use crate::cs::circuit::LookupQueryTableTypeExt;
use crate::definitions::{Degree1Constraint, GKRAddress, Variable};
use crate::gkr_compiler::graph::{graph_element_equals_if_eq, GKRGraph, GraphElement, GraphHolder};
use crate::gkr_compiler::lookup_nodes::LookupRationalPair;
use crate::one_row_compiler::LookupInput;
use crate::tables::TableType;

use super::compiled_constraint::GKRCompiledLinearConstraint;
use super::*;

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupInputRelation<F: PrimeField, const TOTAL_WIDTH: usize> {
    #[serde(bound(
        deserialize = "arrayvec::ArrayVec<Degree1Constraint<F>, TOTAL_WIDTH>: serde::Deserialize<'de>"
    ))]
    #[serde(bound(
        serialize = "arrayvec::ArrayVec<Degree1Constraint<F>, TOTAL_WIDTH>: serde::Serialize"
    ))]
    pub inputs: arrayvec::ArrayVec<Degree1Constraint<F>, TOTAL_WIDTH>,
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupInputRelation<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        for c in self.inputs.iter() {
            for (_, v) in c.linear_terms.iter() {
                let idx = graph.get_node_index_for_variable(*v);
                dst.push(idx);
            }
        }
    }
}

// This node takes two expressions like 1/(witness + gamma) and will produce 1 new GKR virtual poly
// that is product of denominators. It is real node, but for the rest we will represent numerator/denominator as a tuple
#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TrivialLookupInputDenominatorNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    #[serde(bound(deserialize = "LookupInputRelation<F, TOTAL_WIDTH>: serde::Deserialize<'de>"))]
    #[serde(bound(serialize = "LookupInputRelation<F, TOTAL_WIDTH>: serde::Serialize"))]
    pub inputs: [LookupInputRelation<F, TOTAL_WIDTH>; 2],
    pub lookup_type: LookupType,
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        for input in self.inputs.iter() {
            input.add_dependencies_into(graph, dst);
        }
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement
    for TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>
{
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        match self.lookup_type {
            LookupType::RangeCheck16 => {
                "Trivial lookup input denominator node in range-check 16".to_string()
            }
            LookupType::TimestampRangeCheck => {
                "Trivial lookup input denominator node in timestamp range-check".to_string()
            }
            LookupType::Generic => {
                "Trivial lookup input denominator node in generic lookup".to_string()
            }
        }
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        // add cached relations
        let parts = self.inputs.each_ref().map(|el| {
            let cached = lookup_input_into_cached_expr(el, &*graph);
            graph.add_cached_relation(cached)
        });

        NoFieldGKRRelation::LookupDenominatorFromCaches(parts)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterializedVectorLookupInputNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    #[serde(bound(deserialize = "LookupInputRelation<F, TOTAL_WIDTH>: serde::Deserialize<'de>"))]
    #[serde(bound(serialize = "LookupInputRelation<F, TOTAL_WIDTH>: serde::Serialize"))]
    pub input: LookupInputRelation<F, TOTAL_WIDTH>,
    pub lookup_type: LookupType,
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for MaterializedVectorLookupInputNode<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        self.input.add_dependencies_into(graph, dst);
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement
    for MaterializedVectorLookupInputNode<F, TOTAL_WIDTH>
{
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        match self.lookup_type {
            LookupType::RangeCheck16 => {
                "Materialize vector lookup input node in range-check 16".to_string()
            }
            LookupType::TimestampRangeCheck => {
                "Materialize vector lookup input node in timestamp range-check".to_string()
            }
            LookupType::Generic => {
                "Materialize vector lookup input node in generic lookup".to_string()
            }
        }
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        let rel = lookup_input_into_cached_expr(&self.input, &*graph);
        let NoFieldGKRCacheRelation::VectorizedLookup(inner) = rel else {
            unreachable!()
        };

        NoFieldGKRRelation::MaterializedVectorLookupInput(inner)
    }
}

// this is just a way of bookkeeping. It is not a real node
#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookupAccumulationInputTuple<F: PrimeField, const TOTAL_WIDTH: usize> {
    #[serde(bound(deserialize = "LookupInputRelation<F, TOTAL_WIDTH>: serde::Deserialize<'de>"))]
    #[serde(bound(serialize = "LookupInputRelation<F, TOTAL_WIDTH>: serde::Serialize"))]
    NaiveInput {
        den: LookupInputRelation<F, TOTAL_WIDTH>,
    },
    CopySingleInput {
        input: GKRAddress,
    },
    MaterializedVectorInput {
        input: MaterializedVectorLookupInputNode<F, TOTAL_WIDTH>,
    },
    TriviallyAccumulated {
        num: [LookupInputRelation<F, TOTAL_WIDTH>; 2],
        den: TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>,
    },
    #[serde(bound(
        deserialize = "arrayvec::ArrayVec<GKRAddress, TOTAL_WIDTH>: serde::Deserialize<'de>"
    ))]
    #[serde(bound(serialize = "arrayvec::ArrayVec<GKRAddress, TOTAL_WIDTH>: serde::Serialize"))]
    MultiplicityInputFromSetup {
        num: Variable,
        den: arrayvec::ArrayVec<GKRAddress, TOTAL_WIDTH>, // setup
    },
    MultiplicityInputFromWitness {
        num: Variable,
        den: LookupInputRelation<F, TOTAL_WIDTH>,
    },
    // CopyInput(CopyNode),
    // CopyVectorInput(CopyNode),
    // CopyRationalPair {
    //     num: CopyNode,
    //     den: CopyNode,
    // }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupAccumulationInputTuple<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        match self {
            Self::NaiveInput { den } => {
                DependentNode::add_dependencies_into(den, graph, dst);
            }
            Self::CopySingleInput { input } => {
                let node_idx = graph.get_node_index_for_address(*input);
                dst.push(node_idx);
            }
            Self::MaterializedVectorInput { input } => {
                let node_idx = graph.get_node_index(input).expect("already placed");
                dst.push(node_idx);
            }
            Self::TriviallyAccumulated { num, den } => {
                assert_eq!(num, &den.inputs);
                for input in num.iter() {
                    DependentNode::add_dependencies_into(input, graph, dst);
                }
                let node_idx = graph.get_node_index(den).expect("already placed");
                dst.push(node_idx);
            }
            Self::MultiplicityInputFromSetup { num, den: _, .. } => {
                let idx = graph.get_node_index_for_variable(*num);
                dst.push(idx);
            }
            Self::MultiplicityInputFromWitness { num, den, .. } => {
                let idx = graph.get_node_index_for_variable(*num);
                dst.push(idx);
                DependentNode::add_dependencies_into(den, graph, dst);
            } // Self::CopyInput(node) | Self::CopyVectorInput(node) => {
              //     let node_idx = graph.get_node_index(node).expect("already placed");
              //     dst.push(node_idx);
              // }
              // Self::CopyRationalPair { num, den } => {
              //     dst.push(graph.get_node_index(num).expect("already placed"));
              //     dst.push(graph.get_node_index(den).expect("already placed"));
              // }
        }
    }
}

// formal accumulation nodes, that take two rational
#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookupAccumulationNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    #[serde(bound(
        deserialize = "LookupAccumulationInputTuple<F, TOTAL_WIDTH>: serde::Deserialize<'de>"
    ))]
    #[serde(bound(serialize = "LookupAccumulationInputTuple<F, TOTAL_WIDTH>: serde::Serialize"))]
    InitialAccumulation {
        lhs: LookupAccumulationInputTuple<F, TOTAL_WIDTH>,
        rhs: LookupAccumulationInputTuple<F, TOTAL_WIDTH>,
    },
    #[serde(bound(
        deserialize = "NumeratorNode<F, TOTAL_WIDTH>: serde::Deserialize<'de>, DenominatorNode<F, TOTAL_WIDTH>: serde::Deserialize<'de>"
    ))]
    #[serde(bound(
        serialize = "NumeratorNode<F, TOTAL_WIDTH>: serde::Serialize, DenominatorNode<F, TOTAL_WIDTH>: serde::Serialize"
    ))]
    Recursive {
        lhs: (
            Box<NumeratorNode<F, TOTAL_WIDTH>>,
            Box<DenominatorNode<F, TOTAL_WIDTH>>,
        ),
        rhs: (
            Box<NumeratorNode<F, TOTAL_WIDTH>>,
            Box<DenominatorNode<F, TOTAL_WIDTH>>,
        ),
    },
    Unbalanced {
        lhs: (
            Box<NumeratorNode<F, TOTAL_WIDTH>>,
            Box<DenominatorNode<F, TOTAL_WIDTH>>,
        ),
        rhs: LookupAccumulationInputTuple<F, TOTAL_WIDTH>,
    },
    CopyBaseInput {
        source: GKRAddress,
    },
    CopyVectorInput {
        source: GKRAddress,
    },
    CopyRationalPair {
        num: GKRAddress,
        den: GKRAddress,
    },
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupAccumulationNode<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        match self {
            Self::InitialAccumulation { lhs, rhs } => {
                DependentNode::add_dependencies_into(lhs, graph, dst);
                DependentNode::add_dependencies_into(rhs, graph, dst);
            }
            Self::Recursive { lhs, rhs } => {
                for node in [
                    lhs.0.as_ref().as_dyn(),
                    lhs.1.as_ref().as_dyn(),
                    rhs.0.as_ref().as_dyn(),
                    rhs.1.as_ref().as_dyn(),
                ] {
                    let node_idx = graph.get_node_index(node).expect("already placed");
                    dst.push(node_idx);
                }
            }
            Self::Unbalanced { lhs, rhs } => {
                for node in [lhs.0.as_ref().as_dyn(), lhs.1.as_ref().as_dyn()] {
                    let node_idx = graph.get_node_index(node).expect("already placed");
                    dst.push(node_idx);
                }
                DependentNode::add_dependencies_into(rhs, graph, dst);
            }
            Self::CopyBaseInput { source } => {
                dst.push(graph.get_node_index_for_address(*source));
            }
            Self::CopyVectorInput { source } => {
                dst.push(graph.get_node_index_for_address(*source));
            }
            Self::CopyRationalPair { num, den } => {
                dst.push(graph.get_node_index_for_address(*num));
                dst.push(graph.get_node_index_for_address(*den));
            }
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct NumeratorNode<F: PrimeField, const TOTAL_WIDTH: usize>(
    pub LookupAccumulationNode<F, TOTAL_WIDTH>,
    String,
);

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct DenominatorNode<F: PrimeField, const TOTAL_WIDTH: usize>(
    pub LookupAccumulationNode<F, TOTAL_WIDTH>,
    String,
);

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode for NumeratorNode<F, TOTAL_WIDTH> {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        DependentNode::add_dependencies_into(&self.0, graph, dst);
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode for DenominatorNode<F, TOTAL_WIDTH> {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        DependentNode::add_dependencies_into(&self.0, graph, dst);
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement for NumeratorNode<F, TOTAL_WIDTH> {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        format!("Lookup aggregation numerator node for {}", &self.1)
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        match &self.0 {
            LookupAccumulationNode::Recursive {
                lhs: (lhs_num, lhs_den),
                rhs: (rhs_num, rhs_den),
            } => match (lhs_num, lhs_den, rhs_num, rhs_den) {
                (lhs_num, lhs_den, rhs_num, rhs_den) => {
                    panic!(
                        "Combination {:?}/{:?} + {:?}/{:?} is not yet supported",
                        lhs_num, lhs_den, rhs_num, rhs_den
                    );
                }
            },
            LookupAccumulationNode::InitialAccumulation { lhs, rhs } => match (lhs, rhs) {
                (lhs, rhs) => {
                    panic!("Combination {:?} + {:?} is not yet supported", lhs, rhs,);
                }
            },
            LookupAccumulationNode::Unbalanced {
                lhs: (lhs_num, lhs_den),
                rhs,
            } => match (lhs_num, lhs_den, rhs) {
                (lhs_num, lhs_den, rhs) => {
                    panic!(
                        "Combination {:?}/{:?} + {:?} is not yet supported",
                        lhs_num, lhs_den, rhs,
                    );
                }
            },
            a @ _ => {
                panic!("Combination {:?} is not yet supported", a,);
            }
        }
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement for DenominatorNode<F, TOTAL_WIDTH> {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        format!("Lookup aggregation denominator node for {}", &self.1)
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!()
    }
}

pub(crate) fn layout_width_1_lookup_expressions<F: PrimeField>(
    graph: &mut GKRGraph,
    expressions: Vec<LookupInput<F>>,
    num_variables: &mut u64,
    all_variables_to_place: &mut BTreeSet<Variable>,
    variable_names: &mut HashMap<Variable, String>,
    lookup_type: &str,
    lookup: LookupType,
) -> (Variable, LookupRationalPair) {
    layout_lookup_expressions::<F, 1>(
        graph,
        expressions
            .into_iter()
            .map(|el| {
                (
                    vec![el],
                    LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder),
                )
            })
            .collect(),
        num_variables,
        all_variables_to_place,
        variable_names,
        lookup_type,
        None,
        lookup,
    )
}

fn lookup_input_node_from_expr<F: PrimeField, const TOTAL_WIDTH: usize>(
    expr: &(Vec<LookupInput<F>>, LookupQueryTableTypeExt<F>),
) -> LookupInputRelation<F, TOTAL_WIDTH> {
    let (expr, table_type) = expr;
    if TOTAL_WIDTH == 1 {
        assert_eq!(expr.len(), 1);
        assert_eq!(
            *table_type,
            LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
        );
    } else {
        assert!(expr.len() + 1 <= TOTAL_WIDTH)
    }

    let mut inputs = arrayvec::ArrayVec::new();

    for el in expr.iter() {
        match el {
            LookupInput::Variable(var) => {
                inputs.push(Degree1Constraint {
                    linear_terms: vec![(F::ONE, *var)].into_boxed_slice(),
                    constant_term: F::ZERO,
                });
            }
            LookupInput::Expression {
                linear_terms,
                constant_coeff,
            } => {
                inputs.push(Degree1Constraint {
                    linear_terms: linear_terms.clone().into_boxed_slice(),
                    constant_term: *constant_coeff,
                });
            }
        }
    }
    if TOTAL_WIDTH > 1 {
        assert_ne!(
            *table_type,
            LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
        );
        match table_type {
            LookupQueryTableTypeExt::Constant(constant) => {
                inputs.push(Degree1Constraint {
                    linear_terms: vec![].into_boxed_slice(),
                    constant_term: F::from_u64_unchecked(constant.to_table_id() as u64),
                });
            }
            LookupQueryTableTypeExt::Variable(var) => {
                inputs.push(Degree1Constraint {
                    linear_terms: vec![(F::ONE, *var)].into_boxed_slice(),
                    constant_term: F::ZERO,
                });
            }
            LookupQueryTableTypeExt::Expression(expr) => match expr {
                LookupInput::Variable(var) => {
                    inputs.push(Degree1Constraint {
                        linear_terms: vec![(F::ONE, *var)].into_boxed_slice(),
                        constant_term: F::ZERO,
                    });
                }
                LookupInput::Expression {
                    linear_terms,
                    constant_coeff,
                } => {
                    inputs.push(Degree1Constraint {
                        linear_terms: linear_terms.clone().into_boxed_slice(),
                        constant_term: *constant_coeff,
                    });
                }
            },
        }
    }

    LookupInputRelation { inputs }
}

pub(crate) fn layout_lookup_expressions<F: PrimeField, const TOTAL_WIDTH: usize>(
    graph: &mut GKRGraph,
    expressions: Vec<(Vec<LookupInput<F>>, LookupQueryTableTypeExt<F>)>,
    num_variables: &mut u64,
    all_variables_to_place: &mut BTreeSet<Variable>,
    variable_names: &mut HashMap<Variable, String>,
    lookup_type: &str,
    decoder_lookup: Option<(Variable, Vec<LookupInput<F>>)>,
    lookup: LookupType,
) -> (Variable, LookupRationalPair) {
    println!(
        "In total of {} lookups of type {}",
        expressions.len(),
        lookup_type
    );

    // create multiplicity
    let multiplicity_var = Variable(*num_variables);
    variable_names.insert(
        multiplicity_var,
        format!("Multiplicity for {}", lookup_type),
    );
    *num_variables += 1;
    all_variables_to_place.insert(multiplicity_var);
    let [multiplicity_pos] =
        graph.layout_witness_subtree_multiple_variables([multiplicity_var], all_variables_to_place);

    for (expr, table_type) in expressions.iter() {
        if TOTAL_WIDTH == 1 {
            assert_eq!(expr.len(), 1);
            assert_eq!(
                *table_type,
                LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
            );
        } else {
            assert!(expr.len() + 1 <= TOTAL_WIDTH)
        }
    }

    let total_rational_terms = expressions.len() + 1 + (decoder_lookup.is_some() as usize);

    let mut initial_reduction_layer_nodes = vec![];
    assert!(total_rational_terms > 0);
    let mut expected_placement_layer = 1;

    if total_rational_terms % 2 == 0 {
        if let Some(decoder_lookup) = decoder_lookup {
            {
                let (decoder_predicate_var, decoder_lookup) = decoder_lookup;
                let decoder_predicate = graph.get_address_for_variable(decoder_predicate_var);
                let input = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&(
                    decoder_lookup,
                    LookupQueryTableTypeExt::Constant(TableType::Decoder),
                ));
                let input = lookup_input_into_relation(&input, &*graph);
                assert_eq!(input.0.len(), graph.setup_addresses(lookup).len());

                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Positive(decoder_predicate),
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<TOTAL_WIDTH>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                if expressions.is_empty() {
                    return (multiplicity_var, next_pair);
                }

                initial_reduction_layer_nodes.push(next_pair);
            }

            // and continue over all other pairs
            assert_eq!(expressions.len() % 2, 0);
            for [a, b] in expressions.as_chunks::<2>().0 {
                // We will take 2 inputs with "1" in the numerator, and some witness in denominator
                let a = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&a);
                let b = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&b);
                let a = lookup_input_into_relation(&a, &*graph);
                let b = lookup_input_into_relation(&b, &*graph);
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                initial_reduction_layer_nodes.push(next_pair);
            }
        } else {
            // we will make a mixed node with one of the witnesses to avoid copying multiplicity
            assert_eq!(expressions.len() % 2, 1);
            let (first, expressions) = expressions.split_at(1);
            let first = &first[0];
            {
                let input = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(first);
                let input = lookup_input_into_relation(&input, &*graph);
                assert_eq!(input.0.len(), graph.setup_addresses(lookup).len());

                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<TOTAL_WIDTH>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                if expressions.is_empty() {
                    return (multiplicity_var, next_pair);
                }

                initial_reduction_layer_nodes.push(next_pair);
            }

            assert_eq!(expressions.len() % 2, 0);
            for [a, b] in expressions.as_chunks::<2>().0 {
                let a = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&a);
                let b = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&b);
                let a = lookup_input_into_relation(&a, &*graph);
                let b = lookup_input_into_relation(&b, &*graph);
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                initial_reduction_layer_nodes.push(next_pair);
            }
        }
    } else {
        // inevitably we will need to copy something, so we will try to copy the simplest case
        if let Some(decoder_lookup) = decoder_lookup {
            assert!(expressions.is_empty() == false);
            {
                let (decoder_predicate_var, decoder_lookup) = decoder_lookup;
                let decoder_predicate = graph.get_address_for_variable(decoder_predicate_var);
                let input = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&(
                    decoder_lookup,
                    LookupQueryTableTypeExt::Constant(TableType::Decoder),
                ));
                let input = lookup_input_into_relation(&input, &*graph);
                assert_eq!(input.0.len(), graph.setup_addresses(lookup).len());

                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Positive(decoder_predicate),
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<TOTAL_WIDTH>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                if expressions.is_empty() {
                    return (multiplicity_var, next_pair);
                }

                initial_reduction_layer_nodes.push(next_pair);
            }

            // and continue over all other pairs
            assert_eq!(expressions.len() % 2, 1);
            for [a, b] in expressions.as_chunks::<2>().0 {
                let a = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&a);
                let b = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&b);
                let a = lookup_input_into_relation(&a, &*graph);
                let b = lookup_input_into_relation(&b, &*graph);
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                initial_reduction_layer_nodes.push(next_pair);
            }
            {
                let last_input = expressions.as_chunks::<2>().1[0].clone();
                let last_input = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&last_input);
                let last_input = lookup_input_into_relation(&last_input, &*graph);
                let last_input = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: copy_single_base_input_or_materialize_vector::<TOTAL_WIDTH>(last_input),
                    den_node: None,
                    lookup_type: lookup,
                };

                let next_pair = LookupRationalPair::add_single_into_graph(last_input, graph);

                initial_reduction_layer_nodes.push(next_pair);
            }
        } else {
            // we will make a mixed node with one of the witnesses to avoid copying multiplicity and setup
            assert_eq!(expressions.len() % 2, 0);
            let (first, expressions) = expressions.split_at(1);
            let first = &first[0];
            {
                let input = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(first);
                let input = lookup_input_into_relation(&input, &*graph);

                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<TOTAL_WIDTH>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                initial_reduction_layer_nodes.push(next_pair);
            }

            assert_eq!(expressions.len() % 2, 1);
            for [a, b] in expressions.as_chunks::<2>().0 {
                let a = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&a);
                let b = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&b);
                let a = lookup_input_into_relation(&a, &*graph);
                let b = lookup_input_into_relation(&b, &*graph);
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<TOTAL_WIDTH>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let next_pair = LookupRationalPair::accumulate_pair_into_graph((a, b), graph);

                initial_reduction_layer_nodes.push(next_pair);
            }
            {
                let last_input = expressions.as_chunks::<2>().1[0].clone();
                let last_input = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&last_input);
                let last_input = lookup_input_into_relation(&last_input, &*graph);
                let last_input = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: copy_single_base_input_or_materialize_vector::<TOTAL_WIDTH>(last_input),
                    den_node: None,
                    lookup_type: lookup,
                };

                let next_pair = LookupRationalPair::add_single_into_graph(last_input, graph);

                initial_reduction_layer_nodes.push(next_pair);
            }
        }
    }

    // now we resolved a problem of copying from base layer, but we still want to have all the relations to be between two
    // nearby layers only

    expected_placement_layer += 1;
    println!(
        "Will continue placement of {} lookup rationals into layer {}",
        initial_reduction_layer_nodes.len(),
        expected_placement_layer
    );

    let mut current_layer = initial_reduction_layer_nodes;

    loop {
        if current_layer.len() == 1 {
            return (multiplicity_var, current_layer.pop().unwrap());
        }

        let mut next_layer = vec![];
        for [a, b] in current_layer.as_chunks::<2>().0.iter() {
            let next_pair =
                LookupRationalPair::accumulate_pair_into_graph((a.clone(), b.clone()), graph);

            next_layer.push(next_pair);
        }
        match current_layer.as_chunks::<2>().1 {
            [] => {}
            [last] => {
                let next_pair = LookupRationalPair::add_single_into_graph(last.clone(), graph);

                next_layer.push(next_pair);
            }
            _ => {
                unreachable!()
            }
        }

        current_layer = next_layer;
    }
}

pub struct GKRLookupInput<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub multiplicity: Option<Variable>,
    pub inputs: arrayvec::ArrayVec<Degree1Constraint<F>, TOTAL_WIDTH>,
}

pub struct CompiledGKRLookupInput<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub multiplicity: Option<GKRAddress>,
    pub inputs: arrayvec::ArrayVec<GKRCompiledLinearConstraint<F>, TOTAL_WIDTH>,
}
