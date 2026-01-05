use crate::cs::circuit::LookupQueryTableTypeExt;
use crate::definitions::{Degree1Constraint, GKRAddress, Variable};
use crate::gkr_compiler::graph::{
    graph_element_equals_if_eq, CopyNode, GKRGraph, GraphElement, GraphHolder, NodeIndex,
};
use crate::one_row_compiler::LookupInput;
use crate::tables::TableType;

use super::compiled_constraint::GKRCompiledLinearConstraint;
use super::*;

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookupNumerator {
    Identity,
    Positive(GKRAddress),
    Negative(GKRAddress),
    LinearNumeratorFromInitialAccumulationCaches([GKRAddress; 2]),
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookupDenominator {
    CopiedBaseInput(GKRAddress),
    CopiedCopiedBaseInput(GKRAddress),
    UseInput(NoFieldLinearRelation),
    UseInputViaCopy(GKRAddress),
    UseVectorInput(NoFieldVectorLookupRelation),
    CopyMaterializedVectorInput(GKRAddress),
    MaterializeBaseInput(NoFieldLinearRelation),
    MaterializeVectorInput(NoFieldVectorLookupRelation),
    Setup(GKRAddress),
    VectorSetup(Box<[GKRAddress]>),
    Explicit(GKRAddress),
}

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

// This is just a logical holder of what is a rational pair itself. It's not a node, but can add itself or pair of
// selves into the graph
#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupRationalPair {
    pub num: LookupNumerator,
    pub num_node: Option<NodeIndex>,
    pub den: LookupDenominator,
    pub den_node: Option<NodeIndex>,
    pub lookup_type: LookupType,
}

impl LookupRationalPair {
    pub fn add_single_into_graph(this: Self, graph: &mut dyn GraphHolder) -> Self {
        // we consider very limited set of options here
        match (this.num, this.den) {
            (LookupNumerator::Identity, LookupDenominator::MaterializeVectorInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_none());

                let node = MaterializeVectorInputNode(input);
                let node_idx = graph.add_node_dyn(Box::new(node));
                let address = graph.get_address_for_node_index(node_idx);

                Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::Explicit(address),
                    den_node: Some(node_idx),
                    lookup_type: this.lookup_type,
                }
            }
            (LookupNumerator::Identity, LookupDenominator::UseInputViaCopy(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_none());

                match input {
                    GKRAddress::BaseLayerMemory(..) | GKRAddress::BaseLayerWitness(..) | GKRAddress::Setup(..) => {},
                    _ => {
                        unreachable!()
                    }
                }

                let node = CopyNode::FromBase(input);
                let node_idx = graph.add_node_dyn(Box::new(node));
                let address = graph.get_address_for_node_index(node_idx);

                Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedBaseInput(address),
                    den_node: Some(node_idx),
                    lookup_type: this.lookup_type,
                }
            }
            (LookupNumerator::Identity, LookupDenominator::CopiedBaseInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_some());

                let GKRAddress::InnerLayer { .. } = input else {
                    unreachable!()
                };

                let node = CopyNode::FromIntermediate(input);
                let node_idx = graph.add_node_dyn(Box::new(node));
                let address = graph.get_address_for_node_index(node_idx);

                Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedCopiedBaseInput(address),
                    den_node: Some(node_idx),
                    lookup_type: this.lookup_type,
                }
            }
            (LookupNumerator::Identity, LookupDenominator::MaterializeBaseInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_none());

                let node = MaterializeSingleInputNode(input);
                let node_idx = graph.add_node_dyn(Box::new(node));
                let address = graph.get_address_for_node_index(node_idx);

                Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedCopiedBaseInput(address),
                    den_node: Some(node_idx),
                    lookup_type: this.lookup_type,
                }
            }
            (LookupNumerator::Identity, LookupDenominator::CopiedCopiedBaseInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_some());

                let GKRAddress::InnerLayer { .. } = input else {
                    unreachable!()
                };

                let node = CopyNode::FromIntermediate(input);
                let node_idx = graph.add_node_dyn(Box::new(node));
                let address = graph.get_address_for_node_index(node_idx);

                Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedCopiedBaseInput(address),
                    den_node: Some(node_idx),
                    lookup_type: this.lookup_type,
                }
            }
            (num, den) => {
                panic!("{:?}/{:?} is not supported", num, den);
            }
        }
    }

    pub fn accumulate_pair_into_graph(pair: (Self, Self), graph: &mut dyn GraphHolder) -> Self {
        let (a, b) = pair;
        assert_eq!(a.lookup_type, b.lookup_type);
        let lookup_type = a.lookup_type;

        match (a.num, a.den, b.num, b.den) {
            (
                LookupNumerator::Positive(var),
                LookupDenominator::UseVectorInput(input),
                LookupNumerator::Negative(multiplicity),
                LookupDenominator::VectorSetup(setup),
            ) => {
                todo!();
            }
            (
                LookupNumerator::Identity,
                LookupDenominator::UseVectorInput(input),
                LookupNumerator::Negative(multiplicity),
                LookupDenominator::VectorSetup(setup),
            ) => {
                todo!();
            }
            (
                LookupNumerator::Identity,
                LookupDenominator::UseInput(input),
                LookupNumerator::Negative(multiplicity),
                LookupDenominator::Setup(setup),
            ) => {
                let numerator_node = LookupSingleColumnWitnessMinusSetupInputNodeNumerator(
                    LookupSingleColumnWitnessMinusSetupInputNode {
                        input: input.clone(),
                        multiplicity,
                        setup,
                    },
                );
                let denominator_node = LookupSingleColumnWitnessMinusSetupInputNodeDenominator(
                    LookupSingleColumnWitnessMinusSetupInputNode {
                        input,
                        multiplicity,
                        setup,
                    },
                );
                let num_node_idx = graph.add_node_dyn(Box::new(numerator_node));
                let den_node_idx = graph.add_node_dyn(Box::new(denominator_node));

                Self {
                    num: LookupNumerator::Positive(graph.get_address_for_node_index(num_node_idx)),
                    num_node: Some(num_node_idx),
                    den: LookupDenominator::Explicit(
                        graph.get_address_for_node_index(den_node_idx),
                    ),
                    den_node: Some(den_node_idx),
                    lookup_type,
                }
            }
            (
                LookupNumerator::Identity,
                LookupDenominator::UseInput(a),
                LookupNumerator::Identity,
                LookupDenominator::UseInput(b),
            ) => {
                let numerator_node = LookupSingleColumnWitnessPairAggregationNodeNumerator(
                    LookupSingleColumnWitnessPairAggregationNode {
                        lhs: a.clone(),
                        rhs: b.clone(),
                    },
                );
                let denominator_node = LookupSingleColumnWitnessPairAggregationNodeDenominator(
                    LookupSingleColumnWitnessPairAggregationNode {
                        lhs: a.clone(),
                        rhs: b.clone(),
                    },
                );
                let num_node_idx = graph.add_node_dyn(Box::new(numerator_node));
                let den_node_idx = graph.add_node_dyn(Box::new(denominator_node));

                Self {
                    num: LookupNumerator::Positive(graph.get_address_for_node_index(num_node_idx)),
                    num_node: Some(num_node_idx),
                    den: LookupDenominator::Explicit(
                        graph.get_address_for_node_index(den_node_idx),
                    ),
                    den_node: Some(den_node_idx),
                    lookup_type,
                }
            }
            (
                LookupNumerator::Positive(a_num),
                LookupDenominator::Explicit(a_den),
                LookupNumerator::Positive(b_num),
                LookupDenominator::Explicit(b_den),
            ) => {
                let numerator_node =
                    LookupExplicitPairAggregationNodeNumerator(LookupExplicitPairAggregationNode {
                        lhs_num: a_num,
                        lhs_den: a_den,
                        rhs_num: b_num,
                        rhs_den: b_den,
                    });
                let denominator_node = LookupExplicitPairAggregationNodeDenominator(
                    LookupExplicitPairAggregationNode {
                        lhs_num: a_num,
                        lhs_den: a_den,
                        rhs_num: b_num,
                        rhs_den: b_den,
                    },
                );
                let num_node_idx = graph.add_node_dyn(Box::new(numerator_node));
                let den_node_idx = graph.add_node_dyn(Box::new(denominator_node));

                Self {
                    num: LookupNumerator::Positive(graph.get_address_for_node_index(num_node_idx)),
                    num_node: Some(num_node_idx),
                    den: LookupDenominator::Explicit(
                        graph.get_address_for_node_index(den_node_idx),
                    ),
                    den_node: Some(den_node_idx),
                    lookup_type,
                }
            }
            (
                LookupNumerator::Positive(a_num),
                LookupDenominator::Explicit(a_den),
                LookupNumerator::Identity,
                LookupDenominator::CopiedCopiedBaseInput(input),
            ) => {
                assert!(a.num_node.is_some());
                assert!(a.den_node.is_some());
                assert!(b.num_node.is_none());
                assert!(b.den_node.is_some());

                let numerator_node =
                    LookupExplicitPairWithSingleColumnInputAggregationNodeNumerator(
                        LookupExplicitPairWithSingleColumnInputAggregationNode {
                            lhs_num: a_num,
                            lhs_den: a_den,
                            base_input: input,
                        },
                    );
                let denominator_node =
                    LookupExplicitPairWithSingleColumnInputAggregationNodeDenominator(
                        LookupExplicitPairWithSingleColumnInputAggregationNode {
                            lhs_num: a_num,
                            lhs_den: a_den,
                            base_input: input,
                        },
                    );
                let num_node_idx = graph.add_node_dyn(Box::new(numerator_node));
                let den_node_idx = graph.add_node_dyn(Box::new(denominator_node));

                Self {
                    num: LookupNumerator::Positive(graph.get_address_for_node_index(num_node_idx)),
                    num_node: Some(num_node_idx),
                    den: LookupDenominator::Explicit(
                        graph.get_address_for_node_index(den_node_idx),
                    ),
                    den_node: Some(den_node_idx),
                    lookup_type,
                }
            }
            (a_num, a_den, b_num, b_den) => {
                panic!(
                    "{:?}/{:?} + {:?}/{:?} is not supported",
                    a_num, a_den, b_num, b_den
                );
            }
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterializeSingleInputNode(pub(crate) NoFieldLinearRelation);

impl DependentNode for MaterializeSingleInputNode {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        DependentNode::add_dependencies_into(&self.0, graph, dst);
    }
}

impl GraphElement for MaterializeSingleInputNode {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Materialize single lookup input node".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        NoFieldGKRRelation::MaterializedSingleLookupInput(self.0.clone())
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterializeVectorInputNode(pub(crate) NoFieldVectorLookupRelation);

impl DependentNode for MaterializeVectorInputNode {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        for el in self.0 .0.iter() {
            let NoFieldLinearRelation { linear_terms, .. } = el;
            for (_, pos) in linear_terms.iter() {
                let node_idx = graph.get_node_index_for_address(*pos);
                dst.push(node_idx);
            }
        }
    }
}

impl GraphElement for MaterializeVectorInputNode {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Materialize vector lookup input node".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        NoFieldGKRRelation::MaterializedVectorLookupInput(self.0.clone())
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupMaskedWitnessMinusSetupInputNode {
    pub mask: GKRAddress,
    pub input: NoFieldVectorLookupRelation,
    pub multiplicity: GKRAddress,
    pub setup: Box<[GKRAddress]>,
}

impl DependentNode for LookupMaskedWitnessMinusSetupInputNode {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        dst.push(graph.get_node_index_for_address(self.mask));
        for el in self.input.0.iter() {
            let NoFieldLinearRelation { linear_terms, .. } = el;
            for (_, pos) in linear_terms.iter() {
                let node_idx = graph.get_node_index_for_address(*pos);
                dst.push(node_idx);
            }
        }
        dst.push(graph.get_node_index_for_address(self.multiplicity));
        dst.extend(
            self.setup
                .iter()
                .map(|el| graph.get_node_index_for_address(*el)),
        );
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupMaskedWitnessMinusSetupInputNodeNumerator(
    pub LookupMaskedWitnessMinusSetupInputNode,
);

impl GraphElement for LookupMaskedWitnessMinusSetupInputNodeNumerator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Numerator for mask/vector_input - multiplicity/vector_setup".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupMaskedWitnessMinusSetupInputNodeDenominator(
    pub LookupMaskedWitnessMinusSetupInputNode,
);

impl GraphElement for LookupMaskedWitnessMinusSetupInputNodeDenominator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Denominator for mask/vector_input - multiplicity/vector_setup".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessMinusSetupInputNode {
    pub input: NoFieldLinearRelation,
    pub multiplicity: GKRAddress,
    pub setup: GKRAddress,
}

impl DependentNode for LookupSingleColumnWitnessMinusSetupInputNode {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        DependentNode::add_dependencies_into(&self.input, graph, dst);
        dst.push(graph.get_node_index_for_address(self.multiplicity));
        dst.push(graph.get_node_index_for_address(self.setup));
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessMinusSetupInputNodeNumerator(
    pub LookupSingleColumnWitnessMinusSetupInputNode,
);

impl GraphElement for LookupSingleColumnWitnessMinusSetupInputNodeNumerator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Numerator for 1/single_input - multiplicity/single_setup".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessMinusSetupInputNodeDenominator(
    pub LookupSingleColumnWitnessMinusSetupInputNode,
);

impl GraphElement for LookupSingleColumnWitnessMinusSetupInputNodeDenominator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Denominator for 1/single_input - multiplicity/single_setup".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessPairAggregationNode {
    pub lhs: NoFieldLinearRelation,
    pub rhs: NoFieldLinearRelation,
}

impl DependentNode for LookupSingleColumnWitnessPairAggregationNode {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        DependentNode::add_dependencies_into(&self.lhs, graph, dst);
        DependentNode::add_dependencies_into(&self.rhs, graph, dst);
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessPairAggregationNodeNumerator(
    pub LookupSingleColumnWitnessPairAggregationNode,
);

impl GraphElement for LookupSingleColumnWitnessPairAggregationNodeNumerator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Numerator for 1/single_input + 1/single_input".to_string()
    }
    fn evaluation_description(&self, _graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        NoFieldGKRRelation::LookupNumeratorFromBaseInputs(
            [self.0.lhs.clone(), self.0.rhs.clone()]
        )
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessPairAggregationNodeDenominator(
    pub LookupSingleColumnWitnessPairAggregationNode,
);

impl GraphElement for LookupSingleColumnWitnessPairAggregationNodeDenominator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Denominator for 1/single_input + 1/single_input".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        NoFieldGKRRelation::LookupDenominatorFromBaseInputs(
            [self.0.lhs.clone(), self.0.rhs.clone()]
        )
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairAggregationNode {
    pub lhs_num: GKRAddress,
    pub lhs_den: GKRAddress,
    pub rhs_num: GKRAddress,
    pub rhs_den: GKRAddress,
}

impl DependentNode for LookupExplicitPairAggregationNode {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        dst.push(graph.get_node_index_for_address(self.lhs_num));
        dst.push(graph.get_node_index_for_address(self.lhs_den));
        dst.push(graph.get_node_index_for_address(self.rhs_num));
        dst.push(graph.get_node_index_for_address(self.rhs_den));
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairAggregationNodeNumerator(pub LookupExplicitPairAggregationNode);

impl GraphElement for LookupExplicitPairAggregationNodeNumerator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Numerator for a/b + c/d".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairAggregationNodeDenominator(pub LookupExplicitPairAggregationNode);

impl GraphElement for LookupExplicitPairAggregationNodeDenominator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Denominator for a/b + c/d".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairWithSingleColumnInputAggregationNode {
    pub lhs_num: GKRAddress,
    pub lhs_den: GKRAddress,
    pub base_input: GKRAddress,
}

impl DependentNode for LookupExplicitPairWithSingleColumnInputAggregationNode {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        dst.push(graph.get_node_index_for_address(self.lhs_num));
        dst.push(graph.get_node_index_for_address(self.lhs_den));
        dst.push(graph.get_node_index_for_address(self.base_input));
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairWithSingleColumnInputAggregationNodeNumerator(
    pub LookupExplicitPairWithSingleColumnInputAggregationNode,
);

impl GraphElement for LookupExplicitPairWithSingleColumnInputAggregationNodeNumerator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Numerator for a/b + 1/single_input".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairWithSingleColumnInputAggregationNodeDenominator(
    pub LookupExplicitPairWithSingleColumnInputAggregationNode,
);

impl GraphElement for LookupExplicitPairWithSingleColumnInputAggregationNodeDenominator {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(&self.0, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Denominator for a/b + 1/single_input".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!();
    }
}
