use std::ops::Range;

use crate::gkr_compiler::graph::{GKRGraph, GraphHolder};

use super::*;

pub struct GKRLayout {
    pub layered_relations: Vec<Vec<NoFieldGKRRelation>>,
    pub layered_cached_relations: Vec<Vec<NoFieldGKRCacheRelation>>,
    pub base_layer_vars_participation_in_layers: BTreeMap<GKRAddress, Vec<usize>>, // transitive via cached relations
}

#[serde_with::serde_as]
#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GKRLayerDescription {
    pub layer: usize,
    // our point of convergence is batched sumchecks for lookup and grand product reduction,
    // but as we move to GKR stage where polynomial sizes are not changing, it'll reflect which random evaluation point to use
    // for the sumcheck like f(r) = \sum_x eq(r, x) a(x) b(x)
    pub gates_with_external_connections: Vec<GateArtifacts>,
    #[serde_as(as = "Vec<(_, _)>")]
    pub cached_relations: BTreeMap<GKRAddress, NoFieldGKRCacheRelation>,
    pub gates: Vec<GateArtifacts>,
    pub additional_base_layer_openings: Vec<GKRAddress>,
}

impl GKRGraph {
    fn dump_base_layer_set(&self) -> BTreeSet<GKRAddress> {
        let mut result = BTreeSet::new();
        result.extend(self.base_layer_memory_rev.keys().copied());
        result.extend(self.base_layer_witness_rev.keys().copied());
        result.extend(self.setups.iter().copied());

        result
    }

    pub(crate) fn layout_layers(
        &mut self,
        grand_product_outputs: [(GKRAddress, NoFieldGKRRelation); 2],
        mut lookup_outputs: BTreeMap<LookupType, ([GKRAddress; 2], NoFieldGKRRelation)>,
    ) -> Vec<GKRLayerDescription> {
        assert!(self.enforced_relations.len() > 0);
        assert!(self.enforced_relations.get(&0).is_none());

        let mut result = vec![];

        let total_layers = self.enforced_relations.len();

        // the only difficult topic is if a layer has any connection to the size-reducing part of GKR, otherwise we just take
        // all relations without splitting

        for layer in (1..(1 + total_layers)).rev() {
            if layer != 1 {
                assert!(self.cached_relations.get(&layer).is_none());
            }

            let mut descr = GKRLayerDescription {
                layer,
                gates_with_external_connections: vec![],
                gates: vec![],
                cached_relations: BTreeMap::new(),
                additional_base_layer_openings: vec![],
            };

            let relations = &self.enforced_relations[&layer];
            let cached_relations_at_layer = self
                .cached_relations
                .get(&layer)
                .cloned()
                .unwrap_or_default();
            let mut external_lookup_connections = BTreeSet::new();

            let mut base_layer_polys_to_open_for_caches = if layer == 1 {
                self.dump_base_layer_set()
            } else {
                BTreeSet::new()
            };

            'outer: for rel in relations.iter() {
                if rel == &grand_product_outputs[0].1 || rel == &grand_product_outputs[1].1 {
                    assert!(rel.cached_addresses().is_empty());
                    let artifact = GateArtifacts {
                        output_layer: layer,
                        enforced_relation: rel.clone(),
                    };
                    descr.gates_with_external_connections.push(artifact);
                    continue 'outer;
                }
                for (k, (_, el)) in lookup_outputs.iter() {
                    if el == rel {
                        assert!(rel.cached_addresses().is_empty());
                        let artifact = GateArtifacts {
                            output_layer: layer,
                            enforced_relation: rel.clone(),
                        };
                        descr.gates_with_external_connections.push(artifact);
                        external_lookup_connections.insert(*k);
                        continue 'outer;
                    }
                }

                // no external connection beyond this point
                let referenced_caches = rel.cached_addresses();
                for cached in referenced_caches.into_iter() {
                    let GKRAddress::Cached { layer: l, offset } = cached else {
                        unreachable!();
                    };
                    assert_eq!(l, layer);
                    let relation = cached_relations_at_layer[offset].clone();
                    // if layer == 1 {
                    //     for el in relation.dependencies().into_iter() {
                    //         base_layer_polys_to_open_for_caches.remove(&el);
                    //     }
                    // }
                    descr.cached_relations.insert(cached, relation);
                }
                if layer == 1 {
                    for claim in rel.created_claims().into_iter() {
                        base_layer_polys_to_open_for_caches.remove(&claim);
                    }
                }
                let artifact = GateArtifacts {
                    output_layer: layer,
                    enforced_relation: rel.clone(),
                };
                descr.gates.push(artifact);
            }

            for k in external_lookup_connections.into_iter() {
                lookup_outputs.remove(&k);
            }

            if layer == 1 {
                descr
                    .additional_base_layer_openings
                    .extend(base_layer_polys_to_open_for_caches.into_iter());
            }

            result.push(descr);
        }

        result
    }
}
