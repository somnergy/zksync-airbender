use super::*;

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldLinearRelation {
    pub linear_terms: Box<[(u32, GKRAddress)]>,
    pub constant: u32,
}

impl NoFieldLinearRelation {
    pub fn from_single_input(input: GKRAddress) -> Self {
        Self {
            linear_terms: vec![(1, input)].into_boxed_slice(),
            constant: 0,
        }
    }
}
