use super::*;

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldVectorLookupRelation(pub Box<[NoFieldLinearRelation]>);
