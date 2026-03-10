#![allow(non_snake_case)]

#[cfg(test)]
pub mod tests;

mod ntt;
pub use ntt::{evals_to_monomials, monomials_to_evals};
pub(crate) use ntt::{
    evals_to_monomials_2_pass, evals_to_monomials_3_pass, monomials_to_evals_2_pass,
    monomials_to_evals_3_pass,
};

mod hypercube;
pub use hypercube::{hypercube_evals_to_monomials, hypercube_monomials_to_evals};
pub(crate) use hypercube::{
    hypercube_evals_to_monomials_2_pass, hypercube_evals_to_monomials_3_pass,
    hypercube_monomials_to_evals_2_pass, hypercube_monomials_to_evals_3_pass,
};
