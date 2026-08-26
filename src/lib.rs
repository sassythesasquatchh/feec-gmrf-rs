//! Reusable finite element exterior calculus (FEEC) and Gaussian Markov
//! random field (GMRF) model construction.
//!
//! The root crate provides the downstream API for FEEC assembly, Gaussian model
//! composition, and inference through geometry-independent operators. Thesis
//! applications live in the case-study crates and use the same implementations.

pub mod boundary;
pub mod diagnostics;
pub mod error;
pub mod hodge;
pub mod infer;
pub mod linear_pde;
pub mod model;
pub mod operator;
pub mod physical;
pub mod prelude;
pub mod prior;
pub mod report;
pub mod spacetime;

#[cfg(feature = "spectral")]
pub mod spectral {
    //! Spectral and low-rank Gaussian-process constructions.

    pub use feg_gp::*;
}

pub use error::{FeecGmrfError, Result};
