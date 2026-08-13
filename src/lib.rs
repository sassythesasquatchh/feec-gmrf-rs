//! Reusable finite element exterior calculus (FEEC) and Gaussian Markov
//! random field (GMRF) model construction.
//!
//! This crate is the supported entry point for downstream applications.  It
//! presents FEEC assembly, Gaussian model composition, and inference through
//! geometry-independent operators. The thesis case studies use the same
//! canonical FEEC, GMRF, and integration implementations but are not part of
//! this downstream stability boundary.

pub mod boundary;
pub mod error;
pub mod infer;
pub mod model;
pub mod operator;
pub mod physical;
pub mod prelude;
pub mod prior;
pub mod spacetime;

#[cfg(feature = "spectral")]
pub mod spectral {
    //! Spectral and low-rank Gaussian-process constructions.

    pub use feg_gp::*;
}

pub use error::{FeecGmrfError, Result};
