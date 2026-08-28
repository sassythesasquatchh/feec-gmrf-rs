//! Gaussian random fields on finite element differential-form spaces.
//!
//! This crate combines FEEC mass, incidence, Hodge, boundary, and
//! reconstruction operators with sparse GMRF priors, observations, constraints,
//! Laplace approximations, physical pushforwards, and uncertainty estimators.
//! The public model types use geometry-independent sparse operators after FEEC
//! assembly.

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
