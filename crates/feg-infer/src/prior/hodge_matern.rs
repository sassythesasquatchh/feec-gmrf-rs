//! Sparse Hodge--Matérn priors for 1-forms.
//!
//! The public distinction is where the requested Matérn spectrum lives:
//! on the latent potential or on the synthesized differential form. Gauge
//! selection makes the latent potential precision proper but does not define a
//! separate spectrum or Hodge branch.

pub use super::sparse_anchor_hodge::{
    build_hodge_matern_1form_prior, build_hodge_matern_1form_prior_with_coords,
    HodgeMatern1FormBranch, HodgeMatern1FormPrior, HodgeMatern1FormPriorConfig,
    HodgeMaternBranchConfig, HodgeMaternSpectrum, HodgePotentialGauge,
};
