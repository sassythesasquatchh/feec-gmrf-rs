//! Canonical sparse Hodge--Matérn prior API.
//!
//! The public distinction is where the requested Matérn spectrum lives:
//! on the latent potential or on the synthesized differential form. Gauge
//! selection used by the form-spectrum implementation is intentionally not
//! part of this API's naming.

pub use super::sparse_anchor_hodge::{
    build_hodge_matern_1form_prior, build_hodge_matern_1form_prior_with_coords,
    HodgeMatern1FormBranch, HodgeMatern1FormPrior, HodgeMatern1FormPriorConfig,
    HodgeMaternBranchConfig, HodgeMaternSpectrum, HodgePotentialGauge,
};
