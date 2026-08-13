//! Case-study workflows for the FEEC/GMRF thesis codebase.
//!
//! The publication-supported API is exposed through [`publication`].  Root-level
//! modules are kept for backwards compatibility with exploratory binaries and
//! historical experiments; not every root module is part of the publication
//! reproducibility surface.

pub mod annulus_baselines;
pub mod annulus_h_formulation;
pub mod cube_mass_inverse_variance;
#[cfg(feature = "experimental")]
pub mod cube_voids_coexact_transform;
pub mod cube_zero_form_kernel_validation;
pub mod de_rham;
#[cfg(feature = "experimental")]
pub mod derham_ladder;
#[cfg(feature = "experimental")]
pub mod genus2_1form_hodge_conditioning;
#[cfg(feature = "experimental")]
pub mod genus2_nondecomposed_period_variance;
#[cfg(feature = "experimental")]
pub mod genus2_sparse_anchor_branch_conditioning;
#[cfg(feature = "experimental")]
pub mod genus2_topological_inverse;
pub mod magnetic_physical_calibration;
pub mod magnetic_prior_uq_comparison;
pub mod matern_functional_convergence;
#[cfg(feature = "experimental")]
pub mod matern_functional_convergence_4d;
pub mod matern_scalar;
pub mod matern_scalar_borderline_4d;
pub mod matern_trace_normalization;
#[cfg(feature = "experimental")]
pub mod nc1_matern_convergence;
#[cfg(feature = "experimental")]
pub mod nonlinear_eddy_current;
#[cfg(feature = "experimental")]
pub mod planar_holes_hodge_flow;
#[cfg(feature = "experimental")]
pub mod sparse_anchor_branch_functional_convergence;
pub mod sphere_branch_observable_convergence;
#[cfg(feature = "experimental")]
pub mod sphere_branch_pushforward_convergence;
#[cfg(feature = "experimental")]
pub mod sphere_nc1_matern_spectral_reference;
#[cfg(feature = "experimental")]
pub mod sphere_sparse_anchor_branch_conditioning;
pub mod sphere_sparse_anchor_kernel_validation;
#[cfg(feature = "experimental")]
pub mod square_zero_form_kernel_validation;
pub mod study;
#[cfg(feature = "experimental")]
pub mod team13;
#[cfg(feature = "experimental")]
mod team13_material;
#[cfg(feature = "experimental")]
pub mod team7;
pub mod toroidal_exact_b_sweeps;
#[cfg(feature = "experimental")]
pub mod toroidal_harmonic_b;
pub mod toroidal_inductor;
mod toroidal_material;
pub mod torus;
pub mod visual_output;
#[cfg(feature = "experimental")]
pub mod weighted_hodge_matern_isolation;

pub mod publication {
    //! Publication-supported thesis workflows.
    //!
    //! These re-exports identify the case-study modules that are intended to be
    //! maintained as the reproducible surface for the thesis/report.  Other
    //! root-level modules are experimental or historical unless promoted here.

    pub mod chapter7 {
        //! Active Chapter 7 numerical-validation workflows.

        pub use crate::cube_mass_inverse_variance;
        pub use crate::matern_functional_convergence;
        pub use crate::matern_scalar;
        pub use crate::matern_scalar_borderline_4d;
        pub use crate::matern_trace_normalization;
        pub use crate::sphere_branch_observable_convergence;
        pub use crate::torus;
    }

    pub mod chapter8 {
        //! Active Chapter 8 electromagnetic UQ workflows.

        pub use crate::annulus_baselines;
        pub use crate::annulus_h_formulation;
        pub use crate::magnetic_physical_calibration;
        pub use crate::magnetic_prior_uq_comparison;
        pub use crate::toroidal_inductor;
    }
}

#[cfg(feature = "experimental")]
pub mod experimental {
    //! Historical and exploratory workflows.
    //!
    //! These modules remain available for local experiments, but they are not
    //! part of the publication-supported API and may still contain duplicated
    //! or less-polished orchestration code.

    pub use crate::cube_voids_coexact_transform;
    pub use crate::cube_zero_form_kernel_validation;
    pub use crate::derham_ladder;
    pub use crate::genus2_1form_hodge_conditioning;
    pub use crate::genus2_nondecomposed_period_variance;
    pub use crate::genus2_sparse_anchor_branch_conditioning;
    pub use crate::genus2_topological_inverse;
    pub use crate::matern_functional_convergence_4d;
    pub use crate::nc1_matern_convergence;
    pub use crate::nonlinear_eddy_current;
    pub use crate::planar_holes_hodge_flow;
    pub use crate::sparse_anchor_branch_functional_convergence;
    pub use crate::sphere_branch_pushforward_convergence;
    pub use crate::sphere_nc1_matern_spectral_reference;
    pub use crate::sphere_sparse_anchor_branch_conditioning;
    pub use crate::sphere_sparse_anchor_kernel_validation;
    pub use crate::square_zero_form_kernel_validation;
    pub use crate::team13;
    pub use crate::team7;
    pub use crate::toroidal_harmonic_b;
    pub use crate::weighted_hodge_matern_isolation;
}

#[cfg(all(test, any(feature = "heavy-tests", feature = "experimental")))]
pub(crate) mod test_util {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    static FEEC_CASE_STUDY_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    pub(crate) fn lock_feec_harmonic_tests() -> MutexGuard<'static, ()> {
        match FEEC_CASE_STUDY_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
        {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}
