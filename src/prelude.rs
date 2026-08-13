//! Common imports for constructing FEEC--GMRF models.

pub use crate::boundary::{
    EliminatedLinearMap, EliminatedResidualModel, EssentialBoundaryConditions,
    EssentialBoundaryElimination,
};
pub use crate::error::{FeecGmrfError, Result};
pub use crate::infer::{
    MonteCarloVarianceConfig, Posterior, VarianceEstimate, VarianceEstimator, VarianceMethod,
};
pub use crate::linear_pde::{
    DeterministicLinearPdeSolution, LinearPdeModelBuilder, LinearPdeSystem, PdeResidualNoise,
};
pub use crate::model::nonlinear::{
    NonlinearLaplaceModelBuilder, NonlinearPosterior, NonlinearResidualEvaluation,
    NonlinearResidualTerm, ResidualModel,
};
pub use crate::model::{
    DerivedQuantity, GaussianNoise, LinearConstraint, LinearGaussianModelBuilder, LinearObservation,
};
pub use crate::operator::{BoundaryLayout, FormDegree, FormOperators, LinearMap, SparseMat};
pub use crate::physical::{
    calibrate_prior_to_physical_rms, calibrate_prior_to_weighted_physical_rms, magnetic_field_map,
    outward_boundary_flux_map_3d, scalar_field_l2_rms_weights, MagneticFieldMaps3d, PhysicalMap,
    PhysicalRmsCalibration,
};
pub use crate::prior::{
    GaussianPrior, MassInversePolicy, MaternAlpha, MaternParameterConvention, MaternParameters,
    MaternPriorBuilder, PriorMahalanobisCalibration, PriorNormalization,
};
pub use crate::spacetime::{SpacetimePrior, SpacetimePriorBuilder, TimeDiscretization};
