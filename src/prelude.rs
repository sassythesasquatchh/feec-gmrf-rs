//! Common imports for constructing FEEC--GMRF models.

pub use crate::boundary::{
    EliminatedLinearMap, EliminatedResidualModel, EssentialBoundaryConditions,
    EssentialBoundaryElimination,
};
pub use crate::error::{FeecGmrfError, Result};
pub use crate::infer::Posterior;
pub use crate::model::nonlinear::{
    NonlinearLaplaceModelBuilder, NonlinearPosterior, NonlinearResidualEvaluation,
    NonlinearResidualTerm, ResidualModel,
};
pub use crate::model::{
    DerivedQuantity, GaussianNoise, LinearConstraint, LinearGaussianModelBuilder, LinearObservation,
};
pub use crate::operator::{BoundaryLayout, FormDegree, FormOperators, LinearMap, SparseMat};
pub use crate::physical::{
    calibrate_prior_to_physical_rms, magnetic_field_map, PhysicalMap, PhysicalRmsCalibration,
};
pub use crate::prior::{
    GaussianPrior, MassInversePolicy, MaternAlpha, MaternParameterConvention, MaternParameters,
    MaternPriorBuilder, PriorNormalization,
};
pub use crate::spacetime::{SpacetimePrior, SpacetimePriorBuilder, TimeDiscretization};
