//! Geometry-independent Gaussian prior construction.

pub mod supported;

use crate::boundary::{
    EliminatedLinearMap, EssentialBoundaryConditions, EssentialBoundaryElimination,
};
use crate::operator::{
    add_sparse, multiply_sparse, BoundaryLayout, ExactBoundaryDomain, ExactFormOperators,
    FormDegree, FormOperatorProvenance, FormOperators, LinearMap, SparseMat,
};
use crate::{FeecGmrfError, Result};
pub use feg_core::MaternAlpha;
use feg_core::SparseTriplet;
use gmrf_core::{Gmrf, Vector};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

/// Convention used to obtain `alpha`, `kappa`, and `tau`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaternParameterConvention {
    /// Parameters were supplied directly as `alpha`, `kappa`, and `tau`.
    DirectAlphaKappaTau,
    /// `kappa` was computed using `range = sqrt(8 nu) / kappa`.
    PracticalRange {
        practical_range: f64,
        intrinsic_dimension: usize,
    },
}

/// Matérn parameters used by the discrete precision recurrence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaternParameters {
    /// Lindgren/Whittle recurrence exponent.
    pub alpha: MaternAlpha,
    /// Inverse correlation-length parameter.
    pub kappa: f64,
    /// Precision amplitude; the final precision is scaled by `tau^2`.
    pub tau: f64,
    /// Convention used to obtain these parameters.
    pub convention: MaternParameterConvention,
}

impl MaternParameters {
    /// Construct validated `alpha`, `kappa`, and `tau` parameters.
    pub fn new(alpha: MaternAlpha, kappa: f64, tau: f64) -> Result<Self> {
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Matérn kappa must be finite and positive".to_string(),
            ));
        }
        if kappa == 0.0 {
            return Err(FeecGmrfError::IntrinsicModelUnsupported(
                "kappa = 0 requires explicit nullspace, harmonic, and quotient semantics"
                    .to_string(),
            ));
        }
        if !tau.is_finite() || tau <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Matérn tau must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            kappa,
            tau,
            convention: MaternParameterConvention::DirectAlphaKappaTau,
        })
    }

    /// Construct parameters from the practical-range convention
    /// `range = sqrt(8 nu) / kappa`, with `nu = alpha - dimension / 2`.
    ///
    /// The `tau` amplitude remains explicit because converting marginal
    /// variance to `tau` depends on the discretization and normalization.
    pub fn from_practical_range(
        alpha: MaternAlpha,
        practical_range: f64,
        intrinsic_dimension: usize,
        tau: f64,
    ) -> Result<Self> {
        if !practical_range.is_finite() || practical_range <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Matérn practical range must be finite and positive".to_string(),
            ));
        }
        let nu = alpha.as_u32() as f64 - intrinsic_dimension as f64 / 2.0;
        if nu <= 0.0 {
            return Err(FeecGmrfError::Unsupported(format!(
                "the practical-range convention requires nu > 0, got {nu}"
            )));
        }
        let mut parameters = Self::new(alpha, (8.0 * nu).sqrt() / practical_range, tau)?;
        parameters.convention = MaternParameterConvention::PracticalRange {
            practical_range,
            intrinsic_dimension,
        };
        Ok(parameters)
    }
}

impl Default for MaternParameters {
    fn default() -> Self {
        Self {
            alpha: MaternAlpha::Two,
            kappa: 1.0,
            tau: 1.0,
            convention: MaternParameterConvention::DirectAlphaKappaTau,
        }
    }
}

/// Policy for the degree-`k` mass inverse between repeated Matérn factors.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RecurrenceMassInversePolicy {
    /// Use the Whitney projected/top-degree inverse supplied by FEEC.
    #[default]
    WhitneyProjected,
    /// Invert the row-sum lumped mass diagonal.
    RowSumLumped,
    /// Invert the assembled diagonal entries. Appropriate for diagonal top-degree masses.
    Diagonal,
    /// Use an explicitly supplied inverse.
    Provided(SparseMat),
}

/// Backwards-compatible policy for the recurrence/load mass inverse.
///
/// This policy never selects the lower-adjoint inverse inside the Hodge
/// Laplacian. New code should use [`RecurrenceMassInversePolicy`] explicitly.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MassInversePolicy {
    RowSumLumped,
    Diagonal,
    #[default]
    Projected,
    Provided(SparseMat),
}

impl From<MassInversePolicy> for RecurrenceMassInversePolicy {
    fn from(policy: MassInversePolicy) -> Self {
        match policy {
            MassInversePolicy::RowSumLumped => Self::RowSumLumped,
            MassInversePolicy::Diagonal => Self::Diagonal,
            MassInversePolicy::Projected => Self::WhitneyProjected,
            MassInversePolicy::Provided(inverse) => Self::Provided(inverse),
        }
    }
}

/// Policy for the lower-degree mass inverse inside the weak Hodge adjoint.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LowerAdjointMassInversePolicy {
    /// Use the Whitney projected sparse inverse supplied by FEEC.
    #[default]
    WhitneyProjected,
    /// Invert the row-sum lumped lower mass diagonal.
    RowSumLumped,
    /// Invert the assembled lower mass diagonal.
    Diagonal,
    /// Use an explicitly supplied inverse on the lower auxiliary space.
    Provided(SparseMat),
}

impl LowerAdjointMassInversePolicy {
    fn kind(&self) -> MassInverseKind {
        match self {
            Self::WhitneyProjected => MassInverseKind::WhitneyProjected,
            Self::RowSumLumped => MassInverseKind::RowSumLumped,
            Self::Diagonal => MassInverseKind::Diagonal,
            Self::Provided(_) => MassInverseKind::Provided,
        }
    }
}

/// Stable, matrix-free description of a resolved inverse construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassInverseKind {
    ExactMaterialized,
    /// The exact metric relation is retained as an auxiliary support constraint.
    ConstrainedSupport,
    WhitneyProjected,
    RowSumLumped,
    Diagonal,
    Provided,
    EmbeddedInCollapsedHodge,
}

impl RecurrenceMassInversePolicy {
    fn kind(&self, degree: FormDegree, complex_dimension: usize) -> MassInverseKind {
        match self {
            Self::WhitneyProjected if degree.get() == complex_dimension => {
                MassInverseKind::Diagonal
            }
            Self::WhitneyProjected => MassInverseKind::WhitneyProjected,
            Self::RowSumLumped => MassInverseKind::RowSumLumped,
            Self::Diagonal => MassInverseKind::Diagonal,
            Self::Provided(_) => MassInverseKind::Provided,
        }
    }
}

/// Whether an inverse role is present and used in a particular Matérn law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseUse {
    NotApplicable,
    NotUsed,
    Applied(MassInverseKind),
}

/// Finite-dimensional representation constructed by a Matérn builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaternRepresentation {
    /// Exact full-metric eliminated law, materialized for reference-sized systems.
    ExactEliminatedReference,
    /// Exact law represented by a sparse energy on a constrained support.
    ExactSupported,
    /// Eliminated adjacent-space law with one exact-materialized inverse and
    /// one caller-supplied inverse. The supplied role is conservatively
    /// recorded as an approximation; equality to the consistent metric is not
    /// certified by this package.
    MixedAdjacentEliminatedApproximation,
    /// Eliminated adjacent-space law whose applicable inverse roles are all
    /// caller-supplied. The package validates their algebraic contracts but
    /// does not certify them as consistent-metric inverses.
    ProvidedAdjacentEliminatedApproximation,
    /// Existing compact Whitney law with at least one approximate metric role.
    WhitneyApproximateEliminated,
    /// A collapsed Hodge operator supplied by the caller.
    ProvidedCollapsedEliminated,
}

/// Realization of one inverse metric in an eliminated adjacent-space law.
///
/// `ExactMaterialized` is intended only for reference-sized systems and is
/// subject to the builder's explicit materialization limit. `Provided` keeps
/// the mathematical site explicit while allowing a caller to supply, for
/// example, a projected sparse inverse assembled by FEEC.
#[derive(Debug, Clone, PartialEq)]
pub enum EliminatedMassInversePolicy {
    ExactMaterialized,
    Provided(SparseMat),
}

impl EliminatedMassInversePolicy {
    fn kind(&self) -> MassInverseKind {
        match self {
            Self::ExactMaterialized => MassInverseKind::ExactMaterialized,
            Self::Provided(_) => MassInverseKind::Provided,
        }
    }
}

/// Origin of the operators used to construct a Matérn prior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaternOperatorSource {
    FeecWhitney,
    UserSuppliedAdjacent { label: String },
    UserSuppliedCollapsed,
}

/// Boundary layouts for the lower, state, and upper spaces of one FEEC law.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjacentBoundaryLayouts {
    lower: Option<crate::operator::BoundaryLayout>,
    state: crate::operator::BoundaryLayout,
    upper: Option<crate::operator::BoundaryLayout>,
}

impl AdjacentBoundaryLayouts {
    fn from_exact(operators: &ExactFormOperators) -> Self {
        Self {
            lower: operators
                .incoming()
                .map(|branch| branch.lower().boundary_layout().clone()),
            state: operators.state().boundary_layout().clone(),
            upper: operators
                .outgoing()
                .map(|branch| branch.upper().boundary_layout().clone()),
        }
    }

    pub fn lower(&self) -> Option<&crate::operator::BoundaryLayout> {
        self.lower.as_ref()
    }

    pub fn state(&self) -> &crate::operator::BoundaryLayout {
        &self.state
    }

    pub fn upper(&self) -> Option<&crate::operator::BoundaryLayout> {
        self.upper.as_ref()
    }
}

/// Boundary operation applied during or after Matérn construction.
#[derive(Debug, Clone, PartialEq)]
pub enum MaternBoundarySemantics {
    /// No essential coefficients were removed from the supplied discrete space.
    UnrestrictedDiscreteComplex,
    /// Adjacent spaces were assembled together on the homogeneous
    /// full-relative FEEC subcomplex before either inverse was formed.
    AssembledHomogeneousFullRelative { layouts: AdjacentBoundaryLayouts },
    /// An already collapsed operator was principally reduced before recurrence.
    PrincipalRestrictedCollapsedOperator {
        state_layout: crate::operator::BoundaryLayout,
    },
    /// A completed finite Gaussian was conditioned on coefficient values.
    ConditionedCompletedDiscreteLaw {
        state_layout: crate::operator::BoundaryLayout,
    },
    /// A downstream caller supplied a compatible adjacent boundary subcomplex.
    UserSuppliedCompatibleSubcomplex {
        name: String,
        layouts: AdjacentBoundaryLayouts,
    },
    /// The caller supplied an already reduced opaque collapsed operator bundle.
    UserSuppliedReducedCollapsedOperator {
        state_layout: crate::operator::BoundaryLayout,
    },
}

/// Auditable description of the finite Matérn law returned by the builder.
#[derive(Debug, Clone, PartialEq)]
pub struct MaternPriorMetadata {
    representation: MaternRepresentation,
    operator_source: MaternOperatorSource,
    boundary: MaternBoundarySemantics,
    form_degree: FormDegree,
    complex_dimension: usize,
    adjacent_dimensions: [Option<usize>; 3],
    parameters: MaternParameters,
    lower_adjoint_inverse: InverseUse,
    recurrence_inverse: InverseUse,
    normalization: PriorNormalization,
    post_construction_precision_scale: f64,
}

impl MaternPriorMetadata {
    pub fn representation(&self) -> MaternRepresentation {
        self.representation
    }

    pub fn operator_source(&self) -> &MaternOperatorSource {
        &self.operator_source
    }

    pub fn boundary(&self) -> &MaternBoundarySemantics {
        &self.boundary
    }

    pub fn form_degree(&self) -> FormDegree {
        self.form_degree
    }

    pub fn complex_dimension(&self) -> usize {
        self.complex_dimension
    }

    /// `[lower, state, upper]` active dimensions when the source exposes them.
    /// Opaque collapsed sources report only the state dimension.
    pub fn adjacent_dimensions(&self) -> [Option<usize>; 3] {
        self.adjacent_dimensions
    }

    pub fn parameters(&self) -> MaternParameters {
        self.parameters
    }

    pub fn lower_adjoint_inverse(&self) -> InverseUse {
        self.lower_adjoint_inverse
    }

    pub fn recurrence_inverse(&self) -> InverseUse {
        self.recurrence_inverse
    }

    pub fn normalization(&self) -> PriorNormalization {
        self.normalization
    }

    /// Additional deterministic precision scaling applied after construction.
    pub fn post_construction_precision_scale(&self) -> f64 {
        self.post_construction_precision_scale
    }
}

/// Optional deterministic scaling applied after Matérn construction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PriorNormalization {
    /// Preserve the precision implied by `tau`.
    #[default]
    None,
    /// Rescale the precision so its matrix trace equals the supplied value.
    TargetPrecisionTrace(f64),
}

/// An assembled Gaussian prior in precision form.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPrior {
    mean: Vec<f64>,
    precision: SparseMat,
    form_degree: Option<FormDegree>,
    boundary_elimination: Option<EssentialBoundaryElimination>,
    matern_metadata: Option<MaternPriorMetadata>,
}

/// Report from scaling a prior so one reference state has a requested
/// Mahalanobis distance from the prior mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriorMahalanobisCalibration {
    pub uncalibrated_distance: f64,
    pub target_distance: f64,
    pub precision_scale: f64,
}

impl GaussianPrior {
    /// Construct a proper prior from an explicit mean and precision.
    ///
    /// Construction validates finite entries and scale-aware symmetry, merges
    /// duplicate sparse coordinates, and performs an unshifted sparse Cholesky
    /// factorization to establish positive definiteness. The factorization can
    /// incur dimension-dependent fill and is not cached by this constructor;
    /// call [`Self::factor`] when a reusable factor is required. No
    /// symmetrization, diagonal shift, clipping, or jitter is applied.
    pub fn new(mean: Vec<f64>, precision: SparseMat) -> Result<Self> {
        if precision.nrows() != precision.ncols() || mean.len() != precision.nrows() {
            return Err(FeecGmrfError::Dimension(format!(
                "prior mean length {} and precision shape {}x{} do not match",
                mean.len(),
                precision.nrows(),
                precision.ncols()
            )));
        }
        if !mean.iter().all(|value| value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "prior mean must contain only finite values".to_string(),
            ));
        }
        let precision = crate::operator::validate_proper_precision("prior precision", precision)?;
        Ok(Self {
            mean,
            precision,
            form_degree: None,
            boundary_elimination: None,
            matern_metadata: None,
        })
    }

    pub(crate) fn with_form_degree(mut self, degree: FormDegree) -> Self {
        self.form_degree = Some(degree);
        self
    }

    pub(crate) fn with_boundary_elimination(
        mut self,
        elimination: EssentialBoundaryElimination,
    ) -> Result<Self> {
        if elimination.active_dimension() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "boundary active dimension {} does not match prior dimension {}",
                elimination.active_dimension(),
                self.dimension()
            )));
        }
        self.boundary_elimination = Some(elimination);
        Ok(self)
    }

    pub(crate) fn with_matern_metadata(mut self, metadata: MaternPriorMetadata) -> Self {
        self.matern_metadata = Some(metadata);
        self
    }

    /// Prior dimension.
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Dimension of the complete FEEC cochain before essential-boundary elimination.
    pub fn cochain_dimension(&self) -> usize {
        self.boundary_elimination.as_ref().map_or_else(
            || self.dimension(),
            EssentialBoundaryElimination::full_dimension,
        )
    }

    /// Prior mean in active coordinates.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Prior mean in full cochain ordering, including prescribed values.
    pub fn cochain_mean(&self) -> Result<Vec<f64>> {
        match &self.boundary_elimination {
            Some(elimination) => elimination.lift_state(&self.mean),
            None => Ok(self.mean.clone()),
        }
    }

    /// Prior precision in active coordinates.
    pub fn precision(&self) -> &SparseMat {
        &self.precision
    }

    /// FEEC form degree, when the prior was assembled from form operators.
    pub fn form_degree(&self) -> Option<FormDegree> {
        self.form_degree
    }

    /// Essential-boundary elimination carried by this prior, when present.
    pub fn boundary_elimination(&self) -> Option<&EssentialBoundaryElimination> {
        self.boundary_elimination.as_ref()
    }

    /// Construction metadata when this prior was built by a Matérn builder.
    pub fn matern_metadata(&self) -> Option<&MaternPriorMetadata> {
        self.matern_metadata.as_ref()
    }

    /// Factor the prior once for repeated variance, covariance, and sampling queries.
    pub fn factor(&self) -> Result<crate::infer::FactoredGaussianPrior> {
        crate::infer::FactoredGaussianPrior::new(self.clone())
    }

    /// Factor the prior while separately recording symbolic analysis and
    /// numeric-factorization wall time and actual factor storage.
    ///
    /// Peak RSS is intentionally attached later from an external fresh-process
    /// monitor. Subsequent covariance actions can append solve timings through
    /// [`crate::infer::FactoredGaussianPrior::covariance_action_with_resource_report`].
    pub fn factor_with_resource_report(
        &self,
    ) -> Result<(
        crate::infer::FactoredGaussianPrior,
        crate::infer::SparseDirectResourceReport,
    )> {
        crate::infer::FactoredGaussianPrior::new_with_resource_report(self.clone())
    }

    /// Estimate active-coordinate marginal variances.
    pub fn latent_variance_estimate(
        &self,
        method: crate::infer::VarianceMethod,
    ) -> Result<crate::infer::VarianceEstimate> {
        self.factor()?.latent_variance_estimate(method)
    }

    /// Estimate variances for an active- or full-cochain pushforward.
    pub fn pushforward_variance_estimate(
        &self,
        map: &LinearMap,
        method: crate::infer::VarianceMethod,
    ) -> Result<crate::infer::VarianceEstimate> {
        self.factor()?.pushforward_variance_estimate(map, method)
    }

    /// Estimate mapped variances and their weighted covariance trace.
    pub fn pushforward_weighted_variance_estimate(
        &self,
        map: &LinearMap,
        output_weights: &[f64],
        method: crate::infer::VarianceMethod,
    ) -> Result<crate::infer::WeightedVarianceEstimate> {
        self.factor()?
            .pushforward_weighted_variance_estimate(map, output_weights, method)
    }

    /// Compute exact marginal variances for an active- or full-cochain map.
    ///
    /// Prescribed essential-boundary coefficients contribute zero variance.
    /// The calculation uses the covariance and the linear part of the map.
    pub fn pushforward_variances(&self, map: &LinearMap) -> Result<Vec<f64>> {
        Ok(self
            .pushforward_variance_estimate(map, crate::infer::VarianceMethod::Exact)?
            .values)
    }

    /// Mahalanobis distance from the prior mean in active coordinates.
    pub fn mahalanobis_distance(&self, state: &[f64]) -> Result<f64> {
        if state.len() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "state length {} does not match prior dimension {}",
                state.len(),
                self.dimension()
            )));
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "Mahalanobis state must contain only finite values".to_string(),
            ));
        }
        let centered = state
            .iter()
            .zip(&self.mean)
            .map(|(state, mean)| state - mean)
            .collect::<Vec<_>>();
        let weighted = self
            .precision
            .apply_checked(&centered)
            .map_err(FeecGmrfError::Dimension)?;
        let squared = centered
            .iter()
            .zip(weighted)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        if !squared.is_finite() || squared < -1.0e-10 {
            return Err(FeecGmrfError::Inference(
                "prior precision produced a negative or non-finite Mahalanobis norm".to_string(),
            ));
        }
        Ok(squared.max(0.0).sqrt())
    }

    /// Scale the precision so `state` has the requested Mahalanobis distance
    /// from the unchanged prior mean.
    pub fn calibrate_to_mahalanobis_distance(
        &self,
        state: &[f64],
        target_distance: f64,
    ) -> Result<(Self, PriorMahalanobisCalibration)> {
        if !target_distance.is_finite() || target_distance <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "target Mahalanobis distance must be finite and positive".to_string(),
            ));
        }
        let uncalibrated_distance = self.mahalanobis_distance(state)?;
        if !uncalibrated_distance.is_finite() || uncalibrated_distance <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "reference state must differ from the prior mean in the precision metric"
                    .to_string(),
            ));
        }
        let precision_scale = (target_distance / uncalibrated_distance).powi(2);
        Ok((
            self.with_precision_scale(precision_scale)?,
            PriorMahalanobisCalibration {
                uncalibrated_distance,
                target_distance,
                precision_scale,
            },
        ))
    }

    /// Condition an assembled full-space Gaussian prior on prescribed
    /// essential-boundary coefficients.
    ///
    /// [`MaternPriorBuilder::essential_boundary_conditions`] performs boundary
    /// reduction before the Matérn recurrence. This method applies exact
    /// Gaussian conditioning to an existing full precision.
    pub fn condition_on_essential_boundary(
        &self,
        conditions: EssentialBoundaryConditions,
    ) -> Result<Self> {
        if self.boundary_elimination.is_some() {
            return Err(FeecGmrfError::InvalidParameter(
                "prior already carries essential-boundary elimination".to_string(),
            ));
        }
        let elimination = conditions.eliminate(self.dimension())?;
        let reduced_precision = elimination.reduce_square(&self.precision)?;
        let centered = self
            .mean
            .iter()
            .zip(elimination.prescribed_values())
            .map(|(mean, prescribed)| mean - prescribed)
            .collect::<Vec<_>>();
        let full_information = self
            .precision
            .apply_checked(&centered)
            .map_err(FeecGmrfError::Dimension)?;
        let information = elimination.reduce_state(&full_information)?;
        let gmrf = Gmrf::from_information_and_precision(
            Vector::from_vec(information),
            crate::infer::gmrf_sparse(&reduced_precision),
        )?;
        let mean = gmrf.mean().iter().copied().collect();
        let mut prior = Self::new(mean, reduced_precision)?;
        prior.form_degree = self.form_degree;
        prior.matern_metadata = self.matern_metadata.clone().map(|mut metadata| {
            metadata.boundary = MaternBoundarySemantics::ConditionedCompletedDiscreteLaw {
                state_layout: elimination.layout().clone(),
            };
            metadata
        });
        prior.with_boundary_elimination(elimination)
    }

    /// Eliminate a full-cochain map, or validate an already-active map.
    pub fn eliminate_map(&self, map: &LinearMap, bias: &[f64]) -> Result<EliminatedLinearMap> {
        if let Some(elimination) = &self.boundary_elimination {
            if map.input_dimension() == elimination.full_dimension() {
                return elimination.eliminate_map(map, bias);
            }
        }
        if map.input_dimension() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "map input dimension {} matches neither active dimension {} nor full cochain dimension {}",
                map.input_dimension(),
                self.dimension(),
                self.cochain_dimension()
            )));
        }
        if bias.len() != map.output_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "map bias length {} does not match output dimension {}",
                bias.len(),
                map.output_dimension()
            )));
        }
        Ok(EliminatedLinearMap::from_active(map.clone(), bias.to_vec()))
    }

    /// Return the same Gaussian location with its precision multiplied by a
    /// positive scalar. Consequently, every covariance is divided by `scale`.
    pub fn with_precision_scale(&self, scale: f64) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "precision scale must be finite and positive".to_string(),
            ));
        }
        let matern_metadata = self.matern_metadata.clone().map(|mut metadata| {
            metadata.post_construction_precision_scale *= scale;
            metadata
        });
        Ok(Self {
            mean: self.mean.clone(),
            precision: self.precision.scaled(scale),
            form_degree: self.form_degree,
            boundary_elimination: self.boundary_elimination.clone(),
            matern_metadata,
        })
    }
}

const DEFAULT_MAXIMUM_MATERIALIZED_DIMENSION: usize = 1024;

/// Eliminated Matérn construction from compatible adjacent FEEC operators.
///
/// The lower-adjoint and recurrence inverse metrics are separate mathematical
/// sites and are therefore configured independently. Exact choices materialize
/// the consistent inverse subject to an explicit size guard. Supplied choices
/// are validated as finite symmetric positive-definite matrices but remain
/// recorded as caller-provided approximations.
pub struct AdjacentMaternPriorBuilder {
    operators: ExactFormOperators,
    parameters: MaternParameters,
    lower_adjoint_inverse: EliminatedMassInversePolicy,
    recurrence_inverse: EliminatedMassInversePolicy,
    maximum_materialized_dimension: usize,
}

impl AdjacentMaternPriorBuilder {
    /// Build from caller-supplied compatible adjacent masses, derivatives, and layouts.
    pub fn from_operators(operators: ExactFormOperators) -> Self {
        Self {
            operators,
            parameters: MaternParameters::default(),
            lower_adjoint_inverse: EliminatedMassInversePolicy::ExactMaterialized,
            recurrence_inverse: EliminatedMassInversePolicy::ExactMaterialized,
            maximum_materialized_dimension: DEFAULT_MAXIMUM_MATERIALIZED_DIMENSION,
        }
    }

    /// Assemble compatible lowest-order Whitney spaces on the homogeneous
    /// full-relative subcomplex before either inverse is formed.
    pub fn from_feec_homogeneous_full_relative(
        topology: &Complex,
        metric: &MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        Ok(Self::from_operators(
            ExactFormOperators::from_feec_homogeneous_full_relative(topology, metric, degree)?,
        ))
    }

    /// Assemble the homogeneous full-relative Whitney complex first, then
    /// construct projected sparse inverse metrics on the already-reduced
    /// adjacent spaces.
    ///
    /// This boundary-first ordering is the coefficient-space construction used
    /// by the projected inverse consistency argument. It generally differs
    /// from assembling a full-space projected inverse and taking an active
    /// principal block afterwards. The returned law remains an explicitly
    /// approximate eliminated Gaussian prior.
    pub fn from_feec_homogeneous_full_relative_with_projected_inverses(
        topology: &Complex,
        metric: &MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        let operators =
            ExactFormOperators::from_feec_homogeneous_full_relative(topology, metric, degree)?;
        let state_inverse = boundary_first_projected_inverse(
            topology,
            metric,
            degree,
            operators.state().boundary_layout(),
            operators.state().mass(),
        )?;
        let lower_inverse = operators
            .incoming()
            .map(|incoming| {
                boundary_first_projected_inverse(
                    topology,
                    metric,
                    degree - 1,
                    incoming.lower().boundary_layout(),
                    incoming.lower().mass(),
                )
            })
            .transpose()?;
        let mut builder = Self::from_operators(operators)
            .recurrence_inverse(EliminatedMassInversePolicy::Provided(state_inverse));
        if let Some(lower_inverse) = lower_inverse {
            builder =
                builder.lower_adjoint_inverse(EliminatedMassInversePolicy::Provided(lower_inverse));
        }
        Ok(builder)
    }

    /// Set direct `alpha`, `kappa`, and `tau` parameters.
    pub fn parameters(mut self, parameters: MaternParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Select the inverse metric inside the lower-adjoint Hodge branch.
    ///
    /// This setting is ignored, and reported as not applicable, for degree
    /// zero where no incoming branch exists.
    pub fn lower_adjoint_inverse(mut self, policy: EliminatedMassInversePolicy) -> Self {
        self.lower_adjoint_inverse = policy;
        self
    }

    /// Select the degree-`k` inverse metric between repeated Matérn factors.
    ///
    /// This setting is ignored, and reported as unused, when `alpha = 1`.
    pub fn recurrence_inverse(mut self, policy: EliminatedMassInversePolicy) -> Self {
        self.recurrence_inverse = policy;
        self
    }

    /// Change the resource guard for each dense inverse that may be materialized.
    ///
    /// Raising this limit changes resource exposure, not the represented law.
    pub fn maximum_materialized_dimension(mut self, maximum: usize) -> Result<Self> {
        if maximum == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "maximum materialized dimension must be positive".to_string(),
            ));
        }
        self.maximum_materialized_dimension = maximum;
        Ok(self)
    }

    /// Assemble the centered eliminated precision.
    pub fn build(self) -> Result<GaussianPrior> {
        let Self {
            operators,
            parameters,
            lower_adjoint_inverse,
            recurrence_inverse,
            maximum_materialized_dimension,
        } = self;

        build_eliminated_adjacent_matern_prior(
            operators,
            parameters,
            lower_adjoint_inverse,
            recurrence_inverse,
            maximum_materialized_dimension,
        )
    }
}

fn boundary_first_projected_inverse(
    topology: &Complex,
    metric: &MeshLengths,
    degree: usize,
    layout: &BoundaryLayout,
    active_mass: &SparseMat,
) -> Result<SparseMat> {
    if degree == topology.dim() {
        return diagonal_inverse(active_mass);
    }
    let feec_layout = formoniq::reduction::DofLayout::from_parts(
        layout.full_dimension(),
        layout.active_dofs().to_vec(),
        layout
            .fixed_dofs()
            .iter()
            .map(|(index, value)| formoniq::reduction::PrescribedDof {
                index: *index,
                value: *value,
            })
            .collect(),
    )
    .map_err(FeecGmrfError::Assembly)?;
    let inverse =
        formoniq::assemble::assemble_reduced_whitney_projected_sparse_inverse_galmat_for_grade(
            topology,
            metric,
            degree,
            &feec_layout,
        )
        .map_err(FeecGmrfError::Assembly)?;
    let inverse = common::linalg::nalgebra::CsrMatrix::from(&inverse);
    Ok(feg_infer::sparse::feec_csr_to_core_triplet(&inverse))
}

/// Small-system convenience constructor for the exact full-metric eliminated
/// FEEC law.
///
/// Every required inverse is materialized without dropping entries. This is a
/// reference implementation for theorem checks and comparator runs, not the
/// scalable constrained representation.
pub struct ExactMaternPriorBuilder {
    inner: AdjacentMaternPriorBuilder,
}

impl ExactMaternPriorBuilder {
    /// Build from caller-supplied adjacent masses, derivatives, and layouts.
    pub fn from_operators(operators: ExactFormOperators) -> Self {
        Self {
            inner: AdjacentMaternPriorBuilder::from_operators(operators),
        }
    }

    /// Assemble compatible lowest-order Whitney spaces on the homogeneous
    /// full-relative subcomplex before either inverse is formed.
    pub fn from_feec_homogeneous_full_relative(
        topology: &Complex,
        metric: &MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        Ok(Self::from_operators(
            ExactFormOperators::from_feec_homogeneous_full_relative(topology, metric, degree)?,
        ))
    }

    /// Set direct `alpha`, `kappa`, and `tau` parameters.
    pub fn parameters(mut self, parameters: MaternParameters) -> Self {
        self.inner = self.inner.parameters(parameters);
        self
    }

    /// Change the resource guard for each dense inverse that may be materialized.
    ///
    /// Raising this limit changes resource exposure, not the represented law.
    pub fn maximum_materialized_dimension(mut self, maximum: usize) -> Result<Self> {
        self.inner = self.inner.maximum_materialized_dimension(maximum)?;
        Ok(self)
    }

    /// Assemble the centered exact eliminated precision.
    pub fn build(self) -> Result<GaussianPrior> {
        self.inner.build()
    }
}

fn build_eliminated_adjacent_matern_prior(
    operators: ExactFormOperators,
    parameters: MaternParameters,
    lower_adjoint_inverse: EliminatedMassInversePolicy,
    recurrence_inverse: EliminatedMassInversePolicy,
    maximum_materialized_dimension: usize,
) -> Result<GaussianPrior> {
    let blocks = assemble_exact_matern_blocks(&operators, parameters.kappa)?;

    let mut system = blocks.local_system.clone();
    let lower_use = if let Some(lower) = &blocks.lower {
        let lower_kind = lower_adjoint_inverse.kind();
        let lower_inverse = resolve_eliminated_mass_inverse(
            "lower-adjoint mass inverse",
            &lower.mass,
            lower_adjoint_inverse,
            maximum_materialized_dimension,
        )?;
        let solved = multiply_sparse(&lower.coupling, &lower_inverse)?;
        let lower_term = multiply_sparse(&solved, &lower.coupling.transpose())?;
        system = add_sparse(&system, &lower_term)?;
        InverseUse::Applied(lower_kind)
    } else {
        InverseUse::NotApplicable
    };
    let system = crate::operator::validate_proper_precision("adjacent Matérn system", system)?;

    let (precision, recurrence_use) = if parameters.alpha == MaternAlpha::One {
        (system.scaled(parameters.tau.powi(2)), InverseUse::NotUsed)
    } else {
        let recurrence_kind = recurrence_inverse.kind();
        let mass_inverse = resolve_eliminated_mass_inverse(
            "recurrence mass inverse",
            &blocks.mass,
            recurrence_inverse,
            maximum_materialized_dimension,
        )?;
        (
            matern_recurrence(&system, &mass_inverse, parameters.alpha, parameters.tau)?,
            InverseUse::Applied(recurrence_kind),
        )
    };

    let representation = eliminated_adjacent_representation(lower_use, recurrence_use);
    let metadata = exact_matern_metadata(
        &operators,
        parameters,
        representation,
        lower_use,
        recurrence_use,
    );

    let state_layout = operators.state().boundary_layout().clone();
    let needs_elimination = state_layout.full_dimension() != operators.state().dimension()
        || !state_layout.fixed_dofs().is_empty();
    let prior = GaussianPrior::new(vec![0.0; operators.state().dimension()], precision)?
        .with_form_degree(operators.degree())
        .with_matern_metadata(metadata);
    if needs_elimination {
        prior.with_boundary_elimination(EssentialBoundaryElimination::from_layout(state_layout)?)
    } else {
        Ok(prior)
    }
}

fn resolve_eliminated_mass_inverse(
    name: &'static str,
    mass: &SparseMat,
    policy: EliminatedMassInversePolicy,
    maximum_materialized_dimension: usize,
) -> Result<SparseMat> {
    match policy {
        EliminatedMassInversePolicy::ExactMaterialized => {
            enforce_materialization_limit(mass.nrows(), maximum_materialized_dimension)?;
            materialize_exact_mass_inverse(mass)
        }
        EliminatedMassInversePolicy::Provided(inverse) => {
            if inverse.nrows() != mass.nrows() || inverse.ncols() != mass.ncols() {
                return Err(FeecGmrfError::Dimension(format!(
                    "provided {name} shape {}x{} does not match mass shape {}x{}",
                    inverse.nrows(),
                    inverse.ncols(),
                    mass.nrows(),
                    mass.ncols()
                )));
            }
            crate::operator::validate_proper_precision(name, inverse)
        }
    }
}

fn eliminated_adjacent_representation(
    lower: InverseUse,
    recurrence: InverseUse,
) -> MaternRepresentation {
    let used = [lower, recurrence]
        .into_iter()
        .filter_map(|usage| match usage {
            InverseUse::Applied(kind) => Some(kind),
            InverseUse::NotApplicable | InverseUse::NotUsed => None,
        })
        .collect::<Vec<_>>();
    let provided = used
        .iter()
        .filter(|kind| **kind == MassInverseKind::Provided)
        .count();
    match (provided, used.len()) {
        (0, _) => MaternRepresentation::ExactEliminatedReference,
        (count, total) if count == total => {
            MaternRepresentation::ProvidedAdjacentEliminatedApproximation
        }
        _ => MaternRepresentation::MixedAdjacentEliminatedApproximation,
    }
}

#[derive(Debug, Clone)]
struct ExactMaternLowerBlocks {
    mass: SparseMat,
    coupling: SparseMat,
}

#[derive(Debug, Clone)]
struct ExactMaternBlocks {
    mass: SparseMat,
    local_system: SparseMat,
    lower: Option<ExactMaternLowerBlocks>,
}

/// Assemble the one canonical set of local blocks shared by exact eliminated
/// and exact supported Matérn representations.
fn assemble_exact_matern_blocks(
    operators: &ExactFormOperators,
    kappa: f64,
) -> Result<ExactMaternBlocks> {
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "exact Matérn block assembly requires finite positive kappa".to_string(),
        ));
    }
    let mass = operators.state().mass().clone();
    let mut local_system = mass.scaled(kappa.powi(2));
    if let Some(outgoing) = operators.outgoing() {
        let derivative = outgoing.active_derivative().matrix();
        let weighted = multiply_sparse(outgoing.upper().mass(), derivative)?;
        let upper_term = multiply_sparse(&derivative.transpose(), &weighted)?;
        local_system = add_sparse(&local_system, &upper_term)?;
    }
    let lower = operators
        .incoming()
        .map(|incoming| -> Result<ExactMaternLowerBlocks> {
            Ok(ExactMaternLowerBlocks {
                mass: incoming.lower().mass().clone(),
                coupling: multiply_sparse(&mass, incoming.active_derivative().matrix())?,
            })
        })
        .transpose()?;
    Ok(ExactMaternBlocks {
        mass,
        local_system,
        lower,
    })
}

/// Describe the shared FEEC law independently of its eliminated or supported
/// finite-dimensional representation.
fn exact_matern_metadata(
    operators: &ExactFormOperators,
    parameters: MaternParameters,
    representation: MaternRepresentation,
    lower_adjoint_inverse: InverseUse,
    recurrence_inverse: InverseUse,
) -> MaternPriorMetadata {
    let layouts = AdjacentBoundaryLayouts::from_exact(operators);
    let boundary = match operators.boundary_domain() {
        ExactBoundaryDomain::HomogeneousFullRelative => {
            MaternBoundarySemantics::AssembledHomogeneousFullRelative {
                layouts: layouts.clone(),
            }
        }
        ExactBoundaryDomain::UserSuppliedCompatibleSubcomplex { name } => {
            MaternBoundarySemantics::UserSuppliedCompatibleSubcomplex {
                name: name.clone(),
                layouts: layouts.clone(),
            }
        }
    };
    let operator_source = match operators.provenance() {
        FormOperatorProvenance::FeecWhitneyHomogeneousFullRelative => {
            MaternOperatorSource::FeecWhitney
        }
        FormOperatorProvenance::UserSuppliedAdjacentOperators { label } => {
            MaternOperatorSource::UserSuppliedAdjacent {
                label: label.clone(),
            }
        }
    };
    MaternPriorMetadata {
        representation,
        operator_source,
        boundary,
        form_degree: operators.degree(),
        complex_dimension: operators.complex_dimension(),
        adjacent_dimensions: [
            operators
                .incoming()
                .map(|branch| branch.lower().dimension()),
            Some(operators.state().dimension()),
            operators
                .outgoing()
                .map(|branch| branch.upper().dimension()),
        ],
        parameters,
        lower_adjoint_inverse,
        recurrence_inverse,
        normalization: PriorNormalization::None,
        post_construction_precision_scale: 1.0,
    }
}

fn enforce_materialization_limit(dimension: usize, maximum: usize) -> Result<()> {
    if dimension > maximum {
        return Err(FeecGmrfError::ExactReferenceSizeExceeded { dimension, maximum });
    }
    Ok(())
}

fn materialize_exact_mass_inverse(mass: &SparseMat) -> Result<SparseMat> {
    if mass.nrows() == 0 && mass.ncols() == 0 {
        return Ok(SparseMat::new(0, 0));
    }
    let mass = feg_infer::sparse::core_triplet_to_feec_csr(mass);
    let inverse = feg_infer::prior::matern::generic::build_exact_dense_mass_inverse(&mass)
        .map_err(FeecGmrfError::Assembly)?;
    Ok(feg_infer::sparse::feec_csr_to_core_triplet(&inverse))
}

enum MaternSource<'a> {
    Operators(Box<FormOperators>),
    Feec {
        topology: &'a Complex,
        metric: &'a MeshLengths,
        degree: FormDegree,
    },
}

/// Builder for Matérn GMRFs on arbitrary FEEC form spaces.
pub struct MaternPriorBuilder<'a> {
    source: MaternSource<'a>,
    parameters: MaternParameters,
    lower_adjoint_mass_inverse: Option<LowerAdjointMassInversePolicy>,
    recurrence_mass_inverse: RecurrenceMassInversePolicy,
    normalization: PriorNormalization,
    mean: Option<Vec<f64>>,
    essential_boundary_conditions: Option<EssentialBoundaryConditions>,
}

impl MaternPriorBuilder<'static> {
    /// Start from user-supplied weak mass and Hodge--Laplacian operators.
    pub fn from_operators(operators: FormOperators) -> Self {
        Self {
            source: MaternSource::Operators(Box::new(operators)),
            parameters: MaternParameters::default(),
            lower_adjoint_mass_inverse: None,
            recurrence_mass_inverse: RecurrenceMassInversePolicy::RowSumLumped,
            normalization: PriorNormalization::None,
            mean: None,
            essential_boundary_conditions: None,
        }
    }
}

impl<'a> MaternPriorBuilder<'a> {
    /// Start from a FEEC topology, metric, and form degree.
    #[deprecated(
        note = "use from_feec_whitney_projected; this compatibility constructor builds the same explicitly approximate Whitney law"
    )]
    pub fn from_feec(
        topology: &'a Complex,
        metric: &'a MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        Self::from_feec_whitney_projected(topology, metric, degree)
    }

    /// Start from the existing compact Whitney construction with projected
    /// lower-adjoint and recurrence inverse policies.
    pub fn from_feec_whitney_projected(
        topology: &'a Complex,
        metric: &'a MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        let degree = FormDegree::new(degree, topology.dim())?;
        Ok(Self {
            source: MaternSource::Feec {
                topology,
                metric,
                degree,
            },
            parameters: MaternParameters::default(),
            lower_adjoint_mass_inverse: Some(LowerAdjointMassInversePolicy::WhitneyProjected),
            recurrence_mass_inverse: RecurrenceMassInversePolicy::WhitneyProjected,
            normalization: PriorNormalization::None,
            mean: None,
            essential_boundary_conditions: None,
        })
    }

    /// Set Matérn parameters.
    pub fn parameters(mut self, parameters: MaternParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set the recurrence/load mass-inverse policy.
    ///
    /// This compatibility method never changes the lower-adjoint inverse
    /// inside the Hodge Laplacian. New code should use
    /// [`Self::recurrence_mass_inverse`].
    #[deprecated(
        note = "use recurrence_mass_inverse; this method never controls the lower-adjoint inverse"
    )]
    pub fn mass_inverse(mut self, policy: MassInversePolicy) -> Self {
        self.recurrence_mass_inverse = policy.into();
        self
    }

    /// Set the lower-adjoint mass-inverse policy used to assemble the Hodge term.
    pub fn lower_adjoint_mass_inverse(mut self, policy: LowerAdjointMassInversePolicy) -> Self {
        self.lower_adjoint_mass_inverse = Some(policy);
        self
    }

    /// Set the degree-`k` recurrence/load mass-inverse policy.
    pub fn recurrence_mass_inverse(mut self, policy: RecurrenceMassInversePolicy) -> Self {
        self.recurrence_mass_inverse = policy;
        self
    }

    /// Set a deterministic post-construction normalization.
    pub fn normalization(mut self, normalization: PriorNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set a non-zero prior mean. For a boundary-aware builder this may use
    /// active ordering or full cochain ordering. A supplied full mean must
    /// agree with every prescribed essential value.
    pub fn mean(mut self, mean: Vec<f64>) -> Self {
        self.mean = Some(mean);
        self
    }

    /// Eliminate prescribed essential-boundary coefficients before applying
    /// the Matérn recurrence.
    pub fn essential_boundary_conditions(
        mut self,
        conditions: EssentialBoundaryConditions,
    ) -> Self {
        self.essential_boundary_conditions = Some(conditions);
        self
    }

    /// Assemble the prior precision through the grade-independent recurrence.
    pub fn build(self) -> Result<GaussianPrior> {
        let Self {
            source,
            parameters,
            lower_adjoint_mass_inverse,
            recurrence_mass_inverse,
            normalization,
            mean,
            essential_boundary_conditions,
        } = self;
        let needs_projected_recurrence = parameters.alpha != MaternAlpha::One
            && matches!(
                recurrence_mass_inverse,
                RecurrenceMassInversePolicy::WhitneyProjected
            );
        let (mut operators, mut projected_inverse, representation, operator_source, lower_use) =
            match source {
                MaternSource::Operators(operators) => {
                    if lower_adjoint_mass_inverse.is_some() {
                        return Err(FeecGmrfError::Unsupported(
                        "the lower-adjoint inverse is already embedded in a caller-supplied collapsed Hodge operator"
                            .to_string(),
                    ));
                    }
                    (
                        *operators,
                        None,
                        MaternRepresentation::ProvidedCollapsedEliminated,
                        MaternOperatorSource::UserSuppliedCollapsed,
                        InverseUse::Applied(MassInverseKind::EmbeddedInCollapsedHodge),
                    )
                }
                MaternSource::Feec {
                    topology,
                    metric,
                    degree,
                } => {
                    let lower_policy = lower_adjoint_mass_inverse.unwrap_or_default();
                    let lower_use = if degree.get() == 0 {
                        InverseUse::NotApplicable
                    } else {
                        InverseUse::Applied(lower_policy.kind())
                    };
                    let hodge = if degree.get() == 0 {
                    feg_infer::prior::matern::generic::build_hodge_laplacian_form(
                        topology,
                        metric,
                        degree.get(),
                    )
                } else {
                    match &lower_policy {
                        LowerAdjointMassInversePolicy::WhitneyProjected => {
                            feg_infer::prior::matern::generic::build_hodge_laplacian_form(
                                topology,
                                metric,
                                degree.get(),
                            )
                        }
                        LowerAdjointMassInversePolicy::RowSumLumped => {
                            feg_infer::prior::matern::generic::build_hodge_laplacian_form_with_row_sum_lower_mass_inverse(
                                topology,
                                metric,
                                degree.get(),
                            )
                        }
                        LowerAdjointMassInversePolicy::Diagonal => {
                            feg_infer::prior::matern::generic::build_hodge_laplacian_form_with_diagonal_lower_mass_inverse(
                                topology,
                                metric,
                                degree.get(),
                            )
                        }
                        LowerAdjointMassInversePolicy::Provided(inverse) => {
                            let inverse =
                                feg_infer::sparse::core_triplet_to_feec_csr(inverse);
                            feg_infer::prior::matern::generic::build_hodge_laplacian_form_with_lower_mass_inverse(
                                topology,
                                metric,
                                degree.get(),
                                &inverse,
                            )
                        }
                    }
                }
                .map_err(FeecGmrfError::Assembly)?;
                    let mass = feg_infer::sparse::feec_csr_to_core_triplet(&hodge.mass_u);
                    let laplacian = feg_infer::sparse::feec_csr_to_core_triplet(&hodge.laplacian);
                    let projected_inverse = needs_projected_recurrence
                    .then(|| {
                        feg_infer::prior::matern::generic::build_projected_or_top_degree_mass_inverse(
                        topology,
                        metric,
                        degree.get(),
                        &hodge.mass_u,
                    )
                    })
                    .transpose()
                    .map_err(FeecGmrfError::Assembly)?
                    .map(|inverse| feg_infer::sparse::feec_csr_to_core_triplet(&inverse));
                    (
                        FormOperators::new(degree, topology.dim(), mass, laplacian)?,
                        projected_inverse,
                        MaternRepresentation::WhitneyApproximateEliminated,
                        MaternOperatorSource::FeecWhitney,
                        lower_use,
                    )
                }
            };

        let original_dimension = operators.dimension();
        let embedded_layout = operators.boundary_layout().clone();
        let already_reduced = embedded_layout.full_dimension() != operators.dimension()
            || !embedded_layout.fixed_dofs().is_empty();
        let mut boundary_semantics = if already_reduced {
            MaternBoundarySemantics::UserSuppliedReducedCollapsedOperator {
                state_layout: embedded_layout.clone(),
            }
        } else {
            MaternBoundarySemantics::UnrestrictedDiscreteComplex
        };
        let mut boundary_elimination = if already_reduced {
            Some(EssentialBoundaryElimination::from_layout(embedded_layout)?)
        } else {
            None
        };
        if let Some(conditions) = essential_boundary_conditions {
            if already_reduced {
                return Err(FeecGmrfError::InvalidParameter(
                    "cannot apply essential boundary conditions to operators that are already boundary-reduced"
                        .to_string(),
                ));
            }
            let elimination = conditions.eliminate(original_dimension)?;
            let reduced_mass = elimination.reduce_square(operators.mass())?;
            let reduced_laplacian = elimination.reduce_square(operators.hodge_laplacian())?;
            operators = FormOperators::new(
                operators.degree(),
                operators.complex_dimension(),
                reduced_mass,
                reduced_laplacian,
            )?
            .with_boundary_layout(elimination.layout().clone())?;
            projected_inverse = projected_inverse
                .as_ref()
                .map(|inverse| elimination.reduce_square(inverse))
                .transpose()?;
            boundary_semantics = MaternBoundarySemantics::PrincipalRestrictedCollapsedOperator {
                state_layout: elimination.layout().clone(),
            };
            boundary_elimination = Some(elimination);
        }

        let system = add_sparse(
            operators.hodge_laplacian(),
            &operators.mass().scaled(parameters.kappa.powi(2)),
        )?;
        let (mut precision, recurrence_use) = if parameters.alpha == MaternAlpha::One {
            (system.scaled(parameters.tau.powi(2)), InverseUse::NotUsed)
        } else {
            let recurrence_kind =
                recurrence_mass_inverse.kind(operators.degree(), operators.complex_dimension());
            let mass_inverse = match recurrence_mass_inverse {
                RecurrenceMassInversePolicy::RowSumLumped => row_sum_inverse(operators.mass())?,
                RecurrenceMassInversePolicy::Diagonal => diagonal_inverse(operators.mass())?,
                RecurrenceMassInversePolicy::WhitneyProjected => {
                    projected_inverse.ok_or_else(|| {
                        FeecGmrfError::Unsupported(
                            "Whitney-projected recurrence inversion requires the FEEC constructor or an explicit inverse"
                                .to_string(),
                        )
                    })?
                }
                RecurrenceMassInversePolicy::Provided(inverse) => {
                    if inverse.nrows() == operators.dimension()
                        && inverse.ncols() == operators.dimension()
                    {
                        inverse
                    } else if let Some(elimination) = &boundary_elimination {
                        elimination.reduce_square(&inverse)?
                    } else {
                        return Err(FeecGmrfError::Dimension(
                            "provided recurrence mass inverse does not match the form-space dimension"
                                .to_string(),
                        ));
                    }
                }
            };
            (
                matern_recurrence(&system, &mass_inverse, parameters.alpha, parameters.tau)?,
                InverseUse::Applied(recurrence_kind),
            )
        };
        apply_normalization(&mut precision, normalization)?;
        let mean =
            resolve_reduced_mean(mean, operators.dimension(), boundary_elimination.as_ref())?;
        let metadata = MaternPriorMetadata {
            representation,
            operator_source,
            boundary: boundary_semantics,
            form_degree: operators.degree(),
            complex_dimension: operators.complex_dimension(),
            adjacent_dimensions: [None, Some(operators.dimension()), None],
            parameters,
            lower_adjoint_inverse: lower_use,
            recurrence_inverse: recurrence_use,
            normalization,
            post_construction_precision_scale: 1.0,
        };
        let prior = GaussianPrior::new(mean, precision)?
            .with_form_degree(operators.degree())
            .with_matern_metadata(metadata);
        match boundary_elimination {
            Some(elimination) => prior.with_boundary_elimination(elimination),
            None => Ok(prior),
        }
    }
}

fn resolve_reduced_mean(
    mean: Option<Vec<f64>>,
    active_dimension: usize,
    elimination: Option<&EssentialBoundaryElimination>,
) -> Result<Vec<f64>> {
    let Some(mean) = mean else {
        return Ok(vec![0.0; active_dimension]);
    };
    let Some(elimination) = elimination else {
        if mean.len() != active_dimension {
            return Err(FeecGmrfError::Dimension(format!(
                "prior mean length {} does not match dimension {active_dimension}",
                mean.len()
            )));
        }
        return Ok(mean);
    };
    if mean.len() == active_dimension {
        return Ok(mean);
    }
    if mean.len() != elimination.full_dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "boundary-aware prior mean length {} matches neither active dimension {} nor full dimension {}",
            mean.len(),
            active_dimension,
            elimination.full_dimension()
        )));
    }
    for &(index, prescribed) in elimination.layout().fixed_dofs() {
        let supplied = mean[index];
        let tolerance = 1.0e-12 * (1.0 + prescribed.abs());
        if (supplied - prescribed).abs() > tolerance {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "prior mean at prescribed dof {index} is {supplied}, expected {prescribed}"
            )));
        }
    }
    elimination.reduce_state(&mean)
}

/// Apply the Matérn precision recurrence to an assembled system and mass inverse.
pub fn matern_recurrence(
    system: &SparseMat,
    mass_inverse: &SparseMat,
    alpha: MaternAlpha,
    tau: f64,
) -> Result<SparseMat> {
    if system.nrows() != system.ncols()
        || mass_inverse.nrows() != system.nrows()
        || mass_inverse.ncols() != system.ncols()
    {
        return Err(FeecGmrfError::Dimension(
            "Matérn system and mass inverse must be square and share a dimension".to_string(),
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "Matérn tau must be finite and positive".to_string(),
        ));
    }
    let system = feg_infer::sparse::core_triplet_to_feec_csr(system);
    let mass_inverse = feg_infer::sparse::core_triplet_to_feec_csr(mass_inverse);
    let precision = feg_infer::prior::matern::build_lindgren_precision_from_system(
        &system,
        &mass_inverse,
        alpha,
        tau,
    );
    Ok(feg_infer::sparse::feec_csr_to_core_triplet(&precision))
}

fn diagonal_inverse(mass: &SparseMat) -> Result<SparseMat> {
    let mut diagonal = vec![0.0; mass.nrows()];
    for (row, col, value) in mass.triplet_iter() {
        if row == col {
            diagonal[row] += value;
        }
    }
    inverse_diagonal(&diagonal, "mass diagonal")
}

fn row_sum_inverse(mass: &SparseMat) -> Result<SparseMat> {
    let mut sums = vec![0.0; mass.nrows()];
    for (row, _, value) in mass.triplet_iter() {
        sums[row] += value;
    }
    inverse_diagonal(&sums, "row-sum lumped mass")
}

fn inverse_diagonal(values: &[f64], name: &str) -> Result<SparseMat> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(FeecGmrfError::InvalidParameter(format!(
            "{name} entries must be finite and positive"
        )));
    }
    Ok(SparseMat::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, value)| SparseTriplet {
                row: index,
                col: index,
                value: value.recip(),
            }),
    ))
}

fn apply_normalization(precision: &mut SparseMat, normalization: PriorNormalization) -> Result<()> {
    let PriorNormalization::TargetPrecisionTrace(target) = normalization else {
        return Ok(());
    };
    if !target.is_finite() || target <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "target precision trace must be finite and positive".to_string(),
        ));
    }
    let trace = precision
        .triplet_iter()
        .filter(|(row, col, _)| row == col)
        .map(|(_, _, value)| value)
        .sum::<f64>();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "assembled precision trace must be finite and positive".to_string(),
        ));
    }
    *precision = precision.scaled(target / trace);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proper_matern_parameters_reject_intrinsic_kappa() {
        let error = MaternParameters::new(MaternAlpha::Two, 0.0, 1.0).unwrap_err();
        assert!(matches!(error, FeecGmrfError::IntrinsicModelUnsupported(_)));

        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                MaternParameters::new(MaternAlpha::Two, invalid, 1.0),
                Err(FeecGmrfError::InvalidParameter(_))
            ));
        }
    }

    #[test]
    fn exact_pushforward_variances_support_non_diagonal_precision() {
        let precision =
            SparseMat::from_rows(2, &[vec![(0, 2.0), (1, 1.0)], vec![(0, 1.0), (1, 2.0)]]).unwrap();
        let prior = GaussianPrior::new(vec![3.0, -2.0], precision).unwrap();
        let map =
            LinearMap::weighted_rows(2, &[vec![(0, 1.0), (1, 1.0)], vec![(0, 1.0), (1, -1.0)]])
                .unwrap();

        let variances = prior.pushforward_variances(&map).unwrap();
        assert!((variances[0] - 2.0 / 3.0).abs() < 1.0e-12);
        assert!((variances[1] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn mahalanobis_distance_is_mean_relative() {
        let prior = GaussianPrior::new(vec![1.0, -1.0], SparseMat::diagonal(2, 4.0)).unwrap();
        assert!((prior.mahalanobis_distance(&[2.0, 1.0]).unwrap() - 20.0_f64.sqrt()).abs() < 1e-12);
        assert!(prior.mahalanobis_distance(&[1.0]).is_err());

        let (calibrated, report) = prior
            .calibrate_to_mahalanobis_distance(&[2.0, 1.0], 1.0)
            .unwrap();
        assert!((calibrated.mahalanobis_distance(&[2.0, 1.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!((report.precision_scale - 0.05).abs() < 1e-12);
        assert_eq!(calibrated.mean(), prior.mean());
    }

    #[test]
    fn recurrence_is_independent_of_form_degree() {
        let a = SparseMat::diagonal(2, 2.0);
        let m_inv = SparseMat::diagonal(2, 0.5);
        let q1 = matern_recurrence(&a, &m_inv, MaternAlpha::One, 1.0).unwrap();
        let q2 = matern_recurrence(&a, &m_inv, MaternAlpha::Two, 1.0).unwrap();
        let q3 = matern_recurrence(&a, &m_inv, MaternAlpha::Three, 1.0).unwrap();
        assert_eq!(q1.apply_checked(&[1.0, 1.0]).unwrap(), vec![2.0, 2.0]);
        assert_eq!(q2.apply_checked(&[1.0, 1.0]).unwrap(), vec![2.0, 2.0]);
        assert_eq!(q3.apply_checked(&[1.0, 1.0]).unwrap(), vec![2.0, 2.0]);
    }

    #[test]
    fn alpha_one_does_not_validate_or_apply_recurrence_inverse() {
        let degree = FormDegree::new(1, 2).unwrap();
        let mass = SparseMat::diagonal(2, 2.0);
        let hodge = SparseMat::diagonal(2, 1.0);
        let parameters = MaternParameters::new(MaternAlpha::One, 1.0, 2.0).unwrap();
        let prior =
            MaternPriorBuilder::from_operators(FormOperators::new(degree, 2, mass, hodge).unwrap())
                .parameters(parameters)
                .recurrence_mass_inverse(RecurrenceMassInversePolicy::Provided(
                    SparseMat::diagonal(1, 1.0),
                ))
                .build()
                .unwrap();

        assert_eq!(
            prior.matern_metadata().unwrap().recurrence_inverse(),
            InverseUse::NotUsed
        );
        assert_eq!(prior.precision(), &SparseMat::diagonal(2, 12.0));
    }
}
