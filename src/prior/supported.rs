//! Exact Matérn laws represented by sparse energies on constrained supports.
//!
//! These types preserve the local FEEC blocks and use a certified indefinite
//! KKT solve. The KKT matrix is not exposed as an ordinary Gaussian precision.

use super::{
    assemble_exact_matern_blocks, exact_matern_metadata, ExactMaternBlocks, InverseUse,
    MassInverseKind, MaternParameters, MaternPriorMetadata, MaternRepresentation,
};
use crate::boundary::{EliminatedLinearMap, EssentialBoundaryElimination};
use crate::infer::{
    exact_estimate, public_estimate, public_weighted_trace, VarianceMethod,
    WeightedCovarianceTraceEstimate, WeightedVarianceEstimate,
};
use crate::model::{GaussianNoise, LinearConstraint, LinearObservation};
use crate::operator::{BoundaryLayout, ExactFormOperators, FormDegree, LinearMap, SparseMat};
use crate::{FeecGmrfError, Result};
use feg_core::SparseTriplet;
use gmrf_core::{
    ht_precision_weighted_observations, ht_weighted_observations, linear_observation_update,
    LinearObservationTerm, SelectedCoordinates, SparseCholeskyDiagnostics, SparseMatrix,
    SparseRowOperator, SupportedGaussianCertificateResiduals, SupportedGaussianFactor,
    SupportedGaussianSystem, TransformedVarianceMode, Vector,
};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};
use std::ops::Range;

/// Version of the interim scale-relative transformed-covariance audit.
///
/// This policy records and projects small sparse-KKT roundoff, but it is not
/// residual-aware. A future policy must replace it once KKT action residuals
/// are exposed by `gmrf-core`.
pub const INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION: &str =
    "supported-transformed-covariance-interim-v1";

/// Frozen relative-skew limit for the interim transformed-covariance audit.
///
/// This is an explicit numerical policy, not a claim about KKT forward error.
/// It must not be increased to accommodate a failing study.
pub const INTERIM_SUPPORTED_COVARIANCE_RELATIVE_SKEW_LIMIT: f64 = 1.0e-10;

/// Diagnostics for the interim audit of a materialized transformed covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportedCovarianceSymmetryAudit {
    /// Stable identifier for the numerical policy that accepted the matrix.
    pub policy_version: &'static str,
    /// Output covariance dimension.
    pub dimension: usize,
    /// Largest absolute entry in the raw covariance.
    pub raw_scale: f64,
    /// Largest absolute difference between transposed raw entries.
    pub raw_maximum_absolute_skew: f64,
    /// Maximum raw skew divided by [`Self::raw_scale`], or zero for a zero matrix.
    pub raw_maximum_relative_skew: f64,
    /// Row of the pair attaining [`Self::raw_maximum_absolute_skew`].
    pub raw_maximum_skew_row: usize,
    /// Column of the pair attaining [`Self::raw_maximum_absolute_skew`].
    pub raw_maximum_skew_column: usize,
    /// Frozen relative-skew acceptance limit.
    pub relative_skew_limit: f64,
    /// Largest absolute entry changed by the symmetric projection.
    pub projection_maximum_absolute_correction: f64,
    /// Frobenius norm of the symmetric-projection correction.
    pub projection_frobenius_correction: f64,
    /// Whether KKT residuals participated in this audit.
    pub residual_aware: bool,
    /// Explicit status explaining the residual-awareness limitation.
    pub residual_status: &'static str,
    /// Acceptance status for machine-readable study archives.
    pub status: &'static str,
}

/// Raw and audited-symmetric forms of a small transformed covariance matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditedSupportedCovariance {
    /// Direct column-by-column covariance actions before symmetry projection.
    pub raw: Vec<Vec<f64>>,
    /// Pairwise symmetric projection used as the covariance result.
    pub symmetric: Vec<Vec<f64>>,
    /// Numerical audit associated with the projection.
    pub audit: SupportedCovarianceSymmetryAudit,
}

/// Ambient block ordering of an exact supported Matérn construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportedMaternLayout {
    state: Range<usize>,
    lower_adjoint_auxiliaries: Vec<Range<usize>>,
    recurrence_states: Vec<Range<usize>>,
    ambient_dimension: usize,
    constraint_dimension: usize,
}

impl SupportedMaternLayout {
    pub fn state(&self) -> Range<usize> {
        self.state.clone()
    }

    pub fn lower_adjoint_auxiliaries(&self) -> &[Range<usize>] {
        &self.lower_adjoint_auxiliaries
    }

    pub fn recurrence_states(&self) -> &[Range<usize>] {
        &self.recurrence_states
    }

    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    pub fn constraint_dimension(&self) -> usize {
        self.constraint_dimension
    }

    pub fn support_dimension(&self) -> usize {
        self.ambient_dimension - self.constraint_dimension
    }
}

/// Setup and solve diagnostics for a certified supported law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportedMaternDiagnostics {
    pub ambient_dimension: usize,
    pub support_dimension: usize,
    pub constraint_dimension: usize,
    pub ambient_energy_nnz: usize,
    pub constraint_nnz: usize,
    pub constraint_gram_factor_nnz: usize,
    pub supported_energy_factor_nnz: usize,
    /// Actual sparse-Cholesky storage for the constraint-rank certificate.
    pub constraint_gram_factor: Option<SparseCholeskyDiagnostics>,
    /// Actual sparse-Cholesky storage for the supported-energy certificate.
    pub supported_energy_factor: SparseCholeskyDiagnostics,
    /// Stored entries of the assembled indefinite KKT coefficient matrix.
    /// This is not LU factor fill.
    pub kkt_nnz: usize,
    pub certificate_residuals: SupportedGaussianCertificateResiduals,
}

/// Geometry-independent builder for the exact supported FEEC Matérn law.
pub struct SupportedMaternPriorBuilder {
    operators: ExactFormOperators,
    parameters: MaternParameters,
}

impl SupportedMaternPriorBuilder {
    pub fn from_operators(operators: ExactFormOperators) -> Self {
        Self {
            operators,
            parameters: MaternParameters::default(),
        }
    }

    /// Assemble compatible lowest-order Whitney spaces on the homogeneous
    /// full-relative subcomplex before constructing sparse graph constraints.
    pub fn from_feec_homogeneous_full_relative(
        topology: &Complex,
        metric: &MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        Ok(Self::from_operators(
            ExactFormOperators::from_feec_homogeneous_full_relative(topology, metric, degree)?,
        ))
    }

    pub fn parameters(mut self, parameters: MaternParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Build and certify the exact sparse supported representation.
    pub fn build(self) -> Result<SupportedMaternPrior> {
        let blocks = assemble_exact_matern_blocks(&self.operators, self.parameters.kappa)?;
        let (energy, constraints, layout) =
            assemble_supported_matern_graph(&blocks, self.parameters)?;
        let system = SupportedGaussianSystem::new(
            feg_infer::sparse::core_triplet_to_gmrf(&energy),
            feg_infer::sparse::core_triplet_to_gmrf(&constraints),
        )?;
        if system.support_dim() != self.operators.state().dimension() {
            return Err(FeecGmrfError::Assembly(format!(
                "supported Matérn graph has support dimension {}, expected state dimension {}",
                system.support_dim(),
                self.operators.state().dimension()
            )));
        }
        let metadata = exact_supported_metadata(&self.operators, self.parameters);
        Ok(SupportedMaternPrior {
            system,
            layout,
            state_layout: self.operators.state().boundary_layout().clone(),
            form_degree: self.operators.degree(),
            metadata,
        })
    }
}

/// Certified exact Matérn probability law on a sparse constrained support.
#[derive(Clone, Debug)]
pub struct SupportedMaternPrior {
    system: SupportedGaussianSystem,
    layout: SupportedMaternLayout,
    state_layout: BoundaryLayout,
    form_degree: FormDegree,
    metadata: MaternPriorMetadata,
}

impl SupportedMaternPrior {
    pub fn state_dimension(&self) -> usize {
        self.layout.state.len()
    }

    pub fn cochain_dimension(&self) -> usize {
        self.state_layout.full_dimension()
    }

    pub fn form_degree(&self) -> FormDegree {
        self.form_degree
    }

    pub fn layout(&self) -> &SupportedMaternLayout {
        &self.layout
    }

    pub fn matern_metadata(&self) -> &MaternPriorMetadata {
        &self.metadata
    }

    /// Factor the prior support equations without adding observations.
    pub fn factor(&self) -> Result<SupportedMaternPosterior> {
        self.finish_factorization(
            self.system.factorize()?,
            Vector::zeros(self.layout.ambient_dimension),
            self.system.energy().nnz(),
            self.system.constraints().nnz(),
        )
    }

    /// Exact covariance matrix of a small active- or full-cochain prior output.
    ///
    /// This convenience method factors the certified support once and delegates
    /// to [`SupportedMaternPosterior::pushforward_covariance`]. Repeated queries
    /// should retain the factored posterior instead.
    pub fn pushforward_covariance(&self, map: &LinearMap) -> Result<Vec<Vec<f64>>> {
        self.factor()?.pushforward_covariance(map)
    }

    /// Estimate mapped variances and their weighted covariance trace.
    ///
    /// This is the support-aware counterpart of
    /// [`crate::prior::GaussianPrior::pushforward_weighted_variance_estimate`].
    /// It uses the constrained covariance action directly and never
    /// materializes an ambient covariance matrix.
    pub fn pushforward_weighted_variance_estimate(
        &self,
        map: &LinearMap,
        output_weights: &[f64],
        method: VarianceMethod,
    ) -> Result<WeightedVarianceEstimate> {
        self.factor()?
            .pushforward_weighted_variance_estimate(map, output_weights, method)
    }

    /// Estimate `trace(W Cov(x))` for a sparse active-state weight matrix.
    pub fn weighted_covariance_trace(
        &self,
        weight: &SparseMat,
        method: VarianceMethod,
    ) -> Result<WeightedCovarianceTraceEstimate> {
        self.factor()?.weighted_covariance_trace(weight, method)
    }

    /// Audited exact covariance of a small active- or full-cochain prior output.
    pub fn pushforward_covariance_audited(
        &self,
        map: &LinearMap,
    ) -> Result<AuditedSupportedCovariance> {
        self.factor()?.pushforward_covariance_audited(map)
    }

    /// Condition state-coordinate observations and append exact state constraints.
    pub fn condition(
        &self,
        observations: &[LinearObservation],
        constraints: &[LinearConstraint],
    ) -> Result<SupportedMaternPosterior> {
        self.prepare(observations, constraints)?.condition()
    }

    /// Prepare a fixed observation/constraint design for repeated data vectors.
    ///
    /// Observation operators, noise, support constraints, and the indefinite
    /// KKT factor are reused. Replacement data change only the information
    /// vector and selected mean solve.
    pub fn prepare(
        &self,
        observations: &[LinearObservation],
        constraints: &[LinearConstraint],
    ) -> Result<PreparedSupportedMaternModel> {
        let ambient_dimension = self.layout.ambient_dimension;
        let mut energy = self.system.energy().clone();
        let mut prepared_observations = Vec::with_capacity(observations.len());
        for observation in observations {
            let eliminated = resolve_state_map(
                &self.state_layout,
                self.state_dimension(),
                &observation.operator,
                &observation.bias,
            )?;
            let ambient_map = embed_state_map(eliminated.reduced(), ambient_dimension)?;
            let matrix = feg_infer::sparse::core_triplet_to_gmrf(&ambient_map);
            let values = Vector::from_vec(observation.values.clone());
            let bias = Vector::from_vec(eliminated.fixed_offset().to_vec());
            let (update, _, noise) = match &observation.noise {
                GaussianNoise::ScalarVariance(variance) => {
                    let (update, information) =
                        linear_observation_update(&LinearObservationTerm::scalar_variance(
                            &matrix,
                            &values,
                            Some(&bias),
                            *variance,
                        ));
                    (
                        update,
                        information,
                        PreparedSupportedNoise::ScalarVariance(*variance),
                    )
                }
                GaussianNoise::Precision(precision) => {
                    let precision = feg_infer::sparse::core_triplet_to_gmrf(precision);
                    let (update, information) =
                        linear_observation_update(&LinearObservationTerm::precision(
                            &matrix,
                            &values,
                            Some(&bias),
                            &precision,
                        ));
                    (
                        update,
                        information,
                        PreparedSupportedNoise::Precision(precision),
                    )
                }
            };
            energy = gmrf_core::add_sparse(&energy, &update);
            prepared_observations.push(PreparedSupportedObservation {
                matrix,
                bias,
                default_values: observation.values.clone(),
                noise,
            });
        }

        let (support_constraints, affine_rhs) = append_state_constraints(
            self.system.constraints(),
            self.system.affine_rhs(),
            &self.state_layout,
            self.state_dimension(),
            ambient_dimension,
            constraints,
        )?;
        let posterior_system =
            SupportedGaussianSystem::new_affine(energy, support_constraints, affine_rhs)?;
        let energy_nnz = posterior_system.energy().nnz();
        let constraint_nnz = posterior_system.constraints().nnz();
        let factor = posterior_system.factorize()?;
        let selection = SelectedCoordinates::prefix(ambient_dimension, self.state_dimension())?;
        let diagnostics = supported_matern_diagnostics(&factor, energy_nnz, constraint_nnz);
        Ok(PreparedSupportedMaternModel {
            factor,
            selection,
            observations: prepared_observations,
            state_layout: self.state_layout.clone(),
            form_degree: self.form_degree,
            metadata: self.metadata.clone(),
            diagnostics,
        })
    }

    fn finish_factorization(
        &self,
        factor: SupportedGaussianFactor,
        information: Vector,
        energy_nnz: usize,
        constraint_nnz: usize,
    ) -> Result<SupportedMaternPosterior> {
        let ambient_dimension = factor.ambient_dim();
        let selection = SelectedCoordinates::prefix(ambient_dimension, self.state_dimension())?;
        let mean = factor.selected_mean(&selection, &information)?;
        let diagnostics = supported_matern_diagnostics(&factor, energy_nnz, constraint_nnz);
        Ok(SupportedMaternPosterior {
            factor,
            selection,
            mean: mean.iter().copied().collect(),
            state_layout: self.state_layout.clone(),
            form_degree: self.form_degree,
            metadata: self.metadata.clone(),
            diagnostics,
        })
    }
}

#[derive(Clone, Debug)]
enum PreparedSupportedNoise {
    ScalarVariance(f64),
    Precision(SparseMatrix),
}

#[derive(Clone, Debug)]
struct PreparedSupportedObservation {
    matrix: SparseMatrix,
    bias: Vector,
    default_values: Vec<f64>,
    noise: PreparedSupportedNoise,
}

/// Fixed exact-supported design with one reusable KKT factorization.
#[derive(Clone, Debug)]
pub struct PreparedSupportedMaternModel {
    factor: SupportedGaussianFactor,
    selection: SelectedCoordinates,
    observations: Vec<PreparedSupportedObservation>,
    state_layout: BoundaryLayout,
    form_degree: FormDegree,
    metadata: MaternPriorMetadata,
    diagnostics: SupportedMaternDiagnostics,
}

impl PreparedSupportedMaternModel {
    /// Observation row counts expected by replacement data vectors.
    pub fn observation_row_counts(&self) -> Vec<usize> {
        self.observations
            .iter()
            .map(|observation| observation.matrix.nrows())
            .collect()
    }

    /// Setup diagnostics for the cached constrained factorization.
    pub fn diagnostics(&self) -> SupportedMaternDiagnostics {
        self.diagnostics
    }

    /// Condition using the data supplied when the design was prepared.
    pub fn condition(&self) -> Result<SupportedMaternPosterior> {
        let values = self
            .observations
            .iter()
            .map(|observation| observation.default_values.clone())
            .collect::<Vec<_>>();
        self.condition_with_observation_values(&values)
    }

    /// Reuse the fixed KKT factor and return only the selected state mean.
    pub fn latent_mean_with_observation_values(&self, values: &[Vec<f64>]) -> Result<Vec<f64>> {
        let information = self.information_with_observation_values(values)?;
        let mean = self.factor.selected_mean(&self.selection, &information)?;
        Ok(mean.iter().copied().collect())
    }

    /// Reuse the fixed KKT factor for replacement observation values.
    pub fn condition_with_observation_values(
        &self,
        values: &[Vec<f64>],
    ) -> Result<SupportedMaternPosterior> {
        let mean = self.latent_mean_with_observation_values(values)?;
        Ok(SupportedMaternPosterior {
            factor: self.factor.clone(),
            selection: self.selection.clone(),
            mean,
            state_layout: self.state_layout.clone(),
            form_degree: self.form_degree,
            metadata: self.metadata.clone(),
            diagnostics: self.diagnostics,
        })
    }

    fn information_with_observation_values(&self, values: &[Vec<f64>]) -> Result<Vector> {
        if values.len() != self.observations.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "prepared supported model expects {} observation vectors, received {}",
                self.observations.len(),
                values.len()
            )));
        }
        let mut information = Vector::zeros(self.factor.ambient_dim());
        for (term_index, (observation, values)) in self.observations.iter().zip(values).enumerate()
        {
            if values.len() != observation.matrix.nrows() {
                return Err(FeecGmrfError::Dimension(format!(
                    "prepared supported observation term {term_index} expects {} rows, received {}",
                    observation.matrix.nrows(),
                    values.len()
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "prepared supported observation term {term_index} contains a non-finite value"
                )));
            }
            let centered = Vector::from_iterator(
                values.len(),
                values
                    .iter()
                    .zip(observation.bias.iter())
                    .map(|(value, bias)| value - bias),
            );
            information += match &observation.noise {
                PreparedSupportedNoise::ScalarVariance(variance) => {
                    ht_weighted_observations(&observation.matrix, &centered, 1.0 / variance)
                }
                PreparedSupportedNoise::Precision(precision) => {
                    ht_precision_weighted_observations(&observation.matrix, &centered, precision)
                }
            };
        }
        Ok(information)
    }
}

fn supported_matern_diagnostics(
    factor: &SupportedGaussianFactor,
    energy_nnz: usize,
    constraint_nnz: usize,
) -> SupportedMaternDiagnostics {
    let certificate = factor.certificate();
    SupportedMaternDiagnostics {
        ambient_dimension: factor.ambient_dim(),
        support_dimension: factor.support_dim(),
        constraint_dimension: factor.constraint_dim(),
        ambient_energy_nnz: energy_nnz,
        constraint_nnz,
        constraint_gram_factor_nnz: certificate.constraint_gram_factor_nnz(),
        supported_energy_factor_nnz: certificate.supported_energy_factor_nnz(),
        constraint_gram_factor: certificate.constraint_gram_factor_diagnostics(),
        supported_energy_factor: certificate.supported_energy_factor_diagnostics(),
        kkt_nnz: factor.kkt_nnz(),
        certificate_residuals: certificate.residuals(),
    }
}

/// Factored prior or linearly conditioned posterior of an exact supported law.
#[derive(Debug)]
pub struct SupportedMaternPosterior {
    factor: SupportedGaussianFactor,
    selection: SelectedCoordinates,
    mean: Vec<f64>,
    state_layout: BoundaryLayout,
    form_degree: FormDegree,
    metadata: MaternPriorMetadata,
    diagnostics: SupportedMaternDiagnostics,
}

impl SupportedMaternPosterior {
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn cochain_mean(&self) -> Result<Vec<f64>> {
        let elimination = EssentialBoundaryElimination::from_layout(self.state_layout.clone())?;
        elimination.lift_state(&self.mean)
    }

    pub fn form_degree(&self) -> FormDegree {
        self.form_degree
    }

    pub fn matern_metadata(&self) -> &MaternPriorMetadata {
        &self.metadata
    }

    pub fn diagnostics(&self) -> SupportedMaternDiagnostics {
        self.diagnostics
    }

    /// Actual LU factor nonzeros for the indefinite KKT solve, when supported
    /// by the configured backend.
    ///
    /// The current faer backend returns a typed unsupported-query error. This
    /// prevents the assembled KKT input count from being mislabeled as factor
    /// fill.
    pub fn kkt_factor_nonzeros(&self) -> Result<usize> {
        Ok(self.factor.factor_nnz()?)
    }

    pub fn covariance_action(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        if rhs.len() != self.mean.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "selected covariance right-hand side length {} does not match state dimension {}",
                rhs.len(),
                self.mean.len()
            )));
        }
        let action = self
            .factor
            .selected_covariance_action(&self.selection, &Vector::from_vec(rhs.to_vec()))?;
        Ok(action.iter().copied().collect())
    }

    /// Exact variances of an active- or full-cochain linear map.
    pub fn pushforward_variances(&self, map: &LinearMap) -> Result<Vec<f64>> {
        let eliminated = resolve_state_map(
            &self.state_layout,
            self.mean.len(),
            map,
            &vec![0.0; map.output_dimension()],
        )?;
        let ambient_map = embed_state_map(eliminated.reduced(), self.factor.ambient_dim())?;
        let operator = SparseRowOperator::from_sparse_matrix(
            &feg_infer::sparse::core_triplet_to_gmrf(&ambient_map),
        )?;
        let variances = self.factor.exact_transformed_variances(&operator)?;
        Ok(variances.iter().copied().collect())
    }

    /// Estimate mapped variances and their weighted covariance trace through
    /// the certified supported covariance action.
    pub fn pushforward_weighted_variance_estimate(
        &self,
        map: &LinearMap,
        output_weights: &[f64],
        method: VarianceMethod,
    ) -> Result<WeightedVarianceEstimate> {
        if output_weights.len() != map.output_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "physical RMS weight count {} does not match output dimension {}",
                output_weights.len(),
                map.output_dimension()
            )));
        }
        if output_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || output_weights.iter().sum::<f64>() <= 0.0
        {
            return Err(FeecGmrfError::InvalidParameter(
                "physical RMS weights must be finite, non-negative, and have positive sum"
                    .to_string(),
            ));
        }
        let eliminated = resolve_state_map(
            &self.state_layout,
            self.mean.len(),
            map,
            &vec![0.0; map.output_dimension()],
        )?;
        let ambient_map = embed_state_map(eliminated.reduced(), self.factor.ambient_dim())?;
        let operator = SparseRowOperator::from_sparse_matrix(
            &feg_infer::sparse::core_triplet_to_gmrf(&ambient_map),
        )?;
        let weights = Vector::from_vec(output_weights.to_vec());
        let mode = method.transformed_mode().ok_or_else(|| {
            FeecGmrfError::Unsupported(
                "supported weighted physical traces require exact or Hutchinson estimation"
                    .to_string(),
            )
        })?;
        let estimate = match mode {
            TransformedVarianceMode::Exact => {
                return exact_supported_weighted_variance(&self.factor, &operator, output_weights)
            }
            TransformedVarianceMode::Auto { exact_max_dofs, .. }
                if self.mean.len() <= exact_max_dofs =>
            {
                return exact_supported_weighted_variance(&self.factor, &operator, output_weights);
            }
            TransformedVarianceMode::Hutchinson {
                config,
                floor,
                distribution,
            }
            | TransformedVarianceMode::Auto {
                config,
                floor,
                distribution,
                ..
            } => gmrf_core::estimate_hutchinson_transformed_variance_weighted_trace_with_solve(
                &operator,
                &weights,
                config,
                floor,
                distribution,
                |rhs| self.factor.covariance_action(rhs),
            )?,
        };
        Ok(WeightedVarianceEstimate {
            variances: public_estimate(estimate.variances),
            weighted_trace: estimate.weighted_trace.value,
            weighted_trace_standard_error: estimate.weighted_trace.batch_standard_error,
            weighted_trace_relative_standard_error: estimate.weighted_trace.relative_standard_error,
        })
    }

    /// Estimate `trace(W Cov(x))` on the selected FEEC state without
    /// materializing the supported ambient covariance.
    pub fn weighted_covariance_trace(
        &self,
        weight: &SparseMat,
        method: VarianceMethod,
    ) -> Result<WeightedCovarianceTraceEstimate> {
        if weight.nrows() != self.mean.len() || weight.ncols() != self.mean.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "supported weighted trace shape {}x{} does not match state dimension {}",
                weight.nrows(),
                weight.ncols(),
                self.mean.len()
            )));
        }
        let weight = feg_infer::sparse::core_triplet_to_gmrf(weight);
        let estimate = match method {
            VarianceMethod::Exact => {
                gmrf_core::exact_weighted_covariance_trace_with_solve(&weight, |rhs| {
                    self.factor.selected_covariance_action(&self.selection, rhs)
                })?
            }
            VarianceMethod::Hutchinson(config) => {
                gmrf_core::estimate_hutchinson_weighted_covariance_trace_with_solve(
                    &weight,
                    config.probe_config(),
                    config.distribution,
                    |rhs| self.factor.selected_covariance_action(&self.selection, rhs),
                )?
            }
            VarianceMethod::Auto {
                exact_max_dofs,
                hutchinson,
            } => {
                if self.mean.len() <= exact_max_dofs {
                    gmrf_core::exact_weighted_covariance_trace_with_solve(&weight, |rhs| {
                        self.factor.selected_covariance_action(&self.selection, rhs)
                    })?
                } else {
                    gmrf_core::estimate_hutchinson_weighted_covariance_trace_with_solve(
                        &weight,
                        hutchinson.probe_config(),
                        hutchinson.distribution,
                        |rhs| self.factor.selected_covariance_action(&self.selection, rhs),
                    )?
                }
            }
            VarianceMethod::MonteCarlo(_) => {
                return Err(FeecGmrfError::Unsupported(
                    "supported sparse weighted traces require exact or Hutchinson estimation"
                        .to_string(),
                ))
            }
        };
        Ok(public_weighted_trace(estimate))
    }

    /// Exact covariance matrix of a small active- or full-cochain output map.
    ///
    /// Only the requested output covariance is materialized. Each column uses
    /// the certified selected covariance action on the supported state; no
    /// ambient covariance or dense inverse is formed.
    pub fn pushforward_covariance(&self, map: &LinearMap) -> Result<Vec<Vec<f64>>> {
        Ok(self.pushforward_covariance_audited(map)?.symmetric)
    }

    /// Materialize covariance actions without enforcing or repairing symmetry.
    ///
    /// This diagnostic API exists so numerical studies can retain the raw KKT
    /// output. Most callers should use [`Self::pushforward_covariance`] or
    /// [`Self::pushforward_covariance_audited`].
    pub fn pushforward_covariance_raw(&self, map: &LinearMap) -> Result<Vec<Vec<f64>>> {
        let eliminated = resolve_state_map(
            &self.state_layout,
            self.mean.len(),
            map,
            &vec![0.0; map.output_dimension()],
        )?;
        let reduced = eliminated.reduced();
        let output_dimension = reduced.output_dimension();
        let mut covariance = vec![vec![0.0; output_dimension]; output_dimension];
        for column in 0..output_dimension {
            let mut output_basis = vec![0.0; output_dimension];
            output_basis[column] = 1.0;
            let state_rhs = reduced.apply_transpose(&output_basis)?;
            let state_action = self.covariance_action(&state_rhs)?;
            let output_action = reduced.apply(&state_action)?;
            for (row, value) in output_action.into_iter().enumerate() {
                if !value.is_finite() {
                    return Err(FeecGmrfError::Inference(format!(
                        "supported transformed covariance entry ({row}, {column}) is non-finite"
                    )));
                }
                covariance[row][column] = value;
            }
        }
        Ok(covariance)
    }

    /// Materialize, audit, and symmetrically project a small output covariance.
    ///
    /// The current named interim audit uses a frozen scale-relative skew limit
    /// and records that KKT residuals are not yet available. No covariance is
    /// projected unless the raw matrix passes that explicit audit.
    pub fn pushforward_covariance_audited(
        &self,
        map: &LinearMap,
    ) -> Result<AuditedSupportedCovariance> {
        audit_transformed_covariance(self.pushforward_covariance_raw(map)?)
    }

    /// Sampling is not available until a support-aware square root is implemented.
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>> {
        self.factor
            .sample(rng)
            .map(|sample| {
                self.selection
                    .indices()
                    .iter()
                    .map(|index| sample[*index])
                    .collect()
            })
            .map_err(Into::into)
    }
}

fn exact_supported_weighted_variance(
    factor: &SupportedGaussianFactor,
    operator: &SparseRowOperator,
    output_weights: &[f64],
) -> Result<WeightedVarianceEstimate> {
    let values = factor.exact_transformed_variances(operator)?;
    let variances = exact_estimate(values.iter().copied().collect());
    let weighted_trace = variances
        .values
        .iter()
        .zip(output_weights)
        .map(|(variance, weight)| variance * weight)
        .sum::<f64>();
    Ok(WeightedVarianceEstimate {
        variances,
        weighted_trace,
        weighted_trace_standard_error: None,
        weighted_trace_relative_standard_error: None,
    })
}

fn exact_supported_metadata(
    operators: &ExactFormOperators,
    parameters: MaternParameters,
) -> MaternPriorMetadata {
    exact_matern_metadata(
        operators,
        parameters,
        MaternRepresentation::ExactSupported,
        if operators.incoming().is_some() {
            InverseUse::Applied(MassInverseKind::ConstrainedSupport)
        } else {
            InverseUse::NotApplicable
        },
        if parameters.alpha == feg_core::MaternAlpha::One {
            InverseUse::NotUsed
        } else {
            InverseUse::Applied(MassInverseKind::ConstrainedSupport)
        },
    )
}

fn assemble_supported_matern_graph(
    blocks: &ExactMaternBlocks,
    parameters: MaternParameters,
) -> Result<(SparseMat, SparseMat, SupportedMaternLayout)> {
    let n = blocks.mass.nrows();
    let p = blocks.lower.as_ref().map_or(0, |lower| lower.mass.nrows());
    let tau2 = parameters.tau.powi(2);
    let (ambient, constraint_dimension, lower_ranges, recurrence_ranges) = match parameters.alpha {
        feg_core::MaternAlpha::One => (
            n + p,
            p,
            (p > 0).then_some(n..n + p).into_iter().collect(),
            Vec::new(),
        ),
        feg_core::MaternAlpha::Two => (
            2 * n + p,
            n + p,
            (p > 0).then_some(n..n + p).into_iter().collect(),
            std::iter::once(n + p..2 * n + p).collect(),
        ),
        feg_core::MaternAlpha::Three => (
            2 * n + 2 * p,
            n + 2 * p,
            if p > 0 {
                vec![n..n + p, 2 * n + p..2 * n + 2 * p]
            } else {
                Vec::new()
            },
            std::iter::once(n + p..2 * n + p).collect(),
        ),
    };
    let layout = SupportedMaternLayout {
        state: 0..n,
        lower_adjoint_auxiliaries: lower_ranges,
        recurrence_states: recurrence_ranges,
        ambient_dimension: ambient,
        constraint_dimension,
    };
    let mut energy = Vec::new();
    let mut constraints = Vec::new();
    let lower = blocks.lower.as_ref();

    match parameters.alpha {
        feg_core::MaternAlpha::One => {
            push_block(&mut energy, 0, 0, &blocks.local_system, tau2);
            if let Some(lower) = lower {
                push_block(&mut energy, n, n, &lower.mass, tau2);
                push_block(&mut constraints, 0, 0, &lower.coupling.transpose(), -1.0);
                push_block(&mut constraints, 0, n, &lower.mass, 1.0);
            }
        }
        feg_core::MaternAlpha::Two => {
            let y = n + p;
            push_block(&mut energy, y, y, &blocks.mass, tau2);
            if let Some(lower) = lower {
                push_block(&mut constraints, 0, 0, &lower.coupling.transpose(), -1.0);
                push_block(&mut constraints, 0, n, &lower.mass, 1.0);
            }
            push_block(&mut constraints, p, 0, &blocks.local_system, -1.0);
            if let Some(lower) = lower {
                push_block(&mut constraints, p, n, &lower.coupling, -1.0);
            }
            push_block(&mut constraints, p, y, &blocks.mass, 1.0);
        }
        feg_core::MaternAlpha::Three => {
            let y = n + p;
            let sigma_one = 2 * n + p;
            push_block(&mut energy, y, y, &blocks.local_system, tau2);
            if let Some(lower) = lower {
                push_block(&mut energy, sigma_one, sigma_one, &lower.mass, tau2);
                push_block(&mut constraints, 0, 0, &lower.coupling.transpose(), -1.0);
                push_block(&mut constraints, 0, n, &lower.mass, 1.0);
            }
            push_block(&mut constraints, p, 0, &blocks.local_system, -1.0);
            if let Some(lower) = lower {
                push_block(&mut constraints, p, n, &lower.coupling, -1.0);
            }
            push_block(&mut constraints, p, y, &blocks.mass, 1.0);
            if let Some(lower) = lower {
                push_block(
                    &mut constraints,
                    p + n,
                    y,
                    &lower.coupling.transpose(),
                    -1.0,
                );
                push_block(&mut constraints, p + n, sigma_one, &lower.mass, 1.0);
            }
        }
    }
    Ok((
        SparseMat::from_triplets(ambient, ambient, energy),
        SparseMat::from_triplets(constraint_dimension, ambient, constraints),
        layout,
    ))
}

fn push_block(
    output: &mut Vec<SparseTriplet>,
    row_offset: usize,
    column_offset: usize,
    block: &SparseMat,
    scale: f64,
) {
    output.extend(
        block
            .triplet_iter()
            .filter(|(_, _, value)| *value != 0.0)
            .map(|(row, column, value)| SparseTriplet {
                row: row_offset + row,
                col: column_offset + column,
                value: scale * value,
            }),
    );
}

fn resolve_state_map(
    layout: &BoundaryLayout,
    state_dimension: usize,
    map: &LinearMap,
    bias: &[f64],
) -> Result<EliminatedLinearMap> {
    if map.input_dimension() == state_dimension {
        if bias.len() != map.output_dimension() {
            return Err(FeecGmrfError::Dimension(
                "map bias length must match map output dimension".to_string(),
            ));
        }
        return Ok(EliminatedLinearMap::from_active(map.clone(), bias.to_vec()));
    }
    if map.input_dimension() != layout.full_dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "map input dimension {} matches neither active dimension {state_dimension} nor full cochain dimension {}",
            map.input_dimension(),
            layout.full_dimension()
        )));
    }
    EssentialBoundaryElimination::from_layout(layout.clone())?.eliminate_map(map, bias)
}

fn embed_state_map(map: &LinearMap, ambient_dimension: usize) -> Result<SparseMat> {
    if map.input_dimension() > ambient_dimension {
        return Err(FeecGmrfError::Dimension(
            "state map cannot be embedded into a smaller ambient space".to_string(),
        ));
    }
    Ok(SparseMat::from_triplets(
        map.output_dimension(),
        ambient_dimension,
        map.matrix()
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet { row, col, value }),
    ))
}

fn append_state_constraints(
    base: &gmrf_core::SparseMatrix,
    base_rhs: &Vector,
    layout: &BoundaryLayout,
    state_dimension: usize,
    ambient_dimension: usize,
    additions: &[LinearConstraint],
) -> Result<(gmrf_core::SparseMatrix, Vector)> {
    let added_rows = additions
        .iter()
        .map(|constraint| constraint.operator.output_dimension())
        .sum::<usize>();
    if added_rows == 0 {
        return Ok((base.clone(), base_rhs.clone()));
    }
    let mut matrix = SparseMat::new(base.nrows() + added_rows, ambient_dimension);
    for (row, col, value) in base.triplet_iter() {
        matrix.push(row, col, *value);
    }
    let mut rhs = base_rhs.iter().copied().collect::<Vec<_>>();
    let mut row_offset = base.nrows();
    for constraint in additions {
        let zero_bias = vec![0.0; constraint.operator.output_dimension()];
        let eliminated =
            resolve_state_map(layout, state_dimension, &constraint.operator, &zero_bias)?;
        let embedded = embed_state_map(eliminated.reduced(), ambient_dimension)?;
        for (row, col, value) in embedded.triplet_iter() {
            matrix.push(row_offset + row, col, value);
        }
        rhs.extend(
            constraint
                .target
                .iter()
                .zip(eliminated.fixed_offset())
                .map(|(target, offset)| target - offset),
        );
        row_offset += constraint.operator.output_dimension();
    }
    Ok((
        feg_infer::sparse::core_triplet_to_gmrf(&matrix),
        Vector::from_vec(rhs),
    ))
}

fn audit_transformed_covariance(covariance: Vec<Vec<f64>>) -> Result<AuditedSupportedCovariance> {
    let dimension = covariance.len();
    let scale = covariance
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let mut maximum = (0_usize, 0_usize, 0.0_f64);
    for (row, values) in covariance.iter().enumerate() {
        if values.len() != dimension {
            return Err(FeecGmrfError::Dimension(
                "supported transformed covariance is not square".to_string(),
            ));
        }
        if let Some(column) = values.iter().position(|value| !value.is_finite()) {
            return Err(FeecGmrfError::NonFiniteMatrixEntry {
                matrix: "supported transformed covariance",
                row,
                column,
            });
        }
        for (column, transpose_values) in covariance.iter().enumerate().skip(row + 1) {
            let difference = (values[column] - transpose_values[row]).abs();
            if difference > maximum.2 {
                maximum = (row, column, difference);
            }
        }
    }
    let relative_skew = if scale == 0.0 { 0.0 } else { maximum.2 / scale };
    let tolerance = INTERIM_SUPPORTED_COVARIANCE_RELATIVE_SKEW_LIMIT * scale;
    if maximum.2 > tolerance {
        return Err(FeecGmrfError::MatrixNotSymmetric {
            matrix: "supported transformed covariance",
            row: maximum.0,
            column: maximum.1,
            max_abs_difference: maximum.2,
            tolerance,
        });
    }
    let mut symmetric = covariance.clone();
    let mut correction_squared = 0.0;
    let mut maximum_correction = 0.0_f64;
    for row in 0..dimension {
        for column in (row + 1)..dimension {
            let projected = 0.5 * covariance[row][column] + 0.5 * covariance[column][row];
            for (target_row, target_column) in [(row, column), (column, row)] {
                let correction = projected - covariance[target_row][target_column];
                maximum_correction = maximum_correction.max(correction.abs());
                correction_squared += correction * correction;
                symmetric[target_row][target_column] = projected;
            }
        }
    }
    Ok(AuditedSupportedCovariance {
        raw: covariance,
        symmetric,
        audit: SupportedCovarianceSymmetryAudit {
            policy_version: INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION,
            dimension,
            raw_scale: scale,
            raw_maximum_absolute_skew: maximum.2,
            raw_maximum_relative_skew: relative_skew,
            raw_maximum_skew_row: maximum.0,
            raw_maximum_skew_column: maximum.1,
            relative_skew_limit: INTERIM_SUPPORTED_COVARIANCE_RELATIVE_SKEW_LIMIT,
            projection_maximum_absolute_correction: maximum_correction,
            projection_frobenius_correction: correction_squared.sqrt(),
            residual_aware: false,
            residual_status:
                "unavailable: gmrf-core KKT covariance-action residuals are not exposed",
            status: "accepted_interim_scale_relative_and_projected",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformed_covariance_audit_records_and_projects_accepted_skew() {
        let accepted = audit_transformed_covariance(vec![
            vec![2.0e-3, 3.0e-4],
            vec![3.0e-4 + 1.0e-14, 4.0e-3],
        ])
        .unwrap();
        assert_eq!(accepted.raw[0][1], 3.0e-4);
        assert_eq!(accepted.symmetric[0][1], accepted.symmetric[1][0]);
        assert!(accepted.audit.raw_maximum_absolute_skew > 0.0);
        assert!(accepted.audit.raw_maximum_relative_skew < 1.0e-10);
        assert!(!accepted.audit.residual_aware);
        assert_eq!(
            accepted.audit.policy_version,
            INTERIM_SUPPORTED_COVARIANCE_AUDIT_POLICY_VERSION
        );
        assert!(
            (accepted.audit.projection_maximum_absolute_correction
                - 0.5 * accepted.audit.raw_maximum_absolute_skew)
                .abs()
                <= f64::EPSILON * accepted.audit.raw_scale
        );
    }

    #[test]
    fn transformed_covariance_audit_is_scale_invariant_and_rejects_material_skew() {
        for scale in [1.0e-20, 1.0, 1.0e20] {
            let accepted = audit_transformed_covariance(vec![
                vec![2.0 * scale, 0.3 * scale],
                vec![(0.3 + 2.0e-11) * scale, 4.0 * scale],
            ])
            .unwrap();
            assert!(accepted.audit.raw_maximum_relative_skew < 1.1e-11);
        }
        let error =
            audit_transformed_covariance(vec![vec![2.0e-20, 3.0e-21], vec![3.1e-21, 4.0e-20]])
                .unwrap_err();
        assert!(matches!(
            error,
            FeecGmrfError::MatrixNotSymmetric {
                matrix: "supported transformed covariance",
                ..
            }
        ));
    }
}
