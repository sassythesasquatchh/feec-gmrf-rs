//! Exact observation-space Gaussian conditioning.
//!
//! This module is the backend-independent parent facade for an exact Gaussian
//! process calculation. A caller supplies the finite covariance blocks
//! `Kyy = H C H^T`, `Kgy = G C H^T`, and `Kgg = G C G^T`; `gmrf-core` owns the
//! dense conditioning algebra and factorization. The facade deliberately does
//! not construct a latent dense covariance.

use crate::operator::LinearMap;
use crate::prior::supported::{
    SupportedCovarianceSymmetryAudit, SupportedMaternPosterior, SupportedMaternPrior,
};
use crate::{FeecGmrfError, Result};
use gmrf_core::types::{DenseMatrix, Vector};
use gmrf_core::{
    ExactObservationDiagonalPosteriorDiagnostics as CoreDiagonalDiagnostics,
    PreparedExactObservationDiagonalPosterior, PreparedExactObservationPosterior,
};

/// Provenance for covariance blocks accepted by the exact conditioner.
///
/// Caller-supplied blocks must already be exactly symmetric: the parent API
/// never silently projects them. Supported-Matérn blocks are obtained from one
/// stacked physical pushforward and retain the named raw-skew audit which
/// authorized that explicit symmetric projection.
#[derive(Debug, Clone, PartialEq)]
pub enum ExactObservationCovarianceProvenance {
    /// The caller supplied finite `Kyy` and `Kgg` blocks that were already
    /// exactly symmetric, so no projection was performed.
    CallerSuppliedExactlySymmetric,
    /// A stacked exact-supported covariance action passed the named interim
    /// audit before projection. The audit retains the raw skew, scale,
    /// correction, policy version, and residual-awareness status.
    SupportedMaternStacked {
        audit: SupportedCovarianceSymmetryAudit,
    },
}

/// Validated prior covariance blocks for exact observation-space conditioning.
///
/// Rows use ordinary parent-owned `Vec<Vec<f64>>` storage, avoiding exposure of
/// a dense backend type. `Kgy` is output-by-observation. Construction is
/// necessarily quadratic in the number of requested observations and outputs;
/// it does not materialize covariance outside those requested maps.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactObservationCovarianceBlocks {
    prior_observation_covariance: Vec<Vec<f64>>,
    output_observation_covariance: Vec<Vec<f64>>,
    prior_output_covariance: Vec<Vec<f64>>,
    provenance: ExactObservationCovarianceProvenance,
}

/// Covariance data required for exact observation-space means and all output
/// marginal variances without constructing a dense output covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactObservationDiagonalBlocks {
    prior_observation_covariance: Vec<Vec<f64>>,
    output_observation_covariance: Vec<Vec<f64>>,
    prior_output_variances: Vec<f64>,
    provenance: ExactObservationCovarianceProvenance,
}

impl ExactObservationDiagonalBlocks {
    /// Build `Kyy`, `Kgy`, and `diag(Kgg)` from exact supported covariance
    /// actions. The caller explicitly caps both the dense observation factor
    /// dimension and the stored output-by-observation cross block.
    pub fn from_supported_matern_factor(
        factor: &SupportedMaternPosterior,
        observation_map: &LinearMap,
        output_map: &LinearMap,
        maximum_observation_count: usize,
        maximum_cross_covariance_entries: usize,
    ) -> Result<Self> {
        if observation_map.input_dimension() != output_map.input_dimension() {
            return Err(FeecGmrfError::Dimension(
                "exact diagonal observation and output maps must share a domain".to_string(),
            ));
        }
        let observation_count = observation_map.output_dimension();
        let output_count = output_map.output_dimension();
        if observation_count == 0 || output_count == 0 {
            return Err(FeecGmrfError::Dimension(
                "exact diagonal conditioning requires observations and outputs".to_string(),
            ));
        }
        if observation_count > maximum_observation_count {
            return Err(FeecGmrfError::DenseDiagnosticSizeExceeded {
                dimension: observation_count,
                maximum: maximum_observation_count,
            });
        }
        let cross_entries = observation_count.checked_mul(output_count).ok_or_else(|| {
            FeecGmrfError::Dimension(
                "exact diagonal output-observation cross-block size overflowed usize".to_string(),
            )
        })?;
        if cross_entries > maximum_cross_covariance_entries {
            return Err(FeecGmrfError::DenseDiagnosticSizeExceeded {
                dimension: cross_entries,
                maximum: maximum_cross_covariance_entries,
            });
        }
        let audited = factor.pushforward_covariance_audited(observation_map)?;
        let prior_output_variances = factor.pushforward_variances(output_map)?;
        let mut output_observation_covariance = vec![vec![0.0; observation_count]; output_count];
        for observation in 0..observation_count {
            let mut selector = vec![0.0; observation_count];
            selector[observation] = 1.0;
            let rhs = observation_map.apply_transpose(&selector)?;
            let covariance_action = factor.covariance_action(&rhs)?;
            let output_action = output_map.apply(&covariance_action)?;
            for (output, value) in output_action.into_iter().enumerate() {
                output_observation_covariance[output][observation] = value;
            }
        }
        Ok(Self {
            prior_observation_covariance: audited.symmetric,
            output_observation_covariance,
            prior_output_variances,
            provenance: ExactObservationCovarianceProvenance::SupportedMaternStacked {
                audit: audited.audit,
            },
        })
    }

    pub fn observation_count(&self) -> usize {
        self.prior_observation_covariance.len()
    }

    pub fn output_count(&self) -> usize {
        self.prior_output_variances.len()
    }
}

impl ExactObservationCovarianceBlocks {
    /// Validate caller-supplied `Kyy`, `Kgy`, and `Kgg` blocks.
    ///
    /// `Kyy` and `Kgg` must be exactly symmetric. A caller with independently
    /// computed covariance actions must perform and document an explicit audit
    /// rather than relying on an undocumented tolerance or projection here.
    pub fn new(
        prior_observation_covariance: Vec<Vec<f64>>,
        output_observation_covariance: Vec<Vec<f64>>,
        prior_output_covariance: Vec<Vec<f64>>,
    ) -> Result<Self> {
        validate_blocks(
            &prior_observation_covariance,
            &output_observation_covariance,
            &prior_output_covariance,
        )?;
        Ok(Self {
            prior_observation_covariance,
            output_observation_covariance,
            prior_output_covariance,
            provenance: ExactObservationCovarianceProvenance::CallerSuppliedExactlySymmetric,
        })
    }

    /// Construct covariance blocks from one audited stacked pushforward of a
    /// factored exact-supported Matérn law.
    ///
    /// `observation_map` and `output_map` must share either the active-state or
    /// the full-cochain domain. The explicit guard makes the unavoidable
    /// `(m + q)^2` diagnostic materialization a caller-visible choice.
    pub fn from_supported_matern_factor(
        factor: &SupportedMaternPosterior,
        observation_map: &LinearMap,
        output_map: &LinearMap,
        maximum_stacked_dimension: usize,
    ) -> Result<Self> {
        validate_stacked_guard(observation_map, output_map, maximum_stacked_dimension)?;
        let observation_count = observation_map.output_dimension();
        let output_count = output_map.output_dimension();
        let stacked = LinearMap::stack(&[observation_map.clone(), output_map.clone()])?;
        let audited = factor.pushforward_covariance_audited(&stacked)?;
        let covariance = audited.symmetric;

        let prior_observation_covariance = covariance[..observation_count]
            .iter()
            .map(|row| row[..observation_count].to_vec())
            .collect::<Vec<_>>();
        let output_observation_covariance = covariance
            [observation_count..observation_count + output_count]
            .iter()
            .map(|row| row[..observation_count].to_vec())
            .collect::<Vec<_>>();
        let prior_output_covariance = covariance
            [observation_count..observation_count + output_count]
            .iter()
            .map(|row| row[observation_count..observation_count + output_count].to_vec())
            .collect::<Vec<_>>();
        validate_blocks(
            &prior_observation_covariance,
            &output_observation_covariance,
            &prior_output_covariance,
        )?;
        Ok(Self {
            prior_observation_covariance,
            output_observation_covariance,
            prior_output_covariance,
            provenance: ExactObservationCovarianceProvenance::SupportedMaternStacked {
                audit: audited.audit,
            },
        })
    }

    pub fn observation_count(&self) -> usize {
        self.prior_observation_covariance.len()
    }

    pub fn output_count(&self) -> usize {
        self.prior_output_covariance.len()
    }

    pub fn prior_observation_covariance(&self) -> &[Vec<f64>] {
        &self.prior_observation_covariance
    }

    pub fn output_observation_covariance(&self) -> &[Vec<f64>] {
        &self.output_observation_covariance
    }

    pub fn prior_output_covariance(&self) -> &[Vec<f64>] {
        &self.prior_output_covariance
    }

    pub fn provenance(&self) -> &ExactObservationCovarianceProvenance {
        &self.provenance
    }
}

/// Backend-independent diagnostics for an exact observation-space posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactObservationPosteriorDiagnostics {
    pub observation_count: usize,
    pub output_count: usize,
    pub dense_factor_dimension: usize,
    pub dense_factor_allocated_entries: usize,
    pub dense_factor_triangular_entries: usize,
    pub output_observation_covariance_entries: usize,
    pub posterior_output_covariance_entries: usize,
    pub persistent_f64_entries: usize,
    /// Raw maximum numerical skew in the posterior output covariance. The
    /// backend does not project this result.
    pub posterior_output_covariance_max_abs_skew: f64,
}

/// Prepared exact observation-space Gaussian posterior.
///
/// The dense observation factor, `Kgy`, and posterior `Kgg` are owned by
/// `gmrf-core`. Replacement observations reuse that factor. No conditioning
/// formula is reimplemented in the integration package.
#[derive(Debug, Clone)]
pub struct ExactObservationPosterior {
    inner: PreparedExactObservationPosterior,
    provenance: ExactObservationCovarianceProvenance,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactObservationDiagonalPosteriorDiagnostics {
    pub observation_count: usize,
    pub output_count: usize,
    pub dense_factor_allocated_entries: usize,
    pub output_observation_covariance_entries: usize,
    pub prior_output_variance_entries: usize,
    pub posterior_output_variance_entries: usize,
    pub persistent_f64_entries: usize,
}

/// Parent facade for diagonal-only exact observation-space conditioning.
#[derive(Debug, Clone)]
pub struct ExactObservationDiagonalPosterior {
    inner: PreparedExactObservationDiagonalPosterior,
    provenance: ExactObservationCovarianceProvenance,
}

impl ExactObservationDiagonalPosterior {
    pub fn from_blocks_with_noise_covariance(
        blocks: ExactObservationDiagonalBlocks,
        noise_covariance: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let prior_observation_covariance = dense_from_rows(&blocks.prior_observation_covariance)?;
        let output_observation_covariance = dense_from_rows(&blocks.output_observation_covariance)?;
        let prior_output_variances = Vector::from_vec(blocks.prior_output_variances);
        let noise_covariance = dense_from_rows(&noise_covariance)?;
        let inner = PreparedExactObservationDiagonalPosterior::from_prior_covariance_blocks_with_noise_covariance(
            &prior_observation_covariance,
            &output_observation_covariance,
            &prior_output_variances,
            &noise_covariance,
        )?;
        Ok(Self {
            inner,
            provenance: blocks.provenance,
        })
    }

    pub fn output_mean(&self, observations: &[f64]) -> Result<Vec<f64>> {
        Ok(self
            .inner
            .output_mean(&Vector::from_vec(observations.to_vec()))?
            .iter()
            .copied()
            .collect())
    }

    pub fn prior_output_variances(&self) -> Vec<f64> {
        self.inner
            .prior_output_variances()
            .iter()
            .copied()
            .collect()
    }

    pub fn posterior_output_variances(&self) -> Vec<f64> {
        self.inner
            .posterior_output_variances()
            .iter()
            .copied()
            .collect()
    }

    pub fn posterior_output_covariance_action_from_prior_action(
        &self,
        input: &[f64],
        prior_action: &[f64],
    ) -> Result<Vec<f64>> {
        Ok(self
            .inner
            .posterior_output_covariance_action_from_prior_action(
                &Vector::from_vec(input.to_vec()),
                &Vector::from_vec(prior_action.to_vec()),
            )?
            .iter()
            .copied()
            .collect())
    }

    pub fn covariance_provenance(&self) -> &ExactObservationCovarianceProvenance {
        &self.provenance
    }

    pub fn diagnostics(&self) -> ExactObservationDiagonalPosteriorDiagnostics {
        let CoreDiagonalDiagnostics {
            observation_count,
            output_count,
            dense_factor_allocated_entries,
            output_observation_covariance_entries,
            prior_output_variance_entries,
            posterior_output_variance_entries,
            persistent_f64_entries,
        } = self.inner.diagnostics();
        ExactObservationDiagonalPosteriorDiagnostics {
            observation_count,
            output_count,
            dense_factor_allocated_entries,
            output_observation_covariance_entries,
            prior_output_variance_entries,
            posterior_output_variance_entries,
            persistent_f64_entries,
        }
    }
}

impl ExactObservationPosterior {
    /// Prepare an exact posterior from prior covariance blocks and scalar
    /// independent observation-noise variance.
    pub fn from_prior_covariance_blocks(
        blocks: ExactObservationCovarianceBlocks,
        noise_variance: f64,
    ) -> Result<Self> {
        let prior_observation_covariance = dense_from_rows(blocks.prior_observation_covariance())?;
        let output_observation_covariance =
            dense_from_rows(blocks.output_observation_covariance())?;
        let prior_output_covariance = dense_from_rows(blocks.prior_output_covariance())?;
        let inner = PreparedExactObservationPosterior::from_prior_covariance_blocks(
            &prior_observation_covariance,
            &output_observation_covariance,
            &prior_output_covariance,
            noise_variance,
        )?;
        Ok(Self {
            inner,
            provenance: blocks.provenance,
        })
    }

    /// Prepare an exact posterior with a caller-supplied dense noise covariance.
    ///
    /// The noise covariance uses ordinary row-major parent storage and must be
    /// finite, exactly symmetric, and dimensionally compatible with `Kyy`.
    pub fn from_prior_covariance_blocks_with_noise_covariance(
        blocks: ExactObservationCovarianceBlocks,
        noise_covariance: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let prior_observation_covariance = dense_from_rows(blocks.prior_observation_covariance())?;
        let output_observation_covariance =
            dense_from_rows(blocks.output_observation_covariance())?;
        let prior_output_covariance = dense_from_rows(blocks.prior_output_covariance())?;
        let noise_covariance = dense_from_rows(&noise_covariance)?;
        let inner =
            PreparedExactObservationPosterior::from_prior_covariance_blocks_with_noise_covariance(
                &prior_observation_covariance,
                &output_observation_covariance,
                &prior_output_covariance,
                &noise_covariance,
            )?;
        Ok(Self {
            inner,
            provenance: blocks.provenance,
        })
    }

    /// Convenience path from an exact-supported FEEC Matérn prior.
    ///
    /// This factors the supported prior once, constructs one audited stacked
    /// covariance pushforward, then delegates conditioning to `gmrf-core`.
    pub fn from_supported_matern_prior(
        prior: &SupportedMaternPrior,
        observation_map: &LinearMap,
        output_map: &LinearMap,
        noise_variance: f64,
        maximum_stacked_dimension: usize,
    ) -> Result<Self> {
        let factor = prior.factor()?;
        let blocks = ExactObservationCovarianceBlocks::from_supported_matern_factor(
            &factor,
            observation_map,
            output_map,
            maximum_stacked_dimension,
        )?;
        Self::from_prior_covariance_blocks(blocks, noise_variance)
    }

    pub fn observation_count(&self) -> usize {
        self.inner.observation_count()
    }

    pub fn output_count(&self) -> usize {
        self.inner.output_count()
    }

    /// Covariance-source and raw-skew audit provenance.
    pub fn covariance_provenance(&self) -> &ExactObservationCovarianceProvenance {
        &self.provenance
    }

    /// Posterior output mean for a zero-mean prior.
    pub fn output_mean(&self, observations: &[f64]) -> Result<Vec<f64>> {
        Ok(self
            .inner
            .output_mean(&Vector::from_vec(observations.to_vec()))?
            .iter()
            .copied()
            .collect())
    }

    /// Posterior output mean with explicit prior means.
    pub fn output_mean_with_prior(
        &self,
        observations: &[f64],
        prior_observation_mean: &[f64],
        prior_output_mean: &[f64],
    ) -> Result<Vec<f64>> {
        Ok(self
            .inner
            .output_mean_with_prior(
                &Vector::from_vec(observations.to_vec()),
                &Vector::from_vec(prior_observation_mean.to_vec()),
                &Vector::from_vec(prior_output_mean.to_vec()),
            )?
            .iter()
            .copied()
            .collect())
    }

    /// Materialize the raw posterior covariance of the requested outputs.
    /// No symmetry projection is applied.
    pub fn posterior_output_covariance(&self) -> Vec<Vec<f64>> {
        rows_from_dense(self.inner.posterior_output_covariance())
    }

    /// Apply the posterior covariance without exposing backend storage.
    pub fn posterior_output_covariance_action(&self, input: &[f64]) -> Result<Vec<f64>> {
        Ok(self
            .inner
            .posterior_output_covariance_action(&Vector::from_vec(input.to_vec()))?
            .iter()
            .copied()
            .collect())
    }

    pub fn diagnostics(&self) -> ExactObservationPosteriorDiagnostics {
        let diagnostics = self.inner.diagnostics();
        ExactObservationPosteriorDiagnostics {
            observation_count: diagnostics.observation_count,
            output_count: diagnostics.output_count,
            dense_factor_dimension: diagnostics.dense_factor_dimension,
            dense_factor_allocated_entries: diagnostics.dense_factor_allocated_entries,
            dense_factor_triangular_entries: diagnostics.dense_factor_triangular_entries,
            output_observation_covariance_entries: diagnostics
                .output_observation_covariance_entries,
            posterior_output_covariance_entries: diagnostics.posterior_output_covariance_entries,
            persistent_f64_entries: diagnostics.persistent_f64_entries,
            posterior_output_covariance_max_abs_skew: diagnostics
                .posterior_output_covariance_max_abs_skew,
        }
    }
}

fn validate_stacked_guard(
    observation_map: &LinearMap,
    output_map: &LinearMap,
    maximum_stacked_dimension: usize,
) -> Result<()> {
    if maximum_stacked_dimension == 0 {
        return Err(FeecGmrfError::InvalidParameter(
            "exact observation covariance guard must be positive".to_string(),
        ));
    }
    if observation_map.input_dimension() != output_map.input_dimension() {
        return Err(FeecGmrfError::Dimension(
            "exact observation and output maps must share a domain".to_string(),
        ));
    }
    let dimension = observation_map
        .output_dimension()
        .checked_add(output_map.output_dimension())
        .ok_or_else(|| {
            FeecGmrfError::Dimension(
                "exact observation stacked output dimension overflowed usize".to_string(),
            )
        })?;
    if dimension > maximum_stacked_dimension {
        return Err(FeecGmrfError::DenseDiagnosticSizeExceeded {
            dimension,
            maximum: maximum_stacked_dimension,
        });
    }
    Ok(())
}

fn validate_blocks(kyy: &[Vec<f64>], kgy: &[Vec<f64>], kgg: &[Vec<f64>]) -> Result<()> {
    validate_symmetric(kyy, "exact prior observation covariance")?;
    validate_symmetric(kgg, "exact prior output covariance")?;
    if kyy.is_empty() {
        return Err(FeecGmrfError::Dimension(
            "exact observation conditioner requires at least one observation".to_string(),
        ));
    }
    if kgg.is_empty() {
        return Err(FeecGmrfError::Dimension(
            "exact observation conditioner requires at least one output".to_string(),
        ));
    }
    if kgy.len() != kgg.len() || kgy.iter().any(|row| row.len() != kyy.len()) {
        return Err(FeecGmrfError::Dimension(
            "Kgy dimensions must be output count by observation count".to_string(),
        ));
    }
    validate_finite(kgy, "exact output-observation covariance")
}

fn validate_symmetric(matrix: &[Vec<f64>], label: &'static str) -> Result<()> {
    let dimension = matrix.len();
    if matrix.iter().any(|row| row.len() != dimension) {
        return Err(FeecGmrfError::Dimension(format!("{label} must be square")));
    }
    validate_finite(matrix, label)?;
    for (row, values) in matrix.iter().enumerate() {
        for (column, transpose_values) in matrix.iter().enumerate().take(row) {
            let difference = (values[column] - transpose_values[row]).abs();
            if difference != 0.0 {
                return Err(FeecGmrfError::MatrixNotSymmetric {
                    matrix: label,
                    row,
                    column,
                    max_abs_difference: difference,
                    tolerance: 0.0,
                });
            }
        }
    }
    Ok(())
}

fn validate_finite(matrix: &[Vec<f64>], label: &'static str) -> Result<()> {
    for (row, values) in matrix.iter().enumerate() {
        if let Some(column) = values.iter().position(|value| !value.is_finite()) {
            return Err(FeecGmrfError::NonFiniteMatrixEntry {
                matrix: label,
                row,
                column,
            });
        }
    }
    Ok(())
}

fn dense_from_rows(rows: &[Vec<f64>]) -> Result<DenseMatrix> {
    let columns = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != columns) {
        return Err(FeecGmrfError::Dimension(
            "dense covariance rows must be rectangular".to_string(),
        ));
    }
    Ok(DenseMatrix::from_fn(rows.len(), columns, |row, column| {
        rows[row][column]
    }))
}

fn rows_from_dense(matrix: &DenseMatrix) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}
