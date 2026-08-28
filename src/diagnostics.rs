//! Diagnostics for Gaussian predictive distributions and operator comparisons.

use std::collections::BTreeMap;

use common::linalg::nalgebra::Matrix;

use crate::operator::SparseMat;
use crate::{FeecGmrfError, Result};

pub use gmrf_core::GaussianPredictiveDiagnostics;

/// Dense finite-mesh certificate comparing a candidate precision with a
/// reference precision in the reference energy norm.
///
/// If `reference_precision` is \(Q\) and `candidate_precision` is
/// \(\widetilde Q\), the reported eigenvalues are those of
/// \(Q^{-1/2}(\widetilde Q-Q)Q^{-1/2}\). Consequently,
/// `relative_error` is the exact finite-dimensional quantity
///
/// \[
/// \delta_Q = \lVert Q^{-1/2}(\widetilde Q-Q)Q^{-1/2}\rVert_2.
/// \]
///
/// This diagnostic deliberately materializes dense matrices and is intended
/// only for reference-sized systems. It is not an asymptotic convergence
/// certificate and it does not infer a continuum error from a finite mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativePrecisionCertificate {
    /// Common matrix dimension.
    pub dimension: usize,
    /// Smallest eigenvalue of the relative precision perturbation.
    pub minimum_relative_eigenvalue: f64,
    /// Largest eigenvalue of the relative precision perturbation.
    pub maximum_relative_eigenvalue: f64,
    /// Maximum absolute relative eigenvalue, equal to `delta_Q` above.
    pub relative_error: f64,
    /// Smallest generalized eigenvalue of `(candidate, reference)`.
    pub minimum_precision_ratio: f64,
    /// Largest generalized eigenvalue of `(candidate, reference)`.
    pub maximum_precision_ratio: f64,
    /// Certified lower covariance multiplier relative to `reference^{-1}`.
    pub minimum_covariance_ratio: f64,
    /// Certified upper covariance multiplier relative to `reference^{-1}`.
    pub maximum_covariance_ratio: f64,
}

impl RelativePrecisionCertificate {
    /// Whether the symmetric relative-error envelope has `delta_Q < 1`.
    ///
    /// This is the convenient sufficient condition
    /// `(1-delta_Q) Q <= candidate <= (1+delta_Q) Q`. The sharper asymmetric
    /// precision and covariance ratios remain available in the other fields.
    pub fn has_unit_relative_envelope(self) -> bool {
        self.relative_error < 1.0
    }
}

/// Compute an exact dense relative-precision certificate on one finite mesh.
///
/// Duplicate sparse coordinates are merged. Both inputs must be finite,
/// symmetric positive-definite precision matrices of the same shape. The
/// explicit `maximum_dimension` guard prevents accidental dense
/// materialization of production-sized systems.
pub fn relative_precision_certificate(
    candidate_precision: &SparseMat,
    reference_precision: &SparseMat,
    maximum_dimension: usize,
) -> Result<RelativePrecisionCertificate> {
    if candidate_precision.nrows() != candidate_precision.ncols()
        || reference_precision.nrows() != reference_precision.ncols()
        || candidate_precision.nrows() != reference_precision.nrows()
    {
        return Err(FeecGmrfError::Dimension(format!(
            "relative precision certificate requires equal square shapes, got {}x{} and {}x{}",
            candidate_precision.nrows(),
            candidate_precision.ncols(),
            reference_precision.nrows(),
            reference_precision.ncols()
        )));
    }
    let dimension = reference_precision.nrows();
    if dimension > maximum_dimension {
        return Err(FeecGmrfError::DenseDiagnosticSizeExceeded {
            dimension,
            maximum: maximum_dimension,
        });
    }
    if dimension == 0 {
        return Err(FeecGmrfError::InvalidParameter(
            "relative precision certificate requires a nonempty precision".to_string(),
        ));
    }

    let candidate = crate::operator::validate_proper_precision(
        "candidate precision",
        candidate_precision.clone(),
    )?;
    let reference = crate::operator::validate_proper_precision(
        "reference precision",
        reference_precision.clone(),
    )?;
    let candidate_dense = dense_from_sparse(&candidate);
    let reference_dense = dense_from_sparse(&reference);
    let reference_cholesky = reference_dense.clone().cholesky().ok_or_else(|| {
        FeecGmrfError::PrecisionNotPositiveDefinite {
            matrix: "reference precision",
            reason: "dense certificate Cholesky factorization failed".to_string(),
        }
    })?;
    let inverse_lower = reference_cholesky.l().try_inverse().ok_or_else(|| {
        FeecGmrfError::PrecisionNotPositiveDefinite {
            matrix: "reference precision",
            reason: "dense certificate Cholesky factor is singular".to_string(),
        }
    })?;
    let difference = candidate_dense - reference_dense;
    let transformed = &inverse_lower * difference * inverse_lower.transpose();
    let transformed = (&transformed + transformed.transpose()) * 0.5;
    let eigenvalues = transformed.symmetric_eigen().eigenvalues;
    let minimum_relative_eigenvalue = eigenvalues.min();
    let maximum_relative_eigenvalue = eigenvalues.max();
    let relative_error = minimum_relative_eigenvalue
        .abs()
        .max(maximum_relative_eigenvalue.abs());
    let minimum_precision_ratio = 1.0 + minimum_relative_eigenvalue;
    let maximum_precision_ratio = 1.0 + maximum_relative_eigenvalue;
    if minimum_precision_ratio <= 0.0 || maximum_precision_ratio <= 0.0 {
        return Err(FeecGmrfError::PrecisionNotPositiveDefinite {
            matrix: "candidate precision relative to reference",
            reason: "computed generalized precision ratio is nonpositive".to_string(),
        });
    }

    Ok(RelativePrecisionCertificate {
        dimension,
        minimum_relative_eigenvalue,
        maximum_relative_eigenvalue,
        relative_error,
        minimum_precision_ratio,
        maximum_precision_ratio,
        minimum_covariance_ratio: 1.0 / maximum_precision_ratio,
        maximum_covariance_ratio: 1.0 / minimum_precision_ratio,
    })
}

fn dense_from_sparse(matrix: &SparseMat) -> Matrix {
    let mut dense = Matrix::zeros(matrix.nrows(), matrix.ncols());
    for (row, column, value) in matrix.triplet_iter() {
        dense[(row, column)] += value;
    }
    dense
}

/// Evaluate predictive residuals, RMSE, NLPD, and interval coverage.
pub fn gaussian_predictive_diagnostics(
    observations: &[f64],
    predictive_means: &[f64],
    latent_variances: &[f64],
    observation_variances: &[f64],
    interval_standard_deviations: f64,
) -> Result<GaussianPredictiveDiagnostics> {
    Ok(gmrf_core::gaussian_predictive_diagnostics(
        observations,
        predictive_means,
        latent_variances,
        observation_variances,
        interval_standard_deviations,
    )?)
}

/// Evaluate the conventional two-sided 95% Gaussian predictive interval.
pub fn gaussian_predictive_diagnostics_95(
    observations: &[f64],
    predictive_means: &[f64],
    latent_variances: &[f64],
    observation_variances: &[f64],
) -> Result<GaussianPredictiveDiagnostics> {
    gaussian_predictive_diagnostics(
        observations,
        predictive_means,
        latent_variances,
        observation_variances,
        1.959_963_984_540_054,
    )
}

/// Relative Frobenius error `||candidate - reference||_F / ||reference||_F`.
///
/// Duplicate sparse coordinates are merged before evaluation. No rescaling or
/// best-fit amplitude is applied, so this diagnostic preserves differences in
/// the represented operator scale.
pub fn relative_sparse_frobenius_error(
    candidate: &SparseMat,
    reference: &SparseMat,
) -> Result<f64> {
    if candidate.nrows() != reference.nrows() || candidate.ncols() != reference.ncols() {
        return Err(FeecGmrfError::Dimension(format!(
            "relative Frobenius error requires equal shapes, got {}x{} and {}x{}",
            candidate.nrows(),
            candidate.ncols(),
            reference.nrows(),
            reference.ncols()
        )));
    }
    let mut difference = BTreeMap::<(usize, usize), f64>::new();
    let mut reference_entries = BTreeMap::<(usize, usize), f64>::new();
    for (row, column, value) in candidate.triplet_iter() {
        *difference.entry((row, column)).or_default() += value;
    }
    for (row, column, value) in reference.triplet_iter() {
        *difference.entry((row, column)).or_default() -= value;
        *reference_entries.entry((row, column)).or_default() += value;
    }
    let denominator = reference_entries
        .values()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if denominator == 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "relative Frobenius error requires a nonzero reference matrix".to_string(),
        ));
    }
    Ok(difference
        .values()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        / denominator)
}

/// Relative Euclidean error `||candidate - reference||_2 / ||reference||_2`.
///
/// No rescaling or amplitude fitting is applied. This is suitable for
/// comparing covariance actions or other vectors that have a common physical
/// meaning and coordinate representation.
pub fn relative_vector_l2_error(candidate: &[f64], reference: &[f64]) -> Result<f64> {
    if candidate.len() != reference.len() {
        return Err(FeecGmrfError::Dimension(format!(
            "relative vector error requires equal lengths, got {} and {}",
            candidate.len(),
            reference.len()
        )));
    }
    if candidate
        .iter()
        .chain(reference)
        .any(|value| !value.is_finite())
    {
        return Err(FeecGmrfError::InvalidParameter(
            "relative vector error requires finite values".to_string(),
        ));
    }
    let denominator = reference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if denominator == 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "relative vector error requires a nonzero reference".to_string(),
        ));
    }
    Ok(candidate
        .iter()
        .zip(reference)
        .map(|(candidate, reference)| (candidate - reference).powi(2))
        .sum::<f64>()
        .sqrt()
        / denominator)
}

/// Relative absolute scalar error `|candidate-reference|/|reference|`.
///
/// No rescaling is applied.
pub fn relative_scalar_error(candidate: f64, reference: f64) -> Result<f64> {
    if !candidate.is_finite() || !reference.is_finite() || reference == 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "relative scalar error requires finite values and a nonzero reference".to_string(),
        ));
    }
    Ok((candidate - reference).abs() / reference.abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use feg_core::SparseTriplet;

    #[test]
    fn sparse_frobenius_error_merges_duplicate_coordinates_without_rescaling() {
        let reference = SparseMat::diagonal(2, 2.0);
        let candidate = SparseMat::from_triplets(
            2,
            2,
            vec![
                SparseTriplet {
                    row: 0,
                    col: 0,
                    value: 1.0,
                },
                SparseTriplet {
                    row: 0,
                    col: 0,
                    value: 2.0,
                },
                SparseTriplet {
                    row: 1,
                    col: 1,
                    value: 2.0,
                },
            ],
        );
        let error = relative_sparse_frobenius_error(&candidate, &reference).unwrap();
        assert!((error - 0.5_f64 / 2.0_f64.sqrt()).abs() <= 1.0e-14);
    }

    #[test]
    fn vector_and_scalar_relative_errors_preserve_amplitude() {
        let vector_error = relative_vector_l2_error(&[2.0, 0.0], &[1.0, 0.0]).unwrap();
        let scalar_error = relative_scalar_error(2.0, 1.0).unwrap();
        assert_eq!(vector_error, 1.0);
        assert_eq!(scalar_error, 1.0);
    }

    #[test]
    fn vector_and_scalar_relative_errors_reject_zero_references() {
        assert!(relative_vector_l2_error(&[1.0], &[0.0]).is_err());
        assert!(relative_scalar_error(1.0, 0.0).is_err());
    }

    #[test]
    fn relative_precision_certificate_reports_exact_loewner_ratios() {
        let reference = SparseMat::from_triplets(
            2,
            2,
            vec![
                SparseTriplet {
                    row: 0,
                    col: 0,
                    value: 2.0,
                },
                SparseTriplet {
                    row: 0,
                    col: 1,
                    value: 0.5,
                },
                SparseTriplet {
                    row: 1,
                    col: 0,
                    value: 0.5,
                },
                SparseTriplet {
                    row: 1,
                    col: 1,
                    value: 1.5,
                },
            ],
        );
        let candidate = SparseMat::from_triplets(
            2,
            2,
            reference
                .triplet_iter()
                .map(|(row, col, value)| SparseTriplet {
                    row,
                    col,
                    value: 1.25 * value,
                }),
        );
        let certificate = relative_precision_certificate(&candidate, &reference, 2).unwrap();
        assert_eq!(certificate.dimension, 2);
        assert!((certificate.minimum_relative_eigenvalue - 0.25).abs() <= 1.0e-13);
        assert!((certificate.maximum_relative_eigenvalue - 0.25).abs() <= 1.0e-13);
        assert!((certificate.relative_error - 0.25).abs() <= 1.0e-13);
        assert!((certificate.minimum_precision_ratio - 1.25).abs() <= 1.0e-13);
        assert!((certificate.maximum_precision_ratio - 1.25).abs() <= 1.0e-13);
        assert!((certificate.minimum_covariance_ratio - 0.8).abs() <= 1.0e-13);
        assert!((certificate.maximum_covariance_ratio - 0.8).abs() <= 1.0e-13);
        assert!(certificate.has_unit_relative_envelope());
    }

    #[test]
    fn relative_precision_certificate_enforces_dimension_guard() {
        let matrix = SparseMat::diagonal(3, 1.0);
        let error = relative_precision_certificate(&matrix, &matrix, 2).unwrap_err();
        assert!(matches!(
            error,
            FeecGmrfError::DenseDiagnosticSizeExceeded {
                dimension: 3,
                maximum: 2
            }
        ));
    }
}
