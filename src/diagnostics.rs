//! Diagnostics for Gaussian predictive distributions and operator comparisons.

use std::collections::BTreeMap;

use crate::operator::SparseMat;
use crate::{FeecGmrfError, Result};

pub use gmrf_core::GaussianPredictiveDiagnostics;

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
}
