//! Stable diagnostics for Gaussian predictive distributions.

use crate::Result;

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
