//! Physical outputs represented as explicit compositions of FEEC operators.

use crate::operator::LinearMap;
use crate::prior::GaussianPrior;
use crate::{FeecGmrfError, Result};
use gmrf_core::types::{DenseMatrix, Vector};
use gmrf_core::Gmrf;

/// A named physical pushforward from latent cochains to reported values.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalMap {
    name: String,
    map: LinearMap,
}

impl PhysicalMap {
    /// Construct a physical map from an explicit linear operator.
    pub fn new(name: impl Into<String>, map: LinearMap) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "physical quantity name must not be empty".to_string(),
            ));
        }
        Ok(Self { name, map })
    }

    /// Physical quantity name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Composed operator.
    pub fn map(&self) -> &LinearMap {
        &self.map
    }

    /// Apply the physical map.
    pub fn apply(&self, cochain: &[f64]) -> Result<Vec<f64>> {
        self.map.apply(cochain)
    }
}

/// Construct the canonical magnetic-field map
/// `A --D1--> flux cochain --reconstruction/Hodge--> physical B`.
pub fn magnetic_field_map(
    exterior_derivative_1: &LinearMap,
    flux_reconstruction: &LinearMap,
) -> Result<PhysicalMap> {
    PhysicalMap::new(
        "magnetic_field",
        flux_reconstruction.compose(exterior_derivative_1)?,
    )
}

/// Result of scaling a prior to a target root-mean-square physical standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalRmsCalibration {
    /// RMS standard deviation before calibration.
    pub uncalibrated_rms: f64,
    /// Requested RMS standard deviation.
    pub target_rms: f64,
    /// Scalar multiplying the original precision.
    pub precision_scale: f64,
}

/// Scale a prior so the mean marginal variance of `physical_map * x` is
/// `target_rms^2`.
///
/// This uses the generic GMRF covariance action and therefore works for any
/// FEEC form degree, geometry, or physical reconstruction. The returned prior
/// has the same mean and form-degree metadata as the input.
pub fn calibrate_prior_to_physical_rms(
    prior: &GaussianPrior,
    physical_map: &LinearMap,
    target_rms: f64,
) -> Result<(GaussianPrior, PhysicalRmsCalibration)> {
    if !target_rms.is_finite() || target_rms <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "target physical RMS must be finite and positive".to_string(),
        ));
    }
    if physical_map.output_dimension() == 0 {
        return Err(FeecGmrfError::InvalidParameter(
            "physical RMS calibration requires at least one output".to_string(),
        ));
    }

    let eliminated =
        prior.eliminate_map(physical_map, &vec![0.0; physical_map.output_dimension()])?;
    let mut gmrf = Gmrf::from_mean_and_precision(
        Vector::zeros(prior.dimension()),
        crate::infer::gmrf_sparse(prior.precision()),
    )?;
    let operator = crate::infer::sparse_row_operator(eliminated.reduced())?;
    let variances = gmrf
        .exact_transformed_variance_decomposition(
            &operator,
            &DenseMatrix::zeros(0, prior.dimension()),
        )?
        .unconstrained_diag;
    let mean_variance = variances.iter().sum::<f64>() / variances.len() as f64;
    if !mean_variance.is_finite() || mean_variance <= 0.0 {
        return Err(FeecGmrfError::Inference(
            "physical pushforward has zero or non-finite prior variance".to_string(),
        ));
    }
    let uncalibrated_rms = mean_variance.sqrt();
    let precision_scale = mean_variance / target_rms.powi(2);
    let calibrated = prior.with_precision_scale(precision_scale)?;
    Ok((
        calibrated,
        PhysicalRmsCalibration {
            uncalibrated_rms,
            target_rms,
            precision_scale,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::EssentialBoundaryConditions;
    use crate::operator::SparseMat;

    #[test]
    fn magnetic_map_is_explicit_operator_composition() {
        let derivative = LinearMap::new(SparseMat::diagonal(2, 2.0)).unwrap();
        let reconstruction = LinearMap::new(SparseMat::diagonal(2, 3.0)).unwrap();
        let magnetic = magnetic_field_map(&derivative, &reconstruction).unwrap();
        assert_eq!(magnetic.apply(&[1.0, -1.0]).unwrap(), vec![6.0, -6.0]);
    }

    #[test]
    fn physical_rms_calibration_uses_pushforward_covariance() {
        let prior = GaussianPrior::new(vec![0.0; 2], SparseMat::diagonal(2, 1.0)).unwrap();
        let map = LinearMap::new(SparseMat::diagonal(2, 2.0)).unwrap();
        let (calibrated, report) = calibrate_prior_to_physical_rms(&prior, &map, 0.5).unwrap();
        assert!((report.uncalibrated_rms - 2.0).abs() < 1e-12);
        assert!((report.precision_scale - 16.0).abs() < 1e-12);
        assert_eq!(calibrated.precision(), &SparseMat::diagonal(2, 16.0));
    }

    #[test]
    fn physical_rms_calibration_accepts_full_boundary_constrained_map() {
        let prior = GaussianPrior::new(vec![0.0; 2], SparseMat::diagonal(2, 1.0))
            .unwrap()
            .condition_on_essential_boundary(
                EssentialBoundaryConditions::prescribed(vec![1], vec![3.0]).unwrap(),
            )
            .unwrap();
        let full_map = LinearMap::identity(2);
        let (calibrated, report) = calibrate_prior_to_physical_rms(&prior, &full_map, 1.0).unwrap();
        assert!((report.uncalibrated_rms - 0.5_f64.sqrt()).abs() < 1e-12);
        assert!((report.precision_scale - 0.5).abs() < 1e-12);
        assert_eq!(calibrated.precision(), &SparseMat::diagonal(1, 0.5));
        assert_eq!(calibrated.cochain_mean().unwrap(), [0.0, 3.0]);
    }
}
