//! Minimal 2D 0-form Matérn conditioning and uncertainty workflow.
//!
//! The field is represented by vertex coefficients on a triangulated unit
//! square. We build a Matérn prior, calibrate its domain-averaged RMS
//! uncertainty, and condition it on three noisy vertex values. A horizontal
//! transect is registered as a named output, while a single query point is
//! evaluated with an ad hoc map. Both exact and seeded Monte Carlo variances are
//! reported.
//!
//! The runtime summary shows three characteristic conditioning outcomes: the
//! observed coefficients move close to their supplied values, their uncertainty
//! contracts sharply, and an unobserved query retains substantially more
//! uncertainty. The exact and Monte Carlo variance summaries provide a quick
//! check of the uncertainty estimator.
//!
//! Continue with `em_1form_uq` for mixed boundaries and PDE residuals.
//!
//! Run with `cargo run --release --example minimal_0form`.

use feec_gmrf::prelude::*;
use manifold::gen::cartesian::CartesianMeshInfo;
use std::f64::consts::PI;

const PRIOR_FIELD_RMS_TARGET: f64 = 1.0;
const SENSOR_STANDARD_DEVIATION: f64 = 0.05;

fn main() -> Result<()> {
    // A 0-form cochain has one coefficient per vertex.
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 32, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    // tau=1 sets the initial precision amplitude. The RMS calibration below
    // gives the prior its requested physical uncertainty.
    let parameters = MaternParameters::from_practical_range(MaternAlpha::Two, 0.25, 2, 1.0)?;
    let uncalibrated_prior = MaternPriorBuilder::from_feec(&topology, &metric, 0)?
        .parameters(parameters)
        .build()?;
    let scalar_rms_weights = scalar_field_l2_rms_weights(&topology, &coords)?;
    let (prior, calibration) = calibrate_prior_to_weighted_physical_rms(
        &uncalibrated_prior,
        &LinearMap::identity(uncalibrated_prior.dimension()),
        &scalar_rms_weights,
        PRIOR_FIELD_RMS_TARGET,
    )?;

    // `at_indices` builds a noisy coefficient selector. Weighted `LinearMap`s
    // provide general linear observations.
    let sensor_points = [(0.25, 0.25), (0.75, 0.25), (0.50, 0.50)];
    let sensor_indices = sensor_points
        .iter()
        .map(|&(x, y)| find_vertex(&coords, x, y))
        .collect::<Result<Vec<_>>>()?;
    let sensor_values = sensor_points
        .iter()
        .map(|&(x, y)| truth(x, y))
        .collect::<Vec<_>>();
    let sensor_map = LinearMap::selector(prior.dimension(), &sensor_indices)?;
    let observations = LinearObservation::at_indices(
        prior.dimension(),
        &sensor_indices,
        sensor_values.clone(),
        GaussianNoise::standard_deviation(SENSOR_STANDARD_DEVIATION)?,
    )?;

    // Register the transect by name. The single-point map will be queried
    // directly after conditioning.
    let mut transect = coords
        .coord_iter()
        .enumerate()
        .filter(|(_, point)| (point[1] - 0.5).abs() < 1.0e-12)
        .map(|(index, point)| (index, point[0]))
        .collect::<Vec<_>>();
    transect.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
    let transect_map = LinearMap::selector(
        prior.dimension(),
        &transect.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
    )?;
    let query_index = find_vertex(&coords, 0.375, 0.625)?;
    let query_map = LinearMap::selector(prior.dimension(), &[query_index])?;
    let prior_sensor_variances = prior.pushforward_variances(&sensor_map)?;
    let prior_query_variance = prior.pushforward_variances(&query_map)?[0];

    // Linear conditioning returns the common `Posterior` type.
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .observe(observations)?
        .derive(DerivedQuantity::new("y=0.5 transect", transect_map)?)?
        .condition()?;
    let mc = VarianceMethod::MonteCarlo(MonteCarloVarianceConfig::new(1024, 8, 42)?);

    let sensor_predictions = posterior.pushforward_mean(&sensor_map)?;
    println!("2D 0-form Matérn workflow");
    println!("  vertices: {}", posterior.mean().len());
    println!(
        "  prior scalar-field RMS calibration: {:.6} -> {:.6}, precision scale {:.6e}",
        calibration.uncalibrated_rms, calibration.target_rms, calibration.precision_scale,
    );
    let sensor_posterior_variances = posterior
        .pushforward_variance_estimate(&sensor_map, VarianceMethod::Exact)?
        .values;
    for (index, (((point, observed), predicted), (prior_variance, posterior_variance))) in
        sensor_points
            .iter()
            .zip(sensor_values.iter())
            .zip(sensor_predictions.iter())
            .zip(
                prior_sensor_variances
                    .iter()
                    .zip(&sensor_posterior_variances),
            )
            .enumerate()
    {
        println!(
            "  sensor {index}: ({:.2}, {:.2}) observed={observed:+.6} posterior={predicted:+.6} residual={:+.3} noise SD, prior/posterior SD={:.6}/{:.6}",
            point.0,
            point.1,
            (predicted - observed) / SENSOR_STANDARD_DEVIATION,
            prior_variance.max(0.0).sqrt(),
            posterior_variance.max(0.0).sqrt(),
        );
    }

    let transect_mean = posterior.derived_mean("y=0.5 transect")?;
    let transect_exact =
        posterior.derived_variance_estimate("y=0.5 transect", VarianceMethod::Exact)?;
    let transect_mc = posterior.derived_variance_estimate("y=0.5 transect", mc)?;
    let transect_truth = transect
        .iter()
        .map(|(_, x)| truth(*x, 0.5))
        .collect::<Vec<_>>();
    let transect_rmse = mean_squared_error(&transect_mean, &transect_truth).sqrt();
    let transect_coverage = coverage_fraction(
        &transect_mean,
        &transect_exact.values,
        &transect_truth,
        1.96,
    );
    println!(
        "  transect: {} points, mean range [{:+.6}, {:+.6}], truth RMSE {:.6}, pointwise 95% coverage {:.1}%, max MC relative SE {:.3}",
        transect_mean.len(),
        transect_mean.iter().copied().fold(f64::INFINITY, f64::min),
        transect_mean
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        transect_rmse,
        100.0 * transect_coverage,
        transect_mc
            .relative_standard_error
            .as_ref()
            .expect("eight batches provide standard errors")
            .iter()
            .copied()
            .fold(0.0, f64::max),
    );
    println!(
        "  transect exact/MC variance relative L2 error: {:.4}",
        relative_l2(&transect_mc.values, &transect_exact.values)
    );

    let query_mean = posterior.pushforward_mean(&query_map)?[0];
    let query_exact = posterior
        .pushforward_variance_estimate(&query_map, VarianceMethod::Exact)?
        .values[0];
    let query_mc = posterior.pushforward_variance_estimate(&query_map, mc)?;
    println!(
        "  ad hoc query (0.375, 0.625): truth={:+.6}, mean={:+.6}, truth z={:+.3}, prior/posterior std={:.6}/{:.6}, MC std={:.6} ± variance SE {:.3e}",
        2.0_f64.sqrt() / 4.0,
        query_mean,
        (query_mean - 2.0_f64.sqrt() / 4.0) / query_exact.sqrt(),
        prior_query_variance.sqrt(),
        query_exact.sqrt(),
        query_mc.values[0].sqrt(),
        query_mc.batch_standard_error.as_ref().unwrap()[0],
    );

    let exact = posterior.latent_variance_estimate(VarianceMethod::Exact)?;
    let estimated = posterior.latent_variance_estimate(mc)?;
    let latent_mc_error = relative_l2(&estimated.values, &exact.values);
    println!(
        "  all-coefficient exact/MC variance relative L2 error: {:.4} ({} samples in {:?})",
        latent_mc_error, estimated.sample_count, estimated.batch_sizes,
    );
    let sensor_max_residual = sensor_predictions
        .iter()
        .zip(&sensor_values)
        .map(|(predicted, observed)| ((predicted - observed) / SENSOR_STANDARD_DEVIATION).abs())
        .fold(0.0, f64::max);
    let sensor_min_variance_reduction = prior_sensor_variances
        .iter()
        .zip(&sensor_posterior_variances)
        .map(|(prior, posterior)| 1.0 - posterior / prior)
        .fold(1.0, f64::min);
    let query_truth_z = (query_mean - 2.0_f64.sqrt() / 4.0).abs() / query_exact.max(0.0).sqrt();
    validate_exemplar_outcomes(
        sensor_max_residual,
        sensor_min_variance_reduction,
        query_truth_z,
        transect_coverage,
        latent_mc_error,
    )?;
    println!(
        "  exemplar checks: PASS (sensor assimilation, uncertainty reduction, truth coverage, exact/MC agreement)"
    );
    Ok(())
}

fn truth(x: f64, y: f64) -> f64 {
    (2.0 * PI * x).sin() + 0.5 * (2.0 * PI * y).cos()
}

fn find_vertex(
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    x: f64,
    y: f64,
) -> Result<usize> {
    coords
        .coord_iter()
        .position(|point| (point[0] - x).abs() < 1.0e-12 && (point[1] - y).abs() < 1.0e-12)
        .ok_or_else(|| {
            FeecGmrfError::InvalidParameter(format!(
                "requested point ({x}, {y}) is not a mesh vertex"
            ))
        })
}

fn relative_l2(estimate: &[f64], exact: &[f64]) -> f64 {
    let numerator = estimate
        .iter()
        .zip(exact)
        .map(|(estimate, exact)| (estimate - exact).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = exact.iter().map(|value| value * value).sum::<f64>().sqrt();
    numerator / denominator
}

fn mean_squared_error(estimate: &[f64], truth: &[f64]) -> f64 {
    estimate
        .iter()
        .zip(truth)
        .map(|(estimate, truth)| (estimate - truth).powi(2))
        .sum::<f64>()
        / estimate.len() as f64
}

fn coverage_fraction(mean: &[f64], variance: &[f64], truth: &[f64], z: f64) -> f64 {
    mean.iter()
        .zip(variance)
        .zip(truth)
        .filter(|((mean, variance), truth)| (*mean - **truth).abs() <= z * variance.max(0.0).sqrt())
        .count() as f64
        / mean.len() as f64
}

fn validate_exemplar_outcomes(
    sensor_max_residual: f64,
    sensor_min_variance_reduction: f64,
    query_truth_z: f64,
    transect_coverage: f64,
    latent_mc_error: f64,
) -> Result<()> {
    if sensor_max_residual > 1.0
        || sensor_min_variance_reduction < 0.9
        || query_truth_z > 2.0
        || transect_coverage < 0.9
        || latent_mc_error > 0.1
    {
        return Err(FeecGmrfError::Inference(format!(
            "minimal exemplar failed: sensor residual {sensor_max_residual:.3}, minimum variance reduction {sensor_min_variance_reduction:.3}, query |z| {query_truth_z:.3}, transect coverage {transect_coverage:.3}, MC error {latent_mc_error:.3}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exemplar_gate_accepts_and_rejects_expected_diagnostics() {
        assert!(validate_exemplar_outcomes(0.1, 0.95, 1.0, 0.95, 0.05).is_ok());
        assert!(validate_exemplar_outcomes(1.1, 0.95, 1.0, 0.95, 0.05).is_err());
        assert!(validate_exemplar_outcomes(0.1, 0.95, 2.1, 0.95, 0.05).is_err());
    }
}
