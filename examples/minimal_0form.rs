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
use std::{f64::consts::PI, fs, path::Path};

const PRIOR_FIELD_RMS_TARGET: f64 = 1.0;
const SENSOR_STANDARD_DEVIATION: f64 = 0.05;
const OUTPUT_ROOT: &str = "out/examples/minimal_0form";

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

    let transect_truth = transect
        .iter()
        .map(|(_, x)| truth(*x, 0.5))
        .collect::<Vec<_>>();
    let query_truth = 2.0_f64.sqrt() / 4.0;
    let full_truth = coords
        .coord_iter()
        .map(|point| truth(point[0], point[1]))
        .collect::<Vec<_>>();
    let sensor_labels = sensor_points
        .iter()
        .enumerate()
        .map(|(index, _)| format!("sensor_{index}"))
        .collect::<Vec<_>>();
    let mut report = PosteriorReportBuilder::new(&mut posterior)
        .field(
            FieldRequest::mapped(
                "sensors",
                "Observed sensor coefficients",
                sensor_map.clone(),
            )
            .unit("field units")
            .truth(sensor_values.clone())
            .baseline_variances(prior_sensor_variances.clone()),
        )
        .prediction(PredictionRequest::mapped(
            "sensor_predictions",
            "Sensor posterior predictions",
            sensor_map,
            sensor_labels,
            sensor_values.clone(),
            vec![SENSOR_STANDARD_DEVIATION.powi(2); sensor_values.len()],
        ))
        .field(
            FieldRequest::derived(
                "transect_exact",
                "y=0.5 transect (exact variance)",
                "y=0.5 transect",
            )
            .truth(transect_truth),
        )
        .field(
            FieldRequest::derived(
                "transect_mc",
                "y=0.5 transect (Monte Carlo variance)",
                "y=0.5 transect",
            )
            .variance_method(mc),
        )
        .field(
            FieldRequest::mapped("query_exact", "Ad hoc query", query_map.clone())
                .truth(vec![query_truth])
                .baseline_variances(vec![prior_query_variance]),
        )
        .field(
            FieldRequest::mapped("query_mc", "Ad hoc query (Monte Carlo)", query_map)
                .variance_method(mc),
        )
        .field(FieldRequest::latent("latent_exact", "All coefficients").truth(full_truth))
        .field(
            FieldRequest::latent("latent_mc", "All coefficients (Monte Carlo)").variance_method(mc),
        )
        .include_factorization_diagnostics(true)
        .build()?;

    let sensor = report.field("sensors").expect("requested field");
    let transect_exact = report.field("transect_exact").expect("requested field");
    let transect_mc = report.field("transect_mc").expect("requested field");
    let query = report.field("query_exact").expect("requested field");
    let latent_exact = report.field("latent_exact").expect("requested field");
    let latent_mc = report.field("latent_mc").expect("requested field");
    let sensor_max_residual = sensor
        .errors
        .as_ref()
        .expect("sensor truth supplied")
        .iter()
        .map(|error| (error / SENSOR_STANDARD_DEVIATION).abs())
        .fold(0.0, f64::max);
    let sensor_min_variance_reduction = sensor
        .variance_reductions
        .as_ref()
        .expect("sensor baseline supplied")
        .iter()
        .flatten()
        .copied()
        .fold(1.0, f64::min);
    let query_truth_z = query.z_scores.as_ref().unwrap()[0].unwrap().abs();
    let transect_rmse = transect_exact
        .truth_rmse()
        .expect("transect truth supplied");
    let transect_coverage = transect_exact
        .truth_coverage(1.96)
        .expect("transect truth supplied");
    let transect_mc_error = relative_l2(
        &transect_mc.variance.values,
        &transect_exact.variance.values,
    );
    let latent_mc_error = relative_l2(&latent_mc.variance.values, &latent_exact.variance.values);

    report.push_metric(ReportMetric::new(
        "transect_rmse",
        "Transect truth RMSE",
        transect_rmse,
    ))?;
    report.push_metric(ReportMetric::new(
        "transect_coverage",
        "Transect pointwise 95% coverage",
        transect_coverage,
    ))?;
    report.push_metric(ReportMetric::new(
        "transect_mc_relative_l2",
        "Transect exact/MC variance relative L2 error",
        transect_mc_error,
    ))?;
    report.push_metric(ReportMetric::new(
        "latent_mc_relative_l2",
        "All-coefficient exact/MC variance relative L2 error",
        latent_mc_error,
    ))?;
    report.push_metric(ReportMetric::new(
        "prior_rms_precision_scale",
        "Prior RMS calibration precision scale",
        calibration.precision_scale,
    ))?;

    println!("2D 0-form Matérn workflow");
    println!(
        "  prior scalar-field RMS calibration: {:.6} -> {:.6}",
        calibration.uncalibrated_rms, calibration.target_rms
    );
    write_console_report(
        &mut std::io::stdout(),
        &report,
        &ConsoleReportOptions::default(),
    )?;
    validate_exemplar_outcomes(
        sensor_max_residual,
        sensor_min_variance_reduction,
        query_truth_z,
        transect_coverage,
        latent_mc_error,
    )?;
    println!("exemplar checks: PASS (sensor assimilation, uncertainty reduction, truth coverage, exact/MC agreement)");

    let output_root = Path::new(OUTPUT_ROOT);
    fs::create_dir_all(output_root)?;
    write_csv_directory(output_root, &report.tables()?)?;
    write_minimal_tables(output_root, &report, &sensor_points, &transect)?;
    let mut vtu = CochainVtuBuilder::new(0);
    vtu.add_field_report("posterior", report.field("latent_exact").unwrap())?
        .add_values(
            "truth",
            report.field("latent_exact").unwrap().truth.clone().unwrap(),
        )?
        .add_values(
            "posterior_variance_mc",
            report.field("latent_mc").unwrap().variance.values.clone(),
        )?
        .add_values(
            "posterior_standard_deviation_mc",
            report
                .field("latent_mc")
                .unwrap()
                .standard_deviations
                .clone(),
        )?;
    vtu.write(output_root.join("posterior_0form.vtu"), &coords, &topology)?;
    println!("wrote reporting artifacts under {}", output_root.display());
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

fn write_minimal_tables(
    output_root: &Path,
    report: &PosteriorReport,
    sensor_points: &[(f64, f64)],
    transect_points: &[(usize, f64)],
) -> Result<()> {
    let sensors = report.field("sensors").expect("requested field");
    let mut sensor_table = ReportTable::new(
        "sensors",
        [
            "index",
            "x",
            "y",
            "observed",
            "posterior_mean",
            "posterior_variance",
            "posterior_standard_deviation",
            "noise_standardized_residual",
            "prior_variance",
            "variance_reduction",
        ]
        .map(str::to_string)
        .to_vec(),
    )?;
    for (index, point) in sensor_points.iter().enumerate() {
        sensor_table.push_row(vec![
            ReportCell::Integer(index as i64),
            ReportCell::Float(point.0),
            ReportCell::Float(point.1),
            ReportCell::Float(sensors.truth.as_ref().unwrap()[index]),
            ReportCell::Float(sensors.mean[index]),
            ReportCell::Float(sensors.variance.values[index]),
            ReportCell::Float(sensors.standard_deviations[index]),
            ReportCell::Float(sensors.errors.as_ref().unwrap()[index] / SENSOR_STANDARD_DEVIATION),
            ReportCell::Float(sensors.baseline_variances.as_ref().unwrap()[index]),
            optional_cell(sensors.variance_reductions.as_ref().unwrap()[index]),
        ])?;
    }
    write_csv(output_root.join("sensors.csv"), &sensor_table)?;

    let exact = report.field("transect_exact").expect("requested field");
    let mc = report.field("transect_mc").expect("requested field");
    let mut transect_table = ReportTable::new(
        "transect",
        [
            "index",
            "x",
            "truth",
            "posterior_mean",
            "exact_variance",
            "exact_standard_deviation",
            "truth_z_score",
            "mc_variance",
            "mc_standard_deviation",
            "mc_relative_standard_error",
        ]
        .map(str::to_string)
        .to_vec(),
    )?;
    for (index, (_, x)) in transect_points.iter().enumerate() {
        transect_table.push_row(vec![
            ReportCell::Integer(index as i64),
            ReportCell::Float(*x),
            ReportCell::Float(exact.truth.as_ref().unwrap()[index]),
            ReportCell::Float(exact.mean[index]),
            ReportCell::Float(exact.variance.values[index]),
            ReportCell::Float(exact.standard_deviations[index]),
            optional_cell(exact.z_scores.as_ref().unwrap()[index]),
            ReportCell::Float(mc.variance.values[index]),
            ReportCell::Float(mc.standard_deviations[index]),
            mc.variance
                .relative_standard_error
                .as_ref()
                .map(|values| ReportCell::Float(values[index]))
                .unwrap_or(ReportCell::Missing),
        ])?;
    }
    write_csv(output_root.join("transect.csv"), &transect_table)?;

    let exact = report.field("query_exact").expect("requested field");
    let mc = report.field("query_mc").expect("requested field");
    let mut query_table = ReportTable::new(
        "query",
        [
            "x",
            "y",
            "truth",
            "posterior_mean",
            "exact_variance",
            "exact_standard_deviation",
            "truth_z_score",
            "prior_variance",
            "variance_reduction",
            "mc_variance",
            "mc_variance_standard_error",
        ]
        .map(str::to_string)
        .to_vec(),
    )?;
    query_table.push_row(vec![
        ReportCell::Float(0.375),
        ReportCell::Float(0.625),
        ReportCell::Float(exact.truth.as_ref().unwrap()[0]),
        ReportCell::Float(exact.mean[0]),
        ReportCell::Float(exact.variance.values[0]),
        ReportCell::Float(exact.standard_deviations[0]),
        optional_cell(exact.z_scores.as_ref().unwrap()[0]),
        ReportCell::Float(exact.baseline_variances.as_ref().unwrap()[0]),
        optional_cell(exact.variance_reductions.as_ref().unwrap()[0]),
        ReportCell::Float(mc.variance.values[0]),
        mc.variance
            .batch_standard_error
            .as_ref()
            .map(|values| ReportCell::Float(values[0]))
            .unwrap_or(ReportCell::Missing),
    ])?;
    write_csv(output_root.join("query.csv"), &query_table)?;

    let exact = report.field("latent_exact").expect("requested field");
    let mc = report.field("latent_mc").expect("requested field");
    let mut estimator_table = ReportTable::new(
        "estimator_comparison",
        [
            "index",
            "exact_variance",
            "mc_variance",
            "difference",
            "mc_batch_standard_error",
            "mc_relative_standard_error",
        ]
        .map(str::to_string)
        .to_vec(),
    )?;
    for index in 0..exact.mean.len() {
        estimator_table.push_row(vec![
            ReportCell::Integer(index as i64),
            ReportCell::Float(exact.variance.values[index]),
            ReportCell::Float(mc.variance.values[index]),
            ReportCell::Float(mc.variance.values[index] - exact.variance.values[index]),
            mc.variance
                .batch_standard_error
                .as_ref()
                .map(|values| ReportCell::Float(values[index]))
                .unwrap_or(ReportCell::Missing),
            mc.variance
                .relative_standard_error
                .as_ref()
                .map(|values| ReportCell::Float(values[index]))
                .unwrap_or(ReportCell::Missing),
        ])?;
    }
    write_csv(
        output_root.join("estimator_comparison.csv"),
        &estimator_table,
    )?;
    Ok(())
}

fn optional_cell(value: Option<f64>) -> ReportCell {
    value.map(ReportCell::Float).unwrap_or(ReportCell::Missing)
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
