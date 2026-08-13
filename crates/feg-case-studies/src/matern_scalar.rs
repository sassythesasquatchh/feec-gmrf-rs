//! Report-backed scalar Matérn prior validation.
//!
//! The submitted diagnostic evaluates the covariance between one central
//! vertex and vertices at integer coordinate-axis lags. It deliberately does
//! not form a dense covariance over every interior vertex.

use feec_gmrf::prelude::{
    DerivedQuantity, LinearGaussianModelBuilder, LinearMap, MassInversePolicy, MaternAlpha,
    MaternParameters, MaternPriorBuilder, SparseMat,
};
use feg_gp::{matern_covariance_euclidean, EuclideanMaternConfig};
use libm::tgamma;
use manifold::gen::cartesian::CartesianMeshInfo;
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
    time::Instant,
};

/// Immutable inputs for the scalar Lindgren-style diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarMaternValidationConfig {
    pub dimension: usize,
    pub range_cells: usize,
    pub level: usize,
}

impl ScalarMaternValidationConfig {
    pub const fn smoke() -> Self {
        Self {
            dimension: 3,
            range_cells: 2,
            level: 8,
        }
    }

    /// Configuration reported in the submitted thesis.
    pub const fn thesis_submitted() -> Self {
        Self {
            dimension: 3,
            range_cells: 10,
            level: 64,
        }
    }
}

/// One center-to-axis-lag covariance comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarMaternCorrelationRow {
    pub lag: usize,
    pub distance: f64,
    pub reference_correlation: f64,
    pub gmrf_correlation: f64,
    pub absolute_error: f64,
    pub point_variance: f64,
}

/// Numerical result and provenance-relevant timings for one validation run.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarMaternValidationReport {
    pub config: ScalarMaternValidationConfig,
    pub alpha: f64,
    pub nu: f64,
    pub sigma2: f64,
    pub ndofs: usize,
    pub h_over_range: f64,
    pub margin_over_range: f64,
    pub kappa: f64,
    pub tau: f64,
    pub center_variance: f64,
    pub variance_relative_error: f64,
    pub correlation_rmse: f64,
    pub factor_seconds: f64,
    pub covariance_seconds: f64,
    pub total_seconds: f64,
    pub correlations: Vec<ScalarMaternCorrelationRow>,
}

/// Run the report's scalar Matérn diagnostic using only the reported axis-lag
/// observables.
pub fn run_scalar_matern_validation(
    config: ScalarMaternValidationConfig,
) -> Result<ScalarMaternValidationReport, String> {
    validate_config(config)?;
    let total_start = Instant::now();
    let alpha = 2.0;
    let nu = alpha - config.dimension as f64 / 2.0;
    let range = config.range_cells as f64;
    let kappa = (8.0 * nu).sqrt() / range;
    let sigma2 = 1.0;
    let tau = (tgamma(nu)
        / (sigma2
            * tgamma(alpha)
            * (4.0 * std::f64::consts::PI).powf(config.dimension as f64 / 2.0)
            * kappa.powf(2.0 * nu)))
    .sqrt();

    let mesh =
        CartesianMeshInfo::new_unit_scaled(config.dimension, config.level, config.level as f64);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let ndofs = coords.nvertices();
    let prior = MaternPriorBuilder::from_feec(&topology, &metric, 0)
        .map_err(|error| error.to_string())?
        .parameters(
            MaternParameters::new(MaternAlpha::Two, kappa, tau)
                .map_err(|error| error.to_string())?,
        )
        .mass_inverse(MassInversePolicy::RowSumLumped)
        .build()
        .map_err(|error| error.to_string())?;

    let center_axis = config.level / 2;
    let max_lag = 2 * config.range_cells;
    let axis_length = config.level + 1;
    let indices = (0..=max_lag)
        .map(|lag| {
            vertex_index(
                config.dimension,
                center_axis + lag,
                center_axis,
                center_axis,
                axis_length,
            )
        })
        .collect::<Vec<_>>();
    let operator = LinearMap::new(
        SparseMat::from_rows(
            ndofs,
            &indices
                .iter()
                .map(|&index| vec![(index, 1.0)])
                .collect::<Vec<_>>(),
        )
        .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let derived =
        DerivedQuantity::new("axis-lag-values", operator).map_err(|error| error.to_string())?;

    let factor_start = Instant::now();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .derive(derived)
        .map_err(|error| error.to_string())?
        .condition()
        .map_err(|error| error.to_string())?;
    let factor_seconds = factor_start.elapsed().as_secs_f64();
    let covariance_start = Instant::now();
    let covariance = posterior
        .derived_covariance("axis-lag-values")
        .map_err(|error| error.to_string())?;
    let covariance_seconds = covariance_start.elapsed().as_secs_f64();

    let center_variance = covariance[0][0];
    let reference_config = EuclideanMaternConfig {
        kappa,
        nu,
        variance: sigma2,
    };
    let mut squared_error = 0.0;
    let mut correlations = Vec::with_capacity(max_lag + 1);
    for (lag, covariance_row) in covariance.iter().enumerate() {
        let distance = lag as f64;
        let reference_correlation = matern_covariance_euclidean(distance, reference_config)
            .map_err(|error| error.to_string())?;
        let point_variance = covariance_row[lag];
        let denominator = (center_variance.max(0.0) * point_variance.max(0.0)).sqrt();
        let gmrf_correlation = if denominator > f64::EPSILON {
            covariance[0][lag] / denominator
        } else {
            0.0
        };
        let error = gmrf_correlation - reference_correlation;
        squared_error += error * error;
        correlations.push(ScalarMaternCorrelationRow {
            lag,
            distance,
            reference_correlation,
            gmrf_correlation,
            absolute_error: error.abs(),
            point_variance,
        });
    }
    let correlation_rmse = (squared_error / correlations.len() as f64).sqrt();

    Ok(ScalarMaternValidationReport {
        config,
        alpha,
        nu,
        sigma2,
        ndofs,
        h_over_range: 1.0 / range,
        margin_over_range: (config.level as f64 / 2.0) / range,
        kappa,
        tau,
        center_variance,
        variance_relative_error: (center_variance - sigma2).abs() / sigma2,
        correlation_rmse,
        factor_seconds,
        covariance_seconds,
        total_seconds: total_start.elapsed().as_secs_f64(),
        correlations,
    })
}

/// Write the two report-facing CSV artifacts.
pub fn write_scalar_matern_validation_outputs(
    report: &ScalarMaternValidationReport,
    output_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let mut summary = BufWriter::new(File::create(output_dir.join("summary.csv"))?);
    writeln!(
        summary,
        "dimension,range_cells,level,ndofs,h_over_range,margin_over_range,kappa,tau,center_variance,variance_relative_error,correlation_rmse,factor_seconds,covariance_seconds,total_seconds"
    )?;
    writeln!(
        summary,
        "{},{},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
        report.config.dimension,
        report.config.range_cells,
        report.config.level,
        report.ndofs,
        report.h_over_range,
        report.margin_over_range,
        report.kappa,
        report.tau,
        report.center_variance,
        report.variance_relative_error,
        report.correlation_rmse,
        report.factor_seconds,
        report.covariance_seconds,
        report.total_seconds,
    )?;

    let mut correlations = BufWriter::new(File::create(output_dir.join("correlations.csv"))?);
    writeln!(
        correlations,
        "lag,distance,reference_correlation,gmrf_correlation,absolute_error,point_variance"
    )?;
    for row in &report.correlations {
        writeln!(
            correlations,
            "{},{:.16},{:.16},{:.16},{:.16},{:.16}",
            row.lag,
            row.distance,
            row.reference_correlation,
            row.gmrf_correlation,
            row.absolute_error,
            row.point_variance,
        )?;
    }
    Ok(())
}

fn validate_config(config: ScalarMaternValidationConfig) -> Result<(), String> {
    if !matches!(config.dimension, 2 | 3) {
        return Err("scalar Matérn validation dimension must be 2 or 3".to_string());
    }
    if config.range_cells == 0 {
        return Err("scalar Matérn validation range_cells must be positive".to_string());
    }
    if config.level < 4 * config.range_cells {
        return Err(
            "scalar Matérn validation level must be at least four times range_cells".to_string(),
        );
    }
    Ok(())
}

fn vertex_index(dimension: usize, x: usize, y: usize, z: usize, axis_length: usize) -> usize {
    match dimension {
        2 => x + axis_length * y,
        3 => x + axis_length * (y + axis_length * z),
        _ => unreachable!("dimension is validated"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submitted_profile_is_the_reported_axis_lag_experiment() {
        let config = ScalarMaternValidationConfig::thesis_submitted();
        assert_eq!(config.dimension, 3);
        assert_eq!(config.range_cells, 10);
        assert_eq!(config.level, 64);
        assert_eq!(2 * config.range_cells + 1, 21);
    }

    #[test]
    fn smoke_run_uses_only_center_to_axis_lag_outputs() {
        let report = run_scalar_matern_validation(ScalarMaternValidationConfig::smoke()).unwrap();
        assert_eq!(report.ndofs, 9_usize.pow(3));
        assert_eq!(report.correlations.len(), 5);
        assert!(report.correlation_rmse.is_finite());
        assert!(report.variance_relative_error.is_finite());
    }
}
