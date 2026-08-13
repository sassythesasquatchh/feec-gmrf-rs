use faer::Mat;
use feg_gp::{
    condition_full_covariance_with_covariance, matern_covariance_matrix_euclidean,
    EuclideanMaternConfig, SpectralMaternConfig, SpectralMaternGp,
};
use feg_infer::prior::matern::zero_form::{
    build_laplace_beltrami_0form, build_matern_precision_0form, MaternConfig, MaternMassInverse,
};
use feg_infer::sparse::feec_csr_to_gmrf;
use gmrf_core::observation::{apply_gaussian_observations, observation_selector};
use gmrf_core::types::{DenseMatrix, Vector as GmrfVector};
use gmrf_core::{Gmrf, SparseRowOperator};
use libm::tgamma;
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::mesh::MeshCoords,
    topology::complex::Complex,
};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{self, BufWriter, Write},
    ops::Index,
    path::{Path, PathBuf},
    time::Instant,
};

const DEFAULT_DIMENSION: usize = 3;
const DEFAULT_ALPHA: f64 = 2.0;
const DEFAULT_SIGMA2: f64 = 1.0;
const DEFAULT_PRACTICAL_RANGE: f64 = 0.20;
const DEFAULT_NOISE_VARIANCE: f64 = 1e-4;
const DEFAULT_INTERIOR_MARGIN: f64 = 0.25;
const DEFAULT_PROBE_ANCHOR_LEVEL: usize = 12;
const DEFAULT_SPECTRAL_K: usize = 64;
const DEFAULT_MAX_CORRELATION_PAIRS: usize = 12_000;
const DEFAULT_CORRELATION_BIN_COUNT: usize = 32;
const GRID_ALIGNMENT_TOLERANCE: f64 = 1e-9;

pub type Point3 = [f64; 3];

#[derive(Debug, Clone)]
pub struct CubeZeroFormKernelValidationConfig {
    pub levels: Vec<usize>,
    pub alpha: f64,
    pub sigma2: f64,
    pub practical_range: f64,
    pub noise_variance: f64,
    pub interior_margin: f64,
    pub probe_anchor_level: usize,
    pub sensor_points: Vec<Point3>,
    pub spectral_k: usize,
    pub spectral_sweep_level: usize,
    pub spectral_sweep_ks: Vec<usize>,
    pub max_correlation_pairs: usize,
    pub correlation_bin_count: usize,
    pub include_spectral: bool,
    pub require_spectral: bool,
}

impl Default for CubeZeroFormKernelValidationConfig {
    fn default() -> Self {
        Self {
            levels: vec![12, 24, 36, 48],
            alpha: DEFAULT_ALPHA,
            sigma2: DEFAULT_SIGMA2,
            practical_range: DEFAULT_PRACTICAL_RANGE,
            noise_variance: DEFAULT_NOISE_VARIANCE,
            interior_margin: DEFAULT_INTERIOR_MARGIN,
            probe_anchor_level: DEFAULT_PROBE_ANCHOR_LEVEL,
            sensor_points: default_sensor_points(),
            spectral_k: DEFAULT_SPECTRAL_K,
            spectral_sweep_level: 12,
            spectral_sweep_ks: vec![64, 128, 256, 512],
            max_correlation_pairs: DEFAULT_MAX_CORRELATION_PAIRS,
            correlation_bin_count: DEFAULT_CORRELATION_BIN_COUNT,
            include_spectral: true,
            require_spectral: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CubeZeroFormKernelHyperparameters {
    pub dimension: usize,
    pub alpha: f64,
    pub nu: f64,
    pub sigma2: f64,
    pub practical_range: f64,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
}

#[derive(Debug, Clone)]
pub struct CubeZeroFormKernelValidationReport {
    pub config: CubeZeroFormKernelValidationConfig,
    pub hyperparameters: CubeZeroFormKernelHyperparameters,
    pub spectral_available: bool,
    pub levels: Vec<CubeZeroFormKernelLevelReport>,
    pub spectral_sweep: Vec<CubeZeroFormSpectralSweepRow>,
}

#[derive(Debug, Clone)]
pub struct CubeZeroFormKernelLevelReport {
    pub level: usize,
    pub ndofs: usize,
    pub eval_indices: Vec<usize>,
    pub eval_points: Vec<Point3>,
    pub observation_indices: Vec<usize>,
    pub observation_local_indices: Vec<usize>,
    pub diagnostics: CubeZeroFormLevelDiagnostics,
    pub metrics: Vec<CubeZeroFormVarianceMetric>,
    pub reference_variances: Vec<f64>,
    pub gmrf_variances: Vec<f64>,
    pub gmrf_calibrated_variances: Vec<f64>,
    pub spectral_variances: Option<Vec<f64>>,
    pub correlation_pairs: Vec<CubeZeroFormCorrelationPair>,
    pub correlation_bins: Vec<CubeZeroFormCorrelationBin>,
}

#[derive(Debug, Clone)]
pub struct CubeZeroFormLevelDiagnostics {
    pub level: usize,
    pub cell_width: f64,
    pub h_over_range: f64,
    pub margin_over_range: f64,
    pub probe_anchor_level: usize,
    pub gmrf_prior_mean_variance: f64,
    pub gmrf_prior_variance_scale: f64,
    pub gmrf_tau_calibration_multiplier: f64,
    pub gmrf_calibrated_tau: f64,
}

#[derive(Debug, Clone)]
pub struct CubeZeroFormVarianceMetric {
    pub level: usize,
    pub method: String,
    pub spectral_k: Option<usize>,
    pub ndofs: usize,
    pub eval_count: usize,
    pub variance_rmse: f64,
    pub relative_variance_rmse: f64,
    pub max_abs_variance_error: f64,
    pub factor_or_eigen_seconds: f64,
    pub covariance_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct CubeZeroFormCorrelationPair {
    pub i: usize,
    pub j: usize,
    pub distance: f64,
    pub reference_correlation: f64,
    pub gmrf_correlation: f64,
    pub spectral_correlation: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CubeZeroFormCorrelationBin {
    pub bin_index: usize,
    pub distance_min: f64,
    pub distance_max: f64,
    pub count: usize,
    pub reference_mean: f64,
    pub gmrf_mean: f64,
    pub spectral_mean: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CubeZeroFormSpectralSweepRow {
    pub level: usize,
    pub spectral_k: usize,
    pub ndofs: usize,
    pub eval_count: usize,
    pub variance_rmse: f64,
    pub relative_variance_rmse: f64,
    pub max_abs_variance_error: f64,
    pub eigen_seconds: f64,
    pub covariance_seconds: f64,
    pub total_seconds: f64,
}

struct MethodCovarianceResult {
    variances: Vec<f64>,
    covariance: DenseMatrix,
    factor_or_eigen_seconds: f64,
    covariance_seconds: f64,
    total_seconds: f64,
}

struct ReferenceCovarianceResult {
    variances: Vec<f64>,
    covariance: DenseMatrix,
    covariance_seconds: f64,
    total_seconds: f64,
}

pub fn default_sensor_points() -> Vec<Point3> {
    let coords = [0.25, 0.50, 0.75];
    let mut points = Vec::with_capacity(27);
    for &z in &coords {
        for &y in &coords {
            for &x in &coords {
                points.push([x, y, z]);
            }
        }
    }
    points
}

pub fn cube_zero_form_kernel_hyperparameters(
    alpha: f64,
    sigma2: f64,
    practical_range: f64,
    noise_variance: f64,
) -> Result<CubeZeroFormKernelHyperparameters, String> {
    if !alpha.is_finite() || alpha <= DEFAULT_DIMENSION as f64 / 2.0 {
        return Err("alpha must be finite and greater than dimension / 2".to_string());
    }
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return Err("sigma2 must be finite and positive".to_string());
    }
    if !practical_range.is_finite() || practical_range <= 0.0 {
        return Err("practical_range must be finite and positive".to_string());
    }
    if !noise_variance.is_finite() || noise_variance <= 0.0 {
        return Err("noise_variance must be finite and positive".to_string());
    }

    let dimension = DEFAULT_DIMENSION;
    let nu = alpha - dimension as f64 / 2.0;
    let kappa = (8.0 * nu).sqrt() / practical_range;
    let tau_squared = tgamma(nu)
        / (sigma2
            * tgamma(alpha)
            * (4.0 * std::f64::consts::PI).powf(dimension as f64 / 2.0)
            * kappa.powf(2.0 * nu));
    if !tau_squared.is_finite() || tau_squared <= 0.0 {
        return Err("computed tau^2 is not finite and positive".to_string());
    }

    Ok(CubeZeroFormKernelHyperparameters {
        dimension,
        alpha,
        nu,
        sigma2,
        practical_range,
        kappa,
        tau: tau_squared.sqrt(),
        noise_variance,
    })
}

pub fn compute_cube_zero_form_kernel_validation_report(
    config: CubeZeroFormKernelValidationConfig,
) -> Result<CubeZeroFormKernelValidationReport, String> {
    validate_config(&config)?;
    let hyperparameters = cube_zero_form_kernel_hyperparameters(
        config.alpha,
        config.sigma2,
        config.practical_range,
        config.noise_variance,
    )?;
    let spectral_available = petsc_solver_available();
    if config.include_spectral && config.require_spectral && !spectral_available {
        return Err(
            "spectral validation was required, but the PETSc eigen solver binary was not found"
                .to_string(),
        );
    }
    let probe_points = fixed_probe_points(
        config.probe_anchor_level,
        config.interior_margin,
        &config.sensor_points,
    )?;

    let mut levels = Vec::with_capacity(config.levels.len());
    for &level in &config.levels {
        levels.push(compute_level_report(
            level,
            &config,
            hyperparameters,
            spectral_available,
            &probe_points,
        )?);
    }

    let spectral_sweep = if config.include_spectral && spectral_available {
        match compute_spectral_sweep(&config, hyperparameters, &probe_points) {
            Ok(rows) => rows,
            Err(err) if config.require_spectral => return Err(err),
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };

    Ok(CubeZeroFormKernelValidationReport {
        config,
        hyperparameters,
        spectral_available,
        levels,
        spectral_sweep,
    })
}

pub fn write_cube_zero_form_kernel_validation_outputs(
    report: &CubeZeroFormKernelValidationReport,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_hyperparameters_csv(report, &out_dir.join("hyperparameters.csv"))?;
    write_calibration_diagnostics_csv(report, &out_dir.join("calibration_diagnostics.csv"))?;
    write_variance_rmse_csv(report, &out_dir.join("variance_rmse.csv"))?;
    write_spectral_sweep_csv(report, &out_dir.join("spectral_k_sweep.csv"))?;
    for level in &report.levels {
        write_level_variances_csv(
            level,
            &out_dir.join(format!("interior_variances_level_{}.csv", level.level)),
        )?;
        write_level_correlation_pairs_csv(
            level,
            &out_dir.join(format!("correlation_pairs_level_{}.csv", level.level)),
        )?;
        write_level_correlation_bins_csv(
            level,
            &out_dir.join(format!("correlation_bins_level_{}.csv", level.level)),
        )?;
    }
    write_figures(report, &out_dir.join("figures"))?;
    Ok(())
}

pub fn interior_vertex_indices(coords: &MeshCoords, margin: f64) -> Result<Vec<usize>, String> {
    if coords.dim() != DEFAULT_DIMENSION {
        return Err(format!(
            "expected {DEFAULT_DIMENSION}D cube coordinates, got {}D",
            coords.dim()
        ));
    }
    if !margin.is_finite() || !(0.0..0.5).contains(&margin) {
        return Err("interior margin must be finite and in [0, 0.5)".to_string());
    }

    Ok(coords
        .coord_iter()
        .enumerate()
        .filter_map(|(idx, coord)| {
            let inside = (0..DEFAULT_DIMENSION)
                .all(|axis| coord[axis] >= margin && coord[axis] <= 1.0 - margin);
            inside.then_some(idx)
        })
        .collect())
}

pub fn deterministic_sensor_indices(
    coords: &MeshCoords,
    sensor_points: &[Point3],
    margin: f64,
) -> Result<Vec<usize>, String> {
    if sensor_points.is_empty() {
        return Err("at least one sensor point is required".to_string());
    }
    let interior = interior_vertex_indices(coords, margin)?
        .into_iter()
        .collect::<HashSet<_>>();
    let mut selected = Vec::with_capacity(sensor_points.len());
    for point in sensor_points {
        if point
            .iter()
            .any(|value| !value.is_finite() || *value < margin || *value > 1.0 - margin)
        {
            return Err("sensor points must be finite and inside the interior margin".to_string());
        }
        let best = coords
            .coord_iter()
            .enumerate()
            .filter(|(idx, _)| interior.contains(idx))
            .min_by(|(_, lhs), (_, rhs)| {
                squared_distance_to_coord(point, lhs)
                    .partial_cmp(&squared_distance_to_coord(point, rhs))
                    .expect("finite sensor distances should compare")
            })
            .map(|(idx, _)| idx)
            .ok_or_else(|| "no interior vertices are available for sensor selection".to_string())?;
        selected.push(best);
    }
    let unique = selected.iter().copied().collect::<HashSet<_>>();
    if unique.len() != selected.len() {
        return Err("sensor selection produced duplicate vertices".to_string());
    }
    Ok(selected)
}

pub fn fixed_probe_points(
    anchor_level: usize,
    margin: f64,
    sensor_points: &[Point3],
) -> Result<Vec<Point3>, String> {
    if anchor_level == 0 {
        return Err("probe anchor level must be positive".to_string());
    }
    let mesh = CartesianMeshInfo::new_unit_scaled(DEFAULT_DIMENSION, anchor_level, 1.0);
    let (_topology, coords) = mesh.compute_coord_complex();
    let probe_indices = interior_vertex_indices(&coords, margin)?;
    if probe_indices.is_empty() {
        return Err(format!(
            "probe anchor level {anchor_level} has no vertices at margin {margin}"
        ));
    }
    let probe_points = probe_indices
        .iter()
        .map(|&idx| coord_to_point(coords.coord(idx)))
        .collect::<Vec<_>>();
    validate_sensor_points_in_probe_set(&probe_points, sensor_points)?;
    Ok(probe_points)
}

pub fn fixed_probe_indices_for_mesh(
    coords: &MeshCoords,
    level: usize,
    probe_points: &[Point3],
) -> Result<Vec<usize>, String> {
    if coords.dim() != DEFAULT_DIMENSION {
        return Err(format!(
            "expected {DEFAULT_DIMENSION}D cube coordinates, got {}D",
            coords.dim()
        ));
    }
    if level == 0 {
        return Err("cube level must be positive".to_string());
    }
    if probe_points.is_empty() {
        return Err("fixed probe set must not be empty".to_string());
    }

    let mut vertex_by_key = HashMap::with_capacity(coords.nvertices());
    for (idx, coord) in coords.coord_iter().enumerate() {
        let key = coord_grid_key(&coord, level)?;
        if vertex_by_key.insert(key, idx).is_some() {
            return Err("cube mesh produced duplicate vertex grid coordinates".to_string());
        }
    }

    let mut mapped = Vec::with_capacity(probe_points.len());
    for point in probe_points {
        let key = point_grid_key(point, level)?;
        let idx = vertex_by_key.get(&key).copied().ok_or_else(|| {
            format!(
                "fixed probe point {:?} is not a vertex on level {level}",
                point
            )
        })?;
        mapped.push(idx);
    }
    let unique = mapped.iter().copied().collect::<HashSet<_>>();
    if unique.len() != mapped.len() {
        return Err("fixed probe mapping produced duplicate vertices".to_string());
    }
    Ok(mapped)
}

pub fn rmse(values: &[f64], reference: &[f64]) -> Result<f64, String> {
    if values.len() != reference.len() {
        return Err(format!(
            "rmse expected equal lengths, got {} and {}",
            values.len(),
            reference.len()
        ));
    }
    if values.is_empty() {
        return Err("rmse requires at least one value".to_string());
    }
    let mut sum = 0.0;
    for (&value, &base) in values.iter().zip(reference) {
        if !value.is_finite() || !base.is_finite() {
            return Err("rmse inputs must be finite".to_string());
        }
        let delta = value - base;
        sum += delta * delta;
    }
    Ok((sum / values.len() as f64).sqrt())
}

pub fn correlation_bins(
    distances: &[f64],
    reference: &[f64],
    gmrf: &[f64],
    spectral: Option<&[f64]>,
    bin_count: usize,
) -> Result<Vec<CubeZeroFormCorrelationBin>, String> {
    if distances.len() != reference.len() || distances.len() != gmrf.len() {
        return Err("correlation bin inputs must have matching lengths".to_string());
    }
    if let Some(spectral) = spectral {
        if spectral.len() != distances.len() {
            return Err("spectral correlation input length does not match distances".to_string());
        }
    }
    if distances.is_empty() {
        return Err("at least one correlation pair is required".to_string());
    }
    if bin_count == 0 {
        return Err("bin_count must be positive".to_string());
    }

    let max_distance = distances
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    if max_distance <= 0.0 {
        return Err("correlation distances must contain a positive entry".to_string());
    }

    let mut counts = vec![0usize; bin_count];
    let mut reference_sums = vec![0.0; bin_count];
    let mut gmrf_sums = vec![0.0; bin_count];
    let mut spectral_sums = spectral.map(|_| vec![0.0; bin_count]);
    for idx in 0..distances.len() {
        let distance = distances[idx];
        if !distance.is_finite() || distance < 0.0 {
            return Err("correlation distances must be finite and nonnegative".to_string());
        }
        let bin =
            (((distance / max_distance) * bin_count as f64).floor() as usize).min(bin_count - 1);
        counts[bin] += 1;
        reference_sums[bin] += reference[idx];
        gmrf_sums[bin] += gmrf[idx];
        if let (Some(values), Some(sums)) = (spectral, spectral_sums.as_mut()) {
            sums[bin] += values[idx];
        }
    }

    let width = max_distance / bin_count as f64;
    let mut bins = Vec::new();
    for bin in 0..bin_count {
        let count = counts[bin];
        if count == 0 {
            continue;
        }
        let denom = count as f64;
        bins.push(CubeZeroFormCorrelationBin {
            bin_index: bin,
            distance_min: bin as f64 * width,
            distance_max: (bin + 1) as f64 * width,
            count,
            reference_mean: reference_sums[bin] / denom,
            gmrf_mean: gmrf_sums[bin] / denom,
            spectral_mean: spectral_sums.as_ref().map(|sums| sums[bin] / denom),
        });
    }
    Ok(bins)
}

pub fn petsc_solver_available() -> bool {
    if let Ok(path) = std::env::var("PETSC_SOLVER_PATH") {
        if !path.is_empty() {
            let candidate = PathBuf::from(path).join("ghiep.out");
            return candidate.exists();
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .map(|ancestor| ancestor.join("feec/petsc-solver/ghiep.out"))
        .any(|candidate| candidate.exists())
}

fn validate_config(config: &CubeZeroFormKernelValidationConfig) -> Result<(), String> {
    if config.levels.is_empty() {
        return Err("at least one cube level is required".to_string());
    }
    if config.levels.contains(&0) {
        return Err("cube levels must be positive".to_string());
    }
    if config.probe_anchor_level == 0 {
        return Err("probe_anchor_level must be positive".to_string());
    }
    if config.spectral_k == 0 {
        return Err("spectral_k must be positive".to_string());
    }
    if config.max_correlation_pairs == 0 {
        return Err("max_correlation_pairs must be positive".to_string());
    }
    if config.correlation_bin_count == 0 {
        return Err("correlation_bin_count must be positive".to_string());
    }
    if config.sensor_points.is_empty() {
        return Err("at least one sensor point is required".to_string());
    }
    cube_zero_form_kernel_hyperparameters(
        config.alpha,
        config.sigma2,
        config.practical_range,
        config.noise_variance,
    )?;
    if !config.interior_margin.is_finite() || !(0.0..0.5).contains(&config.interior_margin) {
        return Err("interior_margin must be finite and in [0, 0.5)".to_string());
    }
    Ok(())
}

fn compute_level_report(
    level: usize,
    config: &CubeZeroFormKernelValidationConfig,
    hyperparameters: CubeZeroFormKernelHyperparameters,
    spectral_available: bool,
    probe_points: &[Point3],
) -> Result<CubeZeroFormKernelLevelReport, String> {
    let mesh = CartesianMeshInfo::new_unit_scaled(DEFAULT_DIMENSION, level, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let ndofs = coords.nvertices();

    let eval_indices = fixed_probe_indices_for_mesh(&coords, level, probe_points)?;
    if eval_indices.is_empty() {
        return Err(format!(
            "level {level} has no fixed probe vertices at margin {}",
            config.interior_margin
        ));
    }
    let eval_points = eval_indices
        .iter()
        .map(|&idx| coord_to_point(coords.coord(idx)))
        .collect::<Vec<_>>();
    let observation_indices =
        deterministic_sensor_indices(&coords, &config.sensor_points, config.interior_margin)?;
    let observation_local_indices = observation_local_indices(&eval_indices, &observation_indices)?;

    let reference =
        compute_reference_covariance(&eval_points, &observation_local_indices, hyperparameters)?;
    let pair_specs = select_correlation_pair_specs(&eval_points, config.max_correlation_pairs)?;
    let reference_correlations = pair_correlations(&reference.covariance, &pair_specs)?;

    let gmrf_prior_mean_variance = compute_gmrf_prior_mean_variance(
        &topology,
        &metric,
        ndofs,
        &eval_indices,
        hyperparameters,
        1.0,
    )?;
    let gmrf_prior_variance_scale =
        gmrf_prior_mean_variance / hyperparameters.sigma2.max(f64::EPSILON);
    let gmrf_tau_calibration_multiplier = gmrf_prior_variance_scale.max(f64::EPSILON).sqrt();
    let diagnostics = CubeZeroFormLevelDiagnostics {
        level,
        cell_width: 1.0 / level as f64,
        h_over_range: (1.0 / level as f64) / hyperparameters.practical_range,
        margin_over_range: config.interior_margin / hyperparameters.practical_range,
        probe_anchor_level: config.probe_anchor_level,
        gmrf_prior_mean_variance,
        gmrf_prior_variance_scale,
        gmrf_tau_calibration_multiplier,
        gmrf_calibrated_tau: hyperparameters.tau * gmrf_tau_calibration_multiplier,
    };

    let gmrf = compute_gmrf_covariance(
        &topology,
        &metric,
        ndofs,
        &eval_indices,
        &observation_indices,
        hyperparameters,
        1.0,
    )?;
    let gmrf_correlations = pair_correlations(&gmrf.covariance, &pair_specs)?;

    let gmrf_calibrated = compute_gmrf_covariance(
        &topology,
        &metric,
        ndofs,
        &eval_indices,
        &observation_indices,
        hyperparameters,
        gmrf_tau_calibration_multiplier,
    )?;

    let spectral = if config.include_spectral && spectral_available {
        match compute_spectral_covariance_checked(
            &topology,
            &metric,
            &eval_indices,
            &observation_local_indices,
            hyperparameters,
            config.spectral_k.min(ndofs).max(1),
        ) {
            Ok(covariance) => Some(covariance),
            Err(err) if config.require_spectral => return Err(err),
            Err(_) => None,
        }
    } else {
        None
    };
    let spectral_correlations = spectral
        .as_ref()
        .map(|spectral| pair_correlations(&spectral.covariance, &pair_specs))
        .transpose()?;

    let mut metrics = Vec::new();
    metrics.push(CubeZeroFormVarianceMetric {
        level,
        method: "euclidean_reference".to_string(),
        spectral_k: None,
        ndofs,
        eval_count: eval_indices.len(),
        variance_rmse: 0.0,
        relative_variance_rmse: 0.0,
        max_abs_variance_error: 0.0,
        factor_or_eigen_seconds: 0.0,
        covariance_seconds: reference.covariance_seconds,
        total_seconds: reference.total_seconds,
    });
    metrics.push(metric_row(
        level,
        "gmrf",
        None,
        ndofs,
        &gmrf.variances,
        &reference.variances,
        gmrf.factor_or_eigen_seconds,
        gmrf.covariance_seconds,
        gmrf.total_seconds,
    )?);
    metrics.push(metric_row(
        level,
        "gmrf_calibrated",
        None,
        ndofs,
        &gmrf_calibrated.variances,
        &reference.variances,
        gmrf_calibrated.factor_or_eigen_seconds,
        gmrf_calibrated.covariance_seconds,
        gmrf_calibrated.total_seconds,
    )?);
    if let Some(spectral) = &spectral {
        metrics.push(metric_row(
            level,
            "spectral",
            Some(config.spectral_k.min(ndofs).max(1)),
            ndofs,
            &spectral.variances,
            &reference.variances,
            spectral.factor_or_eigen_seconds,
            spectral.covariance_seconds,
            spectral.total_seconds,
        )?);
    }

    let correlation_pairs = pair_specs
        .iter()
        .zip(reference_correlations.iter().copied())
        .zip(gmrf_correlations.iter().copied())
        .enumerate()
        .map(
            |(pair_index, ((spec, reference_correlation), gmrf_correlation))| {
                CubeZeroFormCorrelationPair {
                    i: spec.i,
                    j: spec.j,
                    distance: spec.distance,
                    reference_correlation,
                    gmrf_correlation,
                    spectral_correlation: spectral_correlations
                        .as_ref()
                        .map(|values| values[pair_index]),
                }
            },
        )
        .collect::<Vec<_>>();

    let distances = pair_specs
        .iter()
        .map(|pair| pair.distance)
        .collect::<Vec<_>>();
    let correlation_bins = correlation_bins(
        &distances,
        &reference_correlations,
        &gmrf_correlations,
        spectral_correlations.as_deref(),
        config.correlation_bin_count,
    )?;

    Ok(CubeZeroFormKernelLevelReport {
        level,
        ndofs,
        eval_indices,
        eval_points,
        observation_indices,
        observation_local_indices,
        diagnostics,
        metrics,
        reference_variances: reference.variances,
        gmrf_variances: gmrf.variances,
        gmrf_calibrated_variances: gmrf_calibrated.variances,
        spectral_variances: spectral.map(|spectral| spectral.variances),
        correlation_pairs,
        correlation_bins,
    })
}

fn compute_reference_covariance(
    eval_points: &[Point3],
    observation_local_indices: &[usize],
    hyperparameters: CubeZeroFormKernelHyperparameters,
) -> Result<ReferenceCovarianceResult, String> {
    let total_start = Instant::now();
    let covariance_start = Instant::now();
    let points = eval_points
        .iter()
        .map(|point| point.to_vec())
        .collect::<Vec<_>>();
    let prior_covariance = matern_covariance_matrix_euclidean(
        &points,
        EuclideanMaternConfig {
            kappa: hyperparameters.kappa,
            nu: hyperparameters.nu,
            variance: hyperparameters.sigma2,
        },
    )
    .map_err(|err| err.to_string())?;
    let observations = vec![0.0; observation_local_indices.len()];
    let conditioned = condition_full_covariance_with_covariance(
        &prior_covariance,
        observation_local_indices,
        &observations,
        hyperparameters.noise_variance,
    )
    .map_err(|err| err.to_string())?;
    let covariance_seconds = covariance_start.elapsed().as_secs_f64();
    Ok(ReferenceCovarianceResult {
        variances: diagonal_variance(&conditioned.covariance)?,
        covariance: conditioned.covariance,
        covariance_seconds,
        total_seconds: total_start.elapsed().as_secs_f64(),
    })
}

fn compute_gmrf_covariance(
    topology: &Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    ndofs: usize,
    eval_indices: &[usize],
    observation_indices: &[usize],
    hyperparameters: CubeZeroFormKernelHyperparameters,
    tau_multiplier: f64,
) -> Result<MethodCovarianceResult, String> {
    if !tau_multiplier.is_finite() || tau_multiplier <= 0.0 {
        return Err("tau multiplier must be finite and positive".to_string());
    }
    let total_start = Instant::now();
    let laplace = build_laplace_beltrami_0form(topology, metric);
    let prior_precision = build_matern_precision_0form(
        &laplace,
        MaternConfig {
            kappa: hyperparameters.kappa,
            tau: hyperparameters.tau * tau_multiplier,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );
    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let observation_matrix = observation_selector(ndofs, observation_indices);
    let observations = GmrfVector::zeros(observation_indices.len());
    let (q_post, _information) = apply_gaussian_observations(
        &q_prior,
        &observation_matrix,
        &observations,
        None,
        hyperparameters.noise_variance,
    );

    let factor_start = Instant::now();
    let factor = q_post
        .cholesky_sqrt_lower()
        .map_err(|err| err.to_string())?;
    let factor_seconds = factor_start.elapsed().as_secs_f64();

    let covariance_start = Instant::now();
    let mut posterior = Gmrf::from_mean_and_precision(GmrfVector::zeros(ndofs), q_post)
        .map_err(|err| err.to_string())?
        .with_precision_sqrt(factor);
    let operator = selector_operator(ndofs, eval_indices)?;
    let covariance = posterior
        .exact_transformed_covariance(&operator)
        .map_err(|err| err.to_string())?;
    let covariance_seconds = covariance_start.elapsed().as_secs_f64();
    Ok(MethodCovarianceResult {
        variances: diagonal_variance(&covariance)?,
        covariance,
        factor_or_eigen_seconds: factor_seconds,
        covariance_seconds,
        total_seconds: total_start.elapsed().as_secs_f64(),
    })
}

fn compute_gmrf_prior_mean_variance(
    topology: &Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    ndofs: usize,
    eval_indices: &[usize],
    hyperparameters: CubeZeroFormKernelHyperparameters,
    tau_multiplier: f64,
) -> Result<f64, String> {
    if !tau_multiplier.is_finite() || tau_multiplier <= 0.0 {
        return Err("tau multiplier must be finite and positive".to_string());
    }
    let laplace = build_laplace_beltrami_0form(topology, metric);
    let prior_precision = build_matern_precision_0form(
        &laplace,
        MaternConfig {
            kappa: hyperparameters.kappa,
            tau: hyperparameters.tau * tau_multiplier,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );
    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let factor = q_prior
        .cholesky_sqrt_lower()
        .map_err(|err| err.to_string())?;
    let mut prior = Gmrf::from_mean_and_precision(GmrfVector::zeros(ndofs), q_prior)
        .map_err(|err| err.to_string())?
        .with_precision_sqrt(factor);
    let operator = selector_operator(ndofs, eval_indices)?;
    let covariance = prior
        .exact_transformed_covariance(&operator)
        .map_err(|err| err.to_string())?;
    Ok(mean(&diagonal_variance(&covariance)?))
}

fn compute_spectral_covariance_checked(
    topology: &Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    eval_indices: &[usize],
    observation_local_indices: &[usize],
    hyperparameters: CubeZeroFormKernelHyperparameters,
    spectral_k: usize,
) -> Result<MethodCovarianceResult, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_spectral_covariance(
            topology,
            metric,
            eval_indices,
            observation_local_indices,
            hyperparameters,
            spectral_k,
        )
    })) {
        Ok(result) => result,
        Err(payload) => Err(format!(
            "spectral covariance computation panicked: {}",
            panic_payload_message(payload)
        )),
    }
}

fn compute_spectral_covariance(
    topology: &Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    eval_indices: &[usize],
    observation_local_indices: &[usize],
    hyperparameters: CubeZeroFormKernelHyperparameters,
    spectral_k: usize,
) -> Result<MethodCovarianceResult, String> {
    let total_start = Instant::now();
    let eigen_start = Instant::now();
    let spectral_gp = SpectralMaternGp::from_hodge_laplace(
        topology,
        metric,
        0,
        SpectralMaternConfig {
            kappa: hyperparameters.kappa,
            alpha: hyperparameters.alpha,
            tau: hyperparameters.tau,
            k: spectral_k,
        },
    )
    .map_err(|err| err.to_string())?;
    let eigen_seconds = eigen_start.elapsed().as_secs_f64();

    let covariance_start = Instant::now();
    let spectral_covariance = selected_spectral_covariance(&spectral_gp, eval_indices)?;
    let observations = vec![0.0; observation_local_indices.len()];
    let conditioned = condition_full_covariance_with_covariance(
        &spectral_covariance,
        observation_local_indices,
        &observations,
        hyperparameters.noise_variance,
    )
    .map_err(|err| err.to_string())?;
    let covariance = conditioned.covariance;
    let covariance_seconds = covariance_start.elapsed().as_secs_f64();
    Ok(MethodCovarianceResult {
        variances: diagonal_variance(&covariance)?,
        covariance,
        factor_or_eigen_seconds: eigen_seconds,
        covariance_seconds,
        total_seconds: total_start.elapsed().as_secs_f64(),
    })
}

fn compute_spectral_sweep(
    config: &CubeZeroFormKernelValidationConfig,
    hyperparameters: CubeZeroFormKernelHyperparameters,
    probe_points: &[Point3],
) -> Result<Vec<CubeZeroFormSpectralSweepRow>, String> {
    let level = config.spectral_sweep_level;
    let mesh = CartesianMeshInfo::new_unit_scaled(DEFAULT_DIMENSION, level, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let ndofs = coords.nvertices();
    let eval_indices = fixed_probe_indices_for_mesh(&coords, level, probe_points)?;
    let eval_points = eval_indices
        .iter()
        .map(|&idx| coord_to_point(coords.coord(idx)))
        .collect::<Vec<_>>();
    let observation_indices =
        deterministic_sensor_indices(&coords, &config.sensor_points, config.interior_margin)?;
    let observation_local_indices = observation_local_indices(&eval_indices, &observation_indices)?;
    let reference =
        compute_reference_covariance(&eval_points, &observation_local_indices, hyperparameters)?;

    let mut rows = Vec::new();
    for &requested_k in &config.spectral_sweep_ks {
        if requested_k == 0 {
            continue;
        }
        let spectral_k = requested_k.min(ndofs).max(1);
        let spectral = compute_spectral_covariance_checked(
            &topology,
            &metric,
            &eval_indices,
            &observation_local_indices,
            hyperparameters,
            spectral_k,
        )?;
        let variance_rmse = rmse(&spectral.variances, &reference.variances)?;
        rows.push(CubeZeroFormSpectralSweepRow {
            level,
            spectral_k,
            ndofs,
            eval_count: eval_indices.len(),
            variance_rmse,
            relative_variance_rmse: variance_rmse / mean(&reference.variances).max(f64::EPSILON),
            max_abs_variance_error: max_abs_error(&spectral.variances, &reference.variances)?,
            eigen_seconds: spectral.factor_or_eigen_seconds,
            covariance_seconds: spectral.covariance_seconds,
            total_seconds: spectral.total_seconds,
        });
    }
    Ok(rows)
}

// This is a report-row constructor: method identity, dimensions, reference values,
// and three timings are retained independently in the reproducibility artifact.
#[allow(clippy::too_many_arguments)]
fn metric_row(
    level: usize,
    method: &str,
    spectral_k: Option<usize>,
    ndofs: usize,
    values: &[f64],
    reference: &[f64],
    factor_or_eigen_seconds: f64,
    covariance_seconds: f64,
    total_seconds: f64,
) -> Result<CubeZeroFormVarianceMetric, String> {
    let variance_rmse = rmse(values, reference)?;
    Ok(CubeZeroFormVarianceMetric {
        level,
        method: method.to_string(),
        spectral_k,
        ndofs,
        eval_count: reference.len(),
        variance_rmse,
        relative_variance_rmse: variance_rmse / mean(reference).max(f64::EPSILON),
        max_abs_variance_error: max_abs_error(values, reference)?,
        factor_or_eigen_seconds,
        covariance_seconds,
        total_seconds,
    })
}

#[derive(Debug, Clone, Copy)]
struct CorrelationPairSpec {
    i: usize,
    j: usize,
    distance: f64,
}

fn select_correlation_pair_specs(
    eval_points: &[Point3],
    max_pairs: usize,
) -> Result<Vec<CorrelationPairSpec>, String> {
    if eval_points.len() < 2 {
        return Err("at least two evaluation points are required for correlations".to_string());
    }
    let total_pairs = eval_points.len() * (eval_points.len() - 1) / 2;
    let step = total_pairs.div_ceil(max_pairs).max(1);
    let mut pairs = Vec::new();
    let mut ordinal = 0usize;
    for i in 0..eval_points.len() {
        for j in (i + 1)..eval_points.len() {
            if ordinal % step == 0 {
                pairs.push(CorrelationPairSpec {
                    i,
                    j,
                    distance: euclidean_distance(&eval_points[i], &eval_points[j]),
                });
            }
            ordinal += 1;
        }
    }
    Ok(pairs)
}

fn pair_correlations(
    covariance: &DenseMatrix,
    pairs: &[CorrelationPairSpec],
) -> Result<Vec<f64>, String> {
    let variances = diagonal_variance(covariance)?;
    pairs
        .iter()
        .map(|pair| {
            let cov = covariance[(pair.i, pair.j)];
            covariance_to_correlation(cov, variances[pair.i], variances[pair.j])
        })
        .collect()
}

fn covariance_to_correlation(covariance: f64, var_i: f64, var_j: f64) -> Result<f64, String> {
    if !covariance.is_finite() || !var_i.is_finite() || !var_j.is_finite() {
        return Err("correlation inputs must be finite".to_string());
    }
    let denom = (var_i.max(0.0) * var_j.max(0.0)).sqrt();
    if denom <= f64::EPSILON {
        return Ok(0.0);
    }
    Ok((covariance / denom).clamp(-1.0, 1.0))
}

fn observation_local_indices(
    eval_indices: &[usize],
    observation_indices: &[usize],
) -> Result<Vec<usize>, String> {
    observation_indices
        .iter()
        .map(|obs| {
            eval_indices
                .iter()
                .position(|idx| idx == obs)
                .ok_or_else(|| {
                    format!("observation vertex {obs} is not in the interior evaluation set")
                })
        })
        .collect()
}

fn selector_operator(ncols: usize, indices: &[usize]) -> Result<SparseRowOperator, String> {
    let rows = indices
        .iter()
        .map(|&idx| vec![(idx, 1.0)])
        .collect::<Vec<_>>();
    SparseRowOperator::new(ncols, rows).map_err(|err| err.to_string())
}

fn selected_spectral_covariance(
    spectral_gp: &SpectralMaternGp,
    indices: &[usize],
) -> Result<Mat<f64>, String> {
    let ambient_dimension = spectral_gp.eigenvectors().nrows();
    if let Some(&idx) = indices.iter().find(|&&idx| idx >= ambient_dimension) {
        return Err(format!(
            "spectral covariance index {idx} is out of bounds for dimension {ambient_dimension}"
        ));
    }
    let n = indices.len();
    let mut covariance = Mat::zeros(n, n);
    for row in 0..n {
        for col in row..n {
            let value = spectral_gp.covariance_entry(indices[row], indices[col]);
            covariance[(row, col)] = value;
            covariance[(col, row)] = value;
        }
    }
    Ok(covariance)
}

fn diagonal_variance(covariance: &DenseMatrix) -> Result<Vec<f64>, String> {
    if covariance.nrows() != covariance.ncols() {
        return Err(format!(
            "covariance must be square, got {}x{}",
            covariance.nrows(),
            covariance.ncols()
        ));
    }
    Ok((0..covariance.nrows())
        .map(|idx| covariance[(idx, idx)].max(0.0))
        .collect())
}

fn max_abs_error(values: &[f64], reference: &[f64]) -> Result<f64, String> {
    if values.len() != reference.len() {
        return Err("max_abs_error inputs must have matching lengths".to_string());
    }
    Ok(values
        .iter()
        .zip(reference)
        .map(|(&value, &base)| (value - base).abs())
        .fold(0.0, f64::max))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn coord_to_point<C>(coord: C) -> Point3
where
    C: Index<usize, Output = f64>,
{
    [coord[0], coord[1], coord[2]]
}

fn validate_sensor_points_in_probe_set(
    probe_points: &[Point3],
    sensor_points: &[Point3],
) -> Result<(), String> {
    for sensor in sensor_points {
        if !probe_points
            .iter()
            .any(|probe| euclidean_distance(probe, sensor) <= GRID_ALIGNMENT_TOLERANCE)
        {
            return Err(format!(
                "sensor point {:?} is not included in the fixed probe set",
                sensor
            ));
        }
    }
    Ok(())
}

fn coord_grid_key<C>(coord: &C, level: usize) -> Result<[i64; DEFAULT_DIMENSION], String>
where
    C: Index<usize, Output = f64> + ?Sized,
{
    point_grid_key(&[coord[0], coord[1], coord[2]], level)
}

fn point_grid_key(point: &Point3, level: usize) -> Result<[i64; DEFAULT_DIMENSION], String> {
    let mut key = [0_i64; DEFAULT_DIMENSION];
    for axis in 0..DEFAULT_DIMENSION {
        if !point[axis].is_finite() {
            return Err("grid point coordinates must be finite".to_string());
        }
        let scaled = point[axis] * level as f64;
        let rounded = scaled.round();
        if (scaled - rounded).abs() > GRID_ALIGNMENT_TOLERANCE {
            return Err(format!(
                "fixed probe coordinate {} is not aligned with level {}",
                point[axis], level
            ));
        }
        key[axis] = rounded as i64;
    }
    Ok(key)
}

fn squared_distance_to_coord<C>(point: &Point3, coord: &C) -> f64
where
    C: Index<usize, Output = f64> + ?Sized,
{
    (0..DEFAULT_DIMENSION)
        .map(|axis| {
            let delta = point[axis] - coord[axis];
            delta * delta
        })
        .sum()
}

fn euclidean_distance(lhs: &Point3, rhs: &Point3) -> f64 {
    (0..DEFAULT_DIMENSION)
        .map(|axis| {
            let delta = lhs[axis] - rhs[axis];
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn write_hyperparameters_csv(
    report: &CubeZeroFormKernelValidationReport,
    path: &Path,
) -> io::Result<()> {
    let hp = report.hyperparameters;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "dimension,alpha,nu,sigma2,practical_range,kappa,tau,noise_variance,interior_margin,probe_anchor_level,spectral_available"
    )?;
    writeln!(
        writer,
        "{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{},{}",
        hp.dimension,
        hp.alpha,
        hp.nu,
        hp.sigma2,
        hp.practical_range,
        hp.kappa,
        hp.tau,
        hp.noise_variance,
        report.config.interior_margin,
        report.config.probe_anchor_level,
        report.spectral_available
    )?;
    Ok(())
}

fn write_variance_rmse_csv(
    report: &CubeZeroFormKernelValidationReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,method,spectral_k,ndofs,eval_count,variance_rmse,relative_variance_rmse,max_abs_variance_error,factor_or_eigen_seconds,covariance_seconds,total_seconds"
    )?;
    for level in &report.levels {
        for metric in &level.metrics {
            writeln!(
                writer,
                "{},{},{},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
                metric.level,
                metric.method,
                optional_usize_csv(metric.spectral_k),
                metric.ndofs,
                metric.eval_count,
                metric.variance_rmse,
                metric.relative_variance_rmse,
                metric.max_abs_variance_error,
                metric.factor_or_eigen_seconds,
                metric.covariance_seconds,
                metric.total_seconds
            )?;
        }
    }
    Ok(())
}

fn write_calibration_diagnostics_csv(
    report: &CubeZeroFormKernelValidationReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,probe_anchor_level,cell_width,h_over_range,margin_over_range,gmrf_prior_mean_variance,gmrf_prior_variance_scale,gmrf_tau_calibration_multiplier,gmrf_calibrated_tau"
    )?;
    for level in &report.levels {
        let diag = &level.diagnostics;
        writeln!(
            writer,
            "{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            diag.level,
            diag.probe_anchor_level,
            diag.cell_width,
            diag.h_over_range,
            diag.margin_over_range,
            diag.gmrf_prior_mean_variance,
            diag.gmrf_prior_variance_scale,
            diag.gmrf_tau_calibration_multiplier,
            diag.gmrf_calibrated_tau
        )?;
    }
    Ok(())
}

fn write_spectral_sweep_csv(
    report: &CubeZeroFormKernelValidationReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,spectral_k,ndofs,eval_count,variance_rmse,relative_variance_rmse,max_abs_variance_error,eigen_seconds,covariance_seconds,total_seconds"
    )?;
    for row in &report.spectral_sweep {
        writeln!(
            writer,
            "{},{},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            row.level,
            row.spectral_k,
            row.ndofs,
            row.eval_count,
            row.variance_rmse,
            row.relative_variance_rmse,
            row.max_abs_variance_error,
            row.eigen_seconds,
            row.covariance_seconds,
            row.total_seconds
        )?;
    }
    Ok(())
}

fn write_level_variances_csv(level: &CubeZeroFormKernelLevelReport, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "vertex_index,x,y,z,euclidean_variance,gmrf_variance,gmrf_error,gmrf_calibrated_variance,gmrf_calibrated_error,spectral_variance,spectral_error,is_observation"
    )?;
    let observation_set = level
        .observation_indices
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    for row in 0..level.eval_indices.len() {
        let spectral_variance = level.spectral_variances.as_ref().map(|values| values[row]);
        let spectral_error = spectral_variance.map(|value| value - level.reference_variances[row]);
        let point = level.eval_points[row];
        writeln!(
            writer,
            "{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{},{},{}",
            level.eval_indices[row],
            point[0],
            point[1],
            point[2],
            level.reference_variances[row],
            level.gmrf_variances[row],
            level.gmrf_variances[row] - level.reference_variances[row],
            level.gmrf_calibrated_variances[row],
            level.gmrf_calibrated_variances[row] - level.reference_variances[row],
            optional_f64_csv(spectral_variance),
            optional_f64_csv(spectral_error),
            observation_set.contains(&level.eval_indices[row])
        )?;
    }
    Ok(())
}

fn write_level_correlation_pairs_csv(
    level: &CubeZeroFormKernelLevelReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "i,j,distance,euclidean_reference_correlation,gmrf_correlation,spectral_correlation"
    )?;
    for pair in &level.correlation_pairs {
        writeln!(
            writer,
            "{},{},{:.16},{:.16},{:.16},{}",
            pair.i,
            pair.j,
            pair.distance,
            pair.reference_correlation,
            pair.gmrf_correlation,
            optional_f64_csv(pair.spectral_correlation)
        )?;
    }
    Ok(())
}

fn write_level_correlation_bins_csv(
    level: &CubeZeroFormKernelLevelReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "bin_index,distance_min,distance_max,count,euclidean_reference_mean,gmrf_mean,spectral_mean"
    )?;
    for bin in &level.correlation_bins {
        writeln!(
            writer,
            "{},{:.16},{:.16},{},{:.16},{:.16},{}",
            bin.bin_index,
            bin.distance_min,
            bin.distance_max,
            bin.count,
            bin.reference_mean,
            bin.gmrf_mean,
            optional_f64_csv(bin.spectral_mean)
        )?;
    }
    Ok(())
}

fn write_figures(
    report: &CubeZeroFormKernelValidationReport,
    figures_dir: &Path,
) -> io::Result<()> {
    fs::create_dir_all(figures_dir)?;
    for level in &report.levels {
        write_gmrf_correlation_svg(
            level,
            &figures_dir.join(format!("correlation_gmrf_level_{}.svg", level.level)),
        )?;
        if level.spectral_variances.is_some() {
            write_spectral_correlation_svg(
                level,
                &figures_dir.join(format!("correlation_spectral_level_{}.svg", level.level)),
            )?;
        }
        write_variance_scatter_svg(
            level,
            &figures_dir.join(format!("variance_scatter_level_{}.svg", level.level)),
        )?;
    }
    if !report.spectral_sweep.is_empty() {
        write_spectral_sweep_svg(
            &report.spectral_sweep,
            &figures_dir.join("spectral_k_sweep.svg"),
        )?;
    }
    Ok(())
}

fn write_gmrf_correlation_svg(
    level: &CubeZeroFormKernelLevelReport,
    path: &Path,
) -> io::Result<()> {
    let scatter = level
        .correlation_pairs
        .iter()
        .map(|pair| (pair.distance, pair.gmrf_correlation))
        .collect::<Vec<_>>();
    let reference_line = level
        .correlation_bins
        .iter()
        .map(|bin| {
            (
                0.5 * (bin.distance_min + bin.distance_max),
                bin.reference_mean,
            )
        })
        .collect::<Vec<_>>();
    write_svg_chart(
        path,
        &format!(
            "GMRF posterior correlation vs Euclidean reference, level {}",
            level.level
        ),
        "distance",
        "posterior correlation",
        &[
            Series::scatter("gmrf pairs", scatter),
            Series::line("reference bins", reference_line),
        ],
    )
}

fn write_spectral_correlation_svg(
    level: &CubeZeroFormKernelLevelReport,
    path: &Path,
) -> io::Result<()> {
    let reference_line = level
        .correlation_bins
        .iter()
        .map(|bin| {
            (
                0.5 * (bin.distance_min + bin.distance_max),
                bin.reference_mean,
            )
        })
        .collect::<Vec<_>>();
    let spectral_line = level
        .correlation_bins
        .iter()
        .filter_map(|bin| {
            bin.spectral_mean
                .map(|value| (0.5 * (bin.distance_min + bin.distance_max), value))
        })
        .collect::<Vec<_>>();
    write_svg_chart(
        path,
        &format!(
            "Spectral posterior correlation vs Euclidean reference, level {}",
            level.level
        ),
        "distance",
        "posterior correlation",
        &[
            Series::line("reference bins", reference_line),
            Series::line("spectral bins", spectral_line),
        ],
    )
}

fn write_variance_scatter_svg(
    level: &CubeZeroFormKernelLevelReport,
    path: &Path,
) -> io::Result<()> {
    let gmrf = level
        .reference_variances
        .iter()
        .zip(&level.gmrf_variances)
        .map(|(&reference, &value)| (reference, value))
        .collect::<Vec<_>>();
    let mut series = vec![Series::scatter("gmrf", gmrf)];
    series.push(Series::scatter(
        "gmrf calibrated",
        level
            .reference_variances
            .iter()
            .zip(&level.gmrf_calibrated_variances)
            .map(|(&reference, &value)| (reference, value))
            .collect::<Vec<_>>(),
    ));
    if let Some(spectral) = &level.spectral_variances {
        series.push(Series::scatter(
            "spectral",
            level
                .reference_variances
                .iter()
                .zip(spectral)
                .map(|(&reference, &value)| (reference, value))
                .collect::<Vec<_>>(),
        ));
    }
    let max_value = level
        .reference_variances
        .iter()
        .chain(level.gmrf_variances.iter())
        .chain(level.gmrf_calibrated_variances.iter())
        .copied()
        .fold(0.0, f64::max);
    series.push(Series::line(
        "identity",
        vec![(0.0, 0.0), (max_value, max_value)],
    ));
    write_svg_chart(
        path,
        &format!("Posterior variance scatter, level {}", level.level),
        "Euclidean reference variance",
        "method variance",
        &series,
    )
}

fn write_spectral_sweep_svg(rows: &[CubeZeroFormSpectralSweepRow], path: &Path) -> io::Result<()> {
    let points = rows
        .iter()
        .map(|row| (row.spectral_k as f64, row.variance_rmse))
        .collect::<Vec<_>>();
    write_svg_chart(
        path,
        "Spectral posterior variance RMSE vs k",
        "spectral k",
        "variance RMSE",
        &[Series::line("spectral", points)],
    )
}

#[derive(Clone)]
enum SeriesKind {
    Line,
    Scatter,
}

#[derive(Clone)]
struct Series {
    label: &'static str,
    points: Vec<(f64, f64)>,
    kind: SeriesKind,
}

impl Series {
    fn line(label: &'static str, points: Vec<(f64, f64)>) -> Self {
        Self {
            label,
            points,
            kind: SeriesKind::Line,
        }
    }

    fn scatter(label: &'static str, points: Vec<(f64, f64)>) -> Self {
        Self {
            label,
            points,
            kind: SeriesKind::Scatter,
        }
    }
}

fn write_svg_chart(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[Series],
) -> io::Result<()> {
    let all_points = series
        .iter()
        .flat_map(|series| series.points.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    if all_points.is_empty() {
        return Ok(());
    }
    let (mut x_min, mut x_max) = min_max(all_points.iter().map(|(x, _)| *x));
    let (mut y_min, mut y_max) = min_max(all_points.iter().map(|(_, y)| *y));
    expand_range(&mut x_min, &mut x_max);
    expand_range(&mut y_min, &mut y_max);

    let width = 900.0;
    let height = 620.0;
    let left = 92.0;
    let right = 36.0;
    let top = 54.0;
    let bottom = 82.0;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;
    let colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"];

    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )?;
    writeln!(
        writer,
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
    )?;
    writeln!(
        writer,
        "<text x=\"{}\" y=\"28\" font-family=\"sans-serif\" font-size=\"18\" fill=\"#111\">{}</text>",
        left,
        escape_xml(title)
    )?;
    writeln!(
        writer,
        "<line x1=\"{left}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#222\" stroke-width=\"1\"/>",
        top + plot_h,
        left + plot_w,
        top + plot_h
    )?;
    writeln!(
        writer,
        "<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{}\" stroke=\"#222\" stroke-width=\"1\"/>",
        top + plot_h
    )?;
    for tick in 0..=5 {
        let t = tick as f64 / 5.0;
        let x = left + t * plot_w;
        let value = x_min + t * (x_max - x_min);
        writeln!(
            writer,
            "<line x1=\"{x:.2}\" y1=\"{:.2}\" x2=\"{x:.2}\" y2=\"{:.2}\" stroke=\"#ddd\"/>",
            top,
            top + plot_h
        )?;
        writeln!(
            writer,
            "<text x=\"{x:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#333\">{value:.3}</text>",
            top + plot_h + 20.0
        )?;
    }
    for tick in 0..=5 {
        let t = tick as f64 / 5.0;
        let y = top + plot_h - t * plot_h;
        let value = y_min + t * (y_max - y_min);
        writeln!(
            writer,
            "<line x1=\"{left}\" y1=\"{y:.2}\" x2=\"{:.2}\" y2=\"{y:.2}\" stroke=\"#eee\"/>",
            left + plot_w
        )?;
        writeln!(
            writer,
            "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#333\">{value:.3}</text>",
            left - 8.0,
            y + 4.0
        )?;
    }

    for (series_index, series) in series.iter().enumerate() {
        let color = colors[series_index % colors.len()];
        match series.kind {
            SeriesKind::Line => {
                let path_data = series
                    .points
                    .iter()
                    .filter(|(x, y)| x.is_finite() && y.is_finite())
                    .enumerate()
                    .map(|(idx, &(x, y))| {
                        let (sx, sy) = scale_point(
                            x, y, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h,
                        );
                        if idx == 0 {
                            format!("M {sx:.2} {sy:.2}")
                        } else {
                            format!("L {sx:.2} {sy:.2}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(
                    writer,
                    "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2.2\"/>",
                    path_data, color
                )?;
            }
            SeriesKind::Scatter => {
                let stride = (series.points.len() / 2500).max(1);
                for &(x, y) in series.points.iter().step_by(stride) {
                    if !x.is_finite() || !y.is_finite() {
                        continue;
                    }
                    let (sx, sy) =
                        scale_point(x, y, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h);
                    writeln!(
                        writer,
                        "<circle cx=\"{sx:.2}\" cy=\"{sy:.2}\" r=\"1.8\" fill=\"{}\" fill-opacity=\"0.42\"/>",
                        color
                    )?;
                }
            }
        }
        let legend_y = top + 18.0 + 20.0 * series_index as f64;
        let legend_x = left + plot_w - 180.0;
        writeln!(
            writer,
            "<rect x=\"{legend_x:.2}\" y=\"{:.2}\" width=\"12\" height=\"12\" fill=\"{}\"/>",
            legend_y - 10.0,
            color
        )?;
        writeln!(
            writer,
            "<text x=\"{:.2}\" y=\"{legend_y:.2}\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#222\">{}</text>",
            legend_x + 18.0,
            escape_xml(series.label)
        )?;
    }

    writeln!(
        writer,
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#111\">{}</text>",
        left + plot_w / 2.0,
        height - 24.0,
        escape_xml(x_label)
    )?;
    writeln!(
        writer,
        "<text transform=\"translate(24,{}) rotate(-90)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\" fill=\"#111\">{}</text>",
        top + plot_h / 2.0,
        escape_xml(y_label)
    )?;
    writeln!(writer, "</svg>")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scale_point(
    x: f64,
    y: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    left: f64,
    top: f64,
    plot_w: f64,
    plot_h: f64,
) -> (f64, f64) {
    let sx = left + ((x - x_min) / (x_max - x_min)) * plot_w;
    let sy = top + plot_h - ((y - y_min) / (y_max - y_min)) * plot_h;
    (sx, sy)
}

fn min_max<I>(values: I) -> (f64, f64)
where
    I: IntoIterator<Item = f64>,
{
    values
        .into_iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        })
}

fn expand_range(min: &mut f64, max: &mut f64) {
    if !min.is_finite() || !max.is_finite() {
        *min = 0.0;
        *max = 1.0;
        return;
    }
    if (*max - *min).abs() <= f64::EPSILON {
        let pad = min.abs().max(1.0) * 0.05;
        *min -= pad;
        *max += pad;
    } else {
        let pad = (*max - *min) * 0.06;
        *min -= pad;
        *max += pad;
    }
}

fn optional_f64_csv(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.16}"))
        .unwrap_or_default()
}

fn optional_usize_csv(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperparameters_match_expected_cube_values() {
        let hp = cube_zero_form_kernel_hyperparameters(2.0, 1.0, 0.20, 1e-4).unwrap();

        assert_eq!(hp.dimension, 3);
        assert!((hp.nu - 0.5).abs() <= 1e-12);
        assert!((hp.kappa - 10.0).abs() <= 1e-12);
        assert!((hp.tau - 6.307_831_305_050_401e-2).abs() <= 1e-12);
    }

    #[test]
    fn interior_selector_and_sensors_stay_away_from_boundary() {
        let mesh = CartesianMeshInfo::new_unit_scaled(DEFAULT_DIMENSION, 4, 1.0);
        let (_topology, coords) = mesh.compute_coord_complex();

        let interior = interior_vertex_indices(&coords, DEFAULT_INTERIOR_MARGIN).unwrap();
        assert_eq!(interior.len(), 27);
        assert!(interior.iter().all(|&idx| {
            let coord = coords.coord(idx);
            (0..DEFAULT_DIMENSION).all(|axis| {
                coord[axis] >= DEFAULT_INTERIOR_MARGIN
                    && coord[axis] <= 1.0 - DEFAULT_INTERIOR_MARGIN
            })
        }));

        let sensors = deterministic_sensor_indices(
            &coords,
            &default_sensor_points(),
            DEFAULT_INTERIOR_MARGIN,
        )
        .unwrap();
        assert_eq!(sensors.len(), 27);
        assert_eq!(sensors.iter().copied().collect::<HashSet<_>>().len(), 27);
    }

    #[test]
    fn rmse_and_bins_are_deterministic() {
        let got = rmse(&[1.0, 3.0, 5.0], &[1.0, 1.0, 1.0]).unwrap();
        assert!((got - (20.0_f64 / 3.0).sqrt()).abs() <= 1e-12);

        let bins = correlation_bins(
            &[0.1, 0.2, 0.8, 1.0],
            &[1.0, 0.8, 0.2, 0.1],
            &[0.9, 0.7, 0.3, 0.2],
            Some(&[0.95, 0.75, 0.25, 0.15]),
            2,
        )
        .unwrap();
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0].count, 2);
        assert_eq!(bins[1].count, 2);
        assert!(bins[0].spectral_mean.is_some());
    }
}
