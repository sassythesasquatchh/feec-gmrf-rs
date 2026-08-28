//! Registry of maintained, report-backed scientific workflows.
//!
//! Data-driven descriptors provide listing, documentation, configuration, and
//! execution to the CLI.

use std::{collections::BTreeMap, path::Path};

/// Function used to execute one resolved study profile.
pub type StudyRunner =
    fn(&str, &StudyRunProfile<'_>, &Path) -> std::result::Result<Vec<(String, f64)>, String>;

/// A registry profile or a strict, file-backed research configuration.
#[derive(Debug, Clone, Copy)]
pub enum StudyRunProfile<'a> {
    Named(&'a str),
    Custom(&'a CustomStudyConfiguration),
}

impl<'a> StudyRunProfile<'a> {
    pub fn base_profile(self) -> &'a str {
        match self {
            Self::Named(profile) => profile,
            Self::Custom(config) => &config.base_profile,
        }
    }

    pub fn label(self) -> &'a str {
        match self {
            Self::Named(profile) => profile,
            Self::Custom(_) => "custom",
        }
    }

    fn raw(self, key: &str) -> Option<&'a str> {
        match self {
            Self::Named(_) => None,
            Self::Custom(config) => config.values.get(key).map(String::as_str),
        }
    }

    fn usize(self, key: &str, default: usize) -> Result<usize, String> {
        self.raw(key)
            .map(|value| parse_usize(key, value))
            .transpose()
            .map(|value| value.unwrap_or(default))
    }

    fn u64(self, key: &str, default: u64) -> Result<u64, String> {
        self.raw(key)
            .map(|value| parse_u64(key, value))
            .transpose()
            .map(|value| value.unwrap_or(default))
    }

    fn f64(self, key: &str, default: f64) -> Result<f64, String> {
        self.raw(key)
            .map(|value| parse_f64(key, value))
            .transpose()
            .map(|value| value.unwrap_or(default))
    }

    fn usize_list(self, key: &str, default: Vec<usize>) -> Result<Vec<usize>, String> {
        self.raw(key)
            .map(|value| parse_list(key, value, parse_usize))
            .transpose()
            .map(|value| value.unwrap_or(default))
    }

    fn f64_list(self, key: &str, default: Vec<f64>) -> Result<Vec<f64>, String> {
        self.raw(key)
            .map(|value| parse_list(key, value, parse_f64))
            .transpose()
            .map(|value| value.unwrap_or(default))
    }

    fn string(self, key: &str, default: String) -> Result<String, String> {
        self.raw(key)
            .map(|value| parse_string(key, value))
            .transpose()
            .map(|value| value.unwrap_or(default))
    }
}

/// Strict custom configuration loaded by `feg-study run --config`.
#[derive(Debug, Clone)]
pub struct CustomStudyConfiguration {
    pub study_id: String,
    pub base_profile: String,
    values: BTreeMap<String, String>,
}

impl CustomStudyConfiguration {
    /// Load a small, auditable TOML subset containing scalar values and arrays.
    /// Unknown keys are rejected for the selected study.
    pub fn from_path(path: &Path, expected_study_id: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read custom config {}: {error}", path.display()))?;
        let mut values = BTreeMap::new();
        for (line_index, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| {
                format!(
                    "{}:{}: expected `key = value`",
                    path.display(),
                    line_index + 1
                )
            })?;
            let key = key.trim().to_string();
            let value = value.trim().to_string();
            if values.insert(key.clone(), value).is_some() {
                return Err(format!("custom config repeats key `{key}`"));
            }
        }
        let schema = take_string(&mut values, "schema")?;
        if schema != "feg-study-custom-v1" {
            return Err(format!(
                "custom config schema must be `feg-study-custom-v1`, found `{schema}`"
            ));
        }
        let study_id = take_string(&mut values, "study_id")?;
        if study_id != expected_study_id {
            return Err(format!(
                "custom config targets `{study_id}`, not requested study `{expected_study_id}`"
            ));
        }
        let base_profile = take_string(&mut values, "base_profile")?;
        if !PROFILES.contains(&base_profile.as_str()) {
            return Err(format!(
                "custom base_profile must be one of: {}",
                PROFILES.join(", ")
            ));
        }
        let allowed = custom_keys(expected_study_id);
        if let Some(key) = values.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(format!(
                "custom key `{key}` is not supported for study `{expected_study_id}`; allowed keys: {}",
                allowed.join(", ")
            ));
        }
        Ok(Self {
            study_id,
            base_profile,
            values,
        })
    }
}

/// Metadata and runner for one maintained study.
#[derive(Debug, Clone, Copy)]
pub struct StudyDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    pub summary: &'static str,
    pub profiles: &'static [&'static str],
    pub requirements: &'static [&'static str],
    /// Repository-relative immutable inputs whose hashes are recorded.
    pub inputs: &'static [&'static str],
    pub run: StudyRunner,
}

const PROFILES: &[&str] = &["smoke", "thesis-submitted"];

/// Maintained study descriptors in identifier order.
pub fn published_studies() -> &'static [StudyDescriptor] {
    &PUBLISHED
}

/// Find a maintained study by identifier.
pub fn find_published_study(id: &str) -> Option<&'static StudyDescriptor> {
    PUBLISHED.iter().find(|study| study.id == id)
}

/// Keys accepted by a study's strict custom research configuration.
pub fn custom_configuration_keys(id: &str) -> &'static [&'static str] {
    custom_keys(id)
}

static PUBLISHED: [StudyDescriptor; 15] = [
    study(
        "hodge-laplacian/cube",
        "deterministic-feec",
        "Mixed-boundary cube Hodge--Laplacian validation",
        run_cube_hodge_laplacian,
        &[],
        &[],
    ),
    study(
        "hodge-laplacian/torus",
        "deterministic-feec",
        "Manufactured torus Hodge--Laplacian validation",
        run_torus_hodge_laplacian,
        &["gmsh"],
        &[
            "meshes/torus_shell_resolution_1.msh",
            "meshes/torus_shell_resolution_2.msh",
            "meshes/torus_shell_resolution_3.msh",
        ],
    ),
    study(
        "hodge-inverse/sparse",
        "deterministic-feec",
        "Sparse inverse-Hodge validation",
        run_sparse_inverse_hodge,
        &[],
        &[],
    ),
    study(
        "matern/scalar",
        "matern",
        "Scalar FEEC Matérn prior validation",
        run_scalar_matern,
        &[],
        &[],
    ),
    study(
        "matern/trace-normalization",
        "matern",
        "Matérn trace-normalization validation",
        run_trace_normalization,
        &[],
        &[],
    ),
    study(
        "matern/marginal-variance-3d",
        "matern",
        "Three-dimensional all-form marginal-variance study",
        run_marginal_variance_3d,
        &[],
        &[],
    ),
    study(
        "matern/marginal-variance-4d",
        "matern",
        "Four-dimensional scalar marginal-variance study",
        run_marginal_variance_4d,
        &[],
        &[],
    ),
    study(
        "hodge/sphere-observables",
        "hodge-observables",
        "Sphere exact/coexact observable convergence",
        run_sphere_observables,
        &[],
        &[],
    ),
    study(
        "hodge/torus-residual-weight",
        "hodge-observables",
        "Torus residual-precision posterior convergence",
        run_torus_residual_weight,
        &[],
        &[
            "meshes/torus_shell_resolution_0.msh",
            "meshes/torus_shell_resolution_1.msh",
            "meshes/torus_shell_resolution_2.msh",
            "meshes/torus_shell_resolution_3.msh",
        ],
    ),
    study(
        "magnetic/calibration",
        "electromagnetic-uq",
        "Physical magnetic-field prior calibration",
        run_magnetic_calibration,
        &[],
        &[],
    ),
    study(
        "magnetic/prior-mismatch",
        "electromagnetic-uq",
        "Magnetic prior-mismatch coverage study",
        run_magnetic_prior_mismatch,
        &[],
        &[],
    ),
    study(
        "annulus/h-formulation",
        "electromagnetic-uq",
        "Annular H-formulation accuracy and efficiency",
        run_annulus_h_formulation,
        &["gmsh"],
        &[],
    ),
    study(
        "toroidal-b/canonical",
        "toroidal-field-uq",
        "Canonical exact toroidal-B source/flux recovery",
        run_toroidal_canonical,
        &[],
        &["meshes/toroidal_inductor.msh"],
    ),
    study(
        "toroidal-b/source-noise",
        "toroidal-field-uq",
        "Toroidal source-noise sweep",
        run_toroidal_source_noise,
        &[],
        &["meshes/toroidal_inductor.msh"],
    ),
    study(
        "toroidal-b/coverage",
        "toroidal-field-uq",
        "Toroidal uncertainty-coverage sweep",
        run_toroidal_coverage,
        &[],
        &["meshes/toroidal_inductor.msh"],
    ),
];

const fn study(
    id: &'static str,
    family: &'static str,
    summary: &'static str,
    run: StudyRunner,
    requirements: &'static [&'static str],
    inputs: &'static [&'static str],
) -> StudyDescriptor {
    StudyDescriptor {
        id,
        family,
        summary,
        profiles: PROFILES,
        requirements,
        inputs,
        run,
    }
}

fn run_cube_hodge_laplacian(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    let default_resolutions: Vec<usize> = match profile.base_profile() {
        "smoke" => vec![2],
        "thesis-submitted" => vec![2, 4, 8, 16, 32],
        _ => return unknown_profile(profile),
    };
    let resolutions = profile.usize_list("resolutions", default_resolutions)?;
    let rows =
        formoniq::mixed_bc_hodge_laplacian_convergence::run_mixed_bc_hodge_laplacian_convergence(
            output,
            &resolutions,
        )
        .map_err(|error| error.to_string())?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!("resolutions = {:?}\nseed = 0\n", resolutions),
    )?;
    let last = rows
        .last()
        .ok_or("cube Hodge--Laplacian produced no rows")?;
    let mut metrics = vec![
        ("levels".into(), rows.len() as f64),
        ("finest_l2_error".into(), last.l2_error),
        ("finest_hd_error".into(), last.hd_error),
    ];
    if last.l2_rate.is_finite() {
        metrics.push(("finest_l2_rate".into(), last.l2_rate));
    }
    if last.hd_rate.is_finite() {
        metrics.push(("finest_hd_rate".into(), last.hd_rate));
    }
    Ok(metrics)
}

fn run_torus_hodge_laplacian(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    let default_resolutions: Vec<usize> = match profile.base_profile() {
        "smoke" => vec![0],
        "thesis-submitted" => (0..=5).collect(),
        _ => return unknown_profile(profile),
    };
    let resolutions = profile.usize_list("resolutions", default_resolutions)?;
    let rows =
        formoniq::torus_convergence::run_torus_convergence_for_resolutions(output, &resolutions)
            .map_err(|error| error.to_string())?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!("resolutions = {:?}\nseed = 0\n", resolutions),
    )?;
    let last = rows
        .last()
        .ok_or("torus Hodge--Laplacian produced no rows")?;
    let mut metrics = vec![
        ("levels".into(), rows.len() as f64),
        ("finest_l2_error".into(), last.l2_error),
        ("finest_hd_error".into(), last.hd_error),
    ];
    if last.l2_rate.is_finite() {
        metrics.push(("finest_l2_rate".into(), last.l2_rate));
    }
    if last.hd_rate.is_finite() {
        metrics.push(("finest_hd_rate".into(), last.hd_rate));
    }
    Ok(metrics)
}

fn run_sparse_inverse_hodge(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use formoniq::sparse_inverse_hodge_validation::{
        run_sparse_inverse_hodge_validation, SparseInverseHodgeValidationConfig,
    };
    let default_max_refinement = match profile.base_profile() {
        "smoke" => 1,
        "thesis-submitted" => 4,
        _ => return unknown_profile(profile),
    };
    let max_refinement =
        u32::try_from(profile.usize("max_refinement", default_max_refinement as usize)?)
            .map_err(|_| "custom key `max_refinement` exceeds u32 range".to_string())?;
    let config = SparseInverseHodgeValidationConfig {
        output_dir: output.to_path_buf(),
        max_refinement,
    };
    let rows = run_sparse_inverse_hodge_validation(&config).map_err(|error| error.to_string())?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!("max_refinement = {max_refinement}\nkappa = 0.0\nseed = 0\n"),
    )?;
    let last = rows
        .last()
        .ok_or("sparse inverse-Hodge validation produced no rows")?;
    Ok(vec![
        ("levels".into(), rows.len() as f64),
        ("reference_l2_error".into(), last.mixed_l2),
        ("projected_l2_error".into(), last.nc1_l2),
        ("barycentric_l2_error".into(), last.bary_l2),
        ("row_sum_l2_error".into(), last.rowsum_l2),
        ("projected_reference_distance".into(), last.mixed_nc1_l2),
        ("barycentric_reference_distance".into(), last.mixed_bary_l2),
        ("row_sum_reference_distance".into(), last.mixed_rowsum_l2),
    ])
}

fn run_scalar_matern(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::matern_scalar::{
        run_scalar_matern_validation, write_scalar_matern_validation_outputs,
        ScalarMaternValidationConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => ScalarMaternValidationConfig::smoke(),
        "thesis-submitted" => ScalarMaternValidationConfig::thesis_submitted(),
        _ => return unknown_profile(profile),
    };
    config.dimension = profile.usize("dimension", config.dimension)?;
    config.range_cells = profile.usize("range_cells", config.range_cells)?;
    config.level = profile.usize("level", config.level)?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "dimension = {}\nrange_cells = {}\nlevel = {}\naxis_lags = {}\nalpha = 2\nsigma2 = 1\nseed = 0\n",
            config.dimension,
            config.range_cells,
            config.level,
            2 * config.range_cells + 1,
        ),
    )?;
    let report = run_scalar_matern_validation(config)?;
    write_scalar_matern_validation_outputs(&report, output).map_err(|error| error.to_string())?;
    Ok(vec![
        ("latent_dimension".into(), report.ndofs as f64),
        ("axis_lag_outputs".into(), report.correlations.len() as f64),
        ("correlation_rmse".into(), report.correlation_rmse),
        (
            "relative_variance_error".into(),
            report.variance_relative_error,
        ),
    ])
}

fn run_trace_normalization(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::matern_trace_normalization::{
        compute_matern_trace_normalization_report, write_matern_trace_normalization_outputs,
        MaternTraceNormalizationConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => MaternTraceNormalizationConfig::smoke(),
        "thesis-submitted" => MaternTraceNormalizationConfig::thesis_submitted(),
        _ => return unknown_profile(profile),
    };
    config.levels = profile.usize_list("levels", config.levels)?;
    config.kappa = profile.f64("kappa", config.kappa)?;
    config.target_mean_trace_variance = profile.f64(
        "target_mean_trace_variance",
        config.target_mean_trace_variance,
    )?;
    config.exact_max_dofs = profile.usize("exact_max_dofs", config.exact_max_dofs)?;
    config.hutchinson_probes = profile.usize("hutchinson_probes", config.hutchinson_probes)?;
    config.hutchinson_batches = profile.usize("hutchinson_batches", config.hutchinson_batches)?;
    config.rng_seed = profile.u64("seed", config.rng_seed)?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "levels = {:?}\nkappa = {}\nhutchinson_probes = {}\nhutchinson_batches = {}\nseed = {}\n",
            config.levels,
            config.kappa,
            config.hutchinson_probes,
            config.hutchinson_batches,
            config.rng_seed,
        ),
    )?;
    let report = compute_matern_trace_normalization_report(config)?;
    write_matern_trace_normalization_outputs(&report, output).map_err(|error| error.to_string())?;
    let max_error = report
        .rows
        .iter()
        .map(|row| (row.normalized_hutchinson_mean_trace_variance - 1.0).abs())
        .fold(0.0, f64::max);
    Ok(vec![
        ("rows".into(), report.rows.len() as f64),
        ("max_normalized_trace_error".into(), max_error),
    ])
}

fn run_marginal_variance_3d(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::matern_functional_convergence::{
        run_functional_convergence_experiment, FunctionalConvergenceConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => FunctionalConvergenceConfig::smoke(output.to_path_buf()),
        "thesis-submitted" => FunctionalConvergenceConfig::thesis_submitted(output.to_path_buf()),
        _ => return unknown_profile(profile),
    };
    config.levels = profile.usize_list("levels", config.levels)?;
    config.kappa = profile.f64("kappa", config.kappa)?;
    config.tau = profile.f64("tau", config.tau)?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "levels = {:?}\nkappa = {}\ntau = {}\nseed = 0\n",
            config.levels, config.kappa, config.tau
        ),
    )?;
    let result =
        run_functional_convergence_experiment(&config).map_err(|error| error.to_string())?;
    Ok(vec![
        ("variance_rows".into(), result.rows.len() as f64),
        ("fit_rows".into(), result.fit_summaries.len() as f64),
    ])
}

fn run_marginal_variance_4d(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::matern_scalar_borderline_4d::{
        run_scalar_borderline_4d_experiment, ScalarBorderline4dConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => ScalarBorderline4dConfig::smoke(output.to_path_buf()),
        "thesis-submitted" => ScalarBorderline4dConfig::thesis_submitted(output.to_path_buf()),
        _ => return unknown_profile(profile),
    };
    config.levels = profile.usize_list("levels", config.levels)?;
    config.kappa = profile.f64("kappa", config.kappa)?;
    config.tau = profile.f64("tau", config.tau)?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "levels = {:?}\nkappa = {}\ntau = {}\nseed = 0\n",
            config.levels, config.kappa, config.tau,
        ),
    )?;
    let result = run_scalar_borderline_4d_experiment(&config).map_err(|error| error.to_string())?;
    Ok(vec![
        ("variance_rows".into(), result.rows.len() as f64),
        ("fit_rows".into(), result.fit_summaries.len() as f64),
    ])
}

fn run_sphere_observables(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::sphere_branch_observable_convergence::{
        run_sphere_branch_observable_convergence, SphereBranchObservableConvergenceConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => SphereBranchObservableConvergenceConfig::smoke(),
        "thesis-submitted" => SphereBranchObservableConvergenceConfig::thesis_submitted(),
        _ => return unknown_profile(profile),
    };
    config.levels = profile.usize_list("levels", config.levels)?;
    config.kappa = profile.f64("kappa", config.kappa)?;
    config.tau = profile.f64("tau", config.tau)?;
    config.lmax = profile.usize("lmax", config.lmax)?;
    config.pointwise_analytic_lmax =
        profile.usize("pointwise_analytic_lmax", config.pointwise_analytic_lmax)?;
    config.output_dir = output.to_path_buf();
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "levels = {:?}\nkappa = {}\ntau = {}\nlmax = {}\npointwise_analytic_lmax = {}\nseed = 0\n",
            config.levels, config.kappa, config.tau, config.lmax, config.pointwise_analytic_lmax
        ),
    )?;
    let result =
        run_sphere_branch_observable_convergence(&config).map_err(|error| error.to_string())?;
    Ok(vec![
        ("observable_rows".into(), result.variance_rows.len() as f64),
        ("pointwise_rows".into(), result.pointwise_rows.len() as f64),
        ("fit_rows".into(), result.fit_summary_rows.len() as f64),
    ])
}

fn run_torus_residual_weight(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::torus::posterior_residual_weight::{
        default_torus_shell_mesh_path, run_torus_1form_pde_posterior_mean_weight_sweep,
        write_torus_1form_pde_posterior_mean_weight_sweep_outputs, Torus1FormPdeMeshLevel,
        Torus1FormPdePosteriorMeanWeightSweepConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => Torus1FormPdePosteriorMeanWeightSweepConfig::smoke(),
        "thesis-submitted" => Torus1FormPdePosteriorMeanWeightSweepConfig::thesis_submitted(),
        _ => return unknown_profile(profile),
    };
    let default_resolutions = config
        .mesh_levels
        .iter()
        .map(|level| level.resolution)
        .collect::<Vec<_>>();
    let resolutions = profile.usize_list("resolutions", default_resolutions)?;
    validate_torus_residual_weight_resolutions(&resolutions)?;
    config.mesh_levels = resolutions
        .iter()
        .map(|&resolution| Torus1FormPdeMeshLevel {
            resolution,
            mesh_path: default_torus_shell_mesh_path(resolution),
        })
        .collect();
    config.weights = profile.f64_list("weights", config.weights)?;
    config.kappa = profile.f64("kappa", config.kappa)?;
    config.tau = profile.f64("tau", config.tau)?;
    let result = run_torus_1form_pde_posterior_mean_weight_sweep(&config)
        .map_err(|error| error.to_string())?;
    write_torus_1form_pde_posterior_mean_weight_sweep_outputs(&result, output)
        .map_err(|error| error.to_string())?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "resolutions = {:?}\nmesh_paths = {:?}\nweights = {:?}\nkappa = {}\ntau = {}\nseed = 0\n",
            resolutions,
            resolutions
                .iter()
                .map(|resolution| format!("meshes/torus_shell_resolution_{resolution}.msh"))
                .collect::<Vec<_>>(),
            config.weights,
            config.kappa,
            config.tau,
        ),
    )?;
    let maximum_high_weight_gap = result
        .summaries
        .iter()
        .map(|row| row.high_weight_posterior_deterministic_relative_l2_error)
        .fold(0.0, f64::max);
    Ok(vec![
        ("mesh_levels".into(), result.summaries.len() as f64),
        ("weight_rows".into(), result.rows.len() as f64),
        ("maximum_high_weight_gap".into(), maximum_high_weight_gap),
    ])
}

fn validate_torus_residual_weight_resolutions(
    resolutions: &[usize],
) -> std::result::Result<(), String> {
    if let Some(resolution) = resolutions
        .iter()
        .copied()
        .find(|resolution| *resolution > 3)
    {
        return Err(format!(
            "torus residual-weight resolution {resolution} is outside the maintained 0..=3 mesh sequence"
        ));
    }
    Ok(())
}

fn run_magnetic_calibration(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::magnetic_physical_calibration::{
        compute_magnetic_physical_calibration_report, write_magnetic_physical_calibration_outputs,
        MagneticPhysicalCalibrationConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => MagneticPhysicalCalibrationConfig::smoke(),
        "thesis-submitted" => MagneticPhysicalCalibrationConfig::thesis_submitted(),
        _ => return unknown_profile(profile),
    };
    config.levels = profile.usize_list("levels", config.levels)?;
    config.practical_range_m = profile.f64("practical_range_m", config.practical_range_m)?;
    config.target_b_rms_tesla = profile.f64("target_b_rms_tesla", config.target_b_rms_tesla)?;
    config.hutchinson_probes = profile.usize("hutchinson_probes", config.hutchinson_probes)?;
    config.hutchinson_batches = profile.usize("hutchinson_batches", config.hutchinson_batches)?;
    config.rng_seed = profile.u64("seed", config.rng_seed)?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "levels = {:?}\npractical_range_m = {}\ntarget_b_rms_tesla = {}\nhutchinson_probes = {}\nhutchinson_batches = {}\nseed = {}\n",
            config.levels,
            config.practical_range_m,
            config.target_b_rms_tesla,
            config.hutchinson_probes,
            config.hutchinson_batches,
            config.rng_seed,
        ),
    )?;
    let report = compute_magnetic_physical_calibration_report(config)?;
    write_magnetic_physical_calibration_outputs(&report, output)
        .map_err(|error| error.to_string())?;
    Ok(vec![
        ("prior_rows".into(), report.prior_rows.len() as f64),
        ("sensor_rows".into(), report.sensor_rows.len() as f64),
        ("variance_rows".into(), report.variance_rows.len() as f64),
    ])
}

fn run_magnetic_prior_mismatch(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::magnetic_prior_uq_comparison::{
        compute_magnetic_prior_uq_comparison_report, write_magnetic_prior_uq_comparison_outputs,
        MagneticPriorUqComparisonConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => MagneticPriorUqComparisonConfig::smoke(),
        "thesis-submitted" => MagneticPriorUqComparisonConfig::thesis_submitted(),
        _ => return unknown_profile(profile),
    };
    config.level = profile.usize("level", config.level)?;
    config.practical_range_m = profile.f64("practical_range_m", config.practical_range_m)?;
    config.smooth_noise_replicates =
        profile.usize("smooth_noise_replicates", config.smooth_noise_replicates)?;
    config.corrected_sample_truth_replicates = profile.usize(
        "corrected_sample_truth_replicates",
        config.corrected_sample_truth_replicates,
    )?;
    config.hutchinson_probes = profile.usize("hutchinson_probes", config.hutchinson_probes)?;
    config.hutchinson_batches = profile.usize("hutchinson_batches", config.hutchinson_batches)?;
    config.rng_seed = profile.u64("seed", config.rng_seed)?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "level = {}\npractical_range_m = {}\nsmooth_noise_replicates = {}\ncorrected_sample_truth_replicates = {}\nhutchinson_probes = {}\nhutchinson_batches = {}\nseed = {}\n",
            config.level,
            config.practical_range_m,
            config.smooth_noise_replicates,
            config.corrected_sample_truth_replicates,
            config.hutchinson_probes,
            config.hutchinson_batches,
            config.rng_seed,
        ),
    )?;
    let report = compute_magnetic_prior_uq_comparison_report(config)?;
    write_magnetic_prior_uq_comparison_outputs(&report, output)
        .map_err(|error| error.to_string())?;
    Ok(vec![
        ("prior_rows".into(), report.prior_rows.len() as f64),
        (
            "heldout_prediction_rows".into(),
            report.heldout_prediction_rows.len() as f64,
        ),
        (
            "aggregate_metric_rows".into(),
            report.aggregate_bin_metric_rows.len() as f64,
        ),
    ])
}

fn run_annulus_h_formulation(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::annulus_h_formulation::{
        run_annulus_h_formulation as run, write_annulus_h_formulation_outputs,
        AnnulusHFormulationConfig,
    };
    let mut config = match profile.base_profile() {
        "smoke" => AnnulusHFormulationConfig::smoke(output.to_path_buf()),
        "thesis-submitted" => AnnulusHFormulationConfig::thesis_submitted(output.to_path_buf()),
        _ => return unknown_profile(profile),
    };
    config.mesh_size = profile.f64("mesh_size", config.mesh_size)?;
    config.noise_trial_count = profile.usize("noise_trial_count", config.noise_trial_count)?;
    config.residual_count = profile.usize("residual_count", config.residual_count)?;
    config.heldout_loop_count = profile.usize("heldout_loop_count", config.heldout_loop_count)?;
    config.rng_seed = profile.u64("seed", config.rng_seed)?;
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "mesh_size = {}\nnoise_trial_count = {}\nresidual_count = {}\nheldout_loop_count = {}\nseed = {}\n",
            config.mesh_size,
            config.noise_trial_count,
            config.residual_count,
            config.heldout_loop_count,
            config.rng_seed,
        ),
    )?;
    let result = run(&config).map_err(|error| error.to_string())?;
    write_annulus_h_formulation_outputs(&result, &config).map_err(|error| error.to_string())?;
    Ok(vec![
        ("trial_metrics".into(), result.trial_metrics.len() as f64),
        ("prediction_rows".into(), result.predictions.len() as f64),
        (
            "harmonic_dimension".into(),
            result.topology.harmonic_1_dimension as f64,
        ),
    ])
}

fn run_toroidal_canonical(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::toroidal_inductor::{
        run_toroidal_exact_b_recovery_experiment,
        toroidal_exact_b_canonical_source_designed_flux_config,
        toroidal_exact_b_thesis_submitted_observation_index_override,
    };
    let mut config = toroidal_exact_b_canonical_source_designed_flux_config();
    match profile.base_profile() {
        "smoke" => {
            config.heldout_count = 8;
            config.surface_flux_azimuth_count = 4;
        }
        "thesis-submitted" => {
            if matches!(profile, StudyRunProfile::Named("thesis-submitted")) {
                config.observation_index_override =
                    Some(toroidal_exact_b_thesis_submitted_observation_index_override());
            }
        }
        _ => return unknown_profile(profile),
    }
    config.base.mesh_path = profile
        .string("mesh_path", config.base.mesh_path.display().to_string())?
        .into();
    config.heldout_count = profile.usize("heldout_count", config.heldout_count)?;
    config.surface_flux_azimuth_count = profile.usize(
        "surface_flux_azimuth_count",
        config.surface_flux_azimuth_count,
    )?;
    config.observation_noise_std =
        profile.f64("observation_noise_std", config.observation_noise_std)?;
    config.observation_seed = profile.u64("seed", config.observation_seed)?;
    config.output_dir = Some(output.to_path_buf());
    config.write_outputs = true;
    let observation_indices = config
        .observation_index_override
        .as_ref()
        .map(|indices| {
            format!(
                "observation_index_source = \"submitted-explicit\"\ntraining_indices = {:?}\nheldout_indices = {:?}\n",
                indices.training_indices, indices.heldout_indices
            )
        })
        .unwrap_or_else(|| "observation_index_source = \"dynamic-source-design\"\n".to_string());
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "mesh_path = {:?}\nheldout_count = {}\nsurface_flux_azimuth_count = {}\nobservation_noise_std = {}\nseed = {}\n{}",
            config.base.mesh_path,
            config.heldout_count,
            config.surface_flux_azimuth_count,
            config.observation_noise_std,
            config.observation_seed,
            observation_indices,
        ),
    )?;
    let report = run_toroidal_exact_b_recovery_experiment(&config)?;
    Ok(vec![
        ("active_dofs".into(), report.summary.active_dofs as f64),
        ("training_rows".into(), report.summary.training_rows as f64),
        ("heldout_rows".into(), report.summary.heldout_rows as f64),
        ("heldout_rmse".into(), report.summary.heldout_rmse),
        (
            "heldout_coverage_fraction".into(),
            report.summary.heldout_coverage_fraction,
        ),
    ])
}

fn run_toroidal_source_noise(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    run_toroidal_sweep(
        study_id,
        profile,
        output,
        crate::toroidal_exact_b_sweeps::ToroidalExactBSweepKind::SourceNoise,
    )
}

fn run_toroidal_coverage(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
) -> std::result::Result<Vec<(String, f64)>, String> {
    run_toroidal_sweep(
        study_id,
        profile,
        output,
        crate::toroidal_exact_b_sweeps::ToroidalExactBSweepKind::FieldCoverage,
    )
}

fn run_toroidal_sweep(
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    output: &Path,
    kind: crate::toroidal_exact_b_sweeps::ToroidalExactBSweepKind,
) -> std::result::Result<Vec<(String, f64)>, String> {
    use crate::toroidal_exact_b_sweeps::{ToroidalExactBSweepKind, ToroidalExactBSweepProfile};
    let resolved_profile = match profile.base_profile() {
        "smoke" => ToroidalExactBSweepProfile::Smoke,
        "thesis-submitted" => ToroidalExactBSweepProfile::ThesisSubmitted,
        _ => return unknown_profile(profile),
    };
    let default_source_count = match resolved_profile {
        ToroidalExactBSweepProfile::Smoke => 1,
        ToroidalExactBSweepProfile::ThesisSubmitted => 6,
    };
    let default_field_count = match resolved_profile {
        ToroidalExactBSweepProfile::Smoke => 1,
        ToroidalExactBSweepProfile::ThesisSubmitted => 4,
    };
    let source_count = profile.usize("source_noise_case_count", default_source_count)?;
    let field_count = profile.usize("field_coverage_case_count", default_field_count)?;
    let report = crate::toroidal_exact_b_sweeps::run_toroidal_exact_b_sweeps_with_case_limits(
        output,
        kind,
        source_count,
        field_count,
    )
    .map_err(|error| error.to_string())?;
    let kind_label = match kind {
        ToroidalExactBSweepKind::SourceNoise => "source-noise",
        ToroidalExactBSweepKind::FieldCoverage => "field-coverage",
        ToroidalExactBSweepKind::Both => "both",
    };
    write_resolved(
        output,
        study_id,
        profile,
        &format!(
            "sweep = \"{kind_label}\"\nsource_noise_cases = {}\nfield_coverage_cases = {}\nseed = {}\n",
            report.source_noise_cases,
            report.field_coverage_cases,
            0xDA7A_2026_u64,
        ),
    )?;
    Ok(vec![
        (
            "source_noise_cases".into(),
            report.source_noise_cases as f64,
        ),
        (
            "field_coverage_cases".into(),
            report.field_coverage_cases as f64,
        ),
        (
            "source_noise_min_coverage".into(),
            report.source_noise_min_coverage,
        ),
        (
            "field_coverage_min_coverage".into(),
            report.field_coverage_min_coverage,
        ),
    ])
}

fn write_resolved(
    output: &Path,
    study_id: &str,
    profile: &StudyRunProfile<'_>,
    body: &str,
) -> std::result::Result<(), String> {
    let text = format!(
        "schema = \"feg-study-profile-v1\"\nstudy_id = \"{study_id}\"\nprofile = \"{}\"\nbase_profile = \"{}\"\n{body}",
        profile.label(),
        profile.base_profile(),
    );
    std::fs::write(output.join("resolved-profile.toml"), text).map_err(|error| error.to_string())
}

fn unknown_profile<T>(profile: &StudyRunProfile<'_>) -> std::result::Result<T, String> {
    Err(format!(
        "unknown study profile `{}`",
        profile.base_profile()
    ))
}

fn take_string(values: &mut BTreeMap<String, String>, key: &str) -> Result<String, String> {
    let value = values
        .remove(key)
        .ok_or_else(|| format!("custom config requires `{key}`"))?;
    parse_string(key, &value)
}

fn parse_string(key: &str, value: &str) -> Result<String, String> {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .map(str::to_string)
        .ok_or_else(|| format!("custom key `{key}` must be a quoted string"))
}

fn parse_usize(key: &str, value: &str) -> Result<usize, String> {
    value
        .replace('_', "")
        .parse::<usize>()
        .map_err(|_| format!("custom key `{key}` must be a non-negative integer"))
}

fn parse_u64(key: &str, value: &str) -> Result<u64, String> {
    let value = value.replace('_', "");
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
    } else {
        value.parse::<u64>()
    }
    .map_err(|_| format!("custom key `{key}` must be an unsigned integer"))
}

fn parse_f64(key: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .replace('_', "")
        .parse::<f64>()
        .map_err(|_| format!("custom key `{key}` must be a real number"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("custom key `{key}` must be finite"))
    }
}

fn parse_list<T>(
    key: &str,
    value: &str,
    parse_item: fn(&str, &str) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let body = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| format!("custom key `{key}` must be an array"))?
        .trim();
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|item| parse_item(key, item.trim()))
        .collect()
}

fn custom_keys(study_id: &str) -> &'static [&'static str] {
    match study_id {
        "hodge-laplacian/cube" | "hodge-laplacian/torus" => &["resolutions"],
        "hodge-inverse/sparse" => &["max_refinement"],
        "matern/scalar" => &["dimension", "range_cells", "level"],
        "matern/trace-normalization" => &[
            "levels",
            "kappa",
            "target_mean_trace_variance",
            "exact_max_dofs",
            "hutchinson_probes",
            "hutchinson_batches",
            "seed",
        ],
        "matern/marginal-variance-3d" => &["levels", "kappa", "tau"],
        "matern/marginal-variance-4d" => &["levels", "kappa", "tau"],
        "hodge/sphere-observables" => {
            &["levels", "kappa", "tau", "lmax", "pointwise_analytic_lmax"]
        }
        "hodge/torus-residual-weight" => &["resolutions", "weights", "kappa", "tau"],
        "magnetic/calibration" => &[
            "levels",
            "practical_range_m",
            "target_b_rms_tesla",
            "hutchinson_probes",
            "hutchinson_batches",
            "seed",
        ],
        "magnetic/prior-mismatch" => &[
            "level",
            "practical_range_m",
            "smooth_noise_replicates",
            "corrected_sample_truth_replicates",
            "hutchinson_probes",
            "hutchinson_batches",
            "seed",
        ],
        "annulus/h-formulation" => &[
            "mesh_size",
            "noise_trial_count",
            "residual_count",
            "heldout_loop_count",
            "seed",
        ],
        "toroidal-b/canonical" => &[
            "mesh_path",
            "heldout_count",
            "surface_flux_azimuth_count",
            "observation_noise_std",
            "seed",
        ],
        "toroidal-b/source-noise" => &["source_noise_case_count"],
        "toroidal-b/coverage" => &["field_coverage_case_count"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_and_profiles_are_unique_and_complete() {
        let mut ids = std::collections::BTreeSet::new();
        for study in published_studies() {
            assert!(ids.insert(study.id));
            assert!(study.profiles.contains(&"smoke"));
            assert!(study.profiles.contains(&"thesis-submitted"));
        }
    }

    #[test]
    fn parses_strict_custom_configuration() {
        let path = std::env::temp_dir().join(format!(
            "feg-study-custom-{}-{}.toml",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(
            &path,
            "schema = \"feg-study-custom-v1\"\nstudy_id = \"matern/scalar\"\nbase_profile = \"smoke\"\ndimension = 2\nrange_cells = 3\nlevel = 12\n",
        )
        .unwrap();
        let custom = CustomStudyConfiguration::from_path(&path, "matern/scalar").unwrap();
        let profile = StudyRunProfile::Custom(&custom);
        assert_eq!(profile.base_profile(), "smoke");
        assert_eq!(profile.usize("dimension", 3).unwrap(), 2);
        assert_eq!(profile.usize("range_cells", 2).unwrap(), 3);
        assert_eq!(profile.usize("level", 8).unwrap(), 12);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_custom_keys_owned_by_another_study() {
        let path = std::env::temp_dir().join(format!(
            "feg-study-invalid-custom-{}.toml",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "schema = \"feg-study-custom-v1\"\nstudy_id = \"matern/scalar\"\nbase_profile = \"smoke\"\nmesh_size = 0.1\n",
        )
        .unwrap();
        let error = CustomStudyConfiguration::from_path(&path, "matern/scalar").unwrap_err();
        assert!(error.contains("not supported"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn torus_residual_weight_accepts_only_available_mesh_resolutions() {
        validate_torus_residual_weight_resolutions(&[0, 1, 2, 3]).unwrap();
        let error = validate_torus_residual_weight_resolutions(&[0, 4]).unwrap_err();
        assert!(error.contains("resolution 4"));
    }
}
