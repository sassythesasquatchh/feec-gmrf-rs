//! Linear probabilistic TEAM 7 eddy-current model.
//!
//! The FEEC state is a complex Whitney 1-cochain represented as doubled real
//! unknowns `[Re(A), Im(A)]`.  The reduced residual is
//! `L x - b - S theta`, with `L = [[K, -omega C], [omega C, K]]`.

use crate::visual_output;
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::{cochain::Cochain, whitney::lsf::WhitneyLsf, ManifoldComplexExt};
use exterior::field::{DiffFormClosure, ExteriorField};
use feg_core::{
    GaussianPriorSpec, LinearGaussianMeasurementSpec, LinearUncertainInputSpec,
    RepresentationPreference, SparseTripletMatrix,
};
use feg_infer::{
    core_triplet_to_feec_csr,
    linear_pde::{
        solve_linear_pde_uq_with_config, LinearPdeDerivedQuantitySpec, LinearPdePrecisionPolicy,
        LinearPdeUqProblem, LinearPdeUqResult, LinearPdeUqSolverConfig, LinearPdeVarianceConfig,
        LinearPdeVarianceMode,
    },
    prior::matern::one_form::{
        build_hodge_laplacian_1form, build_matern_precision_1form,
        MaternConfig as Matern1FormConfig, MaternMassInverse as Matern1FormMassInverse,
    },
    sparse::{
        apply_sparse_row, feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec,
        sparse_rows_to_triplet as rows_to_triplet,
    },
};
use formoniq::{
    assemble::{assemble_galvec, boundary_simplices_where_barycenter},
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::{
        eddy_current::{build_reduced_eddy_current_1form_system, reduce_eddy_current_source},
        reduced_linear::ReducedLinearPdeAssembly,
    },
    reduction::{DofLayout, EssentialBoundarySpec, PrescribedDof},
};
use gmrf_core::{SparseLuFactor, SparseRowOperator};
use manifold::{
    geometry::{
        coord::{mesh::MeshCoords, simplex::SimplexHandleExt, CoordRef},
        metric::mesh::MeshLengths,
    },
    topology::{complex::Complex, handle::SimplexHandle},
};
use std::{
    collections::BTreeMap,
    f64::consts::PI,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const TEAM7_FREQUENCY_HZ: f64 = 50.0;
pub const TEAM7_OMEGA: f64 = 2.0 * PI * TEAM7_FREQUENCY_HZ;
pub const TEAM7_MU0: f64 = 4.0 * PI * 1e-7;
pub const TEAM7_NU0: f64 = 1.0 / TEAM7_MU0;
pub const TEAM7_ALUMINUM_SIGMA: f64 = 35.26e6;
pub const TEAM7_BACKGROUND_SIGMA: f64 = 1.0;
pub const TEAM7_TURNS: f64 = 2742.0;
pub const TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD: f64 = 0.02;
pub const TEAM7_DEFAULT_SOURCE_RELATIVE_STD: f64 = TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD;
pub const TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD: f64 = PI / 180.0;
pub const TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD: f64 = 0.02;
pub const TEAM7_DEFAULT_SOURCE_SHAPE_MODES: usize = 2;
pub const TEAM7_DEFAULT_SOURCE_GUARD_Z_SCORE: f64 = 5.0;
pub const TEAM7_DEFAULT_SOURCE_MIN_ALPHA: f64 = 0.5;
pub const TEAM7_DEFAULT_SOURCE_MIN_MULTIPLIER: f64 = TEAM7_DEFAULT_SOURCE_MIN_ALPHA;

pub const TEAM7_AIRBOX_MIN: f64 = -0.2;
pub const TEAM7_AIRBOX_MAX: f64 = 0.5;
pub const TEAM7_B_REAL_DERIVED_NAME: &str = "B_R";
pub const TEAM7_B_IMAG_DERIVED_NAME: &str = "B_I";
pub const TEAM7_SOURCE_CALIBRATION_INPUT_NAME: &str = "team7_harmonic_source_calibration";

const OBS_EPS: f64 = 1e-12;
const TEAM7_SOURCE_GN_MAX_ITERS: usize = 6;
const TEAM7_SOURCE_GN_TOL: f64 = 1e-7;

const XI_MSM: [f64; 17] = [
    0.000, 0.018, 0.036, 0.054, 0.072, 0.090, 0.108, 0.126, 0.144, 0.162, 0.180, 0.198, 0.216,
    0.234, 0.252, 0.270, 0.288,
];

const BZ_A1B1_REF: [(f64, f64); 17] = [
    (-4.9, -1.16),
    (-17.88, 2.48),
    (-22.13, 4.15),
    (-20.19, 4.0),
    (-15.67, 3.07),
    (0.36, 2.31),
    (43.64, 1.89),
    (78.11, 4.97),
    (71.55, 12.61),
    (60.44, 14.15),
    (53.91, 13.04),
    (52.62, 12.4),
    (53.81, 12.05),
    (56.91, 12.27),
    (59.24, 12.66),
    (52.78, 9.96),
    (27.61, 2.26),
];

const BZ_A2B2_REF: [(f64, f64); 17] = [
    (-1.83, -1.63),
    (-8.5, -0.6),
    (-13.6, -0.43),
    (-15.21, 0.11),
    (-14.48, 1.26),
    (-5.62, 3.4),
    (28.77, 6.53),
    (60.34, 10.25),
    (61.84, 11.83),
    (56.64, 11.83),
    (53.4, 11.01),
    (52.36, 10.58),
    (53.93, 10.8),
    (56.82, 10.54),
    (59.48, 10.62),
    (52.08, 9.03),
    (26.56, 1.79),
];

const JY_A3B3_REF: [(f64, f64); 17] = [
    (0.249, -0.629),
    (0.685, -0.873),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (-0.015, -0.593),
    (-0.103, -0.249),
    (-0.061, -0.101),
    (-0.004, -0.001),
    (0.051, 0.087),
    (0.095, 0.182),
    (0.135, 0.322),
    (0.104, 0.555),
    (-0.321, 0.822),
    (-0.687, 0.855),
];

const JY_A4B4_REF: [(f64, f64); 17] = [
    (0.461, -0.662),
    (0.621, -0.664),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.573, -1.027),
    (0.556, -0.757),
    (0.237, -0.364),
    (0.097, -0.149),
    (-0.034, 0.015),
    (-0.157, 0.154),
    (-0.305, 0.311),
    (-0.478, 0.508),
    (-0.66, 0.747),
    (-1.217, 1.034),
];

#[derive(Debug, Clone)]
pub struct Team7Config {
    pub mesh_path: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub frequency_hz: f64,
    pub aluminum_sigma: f64,
    pub background_sigma: f64,
    pub turns: f64,
    pub pde_residual_std: f64,
    pub b_observation_std_tesla: f64,
    pub jy_observation_std: f64,
    pub source_model: Team7SourceModel,
    pub matern_kappa: f64,
    pub matern_tau: f64,
    pub prior: Team7PriorConfig,
    pub prior_mean: Team7PriorMeanMode,
    pub solver: LinearPdeUqSolverConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Team7SourceModel {
    #[default]
    Deterministic,
    /// Gauss-Newton/Laplace source calibration for alpha exp(i phi) j_nom + U beta.
    /// Each iteration solves the sparse linear tangent model and the report guards
    /// against accepting amplitudes outside the credible calibration range.
    HarmonicCalibration {
        log_alpha_std: f64,
        phase_std_rad: f64,
        shape_relative_std: f64,
        shape_modes: usize,
        guard_z_score: f64,
        min_alpha: f64,
    },
}

impl Team7SourceModel {
    pub fn default_global_calibration() -> Self {
        Self::default_harmonic_calibration()
    }

    pub fn default_harmonic_calibration() -> Self {
        Self::HarmonicCalibration {
            log_alpha_std: TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD,
            phase_std_rad: TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD,
            shape_relative_std: TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD,
            shape_modes: TEAM7_DEFAULT_SOURCE_SHAPE_MODES,
            guard_z_score: TEAM7_DEFAULT_SOURCE_GUARD_Z_SCORE,
            min_alpha: TEAM7_DEFAULT_SOURCE_MIN_ALPHA,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Team7PriorConfig {
    pub physical_weight: f64,
    pub matern_stabilizer: Team7MaternStabilizer,
}

impl Default for Team7PriorConfig {
    fn default() -> Self {
        Self {
            physical_weight: 1.0,
            matern_stabilizer: Team7MaternStabilizer::RelativeToPhysicalDiagonal(1e-6),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Team7MaternStabilizer {
    Disabled,
    Absolute(f64),
    RelativeToPhysicalDiagonal(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Team7PriorMeanMode {
    Zero,
    #[default]
    DeterministicNominal,
}

impl Default for Team7Config {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("meshes/team7.msh"),
            output_dir: Some(PathBuf::from("out/team7_linear_probabilistic")),
            frequency_hz: TEAM7_FREQUENCY_HZ,
            aluminum_sigma: TEAM7_ALUMINUM_SIGMA,
            background_sigma: TEAM7_BACKGROUND_SIGMA,
            turns: TEAM7_TURNS,
            pde_residual_std: 1e-3,
            b_observation_std_tesla: 1e-4,
            jy_observation_std: 5e-2,
            source_model: Team7SourceModel::Deterministic,
            matern_kappa: 25.0,
            matern_tau: 1.0,
            prior: Team7PriorConfig::default(),
            prior_mean: Team7PriorMeanMode::default(),
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::Hutchinson,
                    num_variance_probes: 32,
                    variance_batch_count: 4,
                    rng_seed: 7,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: false,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team7ProblemAssembly {
    pub problem: LinearPdeUqProblem,
    pub observation_rows: Vec<Team7ObservationRow>,
    pub single_edge_layout: DofLayout,
    pub edge_count: usize,
    pub face_count: usize,
}

#[derive(Debug, Clone)]
pub struct Team7ObservationRow {
    pub family: String,
    pub line: String,
    pub component: String,
    pub point: [f64; 3],
    pub observed: f64,
    pub row: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
pub struct Team7ObservationReport {
    pub family: String,
    pub line: String,
    pub component: String,
    pub point: [f64; 3],
    pub observed: f64,
    pub predicted: f64,
    pub residual: f64,
}

#[derive(Debug, Clone)]
pub struct Team7RmseReport {
    pub family: String,
    pub rmse: f64,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Team7SourceCalibrationReport {
    pub log_alpha_mean: f64,
    pub log_alpha_std: f64,
    pub alpha_mean: f64,
    pub phase_mean_rad: f64,
    pub phase_std_rad: f64,
    pub log_alpha_z_score: f64,
    pub phase_z_score: f64,
    pub shape_mode_means: Vec<f64>,
    pub shape_mode_stds: Vec<f64>,
    pub max_shape_z_score: f64,
    pub accepted: bool,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Team7LinearSolveResult {
    pub posterior: LinearPdeUqResult,
    pub observations: Vec<Team7ObservationReport>,
    pub rmse: Vec<Team7RmseReport>,
    pub source_calibration: Option<Team7SourceCalibrationReport>,
}

pub fn solve_team7_linear_probabilistic(
    config: &Team7Config,
) -> Result<Team7LinearSolveResult, String> {
    validate_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 7 mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 7 requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    match config.source_model {
        Team7SourceModel::Deterministic => {
            let assembly = build_team7_problem(&topology, &coords, config)?;
            solve_team7_assembled(&topology, &coords, config, &assembly, true)
        }
        Team7SourceModel::HarmonicCalibration { shape_modes, .. } => {
            solve_team7_harmonic_calibration(&topology, &coords, config, shape_modes)
        }
    }
}

fn solve_team7_harmonic_calibration(
    topology: &Complex,
    coords: &MeshCoords,
    config: &Team7Config,
    shape_modes: usize,
) -> Result<Team7LinearSolveResult, String> {
    let mut linearization = vec![0.0; 2 + shape_modes];
    for _ in 0..TEAM7_SOURCE_GN_MAX_ITERS {
        let assembly = build_team7_problem_with_source_linearization(
            topology,
            coords,
            config,
            Some(&linearization),
        )?;
        let result = solve_team7_assembled(topology, coords, config, &assembly, false)?;
        let next = source_calibration_parameters(&result)?;
        let delta = max_abs_parameter_delta(&linearization, &next);
        if delta <= TEAM7_SOURCE_GN_TOL {
            if let Some(output_dir) = &config.output_dir {
                write_team7_outputs(output_dir, topology, coords, &assembly, &result)?;
            }
            return Ok(result);
        }
        linearization = next;
    }

    let assembly = build_team7_problem_with_source_linearization(
        topology,
        coords,
        config,
        Some(&linearization),
    )?;
    solve_team7_assembled(topology, coords, config, &assembly, true)
}

fn solve_team7_assembled(
    topology: &Complex,
    coords: &MeshCoords,
    config: &Team7Config,
    assembly: &Team7ProblemAssembly,
    write_outputs: bool,
) -> Result<Team7LinearSolveResult, String> {
    let posterior = solve_linear_pde_uq_with_config(&assembly.problem, &config.solver)?;
    let observations = evaluate_observation_reports(&assembly.observation_rows, &posterior);
    let rmse = compute_rmse(&observations);
    let source_calibration = build_source_calibration_report(config, &posterior)?;

    let result = Team7LinearSolveResult {
        posterior,
        observations,
        rmse,
        source_calibration,
    };

    if write_outputs {
        if let Some(output_dir) = &config.output_dir {
            write_team7_outputs(output_dir, topology, coords, assembly, &result)?;
        }
    }

    Ok(result)
}

fn source_calibration_parameters(result: &Team7LinearSolveResult) -> Result<Vec<f64>, String> {
    let report = result
        .source_calibration
        .as_ref()
        .ok_or_else(|| "TEAM 7 harmonic calibration report is missing".to_string())?;
    let mut parameters = Vec::with_capacity(2 + report.shape_mode_means.len());
    parameters.push(report.log_alpha_mean);
    parameters.push(report.phase_mean_rad);
    parameters.extend(report.shape_mode_means.iter().copied());
    Ok(parameters)
}

fn max_abs_parameter_delta(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

pub fn build_team7_problem(
    topology: &Complex,
    coords: &MeshCoords,
    config: &Team7Config,
) -> Result<Team7ProblemAssembly, String> {
    build_team7_problem_with_source_linearization(topology, coords, config, None)
}

fn build_team7_problem_with_source_linearization(
    topology: &Complex,
    coords: &MeshCoords,
    config: &Team7Config,
    source_linearization: Option<&[f64]>,
) -> Result<Team7ProblemAssembly, String> {
    validate_config(config)?;
    let metric = coords.to_edge_lengths(topology);
    let omega = 2.0 * PI * config.frequency_hz;
    let boundary = build_outer_boundary(topology, coords);
    let inverse_permeability = InnerProductWeightClosure::new(|_| TEAM7_NU0);
    let conductivity = {
        let aluminum_sigma = config.aluminum_sigma;
        let background_sigma = config.background_sigma;
        InnerProductWeightClosure::new(move |point| {
            if is_plate_point(point) {
                aluminum_sigma
            } else {
                background_sigma
            }
        })
    };
    let eddy = build_reduced_eddy_current_1form_system(
        topology,
        &metric,
        coords,
        None,
        &inverse_permeability,
        &conductivity,
        &boundary,
    )?;
    let source_full = assemble_team7_source(topology, &metric, coords, config.turns);
    let source_reduced = reduce_eddy_current_source(&eddy.layout, &source_full)?;
    let source_shape_modes =
        build_source_shape_modes(topology, &metric, coords, &eddy, &source_reduced, config)?;
    let source_parameters = source_linearization_parameters(config, source_linearization)?;
    let system = build_doubled_reduced_system_with_source_linearization(
        &eddy,
        &source_reduced,
        &source_shape_modes,
        omega,
        source_parameters.as_deref(),
    )?;
    let nominal_residual_bias =
        source_linearized_residual_bias(&eddy, &source_reduced, &[], omega, None)?;
    let system_operator = csr_to_triplet(&system.operator);
    let state_prior_mean =
        build_team7_prior_mean(config.prior_mean, &system_operator, &nominal_residual_bias)?;
    let forcing_precision = block_diag_scaled(
        &[
            csr_to_triplet(&eddy.state_mass_inverse),
            csr_to_triplet(&eddy.state_mass_inverse),
        ],
        1.0,
    );
    let state_prior = build_team7_state_prior(
        topology,
        &metric,
        &eddy.layout,
        &system_operator,
        &forcing_precision,
        state_prior_mean,
        config,
    )?;
    let pde_precision = Some(scale_triplet_matrix(
        &forcing_precision,
        1.0 / (config.pde_residual_std * config.pde_residual_std),
    ));
    let observations =
        build_team7_observation_rows(topology, coords, &system.layout, omega, config)?;
    let physical_measurements =
        group_observation_specs(&observations, system.layout.full_dimension, config);
    let mut uncertain_inputs = Vec::new();
    if let Team7SourceModel::HarmonicCalibration {
        log_alpha_std,
        phase_std_rad,
        shape_relative_std,
        ..
    } = config.source_model
    {
        let parameters = source_parameters
            .as_deref()
            .ok_or_else(|| "TEAM 7 source calibration linearization is missing".to_string())?;
        uncertain_inputs.push(source_calibration_input(
            &source_reduced,
            &source_shape_modes,
            parameters,
            log_alpha_std,
            phase_std_rad,
            shape_relative_std,
        )?);
    }

    let problem = LinearPdeUqProblem {
        state_prior,
        system,
        uncertain_inputs,
        joint_measurements: Vec::new(),
        physical_measurements,
        derived_quantities: build_derived_quantities(topology, &observations)?,
        joint_derived_quantities: Vec::new(),
        pde_variance: None,
        pde_precision,
    };

    Ok(Team7ProblemAssembly {
        problem,
        observation_rows: observations,
        single_edge_layout: eddy.layout,
        edge_count: topology.skeleton(1).len(),
        face_count: topology.skeleton(2).len(),
    })
}

pub fn generate_team7_geo(path: impl AsRef<Path>, mesh_size: f64) -> Result<(), String> {
    if !mesh_size.is_finite() || mesh_size <= 0.0 {
        return Err("mesh_size must be finite and positive".to_string());
    }
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create `{}`: {err}", parent.display()))?;
    }
    fs::write(path, team7_geo_source(mesh_size))
        .map_err(|err| format!("failed to write `{}`: {err}", path.display()))
}

pub fn team7_geo_source(mesh_size: f64) -> String {
    format!(
        r#"SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 4.1;
Mesh.ElementOrder = 1;
Mesh.MeshSizeMin = {mesh_size};
Mesh.MeshSizeMax = {mesh_size};

Box(1) = {{-0.2, -0.2, -0.2, 0.7, 0.7, 0.7}};
Box(2) = {{0.0, 0.0, 0.0, 0.294, 0.294, 0.019}};
Box(3) = {{0.018, 0.018, -0.001, 0.108, 0.108, 0.021}};
BooleanDifference{{ Volume{{2}}; Delete; }}{{ Volume{{3}}; Delete; }}

Box(10) = {{0.094, 0.000, 0.049, 0.050, 0.200, 0.100}};
Box(11) = {{0.244, 0.000, 0.049, 0.050, 0.200, 0.100}};
Box(12) = {{0.144, 0.000, 0.049, 0.100, 0.050, 0.100}};
Box(13) = {{0.144, 0.150, 0.049, 0.100, 0.050, 0.100}};

BooleanFragments{{ Volume{{1}}; Delete; }}{{ Volume{{2,10,11,12,13}}; Delete; }}
"#
    )
}

fn validate_config(config: &Team7Config) -> Result<(), String> {
    if !config.frequency_hz.is_finite() || config.frequency_hz <= 0.0 {
        return Err("frequency_hz must be finite and positive".to_string());
    }
    if !config.aluminum_sigma.is_finite() || config.aluminum_sigma <= 0.0 {
        return Err("aluminum_sigma must be finite and positive".to_string());
    }
    if !config.background_sigma.is_finite() || config.background_sigma <= 0.0 {
        return Err("background_sigma must be finite and positive".to_string());
    }
    if !config.pde_residual_std.is_finite() || config.pde_residual_std <= 0.0 {
        return Err("pde_residual_std must be finite and positive".to_string());
    }
    if !config.b_observation_std_tesla.is_finite() || config.b_observation_std_tesla <= 0.0 {
        return Err("b_observation_std_tesla must be finite and positive".to_string());
    }
    if !config.jy_observation_std.is_finite() || config.jy_observation_std <= 0.0 {
        return Err("jy_observation_std must be finite and positive".to_string());
    }
    validate_source_model(config.source_model)?;
    if !config.matern_kappa.is_finite() || config.matern_kappa <= 0.0 {
        return Err("matern_kappa must be finite and positive".to_string());
    }
    if !config.matern_tau.is_finite() || config.matern_tau <= 0.0 {
        return Err("matern_tau must be finite and positive".to_string());
    }
    validate_prior_config(config.prior)?;
    Ok(())
}

fn validate_source_model(source_model: Team7SourceModel) -> Result<(), String> {
    match source_model {
        Team7SourceModel::Deterministic => Ok(()),
        Team7SourceModel::HarmonicCalibration {
            log_alpha_std,
            phase_std_rad,
            shape_relative_std,
            shape_modes,
            guard_z_score,
            min_alpha,
        } => {
            if !log_alpha_std.is_finite() || log_alpha_std <= 0.0 {
                return Err("source_model.log_alpha_std must be finite and positive".to_string());
            }
            if !phase_std_rad.is_finite() || phase_std_rad <= 0.0 {
                return Err("source_model.phase_std_rad must be finite and positive".to_string());
            }
            if !shape_relative_std.is_finite() || shape_relative_std <= 0.0 {
                return Err(
                    "source_model.shape_relative_std must be finite and positive".to_string(),
                );
            }
            if shape_modes > 2 {
                return Err(
                    "source_model.shape_modes currently supports at most 2 modes".to_string(),
                );
            }
            if !guard_z_score.is_finite() || guard_z_score <= 0.0 {
                return Err("source_model.guard_z_score must be finite and positive".to_string());
            }
            if !min_alpha.is_finite() || min_alpha <= 0.0 {
                return Err("source_model.min_alpha must be finite and positive".to_string());
            }
            Ok(())
        }
    }
}

fn source_linearization_parameters(
    config: &Team7Config,
    source_linearization: Option<&[f64]>,
) -> Result<Option<Vec<f64>>, String> {
    let Team7SourceModel::HarmonicCalibration { shape_modes, .. } = config.source_model else {
        return Ok(None);
    };
    let dimension = 2 + shape_modes;
    match source_linearization {
        Some(values) => {
            if values.len() != dimension {
                return Err(format!(
                    "TEAM 7 source calibration linearization must have dimension {dimension}, got {}",
                    values.len()
                ));
            }
            Ok(Some(values.to_vec()))
        }
        None => Ok(Some(vec![0.0; dimension])),
    }
}

fn validate_prior_config(prior: Team7PriorConfig) -> Result<(), String> {
    if !prior.physical_weight.is_finite() || prior.physical_weight < 0.0 {
        return Err("prior.physical_weight must be finite and non-negative".to_string());
    }
    let stabilizer_weight = match prior.matern_stabilizer {
        Team7MaternStabilizer::Disabled => 0.0,
        Team7MaternStabilizer::Absolute(value)
        | Team7MaternStabilizer::RelativeToPhysicalDiagonal(value) => {
            if !value.is_finite() || value < 0.0 {
                return Err(
                    "prior.matern_stabilizer weight must be finite and non-negative".to_string(),
                );
            }
            value
        }
    };
    if prior.physical_weight == 0.0 && stabilizer_weight == 0.0 {
        return Err("at least one TEAM 7 prior component weight must be positive".to_string());
    }
    Ok(())
}

fn build_team7_prior_mean(
    mode: Team7PriorMeanMode,
    maxwell_operator: &SparseTripletMatrix,
    nominal_residual_bias: &[f64],
) -> Result<Vec<f64>, String> {
    match mode {
        Team7PriorMeanMode::Zero => Ok(vec![0.0; maxwell_operator.ncols()]),
        Team7PriorMeanMode::DeterministicNominal => {
            deterministic_nominal_solution(maxwell_operator, nominal_residual_bias)
        }
    }
}

fn build_outer_boundary(topology: &Complex, coords: &MeshCoords) -> EssentialBoundarySpec {
    let dofs = boundary_simplices_where_barycenter(topology, coords, 1, |point| {
        is_outer_boundary_point(point)
    });
    EssentialBoundarySpec {
        state: dofs
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
        auxiliary: Vec::new(),
    }
}

pub fn is_outer_boundary_point(point: CoordRef<'_>) -> bool {
    (point[0] - TEAM7_AIRBOX_MIN).abs() < 1e-9
        || (point[0] - TEAM7_AIRBOX_MAX).abs() < 1e-9
        || (point[1] - TEAM7_AIRBOX_MIN).abs() < 1e-9
        || (point[1] - TEAM7_AIRBOX_MAX).abs() < 1e-9
        || (point[2] - TEAM7_AIRBOX_MIN).abs() < 1e-9
        || (point[2] - TEAM7_AIRBOX_MAX).abs() < 1e-9
}

pub fn is_plate_point(point: CoordRef<'_>) -> bool {
    let in_plate_box = in_closed(point[0], 0.0, 0.294)
        && in_closed(point[1], 0.0, 0.294)
        && in_closed(point[2], 0.0, 0.019);
    let in_hole = in_closed(point[0], 0.018, 0.126) && in_closed(point[1], 0.018, 0.126);
    in_plate_box && !in_hole
}

pub fn is_coil_point(point: CoordRef<'_>) -> bool {
    if !in_closed(point[2], 0.049, 0.149) {
        return false;
    }
    let dx = point[0] - clamp(point[0], 0.144, 0.244);
    let dy = point[1] - clamp(point[1], 0.050, 0.150);
    let dist = (dx * dx + dy * dy).sqrt();
    in_closed(dist, 0.025, 0.050)
}

fn in_closed(value: f64, min: f64, max: f64) -> bool {
    value >= min - 1e-12 && value <= max + 1e-12
}

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

#[derive(Debug, Clone, Copy)]
enum Team7SourceDensityMode {
    Nominal,
    XBalance,
    YBalance,
}

impl Team7SourceDensityMode {
    fn shape_mode(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::XBalance),
            1 => Some(Self::YBalance),
            _ => None,
        }
    }

    fn weight(self, point: CoordRef<'_>) -> f64 {
        match self {
            Self::Nominal => 1.0,
            Self::XBalance => (point[0] - 0.194) / 0.1,
            Self::YBalance => (point[1] - 0.100) / 0.1,
        }
    }
}

fn team7_current_density(turns: f64) -> DiffFormClosure {
    team7_current_density_mode(turns, Team7SourceDensityMode::Nominal)
}

fn team7_current_density_mode(turns: f64, mode: Team7SourceDensityMode) -> DiffFormClosure {
    DiffFormClosure::one_form(
        move |point| {
            if !is_coil_point(point) {
                return FeecVector::zeros(3);
            }
            let projx = clamp(point[0], 0.144, 0.244);
            let projy = clamp(point[1], 0.050, 0.150);
            let tau_x = projy - point[1];
            let tau_y = point[0] - projx;
            let norm = (tau_x * tau_x + tau_y * tau_y).sqrt();
            if norm < OBS_EPS {
                return FeecVector::zeros(3);
            }
            let scale = -turns / (0.025 * 0.100);
            let weight = mode.weight(point);
            FeecVector::from_column_slice(&[
                weight * scale * tau_x / norm,
                weight * scale * tau_y / norm,
                0.0,
            ])
        },
        3,
    )
}

fn assemble_team7_source(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    turns: f64,
) -> FeecVector {
    let source = team7_current_density(turns);
    assemble_galvec(topology, metric, SourceElVec::new(&source, coords, None))
}

fn build_source_shape_modes(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    eddy: &formoniq::problems::eddy_current::ReducedEddyCurrent1FormSystem,
    nominal_source: &FeecVector,
    config: &Team7Config,
) -> Result<Vec<FeecVector>, String> {
    let shape_modes = match config.source_model {
        Team7SourceModel::HarmonicCalibration { shape_modes, .. } => shape_modes,
        Team7SourceModel::Deterministic => 0,
    };
    let weight = eddy.state_mass_inverse.clone();
    let mut modes = Vec::with_capacity(shape_modes);
    for index in 0..shape_modes {
        let mode = Team7SourceDensityMode::shape_mode(index)
            .ok_or_else(|| format!("unsupported TEAM 7 source shape mode index {index}"))?;
        let source = team7_current_density_mode(config.turns, mode);
        let full = assemble_galvec(topology, metric, SourceElVec::new(&source, coords, None));
        let reduced = reduce_eddy_current_source(&eddy.layout, &full)?;
        modes.push(orthogonalized_source_shape_mode(
            nominal_source,
            &reduced,
            &weight,
        )?);
    }
    Ok(modes)
}

fn orthogonalized_source_shape_mode(
    nominal_source: &FeecVector,
    raw_mode: &FeecVector,
    weight: &FeecCsr,
) -> Result<FeecVector, String> {
    let nominal_norm_sq = weighted_dot(weight, nominal_source, nominal_source);
    if nominal_norm_sq <= 0.0 || !nominal_norm_sq.is_finite() {
        return Err("nominal source has non-positive weighted norm".to_string());
    }
    let projection = weighted_dot(weight, nominal_source, raw_mode) / nominal_norm_sq;
    let mut mode = raw_mode.clone();
    for index in 0..mode.len() {
        mode[index] -= projection * nominal_source[index];
    }
    let mode_norm_sq = weighted_dot(weight, &mode, &mode);
    if mode_norm_sq <= 0.0 || !mode_norm_sq.is_finite() {
        return Err("orthogonalized source shape mode has non-positive weighted norm".to_string());
    }
    let scale = (nominal_norm_sq / mode_norm_sq).sqrt();
    for value in mode.iter_mut() {
        *value *= scale;
    }
    Ok(mode)
}

fn weighted_dot(weight: &FeecCsr, lhs: &FeecVector, rhs: &FeecVector) -> f64 {
    let mut weighted_rhs = vec![0.0; weight.nrows()];
    for (row, col, value) in weight.triplet_iter() {
        weighted_rhs[row] += *value * rhs[col];
    }
    lhs.iter()
        .zip(weighted_rhs.iter())
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
fn build_doubled_reduced_system(
    eddy: &formoniq::problems::eddy_current::ReducedEddyCurrent1FormSystem,
    source_reduced: &FeecVector,
    omega: f64,
) -> Result<ReducedLinearPdeAssembly, String> {
    build_doubled_reduced_system_with_source_linearization(eddy, source_reduced, &[], omega, None)
}

fn build_doubled_reduced_system_with_source_linearization(
    eddy: &formoniq::problems::eddy_current::ReducedEddyCurrent1FormSystem,
    source_reduced: &FeecVector,
    shape_modes: &[FeecVector],
    omega: f64,
    source_linearization: Option<&[f64]>,
) -> Result<ReducedLinearPdeAssembly, String> {
    let k = eddy.curl_curl.clone();
    let c = eddy.conductivity_mass.clone();
    let operator = build_real_eddy_block_operator(&k, &c, omega)?;
    let n = eddy.reduced_dimension();
    if source_reduced.len() != n {
        return Err(format!(
            "source length {} must match reduced dimension {n}",
            source_reduced.len()
        ));
    }
    for mode in shape_modes {
        if mode.len() != n {
            return Err(format!(
                "source shape mode length {} must match reduced dimension {n}",
                mode.len()
            ));
        }
    }
    let residual_bias = source_linearized_residual_bias(
        eddy,
        source_reduced,
        shape_modes,
        omega,
        source_linearization,
    )?;

    Ok(ReducedLinearPdeAssembly {
        operator,
        residual_bias: FeecVector::from_vec(residual_bias),
        state_mass: core_triplet_to_feec_csr(&block_diag_scaled(
            &[
                csr_to_triplet(&eddy.state_mass),
                csr_to_triplet(&eddy.state_mass),
            ],
            1.0,
        )),
        state_mass_inverse: Some(core_triplet_to_feec_csr(&block_diag_scaled(
            &[
                csr_to_triplet(&eddy.state_mass_inverse),
                csr_to_triplet(&eddy.state_mass_inverse),
            ],
            1.0,
        ))),
        layout: doubled_state_layout(&eddy.layout),
        forcing_operator: core_triplet_to_feec_csr(&identity_triplet(2 * n, -1.0)),
        neumann_operator: core_triplet_to_feec_csr(&identity_triplet(2 * n, -1.0)),
    })
}

fn source_linearized_residual_bias(
    eddy: &formoniq::problems::eddy_current::ReducedEddyCurrent1FormSystem,
    source_reduced: &FeecVector,
    shape_modes: &[FeecVector],
    omega: f64,
    source_linearization: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = eddy.reduced_dimension();
    let mut residual_bias = Vec::with_capacity(2 * n);
    let Some(parameters) = source_linearization else {
        for i in 0..n {
            residual_bias.push(eddy.curl_curl_fixed_bias[i] - source_reduced[i]);
        }
        for i in 0..n {
            residual_bias.push(omega * eddy.conductivity_fixed_bias[i]);
        }
        return Ok(residual_bias);
    };
    if parameters.len() != 2 + shape_modes.len() {
        return Err(format!(
            "source linearization must have dimension {}, got {}",
            2 + shape_modes.len(),
            parameters.len()
        ));
    }

    let eta = parameters[0];
    let phase = parameters[1];
    let alpha = eta.exp();
    let cos_phase = phase.cos();
    let sin_phase = phase.sin();
    let tangent = source_calibration_tangent(source_reduced, shape_modes, parameters)?;

    for i in 0..n {
        let mut source_at = alpha * cos_phase * source_reduced[i];
        for (mode_index, mode) in shape_modes.iter().enumerate() {
            source_at += parameters[2 + mode_index] * mode[i];
        }
        let tangent_at = tangent[0][i] * eta + tangent[1][i] * phase;
        let shape_tangent_at = shape_modes
            .iter()
            .enumerate()
            .map(|(mode_index, _)| tangent[2 + mode_index][i] * parameters[2 + mode_index])
            .sum::<f64>();
        residual_bias
            .push(eddy.curl_curl_fixed_bias[i] - source_at - tangent_at - shape_tangent_at);
    }
    for i in 0..n {
        let source_at = alpha * sin_phase * source_reduced[i];
        let tangent_at = tangent[0][n + i] * eta + tangent[1][n + i] * phase;
        residual_bias.push(omega * eddy.conductivity_fixed_bias[i] - source_at - tangent_at);
    }
    Ok(residual_bias)
}

fn deterministic_nominal_solution(
    maxwell_operator: &SparseTripletMatrix,
    nominal_residual_bias: &[f64],
) -> Result<Vec<f64>, String> {
    if maxwell_operator.nrows() != maxwell_operator.ncols() {
        return Err(format!(
            "deterministic TEAM 7 operator must be square, got {}x{}",
            maxwell_operator.nrows(),
            maxwell_operator.ncols()
        ));
    }
    if nominal_residual_bias.len() != maxwell_operator.nrows() {
        return Err(format!(
            "nominal residual bias length {} must match operator dimension {}",
            nominal_residual_bias.len(),
            maxwell_operator.nrows()
        ));
    }
    let operator = core_triplet_to_feec_csr(maxwell_operator);
    let rhs = FeecVector::from_iterator(
        nominal_residual_bias.len(),
        nominal_residual_bias.iter().map(|value| -*value),
    );
    let factor = SparseLuFactor::factorize(&feec_csr_to_gmrf(&operator))
        .map_err(|err| format!("failed to factor deterministic TEAM 7 Maxwell operator: {err}"))?;
    let solution = factor
        .solve(&feec_vec_to_gmrf(&rhs))
        .map_err(|err| format!("failed to solve deterministic TEAM 7 Maxwell problem: {err}"))?;
    Ok(gmrf_vec_to_feec(&solution).iter().copied().collect())
}

fn build_real_eddy_block_operator(k: &FeecCsr, c: &FeecCsr, omega: f64) -> Result<FeecCsr, String> {
    if k.nrows() != k.ncols() || c.nrows() != c.ncols() || k.nrows() != c.nrows() {
        return Err(format!(
            "K and C must be square with matching dimensions, got K {}x{} and C {}x{}",
            k.nrows(),
            k.ncols(),
            c.nrows(),
            c.ncols()
        ));
    }
    let n = k.nrows();
    let mut coo = FeecCoo::new(2 * n, 2 * n);
    for (row, col, value) in k.triplet_iter() {
        coo.push(row, col, *value);
        coo.push(n + row, n + col, *value);
    }
    for (row, col, value) in c.triplet_iter() {
        coo.push(row, n + col, -omega * *value);
        coo.push(n + row, col, omega * *value);
    }
    Ok(FeecCsr::from(&coo))
}

fn doubled_state_layout(single: &DofLayout) -> DofLayout {
    let n = single.full_dimension;
    let mut active_dofs = Vec::with_capacity(2 * single.active_dofs.len());
    active_dofs.extend(single.active_dofs.iter().copied());
    active_dofs.extend(single.active_dofs.iter().map(|index| n + *index));

    let mut prescribed_dofs = Vec::with_capacity(2 * single.prescribed_dofs.len());
    prescribed_dofs.extend(single.prescribed_dofs.iter().copied());
    prescribed_dofs.extend(single.prescribed_dofs.iter().map(|fixed| PrescribedDof {
        index: n + fixed.index,
        value: 0.0,
    }));
    DofLayout::new(2 * n, active_dofs, prescribed_dofs)
}

fn build_team7_state_prior(
    topology: &Complex,
    metric: &MeshLengths,
    single_layout: &DofLayout,
    maxwell_operator: &SparseTripletMatrix,
    forcing_precision: &SparseTripletMatrix,
    mean: Vec<f64>,
    config: &Team7Config,
) -> Result<GaussianPriorSpec, String> {
    let hodge = build_hodge_laplacian_1form(topology, metric);
    let full_precision = build_matern_precision_1form(
        topology,
        metric,
        &hodge,
        Matern1FormConfig {
            kappa: config.matern_kappa,
            tau: config.matern_tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    );
    let reduced = reduce_square_with_layout(&full_precision, single_layout)?;
    let matern_precision = core_triplet_to_feec_csr(&block_diag_scaled(
        &[csr_to_triplet(&reduced), csr_to_triplet(&reduced)],
        1.0,
    ));
    let maxwell_operator = core_triplet_to_feec_csr(maxwell_operator);
    let forcing_precision = core_triplet_to_feec_csr(forcing_precision);
    let physical_precision =
        build_physical_complex_prior_precision(&maxwell_operator, &forcing_precision)?;
    let precision =
        combine_team7_prior_precisions(&physical_precision, &matern_precision, config.prior)?;
    let expected_dimension = 2 * single_layout.reduced_dimension();
    if mean.len() != expected_dimension {
        return Err(format!(
            "TEAM 7 prior mean length {} must match doubled reduced dimension {expected_dimension}",
            mean.len()
        ));
    }
    Ok(GaussianPriorSpec {
        mean,
        precision: csr_to_triplet(&precision),
    })
}

fn build_physical_complex_prior_precision(
    maxwell_operator: &FeecCsr,
    forcing_precision: &FeecCsr,
) -> Result<FeecCsr, String> {
    if maxwell_operator.nrows() != forcing_precision.nrows()
        || maxwell_operator.nrows() != forcing_precision.ncols()
    {
        return Err(format!(
            "forcing precision must be square with dimension {}, got {}x{}",
            maxwell_operator.nrows(),
            forcing_precision.nrows(),
            forcing_precision.ncols()
        ));
    }
    let weighted = forcing_precision * maxwell_operator;
    let transpose = maxwell_operator.transpose();
    Ok(&transpose * &weighted)
}

fn combine_team7_prior_precisions(
    physical_precision: &FeecCsr,
    matern_precision: &FeecCsr,
    prior: Team7PriorConfig,
) -> Result<FeecCsr, String> {
    if physical_precision.nrows() != physical_precision.ncols()
        || matern_precision.nrows() != matern_precision.ncols()
        || physical_precision.nrows() != matern_precision.nrows()
    {
        return Err(format!(
            "TEAM 7 prior components must be square with matching dimensions, got physical {}x{} and Matérn {}x{}",
            physical_precision.nrows(),
            physical_precision.ncols(),
            matern_precision.nrows(),
            matern_precision.ncols()
        ));
    }
    let matern_weight = resolve_matern_stabilizer_weight(
        prior.matern_stabilizer,
        physical_precision,
        matern_precision,
    )?;
    Ok(add_scaled_sparse(
        physical_precision,
        prior.physical_weight,
        matern_precision,
        matern_weight,
    ))
}

fn resolve_matern_stabilizer_weight(
    stabilizer: Team7MaternStabilizer,
    physical_precision: &FeecCsr,
    matern_precision: &FeecCsr,
) -> Result<f64, String> {
    match stabilizer {
        Team7MaternStabilizer::Disabled => Ok(0.0),
        Team7MaternStabilizer::Absolute(value) => Ok(value),
        Team7MaternStabilizer::RelativeToPhysicalDiagonal(relative) => {
            if relative == 0.0 {
                return Ok(0.0);
            }
            let physical_median = positive_diagonal_median(physical_precision)
                .ok_or_else(|| "physical prior has no positive diagonal entries".to_string())?;
            let matern_median = positive_diagonal_median(matern_precision)
                .ok_or_else(|| "Matérn prior has no positive diagonal entries".to_string())?;
            Ok(relative * physical_median / matern_median)
        }
    }
}

fn source_calibration_input(
    source_reduced: &FeecVector,
    shape_modes: &[FeecVector],
    linearization: &[f64],
    log_alpha_std: f64,
    phase_std_rad: f64,
    shape_relative_std: f64,
) -> Result<LinearUncertainInputSpec, String> {
    let n = source_reduced.len();
    let dimension = 2 + shape_modes.len();
    if linearization.len() != dimension {
        return Err(format!(
            "source calibration linearization must have dimension {dimension}, got {}",
            linearization.len()
        ));
    }
    let tangent = source_calibration_tangent(source_reduced, shape_modes, linearization)?;
    let mut operator = SparseTripletMatrix::new(2 * n, dimension);
    // Tangent source model for alpha exp(i phi) j about the current
    // (eta, phi) point. Shape modes remain exactly linear in-phase terms.
    for (col, column) in tangent.iter().enumerate() {
        for (row, value) in column.iter().copied().enumerate() {
            if value != 0.0 {
                operator.push(row, col, value);
            }
        }
    }
    let mut precision = SparseTripletMatrix::new(dimension, dimension);
    precision.push(0, 0, 1.0 / log_alpha_std.powi(2));
    precision.push(1, 1, 1.0 / phase_std_rad.powi(2));
    for index in 0..shape_modes.len() {
        precision.push(2 + index, 2 + index, 1.0 / shape_relative_std.powi(2));
    }
    Ok(LinearUncertainInputSpec {
        name: TEAM7_SOURCE_CALIBRATION_INPUT_NAME.to_string(),
        operator,
        prior: GaussianPriorSpec {
            mean: vec![0.0; dimension],
            precision,
        },
        preference: RepresentationPreference::ForceLatent,
        collapsed_precision: None,
    })
}

fn source_calibration_tangent(
    source_reduced: &FeecVector,
    shape_modes: &[FeecVector],
    linearization: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    let n = source_reduced.len();
    if linearization.len() != 2 + shape_modes.len() {
        return Err(format!(
            "source calibration linearization must have dimension {}, got {}",
            2 + shape_modes.len(),
            linearization.len()
        ));
    }
    for mode in shape_modes {
        if mode.len() != n {
            return Err(format!(
                "source shape mode length {} must match source length {n}",
                mode.len()
            ));
        }
    }

    let eta = linearization[0];
    let phase = linearization[1];
    let alpha = eta.exp();
    let cos_phase = phase.cos();
    let sin_phase = phase.sin();
    let mut columns = Vec::with_capacity(2 + shape_modes.len());
    let mut eta_column = vec![0.0; 2 * n];
    let mut phase_column = vec![0.0; 2 * n];
    for row in 0..n {
        let source = source_reduced[row];
        eta_column[row] = -alpha * cos_phase * source;
        eta_column[n + row] = -alpha * sin_phase * source;
        phase_column[row] = alpha * sin_phase * source;
        phase_column[n + row] = -alpha * cos_phase * source;
    }
    columns.push(eta_column);
    columns.push(phase_column);
    for mode in shape_modes {
        let mut column = vec![0.0; 2 * n];
        for row in 0..n {
            column[row] = -mode[row];
        }
        columns.push(column);
    }
    Ok(columns)
}

fn build_source_calibration_report(
    config: &Team7Config,
    posterior: &LinearPdeUqResult,
) -> Result<Option<Team7SourceCalibrationReport>, String> {
    let Team7SourceModel::HarmonicCalibration {
        log_alpha_std,
        phase_std_rad,
        shape_relative_std,
        shape_modes,
        guard_z_score,
        min_alpha,
    } = config.source_model
    else {
        return Ok(None);
    };
    let input = posterior
        .latent_inputs
        .iter()
        .find(|input| input.name == TEAM7_SOURCE_CALIBRATION_INPUT_NAME)
        .ok_or_else(|| "TEAM 7 source calibration posterior input is missing".to_string())?;
    let expected_dimension = 2 + shape_modes;
    if input.mean.len() != expected_dimension || input.variance.len() != expected_dimension {
        return Err(format!(
            "TEAM 7 source calibration posterior must have dimension {expected_dimension}, got mean dimension {} and variance dimension {}",
            input.mean.len(),
            input.variance.len()
        ));
    }
    Ok(Some(source_calibration_report_from_moments(
        &input.mean,
        &input.variance,
        log_alpha_std,
        phase_std_rad,
        shape_relative_std,
        guard_z_score,
        min_alpha,
    )))
}

fn source_calibration_report_from_moments(
    means: &[f64],
    variances: &[f64],
    log_alpha_prior_std: f64,
    phase_prior_std_rad: f64,
    shape_relative_std: f64,
    guard_z_score: f64,
    min_alpha: f64,
) -> Team7SourceCalibrationReport {
    let log_alpha_mean = means[0];
    let log_alpha_std = if variances[0] >= 0.0 {
        variances[0].sqrt()
    } else {
        f64::NAN
    };
    let phase_mean_rad = means[1];
    let phase_std_rad = if variances[1] >= 0.0 {
        variances[1].sqrt()
    } else {
        f64::NAN
    };
    let shape_mode_means = means[2..].to_vec();
    let shape_mode_stds = variances[2..]
        .iter()
        .map(|variance| {
            if *variance >= 0.0 {
                variance.sqrt()
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    let alpha_mean = log_alpha_mean.exp();
    let log_alpha_z_score = log_alpha_mean.abs() / log_alpha_prior_std;
    let phase_z_score = phase_mean_rad.abs() / phase_prior_std_rad;
    let max_shape_z_score = shape_mode_means
        .iter()
        .map(|value| value.abs() / shape_relative_std)
        .fold(0.0, f64::max);
    let mut reasons = Vec::new();
    if log_alpha_z_score > guard_z_score {
        reasons.push(format!(
            "log-amplitude z-score {log_alpha_z_score:.6e} exceeds guard {guard_z_score:.6e}"
        ));
    }
    if phase_z_score > guard_z_score {
        reasons.push(format!(
            "phase z-score {phase_z_score:.6e} exceeds guard {guard_z_score:.6e}"
        ));
    }
    if max_shape_z_score > guard_z_score {
        reasons.push(format!(
            "shape-mode z-score {max_shape_z_score:.6e} exceeds guard {guard_z_score:.6e}"
        ));
    }
    if alpha_mean < min_alpha {
        reasons.push(format!(
            "source alpha {alpha_mean:.6e} is below minimum {min_alpha:.6e}"
        ));
    }
    Team7SourceCalibrationReport {
        log_alpha_mean,
        log_alpha_std,
        alpha_mean,
        phase_mean_rad,
        phase_std_rad,
        log_alpha_z_score,
        phase_z_score,
        shape_mode_means,
        shape_mode_stds,
        max_shape_z_score,
        accepted: reasons.is_empty(),
        rejection_reason: if reasons.is_empty() {
            None
        } else {
            Some(reasons.join("; "))
        },
    }
}

fn build_team7_observation_rows(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
    omega: f64,
    config: &Team7Config,
) -> Result<Vec<Team7ObservationRow>, String> {
    let edge_count = topology.skeleton(1).len();
    if layout.full_dimension != 2 * edge_count {
        return Err("TEAM 7 observation layout must be doubled edge layout".to_string());
    }
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    let mut rows = Vec::new();
    add_bz_line(
        &mut rows,
        topology,
        coords,
        &d1,
        edge_count,
        "A1B1",
        0.072,
        BZ_A1B1_REF,
    )?;
    add_bz_line(
        &mut rows,
        topology,
        coords,
        &d1,
        edge_count,
        "A2B2",
        0.144,
        BZ_A2B2_REF,
    )?;
    // The NGSolve notebook evaluates the top line against the array named
    // Jy_A4B4 and the bottom line against Jy_A3B3; we keep that benchmark
    // convention so the CSV reproduces the reference comparison.
    add_jy_line(
        &mut rows,
        topology,
        coords,
        edge_count,
        "A3B3_top",
        0.072,
        0.019 - 1e-5,
        JY_A4B4_REF,
        omega,
        config.aluminum_sigma,
    )?;
    add_jy_line(
        &mut rows,
        topology,
        coords,
        edge_count,
        "A4B4_bottom",
        0.072,
        1e-5,
        JY_A3B3_REF,
        omega,
        config.aluminum_sigma,
    )?;
    Ok(rows)
}

fn add_bz_line(
    rows: &mut Vec<Team7ObservationRow>,
    topology: &Complex,
    coords: &MeshCoords,
    d1: &FeecCsr,
    edge_count: usize,
    line: &str,
    y: f64,
    reference: [(f64, f64); 17],
) -> Result<(), String> {
    for (index, x) in XI_MSM.iter().copied().enumerate() {
        let point = [x, y, 0.034];
        let row = point_bz_edge_row(topology, coords, d1, &point)?;
        let (obs_r, obs_i) = team7_b_reference_to_tesla(reference[index]);
        rows.push(Team7ObservationRow {
            family: "Bz".to_string(),
            line: line.to_string(),
            component: "real".to_string(),
            point,
            observed: obs_r,
            row: lift_row_to_doubled(&row, 0),
        });
        rows.push(Team7ObservationRow {
            family: "Bz".to_string(),
            line: line.to_string(),
            component: "imag".to_string(),
            point,
            observed: obs_i,
            row: lift_row_to_doubled(&row, edge_count),
        });
    }
    Ok(())
}

fn add_jy_line(
    rows: &mut Vec<Team7ObservationRow>,
    topology: &Complex,
    coords: &MeshCoords,
    edge_count: usize,
    line: &str,
    y: f64,
    z: f64,
    reference: [(f64, f64); 17],
    omega: f64,
    aluminum_sigma: f64,
) -> Result<(), String> {
    for (index, x) in XI_MSM.iter().copied().enumerate() {
        let point = [x, y, z];
        let mut row = point_ay_edge_row(topology, coords, &point)?;
        let sigma = if is_plate_point(FeecVector::from_column_slice(&point).as_view()) {
            aluminum_sigma
        } else {
            0.0
        };
        for (_, value) in &mut row {
            *value *= 1e-6 * omega * sigma;
        }
        let (obs_r, obs_i) = reference[index];
        rows.push(Team7ObservationRow {
            family: "Jy".to_string(),
            line: line.to_string(),
            component: "real".to_string(),
            point,
            observed: obs_r,
            row: lift_row_to_doubled(&row, 0),
        });
        rows.push(Team7ObservationRow {
            family: "Jy".to_string(),
            line: line.to_string(),
            component: "imag".to_string(),
            point,
            observed: obs_i,
            row: lift_row_to_doubled(&row, edge_count),
        });
    }
    Ok(())
}

fn point_ay_edge_row(
    topology: &Complex,
    coords: &MeshCoords,
    point: &[f64; 3],
) -> Result<Vec<(usize, f64)>, String> {
    let (cell, point_vec) = find_cell_containing_with_nudges(topology, coords, point)?;
    let cell_coords = cell.coord_simplex(coords);
    let local = cell_coords.global2local(point_vec.as_view());
    let mut accum = BTreeMap::<usize, f64>::new();
    for dof_simp in cell.mesh_subsimps(1) {
        let local_dof = dof_simp.relative_to(&cell);
        let local_value = WhitneyLsf::standard(topology.dim(), local_dof).at_point(local.as_view());
        let ambient = cell_coords.lift_form(&local_value).into_grade1();
        let value = ambient[1];
        if value.abs() > OBS_EPS {
            *accum.entry(dof_simp.kidx()).or_insert(0.0) += value;
        }
    }
    Ok(map_to_row(accum))
}

fn point_bz_edge_row(
    topology: &Complex,
    coords: &MeshCoords,
    d1: &FeecCsr,
    point: &[f64; 3],
) -> Result<Vec<(usize, f64)>, String> {
    let (cell, point_vec) = find_cell_containing_with_nudges(topology, coords, point)?;
    let cell_coords = cell.coord_simplex(coords);
    let local = cell_coords.global2local(point_vec.as_view());
    let mut face_coeff = BTreeMap::<usize, f64>::new();
    for dof_simp in cell.mesh_subsimps(2) {
        let local_dof = dof_simp.relative_to(&cell);
        let local_value = WhitneyLsf::standard(topology.dim(), local_dof).at_point(local.as_view());
        let ambient = cell_coords.lift_form(&local_value);
        let coeffs = ambient.coeffs();
        let value = coeffs[0];
        if value.abs() > OBS_EPS {
            face_coeff.insert(dof_simp.kidx(), value);
        }
    }
    let mut edge_accum = BTreeMap::<usize, f64>::new();
    for (face, edge, incidence) in d1.triplet_iter() {
        if let Some(weight) = face_coeff.get(&face) {
            let value = *weight * *incidence;
            if value.abs() > OBS_EPS {
                *edge_accum.entry(edge).or_insert(0.0) += value;
            }
        }
    }
    Ok(map_to_row(edge_accum))
}

fn find_cell_containing_with_nudges<'a>(
    topology: &'a Complex,
    coords: &MeshCoords,
    point: &[f64; 3],
) -> Result<(SimplexHandle<'a>, FeecVector), String> {
    const NUDGE: f64 = 1e-9;
    let candidates = [
        [0.0, 0.0, 0.0],
        [NUDGE, 0.0, 0.0],
        [-NUDGE, 0.0, 0.0],
        [0.0, NUDGE, 0.0],
        [0.0, -NUDGE, 0.0],
        [0.0, 0.0, NUDGE],
        [0.0, 0.0, -NUDGE],
        [NUDGE, NUDGE, 0.0],
        [NUDGE, 0.0, NUDGE],
    ];
    for delta in candidates {
        let candidate = FeecVector::from_column_slice(&[
            point[0] + delta[0],
            point[1] + delta[1],
            point[2] + delta[2],
        ]);
        if let Some(cell) = coords.find_cell_containing(topology, candidate.as_view()) {
            return Ok((cell, candidate));
        }
    }
    Err(format!("point {:?} is outside the mesh", point))
}

fn map_to_row(accum: BTreeMap<usize, f64>) -> Vec<(usize, f64)> {
    accum
        .into_iter()
        .filter(|(_, value)| value.abs() > OBS_EPS)
        .collect()
}

fn lift_row_to_doubled(row: &[(usize, f64)], offset: usize) -> Vec<(usize, f64)> {
    row.iter()
        .map(|(col, value)| (offset + *col, *value))
        .collect()
}

fn team7_b_reference_to_tesla(reference: (f64, f64)) -> (f64, f64) {
    (-reference.0 * 1e-4, reference.1 * 1e-4)
}

fn group_observation_specs(
    rows: &[Team7ObservationRow],
    full_dimension: usize,
    config: &Team7Config,
) -> Vec<LinearGaussianMeasurementSpec> {
    let mut b_rows = Vec::new();
    let mut b_obs = Vec::new();
    let mut jy_rows = Vec::new();
    let mut jy_obs = Vec::new();
    for row in rows {
        match row.family.as_str() {
            "Bz" => {
                b_rows.push(row.row.clone());
                b_obs.push(row.observed);
            }
            "Jy" => {
                jy_rows.push(row.row.clone());
                jy_obs.push(row.observed);
            }
            _ => {}
        }
    }
    vec![
        LinearGaussianMeasurementSpec {
            name: "team7_bz_line_measurements".to_string(),
            operator: rows_to_triplet(b_rows.len(), full_dimension, &b_rows),
            observations: b_obs,
            bias: vec![0.0; b_rows.len()],
            variance: config.b_observation_std_tesla.powi(2),
        },
        LinearGaussianMeasurementSpec {
            name: "team7_jy_line_measurements".to_string(),
            operator: rows_to_triplet(jy_rows.len(), full_dimension, &jy_rows),
            observations: jy_obs,
            bias: vec![0.0; jy_rows.len()],
            variance: config.jy_observation_std.powi(2),
        },
    ]
}

fn build_derived_quantities(
    topology: &Complex,
    observation_rows: &[Team7ObservationRow],
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    let edge_count = topology.skeleton(1).len();
    let face_count = topology.skeleton(2).len();
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    let mut b_real_rows = vec![Vec::new(); face_count];
    let mut b_imag_rows = vec![Vec::new(); face_count];
    for (face, edge, value) in d1.triplet_iter() {
        b_real_rows[face].push((edge, *value));
        b_imag_rows[face].push((edge_count + edge, *value));
    }
    let mut quantities = vec![
        LinearPdeDerivedQuantitySpec {
            name: TEAM7_B_REAL_DERIVED_NAME.to_string(),
            operator: SparseRowOperator::new(2 * edge_count, b_real_rows)
                .map_err(|err| err.to_string())?,
        },
        LinearPdeDerivedQuantitySpec {
            name: TEAM7_B_IMAG_DERIVED_NAME.to_string(),
            operator: SparseRowOperator::new(2 * edge_count, b_imag_rows)
                .map_err(|err| err.to_string())?,
        },
    ];

    let measurement_operator = SparseRowOperator::new(
        2 * edge_count,
        observation_rows.iter().map(|row| row.row.clone()).collect(),
    )
    .map_err(|err| err.to_string())?;
    quantities.push(LinearPdeDerivedQuantitySpec {
        name: "team7_physical_observations".to_string(),
        operator: measurement_operator,
    });
    Ok(quantities)
}

fn evaluate_observation_reports(
    rows: &[Team7ObservationRow],
    posterior: &LinearPdeUqResult,
) -> Vec<Team7ObservationReport> {
    rows.iter()
        .map(|row| {
            let predicted = apply_sparse_row(&row.row, posterior.posterior_mean.as_slice())
                .expect("observation row should match posterior dimension");
            Team7ObservationReport {
                family: row.family.clone(),
                line: row.line.clone(),
                component: row.component.clone(),
                point: row.point,
                observed: row.observed,
                predicted,
                residual: predicted - row.observed,
            }
        })
        .collect()
}

fn compute_rmse(reports: &[Team7ObservationReport]) -> Vec<Team7RmseReport> {
    let mut grouped = BTreeMap::<String, (f64, usize)>::new();
    for report in reports {
        let entry = grouped.entry(report.family.clone()).or_insert((0.0, 0));
        entry.0 += report.residual * report.residual;
        entry.1 += 1;
    }
    grouped
        .into_iter()
        .map(|(family, (sum_sq, count))| Team7RmseReport {
            family,
            rmse: (sum_sq / count.max(1) as f64).sqrt(),
            count,
        })
        .collect()
}

fn write_team7_outputs(
    output_dir: &Path,
    topology: &Complex,
    coords: &MeshCoords,
    assembly: &Team7ProblemAssembly,
    result: &Team7LinearSolveResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir)
        .map_err(|err| format!("failed to create `{}`: {err}", output_dir.display()))?;
    write_observation_csv(output_dir, result)?;
    write_field_outputs(output_dir, topology, coords, assembly, &result.posterior)?;
    Ok(())
}

fn write_observation_csv(output_dir: &Path, result: &Team7LinearSolveResult) -> Result<(), String> {
    let comparison_path = output_dir.join("team7_line_comparison.csv");
    let mut comparison = fs::File::create(&comparison_path)
        .map_err(|err| format!("failed to create `{}`: {err}", comparison_path.display()))?;
    writeln!(
        comparison,
        "family,line,component,x,y,z,observed,predicted,residual"
    )
    .map_err(|err| err.to_string())?;
    for report in &result.observations {
        writeln!(
            comparison,
            "{},{},{},{:.12},{:.12},{:.12},{:.12e},{:.12e},{:.12e}",
            report.family,
            report.line,
            report.component,
            report.point[0],
            report.point[1],
            report.point[2],
            report.observed,
            report.predicted,
            report.residual
        )
        .map_err(|err| err.to_string())?;
    }

    let rmse_path = output_dir.join("team7_rmse.csv");
    let mut rmse = fs::File::create(&rmse_path)
        .map_err(|err| format!("failed to create `{}`: {err}", rmse_path.display()))?;
    writeln!(rmse, "family,count,rmse").map_err(|err| err.to_string())?;
    for report in &result.rmse {
        writeln!(
            rmse,
            "{},{},{:.12e}",
            report.family, report.count, report.rmse
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn write_field_outputs(
    output_dir: &Path,
    topology: &Complex,
    coords: &MeshCoords,
    assembly: &Team7ProblemAssembly,
    posterior: &LinearPdeUqResult,
) -> Result<(), String> {
    let edge_count = assembly.edge_count;
    let mean = &posterior.posterior_mean;
    let var = &posterior.posterior_variance;
    let a_r = Cochain::new(
        1,
        FeecVector::from_vec(mean.as_slice()[0..edge_count].to_vec()),
    );
    let a_i = Cochain::new(
        1,
        FeecVector::from_vec(mean.as_slice()[edge_count..2 * edge_count].to_vec()),
    );
    let var_r = Cochain::new(
        1,
        FeecVector::from_vec(var.as_slice()[0..edge_count].to_vec()),
    );
    let var_i = Cochain::new(
        1,
        FeecVector::from_vec(var.as_slice()[edge_count..2 * edge_count].to_vec()),
    );
    visual_output::write_1cochain_fields(
        output_dir.join("A_edge_fields.vtu"),
        coords,
        topology,
        &[
            ("A_R", &a_r),
            ("A_I", &a_i),
            ("A_R_variance", &var_r),
            ("A_I_variance", &var_i),
        ],
    )
    .map_err(|err| format!("failed to write A VTU: {err}"))?;

    let b_r = a_r.dif(topology);
    let b_i = a_i.dif(topology);
    visual_output::write_cochain(
        output_dir.join("B_R_faces.vtu"),
        coords,
        topology,
        &b_r,
        "B_R",
    )
    .map_err(|err| format!("failed to write B_R face VTU: {err}"))?;
    visual_output::write_cochain(
        output_dir.join("B_I_faces.vtu"),
        coords,
        topology,
        &b_i,
        "B_I",
    )
    .map_err(|err| format!("failed to write B_I face VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_R_vector_field.vtu"),
        coords,
        topology,
        &b_r,
        "B_R",
    )
    .map_err(|err| format!("failed to write B_R vector VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_I_vector_field.vtu"),
        coords,
        topology,
        &b_i,
        "B_I",
    )
    .map_err(|err| format!("failed to write B_I vector VTU: {err}"))?;

    if let Some(b_r_vars) = posterior.derived_variances.get(TEAM7_B_REAL_DERIVED_NAME) {
        let b_var = Cochain::new(2, b_r_vars.posterior_variance.clone());
        visual_output::write_cochain(
            output_dir.join("B_R_marginal_variance.vtu"),
            coords,
            topology,
            &b_var,
            "B_R_variance",
        )
        .map_err(|err| format!("failed to write B_R variance VTU: {err}"))?;
    }
    if let Some(b_i_vars) = posterior.derived_variances.get(TEAM7_B_IMAG_DERIVED_NAME) {
        let b_var = Cochain::new(2, b_i_vars.posterior_variance.clone());
        visual_output::write_cochain(
            output_dir.join("B_I_marginal_variance.vtu"),
            coords,
            topology,
            &b_var,
            "B_I_variance",
        )
        .map_err(|err| format!("failed to write B_I variance VTU: {err}"))?;
    }
    Ok(())
}

fn reduce_square_with_layout(matrix: &FeecCsr, layout: &DofLayout) -> Result<FeecCsr, String> {
    if matrix.nrows() != layout.full_dimension || matrix.ncols() != layout.full_dimension {
        return Err(format!(
            "matrix dimensions {}x{} do not match layout dimension {}",
            matrix.nrows(),
            matrix.ncols(),
            layout.full_dimension
        ));
    }
    let mut reduced_index = vec![None; layout.full_dimension];
    for (position, dof) in layout.active_dofs.iter().copied().enumerate() {
        reduced_index[dof] = Some(position);
    }
    let mut coo = FeecCoo::new(layout.reduced_dimension(), layout.reduced_dimension());
    for (row, col, value) in matrix.triplet_iter() {
        if let (Some(r), Some(c)) = (reduced_index[row], reduced_index[col]) {
            coo.push(r, c, *value);
        }
    }
    Ok(FeecCsr::from(&coo))
}

fn csr_to_triplet(matrix: &FeecCsr) -> SparseTripletMatrix {
    let mut out = SparseTripletMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            out.push(row, col, *value);
        }
    }
    out
}

fn scale_triplet_matrix(matrix: &SparseTripletMatrix, scale: f64) -> SparseTripletMatrix {
    let mut out = SparseTripletMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        let scaled = scale * value;
        if scaled != 0.0 {
            out.push(row, col, scaled);
        }
    }
    out
}

fn add_scaled_sparse(lhs: &FeecCsr, lhs_scale: f64, rhs: &FeecCsr, rhs_scale: f64) -> FeecCsr {
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    let mut coo = FeecCoo::new(lhs.nrows(), lhs.ncols());
    for (row, col, value) in lhs.triplet_iter() {
        let scaled = lhs_scale * *value;
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    for (row, col, value) in rhs.triplet_iter() {
        let scaled = rhs_scale * *value;
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

fn positive_diagonal_median(matrix: &FeecCsr) -> Option<f64> {
    let mut diagonal = vec![0.0; matrix.nrows().min(matrix.ncols())];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    let mut positive = diagonal
        .into_iter()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return None;
    }
    positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = positive.len() / 2;
    if positive.len() % 2 == 1 {
        Some(positive[mid])
    } else {
        Some(0.5 * (positive[mid - 1] + positive[mid]))
    }
}

fn block_diag_scaled(blocks: &[SparseTripletMatrix], scale: f64) -> SparseTripletMatrix {
    let nrows = blocks.iter().map(SparseTripletMatrix::nrows).sum();
    let ncols = blocks.iter().map(SparseTripletMatrix::ncols).sum();
    let mut out = SparseTripletMatrix::new(nrows, ncols);
    let mut row_offset = 0;
    let mut col_offset = 0;
    for block in blocks {
        for (row, col, value) in block.triplet_iter() {
            let scaled = scale * value;
            if scaled != 0.0 {
                out.push(row_offset + row, col_offset + col, scaled);
            }
        }
        row_offset += block.nrows();
        col_offset += block.ncols();
    }
    out
}

fn diagonal_triplet(size: usize, value: f64) -> SparseTripletMatrix {
    let mut matrix = SparseTripletMatrix::new(size, size);
    for index in 0..size {
        if value != 0.0 {
            matrix.push(index, index, value);
        }
    }
    matrix
}

fn identity_triplet(size: usize, value: f64) -> SparseTripletMatrix {
    diagonal_triplet(size, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn dense_entry(matrix: &FeecCsr, row: usize, col: usize) -> f64 {
        matrix
            .triplet_iter()
            .filter(|(r, c, _)| *r == row && *c == col)
            .map(|(_, _, value)| *value)
            .sum()
    }

    fn diagonal_csr(diagonal: &[f64]) -> FeecCsr {
        let mut coo = FeecCoo::new(diagonal.len(), diagonal.len());
        for (index, value) in diagonal.iter().copied().enumerate() {
            coo.push(index, index, value);
        }
        FeecCsr::from(&coo)
    }

    fn dense(matrix: &FeecCsr) -> Vec<Vec<f64>> {
        let mut values = vec![vec![0.0; matrix.ncols()]; matrix.nrows()];
        for (row, col, value) in matrix.triplet_iter() {
            values[row][col] += *value;
        }
        values
    }

    fn dense_mul(lhs: &[Vec<f64>], rhs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = lhs.len();
        let cols = rhs.first().map_or(0, Vec::len);
        let inner = rhs.len();
        let mut out = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for k in 0..inner {
                for j in 0..cols {
                    out[i][j] += lhs[i][k] * rhs[k][j];
                }
            }
        }
        out
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-10 * (1.0 + actual.abs().max(expected.abs())),
            "actual {actual} expected {expected}"
        );
    }

    fn max_abs_diff(lhs: &FeecCsr, rhs: &FeecCsr) -> f64 {
        let lhs = dense(lhs);
        let rhs = dense(rhs);
        lhs.iter()
            .zip(rhs.iter())
            .flat_map(|(lrow, rrow)| lrow.iter().zip(rrow.iter()))
            .map(|(l, r)| (l - r).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn real_block_sign_convention_matches_complex_system() {
        let mut k_coo = FeecCoo::new(2, 2);
        k_coo.push(0, 0, 2.0);
        k_coo.push(1, 1, 3.0);
        let mut c_coo = FeecCoo::new(2, 2);
        c_coo.push(0, 1, 5.0);
        c_coo.push(1, 0, 7.0);
        let block =
            build_real_eddy_block_operator(&FeecCsr::from(&k_coo), &FeecCsr::from(&c_coo), 11.0)
                .unwrap();
        assert_eq!(dense_entry(&block, 0, 0), 2.0);
        assert_eq!(dense_entry(&block, 1, 1), 3.0);
        assert_eq!(dense_entry(&block, 0, 3), -55.0);
        assert_eq!(dense_entry(&block, 3, 0), 77.0);
    }

    #[test]
    fn physical_prior_is_normal_operator_not_maxwell_operator() {
        let mut l0 = FeecCoo::new(2, 2);
        l0.push(0, 0, 2.0);
        l0.push(1, 1, 3.0);
        let mut m0 = FeecCoo::new(2, 2);
        m0.push(0, 1, 5.0);
        m0.push(1, 0, 5.0);
        let b =
            build_real_eddy_block_operator(&FeecCsr::from(&l0), &FeecCsr::from(&m0), 1.0).unwrap();
        let q = build_physical_complex_prior_precision(&b, &diagonal_csr(&[1.0; 4])).unwrap();

        for row in 0..q.nrows() {
            for col in 0..q.ncols() {
                assert_close(dense_entry(&q, row, col), dense_entry(&q, col, row));
            }
        }
        for vector in [
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.2, -1.0, 3.0, 0.5],
            vec![-2.0, 4.0, 1.0, -0.25],
        ] {
            let quadratic = vector
                .iter()
                .enumerate()
                .map(|(i, xi)| {
                    vector
                        .iter()
                        .enumerate()
                        .map(|(j, xj)| xi * dense_entry(&q, i, j) * xj)
                        .sum::<f64>()
                })
                .sum::<f64>();
            assert!(quadratic >= -1e-10, "quadratic form was {quadratic}");
        }
        assert!(max_abs_diff(&q, &b) > 1e-6);
    }

    #[test]
    fn white_forcing_physical_prior_has_noncommuting_off_diagonal_blocks() {
        let mut l0_coo = FeecCoo::new(2, 2);
        l0_coo.push(0, 0, 2.0);
        l0_coo.push(0, 1, 1.0);
        l0_coo.push(1, 0, 1.0);
        l0_coo.push(1, 1, 3.0);
        let mut m0_coo = FeecCoo::new(2, 2);
        m0_coo.push(0, 1, 4.0);
        m0_coo.push(1, 0, 4.0);
        m0_coo.push(1, 1, 5.0);
        let l0 = FeecCsr::from(&l0_coo);
        let m0 = FeecCsr::from(&m0_coo);
        let b = build_real_eddy_block_operator(&l0, &m0, 1.0).unwrap();
        let q = build_physical_complex_prior_precision(&b, &diagonal_csr(&[1.0; 4])).unwrap();
        let l_dense = dense(&l0);
        let m_dense = dense(&m0);
        let ml = dense_mul(&m_dense, &l_dense);
        let lm = dense_mul(&l_dense, &m_dense);

        for row in 0..2 {
            for col in 0..2 {
                assert_close(dense_entry(&q, row, 2 + col), ml[row][col] - lm[row][col]);
                assert_close(dense_entry(&q, 2 + row, col), lm[row][col] - ml[row][col]);
            }
        }
        let mut max_offdiag = 0.0_f64;
        for row in 0..2 {
            for col in 0..2 {
                max_offdiag = max_offdiag.max(dense_entry(&q, row, 2 + col).abs());
            }
        }
        assert!(max_offdiag > 1e-6);
    }

    #[test]
    fn physical_prior_uses_supplied_mass_inverse_weighting() {
        let l0 = diagonal_csr(&[1.0, 1.0]);
        let m0 = diagonal_csr(&[0.0, 0.0]);
        let b = build_real_eddy_block_operator(&l0, &m0, 1.0).unwrap();
        let mass_inverse = diagonal_csr(&[2.0, 3.0, 5.0, 7.0]);
        let q = build_physical_complex_prior_precision(&b, &mass_inverse).unwrap();
        assert!(max_abs_diff(&q, &mass_inverse) <= 1e-12);
    }

    #[test]
    fn prior_component_weights_are_combined_linearly() {
        let q_phys = diagonal_csr(&[2.0, 4.0]);
        let q_matern = diagonal_csr(&[10.0, 20.0]);
        let config = Team7PriorConfig {
            physical_weight: 3.0,
            matern_stabilizer: Team7MaternStabilizer::Absolute(0.25),
        };
        let combined = combine_team7_prior_precisions(&q_phys, &q_matern, config).unwrap();
        assert_close(dense_entry(&combined, 0, 0), 3.0 * 2.0 + 0.25 * 10.0);
        assert_close(dense_entry(&combined, 1, 1), 3.0 * 4.0 + 0.25 * 20.0);
    }

    #[test]
    fn default_relative_matern_stabilizer_resolves_to_positive_weight() {
        let q_phys = diagonal_csr(&[2.0, 4.0]);
        let q_matern = diagonal_csr(&[10.0, 20.0]);
        let weight = resolve_matern_stabilizer_weight(
            Team7PriorConfig::default().matern_stabilizer,
            &q_phys,
            &q_matern,
        )
        .unwrap();
        assert!(weight.is_finite() && weight > 0.0);
        assert_close(weight, 1e-6 * 3.0 / 15.0);
    }

    #[test]
    fn doubled_layout_eliminates_same_real_and_imag_edge_dofs() {
        let single = DofLayout::new(
            4,
            vec![1, 3],
            vec![
                PrescribedDof {
                    index: 0,
                    value: 2.0,
                },
                PrescribedDof {
                    index: 2,
                    value: -1.0,
                },
            ],
        );
        let doubled = doubled_state_layout(&single);
        assert_eq!(doubled.full_dimension, 8);
        assert_eq!(doubled.active_dofs, vec![1, 3, 5, 7]);
        assert_eq!(
            doubled
                .prescribed_dofs
                .iter()
                .map(|dof| (dof.index, dof.value))
                .collect::<Vec<_>>(),
            vec![(0, 2.0), (2, -1.0), (4, 0.0), (6, 0.0)]
        );
    }

    #[test]
    fn team7_reference_conversion_matches_notebook_signs() {
        let (real, imag) = team7_b_reference_to_tesla((12.5, -2.0));
        assert_eq!(real, -12.5e-4);
        assert_eq!(imag, -2.0e-4);
    }

    #[test]
    fn default_source_model_is_deterministic() {
        assert_eq!(Team7SourceModel::default(), Team7SourceModel::Deterministic);
        assert_eq!(
            Team7Config::default().source_model,
            Team7SourceModel::Deterministic
        );
        assert_eq!(
            Team7Config::default().prior_mean,
            Team7PriorMeanMode::DeterministicNominal
        );
    }

    #[test]
    fn deterministic_nominal_prior_mean_solves_reduced_residual_equation() {
        let mut operator = SparseTripletMatrix::new(2, 2);
        operator.push(0, 0, 2.0);
        operator.push(1, 1, 4.0);
        let bias = vec![-6.0, 8.0];
        let mean = deterministic_nominal_solution(&operator, &bias).unwrap();
        assert_close(mean[0], 3.0);
        assert_close(mean[1], -2.0);

        let zero = build_team7_prior_mean(Team7PriorMeanMode::Zero, &operator, &bias).unwrap();
        assert_eq!(zero, vec![0.0, 0.0]);
    }

    #[test]
    fn source_calibration_operator_enters_pde_residual_with_negative_sign() {
        let source = FeecVector::from_column_slice(&[2.0, -3.0]);
        let shape = FeecVector::from_column_slice(&[0.5, -0.25]);
        let input = source_calibration_input(
            &source,
            &[shape],
            &[0.0, 0.0, 0.0],
            TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD,
            TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD,
            TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD,
        )
        .unwrap();
        let entries = input.operator.triplet_iter().collect::<Vec<_>>();
        assert!(entries.contains(&(0, 0, -2.0)));
        assert!(entries.contains(&(1, 0, 3.0)));
        assert!(entries.contains(&(2, 1, -2.0)));
        assert!(entries.contains(&(3, 1, 3.0)));
        assert!(entries.contains(&(0, 2, -0.5)));
        assert!(entries.contains(&(1, 2, 0.25)));
    }

    #[test]
    fn source_calibration_default_precision_matches_two_percent_prior() {
        let source = FeecVector::from_column_slice(&[1.0]);
        let shape = FeecVector::from_column_slice(&[1.0]);
        let input = source_calibration_input(
            &source,
            &[shape],
            &[0.0, 0.0, 0.0],
            TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD,
            TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD,
            TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD,
        )
        .unwrap();
        let precision_entries = input.prior.precision.triplet_iter().collect::<Vec<_>>();
        assert_eq!(precision_entries.len(), 3);
        let log_alpha_precision = precision_entries
            .iter()
            .find(|(row, col, _)| *row == 0 && *col == 0)
            .map(|(_, _, value)| *value)
            .unwrap();
        let phase_precision = precision_entries
            .iter()
            .find(|(row, col, _)| *row == 1 && *col == 1)
            .map(|(_, _, value)| *value)
            .unwrap();
        let shape_precision = precision_entries
            .iter()
            .find(|(row, col, _)| *row == 2 && *col == 2)
            .map(|(_, _, value)| *value)
            .unwrap();
        assert_close(
            log_alpha_precision,
            1.0 / TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD.powi(2),
        );
        assert_close(
            phase_precision,
            1.0 / TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD.powi(2),
        );
        assert_close(
            shape_precision,
            1.0 / TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD.powi(2),
        );
    }

    #[test]
    fn source_calibration_tangent_matches_harmonic_amplitude_phase_derivatives() {
        let source = FeecVector::from_column_slice(&[2.0]);
        let shape = FeecVector::from_column_slice(&[0.5]);
        let tangent =
            source_calibration_tangent(&source, &[shape], &[2.0_f64.ln(), PI / 6.0, -0.1]).unwrap();
        assert_eq!(tangent.len(), 3);
        assert_close(tangent[0][0], -2.0 * (3.0_f64).sqrt());
        assert_close(tangent[0][1], -2.0);
        assert_close(tangent[1][0], 2.0);
        assert_close(tangent[1][1], -2.0 * (3.0_f64).sqrt());
        assert_close(tangent[2][0], -0.5);
        assert_close(tangent[2][1], 0.0);
    }

    #[test]
    fn source_calibration_guard_accepts_small_shift_and_rejects_collapse() {
        let accepted = source_calibration_report_from_moments(
            &[0.02, 0.0],
            &[1.0e-4, 1.0e-4],
            TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD,
            TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD,
            TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD,
            TEAM7_DEFAULT_SOURCE_GUARD_Z_SCORE,
            TEAM7_DEFAULT_SOURCE_MIN_ALPHA,
        );
        assert!(accepted.accepted);
        assert!(accepted.rejection_reason.is_none());

        let rejected = source_calibration_report_from_moments(
            &[-1.0, 0.0],
            &[1.0e-12, 1.0e-12],
            TEAM7_DEFAULT_SOURCE_LOG_ALPHA_STD,
            TEAM7_DEFAULT_SOURCE_PHASE_STD_RAD,
            TEAM7_DEFAULT_SOURCE_SHAPE_RELATIVE_STD,
            TEAM7_DEFAULT_SOURCE_GUARD_Z_SCORE,
            TEAM7_DEFAULT_SOURCE_MIN_ALPHA,
        );
        assert!(!rejected.accepted);
        assert!(rejected
            .rejection_reason
            .as_deref()
            .unwrap_or("")
            .contains("exceeds guard"));
        assert!(rejected
            .rejection_reason
            .as_deref()
            .unwrap_or("")
            .contains("below minimum"));
    }

    #[test]
    fn source_shape_modes_are_orthogonalized_against_nominal_source() {
        let nominal = FeecVector::from_column_slice(&[1.0, 1.0]);
        let raw_mode = FeecVector::from_column_slice(&[2.0, 0.0]);
        let weight = diagonal_csr(&[1.0, 1.0]);
        let mode = orthogonalized_source_shape_mode(&nominal, &raw_mode, &weight).unwrap();
        assert_close(weighted_dot(&weight, &nominal, &mode), 0.0);
        assert_close(
            weighted_dot(&weight, &mode, &mode),
            weighted_dot(&weight, &nominal, &nominal),
        );
    }

    #[test]
    fn point_observation_rows_reconstruct_ay_and_bz_on_unit_tet_mesh() {
        let (topology, coords) = CartesianMeshInfo::new_unit(3, 1).compute_coord_complex();
        let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
        let point = [0.2, 0.2, 0.2];
        let ay = point_ay_edge_row(&topology, &coords, &point).unwrap();
        let bz = point_bz_edge_row(&topology, &coords, &d1, &point).unwrap();
        assert!(!ay.is_empty());
        assert!(!bz.is_empty());
        assert!(ay.iter().all(|(_, value)| value.is_finite()));
        assert!(bz.iter().all(|(_, value)| value.is_finite()));
    }

    #[test]
    fn nonzero_fixed_bias_is_folded_into_doubled_residual() {
        let eddy = formoniq::problems::eddy_current::ReducedEddyCurrent1FormSystem {
            curl_curl: core_triplet_to_feec_csr(&diagonal_triplet(1, 2.0)),
            conductivity_mass: core_triplet_to_feec_csr(&diagonal_triplet(1, 3.0)),
            state_mass: core_triplet_to_feec_csr(&diagonal_triplet(1, 1.0)),
            state_mass_inverse: core_triplet_to_feec_csr(&diagonal_triplet(1, 1.0)),
            layout: DofLayout::new(
                2,
                vec![1],
                vec![PrescribedDof {
                    index: 0,
                    value: 4.0,
                }],
            ),
            curl_curl_fixed_bias: FeecVector::from_vec(vec![8.0]),
            conductivity_fixed_bias: FeecVector::from_vec(vec![12.0]),
        };
        let source = FeecVector::from_column_slice(&[5.0]);
        let system = build_doubled_reduced_system(&eddy, &source, 7.0).unwrap();
        assert_eq!(system.residual_bias.as_slice(), &[3.0, 84.0]);
    }
}
