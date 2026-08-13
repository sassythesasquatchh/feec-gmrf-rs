use crate::visual_output;
use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use ddf::cochain::Cochain;
use ddf::whitney::lsf::WhitneyLsf;
use ddf::ManifoldComplexExt;
use exterior::field::{DiffFormClosure, ExteriorField};
use feg_core::{
    GaussianPriorSpec, LinearGaussianMeasurementSpec, LinearUncertainInputSpec,
    RepresentationPreference, SparseTriplet, SparseTripletMatrix,
};
use feg_gp::{matern_covariance_euclidean, EuclideanMaternConfig};
use feg_infer::{
    core_triplet_to_feec_csr,
    linear_pde::{
        solve_linear_pde_uq_with_config, solve_linear_pde_uq_with_pushforward_covariance,
        solve_linear_pde_uq_with_pushforward_mean_covariance, LinearPdeDerivedMarginalResult,
        LinearPdeDerivedQuantitySpec, LinearPdeJointDerivedQuantitySpec,
        LinearPdeJointMeasurementSpec, LinearPdeLatentDerivedBlockSpec,
        LinearPdeLatentMeasurementBlockSpec, LinearPdePrecisionPolicy,
        LinearPdePushforwardCovarianceResult, LinearPdePushforwardMeanCovarianceResult,
        LinearPdeUqProblem, LinearPdeUqResult, LinearPdeUqSolverConfig, LinearPdeVarianceConfig,
        LinearPdeVarianceMode,
    },
    prior::matern::one_form::feec_csr_to_gmrf,
};
use formoniq::{
    assemble::{self, assemble_galvec, assemble_whitney_projected_sparse_inverse_galmat_weighted},
    io::sample_2form_cell_vectors,
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::{
        hodge_laplace::{self, MixedGalmats},
        reduced_linear::{
            build_reduced_hodge_laplace_1form_system_with_galmats,
            reduce_reduced_hodge_laplace_1form_rhs_with_galmats, ReducedLinearPdeAssembly,
        },
    },
    reduction::{DofLayout, EssentialBoundarySpec, PrescribedDof},
};
use gmrf_core::SparseRowOperator;
use manifold::{
    geometry::{
        coord::{
            mesh::MeshCoords,
            simplex::{barycenter_local, SimplexCoords, SimplexHandleExt},
            CoordRef,
        },
        metric::mesh::MeshLengths,
    },
    topology::{
        complex::Complex,
        handle::{KSimplexIdx, SimplexIdx},
    },
};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    f64::consts::PI,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

const MESH_PATH: &str = "meshes/toroidal_inductor.msh";
const OUT_DIR: &str = "out/examples/toroidal_harmonic_b_recovery";
const FIELD_ONLY_BETA_OUT_DIR: &str = "out/examples/toroidal_harmonic_b_field_only_beta_recovery";
const EMBEDDED_FIELD_OUT_DIR: &str = "out/examples/toroidal_harmonic_b_embedded_field_recovery";
const SOURCE_GENERATED_OUT_DIR: &str = "out/examples/toroidal_source_generated_harmonic_recovery";
const SOURCE_GENERATED_FULL_STATE_OUT_DIR: &str =
    "out/examples/toroidal_source_generated_harmonic_full_state_recovery";
const TOPOLOGY_PUSHFORWARD_OUT_DIR: &str = "out/examples/toroidal_topology_pushforward_uq";
const TOPOLOGY_SPARSE_NOISY_OUT_DIR: &str =
    "out/examples/toroidal_topology_sparse_noisy_prediction";
const TOPOLOGY_GP_BASELINE_OUT_DIR: &str = "out/examples/toroidal_topology_gp_baseline";
const HARMONIC_COUPLING_CALIBRATION_OUT_DIR: &str =
    "out/examples/toroidal_harmonic_coupling_calibration";
const AMPERE_LOOP_OUT_DIR: &str = "out/examples/toroidal_harmonic_b_ampere_loop_recovery";
const FULL_STATE_AMPERE_LOOP_OUT_DIR: &str =
    "out/examples/toroidal_harmonic_b_full_state_ampere_loop_recovery";
const BETA_INPUT_NAME: &str = "beta";
const ALPHA_INPUT_NAME: &str = "alpha";
const COUPLING_INPUT_NAME: &str = "c_H";
const FIELD_ONLY_BETA_STAGE_NAME: &str = "H8_field_only_beta_recovery";
const EMBEDDED_PDE_STAGE_NAME: &str = "E1_embedded_pde_only";
const EMBEDDED_FIELD_BETA_STAGE_NAME: &str = "E2_embedded_field_beta_recovery";
const EMBEDDED_AMPERE_ALPHA_BETA_STAGE_NAME: &str = "E3_embedded_ampere_alpha_beta_recovery";
const SOURCE_GENERATED_PDE_STAGE_NAME: &str = "SG1_pde_only_source_prior";
const SOURCE_GENERATED_FIELD_STAGE_NAME: &str = "SG2_embedded_field_source_recovery";
const SOURCE_GENERATED_AMPERE_STAGE_NAME: &str = "SG3_ampere_loop_sharpening";
const SOURCE_GENERATED_FULL_STATE_PDE_STAGE_NAME: &str = "SGF1_pde_only_source_prior";
const SOURCE_GENERATED_FULL_STATE_FIELD_STAGE_NAME: &str = "SGF2_embedded_field_source_recovery";
const SOURCE_GENERATED_FULL_STATE_AMPERE_STAGE_NAME: &str = "SGF3_ampere_loop_sharpening";
const TOPOLOGY_PUSHFORWARD_PRIOR_STAGE_NAME: &str = "S0_prior_only";
const TOPOLOGY_PUSHFORWARD_PDE_STAGE_NAME: &str = "S1_pde_residual";
const TOPOLOGY_PUSHFORWARD_FIELD_STAGE_NAME: &str = "S2_field_probes";
const TOPOLOGY_PUSHFORWARD_FLUX_STAGE_NAME: &str = "S3_flux_panels";
const TOPOLOGY_PUSHFORWARD_AMPERE_STAGE_NAME: &str = "S4_ampere_loop";
const TOPOLOGY_SPARSE_NOISY_PRIOR_STAGE_NAME: &str = "N0_nominal_prior";
const TOPOLOGY_SPARSE_NOISY_PDE_STAGE_NAME: &str = "N1_nominal_pde_residual";
const TOPOLOGY_SPARSE_NOISY_OBS_STAGE_NAME: &str = "N2_sparse_noisy_physical";
const SPARSE_GP_COMPARISON_PDE_ONLY_STAGE_NAME: &str = "C1_pde_only";
const FLUCTUATION_ALPHA_BETA_STAGE_NAME: &str = "H5_fluctuation_alpha_beta_recovery";
const AMPERE_LOOP_ALPHA_BETA_STAGE_NAME: &str = "H6_ampere_loop_alpha_beta_recovery";
const FULL_STATE_AMPERE_LOOP_ALPHA_BETA_STAGE_NAME: &str =
    "H7_ampere_loop_full_state_alpha_beta_recovery";
const AMPERE_LOOP_SENSOR_TYPE: &str = "ampere_loop";
const AMPERE_LOOP_NAME: &str = "ampere_loop_poloidal_phi0";
const AMPERE_LOOP_RADIUS: f64 = 0.95;
const AMPERE_LOOP_SEGMENTS: usize = 128;
const B_TOTAL_DERIVED_NAME: &str = "B_total";
const B_EXACT_DERIVED_NAME: &str = "B_exact";
const SOURCE_BETA_DERIVED_NAME: &str = "source_beta";
const SOURCE_LINKED_CURRENT_DERIVED_NAME: &str = "source_linked_current";
const SOURCE_HARMONIC_PROJECTION_DERIVED_NAME: &str = "source_harmonic_projection";
const QOI_SOURCE_NAME: &str = "qoi::s";
const QOI_SOURCE_BETA_NAME: &str = "qoi::beta_H";
const QOI_HARMONIC_PROJECTION_NAME: &str = "qoi::eta_H";
const QOI_LINK_FLUX_NAME: &str = "qoi::Phi_link";
const QOI_LOCAL_FLUX_NAME: &str = "qoi::Phi_loc";
const QOI_AMPERE_LOOP_NAME: &str = "qoi::I_gamma";
const QOI_FIELD_X_NAME: &str = "qoi::B_x_xi1";
const QOI_FIELD_Y_NAME: &str = "qoi::B_y_xi1";
const QOI_FIELD_Z_NAME: &str = "qoi::B_z_xi1";
const BRANCH_LINK_FLUX_EXACT_NAME: &str = "branch::Phi_link_exact";
const OBS_DERIVED_PREFIX: &str = "obs::";
const EPS: f64 = 1e-12;
const MIN_OBSERVATION_VARIANCE: f64 = 1e-30;

#[derive(Debug, Clone, Copy)]
struct ToroidalInductorGeometry {
    major_radius: f64,
    core_minor_radius: f64,
    coil_minor_radius: f64,
    box_half_length: f64,
    target_air_cell_size: f64,
}

impl Default for ToroidalInductorGeometry {
    fn default() -> Self {
        Self {
            major_radius: 2.0,
            core_minor_radius: 0.60,
            coil_minor_radius: 0.85,
            box_half_length: 6.0,
            target_air_cell_size: 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalObservationLayout {
    Legacy,
    Embedded,
    TopologyPushforward,
    TopologySparseNoisyPrediction,
}

#[derive(Debug, Clone)]
pub struct ToroidalHarmonicBConfig {
    pub mesh_path: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub pde_variance: f64,
    pub beta_energy_fraction: f64,
    pub beta_prior_std_scale: f64,
    pub source_alpha_true: f64,
    pub source_prior_std: f64,
    pub fluctuation_state_prior_precision_scale: f64,
    pub use_mass_weighted_pde_residual: bool,
    pub normalize_mass_weighted_pde_residual: bool,
    pub mass_weighted_pde_precision_scale: f64,
    pub relative_noise_std: f64,
    pub noise_floor: f64,
    pub observation_layout: ToroidalObservationLayout,
    pub run_alpha_beta_stress: bool,
    pub include_harmonic_projection_observation: bool,
    pub include_ampere_loop_observation: bool,
    pub include_full_field_variance_maps: bool,
    pub sparse_prediction_training_fraction: f64,
    pub sparse_prediction_noise_seed: u64,
    pub sparse_prediction_field_phi_count: usize,
    pub sparse_prediction_field_theta_count: usize,
    pub sparse_prediction_linked_flux_count: usize,
    pub sparse_prediction_local_flux_phi_count: usize,
    pub sparse_prediction_local_flux_theta_count: usize,
    pub sparse_prediction_ampere_loop_count: usize,
    pub solver: LinearPdeUqSolverConfig,
}

impl Default for ToroidalHarmonicBConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from(MESH_PATH),
            output_dir: Some(PathBuf::from(OUT_DIR)),
            pde_variance: 1e-6,
            beta_energy_fraction: 0.05,
            beta_prior_std_scale: 4.0,
            source_alpha_true: 1.15,
            source_prior_std: 0.10,
            fluctuation_state_prior_precision_scale: 1.0,
            use_mass_weighted_pde_residual: false,
            normalize_mass_weighted_pde_residual: false,
            mass_weighted_pde_precision_scale: 1.0,
            relative_noise_std: 0.01,
            noise_floor: 1e-8,
            observation_layout: ToroidalObservationLayout::Legacy,
            run_alpha_beta_stress: true,
            include_harmonic_projection_observation: true,
            include_ampere_loop_observation: false,
            include_full_field_variance_maps: true,
            sparse_prediction_training_fraction: 0.10,
            sparse_prediction_noise_seed: 20260514,
            sparse_prediction_field_phi_count: 8,
            sparse_prediction_field_theta_count: 8,
            sparse_prediction_linked_flux_count: 12,
            sparse_prediction_local_flux_phi_count: 4,
            sparse_prediction_local_flux_theta_count: 4,
            sparse_prediction_ampere_loop_count: 8,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::ExactSolves,
                    num_variance_probes: 32,
                    variance_batch_count: 4,
                    rng_seed: 97,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: true,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalTopologySummary {
    pub betti_numbers: Vec<usize>,
    pub harmonic_2_dimension: usize,
    pub harmonic_2_mass_norm: f64,
    pub deterministic_harmonic_projection: f64,
    pub deterministic_harmonic_projection_relative: f64,
    pub deterministic_b_energy: f64,
    pub beta_true: f64,
    pub source_harmonic_energy_fraction: f64,
    pub source_harmonic_kappa: f64,
    pub ampere_loop_exact_nominal: f64,
    pub ampere_loop_harmonic_sensitivity: f64,
    pub linked_current_unit: f64,
    pub linked_current_true: f64,
    pub source_harmonic_projection_unit: f64,
    pub source_harmonic_projection_true: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalStageSummary {
    pub stage: String,
    pub latent_dimension: usize,
    pub pde_residual_norm: f64,
    pub hall_rmse: f64,
    pub flux_rmse: f64,
    pub harmonic_observation_residual: f64,
    pub b_variance_ratio_mean: f64,
    pub beta_prior_mean: Option<f64>,
    pub beta_prior_variance: Option<f64>,
    pub beta_posterior_mean: Option<f64>,
    pub beta_posterior_variance: Option<f64>,
    pub beta_error: Option<f64>,
    pub alpha_prior_mean: Option<f64>,
    pub alpha_prior_variance: Option<f64>,
    pub alpha_posterior_mean: Option<f64>,
    pub alpha_posterior_variance: Option<f64>,
    pub alpha_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalObservationUncertaintyRow {
    pub stage: String,
    pub sensor_type: String,
    pub name: String,
    pub observed: f64,
    pub prediction: f64,
    pub residual: f64,
    pub prior_variance: f64,
    pub posterior_variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalPushforwardQoiRow {
    pub stage: String,
    pub qoi: String,
    pub role: String,
    pub truth: f64,
    pub mean: f64,
    pub sd: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalPushforwardCovarianceRow {
    pub stage: String,
    pub qoi_i: String,
    pub qoi_j: String,
    pub prior_covariance: f64,
    pub posterior_covariance: f64,
    pub posterior_correlation: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalHeldoutPredictionRow {
    pub stage: String,
    pub sensor_type: String,
    pub name: String,
    pub truth: f64,
    pub prediction: f64,
    pub residual: f64,
    pub posterior_sd: f64,
    pub standardized_residual: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub covered95: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalBranchDecompositionRow {
    pub stage: String,
    pub functional: String,
    pub prior_exact_variance: f64,
    pub prior_source_harmonic_variance: f64,
    pub prior_coupling_variance: f64,
    pub prior_total_variance: f64,
    pub prior_reported_variance: f64,
    pub posterior_exact_variance: f64,
    pub posterior_source_harmonic_variance: f64,
    pub posterior_coupling_variance: f64,
    pub posterior_total_variance: f64,
    pub posterior_reported_variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalFieldTraceVarianceRow {
    pub stage: String,
    pub point: String,
    pub prior_trace_variance: f64,
    pub posterior_trace_variance: f64,
    pub variance_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct ToroidalStageResult {
    pub summary: ToroidalStageSummary,
    pub solve: LinearPdeUqResult,
    pub observations: Vec<ToroidalObservationUncertaintyRow>,
    pub pushforward_qois: Vec<ToroidalPushforwardQoiRow>,
    pub pushforward_covariances: Vec<ToroidalPushforwardCovarianceRow>,
    pub heldout_predictions: Vec<ToroidalHeldoutPredictionRow>,
    pub branch_decomposition: Vec<ToroidalBranchDecompositionRow>,
    pub field_trace_variance: Vec<ToroidalFieldTraceVarianceRow>,
}

#[derive(Debug, Clone)]
pub struct ToroidalHarmonicBResult {
    pub topology_summary: ToroidalTopologySummary,
    pub stages: Vec<ToroidalStageResult>,
}

#[derive(Debug, Clone)]
pub struct ToroidalGpBaselineConfig {
    pub toroidal: ToroidalHarmonicBConfig,
    pub output_dir: Option<PathBuf>,
    pub matern_nu: f64,
    pub length_scales: Vec<f64>,
    pub signal_std_factors: Vec<f64>,
    pub jitter: f64,
}

impl Default for ToroidalGpBaselineConfig {
    fn default() -> Self {
        Self {
            toroidal: ToroidalHarmonicBConfig::default(),
            output_dir: Some(PathBuf::from(TOPOLOGY_GP_BASELINE_OUT_DIR)),
            matern_nu: 1.5,
            length_scales: vec![0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5],
            signal_std_factors: vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            jitter: 1e-10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToroidalHarmonicCouplingCalibrationConfig {
    pub toroidal: ToroidalHarmonicBConfig,
    pub output_dir: Option<PathBuf>,
    pub drive_currents: Vec<f64>,
    pub coupling_prior_std_scale: f64,
}

impl Default for ToroidalHarmonicCouplingCalibrationConfig {
    fn default() -> Self {
        Self {
            toroidal: ToroidalHarmonicBConfig::default(),
            output_dir: Some(PathBuf::from(HARMONIC_COUPLING_CALIBRATION_OUT_DIR)),
            drive_currents: vec![0.6, 0.8, 1.0, 1.2, 1.4],
            coupling_prior_std_scale: 5.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalGpBaselineSummaryRow {
    pub stage: String,
    pub matched_feec_stage: String,
    pub kernel: String,
    pub matern_nu: f64,
    pub length_scale: f64,
    pub signal_variance: f64,
    pub log_marginal_likelihood: f64,
    pub training_rows: usize,
    pub hall_training_rows: usize,
    pub flux_training_rows: usize,
    pub ampere_training_rows: usize,
    pub heldout_rows: usize,
    pub heldout_rmse: f64,
    pub heldout_nlpd: f64,
    pub heldout_covered95: usize,
    pub heldout_coverage_fraction: f64,
    pub heldout_max_abs_standardized_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalGpBaselinePredictionRow {
    pub stage: String,
    pub sensor_type: String,
    pub name: String,
    pub truth: f64,
    pub prediction: f64,
    pub residual: f64,
    pub posterior_sd: f64,
    pub standardized_residual: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub covered95: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalGpBaselineQoiRow {
    pub stage: String,
    pub qoi: String,
    pub role: String,
    pub truth: f64,
    pub mean: f64,
    pub sd: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub abs_error: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalGpBaselineStageResult {
    pub summary: ToroidalGpBaselineSummaryRow,
    pub heldout_predictions: Vec<ToroidalGpBaselinePredictionRow>,
    pub qois: Vec<ToroidalGpBaselineQoiRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalGpBaselineResult {
    pub topology_summary: ToroidalTopologySummary,
    pub stages: Vec<ToroidalGpBaselineStageResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalSparseComparisonMetricRow {
    pub model: String,
    pub stage: String,
    pub sensor_family: String,
    pub training_rows: usize,
    pub heldout_rows: usize,
    pub rmse: f64,
    pub nlpd: f64,
    pub covered95: usize,
    pub coverage_fraction: f64,
    pub max_abs_standardized_residual: f64,
    pub mean_abs_standardized_residual: f64,
    pub source_posterior_available: bool,
    pub topology_posterior_available: bool,
    pub pde_residual_used: bool,
}

#[derive(Debug, Clone)]
pub struct ToroidalSparseGpComparisonResult {
    pub topology_summary: ToroidalTopologySummary,
    pub feec_observation_only_stages: Vec<ToroidalStageResult>,
    pub feec_full_stages: Vec<ToroidalStageResult>,
    pub gp_stages: Vec<ToroidalGpBaselineStageResult>,
    pub metrics: Vec<ToroidalSparseComparisonMetricRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalSourceTemplateGpSummaryRow {
    pub model: String,
    pub stage: String,
    pub template_kind: String,
    pub sensor_family: String,
    pub kernel: String,
    pub matern_nu: f64,
    pub length_scale: f64,
    pub signal_variance: f64,
    pub log_marginal_likelihood: f64,
    pub training_rows: usize,
    pub hall_training_rows: usize,
    pub flux_training_rows: usize,
    pub ampere_training_rows: usize,
    pub heldout_rows: usize,
    pub heldout_rmse: f64,
    pub heldout_nlpd: f64,
    pub heldout_covered95: usize,
    pub heldout_coverage_fraction: f64,
    pub heldout_max_abs_standardized_residual: f64,
    pub source_prior_mean: f64,
    pub source_prior_variance: f64,
    pub source_posterior_mean: f64,
    pub source_posterior_variance: f64,
    pub source_truth: f64,
    pub source_abs_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalSourceTemplateGpStageResult {
    pub summary: ToroidalSourceTemplateGpSummaryRow,
    pub heldout_predictions: Vec<ToroidalGpBaselinePredictionRow>,
    pub qois: Vec<ToroidalGpBaselineQoiRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalSourceTemplateGpResult {
    pub topology_summary: ToroidalTopologySummary,
    pub stages: Vec<ToroidalSourceTemplateGpStageResult>,
    pub metrics: Vec<ToroidalSparseComparisonMetricRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalCouplingCalibrationSummaryRow {
    pub stage: String,
    pub training_rows: usize,
    pub hall_training_rows: usize,
    pub flux_training_rows: usize,
    pub ampere_training_rows: usize,
    pub heldout_rows: usize,
    pub heldout_rmse: f64,
    pub heldout_nlpd: f64,
    pub heldout_covered95: usize,
    pub heldout_coverage_fraction: f64,
    pub heldout_max_abs_standardized_residual: f64,
    pub coupling_prior_mean: f64,
    pub coupling_prior_variance: f64,
    pub coupling_posterior_mean: f64,
    pub coupling_posterior_variance: f64,
    pub coupling_truth: f64,
    pub coupling_abs_error: f64,
    pub coupling_variance_ratio: f64,
    pub posterior_precision_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub posterior_fill_in: f64,
    pub posterior_factor_mib: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalCouplingCalibrationQoiRow {
    pub stage: String,
    pub drive_index: Option<usize>,
    pub drive_current: Option<f64>,
    pub qoi: String,
    pub role: String,
    pub truth: f64,
    pub mean: f64,
    pub sd: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalCouplingCalibrationObservationRow {
    pub stage: String,
    pub drive_index: usize,
    pub drive_current: f64,
    pub sensor_type: String,
    pub name: String,
    pub truth: f64,
    pub observed: f64,
    pub prediction: f64,
    pub residual: f64,
    pub posterior_sd: f64,
    pub standardized_residual: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub covered95: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalCouplingCalibrationCovarianceRow {
    pub stage: String,
    pub qoi_i: String,
    pub qoi_j: String,
    pub prior_covariance: f64,
    pub posterior_covariance: f64,
    pub posterior_correlation: f64,
}

#[derive(Debug, Clone)]
pub struct ToroidalCouplingCalibrationStageResult {
    pub summary: ToroidalCouplingCalibrationSummaryRow,
    pub pushforward: LinearPdePushforwardMeanCovarianceResult,
    pub qois: Vec<ToroidalCouplingCalibrationQoiRow>,
    pub heldout_predictions: Vec<ToroidalCouplingCalibrationObservationRow>,
    pub observations: Vec<ToroidalCouplingCalibrationObservationRow>,
    pub covariances: Vec<ToroidalCouplingCalibrationCovarianceRow>,
}

#[derive(Debug, Clone)]
pub struct ToroidalHarmonicCouplingCalibrationResult {
    pub topology_summary: ToroidalTopologySummary,
    pub drive_currents: Vec<f64>,
    pub stages: Vec<ToroidalCouplingCalibrationStageResult>,
}

#[derive(Debug, Clone, PartialEq)]
struct GpFunctionalTerm {
    point: [f64; 3],
    weight: [f64; 3],
}

#[derive(Debug, Clone)]
struct ObservationSpec {
    sensor_type: String,
    name: String,
    state_operator: SparseTripletMatrix,
    gp_terms: Vec<GpFunctionalTerm>,
    beta_operator_value: f64,
    observation_beta_truth: f64,
    observation_alpha_beta_truth: f64,
    observation_source_generated_truth: f64,
    observation_source_generated_observed: f64,
    variance_beta_truth: f64,
    variance_alpha_beta_truth: f64,
    variance_source_generated_truth: f64,
}

#[derive(Debug, Clone)]
struct ExperimentWorkspace {
    topology: Complex,
    coords: MeshCoords,
    system: ReducedLinearPdeAssembly,
    source_rhs: FeecVector,
    source_operator: SparseTripletMatrix,
    state_prior: GaussianPriorSpec,
    h2: FeecVector,
    mass_2: FeecCsr,
    topology_summary: ToroidalTopologySummary,
    truth_a_nominal: FeecVector,
    truth_a_alpha: FeecVector,
    truth_b_exact_nominal: FeecVector,
    truth_b_exact_alpha: FeecVector,
    truth_b_total_nominal: FeecVector,
    truth_b_total_alpha: FeecVector,
    truth_b_source_generated_unit: FeecVector,
    truth_b_source_generated_alpha: FeecVector,
    ampere_loop_state_operator: SparseTripletMatrix,
    observations: Vec<ObservationSpec>,
    heldout_observations: Vec<ObservationSpec>,
}

pub fn run_toroidal_harmonic_b_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let workspace = build_workspace(config)?;
    let output_dir = prepare_output_dir(config, &workspace)?;
    let stages = run_stages(config, &workspace, output_dir)?;

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_harmonic_b_fluctuation_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let workspace = build_workspace(config)?;
    let output_dir = prepare_output_dir(config, &workspace)?;
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        FLUCTUATION_ALPHA_BETA_STAGE_NAME,
        config,
        &workspace,
        build_problem(
            &workspace,
            config,
            StageProblemKind::FluctuationAlphaBetaRecovery,
            beta_prior_std,
            ObservationTruth::AlphaBeta,
        )?,
        ObservationTruth::AlphaBeta,
        StageObservationPrediction::FluctuationAlphaBeta,
        output_dir,
    )?;

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_harmonic_b_field_only_beta_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut config = config.clone();
    config.include_harmonic_projection_observation = false;
    config.include_ampere_loop_observation = false;
    if config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        config.output_dir = Some(PathBuf::from(FIELD_ONLY_BETA_OUT_DIR));
    }
    let workspace = build_workspace(&config)?;
    let output_dir = prepare_output_dir(&config, &workspace)?;
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        FIELD_ONLY_BETA_STAGE_NAME,
        &config,
        &workspace,
        build_problem(
            &workspace,
            &config,
            StageProblemKind::JointBetaRecovery,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(&config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_harmonic_b_embedded_field_beta_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut config = config.clone();
    config.observation_layout = ToroidalObservationLayout::Embedded;
    config.include_harmonic_projection_observation = false;
    config.include_ampere_loop_observation = false;
    if config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        config.output_dir = Some(PathBuf::from(EMBEDDED_FIELD_OUT_DIR));
    }
    let workspace = build_workspace(&config)?;
    let output_dir = prepare_output_dir(&config, &workspace)?;
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        EMBEDDED_FIELD_BETA_STAGE_NAME,
        &config,
        &workspace,
        build_problem(
            &workspace,
            &config,
            StageProblemKind::JointBetaRecovery,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(&config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_harmonic_b_embedded_field_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut field_config = config.clone();
    field_config.observation_layout = ToroidalObservationLayout::Embedded;
    field_config.include_harmonic_projection_observation = false;
    field_config.include_ampere_loop_observation = false;
    if field_config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        field_config.output_dir = Some(PathBuf::from(EMBEDDED_FIELD_OUT_DIR));
    }

    let field_workspace = build_workspace(&field_config)?;
    let output_dir = prepare_output_dir(&field_config, &field_workspace)?;
    let beta_prior_std = (field_config.beta_prior_std_scale
        * field_workspace.topology_summary.beta_true.abs())
    .max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        EMBEDDED_PDE_STAGE_NAME,
        &field_config,
        &field_workspace,
        build_problem(
            &field_workspace,
            &field_config,
            StageProblemKind::FixedSourceBetaLatentPdeOnly,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;
    push_stage(
        &mut stages,
        EMBEDDED_FIELD_BETA_STAGE_NAME,
        &field_config,
        &field_workspace,
        build_problem(
            &field_workspace,
            &field_config,
            StageProblemKind::JointBetaRecovery,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;

    let mut loop_config = field_config.clone();
    loop_config.include_ampere_loop_observation = true;
    let loop_workspace = build_workspace(&loop_config)?;
    let loop_beta_prior_std = (loop_config.beta_prior_std_scale
        * loop_workspace.topology_summary.beta_true.abs())
    .max(1e-10);
    push_stage(
        &mut stages,
        EMBEDDED_AMPERE_ALPHA_BETA_STAGE_NAME,
        &loop_config,
        &loop_workspace,
        build_problem(
            &loop_workspace,
            &loop_config,
            StageProblemKind::JointAlphaBetaStress,
            loop_beta_prior_std,
            ObservationTruth::AlphaBeta,
        )?,
        ObservationTruth::AlphaBeta,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;

    Ok(ToroidalHarmonicBResult {
        topology_summary: field_workspace.topology_summary.clone(),
        stages,
    })
}

pub fn run_toroidal_source_generated_harmonic_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut field_config = config.clone();
    field_config.observation_layout = ToroidalObservationLayout::Embedded;
    field_config.include_harmonic_projection_observation = false;
    field_config.include_ampere_loop_observation = false;
    if field_config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        field_config.output_dir = Some(PathBuf::from(SOURCE_GENERATED_OUT_DIR));
    }

    let field_workspace = build_workspace(&field_config)?;
    let output_dir = prepare_output_dir(&field_config, &field_workspace)?;
    let beta_prior_std = (field_config.beta_prior_std_scale
        * field_workspace.topology_summary.beta_true.abs())
    .max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        SOURCE_GENERATED_PDE_STAGE_NAME,
        &field_config,
        &field_workspace,
        build_problem(
            &field_workspace,
            &field_config,
            StageProblemKind::SourceGeneratedPdeOnly,
            beta_prior_std,
            ObservationTruth::SourceGenerated,
        )?,
        ObservationTruth::SourceGenerated,
        StageObservationPrediction::SourceGeneratedAlpha,
        output_dir,
    )?;
    push_stage(
        &mut stages,
        SOURCE_GENERATED_FIELD_STAGE_NAME,
        &field_config,
        &field_workspace,
        build_problem(
            &field_workspace,
            &field_config,
            StageProblemKind::SourceGeneratedFieldRecovery,
            beta_prior_std,
            ObservationTruth::SourceGenerated,
        )?,
        ObservationTruth::SourceGenerated,
        StageObservationPrediction::SourceGeneratedAlpha,
        output_dir,
    )?;

    let mut loop_config = field_config.clone();
    loop_config.include_ampere_loop_observation = true;
    let loop_workspace = build_workspace(&loop_config)?;
    let loop_beta_prior_std = (loop_config.beta_prior_std_scale
        * loop_workspace.topology_summary.beta_true.abs())
    .max(1e-10);
    push_stage(
        &mut stages,
        SOURCE_GENERATED_AMPERE_STAGE_NAME,
        &loop_config,
        &loop_workspace,
        build_problem(
            &loop_workspace,
            &loop_config,
            StageProblemKind::SourceGeneratedAmpereLoop,
            loop_beta_prior_std,
            ObservationTruth::SourceGenerated,
        )?,
        ObservationTruth::SourceGenerated,
        StageObservationPrediction::SourceGeneratedAlpha,
        output_dir,
    )?;

    Ok(ToroidalHarmonicBResult {
        topology_summary: field_workspace.topology_summary.clone(),
        stages,
    })
}

pub fn run_toroidal_source_generated_harmonic_full_state_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut field_config = config.clone();
    field_config.observation_layout = ToroidalObservationLayout::Embedded;
    field_config.include_harmonic_projection_observation = false;
    field_config.include_ampere_loop_observation = false;
    if field_config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        field_config.output_dir = Some(PathBuf::from(SOURCE_GENERATED_FULL_STATE_OUT_DIR));
    }

    let field_workspace = build_workspace(&field_config)?;
    let output_dir = prepare_output_dir(&field_config, &field_workspace)?;
    let beta_prior_std = (field_config.beta_prior_std_scale
        * field_workspace.topology_summary.beta_true.abs())
    .max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        SOURCE_GENERATED_FULL_STATE_PDE_STAGE_NAME,
        &field_config,
        &field_workspace,
        build_problem(
            &field_workspace,
            &field_config,
            StageProblemKind::SourceGeneratedFullStatePdeOnly,
            beta_prior_std,
            ObservationTruth::SourceGenerated,
        )?,
        ObservationTruth::SourceGenerated,
        StageObservationPrediction::SourceGeneratedFullStateAlpha,
        output_dir,
    )?;
    push_stage(
        &mut stages,
        SOURCE_GENERATED_FULL_STATE_FIELD_STAGE_NAME,
        &field_config,
        &field_workspace,
        build_problem(
            &field_workspace,
            &field_config,
            StageProblemKind::SourceGeneratedFullStateFieldRecovery,
            beta_prior_std,
            ObservationTruth::SourceGenerated,
        )?,
        ObservationTruth::SourceGenerated,
        StageObservationPrediction::SourceGeneratedFullStateAlpha,
        output_dir,
    )?;

    let mut loop_config = field_config.clone();
    loop_config.include_ampere_loop_observation = true;
    let loop_workspace = build_workspace(&loop_config)?;
    let loop_beta_prior_std = (loop_config.beta_prior_std_scale
        * loop_workspace.topology_summary.beta_true.abs())
    .max(1e-10);
    push_stage(
        &mut stages,
        SOURCE_GENERATED_FULL_STATE_AMPERE_STAGE_NAME,
        &loop_config,
        &loop_workspace,
        build_problem(
            &loop_workspace,
            &loop_config,
            StageProblemKind::SourceGeneratedFullStateAmpereLoop,
            loop_beta_prior_std,
            ObservationTruth::SourceGenerated,
        )?,
        ObservationTruth::SourceGenerated,
        StageObservationPrediction::SourceGeneratedFullStateAlpha,
        output_dir,
    )?;

    Ok(ToroidalHarmonicBResult {
        topology_summary: field_workspace.topology_summary.clone(),
        stages,
    })
}

pub fn run_toroidal_topology_pushforward_uq(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut config = config.clone();
    config.observation_layout = ToroidalObservationLayout::TopologyPushforward;
    config.include_harmonic_projection_observation = false;
    config.include_ampere_loop_observation = true;
    config.include_full_field_variance_maps = false;
    config.use_mass_weighted_pde_residual = true;
    config.normalize_mass_weighted_pde_residual = true;
    if config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        config.output_dir = Some(PathBuf::from(TOPOLOGY_PUSHFORWARD_OUT_DIR));
    }

    let workspace = build_workspace(&config)?;
    let output_dir = prepare_output_dir(&config, &workspace)?;
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);
    let mut stages = Vec::new();
    for (name, kind) in [
        (
            TOPOLOGY_PUSHFORWARD_PRIOR_STAGE_NAME,
            StageProblemKind::TopologyPushforwardPriorOnly,
        ),
        (
            TOPOLOGY_PUSHFORWARD_PDE_STAGE_NAME,
            StageProblemKind::TopologyPushforwardPdeOnly,
        ),
        (
            TOPOLOGY_PUSHFORWARD_FIELD_STAGE_NAME,
            StageProblemKind::TopologyPushforwardFieldProbes,
        ),
        (
            TOPOLOGY_PUSHFORWARD_FLUX_STAGE_NAME,
            StageProblemKind::TopologyPushforwardFluxPanels,
        ),
        (
            TOPOLOGY_PUSHFORWARD_AMPERE_STAGE_NAME,
            StageProblemKind::TopologyPushforwardAmpereLoop,
        ),
    ] {
        push_stage(
            &mut stages,
            name,
            &config,
            &workspace,
            build_problem(
                &workspace,
                &config,
                kind,
                beta_prior_std,
                ObservationTruth::SourceGenerated,
            )?,
            ObservationTruth::SourceGenerated,
            StageObservationPrediction::SourceGeneratedFullStateAlpha,
            output_dir,
        )?;
    }

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(&config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_topology_sparse_noisy_prediction(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut config = config.clone();
    config.observation_layout = ToroidalObservationLayout::TopologySparseNoisyPrediction;
    config.source_alpha_true = 1.0;
    config.include_harmonic_projection_observation = false;
    config.include_ampere_loop_observation = true;
    config.include_full_field_variance_maps = false;
    config.use_mass_weighted_pde_residual = true;
    config.normalize_mass_weighted_pde_residual = true;
    if config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        config.output_dir = Some(PathBuf::from(TOPOLOGY_SPARSE_NOISY_OUT_DIR));
    }

    let workspace = build_workspace(&config)?;
    let output_dir = prepare_output_dir(&config, &workspace)?;
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);
    let mut stages = Vec::new();
    for (name, kind) in [
        (
            TOPOLOGY_SPARSE_NOISY_PRIOR_STAGE_NAME,
            StageProblemKind::TopologySparseNoisyPriorOnly,
        ),
        (
            TOPOLOGY_SPARSE_NOISY_PDE_STAGE_NAME,
            StageProblemKind::TopologySparseNoisyPdeOnly,
        ),
        (
            TOPOLOGY_SPARSE_NOISY_OBS_STAGE_NAME,
            StageProblemKind::TopologySparseNoisyObservations,
        ),
    ] {
        push_stage(
            &mut stages,
            name,
            &config,
            &workspace,
            build_problem(
                &workspace,
                &config,
                kind,
                beta_prior_std,
                ObservationTruth::SourceGenerated,
            )?,
            ObservationTruth::SourceGenerated,
            StageObservationPrediction::SourceGeneratedFullStateAlpha,
            output_dir,
        )?;
    }

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(&config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_topology_gp_baseline(
    config: &ToroidalGpBaselineConfig,
) -> Result<ToroidalGpBaselineResult, Box<dyn Error>> {
    let mut toroidal_config = config.toroidal.clone();
    toroidal_config.observation_layout = ToroidalObservationLayout::TopologyPushforward;
    toroidal_config.include_harmonic_projection_observation = false;
    toroidal_config.include_ampere_loop_observation = true;
    toroidal_config.include_full_field_variance_maps = false;
    toroidal_config.output_dir = None;

    let workspace = build_workspace(&toroidal_config)?;
    let stages = [
        (
            "G2_field_probes",
            TOPOLOGY_PUSHFORWARD_FIELD_STAGE_NAME,
            StageProblemKind::TopologyPushforwardFieldProbes,
        ),
        (
            "G3_flux_panels",
            TOPOLOGY_PUSHFORWARD_FLUX_STAGE_NAME,
            StageProblemKind::TopologyPushforwardFluxPanels,
        ),
        (
            "G4_ampere_loop",
            TOPOLOGY_PUSHFORWARD_AMPERE_STAGE_NAME,
            StageProblemKind::TopologyPushforwardAmpereLoop,
        ),
    ];

    let mut results = Vec::with_capacity(stages.len());
    for (stage_name, matched_feec_stage, kind) in stages {
        eprintln!("[toroidal-gp] stage_start name={stage_name}");
        let training = topology_pushforward_training_observations(&workspace, kind);
        let stage = run_toroidal_gp_baseline_stage(
            config,
            &workspace,
            stage_name,
            matched_feec_stage,
            training,
        )?;
        report_gp_baseline_progress(&stage);
        results.push(stage);
    }

    let result = ToroidalGpBaselineResult {
        topology_summary: workspace.topology_summary.clone(),
        stages: results,
    };

    if let Some(out_dir) = &config.output_dir {
        fs::create_dir_all(out_dir)?;
        write_topology_summary(
            &result.topology_summary,
            &out_dir.join("topology_summary.json"),
        )?;
        write_topology_summary_csv(
            &result.topology_summary,
            &out_dir.join("topology_summary.csv"),
        )?;
        write_gp_baseline_outputs(&result, out_dir)?;
        write_gp_baseline_comparison_csv(
            &result,
            &PathBuf::from(TOPOLOGY_PUSHFORWARD_OUT_DIR),
            &out_dir.join("gp_baseline_comparison.csv"),
        )?;
    }

    Ok(result)
}

pub fn run_toroidal_topology_sparse_noisy_gp_comparison(
    config: &ToroidalGpBaselineConfig,
) -> Result<ToroidalSparseGpComparisonResult, Box<dyn Error>> {
    let mut toroidal_config = config.toroidal.clone();
    toroidal_config.observation_layout = ToroidalObservationLayout::TopologySparseNoisyPrediction;
    toroidal_config.source_alpha_true = 1.15;
    toroidal_config.include_harmonic_projection_observation = false;
    toroidal_config.include_ampere_loop_observation = true;
    toroidal_config.include_full_field_variance_maps = false;
    toroidal_config.use_mass_weighted_pde_residual = true;
    toroidal_config.normalize_mass_weighted_pde_residual = true;
    toroidal_config.output_dir = None;

    let workspace = build_workspace(&toroidal_config)?;
    let beta_prior_std = (toroidal_config.beta_prior_std_scale
        * workspace.topology_summary.beta_true.abs())
    .max(1e-10);

    let specs = [
        SparseComparisonSpec {
            label: "field",
            gp_stage: "G2_sparse_field",
            obs_only_stage: "O2_sparse_field",
            full_stage: "C2_sparse_field",
            obs_only_kind: StageProblemKind::TopologySparseNoisyFieldObservationsNoPde,
            full_kind: StageProblemKind::TopologySparseNoisyFieldObservations,
        },
        SparseComparisonSpec {
            label: "field_flux",
            gp_stage: "G3_sparse_field_flux",
            obs_only_stage: "O3_sparse_field_flux",
            full_stage: "C3_sparse_field_flux",
            obs_only_kind: StageProblemKind::TopologySparseNoisyFieldFluxObservationsNoPde,
            full_kind: StageProblemKind::TopologySparseNoisyFieldFluxObservations,
        },
        SparseComparisonSpec {
            label: "field_flux_loop",
            gp_stage: "G4_sparse_field_flux_loop",
            obs_only_stage: "O4_sparse_field_flux_loop",
            full_stage: "C4_sparse_field_flux_loop",
            obs_only_kind: StageProblemKind::TopologySparseNoisyObservationsNoPde,
            full_kind: StageProblemKind::TopologySparseNoisyObservations,
        },
    ];

    let mut feec_observation_only_stages = Vec::new();
    let mut feec_full_stages = Vec::new();
    let mut gp_stages = Vec::new();
    let mut metrics = Vec::new();

    let pde_only = run_sparse_comparison_feec_stage(
        SPARSE_GP_COMPARISON_PDE_ONLY_STAGE_NAME,
        &toroidal_config,
        &workspace,
        StageProblemKind::TopologySparseNoisyPdeOnly,
        beta_prior_std,
        workspace
            .observations
            .iter()
            .chain(workspace.heldout_observations.iter())
            .collect(),
    )?;
    report_stage_progress(&pde_only);
    feec_full_stages.push(pde_only);

    for spec in specs {
        let training = topology_pushforward_training_observations(&workspace, spec.full_kind);
        let heldout = sparse_comparison_heldout_observations(&workspace, &training);

        let obs_only = run_sparse_comparison_feec_stage(
            spec.obs_only_stage,
            &toroidal_config,
            &workspace,
            spec.obs_only_kind,
            beta_prior_std,
            heldout.clone(),
        )?;
        report_stage_progress(&obs_only);
        metrics.extend(sparse_feec_metric_rows(
            "FEEC-GMRF observation-only",
            &obs_only,
            training.len(),
            false,
        ));
        feec_observation_only_stages.push(obs_only);

        let full = run_sparse_comparison_feec_stage(
            spec.full_stage,
            &toroidal_config,
            &workspace,
            spec.full_kind,
            beta_prior_std,
            heldout.clone(),
        )?;
        report_stage_progress(&full);
        metrics.extend(sparse_feec_metric_rows(
            "FEEC-GMRF full",
            &full,
            training.len(),
            true,
        ));
        feec_full_stages.push(full);

        let gp = run_toroidal_gp_baseline_stage_with_predictions(
            config,
            &workspace,
            spec.gp_stage,
            spec.label,
            training,
            heldout,
        )?;
        report_gp_baseline_progress(&gp);
        metrics.extend(sparse_gp_metric_rows(&gp));
        gp_stages.push(gp);
    }

    let result = ToroidalSparseGpComparisonResult {
        topology_summary: workspace.topology_summary.clone(),
        feec_observation_only_stages,
        feec_full_stages,
        gp_stages,
        metrics,
    };

    if let Some(out_dir) = &config.output_dir {
        fs::create_dir_all(out_dir)?;
        write_topology_summary(
            &result.topology_summary,
            &out_dir.join("topology_summary.json"),
        )?;
        write_topology_summary_csv(
            &result.topology_summary,
            &out_dir.join("topology_summary.csv"),
        )?;
        write_sparse_prediction_split_csv(
            &workspace,
            &out_dir.join("sparse_prediction_split.csv"),
        )?;
        write_sparse_gp_comparison_outputs(&result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_topology_source_template_gp_baseline(
    config: &ToroidalGpBaselineConfig,
) -> Result<ToroidalSourceTemplateGpResult, Box<dyn Error>> {
    let mut toroidal_config = config.toroidal.clone();
    toroidal_config.observation_layout = ToroidalObservationLayout::TopologySparseNoisyPrediction;
    toroidal_config.source_alpha_true = 1.15;
    toroidal_config.include_harmonic_projection_observation = false;
    toroidal_config.include_ampere_loop_observation = true;
    toroidal_config.include_full_field_variance_maps = false;
    toroidal_config.use_mass_weighted_pde_residual = true;
    toroidal_config.normalize_mass_weighted_pde_residual = true;
    toroidal_config.output_dir = None;

    let workspace = build_workspace(&toroidal_config)?;
    let specs = [
        (
            "field",
            StageProblemKind::TopologySparseNoisyFieldObservations,
        ),
        (
            "field_flux",
            StageProblemKind::TopologySparseNoisyFieldFluxObservations,
        ),
        (
            "field_flux_loop",
            StageProblemKind::TopologySparseNoisyObservations,
        ),
    ];

    let mut stages = Vec::new();
    let mut metrics = Vec::new();
    for template_kind in [
        SourceTemplateKind::ExactSource,
        SourceTemplateKind::TopologyOracle,
    ] {
        for (sensor_family, kind) in specs {
            let stage_name = template_kind.stage_name(sensor_family);
            eprintln!(
                "[toroidal-source-template-gp] stage_start model={} stage={stage_name}",
                template_kind.model_label()
            );
            let training = topology_pushforward_training_observations(&workspace, kind);
            let heldout = sparse_comparison_heldout_observations(&workspace, &training);
            let stage = run_source_template_gp_stage(
                config,
                &workspace,
                template_kind,
                stage_name,
                sensor_family,
                training,
                heldout,
            )?;
            report_source_template_gp_progress(&stage);
            metrics.extend(source_template_gp_metric_rows(&stage));
            stages.push(stage);
        }
    }

    let result = ToroidalSourceTemplateGpResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
        metrics,
    };

    if let Some(out_dir) = &config.output_dir {
        fs::create_dir_all(out_dir)?;
        write_topology_summary(
            &result.topology_summary,
            &out_dir.join("topology_summary.json"),
        )?;
        write_topology_summary_csv(
            &result.topology_summary,
            &out_dir.join("topology_summary.csv"),
        )?;
        write_sparse_prediction_split_csv(
            &workspace,
            &out_dir.join("sparse_prediction_split.csv"),
        )?;
        write_source_template_gp_outputs(&result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_harmonic_coupling_calibration(
    config: &ToroidalHarmonicCouplingCalibrationConfig,
) -> Result<ToroidalHarmonicCouplingCalibrationResult, Box<dyn Error>> {
    validate_coupling_calibration_config(config)?;
    let mut toroidal_config = config.toroidal.clone();
    toroidal_config.observation_layout = ToroidalObservationLayout::TopologySparseNoisyPrediction;
    toroidal_config.source_alpha_true = 1.0;
    toroidal_config.include_harmonic_projection_observation = false;
    toroidal_config.include_ampere_loop_observation = true;
    toroidal_config.include_full_field_variance_maps = false;
    toroidal_config.use_mass_weighted_pde_residual = true;
    toroidal_config.normalize_mass_weighted_pde_residual = true;
    toroidal_config.output_dir = None;

    let workspace = build_workspace(&toroidal_config)?;
    let prior_std = (config.coupling_prior_std_scale
        * workspace.topology_summary.source_harmonic_kappa.abs())
    .max(1e-14);
    let stage_specs = [
        (
            "K0_prior_only",
            CouplingCalibrationStageKind::PriorOnly,
            StageProblemKind::TopologySparseNoisyPriorOnly,
        ),
        (
            "K1_pde_residual",
            CouplingCalibrationStageKind::PdeResidualOnly,
            StageProblemKind::TopologySparseNoisyPdeOnly,
        ),
        (
            "K2_field_probes",
            CouplingCalibrationStageKind::FieldProbes,
            StageProblemKind::TopologySparseNoisyFieldObservations,
        ),
        (
            "K3_flux_panels",
            CouplingCalibrationStageKind::FluxPanels,
            StageProblemKind::TopologySparseNoisyFieldFluxObservations,
        ),
        (
            "K4_ampere_loops",
            CouplingCalibrationStageKind::AmpereLoops,
            StageProblemKind::TopologySparseNoisyObservations,
        ),
    ];

    let mut stages = Vec::with_capacity(stage_specs.len());
    for (stage_name, calibration_kind, training_kind) in stage_specs {
        eprintln!("[toroidal-coupling-calibration] stage_start name={stage_name}");
        let training = if matches!(
            calibration_kind,
            CouplingCalibrationStageKind::PriorOnly | CouplingCalibrationStageKind::PdeResidualOnly
        ) {
            Vec::new()
        } else {
            topology_pushforward_training_observations(&workspace, training_kind)
        };
        let heldout = sparse_comparison_heldout_observations(&workspace, &training);
        let stage = run_coupling_calibration_stage(
            stage_name,
            config,
            &toroidal_config,
            &workspace,
            calibration_kind,
            training,
            heldout,
            prior_std,
        )?;
        report_coupling_calibration_progress(&stage);
        stages.push(stage);
    }

    let result = ToroidalHarmonicCouplingCalibrationResult {
        topology_summary: workspace.topology_summary.clone(),
        drive_currents: config.drive_currents.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        fs::create_dir_all(out_dir)?;
        write_topology_summary(
            &result.topology_summary,
            &out_dir.join("topology_summary.json"),
        )?;
        write_topology_summary_csv(
            &result.topology_summary,
            &out_dir.join("topology_summary.csv"),
        )?;
        write_coupling_calibration_outputs(
            &result,
            config,
            prior_std,
            &out_dir.join("coupling_calibration.json"),
            out_dir,
        )?;
    }

    Ok(result)
}

pub fn run_toroidal_harmonic_b_ampere_loop_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut config = config.clone();
    config.include_ampere_loop_observation = true;
    if config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        config.output_dir = Some(PathBuf::from(AMPERE_LOOP_OUT_DIR));
    }
    let workspace = build_workspace(&config)?;
    let output_dir = prepare_output_dir(&config, &workspace)?;
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        AMPERE_LOOP_ALPHA_BETA_STAGE_NAME,
        &config,
        &workspace,
        build_problem(
            &workspace,
            &config,
            StageProblemKind::AmpereLoopAlphaBetaRecovery,
            beta_prior_std,
            ObservationTruth::AlphaBeta,
        )?,
        ObservationTruth::AlphaBeta,
        StageObservationPrediction::FluctuationAlphaBeta,
        output_dir,
    )?;

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(&config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn run_toroidal_harmonic_b_full_state_ampere_loop_recovery(
    config: &ToroidalHarmonicBConfig,
) -> Result<ToroidalHarmonicBResult, Box<dyn Error>> {
    let mut config = config.clone();
    config.include_ampere_loop_observation = true;
    if config.output_dir.as_deref() == Some(Path::new(OUT_DIR)) {
        config.output_dir = Some(PathBuf::from(FULL_STATE_AMPERE_LOOP_OUT_DIR));
    }
    let workspace = build_workspace(&config)?;
    let output_dir = prepare_output_dir(&config, &workspace)?;
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);
    let mut stages = Vec::new();
    push_stage(
        &mut stages,
        FULL_STATE_AMPERE_LOOP_ALPHA_BETA_STAGE_NAME,
        &config,
        &workspace,
        build_problem(
            &workspace,
            &config,
            StageProblemKind::JointAlphaBetaStress,
            beta_prior_std,
            ObservationTruth::AlphaBeta,
        )?,
        ObservationTruth::AlphaBeta,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;

    let result = ToroidalHarmonicBResult {
        topology_summary: workspace.topology_summary.clone(),
        stages,
    };

    if let Some(out_dir) = &config.output_dir {
        write_outputs(&config, &workspace, &result, out_dir)?;
    }

    Ok(result)
}

pub fn compute_toroidal_harmonic_topology_summary(
    mesh_path: impl AsRef<Path>,
) -> Result<ToroidalTopologySummary, Box<dyn Error>> {
    let config = ToroidalHarmonicBConfig {
        mesh_path: mesh_path.as_ref().to_path_buf(),
        output_dir: None,
        run_alpha_beta_stress: false,
        ..ToroidalHarmonicBConfig::default()
    };
    Ok(build_workspace(&config)?.topology_summary)
}

fn prepare_output_dir<'a>(
    config: &'a ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
) -> Result<Option<&'a Path>, Box<dyn Error>> {
    let Some(out_dir) = config.output_dir.as_deref() else {
        return Ok(None);
    };
    if out_dir.exists() {
        fs::remove_dir_all(out_dir)?;
    }
    fs::create_dir_all(out_dir)?;
    write_topology_summary(
        &workspace.topology_summary,
        &out_dir.join("topology_summary.json"),
    )?;
    write_topology_summary_csv(
        &workspace.topology_summary,
        &out_dir.join("topology_summary.csv"),
    )?;
    write_truth_vtus(workspace, out_dir)?;
    Ok(Some(out_dir))
}

fn build_workspace(
    config: &ToroidalHarmonicBConfig,
) -> Result<ExperimentWorkspace, Box<dyn Error>> {
    if config.beta_energy_fraction <= 0.0 {
        return Err(invalid_input("beta_energy_fraction must be positive").into());
    }
    if config.relative_noise_std <= 0.0 {
        return Err(invalid_input("relative_noise_std must be positive").into());
    }
    if config.source_prior_std <= 0.0 {
        return Err(invalid_input("source_prior_std must be positive").into());
    }
    if !config.fluctuation_state_prior_precision_scale.is_finite()
        || config.fluctuation_state_prior_precision_scale <= 0.0
    {
        return Err(invalid_input(
            "fluctuation_state_prior_precision_scale must be finite and positive",
        )
        .into());
    }
    if !config.mass_weighted_pde_precision_scale.is_finite()
        || config.mass_weighted_pde_precision_scale <= 0.0
    {
        return Err(
            invalid_input("mass_weighted_pde_precision_scale must be finite and positive").into(),
        );
    }
    if matches!(
        config.observation_layout,
        ToroidalObservationLayout::TopologySparseNoisyPrediction
    ) {
        if !(0.0..=1.0).contains(&config.sparse_prediction_training_fraction)
            || config.sparse_prediction_training_fraction <= 0.0
        {
            return Err(invalid_input(
                "sparse_prediction_training_fraction must be in the interval (0, 1]",
            )
            .into());
        }
        if config.sparse_prediction_field_phi_count == 0
            || config.sparse_prediction_field_theta_count == 0
            || config.sparse_prediction_linked_flux_count == 0
            || config.sparse_prediction_local_flux_phi_count == 0
            || config.sparse_prediction_local_flux_theta_count == 0
            || config.sparse_prediction_ampere_loop_count == 0
        {
            return Err(
                invalid_input("sparse prediction sensor counts must all be positive").into(),
            );
        }
    }

    let mesh_path = resolve_mesh_path(&config.mesh_path);
    let mesh_bytes = fs::read(&mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(invalid_input(format!(
            "toroidal harmonic-B recovery requires a 3D mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ))
        .into());
    }
    let metric = coords.to_edge_lengths(&topology);
    let geom = ToroidalInductorGeometry::default();
    let mu_0 = 4e-7 * PI;
    let mu_0_inverse = 1.0 / mu_0;
    let inverse_permeability = InnerProductWeightClosure::new(move |_| mu_0_inverse);

    let outer_state_dofs = sorted_boundary_dofs(&topology, &coords, 1, |point| {
        outer_boundary_predicate(point, geom)
    });
    let outer_aux_dofs = sorted_boundary_dofs(&topology, &coords, 0, |point| {
        outer_boundary_predicate(point, geom)
    });
    let boundary = EssentialBoundarySpec {
        state: outer_state_dofs
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
        auxiliary: outer_aux_dofs
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
    };

    let galmats_1 =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &inverse_permeability);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &inverse_permeability,
        ));
    let system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats_1,
        &boundary,
        &state_mass_inverse,
    )
    .map_err(invalid_input)?;

    let nominal_source_full = assemble_weighted_source(
        &topology,
        &metric,
        &coords,
        &inverse_permeability,
        &coil_mode(geom, mu_0),
    );
    let source_rhs = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        &galmats_1,
        &boundary,
        &FeecVector::zeros(galmats_1.sigma_len()),
        &nominal_source_full,
    )
    .map_err(invalid_input)?;
    let source_operator = columns_to_sparse_matrix(&[source_rhs.scale(-1.0)]);
    let state_prior = build_whittle_prior(&system, true);

    let truth_a = solve_full_feec_deterministic_reference(
        &topology,
        &metric,
        &coords,
        &inverse_permeability,
        &nominal_source_full,
        geom,
    );
    let truth_a_nominal = truth_a.coeffs.clone();
    let truth_a_alpha = truth_a_nominal.scale(config.source_alpha_true);
    let truth_b_exact_nominal = truth_a.dif(&topology).coeffs;
    let truth_b_exact_alpha = truth_b_exact_nominal.scale(config.source_alpha_true);

    let galmats_2 =
        MixedGalmats::compute_weighted(&topology, &metric, 2, &coords, None, &inverse_permeability);
    let harmonic_basis_raw = hodge_laplace::solve_hodge_laplace_harmonics_with_galmats(
        &topology, &galmats_2, 2, 1, None, None,
    );
    if harmonic_basis_raw.ncols() != 1 {
        return Err(invalid_input(format!(
            "expected a 1D harmonic 2-form basis on {}, found {} columns",
            mesh_path.display(),
            harmonic_basis_raw.ncols()
        ))
        .into());
    }
    let mass_2 = FeecCsr::from(galmats_2.mass_u());
    let harmonic_basis =
        mass_orthonormalize_harmonic_basis(&harmonic_basis_raw, &mass_2).map_err(invalid_input)?;
    let mut h2 = harmonic_basis.column(0).into_owned();

    let d1_operator = build_exterior_derivative_row_operator(&topology).map_err(invalid_input)?;
    let component_operators = [
        compose_sparse_row_operators(
            &build_sampled_magnetic_field_component_operator(&topology, &coords, 0)
                .map_err(invalid_input)?,
            &d1_operator,
        )
        .map_err(invalid_input)?,
        compose_sparse_row_operators(
            &build_sampled_magnetic_field_component_operator(&topology, &coords, 1)
                .map_err(invalid_input)?,
            &d1_operator,
        )
        .map_err(invalid_input)?,
        compose_sparse_row_operators(
            &build_sampled_magnetic_field_component_operator(&topology, &coords, 2)
                .map_err(invalid_input)?,
            &d1_operator,
        )
        .map_err(invalid_input)?,
    ];
    let mut h2_vectors =
        sample_2form_cell_vectors(&coords, &topology, &Cochain::new(2, h2.clone()))
            .map_err(|err| invalid_input(err.to_string()))?;
    let mut ampere_loop =
        build_ampere_loop_operator(&topology, &coords, geom, &component_operators, &h2_vectors)
            .map_err(invalid_input)?;
    if ampere_loop.beta_operator_value.abs() <= EPS {
        return Err(invalid_input(
            "source-generated harmonic calibration requires nonzero Ampere-loop sensitivity to h2",
        )
        .into());
    }
    if ampere_loop.beta_operator_value < 0.0 {
        h2 = h2.scale(-1.0);
        for vector in &mut h2_vectors {
            for component in vector {
                *component = -*component;
            }
        }
        ampere_loop.beta_operator_value = -ampere_loop.beta_operator_value;
    }

    let harmonic_2_mass_norm = mass_inner_product(&h2, &h2, &mass_2).sqrt();
    let deterministic_harmonic_projection =
        mass_inner_product(&h2, &truth_b_exact_nominal, &mass_2);
    let deterministic_b_energy =
        mass_inner_product(&truth_b_exact_nominal, &truth_b_exact_nominal, &mass_2)
            .max(0.0)
            .sqrt();
    let source_harmonic_kappa = config.beta_energy_fraction.sqrt() * deterministic_b_energy;
    let source_harmonic_energy_fraction = if deterministic_b_energy > EPS {
        (source_harmonic_kappa * source_harmonic_kappa)
            / (deterministic_b_energy * deterministic_b_energy)
    } else {
        f64::NAN
    };
    let ampere_loop_exact_nominal =
        apply_triplet_row(&ampere_loop.state_operator, &truth_a_nominal).map_err(invalid_input)?;
    let linked_current_unit =
        ampere_loop_exact_nominal + source_harmonic_kappa * ampere_loop.beta_operator_value;
    let linked_current_true = config.source_alpha_true * linked_current_unit;
    let source_harmonic_projection_unit = deterministic_harmonic_projection + source_harmonic_kappa;
    let source_harmonic_projection_true =
        config.source_alpha_true * source_harmonic_projection_unit;
    let beta_true = source_harmonic_kappa;
    let truth_b_harmonic = h2.scale(beta_true);
    let truth_b_total_nominal = &truth_b_exact_nominal + &truth_b_harmonic;
    let truth_b_total_alpha = &truth_b_exact_alpha + &truth_b_harmonic;
    let truth_b_source_generated_unit = &truth_b_exact_nominal + &truth_b_harmonic;
    let truth_b_source_generated_alpha =
        truth_b_source_generated_unit.scale(config.source_alpha_true);
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    let face_rows = csr_rows(&d1);

    let observations = build_observations(
        config,
        &topology,
        &coords,
        geom,
        &truth_a_nominal,
        &truth_a_alpha,
        &h2,
        beta_true,
        source_harmonic_kappa,
        &mass_2,
    )
    .map_err(invalid_input)?;
    let (observations, heldout_observations) = if matches!(
        config.observation_layout,
        ToroidalObservationLayout::TopologySparseNoisyPrediction
    ) {
        split_sparse_noisy_prediction_observations(observations, config).map_err(invalid_input)?
    } else {
        let heldout_observations = build_heldout_observations(
            config,
            &topology,
            &coords,
            geom,
            &component_operators,
            &h2_vectors,
            &face_rows,
            &h2,
            &truth_a_nominal,
            &truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
        )
        .map_err(invalid_input)?;
        (observations, heldout_observations)
    };

    let betti_numbers = crate::de_rham::betti_numbers(&topology);
    let topology_summary = ToroidalTopologySummary {
        betti_numbers,
        harmonic_2_dimension: harmonic_basis.ncols(),
        harmonic_2_mass_norm,
        deterministic_harmonic_projection,
        deterministic_harmonic_projection_relative: deterministic_harmonic_projection.abs()
            / deterministic_b_energy.max(EPS),
        deterministic_b_energy,
        beta_true,
        source_harmonic_energy_fraction,
        source_harmonic_kappa,
        ampere_loop_exact_nominal,
        ampere_loop_harmonic_sensitivity: ampere_loop.beta_operator_value,
        linked_current_unit,
        linked_current_true,
        source_harmonic_projection_unit,
        source_harmonic_projection_true,
    };

    Ok(ExperimentWorkspace {
        topology,
        coords,
        system,
        source_rhs,
        source_operator,
        state_prior,
        h2,
        mass_2,
        topology_summary,
        truth_a_nominal,
        truth_a_alpha,
        truth_b_exact_nominal,
        truth_b_exact_alpha,
        truth_b_total_nominal,
        truth_b_total_alpha,
        truth_b_source_generated_unit,
        truth_b_source_generated_alpha,
        ampere_loop_state_operator: ampere_loop.state_operator,
        observations,
        heldout_observations,
    })
}

fn run_stages(
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    output_dir: Option<&Path>,
) -> Result<Vec<ToroidalStageResult>, Box<dyn Error>> {
    let mut stages = Vec::new();
    let beta_prior_std =
        (config.beta_prior_std_scale * workspace.topology_summary.beta_true.abs()).max(1e-10);

    push_stage(
        &mut stages,
        "H0_prior_only",
        config,
        workspace,
        build_problem(
            workspace,
            config,
            StageProblemKind::PriorOnly,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;
    push_stage(
        &mut stages,
        "H1_pde_only",
        config,
        workspace,
        build_problem(
            workspace,
            config,
            StageProblemKind::FixedSourceBetaLatentPdeOnly,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;
    push_stage(
        &mut stages,
        "H2_fixed_beta_control",
        config,
        workspace,
        build_problem(
            workspace,
            config,
            StageProblemKind::FixedBetaControl,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;
    push_stage(
        &mut stages,
        "H3_joint_beta_recovery",
        config,
        workspace,
        build_problem(
            workspace,
            config,
            StageProblemKind::JointBetaRecovery,
            beta_prior_std,
            ObservationTruth::BetaOnly,
        )?,
        ObservationTruth::BetaOnly,
        StageObservationPrediction::StatePlusBeta,
        output_dir,
    )?;

    if config.run_alpha_beta_stress {
        push_stage(
            &mut stages,
            "H4_joint_alpha_beta_stress",
            config,
            workspace,
            build_problem(
                workspace,
                config,
                StageProblemKind::JointAlphaBetaStress,
                beta_prior_std,
                ObservationTruth::AlphaBeta,
            )?,
            ObservationTruth::AlphaBeta,
            StageObservationPrediction::StatePlusBeta,
            output_dir,
        )?;
    }

    Ok(stages)
}

fn push_stage(
    stages: &mut Vec<ToroidalStageResult>,
    name: &str,
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    problem: LinearPdeUqProblem,
    truth: ObservationTruth,
    prediction: StageObservationPrediction,
    output_dir: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    eprintln!(
        "[toroidal] stage_start name={} variance_mode={}",
        name,
        variance_mode_name(config.solver.variance.mode)
    );
    let problem_for_pushforward = problem.clone();
    let mut stage = run_stage(name, config, workspace, problem, truth, prediction)?;
    if is_topology_pushforward_stage_name(name) {
        attach_topology_pushforward_reports(
            &mut stage,
            config,
            workspace,
            &problem_for_pushforward,
            prediction,
        )?;
    }
    report_stage_progress(&stage);
    report_topology_pushforward_progress(&stage);
    stages.push(stage);
    if let Some(out_dir) = output_dir {
        let latest = stages
            .last()
            .expect("stage was just pushed and should be available");
        write_stage_vtus(workspace, latest, out_dir)?;
        write_stage_summary_csv(stages, &out_dir.join("stage_summary.csv"))?;
        write_harmonic_posterior_csv(
            stages,
            workspace.topology_summary.beta_true,
            &out_dir.join("harmonic_posterior.csv"),
        )?;
        write_source_generated_summary_csv(
            workspace,
            stages,
            &out_dir.join("source_generated_summary.csv"),
        )?;
        write_source_generated_objective_diagnostics_csv(
            config,
            workspace,
            stages,
            &out_dir.join("source_generated_objective_diagnostics.csv"),
        )?;
        write_source_generated_field_errors_csv(
            workspace,
            stages,
            &out_dir.join("source_generated_field_errors.csv"),
        )?;
        write_observation_uncertainty_csv(stages, &out_dir.join("observation_uncertainty.csv"))?;
        write_ampere_loop_summary_csv(workspace, stages, &out_dir.join("ampere_loop_summary.csv"))?;
        write_pushforward_qoi_summary_csv(stages, &out_dir.join("pushforward_qoi_summary.csv"))?;
        write_pushforward_qoi_covariance_csv(
            stages,
            &out_dir.join("pushforward_qoi_covariance.csv"),
        )?;
        write_pushforward_variance_ratios_csv(
            stages,
            &out_dir.join("pushforward_variance_ratios.csv"),
        )?;
        write_heldout_prediction_csv(stages, &out_dir.join("heldout_prediction.csv"))?;
        write_branch_decomposition_csv(stages, &out_dir.join("branch_decomposition.csv"))?;
        write_field_trace_variance_csv(stages, &out_dir.join("field_trace_variance.csv"))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageProblemKind {
    PriorOnly,
    FixedSourceBetaLatentPdeOnly,
    FixedBetaControl,
    JointBetaRecovery,
    JointAlphaBetaStress,
    FluctuationAlphaBetaRecovery,
    AmpereLoopAlphaBetaRecovery,
    SourceGeneratedPdeOnly,
    SourceGeneratedFieldRecovery,
    SourceGeneratedAmpereLoop,
    SourceGeneratedFullStatePdeOnly,
    SourceGeneratedFullStateFieldRecovery,
    SourceGeneratedFullStateAmpereLoop,
    TopologyPushforwardPriorOnly,
    TopologyPushforwardPdeOnly,
    TopologyPushforwardFieldProbes,
    TopologyPushforwardFluxPanels,
    TopologyPushforwardAmpereLoop,
    TopologySparseNoisyPriorOnly,
    TopologySparseNoisyPdeOnly,
    TopologySparseNoisyFieldObservations,
    TopologySparseNoisyFieldFluxObservations,
    TopologySparseNoisyObservations,
    TopologySparseNoisyFieldObservationsNoPde,
    TopologySparseNoisyFieldFluxObservationsNoPde,
    TopologySparseNoisyObservationsNoPde,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservationTruth {
    BetaOnly,
    AlphaBeta,
    SourceGenerated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageObservationPrediction {
    StatePlusBeta,
    FluctuationAlphaBeta,
    SourceGeneratedAlpha,
    SourceGeneratedFullStateAlpha,
}

fn is_topology_pushforward_kind(kind: StageProblemKind) -> bool {
    matches!(
        kind,
        StageProblemKind::TopologyPushforwardPriorOnly
            | StageProblemKind::TopologyPushforwardPdeOnly
            | StageProblemKind::TopologyPushforwardFieldProbes
            | StageProblemKind::TopologyPushforwardFluxPanels
            | StageProblemKind::TopologyPushforwardAmpereLoop
            | StageProblemKind::TopologySparseNoisyPriorOnly
            | StageProblemKind::TopologySparseNoisyPdeOnly
            | StageProblemKind::TopologySparseNoisyFieldObservations
            | StageProblemKind::TopologySparseNoisyFieldFluxObservations
            | StageProblemKind::TopologySparseNoisyObservations
            | StageProblemKind::TopologySparseNoisyFieldObservationsNoPde
            | StageProblemKind::TopologySparseNoisyFieldFluxObservationsNoPde
            | StageProblemKind::TopologySparseNoisyObservationsNoPde
    )
}

fn topology_pushforward_training_observations(
    workspace: &ExperimentWorkspace,
    kind: StageProblemKind,
) -> Vec<&ObservationSpec> {
    workspace
        .observations
        .iter()
        .filter(|observation| match kind {
            StageProblemKind::TopologyPushforwardFieldProbes => observation.sensor_type == "hall",
            StageProblemKind::TopologyPushforwardFluxPanels => {
                observation.sensor_type == "hall" || observation.sensor_type == "flux"
            }
            StageProblemKind::TopologyPushforwardAmpereLoop => {
                observation.sensor_type == "hall"
                    || observation.sensor_type == "flux"
                    || observation.sensor_type == AMPERE_LOOP_SENSOR_TYPE
            }
            StageProblemKind::TopologySparseNoisyFieldObservations
            | StageProblemKind::TopologySparseNoisyFieldObservationsNoPde => {
                observation.sensor_type == "hall"
            }
            StageProblemKind::TopologySparseNoisyFieldFluxObservations
            | StageProblemKind::TopologySparseNoisyFieldFluxObservationsNoPde => {
                observation.sensor_type == "hall" || observation.sensor_type == "flux"
            }
            StageProblemKind::TopologySparseNoisyObservations
            | StageProblemKind::TopologySparseNoisyObservationsNoPde => true,
            _ => false,
        })
        .collect()
}

fn build_problem(
    workspace: &ExperimentWorkspace,
    config: &ToroidalHarmonicBConfig,
    kind: StageProblemKind,
    beta_prior_std: f64,
    truth: ObservationTruth,
) -> Result<LinearPdeUqProblem, String> {
    let mut system = workspace.system.clone();
    let is_fluctuation_recovery = matches!(
        kind,
        StageProblemKind::FluctuationAlphaBetaRecovery
            | StageProblemKind::AmpereLoopAlphaBetaRecovery
    );
    let is_source_generated_fluctuation = matches!(
        kind,
        StageProblemKind::SourceGeneratedPdeOnly
            | StageProblemKind::SourceGeneratedFieldRecovery
            | StageProblemKind::SourceGeneratedAmpereLoop
    );
    let is_source_generated_full_state = matches!(
        kind,
        StageProblemKind::SourceGeneratedFullStatePdeOnly
            | StageProblemKind::SourceGeneratedFullStateFieldRecovery
            | StageProblemKind::SourceGeneratedFullStateAmpereLoop
            | StageProblemKind::TopologyPushforwardPriorOnly
            | StageProblemKind::TopologyPushforwardPdeOnly
            | StageProblemKind::TopologyPushforwardFieldProbes
            | StageProblemKind::TopologyPushforwardFluxPanels
            | StageProblemKind::TopologyPushforwardAmpereLoop
            | StageProblemKind::TopologySparseNoisyPriorOnly
            | StageProblemKind::TopologySparseNoisyPdeOnly
            | StageProblemKind::TopologySparseNoisyFieldObservations
            | StageProblemKind::TopologySparseNoisyFieldFluxObservations
            | StageProblemKind::TopologySparseNoisyObservations
            | StageProblemKind::TopologySparseNoisyFieldObservationsNoPde
            | StageProblemKind::TopologySparseNoisyFieldFluxObservationsNoPde
            | StageProblemKind::TopologySparseNoisyObservationsNoPde
    );
    let is_source_generated = is_source_generated_fluctuation || is_source_generated_full_state;
    if is_fluctuation_recovery || is_source_generated {
        system.residual_bias.fill(0.0);
    }
    let use_fixed_nominal_source = matches!(
        kind,
        StageProblemKind::PriorOnly
            | StageProblemKind::FixedSourceBetaLatentPdeOnly
            | StageProblemKind::FixedBetaControl
            | StageProblemKind::JointBetaRecovery
    );
    if use_fixed_nominal_source {
        subtract_from_bias(system.residual_bias.as_mut_slice(), &workspace.source_rhs)?;
    }

    let mut uncertain_inputs = Vec::new();
    if matches!(
        kind,
        StageProblemKind::PriorOnly
            | StageProblemKind::FixedSourceBetaLatentPdeOnly
            | StageProblemKind::JointBetaRecovery
            | StageProblemKind::JointAlphaBetaStress
            | StageProblemKind::FluctuationAlphaBetaRecovery
            | StageProblemKind::AmpereLoopAlphaBetaRecovery
    ) {
        uncertain_inputs.push(LinearUncertainInputSpec {
            name: BETA_INPUT_NAME.to_string(),
            operator: SparseTripletMatrix::new(system.residual_dimension(), 1),
            prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: diagonal_precision(1, 1.0 / (beta_prior_std * beta_prior_std)),
            },
            preference: RepresentationPreference::ForceLatent,
            collapsed_precision: None,
        });
    }
    if matches!(
        kind,
        StageProblemKind::JointAlphaBetaStress
            | StageProblemKind::FluctuationAlphaBetaRecovery
            | StageProblemKind::AmpereLoopAlphaBetaRecovery
            | StageProblemKind::SourceGeneratedPdeOnly
            | StageProblemKind::SourceGeneratedFieldRecovery
            | StageProblemKind::SourceGeneratedAmpereLoop
            | StageProblemKind::SourceGeneratedFullStatePdeOnly
            | StageProblemKind::SourceGeneratedFullStateFieldRecovery
            | StageProblemKind::SourceGeneratedFullStateAmpereLoop
            | StageProblemKind::TopologyPushforwardPriorOnly
            | StageProblemKind::TopologyPushforwardPdeOnly
            | StageProblemKind::TopologyPushforwardFieldProbes
            | StageProblemKind::TopologyPushforwardFluxPanels
            | StageProblemKind::TopologyPushforwardAmpereLoop
            | StageProblemKind::TopologySparseNoisyPriorOnly
            | StageProblemKind::TopologySparseNoisyPdeOnly
            | StageProblemKind::TopologySparseNoisyFieldObservations
            | StageProblemKind::TopologySparseNoisyFieldFluxObservations
            | StageProblemKind::TopologySparseNoisyObservations
            | StageProblemKind::TopologySparseNoisyFieldObservationsNoPde
            | StageProblemKind::TopologySparseNoisyFieldFluxObservationsNoPde
            | StageProblemKind::TopologySparseNoisyObservationsNoPde
    ) {
        let operator = if is_fluctuation_recovery || is_source_generated_fluctuation {
            zero_source_operator(system.residual_dimension(), 1)
        } else {
            workspace.source_operator.clone()
        };
        uncertain_inputs.push(LinearUncertainInputSpec {
            name: ALPHA_INPUT_NAME.to_string(),
            operator,
            prior: GaussianPriorSpec {
                mean: vec![1.0],
                precision: diagonal_precision(1, 1.0 / (config.source_prior_std.powi(2))),
            },
            preference: RepresentationPreference::ForceLatent,
            collapsed_precision: None,
        });
    }

    let physical_measurements = if matches!(kind, StageProblemKind::FixedBetaControl) {
        workspace
            .observations
            .iter()
            .map(|observation| LinearGaussianMeasurementSpec {
                name: observation.name.clone(),
                operator: observation.state_operator.clone(),
                observations: vec![observation_value(observation, truth)],
                bias: vec![0.0],
                variance: observation_variance(observation, truth),
            })
            .collect()
    } else {
        Vec::new()
    };
    let joint_measurements = if is_topology_pushforward_kind(kind) {
        topology_pushforward_training_observations(workspace, kind)
            .into_iter()
            .map(|observation| {
                source_generated_joint_measurement_spec(
                    observation,
                    workspace,
                    is_source_generated_full_state,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if is_fluctuation_recovery {
        workspace
            .observations
            .iter()
            .map(|observation| fluctuation_joint_measurement_spec(observation, workspace, truth))
            .collect::<Result<Vec<_>, _>>()?
    } else if matches!(
        kind,
        StageProblemKind::SourceGeneratedFieldRecovery
            | StageProblemKind::SourceGeneratedAmpereLoop
            | StageProblemKind::SourceGeneratedFullStateFieldRecovery
            | StageProblemKind::SourceGeneratedFullStateAmpereLoop
    ) {
        workspace
            .observations
            .iter()
            .map(|observation| {
                source_generated_joint_measurement_spec(
                    observation,
                    workspace,
                    is_source_generated_full_state,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if matches!(
        kind,
        StageProblemKind::JointBetaRecovery | StageProblemKind::JointAlphaBetaStress
    ) {
        workspace
            .observations
            .iter()
            .map(|observation| joint_measurement_spec(observation, truth))
            .collect()
    } else {
        Vec::new()
    };

    let has_beta = uncertain_inputs
        .iter()
        .any(|input| input.name == BETA_INPUT_NAME);
    let (derived_quantities, joint_derived_quantities) = build_stage_derived_quantities(
        workspace,
        has_beta,
        is_fluctuation_recovery,
        is_source_generated_fluctuation,
        is_source_generated_full_state,
        &workspace.observations,
        &workspace.heldout_observations,
        is_topology_pushforward_kind(kind),
        config.include_full_field_variance_maps,
    )?;

    let (pde_variance, pde_precision) = match kind {
        StageProblemKind::PriorOnly
        | StageProblemKind::TopologyPushforwardPriorOnly
        | StageProblemKind::TopologySparseNoisyPriorOnly
        | StageProblemKind::TopologySparseNoisyFieldObservationsNoPde
        | StageProblemKind::TopologySparseNoisyFieldFluxObservationsNoPde
        | StageProblemKind::TopologySparseNoisyObservationsNoPde => (None, None),
        _ if config.use_mass_weighted_pde_residual => {
            (None, Some(mass_weighted_pde_precision(&system, config)?))
        }
        _ => (Some(config.pde_variance), None),
    };

    let state_prior = if is_fluctuation_recovery || is_source_generated {
        scale_gaussian_prior_precision(
            workspace.state_prior.clone(),
            config.fluctuation_state_prior_precision_scale,
        )?
    } else {
        workspace.state_prior.clone()
    };

    Ok(LinearPdeUqProblem {
        state_prior,
        system,
        uncertain_inputs,
        physical_measurements,
        joint_measurements,
        derived_quantities,
        joint_derived_quantities,
        pde_variance,
        pde_precision,
    })
}

fn build_stage_derived_quantities(
    workspace: &ExperimentWorkspace,
    has_beta: bool,
    is_fluctuation_recovery: bool,
    is_source_generated_fluctuation: bool,
    is_source_generated_full_state: bool,
    observations: &[ObservationSpec],
    heldout_observations: &[ObservationSpec],
    include_topology_pushforward_qois: bool,
    include_full_field_variance_maps: bool,
) -> Result<
    (
        Vec<LinearPdeDerivedQuantitySpec>,
        Vec<LinearPdeJointDerivedQuantitySpec>,
    ),
    String,
> {
    let b_exact = if include_full_field_variance_maps {
        Some(build_exterior_derivative_row_operator(&workspace.topology)?)
    } else {
        None
    };
    let mut derived_quantities = Vec::new();
    let mut joint_derived_quantities = Vec::new();

    let is_source_generated = is_source_generated_fluctuation || is_source_generated_full_state;
    if is_source_generated {
        if let Some(b_exact) = b_exact {
            derived_quantities.push(LinearPdeDerivedQuantitySpec {
                name: B_EXACT_DERIVED_NAME.to_string(),
                operator: b_exact.clone(),
            });
            let source_generated_b_operator = if is_source_generated_full_state {
                workspace
                    .h2
                    .scale(workspace.topology_summary.source_harmonic_kappa)
            } else {
                workspace.truth_b_source_generated_unit.clone()
            };
            joint_derived_quantities.push(LinearPdeJointDerivedQuantitySpec {
                name: B_TOTAL_DERIVED_NAME.to_string(),
                state_operator: Some(b_exact),
                latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                    input_name: ALPHA_INPUT_NAME.to_string(),
                    operator: vector_column_row_operator(&source_generated_b_operator)?,
                }],
            });
        }
        joint_derived_quantities.extend(source_generated_scalar_derived_specs(
            workspace,
            is_source_generated_full_state,
        )?);
        if include_topology_pushforward_qois {
            joint_derived_quantities.extend(topology_pushforward_qoi_derived_specs(workspace)?);
        }
        for observation in observations.iter().chain(heldout_observations.iter()) {
            joint_derived_quantities.push(source_generated_observation_derived_spec(
                observation,
                workspace,
                is_source_generated_full_state,
            )?);
        }
    } else if has_beta {
        if let Some(b_exact) = b_exact {
            derived_quantities.push(LinearPdeDerivedQuantitySpec {
                name: B_EXACT_DERIVED_NAME.to_string(),
                operator: b_exact.clone(),
            });
            let mut latent_operators = vec![LinearPdeLatentDerivedBlockSpec {
                input_name: BETA_INPUT_NAME.to_string(),
                operator: vector_column_row_operator(&workspace.h2)?,
            }];
            if is_fluctuation_recovery {
                latent_operators.push(LinearPdeLatentDerivedBlockSpec {
                    input_name: ALPHA_INPUT_NAME.to_string(),
                    operator: vector_column_row_operator(&workspace.truth_b_exact_nominal)?,
                });
            }
            joint_derived_quantities.push(LinearPdeJointDerivedQuantitySpec {
                name: B_TOTAL_DERIVED_NAME.to_string(),
                state_operator: Some(b_exact),
                latent_operators,
            });
        }
        for observation in observations {
            joint_derived_quantities.push(joint_observation_derived_spec(
                observation,
                workspace,
                is_fluctuation_recovery,
            )?);
        }
    } else {
        if let Some(b_exact) = b_exact {
            derived_quantities.push(LinearPdeDerivedQuantitySpec {
                name: B_TOTAL_DERIVED_NAME.to_string(),
                operator: b_exact,
            });
        }
        for observation in observations {
            derived_quantities.push(LinearPdeDerivedQuantitySpec {
                name: observation_derived_name(&observation.name),
                operator: triplet_to_row_operator(&observation.state_operator)?,
            });
        }
    }

    Ok((derived_quantities, joint_derived_quantities))
}

fn run_stage(
    name: &str,
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    problem: LinearPdeUqProblem,
    truth: ObservationTruth,
    prediction: StageObservationPrediction,
) -> Result<ToroidalStageResult, Box<dyn Error>> {
    let result = solve_linear_pde_uq_with_config(&problem, &config.solver)
        .map_err(|err| invalid_input(format!("{name} failed: {err}")))?;
    let reports = observation_reports(name, workspace, &result, truth, prediction)?;
    let hall_rmse = rmse_for_kind(&reports, "hall");
    let flux_rmse = rmse_for_kind(&reports, "flux");
    let harmonic_observation_residual = reports
        .iter()
        .find(|row| row.sensor_type == "harmonic")
        .map(|row| row.residual)
        .unwrap_or(0.0);
    let b_variance_ratio_mean = result
        .derived_variances
        .get(B_TOTAL_DERIVED_NAME)
        .map(mean_variance_ratio)
        .unwrap_or(f64::NAN);
    let beta_prior = latent_prior_summary(&problem, BETA_INPUT_NAME);
    let alpha_prior = latent_prior_summary(&problem, ALPHA_INPUT_NAME);
    let beta_post = latent_posterior_summary(&result, BETA_INPUT_NAME);
    let alpha_post = latent_posterior_summary(&result, ALPHA_INPUT_NAME);

    Ok(ToroidalStageResult {
        summary: ToroidalStageSummary {
            stage: name.to_string(),
            latent_dimension: result.debug.joint_dimension,
            pde_residual_norm: result.pde_residual_mean.norm(),
            hall_rmse,
            flux_rmse,
            harmonic_observation_residual,
            b_variance_ratio_mean,
            beta_prior_mean: beta_prior.map(|summary| summary.0),
            beta_prior_variance: beta_prior.map(|summary| summary.1),
            beta_posterior_mean: beta_post.map(|summary| summary.0),
            beta_posterior_variance: beta_post.map(|summary| summary.1),
            beta_error: beta_post.map(|summary| summary.0 - workspace.topology_summary.beta_true),
            alpha_prior_mean: alpha_prior.map(|summary| summary.0),
            alpha_prior_variance: alpha_prior.map(|summary| summary.1),
            alpha_posterior_mean: alpha_post.map(|summary| summary.0),
            alpha_posterior_variance: alpha_post.map(|summary| summary.1),
            alpha_error: alpha_post.map(|summary| summary.0 - config.source_alpha_true),
        },
        solve: result,
        observations: reports,
        pushforward_qois: Vec::new(),
        pushforward_covariances: Vec::new(),
        heldout_predictions: Vec::new(),
        branch_decomposition: Vec::new(),
        field_trace_variance: Vec::new(),
    })
}

fn report_stage_progress(stage: &ToroidalStageResult) {
    let s = &stage.summary;
    let prior = stage.solve.debug.prior_factorization;
    let posterior = stage.solve.debug.posterior_factorization;
    eprintln!(
        concat!(
            "[toroidal] stage_done name={} latent_dimension={} pde_residual={:.3e} ",
            "hall_rmse={:.3e} flux_rmse={:.3e} harmonic_residual={:.3e} ",
            "beta_mean={} beta_var={} alpha_mean={} alpha_var={} ",
            "prior_precision_nnz={} prior_factor_nnz={} prior_fill={:.3}x ",
            "posterior_precision_nnz={} posterior_factor_nnz={} posterior_fill={:.3}x ",
            "posterior_factor_values_mib={:.3}"
        ),
        s.stage,
        s.latent_dimension,
        s.pde_residual_norm,
        s.hall_rmse,
        s.flux_rmse,
        s.harmonic_observation_residual,
        display_option(s.beta_posterior_mean),
        display_option(s.beta_posterior_variance),
        display_option(s.alpha_posterior_mean),
        display_option(s.alpha_posterior_variance),
        prior.matrix_nnz,
        prior.factor_nnz,
        prior.fill_in_ratio_vs_lower_triangle,
        posterior.matrix_nnz,
        posterior.factor_nnz,
        posterior.fill_in_ratio_vs_lower_triangle,
        posterior.factor_numeric_values_mib
    );
}

fn report_topology_pushforward_progress(stage: &ToroidalStageResult) {
    if stage.pushforward_qois.is_empty() {
        return;
    }
    let qoi = |name: &str| {
        stage
            .pushforward_qois
            .iter()
            .find(|row| row.qoi == name)
            .map(|row| (row.mean, row.posterior_variance, row.variance_ratio))
    };
    let source = qoi(QOI_SOURCE_NAME);
    let beta = qoi(QOI_SOURCE_BETA_NAME);
    let harmonic = qoi(QOI_HARMONIC_PROJECTION_NAME);
    let linked = qoi(QOI_LINK_FLUX_NAME);
    let loop_current = qoi(QOI_AMPERE_LOOP_NAME);
    eprintln!(
        concat!(
            "[toroidal] pushforward_done stage={} ",
            "s={} beta_H={} eta_H={} Phi_link={} I_gamma={} ",
            "var_ratio_s={} var_ratio_eta={} var_ratio_I={}"
        ),
        stage.summary.stage,
        display_option(source.map(|value| value.0)),
        display_option(beta.map(|value| value.0)),
        display_option(harmonic.map(|value| value.0)),
        display_option(linked.map(|value| value.0)),
        display_option(loop_current.map(|value| value.0)),
        display_option(source.map(|value| value.2)),
        display_option(harmonic.map(|value| value.2)),
        display_option(loop_current.map(|value| value.2)),
    );
}

fn attach_topology_pushforward_reports(
    stage: &mut ToroidalStageResult,
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    problem: &LinearPdeUqProblem,
    prediction: StageObservationPrediction,
) -> Result<(), Box<dyn Error>> {
    let covariance_names = topology_pushforward_covariance_names();
    let covariance =
        solve_linear_pde_uq_with_pushforward_covariance(problem, &config.solver, &covariance_names)
            .map_err(|err| invalid_input(format!("pushforward covariance failed: {err}")))?;
    stage.pushforward_qois = topology_pushforward_qoi_rows(stage, config, workspace, &covariance)?;
    stage.pushforward_covariances =
        topology_pushforward_covariance_rows(&stage.summary.stage, &covariance);
    stage.heldout_predictions = topology_pushforward_heldout_rows(stage, workspace, prediction)?;
    stage.branch_decomposition =
        topology_pushforward_branch_decomposition(stage, workspace, &covariance)?;
    stage.field_trace_variance = topology_pushforward_field_trace_rows(stage);
    Ok(())
}

fn topology_pushforward_covariance_names() -> Vec<String> {
    topology_pushforward_main_qoi_names()
        .into_iter()
        .chain([BRANCH_LINK_FLUX_EXACT_NAME.to_string()])
        .collect()
}

fn topology_pushforward_main_qoi_names() -> Vec<String> {
    [
        QOI_SOURCE_NAME,
        QOI_SOURCE_BETA_NAME,
        QOI_HARMONIC_PROJECTION_NAME,
        QOI_LINK_FLUX_NAME,
        QOI_LOCAL_FLUX_NAME,
        QOI_AMPERE_LOOP_NAME,
        QOI_FIELD_X_NAME,
        QOI_FIELD_Y_NAME,
        QOI_FIELD_Z_NAME,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn topology_pushforward_qoi_rows(
    stage: &ToroidalStageResult,
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    covariance: &LinearPdePushforwardCovarianceResult,
) -> Result<Vec<ToroidalPushforwardQoiRow>, Box<dyn Error>> {
    let main_names = topology_pushforward_main_qoi_names();
    let mut rows = Vec::with_capacity(main_names.len());
    for name in main_names {
        let index = covariance_index(covariance, &name)?;
        let prior_variance = covariance.prior_covariance[(index, index)].max(0.0);
        let posterior_variance = covariance.posterior_covariance[(index, index)].max(0.0);
        let sd = posterior_variance.sqrt();
        let mean = topology_pushforward_qoi_mean(stage, workspace, &name)?;
        let truth = topology_pushforward_qoi_truth(config, workspace, &name)?;
        rows.push(ToroidalPushforwardQoiRow {
            stage: stage.summary.stage.clone(),
            qoi: name.clone(),
            role: topology_pushforward_qoi_role(&name).to_string(),
            truth,
            mean,
            sd,
            lower95: mean - 1.96 * sd,
            upper95: mean + 1.96 * sd,
            prior_variance,
            posterior_variance,
            variance_ratio: if prior_variance > EPS {
                posterior_variance / prior_variance
            } else {
                f64::NAN
            },
            unit: topology_pushforward_qoi_unit(&name).to_string(),
        });
    }
    Ok(rows)
}

fn topology_pushforward_covariance_rows(
    stage: &str,
    covariance: &LinearPdePushforwardCovarianceResult,
) -> Vec<ToroidalPushforwardCovarianceRow> {
    let main_names = topology_pushforward_main_qoi_names();
    let mut rows = Vec::new();
    for (i, qoi_i) in main_names.iter().enumerate() {
        for (j, qoi_j) in main_names.iter().enumerate() {
            let prior_covariance = covariance.prior_covariance[(i, j)];
            let posterior_covariance = covariance.posterior_covariance[(i, j)];
            let denom = (covariance.posterior_covariance[(i, i)].max(0.0)
                * covariance.posterior_covariance[(j, j)].max(0.0))
            .sqrt();
            rows.push(ToroidalPushforwardCovarianceRow {
                stage: stage.to_string(),
                qoi_i: qoi_i.clone(),
                qoi_j: qoi_j.clone(),
                prior_covariance,
                posterior_covariance,
                posterior_correlation: if denom > EPS {
                    posterior_covariance / denom
                } else {
                    f64::NAN
                },
            });
        }
    }
    rows
}

fn topology_pushforward_heldout_rows(
    stage: &ToroidalStageResult,
    workspace: &ExperimentWorkspace,
    prediction: StageObservationPrediction,
) -> Result<Vec<ToroidalHeldoutPredictionRow>, Box<dyn Error>> {
    topology_pushforward_prediction_rows_for_observations(
        stage,
        workspace,
        prediction,
        workspace.heldout_observations.iter().collect(),
    )
}

fn topology_pushforward_prediction_rows_for_observations(
    stage: &ToroidalStageResult,
    workspace: &ExperimentWorkspace,
    prediction: StageObservationPrediction,
    observations: Vec<&ObservationSpec>,
) -> Result<Vec<ToroidalHeldoutPredictionRow>, Box<dyn Error>> {
    observations
        .into_iter()
        .map(|observation| {
            let prediction_value =
                predict_observation(workspace, &stage.solve, observation, prediction)
                    .map_err(invalid_input)?;
            let truth = observation.observation_source_generated_truth;
            let derived_name = observation_derived_name(&observation.name);
            let variance = stage
                .solve
                .derived_variances
                .get(&derived_name)
                .ok_or_else(|| {
                    invalid_input(format!("missing held-out variance `{derived_name}`"))
                })?
                .posterior_variance[0]
                .max(0.0);
            let sd = variance.sqrt();
            let residual = prediction_value - truth;
            Ok(ToroidalHeldoutPredictionRow {
                stage: stage.summary.stage.clone(),
                sensor_type: observation.sensor_type.clone(),
                name: observation.name.clone(),
                truth,
                prediction: prediction_value,
                residual,
                posterior_sd: sd,
                standardized_residual: if sd > EPS { residual / sd } else { f64::NAN },
                lower95: prediction_value - 1.96 * sd,
                upper95: prediction_value + 1.96 * sd,
                covered95: truth >= prediction_value - 1.96 * sd
                    && truth <= prediction_value + 1.96 * sd,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()
}

fn topology_pushforward_branch_decomposition(
    stage: &ToroidalStageResult,
    workspace: &ExperimentWorkspace,
    covariance: &LinearPdePushforwardCovarianceResult,
) -> Result<Vec<ToroidalBranchDecompositionRow>, Box<dyn Error>> {
    let exact = covariance_index(covariance, BRANCH_LINK_FLUX_EXACT_NAME)?;
    let source = covariance_index(covariance, QOI_SOURCE_NAME)?;
    let total = covariance_index(covariance, QOI_LINK_FLUX_NAME)?;
    let link_flux = topology_pushforward_observation(workspace, "embedded_flux_panel_p0")
        .map_err(invalid_input)?;
    let harmonic_weight =
        workspace.topology_summary.source_harmonic_kappa * link_flux.beta_operator_value;
    let decompose = |matrix: &gmrf_core::types::DenseMatrix| {
        let exact_variance = matrix[(exact, exact)];
        let source_variance = harmonic_weight * harmonic_weight * matrix[(source, source)];
        let coupling = 2.0 * harmonic_weight * matrix[(exact, source)];
        let total_variance = exact_variance + source_variance + coupling;
        let reported = matrix[(total, total)];
        (
            exact_variance,
            source_variance,
            coupling,
            total_variance,
            reported,
        )
    };
    let prior = decompose(&covariance.prior_covariance);
    let posterior = decompose(&covariance.posterior_covariance);
    Ok(vec![ToroidalBranchDecompositionRow {
        stage: stage.summary.stage.clone(),
        functional: QOI_LINK_FLUX_NAME.to_string(),
        prior_exact_variance: prior.0,
        prior_source_harmonic_variance: prior.1,
        prior_coupling_variance: prior.2,
        prior_total_variance: prior.3,
        prior_reported_variance: prior.4,
        posterior_exact_variance: posterior.0,
        posterior_source_harmonic_variance: posterior.1,
        posterior_coupling_variance: posterior.2,
        posterior_total_variance: posterior.3,
        posterior_reported_variance: posterior.4,
    }])
}

fn topology_pushforward_field_trace_rows(
    stage: &ToroidalStageResult,
) -> Vec<ToroidalFieldTraceVarianceRow> {
    let mut rows = Vec::new();
    for phi in 0..4 {
        for theta in 0..4 {
            let point = format!("embedded_hall_p{phi}_t{theta}");
            let names = ["bx", "by", "bz"].map(|component| {
                observation_derived_name(&format!("embedded_hall_p{phi}_t{theta}_{component}"))
            });
            let mut prior_trace = 0.0;
            let mut posterior_trace = 0.0;
            let mut complete = true;
            for name in names {
                if let Some(variance) = stage.solve.derived_variances.get(&name) {
                    prior_trace += variance.prior_variance[0].max(0.0);
                    posterior_trace += variance.posterior_variance[0].max(0.0);
                } else {
                    complete = false;
                }
            }
            if complete {
                rows.push(ToroidalFieldTraceVarianceRow {
                    stage: stage.summary.stage.clone(),
                    point,
                    prior_trace_variance: prior_trace,
                    posterior_trace_variance: posterior_trace,
                    variance_ratio: if prior_trace > EPS {
                        posterior_trace / prior_trace
                    } else {
                        f64::NAN
                    },
                });
            }
        }
    }
    rows
}

fn covariance_index(
    covariance: &LinearPdePushforwardCovarianceResult,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    covariance
        .names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| invalid_input(format!("missing pushforward covariance row `{name}`")).into())
}

fn topology_pushforward_qoi_mean(
    stage: &ToroidalStageResult,
    workspace: &ExperimentWorkspace,
    name: &str,
) -> Result<f64, Box<dyn Error>> {
    let alpha = stage.summary.alpha_posterior_mean.unwrap_or(1.0);
    match name {
        QOI_SOURCE_NAME => Ok(alpha),
        QOI_SOURCE_BETA_NAME => Ok(workspace.topology_summary.source_harmonic_kappa * alpha),
        QOI_HARMONIC_PROJECTION_NAME => source_generated_harmonic_projection_mean(workspace, stage)
            .ok_or_else(|| invalid_input("missing harmonic projection mean").into()),
        QOI_LINK_FLUX_NAME | QOI_LOCAL_FLUX_NAME | QOI_AMPERE_LOOP_NAME | QOI_FIELD_X_NAME
        | QOI_FIELD_Y_NAME | QOI_FIELD_Z_NAME => {
            let observation_name = topology_pushforward_qoi_observation_name(name)
                .ok_or_else(|| invalid_input(format!("QoI `{name}` has no observation mapping")))?;
            let observation = topology_pushforward_observation(workspace, observation_name)
                .map_err(invalid_input)?;
            predict_observation(
                workspace,
                &stage.solve,
                observation,
                StageObservationPrediction::SourceGeneratedFullStateAlpha,
            )
            .map_err(|err| invalid_input(err).into())
        }
        _ => Err(invalid_input(format!("unknown topology pushforward QoI `{name}`")).into()),
    }
}

fn topology_pushforward_qoi_truth(
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    name: &str,
) -> Result<f64, Box<dyn Error>> {
    match name {
        QOI_SOURCE_NAME => Ok(config.source_alpha_true),
        QOI_SOURCE_BETA_NAME => {
            Ok(workspace.topology_summary.source_harmonic_kappa * config.source_alpha_true)
        }
        QOI_HARMONIC_PROJECTION_NAME => {
            Ok(workspace.topology_summary.source_harmonic_projection_true)
        }
        QOI_LINK_FLUX_NAME | QOI_LOCAL_FLUX_NAME | QOI_AMPERE_LOOP_NAME | QOI_FIELD_X_NAME
        | QOI_FIELD_Y_NAME | QOI_FIELD_Z_NAME => {
            let observation_name = topology_pushforward_qoi_observation_name(name)
                .ok_or_else(|| invalid_input(format!("QoI `{name}` has no observation mapping")))?;
            Ok(
                topology_pushforward_observation(workspace, observation_name)
                    .map_err(invalid_input)?
                    .observation_source_generated_truth,
            )
        }
        _ => Err(invalid_input(format!("unknown topology pushforward QoI `{name}`")).into()),
    }
}

fn topology_pushforward_qoi_observation_name(name: &str) -> Option<&'static str> {
    match name {
        QOI_LINK_FLUX_NAME => Some("embedded_flux_panel_p0"),
        QOI_LOCAL_FLUX_NAME => Some("flux_top"),
        QOI_AMPERE_LOOP_NAME => Some(AMPERE_LOOP_NAME),
        QOI_FIELD_X_NAME => Some("embedded_hall_p0_t0_bx"),
        QOI_FIELD_Y_NAME => Some("embedded_hall_p0_t0_by"),
        QOI_FIELD_Z_NAME => Some("embedded_hall_p0_t0_bz"),
        _ => None,
    }
}

fn topology_pushforward_qoi_role(name: &str) -> &'static str {
    match name {
        QOI_SOURCE_NAME => "source",
        QOI_SOURCE_BETA_NAME | QOI_HARMONIC_PROJECTION_NAME => "topology",
        QOI_LINK_FLUX_NAME | QOI_LOCAL_FLUX_NAME => "flux",
        QOI_AMPERE_LOOP_NAME => "circulation",
        QOI_FIELD_X_NAME | QOI_FIELD_Y_NAME | QOI_FIELD_Z_NAME => "field",
        _ => "internal",
    }
}

fn topology_pushforward_qoi_unit(name: &str) -> &'static str {
    match name {
        QOI_SOURCE_NAME => "dimensionless",
        QOI_SOURCE_BETA_NAME | QOI_HARMONIC_PROJECTION_NAME => "weighted flux coordinate",
        QOI_LINK_FLUX_NAME | QOI_LOCAL_FLUX_NAME => "flux proxy",
        QOI_AMPERE_LOOP_NAME => "A",
        QOI_FIELD_X_NAME | QOI_FIELD_Y_NAME | QOI_FIELD_Z_NAME => "T proxy",
        _ => "",
    }
}

#[derive(Debug, Clone, Copy)]
struct SparseComparisonSpec {
    label: &'static str,
    gp_stage: &'static str,
    obs_only_stage: &'static str,
    full_stage: &'static str,
    obs_only_kind: StageProblemKind,
    full_kind: StageProblemKind,
}

fn sparse_comparison_heldout_observations<'a>(
    workspace: &'a ExperimentWorkspace,
    training: &[&'a ObservationSpec],
) -> Vec<&'a ObservationSpec> {
    let training_names = training
        .iter()
        .map(|observation| observation.name.clone())
        .collect::<HashSet<_>>();
    workspace
        .observations
        .iter()
        .filter(|observation| !training_names.contains(&observation.name))
        .chain(workspace.heldout_observations.iter())
        .collect()
}

fn run_sparse_comparison_feec_stage(
    stage_name: &str,
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    kind: StageProblemKind,
    beta_prior_std: f64,
    heldout: Vec<&ObservationSpec>,
) -> Result<ToroidalStageResult, Box<dyn Error>> {
    eprintln!(
        "[toroidal-sparse-comparison] stage_start name={} variance_mode={}",
        stage_name,
        variance_mode_name(config.solver.variance.mode)
    );
    let problem = build_problem(
        workspace,
        config,
        kind,
        beta_prior_std,
        ObservationTruth::SourceGenerated,
    )?;
    let problem_for_pushforward = problem.clone();
    let mut stage = run_stage(
        stage_name,
        config,
        workspace,
        problem,
        ObservationTruth::SourceGenerated,
        StageObservationPrediction::SourceGeneratedFullStateAlpha,
    )?;
    attach_topology_pushforward_reports(
        &mut stage,
        config,
        workspace,
        &problem_for_pushforward,
        StageObservationPrediction::SourceGeneratedFullStateAlpha,
    )?;
    stage.heldout_predictions = topology_pushforward_prediction_rows_for_observations(
        &stage,
        workspace,
        StageObservationPrediction::SourceGeneratedFullStateAlpha,
        heldout,
    )?;
    Ok(stage)
}

fn sparse_feec_metric_rows(
    model: &str,
    stage: &ToroidalStageResult,
    training_rows: usize,
    pde_residual_used: bool,
) -> Vec<ToroidalSparseComparisonMetricRow> {
    let values = stage
        .heldout_predictions
        .iter()
        .map(|row| PredictionMetricValue {
            sensor_family: sensor_family(&row.sensor_type).to_string(),
            residual: row.residual,
            sd: row.posterior_sd,
            covered95: row.covered95,
            standardized_residual: row.standardized_residual,
        })
        .collect::<Vec<_>>();
    sparse_metric_rows_from_values(
        model,
        &stage.summary.stage,
        training_rows,
        pde_residual_used,
        true,
        true,
        &values,
    )
}

fn sparse_gp_metric_rows(
    stage: &ToroidalGpBaselineStageResult,
) -> Vec<ToroidalSparseComparisonMetricRow> {
    let values = stage
        .heldout_predictions
        .iter()
        .map(|row| PredictionMetricValue {
            sensor_family: sensor_family(&row.sensor_type).to_string(),
            residual: row.residual,
            sd: row.posterior_sd,
            covered95: row.covered95,
            standardized_residual: row.standardized_residual,
        })
        .collect::<Vec<_>>();
    sparse_metric_rows_from_values(
        "independent-output GP",
        &stage.summary.stage,
        stage.summary.training_rows,
        false,
        false,
        false,
        &values,
    )
}

fn source_template_gp_metric_rows(
    stage: &ToroidalSourceTemplateGpStageResult,
) -> Vec<ToroidalSparseComparisonMetricRow> {
    let values = stage
        .heldout_predictions
        .iter()
        .map(|row| PredictionMetricValue {
            sensor_family: sensor_family(&row.sensor_type).to_string(),
            residual: row.residual,
            sd: row.posterior_sd,
            covered95: row.covered95,
            standardized_residual: row.standardized_residual,
        })
        .collect::<Vec<_>>();
    sparse_metric_rows_from_values(
        &stage.summary.model,
        &stage.summary.stage,
        stage.summary.training_rows,
        false,
        true,
        stage.summary.template_kind == SourceTemplateKind::TopologyOracle.label(),
        &values,
    )
}

#[derive(Debug, Clone)]
struct PredictionMetricValue {
    sensor_family: String,
    residual: f64,
    sd: f64,
    covered95: bool,
    standardized_residual: f64,
}

fn sparse_metric_rows_from_values(
    model: &str,
    stage: &str,
    training_rows: usize,
    pde_residual_used: bool,
    source_posterior_available: bool,
    topology_posterior_available: bool,
    values: &[PredictionMetricValue],
) -> Vec<ToroidalSparseComparisonMetricRow> {
    let mut rows = Vec::new();
    for family in ["all", "hall", "flux", "ampere_loop"] {
        let subset = values
            .iter()
            .filter(|value| family == "all" || value.sensor_family == family)
            .collect::<Vec<_>>();
        if subset.is_empty() {
            continue;
        }
        let heldout_rows = subset.len();
        let rmse = (subset
            .iter()
            .map(|value| value.residual * value.residual)
            .sum::<f64>()
            / heldout_rows as f64)
            .sqrt();
        let nlpd = subset
            .iter()
            .map(|value| {
                let variance = value.sd * value.sd;
                if variance <= 0.0 {
                    f64::INFINITY
                } else {
                    0.5 * ((2.0 * PI * variance).ln() + value.residual * value.residual / variance)
                }
            })
            .sum::<f64>()
            / heldout_rows as f64;
        let covered95 = subset.iter().filter(|value| value.covered95).count();
        let max_abs_standardized_residual = subset
            .iter()
            .map(|value| value.standardized_residual.abs())
            .fold(0.0_f64, f64::max);
        let mean_abs_standardized_residual = subset
            .iter()
            .map(|value| value.standardized_residual.abs())
            .sum::<f64>()
            / heldout_rows as f64;
        rows.push(ToroidalSparseComparisonMetricRow {
            model: model.to_string(),
            stage: stage.to_string(),
            sensor_family: family.to_string(),
            training_rows,
            heldout_rows,
            rmse,
            nlpd,
            covered95,
            coverage_fraction: covered95 as f64 / heldout_rows as f64,
            max_abs_standardized_residual,
            mean_abs_standardized_residual,
            source_posterior_available,
            topology_posterior_available,
            pde_residual_used,
        });
    }
    rows
}

fn sensor_family(sensor_type: &str) -> &'static str {
    if sensor_type.contains("hall") {
        "hall"
    } else if sensor_type.contains("flux") {
        "flux"
    } else if sensor_type.contains("ampere") {
        "ampere_loop"
    } else {
        "other"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CouplingCalibrationStageKind {
    PriorOnly,
    PdeResidualOnly,
    FieldProbes,
    FluxPanels,
    AmpereLoops,
}

#[derive(Debug, Clone)]
struct CouplingObservationInstance<'a> {
    drive_index: usize,
    drive_current: f64,
    observation: &'a ObservationSpec,
    truth: f64,
    observed: f64,
    variance: f64,
}

fn validate_coupling_calibration_config(
    config: &ToroidalHarmonicCouplingCalibrationConfig,
) -> Result<(), Box<dyn Error>> {
    if config.drive_currents.is_empty()
        || config
            .drive_currents
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(invalid_input(
            "coupling calibration drive currents must be finite, positive, and nonempty",
        )
        .into());
    }
    if !config.coupling_prior_std_scale.is_finite() || config.coupling_prior_std_scale <= 0.0 {
        return Err(invalid_input("coupling_prior_std_scale must be finite and positive").into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_coupling_calibration_stage(
    stage_name: &str,
    config: &ToroidalHarmonicCouplingCalibrationConfig,
    toroidal_config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    kind: CouplingCalibrationStageKind,
    training: Vec<&ObservationSpec>,
    heldout: Vec<&ObservationSpec>,
    coupling_prior_std: f64,
) -> Result<ToroidalCouplingCalibrationStageResult, Box<dyn Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(toroidal_config.sparse_prediction_noise_seed);
    let training_instances = coupling_observation_instances(
        workspace,
        toroidal_config,
        &config.drive_currents,
        &training,
        true,
        &mut rng,
    );
    let heldout_instances = coupling_observation_instances(
        workspace,
        toroidal_config,
        &config.drive_currents,
        &heldout,
        false,
        &mut rng,
    );
    let problem = build_coupling_calibration_problem(
        workspace,
        toroidal_config,
        &config.drive_currents,
        kind,
        &training_instances,
        &heldout_instances,
        coupling_prior_std,
    )?;
    let covariance_names = coupling_calibration_covariance_names(&config.drive_currents);
    let pushforward = solve_linear_pde_uq_with_pushforward_mean_covariance(
        &problem,
        &toroidal_config.solver,
        &covariance_names,
    )
    .map_err(|err| {
        invalid_input(format!(
            "coupling calibration pushforward solve failed: {err}"
        ))
    })?;
    let c_h_cov_index = pushforward
        .names
        .iter()
        .position(|name| name == &coupling_scalar_qoi_name("c_H", None))
        .ok_or_else(|| invalid_input("missing c_H pushforward covariance row"))?;
    let c_h_mean = pushforward.posterior_mean[c_h_cov_index];
    let c_h_prior_var = pushforward.prior_covariance[(c_h_cov_index, c_h_cov_index)];
    let c_h_var = pushforward.posterior_covariance[(c_h_cov_index, c_h_cov_index)];
    let c_h_truth = workspace.topology_summary.source_harmonic_kappa;

    let heldout_predictions = coupling_observation_rows(
        stage_name,
        &pushforward,
        workspace,
        &heldout_instances,
        c_h_mean,
    )?;
    let observations = coupling_observation_rows(
        stage_name,
        &pushforward,
        workspace,
        &training_instances,
        c_h_mean,
    )?;
    let qois = coupling_qoi_rows(stage_name, workspace, &config.drive_currents, &pushforward)?;
    let covariances = coupling_covariance_rows(stage_name, &pushforward);
    let heldout_rmse = prediction_rmse(
        heldout_predictions
            .iter()
            .map(|row| row.residual)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let heldout_nlpd = coupling_prediction_nlpd(&heldout_predictions);
    let heldout_covered95 = heldout_predictions
        .iter()
        .filter(|row| row.covered95)
        .count();
    let heldout_has_finite_sd = heldout_predictions
        .iter()
        .all(|row| row.standardized_residual.is_finite());
    let heldout_max_abs_standardized_residual = if heldout_has_finite_sd {
        heldout_predictions
            .iter()
            .map(|row| row.standardized_residual.abs())
            .fold(0.0_f64, f64::max)
    } else {
        f64::NAN
    };
    let count_kind = |sensor_type: &str| {
        training_instances
            .iter()
            .filter(|instance| instance.observation.sensor_type == sensor_type)
            .count()
    };
    let summary = ToroidalCouplingCalibrationSummaryRow {
        stage: stage_name.to_string(),
        training_rows: training_instances.len(),
        hall_training_rows: count_kind("hall"),
        flux_training_rows: count_kind("flux"),
        ampere_training_rows: count_kind(AMPERE_LOOP_SENSOR_TYPE),
        heldout_rows: heldout_predictions.len(),
        heldout_rmse,
        heldout_nlpd,
        heldout_covered95,
        heldout_coverage_fraction: if heldout_predictions.is_empty() || !heldout_has_finite_sd {
            f64::NAN
        } else {
            heldout_covered95 as f64 / heldout_predictions.len() as f64
        },
        heldout_max_abs_standardized_residual,
        coupling_prior_mean: 0.0,
        coupling_prior_variance: c_h_prior_var,
        coupling_posterior_mean: c_h_mean,
        coupling_posterior_variance: c_h_var,
        coupling_truth: c_h_truth,
        coupling_abs_error: (c_h_mean - c_h_truth).abs(),
        coupling_variance_ratio: c_h_var / c_h_prior_var,
        posterior_precision_nnz: pushforward.debug.posterior_factorization.matrix_nnz,
        posterior_factor_nnz: pushforward.debug.posterior_factorization.factor_nnz,
        posterior_fill_in: pushforward
            .debug
            .posterior_factorization
            .fill_in_ratio_vs_lower_triangle,
        posterior_factor_mib: pushforward
            .debug
            .posterior_factorization
            .factor_numeric_values_mib,
    };
    Ok(ToroidalCouplingCalibrationStageResult {
        summary,
        pushforward,
        qois,
        heldout_predictions,
        observations,
        covariances,
    })
}

fn coupling_observation_instances<'a>(
    workspace: &ExperimentWorkspace,
    config: &ToroidalHarmonicBConfig,
    drive_currents: &[f64],
    observations: &[&'a ObservationSpec],
    sample_noise: bool,
    rng: &mut rand::rngs::StdRng,
) -> Vec<CouplingObservationInstance<'a>> {
    let c_h_truth = workspace.topology_summary.source_harmonic_kappa;
    let mut instances = Vec::new();
    for (drive_index, &drive_current) in drive_currents.iter().enumerate() {
        for &observation in observations {
            let truth = drive_current
                * (observation_nominal_state_value(workspace, observation)
                    + c_h_truth * observation.beta_operator_value);
            let variance = observation_variance_from_value(truth, config);
            let observed = if sample_noise {
                truth + rng.sample::<f64, _>(StandardNormal) * variance.sqrt()
            } else {
                truth
            };
            instances.push(CouplingObservationInstance {
                drive_index,
                drive_current,
                observation,
                truth,
                observed,
                variance,
            });
        }
    }
    instances
}

fn observation_nominal_state_value(
    workspace: &ExperimentWorkspace,
    observation: &ObservationSpec,
) -> f64 {
    observation.observation_source_generated_truth
        - workspace.topology_summary.source_harmonic_kappa * observation.beta_operator_value
}

#[allow(clippy::too_many_arguments)]
fn build_coupling_calibration_problem(
    workspace: &ExperimentWorkspace,
    config: &ToroidalHarmonicBConfig,
    drive_currents: &[f64],
    kind: CouplingCalibrationStageKind,
    training: &[CouplingObservationInstance<'_>],
    _heldout: &[CouplingObservationInstance<'_>],
    coupling_prior_std: f64,
) -> Result<LinearPdeUqProblem, Box<dyn Error>> {
    let drive_count = drive_currents.len();
    let state_dim = workspace.system.state_dimension();
    let residual_dim = workspace.system.residual_dimension();
    let block_state_dim = state_dim * drive_count;
    let block_residual_dim = residual_dim * drive_count;
    let state_prior = GaussianPriorSpec {
        mean: vec![0.0; block_state_dim],
        precision: block_diag_repeat_triplet(&workspace.state_prior.precision, drive_count),
    };
    let system = ReducedLinearPdeAssembly {
        operator: core_triplet_to_feec_csr(&block_diag_repeat_triplet(
            &csr_to_triplet(&workspace.system.operator),
            drive_count,
        )),
        residual_bias: FeecVector::from_vec(coupling_residual_bias(workspace, drive_currents)),
        state_mass: core_triplet_to_feec_csr(&block_diag_repeat_triplet(
            &csr_to_triplet(&workspace.system.state_mass),
            drive_count,
        )),
        state_mass_inverse: workspace.system.state_mass_inverse.as_ref().map(|matrix| {
            core_triplet_to_feec_csr(&block_diag_repeat_triplet(
                &csr_to_triplet(matrix),
                drive_count,
            ))
        }),
        layout: DofLayout::new(block_state_dim, (0..block_state_dim).collect(), Vec::new()),
        forcing_operator: core_triplet_to_feec_csr(&SparseTripletMatrix::new(
            block_residual_dim,
            block_residual_dim,
        )),
        neumann_operator: core_triplet_to_feec_csr(&SparseTripletMatrix::new(
            block_residual_dim,
            block_residual_dim,
        )),
    };
    let uncertain_inputs = vec![LinearUncertainInputSpec {
        name: COUPLING_INPUT_NAME.to_string(),
        operator: SparseTripletMatrix::new(block_residual_dim, 1),
        prior: GaussianPriorSpec {
            mean: vec![0.0],
            precision: diagonal_precision(1, 1.0 / (coupling_prior_std * coupling_prior_std)),
        },
        preference: RepresentationPreference::ForceLatent,
        collapsed_precision: None,
    }];
    let joint_measurements = if matches!(
        kind,
        CouplingCalibrationStageKind::FieldProbes
            | CouplingCalibrationStageKind::FluxPanels
            | CouplingCalibrationStageKind::AmpereLoops
    ) {
        training
            .iter()
            .map(|instance| {
                coupling_measurement_spec(drive_count, &workspace.system.layout, instance)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(invalid_input)?
    } else {
        Vec::new()
    };
    let joint_derived_quantities =
        coupling_qoi_derived_specs(workspace, drive_currents, drive_count)?;
    let (pde_variance, pde_precision) = if matches!(kind, CouplingCalibrationStageKind::PriorOnly) {
        (None, None)
    } else if config.use_mass_weighted_pde_residual {
        let base = mass_weighted_pde_precision(&workspace.system, config).map_err(invalid_input)?;
        (None, Some(block_diag_repeat_triplet(&base, drive_count)))
    } else {
        (Some(config.pde_variance), None)
    };
    Ok(LinearPdeUqProblem {
        state_prior,
        system,
        uncertain_inputs,
        physical_measurements: Vec::new(),
        joint_measurements,
        derived_quantities: Vec::new(),
        joint_derived_quantities,
        pde_variance,
        pde_precision,
    })
}

fn coupling_residual_bias(workspace: &ExperimentWorkspace, drive_currents: &[f64]) -> Vec<f64> {
    let residual_dim = workspace.system.residual_dimension();
    let mut bias = vec![0.0; residual_dim * drive_currents.len()];
    for (drive_index, &drive_current) in drive_currents.iter().enumerate() {
        for row in 0..residual_dim {
            bias[drive_index * residual_dim + row] = -drive_current * workspace.source_rhs[row];
        }
    }
    bias
}

fn block_diag_repeat_triplet(matrix: &SparseTripletMatrix, count: usize) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        matrix.nrows() * count,
        matrix.ncols() * count,
        (0..count).flat_map(|block| {
            let row_offset = block * matrix.nrows();
            let col_offset = block * matrix.ncols();
            matrix
                .triplet_iter()
                .map(move |(row, col, value)| SparseTriplet {
                    row: row_offset + row,
                    col: col_offset + col,
                    value,
                })
        }),
    )
}

fn block_state_triplet(
    operator: &SparseTripletMatrix,
    drive_index: usize,
    layout: &DofLayout,
    drive_count: usize,
) -> Result<SparseTripletMatrix, String> {
    let reduced = restrict_state_operator_to_reduced_layout(operator, layout)?;
    let base_state_dim = layout.reduced_dimension();
    Ok(SparseTripletMatrix::from_triplets(
        reduced.nrows(),
        base_state_dim * drive_count,
        reduced
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col: drive_index * base_state_dim + col,
                value,
            }),
    ))
}

fn block_state_row_operator(
    operator: &SparseTripletMatrix,
    drive_index: usize,
    layout: &DofLayout,
    drive_count: usize,
) -> Result<SparseRowOperator, String> {
    triplet_to_row_operator(&block_state_triplet(
        operator,
        drive_index,
        layout,
        drive_count,
    )?)
}

fn restrict_state_operator_to_reduced_layout(
    operator: &SparseTripletMatrix,
    layout: &DofLayout,
) -> Result<SparseTripletMatrix, String> {
    if operator.ncols() == layout.reduced_dimension() {
        return Ok(operator.clone());
    }
    if operator.ncols() != layout.full_dimension {
        return Err(format!(
            "state operator has {} columns but layout has full dimension {} and reduced dimension {}",
            operator.ncols(),
            layout.full_dimension,
            layout.reduced_dimension()
        ));
    }
    let active_map = layout
        .active_dofs
        .iter()
        .enumerate()
        .map(|(reduced, &full)| (full, reduced))
        .collect::<BTreeMap<_, _>>();
    Ok(SparseTripletMatrix::from_triplets(
        operator.nrows(),
        layout.reduced_dimension(),
        operator.triplet_iter().filter_map(|(row, col, value)| {
            active_map
                .get(&col)
                .copied()
                .map(|reduced_col| SparseTriplet {
                    row,
                    col: reduced_col,
                    value,
                })
        }),
    ))
}

fn coupling_measurement_spec(
    drive_count: usize,
    layout: &DofLayout,
    instance: &CouplingObservationInstance<'_>,
) -> Result<LinearPdeJointMeasurementSpec, String> {
    Ok(LinearPdeJointMeasurementSpec {
        name: coupling_observation_name(instance.drive_index, &instance.observation.name),
        state_operator: Some(block_state_triplet(
            &instance.observation.state_operator,
            instance.drive_index,
            layout,
            drive_count,
        )?),
        latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
            input_name: COUPLING_INPUT_NAME.to_string(),
            operator: scalar_column_triplet(
                instance.drive_current * instance.observation.beta_operator_value,
            ),
        }],
        observations: vec![instance.observed],
        bias: vec![0.0],
        variance: instance.variance,
    })
}

fn coupling_qoi_derived_specs(
    workspace: &ExperimentWorkspace,
    drive_currents: &[f64],
    drive_count: usize,
) -> Result<Vec<LinearPdeJointDerivedQuantitySpec>, String> {
    let harmonic_projection_operator_full = build_harmonic_projection_state_operator(
        &workspace.topology,
        &workspace.h2,
        &workspace.mass_2,
    )?;
    let harmonic_projection_operator = restrict_state_operator_to_reduced_layout(
        &harmonic_projection_operator_full,
        &workspace.system.layout,
    )?;
    let link_flux = topology_pushforward_observation(workspace, "embedded_flux_panel_p0")?;
    let local_flux = topology_pushforward_observation(workspace, "flux_top")?;
    let ampere_loop = topology_pushforward_observation(workspace, AMPERE_LOOP_NAME)?;
    let field_x = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_bx")?;
    let field_y = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_by")?;
    let field_z = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_bz")?;
    let mut specs = vec![LinearPdeJointDerivedQuantitySpec {
        name: coupling_scalar_qoi_name("c_H", None),
        state_operator: None,
        latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
            input_name: COUPLING_INPUT_NAME.to_string(),
            operator: scalar_row_operator(1.0)?,
        }],
    }];
    for (drive_index, &drive_current) in drive_currents.iter().enumerate() {
        specs.push(LinearPdeJointDerivedQuantitySpec {
            name: coupling_scalar_qoi_name("beta_H", Some(drive_index)),
            state_operator: None,
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: COUPLING_INPUT_NAME.to_string(),
                operator: scalar_row_operator(drive_current)?,
            }],
        });
        specs.push(LinearPdeJointDerivedQuantitySpec {
            name: coupling_scalar_qoi_name("eta_H", Some(drive_index)),
            state_operator: Some(block_state_row_operator(
                &harmonic_projection_operator,
                drive_index,
                &workspace.system.layout,
                drive_count,
            )?),
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: COUPLING_INPUT_NAME.to_string(),
                operator: scalar_row_operator(drive_current)?,
            }],
        });
        for (label, observation) in [
            ("Phi_link", link_flux),
            ("Phi_loc", local_flux),
            ("I_gamma", ampere_loop),
            ("B_x_xi1", field_x),
            ("B_y_xi1", field_y),
            ("B_z_xi1", field_z),
        ] {
            specs.push(LinearPdeJointDerivedQuantitySpec {
                name: coupling_scalar_qoi_name(label, Some(drive_index)),
                state_operator: Some(block_state_row_operator(
                    &observation.state_operator,
                    drive_index,
                    &workspace.system.layout,
                    drive_count,
                )?),
                latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                    input_name: COUPLING_INPUT_NAME.to_string(),
                    operator: scalar_row_operator(drive_current * observation.beta_operator_value)?,
                }],
            });
        }
    }
    Ok(specs)
}

fn coupling_scalar_qoi_name(label: &str, drive_index: Option<usize>) -> String {
    match drive_index {
        Some(index) => format!("coupling::{label}::d{index}"),
        None => format!("coupling::{label}"),
    }
}

fn coupling_observation_name(drive_index: usize, observation_name: &str) -> String {
    format!("coupling::obs::d{drive_index}::{observation_name}")
}

fn coupling_calibration_covariance_names(drive_currents: &[f64]) -> Vec<String> {
    let mut names = vec![coupling_scalar_qoi_name("c_H", None)];
    for index in 0..drive_currents.len() {
        names.push(coupling_scalar_qoi_name("beta_H", Some(index)));
        names.push(coupling_scalar_qoi_name("eta_H", Some(index)));
        names.push(coupling_scalar_qoi_name("Phi_link", Some(index)));
        names.push(coupling_scalar_qoi_name("Phi_loc", Some(index)));
        names.push(coupling_scalar_qoi_name("I_gamma", Some(index)));
        names.push(coupling_scalar_qoi_name("B_x_xi1", Some(index)));
        names.push(coupling_scalar_qoi_name("B_y_xi1", Some(index)));
        names.push(coupling_scalar_qoi_name("B_z_xi1", Some(index)));
    }
    names
}

fn coupling_observation_rows(
    stage_name: &str,
    pushforward: &LinearPdePushforwardMeanCovarianceResult,
    workspace: &ExperimentWorkspace,
    instances: &[CouplingObservationInstance<'_>],
    c_h_mean: f64,
) -> Result<Vec<ToroidalCouplingCalibrationObservationRow>, Box<dyn Error>> {
    instances
        .iter()
        .map(|instance| {
            let prediction = coupling_observation_mean(workspace, pushforward, instance, c_h_mean)?;
            Ok(coupling_observation_row(stage_name, instance, prediction))
        })
        .collect()
}

fn coupling_observation_mean(
    workspace: &ExperimentWorkspace,
    pushforward: &LinearPdePushforwardMeanCovarianceResult,
    instance: &CouplingObservationInstance<'_>,
    c_h_mean: f64,
) -> Result<f64, Box<dyn Error>> {
    let state_mean = coupling_apply_block_state_row(
        &instance.observation.state_operator,
        &pushforward.joint_posterior_mean,
        instance.drive_index,
        &workspace.system.layout,
    )?;
    Ok(state_mean + instance.drive_current * c_h_mean * instance.observation.beta_operator_value)
}

fn coupling_apply_block_state_row(
    operator: &SparseTripletMatrix,
    joint_posterior_mean: &[f64],
    drive_index: usize,
    layout: &DofLayout,
) -> Result<f64, Box<dyn Error>> {
    let reduced =
        restrict_state_operator_to_reduced_layout(operator, layout).map_err(invalid_input)?;
    let base_state_dim = layout.reduced_dimension();
    let offset = drive_index * base_state_dim;
    let mut value = 0.0;
    for (row, col, entry) in reduced.triplet_iter() {
        if row != 0 {
            return Err(invalid_input("coupling scalar row expected exactly one row").into());
        }
        value += entry * joint_posterior_mean[offset + col];
    }
    Ok(value)
}

fn coupling_observation_row(
    stage_name: &str,
    instance: &CouplingObservationInstance<'_>,
    prediction: f64,
) -> ToroidalCouplingCalibrationObservationRow {
    let residual = prediction - instance.truth;
    ToroidalCouplingCalibrationObservationRow {
        stage: stage_name.to_string(),
        drive_index: instance.drive_index,
        drive_current: instance.drive_current,
        sensor_type: instance.observation.sensor_type.clone(),
        name: instance.observation.name.clone(),
        truth: instance.truth,
        observed: instance.observed,
        prediction,
        residual,
        posterior_sd: f64::NAN,
        standardized_residual: f64::NAN,
        lower95: f64::NAN,
        upper95: f64::NAN,
        covered95: false,
    }
}

fn coupling_qoi_rows(
    stage_name: &str,
    workspace: &ExperimentWorkspace,
    drive_currents: &[f64],
    pushforward: &LinearPdePushforwardMeanCovarianceResult,
) -> Result<Vec<ToroidalCouplingCalibrationQoiRow>, Box<dyn Error>> {
    let mut rows = vec![coupling_qoi_from_pushforward(
        stage_name,
        pushforward,
        None,
        None,
        "c_H",
        "coupling",
        workspace.topology_summary.source_harmonic_kappa,
        "weighted flux per source",
    )?];
    let link_flux = topology_pushforward_observation(workspace, "embedded_flux_panel_p0")
        .map_err(invalid_input)?;
    let local_flux =
        topology_pushforward_observation(workspace, "flux_top").map_err(invalid_input)?;
    let ampere_loop =
        topology_pushforward_observation(workspace, AMPERE_LOOP_NAME).map_err(invalid_input)?;
    let field_x = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_bx")
        .map_err(invalid_input)?;
    let field_y = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_by")
        .map_err(invalid_input)?;
    let field_z = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_bz")
        .map_err(invalid_input)?;
    for (drive_index, &drive_current) in drive_currents.iter().enumerate() {
        rows.push(coupling_qoi_from_pushforward(
            stage_name,
            pushforward,
            Some(drive_index),
            Some(drive_current),
            "beta_H",
            "topology",
            drive_current * workspace.topology_summary.source_harmonic_kappa,
            "weighted flux coordinate",
        )?);
        rows.push(coupling_qoi_from_pushforward(
            stage_name,
            pushforward,
            Some(drive_index),
            Some(drive_current),
            "eta_H",
            "topology",
            drive_current * workspace.topology_summary.source_harmonic_projection_unit,
            "weighted flux coordinate",
        )?);
        for (label, role, unit, observation) in [
            ("Phi_link", "flux", "flux proxy", link_flux),
            ("Phi_loc", "flux", "flux proxy", local_flux),
            ("I_gamma", "circulation", "A", ampere_loop),
            ("B_x_xi1", "field", "T proxy", field_x),
            ("B_y_xi1", "field", "T proxy", field_y),
            ("B_z_xi1", "field", "T proxy", field_z),
        ] {
            let truth = drive_current
                * (observation_nominal_state_value(workspace, observation)
                    + workspace.topology_summary.source_harmonic_kappa
                        * observation.beta_operator_value);
            rows.push(coupling_qoi_from_pushforward(
                stage_name,
                pushforward,
                Some(drive_index),
                Some(drive_current),
                label,
                role,
                truth,
                unit,
            )?);
        }
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn coupling_qoi_from_pushforward(
    stage_name: &str,
    pushforward: &LinearPdePushforwardMeanCovarianceResult,
    drive_index: Option<usize>,
    drive_current: Option<f64>,
    qoi: &str,
    role: &str,
    truth: f64,
    unit: &str,
) -> Result<ToroidalCouplingCalibrationQoiRow, Box<dyn Error>> {
    let name = coupling_scalar_qoi_name(qoi, drive_index);
    let index = coupling_pushforward_index(pushforward, &name)?;
    Ok(coupling_qoi_row(
        stage_name,
        drive_index,
        drive_current,
        qoi,
        role,
        truth,
        pushforward.posterior_mean[index],
        pushforward.prior_covariance[(index, index)],
        pushforward.posterior_covariance[(index, index)],
        unit,
    ))
}

fn coupling_pushforward_index(
    pushforward: &LinearPdePushforwardMeanCovarianceResult,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    pushforward
        .names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| invalid_input(format!("missing coupling pushforward `{name}`")).into())
}

#[allow(clippy::too_many_arguments)]
fn coupling_qoi_row(
    stage_name: &str,
    drive_index: Option<usize>,
    drive_current: Option<f64>,
    qoi: &str,
    role: &str,
    truth: f64,
    mean: f64,
    prior_variance: f64,
    posterior_variance: f64,
    unit: &str,
) -> ToroidalCouplingCalibrationQoiRow {
    let sd = posterior_variance.max(0.0).sqrt();
    ToroidalCouplingCalibrationQoiRow {
        stage: stage_name.to_string(),
        drive_index,
        drive_current,
        qoi: qoi.to_string(),
        role: role.to_string(),
        truth,
        mean,
        sd,
        lower95: mean - 1.96 * sd,
        upper95: mean + 1.96 * sd,
        prior_variance,
        posterior_variance,
        variance_ratio: if prior_variance.abs() > EPS {
            posterior_variance / prior_variance
        } else {
            f64::NAN
        },
        unit: unit.to_string(),
    }
}

fn coupling_covariance_rows(
    stage_name: &str,
    covariance: &LinearPdePushforwardMeanCovarianceResult,
) -> Vec<ToroidalCouplingCalibrationCovarianceRow> {
    let mut rows = Vec::new();
    for i in 0..covariance.names.len() {
        for j in 0..covariance.names.len() {
            let prior = covariance.prior_covariance[(i, j)];
            let posterior = covariance.posterior_covariance[(i, j)];
            let denom = (covariance.posterior_covariance[(i, i)]
                * covariance.posterior_covariance[(j, j)])
                .max(0.0)
                .sqrt();
            rows.push(ToroidalCouplingCalibrationCovarianceRow {
                stage: stage_name.to_string(),
                qoi_i: covariance.names[i].clone(),
                qoi_j: covariance.names[j].clone(),
                prior_covariance: prior,
                posterior_covariance: posterior,
                posterior_correlation: if denom > EPS {
                    posterior / denom
                } else {
                    f64::NAN
                },
            });
        }
    }
    rows
}

fn coupling_prediction_nlpd(rows: &[ToroidalCouplingCalibrationObservationRow]) -> f64 {
    if rows.is_empty()
        || rows
            .iter()
            .any(|row| !row.posterior_sd.is_finite() || row.posterior_sd <= 0.0)
    {
        return f64::NAN;
    }
    rows.iter()
        .map(|row| {
            let variance = row.posterior_sd * row.posterior_sd;
            0.5 * ((2.0 * PI * variance).ln() + row.residual * row.residual / variance)
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn report_coupling_calibration_progress(stage: &ToroidalCouplingCalibrationStageResult) {
    println!(
        "{}: c_H={:.6e}±{:.3e} truth={:.6e} var_ratio={:.3e} train={} heldout={} rmse={:.3e} coverage={}/{} max|z|={:.3} factor_nnz={} fill={:.3}x",
        stage.summary.stage,
        stage.summary.coupling_posterior_mean,
        stage.summary.coupling_posterior_variance.max(0.0).sqrt(),
        stage.summary.coupling_truth,
        stage.summary.coupling_variance_ratio,
        stage.summary.training_rows,
        stage.summary.heldout_rows,
        stage.summary.heldout_rmse,
        stage.summary.heldout_covered95,
        stage.summary.heldout_rows,
        stage.summary.heldout_max_abs_standardized_residual,
        stage.summary.posterior_factor_nnz,
        stage.summary.posterior_fill_in,
    );
}

fn run_toroidal_gp_baseline_stage(
    config: &ToroidalGpBaselineConfig,
    workspace: &ExperimentWorkspace,
    stage_name: &str,
    matched_feec_stage: &str,
    training: Vec<&ObservationSpec>,
) -> Result<ToroidalGpBaselineStageResult, Box<dyn Error>> {
    run_toroidal_gp_baseline_stage_with_predictions(
        config,
        workspace,
        stage_name,
        matched_feec_stage,
        training,
        workspace.heldout_observations.iter().collect(),
    )
}

fn run_toroidal_gp_baseline_stage_with_predictions(
    config: &ToroidalGpBaselineConfig,
    workspace: &ExperimentWorkspace,
    stage_name: &str,
    matched_feec_stage: &str,
    training: Vec<&ObservationSpec>,
    heldout: Vec<&ObservationSpec>,
) -> Result<ToroidalGpBaselineStageResult, Box<dyn Error>> {
    let training_targets = training
        .into_iter()
        .map(gp_target_from_observation)
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_input)?;
    let heldout_targets = heldout
        .into_iter()
        .map(gp_target_from_observation)
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_input)?;
    let qoi_targets = gp_baseline_qoi_targets(workspace).map_err(invalid_input)?;
    let prediction_targets = heldout_targets
        .iter()
        .cloned()
        .chain(qoi_targets.iter().cloned())
        .collect::<Vec<_>>();

    let hyper = fit_gp_hyperparameters(config, &training_targets).map_err(invalid_input)?;
    let posterior = condition_gp_functionals(
        &hyper,
        &training_targets,
        &prediction_targets,
        config.jitter,
    )
    .map_err(invalid_input)?;
    let heldout_count = heldout_targets.len();
    let heldout_predictions = heldout_targets
        .iter()
        .enumerate()
        .map(|(index, target)| {
            gp_prediction_row(
                stage_name,
                &target.sensor_type,
                &target.name,
                target.truth,
                posterior.mean[index],
                posterior.variance[index],
            )
        })
        .collect::<Vec<_>>();
    let qois = qoi_targets
        .iter()
        .enumerate()
        .map(|(index, target)| {
            let posterior_index = heldout_count + index;
            let variance = posterior.variance[posterior_index].max(0.0);
            let sd = variance.sqrt();
            let mean = posterior.mean[posterior_index];
            ToroidalGpBaselineQoiRow {
                stage: stage_name.to_string(),
                qoi: target.name.clone(),
                role: topology_pushforward_qoi_role(&target.name).to_string(),
                truth: target.truth,
                mean,
                sd,
                lower95: mean - 1.96 * sd,
                upper95: mean + 1.96 * sd,
                abs_error: (mean - target.truth).abs(),
                unit: topology_pushforward_qoi_unit(&target.name).to_string(),
            }
        })
        .collect::<Vec<_>>();

    let heldout_rmse = prediction_rmse(
        heldout_predictions
            .iter()
            .map(|row| row.residual)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let heldout_nlpd = prediction_nlpd(&heldout_predictions);
    let heldout_covered95 = heldout_predictions
        .iter()
        .filter(|row| row.covered95)
        .count();
    let heldout_max_abs_standardized_residual = heldout_predictions
        .iter()
        .map(|row| row.standardized_residual.abs())
        .fold(0.0_f64, f64::max);
    let count_kind = |kind: &str| {
        training_targets
            .iter()
            .filter(|target| target.sensor_type == kind)
            .count()
    };
    let summary = ToroidalGpBaselineSummaryRow {
        stage: stage_name.to_string(),
        matched_feec_stage: matched_feec_stage.to_string(),
        kernel: "independent_output_matern32".to_string(),
        matern_nu: config.matern_nu,
        length_scale: hyper.length_scale,
        signal_variance: hyper.signal_variance,
        log_marginal_likelihood: hyper.log_marginal_likelihood,
        training_rows: training_targets.len(),
        hall_training_rows: count_kind("hall"),
        flux_training_rows: count_kind("flux"),
        ampere_training_rows: count_kind(AMPERE_LOOP_SENSOR_TYPE),
        heldout_rows: heldout_predictions.len(),
        heldout_rmse,
        heldout_nlpd,
        heldout_covered95,
        heldout_coverage_fraction: if heldout_predictions.is_empty() {
            f64::NAN
        } else {
            heldout_covered95 as f64 / heldout_predictions.len() as f64
        },
        heldout_max_abs_standardized_residual,
    };
    Ok(ToroidalGpBaselineStageResult {
        summary,
        heldout_predictions,
        qois,
    })
}

fn report_gp_baseline_progress(stage: &ToroidalGpBaselineStageResult) {
    println!(
        "{}: ell={:.3e} signal_var={:.3e} logml={:.3e} heldout_rmse={:.3e} heldout_nlpd={:.3e} coverage={}/{} max|z|={:.3}",
        stage.summary.stage,
        stage.summary.length_scale,
        stage.summary.signal_variance,
        stage.summary.log_marginal_likelihood,
        stage.summary.heldout_rmse,
        stage.summary.heldout_nlpd,
        stage.summary.heldout_covered95,
        stage.summary.heldout_rows,
        stage.summary.heldout_max_abs_standardized_residual
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceTemplateKind {
    ExactSource,
    TopologyOracle,
}

impl SourceTemplateKind {
    fn label(self) -> &'static str {
        match self {
            SourceTemplateKind::ExactSource => "source_template_exact",
            SourceTemplateKind::TopologyOracle => "source_template_topology_oracle",
        }
    }

    fn model_label(self) -> &'static str {
        match self {
            SourceTemplateKind::ExactSource => "source-template GP exact",
            SourceTemplateKind::TopologyOracle => "source-template GP topology oracle",
        }
    }

    fn stage_name(self, sensor_family: &str) -> &'static str {
        match (self, sensor_family) {
            (SourceTemplateKind::ExactSource, "field") => "T2_exact_field",
            (SourceTemplateKind::ExactSource, "field_flux") => "T3_exact_field_flux",
            (SourceTemplateKind::ExactSource, "field_flux_loop") => "T4_exact_field_flux_loop",
            (SourceTemplateKind::TopologyOracle, "field") => "U2_oracle_field",
            (SourceTemplateKind::TopologyOracle, "field_flux") => "U3_oracle_field_flux",
            (SourceTemplateKind::TopologyOracle, "field_flux_loop") => "U4_oracle_field_flux_loop",
            _ => "source_template_gp",
        }
    }

    fn includes_harmonic_template(self) -> bool {
        matches!(self, SourceTemplateKind::TopologyOracle)
    }
}

#[derive(Debug, Clone)]
struct SourceTemplateGpTarget {
    target: GpPredictionTarget,
    exact_template_value: f64,
    oracle_template_value: f64,
}

impl SourceTemplateGpTarget {
    fn template_value(&self, kind: SourceTemplateKind) -> f64 {
        match kind {
            SourceTemplateKind::ExactSource => self.exact_template_value,
            SourceTemplateKind::TopologyOracle => self.oracle_template_value,
        }
    }
}

#[derive(Debug, Clone)]
struct SourceTemplateGpPosteriorPrediction {
    mean: Vec<f64>,
    variance: Vec<f64>,
    source_mean: f64,
    source_variance: f64,
}

fn run_source_template_gp_stage(
    config: &ToroidalGpBaselineConfig,
    workspace: &ExperimentWorkspace,
    template_kind: SourceTemplateKind,
    stage_name: &str,
    sensor_family_label: &str,
    training: Vec<&ObservationSpec>,
    heldout: Vec<&ObservationSpec>,
) -> Result<ToroidalSourceTemplateGpStageResult, Box<dyn Error>> {
    let training_targets = training
        .into_iter()
        .map(|observation| source_template_target_from_observation(workspace, observation))
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_input)?;
    let heldout_targets = heldout
        .into_iter()
        .map(|observation| source_template_target_from_observation(workspace, observation))
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid_input)?;
    let qoi_targets = source_template_gp_qoi_targets(workspace).map_err(invalid_input)?;
    let prediction_targets = heldout_targets
        .iter()
        .cloned()
        .chain(qoi_targets.iter().cloned())
        .collect::<Vec<_>>();

    let source_prior_mean = 1.0;
    let source_prior_variance = config.toroidal.source_prior_std.powi(2);
    let source_truth = config.toroidal.source_alpha_true;
    let hyper = fit_source_template_gp_hyperparameters(
        config,
        &training_targets,
        template_kind,
        source_prior_mean,
        source_prior_variance,
    )
    .map_err(invalid_input)?;
    let posterior = condition_source_template_gp_functionals(
        &hyper,
        &training_targets,
        &prediction_targets,
        template_kind,
        source_prior_mean,
        source_prior_variance,
        config.jitter,
    )
    .map_err(invalid_input)?;

    let heldout_count = heldout_targets.len();
    let heldout_predictions = heldout_targets
        .iter()
        .enumerate()
        .map(|(index, target)| {
            gp_prediction_row(
                stage_name,
                &target.target.sensor_type,
                &target.target.name,
                target.target.truth,
                posterior.mean[index],
                posterior.variance[index],
            )
        })
        .collect::<Vec<_>>();

    let mut qois = Vec::new();
    qois.push(source_template_scalar_qoi_row(
        stage_name,
        QOI_SOURCE_NAME,
        source_truth,
        posterior.source_mean,
        posterior.source_variance,
    ));
    if template_kind.includes_harmonic_template() {
        let c_h = workspace.topology_summary.source_harmonic_kappa;
        qois.push(source_template_scalar_qoi_row(
            stage_name,
            QOI_SOURCE_BETA_NAME,
            c_h * source_truth,
            c_h * posterior.source_mean,
            c_h * c_h * posterior.source_variance,
        ));
        let eta_unit = workspace.topology_summary.source_harmonic_projection_unit;
        qois.push(source_template_scalar_qoi_row(
            stage_name,
            QOI_HARMONIC_PROJECTION_NAME,
            workspace.topology_summary.source_harmonic_projection_true,
            eta_unit * posterior.source_mean,
            eta_unit * eta_unit * posterior.source_variance,
        ));
    }
    qois.extend(qoi_targets.iter().enumerate().map(|(index, target)| {
        let posterior_index = heldout_count + index;
        let variance = posterior.variance[posterior_index].max(0.0);
        let sd = variance.sqrt();
        let mean = posterior.mean[posterior_index];
        ToroidalGpBaselineQoiRow {
            stage: stage_name.to_string(),
            qoi: target.target.name.clone(),
            role: topology_pushforward_qoi_role(&target.target.name).to_string(),
            truth: target.target.truth,
            mean,
            sd,
            lower95: mean - 1.96 * sd,
            upper95: mean + 1.96 * sd,
            abs_error: (mean - target.target.truth).abs(),
            unit: topology_pushforward_qoi_unit(&target.target.name).to_string(),
        }
    }));

    let heldout_rmse = prediction_rmse(
        heldout_predictions
            .iter()
            .map(|row| row.residual)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let heldout_nlpd = prediction_nlpd(&heldout_predictions);
    let heldout_covered95 = heldout_predictions
        .iter()
        .filter(|row| row.covered95)
        .count();
    let heldout_max_abs_standardized_residual = heldout_predictions
        .iter()
        .map(|row| row.standardized_residual.abs())
        .fold(0.0_f64, f64::max);
    let count_kind = |kind: &str| {
        training_targets
            .iter()
            .filter(|target| target.target.sensor_type == kind)
            .count()
    };
    let summary = ToroidalSourceTemplateGpSummaryRow {
        model: template_kind.model_label().to_string(),
        stage: stage_name.to_string(),
        template_kind: template_kind.label().to_string(),
        sensor_family: sensor_family_label.to_string(),
        kernel: "source_template_plus_independent_output_matern32".to_string(),
        matern_nu: config.matern_nu,
        length_scale: hyper.length_scale,
        signal_variance: hyper.signal_variance,
        log_marginal_likelihood: hyper.log_marginal_likelihood,
        training_rows: training_targets.len(),
        hall_training_rows: count_kind("hall"),
        flux_training_rows: count_kind("flux"),
        ampere_training_rows: count_kind(AMPERE_LOOP_SENSOR_TYPE),
        heldout_rows: heldout_predictions.len(),
        heldout_rmse,
        heldout_nlpd,
        heldout_covered95,
        heldout_coverage_fraction: if heldout_predictions.is_empty() {
            f64::NAN
        } else {
            heldout_covered95 as f64 / heldout_predictions.len() as f64
        },
        heldout_max_abs_standardized_residual,
        source_prior_mean,
        source_prior_variance,
        source_posterior_mean: posterior.source_mean,
        source_posterior_variance: posterior.source_variance,
        source_truth,
        source_abs_error: (posterior.source_mean - source_truth).abs(),
    };
    Ok(ToroidalSourceTemplateGpStageResult {
        summary,
        heldout_predictions,
        qois,
    })
}

fn report_source_template_gp_progress(stage: &ToroidalSourceTemplateGpStageResult) {
    println!(
        "{} {}: s={:.6}±{:.3e} ell={:.3e} signal_var={:.3e} logml={:.3e} heldout_rmse={:.3e} coverage={}/{} max|z|={:.3}",
        stage.summary.model,
        stage.summary.stage,
        stage.summary.source_posterior_mean,
        stage.summary.source_posterior_variance.max(0.0).sqrt(),
        stage.summary.length_scale,
        stage.summary.signal_variance,
        stage.summary.log_marginal_likelihood,
        stage.summary.heldout_rmse,
        stage.summary.heldout_covered95,
        stage.summary.heldout_rows,
        stage.summary.heldout_max_abs_standardized_residual
    );
}

fn source_template_scalar_qoi_row(
    stage_name: &str,
    qoi: &str,
    truth: f64,
    mean: f64,
    variance: f64,
) -> ToroidalGpBaselineQoiRow {
    let sd = variance.max(0.0).sqrt();
    ToroidalGpBaselineQoiRow {
        stage: stage_name.to_string(),
        qoi: qoi.to_string(),
        role: topology_pushforward_qoi_role(qoi).to_string(),
        truth,
        mean,
        sd,
        lower95: mean - 1.96 * sd,
        upper95: mean + 1.96 * sd,
        abs_error: (mean - truth).abs(),
        unit: topology_pushforward_qoi_unit(qoi).to_string(),
    }
}

#[derive(Debug, Clone)]
struct GpPredictionTarget {
    sensor_type: String,
    name: String,
    truth: f64,
    observed: f64,
    noise_variance: f64,
    terms: Vec<GpFunctionalTerm>,
}

#[derive(Debug, Clone, Copy)]
struct GpHyperparameters {
    matern_nu: f64,
    length_scale: f64,
    signal_variance: f64,
    log_marginal_likelihood: f64,
}

#[derive(Debug, Clone)]
struct GpPosteriorPrediction {
    mean: Vec<f64>,
    variance: Vec<f64>,
}

fn gp_target_from_observation(observation: &ObservationSpec) -> Result<GpPredictionTarget, String> {
    if observation.gp_terms.is_empty() {
        return Err(format!(
            "observation `{}` has no GP functional terms",
            observation.name
        ));
    }
    Ok(GpPredictionTarget {
        sensor_type: observation.sensor_type.clone(),
        name: observation.name.clone(),
        truth: observation.observation_source_generated_truth,
        observed: observation.observation_source_generated_observed,
        noise_variance: observation.variance_source_generated_truth,
        terms: observation.gp_terms.clone(),
    })
}

fn gp_baseline_qoi_targets(
    workspace: &ExperimentWorkspace,
) -> Result<Vec<GpPredictionTarget>, String> {
    [
        QOI_LINK_FLUX_NAME,
        QOI_LOCAL_FLUX_NAME,
        QOI_AMPERE_LOOP_NAME,
        QOI_FIELD_X_NAME,
        QOI_FIELD_Y_NAME,
        QOI_FIELD_Z_NAME,
    ]
    .into_iter()
    .map(|name| {
        let observation_name = topology_pushforward_qoi_observation_name(name)
            .ok_or_else(|| format!("QoI `{name}` has no observation mapping"))?;
        let observation = topology_pushforward_observation(workspace, observation_name)?;
        let mut target = gp_target_from_observation(observation)?;
        target.name = name.to_string();
        target.sensor_type = topology_pushforward_qoi_role(name).to_string();
        Ok(target)
    })
    .collect()
}

fn source_template_target_from_observation(
    workspace: &ExperimentWorkspace,
    observation: &ObservationSpec,
) -> Result<SourceTemplateGpTarget, String> {
    let target = gp_target_from_observation(observation)?;
    let exact_template_value =
        apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)?;
    let oracle_template_value = exact_template_value
        + workspace.topology_summary.source_harmonic_kappa * observation.beta_operator_value;
    Ok(SourceTemplateGpTarget {
        target,
        exact_template_value,
        oracle_template_value,
    })
}

fn source_template_gp_qoi_targets(
    workspace: &ExperimentWorkspace,
) -> Result<Vec<SourceTemplateGpTarget>, String> {
    [
        QOI_LINK_FLUX_NAME,
        QOI_LOCAL_FLUX_NAME,
        QOI_AMPERE_LOOP_NAME,
        QOI_FIELD_X_NAME,
        QOI_FIELD_Y_NAME,
        QOI_FIELD_Z_NAME,
    ]
    .into_iter()
    .map(|name| {
        let observation_name = topology_pushforward_qoi_observation_name(name)
            .ok_or_else(|| format!("QoI `{name}` has no observation mapping"))?;
        let observation = topology_pushforward_observation(workspace, observation_name)?;
        let mut target = source_template_target_from_observation(workspace, observation)?;
        target.target.name = name.to_string();
        target.target.sensor_type = topology_pushforward_qoi_role(name).to_string();
        Ok(target)
    })
    .collect()
}

fn fit_gp_hyperparameters(
    config: &ToroidalGpBaselineConfig,
    training: &[GpPredictionTarget],
) -> Result<GpHyperparameters, String> {
    if training.is_empty() {
        return Err("GP baseline requires at least one training row".to_string());
    }
    let mut best = None;
    for &length_scale in &config.length_scales {
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(format!("invalid GP length scale {length_scale}"));
        }
        let unit_hyper = GpHyperparameters {
            matern_nu: config.matern_nu,
            length_scale,
            signal_variance: 1.0,
            log_marginal_likelihood: f64::NAN,
        };
        let unit_cov = gp_covariance_matrix(training, training, &unit_hyper)?;
        let base_signal_std = empirical_signal_std(training, &unit_cov);
        for &factor in &config.signal_std_factors {
            if !factor.is_finite() || factor <= 0.0 {
                return Err(format!("invalid GP signal std factor {factor}"));
            }
            let signal_std = (base_signal_std * factor).max(1e-16);
            let signal_variance = signal_std * signal_std;
            let hyper = GpHyperparameters {
                matern_nu: config.matern_nu,
                length_scale,
                signal_variance,
                log_marginal_likelihood: f64::NAN,
            };
            let logml = gp_log_marginal_likelihood(&hyper, training, config.jitter)?;
            let candidate = GpHyperparameters {
                log_marginal_likelihood: logml,
                ..hyper
            };
            if best
                .as_ref()
                .map(|current: &GpHyperparameters| {
                    candidate.log_marginal_likelihood > current.log_marginal_likelihood
                })
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| "GP hyperparameter grid was empty".to_string())
}

fn empirical_signal_std(training: &[GpPredictionTarget], unit_cov: &FeecMatrix) -> f64 {
    let mut acc = 0.0;
    let mut count = 0usize;
    for (index, target) in training.iter().enumerate() {
        let diag = unit_cov[(index, index)].abs().max(1e-300);
        acc += target.observed * target.observed / diag;
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    (acc / count as f64).max(1e-300).sqrt()
}

fn gp_log_marginal_likelihood(
    hyper: &GpHyperparameters,
    training: &[GpPredictionTarget],
    jitter: f64,
) -> Result<f64, String> {
    let mut cov = gp_covariance_matrix(training, training, hyper)?;
    add_gp_noise_and_jitter(&mut cov, training, jitter);
    let observations =
        FeecVector::from_iterator(training.len(), training.iter().map(|t| t.observed));
    let Some(chol) = cov.clone().cholesky() else {
        return Err("GP training covariance Cholesky failed".to_string());
    };
    let alpha = chol.solve(&observations);
    let quadratic = observations.dot(&alpha);
    let lower = chol.l();
    let logdet = 2.0
        * (0..training.len())
            .map(|index| lower[(index, index)].abs().ln())
            .sum::<f64>();
    Ok(-0.5 * quadratic - 0.5 * logdet - 0.5 * training.len() as f64 * (2.0 * PI).ln())
}

fn condition_gp_functionals(
    hyper: &GpHyperparameters,
    training: &[GpPredictionTarget],
    predictions: &[GpPredictionTarget],
    jitter: f64,
) -> Result<GpPosteriorPrediction, String> {
    if predictions.is_empty() {
        return Ok(GpPosteriorPrediction {
            mean: Vec::new(),
            variance: Vec::new(),
        });
    }
    let mut k_tt = gp_covariance_matrix(training, training, hyper)?;
    add_gp_noise_and_jitter(&mut k_tt, training, jitter);
    let k_pt = gp_covariance_matrix(predictions, training, hyper)?;
    let k_pp = gp_covariance_matrix(predictions, predictions, hyper)?;
    let observations =
        FeecVector::from_iterator(training.len(), training.iter().map(|t| t.observed));
    let Some(chol) = k_tt.clone().cholesky() else {
        return Err("GP training covariance Cholesky failed".to_string());
    };
    let alpha = chol.solve(&observations);
    let posterior_mean = &k_pt * alpha;
    let solved = chol.solve(&k_pt.transpose());
    let correction = &k_pt * solved;
    let mut variance = Vec::with_capacity(predictions.len());
    for index in 0..predictions.len() {
        variance.push((k_pp[(index, index)] - correction[(index, index)]).max(0.0));
    }
    Ok(GpPosteriorPrediction {
        mean: posterior_mean.iter().copied().collect(),
        variance,
    })
}

fn fit_source_template_gp_hyperparameters(
    config: &ToroidalGpBaselineConfig,
    training: &[SourceTemplateGpTarget],
    template_kind: SourceTemplateKind,
    source_prior_mean: f64,
    source_prior_variance: f64,
) -> Result<GpHyperparameters, String> {
    if training.is_empty() {
        return Err("source-template GP requires at least one training row".to_string());
    }
    let mut best = None;
    for &length_scale in &config.length_scales {
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(format!("invalid GP length scale {length_scale}"));
        }
        let unit_hyper = GpHyperparameters {
            matern_nu: config.matern_nu,
            length_scale,
            signal_variance: 1.0,
            log_marginal_likelihood: f64::NAN,
        };
        let unit_cov =
            source_template_gp_discrepancy_covariance_matrix(training, training, &unit_hyper)?;
        let base_signal_std = empirical_source_template_signal_std(
            training,
            &unit_cov,
            template_kind,
            source_prior_mean,
        );
        for &factor in &config.signal_std_factors {
            if !factor.is_finite() || factor <= 0.0 {
                return Err(format!("invalid GP signal std factor {factor}"));
            }
            let signal_std = (base_signal_std * factor).max(1e-16);
            let signal_variance = signal_std * signal_std;
            let hyper = GpHyperparameters {
                matern_nu: config.matern_nu,
                length_scale,
                signal_variance,
                log_marginal_likelihood: f64::NAN,
            };
            let logml = source_template_gp_log_marginal_likelihood(
                &hyper,
                training,
                template_kind,
                source_prior_mean,
                source_prior_variance,
                config.jitter,
            )?;
            let candidate = GpHyperparameters {
                log_marginal_likelihood: logml,
                ..hyper
            };
            if best
                .as_ref()
                .map(|current: &GpHyperparameters| {
                    candidate.log_marginal_likelihood > current.log_marginal_likelihood
                })
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| "source-template GP hyperparameter grid was empty".to_string())
}

fn empirical_source_template_signal_std(
    training: &[SourceTemplateGpTarget],
    unit_cov: &FeecMatrix,
    template_kind: SourceTemplateKind,
    source_prior_mean: f64,
) -> f64 {
    let mut acc = 0.0;
    let mut count = 0usize;
    for (index, target) in training.iter().enumerate() {
        let diag = unit_cov[(index, index)].abs().max(1e-300);
        let residual =
            target.target.observed - source_prior_mean * target.template_value(template_kind);
        acc += residual * residual / diag;
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    (acc / count as f64).max(1e-300).sqrt()
}

fn source_template_gp_log_marginal_likelihood(
    hyper: &GpHyperparameters,
    training: &[SourceTemplateGpTarget],
    template_kind: SourceTemplateKind,
    source_prior_mean: f64,
    source_prior_variance: f64,
    jitter: f64,
) -> Result<f64, String> {
    let mut cov = source_template_gp_covariance_matrix(
        training,
        training,
        hyper,
        template_kind,
        source_prior_variance,
    )?;
    add_source_template_noise_and_jitter(&mut cov, training, jitter);
    let residual =
        source_template_centered_observations(training, template_kind, source_prior_mean);
    let Some(chol) = cov.clone().cholesky() else {
        return Err("source-template GP training covariance Cholesky failed".to_string());
    };
    let alpha = chol.solve(&residual);
    let quadratic = residual.dot(&alpha);
    let lower = chol.l();
    let logdet = 2.0
        * (0..training.len())
            .map(|index| lower[(index, index)].abs().ln())
            .sum::<f64>();
    Ok(-0.5 * quadratic - 0.5 * logdet - 0.5 * training.len() as f64 * (2.0 * PI).ln())
}

fn condition_source_template_gp_functionals(
    hyper: &GpHyperparameters,
    training: &[SourceTemplateGpTarget],
    predictions: &[SourceTemplateGpTarget],
    template_kind: SourceTemplateKind,
    source_prior_mean: f64,
    source_prior_variance: f64,
    jitter: f64,
) -> Result<SourceTemplateGpPosteriorPrediction, String> {
    let mut k_tt = source_template_gp_covariance_matrix(
        training,
        training,
        hyper,
        template_kind,
        source_prior_variance,
    )?;
    add_source_template_noise_and_jitter(&mut k_tt, training, jitter);
    let residual =
        source_template_centered_observations(training, template_kind, source_prior_mean);
    let Some(chol) = k_tt.clone().cholesky() else {
        return Err("source-template GP training covariance Cholesky failed".to_string());
    };
    let alpha = chol.solve(&residual);

    let source_cross = FeecVector::from_iterator(
        training.len(),
        training
            .iter()
            .map(|target| source_prior_variance * target.template_value(template_kind)),
    );
    let source_solved = chol.solve(&source_cross);
    let source_mean = source_prior_mean + source_cross.dot(&alpha);
    let source_variance = (source_prior_variance - source_cross.dot(&source_solved)).max(0.0);

    if predictions.is_empty() {
        return Ok(SourceTemplateGpPosteriorPrediction {
            mean: Vec::new(),
            variance: Vec::new(),
            source_mean,
            source_variance,
        });
    }

    let k_pt = source_template_gp_covariance_matrix(
        predictions,
        training,
        hyper,
        template_kind,
        source_prior_variance,
    )?;
    let k_pp = source_template_gp_covariance_matrix(
        predictions,
        predictions,
        hyper,
        template_kind,
        source_prior_variance,
    )?;
    let posterior_correction_mean = &k_pt * alpha;
    let mut posterior_mean = Vec::with_capacity(predictions.len());
    for (index, target) in predictions.iter().enumerate() {
        posterior_mean.push(
            source_prior_mean * target.template_value(template_kind)
                + posterior_correction_mean[index],
        );
    }
    let solved = chol.solve(&k_pt.transpose());
    let correction = &k_pt * solved;
    let mut variance = Vec::with_capacity(predictions.len());
    for index in 0..predictions.len() {
        variance.push((k_pp[(index, index)] - correction[(index, index)]).max(0.0));
    }
    Ok(SourceTemplateGpPosteriorPrediction {
        mean: posterior_mean,
        variance,
        source_mean,
        source_variance,
    })
}

fn source_template_centered_observations(
    training: &[SourceTemplateGpTarget],
    template_kind: SourceTemplateKind,
    source_prior_mean: f64,
) -> FeecVector {
    FeecVector::from_iterator(
        training.len(),
        training.iter().map(|target| {
            target.target.observed - source_prior_mean * target.template_value(template_kind)
        }),
    )
}

fn source_template_gp_discrepancy_covariance_matrix(
    rows: &[SourceTemplateGpTarget],
    cols: &[SourceTemplateGpTarget],
    hyper: &GpHyperparameters,
) -> Result<FeecMatrix, String> {
    let mut matrix = FeecMatrix::zeros(rows.len(), cols.len());
    for (row_index, row) in rows.iter().enumerate() {
        for (col_index, col) in cols.iter().enumerate() {
            matrix[(row_index, col_index)] =
                gp_functional_covariance(&row.target.terms, &col.target.terms, hyper)?;
        }
    }
    Ok(matrix)
}

fn source_template_gp_covariance_matrix(
    rows: &[SourceTemplateGpTarget],
    cols: &[SourceTemplateGpTarget],
    hyper: &GpHyperparameters,
    template_kind: SourceTemplateKind,
    source_prior_variance: f64,
) -> Result<FeecMatrix, String> {
    let mut matrix = source_template_gp_discrepancy_covariance_matrix(rows, cols, hyper)?;
    for (row_index, row) in rows.iter().enumerate() {
        let row_template = row.template_value(template_kind);
        for (col_index, col) in cols.iter().enumerate() {
            matrix[(row_index, col_index)] +=
                source_prior_variance * row_template * col.template_value(template_kind);
        }
    }
    Ok(matrix)
}

fn add_source_template_noise_and_jitter(
    covariance: &mut FeecMatrix,
    training: &[SourceTemplateGpTarget],
    jitter: f64,
) {
    let diag_scale = (0..covariance.nrows())
        .map(|index| covariance[(index, index)].abs() + training[index].target.noise_variance)
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    for index in 0..covariance.nrows() {
        covariance[(index, index)] +=
            training[index].target.noise_variance + jitter.max(0.0) * diag_scale;
    }
}

fn add_gp_noise_and_jitter(
    covariance: &mut FeecMatrix,
    training: &[GpPredictionTarget],
    jitter: f64,
) {
    let diag_scale = (0..covariance.nrows())
        .map(|index| covariance[(index, index)].abs() + training[index].noise_variance)
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    for index in 0..covariance.nrows() {
        covariance[(index, index)] += training[index].noise_variance + jitter.max(0.0) * diag_scale;
    }
}

fn gp_covariance_matrix(
    rows: &[GpPredictionTarget],
    cols: &[GpPredictionTarget],
    hyper: &GpHyperparameters,
) -> Result<FeecMatrix, String> {
    let mut matrix = FeecMatrix::zeros(rows.len(), cols.len());
    for (row_index, row) in rows.iter().enumerate() {
        for (col_index, col) in cols.iter().enumerate() {
            matrix[(row_index, col_index)] =
                gp_functional_covariance(&row.terms, &col.terms, hyper)?;
        }
    }
    Ok(matrix)
}

fn gp_functional_covariance(
    lhs: &[GpFunctionalTerm],
    rhs: &[GpFunctionalTerm],
    hyper: &GpHyperparameters,
) -> Result<f64, String> {
    if lhs.is_empty() || rhs.is_empty() {
        return Err("GP functionals must have at least one quadrature term".to_string());
    }
    let mut covariance = 0.0;
    let kernel = EuclideanMaternConfig {
        kappa: 3.0_f64.sqrt() / hyper.length_scale,
        nu: hyper.matern_nu,
        variance: hyper.signal_variance,
    };
    for left in lhs {
        for right in rhs {
            let distance = distance3(left.point, right.point);
            let kernel_value =
                matern_covariance_euclidean(distance, kernel).map_err(|err| err.to_string())?;
            covariance += dot3(left.weight, right.weight) * kernel_value;
        }
    }
    Ok(covariance)
}

fn gp_prediction_row(
    stage: &str,
    sensor_type: &str,
    name: &str,
    truth: f64,
    prediction: f64,
    variance: f64,
) -> ToroidalGpBaselinePredictionRow {
    let sd = variance.max(0.0).sqrt();
    let residual = prediction - truth;
    ToroidalGpBaselinePredictionRow {
        stage: stage.to_string(),
        sensor_type: sensor_type.to_string(),
        name: name.to_string(),
        truth,
        prediction,
        residual,
        posterior_sd: sd,
        standardized_residual: if sd > EPS { residual / sd } else { f64::NAN },
        lower95: prediction - 1.96 * sd,
        upper95: prediction + 1.96 * sd,
        covered95: truth >= prediction - 1.96 * sd && truth <= prediction + 1.96 * sd,
    }
}

fn prediction_rmse(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return f64::NAN;
    }
    (residuals.iter().map(|value| value * value).sum::<f64>() / residuals.len() as f64).sqrt()
}

fn prediction_nlpd(rows: &[ToroidalGpBaselinePredictionRow]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    rows.iter()
        .map(|row| {
            let variance = row.posterior_sd * row.posterior_sd;
            if variance <= 0.0 {
                f64::INFINITY
            } else {
                0.5 * ((2.0 * PI * variance).ln() + row.residual * row.residual / variance)
            }
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn display_option(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "fixed".to_string())
}

fn variance_mode_name(mode: LinearPdeVarianceMode) -> &'static str {
    match mode {
        LinearPdeVarianceMode::Exact => "exact",
        LinearPdeVarianceMode::ExactSolves => "exact-solves",
        LinearPdeVarianceMode::MonteCarlo => "monte-carlo",
        LinearPdeVarianceMode::Hutchinson => "hutchinson",
        LinearPdeVarianceMode::LocalRbmc => "local-rbmc",
        LinearPdeVarianceMode::SelectedInverse => "selected-inverse",
    }
}

fn build_observations(
    config: &ToroidalHarmonicBConfig,
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    h2: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
    mass_2: &FeecCsr,
) -> Result<Vec<ObservationSpec>, String> {
    let d1_operator = build_exterior_derivative_row_operator(topology)?;
    let component_operators = [
        compose_sparse_row_operators(
            &build_sampled_magnetic_field_component_operator(topology, coords, 0)?,
            &d1_operator,
        )?,
        compose_sparse_row_operators(
            &build_sampled_magnetic_field_component_operator(topology, coords, 1)?,
            &d1_operator,
        )?,
        compose_sparse_row_operators(
            &build_sampled_magnetic_field_component_operator(topology, coords, 2)?,
            &d1_operator,
        )?,
    ];
    let h2_vectors = sample_2form_cell_vectors(coords, topology, &Cochain::new(2, h2.clone()))
        .map_err(|err| err.to_string())?;
    let mut observations = Vec::new();

    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    let face_rows = csr_rows(&d1);

    match config.observation_layout {
        ToroidalObservationLayout::Legacy => {
            push_legacy_field_observations(
                &mut observations,
                config,
                topology,
                coords,
                geom,
                &component_operators,
                &h2_vectors,
                &face_rows,
                h2,
                truth_a_nominal,
                truth_a_alpha,
                beta_true,
                source_harmonic_kappa,
            )?;
        }
        ToroidalObservationLayout::Embedded => {
            push_embedded_field_observations(
                &mut observations,
                config,
                topology,
                coords,
                geom,
                &component_operators,
                &h2_vectors,
                &face_rows,
                h2,
                truth_a_nominal,
                truth_a_alpha,
                beta_true,
                source_harmonic_kappa,
            )?;
        }
        ToroidalObservationLayout::TopologyPushforward => {
            push_embedded_field_observations(
                &mut observations,
                config,
                topology,
                coords,
                geom,
                &component_operators,
                &h2_vectors,
                &face_rows,
                h2,
                truth_a_nominal,
                truth_a_alpha,
                beta_true,
                source_harmonic_kappa,
            )?;
            push_local_flux_patch_observations(
                &mut observations,
                config,
                topology,
                coords,
                geom,
                &face_rows,
                h2,
                truth_a_nominal,
                truth_a_alpha,
                beta_true,
                source_harmonic_kappa,
            )?;
        }
        ToroidalObservationLayout::TopologySparseNoisyPrediction => {
            push_sparse_prediction_observation_bank(
                &mut observations,
                config,
                topology,
                coords,
                geom,
                &component_operators,
                &h2_vectors,
                &face_rows,
                h2,
                truth_a_nominal,
                truth_a_alpha,
                beta_true,
                source_harmonic_kappa,
            )?;
        }
    }

    if config.include_harmonic_projection_observation {
        let harmonic_operator = build_harmonic_projection_state_operator(topology, h2, mass_2)?;
        let beta_operator_value = mass_inner_product(h2, h2, mass_2);
        observations.push(make_observation(
            "harmonic",
            "harmonic_component",
            harmonic_operator,
            beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }

    if config.include_ampere_loop_observation
        && !matches!(
            config.observation_layout,
            ToroidalObservationLayout::TopologySparseNoisyPrediction
        )
    {
        let ampere_loop =
            build_ampere_loop_operator(topology, coords, geom, &component_operators, &h2_vectors)?;
        observations.push(make_observation_with_gp_terms(
            AMPERE_LOOP_SENSOR_TYPE,
            AMPERE_LOOP_NAME,
            ampere_loop.state_operator,
            ampere_loop.gp_terms,
            ampere_loop.beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }

    Ok(observations)
}

#[allow(clippy::too_many_arguments)]
fn build_heldout_observations(
    config: &ToroidalHarmonicBConfig,
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    component_operators: &[SparseRowOperator; 3],
    h2_vectors: &[[f64; 3]],
    face_rows: &[Vec<(usize, f64)>],
    h2: &FeecVector,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
) -> Result<Vec<ObservationSpec>, String> {
    let mut observations = Vec::new();
    let heldout_point = toroidal_shell_point(geom, 0.95, 0.25 * PI, 0.25 * PI);
    let cell_index = nearest_cell(topology, coords, heldout_point)?;
    for (component, suffix) in ["x", "y", "z"].into_iter().enumerate() {
        let mut direction = [0.0; 3];
        direction[component] = 1.0;
        let operator = combine_component_rows(component_operators, cell_index, direction)?;
        let beta_operator_value = h2_vectors[cell_index][component];
        let gp_terms = vec![GpFunctionalTerm {
            point: cell_barycenter(topology, coords, cell_index)?,
            weight: direction,
        }];
        let name = format!("heldout_hall_mid_b{suffix}");
        observations.push(make_observation_with_gp_terms(
            "heldout_hall",
            &name,
            operator,
            gp_terms,
            beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }

    let local_patch = build_flux_patch_operator(
        topology,
        coords,
        face_rows,
        [geom.major_radius, 0.0, -1.05],
        0.45,
        0.18,
    )?;
    let local_beta = local_patch
        .face_weights
        .iter()
        .map(|(face, value)| *value * h2[*face])
        .sum::<f64>();
    observations.push(make_observation_with_gp_terms(
        "heldout_flux",
        "heldout_flux_local_bottom",
        local_patch.state_operator,
        local_patch.gp_terms,
        local_beta,
        truth_a_nominal,
        truth_a_alpha,
        beta_true,
        source_harmonic_kappa,
        config,
    )?);

    let linked_panel = build_meridional_flux_panel_operator(
        topology,
        coords,
        face_rows,
        geom,
        0.25 * PI,
        0.75,
        0.75,
        0.20,
    )?;
    let linked_beta = linked_panel
        .face_weights
        .iter()
        .map(|(face, value)| *value * h2[*face])
        .sum::<f64>();
    observations.push(make_observation_with_gp_terms(
        "heldout_flux",
        "heldout_flux_link_p45",
        linked_panel.state_operator,
        linked_panel.gp_terms,
        linked_beta,
        truth_a_nominal,
        truth_a_alpha,
        beta_true,
        source_harmonic_kappa,
        config,
    )?);

    let ampere_loop = build_ampere_loop_operator_at_phi(
        topology,
        coords,
        geom,
        0.5 * PI,
        component_operators,
        h2_vectors,
    )?;
    observations.push(make_observation_with_gp_terms(
        "heldout_ampere_loop",
        "heldout_ampere_loop_poloidal_phi90",
        ampere_loop.state_operator,
        ampere_loop.gp_terms,
        ampere_loop.beta_operator_value,
        truth_a_nominal,
        truth_a_alpha,
        beta_true,
        source_harmonic_kappa,
        config,
    )?);

    Ok(observations)
}

#[allow(clippy::too_many_arguments)]
fn push_legacy_field_observations(
    observations: &mut Vec<ObservationSpec>,
    config: &ToroidalHarmonicBConfig,
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    component_operators: &[SparseRowOperator; 3],
    h2_vectors: &[[f64; 3]],
    face_rows: &[Vec<(usize, f64)>],
    h2: &FeecVector,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
) -> Result<(), String> {
    let hall_targets = [
        ("hall_inner_x", [geom.major_radius - 0.95, 0.0, 0.0]),
        ("hall_outer_x", [geom.major_radius + 0.95, 0.0, 0.0]),
        ("hall_top_x", [geom.major_radius, 0.0, 0.95]),
        ("hall_inner_y", [0.0, geom.major_radius - 0.95, 0.0]),
        ("hall_outer_y", [0.0, geom.major_radius + 0.95, 0.0]),
        ("hall_bottom_y", [0.0, geom.major_radius, -0.95]),
    ];
    let mut used_cells = HashSet::new();
    for (name, target) in hall_targets {
        let cell_index = nearest_unused_cell(topology, coords, target, &mut used_cells)?;
        let cell = SimplexIdx::new(topology.dim(), cell_index).handle(topology);
        let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
        let bary = cell_coords.barycenter();
        let direction = toroidal_direction(bary.as_view());
        let operator = combine_component_rows(component_operators, cell_index, direction)?;
        let beta_operator_value = dot3(h2_vectors[cell_index], direction);
        observations.push(make_observation(
            "hall",
            name,
            operator,
            beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }

    let flux_patches = [
        ("flux_inner", [geom.major_radius - 1.05, 0.0, 0.0]),
        ("flux_top", [geom.major_radius, 0.0, 1.05]),
        ("flux_outer", [geom.major_radius + 1.05, 0.0, 0.0]),
    ];
    for (name, center) in flux_patches {
        let patch = build_flux_patch_operator(topology, coords, face_rows, center, 0.45, 0.18)?;
        let beta_operator_value = patch
            .face_weights
            .iter()
            .map(|(face, value)| *value * h2[*face])
            .sum::<f64>();
        observations.push(make_observation_with_gp_terms(
            "flux",
            name,
            patch.state_operator,
            patch.gp_terms,
            beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_local_flux_patch_observations(
    observations: &mut Vec<ObservationSpec>,
    config: &ToroidalHarmonicBConfig,
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    face_rows: &[Vec<(usize, f64)>],
    h2: &FeecVector,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
) -> Result<(), String> {
    let flux_patches = [
        ("flux_inner", [geom.major_radius - 1.05, 0.0, 0.0]),
        ("flux_top", [geom.major_radius, 0.0, 1.05]),
        ("flux_outer", [geom.major_radius + 1.05, 0.0, 0.0]),
    ];
    for (name, center) in flux_patches {
        let patch = build_flux_patch_operator(topology, coords, face_rows, center, 0.45, 0.18)?;
        let beta_operator_value = patch
            .face_weights
            .iter()
            .map(|(face, value)| *value * h2[*face])
            .sum::<f64>();
        observations.push(make_observation_with_gp_terms(
            "flux",
            name,
            patch.state_operator,
            patch.gp_terms,
            beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_embedded_field_observations(
    observations: &mut Vec<ObservationSpec>,
    config: &ToroidalHarmonicBConfig,
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    component_operators: &[SparseRowOperator; 3],
    h2_vectors: &[[f64; 3]],
    face_rows: &[Vec<(usize, f64)>],
    h2: &FeecVector,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
) -> Result<(), String> {
    let sensor_minor_radius = 0.95;
    let phis = [0.0, 0.5 * PI, PI, 1.5 * PI];
    let thetas = [0.0, 0.5 * PI, PI, 1.5 * PI];
    for (phi_index, phi) in phis.into_iter().enumerate() {
        for (theta_index, theta) in thetas.into_iter().enumerate() {
            let target = toroidal_shell_point(geom, sensor_minor_radius, phi, theta);
            let cell_index = nearest_cell(topology, coords, target)?;
            for (component, suffix) in ["x", "y", "z"].into_iter().enumerate() {
                let mut direction = [0.0; 3];
                direction[component] = 1.0;
                let operator = combine_component_rows(component_operators, cell_index, direction)?;
                let beta_operator_value = h2_vectors[cell_index][component];
                let gp_terms = vec![GpFunctionalTerm {
                    point: cell_barycenter(topology, coords, cell_index)?,
                    weight: direction,
                }];
                let name = format!("embedded_hall_p{phi_index}_t{theta_index}_b{suffix}");
                observations.push(make_observation_with_gp_terms(
                    "hall",
                    &name,
                    operator,
                    gp_terms,
                    beta_operator_value,
                    truth_a_nominal,
                    truth_a_alpha,
                    beta_true,
                    source_harmonic_kappa,
                    config,
                )?);
            }
        }
    }

    for (phi_index, phi) in phis.into_iter().enumerate() {
        let panel = build_meridional_flux_panel_operator(
            topology, coords, face_rows, geom, phi, 0.75, 0.75, 0.20,
        )?;
        let beta_operator_value = panel
            .face_weights
            .iter()
            .map(|(face, value)| *value * h2[*face])
            .sum::<f64>();
        let name = format!("embedded_flux_panel_p{phi_index}");
        observations.push(make_observation_with_gp_terms(
            "flux",
            &name,
            panel.state_operator,
            panel.gp_terms,
            beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_sparse_prediction_observation_bank(
    observations: &mut Vec<ObservationSpec>,
    config: &ToroidalHarmonicBConfig,
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    component_operators: &[SparseRowOperator; 3],
    h2_vectors: &[[f64; 3]],
    face_rows: &[Vec<(usize, f64)>],
    h2: &FeecVector,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
) -> Result<(), String> {
    let sensor_minor_radius = 0.95;
    for phi_index in 0..config.sparse_prediction_field_phi_count {
        let phi = 2.0 * PI * phi_index as f64 / config.sparse_prediction_field_phi_count as f64;
        for theta_index in 0..config.sparse_prediction_field_theta_count {
            let theta =
                2.0 * PI * theta_index as f64 / config.sparse_prediction_field_theta_count as f64;
            let target = toroidal_shell_point(geom, sensor_minor_radius, phi, theta);
            let cell_index = nearest_cell(topology, coords, target)?;
            for (component, suffix) in ["x", "y", "z"].into_iter().enumerate() {
                let mut direction = [0.0; 3];
                direction[component] = 1.0;
                let operator = combine_component_rows(component_operators, cell_index, direction)?;
                let beta_operator_value = h2_vectors[cell_index][component];
                let name = format!("embedded_hall_p{phi_index}_t{theta_index}_b{suffix}");
                let gp_terms = vec![GpFunctionalTerm {
                    point: cell_barycenter(topology, coords, cell_index)?,
                    weight: direction,
                }];
                observations.push(make_observation_with_gp_terms(
                    "hall",
                    &name,
                    operator,
                    gp_terms,
                    beta_operator_value,
                    truth_a_nominal,
                    truth_a_alpha,
                    beta_true,
                    source_harmonic_kappa,
                    config,
                )?);
            }
        }
    }

    for phi_index in 0..config.sparse_prediction_linked_flux_count {
        let phi = 2.0 * PI * phi_index as f64 / config.sparse_prediction_linked_flux_count as f64;
        let panel = build_meridional_flux_panel_operator(
            topology, coords, face_rows, geom, phi, 0.75, 0.75, 0.20,
        )?;
        let beta_operator_value = panel
            .face_weights
            .iter()
            .map(|(face, value)| *value * h2[*face])
            .sum::<f64>();
        let name = format!("embedded_flux_panel_p{phi_index}");
        observations.push(make_observation_with_gp_terms(
            "flux",
            &name,
            panel.state_operator,
            panel.gp_terms,
            beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }

    push_local_flux_patch_observations(
        observations,
        config,
        topology,
        coords,
        geom,
        face_rows,
        h2,
        truth_a_nominal,
        truth_a_alpha,
        beta_true,
        source_harmonic_kappa,
    )?;
    let local_minor_radius = 1.05;
    for phi_index in 0..config.sparse_prediction_local_flux_phi_count {
        let phi =
            2.0 * PI * phi_index as f64 / config.sparse_prediction_local_flux_phi_count as f64;
        for theta_index in 0..config.sparse_prediction_local_flux_theta_count {
            let theta = 2.0 * PI * theta_index as f64
                / config.sparse_prediction_local_flux_theta_count as f64;
            let center = toroidal_shell_point(geom, local_minor_radius, phi, theta);
            let patch = build_flux_patch_operator(topology, coords, face_rows, center, 0.45, 0.18)?;
            let beta_operator_value = patch
                .face_weights
                .iter()
                .map(|(face, value)| *value * h2[*face])
                .sum::<f64>();
            let name = format!("dense_flux_local_p{phi_index}_t{theta_index}");
            observations.push(make_observation_with_gp_terms(
                "flux",
                &name,
                patch.state_operator,
                patch.gp_terms,
                beta_operator_value,
                truth_a_nominal,
                truth_a_alpha,
                beta_true,
                source_harmonic_kappa,
                config,
            )?);
        }
    }

    for phi_index in 0..config.sparse_prediction_ampere_loop_count {
        let phi = 2.0 * PI * phi_index as f64 / config.sparse_prediction_ampere_loop_count as f64;
        let loop_operator = build_ampere_loop_operator_at_phi(
            topology,
            coords,
            geom,
            phi,
            component_operators,
            h2_vectors,
        )?;
        let name = if phi_index == 0 {
            AMPERE_LOOP_NAME.to_string()
        } else {
            format!("dense_ampere_loop_p{phi_index}")
        };
        observations.push(make_observation_with_gp_terms(
            AMPERE_LOOP_SENSOR_TYPE,
            &name,
            loop_operator.state_operator,
            loop_operator.gp_terms,
            loop_operator.beta_operator_value,
            truth_a_nominal,
            truth_a_alpha,
            beta_true,
            source_harmonic_kappa,
            config,
        )?);
    }

    Ok(())
}

fn split_sparse_noisy_prediction_observations(
    observations: Vec<ObservationSpec>,
    config: &ToroidalHarmonicBConfig,
) -> Result<(Vec<ObservationSpec>, Vec<ObservationSpec>), String> {
    let mut by_kind = BTreeMap::<String, Vec<usize>>::new();
    for (index, observation) in observations.iter().enumerate() {
        by_kind
            .entry(observation.sensor_type.clone())
            .or_default()
            .push(index);
    }

    let mut training_indices = HashSet::<usize>::new();
    for indices in by_kind.values() {
        for index in
            stratified_training_indices(indices, config.sparse_prediction_training_fraction)
        {
            training_indices.insert(index);
        }
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.sparse_prediction_noise_seed);
    let mut training = Vec::new();
    let mut heldout = Vec::new();
    for (index, observation) in observations.into_iter().enumerate() {
        if training_indices.contains(&index) {
            training.push(with_noisy_source_generated_observation(
                observation,
                &mut rng,
            ));
        } else {
            heldout.push(observation);
        }
    }
    if training.is_empty() || heldout.is_empty() {
        return Err(format!(
            "sparse noisy split produced {} training and {} held-out observations",
            training.len(),
            heldout.len()
        ));
    }
    Ok((training, heldout))
}

fn stratified_training_indices(indices: &[usize], fraction: f64) -> Vec<usize> {
    if indices.is_empty() {
        return Vec::new();
    }
    let count = ((indices.len() as f64 * fraction).ceil() as usize)
        .max(1)
        .min(indices.len());
    let mut selected = Vec::with_capacity(count);
    let mut used = HashSet::<usize>::new();
    for rank in 0..count {
        let local = (((rank as f64 + 0.5) * indices.len() as f64 / count as f64).floor() as usize)
            .min(indices.len() - 1);
        let mut candidate = local;
        while used.contains(&candidate) {
            candidate = (candidate + 1).min(indices.len() - 1);
            if used.contains(&candidate) && candidate == indices.len() - 1 {
                break;
            }
        }
        if used.insert(candidate) {
            selected.push(indices[candidate]);
        }
    }
    selected
}

fn with_noisy_source_generated_observation(
    mut observation: ObservationSpec,
    rng: &mut rand::rngs::StdRng,
) -> ObservationSpec {
    let noise =
        rng.sample::<f64, _>(StandardNormal) * observation.variance_source_generated_truth.sqrt();
    observation.observation_source_generated_observed =
        observation.observation_source_generated_truth + noise;
    observation
}

fn make_observation(
    sensor_type: &str,
    name: &str,
    state_operator: SparseTripletMatrix,
    beta_operator_value: f64,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
    config: &ToroidalHarmonicBConfig,
) -> Result<ObservationSpec, String> {
    make_observation_with_gp_terms(
        sensor_type,
        name,
        state_operator,
        Vec::new(),
        beta_operator_value,
        truth_a_nominal,
        truth_a_alpha,
        beta_true,
        source_harmonic_kappa,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn make_observation_with_gp_terms(
    sensor_type: &str,
    name: &str,
    state_operator: SparseTripletMatrix,
    gp_terms: Vec<GpFunctionalTerm>,
    beta_operator_value: f64,
    truth_a_nominal: &FeecVector,
    truth_a_alpha: &FeecVector,
    beta_true: f64,
    source_harmonic_kappa: f64,
    config: &ToroidalHarmonicBConfig,
) -> Result<ObservationSpec, String> {
    let state_nominal = apply_triplet_row(&state_operator, truth_a_nominal)?;
    let state_alpha = apply_triplet_row(&state_operator, truth_a_alpha)?;
    let observation_beta_truth = state_nominal + beta_operator_value * beta_true;
    let observation_alpha_beta_truth = state_alpha + beta_operator_value * beta_true;
    let observation_source_generated_truth =
        config.source_alpha_true * (state_nominal + beta_operator_value * source_harmonic_kappa);
    Ok(ObservationSpec {
        sensor_type: sensor_type.to_string(),
        name: name.to_string(),
        state_operator,
        gp_terms,
        beta_operator_value,
        observation_beta_truth,
        observation_alpha_beta_truth,
        observation_source_generated_truth,
        observation_source_generated_observed: observation_source_generated_truth,
        variance_beta_truth: observation_variance_from_value(observation_beta_truth, config),
        variance_alpha_beta_truth: observation_variance_from_value(
            observation_alpha_beta_truth,
            config,
        ),
        variance_source_generated_truth: observation_variance_from_value(
            observation_source_generated_truth,
            config,
        ),
    })
}

fn observation_variance_from_value(value: f64, config: &ToroidalHarmonicBConfig) -> f64 {
    let std = config.relative_noise_std * value.abs().max(config.noise_floor);
    (std * std).max(MIN_OBSERVATION_VARIANCE)
}

fn joint_measurement_spec(
    observation: &ObservationSpec,
    truth: ObservationTruth,
) -> LinearPdeJointMeasurementSpec {
    LinearPdeJointMeasurementSpec {
        name: observation.name.clone(),
        state_operator: Some(observation.state_operator.clone()),
        latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
            input_name: BETA_INPUT_NAME.to_string(),
            operator: beta_column_triplet(observation.beta_operator_value),
        }],
        observations: vec![observation_value(observation, truth)],
        bias: vec![0.0],
        variance: observation_variance(observation, truth),
    }
}

fn fluctuation_joint_measurement_spec(
    observation: &ObservationSpec,
    workspace: &ExperimentWorkspace,
    truth: ObservationTruth,
) -> Result<LinearPdeJointMeasurementSpec, String> {
    let alpha_operator_value =
        apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)?;
    Ok(LinearPdeJointMeasurementSpec {
        name: observation.name.clone(),
        state_operator: Some(observation.state_operator.clone()),
        latent_operators: vec![
            LinearPdeLatentMeasurementBlockSpec {
                input_name: ALPHA_INPUT_NAME.to_string(),
                operator: scalar_column_triplet(alpha_operator_value),
            },
            LinearPdeLatentMeasurementBlockSpec {
                input_name: BETA_INPUT_NAME.to_string(),
                operator: scalar_column_triplet(observation.beta_operator_value),
            },
        ],
        observations: vec![observation_value(observation, truth)],
        bias: vec![0.0],
        variance: observation_variance(observation, truth),
    })
}

fn source_generated_joint_measurement_spec(
    observation: &ObservationSpec,
    workspace: &ExperimentWorkspace,
    is_full_state: bool,
) -> Result<LinearPdeJointMeasurementSpec, String> {
    Ok(LinearPdeJointMeasurementSpec {
        name: observation.name.clone(),
        state_operator: Some(observation.state_operator.clone()),
        latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
            input_name: ALPHA_INPUT_NAME.to_string(),
            operator: scalar_column_triplet(source_generated_alpha_sensitivity(
                observation,
                workspace,
                is_full_state,
            )?),
        }],
        observations: vec![observation.observation_source_generated_observed],
        bias: vec![0.0],
        variance: observation.variance_source_generated_truth,
    })
}

fn joint_observation_derived_spec(
    observation: &ObservationSpec,
    workspace: &ExperimentWorkspace,
    is_fluctuation_recovery: bool,
) -> Result<LinearPdeJointDerivedQuantitySpec, String> {
    let mut latent_operators = vec![LinearPdeLatentDerivedBlockSpec {
        input_name: BETA_INPUT_NAME.to_string(),
        operator: SparseRowOperator::new(1, vec![vec![(0, observation.beta_operator_value)]])
            .map_err(|err| err.to_string())?,
    }];
    if is_fluctuation_recovery {
        let alpha_operator_value =
            apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)?;
        latent_operators.push(LinearPdeLatentDerivedBlockSpec {
            input_name: ALPHA_INPUT_NAME.to_string(),
            operator: SparseRowOperator::new(1, vec![vec![(0, alpha_operator_value)]])
                .map_err(|err| err.to_string())?,
        });
    }
    Ok(LinearPdeJointDerivedQuantitySpec {
        name: observation_derived_name(&observation.name),
        state_operator: Some(triplet_to_row_operator(&observation.state_operator)?),
        latent_operators,
    })
}

fn source_generated_observation_derived_spec(
    observation: &ObservationSpec,
    workspace: &ExperimentWorkspace,
    is_full_state: bool,
) -> Result<LinearPdeJointDerivedQuantitySpec, String> {
    source_generated_observation_derived_spec_named(
        &observation_derived_name(&observation.name),
        observation,
        workspace,
        is_full_state,
    )
}

fn source_generated_observation_derived_spec_named(
    name: &str,
    observation: &ObservationSpec,
    workspace: &ExperimentWorkspace,
    is_full_state: bool,
) -> Result<LinearPdeJointDerivedQuantitySpec, String> {
    Ok(LinearPdeJointDerivedQuantitySpec {
        name: name.to_string(),
        state_operator: Some(triplet_to_row_operator(&observation.state_operator)?),
        latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
            input_name: ALPHA_INPUT_NAME.to_string(),
            operator: scalar_row_operator(source_generated_alpha_sensitivity(
                observation,
                workspace,
                is_full_state,
            )?)?,
        }],
    })
}

fn topology_pushforward_qoi_derived_specs(
    workspace: &ExperimentWorkspace,
) -> Result<Vec<LinearPdeJointDerivedQuantitySpec>, String> {
    let harmonic_projection_operator = build_harmonic_projection_state_operator(
        &workspace.topology,
        &workspace.h2,
        &workspace.mass_2,
    )?;
    let link_flux = topology_pushforward_observation(workspace, "embedded_flux_panel_p0")?;
    let local_flux = topology_pushforward_observation(workspace, "flux_top")?;
    let ampere_loop = topology_pushforward_observation(workspace, AMPERE_LOOP_NAME)?;
    let field_x = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_bx")?;
    let field_y = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_by")?;
    let field_z = topology_pushforward_observation(workspace, "embedded_hall_p0_t0_bz")?;

    let mut specs = vec![
        LinearPdeJointDerivedQuantitySpec {
            name: QOI_SOURCE_NAME.to_string(),
            state_operator: None,
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: ALPHA_INPUT_NAME.to_string(),
                operator: scalar_row_operator(1.0)?,
            }],
        },
        LinearPdeJointDerivedQuantitySpec {
            name: QOI_SOURCE_BETA_NAME.to_string(),
            state_operator: None,
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: ALPHA_INPUT_NAME.to_string(),
                operator: scalar_row_operator(workspace.topology_summary.source_harmonic_kappa)?,
            }],
        },
        LinearPdeJointDerivedQuantitySpec {
            name: QOI_HARMONIC_PROJECTION_NAME.to_string(),
            state_operator: Some(triplet_to_row_operator(&harmonic_projection_operator)?),
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: ALPHA_INPUT_NAME.to_string(),
                operator: scalar_row_operator(workspace.topology_summary.source_harmonic_kappa)?,
            }],
        },
        LinearPdeJointDerivedQuantitySpec {
            name: BRANCH_LINK_FLUX_EXACT_NAME.to_string(),
            state_operator: Some(triplet_to_row_operator(&link_flux.state_operator)?),
            latent_operators: Vec::new(),
        },
    ];
    for (name, observation) in [
        (QOI_LINK_FLUX_NAME, link_flux),
        (QOI_LOCAL_FLUX_NAME, local_flux),
        (QOI_AMPERE_LOOP_NAME, ampere_loop),
        (QOI_FIELD_X_NAME, field_x),
        (QOI_FIELD_Y_NAME, field_y),
        (QOI_FIELD_Z_NAME, field_z),
    ] {
        specs.push(source_generated_observation_derived_spec_named(
            name,
            observation,
            workspace,
            true,
        )?);
    }
    Ok(specs)
}

fn topology_pushforward_observation<'a>(
    workspace: &'a ExperimentWorkspace,
    name: &str,
) -> Result<&'a ObservationSpec, String> {
    workspace
        .observations
        .iter()
        .chain(workspace.heldout_observations.iter())
        .find(|observation| observation.name == name)
        .ok_or_else(|| format!("topology pushforward observation `{name}` is missing"))
}

fn source_generated_scalar_derived_specs(
    workspace: &ExperimentWorkspace,
    is_full_state: bool,
) -> Result<Vec<LinearPdeJointDerivedQuantitySpec>, String> {
    let harmonic_projection_operator = build_harmonic_projection_state_operator(
        &workspace.topology,
        &workspace.h2,
        &workspace.mass_2,
    )?;
    let linked_current_state_operator = is_full_state
        .then(|| triplet_to_row_operator(&workspace.ampere_loop_state_operator))
        .transpose()?;
    let linked_current_latent_operator = if is_full_state {
        workspace.topology_summary.source_harmonic_kappa
            * workspace.topology_summary.ampere_loop_harmonic_sensitivity
    } else {
        workspace.topology_summary.linked_current_unit
    };
    let harmonic_projection_latent_operator = if is_full_state {
        workspace.topology_summary.source_harmonic_kappa
    } else {
        workspace.topology_summary.source_harmonic_projection_unit
    };
    Ok(vec![
        LinearPdeJointDerivedQuantitySpec {
            name: SOURCE_BETA_DERIVED_NAME.to_string(),
            state_operator: None,
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: ALPHA_INPUT_NAME.to_string(),
                operator: scalar_row_operator(workspace.topology_summary.source_harmonic_kappa)?,
            }],
        },
        LinearPdeJointDerivedQuantitySpec {
            name: SOURCE_LINKED_CURRENT_DERIVED_NAME.to_string(),
            state_operator: linked_current_state_operator,
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: ALPHA_INPUT_NAME.to_string(),
                operator: scalar_row_operator(linked_current_latent_operator)?,
            }],
        },
        LinearPdeJointDerivedQuantitySpec {
            name: SOURCE_HARMONIC_PROJECTION_DERIVED_NAME.to_string(),
            state_operator: Some(triplet_to_row_operator(&harmonic_projection_operator)?),
            latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
                input_name: ALPHA_INPUT_NAME.to_string(),
                operator: scalar_row_operator(harmonic_projection_latent_operator)?,
            }],
        },
    ])
}

fn source_generated_alpha_sensitivity(
    observation: &ObservationSpec,
    workspace: &ExperimentWorkspace,
    is_full_state: bool,
) -> Result<f64, String> {
    if is_full_state {
        Ok(workspace.topology_summary.source_harmonic_kappa * observation.beta_operator_value)
    } else {
        Ok(
            apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)?
                + workspace.topology_summary.source_harmonic_kappa
                    * observation.beta_operator_value,
        )
    }
}

fn observation_reports(
    stage: &str,
    workspace: &ExperimentWorkspace,
    result: &LinearPdeUqResult,
    truth: ObservationTruth,
    prediction: StageObservationPrediction,
) -> Result<Vec<ToroidalObservationUncertaintyRow>, String> {
    workspace
        .observations
        .iter()
        .map(|observation| {
            let prediction = predict_observation(workspace, result, observation, prediction)?;
            let observed = observation_value(observation, truth);
            let derived_name = observation_derived_name(&observation.name);
            let variances = result
                .derived_variances
                .get(&derived_name)
                .ok_or_else(|| format!("missing derived variance for `{derived_name}`"))?;
            Ok(ToroidalObservationUncertaintyRow {
                stage: stage.to_string(),
                sensor_type: observation.sensor_type.clone(),
                name: observation.name.clone(),
                observed,
                prediction,
                residual: prediction - observed,
                prior_variance: variances.prior_variance[0],
                posterior_variance: variances.posterior_variance[0],
            })
        })
        .collect()
}

fn predict_observation(
    workspace: &ExperimentWorkspace,
    result: &LinearPdeUqResult,
    observation: &ObservationSpec,
    prediction: StageObservationPrediction,
) -> Result<f64, String> {
    let beta_mean = latent_posterior_summary(result, BETA_INPUT_NAME)
        .map(|summary| summary.0)
        .unwrap_or(0.0);
    let alpha_mean = latent_posterior_summary(result, ALPHA_INPUT_NAME)
        .map(|summary| summary.0)
        .unwrap_or(0.0);
    let state_prediction = apply_triplet_row(&observation.state_operator, &result.posterior_mean)?;
    let alpha_prediction = if prediction == StageObservationPrediction::FluctuationAlphaBeta {
        alpha_mean * apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)?
    } else if prediction == StageObservationPrediction::SourceGeneratedAlpha {
        alpha_mean * source_generated_alpha_sensitivity(observation, workspace, false)?
    } else if prediction == StageObservationPrediction::SourceGeneratedFullStateAlpha {
        alpha_mean * source_generated_alpha_sensitivity(observation, workspace, true)?
    } else {
        0.0
    };
    Ok(state_prediction + alpha_prediction + beta_mean * observation.beta_operator_value)
}

fn observation_value(observation: &ObservationSpec, truth: ObservationTruth) -> f64 {
    match truth {
        ObservationTruth::BetaOnly => observation.observation_beta_truth,
        ObservationTruth::AlphaBeta => observation.observation_alpha_beta_truth,
        ObservationTruth::SourceGenerated => observation.observation_source_generated_observed,
    }
}

fn observation_variance(observation: &ObservationSpec, truth: ObservationTruth) -> f64 {
    match truth {
        ObservationTruth::BetaOnly => observation.variance_beta_truth,
        ObservationTruth::AlphaBeta => observation.variance_alpha_beta_truth,
        ObservationTruth::SourceGenerated => observation.variance_source_generated_truth,
    }
}

fn observation_chi2_for_stage(
    workspace: &ExperimentWorkspace,
    stage: &ToroidalStageResult,
    sensor_type: Option<&str>,
) -> f64 {
    stage
        .observations
        .iter()
        .filter(|row| sensor_type.map_or(true, |kind| row.sensor_type == kind))
        .map(|row| {
            let variance = workspace
                .observations
                .iter()
                .find(|observation| observation.name == row.name)
                .map(|observation| observation.variance_source_generated_truth)
                .unwrap_or(1.0);
            row.residual * row.residual / variance
        })
        .sum()
}

fn observation_derived_name(name: &str) -> String {
    format!("{OBS_DERIVED_PREFIX}{name}")
}

fn latent_prior_summary(problem: &LinearPdeUqProblem, name: &str) -> Option<(f64, f64)> {
    problem.uncertain_inputs.iter().find_map(|input| {
        (input.name == name).then(|| (input.prior.mean[0], 1.0 / input.prior.precision_value(0, 0)))
    })
}

trait PrecisionLookup {
    fn precision_value(&self, row: usize, col: usize) -> f64;
}

impl PrecisionLookup for GaussianPriorSpec {
    fn precision_value(&self, row: usize, col: usize) -> f64 {
        self.precision
            .triplet_iter()
            .find_map(|(r, c, value)| (r == row && c == col).then_some(value))
            .unwrap_or(0.0)
    }
}

fn latent_posterior_summary(result: &LinearPdeUqResult, name: &str) -> Option<(f64, f64)> {
    result
        .latent_inputs
        .iter()
        .find(|input| input.name == name)
        .map(|input| (input.mean[0], input.variance[0]))
}

fn mean_variance_ratio(variances: &LinearPdeDerivedMarginalResult) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (prior, posterior) in variances
        .prior_variance
        .iter()
        .zip(variances.posterior_variance.iter())
    {
        if *prior > EPS && prior.is_finite() && posterior.is_finite() {
            sum += (*posterior / *prior).max(0.0);
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn rmse_for_kind(rows: &[ToroidalObservationUncertaintyRow], kind: &str) -> f64 {
    let values = rows
        .iter()
        .filter(|row| row.sensor_type == kind)
        .map(|row| row.residual)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn write_outputs(
    _config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    result: &ToroidalHarmonicBResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    write_topology_summary(
        &result.topology_summary,
        &out_dir.join("topology_summary.json"),
    )?;
    write_topology_summary_csv(
        &result.topology_summary,
        &out_dir.join("topology_summary.csv"),
    )?;
    write_stage_summary_csv(&result.stages, &out_dir.join("stage_summary.csv"))?;
    write_harmonic_posterior_csv(
        &result.stages,
        result.topology_summary.beta_true,
        &out_dir.join("harmonic_posterior.csv"),
    )?;
    write_source_generated_summary_csv(
        workspace,
        &result.stages,
        &out_dir.join("source_generated_summary.csv"),
    )?;
    write_source_generated_objective_diagnostics_csv(
        _config,
        workspace,
        &result.stages,
        &out_dir.join("source_generated_objective_diagnostics.csv"),
    )?;
    write_source_generated_field_errors_csv(
        workspace,
        &result.stages,
        &out_dir.join("source_generated_field_errors.csv"),
    )?;
    write_observation_uncertainty_csv(
        &result.stages,
        &out_dir.join("observation_uncertainty.csv"),
    )?;
    write_ampere_loop_summary_csv(
        workspace,
        &result.stages,
        &out_dir.join("ampere_loop_summary.csv"),
    )?;
    write_pushforward_qoi_summary_csv(
        &result.stages,
        &out_dir.join("pushforward_qoi_summary.csv"),
    )?;
    write_pushforward_qoi_covariance_csv(
        &result.stages,
        &out_dir.join("pushforward_qoi_covariance.csv"),
    )?;
    write_pushforward_variance_ratios_csv(
        &result.stages,
        &out_dir.join("pushforward_variance_ratios.csv"),
    )?;
    write_heldout_prediction_csv(&result.stages, &out_dir.join("heldout_prediction.csv"))?;
    write_sparse_prediction_split_csv(workspace, &out_dir.join("sparse_prediction_split.csv"))?;
    write_branch_decomposition_csv(&result.stages, &out_dir.join("branch_decomposition.csv"))?;
    write_field_trace_variance_csv(&result.stages, &out_dir.join("field_trace_variance.csv"))?;
    write_truth_vtus(workspace, out_dir)?;
    for stage in &result.stages {
        write_stage_vtus(workspace, stage, out_dir)?;
    }
    Ok(())
}

fn write_topology_summary(summary: &ToroidalTopologySummary, path: &Path) -> io::Result<()> {
    let betti = summary
        .betti_numbers
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        concat!(
            "{{\n",
            "  \"betti_numbers\": [{}],\n",
            "  \"harmonic_2_dimension\": {},\n",
            "  \"harmonic_2_mass_norm\": {:.16e},\n",
            "  \"deterministic_harmonic_projection\": {:.16e},\n",
            "  \"deterministic_harmonic_projection_relative\": {:.16e},\n",
            "  \"deterministic_b_energy\": {:.16e},\n",
            "  \"beta_true\": {:.16e},\n",
            "  \"source_harmonic_energy_fraction\": {:.16e},\n",
            "  \"source_harmonic_kappa\": {:.16e},\n",
            "  \"ampere_loop_exact_nominal\": {:.16e},\n",
            "  \"ampere_loop_harmonic_sensitivity\": {:.16e},\n",
            "  \"linked_current_unit\": {:.16e},\n",
            "  \"linked_current_true\": {:.16e},\n",
            "  \"source_harmonic_projection_unit\": {:.16e},\n",
            "  \"source_harmonic_projection_true\": {:.16e}\n",
            "}}\n"
        ),
        betti,
        summary.harmonic_2_dimension,
        summary.harmonic_2_mass_norm,
        summary.deterministic_harmonic_projection,
        summary.deterministic_harmonic_projection_relative,
        summary.deterministic_b_energy,
        summary.beta_true,
        summary.source_harmonic_energy_fraction,
        summary.source_harmonic_kappa,
        summary.ampere_loop_exact_nominal,
        summary.ampere_loop_harmonic_sensitivity,
        summary.linked_current_unit,
        summary.linked_current_true,
        summary.source_harmonic_projection_unit,
        summary.source_harmonic_projection_true
    );
    fs::write(path, json)
}

fn write_topology_summary_csv(summary: &ToroidalTopologySummary, path: &Path) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    let betti_numbers = format!("{:?}", summary.betti_numbers).replace('"', "\"\"");
    writeln!(file, "metric,value")?;
    writeln!(file, "betti_numbers,\"{}\"", betti_numbers)?;
    writeln!(
        file,
        "harmonic_2_dimension,{}",
        summary.harmonic_2_dimension
    )?;
    writeln!(
        file,
        "harmonic_2_mass_norm,{:.16e}",
        summary.harmonic_2_mass_norm
    )?;
    writeln!(
        file,
        "deterministic_harmonic_projection,{:.16e}",
        summary.deterministic_harmonic_projection
    )?;
    writeln!(
        file,
        "deterministic_harmonic_projection_relative,{:.16e}",
        summary.deterministic_harmonic_projection_relative
    )?;
    writeln!(
        file,
        "deterministic_b_energy,{:.16e}",
        summary.deterministic_b_energy
    )?;
    writeln!(file, "beta_true,{:.16e}", summary.beta_true)?;
    writeln!(
        file,
        "source_harmonic_energy_fraction,{:.16e}",
        summary.source_harmonic_energy_fraction
    )?;
    writeln!(
        file,
        "source_harmonic_kappa,{:.16e}",
        summary.source_harmonic_kappa
    )?;
    writeln!(
        file,
        "ampere_loop_exact_nominal,{:.16e}",
        summary.ampere_loop_exact_nominal
    )?;
    writeln!(
        file,
        "ampere_loop_harmonic_sensitivity,{:.16e}",
        summary.ampere_loop_harmonic_sensitivity
    )?;
    writeln!(
        file,
        "linked_current_unit,{:.16e}",
        summary.linked_current_unit
    )?;
    writeln!(
        file,
        "linked_current_true,{:.16e}",
        summary.linked_current_true
    )?;
    writeln!(
        file,
        "source_harmonic_projection_unit,{:.16e}",
        summary.source_harmonic_projection_unit
    )?;
    writeln!(
        file,
        "source_harmonic_projection_true,{:.16e}",
        summary.source_harmonic_projection_true
    )?;
    Ok(())
}

fn write_stage_summary_csv(stages: &[ToroidalStageResult], path: &Path) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,latent_dimension,pde_residual_norm,hall_rmse,flux_rmse,harmonic_observation_residual,b_variance_ratio_mean,beta_prior_mean,beta_prior_variance,beta_posterior_mean,beta_posterior_variance,beta_error,alpha_prior_mean,alpha_prior_variance,alpha_posterior_mean,alpha_posterior_variance,alpha_error,prior_precision_nnz,prior_precision_lower_triangle_nnz,prior_factor_nnz,prior_factor_fill_in,prior_factor_values_mib,posterior_precision_nnz,posterior_precision_lower_triangle_nnz,posterior_factor_nnz,posterior_factor_fill_in,posterior_factor_values_mib"
    )?;
    for stage in stages {
        let s = &stage.summary;
        let prior = stage.solve.debug.prior_factorization;
        let posterior = stage.solve.debug.posterior_factorization;
        writeln!(
            file,
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{},{},{},{:.16e},{:.16e}",
            s.stage,
            s.latent_dimension,
            s.pde_residual_norm,
            s.hall_rmse,
            s.flux_rmse,
            s.harmonic_observation_residual,
            s.b_variance_ratio_mean,
            csv_option(s.beta_prior_mean),
            csv_option(s.beta_prior_variance),
            csv_option(s.beta_posterior_mean),
            csv_option(s.beta_posterior_variance),
            csv_option(s.beta_error),
            csv_option(s.alpha_prior_mean),
            csv_option(s.alpha_prior_variance),
            csv_option(s.alpha_posterior_mean),
            csv_option(s.alpha_posterior_variance),
            csv_option(s.alpha_error),
            prior.matrix_nnz,
            prior.matrix_lower_triangle_nnz,
            prior.factor_nnz,
            prior.fill_in_ratio_vs_lower_triangle,
            prior.factor_numeric_values_mib,
            posterior.matrix_nnz,
            posterior.matrix_lower_triangle_nnz,
            posterior.factor_nnz,
            posterior.fill_in_ratio_vs_lower_triangle,
            posterior.factor_numeric_values_mib
        )?;
    }
    Ok(())
}

fn write_harmonic_posterior_csv(
    stages: &[ToroidalStageResult],
    beta_true: f64,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,beta_prior_mean,beta_prior_variance,beta_posterior_mean,beta_posterior_variance,beta_truth,beta_error,posterior_prior_variance_ratio"
    )?;
    for stage in stages {
        let s = &stage.summary;
        let ratio = match (s.beta_prior_variance, s.beta_posterior_variance) {
            (Some(prior), Some(post)) if prior > 0.0 => Some(post / prior),
            _ => None,
        };
        writeln!(
            file,
            "{},{},{},{},{},{:.16e},{},{}",
            s.stage,
            csv_option(s.beta_prior_mean),
            csv_option(s.beta_prior_variance),
            csv_option(s.beta_posterior_mean),
            csv_option(s.beta_posterior_variance),
            beta_true,
            csv_option(s.beta_error),
            csv_option(ratio)
        )?;
    }
    Ok(())
}

fn write_source_generated_summary_csv(
    workspace: &ExperimentWorkspace,
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,alpha_prior_mean,alpha_prior_variance,alpha_posterior_mean,alpha_posterior_variance,alpha_truth,alpha_error,beta_source_prior_mean,beta_source_prior_variance,beta_source_posterior_mean,beta_source_posterior_variance,beta_source_truth,beta_source_error,linked_current_prior_mean,linked_current_prior_variance,linked_current_posterior_mean,linked_current_posterior_variance,linked_current_truth,linked_current_error,harmonic_projection_prior_mean,harmonic_projection_prior_variance,harmonic_projection_posterior_mean,harmonic_projection_posterior_variance,harmonic_projection_truth,harmonic_projection_error"
    )?;
    for stage in stages
        .iter()
        .filter(|stage| is_source_generated_stage(&stage.summary.stage))
    {
        let alpha_prior_mean = stage.summary.alpha_prior_mean.unwrap_or(1.0);
        let alpha_post_mean = stage
            .summary
            .alpha_posterior_mean
            .unwrap_or(alpha_prior_mean);
        let alpha_truth = workspace.topology_summary.linked_current_true
            / workspace.topology_summary.linked_current_unit;
        let beta_prior_mean = workspace.topology_summary.source_harmonic_kappa * alpha_prior_mean;
        let beta_post_mean = workspace.topology_summary.source_harmonic_kappa * alpha_post_mean;
        let is_full_state = is_source_generated_full_state_stage(&stage.summary.stage);
        let linked_prior_mean = if is_full_state {
            workspace.topology_summary.source_harmonic_kappa
                * workspace.topology_summary.ampere_loop_harmonic_sensitivity
                * alpha_prior_mean
        } else {
            workspace.topology_summary.linked_current_unit * alpha_prior_mean
        };
        let linked_post_mean =
            source_generated_linked_current_mean(workspace, stage).unwrap_or(if is_full_state {
                workspace.topology_summary.source_harmonic_kappa
                    * workspace.topology_summary.ampere_loop_harmonic_sensitivity
                    * alpha_post_mean
            } else {
                workspace.topology_summary.linked_current_unit * alpha_post_mean
            });
        let projection_prior_mean = if is_full_state {
            workspace.topology_summary.source_harmonic_kappa * alpha_prior_mean
        } else {
            workspace.topology_summary.source_harmonic_projection_unit * alpha_prior_mean
        };
        let projection_post_mean = source_generated_harmonic_projection_mean(workspace, stage)
            .unwrap_or(
                workspace.topology_summary.source_harmonic_projection_unit * alpha_post_mean,
            );
        let beta_variance = derived_scalar_variances(stage, SOURCE_BETA_DERIVED_NAME);
        let linked_variance = derived_scalar_variances(stage, SOURCE_LINKED_CURRENT_DERIVED_NAME);
        let projection_variance =
            derived_scalar_variances(stage, SOURCE_HARMONIC_PROJECTION_DERIVED_NAME);
        writeln!(
            file,
            "{},{},{},{},{},{:.16e},{},{},{},{},{},{:.16e},{},{},{},{},{},{:.16e},{},{},{},{},{},{:.16e},{}",
            stage.summary.stage,
            csv_option(stage.summary.alpha_prior_mean),
            csv_option(stage.summary.alpha_prior_variance),
            csv_option(stage.summary.alpha_posterior_mean),
            csv_option(stage.summary.alpha_posterior_variance),
            alpha_truth,
            csv_option(stage.summary.alpha_error),
            csv_option(Some(beta_prior_mean)),
            csv_option(beta_variance.map(|value| value.0)),
            csv_option(Some(beta_post_mean)),
            csv_option(beta_variance.map(|value| value.1)),
            workspace.topology_summary.source_harmonic_kappa * alpha_truth,
            csv_option(Some(beta_post_mean - workspace.topology_summary.source_harmonic_kappa * alpha_truth)),
            csv_option(Some(linked_prior_mean)),
            csv_option(linked_variance.map(|value| value.0)),
            csv_option(Some(linked_post_mean)),
            csv_option(linked_variance.map(|value| value.1)),
            workspace.topology_summary.linked_current_true,
            csv_option(Some(linked_post_mean - workspace.topology_summary.linked_current_true)),
            csv_option(Some(projection_prior_mean)),
            csv_option(projection_variance.map(|value| value.0)),
            csv_option(Some(projection_post_mean)),
            csv_option(projection_variance.map(|value| value.1)),
            workspace.topology_summary.source_harmonic_projection_true,
            csv_option(Some(projection_post_mean - workspace.topology_summary.source_harmonic_projection_true))
        )?;
    }
    Ok(())
}

fn write_source_generated_objective_diagnostics_csv(
    config: &ToroidalHarmonicBConfig,
    workspace: &ExperimentWorkspace,
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,pde_metric_used,pde_mass_precision_scale,alpha_mean,alpha_error,state_prior_quadratic,alpha_prior_quadratic,pde_residual_euclidean_sq,pde_residual_mass_inv_sq,pde_mass_to_euclidean_ratio,pde_euclidean_likelihood_quadratic,pde_mass_raw_likelihood_quadratic,pde_mass_likelihood_quadratic,observation_chi2,hall_chi2,flux_chi2,ampere_loop_chi2,posterior_total_euclidean_metric,posterior_total_mass_metric,source_explanation_alpha_prior_quadratic,source_explanation_total,matching_scaled_mass_pde_variance_for_euclidean_penalty"
    )?;
    let state_prior = scale_gaussian_prior_precision(
        workspace.state_prior.clone(),
        config.fluctuation_state_prior_precision_scale,
    )
    .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    let source_alpha_prior_quadratic =
        ((config.source_alpha_true - 1.0) / config.source_prior_std).powi(2);
    let mass_precision_scale =
        effective_mass_weighted_pde_precision_scale(config, &workspace.system)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    for stage in stages
        .iter()
        .filter(|stage| is_source_generated_stage(&stage.summary.stage))
    {
        let alpha_mean = stage.summary.alpha_posterior_mean.unwrap_or(1.0);
        let alpha_error = alpha_mean - config.source_alpha_true;
        let state_prior_quadratic =
            gaussian_prior_quadratic(&state_prior, &stage.solve.reduced_posterior_mean)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        let alpha_prior_quadratic = ((alpha_mean - 1.0) / config.source_prior_std).powi(2);
        let pde_residual_euclidean_sq = stage
            .solve
            .pde_residual_mean
            .dot(&stage.solve.pde_residual_mean);
        let pde_residual_mass_inv_sq = sparse_quadratic(
            &csr_to_triplet(
                workspace
                    .system
                    .state_mass_inverse
                    .as_ref()
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "source-generated objective diagnostics require state_mass_inverse",
                        )
                    })?,
            ),
            &stage.solve.pde_residual_mean,
        )
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        let pde_mass_to_euclidean_ratio = if pde_residual_euclidean_sq > EPS {
            pde_residual_mass_inv_sq / pde_residual_euclidean_sq
        } else {
            f64::NAN
        };
        let pde_euclidean_likelihood_quadratic = pde_residual_euclidean_sq / config.pde_variance;
        let pde_mass_raw_likelihood_quadratic = pde_residual_mass_inv_sq / config.pde_variance;
        let pde_mass_likelihood_quadratic =
            mass_precision_scale * pde_mass_raw_likelihood_quadratic;
        let observation_chi2 = observation_chi2_for_stage(workspace, stage, None);
        let hall_chi2 = observation_chi2_for_stage(workspace, stage, Some("hall"));
        let flux_chi2 = observation_chi2_for_stage(workspace, stage, Some("flux"));
        let ampere_loop_chi2 =
            observation_chi2_for_stage(workspace, stage, Some(AMPERE_LOOP_SENSOR_TYPE));
        let posterior_total_euclidean_metric = state_prior_quadratic
            + alpha_prior_quadratic
            + pde_euclidean_likelihood_quadratic
            + observation_chi2;
        let posterior_total_mass_metric = state_prior_quadratic
            + alpha_prior_quadratic
            + pde_mass_likelihood_quadratic
            + observation_chi2;
        let matching_scaled_mass_pde_variance_for_euclidean_penalty =
            if pde_residual_euclidean_sq > EPS {
                config.pde_variance * mass_precision_scale * pde_residual_mass_inv_sq
                    / pde_residual_euclidean_sq
            } else {
                f64::NAN
            };
        let numeric_values = [
            mass_precision_scale,
            alpha_mean,
            alpha_error,
            state_prior_quadratic,
            alpha_prior_quadratic,
            pde_residual_euclidean_sq,
            pde_residual_mass_inv_sq,
            pde_mass_to_euclidean_ratio,
            pde_euclidean_likelihood_quadratic,
            pde_mass_raw_likelihood_quadratic,
            pde_mass_likelihood_quadratic,
            observation_chi2,
            hall_chi2,
            flux_chi2,
            ampere_loop_chi2,
            posterior_total_euclidean_metric,
            posterior_total_mass_metric,
            source_alpha_prior_quadratic,
            source_alpha_prior_quadratic,
            matching_scaled_mass_pde_variance_for_euclidean_penalty,
        ];
        let numeric_csv = numeric_values
            .iter()
            .map(|value| format!("{value:.16e}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            file,
            "{},{},{}",
            stage.summary.stage,
            if config.use_mass_weighted_pde_residual {
                if config.normalize_mass_weighted_pde_residual {
                    "mass_inverse_diag_normalized"
                } else {
                    "mass_inverse"
                }
            } else {
                "euclidean"
            },
            numeric_csv
        )?;
    }
    Ok(())
}

fn write_source_generated_field_errors_csv(
    workspace: &ExperimentWorkspace,
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,a_exact_relative_error,b_exact_relative_error,b_harmonic_relative_error,b_total_relative_error,harmonic_projection_exact_part,harmonic_projection_source_part,harmonic_projection_mean,harmonic_projection_truth,harmonic_projection_error"
    )?;
    let a_truth = workspace.truth_a_nominal.scale(
        workspace.topology_summary.linked_current_true
            / workspace.topology_summary.linked_current_unit,
    );
    let b_harmonic_truth = workspace.h2.scale(
        workspace.topology_summary.source_harmonic_kappa
            * workspace.topology_summary.linked_current_true
            / workspace.topology_summary.linked_current_unit,
    );
    for stage in stages
        .iter()
        .filter(|stage| is_source_generated_stage(&stage.summary.stage))
    {
        let Some(fields) = source_generated_mean_fields(workspace, stage) else {
            continue;
        };
        let harmonic_projection_exact_part =
            mass_inner_product(&workspace.h2, &fields.b_exact, &workspace.mass_2);
        let harmonic_projection_source_part = stage.summary.alpha_posterior_mean.unwrap_or(1.0)
            * workspace.topology_summary.source_harmonic_kappa;
        let harmonic_projection_mean =
            harmonic_projection_exact_part + harmonic_projection_source_part;
        writeln!(
            file,
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            stage.summary.stage,
            relative_euclidean_error(&fields.a_exact, &a_truth),
            relative_mass_error(
                &fields.b_exact,
                &workspace.truth_b_exact_alpha,
                &workspace.mass_2
            ),
            relative_mass_error(&fields.b_harmonic, &b_harmonic_truth, &workspace.mass_2),
            relative_mass_error(
                &fields.b_total,
                &workspace.truth_b_source_generated_alpha,
                &workspace.mass_2,
            ),
            harmonic_projection_exact_part,
            harmonic_projection_source_part,
            harmonic_projection_mean,
            workspace.topology_summary.source_harmonic_projection_true,
            harmonic_projection_mean - workspace.topology_summary.source_harmonic_projection_true
        )?;
    }
    Ok(())
}

fn write_observation_uncertainty_csv(
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,sensor_type,name,observed,prediction,residual,prior_variance,posterior_variance"
    )?;
    for stage in stages {
        for row in &stage.observations {
            writeln!(
                file,
                "{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
                row.stage,
                row.sensor_type,
                row.name,
                row.observed,
                row.prediction,
                row.residual,
                row.prior_variance,
                row.posterior_variance
            )?;
        }
    }
    Ok(())
}

fn write_ampere_loop_summary_csv(
    workspace: &ExperimentWorkspace,
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,name,observed,prediction,residual,nominal_loop_value,alpha_sensitivity,beta_sensitivity,prior_variance,posterior_variance,center_x,center_y,center_z,radius,segments"
    )?;
    let Some(observation) = workspace
        .observations
        .iter()
        .find(|observation| observation.sensor_type == AMPERE_LOOP_SENSOR_TYPE)
    else {
        return Ok(());
    };
    let nominal_loop_value =
        apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let geom = ToroidalInductorGeometry::default();
    let center = [geom.major_radius, 0.0, 0.0];
    for stage in stages {
        let Some(row) = stage
            .observations
            .iter()
            .find(|row| row.sensor_type == AMPERE_LOOP_SENSOR_TYPE)
        else {
            continue;
        };
        let alpha_sensitivity = if is_source_generated_full_state_stage(&stage.summary.stage) {
            workspace.topology_summary.source_harmonic_kappa * observation.beta_operator_value
        } else if is_source_generated_stage(&stage.summary.stage) {
            nominal_loop_value
                + workspace.topology_summary.source_harmonic_kappa * observation.beta_operator_value
        } else {
            nominal_loop_value
        };
        writeln!(
            file,
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            stage.summary.stage,
            row.name,
            row.observed,
            row.prediction,
            row.residual,
            nominal_loop_value,
            alpha_sensitivity,
            observation.beta_operator_value,
            row.prior_variance,
            row.posterior_variance,
            center[0],
            center[1],
            center[2],
            AMPERE_LOOP_RADIUS,
            AMPERE_LOOP_SEGMENTS
        )?;
    }
    Ok(())
}

fn write_pushforward_qoi_summary_csv(
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,qoi,role,truth,mean,sd,lower95,upper95,prior_variance,posterior_variance,variance_ratio,unit"
    )?;
    for row in stages.iter().flat_map(|stage| &stage.pushforward_qois) {
        writeln!(
            file,
            "{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            row.stage,
            row.qoi,
            row.role,
            row.truth,
            row.mean,
            row.sd,
            row.lower95,
            row.upper95,
            row.prior_variance,
            row.posterior_variance,
            row.variance_ratio,
            row.unit
        )?;
    }
    Ok(())
}

fn write_pushforward_qoi_covariance_csv(
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,qoi_i,qoi_j,prior_covariance,posterior_covariance,posterior_correlation"
    )?;
    for row in stages
        .iter()
        .flat_map(|stage| &stage.pushforward_covariances)
    {
        writeln!(
            file,
            "{},{},{},{:.16e},{:.16e},{:.16e}",
            row.stage,
            row.qoi_i,
            row.qoi_j,
            row.prior_covariance,
            row.posterior_covariance,
            row.posterior_correlation
        )?;
    }
    Ok(())
}

fn write_pushforward_variance_ratios_csv(
    stages: &[ToroidalStageResult],
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    let has_standard_topology_stages = stages
        .iter()
        .any(|stage| stage.summary.stage == TOPOLOGY_PUSHFORWARD_PDE_STAGE_NAME);
    if !has_standard_topology_stages {
        let reported_stages = stages
            .iter()
            .filter(|stage| !stage.pushforward_qois.is_empty())
            .collect::<Vec<_>>();
        write!(file, "qoi")?;
        for stage in &reported_stages {
            write!(file, ",{}", stage.summary.stage)?;
        }
        writeln!(file)?;
        for qoi in topology_pushforward_main_qoi_names() {
            write!(file, "{qoi}")?;
            for stage in &reported_stages {
                let ratio = stage
                    .pushforward_qois
                    .iter()
                    .find(|row| row.qoi == qoi)
                    .map(|row| row.variance_ratio);
                write!(file, ",{}", csv_option(ratio))?;
            }
            writeln!(file)?;
        }
        return Ok(());
    }
    writeln!(file, "qoi,S1_prior,S2_prior,S3_prior,S4_prior")?;
    for qoi in topology_pushforward_main_qoi_names() {
        let ratio = |stage_name: &str| {
            stages
                .iter()
                .find(|stage| stage.summary.stage == stage_name)
                .and_then(|stage| stage.pushforward_qois.iter().find(|row| row.qoi == qoi))
                .map(|row| row.variance_ratio)
        };
        writeln!(
            file,
            "{},{},{},{},{}",
            qoi,
            csv_option(ratio(TOPOLOGY_PUSHFORWARD_PDE_STAGE_NAME)),
            csv_option(ratio(TOPOLOGY_PUSHFORWARD_FIELD_STAGE_NAME)),
            csv_option(ratio(TOPOLOGY_PUSHFORWARD_FLUX_STAGE_NAME)),
            csv_option(ratio(TOPOLOGY_PUSHFORWARD_AMPERE_STAGE_NAME))
        )?;
    }
    Ok(())
}

fn write_heldout_prediction_csv(stages: &[ToroidalStageResult], path: &Path) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,sensor_type,name,truth,prediction,residual,posterior_sd,standardized_residual,lower95,upper95,covered95"
    )?;
    for row in stages.iter().flat_map(|stage| &stage.heldout_predictions) {
        writeln!(
            file,
            "{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            row.stage,
            row.sensor_type,
            row.name,
            row.truth,
            row.prediction,
            row.residual,
            row.posterior_sd,
            row.standardized_residual,
            row.lower95,
            row.upper95,
            row.covered95
        )?;
    }
    Ok(())
}

fn write_sparse_prediction_split_csv(
    workspace: &ExperimentWorkspace,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "role,sensor_type,name,truth,observed,noise,variance")?;
    for observation in &workspace.observations {
        writeln!(
            file,
            "training,{},{},{:.16e},{:.16e},{:.16e},{:.16e}",
            observation.sensor_type,
            observation.name,
            observation.observation_source_generated_truth,
            observation.observation_source_generated_observed,
            observation.observation_source_generated_observed
                - observation.observation_source_generated_truth,
            observation.variance_source_generated_truth
        )?;
    }
    for observation in &workspace.heldout_observations {
        writeln!(
            file,
            "heldout,{},{},{:.16e},{:.16e},{:.16e},{:.16e}",
            observation.sensor_type,
            observation.name,
            observation.observation_source_generated_truth,
            observation.observation_source_generated_observed,
            observation.observation_source_generated_observed
                - observation.observation_source_generated_truth,
            observation.variance_source_generated_truth
        )?;
    }
    Ok(())
}

fn write_branch_decomposition_csv(stages: &[ToroidalStageResult], path: &Path) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,functional,prior_exact_variance,prior_source_harmonic_variance,prior_coupling_variance,prior_total_variance,prior_reported_variance,posterior_exact_variance,posterior_source_harmonic_variance,posterior_coupling_variance,posterior_total_variance,posterior_reported_variance"
    )?;
    for row in stages.iter().flat_map(|stage| &stage.branch_decomposition) {
        writeln!(
            file,
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            row.stage,
            row.functional,
            row.prior_exact_variance,
            row.prior_source_harmonic_variance,
            row.prior_coupling_variance,
            row.prior_total_variance,
            row.prior_reported_variance,
            row.posterior_exact_variance,
            row.posterior_source_harmonic_variance,
            row.posterior_coupling_variance,
            row.posterior_total_variance,
            row.posterior_reported_variance
        )?;
    }
    Ok(())
}

fn write_field_trace_variance_csv(stages: &[ToroidalStageResult], path: &Path) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,point,prior_trace_variance,posterior_trace_variance,variance_ratio"
    )?;
    for row in stages.iter().flat_map(|stage| &stage.field_trace_variance) {
        writeln!(
            file,
            "{},{},{:.16e},{:.16e},{:.16e}",
            row.stage,
            row.point,
            row.prior_trace_variance,
            row.posterior_trace_variance,
            row.variance_ratio
        )?;
    }
    Ok(())
}

fn write_gp_baseline_outputs(result: &ToroidalGpBaselineResult, out_dir: &Path) -> io::Result<()> {
    write_gp_baseline_summary_csv(result, &out_dir.join("gp_baseline_summary.csv"))?;
    write_gp_baseline_heldout_prediction_csv(
        result,
        &out_dir.join("gp_baseline_heldout_prediction.csv"),
    )?;
    write_gp_baseline_qoi_summary_csv(result, &out_dir.join("gp_baseline_qoi_summary.csv"))?;
    Ok(())
}

fn write_gp_baseline_summary_csv(result: &ToroidalGpBaselineResult, path: &Path) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,matched_feec_stage,kernel,matern_nu,length_scale,signal_variance,log_marginal_likelihood,training_rows,hall_training_rows,flux_training_rows,ampere_training_rows,heldout_rows,heldout_rmse,heldout_nlpd,heldout_covered95,heldout_coverage_fraction,heldout_max_abs_standardized_residual"
    )?;
    for stage in &result.stages {
        let row = &stage.summary;
        writeln!(
            file,
            "{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e}",
            row.stage,
            row.matched_feec_stage,
            row.kernel,
            row.matern_nu,
            row.length_scale,
            row.signal_variance,
            row.log_marginal_likelihood,
            row.training_rows,
            row.hall_training_rows,
            row.flux_training_rows,
            row.ampere_training_rows,
            row.heldout_rows,
            row.heldout_rmse,
            row.heldout_nlpd,
            row.heldout_covered95,
            row.heldout_coverage_fraction,
            row.heldout_max_abs_standardized_residual
        )?;
    }
    Ok(())
}

fn write_gp_baseline_heldout_prediction_csv(
    result: &ToroidalGpBaselineResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,sensor_type,name,truth,prediction,residual,posterior_sd,standardized_residual,lower95,upper95,covered95"
    )?;
    for row in result
        .stages
        .iter()
        .flat_map(|stage| &stage.heldout_predictions)
    {
        writeln!(
            file,
            "{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            row.stage,
            row.sensor_type,
            row.name,
            row.truth,
            row.prediction,
            row.residual,
            row.posterior_sd,
            row.standardized_residual,
            row.lower95,
            row.upper95,
            row.covered95
        )?;
    }
    Ok(())
}

fn write_gp_baseline_qoi_summary_csv(
    result: &ToroidalGpBaselineResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,qoi,role,truth,mean,sd,lower95,upper95,abs_error,unit,source_posterior_available,topology_posterior_available"
    )?;
    for row in result.stages.iter().flat_map(|stage| &stage.qois) {
        writeln!(
            file,
            "{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},false,false",
            row.stage,
            row.qoi,
            row.role,
            row.truth,
            row.mean,
            row.sd,
            row.lower95,
            row.upper95,
            row.abs_error,
            row.unit
        )?;
    }
    Ok(())
}

fn write_gp_baseline_comparison_csv(
    result: &ToroidalGpBaselineResult,
    feec_out_dir: &Path,
    path: &Path,
) -> io::Result<()> {
    let feec_metrics = read_feec_heldout_metrics(&feec_out_dir.join("heldout_prediction.csv"))?;
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,model,matched_stage,training_rows,heldout_rows,heldout_rmse,heldout_nlpd,heldout_covered95,heldout_coverage_fraction,heldout_max_abs_standardized_residual,source_posterior_available,topology_posterior_available"
    )?;
    for stage in &result.stages {
        if let Some(feec) = feec_metrics.get(&stage.summary.matched_feec_stage) {
            writeln!(
                file,
                "{},FEEC-GMRF,{},,{},{:.16e},{:.16e},{},{:.16e},{:.16e},true,true",
                stage.summary.matched_feec_stage,
                stage.summary.matched_feec_stage,
                feec.rows,
                feec.rmse,
                feec.nlpd,
                feec.covered95,
                feec.coverage_fraction,
                feec.max_abs_standardized_residual
            )?;
        }
        writeln!(
            file,
            "{},independent-output GP,{},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e},false,false",
            stage.summary.stage,
            stage.summary.matched_feec_stage,
            stage.summary.training_rows,
            stage.summary.heldout_rows,
            stage.summary.heldout_rmse,
            stage.summary.heldout_nlpd,
            stage.summary.heldout_covered95,
            stage.summary.heldout_coverage_fraction,
            stage.summary.heldout_max_abs_standardized_residual
        )?;
    }
    Ok(())
}

fn write_sparse_gp_comparison_outputs(
    result: &ToroidalSparseGpComparisonResult,
    out_dir: &Path,
) -> io::Result<()> {
    write_sparse_gp_comparison_metric_csv(
        result
            .metrics
            .iter()
            .filter(|row| row.sensor_family == "all"),
        &out_dir.join("sparse_gp_comparison_summary.csv"),
    )?;
    write_sparse_gp_comparison_metric_csv(
        result.metrics.iter(),
        &out_dir.join("sparse_gp_comparison_by_sensor.csv"),
    )?;
    write_sparse_gp_heldout_prediction_csv(
        result,
        &out_dir.join("sparse_gp_heldout_prediction.csv"),
    )?;
    write_sparse_feec_qoi_summary_csv(result, &out_dir.join("sparse_feec_qoi_summary.csv"))?;
    write_sparse_gp_qoi_summary_csv(result, &out_dir.join("sparse_gp_qoi_summary.csv"))?;
    Ok(())
}

fn write_sparse_gp_comparison_metric_csv<'a>(
    rows: impl IntoIterator<Item = &'a ToroidalSparseComparisonMetricRow>,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,sensor_family,training_rows,heldout_rows,rmse,nlpd,covered95,coverage_fraction,max_abs_standardized_residual,mean_abs_standardized_residual,source_posterior_available,topology_posterior_available,pde_residual_used"
    )?;
    for row in rows {
        writeln!(
            file,
            "{},{},{},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{},{},{}",
            row.model,
            row.stage,
            row.sensor_family,
            row.training_rows,
            row.heldout_rows,
            row.rmse,
            row.nlpd,
            row.covered95,
            row.coverage_fraction,
            row.max_abs_standardized_residual,
            row.mean_abs_standardized_residual,
            row.source_posterior_available,
            row.topology_posterior_available,
            row.pde_residual_used
        )?;
    }
    Ok(())
}

fn write_sparse_gp_heldout_prediction_csv(
    result: &ToroidalSparseGpComparisonResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,pde_residual_used,sensor_family,sensor_type,name,truth,prediction,residual,posterior_sd,standardized_residual,lower95,upper95,covered95"
    )?;
    for (model, pde, stage) in result
        .feec_observation_only_stages
        .iter()
        .map(|stage| ("FEEC-GMRF observation-only", false, stage))
        .chain(
            result
                .feec_full_stages
                .iter()
                .map(|stage| ("FEEC-GMRF full", true, stage)),
        )
    {
        for row in &stage.heldout_predictions {
            writeln!(
                file,
                "{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
                model,
                row.stage,
                pde,
                sensor_family(&row.sensor_type),
                row.sensor_type,
                row.name,
                row.truth,
                row.prediction,
                row.residual,
                row.posterior_sd,
                row.standardized_residual,
                row.lower95,
                row.upper95,
                row.covered95
            )?;
        }
    }
    for stage in &result.gp_stages {
        for row in &stage.heldout_predictions {
            writeln!(
                file,
                "independent-output GP,{},false,{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
                row.stage,
                sensor_family(&row.sensor_type),
                row.sensor_type,
                row.name,
                row.truth,
                row.prediction,
                row.residual,
                row.posterior_sd,
                row.standardized_residual,
                row.lower95,
                row.upper95,
                row.covered95
            )?;
        }
    }
    Ok(())
}

fn write_sparse_feec_qoi_summary_csv(
    result: &ToroidalSparseGpComparisonResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,pde_residual_used,qoi,role,truth,mean,sd,lower95,upper95,prior_variance,posterior_variance,variance_ratio,unit"
    )?;
    for (model, pde, stage) in result
        .feec_observation_only_stages
        .iter()
        .map(|stage| ("FEEC-GMRF observation-only", false, stage))
        .chain(
            result
                .feec_full_stages
                .iter()
                .map(|stage| ("FEEC-GMRF full", true, stage)),
        )
    {
        for row in &stage.pushforward_qois {
            writeln!(
                file,
                "{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
                model,
                row.stage,
                pde,
                row.qoi,
                row.role,
                row.truth,
                row.mean,
                row.sd,
                row.lower95,
                row.upper95,
                row.prior_variance,
                row.posterior_variance,
                row.variance_ratio,
                row.unit
            )?;
        }
    }
    Ok(())
}

fn write_sparse_gp_qoi_summary_csv(
    result: &ToroidalSparseGpComparisonResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,qoi,role,truth,mean,sd,lower95,upper95,abs_error,unit,source_posterior_available,topology_posterior_available"
    )?;
    for stage in &result.gp_stages {
        for row in &stage.qois {
            writeln!(
                file,
                "independent-output GP,{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},false,false",
                row.stage,
                row.qoi,
                row.role,
                row.truth,
                row.mean,
                row.sd,
                row.lower95,
                row.upper95,
                row.abs_error,
                row.unit
            )?;
        }
    }
    Ok(())
}

fn write_source_template_gp_outputs(
    result: &ToroidalSourceTemplateGpResult,
    out_dir: &Path,
) -> io::Result<()> {
    write_source_template_gp_summary_csv(result, &out_dir.join("source_template_gp_summary.csv"))?;
    write_source_template_gp_heldout_prediction_csv(
        result,
        &out_dir.join("source_template_gp_heldout_prediction.csv"),
    )?;
    write_source_template_gp_qoi_summary_csv(
        result,
        &out_dir.join("source_template_gp_qoi_summary.csv"),
    )?;
    write_source_template_gp_comparison_csv(
        result,
        &out_dir.join("source_template_gp_comparison.csv"),
    )?;
    Ok(())
}

fn write_source_template_gp_summary_csv(
    result: &ToroidalSourceTemplateGpResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,template_kind,sensor_family,kernel,matern_nu,length_scale,signal_variance,log_marginal_likelihood,training_rows,hall_training_rows,flux_training_rows,ampere_training_rows,heldout_rows,heldout_rmse,heldout_nlpd,heldout_covered95,heldout_coverage_fraction,heldout_max_abs_standardized_residual,source_prior_mean,source_prior_variance,source_posterior_mean,source_posterior_variance,source_truth,source_abs_error"
    )?;
    for stage in &result.stages {
        let row = &stage.summary;
        writeln!(
            file,
            "{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            row.model,
            row.stage,
            row.template_kind,
            row.sensor_family,
            row.kernel,
            row.matern_nu,
            row.length_scale,
            row.signal_variance,
            row.log_marginal_likelihood,
            row.training_rows,
            row.hall_training_rows,
            row.flux_training_rows,
            row.ampere_training_rows,
            row.heldout_rows,
            row.heldout_rmse,
            row.heldout_nlpd,
            row.heldout_covered95,
            row.heldout_coverage_fraction,
            row.heldout_max_abs_standardized_residual,
            row.source_prior_mean,
            row.source_prior_variance,
            row.source_posterior_mean,
            row.source_posterior_variance,
            row.source_truth,
            row.source_abs_error
        )?;
    }
    Ok(())
}

fn write_source_template_gp_heldout_prediction_csv(
    result: &ToroidalSourceTemplateGpResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,template_kind,sensor_family,sensor_type,name,truth,prediction,residual,posterior_sd,standardized_residual,lower95,upper95,covered95"
    )?;
    for stage in &result.stages {
        for row in &stage.heldout_predictions {
            writeln!(
                file,
                "{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
                stage.summary.model,
                row.stage,
                stage.summary.template_kind,
                sensor_family(&row.sensor_type),
                row.sensor_type,
                row.name,
                row.truth,
                row.prediction,
                row.residual,
                row.posterior_sd,
                row.standardized_residual,
                row.lower95,
                row.upper95,
                row.covered95
            )?;
        }
    }
    Ok(())
}

fn write_source_template_gp_qoi_summary_csv(
    result: &ToroidalSourceTemplateGpResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,template_kind,qoi,role,truth,mean,sd,lower95,upper95,abs_error,unit,source_posterior_available,topology_posterior_available"
    )?;
    for stage in &result.stages {
        let topology_available =
            stage.summary.template_kind == SourceTemplateKind::TopologyOracle.label();
        for row in &stage.qois {
            writeln!(
                file,
                "{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},true,{}",
                stage.summary.model,
                row.stage,
                stage.summary.template_kind,
                row.qoi,
                row.role,
                row.truth,
                row.mean,
                row.sd,
                row.lower95,
                row.upper95,
                row.abs_error,
                row.unit,
                topology_available
            )?;
        }
    }
    Ok(())
}

fn write_source_template_gp_comparison_csv(
    result: &ToroidalSourceTemplateGpResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "model,stage,sensor_family,training_rows,heldout_rows,rmse,nlpd,covered95,coverage_fraction,max_abs_standardized_residual,mean_abs_standardized_residual,source_posterior_available,topology_posterior_available,pde_residual_used"
    )?;
    let sparse_summary_path = Path::new(
        "out/examples/toroidal_topology_sparse_noisy_gp_comparison/sparse_gp_comparison_summary.csv",
    );
    if sparse_summary_path.exists() {
        let text = fs::read_to_string(sparse_summary_path)?;
        for line in text.lines().skip(1) {
            if !line.trim().is_empty() {
                writeln!(file, "{line}")?;
            }
        }
    }
    for row in result
        .metrics
        .iter()
        .filter(|row| row.sensor_family == "all")
    {
        writeln!(
            file,
            "{},{},{},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{},{},{}",
            row.model,
            row.stage,
            row.sensor_family,
            row.training_rows,
            row.heldout_rows,
            row.rmse,
            row.nlpd,
            row.covered95,
            row.coverage_fraction,
            row.max_abs_standardized_residual,
            row.mean_abs_standardized_residual,
            row.source_posterior_available,
            row.topology_posterior_available,
            row.pde_residual_used
        )?;
    }
    Ok(())
}

fn write_coupling_calibration_outputs(
    result: &ToroidalHarmonicCouplingCalibrationResult,
    config: &ToroidalHarmonicCouplingCalibrationConfig,
    coupling_prior_std: f64,
    json_path: &Path,
    out_dir: &Path,
) -> io::Result<()> {
    write_coupling_calibration_json(result, config, coupling_prior_std, json_path)?;
    write_coupling_calibration_summary_csv(
        result,
        &out_dir.join("coupling_calibration_summary.csv"),
    )?;
    write_coupling_pushforward_qoi_summary_csv(
        result,
        &out_dir.join("coupling_pushforward_qoi_summary.csv"),
    )?;
    write_coupling_observation_rows_csv(
        result
            .stages
            .iter()
            .flat_map(|stage| &stage.heldout_predictions),
        &out_dir.join("coupling_heldout_prediction.csv"),
    )?;
    write_coupling_observation_rows_csv(
        result.stages.iter().flat_map(|stage| &stage.observations),
        &out_dir.join("coupling_observation_uncertainty.csv"),
    )?;
    write_coupling_covariance_csv(result, &out_dir.join("coupling_covariance.csv"))?;
    Ok(())
}

fn write_coupling_calibration_json(
    result: &ToroidalHarmonicCouplingCalibrationResult,
    config: &ToroidalHarmonicCouplingCalibrationConfig,
    coupling_prior_std: f64,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(
        file,
        "  \"betti_numbers\": [{},{},{},{}],",
        result.topology_summary.betti_numbers[0],
        result.topology_summary.betti_numbers[1],
        result.topology_summary.betti_numbers[2],
        result.topology_summary.betti_numbers[3]
    )?;
    writeln!(
        file,
        "  \"harmonic_2_dimension\": {},",
        result.topology_summary.harmonic_2_dimension
    )?;
    writeln!(
        file,
        "  \"drive_currents\": [{}],",
        result
            .drive_currents
            .iter()
            .map(|value| format!("{value:.16e}"))
            .collect::<Vec<_>>()
            .join(",")
    )?;
    writeln!(
        file,
        "  \"coupling_truth\": {:.16e},",
        result.topology_summary.source_harmonic_kappa
    )?;
    writeln!(file, "  \"coupling_prior_mean\": 0.0000000000000000e0,")?;
    writeln!(
        file,
        "  \"coupling_prior_std\": {:.16e},",
        coupling_prior_std
    )?;
    writeln!(
        file,
        "  \"coupling_prior_std_scale\": {:.16e},",
        config.coupling_prior_std_scale
    )?;
    writeln!(
        file,
        "  \"relative_noise_std\": {:.16e},",
        config.toroidal.relative_noise_std
    )?;
    writeln!(
        file,
        "  \"noise_floor\": {:.16e},",
        config.toroidal.noise_floor
    )?;
    writeln!(
        file,
        "  \"pde_variance\": {:.16e},",
        config.toroidal.pde_variance
    )?;
    writeln!(
        file,
        "  \"use_mass_weighted_pde_residual\": {},",
        config.toroidal.use_mass_weighted_pde_residual
    )?;
    writeln!(
        file,
        "  \"normalize_mass_weighted_pde_residual\": {},",
        config.toroidal.normalize_mass_weighted_pde_residual
    )?;
    if let Some(last) = result.stages.last() {
        writeln!(
            file,
            "  \"posterior_precision_nnz_final\": {},",
            last.summary.posterior_precision_nnz
        )?;
        writeln!(
            file,
            "  \"posterior_factor_nnz_final\": {},",
            last.summary.posterior_factor_nnz
        )?;
        writeln!(
            file,
            "  \"posterior_fill_in_final\": {:.16e},",
            last.summary.posterior_fill_in
        )?;
        writeln!(
            file,
            "  \"posterior_factor_mib_final\": {:.16e}",
            last.summary.posterior_factor_mib
        )?;
    } else {
        writeln!(file, "  \"posterior_precision_nnz_final\": 0")?;
    }
    writeln!(file, "}}")?;
    Ok(())
}

fn write_coupling_calibration_summary_csv(
    result: &ToroidalHarmonicCouplingCalibrationResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,training_rows,hall_training_rows,flux_training_rows,ampere_training_rows,heldout_rows,heldout_rmse,heldout_nlpd,heldout_covered95,heldout_coverage_fraction,heldout_max_abs_standardized_residual,coupling_prior_mean,coupling_prior_variance,coupling_posterior_mean,coupling_posterior_variance,coupling_truth,coupling_abs_error,coupling_variance_ratio,posterior_precision_nnz,posterior_factor_nnz,posterior_fill_in,posterior_factor_mib"
    )?;
    for row in result.stages.iter().map(|stage| &stage.summary) {
        writeln!(
            file,
            "{},{},{},{},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{:.16e}",
            row.stage,
            row.training_rows,
            row.hall_training_rows,
            row.flux_training_rows,
            row.ampere_training_rows,
            row.heldout_rows,
            row.heldout_rmse,
            row.heldout_nlpd,
            row.heldout_covered95,
            row.heldout_coverage_fraction,
            row.heldout_max_abs_standardized_residual,
            row.coupling_prior_mean,
            row.coupling_prior_variance,
            row.coupling_posterior_mean,
            row.coupling_posterior_variance,
            row.coupling_truth,
            row.coupling_abs_error,
            row.coupling_variance_ratio,
            row.posterior_precision_nnz,
            row.posterior_factor_nnz,
            row.posterior_fill_in,
            row.posterior_factor_mib
        )?;
    }
    Ok(())
}

fn write_coupling_pushforward_qoi_summary_csv(
    result: &ToroidalHarmonicCouplingCalibrationResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,drive_index,drive_current,qoi,role,truth,mean,sd,lower95,upper95,prior_variance,posterior_variance,variance_ratio,unit"
    )?;
    for row in result.stages.iter().flat_map(|stage| &stage.qois) {
        writeln!(
            file,
            "{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            row.stage,
            row.drive_index
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.drive_current
                .map(|value| format!("{value:.16e}"))
                .unwrap_or_default(),
            row.qoi,
            row.role,
            row.truth,
            row.mean,
            row.sd,
            row.lower95,
            row.upper95,
            row.prior_variance,
            row.posterior_variance,
            row.variance_ratio,
            row.unit
        )?;
    }
    Ok(())
}

fn write_coupling_observation_rows_csv<'a>(
    rows: impl IntoIterator<Item = &'a ToroidalCouplingCalibrationObservationRow>,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,drive_index,drive_current,sensor_family,sensor_type,name,truth,observed,prediction,residual,posterior_sd,standardized_residual,lower95,upper95,covered95"
    )?;
    for row in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}",
            row.stage,
            row.drive_index,
            row.drive_current,
            sensor_family(&row.sensor_type),
            row.sensor_type,
            row.name,
            row.truth,
            row.observed,
            row.prediction,
            row.residual,
            row.posterior_sd,
            row.standardized_residual,
            row.lower95,
            row.upper95,
            row.covered95
        )?;
    }
    Ok(())
}

fn write_coupling_covariance_csv(
    result: &ToroidalHarmonicCouplingCalibrationResult,
    path: &Path,
) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "stage,qoi_i,qoi_j,prior_covariance,posterior_covariance,posterior_correlation"
    )?;
    for row in result.stages.iter().flat_map(|stage| &stage.covariances) {
        writeln!(
            file,
            "{},{},{},{:.16e},{:.16e},{:.16e}",
            row.stage,
            row.qoi_i,
            row.qoi_j,
            row.prior_covariance,
            row.posterior_covariance,
            row.posterior_correlation
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct HeldoutMetricSummary {
    rows: usize,
    rmse: f64,
    nlpd: f64,
    covered95: usize,
    coverage_fraction: f64,
    max_abs_standardized_residual: f64,
}

fn read_feec_heldout_metrics(path: &Path) -> io::Result<BTreeMap<String, HeldoutMetricSummary>> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let text = fs::read_to_string(path)?;
    let mut by_stage = BTreeMap::<String, Vec<ToroidalGpBaselinePredictionRow>>::new();
    for line in text.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() < 11 {
            continue;
        }
        let stage = fields[0].to_string();
        let row = ToroidalGpBaselinePredictionRow {
            stage: stage.clone(),
            sensor_type: fields[1].to_string(),
            name: fields[2].to_string(),
            truth: fields[3].parse().unwrap_or(f64::NAN),
            prediction: fields[4].parse().unwrap_or(f64::NAN),
            residual: fields[5].parse().unwrap_or(f64::NAN),
            posterior_sd: fields[6].parse().unwrap_or(f64::NAN),
            standardized_residual: fields[7].parse().unwrap_or(f64::NAN),
            lower95: fields[8].parse().unwrap_or(f64::NAN),
            upper95: fields[9].parse().unwrap_or(f64::NAN),
            covered95: fields[10].parse().unwrap_or(false),
        };
        by_stage.entry(stage).or_default().push(row);
    }
    Ok(by_stage
        .into_iter()
        .map(|(stage, rows)| {
            let rmse = prediction_rmse(
                rows.iter()
                    .map(|row| row.residual)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            let nlpd = prediction_nlpd(&rows);
            let covered95 = rows.iter().filter(|row| row.covered95).count();
            let max_abs_standardized_residual = rows
                .iter()
                .map(|row| row.standardized_residual.abs())
                .fold(0.0_f64, f64::max);
            let summary = HeldoutMetricSummary {
                rows: rows.len(),
                rmse,
                nlpd,
                covered95,
                coverage_fraction: if rows.is_empty() {
                    f64::NAN
                } else {
                    covered95 as f64 / rows.len() as f64
                },
                max_abs_standardized_residual,
            };
            (stage, summary)
        })
        .collect())
}

fn write_truth_vtus(workspace: &ExperimentWorkspace, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let truth_a_nominal = Cochain::new(1, workspace.truth_a_nominal.clone());
    let truth_a_alpha = Cochain::new(1, workspace.truth_a_alpha.clone());
    let truth_b_exact_nominal = Cochain::new(2, workspace.truth_b_exact_nominal.clone());
    let truth_b_exact_alpha = Cochain::new(2, workspace.truth_b_exact_alpha.clone());
    let truth_b_harmonic =
        Cochain::new(2, workspace.h2.scale(workspace.topology_summary.beta_true));
    let truth_b_total_nominal = Cochain::new(2, workspace.truth_b_total_nominal.clone());
    let truth_b_total_alpha = Cochain::new(2, workspace.truth_b_total_alpha.clone());
    let truth_b_source_generated_unit =
        Cochain::new(2, workspace.truth_b_source_generated_unit.clone());
    let truth_b_source_generated_alpha =
        Cochain::new(2, workspace.truth_b_source_generated_alpha.clone());

    visual_output::write_1cochain_fields(
        out_dir.join("truth_A_fields.vtu"),
        &workspace.coords,
        &workspace.topology,
        &[
            ("A_det_nominal", &truth_a_nominal),
            ("A_det_alpha_true", &truth_a_alpha),
        ],
    )?;
    visual_output::write_2form_vector_field(
        out_dir.join("truth_B_exact_nominal_vector.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_b_exact_nominal,
        "B_exact_nominal",
    )?;
    visual_output::write_2form_vector_field(
        out_dir.join("truth_B_exact_alpha_vector.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_b_exact_alpha,
        "B_exact_alpha",
    )?;
    visual_output::write_2form_vector_field(
        out_dir.join("truth_B_harmonic_vector.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_b_harmonic,
        "B_harmonic",
    )?;
    visual_output::write_2form_vector_field(
        out_dir.join("truth_B_total_nominal_vector.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_b_total_nominal,
        "B_total_nominal",
    )?;
    visual_output::write_2form_vector_field(
        out_dir.join("truth_B_total_alpha_vector.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_b_total_alpha,
        "B_total_alpha",
    )?;
    visual_output::write_2form_vector_field(
        out_dir.join("truth_B_source_generated_unit_vector.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_b_source_generated_unit,
        "B_source_generated_unit",
    )?;
    visual_output::write_2form_vector_field(
        out_dir.join("truth_B_source_generated_alpha_vector.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_b_source_generated_alpha,
        "B_source_generated_alpha",
    )?;
    Ok(())
}

fn write_stage_vtus(
    workspace: &ExperimentWorkspace,
    stage: &ToroidalStageResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let stage_dir = out_dir.join(&stage.summary.stage);
    fs::create_dir_all(&stage_dir)?;

    let mut a_mean_coeffs = stage.solve.posterior_mean.clone();
    if is_fluctuation_alpha_beta_stage(&stage.summary.stage)
        || is_source_generated_fluctuation_stage(&stage.summary.stage)
    {
        let alpha_mean = stage.summary.alpha_posterior_mean.unwrap_or(0.0);
        a_mean_coeffs += workspace.truth_a_nominal.scale(alpha_mean);
    }
    let a_mean = Cochain::new(1, a_mean_coeffs);
    let b_exact_mean = a_mean.dif(&workspace.topology);
    let beta_mean = if is_source_generated_stage(&stage.summary.stage) {
        stage.summary.alpha_posterior_mean.unwrap_or(0.0)
            * workspace.topology_summary.source_harmonic_kappa
    } else {
        stage.summary.beta_posterior_mean.unwrap_or(0.0)
    };
    let b_harmonic_mean = Cochain::new(2, workspace.h2.scale(beta_mean));
    let b_total_mean = Cochain::new(2, &b_exact_mean.coeffs + &b_harmonic_mean.coeffs);
    let truth_total = if is_source_generated_stage(&stage.summary.stage) {
        Cochain::new(2, workspace.truth_b_source_generated_alpha.clone())
    } else if stage_uses_alpha_truth(&stage.summary.stage) {
        Cochain::new(2, workspace.truth_b_total_alpha.clone())
    } else {
        Cochain::new(2, workspace.truth_b_total_nominal.clone())
    };
    let absolute_error = Cochain::new(
        2,
        FeecVector::from_iterator(
            truth_total.len(),
            (0..truth_total.len())
                .map(|idx| (b_total_mean.coeffs[idx] - truth_total.coeffs[idx]).abs()),
        ),
    );

    visual_output::write_1cochain_fields(
        stage_dir.join("A_mean.vtu"),
        &workspace.coords,
        &workspace.topology,
        &[("A_mean", &a_mean)],
    )?;
    visual_output::write_2form_vector_field(
        stage_dir.join("B_exact_mean.vtu"),
        &workspace.coords,
        &workspace.topology,
        &b_exact_mean,
        "B_exact_mean",
    )?;
    visual_output::write_2form_vector_field(
        stage_dir.join("B_harmonic_mean.vtu"),
        &workspace.coords,
        &workspace.topology,
        &b_harmonic_mean,
        "B_harmonic_mean",
    )?;
    visual_output::write_2form_vector_field(
        stage_dir.join("B_total_mean.vtu"),
        &workspace.coords,
        &workspace.topology,
        &b_total_mean,
        "B_total_mean",
    )?;
    visual_output::write_2form_vector_field(
        stage_dir.join("B_total_truth.vtu"),
        &workspace.coords,
        &workspace.topology,
        &truth_total,
        "B_total_truth",
    )?;
    visual_output::write_cochain(
        stage_dir.join("B_total_absolute_error.vtu"),
        &workspace.coords,
        &workspace.topology,
        &absolute_error,
        "B_total_absolute_error",
    )?;

    if let Some(variance) = stage.solve.derived_variances.get(B_TOTAL_DERIVED_NAME) {
        let prior_variance = Cochain::new(2, variance.prior_variance.clone());
        let posterior_variance = Cochain::new(2, variance.posterior_variance.clone());
        let variance_ratio = Cochain::new(
            2,
            pointwise_variance_ratio(&variance.prior_variance, &variance.posterior_variance),
        );
        visual_output::write_cochain(
            stage_dir.join("B_total_prior_variance.vtu"),
            &workspace.coords,
            &workspace.topology,
            &prior_variance,
            "B_total_prior_variance",
        )?;
        visual_output::write_cochain(
            stage_dir.join("B_total_posterior_variance.vtu"),
            &workspace.coords,
            &workspace.topology,
            &posterior_variance,
            "B_total_posterior_variance",
        )?;
        visual_output::write_cochain(
            stage_dir.join("B_total_variance_ratio.vtu"),
            &workspace.coords,
            &workspace.topology,
            &variance_ratio,
            "B_total_variance_ratio",
        )?;
    }

    Ok(())
}

struct SourceGeneratedMeanFields {
    a_exact: FeecVector,
    b_exact: FeecVector,
    b_harmonic: FeecVector,
    b_total: FeecVector,
}

fn source_generated_mean_fields(
    workspace: &ExperimentWorkspace,
    stage: &ToroidalStageResult,
) -> Option<SourceGeneratedMeanFields> {
    let alpha_mean = stage.summary.alpha_posterior_mean?;
    let mut a_exact = stage.solve.posterior_mean.clone();
    if is_source_generated_fluctuation_stage(&stage.summary.stage) {
        a_exact += workspace.truth_a_nominal.scale(alpha_mean);
    }
    let b_exact = Cochain::new(1, a_exact.clone())
        .dif(&workspace.topology)
        .coeffs;
    let b_harmonic = workspace
        .h2
        .scale(alpha_mean * workspace.topology_summary.source_harmonic_kappa);
    let b_total = &b_exact + &b_harmonic;
    Some(SourceGeneratedMeanFields {
        a_exact,
        b_exact,
        b_harmonic,
        b_total,
    })
}

fn source_generated_harmonic_projection_mean(
    workspace: &ExperimentWorkspace,
    stage: &ToroidalStageResult,
) -> Option<f64> {
    let fields = source_generated_mean_fields(workspace, stage)?;
    Some(mass_inner_product(
        &workspace.h2,
        &fields.b_total,
        &workspace.mass_2,
    ))
}

fn source_generated_linked_current_mean(
    workspace: &ExperimentWorkspace,
    stage: &ToroidalStageResult,
) -> Option<f64> {
    let alpha_mean = stage.summary.alpha_posterior_mean?;
    if is_source_generated_full_state_stage(&stage.summary.stage) {
        let state_loop = apply_triplet_row(
            &workspace.ampere_loop_state_operator,
            &stage.solve.posterior_mean,
        )
        .ok()?;
        Some(
            state_loop
                + alpha_mean
                    * workspace.topology_summary.source_harmonic_kappa
                    * workspace.topology_summary.ampere_loop_harmonic_sensitivity,
        )
    } else {
        Some(workspace.topology_summary.linked_current_unit * alpha_mean)
    }
}

fn derived_scalar_variances(stage: &ToroidalStageResult, name: &str) -> Option<(f64, f64)> {
    stage
        .solve
        .derived_variances
        .get(name)
        .map(|variance| (variance.prior_variance[0], variance.posterior_variance[0]))
}

fn relative_euclidean_error(estimate: &FeecVector, truth: &FeecVector) -> f64 {
    if estimate.len() != truth.len() {
        return f64::NAN;
    }
    let diff = estimate - truth;
    diff.norm() / truth.norm().max(EPS)
}

fn relative_mass_error(estimate: &FeecVector, truth: &FeecVector, mass: &FeecCsr) -> f64 {
    if estimate.len() != truth.len() {
        return f64::NAN;
    }
    let diff = estimate - truth;
    mass_inner_product(&diff, &diff, mass).max(0.0).sqrt()
        / mass_inner_product(truth, truth, mass)
            .max(0.0)
            .sqrt()
            .max(EPS)
}

fn csv_option(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.16e}"))
        .unwrap_or_default()
}

fn stage_uses_alpha_truth(stage: &str) -> bool {
    stage == "H4_joint_alpha_beta_stress"
        || stage == EMBEDDED_AMPERE_ALPHA_BETA_STAGE_NAME
        || is_source_generated_stage(stage)
        || is_fluctuation_alpha_beta_stage(stage)
}

fn is_fluctuation_alpha_beta_stage(stage: &str) -> bool {
    stage == FLUCTUATION_ALPHA_BETA_STAGE_NAME || stage == AMPERE_LOOP_ALPHA_BETA_STAGE_NAME
}

fn is_source_generated_stage(stage: &str) -> bool {
    is_source_generated_fluctuation_stage(stage)
        || is_source_generated_full_state_stage(stage)
        || is_topology_pushforward_stage_name(stage)
}

fn is_source_generated_fluctuation_stage(stage: &str) -> bool {
    stage == SOURCE_GENERATED_PDE_STAGE_NAME
        || stage == SOURCE_GENERATED_FIELD_STAGE_NAME
        || stage == SOURCE_GENERATED_AMPERE_STAGE_NAME
}

fn is_source_generated_full_state_stage(stage: &str) -> bool {
    stage == SOURCE_GENERATED_FULL_STATE_PDE_STAGE_NAME
        || stage == SOURCE_GENERATED_FULL_STATE_FIELD_STAGE_NAME
        || stage == SOURCE_GENERATED_FULL_STATE_AMPERE_STAGE_NAME
        || is_topology_pushforward_stage_name(stage)
}

fn is_topology_pushforward_stage_name(stage: &str) -> bool {
    stage == TOPOLOGY_PUSHFORWARD_PRIOR_STAGE_NAME
        || stage == TOPOLOGY_PUSHFORWARD_PDE_STAGE_NAME
        || stage == TOPOLOGY_PUSHFORWARD_FIELD_STAGE_NAME
        || stage == TOPOLOGY_PUSHFORWARD_FLUX_STAGE_NAME
        || stage == TOPOLOGY_PUSHFORWARD_AMPERE_STAGE_NAME
        || stage == TOPOLOGY_SPARSE_NOISY_PRIOR_STAGE_NAME
        || stage == TOPOLOGY_SPARSE_NOISY_PDE_STAGE_NAME
        || stage == TOPOLOGY_SPARSE_NOISY_OBS_STAGE_NAME
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn resolve_mesh_path(path: &Path) -> PathBuf {
    if path.exists() || path.is_absolute() {
        return path.to_path_buf();
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

fn outer_boundary_predicate(point: CoordRef<'_>, geom: ToroidalInductorGeometry) -> bool {
    let d = (geom.box_half_length - point[0].abs())
        .min(geom.box_half_length - point[1].abs())
        .min(geom.box_half_length - point[2].abs());
    d < 10.0 * geom.target_air_cell_size
}

fn toroidal_radius(point: CoordRef<'_>, geom: ToroidalInductorGeometry) -> f64 {
    let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
    ((rho - geom.major_radius).powi(2) + point[2] * point[2]).sqrt()
}

fn toroidal_direction(point: CoordRef<'_>) -> [f64; 3] {
    let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
    if rho < EPS {
        [0.0, 0.0, 0.0]
    } else {
        [-point[1] / rho, point[0] / rho, 0.0]
    }
}

fn toroidal_shell_point(
    geom: ToroidalInductorGeometry,
    minor_radius: f64,
    phi: f64,
    theta: f64,
) -> [f64; 3] {
    let rho = geom.major_radius + minor_radius * theta.cos();
    [rho * phi.cos(), rho * phi.sin(), minor_radius * theta.sin()]
}

fn coil_mode(geom: ToroidalInductorGeometry, mu_0: f64) -> DiffFormClosure {
    let j0 = 1.0;
    let sigma = 0.18;
    let eps = 0.03;
    DiffFormClosure::one_form(
        move |point| {
            let s = toroidal_radius(point, geom);
            let smoothstep = |t: f64| t * t * (3.0 - 2.0 * t);
            let inner = geom.core_minor_radius + eps;
            let outer = geom.coil_minor_radius - eps;
            let tin = ((s - inner) / eps).clamp(0.0, 1.0);
            let tout = ((outer - s) / eps).clamp(0.0, 1.0);
            let cutoff = smoothstep(tin) * smoothstep(tout);
            let s0 = 0.5 * (geom.core_minor_radius + geom.coil_minor_radius);
            let gauss = (-((s - s0) * (s - s0)) / (sigma * sigma)).exp();
            let amplitude = mu_0 * j0 * gauss * cutoff;
            let direction = toroidal_direction(point);
            FeecVector::from_column_slice(&[
                amplitude * direction[0],
                amplitude * direction[1],
                amplitude * direction[2],
            ])
        },
        3,
    )
}

fn assemble_weighted_source(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    inverse_permeability: &InnerProductWeightClosure,
    source: &DiffFormClosure,
) -> FeecVector {
    assemble_galvec(
        topology,
        metric,
        SourceElVec::new_weighted(source, coords, None, inverse_permeability),
    )
}

fn sorted_boundary_dofs<P>(
    topology: &Complex,
    coords: &MeshCoords,
    dim: usize,
    predicate: P,
) -> Vec<usize>
where
    P: Fn(CoordRef<'_>) -> bool + Sync,
{
    let mut dofs = assemble::boundary_simplices_where_barycenter(topology, coords, dim, predicate);
    dofs.sort_unstable();
    dofs
}

fn build_whittle_prior(
    system: &ReducedLinearPdeAssembly,
    stabilize_precision: bool,
) -> GaussianPriorSpec {
    let laplacian = system.operator.clone();
    let mass = system.state_mass.clone();
    let mass_inverse = system
        .state_mass_inverse
        .as_ref()
        .expect("1-form reduced system should expose an NC1 projected mass inverse")
        .clone();
    let a = add_sparse(&laplacian, &mass);
    let precision = &a.transpose() * &(&mass_inverse * &a);
    let precision = if stabilize_precision {
        stabilize_spd_precision(precision)
    } else {
        precision
    };
    GaussianPriorSpec {
        mean: vec![0.0; system.state_dimension()],
        precision: csr_to_triplet(&precision),
    }
}

fn zero_source_operator(residual_dimension: usize, source_dimension: usize) -> SparseTripletMatrix {
    SparseTripletMatrix::new(residual_dimension, source_dimension)
}

fn scale_gaussian_prior_precision(
    mut prior: GaussianPriorSpec,
    scale: f64,
) -> Result<GaussianPriorSpec, String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("prior precision scale must be finite and positive".to_string());
    }
    prior.precision = scale_triplet_matrix(&prior.precision, scale);
    Ok(prior)
}

fn mass_weighted_pde_precision(
    system: &ReducedLinearPdeAssembly,
    config: &ToroidalHarmonicBConfig,
) -> Result<SparseTripletMatrix, String> {
    let pde_variance = config.pde_variance;
    if !pde_variance.is_finite() || pde_variance <= 0.0 {
        return Err("pde_variance must be finite and positive".to_string());
    }
    let precision_scale = effective_mass_weighted_pde_precision_scale(config, system)?;
    let mass_inverse = system
        .state_mass_inverse
        .as_ref()
        .ok_or_else(|| "mass-weighted PDE residual requires state_mass_inverse".to_string())?;
    let residual_dimension = system.residual_dimension();
    if mass_inverse.nrows() != residual_dimension || mass_inverse.ncols() != residual_dimension {
        return Err(format!(
            "mass-weighted PDE precision has shape {}x{}, expected {}x{}",
            mass_inverse.nrows(),
            mass_inverse.ncols(),
            residual_dimension,
            residual_dimension
        ));
    }
    Ok(scale_triplet_matrix(
        &csr_to_triplet(mass_inverse),
        precision_scale / pde_variance,
    ))
}

fn effective_mass_weighted_pde_precision_scale(
    config: &ToroidalHarmonicBConfig,
    system: &ReducedLinearPdeAssembly,
) -> Result<f64, String> {
    let mut scale = config.mass_weighted_pde_precision_scale;
    if config.normalize_mass_weighted_pde_residual {
        let mass_inverse = system
            .state_mass_inverse
            .as_ref()
            .ok_or_else(|| "mass-weighted PDE residual requires state_mass_inverse".to_string())?;
        scale *= reciprocal_mean_diagonal(mass_inverse)?;
    }
    Ok(scale)
}

fn reciprocal_mean_diagonal(matrix: &FeecCsr) -> Result<f64, String> {
    let dimension = matrix.nrows();
    if dimension == 0 || matrix.ncols() != dimension {
        return Err(format!(
            "mean-diagonal normalization requires a nonempty square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut trace = 0.0;
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            trace += value;
        }
    }
    if !trace.is_finite() || trace <= 0.0 {
        return Err(format!(
            "mean-diagonal normalization requires positive finite trace, got {trace}"
        ));
    }
    Ok(dimension as f64 / trace)
}

fn scale_triplet_matrix(matrix: &SparseTripletMatrix, scale: f64) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col,
                value: scale * value,
            }),
    )
}

fn solve_full_feec_deterministic_reference(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    inverse_permeability: &InnerProductWeightClosure,
    source_galvec: &FeecVector,
    geom: ToroidalInductorGeometry,
) -> Cochain {
    let strong_state_dofs = sorted_boundary_dofs(topology, coords, 1, |point| {
        outer_boundary_predicate(point, geom)
    })
    .into_iter()
    .collect::<HashSet<_>>();
    let strong_aux_dofs = sorted_boundary_dofs(topology, coords, 0, |point| {
        outer_boundary_predicate(point, geom)
    })
    .into_iter()
    .collect::<HashSet<_>>();
    let strong_state_predicate = |sidx: KSimplexIdx| strong_state_dofs.contains(&sidx);
    let strong_aux_predicate = |sidx: KSimplexIdx| strong_aux_dofs.contains(&sidx);
    let zero_data = |_sidx: KSimplexIdx| 0.0;

    let (_, truth_a, _) =
        hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
            topology,
            metric,
            None,
            source_galvec.clone(),
            1,
            1,
            coords,
            None,
            inverse_permeability,
            &strong_state_predicate,
            &zero_data,
            &strong_aux_predicate,
            &zero_data,
        );
    truth_a
}

fn build_exterior_derivative_row_operator(topology: &Complex) -> Result<SparseRowOperator, String> {
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    SparseRowOperator::new(d1.ncols(), csr_rows(&d1)).map_err(|err| err.to_string())
}

fn build_sampled_magnetic_field_component_operator(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    if component_index >= 3 {
        return Err(format!(
            "magnetic-field vector component index {component_index} is out of range"
        ));
    }
    let topo_dim = topology.dim();
    let bary_local = barycenter_local(topo_dim);
    let mut rows = Vec::with_capacity(topology.cells().len());
    for cell in topology.skeleton(topo_dim).handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let mut row = Vec::new();
        for dof_simp in cell.mesh_subsimps(2) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let ambient_value = cell_coords.lift_form(&lsf.at_point(&bary_local));
            let coeffs = ambient_value.coeffs();
            let coefficient = match component_index {
                0 => coeffs[2],
                1 => -coeffs[1],
                2 => coeffs[0],
                _ => unreachable!(),
            };
            if coefficient.abs() > EPS {
                row.push((dof_simp.kidx(), coefficient));
            }
        }
        rows.push(row);
    }
    SparseRowOperator::new(topology.nsimplices(2), rows).map_err(|err| err.to_string())
}

fn compose_sparse_row_operators(
    lhs: &SparseRowOperator,
    rhs: &SparseRowOperator,
) -> Result<SparseRowOperator, String> {
    if lhs.ncols != rhs.nrows() {
        return Err(format!(
            "cannot compose sparse row operators with dimensions {}x{} and {}x{}",
            lhs.nrows(),
            lhs.ncols,
            rhs.nrows(),
            rhs.ncols
        ));
    }

    let mut rows = Vec::with_capacity(lhs.nrows());
    for lhs_row in &lhs.rows {
        let mut entries = BTreeMap::<usize, f64>::new();
        for (mid, lhs_value) in lhs_row {
            for (col, rhs_value) in &rhs.rows[*mid] {
                *entries.entry(*col).or_insert(0.0) += *lhs_value * *rhs_value;
            }
        }
        rows.push(
            entries
                .into_iter()
                .filter(|(_, value)| value.abs() > EPS)
                .collect(),
        );
    }

    SparseRowOperator::new(rhs.ncols, rows).map_err(|err| err.to_string())
}

fn combine_component_rows(
    operators: &[SparseRowOperator; 3],
    row_index: usize,
    direction: [f64; 3],
) -> Result<SparseTripletMatrix, String> {
    let ncols = operators[0].ncols;
    let mut entries = BTreeMap::<usize, f64>::new();
    for component in 0..3 {
        if operators[component].ncols != ncols {
            return Err("component operators must have matching column counts".to_string());
        }
        for (col, value) in &operators[component].rows[row_index] {
            *entries.entry(*col).or_insert(0.0) += direction[component] * *value;
        }
    }
    Ok(SparseTripletMatrix::from_triplets(
        1,
        ncols,
        entries
            .into_iter()
            .filter(|(_, value)| value.abs() > EPS)
            .map(|(col, value)| SparseTriplet { row: 0, col, value }),
    ))
}

#[derive(Debug, Clone)]
struct AmpereLoopOperator {
    state_operator: SparseTripletMatrix,
    gp_terms: Vec<GpFunctionalTerm>,
    beta_operator_value: f64,
}

fn build_ampere_loop_operator(
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    component_operators: &[SparseRowOperator; 3],
    h2_vectors: &[[f64; 3]],
) -> Result<AmpereLoopOperator, String> {
    build_ampere_loop_operator_at_phi(topology, coords, geom, 0.0, component_operators, h2_vectors)
}

fn build_ampere_loop_operator_at_phi(
    topology: &Complex,
    coords: &MeshCoords,
    geom: ToroidalInductorGeometry,
    phi: f64,
    component_operators: &[SparseRowOperator; 3],
    h2_vectors: &[[f64; 3]],
) -> Result<AmpereLoopOperator, String> {
    if AMPERE_LOOP_SEGMENTS < 3 {
        return Err("Ampere-loop quadrature requires at least three segments".to_string());
    }
    let ncols = component_operators[0].ncols;
    let inv_mu0 = 1.0 / vacuum_permeability();
    let mut entries = BTreeMap::<usize, f64>::new();
    let mut gp_terms = Vec::with_capacity(AMPERE_LOOP_SEGMENTS);
    let mut beta_operator_value = 0.0;
    for segment in 0..AMPERE_LOOP_SEGMENTS {
        let theta0 = 2.0 * PI * segment as f64 / AMPERE_LOOP_SEGMENTS as f64;
        let theta1 = 2.0 * PI * (segment + 1) as f64 / AMPERE_LOOP_SEGMENTS as f64;
        let theta_mid = 0.5 * (theta0 + theta1);
        let p0 = ampere_loop_point(geom, phi, theta0);
        let p1 = ampere_loop_point(geom, phi, theta1);
        let midpoint = ampere_loop_point(geom, phi, theta_mid);
        let dl = [
            inv_mu0 * (p1[0] - p0[0]),
            inv_mu0 * (p1[1] - p0[1]),
            inv_mu0 * (p1[2] - p0[2]),
        ];
        gp_terms.push(GpFunctionalTerm {
            point: midpoint,
            weight: dl,
        });
        let cell_index = nearest_cell(topology, coords, midpoint)?;
        for component in 0..3 {
            for (col, value) in &component_operators[component].rows[cell_index] {
                *entries.entry(*col).or_insert(0.0) += dl[component] * *value;
            }
        }
        let h2_vector = h2_vectors.get(cell_index).ok_or_else(|| {
            format!(
                "Ampere-loop sample selected cell {cell_index}, but h2 vector table has {} rows",
                h2_vectors.len()
            )
        })?;
        beta_operator_value += dot3(*h2_vector, dl);
    }
    if entries.is_empty() {
        return Err("Ampere-loop field operator has no nonzero entries".to_string());
    }

    Ok(AmpereLoopOperator {
        state_operator: SparseTripletMatrix::from_triplets(
            1,
            ncols,
            entries
                .into_iter()
                .filter(|(_, value)| value.abs() > EPS)
                .map(|(col, value)| SparseTriplet { row: 0, col, value }),
        ),
        gp_terms,
        beta_operator_value,
    })
}

fn ampere_loop_point(geom: ToroidalInductorGeometry, phi: f64, theta: f64) -> [f64; 3] {
    let radial = geom.major_radius + AMPERE_LOOP_RADIUS * theta.cos();
    [
        radial * phi.cos(),
        radial * phi.sin(),
        AMPERE_LOOP_RADIUS * theta.sin(),
    ]
}

fn vacuum_permeability() -> f64 {
    4e-7 * PI
}

#[derive(Debug, Clone)]
struct FluxPatchOperator {
    state_operator: SparseTripletMatrix,
    face_weights: Vec<(usize, f64)>,
    gp_terms: Vec<GpFunctionalTerm>,
}

fn build_flux_patch_operator(
    topology: &Complex,
    coords: &MeshCoords,
    face_rows: &[Vec<(usize, f64)>],
    center: [f64; 3],
    patch_radius: f64,
    y_half_width: f64,
) -> Result<FluxPatchOperator, String> {
    let mut edge_weights = BTreeMap::<usize, f64>::new();
    let mut face_weights = Vec::new();
    let mut gp_terms = Vec::new();
    for face_index in 0..topology.nsimplices(2) {
        let face = SimplexIdx::new(2, face_index).handle(topology);
        let face_coords = SimplexCoords::from_simplex_and_coords(&face, coords);
        let bary = face_coords.barycenter();
        let dx = bary[0] - center[0];
        let dy = bary[1] - center[1];
        let dz = bary[2] - center[2];
        let radial = (dx * dx + dz * dz).sqrt();
        if radial > patch_radius || dy.abs() > y_half_width {
            continue;
        }

        let normal = face_normal(&face_coords);
        let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if norm <= EPS {
            continue;
        }
        let alignment = normal[1].abs() / norm;
        if alignment < 0.55 {
            continue;
        }
        let sign = if normal[1] >= 0.0 { 1.0 } else { -1.0 };
        face_weights.push((face_index, sign));
        gp_terms.push(GpFunctionalTerm {
            point: [bary[0], bary[1], bary[2]],
            weight: [
                0.5 * sign * normal[0],
                0.5 * sign * normal[1],
                0.5 * sign * normal[2],
            ],
        });
        for (edge, value) in &face_rows[face_index] {
            *edge_weights.entry(*edge).or_insert(0.0) += sign * *value;
        }
    }

    if face_weights.is_empty() {
        return Err(format!(
            "flux sensor patch at ({:.3}, {:.3}, {:.3}) selected no faces",
            center[0], center[1], center[2]
        ));
    }

    Ok(FluxPatchOperator {
        state_operator: SparseTripletMatrix::from_triplets(
            1,
            topology.nsimplices(1),
            edge_weights
                .into_iter()
                .filter(|(_, value)| value.abs() > EPS)
                .map(|(col, value)| SparseTriplet { row: 0, col, value }),
        ),
        face_weights,
        gp_terms,
    })
}

fn build_meridional_flux_panel_operator(
    topology: &Complex,
    coords: &MeshCoords,
    face_rows: &[Vec<(usize, f64)>],
    geom: ToroidalInductorGeometry,
    phi: f64,
    radial_half_width: f64,
    vertical_half_width: f64,
    plane_half_width: f64,
) -> Result<FluxPatchOperator, String> {
    let e_r = [phi.cos(), phi.sin(), 0.0];
    let e_phi = [-phi.sin(), phi.cos(), 0.0];
    let mut edge_weights = BTreeMap::<usize, f64>::new();
    let mut face_weights = Vec::new();
    let mut gp_terms = Vec::new();
    for face_index in 0..topology.nsimplices(2) {
        let face = SimplexIdx::new(2, face_index).handle(topology);
        let face_coords = SimplexCoords::from_simplex_and_coords(&face, coords);
        let bary = face_coords.barycenter();
        let radial_offset = bary[0] * e_r[0] + bary[1] * e_r[1] - geom.major_radius;
        let plane_offset = bary[0] * e_phi[0] + bary[1] * e_phi[1];
        if radial_offset.abs() > radial_half_width
            || bary[2].abs() > vertical_half_width
            || plane_offset.abs() > plane_half_width
        {
            continue;
        }

        let normal = face_normal(&face_coords);
        let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if norm <= EPS {
            continue;
        }
        let normal_dot = normal[0] * e_phi[0] + normal[1] * e_phi[1] + normal[2] * e_phi[2];
        let alignment = normal_dot.abs() / norm;
        if alignment < 0.45 {
            continue;
        }
        let sign = if normal_dot >= 0.0 { 1.0 } else { -1.0 };
        face_weights.push((face_index, sign));
        gp_terms.push(GpFunctionalTerm {
            point: [bary[0], bary[1], bary[2]],
            weight: [
                0.5 * sign * normal[0],
                0.5 * sign * normal[1],
                0.5 * sign * normal[2],
            ],
        });
        for (edge, value) in &face_rows[face_index] {
            *edge_weights.entry(*edge).or_insert(0.0) += sign * *value;
        }
    }

    if face_weights.is_empty() {
        return Err(format!(
            "embedded flux panel at phi={phi:.3} selected no faces"
        ));
    }

    Ok(FluxPatchOperator {
        state_operator: SparseTripletMatrix::from_triplets(
            1,
            topology.nsimplices(1),
            edge_weights
                .into_iter()
                .filter(|(_, value)| value.abs() > EPS)
                .map(|(col, value)| SparseTriplet { row: 0, col, value }),
        ),
        face_weights,
        gp_terms,
    })
}

fn build_harmonic_projection_state_operator(
    topology: &Complex,
    h2: &FeecVector,
    mass_2: &FeecCsr,
) -> Result<SparseTripletMatrix, String> {
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    let weighted = mass_2 * h2;
    let mut edge_weights = BTreeMap::<usize, f64>::new();
    for (face, edge, value) in d1.triplet_iter() {
        *edge_weights.entry(edge).or_insert(0.0) += weighted[face] * *value;
    }
    Ok(SparseTripletMatrix::from_triplets(
        1,
        topology.nsimplices(1),
        edge_weights
            .into_iter()
            .filter(|(_, value)| value.abs() > EPS)
            .map(|(col, value)| SparseTriplet { row: 0, col, value }),
    ))
}

fn nearest_cell(
    topology: &Complex,
    coords: &MeshCoords,
    target: [f64; 3],
) -> Result<usize, String> {
    let mut used_cells = HashSet::new();
    nearest_unused_cell(topology, coords, target, &mut used_cells)
}

fn nearest_unused_cell(
    topology: &Complex,
    coords: &MeshCoords,
    target: [f64; 3],
    used_cells: &mut HashSet<usize>,
) -> Result<usize, String> {
    let mut best = None;
    for cell in topology.skeleton(topology.dim()).handle_iter() {
        let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
        let bary = cell_coords.barycenter();
        let distance_sq = (bary[0] - target[0]).powi(2)
            + (bary[1] - target[1]).powi(2)
            + (bary[2] - target[2]).powi(2);
        let index = cell.kidx();
        if used_cells.contains(&index) {
            continue;
        }
        match best {
            Some((_, best_distance)) if best_distance <= distance_sq => {}
            _ => best = Some((index, distance_sq)),
        }
    }
    let Some((index, _)) = best else {
        return Err("could not select an unused Hall-probe cell".to_string());
    };
    used_cells.insert(index);
    Ok(index)
}

fn cell_barycenter(
    topology: &Complex,
    coords: &MeshCoords,
    cell_index: usize,
) -> Result<[f64; 3], String> {
    let cell = SimplexIdx::new(topology.dim(), cell_index).handle(topology);
    let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
    let bary = cell_coords.barycenter();
    Ok([bary[0], bary[1], bary[2]])
}

fn face_normal(face_coords: &SimplexCoords) -> [f64; 3] {
    let p0 = face_coords.coord(0);
    let p1 = face_coords.coord(1);
    let p2 = face_coords.coord(2);
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn apply_triplet_row(operator: &SparseTripletMatrix, state: &FeecVector) -> Result<f64, String> {
    if operator.nrows() != 1 {
        return Err(format!(
            "expected a single-row operator, found {} rows",
            operator.nrows()
        ));
    }
    if operator.ncols() != state.len() {
        return Err(format!(
            "operator column count {} must match state length {}",
            operator.ncols(),
            state.len()
        ));
    }
    let mut out = 0.0;
    for (row, col, value) in operator.triplet_iter() {
        debug_assert_eq!(row, 0);
        out += value * state[col];
    }
    Ok(out)
}

fn gaussian_prior_quadratic(prior: &GaussianPriorSpec, value: &FeecVector) -> Result<f64, String> {
    if prior.mean.len() != value.len() {
        return Err(format!(
            "prior mean length {} must match vector length {}",
            prior.mean.len(),
            value.len()
        ));
    }
    let centered = FeecVector::from_iterator(
        value.len(),
        value
            .iter()
            .zip(prior.mean.iter())
            .map(|(x, mean)| x - mean),
    );
    sparse_quadratic(&prior.precision, &centered)
}

fn sparse_quadratic(matrix: &SparseTripletMatrix, value: &FeecVector) -> Result<f64, String> {
    if matrix.nrows() != value.len() || matrix.ncols() != value.len() {
        return Err(format!(
            "quadratic matrix shape {}x{} must match vector length {}",
            matrix.nrows(),
            matrix.ncols(),
            value.len()
        ));
    }
    let mut out = 0.0;
    for (row, col, entry) in matrix.triplet_iter() {
        out += value[row] * entry * value[col];
    }
    Ok(out)
}

fn vector_column_row_operator(vector: &FeecVector) -> Result<SparseRowOperator, String> {
    SparseRowOperator::new(
        1,
        (0..vector.len())
            .map(|row| {
                if vector[row].abs() > EPS {
                    vec![(0, vector[row])]
                } else {
                    Vec::new()
                }
            })
            .collect(),
    )
    .map_err(|err| err.to_string())
}

fn beta_column_triplet(value: f64) -> SparseTripletMatrix {
    scalar_column_triplet(value)
}

fn scalar_column_triplet(value: f64) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        1,
        1,
        (value.abs() > EPS).then_some(SparseTriplet {
            row: 0,
            col: 0,
            value,
        }),
    )
}

fn scalar_row_operator(value: f64) -> Result<SparseRowOperator, String> {
    SparseRowOperator::new(
        1,
        vec![if value.abs() > EPS {
            vec![(0, value)]
        } else {
            Vec::new()
        }],
    )
    .map_err(|err| err.to_string())
}

fn triplet_to_row_operator(operator: &SparseTripletMatrix) -> Result<SparseRowOperator, String> {
    let mut rows = vec![Vec::new(); operator.nrows()];
    for (row, col, value) in operator.triplet_iter() {
        if value.abs() > EPS {
            rows[row].push((col, value));
        }
    }
    SparseRowOperator::new(operator.ncols(), rows).map_err(|err| err.to_string())
}

fn columns_to_sparse_matrix(columns: &[FeecVector]) -> SparseTripletMatrix {
    let nrows = columns.first().map(|column| column.len()).unwrap_or(0);
    SparseTripletMatrix::from_triplets(
        nrows,
        columns.len(),
        columns.iter().enumerate().flat_map(|(col, column)| {
            column.iter().enumerate().filter_map(move |(row, value)| {
                (value.abs() > EPS).then_some(SparseTriplet {
                    row,
                    col,
                    value: *value,
                })
            })
        }),
    )
}

fn diagonal_precision(dimension: usize, diagonal_value: f64) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        dimension,
        dimension,
        (0..dimension).map(|index| SparseTriplet {
            row: index,
            col: index,
            value: diagonal_value,
        }),
    )
}

fn csr_to_triplet(matrix: &FeecCsr) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col,
                value: *value,
            }),
    )
}

fn add_sparse(lhs: &FeecCsr, rhs: &FeecCsr) -> FeecCsr {
    let mut coo = FeecCoo::new(lhs.nrows(), lhs.ncols());
    for (row, col, value) in lhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    for (row, col, value) in rhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
}

fn add_diagonal_shift(matrix: &FeecCsr, shift: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for index in 0..matrix.nrows() {
        coo.push(index, index, shift);
    }
    FeecCsr::from(&coo)
}

fn symmetrize_feec_csr(matrix: &FeecCsr) -> FeecCsr {
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            coo.push(row, col, *value);
        } else {
            coo.push(row, col, 0.5 * *value);
            coo.push(col, row, 0.5 * *value);
        }
    }
    FeecCsr::from(&coo)
}

fn diagonal_stats_feec(matrix: &FeecCsr) -> (f64, f64) {
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    let min_diag = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_diag = diagonal.iter().copied().map(f64::abs).fold(0.0, f64::max);
    (min_diag, max_abs_diag.max(1.0))
}

fn stabilize_spd_precision(mut precision: FeecCsr) -> FeecCsr {
    if feec_csr_to_gmrf(&precision).cholesky_sqrt_lower().is_ok() {
        return precision;
    }
    precision = symmetrize_feec_csr(&precision);
    if feec_csr_to_gmrf(&precision).cholesky_sqrt_lower().is_ok() {
        return precision;
    }
    let (min_diag, max_abs_diag) = diagonal_stats_feec(&precision);
    let mut shift = if min_diag.is_finite() && min_diag <= 0.0 {
        (-min_diag) + max_abs_diag * 1e-8
    } else {
        max_abs_diag * 1e-12
    }
    .max(1e-10);
    for _ in 0..12 {
        let shifted = add_diagonal_shift(&precision, shift);
        if feec_csr_to_gmrf(&shifted).cholesky_sqrt_lower().is_ok() {
            return shifted;
        }
        shift *= 10.0;
        precision = shifted;
    }
    precision
}

fn csr_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        rows[row].push((col, *value));
    }
    rows
}

fn subtract_from_bias(bias: &mut [f64], rhs: &FeecVector) -> Result<(), String> {
    if bias.len() != rhs.len() {
        return Err(format!(
            "bias length {} must match rhs length {}",
            bias.len(),
            rhs.len()
        ));
    }
    for (entry, value) in bias.iter_mut().zip(rhs.iter()) {
        *entry -= *value;
    }
    Ok(())
}

fn mass_orthonormalize_harmonic_basis(
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<FeecMatrix, String> {
    if harmonic_basis.ncols() == 0 {
        return Err("harmonic basis must contain at least one column".to_string());
    }

    let mut columns = Vec::with_capacity(harmonic_basis.ncols());
    for j in 0..harmonic_basis.ncols() {
        let mut column = harmonic_basis.column(j).into_owned();
        for previous in &columns {
            let coeff = mass_inner_product(previous, &column, mass_u);
            column -= previous * coeff;
        }

        let norm_sq = mass_inner_product(&column, &column, mass_u);
        if !norm_sq.is_finite() || norm_sq <= EPS {
            return Err(format!(
                "harmonic basis column {j} became singular during orthonormalization"
            ));
        }
        column /= norm_sq.sqrt();
        columns.push(column);
    }

    Ok(FeecMatrix::from_columns(&columns))
}

fn mass_inner_product(lhs: &FeecVector, rhs: &FeecVector, mass_u: &FeecCsr) -> f64 {
    let weighted_rhs = mass_u * rhs;
    lhs.dot(&weighted_rhs)
}

fn pointwise_variance_ratio(prior: &FeecVector, posterior: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(
        prior.len(),
        prior
            .iter()
            .zip(posterior.iter())
            .map(|(prior, posterior)| {
                if *prior > EPS {
                    (posterior / prior).max(0.0)
                } else {
                    0.0
                }
            }),
    )
}

fn dot3(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use formoniq::reduction::DofLayout;

    #[cfg(feature = "heavy-tests")]
    use crate::test_util::lock_feec_harmonic_tests;

    #[test]
    fn mass_weighted_pde_precision_scales_reduced_mass_inverse() {
        let system = ReducedLinearPdeAssembly {
            operator: core_triplet_to_feec_csr(&diagonal_precision(2, 1.0)),
            residual_bias: FeecVector::zeros(2),
            state_mass: core_triplet_to_feec_csr(&diagonal_precision(2, 1.0)),
            state_mass_inverse: Some(core_triplet_to_feec_csr(
                &SparseTripletMatrix::from_triplets(
                    2,
                    2,
                    [
                        SparseTriplet {
                            row: 0,
                            col: 0,
                            value: 2.0,
                        },
                        SparseTriplet {
                            row: 1,
                            col: 1,
                            value: 3.0,
                        },
                    ],
                ),
            )),
            layout: DofLayout::identity(2),
            forcing_operator: core_triplet_to_feec_csr(&diagonal_precision(2, -1.0)),
            neumann_operator: core_triplet_to_feec_csr(&diagonal_precision(2, -1.0)),
        };

        let config = ToroidalHarmonicBConfig {
            pde_variance: 0.25,
            use_mass_weighted_pde_residual: true,
            ..ToroidalHarmonicBConfig::default()
        };
        let precision = mass_weighted_pde_precision(&system, &config).unwrap();
        assert_eq!(precision.nrows(), 2);
        assert_eq!(precision.ncols(), 2);
        let diagonal_values: Vec<f64> = (0..2)
            .map(|index| {
                precision
                    .triplet_iter()
                    .filter(|(row, col, _)| *row == index && *col == index)
                    .map(|(_, _, value)| value)
                    .sum()
            })
            .collect();
        assert!((diagonal_values[0] - 8.0).abs() < 1e-14);
        assert!((diagonal_values[1] - 12.0).abs() < 1e-14);

        let normalized_config = ToroidalHarmonicBConfig {
            pde_variance: 0.25,
            use_mass_weighted_pde_residual: true,
            normalize_mass_weighted_pde_residual: true,
            ..ToroidalHarmonicBConfig::default()
        };
        let normalized = mass_weighted_pde_precision(&system, &normalized_config).unwrap();
        let normalized_diagonal: Vec<f64> = (0..2)
            .map(|index| {
                normalized
                    .triplet_iter()
                    .filter(|(row, col, _)| *row == index && *col == index)
                    .map(|(_, _, value)| value)
                    .sum()
            })
            .collect();
        assert!((normalized_diagonal[0] - 3.2).abs() < 1e-14);
        assert!((normalized_diagonal[1] - 4.8).abs() < 1e-14);
    }

    #[test]
    fn topology_summary_csv_quotes_betti_numbers() {
        let summary = ToroidalTopologySummary {
            betti_numbers: vec![1, 1, 1, 0],
            harmonic_2_dimension: 1,
            harmonic_2_mass_norm: 1.0,
            deterministic_harmonic_projection: 0.0,
            deterministic_harmonic_projection_relative: 0.0,
            deterministic_b_energy: 1.0,
            beta_true: 0.1,
            source_harmonic_energy_fraction: 0.05,
            source_harmonic_kappa: 0.1,
            ampere_loop_exact_nominal: 1.0,
            ampere_loop_harmonic_sensitivity: 2.0,
            linked_current_unit: 3.0,
            linked_current_true: 4.0,
            source_harmonic_projection_unit: 5.0,
            source_harmonic_projection_true: 6.0,
        };
        let path = std::env::temp_dir().join(format!(
            "toroidal_topology_summary_{}_{}.csv",
            std::process::id(),
            "betti"
        ));
        write_topology_summary_csv(&summary, &path).expect("CSV should write");
        let contents = fs::read_to_string(&path).expect("CSV should read");
        let _ = fs::remove_file(&path);
        assert!(
            contents.contains("betti_numbers,\"[1, 1, 1, 0]\"\n"),
            "betti_numbers row should be quoted CSV, got {contents:?}"
        );
    }

    #[test]
    fn toroidal_gp_baseline_functional_covariance_is_symmetric_and_component_independent() {
        let hyper = GpHyperparameters {
            matern_nu: 1.5,
            length_scale: 1.0,
            signal_variance: 2.0,
            log_marginal_likelihood: f64::NAN,
        };
        let x = vec![GpFunctionalTerm {
            point: [0.0, 0.0, 0.0],
            weight: [1.0, 0.0, 0.0],
        }];
        let y = vec![GpFunctionalTerm {
            point: [0.0, 0.0, 0.0],
            weight: [0.0, 1.0, 0.0],
        }];
        let shifted_x = vec![GpFunctionalTerm {
            point: [0.25, 0.0, 0.0],
            weight: [3.0, 0.0, 0.0],
        }];

        let xx = gp_functional_covariance(&x, &x, &hyper).unwrap();
        let xy = gp_functional_covariance(&x, &y, &hyper).unwrap();
        let xs = gp_functional_covariance(&x, &shifted_x, &hyper).unwrap();
        let sx = gp_functional_covariance(&shifted_x, &x, &hyper).unwrap();

        assert!(xx.is_finite() && xx > 0.0);
        assert!(xy.abs() < 1e-14, "independent components should not covary");
        assert!(
            (xs - sx).abs() < 1e-14,
            "functional covariance must be symmetric"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_topology_reports_one_mass_normalized_harmonic_2form() {
        let _lock = lock_feec_harmonic_tests();
        let summary = compute_toroidal_harmonic_topology_summary(MESH_PATH)
            .expect("topology summary should compute");
        assert_eq!(summary.harmonic_2_dimension, 1);
        assert!((summary.harmonic_2_mass_norm - 1.0).abs() < 1e-8);
        assert!(
            summary.deterministic_harmonic_projection_relative < 1e-7,
            "deterministic exact field should have negligible harmonic projection, got {}",
            summary.deterministic_harmonic_projection_relative
        );
        assert!(summary.ampere_loop_harmonic_sensitivity > 0.0);
        assert!((summary.source_harmonic_kappa - summary.beta_true).abs() < 1e-14);
        assert!((summary.source_harmonic_energy_fraction - 0.05).abs() < 1e-12);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_gp_baseline_smoke_reports_finite_predictions() {
        let _lock = lock_feec_harmonic_tests();
        let config = ToroidalGpBaselineConfig {
            output_dir: None,
            length_scales: vec![0.75, 1.5],
            signal_std_factors: vec![0.5, 1.0, 2.0],
            toroidal: ToroidalHarmonicBConfig {
                output_dir: None,
                ..ToroidalHarmonicBConfig::default()
            },
            ..ToroidalGpBaselineConfig::default()
        };
        let result = run_toroidal_topology_gp_baseline(&config)
            .expect("GP baseline should run on toroidal topology workspace");
        assert_eq!(result.topology_summary.harmonic_2_dimension, 1);
        assert_eq!(result.stages.len(), 3);
        for stage in &result.stages {
            assert!(stage.summary.length_scale.is_finite());
            assert!(stage.summary.signal_variance.is_finite());
            assert!(stage.summary.signal_variance > 0.0);
            assert!(!stage.heldout_predictions.is_empty());
            assert!(!stage.qois.is_empty());
            assert!(stage.heldout_predictions.iter().all(|row| {
                row.prediction.is_finite()
                    && row.posterior_sd.is_finite()
                    && row.posterior_sd >= 0.0
                    && row.standardized_residual.is_finite()
            }));
            assert!(stage
                .qois
                .iter()
                .all(|row| { row.mean.is_finite() && row.sd.is_finite() && row.sd >= 0.0 }));
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_sparse_gp_comparison_reports_finite_metrics() {
        let _lock = lock_feec_harmonic_tests();
        let config = ToroidalGpBaselineConfig {
            output_dir: None,
            length_scales: vec![0.75, 1.5],
            signal_std_factors: vec![0.5, 1.0],
            toroidal: ToroidalHarmonicBConfig {
                output_dir: None,
                include_full_field_variance_maps: false,
                use_mass_weighted_pde_residual: true,
                normalize_mass_weighted_pde_residual: true,
                source_alpha_true: 1.15,
                sparse_prediction_training_fraction: 0.10,
                sparse_prediction_noise_seed: 12345,
                sparse_prediction_field_phi_count: 4,
                sparse_prediction_field_theta_count: 4,
                sparse_prediction_linked_flux_count: 4,
                sparse_prediction_local_flux_phi_count: 2,
                sparse_prediction_local_flux_theta_count: 2,
                sparse_prediction_ampere_loop_count: 4,
                ..ToroidalHarmonicBConfig::default()
            },
            ..ToroidalGpBaselineConfig::default()
        };
        let mut toroidal_config = config.toroidal.clone();
        toroidal_config.observation_layout =
            ToroidalObservationLayout::TopologySparseNoisyPrediction;
        let workspace = build_workspace(&toroidal_config).expect("sparse workspace should build");
        assert_eq!(workspace.topology_summary.harmonic_2_dimension, 1);

        let field_training = topology_pushforward_training_observations(
            &workspace,
            StageProblemKind::TopologySparseNoisyFieldObservations,
        );
        assert!(
            field_training
                .iter()
                .all(|observation| observation.sensor_type == "hall"),
            "field stage should train on Hall rows only"
        );
        let field_heldout = sparse_comparison_heldout_observations(&workspace, &field_training);
        assert!(
            field_heldout.len() > workspace.heldout_observations.len(),
            "field-stage heldout should include withheld training flux/loop rows plus the held-out bank"
        );
        for family in ["hall", "flux", "ampere_loop"] {
            assert!(
                field_heldout
                    .iter()
                    .any(|observation| sensor_family(&observation.sensor_type) == family),
                "missing heldout rows for sensor family {family}"
            );
        }

        let values = field_heldout
            .iter()
            .enumerate()
            .map(|(index, observation)| PredictionMetricValue {
                sensor_family: sensor_family(&observation.sensor_type).to_string(),
                residual: 0.01 * (index as f64 + 1.0),
                sd: 0.1,
                covered95: true,
                standardized_residual: 0.1 * (index as f64 + 1.0),
            })
            .collect::<Vec<_>>();
        let rows = sparse_metric_rows_from_values(
            "FEEC-GMRF full",
            "C2_sparse_field",
            field_training.len(),
            true,
            true,
            true,
            &values,
        );
        assert!(rows.iter().any(|row| row.sensor_family == "all"));
        assert!(rows
            .iter()
            .all(|row| row.rmse.is_finite() && row.nlpd.is_finite()));
        assert!(rows.iter().all(|row| {
            row.training_rows > 0
                && row.heldout_rows > 0
                && row.rmse.is_finite()
                && row.nlpd.is_finite()
                && row.coverage_fraction.is_finite()
                && row.coverage_fraction >= 0.0
                && row.coverage_fraction <= 1.0
                && row.max_abs_standardized_residual.is_finite()
                && row.mean_abs_standardized_residual.is_finite()
        }));
    }

    #[test]
    fn source_template_gp_dense_conditioning_matches_scalar_posterior() {
        let train = vec![SourceTemplateGpTarget {
            target: GpPredictionTarget {
                sensor_type: "scalar".to_string(),
                name: "source_template_scalar".to_string(),
                truth: 5.0,
                observed: 5.0,
                noise_variance: 0.25,
                terms: vec![GpFunctionalTerm {
                    point: [0.0, 0.0, 0.0],
                    weight: [0.0, 0.0, 0.0],
                }],
            },
            exact_template_value: 2.0,
            oracle_template_value: 2.0,
        }];
        let pred = vec![SourceTemplateGpTarget {
            target: GpPredictionTarget {
                sensor_type: "scalar".to_string(),
                name: "predicted_template_scalar".to_string(),
                truth: 0.0,
                observed: 0.0,
                noise_variance: 0.0,
                terms: vec![GpFunctionalTerm {
                    point: [0.0, 0.0, 0.0],
                    weight: [0.0, 0.0, 0.0],
                }],
            },
            exact_template_value: 3.0,
            oracle_template_value: 3.0,
        }];
        let hyper = GpHyperparameters {
            matern_nu: 1.5,
            length_scale: 1.0,
            signal_variance: 1.0,
            log_marginal_likelihood: f64::NAN,
        };
        let posterior = condition_source_template_gp_functionals(
            &hyper,
            &train,
            &pred,
            SourceTemplateKind::ExactSource,
            1.0,
            4.0,
            0.0,
        )
        .expect("scalar source-template posterior should condition");
        let analytic_var = 1.0 / (1.0 / 4.0 + 2.0 * 2.0 / 0.25);
        let analytic_mean = analytic_var * (1.0 / 4.0 + 2.0 * 5.0 / 0.25);
        assert!((posterior.source_mean - analytic_mean).abs() < 1e-12);
        assert!((posterior.source_variance - analytic_var).abs() < 1e-12);
        assert!((posterior.mean[0] - 3.0 * analytic_mean).abs() < 1e-12);
        assert!((posterior.variance[0] - 9.0 * analytic_var).abs() < 1e-12);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn source_template_gp_template_values_match_exact_and_oracle_rows() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            source_alpha_true: 1.15,
            include_full_field_variance_maps: false,
            sparse_prediction_training_fraction: 0.10,
            sparse_prediction_noise_seed: 12345,
            sparse_prediction_field_phi_count: 4,
            sparse_prediction_field_theta_count: 4,
            sparse_prediction_linked_flux_count: 4,
            sparse_prediction_local_flux_phi_count: 2,
            sparse_prediction_local_flux_theta_count: 2,
            sparse_prediction_ampere_loop_count: 4,
            ..ToroidalHarmonicBConfig::default()
        };
        config.observation_layout = ToroidalObservationLayout::TopologySparseNoisyPrediction;
        let workspace = build_workspace(&config).expect("sparse workspace should build");
        let observation = workspace
            .observations
            .iter()
            .find(|observation| observation.sensor_type == AMPERE_LOOP_SENSOR_TYPE)
            .expect("Ampere-loop row should exist");
        let target = source_template_target_from_observation(&workspace, observation)
            .expect("source-template target should build");
        let exact = apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)
            .expect("template row should apply to nominal potential");
        let oracle = exact
            + workspace.topology_summary.source_harmonic_kappa * observation.beta_operator_value;
        assert!((target.exact_template_value - exact).abs() < 1e-12);
        assert!((target.oracle_template_value - oracle).abs() < 1e-12);
        assert!(
            (target.oracle_template_value - target.exact_template_value).abs() > 1e-14,
            "oracle template should add a nonzero topological source direction"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn source_template_gp_smoke_reports_finite_predictions() {
        let _lock = lock_feec_harmonic_tests();
        let config = ToroidalGpBaselineConfig {
            output_dir: None,
            length_scales: vec![0.75, 1.5],
            signal_std_factors: vec![0.5, 1.0],
            toroidal: ToroidalHarmonicBConfig {
                output_dir: None,
                include_full_field_variance_maps: false,
                use_mass_weighted_pde_residual: true,
                normalize_mass_weighted_pde_residual: true,
                source_alpha_true: 1.15,
                sparse_prediction_training_fraction: 0.10,
                sparse_prediction_noise_seed: 12345,
                sparse_prediction_field_phi_count: 4,
                sparse_prediction_field_theta_count: 4,
                sparse_prediction_linked_flux_count: 4,
                sparse_prediction_local_flux_phi_count: 2,
                sparse_prediction_local_flux_theta_count: 2,
                sparse_prediction_ampere_loop_count: 4,
                ..ToroidalHarmonicBConfig::default()
            },
            ..ToroidalGpBaselineConfig::default()
        };
        let result = run_toroidal_topology_source_template_gp_baseline(&config)
            .expect("source-template GP baseline should run");
        assert_eq!(result.topology_summary.harmonic_2_dimension, 1);
        assert_eq!(result.stages.len(), 6);
        assert!(result.metrics.iter().any(|row| row.sensor_family == "all"));
        for stage in &result.stages {
            assert!(stage.summary.source_posterior_mean.is_finite());
            assert!(stage.summary.source_posterior_variance.is_finite());
            assert!(stage.summary.source_posterior_variance >= 0.0);
            assert!(stage.summary.length_scale.is_finite());
            assert!(stage.summary.signal_variance.is_finite());
            assert!(!stage.heldout_predictions.is_empty());
            assert!(stage.heldout_predictions.iter().all(|row| {
                row.prediction.is_finite()
                    && row.posterior_sd.is_finite()
                    && row.posterior_sd >= 0.0
                    && row.standardized_residual.is_finite()
            }));
            assert!(!stage.qois.is_empty());
            assert!(stage
                .qois
                .iter()
                .all(|row| row.mean.is_finite() && row.sd.is_finite() && row.sd >= 0.0));
        }
    }

    #[test]
    fn harmonic_coupling_calibration_scalar_posterior_matches_hand_calculation() {
        let drive_currents = [0.6, 1.0, 1.4];
        let c_h_true = 2.0;
        let prior_variance = 25.0;
        let observation_variance = 0.25;
        let system = ReducedLinearPdeAssembly {
            operator: core_triplet_to_feec_csr(&SparseTripletMatrix::new(1, 1)),
            residual_bias: FeecVector::zeros(1),
            state_mass: core_triplet_to_feec_csr(&diagonal_precision(1, 1.0)),
            state_mass_inverse: Some(core_triplet_to_feec_csr(&diagonal_precision(1, 1.0))),
            layout: DofLayout::identity(1),
            forcing_operator: core_triplet_to_feec_csr(&SparseTripletMatrix::new(1, 1)),
            neumann_operator: core_triplet_to_feec_csr(&SparseTripletMatrix::new(1, 1)),
        };
        let problem = LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0],
                precision: diagonal_precision(1, 1.0),
            },
            system,
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: COUPLING_INPUT_NAME.to_string(),
                operator: SparseTripletMatrix::new(1, 1),
                prior: GaussianPriorSpec {
                    mean: vec![0.0],
                    precision: diagonal_precision(1, 1.0 / prior_variance),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            physical_measurements: Vec::new(),
            joint_measurements: drive_currents
                .iter()
                .enumerate()
                .map(|(index, &drive_current)| LinearPdeJointMeasurementSpec {
                    name: format!("calibration_scalar_{index}"),
                    state_operator: None,
                    latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
                        input_name: COUPLING_INPUT_NAME.to_string(),
                        operator: scalar_column_triplet(drive_current),
                    }],
                    observations: vec![drive_current * c_h_true],
                    bias: vec![0.0],
                    variance: observation_variance,
                })
                .collect(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: None,
            pde_precision: None,
        };
        let solver = LinearPdeUqSolverConfig {
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::ExactSolves,
                ..LinearPdeVarianceConfig::default()
            },
            precision_policy: LinearPdePrecisionPolicy::default(),
            log_diagnostics: false,
        };
        let result = solve_linear_pde_uq_with_config(&problem, &solver)
            .expect("scalar coupling calibration should solve");
        let posterior = result
            .latent_inputs
            .iter()
            .find(|input| input.name == COUPLING_INPUT_NAME)
            .expect("c_H posterior should be present");
        let precision = 1.0 / prior_variance
            + drive_currents
                .iter()
                .map(|drive_current| drive_current * drive_current / observation_variance)
                .sum::<f64>();
        let analytic_variance = 1.0 / precision;
        let analytic_mean = analytic_variance
            * drive_currents
                .iter()
                .map(|drive_current| {
                    drive_current * (drive_current * c_h_true) / observation_variance
                })
                .sum::<f64>();
        assert!((posterior.mean[0] - analytic_mean).abs() < 1e-12);
        assert!((posterior.variance[0] - analytic_variance).abs() < 1e-12);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn harmonic_coupling_calibration_operator_has_blocks_and_sensitivities() {
        let _lock = lock_feec_harmonic_tests();
        let mut toroidal_config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            use_mass_weighted_pde_residual: true,
            normalize_mass_weighted_pde_residual: true,
            sparse_prediction_training_fraction: 0.25,
            sparse_prediction_noise_seed: 12345,
            sparse_prediction_field_phi_count: 2,
            sparse_prediction_field_theta_count: 2,
            sparse_prediction_linked_flux_count: 2,
            sparse_prediction_local_flux_phi_count: 1,
            sparse_prediction_local_flux_theta_count: 1,
            sparse_prediction_ampere_loop_count: 2,
            ..ToroidalHarmonicBConfig::default()
        };
        toroidal_config.observation_layout =
            ToroidalObservationLayout::TopologySparseNoisyPrediction;
        toroidal_config.include_harmonic_projection_observation = false;
        toroidal_config.include_ampere_loop_observation = true;
        let workspace = build_workspace(&toroidal_config).expect("sparse workspace should build");
        let drive_currents = vec![0.8, 1.2];
        let training = topology_pushforward_training_observations(
            &workspace,
            StageProblemKind::TopologySparseNoisyObservations,
        );
        let heldout = sparse_comparison_heldout_observations(&workspace, &training);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let training_instances = coupling_observation_instances(
            &workspace,
            &toroidal_config,
            &drive_currents,
            &training,
            false,
            &mut rng,
        );
        let heldout_instances = coupling_observation_instances(
            &workspace,
            &toroidal_config,
            &drive_currents,
            &heldout,
            false,
            &mut rng,
        );
        let problem = build_coupling_calibration_problem(
            &workspace,
            &toroidal_config,
            &drive_currents,
            CouplingCalibrationStageKind::AmpereLoops,
            &training_instances,
            &heldout_instances,
            5.0 * workspace.topology_summary.source_harmonic_kappa.abs(),
        )
        .expect("coupling calibration problem should assemble");
        assert_eq!(
            problem.system.state_dimension(),
            drive_currents.len() * workspace.system.state_dimension()
        );
        assert_eq!(problem.uncertain_inputs.len(), 1);
        assert_eq!(problem.uncertain_inputs[0].name, COUPLING_INPUT_NAME);
        assert!(problem
            .joint_measurements
            .iter()
            .all(|measurement| !measurement.name.contains("eta_H")
                && !measurement.name.contains("harmonic_component")));
        for family in ["hall", "flux", AMPERE_LOOP_SENSOR_TYPE] {
            let sensitivity = training_instances
                .iter()
                .filter(|instance| instance.observation.sensor_type == family)
                .map(|instance| instance.drive_current * instance.observation.beta_operator_value)
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            assert!(
                sensitivity.is_finite() && sensitivity > 0.0,
                "missing finite nonzero c_H sensitivity for {family}"
            );
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn harmonic_coupling_calibration_smoke_recovers_coupling() {
        let _lock = lock_feec_harmonic_tests();
        let config = ToroidalHarmonicCouplingCalibrationConfig {
            output_dir: None,
            drive_currents: vec![0.8, 1.2],
            coupling_prior_std_scale: 5.0,
            toroidal: ToroidalHarmonicBConfig {
                output_dir: None,
                include_full_field_variance_maps: false,
                use_mass_weighted_pde_residual: true,
                normalize_mass_weighted_pde_residual: true,
                sparse_prediction_training_fraction: 0.25,
                sparse_prediction_noise_seed: 12345,
                sparse_prediction_field_phi_count: 2,
                sparse_prediction_field_theta_count: 2,
                sparse_prediction_linked_flux_count: 2,
                sparse_prediction_local_flux_phi_count: 1,
                sparse_prediction_local_flux_theta_count: 1,
                sparse_prediction_ampere_loop_count: 2,
                solver: LinearPdeUqSolverConfig {
                    variance: LinearPdeVarianceConfig {
                        mode: LinearPdeVarianceMode::Hutchinson,
                        num_variance_probes: 16,
                        variance_batch_count: 2,
                        rng_seed: 59,
                        local_rb_block_size: 8,
                    },
                    precision_policy: LinearPdePrecisionPolicy::default(),
                    log_diagnostics: false,
                },
                ..ToroidalHarmonicBConfig::default()
            },
        };
        let mut toroidal_config = config.toroidal.clone();
        toroidal_config.observation_layout =
            ToroidalObservationLayout::TopologySparseNoisyPrediction;
        toroidal_config.source_alpha_true = 1.0;
        toroidal_config.include_harmonic_projection_observation = false;
        toroidal_config.include_ampere_loop_observation = true;
        let workspace = build_workspace(&toroidal_config).expect("sparse workspace should build");
        assert_eq!(workspace.topology_summary.harmonic_2_dimension, 1);
        let training = topology_pushforward_training_observations(
            &workspace,
            StageProblemKind::TopologySparseNoisyObservations,
        );
        let heldout = sparse_comparison_heldout_observations(&workspace, &training);
        let coupling_prior_std = 5.0
            * workspace
                .topology_summary
                .source_harmonic_kappa
                .abs()
                .max(1e-14);
        let k4 = run_coupling_calibration_stage(
            "K4_ampere_loops",
            &config,
            &toroidal_config,
            &workspace,
            CouplingCalibrationStageKind::AmpereLoops,
            training,
            heldout,
            coupling_prior_std,
        )
        .expect("K4 harmonic coupling calibration should run");
        assert!(k4.summary.coupling_posterior_mean.is_finite());
        assert!(k4.summary.coupling_posterior_variance.is_finite());
        assert!(k4.summary.coupling_posterior_variance >= 0.0);
        assert!(
            k4.summary.coupling_abs_error < k4.summary.coupling_truth.abs(),
            "posterior c_H should be closer to truth than the broad zero-mean prior"
        );
        assert!(!k4.qois.is_empty());
        assert!(k4.qois.iter().all(|row| {
            row.mean.is_finite()
                && row.sd.is_finite()
                && row.sd >= 0.0
                && row.posterior_variance.is_finite()
                && row.posterior_variance >= 0.0
        }));
        assert!(k4.heldout_predictions.iter().all(|row| {
            row.prediction.is_finite()
                && row.residual.is_finite()
                && row.posterior_sd.is_nan()
                && row.standardized_residual.is_nan()
        }));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_beta_smoke_recovers_beta_better_than_prior() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            run_alpha_beta_stress: false,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 11,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_harmonic_b_recovery(&config)
            .expect("beta-only toroidal harmonic experiment should run");
        let beta_stage = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == "H3_joint_beta_recovery")
            .expect("H3 stage should be present");
        let beta_truth = result.topology_summary.beta_true;
        let beta_mean = beta_stage
            .summary
            .beta_posterior_mean
            .expect("beta posterior should be present");
        let beta_variance = beta_stage
            .summary
            .beta_posterior_variance
            .expect("beta posterior variance should be present");
        assert!(beta_variance.is_finite() && beta_variance >= 0.0);
        assert!(beta_stage
            .solve
            .posterior_variance
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        assert!(
            (beta_mean - beta_truth).abs() < beta_truth.abs(),
            "posterior beta should be closer to truth than the zero prior mean"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_field_only_beta_smoke_reports_finite_variances() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 31,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_harmonic_b_field_only_beta_recovery(&config)
            .expect("field-only beta recovery experiment should run");
        let stage = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == FIELD_ONLY_BETA_STAGE_NAME)
            .expect("field-only beta stage should be present");
        assert!(stage
            .observations
            .iter()
            .all(|row| row.sensor_type == "hall" || row.sensor_type == "flux"));
        let beta_mean = stage
            .summary
            .beta_posterior_mean
            .expect("beta posterior should be present");
        let beta_variance = stage
            .summary
            .beta_posterior_variance
            .expect("beta posterior variance should be present");
        assert!(beta_mean.is_finite());
        assert!(beta_variance.is_finite() && beta_variance >= 0.0);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_embedded_field_beta_recovers_without_harmonic_projection() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 37,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_harmonic_b_embedded_field_beta_recovery(&config)
            .expect("embedded field beta recovery experiment should run");
        let stage = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == EMBEDDED_FIELD_BETA_STAGE_NAME)
            .expect("embedded field beta stage should be present");
        assert!(stage
            .observations
            .iter()
            .all(|row| row.sensor_type == "hall" || row.sensor_type == "flux"));
        assert_eq!(
            stage
                .observations
                .iter()
                .filter(|row| row.sensor_type == "hall")
                .count(),
            48
        );
        assert_eq!(
            stage
                .observations
                .iter()
                .filter(|row| row.sensor_type == "flux")
                .count(),
            4
        );
        let beta_truth = result.topology_summary.beta_true;
        let beta_mean = stage
            .summary
            .beta_posterior_mean
            .expect("beta posterior should be present");
        let beta_variance = stage
            .summary
            .beta_posterior_variance
            .expect("beta posterior variance should be present");
        assert!(beta_variance.is_finite() && beta_variance >= 0.0);
        assert!(
            (beta_mean - beta_truth).abs() < 0.01 * beta_truth.abs(),
            "embedded field sensors should recover beta to within 1%, got mean {beta_mean} truth {beta_truth}"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_source_generated_harmonic_recovers_alpha_without_harmonic_projection() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 41,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_source_generated_harmonic_recovery(&config)
            .expect("source-generated harmonic experiment should run");
        assert!(result.topology_summary.ampere_loop_harmonic_sensitivity > 0.0);
        assert!(
            (result.topology_summary.source_harmonic_energy_fraction - config.beta_energy_fraction)
                .abs()
                < 1e-12
        );
        let sg2 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == SOURCE_GENERATED_FIELD_STAGE_NAME)
            .expect("SG2 field recovery stage should be present");
        let sg3 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == SOURCE_GENERATED_AMPERE_STAGE_NAME)
            .expect("SG3 Ampere stage should be present");
        assert!(sg2
            .observations
            .iter()
            .all(|row| row.sensor_type == "hall" || row.sensor_type == "flux"));
        let alpha_mean = sg2
            .summary
            .alpha_posterior_mean
            .expect("SG2 alpha posterior should be present");
        let alpha_variance = sg2
            .summary
            .alpha_posterior_variance
            .expect("SG2 alpha posterior variance should be present");
        assert!(alpha_mean.is_finite());
        assert!(alpha_variance.is_finite() && alpha_variance >= 0.0);
        assert!(
            (alpha_mean - config.source_alpha_true).abs() < (1.0 - config.source_alpha_true).abs(),
            "SG2 posterior alpha should be closer to truth than the prior mean"
        );
        let sg3_variance = sg3
            .summary
            .alpha_posterior_variance
            .expect("SG3 alpha posterior variance should be present");
        assert!(sg3_variance.is_finite() && sg3_variance >= 0.0);
        assert!(
            sg3_variance <= alpha_variance,
            "Ampere loop should not increase alpha uncertainty"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_source_generated_harmonic_full_state_reports_finite_alpha() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            use_mass_weighted_pde_residual: true,
            normalize_mass_weighted_pde_residual: true,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 43,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_source_generated_harmonic_full_state_recovery(&config)
            .expect("full-state source-generated harmonic experiment should run");
        let sgf2 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == SOURCE_GENERATED_FULL_STATE_FIELD_STAGE_NAME)
            .expect("SGF2 field recovery stage should be present");
        let sgf3 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == SOURCE_GENERATED_FULL_STATE_AMPERE_STAGE_NAME)
            .expect("SGF3 Ampere stage should be present");
        assert!(sgf2
            .observations
            .iter()
            .all(|row| row.sensor_type == "hall" || row.sensor_type == "flux"));
        for stage in [sgf2, sgf3] {
            let alpha_mean = stage
                .summary
                .alpha_posterior_mean
                .expect("alpha posterior should be present");
            let alpha_variance = stage
                .summary
                .alpha_posterior_variance
                .expect("alpha posterior variance should be present");
            assert!(alpha_mean.is_finite());
            assert!(alpha_variance.is_finite() && alpha_variance >= 0.0);
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_topology_pushforward_reports_physical_qoi_uncertainty() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            use_mass_weighted_pde_residual: true,
            normalize_mass_weighted_pde_residual: true,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 47,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_topology_pushforward_uq(&config)
            .expect("topology-aware pushforward experiment should run");
        assert_eq!(result.topology_summary.harmonic_2_dimension, 1);
        assert!((result.topology_summary.harmonic_2_mass_norm - 1.0).abs() < 1e-8);
        let s2 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == TOPOLOGY_PUSHFORWARD_FIELD_STAGE_NAME)
            .expect("S2 should be present");
        let s3 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == TOPOLOGY_PUSHFORWARD_FLUX_STAGE_NAME)
            .expect("S3 should be present");
        let s4 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == TOPOLOGY_PUSHFORWARD_AMPERE_STAGE_NAME)
            .expect("S4 should be present");
        assert!(
            s2.observations
                .iter()
                .filter(|row| row.sensor_type == "harmonic")
                .count()
                == 0
        );
        assert!(s4
            .observations
            .iter()
            .any(|row| row.sensor_type == AMPERE_LOOP_SENSOR_TYPE));
        for stage in [&result.stages[0], s2, s3, s4] {
            assert_eq!(stage.pushforward_qois.len(), 9);
            assert!(!stage.heldout_predictions.is_empty());
            assert!(!stage.branch_decomposition.is_empty());
            assert!(!stage.field_trace_variance.is_empty());
            assert!(stage.pushforward_qois.iter().all(|row| {
                row.mean.is_finite()
                    && row.sd.is_finite()
                    && row.sd >= 0.0
                    && row.posterior_variance.is_finite()
                    && row.posterior_variance >= 0.0
            }));
            assert!(stage.pushforward_covariances.iter().all(|row| {
                row.prior_covariance.is_finite() && row.posterior_covariance.is_finite()
            }));
        }
        let eta_s2 = s2
            .pushforward_qois
            .iter()
            .find(|row| row.qoi == QOI_HARMONIC_PROJECTION_NAME)
            .expect("eta_H should be reported");
        let eta_s4 = s4
            .pushforward_qois
            .iter()
            .find(|row| row.qoi == QOI_HARMONIC_PROJECTION_NAME)
            .expect("eta_H should be reported");
        assert!(
            (eta_s4.mean - eta_s4.truth).abs() < (eta_s2.truth - 0.0).abs(),
            "posterior harmonic projection should be finite and truth-directed"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_topology_sparse_noisy_prediction_reports_heldout_coverage() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            use_mass_weighted_pde_residual: true,
            normalize_mass_weighted_pde_residual: true,
            source_alpha_true: 1.0,
            sparse_prediction_training_fraction: 0.10,
            sparse_prediction_noise_seed: 12345,
            sparse_prediction_field_phi_count: 4,
            sparse_prediction_field_theta_count: 4,
            sparse_prediction_linked_flux_count: 4,
            sparse_prediction_local_flux_phi_count: 2,
            sparse_prediction_local_flux_theta_count: 2,
            sparse_prediction_ampere_loop_count: 4,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 53,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_topology_sparse_noisy_prediction(&config)
            .expect("sparse noisy topology prediction experiment should run");
        let n2 = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == TOPOLOGY_SPARSE_NOISY_OBS_STAGE_NAME)
            .expect("N2 should be present");
        assert_eq!(result.topology_summary.harmonic_2_dimension, 1);
        assert!(n2
            .observations
            .iter()
            .any(|row| row.sensor_type == AMPERE_LOOP_SENSOR_TYPE));
        assert!(n2.heldout_predictions.len() > n2.observations.len());
        assert!(n2.heldout_predictions.iter().all(|row| {
            row.posterior_sd.is_finite()
                && row.posterior_sd >= 0.0
                && row.standardized_residual.is_finite()
        }));
        let covered = n2
            .heldout_predictions
            .iter()
            .filter(|row| row.covered95)
            .count();
        assert!(
            covered * 100 >= 80 * n2.heldout_predictions.len(),
            "held-out coverage should be broadly calibrated for the synthetic nominal run"
        );
        let source = n2
            .pushforward_qois
            .iter()
            .find(|row| row.qoi == QOI_SOURCE_NAME)
            .expect("source QoI should be reported");
        assert!((source.mean - 1.0).abs() < 5.0 * source.sd.max(EPS));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_fluctuation_smoke_reports_finite_alpha_and_beta() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            fluctuation_state_prior_precision_scale: 1.0e8,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 19,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_harmonic_b_fluctuation_recovery(&config)
            .expect("fluctuation toroidal harmonic experiment should run");
        let stage = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == FLUCTUATION_ALPHA_BETA_STAGE_NAME)
            .expect("fluctuation stage should be present");
        let beta_truth = result.topology_summary.beta_true;
        let beta_mean = stage
            .summary
            .beta_posterior_mean
            .expect("beta posterior should be present");
        let beta_variance = stage
            .summary
            .beta_posterior_variance
            .expect("beta posterior variance should be present");
        let alpha_mean = stage
            .summary
            .alpha_posterior_mean
            .expect("alpha posterior should be present");
        let alpha_variance = stage
            .summary
            .alpha_posterior_variance
            .expect("alpha posterior variance should be present");
        assert!(beta_variance.is_finite() && beta_variance >= 0.0);
        assert!(alpha_mean.is_finite());
        assert!(alpha_variance.is_finite() && alpha_variance >= 0.0);
        assert!(
            (beta_mean - beta_truth).abs() < beta_truth.abs(),
            "posterior beta should be closer to truth than the zero prior mean"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_ampere_loop_operator_reports_finite_response() {
        let _lock = lock_feec_harmonic_tests();
        let config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_ampere_loop_observation: true,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        let workspace = build_workspace(&config).expect("workspace should build");
        let observation = workspace
            .observations
            .iter()
            .find(|observation| observation.sensor_type == AMPERE_LOOP_SENSOR_TYPE)
            .expect("Ampere-loop observation should be present");
        let nonzeros = observation
            .state_operator
            .triplet_iter()
            .collect::<Vec<_>>();
        assert!(!nonzeros.is_empty(), "Ampere-loop row must be nonempty");
        assert!(nonzeros.iter().all(|(_, _, value)| value.is_finite()));
        let nominal = apply_triplet_row(&observation.state_operator, &workspace.truth_a_nominal)
            .expect("nominal loop response should evaluate");
        assert!(nominal.is_finite() && nominal.abs() > EPS);
        assert!(observation.beta_operator_value.is_finite());
        let expected_alpha_delta = (config.source_alpha_true - 1.0) * nominal;
        let actual_alpha_delta =
            observation.observation_alpha_beta_truth - observation.observation_beta_truth;
        assert!(
            (actual_alpha_delta - expected_alpha_delta).abs()
                <= 1e-8 * expected_alpha_delta.abs().max(1.0),
            "Ampere-loop truth delta should match alpha-scaled nominal response"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_ampere_loop_smoke_reports_finite_alpha_and_beta() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            fluctuation_state_prior_precision_scale: 1.0e8,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 23,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_harmonic_b_ampere_loop_recovery(&config)
            .expect("Ampere-loop toroidal harmonic experiment should run");
        let stage = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == AMPERE_LOOP_ALPHA_BETA_STAGE_NAME)
            .expect("Ampere-loop stage should be present");
        assert!(stage
            .observations
            .iter()
            .any(|row| row.sensor_type == AMPERE_LOOP_SENSOR_TYPE));
        let beta_truth = result.topology_summary.beta_true;
        let beta_mean = stage
            .summary
            .beta_posterior_mean
            .expect("beta posterior should be present");
        let beta_variance = stage
            .summary
            .beta_posterior_variance
            .expect("beta posterior variance should be present");
        let alpha_mean = stage
            .summary
            .alpha_posterior_mean
            .expect("alpha posterior should be present");
        let alpha_variance = stage
            .summary
            .alpha_posterior_variance
            .expect("alpha posterior variance should be present");
        assert!(beta_variance.is_finite() && beta_variance >= 0.0);
        assert!(alpha_mean.is_finite());
        assert!(alpha_variance.is_finite() && alpha_variance >= 0.0);
        assert!(
            (beta_mean - beta_truth).abs() < beta_truth.abs(),
            "posterior beta should be closer to truth than the zero prior mean"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_harmonic_full_state_ampere_loop_smoke_reports_finite_alpha_and_beta() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.variance = LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            num_variance_probes: 8,
            variance_batch_count: 2,
            rng_seed: 29,
            local_rb_block_size: 8,
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_harmonic_b_full_state_ampere_loop_recovery(&config)
            .expect("full-state Ampere-loop toroidal harmonic experiment should run");
        let stage = result
            .stages
            .iter()
            .find(|stage| stage.summary.stage == FULL_STATE_AMPERE_LOOP_ALPHA_BETA_STAGE_NAME)
            .expect("full-state Ampere-loop stage should be present");
        assert!(stage
            .observations
            .iter()
            .any(|row| row.sensor_type == AMPERE_LOOP_SENSOR_TYPE));
        let beta_truth = result.topology_summary.beta_true;
        let beta_mean = stage
            .summary
            .beta_posterior_mean
            .expect("beta posterior should be present");
        let beta_variance = stage
            .summary
            .beta_posterior_variance
            .expect("beta posterior variance should be present");
        let alpha_mean = stage
            .summary
            .alpha_posterior_mean
            .expect("alpha posterior should be present");
        let alpha_variance = stage
            .summary
            .alpha_posterior_variance
            .expect("alpha posterior variance should be present");
        assert!(beta_variance.is_finite() && beta_variance >= 0.0);
        assert!(alpha_mean.is_finite());
        assert!(alpha_variance.is_finite() && alpha_variance >= 0.0);
        assert!(
            (beta_mean - beta_truth).abs() < beta_truth.abs(),
            "posterior beta should be closer to truth than the zero prior mean"
        );
    }
}
