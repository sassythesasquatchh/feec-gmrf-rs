//! 2D annulus H-formulation topology benchmark.
//!
//! The unknown is a discrete 1-form on `1 < r < 2`.  The experiment compares
//! the full FEEC-GMRF model with reusable baselines on local line observations,
//! local curl/residual observations, and global circulation.

use crate::{
    annulus_baselines::{
        build_componentwise_gp_correction_model, build_componentwise_gp_model,
        build_exact_only_feec_model, build_feec_gmrf_model,
        build_feec_split_no_spectral_correction_model, build_scalar_potential_gp_correction_model,
        build_scalar_potential_gp_model, AnnulusComponentwiseGpConfig, AnnulusLinearModel,
        AnnulusModelKind, AnnulusPotentialPriorConfig, AnnulusScalarPotentialGpConfig,
    },
    de_rham, visual_output,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::{cochain::Cochain, whitney::lsf::WhitneyLsf};
use exterior::field::ExteriorField;
use feec_gmrf::prelude::{
    sparse_mat_from_feec_csr, GaussianNoise, GaussianPrior, LinearGaussianModelBuilder, LinearMap,
    LinearObservation, Posterior, VarianceMethod,
};
use feg_infer::{
    conditioning::linear::{feec_observation_matrix, FeecWeightedObservationRow},
    prior::hodge::{compute_harmonic_basis_1form, mass_orthonormalize_harmonic_basis_1form},
};
use manifold::{
    geometry::{
        coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
        metric::mesh::MeshLengths,
    },
    io::gmsh::gmsh2coord_complex,
    topology::complex::Complex,
};
use plotters::prelude::*;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

const INNER_RADIUS: f64 = 1.0;
const OUTER_RADIUS: f64 = 2.0;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const EPS: f64 = 1e-12;
const VARIANCE_FLOOR: f64 = 1e-14;
const MESH_VARIANCE_QOI_FAMILIES: [(&str, f64); 9] = [
    ("q_period_variance", 0.02),
    ("circulation_mean_variance", 0.02),
    ("short_line_median_variance", 0.10),
    ("long_line_median_variance", 0.10),
    ("dense_line_away_median_variance", 0.10),
    ("dense_line_away_p90_variance", 0.10),
    ("field_x_away_median_variance", 0.10),
    ("field_y_away_median_variance", 0.10),
    ("field_magnitude_away_median_variance", 0.10),
];
const PRIMARY_MESH_QOI_FAMILIES: [&str; 6] = [
    "q_period_variance",
    "circulation_mean_variance",
    "dense_line_away_median_variance",
    "field_x_away_median_variance",
    "field_y_away_median_variance",
    "field_magnitude_away_median_variance",
];

#[derive(Debug, Clone)]
pub struct AnnulusHFormulationConfig {
    pub mesh_path: PathBuf,
    pub geo_path: PathBuf,
    pub output_dir: PathBuf,
    pub force_mesh: bool,
    pub mesh_size: f64,
    pub tau0: f64,
    pub tau1: f64,
    pub harmonic_prior_std: f64,
    pub line_noise_variance: f64,
    pub residual_noise_variance: f64,
    pub circulation_noise_variance: f64,
    pub line_tangential_count: usize,
    pub line_radial_count: usize,
    pub line_random_count: usize,
    pub residual_count: usize,
    pub validation_line_count: usize,
    pub validation_residual_count: usize,
    pub heldout_line_count: usize,
    pub heldout_residual_count: usize,
    pub heldout_loop_count: usize,
    pub noise_trial_count: usize,
    pub sample_observation_noise: bool,
    pub rng_seed: u64,
    pub potential_tau0_grid: Vec<f64>,
    pub potential_tau1_grid: Vec<f64>,
    pub component_kappa_grid: Vec<f64>,
    pub component_tau_grid: Vec<f64>,
    pub scalar_potential_kappa_grid: Vec<f64>,
    pub scalar_potential_tau_grid: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnnulusMeshInvarianceProfile {
    Quick,
    Thesis,
}

impl AnnulusMeshInvarianceProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Thesis => "thesis",
        }
    }

    pub fn default_mesh_sizes(self) -> Vec<f64> {
        match self {
            Self::Quick => vec![0.42, 0.36, 0.30],
            Self::Thesis => vec![0.24, 0.18, 0.135, 0.10, 0.075, 0.055],
        }
    }
}

impl std::str::FromStr for AnnulusMeshInvarianceProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "quick" => Ok(Self::Quick),
            "thesis" => Ok(Self::Thesis),
            other => Err(format!(
                "unknown annulus mesh-invariance profile `{other}`; expected quick or thesis"
            )),
        }
    }
}

impl Default for AnnulusHFormulationConfig {
    fn default() -> Self {
        let output_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/annulus_h_formulation");
        Self {
            mesh_path: output_dir.join("annulus_h_formulation.msh"),
            geo_path: output_dir.join("annulus_h_formulation.geo"),
            output_dir,
            force_mesh: true,
            mesh_size: 0.12,
            tau0: 1e-3,
            tau1: 1.0,
            harmonic_prior_std: 2.0,
            line_noise_variance: 1e-4,
            residual_noise_variance: 1e-6,
            circulation_noise_variance: 1e-4,
            line_tangential_count: 40,
            line_radial_count: 30,
            line_random_count: 30,
            residual_count: 500,
            validation_line_count: 60,
            validation_residual_count: 120,
            heldout_line_count: 120,
            heldout_residual_count: 200,
            heldout_loop_count: 16,
            noise_trial_count: 10,
            sample_observation_noise: true,
            rng_seed: 20240519,
            potential_tau0_grid: vec![1e-3],
            potential_tau1_grid: vec![1.0],
            component_kappa_grid: vec![1.0, 2.0, 4.0],
            component_tau_grid: vec![0.5, 1.0],
            scalar_potential_kappa_grid: vec![1.0, 2.0, 4.0],
            scalar_potential_tau_grid: vec![0.5, 1.0],
        }
    }
}

impl AnnulusHFormulationConfig {
    /// Cheap deterministic configuration intended for continuous integration.
    pub fn smoke(output_dir: PathBuf) -> Self {
        Self {
            mesh_path: output_dir.join("annulus_h_formulation.msh"),
            geo_path: output_dir.join("annulus_h_formulation.geo"),
            output_dir,
            mesh_size: 0.45,
            line_tangential_count: 4,
            line_radial_count: 3,
            line_random_count: 3,
            residual_count: 12,
            validation_line_count: 6,
            validation_residual_count: 6,
            heldout_line_count: 8,
            heldout_residual_count: 8,
            heldout_loop_count: 3,
            noise_trial_count: 1,
            component_kappa_grid: vec![2.0],
            component_tau_grid: vec![1.0],
            scalar_potential_kappa_grid: vec![2.0],
            scalar_potential_tau_grid: vec![1.0],
            ..Self::default()
        }
    }

    /// Immutable configuration used by the submitted thesis.
    pub fn thesis_submitted(output_dir: PathBuf) -> Self {
        Self {
            mesh_path: output_dir.join("annulus_h_formulation.msh"),
            geo_path: output_dir.join("annulus_h_formulation.geo"),
            output_dir,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnnulusRegime {
    A,
    B,
    C,
    D,
}

impl AnnulusRegime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::A => "A_local",
            Self::B => "B_local_residual",
            Self::C => "C_local_circulation",
            Self::D => "D_local_residual_circulation",
        }
    }

    fn all() -> [Self; 4] {
        [Self::A, Self::B, Self::C, Self::D]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnnulusObservationKind {
    TrainLine,
    TrainResidual,
    TrainCirculation,
    ValidationLine,
    ValidationResidual,
    HeldoutLine,
    HeldoutResidual,
    HeldoutCirculation,
    HeldoutHomologousDifference,
    DenseFieldX,
    DenseFieldY,
}

impl AnnulusObservationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TrainLine => "train_line",
            Self::TrainResidual => "train_residual",
            Self::TrainCirculation => "train_circulation",
            Self::ValidationLine => "validation_line",
            Self::ValidationResidual => "validation_residual",
            Self::HeldoutLine => "heldout_line",
            Self::HeldoutResidual => "heldout_residual",
            Self::HeldoutCirculation => "heldout_circulation",
            Self::HeldoutHomologousDifference => "heldout_homologous_difference",
            Self::DenseFieldX => "dense_field_x",
            Self::DenseFieldY => "dense_field_y",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnulusTopologySummary {
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub harmonic_1_dimension: usize,
    pub reference_period_truth: f64,
    pub reference_period_psi: f64,
    pub truth_closure_l2: f64,
    pub truth_closure_max_abs: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusMetricRow {
    pub trial: usize,
    pub regime: AnnulusRegime,
    pub model: AnnulusModelKind,
    pub training_observation_count: usize,
    pub selected_kappa: f64,
    pub selected_tau0: f64,
    pub selected_tau1: f64,
    pub validation_nlpd: f64,
    pub rmse_line: f64,
    pub residual_rmse: f64,
    pub circ_rmse: f64,
    pub homologous_difference_rmse: f64,
    pub z_mean: f64,
    pub z_variance: f64,
    pub coverage_90: f64,
    pub coverage_95: f64,
    pub nlpd: f64,
    pub circ_nlpd: f64,
    pub topo_spread: f64,
    pub q_mean: f64,
    pub q_std: f64,
    pub selected_build_seconds: f64,
    pub selection_seconds: f64,
    pub conditioning_seconds: f64,
    pub prediction_seconds: f64,
    pub selected_total_seconds: f64,
    pub pipeline_total_seconds: f64,
    pub latent_dimension: usize,
    pub prior_precision_nnz: usize,
    pub prior_precision_density: f64,
    pub posterior_precision_nnz: usize,
    pub posterior_precision_density: f64,
    pub posterior_factor_nnz: usize,
    pub posterior_fill_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusSummaryRow {
    pub regime: AnnulusRegime,
    pub model: AnnulusModelKind,
    pub trial_count: usize,
    pub rmse_line: f64,
    pub residual_rmse: f64,
    pub circ_rmse: f64,
    pub homologous_difference_rmse: f64,
    pub z_mean: f64,
    pub z_variance: f64,
    pub coverage_90: f64,
    pub coverage_95: f64,
    pub nlpd: f64,
    pub circ_nlpd: f64,
    pub topo_spread: f64,
    pub q_mean: f64,
    pub q_std: f64,
    pub selected_build_seconds: f64,
    pub selection_seconds: f64,
    pub conditioning_seconds: f64,
    pub prediction_seconds: f64,
    pub selected_total_seconds: f64,
    pub pipeline_total_seconds: f64,
    pub latent_dimension: f64,
    pub prior_precision_nnz: f64,
    pub prior_precision_density: f64,
    pub posterior_precision_nnz: f64,
    pub posterior_precision_density: f64,
    pub posterior_factor_nnz: f64,
    pub posterior_fill_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusPredictionRow {
    pub trial: usize,
    pub regime: AnnulusRegime,
    pub model: AnnulusModelKind,
    pub kind: AnnulusObservationKind,
    pub label: String,
    pub truth: f64,
    pub mean: f64,
    pub latent_variance: f64,
    pub predictive_variance: f64,
    pub z: f64,
    pub nlpd: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusTuningRow {
    pub model: AnnulusModelKind,
    pub selected: bool,
    pub kappa: f64,
    pub tau0: f64,
    pub tau1: f64,
    pub validation_nlpd: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusPosteriorField {
    pub regime: AnnulusRegime,
    pub model: AnnulusModelKind,
    pub posterior_mean: FeecVector,
}

#[derive(Debug, Clone)]
pub struct AnnulusHFormulationResult {
    pub topology: AnnulusTopologySummary,
    pub trial_metrics: Vec<AnnulusMetricRow>,
    pub summary_rows: Vec<AnnulusSummaryRow>,
    pub predictions: Vec<AnnulusPredictionRow>,
    pub tuning_rows: Vec<AnnulusTuningRow>,
    pub truth_h: FeecVector,
    pub psi: FeecVector,
    pub posterior_fields: Vec<AnnulusPosteriorField>,
}

#[derive(Debug, Clone)]
pub struct AnnulusMeshInvarianceConfig {
    pub profile: AnnulusMeshInvarianceProfile,
    pub mesh_sizes: Vec<f64>,
    pub output_dir: PathBuf,
    pub residual_count: usize,
    pub heldout_loop_count: usize,
    pub sample_observation_noise: bool,
    pub model_kinds: Vec<AnnulusModelKind>,
    pub convergence_tail_count: usize,
    pub base: AnnulusHFormulationConfig,
}

impl Default for AnnulusMeshInvarianceConfig {
    fn default() -> Self {
        let output_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/annulus_h_mesh_invariance");
        let base = AnnulusHFormulationConfig {
            sample_observation_noise: false,
            noise_trial_count: 1,
            ..AnnulusHFormulationConfig::default()
        };
        let profile = AnnulusMeshInvarianceProfile::Thesis;
        Self {
            profile,
            mesh_sizes: profile.default_mesh_sizes(),
            output_dir,
            residual_count: base.residual_count,
            heldout_loop_count: base.heldout_loop_count,
            sample_observation_noise: false,
            model_kinds: vec![
                AnnulusModelKind::FeecGmrf,
                AnnulusModelKind::FeecSplitNoSpectralCorrection,
            ],
            convergence_tail_count: 3,
            base,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnulusEfficiencyConfig {
    pub mesh_sizes: Vec<f64>,
    pub output_dir: PathBuf,
    pub sample_observation_noise: bool,
    pub dense_gp_vertex_limit: Option<usize>,
    pub base: AnnulusHFormulationConfig,
}

impl Default for AnnulusEfficiencyConfig {
    fn default() -> Self {
        let output_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/annulus_h_efficiency");
        let base = AnnulusHFormulationConfig {
            sample_observation_noise: false,
            noise_trial_count: 1,
            ..AnnulusHFormulationConfig::default()
        };
        Self {
            mesh_sizes: vec![0.24, 0.18, 0.12, 0.09, 0.0675, 0.050625],
            output_dir,
            sample_observation_noise: false,
            dense_gp_vertex_limit: None,
            base,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnulusEfficiencyRow {
    pub mesh_index: usize,
    pub mesh_size: f64,
    pub model: AnnulusModelKind,
    pub status: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub selected_kappa: f64,
    pub selected_tau0: f64,
    pub selected_tau1: f64,
    pub build_seconds: f64,
    pub conditioning_seconds: f64,
    pub prediction_seconds: f64,
    pub selected_total_seconds: f64,
    pub latent_dimension: usize,
    pub prior_precision_nnz: usize,
    pub prior_precision_density: f64,
    pub posterior_precision_nnz: usize,
    pub posterior_precision_density: f64,
    pub posterior_factor_nnz: usize,
    pub posterior_fill_ratio: f64,
    pub rmse_line: f64,
    pub residual_rmse: f64,
    pub circ_rmse: f64,
    pub homologous_difference_rmse: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusEfficiencySpeedupRow {
    pub mesh_index: usize,
    pub mesh_size: f64,
    pub vertex_count: usize,
    pub gp_selected_total_seconds: f64,
    pub feec_selected_total_seconds: f64,
    pub speedup: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusEfficiencyResult {
    pub rows: Vec<AnnulusEfficiencyRow>,
    pub speedup_rows: Vec<AnnulusEfficiencySpeedupRow>,
}

#[derive(Debug, Clone)]
pub struct AnnulusMeshMetadataRow {
    pub mesh_index: usize,
    pub mesh_size: f64,
    pub model: AnnulusModelKind,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub harmonic_1_dimension: usize,
    pub reference_period_truth: f64,
    pub reference_period_psi: f64,
    pub truth_closure_l2: f64,
    pub truth_closure_max_abs: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusMeshQoiRow {
    pub mesh_index: usize,
    pub mesh_size: f64,
    pub model: AnnulusModelKind,
    pub regime: AnnulusRegime,
    pub qoi_kind: String,
    pub qoi_label: String,
    pub truth: f64,
    pub mean: f64,
    pub latent_variance: f64,
    pub posterior_sd: f64,
    pub support_entries: usize,
    pub direct_support_overlap_count: usize,
    pub is_away_probe: bool,
    pub probe_family: String,
}

#[derive(Debug, Clone)]
pub struct AnnulusMeshInvarianceSummaryRow {
    pub profile: AnnulusMeshInvarianceProfile,
    pub mesh_index: usize,
    pub mesh_size: f64,
    pub model: AnnulusModelKind,
    pub regime: AnnulusRegime,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub q_period_variance: f64,
    pub q_period_sd: f64,
    pub circulation_count: usize,
    pub circulation_mean_variance: f64,
    pub circulation_median_variance: f64,
    pub circulation_mean_sd: f64,
    pub circulation_median_sd: f64,
    pub line_count: usize,
    pub line_mean_variance: f64,
    pub line_median_variance: f64,
    pub line_mean_sd: f64,
    pub line_median_sd: f64,
    pub short_line_count: usize,
    pub short_line_mean_variance: f64,
    pub short_line_median_variance: f64,
    pub short_line_mean_sd: f64,
    pub short_line_median_sd: f64,
    pub long_line_count: usize,
    pub long_line_mean_variance: f64,
    pub long_line_median_variance: f64,
    pub long_line_mean_sd: f64,
    pub long_line_median_sd: f64,
    pub dense_line_away_count: usize,
    pub dense_line_away_median_variance: f64,
    pub dense_line_away_p90_variance: f64,
    pub dense_line_away_median_sd: f64,
    pub dense_line_away_p90_sd: f64,
    pub field_x_away_count: usize,
    pub field_x_away_median_variance: f64,
    pub field_x_away_median_sd: f64,
    pub field_y_away_count: usize,
    pub field_y_away_median_variance: f64,
    pub field_y_away_median_sd: f64,
    pub field_magnitude_away_count: usize,
    pub field_magnitude_away_median_variance: f64,
    pub field_magnitude_away_median_sd: f64,
    pub homologous_count: usize,
    pub homologous_mean_variance: f64,
    pub homologous_median_variance: f64,
    pub homologous_mean_sd: f64,
    pub homologous_median_sd: f64,
    pub max_relative_change_to_finest_variance: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusMeshInvarianceFitRow {
    pub profile: AnnulusMeshInvarianceProfile,
    pub model: AnnulusModelKind,
    pub regime: AnnulusRegime,
    pub qoi_family: String,
    pub coarser_mesh_size: f64,
    pub finest_mesh_size: f64,
    pub coarser_value: f64,
    pub finest_value: f64,
    pub relative_change: f64,
    pub coarse_to_finest_relative_change: f64,
    pub slope_log_variance_vs_log_h: f64,
    pub tail_relative_spread: f64,
    pub tail_count: usize,
    pub model_ratio_to_full: f64,
    pub threshold: f64,
    pub trend: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct AnnulusMeshModelContrastRow {
    pub profile: AnnulusMeshInvarianceProfile,
    pub mesh_index: usize,
    pub mesh_size: f64,
    pub regime: AnnulusRegime,
    pub qoi_family: String,
    pub no_correction_variance: f64,
    pub full_variance: f64,
    pub ratio_no_correction_to_full: f64,
}

#[derive(Debug, Clone)]
pub struct AnnulusMeshInvarianceResult {
    pub mesh_metadata: Vec<AnnulusMeshMetadataRow>,
    pub qoi_rows: Vec<AnnulusMeshQoiRow>,
    pub summary_rows: Vec<AnnulusMeshInvarianceSummaryRow>,
    pub fit_rows: Vec<AnnulusMeshInvarianceFitRow>,
    pub contrast_rows: Vec<AnnulusMeshModelContrastRow>,
}

#[derive(Debug, Clone)]
struct AnnulusSupport {
    label: String,
    entries: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
struct AnnulusProbeMetadata {
    direct_support_overlap_count: usize,
    is_away_probe: bool,
    probe_family: String,
}

#[derive(Debug, Clone)]
struct AnnulusFieldSamplePoint {
    label: String,
    point: [f64; 3],
}

#[derive(Debug, Clone)]
struct AnnulusObservation {
    kind: AnnulusObservationKind,
    label: String,
    entries: Vec<(usize, f64)>,
    truth_value: f64,
    observed_value: f64,
    noise_variance: f64,
}

#[derive(Debug, Clone)]
struct SelectedModel {
    model: AnnulusLinearModel,
    validation_nlpd: f64,
    selected_build_seconds: f64,
    selection_seconds: f64,
}

#[derive(Debug, Clone)]
struct AnnulusModelCandidate {
    model: AnnulusLinearModel,
    build_seconds: f64,
}

struct ConditionedAnnulusModel {
    latent_to_h: LinearMap,
    posterior: Posterior,
    posterior_precision_nnz: usize,
    posterior_factor_nnz: usize,
    posterior_fill_ratio: f64,
    h_mean: FeecVector,
    q_mean: f64,
    q_std: f64,
}

struct AnnulusWorkspace {
    topology: Complex,
    coords: MeshCoords,
    metric: MeshLengths,
    mass_1form: FeecCsr,
    reference_loop: AnnulusSupport,
    heldout_loop_supports: Vec<AnnulusSupport>,
    truth_h: FeecVector,
    psi: FeecVector,
    topology_summary: AnnulusTopologySummary,
}

pub fn run_annulus_h_formulation(
    config: &AnnulusHFormulationConfig,
) -> Result<AnnulusHFormulationResult, Box<dyn Error>> {
    validate_config(config)?;
    let workspace = build_workspace(config)?;
    let observation_sets = build_observation_sets(config, &workspace)?;

    let validation_train = observation_sets
        .train_line
        .iter()
        .chain(observation_sets.train_residual.iter())
        .cloned()
        .collect::<Vec<_>>();
    let validation_heldout = observation_sets
        .validation_line
        .iter()
        .chain(observation_sets.validation_residual.iter())
        .cloned()
        .collect::<Vec<_>>();

    let mut tuning_rows = Vec::new();
    let selected_models = select_models(
        config,
        &workspace,
        &validation_train,
        &validation_heldout,
        &mut tuning_rows,
    )?;

    let heldout_observations = observation_sets.heldout_all();
    let mut trial_metrics = Vec::new();
    let mut predictions = Vec::new();
    let mut posterior_fields = Vec::new();

    for trial in 0..config.noise_trial_count {
        let noisy_line = sample_noisy_observations(
            &observation_sets.train_line,
            config.sample_observation_noise,
            config.rng_seed + 10_000 + trial as u64,
        );
        let noisy_residual = sample_noisy_observations(
            &observation_sets.train_residual,
            config.sample_observation_noise,
            config.rng_seed + 20_000 + trial as u64,
        );
        let noisy_circulation = sample_noisy_observations(
            &observation_sets.train_circulation,
            config.sample_observation_noise,
            config.rng_seed + 30_000 + trial as u64,
        );

        for regime in AnnulusRegime::all() {
            let training =
                training_for_regime(regime, &noisy_line, &noisy_residual, &noisy_circulation);
            for selected in &selected_models {
                let conditioning_start = Instant::now();
                let mut conditioned =
                    condition_model(&selected.model, &training, workspace.topology.nsimplices(1))?;
                let conditioning_seconds = conditioning_start.elapsed().as_secs_f64();
                let prediction_start = Instant::now();
                let mut model_predictions = predict_observables(
                    trial,
                    regime,
                    selected.model.model_kind,
                    &mut conditioned,
                    &heldout_observations,
                    workspace.topology.nsimplices(1),
                )?;
                let topo_spread =
                    compute_topo_spread(&mut conditioned, &workspace.heldout_loop_supports)?;
                let prediction_seconds = prediction_start.elapsed().as_secs_f64();
                let metric = compute_metric_row(
                    trial,
                    regime,
                    selected,
                    training.len(),
                    &conditioned,
                    &model_predictions,
                    topo_spread,
                    conditioning_seconds,
                    prediction_seconds,
                );
                if trial == 0 && regime == AnnulusRegime::D {
                    posterior_fields.push(AnnulusPosteriorField {
                        regime,
                        model: selected.model.model_kind,
                        posterior_mean: conditioned.h_mean.clone(),
                    });
                }
                predictions.append(&mut model_predictions);
                trial_metrics.push(metric);
            }
        }
    }

    let summary_rows = aggregate_summary_rows(&trial_metrics);
    Ok(AnnulusHFormulationResult {
        topology: workspace.topology_summary,
        trial_metrics,
        summary_rows,
        predictions,
        tuning_rows,
        truth_h: workspace.truth_h,
        psi: workspace.psi,
        posterior_fields,
    })
}

pub fn write_annulus_h_formulation_outputs(
    result: &AnnulusHFormulationResult,
    config: &AnnulusHFormulationConfig,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(&config.output_dir)?;
    write_topology_summary_csv(result, &config.output_dir.join("topology_summary.csv"))?;
    write_trial_metrics_csv(result, &config.output_dir.join("trial_metrics.csv"))?;
    write_summary_metrics_csv(result, &config.output_dir.join("metrics_summary.csv"))?;
    write_predictions_csv(result, &config.output_dir.join("heldout_predictions.csv"))?;
    write_tuning_csv(result, &config.output_dir.join("hyperparameter_tuning.csv"))?;
    write_circulation_rmse_plot(result, &config.output_dir.join("circulation_rmse.png"))?;
    write_coverage_plot(
        result,
        &config.output_dir.join("circulation_coverage_95.png"),
    )?;
    write_vtk_outputs(result, config)?;
    Ok(())
}

pub fn run_annulus_h_mesh_invariance(
    config: &AnnulusMeshInvarianceConfig,
) -> Result<AnnulusMeshInvarianceResult, Box<dyn Error>> {
    validate_mesh_invariance_config(config)?;
    let mut mesh_metadata = Vec::new();
    let mut qoi_rows = Vec::new();
    let mut summary_rows = Vec::new();

    for (mesh_index, &mesh_size) in config.mesh_sizes.iter().enumerate() {
        let mut annulus_config = config.base.clone();
        let mesh_dir = config.output_dir.join(format!(
            "mesh_{mesh_index:02}_h{}",
            mesh_size_tag(mesh_size)
        ));
        annulus_config.output_dir = mesh_dir.clone();
        annulus_config.mesh_path = mesh_dir.join("annulus_h_formulation.msh");
        annulus_config.geo_path = mesh_dir.join("annulus_h_formulation.geo");
        annulus_config.force_mesh = true;
        annulus_config.mesh_size = mesh_size;
        annulus_config.noise_trial_count = 1;
        annulus_config.sample_observation_noise = config.sample_observation_noise;
        annulus_config.residual_count = config.residual_count;
        annulus_config.heldout_loop_count = config.heldout_loop_count;

        validate_config(&annulus_config)?;
        let workspace = build_workspace(&annulus_config)?;
        let observation_sets = build_observation_sets(&annulus_config, &workspace)?;

        let noisy_line = sample_noisy_observations(
            &observation_sets.train_line,
            config.sample_observation_noise,
            annulus_config.rng_seed + 10_000,
        );
        let noisy_residual = sample_noisy_observations(
            &observation_sets.train_residual,
            config.sample_observation_noise,
            annulus_config.rng_seed + 20_000,
        );
        let noisy_circulation = sample_noisy_observations(
            &observation_sets.train_circulation,
            config.sample_observation_noise,
            annulus_config.rng_seed + 30_000,
        );
        let qoi_line_supports =
            build_mesh_invariance_line_supports(&workspace.topology, &workspace.coords)?;
        let dense_line_supports =
            build_dense_posterior_line_supports(&workspace.topology, &workspace.coords)?;
        let dense_field_observations = build_dense_field_observations(
            &workspace.topology,
            &workspace.coords,
            &workspace.truth_h,
            annulus_config.line_noise_variance,
        )?;
        let mut qoi_observations = observation_sets.heldout_line.to_vec();
        qoi_observations.extend(observations_from_supports(
            AnnulusObservationKind::HeldoutLine,
            &qoi_line_supports,
            &workspace.truth_h,
            annulus_config.line_noise_variance,
        ));
        qoi_observations.extend(observations_from_supports(
            AnnulusObservationKind::HeldoutLine,
            &dense_line_supports,
            &workspace.truth_h,
            annulus_config.line_noise_variance,
        ));
        qoi_observations.extend(dense_field_observations);
        qoi_observations.extend(observation_sets.heldout_residual.iter().cloned());
        qoi_observations.extend(observation_sets.heldout_circulation.iter().cloned());
        qoi_observations.extend(
            observation_sets
                .heldout_homologous_difference
                .iter()
                .cloned(),
        );

        for &model_kind in &config.model_kinds {
            let model = build_mesh_invariance_model(model_kind, &annulus_config, &workspace)?;
            mesh_metadata.push(AnnulusMeshMetadataRow {
                mesh_index,
                mesh_size,
                model: model.model_kind,
                vertex_count: workspace.topology_summary.vertex_count,
                edge_count: workspace.topology_summary.edge_count,
                face_count: workspace.topology_summary.face_count,
                harmonic_1_dimension: workspace.topology_summary.harmonic_1_dimension,
                reference_period_truth: workspace.topology_summary.reference_period_truth,
                reference_period_psi: workspace.topology_summary.reference_period_psi,
                truth_closure_l2: workspace.topology_summary.truth_closure_l2,
                truth_closure_max_abs: workspace.topology_summary.truth_closure_max_abs,
            });

            for regime in [AnnulusRegime::B, AnnulusRegime::D] {
                let training =
                    training_for_regime(regime, &noisy_line, &noisy_residual, &noisy_circulation);
                let direct_support = direct_support_edges(regime, &noisy_line, &noisy_circulation);
                let mut conditioned =
                    condition_model(&model, &training, workspace.topology.nsimplices(1))?;
                let reference_period_observation = AnnulusObservation {
                    kind: AnnulusObservationKind::HeldoutCirculation,
                    label: "q_period".to_string(),
                    entries: workspace.reference_loop.entries.clone(),
                    truth_value: 1.0,
                    observed_value: 1.0,
                    noise_variance: 0.0,
                };
                let reference_period_prediction = predict_observables(
                    0,
                    regime,
                    model.model_kind,
                    &mut conditioned,
                    std::slice::from_ref(&reference_period_observation),
                    workspace.topology.nsimplices(1),
                )?
                .into_iter()
                .next()
                .ok_or_else(|| invalid_data("reference period prediction was empty"))?;
                let q_period_row = AnnulusMeshQoiRow {
                    mesh_index,
                    mesh_size,
                    model: model.model_kind,
                    regime,
                    qoi_kind: "q_period".to_string(),
                    qoi_label: "q_period".to_string(),
                    truth: reference_period_prediction.truth,
                    mean: reference_period_prediction.mean,
                    latent_variance: reference_period_prediction.latent_variance,
                    posterior_sd: reference_period_prediction.latent_variance.max(0.0).sqrt(),
                    support_entries: reference_period_observation.entries.len(),
                    direct_support_overlap_count: 0,
                    is_away_probe: false,
                    probe_family: "period".to_string(),
                };
                qoi_rows.push(q_period_row.clone());

                let predictions = predict_observables(
                    0,
                    regime,
                    model.model_kind,
                    &mut conditioned,
                    &qoi_observations,
                    workspace.topology.nsimplices(1),
                )?;
                let mut mesh_regime_rows = vec![q_period_row];
                for prediction in predictions {
                    let qoi_kind = mesh_qoi_kind(prediction.kind, &prediction.label);
                    let support_entries = qoi_observations
                        .iter()
                        .find(|observation| {
                            observation.kind == prediction.kind
                                && observation.label == prediction.label
                        })
                        .map(|observation| observation.entries.len())
                        .unwrap_or(0);
                    let metadata = qoi_observations
                        .iter()
                        .find(|observation| {
                            observation.kind == prediction.kind
                                && observation.label == prediction.label
                        })
                        .map(|observation| probe_metadata(observation, &direct_support))
                        .unwrap_or_else(|| AnnulusProbeMetadata {
                            direct_support_overlap_count: 0,
                            is_away_probe: false,
                            probe_family: qoi_kind.clone(),
                        });
                    let row = AnnulusMeshQoiRow {
                        mesh_index,
                        mesh_size,
                        model: model.model_kind,
                        regime,
                        qoi_kind,
                        qoi_label: prediction.label,
                        truth: prediction.truth,
                        mean: prediction.mean,
                        latent_variance: prediction.latent_variance,
                        posterior_sd: prediction.latent_variance.max(0.0).sqrt(),
                        support_entries,
                        direct_support_overlap_count: metadata.direct_support_overlap_count,
                        is_away_probe: metadata.is_away_probe,
                        probe_family: metadata.probe_family,
                    };
                    mesh_regime_rows.push(row.clone());
                    qoi_rows.push(row);
                }
                summary_rows.push(summarize_mesh_regime_qois(
                    config.profile,
                    mesh_index,
                    mesh_size,
                    model.model_kind,
                    regime,
                    &workspace.topology_summary,
                    &mesh_regime_rows,
                ));
            }
        }
    }

    annotate_relative_changes_to_finest(&mut summary_rows);
    let fit_rows =
        mesh_invariance_fit_rows(config.profile, config.convergence_tail_count, &summary_rows);
    let contrast_rows = mesh_invariance_model_contrast_rows(config.profile, &summary_rows);
    Ok(AnnulusMeshInvarianceResult {
        mesh_metadata,
        qoi_rows,
        summary_rows,
        fit_rows,
        contrast_rows,
    })
}

pub fn write_annulus_h_mesh_invariance_outputs(
    result: &AnnulusMeshInvarianceResult,
    config: &AnnulusMeshInvarianceConfig,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(&config.output_dir)?;
    write_mesh_metadata_csv(result, &config.output_dir.join("mesh_metadata.csv"))?;
    write_mesh_qoi_csv(result, &config.output_dir.join("qoi_variance_by_mesh.csv"))?;
    write_mesh_invariance_summary_csv(
        result,
        &config.output_dir.join("mesh_invariance_summary.csv"),
    )?;
    write_mesh_invariance_fit_csv(result, &config.output_dir.join("mesh_invariance_fit.csv"))?;
    write_mesh_model_contrast_csv(
        result,
        &config.output_dir.join("mesh_invariance_model_contrast.csv"),
    )?;
    Ok(())
}

pub fn run_annulus_h_efficiency_sweep(
    config: &AnnulusEfficiencyConfig,
) -> Result<AnnulusEfficiencyResult, Box<dyn Error>> {
    validate_efficiency_config(config)?;
    let mut rows = Vec::new();

    for (mesh_index, &mesh_size) in config.mesh_sizes.iter().enumerate() {
        let mut annulus_config = config.base.clone();
        let mesh_dir = config.output_dir.join(format!(
            "mesh_{mesh_index:02}_h{}",
            mesh_size_tag(mesh_size)
        ));
        annulus_config.output_dir = mesh_dir.clone();
        annulus_config.mesh_path = mesh_dir.join("annulus_h_formulation.msh");
        annulus_config.geo_path = mesh_dir.join("annulus_h_formulation.geo");
        annulus_config.force_mesh = true;
        annulus_config.mesh_size = mesh_size;
        annulus_config.noise_trial_count = 1;
        annulus_config.sample_observation_noise = config.sample_observation_noise;

        validate_config(&annulus_config)?;
        let workspace = build_workspace(&annulus_config)?;
        let observation_sets = build_observation_sets(&annulus_config, &workspace)?;
        let noisy_line = sample_noisy_observations(
            &observation_sets.train_line,
            config.sample_observation_noise,
            annulus_config.rng_seed + 10_000,
        );
        let noisy_residual = sample_noisy_observations(
            &observation_sets.train_residual,
            config.sample_observation_noise,
            annulus_config.rng_seed + 20_000,
        );
        let noisy_circulation = sample_noisy_observations(
            &observation_sets.train_circulation,
            config.sample_observation_noise,
            annulus_config.rng_seed + 30_000,
        );
        let training = training_for_regime(
            AnnulusRegime::D,
            &noisy_line,
            &noisy_residual,
            &noisy_circulation,
        );
        let heldout_observations = observation_sets.heldout_all();

        for model_kind in [
            AnnulusModelKind::ScalarPotentialGpCorrection,
            AnnulusModelKind::FeecGmrf,
        ] {
            if model_kind == AnnulusModelKind::ScalarPotentialGpCorrection {
                if let Some(limit) = config.dense_gp_vertex_limit {
                    if workspace.topology_summary.vertex_count > limit {
                        rows.push(failed_efficiency_row(
                            mesh_index,
                            mesh_size,
                            model_kind,
                            "skipped_dense_gp_vertex_limit",
                            &workspace.topology_summary,
                        ));
                        continue;
                    }
                }
            }

            let build_start = Instant::now();
            let model = match build_efficiency_model(model_kind, &annulus_config, &workspace) {
                Ok(model) => model,
                Err(err) => {
                    rows.push(failed_efficiency_row(
                        mesh_index,
                        mesh_size,
                        model_kind,
                        &format!("build_failed:{err}"),
                        &workspace.topology_summary,
                    ));
                    continue;
                }
            };
            let build_seconds = build_start.elapsed().as_secs_f64();

            let conditioning_start = Instant::now();
            let conditioned =
                match condition_model(&model, &training, workspace.topology.nsimplices(1)) {
                    Ok(conditioned) => conditioned,
                    Err(err) => {
                        rows.push(failed_efficiency_row(
                            mesh_index,
                            mesh_size,
                            model_kind,
                            &format!("conditioning_failed:{err}"),
                            &workspace.topology_summary,
                        ));
                        continue;
                    }
                };
            let conditioning_seconds = conditioning_start.elapsed().as_secs_f64();

            let prediction_start = Instant::now();
            let predictions = match predict_observable_means(
                0,
                AnnulusRegime::D,
                model.model_kind,
                &conditioned,
                &heldout_observations,
                workspace.topology.nsimplices(1),
            ) {
                Ok(predictions) => predictions,
                Err(err) => {
                    rows.push(failed_efficiency_row(
                        mesh_index,
                        mesh_size,
                        model_kind,
                        &format!("prediction_failed:{err}"),
                        &workspace.topology_summary,
                    ));
                    continue;
                }
            };
            let prediction_seconds = prediction_start.elapsed().as_secs_f64();
            let selected = SelectedModel {
                model,
                validation_nlpd: f64::NAN,
                selected_build_seconds: build_seconds,
                selection_seconds: 0.0,
            };
            let metric = compute_metric_row(
                0,
                AnnulusRegime::D,
                &selected,
                training.len(),
                &conditioned,
                &predictions,
                f64::NAN,
                conditioning_seconds,
                prediction_seconds,
            );
            rows.push(AnnulusEfficiencyRow {
                mesh_index,
                mesh_size,
                model: model_kind,
                status: "ok".to_string(),
                vertex_count: workspace.topology_summary.vertex_count,
                edge_count: workspace.topology_summary.edge_count,
                face_count: workspace.topology_summary.face_count,
                selected_kappa: metric.selected_kappa,
                selected_tau0: metric.selected_tau0,
                selected_tau1: metric.selected_tau1,
                build_seconds,
                conditioning_seconds,
                prediction_seconds,
                selected_total_seconds: metric.selected_total_seconds,
                latent_dimension: metric.latent_dimension,
                prior_precision_nnz: metric.prior_precision_nnz,
                prior_precision_density: metric.prior_precision_density,
                posterior_precision_nnz: metric.posterior_precision_nnz,
                posterior_precision_density: metric.posterior_precision_density,
                posterior_factor_nnz: metric.posterior_factor_nnz,
                posterior_fill_ratio: metric.posterior_fill_ratio,
                rmse_line: metric.rmse_line,
                residual_rmse: metric.residual_rmse,
                circ_rmse: metric.circ_rmse,
                homologous_difference_rmse: metric.homologous_difference_rmse,
            });
        }
    }

    let speedup_rows = efficiency_speedup_rows(&rows);
    Ok(AnnulusEfficiencyResult { rows, speedup_rows })
}

pub fn write_annulus_h_efficiency_outputs(
    result: &AnnulusEfficiencyResult,
    config: &AnnulusEfficiencyConfig,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(&config.output_dir)?;
    write_efficiency_rows_csv(result, &config.output_dir.join("efficiency_summary.csv"))?;
    write_efficiency_speedup_csv(result, &config.output_dir.join("efficiency_speedup.csv"))?;
    Ok(())
}

fn build_workspace(config: &AnnulusHFormulationConfig) -> Result<AnnulusWorkspace, Box<dyn Error>> {
    ensure_annulus_mesh(config)?;
    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    if topology.homology_dim(1) != 1 {
        return Err(invalid_data(format!(
            "annulus mesh should have b1=1, got {}",
            topology.homology_dim(1)
        ))
        .into());
    }
    let mass_1form = de_rham::mass_matrix_form(&topology, &metric, 1).map_err(invalid_data)?;
    let reference_loop = build_loop_support(&topology, &coords, 1.5, 0.0, false, "reference_loop")?;
    let psi = normalized_harmonic_one_form(&topology, &metric, &mass_1form, &reference_loop)?;
    let truth_h = analytic_truth_h(&topology, &coords);
    let reference_period_truth = row_dot(&reference_loop.entries, &truth_h);
    let reference_period_psi = row_dot(&reference_loop.entries, &psi);
    let d1_truth = de_rham::derivative(&topology, 1, &truth_h);
    let truth_closure_l2 = d1_truth.norm();
    let truth_closure_max_abs = d1_truth.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let heldout_loop_supports = build_heldout_loop_supports(
        &topology,
        &coords,
        config.heldout_loop_count,
        config.rng_seed + 701,
        "heldout_loop",
    )?;
    let topology_summary = AnnulusTopologySummary {
        vertex_count: topology.nsimplices(0),
        edge_count: topology.nsimplices(1),
        face_count: topology.nsimplices(2),
        harmonic_1_dimension: topology.homology_dim(1),
        reference_period_truth,
        reference_period_psi,
        truth_closure_l2,
        truth_closure_max_abs,
    };
    Ok(AnnulusWorkspace {
        topology,
        coords,
        metric,
        mass_1form,
        reference_loop,
        heldout_loop_supports,
        truth_h,
        psi,
        topology_summary,
    })
}

struct ObservationSets {
    train_line: Vec<AnnulusObservation>,
    train_residual: Vec<AnnulusObservation>,
    train_circulation: Vec<AnnulusObservation>,
    validation_line: Vec<AnnulusObservation>,
    validation_residual: Vec<AnnulusObservation>,
    heldout_line: Vec<AnnulusObservation>,
    heldout_residual: Vec<AnnulusObservation>,
    heldout_circulation: Vec<AnnulusObservation>,
    heldout_homologous_difference: Vec<AnnulusObservation>,
}

impl ObservationSets {
    fn heldout_all(&self) -> Vec<AnnulusObservation> {
        self.heldout_line
            .iter()
            .chain(self.heldout_residual.iter())
            .chain(self.heldout_circulation.iter())
            .chain(self.heldout_homologous_difference.iter())
            .cloned()
            .collect()
    }
}

fn build_observation_sets(
    config: &AnnulusHFormulationConfig,
    workspace: &AnnulusWorkspace,
) -> Result<ObservationSets, Box<dyn Error>> {
    let train_line_supports = build_line_supports(
        &workspace.topology,
        &workspace.coords,
        config.line_tangential_count,
        config.line_radial_count,
        config.line_random_count,
        config.rng_seed + 101,
        "train_line",
    )?;
    let (validation_tangent, validation_radial, validation_random) =
        split_line_counts(config.validation_line_count);
    let validation_line_supports = build_line_supports(
        &workspace.topology,
        &workspace.coords,
        validation_tangent,
        validation_radial,
        validation_random,
        config.rng_seed + 102,
        "validation_line",
    )?;
    let (heldout_tangent, heldout_radial, heldout_random) =
        split_line_counts(config.heldout_line_count);
    let heldout_line_supports = build_line_supports(
        &workspace.topology,
        &workspace.coords,
        heldout_tangent,
        heldout_radial,
        heldout_random,
        config.rng_seed + 103,
        "heldout_line",
    )?;
    let train_residual_supports = build_residual_supports(
        &workspace.topology,
        config.residual_count,
        config.rng_seed + 201,
        "train_residual",
    );
    let validation_residual_supports = build_residual_supports(
        &workspace.topology,
        config.validation_residual_count,
        config.rng_seed + 202,
        "validation_residual",
    );
    let heldout_residual_supports = build_residual_supports(
        &workspace.topology,
        config.heldout_residual_count,
        config.rng_seed + 203,
        "heldout_residual",
    );
    let train_circulation_supports = vec![workspace.reference_loop.clone()];
    let heldout_difference_supports =
        build_homologous_difference_supports(&workspace.heldout_loop_supports);

    Ok(ObservationSets {
        train_line: observations_from_supports(
            AnnulusObservationKind::TrainLine,
            &train_line_supports,
            &workspace.truth_h,
            config.line_noise_variance,
        ),
        train_residual: observations_from_supports(
            AnnulusObservationKind::TrainResidual,
            &train_residual_supports,
            &workspace.truth_h,
            config.residual_noise_variance,
        ),
        train_circulation: observations_from_supports(
            AnnulusObservationKind::TrainCirculation,
            &train_circulation_supports,
            &workspace.truth_h,
            config.circulation_noise_variance,
        ),
        validation_line: observations_from_supports(
            AnnulusObservationKind::ValidationLine,
            &validation_line_supports,
            &workspace.truth_h,
            config.line_noise_variance,
        ),
        validation_residual: observations_from_supports(
            AnnulusObservationKind::ValidationResidual,
            &validation_residual_supports,
            &workspace.truth_h,
            config.residual_noise_variance,
        ),
        heldout_line: observations_from_supports(
            AnnulusObservationKind::HeldoutLine,
            &heldout_line_supports,
            &workspace.truth_h,
            config.line_noise_variance,
        ),
        heldout_residual: observations_from_supports(
            AnnulusObservationKind::HeldoutResidual,
            &heldout_residual_supports,
            &workspace.truth_h,
            config.residual_noise_variance,
        ),
        heldout_circulation: observations_from_supports(
            AnnulusObservationKind::HeldoutCirculation,
            &workspace.heldout_loop_supports,
            &workspace.truth_h,
            config.circulation_noise_variance,
        ),
        heldout_homologous_difference: observations_from_supports(
            AnnulusObservationKind::HeldoutHomologousDifference,
            &heldout_difference_supports,
            &workspace.truth_h,
            config.circulation_noise_variance,
        ),
    })
}

fn select_models(
    config: &AnnulusHFormulationConfig,
    workspace: &AnnulusWorkspace,
    validation_train: &[AnnulusObservation],
    validation_heldout: &[AnnulusObservation],
    tuning_rows: &mut Vec<AnnulusTuningRow>,
) -> Result<Vec<SelectedModel>, Box<dyn Error>> {
    let mut selected = Vec::new();
    for kind in AnnulusModelKind::benchmark_models() {
        selected.push(select_model(
            kind,
            config,
            workspace,
            validation_train,
            validation_heldout,
            tuning_rows,
        )?);
    }
    Ok(selected)
}

fn select_model(
    kind: AnnulusModelKind,
    config: &AnnulusHFormulationConfig,
    workspace: &AnnulusWorkspace,
    validation_train: &[AnnulusObservation],
    validation_heldout: &[AnnulusObservation],
    tuning_rows: &mut Vec<AnnulusTuningRow>,
) -> Result<SelectedModel, Box<dyn Error>> {
    let selection_start = Instant::now();
    let candidates = build_model_candidates(kind, config, workspace)?;
    let mut best: Option<SelectedModel> = None;
    let mut candidate_rows = Vec::new();
    for candidate in candidates {
        let validation_nlpd = match condition_model(
            &candidate.model,
            validation_train,
            workspace.topology.nsimplices(1),
        )
        .and_then(|mut conditioned| {
            predict_observables(
                0,
                AnnulusRegime::A,
                candidate.model.model_kind,
                &mut conditioned,
                validation_heldout,
                workspace.topology.nsimplices(1),
            )
        }) {
            Ok(predictions) => mean(predictions.iter().map(|row| row.nlpd)),
            Err(_) => f64::INFINITY,
        };
        candidate_rows.push((candidate.clone(), validation_nlpd));
        if validation_nlpd.is_finite()
            && best
                .as_ref()
                .map(|current| validation_nlpd < current.validation_nlpd)
                .unwrap_or(true)
        {
            best = Some(SelectedModel {
                model: candidate.model.clone(),
                validation_nlpd,
                selected_build_seconds: candidate.build_seconds,
                selection_seconds: 0.0,
            });
        }
    }
    let mut best = best.ok_or_else(|| {
        invalid_data(format!(
            "no finite validation candidate for model {}",
            kind.as_str()
        ))
    })?;
    best.selection_seconds = selection_start.elapsed().as_secs_f64();
    for (candidate, validation_nlpd) in candidate_rows {
        tuning_rows.push(AnnulusTuningRow {
            model: candidate.model.model_kind,
            selected: candidate.model.model_kind == best.model.model_kind
                && same_f64_or_nan(candidate.model.selected_kappa, best.model.selected_kappa)
                && same_f64_or_nan(candidate.model.selected_tau0, best.model.selected_tau0)
                && same_f64_or_nan(candidate.model.selected_tau1, best.model.selected_tau1),
            kappa: candidate.model.selected_kappa,
            tau0: candidate.model.selected_tau0,
            tau1: candidate.model.selected_tau1,
            validation_nlpd,
        });
    }
    Ok(best)
}

fn same_f64_or_nan(lhs: f64, rhs: f64) -> bool {
    (lhs.is_nan() && rhs.is_nan()) || lhs == rhs
}

fn build_model_candidates(
    kind: AnnulusModelKind,
    config: &AnnulusHFormulationConfig,
    workspace: &AnnulusWorkspace,
) -> Result<Vec<AnnulusModelCandidate>, Box<dyn Error>> {
    let mut models = Vec::new();
    match kind {
        AnnulusModelKind::ComponentwiseGp => {
            for &kappa in &config.component_kappa_grid {
                for &tau in &config.component_tau_grid {
                    push_timed_candidate(&mut models, || {
                        build_componentwise_gp_model(
                            &workspace.topology,
                            &workspace.coords,
                            AnnulusComponentwiseGpConfig {
                                kappa,
                                tau,
                                jitter_scale: 1e-8,
                            },
                        )
                    })?;
                }
            }
        }
        AnnulusModelKind::ComponentwiseGpCorrection => {
            for &kappa in &config.component_kappa_grid {
                for &tau in &config.component_tau_grid {
                    push_timed_candidate(&mut models, || {
                        build_componentwise_gp_correction_model(
                            &workspace.topology,
                            &workspace.coords,
                            &workspace.psi,
                            AnnulusComponentwiseGpConfig {
                                kappa,
                                tau,
                                jitter_scale: 1e-8,
                            },
                        )
                    })?;
                }
            }
        }
        AnnulusModelKind::ScalarPotentialGp => {
            for &kappa in &config.scalar_potential_kappa_grid {
                for &tau in &config.scalar_potential_tau_grid {
                    push_timed_candidate(&mut models, || {
                        build_scalar_potential_gp_model(
                            &workspace.topology,
                            &workspace.coords,
                            AnnulusScalarPotentialGpConfig {
                                kappa,
                                tau,
                                jitter_scale: 1e-8,
                            },
                        )
                    })?;
                }
            }
        }
        AnnulusModelKind::ScalarPotentialGpCorrection => {
            for &kappa in &config.scalar_potential_kappa_grid {
                for &tau in &config.scalar_potential_tau_grid {
                    push_timed_candidate(&mut models, || {
                        build_scalar_potential_gp_correction_model(
                            &workspace.topology,
                            &workspace.coords,
                            &workspace.psi,
                            AnnulusScalarPotentialGpConfig {
                                kappa,
                                tau,
                                jitter_scale: 1e-8,
                            },
                        )
                    })?;
                }
            }
        }
        AnnulusModelKind::ExactOnlyFeec => {
            for &tau0 in &config.potential_tau0_grid {
                for &tau1 in &config.potential_tau1_grid {
                    push_timed_candidate(&mut models, || {
                        build_exact_only_feec_model(
                            &workspace.topology,
                            &workspace.metric,
                            &workspace.mass_1form,
                            AnnulusPotentialPriorConfig {
                                tau0,
                                tau1,
                                sigma_q: config.harmonic_prior_std,
                            },
                        )
                    })?;
                }
            }
        }
        AnnulusModelKind::FeecSplitNoSpectralCorrection => {
            for &tau0 in &config.potential_tau0_grid {
                for &tau1 in &config.potential_tau1_grid {
                    push_timed_candidate(&mut models, || {
                        build_feec_split_no_spectral_correction_model(
                            &workspace.topology,
                            &workspace.metric,
                            &workspace.mass_1form,
                            &workspace.psi,
                            AnnulusPotentialPriorConfig {
                                tau0,
                                tau1,
                                sigma_q: config.harmonic_prior_std,
                            },
                        )
                    })?;
                }
            }
        }
        AnnulusModelKind::FeecGmrf => {
            for &tau0 in &config.potential_tau0_grid {
                for &tau1 in &config.potential_tau1_grid {
                    push_timed_candidate(&mut models, || {
                        build_feec_gmrf_model(
                            &workspace.topology,
                            &workspace.metric,
                            &workspace.mass_1form,
                            &workspace.psi,
                            AnnulusPotentialPriorConfig {
                                tau0,
                                tau1,
                                sigma_q: config.harmonic_prior_std,
                            },
                        )
                    })?;
                }
            }
        }
    }
    Ok(models)
}

fn push_timed_candidate(
    models: &mut Vec<AnnulusModelCandidate>,
    build: impl FnOnce() -> Result<AnnulusLinearModel, String>,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let model = build().map_err(invalid_data)?;
    models.push(AnnulusModelCandidate {
        model,
        build_seconds: start.elapsed().as_secs_f64(),
    });
    Ok(())
}

fn build_mesh_invariance_model(
    kind: AnnulusModelKind,
    config: &AnnulusHFormulationConfig,
    workspace: &AnnulusWorkspace,
) -> Result<AnnulusLinearModel, Box<dyn Error>> {
    let potential = AnnulusPotentialPriorConfig {
        tau0: config.tau0,
        tau1: config.tau1,
        sigma_q: config.harmonic_prior_std,
    };
    match kind {
        AnnulusModelKind::FeecGmrf => Ok(build_feec_gmrf_model(
            &workspace.topology,
            &workspace.metric,
            &workspace.mass_1form,
            &workspace.psi,
            potential,
        )?),
        AnnulusModelKind::FeecSplitNoSpectralCorrection => {
            Ok(build_feec_split_no_spectral_correction_model(
                &workspace.topology,
                &workspace.metric,
                &workspace.mass_1form,
                &workspace.psi,
                potential,
            )?)
        }
        other => Err(invalid_input(format!(
            "mesh invariance supports only feec_gmrf and feec_split_no_spectral_correction, got {}",
            other.as_str()
        ))
        .into()),
    }
}

fn build_efficiency_model(
    kind: AnnulusModelKind,
    config: &AnnulusHFormulationConfig,
    workspace: &AnnulusWorkspace,
) -> Result<AnnulusLinearModel, Box<dyn Error>> {
    match kind {
        AnnulusModelKind::ScalarPotentialGpCorrection => {
            Ok(build_scalar_potential_gp_correction_model(
                &workspace.topology,
                &workspace.coords,
                &workspace.psi,
                AnnulusScalarPotentialGpConfig {
                    kappa: 1.0,
                    tau: 1.0,
                    jitter_scale: 1e-8,
                },
            )?)
        }
        AnnulusModelKind::FeecGmrf => Ok(build_feec_gmrf_model(
            &workspace.topology,
            &workspace.metric,
            &workspace.mass_1form,
            &workspace.psi,
            AnnulusPotentialPriorConfig {
                tau0: config.tau0,
                tau1: config.tau1,
                sigma_q: config.harmonic_prior_std,
            },
        )?),
        other => Err(invalid_input(format!(
            "efficiency sweep supports scalar_potential_gp_correction and feec_gmrf, got {}",
            other.as_str()
        ))
        .into()),
    }
}

fn condition_model(
    model: &AnnulusLinearModel,
    observations: &[AnnulusObservation],
    h_dimension: usize,
) -> Result<ConditionedAnnulusModel, Box<dyn Error>> {
    let prior = GaussianPrior::new(
        vec![0.0; model.latent_dimension()],
        sparse_mat_from_feec_csr(&model.prior_precision),
    )?;
    let latent_to_h = LinearMap::from_feec_csr(&model.latent_to_h)?;
    if latent_to_h.output_dimension() != h_dimension {
        return Err(invalid_data(format!(
            "latent-to-H output dimension {} does not match H dimension {h_dimension}",
            latent_to_h.output_dimension()
        ))
        .into());
    }
    let h_observation_matrix = observation_matrix(observations, h_dimension, false)?;
    let h_observation_map = LinearMap::from_feec_csr(&h_observation_matrix)?;
    let latent_observation_map = h_observation_map.compose(&latent_to_h)?;
    let h_offset = model.h_offset.iter().copied().collect::<Vec<_>>();
    let observation_bias = h_observation_map.apply(&h_offset)?;
    let observation_values = observations
        .iter()
        .map(|observation| observation.observed_value)
        .collect::<Vec<_>>();
    let observation_variances = observations
        .iter()
        .map(|observation| observation.noise_variance.max(VARIANCE_FLOOR))
        .collect::<Vec<_>>();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .observe(LinearObservation::with_bias(
            latent_observation_map,
            observation_values,
            observation_bias,
            GaussianNoise::independent_variances(&observation_variances)?,
        )?)?
        .condition()?;
    let diagnostics = posterior.factorization_diagnostics()?;
    let latent_mean = FeecVector::from_vec(posterior.mean().to_vec());
    let h_mean = FeecVector::from_vec(
        latent_to_h
            .apply(posterior.mean())?
            .into_iter()
            .zip(h_offset)
            .map(|(value, offset)| value + offset)
            .collect(),
    );
    let (q_mean, q_std) = if let Some(q_column) = model.q_column {
        if q_column >= latent_mean.len() {
            return Err(invalid_data(format!(
                "q column {q_column} exceeds latent dimension {}",
                latent_mean.len()
            ))
            .into());
        }
        let q_operator = LinearMap::selector(latent_mean.len(), &[q_column])?;
        let q_variance = posterior
            .pushforward_variance_estimate(&q_operator, VarianceMethod::Exact)?
            .values[0]
            .max(VARIANCE_FLOOR);
        (latent_mean[q_column], q_variance.sqrt())
    } else {
        (f64::NAN, f64::NAN)
    };
    Ok(ConditionedAnnulusModel {
        latent_to_h,
        posterior,
        posterior_precision_nnz: diagnostics.precision_nonzeros,
        posterior_factor_nnz: diagnostics.factor_nonzeros,
        posterior_fill_ratio: diagnostics.fill_ratio,
        h_mean,
        q_mean,
        q_std,
    })
}

fn predict_observables(
    trial: usize,
    regime: AnnulusRegime,
    model: AnnulusModelKind,
    conditioned: &mut ConditionedAnnulusModel,
    observations: &[AnnulusObservation],
    h_dimension: usize,
) -> Result<Vec<AnnulusPredictionRow>, Box<dyn Error>> {
    let observation_matrix = observation_matrix(observations, h_dimension, false)?;
    let observation_map = LinearMap::from_feec_csr(&observation_matrix)?;
    let latent_operator = observation_map.compose(&conditioned.latent_to_h)?;
    let variance = conditioned
        .posterior
        .pushforward_variance_estimate(&latent_operator, VarianceMethod::Exact)?
        .values;
    let means = &observation_matrix * &conditioned.h_mean;
    Ok(observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            let latent_variance = variance[index].max(VARIANCE_FLOOR);
            let predictive_variance =
                (latent_variance + observation.noise_variance).max(VARIANCE_FLOOR);
            let z = (observation.truth_value - means[index]) / latent_variance.sqrt();
            AnnulusPredictionRow {
                trial,
                regime,
                model,
                kind: observation.kind,
                label: observation.label.clone(),
                truth: observation.truth_value,
                mean: means[index],
                latent_variance,
                predictive_variance,
                z,
                nlpd: gaussian_nlpd(observation.truth_value, means[index], predictive_variance),
            }
        })
        .collect())
}

fn predict_observable_means(
    trial: usize,
    regime: AnnulusRegime,
    model: AnnulusModelKind,
    conditioned: &ConditionedAnnulusModel,
    observations: &[AnnulusObservation],
    h_dimension: usize,
) -> Result<Vec<AnnulusPredictionRow>, Box<dyn Error>> {
    let observation_matrix = observation_matrix(observations, h_dimension, false)?;
    let means = &observation_matrix * &conditioned.h_mean;
    Ok(observations
        .iter()
        .enumerate()
        .map(|(index, observation)| AnnulusPredictionRow {
            trial,
            regime,
            model,
            kind: observation.kind,
            label: observation.label.clone(),
            truth: observation.truth_value,
            mean: means[index],
            latent_variance: f64::NAN,
            predictive_variance: f64::NAN,
            z: f64::NAN,
            nlpd: f64::NAN,
        })
        .collect())
}

fn compute_topo_spread(
    conditioned: &mut ConditionedAnnulusModel,
    loops: &[AnnulusSupport],
) -> Result<f64, Box<dyn Error>> {
    if loops.is_empty() {
        return Ok(f64::NAN);
    }
    let loop_observations = loops
        .iter()
        .map(|support| AnnulusObservation {
            kind: AnnulusObservationKind::HeldoutCirculation,
            label: support.label.clone(),
            entries: support.entries.clone(),
            truth_value: 1.0,
            observed_value: 1.0,
            noise_variance: 0.0,
        })
        .collect::<Vec<_>>();
    let loop_matrix = observation_matrix(&loop_observations, conditioned.h_mean.len(), false)?;
    let loop_map = LinearMap::from_feec_csr(&loop_matrix)?;
    let latent_loop_map = loop_map.compose(&conditioned.latent_to_h)?;
    let covariance = conditioned
        .posterior
        .pushforward_covariance(&latent_loop_map)?;
    let means = &loop_matrix * &conditioned.h_mean;
    let mean_bar = mean(means.iter().copied());
    let mean_spread = means
        .iter()
        .map(|value| {
            let delta = value - mean_bar;
            delta * delta
        })
        .sum::<f64>()
        / loops.len() as f64;
    let trace = covariance
        .iter()
        .enumerate()
        .take(loops.len())
        .map(|(index, row)| row[index])
        .sum::<f64>();
    let total = covariance
        .iter()
        .take(loops.len())
        .flat_map(|row| row.iter().take(loops.len()))
        .sum::<f64>();
    Ok(mean_spread + trace / loops.len() as f64 - total / (loops.len() * loops.len()) as f64)
}

// Case-study row assembly keeps trial, regime, sparsity, and timing provenance
// explicit. The annulus smoke and thesis profiles exercise this exact report path.
#[allow(clippy::too_many_arguments)]
fn compute_metric_row(
    trial: usize,
    regime: AnnulusRegime,
    selected: &SelectedModel,
    training_observation_count: usize,
    conditioned: &ConditionedAnnulusModel,
    predictions: &[AnnulusPredictionRow],
    topo_spread: f64,
    conditioning_seconds: f64,
    prediction_seconds: f64,
) -> AnnulusMetricRow {
    let line = predictions
        .iter()
        .filter(|row| row.kind == AnnulusObservationKind::HeldoutLine)
        .collect::<Vec<_>>();
    let residual = predictions
        .iter()
        .filter(|row| row.kind == AnnulusObservationKind::HeldoutResidual)
        .collect::<Vec<_>>();
    let circ = predictions
        .iter()
        .filter(|row| row.kind == AnnulusObservationKind::HeldoutCirculation)
        .collect::<Vec<_>>();
    let homologous = predictions
        .iter()
        .filter(|row| row.kind == AnnulusObservationKind::HeldoutHomologousDifference)
        .collect::<Vec<_>>();
    let prior_precision_nnz = sparse_nnz(&selected.model.prior_precision);
    let latent_dimension = selected.model.latent_dimension();
    AnnulusMetricRow {
        trial,
        regime,
        model: selected.model.model_kind,
        training_observation_count,
        selected_kappa: selected.model.selected_kappa,
        selected_tau0: selected.model.selected_tau0,
        selected_tau1: selected.model.selected_tau1,
        validation_nlpd: selected.validation_nlpd,
        rmse_line: rmse(line.iter().map(|row| row.mean - row.truth)),
        residual_rmse: rmse(residual.iter().map(|row| row.mean)),
        circ_rmse: rmse(circ.iter().map(|row| row.mean - row.truth)),
        homologous_difference_rmse: rmse(homologous.iter().map(|row| row.mean - row.truth)),
        z_mean: mean(circ.iter().map(|row| row.z)),
        z_variance: variance(circ.iter().map(|row| row.z)),
        coverage_90: coverage(circ.iter().map(|row| row.z), 1.644_853_626_951_472_2),
        coverage_95: coverage(circ.iter().map(|row| row.z), 1.959_963_984_540_054),
        nlpd: mean(predictions.iter().map(|row| row.nlpd)),
        circ_nlpd: mean(circ.iter().map(|row| row.nlpd)),
        topo_spread,
        q_mean: conditioned.q_mean,
        q_std: conditioned.q_std,
        selected_build_seconds: selected.selected_build_seconds,
        selection_seconds: selected.selection_seconds,
        conditioning_seconds,
        prediction_seconds,
        selected_total_seconds: selected.selected_build_seconds
            + conditioning_seconds
            + prediction_seconds,
        pipeline_total_seconds: selected.selection_seconds
            + conditioning_seconds
            + prediction_seconds,
        latent_dimension,
        prior_precision_nnz,
        prior_precision_density: square_density(latent_dimension, prior_precision_nnz),
        posterior_precision_nnz: conditioned.posterior_precision_nnz,
        posterior_precision_density: square_density(
            latent_dimension,
            conditioned.posterior_precision_nnz,
        ),
        posterior_factor_nnz: conditioned.posterior_factor_nnz,
        posterior_fill_ratio: conditioned.posterior_fill_ratio,
    }
}

fn aggregate_summary_rows(metrics: &[AnnulusMetricRow]) -> Vec<AnnulusSummaryRow> {
    let mut groups = BTreeMap::<(AnnulusRegime, AnnulusModelKind), Vec<&AnnulusMetricRow>>::new();
    for row in metrics {
        groups.entry((row.regime, row.model)).or_default().push(row);
    }
    groups
        .into_iter()
        .map(|((regime, model), rows)| AnnulusSummaryRow {
            regime,
            model,
            trial_count: rows.len(),
            rmse_line: mean(rows.iter().map(|row| row.rmse_line)),
            residual_rmse: mean(rows.iter().map(|row| row.residual_rmse)),
            circ_rmse: mean(rows.iter().map(|row| row.circ_rmse)),
            homologous_difference_rmse: mean(rows.iter().map(|row| row.homologous_difference_rmse)),
            z_mean: mean(rows.iter().map(|row| row.z_mean)),
            z_variance: mean(rows.iter().map(|row| row.z_variance)),
            coverage_90: mean(rows.iter().map(|row| row.coverage_90)),
            coverage_95: mean(rows.iter().map(|row| row.coverage_95)),
            nlpd: mean(rows.iter().map(|row| row.nlpd)),
            circ_nlpd: mean(rows.iter().map(|row| row.circ_nlpd)),
            topo_spread: mean(rows.iter().map(|row| row.topo_spread)),
            q_mean: mean(rows.iter().map(|row| row.q_mean)),
            q_std: mean(rows.iter().map(|row| row.q_std)),
            selected_build_seconds: mean(rows.iter().map(|row| row.selected_build_seconds)),
            selection_seconds: mean(rows.iter().map(|row| row.selection_seconds)),
            conditioning_seconds: mean(rows.iter().map(|row| row.conditioning_seconds)),
            prediction_seconds: mean(rows.iter().map(|row| row.prediction_seconds)),
            selected_total_seconds: mean(rows.iter().map(|row| row.selected_total_seconds)),
            pipeline_total_seconds: mean(rows.iter().map(|row| row.pipeline_total_seconds)),
            latent_dimension: mean(rows.iter().map(|row| row.latent_dimension as f64)),
            prior_precision_nnz: mean(rows.iter().map(|row| row.prior_precision_nnz as f64)),
            prior_precision_density: mean(rows.iter().map(|row| row.prior_precision_density)),
            posterior_precision_nnz: mean(
                rows.iter().map(|row| row.posterior_precision_nnz as f64),
            ),
            posterior_precision_density: mean(
                rows.iter().map(|row| row.posterior_precision_density),
            ),
            posterior_factor_nnz: mean(rows.iter().map(|row| row.posterior_factor_nnz as f64)),
            posterior_fill_ratio: mean(rows.iter().map(|row| row.posterior_fill_ratio)),
        })
        .collect()
}

fn sparse_nnz(matrix: &FeecCsr) -> usize {
    matrix
        .triplet_iter()
        .filter(|(_, _, value)| **value != 0.0)
        .count()
}

fn square_density(dimension: usize, nnz: usize) -> f64 {
    if dimension == 0 {
        return f64::NAN;
    }
    nnz as f64 / ((dimension * dimension) as f64)
}

fn failed_efficiency_row(
    mesh_index: usize,
    mesh_size: f64,
    model: AnnulusModelKind,
    status: &str,
    topology: &AnnulusTopologySummary,
) -> AnnulusEfficiencyRow {
    AnnulusEfficiencyRow {
        mesh_index,
        mesh_size,
        model,
        status: status.to_string(),
        vertex_count: topology.vertex_count,
        edge_count: topology.edge_count,
        face_count: topology.face_count,
        selected_kappa: f64::NAN,
        selected_tau0: f64::NAN,
        selected_tau1: f64::NAN,
        build_seconds: f64::NAN,
        conditioning_seconds: f64::NAN,
        prediction_seconds: f64::NAN,
        selected_total_seconds: f64::NAN,
        latent_dimension: 0,
        prior_precision_nnz: 0,
        prior_precision_density: f64::NAN,
        posterior_precision_nnz: 0,
        posterior_precision_density: f64::NAN,
        posterior_factor_nnz: 0,
        posterior_fill_ratio: f64::NAN,
        rmse_line: f64::NAN,
        residual_rmse: f64::NAN,
        circ_rmse: f64::NAN,
        homologous_difference_rmse: f64::NAN,
    }
}

fn efficiency_speedup_rows(rows: &[AnnulusEfficiencyRow]) -> Vec<AnnulusEfficiencySpeedupRow> {
    let mut grouped = BTreeMap::<(usize, u64), Vec<&AnnulusEfficiencyRow>>::new();
    for row in rows.iter().filter(|row| row.status == "ok") {
        grouped
            .entry((row.mesh_index, row.mesh_size.to_bits()))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .filter_map(|((mesh_index, mesh_bits), rows)| {
            let gp = rows
                .iter()
                .find(|row| row.model == AnnulusModelKind::ScalarPotentialGpCorrection)?;
            let feec = rows
                .iter()
                .find(|row| row.model == AnnulusModelKind::FeecGmrf)?;
            Some(AnnulusEfficiencySpeedupRow {
                mesh_index,
                mesh_size: f64::from_bits(mesh_bits),
                vertex_count: feec.vertex_count,
                gp_selected_total_seconds: gp.selected_total_seconds,
                feec_selected_total_seconds: feec.selected_total_seconds,
                speedup: gp.selected_total_seconds / feec.selected_total_seconds,
            })
        })
        .collect()
}

fn summarize_mesh_regime_qois(
    profile: AnnulusMeshInvarianceProfile,
    mesh_index: usize,
    mesh_size: f64,
    model: AnnulusModelKind,
    regime: AnnulusRegime,
    topology: &AnnulusTopologySummary,
    rows: &[AnnulusMeshQoiRow],
) -> AnnulusMeshInvarianceSummaryRow {
    let q_period = rows
        .iter()
        .find(|row| row.qoi_kind == "q_period")
        .map(|row| row.latent_variance)
        .unwrap_or(f64::NAN);
    let circulation = summarize_qoi_rows(rows, "heldout_circulation");
    let line = summarize_qoi_rows_matching(rows, |row| row.qoi_kind.starts_with("heldout_line"));
    let short_line = summarize_qoi_rows(rows, "heldout_line_short");
    let long_line = summarize_qoi_rows(rows, "heldout_line_physical_long");
    let dense_line = summarize_qoi_rows_matching(rows, |row| {
        row.qoi_kind == "dense_line" && row.is_away_probe
    });
    let field_x =
        summarize_qoi_rows_matching(rows, |row| row.qoi_kind == "field_x" && row.is_away_probe);
    let field_y =
        summarize_qoi_rows_matching(rows, |row| row.qoi_kind == "field_y" && row.is_away_probe);
    let field_magnitude = summarize_field_magnitude_rows(rows);
    let homologous = summarize_qoi_rows(rows, "heldout_homologous_difference");
    AnnulusMeshInvarianceSummaryRow {
        profile,
        mesh_index,
        mesh_size,
        model,
        regime,
        vertex_count: topology.vertex_count,
        edge_count: topology.edge_count,
        face_count: topology.face_count,
        q_period_variance: q_period,
        q_period_sd: q_period.max(0.0).sqrt(),
        circulation_count: circulation.count,
        circulation_mean_variance: circulation.mean_variance,
        circulation_median_variance: circulation.median_variance,
        circulation_mean_sd: circulation.mean_sd,
        circulation_median_sd: circulation.median_sd,
        line_count: line.count,
        line_mean_variance: line.mean_variance,
        line_median_variance: line.median_variance,
        line_mean_sd: line.mean_sd,
        line_median_sd: line.median_sd,
        short_line_count: short_line.count,
        short_line_mean_variance: short_line.mean_variance,
        short_line_median_variance: short_line.median_variance,
        short_line_mean_sd: short_line.mean_sd,
        short_line_median_sd: short_line.median_sd,
        long_line_count: long_line.count,
        long_line_mean_variance: long_line.mean_variance,
        long_line_median_variance: long_line.median_variance,
        long_line_mean_sd: long_line.mean_sd,
        long_line_median_sd: long_line.median_sd,
        dense_line_away_count: dense_line.count,
        dense_line_away_median_variance: dense_line.median_variance,
        dense_line_away_p90_variance: dense_line.p90_variance,
        dense_line_away_median_sd: dense_line.median_sd,
        dense_line_away_p90_sd: dense_line.p90_sd,
        field_x_away_count: field_x.count,
        field_x_away_median_variance: field_x.median_variance,
        field_x_away_median_sd: field_x.median_sd,
        field_y_away_count: field_y.count,
        field_y_away_median_variance: field_y.median_variance,
        field_y_away_median_sd: field_y.median_sd,
        field_magnitude_away_count: field_magnitude.count,
        field_magnitude_away_median_variance: field_magnitude.median_variance,
        field_magnitude_away_median_sd: field_magnitude.median_sd,
        homologous_count: homologous.count,
        homologous_mean_variance: homologous.mean_variance,
        homologous_median_variance: homologous.median_variance,
        homologous_mean_sd: homologous.mean_sd,
        homologous_median_sd: homologous.median_sd,
        max_relative_change_to_finest_variance: f64::NAN,
    }
}

#[derive(Debug, Clone, Copy)]
struct QoiSummaryStats {
    count: usize,
    mean_variance: f64,
    median_variance: f64,
    p90_variance: f64,
    mean_sd: f64,
    median_sd: f64,
    p90_sd: f64,
}

fn summarize_qoi_rows(rows: &[AnnulusMeshQoiRow], qoi_kind: &str) -> QoiSummaryStats {
    summarize_qoi_rows_matching(rows, |row| row.qoi_kind == qoi_kind)
}

fn summarize_qoi_rows_matching(
    rows: &[AnnulusMeshQoiRow],
    predicate: impl Fn(&AnnulusMeshQoiRow) -> bool,
) -> QoiSummaryStats {
    let variances = rows
        .iter()
        .filter(|row| predicate(row))
        .map(|row| row.latent_variance)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .collect::<Vec<_>>();
    let sds = variances
        .iter()
        .map(|variance| variance.sqrt())
        .collect::<Vec<_>>();
    QoiSummaryStats {
        count: variances.len(),
        mean_variance: mean(variances.iter().copied()),
        median_variance: median(variances.iter().copied()),
        p90_variance: percentile(variances.iter().copied(), 0.90),
        mean_sd: mean(sds.iter().copied()),
        median_sd: median(sds.iter().copied()),
        p90_sd: percentile(sds.iter().copied(), 0.90),
    }
}

fn summarize_field_magnitude_rows(rows: &[AnnulusMeshQoiRow]) -> QoiSummaryStats {
    let mut by_label = BTreeMap::<String, (Option<f64>, Option<f64>, bool)>::new();
    for row in rows.iter().filter(|row| row.is_away_probe) {
        if row.qoi_kind == "field_x" {
            by_label
                .entry(field_sample_base_label(&row.qoi_label).to_string())
                .or_insert((None, None, true))
                .0 = Some(row.latent_variance);
        } else if row.qoi_kind == "field_y" {
            by_label
                .entry(field_sample_base_label(&row.qoi_label).to_string())
                .or_insert((None, None, true))
                .1 = Some(row.latent_variance);
        }
    }
    let variances = by_label
        .values()
        .filter_map(|(x, y, _)| match (x, y) {
            (Some(x), Some(y)) if x.is_finite() && y.is_finite() && *x >= 0.0 && *y >= 0.0 => {
                Some(*x + *y)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let sds = variances
        .iter()
        .map(|variance| variance.sqrt())
        .collect::<Vec<_>>();
    QoiSummaryStats {
        count: variances.len(),
        mean_variance: mean(variances.iter().copied()),
        median_variance: median(variances.iter().copied()),
        p90_variance: percentile(variances.iter().copied(), 0.90),
        mean_sd: mean(sds.iter().copied()),
        median_sd: median(sds.iter().copied()),
        p90_sd: percentile(sds.iter().copied(), 0.90),
    }
}

fn field_sample_base_label(label: &str) -> &str {
    label
        .strip_suffix("_x")
        .or_else(|| label.strip_suffix("_y"))
        .unwrap_or(label)
}

fn annotate_relative_changes_to_finest(rows: &mut [AnnulusMeshInvarianceSummaryRow]) {
    let models = rows.iter().map(|row| row.model).collect::<BTreeSet<_>>();
    for model in models {
        for regime in [AnnulusRegime::B, AnnulusRegime::D] {
            let finest = rows
                .iter()
                .filter(|row| row.model == model && row.regime == regime)
                .min_by(|lhs, rhs| lhs.mesh_size.total_cmp(&rhs.mesh_size))
                .cloned();
            let Some(finest) = finest else {
                continue;
            };
            for row in rows
                .iter_mut()
                .filter(|row| row.model == model && row.regime == regime)
            {
                row.max_relative_change_to_finest_variance = [
                    relative_change(row.q_period_variance, finest.q_period_variance),
                    relative_change(
                        row.circulation_mean_variance,
                        finest.circulation_mean_variance,
                    ),
                    relative_change(
                        row.short_line_median_variance,
                        finest.short_line_median_variance,
                    ),
                    relative_change(
                        row.long_line_median_variance,
                        finest.long_line_median_variance,
                    ),
                ]
                .into_iter()
                .filter(|value| value.is_finite())
                .fold(0.0_f64, f64::max);
            }
        }
    }
}

fn mesh_invariance_fit_rows(
    profile: AnnulusMeshInvarianceProfile,
    convergence_tail_count: usize,
    summaries: &[AnnulusMeshInvarianceSummaryRow],
) -> Vec<AnnulusMeshInvarianceFitRow> {
    let mut rows = Vec::new();
    let models = summaries
        .iter()
        .map(|row| row.model)
        .collect::<BTreeSet<_>>();
    for model in models {
        for regime in [AnnulusRegime::B, AnnulusRegime::D] {
            let mut regime_rows = summaries
                .iter()
                .filter(|row| row.model == model && row.regime == regime)
                .collect::<Vec<_>>();
            regime_rows.sort_by(|lhs, rhs| lhs.mesh_size.total_cmp(&rhs.mesh_size));
            if regime_rows.len() < 2 {
                continue;
            }
            let tail_count = convergence_tail_count.clamp(2, regime_rows.len());
            let finest = regime_rows[0];
            let coarser = regime_rows[1];
            for (family, threshold) in MESH_VARIANCE_QOI_FAMILIES {
                let coarser_value = mesh_summary_value(coarser, family);
                let finest_value = mesh_summary_value(finest, family);
                let final_two_relative_change = relative_change(coarser_value, finest_value);
                let coarse_to_fine_pairs = regime_rows
                    .iter()
                    .rev()
                    .map(|row| (row.mesh_size, mesh_summary_value(row, family)))
                    .collect::<Vec<_>>();
                let coarse_to_fine_values = coarse_to_fine_pairs
                    .iter()
                    .map(|(_, value)| *value)
                    .collect::<Vec<_>>();
                let coarse_to_finest_relative_change = relative_change(
                    coarse_to_fine_values[0],
                    *coarse_to_fine_values.last().unwrap(),
                );
                let trend = mesh_refinement_trend(&coarse_to_fine_values, threshold);
                let tail_pairs = regime_rows
                    .iter()
                    .take(tail_count)
                    .map(|row| (row.mesh_size, mesh_summary_value(row, family)))
                    .collect::<Vec<_>>();
                let tail_values = tail_pairs
                    .iter()
                    .map(|(_, value)| *value)
                    .collect::<Vec<_>>();
                let tail_relative_spread = relative_spread(&tail_values);
                let slope_log_variance_vs_log_h = log_log_slope(&tail_pairs);
                let model_ratio_to_full = full_finest_value(summaries, regime, family)
                    .map(|full_value| finest_value / full_value)
                    .unwrap_or(f64::NAN);
                let status = mesh_fit_status(
                    model,
                    family,
                    threshold,
                    tail_relative_spread,
                    slope_log_variance_vs_log_h,
                );
                rows.push(AnnulusMeshInvarianceFitRow {
                    profile,
                    model,
                    regime,
                    qoi_family: family.to_string(),
                    coarser_mesh_size: coarser.mesh_size,
                    finest_mesh_size: finest.mesh_size,
                    coarser_value,
                    finest_value,
                    relative_change: final_two_relative_change,
                    coarse_to_finest_relative_change,
                    slope_log_variance_vs_log_h,
                    tail_relative_spread,
                    tail_count,
                    model_ratio_to_full,
                    threshold,
                    trend: trend.to_string(),
                    status,
                });
            }
        }
    }
    rows
}

fn mesh_invariance_model_contrast_rows(
    profile: AnnulusMeshInvarianceProfile,
    summaries: &[AnnulusMeshInvarianceSummaryRow],
) -> Vec<AnnulusMeshModelContrastRow> {
    let mut rows = Vec::new();
    for no_correction in summaries
        .iter()
        .filter(|row| row.model == AnnulusModelKind::FeecSplitNoSpectralCorrection)
    {
        let Some(full) = summaries.iter().find(|row| {
            row.model == AnnulusModelKind::FeecGmrf
                && row.mesh_index == no_correction.mesh_index
                && row.regime == no_correction.regime
        }) else {
            continue;
        };
        for (family, _) in MESH_VARIANCE_QOI_FAMILIES {
            let no_correction_variance = mesh_summary_value(no_correction, family);
            let full_variance = mesh_summary_value(full, family);
            rows.push(AnnulusMeshModelContrastRow {
                profile,
                mesh_index: no_correction.mesh_index,
                mesh_size: no_correction.mesh_size,
                regime: no_correction.regime,
                qoi_family: family.to_string(),
                no_correction_variance,
                full_variance,
                ratio_no_correction_to_full: no_correction_variance / full_variance,
            });
        }
    }
    rows
}

fn mesh_summary_value(row: &AnnulusMeshInvarianceSummaryRow, family: &str) -> f64 {
    match family {
        "q_period_variance" => row.q_period_variance,
        "circulation_mean_variance" => row.circulation_mean_variance,
        "short_line_median_variance" => row.short_line_median_variance,
        "long_line_median_variance" => row.long_line_median_variance,
        "line_median_variance" => row.line_median_variance,
        "dense_line_away_median_variance" => row.dense_line_away_median_variance,
        "dense_line_away_p90_variance" => row.dense_line_away_p90_variance,
        "field_x_away_median_variance" => row.field_x_away_median_variance,
        "field_y_away_median_variance" => row.field_y_away_median_variance,
        "field_magnitude_away_median_variance" => row.field_magnitude_away_median_variance,
        _ => f64::NAN,
    }
}

fn full_finest_value(
    summaries: &[AnnulusMeshInvarianceSummaryRow],
    regime: AnnulusRegime,
    family: &str,
) -> Option<f64> {
    summaries
        .iter()
        .filter(|row| row.model == AnnulusModelKind::FeecGmrf && row.regime == regime)
        .min_by(|lhs, rhs| lhs.mesh_size.total_cmp(&rhs.mesh_size))
        .map(|row| mesh_summary_value(row, family))
        .filter(|value| value.is_finite() && value.abs() > EPS)
}

fn relative_spread(values: &[f64]) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.len() < 2 {
        return f64::NAN;
    }
    let min_value = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max_value = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let center = median(finite.iter().copied()).abs().max(EPS);
    (max_value - min_value).abs() / center
}

fn log_log_slope(mesh_value_pairs: &[(f64, f64)]) -> f64 {
    let points = mesh_value_pairs
        .iter()
        .copied()
        .filter(|(mesh_size, value)| {
            mesh_size.is_finite() && *mesh_size > 0.0 && value.is_finite() && *value > 0.0
        })
        .map(|(mesh_size, value)| (mesh_size.ln(), value.ln()))
        .collect::<Vec<_>>();
    if points.len() < 2 {
        return f64::NAN;
    }
    let x_mean = mean(points.iter().map(|(x, _)| *x));
    let y_mean = mean(points.iter().map(|(_, y)| *y));
    let numerator = points
        .iter()
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    let denominator = points
        .iter()
        .map(|(x, _)| {
            let delta = x - x_mean;
            delta * delta
        })
        .sum::<f64>();
    if denominator <= EPS {
        f64::NAN
    } else {
        numerator / denominator
    }
}

fn mesh_fit_status(
    model: AnnulusModelKind,
    family: &str,
    threshold: f64,
    tail_relative_spread: f64,
    slope_log_variance_vs_log_h: f64,
) -> String {
    let is_primary = PRIMARY_MESH_QOI_FAMILIES.contains(&family);
    let is_topological = matches!(family, "q_period_variance" | "circulation_mean_variance");
    if model == AnnulusModelKind::FeecSplitNoSpectralCorrection && !is_topological {
        return "baseline".to_string();
    }
    let spread_ok = tail_relative_spread.is_finite() && tail_relative_spread <= threshold;
    let slope_ok = !is_primary
        || (slope_log_variance_vs_log_h.is_finite() && slope_log_variance_vs_log_h.abs() <= 0.25);
    if spread_ok && slope_ok {
        "pass".to_string()
    } else {
        "fail".to_string()
    }
}

fn mesh_refinement_trend(values: &[f64], threshold: f64) -> &'static str {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return "nonfinite";
    }
    let total_change = relative_change(values[0], *values.last().unwrap());
    if total_change <= threshold {
        return "stable";
    }
    let monotone_growth = values.windows(2).all(|pair| {
        let tolerance = pair[0].abs().max(pair[1].abs()).max(1.0) * 1.0e-8;
        pair[1] > pair[0] + tolerance
    });
    if monotone_growth {
        "monotone_growth"
    } else {
        "mixed"
    }
}

fn mesh_qoi_kind(kind: AnnulusObservationKind, label: &str) -> String {
    match kind {
        AnnulusObservationKind::DenseFieldX => "field_x",
        AnnulusObservationKind::DenseFieldY => "field_y",
        AnnulusObservationKind::HeldoutLine if label.starts_with("dense_line_") => "dense_line",
        AnnulusObservationKind::HeldoutLine if label.starts_with("mesh_qoi_line_") => {
            "heldout_line_physical_long"
        }
        AnnulusObservationKind::HeldoutLine if label.starts_with("heldout_line_") => {
            "heldout_line_short"
        }
        AnnulusObservationKind::HeldoutLine => "heldout_line_other",
        AnnulusObservationKind::HeldoutResidual => "heldout_residual",
        AnnulusObservationKind::HeldoutCirculation => "heldout_circulation",
        AnnulusObservationKind::HeldoutHomologousDifference => "heldout_homologous_difference",
        _ => kind.as_str(),
    }
    .to_string()
}

fn probe_metadata(
    observation: &AnnulusObservation,
    direct_support: &BTreeSet<usize>,
) -> AnnulusProbeMetadata {
    let overlap = observation
        .entries
        .iter()
        .map(|(col, _)| *col)
        .collect::<BTreeSet<_>>()
        .intersection(direct_support)
        .count();
    AnnulusProbeMetadata {
        direct_support_overlap_count: overlap,
        is_away_probe: overlap == 0,
        probe_family: mesh_probe_family(observation.kind, &observation.label),
    }
}

fn mesh_probe_family(kind: AnnulusObservationKind, label: &str) -> String {
    match kind {
        AnnulusObservationKind::DenseFieldX => "field_x",
        AnnulusObservationKind::DenseFieldY => "field_y",
        AnnulusObservationKind::HeldoutLine if label.starts_with("dense_line_tangential") => {
            "dense_line_tangential"
        }
        AnnulusObservationKind::HeldoutLine if label.starts_with("dense_line_radial") => {
            "dense_line_radial"
        }
        AnnulusObservationKind::HeldoutLine if label.starts_with("dense_line_diagonal") => {
            "dense_line_diagonal"
        }
        AnnulusObservationKind::HeldoutLine if label.starts_with("mesh_qoi_line_") => {
            "heldout_line_physical_long"
        }
        AnnulusObservationKind::HeldoutLine if label.starts_with("heldout_line_") => {
            "heldout_line_short"
        }
        _ => kind.as_str(),
    }
    .to_string()
}

fn direct_support_edges(
    regime: AnnulusRegime,
    line: &[AnnulusObservation],
    circulation: &[AnnulusObservation],
) -> BTreeSet<usize> {
    let mut support = BTreeSet::new();
    for observation in line {
        support.extend(observation.entries.iter().map(|(col, _)| *col));
    }
    if regime == AnnulusRegime::D {
        for observation in circulation {
            support.extend(observation.entries.iter().map(|(col, _)| *col));
        }
    }
    support
}

fn training_for_regime(
    regime: AnnulusRegime,
    line: &[AnnulusObservation],
    residual: &[AnnulusObservation],
    circulation: &[AnnulusObservation],
) -> Vec<AnnulusObservation> {
    let mut training = line.to_vec();
    match regime {
        AnnulusRegime::A => {}
        AnnulusRegime::B => training.extend_from_slice(residual),
        AnnulusRegime::C => training.extend_from_slice(circulation),
        AnnulusRegime::D => {
            training.extend_from_slice(residual);
            training.extend_from_slice(circulation);
        }
    }
    training
}

fn observations_from_supports(
    kind: AnnulusObservationKind,
    supports: &[AnnulusSupport],
    truth: &FeecVector,
    noise_variance: f64,
) -> Vec<AnnulusObservation> {
    supports
        .iter()
        .map(|support| {
            let truth_value = row_dot(&support.entries, truth);
            AnnulusObservation {
                kind,
                label: support.label.clone(),
                entries: support.entries.clone(),
                truth_value,
                observed_value: truth_value,
                noise_variance,
            }
        })
        .collect()
}

fn sample_noisy_observations(
    observations: &[AnnulusObservation],
    sample_noise: bool,
    seed: u64,
) -> Vec<AnnulusObservation> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    observations
        .iter()
        .cloned()
        .map(|mut observation| {
            let noise = if sample_noise {
                rng.sample::<f64, _>(StandardNormal) * observation.noise_variance.sqrt()
            } else {
                0.0
            };
            observation.observed_value = observation.truth_value + noise;
            observation
        })
        .collect()
}

fn annulus_weighted_observation_rows(
    observations: &[AnnulusObservation],
) -> Vec<FeecWeightedObservationRow> {
    observations
        .iter()
        .map(|observation| FeecWeightedObservationRow {
            entries: observation.entries.clone(),
            observed_value: observation.observed_value,
            noise_variance: observation.noise_variance,
        })
        .collect()
}

fn observation_matrix(
    observations: &[AnnulusObservation],
    ncols: usize,
    scaled: bool,
) -> Result<FeecCsr, Box<dyn Error>> {
    feec_observation_matrix(
        &annulus_weighted_observation_rows(observations),
        ncols,
        scaled,
        VARIANCE_FLOOR,
    )
}

fn normalized_harmonic_one_form(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    reference_loop: &AnnulusSupport,
) -> Result<FeecVector, Box<dyn Error>> {
    let harmonic_basis_raw =
        compute_harmonic_basis_1form(topology, metric, 1, None).map_err(invalid_data)?;
    let harmonic_basis = mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, mass_1form)
        .map_err(invalid_data)?;
    if harmonic_basis.ncols() != 1 {
        return Err(invalid_data(format!(
            "expected one harmonic 1-form, got {}",
            harmonic_basis.ncols()
        ))
        .into());
    }
    let column = harmonic_basis.column(0).into_owned();
    let period = row_dot(&reference_loop.entries, &column);
    if !period.is_finite() || period.abs() <= 1e-10 {
        return Err(invalid_data("harmonic basis has near-zero reference period").into());
    }
    Ok(column / period)
}

fn analytic_truth_h(topology: &Complex, coords: &MeshCoords) -> FeecVector {
    let mut values = FeecVector::zeros(topology.nsimplices(1));
    for edge in topology.edges().handle_iter() {
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        let ca = coords.coord(a);
        let cb = coords.coord(b);
        let phi_a = analytic_phi(ca[0], ca[1]);
        let phi_b = analytic_phi(cb[0], cb[1]);
        let theta_a = ca[1].atan2(ca[0]);
        let theta_b = cb[1].atan2(cb[0]);
        values[edge.kidx()] = (phi_b - phi_a) + unwrap_angle_delta(theta_b - theta_a) / TWO_PI;
    }
    values
}

fn analytic_phi(x: f64, y: f64) -> f64 {
    0.3 * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
}

fn unwrap_angle_delta(mut delta: f64) -> f64 {
    while delta <= -std::f64::consts::PI {
        delta += TWO_PI;
    }
    while delta > std::f64::consts::PI {
        delta -= TWO_PI;
    }
    delta
}

fn build_line_supports(
    topology: &Complex,
    coords: &MeshCoords,
    tangential_count: usize,
    radial_count: usize,
    random_count: usize,
    seed: u64,
    prefix: &str,
) -> Result<Vec<AnnulusSupport>, Box<dyn Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut supports = Vec::new();
    append_segment_supports(
        topology,
        coords,
        &mut supports,
        &mut rng,
        tangential_count,
        prefix,
        "tangential",
        SegmentKind::Tangential,
    )?;
    append_segment_supports(
        topology,
        coords,
        &mut supports,
        &mut rng,
        radial_count,
        prefix,
        "radial",
        SegmentKind::Radial,
    )?;
    append_segment_supports(
        topology,
        coords,
        &mut supports,
        &mut rng,
        random_count,
        prefix,
        "random",
        SegmentKind::Random,
    )?;
    Ok(supports)
}

fn build_mesh_invariance_line_supports(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<AnnulusSupport>, Box<dyn Error>> {
    let mut supports = Vec::new();
    let arcs = [
        (
            1.20,
            0.15 * std::f64::consts::PI,
            0.65 * std::f64::consts::PI,
        ),
        (
            1.38,
            0.75 * std::f64::consts::PI,
            1.25 * std::f64::consts::PI,
        ),
        (
            1.56,
            1.35 * std::f64::consts::PI,
            1.85 * std::f64::consts::PI,
        ),
        (
            1.76,
            1.85 * std::f64::consts::PI,
            2.35 * std::f64::consts::PI,
        ),
    ];
    for (index, (radius, start, end)) in arcs.into_iter().enumerate() {
        let points = (0..=16)
            .map(|i| {
                let t = i as f64 / 16.0;
                polar_point(radius, start + t * (end - start))
            })
            .collect::<Vec<_>>();
        supports.push(polyline_support(
            topology,
            coords,
            &format!("mesh_qoi_line_arc_{index}"),
            &points,
            false,
        )?);
    }

    for (index, angle) in [
        0.10 * std::f64::consts::PI,
        0.60 * std::f64::consts::PI,
        1.10 * std::f64::consts::PI,
        1.60 * std::f64::consts::PI,
    ]
    .into_iter()
    .enumerate()
    {
        let points = (0..=12)
            .map(|i| {
                let t = i as f64 / 12.0;
                polar_point(1.14 + 0.72 * t, angle)
            })
            .collect::<Vec<_>>();
        supports.push(polyline_support(
            topology,
            coords,
            &format!("mesh_qoi_line_radial_{index}"),
            &points,
            false,
        )?);
    }
    Ok(supports)
}

fn build_dense_posterior_line_supports(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<AnnulusSupport>, Box<dyn Error>> {
    let mut supports = Vec::new();
    let tangential_radii = [1.12, 1.24, 1.36, 1.48, 1.60, 1.72, 1.84];
    for (radius_index, radius) in tangential_radii.into_iter().enumerate() {
        for angle_index in 0..12 {
            let center = TWO_PI * (angle_index as f64 + 0.31) / 12.0;
            let half_angle = 0.14 / radius;
            let points = (0..=4)
                .map(|i| {
                    let t = -half_angle + 2.0 * half_angle * i as f64 / 4.0;
                    polar_point(radius, center + t)
                })
                .collect::<Vec<_>>();
            supports.push(polyline_support(
                topology,
                coords,
                &format!("dense_line_tangential_r{radius_index}_a{angle_index}"),
                &points,
                false,
            )?);
        }
    }

    let radial_bands = [(1.10, 1.36), (1.30, 1.56), (1.50, 1.76), (1.66, 1.92)];
    for (band_index, (start_radius, end_radius)) in radial_bands.into_iter().enumerate() {
        for angle_index in 0..24 {
            let angle = TWO_PI * (angle_index as f64 + 0.17) / 24.0;
            let points = vec![
                polar_point(start_radius, angle),
                polar_point(end_radius, angle),
            ];
            supports.push(polyline_support(
                topology,
                coords,
                &format!("dense_line_radial_b{band_index}_a{angle_index}"),
                &points,
                false,
            )?);
        }
    }

    let mut diagonal_count = 0usize;
    for radius_index in 0..8 {
        let radius = 1.16 + 0.68 * radius_index as f64 / 7.0;
        for angle_index in 0..15 {
            let angle = TWO_PI * (angle_index as f64 + 0.43) / 15.0;
            let center = polar_point(radius, angle);
            let direction = angle + 0.35 * std::f64::consts::PI;
            let half_length = 0.15;
            let a = [
                center[0] - half_length * direction.cos(),
                center[1] - half_length * direction.sin(),
                0.0,
            ];
            let b = [
                center[0] + half_length * direction.cos(),
                center[1] + half_length * direction.sin(),
                0.0,
            ];
            if !inside_annulus(a) || !inside_annulus(b) {
                continue;
            }
            supports.push(polyline_support(
                topology,
                coords,
                &format!("dense_line_diagonal_{diagonal_count}"),
                &[a, b],
                false,
            )?);
            diagonal_count += 1;
        }
    }
    Ok(supports)
}

fn build_dense_field_observations(
    topology: &Complex,
    coords: &MeshCoords,
    truth: &FeecVector,
    noise_variance: f64,
) -> Result<Vec<AnnulusObservation>, Box<dyn Error>> {
    let points = dense_field_sample_points();
    let mut observations = Vec::with_capacity(2 * points.len());
    for sample in points {
        let Some(x_support) =
            field_component_support(topology, coords, &sample, 0, &format!("{}_x", sample.label))?
        else {
            continue;
        };
        let Some(y_support) =
            field_component_support(topology, coords, &sample, 1, &format!("{}_y", sample.label))?
        else {
            continue;
        };
        observations.extend(observations_from_supports(
            AnnulusObservationKind::DenseFieldX,
            &[x_support],
            truth,
            noise_variance,
        ));
        observations.extend(observations_from_supports(
            AnnulusObservationKind::DenseFieldY,
            &[y_support],
            truth,
            noise_variance,
        ));
    }
    Ok(observations)
}

fn dense_field_sample_points() -> Vec<AnnulusFieldSamplePoint> {
    let mut points = Vec::with_capacity(120);
    for radius_index in 0..10 {
        let radius = 1.12 + 0.76 * radius_index as f64 / 9.0;
        for angle_index in 0..12 {
            let theta = TWO_PI * (angle_index as f64 + 0.5) / 12.0 + 0.037 * radius_index as f64;
            points.push(AnnulusFieldSamplePoint {
                label: format!("dense_field_{:03}", points.len()),
                point: polar_point(radius, theta),
            });
        }
    }
    points
}

fn field_component_support(
    topology: &Complex,
    coords: &MeshCoords,
    sample: &AnnulusFieldSamplePoint,
    component_index: usize,
    label: &str,
) -> Result<Option<AnnulusSupport>, Box<dyn Error>> {
    if component_index >= coords.dim() {
        return Ok(None);
    }
    let point = point_coord_vector(coords, sample.point);
    for cell in topology.cells().handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let bary = cell_coords.global2bary(&point);
        if !barycentric_inside(&bary, 1e-9) {
            continue;
        }
        let local_point = cell_coords.global2local(&point);
        let mut entries = BTreeMap::<usize, f64>::new();
        for dof_simp in cell.mesh_subsimps(1) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topology.dim(), local_dof_simp);
            let ambient_value = cell_coords
                .lift_form(&lsf.at_point(&local_point))
                .into_grade1();
            let coefficient = ambient_value[component_index];
            if coefficient.abs() > EPS {
                *entries.entry(dof_simp.kidx()).or_insert(0.0) += coefficient;
            }
        }
        let entries = entries
            .into_iter()
            .filter(|(_, value)| value.abs() > EPS)
            .collect::<Vec<_>>();
        return Ok(Some(AnnulusSupport {
            label: label.to_string(),
            entries,
        }));
    }
    Ok(None)
}

fn point_coord_vector(coords: &MeshCoords, point: [f64; 3]) -> FeecVector {
    FeecVector::from_iterator(coords.dim(), (0..coords.dim()).map(|index| point[index]))
}

fn barycentric_inside(bary: &FeecVector, tolerance: f64) -> bool {
    (bary.sum() - 1.0).abs() <= 10.0 * tolerance
        && bary
            .iter()
            .all(|value| *value >= -tolerance && *value <= 1.0 + tolerance)
}

#[derive(Clone, Copy)]
enum SegmentKind {
    Tangential,
    Radial,
    Random,
}

// Segment support generation records the geometry-independent mesh inputs and the
// annulus study's sampling labels separately; annulus smoke tests validate the set.
#[allow(clippy::too_many_arguments)]
fn append_segment_supports(
    topology: &Complex,
    coords: &MeshCoords,
    supports: &mut Vec<AnnulusSupport>,
    rng: &mut rand::rngs::StdRng,
    count: usize,
    prefix: &str,
    label: &str,
    kind: SegmentKind,
) -> Result<(), Box<dyn Error>> {
    let mut attempts = 0usize;
    while supports
        .iter()
        .filter(|support| support.label.contains(label))
        .count()
        < count
        && attempts < count.max(1) * 80
    {
        attempts += 1;
        let local_index = supports
            .iter()
            .filter(|support| support.label.contains(label))
            .count();
        let angle = biased_angle(rng, local_index < (count * 7 / 10));
        let radius = rng.gen_range(1.15..1.85);
        let points = match kind {
            SegmentKind::Tangential => {
                let half_angle = 0.055 / radius;
                (0..=4)
                    .map(|i| {
                        let t = -half_angle + 2.0 * half_angle * i as f64 / 4.0;
                        polar_point(radius, angle + t)
                    })
                    .collect::<Vec<_>>()
            }
            SegmentKind::Radial => {
                let dr = 0.11;
                vec![
                    polar_point(radius - dr, angle),
                    polar_point(radius + dr, angle),
                ]
            }
            SegmentKind::Random => {
                let center = polar_point(radius, angle);
                let direction = rng.gen_range(0.0..TWO_PI);
                let half_length = 0.08;
                let a = [
                    center[0] - half_length * direction.cos(),
                    center[1] - half_length * direction.sin(),
                    0.0,
                ];
                let b = [
                    center[0] + half_length * direction.cos(),
                    center[1] + half_length * direction.sin(),
                    0.0,
                ];
                if !inside_annulus(a) || !inside_annulus(b) {
                    continue;
                }
                vec![a, b]
            }
        };
        let support = match polyline_support(
            topology,
            coords,
            &format!("{prefix}_{label}_{local_index}"),
            &points,
            false,
        ) {
            Ok(support) if !support.entries.is_empty() => support,
            _ => continue,
        };
        supports.push(support);
    }
    if supports
        .iter()
        .filter(|support| support.label.contains(label))
        .count()
        < count
    {
        return Err(invalid_data(format!(
            "could only build {} {label} supports out of requested {count}",
            supports
                .iter()
                .filter(|support| support.label.contains(label))
                .count()
        ))
        .into());
    }
    Ok(())
}

fn build_residual_supports(
    topology: &Complex,
    count: usize,
    seed: u64,
    prefix: &str,
) -> Vec<AnnulusSupport> {
    let d1 = de_rham::exterior_derivative_matrix(topology, 1);
    let rows = csr_rows(&d1);
    let mut indices = (0..rows.len()).collect::<Vec<_>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices
        .into_iter()
        .take(count.min(rows.len()))
        .map(|face| AnnulusSupport {
            label: format!("{prefix}_face_{face}"),
            entries: rows[face].clone(),
        })
        .collect()
}

fn build_heldout_loop_supports(
    topology: &Complex,
    coords: &MeshCoords,
    count: usize,
    seed: u64,
    prefix: &str,
) -> Result<Vec<AnnulusSupport>, Box<dyn Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut supports = Vec::new();
    for index in 0..count {
        let radius = 1.12 + 0.76 * (index as f64 + 0.5) / count.max(1) as f64;
        let offset = rng.gen_range(0.0..TWO_PI);
        let deformed = index % 3 == 2;
        supports.push(build_loop_support(
            topology,
            coords,
            radius,
            offset,
            deformed,
            &format!("{prefix}_{index}"),
        )?);
    }
    Ok(supports)
}

fn build_loop_support(
    topology: &Complex,
    coords: &MeshCoords,
    radius: f64,
    angle_offset: f64,
    deformed: bool,
    label: &str,
) -> Result<AnnulusSupport, Box<dyn Error>> {
    let points = (0..64)
        .map(|i| {
            let theta = angle_offset + TWO_PI * i as f64 / 64.0;
            let local_radius = if deformed {
                (radius + 0.06 * (3.0 * theta + 0.4).sin()).clamp(1.08, 1.92)
            } else {
                radius
            };
            polar_point(local_radius, theta)
        })
        .collect::<Vec<_>>();
    polyline_support(topology, coords, label, &points, true)
}

fn build_homologous_difference_supports(loops: &[AnnulusSupport]) -> Vec<AnnulusSupport> {
    let mut supports = Vec::new();
    for i in 0..loops.len() {
        for j in i + 1..loops.len() {
            let mut entries = loops[i].entries.clone();
            entries.extend(loops[j].entries.iter().map(|(col, value)| (*col, -*value)));
            supports.push(AnnulusSupport {
                label: format!("{}_minus_{}", loops[i].label, loops[j].label),
                entries,
            });
        }
    }
    supports
}

fn polyline_support(
    topology: &Complex,
    coords: &MeshCoords,
    label: &str,
    points: &[[f64; 3]],
    close_path: bool,
) -> Result<AnnulusSupport, Box<dyn Error>> {
    let support = de_rham::edge_path_integral_support(topology, coords, label, points, close_path)
        .map_err(invalid_data)?;
    Ok(AnnulusSupport {
        label: support.label,
        entries: support.entries,
    })
}

fn split_line_counts(total: usize) -> (usize, usize, usize) {
    let tangential = (0.4 * total as f64).round() as usize;
    let radial = (0.3 * total as f64).round() as usize;
    let random = total.saturating_sub(tangential + radial);
    (tangential, radial, random)
}

fn biased_angle(rng: &mut rand::rngs::StdRng, sector_bias: bool) -> f64 {
    if sector_bias {
        rng.gen_range(0.15 * std::f64::consts::PI..0.55 * std::f64::consts::PI)
    } else {
        rng.gen_range(0.0..TWO_PI)
    }
}

fn polar_point(radius: f64, theta: f64) -> [f64; 3] {
    [radius * theta.cos(), radius * theta.sin(), 0.0]
}

fn inside_annulus(point: [f64; 3]) -> bool {
    let r = (point[0] * point[0] + point[1] * point[1]).sqrt();
    r > INNER_RADIUS + 0.03 && r < OUTER_RADIUS - 0.03
}

fn csr_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            rows[row].push((col, *value));
        }
    }
    rows
}

fn row_dot(entries: &[(usize, f64)], values: &FeecVector) -> f64 {
    entries
        .iter()
        .filter(|(index, _)| *index < values.len())
        .map(|(index, weight)| weight * values[*index])
        .sum()
}

fn gaussian_nlpd(truth: f64, mean: f64, variance: f64) -> f64 {
    let variance = variance.max(VARIANCE_FLOOR);
    let residual = truth - mean;
    0.5 * ((TWO_PI * variance).ln() + residual * residual / variance)
}

fn rmse(values: impl Iterator<Item = f64>) -> f64 {
    let finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    (finite.iter().map(|value| value * value).sum::<f64>() / finite.len() as f64).sqrt()
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let mid = finite.len() / 2;
    if finite.len() % 2 == 0 {
        0.5 * (finite[mid - 1] + finite[mid])
    } else {
        finite[mid]
    }
}

fn percentile(values: impl Iterator<Item = f64>, probability: f64) -> f64 {
    let mut finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() || !probability.is_finite() {
        return f64::NAN;
    }
    finite.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let p = probability.clamp(0.0, 1.0);
    let index = ((finite.len() - 1) as f64 * p).round() as usize;
    finite[index]
}

fn variance(values: impl Iterator<Item = f64>) -> f64 {
    let finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.len() < 2 {
        return f64::NAN;
    }
    let avg = finite.iter().sum::<f64>() / finite.len() as f64;
    finite
        .iter()
        .map(|value| {
            let delta = value - avg;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64
}

fn coverage(values: impl Iterator<Item = f64>, z: f64) -> f64 {
    let finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().filter(|value| value.abs() <= z).count() as f64 / finite.len() as f64
}

fn relative_change(value: f64, reference: f64) -> f64 {
    if !value.is_finite() || !reference.is_finite() {
        return f64::NAN;
    }
    (value - reference).abs() / reference.abs().max(EPS)
}

fn mesh_size_tag(mesh_size: f64) -> String {
    format!("{mesh_size:.5}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .replace('.', "p")
}

fn ensure_annulus_mesh(config: &AnnulusHFormulationConfig) -> Result<(), Box<dyn Error>> {
    if !config.force_mesh && config.mesh_path.is_file() {
        return Ok(());
    }
    if let Some(parent) = config.geo_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = config.mesh_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.geo_path, annulus_geo(config.mesh_size))?;
    let status = Command::new("gmsh")
        .arg("-2")
        .arg(&config.geo_path)
        .arg("-format")
        .arg("msh4")
        .arg("-o")
        .arg(&config.mesh_path)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "gmsh failed while generating `{}`",
            config.mesh_path.display()
        ))
        .into());
    }
    Ok(())
}

fn annulus_geo(mesh_size: f64) -> String {
    format!(
        r#"lc = {mesh_size:.12};
Mesh.Algorithm = 6;
Point(1) = {{0, 0, 0, lc}};
Point(2) = {{2, 0, 0, lc}};
Point(3) = {{0, 2, 0, lc}};
Point(4) = {{-2, 0, 0, lc}};
Point(5) = {{0, -2, 0, lc}};
Point(6) = {{1, 0, 0, lc}};
Point(7) = {{0, 1, 0, lc}};
Point(8) = {{-1, 0, 0, lc}};
Point(9) = {{0, -1, 0, lc}};
Circle(1) = {{2, 1, 3}};
Circle(2) = {{3, 1, 4}};
Circle(3) = {{4, 1, 5}};
Circle(4) = {{5, 1, 2}};
Circle(5) = {{6, 1, 7}};
Circle(6) = {{7, 1, 8}};
Circle(7) = {{8, 1, 9}};
Circle(8) = {{9, 1, 6}};
Curve Loop(1) = {{1, 2, 3, 4}};
Curve Loop(2) = {{5, 6, 7, 8}};
Plane Surface(1) = {{1, 2}};
Physical Surface("annulus") = {{1}};
"#
    )
}

fn validate_config(config: &AnnulusHFormulationConfig) -> Result<(), Box<dyn Error>> {
    for (name, value) in [
        ("mesh_size", config.mesh_size),
        ("tau0", config.tau0),
        ("tau1", config.tau1),
        ("harmonic_prior_std", config.harmonic_prior_std),
        ("line_noise_variance", config.line_noise_variance),
        ("residual_noise_variance", config.residual_noise_variance),
        (
            "circulation_noise_variance",
            config.circulation_noise_variance,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_input(format!("{name} must be finite and positive")).into());
        }
    }
    if config.line_tangential_count + config.line_radial_count + config.line_random_count == 0 {
        return Err(invalid_input("at least one line observation is required").into());
    }
    if config.heldout_loop_count < 2 {
        return Err(invalid_input("at least two held-out loops are required").into());
    }
    if config.noise_trial_count == 0 {
        return Err(invalid_input("noise_trial_count must be positive").into());
    }
    Ok(())
}

fn validate_mesh_invariance_config(
    config: &AnnulusMeshInvarianceConfig,
) -> Result<(), Box<dyn Error>> {
    if config.mesh_sizes.len() < 2 {
        return Err(invalid_input("mesh invariance requires at least two mesh sizes").into());
    }
    for mesh_size in &config.mesh_sizes {
        if !mesh_size.is_finite() || *mesh_size <= 0.0 {
            return Err(invalid_input("all mesh sizes must be finite and positive").into());
        }
    }
    if config.residual_count == 0 {
        return Err(invalid_input("residual_count must be positive").into());
    }
    if config.heldout_loop_count < 2 {
        return Err(invalid_input("heldout_loop_count must be at least two").into());
    }
    if config.convergence_tail_count < 2 {
        return Err(invalid_input("convergence_tail_count must be at least two").into());
    }
    if config.model_kinds.is_empty() {
        return Err(invalid_input("mesh invariance requires at least one model").into());
    }
    for model in &config.model_kinds {
        if !matches!(
            model,
            AnnulusModelKind::FeecGmrf | AnnulusModelKind::FeecSplitNoSpectralCorrection
        ) {
            return Err(invalid_input(format!(
                "mesh invariance supports only feec_gmrf and feec_split_no_spectral_correction, got {}",
                model.as_str()
            ))
            .into());
        }
    }
    Ok(())
}

fn validate_efficiency_config(config: &AnnulusEfficiencyConfig) -> Result<(), Box<dyn Error>> {
    if config.mesh_sizes.is_empty() {
        return Err(invalid_input("efficiency sweep requires at least one mesh size").into());
    }
    for mesh_size in &config.mesh_sizes {
        if !mesh_size.is_finite() || *mesh_size <= 0.0 {
            return Err(invalid_input("all mesh sizes must be finite and positive").into());
        }
    }
    if let Some(limit) = config.dense_gp_vertex_limit {
        if limit == 0 {
            return Err(invalid_input("dense_gp_vertex_limit must be positive").into());
        }
    }
    validate_config(&config.base)
}

fn write_mesh_metadata_csv(result: &AnnulusMeshInvarianceResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "mesh_index,mesh_size,model,vertex_count,edge_count,face_count,harmonic_1_dimension,reference_period_truth,reference_period_psi,truth_closure_l2,truth_closure_max_abs"
    )?;
    for row in &result.mesh_metadata {
        writeln!(
            writer,
            "{},{:.12e},{},{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.mesh_index,
            row.mesh_size,
            row.model.as_str(),
            row.vertex_count,
            row.edge_count,
            row.face_count,
            row.harmonic_1_dimension,
            row.reference_period_truth,
            row.reference_period_psi,
            row.truth_closure_l2,
            row.truth_closure_max_abs
        )?;
    }
    Ok(())
}

fn write_mesh_qoi_csv(result: &AnnulusMeshInvarianceResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "mesh_index,mesh_size,model,regime,qoi_kind,qoi_label,truth,mean,latent_variance,posterior_sd,support_entries,direct_support_overlap_count,is_away_probe,probe_family"
    )?;
    for row in &result.qoi_rows {
        writeln!(
            writer,
            "{},{:.12e},{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{}",
            row.mesh_index,
            row.mesh_size,
            row.model.as_str(),
            row.regime.as_str(),
            row.qoi_kind,
            row.qoi_label,
            row.truth,
            row.mean,
            row.latent_variance,
            row.posterior_sd,
            row.support_entries,
            row.direct_support_overlap_count,
            row.is_away_probe,
            row.probe_family
        )?;
    }
    Ok(())
}

fn write_mesh_invariance_summary_csv(
    result: &AnnulusMeshInvarianceResult,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "profile,mesh_index,mesh_size,model,regime,vertex_count,edge_count,face_count,q_period_variance,q_period_sd,circulation_count,circulation_mean_variance,circulation_median_variance,circulation_mean_sd,circulation_median_sd,line_count,line_mean_variance,line_median_variance,line_mean_sd,line_median_sd,short_line_count,short_line_mean_variance,short_line_median_variance,short_line_mean_sd,short_line_median_sd,long_line_count,long_line_mean_variance,long_line_median_variance,long_line_mean_sd,long_line_median_sd,dense_line_away_count,dense_line_away_median_variance,dense_line_away_p90_variance,dense_line_away_median_sd,dense_line_away_p90_sd,field_x_away_count,field_x_away_median_variance,field_x_away_median_sd,field_y_away_count,field_y_away_median_variance,field_y_away_median_sd,field_magnitude_away_count,field_magnitude_away_median_variance,field_magnitude_away_median_sd,homologous_count,homologous_mean_variance,homologous_median_variance,homologous_mean_sd,homologous_median_sd,max_relative_change_to_finest_variance"
    )?;
    for row in &result.summary_rows {
        writeln!(
            writer,
            "{},{},{:.12e},{},{},{},{},{},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.12e},{:.12e},{},{:.12e},{:.12e},{},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.profile.as_str(),
            row.mesh_index,
            row.mesh_size,
            row.model.as_str(),
            row.regime.as_str(),
            row.vertex_count,
            row.edge_count,
            row.face_count,
            row.q_period_variance,
            row.q_period_sd,
            row.circulation_count,
            row.circulation_mean_variance,
            row.circulation_median_variance,
            row.circulation_mean_sd,
            row.circulation_median_sd,
            row.line_count,
            row.line_mean_variance,
            row.line_median_variance,
            row.line_mean_sd,
            row.line_median_sd,
            row.short_line_count,
            row.short_line_mean_variance,
            row.short_line_median_variance,
            row.short_line_mean_sd,
            row.short_line_median_sd,
            row.long_line_count,
            row.long_line_mean_variance,
            row.long_line_median_variance,
            row.long_line_mean_sd,
            row.long_line_median_sd,
            row.dense_line_away_count,
            row.dense_line_away_median_variance,
            row.dense_line_away_p90_variance,
            row.dense_line_away_median_sd,
            row.dense_line_away_p90_sd,
            row.field_x_away_count,
            row.field_x_away_median_variance,
            row.field_x_away_median_sd,
            row.field_y_away_count,
            row.field_y_away_median_variance,
            row.field_y_away_median_sd,
            row.field_magnitude_away_count,
            row.field_magnitude_away_median_variance,
            row.field_magnitude_away_median_sd,
            row.homologous_count,
            row.homologous_mean_variance,
            row.homologous_median_variance,
            row.homologous_mean_sd,
            row.homologous_median_sd,
            row.max_relative_change_to_finest_variance
        )?;
    }
    Ok(())
}

fn write_mesh_invariance_fit_csv(
    result: &AnnulusMeshInvarianceResult,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "profile,model,regime,qoi_family,coarser_mesh_size,finest_mesh_size,coarser_value,finest_value,relative_change,coarse_to_finest_relative_change,slope_log_variance_vs_log_h,tail_relative_spread,tail_count,model_ratio_to_full,threshold,trend,status"
    )?;
    for row in &result.fit_rows {
        writeln!(
            writer,
            "{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.12e},{:.12e},{},{}",
            row.profile.as_str(),
            row.model.as_str(),
            row.regime.as_str(),
            row.qoi_family,
            row.coarser_mesh_size,
            row.finest_mesh_size,
            row.coarser_value,
            row.finest_value,
            row.relative_change,
            row.coarse_to_finest_relative_change,
            row.slope_log_variance_vs_log_h,
            row.tail_relative_spread,
            row.tail_count,
            row.model_ratio_to_full,
            row.threshold,
            row.trend,
            row.status
        )?;
    }
    Ok(())
}

fn write_mesh_model_contrast_csv(
    result: &AnnulusMeshInvarianceResult,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "profile,mesh_index,mesh_size,regime,qoi_family,no_correction_variance,full_variance,ratio_no_correction_to_full"
    )?;
    for row in &result.contrast_rows {
        writeln!(
            writer,
            "{},{},{:.12e},{},{},{:.12e},{:.12e},{:.12e}",
            row.profile.as_str(),
            row.mesh_index,
            row.mesh_size,
            row.regime.as_str(),
            row.qoi_family,
            row.no_correction_variance,
            row.full_variance,
            row.ratio_no_correction_to_full
        )?;
    }
    Ok(())
}

fn write_topology_summary_csv(result: &AnnulusHFormulationResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "vertex_count,edge_count,face_count,harmonic_1_dimension,reference_period_truth,reference_period_psi,truth_closure_l2,truth_closure_max_abs"
    )?;
    writeln!(
        writer,
        "{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e}",
        result.topology.vertex_count,
        result.topology.edge_count,
        result.topology.face_count,
        result.topology.harmonic_1_dimension,
        result.topology.reference_period_truth,
        result.topology.reference_period_psi,
        result.topology.truth_closure_l2,
        result.topology.truth_closure_max_abs
    )
}

fn write_trial_metrics_csv(result: &AnnulusHFormulationResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "trial,regime,model,training_observation_count,selected_kappa,selected_tau0,selected_tau1,validation_nlpd,rmse_line,residual_rmse,circ_rmse,homologous_difference_rmse,z_mean,z_variance,coverage_90,coverage_95,nlpd,circ_nlpd,topo_spread,q_mean,q_std,selected_build_seconds,selection_seconds,conditioning_seconds,prediction_seconds,selected_total_seconds,pipeline_total_seconds,latent_dimension,prior_precision_nnz,prior_precision_density,posterior_precision_nnz,posterior_precision_density,posterior_factor_nnz,posterior_fill_ratio"
    )?;
    for row in &result.trial_metrics {
        writeln!(
            writer,
            "{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{:.12e},{},{:.12e},{},{:.12e}",
            row.trial,
            row.regime.as_str(),
            row.model.as_str(),
            row.training_observation_count,
            row.selected_kappa,
            row.selected_tau0,
            row.selected_tau1,
            row.validation_nlpd,
            row.rmse_line,
            row.residual_rmse,
            row.circ_rmse,
            row.homologous_difference_rmse,
            row.z_mean,
            row.z_variance,
            row.coverage_90,
            row.coverage_95,
            row.nlpd,
            row.circ_nlpd,
            row.topo_spread,
            row.q_mean,
            row.q_std,
            row.selected_build_seconds,
            row.selection_seconds,
            row.conditioning_seconds,
            row.prediction_seconds,
            row.selected_total_seconds,
            row.pipeline_total_seconds,
            row.latent_dimension,
            row.prior_precision_nnz,
            row.prior_precision_density,
            row.posterior_precision_nnz,
            row.posterior_precision_density,
            row.posterior_factor_nnz,
            row.posterior_fill_ratio
        )?;
    }
    Ok(())
}

fn write_summary_metrics_csv(result: &AnnulusHFormulationResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "regime,model,trial_count,rmse_line,residual_rmse,circ_rmse,homologous_difference_rmse,z_mean,z_variance,coverage_90,coverage_95,nlpd,circ_nlpd,topo_spread,q_mean,q_std,selected_build_seconds,selection_seconds,conditioning_seconds,prediction_seconds,selected_total_seconds,pipeline_total_seconds,latent_dimension,prior_precision_nnz,prior_precision_density,posterior_precision_nnz,posterior_precision_density,posterior_factor_nnz,posterior_fill_ratio"
    )?;
    for row in &result.summary_rows {
        writeln!(
            writer,
            "{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.regime.as_str(),
            row.model.as_str(),
            row.trial_count,
            row.rmse_line,
            row.residual_rmse,
            row.circ_rmse,
            row.homologous_difference_rmse,
            row.z_mean,
            row.z_variance,
            row.coverage_90,
            row.coverage_95,
            row.nlpd,
            row.circ_nlpd,
            row.topo_spread,
            row.q_mean,
            row.q_std,
            row.selected_build_seconds,
            row.selection_seconds,
            row.conditioning_seconds,
            row.prediction_seconds,
            row.selected_total_seconds,
            row.pipeline_total_seconds,
            row.latent_dimension,
            row.prior_precision_nnz,
            row.prior_precision_density,
            row.posterior_precision_nnz,
            row.posterior_precision_density,
            row.posterior_factor_nnz,
            row.posterior_fill_ratio
        )?;
    }
    Ok(())
}

fn write_predictions_csv(result: &AnnulusHFormulationResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "trial,regime,model,kind,label,truth,mean,latent_variance,predictive_variance,z,nlpd"
    )?;
    for row in &result.predictions {
        writeln!(
            writer,
            "{},{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.trial,
            row.regime.as_str(),
            row.model.as_str(),
            row.kind.as_str(),
            row.label,
            row.truth,
            row.mean,
            row.latent_variance,
            row.predictive_variance,
            row.z,
            row.nlpd
        )?;
    }
    Ok(())
}

fn write_tuning_csv(result: &AnnulusHFormulationResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "model,selected,kappa,tau0,tau1,validation_nlpd")?;
    for row in &result.tuning_rows {
        writeln!(
            writer,
            "{},{},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.model.as_str(),
            row.selected,
            row.kappa,
            row.tau0,
            row.tau1,
            row.validation_nlpd
        )?;
    }
    Ok(())
}

fn write_efficiency_rows_csv(result: &AnnulusEfficiencyResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "mesh_index,mesh_size,model,status,vertex_count,edge_count,face_count,selected_kappa,selected_tau0,selected_tau1,build_seconds,conditioning_seconds,prediction_seconds,selected_total_seconds,latent_dimension,prior_precision_nnz,prior_precision_density,posterior_precision_nnz,posterior_precision_density,posterior_factor_nnz,posterior_fill_ratio,line_rmse,closure_rmse,period_rmse,homologous_consistency_rmse"
    )?;
    for row in &result.rows {
        writeln!(
            writer,
            "{},{:.12e},{},{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{:.12e},{},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.mesh_index,
            row.mesh_size,
            row.model.as_str(),
            row.status,
            row.vertex_count,
            row.edge_count,
            row.face_count,
            row.selected_kappa,
            row.selected_tau0,
            row.selected_tau1,
            row.build_seconds,
            row.conditioning_seconds,
            row.prediction_seconds,
            row.selected_total_seconds,
            row.latent_dimension,
            row.prior_precision_nnz,
            row.prior_precision_density,
            row.posterior_precision_nnz,
            row.posterior_precision_density,
            row.posterior_factor_nnz,
            row.posterior_fill_ratio,
            row.rmse_line,
            row.residual_rmse,
            row.circ_rmse,
            row.homologous_difference_rmse
        )?;
    }
    Ok(())
}

fn write_efficiency_speedup_csv(result: &AnnulusEfficiencyResult, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "mesh_index,mesh_size,vertex_count,gp_selected_total_seconds,feec_selected_total_seconds,speedup"
    )?;
    for row in &result.speedup_rows {
        writeln!(
            writer,
            "{},{:.12e},{},{:.12e},{:.12e},{:.12e}",
            row.mesh_index,
            row.mesh_size,
            row.vertex_count,
            row.gp_selected_total_seconds,
            row.feec_selected_total_seconds,
            row.speedup
        )?;
    }
    Ok(())
}

fn write_circulation_rmse_plot(
    result: &AnnulusHFormulationResult,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    let rows = result
        .summary_rows
        .iter()
        .filter(|row| row.regime == AnnulusRegime::D)
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Ok(());
    }
    let y_max = rows
        .iter()
        .map(|row| row.circ_rmse)
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max)
        .max(1e-6);
    let root = BitMapBackend::new(path, (1000, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Annulus Held-out Circulation RMSE (Regime D)",
            ("sans-serif", 24),
        )
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(70)
        .build_cartesian_2d(0..rows.len(), 0.0..(1.15 * y_max))?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(rows.len())
        .x_label_formatter(&|index| {
            rows.get(*index)
                .map(|row| row.model.as_str().replace('_', "\n"))
                .unwrap_or_default()
        })
        .y_desc("RMSE")
        .draw()?;
    chart.draw_series(rows.iter().enumerate().map(|(index, row)| {
        Rectangle::new(
            [(index, 0.0), (index + 1, row.circ_rmse)],
            BLUE.mix(0.65).filled(),
        )
    }))?;
    root.present()?;
    Ok(())
}

fn write_coverage_plot(
    result: &AnnulusHFormulationResult,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    let rows = result
        .summary_rows
        .iter()
        .filter(|row| row.regime == AnnulusRegime::D)
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Ok(());
    }
    let root = BitMapBackend::new(path, (1000, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Annulus Circulation 95% Coverage (Regime D)",
            ("sans-serif", 24),
        )
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(70)
        .build_cartesian_2d(0..rows.len(), 0.0..1.05)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(rows.len())
        .x_label_formatter(&|index| {
            rows.get(*index)
                .map(|row| row.model.as_str().replace('_', "\n"))
                .unwrap_or_default()
        })
        .y_desc("coverage")
        .draw()?;
    chart.draw_series(rows.iter().enumerate().map(|(index, row)| {
        Rectangle::new(
            [(index, 0.0), (index + 1, row.coverage_95)],
            GREEN.mix(0.65).filled(),
        )
    }))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0, 0.95), (rows.len(), 0.95)],
        RED,
    )))?;
    root.present()?;
    Ok(())
}

fn write_vtk_outputs(
    result: &AnnulusHFormulationResult,
    config: &AnnulusHFormulationConfig,
) -> Result<(), Box<dyn Error>> {
    if !config.mesh_path.is_file() {
        ensure_annulus_mesh(config)?;
    }
    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let truth = Cochain::new(1, result.truth_h.clone());
    let psi = Cochain::new(1, result.psi.clone());
    visual_output::write_1cochain_fields(
        config.output_dir.join("annulus_truth_fields.vtu"),
        &coords,
        &topology,
        &[("truth_h", &truth), ("psi", &psi)],
    )?;
    for field in &result.posterior_fields {
        let posterior = Cochain::new(1, field.posterior_mean.clone());
        let error = Cochain::new(1, &field.posterior_mean - &result.truth_h);
        visual_output::write_1cochain_fields(
            config.output_dir.join(format!(
                "posterior_{}_{}.vtu",
                field.regime.as_str(),
                field.model.as_str()
            )),
            &coords,
            &topology,
            &[
                ("truth_h", &truth),
                ("posterior_mean", &posterior),
                ("posterior_error", &error),
            ],
        )?;
    }
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod root_api_tests {
    use super::*;
    use common::linalg::nalgebra::CooMatrix as FeecCoo;

    fn identity(dimension: usize) -> FeecCsr {
        let mut matrix = FeecCoo::new(dimension, dimension);
        for index in 0..dimension {
            matrix.push(index, index, 1.0);
        }
        FeecCsr::from(&matrix)
    }

    #[test]
    fn annulus_conditioning_uses_affine_root_model_and_pushforward_variance() {
        let model = AnnulusLinearModel {
            model_kind: AnnulusModelKind::FeecGmrf,
            prior_precision: identity(2),
            latent_to_h: identity(2),
            h_offset: FeecVector::from_vec(vec![1.0, -1.0]),
            q_column: Some(0),
            selected_kappa: 1.0,
            selected_tau0: 1.0,
            selected_tau1: 1.0,
        };
        let training = AnnulusObservation {
            kind: AnnulusObservationKind::TrainLine,
            label: "training".to_string(),
            entries: vec![(0, 1.0)],
            truth_value: 2.0,
            observed_value: 2.0,
            noise_variance: 1.0,
        };
        let mut conditioned =
            condition_model(&model, &[training], 2).expect("root conditioning should succeed");
        assert!((conditioned.h_mean[0] - 1.5).abs() < 1e-12);
        assert!((conditioned.h_mean[1] + 1.0).abs() < 1e-12);
        assert!((conditioned.q_mean - 0.5).abs() < 1e-12);
        assert!((conditioned.q_std - 0.5_f64.sqrt()).abs() < 1e-12);

        let heldout = AnnulusObservation {
            kind: AnnulusObservationKind::HeldoutLine,
            label: "heldout".to_string(),
            entries: vec![(0, 1.0)],
            truth_value: 1.5,
            observed_value: 1.5,
            noise_variance: 1.0,
        };
        let predictions = predict_observables(
            0,
            AnnulusRegime::A,
            model.model_kind,
            &mut conditioned,
            &[heldout],
            2,
        )
        .expect("root pushforward should succeed");
        assert_eq!(predictions.len(), 1);
        assert!((predictions[0].mean - 1.5).abs() < 1e-12);
        assert!((predictions[0].latent_variance - 0.5).abs() < 1e-12);
    }
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use crate::annulus_baselines::build_spectrum_matched_potential_precision_full;
    use feg_infer::prior::matern::MaternAlpha;
    use feg_infer::sparse::feec_csr_to_gmrf;

    use super::*;

    fn gmsh_available() -> bool {
        Command::new("gmsh").arg("-version").output().is_ok()
    }

    fn test_config(name: &str) -> AnnulusHFormulationConfig {
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/annulus_h_formulation_tests")
            .join(name);
        AnnulusHFormulationConfig {
            output_dir: output_dir.clone(),
            mesh_path: output_dir.join("annulus_test.msh"),
            geo_path: output_dir.join("annulus_test.geo"),
            force_mesh: true,
            mesh_size: 0.42,
            line_tangential_count: 4,
            line_radial_count: 3,
            line_random_count: 3,
            residual_count: 12,
            validation_line_count: 6,
            validation_residual_count: 8,
            heldout_line_count: 8,
            heldout_residual_count: 8,
            heldout_loop_count: 4,
            noise_trial_count: 1,
            sample_observation_noise: false,
            potential_tau0_grid: vec![1e-3],
            potential_tau1_grid: vec![1.0],
            component_kappa_grid: vec![2.0],
            component_tau_grid: vec![1.0],
            scalar_potential_kappa_grid: vec![2.0],
            scalar_potential_tau_grid: vec![1.0],
            ..AnnulusHFormulationConfig::default()
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn annulus_topology_harmonic_and_truth_are_consistent() {
        if !gmsh_available() {
            eprintln!("skipping annulus topology test because gmsh is unavailable");
            return;
        }
        let _lock = crate::test_util::lock_feec_harmonic_tests();
        let config = test_config("topology");
        let workspace = build_workspace(&config).expect("workspace should build");
        assert_eq!(workspace.topology_summary.harmonic_1_dimension, 1);
        assert!((workspace.topology_summary.reference_period_psi - 1.0).abs() < 1e-8);
        assert!((workspace.topology_summary.reference_period_truth - 1.0).abs() < 1e-8);
        assert!(workspace.topology_summary.truth_closure_max_abs < 1e-8);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn annulus_baselines_factor_and_exact_models_have_zero_closed_period() {
        if !gmsh_available() {
            eprintln!("skipping annulus baseline test because gmsh is unavailable");
            return;
        }
        let _lock = crate::test_util::lock_feec_harmonic_tests();
        let config = test_config("baselines");
        let workspace = build_workspace(&config).expect("workspace should build");
        let models = AnnulusModelKind::benchmark_models()
            .into_iter()
            .flat_map(|kind| {
                build_model_candidates(kind, &config, &workspace)
                    .unwrap()
                    .into_iter()
                    .map(|candidate| candidate.model)
            })
            .collect::<Vec<_>>();
        assert_eq!(models.len(), AnnulusModelKind::benchmark_models().len());
        for model in &models {
            feec_csr_to_gmrf(&model.prior_precision)
                .cholesky_sqrt_lower()
                .expect("model precision should factor");
            assert_eq!(model.prior_precision.nrows(), model.latent_to_h.ncols());
            assert_eq!(workspace.topology.nsimplices(1), model.latent_to_h.nrows());
        }
        let loop_obs = AnnulusObservation {
            kind: AnnulusObservationKind::HeldoutCirculation,
            label: "reference".to_string(),
            entries: workspace.reference_loop.entries.clone(),
            truth_value: 1.0,
            observed_value: 1.0,
            noise_variance: 1e-4,
        };
        let loop_matrix = observation_matrix(&[loop_obs], workspace.topology.nsimplices(1), false)
            .expect("loop observation matrix should build");
        let scalar_correction = models
            .iter()
            .find(|model| model.model_kind == AnnulusModelKind::ScalarPotentialGpCorrection)
            .expect("scalar-potential correction model should exist");
        let scalar_loop_action = &loop_matrix * &scalar_correction.latent_to_h;
        assert!(
            scalar_loop_action
                .triplet_iter()
                .all(|(_, _, value)| value.abs() < 1e-10),
            "scalar-potential correction should leave the closed period to the offset"
        );
        assert!(
            (row_dot(
                &workspace.reference_loop.entries,
                &scalar_correction.h_offset
            ) - 1.0)
                .abs()
                < 1e-8,
            "scalar-potential correction offset should carry the reference period"
        );
        assert!(scalar_correction.q_column.is_none());

        for kind in [
            AnnulusModelKind::FeecSplitNoSpectralCorrection,
            AnnulusModelKind::FeecGmrf,
        ] {
            let model = models
                .iter()
                .find(|model| model.model_kind == kind)
                .expect("FEEC harmonic model should exist");
            let loop_action = &loop_matrix * &model.latent_to_h;
            let q_column = model.q_column.expect("FEEC model should expose q column");
            assert!(
                (row_dot(&workspace.reference_loop.entries, &model.h_offset)).abs() < 1e-10,
                "{} should not use a deterministic circulation offset",
                kind.as_str()
            );
            for (row, col, value) in loop_action.triplet_iter() {
                assert_eq!(row, 0);
                if col == q_column {
                    assert!(
                        (*value - 1.0).abs() < 1e-8,
                        "{} harmonic column should have unit reference period, got {}",
                        kind.as_str(),
                        value
                    );
                } else {
                    assert!(
                        value.abs() < 1e-10,
                        "{} exact branch column {col} should have zero closed period",
                        kind.as_str()
                    );
                }
            }
        }
        let component_correction = models
            .iter()
            .find(|model| model.model_kind == AnnulusModelKind::ComponentwiseGpCorrection)
            .expect("componentwise correction should exist");
        assert!(
            (row_dot(
                &workspace.reference_loop.entries,
                &component_correction.h_offset
            ) - 1.0)
                .abs()
                < 1e-8
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn annulus_feec_priors_preserve_sparse_precision_structure() {
        if !gmsh_available() {
            eprintln!("skipping annulus FEEC sparsity test because gmsh is unavailable");
            return;
        }
        let _lock = crate::test_util::lock_feec_harmonic_tests();
        let mut config = test_config("feec_sparsity");
        config.mesh_size = 0.24;
        let workspace = build_workspace(&config).expect("workspace should build");
        let ordinary = build_feec_split_no_spectral_correction_model(
            &workspace.topology,
            &workspace.metric,
            &workspace.mass_1form,
            &workspace.psi,
            AnnulusPotentialPriorConfig {
                tau0: config.tau0,
                tau1: config.tau1,
                sigma_q: config.harmonic_prior_std,
            },
        )
        .expect("ordinary FEEC prior should build");
        assert_eq!(
            ordinary.prior_precision.nrows(),
            workspace.topology_summary.vertex_count + 1,
            "ordinary FEEC should use full SPD potential coordinates plus q"
        );
        feec_csr_to_gmrf(&ordinary.prior_precision)
            .cholesky_sqrt_lower()
            .expect("ordinary FEEC precision should factor without a gauge");

        let kappa = (config.tau0 / config.tau1).sqrt();
        let tau = config.tau1.sqrt();
        let unanchored_spectral = build_spectrum_matched_potential_precision_full(
            &workspace.topology,
            &workspace.metric,
            MaternAlpha::Two,
            kappa,
            tau,
        )
        .expect("unanchored spectrally corrected precision should build");
        assert_eq!(
            unanchored_spectral.nrows(),
            workspace.topology_summary.vertex_count,
            "unanchored spectrally corrected precision should live on full potential coordinates"
        );

        let spectral = build_feec_gmrf_model(
            &workspace.topology,
            &workspace.metric,
            &workspace.mass_1form,
            &workspace.psi,
            AnnulusPotentialPriorConfig {
                tau0: config.tau0,
                tau1: config.tau1,
                sigma_q: config.harmonic_prior_std,
            },
        )
        .expect("spectrally corrected FEEC prior should build");
        assert_eq!(
            spectral.prior_precision.nrows(),
            workspace.topology_summary.vertex_count,
            "spectrally corrected FEEC should drop one sparse anchor coordinate and append q"
        );
        feec_csr_to_gmrf(&spectral.prior_precision)
            .cholesky_sqrt_lower()
            .expect("sparse-anchored spectrally corrected precision should factor");

        for model in [&ordinary, &spectral] {
            let density = square_density(
                model.prior_precision.nrows(),
                sparse_nnz(&model.prior_precision),
            );
            assert!(
                density < 0.15,
                "{} density {density:.3e} should stay sparse on the test mesh",
                model.model_kind.as_str()
            );
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn annulus_experiment_smoke_reports_finite_metrics_and_circulation_recovery() {
        if !gmsh_available() {
            eprintln!("skipping annulus experiment smoke test because gmsh is unavailable");
            return;
        }
        let _lock = crate::test_util::lock_feec_harmonic_tests();
        let config = test_config("smoke");
        let result = run_annulus_h_formulation(&config).expect("annulus experiment should run");
        assert!(!result.summary_rows.is_empty());
        assert!(result
            .summary_rows
            .iter()
            .all(|row| row.rmse_line.is_finite() && row.circ_rmse.is_finite()));
        assert!(result.trial_metrics.iter().all(|row| {
            row.selected_build_seconds.is_finite()
                && row.selection_seconds.is_finite()
                && row.conditioning_seconds.is_finite()
                && row.prediction_seconds.is_finite()
                && row.selected_total_seconds.is_finite()
                && row.pipeline_total_seconds.is_finite()
                && row.latent_dimension > 0
                && row.prior_precision_nnz > 0
                && row.posterior_precision_nnz > 0
                && row.posterior_factor_nnz > 0
        }));
        let trial_csv = config.output_dir.join("trial_metrics_schema.csv");
        write_trial_metrics_csv(&result, &trial_csv).expect("trial metrics CSV should write");
        let header = fs::read_to_string(&trial_csv)
            .expect("trial metrics CSV should be readable")
            .lines()
            .next()
            .expect("trial metrics CSV should have a header")
            .to_string();
        for required in [
            "selected_build_seconds",
            "selection_seconds",
            "selected_total_seconds",
            "pipeline_total_seconds",
            "prior_precision_density",
            "posterior_precision_density",
            "posterior_factor_nnz",
            "posterior_fill_ratio",
        ] {
            assert!(
                header.contains(required),
                "trial metrics schema should contain {required}"
            );
        }
        let d_models = result
            .summary_rows
            .iter()
            .filter(|row| row.regime == AnnulusRegime::D)
            .map(|row| row.model)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            d_models,
            AnnulusModelKind::benchmark_models()
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        for model in [
            AnnulusModelKind::FeecGmrf,
            AnnulusModelKind::FeecSplitNoSpectralCorrection,
        ] {
            let row = result
                .trial_metrics
                .iter()
                .find(|row| row.regime == AnnulusRegime::D && row.model == model)
                .expect("regime D topology-aware metric should exist");
            assert!(
                row.circ_rmse < 1e-4,
                "{} circ_rmse={} should recover circulation within the observation noise scale",
                model.as_str(),
                row.circ_rmse
            );
            assert!(
                (row.q_mean - 1.0).abs() < 1e-4 && row.q_std.is_finite() && row.q_std > 0.0,
                "{} q posterior should identify the unit circulation, got mean={} std={}",
                model.as_str(),
                row.q_mean,
                row.q_std
            );
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn annulus_mesh_invariance_reports_stable_physical_qoi_variances() {
        if !gmsh_available() {
            eprintln!("skipping annulus mesh invariance test because gmsh is unavailable");
            return;
        }
        let _lock = crate::test_util::lock_feec_harmonic_tests();
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/annulus_h_formulation_tests")
            .join("mesh_invariance");
        let config = AnnulusMeshInvarianceConfig {
            profile: AnnulusMeshInvarianceProfile::Quick,
            mesh_sizes: AnnulusMeshInvarianceProfile::Quick.default_mesh_sizes(),
            output_dir,
            residual_count: 10,
            heldout_loop_count: 4,
            sample_observation_noise: false,
            model_kinds: vec![
                AnnulusModelKind::FeecGmrf,
                AnnulusModelKind::FeecSplitNoSpectralCorrection,
            ],
            convergence_tail_count: 3,
            base: test_config("mesh_invariance_base"),
        };
        let result =
            run_annulus_h_mesh_invariance(&config).expect("mesh invariance run should complete");
        assert_eq!(result.mesh_metadata.len(), 6);
        assert!(result
            .mesh_metadata
            .iter()
            .all(|row| row.harmonic_1_dimension == 1
                && (row.reference_period_truth - 1.0).abs() < 1e-8
                && (row.reference_period_psi - 1.0).abs() < 1e-8));
        assert!(result
            .qoi_rows
            .iter()
            .all(|row| row.latent_variance.is_finite() && row.latent_variance >= 0.0));
        let invalid_fit_rows = result
            .fit_rows
            .iter()
            .filter(|row| {
                let deterministic_topological_ratio = matches!(
                    row.qoi_family.as_str(),
                    "q_period_variance" | "circulation_mean_variance"
                ) && row.model_ratio_to_full.is_nan();
                !(row.slope_log_variance_vs_log_h.is_finite()
                    && row.tail_relative_spread.is_finite()
                    && row.tail_count == 3
                    && (row.model_ratio_to_full.is_finite() || deterministic_topological_ratio))
            })
            .map(|row| {
                format!(
                    "{} {} {} slope={} spread={} ratio={}",
                    row.model.as_str(),
                    row.regime.as_str(),
                    row.qoi_family,
                    row.slope_log_variance_vs_log_h,
                    row.tail_relative_spread,
                    row.model_ratio_to_full
                )
            })
            .collect::<Vec<_>>();
        assert!(
            invalid_fit_rows.is_empty(),
            "invalid mesh fit rows: {}",
            invalid_fit_rows.join("; ")
        );
        assert!(!result.contrast_rows.is_empty());
        assert!(result.contrast_rows.iter().all(|row| {
            let deterministic_topological_ratio = matches!(
                row.qoi_family.as_str(),
                "q_period_variance" | "circulation_mean_variance"
            ) && row.ratio_no_correction_to_full.is_nan();
            (row.ratio_no_correction_to_full.is_finite() && row.ratio_no_correction_to_full >= 0.0)
                || deterministic_topological_ratio
        }));
        assert!(result
            .qoi_rows
            .iter()
            .filter(|row| {
                row.regime == AnnulusRegime::D && row.qoi_kind == "heldout_circulation"
            })
            .all(|row| (row.truth - 1.0).abs() < 1e-8));
        for (regime, qoi_kind) in [
            (AnnulusRegime::B, "heldout_line_short"),
            (AnnulusRegime::B, "heldout_line_physical_long"),
            (AnnulusRegime::B, "dense_line"),
            (AnnulusRegime::B, "field_x"),
            (AnnulusRegime::B, "field_y"),
            (AnnulusRegime::D, "heldout_line_short"),
            (AnnulusRegime::D, "heldout_line_physical_long"),
            (AnnulusRegime::D, "dense_line"),
            (AnnulusRegime::D, "field_x"),
            (AnnulusRegime::D, "field_y"),
        ] {
            for model in &config.model_kinds {
                assert!(
                    result.qoi_rows.iter().any(|row| {
                        row.model == *model && row.regime == regime && row.qoi_kind == qoi_kind
                    }),
                    "{} {} should include {qoi_kind} QoIs",
                    model.as_str(),
                    regime.as_str()
                );
            }
        }
        for row in &result.summary_rows {
            assert!(
                row.dense_line_away_count > 0 && row.field_magnitude_away_count > 0,
                "{} h={} should retain away dense line and field probes",
                row.regime.as_str(),
                row.mesh_size
            );
        }
        for (model, regime, mesh_size) in result
            .summary_rows
            .iter()
            .map(|row| (row.model, row.regime, row.mesh_size))
        {
            let field_x_count = result
                .qoi_rows
                .iter()
                .filter(|row| {
                    row.model == model
                        && row.regime == regime
                        && row.mesh_size == mesh_size
                        && row.qoi_kind == "field_x"
                })
                .count();
            let field_y_count = result
                .qoi_rows
                .iter()
                .filter(|row| {
                    row.model == model
                        && row.regime == regime
                        && row.mesh_size == mesh_size
                        && row.qoi_kind == "field_y"
                })
                .count();
            assert!(
                field_x_count >= 96 && field_y_count >= 96,
                "{} {} h={mesh_size} should locate at least 80% of dense field points, got x={field_x_count} y={field_y_count}",
                model.as_str(),
                regime.as_str()
            );
        }
        for (regime, family) in [
            (AnnulusRegime::B, "q_period_variance"),
            (AnnulusRegime::B, "short_line_median_variance"),
            (AnnulusRegime::B, "dense_line_away_median_variance"),
            (AnnulusRegime::B, "field_magnitude_away_median_variance"),
            (AnnulusRegime::D, "q_period_variance"),
            (AnnulusRegime::D, "short_line_median_variance"),
            (AnnulusRegime::D, "long_line_median_variance"),
            (AnnulusRegime::D, "dense_line_away_median_variance"),
            (AnnulusRegime::D, "field_magnitude_away_median_variance"),
        ] {
            for model in &config.model_kinds {
                assert!(
                    result.fit_rows.iter().any(|row| {
                        row.model == *model && row.regime == regime && row.qoi_family == family
                    }),
                    "{} {} fit rows should include {family}",
                    model.as_str(),
                    regime.as_str()
                );
            }
        }

        for model in &config.model_kinds {
            for family in ["q_period_variance", "circulation_mean_variance"] {
                let fit = result
                    .fit_rows
                    .iter()
                    .find(|row| {
                        row.model == *model
                            && row.regime == AnnulusRegime::D
                            && row.qoi_family == family
                    })
                    .expect("D-regime topological fit row should exist");
                assert_eq!(
                    fit.status,
                    "pass",
                    "{} {family} should pass quick mesh stability: tail_spread={} slope={}",
                    model.as_str(),
                    fit.tail_relative_spread,
                    fit.slope_log_variance_vs_log_h
                );
            }
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn annulus_mesh_invariance_single_full_model_emits_dense_field_qois() {
        if !gmsh_available() {
            eprintln!(
                "skipping annulus full-model mesh invariance test because gmsh is unavailable"
            );
            return;
        }
        let _lock = crate::test_util::lock_feec_harmonic_tests();
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/annulus_h_formulation_tests")
            .join("mesh_invariance_spectrum_matched");
        let config = AnnulusMeshInvarianceConfig {
            profile: AnnulusMeshInvarianceProfile::Quick,
            mesh_sizes: AnnulusMeshInvarianceProfile::Quick.default_mesh_sizes(),
            output_dir,
            residual_count: 10,
            heldout_loop_count: 4,
            sample_observation_noise: false,
            model_kinds: vec![AnnulusModelKind::FeecGmrf],
            convergence_tail_count: 3,
            base: test_config("mesh_invariance_spectrum_matched_base"),
        };
        let result = run_annulus_h_mesh_invariance(&config)
            .expect("full-model mesh invariance run should complete");
        assert_eq!(result.mesh_metadata.len(), 3);
        assert!(result
            .mesh_metadata
            .iter()
            .all(|row| row.model == AnnulusModelKind::FeecGmrf
                && row.harmonic_1_dimension == 1
                && (row.reference_period_truth - 1.0).abs() < 1e-8));
        assert!(result
            .qoi_rows
            .iter()
            .all(|row| row.latent_variance.is_finite() && row.latent_variance >= 0.0));
        assert!(result
            .summary_rows
            .iter()
            .all(|row| row.dense_line_away_count > 0
                && row.field_magnitude_away_count > 0
                && row.field_magnitude_away_median_variance.is_finite()
                && row.field_magnitude_away_median_variance >= 0.0));
        assert!(result.fit_rows.iter().any(|row| {
            row.model == AnnulusModelKind::FeecGmrf
                && row.regime == AnnulusRegime::B
                && row.qoi_family == "field_magnitude_away_median_variance"
        }));
        assert!(result.fit_rows.iter().any(|row| {
            row.model == AnnulusModelKind::FeecGmrf
                && row.regime == AnnulusRegime::D
                && row.qoi_family == "dense_line_away_median_variance"
        }));
    }
}
