use crate::toroidal_material::ToroidalReluctivityLaw;
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::{cochain::Cochain, ManifoldComplexExt};
use exterior::field::DiffFormClosure;
use feg_core::{
    BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, GaussianPriorSpec,
    LinearGaussianMeasurementSpec, LinearUncertainInputSpec, NonlinearResidualModel,
    RepresentationPreference, SparseTriplet, SparseTripletMatrix,
};
use feg_infer::linear_pde::{
    build_linear_pde_joint_posterior_with_config, solve_linear_pde_uq_with_config,
    solve_scaled_linear_pde_uq_with_config, LinearPdeCoordinateScaling,
    LinearPdeDerivedQuantitySpec, LinearPdeLatentCoordinateScaling, LinearPdePrecisionPolicy,
    LinearPdeUqProblem, LinearPdeUqResult, LinearPdeUqSolverConfig, LinearPdeVarianceConfig,
    LinearPdeVarianceMode,
};
use feg_infer::nonlinear::{
    solve_nonlinear_laplace, trace_normalized_precision, GaussNewtonConfig, GaussNewtonLinearSolve,
    GaussNewtonLinearSolveMode, GaussianNoiseModel, NonlinearLaplaceProblem,
    NonlinearLaplaceResult, NonlinearResidualTerm, SelectedResidualModel,
};
use feg_infer::physical::{
    build_full_magnetic_flux_density_operator_3d, build_reduced_magnetic_flux_density_operator_3d,
};
use feg_infer::prior::exact_potential::{
    build_exact_two_form_potential_prior, ExactTwoFormPotentialPriorConfig,
};
use feg_infer::prior::matern::one_form::{
    build_reduced_linear_proxy_matern_alpha2_prior, default_reduced_linear_proxy_matern_kappa,
    MaternMassInverse as Matern1FormMassInverse, ReducedLinearProxyMaternAlpha2Config,
};
use feg_infer::prior::trace_normalization::trace_normalization_from_target_trace;
use feg_infer::prior::zero_mean_diagonal_prior;
use feg_infer::sparse::core_triplet_to_gmrf as sparse_from_core;
use feg_infer::sparse::{
    core_triplet_to_feec_csr, feec_csr_to_core_triplet, feec_csr_to_gmrf, lift_vector_with_layout,
    restrict_triplet_columns_and_fold_fixed, select_square_triplet_rows_cols,
    sparse_row_operator_to_triplet, triplet_to_sparse_row_operator,
};
use feg_infer::{adapters::FeecResidualAdapter, boundary::adapt_boundary_spec};
use formoniq::{
    assemble::{self, assemble_galvec},
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::{
        hodge_laplace::{self, MixedGalmats},
        nonlinear_magnetostatic::{
            build_reduced_vector_potential_magnetostatic_3d, LocalMagneticStrongProbe3d,
            NonlinearMagnetostaticAssemblyConfig, ReducedVectorPotentialMagnetostatic3d,
        },
        reduced_linear::{
            build_reduced_weighted_hodge_laplace_1form_system, ReducedLinearPdeAssembly,
        },
        residual::ResidualModel as FeecResidualModel,
    },
    reduction::EssentialBoundarySpec,
};
use gmrf_core::{
    apply_gaussian_observations, exact_transformed_variance_weighted_trace, SparseRowOperator,
    Vector as GmrfVector,
};
use manifold::{
    geometry::coord::{
        mesh::MeshCoords,
        simplex::{SimplexCoords, SimplexHandleExt},
        CoordRef,
    },
    topology::{complex::Complex, handle::SimplexIdx},
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::{
    collections::{BTreeMap, HashSet},
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

pub const SOURCE_MODE_COUNT: usize = 4;
pub const DEFAULT_MESH_PATH: &str = "meshes/toroidal_inductor.msh";
pub const DEFAULT_OUTPUT_DIR: &str = "out/examples/nonlinear_toroidal_inductor";
pub const TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_CALIBRATION_LABEL: &str = "physical_b_nominal_rms_x10";
pub const TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_RMS_MULTIPLIER: f64 = 10.0;
pub const TOROIDAL_EXACT_B_CALIBRATION_NOMINAL_B_RMS: f64 = 2.5839736950567727e-8;
pub const TOROIDAL_EXACT_B_CALIBRATION_TARGET_PRIOR_B_RMS: f64 = 2.5839736950567726e-7;
pub const TOROIDAL_EXACT_B_CALIBRATED_PRIOR_TAU: f64 = 1.0698378444558117e6;
pub const TOROIDAL_EXACT_B_CANONICAL_PRIOR_RMS_MULTIPLIER: f64 = 100.0;
pub const TOROIDAL_EXACT_B_CANONICAL_PRIOR_TAU: f64 = 1.0698378444558117e5;
pub const TOROIDAL_EXACT_B_DETERMINISTIC_REFERENCE_DIAGONAL_SHIFT: f64 = 1.0e-4;
pub const TOROIDAL_EXACT_B_SYNTHETIC_OBSERVATION_NOISE_SEED: u64 = 0x5EED_F10C_2026;
pub const TOROIDAL_EXACT_B_SYNTHETIC_HELDOUT_NOISE_SEED: u64 = 0xA11D_0A75_2026;
const SOURCE_DISCREPANCY_INPUT: &str = "source_discrepancy";
const EPS: f64 = 1e-12;

pub fn toroidal_exact_b_prior_tau_for_physical_b_multiplier(
    target_multiplier: f64,
) -> Result<f64, String> {
    if !target_multiplier.is_finite() || target_multiplier <= 0.0 {
        return Err(format!(
            "exact-B physical-B prior multiplier must be finite and positive, got {target_multiplier}"
        ));
    }
    Ok(
        TOROIDAL_EXACT_B_CALIBRATED_PRIOR_TAU * TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_RMS_MULTIPLIER
            / target_multiplier,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalPriorMode {
    WeakDiagonal,
    LinearProxyMaternAlpha2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalResidualWeighting {
    Euclidean,
    MassInverseTraceNormalized,
}

impl ToroidalResidualWeighting {
    pub fn label(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::MassInverseTraceNormalized => "mass_inverse_trace_normalized",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalPdeObservationMode {
    WeakGalerkinRows,
    LocalMagneticStrongCells,
}

impl ToroidalPdeObservationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::WeakGalerkinRows => "weak_galerkin_rows",
            Self::LocalMagneticStrongCells => "local_strong_cells",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToroidalResidualSelection {
    Full,
    Stride { step: usize },
    ShuffledStride { step: usize, seed: u64 },
    ShuffledCount { count: usize, seed: u64 },
}

impl ToroidalResidualSelection {
    pub fn label(&self) -> String {
        match self {
            Self::Full => "full".to_string(),
            Self::Stride { step } => format!("stride_{step}"),
            Self::ShuffledStride { step, seed } => format!("shuffled_stride_{step}_seed_{seed}"),
            Self::ShuffledCount { count, seed } => format!("shuffled_count_{count}_seed_{seed}"),
        }
    }
}

impl ToroidalPriorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::WeakDiagonal => "weak_diagonal",
            Self::LinearProxyMaternAlpha2 => "linear_proxy_matern_alpha2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToroidalInductorGeometry {
    pub major_radius: f64,
    pub core_minor_radius: f64,
    pub coil_minor_radius: f64,
    pub box_half_length: f64,
    pub target_air_cell_size: f64,
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

#[derive(Debug, Clone)]
pub struct NonlinearToroidalConfig {
    pub mesh_path: std::path::PathBuf,
    pub output_dir: Option<std::path::PathBuf>,
    pub geometry: ToroidalInductorGeometry,
    pub nu_air: f64,
    pub nu_core0: f64,
    pub beta_core: f64,
    pub pde_variance: f64,
    pub pde_observation_mode: ToroidalPdeObservationMode,
    pub residual_weighting: ToroidalResidualWeighting,
    pub residual_selection: ToroidalResidualSelection,
    pub prior_precision: f64,
    pub prior_mode: ToroidalPriorMode,
    pub linear_proxy_kappa: Option<f64>,
    pub linear_proxy_tau: f64,
    pub linear_proxy_allow_kappa_fallback: bool,
    pub include_nonlinear_residual: bool,
    pub linear_measurements: Vec<LinearGaussianMeasurementSpec>,
    pub extra_derived_quantities: Vec<LinearPdeDerivedQuantitySpec>,
    pub linear_solve: GaussNewtonLinearSolve,
    pub variance: LinearPdeVarianceConfig,
    pub max_iterations: usize,
    pub write_outputs: bool,
    pub include_cell_b_variance: bool,
    pub compute_harmonic_diagnostics: bool,
}

impl Default for NonlinearToroidalConfig {
    fn default() -> Self {
        let mu0 = 4e-7 * PI;
        let nu_air = 1.0 / mu0;
        Self {
            mesh_path: DEFAULT_MESH_PATH.into(),
            output_dir: Some(DEFAULT_OUTPUT_DIR.into()),
            geometry: ToroidalInductorGeometry::default(),
            nu_air,
            nu_core0: nu_air / 200.0,
            beta_core: 0.0,
            pde_variance: 1.0,
            pde_observation_mode: ToroidalPdeObservationMode::WeakGalerkinRows,
            residual_weighting: ToroidalResidualWeighting::Euclidean,
            residual_selection: ToroidalResidualSelection::Full,
            prior_precision: 1e-12,
            prior_mode: ToroidalPriorMode::WeakDiagonal,
            linear_proxy_kappa: None,
            linear_proxy_tau: 1e-6,
            linear_proxy_allow_kappa_fallback: true,
            include_nonlinear_residual: true,
            linear_measurements: Vec::new(),
            extra_derived_quantities: Vec::new(),
            linear_solve: GaussNewtonLinearSolve::DirectCholesky,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::Exact,
                ..LinearPdeVarianceConfig::default()
            },
            max_iterations: 12,
            write_outputs: true,
            include_cell_b_variance: true,
            compute_harmonic_diagnostics: true,
        }
    }
}

pub struct NonlinearToroidalReport {
    pub active_dofs: usize,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub boundary_edge_dofs: usize,
    pub gauge_edge_dofs: usize,
    pub prior_mode: ToroidalPriorMode,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_kappa_fallback_used: bool,
    pub residual_variance: f64,
    pub pde_observation_mode: ToroidalPdeObservationMode,
    pub residual_weighting: ToroidalResidualWeighting,
    pub residual_precision_normalization: Option<f64>,
    pub residual_rows_used: usize,
    pub residual_rows_total: usize,
    pub nonlinear_residual_likelihood: bool,
    pub observation_initial_residual_norm: Option<f64>,
    pub observation_final_residual_norm: Option<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub map_relative_distance_from_linear_mean: f64,
    pub direct_linear_relative_error: Option<f64>,
    pub mixed_reference_b_relative_error: Option<f64>,
    pub harmonic_coefficients: Vec<f64>,
    pub harmonic_coefficient_norm: f64,
    pub sensor_reports: Vec<ToroidalSensorReport>,
    pub posterior_precision_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub final_factorization_seconds: f64,
    pub linear_solve_mode: GaussNewtonLinearSolveMode,
    pub linear_solve_iteration_sum: usize,
    pub linear_solve_residual_max: f64,
    pub posterior_factorizes: bool,
    pub latent_variance_len: usize,
    pub b_variance_len: Option<usize>,
    pub b_variance_min: Option<f64>,
    pub b_variance_max: Option<f64>,
    pub result: NonlinearLaplaceResult,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalSensorReport {
    pub name: String,
    pub nonlinear_value: f64,
    pub beta_zero_value: Option<f64>,
    pub mixed_reference_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalFluxSensorSpec {
    pub name: String,
    pub center: [f64; 3],
    pub normal: [f64; 3],
    pub patch_radius: f64,
    pub normal_half_width: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalReducedFluxSensor {
    pub spec: ToroidalFluxSensorSpec,
    pub operator: SparseTripletMatrix,
    pub bias: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ToroidalResidualBudgetConfig {
    pub base: NonlinearToroidalConfig,
    pub prior_modes: Vec<ToroidalPriorMode>,
    pub residual_strides: Vec<usize>,
    pub shuffled: bool,
    pub seed: u64,
}

impl Default for ToroidalResidualBudgetConfig {
    fn default() -> Self {
        Self {
            base: NonlinearToroidalConfig {
                beta_core: 1e16,
                write_outputs: false,
                include_cell_b_variance: false,
                ..NonlinearToroidalConfig::default()
            },
            prior_modes: vec![
                ToroidalPriorMode::WeakDiagonal,
                ToroidalPriorMode::LinearProxyMaternAlpha2,
            ],
            residual_strides: vec![64, 1],
            shuffled: true,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalResidualBudgetRow {
    pub prior_mode: ToroidalPriorMode,
    pub selection_label: String,
    pub residual_rows_used: usize,
    pub residual_rows_total: usize,
    pub iterations: usize,
    pub final_residual_norm: f64,
    pub linear_solve_iteration_sum: usize,
    pub linear_solve_residual_max: f64,
    pub posterior_factor_nnz: usize,
    pub final_factorization_seconds: f64,
    pub map_relative_error_to_reference: f64,
    pub cell_b_relative_error_to_reference: f64,
    pub flux_sensor_rmse_to_reference: f64,
    pub posterior_factorizes: bool,
}

pub struct ToroidalResidualBudgetReport {
    pub reference: NonlinearToroidalReport,
    pub rows: Vec<ToroidalResidualBudgetRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalSensorRegressionVariant {
    PriorOnly,
    SensorsOnly,
    ResidualBudgetOnly,
    SensorsPlusResidualBudget,
}

impl ToroidalSensorRegressionVariant {
    pub fn label(self) -> &'static str {
        match self {
            Self::PriorOnly => "prior_only",
            Self::SensorsOnly => "sensors_only",
            Self::ResidualBudgetOnly => "residual_budget_only",
            Self::SensorsPlusResidualBudget => "sensors_plus_residual_budget",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToroidalSensorRegressionConfig {
    pub base: NonlinearToroidalConfig,
    pub prior_modes: Vec<ToroidalPriorMode>,
    pub variants: Vec<ToroidalSensorRegressionVariant>,
    pub azimuth_count: usize,
    pub train_counts: Vec<usize>,
    pub residual_stride: usize,
    pub sensor_variance: f64,
    pub synthetic_noise_std: f64,
    pub seed: u64,
}

impl Default for ToroidalSensorRegressionConfig {
    fn default() -> Self {
        Self {
            base: NonlinearToroidalConfig {
                beta_core: 1e16,
                write_outputs: false,
                include_cell_b_variance: false,
                ..NonlinearToroidalConfig::default()
            },
            prior_modes: vec![ToroidalPriorMode::LinearProxyMaternAlpha2],
            variants: vec![
                ToroidalSensorRegressionVariant::PriorOnly,
                ToroidalSensorRegressionVariant::SensorsPlusResidualBudget,
            ],
            azimuth_count: 2,
            train_counts: vec![3],
            residual_stride: 64,
            sensor_variance: 1e-20,
            synthetic_noise_std: 0.0,
            seed: 0x51a7,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalSensorRegressionRow {
    pub prior_mode: ToroidalPriorMode,
    pub variant: ToroidalSensorRegressionVariant,
    pub train_count: usize,
    pub holdout_count: usize,
    pub train_rmse: f64,
    pub holdout_rmse: f64,
    pub mean_abs_z: f64,
    pub coverage_2sigma: f64,
    pub final_residual_norm: f64,
    pub linear_solve_iteration_sum: usize,
    pub linear_solve_residual_max: f64,
    pub posterior_factor_nnz: usize,
    pub final_factorization_seconds: f64,
    pub posterior_factorizes: bool,
}

pub struct ToroidalSensorRegressionReport {
    pub reference: NonlinearToroidalReport,
    pub sensors: Vec<ToroidalFluxSensorSpec>,
    pub rows: Vec<ToroidalSensorRegressionRow>,
}

#[derive(Debug, Clone)]
pub struct ToroidalWeilandComparisonConfig {
    pub base: NonlinearToroidalConfig,
    pub prior_modes: Vec<ToroidalPriorMode>,
    pub residual_fractions: Vec<f64>,
    pub explicit_kappas: Vec<f64>,
    pub kappa_diameter_scales: Vec<f64>,
    pub include_default_kappa: bool,
    pub row_selection_repetitions: usize,
    pub seed: u64,
    pub sensor_azimuth_count: usize,
}

impl Default for ToroidalWeilandComparisonConfig {
    fn default() -> Self {
        Self {
            base: NonlinearToroidalConfig {
                beta_core: 1e16,
                write_outputs: false,
                include_cell_b_variance: false,
                compute_harmonic_diagnostics: false,
                linear_proxy_allow_kappa_fallback: false,
                pde_observation_mode: ToroidalPdeObservationMode::LocalMagneticStrongCells,
                ..NonlinearToroidalConfig::default()
            },
            prior_modes: vec![
                ToroidalPriorMode::WeakDiagonal,
                ToroidalPriorMode::LinearProxyMaternAlpha2,
            ],
            residual_fractions: vec![
                0.0,
                1.0 / 256.0,
                1.0 / 128.0,
                1.0 / 64.0,
                1.0 / 32.0,
                1.0 / 16.0,
                1.0 / 8.0,
                1.0 / 4.0,
                1.0 / 2.0,
                1.0,
            ],
            explicit_kappas: Vec::new(),
            kappa_diameter_scales: vec![1.0, 1e-1, 1e-2, 1e-3, 1e-4],
            include_default_kappa: true,
            row_selection_repetitions: 3,
            seed: 0x1A15_7A75,
            sensor_azimuth_count: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalWeilandReferenceRow {
    pub requested_kappa: f64,
    pub actual_kappa: f64,
    pub kappa_times_diameter: f64,
    pub success: bool,
    pub failure_reason: Option<String>,
    pub iterations: usize,
    pub final_residual_norm: f64,
    pub posterior_factor_nnz: usize,
    pub final_factorization_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalWeilandComparisonRow {
    pub reference_kappa: f64,
    pub prior_kappa: f64,
    pub kappa_times_diameter: f64,
    pub prior_mode: ToroidalPriorMode,
    pub nonlinear_residual_likelihood: bool,
    pub selection_label: String,
    pub seed: u64,
    pub residual_rows_requested: usize,
    pub residual_rows_used: usize,
    pub residual_rows_total: usize,
    pub residual_fraction: f64,
    pub success: bool,
    pub failure_reason: Option<String>,
    pub iterations: usize,
    pub damping_steps: usize,
    pub final_residual_norm: f64,
    pub linear_solve_iteration_sum: usize,
    pub linear_solve_residual_max: f64,
    pub posterior_factor_nnz: usize,
    pub final_factorization_seconds: f64,
    pub map_relative_error_to_reference: f64,
    pub cell_b_relative_error_to_reference: f64,
    pub flux_sensor_rmse_to_reference: f64,
    pub sensor_variance_min: f64,
    pub sensor_variance_max: f64,
    pub sensor_mean_abs_z: f64,
    pub sensor_coverage_2sigma: f64,
    pub posterior_factorizes: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalWeilandComparisonSummaryRow {
    pub reference_kappa: f64,
    pub prior_mode: ToroidalPriorMode,
    pub nonlinear_residual_likelihood: bool,
    pub residual_rows_requested: usize,
    pub residual_rows_total: usize,
    pub residual_fraction: f64,
    pub success_count: usize,
    pub failure_count: usize,
    pub final_residual_mean: f64,
    pub final_residual_std: f64,
    pub cell_b_relative_error_mean: f64,
    pub cell_b_relative_error_std: f64,
    pub flux_sensor_rmse_mean: f64,
    pub flux_sensor_rmse_std: f64,
    pub map_relative_error_mean: f64,
    pub map_relative_error_std: f64,
    pub sensor_mean_abs_z_mean: f64,
    pub sensor_coverage_2sigma_mean: f64,
}

pub struct ToroidalWeilandComparisonReport {
    pub mesh_path: PathBuf,
    pub bounding_box_diameter: f64,
    pub active_dofs: usize,
    pub residual_rows_total: usize,
    pub sensor_count: usize,
    pub references: Vec<ToroidalWeilandReferenceRow>,
    pub rows: Vec<ToroidalWeilandComparisonRow>,
    pub summaries: Vec<ToroidalWeilandComparisonSummaryRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalExactBReferenceMode {
    NominalDebug,
    PerturbedSource,
}

impl ToroidalExactBReferenceMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::NominalDebug => "nominal_debug",
            Self::PerturbedSource => "perturbed_source",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalExactBReferenceSolveMode {
    RegularizedMap,
    DeterministicPde,
}

impl ToroidalExactBReferenceSolveMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::RegularizedMap => "regularized_map",
            Self::DeterministicPde => "deterministic_pde",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalExactBObservationMode {
    CellMagneticComponents,
    SurfaceFluxes,
    SourceDesignedFluxes,
}

impl ToroidalExactBObservationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::CellMagneticComponents => "cell_magnetic_components",
            Self::SurfaceFluxes => "surface_fluxes",
            Self::SourceDesignedFluxes => "source_designed_fluxes",
        }
    }
}

impl std::str::FromStr for ToroidalExactBObservationMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cell"
            | "cells"
            | "cell-components"
            | "cell_components"
            | "cell-magnetic-components"
            | "cell_magnetic_components" => Ok(Self::CellMagneticComponents),
            "flux" | "surface-flux" | "surface_flux" | "surface-fluxes" | "surface_fluxes" => {
                Ok(Self::SurfaceFluxes)
            }
            "source-designed-flux"
            | "source_designed_flux"
            | "source-designed-fluxes"
            | "source_designed_fluxes"
            | "designed-flux"
            | "designed_flux" => Ok(Self::SourceDesignedFluxes),
            other => Err(format!(
                "unknown observation mode `{other}`; expected cell-components, surface-flux, or source-designed-flux"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalExactBNondimensionalizationMode {
    Off,
    PdeColumnNorm,
}

impl ToroidalExactBNondimensionalizationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::PdeColumnNorm => "pde_column_norm",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToroidalExactBBaseConfig {
    pub mesh_path: PathBuf,
    pub geometry: ToroidalInductorGeometry,
    pub nu_air: f64,
    pub nu_core0: f64,
    pub pde_variance: f64,
    pub residual_selection: ToroidalResidualSelection,
    pub prior_precision: f64,
    pub variance: LinearPdeVarianceConfig,
    pub precision_policy: LinearPdePrecisionPolicy,
}

impl Default for ToroidalExactBBaseConfig {
    fn default() -> Self {
        let mu0 = 4e-7 * PI;
        let nu_air = 1.0 / mu0;
        Self {
            mesh_path: DEFAULT_MESH_PATH.into(),
            geometry: ToroidalInductorGeometry::default(),
            nu_air,
            nu_core0: nu_air / 200.0,
            pde_variance: 1e-8,
            residual_selection: ToroidalResidualSelection::Full,
            prior_precision: 1e-8,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::Exact,
                ..LinearPdeVarianceConfig::default()
            },
            precision_policy: LinearPdePrecisionPolicy::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToroidalExactBRecoveryConfig {
    pub base: ToroidalExactBBaseConfig,
    pub reference_mode: ToroidalExactBReferenceMode,
    pub reference_solve_mode: ToroidalExactBReferenceSolveMode,
    pub reference_solver_diagonal_shift: f64,
    pub observation_mode: ToroidalExactBObservationMode,
    pub source_deltas: [f64; SOURCE_MODE_COUNT],
    pub source_prior_std: f64,
    pub reference_pde_variance: Option<f64>,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub potential_mass_inverse: Matern1FormMassInverse,
    pub observation_train_fraction: f64,
    pub observation_noise_std: f64,
    pub synthetic_observation_noise_seed: Option<u64>,
    pub synthetic_heldout_observation_noise_seed: Option<u64>,
    pub observation_seed: u64,
    pub heldout_count: usize,
    pub observation_index_override: Option<ToroidalExactBObservationIndexOverride>,
    pub surface_flux_azimuth_count: usize,
    pub nondimensionalization: ToroidalExactBNondimensionalizationMode,
    pub reference_observation_csv_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub write_outputs: bool,
}

impl Default for ToroidalExactBRecoveryConfig {
    fn default() -> Self {
        Self {
            base: ToroidalExactBBaseConfig::default(),
            reference_mode: ToroidalExactBReferenceMode::PerturbedSource,
            reference_solve_mode: ToroidalExactBReferenceSolveMode::RegularizedMap,
            reference_solver_diagonal_shift: 0.0,
            observation_mode: ToroidalExactBObservationMode::CellMagneticComponents,
            source_deltas: [0.0, 0.15, -0.10, 0.05],
            source_prior_std: 0.25,
            reference_pde_variance: None,
            prior_kappa: 1.0,
            prior_tau: 1e-6,
            potential_mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
            observation_train_fraction: 0.10,
            observation_noise_std: 1e-10,
            synthetic_observation_noise_seed: None,
            synthetic_heldout_observation_noise_seed: None,
            observation_seed: 0xDA7A_2026,
            heldout_count: 128,
            observation_index_override: None,
            surface_flux_azimuth_count: 4,
            nondimensionalization: ToroidalExactBNondimensionalizationMode::Off,
            reference_observation_csv_path: None,
            output_dir: Some(PathBuf::from("out/examples/toroidal_exact_b_recovery")),
            write_outputs: true,
        }
    }
}

pub fn toroidal_exact_b_canonical_source_designed_flux_config() -> ToroidalExactBRecoveryConfig {
    ToroidalExactBRecoveryConfig {
        reference_mode: ToroidalExactBReferenceMode::PerturbedSource,
        reference_solve_mode: ToroidalExactBReferenceSolveMode::DeterministicPde,
        reference_solver_diagonal_shift: TOROIDAL_EXACT_B_DETERMINISTIC_REFERENCE_DIAGONAL_SHIFT,
        observation_mode: ToroidalExactBObservationMode::SourceDesignedFluxes,
        source_deltas: [0.0, 0.15, -0.10, 0.05],
        source_prior_std: 0.25,
        reference_pde_variance: Some(1e-8),
        prior_kappa: 1.0,
        prior_tau: TOROIDAL_EXACT_B_CANONICAL_PRIOR_TAU,
        observation_train_fraction: 0.33,
        observation_noise_std: 3e-10,
        synthetic_observation_noise_seed: Some(TOROIDAL_EXACT_B_SYNTHETIC_OBSERVATION_NOISE_SEED),
        synthetic_heldout_observation_noise_seed: Some(
            TOROIDAL_EXACT_B_SYNTHETIC_HELDOUT_NOISE_SEED,
        ),
        heldout_count: 128,
        surface_flux_azimuth_count: 12,
        nondimensionalization: ToroidalExactBNondimensionalizationMode::PdeColumnNorm,
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_exact_b_source_designed_flux_workflow",
        )),
        base: ToroidalExactBBaseConfig {
            pde_variance: 3e-8,
            precision_policy: LinearPdePrecisionPolicy::DiagonalEquilibrated {
                max_relative_asymmetry: 1.0e-10,
            },
            ..Default::default()
        },
        ..ToroidalExactBRecoveryConfig::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToroidalExactBObservationIndexOverride {
    pub training_indices: Vec<usize>,
    pub heldout_indices: Vec<usize>,
}

/// Exact observation split used to generate the submitted canonical toroidal
/// source-designed-flux artifacts.
///
/// The generic greedy design remains the default for research configurations.
/// This explicit split belongs to the immutable publication profile because
/// the final greedy choice is a near-tie and can vary with solver arithmetic.
pub fn toroidal_exact_b_thesis_submitted_observation_index_override(
) -> ToroidalExactBObservationIndexOverride {
    ToroidalExactBObservationIndexOverride {
        training_indices: vec![0, 1, 6, 9, 10, 12, 16, 19, 22, 23, 29, 32],
        heldout_indices: vec![
            2, 3, 4, 5, 7, 8, 11, 13, 14, 15, 17, 18, 20, 21, 24, 25, 26, 27, 28, 30, 31, 33, 34,
            35,
        ],
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBProbeRow {
    pub name: String,
    pub cell_index: usize,
    pub component: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBStageSummary {
    pub reference_mode: ToroidalExactBReferenceMode,
    pub reference_solve_mode: ToroidalExactBReferenceSolveMode,
    pub reference_solver_diagonal_shift: f64,
    pub observation_mode: ToroidalExactBObservationMode,
    pub prior_tau: f64,
    pub prior_calibration_label: String,
    pub prior_calibration_nominal_b_rms: Option<f64>,
    pub prior_calibration_target_b_rms: Option<f64>,
    pub prior_calibration_multiplier: Option<f64>,
    pub active_dofs: usize,
    pub source_modes: usize,
    pub training_rows: usize,
    pub heldout_rows: usize,
    pub residual_rows_used: usize,
    pub residual_rows_total: usize,
    pub final_residual_norm: f64,
    pub train_rmse: f64,
    pub heldout_rmse: f64,
    pub heldout_nlpd: f64,
    pub heldout_covered95: usize,
    pub heldout_coverage_fraction: f64,
    pub heldout_mean_posterior_flux_sd: f64,
    pub heldout_max_abs_z: f64,
    pub heldout_rms_z: f64,
    pub heldout_mean_abs_residual: f64,
    pub heldout_noisy_rmse: f64,
    pub heldout_noisy_nlpd: f64,
    pub heldout_noisy_covered95: usize,
    pub heldout_noisy_coverage_fraction: f64,
    pub heldout_mean_predictive_sd: f64,
    pub heldout_noisy_max_abs_z: f64,
    pub heldout_noisy_rms_z: f64,
    pub heldout_noisy_mean_abs_residual: f64,
    pub b_relative_error: Option<f64>,
    pub posterior_factor_nnz: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBSourcePosteriorRow {
    pub mode_index: usize,
    pub prior_mean: f64,
    pub prior_variance: f64,
    pub truth: f64,
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    pub error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBHeldoutPredictionRow {
    pub name: String,
    pub truth: f64,
    pub noisy_observation: f64,
    pub prediction: f64,
    pub residual: f64,
    pub noisy_residual: f64,
    pub posterior_sd: f64,
    pub predictive_sd: f64,
    pub standardized_residual: f64,
    pub noisy_standardized_residual: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub covered95: bool,
    pub noisy_lower95: f64,
    pub noisy_upper95: f64,
    pub noisy_covered95: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBFluxRowNorm {
    pub row_index: usize,
    pub name: String,
    pub nnz: usize,
    pub l2_norm: f64,
    pub max_abs_entry: f64,
    pub max_diagonal_contribution: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBPhysicalBPriorCalibrationReport {
    pub calibration_label: String,
    pub mesh_path: PathBuf,
    pub reference_solve_mode: ToroidalExactBReferenceSolveMode,
    pub reference_solver_diagonal_shift: f64,
    pub active_dofs: usize,
    pub cells: usize,
    pub b_rows: usize,
    pub domain_volume: f64,
    pub prior_kappa: f64,
    pub raw_prior_tau: f64,
    pub target_multiplier: f64,
    pub nominal_b_rms: f64,
    pub target_prior_b_rms: f64,
    pub raw_trace: f64,
    pub raw_mean_b2: f64,
    pub target_trace: f64,
    pub precision_scale: f64,
    pub tau_multiplier: f64,
    pub effective_prior_tau: f64,
    pub normalized_mean_b2: f64,
    pub normalization_relative_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBPrecisionScaleAudit {
    pub joint_dimension: usize,
    pub state_dimension: usize,
    pub source_dimension: usize,
    pub training_rows: usize,
    pub heldout_rows: usize,
    pub nondimensionalization: ToroidalExactBNondimensionalizationMode,
    pub state_scale_min: f64,
    pub state_scale_max: f64,
    pub source_scale_min: f64,
    pub source_scale_max: f64,
    pub scaled_posterior_max_abs_diagonal: f64,
    pub scaled_posterior_diagonal_ratio: f64,
    pub terms: Vec<ToroidalExactBPrecisionTermScale>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBPrecisionTermScale {
    pub term: String,
    pub nonzero_diagonal_entries: usize,
    pub min_positive_diagonal: f64,
    pub max_abs_diagonal: f64,
    pub diagonal_sum: f64,
    pub max_index: usize,
    pub max_block: String,
    pub max_local_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToroidalExactBSourceResponseSummary {
    pub condition: f64,
    pub singular_values: [f64; SOURCE_MODE_COUNT],
    pub column_norms: [f64; SOURCE_MODE_COUNT],
    pub snr_min: f64,
    pub snr_max: f64,
}

pub struct ToroidalExactBRecoveryReport {
    pub summary: ToroidalExactBStageSummary,
    pub source_response: ToroidalExactBSourceResponseSummary,
    pub source_posterior: Vec<ToroidalExactBSourcePosteriorRow>,
    pub heldout_predictions: Vec<ToroidalExactBHeldoutPredictionRow>,
    pub probes: Vec<ToroidalExactBProbeRow>,
    pub result: LinearPdeUqResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalExactBDiagnosticObservationMode {
    CellMagneticComponents,
    SurfaceFluxes,
    SourceDesignedFluxes,
    FullFieldOracle,
    PdeOnly,
}

impl ToroidalExactBDiagnosticObservationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::CellMagneticComponents => "cell_magnetic_components",
            Self::SurfaceFluxes => "surface_fluxes",
            Self::SourceDesignedFluxes => "source_designed_fluxes",
            Self::FullFieldOracle => "full_field_oracle",
            Self::PdeOnly => "pde_only",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToroidalExactBDiagnosticsConfig {
    pub base: ToroidalExactBRecoveryConfig,
    pub observation_modes: Vec<ToroidalExactBDiagnosticObservationMode>,
    pub pde_variances: Vec<f64>,
    pub prior_taus: Vec<f64>,
    pub source_prior_stds: Vec<f64>,
    pub observation_noise_stds: Vec<f64>,
    pub include_source_response: bool,
    pub output_dir: Option<PathBuf>,
    pub write_outputs: bool,
}

impl Default for ToroidalExactBDiagnosticsConfig {
    fn default() -> Self {
        Self {
            base: ToroidalExactBRecoveryConfig::default(),
            observation_modes: vec![
                ToroidalExactBDiagnosticObservationMode::CellMagneticComponents,
                ToroidalExactBDiagnosticObservationMode::SurfaceFluxes,
                ToroidalExactBDiagnosticObservationMode::SourceDesignedFluxes,
                ToroidalExactBDiagnosticObservationMode::FullFieldOracle,
                ToroidalExactBDiagnosticObservationMode::PdeOnly,
            ],
            pde_variances: vec![1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
            prior_taus: vec![1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            source_prior_stds: vec![0.025, 0.25, 2.5, 25.0],
            observation_noise_stds: vec![1e-10, 1e-9, 1e-8],
            include_source_response: true,
            output_dir: Some(PathBuf::from("out/examples/toroidal_exact_b_diagnostics")),
            write_outputs: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalExactBDiagnosticRow {
    pub sweep: String,
    pub observation_mode: ToroidalExactBDiagnosticObservationMode,
    pub pde_variance: f64,
    pub prior_tau: f64,
    pub source_prior_std: f64,
    pub observation_noise_std: f64,
    pub training_rows: usize,
    pub heldout_rows: usize,
    pub train_rmse: f64,
    pub heldout_rmse: f64,
    pub b_relative_error: f64,
    pub final_residual_norm: f64,
    pub source_response_condition: f64,
    pub source_response_singular_values: [f64; SOURCE_MODE_COUNT],
    pub source_response_column_norms: [f64; SOURCE_MODE_COUNT],
    pub source_response_snr_min: f64,
    pub source_response_snr_max: f64,
    pub eta_truth: [f64; SOURCE_MODE_COUNT],
    pub eta_posterior_mean: [f64; SOURCE_MODE_COUNT],
    pub eta_posterior_variance: [f64; SOURCE_MODE_COUNT],
    pub eta_error: [f64; SOURCE_MODE_COUNT],
    pub posterior_factor_nnz: usize,
    pub runtime_seconds: f64,
}

pub struct ToroidalExactBDiagnosticsReport {
    pub rows: Vec<ToroidalExactBDiagnosticRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToroidalObservationDiagnosticReport {
    pub active_dofs: usize,
    pub cells: usize,
    pub pde_observation_mode: ToroidalPdeObservationMode,
    pub residual_rows_used: usize,
    pub residual_rows_total: usize,
    pub weak_residual_at_zero: f64,
    pub weak_residual_at_linear_mean: f64,
    pub observation_residual_at_zero: Option<f64>,
    pub observation_residual_at_linear_mean: Option<f64>,
    pub weighted_observation_residual_at_zero: Option<f64>,
    pub weighted_observation_residual_at_linear_mean: Option<f64>,
    pub observation_delta_from_zero_to_linear_mean: Option<f64>,
    pub observation_source_norm: Option<f64>,
    pub observation_field_response_norm: Option<f64>,
    pub observation_best_source_scale: Option<f64>,
    pub observation_field_source_cosine: Option<f64>,
    pub observation_jacobian_nnz_at_linear_mean: Option<usize>,
}

pub fn diagnose_toroidal_observation(
    config: &NonlinearToroidalConfig,
) -> Result<ToroidalObservationDiagnosticReport, String> {
    let (topology, coords) = load_mesh(&config.mesh_path)?;
    let boundary = outer_boundary(&topology, &coords, config.geometry);
    let linear_material = ToroidalReluctivityLaw::new(
        config.geometry.major_radius,
        config.geometry.core_minor_radius,
        config.nu_air,
        config.nu_core0,
        0.0,
    )?;
    let material = ToroidalReluctivityLaw::new(
        config.geometry.major_radius,
        config.geometry.core_minor_radius,
        config.nu_air,
        config.nu_core0,
        config.beta_core,
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(
            material,
            toroidal_essential_boundary(&topology, &boundary, false)?,
        ),
    )?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(
            linear_material,
            toroidal_essential_boundary(&topology, &boundary, false)?,
        ),
    )?;
    let source_full = assemble_toroidal_source(&topology, &coords, config.geometry, config.nu_air);
    let source_reduced = reduce_full_edge_vector(source_free.layout(), &source_full)?;
    let model = source_free.with_source(source_reduced)?;
    let linear_model = linear_source_free.with_source(model.source())?;
    let zero_state = vec![0.0; model.reduced_dimension()];
    let linear_mean = direct_reduced_hodge_linear_solution(
        &linear_model,
        config.prior_precision,
        config.pde_variance,
    )?;
    let weak_residual_at_zero = l2_norm(
        model
            .residual_and_jacobian(&zero_state)?
            .residual
            .as_slice(),
    );
    let weak_residual_at_linear_mean = l2_norm(
        model
            .residual_and_jacobian(&linear_mean)?
            .residual
            .as_slice(),
    );
    let rows_total = toroidal_observation_residual_dimension(&model, config.pde_observation_mode);
    let observation = if config.include_nonlinear_residual {
        Some(build_toroidal_observation_residual(
            config, &topology, &coords, &model, rows_total,
        )?)
    } else {
        None
    };
    let zero_observation_evaluation = observation
        .as_ref()
        .filter(|observation| observation.rows_used > 0)
        .map(|observation| observation.model.residual_and_jacobian(&zero_state))
        .transpose()?;
    let linear_observation_evaluation = observation
        .as_ref()
        .filter(|observation| observation.rows_used > 0)
        .map(|observation| observation.model.residual_and_jacobian(&linear_mean))
        .transpose()?;
    let weighted_zero = observation
        .as_ref()
        .zip(zero_observation_evaluation.as_ref())
        .map(|(observation, evaluation)| {
            weighted_residual_norm(&evaluation.residual, &observation.noise.noise)
        })
        .transpose()?;
    let weighted_linear = observation
        .as_ref()
        .zip(linear_observation_evaluation.as_ref())
        .map(|(observation, evaluation)| {
            weighted_residual_norm(&evaluation.residual, &observation.noise.noise)
        })
        .transpose()?;
    let observation_delta = zero_observation_evaluation
        .as_ref()
        .zip(linear_observation_evaluation.as_ref())
        .map(|(zero, linear)| {
            linear
                .residual
                .iter()
                .zip(zero.residual.iter())
                .map(|(linear, zero)| (linear - zero).powi(2))
                .sum::<f64>()
                .sqrt()
        });
    let observation_source_norm = zero_observation_evaluation
        .as_ref()
        .map(|zero| l2_norm(&zero.residual));
    let observation_field_response_norm = observation_delta;
    let source_scale_and_cosine = zero_observation_evaluation
        .as_ref()
        .zip(linear_observation_evaluation.as_ref())
        .map(|(zero, linear)| {
            let mut source_norm_sq = 0.0;
            let mut field_norm_sq = 0.0;
            let mut dot = 0.0;
            for (zero, linear) in zero.residual.iter().zip(linear.residual.iter()) {
                let source = -*zero;
                let field = linear - zero;
                source_norm_sq += source * source;
                field_norm_sq += field * field;
                dot += field * source;
            }
            let best_scale = dot / source_norm_sq.max(1e-300);
            let cosine = dot / (source_norm_sq.sqrt() * field_norm_sq.sqrt()).max(1e-300);
            (best_scale, cosine)
        });
    Ok(ToroidalObservationDiagnosticReport {
        active_dofs: model.reduced_dimension(),
        cells: topology.nsimplices(3),
        pde_observation_mode: config.pde_observation_mode,
        residual_rows_used: observation
            .as_ref()
            .map(|observation| observation.rows_used)
            .unwrap_or(0),
        residual_rows_total: rows_total,
        weak_residual_at_zero,
        weak_residual_at_linear_mean,
        observation_residual_at_zero: zero_observation_evaluation
            .as_ref()
            .map(|evaluation| l2_norm(&evaluation.residual)),
        observation_residual_at_linear_mean: linear_observation_evaluation
            .as_ref()
            .map(|evaluation| l2_norm(&evaluation.residual)),
        weighted_observation_residual_at_zero: weighted_zero,
        weighted_observation_residual_at_linear_mean: weighted_linear,
        observation_delta_from_zero_to_linear_mean: observation_delta,
        observation_source_norm,
        observation_field_response_norm,
        observation_best_source_scale: source_scale_and_cosine.map(|(scale, _)| scale),
        observation_field_source_cosine: source_scale_and_cosine.map(|(_, cosine)| cosine),
        observation_jacobian_nnz_at_linear_mean: linear_observation_evaluation
            .as_ref()
            .map(|evaluation| evaluation.jacobian.nnz()),
    })
}

pub fn run_nonlinear_toroidal_inductor(
    config: &NonlinearToroidalConfig,
) -> Result<NonlinearToroidalReport, String> {
    let (topology, coords) = load_mesh(&config.mesh_path)?;
    let boundary = outer_boundary(&topology, &coords, config.geometry);
    let linear_material = ToroidalReluctivityLaw::new(
        config.geometry.major_radius,
        config.geometry.core_minor_radius,
        config.nu_air,
        config.nu_core0,
        0.0,
    )?;
    let material = ToroidalReluctivityLaw::new(
        config.geometry.major_radius,
        config.geometry.core_minor_radius,
        config.nu_air,
        config.nu_core0,
        config.beta_core,
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(
            material,
            toroidal_essential_boundary(&topology, &boundary, false)?,
        ),
    )?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(
            linear_material,
            toroidal_essential_boundary(&topology, &boundary, false)?,
        ),
    )?;
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("nonlinear and beta-zero toroidal layouts differ".to_string());
    }
    let source_full = assemble_toroidal_source(&topology, &coords, config.geometry, config.nu_air);
    let source_reduced = reduce_full_edge_vector(source_free.layout(), &source_full)?;
    let model = source_free.with_source(source_reduced)?;
    let linear_model = linear_source_free.with_source(model.source())?;
    let zero_state = vec![0.0; model.reduced_dimension()];
    let initial_residual_norm = l2_norm(
        model
            .residual_and_jacobian(&zero_state)?
            .residual
            .as_slice(),
    );
    let linear_mean = direct_reduced_hodge_linear_solution(
        &linear_model,
        config.prior_precision,
        config.pde_variance,
    )?;
    let prior = build_toroidal_prior(
        config,
        &topology,
        &coords,
        &linear_model,
        linear_mean.clone(),
    )?;
    let mut derived_quantities = config.extra_derived_quantities.clone();
    if config.include_cell_b_variance {
        derived_quantities.push(LinearPdeDerivedQuantitySpec {
            name: "cell_B".to_string(),
            operator: build_reduced_magnetic_flux_density_operator_3d(
                &topology,
                &coords,
                model.layout(),
            )?,
        });
    }
    let observation_rows_total =
        toroidal_observation_residual_dimension(&model, config.pde_observation_mode);
    let observation_build = if config.include_nonlinear_residual {
        Some(build_toroidal_observation_residual(
            config,
            &topology,
            &coords,
            &model,
            observation_rows_total,
        )?)
    } else {
        None
    };
    let residual_terms = if let Some(observation) = observation_build.as_ref() {
        if observation.rows_used == 0 {
            Vec::new()
        } else {
            vec![NonlinearResidualTerm::zero(
                "nonlinear_toroidal_magnetostatic",
                &observation.model,
                observation.noise.noise.clone(),
            )]
        }
    } else {
        Vec::new()
    };
    let observation_initial_residual_norm = observation_build
        .as_ref()
        .filter(|observation| observation.rows_used > 0)
        .map(|observation| {
            observation
                .model
                .residual_and_jacobian(&linear_mean)
                .map(|evaluation| l2_norm(&evaluation.residual))
        })
        .transpose()?;
    let problem = NonlinearLaplaceProblem {
        prior: prior.spec,
        residual_terms,
        linear_measurements: config.linear_measurements.clone(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities,
    };
    let result = solve_nonlinear_laplace(
        &problem,
        &GaussNewtonConfig {
            initial_guess: Some(linear_mean.clone()),
            max_iterations: config.max_iterations,
            step_tolerance: 1e-10,
            gradient_tolerance: 1e-5,
            max_line_search_steps: 20,
            linear_solve: config.linear_solve,
            variance: config.variance,
            ..GaussNewtonConfig::default()
        },
    )?;
    let true_final_residual = model.residual_and_jacobian(&result.map)?.residual;
    let final_residual_norm = l2_norm(true_final_residual.as_slice());
    let observation_final_residual_norm = observation_build
        .as_ref()
        .filter(|observation| observation.rows_used > 0)
        .map(|observation| {
            observation
                .model
                .residual_and_jacobian(&result.map)
                .map(|evaluation| l2_norm(&evaluation.residual))
        })
        .transpose()?;
    let effective_converged =
        result.converged || final_residual_norm <= 1e-10 * initial_residual_norm.max(1.0);
    let posterior_factorizes = true;

    let beta_zero = if config.beta_core == 0.0 {
        Some(linear_mean.clone())
    } else {
        None
    };
    let direct_linear_relative_error = beta_zero
        .as_ref()
        .map(|linear| relative_error(&result.map, linear));
    let mixed_reference = if config.beta_core == 0.0 {
        Some(solve_mixed_linear_reference(
            &topology,
            &coords,
            &source_full,
            config.geometry,
            config.nu_air,
        )?)
    } else {
        None
    };
    let mixed_reference_b_relative_error = mixed_reference
        .as_ref()
        .map(|reference| b_relative_error(&topology, &coords, &model, &result.map, reference))
        .transpose()?;
    let sensor_reports = build_sensor_reports(
        &topology,
        &coords,
        &model,
        &result.map,
        beta_zero.as_deref(),
        mixed_reference.as_ref(),
        config.geometry,
    )?;
    let harmonic_coefficients = if config.compute_harmonic_diagnostics {
        compute_harmonic_coefficients(
            &topology,
            &coords,
            &model,
            &result.map,
            config.geometry,
            config.nu_air,
        )?
    } else {
        Vec::new()
    };
    let harmonic_coefficient_norm = if config.compute_harmonic_diagnostics {
        l2_norm(&harmonic_coefficients)
    } else {
        f64::NAN
    };
    let (b_variance_len, b_variance_min, b_variance_max) =
        if let Some(variance) = result.derived_variances.get("cell_B") {
            let (min, max) = finite_min_max(variance.posterior_variance.iter().copied());
            (
                Some(variance.posterior_variance.len()),
                Some(min),
                Some(max),
            )
        } else {
            (None, None, None)
        };

    if config.write_outputs {
        if let Some(output_dir) = &config.output_dir {
            write_nonlinear_toroidal_outputs(output_dir, &topology, &coords, &model, &result.map)?;
        }
    }

    Ok(NonlinearToroidalReport {
        active_dofs: model.reduced_dimension(),
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        gauge_edge_dofs: model.gauge_edge_dofs().len(),
        prior_mode: config.prior_mode,
        prior_kappa: prior.kappa,
        prior_tau: prior.tau,
        prior_kappa_fallback_used: prior.kappa_fallback_used,
        residual_variance: config.pde_variance,
        pde_observation_mode: config.pde_observation_mode,
        residual_weighting: config.residual_weighting,
        residual_precision_normalization: observation_build
            .as_ref()
            .and_then(|observation| observation.noise.normalization_scale),
        residual_rows_used: observation_build
            .as_ref()
            .map(|observation| observation.rows_used)
            .unwrap_or(0),
        residual_rows_total: observation_build
            .as_ref()
            .map(|observation| observation.rows_total)
            .unwrap_or(observation_rows_total),
        nonlinear_residual_likelihood: config.include_nonlinear_residual
            && observation_build
                .as_ref()
                .map(|observation| observation.rows_used > 0)
                .unwrap_or(false),
        observation_initial_residual_norm,
        observation_final_residual_norm,
        converged: effective_converged,
        iterations: result.history.len(),
        initial_residual_norm,
        final_residual_norm,
        map_relative_distance_from_linear_mean: relative_error(&result.map, &linear_mean),
        direct_linear_relative_error,
        mixed_reference_b_relative_error,
        harmonic_coefficients,
        harmonic_coefficient_norm,
        sensor_reports,
        posterior_precision_nnz: result.posterior_precision.nnz(),
        posterior_factor_nnz: result.final_factorization.nnz,
        final_factorization_seconds: result.final_factorization.elapsed_seconds,
        linear_solve_mode: result
            .history
            .first()
            .map(|entry| entry.linear_solve.mode)
            .unwrap_or(match config.linear_solve {
                GaussNewtonLinearSolve::DirectCholesky => {
                    GaussNewtonLinearSolveMode::DirectCholesky
                }
                GaussNewtonLinearSolve::IterativeCg { .. } => {
                    GaussNewtonLinearSolveMode::IterativeCg
                }
            }),
        linear_solve_iteration_sum: result
            .history
            .iter()
            .map(|entry| entry.linear_solve.iterations)
            .sum(),
        linear_solve_residual_max: result
            .history
            .iter()
            .map(|entry| entry.linear_solve.final_residual_norm)
            .fold(0.0, f64::max),
        posterior_factorizes,
        latent_variance_len: result.posterior_variance.len(),
        b_variance_len,
        b_variance_min,
        b_variance_max,
        result,
    })
}

pub fn run_toroidal_exact_b_recovery_experiment(
    config: &ToroidalExactBRecoveryConfig,
) -> Result<ToroidalExactBRecoveryReport, String> {
    validate_exact_b_config(config)?;

    let (topology, coords) = load_mesh(&config.base.mesh_path)?;
    let boundary = outer_boundary(&topology, &coords, config.base.geometry);
    let linear_system = build_toroidal_exact_b_linear_system(&topology, &coords, boundary, config)?;

    let nominal_source_full =
        assemble_toroidal_source(&topology, &coords, config.base.geometry, config.base.nu_air);
    let nominal_source = reduce_full_edge_vector(&linear_system.layout, &nominal_source_full)?;
    let source_modes = assemble_toroidal_source_modes(
        &topology,
        &coords,
        &linear_system.layout,
        config.base.geometry,
        config.base.nu_air,
    )?;
    let truth_eta = match config.reference_mode {
        ToroidalExactBReferenceMode::NominalDebug => [0.0; SOURCE_MODE_COUNT],
        ToroidalExactBReferenceMode::PerturbedSource => config.source_deltas,
    };
    let truth_source = source_with_sector_deltas(&nominal_source, &source_modes, &truth_eta)?;
    let reference_pde_variance = config
        .reference_pde_variance
        .unwrap_or(config.base.pde_variance);

    let linear_mean = solve_exact_b_reference_state(
        &linear_system,
        &nominal_source,
        config.reference_solve_mode,
        config.reference_solver_diagonal_shift,
        config.base.prior_precision,
        reference_pde_variance,
    )?;
    let exact_prior = build_exact_two_form_potential_prior(
        &topology,
        &coords,
        &linear_system.layout,
        linear_mean.clone(),
        ExactTwoFormPotentialPriorConfig {
            kappa: config.prior_kappa,
            tau: config.prior_tau,
            mass_inverse: config.potential_mass_inverse,
            diagonal_shift: 0.0,
        },
    )?;

    let reference_state = if config.reference_observation_csv_path.is_none() {
        Some(solve_exact_b_reference_state(
            &linear_system,
            &truth_source,
            config.reference_solve_mode,
            config.reference_solver_diagonal_shift,
            config.base.prior_precision,
            reference_pde_variance,
        )?)
    } else {
        None
    };
    let reference_full = reference_state
        .as_ref()
        .map(|state| {
            lift_vector_with_layout(&linear_system.layout, &FeecVector::from_vec(state.clone()))
                .map(|full| full.as_slice().to_vec())
        })
        .transpose()?;

    let observation_system = build_exact_b_observation_system(
        &topology,
        &coords,
        config.observation_mode,
        config.base.geometry,
        config.surface_flux_azimuth_count,
    )?;
    let (training_indices, heldout_indices) = resolve_exact_b_observation_indices(
        config,
        &linear_system,
        &source_modes,
        &observation_system.operator,
    )?;
    let probe_roles = exact_b_probe_roles(
        observation_system.operator.nrows(),
        &training_indices,
        &heldout_indices,
    );
    let probes = observation_system.probes_with_roles(&probe_roles)?;
    let reference_values = if let Some(path) = config.reference_observation_csv_path.as_deref() {
        load_exact_b_reference_observations(path, &probes, &training_indices, &heldout_indices)?
    } else {
        let truth = reference_state
            .as_ref()
            .ok_or_else(|| "synthetic reference state was not produced".to_string())?;
        let truth_full =
            lift_vector_with_layout(&linear_system.layout, &FeecVector::from_vec(truth.clone()))?;
        apply_sparse_triplet(&observation_system.operator, truth_full.as_slice())
    };

    let measurement = build_exact_b_measurement(
        &observation_system.operator,
        &reference_values,
        &training_indices,
        config.observation_noise_std * config.observation_noise_std,
        config.synthetic_observation_noise_seed,
    )?;
    let source_response = exact_b_source_response_summary(
        &linear_system,
        &source_modes,
        Some(&measurement.operator),
        config.base.prior_precision,
        config.base.pde_variance,
        config.observation_noise_std,
    )?;
    let derived_quantities = build_exact_b_heldout_derived_quantities(
        &observation_system.operator,
        &probes,
        &heldout_indices,
    )?;

    let selected_rows = select_residual_rows(
        linear_system.residual_dimension(),
        &config.base.residual_selection,
    )?;
    let nominal_source_selected = select_vector_entries(&nominal_source, &selected_rows)?;
    let source_modes_selected = source_modes
        .iter()
        .map(|mode| select_vector_entries(mode, &selected_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let mut inference_system = select_linear_system_residual_rows(&linear_system, &selected_rows)?;
    for (bias, source) in inference_system
        .residual_bias
        .iter_mut()
        .zip(&nominal_source_selected)
    {
        *bias -= *source;
    }
    let source_operator = build_source_discrepancy_operator(&source_modes_selected)?;
    let source_prior_variance = config.source_prior_std * config.source_prior_std;
    let inference_operator = feec_csr_to_core_triplet(&inference_system.operator);
    let coordinate_scaling = exact_b_coordinate_scaling(
        config.nondimensionalization,
        &inference_operator,
        &source_operator,
    )?;
    let problem = LinearPdeUqProblem {
        state_prior: exact_prior.spec,
        system: inference_system,
        uncertain_inputs: vec![LinearUncertainInputSpec {
            name: SOURCE_DISCREPANCY_INPUT.to_string(),
            operator: source_operator,
            prior: zero_mean_diagonal_prior(SOURCE_MODE_COUNT, 1.0 / source_prior_variance),
            preference: RepresentationPreference::ForceLatent,
            collapsed_precision: None,
        }],
        physical_measurements: vec![measurement],
        joint_measurements: Vec::new(),
        derived_quantities,
        joint_derived_quantities: Vec::new(),
        pde_variance: Some(config.base.pde_variance),
        pde_precision: None,
    };
    let solver_config = LinearPdeUqSolverConfig {
        variance: config.base.variance,
        precision_policy: config.base.precision_policy,
        ..LinearPdeUqSolverConfig::default()
    };
    let result = if let Some(scaling) = coordinate_scaling.as_ref() {
        solve_scaled_linear_pde_uq_with_config(&problem, scaling, &solver_config)?
    } else {
        solve_linear_pde_uq_with_config(&problem, &solver_config)?
    };

    let predictions = apply_sparse_triplet(
        &observation_system.operator,
        result.posterior_mean.as_slice(),
    );
    let train_rmse = indexed_rmse(&predictions, &reference_values, &training_indices);
    let heldout_predictions = exact_b_heldout_prediction_rows(
        &result,
        &probes,
        &predictions,
        &reference_values,
        &heldout_indices,
        config.observation_noise_std,
        config.synthetic_heldout_observation_noise_seed,
    )?;
    let heldout_rmse = prediction_rmse_exact_b(&heldout_predictions);
    let heldout_nlpd = prediction_nlpd_exact_b(&heldout_predictions);
    let heldout_covered95 = heldout_predictions
        .iter()
        .filter(|row| row.covered95)
        .count();
    let heldout_noisy_rmse = noisy_prediction_rmse_exact_b(&heldout_predictions);
    let heldout_noisy_nlpd = noisy_prediction_nlpd_exact_b(&heldout_predictions);
    let heldout_noisy_covered95 = heldout_predictions
        .iter()
        .filter(|row| row.noisy_covered95)
        .count();
    let heldout_metrics = exact_b_prediction_summary_metrics(&heldout_predictions);
    let final_residual_norm = l2_norm(result.pde_residual_mean.as_slice());
    let cell_b_operator = build_full_magnetic_flux_density_triplet_operator(&topology, &coords)?;
    let b_relative_error = reference_full.as_ref().map(|truth| {
        relative_error(
            &apply_sparse_triplet(&cell_b_operator, result.posterior_mean.as_slice()),
            &apply_sparse_triplet(&cell_b_operator, truth),
        )
    });
    let source_posterior =
        exact_b_source_posterior_rows(&result, &truth_eta, config.source_prior_std)?;
    let calibration_metadata = exact_b_prior_calibration_metadata(config.prior_tau);
    let summary = ToroidalExactBStageSummary {
        reference_mode: config.reference_mode,
        reference_solve_mode: config.reference_solve_mode,
        reference_solver_diagonal_shift: config.reference_solver_diagonal_shift,
        observation_mode: config.observation_mode,
        prior_tau: config.prior_tau,
        prior_calibration_label: calibration_metadata.0,
        prior_calibration_nominal_b_rms: calibration_metadata.1,
        prior_calibration_target_b_rms: calibration_metadata.2,
        prior_calibration_multiplier: calibration_metadata.3,
        active_dofs: linear_system.state_dimension(),
        source_modes: SOURCE_MODE_COUNT,
        training_rows: training_indices.len(),
        heldout_rows: heldout_indices.len(),
        residual_rows_used: selected_rows.len(),
        residual_rows_total: linear_system.residual_dimension(),
        final_residual_norm,
        train_rmse,
        heldout_rmse,
        heldout_nlpd,
        heldout_covered95,
        heldout_coverage_fraction: if heldout_indices.is_empty() {
            f64::NAN
        } else {
            heldout_covered95 as f64 / heldout_indices.len() as f64
        },
        heldout_mean_posterior_flux_sd: heldout_metrics.mean_posterior_sd,
        heldout_max_abs_z: heldout_metrics.max_abs_z,
        heldout_rms_z: heldout_metrics.rms_z,
        heldout_mean_abs_residual: heldout_metrics.mean_abs_residual,
        heldout_noisy_rmse,
        heldout_noisy_nlpd,
        heldout_noisy_covered95,
        heldout_noisy_coverage_fraction: if heldout_indices.is_empty() {
            f64::NAN
        } else {
            heldout_noisy_covered95 as f64 / heldout_indices.len() as f64
        },
        heldout_mean_predictive_sd: heldout_metrics.mean_predictive_sd,
        heldout_noisy_max_abs_z: heldout_metrics.noisy_max_abs_z,
        heldout_noisy_rms_z: heldout_metrics.noisy_rms_z,
        heldout_noisy_mean_abs_residual: heldout_metrics.noisy_mean_abs_residual,
        b_relative_error,
        posterior_factor_nnz: result.debug.posterior_factorization.factor_nnz,
    };

    if config.write_outputs {
        if let Some(output_dir) = &config.output_dir {
            write_exact_b_recovery_outputs(
                output_dir,
                &topology,
                &coords,
                result.posterior_mean.as_slice(),
                reference_full.as_deref(),
                &summary,
                &source_response,
                &source_posterior,
                &heldout_predictions,
                &probes,
            )?;
        }
    }

    Ok(ToroidalExactBRecoveryReport {
        summary,
        source_response,
        source_posterior,
        heldout_predictions,
        probes,
        result,
    })
}

pub fn run_toroidal_exact_b_diagnostics(
    config: &ToroidalExactBDiagnosticsConfig,
) -> Result<ToroidalExactBDiagnosticsReport, String> {
    validate_exact_b_config(&config.base)?;
    let mut rows = Vec::new();
    if config.write_outputs {
        if let Some(output_dir) = &config.output_dir {
            fs::create_dir_all(output_dir).map_err(|err| {
                format!(
                    "failed to create exact-B diagnostics directory `{}`: {err}",
                    output_dir.display()
                )
            })?;
        }
    }
    for (sweep, mode, case_config) in exact_b_diagnostic_cases(config) {
        eprintln!("exact-B diagnostic: {} {sweep}", mode.label());
        rows.push(run_toroidal_exact_b_diagnostic_case(
            &case_config,
            mode,
            &sweep,
            config.include_source_response,
        )?);
        if config.write_outputs {
            if let Some(output_dir) = &config.output_dir {
                fs::write(
                    output_dir.join("diagnostics_summary.csv"),
                    exact_b_diagnostics_csv(&rows),
                )
                .map_err(|err| err.to_string())?;
            }
        }
    }
    if config.write_outputs {
        if let Some(output_dir) = &config.output_dir {
            fs::write(
                output_dir.join("diagnostics_summary.csv"),
                exact_b_diagnostics_csv(&rows),
            )
            .map_err(|err| err.to_string())?;
        }
    }
    Ok(ToroidalExactBDiagnosticsReport { rows })
}

fn exact_b_diagnostic_cases(
    config: &ToroidalExactBDiagnosticsConfig,
) -> Vec<(
    String,
    ToroidalExactBDiagnosticObservationMode,
    ToroidalExactBRecoveryConfig,
)> {
    let mut cases = Vec::new();
    for mode in &config.observation_modes {
        let mut baseline = exact_b_diagnostic_case_config(&config.base, *mode);
        if baseline.reference_pde_variance.is_none() {
            baseline.reference_pde_variance = Some(config.base.base.pde_variance);
        }
        cases.push(("baseline".to_string(), *mode, baseline.clone()));

        for value in &config.pde_variances {
            if approx_equal(*value, config.base.base.pde_variance) {
                continue;
            }
            let mut case = baseline.clone();
            case.base.pde_variance = *value;
            cases.push((format!("pde_variance={value:.0e}"), *mode, case));
        }
        for value in &config.prior_taus {
            if approx_equal(*value, config.base.prior_tau) {
                continue;
            }
            let mut case = baseline.clone();
            case.prior_tau = *value;
            cases.push((format!("prior_tau={value:.0e}"), *mode, case));
        }
        for value in &config.source_prior_stds {
            if approx_equal(*value, config.base.source_prior_std) {
                continue;
            }
            let mut case = baseline.clone();
            case.source_prior_std = *value;
            cases.push((format!("source_prior_std={value:.3e}"), *mode, case));
        }
        for value in &config.observation_noise_stds {
            if approx_equal(*value, config.base.observation_noise_std) {
                continue;
            }
            let mut case = baseline.clone();
            case.observation_noise_std = *value;
            cases.push((format!("observation_noise_std={value:.0e}"), *mode, case));
        }
    }
    cases
}

fn exact_b_diagnostic_case_config(
    base: &ToroidalExactBRecoveryConfig,
    mode: ToroidalExactBDiagnosticObservationMode,
) -> ToroidalExactBRecoveryConfig {
    let mut config = base.clone();
    match mode {
        ToroidalExactBDiagnosticObservationMode::CellMagneticComponents
        | ToroidalExactBDiagnosticObservationMode::FullFieldOracle
        | ToroidalExactBDiagnosticObservationMode::PdeOnly => {
            config.observation_mode = ToroidalExactBObservationMode::CellMagneticComponents;
        }
        ToroidalExactBDiagnosticObservationMode::SurfaceFluxes
        | ToroidalExactBDiagnosticObservationMode::SourceDesignedFluxes => {
            config.observation_mode = match mode {
                ToroidalExactBDiagnosticObservationMode::SourceDesignedFluxes => {
                    ToroidalExactBObservationMode::SourceDesignedFluxes
                }
                _ => ToroidalExactBObservationMode::SurfaceFluxes,
            };
        }
    }
    config.write_outputs = false;
    config.output_dir = None;
    config
}

fn run_toroidal_exact_b_diagnostic_case(
    config: &ToroidalExactBRecoveryConfig,
    diagnostic_mode: ToroidalExactBDiagnosticObservationMode,
    sweep: &str,
    include_source_response: bool,
) -> Result<ToroidalExactBDiagnosticRow, String> {
    validate_exact_b_config(config)?;
    let started = Instant::now();
    let (topology, coords) = load_mesh(&config.base.mesh_path)?;
    let boundary = outer_boundary(&topology, &coords, config.base.geometry);
    let linear_system = build_toroidal_exact_b_linear_system(&topology, &coords, boundary, config)?;

    let nominal_source_full =
        assemble_toroidal_source(&topology, &coords, config.base.geometry, config.base.nu_air);
    let nominal_source = reduce_full_edge_vector(&linear_system.layout, &nominal_source_full)?;
    let source_modes = assemble_toroidal_source_modes(
        &topology,
        &coords,
        &linear_system.layout,
        config.base.geometry,
        config.base.nu_air,
    )?;
    let truth_eta = match config.reference_mode {
        ToroidalExactBReferenceMode::NominalDebug => [0.0; SOURCE_MODE_COUNT],
        ToroidalExactBReferenceMode::PerturbedSource => config.source_deltas,
    };
    let truth_source = source_with_sector_deltas(&nominal_source, &source_modes, &truth_eta)?;
    let reference_pde_variance = config
        .reference_pde_variance
        .unwrap_or(config.base.pde_variance);
    let linear_mean = solve_exact_b_reference_state(
        &linear_system,
        &nominal_source,
        config.reference_solve_mode,
        config.reference_solver_diagonal_shift,
        config.base.prior_precision,
        reference_pde_variance,
    )?;
    let exact_prior = build_exact_two_form_potential_prior(
        &topology,
        &coords,
        &linear_system.layout,
        linear_mean.clone(),
        ExactTwoFormPotentialPriorConfig {
            kappa: config.prior_kappa,
            tau: config.prior_tau,
            mass_inverse: config.potential_mass_inverse,
            diagonal_shift: 0.0,
        },
    )?;
    let reference_state = solve_exact_b_reference_state(
        &linear_system,
        &truth_source,
        config.reference_solve_mode,
        config.reference_solver_diagonal_shift,
        config.base.prior_precision,
        reference_pde_variance,
    )?;
    let reference_full = lift_vector_with_layout(
        &linear_system.layout,
        &FeecVector::from_vec(reference_state.clone()),
    )?;

    let maybe_observation_system = match diagnostic_mode {
        ToroidalExactBDiagnosticObservationMode::PdeOnly => None,
        ToroidalExactBDiagnosticObservationMode::CellMagneticComponents
        | ToroidalExactBDiagnosticObservationMode::FullFieldOracle => {
            Some(build_exact_b_observation_system(
                &topology,
                &coords,
                ToroidalExactBObservationMode::CellMagneticComponents,
                config.base.geometry,
                config.surface_flux_azimuth_count,
            )?)
        }
        ToroidalExactBDiagnosticObservationMode::SurfaceFluxes => {
            Some(build_exact_b_observation_system(
                &topology,
                &coords,
                ToroidalExactBObservationMode::SurfaceFluxes,
                config.base.geometry,
                config.surface_flux_azimuth_count,
            )?)
        }
        ToroidalExactBDiagnosticObservationMode::SourceDesignedFluxes => {
            Some(build_exact_b_observation_system(
                &topology,
                &coords,
                ToroidalExactBObservationMode::SourceDesignedFluxes,
                config.base.geometry,
                config.surface_flux_azimuth_count,
            )?)
        }
    };

    let (physical_measurements, training_indices, heldout_indices, reference_values) =
        if let Some(observation_system) = maybe_observation_system.as_ref() {
            let (training_indices, heldout_indices) = if config.observation_index_override.is_some()
            {
                resolve_exact_b_observation_indices(
                    config,
                    &linear_system,
                    &source_modes,
                    &observation_system.operator,
                )?
            } else {
                match diagnostic_mode {
                    ToroidalExactBDiagnosticObservationMode::FullFieldOracle => {
                        let dimension = observation_system.operator.nrows();
                        let heldout_count = config.heldout_count.min(dimension.saturating_sub(1));
                        let train_fraction = 1.0 - heldout_count as f64 / dimension.max(1) as f64;
                        split_exact_b_observation_indices(
                            dimension,
                            train_fraction,
                            heldout_count,
                            config.observation_seed,
                        )?
                    }
                    ToroidalExactBDiagnosticObservationMode::SourceDesignedFluxes => {
                        split_source_designed_flux_observation_indices(
                            &linear_system,
                            &source_modes,
                            &observation_system.operator,
                            config.base.prior_precision,
                            config.base.pde_variance,
                            config.observation_train_fraction,
                            config.heldout_count,
                        )?
                    }
                    _ => split_exact_b_observation_indices(
                        observation_system.operator.nrows(),
                        config.observation_train_fraction,
                        config.heldout_count,
                        config.observation_seed,
                    )?,
                }
            };
            let reference_values =
                apply_sparse_triplet(&observation_system.operator, reference_full.as_slice());
            let measurement = build_exact_b_measurement(
                &observation_system.operator,
                &reference_values,
                &training_indices,
                config.observation_noise_std * config.observation_noise_std,
                config.synthetic_observation_noise_seed,
            )?;
            (
                vec![measurement],
                training_indices,
                heldout_indices,
                reference_values,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

    let selected_rows = select_residual_rows(
        linear_system.residual_dimension(),
        &config.base.residual_selection,
    )?;
    let nominal_source_selected = select_vector_entries(&nominal_source, &selected_rows)?;
    let source_modes_selected = source_modes
        .iter()
        .map(|mode| select_vector_entries(mode, &selected_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let mut inference_system = select_linear_system_residual_rows(&linear_system, &selected_rows)?;
    for (bias, source) in inference_system
        .residual_bias
        .iter_mut()
        .zip(&nominal_source_selected)
    {
        *bias -= *source;
    }
    let source_operator = build_source_discrepancy_operator(&source_modes_selected)?;
    let source_prior_variance = config.source_prior_std * config.source_prior_std;
    let training_observation_operator = physical_measurements
        .first()
        .map(|measurement| measurement.operator.clone());
    let problem = LinearPdeUqProblem {
        state_prior: exact_prior.spec,
        system: inference_system.clone(),
        uncertain_inputs: vec![LinearUncertainInputSpec {
            name: SOURCE_DISCREPANCY_INPUT.to_string(),
            operator: source_operator.clone(),
            prior: zero_mean_diagonal_prior(SOURCE_MODE_COUNT, 1.0 / source_prior_variance),
            preference: RepresentationPreference::ForceLatent,
            collapsed_precision: None,
        }],
        physical_measurements,
        joint_measurements: Vec::new(),
        derived_quantities: Vec::new(),
        joint_derived_quantities: Vec::new(),
        pde_variance: Some(config.base.pde_variance),
        pde_precision: None,
    };
    let mut joint = build_linear_pde_joint_posterior_with_config(
        &problem,
        &LinearPdeUqSolverConfig {
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::ExactSolves,
                ..LinearPdeVarianceConfig::default()
            },
            precision_policy: LinearPdePrecisionPolicy::default(),
            ..LinearPdeUqSolverConfig::default()
        },
    )?;
    let centered_mean = joint.posterior.mean_vector().as_slice().to_vec();
    let state_dimension = joint.state_dimension;
    let mut reduced_posterior_mean = vec![0.0; state_dimension];
    for index in 0..state_dimension {
        reduced_posterior_mean[index] = centered_mean[index] + linear_mean[index];
    }
    let posterior_full = lift_vector_with_layout(
        &linear_system.layout,
        &FeecVector::from_vec(reduced_posterior_mean.clone()),
    )?;

    let source_selector = source_block_selector(joint.joint_dimension, state_dimension)?;
    let source_covariance = joint
        .posterior
        .exact_transformed_covariance(&source_selector)
        .map_err(|err| err.to_string())?;
    let mut eta_posterior_mean = [0.0; SOURCE_MODE_COUNT];
    let mut eta_posterior_variance = [0.0; SOURCE_MODE_COUNT];
    let mut eta_error = [0.0; SOURCE_MODE_COUNT];
    for mode_index in 0..SOURCE_MODE_COUNT {
        eta_posterior_mean[mode_index] = centered_mean[state_dimension + mode_index];
        eta_posterior_variance[mode_index] = source_covariance[(mode_index, mode_index)].max(0.0);
        eta_error[mode_index] = eta_posterior_mean[mode_index] - truth_eta[mode_index];
    }

    let predictions = maybe_observation_system
        .as_ref()
        .map(|observation_system| {
            apply_sparse_triplet(&observation_system.operator, posterior_full.as_slice())
        })
        .unwrap_or_default();
    let train_rmse = indexed_rmse(&predictions, &reference_values, &training_indices);
    let heldout_rmse = indexed_rmse(&predictions, &reference_values, &heldout_indices);
    let cell_b_operator = build_full_magnetic_flux_density_triplet_operator(&topology, &coords)?;
    let b_relative_error = relative_error(
        &apply_sparse_triplet(&cell_b_operator, posterior_full.as_slice()),
        &apply_sparse_triplet(&cell_b_operator, reference_full.as_slice()),
    );
    let final_residual_norm = exact_b_linear_residual_norm(
        &inference_system,
        &source_operator,
        &reduced_posterior_mean,
        &eta_posterior_mean,
    )?;
    let source_response = if include_source_response {
        exact_b_source_response_summary(
            &linear_system,
            &source_modes,
            training_observation_operator.as_ref(),
            config.base.prior_precision,
            config.base.pde_variance,
            config.observation_noise_std,
        )?
    } else {
        ToroidalExactBSourceResponseSummary {
            condition: f64::NAN,
            singular_values: [f64::NAN; SOURCE_MODE_COUNT],
            column_norms: [f64::NAN; SOURCE_MODE_COUNT],
            snr_min: f64::NAN,
            snr_max: f64::NAN,
        }
    };
    let posterior_factor_nnz = joint
        .posterior
        .precision_factor()
        .map(|factor| factor.nnz())
        .unwrap_or(0);

    Ok(ToroidalExactBDiagnosticRow {
        sweep: sweep.to_string(),
        observation_mode: diagnostic_mode,
        pde_variance: config.base.pde_variance,
        prior_tau: config.prior_tau,
        source_prior_std: config.source_prior_std,
        observation_noise_std: config.observation_noise_std,
        training_rows: training_indices.len(),
        heldout_rows: heldout_indices.len(),
        train_rmse,
        heldout_rmse,
        b_relative_error,
        final_residual_norm,
        source_response_condition: source_response.condition,
        source_response_singular_values: source_response.singular_values,
        source_response_column_norms: source_response.column_norms,
        source_response_snr_min: source_response.snr_min,
        source_response_snr_max: source_response.snr_max,
        eta_truth: truth_eta,
        eta_posterior_mean,
        eta_posterior_variance,
        eta_error,
        posterior_factor_nnz,
        runtime_seconds: started.elapsed().as_secs_f64(),
    })
}

pub fn run_toroidal_residual_budget_experiment(
    config: &ToroidalResidualBudgetConfig,
) -> Result<ToroidalResidualBudgetReport, String> {
    let mut reference_config = config.base.clone();
    reference_config.prior_mode = ToroidalPriorMode::LinearProxyMaternAlpha2;
    reference_config.residual_selection = ToroidalResidualSelection::Full;
    reference_config.include_nonlinear_residual = true;
    reference_config.linear_measurements.clear();
    reference_config.extra_derived_quantities.clear();
    reference_config.include_cell_b_variance = false;
    reference_config.compute_harmonic_diagnostics = false;
    reference_config.variance = sweep_variance_config(config.seed);
    let reference = run_nonlinear_toroidal_inductor(&reference_config)?;

    let (topology, coords, reference_model) = build_toroidal_model_for_diagnostics(&config.base)?;
    let reference_sensors = reference
        .sensor_reports
        .iter()
        .map(|report| report.nonlinear_value)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for prior_mode in &config.prior_modes {
        for stride in &config.residual_strides {
            let selection = if *stride == 1 {
                ToroidalResidualSelection::Full
            } else if config.shuffled {
                ToroidalResidualSelection::ShuffledStride {
                    step: *stride,
                    seed: config.seed,
                }
            } else {
                ToroidalResidualSelection::Stride { step: *stride }
            };
            let mut run_config = config.base.clone();
            run_config.prior_mode = *prior_mode;
            run_config.residual_selection = selection.clone();
            run_config.include_nonlinear_residual = true;
            run_config.linear_measurements.clear();
            run_config.extra_derived_quantities.clear();
            run_config.include_cell_b_variance = false;
            run_config.compute_harmonic_diagnostics = false;
            run_config.variance = sweep_variance_config(config.seed);
            let report = run_nonlinear_toroidal_inductor(&run_config)?;
            let sensors = report
                .sensor_reports
                .iter()
                .map(|sensor| sensor.nonlinear_value)
                .collect::<Vec<_>>();
            rows.push(ToroidalResidualBudgetRow {
                prior_mode: *prior_mode,
                selection_label: selection.label(),
                residual_rows_used: report.residual_rows_used,
                residual_rows_total: report.residual_rows_total,
                iterations: report.iterations,
                final_residual_norm: report.final_residual_norm,
                linear_solve_iteration_sum: report.linear_solve_iteration_sum,
                linear_solve_residual_max: report.linear_solve_residual_max,
                posterior_factor_nnz: report.posterior_factor_nnz,
                final_factorization_seconds: report.final_factorization_seconds,
                map_relative_error_to_reference: relative_error(
                    &report.result.map,
                    &reference.result.map,
                ),
                cell_b_relative_error_to_reference: cell_b_relative_error(
                    &topology,
                    &coords,
                    &reference_model,
                    &report.result.map,
                    &reference.result.map,
                )?,
                flux_sensor_rmse_to_reference: rmse(&sensors, &reference_sensors),
                posterior_factorizes: report.posterior_factorizes,
            });
        }
    }
    Ok(ToroidalResidualBudgetReport { reference, rows })
}

pub fn run_toroidal_sensor_regression_experiment(
    config: &ToroidalSensorRegressionConfig,
) -> Result<ToroidalSensorRegressionReport, String> {
    if config.azimuth_count == 0 {
        return Err("sensor regression azimuth_count must be at least one".to_string());
    }
    if config.residual_stride == 0 {
        return Err("sensor regression residual_stride must be at least one".to_string());
    }
    if !config.sensor_variance.is_finite() || config.sensor_variance <= 0.0 {
        return Err("sensor regression sensor_variance must be finite and positive".to_string());
    }
    if !config.synthetic_noise_std.is_finite() || config.synthetic_noise_std < 0.0 {
        return Err(
            "sensor regression synthetic_noise_std must be finite and nonnegative".to_string(),
        );
    }

    let mut reference_config = config.base.clone();
    reference_config.prior_mode = ToroidalPriorMode::LinearProxyMaternAlpha2;
    reference_config.residual_selection = ToroidalResidualSelection::Full;
    reference_config.include_nonlinear_residual = true;
    reference_config.linear_measurements.clear();
    reference_config.extra_derived_quantities.clear();
    reference_config.include_cell_b_variance = false;
    reference_config.compute_harmonic_diagnostics = false;
    reference_config.variance = sweep_variance_config(config.seed);
    let reference = run_nonlinear_toroidal_inductor(&reference_config)?;

    let (topology, coords, model) = build_toroidal_model_for_diagnostics(&config.base)?;
    let sensor_specs =
        default_toroidal_flux_sensor_specs(config.base.geometry, config.azimuth_count);
    let sensors = build_reduced_flux_sensors(&topology, &coords, &model, &sensor_specs)?;
    let reference_values = apply_reduced_flux_sensors(&sensors, &reference.result.map)?;
    let mut observed_values = reference_values.clone();
    if config.synthetic_noise_std > 0.0 {
        let normal = Normal::new(0.0, config.synthetic_noise_std)
            .map_err(|err| format!("invalid synthetic sensor noise std: {err}"))?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        for value in &mut observed_values {
            *value += normal.sample(&mut rng);
        }
    }
    let mut shuffled_indices = (0..sensors.len()).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(config.seed);
    shuffled_indices.shuffle(&mut rng);

    let all_derived = sensors
        .iter()
        .map(|sensor| {
            Ok(LinearPdeDerivedQuantitySpec {
                name: sensor.spec.name.clone(),
                operator: triplet_to_sparse_row_operator(&sensor.operator)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut rows = Vec::new();
    for prior_mode in &config.prior_modes {
        for train_count in &config.train_counts {
            let train_count = (*train_count).min(sensors.len());
            let train = shuffled_indices[..train_count].to_vec();
            let holdout = shuffled_indices[train_count..].to_vec();
            let measurement = build_sensor_measurement(
                &sensors,
                &observed_values,
                &train,
                config.sensor_variance,
            )?;
            for variant in &config.variants {
                let mut run_config = config.base.clone();
                run_config.prior_mode = *prior_mode;
                run_config.include_cell_b_variance = false;
                run_config.compute_harmonic_diagnostics = false;
                run_config.variance = sweep_variance_config(config.seed);
                run_config.extra_derived_quantities = all_derived.clone();
                run_config.linear_measurements.clear();
                run_config.residual_selection = ToroidalResidualSelection::ShuffledStride {
                    step: config.residual_stride,
                    seed: config.seed,
                };
                run_config.include_nonlinear_residual = matches!(
                    *variant,
                    ToroidalSensorRegressionVariant::ResidualBudgetOnly
                        | ToroidalSensorRegressionVariant::SensorsPlusResidualBudget
                );
                if matches!(
                    *variant,
                    ToroidalSensorRegressionVariant::SensorsOnly
                        | ToroidalSensorRegressionVariant::SensorsPlusResidualBudget
                ) {
                    run_config.linear_measurements = vec![measurement.clone()];
                }
                let report = run_nonlinear_toroidal_inductor(&run_config)?;
                let predictions = apply_reduced_flux_sensors(&sensors, &report.result.map)?;
                let variances =
                    sensor_predictive_variances(&sensors, &report, config.sensor_variance)?;
                rows.push(ToroidalSensorRegressionRow {
                    prior_mode: *prior_mode,
                    variant: *variant,
                    train_count,
                    holdout_count: holdout.len(),
                    train_rmse: indexed_rmse(&predictions, &reference_values, &train),
                    holdout_rmse: indexed_rmse(&predictions, &reference_values, &holdout),
                    mean_abs_z: indexed_mean_abs_z(
                        &predictions,
                        &reference_values,
                        &variances,
                        &holdout,
                    ),
                    coverage_2sigma: indexed_coverage(
                        &predictions,
                        &reference_values,
                        &variances,
                        &holdout,
                        2.0,
                    ),
                    final_residual_norm: report.final_residual_norm,
                    linear_solve_iteration_sum: report.linear_solve_iteration_sum,
                    linear_solve_residual_max: report.linear_solve_residual_max,
                    posterior_factor_nnz: report.posterior_factor_nnz,
                    final_factorization_seconds: report.final_factorization_seconds,
                    posterior_factorizes: report.posterior_factorizes,
                });
            }
        }
    }
    Ok(ToroidalSensorRegressionReport {
        reference,
        sensors: sensor_specs,
        rows,
    })
}

pub fn run_toroidal_weiland_comparison_experiment(
    config: &ToroidalWeilandComparisonConfig,
) -> Result<ToroidalWeilandComparisonReport, String> {
    run_toroidal_weiland_comparison_experiment_with_row_callback(config, |_| {})
}

pub fn run_toroidal_weiland_comparison_experiment_with_row_callback(
    config: &ToroidalWeilandComparisonConfig,
    mut on_row: impl FnMut(&ToroidalWeilandComparisonRow),
) -> Result<ToroidalWeilandComparisonReport, String> {
    if config.row_selection_repetitions == 0 {
        return Err(
            "Weiland comparison requires at least one row-selection repetition".to_string(),
        );
    }
    if config.sensor_azimuth_count == 0 {
        return Err("Weiland comparison sensor_azimuth_count must be at least one".to_string());
    }
    if config.prior_modes.is_empty() {
        return Err("Weiland comparison requires at least one prior mode".to_string());
    }

    let (topology, coords, model) = build_toroidal_model_for_diagnostics(&config.base)?;
    let diameter = mesh_bounding_box_diameter(&coords).max(1e-12);
    let residual_rows_total =
        toroidal_observation_residual_dimension(&model, config.base.pde_observation_mode);
    let residual_counts =
        residual_counts_from_fractions(residual_rows_total, &config.residual_fractions)?;
    let kappa_candidates = weiland_kappa_candidates(
        config,
        diameter,
        default_reduced_linear_proxy_matern_kappa(&coords),
    )?;
    let sensor_specs =
        default_toroidal_flux_sensor_specs(config.base.geometry, config.sensor_azimuth_count);
    let sensors = build_reduced_flux_sensors(&topology, &coords, &model, &sensor_specs)?;
    let sensor_derived = sensors
        .iter()
        .map(|sensor| {
            Ok(LinearPdeDerivedQuantitySpec {
                name: sensor.spec.name.clone(),
                operator: triplet_to_sparse_row_operator(&sensor.operator)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut references = Vec::new();
    let mut rows = Vec::new();
    for requested_kappa in kappa_candidates {
        let mut reference_config = config.base.clone();
        configure_weiland_base_run(
            &mut reference_config,
            requested_kappa,
            config.seed,
            &sensor_derived,
        );
        reference_config.prior_mode = ToroidalPriorMode::LinearProxyMaternAlpha2;
        reference_config.pde_observation_mode = ToroidalPdeObservationMode::WeakGalerkinRows;
        reference_config.residual_selection = ToroidalResidualSelection::Full;
        reference_config.include_nonlinear_residual = true;

        let reference_result = run_nonlinear_toroidal_inductor(&reference_config);
        let reference = match reference_result {
            Ok(reference) => {
                references.push(ToroidalWeilandReferenceRow {
                    requested_kappa,
                    actual_kappa: reference.prior_kappa,
                    kappa_times_diameter: reference.prior_kappa * diameter,
                    success: true,
                    failure_reason: None,
                    iterations: reference.iterations,
                    final_residual_norm: reference.final_residual_norm,
                    posterior_factor_nnz: reference.posterior_factor_nnz,
                    final_factorization_seconds: reference.final_factorization_seconds,
                });
                reference
            }
            Err(err) => {
                references.push(ToroidalWeilandReferenceRow {
                    requested_kappa,
                    actual_kappa: f64::NAN,
                    kappa_times_diameter: requested_kappa * diameter,
                    success: false,
                    failure_reason: Some(err),
                    iterations: 0,
                    final_residual_norm: f64::NAN,
                    posterior_factor_nnz: 0,
                    final_factorization_seconds: f64::NAN,
                });
                continue;
            }
        };

        let reference_sensors = apply_reduced_flux_sensors(&sensors, &reference.result.map)?;
        for prior_mode in &config.prior_modes {
            for residual_rows_requested in &residual_counts {
                let repetitions = if *residual_rows_requested == 0
                    || *residual_rows_requested == residual_rows_total
                {
                    1
                } else {
                    config.row_selection_repetitions
                };
                for repetition in 0..repetitions {
                    let seed = config
                        .seed
                        .wrapping_add(repetition as u64)
                        .wrapping_add((*residual_rows_requested as u64) << 16)
                        .wrapping_add(match prior_mode {
                            ToroidalPriorMode::WeakDiagonal => 0xD1A9,
                            ToroidalPriorMode::LinearProxyMaternAlpha2 => 0xA2A2,
                        });
                    let selection = if *residual_rows_requested == residual_rows_total {
                        ToroidalResidualSelection::Full
                    } else {
                        ToroidalResidualSelection::ShuffledCount {
                            count: *residual_rows_requested,
                            seed,
                        }
                    };
                    let mut run_config = config.base.clone();
                    configure_weiland_base_run(
                        &mut run_config,
                        requested_kappa,
                        seed,
                        &sensor_derived,
                    );
                    run_config.prior_mode = *prior_mode;
                    run_config.residual_selection = selection.clone();
                    run_config.include_nonlinear_residual = *residual_rows_requested > 0;

                    let row = match run_nonlinear_toroidal_inductor(&run_config) {
                        Ok(report) => build_weiland_success_row(
                            requested_kappa,
                            diameter,
                            *residual_rows_requested,
                            seed,
                            selection.label(),
                            &topology,
                            &coords,
                            &model,
                            &sensors,
                            &reference,
                            &reference_sensors,
                            report,
                        ),
                        Err(err) => build_weiland_failure_row(
                            requested_kappa,
                            diameter,
                            *prior_mode,
                            *residual_rows_requested,
                            residual_rows_total,
                            *residual_rows_requested > 0,
                            seed,
                            selection.label(),
                            err,
                        ),
                    }?;
                    on_row(&row);
                    rows.push(row);
                }
            }
        }
    }

    let summaries = summarize_weiland_comparison_rows(&rows);
    Ok(ToroidalWeilandComparisonReport {
        mesh_path: config.base.mesh_path.clone(),
        bounding_box_diameter: diameter,
        active_dofs: model.reduced_dimension(),
        residual_rows_total,
        sensor_count: sensors.len(),
        references,
        rows,
        summaries,
    })
}

fn configure_weiland_base_run(
    run_config: &mut NonlinearToroidalConfig,
    kappa: f64,
    seed: u64,
    sensor_derived: &[LinearPdeDerivedQuantitySpec],
) {
    run_config.linear_proxy_kappa = Some(kappa);
    run_config.linear_proxy_allow_kappa_fallback = false;
    run_config.write_outputs = false;
    run_config.include_cell_b_variance = false;
    run_config.compute_harmonic_diagnostics = false;
    run_config.linear_measurements.clear();
    run_config.extra_derived_quantities = sensor_derived.to_vec();
    run_config.variance = sweep_variance_config(seed);
}

fn residual_counts_from_fractions(
    residual_rows_total: usize,
    fractions: &[f64],
) -> Result<Vec<usize>, String> {
    if residual_rows_total == 0 {
        return Err("Weiland comparison requires a nonempty residual".to_string());
    }
    if fractions.is_empty() {
        return Err("Weiland comparison requires at least one residual fraction".to_string());
    }
    let mut counts = Vec::with_capacity(fractions.len());
    for fraction in fractions {
        if !fraction.is_finite() || *fraction < 0.0 || *fraction > 1.0 {
            return Err(format!(
                "residual fraction must be finite and in [0, 1], got {fraction:.6e}"
            ));
        }
        let count = if *fraction == 0.0 {
            0
        } else {
            ((*fraction * residual_rows_total as f64).ceil() as usize).clamp(1, residual_rows_total)
        };
        if !counts.contains(&count) {
            counts.push(count);
        }
    }
    counts.sort_unstable();
    Ok(counts)
}

fn weiland_kappa_candidates(
    config: &ToroidalWeilandComparisonConfig,
    diameter: f64,
    default_kappa: f64,
) -> Result<Vec<f64>, String> {
    let mut kappas = Vec::new();
    if !config.explicit_kappas.is_empty() {
        kappas.extend(config.explicit_kappas.iter().copied());
    } else if config.include_default_kappa {
        kappas.push(default_kappa);
    }
    if config.explicit_kappas.is_empty() {
        for scale in &config.kappa_diameter_scales {
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(format!(
                    "kappa diameter scale must be finite and positive, got {scale:.6e}"
                ));
            }
            kappas.push(*scale / diameter);
        }
    }
    if kappas.is_empty() {
        return Err("Weiland comparison requires at least one kappa candidate".to_string());
    }
    let mut unique = Vec::new();
    for kappa in kappas {
        if !kappa.is_finite() || kappa <= 0.0 {
            return Err(format!(
                "kappa candidate must be finite and positive, got {kappa:.6e}"
            ));
        }
        if !unique
            .iter()
            .any(|existing: &f64| (*existing - kappa).abs() <= 1e-12 * existing.abs().max(1.0))
        {
            unique.push(kappa);
        }
    }
    Ok(unique)
}

// Success rows preserve solver selection, mesh, sensor, seed, and reference fields
// for publication provenance. Toroidal smoke profiles exercise this constructor.
#[allow(clippy::too_many_arguments)]
fn build_weiland_success_row(
    reference_kappa: f64,
    diameter: f64,
    residual_rows_requested: usize,
    seed: u64,
    selection_label: String,
    topology: &Complex,
    coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
    sensors: &[ToroidalReducedFluxSensor],
    reference: &NonlinearToroidalReport,
    reference_sensors: &[f64],
    report: NonlinearToroidalReport,
) -> Result<ToroidalWeilandComparisonRow, String> {
    let predictions = apply_reduced_flux_sensors(sensors, &report.result.map)?;
    let variances = sensor_predictive_variances(sensors, &report, 0.0)?;
    let sensor_indices = (0..sensors.len()).collect::<Vec<_>>();
    let (sensor_variance_min, sensor_variance_max) = finite_min_max(variances.iter().copied());
    Ok(ToroidalWeilandComparisonRow {
        reference_kappa,
        prior_kappa: report.prior_kappa,
        kappa_times_diameter: report.prior_kappa * diameter,
        prior_mode: report.prior_mode,
        nonlinear_residual_likelihood: report.nonlinear_residual_likelihood,
        selection_label,
        seed,
        residual_rows_requested,
        residual_rows_used: report.residual_rows_used,
        residual_rows_total: report.residual_rows_total,
        residual_fraction: report.residual_rows_used as f64 / report.residual_rows_total as f64,
        success: true,
        failure_reason: None,
        iterations: report.iterations,
        damping_steps: report
            .result
            .history
            .iter()
            .filter(|entry| entry.alpha < 1.0)
            .count(),
        final_residual_norm: report.final_residual_norm,
        linear_solve_iteration_sum: report.linear_solve_iteration_sum,
        linear_solve_residual_max: report.linear_solve_residual_max,
        posterior_factor_nnz: report.posterior_factor_nnz,
        final_factorization_seconds: report.final_factorization_seconds,
        map_relative_error_to_reference: relative_error(&report.result.map, &reference.result.map),
        cell_b_relative_error_to_reference: cell_b_relative_error(
            topology,
            coords,
            model,
            &report.result.map,
            &reference.result.map,
        )?,
        flux_sensor_rmse_to_reference: rmse(&predictions, reference_sensors),
        sensor_variance_min,
        sensor_variance_max,
        sensor_mean_abs_z: indexed_mean_abs_z(
            &predictions,
            reference_sensors,
            &variances,
            &sensor_indices,
        ),
        sensor_coverage_2sigma: indexed_coverage(
            &predictions,
            reference_sensors,
            &variances,
            &sensor_indices,
            2.0,
        ),
        posterior_factorizes: report.posterior_factorizes,
    })
}

// Failure and success rows share a provenance schema, keeping every publication
// run auditable.
#[allow(clippy::too_many_arguments)]
fn build_weiland_failure_row(
    reference_kappa: f64,
    diameter: f64,
    prior_mode: ToroidalPriorMode,
    residual_rows_requested: usize,
    residual_rows_total: usize,
    nonlinear_residual_likelihood: bool,
    seed: u64,
    selection_label: String,
    failure_reason: String,
) -> Result<ToroidalWeilandComparisonRow, String> {
    Ok(ToroidalWeilandComparisonRow {
        reference_kappa,
        prior_kappa: match prior_mode {
            ToroidalPriorMode::WeakDiagonal => 0.0,
            ToroidalPriorMode::LinearProxyMaternAlpha2 => reference_kappa,
        },
        kappa_times_diameter: match prior_mode {
            ToroidalPriorMode::WeakDiagonal => 0.0,
            ToroidalPriorMode::LinearProxyMaternAlpha2 => reference_kappa * diameter,
        },
        prior_mode,
        nonlinear_residual_likelihood,
        selection_label,
        seed,
        residual_rows_requested,
        residual_rows_used: residual_rows_requested,
        residual_rows_total,
        residual_fraction: residual_rows_requested as f64 / residual_rows_total as f64,
        success: false,
        failure_reason: Some(failure_reason),
        iterations: 0,
        damping_steps: 0,
        final_residual_norm: f64::NAN,
        linear_solve_iteration_sum: 0,
        linear_solve_residual_max: f64::NAN,
        posterior_factor_nnz: 0,
        final_factorization_seconds: f64::NAN,
        map_relative_error_to_reference: f64::NAN,
        cell_b_relative_error_to_reference: f64::NAN,
        flux_sensor_rmse_to_reference: f64::NAN,
        sensor_variance_min: f64::NAN,
        sensor_variance_max: f64::NAN,
        sensor_mean_abs_z: f64::NAN,
        sensor_coverage_2sigma: f64::NAN,
        posterior_factorizes: false,
    })
}

fn summarize_weiland_comparison_rows(
    rows: &[ToroidalWeilandComparisonRow],
) -> Vec<ToroidalWeilandComparisonSummaryRow> {
    let mut keys = Vec::<(f64, ToroidalPriorMode, bool, usize, usize)>::new();
    for row in rows {
        let key = (
            row.reference_kappa,
            row.prior_mode,
            row.nonlinear_residual_likelihood,
            row.residual_rows_requested,
            row.residual_rows_total,
        );
        if !keys.iter().any(|existing| {
            existing.0 == key.0
                && existing.1 == key.1
                && existing.2 == key.2
                && existing.3 == key.3
                && existing.4 == key.4
        }) {
            keys.push(key);
        }
    }
    keys.into_iter()
        .map(
            |(
                reference_kappa,
                prior_mode,
                nonlinear_residual_likelihood,
                residual_rows_requested,
                residual_rows_total,
            )| {
                let group = rows
                    .iter()
                    .filter(|row| {
                        row.reference_kappa == reference_kappa
                            && row.prior_mode == prior_mode
                            && row.nonlinear_residual_likelihood == nonlinear_residual_likelihood
                            && row.residual_rows_requested == residual_rows_requested
                            && row.residual_rows_total == residual_rows_total
                    })
                    .collect::<Vec<_>>();
                let success = group
                    .iter()
                    .copied()
                    .filter(|row| row.success)
                    .collect::<Vec<_>>();
                let residuals = success
                    .iter()
                    .map(|row| row.final_residual_norm)
                    .collect::<Vec<_>>();
                let cell_b = success
                    .iter()
                    .map(|row| row.cell_b_relative_error_to_reference)
                    .collect::<Vec<_>>();
                let sensors = success
                    .iter()
                    .map(|row| row.flux_sensor_rmse_to_reference)
                    .collect::<Vec<_>>();
                let map = success
                    .iter()
                    .map(|row| row.map_relative_error_to_reference)
                    .collect::<Vec<_>>();
                let mean_abs_z = success
                    .iter()
                    .map(|row| row.sensor_mean_abs_z)
                    .collect::<Vec<_>>();
                let coverage = success
                    .iter()
                    .map(|row| row.sensor_coverage_2sigma)
                    .collect::<Vec<_>>();
                let (final_residual_mean, final_residual_std) = mean_std(&residuals);
                let (cell_b_relative_error_mean, cell_b_relative_error_std) = mean_std(&cell_b);
                let (flux_sensor_rmse_mean, flux_sensor_rmse_std) = mean_std(&sensors);
                let (map_relative_error_mean, map_relative_error_std) = mean_std(&map);
                let (sensor_mean_abs_z_mean, _) = mean_std(&mean_abs_z);
                let (sensor_coverage_2sigma_mean, _) = mean_std(&coverage);
                ToroidalWeilandComparisonSummaryRow {
                    reference_kappa,
                    prior_mode,
                    nonlinear_residual_likelihood,
                    residual_rows_requested,
                    residual_rows_total,
                    residual_fraction: residual_rows_requested as f64 / residual_rows_total as f64,
                    success_count: success.len(),
                    failure_count: group.len() - success.len(),
                    final_residual_mean,
                    final_residual_std,
                    cell_b_relative_error_mean,
                    cell_b_relative_error_std,
                    flux_sensor_rmse_mean,
                    flux_sensor_rmse_std,
                    map_relative_error_mean,
                    map_relative_error_std,
                    sensor_mean_abs_z_mean,
                    sensor_coverage_2sigma_mean,
                }
            },
        )
        .collect()
}

pub fn load_mesh(path: &Path) -> Result<(Complex, MeshCoords), String> {
    let resolved = resolve_workspace_path(path);
    let mesh_bytes = fs::read(&resolved)
        .map_err(|err| format!("failed to read mesh `{}`: {err}", resolved.display()))?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "toroidal inductor requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    Ok((topology, coords))
}

fn resolve_workspace_path(path: &Path) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path.to_path_buf();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

pub fn outer_boundary(
    topology: &Complex,
    coords: &MeshCoords,
    geometry: ToroidalInductorGeometry,
) -> BoundarySpec {
    let outer_edges = sorted_boundary_dofs(topology, coords, 1, |point| {
        outer_boundary_predicate(point, geometry)
    });
    let outer_vertices = sorted_boundary_dofs(topology, coords, 0, |point| {
        outer_boundary_predicate(point, geometry)
    });
    BoundarySpec::default()
        .with_state_region(BoundaryRegionSpec::new(
            "outer_tangential_edges",
            outer_edges.clone(),
            vec![0.0; outer_edges.len()],
            BoundaryTreatment::HardEssential,
        ))
        .with_auxiliary_region(BoundaryRegionSpec::new(
            "outer_coulomb_vertices",
            outer_vertices.clone(),
            vec![0.0; outer_vertices.len()],
            BoundaryTreatment::HardEssential,
        ))
}

fn toroidal_essential_boundary(
    topology: &Complex,
    boundary: &BoundarySpec,
    include_auxiliary: bool,
) -> Result<EssentialBoundarySpec, String> {
    let adapted = adapt_boundary_spec(boundary, topology.nsimplices(1), topology.nsimplices(0))?;
    if !adapted.soft_state_measurements.is_empty() {
        return Err(
            "toroidal deterministic assembly requires soft boundaries to be supplied as inference measurements"
                .to_string(),
        );
    }
    Ok(EssentialBoundarySpec {
        state: adapted.essential.state,
        auxiliary: if include_auxiliary {
            adapted.essential.auxiliary
        } else {
            Vec::new()
        },
    })
}

pub fn outer_boundary_predicate(point: CoordRef<'_>, geometry: ToroidalInductorGeometry) -> bool {
    let d = (geometry.box_half_length - point[0].abs())
        .min(geometry.box_half_length - point[1].abs())
        .min(geometry.box_half_length - point[2].abs());
    d < 10.0 * geometry.target_air_cell_size
}

pub fn toroidal_radius(point: CoordRef<'_>, geometry: ToroidalInductorGeometry) -> f64 {
    let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
    ((rho - geometry.major_radius).powi(2) + point[2] * point[2]).sqrt()
}

fn toroidal_radius_from_point(point: [f64; 3], geometry: ToroidalInductorGeometry) -> f64 {
    let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
    ((rho - geometry.major_radius).powi(2) + point[2] * point[2]).sqrt()
}

fn coord_vec_to_point3(coord: FeecVector) -> [f64; 3] {
    let mut point = [0.0, 0.0, 0.0];
    for index in 0..coord.len().min(3) {
        point[index] = coord[index];
    }
    point
}

pub fn toroidal_direction(point: CoordRef<'_>) -> [f64; 3] {
    let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
    if rho < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [-point[1] / rho, point[0] / rho, 0.0]
    }
}

pub fn coil_current_density(
    geometry: ToroidalInductorGeometry,
    mu0: f64,
    sector: Option<usize>,
) -> DiffFormClosure {
    let sigma = 0.18;
    let eps = 0.03;
    DiffFormClosure::one_form(
        move |point| {
            let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
            if rho < 1e-12 {
                return FeecVector::from_column_slice(&[0.0, 0.0, 0.0]);
            }
            let angle = point[1].atan2(point[0]);
            if let Some(sector_index) = sector {
                let width = 2.0 * PI / SOURCE_MODE_COUNT as f64;
                let wrapped = (angle + PI).rem_euclid(2.0 * PI);
                let candidate = ((wrapped / width).floor() as usize).min(SOURCE_MODE_COUNT - 1);
                if candidate != sector_index {
                    return FeecVector::from_column_slice(&[0.0, 0.0, 0.0]);
                }
            }

            let s = toroidal_radius(point, geometry);
            let smoothstep = |t: f64| t * t * (3.0 - 2.0 * t);
            let inner = geometry.core_minor_radius + eps;
            let outer = geometry.coil_minor_radius - eps;
            let tin = ((s - inner) / eps).clamp(0.0, 1.0);
            let tout = ((outer - s) / eps).clamp(0.0, 1.0);
            let cutoff = smoothstep(tin) * smoothstep(tout);
            let s0 = 0.5 * (geometry.core_minor_radius + geometry.coil_minor_radius);
            let gauss = (-((s - s0) * (s - s0)) / (sigma * sigma)).exp();
            let amplitude = mu0 * gauss * cutoff;
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

pub fn assemble_toroidal_source(
    topology: &Complex,
    coords: &MeshCoords,
    geometry: ToroidalInductorGeometry,
    nu_air: f64,
) -> FeecVector {
    assemble_toroidal_source_with_sector(topology, coords, geometry, nu_air, None)
}

pub fn assemble_toroidal_source_sector(
    topology: &Complex,
    coords: &MeshCoords,
    geometry: ToroidalInductorGeometry,
    nu_air: f64,
    sector: usize,
) -> FeecVector {
    assemble_toroidal_source_with_sector(topology, coords, geometry, nu_air, Some(sector))
}

fn assemble_toroidal_source_with_sector(
    topology: &Complex,
    coords: &MeshCoords,
    geometry: ToroidalInductorGeometry,
    nu_air: f64,
    sector: Option<usize>,
) -> FeecVector {
    let metric = coords.to_edge_lengths(topology);
    let mu0 = 1.0 / nu_air;
    let weight = InnerProductWeightClosure::new(move |_| nu_air);
    assemble_galvec(
        topology,
        &metric,
        SourceElVec::new_weighted(
            &coil_current_density(geometry, mu0, sector),
            coords,
            None,
            &weight,
        ),
    )
}

fn build_toroidal_model_for_diagnostics(
    config: &NonlinearToroidalConfig,
) -> Result<(Complex, MeshCoords, ReducedVectorPotentialMagnetostatic3d), String> {
    let (topology, coords) = load_mesh(&config.mesh_path)?;
    let boundary = outer_boundary(&topology, &coords, config.geometry);
    let material = ToroidalReluctivityLaw::new(
        config.geometry.major_radius,
        config.geometry.core_minor_radius,
        config.nu_air,
        config.nu_core0,
        config.beta_core,
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(
            material,
            toroidal_essential_boundary(&topology, &boundary, false)?,
        ),
    )?;
    let source_full = assemble_toroidal_source(&topology, &coords, config.geometry, config.nu_air);
    let source_reduced = reduce_full_edge_vector(source_free.layout(), &source_full)?;
    Ok((topology, coords, source_free.with_source(source_reduced)?))
}

pub fn default_toroidal_flux_sensor_specs(
    geometry: ToroidalInductorGeometry,
    azimuth_count: usize,
) -> Vec<ToroidalFluxSensorSpec> {
    let mut sensors = Vec::new();
    let (patch_radius, normal_half_width) = if azimuth_count > 4 {
        (0.65, 0.28)
    } else {
        (0.45, 0.18)
    };
    for index in 0..azimuth_count {
        let theta = 2.0 * PI * index as f64 / azimuth_count.max(1) as f64;
        let radial = [theta.cos(), theta.sin(), 0.0];
        let normal = [-theta.sin(), theta.cos(), 0.0];
        for (label, radial_offset, z_offset) in [
            ("inner", -1.05, 0.0),
            ("top", 0.0, 1.05),
            ("outer", 1.05, 0.0),
        ] {
            let radius = geometry.major_radius + radial_offset;
            sensors.push(ToroidalFluxSensorSpec {
                name: format!("flux_{label}_{index:02}"),
                center: [radius * radial[0], radius * radial[1], z_offset],
                normal,
                patch_radius,
                normal_half_width,
            });
        }
    }
    sensors
}

const EXACT_B_FIELD_RECOVERY_AZIMUTH_COUNT: usize = 16;
const EXACT_B_FLUX_SURFACE_COUNT: usize = 3;
const EXACT_B_FIELD_RECOVERY_HELDOUT_AZIMUTHS: [usize; 4] = [1, 5, 9, 13];
const EXACT_B_FIELD_RECOVERY_TRAINING_AZIMUTHS: [usize; 12] =
    [0, 8, 4, 12, 2, 10, 6, 14, 3, 11, 7, 15];

pub fn toroidal_exact_b_field_recovery_observation_indices(
    azimuth_count: usize,
    training_count: usize,
) -> Result<ToroidalExactBObservationIndexOverride, String> {
    if azimuth_count != EXACT_B_FIELD_RECOVERY_AZIMUTH_COUNT {
        return Err(format!(
            "exact-B field-recovery design expects {EXACT_B_FIELD_RECOVERY_AZIMUTH_COUNT} azimuths, got {azimuth_count}"
        ));
    }
    let training_order = toroidal_exact_b_field_recovery_training_order(azimuth_count)?;
    if training_count == 0 || training_count > training_order.len() {
        return Err(format!(
            "exact-B field-recovery training count {training_count} must be in 1..={}",
            training_order.len()
        ));
    }
    let heldout_indices = EXACT_B_FIELD_RECOVERY_HELDOUT_AZIMUTHS
        .iter()
        .flat_map(|azimuth| {
            (0..EXACT_B_FLUX_SURFACE_COUNT)
                .map(move |surface| toroidal_exact_b_flux_row_index(*azimuth, surface))
        })
        .collect::<Vec<_>>();
    let override_indices = ToroidalExactBObservationIndexOverride {
        training_indices: training_order[..training_count].to_vec(),
        heldout_indices,
    };
    validate_exact_b_observation_index_override(
        &override_indices,
        azimuth_count * EXACT_B_FLUX_SURFACE_COUNT,
    )?;
    Ok(override_indices)
}

fn toroidal_exact_b_field_recovery_training_order(
    azimuth_count: usize,
) -> Result<Vec<usize>, String> {
    if azimuth_count != EXACT_B_FIELD_RECOVERY_AZIMUTH_COUNT {
        return Err(format!(
            "exact-B field-recovery design expects {EXACT_B_FIELD_RECOVERY_AZIMUTH_COUNT} azimuths, got {azimuth_count}"
        ));
    }
    let mut order = Vec::with_capacity(
        EXACT_B_FIELD_RECOVERY_TRAINING_AZIMUTHS.len() * EXACT_B_FLUX_SURFACE_COUNT,
    );
    for round in 0..EXACT_B_FLUX_SURFACE_COUNT {
        for (position, azimuth) in EXACT_B_FIELD_RECOVERY_TRAINING_AZIMUTHS
            .iter()
            .copied()
            .enumerate()
        {
            let surface = (position + round) % EXACT_B_FLUX_SURFACE_COUNT;
            order.push(toroidal_exact_b_flux_row_index(azimuth, surface));
        }
    }
    Ok(order)
}

fn toroidal_exact_b_flux_row_index(azimuth_index: usize, surface_index: usize) -> usize {
    azimuth_index * EXACT_B_FLUX_SURFACE_COUNT + surface_index
}

pub fn toroidal_exact_b_surface_flux_row_norms(
    config: &ToroidalExactBRecoveryConfig,
) -> Result<Vec<ToroidalExactBFluxRowNorm>, String> {
    if config.observation_noise_std <= 0.0 || !config.observation_noise_std.is_finite() {
        return Err(
            "exact-B flux row norm audit requires finite positive observation noise".into(),
        );
    }
    let (topology, coords) = load_mesh(&config.base.mesh_path)?;
    let observation_system = build_exact_b_observation_system(
        &topology,
        &coords,
        ToroidalExactBObservationMode::SurfaceFluxes,
        config.base.geometry,
        config.surface_flux_azimuth_count,
    )?;
    let variance = config.observation_noise_std * config.observation_noise_std;
    let mut sums = vec![0.0; observation_system.operator.nrows()];
    let mut max_abs = vec![0.0_f64; observation_system.operator.nrows()];
    let mut nnz = vec![0usize; observation_system.operator.nrows()];
    for (row, _, value) in observation_system.operator.triplet_iter() {
        sums[row] += value * value;
        max_abs[row] = max_abs[row].max(value.abs());
        nnz[row] += 1;
    }
    observation_system
        .probes
        .iter()
        .enumerate()
        .map(|(row_index, probe)| {
            Ok(ToroidalExactBFluxRowNorm {
                row_index,
                name: probe.name.clone(),
                nnz: nnz[row_index],
                l2_norm: sums[row_index].sqrt(),
                max_abs_entry: max_abs[row_index],
                max_diagonal_contribution: max_abs[row_index] * max_abs[row_index] / variance,
            })
        })
        .collect()
}

pub fn toroidal_exact_b_physical_b_prior_calibration(
    config: &ToroidalExactBRecoveryConfig,
    target_multiplier: f64,
) -> Result<ToroidalExactBPhysicalBPriorCalibrationReport, String> {
    validate_exact_b_config(config)?;
    if !target_multiplier.is_finite() || target_multiplier <= 0.0 {
        return Err(format!(
            "exact-B physical-B prior calibration multiplier must be finite and positive, got {target_multiplier}"
        ));
    }

    let (topology, coords) = load_mesh(&config.base.mesh_path)?;
    let boundary = outer_boundary(&topology, &coords, config.base.geometry);
    let linear_system = build_toroidal_exact_b_linear_system(&topology, &coords, boundary, config)?;
    let nominal_source_full =
        assemble_toroidal_source(&topology, &coords, config.base.geometry, config.base.nu_air);
    let nominal_source = reduce_full_edge_vector(&linear_system.layout, &nominal_source_full)?;
    let reference_pde_variance = config
        .reference_pde_variance
        .unwrap_or(config.base.pde_variance);
    let linear_mean = robust_direct_linear_system_solution(
        &linear_system,
        &nominal_source,
        config.base.prior_precision,
        reference_pde_variance,
    )?;

    let cell_volumes = toroidal_cell_volumes(&topology, &coords)?;
    let domain_volume = cell_volumes.iter().sum::<f64>();
    if !domain_volume.is_finite() || domain_volume <= 0.0 {
        return Err(format!(
            "exact-B physical-B prior calibration requires positive domain volume, got {domain_volume}"
        ));
    }
    let b_operator =
        build_reduced_magnetic_flux_density_operator_3d(&topology, &coords, &linear_system.layout)?;
    let weights = cell_major_b_weights(&cell_volumes);
    if b_operator.nrows() != weights.len() {
        return Err(format!(
            "physical B operator row count {} must match weight count {}",
            b_operator.nrows(),
            weights.len()
        ));
    }
    let nominal_b = b_operator
        .apply(&GmrfVector::from_vec(linear_mean.clone()))
        .map_err(|err| format!("failed to apply nominal physical B operator: {err}"))?;
    let nominal_b_rms = weighted_rms(&nominal_b, &weights, domain_volume)?;
    if nominal_b_rms <= 0.0 {
        return Err("exact-B physical-B prior calibration found zero nominal B RMS".to_string());
    }
    let target_prior_b_rms = target_multiplier * nominal_b_rms;
    let target_trace = target_prior_b_rms * target_prior_b_rms * domain_volume;

    let raw_prior = build_exact_two_form_potential_prior(
        &topology,
        &coords,
        &linear_system.layout,
        linear_mean,
        ExactTwoFormPotentialPriorConfig {
            kappa: config.prior_kappa,
            tau: 1.0,
            mass_inverse: config.potential_mass_inverse,
            diagonal_shift: 0.0,
        },
    )?;
    let raw_q = feec_csr_to_gmrf(&raw_prior.precision);
    let raw_factor = raw_q
        .cholesky_sqrt_lower()
        .map_err(|err| format!("raw exact-B prior precision factorization failed: {err}"))?;
    let raw_variance_trace =
        exact_transformed_variance_weighted_trace(&raw_factor, &b_operator, &weights)
            .map_err(|err| format!("failed to compute physical B prior trace: {err}"))?;
    let raw_trace = raw_variance_trace.weighted_trace.value;
    let normalization = trace_normalization_from_target_trace(raw_trace, target_trace)?;
    let normalized_mean_b2 = normalization.normalized_mean_trace_variance(domain_volume)?;
    let target_mean_b2 = target_prior_b_rms * target_prior_b_rms;
    let normalization_relative_error =
        ((normalized_mean_b2 - target_mean_b2) / target_mean_b2).abs();

    Ok(ToroidalExactBPhysicalBPriorCalibrationReport {
        calibration_label: TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_CALIBRATION_LABEL.to_string(),
        mesh_path: config.base.mesh_path.clone(),
        reference_solve_mode: config.reference_solve_mode,
        reference_solver_diagonal_shift: config.reference_solver_diagonal_shift,
        active_dofs: linear_system.state_dimension(),
        cells: cell_volumes.len(),
        b_rows: b_operator.nrows(),
        domain_volume,
        prior_kappa: config.prior_kappa,
        raw_prior_tau: 1.0,
        target_multiplier,
        nominal_b_rms,
        target_prior_b_rms,
        raw_trace,
        raw_mean_b2: raw_trace / domain_volume,
        target_trace,
        precision_scale: normalization.precision_scale,
        tau_multiplier: normalization.tau_multiplier,
        effective_prior_tau: normalization.tau_multiplier,
        normalized_mean_b2,
        normalization_relative_error,
    })
}

pub fn toroidal_exact_b_physical_b_prior_calibration_csv(
    report: &ToroidalExactBPhysicalBPriorCalibrationReport,
) -> String {
    format!(
        "calibration_label,mesh_path,reference_solve_mode,reference_solver_diagonal_shift,active_dofs,cells,b_rows,domain_volume,prior_kappa,raw_prior_tau,target_multiplier,nominal_b_rms,target_prior_b_rms,raw_trace,raw_mean_b2,target_trace,precision_scale,tau_multiplier,effective_prior_tau,normalized_mean_b2,normalization_relative_error\n{},{},{},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
        report.calibration_label,
        report.mesh_path.display(),
        report.reference_solve_mode.label(),
        report.reference_solver_diagonal_shift,
        report.active_dofs,
        report.cells,
        report.b_rows,
        report.domain_volume,
        report.prior_kappa,
        report.raw_prior_tau,
        report.target_multiplier,
        report.nominal_b_rms,
        report.target_prior_b_rms,
        report.raw_trace,
        report.raw_mean_b2,
        report.target_trace,
        report.precision_scale,
        report.tau_multiplier,
        report.effective_prior_tau,
        report.normalized_mean_b2,
        report.normalization_relative_error
    )
}

pub fn write_toroidal_exact_b_physical_b_prior_calibration_csv(
    report: &ToroidalExactBPhysicalBPriorCalibrationReport,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
        toroidal_exact_b_physical_b_prior_calibration_csv(report),
    )
}

fn exact_b_coordinate_scaling(
    mode: ToroidalExactBNondimensionalizationMode,
    state_operator: &SparseTripletMatrix,
    source_operator: &SparseTripletMatrix,
) -> Result<Option<LinearPdeCoordinateScaling>, String> {
    match mode {
        ToroidalExactBNondimensionalizationMode::Off => Ok(None),
        ToroidalExactBNondimensionalizationMode::PdeColumnNorm => {
            let state_scale = exact_b_column_norm_coordinate_scales(state_operator)?;
            let source_scale = exact_b_column_norm_coordinate_scales(source_operator)?;
            Ok(Some(LinearPdeCoordinateScaling {
                state_scale,
                latent_scales: vec![LinearPdeLatentCoordinateScaling {
                    input_name: SOURCE_DISCREPANCY_INPUT.to_string(),
                    scale: source_scale,
                }],
            }))
        }
    }
}

fn exact_b_column_norm_coordinate_scales(
    operator: &SparseTripletMatrix,
) -> Result<Vec<f64>, String> {
    let mut sums = vec![0.0; operator.ncols()];
    for (_, col, value) in operator.triplet_iter() {
        sums[col] += value * value;
    }
    let positive_norms = sums
        .iter()
        .map(|sum| sum.sqrt())
        .filter(|norm| *norm > 0.0 && norm.is_finite())
        .collect::<Vec<_>>();
    if positive_norms.is_empty() {
        return Err(
            "exact-B nondimensionalization requires at least one nonzero PDE column".into(),
        );
    }
    let median = median_f64(positive_norms)?;
    let floor = (median * 1.0e-12).max(f64::MIN_POSITIVE);
    sums.into_iter()
        .map(|sum| {
            let norm = sum.sqrt();
            let denominator = if norm.is_finite() {
                norm.max(floor)
            } else {
                return Err("exact-B nondimensionalization found non-finite column norm".into());
            };
            Ok(1.0 / denominator)
        })
        .collect()
}

fn median_f64(mut values: Vec<f64>) -> Result<f64, String> {
    if values.is_empty() {
        return Err("median requires at least one value".into());
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Ok(0.5 * (values[mid - 1] + values[mid]))
    } else {
        Ok(values[mid])
    }
}

fn exact_b_joint_coordinate_scale(
    joint_dimension: usize,
    state_dimension: usize,
    source_dimension: usize,
    coordinate_scaling: Option<&LinearPdeCoordinateScaling>,
) -> Result<Vec<f64>, String> {
    let mut scale = vec![1.0; joint_dimension];
    if let Some(coordinate_scaling) = coordinate_scaling {
        if coordinate_scaling.state_scale.len() != state_dimension {
            return Err(format!(
                "state coordinate scale length {} must match state dimension {state_dimension}",
                coordinate_scaling.state_scale.len()
            ));
        }
        scale[..state_dimension].copy_from_slice(&coordinate_scaling.state_scale);
        let source_scale = coordinate_scaling
            .latent_scales
            .iter()
            .find(|latent| latent.input_name == SOURCE_DISCREPANCY_INPUT)
            .ok_or_else(|| "missing exact-B source coordinate scale".to_string())?;
        if source_scale.scale.len() != source_dimension {
            return Err(format!(
                "source coordinate scale length {} must match source dimension {source_dimension}",
                source_scale.scale.len()
            ));
        }
        scale[state_dimension..state_dimension + source_dimension]
            .copy_from_slice(&source_scale.scale);
    }
    Ok(scale)
}

fn finite_scale_min_max(scale: &[f64]) -> (f64, f64) {
    scale
        .iter()
        .copied()
        .fold((f64::INFINITY, 0.0_f64), |(min, max), value| {
            (min.min(value), max.max(value))
        })
}

pub fn toroidal_exact_b_precision_scale_audit(
    config: &ToroidalExactBRecoveryConfig,
) -> Result<ToroidalExactBPrecisionScaleAudit, String> {
    validate_exact_b_config(config)?;

    let (topology, coords) = load_mesh(&config.base.mesh_path)?;
    let boundary = outer_boundary(&topology, &coords, config.base.geometry);
    let linear_system = build_toroidal_exact_b_linear_system(&topology, &coords, boundary, config)?;
    let state_dimension = linear_system.state_dimension();
    let source_dimension = SOURCE_MODE_COUNT;
    let joint_dimension = state_dimension + source_dimension;

    let exact_prior = build_exact_two_form_potential_prior(
        &topology,
        &coords,
        &linear_system.layout,
        vec![0.0; state_dimension],
        ExactTwoFormPotentialPriorConfig {
            kappa: config.prior_kappa,
            tau: config.prior_tau,
            mass_inverse: config.potential_mass_inverse,
            diagonal_shift: 0.0,
        },
    )?;

    let source_modes = assemble_toroidal_source_modes(
        &topology,
        &coords,
        &linear_system.layout,
        config.base.geometry,
        config.base.nu_air,
    )?;
    let observation_system = build_exact_b_observation_system(
        &topology,
        &coords,
        config.observation_mode,
        config.base.geometry,
        config.surface_flux_azimuth_count,
    )?;
    let (training_indices, heldout_indices) = resolve_exact_b_observation_indices(
        config,
        &linear_system,
        &source_modes,
        &observation_system.operator,
    )?;

    let selected_rows = select_residual_rows(
        linear_system.residual_dimension(),
        &config.base.residual_selection,
    )?;
    let inference_system = select_linear_system_residual_rows(&linear_system, &selected_rows)?;
    let source_modes_selected = source_modes
        .iter()
        .map(|mode| select_vector_entries(mode, &selected_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let source_operator = build_source_discrepancy_operator(&source_modes_selected)?;
    let inference_operator = feec_csr_to_core_triplet(&inference_system.operator);
    let coordinate_scaling = exact_b_coordinate_scaling(
        config.nondimensionalization,
        &inference_operator,
        &source_operator,
    )?;

    let mut state_prior = vec![0.0; joint_dimension];
    add_sparse_diagonal_to_precision_audit(
        &mut state_prior,
        &exact_prior.spec.precision,
        0,
        "state_prior",
    )?;

    let mut source_prior = vec![0.0; joint_dimension];
    let source_prior_precision = 1.0 / (config.source_prior_std * config.source_prior_std);
    for mode_index in 0..source_dimension {
        source_prior[state_dimension + mode_index] += source_prior_precision;
    }

    let mut pde_state = vec![0.0; joint_dimension];
    add_normal_diagonal_to_precision_audit(
        &mut pde_state,
        &inference_operator,
        0,
        config.base.pde_variance,
        "pde_state",
    )?;

    let mut pde_source = vec![0.0; joint_dimension];
    add_normal_diagonal_to_precision_audit(
        &mut pde_source,
        &source_operator,
        state_dimension,
        config.base.pde_variance,
        "pde_source",
    )?;

    let selected_observation_full =
        select_triplet_rows(&observation_system.operator, &training_indices)?;
    let selected_bias = vec![0.0; training_indices.len()];
    let (selected_observation_reduced, _) = restrict_triplet_columns_and_fold_fixed(
        &selected_observation_full,
        &selected_bias,
        &linear_system.layout,
    )?;
    let mut flux_observations = vec![0.0; joint_dimension];
    add_normal_diagonal_to_precision_audit(
        &mut flux_observations,
        &selected_observation_reduced,
        0,
        config.observation_noise_std * config.observation_noise_std,
        "flux_observations",
    )?;

    let posterior_total = sum_precision_audit_diagonals(&[
        &state_prior,
        &source_prior,
        &pde_state,
        &pde_source,
        &flux_observations,
    ]);
    let joint_scale = exact_b_joint_coordinate_scale(
        joint_dimension,
        state_dimension,
        source_dimension,
        coordinate_scaling.as_ref(),
    )?;
    let scaled_posterior_total = scale_precision_audit_diagonal(&posterior_total, &joint_scale)?;
    let scaled_posterior = summarize_precision_audit_diagonal(
        "scaled_posterior_total",
        &scaled_posterior_total,
        state_dimension,
    );
    let (state_scale_min, state_scale_max) = finite_scale_min_max(&joint_scale[..state_dimension]);
    let (source_scale_min, source_scale_max) =
        finite_scale_min_max(&joint_scale[state_dimension..state_dimension + source_dimension]);
    let terms = vec![
        summarize_precision_audit_diagonal("state_prior", &state_prior, state_dimension),
        summarize_precision_audit_diagonal("source_prior", &source_prior, state_dimension),
        summarize_precision_audit_diagonal("pde_state", &pde_state, state_dimension),
        summarize_precision_audit_diagonal("pde_source", &pde_source, state_dimension),
        summarize_precision_audit_diagonal(
            "flux_observations",
            &flux_observations,
            state_dimension,
        ),
        summarize_precision_audit_diagonal("posterior_total", &posterior_total, state_dimension),
    ];

    Ok(ToroidalExactBPrecisionScaleAudit {
        joint_dimension,
        state_dimension,
        source_dimension,
        training_rows: training_indices.len(),
        heldout_rows: heldout_indices.len(),
        nondimensionalization: config.nondimensionalization,
        state_scale_min,
        state_scale_max,
        source_scale_min,
        source_scale_max,
        scaled_posterior_max_abs_diagonal: scaled_posterior.max_abs_diagonal,
        scaled_posterior_diagonal_ratio: scaled_posterior_diagonal_ratio(&scaled_posterior_total),
        terms,
    })
}

fn add_sparse_diagonal_to_precision_audit(
    diagonal: &mut [f64],
    matrix: &SparseTripletMatrix,
    offset: usize,
    label: &str,
) -> Result<(), String> {
    if offset + matrix.nrows() > diagonal.len() || matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "{label} precision audit expected a square block fitting at offset {offset}, got {}x{} into dimension {}",
            matrix.nrows(),
            matrix.ncols(),
            diagonal.len()
        ));
    }
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            let entry = &mut diagonal[offset + row];
            *entry += value;
            if !entry.is_finite() {
                return Err(format!(
                    "{label} precision audit produced non-finite diagonal at {}",
                    offset + row
                ));
            }
        }
    }
    Ok(())
}

fn add_normal_diagonal_to_precision_audit(
    diagonal: &mut [f64],
    operator: &SparseTripletMatrix,
    offset: usize,
    variance: f64,
    label: &str,
) -> Result<(), String> {
    if !variance.is_finite() || variance <= 0.0 {
        return Err(format!(
            "{label} precision audit requires finite positive variance"
        ));
    }
    if offset + operator.ncols() > diagonal.len() {
        return Err(format!(
            "{label} precision audit operator with {} columns does not fit at offset {offset} into dimension {}",
            operator.ncols(),
            diagonal.len()
        ));
    }
    for (_, col, value) in operator.triplet_iter() {
        let contribution = value * value / variance;
        let entry = &mut diagonal[offset + col];
        *entry += contribution;
        if !entry.is_finite() {
            return Err(format!(
                "{label} precision audit produced non-finite diagonal at {}",
                offset + col
            ));
        }
    }
    Ok(())
}

fn sum_precision_audit_diagonals(diagonals: &[&[f64]]) -> Vec<f64> {
    let dimension = diagonals
        .first()
        .map(|diagonal| diagonal.len())
        .unwrap_or(0);
    let mut total = vec![0.0; dimension];
    for diagonal in diagonals {
        debug_assert_eq!(diagonal.len(), dimension);
        for (total_entry, value) in total.iter_mut().zip(*diagonal) {
            *total_entry += *value;
        }
    }
    total
}

fn scale_precision_audit_diagonal(diagonal: &[f64], scale: &[f64]) -> Result<Vec<f64>, String> {
    if diagonal.len() != scale.len() {
        return Err(format!(
            "precision audit diagonal length {} must match coordinate scale length {}",
            diagonal.len(),
            scale.len()
        ));
    }
    Ok(diagonal
        .iter()
        .zip(scale)
        .map(|(value, scale)| value * scale * scale)
        .collect())
}

fn scaled_posterior_diagonal_ratio(diagonal: &[f64]) -> f64 {
    let min_positive = diagonal
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    let max_abs = diagonal.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if min_positive > 0.0 && min_positive.is_finite() {
        max_abs / min_positive
    } else {
        f64::INFINITY
    }
}

fn summarize_precision_audit_diagonal(
    term: &str,
    diagonal: &[f64],
    state_dimension: usize,
) -> ToroidalExactBPrecisionTermScale {
    let nonzero_diagonal_entries = diagonal.iter().filter(|value| **value != 0.0).count();
    let min_positive_diagonal = diagonal
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    let min_positive_diagonal = if min_positive_diagonal.is_finite() {
        min_positive_diagonal
    } else {
        0.0
    };
    let (max_index, max_abs_diagonal) =
        diagonal
            .iter()
            .copied()
            .enumerate()
            .fold((0usize, 0.0_f64), |best, (index, value)| {
                let abs = value.abs();
                if abs > best.1 {
                    (index, abs)
                } else {
                    best
                }
            });
    let (max_block, max_local_index) = if max_index < state_dimension {
        ("state".to_string(), max_index)
    } else {
        ("source".to_string(), max_index - state_dimension)
    };

    ToroidalExactBPrecisionTermScale {
        term: term.to_string(),
        nonzero_diagonal_entries,
        min_positive_diagonal,
        max_abs_diagonal,
        diagonal_sum: diagonal.iter().sum(),
        max_index,
        max_block,
        max_local_index,
    }
}

pub fn build_reduced_flux_sensors(
    topology: &Complex,
    coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
    specs: &[ToroidalFluxSensorSpec],
) -> Result<Vec<ToroidalReducedFluxSensor>, String> {
    let face_rows = csr_rows(&FeecCsr::from(&topology.exterior_derivative_operator(1)));
    specs
        .iter()
        .map(|spec| {
            let full = build_oriented_flux_patch_operator(
                topology,
                coords,
                &face_rows,
                spec.center,
                spec.normal,
                spec.patch_radius,
                spec.normal_half_width,
            )?;
            let (operator, bias) =
                restrict_triplet_columns_and_fold_fixed(&full, &[0.0], model.layout())?;
            Ok(ToroidalReducedFluxSensor {
                spec: spec.clone(),
                operator,
                bias,
            })
        })
        .collect()
}

pub fn apply_reduced_flux_sensors(
    sensors: &[ToroidalReducedFluxSensor],
    reduced_state: &[f64],
) -> Result<Vec<f64>, String> {
    sensors
        .iter()
        .map(|sensor| {
            if sensor.bias.len() != 1 {
                return Err(format!("sensor `{}` has non-scalar bias", sensor.spec.name));
            }
            Ok(apply_sparse_triplet(&sensor.operator, reduced_state)[0] + sensor.bias[0])
        })
        .collect()
}

pub fn sorted_boundary_dofs<P>(
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

pub fn write_nonlinear_toroidal_outputs(
    output_dir: &Path,
    topology: &Complex,
    coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
    reduced_state: &[f64],
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    let full = model.lift_reduced_state(reduced_state)?;
    let a = Cochain::new(1, FeecVector::from_vec(full));
    let b = a.dif(topology);
    crate::visual_output::write_cochain(
        output_dir.join("nonlinear_A.vtu"),
        coords,
        topology,
        &a,
        "A",
    )
    .map_err(|err| err.to_string())?;
    crate::visual_output::write_1form_vector_field(
        output_dir.join("nonlinear_A_vector.vtu"),
        coords,
        topology,
        &a,
        "A_vector",
    )
    .map_err(|err| err.to_string())?;
    crate::visual_output::write_cochain(
        output_dir.join("nonlinear_B.vtu"),
        coords,
        topology,
        &b,
        "B",
    )
    .map_err(|err| err.to_string())?;
    crate::visual_output::write_2form_vector_field(
        output_dir.join("nonlinear_B_vector.vtu"),
        coords,
        topology,
        &b,
        "B_vector",
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

// Exact-B output assembly retains each scientific artifact as a separately named
// input; toroidal canonical/source/coverage smoke profiles verify the inventory.
#[allow(clippy::too_many_arguments)]
fn write_exact_b_recovery_outputs(
    output_dir: &Path,
    topology: &Complex,
    coords: &MeshCoords,
    posterior_full: &[f64],
    reference_full: Option<&[f64]>,
    summary: &ToroidalExactBStageSummary,
    source_response: &ToroidalExactBSourceResponseSummary,
    source_posterior: &[ToroidalExactBSourcePosteriorRow],
    heldout_predictions: &[ToroidalExactBHeldoutPredictionRow],
    probes: &[ToroidalExactBProbeRow],
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create exact-B output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    fs::write(
        output_dir.join("stage_summary.csv"),
        exact_b_stage_summary_csv(summary),
    )
    .map_err(|err| err.to_string())?;
    fs::write(
        output_dir.join("source_discrepancy_posterior.csv"),
        exact_b_source_posterior_csv(source_posterior),
    )
    .map_err(|err| err.to_string())?;
    fs::write(
        output_dir.join("source_design_observability.csv"),
        exact_b_source_response_csv(source_response),
    )
    .map_err(|err| err.to_string())?;
    fs::write(
        output_dir.join("heldout_predictions.csv"),
        exact_b_heldout_prediction_csv(heldout_predictions),
    )
    .map_err(|err| err.to_string())?;
    fs::write(
        output_dir.join("probe_points.csv"),
        exact_b_probe_csv(probes),
    )
    .map_err(|err| err.to_string())?;

    if posterior_full.len() != topology.nsimplices(1) {
        return Err(format!(
            "posterior full state length {} must match edge count {}",
            posterior_full.len(),
            topology.nsimplices(1)
        ));
    }
    let posterior_a = Cochain::new(1, FeecVector::from_vec(posterior_full.to_vec()));
    let posterior_b = posterior_a.dif(topology);
    crate::visual_output::write_cochain(
        output_dir.join("posterior_A.vtu"),
        coords,
        topology,
        &posterior_a,
        "posterior_A",
    )
    .map_err(|err| err.to_string())?;
    crate::visual_output::write_cochain(
        output_dir.join("posterior_B.vtu"),
        coords,
        topology,
        &posterior_b,
        "posterior_B",
    )
    .map_err(|err| err.to_string())?;
    crate::visual_output::write_2form_vector_field(
        output_dir.join("posterior_B_vector.vtu"),
        coords,
        topology,
        &posterior_b,
        "posterior_B_vector",
    )
    .map_err(|err| err.to_string())?;

    if let Some(reference_full) = reference_full {
        if reference_full.len() != topology.nsimplices(1) {
            return Err(format!(
                "reference full state length {} must match edge count {}",
                reference_full.len(),
                topology.nsimplices(1)
            ));
        }
        let reference_a = Cochain::new(1, FeecVector::from_vec(reference_full.to_vec()));
        let reference_b = reference_a.dif(topology);
        let error_b = Cochain::new(2, &posterior_b.coeffs - &reference_b.coeffs);
        crate::visual_output::write_cochain(
            output_dir.join("reference_B.vtu"),
            coords,
            topology,
            &reference_b,
            "reference_B",
        )
        .map_err(|err| err.to_string())?;
        crate::visual_output::write_2form_vector_field(
            output_dir.join("reference_B_vector.vtu"),
            coords,
            topology,
            &reference_b,
            "reference_B_vector",
        )
        .map_err(|err| err.to_string())?;
        crate::visual_output::write_cochain(
            output_dir.join("posterior_B_error.vtu"),
            coords,
            topology,
            &error_b,
            "posterior_B_error",
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn exact_b_stage_summary_csv(summary: &ToroidalExactBStageSummary) -> String {
    format!(
        "reference_mode,reference_solve_mode,reference_solver_diagonal_shift,observation_mode,prior_tau,prior_calibration_label,prior_calibration_nominal_b_rms,prior_calibration_target_b_rms,prior_calibration_multiplier,active_dofs,source_modes,training_rows,heldout_rows,residual_rows_used,residual_rows_total,final_residual_norm,train_rmse,heldout_latent_rmse,heldout_latent_nlpd,heldout_latent_covered95,heldout_latent_coverage_fraction,heldout_mean_posterior_flux_sd,heldout_latent_max_abs_z,heldout_latent_rms_z,heldout_latent_mean_abs_residual,heldout_noisy_rmse,heldout_noisy_nlpd,heldout_noisy_covered95,heldout_noisy_coverage_fraction,heldout_mean_predictive_sd,heldout_noisy_max_abs_z,heldout_noisy_rms_z,heldout_noisy_mean_abs_residual,b_relative_error,posterior_factor_nnz\n{},{},{:.16e},{},{:.16e},{},{},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{}\n",
        summary.reference_mode.label(),
        summary.reference_solve_mode.label(),
        summary.reference_solver_diagonal_shift,
        summary.observation_mode.label(),
        summary.prior_tau,
        summary.prior_calibration_label,
        csv_option(summary.prior_calibration_nominal_b_rms),
        csv_option(summary.prior_calibration_target_b_rms),
        csv_option(summary.prior_calibration_multiplier),
        summary.active_dofs,
        summary.source_modes,
        summary.training_rows,
        summary.heldout_rows,
        summary.residual_rows_used,
        summary.residual_rows_total,
        summary.final_residual_norm,
        summary.train_rmse,
        summary.heldout_rmse,
        summary.heldout_nlpd,
        summary.heldout_covered95,
        summary.heldout_coverage_fraction,
        summary.heldout_mean_posterior_flux_sd,
        summary.heldout_max_abs_z,
        summary.heldout_rms_z,
        summary.heldout_mean_abs_residual,
        summary.heldout_noisy_rmse,
        summary.heldout_noisy_nlpd,
        summary.heldout_noisy_covered95,
        summary.heldout_noisy_coverage_fraction,
        summary.heldout_mean_predictive_sd,
        summary.heldout_noisy_max_abs_z,
        summary.heldout_noisy_rms_z,
        summary.heldout_noisy_mean_abs_residual,
        csv_option(summary.b_relative_error),
        summary.posterior_factor_nnz
    )
}

fn exact_b_source_posterior_csv(rows: &[ToroidalExactBSourcePosteriorRow]) -> String {
    let mut csv =
        "mode_index,prior_mean,prior_variance,truth,posterior_mean,posterior_variance,error\n"
            .to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            row.mode_index,
            row.prior_mean,
            row.prior_variance,
            row.truth,
            row.posterior_mean,
            row.posterior_variance,
            row.error
        ));
    }
    csv
}

fn exact_b_source_response_csv(row: &ToroidalExactBSourceResponseSummary) -> String {
    let mut csv =
        "condition,snr_min,snr_max,singular_0,singular_1,singular_2,singular_3,column_norm_0,column_norm_1,column_norm_2,column_norm_3\n"
            .to_string();
    csv.push_str(&format!(
        "{:.16e},{:.16e},{:.16e}",
        row.condition, row.snr_min, row.snr_max
    ));
    for value in row.singular_values {
        csv.push_str(&format!(",{value:.16e}"));
    }
    for value in row.column_norms {
        csv.push_str(&format!(",{value:.16e}"));
    }
    csv.push('\n');
    csv
}

fn exact_b_heldout_prediction_csv(rows: &[ToroidalExactBHeldoutPredictionRow]) -> String {
    let mut csv = "name,latent_truth,noisy_observation,prediction,latent_residual,noisy_residual,posterior_flux_sd,predictive_sd,latent_standardized_residual,noisy_standardized_residual,latent_lower95,latent_upper95,latent_covered95,noisy_lower95,noisy_upper95,noisy_covered95\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{}\n",
            row.name,
            row.truth,
            row.noisy_observation,
            row.prediction,
            row.residual,
            row.noisy_residual,
            row.posterior_sd,
            row.predictive_sd,
            row.standardized_residual,
            row.noisy_standardized_residual,
            row.lower95,
            row.upper95,
            row.covered95,
            row.noisy_lower95,
            row.noisy_upper95,
            row.noisy_covered95
        ));
    }
    csv
}

fn exact_b_probe_csv(rows: &[ToroidalExactBProbeRow]) -> String {
    let mut csv = "name,cell_index,component,x,y,z,role\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{:.16e},{:.16e},{:.16e},{}\n",
            row.name, row.cell_index, row.component, row.x, row.y, row.z, row.role
        ));
    }
    csv
}

fn exact_b_diagnostics_csv(rows: &[ToroidalExactBDiagnosticRow]) -> String {
    let mut csv = String::from(
        "sweep,observation_mode,pde_variance,prior_tau,source_prior_std,observation_noise_std,training_rows,heldout_rows,train_rmse,heldout_rmse,b_relative_error,final_residual_norm,source_response_condition,source_response_snr_min,source_response_snr_max,posterior_factor_nnz,runtime_seconds",
    );
    for prefix in [
        "singular",
        "column_norm",
        "eta_truth",
        "eta_mean",
        "eta_variance",
        "eta_error",
    ] {
        for mode_index in 0..SOURCE_MODE_COUNT {
            csv.push_str(&format!(",{prefix}_{mode_index}"));
        }
    }
    csv.push('\n');
    for row in rows {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.6}",
            row.sweep,
            row.observation_mode.label(),
            row.pde_variance,
            row.prior_tau,
            row.source_prior_std,
            row.observation_noise_std,
            row.training_rows,
            row.heldout_rows,
            row.train_rmse,
            row.heldout_rmse,
            row.b_relative_error,
            row.final_residual_norm,
            row.source_response_condition,
            row.source_response_snr_min,
            row.source_response_snr_max,
            row.posterior_factor_nnz,
            row.runtime_seconds
        ));
        for values in [
            row.source_response_singular_values,
            row.source_response_column_norms,
            row.eta_truth,
            row.eta_posterior_mean,
            row.eta_posterior_variance,
            row.eta_error,
        ] {
            for value in values {
                csv.push_str(&format!(",{value:.16e}"));
            }
        }
        csv.push('\n');
    }
    csv
}

fn csv_option(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.16e}"))
        .unwrap_or_else(|| "nan".to_string())
}

fn exact_b_prior_calibration_metadata(
    prior_tau: f64,
) -> (String, Option<f64>, Option<f64>, Option<f64>) {
    if approx_equal(prior_tau, TOROIDAL_EXACT_B_CALIBRATED_PRIOR_TAU) {
        return (
            TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_CALIBRATION_LABEL.to_string(),
            Some(TOROIDAL_EXACT_B_CALIBRATION_NOMINAL_B_RMS),
            Some(TOROIDAL_EXACT_B_CALIBRATION_TARGET_PRIOR_B_RMS),
            Some(TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_RMS_MULTIPLIER),
        );
    }

    if prior_tau.is_finite() && prior_tau > 0.0 {
        let inferred_multiplier = TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_RMS_MULTIPLIER
            * TOROIDAL_EXACT_B_CALIBRATED_PRIOR_TAU
            / prior_tau;
        for supported in [30.0, 100.0] {
            if approx_equal(inferred_multiplier, supported) {
                return (
                    format!("physical_b_nominal_rms_x{}", supported as usize),
                    Some(TOROIDAL_EXACT_B_CALIBRATION_NOMINAL_B_RMS),
                    Some(TOROIDAL_EXACT_B_CALIBRATION_NOMINAL_B_RMS * supported),
                    Some(supported),
                );
            }
        }
    }

    ("uncalibrated".to_string(), None, None, None)
}

fn approx_equal(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= 1e-12 * lhs.abs().max(rhs.abs()).max(1.0)
}

struct ToroidalPriorBuild {
    spec: GaussianPriorSpec,
    kappa: f64,
    tau: f64,
    kappa_fallback_used: bool,
}

#[derive(Debug, Clone)]
struct ToroidalResidualNoise {
    noise: GaussianNoiseModel,
    normalization_scale: Option<f64>,
}

enum ToroidalObservationResidualModel<'a> {
    Weak {
        model: &'a ReducedVectorPotentialMagnetostatic3d,
        rows: Vec<usize>,
    },
    LocalStrong(Box<LocalMagneticStrongProbe3d>),
}

impl NonlinearResidualModel for ToroidalObservationResidualModel<'_> {
    fn state_dimension(&self) -> usize {
        match self {
            Self::Weak { model, .. } => FeecResidualModel::state_dimension(*model),
            Self::LocalStrong(model) => FeecResidualModel::state_dimension(model.as_ref()),
        }
    }

    fn residual_dimension(&self) -> usize {
        match self {
            Self::Weak { rows, .. } => rows.len(),
            Self::LocalStrong(model) => FeecResidualModel::residual_dimension(model.as_ref()),
        }
    }

    fn residual_and_jacobian(
        &self,
        state: &[f64],
    ) -> Result<feg_core::NonlinearResidualEvaluation, String> {
        match self {
            Self::Weak { model, rows } => {
                let adapter = FeecResidualAdapter::new(*model);
                SelectedResidualModel::new(&adapter, rows.clone())?.residual_and_jacobian(state)
            }
            Self::LocalStrong(model) => {
                FeecResidualAdapter::new(model.as_ref()).residual_and_jacobian(state)
            }
        }
    }
}

struct ToroidalObservationBuild<'a> {
    model: ToroidalObservationResidualModel<'a>,
    noise: ToroidalResidualNoise,
    rows_used: usize,
    rows_total: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct StratifiedCellSelection {
    cells: Vec<usize>,
    weights: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ToroidalExactBObservationSystem {
    operator: SparseTripletMatrix,
    probes: Vec<ToroidalExactBProbeRow>,
}

impl ToroidalExactBObservationSystem {
    fn probes_with_roles(&self, roles: &[String]) -> Result<Vec<ToroidalExactBProbeRow>, String> {
        if roles.len() != self.probes.len() {
            return Err(format!(
                "exact-B role count {} must match observation count {}",
                roles.len(),
                self.probes.len()
            ));
        }
        Ok(self
            .probes
            .iter()
            .zip(roles)
            .map(|(probe, role)| {
                let mut probe = probe.clone();
                probe.role = role.clone();
                probe
            })
            .collect())
    }
}

fn validate_exact_b_config(config: &ToroidalExactBRecoveryConfig) -> Result<(), String> {
    if !config.source_prior_std.is_finite() || config.source_prior_std <= 0.0 {
        return Err("exact-B source_prior_std must be finite and positive".to_string());
    }
    if !config.prior_kappa.is_finite() || config.prior_kappa <= 0.0 {
        return Err("exact-B prior_kappa must be finite and positive".to_string());
    }
    if !config.prior_tau.is_finite() || config.prior_tau <= 0.0 {
        return Err("exact-B prior_tau must be finite and positive".to_string());
    }
    if !config.observation_noise_std.is_finite() || config.observation_noise_std <= 0.0 {
        return Err("exact-B observation_noise_std must be finite and positive".to_string());
    }
    if let Some(reference_pde_variance) = config.reference_pde_variance {
        if !reference_pde_variance.is_finite() || reference_pde_variance <= 0.0 {
            return Err("exact-B reference_pde_variance must be finite and positive".to_string());
        }
    }
    if !config.reference_solver_diagonal_shift.is_finite()
        || config.reference_solver_diagonal_shift < 0.0
    {
        return Err(
            "exact-B reference_solver_diagonal_shift must be finite and nonnegative".to_string(),
        );
    }
    if config.reference_solve_mode == ToroidalExactBReferenceSolveMode::RegularizedMap
        && config.reference_solver_diagonal_shift != 0.0
    {
        return Err(
            "exact-B reference_solver_diagonal_shift is only used by deterministic_pde reference solves"
                .to_string(),
        );
    }
    if !config.observation_train_fraction.is_finite()
        || config.observation_train_fraction <= 0.0
        || config.observation_train_fraction >= 1.0
    {
        return Err("exact-B observation_train_fraction must lie in (0, 1)".to_string());
    }
    if !config.source_deltas.iter().all(|value| value.is_finite()) {
        return Err("exact-B source_deltas must be finite".to_string());
    }
    if config.base.variance.mode == LinearPdeVarianceMode::SelectedInverse {
        return Err("exact-B variance estimation must not use selected inverse".to_string());
    }
    if matches!(
        config.observation_mode,
        ToroidalExactBObservationMode::SurfaceFluxes
            | ToroidalExactBObservationMode::SourceDesignedFluxes
    ) && config.surface_flux_azimuth_count == 0
    {
        return Err("exact-B surface_flux_azimuth_count must be positive".to_string());
    }
    Ok(())
}

fn build_toroidal_exact_b_linear_system(
    topology: &Complex,
    coords: &MeshCoords,
    boundary: BoundarySpec,
    config: &ToroidalExactBRecoveryConfig,
) -> Result<ReducedLinearPdeAssembly, String> {
    let metric = coords.to_edge_lengths(topology);
    let weight = toroidal_linear_reluctivity_weight(
        config.base.geometry,
        config.base.nu_air,
        config.base.nu_core0,
    );
    let essential = toroidal_essential_boundary(topology, &boundary, true)?;
    build_reduced_weighted_hodge_laplace_1form_system(
        topology, &metric, coords, None, &weight, &essential,
    )
}

fn toroidal_linear_reluctivity_weight(
    geometry: ToroidalInductorGeometry,
    nu_air: f64,
    nu_core0: f64,
) -> InnerProductWeightClosure {
    InnerProductWeightClosure::new(move |point| {
        let radius = (point[0] * point[0] + point[1] * point[1]).sqrt();
        let core_distance = ((radius - geometry.major_radius).powi(2) + point[2] * point[2]).sqrt();
        if core_distance <= geometry.core_minor_radius {
            nu_core0
        } else {
            nu_air
        }
    })
}

fn select_linear_system_residual_rows(
    system: &ReducedLinearPdeAssembly,
    rows: &[usize],
) -> Result<ReducedLinearPdeAssembly, String> {
    Ok(ReducedLinearPdeAssembly {
        operator: core_triplet_to_feec_csr(&select_triplet_rows(
            &feec_csr_to_core_triplet(&system.operator),
            rows,
        )?),
        residual_bias: FeecVector::from_vec(select_vector_entries(
            system.residual_bias.as_slice(),
            rows,
        )?),
        state_mass: system.state_mass.clone(),
        state_mass_inverse: system.state_mass_inverse.clone(),
        layout: system.layout.clone(),
        forcing_operator: core_triplet_to_feec_csr(&select_triplet_rows(
            &feec_csr_to_core_triplet(&system.forcing_operator),
            rows,
        )?),
        neumann_operator: core_triplet_to_feec_csr(&select_triplet_rows(
            &feec_csr_to_core_triplet(&system.neumann_operator),
            rows,
        )?),
    })
}

fn select_triplet_rows(
    matrix: &SparseTripletMatrix,
    rows: &[usize],
) -> Result<SparseTripletMatrix, String> {
    let row_map = rows
        .iter()
        .copied()
        .enumerate()
        .map(|(selected, source)| {
            if source >= matrix.nrows() {
                Err(format!(
                    "selected row {source} is outside matrix row count {}",
                    matrix.nrows()
                ))
            } else {
                Ok((source, selected))
            }
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut selected = SparseTripletMatrix::new(rows.len(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if let Some(&selected_row) = row_map.get(&row) {
            selected.push(selected_row, col, value);
        }
    }
    Ok(selected)
}

fn select_vector_entries(values: &[f64], rows: &[usize]) -> Result<Vec<f64>, String> {
    rows.iter()
        .copied()
        .map(|row| {
            values.get(row).copied().ok_or_else(|| {
                format!(
                    "selected row {row} is outside vector length {}",
                    values.len()
                )
            })
        })
        .collect()
}

fn build_source_discrepancy_operator(
    source_modes: &[Vec<f64>],
) -> Result<SparseTripletMatrix, String> {
    if source_modes.len() != SOURCE_MODE_COUNT {
        return Err(format!(
            "expected {SOURCE_MODE_COUNT} source modes, found {}",
            source_modes.len()
        ));
    }
    let residual_dimension = source_modes.first().map(|mode| mode.len()).unwrap_or(0);
    let mut operator = SparseTripletMatrix::new(residual_dimension, SOURCE_MODE_COUNT);
    for (mode_index, mode) in source_modes.iter().enumerate() {
        if mode.len() != residual_dimension {
            return Err(format!(
                "source mode {mode_index} length {} must match residual dimension {residual_dimension}",
                mode.len()
            ));
        }
        for (row, value) in mode.iter().copied().enumerate() {
            if value != 0.0 {
                operator.push(row, mode_index, -value);
            }
        }
    }
    Ok(operator)
}

fn assemble_toroidal_source_modes(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &formoniq::reduction::DofLayout,
    geometry: ToroidalInductorGeometry,
    nu_air: f64,
) -> Result<Vec<Vec<f64>>, String> {
    (0..SOURCE_MODE_COUNT)
        .map(|sector| {
            let full = assemble_toroidal_source_sector(topology, coords, geometry, nu_air, sector);
            reduce_full_edge_vector(layout, &full)
        })
        .collect()
}

fn source_with_sector_deltas(
    nominal_source: &[f64],
    source_modes: &[Vec<f64>],
    deltas: &[f64; SOURCE_MODE_COUNT],
) -> Result<Vec<f64>, String> {
    let mut source = nominal_source.to_vec();
    for (mode_index, (delta, mode)) in deltas.iter().zip(source_modes.iter()).enumerate() {
        if mode.len() != source.len() {
            return Err(format!(
                "source mode {mode_index} length {} does not match source length {}",
                mode.len(),
                source.len()
            ));
        }
        for (value, mode_value) in source.iter_mut().zip(mode) {
            *value += *delta * *mode_value;
        }
    }
    Ok(source)
}

fn solve_exact_b_reference_state(
    system: &ReducedLinearPdeAssembly,
    truth_source: &[f64],
    solve_mode: ToroidalExactBReferenceSolveMode,
    diagonal_shift: f64,
    prior_precision: f64,
    pde_variance: f64,
) -> Result<Vec<f64>, String> {
    match solve_mode {
        ToroidalExactBReferenceSolveMode::RegularizedMap => robust_direct_linear_system_solution(
            system,
            truth_source,
            prior_precision,
            pde_variance,
        ),
        ToroidalExactBReferenceSolveMode::DeterministicPde => {
            deterministic_linear_system_solution(system, truth_source, diagonal_shift)
        }
    }
}

fn deterministic_linear_system_solution(
    system: &ReducedLinearPdeAssembly,
    source: &[f64],
    diagonal_shift: f64,
) -> Result<Vec<f64>, String> {
    if source.len() != system.residual_dimension() {
        return Err(format!(
            "deterministic exact-B source length {} must match residual dimension {}",
            source.len(),
            system.residual_dimension()
        ));
    }
    if system.operator.nrows() != system.operator.ncols() {
        return Err(format!(
            "deterministic exact-B solve requires a square operator, got {}x{}",
            system.operator.nrows(),
            system.operator.ncols()
        ));
    }
    if system.operator.ncols() != system.state_dimension() {
        return Err(format!(
            "deterministic exact-B operator column count {} must match state dimension {}",
            system.operator.ncols(),
            system.state_dimension()
        ));
    }
    if system.residual_bias.len() != system.residual_dimension() {
        return Err(format!(
            "deterministic exact-B residual bias length {} must match residual dimension {}",
            system.residual_bias.len(),
            system.residual_dimension()
        ));
    }
    if !diagonal_shift.is_finite() || diagonal_shift < 0.0 {
        return Err(format!(
            "deterministic exact-B diagonal shift must be finite and nonnegative, got {diagonal_shift}"
        ));
    }

    let native_operator = feec_csr_to_core_triplet(&system.operator);
    let operator = if diagonal_shift > 0.0 {
        let mut shifted =
            SparseTripletMatrix::new(system.operator.nrows(), system.operator.ncols());
        for (row, col, value) in native_operator.triplet_iter() {
            shifted.push(row, col, value);
        }
        for index in 0..system.operator.nrows() {
            shifted.push(index, index, diagonal_shift);
        }
        shifted
    } else {
        native_operator
    };
    let rhs = source
        .iter()
        .zip(&system.residual_bias)
        .map(|(source_value, bias)| source_value - bias)
        .collect::<Vec<_>>();
    let factor = sparse_from_core(&operator)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor deterministic exact-B PDE operator: {err}"))?;
    factor
        .solve(&GmrfVector::from_vec(rhs))
        .map(|solution| solution.as_slice().to_vec())
        .map_err(|err| format!("failed to solve deterministic exact-B PDE operator: {err}"))
}

fn robust_direct_linear_system_solution(
    system: &ReducedLinearPdeAssembly,
    source: &[f64],
    prior_precision: f64,
    pde_variance: f64,
) -> Result<Vec<f64>, String> {
    let prior_floor = prior_precision.max(1e-12);
    let variance_floor = pde_variance.max(1e-12);
    let candidates = [
        (prior_floor, variance_floor),
        (prior_floor.max(1e-8), variance_floor),
        (prior_floor.max(1e-6), variance_floor),
        (prior_floor.max(1e-4), variance_floor),
        (prior_floor.max(1e-2), variance_floor),
        (prior_floor.max(1.0), variance_floor),
        (prior_floor.max(1.0), variance_floor.max(1e-6)),
        (prior_floor.max(1.0), variance_floor.max(1.0)),
    ];
    let mut last_error = None;
    for (candidate_prior, candidate_variance) in candidates {
        match direct_linear_system_solution(system, source, candidate_prior, candidate_variance) {
            Ok(solution) => return Ok(solution),
            Err(err) => last_error = Some(err),
        }
    }
    Err(format!(
        "failed to compute robust direct linear exact-B solution: {}",
        last_error.unwrap_or_else(|| "no candidate was attempted".to_string())
    ))
}

fn direct_linear_system_solution(
    system: &ReducedLinearPdeAssembly,
    source: &[f64],
    prior_precision: f64,
    pde_variance: f64,
) -> Result<Vec<f64>, String> {
    if source.len() != system.residual_dimension() {
        return Err(format!(
            "linear source length {} must match residual dimension {}",
            source.len(),
            system.residual_dimension()
        ));
    }
    let prior = sparse_from_core(
        &zero_mean_diagonal_prior(system.state_dimension(), prior_precision).precision,
    );
    let observations = GmrfVector::from_vec(source.to_vec());
    let bias = GmrfVector::from_vec(system.residual_bias.as_slice().to_vec());
    let (precision, information) = apply_gaussian_observations(
        &prior,
        &feec_csr_to_gmrf(&system.operator),
        &observations,
        Some(&bias),
        pde_variance,
    );
    let factor = precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor direct linear exact-B system: {err}"))?;
    factor
        .solve(&information)
        .map(|solution| solution.as_slice().to_vec())
        .map_err(|err| format!("failed to solve direct linear exact-B system: {err}"))
}

fn direct_linear_response_solutions(
    system: &ReducedLinearPdeAssembly,
    source_responses: &[Vec<f64>],
    prior_precision: f64,
    pde_variance: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if source_responses.is_empty() {
        return Ok(Vec::new());
    }
    for (index, source_response) in source_responses.iter().enumerate() {
        if source_response.len() != system.residual_dimension() {
            return Err(format!(
                "linear source response {index} length {} must match residual dimension {}",
                source_response.len(),
                system.residual_dimension()
            ));
        }
    }
    let prior = sparse_from_core(
        &zero_mean_diagonal_prior(system.state_dimension(), prior_precision).precision,
    );
    let operator = feec_csr_to_gmrf(&system.operator);
    let mut precision = None;
    let mut informations = Vec::with_capacity(source_responses.len());
    for source_response in source_responses {
        let observations = GmrfVector::from_vec(source_response.clone());
        let (candidate_precision, information) =
            apply_gaussian_observations(&prior, &operator, &observations, None, pde_variance);
        if precision.is_none() {
            precision = Some(candidate_precision);
        }
        informations.push(information);
    }
    let factor = precision
        .ok_or_else(|| "no linear exact-B response precision was assembled".to_string())?
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor linear exact-B response system: {err}"))?;
    informations
        .iter()
        .map(|information| {
            factor
                .solve(information)
                .map(|solution| solution.as_slice().to_vec())
                .map_err(|err| format!("failed to solve linear exact-B response system: {err}"))
        })
        .collect()
}

fn robust_direct_linear_response_solutions(
    system: &ReducedLinearPdeAssembly,
    source_responses: &[Vec<f64>],
    prior_precision: f64,
    pde_variance: f64,
) -> Result<Vec<Vec<f64>>, String> {
    let prior_floor = prior_precision.max(1e-12);
    let variance_floor = pde_variance.max(1e-12);
    let candidates = [
        (prior_floor, variance_floor),
        (prior_floor.max(1e-8), variance_floor),
        (prior_floor.max(1e-6), variance_floor),
        (prior_floor.max(1e-4), variance_floor),
        (prior_floor.max(1e-2), variance_floor),
        (prior_floor.max(1.0), variance_floor),
        (prior_floor.max(1.0), variance_floor.max(1e-6)),
        (prior_floor.max(1.0), variance_floor.max(1.0)),
    ];
    let mut last_error = None;
    for (candidate_prior, candidate_variance) in candidates {
        match direct_linear_response_solutions(
            system,
            source_responses,
            candidate_prior,
            candidate_variance,
        ) {
            Ok(solutions) => return Ok(solutions),
            Err(err) => last_error = Some(err),
        }
    }
    Err(format!(
        "failed to compute robust direct linear exact-B responses: {}",
        last_error.unwrap_or_else(|| "no candidate was attempted".to_string())
    ))
}

fn source_block_selector(
    joint_dimension: usize,
    state_dimension: usize,
) -> Result<SparseRowOperator, String> {
    SparseRowOperator::new(
        joint_dimension,
        (0..SOURCE_MODE_COUNT)
            .map(|mode_index| vec![(state_dimension + mode_index, 1.0)])
            .collect(),
    )
    .map_err(|err| err.to_string())
}

fn exact_b_linear_residual_norm(
    system: &ReducedLinearPdeAssembly,
    source_operator: &SparseTripletMatrix,
    reduced_state: &[f64],
    source_coefficients: &[f64; SOURCE_MODE_COUNT],
) -> Result<f64, String> {
    if reduced_state.len() != system.state_dimension() {
        return Err(format!(
            "exact-B residual state length {} must match dimension {}",
            reduced_state.len(),
            system.state_dimension()
        ));
    }
    let mut residual =
        apply_sparse_triplet(&feec_csr_to_core_triplet(&system.operator), reduced_state);
    for (value, bias) in residual.iter_mut().zip(&system.residual_bias) {
        *value += *bias;
    }
    let source_residual = apply_sparse_triplet(source_operator, source_coefficients);
    for (value, source_value) in residual.iter_mut().zip(source_residual) {
        *value += source_value;
    }
    Ok(l2_norm(&residual))
}

fn exact_b_source_response_summary(
    system: &ReducedLinearPdeAssembly,
    source_modes: &[Vec<f64>],
    observation_operator: Option<&SparseTripletMatrix>,
    prior_precision: f64,
    pde_variance: f64,
    observation_noise_std: f64,
) -> Result<ToroidalExactBSourceResponseSummary, String> {
    let Some(observation_operator) = observation_operator else {
        return Ok(ToroidalExactBSourceResponseSummary {
            condition: f64::INFINITY,
            singular_values: [0.0; SOURCE_MODE_COUNT],
            column_norms: [0.0; SOURCE_MODE_COUNT],
            snr_min: f64::NAN,
            snr_max: f64::NAN,
        });
    };
    let responses = exact_b_source_observation_responses(
        system,
        source_modes,
        observation_operator,
        prior_precision,
        pde_variance,
    )?;
    let mut column_norms = [0.0; SOURCE_MODE_COUNT];
    for (mode_index, response) in responses.iter().enumerate() {
        column_norms[mode_index] = l2_norm(response);
    }
    let mut gram = [[0.0; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT];
    for i in 0..SOURCE_MODE_COUNT {
        for j in i..SOURCE_MODE_COUNT {
            let value = responses[i]
                .iter()
                .zip(&responses[j])
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f64>();
            gram[i][j] = value;
            gram[j][i] = value;
        }
    }
    let singular_values = source_response_singular_values(gram);
    let min_positive = singular_values
        .iter()
        .rev()
        .copied()
        .find(|value| *value > 1e-14)
        .unwrap_or(0.0);
    let condition = if min_positive > 0.0 {
        singular_values[0] / min_positive
    } else {
        f64::INFINITY
    };
    let noise = observation_noise_std.max(EPS);
    let snrs = singular_values.map(|value| value / noise);
    let snr_min = snrs
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    let snr_max = snrs.iter().copied().fold(0.0_f64, f64::max);
    Ok(ToroidalExactBSourceResponseSummary {
        condition,
        singular_values,
        column_norms,
        snr_min,
        snr_max,
    })
}

fn exact_b_source_observation_responses(
    system: &ReducedLinearPdeAssembly,
    source_modes: &[Vec<f64>],
    observation_operator: &SparseTripletMatrix,
    prior_precision: f64,
    pde_variance: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if source_modes.len() != SOURCE_MODE_COUNT {
        return Err(format!(
            "expected {SOURCE_MODE_COUNT} source modes, found {}",
            source_modes.len()
        ));
    }
    let responses = robust_direct_linear_response_solutions(
        system,
        source_modes,
        prior_precision,
        pde_variance,
    )?;
    responses
        .into_iter()
        .map(|response| {
            let full_response =
                lift_vector_with_layout(&system.layout, &FeecVector::from_vec(response))?;
            Ok(apply_sparse_triplet(
                observation_operator,
                full_response.as_slice(),
            ))
        })
        .collect()
}

fn source_response_singular_values(
    gram: [[f64; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT],
) -> [f64; SOURCE_MODE_COUNT] {
    let eigenvalues = symmetric_eigenvalues_4(gram);
    let mut singular_values = eigenvalues.map(|value| value.max(0.0).sqrt());
    singular_values.sort_by(|lhs, rhs| rhs.partial_cmp(lhs).unwrap_or(std::cmp::Ordering::Equal));
    singular_values
}

// Fixed-size Jacobi rotations require coordinated symmetric row/column updates;
// the source-response singular-value tests compare this kernel with dense references.
#[allow(clippy::needless_range_loop)]
fn symmetric_eigenvalues_4(mut matrix: [[f64; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT]) -> [f64; 4] {
    for _ in 0..64 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_offdiag = 0.0_f64;
        for i in 0..SOURCE_MODE_COUNT {
            for j in i + 1..SOURCE_MODE_COUNT {
                let value = matrix[i][j].abs();
                if value > max_offdiag {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max_offdiag <= 1e-18 {
            break;
        }
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..SOURCE_MODE_COUNT {
            if k == p || k == q {
                continue;
            }
            let akp = matrix[k][p];
            let akq = matrix[k][q];
            matrix[k][p] = c * akp - s * akq;
            matrix[p][k] = matrix[k][p];
            matrix[k][q] = s * akp + c * akq;
            matrix[q][k] = matrix[k][q];
        }
        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
    }
    [matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][3]]
}

fn split_exact_b_observation_indices(
    dimension: usize,
    train_fraction: f64,
    heldout_count: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if dimension < 2 {
        return Err("exact-B observation split requires at least two rows".to_string());
    }
    let mut indices = (0..dimension).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    let mut train_count = (train_fraction * dimension as f64).ceil() as usize;
    train_count = train_count.clamp(1, dimension - 1);
    let mut training = indices[..train_count].to_vec();
    let mut heldout = indices[train_count..].to_vec();
    if heldout_count > 0 && heldout.len() > heldout_count {
        heldout.truncate(heldout_count);
    }
    training.sort_unstable();
    heldout.sort_unstable();
    Ok((training, heldout))
}

fn resolve_exact_b_observation_indices(
    config: &ToroidalExactBRecoveryConfig,
    system: &ReducedLinearPdeAssembly,
    source_modes: &[Vec<f64>],
    observation_operator: &SparseTripletMatrix,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if let Some(override_indices) = &config.observation_index_override {
        validate_exact_b_observation_index_override(
            override_indices,
            observation_operator.nrows(),
        )?;
        return Ok((
            override_indices.training_indices.clone(),
            override_indices.heldout_indices.clone(),
        ));
    }

    match config.observation_mode {
        ToroidalExactBObservationMode::SourceDesignedFluxes => {
            split_source_designed_flux_observation_indices(
                system,
                source_modes,
                observation_operator,
                config.base.prior_precision,
                config.base.pde_variance,
                config.observation_train_fraction,
                config.heldout_count,
            )
        }
        _ => split_exact_b_observation_indices(
            observation_operator.nrows(),
            config.observation_train_fraction,
            config.heldout_count,
            config.observation_seed,
        ),
    }
}

fn validate_exact_b_observation_index_override(
    override_indices: &ToroidalExactBObservationIndexOverride,
    dimension: usize,
) -> Result<(), String> {
    if dimension == 0 {
        return Err("exact-B explicit observation indices require at least one row".to_string());
    }
    if override_indices.training_indices.is_empty() {
        return Err("exact-B explicit training observation indices must be nonempty".to_string());
    }
    if override_indices.heldout_indices.is_empty() {
        return Err("exact-B explicit heldout observation indices must be nonempty".to_string());
    }
    validate_exact_b_observation_index_set(
        "training",
        &override_indices.training_indices,
        dimension,
    )?;
    validate_exact_b_observation_index_set(
        "heldout",
        &override_indices.heldout_indices,
        dimension,
    )?;

    let training = override_indices
        .training_indices
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    if let Some(row) = override_indices
        .heldout_indices
        .iter()
        .copied()
        .find(|row| training.contains(row))
    {
        return Err(format!(
            "exact-B explicit observation row {row} appears in both training and heldout sets"
        ));
    }
    Ok(())
}

fn validate_exact_b_observation_index_set(
    label: &str,
    indices: &[usize],
    dimension: usize,
) -> Result<(), String> {
    let mut seen = HashSet::new();
    for &row in indices {
        if row >= dimension {
            return Err(format!(
                "exact-B explicit {label} observation row {row} is out of bounds for {dimension} rows"
            ));
        }
        if !seen.insert(row) {
            return Err(format!(
                "exact-B explicit {label} observation row {row} is duplicated"
            ));
        }
    }
    Ok(())
}

fn split_source_designed_flux_observation_indices(
    system: &ReducedLinearPdeAssembly,
    source_modes: &[Vec<f64>],
    observation_operator: &SparseTripletMatrix,
    prior_precision: f64,
    pde_variance: f64,
    train_fraction: f64,
    heldout_count: usize,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let dimension = observation_operator.nrows();
    if dimension < 2 {
        return Err("source-designed flux observations require at least two rows".to_string());
    }
    let source_responses = exact_b_source_observation_responses(
        system,
        source_modes,
        observation_operator,
        prior_precision,
        pde_variance,
    )?;
    let mut train_count = (train_fraction * dimension as f64).ceil() as usize;
    train_count = train_count.max(SOURCE_MODE_COUNT).min(dimension - 1).max(1);
    let training = select_source_designed_flux_rows(&source_responses, train_count)?;
    let training_set = training.iter().copied().collect::<HashSet<_>>();
    let mut heldout = (0..dimension)
        .filter(|row| !training_set.contains(row))
        .collect::<Vec<_>>();
    if heldout_count > 0 && heldout.len() > heldout_count {
        heldout.truncate(heldout_count);
    }
    Ok((training, heldout))
}

fn select_source_designed_flux_rows(
    source_responses: &[Vec<f64>],
    train_count: usize,
) -> Result<Vec<usize>, String> {
    if source_responses.len() != SOURCE_MODE_COUNT {
        return Err(format!(
            "expected {SOURCE_MODE_COUNT} source response columns, found {}",
            source_responses.len()
        ));
    }
    let dimension = source_responses.first().map(Vec::len).unwrap_or(0);
    if dimension == 0 {
        return Err("source-designed flux response matrix has no rows".to_string());
    }
    if train_count == 0 || train_count > dimension {
        return Err(format!(
            "source-designed flux train count {train_count} must be in 1..={dimension}"
        ));
    }
    if source_responses
        .iter()
        .any(|response| response.len() != dimension)
    {
        return Err("source-designed flux response columns must have equal length".to_string());
    }

    let mut gram = [[0.0; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT];
    let mut selected = Vec::with_capacity(train_count);
    let mut selected_set = HashSet::new();
    for _ in 0..train_count {
        let mut best = None;
        for row in 0..dimension {
            if selected_set.contains(&row) {
                continue;
            }
            let candidate_gram = source_design_gram_with_row(gram, source_responses, row);
            let score = source_design_score(candidate_gram);
            let replace = best
                .as_ref()
                .map(|(best_score, best_row, _)| {
                    source_design_score_is_better(score, row, *best_score, *best_row)
                })
                .unwrap_or(true);
            if replace {
                best = Some((score, row, candidate_gram));
            }
        }
        let Some((_, row, next_gram)) = best else {
            return Err("failed to select a source-designed flux row".to_string());
        };
        selected.push(row);
        selected_set.insert(row);
        gram = next_gram;
    }
    selected.sort_unstable();
    Ok(selected)
}

fn source_design_score_is_better(score: f64, row: usize, best_score: f64, best_row: usize) -> bool {
    let tolerance = 16.0 * f64::EPSILON * score.abs().max(best_score.abs()).max(1.0);
    score > best_score + tolerance || ((score - best_score).abs() <= tolerance && row < best_row)
}

fn source_design_gram_with_row(
    mut gram: [[f64; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT],
    source_responses: &[Vec<f64>],
    row: usize,
) -> [[f64; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT] {
    for i in 0..SOURCE_MODE_COUNT {
        for j in i..SOURCE_MODE_COUNT {
            let value = source_responses[i][row] * source_responses[j][row];
            gram[i][j] += value;
            if i != j {
                gram[j][i] += value;
            }
        }
    }
    gram
}

fn source_design_score(gram: [[f64; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT]) -> f64 {
    let singular_values = source_response_singular_values(gram);
    let largest = singular_values[0].max(EPS);
    let threshold = largest * 1e-10;
    let rank = singular_values
        .iter()
        .filter(|value| **value > threshold)
        .count();
    let ridge = (largest * largest * 1e-24).max(1e-300);
    let logdet = singular_values
        .iter()
        .map(|value| (value * value + ridge).ln())
        .sum::<f64>();
    let smallest = singular_values[SOURCE_MODE_COUNT - 1].max(1e-300);
    let condition_penalty = (largest / smallest).ln();
    rank as f64 * 1e9 + logdet - condition_penalty
}

fn exact_b_probe_roles(
    dimension: usize,
    training_indices: &[usize],
    heldout_indices: &[usize],
) -> Vec<String> {
    let training = training_indices.iter().copied().collect::<HashSet<_>>();
    let heldout = heldout_indices.iter().copied().collect::<HashSet<_>>();
    (0..dimension)
        .map(|index| {
            if training.contains(&index) {
                "train".to_string()
            } else if heldout.contains(&index) {
                "heldout".to_string()
            } else {
                "unused".to_string()
            }
        })
        .collect()
}

fn build_exact_b_observation_system(
    topology: &Complex,
    coords: &MeshCoords,
    mode: ToroidalExactBObservationMode,
    geometry: ToroidalInductorGeometry,
    surface_flux_azimuth_count: usize,
) -> Result<ToroidalExactBObservationSystem, String> {
    match mode {
        ToroidalExactBObservationMode::CellMagneticComponents => {
            let operator = build_full_magnetic_flux_density_triplet_operator(topology, coords)?;
            let roles = vec!["unused".to_string(); operator.nrows()];
            Ok(ToroidalExactBObservationSystem {
                probes: exact_b_cell_probe_rows(topology, coords, &roles)?,
                operator,
            })
        }
        ToroidalExactBObservationMode::SurfaceFluxes
        | ToroidalExactBObservationMode::SourceDesignedFluxes => {
            let specs = default_toroidal_flux_sensor_specs(geometry, surface_flux_azimuth_count);
            build_exact_b_surface_flux_observation_system(topology, coords, &specs)
        }
    }
}

fn build_full_magnetic_flux_density_triplet_operator(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<SparseTripletMatrix, String> {
    let operator = build_full_magnetic_flux_density_operator_3d(topology, coords)?;
    sparse_row_operator_to_triplet(&operator)
}

fn build_reduced_magnetic_flux_density_triplet_operator(
    topology: &Complex,
    coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
) -> Result<SparseTripletMatrix, String> {
    let operator =
        build_reduced_magnetic_flux_density_operator_3d(topology, coords, model.layout())?;
    sparse_row_operator_to_triplet(&operator)
}

fn build_exact_b_surface_flux_observation_system(
    topology: &Complex,
    coords: &MeshCoords,
    specs: &[ToroidalFluxSensorSpec],
) -> Result<ToroidalExactBObservationSystem, String> {
    let face_rows = csr_rows(&FeecCsr::from(&topology.exterior_derivative_operator(1)));
    let mut operator = SparseTripletMatrix::new(specs.len(), topology.nsimplices(1));
    let mut probes = Vec::with_capacity(specs.len());
    for (sensor_index, spec) in specs.iter().enumerate() {
        let sensor_operator = build_oriented_flux_patch_operator(
            topology,
            coords,
            &face_rows,
            spec.center,
            spec.normal,
            spec.patch_radius,
            spec.normal_half_width,
        )?;
        for (_, col, value) in sensor_operator.triplet_iter() {
            operator.push(sensor_index, col, value);
        }
        probes.push(ToroidalExactBProbeRow {
            name: spec.name.clone(),
            cell_index: sensor_index,
            component: 0,
            x: spec.center[0],
            y: spec.center[1],
            z: spec.center[2],
            role: "unused".to_string(),
        });
    }
    Ok(ToroidalExactBObservationSystem { operator, probes })
}

fn exact_b_cell_probe_rows(
    topology: &Complex,
    coords: &MeshCoords,
    roles: &[String],
) -> Result<Vec<ToroidalExactBProbeRow>, String> {
    let expected = 3 * topology.nsimplices(3);
    if roles.len() != expected {
        return Err(format!(
            "exact-B probe role count {} must match 3 * cell count {}",
            roles.len(),
            expected
        ));
    }
    let mut rows = Vec::with_capacity(expected);
    for (cell_index, cell) in topology.cells().handle_iter().enumerate() {
        let point = coord_vec_to_point3(cell.coord_simplex(coords).barycenter());
        for component in 0..3 {
            let row = 3 * cell_index + component;
            rows.push(ToroidalExactBProbeRow {
                name: exact_b_probe_name(cell_index, component),
                cell_index,
                component,
                x: point[0],
                y: point[1],
                z: point[2],
                role: roles[row].clone(),
            });
        }
    }
    Ok(rows)
}

fn exact_b_probe_name(cell_index: usize, component: usize) -> String {
    format!("cell_{cell_index:06}_b{}", component_label(component))
}

fn component_label(component: usize) -> &'static str {
    match component {
        0 => "x",
        1 => "y",
        2 => "z",
        _ => "unknown",
    }
}

fn load_exact_b_reference_observations(
    path: &Path,
    probes: &[ToroidalExactBProbeRow],
    training_indices: &[usize],
    heldout_indices: &[usize],
) -> Result<Vec<f64>, String> {
    let resolved = resolve_workspace_path(path);
    let contents = fs::read_to_string(&resolved).map_err(|err| {
        format!(
            "failed to read exact-B reference observation CSV `{}`: {err}",
            resolved.display()
        )
    })?;
    let mut lines = contents.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| "exact-B reference observation CSV is empty".to_string())?;
    let header_cols = header
        .split(',')
        .map(|value| value.trim())
        .collect::<Vec<_>>();
    let name_col = header_cols
        .iter()
        .position(|name| *name == "name")
        .ok_or_else(|| "exact-B reference CSV header must contain `name`".to_string())?;
    let value_col = header_cols
        .iter()
        .position(|name| *name == "value")
        .ok_or_else(|| "exact-B reference CSV header must contain `value`".to_string())?;
    let mut values_by_name = BTreeMap::new();
    for (line_index, line) in lines.enumerate() {
        let cols = line
            .split(',')
            .map(|value| value.trim())
            .collect::<Vec<_>>();
        if cols.len() <= name_col.max(value_col) {
            return Err(format!(
                "exact-B reference CSV `{}` line {} has too few columns",
                resolved.display(),
                line_index + 2
            ));
        }
        let name = cols[name_col];
        let value = cols[value_col].parse::<f64>().map_err(|err| {
            format!(
                "exact-B reference CSV `{}` line {} has invalid value `{}`: {err}",
                resolved.display(),
                line_index + 2,
                cols[value_col]
            )
        })?;
        if !value.is_finite() {
            return Err(format!(
                "exact-B reference CSV `{}` line {} has non-finite value",
                resolved.display(),
                line_index + 2
            ));
        }
        values_by_name.insert(name.to_string(), value);
    }

    let mut values = vec![0.0; probes.len()];
    for index in training_indices.iter().chain(heldout_indices).copied() {
        let probe = probes
            .get(index)
            .ok_or_else(|| format!("exact-B probe index {index} is out of bounds"))?;
        values[index] = *values_by_name.get(&probe.name).ok_or_else(|| {
            format!(
                "exact-B reference CSV `{}` is missing probe `{}`",
                resolved.display(),
                probe.name
            )
        })?;
    }
    Ok(values)
}

fn build_exact_b_measurement(
    observation_operator: &SparseTripletMatrix,
    reference_values: &[f64],
    indices: &[usize],
    variance: f64,
    synthetic_noise_seed: Option<u64>,
) -> Result<LinearGaussianMeasurementSpec, String> {
    if reference_values.len() != observation_operator.nrows() {
        return Err("exact-B reference value count must match observation row count".to_string());
    }
    if !variance.is_finite() || variance <= 0.0 {
        return Err("exact-B measurement variance must be finite and positive".to_string());
    }
    let selected = indices
        .iter()
        .copied()
        .enumerate()
        .map(|(row, source_row)| (source_row, row))
        .collect::<BTreeMap<_, _>>();
    let mut operator = SparseTripletMatrix::new(indices.len(), observation_operator.ncols());
    for (row, col, value) in observation_operator.triplet_iter() {
        if let Some(selected_row) = selected.get(&row) {
            operator.push(*selected_row, col, value);
        }
    }
    Ok(LinearGaussianMeasurementSpec {
        name: "toroidal_exact_b_sparse_observations".to_string(),
        operator,
        observations: indices
            .iter()
            .map(|index| {
                exact_b_training_observation_value(
                    reference_values[*index],
                    variance.sqrt(),
                    synthetic_noise_seed,
                    *index,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        bias: vec![0.0; indices.len()],
        variance,
    })
}

fn exact_b_training_observation_value(
    reference_value: f64,
    noise_std: f64,
    synthetic_noise_seed: Option<u64>,
    global_row: usize,
) -> Result<f64, String> {
    let Some(seed) = synthetic_noise_seed else {
        return Ok(reference_value);
    };
    if !noise_std.is_finite() || noise_std <= 0.0 {
        return Err(
            "exact-B synthetic observation noise std must be finite and positive".to_string(),
        );
    }
    let normal = Normal::new(0.0, noise_std)
        .map_err(|err| format!("failed to construct exact-B synthetic observation noise: {err}"))?;
    let mut rng = StdRng::seed_from_u64(exact_b_observation_noise_row_seed(seed, global_row));
    Ok(reference_value + normal.sample(&mut rng))
}

fn exact_b_observation_noise_row_seed(seed: u64, global_row: usize) -> u64 {
    let mut value = seed ^ (global_row as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn build_exact_b_heldout_derived_quantities(
    observation_operator: &SparseTripletMatrix,
    probes: &[ToroidalExactBProbeRow],
    heldout_indices: &[usize],
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    if probes.len() != observation_operator.nrows() {
        return Err("exact-B probe count must match observation row count".to_string());
    }
    heldout_indices
        .iter()
        .copied()
        .map(|index| {
            let probe = probes
                .get(index)
                .ok_or_else(|| format!("heldout probe index {index} is out of bounds"))?;
            let row = observation_operator
                .triplet_iter()
                .filter_map(|(row, col, value)| (row == index).then_some((col, value)))
                .collect::<Vec<_>>();
            Ok(LinearPdeDerivedQuantitySpec {
                name: exact_b_derived_name(&probe.name),
                operator: SparseRowOperator::new(observation_operator.ncols(), vec![row])
                    .map_err(|err| err.to_string())?,
            })
        })
        .collect()
}

fn exact_b_derived_name(probe_name: &str) -> String {
    format!("exact_b::{probe_name}")
}

fn exact_b_heldout_prediction_rows(
    result: &LinearPdeUqResult,
    probes: &[ToroidalExactBProbeRow],
    predictions: &[f64],
    reference_values: &[f64],
    heldout_indices: &[usize],
    observation_noise_std: f64,
    synthetic_heldout_noise_seed: Option<u64>,
) -> Result<Vec<ToroidalExactBHeldoutPredictionRow>, String> {
    heldout_indices
        .iter()
        .copied()
        .map(|index| {
            let probe = probes
                .get(index)
                .ok_or_else(|| format!("heldout probe index {index} is out of bounds"))?;
            let variance = result
                .derived_variances
                .get(&exact_b_derived_name(&probe.name))
                .ok_or_else(|| format!("missing heldout variance for `{}`", probe.name))?;
            if variance.posterior_variance.len() != 1 {
                return Err(format!(
                    "heldout variance for `{}` must be scalar",
                    probe.name
                ));
            }
            let truth = reference_values[index];
            let prediction = predictions[index];
            let residual = prediction - truth;
            let posterior_sd = variance.posterior_variance[0].max(0.0).sqrt();
            let noisy_observation = exact_b_training_observation_value(
                truth,
                observation_noise_std,
                synthetic_heldout_noise_seed,
                index,
            )?;
            let noisy_residual = prediction - noisy_observation;
            let predictive_sd = (variance.posterior_variance[0]
                + observation_noise_std * observation_noise_std)
                .max(0.0)
                .sqrt();
            let standardized_residual = residual / posterior_sd.max(EPS);
            let noisy_standardized_residual = noisy_residual / predictive_sd.max(EPS);
            let lower95 = prediction - 1.96 * posterior_sd;
            let upper95 = prediction + 1.96 * posterior_sd;
            let noisy_lower95 = prediction - 1.96 * predictive_sd;
            let noisy_upper95 = prediction + 1.96 * predictive_sd;
            Ok(ToroidalExactBHeldoutPredictionRow {
                name: probe.name.clone(),
                truth,
                noisy_observation,
                prediction,
                residual,
                noisy_residual,
                posterior_sd,
                predictive_sd,
                standardized_residual,
                noisy_standardized_residual,
                lower95,
                upper95,
                covered95: truth >= lower95 && truth <= upper95,
                noisy_lower95,
                noisy_upper95,
                noisy_covered95: noisy_observation >= noisy_lower95
                    && noisy_observation <= noisy_upper95,
            })
        })
        .collect()
}

fn exact_b_source_posterior_rows(
    result: &LinearPdeUqResult,
    truth_eta: &[f64; SOURCE_MODE_COUNT],
    source_prior_std: f64,
) -> Result<Vec<ToroidalExactBSourcePosteriorRow>, String> {
    let latent = result
        .latent_inputs
        .iter()
        .find(|input| input.name == SOURCE_DISCREPANCY_INPUT)
        .ok_or_else(|| "missing exact-B source discrepancy posterior block".to_string())?;
    if latent.mean.len() != SOURCE_MODE_COUNT || latent.variance.len() != SOURCE_MODE_COUNT {
        return Err(format!(
            "source discrepancy posterior block must have {SOURCE_MODE_COUNT} entries"
        ));
    }
    let prior_variance = source_prior_std * source_prior_std;
    Ok((0..SOURCE_MODE_COUNT)
        .map(|mode_index| {
            let posterior_mean = latent.mean[mode_index];
            let posterior_variance = latent.variance[mode_index];
            ToroidalExactBSourcePosteriorRow {
                mode_index,
                prior_mean: 0.0,
                prior_variance,
                truth: truth_eta[mode_index],
                posterior_mean,
                posterior_variance,
                error: posterior_mean - truth_eta[mode_index],
            }
        })
        .collect())
}

fn prediction_rmse_exact_b(rows: &[ToroidalExactBHeldoutPredictionRow]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    (rows
        .iter()
        .map(|row| row.residual * row.residual)
        .sum::<f64>()
        / rows.len() as f64)
        .sqrt()
}

fn noisy_prediction_rmse_exact_b(rows: &[ToroidalExactBHeldoutPredictionRow]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    (rows
        .iter()
        .map(|row| row.noisy_residual * row.noisy_residual)
        .sum::<f64>()
        / rows.len() as f64)
        .sqrt()
}

fn prediction_nlpd_exact_b(rows: &[ToroidalExactBHeldoutPredictionRow]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    rows.iter()
        .map(|row| {
            let variance = row.posterior_sd.max(EPS).powi(2);
            0.5 * ((row.residual * row.residual) / variance + (2.0 * PI * variance).ln())
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn noisy_prediction_nlpd_exact_b(rows: &[ToroidalExactBHeldoutPredictionRow]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    rows.iter()
        .map(|row| {
            let variance = row.predictive_sd.max(EPS).powi(2);
            0.5 * ((row.noisy_residual * row.noisy_residual) / variance
                + (2.0 * PI * variance).ln())
        })
        .sum::<f64>()
        / rows.len() as f64
}

#[derive(Debug, Clone, Copy)]
struct ExactBPredictionSummaryMetrics {
    mean_posterior_sd: f64,
    max_abs_z: f64,
    rms_z: f64,
    mean_abs_residual: f64,
    mean_predictive_sd: f64,
    noisy_max_abs_z: f64,
    noisy_rms_z: f64,
    noisy_mean_abs_residual: f64,
}

fn exact_b_prediction_summary_metrics(
    rows: &[ToroidalExactBHeldoutPredictionRow],
) -> ExactBPredictionSummaryMetrics {
    if rows.is_empty() {
        return ExactBPredictionSummaryMetrics {
            mean_posterior_sd: f64::NAN,
            max_abs_z: f64::NAN,
            rms_z: f64::NAN,
            mean_abs_residual: f64::NAN,
            mean_predictive_sd: f64::NAN,
            noisy_max_abs_z: f64::NAN,
            noisy_rms_z: f64::NAN,
            noisy_mean_abs_residual: f64::NAN,
        };
    }
    let count = rows.len() as f64;
    ExactBPredictionSummaryMetrics {
        mean_posterior_sd: rows.iter().map(|row| row.posterior_sd).sum::<f64>() / count,
        max_abs_z: rows
            .iter()
            .map(|row| row.standardized_residual.abs())
            .fold(0.0, f64::max),
        rms_z: (rows
            .iter()
            .map(|row| row.standardized_residual * row.standardized_residual)
            .sum::<f64>()
            / count)
            .sqrt(),
        mean_abs_residual: rows.iter().map(|row| row.residual.abs()).sum::<f64>() / count,
        mean_predictive_sd: rows.iter().map(|row| row.predictive_sd).sum::<f64>() / count,
        noisy_max_abs_z: rows
            .iter()
            .map(|row| row.noisy_standardized_residual.abs())
            .fold(0.0, f64::max),
        noisy_rms_z: (rows
            .iter()
            .map(|row| row.noisy_standardized_residual * row.noisy_standardized_residual)
            .sum::<f64>()
            / count)
            .sqrt(),
        noisy_mean_abs_residual: rows.iter().map(|row| row.noisy_residual.abs()).sum::<f64>()
            / count,
    }
}

fn toroidal_observation_residual_dimension(
    model: &ReducedVectorPotentialMagnetostatic3d,
    mode: ToroidalPdeObservationMode,
) -> usize {
    match mode {
        ToroidalPdeObservationMode::WeakGalerkinRows => model.residual_dimension(),
        ToroidalPdeObservationMode::LocalMagneticStrongCells => 3 * model.num_elements(),
    }
}

fn build_toroidal_observation_residual<'a>(
    config: &NonlinearToroidalConfig,
    topology: &Complex,
    coords: &MeshCoords,
    model: &'a ReducedVectorPotentialMagnetostatic3d,
    observation_rows_total: usize,
) -> Result<ToroidalObservationBuild<'a>, String> {
    match config.pde_observation_mode {
        ToroidalPdeObservationMode::WeakGalerkinRows => {
            let selected_rows =
                select_residual_rows(model.residual_dimension(), &config.residual_selection)?;
            let noise = build_toroidal_residual_noise(
                model,
                &selected_rows,
                config.residual_weighting,
                config.pde_variance,
            )?;
            let rows_used = selected_rows.len();
            Ok(ToroidalObservationBuild {
                model: ToroidalObservationResidualModel::Weak {
                    model,
                    rows: selected_rows,
                },
                noise,
                rows_used,
                rows_total: model.residual_dimension(),
            })
        }
        ToroidalPdeObservationMode::LocalMagneticStrongCells => {
            if config.residual_weighting != ToroidalResidualWeighting::Euclidean {
                return Err(
                    "local magnetic strong-cell probes currently support Euclidean probe weighting only"
                        .to_string(),
                );
            }
            let requested_rows =
                requested_rows_from_selection(observation_rows_total, &config.residual_selection)?;
            let requested_cells = if requested_rows == 0 {
                0
            } else {
                requested_rows.div_ceil(3).clamp(1, model.num_elements())
            };
            let selection = select_stratified_toroidal_cells(
                topology,
                coords,
                config.geometry,
                requested_cells,
                selection_seed(&config.residual_selection),
            )?;
            let probe = LocalMagneticStrongProbe3d::from_vector_potential_model(
                model,
                selection.cells.clone(),
            )?;
            let noise = build_local_strong_probe_noise(&selection.weights, config.pde_variance)?;
            Ok(ToroidalObservationBuild {
                rows_used: probe.residual_dimension(),
                rows_total: observation_rows_total,
                model: ToroidalObservationResidualModel::LocalStrong(Box::new(probe)),
                noise,
            })
        }
    }
}

fn requested_rows_from_selection(
    residual_dimension: usize,
    selection: &ToroidalResidualSelection,
) -> Result<usize, String> {
    match selection {
        ToroidalResidualSelection::Full => Ok(residual_dimension),
        ToroidalResidualSelection::Stride { step } => {
            if *step == 0 {
                return Err("residual stride must be at least one".to_string());
            }
            Ok(residual_dimension.div_ceil(*step))
        }
        ToroidalResidualSelection::ShuffledStride { step, .. } => {
            if *step == 0 {
                return Err("shuffled residual stride must be at least one".to_string());
            }
            Ok(residual_dimension.div_ceil(*step))
        }
        ToroidalResidualSelection::ShuffledCount { count, .. } => {
            if *count > residual_dimension {
                return Err(format!(
                    "shuffled residual count {count} exceeds residual dimension {residual_dimension}"
                ));
            }
            Ok(*count)
        }
    }
}

fn selection_seed(selection: &ToroidalResidualSelection) -> u64 {
    match selection {
        ToroidalResidualSelection::Full | ToroidalResidualSelection::Stride { .. } => 0xC311,
        ToroidalResidualSelection::ShuffledStride { seed, .. }
        | ToroidalResidualSelection::ShuffledCount { seed, .. } => *seed,
    }
}

fn build_local_strong_probe_noise(
    cell_weights: &[f64],
    variance: f64,
) -> Result<ToroidalResidualNoise, String> {
    if !variance.is_finite() || variance <= 0.0 {
        return Err("local strong-probe residual variance must be finite and positive".to_string());
    }
    let dimension = 3 * cell_weights.len();
    if dimension == 0 {
        return Ok(ToroidalResidualNoise {
            noise: GaussianNoiseModel::ScalarVariance(variance),
            normalization_scale: None,
        });
    }
    let all_unit = cell_weights
        .iter()
        .all(|weight| (*weight - 1.0).abs() <= 1e-12);
    if all_unit {
        return Ok(ToroidalResidualNoise {
            noise: GaussianNoiseModel::ScalarVariance(variance),
            normalization_scale: None,
        });
    }
    let mut precision = SparseTripletMatrix::new(dimension, dimension);
    for (cell_offset, weight) in cell_weights.iter().copied().enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(format!(
                "local strong-probe precision weight must be finite and positive, got {weight:.6e}"
            ));
        }
        for component in 0..3 {
            let row = 3 * cell_offset + component;
            precision.push(row, row, weight / variance);
        }
    }
    Ok(ToroidalResidualNoise {
        noise: GaussianNoiseModel::Precision(precision),
        normalization_scale: None,
    })
}

fn weighted_residual_norm(residual: &[f64], noise: &GaussianNoiseModel) -> Result<f64, String> {
    match noise {
        GaussianNoiseModel::ScalarVariance(variance) => {
            if !variance.is_finite() || *variance <= 0.0 {
                return Err("scalar residual variance must be finite and positive".to_string());
            }
            Ok((residual.iter().map(|value| value * value).sum::<f64>() / variance).sqrt())
        }
        GaussianNoiseModel::Precision(precision) => {
            if precision.nrows() != residual.len() || precision.ncols() != residual.len() {
                return Err(format!(
                    "residual precision dimension {}x{} must match residual length {}",
                    precision.nrows(),
                    precision.ncols(),
                    residual.len()
                ));
            }
            let mut quadratic = 0.0;
            for (row, col, value) in precision.triplet_iter() {
                quadratic += residual[row] * value * residual[col];
            }
            Ok(quadratic.max(0.0).sqrt())
        }
    }
}

fn select_residual_rows(
    residual_dimension: usize,
    selection: &ToroidalResidualSelection,
) -> Result<Vec<usize>, String> {
    if residual_dimension == 0 {
        return Err("toroidal residual selection requires a nonempty residual".to_string());
    }
    match selection {
        ToroidalResidualSelection::Full => Ok((0..residual_dimension).collect()),
        ToroidalResidualSelection::Stride { step } => {
            if *step == 0 {
                return Err("residual stride must be at least one".to_string());
            }
            let mut rows = (0..residual_dimension).step_by(*step).collect::<Vec<_>>();
            if rows.last().copied() != Some(residual_dimension - 1) {
                rows.push(residual_dimension - 1);
            }
            Ok(rows)
        }
        ToroidalResidualSelection::ShuffledStride { step, seed } => {
            if *step == 0 {
                return Err("shuffled residual stride must be at least one".to_string());
            }
            let count = residual_dimension.div_ceil(*step);
            let mut rows = (0..residual_dimension).collect::<Vec<_>>();
            let mut rng = StdRng::seed_from_u64(*seed);
            rows.shuffle(&mut rng);
            rows.truncate(count.max(1));
            rows.sort_unstable();
            Ok(rows)
        }
        ToroidalResidualSelection::ShuffledCount { count, seed } => {
            if *count > residual_dimension {
                return Err(format!(
                    "shuffled residual count {count} exceeds residual dimension {residual_dimension}"
                ));
            }
            let mut rows = (0..residual_dimension).collect::<Vec<_>>();
            let mut rng = StdRng::seed_from_u64(*seed);
            rows.shuffle(&mut rng);
            rows.truncate(*count);
            rows.sort_unstable();
            Ok(rows)
        }
    }
}

fn select_stratified_toroidal_cells(
    topology: &Complex,
    coords: &MeshCoords,
    geometry: ToroidalInductorGeometry,
    count: usize,
    seed: u64,
) -> Result<StratifiedCellSelection, String> {
    let cell_count = topology.nsimplices(3);
    if count > cell_count {
        return Err(format!(
            "requested {count} stratified cells but mesh has only {cell_count} cells"
        ));
    }
    if count == 0 {
        return Ok(StratifiedCellSelection {
            cells: Vec::new(),
            weights: Vec::new(),
        });
    }

    let mut strata = [Vec::<usize>::new(), Vec::new(), Vec::new()];
    for (cell_index, cell) in topology.cells().handle_iter().enumerate() {
        let point = coord_vec_to_point3(cell.coord_simplex(coords).barycenter());
        let radius = toroidal_radius_from_point(point, geometry);
        let stratum = if radius <= geometry.core_minor_radius {
            0
        } else if radius <= geometry.coil_minor_radius {
            1
        } else {
            2
        };
        strata[stratum].push(cell_index);
    }

    let nonempty = strata
        .iter()
        .enumerate()
        .filter_map(|(index, cells)| (!cells.is_empty()).then_some(index))
        .collect::<Vec<_>>();
    if nonempty.is_empty() {
        return Err(
            "stratified local probe selection requires at least one tetrahedron".to_string(),
        );
    }

    let mut counts = vec![0usize; strata.len()];
    let priority = [0usize, 1, 2];
    if count < nonempty.len() {
        for stratum in priority
            .iter()
            .copied()
            .filter(|stratum| !strata[*stratum].is_empty())
            .take(count)
        {
            counts[stratum] = 1;
        }
    } else {
        for stratum in &nonempty {
            counts[*stratum] = 1;
        }
        let assigned = counts.iter().sum::<usize>();
        let remaining = count - assigned;
        let total = cell_count.max(1) as f64;
        let mut remainders = nonempty
            .iter()
            .map(|stratum| {
                let ideal = remaining as f64 * strata[*stratum].len() as f64 / total;
                let add = ideal.floor() as usize;
                counts[*stratum] += add;
                (*stratum, ideal - add as f64)
            })
            .collect::<Vec<_>>();
        let mut assigned = counts.iter().sum::<usize>();
        remainders.sort_by(|lhs, rhs| {
            rhs.1
                .partial_cmp(&lhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| lhs.0.cmp(&rhs.0))
        });
        for (stratum, _) in &remainders {
            if assigned >= count {
                break;
            }
            if counts[*stratum] < strata[*stratum].len() {
                counts[*stratum] += 1;
                assigned += 1;
            }
        }
        while assigned < count {
            let mut progressed = false;
            for stratum in &nonempty {
                if counts[*stratum] < strata[*stratum].len() {
                    counts[*stratum] += 1;
                    assigned += 1;
                    progressed = true;
                    if assigned == count {
                        break;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
    }

    let mut selected = Vec::<(usize, f64)>::new();
    for (stratum, cells) in strata.iter_mut().enumerate() {
        let take = counts[stratum].min(cells.len());
        if take == 0 {
            continue;
        }
        let mut rng = StdRng::seed_from_u64(seed ^ ((stratum as u64 + 1) * 0x9E37_79B9));
        cells.shuffle(&mut rng);
        let multiplier = cells.len() as f64 / take as f64;
        selected.extend(cells.iter().take(take).map(|cell| {
            let handle = SimplexIdx::new(3, *cell).handle(topology);
            let volume = handle.coord_simplex(coords).vol();
            (*cell, multiplier * volume)
        }));
    }
    selected.sort_by_key(|(cell, _)| *cell);
    let (cells, weights): (Vec<_>, Vec<_>) = selected.into_iter().unzip();
    Ok(StratifiedCellSelection { cells, weights })
}

fn build_toroidal_residual_noise(
    model: &ReducedVectorPotentialMagnetostatic3d,
    residual_rows: &[usize],
    weighting: ToroidalResidualWeighting,
    variance: f64,
) -> Result<ToroidalResidualNoise, String> {
    match weighting {
        ToroidalResidualWeighting::Euclidean => Ok(ToroidalResidualNoise {
            noise: GaussianNoiseModel::ScalarVariance(variance),
            normalization_scale: None,
        }),
        ToroidalResidualWeighting::MassInverseTraceNormalized => {
            let full_mass_inverse = model.state_mass_inverse().ok_or_else(|| {
                "mass-inverse residual weighting requires FEEC state_mass_inverse".to_string()
            })?;
            let full_mass_inverse = feec_csr_to_core_triplet(full_mass_inverse);
            let selected_mass_inverse = if residual_rows.len() == model.residual_dimension() {
                full_mass_inverse.clone()
            } else {
                select_square_triplet_rows_cols(&full_mass_inverse, residual_rows)?
            };
            let normalized = trace_normalized_precision(&selected_mass_inverse, variance)?;
            Ok(ToroidalResidualNoise {
                noise: GaussianNoiseModel::Precision(normalized.precision),
                normalization_scale: Some(normalized.normalization_scale),
            })
        }
    }
}

fn sweep_variance_config(seed: u64) -> LinearPdeVarianceConfig {
    LinearPdeVarianceConfig {
        mode: LinearPdeVarianceMode::MonteCarlo,
        num_variance_probes: 1,
        variance_batch_count: 1,
        rng_seed: seed,
        ..LinearPdeVarianceConfig::default()
    }
}

fn build_toroidal_prior(
    config: &NonlinearToroidalConfig,
    topology: &Complex,
    coords: &MeshCoords,
    linear_model: &ReducedVectorPotentialMagnetostatic3d,
    linear_mean: Vec<f64>,
) -> Result<ToroidalPriorBuild, String> {
    match config.prior_mode {
        ToroidalPriorMode::WeakDiagonal => Ok(ToroidalPriorBuild {
            spec: zero_mean_diagonal_prior(
                linear_model.reduced_dimension(),
                config.prior_precision,
            ),
            kappa: 0.0,
            tau: 0.0,
            kappa_fallback_used: false,
        }),
        ToroidalPriorMode::LinearProxyMaternAlpha2 => {
            build_linear_proxy_matern_prior(config, topology, coords, linear_model, linear_mean)
        }
    }
}

fn build_linear_proxy_matern_prior(
    config: &NonlinearToroidalConfig,
    topology: &Complex,
    coords: &MeshCoords,
    linear_model: &ReducedVectorPotentialMagnetostatic3d,
    linear_mean: Vec<f64>,
) -> Result<ToroidalPriorBuild, String> {
    let evaluation =
        linear_model
            .source_free_residual_and_jacobian(&vec![0.0; linear_model.reduced_dimension()])?;
    let prior = build_reduced_linear_proxy_matern_alpha2_prior(
        topology,
        coords,
        linear_model.layout(),
        &feec_csr_to_core_triplet(&evaluation.jacobian),
        linear_mean,
        ReducedLinearProxyMaternAlpha2Config {
            kappa: config
                .linear_proxy_kappa
                .unwrap_or_else(|| default_reduced_linear_proxy_matern_kappa(coords)),
            tau: config.linear_proxy_tau,
            allow_kappa_fallback: config.linear_proxy_allow_kappa_fallback,
            ..ReducedLinearProxyMaternAlpha2Config::default()
        },
    )?;
    Ok(ToroidalPriorBuild {
        spec: prior.spec,
        kappa: prior.kappa,
        tau: prior.tau,
        kappa_fallback_used: prior.kappa_fallback_used,
    })
}

fn direct_reduced_hodge_linear_solution(
    model: &ReducedVectorPotentialMagnetostatic3d,
    prior_precision: f64,
    pde_variance: f64,
) -> Result<Vec<f64>, String> {
    let evaluation =
        model.source_free_residual_and_jacobian(&vec![0.0; model.reduced_dimension()])?;
    let prior = sparse_from_core(
        &zero_mean_diagonal_prior(model.reduced_dimension(), prior_precision).precision,
    );
    let observations = GmrfVector::from_vec(model.source().to_vec());
    let (precision, information) = apply_gaussian_observations(
        &prior,
        &feec_csr_to_gmrf(&evaluation.jacobian),
        &observations,
        None,
        pde_variance,
    );
    let factor = precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor direct reduced linear normal system: {err}"))?;
    factor
        .solve(&information)
        .map(|solution| solution.as_slice().to_vec())
        .map_err(|err| format!("failed to solve direct reduced linear normal system: {err}"))
}

fn solve_mixed_linear_reference(
    topology: &Complex,
    coords: &MeshCoords,
    source: &FeecVector,
    geometry: ToroidalInductorGeometry,
    nu_air: f64,
) -> Result<Cochain, String> {
    let metric = coords.to_edge_lengths(topology);
    let strong_state_dofs = sorted_boundary_dofs(topology, coords, 1, |point| {
        outer_boundary_predicate(point, geometry)
    })
    .into_iter()
    .collect::<std::collections::HashSet<_>>();
    let strong_aux_dofs = sorted_boundary_dofs(topology, coords, 0, |point| {
        outer_boundary_predicate(point, geometry)
    })
    .into_iter()
    .collect::<std::collections::HashSet<_>>();
    let strong_state_predicate =
        |sidx: manifold::topology::handle::KSimplexIdx| strong_state_dofs.contains(&sidx);
    let strong_aux_predicate =
        |sidx: manifold::topology::handle::KSimplexIdx| strong_aux_dofs.contains(&sidx);
    let zero_data = |_sidx: manifold::topology::handle::KSimplexIdx| 0.0;
    let weight = InnerProductWeightClosure::new(move |_| nu_air);
    let (_, a, _) = formoniq::problems::hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
        topology,
        &metric,
        None,
        source.clone(),
        1,
        1,
        coords,
        None,
        &weight,
        &strong_state_predicate,
        &zero_data,
        &strong_aux_predicate,
        &zero_data,
    );
    Ok(a)
}

fn compute_harmonic_coefficients(
    topology: &Complex,
    coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
    reduced_state: &[f64],
    geometry: ToroidalInductorGeometry,
    nu_air: f64,
) -> Result<Vec<f64>, String> {
    let metric = coords.to_edge_lengths(topology);
    let weight = InnerProductWeightClosure::new(move |_| nu_air);
    let galmats = MixedGalmats::compute_weighted(topology, &metric, 1, coords, None, &weight);
    let strong_aux_dofs = sorted_boundary_dofs(topology, coords, 0, |point| {
        outer_boundary_predicate(point, geometry)
    })
    .into_iter()
    .collect::<HashSet<_>>();
    let harmonics = hodge_laplace::solve_hodge_laplace_harmonics_with_galmats(
        topology,
        &galmats,
        1,
        1,
        Some(&|kidx| strong_aux_dofs.contains(&kidx)),
        None,
    );
    let full = FeecVector::from_vec(model.lift_reduced_state(reduced_state)?);
    let mass = FeecCsr::from(galmats.mass_u());
    let weighted_full = &mass * &full;
    Ok((0..harmonics.ncols())
        .map(|col| harmonics.column(col).dot(&weighted_full))
        .collect())
}

fn build_sensor_reports(
    topology: &Complex,
    coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
    nonlinear_state: &[f64],
    beta_zero_state: Option<&[f64]>,
    mixed_reference: Option<&Cochain>,
    geometry: ToroidalInductorGeometry,
) -> Result<Vec<ToroidalSensorReport>, String> {
    let nonlinear_full = model.lift_reduced_state(nonlinear_state)?;
    let beta_zero_full = beta_zero_state
        .map(|state| model.lift_reduced_state(state))
        .transpose()?;
    let face_rows = csr_rows(&FeecCsr::from(&topology.exterior_derivative_operator(1)));
    let sensors = [
        ("flux_inner", [geometry.major_radius - 1.05, 0.0, 0.00]),
        ("flux_top", [geometry.major_radius, 0.0, 1.05]),
        ("flux_outer", [geometry.major_radius + 1.05, 0.0, 0.00]),
    ];
    let mut reports = Vec::with_capacity(sensors.len());
    for (name, center) in sensors {
        let operator = build_flux_patch_operator(topology, coords, &face_rows, center, 0.45, 0.18)?;
        let nonlinear_value = apply_sparse_triplet(&operator, &nonlinear_full)[0];
        let beta_zero_value = beta_zero_full
            .as_ref()
            .map(|state| apply_sparse_triplet(&operator, state)[0]);
        let mixed_reference_value = mixed_reference
            .map(|reference| apply_sparse_triplet(&operator, reference.coeffs.as_slice())[0]);
        reports.push(ToroidalSensorReport {
            name: name.to_string(),
            nonlinear_value,
            beta_zero_value,
            mixed_reference_value,
        });
    }
    Ok(reports)
}

fn b_relative_error(
    topology: &Complex,
    _coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
    reduced_state: &[f64],
    reference: &Cochain,
) -> Result<f64, String> {
    let full = model.lift_reduced_state(reduced_state)?;
    let b = Cochain::new(1, FeecVector::from_vec(full)).dif(topology);
    Ok((&b.coeffs - &reference.clone().dif(topology).coeffs).norm()
        / reference.clone().dif(topology).coeffs.norm().max(1e-12))
}

fn build_flux_patch_operator(
    topology: &Complex,
    coords: &MeshCoords,
    face_rows: &[Vec<(usize, f64)>],
    center: [f64; 3],
    patch_radius: f64,
    y_half_width: f64,
) -> Result<SparseTripletMatrix, String> {
    build_oriented_flux_patch_operator(
        topology,
        coords,
        face_rows,
        center,
        [0.0, 1.0, 0.0],
        patch_radius,
        y_half_width,
    )
}

fn build_oriented_flux_patch_operator(
    topology: &Complex,
    coords: &MeshCoords,
    face_rows: &[Vec<(usize, f64)>],
    center: [f64; 3],
    normal: [f64; 3],
    patch_radius: f64,
    normal_half_width: f64,
) -> Result<SparseTripletMatrix, String> {
    let normal_norm =
        (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !normal_norm.is_finite() || normal_norm <= 1e-12 {
        return Err("flux sensor normal must be finite and nonzero".to_string());
    }
    let unit_normal = [
        normal[0] / normal_norm,
        normal[1] / normal_norm,
        normal[2] / normal_norm,
    ];
    let mut edge_weights = BTreeMap::<usize, f64>::new();
    let mut selected_face_count = 0usize;
    for (face_index, face_row) in face_rows.iter().enumerate().take(topology.nsimplices(2)) {
        let face = SimplexIdx::new(2, face_index).handle(topology);
        let face_coords = SimplexCoords::from_simplex_and_coords(&face, coords);
        let bary = face_coords.barycenter();
        let delta = [
            bary[0] - center[0],
            bary[1] - center[1],
            bary[2] - center[2],
        ];
        let signed_distance =
            delta[0] * unit_normal[0] + delta[1] * unit_normal[1] + delta[2] * unit_normal[2];
        let perpendicular_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
            - signed_distance * signed_distance;
        let radial = perpendicular_sq.max(0.0).sqrt();
        if radial > patch_radius || signed_distance.abs() > normal_half_width {
            continue;
        }

        let normal = face_normal(&face_coords);
        let face_norm =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if face_norm <= 1e-12 {
            continue;
        }
        let dot_normal =
            (normal[0] * unit_normal[0] + normal[1] * unit_normal[1] + normal[2] * unit_normal[2])
                / face_norm;
        let alignment = dot_normal.abs();
        if alignment < 0.55 {
            continue;
        }
        let sign = if dot_normal >= 0.0 { 1.0 } else { -1.0 };
        selected_face_count += 1;
        for (edge, value) in face_row {
            *edge_weights.entry(*edge).or_insert(0.0) += sign * *value;
        }
    }
    if selected_face_count == 0 {
        return Err(format!(
            "flux sensor patch at ({:.3}, {:.3}, {:.3}) selected no faces",
            center[0], center[1], center[2]
        ));
    }
    Ok(SparseTripletMatrix::from_triplets(
        1,
        topology.nsimplices(1),
        edge_weights
            .into_iter()
            .filter(|(_, value)| value.abs() > 1e-12)
            .map(|(col, value)| SparseTriplet { row: 0, col, value }),
    ))
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

fn reduce_full_edge_vector(
    layout: &formoniq::reduction::DofLayout,
    full: &FeecVector,
) -> Result<Vec<f64>, String> {
    if full.len() != layout.full_dimension {
        return Err(format!(
            "full edge vector length {} does not match layout dimension {}",
            full.len(),
            layout.full_dimension
        ));
    }
    Ok(layout
        .active_dofs
        .iter()
        .map(|&index| full[index])
        .collect())
}

fn csr_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if value.abs() > 1e-12 {
            rows[row].push((col, *value));
        }
    }
    rows
}

fn apply_sparse_triplet(operator: &SparseTripletMatrix, values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; operator.nrows()];
    for (row, col, value) in operator.triplet_iter() {
        out[row] += value * values[col];
    }
    out
}

fn build_sensor_measurement(
    sensors: &[ToroidalReducedFluxSensor],
    observations: &[f64],
    indices: &[usize],
    variance: f64,
) -> Result<LinearGaussianMeasurementSpec, String> {
    if observations.len() != sensors.len() {
        return Err("sensor observations must match sensor count".to_string());
    }
    let ncols = sensors
        .first()
        .map(|sensor| sensor.operator.ncols())
        .unwrap_or(0);
    let mut operator = SparseTripletMatrix::new(indices.len(), ncols);
    let mut selected_observations = Vec::with_capacity(indices.len());
    let mut bias = Vec::with_capacity(indices.len());
    for (row, sensor_index) in indices.iter().copied().enumerate() {
        let sensor = sensors
            .get(sensor_index)
            .ok_or_else(|| format!("sensor index {sensor_index} is outside sensor list"))?;
        if sensor.operator.ncols() != ncols
            || sensor.operator.nrows() != 1
            || sensor.bias.len() != 1
        {
            return Err(format!(
                "sensor `{}` is not a scalar reduced operator",
                sensor.spec.name
            ));
        }
        for (_, col, value) in sensor.operator.triplet_iter() {
            operator.push(row, col, value);
        }
        selected_observations.push(observations[sensor_index]);
        bias.push(sensor.bias[0]);
    }
    Ok(LinearGaussianMeasurementSpec {
        name: "toroidal_flux_sensors".to_string(),
        operator,
        observations: selected_observations,
        bias,
        variance,
    })
}

fn sensor_predictive_variances(
    sensors: &[ToroidalReducedFluxSensor],
    report: &NonlinearToroidalReport,
    observation_variance: f64,
) -> Result<Vec<f64>, String> {
    sensors
        .iter()
        .map(|sensor| {
            let variance = report
                .result
                .derived_variances
                .get(&sensor.spec.name)
                .ok_or_else(|| format!("missing sensor derived variance `{}`", sensor.spec.name))?;
            if variance.posterior_variance.len() != 1 {
                return Err(format!(
                    "sensor derived variance `{}` must be scalar",
                    sensor.spec.name
                ));
            }
            Ok((variance.posterior_variance[0] + observation_variance).max(0.0))
        })
        .collect()
}

fn cell_b_relative_error(
    topology: &Complex,
    coords: &MeshCoords,
    model: &ReducedVectorPotentialMagnetostatic3d,
    lhs: &[f64],
    rhs: &[f64],
) -> Result<f64, String> {
    let operator = build_reduced_magnetic_flux_density_triplet_operator(topology, coords, model)?;
    Ok(relative_error(
        &apply_sparse_triplet(&operator, lhs),
        &apply_sparse_triplet(&operator, rhs),
    ))
}

fn toroidal_cell_volumes(topology: &Complex, coords: &MeshCoords) -> Result<Vec<f64>, String> {
    topology
        .cells()
        .handle_iter()
        .map(|cell| {
            let volume = cell.coord_simplex(coords).vol();
            if volume.is_finite() && volume > 0.0 {
                Ok(volume)
            } else {
                Err(format!(
                    "toroidal cell {} has non-positive volume {volume}",
                    cell.kidx()
                ))
            }
        })
        .collect()
}

fn cell_major_b_weights(cell_volumes: &[f64]) -> GmrfVector {
    GmrfVector::from_iterator(
        3 * cell_volumes.len(),
        cell_volumes
            .iter()
            .flat_map(|volume| [*volume, *volume, *volume]),
    )
}

fn weighted_rms(
    values: &GmrfVector,
    weights: &GmrfVector,
    domain_measure: f64,
) -> Result<f64, String> {
    if values.len() != weights.len() {
        return Err(format!(
            "weighted RMS value count {} must match weight count {}",
            values.len(),
            weights.len()
        ));
    }
    if !domain_measure.is_finite() || domain_measure <= 0.0 {
        return Err(format!(
            "weighted RMS domain measure must be finite and positive, got {domain_measure}"
        ));
    }
    let mut sum = 0.0;
    for (index, (value, weight)) in values.iter().zip(weights.iter()).enumerate() {
        if !value.is_finite() || !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "weighted RMS found invalid entry {index}: value={value}, weight={weight}"
            ));
        }
        sum += *weight * *value * *value;
    }
    Ok((sum / domain_measure).sqrt())
}

fn relative_error(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / rhs
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1e-12)
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn rmse(lhs: &[f64], rhs: &[f64]) -> f64 {
    if lhs.is_empty() || lhs.len() != rhs.len() {
        return f64::NAN;
    }
    (lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / lhs.len() as f64)
        .sqrt()
}

fn indexed_rmse(predictions: &[f64], truth: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return f64::NAN;
    }
    (indices
        .iter()
        .map(|index| (predictions[*index] - truth[*index]).powi(2))
        .sum::<f64>()
        / indices.len() as f64)
        .sqrt()
}

fn indexed_mean_abs_z(
    predictions: &[f64],
    truth: &[f64],
    variances: &[f64],
    indices: &[usize],
) -> f64 {
    if indices.is_empty() {
        return f64::NAN;
    }
    indices
        .iter()
        .map(|index| {
            let std = variances[*index].max(1e-300).sqrt();
            ((predictions[*index] - truth[*index]) / std).abs()
        })
        .sum::<f64>()
        / indices.len() as f64
}

fn indexed_coverage(
    predictions: &[f64],
    truth: &[f64],
    variances: &[f64],
    indices: &[usize],
    sigma: f64,
) -> f64 {
    if indices.is_empty() {
        return f64::NAN;
    }
    let covered = indices
        .iter()
        .filter(|index| {
            let std = variances[**index].max(1e-300).sqrt();
            (predictions[**index] - truth[**index]).abs() <= sigma * std
        })
        .count();
    covered as f64 / indices.len() as f64
}

fn finite_min_max(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
        (min.min(value), max.max(value))
    })
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / finite.len() as f64;
    (mean, variance.sqrt())
}

fn mesh_bounding_box_diameter(coords: &MeshCoords) -> f64 {
    if coords.nvertices() == 0 {
        return 1.0;
    }
    let first = coords.coord(0);
    let mut min = vec![0.0; coords.dim()];
    let mut max = vec![0.0; coords.dim()];
    for d in 0..coords.dim() {
        min[d] = first[d];
        max[d] = first[d];
    }
    for vertex in 1..coords.nvertices() {
        let point = coords.coord(vertex);
        for d in 0..coords.dim() {
            min[d] = min[d].min(point[d]);
            max[d] = max[d].max(point[d]);
        }
    }
    min.iter()
        .zip(max.iter())
        .map(|(min, max)| (max - min).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[allow(dead_code)]
fn sparse_row_operator_from_triplet(
    matrix: &SparseTripletMatrix,
) -> Result<SparseRowOperator, String> {
    triplet_to_sparse_row_operator(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn matrix_entry_case(matrix: &SparseTripletMatrix, row: usize, col: usize) -> f64 {
        matrix
            .triplet_iter()
            .filter(|(r, c, _)| *r == row && *c == col)
            .map(|(_, _, value)| value)
            .sum()
    }

    fn relative_scalar_error(lhs: f64, rhs: f64) -> f64 {
        (lhs - rhs).abs() / rhs.abs().max(1e-300)
    }

    #[test]
    fn nonlinear_toroidal_weiland_shuffled_count_selection_is_exact_and_deterministic() {
        let rows_a = select_residual_rows(
            17,
            &ToroidalResidualSelection::ShuffledCount {
                count: 5,
                seed: 123,
            },
        )
        .expect("shuffled count should select rows");
        let rows_b = select_residual_rows(
            17,
            &ToroidalResidualSelection::ShuffledCount {
                count: 5,
                seed: 123,
            },
        )
        .expect("shuffled count should be deterministic");
        assert_eq!(rows_a, rows_b);
        assert_eq!(rows_a.len(), 5);
        assert!(rows_a.windows(2).all(|window| window[0] < window[1]));
        assert!(rows_a.iter().all(|row| *row < 17));
        assert!(select_residual_rows(
            17,
            &ToroidalResidualSelection::ShuffledCount {
                count: 18,
                seed: 123,
            },
        )
        .is_err());
    }

    fn exact_b_test_config(
        reference_mode: ToroidalExactBReferenceMode,
    ) -> ToroidalExactBRecoveryConfig {
        ToroidalExactBRecoveryConfig {
            base: ToroidalExactBBaseConfig {
                prior_precision: 1e-8,
                pde_variance: 1e-8,
                residual_selection: ToroidalResidualSelection::ShuffledCount {
                    count: 16,
                    seed: 42,
                },
                ..ToroidalExactBBaseConfig::default()
            },
            reference_mode,
            source_deltas: [0.0, 0.20, -0.10, 0.05],
            source_prior_std: 0.50,
            prior_kappa: 1.0,
            prior_tau: 1e-6,
            observation_train_fraction: 0.08,
            observation_noise_std: 1e-10,
            observation_seed: 12345,
            heldout_count: 12,
            write_outputs: false,
            output_dir: None,
            ..ToroidalExactBRecoveryConfig::default()
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_linear_source_input_has_mode_columns() {
        let config = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        let (topology, coords) = load_mesh(&config.base.mesh_path).expect("mesh should load");
        let boundary = outer_boundary(&topology, &coords, config.base.geometry);
        let linear_system =
            build_toroidal_exact_b_linear_system(&topology, &coords, boundary, &config)
                .expect("linear exact-B system should build");
        let source_modes = assemble_toroidal_source_modes(
            &topology,
            &coords,
            &linear_system.layout,
            config.base.geometry,
            config.base.nu_air,
        )
        .expect("source modes should assemble");
        let selected_rows = (0..linear_system.residual_dimension()).collect::<Vec<_>>();
        let selected_modes = source_modes
            .iter()
            .map(|mode| select_vector_entries(mode, &selected_rows))
            .collect::<Result<Vec<_>, _>>()
            .expect("source modes should select");
        let operator = build_source_discrepancy_operator(&selected_modes)
            .expect("source discrepancy operator should build");

        assert_eq!(operator.nrows(), selected_rows.len());
        assert_eq!(operator.ncols(), SOURCE_MODE_COUNT);
        for mode_index in 0..SOURCE_MODE_COUNT {
            assert!(
                operator
                    .triplet_iter()
                    .any(|(_, candidate_col, value)| candidate_col == mode_index && value != 0.0),
                "source mode column {mode_index} should be present"
            );
        }
    }

    #[test]
    fn toroidal_exact_b_observation_mode_labels_include_source_designed_fluxes() {
        assert_eq!(
            ToroidalExactBObservationMode::CellMagneticComponents.label(),
            "cell_magnetic_components"
        );
        assert_eq!(
            ToroidalExactBObservationMode::SurfaceFluxes.label(),
            "surface_fluxes"
        );
        assert_eq!(
            ToroidalExactBObservationMode::SourceDesignedFluxes.label(),
            "source_designed_fluxes"
        );
    }

    #[test]
    fn toroidal_exact_b_observation_mode_parses_source_designed_fluxes() {
        assert_eq!(
            "source-designed-flux"
                .parse::<ToroidalExactBObservationMode>()
                .expect("hyphenated source-designed flux mode should parse"),
            ToroidalExactBObservationMode::SourceDesignedFluxes
        );
        assert_eq!(
            "source_designed_fluxes"
                .parse::<ToroidalExactBObservationMode>()
                .expect("underscored plural source-designed flux mode should parse"),
            ToroidalExactBObservationMode::SourceDesignedFluxes
        );
        assert_eq!(
            "surface-flux"
                .parse::<ToroidalExactBObservationMode>()
                .expect("surface flux alias should still parse"),
            ToroidalExactBObservationMode::SurfaceFluxes
        );
        assert!("unknown-mode"
            .parse::<ToroidalExactBObservationMode>()
            .expect_err("unknown mode should be rejected")
            .contains("source-designed-flux"));
    }

    #[test]
    fn toroidal_exact_b_reference_solve_modes_are_labeled_in_stage_summary() {
        assert_eq!(
            ToroidalExactBReferenceSolveMode::RegularizedMap.label(),
            "regularized_map"
        );
        assert_eq!(
            ToroidalExactBReferenceSolveMode::DeterministicPde.label(),
            "deterministic_pde"
        );

        let summary = ToroidalExactBStageSummary {
            reference_mode: ToroidalExactBReferenceMode::PerturbedSource,
            reference_solve_mode: ToroidalExactBReferenceSolveMode::DeterministicPde,
            reference_solver_diagonal_shift: 1.0e-12,
            observation_mode: ToroidalExactBObservationMode::SourceDesignedFluxes,
            prior_tau: 1.0,
            prior_calibration_label: "test".to_string(),
            prior_calibration_nominal_b_rms: Some(2.0),
            prior_calibration_target_b_rms: Some(20.0),
            prior_calibration_multiplier: Some(10.0),
            active_dofs: 3,
            source_modes: SOURCE_MODE_COUNT,
            training_rows: 4,
            heldout_rows: 8,
            residual_rows_used: 3,
            residual_rows_total: 3,
            final_residual_norm: 0.0,
            train_rmse: 0.0,
            heldout_rmse: 0.0,
            heldout_nlpd: 0.0,
            heldout_covered95: 8,
            heldout_coverage_fraction: 1.0,
            heldout_mean_posterior_flux_sd: 1.0,
            heldout_max_abs_z: 0.0,
            heldout_rms_z: 0.0,
            heldout_mean_abs_residual: 0.0,
            heldout_noisy_rmse: 0.0,
            heldout_noisy_nlpd: 0.0,
            heldout_noisy_covered95: 8,
            heldout_noisy_coverage_fraction: 1.0,
            heldout_mean_predictive_sd: 1.0,
            heldout_noisy_max_abs_z: 0.0,
            heldout_noisy_rms_z: 0.0,
            heldout_noisy_mean_abs_residual: 0.0,
            b_relative_error: Some(0.0),
            posterior_factor_nnz: 9,
        };
        let csv = exact_b_stage_summary_csv(&summary);
        assert!(csv.starts_with("reference_mode,reference_solve_mode"));
        assert!(csv.contains("perturbed_source,deterministic_pde"));
        assert!(csv.contains("1.000000000000000"));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_deterministic_reference_solves_pde_system() {
        let mut config = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        config.reference_solve_mode = ToroidalExactBReferenceSolveMode::DeterministicPde;
        config.reference_solver_diagonal_shift =
            TOROIDAL_EXACT_B_DETERMINISTIC_REFERENCE_DIAGONAL_SHIFT;
        let (topology, coords) = load_mesh(&config.base.mesh_path).expect("mesh should load");
        let boundary = outer_boundary(&topology, &coords, config.base.geometry);
        let linear_system =
            build_toroidal_exact_b_linear_system(&topology, &coords, boundary, &config)
                .expect("linear exact-B system should build");
        let nominal_source_full =
            assemble_toroidal_source(&topology, &coords, config.base.geometry, config.base.nu_air);
        let nominal_source = reduce_full_edge_vector(&linear_system.layout, &nominal_source_full)
            .expect("nominal source should reduce");
        let source_modes = assemble_toroidal_source_modes(
            &topology,
            &coords,
            &linear_system.layout,
            config.base.geometry,
            config.base.nu_air,
        )
        .expect("source modes should assemble");
        let truth_source =
            source_with_sector_deltas(&nominal_source, &source_modes, &config.source_deltas)
                .expect("truth source should build");
        let state = deterministic_linear_system_solution(
            &linear_system,
            &truth_source,
            config.reference_solver_diagonal_shift,
        )
        .expect("deterministic reference solve should factor");

        let mut residual = linear_system.residual_bias.clone();
        for (row, col, value) in linear_system.operator.triplet_iter() {
            residual[row] += value * state[col];
        }
        for (entry, source) in residual.iter_mut().zip(&truth_source) {
            *entry -= *source;
        }
        for (entry, state_value) in residual.iter_mut().zip(&state) {
            *entry += config.reference_solver_diagonal_shift * *state_value;
        }
        let residual_norm = l2_norm(residual.as_slice());
        let source_norm = l2_norm(&truth_source).max(1.0);
        assert!(
            residual_norm / source_norm < 1.0e-8,
            "relative shifted deterministic PDE residual was {:.3e}",
            residual_norm / source_norm
        );
    }

    #[test]
    fn toroidal_exact_b_synthetic_observation_noise_is_deterministic() {
        let mut operator = SparseTripletMatrix::new(4, 4);
        for index in 0..4 {
            operator.push(index, index, 1.0);
        }
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![0, 2, 3];
        let variance = 0.25;

        let exact = build_exact_b_measurement(&operator, &reference, &indices, variance, None)
            .expect("exact measurement should build");
        assert_eq!(exact.observations, vec![1.0, 3.0, 4.0]);

        let noisy_a =
            build_exact_b_measurement(&operator, &reference, &indices, variance, Some(17))
                .expect("noisy measurement should build");
        let noisy_b =
            build_exact_b_measurement(&operator, &reference, &indices, variance, Some(17))
                .expect("same-seed noisy measurement should build");
        let noisy_c =
            build_exact_b_measurement(&operator, &reference, &indices, variance, Some(18))
                .expect("different-seed noisy measurement should build");

        assert_eq!(noisy_a.observations, noisy_b.observations);
        assert_ne!(noisy_a.observations, exact.observations);
        assert_ne!(noisy_a.observations, noisy_c.observations);

        let row_two_a = exact_b_training_observation_value(reference[2], 0.5, Some(17), 2)
            .expect("row-specific noise should build");
        let row_two_b = exact_b_training_observation_value(reference[2], 0.5, Some(17), 2)
            .expect("row-specific noise should be reproducible");
        assert_eq!(row_two_a, row_two_b);
        assert_eq!(noisy_a.observations[1], row_two_a);
    }

    #[test]
    fn toroidal_exact_b_heldout_metrics_split_latent_and_noisy_predictions() {
        let rows = vec![ToroidalExactBHeldoutPredictionRow {
            name: "flux_test".to_string(),
            truth: 1.0,
            noisy_observation: 1.5,
            prediction: 1.2,
            residual: 0.2,
            noisy_residual: -0.3,
            posterior_sd: 0.1,
            predictive_sd: 0.5,
            standardized_residual: 2.0,
            noisy_standardized_residual: -0.6,
            lower95: 1.004,
            upper95: 1.396,
            covered95: false,
            noisy_lower95: 0.22,
            noisy_upper95: 2.18,
            noisy_covered95: true,
        }];

        let metrics = exact_b_prediction_summary_metrics(&rows);
        assert!((prediction_rmse_exact_b(&rows) - 0.2).abs() < 1.0e-12);
        assert!((noisy_prediction_rmse_exact_b(&rows) - 0.3).abs() < 1.0e-12);
        assert!((metrics.mean_posterior_sd - 0.1).abs() < 1.0e-12);
        assert!((metrics.mean_predictive_sd - 0.5).abs() < 1.0e-12);
        assert_eq!(metrics.max_abs_z, 2.0);
        assert_eq!(metrics.noisy_max_abs_z, 0.6);
    }

    #[test]
    fn toroidal_exact_b_observation_index_override_validates_rows_and_roles() {
        let valid = ToroidalExactBObservationIndexOverride {
            training_indices: vec![3, 1],
            heldout_indices: vec![4, 0],
        };
        validate_exact_b_observation_index_override(&valid, 5)
            .expect("valid explicit observation indices should pass");
        assert_eq!(
            exact_b_probe_roles(5, &valid.training_indices, &valid.heldout_indices),
            vec!["heldout", "train", "unused", "train", "heldout"]
        );

        let empty_training = ToroidalExactBObservationIndexOverride {
            training_indices: Vec::new(),
            heldout_indices: vec![0],
        };
        assert!(
            validate_exact_b_observation_index_override(&empty_training, 5)
                .expect_err("empty training set should fail")
                .contains("training")
        );

        let out_of_bounds = ToroidalExactBObservationIndexOverride {
            training_indices: vec![5],
            heldout_indices: vec![0],
        };
        assert!(
            validate_exact_b_observation_index_override(&out_of_bounds, 5)
                .expect_err("out-of-bounds row should fail")
                .contains("out of bounds")
        );

        let duplicate = ToroidalExactBObservationIndexOverride {
            training_indices: vec![1, 1],
            heldout_indices: vec![0],
        };
        assert!(validate_exact_b_observation_index_override(&duplicate, 5)
            .expect_err("duplicate row should fail")
            .contains("duplicated"));

        let overlap = ToroidalExactBObservationIndexOverride {
            training_indices: vec![1, 2],
            heldout_indices: vec![2, 3],
        };
        assert!(validate_exact_b_observation_index_override(&overlap, 5)
            .expect_err("overlapping row should fail")
            .contains("both training and heldout"));
    }

    #[test]
    fn toroidal_exact_b_submitted_observation_split_is_complete() {
        let submitted = toroidal_exact_b_thesis_submitted_observation_index_override();
        validate_exact_b_observation_index_override(&submitted, 36)
            .expect("submitted observation split should validate");
        assert_eq!(submitted.training_indices.len(), 12);
        assert_eq!(submitted.heldout_indices.len(), 24);
        let mut all = submitted.training_indices.clone();
        all.extend(submitted.heldout_indices);
        all.sort_unstable();
        assert_eq!(all, (0..36).collect::<Vec<_>>());
    }

    #[test]
    fn toroidal_exact_b_field_recovery_design_is_fixed_nested_and_disjoint() {
        let design6 = toroidal_exact_b_field_recovery_observation_indices(16, 6)
            .expect("6-row field design should build");
        let design12 = toroidal_exact_b_field_recovery_observation_indices(16, 12)
            .expect("12-row field design should build");
        let design24 = toroidal_exact_b_field_recovery_observation_indices(16, 24)
            .expect("24-row field design should build");
        let design36 = toroidal_exact_b_field_recovery_observation_indices(16, 36)
            .expect("36-row field design should build");

        assert_eq!(design36.heldout_indices.len(), 12);
        assert_eq!(design36.training_indices.len(), 36);
        assert_eq!(
            design36
                .heldout_indices
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            12
        );
        assert_eq!(
            design36
                .training_indices
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            36
        );
        assert!(design36
            .training_indices
            .iter()
            .chain(&design36.heldout_indices)
            .all(|row| *row < 48));
        assert!(design36
            .training_indices
            .iter()
            .all(|row| !design36.heldout_indices.contains(row)));
        assert!(design12
            .training_indices
            .starts_with(&design6.training_indices));
        assert!(design24
            .training_indices
            .starts_with(&design12.training_indices));
        assert!(design36
            .training_indices
            .starts_with(&design24.training_indices));
        assert_eq!(
            design36.heldout_indices,
            vec![3, 4, 5, 15, 16, 17, 27, 28, 29, 39, 40, 41]
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_surface_flux_row_norm_audit_reports_field_rows() {
        let mut config = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        config.surface_flux_azimuth_count = 16;
        config.observation_noise_std = 1.0e-9;
        let rows =
            toroidal_exact_b_surface_flux_row_norms(&config).expect("row norm audit should build");
        assert_eq!(rows.len(), 48);
        assert_eq!(rows[0].name, "flux_inner_00");
        assert_eq!(rows[47].name, "flux_outer_15");
        assert!(rows.iter().all(|row| {
            row.nnz > 0
                && row.l2_norm.is_finite()
                && row.max_abs_entry.is_finite()
                && row.max_diagonal_contribution.is_finite()
        }));
        assert!(
            rows.iter()
                .map(|row| row.max_diagonal_contribution)
                .fold(0.0, f64::max)
                >= 1.0e18
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_physical_b_prior_calibration_matches_target() {
        let mut config = toroidal_exact_b_canonical_source_designed_flux_config();
        config.prior_tau = 1.0;
        config.output_dir = None;
        config.write_outputs = false;
        let report = toroidal_exact_b_physical_b_prior_calibration(
            &config,
            TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_RMS_MULTIPLIER,
        )
        .expect("physical B prior calibration should run");

        assert_eq!(
            report.reference_solve_mode,
            ToroidalExactBReferenceSolveMode::DeterministicPde
        );
        assert_eq!(
            report.reference_solver_diagonal_shift,
            TOROIDAL_EXACT_B_DETERMINISTIC_REFERENCE_DIAGONAL_SHIFT
        );
        assert!(report.effective_prior_tau.is_finite() && report.effective_prior_tau > 0.0);
        assert!(report.nominal_b_rms.is_finite() && report.nominal_b_rms > 0.0);
        assert_eq!(
            report.b_rows,
            3 * report.cells,
            "physical B rows should be cell-major vector components"
        );
        assert!(
            report.normalization_relative_error < 1.0e-10,
            "calibrated mean B2 should match target, relative error {:.3e}",
            report.normalization_relative_error
        );
        assert!(
            relative_scalar_error(
                report.effective_prior_tau,
                TOROIDAL_EXACT_B_CALIBRATED_PRIOR_TAU
            ) < 1.0e-12
        );
        assert!(
            relative_scalar_error(
                report.nominal_b_rms,
                TOROIDAL_EXACT_B_CALIBRATION_NOMINAL_B_RMS
            ) < 1.0e-12
        );
    }

    #[test]
    fn toroidal_exact_b_canonical_config_uses_hardcoded_calibrated_tau() {
        let config = toroidal_exact_b_canonical_source_designed_flux_config();
        assert_eq!(config.prior_tau, TOROIDAL_EXACT_B_CANONICAL_PRIOR_TAU);
        assert!(
            relative_scalar_error(
                config.prior_tau,
                toroidal_exact_b_prior_tau_for_physical_b_multiplier(
                    TOROIDAL_EXACT_B_CANONICAL_PRIOR_RMS_MULTIPLIER
                )
                .expect("canonical prior multiplier should be valid")
            ) < 1.0e-12
        );
        assert_eq!(
            config.observation_mode,
            ToroidalExactBObservationMode::SourceDesignedFluxes
        );
        assert_eq!(
            config.reference_solve_mode,
            ToroidalExactBReferenceSolveMode::DeterministicPde
        );
        assert_eq!(
            config.reference_solver_diagonal_shift,
            TOROIDAL_EXACT_B_DETERMINISTIC_REFERENCE_DIAGONAL_SHIFT
        );
        assert_eq!(
            TOROIDAL_EXACT_B_PHYSICAL_B_PRIOR_CALIBRATION_LABEL,
            "physical_b_nominal_rms_x10"
        );
        assert_eq!(
            exact_b_prior_calibration_metadata(config.prior_tau).0,
            "physical_b_nominal_rms_x100"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_precision_scale_audit_separates_field_terms() {
        let mut config = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        config.observation_mode = ToroidalExactBObservationMode::SurfaceFluxes;
        config.surface_flux_azimuth_count = 16;
        config.observation_noise_std = 1.0e-9;
        config.observation_index_override =
            Some(toroidal_exact_b_field_recovery_observation_indices(16, 6).unwrap());

        let audit = toroidal_exact_b_precision_scale_audit(&config)
            .expect("precision scale audit should build");
        assert_eq!(audit.source_dimension, SOURCE_MODE_COUNT);
        assert_eq!(audit.training_rows, 6);
        assert_eq!(audit.heldout_rows, 12);
        assert_eq!(
            audit.nondimensionalization,
            ToroidalExactBNondimensionalizationMode::Off
        );
        assert_eq!(audit.state_scale_min, 1.0);
        assert_eq!(audit.state_scale_max, 1.0);

        let terms = audit
            .terms
            .iter()
            .map(|term| term.term.as_str())
            .collect::<HashSet<_>>();
        for expected in [
            "state_prior",
            "source_prior",
            "pde_state",
            "pde_source",
            "flux_observations",
            "posterior_total",
        ] {
            assert!(terms.contains(expected), "missing term {expected}");
        }

        let term = |name: &str| {
            audit
                .terms
                .iter()
                .find(|term| term.term == name)
                .expect("term should exist")
        };
        assert!(term("state_prior").max_abs_diagonal.is_finite());
        assert_eq!(term("source_prior").max_block, "source");
        assert!(
            term("flux_observations").max_abs_diagonal >= 1.0e18,
            "high-confidence flux rows should make a visible diagonal contribution"
        );
        assert!(term("posterior_total").max_abs_diagonal >= term("pde_source").max_abs_diagonal);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_pde_column_norm_nondimensionalization_reduces_audit_ratio() {
        let mut raw_config = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        raw_config.observation_mode = ToroidalExactBObservationMode::SurfaceFluxes;
        raw_config.surface_flux_azimuth_count = 16;
        raw_config.observation_noise_std = 1.0e-9;
        raw_config.base.residual_selection = ToroidalResidualSelection::Full;
        raw_config.base.pde_variance = 3.0e-8;
        raw_config.observation_index_override =
            Some(toroidal_exact_b_field_recovery_observation_indices(16, 6).unwrap());
        let raw_audit = toroidal_exact_b_precision_scale_audit(&raw_config)
            .expect("raw precision scale audit should build");
        let raw_ratio = raw_audit.scaled_posterior_diagonal_ratio;

        let mut scaled_config = raw_config.clone();
        scaled_config.nondimensionalization =
            ToroidalExactBNondimensionalizationMode::PdeColumnNorm;
        let scaled_audit = toroidal_exact_b_precision_scale_audit(&scaled_config)
            .expect("scaled precision scale audit should build");

        assert_eq!(
            scaled_audit.nondimensionalization,
            ToroidalExactBNondimensionalizationMode::PdeColumnNorm
        );
        assert!(scaled_audit.state_scale_min.is_finite() && scaled_audit.state_scale_min > 0.0);
        assert!(scaled_audit.state_scale_max.is_finite() && scaled_audit.state_scale_max > 0.0);
        assert!(scaled_audit.source_scale_min.is_finite() && scaled_audit.source_scale_min > 0.0);
        assert!(scaled_audit.source_scale_max.is_finite() && scaled_audit.source_scale_max > 0.0);
        assert!(
            scaled_audit.scaled_posterior_diagonal_ratio < raw_ratio * 1.0e-6,
            "scaled ratio {:.3e} should be many orders below raw ratio {:.3e}",
            scaled_audit.scaled_posterior_diagonal_ratio,
            raw_ratio
        );
    }

    #[test]
    fn toroidal_exact_b_rejects_selected_inverse_variance() {
        let mut config = exact_b_test_config(ToroidalExactBReferenceMode::NominalDebug);
        config.base.variance.mode = LinearPdeVarianceMode::SelectedInverse;
        let err =
            validate_exact_b_config(&config).expect_err("selected inverse should be rejected");
        assert!(err.contains("selected inverse"));
    }

    #[test]
    fn toroidal_exact_b_diagnostic_cases_use_one_axis_sweeps() {
        let base = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        let config = ToroidalExactBDiagnosticsConfig {
            observation_modes: vec![ToroidalExactBDiagnosticObservationMode::PdeOnly],
            pde_variances: vec![base.base.pde_variance, 1e-6],
            prior_taus: vec![base.prior_tau],
            source_prior_stds: vec![base.source_prior_std, 2.5],
            observation_noise_stds: vec![base.observation_noise_std],
            include_source_response: false,
            output_dir: None,
            write_outputs: false,
            base,
        };
        let cases = exact_b_diagnostic_cases(&config);
        let labels = cases
            .iter()
            .map(|(label, mode, _)| (label.as_str(), *mode))
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                ("baseline", ToroidalExactBDiagnosticObservationMode::PdeOnly),
                (
                    "pde_variance=1e-6",
                    ToroidalExactBDiagnosticObservationMode::PdeOnly
                ),
                (
                    "source_prior_std=2.500e0",
                    ToroidalExactBDiagnosticObservationMode::PdeOnly
                ),
            ]
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_source_designed_mode_selects_four_training_fluxes() {
        let mut config = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        config.observation_mode = ToroidalExactBObservationMode::SourceDesignedFluxes;
        config.surface_flux_azimuth_count = 4;
        config.heldout_count = 128;
        let (topology, coords) = load_mesh(&config.base.mesh_path).expect("mesh should load");
        let boundary = outer_boundary(&topology, &coords, config.base.geometry);
        let linear_system =
            build_toroidal_exact_b_linear_system(&topology, &coords, boundary, &config)
                .expect("linear exact-B system should build");
        let source_modes = assemble_toroidal_source_modes(
            &topology,
            &coords,
            &linear_system.layout,
            config.base.geometry,
            config.base.nu_air,
        )
        .expect("source modes should assemble");
        let observation_system = build_exact_b_observation_system(
            &topology,
            &coords,
            config.observation_mode,
            config.base.geometry,
            config.surface_flux_azimuth_count,
        )
        .expect("source-designed flux observation system should build");

        let (training, heldout) = split_source_designed_flux_observation_indices(
            &linear_system,
            &source_modes,
            &observation_system.operator,
            config.base.prior_precision,
            config.base.pde_variance,
            config.observation_train_fraction,
            config.heldout_count,
        )
        .expect("source-designed split should build");

        assert_eq!(training.len(), SOURCE_MODE_COUNT);
        assert_eq!(
            heldout.len(),
            observation_system.operator.nrows() - SOURCE_MODE_COUNT
        );
        assert_eq!(heldout.len(), 8);
        assert_eq!(
            training.iter().copied().collect::<HashSet<_>>().len(),
            SOURCE_MODE_COUNT
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_source_designed_mode_supports_dense_flux_training_splits() {
        let mut config = exact_b_test_config(ToroidalExactBReferenceMode::PerturbedSource);
        config.observation_mode = ToroidalExactBObservationMode::SourceDesignedFluxes;
        config.surface_flux_azimuth_count = 12;
        config.observation_train_fraction = 0.33;
        config.heldout_count = 128;
        let (topology, coords) = load_mesh(&config.base.mesh_path).expect("mesh should load");
        let boundary = outer_boundary(&topology, &coords, config.base.geometry);
        let linear_system =
            build_toroidal_exact_b_linear_system(&topology, &coords, boundary, &config)
                .expect("linear exact-B system should build");
        let source_modes = assemble_toroidal_source_modes(
            &topology,
            &coords,
            &linear_system.layout,
            config.base.geometry,
            config.base.nu_air,
        )
        .expect("source modes should assemble");
        let observation_system = build_exact_b_observation_system(
            &topology,
            &coords,
            config.observation_mode,
            config.base.geometry,
            config.surface_flux_azimuth_count,
        )
        .expect("dense source-designed flux observation system should build");
        let (training, heldout) = split_source_designed_flux_observation_indices(
            &linear_system,
            &source_modes,
            &observation_system.operator,
            config.base.prior_precision,
            config.base.pde_variance,
            config.observation_train_fraction,
            config.heldout_count,
        )
        .expect("dense source-designed split should build");

        assert_eq!(observation_system.operator.nrows(), 36);
        assert_eq!(training.len(), 12);
        assert_eq!(heldout.len(), 24);
        assert_eq!(training.iter().copied().collect::<HashSet<_>>().len(), 12);
        assert_eq!(heldout.iter().copied().collect::<HashSet<_>>().len(), 24);
        assert!(training.iter().all(|row| !heldout.contains(row)));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn toroidal_exact_b_source_designed_equilibrated_recovery_is_noncollapsed() {
        let mut config = ToroidalExactBRecoveryConfig {
            base: ToroidalExactBBaseConfig {
                pde_variance: 3e-8,
                precision_policy: LinearPdePrecisionPolicy::DiagonalEquilibrated {
                    max_relative_asymmetry: 1.0e-10,
                },
                ..ToroidalExactBBaseConfig::default()
            },
            reference_mode: ToroidalExactBReferenceMode::PerturbedSource,
            observation_mode: ToroidalExactBObservationMode::SourceDesignedFluxes,
            source_deltas: [0.0, 0.15, -0.10, 0.05],
            source_prior_std: 0.25,
            reference_pde_variance: Some(1e-8),
            prior_kappa: 1.0,
            prior_tau: 1e-6,
            observation_train_fraction: 0.10,
            observation_noise_std: 1e-10,
            heldout_count: 128,
            surface_flux_azimuth_count: 4,
            nondimensionalization: ToroidalExactBNondimensionalizationMode::PdeColumnNorm,
            write_outputs: false,
            output_dir: None,
            ..ToroidalExactBRecoveryConfig::default()
        };
        config.base.variance.mode = LinearPdeVarianceMode::Exact;

        let report = run_toroidal_exact_b_recovery_experiment(&config)
            .expect("source-designed equilibrated recovery should solve");
        let eta_l2 = report
            .source_posterior
            .iter()
            .map(|row| row.error * row.error)
            .sum::<f64>()
            .sqrt();

        assert_eq!(report.source_posterior.len(), SOURCE_MODE_COUNT);
        assert_eq!(report.summary.training_rows, SOURCE_MODE_COUNT);
        assert_eq!(report.summary.heldout_rows, 8);
        assert!(report.heldout_predictions.iter().all(|row| {
            row.posterior_sd.is_finite()
                && row.lower95.is_finite()
                && row.upper95.is_finite()
                && row.standardized_residual.is_finite()
                && row.predictive_sd.is_finite()
                && row.noisy_lower95.is_finite()
                && row.noisy_upper95.is_finite()
                && row.noisy_standardized_residual.is_finite()
        }));
        assert!(
            report
                .source_posterior
                .iter()
                .map(|row| row.posterior_mean.abs())
                .fold(0.0, f64::max)
                > 1.0e-2,
            "source posterior should not collapse to zero"
        );
        assert!(eta_l2 < 3.0e-2, "source eta L2 error was {eta_l2:.3e}");
        assert_eq!(
            report
                .result
                .debug
                .posterior_factorization
                .precision_policy
                .label(),
            "diagonal_equilibrated"
        );
    }

    #[test]
    fn toroidal_exact_b_source_designed_flux_rows_span_source_modes() {
        let source_responses = vec![
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        ];
        let rows = select_source_designed_flux_rows(&source_responses, SOURCE_MODE_COUNT)
            .expect("source-designed rows should select");
        assert_eq!(rows.len(), SOURCE_MODE_COUNT);
        assert_eq!(
            rows.iter().copied().collect::<HashSet<_>>().len(),
            SOURCE_MODE_COUNT
        );

        let mut gram = [[0.0; SOURCE_MODE_COUNT]; SOURCE_MODE_COUNT];
        for row in rows {
            gram = source_design_gram_with_row(gram, &source_responses, row);
        }
        let singular_values = source_response_singular_values(gram);
        assert!(
            singular_values[SOURCE_MODE_COUNT - 1] > 1e-8,
            "selected rows should observe all source modes"
        );
    }

    #[test]
    fn toroidal_exact_b_source_design_near_ties_prefer_stable_row_order() {
        assert!(source_design_score_is_better(10.0, 2, 10.0, 5));
        assert!(!source_design_score_is_better(10.0, 5, 10.0, 2));
        assert!(source_design_score_is_better(10.0 + 1.0e-8, 5, 10.0, 2));
        assert!(source_design_score_is_better(11.0, 5, 10.0, 2));
        assert!(!source_design_score_is_better(9.0, 1, 10.0, 2));
    }

    #[test]
    fn toroidal_exact_b_probe_reference_csv_contract_roundtrips() {
        let probes = vec![
            ToroidalExactBProbeRow {
                name: "cell_000000_bx".to_string(),
                cell_index: 0,
                component: 0,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                role: "train".to_string(),
            },
            ToroidalExactBProbeRow {
                name: "cell_000001_by".to_string(),
                cell_index: 1,
                component: 1,
                x: 4.0,
                y: 5.0,
                z: 6.0,
                role: "heldout".to_string(),
            },
        ];
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("exact_b_reference_{stamp}.csv"));
        fs::write(
            &path,
            "name,value,noise_std,role\ncell_000000_bx,1.25,0.01,train\ncell_000001_by,-2.5,0.01,heldout\n",
        )
        .expect("reference CSV should write");

        let values = load_exact_b_reference_observations(&path, &probes, &[0], &[1])
            .expect("reference CSV should load");
        assert_eq!(values, vec![1.25, -2.5]);
        let probe_csv = exact_b_probe_csv(&probes);
        assert!(probe_csv.contains("name,cell_index,component,x,y,z,role"));
        assert!(probe_csv.contains("cell_000001_by"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn nonlinear_toroidal_weiland_summary_aggregates_successes_and_failures() {
        let success = ToroidalWeilandComparisonRow {
            reference_kappa: 1.0,
            prior_kappa: 1.0,
            kappa_times_diameter: 12.0,
            prior_mode: ToroidalPriorMode::LinearProxyMaternAlpha2,
            nonlinear_residual_likelihood: true,
            selection_label: "shuffled_count_4_seed_1".to_string(),
            seed: 1,
            residual_rows_requested: 4,
            residual_rows_used: 4,
            residual_rows_total: 16,
            residual_fraction: 0.25,
            success: true,
            failure_reason: None,
            iterations: 2,
            damping_steps: 1,
            final_residual_norm: 2.0,
            linear_solve_iteration_sum: 0,
            linear_solve_residual_max: 0.0,
            posterior_factor_nnz: 12,
            final_factorization_seconds: 0.1,
            map_relative_error_to_reference: 0.3,
            cell_b_relative_error_to_reference: 0.4,
            flux_sensor_rmse_to_reference: 0.5,
            sensor_variance_min: 0.01,
            sensor_variance_max: 0.02,
            sensor_mean_abs_z: 0.6,
            sensor_coverage_2sigma: 1.0,
            posterior_factorizes: true,
        };
        let mut failure = success.clone();
        failure.seed = 2;
        failure.success = false;
        failure.failure_reason = Some("factorization failed".to_string());
        failure.final_residual_norm = f64::NAN;
        let summaries = summarize_weiland_comparison_rows(&[success, failure]);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].success_count, 1);
        assert_eq!(summaries[0].failure_count, 1);
        assert_eq!(summaries[0].final_residual_mean, 2.0);
        assert_eq!(summaries[0].cell_b_relative_error_mean, 0.4);
        assert_eq!(summaries[0].sensor_coverage_2sigma_mean, 1.0);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_strong_probe_stratified_selector_is_exact_and_deterministic() {
        let (topology, coords) = load_mesh(Path::new(DEFAULT_MESH_PATH)).unwrap();
        let geometry = ToroidalInductorGeometry::default();
        let first = select_stratified_toroidal_cells(&topology, &coords, geometry, 3, 99).unwrap();
        let second = select_stratified_toroidal_cells(&topology, &coords, geometry, 3, 99).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.cells.len(), 3);
        assert_eq!(first.weights.len(), 3);
        assert!(first.cells.windows(2).all(|window| window[0] < window[1]));
        assert!(first
            .weights
            .iter()
            .all(|weight| weight.is_finite() && *weight > 0.0));

        let selected_strata = first
            .cells
            .iter()
            .map(|cell_index| {
                let cell = SimplexIdx::new(3, *cell_index).handle(&topology);
                let point = coord_vec_to_point3(cell.coord_simplex(&coords).barycenter());
                let radius = toroidal_radius_from_point(point, geometry);
                if radius <= geometry.core_minor_radius {
                    0
                } else if radius <= geometry.coil_minor_radius {
                    1
                } else {
                    2
                }
            })
            .collect::<HashSet<_>>();
        assert!(
            selected_strata.contains(&0),
            "selector should include core cells"
        );
        assert!(
            selected_strata.contains(&1),
            "selector should include source-shell cells"
        );
    }

    #[test]
    fn nonlinear_toroidal_strong_probe_precision_scaling_uses_cell_weights() {
        let noise = build_local_strong_probe_noise(&[2.0, 4.0], 0.5).unwrap();
        let GaussianNoiseModel::Precision(precision) = noise.noise else {
            panic!("non-unit local probe weights should produce a diagonal precision");
        };
        assert_eq!(precision.nrows(), 6);
        assert_eq!(precision.ncols(), 6);
        assert_eq!(precision.nnz(), 6);
        assert_eq!(matrix_entry_case(&precision, 0, 0), 4.0);
        assert_eq!(matrix_entry_case(&precision, 3, 3), 8.0);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_strong_probe_builds_finite_local_residual() {
        let config = NonlinearToroidalConfig {
            pde_observation_mode: ToroidalPdeObservationMode::LocalMagneticStrongCells,
            residual_selection: ToroidalResidualSelection::ShuffledCount { count: 9, seed: 7 },
            write_outputs: false,
            include_cell_b_variance: false,
            compute_harmonic_diagnostics: false,
            ..NonlinearToroidalConfig::default()
        };
        let (topology, coords, model) = build_toroidal_model_for_diagnostics(&config).unwrap();
        let observation_total =
            toroidal_observation_residual_dimension(&model, config.pde_observation_mode);
        let observation = build_toroidal_observation_residual(
            &config,
            &topology,
            &coords,
            &model,
            observation_total,
        )
        .unwrap();
        assert_eq!(observation.rows_total, 3 * topology.nsimplices(3));
        assert_eq!(observation.rows_used, 9);
        let residual = observation
            .model
            .residual_and_jacobian(&vec![0.0; model.reduced_dimension()])
            .unwrap();
        assert_eq!(residual.residual.len(), 9);
        assert_eq!(residual.jacobian.nrows(), 9);
        assert_eq!(residual.jacobian.ncols(), model.reduced_dimension());
        assert!(residual.residual.iter().all(|value| value.is_finite()));
        assert!(residual
            .jacobian
            .triplet_iter()
            .all(|(_, _, value)| value.is_finite()));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_beta_zero_solves_and_reports_finite_diagnostics() {
        let config = NonlinearToroidalConfig {
            residual_weighting: ToroidalResidualWeighting::Euclidean,
            write_outputs: false,
            include_cell_b_variance: false,
            max_iterations: 12,
            ..NonlinearToroidalConfig::default()
        };
        let report = run_nonlinear_toroidal_inductor(&config)
            .expect("beta=0 toroidal nonlinear path should solve");
        assert!(report.converged);
        assert!(report.final_residual_norm <= 1e-3 * report.initial_residual_norm);
        assert!(report.posterior_factorizes);
        assert_eq!(
            report.linear_solve_mode,
            GaussNewtonLinearSolveMode::DirectCholesky
        );
        assert!(report.posterior_factor_nnz > 0);
        assert!(report.final_factorization_seconds.is_finite());
        assert_eq!(report.gauge_edge_dofs, 0);
        assert!(report.direct_linear_relative_error.unwrap() <= 1e-6);
        assert!(report.mixed_reference_b_relative_error.unwrap().is_finite());
        assert!(report.harmonic_coefficient_norm.is_finite());
        assert!(report.harmonic_coefficient_norm > 1e-16);
        assert_eq!(report.sensor_reports.len(), 3);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_beta_positive_reduces_residual() {
        let config = NonlinearToroidalConfig {
            beta_core: 1e16,
            write_outputs: false,
            include_cell_b_variance: false,
            max_iterations: 12,
            ..NonlinearToroidalConfig::default()
        };
        let report =
            run_nonlinear_toroidal_inductor(&config).expect("nonlinear toroidal path should solve");
        assert_eq!(
            report.residual_weighting,
            ToroidalResidualWeighting::Euclidean
        );
        assert!(report.residual_precision_normalization.is_none());
        assert!(report.converged);
        assert!(report.final_residual_norm <= 1e-3 * report.initial_residual_norm);
        assert!(report.posterior_factorizes);
        assert_eq!(
            report.linear_solve_mode,
            GaussNewtonLinearSolveMode::DirectCholesky
        );
        assert_eq!(report.linear_solve_iteration_sum, 0);
        assert!(report.linear_solve_residual_max.is_finite());
        assert!(report.posterior_factor_nnz > 0);
        assert!(report.harmonic_coefficient_norm.is_finite());
        assert!(report
            .result
            .posterior_variance
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_linear_proxy_prior_is_corrected_by_residual_likelihood() {
        let base = NonlinearToroidalConfig {
            beta_core: 1e16,
            prior_mode: ToroidalPriorMode::LinearProxyMaternAlpha2,
            write_outputs: false,
            include_cell_b_variance: false,
            max_iterations: 12,
            ..NonlinearToroidalConfig::default()
        };
        let prior_only = run_nonlinear_toroidal_inductor(&NonlinearToroidalConfig {
            include_nonlinear_residual: false,
            ..base.clone()
        })
        .expect("linear-proxy prior-only toroidal path should solve");
        assert_eq!(
            prior_only.residual_weighting,
            ToroidalResidualWeighting::Euclidean
        );
        assert!(prior_only.residual_precision_normalization.is_none());
        assert!(prior_only.converged);
        assert!(prior_only.posterior_factorizes);
        assert_eq!(
            prior_only.linear_solve_mode,
            GaussNewtonLinearSolveMode::DirectCholesky
        );
        assert_eq!(
            prior_only.prior_mode,
            ToroidalPriorMode::LinearProxyMaternAlpha2
        );
        assert!(prior_only.prior_kappa.is_finite() && prior_only.prior_kappa > 0.0);
        assert!(!prior_only.prior_kappa_fallback_used);
        assert!(prior_only.prior_tau.is_finite() && prior_only.prior_tau > 0.0);
        assert!(prior_only
            .result
            .posterior_precision
            .triplet_iter()
            .all(|(_, _, value)| value.is_finite()));
        assert!(prior_only.map_relative_distance_from_linear_mean <= 1e-12);

        let corrected = run_nonlinear_toroidal_inductor(&NonlinearToroidalConfig {
            include_nonlinear_residual: true,
            ..base
        })
        .expect("linear-proxy plus nonlinear residual toroidal path should solve");
        assert_eq!(
            corrected.residual_weighting,
            ToroidalResidualWeighting::Euclidean
        );
        assert!(corrected.residual_precision_normalization.is_none());
        assert!(corrected.converged);
        assert!(corrected.posterior_factorizes);
        assert_eq!(
            corrected.linear_solve_mode,
            GaussNewtonLinearSolveMode::DirectCholesky
        );
        assert_eq!(corrected.linear_solve_iteration_sum, 0);
        assert!(corrected.harmonic_coefficient_norm.is_finite());
        assert!(
            corrected.final_residual_norm <= 1e-2 * prior_only.final_residual_norm,
            "residual likelihood should improve true nonlinear residual: prior-only {}, corrected {}",
            prior_only.final_residual_norm,
            corrected.final_residual_norm
        );
        assert!(corrected
            .sensor_reports
            .iter()
            .all(|sensor| sensor.nonlinear_value.is_finite()));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_linear_proxy_uses_explicit_kappa_without_fallback() {
        let report = run_nonlinear_toroidal_inductor(&NonlinearToroidalConfig {
            beta_core: 1e16,
            prior_mode: ToroidalPriorMode::LinearProxyMaternAlpha2,
            linear_proxy_kappa: Some(1.0),
            linear_proxy_allow_kappa_fallback: false,
            include_nonlinear_residual: false,
            write_outputs: false,
            include_cell_b_variance: false,
            compute_harmonic_diagnostics: false,
            max_iterations: 4,
            ..NonlinearToroidalConfig::default()
        })
        .expect("explicit-kappa linear proxy prior should solve");
        assert_eq!(
            report.prior_mode,
            ToroidalPriorMode::LinearProxyMaternAlpha2
        );
        assert!((report.prior_kappa - 1.0).abs() <= 1e-14);
        assert!(!report.prior_kappa_fallback_used);
        assert!(report.posterior_factorizes);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_residual_budget_reports_reference_errors() {
        let config = ToroidalResidualBudgetConfig {
            base: NonlinearToroidalConfig {
                beta_core: 1e16,
                write_outputs: false,
                include_cell_b_variance: false,
                max_iterations: 6,
                ..NonlinearToroidalConfig::default()
            },
            prior_modes: vec![ToroidalPriorMode::LinearProxyMaternAlpha2],
            residual_strides: vec![64, 1],
            shuffled: true,
            seed: 7,
        };
        let report = run_toroidal_residual_budget_experiment(&config)
            .expect("residual-budget toroidal experiment should run");
        assert_eq!(report.rows.len(), 2);
        assert!(report.reference.posterior_factorizes);
        assert!(report.rows.iter().all(|row| {
            row.posterior_factorizes
                && row.final_residual_norm.is_finite()
                && row.linear_solve_residual_max.is_finite()
                && row.posterior_factor_nnz > 0
        }));
        let coarse = &report.rows[0];
        let full = &report.rows[1];
        assert!(full.residual_rows_used >= coarse.residual_rows_used);
        assert!(
            full.cell_b_relative_error_to_reference
                <= coarse.cell_b_relative_error_to_reference + 1e-10
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_toroidal_sensor_regression_reports_finite_holdout_uq() {
        let config = ToroidalSensorRegressionConfig {
            base: NonlinearToroidalConfig {
                beta_core: 1e16,
                write_outputs: false,
                include_cell_b_variance: false,
                max_iterations: 4,
                ..NonlinearToroidalConfig::default()
            },
            prior_modes: vec![ToroidalPriorMode::LinearProxyMaternAlpha2],
            variants: vec![
                ToroidalSensorRegressionVariant::PriorOnly,
                ToroidalSensorRegressionVariant::SensorsPlusResidualBudget,
            ],
            azimuth_count: 2,
            train_counts: vec![3],
            residual_stride: 64,
            sensor_variance: 1e-20,
            synthetic_noise_std: 0.0,
            seed: 11,
        };
        let report = run_toroidal_sensor_regression_experiment(&config)
            .expect("sensor-regression toroidal experiment should run");
        assert_eq!(report.sensors.len(), 6);
        assert_eq!(report.rows.len(), 2);
        assert!(report.rows.iter().all(|row| {
            row.posterior_factorizes
                && row.holdout_rmse.is_finite()
                && row.linear_solve_residual_max.is_finite()
                && row.posterior_factor_nnz > 0
        }));
        let prior_only = report
            .rows
            .iter()
            .find(|row| row.variant == ToroidalSensorRegressionVariant::PriorOnly)
            .unwrap();
        let sensors_plus_residual = report
            .rows
            .iter()
            .find(|row| row.variant == ToroidalSensorRegressionVariant::SensorsPlusResidualBudget)
            .unwrap();
        assert!(sensors_plus_residual.holdout_rmse <= prior_only.holdout_rmse + 1e-7);
    }

    #[test]
    #[ignore = "runs several nonlinear toroidal solves; use for local integration smoke"]
    fn nonlinear_toroidal_weiland_comparison_records_budget_curve() {
        let config = ToroidalWeilandComparisonConfig {
            base: NonlinearToroidalConfig {
                beta_core: 1e16,
                write_outputs: false,
                include_cell_b_variance: false,
                compute_harmonic_diagnostics: false,
                max_iterations: 2,
                ..NonlinearToroidalConfig::default()
            },
            prior_modes: vec![ToroidalPriorMode::LinearProxyMaternAlpha2],
            residual_fractions: vec![0.0, 1.0 / 64.0, 1.0],
            explicit_kappas: Vec::new(),
            kappa_diameter_scales: vec![20.0],
            include_default_kappa: true,
            row_selection_repetitions: 1,
            seed: 31,
            sensor_azimuth_count: 1,
        };
        let report = run_toroidal_weiland_comparison_experiment(&config)
            .expect("Weiland-style toroidal comparison should run");
        assert_eq!(report.references.len(), 2);
        assert!(report.references.iter().all(|reference| reference.success));
        assert_eq!(
            report.residual_rows_total,
            report.rows[0].residual_rows_total
        );
        assert!(report.rows.iter().all(|row| {
            row.success
                && row.final_residual_norm.is_finite()
                && row.cell_b_relative_error_to_reference.is_finite()
                && row.sensor_variance_min.is_finite()
                && row.sensor_variance_min >= 0.0
                && row.sensor_variance_max.is_finite()
                && row.sensor_variance_max >= row.sensor_variance_min
        }));
        for reference in &report.references {
            let coarse = report
                .rows
                .iter()
                .find(|row| {
                    row.reference_kappa == reference.requested_kappa
                        && row.residual_rows_requested > 0
                        && row.residual_rows_requested < row.residual_rows_total
                })
                .expect("coarse residual-budget row should be present");
            let full = report
                .rows
                .iter()
                .find(|row| {
                    row.reference_kappa == reference.requested_kappa
                        && row.residual_rows_requested == row.residual_rows_total
                })
                .expect("full residual-budget row should be present");
            assert!(
                full.cell_b_relative_error_to_reference
                    <= coarse.cell_b_relative_error_to_reference + 1e-10
            );
        }
        assert!(report.summaries.iter().all(|summary| {
            summary.success_count > 0
                && summary.final_residual_mean.is_finite()
                && summary.cell_b_relative_error_mean.is_finite()
        }));
    }
}
