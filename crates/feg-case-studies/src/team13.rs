//! Linear probabilistic TEAM 13 magnetostatic model.
//!
//! The FEEC state is a Whitney 1-cochain `a_h` with `b_h = d a_h` and a
//! weighted Coulomb-gauge auxiliary 0-cochain `sigma_h`. The mixed system is
//! the reluctivity-weighted Hodge-Laplacian:
//!
//! `(sigma_h, tau_h)_nu - (a_h, d tau_h)_nu = 0`
//!
//! `(d sigma_h, v_h)_nu + (d a_h, d v_h)_nu = integral J(theta) dot v dx`
//!
//! Hard essential conditions are applied only on the outer truncation boundary
//! for both `a_h` and `sigma_h`; the half-domain plane `z = 0` is natural.
//! There is no diagonal PDE regularization term.
//!
//! The selected linear B/H curve is `H = nu B` with `nu_air = 1/mu0` in
//! air/coils and `nu_iron = 342` in iron.

use crate::{
    team13_material::{
        Team13BhSample, Team13SmoothIronReluctivityLaw, Team13TabulatedReluctivityLaw,
    },
    visual_output,
};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::{cochain::Cochain, ManifoldComplexExt};
use exterior::field::{DiffFormClosure, ExteriorField};
use feg_core::{
    GaussianPriorSpec, LinearGaussianMeasurementSpec, LinearUncertainInputSpec,
    NonlinearResidualEvaluation, NonlinearResidualModel, RepresentationPreference, SparseTriplet,
    SparseTripletMatrix,
};
use feg_infer::sparse::core_triplet_to_gmrf as sparse_from_core;
use feg_infer::{
    adapters::FeecResidualAdapter,
    core_triplet_to_feec_csr, lift_vector_with_layout,
    linear_pde::{
        solve_linear_pde_uq_with_config, LinearPdeDerivedMarginalResult,
        LinearPdeDerivedQuantitySpec, LinearPdeJointDerivedQuantitySpec,
        LinearPdeJointMeasurementSpec, LinearPdeLatentDerivedBlockSpec,
        LinearPdeLatentInputPosterior, LinearPdeLatentMeasurementBlockSpec,
        LinearPdePrecisionPolicy, LinearPdeUqProblem, LinearPdeUqResult, LinearPdeUqSolverConfig,
        LinearPdeVarianceConfig, LinearPdeVarianceMode,
    },
    nonlinear::{
        diagnose_gauss_newton_first_step, solve_nonlinear_laplace, solve_square_nonlinear_system,
        GaussNewtonConfig, GaussNewtonFirstStepDiagnostics, GaussNewtonIteration,
        GaussNewtonLinearSolve, GaussNewtonLinearSolveMode, GaussNewtonRunDiagnostics,
        GaussNewtonStepRegularization, GaussianNoiseModel, LaplaceFactorizationStats,
        NonlinearAssemblyStats, NonlinearAssemblyTermKind, NonlinearLaplaceProblem,
        NonlinearLaplaceResult, NonlinearResidualReport, NonlinearResidualTerm,
        SmoothAbsLinearResidualModel, SmoothGroupedNormLinearResidualModel,
        SmoothGroupedNormObservation, SmoothGroupedNormSample, SquareNewtonConfig,
        SquareNewtonIteration,
    },
    prior::{
        exact_potential::{
            build_exact_two_form_potential_prior_with_metric, ExactTwoFormPotentialPriorConfig,
        },
        matern::one_form::{
            build_matern_precision_1form_with_mass_inverse_for_alpha,
            build_reconstructed_barycenter_field_operator,
            build_split_graph_matern_precision_1form_with_mass_inverse_for_alpha,
            HodgeLaplacian1Form, MaternAlpha, MaternMassInverse,
        },
    },
    reduce_vector_with_layout,
    sparse::{
        feec_csr_to_gmrf, restrict_columns_and_fold_fixed,
        sparse_row_operator_apply_feec as apply_operator_to_feec, sparse_row_to_triplet,
        symmetrize_feec_csr, triplet_to_sparse_row_operator,
    },
};
use formoniq::{
    assemble::{self, assemble_galvec, assemble_whitney_projected_sparse_inverse_galmat_weighted},
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::{
        hodge_laplace::{self, MixedGalmats},
        nonlinear_magnetostatic::{
            build_reduced_vector_potential_magnetostatic_3d, NonlinearMagnetostaticAssemblyConfig,
            ReducedVectorPotentialMagnetostatic3d, SpatialReluctivity,
        },
        reduced_linear::{
            build_reduced_hodge_laplace_1form_system,
            build_reduced_hodge_laplace_1form_system_with_galmats,
            reduce_reduced_hodge_laplace_1form_rhs_with_galmats,
        },
        residual::ResidualModel as FeecResidualModel,
    },
    reduction::{DofLayout, EssentialBoundarySpec, PrescribedDof},
};
use gmrf_core::{
    estimate_hutchinson_transformed_variances, estimate_monte_carlo_transformed_variances,
    exact_solve_diag, exact_solve_transformed_diag, ht_weighted_h,
    selected_inverse_transformed_diag,
    types::{SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector},
    ProbeDistribution, SparseCholeskyFactor, SparseRowOperator,
};
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
        handle::{KSimplexIdx, SimplexHandle, SimplexIdx},
        simplex::Simplex,
    },
};
use std::{
    collections::{BTreeMap, HashSet},
    f64::consts::PI,
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

pub const MU0: f64 = 4.0 * PI * 1e-7;
pub const NU_AIR: f64 = 1.0 / MU0;
pub const NU_IRON: f64 = 342.0;
pub const MU_R_IRON: f64 = 1.0 / (NU_IRON * MU0);
pub const TEAM13_OBSERVATION_COUNT: usize = 25;
pub const TEAM13_BENCHMARK_MEASUREMENT_COUNT: usize = 40;

pub const A_VECTOR_X_DERIVED_NAME: &str = "A_vector_x";
pub const A_VECTOR_Y_DERIVED_NAME: &str = "A_vector_y";
pub const A_VECTOR_Z_DERIVED_NAME: &str = "A_vector_z";
pub const B_COCHAIN_DERIVED_NAME: &str = "B_cochain";
pub const B_VECTOR_X_DERIVED_NAME: &str = "B_vector_x";
pub const B_VECTOR_Y_DERIVED_NAME: &str = "B_vector_y";
pub const B_VECTOR_Z_DERIVED_NAME: &str = "B_vector_z";
const A_PHYSICAL_COCHAIN_DERIVED_NAME: &str = "A_physical_cochain";
pub const SOURCE_ALPHA_INPUT_NAME: &str = "source_alpha";
pub const SOURCE_MODE_INPUT_NAME: &str = "team13_coil_modes";
pub const DEFAULT_SOURCE_ALPHA_TRUE: f64 = 1.15;
pub const DEFAULT_SOURCE_ALPHA_TRUE_MODES: [f64; 8] =
    [1.15, 1.08, 0.93, 1.04, 1.12, 0.88, 1.05, 0.97];
pub const TEAM13_COIL_MODE_COUNT: usize = 8;

const COIL_MODE_COUNT: usize = TEAM13_COIL_MODE_COUNT;
const EPS: f64 = 1e-12;
const SENSOR_DERIVED_PREFIX: &str = "sensor::";
const PERIOD_DERIVED_PREFIX: &str = "period::";
const SOURCE_FREE_H_TOP_LOOP_NAME: &str = "source_free_air_top_h_circulation";
const SOURCE_RECOVERY_STATE_PRIOR_PRECISION_SCALE: f64 = 1e-12;
const EIGHT_MODE_SOURCE_ERROR_PRIOR_STD_FRACTION_GATE: f64 = 0.5;
const EIGHT_MODE_STRONG_SHRINKAGE_VARIANCE_RATIO_GATE: f64 = 0.75;
const EIGHT_MODE_FIELD_RELATIVE_ERROR_GATE: f64 = 0.05;
const TEAM13_NGSOLVE_SURFACE_X_SAMPLES: usize = 100;
const TEAM13_NGSOLVE_SURFACE_Y_SAMPLES: usize = 300;
const TEAM13_NGSOLVE_SURFACE_Z_SAMPLES: usize = 100;
const TEAM13_FACE_COHERENT_COORD_TOL: f64 = 1.0e-8;
const TEAM13_FACE_COHERENT_NORMAL_TOL: f64 = 1.0e-8;
const TEAM13_FACE_COHERENT_AREA_REL_TOL: f64 = 1.0e-6;
const TEAM13_FACE_COHERENT_AREA_ABS_TOL: f64 = 1.0e-12;
const TEAM13_STEEL_OBSERVATION_CACHE_VERSION: u64 = 1;
const TEAM13_STEEL_OBSERVATION_CACHE_DIR: &str = "target/team13_observation_cache";
const TEAM13_STEEL_OBSERVATION_CACHE_MAGIC: &[u8; 16] = b"TEAM13STEELC0001";
const TEAM13_TRUTH_CACHE_VERSION: u64 = 1;
const TEAM13_TRUTH_CACHE_DIR: &str = "target/team13_truth_cache";
const TEAM13_TRUTH_CACHE_MAGIC: &[u8; 16] = b"TEAM13TRUTHC0001";
const TEAM13_NGSOLVE_MESH_PATH: &str = "target/team13_ngsolve_mesh/team13_ngsolve_half_msh4.msh";
const TEAM13_NGSOLVE_MEASUREMENT_SLICES_MESH_PATH: &str =
    "target/team13_ngsolve_measurement_slices_mesh/team13_ngsolve_half_measurement_slices_msh4.msh";
const TEAM13_DETERMINISTIC_LINEAR_OUTPUT_DIR: &str =
    "target/team13_deterministic_benchmark/feec_linear";
const TEAM13_NGSOLVE_LINEAR_REFERENCE_DIR: &str =
    "../ngsolve/out/team13_linear_parity_order1_curve1";

static TEAM13_NGSOLVE_BH_SAMPLES: &[Team13BhSample] = &[
    Team13BhSample {
        b_tesla: 0.0,
        h_ampere_per_meter: 0.0,
    },
    Team13BhSample {
        b_tesla: 0.0025,
        h_ampere_per_meter: 16.0,
    },
    Team13BhSample {
        b_tesla: 0.005,
        h_ampere_per_meter: 30.0,
    },
    Team13BhSample {
        b_tesla: 0.0125,
        h_ampere_per_meter: 54.0,
    },
    Team13BhSample {
        b_tesla: 0.025,
        h_ampere_per_meter: 93.0,
    },
    Team13BhSample {
        b_tesla: 0.05,
        h_ampere_per_meter: 143.0,
    },
    Team13BhSample {
        b_tesla: 0.1,
        h_ampere_per_meter: 191.0,
    },
    Team13BhSample {
        b_tesla: 0.2,
        h_ampere_per_meter: 210.0,
    },
    Team13BhSample {
        b_tesla: 0.3,
        h_ampere_per_meter: 222.0,
    },
    Team13BhSample {
        b_tesla: 0.4,
        h_ampere_per_meter: 233.0,
    },
    Team13BhSample {
        b_tesla: 0.5,
        h_ampere_per_meter: 247.0,
    },
    Team13BhSample {
        b_tesla: 0.6,
        h_ampere_per_meter: 258.0,
    },
    Team13BhSample {
        b_tesla: 0.7,
        h_ampere_per_meter: 272.0,
    },
    Team13BhSample {
        b_tesla: 0.8,
        h_ampere_per_meter: 289.0,
    },
    Team13BhSample {
        b_tesla: 0.9,
        h_ampere_per_meter: 313.0,
    },
    Team13BhSample {
        b_tesla: 1.0,
        h_ampere_per_meter: 342.0,
    },
    Team13BhSample {
        b_tesla: 1.1,
        h_ampere_per_meter: 377.0,
    },
    Team13BhSample {
        b_tesla: 1.2,
        h_ampere_per_meter: 433.0,
    },
    Team13BhSample {
        b_tesla: 1.3,
        h_ampere_per_meter: 509.0,
    },
    Team13BhSample {
        b_tesla: 1.4,
        h_ampere_per_meter: 648.0,
    },
    Team13BhSample {
        b_tesla: 1.5,
        h_ampere_per_meter: 933.0,
    },
    Team13BhSample {
        b_tesla: 1.55,
        h_ampere_per_meter: 1228.0,
    },
    Team13BhSample {
        b_tesla: 1.6,
        h_ampere_per_meter: 1934.0,
    },
    Team13BhSample {
        b_tesla: 1.65,
        h_ampere_per_meter: 2913.0,
    },
    Team13BhSample {
        b_tesla: 1.7,
        h_ampere_per_meter: 4993.0,
    },
    Team13BhSample {
        b_tesla: 1.75,
        h_ampere_per_meter: 7189.0,
    },
    Team13BhSample {
        b_tesla: 1.8,
        h_ampere_per_meter: 9423.0,
    },
    Team13BhSample {
        b_tesla: 1.8,
        h_ampere_per_meter: 9423.0,
    },
    Team13BhSample {
        b_tesla: 1.86530612,
        h_ampere_per_meter: 12820.3768,
    },
    Team13BhSample {
        b_tesla: 1.93061224,
        h_ampere_per_meter: 16544.7489,
    },
    Team13BhSample {
        b_tesla: 1.99591837,
        h_ampere_per_meter: 20716.3957,
    },
    Team13BhSample {
        b_tesla: 2.06122449,
        h_ampere_per_meter: 25550.0961,
    },
    Team13BhSample {
        b_tesla: 2.12653061,
        h_ampere_per_meter: 31520.6135,
    },
    Team13BhSample {
        b_tesla: 2.19183673,
        h_ampere_per_meter: 40320.4637,
    },
    Team13BhSample {
        b_tesla: 2.25714286,
        h_ampere_per_meter: 77303.8295,
    },
    Team13BhSample {
        b_tesla: 2.32244898,
        h_ampere_per_meter: 129272.791,
    },
    Team13BhSample {
        b_tesla: 2.3877551,
        h_ampere_per_meter: 181241.752,
    },
    Team13BhSample {
        b_tesla: 2.45306122,
        h_ampere_per_meter: 233210.713,
    },
    Team13BhSample {
        b_tesla: 2.51836735,
        h_ampere_per_meter: 285179.674,
    },
    Team13BhSample {
        b_tesla: 2.58367347,
        h_ampere_per_meter: 337148.635,
    },
    Team13BhSample {
        b_tesla: 2.64897959,
        h_ampere_per_meter: 389117.596,
    },
    Team13BhSample {
        b_tesla: 2.71428571,
        h_ampere_per_meter: 441086.557,
    },
    Team13BhSample {
        b_tesla: 2.77959184,
        h_ampere_per_meter: 493055.518,
    },
    Team13BhSample {
        b_tesla: 2.84489796,
        h_ampere_per_meter: 545024.479,
    },
    Team13BhSample {
        b_tesla: 2.91020408,
        h_ampere_per_meter: 596993.44,
    },
    Team13BhSample {
        b_tesla: 2.9755102,
        h_ampere_per_meter: 648962.401,
    },
    Team13BhSample {
        b_tesla: 3.04081633,
        h_ampere_per_meter: 700931.362,
    },
    Team13BhSample {
        b_tesla: 3.10612245,
        h_ampere_per_meter: 752900.323,
    },
    Team13BhSample {
        b_tesla: 3.17142857,
        h_ampere_per_meter: 804869.284,
    },
    Team13BhSample {
        b_tesla: 3.23673469,
        h_ampere_per_meter: 856838.245,
    },
    Team13BhSample {
        b_tesla: 3.30204082,
        h_ampere_per_meter: 908807.206,
    },
    Team13BhSample {
        b_tesla: 3.36734694,
        h_ampere_per_meter: 960776.167,
    },
    Team13BhSample {
        b_tesla: 3.43265306,
        h_ampere_per_meter: 1012745.13,
    },
    Team13BhSample {
        b_tesla: 3.49795918,
        h_ampere_per_meter: 1064714.09,
    },
    Team13BhSample {
        b_tesla: 3.56326531,
        h_ampere_per_meter: 1116683.05,
    },
    Team13BhSample {
        b_tesla: 3.62857143,
        h_ampere_per_meter: 1168652.01,
    },
    Team13BhSample {
        b_tesla: 3.69387755,
        h_ampere_per_meter: 1220620.97,
    },
    Team13BhSample {
        b_tesla: 3.75918367,
        h_ampere_per_meter: 1272589.93,
    },
    Team13BhSample {
        b_tesla: 3.8244898,
        h_ampere_per_meter: 1324558.89,
    },
    Team13BhSample {
        b_tesla: 3.88979592,
        h_ampere_per_meter: 1376527.85,
    },
    Team13BhSample {
        b_tesla: 3.95510204,
        h_ampere_per_meter: 1428496.82,
    },
    Team13BhSample {
        b_tesla: 4.02040816,
        h_ampere_per_meter: 1480465.78,
    },
    Team13BhSample {
        b_tesla: 4.08571429,
        h_ampere_per_meter: 1532434.74,
    },
    Team13BhSample {
        b_tesla: 4.15102041,
        h_ampere_per_meter: 1584403.7,
    },
    Team13BhSample {
        b_tesla: 4.21632653,
        h_ampere_per_meter: 1636372.66,
    },
    Team13BhSample {
        b_tesla: 4.28163265,
        h_ampere_per_meter: 1688341.62,
    },
    Team13BhSample {
        b_tesla: 4.34693878,
        h_ampere_per_meter: 1740310.58,
    },
    Team13BhSample {
        b_tesla: 4.4122449,
        h_ampere_per_meter: 1792279.54,
    },
    Team13BhSample {
        b_tesla: 4.47755102,
        h_ampere_per_meter: 1844248.5,
    },
    Team13BhSample {
        b_tesla: 4.54285714,
        h_ampere_per_meter: 1896217.46,
    },
    Team13BhSample {
        b_tesla: 4.60816327,
        h_ampere_per_meter: 1948186.43,
    },
    Team13BhSample {
        b_tesla: 4.67346939,
        h_ampere_per_meter: 2000155.39,
    },
    Team13BhSample {
        b_tesla: 4.73877551,
        h_ampere_per_meter: 2052124.35,
    },
    Team13BhSample {
        b_tesla: 4.80408163,
        h_ampere_per_meter: 2104093.31,
    },
    Team13BhSample {
        b_tesla: 4.86938776,
        h_ampere_per_meter: 2156062.27,
    },
    Team13BhSample {
        b_tesla: 4.93469388,
        h_ampere_per_meter: 2208031.23,
    },
    Team13BhSample {
        b_tesla: 5.0,
        h_ampere_per_meter: 2260000.19,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13DomainMode {
    HalfZNonnegative,
    Full,
}

impl Team13DomainMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HalfZNonnegative => "half",
            Self::Full => "full",
        }
    }

    fn solid_z_min(self) -> f64 {
        match self {
            Self::HalfZNonnegative => 0.0,
            Self::Full => -0.05,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13MeasurementMode {
    BenchmarkExact,
    FaceCochain,
    LegacyBand,
}

impl Team13MeasurementMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BenchmarkExact => "exact",
            Self::FaceCochain => "face-cochain",
            Self::LegacyBand => "legacy-band",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13PdeResidualWeighting {
    Euclidean,
    MassInverse,
    MassInverseTraceNormalized,
}

impl Team13PdeResidualWeighting {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::MassInverse => "mass-inverse",
            Self::MassInverseTraceNormalized => "mass-inverse-trace-normalized",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13NonlinearMaterialKind {
    NgsolveTabulatedLinear,
    SmoothQuadratic,
}

impl Team13NonlinearMaterialKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NgsolveTabulatedLinear => "ngsolve-tabulated-linear",
            Self::SmoothQuadratic => "smooth-quadratic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13DiscrepancyPriorKind {
    Flat,
    WeightedWhittle,
}

impl Team13DiscrepancyPriorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::WeightedWhittle => "weighted-whittle",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Team13FieldPriorKind {
    UnweightedHodgeMatern,
    SplitGraphHodgeMatern,
}

impl Team13FieldPriorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnweightedHodgeMatern => "unweighted-hodge-matern",
            Self::SplitGraphHodgeMatern => "split-graph-hodge-matern",
        }
    }

    fn stage_slug(self) -> &'static str {
        match self {
            Self::UnweightedHodgeMatern => "unweighted",
            Self::SplitGraphHodgeMatern => "split_graph",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13LinearConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub coil_relative_std: f64,
    pub pde_variance: f64,
    pub b_observation_std_tesla: f64,
    pub measurement_mode: Team13MeasurementMode,
    pub legacy_measurement_band: f64,
    pub observation_csv_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub solver: LinearPdeUqSolverConfig,
}

impl Default for Team13LinearConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("target/team13_linear/team13_half_linear.msh"),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            coil_relative_std: 0.05,
            pde_variance: 1e-8,
            b_observation_std_tesla: 0.02,
            measurement_mode: Team13MeasurementMode::BenchmarkExact,
            legacy_measurement_band: 0.03,
            observation_csv_path: None,
            output_dir: None,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::Exact,
                    num_variance_probes: 32,
                    variance_batch_count: 4,
                    rng_seed: 13,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: true,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13SourceRecoveryConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub source_alpha_true: f64,
    pub source_alpha_true_modes: [f64; TEAM13_COIL_MODE_COUNT],
    pub run_eight_mode_recovery: bool,
    pub source_prior_std: f64,
    pub pde_variance: f64,
    pub pde_residual_weighting: Team13PdeResidualWeighting,
    pub discrepancy_prior: Team13DiscrepancyPriorKind,
    pub discrepancy_prior_precision_scale: f64,
    pub field_prior_precision_scale: f64,
    pub field_priors: Vec<Team13FieldPriorKind>,
    pub b_observation_std_tesla: f64,
    pub measurement_mode: Team13MeasurementMode,
    pub legacy_measurement_band: f64,
    pub nominal_observation_csv_path: Option<PathBuf>,
    pub perturbed_observation_csv_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub solver: LinearPdeUqSolverConfig,
}

impl Default for Team13SourceRecoveryConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("target/team13_linear/team13_half_linear.msh"),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            source_alpha_true: DEFAULT_SOURCE_ALPHA_TRUE,
            source_alpha_true_modes: DEFAULT_SOURCE_ALPHA_TRUE_MODES,
            run_eight_mode_recovery: false,
            source_prior_std: 0.10,
            pde_variance: 1e-6,
            pde_residual_weighting: Team13PdeResidualWeighting::Euclidean,
            discrepancy_prior: Team13DiscrepancyPriorKind::Flat,
            discrepancy_prior_precision_scale: SOURCE_RECOVERY_STATE_PRIOR_PRECISION_SCALE,
            field_prior_precision_scale: SOURCE_RECOVERY_STATE_PRIOR_PRECISION_SCALE,
            field_priors: vec![
                Team13FieldPriorKind::UnweightedHodgeMatern,
                Team13FieldPriorKind::SplitGraphHodgeMatern,
            ],
            b_observation_std_tesla: 0.02,
            measurement_mode: Team13MeasurementMode::BenchmarkExact,
            legacy_measurement_band: 0.03,
            nominal_observation_csv_path: None,
            perturbed_observation_csv_path: None,
            output_dir: None,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::Exact,
                    num_variance_probes: 32,
                    variance_batch_count: 4,
                    rng_seed: 115,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: true,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13SourceRecoveryStageSummary {
    pub name: String,
    pub latent_dimension: usize,
    pub pde_residual_norm: f64,
    pub sensor_rmse: f64,
    pub field_reference: String,
    pub a_active_rmse: f64,
    pub a_active_relative_l2_error: f64,
    pub b_cochain_rmse: f64,
    pub b_cochain_relative_l2_error: f64,
    pub b_vector_rmse: f64,
    pub b_vector_relative_l2_error: f64,
    pub a_variance_ratio_mean: f64,
    pub b_variance_ratio_mean: f64,
    pub b_vector_trace_variance_ratio_mean: f64,
}

#[derive(Debug, Clone)]
pub struct Team13SourcePosteriorSummary {
    pub prior_mean: f64,
    pub prior_variance: f64,
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
    pub true_alpha: f64,
    pub recovery_error: f64,
}

#[derive(Debug, Clone)]
pub struct Team13SourceModePosteriorSummary {
    pub mode_index: usize,
    pub mode_name: String,
    pub prior_mean: f64,
    pub prior_variance: f64,
    pub true_alpha: f64,
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
    pub recovery_error: f64,
    pub z_score: f64,
}

#[derive(Debug, Clone)]
pub struct Team13FieldPriorComparison {
    pub prior_kind: Team13FieldPriorKind,
    pub stage_name: String,
    pub source_posterior: Team13SourcePosteriorSummary,
    pub sensor_rmse: f64,
    pub a_active_relative_l2_error: f64,
    pub b_cochain_relative_l2_error: f64,
    pub b_vector_relative_l2_error: f64,
    pub a_variance_ratio_mean: f64,
    pub b_variance_ratio_mean: f64,
    pub b_vector_trace_variance_ratio_mean: f64,
    pub prior_factor_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub all_finite_variances: bool,
    pub nonnegative_variances: bool,
}

#[derive(Debug, Clone)]
pub struct Team13SourceModeObservabilitySummary {
    pub mode_index: usize,
    pub mode_name: String,
    pub sensor_sensitivity_norm: f64,
    pub posterior_shrinkage: f64,
    pub identifiable: bool,
}

#[derive(Debug, Clone)]
pub struct Team13ReplacementDecision {
    pub technical_pass: bool,
    pub recommendation: String,
    pub all_finite: bool,
    pub variances_shrink: bool,
    pub source_recovery_convincing: bool,
    pub field_recovery_convincing: bool,
    pub modes_within_two_sigma: usize,
    pub modes_beyond_three_sigma: usize,
    pub modes_within_half_prior_std: usize,
    pub modes_with_strong_variance_shrinkage: usize,
    pub sensor_rmse_improves: bool,
    pub fixed_source_sensor_rmse: f64,
    pub eight_mode_sensor_rmse: f64,
    pub fixed_source_b_vector_relative_l2_error: f64,
    pub eight_mode_b_vector_relative_l2_error: f64,
}

#[derive(Debug, Clone)]
pub struct Team13SensorUncertaintyReport {
    pub stage: String,
    pub name: String,
    pub area: f64,
    pub observed: f64,
    pub prediction: f64,
    pub residual: f64,
    pub prior_variance: f64,
    pub posterior_variance: f64,
}

#[derive(Debug, Clone)]
pub struct Team13PeriodDiagnosticReport {
    pub stage: String,
    pub name: String,
    pub truth: f64,
    pub prediction: f64,
    pub residual: f64,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct Team13SourceRecoveryStageResult {
    pub summary: Team13SourceRecoveryStageSummary,
    pub solve: Team13LinearSolveResult,
    pub sensor_uncertainty: Vec<Team13SensorUncertaintyReport>,
    pub period_diagnostics: Vec<Team13PeriodDiagnosticReport>,
}

#[derive(Debug, Clone)]
pub struct Team13EightModeRecoveryResult {
    pub stage: Team13SourceRecoveryStageResult,
    pub fluctuation_stage: Option<Team13SourceRecoveryStageResult>,
    pub source_modes: Vec<Team13SourceModePosteriorSummary>,
    pub fluctuation_source_modes: Vec<Team13SourceModePosteriorSummary>,
    pub observability: Vec<Team13SourceModeObservabilitySummary>,
    pub decision: Team13ReplacementDecision,
}

#[derive(Debug, Clone)]
pub struct Team13SourceRecoveryResult {
    pub stages: Vec<Team13SourceRecoveryStageResult>,
    pub field_prior_comparisons: Vec<Team13FieldPriorComparison>,
    pub source_posterior: Team13SourcePosteriorSummary,
    pub source_scaling_proxy: Team13SourcePosteriorSummary,
    pub baseline_source_posterior: Team13SourcePosteriorSummary,
    pub fluctuation_source_posterior: Team13SourcePosteriorSummary,
    pub eight_mode: Option<Team13EightModeRecoveryResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13CoilRegion {
    BrickBack,
    BrickFront,
    BrickLeft,
    BrickRight,
    CornerRightBack,
    CornerLeftBack,
    CornerLeftFront,
    CornerRightFront,
}

impl Team13CoilRegion {
    pub fn all() -> [Self; COIL_MODE_COUNT] {
        [
            Self::BrickBack,
            Self::BrickFront,
            Self::BrickLeft,
            Self::BrickRight,
            Self::CornerRightBack,
            Self::CornerLeftBack,
            Self::CornerLeftFront,
            Self::CornerRightFront,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::BrickBack => "brick_back",
            Self::BrickFront => "brick_front",
            Self::BrickLeft => "brick_left",
            Self::BrickRight => "brick_right",
            Self::CornerRightBack => "corner_right_back",
            Self::CornerLeftBack => "corner_left_back",
            Self::CornerLeftFront => "corner_left_front",
            Self::CornerRightFront => "corner_right_front",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13SensorReport {
    pub name: String,
    pub observed: f64,
    pub nominal_prediction: f64,
    pub posterior_prediction: f64,
    pub residual: f64,
    pub linearization_direction: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct Team13BenchmarkReport {
    pub name: String,
    pub observed: Option<f64>,
    pub nominal_prediction: f64,
    pub posterior_prediction: f64,
}

#[derive(Debug, Clone)]
pub struct Team13VectorPushforward {
    pub name: String,
    pub mean_vectors: Vec<[f64; 3]>,
    pub prior_trace_variance: FeecVector,
    pub posterior_trace_variance: FeecVector,
    pub trace_variance_ratio: FeecVector,
}

#[derive(Debug, Clone)]
pub struct Team13LinearSolveResult {
    pub domain_mode: Team13DomainMode,
    pub posterior: LinearPdeUqResult,
    pub nominal_a: FeecVector,
    pub field_reference_name: String,
    pub field_reference_a: FeecVector,
    pub state_active_mask: FeecVector,
    pub a_variance_ratio: FeecVector,
    pub b_variance_ratio: FeecVector,
    pub sensor_reports: Vec<Team13SensorReport>,
    pub benchmark_reports: Vec<Team13BenchmarkReport>,
    pub vector_pushforwards: BTreeMap<String, Team13VectorPushforward>,
}

#[derive(Debug, Clone)]
pub struct Team13LinearNominalResult {
    pub domain_mode: Team13DomainMode,
    pub nominal_a: FeecVector,
    pub benchmark_reports: Vec<Team13BenchmarkReport>,
}

#[derive(Debug, Clone)]
pub struct Team13NonlinearConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub pde_variance: f64,
    pub field_prior_kind: Team13FieldPriorKind,
    pub field_prior_precision_scale: f64,
    pub prior_kappa: Option<f64>,
    pub prior_tau: f64,
    pub prior_allow_kappa_fallback: bool,
    pub max_iterations: usize,
    pub b_observation_std_tesla: f64,
    pub measurement_mode: Team13MeasurementMode,
    pub legacy_measurement_band: f64,
    pub observation_csv_path: Option<PathBuf>,
    pub assimilate_measurements: bool,
    pub output_dir: Option<PathBuf>,
    pub write_outputs: bool,
    pub sensor_variance_count: usize,
    pub linear_solve: GaussNewtonLinearSolve,
    pub step_regularization: GaussNewtonStepRegularization,
    pub variance: LinearPdeVarianceConfig,
}

impl Default for Team13NonlinearConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("target/team13_nonlinear/team13_half_nonlinear.msh"),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
            beta_iron: 10.0,
            b_scale_tesla: 1.0,
            pde_variance: 1.2e4,
            field_prior_kind: Team13FieldPriorKind::UnweightedHodgeMatern,
            field_prior_precision_scale: SOURCE_RECOVERY_STATE_PRIOR_PRECISION_SCALE,
            prior_kappa: None,
            prior_tau: 1.0,
            prior_allow_kappa_fallback: false,
            max_iterations: 12,
            b_observation_std_tesla: 0.02,
            measurement_mode: Team13MeasurementMode::BenchmarkExact,
            legacy_measurement_band: 0.03,
            observation_csv_path: None,
            assimilate_measurements: false,
            output_dir: None,
            write_outputs: true,
            sensor_variance_count: 5,
            linear_solve: GaussNewtonLinearSolve::DirectCholesky,
            step_regularization: GaussNewtonStepRegularization::LevenbergMarquardtGrid,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::MonteCarlo,
                num_variance_probes: 8,
                variance_batch_count: 1,
                rng_seed: 1313,
                local_rb_block_size: 16,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13NonlinearSensorVarianceReport {
    pub name: String,
    pub prior_variance: f64,
    pub posterior_variance: f64,
}

#[derive(Debug, Clone)]
pub struct Team13NonlinearSolveResult {
    pub domain_mode: Team13DomainMode,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub prior_kind: Team13FieldPriorKind,
    pub field_prior_precision_scale: f64,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_kappa_fallback_used: bool,
    pub initial_residual_norm: f64,
    pub linear_mean_residual_norm: f64,
    pub final_residual_norm: f64,
    pub map_distance_from_linear_mean: f64,
    pub map_relative_distance_from_linear_mean: f64,
    pub beta_zero_relative_error: Option<f64>,
    pub field_metrics_vs_linear: Team13FieldRecoveryMetrics,
    pub linear_sensor_rmse: f64,
    pub sensor_rmse: f64,
    pub sensor_rmse_improvement_ratio: f64,
    pub assimilated_measurements: usize,
    pub sensor_reports: Vec<Team13SensorReport>,
    pub sensor_variances: Vec<Team13NonlinearSensorVarianceReport>,
    pub benchmark_reports: Vec<Team13BenchmarkReport>,
    pub history: Vec<GaussNewtonIteration>,
    pub assembly: NonlinearAssemblyStats,
    pub final_factorization: LaplaceFactorizationStats,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct Team13NonlinearDiagnosticsConfig {
    pub solve: Team13NonlinearConfig,
    pub pde_variance_values: Vec<f64>,
}

impl Default for Team13NonlinearDiagnosticsConfig {
    fn default() -> Self {
        Self {
            solve: Team13NonlinearConfig {
                write_outputs: false,
                sensor_variance_count: 0,
                ..Team13NonlinearConfig::default()
            },
            pde_variance_values: vec![1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1.0e4, 1.2e4],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13FirstStepConditioningClass {
    WellConditioned,
    IllConditioned,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Team13FirstStepDiagnosticReport {
    pub pde_variance: f64,
    pub step_regularization: GaussNewtonStepRegularization,
    pub classification: Team13FirstStepConditioningClass,
    pub diagnostics: Option<GaussNewtonFirstStepDiagnostics>,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Team13NonlinearDiagnosticsResult {
    pub domain_mode: Team13DomainMode,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub reduced_physical_rhs_norm: f64,
    pub beta_zero_residual_norm: f64,
    pub source_free_affine_solve_residual_norm: f64,
    pub nonlinear_residual_at_linear_mean_norm: f64,
    pub prior_kind: Team13FieldPriorKind,
    pub field_prior_precision_scale: f64,
    pub assimilated_measurements: usize,
    pub first_steps: Vec<Team13FirstStepDiagnosticReport>,
}

#[derive(Debug, Clone)]
pub struct Team13JacobianSparsityAuditConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
}

impl Default for Team13JacobianSparsityAuditConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("target/team13_nonlinear/team13_half_nonlinear.msh"),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
            beta_iron: 10.0,
            b_scale_tesla: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Team13JacobianSparsityAuditResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub material_kind: Team13NonlinearMaterialKind,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub residual_dimension: usize,
    pub jacobian_rows: usize,
    pub jacobian_cols: usize,
    pub jacobian_nnz: usize,
    pub jacobian_lower_triangle_nnz: usize,
    pub jacobian_density: f64,
    pub jacobian_lower_triangle_density: f64,
    pub normal_rows: usize,
    pub normal_cols: usize,
    pub normal_nnz: usize,
    pub normal_lower_triangle_nnz: usize,
    pub normal_density: f64,
    pub normal_lower_triangle_density: f64,
    pub normal_to_jacobian_nnz_ratio: f64,
    pub reduced_physical_rhs_norm: f64,
    pub linear_mean_residual_norm: f64,
    pub nonlinear_residual_at_linear_mean_norm: f64,
    pub jacobian_seconds: f64,
    pub normal_product_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13SyntheticObservationModelKind {
    SmoothMagnitude,
    SignedLinearProxy,
}

impl Team13SyntheticObservationModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SmoothMagnitude => "smooth_magnitude",
            Self::SignedLinearProxy => "signed_linear_proxy",
        }
    }

    pub fn default_comparison() -> Vec<Self> {
        vec![Self::SmoothMagnitude, Self::SignedLinearProxy]
    }
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticNonlinearBaselineConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub truth_prior_precision: f64,
    pub truth_pde_variance: f64,
    pub pde_variance: f64,
    pub observation_std_tesla: f64,
    pub magnitude_smoothing_tesla: f64,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_diagonal_shift: f64,
    pub observation_models: Vec<Team13SyntheticObservationModelKind>,
    pub truth_max_iterations: usize,
    pub max_iterations: usize,
    pub linear_solve: GaussNewtonLinearSolve,
    pub step_regularization: GaussNewtonStepRegularization,
    pub variance: LinearPdeVarianceConfig,
}

impl Default for Team13SyntheticNonlinearBaselineConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("target/team13_nonlinear/team13_half_synthetic_baseline.msh"),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            beta_iron: 10.0,
            b_scale_tesla: 1.0,
            truth_prior_precision: 1.0e-16,
            truth_pde_variance: 1.0e-10,
            pde_variance: 1.0e-8,
            observation_std_tesla: 1.0e-4,
            magnitude_smoothing_tesla: 1.0e-8,
            prior_kappa: 1.0,
            prior_tau: 1.0e-6,
            prior_diagonal_shift: 1.0e-12,
            observation_models: Team13SyntheticObservationModelKind::default_comparison(),
            truth_max_iterations: 24,
            max_iterations: 24,
            linear_solve: GaussNewtonLinearSolve::DirectCholesky,
            step_regularization: GaussNewtonStepRegularization::LevenbergMarquardtGrid,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::Exact,
                num_variance_probes: 1,
                variance_batch_count: 1,
                rng_seed: 1314,
                local_rb_block_size: 16,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticObservationRunResult {
    pub model_kind: Team13SyntheticObservationModelKind,
    pub synthetic_sensor_count: usize,
    pub sign_mismatch_count: usize,
    pub initial_relative_error: f64,
    pub posterior_relative_error: f64,
    pub initial_residual_norm: f64,
    pub truth_residual_norm: f64,
    pub posterior_residual_norm: f64,
    pub initial_sensor_rmse: f64,
    pub posterior_sensor_rmse: f64,
    pub sensor_rmse_improvement_ratio: f64,
    pub posterior_converged: bool,
    pub all_finite_variances: bool,
    pub nonnegative_variances: bool,
    pub assembly: NonlinearAssemblyStats,
    pub final_factorization: LaplaceFactorizationStats,
    pub posterior_history: Vec<GaussNewtonIteration>,
    pub sensor_reports: Vec<Team13SensorReport>,
    pub sensor_variances: Vec<Team13NonlinearSensorVarianceReport>,
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticNonlinearBaselineResult {
    pub domain_mode: Team13DomainMode,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_diagonal_shift: f64,
    pub magnitude_smoothing_tesla: f64,
    pub synthetic_sensor_count: usize,
    pub initial_residual_norm: f64,
    pub truth_residual_norm: f64,
    pub truth_converged: bool,
    pub truth_history: Vec<SquareNewtonIteration>,
    pub observation_runs: Vec<Team13SyntheticObservationRunResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13SyntheticBenchmarkObservationGroup {
    SteelAverage,
    AirPoint,
}

impl Team13SyntheticBenchmarkObservationGroup {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SteelAverage => "steel_average",
            Self::AirPoint => "air_point",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13SteelSurfaceGroup {
    MidSheet,
    BackRightTop,
    BackRightEdge,
}

impl Team13SteelSurfaceGroup {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MidSheet => "mid_sheet",
            Self::BackRightTop => "back_right_top",
            Self::BackRightEdge => "back_right_edge",
        }
    }

    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "mid_sheet" => Ok(Self::MidSheet),
            "back_right_top" => Ok(Self::BackRightTop),
            "back_right_edge" => Ok(Self::BackRightEdge),
            other => Err(format!("unknown TEAM 13 steel surface group `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13SteelObservationQuadratureMode {
    NgsolveStyle,
    FaceCochain,
}

impl Team13SteelObservationQuadratureMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NgsolveStyle => "ngsolve-style",
            Self::FaceCochain => "face-cochain",
        }
    }
}

impl std::str::FromStr for Team13SteelObservationQuadratureMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ngsolve-style" | "ngsolve" | "quadrature" => Ok(Self::NgsolveStyle),
            "face-cochain" | "cochain" => Ok(Self::FaceCochain),
            other => Err(format!(
                "unknown TEAM 13 steel observation mode `{other}`; expected ngsolve-style or face-cochain"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13MapParityPriorKind {
    ExactPotential,
    OrdinaryMaternAlpha2,
    WeakRidge,
}

impl Team13MapParityPriorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactPotential => "exact-potential",
            Self::OrdinaryMaternAlpha2 => "ordinary-matern-alpha2",
            Self::WeakRidge => "weak-ridge",
        }
    }
}

impl std::str::FromStr for Team13MapParityPriorKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "exact-potential" | "exact" => Ok(Self::ExactPotential),
            "ordinary-matern-alpha2" | "ordinary-matern" | "matern-alpha2" | "ordinary" => {
                Ok(Self::OrdinaryMaternAlpha2)
            }
            "weak-ridge" | "ridge" => Ok(Self::WeakRidge),
            other => Err(format!(
                "unknown TEAM 13 MAP parity prior `{other}`; expected exact-potential, ordinary-matern-alpha2, or weak-ridge"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13MapParityPdeResidualKind {
    GaugeFixed,
    UngaugedCurl,
}

impl Team13MapParityPdeResidualKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GaugeFixed => "gauge-fixed",
            Self::UngaugedCurl => "ungauged-curl",
        }
    }
}

impl std::str::FromStr for Team13MapParityPdeResidualKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "gauge-fixed" | "gauge" | "coulomb-gauge" => Ok(Self::GaugeFixed),
            "ungauged-curl" | "ungauged" | "curl-only" | "no-gauge" => Ok(Self::UngaugedCurl),
            other => Err(format!(
                "unknown TEAM 13 MAP parity PDE residual `{other}`; expected gauge-fixed or ungauged-curl"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13PublishedSteelGap {
    G052,
    G047,
}

impl Team13PublishedSteelGap {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::G052 => "G=0.52mm",
            Self::G047 => "G=0.47mm",
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            Self::G052 => "g052",
            Self::G047 => "g047",
        }
    }

    pub fn steel_gap_m(self) -> f64 {
        match self {
            Self::G052 => 0.00052,
            Self::G047 => 0.00047,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticBenchmarkGeometryConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub truth_prior_precision: f64,
    pub truth_pde_variance: f64,
    pub pde_variance: f64,
    pub observation_std_tesla: f64,
    pub magnitude_smoothing_tesla: f64,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_diagonal_shift: f64,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub source_scale_diagnostic_values: Vec<f64>,
    pub sweep_pde_variances: Vec<f64>,
    pub sweep_observation_std_tesla: Vec<f64>,
    pub truth_max_iterations: usize,
    pub max_iterations: usize,
    pub linear_solve: GaussNewtonLinearSolve,
    pub step_regularization: GaussNewtonStepRegularization,
    pub variance: LinearPdeVarianceConfig,
}

impl Default for Team13SyntheticBenchmarkGeometryConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from(
                "target/team13_nonlinear/team13_half_synthetic_benchmark_geometry.msh",
            ),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
            beta_iron: 10.0,
            b_scale_tesla: 1.0,
            truth_prior_precision: 1.0e-16,
            truth_pde_variance: 1.0e-10,
            pde_variance: 1.0e4,
            observation_std_tesla: 1.0e-3,
            magnitude_smoothing_tesla: 1.0e-8,
            prior_kappa: 1.0,
            prior_tau: 1.0e-6,
            prior_diagonal_shift: 1.0e-12,
            steel_observation_quadrature: Team13SteelObservationQuadratureMode::NgsolveStyle,
            source_scale_diagnostic_values: vec![0.75, 0.875, 1.0, 1.125, 1.25],
            sweep_pde_variances: vec![1.0e2, 1.0e4, 1.0e6],
            sweep_observation_std_tesla: vec![1.0e-4, 1.0e-3, 1.0e-2],
            truth_max_iterations: 24,
            max_iterations: 24,
            linear_solve: GaussNewtonLinearSolve::DirectCholesky,
            step_regularization: GaussNewtonStepRegularization::LevenbergMarquardtGrid,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::Exact,
                num_variance_probes: 1,
                variance_batch_count: 1,
                rng_seed: 1315,
                local_rb_block_size: 16,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticBenchmarkObservationReport {
    pub name: String,
    pub group: Team13SyntheticBenchmarkObservationGroup,
    pub steel_surface_group: Option<Team13SteelSurfaceGroup>,
    pub observed: f64,
    pub initial_prediction: f64,
    pub posterior_prediction: f64,
    pub residual: f64,
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticBenchmarkObservationGroupSummary {
    pub group: Team13SyntheticBenchmarkObservationGroup,
    pub count: usize,
    pub initial_rmse: f64,
    pub posterior_rmse: f64,
    pub posterior_relative_rmse: f64,
    pub posterior_max_abs_residual: f64,
}

#[derive(Debug, Clone)]
pub struct Team13PriorVarianceDiagnostics {
    pub dimension: usize,
    pub min_variance: f64,
    pub max_variance: f64,
    pub max_to_min_variance_ratio: f64,
    pub all_finite: bool,
    pub nonnegative: bool,
    pub factor_nnz: usize,
}

#[derive(Debug, Clone)]
pub struct Team13PublishedSteelBenchmarkReport {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub observed_g_052: f64,
    pub observed_g_047: f64,
    pub nominal_prediction: f64,
    pub posterior_prediction: f64,
}

#[derive(Debug, Clone)]
pub struct Team13PublishedSteelGroupSummary {
    pub group: Team13SteelSurfaceGroup,
    pub count: usize,
    pub rmse_g_052: f64,
    pub rmse_g_047: f64,
    pub max_abs_residual_g_052: f64,
    pub max_abs_residual_g_047: f64,
}

#[derive(Debug, Clone)]
pub struct Team13SourceScaleDiagnosticRun {
    pub source_scale: f64,
    pub converged: bool,
    pub error: Option<String>,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub steel_rmse_g_052: f64,
    pub steel_rmse_g_047: f64,
    pub group_summaries: Vec<Team13PublishedSteelGroupSummary>,
}

#[derive(Debug, Clone)]
pub struct Team13ForwardBenchmarkDiagnosticResult {
    pub domain_mode: Team13DomainMode,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_kind: Team13NonlinearMaterialKind,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub observation_count: usize,
    pub assimilated_observation_count: usize,
    pub source_scale_diagnostics: Vec<Team13SourceScaleDiagnosticRun>,
}

#[derive(Debug, Clone)]
pub struct Team13SameMeshLinearParityConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13SameMeshLinearParityConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("target/team13_map_parity/team13_half_map_parity.msh"),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            steel_observation_quadrature: Team13SteelObservationQuadratureMode::NgsolveStyle,
            output_dir: Some(PathBuf::from("target/team13_same_mesh_linear_parity/feec")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13DeterministicBenchmarkConfig {
    pub linear: Team13SameMeshLinearParityConfig,
    pub ngsolve_linear_reference_dir: PathBuf,
    pub nonlinear: Option<Team13NonlinearForwardParityConfig>,
    pub ngsolve_nonlinear_reference_dir: Option<PathBuf>,
}

impl Default for Team13DeterministicBenchmarkConfig {
    fn default() -> Self {
        let mut linear = Team13SameMeshLinearParityConfig::default();
        linear.mesh_path = PathBuf::from(TEAM13_NGSOLVE_MESH_PATH);
        linear.output_dir = Some(PathBuf::from(TEAM13_DETERMINISTIC_LINEAR_OUTPUT_DIR));

        Self {
            linear,
            ngsolve_linear_reference_dir: PathBuf::from(TEAM13_NGSOLVE_LINEAR_REFERENCE_DIR),
            nonlinear: None,
            ngsolve_nonlinear_reference_dir: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13NonlinearForwardParityConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub magnitude_smoothing_tesla: f64,
    pub max_iterations: usize,
    pub linear_solve: GaussNewtonLinearSolve,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13NonlinearForwardParityConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from(TEAM13_NGSOLVE_MESH_PATH),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
            beta_iron: 10.0,
            b_scale_tesla: 1.0,
            magnitude_smoothing_tesla: 1.0e-8,
            max_iterations: 24,
            linear_solve: GaussNewtonLinearSolve::DirectCholesky,
            steel_observation_quadrature: Team13SteelObservationQuadratureMode::NgsolveStyle,
            output_dir: Some(PathBuf::from("target/team13_nonlinear_forward_parity/feec")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13MapParityConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub pde_variance: f64,
    pub observation_std_tesla: f64,
    pub magnitude_smoothing_tesla: f64,
    pub prior_kind: Team13MapParityPriorKind,
    pub pde_residual_kind: Team13MapParityPdeResidualKind,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_diagonal_shift: f64,
    pub sweep_pde_variances: Vec<f64>,
    pub sweep_observation_std_tesla: Vec<f64>,
    pub truth_max_iterations: usize,
    pub max_iterations: usize,
    pub linear_solve: GaussNewtonLinearSolve,
    pub step_regularization: GaussNewtonStepRegularization,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub variance: LinearPdeVarianceConfig,
    pub estimate_latent_variance: bool,
    pub force_truth_solve: bool,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13MapParityConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from("target/team13_map_parity/team13_half_map_parity.msh"),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
            beta_iron: 10.0,
            b_scale_tesla: 1.0,
            pde_variance: 1.0e4,
            observation_std_tesla: 1.0e-3,
            magnitude_smoothing_tesla: 1.0e-8,
            prior_kind: Team13MapParityPriorKind::ExactPotential,
            pde_residual_kind: Team13MapParityPdeResidualKind::UngaugedCurl,
            prior_kappa: 1.0,
            prior_tau: 1.0e-6,
            prior_diagonal_shift: 1.0e-12,
            sweep_pde_variances: vec![1.0e2, 1.0e4, 1.0e6],
            sweep_observation_std_tesla: vec![1.0e-4, 1.0e-3, 1.0e-2],
            truth_max_iterations: 24,
            max_iterations: 24,
            linear_solve: GaussNewtonLinearSolve::DirectCholesky,
            step_regularization: GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt,
            steel_observation_quadrature: Team13SteelObservationQuadratureMode::NgsolveStyle,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::MonteCarlo,
                num_variance_probes: 8,
                variance_batch_count: 1,
                rng_seed: 1317,
                local_rb_block_size: 16,
            },
            estimate_latent_variance: false,
            force_truth_solve: false,
            output_dir: Some(PathBuf::from("target/team13_map_parity")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13OperatorUncertaintyTangentKind {
    Nonlinear,
    LinearBetaZero,
}

impl Team13OperatorUncertaintyTangentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nonlinear => "nonlinear",
            Self::LinearBetaZero => "linear-beta-zero",
        }
    }
}

impl std::str::FromStr for Team13OperatorUncertaintyTangentKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "nonlinear" | "nonlinear-tabulated" | "tabulated" => Ok(Self::Nonlinear),
            "linear" | "linear-beta-zero" | "beta-zero" => Ok(Self::LinearBetaZero),
            other => Err(format!(
                "unknown TEAM 13 operator-uncertainty tangent `{other}`; expected nonlinear or linear-beta-zero"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13OperatorUncertaintyConfig {
    pub mesh_path: PathBuf,
    pub domain_mode: Team13DomainMode,
    pub ampere_turns: f64,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub material_log_iron_nu_scale: f64,
    pub tangent_kind: Team13OperatorUncertaintyTangentKind,
    pub prior_kind: Team13MapParityPriorKind,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_diagonal_shift: f64,
    pub pde_residual_kind: Team13MapParityPdeResidualKind,
    pub pde_residual_weighting: Team13PdeResidualWeighting,
    pub pde_variance: f64,
    pub include_steel_observations: bool,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub observation_std_tesla: f64,
    pub truth_max_iterations: usize,
    pub linear_solve: GaussNewtonLinearSolve,
    pub field_variance: LinearPdeVarianceConfig,
    pub estimate_field_variance: bool,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13OperatorUncertaintyConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from(TEAM13_NGSOLVE_MEASUREMENT_SLICES_MESH_PATH),
            domain_mode: Team13DomainMode::HalfZNonnegative,
            ampere_turns: 1000.0,
            material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
            beta_iron: 10.0,
            b_scale_tesla: 1.0,
            material_log_iron_nu_scale: 0.0,
            tangent_kind: Team13OperatorUncertaintyTangentKind::Nonlinear,
            prior_kind: Team13MapParityPriorKind::WeakRidge,
            prior_kappa: 1.0,
            prior_tau: 1.0e-6,
            prior_diagonal_shift: 1.0e-12,
            pde_residual_kind: Team13MapParityPdeResidualKind::UngaugedCurl,
            pde_residual_weighting: Team13PdeResidualWeighting::Euclidean,
            pde_variance: 1.0e4,
            include_steel_observations: false,
            steel_observation_quadrature: Team13SteelObservationQuadratureMode::FaceCochain,
            observation_std_tesla: 1.0e-3,
            truth_max_iterations: 24,
            linear_solve: GaussNewtonLinearSolve::DirectCholesky,
            field_variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::Hutchinson,
                num_variance_probes: 64,
                variance_batch_count: 4,
                rng_seed: 1319,
                local_rb_block_size: 16,
            },
            estimate_field_variance: true,
            output_dir: Some(PathBuf::from(
                "target/team13_operator_uncertainty_diagnostic",
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13OperatorSteelPatchVarianceReport {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub prediction: f64,
    pub signed_prediction: f64,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub posterior_std: f64,
    pub observed_g_052: f64,
    pub observed_g_047: f64,
    pub residual_g_052: f64,
    pub residual_g_047: f64,
    pub abs_residual_over_std_g_052: f64,
    pub abs_residual_over_std_g_047: f64,
    pub row_nnz: usize,
}

#[derive(Debug, Clone)]
pub struct Team13OperatorRegionVarianceSummary {
    pub region: String,
    pub count: usize,
    pub mean_variance: f64,
    pub median_variance: f64,
    pub p90_variance: f64,
    pub max_variance: f64,
    pub mean_std: f64,
    pub mean_b_magnitude: f64,
    pub mean_gradient_indicator: f64,
    pub variance_ratio_to_iron_bulk: f64,
    pub variance_ratio_to_air_bulk: f64,
}

#[derive(Debug, Clone)]
pub struct Team13OperatorVarianceIndicatorCorrelation {
    pub indicator: String,
    pub count: usize,
    pub pearson_with_variance: f64,
    pub pearson_with_std: f64,
    pub indicator_mean: f64,
    pub mean_variance_indicator_positive: f64,
    pub mean_variance_indicator_zero: f64,
}

#[derive(Debug, Clone)]
pub struct Team13OperatorUncertaintyResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_kind: Team13NonlinearMaterialKind,
    pub material_log_iron_nu_scale: f64,
    pub tangent_kind: Team13OperatorUncertaintyTangentKind,
    pub prior_kind: Team13MapParityPriorKind,
    pub pde_residual_kind: Team13MapParityPdeResidualKind,
    pub pde_residual_weighting: Team13PdeResidualWeighting,
    pub pde_variance: f64,
    pub include_steel_observations: bool,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub observation_std_tesla: f64,
    pub deterministic_converged: bool,
    pub deterministic_residual_norm: f64,
    pub posterior_converged: bool,
    pub posterior_precision_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub fill_ratio_vs_lower: f64,
    pub field_variance_estimator: String,
    pub field_variance_available: bool,
    pub steel_patch_reports: Vec<Team13OperatorSteelPatchVarianceReport>,
    pub region_summaries: Vec<Team13OperatorRegionVarianceSummary>,
    pub indicator_correlations: Vec<Team13OperatorVarianceIndicatorCorrelation>,
    pub audit: Team13RegionAudit,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialGapMeshCase {
    pub label: String,
    pub steel_gap_m: f64,
    pub mesh_path: PathBuf,
    pub observed_gap: Option<Team13PublishedSteelGap>,
    pub weight: f64,
}

impl Team13MaterialGapMeshCase {
    pub fn from_published_gap(gap: Team13PublishedSteelGap, mesh_path: PathBuf) -> Self {
        Self {
            label: gap.token().to_string(),
            steel_gap_m: gap.steel_gap_m(),
            mesh_path,
            observed_gap: Some(gap),
            weight: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13MaterialLogScaleNode {
    pub label: String,
    pub log_iron_nu_scale: f64,
    pub weight: f64,
}

pub fn team13_material_log_scale_sigma_points(
    std: f64,
) -> Result<Vec<Team13MaterialLogScaleNode>, String> {
    validate_positive_finite(std, "TEAM 13 material log-scale standard deviation")?;
    let radius = 3.0_f64.sqrt() * std;
    Ok(vec![
        Team13MaterialLogScaleNode {
            label: "theta_minus".to_string(),
            log_iron_nu_scale: -radius,
            weight: 1.0 / 6.0,
        },
        Team13MaterialLogScaleNode {
            label: "theta_zero".to_string(),
            log_iron_nu_scale: 0.0,
            weight: 2.0 / 3.0,
        },
        Team13MaterialLogScaleNode {
            label: "theta_plus".to_string(),
            log_iron_nu_scale: radius,
            weight: 1.0 / 6.0,
        },
    ])
}

#[derive(Debug, Clone)]
pub struct Team13MaterialGapUqConfig {
    pub operator: Team13OperatorUncertaintyConfig,
    pub gap_cases: Vec<Team13MaterialGapMeshCase>,
    pub material_nodes: Vec<Team13MaterialLogScaleNode>,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13MaterialGapUqConfig {
    fn default() -> Self {
        let mut operator = Team13OperatorUncertaintyConfig::default();
        operator.estimate_field_variance = false;
        Self {
            operator,
            gap_cases: vec![
                Team13MaterialGapMeshCase::from_published_gap(
                    Team13PublishedSteelGap::G047,
                    PathBuf::from("target/team13_material_gap_uq/meshes/team13_half_g047.msh"),
                ),
                Team13MaterialGapMeshCase::from_published_gap(
                    Team13PublishedSteelGap::G052,
                    PathBuf::from("target/team13_material_gap_uq/meshes/team13_half_g052.msh"),
                ),
            ],
            material_nodes: team13_material_log_scale_sigma_points(0.10)
                .expect("default TEAM 13 material log-scale sigma points are valid"),
            output_dir: Some(PathBuf::from("target/team13_material_gap_uq")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13MaterialGapUqCaseResult {
    pub gap_label: String,
    pub steel_gap_m: f64,
    pub observed_gap: Option<Team13PublishedSteelGap>,
    pub material_label: String,
    pub material_log_iron_nu_scale: f64,
    pub normalized_weight: f64,
    pub rmse_vs_observed_gap: f64,
    pub mean_patch_std: f64,
    pub operator_result: Team13OperatorUncertaintyResult,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialGapVarianceDecomposition {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub mean_prediction: f64,
    pub expected_operator_variance: f64,
    pub between_gap_variance: f64,
    pub between_material_variance: f64,
    pub gap_material_interaction_variance: f64,
    pub total_between_case_variance: f64,
    pub total_variance: f64,
    pub operator_fraction: f64,
    pub gap_fraction: f64,
    pub material_fraction: f64,
    pub interaction_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialGapUqResult {
    pub case_results: Vec<Team13MaterialGapUqCaseResult>,
    pub variance_decomposition: Vec<Team13MaterialGapVarianceDecomposition>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13MaterialPriorCalibrationMode {
    Fixed,
    SteelPriorPredictiveRms,
}

impl Team13MaterialPriorCalibrationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::SteelPriorPredictiveRms => "steel-prior-predictive-rms",
        }
    }
}

impl std::str::FromStr for Team13MaterialPriorCalibrationMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "steel-prior-predictive-rms" | "steel-rms" | "prior-predictive-rms" => {
                Ok(Self::SteelPriorPredictiveRms)
            }
            other => Err(format!(
                "unknown TEAM 13 material prior calibration `{other}`; expected fixed or steel-prior-predictive-rms"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team13MaterialPriorCalibrationTarget {
    ObservationStd,
    PublishedGapDifference,
    Explicit,
}

impl Team13MaterialPriorCalibrationTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ObservationStd => "observation-std",
            Self::PublishedGapDifference => "published-gap-difference",
            Self::Explicit => "explicit",
        }
    }
}

impl std::str::FromStr for Team13MaterialPriorCalibrationTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "observation-std" | "observation" | "obs-std" => Ok(Self::ObservationStd),
            "published-gap-difference" | "gap-difference" | "gap-rms" => {
                Ok(Self::PublishedGapDifference)
            }
            "explicit" => Ok(Self::Explicit),
            other => Err(format!(
                "unknown TEAM 13 material prior calibration target `{other}`; expected observation-std, published-gap-difference, or explicit"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13JointMaterialUqConfig {
    pub operator: Team13OperatorUncertaintyConfig,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub material_anchor_b_tesla: [f64; 3],
    pub material_prior_std: f64,
    pub material_prior_calibration: Team13MaterialPriorCalibrationMode,
    pub material_prior_calibration_target: Team13MaterialPriorCalibrationTarget,
    pub material_prior_target_steel_rms_tesla: Option<f64>,
    pub material_prior_std_floor: f64,
    pub material_prior_std_ceiling: f64,
    pub magnitude_smoothing_tesla: f64,
    pub max_iterations: usize,
    pub step_regularization: GaussNewtonStepRegularization,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13JointMaterialUqConfig {
    fn default() -> Self {
        let mut operator = Team13OperatorUncertaintyConfig::default();
        operator.include_steel_observations = true;
        operator.output_dir = None;
        operator.estimate_field_variance = false;
        Self {
            operator,
            observed_steel_gap: Team13PublishedSteelGap::G052,
            material_anchor_b_tesla: [0.5, 1.7, 2.3],
            material_prior_std: 0.10,
            material_prior_calibration: Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms,
            material_prior_calibration_target:
                Team13MaterialPriorCalibrationTarget::PublishedGapDifference,
            material_prior_target_steel_rms_tesla: None,
            material_prior_std_floor: 1.0e-6,
            material_prior_std_ceiling: 0.5,
            magnitude_smoothing_tesla: 1.0e-8,
            max_iterations: 24,
            step_regularization: GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt,
            output_dir: Some(PathBuf::from("target/team13_joint_material_uq")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13MaterialPriorCalibrationReport {
    pub mode: Team13MaterialPriorCalibrationMode,
    pub target: Team13MaterialPriorCalibrationTarget,
    pub target_steel_rms_tesla: Option<f64>,
    pub configured_material_prior_std: f64,
    pub material_prior_std: f64,
    pub unclamped_material_prior_std: Option<f64>,
    pub material_prior_std_floor: f64,
    pub material_prior_std_ceiling: f64,
    pub unit_theta_steel_rms_tesla: Option<f64>,
    pub sensitivity_frobenius_norm_tesla: Option<f64>,
    pub max_abs_sensitivity_tesla: Option<f64>,
    pub theta_column_norms_tesla: [f64; 3],
    pub steel_row_count: usize,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialOnlyUqConfig {
    pub operator: Team13OperatorUncertaintyConfig,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub material_anchor_b_tesla: [f64; 3],
    pub material_prior_std: f64,
    pub material_prior_calibration: Team13MaterialPriorCalibrationMode,
    pub material_prior_calibration_target: Team13MaterialPriorCalibrationTarget,
    pub material_prior_target_steel_rms_tesla: Option<f64>,
    pub material_prior_std_floor: f64,
    pub material_prior_std_ceiling: f64,
    pub magnitude_smoothing_tesla: f64,
    pub max_iterations: usize,
    pub max_line_search_steps: usize,
    pub max_theta_step_norm: f64,
    pub step_regularization: GaussNewtonStepRegularization,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13MaterialOnlyUqConfig {
    fn default() -> Self {
        let mut operator = Team13OperatorUncertaintyConfig::default();
        operator.output_dir = None;
        operator.estimate_field_variance = false;
        Self {
            operator,
            observed_steel_gap: Team13PublishedSteelGap::G052,
            material_anchor_b_tesla: [0.5, 1.7, 2.3],
            material_prior_std: 0.10,
            material_prior_calibration: Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms,
            material_prior_calibration_target:
                Team13MaterialPriorCalibrationTarget::PublishedGapDifference,
            material_prior_target_steel_rms_tesla: None,
            material_prior_std_floor: 1.0e-6,
            material_prior_std_ceiling: 0.5,
            magnitude_smoothing_tesla: 1.0e-8,
            max_iterations: 12,
            max_line_search_steps: 8,
            max_theta_step_norm: 1.0e-4,
            step_regularization: GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt,
            output_dir: Some(PathBuf::from("target/team13_material_only_uq")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableMaterialUqConfig {
    pub operator: Team13OperatorUncertaintyConfig,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub material_anchor_b_tesla: [f64; 3],
    pub eta_prior_std: f64,
    pub eta_prior_calibration: Team13MaterialPriorCalibrationMode,
    pub eta_prior_calibration_target: Team13MaterialPriorCalibrationTarget,
    pub eta_prior_target_steel_rms_tesla: Option<f64>,
    pub eta_prior_std_floor: f64,
    pub eta_prior_std_ceiling: f64,
    pub svd_relative_tolerance: f64,
    pub svd_absolute_tolerance: f64,
    pub perturbation_rms_fraction_of_gap: f64,
    pub continuation_steps: usize,
    pub magnitude_smoothing_tesla: f64,
    pub max_iterations: usize,
    pub max_line_search_steps: usize,
    pub max_eta_step_norm: f64,
    pub step_regularization: GaussNewtonStepRegularization,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13IdentifiableMaterialUqConfig {
    fn default() -> Self {
        let mut operator = Team13OperatorUncertaintyConfig::default();
        operator.output_dir = None;
        operator.estimate_field_variance = false;
        Self {
            operator,
            observed_steel_gap: Team13PublishedSteelGap::G052,
            material_anchor_b_tesla: [0.5, 1.7, 2.3],
            eta_prior_std: 0.10,
            eta_prior_calibration: Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms,
            eta_prior_calibration_target:
                Team13MaterialPriorCalibrationTarget::PublishedGapDifference,
            eta_prior_target_steel_rms_tesla: None,
            eta_prior_std_floor: 1.0e-6,
            eta_prior_std_ceiling: 0.5,
            svd_relative_tolerance: 1.0e-3,
            svd_absolute_tolerance: 1.0e-12,
            perturbation_rms_fraction_of_gap: 0.5,
            continuation_steps: 4,
            magnitude_smoothing_tesla: 1.0e-8,
            max_iterations: 24,
            max_line_search_steps: 8,
            max_eta_step_norm: 1.0e-2,
            step_regularization: GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt,
            output_dir: Some(PathBuf::from("target/team13_identifiable_material_uq")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableJointMaterialUqConfig {
    pub operator: Team13OperatorUncertaintyConfig,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub material_anchor_b_tesla: [f64; 3],
    pub eta_prior_std: f64,
    pub eta_prior_calibration: Team13MaterialPriorCalibrationMode,
    pub eta_prior_calibration_target: Team13MaterialPriorCalibrationTarget,
    pub eta_prior_target_steel_rms_tesla: Option<f64>,
    pub eta_prior_std_floor: f64,
    pub eta_prior_std_ceiling: f64,
    pub svd_relative_tolerance: f64,
    pub svd_absolute_tolerance: f64,
    pub perturbation_rms_fraction_of_gap: f64,
    pub continuation_steps: usize,
    pub magnitude_smoothing_tesla: f64,
    pub max_iterations: usize,
    pub step_regularization: GaussNewtonStepRegularization,
    pub output_dir: Option<PathBuf>,
}

impl Default for Team13IdentifiableJointMaterialUqConfig {
    fn default() -> Self {
        let mut operator = Team13OperatorUncertaintyConfig::default();
        operator.include_steel_observations = true;
        operator.output_dir = None;
        operator.estimate_field_variance = false;
        Self {
            operator,
            observed_steel_gap: Team13PublishedSteelGap::G052,
            material_anchor_b_tesla: [0.5, 1.7, 2.3],
            eta_prior_std: 0.10,
            eta_prior_calibration: Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms,
            eta_prior_calibration_target:
                Team13MaterialPriorCalibrationTarget::PublishedGapDifference,
            eta_prior_target_steel_rms_tesla: None,
            eta_prior_std_floor: 1.0e-6,
            eta_prior_std_ceiling: 0.5,
            svd_relative_tolerance: 1.0e-3,
            svd_absolute_tolerance: 1.0e-12,
            perturbation_rms_fraction_of_gap: 0.5,
            continuation_steps: 4,
            magnitude_smoothing_tesla: 1.0e-8,
            max_iterations: 24,
            step_regularization: GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt,
            output_dir: Some(PathBuf::from(
                "target/team13_identifiable_joint_material_uq",
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13MaterialOnlyIteration {
    pub iteration: usize,
    pub objective: f64,
    pub trial_objective: f64,
    pub steel_weighted_residual_norm: f64,
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub alpha: f64,
    pub regularization_lambda: f64,
    pub forward_residual_norm: f64,
    pub trial_forward_residual_norm: f64,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialOnlyObjectiveComponents {
    pub prior_material: f64,
    pub steel_observation: f64,
    pub total: f64,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialOnlySteelPredictionReport {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub observed: f64,
    pub nominal_prediction: f64,
    pub map_prediction: f64,
    pub nominal_residual: f64,
    pub map_residual: f64,
    pub nominal_signed_prediction: f64,
    pub map_signed_prediction: f64,
    pub row_nnz: usize,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialOnlySensitivityReport {
    pub singular_values: [f64; 3],
    pub rank: usize,
    pub condition_number: f64,
    pub frobenius_norm_tesla: f64,
    pub max_abs_sensitivity_tesla: f64,
    pub theta_column_norms_tesla: [f64; 3],
    pub steel_row_count: usize,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialOnlyComparisonRow {
    pub label: String,
    pub theta: [f64; 3],
    pub steel_rmse_tesla: f64,
    pub steel_max_abs_residual_tesla: f64,
    pub objective: f64,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialOnlyUqResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_anchor_b_tesla: [f64; 3],
    pub material_prior_std: f64,
    pub material_prior_calibration: Team13MaterialPriorCalibrationReport,
    pub magnitude_smoothing_tesla: f64,
    pub nominal_forward_converged: bool,
    pub nominal_forward_residual_norm: f64,
    pub map_forward_converged: bool,
    pub map_forward_residual_norm: f64,
    pub posterior_converged: bool,
    pub theta_map: [f64; 3],
    pub material_posterior: Vec<Team13MaterialParameterPosteriorReport>,
    pub material_posterior_covariance: [[f64; 3]; 3],
    pub material_posterior_correlation: [[f64; 3]; 3],
    pub bh_curve_bands: Vec<Team13BhCurveBandReport>,
    pub steel_predictions: Vec<Team13MaterialOnlySteelPredictionReport>,
    pub objective_components: Team13MaterialOnlyObjectiveComponents,
    pub sensitivity: Team13MaterialOnlySensitivityReport,
    pub history: Vec<Team13MaterialOnlyIteration>,
    pub comparison: Vec<Team13MaterialOnlyComparisonRow>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableEtaPriorCalibrationReport {
    pub mode: Team13MaterialPriorCalibrationMode,
    pub target: Team13MaterialPriorCalibrationTarget,
    pub target_steel_rms_tesla: Option<f64>,
    pub configured_eta_prior_std: f64,
    pub eta_prior_std: f64,
    pub unclamped_eta_prior_std: Option<f64>,
    pub eta_prior_std_floor: f64,
    pub eta_prior_std_ceiling: f64,
    pub unit_eta_steel_rms_tesla: Option<f64>,
    pub retained_rank: usize,
    pub retained_mode_norms_tesla: Vec<f64>,
    pub steel_row_count: usize,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableMaterialModeReport {
    pub mode_index: usize,
    pub retained: bool,
    pub singular_value: f64,
    pub relative_singular_value: f64,
    pub theta_coefficients: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableBaselinePerturbationReport {
    pub target_rms_tesla: f64,
    pub achieved_linearized_rms_tesla: f64,
    pub gap_difference_rms_tesla: f64,
    pub eta_bias: Vec<f64>,
    pub theta_bias: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableEtaPosteriorReport {
    pub name: String,
    pub prior_mean: f64,
    pub posterior_mean: f64,
    pub prior_std: f64,
    pub posterior_std: f64,
    pub eta_bias: f64,
    pub recovery_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableThetaPosteriorReport {
    pub name: String,
    pub anchor_b_tesla: f64,
    pub benchmark_mean: f64,
    pub biased_baseline_mean: f64,
    pub posterior_mean: f64,
    pub posterior_std: f64,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableSteelPredictionReport {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub observed: f64,
    pub benchmark_prediction: f64,
    pub biased_prediction: f64,
    pub corrected_prediction: f64,
    pub benchmark_residual: f64,
    pub biased_residual: f64,
    pub corrected_residual: f64,
    pub row_nnz: usize,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableMaterialComparisonRow {
    pub label: String,
    pub theta: [f64; 3],
    pub eta: Vec<f64>,
    pub steel_rmse_tesla: f64,
    pub steel_max_abs_residual_tesla: f64,
    pub objective: f64,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableBhCurveBandReport {
    pub b_tesla: f64,
    pub benchmark_h_ampere_per_meter: f64,
    pub biased_baseline_h_ampere_per_meter: f64,
    pub corrected_mean_h_ampere_per_meter: f64,
    pub corrected_std_h_ampere_per_meter: f64,
    pub corrected_lower_2sigma_h_ampere_per_meter: f64,
    pub corrected_upper_2sigma_h_ampere_per_meter: f64,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableMaterialUqResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_anchor_b_tesla: [f64; 3],
    pub eta_prior_std: f64,
    pub eta_prior_calibration: Team13IdentifiableEtaPriorCalibrationReport,
    pub magnitude_smoothing_tesla: f64,
    pub svd_relative_tolerance: f64,
    pub svd_absolute_tolerance: f64,
    pub retained_rank: usize,
    pub singular_values: [f64; 3],
    pub material_basis: Vec<Team13IdentifiableMaterialModeReport>,
    pub baseline_perturbation: Team13IdentifiableBaselinePerturbationReport,
    pub benchmark_forward_converged: bool,
    pub benchmark_forward_residual_norm: f64,
    pub biased_forward_converged: bool,
    pub biased_forward_residual_norm: f64,
    pub map_forward_converged: bool,
    pub map_forward_residual_norm: f64,
    pub posterior_converged: bool,
    pub eta_map: Vec<f64>,
    pub theta_map: [f64; 3],
    pub eta_posterior: Vec<Team13IdentifiableEtaPosteriorReport>,
    pub theta_posterior: Vec<Team13IdentifiableThetaPosteriorReport>,
    pub eta_posterior_covariance: Vec<Vec<f64>>,
    pub theta_posterior_covariance: [[f64; 3]; 3],
    pub bh_curve_bands: Vec<Team13IdentifiableBhCurveBandReport>,
    pub steel_predictions: Vec<Team13IdentifiableSteelPredictionReport>,
    pub objective_components: Team13MaterialOnlyObjectiveComponents,
    pub history: Vec<Team13MaterialOnlyIteration>,
    pub comparison: Vec<Team13IdentifiableMaterialComparisonRow>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableJointSteelPatchReport {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub prediction: f64,
    pub signed_prediction: f64,
    pub observed: f64,
    pub residual: f64,
    pub total_variance: f64,
    pub state_conditional_variance: f64,
    pub eta_explained_variance: f64,
    pub posterior_std: f64,
    pub row_nnz: usize,
}

#[derive(Debug, Clone)]
pub struct Team13IdentifiableJointMaterialUqResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_anchor_b_tesla: [f64; 3],
    pub eta_prior_std: f64,
    pub eta_prior_calibration: Team13IdentifiableEtaPriorCalibrationReport,
    pub magnitude_smoothing_tesla: f64,
    pub svd_relative_tolerance: f64,
    pub svd_absolute_tolerance: f64,
    pub retained_rank: usize,
    pub singular_values: [f64; 3],
    pub material_basis: Vec<Team13IdentifiableMaterialModeReport>,
    pub baseline_perturbation: Team13IdentifiableBaselinePerturbationReport,
    pub benchmark_forward_converged: bool,
    pub benchmark_forward_residual_norm: f64,
    pub biased_forward_converged: bool,
    pub biased_forward_residual_norm: f64,
    pub posterior_converged: bool,
    pub posterior_residual_norm: f64,
    pub posterior_precision_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub eta_map: Vec<f64>,
    pub theta_map: [f64; 3],
    pub eta_posterior: Vec<Team13IdentifiableEtaPosteriorReport>,
    pub theta_posterior: Vec<Team13IdentifiableThetaPosteriorReport>,
    pub eta_posterior_covariance: Vec<Vec<f64>>,
    pub theta_posterior_covariance: [[f64; 3]; 3],
    pub bh_curve_bands: Vec<Team13IdentifiableBhCurveBandReport>,
    pub steel_patch_reports: Vec<Team13IdentifiableJointSteelPatchReport>,
    pub fixed_biased_material_solve: Team13JointMaterialSolveDiagnostics,
    pub joint_identifiable_material_solve: Team13JointMaterialSolveDiagnostics,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13MaterialParameterPosteriorReport {
    pub name: String,
    pub anchor_b_tesla: f64,
    pub prior_mean: f64,
    pub posterior_mean: f64,
    pub prior_std: f64,
    pub posterior_std: f64,
}

#[derive(Debug, Clone)]
pub struct Team13BhCurveBandReport {
    pub b_tesla: f64,
    pub nominal_h_ampere_per_meter: f64,
    pub posterior_mean_h_ampere_per_meter: f64,
    pub posterior_std_h_ampere_per_meter: f64,
    pub lower_2sigma_h_ampere_per_meter: f64,
    pub upper_2sigma_h_ampere_per_meter: f64,
}

#[derive(Debug, Clone)]
pub struct Team13JointSteelPatchVarianceReport {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub prediction: f64,
    pub signed_prediction: f64,
    pub observed: f64,
    pub residual: f64,
    pub total_variance: f64,
    pub state_conditional_variance: f64,
    pub material_explained_variance: f64,
    pub posterior_std: f64,
    pub row_nnz: usize,
}

#[derive(Debug, Clone)]
pub struct Team13JointMaterialObjectiveComponents {
    pub label: String,
    pub prior_state: f64,
    pub prior_material: f64,
    pub steel_observation: f64,
    pub pde_residual: f64,
    pub total: f64,
    pub solver_objective: Option<f64>,
    pub solver_objective_gap: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Team13JointMaterialFinalStepDiagnostic {
    pub available: bool,
    pub error: Option<String>,
    pub objective: f64,
    pub weighted_residual_norm: f64,
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub directional_derivative: f64,
    pub accepted_alpha: Option<f64>,
    pub accepted_objective: Option<f64>,
    pub regularization_lambda: f64,
    pub linear_solve_absolute_residual_norm: f64,
    pub linear_solve_relative_residual_norm: f64,
}

impl Team13JointMaterialFinalStepDiagnostic {
    fn unavailable(error: String) -> Self {
        Self {
            available: false,
            error: Some(error),
            objective: f64::NAN,
            weighted_residual_norm: f64::NAN,
            gradient_norm: f64::NAN,
            step_norm: f64::NAN,
            directional_derivative: f64::NAN,
            accepted_alpha: None,
            accepted_objective: None,
            regularization_lambda: f64::NAN,
            linear_solve_absolute_residual_norm: f64::NAN,
            linear_solve_relative_residual_norm: f64::NAN,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Team13JointMaterialSolveDiagnostics {
    pub label: String,
    pub state_dimension: usize,
    pub theta_dimension: usize,
    pub prior_kind: Team13MapParityPriorKind,
    pub pde_residual_kind: Team13MapParityPdeResidualKind,
    pub linear_measurement_count: usize,
    pub linear_measurement_rows: usize,
    pub converged: bool,
    pub posterior_residual_norm: f64,
    pub posterior_precision_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub history: Vec<GaussNewtonIteration>,
    pub diagnostics: GaussNewtonRunDiagnostics,
    pub assembly: NonlinearAssemblyStats,
    pub final_factorization: LaplaceFactorizationStats,
    pub final_residuals: Vec<NonlinearResidualReport>,
    pub final_step: Team13JointMaterialFinalStepDiagnostic,
    pub objective_components: Team13JointMaterialObjectiveComponents,
}

#[derive(Debug, Clone)]
pub struct Team13JointMaterialUqResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub observed_steel_gap: Team13PublishedSteelGap,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_anchor_b_tesla: [f64; 3],
    pub material_prior_std: f64,
    pub material_prior_calibration: Team13MaterialPriorCalibrationReport,
    pub magnitude_smoothing_tesla: f64,
    pub deterministic_converged: bool,
    pub deterministic_residual_norm: f64,
    pub posterior_converged: bool,
    pub posterior_residual_norm: f64,
    pub posterior_precision_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub material_posterior: Vec<Team13MaterialParameterPosteriorReport>,
    pub material_posterior_covariance: [[f64; 3]; 3],
    pub material_posterior_correlation: [[f64; 3]; 3],
    pub bh_curve_bands: Vec<Team13BhCurveBandReport>,
    pub steel_patch_reports: Vec<Team13JointSteelPatchVarianceReport>,
    pub fixed_material_solve: Team13JointMaterialSolveDiagnostics,
    pub joint_material_solve: Team13JointMaterialSolveDiagnostics,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13RegionAuditEntry {
    pub name: String,
    pub cell_count: usize,
    pub volume: f64,
    pub volume_fraction: f64,
    pub current_integral: [f64; 3],
    pub current_l2_norm: f64,
}

#[derive(Debug, Clone)]
pub struct Team13RegionAudit {
    pub entries: Vec<Team13RegionAuditEntry>,
    pub total_volume: f64,
    pub iron_and_coil_cells: usize,
    pub multiple_coil_cells: usize,
    pub unclassified_cells: usize,
}

#[derive(Debug, Clone)]
pub struct Team13SameMeshLinearParityResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub operator_dimension: usize,
    pub operator_nnz: usize,
    pub full_source_l2: f64,
    pub rhs_l2: f64,
    pub solution_l2: f64,
    pub linear_residual_l2: f64,
    pub energy: f64,
    pub work: f64,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub steel_predictions: Vec<Team13PublishedSteelBenchmarkReport>,
    pub steel_group_summaries_g052: Vec<Team13PublishedSteelGroupSummary>,
    pub steel_rmse_g052: f64,
    pub steel_rmse_g047: f64,
    pub audit: Team13RegionAudit,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13SteelNgsolveComparisonReport {
    pub name: String,
    pub group: Team13SteelSurfaceGroup,
    pub ngsolve_prediction: f64,
    pub feec_prediction: f64,
    pub observed_g_052: f64,
    pub observed_g_047: f64,
    pub feec_minus_ngsolve: f64,
    pub feec_residual_g_052: f64,
    pub feec_residual_g_047: f64,
}

#[derive(Debug, Clone)]
pub struct Team13DeterministicBenchmarkResult {
    pub linear: Team13SameMeshLinearParityResult,
    pub linear_comparison: Vec<Team13SteelNgsolveComparisonReport>,
    pub nonlinear: Option<Team13NonlinearForwardParityResult>,
    pub nonlinear_comparison: Option<Vec<Team13SteelNgsolveComparisonReport>>,
}

#[derive(Debug, Clone)]
pub struct Team13NonlinearForwardParityResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_kind: Team13NonlinearMaterialKind,
    pub ampere_turns: f64,
    pub residual_dimension: usize,
    pub initial_jacobian_nnz: usize,
    pub final_jacobian_nnz: usize,
    pub rhs_l2: f64,
    pub initial_solution_l2: f64,
    pub nonlinear_solution_l2: f64,
    pub initial_residual_l2: f64,
    pub final_residual_l2: f64,
    pub converged: bool,
    pub iterations: usize,
    pub final_step_norm: f64,
    pub final_step_alpha: f64,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub steel_predictions: Vec<Team13PublishedSteelBenchmarkReport>,
    pub initial_steel_group_summaries: Vec<Team13PublishedSteelGroupSummary>,
    pub nonlinear_steel_group_summaries: Vec<Team13PublishedSteelGroupSummary>,
    pub initial_steel_rmse_g052: f64,
    pub initial_steel_rmse_g047: f64,
    pub nonlinear_steel_rmse_g052: f64,
    pub nonlinear_steel_rmse_g047: f64,
    pub audit: Team13RegionAudit,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13MapParityRunResult {
    pub label: String,
    pub pde_variance: f64,
    pub observation_std_tesla: f64,
    pub total_residual_rows: usize,
    pub steel_observation_count: usize,
    pub initial_relative_error: f64,
    pub posterior_relative_error: f64,
    pub initial_residual_norm: f64,
    pub truth_residual_norm: f64,
    pub posterior_residual_norm: f64,
    pub initial_steel_rmse: f64,
    pub posterior_steel_rmse: f64,
    pub posterior_steel_relative_rmse: f64,
    pub posterior_steel_max_abs_residual: f64,
    pub steel_rmse_improvement_ratio: f64,
    pub posterior_converged: bool,
    pub all_finite_variances: bool,
    pub nonnegative_variances: bool,
    pub b_quantity_variances_finite: bool,
    pub b_quantity_variances_nonnegative: bool,
    pub latent_variance_count: usize,
    pub latent_variances_finite: bool,
    pub latent_variances_nonnegative: bool,
    pub assembly: NonlinearAssemblyStats,
    pub final_factorization: LaplaceFactorizationStats,
    pub diagnostics: GaussNewtonRunDiagnostics,
    pub posterior_history: Vec<GaussNewtonIteration>,
    pub internal_steel_reports: Vec<Team13SyntheticBenchmarkObservationReport>,
    pub internal_steel_variances: Vec<Team13NonlinearSensorVarianceReport>,
    pub published_steel_benchmark_reports: Vec<Team13PublishedSteelBenchmarkReport>,
}

#[derive(Debug, Clone)]
pub struct Team13MapParityResult {
    pub domain_mode: Team13DomainMode,
    pub mesh_path: PathBuf,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub prior_kind: Team13MapParityPriorKind,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_diagonal_shift: f64,
    pub pde_residual_kind: Team13MapParityPdeResidualKind,
    pub step_regularization: GaussNewtonStepRegularization,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub magnitude_smoothing_tesla: f64,
    pub initial_residual_norm: f64,
    pub truth_residual_norm: f64,
    pub truth_converged: bool,
    pub truth_cache_hit: bool,
    pub truth_history: Vec<SquareNewtonIteration>,
    pub default_run: Team13MapParityRunResult,
    pub sweep_runs: Vec<Team13MapParityRunResult>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticBenchmarkGeometryRunResult {
    pub pde_variance: f64,
    pub observation_std_tesla: f64,
    pub total_residual_rows: usize,
    pub observation_count: usize,
    pub assimilated_observation_count: usize,
    pub steel_observation_count: usize,
    pub air_observation_count: usize,
    pub initial_relative_error: f64,
    pub posterior_relative_error: f64,
    pub initial_residual_norm: f64,
    pub truth_residual_norm: f64,
    pub posterior_residual_norm: f64,
    pub initial_sensor_rmse: f64,
    pub posterior_sensor_rmse: f64,
    pub posterior_sensor_relative_rmse: f64,
    pub posterior_sensor_max_abs_residual: f64,
    pub sensor_rmse_improvement_ratio: f64,
    pub posterior_converged: bool,
    pub all_finite_variances: bool,
    pub nonnegative_variances: bool,
    pub prior_variance_diagnostics: Team13PriorVarianceDiagnostics,
    pub assembly: NonlinearAssemblyStats,
    pub final_factorization: LaplaceFactorizationStats,
    pub posterior_history: Vec<GaussNewtonIteration>,
    pub group_summaries: Vec<Team13SyntheticBenchmarkObservationGroupSummary>,
    pub observation_reports: Vec<Team13SyntheticBenchmarkObservationReport>,
    pub published_steel_benchmark_reports: Vec<Team13PublishedSteelBenchmarkReport>,
    pub observation_variances: Vec<Team13NonlinearSensorVarianceReport>,
}

#[derive(Debug, Clone)]
pub struct Team13SyntheticBenchmarkGeometryResult {
    pub domain_mode: Team13DomainMode,
    pub vertices: usize,
    pub edges: usize,
    pub cells: usize,
    pub active_dofs: usize,
    pub boundary_edge_dofs: usize,
    pub material_kind: Team13NonlinearMaterialKind,
    pub beta_iron: f64,
    pub b_scale_tesla: f64,
    pub prior_kappa: f64,
    pub prior_tau: f64,
    pub prior_diagonal_shift: f64,
    pub steel_observation_quadrature: Team13SteelObservationQuadratureMode,
    pub magnitude_smoothing_tesla: f64,
    pub observation_count: usize,
    pub assimilated_observation_count: usize,
    pub steel_observation_count: usize,
    pub air_observation_count: usize,
    pub initial_residual_norm: f64,
    pub truth_residual_norm: f64,
    pub truth_converged: bool,
    pub truth_history: Vec<SquareNewtonIteration>,
    pub default_run: Team13SyntheticBenchmarkGeometryRunResult,
    pub sweep_runs: Vec<Team13SyntheticBenchmarkGeometryRunResult>,
    pub source_scale_diagnostics: Vec<Team13SourceScaleDiagnosticRun>,
}

struct Team13NonlinearPriorBuild {
    spec: GaussianPriorSpec,
    kind: Team13FieldPriorKind,
    precision_scale: f64,
    kappa: f64,
    tau: f64,
    kappa_fallback_used: bool,
}

struct Team13Operators {
    a_components: Vec<SparseRowOperator>,
    b_components: Vec<SparseRowOperator>,
    b_cochain: SparseRowOperator,
}

#[derive(Debug, Clone)]
struct Team13PeriodDiagnosticSpec {
    name: String,
    operator: SparseRowOperator,
}

#[derive(Debug, Clone, Copy)]
pub struct Team13FieldRecoveryMetrics {
    pub a_active_rmse: f64,
    pub a_active_relative_l2_error: f64,
    pub b_cochain_rmse: f64,
    pub b_cochain_relative_l2_error: f64,
    pub b_vector_rmse: f64,
    pub b_vector_relative_l2_error: f64,
}

#[derive(Debug, Clone)]
struct Team13SurfaceMeasurementDefinition {
    name: String,
    ngsolve_name: String,
    observation: f64,
    component_index: usize,
    normal_axis: usize,
    target: f64,
    x_range: (f64, f64),
    y_range: (f64, f64),
    z_range: (f64, f64),
    quadrature_counts: [usize; 3],
}

#[derive(Debug, Clone)]
struct Team13PointMeasurementDefinition {
    name: String,
    observation: Option<f64>,
    point: [f64; 3],
}

#[derive(Debug, Clone)]
struct Team13NgsolveSteelPrediction {
    name: String,
    group: Team13SteelSurfaceGroup,
    prediction: f64,
    observed_g_052: f64,
    observed_g_047: f64,
}

#[derive(Debug, Clone)]
struct Team13LinearizedMeasurement {
    spec: LinearGaussianMeasurementSpec,
    nominal_prediction: f64,
    linearization_direction: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
struct Team13SyntheticBenchmarkObservationSpec {
    name: String,
    group: Team13SyntheticBenchmarkObservationGroup,
    steel_surface_group: Option<Team13SteelSurfaceGroup>,
}

#[derive(Debug, Clone, PartialEq)]
struct Team13SteelFullObservationPart {
    row_count: usize,
    triplets: Vec<SparseTriplet>,
    groups: Vec<SmoothGroupedNormObservation>,
    specs: Vec<Team13SyntheticBenchmarkObservationSpec>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Team13FaceCochainSurfaceRow {
    row: Vec<(usize, f64)>,
    face_count: usize,
    selected_area: f64,
    expected_area: f64,
}

struct Team13SyntheticBenchmarkObservationBuild {
    model: SmoothGroupedNormLinearResidualModel,
    observations: Vec<f64>,
    initial_predictions: Vec<f64>,
    specs: Vec<Team13SyntheticBenchmarkObservationSpec>,
    assimilated_model: SmoothGroupedNormLinearResidualModel,
    assimilated_observations: Vec<f64>,
    assimilated_specs: Vec<Team13SyntheticBenchmarkObservationSpec>,
    #[cfg(test)]
    full_operator: SparseTripletMatrix,
    #[cfg(test)]
    full_bias: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Team13PublishedSteelSmoothObservationBuild {
    model: SmoothGroupedNormLinearResidualModel,
    observations: Vec<f64>,
    specs: Vec<Team13SyntheticBenchmarkObservationSpec>,
}

#[derive(Debug, Clone)]
struct Team13CellGeometry {
    coords: SimplexCoords,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    faces: Vec<Team13CellFaceGeometry>,
}

#[derive(Debug, Clone)]
struct Team13CellFaceGeometry {
    face_index: usize,
    local_face: Simplex,
}

pub fn solve_team13_linear_uq(
    config: &Team13LinearConfig,
) -> Result<Team13LinearSolveResult, String> {
    validate_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    let metric = coords.to_edge_lengths(&topology);
    let reluctivity = reluctivity_weight();
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &reluctivity,
        ));
    let system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;
    let source_operator = build_source_mode_operator(
        &topology,
        &metric,
        &coords,
        &galmats,
        &boundary,
        config.domain_mode,
        config.ampere_turns,
    )?;
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let nominal_a = solve_nominal_a(
        &topology,
        &metric,
        &coords,
        &reluctivity,
        &boundary,
        nominal_source,
    );
    let operators = build_team13_operators(&topology, &coords)?;
    let observation_overrides =
        load_surface_observation_overrides(config.observation_csv_path.as_deref())?;
    let measurements = build_linearized_b_measurements(
        &topology,
        &coords,
        &operators.b_cochain,
        &operators.b_components,
        &nominal_a,
        config.b_observation_std_tesla * config.b_observation_std_tesla,
        config.measurement_mode,
        config.legacy_measurement_band,
        observation_overrides.as_ref(),
    )?;
    let reduced_nominal_a = reduce_vector_with_layout(&system.layout, &nominal_a)?;
    let state_active_mask = active_dof_mask(&system.layout);
    let state_prior = build_weighted_whittle_prior(&system, &reduced_nominal_a)?;
    let source_prior = LinearUncertainInputSpec {
        name: "team13_coil_modes".to_string(),
        operator: source_operator,
        prior: GaussianPriorSpec {
            mean: vec![1.0; COIL_MODE_COUNT],
            precision: diagonal_precision(
                COIL_MODE_COUNT,
                1.0 / (config.coil_relative_std * config.coil_relative_std),
            ),
        },
        preference: RepresentationPreference::ForceLatent,
        collapsed_precision: None,
    };

    let problem = LinearPdeUqProblem {
        state_prior,
        system,
        uncertain_inputs: vec![source_prior],
        joint_measurements: Vec::new(),
        physical_measurements: measurements
            .iter()
            .map(|measurement| measurement.spec.clone())
            .collect(),
        derived_quantities: build_derived_quantities(&operators)?,
        joint_derived_quantities: Vec::new(),
        pde_variance: Some(config.pde_variance),
        pde_precision: None,
    };
    let posterior = solve_linear_pde_uq_with_config(&problem, &config.solver)?;
    let sensor_reports = evaluate_sensor_reports(&measurements, &posterior.posterior_mean)?;
    let benchmark_reports = evaluate_benchmark_reports(
        &topology,
        &coords,
        &operators,
        &nominal_a,
        &posterior.posterior_mean,
        config.measurement_mode,
        config.legacy_measurement_band,
        observation_overrides.as_ref(),
    )?;
    let a_variance_ratio = variance_ratio(&posterior.posterior_variance, &posterior.prior_variance);
    let b_variance_ratio = derived_variance_ratio(&posterior, B_COCHAIN_DERIVED_NAME)?;
    let vector_pushforwards =
        build_vector_pushforwards(&operators, &posterior.posterior_mean, &posterior)?;

    let result = Team13LinearSolveResult {
        domain_mode: config.domain_mode,
        posterior,
        nominal_a: nominal_a.clone(),
        field_reference_name: "nominal_alpha_1".to_string(),
        field_reference_a: nominal_a,
        state_active_mask,
        a_variance_ratio,
        b_variance_ratio,
        sensor_reports,
        benchmark_reports,
        vector_pushforwards,
    };

    if let Some(output_dir) = &config.output_dir {
        write_team13_outputs(output_dir, &topology, &coords, &result)?;
    }

    Ok(result)
}

pub fn solve_team13_linear_nominal(
    config: &Team13LinearConfig,
) -> Result<Team13LinearNominalResult, String> {
    validate_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    let metric = coords.to_edge_lengths(&topology);
    let reluctivity = reluctivity_weight();
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let nominal_a = solve_nominal_a(
        &topology,
        &metric,
        &coords,
        &reluctivity,
        &boundary,
        nominal_source,
    );
    let operators = build_team13_operators(&topology, &coords)?;
    let observation_overrides =
        load_surface_observation_overrides(config.observation_csv_path.as_deref())?;
    let benchmark_reports = evaluate_benchmark_reports(
        &topology,
        &coords,
        &operators,
        &nominal_a,
        &nominal_a,
        config.measurement_mode,
        config.legacy_measurement_band,
        observation_overrides.as_ref(),
    )?;

    let result = Team13LinearNominalResult {
        domain_mode: config.domain_mode,
        nominal_a,
        benchmark_reports,
    };

    if let Some(output_dir) = &config.output_dir {
        write_team13_nominal_outputs(output_dir, &topology, &coords, &result)?;
    }

    Ok(result)
}

pub fn run_team13_nonlinear_uq(
    config: &Team13NonlinearConfig,
) -> Result<Team13NonlinearSolveResult, String> {
    validate_nonlinear_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 nonlinear solve requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &linear_reluctivity,
        ));
    let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;
    let unweighted_prior_system =
        build_reduced_hodge_laplace_1form_system(&topology, &metric, &boundary)?;
    if unweighted_prior_system.layout != linear_system.layout {
        return Err(
            "TEAM13 nonlinear unweighted prior layout does not match weighted PDE layout"
                .to_string(),
        );
    }

    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;

    let nonlinear_material = build_team13_nonlinear_material(config)?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary.clone(),
        ),
    )?;
    if linear_source_free.layout().active_dofs != linear_system.layout.active_dofs {
        return Err("TEAM13 nonlinear beta-zero layout does not match linear layout".to_string());
    }
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 nonlinear and beta-zero layouts differ".to_string());
    }
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let nominal_a = lift_vector_with_layout(
        &linear_system.layout,
        &FeecVector::from_vec(linear_mean.clone()),
    )?;
    let linear_model = linear_source_free.with_source(reduced_source.clone())?;
    let beta_zero_parity_residual =
        l2_norm(&linear_model.residual_and_jacobian(&linear_mean)?.residual);
    let beta_zero_parity_tolerance = (1e-8 * l2_norm(&reduced_source).max(1.0)).max(1e-8);
    if beta_zero_parity_residual > beta_zero_parity_tolerance {
        return Err(format!(
            "TEAM13 beta-zero nonlinear residual does not match the reduced physical coil RHS: residual norm {beta_zero_parity_residual:.6e} exceeds tolerance {beta_zero_parity_tolerance:.6e}"
        ));
    }
    let model = source_free.clone().with_source(reduced_source.clone())?;

    let zero_state = vec![0.0; model.reduced_dimension()];
    let initial_residual_norm = l2_norm(&model.residual_and_jacobian(&zero_state)?.residual);
    let linear_mean_residual_norm = l2_norm(&model.residual_and_jacobian(&linear_mean)?.residual);
    let prior = build_team13_nonlinear_prior(
        config,
        &unweighted_prior_system,
        &topology,
        &coords,
        &FeecVector::from_vec(linear_mean.clone()),
    )?;
    let operators = build_team13_operators(&topology, &coords)?;
    let observation_overrides =
        load_surface_observation_overrides(config.observation_csv_path.as_deref())?;
    let measurements = build_linearized_b_measurements(
        &topology,
        &coords,
        &operators.b_cochain,
        &operators.b_components,
        &nominal_a,
        config.b_observation_std_tesla * config.b_observation_std_tesla,
        config.measurement_mode,
        config.legacy_measurement_band,
        observation_overrides.as_ref(),
    )?;
    let linear_sensor_reports = evaluate_sensor_reports(&measurements, &nominal_a)?;
    let linear_sensor_rmse = sensor_rmse_from_reports(&linear_sensor_reports);
    let reduced_measurements = if config.assimilate_measurements {
        restrict_team13_measurements_to_layout(&measurements, model.layout())?
    } else {
        Vec::new()
    };
    let assimilated_measurements = reduced_measurements.len();
    let derived_quantities = selected_sensor_derived_quantities(
        &measurements,
        model.layout(),
        config.sensor_variance_count,
    )?;
    let gradient_tolerance = if !config.assimilate_measurements
        && config.beta_iron == 0.0
        && linear_mean_residual_norm <= beta_zero_parity_tolerance
    {
        // The beta-zero run starts at the physical reduced linear solution. The
        // remaining residual is sparse-solve roundoff, so skip an unnecessary
        // Gauss-Newton correction and still build the final Laplace precision.
        1.0e300
    } else {
        1e-5
    };
    let model_adapter = FeecResidualAdapter::new(&model);
    let problem = NonlinearLaplaceProblem {
        prior: prior.spec,
        residual_terms: vec![NonlinearResidualTerm::zero(
            "team13_nonlinear_weak_residual",
            &model_adapter,
            GaussianNoiseModel::ScalarVariance(config.pde_variance),
        )],
        linear_measurements: reduced_measurements,
        precision_weighted_measurements: Vec::new(),
        derived_quantities,
    };
    let nonlinear = solve_nonlinear_laplace(
        &problem,
        &GaussNewtonConfig {
            initial_guess: Some(linear_mean.clone()),
            max_iterations: config.max_iterations,
            step_tolerance: 1e-10,
            gradient_tolerance,
            max_line_search_steps: 40,
            linear_solve: config.linear_solve,
            step_regularization: config.step_regularization,
            variance: config.variance,
            ..GaussNewtonConfig::default()
        },
    )?;

    let final_residual_norm = l2_norm(&model.residual_and_jacobian(&nonlinear.map)?.residual);
    let nonlinear_a =
        lift_vector_with_layout(model.layout(), &FeecVector::from_vec(nonlinear.map.clone()))?;
    let state_active_mask = active_dof_mask(model.layout());
    let field_metrics_vs_linear =
        field_recovery_metrics(&operators, &nominal_a, &nonlinear_a, &state_active_mask)?;
    let sensor_reports = evaluate_sensor_reports(&measurements, &nonlinear_a)?;
    let sensor_rmse = sensor_rmse_from_reports(&sensor_reports);
    let sensor_rmse_improvement_ratio = safe_ratio(sensor_rmse, linear_sensor_rmse);
    let benchmark_reports = evaluate_benchmark_reports(
        &topology,
        &coords,
        &operators,
        &nominal_a,
        &nonlinear_a,
        config.measurement_mode,
        config.legacy_measurement_band,
        observation_overrides.as_ref(),
    )?;
    let sensor_variances = sensor_variance_reports(&measurements, &nonlinear.derived_variances);
    let map_distance_from_linear_mean = l2_distance(&nonlinear.map, &linear_mean)?;
    let map_relative_distance_from_linear_mean =
        relative_l2_distance(&nonlinear.map, &linear_mean)?;
    let beta_zero_relative_error = if config.beta_iron == 0.0 {
        Some(map_relative_distance_from_linear_mean)
    } else {
        None
    };

    let result = Team13NonlinearSolveResult {
        domain_mode: config.domain_mode,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: model.reduced_dimension(),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        material_kind: config.material_kind,
        beta_iron: config.beta_iron,
        b_scale_tesla: config.b_scale_tesla,
        prior_kind: prior.kind,
        field_prior_precision_scale: prior.precision_scale,
        prior_kappa: prior.kappa,
        prior_tau: prior.tau,
        prior_kappa_fallback_used: prior.kappa_fallback_used,
        initial_residual_norm,
        linear_mean_residual_norm,
        final_residual_norm,
        map_distance_from_linear_mean,
        map_relative_distance_from_linear_mean,
        beta_zero_relative_error,
        field_metrics_vs_linear,
        linear_sensor_rmse,
        sensor_rmse,
        sensor_rmse_improvement_ratio,
        assimilated_measurements,
        sensor_reports,
        sensor_variances,
        benchmark_reports,
        history: nonlinear.history,
        assembly: nonlinear.assembly,
        final_factorization: nonlinear.final_factorization,
        converged: nonlinear.converged,
    };

    if config.write_outputs {
        if let Some(output_dir) = &config.output_dir {
            write_team13_nonlinear_outputs(
                output_dir,
                &topology,
                &coords,
                &nominal_a,
                &nonlinear_a,
                &result,
            )?;
        }
    }

    Ok(result)
}

pub fn run_team13_nonlinear_diagnostics(
    config: &Team13NonlinearDiagnosticsConfig,
) -> Result<Team13NonlinearDiagnosticsResult, String> {
    validate_nonlinear_config(&config.solve)?;
    if config.pde_variance_values.is_empty() {
        return Err("diagnostic pde_variance_values must not be empty".to_string());
    }
    if !config
        .pde_variance_values
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err("diagnostic pde_variance_values must be finite and positive".to_string());
    }

    let mesh_bytes = fs::read(&config.solve.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.solve.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 nonlinear diagnostics require a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.solve.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &linear_reluctivity,
        ));
    let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;
    let unweighted_prior_system =
        build_reduced_hodge_laplace_1form_system(&topology, &metric, &boundary)?;
    if unweighted_prior_system.layout != linear_system.layout {
        return Err(
            "TEAM13 nonlinear diagnostic unweighted prior layout does not match weighted PDE layout"
                .to_string(),
        );
    }

    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.solve.domain_mode, config.solve.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let nonlinear_material = build_team13_nonlinear_material(&config.solve)?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary.clone(),
        ),
    )?;
    if linear_source_free.layout().active_dofs != linear_system.layout.active_dofs {
        return Err(
            "TEAM13 nonlinear diagnostic beta-zero layout does not match linear layout".to_string(),
        );
    }
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 nonlinear diagnostic layouts differ".to_string());
    }

    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let linear_model = linear_source_free.with_source(reduced_source.clone())?;
    let model = source_free.with_source(reduced_source.clone())?;
    let beta_zero_residual_norm =
        l2_norm(&linear_model.residual_and_jacobian(&linear_mean)?.residual);
    let source_free_at_linear_mean = linear_model
        .source_free_residual_and_jacobian(&linear_mean)?
        .residual;
    let source_free_affine_solve_residual = source_free_at_linear_mean
        .iter()
        .zip(reduced_source.iter())
        .map(|(lhs, rhs)| *lhs - *rhs)
        .collect::<Vec<_>>();
    let nonlinear_residual_at_linear_mean_norm =
        l2_norm(&model.residual_and_jacobian(&linear_mean)?.residual);
    let prior = build_team13_nonlinear_prior(
        &config.solve,
        &unweighted_prior_system,
        &topology,
        &coords,
        &FeecVector::from_vec(linear_mean.clone()),
    )?;
    let nominal_a = lift_vector_with_layout(
        &linear_system.layout,
        &FeecVector::from_vec(linear_mean.clone()),
    )?;
    let operators = build_team13_operators(&topology, &coords)?;
    let observation_overrides =
        load_surface_observation_overrides(config.solve.observation_csv_path.as_deref())?;
    let measurements = build_linearized_b_measurements(
        &topology,
        &coords,
        &operators.b_cochain,
        &operators.b_components,
        &nominal_a,
        config.solve.b_observation_std_tesla * config.solve.b_observation_std_tesla,
        config.solve.measurement_mode,
        config.solve.legacy_measurement_band,
        observation_overrides.as_ref(),
    )?;
    let reduced_measurements = if config.solve.assimilate_measurements {
        restrict_team13_measurements_to_layout(&measurements, model.layout())?
    } else {
        Vec::new()
    };
    let assimilated_measurements = reduced_measurements.len();

    let diagnostic_regularizations = [
        GaussNewtonStepRegularization::None,
        GaussNewtonStepRegularization::LevenbergMarquardtGrid,
    ];
    let mut first_steps =
        Vec::with_capacity(config.pde_variance_values.len() * diagnostic_regularizations.len());
    let model_adapter = FeecResidualAdapter::new(&model);
    for pde_variance in &config.pde_variance_values {
        for step_regularization in diagnostic_regularizations {
            let problem = NonlinearLaplaceProblem {
                prior: prior.spec.clone(),
                residual_terms: vec![NonlinearResidualTerm::zero(
                    "team13_nonlinear_weak_residual",
                    &model_adapter,
                    GaussianNoiseModel::ScalarVariance(*pde_variance),
                )],
                linear_measurements: reduced_measurements.clone(),
                precision_weighted_measurements: Vec::new(),
                derived_quantities: Vec::new(),
            };
            let diagnostic = diagnose_gauss_newton_first_step(
                &problem,
                &GaussNewtonConfig {
                    initial_guess: Some(linear_mean.clone()),
                    max_iterations: config.solve.max_iterations,
                    step_tolerance: 1e-10,
                    gradient_tolerance: 1e-5,
                    max_line_search_steps: 40,
                    linear_solve: config.solve.linear_solve,
                    step_regularization,
                    variance: config.solve.variance,
                    ..GaussNewtonConfig::default()
                },
            );
            match diagnostic {
                Ok(diagnostics) => first_steps.push(Team13FirstStepDiagnosticReport {
                    pde_variance: *pde_variance,
                    step_regularization,
                    classification: classify_first_step_diagnostics(&diagnostics),
                    diagnostics: Some(diagnostics),
                    failure_reason: None,
                }),
                Err(err) => first_steps.push(Team13FirstStepDiagnosticReport {
                    pde_variance: *pde_variance,
                    step_regularization,
                    classification: Team13FirstStepConditioningClass::Failed,
                    diagnostics: None,
                    failure_reason: Some(err),
                }),
            }
        }
    }

    Ok(Team13NonlinearDiagnosticsResult {
        domain_mode: config.solve.domain_mode,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: model.reduced_dimension(),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        material_kind: config.solve.material_kind,
        beta_iron: config.solve.beta_iron,
        reduced_physical_rhs_norm: l2_norm(&reduced_source),
        beta_zero_residual_norm,
        source_free_affine_solve_residual_norm: l2_norm(&source_free_affine_solve_residual),
        nonlinear_residual_at_linear_mean_norm,
        prior_kind: prior.kind,
        field_prior_precision_scale: prior.precision_scale,
        assimilated_measurements,
        first_steps,
    })
}

pub fn run_team13_jacobian_sparsity_audit(
    config: &Team13JacobianSparsityAuditConfig,
) -> Result<Team13JacobianSparsityAuditResult, String> {
    let solve_config = Team13NonlinearConfig {
        mesh_path: config.mesh_path.clone(),
        domain_mode: config.domain_mode,
        ampere_turns: config.ampere_turns,
        material_kind: config.material_kind,
        beta_iron: config.beta_iron,
        b_scale_tesla: config.b_scale_tesla,
        write_outputs: false,
        ..Team13NonlinearConfig::default()
    };
    validate_nonlinear_config(&solve_config)?;

    let mesh_bytes = fs::read(&solve_config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            solve_config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 Jacobian sparsity audit requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, solve_config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &linear_reluctivity,
        ));
    let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(solve_config.domain_mode, solve_config.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let nonlinear_material = build_team13_nonlinear_material(&solve_config)?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary.clone(),
        ),
    )?;
    if linear_source_free.layout().active_dofs != linear_system.layout.active_dofs {
        return Err(
            "TEAM13 Jacobian sparsity audit beta-zero layout does not match linear layout"
                .to_string(),
        );
    }
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 Jacobian sparsity audit layouts differ".to_string());
    }

    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let linear_model = linear_source_free.with_source(reduced_source.clone())?;
    let linear_mean_residual_norm =
        l2_norm(&linear_model.residual_and_jacobian(&linear_mean)?.residual);
    let model = source_free.with_source(reduced_source.clone())?;

    let jacobian_start = Instant::now();
    let evaluation = model.residual_and_jacobian(&linear_mean)?;
    let jacobian_seconds = jacobian_start.elapsed().as_secs_f64();
    let nonlinear_residual_at_linear_mean_norm = l2_norm(&evaluation.residual);
    let jacobian = feec_csr_to_gmrf(&evaluation.jacobian);
    let jacobian_rows = jacobian.nrows();
    let jacobian_cols = jacobian.ncols();
    let jacobian_nnz = jacobian.nnz();
    let jacobian_lower_triangle_nnz = sparse_lower_triangle_nnz(&jacobian);

    let normal_start = Instant::now();
    let normal = ht_weighted_h(&jacobian, 1.0);
    let normal_product_seconds = normal_start.elapsed().as_secs_f64();
    let normal_rows = normal.nrows();
    let normal_cols = normal.ncols();
    let normal_nnz = normal.nnz();
    let normal_lower_triangle_nnz = sparse_lower_triangle_nnz(&normal);

    Ok(Team13JacobianSparsityAuditResult {
        domain_mode: solve_config.domain_mode,
        mesh_path: solve_config.mesh_path,
        material_kind: solve_config.material_kind,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: model.reduced_dimension(),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        residual_dimension: model.residual_dimension(),
        jacobian_rows,
        jacobian_cols,
        jacobian_nnz,
        jacobian_lower_triangle_nnz,
        jacobian_density: sparse_density(jacobian_rows, jacobian_cols, jacobian_nnz),
        jacobian_lower_triangle_density: sparse_lower_triangle_density(
            jacobian_rows,
            jacobian_cols,
            jacobian_lower_triangle_nnz,
        ),
        normal_rows,
        normal_cols,
        normal_nnz,
        normal_lower_triangle_nnz,
        normal_density: sparse_density(normal_rows, normal_cols, normal_nnz),
        normal_lower_triangle_density: sparse_lower_triangle_density(
            normal_rows,
            normal_cols,
            normal_lower_triangle_nnz,
        ),
        normal_to_jacobian_nnz_ratio: normal_nnz as f64 / jacobian_nnz.max(1) as f64,
        reduced_physical_rhs_norm: l2_norm(&reduced_source),
        linear_mean_residual_norm,
        nonlinear_residual_at_linear_mean_norm,
        jacobian_seconds,
        normal_product_seconds,
    })
}

fn sparse_density(rows: usize, cols: usize, nnz: usize) -> f64 {
    let capacity = (rows as f64) * (cols as f64);
    if capacity == 0.0 {
        0.0
    } else {
        nnz as f64 / capacity
    }
}

fn sparse_lower_triangle_nnz(matrix: &GmrfSparseMatrix) -> usize {
    matrix
        .triplet_iter()
        .filter(|(row, col, _)| row >= col)
        .count()
}

fn sparse_lower_triangle_density(rows: usize, cols: usize, nnz: usize) -> f64 {
    let capacity = lower_triangle_capacity(rows, cols);
    if capacity == 0 {
        0.0
    } else {
        nnz as f64 / capacity as f64
    }
}

fn lower_triangle_capacity(rows: usize, cols: usize) -> usize {
    (0..rows).map(|row| cols.min(row + 1)).sum()
}

pub fn run_team13_synthetic_nonlinear_baseline(
    config: &Team13SyntheticNonlinearBaselineConfig,
) -> Result<Team13SyntheticNonlinearBaselineResult, String> {
    validate_synthetic_nonlinear_baseline_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 synthetic nonlinear baseline requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &linear_reluctivity,
        ));
    let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;

    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let linear_material =
        Team13SmoothIronReluctivityLaw::new(NU_AIR, NU_IRON, 0.0, config.b_scale_tesla)?;
    let nonlinear_material = Team13SmoothIronReluctivityLaw::new(
        NU_AIR,
        NU_IRON,
        config.beta_iron,
        config.b_scale_tesla,
    )?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(linear_material, boundary.clone()),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(nonlinear_material, boundary.clone()),
    )?;
    if linear_source_free.layout().active_dofs != linear_system.layout.active_dofs {
        return Err(
            "TEAM13 synthetic baseline beta-zero layout does not match linear layout".to_string(),
        );
    }
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 synthetic baseline nonlinear and beta-zero layouts differ".to_string());
    }

    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let model = source_free.clone().with_source(reduced_source.clone())?;
    let initial_residual_norm = l2_norm(&model.residual_and_jacobian(&linear_mean)?.residual);

    let truth = solve_team13_feec_forward_newton(
        &model,
        linear_mean.clone(),
        config.truth_max_iterations,
        config.linear_solve,
    )?;
    let truth_residual_norm = l2_norm(&model.residual_and_jacobian(&truth.solution)?.residual);

    let initial_a =
        lift_vector_with_layout(model.layout(), &FeecVector::from_vec(linear_mean.clone()))?;
    let truth_a = lift_vector_with_layout(
        model.layout(),
        &FeecVector::from_vec(truth.solution.clone()),
    )?;
    let operators = build_team13_operators(&topology, &coords)?;
    let synthetic_measurements = build_synthetic_surface_flux_measurements(
        &topology,
        &coords,
        &operators,
        &initial_a,
        &truth_a,
        config.observation_std_tesla * config.observation_std_tesla,
    )?;
    let exact_prior = build_exact_two_form_potential_prior_with_metric(
        &topology,
        &coords,
        &metric,
        model.layout(),
        linear_mean.clone(),
        ExactTwoFormPotentialPriorConfig {
            kappa: config.prior_kappa,
            tau: config.prior_tau,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
            diagonal_shift: config.prior_diagonal_shift,
        },
    )?;
    let prior_factor = sparse_from_core(&exact_prior.spec.precision)
        .cholesky_sqrt_lower()
        .map_err(|err| {
            format!("failed to factor synthetic baseline prior for sensor variances: {err}")
        })?;
    let mut observation_runs = Vec::with_capacity(config.observation_models.len());
    for kind in &config.observation_models {
        observation_runs.push(run_team13_synthetic_observation_model(
            *kind,
            config,
            &model,
            model.layout(),
            &linear_mean,
            &truth.solution,
            &initial_a,
            &exact_prior.spec,
            &prior_factor,
            &synthetic_measurements,
            initial_residual_norm,
            truth_residual_norm,
        )?);
    }

    Ok(Team13SyntheticNonlinearBaselineResult {
        domain_mode: config.domain_mode,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: model.reduced_dimension(),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        beta_iron: config.beta_iron,
        b_scale_tesla: config.b_scale_tesla,
        prior_kappa: config.prior_kappa,
        prior_tau: config.prior_tau,
        prior_diagonal_shift: config.prior_diagonal_shift,
        magnitude_smoothing_tesla: config.magnitude_smoothing_tesla,
        synthetic_sensor_count: synthetic_measurements.len(),
        initial_residual_norm,
        truth_residual_norm,
        truth_converged: truth.converged,
        truth_history: truth.history,
        observation_runs,
    })
}

fn run_team13_synthetic_observation_model(
    kind: Team13SyntheticObservationModelKind,
    config: &Team13SyntheticNonlinearBaselineConfig,
    model: &ReducedVectorPotentialMagnetostatic3d,
    layout: &DofLayout,
    linear_mean: &[f64],
    truth_map: &[f64],
    initial_a: &FeecVector,
    exact_prior: &GaussianPriorSpec,
    prior_factor: &SparseCholeskyFactor,
    synthetic_measurements: &[Team13LinearizedMeasurement],
    initial_residual_norm: f64,
    truth_residual_norm: f64,
) -> Result<Team13SyntheticObservationRunResult, String> {
    let model_adapter = FeecResidualAdapter::new(model);
    let reduced_component_measurements =
        restrict_team13_measurements_to_layout(synthetic_measurements, layout)?;
    let observation_variance = config.observation_std_tesla * config.observation_std_tesla;
    let posterior = match kind {
        Team13SyntheticObservationModelKind::SmoothMagnitude => {
            let (smooth_abs_model, observations) =
                smooth_abs_model_and_observations_from_measurements(
                    &reduced_component_measurements,
                    config.magnitude_smoothing_tesla,
                )?;
            let posterior_problem = NonlinearLaplaceProblem {
                prior: exact_prior.clone(),
                residual_terms: vec![
                    NonlinearResidualTerm::zero(
                        "team13_synthetic_posterior_residual",
                        &model_adapter,
                        GaussianNoiseModel::ScalarVariance(config.pde_variance),
                    ),
                    NonlinearResidualTerm {
                        name: "team13_synthetic_surface_smooth_magnitude".to_string(),
                        model: &smooth_abs_model,
                        observations,
                        noise: GaussianNoiseModel::ScalarVariance(observation_variance),
                    },
                ],
                linear_measurements: Vec::new(),
                precision_weighted_measurements: Vec::new(),
                derived_quantities: Vec::new(),
            };
            solve_nonlinear_laplace(
                &posterior_problem,
                &GaussNewtonConfig {
                    initial_guess: Some(linear_mean.to_vec()),
                    max_iterations: config.max_iterations,
                    step_tolerance: 1e-10,
                    gradient_tolerance: 1e-9,
                    max_line_search_steps: 40,
                    linear_solve: config.linear_solve,
                    step_regularization: config.step_regularization,
                    variance: config.variance,
                    ..GaussNewtonConfig::default()
                },
            )?
        }
        Team13SyntheticObservationModelKind::SignedLinearProxy => {
            let signed_measurements = signed_linear_proxy_measurements(
                &reduced_component_measurements,
                synthetic_measurements,
            )?;
            let posterior_problem = NonlinearLaplaceProblem {
                prior: exact_prior.clone(),
                residual_terms: vec![NonlinearResidualTerm::zero(
                    "team13_synthetic_posterior_residual",
                    &model_adapter,
                    GaussianNoiseModel::ScalarVariance(config.pde_variance),
                )],
                linear_measurements: signed_measurements,
                precision_weighted_measurements: Vec::new(),
                derived_quantities: Vec::new(),
            };
            solve_nonlinear_laplace(
                &posterior_problem,
                &GaussNewtonConfig {
                    initial_guess: Some(linear_mean.to_vec()),
                    max_iterations: config.max_iterations,
                    step_tolerance: 1e-10,
                    gradient_tolerance: 1e-9,
                    max_line_search_steps: 40,
                    linear_solve: config.linear_solve,
                    step_regularization: config.step_regularization,
                    variance: config.variance,
                    ..GaussNewtonConfig::default()
                },
            )?
        }
    };

    let posterior_residual_norm = l2_norm(&model.residual_and_jacobian(&posterior.map)?.residual);
    let posterior_a =
        lift_vector_with_layout(layout, &FeecVector::from_vec(posterior.map.clone()))?;
    let initial_sensor_reports = evaluate_smooth_abs_sensor_reports(
        synthetic_measurements,
        initial_a,
        config.magnitude_smoothing_tesla,
    )?;
    let sensor_reports = evaluate_smooth_abs_sensor_reports(
        synthetic_measurements,
        &posterior_a,
        config.magnitude_smoothing_tesla,
    )?;
    let initial_sensor_rmse = sensor_rmse_from_reports(&initial_sensor_reports);
    let posterior_sensor_rmse = sensor_rmse_from_reports(&sensor_reports);
    let sensor_variances = smooth_abs_sensor_variance_reports(
        synthetic_measurements,
        &reduced_component_measurements,
        config.magnitude_smoothing_tesla,
        prior_factor,
        &posterior,
    )?;
    let initial_relative_error = relative_l2_distance(linear_mean, truth_map)?;
    let posterior_relative_error = relative_l2_distance(&posterior.map, truth_map)?;
    let all_finite_variances = posterior
        .posterior_variance
        .iter()
        .all(|value| value.is_finite())
        && sensor_variances.iter().all(|report| {
            report.prior_variance.is_finite() && report.posterior_variance.is_finite()
        });
    let nonnegative_variances = posterior
        .posterior_variance
        .iter()
        .all(|value| *value >= -1e-12)
        && sensor_variances
            .iter()
            .all(|report| report.prior_variance >= -1e-12 && report.posterior_variance >= -1e-12);

    Ok(Team13SyntheticObservationRunResult {
        model_kind: kind,
        synthetic_sensor_count: synthetic_measurements.len(),
        sign_mismatch_count: synthetic_surface_sign_mismatch_count(synthetic_measurements),
        initial_relative_error,
        posterior_relative_error,
        initial_residual_norm,
        truth_residual_norm,
        posterior_residual_norm,
        initial_sensor_rmse,
        posterior_sensor_rmse,
        sensor_rmse_improvement_ratio: safe_ratio(posterior_sensor_rmse, initial_sensor_rmse),
        posterior_converged: posterior.converged,
        all_finite_variances,
        nonnegative_variances,
        assembly: posterior.assembly,
        final_factorization: posterior.final_factorization,
        posterior_history: posterior.history,
        sensor_reports,
        sensor_variances,
    })
}

pub fn run_team13_synthetic_benchmark_geometry(
    config: &Team13SyntheticBenchmarkGeometryConfig,
) -> Result<Team13SyntheticBenchmarkGeometryResult, String> {
    validate_synthetic_benchmark_geometry_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 synthetic benchmark-geometry baseline requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &linear_reluctivity,
        ));
    let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;

    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let nonlinear_material = build_team13_synthetic_benchmark_material(config)?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary.clone(),
        ),
    )?;
    if linear_source_free.layout().active_dofs != linear_system.layout.active_dofs {
        return Err(
            "TEAM13 synthetic benchmark beta-zero layout does not match linear layout".to_string(),
        );
    }
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err(
            "TEAM13 synthetic benchmark nonlinear and beta-zero layouts differ".to_string(),
        );
    }

    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let model = source_free.clone().with_source(reduced_source.clone())?;
    let initial_residual_norm = l2_norm(&model.residual_and_jacobian(&linear_mean)?.residual);

    let truth = solve_team13_feec_forward_newton(
        &model,
        linear_mean.clone(),
        config.truth_max_iterations,
        config.linear_solve,
    )?;
    let truth_residual_norm = l2_norm(&model.residual_and_jacobian(&truth.solution)?.residual);

    let exact_prior = build_exact_two_form_potential_prior_with_metric(
        &topology,
        &coords,
        &metric,
        model.layout(),
        linear_mean.clone(),
        ExactTwoFormPotentialPriorConfig {
            kappa: config.prior_kappa,
            tau: config.prior_tau,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
            diagonal_shift: config.prior_diagonal_shift,
        },
    )?;
    let prior_factor = sparse_from_core(&exact_prior.spec.precision)
        .cholesky_sqrt_lower()
        .map_err(|err| {
            format!("failed to factor synthetic benchmark prior for sensor variances: {err}")
        })?;
    let exact_prior_diagnostics = prior_variance_diagnostics(&prior_factor)?;

    let operators = build_team13_operators(&topology, &coords)?;
    let observations = build_team13_synthetic_benchmark_geometry_observations(
        &topology,
        &coords,
        &operators,
        model.layout(),
        &linear_mean,
        &truth.solution,
        config.steel_observation_quadrature,
        config.magnitude_smoothing_tesla,
    )?;

    let default_run = run_team13_synthetic_benchmark_geometry_fit(
        config,
        &topology,
        &coords,
        &operators,
        model.layout(),
        config.pde_variance,
        config.observation_std_tesla,
        &model,
        &linear_mean,
        &truth.solution,
        &exact_prior.spec,
        &prior_factor,
        &exact_prior_diagnostics,
        &observations,
        initial_residual_norm,
        truth_residual_norm,
    )?;

    let mut sweep_runs = Vec::new();
    for pde_variance in &config.sweep_pde_variances {
        for observation_std_tesla in &config.sweep_observation_std_tesla {
            if same_positive_scale(*pde_variance, config.pde_variance)
                && same_positive_scale(*observation_std_tesla, config.observation_std_tesla)
            {
                continue;
            }
            sweep_runs.push(run_team13_synthetic_benchmark_geometry_fit(
                config,
                &topology,
                &coords,
                &operators,
                model.layout(),
                *pde_variance,
                *observation_std_tesla,
                &model,
                &linear_mean,
                &truth.solution,
                &exact_prior.spec,
                &prior_factor,
                &exact_prior_diagnostics,
                &observations,
                initial_residual_norm,
                truth_residual_norm,
            )?);
        }
    }

    let source_scale_diagnostics = run_team13_source_scale_diagnostics(
        config,
        &linear_source_free,
        &source_free,
        &reduced_source,
        &observations,
    )?;

    let steel_observation_count = observations
        .specs
        .iter()
        .filter(|spec| spec.group == Team13SyntheticBenchmarkObservationGroup::SteelAverage)
        .count();
    let air_observation_count = observations
        .specs
        .iter()
        .filter(|spec| spec.group == Team13SyntheticBenchmarkObservationGroup::AirPoint)
        .count();
    let assimilated_observation_count = observations.assimilated_specs.len();

    Ok(Team13SyntheticBenchmarkGeometryResult {
        domain_mode: config.domain_mode,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: model.reduced_dimension(),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        material_kind: config.material_kind,
        beta_iron: config.beta_iron,
        b_scale_tesla: config.b_scale_tesla,
        prior_kappa: config.prior_kappa,
        prior_tau: config.prior_tau,
        prior_diagonal_shift: config.prior_diagonal_shift,
        steel_observation_quadrature: config.steel_observation_quadrature,
        magnitude_smoothing_tesla: config.magnitude_smoothing_tesla,
        observation_count: observations.specs.len(),
        assimilated_observation_count,
        steel_observation_count,
        air_observation_count,
        initial_residual_norm,
        truth_residual_norm,
        truth_converged: truth.converged,
        truth_history: truth.history,
        default_run,
        sweep_runs,
        source_scale_diagnostics,
    })
}

pub fn run_team13_map_parity(
    config: &Team13MapParityConfig,
) -> Result<Team13MapParityResult, String> {
    validate_team13_map_parity_config(config)?;
    let total_start = Instant::now();
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 MAP parity: reading mesh `{}`",
        config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 MAP parity mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 MAP parity requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 MAP parity: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    eprintln!(
        "TEAM 13 MAP parity: assembled metric/source data in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let nonlinear_material = build_team13_material_from_kind(
        config.material_kind,
        config.beta_iron,
        config.b_scale_tesla,
    )?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary.clone(),
        ),
    )?;
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 MAP parity nonlinear and beta-zero layouts differ".to_string());
    }
    eprintln!(
        "TEAM 13 MAP parity: built nonlinear models in {:.2?} (active_dofs={})",
        phase_start.elapsed(),
        source_free.reduced_dimension()
    );

    phase_start = Instant::now();
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let model = source_free.clone().with_source(reduced_source)?;
    let posterior_likelihood_model = match config.pde_residual_kind {
        Team13MapParityPdeResidualKind::GaugeFixed => model.clone(),
        Team13MapParityPdeResidualKind::UngaugedCurl => model.clone().without_coulomb_gauge(),
    };
    let initial_residual_norm = l2_norm(&model.residual_and_jacobian(&linear_mean)?.residual);
    let truth_cache_path = team13_truth_cache_path(&mesh_bytes, config, model.reduced_dimension());
    let truth_residual_tolerance = 1.0e-6 * initial_residual_norm.max(1.0);
    let cached_truth = if config.force_truth_solve {
        None
    } else {
        let model_adapter = FeecResidualAdapter::new(&model);
        try_load_team13_truth_cache(
            &truth_cache_path,
            &model_adapter,
            model.reduced_dimension(),
            truth_residual_tolerance,
        )
    };
    let (truth, truth_cache_hit) = if let Some(truth) = cached_truth {
        eprintln!(
            "TEAM 13 MAP parity: truth cache hit `{}` (residual={:.6e})",
            truth_cache_path.display(),
            truth.residual_norm
        );
        (truth, true)
    } else {
        eprintln!(
            "TEAM 13 MAP parity: truth cache miss `{}`",
            truth_cache_path.display()
        );
        let truth = solve_team13_feec_forward_newton(
            &model,
            linear_mean.clone(),
            config.truth_max_iterations,
            config.linear_solve,
        )?;
        if truth.converged && truth.residual_norm <= truth_residual_tolerance {
            if let Err(err) = write_team13_truth_cache(&truth_cache_path, &truth) {
                eprintln!(
                    "TEAM 13 MAP parity: truth cache write skipped for `{}`: {err}",
                    truth_cache_path.display()
                );
            } else {
                eprintln!(
                    "TEAM 13 MAP parity: truth cache wrote `{}`",
                    truth_cache_path.display()
                );
            }
        } else {
            eprintln!(
                "TEAM 13 MAP parity: truth cache write skipped because residual {:.6e} exceeds tolerance {:.6e} or solve did not converge",
                truth.residual_norm,
                truth_residual_tolerance
            );
        }
        (truth, false)
    };
    let truth_residual_norm = l2_norm(&model.residual_and_jacobian(&truth.solution)?.residual);
    let likelihood_initial_residual_norm = l2_norm(
        &posterior_likelihood_model
            .residual_and_jacobian(&linear_mean)?
            .residual,
    );
    let likelihood_truth_residual_norm = l2_norm(
        &posterior_likelihood_model
            .residual_and_jacobian(&truth.solution)?
            .residual,
    );
    eprintln!(
        "TEAM 13 MAP parity: solved internal truth in {:.2?} (converged={} residual={:.6e}->{:.6e})",
        phase_start.elapsed(),
        truth.converged,
        initial_residual_norm,
        truth_residual_norm
    );

    phase_start = Instant::now();
    let prior_spec = match config.prior_kind {
        Team13MapParityPriorKind::ExactPotential => {
            build_exact_two_form_potential_prior_with_metric(
                &topology,
                &coords,
                &metric,
                model.layout(),
                linear_mean.clone(),
                ExactTwoFormPotentialPriorConfig {
                    kappa: config.prior_kappa,
                    tau: config.prior_tau,
                    mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
                    diagonal_shift: config.prior_diagonal_shift,
                },
            )?
            .spec
        }
        Team13MapParityPriorKind::OrdinaryMaternAlpha2 => {
            let state_mass_inverse =
                FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
                    &topology,
                    &metric,
                    &coords,
                    None,
                    &linear_reluctivity,
                ));
            let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
                &galmats,
                &boundary,
                &state_mass_inverse,
            )?;
            if linear_system.state_dimension() != model.reduced_dimension() {
                return Err(format!(
                    "ordinary TEAM 13 MAP parity prior dimension {} does not match nonlinear model dimension {}",
                    linear_system.state_dimension(),
                    model.reduced_dimension()
                ));
            }
            let prior = build_hodge_matern_prior_from_reduced_system_with_params(
                &linear_system,
                &FeecVector::from_vec(linear_mean.clone()),
                config.prior_kappa,
                config.prior_tau,
            )?;
            add_diagonal_shift_to_gaussian_prior(prior, config.prior_diagonal_shift)?
        }
        Team13MapParityPriorKind::WeakRidge => build_weak_ridge_prior(
            &linear_mean,
            config.prior_tau * config.prior_tau + config.prior_diagonal_shift,
        )?,
    };
    let prior_factor = sparse_from_core(&prior_spec.precision)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor TEAM 13 MAP parity prior: {err}"))?;
    eprintln!(
        "TEAM 13 MAP parity: built/factored {} prior in {:.2?} (prior_nnz={} factor_nnz={})",
        config.prior_kind.as_str(),
        phase_start.elapsed(),
        prior_spec.precision.nnz(),
        prior_factor.nnz()
    );

    phase_start = Instant::now();
    let operators = build_team13_operators(&topology, &coords)?;
    let observations = build_team13_synthetic_benchmark_geometry_observations(
        &topology,
        &coords,
        &operators,
        model.layout(),
        &linear_mean,
        &truth.solution,
        config.steel_observation_quadrature,
        config.magnitude_smoothing_tesla,
    )?;
    eprintln!(
        "TEAM 13 MAP parity: built internal steel observations in {:.2?} (steel={})",
        phase_start.elapsed(),
        observations.assimilated_specs.len()
    );
    eprintln!(
        "TEAM 13 MAP parity: posterior PDE residual mode {} has residual {:.6e}->{:.6e} at linear/truth states",
        config.pde_residual_kind.as_str(),
        likelihood_initial_residual_norm,
        likelihood_truth_residual_norm
    );

    let default_run = run_team13_map_parity_fit(
        "default",
        config,
        config.pde_variance,
        config.observation_std_tesla,
        &posterior_likelihood_model,
        &linear_mean,
        &truth.solution,
        &prior_spec,
        &prior_factor,
        &observations,
        likelihood_initial_residual_norm,
        likelihood_truth_residual_norm,
    )?;

    let mut sweep_runs = Vec::new();
    for pde_variance in &config.sweep_pde_variances {
        for observation_std_tesla in &config.sweep_observation_std_tesla {
            if same_positive_scale(*pde_variance, config.pde_variance)
                && same_positive_scale(*observation_std_tesla, config.observation_std_tesla)
            {
                continue;
            }
            sweep_runs.push(run_team13_map_parity_fit(
                "sweep",
                config,
                *pde_variance,
                *observation_std_tesla,
                &posterior_likelihood_model,
                &linear_mean,
                &truth.solution,
                &prior_spec,
                &prior_factor,
                &observations,
                likelihood_initial_residual_norm,
                likelihood_truth_residual_norm,
            )?);
        }
    }

    let result = Team13MapParityResult {
        domain_mode: config.domain_mode,
        mesh_path: config.mesh_path.clone(),
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: model.reduced_dimension(),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        material_kind: config.material_kind,
        beta_iron: config.beta_iron,
        b_scale_tesla: config.b_scale_tesla,
        prior_kind: config.prior_kind,
        prior_kappa: config.prior_kappa,
        prior_tau: config.prior_tau,
        prior_diagonal_shift: config.prior_diagonal_shift,
        pde_residual_kind: config.pde_residual_kind,
        step_regularization: config.step_regularization,
        steel_observation_quadrature: config.steel_observation_quadrature,
        magnitude_smoothing_tesla: config.magnitude_smoothing_tesla,
        initial_residual_norm,
        truth_residual_norm,
        truth_converged: truth.converged,
        truth_cache_hit,
        truth_history: truth.history,
        default_run,
        sweep_runs,
        output_dir: config.output_dir.clone(),
    };

    if let Some(output_dir) = &config.output_dir {
        write_team13_map_parity_outputs(output_dir, &result)?;
    }
    eprintln!(
        "TEAM 13 MAP parity: completed in {:.2?}",
        total_start.elapsed()
    );
    Ok(result)
}

pub fn run_team13_operator_uncertainty_diagnostic(
    config: &Team13OperatorUncertaintyConfig,
) -> Result<Team13OperatorUncertaintyResult, String> {
    validate_team13_operator_uncertainty_config(config)?;
    let total_start = Instant::now();
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 operator uncertainty: reading mesh `{}`",
        config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 operator uncertainty mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 operator uncertainty requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 operator uncertainty: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let operators = build_team13_operators(&topology, &coords)?;
    let audit = audit_team13_regions(&topology, &coords, config.domain_mode, config.ampere_turns)?;
    eprintln!(
        "TEAM 13 operator uncertainty: assembled geometry/source/operators in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let nonlinear_material = build_team13_material_from_kind_with_log_scale(
        config.material_kind,
        config.beta_iron,
        config.b_scale_tesla,
        config.material_log_iron_nu_scale,
    )?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let nonlinear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary.clone(),
        ),
    )?;
    if nonlinear_source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err(
            "TEAM13 operator uncertainty nonlinear and beta-zero layouts differ".to_string(),
        );
    }
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let nonlinear_model = nonlinear_source_free
        .clone()
        .with_source(reduced_source.clone())?;
    let linear_model = linear_source_free.with_source(reduced_source)?;
    let deterministic = match config.tangent_kind {
        Team13OperatorUncertaintyTangentKind::Nonlinear => solve_team13_feec_forward_newton(
            &nonlinear_model,
            linear_mean.clone(),
            config.truth_max_iterations,
            config.linear_solve,
        )?,
        Team13OperatorUncertaintyTangentKind::LinearBetaZero => solve_team13_feec_forward_newton(
            &linear_model,
            linear_mean.clone(),
            config.truth_max_iterations,
            config.linear_solve,
        )?,
    };
    let deterministic_model = match config.tangent_kind {
        Team13OperatorUncertaintyTangentKind::Nonlinear => nonlinear_model,
        Team13OperatorUncertaintyTangentKind::LinearBetaZero => linear_model,
    };
    let posterior_likelihood_model = match config.pde_residual_kind {
        Team13MapParityPdeResidualKind::GaugeFixed => deterministic_model.clone(),
        Team13MapParityPdeResidualKind::UngaugedCurl => {
            deterministic_model.clone().without_coulomb_gauge()
        }
    };
    let deterministic_residual_norm =
        l2_norm(&deterministic_model.residual(&deterministic.solution)?);
    eprintln!(
        "TEAM 13 operator uncertainty: built models and deterministic {} state in {:.2?} (converged={} residual={:.6e})",
        config.tangent_kind.as_str(),
        phase_start.elapsed(),
        deterministic.converged,
        deterministic_residual_norm
    );

    phase_start = Instant::now();
    let prior_spec = build_team13_map_style_prior(
        config.prior_kind,
        &topology,
        &coords,
        &metric,
        &galmats,
        &boundary,
        deterministic_model.layout(),
        &linear_reluctivity,
        &FeecVector::from_vec(linear_mean.clone()),
        config.prior_kappa,
        config.prior_tau,
        config.prior_diagonal_shift,
    )?;
    let prior_factor = sparse_from_core(&prior_spec.precision)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor TEAM 13 operator uncertainty prior: {err}"))?;
    eprintln!(
        "TEAM 13 operator uncertainty: built/factored {} prior in {:.2?} (prior_nnz={} factor_nnz={})",
        config.prior_kind.as_str(),
        phase_start.elapsed(),
        prior_spec.precision.nnz(),
        prior_factor.nnz()
    );

    phase_start = Instant::now();
    let pde_noise =
        team13_operator_pde_noise(config, posterior_likelihood_model.state_mass_inverse())?;
    let full_deterministic = lift_vector_with_layout(
        deterministic_model.layout(),
        &FeecVector::from_vec(deterministic.solution.clone()),
    )?;
    let linear_measurements = if config.include_steel_observations {
        let full_measurements = build_linearized_b_measurements(
            &topology,
            &coords,
            &operators.b_cochain,
            &operators.b_components,
            &full_deterministic,
            config.observation_std_tesla * config.observation_std_tesla,
            team13_measurement_mode_from_steel_quadrature(config.steel_observation_quadrature),
            0.03,
            None,
        )?;
        restrict_team13_measurements_to_layout(&full_measurements, deterministic_model.layout())?
    } else {
        Vec::new()
    };
    let posterior_likelihood_adapter = FeecResidualAdapter::new(&posterior_likelihood_model);
    let posterior_problem = NonlinearLaplaceProblem {
        prior: prior_spec.clone(),
        residual_terms: vec![NonlinearResidualTerm::zero(
            "team13_operator_uncertainty_pde_residual",
            &posterior_likelihood_adapter,
            pde_noise,
        )],
        linear_measurements,
        precision_weighted_measurements: Vec::new(),
        derived_quantities: Vec::new(),
    };
    let mut posterior = solve_nonlinear_laplace(
        &posterior_problem,
        &GaussNewtonConfig {
            initial_guess: Some(deterministic.solution.clone()),
            max_iterations: 1,
            gradient_tolerance: 1.0e300,
            step_tolerance: 1.0e-10,
            max_line_search_steps: 1,
            linear_solve: config.linear_solve,
            step_regularization: GaussNewtonStepRegularization::None,
            reuse_cholesky_stabilization_shift: true,
            estimate_latent_variance: false,
            variance: config.field_variance,
            ..GaussNewtonConfig::default()
        },
    )?;
    if relative_l2_distance(&posterior.map, &deterministic.solution)? > 1.0e-10 {
        return Err(
            "operator uncertainty fixed-state Laplace moved away from deterministic state"
                .to_string(),
        );
    }
    eprintln!(
        "TEAM 13 operator uncertainty: assembled/factored fixed-state posterior in {:.2?} (posterior_nnz={} factor_nnz={})",
        phase_start.elapsed(),
        posterior.assembly.posterior_precision_nnz,
        posterior.final_factorization.nnz
    );

    phase_start = Instant::now();
    let steel_patch_operator = build_team13_reduced_steel_patch_operator(
        &topology,
        &coords,
        &operators,
        deterministic_model.layout(),
        config.steel_observation_quadrature,
    )?;
    let steel_patch_reports = team13_operator_steel_patch_variances(
        &steel_patch_operator,
        &deterministic.solution,
        &prior_factor,
        &posterior,
    )?;
    eprintln!(
        "TEAM 13 operator uncertainty: computed exact steel patch variances in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let (field_variance_estimator, field_cell_variances) = if config.estimate_field_variance {
        let b_operator =
            build_interleaved_b_vector_operator(&operators, deterministic_model.layout())?;
        let (estimator, variances) = estimate_team13_b_vector_variances(
            &mut posterior,
            &b_operator,
            &config.field_variance,
        )?;
        (
            estimator,
            Some(cell_trace_variances_from_interleaved(&variances)?),
        )
    } else {
        ("disabled".to_string(), None)
    };
    let b_field = evaluate_cell_b_field(&operators, full_deterministic.as_slice())?;
    let region_diagnostics = build_team13_operator_region_diagnostics(
        &topology,
        &coords,
        &b_field,
        field_cell_variances.as_deref(),
    )?;
    let (region_summaries, indicator_correlations) =
        summarize_team13_operator_regions(&region_diagnostics);
    eprintln!(
        "TEAM 13 operator uncertainty: computed field/region diagnostics in {:.2?} (field_variance={})",
        phase_start.elapsed(),
        field_variance_estimator
    );

    let result = Team13OperatorUncertaintyResult {
        domain_mode: config.domain_mode,
        mesh_path: config.mesh_path.clone(),
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: deterministic_model.reduced_dimension(),
        boundary_edge_dofs: deterministic_model.boundary_edge_dofs().len(),
        material_kind: config.material_kind,
        material_log_iron_nu_scale: config.material_log_iron_nu_scale,
        tangent_kind: config.tangent_kind,
        prior_kind: config.prior_kind,
        pde_residual_kind: config.pde_residual_kind,
        pde_residual_weighting: config.pde_residual_weighting,
        pde_variance: config.pde_variance,
        include_steel_observations: config.include_steel_observations,
        steel_observation_quadrature: config.steel_observation_quadrature,
        observation_std_tesla: config.observation_std_tesla,
        deterministic_converged: deterministic.converged,
        deterministic_residual_norm,
        posterior_converged: posterior.converged,
        posterior_precision_nnz: posterior.assembly.posterior_precision_nnz,
        posterior_factor_nnz: posterior.final_factorization.nnz,
        fill_ratio_vs_lower: posterior
            .assembly
            .fill_ratio_vs_lower_triangle
            .unwrap_or(f64::NAN),
        field_variance_estimator,
        field_variance_available: field_cell_variances.is_some(),
        steel_patch_reports,
        region_summaries,
        indicator_correlations,
        audit,
        output_dir: config.output_dir.clone(),
    };

    if let Some(output_dir) = &config.output_dir {
        write_team13_operator_uncertainty_outputs(output_dir, &result)?;
    }
    eprintln!(
        "TEAM 13 operator uncertainty: completed in {:.2?}",
        total_start.elapsed()
    );
    Ok(result)
}

pub fn run_team13_material_gap_uq(
    config: &Team13MaterialGapUqConfig,
) -> Result<Team13MaterialGapUqResult, String> {
    validate_team13_material_gap_uq_config(config)?;
    let total_weight = team13_material_gap_case_total_weight(config);
    let mut case_results = Vec::with_capacity(config.gap_cases.len() * config.material_nodes.len());

    for gap_case in &config.gap_cases {
        for material_node in &config.material_nodes {
            let mut operator_config = config.operator.clone();
            operator_config.mesh_path = gap_case.mesh_path.clone();
            operator_config.material_log_iron_nu_scale = material_node.log_iron_nu_scale;
            operator_config.output_dir = config.output_dir.as_ref().map(|root| {
                root.join("cases").join(format!(
                    "{}_{}",
                    sanitize_path_token(&gap_case.label),
                    sanitize_path_token(&material_node.label)
                ))
            });

            let normalized_weight = gap_case.weight * material_node.weight / total_weight;
            eprintln!(
                "TEAM 13 material/gap UQ: running gap={} ({:.6e} m), material={} ({:.6e}), weight={:.6e}",
                gap_case.label,
                gap_case.steel_gap_m,
                material_node.label,
                material_node.log_iron_nu_scale,
                normalized_weight
            );
            let operator_result = run_team13_operator_uncertainty_diagnostic(&operator_config)?;
            let rmse_vs_observed_gap = gap_case.observed_gap.map_or(f64::NAN, |gap| {
                team13_operator_patch_rmse_vs_gap(&operator_result.steel_patch_reports, gap)
            });
            let mean_patch_std = mean_or_nan(
                &operator_result
                    .steel_patch_reports
                    .iter()
                    .map(|report| report.posterior_std)
                    .collect::<Vec<_>>(),
            );

            case_results.push(Team13MaterialGapUqCaseResult {
                gap_label: gap_case.label.clone(),
                steel_gap_m: gap_case.steel_gap_m,
                observed_gap: gap_case.observed_gap,
                material_label: material_node.label.clone(),
                material_log_iron_nu_scale: material_node.log_iron_nu_scale,
                normalized_weight,
                rmse_vs_observed_gap,
                mean_patch_std,
                operator_result,
            });
        }
    }

    let variance_decomposition = team13_material_gap_variance_decomposition(&case_results)?;
    let result = Team13MaterialGapUqResult {
        case_results,
        variance_decomposition,
        output_dir: config.output_dir.clone(),
    };
    if let Some(output_dir) = &config.output_dir {
        write_team13_material_gap_uq_outputs(output_dir, &result)?;
    }
    Ok(result)
}

pub fn run_team13_material_only_uq(
    config: &Team13MaterialOnlyUqConfig,
) -> Result<Team13MaterialOnlyUqResult, String> {
    validate_team13_material_only_uq_config(config)?;
    let total_start = Instant::now();
    let operator_config = &config.operator;
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 material-only UQ: reading mesh `{}`",
        operator_config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&operator_config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 material-only UQ mesh `{}`: {err}",
            operator_config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 material-only UQ requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 material-only UQ: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, operator_config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(
            operator_config.domain_mode,
            operator_config.ampere_turns,
            None,
        ),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let operators = build_team13_operators(&topology, &coords)?;
    eprintln!(
        "TEAM 13 material-only UQ: assembled geometry/source/operators in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let nominal_material =
        build_team13_tabulated_material_with_log_h_shape(config.material_anchor_b_tesla, [0.0; 3])?;
    let linear_material = nominal_material.linear_reference_law();
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(linear_material, boundary.clone()),
    )?;
    let nominal_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(nominal_material, boundary.clone()),
    )?;
    if nominal_source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 material-only UQ nominal and beta-zero layouts differ".to_string());
    }
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let nominal_model = nominal_source_free
        .clone()
        .with_source(reduced_source.clone())?;
    let nominal_forward = solve_team13_feec_forward_newton(
        &nominal_model,
        linear_mean.clone(),
        operator_config.truth_max_iterations,
        operator_config.linear_solve,
    )?;
    eprintln!(
        "TEAM 13 material-only UQ: nominal forward state in {:.2?} (converged={} residual={:.6e})",
        phase_start.elapsed(),
        nominal_forward.converged,
        nominal_forward.residual_norm
    );
    if !nominal_forward.converged {
        return Err(format!(
            "TEAM 13 material-only UQ nominal forward solve did not converge (residual={:.6e})",
            nominal_forward.residual_norm
        ));
    }

    phase_start = Instant::now();
    let steel_patch_operator = build_team13_reduced_steel_patch_operator(
        &topology,
        &coords,
        &operators,
        nominal_model.layout(),
        operator_config.steel_observation_quadrature,
    )?;
    let calibration_augmented_model = Team13MaterialShapeAugmentedResidualModel::new(
        nominal_model.clone(),
        config.material_anchor_b_tesla,
        nominal_model.layout().clone(),
    )?;
    let material_prior_calibration = team13_calibrated_material_only_prior(
        config,
        &calibration_augmented_model,
        &nominal_forward.solution,
        &steel_patch_operator,
    )?;
    let material_prior_std = material_prior_calibration.material_prior_std;
    let observations = team13_published_steel_observations(config.observed_steel_gap);
    let context = Team13MaterialOnlyForwardContext {
        model: nominal_model.clone(),
        augmented_model: calibration_augmented_model,
        patch_operator: steel_patch_operator,
        anchors: config.material_anchor_b_tesla,
        observations,
        smoothing: config.magnitude_smoothing_tesla,
        observation_std_tesla: operator_config.observation_std_tesla,
        forward_max_iterations: operator_config.truth_max_iterations,
        linear_solve: operator_config.linear_solve,
    };
    eprintln!(
        "TEAM 13 material-only UQ: built theta-only context in {:.2?} (material_prior_std={:.6e} calibration={} target={})",
        phase_start.elapsed(),
        material_prior_std,
        material_prior_calibration.mode.as_str(),
        material_prior_calibration.target.as_str()
    );

    phase_start = Instant::now();
    let initial_theta_evaluation = context.evaluate_known_forward(
        [0.0; 3],
        nominal_forward.solution.clone(),
        nominal_forward.converged,
        nominal_forward.residual_norm,
        true,
    )?;
    let solve = solve_team13_material_only_theta(
        &context,
        initial_theta_evaluation,
        material_prior_std,
        config.max_iterations,
        config.max_line_search_steps,
        config.max_theta_step_norm,
        config.step_regularization,
    )?;
    eprintln!(
        "TEAM 13 material-only UQ: solved theta-only problem in {:.2?} (converged={} theta=[{:.6e}, {:.6e}, {:.6e}] objective={:.6e})",
        phase_start.elapsed(),
        solve.converged,
        solve.theta[0],
        solve.theta[1],
        solve.theta[2],
        solve.objective_components.total
    );

    phase_start = Instant::now();
    let covariance = invert_3x3(solve.final_hessian)
        .ok_or_else(|| "TEAM 13 material-only posterior precision is singular".to_string())?;
    let correlation = correlation_from_covariance(covariance);
    let material_posterior = team13_material_parameter_reports(
        config.material_anchor_b_tesla,
        material_prior_std,
        solve.theta,
        covariance,
    );
    let bh_curve_bands =
        team13_bh_curve_bands(config.material_anchor_b_tesla, solve.theta, covariance)?;
    let steel_predictions = team13_material_only_steel_prediction_reports(
        &context,
        &nominal_forward.solution,
        &solve.final_evaluation.forward_state,
        &solve.final_evaluation,
    )?;
    let sensitivity = team13_material_only_sensitivity_report(&solve.final_evaluation.jacobian)?;
    let comparison = team13_material_only_comparison_rows(
        &solve,
        &steel_predictions,
        operator_config.observation_std_tesla,
    );
    eprintln!(
        "TEAM 13 material-only UQ: computed reports in {:.2?}",
        phase_start.elapsed()
    );

    let result = Team13MaterialOnlyUqResult {
        domain_mode: operator_config.domain_mode,
        mesh_path: operator_config.mesh_path.clone(),
        observed_steel_gap: config.observed_steel_gap,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: nominal_model.reduced_dimension(),
        boundary_edge_dofs: nominal_model.boundary_edge_dofs().len(),
        material_anchor_b_tesla: config.material_anchor_b_tesla,
        material_prior_std,
        material_prior_calibration,
        magnitude_smoothing_tesla: config.magnitude_smoothing_tesla,
        nominal_forward_converged: nominal_forward.converged,
        nominal_forward_residual_norm: nominal_forward.residual_norm,
        map_forward_converged: solve.final_evaluation.forward_converged,
        map_forward_residual_norm: solve.final_evaluation.forward_residual_norm,
        posterior_converged: solve.converged,
        theta_map: solve.theta,
        material_posterior,
        material_posterior_covariance: covariance,
        material_posterior_correlation: correlation,
        bh_curve_bands,
        steel_predictions,
        objective_components: solve.objective_components,
        sensitivity,
        history: solve.history,
        comparison,
        output_dir: config.output_dir.clone(),
    };
    if let Some(output_dir) = &config.output_dir {
        write_team13_material_only_uq_outputs(output_dir, &result)?;
    }
    eprintln!(
        "TEAM 13 material-only UQ: completed in {:.2?}",
        total_start.elapsed()
    );
    Ok(result)
}

pub fn run_team13_identifiable_material_uq(
    config: &Team13IdentifiableMaterialUqConfig,
) -> Result<Team13IdentifiableMaterialUqResult, String> {
    validate_team13_identifiable_material_uq_config(config)?;
    let total_start = Instant::now();
    let operator_config = &config.operator;
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 identifiable material UQ: reading mesh `{}`",
        operator_config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&operator_config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 identifiable material UQ mesh `{}`: {err}",
            operator_config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 identifiable material UQ requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 identifiable material UQ: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, operator_config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(
            operator_config.domain_mode,
            operator_config.ampere_turns,
            None,
        ),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let operators = build_team13_operators(&topology, &coords)?;
    eprintln!(
        "TEAM 13 identifiable material UQ: assembled geometry/source/operators in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let benchmark_material =
        build_team13_tabulated_material_with_log_h_shape(config.material_anchor_b_tesla, [0.0; 3])?;
    let linear_material = benchmark_material.linear_reference_law();
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(linear_material, boundary.clone()),
    )?;
    let benchmark_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(benchmark_material, boundary.clone()),
    )?;
    if benchmark_source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err(
            "TEAM13 identifiable material UQ benchmark and beta-zero layouts differ".to_string(),
        );
    }
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let benchmark_model = benchmark_source_free
        .clone()
        .with_source(reduced_source.clone())?;
    let benchmark_forward = solve_team13_feec_forward_newton(
        &benchmark_model,
        linear_mean,
        operator_config.truth_max_iterations,
        operator_config.linear_solve,
    )?;
    eprintln!(
        "TEAM 13 identifiable material UQ: benchmark law forward state in {:.2?} (converged={} residual={:.6e})",
        phase_start.elapsed(),
        benchmark_forward.converged,
        benchmark_forward.residual_norm
    );
    if !benchmark_forward.converged {
        return Err(format!(
            "TEAM 13 identifiable material UQ benchmark forward solve did not converge (residual={:.6e})",
            benchmark_forward.residual_norm
        ));
    }

    phase_start = Instant::now();
    let steel_patch_operator = build_team13_reduced_steel_patch_operator(
        &topology,
        &coords,
        &operators,
        benchmark_model.layout(),
        operator_config.steel_observation_quadrature,
    )?;
    let augmented_model = Team13MaterialShapeAugmentedResidualModel::new(
        benchmark_model.clone(),
        config.material_anchor_b_tesla,
        benchmark_model.layout().clone(),
    )?;
    let observations = team13_published_steel_observations(config.observed_steel_gap);
    let context = Team13MaterialOnlyForwardContext {
        model: benchmark_model.clone(),
        augmented_model,
        patch_operator: steel_patch_operator,
        anchors: config.material_anchor_b_tesla,
        observations,
        smoothing: config.magnitude_smoothing_tesla,
        observation_std_tesla: operator_config.observation_std_tesla,
        forward_max_iterations: operator_config.truth_max_iterations,
        linear_solve: operator_config.linear_solve,
    };
    let benchmark_evaluation = context.evaluate_known_forward(
        [0.0; 3],
        benchmark_forward.solution.clone(),
        benchmark_forward.converged,
        benchmark_forward.residual_norm,
        true,
    )?;
    let basis = team13_identifiable_material_basis(
        &benchmark_evaluation.jacobian,
        config.svd_relative_tolerance,
        config.svd_absolute_tolerance,
    )?;
    if basis.retained_modes.is_empty() {
        return Err(
            "TEAM 13 identifiable material UQ found no observable material SVD modes".to_string(),
        );
    }
    let eta_prior_calibration = team13_calibrated_identifiable_eta_prior(
        config,
        &basis,
        benchmark_evaluation.jacobian.len(),
    )?;
    let eta_prior_std = eta_prior_calibration.eta_prior_std;
    let target_gap_rms = team13_published_steel_gap_difference_rms();
    let baseline_perturbation = team13_identifiable_baseline_perturbation(
        &basis,
        &benchmark_evaluation.jacobian,
        config.perturbation_rms_fraction_of_gap * target_gap_rms,
        target_gap_rms,
    )?;
    eprintln!(
        "TEAM 13 identifiable material UQ: built SVD basis in {:.2?} (rank={} singular=[{:.6e}, {:.6e}, {:.6e}] eta_prior_std={:.6e} theta_bias=[{:.6e}, {:.6e}, {:.6e}])",
        phase_start.elapsed(),
        basis.retained_modes.len(),
        basis.singular_values[0],
        basis.singular_values[1],
        basis.singular_values[2],
        eta_prior_std,
        baseline_perturbation.theta_bias[0],
        baseline_perturbation.theta_bias[1],
        baseline_perturbation.theta_bias[2]
    );

    phase_start = Instant::now();
    let mut biased_forward_state = benchmark_forward.solution.clone();
    let mut biased_forward_converged = benchmark_forward.converged;
    let mut biased_forward_residual_norm = benchmark_forward.residual_norm;
    for step in 1..=config.continuation_steps {
        let scale = step as f64 / config.continuation_steps as f64;
        let theta = scale_theta(baseline_perturbation.theta_bias, scale);
        let forward = solve_team13_material_only_forward(
            &benchmark_model,
            config.material_anchor_b_tesla,
            theta,
            biased_forward_state,
            operator_config.truth_max_iterations,
            operator_config.linear_solve,
        )?;
        eprintln!(
            "TEAM 13 identifiable material UQ: bias continuation step {}/{} scale={:.3e} converged={} residual={:.6e}",
            step,
            config.continuation_steps,
            scale,
            forward.converged,
            forward.residual_norm
        );
        if !forward.converged {
            return Err(format!(
                "TEAM 13 identifiable material UQ biased-baseline continuation failed at step {step}/{} (residual={:.6e})",
                config.continuation_steps,
                forward.residual_norm
            ));
        }
        biased_forward_converged = forward.converged;
        biased_forward_residual_norm = forward.residual_norm;
        biased_forward_state = forward.solution;
    }
    let biased_evaluation = context.evaluate_known_forward(
        baseline_perturbation.theta_bias,
        biased_forward_state,
        biased_forward_converged,
        biased_forward_residual_norm,
        true,
    )?;
    let initial_eta_evaluation = team13_identifiable_eta_evaluation_from_theta_evaluation(
        &baseline_perturbation.theta_bias,
        &basis.retained_modes,
        &vec![0.0; basis.retained_modes.len()],
        biased_evaluation.clone(),
    )?;
    eprintln!(
        "TEAM 13 identifiable material UQ: solved biased baseline in {:.2?} (residual={:.6e})",
        phase_start.elapsed(),
        biased_forward_residual_norm
    );

    phase_start = Instant::now();
    let solve = solve_team13_identifiable_eta(
        &context,
        baseline_perturbation.theta_bias,
        &basis.retained_modes,
        initial_eta_evaluation,
        eta_prior_std,
        config.max_iterations,
        config.max_line_search_steps,
        config.max_eta_step_norm,
        config.step_regularization,
    )?;
    eprintln!(
        "TEAM 13 identifiable material UQ: solved eta problem in {:.2?} (converged={} rank={} objective={:.6e})",
        phase_start.elapsed(),
        solve.converged,
        solve.eta.len(),
        solve.objective_components.total
    );

    phase_start = Instant::now();
    let eta_covariance = invert_dense_matrix(&solve.final_hessian)?;
    let theta_covariance =
        team13_theta_covariance_from_eta_covariance(&basis.retained_modes, &eta_covariance)?;
    let eta_posterior = team13_identifiable_eta_posterior_reports(
        eta_prior_std,
        &baseline_perturbation.eta_bias,
        &solve.eta,
        &eta_covariance,
    );
    let theta_posterior = team13_identifiable_theta_posterior_reports(
        config.material_anchor_b_tesla,
        baseline_perturbation.theta_bias,
        solve.final_evaluation.theta,
        theta_covariance,
    );
    let bh_curve_bands = team13_identifiable_bh_curve_bands(
        config.material_anchor_b_tesla,
        baseline_perturbation.theta_bias,
        solve.final_evaluation.theta,
        theta_covariance,
    )?;
    let steel_predictions = team13_identifiable_steel_prediction_reports(
        &context,
        &benchmark_evaluation,
        &biased_evaluation,
        &solve.final_evaluation,
    )?;
    let comparison = team13_identifiable_comparison_rows(
        &benchmark_evaluation,
        &biased_evaluation,
        &solve.final_evaluation,
        &baseline_perturbation,
        &basis.retained_modes,
        eta_prior_std,
        operator_config.observation_std_tesla,
    );
    eprintln!(
        "TEAM 13 identifiable material UQ: computed reports in {:.2?}",
        phase_start.elapsed()
    );

    let result = Team13IdentifiableMaterialUqResult {
        domain_mode: operator_config.domain_mode,
        mesh_path: operator_config.mesh_path.clone(),
        observed_steel_gap: config.observed_steel_gap,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: benchmark_model.reduced_dimension(),
        boundary_edge_dofs: benchmark_model.boundary_edge_dofs().len(),
        material_anchor_b_tesla: config.material_anchor_b_tesla,
        eta_prior_std,
        eta_prior_calibration,
        magnitude_smoothing_tesla: config.magnitude_smoothing_tesla,
        svd_relative_tolerance: config.svd_relative_tolerance,
        svd_absolute_tolerance: config.svd_absolute_tolerance,
        retained_rank: basis.retained_modes.len(),
        singular_values: basis.singular_values,
        material_basis: basis.mode_reports,
        baseline_perturbation,
        benchmark_forward_converged: benchmark_forward.converged,
        benchmark_forward_residual_norm: benchmark_forward.residual_norm,
        biased_forward_converged,
        biased_forward_residual_norm,
        map_forward_converged: solve.final_evaluation.forward_converged,
        map_forward_residual_norm: solve.final_evaluation.forward_residual_norm,
        posterior_converged: solve.converged,
        eta_map: solve.eta,
        theta_map: solve.final_evaluation.theta,
        eta_posterior,
        theta_posterior,
        eta_posterior_covariance: eta_covariance,
        theta_posterior_covariance: theta_covariance,
        bh_curve_bands,
        steel_predictions,
        objective_components: solve.objective_components,
        history: solve.history,
        comparison,
        output_dir: config.output_dir.clone(),
    };
    if let Some(output_dir) = &config.output_dir {
        write_team13_identifiable_material_uq_outputs(output_dir, &result)?;
    }
    eprintln!(
        "TEAM 13 identifiable material UQ: completed in {:.2?}",
        total_start.elapsed()
    );
    Ok(result)
}

pub fn run_team13_identifiable_joint_material_uq(
    config: &Team13IdentifiableJointMaterialUqConfig,
) -> Result<Team13IdentifiableJointMaterialUqResult, String> {
    validate_team13_identifiable_joint_material_uq_config(config)?;
    let total_start = Instant::now();
    let operator_config = &config.operator;
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 identifiable joint material UQ: reading mesh `{}`",
        operator_config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&operator_config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 identifiable joint material UQ mesh `{}`: {err}",
            operator_config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 identifiable joint material UQ requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 identifiable joint material UQ: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, operator_config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(
            operator_config.domain_mode,
            operator_config.ampere_turns,
            None,
        ),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let operators = build_team13_operators(&topology, &coords)?;
    eprintln!(
        "TEAM 13 identifiable joint material UQ: assembled geometry/source/operators in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let benchmark_material =
        build_team13_tabulated_material_with_log_h_shape(config.material_anchor_b_tesla, [0.0; 3])?;
    let linear_material = benchmark_material.linear_reference_law();
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(linear_material, boundary.clone()),
    )?;
    let benchmark_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(benchmark_material, boundary.clone()),
    )?;
    if benchmark_source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err(
            "TEAM13 identifiable joint material UQ benchmark and beta-zero layouts differ"
                .to_string(),
        );
    }
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let benchmark_model = benchmark_source_free
        .clone()
        .with_source(reduced_source.clone())?;
    let benchmark_forward = solve_team13_feec_forward_newton(
        &benchmark_model,
        linear_mean.clone(),
        operator_config.truth_max_iterations,
        operator_config.linear_solve,
    )?;
    eprintln!(
        "TEAM 13 identifiable joint material UQ: benchmark law forward state in {:.2?} (converged={} residual={:.6e})",
        phase_start.elapsed(),
        benchmark_forward.converged,
        benchmark_forward.residual_norm
    );
    if !benchmark_forward.converged {
        return Err(format!(
            "TEAM 13 identifiable joint material UQ benchmark forward solve did not converge (residual={:.6e})",
            benchmark_forward.residual_norm
        ));
    }

    phase_start = Instant::now();
    let state_prior = build_team13_map_style_prior(
        operator_config.prior_kind,
        &topology,
        &coords,
        &metric,
        &galmats,
        &boundary,
        benchmark_model.layout(),
        &linear_reluctivity,
        &FeecVector::from_vec(linear_mean.clone()),
        operator_config.prior_kappa,
        operator_config.prior_tau,
        operator_config.prior_diagonal_shift,
    )?;
    let steel_patch_operator = build_team13_reduced_steel_patch_operator(
        &topology,
        &coords,
        &operators,
        benchmark_model.layout(),
        operator_config.steel_observation_quadrature,
    )?;
    let calibration_augmented_model = Team13MaterialShapeAugmentedResidualModel::new(
        benchmark_model.clone(),
        config.material_anchor_b_tesla,
        benchmark_model.layout().clone(),
    )?;
    let observations = team13_published_steel_observations(config.observed_steel_gap);
    let material_context = Team13MaterialOnlyForwardContext {
        model: benchmark_model.clone(),
        augmented_model: calibration_augmented_model,
        patch_operator: steel_patch_operator.clone(),
        anchors: config.material_anchor_b_tesla,
        observations,
        smoothing: config.magnitude_smoothing_tesla,
        observation_std_tesla: operator_config.observation_std_tesla,
        forward_max_iterations: operator_config.truth_max_iterations,
        linear_solve: operator_config.linear_solve,
    };
    let benchmark_evaluation = material_context.evaluate_known_forward(
        [0.0; 3],
        benchmark_forward.solution.clone(),
        benchmark_forward.converged,
        benchmark_forward.residual_norm,
        true,
    )?;
    let basis = team13_identifiable_material_basis(
        &benchmark_evaluation.jacobian,
        config.svd_relative_tolerance,
        config.svd_absolute_tolerance,
    )?;
    if basis.retained_modes.is_empty() {
        return Err(
            "TEAM 13 identifiable joint material UQ found no observable material SVD modes"
                .to_string(),
        );
    }
    let eta_prior_calibration = team13_calibrated_identifiable_joint_eta_prior(
        config,
        &basis,
        benchmark_evaluation.jacobian.len(),
    )?;
    let eta_prior_std = eta_prior_calibration.eta_prior_std;
    let target_gap_rms = team13_published_steel_gap_difference_rms();
    let baseline_perturbation = team13_identifiable_baseline_perturbation(
        &basis,
        &benchmark_evaluation.jacobian,
        config.perturbation_rms_fraction_of_gap * target_gap_rms,
        target_gap_rms,
    )?;
    let joint_prior = append_independent_material_prior(
        state_prior.clone(),
        basis.retained_modes.len(),
        1.0 / (eta_prior_std * eta_prior_std),
    )?;
    eprintln!(
        "TEAM 13 identifiable joint material UQ: built SVD basis/prior in {:.2?} (rank={} singular=[{:.6e}, {:.6e}, {:.6e}] eta_prior_std={:.6e} theta_bias=[{:.6e}, {:.6e}, {:.6e}])",
        phase_start.elapsed(),
        basis.retained_modes.len(),
        basis.singular_values[0],
        basis.singular_values[1],
        basis.singular_values[2],
        eta_prior_std,
        baseline_perturbation.theta_bias[0],
        baseline_perturbation.theta_bias[1],
        baseline_perturbation.theta_bias[2]
    );

    phase_start = Instant::now();
    let mut biased_forward_state = benchmark_forward.solution.clone();
    let mut biased_forward_converged = benchmark_forward.converged;
    let mut biased_forward_residual_norm = benchmark_forward.residual_norm;
    for step in 1..=config.continuation_steps {
        let scale = step as f64 / config.continuation_steps as f64;
        let theta = scale_theta(baseline_perturbation.theta_bias, scale);
        let forward = solve_team13_material_only_forward(
            &benchmark_model,
            config.material_anchor_b_tesla,
            theta,
            biased_forward_state,
            operator_config.truth_max_iterations,
            operator_config.linear_solve,
        )?;
        eprintln!(
            "TEAM 13 identifiable joint material UQ: bias continuation step {}/{} scale={:.3e} converged={} residual={:.6e}",
            step,
            config.continuation_steps,
            scale,
            forward.converged,
            forward.residual_norm
        );
        if !forward.converged {
            return Err(format!(
                "TEAM 13 identifiable joint material UQ biased-baseline continuation failed at step {step}/{} (residual={:.6e})",
                config.continuation_steps,
                forward.residual_norm
            ));
        }
        biased_forward_converged = forward.converged;
        biased_forward_residual_norm = forward.residual_norm;
        biased_forward_state = forward.solution;
    }
    eprintln!(
        "TEAM 13 identifiable joint material UQ: solved biased baseline in {:.2?} (residual={:.6e})",
        phase_start.elapsed(),
        biased_forward_residual_norm
    );

    phase_start = Instant::now();
    let likelihood_template_model = match operator_config.pde_residual_kind {
        Team13MapParityPdeResidualKind::GaugeFixed => benchmark_model.clone(),
        Team13MapParityPdeResidualKind::UngaugedCurl => {
            benchmark_model.clone().without_coulomb_gauge()
        }
    };
    let pde_noise = team13_operator_pde_noise(
        operator_config,
        likelihood_template_model.state_mass_inverse(),
    )?;
    let biased_material = build_team13_tabulated_material_with_log_h_shape(
        config.material_anchor_b_tesla,
        baseline_perturbation.theta_bias,
    )?;
    let fixed_biased_likelihood_model = Team13FixedThetaForwardModel {
        model: likelihood_template_model.clone(),
        material: biased_material,
    };
    let steel_observations = if operator_config.include_steel_observations {
        Some(build_team13_published_steel_smooth_observations(
            &topology,
            &coords,
            &operators,
            benchmark_model.layout(),
            operator_config.steel_observation_quadrature,
            config.magnitude_smoothing_tesla,
            config.observed_steel_gap,
        )?)
    } else {
        None
    };
    let fixed_solver_config = GaussNewtonConfig {
        initial_guess: Some(biased_forward_state.clone()),
        max_iterations: config.max_iterations,
        step_tolerance: 1.0e-10,
        gradient_tolerance: 1.0e-9,
        max_line_search_steps: 40,
        linear_solve: operator_config.linear_solve,
        step_regularization: config.step_regularization,
        reuse_cholesky_stabilization_shift: true,
        estimate_latent_variance: false,
        variance: operator_config.field_variance,
        ..GaussNewtonConfig::default()
    };
    let mut fixed_residual_terms = vec![NonlinearResidualTerm::zero(
        "team13_fixed_biased_material_pde_residual",
        &fixed_biased_likelihood_model,
        pde_noise.clone(),
    )];
    if let Some(observations) = &steel_observations {
        fixed_residual_terms.push(NonlinearResidualTerm {
            name: "team13_fixed_biased_material_steel_smooth_magnitude".to_string(),
            model: &observations.model,
            observations: observations.observations.clone(),
            noise: GaussianNoiseModel::ScalarVariance(
                operator_config.observation_std_tesla * operator_config.observation_std_tesla,
            ),
        });
    }
    let fixed_problem = NonlinearLaplaceProblem {
        prior: state_prior.clone(),
        residual_terms: fixed_residual_terms,
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: Vec::new(),
    };
    let fixed_posterior = solve_nonlinear_laplace(&fixed_problem, &fixed_solver_config)?;
    let fixed_posterior_residual_norm =
        l2_norm(&fixed_biased_likelihood_model.residual(&fixed_posterior.map)?);
    let fixed_final_step =
        team13_final_step_diagnostic(&fixed_problem, &fixed_solver_config, &fixed_posterior.map);
    let fixed_objective_components = team13_joint_material_objective_components(
        "fixed_biased_material",
        &state_prior,
        None,
        &fixed_posterior.map,
        &fixed_problem.linear_measurements,
        &fixed_posterior.final_residuals,
        fixed_posterior
            .history
            .last()
            .map(|iteration| iteration.trial_objective),
    )?;
    let fixed_biased_material_solve = team13_material_solve_diagnostics(
        "fixed_biased_material",
        benchmark_model.reduced_dimension(),
        0,
        operator_config.prior_kind,
        operator_config.pde_residual_kind,
        &fixed_problem.linear_measurements,
        fixed_posterior_residual_norm,
        &fixed_posterior,
        fixed_final_step,
        fixed_objective_components,
    );
    eprintln!(
        "TEAM 13 identifiable joint material UQ: solved fixed-biased comparator in {:.2?} (converged={} residual={:.6e} iterations={})",
        phase_start.elapsed(),
        fixed_biased_material_solve.converged,
        fixed_biased_material_solve.posterior_residual_norm,
        fixed_biased_material_solve.history.len()
    );

    phase_start = Instant::now();
    let joint_model = Team13IdentifiableJointMaterialResidualModel::new(
        likelihood_template_model,
        config.material_anchor_b_tesla,
        baseline_perturbation.theta_bias,
        basis.retained_modes.clone(),
        benchmark_model.layout().clone(),
    )?;
    let joint_steel_observation_model = steel_observations
        .as_ref()
        .map(|observations| {
            append_zero_theta_columns_to_smooth_grouped_norm_model(
                &observations.model,
                basis.retained_modes.len(),
            )
        })
        .transpose()?;
    let mut initial_guess = biased_forward_state.clone();
    initial_guess.extend(std::iter::repeat(0.0).take(basis.retained_modes.len()));
    let joint_solver_config = GaussNewtonConfig {
        initial_guess: Some(initial_guess),
        max_iterations: config.max_iterations,
        step_tolerance: 1.0e-10,
        gradient_tolerance: 1.0e-9,
        max_line_search_steps: 40,
        linear_solve: operator_config.linear_solve,
        step_regularization: config.step_regularization,
        reuse_cholesky_stabilization_shift: true,
        estimate_latent_variance: false,
        variance: operator_config.field_variance,
        ..GaussNewtonConfig::default()
    };
    let mut joint_residual_terms = vec![NonlinearResidualTerm::zero(
        "team13_identifiable_joint_material_pde_residual",
        &joint_model,
        pde_noise,
    )];
    if let (Some(observations), Some(model)) = (&steel_observations, &joint_steel_observation_model)
    {
        joint_residual_terms.push(NonlinearResidualTerm {
            name: "team13_identifiable_joint_material_steel_smooth_magnitude".to_string(),
            model,
            observations: observations.observations.clone(),
            noise: GaussianNoiseModel::ScalarVariance(
                operator_config.observation_std_tesla * operator_config.observation_std_tesla,
            ),
        });
    }
    let posterior_problem = NonlinearLaplaceProblem {
        prior: joint_prior,
        residual_terms: joint_residual_terms,
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: Vec::new(),
    };
    let mut posterior = solve_nonlinear_laplace(&posterior_problem, &joint_solver_config)?;
    let posterior_residual_norm = l2_norm(&joint_model.residual(&posterior.map)?);
    let joint_final_step =
        team13_final_step_diagnostic(&posterior_problem, &joint_solver_config, &posterior.map);
    let joint_objective_components = team13_joint_material_objective_components(
        "joint_identifiable_material",
        &state_prior,
        Some(eta_prior_std),
        &posterior.map,
        &posterior_problem.linear_measurements,
        &posterior.final_residuals,
        posterior
            .history
            .last()
            .map(|iteration| iteration.trial_objective),
    )?;
    let joint_identifiable_material_solve = team13_material_solve_diagnostics(
        "joint_identifiable_material",
        benchmark_model.reduced_dimension(),
        basis.retained_modes.len(),
        operator_config.prior_kind,
        operator_config.pde_residual_kind,
        &posterior_problem.linear_measurements,
        posterior_residual_norm,
        &posterior,
        joint_final_step,
        joint_objective_components,
    );
    eprintln!(
        "TEAM 13 identifiable joint material UQ: solved joint Laplace in {:.2?} (converged={} residual={:.6e} posterior_nnz={} factor_nnz={})",
        phase_start.elapsed(),
        posterior.converged,
        posterior_residual_norm,
        posterior.assembly.posterior_precision_nnz,
        posterior.final_factorization.nnz
    );

    phase_start = Instant::now();
    let n_state = benchmark_model.reduced_dimension();
    let eta_mean = posterior.map[n_state..].to_vec();
    let theta_mean = team13_eta_to_theta(
        baseline_perturbation.theta_bias,
        &basis.retained_modes,
        &eta_mean,
    )?;
    let posterior_state_mean = posterior.map[..n_state].to_vec();
    let (eta_covariance, steel_patch_reports) = team13_identifiable_joint_patch_reports(
        &mut posterior,
        &steel_patch_operator,
        &posterior_state_mean,
        &observations,
        n_state,
        basis.retained_modes.len(),
    )?;
    let theta_covariance =
        team13_theta_covariance_from_eta_covariance(&basis.retained_modes, &eta_covariance)?;
    let eta_posterior = team13_identifiable_eta_posterior_reports(
        eta_prior_std,
        &baseline_perturbation.eta_bias,
        &eta_mean,
        &eta_covariance,
    );
    let theta_posterior = team13_identifiable_theta_posterior_reports(
        config.material_anchor_b_tesla,
        baseline_perturbation.theta_bias,
        theta_mean,
        theta_covariance,
    );
    let bh_curve_bands = team13_identifiable_bh_curve_bands(
        config.material_anchor_b_tesla,
        baseline_perturbation.theta_bias,
        theta_mean,
        theta_covariance,
    )?;
    eprintln!(
        "TEAM 13 identifiable joint material UQ: computed eta/material/patch reports in {:.2?}",
        phase_start.elapsed()
    );

    let result = Team13IdentifiableJointMaterialUqResult {
        domain_mode: operator_config.domain_mode,
        mesh_path: operator_config.mesh_path.clone(),
        observed_steel_gap: config.observed_steel_gap,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: n_state,
        boundary_edge_dofs: benchmark_model.boundary_edge_dofs().len(),
        material_anchor_b_tesla: config.material_anchor_b_tesla,
        eta_prior_std,
        eta_prior_calibration,
        magnitude_smoothing_tesla: config.magnitude_smoothing_tesla,
        svd_relative_tolerance: config.svd_relative_tolerance,
        svd_absolute_tolerance: config.svd_absolute_tolerance,
        retained_rank: basis.retained_modes.len(),
        singular_values: basis.singular_values,
        material_basis: basis.mode_reports,
        baseline_perturbation,
        benchmark_forward_converged: benchmark_forward.converged,
        benchmark_forward_residual_norm: benchmark_forward.residual_norm,
        biased_forward_converged,
        biased_forward_residual_norm,
        posterior_converged: posterior.converged,
        posterior_residual_norm,
        posterior_precision_nnz: posterior.assembly.posterior_precision_nnz,
        posterior_factor_nnz: posterior.final_factorization.nnz,
        eta_map: eta_mean,
        theta_map: theta_mean,
        eta_posterior,
        theta_posterior,
        eta_posterior_covariance: eta_covariance,
        theta_posterior_covariance: theta_covariance,
        bh_curve_bands,
        steel_patch_reports,
        fixed_biased_material_solve,
        joint_identifiable_material_solve,
        output_dir: config.output_dir.clone(),
    };
    if let Some(output_dir) = &config.output_dir {
        write_team13_identifiable_joint_material_uq_outputs(output_dir, &result)?;
    }
    eprintln!(
        "TEAM 13 identifiable joint material UQ: completed in {:.2?}",
        total_start.elapsed()
    );
    Ok(result)
}

pub fn run_team13_joint_material_uq(
    config: &Team13JointMaterialUqConfig,
) -> Result<Team13JointMaterialUqResult, String> {
    validate_team13_joint_material_uq_config(config)?;
    let total_start = Instant::now();
    let operator_config = &config.operator;
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 joint material UQ: reading mesh `{}`",
        operator_config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&operator_config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 joint material UQ mesh `{}`: {err}",
            operator_config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 joint material UQ requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 joint material UQ: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, operator_config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(
            operator_config.domain_mode,
            operator_config.ampere_turns,
            None,
        ),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let operators = build_team13_operators(&topology, &coords)?;
    eprintln!(
        "TEAM 13 joint material UQ: assembled geometry/source/operators in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let nominal_material =
        build_team13_tabulated_material_with_log_h_shape(config.material_anchor_b_tesla, [0.0; 3])?;
    let linear_material = nominal_material.linear_reference_law();
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(linear_material, boundary.clone()),
    )?;
    let nominal_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(nominal_material, boundary.clone()),
    )?;
    if nominal_source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 joint material UQ nominal and beta-zero layouts differ".to_string());
    }
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let nominal_model = nominal_source_free
        .clone()
        .with_source(reduced_source.clone())?;
    let deterministic = solve_team13_feec_forward_newton(
        &nominal_model,
        linear_mean.clone(),
        operator_config.truth_max_iterations,
        operator_config.linear_solve,
    )?;
    let deterministic_likelihood_model = match operator_config.pde_residual_kind {
        Team13MapParityPdeResidualKind::GaugeFixed => nominal_model.clone(),
        Team13MapParityPdeResidualKind::UngaugedCurl => {
            nominal_model.clone().without_coulomb_gauge()
        }
    };
    let deterministic_residual_norm =
        l2_norm(&deterministic_likelihood_model.residual(&deterministic.solution)?);
    eprintln!(
        "TEAM 13 joint material UQ: deterministic nominal state in {:.2?} (converged={} residual={:.6e})",
        phase_start.elapsed(),
        deterministic.converged,
        deterministic_residual_norm
    );

    phase_start = Instant::now();
    let state_prior = build_team13_map_style_prior(
        operator_config.prior_kind,
        &topology,
        &coords,
        &metric,
        &galmats,
        &boundary,
        nominal_model.layout(),
        &linear_reluctivity,
        &FeecVector::from_vec(linear_mean.clone()),
        operator_config.prior_kappa,
        operator_config.prior_tau,
        operator_config.prior_diagonal_shift,
    )?;
    let steel_patch_operator = build_team13_reduced_steel_patch_operator(
        &topology,
        &coords,
        &operators,
        nominal_model.layout(),
        operator_config.steel_observation_quadrature,
    )?;
    let calibration_augmented_model = Team13MaterialShapeAugmentedResidualModel::new(
        nominal_model.clone(),
        config.material_anchor_b_tesla,
        nominal_model.layout().clone(),
    )?;
    let material_prior_calibration = team13_calibrated_material_prior(
        config,
        &calibration_augmented_model,
        &deterministic.solution,
        &steel_patch_operator,
    )?;
    let material_prior_std = material_prior_calibration.material_prior_std;
    let joint_prior = append_independent_material_prior(
        state_prior.clone(),
        3,
        1.0 / (material_prior_std * material_prior_std),
    )?;
    eprintln!(
        "TEAM 13 joint material UQ: built joint prior in {:.2?} (dimension={} nnz={} material_prior_std={:.6e} calibration={} target={})",
        phase_start.elapsed(),
        joint_prior.dimension(),
        joint_prior.precision.nnz(),
        material_prior_std,
        material_prior_calibration.mode.as_str(),
        material_prior_calibration.target.as_str()
    );

    phase_start = Instant::now();
    let augmented_template_model = match operator_config.pde_residual_kind {
        Team13MapParityPdeResidualKind::GaugeFixed => nominal_model.clone(),
        Team13MapParityPdeResidualKind::UngaugedCurl => {
            nominal_model.clone().without_coulomb_gauge()
        }
    };
    let augmented_model = Team13MaterialShapeAugmentedResidualModel::new(
        augmented_template_model,
        config.material_anchor_b_tesla,
        nominal_model.layout().clone(),
    )?;
    let pde_noise = team13_operator_pde_noise(
        operator_config,
        deterministic_likelihood_model.state_mass_inverse(),
    )?;
    let steel_observations = if operator_config.include_steel_observations {
        Some(build_team13_published_steel_smooth_observations(
            &topology,
            &coords,
            &operators,
            nominal_model.layout(),
            operator_config.steel_observation_quadrature,
            config.magnitude_smoothing_tesla,
            config.observed_steel_gap,
        )?)
    } else {
        None
    };
    let joint_steel_observation_model = steel_observations
        .as_ref()
        .map(|observations| {
            append_zero_theta_columns_to_smooth_grouped_norm_model(&observations.model, 3)
        })
        .transpose()?;
    if let Some(observations) = &steel_observations {
        eprintln!(
            "TEAM 13 joint material UQ: using {} published steel smooth-magnitude observations ({}, smoothing={:.3e})",
            observations.specs.len(),
            config.observed_steel_gap.as_str(),
            config.magnitude_smoothing_tesla
        );
    }
    let fixed_solver_config = GaussNewtonConfig {
        initial_guess: Some(deterministic.solution.clone()),
        max_iterations: config.max_iterations,
        step_tolerance: 1.0e-10,
        gradient_tolerance: 1.0e-9,
        max_line_search_steps: 40,
        linear_solve: operator_config.linear_solve,
        step_regularization: config.step_regularization,
        reuse_cholesky_stabilization_shift: true,
        estimate_latent_variance: false,
        variance: operator_config.field_variance,
        ..GaussNewtonConfig::default()
    };
    let deterministic_likelihood_adapter =
        FeecResidualAdapter::new(&deterministic_likelihood_model);
    let mut fixed_residual_terms = vec![NonlinearResidualTerm::zero(
        "team13_fixed_material_pde_residual",
        &deterministic_likelihood_adapter,
        pde_noise.clone(),
    )];
    if let Some(observations) = &steel_observations {
        fixed_residual_terms.push(NonlinearResidualTerm {
            name: "team13_fixed_material_steel_smooth_magnitude".to_string(),
            model: &observations.model,
            observations: observations.observations.clone(),
            noise: GaussianNoiseModel::ScalarVariance(
                operator_config.observation_std_tesla * operator_config.observation_std_tesla,
            ),
        });
    }
    let fixed_problem = NonlinearLaplaceProblem {
        prior: state_prior.clone(),
        residual_terms: fixed_residual_terms,
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: Vec::new(),
    };
    let fixed_posterior = solve_nonlinear_laplace(&fixed_problem, &fixed_solver_config)?;
    let fixed_posterior_residual_norm =
        l2_norm(&deterministic_likelihood_model.residual(&fixed_posterior.map)?);
    let fixed_final_step =
        team13_final_step_diagnostic(&fixed_problem, &fixed_solver_config, &fixed_posterior.map);
    let fixed_objective_components = team13_joint_material_objective_components(
        "fixed_material",
        &state_prior,
        None,
        &fixed_posterior.map,
        &fixed_problem.linear_measurements,
        &fixed_posterior.final_residuals,
        fixed_posterior
            .history
            .last()
            .map(|iteration| iteration.trial_objective),
    )?;
    let fixed_material_solve = team13_material_solve_diagnostics(
        "fixed_material",
        nominal_model.reduced_dimension(),
        0,
        operator_config.prior_kind,
        operator_config.pde_residual_kind,
        &fixed_problem.linear_measurements,
        fixed_posterior_residual_norm,
        &fixed_posterior,
        fixed_final_step,
        fixed_objective_components,
    );
    eprintln!(
        "TEAM 13 joint material UQ: solved fixed-material comparator (converged={} residual={:.6e} iterations={})",
        fixed_material_solve.converged,
        fixed_material_solve.posterior_residual_norm,
        fixed_material_solve.history.len()
    );

    let mut initial_guess = deterministic.solution.clone();
    initial_guess.extend([0.0; 3]);
    let joint_solver_config = GaussNewtonConfig {
        initial_guess: Some(initial_guess.clone()),
        max_iterations: config.max_iterations,
        step_tolerance: 1.0e-10,
        gradient_tolerance: 1.0e-9,
        max_line_search_steps: 40,
        linear_solve: operator_config.linear_solve,
        step_regularization: config.step_regularization,
        reuse_cholesky_stabilization_shift: true,
        estimate_latent_variance: false,
        variance: operator_config.field_variance,
        ..GaussNewtonConfig::default()
    };
    let mut joint_residual_terms = vec![NonlinearResidualTerm::zero(
        "team13_joint_material_pde_residual",
        &augmented_model,
        pde_noise,
    )];
    if let (Some(observations), Some(model)) = (&steel_observations, &joint_steel_observation_model)
    {
        joint_residual_terms.push(NonlinearResidualTerm {
            name: "team13_joint_material_steel_smooth_magnitude".to_string(),
            model,
            observations: observations.observations.clone(),
            noise: GaussianNoiseModel::ScalarVariance(
                operator_config.observation_std_tesla * operator_config.observation_std_tesla,
            ),
        });
    }
    let posterior_problem = NonlinearLaplaceProblem {
        prior: joint_prior,
        residual_terms: joint_residual_terms,
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: Vec::new(),
    };
    let mut posterior = solve_nonlinear_laplace(&posterior_problem, &joint_solver_config)?;
    let posterior_residual_norm = l2_norm(&augmented_model.residual(&posterior.map)?);
    let joint_final_step =
        team13_final_step_diagnostic(&posterior_problem, &joint_solver_config, &posterior.map);
    let joint_objective_components = team13_joint_material_objective_components(
        "joint_material",
        &state_prior,
        Some(material_prior_std),
        &posterior.map,
        &posterior_problem.linear_measurements,
        &posterior.final_residuals,
        posterior
            .history
            .last()
            .map(|iteration| iteration.trial_objective),
    )?;
    let joint_material_solve = team13_material_solve_diagnostics(
        "joint_material",
        nominal_model.reduced_dimension(),
        3,
        operator_config.prior_kind,
        operator_config.pde_residual_kind,
        &posterior_problem.linear_measurements,
        posterior_residual_norm,
        &posterior,
        joint_final_step,
        joint_objective_components,
    );
    eprintln!(
        "TEAM 13 joint material UQ: solved joint Laplace in {:.2?} (converged={} residual={:.6e} posterior_nnz={} factor_nnz={})",
        phase_start.elapsed(),
        posterior.converged,
        posterior_residual_norm,
        posterior.assembly.posterior_precision_nnz,
        posterior.final_factorization.nnz
    );

    phase_start = Instant::now();
    let n_state = nominal_model.reduced_dimension();
    let theta_mean = [
        posterior.map[n_state],
        posterior.map[n_state + 1],
        posterior.map[n_state + 2],
    ];
    let observations = team13_published_steel_observations(config.observed_steel_gap);
    let posterior_state_mean = posterior.map[..n_state].to_vec();
    let (material_covariance, material_correlation, steel_patch_reports) =
        team13_joint_material_patch_reports(
            &mut posterior,
            &steel_patch_operator,
            &posterior_state_mean,
            &observations,
            n_state,
        )?;
    let material_posterior = team13_material_parameter_reports(
        config.material_anchor_b_tesla,
        material_prior_std,
        theta_mean,
        material_covariance,
    );
    let bh_curve_bands = team13_bh_curve_bands(
        config.material_anchor_b_tesla,
        theta_mean,
        material_covariance,
    )?;
    eprintln!(
        "TEAM 13 joint material UQ: computed material/patch reports in {:.2?}",
        phase_start.elapsed()
    );

    let result = Team13JointMaterialUqResult {
        domain_mode: operator_config.domain_mode,
        mesh_path: operator_config.mesh_path.clone(),
        observed_steel_gap: config.observed_steel_gap,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: n_state,
        boundary_edge_dofs: nominal_model.boundary_edge_dofs().len(),
        material_anchor_b_tesla: config.material_anchor_b_tesla,
        material_prior_std,
        material_prior_calibration,
        magnitude_smoothing_tesla: config.magnitude_smoothing_tesla,
        deterministic_converged: deterministic.converged,
        deterministic_residual_norm,
        posterior_converged: posterior.converged,
        posterior_residual_norm,
        posterior_precision_nnz: posterior.assembly.posterior_precision_nnz,
        posterior_factor_nnz: posterior.final_factorization.nnz,
        material_posterior,
        material_posterior_covariance: material_covariance,
        material_posterior_correlation: material_correlation,
        bh_curve_bands,
        steel_patch_reports,
        fixed_material_solve,
        joint_material_solve,
        output_dir: config.output_dir.clone(),
    };
    if let Some(output_dir) = &config.output_dir {
        write_team13_joint_material_uq_outputs(output_dir, &result)?;
    }
    eprintln!(
        "TEAM 13 joint material UQ: completed in {:.2?}",
        total_start.elapsed()
    );
    Ok(result)
}

pub fn run_team13_forward_benchmark_diagnostic(
    config: &Team13SyntheticBenchmarkGeometryConfig,
) -> Result<Team13ForwardBenchmarkDiagnosticResult, String> {
    validate_synthetic_benchmark_geometry_config(config)?;
    let total_start = Instant::now();
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 forward diagnostic: reading mesh `{}`",
        config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    eprintln!(
        "TEAM 13 forward diagnostic: read mesh bytes in {:.2?}",
        phase_start.elapsed()
    );
    phase_start = Instant::now();
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 forward benchmark diagnostic requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 forward diagnostic: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    eprintln!(
        "TEAM 13 forward diagnostic: built metric/boundary in {:.2?} (boundary_edge_dofs={})",
        phase_start.elapsed(),
        boundary.state.len()
    );
    phase_start = Instant::now();
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    eprintln!(
        "TEAM 13 forward diagnostic: assembled weighted Galerkin matrices in {:.2?}",
        phase_start.elapsed()
    );
    phase_start = Instant::now();
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &linear_reluctivity,
        ));
    eprintln!(
        "TEAM 13 forward diagnostic: assembled projected sparse inverse in {:.2?}",
        phase_start.elapsed()
    );
    phase_start = Instant::now();
    let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;
    eprintln!(
        "TEAM 13 forward diagnostic: built reduced linear system in {:.2?} (active_dofs={})",
        phase_start.elapsed(),
        linear_system.layout.reduced_dimension()
    );

    phase_start = Instant::now();
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let nonlinear_material = build_team13_synthetic_benchmark_material(config)?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary,
        ),
    )?;
    eprintln!(
        "TEAM 13 forward diagnostic: built source/material/nonlinear models in {:.2?}",
        phase_start.elapsed()
    );
    if linear_source_free.layout().active_dofs != linear_system.layout.active_dofs {
        return Err("TEAM13 forward beta-zero layout does not match linear layout".to_string());
    }
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 forward nonlinear and beta-zero layouts differ".to_string());
    }

    phase_start = Instant::now();
    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    eprintln!(
        "TEAM 13 forward diagnostic: solved beta-zero linear model in {:.2?}",
        phase_start.elapsed()
    );
    phase_start = Instant::now();
    let operators = build_team13_operators(&topology, &coords)?;
    let observations = build_team13_synthetic_benchmark_geometry_observations(
        &topology,
        &coords,
        &operators,
        source_free.layout(),
        &linear_mean,
        &linear_mean,
        config.steel_observation_quadrature,
        config.magnitude_smoothing_tesla,
    )?;
    eprintln!(
        "TEAM 13 forward diagnostic: built observation rows in {:.2?} (total={} assimilated={})",
        phase_start.elapsed(),
        observations.specs.len(),
        observations.assimilated_specs.len()
    );
    phase_start = Instant::now();
    let source_scale_diagnostics = run_team13_source_scale_diagnostics(
        config,
        &linear_source_free,
        &source_free,
        &reduced_source,
        &observations,
    )?;
    eprintln!(
        "TEAM 13 forward diagnostic: completed source-scale diagnostics in {:.2?} (total {:.2?})",
        phase_start.elapsed(),
        total_start.elapsed()
    );

    Ok(Team13ForwardBenchmarkDiagnosticResult {
        domain_mode: config.domain_mode,
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: source_free.reduced_dimension(),
        boundary_edge_dofs: source_free.boundary_edge_dofs().len(),
        material_kind: config.material_kind,
        steel_observation_quadrature: config.steel_observation_quadrature,
        observation_count: observations.specs.len(),
        assimilated_observation_count: observations.assimilated_specs.len(),
        source_scale_diagnostics,
    })
}

pub fn run_team13_same_mesh_linear_parity(
    config: &Team13SameMeshLinearParityConfig,
) -> Result<Team13SameMeshLinearParityResult, String> {
    validate_same_mesh_linear_parity_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 parity mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 same-mesh parity diagnostic requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &linear_reluctivity,
        ));
    let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;

    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let full_source_l2 = l2_norm(nominal_source.as_slice());
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;

    let linear_material = Team13SmoothIronReluctivityLaw::new(NU_AIR, NU_IRON, 0.0, 1.0)?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(linear_material, boundary.clone()),
    )?;
    if linear_source_free.layout().active_dofs != linear_system.layout.active_dofs {
        return Err(format!(
            "TEAM 13 same-mesh layout mismatch: nonlinear beta-zero active dofs {} vs mixed system {}",
            linear_source_free.layout().active_dofs.len(),
            linear_system.layout.active_dofs.len()
        ));
    }

    let linear_mean = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let mut evaluation = linear_source_free.source_free_residual_and_jacobian(&linear_mean)?;
    for (residual, source) in evaluation.residual.iter_mut().zip(reduced_source.iter()) {
        *residual -= *source;
    }
    let operator = feec_csr_to_gmrf(&evaluation.jacobian);
    let state = GmrfVector::from_vec(linear_mean.clone());
    let operator_state = operator.mul_vec(&state);
    let energy = state.dot(&operator_state);
    let source_vector = GmrfVector::from_vec(reduced_source.clone());
    let work = source_vector.dot(&state);

    let operators = build_team13_operators(&topology, &coords)?;
    let observations = build_team13_synthetic_benchmark_geometry_observations(
        &topology,
        &coords,
        &operators,
        linear_source_free.layout(),
        &linear_mean,
        &linear_mean,
        config.steel_observation_quadrature,
        1.0e-8,
    )?;
    let steel_predictions = observations
        .assimilated_model
        .smooth_norm_values(&linear_mean)?;
    let steel_reports = published_steel_reports_from_predictions(
        &observations.assimilated_specs,
        &steel_predictions,
        &steel_predictions,
    )?;
    let steel_rmse_g052 =
        published_steel_rmse(&steel_reports, Team13PublishedSteelGap::G052, false);
    let steel_rmse_g047 =
        published_steel_rmse(&steel_reports, Team13PublishedSteelGap::G047, false);
    let steel_group_summaries_g052 = published_steel_group_summaries(&steel_reports, false);
    let audit = audit_team13_regions(&topology, &coords, config.domain_mode, config.ampere_turns)?;

    let result = Team13SameMeshLinearParityResult {
        domain_mode: config.domain_mode,
        mesh_path: config.mesh_path.clone(),
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: linear_source_free.reduced_dimension(),
        boundary_edge_dofs: linear_source_free.boundary_edge_dofs().len(),
        operator_dimension: evaluation.jacobian.nrows(),
        operator_nnz: evaluation.jacobian.nnz(),
        full_source_l2,
        rhs_l2: l2_norm(&reduced_source),
        solution_l2: l2_norm(&linear_mean),
        linear_residual_l2: l2_norm(&evaluation.residual),
        energy,
        work,
        steel_observation_quadrature: config.steel_observation_quadrature,
        steel_predictions: steel_reports,
        steel_group_summaries_g052,
        steel_rmse_g052,
        steel_rmse_g047,
        audit,
        output_dir: config.output_dir.clone(),
    };

    if let Some(output_dir) = &config.output_dir {
        write_team13_same_mesh_linear_parity_outputs(output_dir, &result)?;
    }

    Ok(result)
}

pub fn run_team13_deterministic_benchmark(
    config: &Team13DeterministicBenchmarkConfig,
) -> Result<Team13DeterministicBenchmarkResult, String> {
    let linear = run_team13_same_mesh_linear_parity(&config.linear)?;
    let ngsolve_linear_predictions =
        read_team13_ngsolve_steel_predictions(&config.ngsolve_linear_reference_dir)?;
    let linear_comparison = compare_team13_steel_predictions_to_ngsolve(
        &ngsolve_linear_predictions,
        &linear.steel_predictions,
        false,
    )?;
    if let Some(output_dir) = &linear.output_dir {
        write_team13_steel_ngsolve_comparison_output(output_dir, &linear_comparison)?;
    }

    let (nonlinear, nonlinear_comparison) = if let Some(nonlinear_config) = &config.nonlinear {
        let nonlinear = run_team13_nonlinear_forward_parity(nonlinear_config)?;
        let comparison = match &config.ngsolve_nonlinear_reference_dir {
            Some(reference_dir) => {
                let ngsolve_nonlinear_predictions =
                    read_team13_ngsolve_steel_predictions(reference_dir)?;
                let comparison = compare_team13_steel_predictions_to_ngsolve(
                    &ngsolve_nonlinear_predictions,
                    &nonlinear.steel_predictions,
                    false,
                )?;
                if let Some(output_dir) = &nonlinear.output_dir {
                    write_team13_steel_ngsolve_comparison_output(output_dir, &comparison)?;
                }
                Some(comparison)
            }
            None => None,
        };
        (Some(nonlinear), comparison)
    } else {
        (None, None)
    };

    Ok(Team13DeterministicBenchmarkResult {
        linear,
        linear_comparison,
        nonlinear,
        nonlinear_comparison,
    })
}

pub fn run_team13_nonlinear_forward_parity(
    config: &Team13NonlinearForwardParityConfig,
) -> Result<Team13NonlinearForwardParityResult, String> {
    validate_nonlinear_forward_parity_config(config)?;
    let total_start = Instant::now();
    let mut phase_start = Instant::now();
    eprintln!(
        "TEAM 13 nonlinear forward parity: reading mesh `{}`",
        config.mesh_path.display()
    );
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read TEAM 13 nonlinear parity mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    eprintln!(
        "TEAM 13 nonlinear forward parity: read mesh bytes in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 nonlinear forward parity requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }
    eprintln!(
        "TEAM 13 nonlinear forward parity: parsed mesh in {:.2?} (vertices={} edges={} cells={})",
        phase_start.elapsed(),
        topology.nsimplices(0),
        topology.nsimplices(1),
        topology.nsimplices(3)
    );

    phase_start = Instant::now();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    eprintln!(
        "TEAM 13 nonlinear forward parity: built metric/boundary in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let linear_reluctivity = reluctivity_weight();
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &linear_reluctivity);
    let reduced_source = reduce_team13_physical_source_rhs(&galmats, &boundary, &nominal_source)?;
    let rhs_l2 = l2_norm(&reduced_source);
    eprintln!(
        "TEAM 13 nonlinear forward parity: assembled source and reduction data in {:.2?}",
        phase_start.elapsed()
    );

    phase_start = Instant::now();
    let nonlinear_material = build_team13_material_from_kind(
        config.material_kind,
        config.beta_iron,
        config.b_scale_tesla,
    )?;
    let linear_source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.linear.clone(),
            boundary.clone(),
        ),
    )?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::with_shared_material(
            nonlinear_material.nonlinear.clone(),
            boundary.clone(),
        ),
    )?;
    if source_free.layout().active_dofs != linear_source_free.layout().active_dofs {
        return Err("TEAM13 nonlinear parity nonlinear and beta-zero layouts differ".to_string());
    }
    eprintln!(
        "TEAM 13 nonlinear forward parity: built beta-zero/nonlinear models in {:.2?} (active_dofs={})",
        phase_start.elapsed(),
        source_free.reduced_dimension()
    );

    phase_start = Instant::now();
    let initial = solve_source_free_linear_model(&linear_source_free, &reduced_source)?;
    let model = source_free.clone().with_source(reduced_source)?;
    let initial_evaluation = model.residual_and_jacobian(&initial)?;
    let initial_residual_l2 = l2_norm(&initial_evaluation.residual);
    eprintln!(
        "TEAM 13 nonlinear forward parity: solved beta-zero initial state in {:.2?} (initial_residual={:.6e})",
        phase_start.elapsed(),
        initial_residual_l2
    );

    phase_start = Instant::now();
    let solve = solve_team13_feec_forward_newton(
        &model,
        initial.clone(),
        config.max_iterations,
        config.linear_solve,
    )?;
    let final_evaluation = model.residual_and_jacobian(&solve.solution)?;
    let final_residual_l2 = l2_norm(&final_evaluation.residual);
    eprintln!(
        "TEAM 13 nonlinear forward parity: nonlinear solve completed in {:.2?} (converged={} iterations={} final_residual={:.6e})",
        phase_start.elapsed(),
        solve.converged,
        solve.history.len(),
        final_residual_l2
    );

    phase_start = Instant::now();
    let operators = build_team13_operators(&topology, &coords)?;
    let observations = build_team13_synthetic_benchmark_geometry_observations(
        &topology,
        &coords,
        &operators,
        model.layout(),
        &initial,
        &solve.solution,
        config.steel_observation_quadrature,
        config.magnitude_smoothing_tesla,
    )?;
    let initial_predictions = observations
        .assimilated_model
        .smooth_norm_values(&initial)?;
    let nonlinear_predictions = observations
        .assimilated_model
        .smooth_norm_values(&solve.solution)?;
    let steel_predictions = published_steel_reports_from_predictions(
        &observations.assimilated_specs,
        &initial_predictions,
        &nonlinear_predictions,
    )?;
    eprintln!(
        "TEAM 13 nonlinear forward parity: evaluated steel observations in {:.2?} (steel={})",
        phase_start.elapsed(),
        observations.assimilated_specs.len()
    );

    phase_start = Instant::now();
    let audit = audit_team13_regions(&topology, &coords, config.domain_mode, config.ampere_turns)?;
    let last_iteration = solve.history.last();
    let result = Team13NonlinearForwardParityResult {
        domain_mode: config.domain_mode,
        mesh_path: config.mesh_path.clone(),
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        cells: topology.nsimplices(3),
        active_dofs: model.reduced_dimension(),
        boundary_edge_dofs: model.boundary_edge_dofs().len(),
        material_kind: config.material_kind,
        ampere_turns: config.ampere_turns,
        residual_dimension: model.residual_dimension(),
        initial_jacobian_nnz: initial_evaluation.jacobian.nnz(),
        final_jacobian_nnz: final_evaluation.jacobian.nnz(),
        rhs_l2,
        initial_solution_l2: l2_norm(&initial),
        nonlinear_solution_l2: l2_norm(&solve.solution),
        initial_residual_l2,
        final_residual_l2,
        converged: solve.converged,
        iterations: solve.history.len(),
        final_step_norm: last_iteration.map_or(f64::NAN, |iteration| iteration.step_norm),
        final_step_alpha: last_iteration.map_or(f64::NAN, |iteration| iteration.alpha),
        steel_observation_quadrature: config.steel_observation_quadrature,
        initial_steel_rmse_g052: published_steel_rmse(
            &steel_predictions,
            Team13PublishedSteelGap::G052,
            true,
        ),
        initial_steel_rmse_g047: published_steel_rmse(
            &steel_predictions,
            Team13PublishedSteelGap::G047,
            true,
        ),
        nonlinear_steel_rmse_g052: published_steel_rmse(
            &steel_predictions,
            Team13PublishedSteelGap::G052,
            false,
        ),
        nonlinear_steel_rmse_g047: published_steel_rmse(
            &steel_predictions,
            Team13PublishedSteelGap::G047,
            false,
        ),
        initial_steel_group_summaries: published_steel_group_summaries(&steel_predictions, true),
        nonlinear_steel_group_summaries: published_steel_group_summaries(&steel_predictions, false),
        steel_predictions,
        audit,
        output_dir: config.output_dir.clone(),
    };

    if let Some(output_dir) = &config.output_dir {
        write_team13_nonlinear_forward_parity_outputs(output_dir, &result)?;
    }
    eprintln!(
        "TEAM 13 nonlinear forward parity: wrote diagnostics in {:.2?} (total {:.2?})",
        phase_start.elapsed(),
        total_start.elapsed()
    );

    Ok(result)
}

fn validate_same_mesh_linear_parity_config(
    config: &Team13SameMeshLinearParityConfig,
) -> Result<(), String> {
    if !config.ampere_turns.is_finite() || config.ampere_turns <= 0.0 {
        return Err("ampere_turns must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_nonlinear_forward_parity_config(
    config: &Team13NonlinearForwardParityConfig,
) -> Result<(), String> {
    if !config.ampere_turns.is_finite() || config.ampere_turns <= 0.0 {
        return Err("ampere_turns must be finite and positive".to_string());
    }
    if !config.beta_iron.is_finite() || config.beta_iron < 0.0 {
        return Err("beta_iron must be finite and nonnegative".to_string());
    }
    if !config.b_scale_tesla.is_finite() || config.b_scale_tesla <= 0.0 {
        return Err("b_scale_tesla must be finite and positive".to_string());
    }
    if !config.magnitude_smoothing_tesla.is_finite() || config.magnitude_smoothing_tesla <= 0.0 {
        return Err("magnitude_smoothing_tesla must be finite and positive".to_string());
    }
    if config.max_iterations == 0 {
        return Err("max_iterations must be at least one".to_string());
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct Team13AuditAccumulator {
    cell_count: usize,
    volume: f64,
    current_integral: [f64; 3],
    current_l2_sq: f64,
}

fn audit_team13_regions(
    topology: &Complex,
    coords: &MeshCoords,
    mode: Team13DomainMode,
    ampere_turns: f64,
) -> Result<Team13RegionAudit, String> {
    let mut entries = BTreeMap::<String, Team13AuditAccumulator>::new();
    for name in [
        "all",
        "iron",
        "air",
        "source_free_air",
        "coil_total",
        "unclassified",
    ] {
        entries.entry(name.to_string()).or_default();
    }
    for region in Team13CoilRegion::all() {
        entries.entry(region.name().to_string()).or_default();
    }

    let mut iron_and_coil_cells = 0usize;
    let mut multiple_coil_cells = 0usize;
    let unclassified_cells = 0usize;
    for cell in topology.skeleton(topology.dim()).handle_iter() {
        let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
        let volume = cell_coords.vol().abs();
        let bary = cell_coords.barycenter();
        let point = FeecVector::from_vec(vec![bary[0], bary[1], bary[2]]);
        let current = team13_current_vector(point.as_view(), mode, ampere_turns, None);
        let iron = is_iron_point(point.as_view());
        let coil_regions = coil_regions_at(point.as_view(), mode);

        add_audit_cell(&mut entries, "all", volume, current);
        if iron {
            add_audit_cell(&mut entries, "iron", volume, current);
        } else {
            add_audit_cell(&mut entries, "air", volume, current);
        }

        if coil_regions.is_empty() {
            if !iron {
                add_audit_cell(&mut entries, "source_free_air", volume, current);
            }
        } else {
            add_audit_cell(&mut entries, "coil_total", volume, current);
            if iron {
                iron_and_coil_cells += 1;
            }
            if coil_regions.len() > 1 {
                multiple_coil_cells += 1;
            }
            for region in coil_regions {
                add_audit_cell(&mut entries, region.name(), volume, current);
            }
        }
    }

    let total_volume = entries.get("all").map_or(0.0, |entry| entry.volume);
    let names = [
        "all",
        "iron",
        "air",
        "source_free_air",
        "coil_total",
        "unclassified",
        "brick_back",
        "brick_front",
        "brick_left",
        "brick_right",
        "corner_right_back",
        "corner_left_back",
        "corner_left_front",
        "corner_right_front",
    ];
    let audit_entries = names
        .into_iter()
        .map(|name| {
            let accumulator = entries.remove(name).unwrap_or_default();
            Team13RegionAuditEntry {
                name: name.to_string(),
                cell_count: accumulator.cell_count,
                volume: accumulator.volume,
                volume_fraction: safe_ratio(accumulator.volume, total_volume),
                current_integral: accumulator.current_integral,
                current_l2_norm: accumulator.current_l2_sq.sqrt(),
            }
        })
        .collect();

    Ok(Team13RegionAudit {
        entries: audit_entries,
        total_volume,
        iron_and_coil_cells,
        multiple_coil_cells,
        unclassified_cells,
    })
}

fn coil_regions_at(point: CoordRef<'_>, mode: Team13DomainMode) -> Vec<Team13CoilRegion> {
    Team13CoilRegion::all()
        .into_iter()
        .filter(|region| point_in_coil_region(point, mode, *region))
        .collect()
}

fn add_audit_cell(
    entries: &mut BTreeMap<String, Team13AuditAccumulator>,
    name: &str,
    volume: f64,
    current: [f64; 3],
) {
    let entry = entries.entry(name.to_string()).or_default();
    entry.cell_count += 1;
    entry.volume += volume;
    for axis in 0..3 {
        entry.current_integral[axis] += volume * current[axis];
    }
    entry.current_l2_sq +=
        volume * (current[0] * current[0] + current[1] * current[1] + current[2] * current[2]);
}

fn write_team13_same_mesh_linear_parity_outputs(
    output_dir: &Path,
    result: &Team13SameMeshLinearParityResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 parity output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    fs::write(
        output_dir.join("linear_parity_diagnostic.json"),
        team13_same_mesh_linear_parity_json(result),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 parity JSON `{}`: {err}",
            output_dir.join("linear_parity_diagnostic.json").display()
        )
    })?;
    fs::write(
        output_dir.join("linear_parity_summary.csv"),
        team13_same_mesh_linear_parity_summary_csv(result),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 parity CSV `{}`: {err}",
            output_dir.join("linear_parity_summary.csv").display()
        )
    })?;
    fs::write(
        output_dir.join("region_audit.csv"),
        team13_region_audit_csv(&result.audit),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 region audit CSV `{}`: {err}",
            output_dir.join("region_audit.csv").display()
        )
    })?;
    fs::write(
        output_dir.join("steel_predictions.csv"),
        team13_steel_predictions_csv(&result.steel_predictions),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 steel prediction CSV `{}`: {err}",
            output_dir.join("steel_predictions.csv").display()
        )
    })?;
    Ok(())
}

fn write_team13_nonlinear_forward_parity_outputs(
    output_dir: &Path,
    result: &Team13NonlinearForwardParityResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 nonlinear parity output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    fs::write(
        output_dir.join("nonlinear_forward_parity.json"),
        team13_nonlinear_forward_parity_json(result),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 nonlinear parity JSON `{}`: {err}",
            output_dir.join("nonlinear_forward_parity.json").display()
        )
    })?;
    fs::write(
        output_dir.join("nonlinear_forward_summary.csv"),
        team13_nonlinear_forward_parity_summary_csv(result),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 nonlinear parity CSV `{}`: {err}",
            output_dir.join("nonlinear_forward_summary.csv").display()
        )
    })?;
    fs::write(
        output_dir.join("region_audit.csv"),
        team13_region_audit_csv(&result.audit),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 nonlinear parity region audit CSV `{}`: {err}",
            output_dir.join("region_audit.csv").display()
        )
    })?;
    fs::write(
        output_dir.join("steel_predictions.csv"),
        team13_nonlinear_steel_predictions_csv(&result.steel_predictions),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 nonlinear parity steel CSV `{}`: {err}",
            output_dir.join("steel_predictions.csv").display()
        )
    })?;
    Ok(())
}

fn write_team13_steel_ngsolve_comparison_output(
    output_dir: &Path,
    reports: &[Team13SteelNgsolveComparisonReport],
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 NGSolve comparison output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    fs::write(
        output_dir.join("steel_ngsolve_comparison.csv"),
        team13_steel_ngsolve_comparison_csv(reports),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 NGSolve comparison CSV `{}`: {err}",
            output_dir.join("steel_ngsolve_comparison.csv").display()
        )
    })?;
    Ok(())
}

fn read_team13_ngsolve_steel_predictions(
    reference_dir: &Path,
) -> Result<Vec<Team13NgsolveSteelPrediction>, String> {
    let path = reference_dir.join("steel_predictions.csv");
    let contents = fs::read_to_string(&path).map_err(|err| {
        format!(
            "failed to read TEAM 13 NGSolve steel predictions `{}`: {err}",
            path.display()
        )
    })?;
    parse_team13_ngsolve_steel_predictions_csv(&path, &contents)
}

fn parse_team13_ngsolve_steel_predictions_csv(
    path: &Path,
    contents: &str,
) -> Result<Vec<Team13NgsolveSteelPrediction>, String> {
    let mut lines = contents.lines();
    let Some(header) = lines.next() else {
        return Err(format!(
            "TEAM 13 NGSolve steel predictions `{}` is empty",
            path.display()
        ));
    };
    let expected_header =
        "name,group,prediction,observed_g052,observed_g047,residual_g052,residual_g047";
    if header.trim_end_matches('\r') != expected_header {
        return Err(format!(
            "TEAM 13 NGSolve steel predictions `{}` has unexpected header `{}`",
            path.display(),
            header
        ));
    }

    let mut predictions = Vec::with_capacity(TEAM13_OBSERVATION_COUNT);
    for (line_index, raw_line) in lines.enumerate() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err(format!(
                "TEAM 13 NGSolve steel predictions `{}` line {} has {} columns, expected 7",
                path.display(),
                line_index + 2,
                fields.len()
            ));
        }
        let prediction = parse_team13_csv_f64(path, line_index + 2, "prediction", fields[2])?;
        let observed_g_052 =
            parse_team13_csv_f64(path, line_index + 2, "observed_g052", fields[3])?;
        let observed_g_047 =
            parse_team13_csv_f64(path, line_index + 2, "observed_g047", fields[4])?;
        predictions.push(Team13NgsolveSteelPrediction {
            name: fields[0].to_string(),
            group: Team13SteelSurfaceGroup::from_str(fields[1])?,
            prediction,
            observed_g_052,
            observed_g_047,
        });
    }
    if predictions.len() != TEAM13_OBSERVATION_COUNT {
        return Err(format!(
            "TEAM 13 NGSolve steel predictions `{}` has {} rows, expected {TEAM13_OBSERVATION_COUNT}",
            path.display(),
            predictions.len()
        ));
    }
    Ok(predictions)
}

fn parse_team13_csv_f64(path: &Path, line: usize, column: &str, raw: &str) -> Result<f64, String> {
    raw.parse::<f64>().map_err(|err| {
        format!(
            "failed to parse TEAM 13 NGSolve CSV `{}` line {line} column `{column}` value `{raw}`: {err}",
            path.display()
        )
    })
}

fn compare_team13_steel_predictions_to_ngsolve(
    ngsolve: &[Team13NgsolveSteelPrediction],
    feec: &[Team13PublishedSteelBenchmarkReport],
    use_nominal_prediction: bool,
) -> Result<Vec<Team13SteelNgsolveComparisonReport>, String> {
    if ngsolve.len() != TEAM13_OBSERVATION_COUNT || feec.len() != TEAM13_OBSERVATION_COUNT {
        return Err(format!(
            "TEAM 13 NGSolve comparison requires {TEAM13_OBSERVATION_COUNT} rows, got {} NGSolve rows and {} FEEC rows",
            ngsolve.len(),
            feec.len()
        ));
    }
    let g_052 = team13_published_steel_observations(Team13PublishedSteelGap::G052);
    let g_047 = team13_published_steel_observations(Team13PublishedSteelGap::G047);
    let mut reports = Vec::with_capacity(TEAM13_OBSERVATION_COUNT);
    for index in 0..TEAM13_OBSERVATION_COUNT {
        let reference = &ngsolve[index];
        let report = &feec[index];
        let expected_name = team13_ngsolve_measurement_name(index);
        if reference.name != expected_name {
            return Err(format!(
                "TEAM 13 NGSolve reference row {} is named `{}`, expected `{expected_name}`",
                index + 1,
                reference.name
            ));
        }
        if reference.group != report.group {
            return Err(format!(
                "TEAM 13 steel group mismatch for `{}`: NGSolve `{}` vs FEEC `{}`",
                reference.name,
                reference.group.as_str(),
                report.group.as_str()
            ));
        }
        if (reference.observed_g_052 - g_052[index]).abs() > 1.0e-12
            || (report.observed_g_052 - g_052[index]).abs() > 1.0e-12
            || (reference.observed_g_047 - g_047[index]).abs() > 1.0e-12
            || (report.observed_g_047 - g_047[index]).abs() > 1.0e-12
        {
            return Err(format!(
                "TEAM 13 published steel value mismatch for `{}`",
                reference.name
            ));
        }
        let feec_prediction = if use_nominal_prediction {
            report.nominal_prediction
        } else {
            report.posterior_prediction
        };
        reports.push(Team13SteelNgsolveComparisonReport {
            name: reference.name.clone(),
            group: reference.group,
            ngsolve_prediction: reference.prediction,
            feec_prediction,
            observed_g_052: g_052[index],
            observed_g_047: g_047[index],
            feec_minus_ngsolve: feec_prediction - reference.prediction,
            feec_residual_g_052: feec_prediction - g_052[index],
            feec_residual_g_047: feec_prediction - g_047[index],
        });
    }
    Ok(reports)
}

pub fn team13_steel_ngsolve_comparison_rmse(reports: &[Team13SteelNgsolveComparisonReport]) -> f64 {
    if reports.is_empty() {
        return f64::NAN;
    }
    (reports
        .iter()
        .map(|report| report.feec_minus_ngsolve * report.feec_minus_ngsolve)
        .sum::<f64>()
        / reports.len() as f64)
        .sqrt()
}

fn team13_same_mesh_linear_parity_json(result: &Team13SameMeshLinearParityResult) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"solver\": \"feec\",\n");
    json.push_str(&format!(
        "  \"mesh_path\": \"{}\",\n",
        json_escape(&result.mesh_path.display().to_string())
    ));
    json.push_str(&format!(
        "  \"domain\": \"{}\",\n",
        result.domain_mode.as_str()
    ));
    json.push_str(&format!(
        "  \"vertices\": {},\n  \"edges\": {},\n  \"cells\": {},\n",
        result.vertices, result.edges, result.cells
    ));
    json.push_str(&format!(
        "  \"active_dofs\": {},\n  \"boundary_edge_dofs\": {},\n",
        result.active_dofs, result.boundary_edge_dofs
    ));
    json.push_str(&format!(
        "  \"operator_dimension\": {},\n  \"operator_nnz\": {},\n",
        result.operator_dimension, result.operator_nnz
    ));
    json.push_str(&format!(
        "  \"full_source_l2\": {:.16e},\n  \"rhs_l2\": {:.16e},\n  \"solution_l2\": {:.16e},\n  \"linear_residual_l2\": {:.16e},\n",
        result.full_source_l2, result.rhs_l2, result.solution_l2, result.linear_residual_l2
    ));
    json.push_str(&format!(
        "  \"energy\": {:.16e},\n  \"work\": {:.16e},\n",
        result.energy, result.work
    ));
    json.push_str(&format!(
        "  \"steel_rmse_g052\": {:.16e},\n  \"steel_rmse_g047\": {:.16e},\n",
        result.steel_rmse_g052, result.steel_rmse_g047
    ));
    json.push_str(&format!(
        "  \"steel_observation_quadrature\": \"{}\",\n",
        result.steel_observation_quadrature.as_str()
    ));
    json.push_str(&format!(
        "  \"iron_volume\": {:.16e},\n  \"coil_total_volume\": {:.16e},\n",
        audit_volume(&result.audit, "iron"),
        audit_volume(&result.audit, "coil_total")
    ));
    json.push_str(&format!(
        "  \"iron_and_coil_cells\": {},\n  \"multiple_coil_cells\": {},\n  \"unclassified_cells\": {},\n",
        result.audit.iron_and_coil_cells,
        result.audit.multiple_coil_cells,
        result.audit.unclassified_cells
    ));
    json.push_str("  \"region_audit\": [\n");
    for (index, entry) in result.audit.entries.iter().enumerate() {
        json.push_str(&format!(
            concat!(
                "    {{\"name\": \"{}\", \"cell_count\": {}, \"volume\": {:.16e}, ",
                "\"volume_fraction\": {:.16e}, \"current_integral\": [{:.16e}, {:.16e}, {:.16e}], ",
                "\"current_l2_norm\": {:.16e}}}{} \n"
            ),
            json_escape(&entry.name),
            entry.cell_count,
            entry.volume,
            entry.volume_fraction,
            entry.current_integral[0],
            entry.current_integral[1],
            entry.current_integral[2],
            entry.current_l2_norm,
            if index + 1 == result.audit.entries.len() {
                ""
            } else {
                ","
            }
        ));
    }
    json.push_str("  ]\n");
    json.push_str("}\n");
    json
}

fn team13_nonlinear_forward_parity_json(result: &Team13NonlinearForwardParityResult) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"solver\": \"feec\",\n");
    json.push_str(&format!(
        "  \"mesh_path\": \"{}\",\n",
        json_escape(&result.mesh_path.display().to_string())
    ));
    json.push_str(&format!(
        "  \"domain\": \"{}\",\n",
        result.domain_mode.as_str()
    ));
    json.push_str(&format!(
        "  \"material_kind\": \"{}\",\n  \"ampere_turns\": {:.16e},\n",
        result.material_kind.as_str(),
        result.ampere_turns
    ));
    json.push_str(&format!(
        "  \"vertices\": {},\n  \"edges\": {},\n  \"cells\": {},\n",
        result.vertices, result.edges, result.cells
    ));
    json.push_str(&format!(
        "  \"active_dofs\": {},\n  \"boundary_edge_dofs\": {},\n",
        result.active_dofs, result.boundary_edge_dofs
    ));
    json.push_str(&format!(
        "  \"residual_dimension\": {},\n  \"initial_jacobian_nnz\": {},\n  \"final_jacobian_nnz\": {},\n",
        result.residual_dimension, result.initial_jacobian_nnz, result.final_jacobian_nnz
    ));
    json.push_str(&format!(
        "  \"rhs_l2\": {:.16e},\n  \"initial_solution_l2\": {:.16e},\n  \"nonlinear_solution_l2\": {:.16e},\n",
        result.rhs_l2, result.initial_solution_l2, result.nonlinear_solution_l2
    ));
    json.push_str(&format!(
        "  \"initial_residual_l2\": {:.16e},\n  \"final_residual_l2\": {:.16e},\n",
        result.initial_residual_l2, result.final_residual_l2
    ));
    json.push_str(&format!(
        "  \"converged\": {},\n  \"iterations\": {},\n  \"final_step_norm\": {:.16e},\n  \"final_step_alpha\": {:.16e},\n",
        result.converged,
        result.iterations,
        result.final_step_norm,
        result.final_step_alpha
    ));
    json.push_str(&format!(
        "  \"steel_observation_quadrature\": \"{}\",\n",
        result.steel_observation_quadrature.as_str()
    ));
    json.push_str(&format!(
        "  \"initial_steel_rmse_g052\": {:.16e},\n  \"initial_steel_rmse_g047\": {:.16e},\n",
        result.initial_steel_rmse_g052, result.initial_steel_rmse_g047
    ));
    json.push_str(&format!(
        "  \"nonlinear_steel_rmse_g052\": {:.16e},\n  \"nonlinear_steel_rmse_g047\": {:.16e},\n",
        result.nonlinear_steel_rmse_g052, result.nonlinear_steel_rmse_g047
    ));
    json.push_str(&format!(
        "  \"iron_volume\": {:.16e},\n  \"coil_total_volume\": {:.16e},\n",
        audit_volume(&result.audit, "iron"),
        audit_volume(&result.audit, "coil_total")
    ));
    json.push_str(&format!(
        "  \"iron_and_coil_cells\": {},\n  \"multiple_coil_cells\": {},\n  \"unclassified_cells\": {},\n",
        result.audit.iron_and_coil_cells,
        result.audit.multiple_coil_cells,
        result.audit.unclassified_cells
    ));
    json.push_str("  \"initial_steel_groups\": [\n");
    for (index, summary) in result.initial_steel_group_summaries.iter().enumerate() {
        json.push_str(&team13_published_steel_group_summary_json(
            summary,
            index + 1 == result.initial_steel_group_summaries.len(),
        ));
    }
    json.push_str("  ],\n");
    json.push_str("  \"nonlinear_steel_groups\": [\n");
    for (index, summary) in result.nonlinear_steel_group_summaries.iter().enumerate() {
        json.push_str(&team13_published_steel_group_summary_json(
            summary,
            index + 1 == result.nonlinear_steel_group_summaries.len(),
        ));
    }
    json.push_str("  ],\n");
    json.push_str("  \"region_audit\": [\n");
    for (index, entry) in result.audit.entries.iter().enumerate() {
        json.push_str(&format!(
            concat!(
                "    {{\"name\": \"{}\", \"cell_count\": {}, \"volume\": {:.16e}, ",
                "\"volume_fraction\": {:.16e}, \"current_integral\": [{:.16e}, {:.16e}, {:.16e}], ",
                "\"current_l2_norm\": {:.16e}}}{} \n"
            ),
            json_escape(&entry.name),
            entry.cell_count,
            entry.volume,
            entry.volume_fraction,
            entry.current_integral[0],
            entry.current_integral[1],
            entry.current_integral[2],
            entry.current_l2_norm,
            if index + 1 == result.audit.entries.len() {
                ""
            } else {
                ","
            }
        ));
    }
    json.push_str("  ]\n");
    json.push_str("}\n");
    json
}

fn team13_published_steel_group_summary_json(
    summary: &Team13PublishedSteelGroupSummary,
    last: bool,
) -> String {
    format!(
        concat!(
            "    {{\"group\": \"{}\", \"count\": {}, \"rmse_g052\": {:.16e}, ",
            "\"rmse_g047\": {:.16e}, \"max_abs_residual_g052\": {:.16e}, ",
            "\"max_abs_residual_g047\": {:.16e}}}{} \n"
        ),
        summary.group.as_str(),
        summary.count,
        summary.rmse_g_052,
        summary.rmse_g_047,
        summary.max_abs_residual_g_052,
        summary.max_abs_residual_g_047,
        if last { "" } else { "," }
    )
}

fn team13_same_mesh_linear_parity_summary_csv(result: &Team13SameMeshLinearParityResult) -> String {
    let coil_names = [
        "brick_back",
        "brick_front",
        "brick_left",
        "brick_right",
        "corner_right_back",
        "corner_left_back",
        "corner_left_front",
        "corner_right_front",
    ];
    let mut header = String::from(
        "solver,mesh_path,domain,vertices,edges,cells,active_dofs,boundary_edge_dofs,operator_dimension,operator_nnz,iron_volume,coil_total_volume,rhs_l2,solution_l2,linear_residual_l2,energy,work,steel_rmse_g052,steel_rmse_g047",
    );
    for name in coil_names {
        header.push_str(&format!(",{}_volume,{}_current_l2_norm", name, name));
    }
    header.push('\n');

    let mut row = format!(
        "feec,{},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
        csv_field(&result.mesh_path.display().to_string()),
        result.domain_mode.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs,
        result.operator_dimension,
        result.operator_nnz,
        audit_volume(&result.audit, "iron"),
        audit_volume(&result.audit, "coil_total"),
        result.rhs_l2,
        result.solution_l2,
        result.linear_residual_l2,
        result.energy,
        result.work,
        result.steel_rmse_g052,
        result.steel_rmse_g047,
    );
    for name in coil_names {
        row.push_str(&format!(
            ",{:.16e},{:.16e}",
            audit_volume(&result.audit, name),
            audit_current_l2(&result.audit, name)
        ));
    }
    row.push('\n');
    header + &row
}

fn team13_nonlinear_forward_parity_summary_csv(
    result: &Team13NonlinearForwardParityResult,
) -> String {
    let coil_names = [
        "brick_back",
        "brick_front",
        "brick_left",
        "brick_right",
        "corner_right_back",
        "corner_left_back",
        "corner_left_front",
        "corner_right_front",
    ];
    let mut header = String::from(
        "solver,mesh_path,domain,material_kind,ampere_turns,vertices,edges,cells,active_dofs,boundary_edge_dofs,residual_dimension,initial_jacobian_nnz,final_jacobian_nnz,iron_volume,coil_total_volume,rhs_l2,initial_solution_l2,nonlinear_solution_l2,initial_residual_l2,final_residual_l2,converged,iterations,final_step_norm,final_step_alpha,initial_steel_rmse_g052,initial_steel_rmse_g047,nonlinear_steel_rmse_g052,nonlinear_steel_rmse_g047",
    );
    for name in coil_names {
        header.push_str(&format!(",{}_volume,{}_current_l2_norm", name, name));
    }
    header.push('\n');

    let mut row = format!(
        "feec,{},{},{},{:.16e},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
        csv_field(&result.mesh_path.display().to_string()),
        result.domain_mode.as_str(),
        result.material_kind.as_str(),
        result.ampere_turns,
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs,
        result.residual_dimension,
        result.initial_jacobian_nnz,
        result.final_jacobian_nnz,
        audit_volume(&result.audit, "iron"),
        audit_volume(&result.audit, "coil_total"),
        result.rhs_l2,
        result.initial_solution_l2,
        result.nonlinear_solution_l2,
        result.initial_residual_l2,
        result.final_residual_l2,
        result.converged,
        result.iterations,
        result.final_step_norm,
        result.final_step_alpha,
        result.initial_steel_rmse_g052,
        result.initial_steel_rmse_g047,
        result.nonlinear_steel_rmse_g052,
        result.nonlinear_steel_rmse_g047,
    );
    for name in coil_names {
        row.push_str(&format!(
            ",{:.16e},{:.16e}",
            audit_volume(&result.audit, name),
            audit_current_l2(&result.audit, name)
        ));
    }
    row.push('\n');
    header + &row
}

fn team13_region_audit_csv(audit: &Team13RegionAudit) -> String {
    let mut csv = "name,cell_count,volume,volume_fraction,current_integral_x,current_integral_y,current_integral_z,current_l2_norm\n".to_string();
    for entry in &audit.entries {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            entry.name,
            entry.cell_count,
            entry.volume,
            entry.volume_fraction,
            entry.current_integral[0],
            entry.current_integral[1],
            entry.current_integral[2],
            entry.current_l2_norm
        ));
    }
    csv
}

fn team13_steel_predictions_csv(reports: &[Team13PublishedSteelBenchmarkReport]) -> String {
    let mut csv = "name,group,prediction,observed_g052,observed_g047,residual_g052,residual_g047\n"
        .to_string();
    for report in reports {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.group.as_str(),
            report.posterior_prediction,
            report.observed_g_052,
            report.observed_g_047,
            report.posterior_prediction - report.observed_g_052,
            report.posterior_prediction - report.observed_g_047
        ));
    }
    csv
}

fn team13_steel_ngsolve_comparison_csv(reports: &[Team13SteelNgsolveComparisonReport]) -> String {
    let mut csv = "name,group,ngsolve_prediction,feec_prediction,observed_g052,observed_g047,feec_minus_ngsolve,feec_residual_g052,feec_residual_g047\n"
        .to_string();
    for report in reports {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            csv_field(&report.name),
            report.group.as_str(),
            report.ngsolve_prediction,
            report.feec_prediction,
            report.observed_g_052,
            report.observed_g_047,
            report.feec_minus_ngsolve,
            report.feec_residual_g_052,
            report.feec_residual_g_047
        ));
    }
    csv
}

fn team13_nonlinear_steel_predictions_csv(
    reports: &[Team13PublishedSteelBenchmarkReport],
) -> String {
    let mut csv = "name,group,initial_prediction,nonlinear_prediction,observed_g052,observed_g047,initial_residual_g052,initial_residual_g047,nonlinear_residual_g052,nonlinear_residual_g047\n"
        .to_string();
    for report in reports {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.group.as_str(),
            report.nominal_prediction,
            report.posterior_prediction,
            report.observed_g_052,
            report.observed_g_047,
            report.nominal_prediction - report.observed_g_052,
            report.nominal_prediction - report.observed_g_047,
            report.posterior_prediction - report.observed_g_052,
            report.posterior_prediction - report.observed_g_047
        ));
    }
    csv
}

fn team13_map_parity_runs_csv(result: &Team13MapParityResult) -> String {
    let mut csv = "label,prior_kind,pde_residual_kind,step_regularization,steel_observation_mode,truth_cache_hit,pde_variance,observation_std_tesla,total_residual_rows,steel_observation_count,posterior_converged,initial_relative_error,posterior_relative_error,relative_error_ratio,initial_residual_norm,truth_residual_norm,posterior_residual_norm,initial_steel_rmse,posterior_steel_rmse,steel_rmse_ratio,posterior_steel_relative_rmse,posterior_steel_max_abs_residual,all_finite_variances,nonnegative_variances,b_quantity_variances_finite,b_quantity_variances_nonnegative,latent_variance_count,latent_variances_finite,latent_variances_nonnegative,step_solve_attempts,accepted_iterations,line_search_residual_evaluations,final_factorizations,metis_cache_hits,metis_cache_misses,cholesky_factor_attempts,cholesky_factor_successes,cholesky_unshifted_attempts,cholesky_symmetrized_attempts,cholesky_cached_shift_attempts,cholesky_cached_shift_successes,cholesky_shifted_attempts,cholesky_shifted_successes,cholesky_max_shift,cholesky_factorization_seconds,prior_precision_nnz,residual_terms_operator_nnz,residual_terms_update_nnz,posterior_precision_nnz,factor_nnz,fill_ratio_vs_lower\n".to_string();
    for run in std::iter::once(&result.default_run).chain(result.sweep_runs.iter()) {
        let fields = vec![
            csv_field(&run.label),
            result.prior_kind.as_str().to_string(),
            result.pde_residual_kind.as_str().to_string(),
            result.step_regularization.as_str().to_string(),
            result.steel_observation_quadrature.as_str().to_string(),
            result.truth_cache_hit.to_string(),
            format!("{:.16e}", run.pde_variance),
            format!("{:.16e}", run.observation_std_tesla),
            run.total_residual_rows.to_string(),
            run.steel_observation_count.to_string(),
            run.posterior_converged.to_string(),
            format!("{:.16e}", run.initial_relative_error),
            format!("{:.16e}", run.posterior_relative_error),
            format!(
                "{:.16e}",
                safe_ratio(run.posterior_relative_error, run.initial_relative_error)
            ),
            format!("{:.16e}", run.initial_residual_norm),
            format!("{:.16e}", run.truth_residual_norm),
            format!("{:.16e}", run.posterior_residual_norm),
            format!("{:.16e}", run.initial_steel_rmse),
            format!("{:.16e}", run.posterior_steel_rmse),
            format!("{:.16e}", run.steel_rmse_improvement_ratio),
            format!("{:.16e}", run.posterior_steel_relative_rmse),
            format!("{:.16e}", run.posterior_steel_max_abs_residual),
            run.all_finite_variances.to_string(),
            run.nonnegative_variances.to_string(),
            run.b_quantity_variances_finite.to_string(),
            run.b_quantity_variances_nonnegative.to_string(),
            run.latent_variance_count.to_string(),
            run.latent_variances_finite.to_string(),
            run.latent_variances_nonnegative.to_string(),
            run.diagnostics.step_solve_attempts.to_string(),
            run.diagnostics.accepted_iterations.to_string(),
            run.diagnostics.line_search_residual_evaluations.to_string(),
            run.diagnostics.final_factorizations.to_string(),
            run.diagnostics.metis_cache_hits.to_string(),
            run.diagnostics.metis_cache_misses.to_string(),
            run.diagnostics.cholesky_factor_attempts.to_string(),
            run.diagnostics.cholesky_factor_successes.to_string(),
            run.diagnostics.cholesky_unshifted_attempts.to_string(),
            run.diagnostics.cholesky_symmetrized_attempts.to_string(),
            run.diagnostics.cholesky_cached_shift_attempts.to_string(),
            run.diagnostics.cholesky_cached_shift_successes.to_string(),
            run.diagnostics.cholesky_shifted_attempts.to_string(),
            run.diagnostics.cholesky_shifted_successes.to_string(),
            format!("{:.16e}", run.diagnostics.cholesky_max_shift),
            format!("{:.16e}", run.diagnostics.cholesky_factorization_seconds),
            run.assembly.prior_precision_nnz.to_string(),
            run.assembly
                .term_operator_nnz(NonlinearAssemblyTermKind::Residual)
                .to_string(),
            run.assembly
                .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual)
                .to_string(),
            run.assembly.posterior_precision_nnz.to_string(),
            run.assembly
                .factor_nnz
                .unwrap_or(run.final_factorization.nnz)
                .to_string(),
            format!(
                "{:.16e}",
                run.assembly
                    .fill_ratio_vs_lower_triangle
                    .unwrap_or(f64::NAN)
            ),
        ];
        csv.push_str(&fields.join(","));
        csv.push('\n');
    }
    csv
}

fn team13_map_parity_internal_steel_csv(run: &Team13MapParityRunResult) -> String {
    let mut csv = "name,group,observed,initial_prediction,posterior_prediction,initial_residual,posterior_residual,prior_variance,posterior_variance\n"
        .to_string();
    for (report, variance) in run
        .internal_steel_reports
        .iter()
        .zip(run.internal_steel_variances.iter())
    {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report
                .steel_surface_group
                .map(|group| group.as_str())
                .unwrap_or("unknown"),
            report.observed,
            report.initial_prediction,
            report.posterior_prediction,
            report.initial_prediction - report.observed,
            report.posterior_prediction - report.observed,
            variance.prior_variance,
            variance.posterior_variance
        ));
    }
    csv
}

fn team13_map_parity_json(result: &Team13MapParityResult) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"experiment\": \"team13_map_parity\",\n");
    json.push_str(&format!(
        "  \"mesh_path\": \"{}\",\n  \"domain\": \"{}\",\n",
        json_escape(&result.mesh_path.display().to_string()),
        result.domain_mode.as_str()
    ));
    json.push_str(&format!(
        "  \"material_kind\": \"{}\",\n  \"vertices\": {},\n  \"edges\": {},\n  \"cells\": {},\n  \"active_dofs\": {},\n  \"boundary_edge_dofs\": {},\n",
        result.material_kind.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs
    ));
    json.push_str(&format!(
        "  \"prior_kind\": \"{}\",\n  \"pde_residual_kind\": \"{}\",\n  \"step_regularization\": \"{}\",\n  \"prior_kappa\": {:.16e},\n  \"prior_tau\": {:.16e},\n  \"prior_diagonal_shift\": {:.16e},\n  \"steel_observation_mode\": \"{}\",\n",
        result.prior_kind.as_str(),
        result.pde_residual_kind.as_str(),
        result.step_regularization.as_str(),
        result.prior_kappa,
        result.prior_tau,
        result.prior_diagonal_shift,
        result.steel_observation_quadrature.as_str()
    ));
    json.push_str(&format!(
        "  \"truth_converged\": {},\n  \"truth_cache_hit\": {},\n  \"initial_residual_norm\": {:.16e},\n  \"truth_residual_norm\": {:.16e},\n",
        result.truth_converged,
        result.truth_cache_hit,
        result.initial_residual_norm,
        result.truth_residual_norm
    ));
    json.push_str("  \"runs\": [\n");
    for (index, run) in std::iter::once(&result.default_run)
        .chain(result.sweep_runs.iter())
        .enumerate()
    {
        json.push_str(&format!(
            concat!(
                "    {{\"label\": \"{}\", \"pde_variance\": {:.16e}, ",
                "\"observation_std_tesla\": {:.16e}, \"posterior_converged\": {}, ",
                "\"initial_relative_error\": {:.16e}, \"posterior_relative_error\": {:.16e}, ",
                "\"initial_steel_rmse\": {:.16e}, \"posterior_steel_rmse\": {:.16e}, ",
                "\"posterior_residual_norm\": {:.16e}, \"prior_precision_nnz\": {}, ",
                "\"posterior_precision_nnz\": {}, \"factor_nnz\": {}, ",
                "\"fill_ratio_vs_lower\": {:.16e}, ",
                "\"step_solve_attempts\": {}, \"accepted_iterations\": {}, ",
                "\"line_search_residual_evaluations\": {}, \"final_factorizations\": {}, ",
                "\"metis_cache_hits\": {}, \"metis_cache_misses\": {}, ",
                "\"cholesky_factor_attempts\": {}, \"cholesky_factor_successes\": {}, ",
                "\"cholesky_unshifted_attempts\": {}, \"cholesky_symmetrized_attempts\": {}, ",
                "\"cholesky_cached_shift_attempts\": {}, \"cholesky_cached_shift_successes\": {}, ",
                "\"cholesky_shifted_attempts\": {}, \"cholesky_shifted_successes\": {}, ",
                "\"cholesky_max_shift\": {:.16e}, \"cholesky_factorization_seconds\": {:.16e}, ",
                "\"all_finite_variances\": {}, \"nonnegative_variances\": {}, ",
                "\"b_quantity_variances_finite\": {}, \"b_quantity_variances_nonnegative\": {}, ",
                "\"latent_variance_count\": {}, ",
                "\"latent_variances_finite\": {}, \"latent_variances_nonnegative\": {}}}{} \n"
            ),
            json_escape(&run.label),
            run.pde_variance,
            run.observation_std_tesla,
            run.posterior_converged,
            run.initial_relative_error,
            run.posterior_relative_error,
            run.initial_steel_rmse,
            run.posterior_steel_rmse,
            run.posterior_residual_norm,
            run.assembly.prior_precision_nnz,
            run.assembly.posterior_precision_nnz,
            run.assembly
                .factor_nnz
                .unwrap_or(run.final_factorization.nnz),
            run.assembly
                .fill_ratio_vs_lower_triangle
                .unwrap_or(f64::NAN),
            run.diagnostics.step_solve_attempts,
            run.diagnostics.accepted_iterations,
            run.diagnostics.line_search_residual_evaluations,
            run.diagnostics.final_factorizations,
            run.diagnostics.metis_cache_hits,
            run.diagnostics.metis_cache_misses,
            run.diagnostics.cholesky_factor_attempts,
            run.diagnostics.cholesky_factor_successes,
            run.diagnostics.cholesky_unshifted_attempts,
            run.diagnostics.cholesky_symmetrized_attempts,
            run.diagnostics.cholesky_cached_shift_attempts,
            run.diagnostics.cholesky_cached_shift_successes,
            run.diagnostics.cholesky_shifted_attempts,
            run.diagnostics.cholesky_shifted_successes,
            run.diagnostics.cholesky_max_shift,
            run.diagnostics.cholesky_factorization_seconds,
            run.all_finite_variances,
            run.nonnegative_variances,
            run.b_quantity_variances_finite,
            run.b_quantity_variances_nonnegative,
            run.latent_variance_count,
            run.latent_variances_finite,
            run.latent_variances_nonnegative,
            if index == result.sweep_runs.len() {
                ""
            } else {
                ","
            }
        ));
    }
    json.push_str("  ]\n");
    json.push_str("}\n");
    json
}

fn write_team13_map_parity_outputs(
    output_dir: &Path,
    result: &Team13MapParityResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 MAP parity output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    fs::write(
        output_dir.join("map_parity.json"),
        team13_map_parity_json(result),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 MAP parity JSON `{}`: {err}",
            output_dir.join("map_parity.json").display()
        )
    })?;
    fs::write(
        output_dir.join("map_parity_runs.csv"),
        team13_map_parity_runs_csv(result),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 MAP parity runs CSV `{}`: {err}",
            output_dir.join("map_parity_runs.csv").display()
        )
    })?;
    fs::write(
        output_dir.join("internal_steel_predictions.csv"),
        team13_map_parity_internal_steel_csv(&result.default_run),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 MAP parity internal steel CSV `{}`: {err}",
            output_dir.join("internal_steel_predictions.csv").display()
        )
    })?;
    fs::write(
        output_dir.join("published_steel_reporting.csv"),
        team13_nonlinear_steel_predictions_csv(
            &result.default_run.published_steel_benchmark_reports,
        ),
    )
    .map_err(|err| {
        format!(
            "failed to write TEAM 13 MAP parity published steel CSV `{}`: {err}",
            output_dir.join("published_steel_reporting.csv").display()
        )
    })?;
    Ok(())
}

fn write_team13_operator_uncertainty_outputs(
    output_dir: &Path,
    result: &Team13OperatorUncertaintyResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 operator uncertainty output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    write_string_file(
        &output_dir.join("operator_uncertainty_summary.csv"),
        &team13_operator_uncertainty_summary_csv(result),
        "TEAM 13 operator uncertainty summary",
    )?;
    write_string_file(
        &output_dir.join("region_variance_summary.csv"),
        &team13_operator_region_variance_csv(result),
        "TEAM 13 operator uncertainty region variance summary",
    )?;
    write_string_file(
        &output_dir.join("steel_patch_variance.csv"),
        &team13_operator_steel_patch_variance_csv(result),
        "TEAM 13 operator uncertainty steel patch variance",
    )?;
    write_string_file(
        &output_dir.join("steel_patch_error_vs_std.csv"),
        &team13_operator_steel_patch_error_csv(result),
        "TEAM 13 operator uncertainty steel patch error/std",
    )?;
    write_string_file(
        &output_dir.join("variance_indicator_correlation.csv"),
        &team13_operator_indicator_correlation_csv(result),
        "TEAM 13 operator uncertainty indicator correlation",
    )?;
    write_string_file(
        &output_dir.join("region_audit.csv"),
        &team13_region_audit_csv(&result.audit),
        "TEAM 13 operator uncertainty region audit",
    )?;
    Ok(())
}

fn write_team13_material_gap_uq_outputs(
    output_dir: &Path,
    result: &Team13MaterialGapUqResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 material/gap UQ output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    write_string_file(
        &output_dir.join("material_gap_case_summary.csv"),
        &team13_material_gap_case_summary_csv(result),
        "TEAM 13 material/gap case summary",
    )?;
    write_string_file(
        &output_dir.join("steel_patch_material_gap_variance.csv"),
        &team13_material_gap_variance_decomposition_csv(result),
        "TEAM 13 material/gap steel patch variance decomposition",
    )?;
    Ok(())
}

fn write_team13_joint_material_uq_outputs(
    output_dir: &Path,
    result: &Team13JointMaterialUqResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 joint material UQ output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    write_string_file(
        &output_dir.join("summary.csv"),
        &team13_joint_material_summary_csv(result),
        "TEAM 13 joint material UQ summary",
    )?;
    write_string_file(
        &output_dir.join("material_posterior.csv"),
        &team13_joint_material_posterior_csv(result),
        "TEAM 13 joint material posterior",
    )?;
    write_string_file(
        &output_dir.join("material_prior_calibration.csv"),
        &team13_joint_material_prior_calibration_csv(result),
        "TEAM 13 joint material prior calibration",
    )?;
    write_string_file(
        &output_dir.join("material_covariance.csv"),
        &team13_joint_material_covariance_csv(result),
        "TEAM 13 joint material covariance",
    )?;
    write_string_file(
        &output_dir.join("bh_curve_bands.csv"),
        &team13_joint_material_bh_curve_bands_csv(result),
        "TEAM 13 joint material B-H curve bands",
    )?;
    write_string_file(
        &output_dir.join("steel_patch_joint_variance.csv"),
        &team13_joint_material_steel_patch_csv(result),
        "TEAM 13 joint material steel patch variance",
    )?;
    write_string_file(
        &output_dir.join("posterior_history.csv"),
        &team13_joint_material_history_csv(result),
        "TEAM 13 joint material solver history",
    )?;
    write_string_file(
        &output_dir.join("solver_diagnostics.csv"),
        &team13_joint_material_solver_diagnostics_csv(result),
        "TEAM 13 joint material solver diagnostics",
    )?;
    write_string_file(
        &output_dir.join("residual_terms.csv"),
        &team13_joint_material_residual_terms_csv(result),
        "TEAM 13 joint material residual terms",
    )?;
    write_string_file(
        &output_dir.join("objective_components.csv"),
        &team13_joint_material_objective_components_csv(result),
        "TEAM 13 joint material objective components",
    )?;
    write_string_file(
        &output_dir.join("fixed_vs_joint_comparison.csv"),
        &team13_joint_material_fixed_vs_joint_comparison_csv(result),
        "TEAM 13 fixed-vs-joint material comparison",
    )?;
    Ok(())
}

fn write_team13_material_only_uq_outputs(
    output_dir: &Path,
    result: &Team13MaterialOnlyUqResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 material-only UQ output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    write_string_file(
        &output_dir.join("summary.csv"),
        &team13_material_only_summary_csv(result),
        "TEAM 13 material-only UQ summary",
    )?;
    write_string_file(
        &output_dir.join("theta_posterior.csv"),
        &team13_joint_material_posterior_csv_from_reports(&result.material_posterior),
        "TEAM 13 material-only theta posterior",
    )?;
    write_string_file(
        &output_dir.join("theta_covariance.csv"),
        &team13_material_only_covariance_csv(result),
        "TEAM 13 material-only theta covariance",
    )?;
    write_string_file(
        &output_dir.join("steel_predictions.csv"),
        &team13_material_only_steel_predictions_csv(result),
        "TEAM 13 material-only steel predictions",
    )?;
    write_string_file(
        &output_dir.join("objective_components.csv"),
        &team13_material_only_objective_components_csv(result),
        "TEAM 13 material-only objective components",
    )?;
    write_string_file(
        &output_dir.join("sensitivity_singular_values.csv"),
        &team13_material_only_sensitivity_csv(result),
        "TEAM 13 material-only sensitivity singular values",
    )?;
    write_string_file(
        &output_dir.join("bh_curve_bands.csv"),
        &team13_joint_material_bh_curve_bands_csv_from_rows(&result.bh_curve_bands),
        "TEAM 13 material-only B-H curve bands",
    )?;
    write_string_file(
        &output_dir.join("solver_history.csv"),
        &team13_material_only_history_csv(result),
        "TEAM 13 material-only solver history",
    )?;
    write_string_file(
        &output_dir.join("comparison.csv"),
        &team13_material_only_comparison_csv(result),
        "TEAM 13 material-only comparison",
    )?;
    Ok(())
}

fn write_team13_identifiable_material_uq_outputs(
    output_dir: &Path,
    result: &Team13IdentifiableMaterialUqResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 identifiable material UQ output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    write_string_file(
        &output_dir.join("summary.csv"),
        &team13_identifiable_material_summary_csv(result),
        "TEAM 13 identifiable material UQ summary",
    )?;
    write_string_file(
        &output_dir.join("material_basis.csv"),
        &team13_identifiable_material_basis_csv(result),
        "TEAM 13 identifiable material basis",
    )?;
    write_string_file(
        &output_dir.join("baseline_perturbation.csv"),
        &team13_identifiable_baseline_perturbation_csv(result),
        "TEAM 13 identifiable material baseline perturbation",
    )?;
    write_string_file(
        &output_dir.join("eta_posterior.csv"),
        &team13_identifiable_eta_posterior_csv(result),
        "TEAM 13 identifiable material eta posterior",
    )?;
    write_string_file(
        &output_dir.join("eta_covariance.csv"),
        &team13_identifiable_eta_covariance_csv(result),
        "TEAM 13 identifiable material eta covariance",
    )?;
    write_string_file(
        &output_dir.join("theta_posterior.csv"),
        &team13_identifiable_theta_posterior_csv(result),
        "TEAM 13 identifiable material theta posterior",
    )?;
    write_string_file(
        &output_dir.join("theta_covariance.csv"),
        &team13_identifiable_theta_covariance_csv(result),
        "TEAM 13 identifiable material theta covariance",
    )?;
    write_string_file(
        &output_dir.join("steel_predictions.csv"),
        &team13_identifiable_steel_predictions_csv(result),
        "TEAM 13 identifiable material steel predictions",
    )?;
    write_string_file(
        &output_dir.join("comparison.csv"),
        &team13_identifiable_comparison_csv(result),
        "TEAM 13 identifiable material comparison",
    )?;
    write_string_file(
        &output_dir.join("bh_curve_bands.csv"),
        &team13_identifiable_bh_curve_bands_csv(result),
        "TEAM 13 identifiable material B-H curve bands",
    )?;
    write_string_file(
        &output_dir.join("solver_history.csv"),
        &team13_material_only_history_csv_from_rows(&result.history),
        "TEAM 13 identifiable material solver history",
    )?;
    Ok(())
}

fn write_team13_identifiable_joint_material_uq_outputs(
    output_dir: &Path,
    result: &Team13IdentifiableJointMaterialUqResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create TEAM 13 identifiable joint material UQ output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    write_string_file(
        &output_dir.join("summary.csv"),
        &team13_identifiable_joint_material_summary_csv(result),
        "TEAM 13 identifiable joint material UQ summary",
    )?;
    write_string_file(
        &output_dir.join("material_basis.csv"),
        &team13_identifiable_joint_material_basis_csv(result),
        "TEAM 13 identifiable joint material basis",
    )?;
    write_string_file(
        &output_dir.join("baseline_perturbation.csv"),
        &team13_identifiable_joint_baseline_perturbation_csv(result),
        "TEAM 13 identifiable joint material baseline perturbation",
    )?;
    write_string_file(
        &output_dir.join("eta_posterior.csv"),
        &team13_identifiable_joint_eta_posterior_csv(result),
        "TEAM 13 identifiable joint material eta posterior",
    )?;
    write_string_file(
        &output_dir.join("eta_covariance.csv"),
        &team13_identifiable_joint_eta_covariance_csv(result),
        "TEAM 13 identifiable joint material eta covariance",
    )?;
    write_string_file(
        &output_dir.join("theta_posterior.csv"),
        &team13_identifiable_joint_theta_posterior_csv(result),
        "TEAM 13 identifiable joint material theta posterior",
    )?;
    write_string_file(
        &output_dir.join("theta_covariance.csv"),
        &team13_identifiable_joint_theta_covariance_csv(result),
        "TEAM 13 identifiable joint material theta covariance",
    )?;
    write_string_file(
        &output_dir.join("bh_curve_bands.csv"),
        &team13_identifiable_joint_bh_curve_bands_csv(result),
        "TEAM 13 identifiable joint material B-H curve bands",
    )?;
    write_string_file(
        &output_dir.join("steel_patch_joint_variance.csv"),
        &team13_identifiable_joint_steel_patch_csv(result),
        "TEAM 13 identifiable joint material steel patch variance",
    )?;
    write_string_file(
        &output_dir.join("posterior_history.csv"),
        &team13_identifiable_joint_history_csv(result),
        "TEAM 13 identifiable joint material solver history",
    )?;
    write_string_file(
        &output_dir.join("solver_diagnostics.csv"),
        &team13_identifiable_joint_solver_diagnostics_csv(result),
        "TEAM 13 identifiable joint material solver diagnostics",
    )?;
    write_string_file(
        &output_dir.join("residual_terms.csv"),
        &team13_identifiable_joint_residual_terms_csv(result),
        "TEAM 13 identifiable joint material residual terms",
    )?;
    write_string_file(
        &output_dir.join("objective_components.csv"),
        &team13_identifiable_joint_objective_components_csv(result),
        "TEAM 13 identifiable joint material objective components",
    )?;
    write_string_file(
        &output_dir.join("fixed_vs_joint_comparison.csv"),
        &team13_identifiable_joint_fixed_vs_joint_comparison_csv(result),
        "TEAM 13 identifiable joint material fixed-vs-joint comparison",
    )?;
    Ok(())
}

fn team13_joint_material_summary_csv(result: &Team13JointMaterialUqResult) -> String {
    format!(
        "mesh_path,domain,observed_gap,vertices,edges,cells,active_dofs,boundary_edge_dofs,anchor_0_tesla,anchor_1_tesla,anchor_2_tesla,material_prior_std,material_prior_calibration,material_prior_calibration_target,material_prior_target_steel_rms_tesla,magnitude_smoothing_tesla,deterministic_converged,deterministic_residual_norm,posterior_converged,posterior_residual_norm,posterior_precision_nnz,posterior_factor_nnz\n{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{:.16e},{},{:.16e},{},{:.16e},{},{}\n",
        result.mesh_path.display(),
        result.domain_mode.as_str(),
        result.observed_steel_gap.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs,
        result.material_anchor_b_tesla[0],
        result.material_anchor_b_tesla[1],
        result.material_anchor_b_tesla[2],
        result.material_prior_std,
        result.material_prior_calibration.mode.as_str(),
        result.material_prior_calibration.target.as_str(),
        csv_optional(result.material_prior_calibration.target_steel_rms_tesla),
        result.magnitude_smoothing_tesla,
        result.deterministic_converged,
        result.deterministic_residual_norm,
        result.posterior_converged,
        result.posterior_residual_norm,
        result.posterior_precision_nnz,
        result.posterior_factor_nnz
    )
}

fn team13_joint_material_prior_calibration_csv(result: &Team13JointMaterialUqResult) -> String {
    let report = &result.material_prior_calibration;
    format!(
        "mode,target,target_steel_rms_tesla,configured_material_prior_std,material_prior_std,unclamped_material_prior_std,material_prior_std_floor,material_prior_std_ceiling,unit_theta_steel_rms_tesla,sensitivity_frobenius_norm_tesla,max_abs_sensitivity_tesla,theta_0_column_norm_tesla,theta_1_column_norm_tesla,theta_2_column_norm_tesla,steel_row_count\n{},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e},{}\n",
        report.mode.as_str(),
        report.target.as_str(),
        csv_optional(report.target_steel_rms_tesla),
        report.configured_material_prior_std,
        report.material_prior_std,
        csv_optional(report.unclamped_material_prior_std),
        report.material_prior_std_floor,
        report.material_prior_std_ceiling,
        csv_optional(report.unit_theta_steel_rms_tesla),
        csv_optional(report.sensitivity_frobenius_norm_tesla),
        csv_optional(report.max_abs_sensitivity_tesla),
        report.theta_column_norms_tesla[0],
        report.theta_column_norms_tesla[1],
        report.theta_column_norms_tesla[2],
        report.steel_row_count
    )
}

fn team13_joint_material_posterior_csv(result: &Team13JointMaterialUqResult) -> String {
    team13_joint_material_posterior_csv_from_reports(&result.material_posterior)
}

fn team13_joint_material_posterior_csv_from_reports(
    reports: &[Team13MaterialParameterPosteriorReport],
) -> String {
    let mut csv =
        "name,anchor_b_tesla,prior_mean,posterior_mean,prior_std,posterior_std\n".to_string();
    for report in reports {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.anchor_b_tesla,
            report.prior_mean,
            report.posterior_mean,
            report.prior_std,
            report.posterior_std
        ));
    }
    csv
}

fn team13_material_only_summary_csv(result: &Team13MaterialOnlyUqResult) -> String {
    format!(
        "mesh_path,domain,observed_gap,vertices,edges,cells,active_dofs,boundary_edge_dofs,anchor_0_tesla,anchor_1_tesla,anchor_2_tesla,material_prior_std,material_prior_calibration,material_prior_calibration_target,material_prior_target_steel_rms_tesla,magnitude_smoothing_tesla,nominal_forward_converged,nominal_forward_residual_norm,map_forward_converged,map_forward_residual_norm,posterior_converged,theta_0,theta_1,theta_2,total_objective,sensitivity_rank,sensitivity_condition_number\n{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{:.16e},{},{:.16e},{},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e}\n",
        result.mesh_path.display(),
        result.domain_mode.as_str(),
        result.observed_steel_gap.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs,
        result.material_anchor_b_tesla[0],
        result.material_anchor_b_tesla[1],
        result.material_anchor_b_tesla[2],
        result.material_prior_std,
        result.material_prior_calibration.mode.as_str(),
        result.material_prior_calibration.target.as_str(),
        csv_optional(result.material_prior_calibration.target_steel_rms_tesla),
        result.magnitude_smoothing_tesla,
        result.nominal_forward_converged,
        result.nominal_forward_residual_norm,
        result.map_forward_converged,
        result.map_forward_residual_norm,
        result.posterior_converged,
        result.theta_map[0],
        result.theta_map[1],
        result.theta_map[2],
        result.objective_components.total,
        result.sensitivity.rank,
        result.sensitivity.condition_number
    )
}

fn team13_material_only_covariance_csv(result: &Team13MaterialOnlyUqResult) -> String {
    let mut csv = "row,col,covariance,correlation\n".to_string();
    for row in 0..3 {
        for col in 0..3 {
            csv.push_str(&format!(
                "theta_h_{},theta_h_{},{:.16e},{:.16e}\n",
                row,
                col,
                result.material_posterior_covariance[row][col],
                result.material_posterior_correlation[row][col]
            ));
        }
    }
    csv
}

fn team13_material_only_steel_predictions_csv(result: &Team13MaterialOnlyUqResult) -> String {
    let mut csv = "name,group,observed,nominal_prediction,map_prediction,nominal_residual,map_residual,nominal_signed_prediction,map_signed_prediction,row_nnz\n"
        .to_string();
    for report in &result.steel_predictions {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}\n",
            report.name,
            report.group.as_str(),
            report.observed,
            report.nominal_prediction,
            report.map_prediction,
            report.nominal_residual,
            report.map_residual,
            report.nominal_signed_prediction,
            report.map_signed_prediction,
            report.row_nnz
        ));
    }
    csv
}

fn team13_material_only_objective_components_csv(result: &Team13MaterialOnlyUqResult) -> String {
    let components = &result.objective_components;
    format!(
        "prior_material,steel_observation,total\n{:.16e},{:.16e},{:.16e}\n",
        components.prior_material, components.steel_observation, components.total
    )
}

fn team13_material_only_sensitivity_csv(result: &Team13MaterialOnlyUqResult) -> String {
    let sensitivity = &result.sensitivity;
    format!(
        "singular_0,singular_1,singular_2,rank,condition_number,frobenius_norm_tesla,max_abs_sensitivity_tesla,theta_0_column_norm_tesla,theta_1_column_norm_tesla,theta_2_column_norm_tesla,steel_row_count\n{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}\n",
        sensitivity.singular_values[0],
        sensitivity.singular_values[1],
        sensitivity.singular_values[2],
        sensitivity.rank,
        sensitivity.condition_number,
        sensitivity.frobenius_norm_tesla,
        sensitivity.max_abs_sensitivity_tesla,
        sensitivity.theta_column_norms_tesla[0],
        sensitivity.theta_column_norms_tesla[1],
        sensitivity.theta_column_norms_tesla[2],
        sensitivity.steel_row_count
    )
}

fn team13_material_only_history_csv(result: &Team13MaterialOnlyUqResult) -> String {
    team13_material_only_history_csv_from_rows(&result.history)
}

fn team13_material_only_history_csv_from_rows(rows: &[Team13MaterialOnlyIteration]) -> String {
    let mut csv = "iteration,objective,trial_objective,steel_weighted_residual_norm,gradient_norm,step_norm,alpha,regularization_lambda,forward_residual_norm,trial_forward_residual_norm\n"
        .to_string();
    for entry in rows {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            entry.iteration,
            entry.objective,
            entry.trial_objective,
            entry.steel_weighted_residual_norm,
            entry.gradient_norm,
            entry.step_norm,
            entry.alpha,
            entry.regularization_lambda,
            entry.forward_residual_norm,
            entry.trial_forward_residual_norm
        ));
    }
    csv
}

fn team13_identifiable_material_summary_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    format!(
        "mesh_path,domain,observed_gap,vertices,edges,cells,active_dofs,boundary_edge_dofs,anchor_0_tesla,anchor_1_tesla,anchor_2_tesla,eta_prior_std,eta_prior_calibration,eta_prior_calibration_target,eta_prior_target_steel_rms_tesla,magnitude_smoothing_tesla,svd_relative_tolerance,svd_absolute_tolerance,retained_rank,perturbation_target_rms_tesla,perturbation_achieved_linearized_rms_tesla,benchmark_forward_converged,benchmark_forward_residual_norm,biased_forward_converged,biased_forward_residual_norm,map_forward_converged,map_forward_residual_norm,posterior_converged,total_objective,eta_dimension,theta_0,theta_1,theta_2\n{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{},{:.16e},{},{:.16e},{},{:.16e},{},{:.16e},{},{:.16e},{:.16e},{:.16e}\n",
        result.mesh_path.display(),
        result.domain_mode.as_str(),
        result.observed_steel_gap.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs,
        result.material_anchor_b_tesla[0],
        result.material_anchor_b_tesla[1],
        result.material_anchor_b_tesla[2],
        result.eta_prior_std,
        result.eta_prior_calibration.mode.as_str(),
        result.eta_prior_calibration.target.as_str(),
        csv_optional(result.eta_prior_calibration.target_steel_rms_tesla),
        result.magnitude_smoothing_tesla,
        result.svd_relative_tolerance,
        result.svd_absolute_tolerance,
        result.retained_rank,
        result.baseline_perturbation.target_rms_tesla,
        result.baseline_perturbation.achieved_linearized_rms_tesla,
        result.benchmark_forward_converged,
        result.benchmark_forward_residual_norm,
        result.biased_forward_converged,
        result.biased_forward_residual_norm,
        result.map_forward_converged,
        result.map_forward_residual_norm,
        result.posterior_converged,
        result.objective_components.total,
        result.eta_map.len(),
        result.theta_map[0],
        result.theta_map[1],
        result.theta_map[2]
    )
}

fn team13_identifiable_material_basis_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    let mut csv =
        "mode_index,retained,singular_value,relative_singular_value,theta_0,theta_1,theta_2\n"
            .to_string();
    for report in &result.material_basis {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.mode_index,
            report.retained,
            report.singular_value,
            report.relative_singular_value,
            report.theta_coefficients[0],
            report.theta_coefficients[1],
            report.theta_coefficients[2]
        ));
    }
    csv
}

fn team13_identifiable_baseline_perturbation_csv(
    result: &Team13IdentifiableMaterialUqResult,
) -> String {
    let mut csv = "quantity,index,value\n".to_string();
    csv.push_str(&format!(
        "gap_difference_rms_tesla,,{:.16e}\n",
        result.baseline_perturbation.gap_difference_rms_tesla
    ));
    csv.push_str(&format!(
        "target_rms_tesla,,{:.16e}\n",
        result.baseline_perturbation.target_rms_tesla
    ));
    csv.push_str(&format!(
        "achieved_linearized_rms_tesla,,{:.16e}\n",
        result.baseline_perturbation.achieved_linearized_rms_tesla
    ));
    for (index, value) in result.baseline_perturbation.eta_bias.iter().enumerate() {
        csv.push_str(&format!("eta_bias,{index},{value:.16e}\n"));
    }
    for (index, value) in result.baseline_perturbation.theta_bias.iter().enumerate() {
        csv.push_str(&format!("theta_bias,{index},{value:.16e}\n"));
    }
    csv
}

fn team13_identifiable_eta_posterior_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    let mut csv =
        "name,prior_mean,posterior_mean,prior_std,posterior_std,eta_bias,recovery_fraction\n"
            .to_string();
    for report in &result.eta_posterior {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.prior_mean,
            report.posterior_mean,
            report.prior_std,
            report.posterior_std,
            report.eta_bias,
            report.recovery_fraction
        ));
    }
    csv
}

fn team13_identifiable_eta_covariance_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    let mut csv = "row,col,covariance,correlation\n".to_string();
    let std = result
        .eta_posterior_covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index].max(0.0).sqrt())
        .collect::<Vec<_>>();
    for row in 0..result.eta_posterior_covariance.len() {
        for col in 0..result.eta_posterior_covariance.len() {
            csv.push_str(&format!(
                "eta_svd_{},eta_svd_{},{:.16e},{:.16e}\n",
                row,
                col,
                result.eta_posterior_covariance[row][col],
                safe_ratio(
                    result.eta_posterior_covariance[row][col],
                    std[row] * std[col]
                )
            ));
        }
    }
    csv
}

fn team13_identifiable_theta_posterior_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    let mut csv =
        "name,anchor_b_tesla,benchmark_mean,biased_baseline_mean,posterior_mean,posterior_std\n"
            .to_string();
    for report in &result.theta_posterior {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.anchor_b_tesla,
            report.benchmark_mean,
            report.biased_baseline_mean,
            report.posterior_mean,
            report.posterior_std
        ));
    }
    csv
}

fn team13_identifiable_theta_covariance_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    let correlation = correlation_from_covariance(result.theta_posterior_covariance);
    let mut csv = "row,col,covariance,correlation\n".to_string();
    for row in 0..3 {
        for col in 0..3 {
            csv.push_str(&format!(
                "theta_h_{},theta_h_{},{:.16e},{:.16e}\n",
                row, col, result.theta_posterior_covariance[row][col], correlation[row][col]
            ));
        }
    }
    csv
}

fn team13_identifiable_steel_predictions_csv(
    result: &Team13IdentifiableMaterialUqResult,
) -> String {
    let mut csv = "name,group,observed,benchmark_prediction,biased_prediction,corrected_prediction,benchmark_residual,biased_residual,corrected_residual,row_nnz\n"
        .to_string();
    for report in &result.steel_predictions {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}\n",
            report.name,
            report.group.as_str(),
            report.observed,
            report.benchmark_prediction,
            report.biased_prediction,
            report.corrected_prediction,
            report.benchmark_residual,
            report.biased_residual,
            report.corrected_residual,
            report.row_nnz
        ));
    }
    csv
}

fn team13_identifiable_comparison_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    let mut csv =
        "label,theta_0,theta_1,theta_2,eta_values,steel_rmse_tesla,steel_max_abs_residual_tesla,objective\n"
            .to_string();
    for row in &result.comparison {
        let eta_values = row
            .eta
            .iter()
            .map(|value| format!("{value:.16e}"))
            .collect::<Vec<_>>()
            .join(";");
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e}\n",
            row.label,
            row.theta[0],
            row.theta[1],
            row.theta[2],
            eta_values,
            row.steel_rmse_tesla,
            row.steel_max_abs_residual_tesla,
            row.objective
        ));
    }
    csv
}

fn team13_identifiable_bh_curve_bands_csv(result: &Team13IdentifiableMaterialUqResult) -> String {
    let mut csv = "b_tesla,benchmark_h_ampere_per_meter,biased_baseline_h_ampere_per_meter,corrected_mean_h_ampere_per_meter,corrected_std_h_ampere_per_meter,corrected_lower_2sigma_h_ampere_per_meter,corrected_upper_2sigma_h_ampere_per_meter\n"
        .to_string();
    for row in &result.bh_curve_bands {
        csv.push_str(&format!(
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            row.b_tesla,
            row.benchmark_h_ampere_per_meter,
            row.biased_baseline_h_ampere_per_meter,
            row.corrected_mean_h_ampere_per_meter,
            row.corrected_std_h_ampere_per_meter,
            row.corrected_lower_2sigma_h_ampere_per_meter,
            row.corrected_upper_2sigma_h_ampere_per_meter
        ));
    }
    csv
}

fn team13_identifiable_joint_material_summary_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    format!(
        "mesh_path,domain,observed_gap,vertices,edges,cells,active_dofs,boundary_edge_dofs,anchor_0_tesla,anchor_1_tesla,anchor_2_tesla,eta_prior_std,eta_prior_calibration,eta_prior_calibration_target,eta_prior_target_steel_rms_tesla,magnitude_smoothing_tesla,svd_relative_tolerance,svd_absolute_tolerance,retained_rank,perturbation_target_rms_tesla,perturbation_achieved_linearized_rms_tesla,benchmark_forward_converged,benchmark_forward_residual_norm,biased_forward_converged,biased_forward_residual_norm,posterior_converged,posterior_residual_norm,posterior_precision_nnz,posterior_factor_nnz,eta_dimension,theta_0,theta_1,theta_2\n{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{},{:.16e},{},{:.16e},{},{:.16e},{},{},{},{:.16e},{:.16e},{:.16e}\n",
        result.mesh_path.display(),
        result.domain_mode.as_str(),
        result.observed_steel_gap.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs,
        result.material_anchor_b_tesla[0],
        result.material_anchor_b_tesla[1],
        result.material_anchor_b_tesla[2],
        result.eta_prior_std,
        result.eta_prior_calibration.mode.as_str(),
        result.eta_prior_calibration.target.as_str(),
        csv_optional(result.eta_prior_calibration.target_steel_rms_tesla),
        result.magnitude_smoothing_tesla,
        result.svd_relative_tolerance,
        result.svd_absolute_tolerance,
        result.retained_rank,
        result.baseline_perturbation.target_rms_tesla,
        result.baseline_perturbation.achieved_linearized_rms_tesla,
        result.benchmark_forward_converged,
        result.benchmark_forward_residual_norm,
        result.biased_forward_converged,
        result.biased_forward_residual_norm,
        result.posterior_converged,
        result.posterior_residual_norm,
        result.posterior_precision_nnz,
        result.posterior_factor_nnz,
        result.eta_map.len(),
        result.theta_map[0],
        result.theta_map[1],
        result.theta_map[2]
    )
}

fn team13_identifiable_joint_material_basis_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv =
        "mode_index,retained,singular_value,relative_singular_value,theta_0,theta_1,theta_2\n"
            .to_string();
    for report in &result.material_basis {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.mode_index,
            report.retained,
            report.singular_value,
            report.relative_singular_value,
            report.theta_coefficients[0],
            report.theta_coefficients[1],
            report.theta_coefficients[2]
        ));
    }
    csv
}

fn team13_identifiable_joint_baseline_perturbation_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "quantity,index,value\n".to_string();
    csv.push_str(&format!(
        "gap_difference_rms_tesla,,{:.16e}\n",
        result.baseline_perturbation.gap_difference_rms_tesla
    ));
    csv.push_str(&format!(
        "target_rms_tesla,,{:.16e}\n",
        result.baseline_perturbation.target_rms_tesla
    ));
    csv.push_str(&format!(
        "achieved_linearized_rms_tesla,,{:.16e}\n",
        result.baseline_perturbation.achieved_linearized_rms_tesla
    ));
    for (index, value) in result.baseline_perturbation.eta_bias.iter().enumerate() {
        csv.push_str(&format!("eta_bias,{index},{value:.16e}\n"));
    }
    for (index, value) in result.baseline_perturbation.theta_bias.iter().enumerate() {
        csv.push_str(&format!("theta_bias,{index},{value:.16e}\n"));
    }
    csv
}

fn team13_identifiable_joint_eta_posterior_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv =
        "name,prior_mean,posterior_mean,prior_std,posterior_std,eta_bias,recovery_fraction\n"
            .to_string();
    for report in &result.eta_posterior {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.prior_mean,
            report.posterior_mean,
            report.prior_std,
            report.posterior_std,
            report.eta_bias,
            report.recovery_fraction
        ));
    }
    csv
}

fn team13_identifiable_joint_eta_covariance_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "row,col,covariance,correlation\n".to_string();
    let std = result
        .eta_posterior_covariance
        .iter()
        .enumerate()
        .map(|(index, row)| row[index].max(0.0).sqrt())
        .collect::<Vec<_>>();
    for row in 0..result.eta_posterior_covariance.len() {
        for col in 0..result.eta_posterior_covariance.len() {
            csv.push_str(&format!(
                "eta_svd_{},eta_svd_{},{:.16e},{:.16e}\n",
                row,
                col,
                result.eta_posterior_covariance[row][col],
                safe_ratio(
                    result.eta_posterior_covariance[row][col],
                    std[row] * std[col]
                )
            ));
        }
    }
    csv
}

fn team13_identifiable_joint_theta_posterior_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv =
        "name,anchor_b_tesla,benchmark_mean,biased_baseline_mean,posterior_mean,posterior_std\n"
            .to_string();
    for report in &result.theta_posterior {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.anchor_b_tesla,
            report.benchmark_mean,
            report.biased_baseline_mean,
            report.posterior_mean,
            report.posterior_std
        ));
    }
    csv
}

fn team13_identifiable_joint_theta_covariance_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let correlation = correlation_from_covariance(result.theta_posterior_covariance);
    let mut csv = "row,col,covariance,correlation\n".to_string();
    for row in 0..3 {
        for col in 0..3 {
            csv.push_str(&format!(
                "theta_h_{},theta_h_{},{:.16e},{:.16e}\n",
                row, col, result.theta_posterior_covariance[row][col], correlation[row][col]
            ));
        }
    }
    csv
}

fn team13_identifiable_joint_bh_curve_bands_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "b_tesla,benchmark_h_ampere_per_meter,biased_baseline_h_ampere_per_meter,corrected_mean_h_ampere_per_meter,corrected_std_h_ampere_per_meter,corrected_lower_2sigma_h_ampere_per_meter,corrected_upper_2sigma_h_ampere_per_meter\n"
        .to_string();
    for row in &result.bh_curve_bands {
        csv.push_str(&format!(
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            row.b_tesla,
            row.benchmark_h_ampere_per_meter,
            row.biased_baseline_h_ampere_per_meter,
            row.corrected_mean_h_ampere_per_meter,
            row.corrected_std_h_ampere_per_meter,
            row.corrected_lower_2sigma_h_ampere_per_meter,
            row.corrected_upper_2sigma_h_ampere_per_meter
        ));
    }
    csv
}

fn team13_identifiable_joint_steel_patch_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "name,group,prediction,signed_prediction,observed,residual,total_variance,state_conditional_variance,eta_explained_variance,posterior_std,row_nnz\n"
        .to_string();
    for report in &result.steel_patch_reports {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}\n",
            report.name,
            report.group.as_str(),
            report.prediction,
            report.signed_prediction,
            report.observed,
            report.residual,
            report.total_variance,
            report.state_conditional_variance,
            report.eta_explained_variance,
            report.posterior_std,
            report.row_nnz
        ));
    }
    csv
}

fn team13_identifiable_joint_material_solves(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> [&Team13JointMaterialSolveDiagnostics; 2] {
    [
        &result.fixed_biased_material_solve,
        &result.joint_identifiable_material_solve,
    ]
}

fn team13_identifiable_joint_history_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "label,iteration,objective,trial_objective,weighted_residual_norm,gradient_norm,step_norm,alpha,regularization,regularization_lambda,linear_solve_mode,linear_solve_iterations,linear_solve_residual_norm,linear_solve_converged,linear_solve_factor_nnz\n".to_string();
    for solve in team13_identifiable_joint_material_solves(result) {
        for entry in &solve.history {
            csv.push_str(&format!(
                "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{},{},{:.16e},{},{}\n",
                solve.label,
                entry.iteration,
                entry.objective,
                entry.trial_objective,
                entry.residual_norm,
                entry.gradient_norm,
                entry.step_norm,
                entry.alpha,
                entry.regularization.as_str(),
                entry.regularization_lambda,
                gauss_newton_linear_solve_mode_label(entry.linear_solve.mode),
                entry.linear_solve.iterations,
                entry.linear_solve.final_residual_norm,
                entry.linear_solve.converged,
                entry
                    .linear_solve
                    .factor_nnz
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            ));
        }
    }
    csv
}

fn team13_identifiable_joint_solver_diagnostics_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "label,converged,state_dimension,theta_dimension,prior_kind,pde_residual_kind,linear_measurement_count,linear_measurement_rows,accepted_iterations,history_len,line_search_residual_evaluations,step_solve_attempts,final_factorizations,cholesky_factor_attempts,cholesky_factor_successes,cholesky_max_shift,cholesky_factorization_seconds,posterior_precision_nnz,posterior_factor_nnz,posterior_residual_norm,final_step_available,final_step_error,final_step_objective,final_step_weighted_residual_norm,final_step_gradient_norm,final_step_step_norm,final_step_directional_derivative,final_step_accepted_alpha,final_step_accepted_objective,final_step_regularization_lambda,final_step_linear_solve_absolute_residual_norm,final_step_linear_solve_relative_residual_norm\n".to_string();
    for solve in team13_identifiable_joint_material_solves(result) {
        let final_step = &solve.final_step;
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{},{},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{:.16e}\n",
            solve.label,
            solve.converged,
            solve.state_dimension,
            solve.theta_dimension,
            solve.prior_kind.as_str(),
            solve.pde_residual_kind.as_str(),
            solve.linear_measurement_count,
            solve.linear_measurement_rows,
            solve.diagnostics.accepted_iterations,
            solve.history.len(),
            solve.diagnostics.line_search_residual_evaluations,
            solve.diagnostics.step_solve_attempts,
            solve.diagnostics.final_factorizations,
            solve.diagnostics.cholesky_factor_attempts,
            solve.diagnostics.cholesky_factor_successes,
            solve.diagnostics.cholesky_max_shift,
            solve.diagnostics.cholesky_factorization_seconds,
            solve.posterior_precision_nnz,
            solve.posterior_factor_nnz,
            solve.posterior_residual_norm,
            final_step.available,
            final_step.error.as_deref().unwrap_or(""),
            final_step.objective,
            final_step.weighted_residual_norm,
            final_step.gradient_norm,
            final_step.step_norm,
            final_step.directional_derivative,
            csv_optional(final_step.accepted_alpha),
            csv_optional(final_step.accepted_objective),
            final_step.regularization_lambda,
            final_step.linear_solve_absolute_residual_norm,
            final_step.linear_solve_relative_residual_norm
        ));
    }
    csv
}

fn team13_identifiable_joint_residual_terms_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "label,name,weighted_norm,unweighted_norm,row_count\n".to_string();
    for solve in team13_identifiable_joint_material_solves(result) {
        for report in &solve.final_residuals {
            csv.push_str(&format!(
                "{},{},{:.16e},{:.16e},{}\n",
                solve.label,
                report.name,
                report.weighted_norm,
                l2_norm(&report.residual),
                report.residual.len()
            ));
        }
    }
    csv
}

fn team13_identifiable_joint_objective_components_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "label,prior_state,prior_material,steel_observation,pde_residual,total,solver_objective,solver_objective_gap\n"
        .to_string();
    for solve in team13_identifiable_joint_material_solves(result) {
        let components = &solve.objective_components;
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{}\n",
            components.label,
            components.prior_state,
            components.prior_material,
            components.steel_observation,
            components.pde_residual,
            components.total,
            csv_optional(components.solver_objective),
            csv_optional(components.solver_objective_gap)
        ));
    }
    csv
}

fn team13_identifiable_joint_fixed_vs_joint_comparison_csv(
    result: &Team13IdentifiableJointMaterialUqResult,
) -> String {
    let mut csv = "label,converged,history_len,prior_kind,pde_residual_kind,linear_measurement_count,linear_measurement_rows,theta_dimension,posterior_residual_norm,weighted_residual_norm,gradient_norm,step_norm,total_objective,posterior_precision_nnz,posterior_factor_nnz\n"
        .to_string();
    for solve in team13_identifiable_joint_material_solves(result) {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{}\n",
            solve.label,
            solve.converged,
            solve.history.len(),
            solve.prior_kind.as_str(),
            solve.pde_residual_kind.as_str(),
            solve.linear_measurement_count,
            solve.linear_measurement_rows,
            solve.theta_dimension,
            solve.posterior_residual_norm,
            solve.final_step.weighted_residual_norm,
            solve.final_step.gradient_norm,
            solve.final_step.step_norm,
            solve.objective_components.total,
            solve.posterior_precision_nnz,
            solve.posterior_factor_nnz
        ));
    }
    csv
}

fn team13_material_only_comparison_csv(result: &Team13MaterialOnlyUqResult) -> String {
    let mut csv =
        "label,theta_0,theta_1,theta_2,steel_rmse_tesla,steel_max_abs_residual_tesla,objective\n"
            .to_string();
    for row in &result.comparison {
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            row.label,
            row.theta[0],
            row.theta[1],
            row.theta[2],
            row.steel_rmse_tesla,
            row.steel_max_abs_residual_tesla,
            row.objective
        ));
    }
    csv
}

fn team13_joint_material_covariance_csv(result: &Team13JointMaterialUqResult) -> String {
    let mut csv = "row,col,covariance,correlation\n".to_string();
    for row in 0..3 {
        for col in 0..3 {
            csv.push_str(&format!(
                "theta_h_{},theta_h_{},{:.16e},{:.16e}\n",
                row,
                col,
                result.material_posterior_covariance[row][col],
                result.material_posterior_correlation[row][col]
            ));
        }
    }
    csv
}

fn team13_joint_material_bh_curve_bands_csv(result: &Team13JointMaterialUqResult) -> String {
    team13_joint_material_bh_curve_bands_csv_from_rows(&result.bh_curve_bands)
}

fn team13_joint_material_bh_curve_bands_csv_from_rows(rows: &[Team13BhCurveBandReport]) -> String {
    let mut csv = "b_tesla,nominal_h_ampere_per_meter,posterior_mean_h_ampere_per_meter,posterior_std_h_ampere_per_meter,lower_2sigma_h_ampere_per_meter,upper_2sigma_h_ampere_per_meter\n"
        .to_string();
    for row in rows {
        csv.push_str(&format!(
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            row.b_tesla,
            row.nominal_h_ampere_per_meter,
            row.posterior_mean_h_ampere_per_meter,
            row.posterior_std_h_ampere_per_meter,
            row.lower_2sigma_h_ampere_per_meter,
            row.upper_2sigma_h_ampere_per_meter
        ));
    }
    csv
}

fn team13_joint_material_steel_patch_csv(result: &Team13JointMaterialUqResult) -> String {
    let mut csv = "name,group,prediction,signed_prediction,observed,residual,total_variance,state_conditional_variance,material_explained_variance,posterior_std,row_nnz\n"
        .to_string();
    for report in &result.steel_patch_reports {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}\n",
            report.name,
            report.group.as_str(),
            report.prediction,
            report.signed_prediction,
            report.observed,
            report.residual,
            report.total_variance,
            report.state_conditional_variance,
            report.material_explained_variance,
            report.posterior_std,
            report.row_nnz
        ));
    }
    csv
}

fn team13_joint_material_solves(
    result: &Team13JointMaterialUqResult,
) -> [&Team13JointMaterialSolveDiagnostics; 2] {
    [&result.fixed_material_solve, &result.joint_material_solve]
}

fn team13_joint_material_history_csv(result: &Team13JointMaterialUqResult) -> String {
    let mut csv = "label,iteration,objective,trial_objective,weighted_residual_norm,gradient_norm,step_norm,alpha,regularization,regularization_lambda,linear_solve_mode,linear_solve_iterations,linear_solve_residual_norm,linear_solve_converged,linear_solve_factor_nnz\n".to_string();
    for solve in team13_joint_material_solves(result) {
        for entry in &solve.history {
            csv.push_str(&format!(
                "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{},{},{:.16e},{},{}\n",
                solve.label,
                entry.iteration,
                entry.objective,
                entry.trial_objective,
                entry.residual_norm,
                entry.gradient_norm,
                entry.step_norm,
                entry.alpha,
                entry.regularization.as_str(),
                entry.regularization_lambda,
                gauss_newton_linear_solve_mode_label(entry.linear_solve.mode),
                entry.linear_solve.iterations,
                entry.linear_solve.final_residual_norm,
                entry.linear_solve.converged,
                entry
                    .linear_solve
                    .factor_nnz
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            ));
        }
    }
    csv
}

fn team13_joint_material_solver_diagnostics_csv(result: &Team13JointMaterialUqResult) -> String {
    let mut csv = "label,converged,state_dimension,theta_dimension,prior_kind,pde_residual_kind,linear_measurement_count,linear_measurement_rows,accepted_iterations,history_len,line_search_residual_evaluations,step_solve_attempts,final_factorizations,metis_cache_hits,metis_cache_misses,cholesky_factor_attempts,cholesky_factor_successes,cholesky_cached_shift_attempts,cholesky_cached_shift_successes,cholesky_shifted_attempts,cholesky_shifted_successes,cholesky_max_shift,cholesky_factorization_seconds,posterior_precision_nnz,posterior_factor_nnz,posterior_residual_norm,final_step_available,final_step_error,final_step_objective,final_step_weighted_residual_norm,final_step_gradient_norm,final_step_step_norm,final_step_directional_derivative,final_step_accepted_alpha,final_step_accepted_objective,final_step_regularization_lambda,final_step_linear_solve_absolute_residual_norm,final_step_linear_solve_relative_residual_norm\n".to_string();
    for solve in team13_joint_material_solves(result) {
        let final_step = &solve.final_step;
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{},{},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{:.16e}\n",
            solve.label,
            solve.converged,
            solve.state_dimension,
            solve.theta_dimension,
            solve.prior_kind.as_str(),
            solve.pde_residual_kind.as_str(),
            solve.linear_measurement_count,
            solve.linear_measurement_rows,
            solve.diagnostics.accepted_iterations,
            solve.history.len(),
            solve.diagnostics.line_search_residual_evaluations,
            solve.diagnostics.step_solve_attempts,
            solve.diagnostics.final_factorizations,
            solve.diagnostics.metis_cache_hits,
            solve.diagnostics.metis_cache_misses,
            solve.diagnostics.cholesky_factor_attempts,
            solve.diagnostics.cholesky_factor_successes,
            solve.diagnostics.cholesky_cached_shift_attempts,
            solve.diagnostics.cholesky_cached_shift_successes,
            solve.diagnostics.cholesky_shifted_attempts,
            solve.diagnostics.cholesky_shifted_successes,
            solve.diagnostics.cholesky_max_shift,
            solve.diagnostics.cholesky_factorization_seconds,
            solve.posterior_precision_nnz,
            solve.posterior_factor_nnz,
            solve.posterior_residual_norm,
            final_step.available,
            final_step.error.as_deref().unwrap_or(""),
            final_step.objective,
            final_step.weighted_residual_norm,
            final_step.gradient_norm,
            final_step.step_norm,
            final_step.directional_derivative,
            csv_optional(final_step.accepted_alpha),
            csv_optional(final_step.accepted_objective),
            final_step.regularization_lambda,
            final_step.linear_solve_absolute_residual_norm,
            final_step.linear_solve_relative_residual_norm
        ));
    }
    csv
}

fn team13_joint_material_residual_terms_csv(result: &Team13JointMaterialUqResult) -> String {
    let mut csv = "label,name,weighted_norm,unweighted_norm,row_count\n".to_string();
    for solve in team13_joint_material_solves(result) {
        for report in &solve.final_residuals {
            csv.push_str(&format!(
                "{},{},{:.16e},{:.16e},{}\n",
                solve.label,
                report.name,
                report.weighted_norm,
                l2_norm(&report.residual),
                report.residual.len()
            ));
        }
    }
    csv
}

fn team13_joint_material_objective_components_csv(result: &Team13JointMaterialUqResult) -> String {
    let mut csv = "label,prior_state,prior_material,steel_observation,pde_residual,total,solver_objective,solver_objective_gap\n"
        .to_string();
    for solve in team13_joint_material_solves(result) {
        let components = &solve.objective_components;
        csv.push_str(&format!(
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{}\n",
            components.label,
            components.prior_state,
            components.prior_material,
            components.steel_observation,
            components.pde_residual,
            components.total,
            csv_optional(components.solver_objective),
            csv_optional(components.solver_objective_gap)
        ));
    }
    csv
}

fn team13_joint_material_fixed_vs_joint_comparison_csv(
    result: &Team13JointMaterialUqResult,
) -> String {
    let mut csv = "label,converged,history_len,prior_kind,pde_residual_kind,linear_measurement_count,linear_measurement_rows,theta_dimension,posterior_residual_norm,weighted_residual_norm,gradient_norm,step_norm,total_objective,posterior_precision_nnz,posterior_factor_nnz\n"
        .to_string();
    for solve in team13_joint_material_solves(result) {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{}\n",
            solve.label,
            solve.converged,
            solve.history.len(),
            solve.prior_kind.as_str(),
            solve.pde_residual_kind.as_str(),
            solve.linear_measurement_count,
            solve.linear_measurement_rows,
            solve.theta_dimension,
            solve.posterior_residual_norm,
            solve.final_step.weighted_residual_norm,
            solve.final_step.gradient_norm,
            solve.final_step.step_norm,
            solve.objective_components.total,
            solve.posterior_precision_nnz,
            solve.posterior_factor_nnz
        ));
    }
    csv
}

fn csv_optional(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.16e}"))
        .unwrap_or_default()
}

fn gauss_newton_linear_solve_mode_label(mode: GaussNewtonLinearSolveMode) -> &'static str {
    match mode {
        GaussNewtonLinearSolveMode::IterativeCg => "iterative-cg",
        GaussNewtonLinearSolveMode::DirectCholesky => "direct-cholesky",
    }
}

fn team13_material_gap_case_summary_csv(result: &Team13MaterialGapUqResult) -> String {
    let mut csv = "gap_label,steel_gap_m,observed_gap,material_label,material_log_iron_nu_scale,normalized_weight,mesh_path,case_output_dir,rmse_vs_observed_gap,mean_patch_std,posterior_converged,deterministic_converged,posterior_precision_nnz,posterior_factor_nnz,field_variance_available\n"
        .to_string();
    for case in &result.case_results {
        csv.push_str(&format!(
            "{},{:.16e},{},{},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{},{},{},{},{}\n",
            case.gap_label,
            case.steel_gap_m,
            case.observed_gap.map(|gap| gap.as_str()).unwrap_or("none"),
            case.material_label,
            case.material_log_iron_nu_scale,
            case.normalized_weight,
            case.operator_result.mesh_path.display(),
            case.operator_result
                .output_dir
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            case.rmse_vs_observed_gap,
            case.mean_patch_std,
            case.operator_result.posterior_converged,
            case.operator_result.deterministic_converged,
            case.operator_result.posterior_precision_nnz,
            case.operator_result.posterior_factor_nnz,
            case.operator_result.field_variance_available,
        ));
    }
    csv
}

fn team13_material_gap_variance_decomposition_csv(result: &Team13MaterialGapUqResult) -> String {
    let mut csv = "name,group,mean_prediction,expected_operator_variance,between_gap_variance,between_material_variance,gap_material_interaction_variance,total_between_case_variance,total_variance,operator_fraction,gap_fraction,material_fraction,interaction_fraction\n"
        .to_string();
    for row in &result.variance_decomposition {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            row.name,
            row.group.as_str(),
            row.mean_prediction,
            row.expected_operator_variance,
            row.between_gap_variance,
            row.between_material_variance,
            row.gap_material_interaction_variance,
            row.total_between_case_variance,
            row.total_variance,
            row.operator_fraction,
            row.gap_fraction,
            row.material_fraction,
            row.interaction_fraction,
        ));
    }
    csv
}

fn write_string_file(path: &Path, contents: &str, label: &str) -> Result<(), String> {
    fs::write(path, contents)
        .map_err(|err| format!("failed to write {label} `{}`: {err}", path.display()))
}

fn team13_operator_uncertainty_summary_csv(result: &Team13OperatorUncertaintyResult) -> String {
    let mean_patch_std = mean_or_nan(
        &result
            .steel_patch_reports
            .iter()
            .map(|report| report.posterior_std)
            .collect::<Vec<_>>(),
    );
    let max_patch_std = result
        .steel_patch_reports
        .iter()
        .map(|report| report.posterior_std)
        .fold(f64::NAN, nan_max);
    let interface_ratio = result
        .region_summaries
        .iter()
        .find(|summary| summary.region == "iron_air_interface_band")
        .map_or(f64::NAN, |summary| summary.variance_ratio_to_iron_bulk);
    let corner_ratio = result
        .region_summaries
        .iter()
        .find(|summary| summary.region == "steel_corner_edge_band")
        .map_or(f64::NAN, |summary| summary.variance_ratio_to_iron_bulk);
    let gap_ratio = result
        .region_summaries
        .iter()
        .find(|summary| summary.region == "central_gap_band")
        .map_or(f64::NAN, |summary| summary.variance_ratio_to_air_bulk);
    let mut csv = "mesh_path,domain,vertices,edges,cells,active_dofs,boundary_edge_dofs,material_kind,material_log_iron_nu_scale,tangent_kind,prior_kind,pde_residual_kind,pde_residual_weighting,pde_variance,include_steel_observations,steel_observation_mode,observation_std_tesla,deterministic_converged,deterministic_residual_norm,posterior_converged,posterior_precision_nnz,posterior_factor_nnz,fill_ratio_vs_lower,field_variance_estimator,field_variance_available,mean_patch_std,max_patch_std,interface_variance_ratio_to_iron_bulk,corner_variance_ratio_to_iron_bulk,gap_variance_ratio_to_air_bulk\n".to_string();
    csv.push_str(&format!(
        "{},{},{},{},{},{},{},{},{:.16e},{},{},{},{},{:.16e},{},{},{:.16e},{},{:.16e},{},{},{},{:.16e},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
        result.mesh_path.display(),
        result.domain_mode.as_str(),
        result.vertices,
        result.edges,
        result.cells,
        result.active_dofs,
        result.boundary_edge_dofs,
        result.material_kind.as_str(),
        result.material_log_iron_nu_scale,
        result.tangent_kind.as_str(),
        result.prior_kind.as_str(),
        result.pde_residual_kind.as_str(),
        result.pde_residual_weighting.as_str(),
        result.pde_variance,
        result.include_steel_observations,
        result.steel_observation_quadrature.as_str(),
        result.observation_std_tesla,
        result.deterministic_converged,
        result.deterministic_residual_norm,
        result.posterior_converged,
        result.posterior_precision_nnz,
        result.posterior_factor_nnz,
        result.fill_ratio_vs_lower,
        result.field_variance_estimator,
        result.field_variance_available,
        mean_patch_std,
        max_patch_std,
        interface_ratio,
        corner_ratio,
        gap_ratio,
    ));
    csv
}

fn team13_operator_region_variance_csv(result: &Team13OperatorUncertaintyResult) -> String {
    let mut csv = "region,count,mean_variance,median_variance,p90_variance,max_variance,mean_std,mean_b_magnitude,mean_gradient_indicator,variance_ratio_to_iron_bulk,variance_ratio_to_air_bulk\n".to_string();
    for summary in &result.region_summaries {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            summary.region,
            summary.count,
            summary.mean_variance,
            summary.median_variance,
            summary.p90_variance,
            summary.max_variance,
            summary.mean_std,
            summary.mean_b_magnitude,
            summary.mean_gradient_indicator,
            summary.variance_ratio_to_iron_bulk,
            summary.variance_ratio_to_air_bulk
        ));
    }
    csv
}

fn team13_operator_steel_patch_variance_csv(result: &Team13OperatorUncertaintyResult) -> String {
    let mut csv = "name,group,prediction,signed_prediction,prior_variance,posterior_variance,posterior_std,observed_g052,observed_g047,residual_g052,residual_g047,abs_residual_over_std_g052,abs_residual_over_std_g047,row_nnz\n".to_string();
    for report in &result.steel_patch_reports {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{}\n",
            report.name,
            report.group.as_str(),
            report.prediction,
            report.signed_prediction,
            report.prior_variance,
            report.posterior_variance,
            report.posterior_std,
            report.observed_g_052,
            report.observed_g_047,
            report.residual_g_052,
            report.residual_g_047,
            report.abs_residual_over_std_g_052,
            report.abs_residual_over_std_g_047,
            report.row_nnz
        ));
    }
    csv
}

fn team13_operator_steel_patch_error_csv(result: &Team13OperatorUncertaintyResult) -> String {
    let mut csv = "name,group,posterior_std,abs_residual_g052,abs_residual_g047,abs_residual_over_std_g052,abs_residual_over_std_g047\n".to_string();
    for report in &result.steel_patch_reports {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            report.name,
            report.group.as_str(),
            report.posterior_std,
            report.residual_g_052.abs(),
            report.residual_g_047.abs(),
            report.abs_residual_over_std_g_052,
            report.abs_residual_over_std_g_047
        ));
    }
    csv
}

fn team13_operator_indicator_correlation_csv(result: &Team13OperatorUncertaintyResult) -> String {
    let mut csv = "indicator,count,pearson_with_variance,pearson_with_std,indicator_mean,mean_variance_indicator_positive,mean_variance_indicator_zero\n".to_string();
    for row in &result.indicator_correlations {
        csv.push_str(&format!(
            "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
            row.indicator,
            row.count,
            row.pearson_with_variance,
            row.pearson_with_std,
            row.indicator_mean,
            row.mean_variance_indicator_positive,
            row.mean_variance_indicator_zero
        ));
    }
    csv
}

fn audit_volume(audit: &Team13RegionAudit, name: &str) -> f64 {
    audit
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .map_or(0.0, |entry| entry.volume)
}

fn audit_current_l2(audit: &Team13RegionAudit, name: &str) -> f64 {
    audit
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .map_or(0.0, |entry| entry.current_l2_norm)
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn run_team13_map_parity_fit(
    label: &str,
    config: &Team13MapParityConfig,
    pde_variance: f64,
    observation_std_tesla: f64,
    model: &ReducedVectorPotentialMagnetostatic3d,
    linear_mean: &[f64],
    truth_map: &[f64],
    exact_prior: &GaussianPriorSpec,
    prior_factor: &SparseCholeskyFactor,
    observations: &Team13SyntheticBenchmarkObservationBuild,
    initial_residual_norm: f64,
    truth_residual_norm: f64,
) -> Result<Team13MapParityRunResult, String> {
    let model_adapter = FeecResidualAdapter::new(model);
    let posterior_problem = NonlinearLaplaceProblem {
        prior: exact_prior.clone(),
        residual_terms: vec![
            NonlinearResidualTerm::zero(
                "team13_map_parity_full_pde_residual",
                &model_adapter,
                GaussianNoiseModel::ScalarVariance(pde_variance),
            ),
            NonlinearResidualTerm {
                name: "team13_map_parity_internal_steel_smooth_magnitude".to_string(),
                model: &observations.assimilated_model,
                observations: observations.assimilated_observations.clone(),
                noise: GaussianNoiseModel::ScalarVariance(
                    observation_std_tesla * observation_std_tesla,
                ),
            },
        ],
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: Vec::new(),
    };
    let posterior = solve_nonlinear_laplace(
        &posterior_problem,
        &GaussNewtonConfig {
            initial_guess: Some(linear_mean.to_vec()),
            max_iterations: config.max_iterations,
            step_tolerance: 1e-10,
            gradient_tolerance: 1e-9,
            max_line_search_steps: 40,
            linear_solve: config.linear_solve,
            step_regularization: config.step_regularization,
            reuse_cholesky_stabilization_shift: true,
            estimate_latent_variance: config.estimate_latent_variance,
            variance: config.variance,
            ..GaussNewtonConfig::default()
        },
    )?;

    let posterior_residual_norm = l2_norm(&model.residual_and_jacobian(&posterior.map)?.residual);
    let initial_steel_predictions = observations
        .assimilated_model
        .smooth_norm_values(linear_mean)?;
    let posterior_steel_predictions = observations
        .assimilated_model
        .smooth_norm_values(&posterior.map)?;
    let internal_steel_reports = synthetic_benchmark_observation_reports(
        &observations.assimilated_specs,
        &observations.assimilated_observations,
        &initial_steel_predictions,
        &posterior_steel_predictions,
    )?;
    let initial_steel_rmse = rmse_from_prediction_pairs(
        &initial_steel_predictions,
        &observations.assimilated_observations,
    )?;
    let posterior_steel_rmse = rmse_from_prediction_pairs(
        &posterior_steel_predictions,
        &observations.assimilated_observations,
    )?;
    let posterior_steel_relative_rmse = relative_rmse_from_prediction_pairs(
        &posterior_steel_predictions,
        &observations.assimilated_observations,
    )?;
    let posterior_steel_max_abs_residual = max_abs_residual_from_prediction_pairs(
        &posterior_steel_predictions,
        &observations.assimilated_observations,
    )?;
    let internal_steel_variances = grouped_norm_sensor_variance_reports(
        &observations.assimilated_specs,
        &observations.assimilated_model,
        prior_factor,
        &posterior,
    )?;
    let published_steel_benchmark_reports = published_steel_reports_from_predictions(
        &observations.assimilated_specs,
        &initial_steel_predictions,
        &posterior_steel_predictions,
    )?;
    let latent_variances_finite = posterior
        .posterior_variance
        .iter()
        .all(|value| value.is_finite());
    let latent_variances_nonnegative = posterior
        .posterior_variance
        .iter()
        .all(|value| *value >= -1e-12);
    let b_quantity_variances_finite = internal_steel_variances
        .iter()
        .all(|report| report.prior_variance.is_finite() && report.posterior_variance.is_finite());
    let b_quantity_variances_nonnegative = internal_steel_variances
        .iter()
        .all(|report| report.prior_variance >= -1e-12 && report.posterior_variance >= -1e-12);
    let all_finite_variances = latent_variances_finite && b_quantity_variances_finite;
    let nonnegative_variances = latent_variances_nonnegative && b_quantity_variances_nonnegative;

    Ok(Team13MapParityRunResult {
        label: label.to_string(),
        pde_variance,
        observation_std_tesla,
        total_residual_rows: model.residual_dimension(),
        steel_observation_count: observations.assimilated_specs.len(),
        initial_relative_error: relative_l2_distance(linear_mean, truth_map)?,
        posterior_relative_error: relative_l2_distance(&posterior.map, truth_map)?,
        initial_residual_norm,
        truth_residual_norm,
        posterior_residual_norm,
        initial_steel_rmse,
        posterior_steel_rmse,
        posterior_steel_relative_rmse,
        posterior_steel_max_abs_residual,
        steel_rmse_improvement_ratio: safe_ratio(posterior_steel_rmse, initial_steel_rmse),
        posterior_converged: posterior.converged,
        all_finite_variances,
        nonnegative_variances,
        b_quantity_variances_finite,
        b_quantity_variances_nonnegative,
        latent_variance_count: posterior.posterior_variance.len(),
        latent_variances_finite,
        latent_variances_nonnegative,
        assembly: posterior.assembly,
        final_factorization: posterior.final_factorization,
        diagnostics: posterior.diagnostics,
        posterior_history: posterior.history,
        internal_steel_reports,
        internal_steel_variances,
        published_steel_benchmark_reports,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_team13_map_style_prior(
    kind: Team13MapParityPriorKind,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    galmats: &MixedGalmats,
    boundary: &EssentialBoundarySpec,
    layout: &DofLayout,
    linear_reluctivity: &InnerProductWeightClosure,
    linear_mean: &FeecVector,
    prior_kappa: f64,
    prior_tau: f64,
    prior_diagonal_shift: f64,
) -> Result<GaussianPriorSpec, String> {
    match kind {
        Team13MapParityPriorKind::ExactPotential => {
            Ok(build_exact_two_form_potential_prior_with_metric(
                topology,
                coords,
                metric,
                layout,
                linear_mean.iter().copied().collect(),
                ExactTwoFormPotentialPriorConfig {
                    kappa: prior_kappa,
                    tau: prior_tau,
                    mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
                    diagonal_shift: prior_diagonal_shift,
                },
            )?
            .spec)
        }
        Team13MapParityPriorKind::OrdinaryMaternAlpha2 => {
            let state_mass_inverse =
                FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
                    topology,
                    metric,
                    coords,
                    None,
                    linear_reluctivity,
                ));
            let linear_system = build_reduced_hodge_laplace_1form_system_with_galmats(
                galmats,
                boundary,
                &state_mass_inverse,
            )?;
            if linear_system.state_dimension() != layout.reduced_dimension() {
                return Err(format!(
                    "ordinary TEAM 13 operator prior dimension {} does not match nonlinear model dimension {}",
                    linear_system.state_dimension(),
                    layout.reduced_dimension()
                ));
            }
            let prior = build_hodge_matern_prior_from_reduced_system_with_params(
                &linear_system,
                linear_mean,
                prior_kappa,
                prior_tau,
            )?;
            add_diagonal_shift_to_gaussian_prior(prior, prior_diagonal_shift)
        }
        Team13MapParityPriorKind::WeakRidge => build_weak_ridge_prior(
            linear_mean.as_slice(),
            prior_tau * prior_tau + prior_diagonal_shift,
        ),
    }
}

fn validate_team13_operator_uncertainty_config(
    config: &Team13OperatorUncertaintyConfig,
) -> Result<(), String> {
    validate_positive_finite(
        config.ampere_turns.abs(),
        "operator uncertainty ampere-turns",
    )?;
    validate_positive_finite(config.pde_variance, "operator uncertainty PDE variance")?;
    validate_positive_finite(
        config.observation_std_tesla,
        "operator uncertainty observation std",
    )?;
    validate_positive_finite(config.prior_kappa, "operator uncertainty prior kappa")?;
    validate_positive_finite(config.prior_tau, "operator uncertainty prior tau")?;
    if !config.material_log_iron_nu_scale.is_finite()
        || !config.material_log_iron_nu_scale.exp().is_finite()
    {
        return Err(
            "operator uncertainty material log iron reluctivity scale must be finite".into(),
        );
    }
    if !config.prior_diagonal_shift.is_finite() || config.prior_diagonal_shift < 0.0 {
        return Err(
            "operator uncertainty prior diagonal shift must be finite and nonnegative".into(),
        );
    }
    if config.truth_max_iterations == 0 {
        return Err("operator uncertainty truth_max_iterations must be >= 1".into());
    }
    if config.field_variance.num_variance_probes == 0 && config.estimate_field_variance {
        return Err("operator uncertainty field variance probes must be >= 1".into());
    }
    Ok(())
}

fn validate_team13_material_gap_uq_config(
    config: &Team13MaterialGapUqConfig,
) -> Result<(), String> {
    let mut operator_config = config.operator.clone();
    operator_config.material_log_iron_nu_scale = 0.0;
    validate_team13_operator_uncertainty_config(&operator_config)?;
    if config.gap_cases.is_empty() {
        return Err("TEAM 13 material/gap UQ requires at least one gap case".to_string());
    }
    if config.material_nodes.is_empty() {
        return Err("TEAM 13 material/gap UQ requires at least one material node".to_string());
    }
    for gap_case in &config.gap_cases {
        if gap_case.label.trim().is_empty() {
            return Err("TEAM 13 material/gap UQ gap labels must be nonempty".to_string());
        }
        validate_positive_finite(gap_case.steel_gap_m, "TEAM 13 steel gap")?;
        validate_positive_finite(gap_case.weight, "TEAM 13 gap weight")?;
        if gap_case.mesh_path.as_os_str().is_empty() {
            return Err(format!(
                "TEAM 13 gap case `{}` has an empty mesh path",
                gap_case.label
            ));
        }
    }
    for material_node in &config.material_nodes {
        if material_node.label.trim().is_empty() {
            return Err("TEAM 13 material/gap UQ material labels must be nonempty".to_string());
        }
        if !material_node.log_iron_nu_scale.is_finite()
            || !material_node.log_iron_nu_scale.exp().is_finite()
        {
            return Err(format!(
                "TEAM 13 material node `{}` has an invalid log iron reluctivity scale",
                material_node.label
            ));
        }
        validate_positive_finite(material_node.weight, "TEAM 13 material node weight")?;
    }
    validate_positive_finite(
        team13_material_gap_case_total_weight(config),
        "TEAM 13 material/gap total weight",
    )?;
    Ok(())
}

fn validate_team13_joint_material_uq_config(
    config: &Team13JointMaterialUqConfig,
) -> Result<(), String> {
    validate_team13_operator_uncertainty_config(&config.operator)?;
    if config.operator.material_kind != Team13NonlinearMaterialKind::NgsolveTabulatedLinear {
        return Err(
            "TEAM 13 joint material UQ currently requires the tabulated NGSolve material law"
                .to_string(),
        );
    }
    if config.operator.tangent_kind != Team13OperatorUncertaintyTangentKind::Nonlinear {
        return Err("TEAM 13 joint material UQ requires nonlinear tangent kind".to_string());
    }
    if !config
        .material_anchor_b_tesla
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err("TEAM 13 material anchors must be finite and nonnegative".to_string());
    }
    if !(config.material_anchor_b_tesla[0] < config.material_anchor_b_tesla[1]
        && config.material_anchor_b_tesla[1] < config.material_anchor_b_tesla[2])
    {
        return Err("TEAM 13 material anchors must be strictly increasing".to_string());
    }
    validate_positive_finite(
        config.material_prior_std,
        "TEAM 13 material prior standard deviation",
    )?;
    if let Some(target) = config.material_prior_target_steel_rms_tesla {
        validate_positive_finite(target, "TEAM 13 material prior target steel RMS")?;
    } else if config.material_prior_calibration
        == Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms
        && config.material_prior_calibration_target
            == Team13MaterialPriorCalibrationTarget::Explicit
    {
        return Err(
            "TEAM 13 material-prior target `explicit` requires material_prior_target_steel_rms_tesla"
                .to_string(),
        );
    }
    validate_positive_finite(
        config.material_prior_std_floor,
        "TEAM 13 material prior standard-deviation floor",
    )?;
    validate_positive_finite(
        config.material_prior_std_ceiling,
        "TEAM 13 material prior standard-deviation ceiling",
    )?;
    if config.material_prior_std_floor > config.material_prior_std_ceiling {
        return Err(
            "TEAM 13 material prior standard-deviation floor must not exceed ceiling".to_string(),
        );
    }
    validate_positive_finite(
        config.magnitude_smoothing_tesla,
        "TEAM 13 joint material magnitude smoothing",
    )?;
    if config.max_iterations == 0 {
        return Err("TEAM 13 joint material max_iterations must be >= 1".to_string());
    }
    Ok(())
}

fn validate_team13_material_only_uq_config(
    config: &Team13MaterialOnlyUqConfig,
) -> Result<(), String> {
    validate_team13_operator_uncertainty_config(&config.operator)?;
    if config.operator.material_kind != Team13NonlinearMaterialKind::NgsolveTabulatedLinear {
        return Err(
            "TEAM 13 material-only UQ currently requires the tabulated NGSolve material law"
                .to_string(),
        );
    }
    if config.operator.tangent_kind != Team13OperatorUncertaintyTangentKind::Nonlinear {
        return Err("TEAM 13 material-only UQ requires nonlinear tangent kind".to_string());
    }
    if !config
        .material_anchor_b_tesla
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err("TEAM 13 material anchors must be finite and nonnegative".to_string());
    }
    if !(config.material_anchor_b_tesla[0] < config.material_anchor_b_tesla[1]
        && config.material_anchor_b_tesla[1] < config.material_anchor_b_tesla[2])
    {
        return Err("TEAM 13 material anchors must be strictly increasing".to_string());
    }
    validate_positive_finite(
        config.material_prior_std,
        "TEAM 13 material-only prior standard deviation",
    )?;
    if let Some(target) = config.material_prior_target_steel_rms_tesla {
        validate_positive_finite(target, "TEAM 13 material-only prior target steel RMS")?;
    } else if config.material_prior_calibration
        == Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms
        && config.material_prior_calibration_target
            == Team13MaterialPriorCalibrationTarget::Explicit
    {
        return Err(
            "TEAM 13 material-only target `explicit` requires material_prior_target_steel_rms_tesla"
                .to_string(),
        );
    }
    validate_positive_finite(
        config.material_prior_std_floor,
        "TEAM 13 material-only prior standard-deviation floor",
    )?;
    validate_positive_finite(
        config.material_prior_std_ceiling,
        "TEAM 13 material-only prior standard-deviation ceiling",
    )?;
    if config.material_prior_std_floor > config.material_prior_std_ceiling {
        return Err(
            "TEAM 13 material-only prior standard-deviation floor must not exceed ceiling"
                .to_string(),
        );
    }
    validate_positive_finite(
        config.magnitude_smoothing_tesla,
        "TEAM 13 material-only magnitude smoothing",
    )?;
    if config.max_iterations == 0 {
        return Err("TEAM 13 material-only max_iterations must be >= 1".to_string());
    }
    if config.max_line_search_steps == 0 {
        return Err("TEAM 13 material-only max_line_search_steps must be >= 1".to_string());
    }
    validate_positive_finite(
        config.max_theta_step_norm,
        "TEAM 13 material-only max theta step norm",
    )?;
    Ok(())
}

fn validate_team13_identifiable_material_uq_config(
    config: &Team13IdentifiableMaterialUqConfig,
) -> Result<(), String> {
    validate_team13_operator_uncertainty_config(&config.operator)?;
    if config.operator.material_kind != Team13NonlinearMaterialKind::NgsolveTabulatedLinear {
        return Err(
            "TEAM 13 identifiable material UQ currently requires the tabulated NGSolve material law"
                .to_string(),
        );
    }
    if config.operator.tangent_kind != Team13OperatorUncertaintyTangentKind::Nonlinear {
        return Err("TEAM 13 identifiable material UQ requires nonlinear tangent kind".to_string());
    }
    if !config
        .material_anchor_b_tesla
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err(
            "TEAM 13 identifiable material anchors must be finite and nonnegative".to_string(),
        );
    }
    if !(config.material_anchor_b_tesla[0] < config.material_anchor_b_tesla[1]
        && config.material_anchor_b_tesla[1] < config.material_anchor_b_tesla[2])
    {
        return Err(
            "TEAM 13 identifiable material anchors must be strictly increasing".to_string(),
        );
    }
    validate_positive_finite(
        config.eta_prior_std,
        "TEAM 13 identifiable material eta prior standard deviation",
    )?;
    if let Some(target) = config.eta_prior_target_steel_rms_tesla {
        validate_positive_finite(
            target,
            "TEAM 13 identifiable material eta prior target steel RMS",
        )?;
    } else if config.eta_prior_calibration
        == Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms
        && config.eta_prior_calibration_target == Team13MaterialPriorCalibrationTarget::Explicit
    {
        return Err(
            "TEAM 13 identifiable material target `explicit` requires eta_prior_target_steel_rms_tesla"
                .to_string(),
        );
    }
    validate_positive_finite(
        config.eta_prior_std_floor,
        "TEAM 13 identifiable material eta prior standard-deviation floor",
    )?;
    validate_positive_finite(
        config.eta_prior_std_ceiling,
        "TEAM 13 identifiable material eta prior standard-deviation ceiling",
    )?;
    if config.eta_prior_std_floor > config.eta_prior_std_ceiling {
        return Err(
            "TEAM 13 identifiable material eta prior standard-deviation floor must not exceed ceiling"
                .to_string(),
        );
    }
    validate_positive_finite(
        config.svd_relative_tolerance,
        "TEAM 13 identifiable material SVD relative tolerance",
    )?;
    validate_positive_finite(
        config.svd_absolute_tolerance,
        "TEAM 13 identifiable material SVD absolute tolerance",
    )?;
    validate_positive_finite(
        config.perturbation_rms_fraction_of_gap,
        "TEAM 13 identifiable material perturbation RMS fraction",
    )?;
    if config.continuation_steps == 0 {
        return Err("TEAM 13 identifiable material continuation_steps must be >= 1".to_string());
    }
    validate_positive_finite(
        config.magnitude_smoothing_tesla,
        "TEAM 13 identifiable material magnitude smoothing",
    )?;
    if config.max_iterations == 0 {
        return Err("TEAM 13 identifiable material max_iterations must be >= 1".to_string());
    }
    if config.max_line_search_steps == 0 {
        return Err("TEAM 13 identifiable material max_line_search_steps must be >= 1".to_string());
    }
    validate_positive_finite(
        config.max_eta_step_norm,
        "TEAM 13 identifiable material max eta step norm",
    )?;
    Ok(())
}

fn validate_team13_identifiable_joint_material_uq_config(
    config: &Team13IdentifiableJointMaterialUqConfig,
) -> Result<(), String> {
    validate_team13_operator_uncertainty_config(&config.operator)?;
    if config.operator.material_kind != Team13NonlinearMaterialKind::NgsolveTabulatedLinear {
        return Err(
            "TEAM 13 identifiable joint material UQ currently requires the tabulated NGSolve material law"
                .to_string(),
        );
    }
    if config.operator.tangent_kind != Team13OperatorUncertaintyTangentKind::Nonlinear {
        return Err(
            "TEAM 13 identifiable joint material UQ requires nonlinear tangent kind".to_string(),
        );
    }
    if !config
        .material_anchor_b_tesla
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err(
            "TEAM 13 identifiable joint material anchors must be finite and nonnegative"
                .to_string(),
        );
    }
    if !(config.material_anchor_b_tesla[0] < config.material_anchor_b_tesla[1]
        && config.material_anchor_b_tesla[1] < config.material_anchor_b_tesla[2])
    {
        return Err(
            "TEAM 13 identifiable joint material anchors must be strictly increasing".to_string(),
        );
    }
    validate_positive_finite(
        config.eta_prior_std,
        "TEAM 13 identifiable joint material eta prior standard deviation",
    )?;
    if let Some(target) = config.eta_prior_target_steel_rms_tesla {
        validate_positive_finite(
            target,
            "TEAM 13 identifiable joint material eta prior target steel RMS",
        )?;
    } else if config.eta_prior_calibration
        == Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms
        && config.eta_prior_calibration_target == Team13MaterialPriorCalibrationTarget::Explicit
    {
        return Err(
            "TEAM 13 identifiable joint material target `explicit` requires eta_prior_target_steel_rms_tesla"
                .to_string(),
        );
    }
    validate_positive_finite(
        config.eta_prior_std_floor,
        "TEAM 13 identifiable joint material eta prior standard-deviation floor",
    )?;
    validate_positive_finite(
        config.eta_prior_std_ceiling,
        "TEAM 13 identifiable joint material eta prior standard-deviation ceiling",
    )?;
    if config.eta_prior_std_floor > config.eta_prior_std_ceiling {
        return Err(
            "TEAM 13 identifiable joint material eta prior standard-deviation floor must not exceed ceiling"
                .to_string(),
        );
    }
    validate_positive_finite(
        config.svd_relative_tolerance,
        "TEAM 13 identifiable joint material SVD relative tolerance",
    )?;
    validate_positive_finite(
        config.svd_absolute_tolerance,
        "TEAM 13 identifiable joint material SVD absolute tolerance",
    )?;
    validate_positive_finite(
        config.perturbation_rms_fraction_of_gap,
        "TEAM 13 identifiable joint material perturbation RMS fraction",
    )?;
    if config.continuation_steps == 0 {
        return Err(
            "TEAM 13 identifiable joint material continuation_steps must be >= 1".to_string(),
        );
    }
    validate_positive_finite(
        config.magnitude_smoothing_tesla,
        "TEAM 13 identifiable joint material magnitude smoothing",
    )?;
    if config.max_iterations == 0 {
        return Err("TEAM 13 identifiable joint material max_iterations must be >= 1".to_string());
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Team13MaterialShapeAugmentedResidualModel {
    model: ReducedVectorPotentialMagnetostatic3d,
    anchor_b_tesla: [f64; 3],
    reference_layout: DofLayout,
    residual_dimension: usize,
}

impl Team13MaterialShapeAugmentedResidualModel {
    fn new(
        model: ReducedVectorPotentialMagnetostatic3d,
        anchor_b_tesla: [f64; 3],
        reference_layout: DofLayout,
    ) -> Result<Self, String> {
        if model.layout().active_dofs != reference_layout.active_dofs {
            return Err("TEAM13 augmented material model reference layout mismatch".to_string());
        }
        Ok(Self {
            anchor_b_tesla,
            residual_dimension: model.residual_dimension(),
            model,
            reference_layout,
        })
    }

    fn state_count(&self) -> usize {
        self.reference_layout.reduced_dimension()
    }

    fn split_state<'a>(&self, state: &'a [f64]) -> Result<(&'a [f64], [f64; 3]), String> {
        if state.len() != self.state_dimension() {
            return Err(format!(
                "TEAM13 augmented material state length {} must be {}",
                state.len(),
                self.state_dimension()
            ));
        }
        let n_state = self.state_count();
        Ok((
            &state[..n_state],
            [state[n_state], state[n_state + 1], state[n_state + 2]],
        ))
    }

    fn material_for_theta(&self, theta: [f64; 3]) -> Result<Team13TabulatedReluctivityLaw, String> {
        build_team13_tabulated_material_with_log_h_shape(self.anchor_b_tesla, theta)
    }
}

impl NonlinearResidualModel for Team13MaterialShapeAugmentedResidualModel {
    fn state_dimension(&self) -> usize {
        self.state_count() + 3
    }

    fn residual_dimension(&self) -> usize {
        self.residual_dimension
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        let (a_state, theta) = self.split_state(state)?;
        let material = self.material_for_theta(theta)?;
        self.model.residual_with_material(&material, a_state)
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        let (a_state, theta) = self.split_state(state)?;
        let material = self.material_for_theta(theta)?;
        let base = self
            .model
            .residual_and_jacobian_with_material(&material, a_state)?;
        base.validate(self.residual_dimension(), self.state_count())?;

        let mut triplets = base
            .jacobian
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col,
                value: *value,
            })
            .collect::<Vec<_>>();
        let material_columns = self.model.material_sensitivity_columns_with_material(
            &material,
            a_state,
            3,
            |material, point, s| Ok(material.d_nu_d_log_h_shape_values(point, s).to_vec()),
        )?;
        for (row, theta_col, value) in material_columns.triplet_iter() {
            if value.abs() > EPS {
                triplets.push(SparseTriplet {
                    row,
                    col: self.state_count() + theta_col,
                    value: *value,
                });
            }
        }

        Ok(NonlinearResidualEvaluation {
            residual: base.residual.as_slice().to_vec(),
            jacobian: SparseTripletMatrix::from_triplets(
                self.residual_dimension(),
                self.state_dimension(),
                triplets,
            ),
        })
    }
}

#[derive(Debug, Clone)]
struct Team13IdentifiableJointMaterialResidualModel {
    model: ReducedVectorPotentialMagnetostatic3d,
    anchor_b_tesla: [f64; 3],
    theta_bias: [f64; 3],
    basis: Vec<[f64; 3]>,
    reference_layout: DofLayout,
    residual_dimension: usize,
}

impl Team13IdentifiableJointMaterialResidualModel {
    fn new(
        model: ReducedVectorPotentialMagnetostatic3d,
        anchor_b_tesla: [f64; 3],
        theta_bias: [f64; 3],
        basis: Vec<[f64; 3]>,
        reference_layout: DofLayout,
    ) -> Result<Self, String> {
        if model.layout().active_dofs != reference_layout.active_dofs {
            return Err(
                "TEAM13 identifiable joint material model reference layout mismatch".to_string(),
            );
        }
        if basis.is_empty() {
            return Err("TEAM13 identifiable joint material basis must not be empty".to_string());
        }
        Ok(Self {
            anchor_b_tesla,
            theta_bias,
            basis,
            residual_dimension: model.residual_dimension(),
            model,
            reference_layout,
        })
    }

    fn state_count(&self) -> usize {
        self.reference_layout.reduced_dimension()
    }

    fn eta_count(&self) -> usize {
        self.basis.len()
    }

    fn split_state<'a>(&self, state: &'a [f64]) -> Result<(&'a [f64], &'a [f64]), String> {
        if state.len() != self.state_dimension() {
            return Err(format!(
                "TEAM13 identifiable joint material state length {} must be {}",
                state.len(),
                self.state_dimension()
            ));
        }
        let n_state = self.state_count();
        Ok((&state[..n_state], &state[n_state..]))
    }

    fn theta_for_eta(&self, eta: &[f64]) -> Result<[f64; 3], String> {
        team13_eta_to_theta(self.theta_bias, &self.basis, eta)
    }

    fn material_for_eta(&self, eta: &[f64]) -> Result<Team13TabulatedReluctivityLaw, String> {
        build_team13_tabulated_material_with_log_h_shape(
            self.anchor_b_tesla,
            self.theta_for_eta(eta)?,
        )
    }
}

impl NonlinearResidualModel for Team13IdentifiableJointMaterialResidualModel {
    fn state_dimension(&self) -> usize {
        self.state_count() + self.eta_count()
    }

    fn residual_dimension(&self) -> usize {
        self.residual_dimension
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        let (a_state, eta) = self.split_state(state)?;
        let material = self.material_for_eta(eta)?;
        self.model.residual_with_material(&material, a_state)
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        let (a_state, eta) = self.split_state(state)?;
        let material = self.material_for_eta(eta)?;
        let base = self
            .model
            .residual_and_jacobian_with_material(&material, a_state)?;
        base.validate(self.residual_dimension(), self.state_count())?;

        let mut triplets = base
            .jacobian
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col,
                value: *value,
            })
            .collect::<Vec<_>>();
        let material_columns = self.model.material_sensitivity_columns_with_material(
            &material,
            a_state,
            3,
            |material, point, s| Ok(material.d_nu_d_log_h_shape_values(point, s).to_vec()),
        )?;
        let mut eta_columns = vec![vec![0.0; self.eta_count()]; self.residual_dimension()];
        for (row, theta_col, value) in material_columns.triplet_iter() {
            for eta_col in 0..self.eta_count() {
                eta_columns[row][eta_col] += *value * self.basis[eta_col][theta_col];
            }
        }
        for (row, values) in eta_columns.iter().enumerate() {
            for (eta_col, value) in values.iter().enumerate() {
                if value.abs() > EPS {
                    triplets.push(SparseTriplet {
                        row,
                        col: self.state_count() + eta_col,
                        value: *value,
                    });
                }
            }
        }

        Ok(NonlinearResidualEvaluation {
            residual: base.residual.as_slice().to_vec(),
            jacobian: SparseTripletMatrix::from_triplets(
                self.residual_dimension(),
                self.state_dimension(),
                triplets,
            ),
        })
    }
}

#[derive(Clone)]
struct Team13FixedThetaForwardModel {
    model: ReducedVectorPotentialMagnetostatic3d,
    material: Team13TabulatedReluctivityLaw,
}

impl NonlinearResidualModel for Team13FixedThetaForwardModel {
    fn state_dimension(&self) -> usize {
        self.model.reduced_dimension()
    }

    fn residual_dimension(&self) -> usize {
        self.model.residual_dimension()
    }

    fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        self.model.residual_with_material(&self.material, state)
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        let evaluation = self
            .model
            .residual_and_jacobian_with_material(&self.material, state)?;
        Ok(NonlinearResidualEvaluation {
            residual: evaluation.residual.as_slice().to_vec(),
            jacobian: csr_to_triplet(&evaluation.jacobian),
        })
    }
}

struct Team13MaterialOnlyForwardContext {
    model: ReducedVectorPotentialMagnetostatic3d,
    augmented_model: Team13MaterialShapeAugmentedResidualModel,
    patch_operator: Team13ReducedSteelPatchOperator,
    anchors: [f64; 3],
    observations: [f64; TEAM13_OBSERVATION_COUNT],
    smoothing: f64,
    observation_std_tesla: f64,
    forward_max_iterations: usize,
    linear_solve: GaussNewtonLinearSolve,
}

#[derive(Clone)]
struct Team13MaterialOnlyThetaEvaluation {
    forward_state: Vec<f64>,
    forward_converged: bool,
    forward_residual_norm: f64,
    residual: Vec<f64>,
    jacobian: Vec<[f64; 3]>,
}

struct Team13MaterialOnlyThetaSolve {
    theta: [f64; 3],
    converged: bool,
    final_evaluation: Team13MaterialOnlyThetaEvaluation,
    final_hessian: [[f64; 3]; 3],
    objective_components: Team13MaterialOnlyObjectiveComponents,
    history: Vec<Team13MaterialOnlyIteration>,
}

#[derive(Clone)]
struct Team13IdentifiableEtaEvaluation {
    theta: [f64; 3],
    forward_state: Vec<f64>,
    forward_converged: bool,
    forward_residual_norm: f64,
    residual: Vec<f64>,
    jacobian: Vec<Vec<f64>>,
}

struct Team13IdentifiableEtaSolve {
    eta: Vec<f64>,
    converged: bool,
    final_evaluation: Team13IdentifiableEtaEvaluation,
    final_hessian: Vec<Vec<f64>>,
    objective_components: Team13MaterialOnlyObjectiveComponents,
    history: Vec<Team13MaterialOnlyIteration>,
}

struct Team13IdentifiableMaterialBasis {
    singular_values: [f64; 3],
    retained_modes: Vec<[f64; 3]>,
    mode_reports: Vec<Team13IdentifiableMaterialModeReport>,
}

struct Symmetric3EigenDecomposition {
    eigenvalues: [f64; 3],
    eigenvectors: [[f64; 3]; 3],
}

impl Team13MaterialOnlyForwardContext {
    fn evaluate_known_forward(
        &self,
        theta: [f64; 3],
        forward_state: Vec<f64>,
        forward_converged: bool,
        forward_residual_norm: f64,
        include_jacobian: bool,
    ) -> Result<Team13MaterialOnlyThetaEvaluation, String> {
        let (predictions, _) =
            team13_smooth_steel_predictions(&self.patch_operator, &forward_state, self.smoothing)?;
        let residual = predictions
            .iter()
            .zip(self.observations.iter())
            .map(|(prediction, observation)| *prediction - *observation)
            .collect::<Vec<_>>();
        let jacobian = if include_jacobian {
            team13_material_steel_output_jacobian(
                &self.augmented_model,
                &forward_state,
                theta,
                &self.patch_operator,
                self.smoothing,
            )?
        } else {
            vec![[0.0; 3]; predictions.len()]
        };
        Ok(Team13MaterialOnlyThetaEvaluation {
            forward_state,
            forward_converged,
            forward_residual_norm,
            residual,
            jacobian,
        })
    }

    fn evaluate(
        &self,
        theta: [f64; 3],
        initial_forward_state: &[f64],
        include_jacobian: bool,
    ) -> Result<Team13MaterialOnlyThetaEvaluation, String> {
        let forward = solve_team13_material_only_forward(
            &self.model,
            self.anchors,
            theta,
            initial_forward_state.to_vec(),
            self.forward_max_iterations,
            self.linear_solve,
        )?;
        if !forward.converged {
            return Err(format!(
                "TEAM 13 material-only forward solve did not converge at theta=[{:.6e}, {:.6e}, {:.6e}] (residual={:.6e})",
                theta[0], theta[1], theta[2], forward.residual_norm
            ));
        }
        self.evaluate_known_forward(
            theta,
            forward.solution,
            forward.converged,
            forward.residual_norm,
            include_jacobian,
        )
    }
}

fn solve_team13_material_only_forward(
    model: &ReducedVectorPotentialMagnetostatic3d,
    anchors: [f64; 3],
    theta: [f64; 3],
    initial: Vec<f64>,
    max_iterations: usize,
    linear_solve: GaussNewtonLinearSolve,
) -> Result<feg_infer::nonlinear::SquareNewtonResult, String> {
    let material = build_team13_tabulated_material_with_log_h_shape(anchors, theta)?;
    let fixed_theta_model = Team13FixedThetaForwardModel {
        model: model.clone(),
        material,
    };
    solve_team13_forward_newton(&fixed_theta_model, initial, max_iterations, linear_solve)
}

fn team13_smooth_steel_predictions(
    patch_operator: &Team13ReducedSteelPatchOperator,
    state: &[f64],
    smoothing: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    validate_positive_finite(smoothing, "TEAM 13 material-only magnitude smoothing")?;
    let signed = patch_operator
        .operator
        .apply(&GmrfVector::from_vec(state.to_vec()))
        .map_err(|err| err.to_string())?;
    let signed_predictions = signed
        .iter()
        .zip(patch_operator.bias.iter())
        .map(|(value, bias)| *value + *bias)
        .collect::<Vec<_>>();
    let predictions = signed_predictions
        .iter()
        .map(|value| (value * value + smoothing * smoothing).sqrt())
        .collect::<Vec<_>>();
    Ok((predictions, signed_predictions))
}

fn team13_material_steel_output_jacobian(
    augmented_model: &Team13MaterialShapeAugmentedResidualModel,
    state: &[f64],
    theta: [f64; 3],
    patch_operator: &Team13ReducedSteelPatchOperator,
    smoothing: f64,
) -> Result<Vec<[f64; 3]>, String> {
    let state_sensitivities =
        team13_state_sensitivities_from_augmented_jacobian(augmented_model, state, theta)?;
    let (_, signed_predictions) =
        team13_smooth_steel_predictions(patch_operator, state, smoothing)?;
    team13_smooth_magnitude_jacobian_from_state_sensitivities(
        &patch_operator.operator.rows,
        &signed_predictions,
        &state_sensitivities,
        smoothing,
    )
}

fn team13_smooth_magnitude_jacobian_from_state_sensitivities(
    rows: &[Vec<(usize, f64)>],
    signed_predictions: &[f64],
    state_sensitivities: &[Vec<f64>],
    smoothing: f64,
) -> Result<Vec<[f64; 3]>, String> {
    if state_sensitivities.len() != 3 {
        return Err(
            "TEAM 13 material output sensitivity requires exactly three theta columns".to_string(),
        );
    }
    if rows.len() != signed_predictions.len() {
        return Err(format!(
            "TEAM 13 material output rows {} must match signed prediction count {}",
            rows.len(),
            signed_predictions.len()
        ));
    }
    let mut output = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let signed_prediction = signed_predictions[row_index];
        let smooth = (signed_prediction * signed_prediction + smoothing * smoothing).sqrt();
        let magnitude_derivative = safe_ratio(signed_prediction, smooth);
        let mut output_row = [0.0; 3];
        for theta_index in 0..3 {
            output_row[theta_index] = magnitude_derivative
                * row
                    .iter()
                    .map(|(col, value)| *value * state_sensitivities[theta_index][*col])
                    .sum::<f64>();
        }
        output.push(output_row);
    }
    Ok(output)
}

fn team13_state_sensitivities_from_augmented_jacobian(
    augmented_model: &Team13MaterialShapeAugmentedResidualModel,
    state: &[f64],
    theta: [f64; 3],
) -> Result<Vec<Vec<f64>>, String> {
    let n_state = augmented_model.state_count();
    if state.len() != n_state {
        return Err(format!(
            "TEAM 13 material sensitivity state length {} must match reduced state dimension {}",
            state.len(),
            n_state
        ));
    }
    let mut joint_state = state.to_vec();
    joint_state.extend(theta);
    let evaluation = augmented_model.residual_and_jacobian(&joint_state)?;
    team13_state_sensitivities_from_augmented_evaluation(&evaluation, n_state)
}

fn team13_state_sensitivities_from_augmented_evaluation(
    evaluation: &NonlinearResidualEvaluation,
    n_state: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if evaluation.jacobian.nrows() != n_state {
        return Err(format!(
            "TEAM 13 material sensitivity hard-PDE Jacobian must be square; got {} rows and {} state columns",
            evaluation.jacobian.nrows(),
            n_state
        ));
    }
    let mut state_triplets = Vec::new();
    let mut material_columns = vec![vec![0.0; n_state]; 3];
    for (row, col, value) in evaluation.jacobian.triplet_iter() {
        if col < n_state {
            state_triplets.push(SparseTriplet { row, col, value });
        } else if col < n_state + 3 {
            material_columns[col - n_state][row] += value;
        }
    }
    let state_jacobian = SparseTripletMatrix::from_triplets(n_state, n_state, state_triplets);
    let factor = sparse_from_core(&state_jacobian)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factorize TEAM 13 hard-PDE state Jacobian: {err}"))?;
    let mut state_sensitivities = Vec::with_capacity(3);
    for column in material_columns {
        let rhs = column.into_iter().map(|value| -value).collect::<Vec<_>>();
        let solution = factor.solve(&GmrfVector::from_vec(rhs)).map_err(|err| {
            format!("failed to solve TEAM 13 hard-PDE material state sensitivity: {err}")
        })?;
        state_sensitivities.push(solution.iter().copied().collect::<Vec<_>>());
    }
    Ok(state_sensitivities)
}

fn solve_team13_material_only_theta(
    context: &Team13MaterialOnlyForwardContext,
    initial_evaluation: Team13MaterialOnlyThetaEvaluation,
    material_prior_std: f64,
    max_iterations: usize,
    max_line_search_steps: usize,
    max_theta_step_norm: f64,
    step_regularization: GaussNewtonStepRegularization,
) -> Result<Team13MaterialOnlyThetaSolve, String> {
    validate_positive_finite(material_prior_std, "TEAM 13 material-only prior std")?;
    validate_positive_finite(
        context.observation_std_tesla,
        "TEAM 13 material-only observation std",
    )?;
    if max_iterations == 0 {
        return Err("TEAM 13 material-only max_iterations must be >= 1".to_string());
    }
    if max_line_search_steps == 0 {
        return Err("TEAM 13 material-only max_line_search_steps must be >= 1".to_string());
    }
    validate_positive_finite(
        max_theta_step_norm,
        "TEAM 13 material-only max theta step norm",
    )?;

    let prior_precision = 1.0 / (material_prior_std * material_prior_std);
    let observation_variance = context.observation_std_tesla * context.observation_std_tesla;
    let mut theta = [0.0; 3];
    let mut evaluation = initial_evaluation;
    let mut history = Vec::new();
    let mut converged = false;

    for iteration in 0..max_iterations {
        let objective_components = team13_material_only_objective_components(
            theta,
            prior_precision,
            &evaluation,
            observation_variance,
        );
        let gradient = team13_material_only_gradient(
            theta,
            prior_precision,
            &evaluation,
            observation_variance,
        );
        let gradient_norm = vector3_norm(gradient);
        let hessian = team13_material_only_hessian(
            prior_precision,
            &evaluation.jacobian,
            observation_variance,
        );
        if gradient_norm <= 1.0e-9 {
            converged = true;
            break;
        }

        let mut accepted = None;
        for lambda in team13_material_only_lm_lambdas(step_regularization) {
            let regularized = add_diagonal_3x3(hessian, lambda);
            let Some(inverse) = invert_3x3(regularized) else {
                continue;
            };
            let mut step = mat3_vec_mul(inverse, [-gradient[0], -gradient[1], -gradient[2]]);
            let full_step_norm = vector3_norm(step);
            if full_step_norm > max_theta_step_norm {
                let scale = max_theta_step_norm / full_step_norm;
                step = [scale * step[0], scale * step[1], scale * step[2]];
            }
            let capped_step_norm = vector3_norm(step);
            if capped_step_norm <= 1.0e-10 {
                converged = true;
                accepted = Some((
                    theta,
                    evaluation.clone(),
                    objective_components.total,
                    0.0,
                    lambda,
                    0.0,
                ));
                break;
            }
            let directional_derivative =
                gradient[0] * step[0] + gradient[1] * step[1] + gradient[2] * step[2];
            if directional_derivative >= 0.0 {
                continue;
            }
            let mut alpha = 1.0;
            for _ in 0..=max_line_search_steps {
                let trial_theta = [
                    theta[0] + alpha * step[0],
                    theta[1] + alpha * step[1],
                    theta[2] + alpha * step[2],
                ];
                eprintln!(
                    "TEAM 13 material-only UQ: iteration {iteration} trial theta=[{:.6e}, {:.6e}, {:.6e}] alpha={:.3e} lambda={:.3e}",
                    trial_theta[0], trial_theta[1], trial_theta[2], alpha, lambda
                );
                let trial_value =
                    match context.evaluate(trial_theta, &evaluation.forward_state, false) {
                        Ok(value) => value,
                        Err(_) => {
                            alpha *= 0.5;
                            continue;
                        }
                    };
                let trial_components = team13_material_only_objective_components(
                    trial_theta,
                    prior_precision,
                    &trial_value,
                    observation_variance,
                );
                if trial_components.total
                    <= objective_components.total + 1.0e-4 * alpha * directional_derivative
                {
                    let trial_evaluation =
                        context.evaluate(trial_theta, &trial_value.forward_state, true)?;
                    accepted = Some((
                        trial_theta,
                        trial_evaluation,
                        trial_components.total,
                        alpha,
                        lambda,
                        alpha * capped_step_norm,
                    ));
                    break;
                }
                alpha *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }

        let Some((trial_theta, trial_evaluation, trial_objective, alpha, lambda, step_norm)) =
            accepted
        else {
            return Err(format!(
                "TEAM 13 material-only theta line search failed at iteration {iteration} (objective={:.6e}, gradient_norm={:.6e})",
                objective_components.total, gradient_norm
            ));
        };
        history.push(Team13MaterialOnlyIteration {
            iteration,
            objective: objective_components.total,
            trial_objective,
            steel_weighted_residual_norm: team13_material_only_weighted_residual_norm(
                &trial_evaluation,
                context.observation_std_tesla,
            ),
            gradient_norm,
            step_norm,
            alpha,
            regularization_lambda: lambda,
            forward_residual_norm: evaluation.forward_residual_norm,
            trial_forward_residual_norm: trial_evaluation.forward_residual_norm,
        });
        theta = trial_theta;
        evaluation = trial_evaluation;
        if step_norm <= 1.0e-10 {
            converged = true;
            break;
        }
    }

    let final_hessian =
        team13_material_only_hessian(prior_precision, &evaluation.jacobian, observation_variance);
    let objective_components = team13_material_only_objective_components(
        theta,
        prior_precision,
        &evaluation,
        observation_variance,
    );
    Ok(Team13MaterialOnlyThetaSolve {
        theta,
        converged,
        final_evaluation: evaluation,
        final_hessian,
        objective_components,
        history,
    })
}

fn team13_identifiable_eta_evaluation_from_theta_evaluation(
    theta_bias: &[f64; 3],
    basis: &[[f64; 3]],
    eta: &[f64],
    theta_evaluation: Team13MaterialOnlyThetaEvaluation,
) -> Result<Team13IdentifiableEtaEvaluation, String> {
    let theta = team13_eta_to_theta(*theta_bias, basis, eta)?;
    let jacobian = team13_transform_theta_jacobian_to_eta(&theta_evaluation.jacobian, basis)?;
    Ok(Team13IdentifiableEtaEvaluation {
        theta,
        forward_state: theta_evaluation.forward_state,
        forward_converged: theta_evaluation.forward_converged,
        forward_residual_norm: theta_evaluation.forward_residual_norm,
        residual: theta_evaluation.residual,
        jacobian,
    })
}

fn team13_identifiable_evaluate_eta(
    context: &Team13MaterialOnlyForwardContext,
    theta_bias: [f64; 3],
    basis: &[[f64; 3]],
    eta: &[f64],
    initial_forward_state: &[f64],
    include_jacobian: bool,
) -> Result<Team13IdentifiableEtaEvaluation, String> {
    let theta = team13_eta_to_theta(theta_bias, basis, eta)?;
    let theta_evaluation = context.evaluate(theta, initial_forward_state, include_jacobian)?;
    team13_identifiable_eta_evaluation_from_theta_evaluation(
        &theta_bias,
        basis,
        eta,
        theta_evaluation,
    )
}

fn solve_team13_identifiable_eta(
    context: &Team13MaterialOnlyForwardContext,
    theta_bias: [f64; 3],
    basis: &[[f64; 3]],
    initial_evaluation: Team13IdentifiableEtaEvaluation,
    eta_prior_std: f64,
    max_iterations: usize,
    max_line_search_steps: usize,
    max_eta_step_norm: f64,
    step_regularization: GaussNewtonStepRegularization,
) -> Result<Team13IdentifiableEtaSolve, String> {
    validate_positive_finite(eta_prior_std, "TEAM 13 identifiable material eta prior std")?;
    validate_positive_finite(
        context.observation_std_tesla,
        "TEAM 13 identifiable material observation std",
    )?;
    if basis.is_empty() {
        return Err("TEAM 13 identifiable material eta basis must not be empty".to_string());
    }
    if max_iterations == 0 {
        return Err("TEAM 13 identifiable material max_iterations must be >= 1".to_string());
    }
    if max_line_search_steps == 0 {
        return Err("TEAM 13 identifiable material max_line_search_steps must be >= 1".to_string());
    }
    validate_positive_finite(
        max_eta_step_norm,
        "TEAM 13 identifiable material max eta step norm",
    )?;

    let rank = basis.len();
    let prior_precision = 1.0 / (eta_prior_std * eta_prior_std);
    let observation_variance = context.observation_std_tesla * context.observation_std_tesla;
    let mut eta = vec![0.0; rank];
    let mut evaluation = initial_evaluation;
    let mut history = Vec::new();
    let mut converged = false;

    for iteration in 0..max_iterations {
        let objective_components = team13_identifiable_eta_objective_components(
            &eta,
            prior_precision,
            &evaluation,
            observation_variance,
        );
        let gradient = team13_identifiable_eta_gradient(
            &eta,
            prior_precision,
            &evaluation,
            observation_variance,
        )?;
        let gradient_norm = dense_vector_norm(&gradient);
        let hessian = team13_identifiable_eta_hessian(
            rank,
            prior_precision,
            &evaluation.jacobian,
            observation_variance,
        )?;
        if gradient_norm <= 1.0e-9 {
            converged = true;
            break;
        }

        let mut accepted = None;
        for lambda in team13_material_only_lm_lambdas(step_regularization) {
            let regularized = add_diagonal_dense(&hessian, lambda)?;
            let rhs = gradient.iter().map(|value| -*value).collect::<Vec<_>>();
            let Some(mut step) = solve_dense_linear_system(&regularized, &rhs) else {
                continue;
            };
            let full_step_norm = dense_vector_norm(&step);
            if full_step_norm > max_eta_step_norm {
                let scale = max_eta_step_norm / full_step_norm;
                for value in &mut step {
                    *value *= scale;
                }
            }
            let capped_step_norm = dense_vector_norm(&step);
            if capped_step_norm <= 1.0e-10 {
                converged = true;
                accepted = Some((
                    eta.clone(),
                    evaluation.clone(),
                    objective_components.total,
                    0.0,
                    lambda,
                    0.0,
                ));
                break;
            }
            let directional_derivative = dot_dense(&gradient, &step)?;
            if directional_derivative >= 0.0 {
                continue;
            }
            let mut alpha = 1.0;
            for _ in 0..=max_line_search_steps {
                let trial_eta = eta
                    .iter()
                    .zip(step.iter())
                    .map(|(value, delta)| value + alpha * delta)
                    .collect::<Vec<_>>();
                let trial_theta = team13_eta_to_theta(theta_bias, basis, &trial_eta)?;
                eprintln!(
                    "TEAM 13 identifiable material UQ: iteration {iteration} trial eta_norm={:.6e} theta=[{:.6e}, {:.6e}, {:.6e}] alpha={:.3e} lambda={:.3e}",
                    dense_vector_norm(&trial_eta),
                    trial_theta[0],
                    trial_theta[1],
                    trial_theta[2],
                    alpha,
                    lambda
                );
                let trial_value = match team13_identifiable_evaluate_eta(
                    context,
                    theta_bias,
                    basis,
                    &trial_eta,
                    &evaluation.forward_state,
                    false,
                ) {
                    Ok(value) => value,
                    Err(_) => {
                        alpha *= 0.5;
                        continue;
                    }
                };
                let trial_components = team13_identifiable_eta_objective_components(
                    &trial_eta,
                    prior_precision,
                    &trial_value,
                    observation_variance,
                );
                if trial_components.total
                    <= objective_components.total + 1.0e-4 * alpha * directional_derivative
                {
                    let trial_evaluation = team13_identifiable_evaluate_eta(
                        context,
                        theta_bias,
                        basis,
                        &trial_eta,
                        &trial_value.forward_state,
                        true,
                    )?;
                    accepted = Some((
                        trial_eta,
                        trial_evaluation,
                        trial_components.total,
                        alpha,
                        lambda,
                        alpha * capped_step_norm,
                    ));
                    break;
                }
                alpha *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }

        let Some((trial_eta, trial_evaluation, trial_objective, alpha, lambda, step_norm)) =
            accepted
        else {
            return Err(format!(
                "TEAM 13 identifiable material eta line search failed at iteration {iteration} (objective={:.6e}, gradient_norm={:.6e})",
                objective_components.total, gradient_norm
            ));
        };
        history.push(Team13MaterialOnlyIteration {
            iteration,
            objective: objective_components.total,
            trial_objective,
            steel_weighted_residual_norm: l2_norm(&trial_evaluation.residual)
                / context.observation_std_tesla,
            gradient_norm,
            step_norm,
            alpha,
            regularization_lambda: lambda,
            forward_residual_norm: evaluation.forward_residual_norm,
            trial_forward_residual_norm: trial_evaluation.forward_residual_norm,
        });
        eta = trial_eta;
        evaluation = trial_evaluation;
        if step_norm <= 1.0e-10 {
            converged = true;
            break;
        }
    }

    let final_hessian = team13_identifiable_eta_hessian(
        rank,
        prior_precision,
        &evaluation.jacobian,
        observation_variance,
    )?;
    let objective_components = team13_identifiable_eta_objective_components(
        &eta,
        prior_precision,
        &evaluation,
        observation_variance,
    );
    Ok(Team13IdentifiableEtaSolve {
        eta,
        converged,
        final_evaluation: evaluation,
        final_hessian,
        objective_components,
        history,
    })
}

fn team13_material_only_objective_components(
    theta: [f64; 3],
    prior_precision: f64,
    evaluation: &Team13MaterialOnlyThetaEvaluation,
    observation_variance: f64,
) -> Team13MaterialOnlyObjectiveComponents {
    let prior_material =
        0.5 * prior_precision * theta.iter().map(|value| value * value).sum::<f64>();
    let steel_observation = 0.5
        * evaluation
            .residual
            .iter()
            .map(|value| value * value / observation_variance)
            .sum::<f64>();
    Team13MaterialOnlyObjectiveComponents {
        prior_material,
        steel_observation,
        total: prior_material + steel_observation,
    }
}

fn team13_material_only_gradient(
    theta: [f64; 3],
    prior_precision: f64,
    evaluation: &Team13MaterialOnlyThetaEvaluation,
    observation_variance: f64,
) -> [f64; 3] {
    let mut gradient = [
        prior_precision * theta[0],
        prior_precision * theta[1],
        prior_precision * theta[2],
    ];
    for (row, residual) in evaluation.jacobian.iter().zip(evaluation.residual.iter()) {
        for theta_index in 0..3 {
            gradient[theta_index] += row[theta_index] * *residual / observation_variance;
        }
    }
    gradient
}

fn team13_material_only_hessian(
    prior_precision: f64,
    jacobian: &[[f64; 3]],
    observation_variance: f64,
) -> [[f64; 3]; 3] {
    let mut hessian = [[0.0; 3]; 3];
    for (index, row) in hessian.iter_mut().enumerate() {
        row[index] = prior_precision;
    }
    for output_row in jacobian {
        for row in 0..3 {
            for col in 0..3 {
                hessian[row][col] += output_row[row] * output_row[col] / observation_variance;
            }
        }
    }
    hessian
}

fn team13_identifiable_eta_objective_components(
    eta: &[f64],
    prior_precision: f64,
    evaluation: &Team13IdentifiableEtaEvaluation,
    observation_variance: f64,
) -> Team13MaterialOnlyObjectiveComponents {
    let prior_material = 0.5 * prior_precision * eta.iter().map(|value| value * value).sum::<f64>();
    let steel_observation = 0.5
        * evaluation
            .residual
            .iter()
            .map(|value| value * value / observation_variance)
            .sum::<f64>();
    Team13MaterialOnlyObjectiveComponents {
        prior_material,
        steel_observation,
        total: prior_material + steel_observation,
    }
}

fn team13_identifiable_eta_gradient(
    eta: &[f64],
    prior_precision: f64,
    evaluation: &Team13IdentifiableEtaEvaluation,
    observation_variance: f64,
) -> Result<Vec<f64>, String> {
    let mut gradient = eta
        .iter()
        .map(|value| prior_precision * *value)
        .collect::<Vec<_>>();
    for (row, residual) in evaluation.jacobian.iter().zip(evaluation.residual.iter()) {
        if row.len() != eta.len() {
            return Err(format!(
                "TEAM 13 identifiable material jacobian row has {} columns but eta dimension is {}",
                row.len(),
                eta.len()
            ));
        }
        for mode in 0..eta.len() {
            gradient[mode] += row[mode] * *residual / observation_variance;
        }
    }
    Ok(gradient)
}

fn team13_identifiable_eta_hessian(
    rank: usize,
    prior_precision: f64,
    jacobian: &[Vec<f64>],
    observation_variance: f64,
) -> Result<Vec<Vec<f64>>, String> {
    let mut hessian = vec![vec![0.0; rank]; rank];
    for (index, row) in hessian.iter_mut().enumerate() {
        row[index] = prior_precision;
    }
    for output_row in jacobian {
        if output_row.len() != rank {
            return Err(format!(
                "TEAM 13 identifiable material jacobian row has {} columns but rank is {}",
                output_row.len(),
                rank
            ));
        }
        for row in 0..rank {
            for col in 0..rank {
                hessian[row][col] += output_row[row] * output_row[col] / observation_variance;
            }
        }
    }
    Ok(hessian)
}

fn team13_material_only_lm_lambdas(regularization: GaussNewtonStepRegularization) -> Vec<f64> {
    match regularization {
        GaussNewtonStepRegularization::None => vec![0.0],
        GaussNewtonStepRegularization::LevenbergMarquardtGrid
        | GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt => vec![
            0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 1.0e2,
        ],
    }
}

fn team13_material_only_weighted_residual_norm(
    evaluation: &Team13MaterialOnlyThetaEvaluation,
    observation_std_tesla: f64,
) -> f64 {
    l2_norm(&evaluation.residual) / observation_std_tesla
}

fn add_diagonal_3x3(mut matrix: [[f64; 3]; 3], diagonal: f64) -> [[f64; 3]; 3] {
    for (index, row) in matrix.iter_mut().enumerate() {
        row[index] += diagonal;
    }
    matrix
}

fn mat3_vec_mul(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn vector3_norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn dense_vector_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn dot_dense(left: &[f64], right: &[f64]) -> Result<f64, String> {
    if left.len() != right.len() {
        return Err(format!(
            "dense dot length mismatch: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum())
}

fn dense_quadratic_form(vector: &[f64], matrix: &[Vec<f64>]) -> Result<f64, String> {
    validate_square_dense(matrix, "dense quadratic form")?;
    if matrix.len() != vector.len() {
        return Err(format!(
            "dense quadratic form length mismatch: vector has {} entries but matrix has {} rows",
            vector.len(),
            matrix.len()
        ));
    }
    let mut value = 0.0;
    for row in 0..vector.len() {
        for col in 0..vector.len() {
            value += vector[row] * matrix[row][col] * vector[col];
        }
    }
    Ok(value)
}

fn add_diagonal_dense(matrix: &[Vec<f64>], diagonal: f64) -> Result<Vec<Vec<f64>>, String> {
    validate_square_dense(matrix, "dense diagonal update")?;
    let mut output = matrix.to_vec();
    for (index, row) in output.iter_mut().enumerate() {
        row[index] += diagonal;
    }
    Ok(output)
}

fn validate_square_dense(matrix: &[Vec<f64>], label: &str) -> Result<(), String> {
    let n = matrix.len();
    if n == 0 {
        return Err(format!("{label} matrix must not be empty"));
    }
    for row in matrix {
        if row.len() != n {
            return Err(format!(
                "{label} matrix must be square; got {} rows and a row with {} columns",
                n,
                row.len()
            ));
        }
    }
    Ok(())
}

fn solve_dense_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    validate_square_dense(matrix, "dense linear solve").ok()?;
    let n = matrix.len();
    if rhs.len() != n {
        return None;
    }
    let mut augmented = vec![vec![0.0; n + 1]; n];
    for row in 0..n {
        for col in 0..n {
            augmented[row][col] = matrix[row][col];
        }
        augmented[row][n] = rhs[row];
    }
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = augmented[pivot][pivot].abs();
        for (row, values) in augmented.iter().enumerate().skip(pivot + 1) {
            if values[pivot].abs() > pivot_abs {
                pivot_abs = values[pivot].abs();
                pivot_row = row;
            }
        }
        if pivot_abs <= 1.0e-18 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot_row != pivot {
            augmented.swap(pivot, pivot_row);
        }
        let pivot_value = augmented[pivot][pivot];
        for col in pivot..=n {
            augmented[pivot][col] /= pivot_value;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..=n {
                augmented[row][col] -= factor * augmented[pivot][col];
            }
        }
    }
    Some((0..n).map(|row| augmented[row][n]).collect())
}

fn team13_material_only_steel_prediction_reports(
    context: &Team13MaterialOnlyForwardContext,
    nominal_state: &[f64],
    map_state: &[f64],
    map_evaluation: &Team13MaterialOnlyThetaEvaluation,
) -> Result<Vec<Team13MaterialOnlySteelPredictionReport>, String> {
    let (nominal_predictions, nominal_signed_predictions) =
        team13_smooth_steel_predictions(&context.patch_operator, nominal_state, context.smoothing)?;
    let (map_predictions, map_signed_predictions) =
        team13_smooth_steel_predictions(&context.patch_operator, map_state, context.smoothing)?;
    context
        .patch_operator
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let observed = context.observations[index];
            Ok(Team13MaterialOnlySteelPredictionReport {
                name: definition.name.clone(),
                group: team13_steel_surface_group(index)?,
                observed,
                nominal_prediction: nominal_predictions[index],
                map_prediction: map_predictions[index],
                nominal_residual: nominal_predictions[index] - observed,
                map_residual: map_evaluation.residual[index],
                nominal_signed_prediction: nominal_signed_predictions[index],
                map_signed_prediction: map_signed_predictions[index],
                row_nnz: context.patch_operator.row_nnz[index],
            })
        })
        .collect()
}

fn team13_material_only_sensitivity_report(
    jacobian: &[[f64; 3]],
) -> Result<Team13MaterialOnlySensitivityReport, String> {
    let summary = team13_material_steel_sensitivity_summary(jacobian);
    let normal = dense_jtj_3(jacobian);
    let eigenvalues = symmetric_3x3_eigenvalues(normal);
    let singular_values = [
        eigenvalues[0].max(0.0).sqrt(),
        eigenvalues[1].max(0.0).sqrt(),
        eigenvalues[2].max(0.0).sqrt(),
    ];
    let max_sv = singular_values[0];
    let rank_tolerance = 1.0e-10 * max_sv.max(1.0);
    let rank = singular_values
        .iter()
        .filter(|value| **value > rank_tolerance)
        .count();
    let min_nonzero = singular_values
        .iter()
        .copied()
        .filter(|value| *value > rank_tolerance)
        .fold(f64::INFINITY, f64::min);
    let condition_number = if min_nonzero.is_finite() {
        safe_ratio(max_sv, min_nonzero)
    } else {
        f64::INFINITY
    };
    Ok(Team13MaterialOnlySensitivityReport {
        singular_values,
        rank,
        condition_number,
        frobenius_norm_tesla: summary.frobenius_norm_tesla,
        max_abs_sensitivity_tesla: summary.max_abs_sensitivity_tesla,
        theta_column_norms_tesla: summary.theta_column_norms_tesla,
        steel_row_count: summary.steel_row_count,
    })
}

fn team13_material_only_comparison_rows(
    solve: &Team13MaterialOnlyThetaSolve,
    reports: &[Team13MaterialOnlySteelPredictionReport],
    observation_std_tesla: f64,
) -> Vec<Team13MaterialOnlyComparisonRow> {
    let observations = reports
        .iter()
        .map(|report| report.observed)
        .collect::<Vec<_>>();
    let nominal_predictions = reports
        .iter()
        .map(|report| report.nominal_prediction)
        .collect::<Vec<_>>();
    let map_predictions = reports
        .iter()
        .map(|report| report.map_prediction)
        .collect::<Vec<_>>();
    let nominal_rmse =
        rmse_from_prediction_pairs(&nominal_predictions, &observations).unwrap_or(f64::NAN);
    let nominal_max = max_abs_residual_from_prediction_pairs(&nominal_predictions, &observations)
        .unwrap_or(f64::NAN);
    let map_rmse = rmse_from_prediction_pairs(&map_predictions, &observations).unwrap_or(f64::NAN);
    let map_max =
        max_abs_residual_from_prediction_pairs(&map_predictions, &observations).unwrap_or(f64::NAN);
    vec![
        Team13MaterialOnlyComparisonRow {
            label: "nominal_material".to_string(),
            theta: [0.0; 3],
            steel_rmse_tesla: nominal_rmse,
            steel_max_abs_residual_tesla: nominal_max,
            objective: 0.5
                * nominal_predictions
                    .iter()
                    .zip(observations.iter())
                    .map(|(prediction, observation)| {
                        let residual = prediction - observation;
                        residual * residual
                    })
                    .sum::<f64>()
                / (observation_std_tesla * observation_std_tesla),
        },
        Team13MaterialOnlyComparisonRow {
            label: "material_only_map".to_string(),
            theta: solve.theta,
            steel_rmse_tesla: map_rmse,
            steel_max_abs_residual_tesla: map_max,
            objective: solve.objective_components.total,
        },
    ]
}

fn dense_jtj_3(jacobian: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in jacobian {
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] += row[i] * row[j];
            }
        }
    }
    out
}

fn symmetric_3x3_eigenvalues(matrix: [[f64; 3]; 3]) -> [f64; 3] {
    symmetric_3x3_eigendecomposition(matrix).eigenvalues
}

fn symmetric_3x3_eigendecomposition(mut matrix: [[f64; 3]; 3]) -> Symmetric3EigenDecomposition {
    let mut eigenvectors = [[0.0; 3]; 3];
    for index in 0..3 {
        eigenvectors[index][index] = 1.0;
    }
    for _ in 0..32 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_offdiag = matrix[0][1].abs();
        for (i, j) in [(0usize, 2usize), (1usize, 2usize)] {
            if matrix[i][j].abs() > max_offdiag {
                max_offdiag = matrix[i][j].abs();
                p = i;
                q = j;
            }
        }
        if max_offdiag <= 1.0e-14 {
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
        for k in 0..3 {
            if k != p && k != q {
                let aik = matrix[p][k];
                let aqk = matrix[q][k];
                matrix[p][k] = c * aik - s * aqk;
                matrix[k][p] = matrix[p][k];
                matrix[q][k] = s * aik + c * aqk;
                matrix[k][q] = matrix[q][k];
            }
        }
        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for row in &mut eigenvectors {
            let vip = row[p];
            let viq = row[q];
            row[p] = c * vip - s * viq;
            row[q] = s * vip + c * viq;
        }
    }
    let mut pairs = (0..3)
        .map(|index| {
            let mut vector = [
                eigenvectors[0][index],
                eigenvectors[1][index],
                eigenvectors[2][index],
            ];
            normalize_and_orient_vector3(&mut vector);
            (matrix[index][index], vector)
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Symmetric3EigenDecomposition {
        eigenvalues: [pairs[0].0, pairs[1].0, pairs[2].0],
        eigenvectors: [pairs[0].1, pairs[1].1, pairs[2].1],
    }
}

fn normalize_and_orient_vector3(vector: &mut [f64; 3]) {
    let norm = vector3_norm(*vector);
    if norm > 0.0 && norm.is_finite() {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
    let mut pivot = 0usize;
    let mut pivot_abs = vector[0].abs();
    for (index, value) in vector.iter().enumerate().skip(1) {
        if value.abs() > pivot_abs {
            pivot = index;
            pivot_abs = value.abs();
        }
    }
    if vector[pivot] < 0.0 {
        for value in vector.iter_mut() {
            *value = -*value;
        }
    }
}

fn team13_identifiable_material_basis(
    jacobian: &[[f64; 3]],
    relative_tolerance: f64,
    absolute_tolerance: f64,
) -> Result<Team13IdentifiableMaterialBasis, String> {
    validate_positive_finite(
        relative_tolerance,
        "TEAM 13 identifiable material SVD relative tolerance",
    )?;
    validate_positive_finite(
        absolute_tolerance,
        "TEAM 13 identifiable material SVD absolute tolerance",
    )?;
    let normal = dense_jtj_3(jacobian);
    let decomposition = symmetric_3x3_eigendecomposition(normal);
    let singular_values = [
        decomposition.eigenvalues[0].max(0.0).sqrt(),
        decomposition.eigenvalues[1].max(0.0).sqrt(),
        decomposition.eigenvalues[2].max(0.0).sqrt(),
    ];
    let max_singular = singular_values[0];
    let threshold = absolute_tolerance.max(relative_tolerance * max_singular);
    let mut retained_modes = Vec::new();
    let mut mode_reports = Vec::new();
    for mode in 0..3 {
        let retained = singular_values[mode] > threshold;
        if retained {
            retained_modes.push(decomposition.eigenvectors[mode]);
        }
        mode_reports.push(Team13IdentifiableMaterialModeReport {
            mode_index: mode,
            retained,
            singular_value: singular_values[mode],
            relative_singular_value: safe_ratio(singular_values[mode], max_singular),
            theta_coefficients: decomposition.eigenvectors[mode],
        });
    }
    Ok(Team13IdentifiableMaterialBasis {
        singular_values,
        retained_modes,
        mode_reports,
    })
}

fn team13_eta_to_theta(
    theta_bias: [f64; 3],
    basis: &[[f64; 3]],
    eta: &[f64],
) -> Result<[f64; 3], String> {
    if basis.len() != eta.len() {
        return Err(format!(
            "TEAM 13 identifiable material eta length {} must match basis rank {}",
            eta.len(),
            basis.len()
        ));
    }
    let mut theta = theta_bias;
    for (mode, value) in basis.iter().zip(eta.iter()) {
        for component in 0..3 {
            theta[component] += mode[component] * *value;
        }
    }
    Ok(theta)
}

fn scale_theta(theta: [f64; 3], scale: f64) -> [f64; 3] {
    [scale * theta[0], scale * theta[1], scale * theta[2]]
}

fn team13_transform_theta_jacobian_to_eta(
    theta_jacobian: &[[f64; 3]],
    basis: &[[f64; 3]],
) -> Result<Vec<Vec<f64>>, String> {
    if basis.is_empty() {
        return Err("TEAM 13 identifiable material basis must not be empty".to_string());
    }
    Ok(theta_jacobian
        .iter()
        .map(|row| {
            basis
                .iter()
                .map(|mode| row[0] * mode[0] + row[1] * mode[1] + row[2] * mode[2])
                .collect::<Vec<_>>()
        })
        .collect())
}

fn team13_material_output_shift_from_theta(
    theta_jacobian: &[[f64; 3]],
    theta: [f64; 3],
) -> Vec<f64> {
    theta_jacobian
        .iter()
        .map(|row| row[0] * theta[0] + row[1] * theta[1] + row[2] * theta[2])
        .collect()
}

fn team13_identifiable_baseline_perturbation(
    basis: &Team13IdentifiableMaterialBasis,
    theta_jacobian: &[[f64; 3]],
    target_rms_tesla: f64,
    gap_difference_rms_tesla: f64,
) -> Result<Team13IdentifiableBaselinePerturbationReport, String> {
    validate_positive_finite(
        target_rms_tesla,
        "TEAM 13 identifiable material baseline target RMS",
    )?;
    if basis.retained_modes.is_empty() {
        return Err(
            "TEAM 13 identifiable material perturbation requires a retained mode".to_string(),
        );
    }
    let leading_singular = basis.singular_values[0];
    validate_positive_finite(
        leading_singular,
        "TEAM 13 identifiable material leading singular value",
    )?;
    let row_count = theta_jacobian.len();
    if row_count == 0 {
        return Err(
            "TEAM 13 identifiable material perturbation requires at least one steel row"
                .to_string(),
        );
    }
    let mut eta_bias = vec![0.0; basis.retained_modes.len()];
    eta_bias[0] = target_rms_tesla * (row_count as f64).sqrt() / leading_singular;
    let theta_bias = team13_eta_to_theta([0.0; 3], &basis.retained_modes, &eta_bias)?;
    let shift = team13_material_output_shift_from_theta(theta_jacobian, theta_bias);
    let achieved_linearized_rms_tesla =
        (shift.iter().map(|value| value * value).sum::<f64>() / row_count as f64).sqrt();
    Ok(Team13IdentifiableBaselinePerturbationReport {
        target_rms_tesla,
        achieved_linearized_rms_tesla,
        gap_difference_rms_tesla,
        eta_bias,
        theta_bias,
    })
}

fn team13_calibrated_identifiable_eta_prior(
    config: &Team13IdentifiableMaterialUqConfig,
    basis: &Team13IdentifiableMaterialBasis,
    steel_row_count: usize,
) -> Result<Team13IdentifiableEtaPriorCalibrationReport, String> {
    let configured_std = config.eta_prior_std;
    let retained_mode_norms_tesla = basis
        .singular_values
        .iter()
        .zip(basis.mode_reports.iter())
        .filter_map(|(singular, report)| report.retained.then_some(*singular))
        .collect::<Vec<_>>();
    if config.eta_prior_calibration == Team13MaterialPriorCalibrationMode::Fixed {
        return Ok(Team13IdentifiableEtaPriorCalibrationReport {
            mode: config.eta_prior_calibration,
            target: config.eta_prior_calibration_target,
            target_steel_rms_tesla: None,
            configured_eta_prior_std: configured_std,
            eta_prior_std: configured_std,
            unclamped_eta_prior_std: None,
            eta_prior_std_floor: config.eta_prior_std_floor,
            eta_prior_std_ceiling: config.eta_prior_std_ceiling,
            unit_eta_steel_rms_tesla: None,
            retained_rank: basis.retained_modes.len(),
            retained_mode_norms_tesla,
            steel_row_count,
        });
    }

    let target_steel_rms = team13_material_prior_target_steel_rms_from_parts(
        config.eta_prior_target_steel_rms_tesla,
        config.eta_prior_calibration_target,
        config.operator.observation_std_tesla,
    )?;
    let frobenius_sq = retained_mode_norms_tesla
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let unit_eta_steel_rms = if steel_row_count == 0 {
        0.0
    } else {
        (frobenius_sq / steel_row_count as f64).sqrt()
    };
    let (eta_prior_std, unclamped_eta_prior_std) = calibrated_material_prior_std_from_unit_rms(
        configured_std,
        target_steel_rms,
        unit_eta_steel_rms,
        config.eta_prior_std_floor,
        config.eta_prior_std_ceiling,
    );
    Ok(Team13IdentifiableEtaPriorCalibrationReport {
        mode: config.eta_prior_calibration,
        target: config.eta_prior_calibration_target,
        target_steel_rms_tesla: Some(target_steel_rms),
        configured_eta_prior_std: configured_std,
        eta_prior_std,
        unclamped_eta_prior_std: Some(unclamped_eta_prior_std),
        eta_prior_std_floor: config.eta_prior_std_floor,
        eta_prior_std_ceiling: config.eta_prior_std_ceiling,
        unit_eta_steel_rms_tesla: Some(unit_eta_steel_rms),
        retained_rank: basis.retained_modes.len(),
        retained_mode_norms_tesla,
        steel_row_count,
    })
}

fn team13_calibrated_identifiable_joint_eta_prior(
    config: &Team13IdentifiableJointMaterialUqConfig,
    basis: &Team13IdentifiableMaterialBasis,
    steel_row_count: usize,
) -> Result<Team13IdentifiableEtaPriorCalibrationReport, String> {
    let configured_std = config.eta_prior_std;
    let retained_mode_norms_tesla = basis
        .singular_values
        .iter()
        .zip(basis.mode_reports.iter())
        .filter_map(|(singular, report)| report.retained.then_some(*singular))
        .collect::<Vec<_>>();
    if config.eta_prior_calibration == Team13MaterialPriorCalibrationMode::Fixed {
        return Ok(Team13IdentifiableEtaPriorCalibrationReport {
            mode: config.eta_prior_calibration,
            target: config.eta_prior_calibration_target,
            target_steel_rms_tesla: None,
            configured_eta_prior_std: configured_std,
            eta_prior_std: configured_std,
            unclamped_eta_prior_std: None,
            eta_prior_std_floor: config.eta_prior_std_floor,
            eta_prior_std_ceiling: config.eta_prior_std_ceiling,
            unit_eta_steel_rms_tesla: None,
            retained_rank: basis.retained_modes.len(),
            retained_mode_norms_tesla,
            steel_row_count,
        });
    }

    let target_steel_rms = team13_material_prior_target_steel_rms_from_parts(
        config.eta_prior_target_steel_rms_tesla,
        config.eta_prior_calibration_target,
        config.operator.observation_std_tesla,
    )?;
    let frobenius_sq = retained_mode_norms_tesla
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let unit_eta_steel_rms = if steel_row_count == 0 {
        0.0
    } else {
        (frobenius_sq / steel_row_count as f64).sqrt()
    };
    let (eta_prior_std, unclamped_eta_prior_std) = calibrated_material_prior_std_from_unit_rms(
        configured_std,
        target_steel_rms,
        unit_eta_steel_rms,
        config.eta_prior_std_floor,
        config.eta_prior_std_ceiling,
    );
    Ok(Team13IdentifiableEtaPriorCalibrationReport {
        mode: config.eta_prior_calibration,
        target: config.eta_prior_calibration_target,
        target_steel_rms_tesla: Some(target_steel_rms),
        configured_eta_prior_std: configured_std,
        eta_prior_std,
        unclamped_eta_prior_std: Some(unclamped_eta_prior_std),
        eta_prior_std_floor: config.eta_prior_std_floor,
        eta_prior_std_ceiling: config.eta_prior_std_ceiling,
        unit_eta_steel_rms_tesla: Some(unit_eta_steel_rms),
        retained_rank: basis.retained_modes.len(),
        retained_mode_norms_tesla,
        steel_row_count,
    })
}

fn team13_theta_covariance_from_eta_covariance(
    basis: &[[f64; 3]],
    eta_covariance: &[Vec<f64>],
) -> Result<[[f64; 3]; 3], String> {
    if eta_covariance.len() != basis.len() {
        return Err(format!(
            "TEAM 13 identifiable material eta covariance has {} rows but basis rank is {}",
            eta_covariance.len(),
            basis.len()
        ));
    }
    for row in eta_covariance {
        if row.len() != basis.len() {
            return Err(format!(
                "TEAM 13 identifiable material eta covariance row has {} columns but basis rank is {}",
                row.len(),
                basis.len()
            ));
        }
    }
    let mut covariance = [[0.0; 3]; 3];
    for eta_row in 0..basis.len() {
        for eta_col in 0..basis.len() {
            for theta_row in 0..3 {
                for theta_col in 0..3 {
                    covariance[theta_row][theta_col] += basis[eta_row][theta_row]
                        * eta_covariance[eta_row][eta_col]
                        * basis[eta_col][theta_col];
                }
            }
        }
    }
    Ok(covariance)
}

fn team13_identifiable_eta_posterior_reports(
    eta_prior_std: f64,
    eta_bias: &[f64],
    eta_map: &[f64],
    eta_covariance: &[Vec<f64>],
) -> Vec<Team13IdentifiableEtaPosteriorReport> {
    eta_map
        .iter()
        .enumerate()
        .map(|(index, posterior_mean)| {
            let bias = eta_bias.get(index).copied().unwrap_or(0.0);
            Team13IdentifiableEtaPosteriorReport {
                name: format!("eta_svd_{}", index),
                prior_mean: 0.0,
                posterior_mean: *posterior_mean,
                prior_std: eta_prior_std,
                posterior_std: eta_covariance[index][index].max(0.0).sqrt(),
                eta_bias: bias,
                recovery_fraction: team13_identifiable_recovery_fraction(*posterior_mean, bias),
            }
        })
        .collect()
}

fn team13_identifiable_recovery_fraction(eta_map: f64, eta_bias: f64) -> f64 {
    if eta_bias.abs() <= EPS {
        0.0
    } else {
        -eta_map / eta_bias
    }
}

fn team13_identifiable_theta_posterior_reports(
    anchors: [f64; 3],
    theta_bias: [f64; 3],
    theta_map: [f64; 3],
    theta_covariance: [[f64; 3]; 3],
) -> Vec<Team13IdentifiableThetaPosteriorReport> {
    (0..3)
        .map(|index| Team13IdentifiableThetaPosteriorReport {
            name: format!("theta_h_{}", index),
            anchor_b_tesla: anchors[index],
            benchmark_mean: 0.0,
            biased_baseline_mean: theta_bias[index],
            posterior_mean: theta_map[index],
            posterior_std: theta_covariance[index][index].max(0.0).sqrt(),
        })
        .collect()
}

fn team13_identifiable_steel_prediction_reports(
    context: &Team13MaterialOnlyForwardContext,
    benchmark: &Team13MaterialOnlyThetaEvaluation,
    biased: &Team13MaterialOnlyThetaEvaluation,
    corrected: &Team13IdentifiableEtaEvaluation,
) -> Result<Vec<Team13IdentifiableSteelPredictionReport>, String> {
    let (benchmark_predictions, _) = team13_smooth_steel_predictions(
        &context.patch_operator,
        &benchmark.forward_state,
        context.smoothing,
    )?;
    let (biased_predictions, _) = team13_smooth_steel_predictions(
        &context.patch_operator,
        &biased.forward_state,
        context.smoothing,
    )?;
    let (corrected_predictions, _) = team13_smooth_steel_predictions(
        &context.patch_operator,
        &corrected.forward_state,
        context.smoothing,
    )?;
    context
        .patch_operator
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let observed = context.observations[index];
            Ok(Team13IdentifiableSteelPredictionReport {
                name: definition.name.clone(),
                group: team13_steel_surface_group(index)?,
                observed,
                benchmark_prediction: benchmark_predictions[index],
                biased_prediction: biased_predictions[index],
                corrected_prediction: corrected_predictions[index],
                benchmark_residual: benchmark_predictions[index] - observed,
                biased_residual: biased_predictions[index] - observed,
                corrected_residual: corrected_predictions[index] - observed,
                row_nnz: context.patch_operator.row_nnz[index],
            })
        })
        .collect()
}

fn team13_identifiable_comparison_rows(
    benchmark: &Team13MaterialOnlyThetaEvaluation,
    biased: &Team13MaterialOnlyThetaEvaluation,
    corrected: &Team13IdentifiableEtaEvaluation,
    baseline: &Team13IdentifiableBaselinePerturbationReport,
    basis: &[[f64; 3]],
    eta_prior_std: f64,
    observation_std_tesla: f64,
) -> Vec<Team13IdentifiableMaterialComparisonRow> {
    let prior_precision = 1.0 / (eta_prior_std * eta_prior_std);
    let observation_variance = observation_std_tesla * observation_std_tesla;
    let benchmark_eta = baseline
        .eta_bias
        .iter()
        .map(|value| -*value)
        .collect::<Vec<_>>();
    let biased_eta = vec![0.0; basis.len()];
    let benchmark_eval = Team13IdentifiableEtaEvaluation {
        theta: [0.0; 3],
        forward_state: benchmark.forward_state.clone(),
        forward_converged: benchmark.forward_converged,
        forward_residual_norm: benchmark.forward_residual_norm,
        residual: benchmark.residual.clone(),
        jacobian: Vec::new(),
    };
    let biased_eval = Team13IdentifiableEtaEvaluation {
        theta: baseline.theta_bias,
        forward_state: biased.forward_state.clone(),
        forward_converged: biased.forward_converged,
        forward_residual_norm: biased.forward_residual_norm,
        residual: biased.residual.clone(),
        jacobian: Vec::new(),
    };
    [
        ("benchmark_law", [0.0; 3], benchmark_eta, benchmark_eval),
        (
            "biased_baseline",
            baseline.theta_bias,
            biased_eta,
            biased_eval,
        ),
        (
            "corrected_map",
            corrected.theta,
            basis
                .iter()
                .map(|mode| {
                    let diff = [
                        corrected.theta[0] - baseline.theta_bias[0],
                        corrected.theta[1] - baseline.theta_bias[1],
                        corrected.theta[2] - baseline.theta_bias[2],
                    ];
                    diff[0] * mode[0] + diff[1] * mode[1] + diff[2] * mode[2]
                })
                .collect::<Vec<_>>(),
            corrected.clone(),
        ),
    ]
    .into_iter()
    .map(|(label, theta, eta, evaluation)| {
        let rmse = (evaluation
            .residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / evaluation.residual.len().max(1) as f64)
            .sqrt();
        let max_abs = evaluation
            .residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let objective = team13_identifiable_eta_objective_components(
            &eta,
            prior_precision,
            &evaluation,
            observation_variance,
        )
        .total;
        Team13IdentifiableMaterialComparisonRow {
            label: label.to_string(),
            theta,
            eta,
            steel_rmse_tesla: rmse,
            steel_max_abs_residual_tesla: max_abs,
            objective,
        }
    })
    .collect()
}

fn team13_identifiable_bh_curve_bands(
    anchors: [f64; 3],
    theta_bias: [f64; 3],
    theta_mean: [f64; 3],
    theta_covariance: [[f64; 3]; 3],
) -> Result<Vec<Team13IdentifiableBhCurveBandReport>, String> {
    let benchmark = Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES)?
        .with_log_h_shape(anchors, [0.0; 3])?;
    let biased = Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES)?
        .with_log_h_shape(anchors, theta_bias)?;
    let corrected = Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES)?
        .with_log_h_shape(anchors, theta_mean)?;
    let mut rows = Vec::new();
    for index in 0..=60 {
        let b = 3.0 * index as f64 / 60.0;
        let basis = corrected.log_h_shape_basis(b);
        let log_variance = quadratic_form_3(basis, theta_covariance).max(0.0);
        let log_std = log_variance.sqrt();
        let corrected_h = corrected.h_ampere_per_meter(b);
        rows.push(Team13IdentifiableBhCurveBandReport {
            b_tesla: b,
            benchmark_h_ampere_per_meter: benchmark.h_ampere_per_meter(b),
            biased_baseline_h_ampere_per_meter: biased.h_ampere_per_meter(b),
            corrected_mean_h_ampere_per_meter: corrected_h,
            corrected_std_h_ampere_per_meter: corrected_h * log_std,
            corrected_lower_2sigma_h_ampere_per_meter: corrected_h * (-2.0 * log_std).exp(),
            corrected_upper_2sigma_h_ampere_per_meter: corrected_h * (2.0 * log_std).exp(),
        });
    }
    Ok(rows)
}

fn team13_final_step_diagnostic(
    problem: &NonlinearLaplaceProblem<'_>,
    config: &GaussNewtonConfig,
    state: &[f64],
) -> Team13JointMaterialFinalStepDiagnostic {
    let mut final_config = config.clone();
    final_config.initial_guess = Some(state.to_vec());
    match diagnose_gauss_newton_first_step(problem, &final_config) {
        Ok(diagnostics) => Team13JointMaterialFinalStepDiagnostic {
            available: true,
            error: None,
            objective: diagnostics.objective,
            weighted_residual_norm: diagnostics.weighted_residual_norm,
            gradient_norm: diagnostics.gradient_norm,
            step_norm: diagnostics.step_norm,
            directional_derivative: diagnostics.directional_derivative,
            accepted_alpha: diagnostics.accepted_alpha,
            accepted_objective: diagnostics.accepted_objective,
            regularization_lambda: diagnostics.regularization_lambda,
            linear_solve_absolute_residual_norm: diagnostics.linear_solve_absolute_residual_norm,
            linear_solve_relative_residual_norm: diagnostics.linear_solve_relative_residual_norm,
        },
        Err(err) => Team13JointMaterialFinalStepDiagnostic::unavailable(err),
    }
}

fn team13_material_solve_diagnostics(
    label: &str,
    state_dimension: usize,
    theta_dimension: usize,
    prior_kind: Team13MapParityPriorKind,
    pde_residual_kind: Team13MapParityPdeResidualKind,
    linear_measurements: &[LinearGaussianMeasurementSpec],
    posterior_residual_norm: f64,
    posterior: &NonlinearLaplaceResult,
    final_step: Team13JointMaterialFinalStepDiagnostic,
    objective_components: Team13JointMaterialObjectiveComponents,
) -> Team13JointMaterialSolveDiagnostics {
    Team13JointMaterialSolveDiagnostics {
        label: label.to_string(),
        state_dimension,
        theta_dimension,
        prior_kind,
        pde_residual_kind,
        linear_measurement_count: linear_measurements.len(),
        linear_measurement_rows: linear_measurements
            .iter()
            .map(|measurement| measurement.operator.nrows())
            .sum(),
        converged: posterior.converged,
        posterior_residual_norm,
        posterior_precision_nnz: posterior.assembly.posterior_precision_nnz,
        posterior_factor_nnz: posterior.final_factorization.nnz,
        history: posterior.history.clone(),
        diagnostics: posterior.diagnostics,
        assembly: posterior.assembly.clone(),
        final_factorization: posterior.final_factorization.clone(),
        final_residuals: posterior.final_residuals.clone(),
        final_step,
        objective_components,
    }
}

fn team13_joint_material_objective_components(
    label: &str,
    state_prior: &GaussianPriorSpec,
    material_prior_std: Option<f64>,
    state: &[f64],
    linear_measurements: &[LinearGaussianMeasurementSpec],
    final_residuals: &[NonlinearResidualReport],
    solver_objective: Option<f64>,
) -> Result<Team13JointMaterialObjectiveComponents, String> {
    let state_dimension = state_prior.dimension();
    if state.len() < state_dimension {
        return Err(format!(
            "objective component state length {} must be at least state prior dimension {}",
            state.len(),
            state_dimension
        ));
    }
    let prior_state = gaussian_prior_penalty(state_prior, &state[..state_dimension])?;
    let prior_material = if let Some(std) = material_prior_std {
        validate_positive_finite(std, "TEAM 13 material objective prior std")?;
        let variance = std * std;
        0.5 * state[state_dimension..]
            .iter()
            .map(|value| value * value / variance)
            .sum::<f64>()
    } else {
        0.0
    };
    let mut steel_observation = linear_measurement_penalty(linear_measurements, state)?;
    let mut pde_residual = 0.0;
    for report in final_residuals {
        let penalty = 0.5 * report.weighted_norm * report.weighted_norm;
        if team13_residual_report_is_steel_observation(&report.name) {
            steel_observation += penalty;
        } else {
            pde_residual += penalty;
        }
    }
    let total = prior_state + prior_material + steel_observation + pde_residual;
    Ok(Team13JointMaterialObjectiveComponents {
        label: label.to_string(),
        prior_state,
        prior_material,
        steel_observation,
        pde_residual,
        total,
        solver_objective,
        solver_objective_gap: solver_objective.map(|objective| total - objective),
    })
}

fn team13_residual_report_is_steel_observation(name: &str) -> bool {
    name.contains("steel") || name.contains("smooth_magnitude")
}

fn gaussian_prior_penalty(prior: &GaussianPriorSpec, state: &[f64]) -> Result<f64, String> {
    prior.validate()?;
    if state.len() != prior.mean.len() {
        return Err(format!(
            "prior penalty state length {} must match prior mean length {}",
            state.len(),
            prior.mean.len()
        ));
    }
    let diff = state
        .iter()
        .zip(prior.mean.iter())
        .map(|(value, mean)| value - mean)
        .collect::<Vec<_>>();
    let mut weighted = vec![0.0; diff.len()];
    for (row, col, value) in prior.precision.triplet_iter() {
        weighted[row] += value * diff[col];
    }
    Ok(0.5
        * diff
            .iter()
            .zip(weighted.iter())
            .map(|(left, right)| left * right)
            .sum::<f64>())
}

fn linear_measurement_penalty(
    measurements: &[LinearGaussianMeasurementSpec],
    state: &[f64],
) -> Result<f64, String> {
    let mut penalty = 0.0;
    for measurement in measurements {
        if measurement.operator.ncols() != state.len() {
            return Err(format!(
                "measurement `{}` has {} columns but objective state has length {}",
                measurement.name,
                measurement.operator.ncols(),
                state.len()
            ));
        }
        validate_positive_finite(measurement.variance, "linear measurement variance")?;
        let mut prediction = measurement.bias.clone();
        if prediction.len() != measurement.operator.nrows()
            || measurement.observations.len() != measurement.operator.nrows()
        {
            return Err(format!(
                "measurement `{}` dimensions are inconsistent for objective components",
                measurement.name
            ));
        }
        for (row, col, value) in measurement.operator.triplet_iter() {
            prediction[row] += value * state[col];
        }
        penalty += 0.5
            * prediction
                .iter()
                .zip(measurement.observations.iter())
                .map(|(predicted, observed)| {
                    let residual = predicted - observed;
                    residual * residual / measurement.variance
                })
                .sum::<f64>();
    }
    Ok(penalty)
}

fn append_independent_material_prior(
    state_prior: GaussianPriorSpec,
    parameter_count: usize,
    parameter_precision: f64,
) -> Result<GaussianPriorSpec, String> {
    state_prior.validate()?;
    validate_positive_finite(parameter_precision, "TEAM 13 material prior precision")?;
    let state_dimension = state_prior.dimension();
    let joint_dimension = state_dimension + parameter_count;
    let mut mean = state_prior.mean;
    mean.extend(std::iter::repeat(0.0).take(parameter_count));
    let mut triplets = state_prior
        .precision
        .triplet_iter()
        .map(|(row, col, value)| SparseTriplet { row, col, value })
        .collect::<Vec<_>>();
    for parameter in 0..parameter_count {
        triplets.push(SparseTriplet {
            row: state_dimension + parameter,
            col: state_dimension + parameter,
            value: parameter_precision,
        });
    }
    Ok(GaussianPriorSpec {
        mean,
        precision: SparseTripletMatrix::from_triplets(joint_dimension, joint_dimension, triplets),
    })
}

fn append_zero_theta_columns_to_smooth_grouped_norm_model(
    model: &SmoothGroupedNormLinearResidualModel,
    theta_count: usize,
) -> Result<SmoothGroupedNormLinearResidualModel, String> {
    let ncols = model.operator().ncols();
    SmoothGroupedNormLinearResidualModel::new(
        SparseTripletMatrix::from_triplets(
            model.operator().nrows(),
            ncols + theta_count,
            model
                .operator()
                .triplet_iter()
                .map(|(row, col, value)| SparseTriplet { row, col, value }),
        ),
        model.bias().to_vec(),
        model.groups().to_vec(),
        model.smoothing(),
    )
}

fn team13_calibrated_material_prior(
    config: &Team13JointMaterialUqConfig,
    calibration_model: &Team13MaterialShapeAugmentedResidualModel,
    deterministic_state: &[f64],
    patch_operator: &Team13ReducedSteelPatchOperator,
) -> Result<Team13MaterialPriorCalibrationReport, String> {
    let configured_std = config.material_prior_std;
    if config.material_prior_calibration == Team13MaterialPriorCalibrationMode::Fixed {
        return Ok(Team13MaterialPriorCalibrationReport {
            mode: config.material_prior_calibration,
            target: config.material_prior_calibration_target,
            target_steel_rms_tesla: None,
            configured_material_prior_std: configured_std,
            material_prior_std: configured_std,
            unclamped_material_prior_std: None,
            material_prior_std_floor: config.material_prior_std_floor,
            material_prior_std_ceiling: config.material_prior_std_ceiling,
            unit_theta_steel_rms_tesla: None,
            sensitivity_frobenius_norm_tesla: None,
            max_abs_sensitivity_tesla: None,
            theta_column_norms_tesla: [0.0; 3],
            steel_row_count: patch_operator.operator.nrows(),
        });
    }

    let target_steel_rms = team13_material_prior_target_steel_rms(config)?;
    let sensitivity = team13_hard_pde_material_steel_sensitivity(
        calibration_model,
        deterministic_state,
        patch_operator,
    )?;
    let (calibrated_std, unclamped_std) = calibrated_material_prior_std_from_unit_rms(
        configured_std,
        target_steel_rms,
        sensitivity.unit_theta_steel_rms_tesla,
        config.material_prior_std_floor,
        config.material_prior_std_ceiling,
    );

    Ok(Team13MaterialPriorCalibrationReport {
        mode: config.material_prior_calibration,
        target: config.material_prior_calibration_target,
        target_steel_rms_tesla: Some(target_steel_rms),
        configured_material_prior_std: configured_std,
        material_prior_std: calibrated_std,
        unclamped_material_prior_std: Some(unclamped_std),
        material_prior_std_floor: config.material_prior_std_floor,
        material_prior_std_ceiling: config.material_prior_std_ceiling,
        unit_theta_steel_rms_tesla: Some(sensitivity.unit_theta_steel_rms_tesla),
        sensitivity_frobenius_norm_tesla: Some(sensitivity.frobenius_norm_tesla),
        max_abs_sensitivity_tesla: Some(sensitivity.max_abs_sensitivity_tesla),
        theta_column_norms_tesla: sensitivity.theta_column_norms_tesla,
        steel_row_count: sensitivity.steel_row_count,
    })
}

fn team13_calibrated_material_only_prior(
    config: &Team13MaterialOnlyUqConfig,
    calibration_model: &Team13MaterialShapeAugmentedResidualModel,
    deterministic_state: &[f64],
    patch_operator: &Team13ReducedSteelPatchOperator,
) -> Result<Team13MaterialPriorCalibrationReport, String> {
    let configured_std = config.material_prior_std;
    if config.material_prior_calibration == Team13MaterialPriorCalibrationMode::Fixed {
        return Ok(Team13MaterialPriorCalibrationReport {
            mode: config.material_prior_calibration,
            target: config.material_prior_calibration_target,
            target_steel_rms_tesla: None,
            configured_material_prior_std: configured_std,
            material_prior_std: configured_std,
            unclamped_material_prior_std: None,
            material_prior_std_floor: config.material_prior_std_floor,
            material_prior_std_ceiling: config.material_prior_std_ceiling,
            unit_theta_steel_rms_tesla: None,
            sensitivity_frobenius_norm_tesla: None,
            max_abs_sensitivity_tesla: None,
            theta_column_norms_tesla: [0.0; 3],
            steel_row_count: patch_operator.operator.nrows(),
        });
    }

    let target_steel_rms = team13_material_prior_target_steel_rms_from_parts(
        config.material_prior_target_steel_rms_tesla,
        config.material_prior_calibration_target,
        config.operator.observation_std_tesla,
    )?;
    let jacobian = team13_material_steel_output_jacobian(
        calibration_model,
        deterministic_state,
        [0.0; 3],
        patch_operator,
        config.magnitude_smoothing_tesla,
    )?;
    let sensitivity = team13_material_steel_sensitivity_summary(&jacobian);
    let (calibrated_std, unclamped_std) = calibrated_material_prior_std_from_unit_rms(
        configured_std,
        target_steel_rms,
        sensitivity.unit_theta_steel_rms_tesla,
        config.material_prior_std_floor,
        config.material_prior_std_ceiling,
    );

    Ok(Team13MaterialPriorCalibrationReport {
        mode: config.material_prior_calibration,
        target: config.material_prior_calibration_target,
        target_steel_rms_tesla: Some(target_steel_rms),
        configured_material_prior_std: configured_std,
        material_prior_std: calibrated_std,
        unclamped_material_prior_std: Some(unclamped_std),
        material_prior_std_floor: config.material_prior_std_floor,
        material_prior_std_ceiling: config.material_prior_std_ceiling,
        unit_theta_steel_rms_tesla: Some(sensitivity.unit_theta_steel_rms_tesla),
        sensitivity_frobenius_norm_tesla: Some(sensitivity.frobenius_norm_tesla),
        max_abs_sensitivity_tesla: Some(sensitivity.max_abs_sensitivity_tesla),
        theta_column_norms_tesla: sensitivity.theta_column_norms_tesla,
        steel_row_count: sensitivity.steel_row_count,
    })
}

fn team13_material_prior_target_steel_rms(
    config: &Team13JointMaterialUqConfig,
) -> Result<f64, String> {
    if let Some(target) = config.material_prior_target_steel_rms_tesla {
        validate_positive_finite(target, "TEAM 13 explicit material-prior target steel RMS")?;
        return Ok(target);
    }
    team13_material_prior_target_steel_rms_from_parts(
        config.material_prior_target_steel_rms_tesla,
        config.material_prior_calibration_target,
        config.operator.observation_std_tesla,
    )
}

fn team13_material_prior_target_steel_rms_from_parts(
    explicit_target: Option<f64>,
    target: Team13MaterialPriorCalibrationTarget,
    observation_std_tesla: f64,
) -> Result<f64, String> {
    if let Some(value) = explicit_target {
        validate_positive_finite(value, "TEAM 13 explicit material-prior target steel RMS")?;
        return Ok(value);
    }
    match target {
        Team13MaterialPriorCalibrationTarget::ObservationStd => {
            Ok(observation_std_tesla)
        }
        Team13MaterialPriorCalibrationTarget::PublishedGapDifference => {
            Ok(team13_published_steel_gap_difference_rms())
        }
        Team13MaterialPriorCalibrationTarget::Explicit => Err(
            "TEAM 13 material-prior calibration target `explicit` requires --material-prior-target-steel-rms-tesla"
                .to_string(),
        ),
    }
}

fn team13_published_steel_gap_difference_rms() -> f64 {
    let g_052 = team13_published_steel_observations(Team13PublishedSteelGap::G052);
    let g_047 = team13_published_steel_observations(Team13PublishedSteelGap::G047);
    (g_052
        .iter()
        .zip(g_047.iter())
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f64>()
        / TEAM13_OBSERVATION_COUNT as f64)
        .sqrt()
}

fn calibrated_material_prior_std_from_unit_rms(
    configured_std: f64,
    target_rms: f64,
    unit_rms: f64,
    floor: f64,
    ceiling: f64,
) -> (f64, f64) {
    let unclamped = if unit_rms.is_finite() && unit_rms > EPS {
        target_rms / unit_rms
    } else {
        configured_std
    };
    (unclamped.max(floor).min(ceiling), unclamped)
}

struct Team13MaterialSteelSensitivitySummary {
    unit_theta_steel_rms_tesla: f64,
    frobenius_norm_tesla: f64,
    max_abs_sensitivity_tesla: f64,
    theta_column_norms_tesla: [f64; 3],
    steel_row_count: usize,
}

fn team13_material_steel_sensitivity_summary(
    jacobian: &[[f64; 3]],
) -> Team13MaterialSteelSensitivitySummary {
    let mut frobenius_sq = 0.0;
    let mut theta_column_norms_sq = [0.0; 3];
    let mut max_abs = 0.0_f64;
    for row in jacobian {
        for theta in 0..3 {
            let squared = row[theta] * row[theta];
            frobenius_sq += squared;
            theta_column_norms_sq[theta] += squared;
            max_abs = max_abs.max(row[theta].abs());
        }
    }
    let steel_row_count = jacobian.len();
    Team13MaterialSteelSensitivitySummary {
        unit_theta_steel_rms_tesla: if steel_row_count == 0 {
            0.0
        } else {
            (frobenius_sq / steel_row_count as f64).sqrt()
        },
        frobenius_norm_tesla: frobenius_sq.sqrt(),
        max_abs_sensitivity_tesla: max_abs,
        theta_column_norms_tesla: [
            theta_column_norms_sq[0].sqrt(),
            theta_column_norms_sq[1].sqrt(),
            theta_column_norms_sq[2].sqrt(),
        ],
        steel_row_count,
    }
}

fn team13_hard_pde_material_steel_sensitivity(
    calibration_model: &Team13MaterialShapeAugmentedResidualModel,
    deterministic_state: &[f64],
    patch_operator: &Team13ReducedSteelPatchOperator,
) -> Result<Team13MaterialSteelSensitivitySummary, String> {
    let n_state = calibration_model.state_count();
    if deterministic_state.len() != n_state {
        return Err(format!(
            "TEAM 13 material calibration state length {} must match reduced state dimension {}",
            deterministic_state.len(),
            n_state
        ));
    }
    if patch_operator.operator.ncols != n_state {
        return Err(format!(
            "TEAM 13 material calibration steel operator has {} columns but state dimension is {}",
            patch_operator.operator.ncols, n_state
        ));
    }

    let mut joint_state = deterministic_state.to_vec();
    joint_state.extend([0.0; 3]);
    let evaluation = calibration_model.residual_and_jacobian(&joint_state)?;
    if evaluation.jacobian.nrows() != n_state {
        return Err(format!(
            "TEAM 13 material calibration hard-PDE Jacobian must be square; got {} rows and {} state columns",
            evaluation.jacobian.nrows(),
            n_state
        ));
    }

    let mut state_triplets = Vec::new();
    let mut material_columns = vec![vec![0.0; n_state]; 3];
    for (row, col, value) in evaluation.jacobian.triplet_iter() {
        if col < n_state {
            state_triplets.push(SparseTriplet { row, col, value });
        } else if col < n_state + 3 {
            material_columns[col - n_state][row] += value;
        }
    }
    let state_jacobian = SparseTripletMatrix::from_triplets(n_state, n_state, state_triplets);
    let factor = sparse_from_core(&state_jacobian)
        .cholesky_sqrt_lower()
        .map_err(|err| {
            format!("failed to factorize TEAM 13 hard-PDE material calibration Jacobian: {err}")
        })?;

    let mut state_sensitivities = Vec::with_capacity(3);
    for column in material_columns {
        let rhs = column.into_iter().map(|value| -value).collect::<Vec<_>>();
        let solution = factor.solve(&GmrfVector::from_vec(rhs)).map_err(|err| {
            format!("failed to solve TEAM 13 hard-PDE material calibration sensitivity: {err}")
        })?;
        state_sensitivities.push(solution.iter().copied().collect::<Vec<_>>());
    }

    let predictions = patch_operator
        .operator
        .apply(&GmrfVector::from_vec(deterministic_state.to_vec()))
        .map_err(|err| err.to_string())?;
    let mut frobenius_sq = 0.0;
    let mut theta_column_norms_sq = [0.0; 3];
    let mut max_abs = 0.0_f64;
    for (row_index, row) in patch_operator.operator.rows.iter().enumerate() {
        let signed_prediction = predictions[row_index] + patch_operator.bias[row_index];
        let sign = if signed_prediction >= 0.0 { 1.0 } else { -1.0 };
        for theta in 0..3 {
            let sensitivity = sign
                * row
                    .iter()
                    .map(|(col, value)| *value * state_sensitivities[theta][*col])
                    .sum::<f64>();
            let squared = sensitivity * sensitivity;
            frobenius_sq += squared;
            theta_column_norms_sq[theta] += squared;
            max_abs = max_abs.max(sensitivity.abs());
        }
    }
    let steel_row_count = patch_operator.operator.nrows();
    let frobenius_norm = frobenius_sq.sqrt();
    let unit_theta_steel_rms = if steel_row_count == 0 {
        0.0
    } else {
        (frobenius_sq / steel_row_count as f64).sqrt()
    };
    Ok(Team13MaterialSteelSensitivitySummary {
        unit_theta_steel_rms_tesla: unit_theta_steel_rms,
        frobenius_norm_tesla: frobenius_norm,
        max_abs_sensitivity_tesla: max_abs,
        theta_column_norms_tesla: [
            theta_column_norms_sq[0].sqrt(),
            theta_column_norms_sq[1].sqrt(),
            theta_column_norms_sq[2].sqrt(),
        ],
        steel_row_count,
    })
}

fn team13_material_parameter_reports(
    anchors: [f64; 3],
    prior_std: f64,
    posterior_mean: [f64; 3],
    posterior_covariance: [[f64; 3]; 3],
) -> Vec<Team13MaterialParameterPosteriorReport> {
    (0..3)
        .map(|index| Team13MaterialParameterPosteriorReport {
            name: format!("theta_h_{}", index),
            anchor_b_tesla: anchors[index],
            prior_mean: 0.0,
            posterior_mean: posterior_mean[index],
            prior_std,
            posterior_std: posterior_covariance[index][index].max(0.0).sqrt(),
        })
        .collect()
}

type Team13JointMaterialPatchReports = (
    [[f64; 3]; 3],
    [[f64; 3]; 3],
    Vec<Team13JointSteelPatchVarianceReport>,
);

fn team13_joint_material_patch_reports(
    posterior: &mut NonlinearLaplaceResult,
    patch_operator: &Team13ReducedSteelPatchOperator,
    state: &[f64],
    observations: &[f64; TEAM13_OBSERVATION_COUNT],
    state_dimension: usize,
) -> Result<Team13JointMaterialPatchReports, String> {
    let predictions = patch_operator
        .operator
        .apply(&GmrfVector::from_vec(state.to_vec()))
        .map_err(|err| err.to_string())?;
    let mut rows = Vec::with_capacity(3 + patch_operator.operator.nrows());
    for theta in 0..3 {
        rows.push(vec![(state_dimension + theta, 1.0)]);
    }
    for (row_index, row) in patch_operator.operator.rows.iter().enumerate() {
        let signed_prediction = predictions[row_index] + patch_operator.bias[row_index];
        let sign = if signed_prediction >= 0.0 { 1.0 } else { -1.0 };
        rows.push(
            row.iter()
                .map(|(col, value)| (*col, sign * *value))
                .collect(),
        );
    }
    let joint_operator =
        SparseRowOperator::new(state_dimension + 3, rows).map_err(|err| err.to_string())?;
    let covariance = posterior
        .posterior_gmrf
        .exact_transformed_covariance(&joint_operator)
        .map_err(|err| err.to_string())?;
    let mut theta_covariance = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            theta_covariance[row][col] = covariance[(row, col)];
        }
    }
    let theta_correlation = correlation_from_covariance(theta_covariance);
    let theta_inverse = invert_3x3(theta_covariance);
    let reports = patch_operator
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let output_row = 3 + index;
            let signed_prediction = predictions[index] + patch_operator.bias[index];
            let prediction = signed_prediction.abs();
            let total_variance = covariance[(output_row, output_row)].max(0.0);
            let theta_covariance_with_output = [
                covariance[(0, output_row)],
                covariance[(1, output_row)],
                covariance[(2, output_row)],
            ];
            let material_explained_variance = theta_inverse
                .map(|inverse| quadratic_form_3(theta_covariance_with_output, inverse).max(0.0))
                .unwrap_or(0.0)
                .min(total_variance);
            let state_conditional_variance =
                (total_variance - material_explained_variance).max(0.0);
            let observed = observations[index];
            Ok(Team13JointSteelPatchVarianceReport {
                name: definition.name.clone(),
                group: team13_steel_surface_group(index)?,
                prediction,
                signed_prediction,
                observed,
                residual: prediction - observed,
                total_variance,
                state_conditional_variance,
                material_explained_variance,
                posterior_std: total_variance.sqrt(),
                row_nnz: patch_operator.row_nnz[index],
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((theta_covariance, theta_correlation, reports))
}

type Team13IdentifiableJointPatchReports =
    (Vec<Vec<f64>>, Vec<Team13IdentifiableJointSteelPatchReport>);

fn team13_identifiable_joint_patch_reports(
    posterior: &mut NonlinearLaplaceResult,
    patch_operator: &Team13ReducedSteelPatchOperator,
    state: &[f64],
    observations: &[f64; TEAM13_OBSERVATION_COUNT],
    state_dimension: usize,
    eta_count: usize,
) -> Result<Team13IdentifiableJointPatchReports, String> {
    if eta_count == 0 {
        return Err(
            "TEAM 13 identifiable joint patch reports require at least one eta mode".to_string(),
        );
    }
    let predictions = patch_operator
        .operator
        .apply(&GmrfVector::from_vec(state.to_vec()))
        .map_err(|err| err.to_string())?;
    let mut rows = Vec::with_capacity(eta_count + patch_operator.operator.nrows());
    for eta in 0..eta_count {
        rows.push(vec![(state_dimension + eta, 1.0)]);
    }
    for (row_index, row) in patch_operator.operator.rows.iter().enumerate() {
        let signed_prediction = predictions[row_index] + patch_operator.bias[row_index];
        let sign = if signed_prediction >= 0.0 { 1.0 } else { -1.0 };
        rows.push(
            row.iter()
                .map(|(col, value)| (*col, sign * *value))
                .collect(),
        );
    }
    let joint_operator =
        SparseRowOperator::new(state_dimension + eta_count, rows).map_err(|err| err.to_string())?;
    let covariance = posterior
        .posterior_gmrf
        .exact_transformed_covariance(&joint_operator)
        .map_err(|err| err.to_string())?;
    let mut eta_covariance = vec![vec![0.0; eta_count]; eta_count];
    for row in 0..eta_count {
        for col in 0..eta_count {
            eta_covariance[row][col] = covariance[(row, col)];
        }
    }
    let eta_inverse = invert_dense_matrix(&eta_covariance).ok();
    let reports = patch_operator
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let output_row = eta_count + index;
            let signed_prediction = predictions[index] + patch_operator.bias[index];
            let prediction = signed_prediction.abs();
            let total_variance = covariance[(output_row, output_row)].max(0.0);
            let covariance_with_output = (0..eta_count)
                .map(|eta| covariance[(eta, output_row)])
                .collect::<Vec<_>>();
            let eta_explained_variance = eta_inverse
                .as_ref()
                .map(|inverse| {
                    dense_quadratic_form(&covariance_with_output, inverse)
                        .unwrap_or(0.0)
                        .max(0.0)
                })
                .unwrap_or(0.0)
                .min(total_variance);
            let state_conditional_variance = (total_variance - eta_explained_variance).max(0.0);
            let observed = observations[index];
            Ok(Team13IdentifiableJointSteelPatchReport {
                name: definition.name.clone(),
                group: team13_steel_surface_group(index)?,
                prediction,
                signed_prediction,
                observed,
                residual: prediction - observed,
                total_variance,
                state_conditional_variance,
                eta_explained_variance,
                posterior_std: total_variance.sqrt(),
                row_nnz: patch_operator.row_nnz[index],
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((eta_covariance, reports))
}

fn team13_bh_curve_bands(
    anchors: [f64; 3],
    theta_mean: [f64; 3],
    theta_covariance: [[f64; 3]; 3],
) -> Result<Vec<Team13BhCurveBandReport>, String> {
    let nominal = Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES)?
        .with_log_h_shape(anchors, [0.0; 3])?;
    let posterior = Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES)?
        .with_log_h_shape(anchors, theta_mean)?;
    let mut rows = Vec::new();
    for index in 0..=60 {
        let b = 3.0 * index as f64 / 60.0;
        let basis = posterior.log_h_shape_basis(b);
        let log_variance = quadratic_form_3(basis, theta_covariance).max(0.0);
        let log_std = log_variance.sqrt();
        let posterior_h = posterior.h_ampere_per_meter(b);
        rows.push(Team13BhCurveBandReport {
            b_tesla: b,
            nominal_h_ampere_per_meter: nominal.h_ampere_per_meter(b),
            posterior_mean_h_ampere_per_meter: posterior_h,
            posterior_std_h_ampere_per_meter: posterior_h * log_std,
            lower_2sigma_h_ampere_per_meter: posterior_h * (-2.0 * log_std).exp(),
            upper_2sigma_h_ampere_per_meter: posterior_h * (2.0 * log_std).exp(),
        });
    }
    Ok(rows)
}

fn correlation_from_covariance(covariance: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let std = [
        covariance[0][0].max(0.0).sqrt(),
        covariance[1][1].max(0.0).sqrt(),
        covariance[2][2].max(0.0).sqrt(),
    ];
    let mut correlation = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            correlation[row][col] = safe_ratio(covariance[row][col], std[row] * std[col]);
        }
    }
    correlation
}

fn quadratic_form_3(vector: [f64; 3], matrix: [[f64; 3]; 3]) -> f64 {
    let mv = [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ];
    vector[0] * mv[0] + vector[1] * mv[1] + vector[2] * mv[2]
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = matrix;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() <= 1.0e-18 || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det,
        ],
    ])
}

fn team13_material_gap_case_total_weight(config: &Team13MaterialGapUqConfig) -> f64 {
    config
        .gap_cases
        .iter()
        .flat_map(|gap_case| {
            config
                .material_nodes
                .iter()
                .map(move |material_node| gap_case.weight * material_node.weight)
        })
        .sum()
}

fn sanitize_path_token(value: &str) -> String {
    let token: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if token.is_empty() {
        "case".to_string()
    } else {
        token
    }
}

fn team13_operator_patch_rmse_vs_gap(
    reports: &[Team13OperatorSteelPatchVarianceReport],
    gap: Team13PublishedSteelGap,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for report in reports {
        let observed = match gap {
            Team13PublishedSteelGap::G052 => report.observed_g_052,
            Team13PublishedSteelGap::G047 => report.observed_g_047,
        };
        let residual = report.prediction - observed;
        sum += residual * residual;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        (sum / count as f64).sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
struct Team13WeightedPatchSample<'a> {
    gap_label: &'a str,
    material_label: &'a str,
    weight: f64,
    prediction: f64,
    within_variance: f64,
}

#[derive(Debug, Clone, Copy)]
struct Team13TwoFactorVarianceTerms {
    mean: f64,
    expected_within: f64,
    between_gap: f64,
    between_material: f64,
    interaction: f64,
    total_between: f64,
    total: f64,
}

fn team13_two_factor_weighted_variance(
    samples: &[Team13WeightedPatchSample<'_>],
) -> Result<Team13TwoFactorVarianceTerms, String> {
    if samples.is_empty() {
        return Err("TEAM 13 variance decomposition requires at least one sample".to_string());
    }
    let total_weight: f64 = samples.iter().map(|sample| sample.weight).sum();
    validate_positive_finite(total_weight, "TEAM 13 variance decomposition total weight")?;

    let mut mean = 0.0;
    for sample in samples {
        if !sample.prediction.is_finite() {
            return Err("TEAM 13 variance decomposition prediction is not finite".to_string());
        }
        if !sample.within_variance.is_finite() || sample.within_variance < -EPS {
            return Err("TEAM 13 variance decomposition variance is invalid".to_string());
        }
        mean += sample.weight * sample.prediction / total_weight;
    }

    let mut expected_within = 0.0;
    let mut total_between = 0.0;
    let mut gap_sums: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    let mut material_sums: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for sample in samples {
        let weight = sample.weight / total_weight;
        expected_within += weight * sample.within_variance.max(0.0);
        total_between += weight * (sample.prediction - mean).powi(2);
        let gap_entry = gap_sums.entry(sample.gap_label).or_insert((0.0, 0.0));
        gap_entry.0 += weight;
        gap_entry.1 += weight * sample.prediction;
        let material_entry = material_sums
            .entry(sample.material_label)
            .or_insert((0.0, 0.0));
        material_entry.0 += weight;
        material_entry.1 += weight * sample.prediction;
    }

    let between_gap = gap_sums
        .values()
        .map(|(weight, weighted_sum)| {
            let group_mean = weighted_sum / weight;
            weight * (group_mean - mean).powi(2)
        })
        .sum::<f64>();
    let between_material = material_sums
        .values()
        .map(|(weight, weighted_sum)| {
            let group_mean = weighted_sum / weight;
            weight * (group_mean - mean).powi(2)
        })
        .sum::<f64>();
    let raw_interaction = total_between - between_gap - between_material;
    let scale = total_between
        .abs()
        .max(between_gap.abs())
        .max(between_material.abs())
        .max(1.0);
    let interaction = if raw_interaction.abs() <= 1.0e-12 * scale {
        0.0
    } else {
        raw_interaction.max(0.0)
    };
    Ok(Team13TwoFactorVarianceTerms {
        mean,
        expected_within,
        between_gap,
        between_material,
        interaction,
        total_between,
        total: expected_within + total_between,
    })
}

fn team13_material_gap_variance_decomposition(
    cases: &[Team13MaterialGapUqCaseResult],
) -> Result<Vec<Team13MaterialGapVarianceDecomposition>, String> {
    let reference = cases
        .first()
        .ok_or_else(|| "TEAM 13 material/gap UQ produced no cases".to_string())?
        .operator_result
        .steel_patch_reports
        .as_slice();
    let mut decompositions = Vec::with_capacity(reference.len());

    for (patch_index, reference_report) in reference.iter().enumerate() {
        let mut samples = Vec::with_capacity(cases.len());
        for case in cases {
            let report = case
                .operator_result
                .steel_patch_reports
                .get(patch_index)
                .ok_or_else(|| {
                    format!(
                        "TEAM 13 material/gap case `{}`/`{}` is missing steel patch row {}",
                        case.gap_label, case.material_label, patch_index
                    )
                })?;
            if report.name != reference_report.name || report.group != reference_report.group {
                return Err(format!(
                    "TEAM 13 material/gap patch mismatch at row {patch_index}: expected `{}`/{}, got `{}`/{}",
                    reference_report.name,
                    reference_report.group.as_str(),
                    report.name,
                    report.group.as_str()
                ));
            }
            samples.push(Team13WeightedPatchSample {
                gap_label: &case.gap_label,
                material_label: &case.material_label,
                weight: case.normalized_weight,
                prediction: report.prediction,
                within_variance: report.posterior_variance,
            });
        }
        let terms = team13_two_factor_weighted_variance(&samples)?;
        decompositions.push(Team13MaterialGapVarianceDecomposition {
            name: reference_report.name.clone(),
            group: reference_report.group,
            mean_prediction: terms.mean,
            expected_operator_variance: terms.expected_within,
            between_gap_variance: terms.between_gap,
            between_material_variance: terms.between_material,
            gap_material_interaction_variance: terms.interaction,
            total_between_case_variance: terms.total_between,
            total_variance: terms.total,
            operator_fraction: safe_ratio(terms.expected_within, terms.total),
            gap_fraction: safe_ratio(terms.between_gap, terms.total),
            material_fraction: safe_ratio(terms.between_material, terms.total),
            interaction_fraction: safe_ratio(terms.interaction, terms.total),
        });
    }

    Ok(decompositions)
}

fn validate_positive_finite(value: f64, name: &str) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{name} must be finite and positive"));
    }
    Ok(())
}

fn team13_measurement_mode_from_steel_quadrature(
    mode: Team13SteelObservationQuadratureMode,
) -> Team13MeasurementMode {
    match mode {
        Team13SteelObservationQuadratureMode::NgsolveStyle => Team13MeasurementMode::BenchmarkExact,
        Team13SteelObservationQuadratureMode::FaceCochain => Team13MeasurementMode::FaceCochain,
    }
}

fn team13_operator_pde_noise(
    config: &Team13OperatorUncertaintyConfig,
    state_mass_inverse: Option<&FeecCsr>,
) -> Result<GaussianNoiseModel, String> {
    match config.pde_residual_weighting {
        Team13PdeResidualWeighting::Euclidean => {
            Ok(GaussianNoiseModel::ScalarVariance(config.pde_variance))
        }
        Team13PdeResidualWeighting::MassInverse
        | Team13PdeResidualWeighting::MassInverseTraceNormalized => {
            let mass_inverse = state_mass_inverse.ok_or_else(|| {
                "mass-weighted operator uncertainty residual requires state_mass_inverse"
                    .to_string()
            })?;
            let mut scale = 1.0 / config.pde_variance;
            if config.pde_residual_weighting
                == Team13PdeResidualWeighting::MassInverseTraceNormalized
            {
                scale *= reciprocal_mean_diagonal(mass_inverse)?;
            }
            Ok(GaussianNoiseModel::Precision(scale_triplet_matrix(
                &csr_to_triplet(mass_inverse),
                scale,
            )))
        }
    }
}

#[derive(Clone, Debug)]
struct Team13ReducedSteelPatchOperator {
    definitions: Vec<Team13SurfaceMeasurementDefinition>,
    operator: SparseRowOperator,
    bias: Vec<f64>,
    row_nnz: Vec<usize>,
}

fn build_team13_reduced_steel_patch_operator(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    layout: &DofLayout,
    quadrature: Team13SteelObservationQuadratureMode,
) -> Result<Team13ReducedSteelPatchOperator, String> {
    let definitions = team13_surface_measurement_definitions(None)?;
    let cell_geometries = top_cell_geometries(topology, coords);
    let mut triplets = Vec::new();
    let mut row_nnz = Vec::with_capacity(definitions.len());
    for (row, definition) in definitions.iter().enumerate() {
        let entries = match quadrature {
            Team13SteelObservationQuadratureMode::NgsolveStyle => {
                surface_flux_component_row(&cell_geometries, &operators.b_cochain, definition)?
            }
            Team13SteelObservationQuadratureMode::FaceCochain => {
                surface_face_cochain_row(topology, coords, &operators.b_cochain, definition)?.row
            }
        };
        row_nnz.push(entries.len());
        for (col, value) in entries {
            triplets.push(SparseTriplet { row, col, value });
        }
    }
    let full_operator =
        SparseTripletMatrix::from_triplets(definitions.len(), topology.nsimplices(1), triplets);
    let full_bias = FeecVector::zeros(definitions.len());
    let (reduced_operator, bias) = restrict_columns_and_fold_fixed(
        &core_triplet_to_feec_csr(&full_operator),
        &full_bias,
        layout,
    )?;
    Ok(Team13ReducedSteelPatchOperator {
        definitions,
        operator: triplet_to_sparse_row_operator(&csr_to_triplet(&reduced_operator))?,
        bias: bias.iter().copied().collect(),
        row_nnz,
    })
}

fn team13_operator_steel_patch_variances(
    patch_operator: &Team13ReducedSteelPatchOperator,
    state: &[f64],
    prior_factor: &SparseCholeskyFactor,
    posterior: &NonlinearLaplaceResult,
) -> Result<Vec<Team13OperatorSteelPatchVarianceReport>, String> {
    let prior_variance = exact_solve_transformed_diag(prior_factor, &patch_operator.operator)
        .map_err(|err| err.to_string())?
        .values;
    let posterior_factor = posterior
        .posterior_gmrf
        .precision_factor()
        .ok_or_else(|| "operator uncertainty posterior precision factor is missing".to_string())?;
    let posterior_variance =
        exact_solve_transformed_diag(posterior_factor, &patch_operator.operator)
            .map_err(|err| err.to_string())?
            .values;
    let predictions = patch_operator
        .operator
        .apply(&GmrfVector::from_vec(state.to_vec()))
        .map_err(|err| err.to_string())?;
    let observed_g_052 = team13_published_steel_observations(Team13PublishedSteelGap::G052);
    let observed_g_047 = team13_published_steel_observations(Team13PublishedSteelGap::G047);
    patch_operator
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let signed_prediction = predictions[index] + patch_operator.bias[index];
            let prediction = signed_prediction.abs();
            let posterior_variance = posterior_variance[index].max(0.0);
            let posterior_std = posterior_variance.sqrt();
            let residual_g_052 = prediction - observed_g_052[index];
            let residual_g_047 = prediction - observed_g_047[index];
            Ok(Team13OperatorSteelPatchVarianceReport {
                name: definition.name.clone(),
                group: team13_steel_surface_group(index)?,
                prediction,
                signed_prediction,
                prior_variance: prior_variance[index].max(0.0),
                posterior_variance,
                posterior_std,
                observed_g_052: observed_g_052[index],
                observed_g_047: observed_g_047[index],
                residual_g_052,
                residual_g_047,
                abs_residual_over_std_g_052: safe_ratio(residual_g_052.abs(), posterior_std),
                abs_residual_over_std_g_047: safe_ratio(residual_g_047.abs(), posterior_std),
                row_nnz: patch_operator.row_nnz[index],
            })
        })
        .collect()
}

fn build_interleaved_b_vector_operator(
    operators: &Team13Operators,
    layout: &DofLayout,
) -> Result<SparseRowOperator, String> {
    let cell_count = operators
        .b_components
        .first()
        .map_or(0, SparseRowOperator::nrows);
    let mut reduced_index = vec![None; layout.full_dimension];
    for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
        reduced_index[full] = Some(reduced);
    }
    let mut rows = Vec::with_capacity(3 * cell_count);
    for cell in 0..cell_count {
        for component in 0..3 {
            rows.push(
                operators.b_components[component].rows[cell]
                    .iter()
                    .filter_map(|(full_col, value)| {
                        reduced_index[*full_col].map(|col| (col, *value))
                    })
                    .collect(),
            );
        }
    }
    SparseRowOperator::new(layout.reduced_dimension(), rows).map_err(|err| err.to_string())
}

fn estimate_team13_b_vector_variances(
    posterior: &mut NonlinearLaplaceResult,
    operator: &SparseRowOperator,
    config: &LinearPdeVarianceConfig,
) -> Result<(String, GmrfVector), String> {
    match config.mode {
        LinearPdeVarianceMode::Exact | LinearPdeVarianceMode::ExactSolves => {
            let factor = posterior
                .posterior_gmrf
                .precision_factor()
                .ok_or_else(|| "exact B variance requires a posterior factor".to_string())?;
            exact_solve_transformed_diag(factor, operator)
                .map(|estimate| ("exact-solves".to_string(), estimate.values))
                .map_err(|err| err.to_string())
        }
        LinearPdeVarianceMode::SelectedInverse => {
            let factor = posterior.posterior_gmrf.precision_factor().ok_or_else(|| {
                "selected-inverse B variance requires a posterior factor".to_string()
            })?;
            match selected_inverse_transformed_diag(factor, operator) {
                Ok(selected) => match selected.estimate {
                    Some(estimate) => Ok(("selected-inverse".to_string(), estimate.values)),
                    None => estimate_hutchinson_transformed_variances(
                        &mut posterior.posterior_gmrf,
                        operator,
                        config.num_variance_probes,
                        config.variance_batch_count,
                        config.rng_seed,
                        ProbeDistribution::Rademacher,
                    )
                    .map(|estimate| {
                        (
                            "selected-inverse-fallback-hutchinson".to_string(),
                            estimate.values,
                        )
                    })
                    .map_err(|err| err.to_string()),
                },
                Err(_) => estimate_hutchinson_transformed_variances(
                    &mut posterior.posterior_gmrf,
                    operator,
                    config.num_variance_probes,
                    config.variance_batch_count,
                    config.rng_seed,
                    ProbeDistribution::Rademacher,
                )
                .map(|estimate| {
                    (
                        "selected-inverse-fallback-hutchinson".to_string(),
                        estimate.values,
                    )
                })
                .map_err(|err| err.to_string()),
            }
        }
        LinearPdeVarianceMode::MonteCarlo => estimate_monte_carlo_transformed_variances(
            &mut posterior.posterior_gmrf,
            operator,
            config.num_variance_probes,
            config.variance_batch_count,
            config.rng_seed,
        )
        .map(|estimate| ("monte-carlo".to_string(), estimate.values))
        .map_err(|err| err.to_string()),
        LinearPdeVarianceMode::Hutchinson | LinearPdeVarianceMode::LocalRbmc => {
            estimate_hutchinson_transformed_variances(
                &mut posterior.posterior_gmrf,
                operator,
                config.num_variance_probes,
                config.variance_batch_count,
                config.rng_seed,
                ProbeDistribution::Rademacher,
            )
            .map(|estimate| ("hutchinson".to_string(), estimate.values))
            .map_err(|err| err.to_string())
        }
    }
}

fn cell_trace_variances_from_interleaved(values: &GmrfVector) -> Result<Vec<f64>, String> {
    if values.len() % 3 != 0 {
        return Err(format!(
            "interleaved B variance length {} is not divisible by 3",
            values.len()
        ));
    }
    Ok((0..values.len() / 3)
        .map(|cell| {
            (values[3 * cell].max(0.0)
                + values[3 * cell + 1].max(0.0)
                + values[3 * cell + 2].max(0.0))
            .max(0.0)
        })
        .collect())
}

fn evaluate_cell_b_field(
    operators: &Team13Operators,
    state: &[f64],
) -> Result<Vec<[f64; 3]>, String> {
    let state = FeecVector::from_vec(state.to_vec());
    let bx = apply_operator_to_feec(&operators.b_components[0], &state)?;
    let by = apply_operator_to_feec(&operators.b_components[1], &state)?;
    let bz = apply_operator_to_feec(&operators.b_components[2], &state)?;
    Ok((0..bx.len())
        .map(|cell| [bx[cell], by[cell], bz[cell]])
        .collect())
}

#[derive(Debug, Clone)]
struct Team13OperatorCellDiagnostic {
    variance: Option<f64>,
    b_magnitude: f64,
    gradient_indicator: f64,
    iron: bool,
    interface: bool,
    central_gap: bool,
    steel_corner: bool,
    measurement_mid_sheet: bool,
    measurement_back_right_top: bool,
    measurement_back_right_edge: bool,
    high_gradient: bool,
    low_b_magnitude: bool,
}

fn build_team13_operator_region_diagnostics(
    topology: &Complex,
    coords: &MeshCoords,
    b_field: &[[f64; 3]],
    cell_variances: Option<&[f64]>,
) -> Result<Vec<Team13OperatorCellDiagnostic>, String> {
    let cell_count = topology.nsimplices(3);
    if b_field.len() != cell_count {
        return Err(format!(
            "B field cell count {} does not match topology cell count {}",
            b_field.len(),
            cell_count
        ));
    }
    if let Some(variances) = cell_variances {
        if variances.len() != cell_count {
            return Err(format!(
                "B variance cell count {} does not match topology cell count {}",
                variances.len(),
                cell_count
            ));
        }
    }
    let barycenters = top_cell_barycenters(topology, coords);
    let iron = barycenters
        .iter()
        .map(|point| is_iron_xyz(*point))
        .collect::<Vec<_>>();
    let face_cells = cell_face_adjacency(topology);
    let mut interface = vec![false; cell_count];
    let mut gradient_indicator = vec![0.0_f64; cell_count];
    let b_magnitudes = b_field.iter().map(|b| vector_norm(*b)).collect::<Vec<_>>();
    for cells in &face_cells {
        if cells.len() != 2 {
            continue;
        }
        let lhs = cells[0];
        let rhs = cells[1];
        if iron[lhs] != iron[rhs] {
            interface[lhs] = true;
            interface[rhs] = true;
        }
        let distance = point_distance_squared(barycenters[lhs], barycenters[rhs]).sqrt();
        if distance > EPS {
            let indicator = (b_magnitudes[lhs] - b_magnitudes[rhs]).abs() / distance;
            gradient_indicator[lhs] = gradient_indicator[lhs].max(indicator);
            gradient_indicator[rhs] = gradient_indicator[rhs].max(indicator);
        }
    }
    let high_gradient_threshold = finite_quantile(gradient_indicator.clone(), 0.90);
    let low_b_threshold = finite_quantile(b_magnitudes.clone(), 0.10);
    let definitions = team13_surface_measurement_definitions(None)?;

    barycenters
        .iter()
        .enumerate()
        .map(|(cell, point)| {
            let mut measurement_mid_sheet = false;
            let mut measurement_back_right_top = false;
            let mut measurement_back_right_edge = false;
            for (index, definition) in definitions.iter().enumerate() {
                if !point_near_measurement_patch(*point, definition, 2.0e-3) {
                    continue;
                }
                match team13_steel_surface_group(index)? {
                    Team13SteelSurfaceGroup::MidSheet => measurement_mid_sheet = true,
                    Team13SteelSurfaceGroup::BackRightTop => measurement_back_right_top = true,
                    Team13SteelSurfaceGroup::BackRightEdge => measurement_back_right_edge = true,
                }
            }
            Ok(Team13OperatorCellDiagnostic {
                variance: cell_variances.map(|variances| variances[cell].max(0.0)),
                b_magnitude: b_magnitudes[cell],
                gradient_indicator: gradient_indicator[cell],
                iron: iron[cell],
                interface: interface[cell],
                central_gap: point_in_central_gap_band(*point),
                steel_corner: point_in_steel_corner_band(*point),
                measurement_mid_sheet,
                measurement_back_right_top,
                measurement_back_right_edge,
                high_gradient: gradient_indicator[cell] >= high_gradient_threshold,
                low_b_magnitude: b_magnitudes[cell] <= low_b_threshold,
            })
        })
        .collect()
}

fn summarize_team13_operator_regions(
    diagnostics: &[Team13OperatorCellDiagnostic],
) -> (
    Vec<Team13OperatorRegionVarianceSummary>,
    Vec<Team13OperatorVarianceIndicatorCorrelation>,
) {
    let iron_bulk_values = diagnostics
        .iter()
        .filter(|cell| cell.iron && !cell.interface && !cell.central_gap && !cell.steel_corner)
        .filter_map(|cell| cell.variance)
        .collect::<Vec<_>>();
    let air_bulk_values = diagnostics
        .iter()
        .filter(|cell| !cell.iron && !cell.interface && !cell.central_gap && !cell.steel_corner)
        .filter_map(|cell| cell.variance)
        .collect::<Vec<_>>();
    let iron_bulk_mean = mean_or_nan(&iron_bulk_values);
    let air_bulk_mean = mean_or_nan(&air_bulk_values);
    let region_specs: [(&str, fn(&Team13OperatorCellDiagnostic) -> bool); 12] = [
        ("iron_bulk", |cell| {
            cell.iron && !cell.interface && !cell.central_gap && !cell.steel_corner
        }),
        ("air_bulk", |cell| {
            !cell.iron && !cell.interface && !cell.central_gap && !cell.steel_corner
        }),
        ("iron_air_interface_band", |cell| cell.interface),
        ("central_gap_band", |cell| cell.central_gap),
        ("steel_corner_edge_band", |cell| cell.steel_corner),
        ("measurement_mid_sheet_band", |cell| {
            cell.measurement_mid_sheet
        }),
        ("measurement_back_right_top_band", |cell| {
            cell.measurement_back_right_top
        }),
        ("measurement_back_right_edge_band", |cell| {
            cell.measurement_back_right_edge
        }),
        ("high_gradient_top_decile", |cell| cell.high_gradient),
        ("low_b_magnitude_bottom_decile", |cell| cell.low_b_magnitude),
        ("all_iron", |cell| cell.iron),
        ("all_air", |cell| !cell.iron),
    ];
    let summaries = region_specs
        .iter()
        .map(|(name, predicate)| {
            summarize_operator_region(
                name,
                diagnostics.iter().filter(|cell| predicate(cell)),
                iron_bulk_mean,
                air_bulk_mean,
            )
        })
        .collect::<Vec<_>>();
    let indicator_correlations = [
        ("iron", indicator_bool(diagnostics, |cell| cell.iron)),
        (
            "iron_air_interface_band",
            indicator_bool(diagnostics, |cell| cell.interface),
        ),
        (
            "central_gap_band",
            indicator_bool(diagnostics, |cell| cell.central_gap),
        ),
        (
            "steel_corner_edge_band",
            indicator_bool(diagnostics, |cell| cell.steel_corner),
        ),
        (
            "measurement_mid_sheet_band",
            indicator_bool(diagnostics, |cell| cell.measurement_mid_sheet),
        ),
        (
            "measurement_back_right_top_band",
            indicator_bool(diagnostics, |cell| cell.measurement_back_right_top),
        ),
        (
            "measurement_back_right_edge_band",
            indicator_bool(diagnostics, |cell| cell.measurement_back_right_edge),
        ),
        (
            "high_gradient_indicator",
            diagnostics
                .iter()
                .map(|cell| cell.gradient_indicator)
                .collect(),
        ),
        (
            "b_magnitude",
            diagnostics.iter().map(|cell| cell.b_magnitude).collect(),
        ),
    ]
    .into_iter()
    .map(|(name, indicator)| summarize_indicator_correlation(name, diagnostics, &indicator))
    .collect();
    (summaries, indicator_correlations)
}

fn summarize_operator_region<'a>(
    name: &str,
    cells: impl Iterator<Item = &'a Team13OperatorCellDiagnostic>,
    iron_bulk_mean: f64,
    air_bulk_mean: f64,
) -> Team13OperatorRegionVarianceSummary {
    let selected = cells.collect::<Vec<_>>();
    let variances = selected
        .iter()
        .filter_map(|cell| cell.variance)
        .collect::<Vec<_>>();
    let stds = variances
        .iter()
        .map(|variance| variance.max(0.0).sqrt())
        .collect::<Vec<_>>();
    let b_magnitudes = selected
        .iter()
        .map(|cell| cell.b_magnitude)
        .collect::<Vec<_>>();
    let gradients = selected
        .iter()
        .map(|cell| cell.gradient_indicator)
        .collect::<Vec<_>>();
    let mean_variance = mean_or_nan(&variances);
    Team13OperatorRegionVarianceSummary {
        region: name.to_string(),
        count: selected.len(),
        mean_variance,
        median_variance: finite_quantile(variances.clone(), 0.50),
        p90_variance: finite_quantile(variances.clone(), 0.90),
        max_variance: variances
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::NAN, nan_max),
        mean_std: mean_or_nan(&stds),
        mean_b_magnitude: mean_or_nan(&b_magnitudes),
        mean_gradient_indicator: mean_or_nan(&gradients),
        variance_ratio_to_iron_bulk: safe_ratio(mean_variance, iron_bulk_mean),
        variance_ratio_to_air_bulk: safe_ratio(mean_variance, air_bulk_mean),
    }
}

fn summarize_indicator_correlation(
    name: &str,
    diagnostics: &[Team13OperatorCellDiagnostic],
    indicator: &[f64],
) -> Team13OperatorVarianceIndicatorCorrelation {
    let mut x = Vec::new();
    let mut variances = Vec::new();
    let mut stds = Vec::new();
    for (cell, indicator) in diagnostics.iter().zip(indicator.iter().copied()) {
        let Some(variance) = cell.variance else {
            continue;
        };
        if indicator.is_finite() && variance.is_finite() {
            x.push(indicator);
            variances.push(variance);
            stds.push(variance.max(0.0).sqrt());
        }
    }
    let mut positive = Vec::new();
    let mut zero = Vec::new();
    for (indicator, variance) in x.iter().copied().zip(variances.iter().copied()) {
        if indicator > 0.0 {
            positive.push(variance);
        } else {
            zero.push(variance);
        }
    }
    Team13OperatorVarianceIndicatorCorrelation {
        indicator: name.to_string(),
        count: x.len(),
        pearson_with_variance: pearson_correlation(&x, &variances),
        pearson_with_std: pearson_correlation(&x, &stds),
        indicator_mean: mean_or_nan(&x),
        mean_variance_indicator_positive: mean_or_nan(&positive),
        mean_variance_indicator_zero: mean_or_nan(&zero),
    }
}

fn indicator_bool(
    diagnostics: &[Team13OperatorCellDiagnostic],
    predicate: fn(&Team13OperatorCellDiagnostic) -> bool,
) -> Vec<f64> {
    diagnostics
        .iter()
        .map(|cell| if predicate(cell) { 1.0 } else { 0.0 })
        .collect()
}

fn cell_face_adjacency(topology: &Complex) -> Vec<Vec<usize>> {
    let mut face_cells = vec![Vec::new(); topology.nsimplices(2)];
    for cell in topology.skeleton(3).handle_iter() {
        let cell_index = cell.kidx();
        for face in cell.mesh_subsimps(2) {
            face_cells[face.kidx()].push(cell_index);
        }
    }
    face_cells
}

fn is_iron_xyz(point: [f64; 3]) -> bool {
    let point = FeecVector::from_column_slice(&point);
    is_iron_point(point.as_view())
}

fn point_in_central_gap_band(point: [f64; 3]) -> bool {
    in_range(point[0], (0.0016 - 1.0e-3, 0.0021 + 1.0e-3))
        && in_range(point[1], (0.015 - 2.0e-3, 0.025 + 2.0e-3))
        && in_range(point[2], (-0.0632, 0.0632))
}

fn point_in_steel_corner_band(point: [f64; 3]) -> bool {
    let band = 2.0e-3;
    let steel_x = [
        -0.1253, -0.1221, -0.0021, -0.0016, 0.0016, 0.0021, 0.1221, 0.1253,
    ];
    let steel_y = [-0.065, -0.025, -0.015, 0.015, 0.025, 0.065];
    let steel_z = [-0.0632, -0.0600, 0.0, 0.0600, 0.0632];
    let near_x = steel_x
        .iter()
        .any(|target| (point[0] - target).abs() <= band);
    let near_y = steel_y
        .iter()
        .any(|target| (point[1] - target).abs() <= band);
    let near_z = steel_z
        .iter()
        .any(|target| (point[2] - target).abs() <= band);
    (near_x && near_y) || (near_x && near_z) || (near_y && near_z)
}

fn point_near_measurement_patch(
    point: [f64; 3],
    definition: &Team13SurfaceMeasurementDefinition,
    band: f64,
) -> bool {
    if (point[definition.normal_axis] - definition.target).abs() > band {
        return false;
    }
    (0..3).all(|axis| {
        let (low, high) = measurement_axis_range(definition, axis);
        point[axis] >= low.min(high) - band && point[axis] <= low.max(high) + band
    })
}

fn finite_quantile(mut values: Vec<f64>, probability: f64) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
    let probability = probability.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * probability).round() as usize;
    values[index.min(values.len() - 1)]
}

fn mean_or_nan(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        sum += value;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn nan_max(lhs: f64, rhs: f64) -> f64 {
    if !lhs.is_finite() {
        rhs
    } else if !rhs.is_finite() {
        lhs
    } else {
        lhs.max(rhs)
    }
}

fn pearson_correlation(lhs: &[f64], rhs: &[f64]) -> f64 {
    if lhs.len() != rhs.len() || lhs.len() < 2 {
        return f64::NAN;
    }
    let mean_lhs = mean_or_nan(lhs);
    let mean_rhs = mean_or_nan(rhs);
    if !mean_lhs.is_finite() || !mean_rhs.is_finite() {
        return f64::NAN;
    }
    let mut numerator = 0.0;
    let mut lhs_sumsq = 0.0;
    let mut rhs_sumsq = 0.0;
    for (lhs, rhs) in lhs.iter().copied().zip(rhs.iter().copied()) {
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        let dl = lhs - mean_lhs;
        let dr = rhs - mean_rhs;
        numerator += dl * dr;
        lhs_sumsq += dl * dl;
        rhs_sumsq += dr * dr;
    }
    let denominator = (lhs_sumsq * rhs_sumsq).sqrt();
    if denominator <= EPS {
        f64::NAN
    } else {
        numerator / denominator
    }
}

fn run_team13_synthetic_benchmark_geometry_fit(
    config: &Team13SyntheticBenchmarkGeometryConfig,
    _topology: &Complex,
    _coords: &MeshCoords,
    _operators: &Team13Operators,
    _layout: &DofLayout,
    pde_variance: f64,
    observation_std_tesla: f64,
    model: &ReducedVectorPotentialMagnetostatic3d,
    linear_mean: &[f64],
    truth_map: &[f64],
    exact_prior: &GaussianPriorSpec,
    prior_factor: &SparseCholeskyFactor,
    prior_variance_diagnostics: &Team13PriorVarianceDiagnostics,
    observations: &Team13SyntheticBenchmarkObservationBuild,
    initial_residual_norm: f64,
    truth_residual_norm: f64,
) -> Result<Team13SyntheticBenchmarkGeometryRunResult, String> {
    let model_adapter = FeecResidualAdapter::new(model);
    let posterior_problem = NonlinearLaplaceProblem {
        prior: exact_prior.clone(),
        residual_terms: vec![
            NonlinearResidualTerm::zero(
                "team13_synthetic_benchmark_posterior_residual",
                &model_adapter,
                GaussianNoiseModel::ScalarVariance(pde_variance),
            ),
            NonlinearResidualTerm {
                name: "team13_synthetic_benchmark_smooth_magnitude".to_string(),
                model: &observations.assimilated_model,
                observations: observations.assimilated_observations.clone(),
                noise: GaussianNoiseModel::ScalarVariance(
                    observation_std_tesla * observation_std_tesla,
                ),
            },
        ],
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: Vec::new(),
    };
    let posterior = solve_nonlinear_laplace(
        &posterior_problem,
        &GaussNewtonConfig {
            initial_guess: Some(linear_mean.to_vec()),
            max_iterations: config.max_iterations,
            step_tolerance: 1e-10,
            gradient_tolerance: 1e-9,
            max_line_search_steps: 40,
            linear_solve: config.linear_solve,
            step_regularization: config.step_regularization,
            variance: config.variance,
            ..GaussNewtonConfig::default()
        },
    )?;

    let posterior_residual_norm = l2_norm(&model.residual_and_jacobian(&posterior.map)?.residual);
    let posterior_predictions = observations.model.smooth_norm_values(&posterior.map)?;
    let published_nominal_predictions = observations
        .assimilated_model
        .smooth_norm_values(linear_mean)?;
    let published_posterior_predictions = observations
        .assimilated_model
        .smooth_norm_values(&posterior.map)?;
    let published_steel_benchmark_reports = published_steel_reports_from_predictions(
        &observations.assimilated_specs,
        &published_nominal_predictions,
        &published_posterior_predictions,
    )?;
    let observation_reports = synthetic_benchmark_observation_reports(
        &observations.specs,
        &observations.observations,
        &observations.initial_predictions,
        &posterior_predictions,
    )?;
    let group_summaries = synthetic_benchmark_group_summaries(&observation_reports);
    let initial_sensor_rmse = rmse_from_prediction_pairs(
        &observations.initial_predictions,
        &observations.observations,
    )?;
    let posterior_sensor_rmse =
        rmse_from_prediction_pairs(&posterior_predictions, &observations.observations)?;
    let posterior_sensor_relative_rmse =
        relative_rmse_from_prediction_pairs(&posterior_predictions, &observations.observations)?;
    let posterior_sensor_max_abs_residual =
        max_abs_residual_from_prediction_pairs(&posterior_predictions, &observations.observations)?;
    let observation_variances = grouped_norm_sensor_variance_reports(
        &observations.specs,
        &observations.model,
        prior_factor,
        &posterior,
    )?;
    let all_finite_variances = posterior
        .posterior_variance
        .iter()
        .all(|value| value.is_finite())
        && observation_variances.iter().all(|report| {
            report.prior_variance.is_finite() && report.posterior_variance.is_finite()
        });
    let nonnegative_variances = posterior
        .posterior_variance
        .iter()
        .all(|value| *value >= -1e-12)
        && observation_variances
            .iter()
            .all(|report| report.prior_variance >= -1e-12 && report.posterior_variance >= -1e-12);

    let steel_observation_count = observations
        .specs
        .iter()
        .filter(|spec| spec.group == Team13SyntheticBenchmarkObservationGroup::SteelAverage)
        .count();
    let air_observation_count = observations
        .specs
        .iter()
        .filter(|spec| spec.group == Team13SyntheticBenchmarkObservationGroup::AirPoint)
        .count();

    Ok(Team13SyntheticBenchmarkGeometryRunResult {
        pde_variance,
        observation_std_tesla,
        total_residual_rows: model.residual_dimension(),
        observation_count: observations.specs.len(),
        assimilated_observation_count: observations.assimilated_specs.len(),
        steel_observation_count,
        air_observation_count,
        initial_relative_error: relative_l2_distance(linear_mean, truth_map)?,
        posterior_relative_error: relative_l2_distance(&posterior.map, truth_map)?,
        initial_residual_norm,
        truth_residual_norm,
        posterior_residual_norm,
        initial_sensor_rmse,
        posterior_sensor_rmse,
        posterior_sensor_relative_rmse,
        posterior_sensor_max_abs_residual,
        sensor_rmse_improvement_ratio: safe_ratio(posterior_sensor_rmse, initial_sensor_rmse),
        posterior_converged: posterior.converged,
        all_finite_variances,
        nonnegative_variances,
        prior_variance_diagnostics: prior_variance_diagnostics.clone(),
        assembly: posterior.assembly,
        final_factorization: posterior.final_factorization,
        posterior_history: posterior.history,
        group_summaries,
        observation_reports,
        published_steel_benchmark_reports,
        observation_variances,
    })
}

pub fn run_team13_source_recovery_experiment(
    config: &Team13SourceRecoveryConfig,
) -> Result<Team13SourceRecoveryResult, String> {
    validate_source_recovery_config(config)?;
    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        format!(
            "failed to read mesh `{}`: {err}",
            config.mesh_path.display()
        )
    })?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "TEAM 13 requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let reluctivity = reluctivity_weight();
    let boundary = build_outer_boundary(&topology, &coords, config.domain_mode);
    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &reluctivity);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &reluctivity,
        ));
    let system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &state_mass_inverse,
    )?;
    let source_recovery_system = system_with_zero_residual_bias(&system);
    let unweighted_prior_system =
        build_reduced_hodge_laplace_1form_system(&topology, &metric, &boundary)?;
    if unweighted_prior_system.layout != source_recovery_system.layout {
        return Err(
            "TEAM13 unweighted prior layout does not match weighted PDE layout".to_string(),
        );
    }
    let nominal_source = assemble_unweighted_source(
        &topology,
        &metric,
        &coords,
        &team13_current_density(config.domain_mode, config.ampere_turns, None),
    );
    let nominal_a = solve_nominal_a(
        &topology,
        &metric,
        &coords,
        &reluctivity,
        &boundary,
        nominal_source,
    );
    let scalar_true_a = nominal_a.scale(config.source_alpha_true);
    let state_active_mask = active_dof_mask(&source_recovery_system.layout);
    let reduced_nominal_a = reduce_vector_with_layout(&source_recovery_system.layout, &nominal_a)?;
    let global_source_operator =
        build_nominal_state_source_operator(&source_recovery_system, &reduced_nominal_a)?;
    let fixed_source_bias = single_column_vector(&global_source_operator)?;
    let fixed_source_system =
        system_with_added_residual_bias(&source_recovery_system, &fixed_source_bias)?;
    let zero_state_mean = FeecVector::zeros(source_recovery_system.state_dimension());
    let mut field_prior_builds = Vec::with_capacity(config.field_priors.len());
    for kind in config.field_priors.iter().copied() {
        let zero_prior = scale_gaussian_prior_precision(
            build_team13_field_prior(
                kind,
                &unweighted_prior_system,
                &topology,
                &coords,
                &zero_state_mean,
            )?,
            config.field_prior_precision_scale,
        )?;
        let nominal_prior = scale_gaussian_prior_precision(
            build_team13_field_prior(
                kind,
                &unweighted_prior_system,
                &topology,
                &coords,
                &reduced_nominal_a,
            )?,
            config.field_prior_precision_scale,
        )?;
        field_prior_builds.push((kind, zero_prior, nominal_prior));
    }
    let (primary_prior_kind, field_state_prior, _joint_state_prior) = field_prior_builds
        .first()
        .cloned()
        .ok_or_else(|| "at least one TEAM13 field prior must be requested".to_string())?;
    let discrepancy_state_prior = match config.discrepancy_prior {
        Team13DiscrepancyPriorKind::Flat => {
            flat_state_prior(source_recovery_system.state_dimension())
        }
        Team13DiscrepancyPriorKind::WeightedWhittle => scale_gaussian_prior_precision(
            build_team13_field_prior(
                primary_prior_kind,
                &unweighted_prior_system,
                &topology,
                &coords,
                &zero_state_mean,
            )?,
            config.discrepancy_prior_precision_scale,
        )?,
    };
    let operators = build_team13_operators(&topology, &coords)?;
    let period_diagnostics =
        build_source_free_period_diagnostics(&topology, &coords, &operators, config.domain_mode)?;

    let nominal_overrides =
        load_surface_observation_overrides(config.nominal_observation_csv_path.as_deref())?;
    let perturbed_overrides = match config.perturbed_observation_csv_path.as_deref() {
        Some(path) => load_surface_observation_overrides(Some(path))?,
        None => None,
    };
    let sensor_variance = config.b_observation_std_tesla * config.b_observation_std_tesla;
    let mut nominal_measurements = build_linearized_b_measurements(
        &topology,
        &coords,
        &operators.b_cochain,
        &operators.b_components,
        &nominal_a,
        sensor_variance,
        config.measurement_mode,
        config.legacy_measurement_band,
        nominal_overrides.as_ref(),
    )?;
    if nominal_overrides.is_none() {
        set_measurement_observations_from_nominal_prediction(&mut nominal_measurements, 1.0)?;
    }
    let perturbed_measurements = if perturbed_overrides.is_some() {
        build_linearized_b_measurements(
            &topology,
            &coords,
            &operators.b_cochain,
            &operators.b_components,
            &nominal_a,
            sensor_variance,
            config.measurement_mode,
            config.legacy_measurement_band,
            perturbed_overrides.as_ref(),
        )?
    } else {
        let mut measurements = nominal_measurements.clone();
        scale_measurement_observations(&mut measurements, config.source_alpha_true)?;
        measurements
    };
    let source_mode_data = if config.run_eight_mode_recovery {
        let source_mode_operator = build_source_mode_operator(
            &topology,
            &metric,
            &coords,
            &galmats,
            &boundary,
            config.domain_mode,
            config.ampere_turns,
        )?;
        let source_mode_fields = build_source_mode_fields(
            &topology,
            &metric,
            &coords,
            &reluctivity,
            &boundary,
            config.domain_mode,
            config.ampere_turns,
        );
        let sensor_sensitivities = source_mode_sensor_sensitivities_from_fields(
            &source_mode_fields,
            &nominal_measurements,
        )?;
        let eight_mode_reference_a =
            weighted_sum_source_mode_fields(&source_mode_fields, &config.source_alpha_true_modes)?;
        let eight_mode_measurements = if perturbed_overrides.is_some() {
            perturbed_measurements.clone()
        } else {
            let mut measurements = nominal_measurements.clone();
            set_measurement_observations_from_mode_sensitivities(
                &mut measurements,
                &sensor_sensitivities,
                &config.source_alpha_true_modes,
            )?;
            measurements
        };
        Some((
            source_mode_operator,
            sensor_sensitivities,
            eight_mode_measurements,
            eight_mode_reference_a,
            source_mode_fields,
        ))
    } else {
        None
    };
    let nominal_report_overrides = measurement_observation_overrides(&nominal_measurements);
    let perturbed_report_overrides = measurement_observation_overrides(&perturbed_measurements);
    let sensor_areas = team13_sensor_areas()?;
    let derived_quantities = build_source_recovery_derived_quantities(
        &operators,
        &nominal_measurements,
        &period_diagnostics,
    )?;

    let mut stages = Vec::new();
    stages.push(run_team13_source_recovery_stage(
        "F0_prior_only",
        config,
        LinearPdeUqProblem {
            state_prior: field_state_prior.clone(),
            system: fixed_source_system.clone(),
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: derived_quantities.clone(),
            joint_derived_quantities: Vec::new(),
            pde_variance: None,
            pde_precision: None,
        },
        &topology,
        &coords,
        &operators,
        &nominal_a,
        "nominal_alpha_1",
        &nominal_a,
        &state_active_mask,
        &nominal_measurements,
        Some(&nominal_report_overrides),
        &sensor_areas,
        &period_diagnostics,
    )?);
    let (pde_variance, pde_precision) = team13_pde_residual_noise(config, &source_recovery_system)?;
    stages.push(run_team13_source_recovery_stage(
        "F1_prior_pde_residual",
        config,
        LinearPdeUqProblem {
            state_prior: field_state_prior.clone(),
            system: fixed_source_system.clone(),
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: derived_quantities.clone(),
            joint_derived_quantities: Vec::new(),
            pde_variance,
            pde_precision,
        },
        &topology,
        &coords,
        &operators,
        &nominal_a,
        "nominal_alpha_1",
        &nominal_a,
        &state_active_mask,
        &nominal_measurements,
        Some(&nominal_report_overrides),
        &sensor_areas,
        &period_diagnostics,
    )?);
    let (pde_variance, pde_precision) = team13_pde_residual_noise(config, &source_recovery_system)?;
    stages.push(run_team13_source_recovery_stage(
        "F2_prior_pde_nominal_measurements",
        config,
        LinearPdeUqProblem {
            state_prior: field_state_prior.clone(),
            system: fixed_source_system.clone(),
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: nominal_measurements
                .iter()
                .map(|measurement| measurement.spec.clone())
                .collect(),
            derived_quantities: derived_quantities.clone(),
            joint_derived_quantities: Vec::new(),
            pde_variance,
            pde_precision,
        },
        &topology,
        &coords,
        &operators,
        &nominal_a,
        "nominal_alpha_1",
        &nominal_a,
        &state_active_mask,
        &nominal_measurements,
        Some(&nominal_report_overrides),
        &sensor_areas,
        &period_diagnostics,
    )?);
    let (pde_variance, pde_precision) = team13_pde_residual_noise(config, &source_recovery_system)?;
    stages.push(run_team13_source_recovery_stage(
        "F2_mismatch_fixed_source",
        config,
        LinearPdeUqProblem {
            state_prior: field_state_prior.clone(),
            system: fixed_source_system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: perturbed_measurements
                .iter()
                .map(|measurement| measurement.spec.clone())
                .collect(),
            derived_quantities: derived_quantities.clone(),
            joint_derived_quantities: Vec::new(),
            pde_variance,
            pde_precision,
        },
        &topology,
        &coords,
        &operators,
        &nominal_a,
        "scalar_alpha_true",
        &scalar_true_a,
        &state_active_mask,
        &perturbed_measurements,
        Some(&perturbed_report_overrides),
        &sensor_areas,
        &period_diagnostics,
    )?);

    let source_prior_variance = config.source_prior_std * config.source_prior_std;
    let source_scaling_proxy = source_posterior_summary_from_sensor_scaling(
        &nominal_measurements,
        &perturbed_measurements,
        source_prior_variance,
        config,
    )?;
    let mut field_prior_comparisons = Vec::with_capacity(field_prior_builds.len());
    for (kind, _, nominal_prior) in &field_prior_builds {
        let (pde_variance, pde_precision) =
            team13_pde_residual_noise(config, &source_recovery_system)?;
        let stage_name = format!("S1_{}_direct_source_recovery", kind.stage_slug());
        let stage = run_team13_source_recovery_stage(
            &stage_name,
            config,
            LinearPdeUqProblem {
                state_prior: nominal_prior.clone(),
                system: source_recovery_system.clone(),
                uncertain_inputs: vec![LinearUncertainInputSpec {
                    name: SOURCE_ALPHA_INPUT_NAME.to_string(),
                    operator: global_source_operator.clone(),
                    prior: GaussianPriorSpec {
                        mean: vec![1.0],
                        precision: diagonal_precision(1, 1.0 / source_prior_variance),
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                }],
                joint_measurements: Vec::new(),
                physical_measurements: perturbed_measurements
                    .iter()
                    .map(|measurement| measurement.spec.clone())
                    .collect(),
                derived_quantities: derived_quantities.clone(),
                joint_derived_quantities: Vec::new(),
                pde_variance,
                pde_precision,
            },
            &topology,
            &coords,
            &operators,
            &nominal_a,
            "scalar_alpha_true",
            &scalar_true_a,
            &state_active_mask,
            &perturbed_measurements,
            Some(&perturbed_report_overrides),
            &sensor_areas,
            &period_diagnostics,
        )?;
        let source_posterior = source_posterior_summary_from_stage(
            &stage,
            SOURCE_ALPHA_INPUT_NAME,
            1.0,
            source_prior_variance,
            config.source_alpha_true,
        )?;
        field_prior_comparisons.push(field_prior_comparison_from_stage(
            *kind,
            &stage,
            source_posterior,
        ));
        stages.push(stage);
    }
    let baseline_source_posterior = field_prior_comparisons
        .first()
        .map(|comparison| comparison.source_posterior.clone())
        .ok_or_else(|| "at least one TEAM13 field prior comparison is required".to_string())?;

    let scalar_source_fields = vec![nominal_a.clone()];
    let scalar_fluctuation_measurements = build_fluctuation_joint_measurements(
        &perturbed_measurements,
        SOURCE_ALPHA_INPUT_NAME,
        &scalar_sensor_sensitivities(&nominal_measurements),
    )?;
    let (pde_variance, pde_precision) = team13_pde_residual_noise(config, &source_recovery_system)?;
    let fluctuation_stage = run_team13_fluctuation_source_recovery_stage(
        "S2_fluctuation_source_recovery",
        config,
        LinearPdeUqProblem {
            state_prior: discrepancy_state_prior.clone(),
            system: source_recovery_system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: SOURCE_ALPHA_INPUT_NAME.to_string(),
                operator: zero_source_operator(source_recovery_system.residual_dimension(), 1),
                prior: GaussianPriorSpec {
                    mean: vec![1.0],
                    precision: diagonal_precision(1, 1.0 / source_prior_variance),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            joint_measurements: scalar_fluctuation_measurements,
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: build_fluctuation_joint_derived_quantities(
                &operators,
                &nominal_measurements,
                &period_diagnostics,
                SOURCE_ALPHA_INPUT_NAME,
                &scalar_source_fields,
            )?,
            pde_variance,
            pde_precision,
        },
        &topology,
        &coords,
        &operators,
        &nominal_a,
        "scalar_alpha_true",
        &scalar_true_a,
        SOURCE_ALPHA_INPUT_NAME,
        &scalar_source_fields,
        &state_active_mask,
        &perturbed_measurements,
        Some(&perturbed_report_overrides),
        &sensor_areas,
        &period_diagnostics,
    )?;
    let fluctuation_source_posterior = source_posterior_summary_from_stage(
        &fluctuation_stage,
        SOURCE_ALPHA_INPUT_NAME,
        1.0,
        source_prior_variance,
        config.source_alpha_true,
    )?;
    stages.push(fluctuation_stage);

    let eight_mode = if let Some((
        source_mode_operator,
        source_mode_sensor_sensitivities,
        eight_mode_measurements,
        eight_mode_reference_a,
        source_mode_fields,
    )) = source_mode_data
    {
        let eight_mode_report_overrides =
            measurement_observation_overrides(&eight_mode_measurements);
        let eight_mode_source_prior_variance = config.source_prior_std * config.source_prior_std;
        let (pde_variance, pde_precision) =
            team13_pde_residual_noise(config, &source_recovery_system)?;
        let mut stage = run_team13_source_recovery_stage(
            "M1_joint_eight_mode_recovery",
            config,
            LinearPdeUqProblem {
                state_prior: field_prior_builds
                    .first()
                    .map(|(_, _, prior)| prior.clone())
                    .ok_or_else(|| {
                        "at least one TEAM13 field prior must be requested".to_string()
                    })?,
                system: source_recovery_system.clone(),
                uncertain_inputs: vec![LinearUncertainInputSpec {
                    name: SOURCE_MODE_INPUT_NAME.to_string(),
                    operator: source_mode_operator,
                    prior: GaussianPriorSpec {
                        mean: vec![1.0; COIL_MODE_COUNT],
                        precision: diagonal_precision(
                            COIL_MODE_COUNT,
                            1.0 / eight_mode_source_prior_variance,
                        ),
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                }],
                joint_measurements: Vec::new(),
                physical_measurements: eight_mode_measurements
                    .iter()
                    .map(|measurement| measurement.spec.clone())
                    .collect(),
                derived_quantities: build_source_recovery_derived_quantities(
                    &operators,
                    &nominal_measurements,
                    &period_diagnostics,
                )?,
                joint_derived_quantities: Vec::new(),
                pde_variance,
                pde_precision,
            },
            &topology,
            &coords,
            &operators,
            &nominal_a,
            "eight_mode_true",
            &eight_mode_reference_a,
            &state_active_mask,
            &eight_mode_measurements,
            Some(&eight_mode_report_overrides),
            &sensor_areas,
            &period_diagnostics,
        )?;
        let source_modes = source_mode_posterior_summary_from_sensor_sensitivities(
            &source_mode_sensor_sensitivities,
            &eight_mode_measurements,
            eight_mode_source_prior_variance,
            config.b_observation_std_tesla * config.b_observation_std_tesla,
            &config.source_alpha_true_modes,
        )?;
        let mode_means = source_modes
            .iter()
            .map(|mode| mode.posterior_mean)
            .collect::<Vec<_>>();
        let eight_mode_sensor_rmse = source_mode_sensor_rmse(
            &source_mode_sensor_sensitivities,
            &mode_means,
            &eight_mode_measurements,
        )?;
        stage.summary.sensor_rmse = eight_mode_sensor_rmse;
        let fluctuation_joint_measurements = build_fluctuation_joint_measurements(
            &eight_mode_measurements,
            SOURCE_MODE_INPUT_NAME,
            &source_mode_sensor_sensitivities,
        )?;
        let eight_mode_state_dimension = source_recovery_system.state_dimension();
        let eight_mode_residual_dimension = source_recovery_system.residual_dimension();
        let (pde_variance, pde_precision) =
            team13_pde_residual_noise(config, &source_recovery_system)?;
        let fluctuation_stage = run_team13_fluctuation_source_recovery_stage(
            "M2_fluctuation_eight_mode_recovery",
            config,
            LinearPdeUqProblem {
                state_prior: match config.discrepancy_prior {
                    Team13DiscrepancyPriorKind::Flat => {
                        flat_state_prior(eight_mode_state_dimension)
                    }
                    Team13DiscrepancyPriorKind::WeightedWhittle => discrepancy_state_prior.clone(),
                },
                system: source_recovery_system,
                uncertain_inputs: vec![LinearUncertainInputSpec {
                    name: SOURCE_MODE_INPUT_NAME.to_string(),
                    operator: zero_source_operator(eight_mode_residual_dimension, COIL_MODE_COUNT),
                    prior: GaussianPriorSpec {
                        mean: vec![1.0; COIL_MODE_COUNT],
                        precision: diagonal_precision(
                            COIL_MODE_COUNT,
                            1.0 / eight_mode_source_prior_variance,
                        ),
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                }],
                joint_measurements: fluctuation_joint_measurements,
                physical_measurements: Vec::new(),
                derived_quantities: Vec::new(),
                joint_derived_quantities: build_fluctuation_joint_derived_quantities(
                    &operators,
                    &nominal_measurements,
                    &period_diagnostics,
                    SOURCE_MODE_INPUT_NAME,
                    &source_mode_fields,
                )?,
                pde_variance,
                pde_precision,
            },
            &topology,
            &coords,
            &operators,
            &nominal_a,
            "eight_mode_true",
            &eight_mode_reference_a,
            SOURCE_MODE_INPUT_NAME,
            &source_mode_fields,
            &state_active_mask,
            &eight_mode_measurements,
            Some(&eight_mode_report_overrides),
            &sensor_areas,
            &period_diagnostics,
        )?;
        let fluctuation_source_modes = source_mode_posterior_summary_from_stage(
            &fluctuation_stage,
            SOURCE_MODE_INPUT_NAME,
            eight_mode_source_prior_variance,
            &config.source_alpha_true_modes,
        )?;
        let observability = source_mode_observability_summary(
            &source_mode_sensor_sensitivities,
            &fluctuation_source_modes,
        )?;
        let fixed_source_sensor_rmse = stages
            .iter()
            .find(|stage| stage.summary.name == "F2_mismatch_fixed_source")
            .map(|stage| stage.summary.sensor_rmse)
            .unwrap_or(f64::NAN);
        let fixed_source_b_vector_relative_l2_error = stages
            .iter()
            .find(|stage| stage.summary.name == "F2_mismatch_fixed_source")
            .map(|stage| stage.summary.b_vector_relative_l2_error)
            .unwrap_or(f64::NAN);
        let decision = replacement_decision(
            &fluctuation_source_modes,
            fixed_source_sensor_rmse,
            fluctuation_stage.summary.sensor_rmse,
            fixed_source_b_vector_relative_l2_error,
            fluctuation_stage.summary.b_vector_relative_l2_error,
        );
        Some(Team13EightModeRecoveryResult {
            stage,
            fluctuation_stage: Some(fluctuation_stage),
            source_modes,
            fluctuation_source_modes,
            observability,
            decision,
        })
    } else {
        None
    };

    let result = Team13SourceRecoveryResult {
        stages,
        field_prior_comparisons,
        source_posterior: baseline_source_posterior.clone(),
        source_scaling_proxy,
        baseline_source_posterior,
        fluctuation_source_posterior,
        eight_mode,
    };

    if let Some(output_dir) = &config.output_dir {
        write_team13_source_recovery_outputs(output_dir, &topology, &coords, &result)?;
    }

    Ok(result)
}

pub fn reluctivity_at(point: CoordRef<'_>) -> f64 {
    if is_iron_point(point) {
        NU_IRON
    } else {
        NU_AIR
    }
}

pub fn is_iron_point(point: CoordRef<'_>) -> bool {
    is_vertical_sheet(point) || is_left_c_sheet(point) || is_right_c_sheet(point)
}

pub fn coil_region_at(point: CoordRef<'_>, mode: Team13DomainMode) -> Option<Team13CoilRegion> {
    Team13CoilRegion::all()
        .into_iter()
        .find(|region| point_in_coil_region(point, mode, *region))
}

pub fn team13_current_vector(
    point: CoordRef<'_>,
    mode: Team13DomainMode,
    ampere_turns: f64,
    region_filter: Option<Team13CoilRegion>,
) -> [f64; 3] {
    let Some(region) = coil_region_at(point, mode) else {
        return [0.0, 0.0, 0.0];
    };
    if region_filter.is_some_and(|filter| filter != region) {
        return [0.0, 0.0, 0.0];
    }
    let direction = coil_direction(point, region);
    let magnitude = ampere_turns / (2500.0 * 1e-6);
    [
        magnitude * direction[0],
        magnitude * direction[1],
        magnitude * direction[2],
    ]
}

pub fn trace_variance_ratio(
    prior_components: &[FeecVector],
    posterior_components: &[FeecVector],
) -> Result<FeecVector, String> {
    let prior_trace = trace_components(prior_components)?;
    let posterior_trace = trace_components(posterior_components)?;
    Ok(variance_ratio(&posterior_trace, &prior_trace))
}

fn validate_config(config: &Team13LinearConfig) -> Result<(), String> {
    if !config.ampere_turns.is_finite() {
        return Err("ampere_turns must be finite".to_string());
    }
    if !config.coil_relative_std.is_finite() || config.coil_relative_std <= 0.0 {
        return Err("coil_relative_std must be finite and positive".to_string());
    }
    if !config.pde_variance.is_finite() || config.pde_variance <= 0.0 {
        return Err("pde_variance must be finite and positive".to_string());
    }
    if !config.b_observation_std_tesla.is_finite() || config.b_observation_std_tesla <= 0.0 {
        return Err("b_observation_std_tesla must be finite and positive".to_string());
    }
    if !config.legacy_measurement_band.is_finite() || config.legacy_measurement_band <= 0.0 {
        return Err("legacy_measurement_band must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_source_recovery_config(config: &Team13SourceRecoveryConfig) -> Result<(), String> {
    if !config.ampere_turns.is_finite() || config.ampere_turns <= 0.0 {
        return Err("ampere_turns must be finite and positive".to_string());
    }
    if !config.source_alpha_true.is_finite() || config.source_alpha_true <= 0.0 {
        return Err("source_alpha_true must be finite and positive".to_string());
    }
    for (index, value) in config.source_alpha_true_modes.iter().enumerate() {
        if !value.is_finite() || *value <= 0.0 {
            return Err(format!(
                "source_alpha_true_modes[{index}] must be finite and positive"
            ));
        }
    }
    if !config.source_prior_std.is_finite() || config.source_prior_std <= 0.0 {
        return Err("source_prior_std must be finite and positive".to_string());
    }
    if !config.pde_variance.is_finite() || config.pde_variance <= 0.0 {
        return Err("pde_variance must be finite and positive".to_string());
    }
    if !config.discrepancy_prior_precision_scale.is_finite()
        || config.discrepancy_prior_precision_scale <= 0.0
    {
        return Err("discrepancy_prior_precision_scale must be finite and positive".to_string());
    }
    if !config.field_prior_precision_scale.is_finite() || config.field_prior_precision_scale <= 0.0
    {
        return Err("field_prior_precision_scale must be finite and positive".to_string());
    }
    if config.field_priors.is_empty() {
        return Err("at least one TEAM13 field prior must be requested".to_string());
    }
    for (index, prior) in config.field_priors.iter().copied().enumerate() {
        if config.field_priors[..index].contains(&prior) {
            return Err(format!(
                "TEAM13 field prior `{}` was requested more than once",
                prior.as_str()
            ));
        }
    }
    if !config.b_observation_std_tesla.is_finite() || config.b_observation_std_tesla <= 0.0 {
        return Err("b_observation_std_tesla must be finite and positive".to_string());
    }
    if !config.legacy_measurement_band.is_finite() || config.legacy_measurement_band <= 0.0 {
        return Err("legacy_measurement_band must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_nonlinear_config(config: &Team13NonlinearConfig) -> Result<(), String> {
    if !config.ampere_turns.is_finite() || config.ampere_turns <= 0.0 {
        return Err("ampere_turns must be finite and positive".to_string());
    }
    if !config.beta_iron.is_finite() || config.beta_iron < 0.0 {
        return Err("beta_iron must be finite and nonnegative".to_string());
    }
    if !config.b_scale_tesla.is_finite() || config.b_scale_tesla <= 0.0 {
        return Err("b_scale_tesla must be finite and positive".to_string());
    }
    if !config.pde_variance.is_finite() || config.pde_variance <= 0.0 {
        return Err("pde_variance must be finite and positive".to_string());
    }
    if !config.field_prior_precision_scale.is_finite() || config.field_prior_precision_scale <= 0.0
    {
        return Err("field_prior_precision_scale must be finite and positive".to_string());
    }
    if let Some(kappa) = config.prior_kappa {
        if !kappa.is_finite() || kappa < 0.0 {
            return Err("prior_kappa must be finite and nonnegative".to_string());
        }
    }
    if !config.prior_tau.is_finite() || config.prior_tau <= 0.0 {
        return Err("prior_tau must be finite and positive".to_string());
    }
    if config.max_iterations == 0 {
        return Err("max_iterations must be at least one".to_string());
    }
    if !config.b_observation_std_tesla.is_finite() || config.b_observation_std_tesla <= 0.0 {
        return Err("b_observation_std_tesla must be finite and positive".to_string());
    }
    if !config.legacy_measurement_band.is_finite() || config.legacy_measurement_band <= 0.0 {
        return Err("legacy_measurement_band must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_synthetic_nonlinear_baseline_config(
    config: &Team13SyntheticNonlinearBaselineConfig,
) -> Result<(), String> {
    if !config.ampere_turns.is_finite() || config.ampere_turns <= 0.0 {
        return Err("ampere_turns must be finite and positive".to_string());
    }
    if !config.beta_iron.is_finite() || config.beta_iron <= 0.0 {
        return Err("beta_iron must be finite and positive".to_string());
    }
    if !config.b_scale_tesla.is_finite() || config.b_scale_tesla <= 0.0 {
        return Err("b_scale_tesla must be finite and positive".to_string());
    }
    if !config.truth_prior_precision.is_finite() || config.truth_prior_precision < 0.0 {
        return Err("truth_prior_precision must be finite and nonnegative".to_string());
    }
    if !config.truth_pde_variance.is_finite() || config.truth_pde_variance <= 0.0 {
        return Err("truth_pde_variance must be finite and positive".to_string());
    }
    if !config.pde_variance.is_finite() || config.pde_variance <= 0.0 {
        return Err("pde_variance must be finite and positive".to_string());
    }
    if !config.observation_std_tesla.is_finite() || config.observation_std_tesla <= 0.0 {
        return Err("observation_std_tesla must be finite and positive".to_string());
    }
    if !config.magnitude_smoothing_tesla.is_finite() || config.magnitude_smoothing_tesla <= 0.0 {
        return Err("magnitude_smoothing_tesla must be finite and positive".to_string());
    }
    if !config.prior_kappa.is_finite() || config.prior_kappa <= 0.0 {
        return Err("prior_kappa must be finite and positive".to_string());
    }
    if !config.prior_tau.is_finite() || config.prior_tau <= 0.0 {
        return Err("prior_tau must be finite and positive".to_string());
    }
    if !config.prior_diagonal_shift.is_finite() || config.prior_diagonal_shift < 0.0 {
        return Err("prior_diagonal_shift must be finite and nonnegative".to_string());
    }
    if config.truth_max_iterations == 0 || config.max_iterations == 0 {
        return Err("iteration counts must be at least one".to_string());
    }
    if config.observation_models.is_empty() {
        return Err("at least one synthetic observation model must be requested".to_string());
    }
    if !matches!(
        config.variance.mode,
        LinearPdeVarianceMode::Exact | LinearPdeVarianceMode::ExactSolves
    ) {
        return Err("synthetic nonlinear baseline variance mode must be exact".to_string());
    }
    Ok(())
}

fn validate_synthetic_benchmark_geometry_config(
    config: &Team13SyntheticBenchmarkGeometryConfig,
) -> Result<(), String> {
    if !config.ampere_turns.is_finite() || config.ampere_turns <= 0.0 {
        return Err("ampere_turns must be finite and positive".to_string());
    }
    if !config.beta_iron.is_finite() || config.beta_iron <= 0.0 {
        return Err("beta_iron must be finite and positive".to_string());
    }
    if !config.b_scale_tesla.is_finite() || config.b_scale_tesla <= 0.0 {
        return Err("b_scale_tesla must be finite and positive".to_string());
    }
    if !config.truth_prior_precision.is_finite() || config.truth_prior_precision < 0.0 {
        return Err("truth_prior_precision must be finite and nonnegative".to_string());
    }
    if !config.truth_pde_variance.is_finite() || config.truth_pde_variance <= 0.0 {
        return Err("truth_pde_variance must be finite and positive".to_string());
    }
    if !config.pde_variance.is_finite() || config.pde_variance <= 0.0 {
        return Err("pde_variance must be finite and positive".to_string());
    }
    if !config.observation_std_tesla.is_finite() || config.observation_std_tesla <= 0.0 {
        return Err("observation_std_tesla must be finite and positive".to_string());
    }
    if !config.magnitude_smoothing_tesla.is_finite() || config.magnitude_smoothing_tesla <= 0.0 {
        return Err("magnitude_smoothing_tesla must be finite and positive".to_string());
    }
    if !config.prior_kappa.is_finite() || config.prior_kappa <= 0.0 {
        return Err("prior_kappa must be finite and positive".to_string());
    }
    if !config.prior_tau.is_finite() || config.prior_tau <= 0.0 {
        return Err("prior_tau must be finite and positive".to_string());
    }
    if !config.prior_diagonal_shift.is_finite() || config.prior_diagonal_shift < 0.0 {
        return Err("prior_diagonal_shift must be finite and nonnegative".to_string());
    }
    if config.truth_max_iterations == 0 || config.max_iterations == 0 {
        return Err("iteration counts must be at least one".to_string());
    }
    if !config
        .sweep_pde_variances
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err("sweep_pde_variances must be finite and positive".to_string());
    }
    if !config
        .sweep_observation_std_tesla
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err("sweep_observation_std_tesla must be finite and positive".to_string());
    }
    if !config
        .source_scale_diagnostic_values
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err("source_scale_diagnostic_values must be finite and positive".to_string());
    }
    if !matches!(
        config.variance.mode,
        LinearPdeVarianceMode::Exact | LinearPdeVarianceMode::ExactSolves
    ) {
        return Err("synthetic benchmark-geometry variance mode must be exact".to_string());
    }
    Ok(())
}

fn validate_team13_map_parity_config(config: &Team13MapParityConfig) -> Result<(), String> {
    if !config.ampere_turns.is_finite() || config.ampere_turns <= 0.0 {
        return Err("ampere_turns must be finite and positive".to_string());
    }
    if !config.beta_iron.is_finite() || config.beta_iron < 0.0 {
        return Err("beta_iron must be finite and nonnegative".to_string());
    }
    if !config.b_scale_tesla.is_finite() || config.b_scale_tesla <= 0.0 {
        return Err("b_scale_tesla must be finite and positive".to_string());
    }
    if !config.pde_variance.is_finite() || config.pde_variance <= 0.0 {
        return Err("pde_variance must be finite and positive".to_string());
    }
    if !config.observation_std_tesla.is_finite() || config.observation_std_tesla <= 0.0 {
        return Err("observation_std_tesla must be finite and positive".to_string());
    }
    if !config.magnitude_smoothing_tesla.is_finite() || config.magnitude_smoothing_tesla <= 0.0 {
        return Err("magnitude_smoothing_tesla must be finite and positive".to_string());
    }
    if !config.prior_kappa.is_finite() || config.prior_kappa <= 0.0 {
        return Err("prior_kappa must be finite and positive".to_string());
    }
    if !config.prior_tau.is_finite() || config.prior_tau <= 0.0 {
        return Err("prior_tau must be finite and positive".to_string());
    }
    if !config.prior_diagonal_shift.is_finite() || config.prior_diagonal_shift < 0.0 {
        return Err("prior_diagonal_shift must be finite and nonnegative".to_string());
    }
    if config.truth_max_iterations == 0 || config.max_iterations == 0 {
        return Err("iteration counts must be at least one".to_string());
    }
    if !config
        .sweep_pde_variances
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err("sweep_pde_variances must be finite and positive".to_string());
    }
    if !config
        .sweep_observation_std_tesla
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err("sweep_observation_std_tesla must be finite and positive".to_string());
    }
    Ok(())
}

#[derive(Clone)]
struct Team13MaterialPair {
    nonlinear: Arc<dyn SpatialReluctivity>,
    linear: Arc<dyn SpatialReluctivity>,
}

fn build_team13_nonlinear_material(
    config: &Team13NonlinearConfig,
) -> Result<Team13MaterialPair, String> {
    build_team13_material_from_kind(config.material_kind, config.beta_iron, config.b_scale_tesla)
}

fn build_team13_synthetic_benchmark_material(
    config: &Team13SyntheticBenchmarkGeometryConfig,
) -> Result<Team13MaterialPair, String> {
    build_team13_material_from_kind(config.material_kind, config.beta_iron, config.b_scale_tesla)
}

fn build_team13_material_from_kind(
    kind: Team13NonlinearMaterialKind,
    beta_iron: f64,
    b_scale_tesla: f64,
) -> Result<Team13MaterialPair, String> {
    build_team13_material_from_kind_with_log_scale(kind, beta_iron, b_scale_tesla, 0.0)
}

fn build_team13_material_from_kind_with_log_scale(
    kind: Team13NonlinearMaterialKind,
    beta_iron: f64,
    b_scale_tesla: f64,
    log_iron_nu_scale: f64,
) -> Result<Team13MaterialPair, String> {
    match kind {
        Team13NonlinearMaterialKind::NgsolveTabulatedLinear => {
            let nonlinear =
                Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES)?
                    .with_log_iron_nu_scale(log_iron_nu_scale)?;
            Ok(Team13MaterialPair {
                linear: Arc::new(nonlinear.linear_reference_law()),
                nonlinear: Arc::new(nonlinear),
            })
        }
        Team13NonlinearMaterialKind::SmoothQuadratic => {
            let nonlinear =
                Team13SmoothIronReluctivityLaw::new(NU_AIR, NU_IRON, beta_iron, b_scale_tesla)?
                    .with_log_iron_nu_scale(log_iron_nu_scale)?;
            Ok(Team13MaterialPair {
                linear: Arc::new(nonlinear.linear_reference_law()),
                nonlinear: Arc::new(nonlinear),
            })
        }
    }
}

fn build_team13_tabulated_material_with_log_h_shape(
    anchors_tesla: [f64; 3],
    theta: [f64; 3],
) -> Result<Team13TabulatedReluctivityLaw, String> {
    Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES)?
        .with_log_h_shape(anchors_tesla, theta)
}

fn build_team13_nonlinear_prior(
    config: &Team13NonlinearConfig,
    unweighted_system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    topology: &Complex,
    coords: &MeshCoords,
    linear_mean: &FeecVector,
) -> Result<Team13NonlinearPriorBuild, String> {
    let kappa = config.prior_kappa.unwrap_or(1.0);
    let prior = build_team13_field_prior_with_matern_params(
        config.field_prior_kind,
        unweighted_system,
        topology,
        coords,
        linear_mean,
        kappa,
        config.prior_tau,
    )?;
    let prior = scale_gaussian_prior_precision(prior, config.field_prior_precision_scale)?;
    Ok(Team13NonlinearPriorBuild {
        spec: prior,
        kind: config.field_prior_kind,
        precision_scale: config.field_prior_precision_scale,
        kappa,
        tau: config.prior_tau,
        kappa_fallback_used: false,
    })
}

fn selected_sensor_derived_quantities(
    measurements: &[Team13LinearizedMeasurement],
    layout: &DofLayout,
    count: usize,
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    measurements
        .iter()
        .take(count)
        .map(|measurement| {
            Ok(LinearPdeDerivedQuantitySpec {
                name: sensor_derived_name(&measurement.spec.name),
                operator: triplet_to_sparse_row_operator(&restrict_triplet_columns_to_layout(
                    &measurement.spec.operator,
                    layout,
                )?)?,
            })
        })
        .collect()
}

fn restrict_team13_measurements_to_layout(
    measurements: &[Team13LinearizedMeasurement],
    layout: &DofLayout,
) -> Result<Vec<LinearGaussianMeasurementSpec>, String> {
    measurements
        .iter()
        .map(|measurement| restrict_team13_measurement_to_layout(measurement, layout))
        .collect()
}

fn restrict_team13_measurement_to_layout(
    measurement: &Team13LinearizedMeasurement,
    layout: &DofLayout,
) -> Result<LinearGaussianMeasurementSpec, String> {
    let full_operator = core_triplet_to_feec_csr(&measurement.spec.operator);
    let full_bias = FeecVector::from_vec(measurement.spec.bias.clone());
    let (operator, bias) = restrict_columns_and_fold_fixed(&full_operator, &full_bias, layout)?;
    Ok(LinearGaussianMeasurementSpec {
        name: measurement.spec.name.clone(),
        operator: csr_to_triplet(&operator),
        observations: measurement.spec.observations.clone(),
        bias: bias.iter().copied().collect(),
        variance: measurement.spec.variance,
    })
}

fn smooth_abs_scalar(value: f64, smoothing: f64) -> f64 {
    (value * value + smoothing * smoothing).sqrt()
}

fn smooth_abs_model_and_observations_from_measurements(
    measurements: &[LinearGaussianMeasurementSpec],
    smoothing: f64,
) -> Result<(SmoothAbsLinearResidualModel, Vec<f64>), String> {
    let Some(first) = measurements.first() else {
        return Err("at least one measurement is required to build smooth-abs model".to_string());
    };
    let state_dimension = first.operator.ncols();
    let mut row_offset = 0;
    let mut triplets = Vec::new();
    let mut bias = Vec::new();
    let mut observations = Vec::new();

    for measurement in measurements {
        measurement.validate(state_dimension)?;
        if measurement.operator.ncols() != state_dimension {
            return Err(format!(
                "measurement `{}` has column count {}, expected {}",
                measurement.name,
                measurement.operator.ncols(),
                state_dimension
            ));
        }
        for (row, col, value) in measurement.operator.triplet_iter() {
            triplets.push(SparseTriplet {
                row: row_offset + row,
                col,
                value,
            });
        }
        bias.extend(measurement.bias.iter().copied());
        observations.extend(
            measurement
                .observations
                .iter()
                .copied()
                .map(|value| smooth_abs_scalar(value, smoothing)),
        );
        row_offset += measurement.operator.nrows();
    }

    let model = SmoothAbsLinearResidualModel::new(
        SparseTripletMatrix::from_triplets(row_offset, state_dimension, triplets),
        bias,
        smoothing,
    )?;
    Ok((model, observations))
}

fn signed_linear_proxy_measurements(
    reduced_component_measurements: &[LinearGaussianMeasurementSpec],
    full_component_measurements: &[Team13LinearizedMeasurement],
) -> Result<Vec<LinearGaussianMeasurementSpec>, String> {
    if reduced_component_measurements.len() != full_component_measurements.len() {
        return Err(format!(
            "reduced measurement count {} must match full measurement count {}",
            reduced_component_measurements.len(),
            full_component_measurements.len()
        ));
    }
    reduced_component_measurements
        .iter()
        .zip(full_component_measurements.iter())
        .map(|(measurement, full)| {
            let sign = nonzero_sign(&measurement.name, full.nominal_prediction)?;
            Ok(LinearGaussianMeasurementSpec {
                name: format!("{}_signed_linear_proxy", measurement.name),
                operator: scale_triplet_matrix(&measurement.operator, sign),
                observations: measurement
                    .observations
                    .iter()
                    .map(|value| sign * *value)
                    .collect(),
                bias: measurement.bias.iter().map(|value| sign * *value).collect(),
                variance: measurement.variance,
            })
        })
        .collect()
}

fn evaluate_smooth_abs_sensor_reports(
    measurements: &[Team13LinearizedMeasurement],
    posterior_mean: &FeecVector,
    smoothing: f64,
) -> Result<Vec<Team13SensorReport>, String> {
    measurements
        .iter()
        .map(|measurement| {
            let operator = triplet_to_sparse_row_operator(&measurement.spec.operator)?;
            let raw_prediction = operator
                .apply(&GmrfVector::from_vec(
                    posterior_mean.iter().copied().collect(),
                ))
                .map_err(|err| err.to_string())?[0]
                + measurement.spec.bias[0];
            let observed = smooth_abs_scalar(measurement.spec.observations[0], smoothing);
            let posterior_prediction = smooth_abs_scalar(raw_prediction, smoothing);
            Ok(Team13SensorReport {
                name: measurement.spec.name.clone(),
                observed,
                nominal_prediction: smooth_abs_scalar(measurement.nominal_prediction, smoothing),
                posterior_prediction,
                residual: posterior_prediction - observed,
                linearization_direction: measurement.linearization_direction,
            })
        })
        .collect()
}

fn smooth_abs_sensor_variance_reports(
    measurements: &[Team13LinearizedMeasurement],
    reduced_component_measurements: &[LinearGaussianMeasurementSpec],
    smoothing: f64,
    prior_factor: &SparseCholeskyFactor,
    posterior: &NonlinearLaplaceResult,
) -> Result<Vec<Team13NonlinearSensorVarianceReport>, String> {
    let (model, _) = smooth_abs_model_and_observations_from_measurements(
        reduced_component_measurements,
        smoothing,
    )?;
    let linearized = model.residual_and_jacobian(&posterior.map)?;
    if linearized.jacobian.nrows() != measurements.len() {
        return Err(format!(
            "linearized smooth-abs sensor row count {} must match measurement count {}",
            linearized.jacobian.nrows(),
            measurements.len()
        ));
    }
    let operator = triplet_to_sparse_row_operator(&linearized.jacobian)?;
    let prior_variance = exact_solve_transformed_diag(prior_factor, &operator)
        .map_err(|err| err.to_string())?
        .values;
    let posterior_factor = posterior
        .posterior_gmrf
        .precision_factor()
        .ok_or_else(|| "posterior precision factor is missing".to_string())?;
    let posterior_variance = exact_solve_transformed_diag(posterior_factor, &operator)
        .map_err(|err| err.to_string())?
        .values;
    if prior_variance.len() != measurements.len() || posterior_variance.len() != measurements.len()
    {
        return Err("smooth-abs transformed variance count did not match measurements".to_string());
    }
    Ok(measurements
        .iter()
        .enumerate()
        .map(|(index, measurement)| Team13NonlinearSensorVarianceReport {
            name: measurement.spec.name.clone(),
            prior_variance: prior_variance[index],
            posterior_variance: posterior_variance[index],
        })
        .collect())
}

fn grouped_norm_sensor_variance_reports(
    specs: &[Team13SyntheticBenchmarkObservationSpec],
    model: &SmoothGroupedNormLinearResidualModel,
    prior_factor: &SparseCholeskyFactor,
    posterior: &NonlinearLaplaceResult,
) -> Result<Vec<Team13NonlinearSensorVarianceReport>, String> {
    let linearized = model.residual_and_jacobian(&posterior.map)?;
    if linearized.jacobian.nrows() != specs.len() {
        return Err(format!(
            "linearized grouped-norm sensor row count {} must match observation count {}",
            linearized.jacobian.nrows(),
            specs.len()
        ));
    }
    let operator = triplet_to_sparse_row_operator(&linearized.jacobian)?;
    let prior_variance = exact_solve_transformed_diag(prior_factor, &operator)
        .map_err(|err| err.to_string())?
        .values;
    let posterior_factor = posterior
        .posterior_gmrf
        .precision_factor()
        .ok_or_else(|| "posterior precision factor is missing".to_string())?;
    let posterior_variance = exact_solve_transformed_diag(posterior_factor, &operator)
        .map_err(|err| err.to_string())?
        .values;
    if prior_variance.len() != specs.len() || posterior_variance.len() != specs.len() {
        return Err(
            "grouped-norm transformed variance count did not match observations".to_string(),
        );
    }
    Ok(specs
        .iter()
        .enumerate()
        .map(|(index, spec)| Team13NonlinearSensorVarianceReport {
            name: spec.name.clone(),
            prior_variance: prior_variance[index],
            posterior_variance: posterior_variance[index],
        })
        .collect())
}

fn prior_variance_diagnostics(
    factor: &SparseCholeskyFactor,
) -> Result<Team13PriorVarianceDiagnostics, String> {
    let variance = exact_solve_diag(factor)
        .map_err(|err| format!("exact prior variance solve failed: {err}"))?
        .values;
    if variance.is_empty() {
        return Err("prior variance diagnostics require a nonempty vector".to_string());
    }
    let min_variance = variance.iter().copied().fold(f64::INFINITY, f64::min);
    let max_variance = variance.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let all_finite = variance.iter().all(|value| value.is_finite());
    let nonnegative = variance.iter().all(|value| *value >= -1e-12);
    Ok(Team13PriorVarianceDiagnostics {
        dimension: variance.len(),
        min_variance,
        max_variance,
        max_to_min_variance_ratio: safe_ratio(max_variance, min_variance),
        all_finite,
        nonnegative,
        factor_nnz: factor.nnz(),
    })
}

fn synthetic_benchmark_observation_reports(
    specs: &[Team13SyntheticBenchmarkObservationSpec],
    observations: &[f64],
    initial_predictions: &[f64],
    posterior_predictions: &[f64],
) -> Result<Vec<Team13SyntheticBenchmarkObservationReport>, String> {
    if specs.len() != observations.len()
        || specs.len() != initial_predictions.len()
        || specs.len() != posterior_predictions.len()
    {
        return Err("synthetic benchmark report vectors must have matching lengths".to_string());
    }
    Ok(specs
        .iter()
        .zip(observations.iter())
        .zip(initial_predictions.iter())
        .zip(posterior_predictions.iter())
        .map(
            |(((spec, observed), initial_prediction), posterior_prediction)| {
                Team13SyntheticBenchmarkObservationReport {
                    name: spec.name.clone(),
                    group: spec.group,
                    steel_surface_group: spec.steel_surface_group,
                    observed: *observed,
                    initial_prediction: *initial_prediction,
                    posterior_prediction: *posterior_prediction,
                    residual: *posterior_prediction - *observed,
                }
            },
        )
        .collect())
}

fn synthetic_benchmark_group_summaries(
    reports: &[Team13SyntheticBenchmarkObservationReport],
) -> Vec<Team13SyntheticBenchmarkObservationGroupSummary> {
    [
        Team13SyntheticBenchmarkObservationGroup::SteelAverage,
        Team13SyntheticBenchmarkObservationGroup::AirPoint,
    ]
    .into_iter()
    .filter_map(|group| {
        let group_reports = reports
            .iter()
            .filter(|report| report.group == group)
            .collect::<Vec<_>>();
        if group_reports.is_empty() {
            return None;
        }
        let count = group_reports.len();
        let initial_rmse = (group_reports
            .iter()
            .map(|report| {
                let residual = report.initial_prediction - report.observed;
                residual * residual
            })
            .sum::<f64>()
            / count as f64)
            .sqrt();
        let posterior_rmse = (group_reports
            .iter()
            .map(|report| report.residual * report.residual)
            .sum::<f64>()
            / count as f64)
            .sqrt();
        let observed_norm = (group_reports
            .iter()
            .map(|report| report.observed * report.observed)
            .sum::<f64>()
            / count as f64)
            .sqrt();
        let posterior_max_abs_residual = group_reports
            .iter()
            .map(|report| report.residual.abs())
            .fold(0.0, f64::max);
        Some(Team13SyntheticBenchmarkObservationGroupSummary {
            group,
            count,
            initial_rmse,
            posterior_rmse,
            posterior_relative_rmse: safe_ratio(posterior_rmse, observed_norm),
            posterior_max_abs_residual,
        })
    })
    .collect()
}

fn published_steel_reports_from_predictions(
    specs: &[Team13SyntheticBenchmarkObservationSpec],
    nominal_predictions: &[f64],
    posterior_predictions: &[f64],
) -> Result<Vec<Team13PublishedSteelBenchmarkReport>, String> {
    if specs.len() != TEAM13_OBSERVATION_COUNT
        || nominal_predictions.len() != specs.len()
        || posterior_predictions.len() != specs.len()
    {
        return Err(
            "published steel report vectors must match the 25 steel observations".to_string(),
        );
    }
    let g_052 = team13_published_steel_observations(Team13PublishedSteelGap::G052);
    let g_047 = team13_published_steel_observations(Team13PublishedSteelGap::G047);
    specs
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            let group = spec.steel_surface_group.ok_or_else(|| {
                format!(
                    "steel benchmark spec `{}` is missing a steel group",
                    spec.name
                )
            })?;
            Ok(Team13PublishedSteelBenchmarkReport {
                name: spec.name.clone(),
                group,
                observed_g_052: g_052[index],
                observed_g_047: g_047[index],
                nominal_prediction: nominal_predictions[index],
                posterior_prediction: posterior_predictions[index],
            })
        })
        .collect()
}

fn published_steel_rmse(
    reports: &[Team13PublishedSteelBenchmarkReport],
    gap: Team13PublishedSteelGap,
    use_nominal: bool,
) -> f64 {
    if reports.is_empty() {
        return f64::NAN;
    }
    (reports
        .iter()
        .map(|report| {
            let observed = match gap {
                Team13PublishedSteelGap::G052 => report.observed_g_052,
                Team13PublishedSteelGap::G047 => report.observed_g_047,
            };
            let prediction = if use_nominal {
                report.nominal_prediction
            } else {
                report.posterior_prediction
            };
            let residual = prediction - observed;
            residual * residual
        })
        .sum::<f64>()
        / reports.len() as f64)
        .sqrt()
}

fn published_steel_group_summaries(
    reports: &[Team13PublishedSteelBenchmarkReport],
    use_nominal: bool,
) -> Vec<Team13PublishedSteelGroupSummary> {
    [
        Team13SteelSurfaceGroup::MidSheet,
        Team13SteelSurfaceGroup::BackRightTop,
        Team13SteelSurfaceGroup::BackRightEdge,
    ]
    .into_iter()
    .filter_map(|group| {
        let group_reports = reports
            .iter()
            .filter(|report| report.group == group)
            .collect::<Vec<_>>();
        if group_reports.is_empty() {
            return None;
        }
        let count = group_reports.len();
        let mut sum_g_052 = 0.0;
        let mut sum_g_047 = 0.0;
        let mut max_g_052 = 0.0_f64;
        let mut max_g_047 = 0.0_f64;
        for report in &group_reports {
            let prediction = if use_nominal {
                report.nominal_prediction
            } else {
                report.posterior_prediction
            };
            let residual_g_052 = prediction - report.observed_g_052;
            let residual_g_047 = prediction - report.observed_g_047;
            sum_g_052 += residual_g_052 * residual_g_052;
            sum_g_047 += residual_g_047 * residual_g_047;
            max_g_052 = max_g_052.max(residual_g_052.abs());
            max_g_047 = max_g_047.max(residual_g_047.abs());
        }
        Some(Team13PublishedSteelGroupSummary {
            group,
            count,
            rmse_g_052: (sum_g_052 / count as f64).sqrt(),
            rmse_g_047: (sum_g_047 / count as f64).sqrt(),
            max_abs_residual_g_052: max_g_052,
            max_abs_residual_g_047: max_g_047,
        })
    })
    .collect()
}

fn run_team13_source_scale_diagnostics(
    config: &Team13SyntheticBenchmarkGeometryConfig,
    linear_source_free: &ReducedVectorPotentialMagnetostatic3d,
    source_free: &ReducedVectorPotentialMagnetostatic3d,
    nominal_source: &[f64],
    observations: &Team13SyntheticBenchmarkObservationBuild,
) -> Result<Vec<Team13SourceScaleDiagnosticRun>, String> {
    let mut diagnostics = Vec::with_capacity(config.source_scale_diagnostic_values.len());
    for source_scale in &config.source_scale_diagnostic_values {
        diagnostics.push(run_team13_source_scale_diagnostic(
            config,
            linear_source_free,
            source_free,
            nominal_source,
            observations,
            *source_scale,
        ));
    }
    Ok(diagnostics)
}

fn run_team13_source_scale_diagnostic(
    config: &Team13SyntheticBenchmarkGeometryConfig,
    linear_source_free: &ReducedVectorPotentialMagnetostatic3d,
    source_free: &ReducedVectorPotentialMagnetostatic3d,
    nominal_source: &[f64],
    observations: &Team13SyntheticBenchmarkObservationBuild,
    source_scale: f64,
) -> Team13SourceScaleDiagnosticRun {
    match try_run_team13_source_scale_diagnostic(
        config,
        linear_source_free,
        source_free,
        nominal_source,
        observations,
        source_scale,
    ) {
        Ok(run) => run,
        Err(error) => Team13SourceScaleDiagnosticRun {
            source_scale,
            converged: false,
            error: Some(error),
            initial_residual_norm: f64::NAN,
            final_residual_norm: f64::NAN,
            steel_rmse_g_052: f64::NAN,
            steel_rmse_g_047: f64::NAN,
            group_summaries: Vec::new(),
        },
    }
}

fn try_run_team13_source_scale_diagnostic(
    config: &Team13SyntheticBenchmarkGeometryConfig,
    linear_source_free: &ReducedVectorPotentialMagnetostatic3d,
    source_free: &ReducedVectorPotentialMagnetostatic3d,
    nominal_source: &[f64],
    observations: &Team13SyntheticBenchmarkObservationBuild,
    source_scale: f64,
) -> Result<Team13SourceScaleDiagnosticRun, String> {
    if !source_scale.is_finite() || source_scale <= 0.0 {
        return Err("source scale must be finite and positive".to_string());
    }
    let scaled_source = nominal_source
        .iter()
        .map(|value| source_scale * *value)
        .collect::<Vec<_>>();
    let initial = solve_source_free_linear_model(linear_source_free, &scaled_source)?;
    let model = source_free.clone().with_source(scaled_source)?;
    let initial_residual_norm = l2_norm(&model.residual_and_jacobian(&initial)?.residual);
    let solve = solve_team13_feec_forward_newton(
        &model,
        initial,
        config.truth_max_iterations,
        config.linear_solve,
    )?;
    let final_residual_norm = l2_norm(&model.residual_and_jacobian(&solve.solution)?.residual);
    let predictions = observations
        .assimilated_model
        .smooth_norm_values(&solve.solution)?;
    let reports = published_steel_reports_from_predictions(
        &observations.assimilated_specs,
        &predictions,
        &predictions,
    )?;
    Ok(Team13SourceScaleDiagnosticRun {
        source_scale,
        converged: solve.converged,
        error: None,
        initial_residual_norm,
        final_residual_norm,
        steel_rmse_g_052: published_steel_rmse(&reports, Team13PublishedSteelGap::G052, false),
        steel_rmse_g_047: published_steel_rmse(&reports, Team13PublishedSteelGap::G047, false),
        group_summaries: published_steel_group_summaries(&reports, false),
    })
}

fn rmse_from_prediction_pairs(predictions: &[f64], observations: &[f64]) -> Result<f64, String> {
    if predictions.len() != observations.len() {
        return Err("prediction and observation counts must match".to_string());
    }
    if predictions.is_empty() {
        return Err("at least one prediction is required for RMSE".to_string());
    }
    Ok((predictions
        .iter()
        .zip(observations.iter())
        .map(|(prediction, observation)| {
            let residual = *prediction - *observation;
            residual * residual
        })
        .sum::<f64>()
        / predictions.len() as f64)
        .sqrt())
}

fn relative_rmse_from_prediction_pairs(
    predictions: &[f64],
    observations: &[f64],
) -> Result<f64, String> {
    let rmse = rmse_from_prediction_pairs(predictions, observations)?;
    let observed_norm = (observations
        .iter()
        .map(|observation| observation * observation)
        .sum::<f64>()
        / observations.len() as f64)
        .sqrt();
    Ok(safe_ratio(rmse, observed_norm))
}

fn max_abs_residual_from_prediction_pairs(
    predictions: &[f64],
    observations: &[f64],
) -> Result<f64, String> {
    if predictions.len() != observations.len() {
        return Err("prediction and observation counts must match".to_string());
    }
    Ok(predictions
        .iter()
        .zip(observations.iter())
        .map(|(prediction, observation)| (*prediction - *observation).abs())
        .fold(0.0, f64::max))
}

fn same_positive_scale(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= 1.0e-12 * lhs.abs().max(rhs.abs()).max(1.0)
}

fn synthetic_surface_sign_mismatch_count(measurements: &[Team13LinearizedMeasurement]) -> usize {
    measurements
        .iter()
        .filter(|measurement| {
            tolerant_sign(measurement.nominal_prediction)
                != tolerant_sign(measurement.spec.observations[0])
        })
        .count()
}

fn classify_first_step_diagnostics(
    diagnostics: &GaussNewtonFirstStepDiagnostics,
) -> Team13FirstStepConditioningClass {
    let finite = diagnostics.objective.is_finite()
        && diagnostics.weighted_residual_norm.is_finite()
        && diagnostics.gradient_norm.is_finite()
        && diagnostics.step_norm.is_finite()
        && diagnostics.directional_derivative.is_finite()
        && diagnostics.linear_solve_absolute_residual_norm.is_finite()
        && diagnostics.linear_solve_relative_residual_norm.is_finite()
        && diagnostics
            .objective_grid
            .iter()
            .all(|sample| sample.alpha.is_finite() && sample.objective.is_finite());
    if !finite || !diagnostics.linear_solve.converged {
        return Team13FirstStepConditioningClass::IllConditioned;
    }
    if diagnostics.linear_solve_relative_residual_norm <= 1e-8
        && diagnostics.accepted_alpha.unwrap_or(0.0) >= 2.0_f64.powi(-20)
    {
        Team13FirstStepConditioningClass::WellConditioned
    } else {
        Team13FirstStepConditioningClass::IllConditioned
    }
}

fn restrict_triplet_columns_to_layout(
    operator: &SparseTripletMatrix,
    layout: &DofLayout,
) -> Result<SparseTripletMatrix, String> {
    if operator.ncols() != layout.full_dimension {
        return Err(format!(
            "operator column count {} must match layout full dimension {}",
            operator.ncols(),
            layout.full_dimension
        ));
    }
    let mut reduced_index = vec![None; layout.full_dimension];
    for (index, full) in layout.active_dofs.iter().copied().enumerate() {
        reduced_index[full] = Some(index);
    }
    Ok(SparseTripletMatrix::from_triplets(
        operator.nrows(),
        layout.reduced_dimension(),
        operator.triplet_iter().filter_map(|(row, col, value)| {
            reduced_index[col].map(|reduced_col| SparseTriplet {
                row,
                col: reduced_col,
                value,
            })
        }),
    ))
}

fn sensor_variance_reports(
    measurements: &[Team13LinearizedMeasurement],
    variances: &BTreeMap<String, LinearPdeDerivedMarginalResult>,
) -> Vec<Team13NonlinearSensorVarianceReport> {
    measurements
        .iter()
        .filter_map(|measurement| {
            let name = sensor_derived_name(&measurement.spec.name);
            let variance = variances.get(&name)?;
            Some(Team13NonlinearSensorVarianceReport {
                name: measurement.spec.name.clone(),
                prior_variance: if variance.prior_variance.is_empty() {
                    f64::NAN
                } else {
                    variance.prior_variance[0]
                },
                posterior_variance: if variance.posterior_variance.is_empty() {
                    f64::NAN
                } else {
                    variance.posterior_variance[0]
                },
            })
        })
        .collect()
}

fn team13_pde_residual_noise(
    config: &Team13SourceRecoveryConfig,
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
) -> Result<(Option<f64>, Option<SparseTripletMatrix>), String> {
    match config.pde_residual_weighting {
        Team13PdeResidualWeighting::Euclidean => Ok((Some(config.pde_variance), None)),
        Team13PdeResidualWeighting::MassInverse
        | Team13PdeResidualWeighting::MassInverseTraceNormalized => Ok((
            None,
            Some(team13_mass_weighted_pde_precision(config, system)?),
        )),
    }
}

fn team13_mass_weighted_pde_precision(
    config: &Team13SourceRecoveryConfig,
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
) -> Result<SparseTripletMatrix, String> {
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
    let mut precision_scale = 1.0;
    if config.pde_residual_weighting == Team13PdeResidualWeighting::MassInverseTraceNormalized {
        precision_scale *= reciprocal_mean_diagonal(mass_inverse)?;
    }
    Ok(scale_triplet_matrix(
        &csr_to_triplet(mass_inverse),
        precision_scale / config.pde_variance,
    ))
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
            "mean-diagonal normalization requires positive finite trace, got {trace:.6e}"
        ));
    }
    Ok(dimension as f64 / trace)
}

fn set_measurement_observations_from_nominal_prediction(
    measurements: &mut [Team13LinearizedMeasurement],
    scale: f64,
) -> Result<(), String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("measurement observation scale must be finite and positive".to_string());
    }
    for measurement in measurements {
        measurement.spec.observations[0] = scale * measurement.nominal_prediction;
    }
    Ok(())
}

fn scale_measurement_observations(
    measurements: &mut [Team13LinearizedMeasurement],
    scale: f64,
) -> Result<(), String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("measurement observation scale must be finite and positive".to_string());
    }
    for measurement in measurements {
        measurement.spec.observations[0] *= scale;
    }
    Ok(())
}

fn measurement_observation_overrides(
    measurements: &[Team13LinearizedMeasurement],
) -> BTreeMap<String, f64> {
    measurements
        .iter()
        .map(|measurement| {
            (
                measurement.spec.name.clone(),
                measurement.spec.observations[0],
            )
        })
        .collect()
}

fn flat_state_prior(dimension: usize) -> GaussianPriorSpec {
    GaussianPriorSpec {
        mean: vec![0.0; dimension],
        precision: SparseTripletMatrix::new(dimension, dimension),
    }
}

fn solve_team13_feec_forward_newton(
    model: &ReducedVectorPotentialMagnetostatic3d,
    initial: Vec<f64>,
    max_iterations: usize,
    linear_solve: GaussNewtonLinearSolve,
) -> Result<feg_infer::nonlinear::SquareNewtonResult, String> {
    let adapter = FeecResidualAdapter::new(model);
    solve_team13_forward_newton(&adapter, initial, max_iterations, linear_solve)
}

fn solve_team13_forward_newton(
    model: &dyn NonlinearResidualModel,
    initial: Vec<f64>,
    max_iterations: usize,
    linear_solve: GaussNewtonLinearSolve,
) -> Result<feg_infer::nonlinear::SquareNewtonResult, String> {
    solve_square_nonlinear_system(
        model,
        &SquareNewtonConfig {
            initial_guess: Some(initial),
            max_iterations,
            residual_tolerance: 1e-8,
            step_tolerance: 1e-10,
            max_line_search_steps: 40,
            linear_solve,
            ..SquareNewtonConfig::default()
        },
    )
}

fn team13_truth_cache_path(
    mesh_bytes: &[u8],
    config: &Team13MapParityConfig,
    active_dofs: usize,
) -> PathBuf {
    PathBuf::from(TEAM13_TRUTH_CACHE_DIR).join(format!(
        "truth_{}.bin",
        team13_truth_cache_key(mesh_bytes, config, active_dofs)
    ))
}

fn team13_truth_cache_key(
    mesh_bytes: &[u8],
    config: &Team13MapParityConfig,
    active_dofs: usize,
) -> String {
    let mut hash = Fnv64::new();
    hash.bytes(b"team13-truth-cache");
    hash.u64(TEAM13_TRUTH_CACHE_VERSION);
    hash.bytes(mesh_bytes);
    hash.bytes(config.domain_mode.as_str().as_bytes());
    hash.bytes(config.material_kind.as_str().as_bytes());
    hash.u64(config.beta_iron.to_bits());
    hash.u64(config.b_scale_tesla.to_bits());
    hash.u64(config.ampere_turns.to_bits());
    hash.usize(active_dofs);
    hash.usize(config.truth_max_iterations);
    hash_gauss_newton_linear_solve(&mut hash, &config.linear_solve);
    format!("{:016x}", hash.finish())
}

fn hash_gauss_newton_linear_solve(hash: &mut Fnv64, linear_solve: &GaussNewtonLinearSolve) {
    match *linear_solve {
        GaussNewtonLinearSolve::DirectCholesky => {
            hash.bytes(b"direct-cholesky");
        }
        GaussNewtonLinearSolve::IterativeCg {
            tolerance,
            max_iterations,
            warm_start,
        } => {
            hash.bytes(b"iterative-cg");
            hash.u64(tolerance.to_bits());
            hash.usize(max_iterations);
            hash.u64(u64::from(warm_start));
        }
    }
}

fn try_load_team13_truth_cache(
    path: &Path,
    model: &dyn NonlinearResidualModel,
    expected_dimension: usize,
    residual_tolerance: f64,
) -> Option<feg_infer::nonlinear::SquareNewtonResult> {
    if !path.exists() {
        return None;
    }
    match fs::read(path)
        .map_err(|err| err.to_string())
        .and_then(|bytes| parse_team13_truth_cache(&bytes, expected_dimension))
        .and_then(|mut result| {
            let residual = model.residual(&result.solution)?;
            let residual_norm = l2_norm(&residual);
            if !residual_norm.is_finite() || residual_norm > residual_tolerance {
                return Err(format!(
                    "cached truth residual {residual_norm:.6e} exceeds tolerance {residual_tolerance:.6e}"
                ));
            }
            result.residual = residual;
            result.residual_norm = residual_norm;
            Ok(result)
        }) {
        Ok(result) => Some(result),
        Err(err) => {
            eprintln!(
                "TEAM 13 MAP parity: truth cache ignored `{}`: {err}",
                path.display()
            );
            None
        }
    }
}

fn parse_team13_truth_cache(
    bytes: &[u8],
    expected_dimension: usize,
) -> Result<feg_infer::nonlinear::SquareNewtonResult, String> {
    let mut cursor = Cursor::new(bytes);
    let mut magic = [0u8; 16];
    cursor
        .read_exact(&mut magic)
        .map_err(|err| format!("truth cache is too short to contain magic: {err}"))?;
    if &magic != TEAM13_TRUTH_CACHE_MAGIC {
        return Err("truth cache magic/version tag did not match".to_string());
    }
    let version = read_cache_u64(&mut cursor)?;
    if version != TEAM13_TRUTH_CACHE_VERSION {
        return Err(format!(
            "truth cache version {version} did not match expected {TEAM13_TRUTH_CACHE_VERSION}"
        ));
    }
    let dimension = read_cache_usize(&mut cursor)?;
    if dimension != expected_dimension {
        return Err(format!(
            "cached truth dimension {dimension} did not match expected {expected_dimension}"
        ));
    }
    let converged = read_cache_bool(&mut cursor)?;
    if !converged {
        return Err("cached truth solve was not converged".to_string());
    }
    let residual_norm = read_cache_f64(&mut cursor)?;
    if !residual_norm.is_finite() {
        return Err("cached truth residual norm is not finite".to_string());
    }
    let solution_len = read_cache_usize(&mut cursor)?;
    if solution_len != dimension {
        return Err(format!(
            "cached truth solution length {solution_len} did not match dimension {dimension}"
        ));
    }
    let mut solution = Vec::with_capacity(solution_len);
    for _ in 0..solution_len {
        let value = read_cache_f64(&mut cursor)?;
        if !value.is_finite() {
            return Err("cached truth solution contains a non-finite value".to_string());
        }
        solution.push(value);
    }
    let history_len = read_cache_usize(&mut cursor)?;
    if history_len > 10_000 {
        return Err("cached truth history length is implausibly large".to_string());
    }
    let mut history = Vec::with_capacity(history_len);
    for _ in 0..history_len {
        history.push(read_cache_square_newton_iteration(&mut cursor)?);
    }
    if cursor.position() != bytes.len() as u64 {
        return Err("truth cache contains trailing bytes".to_string());
    }
    Ok(feg_infer::nonlinear::SquareNewtonResult {
        solution,
        residual: Vec::new(),
        residual_norm,
        history,
        converged,
    })
}

fn write_team13_truth_cache(
    path: &Path,
    result: &feg_infer::nonlinear::SquareNewtonResult,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create TEAM 13 truth cache directory `{}`: {err}",
                parent.display()
            )
        })?;
    }
    let mut bytes = Vec::with_capacity(64 + 8 * result.solution.len());
    bytes.extend_from_slice(TEAM13_TRUTH_CACHE_MAGIC);
    write_cache_u64(&mut bytes, TEAM13_TRUTH_CACHE_VERSION);
    write_cache_usize(&mut bytes, result.solution.len());
    write_cache_bool(&mut bytes, result.converged);
    write_cache_f64(&mut bytes, result.residual_norm);
    write_cache_usize(&mut bytes, result.solution.len());
    for value in &result.solution {
        write_cache_f64(&mut bytes, *value);
    }
    write_cache_usize(&mut bytes, result.history.len());
    for iteration in &result.history {
        write_cache_square_newton_iteration(&mut bytes, iteration);
    }
    fs::write(path, bytes)
        .map_err(|err| format!("failed to write truth cache `{}`: {err}", path.display()))
}

fn zero_source_operator(residual_dimension: usize, source_dimension: usize) -> SparseTripletMatrix {
    SparseTripletMatrix::new(residual_dimension, source_dimension)
}

fn scalar_sensor_sensitivities(measurements: &[Team13LinearizedMeasurement]) -> Vec<Vec<f64>> {
    measurements
        .iter()
        .map(|measurement| vec![measurement.nominal_prediction])
        .collect()
}

fn build_fluctuation_joint_measurements(
    measurements: &[Team13LinearizedMeasurement],
    input_name: &str,
    sensitivities: &[Vec<f64>],
) -> Result<Vec<LinearPdeJointMeasurementSpec>, String> {
    if measurements.len() != sensitivities.len() {
        return Err(format!(
            "measurement count {} must match source sensitivity row count {}",
            measurements.len(),
            sensitivities.len()
        ));
    }
    measurements
        .iter()
        .zip(sensitivities.iter())
        .map(|(measurement, sensitivity)| {
            Ok(LinearPdeJointMeasurementSpec {
                name: measurement.spec.name.clone(),
                state_operator: Some(measurement.spec.operator.clone()),
                latent_operators: vec![LinearPdeLatentMeasurementBlockSpec {
                    input_name: input_name.to_string(),
                    operator: dense_row_to_sparse_matrix(sensitivity),
                }],
                observations: measurement.spec.observations.clone(),
                bias: vec![0.0; measurement.spec.observations.len()],
                variance: measurement.spec.variance,
            })
        })
        .collect()
}

#[cfg(test)]
fn build_global_source_operator(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    galmats: &MixedGalmats,
    boundary: &EssentialBoundarySpec,
    mode: Team13DomainMode,
    ampere_turns: f64,
) -> Result<SparseTripletMatrix, String> {
    let source = team13_current_density(mode, ampere_turns, None);
    let rhs = assemble_unweighted_source(topology, metric, coords, &source);
    let reduced = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        galmats,
        boundary,
        &FeecVector::zeros(galmats.sigma_len()),
        &rhs,
    )?;
    Ok(columns_to_sparse_matrix(&[reduced.scale(-1.0)]))
}

fn reduce_team13_physical_source_rhs(
    galmats: &MixedGalmats,
    boundary: &EssentialBoundarySpec,
    source: &FeecVector,
) -> Result<Vec<f64>, String> {
    let sigma_zero = FeecVector::zeros(galmats.sigma_len());
    let source_rhs = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        galmats,
        boundary,
        &sigma_zero,
        source,
    )?;
    let zero_rhs = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        galmats,
        boundary,
        &sigma_zero,
        &FeecVector::zeros(galmats.u_len()),
    )?;
    Ok(source_rhs
        .iter()
        .zip(zero_rhs.iter())
        .map(|(source_value, zero_value)| *source_value - *zero_value)
        .collect())
}

fn solve_source_free_linear_model(
    model: &ReducedVectorPotentialMagnetostatic3d,
    source: &[f64],
) -> Result<Vec<f64>, String> {
    if source.len() != model.reduced_dimension() {
        return Err(format!(
            "reduced source length {} must match TEAM13 beta-zero model dimension {}",
            source.len(),
            model.reduced_dimension()
        ));
    }
    let zero_state = vec![0.0; model.reduced_dimension()];
    let evaluation = model.source_free_residual_and_jacobian(&zero_state)?;
    let rhs = source
        .iter()
        .zip(evaluation.residual.iter())
        .map(|(source_value, bias_value)| *source_value - *bias_value)
        .collect::<Vec<_>>();
    let factor = feec_csr_to_gmrf(&evaluation.jacobian)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factorize beta-zero TEAM13 nonlinear operator: {err}"))?;
    Ok(factor
        .solve(&GmrfVector::from_vec(rhs))
        .map_err(|err| format!("failed to solve beta-zero TEAM13 nonlinear operator: {err}"))?
        .iter()
        .copied()
        .collect())
}

fn build_nominal_state_source_operator(
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    reduced_nominal_state: &FeecVector,
) -> Result<SparseTripletMatrix, String> {
    if reduced_nominal_state.len() != system.state_dimension() {
        return Err(format!(
            "nominal state length {} must match reduced state dimension {}",
            reduced_nominal_state.len(),
            system.state_dimension()
        ));
    }
    let implied_source = &system.operator * reduced_nominal_state;
    Ok(columns_to_sparse_matrix(&[implied_source.scale(-1.0)]))
}

fn system_with_zero_residual_bias(
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
) -> formoniq::problems::reduced_linear::ReducedLinearPdeAssembly {
    let mut system = system.clone();
    system.residual_bias.fill(0.0);
    system
}

fn single_column_vector(matrix: &SparseTripletMatrix) -> Result<FeecVector, String> {
    if matrix.ncols() != 1 {
        return Err(format!(
            "expected a single-column source operator, got {} columns",
            matrix.ncols()
        ));
    }
    let mut vector = FeecVector::zeros(matrix.nrows());
    for (row, col, value) in matrix.triplet_iter() {
        if col != 0 {
            return Err("single-column source operator contains a nonzero outside column 0".into());
        }
        vector[row] += value;
    }
    Ok(vector)
}

fn system_with_added_residual_bias(
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    bias: &FeecVector,
) -> Result<formoniq::problems::reduced_linear::ReducedLinearPdeAssembly, String> {
    if bias.len() != system.residual_dimension() {
        return Err(format!(
            "source bias length {} must match residual dimension {}",
            bias.len(),
            system.residual_dimension()
        ));
    }
    let mut system = system.clone();
    for (entry, value) in system.residual_bias.iter_mut().zip(bias.iter()) {
        *entry += value;
    }
    Ok(system)
}

fn build_source_recovery_derived_quantities(
    operators: &Team13Operators,
    measurements: &[Team13LinearizedMeasurement],
    period_diagnostics: &[Team13PeriodDiagnosticSpec],
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    let mut derived = build_derived_quantities(operators)?;
    for measurement in measurements {
        derived.push(LinearPdeDerivedQuantitySpec {
            name: sensor_derived_name(&measurement.spec.name),
            operator: triplet_to_sparse_row_operator(&measurement.spec.operator)?,
        });
    }
    for diagnostic in period_diagnostics {
        derived.push(LinearPdeDerivedQuantitySpec {
            name: period_derived_name(&diagnostic.name),
            operator: diagnostic.operator.clone(),
        });
    }
    Ok(derived)
}

fn build_fluctuation_joint_derived_quantities(
    operators: &Team13Operators,
    measurements: &[Team13LinearizedMeasurement],
    period_diagnostics: &[Team13PeriodDiagnosticSpec],
    input_name: &str,
    source_fields: &[FeecVector],
) -> Result<Vec<LinearPdeJointDerivedQuantitySpec>, String> {
    let mut derived = Vec::new();
    derived.push(fluctuation_joint_derived_quantity(
        A_PHYSICAL_COCHAIN_DERIVED_NAME,
        identity_sparse_row_operator(source_fields.first().map_or(0, FeecVector::len))?,
        input_name,
        source_fields,
    )?);
    let names = [
        A_VECTOR_X_DERIVED_NAME,
        A_VECTOR_Y_DERIVED_NAME,
        A_VECTOR_Z_DERIVED_NAME,
    ];
    for (name, operator) in names.iter().zip(operators.a_components.iter()) {
        derived.push(fluctuation_joint_derived_quantity(
            name,
            operator.clone(),
            input_name,
            source_fields,
        )?);
    }
    derived.push(fluctuation_joint_derived_quantity(
        B_COCHAIN_DERIVED_NAME,
        operators.b_cochain.clone(),
        input_name,
        source_fields,
    )?);
    let names = [
        B_VECTOR_X_DERIVED_NAME,
        B_VECTOR_Y_DERIVED_NAME,
        B_VECTOR_Z_DERIVED_NAME,
    ];
    for (name, operator) in names.iter().zip(operators.b_components.iter()) {
        derived.push(fluctuation_joint_derived_quantity(
            name,
            operator.clone(),
            input_name,
            source_fields,
        )?);
    }
    for measurement in measurements {
        derived.push(fluctuation_joint_derived_quantity(
            &sensor_derived_name(&measurement.spec.name),
            triplet_to_sparse_row_operator(&measurement.spec.operator)?,
            input_name,
            source_fields,
        )?);
    }
    for diagnostic in period_diagnostics {
        derived.push(fluctuation_joint_derived_quantity(
            &period_derived_name(&diagnostic.name),
            diagnostic.operator.clone(),
            input_name,
            source_fields,
        )?);
    }
    Ok(derived)
}

fn fluctuation_joint_derived_quantity(
    name: &str,
    state_operator: SparseRowOperator,
    input_name: &str,
    source_fields: &[FeecVector],
) -> Result<LinearPdeJointDerivedQuantitySpec, String> {
    Ok(LinearPdeJointDerivedQuantitySpec {
        name: name.to_string(),
        state_operator: Some(state_operator.clone()),
        latent_operators: vec![LinearPdeLatentDerivedBlockSpec {
            input_name: input_name.to_string(),
            operator: source_field_columns_to_operator(&state_operator, source_fields)?,
        }],
    })
}

fn identity_sparse_row_operator(dimension: usize) -> Result<SparseRowOperator, String> {
    SparseRowOperator::new(
        dimension,
        (0..dimension)
            .map(|index| vec![(index, 1.0)])
            .collect::<Vec<_>>(),
    )
    .map_err(|err| err.to_string())
}

fn source_field_columns_to_operator(
    operator: &SparseRowOperator,
    source_fields: &[FeecVector],
) -> Result<SparseRowOperator, String> {
    let columns = source_fields
        .iter()
        .map(|field| apply_operator_to_feec(operator, field))
        .collect::<Result<Vec<_>, _>>()?;
    columns_to_sparse_row_operator(&columns)
}

fn sensor_derived_name(name: &str) -> String {
    format!("{SENSOR_DERIVED_PREFIX}{name}")
}

fn period_derived_name(name: &str) -> String {
    format!("{PERIOD_DERIVED_PREFIX}{name}")
}

fn team13_sensor_areas() -> Result<BTreeMap<String, f64>, String> {
    team13_surface_measurement_definitions(None).map(|definitions| {
        definitions
            .into_iter()
            .map(|definition| {
                (
                    definition.name.clone(),
                    surface_measurement_area(&definition),
                )
            })
            .collect()
    })
}

fn surface_measurement_area(definition: &Team13SurfaceMeasurementDefinition) -> f64 {
    let lengths = [
        (definition.x_range.1 - definition.x_range.0).abs(),
        (definition.y_range.1 - definition.y_range.0).abs(),
        (definition.z_range.1 - definition.z_range.0).abs(),
    ];
    lengths
        .iter()
        .enumerate()
        .filter_map(|(axis, length)| (axis != definition.normal_axis).then_some(*length))
        .product::<f64>()
}

#[allow(clippy::too_many_arguments)]
fn run_team13_source_recovery_stage(
    name: &str,
    config: &Team13SourceRecoveryConfig,
    problem: LinearPdeUqProblem,
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    nominal_a: &FeecVector,
    field_reference: &str,
    reference_a: &FeecVector,
    state_active_mask: &FeecVector,
    report_measurements: &[Team13LinearizedMeasurement],
    observation_overrides: Option<&BTreeMap<String, f64>>,
    sensor_areas: &BTreeMap<String, f64>,
    period_diagnostics: &[Team13PeriodDiagnosticSpec],
) -> Result<Team13SourceRecoveryStageResult, String> {
    let posterior = solve_linear_pde_uq_with_config(&problem, &config.solver)?;
    let sensor_reports = evaluate_sensor_reports(report_measurements, &posterior.posterior_mean)?;
    let benchmark_reports = evaluate_benchmark_reports(
        topology,
        coords,
        operators,
        nominal_a,
        &posterior.posterior_mean,
        config.measurement_mode,
        config.legacy_measurement_band,
        observation_overrides,
    )?;
    let a_variance_ratio = variance_ratio(&posterior.posterior_variance, &posterior.prior_variance);
    let b_variance_ratio = derived_variance_ratio(&posterior, B_COCHAIN_DERIVED_NAME)?;
    let vector_pushforwards =
        build_vector_pushforwards(operators, &posterior.posterior_mean, &posterior)?;
    let field_metrics = field_recovery_metrics(
        operators,
        reference_a,
        &posterior.posterior_mean,
        state_active_mask,
    )?;
    let sensor_uncertainty =
        build_sensor_uncertainty_reports(name, &sensor_reports, &posterior, sensor_areas)?;
    let period_diagnostics =
        build_period_diagnostic_reports(name, period_diagnostics, reference_a, &posterior)?;
    let b_vector_trace_variance_ratio_mean = vector_pushforwards
        .get("B")
        .map(|pushforward| finite_mean(pushforward.trace_variance_ratio.iter().copied()))
        .unwrap_or(0.0);
    let summary = Team13SourceRecoveryStageSummary {
        name: name.to_string(),
        latent_dimension: posterior.debug.joint_dimension,
        pde_residual_norm: posterior.pde_residual_mean.norm(),
        sensor_rmse: sensor_rmse_from_reports(&sensor_reports),
        field_reference: field_reference.to_string(),
        a_active_rmse: field_metrics.a_active_rmse,
        a_active_relative_l2_error: field_metrics.a_active_relative_l2_error,
        b_cochain_rmse: field_metrics.b_cochain_rmse,
        b_cochain_relative_l2_error: field_metrics.b_cochain_relative_l2_error,
        b_vector_rmse: field_metrics.b_vector_rmse,
        b_vector_relative_l2_error: field_metrics.b_vector_relative_l2_error,
        a_variance_ratio_mean: masked_finite_mean(
            a_variance_ratio
                .iter()
                .copied()
                .zip(state_active_mask.iter().copied()),
        ),
        b_variance_ratio_mean: finite_mean(b_variance_ratio.iter().copied()),
        b_vector_trace_variance_ratio_mean,
    };
    let solve = Team13LinearSolveResult {
        domain_mode: config.domain_mode,
        posterior,
        nominal_a: nominal_a.clone(),
        field_reference_name: field_reference.to_string(),
        field_reference_a: reference_a.clone(),
        state_active_mask: state_active_mask.clone(),
        a_variance_ratio,
        b_variance_ratio,
        sensor_reports,
        benchmark_reports,
        vector_pushforwards,
    };
    Ok(Team13SourceRecoveryStageResult {
        summary,
        solve,
        sensor_uncertainty,
        period_diagnostics,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_team13_fluctuation_source_recovery_stage(
    name: &str,
    config: &Team13SourceRecoveryConfig,
    problem: LinearPdeUqProblem,
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    nominal_a: &FeecVector,
    field_reference: &str,
    reference_a: &FeecVector,
    source_input_name: &str,
    source_fields: &[FeecVector],
    state_active_mask: &FeecVector,
    report_measurements: &[Team13LinearizedMeasurement],
    observation_overrides: Option<&BTreeMap<String, f64>>,
    sensor_areas: &BTreeMap<String, f64>,
    period_diagnostics: &[Team13PeriodDiagnosticSpec],
) -> Result<Team13SourceRecoveryStageResult, String> {
    let delta_posterior = solve_linear_pde_uq_with_config(&problem, &config.solver)?;
    let physical_mean =
        reconstruct_physical_field(&delta_posterior, source_input_name, source_fields)?;
    let physical_a_variance =
        derived_variance(&delta_posterior, A_PHYSICAL_COCHAIN_DERIVED_NAME)?.clone();
    let mut posterior = delta_posterior;
    posterior.posterior_mean = physical_mean;
    posterior.posterior_variance = physical_a_variance.posterior_variance;
    posterior.prior_variance = physical_a_variance.prior_variance;

    let sensor_reports = evaluate_sensor_reports(report_measurements, &posterior.posterior_mean)?;
    let benchmark_reports = evaluate_benchmark_reports(
        topology,
        coords,
        operators,
        nominal_a,
        &posterior.posterior_mean,
        config.measurement_mode,
        config.legacy_measurement_band,
        observation_overrides,
    )?;
    let a_variance_ratio = variance_ratio(&posterior.posterior_variance, &posterior.prior_variance);
    let b_variance_ratio = derived_variance_ratio(&posterior, B_COCHAIN_DERIVED_NAME)?;
    let vector_pushforwards =
        build_vector_pushforwards(operators, &posterior.posterior_mean, &posterior)?;
    let field_metrics = field_recovery_metrics(
        operators,
        reference_a,
        &posterior.posterior_mean,
        state_active_mask,
    )?;
    let sensor_uncertainty =
        build_sensor_uncertainty_reports(name, &sensor_reports, &posterior, sensor_areas)?;
    let period_diagnostics =
        build_period_diagnostic_reports(name, period_diagnostics, reference_a, &posterior)?;
    let b_vector_trace_variance_ratio_mean = vector_pushforwards
        .get("B")
        .map(|pushforward| finite_mean(pushforward.trace_variance_ratio.iter().copied()))
        .unwrap_or(0.0);
    let summary = Team13SourceRecoveryStageSummary {
        name: name.to_string(),
        latent_dimension: posterior.debug.joint_dimension,
        pde_residual_norm: posterior.pde_residual_mean.norm(),
        sensor_rmse: sensor_rmse_from_reports(&sensor_reports),
        field_reference: field_reference.to_string(),
        a_active_rmse: field_metrics.a_active_rmse,
        a_active_relative_l2_error: field_metrics.a_active_relative_l2_error,
        b_cochain_rmse: field_metrics.b_cochain_rmse,
        b_cochain_relative_l2_error: field_metrics.b_cochain_relative_l2_error,
        b_vector_rmse: field_metrics.b_vector_rmse,
        b_vector_relative_l2_error: field_metrics.b_vector_relative_l2_error,
        a_variance_ratio_mean: masked_finite_mean(
            a_variance_ratio
                .iter()
                .copied()
                .zip(state_active_mask.iter().copied()),
        ),
        b_variance_ratio_mean: finite_mean(b_variance_ratio.iter().copied()),
        b_vector_trace_variance_ratio_mean,
    };
    let solve = Team13LinearSolveResult {
        domain_mode: config.domain_mode,
        posterior,
        nominal_a: nominal_a.clone(),
        field_reference_name: field_reference.to_string(),
        field_reference_a: reference_a.clone(),
        state_active_mask: state_active_mask.clone(),
        a_variance_ratio,
        b_variance_ratio,
        sensor_reports,
        benchmark_reports,
        vector_pushforwards,
    };
    Ok(Team13SourceRecoveryStageResult {
        summary,
        solve,
        sensor_uncertainty,
        period_diagnostics,
    })
}

fn reconstruct_physical_field(
    posterior: &LinearPdeUqResult,
    source_input_name: &str,
    source_fields: &[FeecVector],
) -> Result<FeecVector, String> {
    let source = latent_input_by_name(&posterior.latent_inputs, source_input_name)?;
    if source.mean.len() != source_fields.len() {
        return Err(format!(
            "source posterior dimension {} does not match source field count {}",
            source.mean.len(),
            source_fields.len()
        ));
    }
    let mut physical = posterior.posterior_mean.clone();
    for (mean, field) in source.mean.iter().zip(source_fields.iter()) {
        if field.len() != physical.len() {
            return Err(format!(
                "source field length {} does not match state length {}",
                field.len(),
                physical.len()
            ));
        }
        physical += field.scale(*mean);
    }
    Ok(physical)
}

fn build_sensor_uncertainty_reports(
    stage: &str,
    reports: &[Team13SensorReport],
    result: &LinearPdeUqResult,
    sensor_areas: &BTreeMap<String, f64>,
) -> Result<Vec<Team13SensorUncertaintyReport>, String> {
    reports
        .iter()
        .map(|report| {
            let derived = derived_variance(result, &sensor_derived_name(&report.name))?;
            let area = sensor_areas
                .get(&report.name)
                .copied()
                .ok_or_else(|| format!("missing surface area for sensor `{}`", report.name))?;
            Ok(Team13SensorUncertaintyReport {
                stage: stage.to_string(),
                name: report.name.clone(),
                area,
                observed: report.observed,
                prediction: report.posterior_prediction,
                residual: report.residual,
                prior_variance: derived.prior_variance[0],
                posterior_variance: derived.posterior_variance[0],
            })
        })
        .collect()
}

fn build_period_diagnostic_reports(
    stage: &str,
    diagnostics: &[Team13PeriodDiagnosticSpec],
    reference_a: &FeecVector,
    result: &LinearPdeUqResult,
) -> Result<Vec<Team13PeriodDiagnosticReport>, String> {
    diagnostics
        .iter()
        .map(|diagnostic| {
            let derived_name = period_derived_name(&diagnostic.name);
            let variance = derived_variance(result, &derived_name)?;
            let truth = diagnostic
                .operator
                .apply(&GmrfVector::from_vec(reference_a.iter().copied().collect()))
                .map_err(|err| err.to_string())?[0];
            let prediction = diagnostic
                .operator
                .apply(&GmrfVector::from_vec(
                    result.posterior_mean.iter().copied().collect(),
                ))
                .map_err(|err| err.to_string())?[0];
            let prior_variance = variance.prior_variance[0];
            let posterior_variance = variance.posterior_variance[0];
            Ok(Team13PeriodDiagnosticReport {
                stage: stage.to_string(),
                name: diagnostic.name.clone(),
                truth,
                prediction,
                residual: prediction - truth,
                prior_variance,
                posterior_variance,
                variance_ratio: safe_ratio(posterior_variance, prior_variance),
            })
        })
        .collect()
}

fn source_posterior_summary_from_sensor_scaling(
    nominal_measurements: &[Team13LinearizedMeasurement],
    perturbed_measurements: &[Team13LinearizedMeasurement],
    prior_variance: f64,
    config: &Team13SourceRecoveryConfig,
) -> Result<Team13SourcePosteriorSummary, String> {
    if nominal_measurements.len() != perturbed_measurements.len() {
        return Err(format!(
            "nominal measurement count {} must match perturbed measurement count {}",
            nominal_measurements.len(),
            perturbed_measurements.len()
        ));
    }
    let sensor_variance = config.b_observation_std_tesla * config.b_observation_std_tesla;
    let mut precision = 1.0 / prior_variance;
    let mut information = 1.0 / prior_variance;
    for (nominal, perturbed) in nominal_measurements
        .iter()
        .zip(perturbed_measurements.iter())
    {
        if nominal.spec.name != perturbed.spec.name {
            return Err(format!(
                "nominal measurement `{}` does not match perturbed measurement `{}`",
                nominal.spec.name, perturbed.spec.name
            ));
        }
        let sensitivity = nominal.spec.observations[0];
        let observed = perturbed.spec.observations[0];
        precision += sensitivity * sensitivity / sensor_variance;
        information += sensitivity * observed / sensor_variance;
    }
    let posterior_variance = 1.0 / precision;
    let posterior_mean = posterior_variance * information;
    Ok(Team13SourcePosteriorSummary {
        prior_mean: 1.0,
        prior_variance,
        posterior_mean,
        posterior_variance,
        variance_ratio: safe_ratio(posterior_variance, prior_variance),
        true_alpha: config.source_alpha_true,
        recovery_error: posterior_mean - config.source_alpha_true,
    })
}

fn source_posterior_summary_from_stage(
    stage: &Team13SourceRecoveryStageResult,
    input_name: &str,
    prior_mean: f64,
    prior_variance: f64,
    true_alpha: f64,
) -> Result<Team13SourcePosteriorSummary, String> {
    let input = latent_input_by_name(&stage.solve.posterior.latent_inputs, input_name)?;
    if input.mean.len() != 1 || input.variance.len() != 1 {
        return Err(format!(
            "latent input `{input_name}` has dimension {}, expected scalar",
            input.mean.len()
        ));
    }
    let posterior_mean = input.mean[0];
    let posterior_variance = input.variance[0];
    Ok(Team13SourcePosteriorSummary {
        prior_mean,
        prior_variance,
        posterior_mean,
        posterior_variance,
        variance_ratio: safe_ratio(posterior_variance, prior_variance),
        true_alpha,
        recovery_error: posterior_mean - true_alpha,
    })
}

fn field_prior_comparison_from_stage(
    prior_kind: Team13FieldPriorKind,
    stage: &Team13SourceRecoveryStageResult,
    source_posterior: Team13SourcePosteriorSummary,
) -> Team13FieldPriorComparison {
    let (all_finite_variances, nonnegative_variances) =
        variance_quality_flags(&stage.solve.posterior);
    Team13FieldPriorComparison {
        prior_kind,
        stage_name: stage.summary.name.clone(),
        source_posterior,
        sensor_rmse: stage.summary.sensor_rmse,
        a_active_relative_l2_error: stage.summary.a_active_relative_l2_error,
        b_cochain_relative_l2_error: stage.summary.b_cochain_relative_l2_error,
        b_vector_relative_l2_error: stage.summary.b_vector_relative_l2_error,
        a_variance_ratio_mean: stage.summary.a_variance_ratio_mean,
        b_variance_ratio_mean: stage.summary.b_variance_ratio_mean,
        b_vector_trace_variance_ratio_mean: stage.summary.b_vector_trace_variance_ratio_mean,
        prior_factor_nnz: stage.solve.posterior.debug.prior_factorization.factor_nnz,
        posterior_factor_nnz: stage
            .solve
            .posterior
            .debug
            .posterior_factorization
            .factor_nnz,
        all_finite_variances,
        nonnegative_variances,
    }
}

fn variance_quality_flags(result: &LinearPdeUqResult) -> (bool, bool) {
    let mut all_finite = true;
    let mut nonnegative = true;
    let mut observe = |value: f64| {
        all_finite &= value.is_finite();
        nonnegative &= value >= 0.0;
    };
    for value in result
        .prior_variance
        .iter()
        .chain(result.posterior_variance.iter())
        .chain(result.reduced_posterior_variance.iter())
    {
        observe(*value);
    }
    for input in &result.latent_inputs {
        for value in &input.variance {
            observe(*value);
        }
    }
    for variance in result.derived_variances.values() {
        for value in variance
            .prior_variance
            .iter()
            .chain(variance.posterior_variance.iter())
        {
            observe(*value);
        }
    }
    (all_finite, nonnegative)
}

fn latent_input_by_name<'a>(
    inputs: &'a [LinearPdeLatentInputPosterior],
    name: &str,
) -> Result<&'a LinearPdeLatentInputPosterior, String> {
    inputs
        .iter()
        .find(|input| input.name == name)
        .ok_or_else(|| format!("missing latent input `{name}`"))
}

fn source_mode_posterior_summary_from_stage(
    stage: &Team13SourceRecoveryStageResult,
    input_name: &str,
    prior_variance: f64,
    true_alphas: &[f64; TEAM13_COIL_MODE_COUNT],
) -> Result<Vec<Team13SourceModePosteriorSummary>, String> {
    let input = latent_input_by_name(&stage.solve.posterior.latent_inputs, input_name)?;
    if input.mean.len() != COIL_MODE_COUNT || input.variance.len() != COIL_MODE_COUNT {
        return Err(format!(
            "latent input `{input_name}` has dimension {}, expected {COIL_MODE_COUNT}",
            input.mean.len()
        ));
    }
    Ok(Team13CoilRegion::all()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, region)| {
            let posterior_mean = input.mean[index];
            let posterior_variance = input.variance[index];
            let recovery_error = posterior_mean - true_alphas[index];
            let posterior_std = posterior_variance.max(0.0).sqrt();
            Team13SourceModePosteriorSummary {
                mode_index: index,
                mode_name: region.name().to_string(),
                prior_mean: 1.0,
                prior_variance,
                true_alpha: true_alphas[index],
                posterior_mean,
                posterior_variance,
                variance_ratio: safe_ratio(posterior_variance, prior_variance),
                recovery_error,
                z_score: if posterior_std > 0.0 {
                    recovery_error.abs() / posterior_std
                } else {
                    f64::INFINITY
                },
            }
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn build_source_mode_fields(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    reluctivity: &InnerProductWeightClosure,
    boundary: &EssentialBoundarySpec,
    mode: Team13DomainMode,
    ampere_turns: f64,
) -> Vec<FeecVector> {
    Team13CoilRegion::all()
        .iter()
        .copied()
        .map(|region| {
            let source = assemble_unweighted_source(
                topology,
                metric,
                coords,
                &team13_current_density(mode, ampere_turns, Some(region)),
            );
            solve_nominal_a(topology, metric, coords, reluctivity, boundary, source)
        })
        .collect()
}

fn source_mode_sensor_sensitivities_from_fields(
    source_mode_fields: &[FeecVector],
    measurements: &[Team13LinearizedMeasurement],
) -> Result<Vec<Vec<f64>>, String> {
    if source_mode_fields.len() != COIL_MODE_COUNT {
        return Err(format!(
            "expected {COIL_MODE_COUNT} source-mode fields, got {}",
            source_mode_fields.len()
        ));
    }
    let measurement_rows = measurements
        .iter()
        .map(|measurement| triplet_to_sparse_row_operator(&measurement.spec.operator))
        .collect::<Result<Vec<_>, _>>()?;
    let mut sensitivities = vec![vec![0.0; COIL_MODE_COUNT]; measurements.len()];
    for (mode_index, mode_a) in source_mode_fields.iter().enumerate() {
        for (sensor_index, row) in measurement_rows.iter().enumerate() {
            let value = apply_operator_to_feec(row, mode_a)?[0];
            sensitivities[sensor_index][mode_index] = value;
        }
    }
    Ok(sensitivities)
}

fn weighted_sum_source_mode_fields(
    source_mode_fields: &[FeecVector],
    source_alphas: &[f64; TEAM13_COIL_MODE_COUNT],
) -> Result<FeecVector, String> {
    if source_mode_fields.len() != COIL_MODE_COUNT {
        return Err(format!(
            "expected {COIL_MODE_COUNT} source-mode fields, got {}",
            source_mode_fields.len()
        ));
    }
    let Some(first) = source_mode_fields.first() else {
        return Err("at least one source-mode field is required".to_string());
    };
    if source_mode_fields
        .iter()
        .any(|field| field.len() != first.len())
    {
        return Err("source-mode field lengths must match".to_string());
    }
    let mut sum = FeecVector::zeros(first.len());
    for (field, alpha) in source_mode_fields.iter().zip(source_alphas.iter()) {
        sum += field.scale(*alpha);
    }
    Ok(sum)
}

fn set_measurement_observations_from_mode_sensitivities(
    measurements: &mut [Team13LinearizedMeasurement],
    sensitivities: &[Vec<f64>],
    source_alphas: &[f64; TEAM13_COIL_MODE_COUNT],
) -> Result<(), String> {
    if measurements.len() != sensitivities.len() {
        return Err(format!(
            "measurement count {} must match sensitivity row count {}",
            measurements.len(),
            sensitivities.len()
        ));
    }
    for (measurement, row) in measurements.iter_mut().zip(sensitivities.iter()) {
        if row.len() != COIL_MODE_COUNT {
            return Err(format!(
                "source-mode sensitivity row has {} columns, expected {COIL_MODE_COUNT}",
                row.len()
            ));
        }
        measurement.spec.observations[0] = row
            .iter()
            .zip(source_alphas.iter())
            .map(|(sensitivity, alpha)| sensitivity * alpha)
            .sum::<f64>();
    }
    Ok(())
}

fn source_mode_posterior_summary_from_sensor_sensitivities(
    sensitivities: &[Vec<f64>],
    measurements: &[Team13LinearizedMeasurement],
    prior_variance: f64,
    sensor_variance: f64,
    true_alphas: &[f64; TEAM13_COIL_MODE_COUNT],
) -> Result<Vec<Team13SourceModePosteriorSummary>, String> {
    if sensitivities.len() != measurements.len() {
        return Err(format!(
            "source sensitivity row count {} must match measurement count {}",
            sensitivities.len(),
            measurements.len()
        ));
    }
    if !prior_variance.is_finite() || prior_variance <= 0.0 {
        return Err("source-mode prior variance must be finite and positive".to_string());
    }
    if !sensor_variance.is_finite() || sensor_variance <= 0.0 {
        return Err("source-mode sensor variance must be finite and positive".to_string());
    }

    let prior_precision = 1.0 / prior_variance;
    let inv_sensor_variance = 1.0 / sensor_variance;
    let mut precision = vec![vec![0.0; COIL_MODE_COUNT]; COIL_MODE_COUNT];
    let mut information = vec![prior_precision; COIL_MODE_COUNT];
    for mode_index in 0..COIL_MODE_COUNT {
        precision[mode_index][mode_index] = prior_precision;
    }
    for (row, measurement) in sensitivities.iter().zip(measurements.iter()) {
        if row.len() != COIL_MODE_COUNT {
            return Err(format!(
                "source-mode sensitivity row has {} columns, expected {COIL_MODE_COUNT}",
                row.len()
            ));
        }
        let observed = measurement.spec.observations[0];
        for i in 0..COIL_MODE_COUNT {
            information[i] += row[i] * observed * inv_sensor_variance;
            for j in 0..COIL_MODE_COUNT {
                precision[i][j] += row[i] * row[j] * inv_sensor_variance;
            }
        }
    }
    let covariance = invert_dense_matrix(&precision)?;
    let posterior_mean = dense_matvec(&covariance, &information)?;
    let modes = Team13CoilRegion::all();
    Ok((0..COIL_MODE_COUNT)
        .map(|index| {
            let posterior_variance = covariance[index][index].max(0.0);
            let recovery_error = posterior_mean[index] - true_alphas[index];
            let posterior_std = posterior_variance.sqrt();
            let z_score = if posterior_std <= EPS {
                if recovery_error.abs() <= EPS {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                recovery_error.abs() / posterior_std
            };
            Team13SourceModePosteriorSummary {
                mode_index: index,
                mode_name: modes[index].name().to_string(),
                prior_mean: 1.0,
                prior_variance,
                true_alpha: true_alphas[index],
                posterior_mean: posterior_mean[index],
                posterior_variance,
                variance_ratio: safe_ratio(posterior_variance, prior_variance),
                recovery_error,
                z_score,
            }
        })
        .collect())
}

fn source_mode_observability_summary(
    sensitivities: &[Vec<f64>],
    source_modes: &[Team13SourceModePosteriorSummary],
) -> Result<Vec<Team13SourceModeObservabilitySummary>, String> {
    let modes = Team13CoilRegion::all();
    Ok((0..COIL_MODE_COUNT)
        .map(|index| {
            let sensor_sensitivity_norm = sensitivities
                .iter()
                .map(|row| row.get(index).copied().unwrap_or(0.0).powi(2))
                .sum::<f64>()
                .sqrt();
            let source = &source_modes[index];
            let variance_ratio = safe_ratio(source.posterior_variance, source.prior_variance);
            let posterior_shrinkage = (1.0 - variance_ratio).max(0.0);
            Team13SourceModeObservabilitySummary {
                mode_index: index,
                mode_name: modes[index].name().to_string(),
                sensor_sensitivity_norm,
                posterior_shrinkage,
                identifiable: source.z_score <= 2.0
                    && posterior_shrinkage >= 1.0 - EIGHT_MODE_STRONG_SHRINKAGE_VARIANCE_RATIO_GATE
                    && source.recovery_error.abs()
                        <= EIGHT_MODE_SOURCE_ERROR_PRIOR_STD_FRACTION_GATE
                            * source.prior_variance.sqrt(),
            }
        })
        .collect())
}

fn replacement_decision(
    source_modes: &[Team13SourceModePosteriorSummary],
    fixed_source_sensor_rmse: f64,
    eight_mode_sensor_rmse: f64,
    fixed_source_b_vector_relative_l2_error: f64,
    eight_mode_b_vector_relative_l2_error: f64,
) -> Team13ReplacementDecision {
    let all_finite = source_modes.iter().all(|mode| {
        mode.posterior_mean.is_finite()
            && mode.posterior_variance.is_finite()
            && mode.true_alpha.is_finite()
            && mode.z_score.is_finite()
    });
    let variances_shrink = source_modes.iter().all(|mode| {
        mode.posterior_variance >= 0.0 && mode.posterior_variance < mode.prior_variance
    });
    let modes_within_two_sigma = source_modes
        .iter()
        .filter(|mode| mode.z_score <= 2.0)
        .count();
    let modes_beyond_three_sigma = source_modes
        .iter()
        .filter(|mode| mode.z_score > 3.0)
        .count();
    let modes_within_half_prior_std = source_modes
        .iter()
        .filter(|mode| {
            mode.prior_variance.is_finite()
                && mode.prior_variance >= 0.0
                && mode.recovery_error.abs()
                    <= EIGHT_MODE_SOURCE_ERROR_PRIOR_STD_FRACTION_GATE * mode.prior_variance.sqrt()
        })
        .count();
    let modes_with_strong_variance_shrinkage = source_modes
        .iter()
        .filter(|mode| {
            safe_ratio(mode.posterior_variance, mode.prior_variance)
                <= EIGHT_MODE_STRONG_SHRINKAGE_VARIANCE_RATIO_GATE
        })
        .count();
    let sensor_rmse_improves = fixed_source_sensor_rmse.is_finite()
        && eight_mode_sensor_rmse.is_finite()
        && eight_mode_sensor_rmse < fixed_source_sensor_rmse;
    let source_recovery_convincing =
        modes_within_half_prior_std >= 6 && modes_with_strong_variance_shrinkage >= 6;
    let field_recovery_convincing = fixed_source_b_vector_relative_l2_error.is_finite()
        && eight_mode_b_vector_relative_l2_error.is_finite()
        && eight_mode_b_vector_relative_l2_error < fixed_source_b_vector_relative_l2_error
        && eight_mode_b_vector_relative_l2_error <= EIGHT_MODE_FIELD_RELATIVE_ERROR_GATE;
    let technical_pass = all_finite
        && variances_shrink
        && modes_within_two_sigma >= 6
        && modes_beyond_three_sigma == 0
        && sensor_rmse_improves
        && source_recovery_convincing
        && field_recovery_convincing;
    Team13ReplacementDecision {
        technical_pass,
        recommendation: if technical_pass {
            "use_eight_mode".to_string()
        } else {
            "fallback_scalar".to_string()
        },
        all_finite,
        variances_shrink,
        source_recovery_convincing,
        field_recovery_convincing,
        modes_within_two_sigma,
        modes_beyond_three_sigma,
        modes_within_half_prior_std,
        modes_with_strong_variance_shrinkage,
        sensor_rmse_improves,
        fixed_source_sensor_rmse,
        eight_mode_sensor_rmse,
        fixed_source_b_vector_relative_l2_error,
        eight_mode_b_vector_relative_l2_error,
    }
}

fn source_mode_sensor_rmse(
    sensitivities: &[Vec<f64>],
    source_means: &[f64],
    measurements: &[Team13LinearizedMeasurement],
) -> Result<f64, String> {
    if source_means.len() != COIL_MODE_COUNT {
        return Err(format!(
            "source mean length {} must match {COIL_MODE_COUNT}",
            source_means.len()
        ));
    }
    if sensitivities.len() != measurements.len() {
        return Err(format!(
            "source sensitivity row count {} must match measurement count {}",
            sensitivities.len(),
            measurements.len()
        ));
    }
    if measurements.is_empty() {
        return Ok(0.0);
    }
    let mut sum_sq = 0.0;
    for (row, measurement) in sensitivities.iter().zip(measurements.iter()) {
        if row.len() != COIL_MODE_COUNT {
            return Err(format!(
                "source-mode sensitivity row has {} columns, expected {COIL_MODE_COUNT}",
                row.len()
            ));
        }
        let prediction = row
            .iter()
            .zip(source_means.iter())
            .map(|(sensitivity, mean)| sensitivity * mean)
            .sum::<f64>();
        let residual = prediction - measurement.spec.observations[0];
        sum_sq += residual * residual;
    }
    Ok((sum_sq / measurements.len() as f64).sqrt())
}

fn invert_dense_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = matrix.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if matrix.iter().any(|row| row.len() != n) {
        return Err("dense inverse requires a square matrix".to_string());
    }
    let mut augmented = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][n + i] = 1.0;
    }
    for pivot_index in 0..n {
        let mut pivot_row = pivot_index;
        let mut pivot_abs = augmented[pivot_index][pivot_index].abs();
        for (row_index, row) in augmented.iter().enumerate().skip(pivot_index + 1) {
            let candidate = row[pivot_index].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row_index;
            }
        }
        if pivot_abs <= 1e-18 {
            return Err("dense inverse failed: singular source posterior precision".to_string());
        }
        if pivot_row != pivot_index {
            augmented.swap(pivot_index, pivot_row);
        }
        let pivot = augmented[pivot_index][pivot_index];
        for col in 0..2 * n {
            augmented[pivot_index][col] /= pivot;
        }
        for row_index in 0..n {
            if row_index == pivot_index {
                continue;
            }
            let factor = augmented[row_index][pivot_index];
            if factor == 0.0 {
                continue;
            }
            for col in 0..2 * n {
                augmented[row_index][col] -= factor * augmented[pivot_index][col];
            }
        }
    }
    Ok(augmented.into_iter().map(|row| row[n..].to_vec()).collect())
}

fn dense_matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>, String> {
    if matrix.iter().any(|row| row.len() != vector.len()) {
        return Err("dense matvec dimension mismatch".to_string());
    }
    Ok(matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
        .collect())
}

fn sensor_rmse_from_reports(reports: &[Team13SensorReport]) -> f64 {
    if reports.is_empty() {
        return 0.0;
    }
    (reports
        .iter()
        .map(|report| report.residual * report.residual)
        .sum::<f64>()
        / reports.len() as f64)
        .sqrt()
}

fn l2_norm<'a>(values: impl IntoIterator<Item = &'a f64>) -> f64 {
    values
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn l2_distance(lhs: &[f64], rhs: &[f64]) -> Result<f64, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "distance dimension mismatch: lhs has {}, rhs has {}",
            lhs.len(),
            rhs.len()
        ));
    }
    Ok(lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt())
}

fn relative_l2_distance(lhs: &[f64], rhs: &[f64]) -> Result<f64, String> {
    Ok(safe_ratio(l2_distance(lhs, rhs)?, l2_norm(rhs)).abs())
}

fn field_recovery_metrics(
    operators: &Team13Operators,
    reference_a: &FeecVector,
    posterior_a: &FeecVector,
    state_active_mask: &FeecVector,
) -> Result<Team13FieldRecoveryMetrics, String> {
    let (a_active_rmse, a_active_relative_l2_error) =
        scalar_error_metrics(reference_a, posterior_a, Some(state_active_mask))?;
    let reference_b = apply_operator_to_feec(&operators.b_cochain, reference_a)?;
    let posterior_b = apply_operator_to_feec(&operators.b_cochain, posterior_a)?;
    let (b_cochain_rmse, b_cochain_relative_l2_error) =
        scalar_error_metrics(&reference_b, &posterior_b, None)?;
    let reference_b_components = operators
        .b_components
        .iter()
        .map(|operator| apply_operator_to_feec(operator, reference_a))
        .collect::<Result<Vec<_>, _>>()?;
    let posterior_b_components = operators
        .b_components
        .iter()
        .map(|operator| apply_operator_to_feec(operator, posterior_a))
        .collect::<Result<Vec<_>, _>>()?;
    let reference_b_vectors = vectors_from_components(&reference_b_components)?;
    let posterior_b_vectors = vectors_from_components(&posterior_b_components)?;
    let (b_vector_rmse, b_vector_relative_l2_error) =
        vector_error_metrics(&reference_b_vectors, &posterior_b_vectors)?;
    Ok(Team13FieldRecoveryMetrics {
        a_active_rmse,
        a_active_relative_l2_error,
        b_cochain_rmse,
        b_cochain_relative_l2_error,
        b_vector_rmse,
        b_vector_relative_l2_error,
    })
}

fn scalar_error_metrics(
    reference: &FeecVector,
    estimate: &FeecVector,
    mask: Option<&FeecVector>,
) -> Result<(f64, f64), String> {
    if reference.len() != estimate.len() {
        return Err(format!(
            "field error dimension mismatch: reference has {}, estimate has {}",
            reference.len(),
            estimate.len()
        ));
    }
    if let Some(mask) = mask {
        if mask.len() != reference.len() {
            return Err(format!(
                "field error mask dimension mismatch: mask has {}, field has {}",
                mask.len(),
                reference.len()
            ));
        }
    }
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    let mut count = 0usize;
    for index in 0..reference.len() {
        if matches!(mask, Some(mask) if mask[index] <= 0.0) {
            continue;
        }
        let error = estimate[index] - reference[index];
        error_sq += error * error;
        reference_sq += reference[index] * reference[index];
        count += 1;
    }
    Ok(rmse_and_relative_l2(error_sq, reference_sq, count))
}

fn vector_error_metrics(
    reference: &[[f64; 3]],
    estimate: &[[f64; 3]],
) -> Result<(f64, f64), String> {
    if reference.len() != estimate.len() {
        return Err(format!(
            "vector field error dimension mismatch: reference has {}, estimate has {}",
            reference.len(),
            estimate.len()
        ));
    }
    let mut error_sq = 0.0;
    let mut reference_sq = 0.0;
    for (reference, estimate) in reference.iter().zip(estimate.iter()) {
        for component in 0..3 {
            let error = estimate[component] - reference[component];
            error_sq += error * error;
            reference_sq += reference[component] * reference[component];
        }
    }
    Ok(rmse_and_relative_l2(
        error_sq,
        reference_sq,
        reference.len(),
    ))
}

fn rmse_and_relative_l2(error_sq: f64, reference_sq: f64, count: usize) -> (f64, f64) {
    let rmse = if count == 0 {
        0.0
    } else {
        (error_sq / count as f64).sqrt()
    };
    (rmse, safe_ratio(error_sq.sqrt(), reference_sq.sqrt()).abs())
}

fn finite_mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn masked_finite_mean(values_and_mask: impl Iterator<Item = (f64, f64)>) -> f64 {
    finite_mean(values_and_mask.filter_map(
        |(value, mask)| {
            if mask > 0.0 {
                Some(value)
            } else {
                None
            }
        },
    ))
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= EPS {
        if numerator.abs() <= EPS {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        numerator / denominator
    }
}

fn write_team13_source_recovery_outputs(
    output_dir: &PathBuf,
    topology: &Complex,
    coords: &MeshCoords,
    result: &Team13SourceRecoveryResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create source-recovery output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    for stage in &result.stages {
        write_team13_outputs(
            &output_dir.join(&stage.summary.name),
            topology,
            coords,
            &stage.solve,
        )?;
    }
    write_source_recovery_stage_summary(output_dir, result)?;
    write_field_prior_comparison_summary(output_dir, result)?;
    write_source_posterior_summary(output_dir, "source_posterior.csv", &result.source_posterior)?;
    write_source_posterior_summary(
        output_dir,
        "source_scaling_proxy.csv",
        &result.source_scaling_proxy,
    )?;
    write_source_posterior_summary(
        output_dir,
        "source_posterior_baseline.csv",
        &result.baseline_source_posterior,
    )?;
    write_source_posterior_summary(
        output_dir,
        "source_posterior_fluctuation.csv",
        &result.fluctuation_source_posterior,
    )?;
    write_sensor_uncertainty_summary(output_dir, result)?;
    write_flux_uncertainty_summary(output_dir, result)?;
    write_period_diagnostic_summary(output_dir, result)?;
    if let Some(eight_mode) = &result.eight_mode {
        write_team13_eight_mode_outputs(
            &output_dir.join("eight_mode"),
            topology,
            coords,
            eight_mode,
        )?;
    }
    Ok(())
}

fn write_source_recovery_stage_summary(
    output_dir: &Path,
    result: &Team13SourceRecoveryResult,
) -> Result<(), String> {
    let mut csv = "stage,latent_dimension,pde_residual_norm,sensor_rmse,field_reference,a_active_rmse,a_active_relative_l2_error,b_cochain_rmse,b_cochain_relative_l2_error,b_vector_rmse,b_vector_relative_l2_error,a_variance_ratio_mean,b_variance_ratio_mean,b_vector_trace_variance_ratio_mean\n".to_string();
    for stage in &result.stages {
        let summary = &stage.summary;
        csv.push_str(&format!(
            "{},{},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n",
            summary.name,
            summary.latent_dimension,
            summary.pde_residual_norm,
            summary.sensor_rmse,
            summary.field_reference,
            summary.a_active_rmse,
            summary.a_active_relative_l2_error,
            summary.b_cochain_rmse,
            summary.b_cochain_relative_l2_error,
            summary.b_vector_rmse,
            summary.b_vector_relative_l2_error,
            summary.a_variance_ratio_mean,
            summary.b_variance_ratio_mean,
            summary.b_vector_trace_variance_ratio_mean,
        ));
    }
    fs::write(output_dir.join("stage_summary.csv"), csv)
        .map_err(|err| format!("failed to write stage summary: {err}"))
}

fn write_field_prior_comparison_summary(
    output_dir: &Path,
    result: &Team13SourceRecoveryResult,
) -> Result<(), String> {
    let mut csv = "prior,stage,source_prior_mean,source_prior_variance,source_posterior_mean,source_posterior_variance,source_variance_ratio,true_alpha,source_recovery_error,sensor_rmse,a_active_relative_l2_error,b_cochain_relative_l2_error,b_vector_relative_l2_error,a_variance_ratio_mean,b_variance_ratio_mean,b_vector_trace_variance_ratio_mean,prior_factor_nnz,posterior_factor_nnz,all_finite_variances,nonnegative_variances\n".to_string();
    for comparison in &result.field_prior_comparisons {
        let source = &comparison.source_posterior;
        csv.push_str(&format!(
            "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{},{},{}\n",
            comparison.prior_kind.as_str(),
            comparison.stage_name,
            source.prior_mean,
            source.prior_variance,
            source.posterior_mean,
            source.posterior_variance,
            source.variance_ratio,
            source.true_alpha,
            source.recovery_error,
            comparison.sensor_rmse,
            comparison.a_active_relative_l2_error,
            comparison.b_cochain_relative_l2_error,
            comparison.b_vector_relative_l2_error,
            comparison.a_variance_ratio_mean,
            comparison.b_variance_ratio_mean,
            comparison.b_vector_trace_variance_ratio_mean,
            comparison.prior_factor_nnz,
            comparison.posterior_factor_nnz,
            comparison.all_finite_variances,
            comparison.nonnegative_variances,
        ));
    }
    fs::write(output_dir.join("prior_comparison.csv"), csv)
        .map_err(|err| format!("failed to write field prior comparison summary: {err}"))
}

fn write_source_posterior_summary(
    output_dir: &Path,
    filename: &str,
    source: &Team13SourcePosteriorSummary,
) -> Result<(), String> {
    let csv = format!(
        "name,prior_mean,prior_variance,posterior_mean,posterior_variance,variance_ratio,true_alpha,recovery_error\n{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n",
        SOURCE_ALPHA_INPUT_NAME,
        source.prior_mean,
        source.prior_variance,
        source.posterior_mean,
        source.posterior_variance,
        source.variance_ratio,
        source.true_alpha,
        source.recovery_error,
    );
    fs::write(output_dir.join(filename), csv)
        .map_err(|err| format!("failed to write source posterior summary: {err}"))
}

fn write_sensor_uncertainty_summary(
    output_dir: &Path,
    result: &Team13SourceRecoveryResult,
) -> Result<(), String> {
    let mut csv =
        "stage,name,observed,prediction,residual,prior_variance,posterior_variance\n".to_string();
    for stage in &result.stages {
        for report in &stage.sensor_uncertainty {
            csv.push_str(&format!(
                "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n",
                report.stage,
                report.name,
                report.observed,
                report.prediction,
                report.residual,
                report.prior_variance,
                report.posterior_variance,
            ));
        }
    }
    fs::write(output_dir.join("sensor_uncertainty.csv"), csv)
        .map_err(|err| format!("failed to write sensor uncertainty summary: {err}"))
}

fn write_flux_uncertainty_summary(
    output_dir: &Path,
    result: &Team13SourceRecoveryResult,
) -> Result<(), String> {
    let mut csv = "stage,name,area,observed_flux,predicted_flux,residual_flux,prior_flux_variance,posterior_flux_variance\n".to_string();
    for stage in &result.stages {
        for report in &stage.sensor_uncertainty {
            let area_sq = report.area * report.area;
            csv.push_str(&format!(
                "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n",
                report.stage,
                report.name,
                report.area,
                report.observed * report.area,
                report.prediction * report.area,
                report.residual * report.area,
                report.prior_variance * area_sq,
                report.posterior_variance * area_sq,
            ));
        }
    }
    fs::write(output_dir.join("flux_uncertainty.csv"), csv)
        .map_err(|err| format!("failed to write flux uncertainty summary: {err}"))
}

fn write_period_diagnostic_summary(
    output_dir: &Path,
    result: &Team13SourceRecoveryResult,
) -> Result<(), String> {
    let mut csv =
        "stage,name,truth,prediction,residual,prior_variance,posterior_variance,variance_ratio\n"
            .to_string();
    for stage in &result.stages {
        for report in &stage.period_diagnostics {
            csv.push_str(&format!(
                "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n",
                report.stage,
                report.name,
                report.truth,
                report.prediction,
                report.residual,
                report.prior_variance,
                report.posterior_variance,
                report.variance_ratio,
            ));
        }
    }
    fs::write(output_dir.join("period_diagnostic.csv"), csv)
        .map_err(|err| format!("failed to write period diagnostic summary: {err}"))
}

fn write_team13_eight_mode_outputs(
    output_dir: &Path,
    topology: &Complex,
    coords: &MeshCoords,
    result: &Team13EightModeRecoveryResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create eight-mode output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    write_team13_outputs(
        &output_dir.join(&result.stage.summary.name),
        topology,
        coords,
        &result.stage.solve,
    )?;
    if let Some(stage) = &result.fluctuation_stage {
        write_team13_outputs(
            &output_dir.join(&stage.summary.name),
            topology,
            coords,
            &stage.solve,
        )?;
    }
    let mut stages = vec![result.stage.clone()];
    if let Some(stage) = &result.fluctuation_stage {
        stages.push(stage.clone());
    }
    let placeholder_source = Team13SourcePosteriorSummary {
        prior_mean: 1.0,
        prior_variance: 0.0,
        posterior_mean: 0.0,
        posterior_variance: 0.0,
        variance_ratio: 0.0,
        true_alpha: 0.0,
        recovery_error: 0.0,
    };
    let stage_result = Team13SourceRecoveryResult {
        stages,
        field_prior_comparisons: Vec::new(),
        source_posterior: placeholder_source.clone(),
        source_scaling_proxy: placeholder_source.clone(),
        baseline_source_posterior: placeholder_source.clone(),
        fluctuation_source_posterior: placeholder_source,
        eight_mode: None,
    };
    write_source_recovery_stage_summary(output_dir, &stage_result)?;
    write_period_diagnostic_summary(output_dir, &stage_result)?;
    write_source_mode_posterior_summary(output_dir, &result.source_modes)?;
    write_source_mode_posterior_summary_named(
        output_dir,
        "source_mode_posterior_fluctuation.csv",
        &result.fluctuation_source_modes,
    )?;
    write_source_mode_observability_summary(output_dir, &result.observability)?;
    write_replacement_decision(output_dir, &result.decision)?;
    Ok(())
}

fn write_source_mode_posterior_summary(
    output_dir: &Path,
    source_modes: &[Team13SourceModePosteriorSummary],
) -> Result<(), String> {
    write_source_mode_posterior_summary_named(output_dir, "source_mode_posterior.csv", source_modes)
}

fn write_source_mode_posterior_summary_named(
    output_dir: &Path,
    filename: &str,
    source_modes: &[Team13SourceModePosteriorSummary],
) -> Result<(), String> {
    let mut csv = "mode_index,mode_name,prior_mean,prior_variance,true_alpha,posterior_mean,posterior_variance,variance_ratio,recovery_error,z_score\n".to_string();
    for mode in source_modes {
        csv.push_str(&format!(
            "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n",
            mode.mode_index,
            mode.mode_name,
            mode.prior_mean,
            mode.prior_variance,
            mode.true_alpha,
            mode.posterior_mean,
            mode.posterior_variance,
            mode.variance_ratio,
            mode.recovery_error,
            mode.z_score,
        ));
    }
    fs::write(output_dir.join(filename), csv)
        .map_err(|err| format!("failed to write source-mode posterior summary: {err}"))
}

fn write_source_mode_observability_summary(
    output_dir: &Path,
    observability: &[Team13SourceModeObservabilitySummary],
) -> Result<(), String> {
    let mut csv = "mode_index,mode_name,sensor_sensitivity_norm,posterior_shrinkage,identifiable\n"
        .to_string();
    for mode in observability {
        csv.push_str(&format!(
            "{},{},{:.12e},{:.12e},{}\n",
            mode.mode_index,
            mode.mode_name,
            mode.sensor_sensitivity_norm,
            mode.posterior_shrinkage,
            mode.identifiable,
        ));
    }
    fs::write(output_dir.join("source_mode_observability.csv"), csv)
        .map_err(|err| format!("failed to write source-mode observability summary: {err}"))
}

fn write_replacement_decision(
    output_dir: &Path,
    decision: &Team13ReplacementDecision,
) -> Result<(), String> {
    let json = format!(
        concat!(
            "{{\n",
            "  \"technical_pass\": {},\n",
            "  \"recommendation\": \"{}\",\n",
            "  \"all_finite\": {},\n",
            "  \"variances_shrink\": {},\n",
            "  \"source_recovery_convincing\": {},\n",
            "  \"field_recovery_convincing\": {},\n",
            "  \"modes_within_two_sigma\": {},\n",
            "  \"modes_beyond_three_sigma\": {},\n",
            "  \"modes_within_half_prior_std\": {},\n",
            "  \"modes_with_strong_variance_shrinkage\": {},\n",
            "  \"sensor_rmse_improves\": {},\n",
            "  \"fixed_source_sensor_rmse\": {:.12e},\n",
            "  \"eight_mode_sensor_rmse\": {:.12e},\n",
            "  \"fixed_source_b_vector_relative_l2_error\": {:.12e},\n",
            "  \"eight_mode_b_vector_relative_l2_error\": {:.12e}\n",
            "}}\n"
        ),
        decision.technical_pass,
        decision.recommendation,
        decision.all_finite,
        decision.variances_shrink,
        decision.source_recovery_convincing,
        decision.field_recovery_convincing,
        decision.modes_within_two_sigma,
        decision.modes_beyond_three_sigma,
        decision.modes_within_half_prior_std,
        decision.modes_with_strong_variance_shrinkage,
        decision.sensor_rmse_improves,
        decision.fixed_source_sensor_rmse,
        decision.eight_mode_sensor_rmse,
        decision.fixed_source_b_vector_relative_l2_error,
        decision.eight_mode_b_vector_relative_l2_error,
    );
    fs::write(output_dir.join("replacement_decision.json"), json)
        .map_err(|err| format!("failed to write eight-mode replacement decision: {err}"))
}

fn load_surface_observation_overrides(
    path: Option<&Path>,
) -> Result<Option<BTreeMap<String, f64>>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let content = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read TEAM 13 observation CSV `{}`: {err}",
            path.display()
        )
    })?;
    let mut observations = BTreeMap::new();
    let mut header = None::<(usize, usize)>;
    for (line_index, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line
            .split(',')
            .map(|field| field.trim().trim_matches('"').to_string())
            .collect::<Vec<_>>();
        if fields.len() < 2 {
            return Err(format!(
                "TEAM 13 observation CSV `{}` line {} has fewer than two columns",
                path.display(),
                line_index + 1
            ));
        }
        if header.is_none() {
            let lower = fields
                .iter()
                .map(|field| field.to_ascii_lowercase())
                .collect::<Vec<_>>();
            if let Some(name_column) = lower.iter().position(|field| field == "name") {
                let observation_column = lower
                    .iter()
                    .position(|field| field == "observed" || field == "observation")
                    .ok_or_else(|| {
                        format!(
                            "TEAM 13 observation CSV `{}` header must contain an observed column",
                            path.display()
                        )
                    })?;
                header = Some((name_column, observation_column));
                continue;
            }
            header = Some((0, 1));
        }
        let (name_column, observation_column) = header.expect("header should be initialized");
        if name_column >= fields.len() || observation_column >= fields.len() {
            return Err(format!(
                "TEAM 13 observation CSV `{}` line {} is missing the configured name/observed columns",
                path.display(),
                line_index + 1
            ));
        }
        let name = fields[name_column].clone();
        if name.is_empty() {
            return Err(format!(
                "TEAM 13 observation CSV `{}` line {} has an empty name",
                path.display(),
                line_index + 1
            ));
        }
        let observed = fields[observation_column].parse::<f64>().map_err(|err| {
            format!(
                "TEAM 13 observation CSV `{}` line {} has invalid observed value `{}`: {err}",
                path.display(),
                line_index + 1,
                fields[observation_column]
            )
        })?;
        if !observed.is_finite() {
            return Err(format!(
                "TEAM 13 observation CSV `{}` line {} has non-finite observed value",
                path.display(),
                line_index + 1
            ));
        }
        if observations.insert(name.clone(), observed).is_some() {
            return Err(format!(
                "TEAM 13 observation CSV `{}` contains duplicate observation `{name}`",
                path.display()
            ));
        }
    }
    if observations.is_empty() {
        return Err(format!(
            "TEAM 13 observation CSV `{}` did not contain any observations",
            path.display()
        ));
    }
    Ok(Some(observations))
}

fn reluctivity_weight() -> InnerProductWeightClosure {
    InnerProductWeightClosure::new(reluctivity_at)
}

fn build_outer_boundary(
    topology: &Complex,
    coords: &MeshCoords,
    mode: Team13DomainMode,
) -> EssentialBoundarySpec {
    let state_dofs =
        sorted_boundary_dofs(topology, coords, 1, |point| is_outer_boundary(point, mode));
    let auxiliary_dofs =
        sorted_boundary_dofs(topology, coords, 0, |point| is_outer_boundary(point, mode));
    EssentialBoundarySpec {
        state: state_dofs
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
        auxiliary: auxiliary_dofs
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
    }
}

fn is_outer_boundary(point: CoordRef<'_>, mode: Team13DomainMode) -> bool {
    let tol = 1e-8;
    near(point[0].abs(), 0.25, tol)
        || near(point[1].abs(), 0.25, tol)
        || near(point[2], 0.25, tol)
        || (mode == Team13DomainMode::Full && near(point[2], -0.25, tol))
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

fn build_source_mode_operator(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    galmats: &MixedGalmats,
    boundary: &EssentialBoundarySpec,
    mode: Team13DomainMode,
    ampere_turns: f64,
) -> Result<SparseTripletMatrix, String> {
    let mut columns = Vec::with_capacity(COIL_MODE_COUNT);
    for region in Team13CoilRegion::all() {
        let source = team13_current_density(mode, ampere_turns, Some(region));
        let rhs = assemble_unweighted_source(topology, metric, coords, &source);
        let reduced = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
            galmats,
            boundary,
            &FeecVector::zeros(galmats.sigma_len()),
            &rhs,
        )?;
        columns.push(reduced.scale(-1.0));
    }
    Ok(columns_to_sparse_matrix(&columns))
}

fn assemble_unweighted_source(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    source: &DiffFormClosure,
) -> FeecVector {
    assemble_galvec(topology, metric, SourceElVec::new(source, coords, None))
}

fn team13_current_density(
    mode: Team13DomainMode,
    ampere_turns: f64,
    region_filter: Option<Team13CoilRegion>,
) -> DiffFormClosure {
    DiffFormClosure::one_form(
        move |point| {
            let value = team13_current_vector(point, mode, ampere_turns, region_filter);
            FeecVector::from_column_slice(&value)
        },
        3,
    )
}

fn solve_nominal_a(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    reluctivity: &InnerProductWeightClosure,
    boundary: &EssentialBoundarySpec,
    source: FeecVector,
) -> FeecVector {
    let state_fixed = boundary
        .state
        .iter()
        .map(|entry| entry.index)
        .collect::<HashSet<_>>();
    let aux_fixed = boundary
        .auxiliary
        .iter()
        .map(|entry| entry.index)
        .collect::<HashSet<_>>();
    let state_predicate = |sidx: KSimplexIdx| state_fixed.contains(&sidx);
    let aux_predicate = |sidx: KSimplexIdx| aux_fixed.contains(&sidx);
    let zero = |_sidx: KSimplexIdx| 0.0;
    let (_, a, _) = hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
        topology,
        metric,
        None,
        source,
        1,
        0,
        coords,
        None,
        reluctivity,
        &state_predicate,
        &zero,
        &aux_predicate,
        &zero,
    );
    a.coeffs
}

fn build_weighted_whittle_prior(
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    mean: &FeecVector,
) -> Result<GaussianPriorSpec, String> {
    build_hodge_matern_prior_from_reduced_system(system, mean)
}

fn build_team13_field_prior(
    kind: Team13FieldPriorKind,
    unweighted_system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    topology: &Complex,
    coords: &MeshCoords,
    mean: &FeecVector,
) -> Result<GaussianPriorSpec, String> {
    build_team13_field_prior_with_matern_params(
        kind,
        unweighted_system,
        topology,
        coords,
        mean,
        1.0,
        1.0,
    )
}

fn build_team13_field_prior_with_matern_params(
    kind: Team13FieldPriorKind,
    unweighted_system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    topology: &Complex,
    coords: &MeshCoords,
    mean: &FeecVector,
    kappa: f64,
    tau: f64,
) -> Result<GaussianPriorSpec, String> {
    match kind {
        Team13FieldPriorKind::UnweightedHodgeMatern => {
            build_hodge_matern_prior_from_reduced_system_with_params(
                unweighted_system,
                mean,
                kappa,
                tau,
            )
        }
        Team13FieldPriorKind::SplitGraphHodgeMatern => {
            build_split_graph_hodge_matern_prior_with_params(
                unweighted_system,
                topology,
                coords,
                mean,
                kappa,
                tau,
            )
        }
    }
}

fn build_hodge_matern_prior_from_reduced_system(
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    mean: &FeecVector,
) -> Result<GaussianPriorSpec, String> {
    build_hodge_matern_prior_from_reduced_system_with_params(system, mean, 1.0, 1.0)
}

fn build_hodge_matern_prior_from_reduced_system_with_params(
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    mean: &FeecVector,
    kappa: f64,
    tau: f64,
) -> Result<GaussianPriorSpec, String> {
    if mean.len() != system.state_dimension() {
        return Err(format!(
            "state prior mean length {} must match reduced state dimension {}",
            mean.len(),
            system.state_dimension()
        ));
    }
    if !kappa.is_finite() || kappa < 0.0 {
        return Err("state prior kappa must be finite and nonnegative".to_string());
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err("state prior tau must be finite and positive".to_string());
    }
    let mass_inverse = system
        .state_mass_inverse
        .as_ref()
        .ok_or_else(|| "1-form system is missing state_mass_inverse".to_string())?
        .clone();
    let hodge = HodgeLaplacian1Form {
        mass_u: system.state_mass.clone(),
        laplacian: system.operator.clone(),
    };
    let precision = build_matern_precision_1form_with_mass_inverse_for_alpha(
        &hodge,
        &mass_inverse,
        MaternAlpha::Two,
        kappa,
        tau,
    );
    Ok(GaussianPriorSpec {
        mean: mean.iter().copied().collect(),
        precision: csr_to_triplet(&stabilize_precision(symmetrize_feec_csr(&precision))),
    })
}

fn build_weak_ridge_prior(mean: &[f64], ridge_precision: f64) -> Result<GaussianPriorSpec, String> {
    if !ridge_precision.is_finite() || ridge_precision <= 0.0 {
        return Err(format!(
            "weak-ridge prior precision must be finite and positive, got {ridge_precision:.6e}"
        ));
    }
    Ok(GaussianPriorSpec {
        mean: mean.to_vec(),
        precision: diagonal_precision(mean.len(), ridge_precision),
    })
}

fn add_diagonal_shift_to_gaussian_prior(
    mut prior: GaussianPriorSpec,
    shift: f64,
) -> Result<GaussianPriorSpec, String> {
    if !shift.is_finite() || shift < 0.0 {
        return Err("prior diagonal shift must be finite and nonnegative".to_string());
    }
    if shift == 0.0 {
        return Ok(prior);
    }
    if prior.precision.nrows() != prior.precision.ncols() {
        return Err(format!(
            "cannot diagonal-shift non-square prior precision {}x{}",
            prior.precision.nrows(),
            prior.precision.ncols()
        ));
    }
    if prior.mean.len() != prior.precision.nrows() {
        return Err(format!(
            "prior mean length {} does not match precision dimension {}",
            prior.mean.len(),
            prior.precision.nrows()
        ));
    }
    let precision = core_triplet_to_feec_csr(&prior.precision);
    prior.precision = csr_to_triplet(&add_diagonal_shift(&precision, shift));
    Ok(prior)
}

fn build_split_graph_hodge_matern_prior_with_params(
    system: &formoniq::problems::reduced_linear::ReducedLinearPdeAssembly,
    topology: &Complex,
    coords: &MeshCoords,
    mean: &FeecVector,
    kappa: f64,
    tau: f64,
) -> Result<GaussianPriorSpec, String> {
    if mean.len() != system.state_dimension() {
        return Err(format!(
            "split-graph prior mean length {} must match reduced state dimension {}",
            mean.len(),
            system.state_dimension()
        ));
    }
    if !kappa.is_finite() || kappa < 0.0 {
        return Err("split-graph prior kappa must be finite and nonnegative".to_string());
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err("split-graph prior tau must be finite and positive".to_string());
    }
    let mass_inverse = system
        .state_mass_inverse
        .as_ref()
        .ok_or_else(|| "split-graph 1-form system is missing state_mass_inverse".to_string())?
        .clone();
    let hodge = HodgeLaplacian1Form {
        mass_u: system.state_mass.clone(),
        laplacian: system.operator.clone(),
    };
    let groups = team13_material_split_graph_groups(topology, coords, &system.layout)?;
    let precision = build_split_graph_matern_precision_1form_with_mass_inverse_for_alpha(
        &hodge,
        &mass_inverse,
        &groups,
        MaternAlpha::Two,
        kappa,
        tau,
    )?;
    Ok(GaussianPriorSpec {
        mean: mean.iter().copied().collect(),
        precision: csr_to_triplet(&stabilize_precision(symmetrize_feec_csr(&precision))),
    })
}

fn team13_material_split_graph_groups(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
) -> Result<Vec<Vec<usize>>, String> {
    let mut iron = Vec::new();
    let mut non_iron = Vec::new();
    for (reduced_index, full_edge_index) in layout.active_dofs.iter().copied().enumerate() {
        if full_edge_index >= topology.nsimplices(1) {
            return Err(format!(
                "active edge {full_edge_index} is outside TEAM13 edge count {}",
                topology.nsimplices(1)
            ));
        }
        let edge = SimplexIdx::new(1, full_edge_index).handle(topology);
        let edge_coords = SimplexCoords::from_simplex_and_coords(&edge, coords);
        if is_iron_point(edge_coords.barycenter().as_view()) {
            iron.push(reduced_index);
        } else {
            non_iron.push(reduced_index);
        }
    }
    if iron.is_empty() || non_iron.is_empty() {
        return Err(format!(
            "TEAM13 split-graph prior requires both iron and non-iron active edge groups, got iron={} non_iron={}",
            iron.len(),
            non_iron.len()
        ));
    }
    Ok(vec![iron, non_iron])
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

fn scale_triplet_matrix(matrix: &SparseTripletMatrix, scale: f64) -> SparseTripletMatrix {
    let mut scaled = SparseTripletMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        scaled.push(row, col, scale * value);
    }
    scaled
}

fn build_team13_operators(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Team13Operators, String> {
    let a_reconstruction = build_reconstructed_barycenter_field_operator(topology, coords)?;
    let mut a_components = Vec::with_capacity(3);
    for component_index in 0..3 {
        let rows = a_reconstruction
            .component_rows(component_index)
            .ok_or_else(|| format!("missing A vector component {component_index}"))?
            .to_vec();
        a_components.push(
            SparseRowOperator::new(topology.nsimplices(1), rows).map_err(|err| err.to_string())?,
        );
    }
    let b_cochain = build_exterior_derivative_operator(topology)?;
    let b_from_faces = build_b_component_from_face_operators(topology, coords)?;
    let mut b_components = Vec::with_capacity(3);
    for face_component in &b_from_faces {
        b_components.push(
            SparseRowOperator::compose(face_component, &b_cochain)
                .map_err(|err| err.to_string())?,
        );
    }
    Ok(Team13Operators {
        a_components,
        b_components,
        b_cochain,
    })
}

fn build_source_free_period_diagnostics(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    mode: Team13DomainMode,
) -> Result<Vec<Team13PeriodDiagnosticSpec>, String> {
    let cell_barycenters = top_cell_barycenters(topology, coords);
    let top_loop = source_free_h_circulation_loop_row(
        &operators.b_components,
        &cell_barycenters,
        mode,
        &[
            [-0.15, -0.15, 0.08],
            [0.15, -0.15, 0.08],
            [0.15, 0.15, 0.08],
            [-0.15, 0.15, 0.08],
        ],
        16,
    )?;
    Ok(vec![Team13PeriodDiagnosticSpec {
        name: SOURCE_FREE_H_TOP_LOOP_NAME.to_string(),
        operator: SparseRowOperator::new(topology.nsimplices(1), vec![top_loop])
            .map_err(|err| err.to_string())?,
    }])
}

fn source_free_h_circulation_loop_row(
    b_components: &[SparseRowOperator],
    cell_barycenters: &[[f64; 3]],
    mode: Team13DomainMode,
    vertices: &[[f64; 3]],
    samples_per_segment: usize,
) -> Result<Vec<(usize, f64)>, String> {
    if b_components.len() != 3 {
        return Err(format!(
            "H circulation diagnostic requires 3 B components, got {}",
            b_components.len()
        ));
    }
    if vertices.len() < 3 {
        return Err("H circulation diagnostic requires at least three loop vertices".to_string());
    }
    if samples_per_segment == 0 {
        return Err(
            "H circulation diagnostic requires at least one sample per segment".to_string(),
        );
    }
    let mut entries = BTreeMap::<usize, f64>::new();
    for segment_index in 0..vertices.len() {
        let start = vertices[segment_index];
        let end = vertices[(segment_index + 1) % vertices.len()];
        accumulate_source_free_h_segment(
            &mut entries,
            b_components,
            cell_barycenters,
            mode,
            start,
            end,
            samples_per_segment,
        )?;
    }
    Ok(entries
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_source_free_h_segment(
    entries: &mut BTreeMap<usize, f64>,
    b_components: &[SparseRowOperator],
    cell_barycenters: &[[f64; 3]],
    mode: Team13DomainMode,
    start: [f64; 3],
    end: [f64; 3],
    samples_per_segment: usize,
) -> Result<(), String> {
    let delta = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if length <= EPS {
        return Ok(());
    }
    let tangent = [delta[0] / length, delta[1] / length, delta[2] / length];
    let quadrature_weight = length / samples_per_segment as f64;
    for sample in 0..samples_per_segment {
        let t = (sample as f64 + 0.5) / samples_per_segment as f64;
        let point = [
            start[0] + t * delta[0],
            start[1] + t * delta[1],
            start[2] + t * delta[2],
        ];
        if !is_source_free_air_xyz(point, mode) {
            return Err(format!(
                "H circulation diagnostic sample [{:.6e}, {:.6e}, {:.6e}] is not in source-free air",
                point[0], point[1], point[2]
            ));
        }
        let cell_index = nearest_source_free_air_cell(cell_barycenters, point, mode)?;
        let barycenter = cell_barycenters[cell_index];
        let reluctivity = reluctivity_at_xyz(barycenter);
        for component in 0..3 {
            let scale = quadrature_weight * tangent[component] * reluctivity;
            if scale.abs() <= EPS {
                continue;
            }
            for (col, value) in &b_components[component].rows[cell_index] {
                *entries.entry(*col).or_insert(0.0) += scale * *value;
            }
        }
    }
    Ok(())
}

fn nearest_source_free_air_cell(
    cell_barycenters: &[[f64; 3]],
    point: [f64; 3],
    mode: Team13DomainMode,
) -> Result<usize, String> {
    cell_barycenters
        .iter()
        .enumerate()
        .filter(|(_, barycenter)| is_source_free_air_xyz(**barycenter, mode))
        .min_by(|(_, lhs), (_, rhs)| {
            point_distance_squared(**lhs, point)
                .partial_cmp(&point_distance_squared(**rhs, point))
                .unwrap()
        })
        .map(|(index, _)| index)
        .ok_or_else(|| {
            "no source-free air cells available for H circulation diagnostic".to_string()
        })
}

fn is_source_free_air_xyz(point: [f64; 3], mode: Team13DomainMode) -> bool {
    let point = FeecVector::from_column_slice(&point);
    !is_iron_point(point.as_view()) && coil_region_at(point.as_view(), mode).is_none()
}

fn reluctivity_at_xyz(point: [f64; 3]) -> f64 {
    let point = FeecVector::from_column_slice(&point);
    reluctivity_at(point.as_view())
}

fn build_derived_quantities(
    operators: &Team13Operators,
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    let names = [
        A_VECTOR_X_DERIVED_NAME,
        A_VECTOR_Y_DERIVED_NAME,
        A_VECTOR_Z_DERIVED_NAME,
    ];
    let mut derived = Vec::new();
    for (name, operator) in names.iter().zip(operators.a_components.iter()) {
        derived.push(LinearPdeDerivedQuantitySpec {
            name: (*name).to_string(),
            operator: operator.clone(),
        });
    }
    derived.push(LinearPdeDerivedQuantitySpec {
        name: B_COCHAIN_DERIVED_NAME.to_string(),
        operator: operators.b_cochain.clone(),
    });
    let names = [
        B_VECTOR_X_DERIVED_NAME,
        B_VECTOR_Y_DERIVED_NAME,
        B_VECTOR_Z_DERIVED_NAME,
    ];
    for (name, operator) in names.iter().zip(operators.b_components.iter()) {
        derived.push(LinearPdeDerivedQuantitySpec {
            name: (*name).to_string(),
            operator: operator.clone(),
        });
    }
    Ok(derived)
}

fn build_exterior_derivative_operator(topology: &Complex) -> Result<SparseRowOperator, String> {
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    SparseRowOperator::new(d1.ncols(), csr_rows(&d1)).map_err(|err| err.to_string())
}

fn build_b_component_from_face_operators(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<SparseRowOperator>, String> {
    let topo_dim = topology.dim();
    let bary_local = barycenter_local(topo_dim);
    let mut component_rows = vec![Vec::with_capacity(topology.cells().len()); 3];
    for cell in topology.skeleton(topo_dim).handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let mut rows = [Vec::new(), Vec::new(), Vec::new()];
        for face in cell.mesh_subsimps(2) {
            let local_face = face.relative_to(&cell);
            let lsf = ddf::whitney::lsf::WhitneyLsf::standard(topo_dim, local_face);
            let ambient_value = cell_coords.lift_form(&lsf.at_point(&bary_local));
            let coeffs = ambient_value.coeffs();
            if coeffs.len() != 3 {
                return Err(format!(
                    "expected 3 coefficients for reconstructed 2-form, found {}",
                    coeffs.len()
                ));
            }
            let coefficients = [coeffs[2], -coeffs[1], coeffs[0]];
            for component_index in 0..3 {
                if coefficients[component_index].abs() > EPS {
                    rows[component_index].push((face.kidx(), coefficients[component_index]));
                }
            }
        }
        for component_index in 0..3 {
            component_rows[component_index].push(std::mem::take(&mut rows[component_index]));
        }
    }
    component_rows
        .into_iter()
        .map(|rows| {
            SparseRowOperator::new(topology.nsimplices(2), rows).map_err(|err| err.to_string())
        })
        .collect()
}

fn build_linearized_b_measurements(
    topology: &Complex,
    coords: &MeshCoords,
    b_cochain: &SparseRowOperator,
    b_component_operators: &[SparseRowOperator],
    nominal_a: &FeecVector,
    variance: f64,
    mode: Team13MeasurementMode,
    legacy_band: f64,
    observation_overrides: Option<&BTreeMap<String, f64>>,
) -> Result<Vec<Team13LinearizedMeasurement>, String> {
    let definitions = team13_surface_measurement_definitions(observation_overrides)?;
    let cell_barycenters = top_cell_barycenters(topology, coords);
    let cell_geometries = top_cell_geometries(topology, coords);
    let mut measurements = Vec::with_capacity(definitions.len());

    for definition in definitions {
        let component_row = measurement_component_row(
            topology,
            coords,
            &cell_geometries,
            b_cochain,
            &b_component_operators[definition.component_index],
            &cell_barycenters,
            &definition,
            mode,
            legacy_band,
        )?;
        let nominal_component = apply_row(&component_row, nominal_a)?;
        if nominal_component.abs() <= 1e-14 {
            return Err(format!(
                "nominal B component for measurement `{}` has zero magnitude and cannot define an absolute-value linearization direction",
                definition.name
            ));
        }
        let sign = if nominal_component >= 0.0 { 1.0 } else { -1.0 };
        let row = scale_row(&component_row, sign);
        let direction = component_direction(definition.component_index, sign);
        measurements.push(Team13LinearizedMeasurement {
            spec: LinearGaussianMeasurementSpec {
                name: definition.name,
                operator: sparse_row_to_triplet(topology.nsimplices(1), &row),
                observations: vec![definition.observation],
                bias: vec![0.0],
                variance,
            },
            nominal_prediction: sign * nominal_component,
            linearization_direction: direction,
        });
    }
    Ok(measurements)
}

fn team13_surface_measurement_definitions(
    observation_overrides: Option<&BTreeMap<String, f64>>,
) -> Result<Vec<Team13SurfaceMeasurementDefinition>, String> {
    // NGSolve TEAM-13 1000 ampere-turn benchmark surface averages. The
    // reference script compares these to signed component magnitudes averaged
    // on the same 25 surfaces.
    let observations = team13_published_steel_observations(Team13PublishedSteelGap::G052);
    let mut definitions = Vec::with_capacity(TEAM13_OBSERVATION_COUNT);
    for (index, z) in [0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060]
        .into_iter()
        .enumerate()
    {
        let name = format!("team13_bz_mid_sheet_{:02}", index + 1);
        let observation = surface_observation(&name, observations[index], observation_overrides)?;
        definitions.push(Team13SurfaceMeasurementDefinition {
            name,
            ngsolve_name: team13_ngsolve_measurement_name(index),
            observation,
            component_index: 2,
            normal_axis: 2,
            target: z,
            x_range: (0.0, 0.0016),
            y_range: (-0.025, 0.025),
            z_range: (z, z),
            quadrature_counts: [
                TEAM13_NGSOLVE_SURFACE_X_SAMPLES,
                TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                1,
            ],
        });
    }
    let xs = [
        0.0021, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.110, 0.1221,
    ];
    for (local_index, x) in xs.into_iter().enumerate() {
        let index = 7 + local_index;
        let name = format!("team13_bx_back_right_top_{:02}", index + 1);
        let observation = surface_observation(&name, observations[index], observation_overrides)?;
        definitions.push(Team13SurfaceMeasurementDefinition {
            name,
            ngsolve_name: team13_ngsolve_measurement_name(index),
            observation,
            component_index: 0,
            normal_axis: 0,
            target: x,
            x_range: (x, x),
            y_range: (0.015, 0.065),
            z_range: (0.060, 0.0632),
            quadrature_counts: [
                1,
                TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                TEAM13_NGSOLVE_SURFACE_Z_SAMPLES,
            ],
        });
    }
    for (local_index, z) in [0.060, 0.050, 0.040, 0.030, 0.020, 0.010, 0.0]
        .into_iter()
        .enumerate()
    {
        let index = 18 + local_index;
        let name = format!("team13_bz_back_right_edge_{:02}", index + 1);
        let observation = surface_observation(&name, observations[index], observation_overrides)?;
        definitions.push(Team13SurfaceMeasurementDefinition {
            name,
            ngsolve_name: team13_ngsolve_measurement_name(index),
            observation,
            component_index: 2,
            normal_axis: 2,
            target: z,
            x_range: (0.1221, 0.1253),
            y_range: (0.015, 0.065),
            z_range: (z, z),
            quadrature_counts: [
                TEAM13_NGSOLVE_SURFACE_X_SAMPLES,
                TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                1,
            ],
        });
    }
    Ok(definitions)
}

fn team13_published_steel_observations(
    gap: Team13PublishedSteelGap,
) -> [f64; TEAM13_OBSERVATION_COUNT] {
    match gap {
        Team13PublishedSteelGap::G052 => [
            1.33, 1.329, 1.286, 1.225, 1.129, 0.985, 0.655, 0.259, 0.453, 0.554, 0.637, 0.698,
            0.755, 0.809, 0.901, 0.945, 0.954, 0.956, 0.960, 0.965, 0.970, 0.974, 0.981, 0.984,
            0.985,
        ],
        Team13PublishedSteelGap::G047 => [
            1.354, 1.339, 1.304, 1.245, 1.138, 0.982, 0.674, 0.263, 0.451, 0.563, 0.641, 0.706,
            0.763, 0.819, 0.907, 0.958, 0.968, 0.968, 0.971, 0.973, 0.982, 0.985, 0.991, 0.995,
            0.995,
        ],
    }
}

fn team13_ngsolve_measurement_name(index: usize) -> String {
    format!("measurement_{:02}", index + 1)
}

fn team13_steel_surface_group(index: usize) -> Result<Team13SteelSurfaceGroup, String> {
    match index {
        0..=6 => Ok(Team13SteelSurfaceGroup::MidSheet),
        7..=17 => Ok(Team13SteelSurfaceGroup::BackRightTop),
        18..=24 => Ok(Team13SteelSurfaceGroup::BackRightEdge),
        _ => Err(format!(
            "TEAM 13 steel surface index {index} is outside 0..25"
        )),
    }
}

fn surface_observation(
    name: &str,
    default_observation: f64,
    observation_overrides: Option<&BTreeMap<String, f64>>,
) -> Result<f64, String> {
    match observation_overrides {
        Some(overrides) => overrides
            .get(name)
            .copied()
            .ok_or_else(|| format!("TEAM 13 observation override is missing `{name}`")),
        None => Ok(default_observation),
    }
}

fn team13_point_measurement_definitions() -> Vec<Team13PointMeasurementDefinition> {
    let points = [
        [0.010, 0.020, 0.055],
        [0.020, 0.020, 0.055],
        [0.030, 0.020, 0.055],
        [0.040, 0.020, 0.055],
        [0.050, 0.020, 0.055],
        [0.060, 0.020, 0.055],
        [0.070, 0.020, 0.055],
        [0.080, 0.020, 0.055],
        [0.090, 0.020, 0.055],
        [0.100, 0.020, 0.055],
        [0.110, 0.020, 0.055],
        [0.0022, 0.0151, 0.0601],
        [0.0020, 0.0149, 0.0509],
        [0.0015, 0.0, 0.0550],
        [0.0015, 0.0, 0.0250],
    ];
    points
        .into_iter()
        .enumerate()
        .map(|(index, point)| Team13PointMeasurementDefinition {
            name: format!("team13_bnorm_point_{:02}", 26 + index),
            observation: None,
            point,
        })
        .collect()
}

fn build_synthetic_surface_flux_measurements(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    initial_a: &FeecVector,
    truth_a: &FeecVector,
    variance: f64,
) -> Result<Vec<Team13LinearizedMeasurement>, String> {
    let cell_geometries = top_cell_geometries(topology, coords);
    team13_synthetic_surface_flux_definitions()
        .into_iter()
        .map(|definition| {
            let row =
                surface_flux_component_row(&cell_geometries, &operators.b_cochain, &definition)?;
            let observation = apply_row(&row, truth_a)?;
            let nominal_prediction = apply_row(&row, initial_a)?;
            let sign = nonzero_sign(&definition.name, nominal_prediction)?;
            Ok(Team13LinearizedMeasurement {
                spec: LinearGaussianMeasurementSpec {
                    name: definition.name,
                    operator: sparse_row_to_triplet(topology.nsimplices(1), &row),
                    observations: vec![observation],
                    bias: vec![0.0],
                    variance,
                },
                nominal_prediction,
                linearization_direction: component_direction(definition.component_index, sign),
            })
        })
        .collect()
}

fn build_team13_synthetic_benchmark_geometry_observations(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    layout: &DofLayout,
    initial_map: &[f64],
    truth_map: &[f64],
    steel_quadrature: Team13SteelObservationQuadratureMode,
    smoothing: f64,
) -> Result<Team13SyntheticBenchmarkObservationBuild, String> {
    let cell_geometries = top_cell_geometries(topology, coords);
    let steel_part = load_or_build_team13_steel_full_observation_part(
        topology,
        coords,
        operators,
        &cell_geometries,
        steel_quadrature,
    )?;
    let mut triplets = steel_part.triplets.clone();
    let mut row_count = steel_part.row_count;
    let mut groups = steel_part.groups.clone();
    let mut specs = steel_part.specs.clone();
    let assimilated_row_count = row_count;
    let assimilated_triplets = triplets.clone();
    let assimilated_groups = groups.clone();
    let assimilated_specs = specs.clone();

    let all_cells = (0..cell_geometries.len()).collect::<Vec<_>>();
    for definition in team13_point_measurement_definitions() {
        let mut last_cell = None;
        let cell_index = locate_surface_sample_cell(
            &cell_geometries,
            &all_cells,
            &mut last_cell,
            definition.point,
        )
        .map_err(|err| {
            format!(
                "point observation `{}` could not locate sample [{:.6e}, {:.6e}, {:.6e}]: {err}",
                definition.name, definition.point[0], definition.point[1], definition.point[2]
            )
        })?;
        let rows = append_point_vector_norm_sample_rows(
            &cell_geometries[cell_index],
            &operators.b_cochain,
            definition.point,
            &mut row_count,
            &mut triplets,
        )?;
        groups.push(SmoothGroupedNormObservation {
            name: definition.name.clone(),
            samples: vec![SmoothGroupedNormSample { rows, weight: 1.0 }],
        });
        specs.push(Team13SyntheticBenchmarkObservationSpec {
            name: definition.name,
            group: Team13SyntheticBenchmarkObservationGroup::AirPoint,
            steel_surface_group: None,
        });
    }

    let full_operator =
        SparseTripletMatrix::from_triplets(row_count, topology.nsimplices(1), triplets);
    let full_bias = vec![0.0; row_count];
    let (reduced_operator, reduced_bias) = restrict_columns_and_fold_fixed(
        &core_triplet_to_feec_csr(&full_operator),
        &FeecVector::from_vec(full_bias.clone()),
        layout,
    )?;
    let model = SmoothGroupedNormLinearResidualModel::new(
        csr_to_triplet(&reduced_operator),
        reduced_bias.iter().copied().collect(),
        groups,
        smoothing,
    )?;
    let assimilated_full_operator = SparseTripletMatrix::from_triplets(
        assimilated_row_count,
        topology.nsimplices(1),
        assimilated_triplets,
    );
    let assimilated_full_bias = vec![0.0; assimilated_row_count];
    let (assimilated_reduced_operator, assimilated_reduced_bias) = restrict_columns_and_fold_fixed(
        &core_triplet_to_feec_csr(&assimilated_full_operator),
        &FeecVector::from_vec(assimilated_full_bias),
        layout,
    )?;
    let assimilated_model = SmoothGroupedNormLinearResidualModel::new(
        csr_to_triplet(&assimilated_reduced_operator),
        assimilated_reduced_bias.iter().copied().collect(),
        assimilated_groups,
        smoothing,
    )?;
    let observations = model.smooth_norm_values(truth_map)?;
    let initial_predictions = model.smooth_norm_values(initial_map)?;
    if observations.len() != specs.len() || initial_predictions.len() != specs.len() {
        return Err("synthetic benchmark observation count mismatch".to_string());
    }
    let assimilated_observations = assimilated_model.smooth_norm_values(truth_map)?;
    if assimilated_observations.len() != assimilated_specs.len() {
        return Err("synthetic benchmark assimilated observation count mismatch".to_string());
    }

    Ok(Team13SyntheticBenchmarkObservationBuild {
        model,
        observations,
        initial_predictions,
        specs,
        assimilated_model,
        assimilated_observations,
        assimilated_specs,
        #[cfg(test)]
        full_operator,
        #[cfg(test)]
        full_bias,
    })
}

fn build_team13_published_steel_smooth_observations(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    layout: &DofLayout,
    steel_quadrature: Team13SteelObservationQuadratureMode,
    smoothing: f64,
    observed_gap: Team13PublishedSteelGap,
) -> Result<Team13PublishedSteelSmoothObservationBuild, String> {
    let cell_geometries = top_cell_geometries(topology, coords);
    let steel_part = load_or_build_team13_steel_full_observation_part(
        topology,
        coords,
        operators,
        &cell_geometries,
        steel_quadrature,
    )?;
    let full_operator = SparseTripletMatrix::from_triplets(
        steel_part.row_count,
        topology.nsimplices(1),
        steel_part.triplets,
    );
    let full_bias = vec![0.0; steel_part.row_count];
    let (reduced_operator, reduced_bias) = restrict_columns_and_fold_fixed(
        &core_triplet_to_feec_csr(&full_operator),
        &FeecVector::from_vec(full_bias),
        layout,
    )?;
    let model = SmoothGroupedNormLinearResidualModel::new(
        csr_to_triplet(&reduced_operator),
        reduced_bias.iter().copied().collect(),
        steel_part.groups,
        smoothing,
    )?;
    let observations = team13_published_steel_observations(observed_gap).to_vec();
    if observations.len() != steel_part.specs.len() {
        return Err(format!(
            "published TEAM 13 steel observation count {} did not match smooth model spec count {}",
            observations.len(),
            steel_part.specs.len()
        ));
    }
    Ok(Team13PublishedSteelSmoothObservationBuild {
        model,
        observations,
        specs: steel_part.specs,
    })
}

fn team13_synthetic_benchmark_surface_definitions(
    _quadrature: Team13SteelObservationQuadratureMode,
) -> Result<Vec<Team13SurfaceMeasurementDefinition>, String> {
    let mut definitions = team13_surface_measurement_definitions(None)?;
    for definition in &mut definitions {
        definition.observation = 0.0;
    }
    Ok(definitions)
}

fn load_or_build_team13_steel_full_observation_part(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    cell_geometries: &[Team13CellGeometry],
    quadrature: Team13SteelObservationQuadratureMode,
) -> Result<Team13SteelFullObservationPart, String> {
    let cache_path = team13_steel_observation_cache_path(topology, coords, quadrature);
    let expected_ncols = topology.nsimplices(1);
    if let Some(cached) = try_load_team13_steel_observation_cache(&cache_path, expected_ncols) {
        eprintln!(
            "TEAM 13 steel observation cache hit: `{}` (rows={} nnz={})",
            cache_path.display(),
            cached.row_count,
            cached.triplets.len()
        );
        return Ok(cached);
    }

    eprintln!(
        "TEAM 13 steel observation cache miss: building {} steel rows `{}`",
        quadrature.as_str(),
        cache_path.display()
    );
    let built = build_team13_steel_full_observation_part(
        topology,
        coords,
        operators,
        cell_geometries,
        quadrature,
    )?;
    if let Err(err) = write_team13_steel_observation_cache(&cache_path, expected_ncols, &built) {
        eprintln!(
            "TEAM 13 steel observation cache write skipped for `{}`: {err}",
            cache_path.display()
        );
    } else {
        eprintln!(
            "TEAM 13 steel observation cache wrote `{}` (rows={} nnz={})",
            cache_path.display(),
            built.row_count,
            built.triplets.len()
        );
    }
    Ok(built)
}

fn build_team13_steel_full_observation_part(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    cell_geometries: &[Team13CellGeometry],
    quadrature: Team13SteelObservationQuadratureMode,
) -> Result<Team13SteelFullObservationPart, String> {
    let mut triplets = Vec::new();
    let mut row_count = 0usize;
    let mut groups = Vec::new();
    let mut specs = Vec::new();
    for (definition_index, definition) in
        team13_synthetic_benchmark_surface_definitions(quadrature)?
            .into_iter()
            .enumerate()
    {
        debug_assert_eq!(
            definition.ngsolve_name,
            team13_ngsolve_measurement_name(definition_index)
        );
        let samples = match quadrature {
            Team13SteelObservationQuadratureMode::NgsolveStyle => {
                append_surface_component_abs_samples(
                    cell_geometries,
                    &operators.b_cochain,
                    &definition,
                    &mut row_count,
                    &mut triplets,
                )?
            }
            Team13SteelObservationQuadratureMode::FaceCochain => {
                append_surface_face_cochain_sample(
                    topology,
                    coords,
                    &operators.b_cochain,
                    &definition,
                    &mut row_count,
                    &mut triplets,
                )?
            }
        };
        let steel_surface_group = team13_steel_surface_group(definition_index)?;
        groups.push(SmoothGroupedNormObservation {
            name: definition.name.clone(),
            samples,
        });
        specs.push(Team13SyntheticBenchmarkObservationSpec {
            name: definition.name,
            group: Team13SyntheticBenchmarkObservationGroup::SteelAverage,
            steel_surface_group: Some(steel_surface_group),
        });
    }
    let ncols = topology.nsimplices(1);
    if triplets.iter().any(|entry| entry.col >= ncols) {
        return Err(
            "TEAM 13 steel observation build produced a column outside edge dimension".into(),
        );
    }
    Ok(Team13SteelFullObservationPart {
        row_count,
        triplets,
        groups,
        specs,
    })
}

fn team13_steel_observation_cache_path(
    topology: &Complex,
    coords: &MeshCoords,
    quadrature: Team13SteelObservationQuadratureMode,
) -> PathBuf {
    PathBuf::from(TEAM13_STEEL_OBSERVATION_CACHE_DIR).join(format!(
        "steel_observation_{}_{}.bin",
        quadrature.as_str(),
        team13_steel_observation_cache_key(topology, coords, quadrature)
    ))
}

fn team13_steel_observation_cache_key(
    topology: &Complex,
    coords: &MeshCoords,
    quadrature: Team13SteelObservationQuadratureMode,
) -> String {
    let mut hash = Fnv64::new();
    hash.bytes(b"team13-steel-observation-cache");
    hash.u64(TEAM13_STEEL_OBSERVATION_CACHE_VERSION);
    hash.bytes(quadrature.as_str().as_bytes());
    hash.usize(TEAM13_NGSOLVE_SURFACE_X_SAMPLES);
    hash.usize(TEAM13_NGSOLVE_SURFACE_Y_SAMPLES);
    hash.usize(TEAM13_NGSOLVE_SURFACE_Z_SAMPLES);
    hash.usize(topology.dim());
    hash.usize(coords.dim());
    hash.usize(coords.nvertices());
    for coord in coords.coord_iter() {
        for value in coord.iter() {
            hash.u64(value.to_bits());
        }
    }
    for dim in 0..=topology.dim() {
        hash.usize(topology.nsimplices(dim));
        for simplex in topology.skeleton(dim).iter() {
            hash.usize(simplex.nvertices());
            for vertex in simplex.iter() {
                hash.usize(vertex);
            }
        }
    }
    format!("{:016x}", hash.finish())
}

fn try_load_team13_steel_observation_cache(
    path: &Path,
    expected_ncols: usize,
) -> Option<Team13SteelFullObservationPart> {
    if !path.exists() {
        return None;
    }
    match fs::read(path)
        .map_err(|err| err.to_string())
        .and_then(|bytes| parse_team13_steel_observation_cache(&bytes, expected_ncols))
    {
        Ok(part) => Some(part),
        Err(err) => {
            eprintln!(
                "TEAM 13 steel observation cache ignored `{}`: {err}",
                path.display()
            );
            None
        }
    }
}

fn parse_team13_steel_observation_cache(
    bytes: &[u8],
    expected_ncols: usize,
) -> Result<Team13SteelFullObservationPart, String> {
    let mut cursor = Cursor::new(bytes);
    let mut magic = [0u8; 16];
    cursor
        .read_exact(&mut magic)
        .map_err(|err| format!("cache is too short to contain magic: {err}"))?;
    if &magic != TEAM13_STEEL_OBSERVATION_CACHE_MAGIC {
        return Err("cache magic/version tag did not match".to_string());
    }
    let version = read_cache_u64(&mut cursor)?;
    if version != TEAM13_STEEL_OBSERVATION_CACHE_VERSION {
        return Err(format!(
            "cache version {version} did not match expected {}",
            TEAM13_STEEL_OBSERVATION_CACHE_VERSION
        ));
    }
    let ncols = read_cache_usize(&mut cursor)?;
    if ncols != expected_ncols {
        return Err(format!(
            "cache edge-column count {ncols} did not match expected {expected_ncols}"
        ));
    }
    let row_count = read_cache_usize(&mut cursor)?;
    let triplet_count = read_cache_usize(&mut cursor)?;
    let group_count = read_cache_usize(&mut cursor)?;
    let spec_count = read_cache_usize(&mut cursor)?;
    if triplet_count > 100_000_000 || group_count > 10_000 || spec_count > 10_000 {
        return Err("cache metadata counts are implausibly large".to_string());
    }

    let mut triplets = Vec::with_capacity(triplet_count);
    for _ in 0..triplet_count {
        let row = read_cache_usize(&mut cursor)?;
        let col = read_cache_usize(&mut cursor)?;
        let value = read_cache_f64(&mut cursor)?;
        if row >= row_count || col >= ncols {
            return Err(format!(
                "cache triplet ({row}, {col}) exceeds dimensions {row_count}x{ncols}"
            ));
        }
        if !value.is_finite() {
            return Err("cache triplet contains a non-finite value".to_string());
        }
        triplets.push(SparseTriplet { row, col, value });
    }

    let mut groups = Vec::with_capacity(group_count);
    for _ in 0..group_count {
        let name = read_cache_string(&mut cursor)?;
        let sample_count = read_cache_usize(&mut cursor)?;
        if sample_count > 10_000_000 {
            return Err("cache sample count is implausibly large".to_string());
        }
        let mut samples = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            let weight = read_cache_f64(&mut cursor)?;
            let rows_count = read_cache_usize(&mut cursor)?;
            if rows_count > 16 {
                return Err("cache grouped-norm sample references too many rows".to_string());
            }
            let mut rows = Vec::with_capacity(rows_count);
            for _ in 0..rows_count {
                let row = read_cache_usize(&mut cursor)?;
                if row >= row_count {
                    return Err(format!(
                        "cache grouped-norm sample row {row} exceeds row count {row_count}"
                    ));
                }
                rows.push(row);
            }
            if !weight.is_finite() {
                return Err("cache grouped-norm sample has a non-finite weight".to_string());
            }
            samples.push(SmoothGroupedNormSample { rows, weight });
        }
        groups.push(SmoothGroupedNormObservation { name, samples });
    }

    let mut specs = Vec::with_capacity(spec_count);
    for _ in 0..spec_count {
        let name = read_cache_string(&mut cursor)?;
        let group = read_cache_observation_group(&mut cursor)?;
        let steel_surface_group = read_cache_steel_surface_group(&mut cursor)?;
        specs.push(Team13SyntheticBenchmarkObservationSpec {
            name,
            group,
            steel_surface_group,
        });
    }
    if specs.len() != TEAM13_OBSERVATION_COUNT || groups.len() != TEAM13_OBSERVATION_COUNT {
        return Err(format!(
            "cache contains {} specs and {} groups, expected {TEAM13_OBSERVATION_COUNT}",
            specs.len(),
            groups.len()
        ));
    }
    if cursor.position() != bytes.len() as u64 {
        return Err("cache contains trailing bytes".to_string());
    }
    Ok(Team13SteelFullObservationPart {
        row_count,
        triplets,
        groups,
        specs,
    })
}

fn write_team13_steel_observation_cache(
    path: &Path,
    ncols: usize,
    part: &Team13SteelFullObservationPart,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create TEAM 13 steel observation cache directory `{}`: {err}",
                parent.display()
            )
        })?;
    }
    let mut bytes = Vec::with_capacity(64 + 24 * part.triplets.len());
    bytes.extend_from_slice(TEAM13_STEEL_OBSERVATION_CACHE_MAGIC);
    write_cache_u64(&mut bytes, TEAM13_STEEL_OBSERVATION_CACHE_VERSION);
    write_cache_usize(&mut bytes, ncols);
    write_cache_usize(&mut bytes, part.row_count);
    write_cache_usize(&mut bytes, part.triplets.len());
    write_cache_usize(&mut bytes, part.groups.len());
    write_cache_usize(&mut bytes, part.specs.len());
    for triplet in &part.triplets {
        write_cache_usize(&mut bytes, triplet.row);
        write_cache_usize(&mut bytes, triplet.col);
        write_cache_f64(&mut bytes, triplet.value);
    }
    for group in &part.groups {
        write_cache_string(&mut bytes, &group.name);
        write_cache_usize(&mut bytes, group.samples.len());
        for sample in &group.samples {
            write_cache_f64(&mut bytes, sample.weight);
            write_cache_usize(&mut bytes, sample.rows.len());
            for row in &sample.rows {
                write_cache_usize(&mut bytes, *row);
            }
        }
    }
    for spec in &part.specs {
        write_cache_string(&mut bytes, &spec.name);
        write_cache_observation_group(&mut bytes, spec.group);
        write_cache_steel_surface_group(&mut bytes, spec.steel_surface_group);
    }
    fs::write(path, bytes)
        .map_err(|err| format!("failed to write cache `{}`: {err}", path.display()))
}

fn read_cache_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    cursor
        .read_exact(&mut bytes)
        .map_err(|err| format!("cache ended while reading u64: {err}"))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_cache_usize(cursor: &mut Cursor<&[u8]>) -> Result<usize, String> {
    usize::try_from(read_cache_u64(cursor)?)
        .map_err(|_| "cache integer does not fit usize".to_string())
}

fn read_cache_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64, String> {
    Ok(f64::from_bits(read_cache_u64(cursor)?))
}

fn read_cache_bool(cursor: &mut Cursor<&[u8]>) -> Result<bool, String> {
    match read_cache_u64(cursor)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(format!("cache boolean code {other} is invalid")),
    }
}

fn read_cache_string(cursor: &mut Cursor<&[u8]>) -> Result<String, String> {
    let len = read_cache_usize(cursor)?;
    if len > 4096 {
        return Err("cache string length is implausibly large".to_string());
    }
    let mut bytes = vec![0u8; len];
    cursor
        .read_exact(&mut bytes)
        .map_err(|err| format!("cache ended while reading string: {err}"))?;
    String::from_utf8(bytes).map_err(|err| format!("cache string is not UTF-8: {err}"))
}

fn read_cache_observation_group(
    cursor: &mut Cursor<&[u8]>,
) -> Result<Team13SyntheticBenchmarkObservationGroup, String> {
    match read_cache_u64(cursor)? {
        0 => Ok(Team13SyntheticBenchmarkObservationGroup::SteelAverage),
        1 => Ok(Team13SyntheticBenchmarkObservationGroup::AirPoint),
        other => Err(format!("unknown cached observation group code {other}")),
    }
}

fn read_cache_steel_surface_group(
    cursor: &mut Cursor<&[u8]>,
) -> Result<Option<Team13SteelSurfaceGroup>, String> {
    match read_cache_u64(cursor)? {
        0 => Ok(Some(Team13SteelSurfaceGroup::MidSheet)),
        1 => Ok(Some(Team13SteelSurfaceGroup::BackRightTop)),
        2 => Ok(Some(Team13SteelSurfaceGroup::BackRightEdge)),
        u64::MAX => Ok(None),
        other => Err(format!("unknown cached steel surface group code {other}")),
    }
}

fn write_cache_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_cache_usize(bytes: &mut Vec<u8>, value: usize) {
    write_cache_u64(bytes, value as u64);
}

fn write_cache_f64(bytes: &mut Vec<u8>, value: f64) {
    write_cache_u64(bytes, value.to_bits());
}

fn write_cache_bool(bytes: &mut Vec<u8>, value: bool) {
    write_cache_u64(bytes, u64::from(value));
}

fn write_cache_string(bytes: &mut Vec<u8>, value: &str) {
    write_cache_usize(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

fn read_cache_square_newton_iteration(
    cursor: &mut Cursor<&[u8]>,
) -> Result<SquareNewtonIteration, String> {
    Ok(SquareNewtonIteration {
        iteration: read_cache_usize(cursor)?,
        residual_norm: read_cache_f64(cursor)?,
        trial_residual_norm: read_cache_f64(cursor)?,
        step_norm: read_cache_f64(cursor)?,
        alpha: read_cache_f64(cursor)?,
        linear_solve: read_cache_linear_solve_stats(cursor)?,
    })
}

fn write_cache_square_newton_iteration(bytes: &mut Vec<u8>, iteration: &SquareNewtonIteration) {
    write_cache_usize(bytes, iteration.iteration);
    write_cache_f64(bytes, iteration.residual_norm);
    write_cache_f64(bytes, iteration.trial_residual_norm);
    write_cache_f64(bytes, iteration.step_norm);
    write_cache_f64(bytes, iteration.alpha);
    write_cache_linear_solve_stats(bytes, &iteration.linear_solve);
}

fn read_cache_linear_solve_stats(
    cursor: &mut Cursor<&[u8]>,
) -> Result<feg_infer::nonlinear::GaussNewtonLinearSolveStats, String> {
    let mode = match read_cache_u64(cursor)? {
        0 => feg_infer::nonlinear::GaussNewtonLinearSolveMode::IterativeCg,
        1 => feg_infer::nonlinear::GaussNewtonLinearSolveMode::DirectCholesky,
        other => return Err(format!("unknown cached linear solve mode code {other}")),
    };
    let iterations = read_cache_usize(cursor)?;
    let final_residual_norm = read_cache_f64(cursor)?;
    let converged = read_cache_bool(cursor)?;
    let factor_nnz_raw = read_cache_u64(cursor)?;
    let factor_nnz = if factor_nnz_raw == u64::MAX {
        None
    } else {
        Some(
            usize::try_from(factor_nnz_raw)
                .map_err(|_| "cached factor nnz does not fit usize".to_string())?,
        )
    };
    Ok(feg_infer::nonlinear::GaussNewtonLinearSolveStats {
        mode,
        iterations,
        final_residual_norm,
        converged,
        factor_nnz,
    })
}

fn write_cache_linear_solve_stats(
    bytes: &mut Vec<u8>,
    stats: &feg_infer::nonlinear::GaussNewtonLinearSolveStats,
) {
    let mode_code = match stats.mode {
        feg_infer::nonlinear::GaussNewtonLinearSolveMode::IterativeCg => 0,
        feg_infer::nonlinear::GaussNewtonLinearSolveMode::DirectCholesky => 1,
    };
    write_cache_u64(bytes, mode_code);
    write_cache_usize(bytes, stats.iterations);
    write_cache_f64(bytes, stats.final_residual_norm);
    write_cache_bool(bytes, stats.converged);
    write_cache_u64(
        bytes,
        stats
            .factor_nnz
            .map(|value| value as u64)
            .unwrap_or(u64::MAX),
    );
}

fn write_cache_observation_group(
    bytes: &mut Vec<u8>,
    group: Team13SyntheticBenchmarkObservationGroup,
) {
    let code = match group {
        Team13SyntheticBenchmarkObservationGroup::SteelAverage => 0,
        Team13SyntheticBenchmarkObservationGroup::AirPoint => 1,
    };
    write_cache_u64(bytes, code);
}

fn write_cache_steel_surface_group(bytes: &mut Vec<u8>, group: Option<Team13SteelSurfaceGroup>) {
    let code = match group {
        Some(Team13SteelSurfaceGroup::MidSheet) => 0,
        Some(Team13SteelSurfaceGroup::BackRightTop) => 1,
        Some(Team13SteelSurfaceGroup::BackRightEdge) => 2,
        None => u64::MAX,
    };
    write_cache_u64(bytes, code);
}

struct Fnv64 {
    hash: u64,
}

impl Fnv64 {
    fn new() -> Self {
        Self {
            hash: 0xcbf29ce484222325,
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.hash ^= u64::from(*byte);
            self.hash = self.hash.wrapping_mul(0x100000001b3);
        }
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn finish(self) -> u64 {
        self.hash
    }
}

fn append_surface_component_abs_samples(
    cell_geometries: &[Team13CellGeometry],
    b_cochain: &SparseRowOperator,
    definition: &Team13SurfaceMeasurementDefinition,
    row_count: &mut usize,
    triplets: &mut Vec<SparseTriplet>,
) -> Result<Vec<SmoothGroupedNormSample>, String> {
    let [axis_u, axis_v] = tangential_axes(definition.normal_axis)?;
    let range_u = measurement_axis_range(definition, axis_u);
    let range_v = measurement_axis_range(definition, axis_v);
    let count_u = definition.quadrature_counts[axis_u];
    let count_v = definition.quadrature_counts[axis_v];
    let weights_u = simpson_uniform_weights(count_u, range_u.0, range_u.1)?;
    let weights_v = simpson_uniform_weights(count_v, range_v.0, range_v.1)?;
    let area = (range_u.1 - range_u.0).abs() * (range_v.1 - range_v.0).abs();
    if !area.is_finite() || area <= 0.0 {
        return Err(format!(
            "measurement `{}` has invalid surface area {area:.6e}",
            definition.name
        ));
    }
    let candidates = surface_candidate_cells(cell_geometries, definition)?;
    let mut samples = Vec::with_capacity(count_u * count_v);
    let mut last_cell = None;
    for (i, weight_u) in weights_u.iter().enumerate() {
        let u = sample_axis_value(range_u, count_u, i)?;
        for (j, weight_v) in weights_v.iter().enumerate() {
            let v = sample_axis_value(range_v, count_v, j)?;
            let mut point = [0.0; 3];
            point[definition.normal_axis] = definition.target;
            point[axis_u] = u;
            point[axis_v] = v;
            let cell_index =
                locate_surface_sample_cell(cell_geometries, &candidates, &mut last_cell, point)
                    .map_err(|err| {
                        format!(
                            "measurement `{}` could not locate quadrature sample [{:.6e}, {:.6e}, {:.6e}]: {err}",
                            definition.name, point[0], point[1], point[2]
                        )
                    })?;
            let row = *row_count;
            *row_count += 1;
            for (col, value) in point_b_component_row(
                &cell_geometries[cell_index],
                b_cochain,
                definition.component_index,
                point,
            )? {
                triplets.push(SparseTriplet { row, col, value });
            }
            samples.push(SmoothGroupedNormSample {
                rows: vec![row],
                weight: *weight_u * *weight_v / area,
            });
        }
    }
    Ok(samples)
}

fn append_surface_face_cochain_sample(
    topology: &Complex,
    coords: &MeshCoords,
    b_cochain: &SparseRowOperator,
    definition: &Team13SurfaceMeasurementDefinition,
    row_count: &mut usize,
    triplets: &mut Vec<SparseTriplet>,
) -> Result<Vec<SmoothGroupedNormSample>, String> {
    let cochain_row = surface_face_cochain_row(topology, coords, b_cochain, definition)?;
    let row = *row_count;
    *row_count += 1;
    for (col, value) in cochain_row.row {
        triplets.push(SparseTriplet { row, col, value });
    }
    Ok(vec![SmoothGroupedNormSample {
        rows: vec![row],
        weight: 1.0,
    }])
}

fn append_point_vector_norm_sample_rows(
    cell: &Team13CellGeometry,
    b_cochain: &SparseRowOperator,
    point: [f64; 3],
    row_count: &mut usize,
    triplets: &mut Vec<SparseTriplet>,
) -> Result<Vec<usize>, String> {
    let mut rows = Vec::with_capacity(3);
    for component_index in 0..3 {
        let row = *row_count;
        *row_count += 1;
        for (col, value) in point_b_component_row(cell, b_cochain, component_index, point)? {
            triplets.push(SparseTriplet { row, col, value });
        }
        rows.push(row);
    }
    Ok(rows)
}

fn team13_synthetic_surface_flux_definitions() -> Vec<Team13SurfaceMeasurementDefinition> {
    let mut definitions = Vec::new();
    for z in [0.010, 0.030, 0.050] {
        let name = format!("synthetic_flux_bz_mid_sheet_z{:.0}mm", 1000.0 * z);
        definitions.push(Team13SurfaceMeasurementDefinition {
            ngsolve_name: name.clone(),
            name,
            observation: 0.0,
            component_index: 2,
            normal_axis: 2,
            target: z,
            x_range: (0.0, 0.0016),
            y_range: (-0.025, 0.025),
            z_range: (z, z),
            quadrature_counts: [5, 9, 1],
        });
    }
    definitions.push(Team13SurfaceMeasurementDefinition {
        name: "synthetic_flux_bx_top_x30mm".to_string(),
        ngsolve_name: "synthetic_flux_bx_top_x30mm".to_string(),
        observation: 0.0,
        component_index: 0,
        normal_axis: 0,
        target: 0.030,
        x_range: (0.030, 0.030),
        y_range: (0.015, 0.065),
        z_range: (0.060, 0.0632),
        quadrature_counts: [1, 9, 5],
    });
    definitions.push(Team13SurfaceMeasurementDefinition {
        name: "synthetic_flux_bz_back_right_edge_z30mm".to_string(),
        ngsolve_name: "synthetic_flux_bz_back_right_edge_z30mm".to_string(),
        observation: 0.0,
        component_index: 2,
        normal_axis: 2,
        target: 0.030,
        x_range: (0.1221, 0.1253),
        y_range: (0.015, 0.065),
        z_range: (0.030, 0.030),
        quadrature_counts: [5, 9, 1],
    });
    definitions
}

fn measurement_component_row(
    topology: &Complex,
    coords: &MeshCoords,
    cell_geometries: &[Team13CellGeometry],
    b_cochain: &SparseRowOperator,
    barycenter_operator: &SparseRowOperator,
    cell_barycenters: &[[f64; 3]],
    definition: &Team13SurfaceMeasurementDefinition,
    mode: Team13MeasurementMode,
    legacy_band: f64,
) -> Result<Vec<(usize, f64)>, String> {
    match mode {
        Team13MeasurementMode::BenchmarkExact => {
            surface_flux_component_row(cell_geometries, b_cochain, definition)
        }
        Team13MeasurementMode::FaceCochain => {
            Ok(surface_face_cochain_row(topology, coords, b_cochain, definition)?.row)
        }
        Team13MeasurementMode::LegacyBand => averaged_component_row(
            barycenter_operator,
            cell_barycenters,
            definition,
            mode,
            legacy_band,
        ),
    }
}

fn surface_flux_component_row(
    cell_geometries: &[Team13CellGeometry],
    b_cochain: &SparseRowOperator,
    definition: &Team13SurfaceMeasurementDefinition,
) -> Result<Vec<(usize, f64)>, String> {
    let [axis_u, axis_v] = tangential_axes(definition.normal_axis)?;
    let range_u = measurement_axis_range(definition, axis_u);
    let range_v = measurement_axis_range(definition, axis_v);
    let count_u = definition.quadrature_counts[axis_u];
    let count_v = definition.quadrature_counts[axis_v];
    let weights_u = simpson_uniform_weights(count_u, range_u.0, range_u.1)?;
    let weights_v = simpson_uniform_weights(count_v, range_v.0, range_v.1)?;
    let area = (range_u.1 - range_u.0).abs() * (range_v.1 - range_v.0).abs();
    if !area.is_finite() || area <= 0.0 {
        return Err(format!(
            "measurement `{}` has invalid surface area {area:.6e}",
            definition.name
        ));
    }
    let candidates = surface_candidate_cells(cell_geometries, definition)?;
    let mut entries = BTreeMap::<usize, f64>::new();
    let mut last_cell = None;
    for (i, weight_u) in weights_u.iter().enumerate() {
        let u = sample_axis_value(range_u, count_u, i)?;
        for (j, weight_v) in weights_v.iter().enumerate() {
            let v = sample_axis_value(range_v, count_v, j)?;
            let mut point = [0.0; 3];
            point[definition.normal_axis] = definition.target;
            point[axis_u] = u;
            point[axis_v] = v;
            let cell_index =
                locate_surface_sample_cell(cell_geometries, &candidates, &mut last_cell, point)
                    .map_err(|err| {
                        format!(
                            "measurement `{}` could not locate quadrature sample [{:.6e}, {:.6e}, {:.6e}]: {err}",
                            definition.name, point[0], point[1], point[2]
                        )
                    })?;
            let sample_weight = *weight_u * *weight_v / area;
            let sample_row = point_b_component_row(
                &cell_geometries[cell_index],
                b_cochain,
                definition.component_index,
                point,
            )?;
            for (col, value) in sample_row {
                *entries.entry(col).or_insert(0.0) += sample_weight * value;
            }
        }
    }
    Ok(entries
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect())
}

fn surface_face_cochain_row(
    topology: &Complex,
    coords: &MeshCoords,
    b_cochain: &SparseRowOperator,
    definition: &Team13SurfaceMeasurementDefinition,
) -> Result<Team13FaceCochainSurfaceRow, String> {
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "face-cochain measurement `{}` requires a 3D tetrahedral mesh, got topology dim {} and coordinate dim {}",
            definition.name,
            topology.dim(),
            coords.dim()
        ));
    }
    if definition.component_index != definition.normal_axis {
        return Err(format!(
            "face-cochain measurement `{}` requires component axis {} to equal normal axis {}",
            definition.name, definition.component_index, definition.normal_axis
        ));
    }
    let expected_area = surface_measurement_area(definition);
    if !expected_area.is_finite() || expected_area <= 0.0 {
        return Err(format!(
            "face-cochain measurement `{}` has invalid expected area {expected_area:.6e}",
            definition.name
        ));
    }

    let mut entries = BTreeMap::<usize, f64>::new();
    let mut selected_area = 0.0;
    let mut face_count = 0usize;
    for face in topology.skeleton(2).handle_iter() {
        let Some(vertices) = measurement_face_vertices(coords, &face, definition)? else {
            continue;
        };
        let normal = triangle_oriented_area_normal(vertices);
        let normal_norm = vector_norm(normal);
        if normal_norm <= EPS {
            return Err(format!(
                "face-cochain measurement `{}` selected degenerate face {}",
                definition.name,
                face.kidx()
            ));
        }
        let axis_component = normal[definition.normal_axis];
        let tangential_normal_norm = (normal_norm * normal_norm - axis_component * axis_component)
            .max(0.0)
            .sqrt();
        if tangential_normal_norm > TEAM13_FACE_COHERENT_NORMAL_TOL * normal_norm {
            return Err(format!(
                "face-cochain measurement `{}` selected non-parallel face {} with normal [{:.6e}, {:.6e}, {:.6e}]",
                definition.name,
                face.kidx(),
                normal[0],
                normal[1],
                normal[2]
            ));
        }
        if axis_component.abs() <= TEAM13_FACE_COHERENT_NORMAL_TOL * normal_norm {
            return Err(format!(
                "face-cochain measurement `{}` selected face {} with ambiguous orientation",
                definition.name,
                face.kidx()
            ));
        }
        let orientation_sign = if axis_component >= 0.0 { 1.0 } else { -1.0 };
        let area = 0.5 * normal_norm;
        selected_area += area;
        face_count += 1;

        let Some(face_row) = b_cochain.rows.get(face.kidx()) else {
            return Err(format!(
                "B cochain operator is missing face row {}",
                face.kidx()
            ));
        };
        for (edge, value) in face_row {
            *entries.entry(*edge).or_insert(0.0) += orientation_sign * *value;
        }
    }

    if face_count == 0 {
        return Err(format!(
            "face-cochain measurement `{}` found no mesh faces on its patch; use the conforming TEAM 13 measurement-plane mesh",
            definition.name
        ));
    }
    let area_error = (selected_area - expected_area).abs();
    let area_tolerance =
        TEAM13_FACE_COHERENT_AREA_ABS_TOL.max(TEAM13_FACE_COHERENT_AREA_REL_TOL * expected_area);
    if area_error > area_tolerance {
        return Err(format!(
            "face-cochain measurement `{}` selected area {:.16e} from {} faces, expected {:.16e} (error {:.3e}, tolerance {:.3e})",
            definition.name, selected_area, face_count, expected_area, area_error, area_tolerance
        ));
    }

    let row = entries
        .into_iter()
        .filter_map(|(col, value)| {
            let scaled = value / selected_area;
            (scaled.abs() > EPS).then_some((col, scaled))
        })
        .collect();
    Ok(Team13FaceCochainSurfaceRow {
        row,
        face_count,
        selected_area,
        expected_area,
    })
}

fn measurement_face_vertices(
    coords: &MeshCoords,
    face: &SimplexHandle<'_>,
    definition: &Team13SurfaceMeasurementDefinition,
) -> Result<Option<[[f64; 3]; 3]>, String> {
    let mut vertices = [[0.0; 3]; 3];
    for (local_index, vertex) in face.iter().enumerate() {
        let point = coord3(coords, vertex)?;
        if !point_on_measurement_patch(point, definition) {
            return Ok(None);
        }
        vertices[local_index] = point;
    }
    Ok(Some(vertices))
}

fn point_on_measurement_patch(
    point: [f64; 3],
    definition: &Team13SurfaceMeasurementDefinition,
) -> bool {
    if (point[definition.normal_axis] - definition.target).abs() > TEAM13_FACE_COHERENT_COORD_TOL {
        return false;
    }
    (0..3).all(|axis| {
        let range = measurement_axis_range(definition, axis);
        let low = range.0.min(range.1) - TEAM13_FACE_COHERENT_COORD_TOL;
        let high = range.0.max(range.1) + TEAM13_FACE_COHERENT_COORD_TOL;
        point[axis] >= low && point[axis] <= high
    })
}

fn coord3(coords: &MeshCoords, vertex: usize) -> Result<[f64; 3], String> {
    if coords.dim() != 3 {
        return Err(format!("expected 3D coordinates, got dim {}", coords.dim()));
    }
    let point = coords.coord(vertex);
    Ok([point[0], point[1], point[2]])
}

fn triangle_oriented_area_normal(vertices: [[f64; 3]; 3]) -> [f64; 3] {
    let edge_a = vector_sub(vertices[1], vertices[0]);
    let edge_b = vector_sub(vertices[2], vertices[0]);
    cross(edge_a, edge_b)
}

fn vector_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vector_norm(value: [f64; 3]) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn point_b_component_row(
    cell: &Team13CellGeometry,
    b_cochain: &SparseRowOperator,
    component_index: usize,
    point: [f64; 3],
) -> Result<Vec<(usize, f64)>, String> {
    let global = FeecVector::from_column_slice(&point);
    let local = cell.coords.global2local(global.as_view());
    let mut entries = BTreeMap::<usize, f64>::new();
    for face in &cell.faces {
        let lsf = ddf::whitney::lsf::WhitneyLsf::standard(3, face.local_face.clone());
        let ambient_value = cell.coords.lift_form(&lsf.at_point(local.as_view()));
        let coeffs = ambient_value.coeffs();
        if coeffs.len() != 3 {
            return Err(format!(
                "expected 3 coefficients for reconstructed 2-form, found {}",
                coeffs.len()
            ));
        }
        let coefficients = [coeffs[2], -coeffs[1], coeffs[0]];
        let coefficient = coefficients[component_index];
        if coefficient.abs() <= EPS {
            continue;
        }
        let Some(face_row) = b_cochain.rows.get(face.face_index) else {
            return Err(format!(
                "B cochain operator is missing face row {}",
                face.face_index
            ));
        };
        for (edge, value) in face_row {
            *entries.entry(*edge).or_insert(0.0) += coefficient * *value;
        }
    }
    Ok(entries
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect())
}

fn tangential_axes(normal_axis: usize) -> Result<[usize; 2], String> {
    match normal_axis {
        0 => Ok([1, 2]),
        1 => Ok([0, 2]),
        2 => Ok([0, 1]),
        _ => Err(format!("invalid surface normal axis {normal_axis}")),
    }
}

fn measurement_axis_range(
    definition: &Team13SurfaceMeasurementDefinition,
    axis: usize,
) -> (f64, f64) {
    match axis {
        0 => definition.x_range,
        1 => definition.y_range,
        2 => definition.z_range,
        _ => unreachable!("TEAM 13 coordinates are three-dimensional"),
    }
}

fn simpson_uniform_weights(count: usize, start: f64, end: f64) -> Result<Vec<f64>, String> {
    if count < 2 {
        return Err(
            "surface quadrature requires at least two points per integrated axis".to_string(),
        );
    }
    if !start.is_finite() || !end.is_finite() || start == end {
        return Err("surface quadrature axis must have finite nonzero length".to_string());
    }
    let dx = (end - start) / (count as f64 - 1.0);
    if count == 2 {
        return Ok(vec![0.5 * dx.abs(), 0.5 * dx.abs()]);
    }
    if count % 2 == 1 {
        return Ok(simpson_odd_count_weights(count, dx.abs()));
    }

    let mut first = vec![0.0; count];
    for (index, weight) in simpson_odd_count_weights(count - 1, dx.abs())
        .into_iter()
        .enumerate()
    {
        first[index] += weight;
    }
    first[count - 2] += 0.5 * dx.abs();
    first[count - 1] += 0.5 * dx.abs();

    let mut last = vec![0.0; count];
    last[0] += 0.5 * dx.abs();
    last[1] += 0.5 * dx.abs();
    for (index, weight) in simpson_odd_count_weights(count - 1, dx.abs())
        .into_iter()
        .enumerate()
    {
        last[index + 1] += weight;
    }
    Ok(first
        .into_iter()
        .zip(last)
        .map(|(lhs, rhs)| 0.5 * (lhs + rhs))
        .collect())
}

fn simpson_odd_count_weights(count: usize, dx: f64) -> Vec<f64> {
    let mut weights = vec![0.0; count];
    for (index, weight) in weights.iter_mut().enumerate() {
        let factor = if index == 0 || index + 1 == count {
            1.0
        } else if index % 2 == 1 {
            4.0
        } else {
            2.0
        };
        *weight = dx * factor / 3.0;
    }
    weights
}

fn sample_axis_value(range: (f64, f64), count: usize, index: usize) -> Result<f64, String> {
    if index >= count {
        return Err("surface quadrature sample index exceeds count".to_string());
    }
    if count == 1 {
        return Ok(0.5 * (range.0 + range.1));
    }
    Ok(range.0 + (range.1 - range.0) * index as f64 / (count as f64 - 1.0))
}

fn surface_candidate_cells(
    cell_geometries: &[Team13CellGeometry],
    definition: &Team13SurfaceMeasurementDefinition,
) -> Result<Vec<usize>, String> {
    let tolerance = 1e-10;
    let candidates = cell_geometries
        .iter()
        .enumerate()
        .filter(|(_, cell)| {
            (0..3).all(|axis| {
                let range = measurement_axis_range(definition, axis);
                cell.bbox_max[axis] + tolerance >= range.0.min(range.1)
                    && cell.bbox_min[axis] - tolerance <= range.0.max(range.1)
            })
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(format!(
            "measurement `{}` has no candidate top-dimensional cells",
            definition.name
        ));
    }
    Ok(candidates)
}

fn locate_surface_sample_cell(
    cell_geometries: &[Team13CellGeometry],
    candidates: &[usize],
    last_cell: &mut Option<usize>,
    point: [f64; 3],
) -> Result<usize, String> {
    if let Some(index) = *last_cell {
        if point_in_cell(&cell_geometries[index], point) {
            return Ok(index);
        }
    }
    for index in candidates {
        if point_in_cell(&cell_geometries[*index], point) {
            *last_cell = Some(*index);
            return Ok(*index);
        }
    }
    Err("sample is outside all candidate cells".to_string())
}

fn point_in_cell(cell: &Team13CellGeometry, point: [f64; 3]) -> bool {
    let tolerance = 1e-9;
    if (0..3).any(|axis| {
        point[axis] < cell.bbox_min[axis] - tolerance
            || point[axis] > cell.bbox_max[axis] + tolerance
    }) {
        return false;
    }
    let global = FeecVector::from_column_slice(&point);
    let bary = cell.coords.global2bary(global.as_view());
    bary.iter()
        .all(|value| *value >= -tolerance && *value <= 1.0 + tolerance)
}

fn averaged_component_row(
    operator: &SparseRowOperator,
    cell_barycenters: &[[f64; 3]],
    definition: &Team13SurfaceMeasurementDefinition,
    mode: Team13MeasurementMode,
    legacy_band: f64,
) -> Result<Vec<(usize, f64)>, String> {
    let mut entries = BTreeMap::<usize, f64>::new();
    let mut count = 0usize;
    for cell_index in select_measurement_cells(cell_barycenters, definition, mode, legacy_band)? {
        count += 1;
        for (col, value) in &operator.rows[cell_index] {
            *entries.entry(*col).or_insert(0.0) += *value;
        }
    }
    if count == 0 {
        return Err(format!(
            "measurement `{}` selected no top-dimensional cells",
            definition.name
        ));
    }
    let scale = 1.0 / count as f64;
    Ok(entries
        .into_iter()
        .map(|(col, value)| (col, value * scale))
        .filter(|(_, value)| value.abs() > EPS)
        .collect())
}

fn select_measurement_cells(
    cell_barycenters: &[[f64; 3]],
    definition: &Team13SurfaceMeasurementDefinition,
    mode: Team13MeasurementMode,
    legacy_band: f64,
) -> Result<Vec<usize>, String> {
    match mode {
        Team13MeasurementMode::BenchmarkExact => {
            select_exact_surface_cells(cell_barycenters, definition)
        }
        Team13MeasurementMode::FaceCochain => {
            select_exact_surface_cells(cell_barycenters, definition)
        }
        Team13MeasurementMode::LegacyBand => {
            select_legacy_band_cells(cell_barycenters, definition, legacy_band)
        }
    }
}

fn select_exact_surface_cells(
    cell_barycenters: &[[f64; 3]],
    definition: &Team13SurfaceMeasurementDefinition,
) -> Result<Vec<usize>, String> {
    let tangential_candidates = cell_barycenters
        .iter()
        .enumerate()
        .filter(|(_, barycenter)| tangential_match(barycenter, definition))
        .map(|(index, barycenter)| {
            (
                index,
                (barycenter[definition.normal_axis] - definition.target).abs(),
            )
        })
        .collect::<Vec<_>>();
    let min_distance = tangential_candidates
        .iter()
        .map(|(_, distance)| *distance)
        .min_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap())
        .ok_or_else(|| {
            format!(
                "measurement `{}` selected no top-dimensional cells in exact mode",
                definition.name
            )
        })?;
    let tolerance = 1e-10_f64.max(1e-8 * min_distance.abs());
    Ok(tangential_candidates
        .into_iter()
        .filter(|(_, distance)| (*distance - min_distance).abs() <= tolerance)
        .map(|(index, _)| index)
        .collect())
}

fn select_legacy_band_cells(
    cell_barycenters: &[[f64; 3]],
    definition: &Team13SurfaceMeasurementDefinition,
    band: f64,
) -> Result<Vec<usize>, String> {
    let (x_range, y_range, z_range) = legacy_ranges(definition, band);
    let cells = cell_barycenters
        .iter()
        .enumerate()
        .filter(|(_, barycenter)| {
            in_range(barycenter[0], x_range)
                && in_range(barycenter[1], y_range)
                && in_range(barycenter[2], z_range)
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if cells.is_empty() {
        return Err(format!(
            "measurement `{}` selected no top-dimensional cells in legacy mode",
            definition.name
        ));
    }
    Ok(cells)
}

fn legacy_ranges(
    definition: &Team13SurfaceMeasurementDefinition,
    band: f64,
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    if definition.normal_axis == 2 && definition.x_range.1 <= 0.002 {
        (
            definition.x_range,
            definition.y_range,
            (definition.target - band, definition.target + band),
        )
    } else if definition.normal_axis == 0 {
        (
            (definition.target - band, definition.target + band),
            definition.y_range,
            (definition.z_range.0 - band, definition.z_range.1 + band),
        )
    } else {
        (
            (definition.x_range.0 - band, definition.x_range.1 + band),
            definition.y_range,
            (definition.target - band, definition.target + band),
        )
    }
}

fn tangential_match(
    barycenter: &[f64; 3],
    definition: &Team13SurfaceMeasurementDefinition,
) -> bool {
    match definition.normal_axis {
        0 => {
            in_range(barycenter[1], definition.y_range)
                && in_range(barycenter[2], definition.z_range)
        }
        2 => {
            in_range(barycenter[0], definition.x_range)
                && in_range(barycenter[1], definition.y_range)
        }
        _ => false,
    }
}

fn component_direction(component_index: usize, sign: f64) -> [f64; 3] {
    let mut direction = [0.0, 0.0, 0.0];
    direction[component_index] = sign;
    direction
}

fn nonzero_sign(name: &str, value: f64) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("value for `{name}` is non-finite"));
    }
    if value.abs() <= 1e-14 {
        return Err(format!(
            "value for `{name}` is too close to zero to define a sign"
        ));
    }
    Ok(if value >= 0.0 { 1.0 } else { -1.0 })
}

fn tolerant_sign(value: f64) -> i8 {
    if value > 1e-14 {
        1
    } else if value < -1e-14 {
        -1
    } else {
        0
    }
}

fn scale_row(row: &[(usize, f64)], factor: f64) -> Vec<(usize, f64)> {
    row.iter()
        .map(|(col, value)| (*col, factor * *value))
        .collect()
}

fn top_cell_barycenters(topology: &Complex, coords: &MeshCoords) -> Vec<[f64; 3]> {
    topology
        .skeleton(topology.dim())
        .handle_iter()
        .map(|cell| {
            let bary = SimplexCoords::from_simplex_and_coords(&cell, coords).barycenter();
            [bary[0], bary[1], bary[2]]
        })
        .collect()
}

fn top_cell_geometries(topology: &Complex, coords: &MeshCoords) -> Vec<Team13CellGeometry> {
    topology
        .skeleton(topology.dim())
        .handle_iter()
        .map(|cell| {
            let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
            let mut bbox_min = [f64::INFINITY; 3];
            let mut bbox_max = [f64::NEG_INFINITY; 3];
            for vertex in cell_coords.vertices.coord_iter() {
                for axis in 0..3 {
                    bbox_min[axis] = bbox_min[axis].min(vertex[axis]);
                    bbox_max[axis] = bbox_max[axis].max(vertex[axis]);
                }
            }
            let faces = cell
                .mesh_subsimps(2)
                .map(|face| Team13CellFaceGeometry {
                    face_index: face.kidx(),
                    local_face: face.relative_to(&cell),
                })
                .collect();
            Team13CellGeometry {
                coords: cell_coords,
                bbox_min,
                bbox_max,
                faces,
            }
        })
        .collect()
}

fn evaluate_sensor_reports(
    measurements: &[Team13LinearizedMeasurement],
    posterior_mean: &FeecVector,
) -> Result<Vec<Team13SensorReport>, String> {
    measurements
        .iter()
        .map(|measurement| {
            let operator = triplet_to_sparse_row_operator(&measurement.spec.operator)?;
            let posterior_prediction = operator
                .apply(&GmrfVector::from_vec(
                    posterior_mean.iter().copied().collect(),
                ))
                .map_err(|err| err.to_string())?[0];
            let observed = measurement.spec.observations[0];
            Ok(Team13SensorReport {
                name: measurement.spec.name.clone(),
                observed,
                nominal_prediction: measurement.nominal_prediction,
                posterior_prediction,
                residual: posterior_prediction - observed,
                linearization_direction: measurement.linearization_direction,
            })
        })
        .collect()
}

fn evaluate_benchmark_reports(
    topology: &Complex,
    coords: &MeshCoords,
    operators: &Team13Operators,
    nominal_a: &FeecVector,
    posterior_mean: &FeecVector,
    mode: Team13MeasurementMode,
    legacy_band: f64,
    observation_overrides: Option<&BTreeMap<String, f64>>,
) -> Result<Vec<Team13BenchmarkReport>, String> {
    let cell_barycenters = top_cell_barycenters(topology, coords);
    let cell_geometries = top_cell_geometries(topology, coords);
    let mut reports = Vec::with_capacity(TEAM13_BENCHMARK_MEASUREMENT_COUNT);

    for definition in team13_surface_measurement_definitions(observation_overrides)? {
        let row = measurement_component_row(
            topology,
            coords,
            &cell_geometries,
            &operators.b_cochain,
            &operators.b_components[definition.component_index],
            &cell_barycenters,
            &definition,
            mode,
            legacy_band,
        )?;
        reports.push(Team13BenchmarkReport {
            name: definition.name,
            observed: Some(definition.observation),
            nominal_prediction: apply_row(&row, nominal_a)?.abs(),
            posterior_prediction: apply_row(&row, posterior_mean)?.abs(),
        });
    }

    for definition in team13_point_measurement_definitions() {
        reports.push(Team13BenchmarkReport {
            name: definition.name,
            observed: definition.observation,
            nominal_prediction: point_b_norm(
                operators,
                &cell_geometries,
                definition.point,
                nominal_a,
            )?,
            posterior_prediction: point_b_norm(
                operators,
                &cell_geometries,
                definition.point,
                posterior_mean,
            )?,
        });
    }

    Ok(reports)
}

fn point_b_norm(
    operators: &Team13Operators,
    cell_geometries: &[Team13CellGeometry],
    point: [f64; 3],
    state: &FeecVector,
) -> Result<f64, String> {
    let candidates = (0..cell_geometries.len()).collect::<Vec<_>>();
    let mut last_cell = None;
    let cell_index =
        locate_surface_sample_cell(cell_geometries, &candidates, &mut last_cell, point)?;
    let bx = apply_row(
        &point_b_component_row(&cell_geometries[cell_index], &operators.b_cochain, 0, point)?,
        state,
    )?;
    let by = apply_row(
        &point_b_component_row(&cell_geometries[cell_index], &operators.b_cochain, 1, point)?,
        state,
    )?;
    let bz = apply_row(
        &point_b_component_row(&cell_geometries[cell_index], &operators.b_cochain, 2, point)?,
        state,
    )?;
    Ok((bx * bx + by * by + bz * bz).sqrt())
}

fn point_distance_squared(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    let dz = lhs[2] - rhs[2];
    dx * dx + dy * dy + dz * dz
}

fn build_vector_pushforwards(
    operators: &Team13Operators,
    posterior_mean: &FeecVector,
    result: &LinearPdeUqResult,
) -> Result<BTreeMap<String, Team13VectorPushforward>, String> {
    let mut pushforwards = BTreeMap::new();
    pushforwards.insert(
        "A".to_string(),
        vector_pushforward(
            "A",
            &operators.a_components,
            [
                A_VECTOR_X_DERIVED_NAME,
                A_VECTOR_Y_DERIVED_NAME,
                A_VECTOR_Z_DERIVED_NAME,
            ],
            posterior_mean,
            result,
        )?,
    );
    pushforwards.insert(
        "B".to_string(),
        vector_pushforward(
            "B",
            &operators.b_components,
            [
                B_VECTOR_X_DERIVED_NAME,
                B_VECTOR_Y_DERIVED_NAME,
                B_VECTOR_Z_DERIVED_NAME,
            ],
            posterior_mean,
            result,
        )?,
    );
    Ok(pushforwards)
}

fn vector_pushforward(
    name: &str,
    operators: &[SparseRowOperator],
    derived_names: [&str; 3],
    posterior_mean: &FeecVector,
    result: &LinearPdeUqResult,
) -> Result<Team13VectorPushforward, String> {
    let component_means = operators
        .iter()
        .map(|operator| apply_operator_to_feec(operator, posterior_mean))
        .collect::<Result<Vec<_>, _>>()?;
    let mean_vectors = vectors_from_components(&component_means)?;
    let prior_components = derived_names
        .iter()
        .map(|derived_name| {
            derived_variance(result, derived_name).map(|variance| variance.prior_variance.clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let posterior_components = derived_names
        .iter()
        .map(|derived_name| {
            derived_variance(result, derived_name)
                .map(|variance| variance.posterior_variance.clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prior_trace_variance = trace_components(&prior_components)?;
    let posterior_trace_variance = trace_components(&posterior_components)?;
    let trace_variance_ratio = trace_variance_ratio(&prior_components, &posterior_components)?;
    Ok(Team13VectorPushforward {
        name: name.to_string(),
        mean_vectors,
        prior_trace_variance,
        posterior_trace_variance,
        trace_variance_ratio,
    })
}

fn write_team13_outputs(
    output_dir: &PathBuf,
    topology: &Complex,
    coords: &MeshCoords,
    result: &Team13LinearSolveResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    fs::write(
        output_dir.join("field_reference.txt"),
        format!("{}\n", result.field_reference_name),
    )
    .map_err(|err| format!("failed to write field reference metadata: {err}"))?;
    let nominal_a = Cochain::new(1, result.nominal_a.clone());
    let reference_a = Cochain::new(1, result.field_reference_a.clone());
    let posterior_a = Cochain::new(1, result.posterior.posterior_mean.clone());
    let state_active_mask = Cochain::new(1, result.state_active_mask.clone());
    let prior_a_variance = Cochain::new(1, result.posterior.prior_variance.clone());
    let posterior_a_variance = Cochain::new(1, result.posterior.posterior_variance.clone());
    let a_variance_ratio = Cochain::new(1, result.a_variance_ratio.clone());
    visual_output::write_1cochain_fields(
        output_dir.join("A_nominal.vtu"),
        coords,
        topology,
        &[("nominal_mean", &nominal_a)],
    )
    .map_err(|err| format!("failed to write A nominal VTU: {err}"))?;
    visual_output::write_1cochain_fields(
        output_dir.join("A_reference.vtu"),
        coords,
        topology,
        &[("reference_mean", &reference_a)],
    )
    .map_err(|err| format!("failed to write A reference VTU: {err}"))?;
    visual_output::write_1cochain_fields(
        output_dir.join("A_posterior.vtu"),
        coords,
        topology,
        &[
            ("posterior_mean", &posterior_a),
            ("state_active_dof_mask", &state_active_mask),
            ("prior_variance", &prior_a_variance),
            ("posterior_variance", &posterior_a_variance),
            ("posterior_prior_variance_ratio", &a_variance_ratio),
        ],
    )
    .map_err(|err| format!("failed to write A posterior VTU: {err}"))?;

    let nominal_b = nominal_a.dif(topology);
    let reference_b = reference_a.dif(topology);
    let posterior_b = posterior_a.dif(topology);
    let b_variance = derived_variance(&result.posterior, B_COCHAIN_DERIVED_NAME)?;
    let b_prior = Cochain::new(2, b_variance.prior_variance.clone());
    let b_post = Cochain::new(2, b_variance.posterior_variance.clone());
    let b_ratio = Cochain::new(2, result.b_variance_ratio.clone());
    visual_output::write_cochain(
        output_dir.join("B_nominal.vtu"),
        coords,
        topology,
        &nominal_b,
        "nominal_B",
    )
    .map_err(|err| format!("failed to write B nominal VTU: {err}"))?;
    visual_output::write_cochain(
        output_dir.join("B_reference.vtu"),
        coords,
        topology,
        &reference_b,
        "reference_B",
    )
    .map_err(|err| format!("failed to write B reference VTU: {err}"))?;
    visual_output::write_cochain(
        output_dir.join("B_posterior.vtu"),
        coords,
        topology,
        &posterior_b,
        "posterior_B",
    )
    .map_err(|err| format!("failed to write B posterior VTU: {err}"))?;
    visual_output::write_cochain(
        output_dir.join("B_variance_ratio.vtu"),
        coords,
        topology,
        &b_ratio,
        "posterior_prior_variance_ratio",
    )
    .map_err(|err| format!("failed to write B variance-ratio VTU: {err}"))?;
    visual_output::write_cochain(
        output_dir.join("B_prior_variance.vtu"),
        coords,
        topology,
        &b_prior,
        "prior_variance",
    )
    .map_err(|err| format!("failed to write B prior variance VTU: {err}"))?;
    visual_output::write_cochain(
        output_dir.join("B_posterior_variance.vtu"),
        coords,
        topology,
        &b_post,
        "posterior_variance",
    )
    .map_err(|err| format!("failed to write B posterior variance VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_nominal_vector_field.vtu"),
        coords,
        topology,
        &nominal_b,
        "nominal_B_vector",
    )
    .map_err(|err| format!("failed to write nominal B vector VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_reference_vector_field.vtu"),
        coords,
        topology,
        &reference_b,
        "reference_B_vector",
    )
    .map_err(|err| format!("failed to write reference B vector VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_posterior_vector_field.vtu"),
        coords,
        topology,
        &posterior_b,
        "posterior_B_vector",
    )
    .map_err(|err| format!("failed to write B vector VTU: {err}"))?;

    for pushforward in result.vector_pushforwards.values() {
        visual_output::write_top_cell_vector_fields(
            output_dir.join(format!("{}_vector_pushforward.vtu", pushforward.name)),
            coords,
            topology,
            &format!("{}_mean_vector", pushforward.name),
            &pushforward.mean_vectors,
            &[
                (
                    "prior_trace_variance",
                    pushforward.prior_trace_variance.as_slice(),
                ),
                (
                    "posterior_trace_variance",
                    pushforward.posterior_trace_variance.as_slice(),
                ),
                (
                    "posterior_prior_trace_variance_ratio",
                    pushforward.trace_variance_ratio.as_slice(),
                ),
            ],
        )
        .map_err(|err| format!("failed to write vector pushforward VTU: {err}"))?;
    }

    let mut sensor_summary =
        "name,observed,nominal_prediction,posterior_prediction,residual,dir_x,dir_y,dir_z\n"
            .to_string();
    for report in &result.sensor_reports {
        sensor_summary.push_str(&format!(
            "{},{:.12e},{:.12e},{:.12e},{:.12e},{:.6},{:.6},{:.6}\n",
            report.name,
            report.observed,
            report.nominal_prediction,
            report.posterior_prediction,
            report.residual,
            report.linearization_direction[0],
            report.linearization_direction[1],
            report.linearization_direction[2],
        ));
    }
    fs::write(output_dir.join("sensor_summary.csv"), sensor_summary)
        .map_err(|err| format!("failed to write sensor summary: {err}"))?;

    let mut benchmark_summary =
        "name,observed,nominal_prediction,posterior_prediction,nominal_residual,posterior_residual\n"
            .to_string();
    for report in &result.benchmark_reports {
        let nominal_residual = report
            .observed
            .map(|observed| report.nominal_prediction - observed)
            .unwrap_or(f64::NAN);
        let posterior_residual = report
            .observed
            .map(|observed| report.posterior_prediction - observed)
            .unwrap_or(f64::NAN);
        benchmark_summary.push_str(&format!(
            "{},{},{:.12e},{:.12e},{},{}\n",
            report.name,
            report
                .observed
                .map(|value| format!("{value:.12e}"))
                .unwrap_or_default(),
            report.nominal_prediction,
            report.posterior_prediction,
            if nominal_residual.is_nan() {
                String::new()
            } else {
                format!("{nominal_residual:.12e}")
            },
            if posterior_residual.is_nan() {
                String::new()
            } else {
                format!("{posterior_residual:.12e}")
            },
        ));
    }
    fs::write(
        output_dir.join("benchmark_measurements.csv"),
        benchmark_summary,
    )
    .map_err(|err| format!("failed to write benchmark summary: {err}"))?;

    write_latent_input_summary(output_dir, &result.posterior.latent_inputs)?;
    write_variance_summary(output_dir, result)?;

    Ok(())
}

fn write_team13_nonlinear_outputs(
    output_dir: &PathBuf,
    topology: &Complex,
    coords: &MeshCoords,
    linear_a: &FeecVector,
    nonlinear_a: &FeecVector,
    result: &Team13NonlinearSolveResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    fs::write(
        output_dir.join("field_reference.txt"),
        "beta-zero TEAM13 linear Hodge-Laplacian mean\n",
    )
    .map_err(|err| format!("failed to write nonlinear field reference metadata: {err}"))?;

    let linear_cochain = Cochain::new(1, linear_a.clone());
    let nonlinear_cochain = Cochain::new(1, nonlinear_a.clone());
    visual_output::write_1cochain_fields(
        output_dir.join("A_nonlinear.vtu"),
        coords,
        topology,
        &[
            ("linear_proxy_mean", &linear_cochain),
            ("nonlinear_map", &nonlinear_cochain),
        ],
    )
    .map_err(|err| format!("failed to write nonlinear A VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_linear_proxy_vector_field.vtu"),
        coords,
        topology,
        &linear_cochain.dif(topology),
        "linear_proxy_B_vector",
    )
    .map_err(|err| format!("failed to write linear proxy B vector VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_nonlinear_vector_field.vtu"),
        coords,
        topology,
        &nonlinear_cochain.dif(topology),
        "nonlinear_B_vector",
    )
    .map_err(|err| format!("failed to write nonlinear B vector VTU: {err}"))?;

    let mut summary = "key,value\n".to_string();
    summary.push_str(&format!("domain,{}\n", result.domain_mode.as_str()));
    summary.push_str(&format!("vertices,{}\n", result.vertices));
    summary.push_str(&format!("edges,{}\n", result.edges));
    summary.push_str(&format!("cells,{}\n", result.cells));
    summary.push_str(&format!("active_dofs,{}\n", result.active_dofs));
    summary.push_str(&format!(
        "material_kind,{}\n",
        result.material_kind.as_str()
    ));
    summary.push_str(&format!("beta_iron,{:.12e}\n", result.beta_iron));
    summary.push_str(&format!("b_scale_tesla,{:.12e}\n", result.b_scale_tesla));
    summary.push_str(&format!("prior_kind,{}\n", result.prior_kind.as_str()));
    summary.push_str(&format!(
        "field_prior_precision_scale,{:.12e}\n",
        result.field_prior_precision_scale
    ));
    summary.push_str(&format!("prior_kappa,{:.12e}\n", result.prior_kappa));
    summary.push_str(&format!("prior_tau,{:.12e}\n", result.prior_tau));
    if let Some(first_step) = result.history.first() {
        summary.push_str(&format!(
            "step_regularization,{:?}\n",
            first_step.regularization
        ));
    }
    summary.push_str(&format!(
        "initial_residual_norm,{:.12e}\n",
        result.initial_residual_norm
    ));
    summary.push_str(&format!(
        "linear_mean_residual_norm,{:.12e}\n",
        result.linear_mean_residual_norm
    ));
    summary.push_str(&format!(
        "final_residual_norm,{:.12e}\n",
        result.final_residual_norm
    ));
    summary.push_str(&format!(
        "linear_sensor_rmse,{:.12e}\n",
        result.linear_sensor_rmse
    ));
    summary.push_str(&format!("sensor_rmse,{:.12e}\n", result.sensor_rmse));
    summary.push_str(&format!(
        "sensor_rmse_improvement_ratio,{:.12e}\n",
        result.sensor_rmse_improvement_ratio
    ));
    summary.push_str(&format!(
        "assimilated_measurements,{}\n",
        result.assimilated_measurements
    ));
    summary.push_str(&format!(
        "assembly_dimension,{}\n",
        result.assembly.dimension
    ));
    summary.push_str(&format!(
        "prior_precision_nnz,{}\n",
        result.assembly.prior_precision_nnz
    ));
    summary.push_str(&format!(
        "residual_jacobian_nnz,{}\n",
        result
            .assembly
            .term_operator_nnz(NonlinearAssemblyTermKind::Residual)
    ));
    summary.push_str(&format!(
        "residual_normal_update_nnz,{}\n",
        result
            .assembly
            .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual)
    ));
    summary.push_str(&format!(
        "linear_measurement_operator_nnz,{}\n",
        result
            .assembly
            .term_operator_nnz(NonlinearAssemblyTermKind::LinearMeasurement)
    ));
    summary.push_str(&format!(
        "linear_measurement_update_nnz,{}\n",
        result
            .assembly
            .term_precision_update_nnz(NonlinearAssemblyTermKind::LinearMeasurement)
    ));
    summary.push_str(&format!(
        "posterior_precision_nnz,{}\n",
        result.assembly.posterior_precision_nnz
    ));
    summary.push_str(&format!(
        "posterior_precision_lower_triangle_nnz,{}\n",
        result.assembly.posterior_precision_lower_triangle_nnz
    ));
    summary.push_str(&format!(
        "factor_nnz,{}\n",
        result
            .assembly
            .factor_nnz
            .unwrap_or(result.final_factorization.nnz)
    ));
    summary.push_str(&format!(
        "fill_ratio_vs_lower_triangle,{:.12e}\n",
        result
            .assembly
            .fill_ratio_vs_lower_triangle
            .unwrap_or(f64::NAN)
    ));
    summary.push_str(&format!("converged,{}\n", result.converged));
    fs::write(output_dir.join("summary.csv"), summary)
        .map_err(|err| format!("failed to write nonlinear TEAM13 summary: {err}"))?;

    let mut sensor_summary =
        "name,observed,linear_prediction,nonlinear_prediction,residual,dir_x,dir_y,dir_z\n"
            .to_string();
    for report in &result.sensor_reports {
        sensor_summary.push_str(&format!(
            "{},{:.12e},{:.12e},{:.12e},{:.12e},{:.6},{:.6},{:.6}\n",
            report.name,
            report.observed,
            report.nominal_prediction,
            report.posterior_prediction,
            report.residual,
            report.linearization_direction[0],
            report.linearization_direction[1],
            report.linearization_direction[2],
        ));
    }
    fs::write(output_dir.join("sensor_summary.csv"), sensor_summary)
        .map_err(|err| format!("failed to write nonlinear sensor summary: {err}"))?;

    let mut sensor_variance = "name,prior_variance,posterior_variance\n".to_string();
    for report in &result.sensor_variances {
        sensor_variance.push_str(&format!(
            "{},{:.12e},{:.12e}\n",
            report.name, report.prior_variance, report.posterior_variance
        ));
    }
    fs::write(output_dir.join("sensor_variance.csv"), sensor_variance)
        .map_err(|err| format!("failed to write nonlinear sensor variances: {err}"))?;

    Ok(())
}

fn write_latent_input_summary(
    output_dir: &Path,
    inputs: &[LinearPdeLatentInputPosterior],
) -> Result<(), String> {
    let mut summary = "name,index,mean,variance\n".to_string();
    for input in inputs {
        for (index, (mean, variance)) in input.mean.iter().zip(input.variance.iter()).enumerate() {
            summary.push_str(&format!(
                "{},{},{:.12e},{:.12e}\n",
                input.name, index, mean, variance
            ));
        }
    }
    fs::write(output_dir.join("latent_inputs.csv"), summary)
        .map_err(|err| format!("failed to write latent input summary: {err}"))?;
    Ok(())
}

fn write_variance_summary(
    output_dir: &Path,
    result: &Team13LinearSolveResult,
) -> Result<(), String> {
    let a_prior_full = variance_stats(result.posterior.prior_variance.iter().copied());
    let a_post_full = variance_stats(result.posterior.posterior_variance.iter().copied());
    let a_ratio_full = variance_stats(result.a_variance_ratio.iter().copied());
    let a_prior_active = variance_stats(
        result
            .posterior
            .prior_variance
            .iter()
            .zip(result.state_active_mask.iter())
            .filter_map(|(value, mask)| if *mask > 0.0 { Some(*value) } else { None }),
    );
    let a_post_active = variance_stats(
        result
            .posterior
            .posterior_variance
            .iter()
            .zip(result.state_active_mask.iter())
            .filter_map(|(value, mask)| if *mask > 0.0 { Some(*value) } else { None }),
    );
    let a_ratio_active = variance_stats(
        result
            .a_variance_ratio
            .iter()
            .zip(result.state_active_mask.iter())
            .filter_map(|(value, mask)| if *mask > 0.0 { Some(*value) } else { None }),
    );

    let mut summary = String::new();
    summary.push_str("# TEAM 13 variance summary\n");
    summary.push_str(
        "# A full-state fields include hard-constrained DOFs padded with zero variance.\n",
    );
    summary.push_str(
        "# Use `state_active_dof_mask` in `A_posterior.vtu` or the `A.active.*` lines below.\n",
    );
    append_variance_stats(&mut summary, "A.full.prior", &a_prior_full);
    append_variance_stats(&mut summary, "A.full.posterior", &a_post_full);
    append_variance_stats(&mut summary, "A.full.ratio", &a_ratio_full);
    summary.push_str(&format!(
        "A.active.count={}\nA.constrained.count={}\n",
        result
            .state_active_mask
            .iter()
            .filter(|value| **value > 0.0)
            .count(),
        result
            .state_active_mask
            .iter()
            .filter(|value| **value == 0.0)
            .count()
    ));
    append_variance_stats(&mut summary, "A.active.prior", &a_prior_active);
    append_variance_stats(&mut summary, "A.active.posterior", &a_post_active);
    append_variance_stats(&mut summary, "A.active.ratio", &a_ratio_active);

    if let Ok(b_variance) = derived_variance(&result.posterior, B_COCHAIN_DERIVED_NAME) {
        append_variance_stats(
            &mut summary,
            "B_cochain.prior",
            &variance_stats(b_variance.prior_variance.iter().copied()),
        );
        append_variance_stats(
            &mut summary,
            "B_cochain.posterior",
            &variance_stats(b_variance.posterior_variance.iter().copied()),
        );
        append_variance_stats(
            &mut summary,
            "B_cochain.ratio",
            &variance_stats(result.b_variance_ratio.iter().copied()),
        );
    }

    for (name, pushforward) in &result.vector_pushforwards {
        append_variance_stats(
            &mut summary,
            &format!("{name}.vector_trace.prior"),
            &variance_stats(pushforward.prior_trace_variance.iter().copied()),
        );
        append_variance_stats(
            &mut summary,
            &format!("{name}.vector_trace.posterior"),
            &variance_stats(pushforward.posterior_trace_variance.iter().copied()),
        );
        append_variance_stats(
            &mut summary,
            &format!("{name}.vector_trace.ratio"),
            &variance_stats(pushforward.trace_variance_ratio.iter().copied()),
        );
    }

    fs::write(output_dir.join("variance_summary.txt"), summary)
        .map_err(|err| format!("failed to write variance summary: {err}"))?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct VarianceStats {
    count: usize,
    positive_count: usize,
    zero_count: usize,
    min: f64,
    max: f64,
    mean: f64,
}

fn variance_stats(values: impl Iterator<Item = f64>) -> VarianceStats {
    let mut count = 0usize;
    let mut positive_count = 0usize;
    let mut zero_count = 0usize;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for value in values.filter(|value| value.is_finite()) {
        count += 1;
        sum += value;
        if value > 0.0 {
            positive_count += 1;
        }
        if value == 0.0 {
            zero_count += 1;
        }
        min = min.min(value);
        max = max.max(value);
    }
    if count == 0 {
        return VarianceStats {
            count: 0,
            positive_count: 0,
            zero_count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
        };
    }
    VarianceStats {
        count,
        positive_count,
        zero_count,
        min,
        max,
        mean: sum / count as f64,
    }
}

fn append_variance_stats(summary: &mut String, prefix: &str, stats: &VarianceStats) {
    summary.push_str(&format!(
        "{prefix}.count={}\n{prefix}.positive={}\n{prefix}.zero={}\n{prefix}.mean={:.12e}\n{prefix}.min={:.12e}\n{prefix}.max={:.12e}\n",
        stats.count, stats.positive_count, stats.zero_count, stats.mean, stats.min, stats.max
    ));
}

fn active_dof_mask(layout: &DofLayout) -> FeecVector {
    let mut mask = FeecVector::zeros(layout.full_dimension);
    for full_index in layout.active_dofs.iter().copied() {
        mask[full_index] = 1.0;
    }
    mask
}

fn write_team13_nominal_outputs(
    output_dir: &PathBuf,
    topology: &Complex,
    coords: &MeshCoords,
    result: &Team13LinearNominalResult,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory `{}`: {err}",
            output_dir.display()
        )
    })?;
    let nominal_a = Cochain::new(1, result.nominal_a.clone());
    visual_output::write_1cochain_fields(
        output_dir.join("A_nominal.vtu"),
        coords,
        topology,
        &[("nominal_mean", &nominal_a)],
    )
    .map_err(|err| format!("failed to write A nominal VTU: {err}"))?;

    let nominal_b = nominal_a.dif(topology);
    visual_output::write_cochain(
        output_dir.join("B_nominal.vtu"),
        coords,
        topology,
        &nominal_b,
        "nominal_B",
    )
    .map_err(|err| format!("failed to write B nominal VTU: {err}"))?;
    visual_output::write_2form_vector_field(
        output_dir.join("B_nominal_vector_field.vtu"),
        coords,
        topology,
        &nominal_b,
        "nominal_B_vector",
    )
    .map_err(|err| format!("failed to write nominal B vector VTU: {err}"))?;

    let mut benchmark_summary = "name,observed,nominal_prediction\n".to_string();
    for report in &result.benchmark_reports {
        let observed = report
            .observed
            .map(|value| value.to_string())
            .unwrap_or_default();
        benchmark_summary.push_str(&format!(
            "{},{},{}\n",
            report.name, observed, report.nominal_prediction
        ));
    }
    fs::write(
        output_dir.join("benchmark_measurements.csv"),
        benchmark_summary,
    )
    .map_err(|err| {
        format!(
            "failed to write benchmark summary `{}`: {err}",
            output_dir.join("benchmark_measurements.csv").display()
        )
    })?;

    Ok(())
}

fn point_in_coil_region(
    point: CoordRef<'_>,
    mode: Team13DomainMode,
    region: Team13CoilRegion,
) -> bool {
    if !in_range(point[2], (mode.solid_z_min(), 0.05)) {
        return false;
    }
    match region {
        Team13CoilRegion::BrickBack => {
            in_range(point[0], (-0.05, 0.05)) && in_range(point[1], (0.075, 0.10))
        }
        Team13CoilRegion::BrickFront => {
            in_range(point[0], (-0.05, 0.05)) && in_range(point[1], (-0.10, -0.075))
        }
        Team13CoilRegion::BrickLeft => {
            in_range(point[0], (-0.10, -0.075)) && in_range(point[1], (-0.05, 0.05))
        }
        Team13CoilRegion::BrickRight => {
            in_range(point[0], (0.075, 0.10)) && in_range(point[1], (-0.05, 0.05))
        }
        Team13CoilRegion::CornerRightBack => point_in_corner(point, (0.05, 0.05), 1.0, 1.0),
        Team13CoilRegion::CornerLeftBack => point_in_corner(point, (-0.05, 0.05), -1.0, 1.0),
        Team13CoilRegion::CornerLeftFront => point_in_corner(point, (-0.05, -0.05), -1.0, -1.0),
        Team13CoilRegion::CornerRightFront => point_in_corner(point, (0.05, -0.05), 1.0, -1.0),
    }
}

fn point_in_corner(point: CoordRef<'_>, center: (f64, f64), sign_x: f64, sign_y: f64) -> bool {
    let dx = point[0] - center.0;
    let dy = point[1] - center.1;
    sign_x * dx >= -EPS
        && sign_y * dy >= -EPS
        && in_range((dx * dx + dy * dy).sqrt(), (0.025, 0.05))
}

fn coil_direction(point: CoordRef<'_>, region: Team13CoilRegion) -> [f64; 3] {
    match region {
        Team13CoilRegion::BrickBack => [1.0, 0.0, 0.0],
        Team13CoilRegion::BrickFront => [-1.0, 0.0, 0.0],
        Team13CoilRegion::BrickLeft => [0.0, 1.0, 0.0],
        Team13CoilRegion::BrickRight => [0.0, -1.0, 0.0],
        Team13CoilRegion::CornerRightBack => corner_direction(point, (0.05, 0.05)),
        Team13CoilRegion::CornerLeftBack => corner_direction(point, (-0.05, 0.05)),
        Team13CoilRegion::CornerLeftFront => corner_direction(point, (-0.05, -0.05)),
        Team13CoilRegion::CornerRightFront => corner_direction(point, (0.05, -0.05)),
    }
}

fn corner_direction(point: CoordRef<'_>, center: (f64, f64)) -> [f64; 3] {
    let dx = point[0] - center.0;
    let dy = point[1] - center.1;
    let r = (dx * dx + dy * dy).sqrt();
    if r <= EPS {
        [0.0, 0.0, 0.0]
    } else {
        [dy / r, -dx / r, 0.0]
    }
}

fn is_vertical_sheet(point: CoordRef<'_>) -> bool {
    in_range(point[0], (-0.0016, 0.0016))
        && in_range(point[1], (-0.025, 0.025))
        && in_range(point[2], (-0.0632, 0.0632))
}

fn is_left_c_sheet(point: CoordRef<'_>) -> bool {
    let outer = in_range(point[0], (-0.1253, -0.0021))
        && in_range(point[1], (-0.065, -0.015))
        && in_range(point[2], (-0.0632, 0.0632));
    let inner = in_range(point[0], (-0.1221, -0.0021)) && in_range(point[2], (-0.0600, 0.0600));
    outer && !inner
}

fn is_right_c_sheet(point: CoordRef<'_>) -> bool {
    let outer = in_range(point[0], (0.0021, 0.1253))
        && in_range(point[1], (0.015, 0.065))
        && in_range(point[2], (-0.0632, 0.0632));
    let inner = in_range(point[0], (0.0021, 0.1221)) && in_range(point[2], (-0.0600, 0.0600));
    outer && !inner
}

fn in_range(value: f64, range: (f64, f64)) -> bool {
    value >= range.0 - EPS && value <= range.1 + EPS
}

fn near(value: f64, target: f64, tolerance: f64) -> bool {
    (value - target).abs() <= tolerance
}

fn columns_to_sparse_matrix(columns: &[FeecVector]) -> SparseTripletMatrix {
    let nrows = columns.first().map_or(0, FeecVector::len);
    SparseTripletMatrix::from_triplets(
        nrows,
        columns.len(),
        columns.iter().enumerate().flat_map(|(col, vector)| {
            vector
                .iter()
                .copied()
                .enumerate()
                .filter_map(move |(row, value)| {
                    (value.abs() > EPS).then_some(SparseTriplet { row, col, value })
                })
        }),
    )
}

fn dense_row_to_sparse_matrix(row: &[f64]) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        1,
        row.len(),
        row.iter().copied().enumerate().filter_map(|(col, value)| {
            (value.abs() > EPS).then_some(SparseTriplet { row: 0, col, value })
        }),
    )
}

fn columns_to_sparse_row_operator(columns: &[FeecVector]) -> Result<SparseRowOperator, String> {
    let nrows = columns.first().map_or(0, FeecVector::len);
    if columns.iter().any(|column| column.len() != nrows) {
        return Err("source-field derived columns must have matching row counts".to_string());
    }
    let mut rows = vec![Vec::new(); nrows];
    for (col, column) in columns.iter().enumerate() {
        for (row, value) in column.iter().copied().enumerate() {
            if value.abs() > EPS {
                rows[row].push((col, value));
            }
        }
    }
    SparseRowOperator::new(columns.len(), rows).map_err(|err| err.to_string())
}

fn diagonal_precision(dimension: usize, value: f64) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        dimension,
        dimension,
        (0..dimension).map(|index| SparseTriplet {
            row: index,
            col: index,
            value,
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

fn stabilize_precision(mut precision: FeecCsr) -> FeecCsr {
    if feg_infer::sparse::feec_csr_to_gmrf(&precision)
        .cholesky_sqrt_lower()
        .is_ok()
    {
        return precision;
    }
    precision = symmetrize(&precision);
    if feg_infer::sparse::feec_csr_to_gmrf(&precision)
        .cholesky_sqrt_lower()
        .is_ok()
    {
        return precision;
    }
    let diagonal = matrix_diag(&precision);
    let min_diag = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_diag = diagonal.iter().copied().map(f64::abs).fold(1.0, f64::max);
    let mut shift = if min_diag <= 0.0 {
        -min_diag + 1e-8 * max_abs_diag
    } else {
        1e-12 * max_abs_diag
    }
    .max(1e-10);
    for _ in 0..10 {
        let shifted = add_diagonal_shift(&precision, shift);
        if feg_infer::sparse::feec_csr_to_gmrf(&shifted)
            .cholesky_sqrt_lower()
            .is_ok()
        {
            return shifted;
        }
        shift *= 10.0;
    }
    add_diagonal_shift(&precision, shift)
}

fn symmetrize(matrix: &FeecCsr) -> FeecCsr {
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

fn matrix_diag(matrix: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diag[row] += *value;
        }
    }
    diag
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

fn csr_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if value.abs() > EPS {
            rows[row].push((col, *value));
        }
    }
    rows
}

fn apply_row(row: &[(usize, f64)], vector: &FeecVector) -> Result<f64, String> {
    let mut value = 0.0;
    for (col, weight) in row {
        if *col >= vector.len() {
            return Err("row references a column outside the vector".to_string());
        }
        value += *weight * vector[*col];
    }
    Ok(value)
}

fn derived_variance<'a>(
    result: &'a LinearPdeUqResult,
    name: &str,
) -> Result<&'a LinearPdeDerivedMarginalResult, String> {
    result
        .derived_variances
        .get(name)
        .ok_or_else(|| format!("missing derived variance `{name}`"))
}

fn derived_variance_ratio(result: &LinearPdeUqResult, name: &str) -> Result<FeecVector, String> {
    let variance = derived_variance(result, name)?;
    Ok(variance_ratio(
        &variance.posterior_variance,
        &variance.prior_variance,
    ))
}

fn variance_ratio(posterior: &FeecVector, prior: &FeecVector) -> FeecVector {
    assert_eq!(posterior.len(), prior.len());
    FeecVector::from_iterator(
        posterior.len(),
        posterior.iter().zip(prior.iter()).map(|(post, pre)| {
            if pre.abs() <= EPS {
                if post.abs() <= EPS {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                (post / pre).max(0.0)
            }
        }),
    )
}

fn trace_components(components: &[FeecVector]) -> Result<FeecVector, String> {
    let Some(first) = components.first() else {
        return Err("at least one component is required".to_string());
    };
    let mut trace = FeecVector::zeros(first.len());
    for component in components {
        if component.len() != first.len() {
            return Err("component lengths must match to compute a trace".to_string());
        }
        trace += component;
    }
    Ok(trace)
}

fn vectors_from_components(components: &[FeecVector]) -> Result<Vec<[f64; 3]>, String> {
    let Some(first) = components.first() else {
        return Err("at least one component is required".to_string());
    };
    let n = first.len();
    if components.iter().any(|component| component.len() != n) {
        return Err("component lengths must match to build vector values".to_string());
    }
    Ok((0..n)
        .map(|index| {
            [
                components.first().map_or(0.0, |component| component[index]),
                components.get(1).map_or(0.0, |component| component[index]),
                components.get(2).map_or(0.0, |component| component[index]),
            ]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use feg_infer::nonlinear::GaussNewtonLinearSolveStats;
    use formoniq::problems::reduced_linear::ReducedLinearPdeAssembly;
    use formoniq::reduction::{DofLayout, PrescribedDof};
    use manifold::gen::cartesian::CartesianMeshInfo;
    use std::{fs, path::Path};

    #[cfg(any(feature = "heavy-tests", feature = "external-reference-tests"))]
    use std::process::Command;

    fn point(values: [f64; 3]) -> FeecVector {
        FeecVector::from_vec(values.to_vec())
    }

    fn matrix_entry(matrix: &SparseTripletMatrix, row: usize, col: usize) -> f64 {
        matrix
            .triplet_iter()
            .filter(|(entry_row, entry_col, _)| *entry_row == row && *entry_col == col)
            .map(|(_, _, value)| value)
            .sum()
    }

    #[test]
    fn team13_bh_constants_match_selected_linear_curve() {
        assert!((NU_IRON - 342.0).abs() < 1e-12);
        assert_eq!(NU_AIR, 1.0 / MU0);
        assert!((MU_R_IRON - (1.0 / (342.0 * MU0))).abs() < 1e-12);
    }

    #[test]
    fn jacobian_sparsity_density_helpers_handle_rectangular_shapes() {
        assert_eq!(lower_triangle_capacity(3, 4), 6);
        assert_eq!(lower_triangle_capacity(5, 3), 12);
        assert!((sparse_density(2, 5, 3) - 0.3).abs() < 1e-12);
        assert!((sparse_lower_triangle_density(3, 4, 3) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn team13_ngsolve_bh_table_matches_reference_curve() {
        let law =
            Team13TabulatedReluctivityLaw::new(NU_AIR, NU_IRON, TEAM13_NGSOLVE_BH_SAMPLES).unwrap();
        assert_eq!(law.samples.first().unwrap().b_tesla, 0.0);
        assert!((law.samples[15].b_tesla - 1.0).abs() < 1e-12);
        assert!((law.samples[15].h_ampere_per_meter - 342.0).abs() < 1e-12);
        assert!((law.samples.last().unwrap().b_tesla - 5.0).abs() < 1e-12);
        assert!((law.samples.last().unwrap().h_ampere_per_meter - 2.26000019e6).abs() < 1e-2);
        assert!(law
            .samples
            .windows(2)
            .all(|window| window[0].b_tesla <= window[1].b_tesla));
        assert!(law.samples.windows(2).any(|window| {
            (window[0].b_tesla - 1.8).abs() < 1e-12
                && (window[1].b_tesla - 1.8).abs() < 1e-12
                && (window[0].h_ampere_per_meter - window[1].h_ampere_per_meter).abs() < 1e-12
        }));
        let iron_point = [0.0, 0.0, 0.02];
        for (b, expected_h) in [
            (0.001, 6.4),
            (1.45, 790.5),
            (1.85, 12024.116710041879),
            (2.0, 21018.501789959897),
            (2.25, 73258.7727153124),
        ] {
            let actual_h = law.nu(iron_point, b * b) * b;
            assert!(
                (actual_h - expected_h).abs() <= 1e-8 * expected_h.max(1.0),
                "H({b}) = {actual_h}, expected {expected_h}"
            );
        }
    }

    #[test]
    fn team13_surface_observations_use_ngsolve_benchmark_values() {
        let definitions = team13_surface_measurement_definitions(None).unwrap();
        assert_eq!(definitions.len(), TEAM13_OBSERVATION_COUNT);
        assert!((definitions[0].observation - 1.33).abs() < 1e-12);
        assert!((definitions[6].observation - 0.655).abs() < 1e-12);
        assert!((definitions[24].observation - 0.985).abs() < 1e-12);
        let g_047 = team13_published_steel_observations(Team13PublishedSteelGap::G047);
        assert!((g_047[0] - 1.354).abs() < 1e-12);
        assert!((g_047[24] - 0.995).abs() < 1e-12);
    }

    #[test]
    fn team13_material_log_scale_sigma_points_match_requested_variance() {
        let std = 0.10;
        let nodes = team13_material_log_scale_sigma_points(std).unwrap();
        let weight_sum = nodes.iter().map(|node| node.weight).sum::<f64>();
        let mean = nodes
            .iter()
            .map(|node| node.weight * node.log_iron_nu_scale)
            .sum::<f64>();
        let variance = nodes
            .iter()
            .map(|node| node.weight * (node.log_iron_nu_scale - mean).powi(2))
            .sum::<f64>();
        assert!((weight_sum - 1.0).abs() <= 1e-12);
        assert!(mean.abs() <= 1e-12);
        assert!((variance - std * std).abs() <= 1e-12);
    }

    #[test]
    fn team13_material_prior_gap_difference_target_uses_published_tables() {
        let rms = team13_published_steel_gap_difference_rms();
        assert!(rms > 1.0e-2);
        assert!(rms < 2.0e-2);

        let mut config = Team13JointMaterialUqConfig::default();
        config.material_prior_calibration_target =
            Team13MaterialPriorCalibrationTarget::PublishedGapDifference;
        assert!((team13_material_prior_target_steel_rms(&config).unwrap() - rms).abs() <= 1e-15);

        config.material_prior_target_steel_rms_tesla = Some(2.5e-3);
        assert!((team13_material_prior_target_steel_rms(&config).unwrap() - 2.5e-3).abs() <= 1e-15);
    }

    #[test]
    fn team13_material_prior_calibration_matches_target_rms_and_clamps() {
        let (std, unclamped) =
            calibrated_material_prior_std_from_unit_rms(0.1, 2.0e-2, 1.0e-1, 1.0e-6, 0.5);
        assert!((std - 0.2).abs() <= 1e-15);
        assert!((unclamped - 0.2).abs() <= 1e-15);

        let (std, unclamped) =
            calibrated_material_prior_std_from_unit_rms(0.1, 1.0, 1.0e-1, 1.0e-6, 0.5);
        assert!((std - 0.5).abs() <= 1e-15);
        assert!((unclamped - 10.0).abs() <= 1e-15);

        let (std, unclamped) =
            calibrated_material_prior_std_from_unit_rms(0.1, 1.0, 0.0, 1.0e-6, 0.5);
        assert!((std - 0.1).abs() <= 1e-15);
        assert!((unclamped - 0.1).abs() <= 1e-15);
    }

    #[test]
    fn team13_material_only_implicit_state_sensitivity_solves_linearized_pde() {
        let evaluation = NonlinearResidualEvaluation {
            residual: vec![0.0, 0.0],
            jacobian: SparseTripletMatrix::from_triplets(
                2,
                5,
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
                    SparseTriplet {
                        row: 0,
                        col: 2,
                        value: 4.0,
                    },
                    SparseTriplet {
                        row: 1,
                        col: 2,
                        value: 8.0,
                    },
                    SparseTriplet {
                        row: 0,
                        col: 3,
                        value: 6.0,
                    },
                    SparseTriplet {
                        row: 1,
                        col: 3,
                        value: 9.0,
                    },
                ],
            ),
        };
        let sensitivities =
            team13_state_sensitivities_from_augmented_evaluation(&evaluation, 2).unwrap();
        assert_eq!(sensitivities.len(), 3);
        assert!((sensitivities[0][0] + 2.0).abs() <= 1e-12);
        assert!((sensitivities[0][1] + 8.0 / 3.0).abs() <= 1e-12);
        assert!((sensitivities[1][0] + 3.0).abs() <= 1e-12);
        assert!((sensitivities[1][1] + 3.0).abs() <= 1e-12);
        assert!(sensitivities[2].iter().all(|value| value.abs() <= 1e-12));
    }

    #[test]
    fn team13_material_only_smooth_observation_jacobian_matches_finite_difference() {
        let rows = vec![vec![(0usize, 2.0), (1usize, -1.0)]];
        let signed_prediction = 0.7;
        let smoothing = 1.0e-3;
        let state_sensitivities = vec![vec![0.5, -0.25], vec![-0.1, 0.2], vec![0.0, 0.3]];
        let jacobian = team13_smooth_magnitude_jacobian_from_state_sensitivities(
            &rows,
            &[signed_prediction],
            &state_sensitivities,
            smoothing,
        )
        .unwrap();

        let eps = 1.0e-6;
        for theta in 0..3 {
            let dz = rows[0]
                .iter()
                .map(|(col, value)| *value * state_sensitivities[theta][*col])
                .sum::<f64>();
            let plus = ((signed_prediction + eps * dz).powi(2) + smoothing * smoothing).sqrt();
            let minus = ((signed_prediction - eps * dz).powi(2) + smoothing * smoothing).sqrt();
            let finite_difference = (plus - minus) / (2.0 * eps);
            assert!(
                (jacobian[0][theta] - finite_difference).abs()
                    <= 1.0e-8 * finite_difference.abs().max(1.0),
                "theta {theta}: analytic {} finite-difference {}",
                jacobian[0][theta],
                finite_difference
            );
        }
    }

    #[test]
    fn team13_identifiable_material_svd_eigendecomposition_is_sorted_and_orthonormal() {
        let angle = 0.37_f64;
        let c = angle.cos();
        let s = angle.sin();
        let eigenvalues = [9.0, 4.0, 1.0];
        let rotation = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let mut matrix = [[0.0; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                for k in 0..3 {
                    matrix[row][col] += rotation[row][k] * eigenvalues[k] * rotation[col][k];
                }
            }
        }
        let decomposition = symmetric_3x3_eigendecomposition(matrix);
        assert!((decomposition.eigenvalues[0] - 9.0).abs() <= 1.0e-10);
        assert!((decomposition.eigenvalues[1] - 4.0).abs() <= 1.0e-10);
        assert!((decomposition.eigenvalues[2] - 1.0).abs() <= 1.0e-10);
        for mode in 0..3 {
            assert!((vector3_norm(decomposition.eigenvectors[mode]) - 1.0).abs() <= 1.0e-12);
            let pivot = decomposition.eigenvectors[mode]
                .iter()
                .enumerate()
                .max_by(|left, right| {
                    left.1
                        .abs()
                        .partial_cmp(&right.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index)
                .unwrap();
            assert!(decomposition.eigenvectors[mode][pivot] >= 0.0);
        }
        for left in 0..3 {
            for right in left + 1..3 {
                let dot = decomposition.eigenvectors[left][0]
                    * decomposition.eigenvectors[right][0]
                    + decomposition.eigenvectors[left][1] * decomposition.eigenvectors[right][1]
                    + decomposition.eigenvectors[left][2] * decomposition.eigenvectors[right][2];
                assert!(dot.abs() <= 1.0e-12);
            }
        }
    }

    #[test]
    fn team13_identifiable_material_rank_filter_drops_null_mode() {
        let jacobian = vec![
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let basis = team13_identifiable_material_basis(&jacobian, 1.0e-3, 1.0e-12).unwrap();
        assert_eq!(basis.singular_values, [3.0, 2.0, 0.0]);
        assert_eq!(basis.retained_modes.len(), 2);
        assert!(basis.mode_reports[0].retained);
        assert!(basis.mode_reports[1].retained);
        assert!(!basis.mode_reports[2].retained);
    }

    #[test]
    fn team13_identifiable_material_basis_transform_matches_dense_product() {
        let jacobian = vec![[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]];
        let basis = vec![[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]];
        let transformed = team13_transform_theta_jacobian_to_eta(&jacobian, &basis).unwrap();
        assert_eq!(transformed.len(), 2);
        for row in 0..2 {
            for col in 0..2 {
                let expected = jacobian[row][0] * basis[col][0]
                    + jacobian[row][1] * basis[col][1]
                    + jacobian[row][2] * basis[col][2];
                assert!((transformed[row][col] - expected).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn team13_identifiable_material_perturbation_sizing_hits_half_gap_rms() {
        let jacobian = vec![
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let basis = team13_identifiable_material_basis(&jacobian, 1.0e-3, 1.0e-12).unwrap();
        let gap_rms = team13_published_steel_gap_difference_rms();
        let report =
            team13_identifiable_baseline_perturbation(&basis, &jacobian, 0.5 * gap_rms, gap_rms)
                .unwrap();
        assert!((report.target_rms_tesla - 0.5 * gap_rms).abs() <= 1.0e-15);
        assert!(
            (report.achieved_linearized_rms_tesla - report.target_rms_tesla).abs()
                <= 1.0e-12 * report.target_rms_tesla.max(1.0)
        );
        assert!(report.eta_bias[0] > 0.0);
    }

    #[test]
    fn team13_identifiable_material_eta_to_theta_and_recovery_fraction() {
        let basis = vec![[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]];
        let theta_bias = [0.1, -0.2, 0.3];
        let theta = team13_eta_to_theta(theta_bias, &basis, &[-0.1, 0.5]).unwrap();
        assert!((theta[0] - 0.0).abs() <= 1e-12);
        assert!((theta[1] - 0.1).abs() <= 1e-12);
        assert!((theta[2] - 0.7).abs() <= 1e-12);
        assert!((team13_identifiable_recovery_fraction(-0.04, 0.05) - 0.8).abs() <= 1e-12);
        assert_eq!(team13_identifiable_recovery_fraction(1.0, 0.0), 0.0);
    }

    #[test]
    fn team13_two_factor_variance_splits_gap_and_material_effects() {
        let samples = [
            Team13WeightedPatchSample {
                gap_label: "g1",
                material_label: "m1",
                weight: 0.25,
                prediction: 10.0,
                within_variance: 1.0,
            },
            Team13WeightedPatchSample {
                gap_label: "g1",
                material_label: "m2",
                weight: 0.25,
                prediction: 12.0,
                within_variance: 1.0,
            },
            Team13WeightedPatchSample {
                gap_label: "g2",
                material_label: "m1",
                weight: 0.25,
                prediction: 14.0,
                within_variance: 1.0,
            },
            Team13WeightedPatchSample {
                gap_label: "g2",
                material_label: "m2",
                weight: 0.25,
                prediction: 16.0,
                within_variance: 1.0,
            },
        ];
        let terms = team13_two_factor_weighted_variance(&samples).unwrap();
        assert!((terms.mean - 13.0).abs() <= 1e-12);
        assert!((terms.expected_within - 1.0).abs() <= 1e-12);
        assert!((terms.between_gap - 4.0).abs() <= 1e-12);
        assert!((terms.between_material - 1.0).abs() <= 1e-12);
        assert!(terms.interaction.abs() <= 1e-12);
        assert!((terms.total - 6.0).abs() <= 1e-12);
    }

    #[test]
    fn team13_joint_material_prior_appends_independent_theta_block() {
        let state_prior = GaussianPriorSpec {
            mean: vec![1.0, -1.0],
            precision: SparseTripletMatrix::from_triplets(
                2,
                2,
                [
                    SparseTriplet {
                        row: 0,
                        col: 0,
                        value: 4.0,
                    },
                    SparseTriplet {
                        row: 1,
                        col: 1,
                        value: 9.0,
                    },
                ],
            ),
        };
        let joint = append_independent_material_prior(state_prior, 3, 25.0).unwrap();
        assert_eq!(joint.mean, vec![1.0, -1.0, 0.0, 0.0, 0.0]);
        assert_eq!(joint.precision.nrows(), 5);
        assert_eq!(joint.precision.ncols(), 5);
        assert!((matrix_entry(&joint.precision, 2, 2) - 25.0).abs() <= 1e-12);
        assert!((matrix_entry(&joint.precision, 4, 4) - 25.0).abs() <= 1e-12);
    }

    #[test]
    fn team13_joint_material_augmented_jacobian_uses_analytic_theta_columns() {
        let anchors = [0.02, 0.08, 0.14];
        let theta = [0.1, -0.05, 0.2];
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            1,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let nominal_material =
            build_team13_tabulated_material_with_log_h_shape(anchors, [0.0; 3]).unwrap();
        let model = build_reduced_vector_potential_magnetostatic_3d(
            &topology,
            &coords,
            NonlinearMagnetostaticAssemblyConfig::new(
                nominal_material,
                EssentialBoundarySpec::default(),
            ),
        )
        .unwrap();
        let state = (0..model.reduced_dimension())
            .map(|index| 0.01 * ((index % 7) as f64 - 3.0))
            .collect::<Vec<_>>();
        let augmented_model = Team13MaterialShapeAugmentedResidualModel::new(
            model.clone(),
            anchors,
            model.layout().clone(),
        )
        .unwrap();
        let mut joint_state = state.clone();
        joint_state.extend(theta);

        let material = build_team13_tabulated_material_with_log_h_shape(anchors, theta).unwrap();
        let base = model
            .residual_and_jacobian_with_material(&material, &state)
            .unwrap();
        let base_jacobian = csr_to_triplet(&base.jacobian);
        let material_columns = model
            .material_sensitivity_columns_with_material(
                &material,
                &state,
                3,
                |material, point, s| Ok(material.d_nu_d_log_h_shape_values(point, s).to_vec()),
            )
            .unwrap();
        let material_columns = csr_to_triplet(&material_columns);
        let augmented = augmented_model.residual_and_jacobian(&joint_state).unwrap();

        assert_eq!(augmented.jacobian.nrows(), base.jacobian.nrows());
        assert_eq!(augmented.jacobian.ncols(), model.reduced_dimension() + 3);
        for row in 0..base.jacobian.nrows() {
            assert!((augmented.residual[row] - base.residual[row]).abs() <= 1e-12);
            for col in 0..model.reduced_dimension() {
                assert!(
                    (matrix_entry(&augmented.jacobian, row, col)
                        - matrix_entry(&base_jacobian, row, col))
                    .abs()
                        <= 1e-12
                );
            }
            for theta_col in 0..3 {
                assert!(
                    (matrix_entry(
                        &augmented.jacobian,
                        row,
                        model.reduced_dimension() + theta_col
                    ) - matrix_entry(&material_columns, row, theta_col))
                    .abs()
                        <= 1e-12
                );
            }
        }
    }

    #[test]
    fn team13_identifiable_joint_material_augmented_jacobian_uses_projected_eta_columns() {
        let anchors = [0.02, 0.08, 0.14];
        let theta_bias = [0.02, -0.01, 0.03];
        let basis = vec![[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]];
        let eta = [0.04, -0.05];
        let theta = team13_eta_to_theta(theta_bias, &basis, &eta).unwrap();
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            1,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let nominal_material =
            build_team13_tabulated_material_with_log_h_shape(anchors, [0.0; 3]).unwrap();
        let model = build_reduced_vector_potential_magnetostatic_3d(
            &topology,
            &coords,
            NonlinearMagnetostaticAssemblyConfig::new(
                nominal_material,
                EssentialBoundarySpec::default(),
            ),
        )
        .unwrap();
        let state = (0..model.reduced_dimension())
            .map(|index| 0.01 * ((index % 7) as f64 - 3.0))
            .collect::<Vec<_>>();
        let identifiable_model = Team13IdentifiableJointMaterialResidualModel::new(
            model.clone(),
            anchors,
            theta_bias,
            basis.clone(),
            model.layout().clone(),
        )
        .unwrap();
        let mut joint_state = state.clone();
        joint_state.extend(eta);

        let material = build_team13_tabulated_material_with_log_h_shape(anchors, theta).unwrap();
        let base = model
            .residual_and_jacobian_with_material(&material, &state)
            .unwrap();
        let base_jacobian = csr_to_triplet(&base.jacobian);
        let material_columns = model
            .material_sensitivity_columns_with_material(
                &material,
                &state,
                3,
                |material, point, s| Ok(material.d_nu_d_log_h_shape_values(point, s).to_vec()),
            )
            .unwrap();
        let material_columns = csr_to_triplet(&material_columns);
        let identifiable = identifiable_model
            .residual_and_jacobian(&joint_state)
            .unwrap();

        assert_eq!(identifiable.jacobian.nrows(), base.jacobian.nrows());
        assert_eq!(
            identifiable.jacobian.ncols(),
            model.reduced_dimension() + basis.len()
        );
        for row in 0..base.jacobian.nrows() {
            assert!((identifiable.residual[row] - base.residual[row]).abs() <= 1e-12);
            for col in 0..model.reduced_dimension() {
                assert!(
                    (matrix_entry(&identifiable.jacobian, row, col)
                        - matrix_entry(&base_jacobian, row, col))
                    .abs()
                        <= 1e-12
                );
            }
            for eta_col in 0..basis.len() {
                let expected = (0..3)
                    .map(|theta_col| {
                        matrix_entry(&material_columns, row, theta_col) * basis[eta_col][theta_col]
                    })
                    .sum::<f64>();
                assert!(
                    (matrix_entry(
                        &identifiable.jacobian,
                        row,
                        model.reduced_dimension() + eta_col
                    ) - expected)
                        .abs()
                        <= 1e-12
                );
            }
        }
    }

    #[test]
    fn team13_material_explained_variance_uses_theta_cross_covariance() {
        let theta_covariance = [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]];
        let inverse = invert_3x3(theta_covariance).unwrap();
        let covariance_with_output = [2.0, 3.0, 4.0];
        let explained = quadratic_form_3(covariance_with_output, inverse);
        assert!((explained - 3.0).abs() <= 1e-12);
    }

    #[test]
    fn team13_joint_material_objective_components_sum_matches_reference_objective() {
        let state_prior = GaussianPriorSpec {
            mean: vec![1.0, -1.0],
            precision: SparseTripletMatrix::from_triplets(
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
                        value: 4.0,
                    },
                ],
            ),
        };
        let measurement = LinearGaussianMeasurementSpec {
            name: "steel".to_string(),
            operator: SparseTripletMatrix::from_triplets(
                1,
                3,
                [
                    SparseTriplet {
                        row: 0,
                        col: 0,
                        value: 1.0,
                    },
                    SparseTriplet {
                        row: 0,
                        col: 1,
                        value: 2.0,
                    },
                ],
            ),
            observations: vec![1.0],
            bias: vec![0.0],
            variance: 4.0,
        };
        let components = team13_joint_material_objective_components(
            "joint_material",
            &state_prior,
            Some(0.1),
            &[2.0, 1.0, 0.2],
            &[measurement],
            &[
                NonlinearResidualReport {
                    name: "pde".to_string(),
                    residual: vec![3.0],
                    weighted_norm: 3.0,
                },
                NonlinearResidualReport {
                    name: "team13_joint_material_steel_smooth_magnitude".to_string(),
                    residual: vec![2.0],
                    weighted_norm: 2.0,
                },
            ],
            Some(18.625),
        )
        .unwrap();
        assert!((components.prior_state - 9.0).abs() <= 1e-12);
        assert!((components.prior_material - 2.0).abs() <= 1e-12);
        assert!((components.steel_observation - 3.125).abs() <= 1e-12);
        assert!((components.pde_residual - 4.5).abs() <= 1e-12);
        assert!((components.total - 18.625).abs() <= 1e-12);
        assert!(components.solver_objective_gap.unwrap().abs() <= 1e-12);
    }

    #[test]
    fn team13_joint_material_smooth_steel_model_appends_zero_theta_columns() {
        let model = SmoothGroupedNormLinearResidualModel::new(
            SparseTripletMatrix::from_triplets(
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
                        value: -3.0,
                    },
                ],
            ),
            vec![0.25, -0.5],
            vec![SmoothGroupedNormObservation {
                name: "steel".to_string(),
                samples: vec![SmoothGroupedNormSample {
                    rows: vec![0, 1],
                    weight: 1.0,
                }],
            }],
            1.0e-8,
        )
        .unwrap();
        let augmented = append_zero_theta_columns_to_smooth_grouped_norm_model(&model, 3).unwrap();
        assert_eq!(augmented.state_dimension(), 5);
        assert_eq!(augmented.residual_dimension(), model.residual_dimension());

        let state = vec![0.4, -0.2];
        let mut joint_state = state.clone();
        joint_state.extend([0.7, -0.8, 0.9]);
        let base = model.residual_and_jacobian(&state).unwrap();
        let lifted = augmented.residual_and_jacobian(&joint_state).unwrap();
        assert_eq!(lifted.residual, base.residual);
        for row in 0..lifted.jacobian.nrows() {
            for col in 0..model.state_dimension() {
                assert!(
                    (matrix_entry(&lifted.jacobian, row, col)
                        - matrix_entry(&base.jacobian, row, col))
                    .abs()
                        <= 1e-12
                );
            }
            for col in model.state_dimension()..augmented.state_dimension() {
                assert_eq!(matrix_entry(&lifted.jacobian, row, col), 0.0);
            }
        }
    }

    #[test]
    fn team13_joint_material_diagnostic_csvs_include_fixed_and_joint_rows() {
        let result = fake_joint_material_result_for_diagnostics();
        let history = team13_joint_material_history_csv(&result);
        assert_eq!(history.lines().count(), 3);
        assert!(history.contains("fixed_material,0,"));
        assert!(history.contains("joint_material,0,"));

        let comparison = team13_joint_material_fixed_vs_joint_comparison_csv(&result);
        assert!(comparison.contains("fixed_material,true"));
        assert!(comparison.contains("joint_material,false"));
        assert_eq!(
            result.fixed_material_solve.linear_measurement_rows,
            result.joint_material_solve.linear_measurement_rows
        );
        assert_eq!(
            result.fixed_material_solve.prior_kind,
            result.joint_material_solve.prior_kind
        );
        assert_eq!(
            result.fixed_material_solve.pde_residual_kind,
            result.joint_material_solve.pde_residual_kind
        );

        let residual_terms = team13_joint_material_residual_terms_csv(&result);
        assert!(residual_terms.contains("fixed_material,pde"));
        assert!(residual_terms.contains("joint_material,pde"));
        assert!(
            residual_terms.contains("joint_material,team13_joint_material_steel_smooth_magnitude")
        );

        let objectives = team13_joint_material_objective_components_csv(&result);
        assert!(objectives.contains("fixed_material"));
        assert!(objectives.contains("joint_material"));

        let calibration = team13_joint_material_prior_calibration_csv(&result);
        assert!(calibration.contains("steel-prior-predictive-rms"));
        assert!(calibration.contains("published-gap-difference"));
    }

    fn fake_joint_material_result_for_diagnostics() -> Team13JointMaterialUqResult {
        let fixed = fake_joint_material_solve("fixed_material", true, 0);
        let joint = fake_joint_material_solve("joint_material", false, 3);
        Team13JointMaterialUqResult {
            domain_mode: Team13DomainMode::HalfZNonnegative,
            mesh_path: PathBuf::from("mesh.msh"),
            observed_steel_gap: Team13PublishedSteelGap::G052,
            vertices: 0,
            edges: 0,
            cells: 0,
            active_dofs: 2,
            boundary_edge_dofs: 0,
            material_anchor_b_tesla: [0.5, 1.7, 2.3],
            material_prior_std: 0.1,
            material_prior_calibration: Team13MaterialPriorCalibrationReport {
                mode: Team13MaterialPriorCalibrationMode::SteelPriorPredictiveRms,
                target: Team13MaterialPriorCalibrationTarget::PublishedGapDifference,
                target_steel_rms_tesla: Some(0.01),
                configured_material_prior_std: 0.1,
                material_prior_std: 0.1,
                unclamped_material_prior_std: Some(0.1),
                material_prior_std_floor: 1.0e-6,
                material_prior_std_ceiling: 0.5,
                unit_theta_steel_rms_tesla: Some(0.1),
                sensitivity_frobenius_norm_tesla: Some(0.5),
                max_abs_sensitivity_tesla: Some(0.2),
                theta_column_norms_tesla: [0.3, 0.4, 0.0],
                steel_row_count: TEAM13_OBSERVATION_COUNT,
            },
            magnitude_smoothing_tesla: 1.0e-8,
            deterministic_converged: true,
            deterministic_residual_norm: 0.0,
            posterior_converged: joint.converged,
            posterior_residual_norm: joint.posterior_residual_norm,
            posterior_precision_nnz: joint.posterior_precision_nnz,
            posterior_factor_nnz: joint.posterior_factor_nnz,
            material_posterior: Vec::new(),
            material_posterior_covariance: [[0.0; 3]; 3],
            material_posterior_correlation: [[0.0; 3]; 3],
            bh_curve_bands: Vec::new(),
            steel_patch_reports: Vec::new(),
            fixed_material_solve: fixed,
            joint_material_solve: joint,
            output_dir: None,
        }
    }

    fn fake_joint_material_solve(
        label: &str,
        converged: bool,
        theta_dimension: usize,
    ) -> Team13JointMaterialSolveDiagnostics {
        Team13JointMaterialSolveDiagnostics {
            label: label.to_string(),
            state_dimension: 2,
            theta_dimension,
            prior_kind: Team13MapParityPriorKind::WeakRidge,
            pde_residual_kind: Team13MapParityPdeResidualKind::UngaugedCurl,
            linear_measurement_count: 1,
            linear_measurement_rows: 25,
            converged,
            posterior_residual_norm: if converged { 1.0 } else { 2.0 },
            posterior_precision_nnz: 11,
            posterior_factor_nnz: 17,
            history: vec![GaussNewtonIteration {
                iteration: 0,
                objective: 10.0,
                trial_objective: 5.0,
                gradient_norm: 4.0,
                step_norm: 3.0,
                alpha: 1.0,
                regularization: GaussNewtonStepRegularization::AdaptiveLevenbergMarquardt,
                regularization_lambda: 1e-4,
                residual_norm: 2.0,
                linear_solve: GaussNewtonLinearSolveStats {
                    mode: GaussNewtonLinearSolveMode::DirectCholesky,
                    iterations: 1,
                    final_residual_norm: 0.0,
                    converged: true,
                    factor_nnz: Some(17),
                },
            }],
            diagnostics: GaussNewtonRunDiagnostics {
                accepted_iterations: 1,
                line_search_residual_evaluations: 1,
                step_solve_attempts: 1,
                cholesky_factor_attempts: 1,
                cholesky_factor_successes: 1,
                ..GaussNewtonRunDiagnostics::default()
            },
            assembly: NonlinearAssemblyStats {
                dimension: 2 + theta_dimension,
                prior_precision_nnz: 2,
                prior_precision_lower_triangle_nnz: 2,
                terms: Vec::new(),
                posterior_precision_nnz: 11,
                posterior_precision_lower_triangle_nnz: 11,
                factor_nnz: Some(17),
                fill_ratio_vs_lower_triangle: Some(17.0 / 11.0),
            },
            final_factorization: LaplaceFactorizationStats {
                nnz: 17,
                elapsed_seconds: 0.1,
            },
            final_residuals: vec![
                NonlinearResidualReport {
                    name: "pde".to_string(),
                    residual: vec![1.0, 2.0],
                    weighted_norm: 3.0,
                },
                NonlinearResidualReport {
                    name: "team13_joint_material_steel_smooth_magnitude".to_string(),
                    residual: vec![0.5],
                    weighted_norm: 1.0,
                },
            ],
            final_step: Team13JointMaterialFinalStepDiagnostic {
                available: true,
                error: None,
                objective: 5.0,
                weighted_residual_norm: 2.0,
                gradient_norm: 4.0,
                step_norm: 3.0,
                directional_derivative: -1.0,
                accepted_alpha: Some(1.0),
                accepted_objective: Some(4.0),
                regularization_lambda: 1e-4,
                linear_solve_absolute_residual_norm: 0.0,
                linear_solve_relative_residual_norm: 0.0,
            },
            objective_components: Team13JointMaterialObjectiveComponents {
                label: label.to_string(),
                prior_state: 1.0,
                prior_material: if theta_dimension == 0 { 0.0 } else { 0.5 },
                steel_observation: 2.0,
                pde_residual: 3.0,
                total: 6.0 + if theta_dimension == 0 { 0.0 } else { 0.5 },
                solver_objective: Some(5.0),
                solver_objective_gap: Some(1.0),
            },
        }
    }

    #[test]
    fn team13_map_parity_prior_and_steel_observation_modes_parse() {
        assert_eq!(
            "exact-potential"
                .parse::<Team13MapParityPriorKind>()
                .unwrap(),
            Team13MapParityPriorKind::ExactPotential
        );
        assert_eq!(
            "ordinary-matern-alpha2"
                .parse::<Team13MapParityPriorKind>()
                .unwrap(),
            Team13MapParityPriorKind::OrdinaryMaternAlpha2
        );
        assert_eq!(
            "weak-ridge".parse::<Team13MapParityPriorKind>().unwrap(),
            Team13MapParityPriorKind::WeakRidge
        );
        assert!("spectral-corrected"
            .parse::<Team13MapParityPriorKind>()
            .is_err());

        assert_eq!(
            "ngsolve-style"
                .parse::<Team13SteelObservationQuadratureMode>()
                .unwrap(),
            Team13SteelObservationQuadratureMode::NgsolveStyle
        );
        assert_eq!(
            "face-cochain"
                .parse::<Team13SteelObservationQuadratureMode>()
                .unwrap(),
            Team13SteelObservationQuadratureMode::FaceCochain
        );
        assert!("point-reconstruction"
            .parse::<Team13SteelObservationQuadratureMode>()
            .is_err());

        assert_eq!(
            "gauge-fixed"
                .parse::<Team13MapParityPdeResidualKind>()
                .unwrap(),
            Team13MapParityPdeResidualKind::GaugeFixed
        );
        assert_eq!(
            "ungauged-curl"
                .parse::<Team13MapParityPdeResidualKind>()
                .unwrap(),
            Team13MapParityPdeResidualKind::UngaugedCurl
        );
        assert!("bad".parse::<Team13MapParityPdeResidualKind>().is_err());
    }

    #[test]
    fn team13_truth_cache_key_distinguishes_solver_inputs() {
        let mesh = b"mesh-a";
        let mut config = Team13MapParityConfig::default();
        let base = team13_truth_cache_key(mesh, &config, 3);

        config.ampere_turns = 1200.0;
        assert_ne!(base, team13_truth_cache_key(mesh, &config, 3));

        config = Team13MapParityConfig::default();
        config.domain_mode = Team13DomainMode::Full;
        assert_ne!(base, team13_truth_cache_key(mesh, &config, 3));

        config = Team13MapParityConfig::default();
        config.truth_max_iterations += 1;
        assert_ne!(base, team13_truth_cache_key(mesh, &config, 3));

        config = Team13MapParityConfig::default();
        assert_ne!(base, team13_truth_cache_key(b"mesh-b", &config, 3));
        assert_ne!(base, team13_truth_cache_key(mesh, &config, 4));
    }

    #[test]
    fn team13_truth_cache_rejects_wrong_dimension_and_high_residual() {
        struct IdentityResidual;

        impl NonlinearResidualModel for IdentityResidual {
            fn state_dimension(&self) -> usize {
                1
            }

            fn residual_dimension(&self) -> usize {
                1
            }

            fn residual(&self, state: &[f64]) -> Result<Vec<f64>, String> {
                Ok(vec![state[0]])
            }

            fn residual_and_jacobian(
                &self,
                state: &[f64],
            ) -> Result<feg_core::NonlinearResidualEvaluation, String> {
                let mut jacobian = SparseTripletMatrix::new(1, 1);
                jacobian.push(0, 0, 1.0);
                Ok(feg_core::NonlinearResidualEvaluation {
                    residual: vec![state[0]],
                    jacobian,
                })
            }
        }

        let cache_path = std::env::temp_dir().join(format!(
            "team13-truth-cache-test-{}-{}.bin",
            std::process::id(),
            1
        ));
        let result = feg_infer::nonlinear::SquareNewtonResult {
            solution: vec![2.0],
            residual: vec![2.0],
            residual_norm: 2.0,
            history: Vec::new(),
            converged: true,
        };
        write_team13_truth_cache(&cache_path, &result).unwrap();

        assert!(parse_team13_truth_cache(&fs::read(&cache_path).unwrap(), 2).is_err());
        assert!(
            try_load_team13_truth_cache(&cache_path, &IdentityResidual, 1, 0.1).is_none(),
            "high-residual cached truth should be ignored"
        );
        let _ = fs::remove_file(cache_path);
    }

    #[test]
    fn team13_published_steel_tables_match_ngsolve_reference() {
        let expected_g052 = [
            1.33, 1.329, 1.286, 1.225, 1.129, 0.985, 0.655, 0.259, 0.453, 0.554, 0.637, 0.698,
            0.755, 0.809, 0.901, 0.945, 0.954, 0.956, 0.960, 0.965, 0.970, 0.974, 0.981, 0.984,
            0.985,
        ];
        let expected_g047 = [
            1.354, 1.339, 1.304, 1.245, 1.138, 0.982, 0.674, 0.263, 0.451, 0.563, 0.641, 0.706,
            0.763, 0.819, 0.907, 0.958, 0.968, 0.968, 0.971, 0.973, 0.982, 0.985, 0.991, 0.995,
            0.995,
        ];
        assert_eq!(
            team13_published_steel_observations(Team13PublishedSteelGap::G052),
            expected_g052
        );
        assert_eq!(
            team13_published_steel_observations(Team13PublishedSteelGap::G047),
            expected_g047
        );
    }

    #[test]
    fn team13_surface_definitions_match_ngsolve_reference() {
        let definitions = team13_surface_measurement_definitions(None).unwrap();
        assert_eq!(definitions.len(), TEAM13_OBSERVATION_COUNT);

        for (index, definition) in definitions.iter().enumerate() {
            assert_eq!(
                definition.ngsolve_name,
                team13_ngsolve_measurement_name(index)
            );
            assert!(team13_steel_surface_group(index).is_ok());
        }

        for (index, z) in [0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060]
            .into_iter()
            .enumerate()
        {
            let definition = &definitions[index];
            assert_eq!(definition.component_index, 2);
            assert_eq!(definition.normal_axis, 2);
            assert!((definition.target - z).abs() < 1e-15);
            assert_eq!(definition.x_range, (0.0, 0.0016));
            assert_eq!(definition.y_range, (-0.025, 0.025));
            assert_eq!(definition.z_range, (z, z));
            assert_eq!(
                definition.quadrature_counts,
                [
                    TEAM13_NGSOLVE_SURFACE_X_SAMPLES,
                    TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                    1
                ]
            );
        }

        for (local_index, x) in [
            0.0021, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.110, 0.1221,
        ]
        .into_iter()
        .enumerate()
        {
            let definition = &definitions[7 + local_index];
            assert_eq!(definition.component_index, 0);
            assert_eq!(definition.normal_axis, 0);
            assert!((definition.target - x).abs() < 1e-15);
            assert_eq!(definition.x_range, (x, x));
            assert_eq!(definition.y_range, (0.015, 0.065));
            assert_eq!(definition.z_range, (0.060, 0.0632));
            assert_eq!(
                definition.quadrature_counts,
                [
                    1,
                    TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                    TEAM13_NGSOLVE_SURFACE_Z_SAMPLES
                ]
            );
        }

        for (local_index, z) in [0.060, 0.050, 0.040, 0.030, 0.020, 0.010, 0.0]
            .into_iter()
            .enumerate()
        {
            let definition = &definitions[18 + local_index];
            assert_eq!(definition.component_index, 2);
            assert_eq!(definition.normal_axis, 2);
            assert!((definition.target - z).abs() < 1e-15);
            assert_eq!(definition.x_range, (0.1221, 0.1253));
            assert_eq!(definition.y_range, (0.015, 0.065));
            assert_eq!(definition.z_range, (z, z));
            assert_eq!(
                definition.quadrature_counts,
                [
                    TEAM13_NGSOLVE_SURFACE_X_SAMPLES,
                    TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                    1
                ]
            );
        }
    }

    #[test]
    fn team13_steel_observation_quadrature_uses_ngsolve_counts() {
        let ngsolve = team13_synthetic_benchmark_surface_definitions(
            Team13SteelObservationQuadratureMode::NgsolveStyle,
        )
        .unwrap();
        assert_eq!(
            ngsolve[0].quadrature_counts,
            [
                TEAM13_NGSOLVE_SURFACE_X_SAMPLES,
                TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                1
            ]
        );
        assert_eq!(
            ngsolve[7].quadrature_counts,
            [
                1,
                TEAM13_NGSOLVE_SURFACE_Y_SAMPLES,
                TEAM13_NGSOLVE_SURFACE_Z_SAMPLES
            ]
        );
    }

    #[test]
    fn team13_ngsolve_reference_csv_matches_rust_published_values() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let reference_dir = workspace.join(TEAM13_NGSOLVE_LINEAR_REFERENCE_DIR);
        if !reference_dir.join("steel_predictions.csv").exists() {
            eprintln!(
                "skipping TEAM 13 NGSolve reference CSV test because `{}` is missing",
                reference_dir.join("steel_predictions.csv").display()
            );
            return;
        }
        let predictions = read_team13_ngsolve_steel_predictions(&reference_dir).unwrap();
        let g_052 = team13_published_steel_observations(Team13PublishedSteelGap::G052);
        let g_047 = team13_published_steel_observations(Team13PublishedSteelGap::G047);
        for (index, prediction) in predictions.iter().enumerate() {
            assert_eq!(prediction.name, team13_ngsolve_measurement_name(index));
            assert_eq!(prediction.group, team13_steel_surface_group(index).unwrap());
            assert!((prediction.observed_g_052 - g_052[index]).abs() < 1e-12);
            assert!((prediction.observed_g_047 - g_047[index]).abs() < 1e-12);
        }
    }

    #[test]
    fn team13_ngsolve_comparison_csv_uses_compact_baseline_columns() {
        let g_052 = team13_published_steel_observations(Team13PublishedSteelGap::G052);
        let g_047 = team13_published_steel_observations(Team13PublishedSteelGap::G047);
        let mut ngsolve_csv =
            "name,group,prediction,observed_g052,observed_g047,residual_g052,residual_g047\n"
                .to_string();
        let mut feec_reports = Vec::new();
        for index in 0..TEAM13_OBSERVATION_COUNT {
            let group = team13_steel_surface_group(index).unwrap();
            let ngsolve_prediction = g_052[index] - 0.01;
            let feec_prediction = g_052[index] + 0.02;
            ngsolve_csv.push_str(&format!(
                "{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}\n",
                team13_ngsolve_measurement_name(index),
                group.as_str(),
                ngsolve_prediction,
                g_052[index],
                g_047[index],
                ngsolve_prediction - g_052[index],
                ngsolve_prediction - g_047[index]
            ));
            feec_reports.push(Team13PublishedSteelBenchmarkReport {
                name: format!("team13_surface_{:02}", index + 1),
                group,
                observed_g_052: g_052[index],
                observed_g_047: g_047[index],
                nominal_prediction: feec_prediction,
                posterior_prediction: feec_prediction,
            });
        }

        let parsed =
            parse_team13_ngsolve_steel_predictions_csv(Path::new("ngsolve.csv"), &ngsolve_csv)
                .unwrap();
        let comparison =
            compare_team13_steel_predictions_to_ngsolve(&parsed, &feec_reports, false).unwrap();
        assert_eq!(comparison[0].name, "measurement_01");
        assert!((comparison[0].feec_minus_ngsolve - 0.03).abs() < 1e-12);
        let compact_csv = team13_steel_ngsolve_comparison_csv(&comparison);
        assert!(compact_csv.starts_with(
            "name,group,ngsolve_prediction,feec_prediction,observed_g052,observed_g047,feec_minus_ngsolve,feec_residual_g052,feec_residual_g047\n"
        ));
    }

    #[test]
    fn team13_steel_observation_cache_roundtrips_grouped_operator() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let cache_dir = workspace.join("target/team13_cache_roundtrip");
        fs::create_dir_all(&cache_dir).unwrap();
        let cache_path = cache_dir.join("steel_cache_roundtrip.bin");
        let part = Team13SteelFullObservationPart {
            row_count: 2,
            triplets: vec![
                SparseTriplet {
                    row: 0,
                    col: 1,
                    value: 2.5,
                },
                SparseTriplet {
                    row: 1,
                    col: 2,
                    value: -0.75,
                },
            ],
            groups: vec![SmoothGroupedNormObservation {
                name: "team13_bz_mid_sheet_01".to_string(),
                samples: vec![
                    SmoothGroupedNormSample {
                        rows: vec![0],
                        weight: 0.25,
                    },
                    SmoothGroupedNormSample {
                        rows: vec![1],
                        weight: 0.75,
                    },
                ],
            }],
            specs: vec![Team13SyntheticBenchmarkObservationSpec {
                name: "team13_bz_mid_sheet_01".to_string(),
                group: Team13SyntheticBenchmarkObservationGroup::SteelAverage,
                steel_surface_group: Some(Team13SteelSurfaceGroup::MidSheet),
            }],
        };
        write_team13_steel_observation_cache(&cache_path, 3, &part).unwrap();
        let short_cache = parse_team13_steel_observation_cache(&fs::read(&cache_path).unwrap(), 3);
        assert_eq!(
            short_cache.unwrap_err(),
            "cache contains 1 specs and 1 groups, expected 25"
        );

        let mut full_part = part.clone();
        for index in 1..TEAM13_OBSERVATION_COUNT {
            full_part.groups.push(SmoothGroupedNormObservation {
                name: format!("team13_cached_group_{index:02}"),
                samples: vec![SmoothGroupedNormSample {
                    rows: vec![0],
                    weight: 1.0,
                }],
            });
            full_part
                .specs
                .push(Team13SyntheticBenchmarkObservationSpec {
                    name: format!("team13_cached_group_{index:02}"),
                    group: Team13SyntheticBenchmarkObservationGroup::SteelAverage,
                    steel_surface_group: Some(Team13SteelSurfaceGroup::MidSheet),
                });
        }
        write_team13_steel_observation_cache(&cache_path, 3, &full_part).unwrap();
        let loaded =
            parse_team13_steel_observation_cache(&fs::read(&cache_path).unwrap(), 3).unwrap();
        assert_eq!(loaded, full_part);
        assert!(parse_team13_steel_observation_cache(&fs::read(&cache_path).unwrap(), 4).is_err());
    }

    #[test]
    fn team13_surface_quadrature_weights_integrate_constants() {
        for count in [3, 4, 5, 100, 300] {
            let weights = simpson_uniform_weights(count, -0.25, 0.75).unwrap();
            assert_eq!(weights.len(), count);
            let sum = weights.iter().sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-12, "count {count} sum {sum}");
            assert!(weights.iter().all(|weight| *weight > 0.0));
        }
    }

    #[test]
    fn active_dof_mask_marks_only_free_state_entries() {
        let layout = DofLayout {
            full_dimension: 5,
            active_dofs: vec![0, 2, 4],
            prescribed_dofs: vec![
                PrescribedDof {
                    index: 1,
                    value: 0.0,
                },
                PrescribedDof {
                    index: 3,
                    value: 0.0,
                },
            ],
        };
        assert_eq!(
            active_dof_mask(&layout),
            FeecVector::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0])
        );
    }

    #[test]
    fn source_recovery_config_rejects_invalid_field_prior_precision_scale() {
        let mut config = Team13SourceRecoveryConfig::default();
        config.field_prior_precision_scale = 0.0;
        let error = validate_source_recovery_config(&config).unwrap_err();
        assert!(error.contains("field_prior_precision_scale"));
    }

    #[test]
    fn team13_surface_observation_overrides_replace_benchmark_values() {
        let defaults = team13_surface_measurement_definitions(None).unwrap();
        let mut overrides = BTreeMap::new();
        for definition in &defaults {
            overrides.insert(definition.name.clone(), 0.125);
        }

        let replaced = team13_surface_measurement_definitions(Some(&overrides)).unwrap();
        assert_eq!(replaced.len(), TEAM13_OBSERVATION_COUNT);
        assert!(replaced
            .iter()
            .all(|definition| (definition.observation - 0.125).abs() < 1e-12));

        overrides.remove(&defaults[0].name);
        let error = team13_surface_measurement_definitions(Some(&overrides)).unwrap_err();
        assert!(error.contains(&defaults[0].name));
    }

    #[test]
    fn team13_geometry_predicates_classify_representative_points() {
        let p = point([0.0, 0.0, 0.02]);
        assert!(is_vertical_sheet(p.as_view()));
        assert!(is_iron_point(p.as_view()));
        assert!(coil_regions_at(p.as_view(), Team13DomainMode::HalfZNonnegative).is_empty());
        let p = point([0.095, 0.0, 0.02]);
        assert_eq!(
            coil_region_at(p.as_view(), Team13DomainMode::HalfZNonnegative),
            Some(Team13CoilRegion::BrickRight)
        );
        let p = point([0.075, 0.075, 0.02]);
        assert_eq!(
            coil_region_at(p.as_view(), Team13DomainMode::HalfZNonnegative),
            Some(Team13CoilRegion::CornerRightBack)
        );
        let p = point([0.20, 0.20, 0.02]);
        assert!(coil_region_at(p.as_view(), Team13DomainMode::HalfZNonnegative).is_none());
        assert!(!is_iron_point(p.as_view()));
        let p = point([0.25, 0.0, 0.10]);
        assert!(is_outer_boundary(
            p.as_view(),
            Team13DomainMode::HalfZNonnegative
        ));
        let p = point([0.0, 0.0, 0.0]);
        assert!(!is_outer_boundary(
            p.as_view(),
            Team13DomainMode::HalfZNonnegative
        ));
    }

    #[test]
    fn team13_coil_regions_are_mutually_exclusive_at_representative_points() {
        let samples = [
            (Team13CoilRegion::BrickBack, [0.0, 0.09, 0.02]),
            (Team13CoilRegion::BrickFront, [0.0, -0.09, 0.02]),
            (Team13CoilRegion::BrickLeft, [-0.09, 0.0, 0.02]),
            (Team13CoilRegion::BrickRight, [0.09, 0.0, 0.02]),
            (Team13CoilRegion::CornerRightBack, [0.085, 0.085, 0.02]),
            (Team13CoilRegion::CornerLeftBack, [-0.085, 0.085, 0.02]),
            (Team13CoilRegion::CornerLeftFront, [-0.085, -0.085, 0.02]),
            (Team13CoilRegion::CornerRightFront, [0.085, -0.085, 0.02]),
        ];
        for (expected, coords) in samples {
            let p = point(coords);
            let regions = coil_regions_at(p.as_view(), Team13DomainMode::HalfZNonnegative);
            assert_eq!(regions, vec![expected], "point {coords:?}");
            assert_eq!(
                coil_region_at(p.as_view(), Team13DomainMode::HalfZNonnegative),
                Some(expected)
            );
        }
    }

    #[test]
    fn team13_current_directions_match_reference_regions() {
        let j0 = 1000.0 / (2500.0 * 1e-6);
        let p = point([0.0, 0.090, 0.02]);
        let current = team13_current_vector(
            p.as_view(),
            Team13DomainMode::HalfZNonnegative,
            1000.0,
            None,
        );
        assert!((current[0] - j0).abs() < 1e-8);
        assert!(current[1].abs() < 1e-8);

        let p = point([0.090, 0.0, 0.02]);
        let current = team13_current_vector(
            p.as_view(),
            Team13DomainMode::HalfZNonnegative,
            1000.0,
            None,
        );
        assert!(current[0].abs() < 1e-8);
        assert!((current[1] + j0).abs() < 1e-8);

        let p = point([0.050, 0.075, 0.02]);
        let current = team13_current_vector(
            p.as_view(),
            Team13DomainMode::HalfZNonnegative,
            1000.0,
            None,
        );
        assert!((current[0] - j0).abs() < 1e-8);
        assert!(current[1].abs() < 1e-8);
    }

    #[test]
    fn team13_nominal_linearization_rejects_zero_direction() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            5,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let error = build_linearized_b_measurements(
            &topology,
            &coords,
            &operators.b_cochain,
            &operators.b_components,
            &FeecVector::zeros(topology.nsimplices(1)),
            1e-4,
            Team13MeasurementMode::LegacyBand,
            0.04,
            None,
        )
        .unwrap_err();
        assert!(error.contains("zero magnitude"));
    }

    #[test]
    fn team13_trace_variance_ratio_uses_componentwise_trace() {
        let prior = vec![
            FeecVector::from_vec(vec![2.0, 4.0]),
            FeecVector::from_vec(vec![3.0, 6.0]),
            FeecVector::from_vec(vec![5.0, 10.0]),
        ];
        let posterior = vec![
            FeecVector::from_vec(vec![1.0, 2.0]),
            FeecVector::from_vec(vec![1.5, 3.0]),
            FeecVector::from_vec(vec![2.5, 5.0]),
        ];
        let ratio = trace_variance_ratio(&prior, &posterior).unwrap();
        assert!((ratio[0] - 0.5).abs() < 1e-12);
        assert!((ratio[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn team13_global_source_operator_matches_sum_of_region_modes() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.11, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.11, 0.07]),
            4,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let reluctivity = reluctivity_weight();
        let boundary = build_outer_boundary(&topology, &coords, Team13DomainMode::HalfZNonnegative);
        let galmats =
            MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &reluctivity);
        let modes = build_source_mode_operator(
            &topology,
            &metric,
            &coords,
            &galmats,
            &boundary,
            Team13DomainMode::HalfZNonnegative,
            1000.0,
        )
        .unwrap();
        let global = build_global_source_operator(
            &topology,
            &metric,
            &coords,
            &galmats,
            &boundary,
            Team13DomainMode::HalfZNonnegative,
            1000.0,
        )
        .unwrap();
        let mut summed = FeecVector::zeros(modes.nrows());
        for (row, _, value) in modes.triplet_iter() {
            summed[row] += value;
        }
        let global = single_column_vector(&global).unwrap();
        assert!(
            (&summed - &global).norm() <= 1e-8 * global.norm().max(1.0),
            "single source-alpha operator should equal the sum of the eight coil-region operators"
        );
    }

    #[test]
    fn team13_unweighted_prior_layout_matches_weighted_pde_layout_and_splits_materials() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            5,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let boundary = build_outer_boundary(&topology, &coords, Team13DomainMode::HalfZNonnegative);
        let reluctivity = reluctivity_weight();
        let galmats =
            MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &reluctivity);
        let state_mass_inverse =
            FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
                &topology,
                &metric,
                &coords,
                None,
                &reluctivity,
            ));
        let weighted = build_reduced_hodge_laplace_1form_system_with_galmats(
            &galmats,
            &boundary,
            &state_mass_inverse,
        )
        .unwrap();
        let unweighted =
            build_reduced_hodge_laplace_1form_system(&topology, &metric, &boundary).unwrap();

        assert_eq!(weighted.layout, unweighted.layout);
        let groups = team13_material_split_graph_groups(&topology, &coords, &unweighted.layout)
            .expect("TEAM13 material split should classify this box mesh");
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|group| !group.is_empty()));
        assert_eq!(
            groups.iter().map(Vec::len).sum::<usize>(),
            unweighted.state_dimension()
        );
    }

    #[test]
    fn nonlinear_team13_prior_uses_unweighted_hodge_matern_system() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            4,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let boundary = build_outer_boundary(&topology, &coords, Team13DomainMode::HalfZNonnegative);
        let unweighted =
            build_reduced_hodge_laplace_1form_system(&topology, &metric, &boundary).unwrap();
        let mean = FeecVector::from_iterator(
            unweighted.state_dimension(),
            (0..unweighted.state_dimension()).map(|index| 0.01 * (index as f64 + 1.0)),
        );
        let config = Team13NonlinearConfig {
            field_prior_kind: Team13FieldPriorKind::UnweightedHodgeMatern,
            field_prior_precision_scale: 2.5e-12,
            prior_kappa: Some(0.75),
            prior_tau: 3.0,
            ..Team13NonlinearConfig::default()
        };

        let prior =
            build_team13_nonlinear_prior(&config, &unweighted, &topology, &coords, &mean).unwrap();
        let expected = scale_gaussian_prior_precision(
            build_hodge_matern_prior_from_reduced_system_with_params(&unweighted, &mean, 0.75, 3.0)
                .unwrap(),
            2.5e-12,
        )
        .unwrap();

        assert_eq!(prior.kind, Team13FieldPriorKind::UnweightedHodgeMatern);
        assert_eq!(prior.precision_scale, 2.5e-12);
        assert_eq!(prior.kappa, 0.75);
        assert_eq!(prior.tau, 3.0);
        assert_eq!(prior.spec, expected);
    }

    #[test]
    fn team13_map_parity_ordinary_matern_prior_factorizes_on_small_mesh() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            3,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let boundary = build_outer_boundary(&topology, &coords, Team13DomainMode::HalfZNonnegative);
        let reluctivity = reluctivity_weight();
        let galmats =
            MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &reluctivity);
        let state_mass_inverse =
            FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
                &topology,
                &metric,
                &coords,
                None,
                &reluctivity,
            ));
        let system = build_reduced_hodge_laplace_1form_system_with_galmats(
            &galmats,
            &boundary,
            &state_mass_inverse,
        )
        .unwrap();
        let mean = FeecVector::zeros(system.state_dimension());
        let prior = add_diagonal_shift_to_gaussian_prior(
            build_hodge_matern_prior_from_reduced_system_with_params(&system, &mean, 1.0, 1.0e-6)
                .unwrap(),
            1.0e-12,
        )
        .unwrap();
        let factor = sparse_from_core(&prior.precision)
            .cholesky_sqrt_lower()
            .unwrap();

        assert_eq!(prior.mean.len(), system.state_dimension());
        assert_eq!(prior.precision.nrows(), system.state_dimension());
        assert_eq!(prior.precision.ncols(), system.state_dimension());
        assert!(prior.precision.nnz() > 0);
        assert!(factor.nnz() > 0);
    }

    #[test]
    fn team13_map_parity_weak_ridge_prior_is_diagonal_and_factorizes() {
        let mean = vec![1.0, -2.0, 3.5, 0.25];
        let ridge_precision = 1.0e-12 + 2.0e-12;
        let prior = build_weak_ridge_prior(&mean, ridge_precision).unwrap();
        let factor = sparse_from_core(&prior.precision)
            .cholesky_sqrt_lower()
            .unwrap();

        assert_eq!(prior.mean, mean);
        assert_eq!(prior.precision.nrows(), 4);
        assert_eq!(prior.precision.ncols(), 4);
        assert_eq!(prior.precision.nnz(), 4);
        for (row, col, value) in prior.precision.triplet_iter() {
            assert_eq!(row, col);
            assert!((value - ridge_precision).abs() <= f64::EPSILON);
        }
        assert_eq!(factor.nnz(), 4);
        assert!(build_weak_ridge_prior(&[0.0], 0.0).is_err());
    }

    #[test]
    fn nonlinear_team13_default_keeps_benchmark_measurements_reporting_only() {
        let config = Team13NonlinearConfig::default();
        assert!(!config.assimilate_measurements);
        assert_eq!(
            config.measurement_mode,
            Team13MeasurementMode::BenchmarkExact
        );
    }

    #[test]
    fn team13_state_prior_is_centered_on_reduced_nominal_solution() {
        let layout = DofLayout::new(
            3,
            vec![0, 2],
            vec![PrescribedDof {
                index: 1,
                value: 0.0,
            }],
        );
        let system = ReducedLinearPdeAssembly {
            operator: core_triplet_to_feec_csr(&diagonal_precision(2, 1.0)),
            residual_bias: FeecVector::zeros(2),
            state_mass: core_triplet_to_feec_csr(&diagonal_precision(2, 1.0)),
            state_mass_inverse: Some(core_triplet_to_feec_csr(&diagonal_precision(2, 1.0))),
            layout,
            forcing_operator: core_triplet_to_feec_csr(&diagonal_precision(2, -1.0)),
            neumann_operator: core_triplet_to_feec_csr(&diagonal_precision(2, -1.0)),
        };
        let nominal_a = FeecVector::from_vec(vec![1.5, 0.0, -2.0]);
        let reduced_nominal_a = reduce_vector_with_layout(&system.layout, &nominal_a).unwrap();

        let prior = build_weighted_whittle_prior(&system, &reduced_nominal_a).unwrap();

        assert_eq!(prior.mean, vec![1.5, -2.0]);
    }

    #[test]
    fn team13_measurement_linearization_builds_nonempty_operator_on_box_mesh() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            5,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let nominal_a = FeecVector::from_iterator(
            topology.nsimplices(1),
            (0..topology.nsimplices(1)).map(|index| 1.0 + index as f64),
        );
        let measurements = build_linearized_b_measurements(
            &topology,
            &coords,
            &operators.b_cochain,
            &operators.b_components,
            &nominal_a,
            1e-4,
            Team13MeasurementMode::LegacyBand,
            0.04,
            None,
        )
        .unwrap();
        assert_eq!(measurements.len(), TEAM13_OBSERVATION_COUNT);
        assert!(measurements[0].spec.operator.nnz() > 0);
        let direction_norm = measurements[0]
            .linearization_direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!((direction_norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn team13_ngsolve_surface_flux_row_builds_on_box_mesh() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            5,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let cell_geometries = top_cell_geometries(&topology, &coords);
        let definition = Team13SurfaceMeasurementDefinition {
            name: "test_surface_flux".to_string(),
            ngsolve_name: "test_surface_flux".to_string(),
            observation: 0.0,
            component_index: 2,
            normal_axis: 2,
            target: 0.035,
            x_range: (-0.05, 0.05),
            y_range: (-0.04, 0.04),
            z_range: (0.035, 0.035),
            quadrature_counts: [5, 5, 1],
        };
        let row = surface_flux_component_row(&cell_geometries, &operators.b_cochain, &definition)
            .unwrap();
        assert!(!row.is_empty());
        assert!(row.iter().all(|(_, value)| value.is_finite()));
    }

    #[test]
    fn team13_face_cochain_row_integrates_constant_bz_on_conforming_mesh() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![0.0, 0.0, 0.0]),
            FeecVector::from_vec(vec![1.0, 1.0, 1.0]),
            1,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let definition = Team13SurfaceMeasurementDefinition {
            name: "unit_bottom_bz".to_string(),
            ngsolve_name: "unit_bottom_bz".to_string(),
            observation: 0.0,
            component_index: 2,
            normal_axis: 2,
            target: 0.0,
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            z_range: (0.0, 0.0),
            quadrature_counts: [1, 1, 1],
        };

        let row = surface_face_cochain_row(&topology, &coords, &operators.b_cochain, &definition)
            .unwrap();
        assert!(row.face_count > 0);
        assert!((row.selected_area - 1.0).abs() < 1e-12);
        assert!((row.expected_area - 1.0).abs() < 1e-12);

        let a = unit_bz_vector_potential_edge_cochain(&topology, &coords);
        let prediction = apply_row(&row.row, &a).unwrap();
        assert!(
            (prediction - 1.0).abs() < 1e-12,
            "face cochain average {prediction}"
        );
    }

    #[test]
    fn team13_face_cochain_row_rejects_nonconforming_mesh() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![0.0, 0.0, 0.0]),
            FeecVector::from_vec(vec![1.0, 1.0, 1.0]),
            1,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let definition = Team13SurfaceMeasurementDefinition {
            name: "unit_cut_bz".to_string(),
            ngsolve_name: "unit_cut_bz".to_string(),
            observation: 0.0,
            component_index: 2,
            normal_axis: 2,
            target: 0.5,
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            z_range: (0.5, 0.5),
            quadrature_counts: [1, 1, 1],
        };

        let error = surface_face_cochain_row(&topology, &coords, &operators.b_cochain, &definition)
            .unwrap_err();
        assert!(
            error.contains("found no mesh faces"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn team13_steel_observation_cache_key_distinguishes_face_cochain_mode() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![0.0, 0.0, 0.0]),
            FeecVector::from_vec(vec![1.0, 1.0, 1.0]),
            1,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let quadrature_key = team13_steel_observation_cache_key(
            &topology,
            &coords,
            Team13SteelObservationQuadratureMode::NgsolveStyle,
        );
        let cochain_key = team13_steel_observation_cache_key(
            &topology,
            &coords,
            Team13SteelObservationQuadratureMode::FaceCochain,
        );
        assert_ne!(quadrature_key, cochain_key);
    }

    #[test]
    fn team13_operator_uncertainty_tangent_kind_parses() {
        assert_eq!(
            "nonlinear"
                .parse::<Team13OperatorUncertaintyTangentKind>()
                .unwrap(),
            Team13OperatorUncertaintyTangentKind::Nonlinear
        );
        assert_eq!(
            "linear-beta-zero"
                .parse::<Team13OperatorUncertaintyTangentKind>()
                .unwrap(),
            Team13OperatorUncertaintyTangentKind::LinearBetaZero
        );
        assert!("bad"
            .parse::<Team13OperatorUncertaintyTangentKind>()
            .is_err());
    }

    #[test]
    fn team13_operator_region_summary_is_posthoc_and_quantitative() {
        let diagnostics = vec![
            Team13OperatorCellDiagnostic {
                variance: Some(1.0),
                b_magnitude: 0.2,
                gradient_indicator: 0.0,
                iron: true,
                interface: false,
                central_gap: false,
                steel_corner: false,
                measurement_mid_sheet: false,
                measurement_back_right_top: false,
                measurement_back_right_edge: false,
                high_gradient: false,
                low_b_magnitude: false,
            },
            Team13OperatorCellDiagnostic {
                variance: Some(4.0),
                b_magnitude: 0.5,
                gradient_indicator: 8.0,
                iron: true,
                interface: true,
                central_gap: false,
                steel_corner: false,
                measurement_mid_sheet: false,
                measurement_back_right_top: false,
                measurement_back_right_edge: false,
                high_gradient: true,
                low_b_magnitude: false,
            },
            Team13OperatorCellDiagnostic {
                variance: Some(2.0),
                b_magnitude: 0.1,
                gradient_indicator: 0.0,
                iron: false,
                interface: false,
                central_gap: false,
                steel_corner: false,
                measurement_mid_sheet: false,
                measurement_back_right_top: false,
                measurement_back_right_edge: false,
                high_gradient: false,
                low_b_magnitude: true,
            },
        ];

        let (regions, correlations) = summarize_team13_operator_regions(&diagnostics);
        let interface = regions
            .iter()
            .find(|summary| summary.region == "iron_air_interface_band")
            .unwrap();
        assert_eq!(interface.count, 1);
        assert!((interface.mean_variance - 4.0).abs() < 1e-12);
        assert!((interface.variance_ratio_to_iron_bulk - 4.0).abs() < 1e-12);
        assert!(correlations
            .iter()
            .any(|row| row.indicator == "high_gradient_indicator"
                && row.pearson_with_variance.is_finite()));
    }

    #[test]
    fn team13_operator_interleaved_b_variance_sums_component_traces() {
        let values = GmrfVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0]);
        let traces = cell_trace_variances_from_interleaved(&values).unwrap();
        assert_eq!(traces, vec![6.0, 10.0]);
    }

    #[test]
    fn team13_operator_uncertainty_default_keeps_regions_out_of_inference() {
        let config = Team13OperatorUncertaintyConfig::default();
        assert_eq!(config.prior_kind, Team13MapParityPriorKind::WeakRidge);
        assert_eq!(
            config.pde_residual_kind,
            Team13MapParityPdeResidualKind::UngaugedCurl
        );
        assert_eq!(
            config.pde_residual_weighting,
            Team13PdeResidualWeighting::Euclidean
        );
        assert!(!config.include_steel_observations);
        assert_eq!(
            config.steel_observation_quadrature,
            Team13SteelObservationQuadratureMode::FaceCochain
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn team13_measurement_plane_mesh_face_cochain_rows_are_finite_nonempty() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping TEAM 13 measurement-plane mesh test because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear_measurement_planes.geo");
        let out_dir = workspace.join("target/team13_measurement_planes_smoke");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_measurement_planes.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("10")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let mesh_bytes = fs::read(&mesh_path).unwrap();
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let operators = build_team13_operators(&topology, &coords).unwrap();
        for definition in team13_surface_measurement_definitions(None).unwrap() {
            let row =
                surface_face_cochain_row(&topology, &coords, &operators.b_cochain, &definition)
                    .unwrap_or_else(|err| panic!("{}: {err}", definition.name));
            assert!(!row.row.is_empty(), "empty row for {}", definition.name);
            assert!(row.row.iter().all(|(_, value)| value.is_finite()));
            assert!(row.face_count > 0);
            let area_error = (row.selected_area - row.expected_area).abs();
            assert!(
                area_error <= TEAM13_FACE_COHERENT_AREA_REL_TOL * row.expected_area,
                "{} selected area {:.16e}, expected {:.16e}",
                definition.name,
                row.selected_area,
                row.expected_area
            );
        }
    }

    fn unit_bz_vector_potential_edge_cochain(
        topology: &Complex,
        coords: &MeshCoords,
    ) -> FeecVector {
        FeecVector::from_iterator(
            topology.nsimplices(1),
            topology.skeleton(1).handle_iter().map(|edge| {
                let mut vertices = edge.iter();
                let first = coord3(coords, vertices.next().unwrap()).unwrap();
                let second = coord3(coords, vertices.next().unwrap()).unwrap();
                0.5 * (first[0] + second[0]) * (second[1] - first[1])
            }),
        )
    }

    #[cfg(feature = "external-reference-tests")]
    #[test]
    fn team13_ngsolve_exported_mesh_heavy_quadrature_rows_are_finite_nonempty() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let mesh_path = workspace.join(TEAM13_NGSOLVE_MESH_PATH);
        if !mesh_path.exists() {
            eprintln!(
                "skipping TEAM 13 NGSolve exported mesh quadrature test because `{}` is missing",
                mesh_path.display()
            );
            return;
        }

        let mesh_bytes = fs::read(&mesh_path).unwrap();
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let cell_geometries = top_cell_geometries(&topology, &coords);
        let observation_part = load_or_build_team13_steel_full_observation_part(
            &topology,
            &coords,
            &operators,
            &cell_geometries,
            Team13SteelObservationQuadratureMode::NgsolveStyle,
        )
        .unwrap();

        assert_eq!(observation_part.specs.len(), TEAM13_OBSERVATION_COUNT);
        assert_eq!(observation_part.groups.len(), TEAM13_OBSERVATION_COUNT);
        assert!(observation_part.row_count > 0);
        assert!(!observation_part.triplets.is_empty());
        let mut row_nnz = vec![0usize; observation_part.row_count];
        for triplet in &observation_part.triplets {
            assert!(triplet.value.is_finite());
            assert!(triplet.row < observation_part.row_count);
            assert!(triplet.col < topology.nsimplices(1));
            row_nnz[triplet.row] += 1;
        }
        for (index, group) in observation_part.groups.iter().enumerate() {
            assert_eq!(group.name, observation_part.specs[index].name);
            assert!(!group.samples.is_empty());
            for sample in &group.samples {
                assert!(sample.weight.is_finite() && sample.weight > 0.0);
                assert!(!sample.rows.is_empty());
                for row in &sample.rows {
                    assert!(*row < observation_part.row_count);
                    assert!(row_nnz[*row] > 0, "empty row {row} in {}", group.name);
                }
            }
        }
    }

    #[test]
    fn team13_measurement_restriction_matches_lifted_full_prediction() {
        let mut operator = SparseTripletMatrix::new(1, 4);
        operator.push(0, 0, 2.0);
        operator.push(0, 1, 99.0);
        operator.push(0, 2, -3.0);
        operator.push(0, 3, 17.0);
        let measurement = Team13LinearizedMeasurement {
            spec: LinearGaussianMeasurementSpec {
                name: "probe".to_string(),
                operator,
                observations: vec![0.0],
                bias: vec![1.25],
                variance: 0.5,
            },
            nominal_prediction: 0.0,
            linearization_direction: [1.0, 0.0, 0.0],
        };
        let layout = DofLayout {
            full_dimension: 4,
            active_dofs: vec![0, 2],
            prescribed_dofs: vec![
                PrescribedDof {
                    index: 1,
                    value: -2.0,
                },
                PrescribedDof {
                    index: 3,
                    value: 4.0,
                },
            ],
        };

        let reduced = restrict_team13_measurement_to_layout(&measurement, &layout).unwrap();
        assert_eq!(reduced.operator.ncols(), layout.reduced_dimension());
        let reduced_state = GmrfVector::from_vec(vec![3.0, 5.0]);
        let full_state = GmrfVector::from_vec(vec![3.0, -2.0, 5.0, 4.0]);
        let reduced_prediction = triplet_to_sparse_row_operator(&reduced.operator)
            .unwrap()
            .apply(&reduced_state)
            .unwrap()[0]
            + reduced.bias[0];
        let full_prediction = triplet_to_sparse_row_operator(&measurement.spec.operator)
            .unwrap()
            .apply(&full_state)
            .unwrap()[0]
            + measurement.spec.bias[0];
        assert!((reduced_prediction - full_prediction).abs() < 1e-12);
    }

    #[test]
    fn team13_synthetic_nonlinear_baseline_surface_rows_match_reduced_layout_predictions() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            5,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let initial_a = FeecVector::from_iterator(
            topology.nsimplices(1),
            (0..topology.nsimplices(1)).map(|index| 1.0 + index as f64),
        );
        let truth_a = FeecVector::from_iterator(
            topology.nsimplices(1),
            (0..topology.nsimplices(1)).map(|index| 0.5 + 0.25 * index as f64),
        );
        let measurements = build_synthetic_surface_flux_measurements(
            &topology, &coords, &operators, &initial_a, &truth_a, 1e-8,
        )
        .unwrap();
        let active_dofs = (0..topology.nsimplices(1))
            .filter(|index| index % 2 == 0)
            .collect::<Vec<_>>();
        let prescribed_dofs = (0..topology.nsimplices(1))
            .filter(|index| index % 2 == 1)
            .map(|index| PrescribedDof {
                index,
                value: -0.01 * index as f64,
            })
            .collect::<Vec<_>>();
        let layout = DofLayout::new(topology.nsimplices(1), active_dofs.clone(), prescribed_dofs);
        let reduced_state = GmrfVector::from_vec(
            active_dofs
                .iter()
                .map(|index| 0.2 + 0.03 * *index as f64)
                .collect(),
        );
        let mut full_values = vec![0.0; topology.nsimplices(1)];
        for (reduced_index, full_index) in active_dofs.iter().copied().enumerate() {
            full_values[full_index] = reduced_state[reduced_index];
        }
        for fixed in &layout.prescribed_dofs {
            full_values[fixed.index] = fixed.value;
        }
        let full_state = GmrfVector::from_vec(full_values);

        for measurement in &measurements {
            let reduced = restrict_team13_measurement_to_layout(measurement, &layout).unwrap();
            let reduced_prediction = triplet_to_sparse_row_operator(&reduced.operator)
                .unwrap()
                .apply(&reduced_state)
                .unwrap()[0]
                + reduced.bias[0];
            let full_prediction = triplet_to_sparse_row_operator(&measurement.spec.operator)
                .unwrap()
                .apply(&full_state)
                .unwrap()[0]
                + measurement.spec.bias[0];
            let tolerance = 1e-10 * full_prediction.abs().max(1.0);
            assert!(
                (reduced_prediction - full_prediction).abs() <= tolerance,
                "{} reduced prediction {} full prediction {}",
                measurement.spec.name,
                reduced_prediction,
                full_prediction
            );
        }
    }

    #[test]
    fn synthetic_benchmark_geometry_rows_match_reduced_layout_predictions() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            5,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let active_dofs = (0..topology.nsimplices(1))
            .filter(|index| index % 2 == 0)
            .collect::<Vec<_>>();
        let prescribed_dofs = (0..topology.nsimplices(1))
            .filter(|index| index % 2 == 1)
            .map(|index| PrescribedDof {
                index,
                value: -0.01 * index as f64,
            })
            .collect::<Vec<_>>();
        let layout = DofLayout::new(topology.nsimplices(1), active_dofs.clone(), prescribed_dofs);
        let reduced_initial = active_dofs
            .iter()
            .map(|index| 0.2 + 0.03 * *index as f64)
            .collect::<Vec<_>>();
        let reduced_truth = active_dofs
            .iter()
            .map(|index| -0.1 + 0.02 * *index as f64)
            .collect::<Vec<_>>();
        let mut full_truth = vec![0.0; topology.nsimplices(1)];
        for (reduced_index, full_index) in active_dofs.iter().copied().enumerate() {
            full_truth[full_index] = reduced_truth[reduced_index];
        }
        for fixed in &layout.prescribed_dofs {
            full_truth[fixed.index] = fixed.value;
        }

        let build = build_team13_synthetic_benchmark_geometry_observations(
            &topology,
            &coords,
            &operators,
            &layout,
            &reduced_initial,
            &reduced_truth,
            Team13SteelObservationQuadratureMode::NgsolveStyle,
            1e-8,
        )
        .unwrap();
        assert_eq!(build.model.operator().ncols(), layout.reduced_dimension());
        assert_eq!(
            build.assimilated_model.operator().ncols(),
            layout.reduced_dimension()
        );
        assert_eq!(build.specs.len(), TEAM13_OBSERVATION_COUNT + 15);
        assert_eq!(build.assimilated_specs.len(), TEAM13_OBSERVATION_COUNT);
        assert_eq!(
            build.assimilated_observations.len(),
            TEAM13_OBSERVATION_COUNT
        );
        assert!(build
            .assimilated_specs
            .iter()
            .all(|spec| spec.group == Team13SyntheticBenchmarkObservationGroup::SteelAverage));
        assert!(build
            .assimilated_model
            .groups()
            .iter()
            .all(|group| group.samples.iter().all(|sample| sample.rows.len() == 1)));
        assert_eq!(
            build
                .specs
                .iter()
                .filter(|spec| spec.group == Team13SyntheticBenchmarkObservationGroup::SteelAverage)
                .count(),
            TEAM13_OBSERVATION_COUNT
        );
        assert_eq!(
            build
                .specs
                .iter()
                .filter(|spec| spec.group == Team13SyntheticBenchmarkObservationGroup::AirPoint)
                .count(),
            15
        );

        let full_model = SmoothGroupedNormLinearResidualModel::new(
            build.full_operator.clone(),
            build.full_bias.clone(),
            build.model.groups().to_vec(),
            1e-8,
        )
        .unwrap();
        let full_predictions = full_model.smooth_norm_values(&full_truth).unwrap();
        for (index, (reduced, full)) in build
            .observations
            .iter()
            .zip(full_predictions.iter())
            .enumerate()
        {
            let tolerance = 1e-10 * full.abs().max(1.0);
            assert!(
                (*reduced - *full).abs() <= tolerance,
                "{} reduced prediction {} full prediction {}",
                build.specs[index].name,
                reduced,
                full
            );
        }
    }

    #[test]
    fn team13_joint_material_published_smooth_observations_use_selected_gap() {
        let mesh = CartesianMeshInfo::new_min_max(
            FeecVector::from_vec(vec![-0.13, -0.07, 0.0]),
            FeecVector::from_vec(vec![0.13, 0.07, 0.07]),
            5,
        );
        let (topology, coords) = mesh.compute_coord_complex();
        let operators = build_team13_operators(&topology, &coords).unwrap();
        let layout = DofLayout::new(
            topology.nsimplices(1),
            (0..topology.nsimplices(1)).collect(),
            Vec::new(),
        );

        let build = build_team13_published_steel_smooth_observations(
            &topology,
            &coords,
            &operators,
            &layout,
            Team13SteelObservationQuadratureMode::NgsolveStyle,
            1.0e-8,
            Team13PublishedSteelGap::G047,
        )
        .unwrap();

        assert_eq!(
            build.observations,
            team13_published_steel_observations(Team13PublishedSteelGap::G047).to_vec()
        );
        assert_eq!(build.specs.len(), TEAM13_OBSERVATION_COUNT);
        assert_eq!(build.model.state_dimension(), layout.reduced_dimension());
        assert_eq!(build.model.residual_dimension(), TEAM13_OBSERVATION_COUNT);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn team13_half_and_full_gmsh_smoke_solve() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping TEAM 13 gmsh smoke test because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_smoke");
        fs::create_dir_all(&out_dir).unwrap();

        for mode in [Team13DomainMode::HalfZNonnegative, Team13DomainMode::Full] {
            let mesh_path = out_dir.join(format!("team13_{}.msh", mode.as_str()));
            let full_domain = if mode == Team13DomainMode::Full {
                "1"
            } else {
                "0"
            };
            let status = Command::new("gmsh")
                .arg("-3")
                .arg(&geo)
                .arg("-setnumber")
                .arg("FullDomain")
                .arg(full_domain)
                .arg("-setnumber")
                .arg("MeshScale")
                .arg("8")
                .arg("-o")
                .arg(&mesh_path)
                .status()
                .unwrap();
            assert!(status.success());
            let config = Team13LinearConfig {
                mesh_path,
                domain_mode: mode,
                coil_relative_std: 0.10,
                pde_variance: 1e-6,
                measurement_mode: Team13MeasurementMode::BenchmarkExact,
                legacy_measurement_band: 0.03,
                output_dir: Some(out_dir.join(format!("out_{}", mode.as_str()))),
                solver: LinearPdeUqSolverConfig {
                    variance: LinearPdeVarianceConfig {
                        mode: LinearPdeVarianceMode::MonteCarlo,
                        num_variance_probes: 2,
                        variance_batch_count: 1,
                        rng_seed: 3,
                        local_rb_block_size: 16,
                    },
                    precision_policy: LinearPdePrecisionPolicy::default(),
                    log_diagnostics: false,
                },
                ..Team13LinearConfig::default()
            };
            let result = solve_team13_linear_uq(&config).unwrap();
            assert_eq!(result.sensor_reports.len(), TEAM13_OBSERVATION_COUNT);
            assert!(result
                .posterior
                .posterior_variance
                .iter()
                .all(|value| (*value).is_finite() && *value >= 0.0));
            assert!(result
                .a_variance_ratio
                .iter()
                .all(|value| (*value).is_finite() && *value >= 0.0));
            let output_dir = config.output_dir.unwrap();
            assert!(output_dir.join("sensor_summary.csv").exists());
            assert!(output_dir.join("variance_summary.txt").exists());
            assert!(output_dir.join("A_posterior.vtu").exists());
            assert!(output_dir.join("A_reference.vtu").exists());
            assert!(output_dir.join("B_vector_pushforward.vtu").exists());
            let a_posterior = fs::read_to_string(output_dir.join("A_posterior.vtu")).unwrap();
            assert!(a_posterior.contains("state_active_dof_mask"));
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn team13_same_mesh_linear_parity_smoke_reports_finite_stats() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping TEAM 13 same-mesh parity smoke test because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_same_mesh_linear_parity_smoke");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_parity.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("10")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let output_dir = out_dir.join("out");
        let result = run_team13_same_mesh_linear_parity(&Team13SameMeshLinearParityConfig {
            mesh_path,
            output_dir: Some(output_dir.clone()),
            steel_observation_quadrature: Team13SteelObservationQuadratureMode::NgsolveStyle,
            ..Team13SameMeshLinearParityConfig::default()
        })
        .unwrap();
        assert!(result.vertices > 0);
        assert!(result.active_dofs > 0);
        assert!(result.operator_nnz > 0);
        assert!(result.rhs_l2.is_finite() && result.rhs_l2 > 0.0);
        assert!(result.solution_l2.is_finite());
        assert!(result.linear_residual_l2.is_finite());
        assert!(
            result.linear_residual_l2 <= 1.0e-4 * result.rhs_l2.max(1.0),
            "linear residual {} rhs {}",
            result.linear_residual_l2,
            result.rhs_l2
        );
        assert!(result.energy.is_finite());
        assert!(result.work.is_finite());
        assert!(result.steel_rmse_g052.is_finite());
        assert!(result.steel_rmse_g047.is_finite());
        assert!(result.audit.total_volume.is_finite() && result.audit.total_volume > 0.0);
        for name in [
            "iron",
            "air",
            "source_free_air",
            "coil_total",
            "brick_back",
            "brick_front",
            "brick_left",
            "brick_right",
            "corner_right_back",
            "corner_left_back",
            "corner_left_front",
            "corner_right_front",
        ] {
            let entry = result
                .audit
                .entries
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("missing audit entry {name}"));
            assert!(entry.volume.is_finite() && entry.volume >= 0.0);
            assert!(entry.current_l2_norm.is_finite() && entry.current_l2_norm >= 0.0);
        }
        assert!(output_dir.join("linear_parity_diagnostic.json").exists());
        assert!(output_dir.join("linear_parity_summary.csv").exists());
        assert!(output_dir.join("region_audit.csv").exists());
        assert!(output_dir.join("steel_predictions.csv").exists());
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn team13_nonlinear_forward_parity_smoke_reports_finite_stats() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping TEAM 13 nonlinear forward parity smoke test because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_nonlinear_forward_parity_smoke");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_nonlinear_parity.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("10")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let output_dir = out_dir.join("out");
        let result = run_team13_nonlinear_forward_parity(&Team13NonlinearForwardParityConfig {
            mesh_path,
            output_dir: Some(output_dir.clone()),
            max_iterations: 8,
            ..Team13NonlinearForwardParityConfig::default()
        })
        .unwrap();
        assert!(result.vertices > 0);
        assert!(result.active_dofs > 0);
        assert_eq!(result.residual_dimension, result.active_dofs);
        assert!(result.initial_jacobian_nnz > 0);
        assert!(result.final_jacobian_nnz > 0);
        assert!(result.rhs_l2.is_finite() && result.rhs_l2 > 0.0);
        assert!(result.initial_solution_l2.is_finite());
        assert!(result.nonlinear_solution_l2.is_finite());
        assert!(result.initial_residual_l2.is_finite());
        assert!(result.final_residual_l2.is_finite());
        assert!(
            result.final_residual_l2 <= result.initial_residual_l2.max(1.0),
            "nonlinear final residual {} should not exceed initial {}",
            result.final_residual_l2,
            result.initial_residual_l2
        );
        assert!(result.initial_steel_rmse_g052.is_finite());
        assert!(result.initial_steel_rmse_g047.is_finite());
        assert!(result.nonlinear_steel_rmse_g052.is_finite());
        assert!(result.nonlinear_steel_rmse_g047.is_finite());
        assert_eq!(result.steel_predictions.len(), TEAM13_OBSERVATION_COUNT);
        assert_eq!(result.nonlinear_steel_group_summaries.len(), 3);
        assert!(result
            .steel_predictions
            .iter()
            .all(|report| report.nominal_prediction.is_finite()
                && report.posterior_prediction.is_finite()));
        assert!(output_dir.join("nonlinear_forward_parity.json").exists());
        assert!(output_dir.join("nonlinear_forward_summary.csv").exists());
        assert!(output_dir.join("region_audit.csv").exists());
        assert!(output_dir.join("steel_predictions.csv").exists());
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn team13_map_parity_internal_reference_smoke() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping TEAM 13 MAP parity smoke test because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_map_parity_smoke");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_map_parity.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("10")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let output_dir = out_dir.join("out");
        let result = run_team13_map_parity(&Team13MapParityConfig {
            mesh_path,
            output_dir: Some(output_dir.clone()),
            max_iterations: 8,
            truth_max_iterations: 8,
            sweep_pde_variances: Vec::new(),
            sweep_observation_std_tesla: Vec::new(),
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::MonteCarlo,
                num_variance_probes: 1,
                variance_batch_count: 1,
                rng_seed: 1317,
                local_rb_block_size: 16,
            },
            ..Team13MapParityConfig::default()
        })
        .unwrap();

        assert!(result.vertices > 0);
        assert_eq!(
            result.default_run.steel_observation_count,
            TEAM13_OBSERVATION_COUNT
        );
        assert!(result.truth_converged);
        assert!(result.truth_residual_norm.is_finite());
        let run = &result.default_run;
        assert!(run.posterior_converged);
        assert!(run.initial_relative_error.is_finite());
        assert!(run.posterior_relative_error.is_finite());
        assert!(
            run.posterior_relative_error < run.initial_relative_error,
            "posterior A error {} did not improve over initial {}",
            run.posterior_relative_error,
            run.initial_relative_error
        );
        assert!(
            run.posterior_steel_rmse < run.initial_steel_rmse,
            "posterior steel RMSE {} did not improve over initial {}",
            run.posterior_steel_rmse,
            run.initial_steel_rmse
        );
        assert!(run.posterior_residual_norm.is_finite());
        assert!(
            run.posterior_residual_norm <= run.initial_residual_norm.max(1.0),
            "posterior residual {} should not exceed initial {}",
            run.posterior_residual_norm,
            run.initial_residual_norm
        );
        assert!(run.final_factorization.nnz > 0);
        assert_eq!(run.assembly.factor_nnz, Some(run.final_factorization.nnz));
        assert!(run.all_finite_variances);
        assert!(run.nonnegative_variances);
        assert_eq!(run.internal_steel_variances.len(), TEAM13_OBSERVATION_COUNT);
        assert!(run.internal_steel_variances.iter().all(|report| {
            report.prior_variance.is_finite()
                && report.posterior_variance.is_finite()
                && report.prior_variance >= -1e-12
                && report.posterior_variance >= -1e-12
        }));
        assert_eq!(
            run.published_steel_benchmark_reports.len(),
            TEAM13_OBSERVATION_COUNT
        );
        assert!(output_dir.join("map_parity.json").exists());
        assert!(output_dir.join("map_parity_runs.csv").exists());
        assert!(output_dir.join("internal_steel_predictions.csv").exists());
        assert!(output_dir.join("published_steel_reporting.csv").exists());
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_team13_beta_zero_parity_and_synthetic_iron_smoke() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping nonlinear TEAM 13 smoke test because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_nonlinear_smoke");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_nonlinear.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("14")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let base_config = Team13NonlinearConfig {
            mesh_path: mesh_path.clone(),
            material_kind: Team13NonlinearMaterialKind::SmoothQuadratic,
            beta_iron: 0.0,
            max_iterations: 4,
            measurement_mode: Team13MeasurementMode::LegacyBand,
            legacy_measurement_band: 0.05,
            assimilate_measurements: false,
            write_outputs: false,
            sensor_variance_count: 2,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::MonteCarlo,
                num_variance_probes: 1,
                variance_batch_count: 1,
                rng_seed: 17,
                local_rb_block_size: 16,
            },
            ..Team13NonlinearConfig::default()
        };
        let beta_zero = run_team13_nonlinear_uq(&base_config).unwrap();
        assert!(beta_zero.beta_zero_relative_error.unwrap() < 1e-6);
        assert!(beta_zero.final_residual_norm <= 1e-6 * beta_zero.initial_residual_norm.max(1.0));
        assert_eq!(beta_zero.assimilated_measurements, 0);
        assert!(beta_zero.assembly.posterior_precision_nnz > 0);
        assert!(beta_zero.assembly.factor_nnz.unwrap() > 0);
        assert!(beta_zero.assembly.posterior_precision_nnz < beta_zero.final_factorization.nnz);
        assert_eq!(beta_zero.sensor_variances.len(), 2);
        assert!(beta_zero
            .sensor_variances
            .iter()
            .all(
                |report| report.posterior_variance.is_finite() && report.posterior_variance >= 0.0
            ));

        let nonlinear = run_team13_nonlinear_uq(&Team13NonlinearConfig {
            material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
            beta_iron: 10.0,
            assimilate_measurements: true,
            ..base_config
        })
        .unwrap();
        assert!(nonlinear.final_residual_norm.is_finite());
        assert_eq!(nonlinear.assimilated_measurements, TEAM13_OBSERVATION_COUNT);
        assert!(nonlinear.linear_sensor_rmse.is_finite());
        assert!(nonlinear.sensor_rmse.is_finite());
        assert!(nonlinear
            .sensor_reports
            .iter()
            .all(|report| report.posterior_prediction.is_finite()));
        assert!(nonlinear.final_factorization.nnz > 0);
        assert_eq!(
            nonlinear.assembly.factor_nnz,
            Some(nonlinear.final_factorization.nnz)
        );
        assert!(nonlinear.assembly.posterior_precision_nnz < nonlinear.final_factorization.nnz);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_team13_diagnostics_smoke_reports_finite_first_step_rows() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping nonlinear TEAM 13 diagnostics smoke because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_nonlinear_diagnostics_smoke");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_diagnostics.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("14")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let diagnostics = run_team13_nonlinear_diagnostics(&Team13NonlinearDiagnosticsConfig {
            solve: Team13NonlinearConfig {
                mesh_path,
                material_kind: Team13NonlinearMaterialKind::NgsolveTabulatedLinear,
                beta_iron: 10.0,
                assimilate_measurements: true,
                max_iterations: 4,
                measurement_mode: Team13MeasurementMode::LegacyBand,
                legacy_measurement_band: 0.05,
                write_outputs: false,
                sensor_variance_count: 0,
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::MonteCarlo,
                    num_variance_probes: 1,
                    variance_batch_count: 1,
                    rng_seed: 17,
                    local_rb_block_size: 16,
                },
                ..Team13NonlinearConfig::default()
            },
            pde_variance_values: vec![1e-2, 1.0, 1e2],
        })
        .unwrap();

        assert!(diagnostics.reduced_physical_rhs_norm.is_finite());
        assert!(diagnostics.beta_zero_residual_norm.is_finite());
        assert!(
            diagnostics.beta_zero_residual_norm
                <= 1e-8 * diagnostics.reduced_physical_rhs_norm.max(1.0)
        );
        assert!(diagnostics
            .source_free_affine_solve_residual_norm
            .is_finite());
        assert!(diagnostics
            .nonlinear_residual_at_linear_mean_norm
            .is_finite());
        assert_eq!(
            diagnostics.assimilated_measurements,
            TEAM13_OBSERVATION_COUNT
        );
        assert_eq!(diagnostics.first_steps.len(), 6);
        assert!(diagnostics
            .first_steps
            .iter()
            .any(|row| { row.step_regularization == GaussNewtonStepRegularization::None }));
        assert!(diagnostics.first_steps.iter().any(|row| {
            row.step_regularization == GaussNewtonStepRegularization::LevenbergMarquardtGrid
        }));
        assert!(diagnostics.first_steps.iter().all(|row| {
            matches!(
                row.classification,
                Team13FirstStepConditioningClass::WellConditioned
                    | Team13FirstStepConditioningClass::IllConditioned
                    | Team13FirstStepConditioningClass::Failed
            ) && row
                .diagnostics
                .as_ref()
                .map(|diagnostic| {
                    diagnostic.objective.is_finite()
                        && diagnostic.gradient_norm.is_finite()
                        && diagnostic.step_norm.is_finite()
                        && diagnostic.objective_grid.len() == 21
                        && diagnostic.assembly.dimension == diagnostics.active_dofs
                        && diagnostic.assembly.prior_precision_nnz > 0
                        && diagnostic.assembly.posterior_precision_nnz > 0
                        && diagnostic
                            .assembly
                            .term_operator_nnz(NonlinearAssemblyTermKind::Residual)
                            > 0
                        && diagnostic
                            .assembly
                            .term_precision_update_nnz(NonlinearAssemblyTermKind::Residual)
                            > 0
                        && diagnostic
                            .assembly
                            .term_operator_nnz(NonlinearAssemblyTermKind::LinearMeasurement)
                            > 0
                        && diagnostic
                            .assembly
                            .term_precision_update_nnz(NonlinearAssemblyTermKind::LinearMeasurement)
                            > 0
                })
                .unwrap_or_else(|| row.failure_reason.is_some())
        }));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn team13_synthetic_nonlinear_baseline_recovers_internal_truth() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!(
                "skipping synthetic nonlinear TEAM 13 baseline because gmsh is not available"
            );
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_synthetic_nonlinear_baseline");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_synthetic_baseline.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("18")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let result =
            run_team13_synthetic_nonlinear_baseline(&Team13SyntheticNonlinearBaselineConfig {
                mesh_path,
                max_iterations: 18,
                truth_max_iterations: 18,
                ..Team13SyntheticNonlinearBaselineConfig::default()
            })
            .unwrap();

        assert_eq!(result.synthetic_sensor_count, 5);
        assert!(result.truth_converged);
        assert!(result.truth_residual_norm.is_finite());
        assert!(
            result.truth_residual_norm <= 1e-5 * result.initial_residual_norm.max(1.0),
            "truth residual {} was not small relative to initial residual {}",
            result.truth_residual_norm,
            result.initial_residual_norm
        );
        assert_eq!(result.observation_runs.len(), 2);
        let smooth = result
            .observation_runs
            .iter()
            .find(|run| run.model_kind == Team13SyntheticObservationModelKind::SmoothMagnitude)
            .expect("smooth magnitude run should be present");
        let signed = result
            .observation_runs
            .iter()
            .find(|run| run.model_kind == Team13SyntheticObservationModelKind::SignedLinearProxy)
            .expect("signed linear proxy run should be present");
        for run in [smooth, signed] {
            assert_eq!(run.synthetic_sensor_count, result.synthetic_sensor_count);
            assert!(run.posterior_converged);
            assert!(run.posterior_residual_norm.is_finite());
            assert!(
                run.posterior_residual_norm <= 1e-3 * run.initial_residual_norm.max(1.0),
                "{} posterior residual {} was not small relative to initial {}",
                run.model_kind.as_str(),
                run.posterior_residual_norm,
                run.initial_residual_norm
            );
            assert!(run.final_factorization.nnz > 0);
            assert_eq!(run.assembly.factor_nnz, Some(run.final_factorization.nnz));
            assert!(run.all_finite_variances);
            assert!(run.nonnegative_variances);
            assert_eq!(run.sensor_variances.len(), run.synthetic_sensor_count);
            assert!(run.sign_mismatch_count <= run.synthetic_sensor_count);
        }
        assert!(
            smooth.posterior_relative_error < smooth.initial_relative_error,
            "posterior relative error {} did not improve over initial {}",
            smooth.posterior_relative_error,
            smooth.initial_relative_error
        );
        assert!(
            smooth.posterior_sensor_rmse < smooth.initial_sensor_rmse,
            "posterior sensor RMSE {} did not improve over initial {}",
            smooth.posterior_sensor_rmse,
            smooth.initial_sensor_rmse
        );
        assert!(signed.posterior_sensor_rmse.is_finite());
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn synthetic_benchmark_geometry_recovers_internal_truth() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!(
                "skipping synthetic benchmark-geometry TEAM 13 baseline because gmsh is not available"
            );
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_synthetic_benchmark_geometry");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_synthetic_benchmark_geometry.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("18")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let result =
            run_team13_synthetic_benchmark_geometry(&Team13SyntheticBenchmarkGeometryConfig {
                mesh_path,
                max_iterations: 18,
                truth_max_iterations: 18,
                source_scale_diagnostic_values: vec![1.0],
                sweep_pde_variances: Vec::new(),
                sweep_observation_std_tesla: Vec::new(),
                ..Team13SyntheticBenchmarkGeometryConfig::default()
            })
            .unwrap();

        assert_eq!(result.observation_count, TEAM13_OBSERVATION_COUNT + 15);
        assert_eq!(
            result.assimilated_observation_count,
            TEAM13_OBSERVATION_COUNT
        );
        assert_eq!(result.steel_observation_count, TEAM13_OBSERVATION_COUNT);
        assert_eq!(result.air_observation_count, 15);
        assert!(result.truth_converged);
        assert!(result.truth_residual_norm.is_finite());
        let run = &result.default_run;
        assert_eq!(run.observation_count, result.observation_count);
        assert_eq!(run.assimilated_observation_count, TEAM13_OBSERVATION_COUNT);
        assert_eq!(run.total_residual_rows, result.active_dofs);
        assert!(run.posterior_converged);
        assert!(run.posterior_residual_norm.is_finite());
        assert!(
            run.posterior_residual_norm <= run.initial_residual_norm,
            "posterior full residual {} should not exceed initial {}",
            run.posterior_residual_norm,
            run.initial_residual_norm
        );
        assert!(
            run.posterior_relative_error < run.initial_relative_error,
            "posterior relative error {} did not improve over initial {}",
            run.posterior_relative_error,
            run.initial_relative_error
        );
        assert!(
            run.posterior_sensor_rmse < run.initial_sensor_rmse,
            "posterior sensor RMSE {} did not improve over initial {}",
            run.posterior_sensor_rmse,
            run.initial_sensor_rmse
        );
        assert!(run.final_factorization.nnz > 0);
        assert_eq!(run.assembly.factor_nnz, Some(run.final_factorization.nnz));
        assert!(run.all_finite_variances);
        assert!(run.nonnegative_variances);
        assert_eq!(run.observation_variances.len(), run.observation_count);
        assert_eq!(
            run.published_steel_benchmark_reports.len(),
            TEAM13_OBSERVATION_COUNT
        );
        assert!(run.published_steel_benchmark_reports.iter().all(|report| {
            report.observed_g_052.is_finite()
                && report.observed_g_047.is_finite()
                && report.nominal_prediction.is_finite()
                && report.posterior_prediction.is_finite()
        }));
        assert_eq!(result.source_scale_diagnostics.len(), 1);
        assert!(result.source_scale_diagnostics[0]
            .steel_rmse_g_052
            .is_finite());
        assert!(result.source_scale_diagnostics[0]
            .steel_rmse_g_047
            .is_finite());
        let steel = run
            .group_summaries
            .iter()
            .find(|summary| summary.group == Team13SyntheticBenchmarkObservationGroup::SteelAverage)
            .expect("steel observation summary should be present");
        assert!(
            steel.posterior_rmse < steel.initial_rmse,
            "steel posterior RMSE {} did not improve over initial {}",
            steel.posterior_rmse,
            steel.initial_rmse
        );
        assert!(run
            .group_summaries
            .iter()
            .all(|summary| summary.posterior_rmse.is_finite()));
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn team13_source_recovery_gmsh_smoke_reports_finite_source_and_field_variances() {
        if Command::new("gmsh").arg("-version").output().is_err() {
            eprintln!("skipping TEAM 13 source-recovery smoke test because gmsh is not available");
            return;
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.join("../..");
        let geo = workspace.join("geometries/team13_linear.geo");
        let out_dir = workspace.join("target/team13_source_recovery_smoke");
        fs::create_dir_all(&out_dir).unwrap();
        let mesh_path = out_dir.join("team13_half_source_recovery.msh");
        let status = Command::new("gmsh")
            .arg("-3")
            .arg(&geo)
            .arg("-setnumber")
            .arg("FullDomain")
            .arg("0")
            .arg("-setnumber")
            .arg("MeshScale")
            .arg("14")
            .arg("-o")
            .arg(&mesh_path)
            .status()
            .unwrap();
        assert!(status.success());

        let result = run_team13_source_recovery_experiment(&Team13SourceRecoveryConfig {
            mesh_path,
            output_dir: Some(out_dir.join("out")),
            run_eight_mode_recovery: true,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::MonteCarlo,
                    num_variance_probes: 2,
                    variance_batch_count: 1,
                    rng_seed: 11,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: false,
            },
            ..Team13SourceRecoveryConfig::default()
        })
        .unwrap();

        assert_eq!(result.stages.len(), 7);
        assert_eq!(result.field_prior_comparisons.len(), 2);
        assert!(result.field_prior_comparisons.iter().any(|comparison| {
            comparison.prior_kind == Team13FieldPriorKind::UnweightedHodgeMatern
        }));
        assert!(result.field_prior_comparisons.iter().any(|comparison| {
            comparison.prior_kind == Team13FieldPriorKind::SplitGraphHodgeMatern
        }));
        assert!(result
            .field_prior_comparisons
            .iter()
            .all(|comparison| comparison.all_finite_variances
                && comparison.nonnegative_variances
                && comparison.source_posterior.posterior_mean.is_finite()
                && comparison.source_posterior.posterior_variance.is_finite()
                && comparison.prior_factor_nnz > 0
                && comparison.posterior_factor_nnz > 0));
        assert!(result
            .stages
            .iter()
            .any(|stage| stage.summary.name == "S1_unweighted_direct_source_recovery"));
        assert!(result
            .stages
            .iter()
            .any(|stage| stage.summary.name == "S1_split_graph_direct_source_recovery"));
        assert!(result
            .stages
            .iter()
            .any(|stage| stage.summary.name == "S2_fluctuation_source_recovery"));
        assert!(result.source_posterior.posterior_mean.is_finite());
        assert!(result.source_posterior.posterior_variance.is_finite());
        assert!(result.source_posterior.posterior_variance >= 0.0);
        assert!(
            result.source_posterior.posterior_variance < result.source_posterior.prior_variance
        );
        assert!(result.fluctuation_source_posterior.recovery_error.abs() < 0.03);
        assert_eq!(
            result.source_posterior.posterior_mean,
            result.baseline_source_posterior.posterior_mean
        );
        assert!(result.baseline_source_posterior.posterior_mean.is_finite());
        assert!(result.source_scaling_proxy.posterior_mean.is_finite());
        let eight_mode = result
            .eight_mode
            .as_ref()
            .expect("eight-mode recovery should be reported");
        assert_eq!(eight_mode.source_modes.len(), COIL_MODE_COUNT);
        assert_eq!(eight_mode.fluctuation_source_modes.len(), COIL_MODE_COUNT);
        assert_eq!(eight_mode.observability.len(), COIL_MODE_COUNT);
        assert!(eight_mode
            .source_modes
            .iter()
            .all(|mode| mode.posterior_mean.is_finite()
                && mode.posterior_variance.is_finite()
                && mode.posterior_variance >= 0.0
                && mode.posterior_variance < mode.prior_variance));
        assert!(eight_mode.fluctuation_source_modes.iter().all(|mode| mode
            .posterior_mean
            .is_finite()
            && mode.posterior_variance.is_finite()
            && mode.posterior_variance >= 0.0));
        let eight_mode_fluctuation = eight_mode
            .fluctuation_stage
            .as_ref()
            .expect("eight-mode fluctuation stage should be reported");
        assert!(eight_mode
            .stage
            .summary
            .a_active_relative_l2_error
            .is_finite());
        assert!(eight_mode
            .stage
            .summary
            .b_cochain_relative_l2_error
            .is_finite());
        assert!(eight_mode
            .stage
            .summary
            .b_vector_relative_l2_error
            .is_finite());
        assert!(eight_mode_fluctuation
            .summary
            .b_vector_relative_l2_error
            .is_finite());
        for stage in &result.stages {
            assert!(!stage.summary.field_reference.is_empty());
            assert!(!stage.period_diagnostics.is_empty());
            assert!(stage.summary.pde_residual_norm.is_finite());
            assert!(stage.summary.sensor_rmse.is_finite());
            assert!(stage.summary.a_active_rmse.is_finite());
            assert!(stage.summary.a_active_relative_l2_error.is_finite());
            assert!(stage.summary.b_cochain_rmse.is_finite());
            assert!(stage.summary.b_cochain_relative_l2_error.is_finite());
            assert!(stage.summary.b_vector_rmse.is_finite());
            assert!(stage.summary.b_vector_relative_l2_error.is_finite());
            assert!(stage
                .solve
                .posterior
                .posterior_variance
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0));
            assert!(stage
                .period_diagnostics
                .iter()
                .all(|report| report.prediction.is_finite()
                    && report.posterior_variance.is_finite()
                    && report.posterior_variance >= 0.0));
        }
        let output_dir = out_dir.join("out");
        assert!(output_dir.join("stage_summary.csv").exists());
        assert!(output_dir.join("prior_comparison.csv").exists());
        let stage_summary = fs::read_to_string(output_dir.join("stage_summary.csv")).unwrap();
        assert!(stage_summary.contains("a_active_relative_l2_error"));
        assert!(stage_summary.contains("b_vector_relative_l2_error"));
        assert!(stage_summary.contains("S1_unweighted_direct_source_recovery"));
        assert!(stage_summary.contains("S1_split_graph_direct_source_recovery"));
        assert!(stage_summary.contains("S2_fluctuation_source_recovery"));
        let prior_comparison = fs::read_to_string(output_dir.join("prior_comparison.csv")).unwrap();
        assert!(prior_comparison.contains("unweighted-hodge-matern"));
        assert!(prior_comparison.contains("split-graph-hodge-matern"));
        assert!(output_dir.join("source_posterior.csv").exists());
        assert!(output_dir.join("source_scaling_proxy.csv").exists());
        assert!(output_dir.join("source_posterior_baseline.csv").exists());
        assert!(output_dir.join("source_posterior_fluctuation.csv").exists());
        assert!(output_dir.join("sensor_uncertainty.csv").exists());
        assert!(output_dir.join("flux_uncertainty.csv").exists());
        assert!(output_dir.join("period_diagnostic.csv").exists());
        let period_diagnostic =
            fs::read_to_string(output_dir.join("period_diagnostic.csv")).unwrap();
        assert!(period_diagnostic.contains(SOURCE_FREE_H_TOP_LOOP_NAME));
        assert!(output_dir
            .join("eight_mode/source_mode_posterior.csv")
            .exists());
        assert!(output_dir
            .join("eight_mode/source_mode_posterior_fluctuation.csv")
            .exists());
        let eight_mode_stage_summary =
            fs::read_to_string(output_dir.join("eight_mode/stage_summary.csv")).unwrap();
        assert!(eight_mode_stage_summary.contains("M2_fluctuation_eight_mode_recovery"));
        assert!(output_dir.join("eight_mode/period_diagnostic.csv").exists());
        assert!(output_dir
            .join("eight_mode/source_mode_observability.csv")
            .exists());
        assert!(output_dir
            .join("eight_mode/replacement_decision.json")
            .exists());
    }
}
