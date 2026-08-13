//! Planar holes Hodge--GMRF flow experiment.
//!
//! This case study compares a decomposed FEEC Hodge--Matérn 1-form prior with
//! nondecomposed and componentwise baselines on a square with circular holes.

use crate::{de_rham, visual_output};
use common::linalg::nalgebra::{
    bilinear_form_sparse, CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix,
    Vector as FeecVector,
};
use ddf::cochain::Cochain;
use faer::Mat;
use feg_core::HodgeBranchKind;
use feg_gp::{
    BoundaryConditionSpec, HodgeBranchConfig as SpectralHodgeBranchConfig,
    HodgeBranchEnergyNormalization, HodgeBranchFeatureStats, HodgeBuildOptions,
    HodgeCompositionalConfig, HodgeCompositionalGp, HodgeDecomposedBasis,
};
use feg_infer::{
    prior::{
        hodge::{
            build_coexact_1form_transform_with_coords, build_exact_1form_transform,
            build_exact_mass_coexact_1form_transform,
            build_hodge_projection_operator_1form_with_basis, build_mass_projection_operator_1form,
            compute_harmonic_basis_1form, mass_orthonormalize_harmonic_basis_1form,
            transformed_mass_expected_energy_from_precision, HodgeProjectionOperator,
        },
        matern::{
            one_form::{
                build_exact_dense_mass_inverse_1form, build_hodge_laplacian_1form,
                build_matern_precision_1form_for_alpha_with_coords,
                build_reconstructed_barycenter_field_operator, MaternConfig as Matern1FormConfig,
                MaternMassInverse as Matern1FormMassInverse,
            },
            two_form::{
                build_hodge_laplacian_2form_with_lower_mass_inverse_coords,
                build_hodge_laplacian_2form_with_lower_mass_inverse_matrix,
                build_matern_mass_inverse_2form_with_coords,
                build_matern_precision_2form_for_alpha_with_coords,
                build_matern_system_matrix_2form, MaternConfig as Matern2FormConfig,
                MaternMassInverse as Matern2FormMassInverse,
            },
            zero_form::{
                build_exact_dense_mass_inverse_0form, build_laplace_beltrami_0form,
                build_matern_mass_inverse_0form, build_matern_precision_0form,
                build_matern_system_matrix_0form, MaternConfig as Matern0FormConfig,
                MaternMassInverse as Matern0FormMassInverse,
            },
            MaternAlpha,
        },
        sparse_anchor_hodge::{
            anchor_hodge_potential_branch, build_ordinary_potential_hodge_1form_prior,
            build_sparse_anchor_hodge_1form_prior_with_coords,
            spectrum_matched_potential_precision, OrdinaryPotentialHodge1FormPriorConfig,
            SparseAnchorBranchConfig, SparseAnchorHodge1FormPrior,
            SparseAnchorHodge1FormPriorConfig,
        },
    },
    sparse::{
        add_sparse, block_diag_feec_csr, dense_to_feec_csr, feec_csr_to_dense, feec_csr_to_gmrf,
        feec_vec_to_gmrf, gmrf_vec_to_feec, hstack_feec_csr, scale_matrix,
        sparse_row_operator_from_feec_csr,
    },
};
use gmrf_core::{
    observation::apply_gaussian_observations,
    types::{
        CholeskyOrdering, DenseMatrix as GmrfDenseMatrix, SparseMatrix as GmrfSparseMatrix,
        Vector as GmrfVector,
    },
    Gmrf,
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
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

const EPS: f64 = 1e-12;
const PERIOD_Z: f64 = 1.96;
const FIELD_COVERAGE_Z: f64 = 1.96;
const FIELD_VARIANCE_FLOOR: f64 = 1e-14;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const HARMONIC_COEFFICIENTS: [f64; 3] = [0.75, -0.55, 0.40];
const DEFAULT_HELDOUT_LOOP_RADIUS_OFFSET: f64 = 0.025;
const DEFAULT_HELDOUT_LOOP_ANGLE_OFFSET: f64 = std::f64::consts::PI / 16.0;
const DEFAULT_SPECTRAL_MODE_COUNT: usize = 160;
const RELATIVE_DENOM_EPS: f64 = 1e-10;
const INCOMPRESSIBLE_TRUTH_CODIFFERENTIAL_LEAKAGE_TOL: f64 = 1e-6;
const SUBSPACE_EIGEN_TOLERANCE: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesCoexactTruthSource {
    SparseAnchor,
    ExactMassCoexact,
    ExactLowerMassCoexact,
    SpectralGp,
    DirichletStreamfunction,
}

impl PlanarHolesCoexactTruthSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SparseAnchor => "sparse_anchor",
            Self::ExactMassCoexact => "exact_mass_coexact",
            Self::ExactLowerMassCoexact => "exact_lower_mass_coexact",
            Self::SpectralGp => "spectral_gp",
            Self::DirichletStreamfunction => "dirichlet_streamfunction",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesExactTruthSource {
    SparseAnchor,
    ExactDenseGmrf,
    SpectralGp,
    AnalyticPotential,
}

impl PlanarHolesExactTruthSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SparseAnchor => "sparse_anchor",
            Self::ExactDenseGmrf => "exact_dense_gmrf",
            Self::SpectralGp => "spectral_gp",
            Self::AnalyticPotential => "analytic_potential",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesHarmonicTruthSource {
    CanonicalFixed,
    SpectralGp,
}

impl PlanarHolesHarmonicTruthSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalFixed => "canonical_fixed",
            Self::SpectralGp => "spectral_gp",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesTruthScaling {
    MassNormTargets,
    RawPriorSamples,
}

impl PlanarHolesTruthScaling {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MassNormTargets => "mass_norm_targets",
            Self::RawPriorSamples => "raw_prior_samples",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesLocalObservationDesign {
    SparseInterior,
    HalfInteriorEdges,
    AllEdges,
}

impl PlanarHolesLocalObservationDesign {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SparseInterior => "sparse_interior",
            Self::HalfInteriorEdges => "half_interior_edges",
            Self::AllEdges => "all_edges",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesSpectralBoundaryCondition {
    Free,
    StrongBoundaryOneForms,
}

impl PlanarHolesSpectralBoundaryCondition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::StrongBoundaryOneForms => "strong_boundary_one_forms",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesFlowConfig {
    pub mesh_path: PathBuf,
    pub geo_path: PathBuf,
    pub output_dir: PathBuf,
    pub force_mesh: bool,
    pub mesh_size: f64,
    pub exact_kappa: f64,
    pub coexact_kappa: f64,
    pub nondecomposed_kappa: f64,
    pub component_kappa: f64,
    pub tau: f64,
    pub exact_tau_scale: f64,
    pub coexact_tau_scale: f64,
    pub alpha: MaternAlpha,
    pub harmonic_precision: f64,
    pub local_observation_count: usize,
    pub heldout_local_count: usize,
    pub local_noise_variance: f64,
    pub loop_noise_variance: f64,
    pub heldout_loop_radius_offset: f64,
    pub heldout_loop_angle_offset: f64,
    pub rng_seed: u64,
    pub exact_truth_mass_norm: f64,
    pub coexact_truth_mass_norm: f64,
    pub harmonic_truth_mass_norm: f64,
    pub exact_truth_source: PlanarHolesExactTruthSource,
    pub coexact_truth_source: PlanarHolesCoexactTruthSource,
    pub harmonic_truth_source: PlanarHolesHarmonicTruthSource,
    pub truth_scaling: PlanarHolesTruthScaling,
    pub local_observation_design: PlanarHolesLocalObservationDesign,
    pub sample_observation_noise: bool,
    pub include_exact_hodge_model: bool,
    pub include_exact_dense_exact_hodge_model: bool,
    pub include_exact_dense_trace_matched_exact_hodge_model: bool,
    pub include_incompressible_hodge_model: bool,
    pub include_exact_mass_incompressible_hodge_model: bool,
    pub include_sparse_lower_trace_matched_incompressible_hodge_model: bool,
    pub include_exact_lower_incompressible_hodge_model: bool,
    pub include_exact_lower_trace_matched_incompressible_hodge_model: bool,
    pub include_spectral_exact_hodge_model: bool,
    pub include_spectral_hodge_model: bool,
    pub include_spectral_incompressible_hodge_model: bool,
    pub include_naive_euclidean_vector_matern_model: bool,
    pub compute_field_coverage: bool,
    pub spectral_boundary_condition: PlanarHolesSpectralBoundaryCondition,
    pub spectral_exact_mode_count: usize,
    pub spectral_coexact_mode_count: usize,
    pub spectral_harmonic_mode_count: usize,
    pub spectral_branch_energy_normalization: bool,
    pub spectral_exact_expected_m1_energy: f64,
    pub spectral_coexact_expected_m1_energy: f64,
    pub spectral_harmonic_expected_m1_energy: f64,
}

impl Default for PlanarHolesFlowConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let output_dir = manifest_dir.join("../../out/planar_holes_hodge_flow");
        Self {
            mesh_path: output_dir.join("planar_holes.msh"),
            geo_path: output_dir.join("planar_holes.geo"),
            output_dir,
            force_mesh: true,
            mesh_size: 0.045,
            exact_kappa: 1.25,
            coexact_kappa: 8.0,
            nondecomposed_kappa: 4.0,
            component_kappa: 4.0,
            tau: 1.0,
            exact_tau_scale: 1.0,
            coexact_tau_scale: 1.0,
            alpha: MaternAlpha::Two,
            harmonic_precision: 1.0,
            local_observation_count: 600,
            heldout_local_count: 200,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
            heldout_loop_radius_offset: DEFAULT_HELDOUT_LOOP_RADIUS_OFFSET,
            heldout_loop_angle_offset: DEFAULT_HELDOUT_LOOP_ANGLE_OFFSET,
            rng_seed: 2026,
            exact_truth_mass_norm: 1.0,
            coexact_truth_mass_norm: 0.8,
            harmonic_truth_mass_norm: 0.7,
            exact_truth_source: PlanarHolesExactTruthSource::SparseAnchor,
            coexact_truth_source: PlanarHolesCoexactTruthSource::SparseAnchor,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            local_observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: true,
            include_exact_hodge_model: false,
            include_exact_dense_exact_hodge_model: false,
            include_exact_dense_trace_matched_exact_hodge_model: false,
            include_incompressible_hodge_model: true,
            include_exact_mass_incompressible_hodge_model: false,
            include_sparse_lower_trace_matched_incompressible_hodge_model: false,
            include_exact_lower_incompressible_hodge_model: false,
            include_exact_lower_trace_matched_incompressible_hodge_model: false,
            include_spectral_exact_hodge_model: false,
            include_spectral_hodge_model: false,
            include_spectral_incompressible_hodge_model: false,
            include_naive_euclidean_vector_matern_model: false,
            compute_field_coverage: true,
            spectral_boundary_condition: PlanarHolesSpectralBoundaryCondition::Free,
            spectral_exact_mode_count: DEFAULT_SPECTRAL_MODE_COUNT,
            spectral_coexact_mode_count: DEFAULT_SPECTRAL_MODE_COUNT,
            spectral_harmonic_mode_count: HARMONIC_COEFFICIENTS.len(),
            spectral_branch_energy_normalization: false,
            spectral_exact_expected_m1_energy: 1.0,
            spectral_coexact_expected_m1_energy: 1.0,
            spectral_harmonic_expected_m1_energy: 0.49,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanarHolesModelKind {
    HodgeMatern,
    ExactHodgeMatern,
    ExactDenseExactHodgeMatern,
    ExactDenseTraceMatchedExactHodgeMatern,
    IncompressibleHodgeMatern,
    ExactMassIncompressibleHodgeMatern,
    SparseLowerTraceMatchedIncompressibleHodgeMatern,
    ExactLowerIncompressibleHodgeMatern,
    ExactLowerTraceMatchedIncompressibleHodgeMatern,
    SpectralExactHodgeGp,
    SpectralHodgeGp,
    SpectralIncompressibleHodgeGp,
    NondecomposedFeec,
    ComponentwiseMatern,
    PostHocProjectedVectorMatern,
    NaiveEuclideanVectorMatern,
}

impl PlanarHolesModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HodgeMatern => "feec_hodge_matern",
            Self::ExactHodgeMatern => "feec_exact_matern",
            Self::ExactDenseExactHodgeMatern => "feec_exact_dense_0form_exact_matern",
            Self::ExactDenseTraceMatchedExactHodgeMatern => {
                "feec_exact_dense_0form_trace_matched_exact_matern"
            }
            Self::IncompressibleHodgeMatern => "feec_coexact_harmonic_matern",
            Self::ExactMassIncompressibleHodgeMatern => "feec_exact_mass_coexact_harmonic_matern",
            Self::SparseLowerTraceMatchedIncompressibleHodgeMatern => {
                "feec_sparse_lower_trace_matched_coexact_harmonic_matern"
            }
            Self::ExactLowerIncompressibleHodgeMatern => "feec_exact_lower_coexact_harmonic_matern",
            Self::ExactLowerTraceMatchedIncompressibleHodgeMatern => {
                "feec_exact_lower_trace_matched_coexact_harmonic_matern"
            }
            Self::SpectralExactHodgeGp => "spectral_exact_gp",
            Self::SpectralHodgeGp => "spectral_hodge_gp",
            Self::SpectralIncompressibleHodgeGp => "spectral_coexact_harmonic_gp",
            Self::NondecomposedFeec => "nondecomposed_feec_hodge_matern",
            Self::ComponentwiseMatern => "componentwise_scalar_matern",
            Self::PostHocProjectedVectorMatern => "posthoc_projected_vector_matern",
            Self::NaiveEuclideanVectorMatern => "naive_euclidean_vector_matern",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanarHolesObservationScenario {
    LocalOnly,
    LocalPlusLoops,
    LocalPlusHarmonicPeriods,
}

impl PlanarHolesObservationScenario {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LocalOnly => "local_only",
            Self::LocalPlusLoops => "local_plus_loops",
            Self::LocalPlusHarmonicPeriods => "local_plus_harmonic_periods",
        }
    }

    fn all() -> [Self; 3] {
        [
            Self::LocalOnly,
            Self::LocalPlusLoops,
            Self::LocalPlusHarmonicPeriods,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesCodifferentialMetricKind {
    RelativeError,
    Leakage,
}

impl PlanarHolesCodifferentialMetricKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RelativeError => "relative_error",
            Self::Leakage => "leakage",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesSpectralBranchDiagnostic {
    pub model: PlanarHolesModelKind,
    pub branch: HodgeBranchKind,
    pub requested_mode_count: usize,
    pub actual_mode_count: usize,
    pub unnormalized_expected_m1_energy: f64,
    pub target_expected_m1_energy: f64,
    pub normalization_scale: f64,
    pub expected_m1_energy: f64,
    pub truth_mass_norm: f64,
    pub projected_truth_mass_norm: f64,
    pub projection_relative_error: f64,
    pub projected_truth_mahalanobis_norm: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesSpectralEnergyDiagnostic {
    pub model: PlanarHolesModelKind,
    pub branch: HodgeBranchKind,
    pub mode_index: usize,
    pub cumulative_projected_energy: f64,
    pub cumulative_projected_energy_fraction: f64,
    pub cumulative_prior_energy: f64,
    pub cumulative_prior_energy_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesPriorEquivalenceConfig {
    pub base: PlanarHolesFlowConfig,
    pub mode_counts: Vec<usize>,
    pub include_max_available: bool,
    pub include_exact_lower_mass: bool,
    pub target_coexact_expected_m1_energy: f64,
    pub dense_eigen_dimension_cap: usize,
}

impl Default for PlanarHolesPriorEquivalenceConfig {
    fn default() -> Self {
        Self {
            base: PlanarHolesFlowConfig::default(),
            mode_counts: vec![64, 160, 512, 1000],
            include_max_available: true,
            include_exact_lower_mass: true,
            target_coexact_expected_m1_energy: 1.0,
            dense_eigen_dimension_cap: 600,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesExactPriorEquivalenceConfig {
    pub base: PlanarHolesFlowConfig,
    pub mode_counts: Vec<usize>,
    pub include_max_available: bool,
    pub target_exact_expected_m1_energy: f64,
    pub dense_eigen_dimension_cap: usize,
}

impl Default for PlanarHolesExactPriorEquivalenceConfig {
    fn default() -> Self {
        Self {
            base: PlanarHolesFlowConfig::default(),
            mode_counts: vec![64, 160, 512, 1000],
            include_max_available: true,
            target_exact_expected_m1_energy: 1.0,
            dense_eigen_dimension_cap: 600,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesPriorEquivalenceSpectralReference {
    Raw,
    EnergyNormalized,
}

impl PlanarHolesPriorEquivalenceSpectralReference {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "spectral_raw",
            Self::EnergyNormalized => "spectral_energy_normalized",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesPriorEquivalenceGmrfVariant {
    SparseLower,
    SparseLowerTraceMatched,
    ExactLower,
    ExactLowerTraceMatched,
    ExactLumped,
    ExactLumpedTraceMatched,
    ExactDense,
    ExactDenseTraceMatched,
    ExactOrdinaryPotentialDense,
    ExactOrdinaryPotentialDenseTraceMatched,
}

impl PlanarHolesPriorEquivalenceGmrfVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SparseLower => "exact_mass_gmrf_sparse_lower",
            Self::SparseLowerTraceMatched => "exact_mass_gmrf_sparse_lower_trace_matched",
            Self::ExactLower => "exact_mass_gmrf_exact_lower",
            Self::ExactLowerTraceMatched => "exact_mass_gmrf_exact_lower_trace_matched",
            Self::ExactLumped => "exact_gmrf_lumped_0form",
            Self::ExactLumpedTraceMatched => "exact_gmrf_lumped_0form_trace_matched",
            Self::ExactDense => "exact_gmrf_dense_0form",
            Self::ExactDenseTraceMatched => "exact_gmrf_dense_0form_trace_matched",
            Self::ExactOrdinaryPotentialDense => "exact_ordinary_potential_dense_0form",
            Self::ExactOrdinaryPotentialDenseTraceMatched => {
                "exact_ordinary_potential_dense_0form_trace_matched"
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesPriorEquivalenceRow {
    pub requested_mode_count: usize,
    pub actual_mode_count: usize,
    pub spectral_reference: PlanarHolesPriorEquivalenceSpectralReference,
    pub gmrf_variant: PlanarHolesPriorEquivalenceGmrfVariant,
    pub spectral_expected_m1_energy: f64,
    pub spectral_unnormalized_expected_m1_energy: f64,
    pub spectral_normalization_scale: f64,
    pub gmrf_expected_m1_energy: f64,
    pub gmrf_tau_scale: f64,
    pub required_tau_scale_to_match_spectral_trace: f64,
    pub trace_relative_error: f64,
    pub m1_frobenius_relative_error: f64,
    pub diag_variance_ratio_min: f64,
    pub diag_variance_ratio_mean: f64,
    pub diag_variance_ratio_max: f64,
    pub source_eigen_count: usize,
    pub source_eigen_relative_l2_error: f64,
    pub source_eigen_max_relative_error: f64,
    pub covariance_eigen_count: usize,
    pub spectral_covariance_eigen_min: f64,
    pub spectral_covariance_eigen_max: f64,
    pub gmrf_covariance_eigen_min: f64,
    pub gmrf_covariance_eigen_max: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesModelMetrics {
    pub scenario: PlanarHolesObservationScenario,
    pub model: PlanarHolesModelKind,
    pub observation_count: usize,
    pub l2_error_absolute: f64,
    pub l2_error: f64,
    pub heldout_nlpd: f64,
    pub heldout_local_nlpd: f64,
    pub heldout_loop_nlpd: f64,
    pub heldout_harmonic_period_nlpd: f64,
    pub exterior_derivative_error_absolute: f64,
    pub exterior_derivative_truth_norm: f64,
    pub exterior_derivative_error: f64,
    pub codifferential_error_absolute: f64,
    pub codifferential_truth_norm: f64,
    pub codifferential_error_relative: f64,
    pub codifferential_leakage: f64,
    pub codifferential_error: f64,
    pub codifferential_metric_kind: PlanarHolesCodifferentialMetricKind,
    pub relative_circulation_error: f64,
    pub relative_harmonic_period_error: f64,
    pub relative_coexact_annular_error: f64,
    pub relative_total_annular_error: f64,
    pub mean_abs_circulation_error: f64,
    pub max_abs_circulation_error: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesTopologySummary {
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub euler_characteristic: isize,
    pub b0: usize,
    pub b1: usize,
    pub b2: usize,
    pub boundary_edge_count: usize,
}

#[derive(Debug, Clone)]
pub struct PlanarHoleSpec {
    pub name: String,
    pub center: [f64; 2],
    pub radius: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHoleCycle {
    pub family: String,
    pub hole_index: usize,
    pub name: String,
    pub radius: f64,
    pub angle_offset: f64,
    pub target_vertices: Vec<usize>,
    pub path_vertices: Vec<usize>,
    pub edge_count: usize,
    pub closure_residual_l1: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesCoexactTransformDiagnostics {
    pub sparse_inverse: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub sparse_coexact_m1_operator_norm: f64,
    pub exact_mass_coexact_m1_operator_norm: f64,
    pub sparse_coexact_codifferential_leakage: f64,
    pub exact_mass_coexact_codifferential_leakage: f64,
    pub sparse_exact_branch_mass_orthogonality: f64,
    pub exact_mass_exact_branch_mass_orthogonality: f64,
    pub sparse_vs_exact_mass_transform_relative_m1_error: f64,
    pub sparse_coexact_rank: usize,
    pub exact_mass_coexact_rank: usize,
    pub principal_cosine_min: f64,
    pub principal_cosine_mean: f64,
    pub principal_angle_max_degrees: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanarHolesObservationKind {
    Local,
    TrainLoop,
    TrainHarmonicPeriod,
    TrainInteriorLoop,
    TrainLongPath,
    TrainPathHomology,
    HeldoutLocal,
    HeldoutLoop,
    HeldoutHarmonicPeriod,
    HeldoutInteriorLoop,
    HeldoutLongPath,
    HeldoutPathHomology,
    HeldoutPathContrast,
}

impl PlanarHolesObservationKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::TrainLoop => "train_loop",
            Self::TrainHarmonicPeriod => "train_harmonic_period",
            Self::TrainInteriorLoop => "train_interior_loop",
            Self::TrainLongPath => "train_long_path",
            Self::TrainPathHomology => "train_path_homology",
            Self::HeldoutLocal => "heldout_local",
            Self::HeldoutLoop => "heldout_loop",
            Self::HeldoutHarmonicPeriod => "heldout_harmonic_period",
            Self::HeldoutInteriorLoop => "heldout_interior_loop",
            Self::HeldoutLongPath => "heldout_long_path",
            Self::HeldoutPathHomology => "heldout_path_homology",
            Self::HeldoutPathContrast => "heldout_path_contrast",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesObservation {
    pub kind: PlanarHolesObservationKind,
    pub label: String,
    pub entries: Vec<(usize, f64)>,
    pub observed_value: f64,
    pub truth_value: f64,
    pub noise_variance: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesPeriodSummary {
    pub scenario: PlanarHolesObservationScenario,
    pub model: PlanarHolesModelKind,
    pub hole_index: usize,
    pub hole_name: String,
    pub truth_period: f64,
    pub posterior_mean: f64,
    pub posterior_std: f64,
    pub posterior_lower_95: f64,
    pub posterior_upper_95: f64,
    pub residual: f64,
    pub abs_residual_over_std: f64,
    pub covered_95: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanarHolesLoopFunctionalKind {
    TotalAnnular,
    HarmonicPeriod,
    CoexactAnnular,
}

impl PlanarHolesLoopFunctionalKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::TotalAnnular => "total_annular",
            Self::HarmonicPeriod => "harmonic_period",
            Self::CoexactAnnular => "coexact_annular",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesLoopFunctionalSummary {
    pub scenario: PlanarHolesObservationScenario,
    pub model: PlanarHolesModelKind,
    pub functional: PlanarHolesLoopFunctionalKind,
    pub hole_index: usize,
    pub hole_name: String,
    pub truth_value: f64,
    pub posterior_mean: f64,
    pub posterior_std: f64,
    pub posterior_lower_95: f64,
    pub posterior_upper_95: f64,
    pub residual: f64,
    pub abs_residual_over_std: f64,
    pub covered_95: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanarHolesFieldCoverageSubset {
    AllEdges,
    HeldoutLocalEdges,
}

impl PlanarHolesFieldCoverageSubset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllEdges => "all_edges",
            Self::HeldoutLocalEdges => "heldout_local_edges",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesFieldCoverageSummary {
    pub scenario: PlanarHolesObservationScenario,
    pub model: PlanarHolesModelKind,
    pub subset: PlanarHolesFieldCoverageSubset,
    pub edge_count: usize,
    pub weight_sum: f64,
    pub coverage_95: f64,
    pub mass_weighted_coverage_95: f64,
    pub mean_abs_z: f64,
    pub rms_z: f64,
    pub p95_abs_z: f64,
    pub mean_posterior_std: f64,
    pub mass_weighted_mean_posterior_std: f64,
    pub latent_nlpd: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesFieldCoverageDiagnostics {
    pub posterior_std: FeecVector,
    pub abs_z_score: FeecVector,
    pub covered_95: FeecVector,
    pub posterior_mean_error: FeecVector,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesBranchRecoverySummary {
    pub scenario: PlanarHolesObservationScenario,
    pub model: PlanarHolesModelKind,
    pub branch: HodgeBranchKind,
    pub truth_mass_norm: f64,
    pub posterior_mass_norm: f64,
    pub error_mass_norm: f64,
    pub relative_error: f64,
    pub mass_correlation: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesHeldoutPrediction {
    pub scenario: PlanarHolesObservationScenario,
    pub model: PlanarHolesModelKind,
    pub kind: PlanarHolesObservationKind,
    pub label: String,
    pub observed_value: f64,
    pub truth_value: f64,
    pub predictive_mean: f64,
    pub predictive_variance: f64,
    pub nlpd: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesModelPosterior {
    pub scenario: PlanarHolesObservationScenario,
    pub model: PlanarHolesModelKind,
    pub posterior_mean: FeecVector,
    pub branch_means: BTreeMap<HodgeBranchKind, FeecVector>,
    pub field_coverage: PlanarHolesFieldCoverageDiagnostics,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesScenarioResult {
    pub scenario: PlanarHolesObservationScenario,
    pub observation_count: usize,
    pub metrics: Vec<PlanarHolesModelMetrics>,
    pub period_summaries: Vec<PlanarHolesPeriodSummary>,
    pub loop_functional_summaries: Vec<PlanarHolesLoopFunctionalSummary>,
    pub field_coverage_summaries: Vec<PlanarHolesFieldCoverageSummary>,
    pub branch_recovery_summaries: Vec<PlanarHolesBranchRecoverySummary>,
    pub heldout_predictions: Vec<PlanarHolesHeldoutPrediction>,
    pub posteriors: Vec<PlanarHolesModelPosterior>,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesFlowResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub metric: MeshLengths,
    pub topology_summary: PlanarHolesTopologySummary,
    pub holes: Vec<PlanarHoleSpec>,
    pub truth: FeecVector,
    pub truth_exact: FeecVector,
    pub truth_coexact: FeecVector,
    pub truth_harmonic: FeecVector,
    pub harmonic_basis: FeecMatrix,
    pub cycle_observation_matrix: FeecCsr,
    pub cycles: Vec<PlanarHoleCycle>,
    pub cycle_harmonic_pairing_rank: usize,
    pub heldout_cycle_observation_matrix: FeecCsr,
    pub heldout_cycles: Vec<PlanarHoleCycle>,
    pub heldout_cycle_harmonic_pairing_rank: usize,
    pub local_observations: Vec<PlanarHolesObservation>,
    pub heldout_observations: Vec<PlanarHolesObservation>,
    pub spectral_branch_diagnostics: Vec<PlanarHolesSpectralBranchDiagnostic>,
    pub spectral_energy_diagnostics: Vec<PlanarHolesSpectralEnergyDiagnostic>,
    pub scenarios: Vec<PlanarHolesScenarioResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanarHolesSpectralTruthFamily {
    SpectralCoexact,
    SparseAnchorCoexact,
    ExactMassCoexactKappa8,
    ExactMassCoexactKappa2,
    StreamfunctionDirichlet,
}

impl PlanarHolesSpectralTruthFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SpectralCoexact => "spectral_coexact",
            Self::SparseAnchorCoexact => "sparse_anchor_coexact",
            Self::ExactMassCoexactKappa8 => "exact_mass_coexact_kappa_8",
            Self::ExactMassCoexactKappa2 => "exact_mass_coexact_kappa_2",
            Self::StreamfunctionDirichlet => "streamfunction_dirichlet",
        }
    }

    pub fn all() -> [Self; 5] {
        [
            Self::SpectralCoexact,
            Self::SparseAnchorCoexact,
            Self::ExactMassCoexactKappa8,
            Self::ExactMassCoexactKappa2,
            Self::StreamfunctionDirichlet,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesSpectralTruthCompatibilityConfig {
    pub base: PlanarHolesFlowConfig,
    pub mode_stages: Vec<usize>,
    pub include_max_available: bool,
    pub spectral_truth_mode_count: usize,
    pub families: Vec<PlanarHolesSpectralTruthFamily>,
    pub boundary_conditions: Vec<PlanarHolesSpectralBoundaryCondition>,
}

impl Default for PlanarHolesSpectralTruthCompatibilityConfig {
    fn default() -> Self {
        Self {
            base: PlanarHolesFlowConfig::default(),
            mode_stages: vec![32, 64, 128, 256, 512, 1000],
            include_max_available: true,
            spectral_truth_mode_count: 1000,
            families: PlanarHolesSpectralTruthFamily::all().to_vec(),
            boundary_conditions: vec![
                PlanarHolesSpectralBoundaryCondition::Free,
                PlanarHolesSpectralBoundaryCondition::StrongBoundaryOneForms,
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesSpectralTruthCompatibilityRow {
    pub truth_family: PlanarHolesSpectralTruthFamily,
    pub boundary_condition: PlanarHolesSpectralBoundaryCondition,
    pub requested_mode_count: usize,
    pub actual_mode_count: usize,
    pub truth_mass_norm: f64,
    pub projected_truth_mass_norm: f64,
    pub projected_energy_fraction: f64,
    pub projection_relative_error: f64,
    pub projected_truth_mahalanobis_norm: f64,
    pub codifferential_leakage: f64,
    pub boundary_lumped_energy_fraction: f64,
    pub unnormalized_expected_m1_energy_sq: f64,
    pub target_expected_m1_energy_sq: f64,
    pub expected_m1_energy_sq: f64,
    pub normalization_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanarHolesSensorDesignKind {
    SparseEdges,
    EdgesHolePeriods,
    EdgesSmallInteriorLoops,
    EdgesMultiscaleInteriorLoops,
    EdgesLongPaths,
    Hybrid,
}

impl PlanarHolesSensorDesignKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SparseEdges => "sparse_edges",
            Self::EdgesHolePeriods => "edges_hole_periods",
            Self::EdgesSmallInteriorLoops => "edges_small_interior_loops",
            Self::EdgesMultiscaleInteriorLoops => "edges_multiscale_interior_loops",
            Self::EdgesLongPaths => "edges_long_paths",
            Self::Hybrid => "hybrid_edges_loops_paths_periods",
        }
    }

    pub fn all() -> [Self; 6] {
        [
            Self::SparseEdges,
            Self::EdgesHolePeriods,
            Self::EdgesSmallInteriorLoops,
            Self::EdgesMultiscaleInteriorLoops,
            Self::EdgesLongPaths,
            Self::Hybrid,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesSensorDesignSweepConfig {
    pub base: PlanarHolesFlowConfig,
    pub total_observation_budget: usize,
    pub heldout_local_count: usize,
    pub heldout_interior_loop_count: usize,
    pub heldout_long_path_count: usize,
    pub interior_loop_noise_variance: f64,
    pub long_path_noise_variance: f64,
    pub designs: Vec<PlanarHolesSensorDesignKind>,
}

impl Default for PlanarHolesSensorDesignSweepConfig {
    fn default() -> Self {
        Self {
            base: PlanarHolesFlowConfig::default(),
            total_observation_budget: 600,
            heldout_local_count: 200,
            heldout_interior_loop_count: 80,
            heldout_long_path_count: 80,
            interior_loop_noise_variance: 1e-5,
            long_path_noise_variance: 1e-4,
            designs: PlanarHolesSensorDesignKind::all().to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesSensorDesignSweepRow {
    pub design: PlanarHolesSensorDesignKind,
    pub model: PlanarHolesModelKind,
    pub model_coexact_tau_scale: f64,
    pub observation_count: usize,
    pub edge_observation_count: usize,
    pub interior_loop_observation_count: usize,
    pub long_path_observation_count: usize,
    pub harmonic_period_observation_count: usize,
    pub l2_error: f64,
    pub heldout_local_nlpd: f64,
    pub heldout_interior_loop_nlpd: f64,
    pub heldout_long_path_nlpd: f64,
    pub heldout_harmonic_period_nlpd: f64,
    pub heldout_interior_loop_relative_error: f64,
    pub heldout_long_path_relative_error: f64,
    pub heldout_harmonic_period_relative_error: f64,
    pub codifferential_leakage: f64,
    pub coexact_relative_error: f64,
    pub coexact_mass_correlation: f64,
    pub harmonic_relative_error: f64,
    pub harmonic_mass_correlation: f64,
    pub all_edge_coverage_95: f64,
    pub heldout_edge_coverage_95: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesTopologyVsNaiveGpConfig {
    pub base: PlanarHolesFlowConfig,
    pub total_observation_budget: usize,
    pub validation_local_count: usize,
    pub validation_interior_loop_count: usize,
    pub validation_long_path_count: usize,
    pub heldout_local_count: usize,
    pub heldout_interior_loop_count: usize,
    pub heldout_long_path_count: usize,
    pub interior_loop_noise_variance: f64,
    pub long_path_noise_variance: f64,
    pub hodge_kappas: Vec<f64>,
    pub hodge_taus: Vec<f64>,
    pub naive_kappas: Vec<f64>,
    pub naive_taus: Vec<f64>,
    pub observation_variance_scales: Vec<f64>,
}

impl Default for PlanarHolesTopologyVsNaiveGpConfig {
    fn default() -> Self {
        let mut base = PlanarHolesFlowConfig::default();
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../out/planar_holes_topology_vs_naive_gp");
        base.output_dir = output_dir.clone();
        base.mesh_path = output_dir.join("planar_holes_topology_vs_naive_gp.msh");
        base.geo_path = output_dir.join("planar_holes_topology_vs_naive_gp.geo");
        base.force_mesh = true;
        base.mesh_size = 0.045;
        base.exact_truth_mass_norm = 0.0;
        base.coexact_truth_mass_norm = 1.0;
        base.harmonic_truth_mass_norm = 2.0;
        base.coexact_truth_source = PlanarHolesCoexactTruthSource::DirichletStreamfunction;
        base.harmonic_truth_source = PlanarHolesHarmonicTruthSource::CanonicalFixed;
        base.truth_scaling = PlanarHolesTruthScaling::MassNormTargets;
        base.sample_observation_noise = false;
        base.compute_field_coverage = true;
        Self {
            base,
            total_observation_budget: 120,
            validation_local_count: 30,
            validation_interior_loop_count: 60,
            validation_long_path_count: 40,
            heldout_local_count: 40,
            heldout_interior_loop_count: 160,
            heldout_long_path_count: 80,
            interior_loop_noise_variance: 1e-5,
            long_path_noise_variance: 1e-4,
            hodge_kappas: vec![1.5, 3.0, 6.0, 10.0],
            hodge_taus: vec![0.03, 0.05, 0.1, 0.2],
            naive_kappas: vec![3.0, 6.0, 10.0],
            naive_taus: vec![0.5, 1.0, 2.0],
            observation_variance_scales: vec![10.0, 1.0, 0.1],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesTopologyVsNaiveGpMetricRow {
    pub model: PlanarHolesModelKind,
    pub selected_kappa: f64,
    pub selected_tau: f64,
    pub selected_observation_variance_scale: f64,
    pub model_coexact_tau_scale: f64,
    pub spectral_coexact_requested_modes: usize,
    pub spectral_coexact_actual_modes: usize,
    pub spectral_coexact_expected_m1_energy: f64,
    pub spectral_harmonic_requested_modes: usize,
    pub spectral_harmonic_actual_modes: usize,
    pub spectral_harmonic_expected_m1_energy: f64,
    pub validation_nlpd: f64,
    pub training_observation_count: usize,
    pub validation_observation_count: usize,
    pub heldout_observation_count: usize,
    pub l2_error: f64,
    pub heldout_nlpd: f64,
    pub heldout_local_nlpd: f64,
    pub heldout_loop_nlpd: f64,
    pub heldout_interior_loop_nlpd: f64,
    pub heldout_long_path_nlpd: f64,
    pub heldout_local_relative_error: f64,
    pub heldout_loop_relative_error: f64,
    pub heldout_interior_loop_relative_error: f64,
    pub heldout_long_path_relative_error: f64,
    pub exterior_derivative_error: f64,
    pub codifferential_leakage: f64,
    pub all_edge_coverage_95: f64,
    pub heldout_edge_coverage_95: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesTopologyVsNaiveGpTuningRow {
    pub model: PlanarHolesModelKind,
    pub kappa: f64,
    pub tau: f64,
    pub observation_variance_scale: f64,
    pub validation_nlpd: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesTopologyVsNaiveGpCalibrationRow {
    pub model: PlanarHolesModelKind,
    pub kind: String,
    pub count: usize,
    pub variance_multiplier: f64,
    pub raw_nlpd: f64,
    pub calibrated_nlpd: f64,
    pub raw_coverage_95: f64,
    pub calibrated_coverage_95: f64,
    pub raw_mean_abs_z: f64,
    pub calibrated_mean_abs_z: f64,
    pub relative_error: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesTopologyVsNaiveGpResult {
    pub topology_summary: PlanarHolesTopologySummary,
    pub train_cycle_harmonic_pairing_rank: usize,
    pub validation_cycle_harmonic_pairing_rank: usize,
    pub heldout_cycle_harmonic_pairing_rank: usize,
    pub rows: Vec<PlanarHolesTopologyVsNaiveGpMetricRow>,
    pub tuning_rows: Vec<PlanarHolesTopologyVsNaiveGpTuningRow>,
    pub calibration_rows: Vec<PlanarHolesTopologyVsNaiveGpCalibrationRow>,
    pub heldout_predictions: Vec<PlanarHolesHeldoutPrediction>,
    pub field_coverage_summaries: Vec<PlanarHolesFieldCoverageSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarHolesDomainKind {
    Default,
    Barrier,
}

impl PlanarHolesDomainKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default_three_holes",
            Self::Barrier => "vertical_barrier_three_holes",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanarHolesBarrierSummaryRow {
    pub model: PlanarHolesModelKind,
    pub train_left_edge_count: usize,
    pub validation_left_edge_count: usize,
    pub heldout_right_edge_count: usize,
    pub heldout_cross_barrier_path_count: usize,
    pub cross_barrier_local_nlpd: f64,
    pub cross_barrier_local_relative_error: f64,
    pub barrier_long_path_nlpd: f64,
    pub barrier_long_path_relative_error: f64,
    pub hole_loop_nlpd: f64,
    pub hole_loop_relative_error: f64,
    pub cross_barrier_local_coverage_95: f64,
    pub barrier_long_path_coverage_95: f64,
    pub hole_loop_coverage_95: f64,
    pub calibrated_cross_barrier_local_nlpd: f64,
    pub calibrated_barrier_long_path_nlpd: f64,
    pub calibrated_hole_loop_nlpd: f64,
    pub calibrated_cross_barrier_local_coverage_95: f64,
    pub calibrated_barrier_long_path_coverage_95: f64,
    pub calibrated_hole_loop_coverage_95: f64,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesBarrierTopologyVsNaiveGpResult {
    pub base: PlanarHolesTopologyVsNaiveGpResult,
    pub domain: PlanarHolesDomainKind,
    pub rows: Vec<PlanarHolesBarrierSummaryRow>,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesPathHomologySummaryRow {
    pub model: PlanarHolesModelKind,
    pub train_path_pair_count: usize,
    pub validation_path_pair_count: usize,
    pub heldout_path_pair_count: usize,
    pub path_integral_nlpd: f64,
    pub path_integral_relative_error: f64,
    pub path_integral_coverage_95: f64,
    pub path_contrast_nlpd: f64,
    pub path_contrast_relative_error: f64,
    pub path_contrast_coverage_95: f64,
    pub path_contrast_mean_abs_z: f64,
    pub hole_loop_nlpd: f64,
    pub hole_loop_relative_error: f64,
    pub calibrated_path_integral_nlpd: f64,
    pub calibrated_path_integral_coverage_95: f64,
    pub calibrated_path_contrast_nlpd: f64,
    pub calibrated_path_contrast_coverage_95: f64,
    pub calibrated_path_contrast_mean_abs_z: f64,
    pub calibrated_hole_loop_nlpd: f64,
    pub calibrated_hole_loop_coverage_95: f64,
    pub path_contrast_harmonic_pairing_rank: usize,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesPathHomologyTopologyVsNaiveGpResult {
    pub base: PlanarHolesTopologyVsNaiveGpResult,
    pub domain: PlanarHolesDomainKind,
    pub path_contrast_harmonic_pairing_rank: usize,
    pub rows: Vec<PlanarHolesPathHomologySummaryRow>,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesVectorFieldFigureSummaryRow {
    pub model: PlanarHolesModelKind,
    pub l2_error: f64,
    pub heldout_nlpd: f64,
    pub calibrated_heldout_nlpd: f64,
    pub path_integral_relative_error: f64,
    pub path_contrast_relative_error: f64,
    pub hole_loop_relative_error: f64,
    pub codifferential_leakage: f64,
    pub all_edge_coverage_95: f64,
    pub heldout_edge_coverage_95: f64,
    pub posterior_vtu_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PlanarHolesPathHomologyVectorFieldsResult {
    pub topology_summary: PlanarHolesTopologySummary,
    pub train_cycle_harmonic_pairing_rank: usize,
    pub validation_cycle_harmonic_pairing_rank: usize,
    pub heldout_cycle_harmonic_pairing_rank: usize,
    pub path_contrast_harmonic_pairing_rank: usize,
    pub output_dir: PathBuf,
    pub truth_vtu_path: PathBuf,
    pub figure_png_path: PathBuf,
    pub summary_csv_path: PathBuf,
    pub rows: Vec<PlanarHolesVectorFieldFigureSummaryRow>,
}

#[derive(Debug, Clone)]
struct ModelOperators {
    model: PlanarHolesModelKind,
    prior_precision: FeecCsr,
    latent_to_ambient: FeecCsr,
    branch_transforms: BTreeMap<HodgeBranchKind, (usize, FeecCsr)>,
    spectral_branch_stats: BTreeMap<HodgeBranchKind, HodgeBranchFeatureStats>,
    coexact_tau_scale: f64,
}

#[derive(Debug, Clone)]
struct ConditionedModel {
    model: PlanarHolesModelKind,
    coexact_tau_scale: f64,
    posterior_precision: GmrfSparseMatrix,
    latent_to_ambient: FeecCsr,
    posterior_mean: FeecVector,
    branch_means: BTreeMap<HodgeBranchKind, FeecVector>,
    branch_transforms: BTreeMap<HodgeBranchKind, (usize, FeecCsr)>,
}

#[derive(Debug, Clone)]
struct VectorFieldPanelData {
    title: String,
    cochain: FeecVector,
    l2_error: Option<f64>,
    path_contrast_relative_error: Option<f64>,
    hole_loop_relative_error: Option<f64>,
    codifferential_leakage: Option<f64>,
}

#[derive(Debug, Clone)]
struct ComputedFieldCoverage {
    summaries: Vec<PlanarHolesFieldCoverageSummary>,
    diagnostics: PlanarHolesFieldCoverageDiagnostics,
}

#[derive(Debug, Clone)]
struct CompatibilityProjectionBasis {
    boundary_condition: PlanarHolesSpectralBoundaryCondition,
    requested_mode_count: usize,
    transform: FeecCsr,
    stats: HodgeBranchFeatureStats,
}

#[derive(Debug, Clone)]
struct SensorRow {
    label: String,
    entries: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PathHomologyPair {
    family: String,
    hole_index: usize,
    hole_name: String,
    upper: SensorRow,
    lower: SensorRow,
    contrast: SensorRow,
    shared_start_vertex: usize,
    shared_end_vertex: usize,
    closure_residual_l1: f64,
}

#[derive(Debug, Clone, Copy)]
struct SensorDesignCounts {
    edge_count: usize,
    interior_loop_count: usize,
    long_path_count: usize,
    harmonic_period_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct EdgeStep {
    to: usize,
    edge_index: usize,
    sign: f64,
    length: f64,
}

#[derive(Debug, Clone)]
struct ShortestPath {
    vertices: Vec<usize>,
    edges: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct QueueState {
    cost: f64,
    vertex: usize,
}

impl Eq for QueueState {}

impl Ord for QueueState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl PartialOrd for QueueState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn one_form_mass_inverse_name(strategy: Matern1FormMassInverse) -> &'static str {
    match strategy {
        Matern1FormMassInverse::RowSumLumped => "row_sum_lumped",
        Matern1FormMassInverse::Nc1ProjectedSparseInverse => "nc1_projected_sparse_inverse",
        Matern1FormMassInverse::BarycentricDualSparseInverse => "barycentric_dual_sparse_inverse",
    }
}

pub fn run_planar_holes_coexact_transform_diagnostics(
    config: &PlanarHolesFlowConfig,
) -> Result<PlanarHolesCoexactTransformDiagnostics, Box<dyn Error>> {
    run_planar_holes_coexact_transform_diagnostics_for_inverse(
        config,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
    )
}

pub fn run_planar_holes_coexact_transform_diagnostics_for_inverse(
    config: &PlanarHolesFlowConfig,
    one_form_mass_inverse: Matern1FormMassInverse,
) -> Result<PlanarHolesCoexactTransformDiagnostics, Box<dyn Error>> {
    validate_config(config)?;
    ensure_planar_holes_mesh(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = summarize_topology(&topology);
    validate_topology(&topology_summary)?;

    build_planar_holes_coexact_transform_diagnostics_for_inverse(
        &topology,
        &coords,
        &metric,
        one_form_mass_inverse,
    )
}

pub fn run_planar_holes_coexact_transform_diagnostics_all(
    config: &PlanarHolesFlowConfig,
) -> Result<Vec<PlanarHolesCoexactTransformDiagnostics>, Box<dyn Error>> {
    validate_config(config)?;
    ensure_planar_holes_mesh(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = summarize_topology(&topology);
    validate_topology(&topology_summary)?;

    [
        Matern1FormMassInverse::RowSumLumped,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
    ]
    .into_iter()
    .map(|strategy| {
        build_planar_holes_coexact_transform_diagnostics_for_inverse(
            &topology, &coords, &metric, strategy,
        )
    })
    .collect()
}

fn build_planar_holes_coexact_transform_diagnostics_for_inverse(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    one_form_mass_inverse: Matern1FormMassInverse,
) -> Result<PlanarHolesCoexactTransformDiagnostics, Box<dyn Error>> {
    let hodge = build_hodge_laplacian_1form(topology, metric);
    let mass_0form = de_rham::mass_matrix_form(topology, metric, 0).map_err(invalid_data)?;
    let exact_transform = build_exact_1form_transform(topology);
    let sparse_coexact_transform = build_coexact_1form_transform_with_coords(
        topology,
        coords,
        metric,
        &hodge.mass_u,
        one_form_mass_inverse,
    )
    .map_err(invalid_data)?;
    let exact_mass_coexact_transform =
        build_exact_mass_coexact_1form_transform(topology, metric, &hodge.mass_u)
            .map_err(invalid_data)?;

    let sparse_norm = operator_mass_norm(&sparse_coexact_transform, &hodge.mass_u);
    let exact_mass_norm = operator_mass_norm(&exact_mass_coexact_transform, &hodge.mass_u);
    let difference = add_sparse(
        &sparse_coexact_transform,
        &scale_matrix(&exact_mass_coexact_transform, -1.0),
    );
    let subspace = mass_principal_angle_summary(
        &sparse_coexact_transform,
        &exact_mass_coexact_transform,
        &hodge.mass_u,
    );

    Ok(PlanarHolesCoexactTransformDiagnostics {
        sparse_inverse: one_form_mass_inverse_name(one_form_mass_inverse).to_string(),
        vertex_count: topology.vertices().len(),
        edge_count: topology.edges().len(),
        face_count: topology.cells().len(),
        sparse_coexact_m1_operator_norm: sparse_norm,
        exact_mass_coexact_m1_operator_norm: exact_mass_norm,
        sparse_coexact_codifferential_leakage: relative_or_nan(
            codifferential_operator_norm(
                &hodge.mass_u,
                &mass_0form,
                &exact_transform,
                &sparse_coexact_transform,
            )?,
            sparse_norm,
        ),
        exact_mass_coexact_codifferential_leakage: relative_or_nan(
            codifferential_operator_norm(
                &hodge.mass_u,
                &mass_0form,
                &exact_transform,
                &exact_mass_coexact_transform,
            )?,
            exact_mass_norm,
        ),
        sparse_exact_branch_mass_orthogonality: mass_orthogonality_ratio(
            &exact_transform,
            &sparse_coexact_transform,
            &hodge.mass_u,
        ),
        exact_mass_exact_branch_mass_orthogonality: mass_orthogonality_ratio(
            &exact_transform,
            &exact_mass_coexact_transform,
            &hodge.mass_u,
        ),
        sparse_vs_exact_mass_transform_relative_m1_error: relative_or_nan(
            operator_mass_norm(&difference, &hodge.mass_u),
            exact_mass_norm,
        ),
        sparse_coexact_rank: subspace.left_rank,
        exact_mass_coexact_rank: subspace.right_rank,
        principal_cosine_min: subspace.min_cosine,
        principal_cosine_mean: subspace.mean_cosine,
        principal_angle_max_degrees: subspace.max_angle_degrees,
    })
}

pub fn run_planar_holes_hodge_flow(
    config: &PlanarHolesFlowConfig,
) -> Result<PlanarHolesFlowResult, Box<dyn Error>> {
    validate_config(config)?;
    ensure_planar_holes_mesh(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = summarize_topology(&topology);
    validate_topology(&topology_summary)?;
    let holes = default_holes();

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw = compute_harmonic_basis_1form(&topology, &metric, holes.len(), None)
        .map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;

    let (cycle_observation_matrix, cycles) =
        build_hole_cycle_observation_matrix(&topology, &coords, &holes, 0.0, 0.0, "train")?;
    let harmonic_basis = canonicalize_harmonic_basis_by_cycles(
        &harmonic_basis_orthonormal,
        &cycle_observation_matrix,
    )?;
    let cycle_harmonic_pairing = &cycle_observation_matrix * &harmonic_basis;
    let cycle_harmonic_pairing_rank = cycle_harmonic_pairing.rank(1e-6);
    if cycle_harmonic_pairing_rank != holes.len() {
        return Err(invalid_data(format!(
            "cycle-harmonic pairing rank {cycle_harmonic_pairing_rank} does not match hole count {}",
            holes.len()
        ))
        .into());
    }
    let (heldout_cycle_observation_matrix, heldout_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        config.heldout_loop_radius_offset,
        config.heldout_loop_angle_offset,
        "heldout",
    )?;
    let heldout_cycle_harmonic_pairing = &heldout_cycle_observation_matrix * &harmonic_basis;
    let heldout_cycle_harmonic_pairing_rank = heldout_cycle_harmonic_pairing.rank(1e-6);
    if heldout_cycle_harmonic_pairing_rank != holes.len() {
        return Err(invalid_data(format!(
            "held-out cycle-harmonic pairing rank {heldout_cycle_harmonic_pairing_rank} does not match hole count {}",
            holes.len()
        ))
        .into());
    }
    if matching_cycle_rows_identical(&cycle_observation_matrix, &heldout_cycle_observation_matrix) {
        return Err(invalid_data(
            "training and held-out loop observation rows should not be identical",
        )
        .into());
    }

    let hodge_prior = build_hodge_prior(&topology, &coords, &metric, &harmonic_basis, config)?;
    let truth = build_truth(
        &topology,
        &coords,
        &metric,
        &hodge.mass_u,
        &hodge_prior,
        config,
    )
    .map_err(|err| invalid_data(format!("truth construction failed: {err}")))?;
    let projection = build_hodge_projection_operator_1form_with_basis(
        &topology,
        &metric,
        &hodge.mass_u,
        harmonic_basis.clone(),
        1e-10,
    )
    .map_err(invalid_data)?;
    let harmonic_period_observation_matrix = &cycle_observation_matrix * &projection.harmonic;
    let heldout_harmonic_period_observation_matrix =
        &heldout_cycle_observation_matrix * &projection.harmonic;
    let model_operators = build_model_operators(
        &topology,
        &coords,
        &metric,
        &hodge,
        &hodge_prior,
        &projection,
        &harmonic_basis,
        config,
    )
    .map_err(|err| invalid_data(format!("model operator construction failed: {err}")))?;
    let (spectral_branch_diagnostics, spectral_energy_diagnostics) =
        compute_spectral_diagnostics(&model_operators, &truth, &hodge.mass_u)?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.rng_seed + 31);
    let mut cycle_edge_set = observation_edge_set(&cycle_observation_matrix);
    cycle_edge_set.extend(observation_edge_set(&heldout_cycle_observation_matrix));
    let local_edges = match config.local_observation_design {
        PlanarHolesLocalObservationDesign::SparseInterior => select_observation_edges(
            &topology,
            &coords,
            &cycle_edge_set,
            &BTreeSet::new(),
            config.local_observation_count,
            &mut rng,
        ),
        PlanarHolesLocalObservationDesign::HalfInteriorEdges => select_fraction_observation_edges(
            &topology,
            &coords,
            &cycle_edge_set,
            &BTreeSet::new(),
            0.5,
            &mut rng,
        ),
        PlanarHolesLocalObservationDesign::AllEdges => {
            (0..topology.edges().len()).collect::<Vec<_>>()
        }
    };
    let local_used = local_edges.iter().copied().collect::<BTreeSet<_>>();
    let heldout_edges = select_observation_edges(
        &topology,
        &coords,
        &cycle_edge_set,
        &local_used,
        config.heldout_local_count,
        &mut rng,
    );

    let local_observations = edge_observations(
        PlanarHolesObservationKind::Local,
        "local",
        &local_edges,
        &truth.mixed,
        config.local_noise_variance,
        config.sample_observation_noise,
        &mut rng,
    );
    let loop_observation_rows = loop_observations(
        PlanarHolesObservationKind::TrainLoop,
        "train_loop",
        &cycle_observation_matrix,
        &cycles,
        &truth.mixed,
        config.loop_noise_variance,
        config.sample_observation_noise,
        &mut rng,
    );
    let harmonic_period_observation_rows = loop_observations(
        PlanarHolesObservationKind::TrainHarmonicPeriod,
        "train_harmonic_period",
        &harmonic_period_observation_matrix,
        &cycles,
        &truth.mixed,
        config.loop_noise_variance,
        config.sample_observation_noise,
        &mut rng,
    );
    let heldout_observations = {
        let mut observations = edge_observations(
            PlanarHolesObservationKind::HeldoutLocal,
            "heldout_local",
            &heldout_edges,
            &truth.mixed,
            config.local_noise_variance,
            config.sample_observation_noise,
            &mut rng,
        );
        observations.extend(loop_observations(
            PlanarHolesObservationKind::HeldoutLoop,
            "heldout_loop",
            &heldout_cycle_observation_matrix,
            &heldout_cycles,
            &truth.mixed,
            config.loop_noise_variance,
            config.sample_observation_noise,
            &mut rng,
        ));
        observations.extend(loop_observations(
            PlanarHolesObservationKind::HeldoutHarmonicPeriod,
            "heldout_harmonic_period",
            &heldout_harmonic_period_observation_matrix,
            &heldout_cycles,
            &truth.mixed,
            config.loop_noise_variance,
            config.sample_observation_noise,
            &mut rng,
        ));
        observations
    };

    let mut scenarios = Vec::new();
    for scenario in PlanarHolesObservationScenario::all() {
        let mut training = local_observations.clone();
        match scenario {
            PlanarHolesObservationScenario::LocalOnly => {}
            PlanarHolesObservationScenario::LocalPlusLoops => {
                training.extend(loop_observation_rows.clone());
            }
            PlanarHolesObservationScenario::LocalPlusHarmonicPeriods => {
                training.extend(harmonic_period_observation_rows.clone());
            }
        }
        scenarios.push(
            run_scenario(
                scenario,
                &topology,
                &metric,
                &hodge.mass_u,
                &truth,
                &cycle_observation_matrix,
                &heldout_cycle_observation_matrix,
                &heldout_cycles,
                &training,
                &heldout_observations,
                &model_operators,
                config.compute_field_coverage,
            )
            .map_err(|err| invalid_data(format!("scenario {} failed: {err}", scenario.as_str())))?,
        );
    }

    Ok(PlanarHolesFlowResult {
        topology,
        coords,
        metric,
        topology_summary,
        holes,
        truth: truth.mixed,
        truth_exact: truth.exact,
        truth_coexact: truth.coexact,
        truth_harmonic: truth.harmonic,
        harmonic_basis,
        cycle_observation_matrix,
        cycles,
        cycle_harmonic_pairing_rank,
        heldout_cycle_observation_matrix,
        heldout_cycles,
        heldout_cycle_harmonic_pairing_rank,
        local_observations,
        heldout_observations,
        spectral_branch_diagnostics,
        spectral_energy_diagnostics,
        scenarios,
    })
}

pub fn run_planar_holes_spectral_truth_compatibility(
    config: &PlanarHolesSpectralTruthCompatibilityConfig,
) -> Result<Vec<PlanarHolesSpectralTruthCompatibilityRow>, Box<dyn Error>> {
    validate_config(&config.base)?;
    ensure_planar_holes_mesh(&config.base)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    validate_topology(&summarize_topology(&topology))?;

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let holes = default_holes();
    let harmonic_basis_raw = compute_harmonic_basis_1form(&topology, &metric, holes.len(), None)
        .map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;
    let (cycle_observation_matrix, _) =
        build_hole_cycle_observation_matrix(&topology, &coords, &holes, 0.0, 0.0, "train")?;
    let harmonic_basis = canonicalize_harmonic_basis_by_cycles(
        &harmonic_basis_orthonormal,
        &cycle_observation_matrix,
    )?;
    let hodge_prior =
        build_hodge_prior(&topology, &coords, &metric, &harmonic_basis, &config.base)?;

    let max_available = topology.vertices().len().max(topology.cells().len());
    let mut stages = config.mode_stages.clone();
    if config.include_max_available && !stages.contains(&max_available) {
        stages.push(max_available);
    }
    stages.sort_unstable();
    stages.dedup();

    let projection_bases = build_compatibility_projection_bases(
        &topology,
        &metric,
        &stages,
        &config.boundary_conditions,
        &config.base,
    )?;

    let mut rows = Vec::new();
    for family in &config.families {
        let truth_config = truth_family_config(config, *family, max_available);
        let truth = build_truth(
            &topology,
            &coords,
            &metric,
            &hodge.mass_u,
            &hodge_prior,
            &truth_config,
        )
        .map_err(|err| {
            invalid_data(format!(
                "truth compatibility family {} failed: {err}",
                family.as_str()
            ))
        })?;
        let truth_coexact = truth.coexact;
        let truth_mass_norm = mass_norm(&truth_coexact, &hodge.mass_u);
        let codifferential_leakage =
            codifferential_leakage(&topology, &metric, &hodge.mass_u, &truth_coexact)?;
        let boundary_lumped_energy_fraction =
            boundary_lumped_energy_fraction(&topology, &hodge.mass_u, &truth_coexact);

        for projection_basis in &projection_bases {
            let (diagnostic, _) = spectral_projection_diagnostic(
                PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
                HodgeBranchKind::Coexact,
                &projection_basis.transform,
                projection_basis.stats,
                &truth_coexact,
                &hodge.mass_u,
            )?;
            rows.push(PlanarHolesSpectralTruthCompatibilityRow {
                truth_family: *family,
                boundary_condition: projection_basis.boundary_condition,
                requested_mode_count: projection_basis.requested_mode_count,
                actual_mode_count: diagnostic.actual_mode_count,
                truth_mass_norm,
                projected_truth_mass_norm: diagnostic.projected_truth_mass_norm,
                projected_energy_fraction: relative_or_nan(
                    diagnostic.projected_truth_mass_norm * diagnostic.projected_truth_mass_norm,
                    truth_mass_norm * truth_mass_norm,
                ),
                projection_relative_error: diagnostic.projection_relative_error,
                projected_truth_mahalanobis_norm: diagnostic.projected_truth_mahalanobis_norm,
                codifferential_leakage,
                boundary_lumped_energy_fraction,
                unnormalized_expected_m1_energy_sq: diagnostic.unnormalized_expected_m1_energy,
                target_expected_m1_energy_sq: diagnostic.target_expected_m1_energy,
                expected_m1_energy_sq: diagnostic.expected_m1_energy,
                normalization_scale: diagnostic.normalization_scale,
            });
        }
    }

    Ok(rows)
}

fn build_compatibility_projection_bases(
    topology: &Complex,
    metric: &MeshLengths,
    stages: &[usize],
    boundary_conditions: &[PlanarHolesSpectralBoundaryCondition],
    base_config: &PlanarHolesFlowConfig,
) -> Result<Vec<CompatibilityProjectionBasis>, Box<dyn Error>> {
    let boundary_edges = boundary_edge_indices(topology);
    let mut projection_bases = Vec::new();
    for boundary_condition in boundary_conditions {
        for requested_mode_count in stages {
            let mut spectral_config = base_config.clone();
            spectral_config.exact_truth_mass_norm = 0.0;
            spectral_config.coexact_truth_mass_norm = 1.0;
            spectral_config.harmonic_truth_mass_norm = 0.0;
            spectral_config.spectral_exact_mode_count = *requested_mode_count;
            spectral_config.spectral_coexact_mode_count = *requested_mode_count;
            spectral_config.spectral_harmonic_mode_count = HARMONIC_COEFFICIENTS.len();
            spectral_config.spectral_boundary_condition = *boundary_condition;
            spectral_config.spectral_branch_energy_normalization = true;
            spectral_config.spectral_exact_expected_m1_energy = 1.0;
            spectral_config.spectral_coexact_expected_m1_energy = 1.0;
            spectral_config.spectral_harmonic_expected_m1_energy = 0.49;
            let basis = build_spectral_hodge_basis(topology, metric, 0, &spectral_config)?;
            let operators = build_spectral_hodge_model_operator(
                basis,
                &spectral_config,
                PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
                &[HodgeBranchKind::Coexact],
            )?;
            let (_, transform) = operators
                .branch_transforms
                .get(&HodgeBranchKind::Coexact)
                .ok_or_else(|| invalid_data("spectral coexact transform is missing"))?;
            let transform = expand_spectral_transform_for_boundary_condition(
                transform,
                topology.edges().len(),
                *boundary_condition,
                &boundary_edges,
            )?;
            let stats = operators
                .spectral_branch_stats
                .get(&HodgeBranchKind::Coexact)
                .copied()
                .ok_or_else(|| invalid_data("spectral coexact stats are missing"))?;
            projection_bases.push(CompatibilityProjectionBasis {
                boundary_condition: *boundary_condition,
                requested_mode_count: *requested_mode_count,
                transform,
                stats,
            });
        }
    }
    Ok(projection_bases)
}

fn expand_spectral_transform_for_boundary_condition(
    transform: &FeecCsr,
    full_rows: usize,
    boundary_condition: PlanarHolesSpectralBoundaryCondition,
    boundary_edges: &BTreeSet<usize>,
) -> Result<FeecCsr, Box<dyn Error>> {
    if transform.nrows() == full_rows {
        return Ok(transform.clone());
    }
    if boundary_condition != PlanarHolesSpectralBoundaryCondition::StrongBoundaryOneForms {
        return Err(invalid_data(format!(
            "spectral transform has {} rows but expected {full_rows}",
            transform.nrows()
        ))
        .into());
    }
    if transform.nrows() + boundary_edges.len() != full_rows {
        return Err(invalid_data(format!(
            "strong-boundary reduced transform has {} rows, boundary has {}, full edge count is {full_rows}",
            transform.nrows(),
            boundary_edges.len()
        ))
        .into());
    }
    let mut reduced_to_full = Vec::with_capacity(transform.nrows());
    for edge in 0..full_rows {
        if !boundary_edges.contains(&edge) {
            reduced_to_full.push(edge);
        }
    }
    let mut coo = FeecCoo::new(full_rows, transform.ncols());
    for (row, col, value) in transform.triplet_iter() {
        coo.push(reduced_to_full[row], col, *value);
    }
    Ok(FeecCsr::from(&coo))
}

fn truth_family_config(
    config: &PlanarHolesSpectralTruthCompatibilityConfig,
    family: PlanarHolesSpectralTruthFamily,
    max_available: usize,
) -> PlanarHolesFlowConfig {
    let mut truth_config = config.base.clone();
    truth_config.exact_truth_mass_norm = 0.0;
    truth_config.coexact_truth_mass_norm = 1.0;
    truth_config.harmonic_truth_mass_norm = 0.0;
    truth_config.truth_scaling = PlanarHolesTruthScaling::MassNormTargets;
    truth_config.spectral_branch_energy_normalization = true;
    truth_config.spectral_exact_expected_m1_energy = 1.0;
    truth_config.spectral_coexact_expected_m1_energy = 1.0;
    truth_config.spectral_harmonic_expected_m1_energy = 0.49;
    truth_config.spectral_exact_mode_count = config.spectral_truth_mode_count.min(max_available);
    truth_config.spectral_coexact_mode_count = config.spectral_truth_mode_count.min(max_available);
    truth_config.spectral_harmonic_mode_count = HARMONIC_COEFFICIENTS.len();
    truth_config.spectral_boundary_condition = PlanarHolesSpectralBoundaryCondition::Free;
    match family {
        PlanarHolesSpectralTruthFamily::SpectralCoexact => {
            truth_config.coexact_truth_source = PlanarHolesCoexactTruthSource::SpectralGp;
            truth_config.harmonic_truth_source = PlanarHolesHarmonicTruthSource::SpectralGp;
            truth_config.truth_scaling = PlanarHolesTruthScaling::RawPriorSamples;
        }
        PlanarHolesSpectralTruthFamily::SparseAnchorCoexact => {
            truth_config.coexact_truth_source = PlanarHolesCoexactTruthSource::SparseAnchor;
            truth_config.harmonic_truth_source = PlanarHolesHarmonicTruthSource::CanonicalFixed;
        }
        PlanarHolesSpectralTruthFamily::ExactMassCoexactKappa8 => {
            truth_config.coexact_truth_source = PlanarHolesCoexactTruthSource::ExactMassCoexact;
            truth_config.coexact_kappa = 8.0;
            truth_config.harmonic_truth_source = PlanarHolesHarmonicTruthSource::CanonicalFixed;
        }
        PlanarHolesSpectralTruthFamily::ExactMassCoexactKappa2 => {
            truth_config.coexact_truth_source = PlanarHolesCoexactTruthSource::ExactMassCoexact;
            truth_config.coexact_kappa = 2.0;
            truth_config.harmonic_truth_source = PlanarHolesHarmonicTruthSource::CanonicalFixed;
        }
        PlanarHolesSpectralTruthFamily::StreamfunctionDirichlet => {
            truth_config.coexact_truth_source =
                PlanarHolesCoexactTruthSource::DirichletStreamfunction;
            truth_config.harmonic_truth_source = PlanarHolesHarmonicTruthSource::CanonicalFixed;
        }
    }
    truth_config
}

pub fn write_planar_holes_spectral_truth_compatibility(
    rows: &[PlanarHolesSpectralTruthCompatibilityRow],
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "truth_family,boundary_condition,requested_mode_count,actual_mode_count,truth_mass_norm,projected_truth_mass_norm,projected_energy_fraction,projection_relative_error,projected_truth_mahalanobis_norm,codifferential_leakage,boundary_lumped_energy_fraction,unnormalized_expected_m1_energy_sq,target_expected_m1_energy_sq,expected_m1_energy_sq,normalization_scale"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.truth_family.as_str(),
            row.boundary_condition.as_str(),
            row.requested_mode_count,
            row.actual_mode_count,
            row.truth_mass_norm,
            row.projected_truth_mass_norm,
            row.projected_energy_fraction,
            row.projection_relative_error,
            row.projected_truth_mahalanobis_norm,
            row.codifferential_leakage,
            row.boundary_lumped_energy_fraction,
            row.unnormalized_expected_m1_energy_sq,
            row.target_expected_m1_energy_sq,
            row.expected_m1_energy_sq,
            row.normalization_scale
        )?;
    }
    Ok(())
}

pub fn run_planar_holes_prior_equivalence(
    config: &PlanarHolesPriorEquivalenceConfig,
) -> Result<Vec<PlanarHolesPriorEquivalenceRow>, Box<dyn Error>> {
    validate_config(&config.base)?;
    ensure_planar_holes_mesh(&config.base)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    validate_topology(&summarize_topology(&topology))?;

    let hodge_1form = build_hodge_laplacian_1form(&topology, &metric);
    let max_available = topology.vertices().len().max(topology.cells().len());
    let mut mode_counts = config.mode_counts.clone();
    if config.include_max_available && !mode_counts.contains(&max_available) {
        mode_counts.push(max_available);
    }
    mode_counts.sort_unstable();
    mode_counts.dedup();

    let exact_lower_mass_inverse = if config.include_exact_lower_mass {
        Some(
            build_exact_dense_mass_inverse_1form(&hodge_1form.mass_u, 1e-14)
                .map_err(invalid_data)?,
        )
    } else {
        None
    };

    let mut rows = Vec::new();
    for requested_mode_count in mode_counts {
        let spectral_raw = build_coexact_spectral_prior_covariance(
            &topology,
            &metric,
            &config.base,
            requested_mode_count,
            false,
            config.target_coexact_expected_m1_energy,
        )?;
        let spectral_normalized = build_coexact_spectral_prior_covariance(
            &topology,
            &metric,
            &config.base,
            requested_mode_count,
            true,
            config.target_coexact_expected_m1_energy,
        )?;

        let sparse_lower = build_coexact_gmrf_prior_covariance(
            &topology,
            &coords,
            &metric,
            &hodge_1form.mass_u,
            &config.base,
            Matern1FormMassInverse::Nc1ProjectedSparseInverse,
            None,
            1.0,
        )?;
        let exact_lower = match &exact_lower_mass_inverse {
            Some(lower_inverse) => Some(build_coexact_gmrf_prior_covariance(
                &topology,
                &coords,
                &metric,
                &hodge_1form.mass_u,
                &config.base,
                Matern1FormMassInverse::Nc1ProjectedSparseInverse,
                Some(lower_inverse),
                1.0,
            )?),
            None => None,
        };

        for spectral in [&spectral_raw, &spectral_normalized] {
            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &sparse_lower,
                PlanarHolesPriorEquivalenceGmrfVariant::SparseLower,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);

            let sparse_tau_scale =
                required_tau_scale(sparse_lower.expected_m1_energy, spectral.expected_m1_energy);
            let sparse_matched = build_coexact_gmrf_prior_covariance(
                &topology,
                &coords,
                &metric,
                &hodge_1form.mass_u,
                &config.base,
                Matern1FormMassInverse::Nc1ProjectedSparseInverse,
                None,
                sparse_tau_scale,
            )?;
            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &sparse_matched,
                PlanarHolesPriorEquivalenceGmrfVariant::SparseLowerTraceMatched,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);

            if let Some(exact_lower) = &exact_lower {
                rows.push(compare_coexact_priors(
                    requested_mode_count,
                    spectral,
                    exact_lower,
                    PlanarHolesPriorEquivalenceGmrfVariant::ExactLower,
                    &hodge_1form.mass_u,
                    config.dense_eigen_dimension_cap,
                )?);
                let exact_tau_scale =
                    required_tau_scale(exact_lower.expected_m1_energy, spectral.expected_m1_energy);
                let exact_matched = build_coexact_gmrf_prior_covariance(
                    &topology,
                    &coords,
                    &metric,
                    &hodge_1form.mass_u,
                    &config.base,
                    Matern1FormMassInverse::Nc1ProjectedSparseInverse,
                    exact_lower_mass_inverse.as_ref(),
                    exact_tau_scale,
                )?;
                rows.push(compare_coexact_priors(
                    requested_mode_count,
                    spectral,
                    &exact_matched,
                    PlanarHolesPriorEquivalenceGmrfVariant::ExactLowerTraceMatched,
                    &hodge_1form.mass_u,
                    config.dense_eigen_dimension_cap,
                )?);
            }
        }
    }

    Ok(rows)
}

pub fn run_planar_holes_exact_prior_equivalence(
    config: &PlanarHolesExactPriorEquivalenceConfig,
) -> Result<Vec<PlanarHolesPriorEquivalenceRow>, Box<dyn Error>> {
    validate_config(&config.base)?;
    ensure_planar_holes_mesh(&config.base)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    validate_topology(&summarize_topology(&topology))?;

    let hodge_1form = build_hodge_laplacian_1form(&topology, &metric);
    let max_available = topology.vertices().len().saturating_sub(1);
    let mut mode_counts = config.mode_counts.clone();
    if config.include_max_available && !mode_counts.contains(&max_available) {
        mode_counts.push(max_available);
    }
    mode_counts.sort_unstable();
    mode_counts.dedup();

    let exact_lumped = build_exact_gmrf_prior_covariance(
        &topology,
        &metric,
        &hodge_1form.mass_u,
        &config.base,
        Exact0FormMassInverseKind::Lumped,
        false,
        1.0,
    )?;
    let exact_dense = build_exact_gmrf_prior_covariance(
        &topology,
        &metric,
        &hodge_1form.mass_u,
        &config.base,
        Exact0FormMassInverseKind::Dense,
        false,
        1.0,
    )?;
    let ordinary_dense = build_exact_gmrf_prior_covariance(
        &topology,
        &metric,
        &hodge_1form.mass_u,
        &config.base,
        Exact0FormMassInverseKind::Dense,
        true,
        1.0,
    )?;

    let mut rows = Vec::new();
    for requested_mode_count in mode_counts {
        let spectral_raw = build_exact_spectral_prior_covariance(
            &topology,
            &metric,
            &config.base,
            requested_mode_count,
            false,
            config.target_exact_expected_m1_energy,
        )?;
        let spectral_normalized = build_exact_spectral_prior_covariance(
            &topology,
            &metric,
            &config.base,
            requested_mode_count,
            true,
            config.target_exact_expected_m1_energy,
        )?;

        for spectral in [&spectral_raw, &spectral_normalized] {
            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &exact_lumped,
                PlanarHolesPriorEquivalenceGmrfVariant::ExactLumped,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);
            let lumped_tau_scale =
                required_tau_scale(exact_lumped.expected_m1_energy, spectral.expected_m1_energy);
            let lumped_matched = build_exact_gmrf_prior_covariance(
                &topology,
                &metric,
                &hodge_1form.mass_u,
                &config.base,
                Exact0FormMassInverseKind::Lumped,
                false,
                lumped_tau_scale,
            )?;
            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &lumped_matched,
                PlanarHolesPriorEquivalenceGmrfVariant::ExactLumpedTraceMatched,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);

            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &exact_dense,
                PlanarHolesPriorEquivalenceGmrfVariant::ExactDense,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);
            let dense_tau_scale =
                required_tau_scale(exact_dense.expected_m1_energy, spectral.expected_m1_energy);
            let dense_matched = build_exact_gmrf_prior_covariance(
                &topology,
                &metric,
                &hodge_1form.mass_u,
                &config.base,
                Exact0FormMassInverseKind::Dense,
                false,
                dense_tau_scale,
            )?;
            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &dense_matched,
                PlanarHolesPriorEquivalenceGmrfVariant::ExactDenseTraceMatched,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);

            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &ordinary_dense,
                PlanarHolesPriorEquivalenceGmrfVariant::ExactOrdinaryPotentialDense,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);
            let ordinary_tau_scale = required_tau_scale(
                ordinary_dense.expected_m1_energy,
                spectral.expected_m1_energy,
            );
            let ordinary_matched = build_exact_gmrf_prior_covariance(
                &topology,
                &metric,
                &hodge_1form.mass_u,
                &config.base,
                Exact0FormMassInverseKind::Dense,
                true,
                ordinary_tau_scale,
            )?;
            rows.push(compare_coexact_priors(
                requested_mode_count,
                spectral,
                &ordinary_matched,
                PlanarHolesPriorEquivalenceGmrfVariant::ExactOrdinaryPotentialDenseTraceMatched,
                &hodge_1form.mass_u,
                config.dense_eigen_dimension_cap,
            )?);
        }
    }

    Ok(rows)
}

pub fn write_planar_holes_prior_equivalence(
    rows: &[PlanarHolesPriorEquivalenceRow],
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "requested_mode_count,actual_mode_count,spectral_reference,gmrf_variant,spectral_expected_m1_energy,spectral_unnormalized_expected_m1_energy,spectral_normalization_scale,gmrf_expected_m1_energy,gmrf_tau_scale,required_tau_scale_to_match_spectral_trace,trace_relative_error,m1_frobenius_relative_error,diag_variance_ratio_min,diag_variance_ratio_mean,diag_variance_ratio_max,source_eigen_count,source_eigen_relative_l2_error,source_eigen_max_relative_error,covariance_eigen_count,spectral_covariance_eigen_min,spectral_covariance_eigen_max,gmrf_covariance_eigen_min,gmrf_covariance_eigen_max"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12}",
            row.requested_mode_count,
            row.actual_mode_count,
            row.spectral_reference.as_str(),
            row.gmrf_variant.as_str(),
            row.spectral_expected_m1_energy,
            row.spectral_unnormalized_expected_m1_energy,
            row.spectral_normalization_scale,
            row.gmrf_expected_m1_energy,
            row.gmrf_tau_scale,
            row.required_tau_scale_to_match_spectral_trace,
            row.trace_relative_error,
            row.m1_frobenius_relative_error,
            row.diag_variance_ratio_min,
            row.diag_variance_ratio_mean,
            row.diag_variance_ratio_max,
            row.source_eigen_count,
            row.source_eigen_relative_l2_error,
            row.source_eigen_max_relative_error,
            row.covariance_eigen_count,
            row.spectral_covariance_eigen_min,
            row.spectral_covariance_eigen_max,
            row.gmrf_covariance_eigen_min,
            row.gmrf_covariance_eigen_max,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct CoexactPriorCovariance {
    spectral_reference: Option<PlanarHolesPriorEquivalenceSpectralReference>,
    actual_mode_count: usize,
    expected_m1_energy: f64,
    unnormalized_expected_m1_energy: f64,
    normalization_scale: f64,
    tau_scale: f64,
    covariance: FeecMatrix,
    source_eigenvalues: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
enum Exact0FormMassInverseKind {
    Lumped,
    Dense,
}

fn build_exact_spectral_prior_covariance(
    topology: &Complex,
    metric: &MeshLengths,
    base_config: &PlanarHolesFlowConfig,
    requested_mode_count: usize,
    energy_normalized: bool,
    target_expected_m1_energy: f64,
) -> Result<CoexactPriorCovariance, Box<dyn Error>> {
    let mut spectral_config = base_config.clone();
    spectral_config.spectral_exact_mode_count = requested_mode_count;
    spectral_config.spectral_coexact_mode_count = 0;
    spectral_config.spectral_harmonic_mode_count = 0;
    spectral_config.spectral_branch_energy_normalization = energy_normalized;
    spectral_config.spectral_exact_expected_m1_energy = target_expected_m1_energy;
    spectral_config.spectral_boundary_condition = PlanarHolesSpectralBoundaryCondition::Free;
    let basis = build_spectral_hodge_basis(topology, metric, 0, &spectral_config)?;
    let spectral_eigenvalues = basis
        .branch_basis(HodgeBranchKind::Exact)
        .eigenvalues()
        .to_vec();
    let operators = build_spectral_hodge_model_operator(
        basis,
        &spectral_config,
        PlanarHolesModelKind::SpectralHodgeGp,
        &[HodgeBranchKind::Exact],
    )?;
    let (_, transform) = operators
        .branch_transforms
        .get(&HodgeBranchKind::Exact)
        .ok_or_else(|| invalid_data("spectral exact transform is missing"))?;
    let stats = operators
        .spectral_branch_stats
        .get(&HodgeBranchKind::Exact)
        .copied()
        .ok_or_else(|| invalid_data("spectral exact stats are missing"))?;
    Ok(CoexactPriorCovariance {
        spectral_reference: Some(if energy_normalized {
            PlanarHolesPriorEquivalenceSpectralReference::EnergyNormalized
        } else {
            PlanarHolesPriorEquivalenceSpectralReference::Raw
        }),
        actual_mode_count: stats.actual_mode_count,
        expected_m1_energy: stats.expected_m1_energy,
        unnormalized_expected_m1_energy: stats.unnormalized_expected_m1_energy,
        normalization_scale: stats.normalization_scale,
        tau_scale: 1.0,
        covariance: feature_covariance_dense(transform),
        source_eigenvalues: spectral_eigenvalues
            .into_iter()
            .take(stats.actual_mode_count)
            .collect(),
    })
}

fn build_exact_gmrf_prior_covariance(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &PlanarHolesFlowConfig,
    mass_inverse_kind: Exact0FormMassInverseKind,
    ordinary_potential: bool,
    tau_scale: f64,
) -> Result<CoexactPriorCovariance, Box<dyn Error>> {
    if !tau_scale.is_finite() || tau_scale <= 0.0 {
        return Err(invalid_data(format!("invalid tau scale {tau_scale}")).into());
    }
    let laplace = build_laplace_beltrami_0form(topology, metric);
    let source_eigenvalues = generalized_positive_eigenvalues(&laplace.laplacian, &laplace.mass)?;
    let mass_inverse = match mass_inverse_kind {
        Exact0FormMassInverseKind::Lumped => {
            build_matern_mass_inverse_0form(&laplace.mass, Matern0FormMassInverse::RowSumLumped)
        }
        Exact0FormMassInverseKind::Dense => {
            build_exact_dense_mass_inverse_0form(&laplace.mass, 1e-14).map_err(invalid_data)?
        }
    };
    let (precision, transform) = if ordinary_potential {
        if matches!(mass_inverse_kind, Exact0FormMassInverseKind::Dense) {
            let system = build_matern_system_matrix_0form(&laplace, config.exact_kappa);
            let precision = feg_infer::prior::matern::build_lindgren_precision_from_system(
                &system,
                &mass_inverse,
                config.alpha,
                config.tau * config.exact_tau_scale * tau_scale,
            );
            (precision, build_exact_1form_transform(topology))
        } else {
            let prior = build_ordinary_potential_hodge_1form_prior(
                topology,
                metric,
                OrdinaryPotentialHodge1FormPriorConfig {
                    branches: vec![HodgeBranchKind::Exact],
                    exact: SparseAnchorBranchConfig {
                        kappa: config.exact_kappa,
                        tau: config.tau * config.exact_tau_scale * tau_scale,
                        alpha: config.alpha,
                    },
                    zero_form_mass_inverse: Matern0FormMassInverse::RowSumLumped,
                    ..OrdinaryPotentialHodge1FormPriorConfig::default()
                },
            )
            .map_err(invalid_data)?;
            let branch = prior
                .branch(HodgeBranchKind::Exact)
                .ok_or_else(|| invalid_data("ordinary exact branch is missing"))?;
            (branch.precision.clone(), branch.transform.clone())
        }
    } else {
        let system = build_matern_system_matrix_0form(&laplace, config.exact_kappa);
        let precision_full = spectrum_matched_potential_precision(
            &system,
            &mass_inverse,
            config.alpha,
            config.exact_kappa,
            config.tau * config.exact_tau_scale * tau_scale,
        )
        .map_err(invalid_data)?;
        let transform_full = build_exact_1form_transform(topology);
        let (precision, transform) =
            anchor_exact_0form_branch(topology, &precision_full, &transform_full)?;
        (precision, transform)
    };
    let covariance = gmrf_field_covariance_dense(&transform, &precision)?;
    let expected_m1_energy = covariance_expected_mass_energy(&covariance, mass_1form);
    Ok(CoexactPriorCovariance {
        spectral_reference: None,
        actual_mode_count: transform.ncols(),
        expected_m1_energy,
        unnormalized_expected_m1_energy: expected_m1_energy,
        normalization_scale: 1.0,
        tau_scale,
        covariance,
        source_eigenvalues,
    })
}

fn build_coexact_spectral_prior_covariance(
    topology: &Complex,
    metric: &MeshLengths,
    base_config: &PlanarHolesFlowConfig,
    requested_mode_count: usize,
    energy_normalized: bool,
    target_expected_m1_energy: f64,
) -> Result<CoexactPriorCovariance, Box<dyn Error>> {
    let mut spectral_config = base_config.clone();
    spectral_config.spectral_exact_mode_count = 0;
    spectral_config.spectral_coexact_mode_count = requested_mode_count;
    spectral_config.spectral_harmonic_mode_count = 0;
    spectral_config.spectral_branch_energy_normalization = energy_normalized;
    spectral_config.spectral_coexact_expected_m1_energy = target_expected_m1_energy;
    spectral_config.spectral_boundary_condition = PlanarHolesSpectralBoundaryCondition::Free;
    let basis = build_spectral_hodge_basis(topology, metric, 0, &spectral_config)?;
    let spectral_eigenvalues = basis
        .branch_basis(HodgeBranchKind::Coexact)
        .eigenvalues()
        .to_vec();
    let operators = build_spectral_hodge_model_operator(
        basis,
        &spectral_config,
        PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
        &[HodgeBranchKind::Coexact],
    )?;
    let (_, transform) = operators
        .branch_transforms
        .get(&HodgeBranchKind::Coexact)
        .ok_or_else(|| invalid_data("spectral coexact transform is missing"))?;
    let stats = operators
        .spectral_branch_stats
        .get(&HodgeBranchKind::Coexact)
        .copied()
        .ok_or_else(|| invalid_data("spectral coexact stats are missing"))?;
    Ok(CoexactPriorCovariance {
        spectral_reference: Some(if energy_normalized {
            PlanarHolesPriorEquivalenceSpectralReference::EnergyNormalized
        } else {
            PlanarHolesPriorEquivalenceSpectralReference::Raw
        }),
        actual_mode_count: stats.actual_mode_count,
        expected_m1_energy: stats.expected_m1_energy,
        unnormalized_expected_m1_energy: stats.unnormalized_expected_m1_energy,
        normalization_scale: stats.normalization_scale,
        tau_scale: 1.0,
        covariance: feature_covariance_dense(transform),
        source_eigenvalues: spectral_eigenvalues
            .into_iter()
            .take(stats.actual_mode_count)
            .collect(),
    })
}

fn build_coexact_gmrf_prior_covariance(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &PlanarHolesFlowConfig,
    lower_mass_inverse: Matern1FormMassInverse,
    exact_lower_mass_inverse: Option<&FeecCsr>,
    tau_scale: f64,
) -> Result<CoexactPriorCovariance, Box<dyn Error>> {
    if !tau_scale.is_finite() || tau_scale <= 0.0 {
        return Err(invalid_data(format!("invalid tau scale {tau_scale}")).into());
    }
    let coexact_transform = build_exact_mass_coexact_1form_transform(topology, metric, mass_1form)
        .map_err(invalid_data)?;
    let hodge_2form = if let Some(lower_inverse) = exact_lower_mass_inverse {
        build_hodge_laplacian_2form_with_lower_mass_inverse_matrix(topology, metric, lower_inverse)
            .map_err(invalid_data)?
    } else {
        build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
            topology,
            coords,
            metric,
            lower_mass_inverse,
        )
        .map_err(invalid_data)?
    };
    let coexact_system = build_matern_system_matrix_2form(&hodge_2form, config.coexact_kappa);
    let coexact_mass_inverse = build_matern_mass_inverse_2form_with_coords(
        topology,
        coords,
        metric,
        &hodge_2form.mass_u,
        Matern2FormMassInverse::default(),
    )
    .map_err(invalid_data)?;
    let coexact_precision_full = spectrum_matched_potential_precision(
        &coexact_system,
        &coexact_mass_inverse,
        config.alpha,
        config.coexact_kappa,
        config.tau * config.coexact_tau_scale * tau_scale,
    )
    .map_err(invalid_data)?;
    let (coexact_precision, coexact_transform, _) = anchor_hodge_potential_branch(
        topology,
        metric,
        2,
        &coexact_precision_full,
        &coexact_transform,
        HodgeBranchKind::Coexact,
    )
    .map_err(invalid_data)?;
    let covariance = gmrf_field_covariance_dense(&coexact_transform, &coexact_precision)?;
    let expected_m1_energy = covariance_expected_mass_energy(&covariance, mass_1form);
    Ok(CoexactPriorCovariance {
        spectral_reference: None,
        actual_mode_count: coexact_transform.ncols(),
        expected_m1_energy,
        unnormalized_expected_m1_energy: expected_m1_energy,
        normalization_scale: 1.0,
        tau_scale,
        covariance,
        source_eigenvalues: generalized_positive_eigenvalues(
            &hodge_2form.laplacian,
            &hodge_2form.mass_u,
        )?,
    })
}

fn compare_coexact_priors(
    requested_mode_count: usize,
    spectral: &CoexactPriorCovariance,
    gmrf: &CoexactPriorCovariance,
    gmrf_variant: PlanarHolesPriorEquivalenceGmrfVariant,
    mass_1form: &FeecCsr,
    dense_eigen_dimension_cap: usize,
) -> Result<PlanarHolesPriorEquivalenceRow, Box<dyn Error>> {
    let spectral_reference = spectral
        .spectral_reference
        .ok_or_else(|| invalid_data("comparison reference must be spectral"))?;
    let (diag_min, diag_mean, diag_max) =
        diagonal_variance_ratio_summary(&gmrf.covariance, &spectral.covariance);
    let (source_count, source_rel_l2, source_max_rel) = eigenvalue_comparison_summary(
        &spectral.source_eigenvalues,
        &gmrf.source_eigenvalues,
        spectral.actual_mode_count,
    );
    let (cov_count, spec_cov_min, spec_cov_max, gmrf_cov_min, gmrf_cov_max) =
        if spectral.covariance.nrows() <= dense_eigen_dimension_cap {
            let spectral_extremes =
                mass_covariance_eigen_extremes(&spectral.covariance, mass_1form)?;
            let gmrf_extremes = mass_covariance_eigen_extremes(&gmrf.covariance, mass_1form)?;
            (
                spectral_extremes.0.min(gmrf_extremes.0),
                spectral_extremes.1,
                spectral_extremes.2,
                gmrf_extremes.1,
                gmrf_extremes.2,
            )
        } else {
            (0, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        };
    Ok(PlanarHolesPriorEquivalenceRow {
        requested_mode_count,
        actual_mode_count: spectral.actual_mode_count,
        spectral_reference,
        gmrf_variant,
        spectral_expected_m1_energy: spectral.expected_m1_energy,
        spectral_unnormalized_expected_m1_energy: spectral.unnormalized_expected_m1_energy,
        spectral_normalization_scale: spectral.normalization_scale,
        gmrf_expected_m1_energy: gmrf.expected_m1_energy,
        gmrf_tau_scale: gmrf.tau_scale,
        required_tau_scale_to_match_spectral_trace: required_tau_scale(
            gmrf.expected_m1_energy,
            spectral.expected_m1_energy,
        ),
        trace_relative_error: relative_or_nan(
            (gmrf.expected_m1_energy - spectral.expected_m1_energy).abs(),
            spectral.expected_m1_energy.abs(),
        ),
        m1_frobenius_relative_error: m1_covariance_relative_frobenius_error(
            &gmrf.covariance,
            &spectral.covariance,
            mass_1form,
        )?,
        diag_variance_ratio_min: diag_min,
        diag_variance_ratio_mean: diag_mean,
        diag_variance_ratio_max: diag_max,
        source_eigen_count: source_count,
        source_eigen_relative_l2_error: source_rel_l2,
        source_eigen_max_relative_error: source_max_rel,
        covariance_eigen_count: cov_count,
        spectral_covariance_eigen_min: spec_cov_min,
        spectral_covariance_eigen_max: spec_cov_max,
        gmrf_covariance_eigen_min: gmrf_cov_min,
        gmrf_covariance_eigen_max: gmrf_cov_max,
    })
}

fn required_tau_scale(current_expected_energy: f64, target_expected_energy: f64) -> f64 {
    if current_expected_energy.is_finite()
        && target_expected_energy.is_finite()
        && current_expected_energy > 0.0
        && target_expected_energy > 0.0
    {
        (current_expected_energy / target_expected_energy).sqrt()
    } else {
        f64::NAN
    }
}

fn feature_covariance_dense(transform: &FeecCsr) -> FeecMatrix {
    let features = feec_csr_to_dense(transform);
    &features * features.transpose()
}

fn gmrf_field_covariance_dense(
    transform: &FeecCsr,
    precision: &FeecCsr,
) -> Result<FeecMatrix, Box<dyn Error>> {
    if precision.nrows() != precision.ncols() || precision.nrows() != transform.ncols() {
        return Err(invalid_data(format!(
            "precision {}x{} does not match transform {}x{}",
            precision.nrows(),
            precision.ncols(),
            transform.nrows(),
            transform.ncols()
        ))
        .into());
    }
    let factor = feec_csr_to_gmrf(precision)
        .cholesky_sqrt_lower()
        .map_err(|err| invalid_data(format!("failed to factor coexact precision: {err}")))?;
    let mut rhs = GmrfDenseMatrix::zeros(transform.ncols(), transform.nrows());
    for (ambient_row, latent_col, value) in transform.triplet_iter() {
        rhs[(latent_col, ambient_row)] += *value;
    }
    factor
        .solve_dense_in_place(&mut rhs)
        .map_err(|err| invalid_data(format!("failed to solve coexact covariance RHS: {err}")))?;
    let mut covariance = FeecMatrix::zeros(transform.nrows(), transform.nrows());
    for (ambient_row, latent_col, value) in transform.triplet_iter() {
        for out_col in 0..transform.nrows() {
            covariance[(ambient_row, out_col)] += *value * rhs[(latent_col, out_col)];
        }
    }
    Ok(symmetrize_dense(&covariance))
}

fn covariance_expected_mass_energy(covariance: &FeecMatrix, mass: &FeecCsr) -> f64 {
    mass.triplet_iter()
        .map(|(row, col, value)| *value * covariance[(col, row)])
        .sum::<f64>()
        .max(0.0)
}

fn diagonal_variance_ratio_summary(
    candidate: &FeecMatrix,
    reference: &FeecMatrix,
) -> (f64, f64, f64) {
    let mut ratios = Vec::new();
    for index in 0..candidate.nrows().min(reference.nrows()) {
        let denom = reference[(index, index)];
        if denom.is_finite() && denom.abs() > RELATIVE_DENOM_EPS {
            ratios.push(candidate[(index, index)] / denom);
        }
    }
    if ratios.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let min = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let max = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min, mean(ratios.iter().copied()), max)
}

fn eigenvalue_comparison_summary(
    reference: &[f64],
    candidate: &[f64],
    count: usize,
) -> (usize, f64, f64) {
    let count = count.min(reference.len()).min(candidate.len());
    if count == 0 {
        return (0, f64::NAN, f64::NAN);
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut max_relative = 0.0_f64;
    for index in 0..count {
        let diff = candidate[index] - reference[index];
        numerator += diff * diff;
        denominator += reference[index] * reference[index];
        if reference[index].abs() > RELATIVE_DENOM_EPS {
            max_relative = max_relative.max((diff / reference[index]).abs());
        }
    }
    (
        count,
        relative_or_nan(numerator.sqrt(), denominator.sqrt()),
        max_relative,
    )
}

fn m1_covariance_relative_frobenius_error(
    candidate: &FeecMatrix,
    reference: &FeecMatrix,
    mass: &FeecCsr,
) -> Result<f64, Box<dyn Error>> {
    let mass_sqrt = dense_spd_power(&feec_csr_to_dense(mass), 0.5)?;
    let diff = candidate - reference;
    let weighted_diff = &mass_sqrt * diff * mass_sqrt.transpose();
    let weighted_reference = &mass_sqrt * reference * mass_sqrt.transpose();
    Ok(relative_or_nan(
        dense_frobenius_norm(&weighted_diff),
        dense_frobenius_norm(&weighted_reference),
    ))
}

fn mass_covariance_eigen_extremes(
    covariance: &FeecMatrix,
    mass: &FeecCsr,
) -> Result<(usize, f64, f64), Box<dyn Error>> {
    let mass_sqrt = dense_spd_power(&feec_csr_to_dense(mass), 0.5)?;
    let weighted = symmetrize_dense(&(&mass_sqrt * covariance * mass_sqrt.transpose()));
    let eigen = weighted.symmetric_eigen();
    let mut values = eigen
        .eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 1e-12)
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues should compare"));
    if values.is_empty() {
        Ok((0, f64::NAN, f64::NAN))
    } else {
        Ok((values.len(), values[0], *values.last().unwrap()))
    }
}

fn generalized_positive_eigenvalues(
    stiffness: &FeecCsr,
    mass: &FeecCsr,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let mass_inv_sqrt = dense_spd_power(&feec_csr_to_dense(mass), -0.5)?;
    let stiffness = symmetrize_dense(&feec_csr_to_dense(stiffness));
    let operator = symmetrize_dense(&(&mass_inv_sqrt * stiffness * mass_inv_sqrt.transpose()));
    let eigen = operator.symmetric_eigen();
    let mut values = eigen
        .eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 1e-10)
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues should compare"));
    Ok(values)
}

fn dense_spd_power(matrix: &FeecMatrix, power: f64) -> Result<FeecMatrix, Box<dyn Error>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(invalid_data("dense SPD power requires a square matrix").into());
    }
    let symmetric = symmetrize_dense(matrix);
    let eigen = symmetric.symmetric_eigen();
    let max_abs = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let tolerance = (1e-12 * max_abs).max(1e-14);
    let mut scaled_vectors = FeecMatrix::zeros(matrix.nrows(), matrix.ncols());
    for col in 0..matrix.ncols() {
        let eigenvalue = eigen.eigenvalues[col];
        if eigenvalue <= tolerance {
            return Err(invalid_data(format!(
                "dense SPD power requires positive eigenvalues, got {eigenvalue:.6e}"
            ))
            .into());
        }
        let scale = eigenvalue.powf(power);
        for row in 0..matrix.nrows() {
            scaled_vectors[(row, col)] = eigen.eigenvectors[(row, col)] * scale;
        }
    }
    Ok(&scaled_vectors * eigen.eigenvectors.transpose())
}

fn dense_frobenius_norm(matrix: &FeecMatrix) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub fn run_planar_holes_sensor_design_sweep(
    config: &PlanarHolesSensorDesignSweepConfig,
) -> Result<Vec<PlanarHolesSensorDesignSweepRow>, Box<dyn Error>> {
    validate_config(&config.base)?;
    validate_sensor_sweep_config(config)?;
    ensure_planar_holes_mesh(&config.base)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    validate_topology(&summarize_topology(&topology))?;
    let holes = default_holes();

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw = compute_harmonic_basis_1form(&topology, &metric, holes.len(), None)
        .map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;
    let (cycle_observation_matrix, cycles) =
        build_hole_cycle_observation_matrix(&topology, &coords, &holes, 0.0, 0.0, "train")?;
    let harmonic_basis = canonicalize_harmonic_basis_by_cycles(
        &harmonic_basis_orthonormal,
        &cycle_observation_matrix,
    )?;
    let (heldout_cycle_observation_matrix, heldout_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        config.base.heldout_loop_radius_offset,
        config.base.heldout_loop_angle_offset,
        "heldout",
    )?;
    let hodge_prior =
        build_hodge_prior(&topology, &coords, &metric, &harmonic_basis, &config.base)?;
    let truth = build_truth(
        &topology,
        &coords,
        &metric,
        &hodge.mass_u,
        &hodge_prior,
        &config.base,
    )
    .map_err(|err| invalid_data(format!("sensor sweep truth construction failed: {err}")))?;
    let projection = build_hodge_projection_operator_1form_with_basis(
        &topology,
        &metric,
        &hodge.mass_u,
        harmonic_basis.clone(),
        1e-10,
    )
    .map_err(invalid_data)?;
    let harmonic_period_observation_matrix = &cycle_observation_matrix * &projection.harmonic;
    let heldout_harmonic_period_observation_matrix =
        &heldout_cycle_observation_matrix * &projection.harmonic;
    let model_operators = build_model_operators(
        &topology,
        &coords,
        &metric,
        &hodge,
        &hodge_prior,
        &projection,
        &harmonic_basis,
        &config.base,
    )
    .map_err(|err| invalid_data(format!("sensor sweep model construction failed: {err}")))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.base.rng_seed + 909);
    let mut cycle_edge_set = observation_edge_set(&cycle_observation_matrix);
    cycle_edge_set.extend(observation_edge_set(&heldout_cycle_observation_matrix));
    let local_edges = select_observation_edges(
        &topology,
        &coords,
        &cycle_edge_set,
        &BTreeSet::new(),
        config.total_observation_budget,
        &mut rng,
    );
    let local_used = local_edges.iter().copied().collect::<BTreeSet<_>>();
    let heldout_edges = select_observation_edges(
        &topology,
        &coords,
        &cycle_edge_set,
        &local_used,
        config.heldout_local_count,
        &mut rng,
    );

    let small_loop_pool = build_contractible_loop_rows(
        &topology,
        &coords,
        &holes,
        &[0.055],
        config.total_observation_budget,
        config.base.rng_seed + 910,
        "small_loop",
    );
    let multiscale_loop_pool = build_contractible_loop_rows(
        &topology,
        &coords,
        &holes,
        &[0.055, 0.095, 0.135],
        config.total_observation_budget,
        config.base.rng_seed + 911,
        "multiscale_loop",
    );
    let heldout_loop_pool = build_contractible_loop_rows(
        &topology,
        &coords,
        &holes,
        &[0.065, 0.105, 0.145],
        config.heldout_interior_loop_count,
        config.base.rng_seed + 912,
        "heldout_interior_loop",
    );
    let long_path_pool = build_long_path_rows(
        &topology,
        &coords,
        &holes,
        config.total_observation_budget,
        config.base.rng_seed + 913,
        "long_path",
    );
    let heldout_long_path_pool = build_long_path_rows(
        &topology,
        &coords,
        &holes,
        config.heldout_long_path_count,
        config.base.rng_seed + 914,
        "heldout_long_path",
    );
    let train_harmonic_periods = loop_observations(
        PlanarHolesObservationKind::TrainHarmonicPeriod,
        "train_harmonic_period",
        &harmonic_period_observation_matrix,
        &cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        config.base.sample_observation_noise,
        &mut rng,
    );
    let heldout_harmonic_periods = loop_observations(
        PlanarHolesObservationKind::HeldoutHarmonicPeriod,
        "heldout_harmonic_period",
        &heldout_harmonic_period_observation_matrix,
        &heldout_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        config.base.sample_observation_noise,
        &mut rng,
    );
    let heldout_observations = {
        let mut observations = edge_observations(
            PlanarHolesObservationKind::HeldoutLocal,
            "heldout_local",
            &heldout_edges,
            &truth.mixed,
            config.base.local_noise_variance,
            config.base.sample_observation_noise,
            &mut rng,
        );
        observations.extend(row_observations(
            PlanarHolesObservationKind::HeldoutInteriorLoop,
            &heldout_loop_pool,
            &truth.mixed,
            config.interior_loop_noise_variance,
            config.base.sample_observation_noise,
            &mut rng,
        ));
        observations.extend(row_observations(
            PlanarHolesObservationKind::HeldoutLongPath,
            &heldout_long_path_pool,
            &truth.mixed,
            config.long_path_noise_variance,
            config.base.sample_observation_noise,
            &mut rng,
        ));
        observations.extend(heldout_harmonic_periods);
        observations
    };

    let mut rows = Vec::new();
    for design in &config.designs {
        let counts = sensor_design_counts(*design, config.total_observation_budget, cycles.len());
        let loop_pool = match design {
            PlanarHolesSensorDesignKind::EdgesSmallInteriorLoops => &small_loop_pool,
            _ => &multiscale_loop_pool,
        };
        let edge_count = counts.edge_count.min(local_edges.len());
        let interior_loop_count = counts.interior_loop_count.min(loop_pool.len());
        let long_path_count = counts.long_path_count.min(long_path_pool.len());
        let harmonic_period_count = counts
            .harmonic_period_count
            .min(train_harmonic_periods.len());
        let mut training = edge_observations(
            PlanarHolesObservationKind::Local,
            "local",
            &local_edges[..edge_count],
            &truth.mixed,
            config.base.local_noise_variance,
            config.base.sample_observation_noise,
            &mut rng,
        );
        training.extend(row_observations(
            PlanarHolesObservationKind::TrainInteriorLoop,
            &loop_pool[..interior_loop_count],
            &truth.mixed,
            config.interior_loop_noise_variance,
            config.base.sample_observation_noise,
            &mut rng,
        ));
        training.extend(row_observations(
            PlanarHolesObservationKind::TrainLongPath,
            &long_path_pool[..long_path_count],
            &truth.mixed,
            config.long_path_noise_variance,
            config.base.sample_observation_noise,
            &mut rng,
        ));
        training.extend(
            train_harmonic_periods
                .iter()
                .take(harmonic_period_count)
                .cloned(),
        );
        let (training_matrix, training_values) =
            scaled_observation_system(&training, topology.edges().len());
        let heldout_matrix =
            observation_matrix(&heldout_observations, topology.edges().len(), false);

        for operators in &model_operators {
            let conditioned = condition_model(operators, &training_matrix, &training_values)
                .map_err(|err| {
                    invalid_data(format!(
                        "sensor sweep conditioning failed for design {} model {}: {err}",
                        design.as_str(),
                        operators.model.as_str()
                    ))
                })?;
            let predictions = compute_heldout_predictions(
                PlanarHolesObservationScenario::LocalOnly,
                conditioned.model,
                &heldout_observations,
                &heldout_matrix,
                &conditioned,
            )?;
            let field_coverage = if config.base.compute_field_coverage {
                compute_field_coverage_diagnostics(
                    PlanarHolesObservationScenario::LocalOnly,
                    conditioned.model,
                    &hodge.mass_u,
                    &truth.mixed,
                    &heldout_observations,
                    &conditioned,
                )?
            } else {
                empty_field_coverage_diagnostics(
                    PlanarHolesObservationScenario::LocalOnly,
                    conditioned.model,
                    &hodge.mass_u,
                    &truth.mixed,
                    &heldout_observations,
                    &conditioned.posterior_mean,
                )
            };
            rows.push(sensor_design_sweep_row(
                *design,
                counts,
                training.len(),
                &topology,
                &metric,
                &hodge.mass_u,
                &truth,
                &conditioned,
                &predictions,
                &field_coverage.summaries,
            )?);
        }
    }

    Ok(rows)
}

pub fn write_planar_holes_sensor_design_sweep(
    rows: &[PlanarHolesSensorDesignSweepRow],
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "design,model,model_coexact_tau_scale,observation_count,edge_observation_count,interior_loop_observation_count,long_path_observation_count,harmonic_period_observation_count,l2_error,heldout_local_nlpd,heldout_interior_loop_nlpd,heldout_long_path_nlpd,heldout_harmonic_period_nlpd,heldout_interior_loop_relative_error,heldout_long_path_relative_error,heldout_harmonic_period_relative_error,codifferential_leakage,coexact_relative_error,coexact_mass_correlation,harmonic_relative_error,harmonic_mass_correlation,all_edge_coverage_95,heldout_edge_coverage_95"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{:.12},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.design.as_str(),
            row.model.as_str(),
            row.model_coexact_tau_scale,
            row.observation_count,
            row.edge_observation_count,
            row.interior_loop_observation_count,
            row.long_path_observation_count,
            row.harmonic_period_observation_count,
            row.l2_error,
            row.heldout_local_nlpd,
            row.heldout_interior_loop_nlpd,
            row.heldout_long_path_nlpd,
            row.heldout_harmonic_period_nlpd,
            row.heldout_interior_loop_relative_error,
            row.heldout_long_path_relative_error,
            row.heldout_harmonic_period_relative_error,
            row.codifferential_leakage,
            row.coexact_relative_error,
            row.coexact_mass_correlation,
            row.harmonic_relative_error,
            row.harmonic_mass_correlation,
            row.all_edge_coverage_95,
            row.heldout_edge_coverage_95
        )?;
    }
    Ok(())
}

pub fn run_planar_holes_topology_vs_naive_gp(
    config: &PlanarHolesTopologyVsNaiveGpConfig,
) -> Result<PlanarHolesTopologyVsNaiveGpResult, Box<dyn Error>> {
    validate_config(&config.base)?;
    validate_topology_vs_naive_config(config)?;
    ensure_planar_holes_mesh(&config.base)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = summarize_topology(&topology);
    validate_topology(&topology_summary)?;
    let holes = default_holes();

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw = compute_harmonic_basis_1form(&topology, &metric, holes.len(), None)
        .map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;
    let (train_cycle_matrix, train_cycles) =
        build_hole_cycle_observation_matrix(&topology, &coords, &holes, 0.0, 0.0, "train")?;
    let harmonic_basis =
        canonicalize_harmonic_basis_by_cycles(&harmonic_basis_orthonormal, &train_cycle_matrix)?;
    let (validation_cycle_matrix, validation_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        0.045,
        std::f64::consts::PI / 11.0,
        "validation",
    )?;
    let (heldout_cycle_matrix, heldout_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        config.base.heldout_loop_radius_offset.max(0.075),
        config.base.heldout_loop_angle_offset,
        "heldout",
    )?;
    let extra_validation_cycle_families = build_hole_cycle_families(
        &topology,
        &coords,
        &holes,
        &[
            (0.030, std::f64::consts::PI / 7.0, "validation_inner"),
            (0.060, std::f64::consts::PI / 5.0, "validation_outer"),
        ],
    )?;
    let extra_heldout_cycle_families = build_hole_cycle_families(
        &topology,
        &coords,
        &holes,
        &[
            (0.025, std::f64::consts::PI / 8.0, "heldout_inner"),
            (0.050, std::f64::consts::PI / 10.0, "heldout_middle"),
            (0.080, std::f64::consts::PI / 13.0, "heldout_outer"),
        ],
    )?;
    let train_cycle_harmonic_pairing_rank = (&train_cycle_matrix * &harmonic_basis).rank(1e-6);
    let validation_cycle_harmonic_pairing_rank =
        (&validation_cycle_matrix * &harmonic_basis).rank(1e-6);
    let heldout_cycle_harmonic_pairing_rank = (&heldout_cycle_matrix * &harmonic_basis).rank(1e-6);
    for (label, rank) in [
        ("train", train_cycle_harmonic_pairing_rank),
        ("validation", validation_cycle_harmonic_pairing_rank),
        ("heldout", heldout_cycle_harmonic_pairing_rank),
    ] {
        if rank != holes.len() {
            return Err(invalid_data(format!(
                "{label} cycle-harmonic pairing rank {rank} does not match hole count {}",
                holes.len()
            ))
            .into());
        }
    }

    let hodge_prior =
        build_hodge_prior(&topology, &coords, &metric, &harmonic_basis, &config.base)?;
    let truth = build_truth(
        &topology,
        &coords,
        &metric,
        &hodge.mass_u,
        &hodge_prior,
        &config.base,
    )
    .map_err(|err| {
        invalid_data(format!(
            "topology-vs-naive truth construction failed: {err}"
        ))
    })?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.base.rng_seed + 202);
    let mut cycle_edge_set = observation_edge_set(&train_cycle_matrix);
    cycle_edge_set.extend(observation_edge_set(&validation_cycle_matrix));
    cycle_edge_set.extend(observation_edge_set(&heldout_cycle_matrix));
    for (matrix, _) in &extra_validation_cycle_families {
        cycle_edge_set.extend(observation_edge_set(matrix));
    }
    for (matrix, _) in &extra_heldout_cycle_families {
        cycle_edge_set.extend(observation_edge_set(matrix));
    }
    let counts = sensor_design_counts(
        PlanarHolesSensorDesignKind::Hybrid,
        config.total_observation_budget,
        holes.len(),
    );
    let train_edges = select_observation_edges(
        &topology,
        &coords,
        &cycle_edge_set,
        &BTreeSet::new(),
        counts.edge_count,
        &mut rng,
    );
    let mut excluded = train_edges.iter().copied().collect::<BTreeSet<_>>();
    let validation_edges = select_observation_edges(
        &topology,
        &coords,
        &cycle_edge_set,
        &excluded,
        config.validation_local_count,
        &mut rng,
    );
    excluded.extend(validation_edges.iter().copied());
    let heldout_edges = select_observation_edges(
        &topology,
        &coords,
        &cycle_edge_set,
        &excluded,
        config.heldout_local_count,
        &mut rng,
    );

    let train_loop_pool = build_contractible_loop_rows(
        &topology,
        &coords,
        &holes,
        &[0.055, 0.095, 0.135],
        counts.interior_loop_count,
        config.base.rng_seed + 203,
        "train_interior_loop",
    );
    let validation_loop_pool = build_contractible_loop_rows(
        &topology,
        &coords,
        &holes,
        &[0.075, 0.115],
        config.validation_interior_loop_count,
        config.base.rng_seed + 204,
        "validation_interior_loop",
    );
    let heldout_loop_pool = build_contractible_loop_rows(
        &topology,
        &coords,
        &holes,
        &[0.065, 0.105, 0.145],
        config.heldout_interior_loop_count,
        config.base.rng_seed + 205,
        "heldout_interior_loop",
    );
    let train_long_path_pool = build_long_path_rows(
        &topology,
        &coords,
        &holes,
        counts.long_path_count,
        config.base.rng_seed + 206,
        "train_long_path",
    );
    let validation_long_path_pool = build_long_path_rows(
        &topology,
        &coords,
        &holes,
        config.validation_long_path_count,
        config.base.rng_seed + 207,
        "validation_long_path",
    );
    let heldout_long_path_pool = build_long_path_rows(
        &topology,
        &coords,
        &holes,
        config.heldout_long_path_count,
        config.base.rng_seed + 208,
        "heldout_long_path",
    );

    let base_train = {
        let mut observations = edge_observations(
            PlanarHolesObservationKind::Local,
            "train_local",
            &train_edges,
            &truth.mixed,
            config.base.local_noise_variance,
            false,
            &mut rng,
        );
        observations.extend(row_observations(
            PlanarHolesObservationKind::TrainInteriorLoop,
            &train_loop_pool,
            &truth.mixed,
            config.interior_loop_noise_variance,
            false,
            &mut rng,
        ));
        observations.extend(row_observations(
            PlanarHolesObservationKind::TrainLongPath,
            &train_long_path_pool,
            &truth.mixed,
            config.long_path_noise_variance,
            false,
            &mut rng,
        ));
        observations.extend(loop_observations(
            PlanarHolesObservationKind::TrainLoop,
            "train_loop",
            &train_cycle_matrix,
            &train_cycles,
            &truth.mixed,
            config.base.loop_noise_variance,
            false,
            &mut rng,
        ));
        observations
    };
    let base_validation = {
        let mut observations = edge_observations(
            PlanarHolesObservationKind::HeldoutLocal,
            "validation_local",
            &validation_edges,
            &truth.mixed,
            config.base.local_noise_variance,
            false,
            &mut rng,
        );
        observations.extend(row_observations(
            PlanarHolesObservationKind::HeldoutInteriorLoop,
            &validation_loop_pool,
            &truth.mixed,
            config.interior_loop_noise_variance,
            false,
            &mut rng,
        ));
        observations.extend(row_observations(
            PlanarHolesObservationKind::HeldoutLongPath,
            &validation_long_path_pool,
            &truth.mixed,
            config.long_path_noise_variance,
            false,
            &mut rng,
        ));
        observations.extend(loop_observations(
            PlanarHolesObservationKind::HeldoutLoop,
            "validation_loop",
            &validation_cycle_matrix,
            &validation_cycles,
            &truth.mixed,
            config.base.loop_noise_variance,
            false,
            &mut rng,
        ));
        for (family_index, (matrix, cycles)) in extra_validation_cycle_families.iter().enumerate() {
            observations.extend(loop_observations(
                PlanarHolesObservationKind::HeldoutLoop,
                &format!("validation_loop_family_{family_index}"),
                matrix,
                cycles,
                &truth.mixed,
                config.base.loop_noise_variance,
                false,
                &mut rng,
            ));
        }
        observations
    };
    let base_heldout = {
        let mut observations = edge_observations(
            PlanarHolesObservationKind::HeldoutLocal,
            "heldout_local",
            &heldout_edges,
            &truth.mixed,
            config.base.local_noise_variance,
            false,
            &mut rng,
        );
        observations.extend(row_observations(
            PlanarHolesObservationKind::HeldoutInteriorLoop,
            &heldout_loop_pool,
            &truth.mixed,
            config.interior_loop_noise_variance,
            false,
            &mut rng,
        ));
        observations.extend(row_observations(
            PlanarHolesObservationKind::HeldoutLongPath,
            &heldout_long_path_pool,
            &truth.mixed,
            config.long_path_noise_variance,
            false,
            &mut rng,
        ));
        observations.extend(loop_observations(
            PlanarHolesObservationKind::HeldoutLoop,
            "heldout_loop",
            &heldout_cycle_matrix,
            &heldout_cycles,
            &truth.mixed,
            config.base.loop_noise_variance,
            false,
            &mut rng,
        ));
        for (family_index, (matrix, cycles)) in extra_heldout_cycle_families.iter().enumerate() {
            observations.extend(loop_observations(
                PlanarHolesObservationKind::HeldoutLoop,
                &format!("heldout_loop_family_{family_index}"),
                matrix,
                cycles,
                &truth.mixed,
                config.base.loop_noise_variance,
                false,
                &mut rng,
            ));
        }
        observations
    };

    let hodge_selection = select_topology_vs_naive_candidate(
        PlanarHolesModelKind::NondecomposedFeec,
        &config.hodge_kappas,
        &config.hodge_taus,
        &config.observation_variance_scales,
        &topology,
        &coords,
        &metric,
        &hodge,
        config.base.alpha,
        &base_train,
        &base_validation,
    )?;
    let naive_selection = select_topology_vs_naive_candidate(
        PlanarHolesModelKind::NaiveEuclideanVectorMatern,
        &config.naive_kappas,
        &config.naive_taus,
        &config.observation_variance_scales,
        &topology,
        &coords,
        &metric,
        &hodge,
        config.base.alpha,
        &base_train,
        &base_validation,
    )?;

    let mut rows = Vec::new();
    let mut selections = vec![hodge_selection, naive_selection];
    let mut tuning_rows = Vec::new();
    for selection in &mut selections {
        tuning_rows.append(&mut selection.tuning_rows);
    }
    let mut heldout_predictions = Vec::new();
    let mut field_coverage_summaries = Vec::new();
    let mut calibration_rows = Vec::new();
    for selection in selections {
        let mut final_train = base_train.clone();
        final_train.extend(base_validation.clone());
        let final_train =
            scaled_observation_variances(&final_train, selection.observation_variance_scale);
        let final_heldout =
            scaled_observation_variances(&base_heldout, selection.observation_variance_scale);
        let operator = build_topology_vs_naive_model_operator(
            selection.model,
            &topology,
            &coords,
            &metric,
            &hodge,
            selection.kappa,
            selection.tau,
            config.base.alpha,
        )?;
        let (training_matrix, training_values) =
            scaled_observation_system(&final_train, topology.edges().len());
        let heldout_matrix = observation_matrix(&final_heldout, topology.edges().len(), false);
        let conditioned = condition_model(&operator, &training_matrix, &training_values)?;
        let predictions = compute_heldout_predictions(
            PlanarHolesObservationScenario::LocalOnly,
            selection.model,
            &final_heldout,
            &heldout_matrix,
            &conditioned,
        )?;
        let field_coverage = compute_field_coverage_diagnostics(
            PlanarHolesObservationScenario::LocalOnly,
            selection.model,
            &hodge.mass_u,
            &truth.mixed,
            &final_heldout,
            &conditioned,
        )?;
        rows.push(topology_vs_naive_metric_row(
            &topology,
            &metric,
            &hodge.mass_u,
            &truth,
            &conditioned,
            &predictions,
            &field_coverage.summaries,
            &operator,
            selection.kappa,
            selection.tau,
            selection.observation_variance_scale,
            selection.validation_nlpd,
            final_train.len(),
            base_validation.len(),
            final_heldout.len(),
        )?);
        calibration_rows.extend(calibration_rows_for_predictions(
            selection.model,
            selection.variance_multiplier,
            &predictions,
        ));
        heldout_predictions.extend(predictions);
        field_coverage_summaries.extend(field_coverage.summaries);
    }

    Ok(PlanarHolesTopologyVsNaiveGpResult {
        topology_summary,
        train_cycle_harmonic_pairing_rank,
        validation_cycle_harmonic_pairing_rank,
        heldout_cycle_harmonic_pairing_rank,
        rows,
        tuning_rows,
        calibration_rows,
        heldout_predictions,
        field_coverage_summaries,
    })
}

pub fn run_planar_holes_barrier_topology_vs_naive_gp(
    config: &PlanarHolesTopologyVsNaiveGpConfig,
) -> Result<PlanarHolesBarrierTopologyVsNaiveGpResult, Box<dyn Error>> {
    validate_config(&config.base)?;
    validate_topology_vs_naive_config(config)?;
    let holes = holes_for_domain(PlanarHolesDomainKind::Barrier);
    ensure_planar_holes_mesh_with_holes(&config.base, &holes)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = summarize_topology(&topology);
    validate_topology(&topology_summary)?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw = compute_harmonic_basis_1form(&topology, &metric, holes.len(), None)
        .map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;
    let (train_cycle_matrix, train_cycles) =
        build_hole_cycle_observation_matrix(&topology, &coords, &holes, 0.0, 0.0, "train")?;
    let harmonic_basis =
        canonicalize_harmonic_basis_by_cycles(&harmonic_basis_orthonormal, &train_cycle_matrix)?;
    let (validation_cycle_matrix, validation_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        0.008,
        std::f64::consts::PI / 13.0,
        "validation",
    )?;
    let (heldout_cycle_matrix, heldout_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        0.015,
        std::f64::consts::PI / 17.0,
        "heldout",
    )?;
    let train_rank = (&train_cycle_matrix * &harmonic_basis).rank(1e-6);
    let validation_rank = (&validation_cycle_matrix * &harmonic_basis).rank(1e-6);
    let heldout_rank = (&heldout_cycle_matrix * &harmonic_basis).rank(1e-6);
    for (label, rank) in [
        ("train", train_rank),
        ("validation", validation_rank),
        ("heldout", heldout_rank),
    ] {
        if rank != holes.len() {
            return Err(invalid_data(format!(
                "{label} cycle-harmonic pairing rank {rank} does not match hole count {}",
                holes.len()
            ))
            .into());
        }
    }

    let hodge_prior =
        build_hodge_prior(&topology, &coords, &metric, &harmonic_basis, &config.base)?;
    let truth = build_truth(
        &topology,
        &coords,
        &metric,
        &hodge.mass_u,
        &hodge_prior,
        &config.base,
    )
    .map_err(|err| invalid_data(format!("barrier truth construction failed: {err}")))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.base.rng_seed + 502);
    let mut cycle_edge_set = observation_edge_set(&train_cycle_matrix);
    cycle_edge_set.extend(observation_edge_set(&validation_cycle_matrix));
    cycle_edge_set.extend(observation_edge_set(&heldout_cycle_matrix));
    let train_edges = select_observation_edges_by_barycenter(
        &topology,
        &coords,
        &cycle_edge_set,
        &BTreeSet::new(),
        config.total_observation_budget,
        &mut rng,
        |point| point[0] >= 0.28 && point[0] <= 0.43,
    );
    let mut excluded = train_edges.iter().copied().collect::<BTreeSet<_>>();
    let validation_edges = select_observation_edges_by_barycenter(
        &topology,
        &coords,
        &cycle_edge_set,
        &excluded,
        config.validation_local_count,
        &mut rng,
        |point| point[0] >= 0.18 && point[0] <= 0.43,
    );
    excluded.extend(validation_edges.iter().copied());
    let heldout_edges = select_observation_edges_by_barycenter(
        &topology,
        &coords,
        &cycle_edge_set,
        &excluded,
        config.heldout_local_count,
        &mut rng,
        |point| point[0] >= 0.57 && point[0] <= 0.72,
    );
    let validation_paths = build_same_side_long_path_rows(
        &topology,
        &coords,
        &holes,
        (0.12, 0.43),
        config.validation_long_path_count,
        config.base.rng_seed + 503,
        "validation_left_path",
    );
    let heldout_paths = build_cross_barrier_path_rows(
        &topology,
        &coords,
        &holes,
        config.heldout_long_path_count,
        config.base.rng_seed + 504,
        "heldout_cross_barrier_path",
    );

    let mut base_train = edge_observations(
        PlanarHolesObservationKind::Local,
        "train_left_local",
        &train_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_train.extend(loop_observations(
        PlanarHolesObservationKind::TrainLoop,
        "train_loop",
        &train_cycle_matrix,
        &train_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    let mut base_validation = edge_observations(
        PlanarHolesObservationKind::HeldoutLocal,
        "validation_left_local",
        &validation_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_validation.extend(row_observations(
        PlanarHolesObservationKind::HeldoutLongPath,
        &validation_paths,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    base_validation.extend(loop_observations(
        PlanarHolesObservationKind::HeldoutLoop,
        "validation_loop",
        &validation_cycle_matrix,
        &validation_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    let mut base_heldout = edge_observations(
        PlanarHolesObservationKind::HeldoutLocal,
        "heldout_right_local",
        &heldout_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_heldout.extend(row_observations(
        PlanarHolesObservationKind::HeldoutLongPath,
        &heldout_paths,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    base_heldout.extend(loop_observations(
        PlanarHolesObservationKind::HeldoutLoop,
        "heldout_loop",
        &heldout_cycle_matrix,
        &heldout_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));

    let base = condition_topology_vs_naive_experiment(
        config,
        topology_summary,
        train_rank,
        validation_rank,
        heldout_rank,
        &topology,
        &coords,
        &metric,
        &hodge,
        &harmonic_basis,
        &truth,
        &base_train,
        &base_validation,
        &base_heldout,
    )?;
    let rows = base
        .rows
        .iter()
        .map(|row| {
            let predictions = base
                .heldout_predictions
                .iter()
                .filter(|prediction| prediction.model == row.model)
                .cloned()
                .collect::<Vec<_>>();
            let local =
                prediction_kind_stats(&predictions, PlanarHolesObservationKind::HeldoutLocal);
            let path =
                prediction_kind_stats(&predictions, PlanarHolesObservationKind::HeldoutLongPath);
            let loop_stats =
                prediction_kind_stats(&predictions, PlanarHolesObservationKind::HeldoutLoop);
            let variance_multiplier =
                calibration_multiplier_for_model(&base.calibration_rows, row.model);
            let calibrated_local = prediction_kind_stats_with_variance_multiplier(
                &predictions,
                PlanarHolesObservationKind::HeldoutLocal,
                variance_multiplier,
            );
            let calibrated_path = prediction_kind_stats_with_variance_multiplier(
                &predictions,
                PlanarHolesObservationKind::HeldoutLongPath,
                variance_multiplier,
            );
            let calibrated_loop = prediction_kind_stats_with_variance_multiplier(
                &predictions,
                PlanarHolesObservationKind::HeldoutLoop,
                variance_multiplier,
            );
            PlanarHolesBarrierSummaryRow {
                model: row.model,
                train_left_edge_count: train_edges.len(),
                validation_left_edge_count: validation_edges.len(),
                heldout_right_edge_count: heldout_edges.len(),
                heldout_cross_barrier_path_count: path.count,
                cross_barrier_local_nlpd: local.mean_nlpd,
                cross_barrier_local_relative_error: local.relative_error,
                barrier_long_path_nlpd: path.mean_nlpd,
                barrier_long_path_relative_error: path.relative_error,
                hole_loop_nlpd: loop_stats.mean_nlpd,
                hole_loop_relative_error: loop_stats.relative_error,
                cross_barrier_local_coverage_95: local.coverage_95,
                barrier_long_path_coverage_95: path.coverage_95,
                hole_loop_coverage_95: loop_stats.coverage_95,
                calibrated_cross_barrier_local_nlpd: calibrated_local.mean_nlpd,
                calibrated_barrier_long_path_nlpd: calibrated_path.mean_nlpd,
                calibrated_hole_loop_nlpd: calibrated_loop.mean_nlpd,
                calibrated_cross_barrier_local_coverage_95: calibrated_local.coverage_95,
                calibrated_barrier_long_path_coverage_95: calibrated_path.coverage_95,
                calibrated_hole_loop_coverage_95: calibrated_loop.coverage_95,
            }
        })
        .collect();
    Ok(PlanarHolesBarrierTopologyVsNaiveGpResult {
        base,
        domain: PlanarHolesDomainKind::Barrier,
        rows,
    })
}

pub fn run_planar_holes_path_homology_vs_naive_gp(
    config: &PlanarHolesTopologyVsNaiveGpConfig,
) -> Result<PlanarHolesPathHomologyTopologyVsNaiveGpResult, Box<dyn Error>> {
    validate_config(&config.base)?;
    validate_topology_vs_naive_config(config)?;
    let holes = holes_for_domain(PlanarHolesDomainKind::Default);
    ensure_planar_holes_mesh_with_holes(&config.base, &holes)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = summarize_topology(&topology);
    validate_topology(&topology_summary)?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw = compute_harmonic_basis_1form(&topology, &metric, holes.len(), None)
        .map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;
    let (train_cycle_matrix, train_cycles) =
        build_hole_cycle_observation_matrix(&topology, &coords, &holes, 0.0, 0.0, "train")?;
    let harmonic_basis =
        canonicalize_harmonic_basis_by_cycles(&harmonic_basis_orthonormal, &train_cycle_matrix)?;
    let (validation_cycle_matrix, validation_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        0.030,
        std::f64::consts::PI / 11.0,
        "validation",
    )?;
    let (heldout_cycle_matrix, heldout_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        0.055,
        std::f64::consts::PI / 17.0,
        "heldout",
    )?;
    let train_rank = (&train_cycle_matrix * &harmonic_basis).rank(1e-6);
    let validation_rank = (&validation_cycle_matrix * &harmonic_basis).rank(1e-6);
    let heldout_rank = (&heldout_cycle_matrix * &harmonic_basis).rank(1e-6);
    for (label, rank) in [
        ("train", train_rank),
        ("validation", validation_rank),
        ("heldout", heldout_rank),
    ] {
        if rank != holes.len() {
            return Err(invalid_data(format!(
                "{label} cycle-harmonic pairing rank {rank} does not match hole count {}",
                holes.len()
            ))
            .into());
        }
    }

    let hodge_prior =
        build_hodge_prior(&topology, &coords, &metric, &harmonic_basis, &config.base)?;
    let truth = build_truth(
        &topology,
        &coords,
        &metric,
        &hodge.mass_u,
        &hodge_prior,
        &config.base,
    )
    .map_err(|err| invalid_data(format!("path-homology truth construction failed: {err}")))?;

    let train_pairs =
        build_path_homology_pairs(&topology, &coords, &holes, 0.025, 0.0, "train_path_family")?;
    let validation_pairs = build_path_homology_pairs(
        &topology,
        &coords,
        &holes,
        0.040,
        std::f64::consts::PI / 10.0,
        "validation_path_family",
    )?;
    let mut heldout_pairs = build_path_homology_pairs(
        &topology,
        &coords,
        &holes,
        0.035,
        std::f64::consts::PI / 5.0,
        "heldout_path_family_a",
    )?;
    heldout_pairs.extend(build_path_homology_pairs(
        &topology,
        &coords,
        &holes,
        0.060,
        -std::f64::consts::PI / 8.0,
        "heldout_path_family_b",
    )?);
    let contrast_rows = heldout_pairs
        .iter()
        .map(|pair| pair.contrast.clone())
        .collect::<Vec<_>>();
    let contrast_matrix = sensor_rows_to_csr(&contrast_rows, topology.edges().len());
    let path_contrast_rank = (&contrast_matrix * &harmonic_basis).rank(1e-6);
    if path_contrast_rank != holes.len() {
        return Err(invalid_data(format!(
            "path contrast-harmonic pairing rank {path_contrast_rank} does not match hole count {}",
            holes.len()
        ))
        .into());
    }

    let mut excluded_edges = observation_edge_set(&train_cycle_matrix);
    excluded_edges.extend(observation_edge_set(&validation_cycle_matrix));
    excluded_edges.extend(observation_edge_set(&heldout_cycle_matrix));
    excluded_edges.extend(edge_set_from_sensor_rows(
        &train_pairs
            .iter()
            .flat_map(|pair| {
                [
                    pair.upper.clone(),
                    pair.lower.clone(),
                    pair.contrast.clone(),
                ]
            })
            .collect::<Vec<_>>(),
    ));
    excluded_edges.extend(edge_set_from_sensor_rows(
        &validation_pairs
            .iter()
            .flat_map(|pair| {
                [
                    pair.upper.clone(),
                    pair.lower.clone(),
                    pair.contrast.clone(),
                ]
            })
            .collect::<Vec<_>>(),
    ));
    excluded_edges.extend(edge_set_from_sensor_rows(
        &heldout_pairs
            .iter()
            .flat_map(|pair| {
                [
                    pair.upper.clone(),
                    pair.lower.clone(),
                    pair.contrast.clone(),
                ]
            })
            .collect::<Vec<_>>(),
    ));
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.base.rng_seed + 602);
    let train_edges = select_observation_edges(
        &topology,
        &coords,
        &excluded_edges,
        &BTreeSet::new(),
        config.total_observation_budget,
        &mut rng,
    );
    let mut used_edges = train_edges.iter().copied().collect::<BTreeSet<_>>();
    let validation_edges = select_observation_edges(
        &topology,
        &coords,
        &excluded_edges,
        &used_edges,
        config.validation_local_count,
        &mut rng,
    );
    used_edges.extend(validation_edges.iter().copied());
    let heldout_edges = select_observation_edges(
        &topology,
        &coords,
        &excluded_edges,
        &used_edges,
        config.heldout_local_count,
        &mut rng,
    );

    let train_path_rows = path_pair_rows(&train_pairs).cloned().collect::<Vec<_>>();
    let validation_path_rows = path_pair_rows(&validation_pairs)
        .cloned()
        .collect::<Vec<_>>();
    let validation_contrast_rows = validation_pairs
        .iter()
        .map(|pair| pair.contrast.clone())
        .collect::<Vec<_>>();
    let heldout_path_rows = path_pair_rows(&heldout_pairs).cloned().collect::<Vec<_>>();
    let heldout_contrast_rows = heldout_pairs
        .iter()
        .map(|pair| pair.contrast.clone())
        .collect::<Vec<_>>();

    let mut base_train = edge_observations(
        PlanarHolesObservationKind::Local,
        "train_local",
        &train_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_train.extend(loop_observations(
        PlanarHolesObservationKind::TrainLoop,
        "train_loop",
        &train_cycle_matrix,
        &train_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    base_train.extend(row_observations(
        PlanarHolesObservationKind::TrainPathHomology,
        &train_path_rows,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    let mut base_validation = edge_observations(
        PlanarHolesObservationKind::HeldoutLocal,
        "validation_local",
        &validation_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_validation.extend(loop_observations(
        PlanarHolesObservationKind::HeldoutLoop,
        "validation_loop",
        &validation_cycle_matrix,
        &validation_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    base_validation.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathHomology,
        &validation_path_rows,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    base_validation.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathContrast,
        &validation_contrast_rows,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    let mut base_heldout = edge_observations(
        PlanarHolesObservationKind::HeldoutLocal,
        "heldout_local",
        &heldout_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_heldout.extend(loop_observations(
        PlanarHolesObservationKind::HeldoutLoop,
        "heldout_loop",
        &heldout_cycle_matrix,
        &heldout_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    base_heldout.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathHomology,
        &heldout_path_rows,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    base_heldout.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathContrast,
        &heldout_contrast_rows,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));

    let base = condition_topology_vs_naive_experiment(
        config,
        topology_summary,
        train_rank,
        validation_rank,
        heldout_rank,
        &topology,
        &coords,
        &metric,
        &hodge,
        &harmonic_basis,
        &truth,
        &base_train,
        &base_validation,
        &base_heldout,
    )?;
    let rows = base
        .rows
        .iter()
        .map(|row| {
            let predictions = base
                .heldout_predictions
                .iter()
                .filter(|prediction| prediction.model == row.model)
                .cloned()
                .collect::<Vec<_>>();
            let path = prediction_kind_stats(
                &predictions,
                PlanarHolesObservationKind::HeldoutPathHomology,
            );
            let contrast = prediction_kind_stats(
                &predictions,
                PlanarHolesObservationKind::HeldoutPathContrast,
            );
            let loop_stats =
                prediction_kind_stats(&predictions, PlanarHolesObservationKind::HeldoutLoop);
            let variance_multiplier =
                calibration_multiplier_for_model(&base.calibration_rows, row.model);
            let calibrated_path = prediction_kind_stats_with_variance_multiplier(
                &predictions,
                PlanarHolesObservationKind::HeldoutPathHomology,
                variance_multiplier,
            );
            let calibrated_contrast = prediction_kind_stats_with_variance_multiplier(
                &predictions,
                PlanarHolesObservationKind::HeldoutPathContrast,
                variance_multiplier,
            );
            let calibrated_loop = prediction_kind_stats_with_variance_multiplier(
                &predictions,
                PlanarHolesObservationKind::HeldoutLoop,
                variance_multiplier,
            );
            PlanarHolesPathHomologySummaryRow {
                model: row.model,
                train_path_pair_count: train_pairs.len(),
                validation_path_pair_count: validation_pairs.len(),
                heldout_path_pair_count: heldout_pairs.len(),
                path_integral_nlpd: path.mean_nlpd,
                path_integral_relative_error: path.relative_error,
                path_integral_coverage_95: path.coverage_95,
                path_contrast_nlpd: contrast.mean_nlpd,
                path_contrast_relative_error: contrast.relative_error,
                path_contrast_coverage_95: contrast.coverage_95,
                path_contrast_mean_abs_z: contrast.mean_abs_z,
                hole_loop_nlpd: loop_stats.mean_nlpd,
                hole_loop_relative_error: loop_stats.relative_error,
                calibrated_path_integral_nlpd: calibrated_path.mean_nlpd,
                calibrated_path_integral_coverage_95: calibrated_path.coverage_95,
                calibrated_path_contrast_nlpd: calibrated_contrast.mean_nlpd,
                calibrated_path_contrast_coverage_95: calibrated_contrast.coverage_95,
                calibrated_path_contrast_mean_abs_z: calibrated_contrast.mean_abs_z,
                calibrated_hole_loop_nlpd: calibrated_loop.mean_nlpd,
                calibrated_hole_loop_coverage_95: calibrated_loop.coverage_95,
                path_contrast_harmonic_pairing_rank: path_contrast_rank,
            }
        })
        .collect();
    Ok(PlanarHolesPathHomologyTopologyVsNaiveGpResult {
        base,
        domain: PlanarHolesDomainKind::Default,
        path_contrast_harmonic_pairing_rank: path_contrast_rank,
        rows,
    })
}

pub fn run_planar_holes_path_homology_vector_fields(
    config: &PlanarHolesTopologyVsNaiveGpConfig,
) -> Result<PlanarHolesPathHomologyVectorFieldsResult, Box<dyn Error>> {
    validate_config(&config.base)?;
    validate_topology_vs_naive_config(config)?;
    let holes = holes_for_domain(PlanarHolesDomainKind::Default);
    ensure_planar_holes_mesh_with_holes(&config.base, &holes)?;
    fs::create_dir_all(&config.base.output_dir)?;

    let mesh_bytes = fs::read(&config.base.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = summarize_topology(&topology);
    validate_topology(&topology_summary)?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw = compute_harmonic_basis_1form(&topology, &metric, holes.len(), None)
        .map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;
    let (train_cycle_matrix, train_cycles) =
        build_hole_cycle_observation_matrix(&topology, &coords, &holes, 0.0, 0.0, "train")?;
    let harmonic_basis =
        canonicalize_harmonic_basis_by_cycles(&harmonic_basis_orthonormal, &train_cycle_matrix)?;
    let (validation_cycle_matrix, validation_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        0.030,
        std::f64::consts::PI / 11.0,
        "validation",
    )?;
    let (heldout_cycle_matrix, heldout_cycles) = build_hole_cycle_observation_matrix(
        &topology,
        &coords,
        &holes,
        0.055,
        std::f64::consts::PI / 17.0,
        "heldout",
    )?;
    let train_rank = (&train_cycle_matrix * &harmonic_basis).rank(1e-6);
    let validation_rank = (&validation_cycle_matrix * &harmonic_basis).rank(1e-6);
    let heldout_rank = (&heldout_cycle_matrix * &harmonic_basis).rank(1e-6);
    for (label, rank) in [
        ("train", train_rank),
        ("validation", validation_rank),
        ("heldout", heldout_rank),
    ] {
        if rank != holes.len() {
            return Err(invalid_data(format!(
                "{label} cycle-harmonic pairing rank {rank} does not match hole count {}",
                holes.len()
            ))
            .into());
        }
    }

    let hodge_prior =
        build_hodge_prior(&topology, &coords, &metric, &harmonic_basis, &config.base)?;
    let truth = build_truth(
        &topology,
        &coords,
        &metric,
        &hodge.mass_u,
        &hodge_prior,
        &config.base,
    )
    .map_err(|err| invalid_data(format!("path-homology truth construction failed: {err}")))?;

    let train_pairs =
        build_path_homology_pairs(&topology, &coords, &holes, 0.025, 0.0, "train_path_family")?;
    let validation_pairs = build_path_homology_pairs(
        &topology,
        &coords,
        &holes,
        0.040,
        std::f64::consts::PI / 10.0,
        "validation_path_family",
    )?;
    let mut heldout_pairs = build_path_homology_pairs(
        &topology,
        &coords,
        &holes,
        0.035,
        std::f64::consts::PI / 5.0,
        "heldout_path_family_a",
    )?;
    heldout_pairs.extend(build_path_homology_pairs(
        &topology,
        &coords,
        &holes,
        0.060,
        -std::f64::consts::PI / 8.0,
        "heldout_path_family_b",
    )?);
    let contrast_rows = heldout_pairs
        .iter()
        .map(|pair| pair.contrast.clone())
        .collect::<Vec<_>>();
    let contrast_matrix = sensor_rows_to_csr(&contrast_rows, topology.edges().len());
    let path_contrast_rank = (&contrast_matrix * &harmonic_basis).rank(1e-6);
    if path_contrast_rank != holes.len() {
        return Err(invalid_data(format!(
            "path contrast-harmonic pairing rank {path_contrast_rank} does not match hole count {}",
            holes.len()
        ))
        .into());
    }

    let mut excluded_edges = observation_edge_set(&train_cycle_matrix);
    excluded_edges.extend(observation_edge_set(&validation_cycle_matrix));
    excluded_edges.extend(observation_edge_set(&heldout_cycle_matrix));
    excluded_edges.extend(edge_set_from_sensor_rows(
        &train_pairs
            .iter()
            .flat_map(|pair| {
                [
                    pair.upper.clone(),
                    pair.lower.clone(),
                    pair.contrast.clone(),
                ]
            })
            .collect::<Vec<_>>(),
    ));
    excluded_edges.extend(edge_set_from_sensor_rows(
        &validation_pairs
            .iter()
            .flat_map(|pair| {
                [
                    pair.upper.clone(),
                    pair.lower.clone(),
                    pair.contrast.clone(),
                ]
            })
            .collect::<Vec<_>>(),
    ));
    excluded_edges.extend(edge_set_from_sensor_rows(
        &heldout_pairs
            .iter()
            .flat_map(|pair| {
                [
                    pair.upper.clone(),
                    pair.lower.clone(),
                    pair.contrast.clone(),
                ]
            })
            .collect::<Vec<_>>(),
    ));
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.base.rng_seed + 602);
    let train_edges = select_observation_edges(
        &topology,
        &coords,
        &excluded_edges,
        &BTreeSet::new(),
        config.total_observation_budget,
        &mut rng,
    );
    let mut used_edges = train_edges.iter().copied().collect::<BTreeSet<_>>();
    let validation_edges = select_observation_edges(
        &topology,
        &coords,
        &excluded_edges,
        &used_edges,
        config.validation_local_count,
        &mut rng,
    );
    used_edges.extend(validation_edges.iter().copied());
    let heldout_edges = select_observation_edges(
        &topology,
        &coords,
        &excluded_edges,
        &used_edges,
        config.heldout_local_count,
        &mut rng,
    );

    let train_path_rows = path_pair_rows(&train_pairs).cloned().collect::<Vec<_>>();
    let validation_path_rows = path_pair_rows(&validation_pairs)
        .cloned()
        .collect::<Vec<_>>();
    let validation_contrast_rows = validation_pairs
        .iter()
        .map(|pair| pair.contrast.clone())
        .collect::<Vec<_>>();
    let heldout_path_rows = path_pair_rows(&heldout_pairs).cloned().collect::<Vec<_>>();
    let heldout_contrast_rows = heldout_pairs
        .iter()
        .map(|pair| pair.contrast.clone())
        .collect::<Vec<_>>();

    let mut base_train = edge_observations(
        PlanarHolesObservationKind::Local,
        "train_local",
        &train_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_train.extend(loop_observations(
        PlanarHolesObservationKind::TrainLoop,
        "train_loop",
        &train_cycle_matrix,
        &train_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    base_train.extend(row_observations(
        PlanarHolesObservationKind::TrainPathHomology,
        &train_path_rows,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    let mut base_validation = edge_observations(
        PlanarHolesObservationKind::HeldoutLocal,
        "validation_local",
        &validation_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_validation.extend(loop_observations(
        PlanarHolesObservationKind::HeldoutLoop,
        "validation_loop",
        &validation_cycle_matrix,
        &validation_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    base_validation.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathHomology,
        &validation_path_rows,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    base_validation.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathContrast,
        &validation_contrast_rows,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    let mut base_heldout = edge_observations(
        PlanarHolesObservationKind::HeldoutLocal,
        "heldout_local",
        &heldout_edges,
        &truth.mixed,
        config.base.local_noise_variance,
        false,
        &mut rng,
    );
    base_heldout.extend(loop_observations(
        PlanarHolesObservationKind::HeldoutLoop,
        "heldout_loop",
        &heldout_cycle_matrix,
        &heldout_cycles,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));
    base_heldout.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathHomology,
        &heldout_path_rows,
        &truth.mixed,
        config.long_path_noise_variance,
        false,
        &mut rng,
    ));
    base_heldout.extend(row_observations(
        PlanarHolesObservationKind::HeldoutPathContrast,
        &heldout_contrast_rows,
        &truth.mixed,
        config.base.loop_noise_variance,
        false,
        &mut rng,
    ));

    let mut selections = vec![
        select_fixed_topology_reference_model(
            PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
            &topology,
            &coords,
            &metric,
            &hodge,
            &harmonic_basis,
            config,
            &base_train,
            &base_validation,
        )?,
        select_fixed_topology_reference_model(
            PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern,
            &topology,
            &coords,
            &metric,
            &hodge,
            &harmonic_basis,
            config,
            &base_train,
            &base_validation,
        )?,
        select_fixed_topology_reference_model(
            PlanarHolesModelKind::IncompressibleHodgeMatern,
            &topology,
            &coords,
            &metric,
            &hodge,
            &harmonic_basis,
            config,
            &base_train,
            &base_validation,
        )?,
        select_topology_vs_naive_candidate(
            PlanarHolesModelKind::NondecomposedFeec,
            &config.hodge_kappas,
            &config.hodge_taus,
            &config.observation_variance_scales,
            &topology,
            &coords,
            &metric,
            &hodge,
            config.base.alpha,
            &base_train,
            &base_validation,
        )?,
        select_topology_vs_naive_candidate(
            PlanarHolesModelKind::NaiveEuclideanVectorMatern,
            &config.naive_kappas,
            &config.naive_taus,
            &config.observation_variance_scales,
            &topology,
            &coords,
            &metric,
            &hodge,
            config.base.alpha,
            &base_train,
            &base_validation,
        )?,
    ];

    let truth_vtu_path = config.base.output_dir.join("truth_total.vtu");
    visual_output::write_1form_vector_field(
        &truth_vtu_path,
        &coords,
        &topology,
        &Cochain::new(1, truth.mixed.clone()),
        "truth_total",
    )?;

    let mut rows = Vec::new();
    let mut panels = vec![VectorFieldPanelData {
        title: "Truth".to_string(),
        cochain: truth.mixed.clone(),
        l2_error: None,
        path_contrast_relative_error: None,
        hole_loop_relative_error: None,
        codifferential_leakage: None,
    }];
    for selection in selections.drain(..) {
        let mut final_train = base_train.clone();
        final_train.extend(base_validation.iter().cloned());
        let final_train =
            scaled_observation_variances(&final_train, selection.observation_variance_scale);
        let final_heldout =
            scaled_observation_variances(&base_heldout, selection.observation_variance_scale);
        let operator = if let Some(operator) = selection.fixed_operator {
            operator
        } else {
            build_topology_vs_naive_model_operator(
                selection.model,
                &topology,
                &coords,
                &metric,
                &hodge,
                selection.kappa,
                selection.tau,
                config.base.alpha,
            )?
        };
        let (training_matrix, training_values) =
            scaled_observation_system(&final_train, topology.edges().len());
        let heldout_matrix = observation_matrix(&final_heldout, topology.edges().len(), false);
        let conditioned = condition_model(&operator, &training_matrix, &training_values)?;
        let predictions = compute_heldout_predictions(
            PlanarHolesObservationScenario::LocalOnly,
            selection.model,
            &final_heldout,
            &heldout_matrix,
            &conditioned,
        )?;
        let field_coverage = compute_field_coverage_diagnostics(
            PlanarHolesObservationScenario::LocalOnly,
            selection.model,
            &hodge.mass_u,
            &truth.mixed,
            &final_heldout,
            &conditioned,
        )?;
        let metric_row = topology_vs_naive_metric_row(
            &topology,
            &metric,
            &hodge.mass_u,
            &truth,
            &conditioned,
            &predictions,
            &field_coverage.summaries,
            &operator,
            selection.kappa,
            selection.tau,
            selection.observation_variance_scale,
            selection.validation_nlpd,
            final_train.len(),
            base_validation.len(),
            final_heldout.len(),
        )?;
        let all_predictions = predictions.iter().collect::<Vec<_>>();
        let calibrated_all =
            prediction_subset_stats(&all_predictions, selection.variance_multiplier);
        let path = prediction_kind_stats(
            &predictions,
            PlanarHolesObservationKind::HeldoutPathHomology,
        );
        let contrast = prediction_kind_stats(
            &predictions,
            PlanarHolesObservationKind::HeldoutPathContrast,
        );
        let loop_stats =
            prediction_kind_stats(&predictions, PlanarHolesObservationKind::HeldoutLoop);
        let posterior_vtu_path = config
            .base
            .output_dir
            .join(format!("posterior_mean_{}.vtu", selection.model.as_str()));
        let posterior_std = Cochain::new(1, field_coverage.diagnostics.posterior_std.clone());
        let abs_z_score = Cochain::new(1, field_coverage.diagnostics.abs_z_score.clone());
        let covered_95 = Cochain::new(1, field_coverage.diagnostics.covered_95.clone());
        let posterior_mean_error =
            Cochain::new(1, field_coverage.diagnostics.posterior_mean_error.clone());
        visual_output::write_1form_vector_proxy_fields(
            &posterior_vtu_path,
            &coords,
            &topology,
            "posterior_mean",
            &Cochain::new(1, conditioned.posterior_mean.clone()),
            &[
                ("posterior_std", &posterior_std),
                ("abs_z_score", &abs_z_score),
                ("covered_95", &covered_95),
                ("posterior_mean_error", &posterior_mean_error),
            ],
        )?;

        rows.push(PlanarHolesVectorFieldFigureSummaryRow {
            model: selection.model,
            l2_error: metric_row.l2_error,
            heldout_nlpd: metric_row.heldout_nlpd,
            calibrated_heldout_nlpd: calibrated_all.mean_nlpd,
            path_integral_relative_error: path.relative_error,
            path_contrast_relative_error: contrast.relative_error,
            hole_loop_relative_error: loop_stats.relative_error,
            codifferential_leakage: metric_row.codifferential_leakage,
            all_edge_coverage_95: metric_row.all_edge_coverage_95,
            heldout_edge_coverage_95: metric_row.heldout_edge_coverage_95,
            posterior_vtu_path: posterior_vtu_path.clone(),
        });
        panels.push(VectorFieldPanelData {
            title: display_model_name(selection.model).to_string(),
            cochain: conditioned.posterior_mean,
            l2_error: Some(metric_row.l2_error),
            path_contrast_relative_error: Some(contrast.relative_error),
            hole_loop_relative_error: Some(loop_stats.relative_error),
            codifferential_leakage: Some(metric_row.codifferential_leakage),
        });
    }

    let summary_csv_path = config
        .base
        .output_dir
        .join("vector_field_figure_summary.csv");
    write_vector_field_figure_summary(&rows, &summary_csv_path)?;
    let figure_png_path = config
        .base
        .output_dir
        .join("posterior_mean_vector_fields.png");
    render_vector_field_panel_png(&figure_png_path, &topology, &coords, &holes, &panels)?;

    Ok(PlanarHolesPathHomologyVectorFieldsResult {
        topology_summary,
        train_cycle_harmonic_pairing_rank: train_rank,
        validation_cycle_harmonic_pairing_rank: validation_rank,
        heldout_cycle_harmonic_pairing_rank: heldout_rank,
        path_contrast_harmonic_pairing_rank: path_contrast_rank,
        output_dir: config.base.output_dir.clone(),
        truth_vtu_path,
        figure_png_path,
        summary_csv_path,
        rows,
    })
}

#[allow(clippy::too_many_arguments)]
fn condition_topology_vs_naive_experiment(
    config: &PlanarHolesTopologyVsNaiveGpConfig,
    topology_summary: PlanarHolesTopologySummary,
    train_cycle_harmonic_pairing_rank: usize,
    validation_cycle_harmonic_pairing_rank: usize,
    heldout_cycle_harmonic_pairing_rank: usize,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    harmonic_basis: &FeecMatrix,
    truth: &TruthFields,
    base_train: &[PlanarHolesObservation],
    base_validation: &[PlanarHolesObservation],
    base_heldout: &[PlanarHolesObservation],
) -> Result<PlanarHolesTopologyVsNaiveGpResult, Box<dyn Error>> {
    let hodge_selection = select_topology_vs_naive_candidate(
        PlanarHolesModelKind::NondecomposedFeec,
        &config.hodge_kappas,
        &config.hodge_taus,
        &config.observation_variance_scales,
        topology,
        coords,
        metric,
        hodge,
        config.base.alpha,
        base_train,
        base_validation,
    )?;
    let naive_selection = select_topology_vs_naive_candidate(
        PlanarHolesModelKind::NaiveEuclideanVectorMatern,
        &config.naive_kappas,
        &config.naive_taus,
        &config.observation_variance_scales,
        topology,
        coords,
        metric,
        hodge,
        config.base.alpha,
        base_train,
        base_validation,
    )?;
    let coexact_harmonic_selection = select_fixed_topology_reference_model(
        PlanarHolesModelKind::IncompressibleHodgeMatern,
        topology,
        coords,
        metric,
        hodge,
        harmonic_basis,
        config,
        base_train,
        base_validation,
    )?;
    let trace_matched_selection = select_fixed_topology_reference_model(
        PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern,
        topology,
        coords,
        metric,
        hodge,
        harmonic_basis,
        config,
        base_train,
        base_validation,
    )?;
    let spectral_selection = select_fixed_topology_reference_model(
        PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
        topology,
        coords,
        metric,
        hodge,
        harmonic_basis,
        config,
        base_train,
        base_validation,
    )?;

    let mut rows = Vec::new();
    let mut selections = vec![
        hodge_selection,
        naive_selection,
        coexact_harmonic_selection,
        trace_matched_selection,
        spectral_selection,
    ];
    let mut tuning_rows = Vec::new();
    for selection in &mut selections {
        tuning_rows.append(&mut selection.tuning_rows);
    }
    let mut heldout_predictions = Vec::new();
    let mut field_coverage_summaries = Vec::new();
    let mut calibration_rows = Vec::new();
    for selection in selections {
        let mut final_train = base_train.to_vec();
        final_train.extend(base_validation.iter().cloned());
        let final_train =
            scaled_observation_variances(&final_train, selection.observation_variance_scale);
        let final_heldout =
            scaled_observation_variances(base_heldout, selection.observation_variance_scale);
        let operator = if let Some(operator) = selection.fixed_operator {
            operator
        } else {
            build_topology_vs_naive_model_operator(
                selection.model,
                topology,
                coords,
                metric,
                hodge,
                selection.kappa,
                selection.tau,
                config.base.alpha,
            )?
        };
        let (training_matrix, training_values) =
            scaled_observation_system(&final_train, topology.edges().len());
        let heldout_matrix = observation_matrix(&final_heldout, topology.edges().len(), false);
        let conditioned = condition_model(&operator, &training_matrix, &training_values)?;
        let predictions = compute_heldout_predictions(
            PlanarHolesObservationScenario::LocalOnly,
            selection.model,
            &final_heldout,
            &heldout_matrix,
            &conditioned,
        )?;
        let field_coverage = compute_field_coverage_diagnostics(
            PlanarHolesObservationScenario::LocalOnly,
            selection.model,
            &hodge.mass_u,
            &truth.mixed,
            &final_heldout,
            &conditioned,
        )?;
        rows.push(topology_vs_naive_metric_row(
            topology,
            metric,
            &hodge.mass_u,
            truth,
            &conditioned,
            &predictions,
            &field_coverage.summaries,
            &operator,
            selection.kappa,
            selection.tau,
            selection.observation_variance_scale,
            selection.validation_nlpd,
            final_train.len(),
            base_validation.len(),
            final_heldout.len(),
        )?);
        calibration_rows.extend(calibration_rows_for_predictions(
            selection.model,
            selection.variance_multiplier,
            &predictions,
        ));
        heldout_predictions.extend(predictions);
        field_coverage_summaries.extend(field_coverage.summaries);
    }

    Ok(PlanarHolesTopologyVsNaiveGpResult {
        topology_summary,
        train_cycle_harmonic_pairing_rank,
        validation_cycle_harmonic_pairing_rank,
        heldout_cycle_harmonic_pairing_rank,
        rows,
        tuning_rows,
        calibration_rows,
        heldout_predictions,
        field_coverage_summaries,
    })
}

#[derive(Debug, Clone)]
struct TopologyVsNaiveCandidateSelection {
    model: PlanarHolesModelKind,
    kappa: f64,
    tau: f64,
    observation_variance_scale: f64,
    validation_nlpd: f64,
    variance_multiplier: f64,
    fixed_operator: Option<ModelOperators>,
    tuning_rows: Vec<PlanarHolesTopologyVsNaiveGpTuningRow>,
}

#[allow(clippy::too_many_arguments)]
fn select_topology_vs_naive_candidate(
    model: PlanarHolesModelKind,
    kappas: &[f64],
    taus: &[f64],
    observation_variance_scales: &[f64],
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    alpha: MaternAlpha,
    base_train: &[PlanarHolesObservation],
    base_validation: &[PlanarHolesObservation],
) -> Result<TopologyVsNaiveCandidateSelection, Box<dyn Error>> {
    let mut best: Option<TopologyVsNaiveCandidateSelection> = None;
    let mut tuning_rows = Vec::new();
    for &kappa in kappas {
        for &tau in taus {
            let operator = build_topology_vs_naive_model_operator(
                model, topology, coords, metric, hodge, kappa, tau, alpha,
            )?;
            for &scale in observation_variance_scales {
                let train = scaled_observation_variances(base_train, scale);
                let validation = scaled_observation_variances(base_validation, scale);
                let (training_matrix, training_values) =
                    scaled_observation_system(&train, topology.edges().len());
                let validation_matrix =
                    observation_matrix(&validation, topology.edges().len(), false);
                let conditioned = condition_model(&operator, &training_matrix, &training_values)?;
                let predictions = compute_heldout_predictions(
                    PlanarHolesObservationScenario::LocalOnly,
                    model,
                    &validation,
                    &validation_matrix,
                    &conditioned,
                )?;
                let validation_nlpd = mean(predictions.iter().map(|row| row.nlpd));
                tuning_rows.push(PlanarHolesTopologyVsNaiveGpTuningRow {
                    model,
                    kappa,
                    tau,
                    observation_variance_scale: scale,
                    validation_nlpd,
                });
                if validation_nlpd.is_finite()
                    && best
                        .as_ref()
                        .is_none_or(|current| validation_nlpd < current.validation_nlpd)
                {
                    let variance_multiplier = fit_predictive_variance_multiplier(&predictions);
                    best = Some(TopologyVsNaiveCandidateSelection {
                        model,
                        kappa,
                        tau,
                        observation_variance_scale: scale,
                        validation_nlpd,
                        variance_multiplier,
                        fixed_operator: None,
                        tuning_rows: Vec::new(),
                    });
                }
            }
        }
    }
    let mut best = best.ok_or_else(|| {
        invalid_data(format!(
            "no finite validation score for topology-vs-naive model {}",
            model.as_str()
        ))
    })?;
    best.tuning_rows = tuning_rows;
    Ok(best)
}

fn build_topology_vs_naive_model_operator(
    model: PlanarHolesModelKind,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    kappa: f64,
    tau: f64,
    alpha: MaternAlpha,
) -> Result<ModelOperators, Box<dyn Error>> {
    match model {
        PlanarHolesModelKind::NondecomposedFeec => build_nondecomposed_feec_model_operator(
            topology, coords, metric, hodge, kappa, tau, alpha,
        ),
        PlanarHolesModelKind::NaiveEuclideanVectorMatern => {
            build_naive_euclidean_vector_matern_model_operator(topology, coords, kappa, tau)
        }
        _ => Err(invalid_input(format!(
            "unsupported topology-vs-naive model {}",
            model.as_str()
        ))
        .into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn select_fixed_topology_reference_model(
    model: PlanarHolesModelKind,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    harmonic_basis: &FeecMatrix,
    config: &PlanarHolesTopologyVsNaiveGpConfig,
    base_train: &[PlanarHolesObservation],
    base_validation: &[PlanarHolesObservation],
) -> Result<TopologyVsNaiveCandidateSelection, Box<dyn Error>> {
    let operator = build_topology_reference_model_operator(
        model,
        topology,
        coords,
        metric,
        hodge,
        harmonic_basis,
        config,
    )?;
    let train = scaled_observation_variances(base_train, 1.0);
    let validation = scaled_observation_variances(base_validation, 1.0);
    let (training_matrix, training_values) =
        scaled_observation_system(&train, topology.edges().len());
    let validation_matrix = observation_matrix(&validation, topology.edges().len(), false);
    let conditioned = condition_model(&operator, &training_matrix, &training_values)?;
    let predictions = compute_heldout_predictions(
        PlanarHolesObservationScenario::LocalOnly,
        model,
        &validation,
        &validation_matrix,
        &conditioned,
    )?;
    let validation_nlpd = mean(predictions.iter().map(|row| row.nlpd));
    Ok(TopologyVsNaiveCandidateSelection {
        model,
        kappa: fixed_reference_kappa(model, &config.base),
        tau: fixed_reference_tau(model, &config.base),
        observation_variance_scale: 1.0,
        validation_nlpd,
        variance_multiplier: fit_predictive_variance_multiplier(&predictions),
        fixed_operator: Some(operator),
        tuning_rows: Vec::new(),
    })
}

fn fixed_reference_kappa(model: PlanarHolesModelKind, config: &PlanarHolesFlowConfig) -> f64 {
    match model {
        PlanarHolesModelKind::IncompressibleHodgeMatern
        | PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
        | PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
        | PlanarHolesModelKind::SpectralIncompressibleHodgeGp => config.coexact_kappa,
        _ => f64::NAN,
    }
}

fn fixed_reference_tau(model: PlanarHolesModelKind, config: &PlanarHolesFlowConfig) -> f64 {
    match model {
        PlanarHolesModelKind::IncompressibleHodgeMatern
        | PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
        | PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
        | PlanarHolesModelKind::SpectralIncompressibleHodgeGp => {
            config.tau * config.coexact_tau_scale
        }
        _ => f64::NAN,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_topology_reference_model_operator(
    model: PlanarHolesModelKind,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    harmonic_basis: &FeecMatrix,
    config: &PlanarHolesTopologyVsNaiveGpConfig,
) -> Result<ModelOperators, Box<dyn Error>> {
    let mut reference_config = config.base.clone();
    reference_config.spectral_exact_mode_count = 0;
    reference_config.spectral_harmonic_mode_count = HARMONIC_COEFFICIENTS.len();
    reference_config.spectral_branch_energy_normalization = true;
    reference_config.spectral_coexact_expected_m1_energy = 1.0;
    reference_config.spectral_harmonic_expected_m1_energy = 4.0;
    match model {
        PlanarHolesModelKind::IncompressibleHodgeMatern => {
            let prior = build_hodge_prior_with_branches(
                topology,
                coords,
                metric,
                harmonic_basis,
                &reference_config,
                vec![HodgeBranchKind::Coexact, HodgeBranchKind::Harmonic],
            )?;
            let branch_transforms = branch_transforms(&prior);
            Ok(ModelOperators {
                model,
                prior_precision: prior.precision,
                latent_to_ambient: prior.latent_to_ambient,
                branch_transforms,
                spectral_branch_stats: BTreeMap::new(),
                coexact_tau_scale: reference_config.coexact_tau_scale,
            })
        }
        PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern => {
            build_exact_mass_incompressible_model_operator(
                topology,
                coords,
                metric,
                &hodge.mass_u,
                harmonic_basis,
                &reference_config,
                true,
            )
        }
        PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern => {
            let exact_lower_mass_inverse =
                build_exact_dense_mass_inverse_1form(&hodge.mass_u, 1e-14).map_err(invalid_data)?;
            build_exact_lower_incompressible_model_operator(
                topology,
                coords,
                metric,
                &hodge.mass_u,
                &exact_lower_mass_inverse,
                harmonic_basis,
                &reference_config,
                true,
            )
        }
        PlanarHolesModelKind::SpectralIncompressibleHodgeGp => {
            let basis = build_spectral_hodge_basis(
                topology,
                metric,
                harmonic_basis.ncols(),
                &reference_config,
            )?;
            build_spectral_hodge_model_operator(
                basis,
                &reference_config,
                model,
                &[HodgeBranchKind::Coexact, HodgeBranchKind::Harmonic],
            )
        }
        _ => Err(invalid_input(format!(
            "unsupported fixed topology reference model {}",
            model.as_str()
        ))
        .into()),
    }
}

fn scaled_observation_variances(
    observations: &[PlanarHolesObservation],
    scale: f64,
) -> Vec<PlanarHolesObservation> {
    observations
        .iter()
        .cloned()
        .map(|mut observation| {
            observation.noise_variance *= scale;
            observation
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn topology_vs_naive_metric_row(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    truth: &TruthFields,
    conditioned: &ConditionedModel,
    predictions: &[PlanarHolesHeldoutPrediction],
    field_coverage_summaries: &[PlanarHolesFieldCoverageSummary],
    operator: &ModelOperators,
    selected_kappa: f64,
    selected_tau: f64,
    selected_observation_variance_scale: f64,
    validation_nlpd: f64,
    training_observation_count: usize,
    validation_observation_count: usize,
    heldout_observation_count: usize,
) -> Result<PlanarHolesTopologyVsNaiveGpMetricRow, Box<dyn Error>> {
    let error = &conditioned.posterior_mean - &truth.mixed;
    let l2_error = relative_or_nan(
        mass_norm(&error, mass_1form),
        mass_norm(&truth.mixed, mass_1form),
    );
    let posterior_d = de_rham::derivative(topology, 1, &conditioned.posterior_mean);
    let truth_d = de_rham::derivative(topology, 1, &truth.mixed);
    let d_error = &posterior_d - &truth_d;
    let mass_2form = de_rham::mass_matrix_form(topology, metric, 2).map_err(invalid_data)?;
    let exterior_derivative_error = relative_or_nan(
        mass_norm(&d_error, &mass_2form),
        mass_norm(&truth_d, &mass_2form),
    );
    let spectral_coexact_stats = operator
        .spectral_branch_stats
        .get(&HodgeBranchKind::Coexact)
        .copied();
    let spectral_harmonic_stats = operator
        .spectral_branch_stats
        .get(&HodgeBranchKind::Harmonic)
        .copied();
    Ok(PlanarHolesTopologyVsNaiveGpMetricRow {
        model: conditioned.model,
        selected_kappa,
        selected_tau,
        selected_observation_variance_scale,
        model_coexact_tau_scale: operator.coexact_tau_scale,
        spectral_coexact_requested_modes: spectral_coexact_stats
            .map(|stats| stats.requested_mode_count)
            .unwrap_or(0),
        spectral_coexact_actual_modes: spectral_coexact_stats
            .map(|stats| stats.actual_mode_count)
            .unwrap_or(0),
        spectral_coexact_expected_m1_energy: spectral_coexact_stats
            .map(|stats| stats.expected_m1_energy)
            .unwrap_or(f64::NAN),
        spectral_harmonic_requested_modes: spectral_harmonic_stats
            .map(|stats| stats.requested_mode_count)
            .unwrap_or(0),
        spectral_harmonic_actual_modes: spectral_harmonic_stats
            .map(|stats| stats.actual_mode_count)
            .unwrap_or(0),
        spectral_harmonic_expected_m1_energy: spectral_harmonic_stats
            .map(|stats| stats.expected_m1_energy)
            .unwrap_or(f64::NAN),
        validation_nlpd,
        training_observation_count,
        validation_observation_count,
        heldout_observation_count,
        l2_error,
        heldout_nlpd: mean(predictions.iter().map(|row| row.nlpd)),
        heldout_local_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutLocal,
        ),
        heldout_loop_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutLoop,
        ),
        heldout_interior_loop_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutInteriorLoop,
        ),
        heldout_long_path_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutLongPath,
        ),
        heldout_local_relative_error: heldout_kind_relative_error(
            predictions,
            PlanarHolesObservationKind::HeldoutLocal,
        ),
        heldout_loop_relative_error: heldout_kind_relative_error(
            predictions,
            PlanarHolesObservationKind::HeldoutLoop,
        ),
        heldout_interior_loop_relative_error: heldout_kind_relative_error(
            predictions,
            PlanarHolesObservationKind::HeldoutInteriorLoop,
        ),
        heldout_long_path_relative_error: heldout_kind_relative_error(
            predictions,
            PlanarHolesObservationKind::HeldoutLongPath,
        ),
        exterior_derivative_error,
        codifferential_leakage: codifferential_leakage(
            topology,
            metric,
            mass_1form,
            &conditioned.posterior_mean,
        )?,
        all_edge_coverage_95: field_coverage_value(
            field_coverage_summaries,
            conditioned.model,
            PlanarHolesFieldCoverageSubset::AllEdges,
        ),
        heldout_edge_coverage_95: field_coverage_value(
            field_coverage_summaries,
            conditioned.model,
            PlanarHolesFieldCoverageSubset::HeldoutLocalEdges,
        ),
    })
}

pub fn write_planar_holes_topology_vs_naive_gp_outputs(
    result: &PlanarHolesTopologyVsNaiveGpResult,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_topology_vs_naive_topology_summary(result, out_dir.join("topology_summary.csv"))?;
    write_topology_vs_naive_metrics(result, out_dir.join("metrics_summary.csv"))?;
    write_topology_vs_naive_tuning(result, out_dir.join("validation_summary.csv"))?;
    write_topology_vs_naive_calibration(result, out_dir.join("calibration_summary.csv"))?;
    write_topology_vs_naive_predictions(result, out_dir.join("heldout_predictions.csv"))?;
    write_topology_vs_naive_field_coverage(result, out_dir.join("field_coverage_summary.csv"))?;
    Ok(())
}

pub fn write_planar_holes_barrier_topology_vs_naive_gp_outputs(
    result: &PlanarHolesBarrierTopologyVsNaiveGpResult,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    write_planar_holes_topology_vs_naive_gp_outputs(&result.base, out_dir)?;
    write_barrier_summary(result, out_dir.join("barrier_summary.csv"))?;
    Ok(())
}

pub fn write_planar_holes_path_homology_vs_naive_gp_outputs(
    result: &PlanarHolesPathHomologyTopologyVsNaiveGpResult,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    write_planar_holes_topology_vs_naive_gp_outputs(&result.base, out_dir)?;
    write_path_homology_summary(result, out_dir.join("path_homology_summary.csv"))?;
    Ok(())
}

pub fn write_planar_holes_path_homology_vector_field_outputs(
    result: &PlanarHolesPathHomologyVectorFieldsResult,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_vector_field_figure_summary(
        &result.rows,
        out_dir.join("vector_field_figure_summary.csv"),
    )
}

fn write_vector_field_figure_summary(
    rows: &[PlanarHolesVectorFieldFigureSummaryRow],
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,l2_error,heldout_nlpd,calibrated_heldout_nlpd,path_integral_relative_error,path_contrast_relative_error,hole_loop_relative_error,codifferential_leakage,all_edge_coverage_95,heldout_edge_coverage_95,posterior_vtu_path"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{}",
            row.model.as_str(),
            row.l2_error,
            row.heldout_nlpd,
            row.calibrated_heldout_nlpd,
            row.path_integral_relative_error,
            row.path_contrast_relative_error,
            row.hole_loop_relative_error,
            row.codifferential_leakage,
            row.all_edge_coverage_95,
            row.heldout_edge_coverage_95,
            row.posterior_vtu_path.display()
        )?;
    }
    Ok(())
}

fn display_model_name(model: PlanarHolesModelKind) -> &'static str {
    match model {
        PlanarHolesModelKind::SpectralIncompressibleHodgeGp => "Spectral coexact+harmonic GP",
        PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern => {
            "Exact-lower trace-matched GMRF"
        }
        PlanarHolesModelKind::IncompressibleHodgeMatern => "Sparse coexact+harmonic GMRF",
        PlanarHolesModelKind::NondecomposedFeec => "Nondecomposed FEEC GMRF",
        PlanarHolesModelKind::NaiveEuclideanVectorMatern => "Naive Euclidean vector GP",
        _ => model.as_str(),
    }
}

fn render_vector_field_panel_png(
    path: impl AsRef<Path>,
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    panels: &[VectorFieldPanelData],
) -> io::Result<()> {
    let reconstruction =
        build_reconstructed_barycenter_field_operator(topology, coords).map_err(invalid_data)?;
    let barycenters = topology
        .cells()
        .handle_iter()
        .map(|cell| {
            let barycenter = cell.coord_simplex(coords).barycenter();
            [barycenter[0], barycenter[1]]
        })
        .collect::<Vec<_>>();
    let mut reconstructed = Vec::with_capacity(panels.len());
    for panel in panels {
        let field = reconstruction
            .apply_to_slice(panel.cochain.as_slice())
            .map_err(invalid_data)?;
        let components = field.components();
        if components.len() < 2 {
            return Err(invalid_data(
                "reconstructed vector field is not two-dimensional",
            ));
        }
        let vectors = (0..field.cell_count())
            .map(|index| [components[0][index], components[1][index]])
            .collect::<Vec<_>>();
        reconstructed.push(vectors);
    }
    let vector_scale = percentile(
        reconstructed
            .iter()
            .flat_map(|vectors| vectors.iter())
            .map(|vector| (vector[0] * vector[0] + vector[1] * vector[1]).sqrt()),
        0.95,
    )
    .max(1e-12);

    let image_size = (1800, 1200);
    let root = BitMapBackend::new(path.as_ref(), image_size).into_drawing_area();
    root.fill(&WHITE).map_err(plotters_io_error)?;
    let areas = root.split_evenly((2, 3));
    let mesh_edges = topology
        .edges()
        .handle_iter()
        .map(|edge| {
            let a = coords.coord(edge.vertices[0]);
            let b = coords.coord(edge.vertices[1]);
            ([a[0], a[1]], [b[0], b[1]])
        })
        .collect::<Vec<_>>();
    for (panel_index, area) in areas.iter().enumerate() {
        area.fill(&WHITE).map_err(plotters_io_error)?;
        if let Some(panel) = panels.get(panel_index) {
            draw_vector_field_panel(
                area,
                panel,
                &reconstructed[panel_index],
                &barycenters,
                &mesh_edges,
                holes,
                vector_scale,
            )?;
        }
    }
    root.present().map_err(plotters_io_error)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_vector_field_panel(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    panel: &VectorFieldPanelData,
    vectors: &[[f64; 2]],
    barycenters: &[[f64; 2]],
    mesh_edges: &[([f64; 2], [f64; 2])],
    holes: &[PlanarHoleSpec],
    vector_scale: f64,
) -> io::Result<()> {
    let (width, height) = area.dim_in_pixel();
    let margin = 28.0;
    let title_height = 78.0;
    let plot_left = margin;
    let plot_right = width as f64 - margin;
    let plot_top = title_height;
    let plot_bottom = height as f64 - margin;
    let plot_width = (plot_right - plot_left).max(1.0);
    let plot_height = (plot_bottom - plot_top).max(1.0);
    let side = plot_width.min(plot_height);
    let x_offset = plot_left + 0.5 * (plot_width - side);
    let y_offset = plot_top + 0.5 * (plot_height - side);
    let to_pixel = |point: [f64; 2]| -> (i32, i32) {
        let x = x_offset + point[0] * side;
        let y = y_offset + (1.0 - point[1]) * side;
        (x.round() as i32, y.round() as i32)
    };

    area.draw(&Rectangle::new(
        [
            (x_offset.round() as i32, y_offset.round() as i32),
            (
                (x_offset + side).round() as i32,
                (y_offset + side).round() as i32,
            ),
        ],
        ShapeStyle::from(&WHITE).filled(),
    ))
    .map_err(plotters_io_error)?;
    for (a, b) in mesh_edges {
        area.draw(&PathElement::new(
            vec![to_pixel(*a), to_pixel(*b)],
            RGBColor(221, 225, 229),
        ))
        .map_err(plotters_io_error)?;
    }
    for hole in holes {
        let center = to_pixel(hole.center);
        let radius = (hole.radius * side).round() as i32;
        area.draw(&Circle::new(
            center,
            radius,
            ShapeStyle::from(&WHITE).filled(),
        ))
        .map_err(plotters_io_error)?;
        area.draw(&Circle::new(
            center,
            radius,
            ShapeStyle::from(&RGBColor(120, 128, 138)).stroke_width(2),
        ))
        .map_err(plotters_io_error)?;
    }

    let stride = barycenters.len().div_ceil(400).max(1);
    for (index, (point, vector)) in barycenters.iter().zip(vectors.iter()).enumerate() {
        if index % stride != 0 {
            continue;
        }
        let norm = (vector[0] * vector[0] + vector[1] * vector[1]).sqrt();
        if norm <= 1e-14 {
            continue;
        }
        let length = (0.035 * norm / vector_scale).clamp(0.006, 0.055);
        let direction = [vector[0] / norm, vector[1] / norm];
        let start = [
            point[0] - 0.5 * length * direction[0],
            point[1] - 0.5 * length * direction[1],
        ];
        let end = [
            point[0] + 0.5 * length * direction[0],
            point[1] + 0.5 * length * direction[1],
        ];
        draw_arrow(area, to_pixel(start), to_pixel(end), &RGBColor(21, 72, 121))?;
    }

    let title = if let (Some(e1), Some(contrast), Some(loop_error), Some(leakage)) = (
        panel.l2_error,
        panel.path_contrast_relative_error,
        panel.hole_loop_relative_error,
        panel.codifferential_leakage,
    ) {
        format!(
            "{}\nE1={:.3}  contrast={:.3}  loop={:.3}  leakage={:.3}",
            panel.title, e1, contrast, loop_error, leakage
        )
    } else {
        panel.title.clone()
    };
    let font = ("sans-serif", 23).into_font().color(&BLACK);
    for (line_index, line) in title.lines().enumerate() {
        area.draw(&Text::new(
            line.to_string(),
            (24, 26 + 28 * line_index as i32),
            font.clone(),
        ))
        .map_err(plotters_io_error)?;
    }
    Ok(())
}

fn draw_arrow(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    start: (i32, i32),
    end: (i32, i32),
    color: &RGBColor,
) -> io::Result<()> {
    let style = ShapeStyle::from(color).stroke_width(1);
    area.draw(&PathElement::new(vec![start, end], style))
        .map_err(plotters_io_error)?;
    let dx = (end.0 - start.0) as f64;
    let dy = (end.1 - start.1) as f64;
    let length = (dx * dx + dy * dy).sqrt();
    if length <= 2.0 {
        return Ok(());
    }
    let ux = dx / length;
    let uy = dy / length;
    let head = 6.0_f64.min(0.45 * length);
    let left = (
        (end.0 as f64 - head * (ux + 0.55 * uy)).round() as i32,
        (end.1 as f64 - head * (uy - 0.55 * ux)).round() as i32,
    );
    let right = (
        (end.0 as f64 - head * (ux - 0.55 * uy)).round() as i32,
        (end.1 as f64 - head * (uy + 0.55 * ux)).round() as i32,
    );
    area.draw(&PathElement::new(vec![left, end, right], style))
        .map_err(plotters_io_error)
}

fn plotters_io_error<E: std::fmt::Debug>(error: E) -> io::Error {
    io::Error::other(format!("{error:?}"))
}

fn write_barrier_summary(
    result: &PlanarHolesBarrierTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "domain,model,train_left_edge_count,validation_left_edge_count,heldout_right_edge_count,heldout_cross_barrier_path_count,cross_barrier_local_nlpd,cross_barrier_local_relative_error,barrier_long_path_nlpd,barrier_long_path_relative_error,hole_loop_nlpd,hole_loop_relative_error,cross_barrier_local_coverage_95,barrier_long_path_coverage_95,hole_loop_coverage_95,calibrated_cross_barrier_local_nlpd,calibrated_barrier_long_path_nlpd,calibrated_hole_loop_nlpd,calibrated_cross_barrier_local_coverage_95,calibrated_barrier_long_path_coverage_95,calibrated_hole_loop_coverage_95"
    )?;
    for row in &result.rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            result.domain.as_str(),
            row.model.as_str(),
            row.train_left_edge_count,
            row.validation_left_edge_count,
            row.heldout_right_edge_count,
            row.heldout_cross_barrier_path_count,
            row.cross_barrier_local_nlpd,
            row.cross_barrier_local_relative_error,
            row.barrier_long_path_nlpd,
            row.barrier_long_path_relative_error,
            row.hole_loop_nlpd,
            row.hole_loop_relative_error,
            row.cross_barrier_local_coverage_95,
            row.barrier_long_path_coverage_95,
            row.hole_loop_coverage_95,
            row.calibrated_cross_barrier_local_nlpd,
            row.calibrated_barrier_long_path_nlpd,
            row.calibrated_hole_loop_nlpd,
            row.calibrated_cross_barrier_local_coverage_95,
            row.calibrated_barrier_long_path_coverage_95,
            row.calibrated_hole_loop_coverage_95
        )?;
    }
    Ok(())
}

fn write_path_homology_summary(
    result: &PlanarHolesPathHomologyTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "domain,model,train_path_pair_count,validation_path_pair_count,heldout_path_pair_count,path_integral_nlpd,path_integral_relative_error,path_integral_coverage_95,path_contrast_nlpd,path_contrast_relative_error,path_contrast_coverage_95,path_contrast_mean_abs_z,hole_loop_nlpd,hole_loop_relative_error,calibrated_path_integral_nlpd,calibrated_path_integral_coverage_95,calibrated_path_contrast_nlpd,calibrated_path_contrast_coverage_95,calibrated_path_contrast_mean_abs_z,calibrated_hole_loop_nlpd,calibrated_hole_loop_coverage_95,path_contrast_harmonic_pairing_rank"
    )?;
    for row in &result.rows {
        writeln!(
            writer,
            "{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{}",
            result.domain.as_str(),
            row.model.as_str(),
            row.train_path_pair_count,
            row.validation_path_pair_count,
            row.heldout_path_pair_count,
            row.path_integral_nlpd,
            row.path_integral_relative_error,
            row.path_integral_coverage_95,
            row.path_contrast_nlpd,
            row.path_contrast_relative_error,
            row.path_contrast_coverage_95,
            row.path_contrast_mean_abs_z,
            row.hole_loop_nlpd,
            row.hole_loop_relative_error,
            row.calibrated_path_integral_nlpd,
            row.calibrated_path_integral_coverage_95,
            row.calibrated_path_contrast_nlpd,
            row.calibrated_path_contrast_coverage_95,
            row.calibrated_path_contrast_mean_abs_z,
            row.calibrated_hole_loop_nlpd,
            row.calibrated_hole_loop_coverage_95,
            row.path_contrast_harmonic_pairing_rank
        )?;
    }
    Ok(())
}

fn write_topology_vs_naive_topology_summary(
    result: &PlanarHolesTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "vertices,edges,faces,euler_characteristic,b0,b1,b2,boundary_edges,train_cycle_harmonic_pairing_rank,validation_cycle_harmonic_pairing_rank,heldout_cycle_harmonic_pairing_rank"
    )?;
    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{}",
        result.topology_summary.vertex_count,
        result.topology_summary.edge_count,
        result.topology_summary.face_count,
        result.topology_summary.euler_characteristic,
        result.topology_summary.b0,
        result.topology_summary.b1,
        result.topology_summary.b2,
        result.topology_summary.boundary_edge_count,
        result.train_cycle_harmonic_pairing_rank,
        result.validation_cycle_harmonic_pairing_rank,
        result.heldout_cycle_harmonic_pairing_rank
    )
}

fn write_topology_vs_naive_metrics(
    result: &PlanarHolesTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,selected_kappa,selected_tau,selected_observation_variance_scale,model_coexact_tau_scale,spectral_coexact_requested_modes,spectral_coexact_actual_modes,spectral_coexact_expected_m1_energy,spectral_harmonic_requested_modes,spectral_harmonic_actual_modes,spectral_harmonic_expected_m1_energy,validation_nlpd,training_observation_count,validation_observation_count,heldout_observation_count,l2_error,heldout_nlpd,heldout_local_nlpd,heldout_loop_nlpd,heldout_interior_loop_nlpd,heldout_long_path_nlpd,heldout_local_relative_error,heldout_loop_relative_error,heldout_interior_loop_relative_error,heldout_long_path_relative_error,exterior_derivative_error,codifferential_leakage,all_edge_coverage_95,heldout_edge_coverage_95"
    )?;
    for row in &result.rows {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12},{},{},{:.12},{},{},{:.12},{:.12},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.model.as_str(),
            row.selected_kappa,
            row.selected_tau,
            row.selected_observation_variance_scale,
            row.model_coexact_tau_scale,
            row.spectral_coexact_requested_modes,
            row.spectral_coexact_actual_modes,
            row.spectral_coexact_expected_m1_energy,
            row.spectral_harmonic_requested_modes,
            row.spectral_harmonic_actual_modes,
            row.spectral_harmonic_expected_m1_energy,
            row.validation_nlpd,
            row.training_observation_count,
            row.validation_observation_count,
            row.heldout_observation_count,
            row.l2_error,
            row.heldout_nlpd,
            row.heldout_local_nlpd,
            row.heldout_loop_nlpd,
            row.heldout_interior_loop_nlpd,
            row.heldout_long_path_nlpd,
            row.heldout_local_relative_error,
            row.heldout_loop_relative_error,
            row.heldout_interior_loop_relative_error,
            row.heldout_long_path_relative_error,
            row.exterior_derivative_error,
            row.codifferential_leakage,
            row.all_edge_coverage_95,
            row.heldout_edge_coverage_95
        )?;
    }
    Ok(())
}

fn write_topology_vs_naive_tuning(
    result: &PlanarHolesTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,kappa,tau,observation_variance_scale,validation_nlpd"
    )?;
    for row in &result.tuning_rows {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12}",
            row.model.as_str(),
            row.kappa,
            row.tau,
            row.observation_variance_scale,
            row.validation_nlpd
        )?;
    }
    Ok(())
}

fn write_topology_vs_naive_calibration(
    result: &PlanarHolesTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,kind,count,variance_multiplier,raw_nlpd,calibrated_nlpd,raw_coverage_95,calibrated_coverage_95,raw_mean_abs_z,calibrated_mean_abs_z,relative_error"
    )?;
    for row in &result.calibration_rows {
        writeln!(
            writer,
            "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.model.as_str(),
            row.kind,
            row.count,
            row.variance_multiplier,
            row.raw_nlpd,
            row.calibrated_nlpd,
            row.raw_coverage_95,
            row.calibrated_coverage_95,
            row.raw_mean_abs_z,
            row.calibrated_mean_abs_z,
            row.relative_error
        )?;
    }
    Ok(())
}

fn write_topology_vs_naive_predictions(
    result: &PlanarHolesTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,kind,label,observed_value,truth_value,predictive_mean,predictive_variance,nlpd"
    )?;
    for row in &result.heldout_predictions {
        writeln!(
            writer,
            "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.scenario.as_str(),
            row.model.as_str(),
            row.kind.as_str(),
            row.label,
            row.observed_value,
            row.truth_value,
            row.predictive_mean,
            row.predictive_variance,
            row.nlpd
        )?;
    }
    Ok(())
}

fn write_topology_vs_naive_field_coverage(
    result: &PlanarHolesTopologyVsNaiveGpResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,subset,edge_count,weight_sum,coverage_95,mass_weighted_coverage_95,mean_abs_z,rms_z,p95_abs_z,mean_posterior_std,mass_weighted_mean_posterior_std,latent_nlpd"
    )?;
    for summary in &result.field_coverage_summaries {
        writeln!(
            writer,
            "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            summary.scenario.as_str(),
            summary.model.as_str(),
            summary.subset.as_str(),
            summary.edge_count,
            summary.weight_sum,
            summary.coverage_95,
            summary.mass_weighted_coverage_95,
            summary.mean_abs_z,
            summary.rms_z,
            summary.p95_abs_z,
            summary.mean_posterior_std,
            summary.mass_weighted_mean_posterior_std,
            summary.latent_nlpd
        )?;
    }
    Ok(())
}

fn validate_sensor_sweep_config(config: &PlanarHolesSensorDesignSweepConfig) -> io::Result<()> {
    if config.total_observation_budget == 0 {
        return Err(invalid_input("total_observation_budget must be positive"));
    }
    for (name, value) in [
        (
            "interior_loop_noise_variance",
            config.interior_loop_noise_variance,
        ),
        ("long_path_noise_variance", config.long_path_noise_variance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_input(format!("{name} must be finite and positive")));
        }
    }
    if config.designs.is_empty() {
        return Err(invalid_input("at least one sensor design is required"));
    }
    Ok(())
}

fn validate_topology_vs_naive_config(
    config: &PlanarHolesTopologyVsNaiveGpConfig,
) -> io::Result<()> {
    if config.total_observation_budget == 0 {
        return Err(invalid_input("total_observation_budget must be positive"));
    }
    for (name, count) in [
        ("validation_local_count", config.validation_local_count),
        (
            "validation_interior_loop_count",
            config.validation_interior_loop_count,
        ),
        (
            "validation_long_path_count",
            config.validation_long_path_count,
        ),
        ("heldout_local_count", config.heldout_local_count),
        (
            "heldout_interior_loop_count",
            config.heldout_interior_loop_count,
        ),
        ("heldout_long_path_count", config.heldout_long_path_count),
    ] {
        if count == 0 {
            return Err(invalid_input(format!("{name} must be positive")));
        }
    }
    for (name, value) in [
        (
            "interior_loop_noise_variance",
            config.interior_loop_noise_variance,
        ),
        ("long_path_noise_variance", config.long_path_noise_variance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_input(format!("{name} must be finite and positive")));
        }
    }
    validate_positive_grid("hodge_kappas", &config.hodge_kappas)?;
    validate_positive_grid("hodge_taus", &config.hodge_taus)?;
    validate_positive_grid("naive_kappas", &config.naive_kappas)?;
    validate_positive_grid("naive_taus", &config.naive_taus)?;
    validate_positive_grid(
        "observation_variance_scales",
        &config.observation_variance_scales,
    )?;
    Ok(())
}

fn validate_positive_grid(name: &str, values: &[f64]) -> io::Result<()> {
    if values.is_empty() {
        return Err(invalid_input(format!("{name} must not be empty")));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(invalid_input(format!(
            "{name} must contain only finite positive values"
        )));
    }
    Ok(())
}

fn sensor_design_counts(
    design: PlanarHolesSensorDesignKind,
    total: usize,
    harmonic_periods: usize,
) -> SensorDesignCounts {
    let clamp = |value: usize| value.min(total);
    match design {
        PlanarHolesSensorDesignKind::SparseEdges => SensorDesignCounts {
            edge_count: total,
            interior_loop_count: 0,
            long_path_count: 0,
            harmonic_period_count: 0,
        },
        PlanarHolesSensorDesignKind::EdgesHolePeriods => {
            let harmonic_period_count = harmonic_periods.min(total);
            SensorDesignCounts {
                edge_count: total - harmonic_period_count,
                interior_loop_count: 0,
                long_path_count: 0,
                harmonic_period_count,
            }
        }
        PlanarHolesSensorDesignKind::EdgesSmallInteriorLoops => {
            let interior_loop_count = clamp((total / 6).max(1));
            SensorDesignCounts {
                edge_count: total - interior_loop_count,
                interior_loop_count,
                long_path_count: 0,
                harmonic_period_count: 0,
            }
        }
        PlanarHolesSensorDesignKind::EdgesMultiscaleInteriorLoops => {
            let interior_loop_count = clamp((total / 4).max(1));
            SensorDesignCounts {
                edge_count: total - interior_loop_count,
                interior_loop_count,
                long_path_count: 0,
                harmonic_period_count: 0,
            }
        }
        PlanarHolesSensorDesignKind::EdgesLongPaths => {
            let long_path_count = clamp((total / 6).max(1));
            SensorDesignCounts {
                edge_count: total - long_path_count,
                interior_loop_count: 0,
                long_path_count,
                harmonic_period_count: 0,
            }
        }
        PlanarHolesSensorDesignKind::Hybrid => {
            let harmonic_period_count = harmonic_periods.min(total);
            let interior_loop_count = clamp(total / 5);
            let long_path_count = clamp(total / 8);
            let used = harmonic_period_count + interior_loop_count + long_path_count;
            SensorDesignCounts {
                edge_count: total.saturating_sub(used),
                interior_loop_count,
                long_path_count,
                harmonic_period_count,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sensor_design_sweep_row(
    design: PlanarHolesSensorDesignKind,
    counts: SensorDesignCounts,
    observation_count: usize,
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    truth: &TruthFields,
    conditioned: &ConditionedModel,
    predictions: &[PlanarHolesHeldoutPrediction],
    field_coverage_summaries: &[PlanarHolesFieldCoverageSummary],
) -> Result<PlanarHolesSensorDesignSweepRow, Box<dyn Error>> {
    let error = &conditioned.posterior_mean - &truth.mixed;
    let l2_error = relative_or_nan(
        mass_norm(&error, mass_1form),
        mass_norm(&truth.mixed, mass_1form),
    );
    let codifferential_leakage =
        codifferential_leakage(topology, metric, mass_1form, &conditioned.posterior_mean)?;
    let (coexact_relative_error, coexact_mass_correlation) =
        branch_recovery_pair(HodgeBranchKind::Coexact, mass_1form, truth, conditioned);
    let (harmonic_relative_error, harmonic_mass_correlation) =
        branch_recovery_pair(HodgeBranchKind::Harmonic, mass_1form, truth, conditioned);
    Ok(PlanarHolesSensorDesignSweepRow {
        design,
        model: conditioned.model,
        model_coexact_tau_scale: conditioned.coexact_tau_scale,
        observation_count,
        edge_observation_count: counts.edge_count,
        interior_loop_observation_count: counts.interior_loop_count,
        long_path_observation_count: counts.long_path_count,
        harmonic_period_observation_count: counts.harmonic_period_count,
        l2_error,
        heldout_local_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutLocal,
        ),
        heldout_interior_loop_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutInteriorLoop,
        ),
        heldout_long_path_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutLongPath,
        ),
        heldout_harmonic_period_nlpd: heldout_kind_mean_nlpd(
            predictions,
            PlanarHolesObservationKind::HeldoutHarmonicPeriod,
        ),
        heldout_interior_loop_relative_error: heldout_kind_relative_error(
            predictions,
            PlanarHolesObservationKind::HeldoutInteriorLoop,
        ),
        heldout_long_path_relative_error: heldout_kind_relative_error(
            predictions,
            PlanarHolesObservationKind::HeldoutLongPath,
        ),
        heldout_harmonic_period_relative_error: heldout_kind_relative_error(
            predictions,
            PlanarHolesObservationKind::HeldoutHarmonicPeriod,
        ),
        codifferential_leakage,
        coexact_relative_error,
        coexact_mass_correlation,
        harmonic_relative_error,
        harmonic_mass_correlation,
        all_edge_coverage_95: field_coverage_value(
            field_coverage_summaries,
            conditioned.model,
            PlanarHolesFieldCoverageSubset::AllEdges,
        ),
        heldout_edge_coverage_95: field_coverage_value(
            field_coverage_summaries,
            conditioned.model,
            PlanarHolesFieldCoverageSubset::HeldoutLocalEdges,
        ),
    })
}

fn field_coverage_value(
    summaries: &[PlanarHolesFieldCoverageSummary],
    model: PlanarHolesModelKind,
    subset: PlanarHolesFieldCoverageSubset,
) -> f64 {
    summaries
        .iter()
        .find(|summary| summary.model == model && summary.subset == subset)
        .map(|summary| summary.coverage_95)
        .unwrap_or(f64::NAN)
}

pub fn write_planar_holes_hodge_flow_outputs(
    result: &PlanarHolesFlowResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_topology_summary(result, out_dir.join("topology_summary.csv"))?;
    write_metrics_summary(result, out_dir.join("metrics_summary.csv"))?;
    write_period_summary(result, out_dir.join("period_summary.csv"))?;
    write_loop_functional_summary(result, out_dir.join("loop_functional_summary.csv"))?;
    write_field_coverage_summary(result, out_dir.join("field_coverage_summary.csv"))?;
    write_heldout_predictions(result, out_dir.join("heldout_predictions.csv"))?;
    write_branch_recovery_summary(result, out_dir.join("branch_recovery_summary.csv"))?;
    write_spectral_branch_diagnostics(result, out_dir.join("spectral_branch_diagnostics.csv"))?;
    write_spectral_energy_diagnostics(result, out_dir.join("spectral_energy_diagnostics.csv"))?;
    write_boundary_diagnostics(result, out_dir.join("boundary_diagnostics.csv"))?;
    write_branch_diagnostics(result, out_dir.join("branch_diagnostics.csv"))?;
    write_truth_outputs(result, out_dir)?;
    write_observation_outputs(result, out_dir)?;
    for scenario in &result.scenarios {
        write_scenario_outputs(result, scenario, out_dir)?;
    }
    Ok(())
}

fn validate_config(config: &PlanarHolesFlowConfig) -> io::Result<()> {
    for (name, value) in [
        ("mesh_size", config.mesh_size),
        ("exact_kappa", config.exact_kappa),
        ("coexact_kappa", config.coexact_kappa),
        ("nondecomposed_kappa", config.nondecomposed_kappa),
        ("component_kappa", config.component_kappa),
        ("tau", config.tau),
        ("exact_tau_scale", config.exact_tau_scale),
        ("coexact_tau_scale", config.coexact_tau_scale),
        ("harmonic_precision", config.harmonic_precision),
        ("local_noise_variance", config.local_noise_variance),
        ("loop_noise_variance", config.loop_noise_variance),
        (
            "heldout_loop_radius_offset",
            config.heldout_loop_radius_offset,
        ),
        (
            "heldout_loop_angle_offset",
            config.heldout_loop_angle_offset,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_input(format!("{name} must be finite and positive")));
        }
    }
    for (name, value) in [
        ("exact_truth_mass_norm", config.exact_truth_mass_norm),
        ("coexact_truth_mass_norm", config.coexact_truth_mass_norm),
        ("harmonic_truth_mass_norm", config.harmonic_truth_mass_norm),
        (
            "spectral_exact_expected_m1_energy",
            config.spectral_exact_expected_m1_energy,
        ),
        (
            "spectral_coexact_expected_m1_energy",
            config.spectral_coexact_expected_m1_energy,
        ),
        (
            "spectral_harmonic_expected_m1_energy",
            config.spectral_harmonic_expected_m1_energy,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid_input(format!(
                "{name} must be finite and nonnegative"
            )));
        }
    }
    if config.exact_truth_mass_norm
        + config.coexact_truth_mass_norm
        + config.harmonic_truth_mass_norm
        <= 0.0
    {
        return Err(invalid_input(
            "at least one truth branch mass norm must be positive",
        ));
    }
    if config.local_observation_count == 0 {
        return Err(invalid_input("local_observation_count must be positive"));
    }
    if config.heldout_local_count == 0 {
        return Err(invalid_input("heldout_local_count must be positive"));
    }
    if config.spectral_exact_mode_count == 0 {
        return Err(invalid_input("spectral_exact_mode_count must be positive"));
    }
    if config.spectral_coexact_mode_count == 0 {
        return Err(invalid_input(
            "spectral_coexact_mode_count must be positive",
        ));
    }
    if config.spectral_harmonic_mode_count == 0 {
        return Err(invalid_input(
            "spectral_harmonic_mode_count must be positive",
        ));
    }
    if config.spectral_harmonic_mode_count > HARMONIC_COEFFICIENTS.len() {
        return Err(invalid_input(format!(
            "spectral_harmonic_mode_count must be at most {}",
            HARMONIC_COEFFICIENTS.len()
        )));
    }
    if config.alpha != MaternAlpha::Two {
        return Err(invalid_input("planar holes v1 expects alpha=2"));
    }
    Ok(())
}

fn ensure_planar_holes_mesh(config: &PlanarHolesFlowConfig) -> Result<(), Box<dyn Error>> {
    ensure_planar_holes_mesh_with_holes(config, &default_holes())
}

fn ensure_planar_holes_mesh_with_holes(
    config: &PlanarHolesFlowConfig,
    holes: &[PlanarHoleSpec],
) -> Result<(), Box<dyn Error>> {
    if !config.force_mesh && config.mesh_path.is_file() {
        return Ok(());
    }
    if let Some(parent) = config.geo_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = config.mesh_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &config.geo_path,
        planar_holes_geo_for_holes(config.mesh_size, holes),
    )?;
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

#[allow(dead_code)]
fn planar_holes_geo(mesh_size: f64) -> String {
    planar_holes_geo_for_holes(mesh_size, &default_holes())
}

fn planar_holes_geo_for_holes(mesh_size: f64, holes: &[PlanarHoleSpec]) -> String {
    let mut geo = String::new();
    geo.push_str(&format!("lc = {:.12};\n", mesh_size));
    geo.push_str("Mesh.Algorithm = 6;\n");
    geo.push_str("Point(1) = {0, 0, 0, lc};\n");
    geo.push_str("Point(2) = {1, 0, 0, lc};\n");
    geo.push_str("Point(3) = {1, 1, 0, lc};\n");
    geo.push_str("Point(4) = {0, 1, 0, lc};\n");
    geo.push_str("Line(1) = {1, 2};\nLine(2) = {2, 3};\nLine(3) = {3, 4};\nLine(4) = {4, 1};\n");
    geo.push_str("Curve Loop(1) = {1, 2, 3, 4};\n");
    let mut next_point = 5;
    let mut next_curve = 5;
    for (index, hole) in holes.iter().enumerate() {
        let center = next_point;
        let east = next_point + 1;
        let north = next_point + 2;
        let west = next_point + 3;
        let south = next_point + 4;
        let loop_id = 2 + index;
        geo.push_str(&format!(
            "Point({center}) = {{{:.12}, {:.12}, 0, lc}};\n",
            hole.center[0], hole.center[1]
        ));
        geo.push_str(&format!(
            "Point({east}) = {{{:.12}, {:.12}, 0, lc}};\n",
            hole.center[0] + hole.radius,
            hole.center[1]
        ));
        geo.push_str(&format!(
            "Point({north}) = {{{:.12}, {:.12}, 0, lc}};\n",
            hole.center[0],
            hole.center[1] + hole.radius
        ));
        geo.push_str(&format!(
            "Point({west}) = {{{:.12}, {:.12}, 0, lc}};\n",
            hole.center[0] - hole.radius,
            hole.center[1]
        ));
        geo.push_str(&format!(
            "Point({south}) = {{{:.12}, {:.12}, 0, lc}};\n",
            hole.center[0],
            hole.center[1] - hole.radius
        ));
        geo.push_str(&format!(
            "Circle({next_curve}) = {{{east}, {center}, {north}}};\n"
        ));
        geo.push_str(&format!(
            "Circle({}) = {{{north}, {center}, {west}}};\n",
            next_curve + 1
        ));
        geo.push_str(&format!(
            "Circle({}) = {{{west}, {center}, {south}}};\n",
            next_curve + 2
        ));
        geo.push_str(&format!(
            "Circle({}) = {{{south}, {center}, {east}}};\n",
            next_curve + 3
        ));
        geo.push_str(&format!(
            "Curve Loop({loop_id}) = {{{}, {}, {}, {}}};\n",
            next_curve,
            next_curve + 1,
            next_curve + 2,
            next_curve + 3
        ));
        next_point += 5;
        next_curve += 4;
    }
    geo.push_str("Plane Surface(1) = {1, 2, 3, 4};\n");
    geo.push_str("Physical Surface(\"domain\") = {1};\n");
    geo
}

fn holes_for_domain(domain: PlanarHolesDomainKind) -> Vec<PlanarHoleSpec> {
    match domain {
        PlanarHolesDomainKind::Default => default_holes(),
        PlanarHolesDomainKind::Barrier => barrier_holes(),
    }
}

fn default_holes() -> Vec<PlanarHoleSpec> {
    vec![
        PlanarHoleSpec {
            name: "lower_left_island".to_string(),
            center: [0.30, 0.34],
            radius: 0.085,
        },
        PlanarHoleSpec {
            name: "upper_island".to_string(),
            center: [0.55, 0.68],
            radius: 0.095,
        },
        PlanarHoleSpec {
            name: "right_island".to_string(),
            center: [0.76, 0.36],
            radius: 0.075,
        },
    ]
}

fn barrier_holes() -> Vec<PlanarHoleSpec> {
    vec![
        PlanarHoleSpec {
            name: "lower_barrier_island".to_string(),
            center: [0.50, 0.28],
            radius: 0.095,
        },
        PlanarHoleSpec {
            name: "middle_barrier_island".to_string(),
            center: [0.50, 0.50],
            radius: 0.095,
        },
        PlanarHoleSpec {
            name: "upper_barrier_island".to_string(),
            center: [0.50, 0.72],
            radius: 0.095,
        },
    ]
}

fn summarize_topology(topology: &Complex) -> PlanarHolesTopologySummary {
    let vertex_count = topology.vertices().len();
    let edge_count = topology.edges().len();
    let face_count = topology.cells().len();
    let euler_characteristic = vertex_count as isize - edge_count as isize + face_count as isize;
    let b0 = connected_component_count(topology);
    let b2 = 0;
    let b1 = (b0 as isize + b2 as isize - euler_characteristic).max(0) as usize;
    PlanarHolesTopologySummary {
        vertex_count,
        edge_count,
        face_count,
        euler_characteristic,
        b0,
        b1,
        b2,
        boundary_edge_count: topology.boundary_subcomplex_simplices(1).len(),
    }
}

fn boundary_vertex_indices(topology: &Complex) -> BTreeSet<usize> {
    topology
        .boundary_subcomplex_simplices(0)
        .into_iter()
        .map(|simplex| simplex.kidx)
        .collect()
}

fn boundary_edge_indices(topology: &Complex) -> BTreeSet<usize> {
    topology
        .boundary_subcomplex_simplices(1)
        .into_iter()
        .map(|simplex| simplex.kidx)
        .collect()
}

fn connected_component_count(topology: &Complex) -> usize {
    let mut parent = (0..topology.vertices().len()).collect::<Vec<_>>();
    for edge in topology.edges().handle_iter() {
        union_vertices(&mut parent, edge.vertices[0], edge.vertices[1]);
    }
    (0..parent.len())
        .map(|vertex| find_vertex_root(&mut parent, vertex))
        .collect::<BTreeSet<_>>()
        .len()
}

fn find_vertex_root(parent: &mut [usize], vertex: usize) -> usize {
    if parent[vertex] != vertex {
        parent[vertex] = find_vertex_root(parent, parent[vertex]);
    }
    parent[vertex]
}

fn union_vertices(parent: &mut [usize], lhs: usize, rhs: usize) {
    let lhs_root = find_vertex_root(parent, lhs);
    let rhs_root = find_vertex_root(parent, rhs);
    if lhs_root != rhs_root {
        parent[rhs_root] = lhs_root;
    }
}

fn validate_topology(summary: &PlanarHolesTopologySummary) -> io::Result<()> {
    if summary.b0 != 1 || summary.b1 != 3 || summary.b2 != 0 {
        return Err(invalid_data(format!(
            "expected connected planar domain with three holes (b0=1,b1=3,b2=0), got b0={} b1={} b2={}",
            summary.b0, summary.b1, summary.b2
        )));
    }
    if summary.boundary_edge_count == 0 {
        return Err(invalid_data(
            "planar holes domain should have boundary edges",
        ));
    }
    Ok(())
}

fn canonicalize_harmonic_basis_by_cycles(
    harmonic_basis: &FeecMatrix,
    cycle_observation_matrix: &FeecCsr,
) -> Result<FeecMatrix, Box<dyn Error>> {
    let pairing = cycle_observation_matrix * harmonic_basis;
    if pairing.nrows() != pairing.ncols() {
        return Err(invalid_data(format!(
            "cycle-harmonic pairing must be square for canonicalization, got {}x{}",
            pairing.nrows(),
            pairing.ncols()
        ))
        .into());
    }
    let inverse = pairing.clone().try_inverse().ok_or_else(|| {
        invalid_data("cycle-harmonic pairing is singular and cannot define a canonical basis")
    })?;
    Ok(harmonic_basis * inverse)
}

fn build_hodge_prior(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    harmonic_basis: &FeecMatrix,
    config: &PlanarHolesFlowConfig,
) -> Result<SparseAnchorHodge1FormPrior, Box<dyn Error>> {
    build_hodge_prior_with_branches(
        topology,
        coords,
        metric,
        harmonic_basis,
        config,
        HodgeBranchKind::ALL.to_vec(),
    )
}

fn build_hodge_prior_with_branches(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    harmonic_basis: &FeecMatrix,
    config: &PlanarHolesFlowConfig,
    branches: Vec<HodgeBranchKind>,
) -> Result<SparseAnchorHodge1FormPrior, Box<dyn Error>> {
    build_sparse_anchor_hodge_1form_prior_with_coords(
        topology,
        coords,
        metric,
        SparseAnchorHodge1FormPriorConfig {
            branches,
            exact: SparseAnchorBranchConfig {
                kappa: config.exact_kappa,
                tau: config.tau * config.exact_tau_scale,
                alpha: config.alpha,
            },
            coexact: SparseAnchorBranchConfig {
                kappa: config.coexact_kappa,
                tau: config.tau * config.coexact_tau_scale,
                alpha: config.alpha,
            },
            harmonic_precision: config.harmonic_precision,
            harmonic_dim: Some(default_holes().len()),
            harmonic_basis_override: Some(harmonic_basis.clone()),
            ..SparseAnchorHodge1FormPriorConfig::default()
        },
    )
    .map_err(|err| invalid_data(err).into())
}

fn build_model_operators(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    hodge_prior: &SparseAnchorHodge1FormPrior,
    projection: &HodgeProjectionOperator,
    harmonic_basis: &FeecMatrix,
    config: &PlanarHolesFlowConfig,
) -> Result<Vec<ModelOperators>, Box<dyn Error>> {
    let mut models = Vec::new();
    models.push(ModelOperators {
        model: PlanarHolesModelKind::HodgeMatern,
        prior_precision: hodge_prior.precision.clone(),
        latent_to_ambient: hodge_prior.latent_to_ambient.clone(),
        branch_transforms: branch_transforms(hodge_prior),
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: config.coexact_tau_scale,
    });
    if config.include_exact_hodge_model {
        let exact_prior = build_hodge_prior_with_branches(
            topology,
            coords,
            metric,
            harmonic_basis,
            config,
            vec![HodgeBranchKind::Exact],
        )?;
        let branch_transforms = branch_transforms(&exact_prior);
        models.push(ModelOperators {
            model: PlanarHolesModelKind::ExactHodgeMatern,
            prior_precision: exact_prior.precision,
            latent_to_ambient: exact_prior.latent_to_ambient,
            branch_transforms,
            spectral_branch_stats: BTreeMap::new(),
            coexact_tau_scale: f64::NAN,
        });
    }
    if config.include_exact_dense_exact_hodge_model
        || config.include_exact_dense_trace_matched_exact_hodge_model
    {
        if config.include_exact_dense_exact_hodge_model {
            models.push(build_exact_dense_exact_model_operator(
                topology,
                metric,
                &hodge.mass_u,
                config,
                false,
            )?);
        }
        if config.include_exact_dense_trace_matched_exact_hodge_model {
            models.push(build_exact_dense_exact_model_operator(
                topology,
                metric,
                &hodge.mass_u,
                config,
                true,
            )?);
        }
    }
    if config.include_incompressible_hodge_model {
        let incompressible_prior = build_hodge_prior_with_branches(
            topology,
            coords,
            metric,
            harmonic_basis,
            config,
            vec![HodgeBranchKind::Coexact, HodgeBranchKind::Harmonic],
        )?;
        let branch_transforms = branch_transforms(&incompressible_prior);
        models.push(ModelOperators {
            model: PlanarHolesModelKind::IncompressibleHodgeMatern,
            prior_precision: incompressible_prior.precision,
            latent_to_ambient: incompressible_prior.latent_to_ambient,
            branch_transforms,
            spectral_branch_stats: BTreeMap::new(),
            coexact_tau_scale: config.coexact_tau_scale,
        });
    }
    if config.include_exact_mass_incompressible_hodge_model
        || config.include_sparse_lower_trace_matched_incompressible_hodge_model
    {
        if config.include_exact_mass_incompressible_hodge_model {
            models.push(build_exact_mass_incompressible_model_operator(
                topology,
                coords,
                metric,
                &hodge.mass_u,
                harmonic_basis,
                config,
                false,
            )?);
        }
        if config.include_sparse_lower_trace_matched_incompressible_hodge_model {
            models.push(build_exact_mass_incompressible_model_operator(
                topology,
                coords,
                metric,
                &hodge.mass_u,
                harmonic_basis,
                config,
                true,
            )?);
        }
    }
    if config.include_exact_lower_incompressible_hodge_model
        || config.include_exact_lower_trace_matched_incompressible_hodge_model
    {
        let exact_lower_mass_inverse =
            build_exact_dense_mass_inverse_1form(&hodge.mass_u, 1e-14).map_err(invalid_data)?;
        if config.include_exact_lower_incompressible_hodge_model {
            models.push(build_exact_lower_incompressible_model_operator(
                topology,
                coords,
                metric,
                &hodge.mass_u,
                &exact_lower_mass_inverse,
                harmonic_basis,
                config,
                false,
            )?);
        }
        if config.include_exact_lower_trace_matched_incompressible_hodge_model {
            models.push(build_exact_lower_incompressible_model_operator(
                topology,
                coords,
                metric,
                &hodge.mass_u,
                &exact_lower_mass_inverse,
                harmonic_basis,
                config,
                true,
            )?);
        }
    }
    if config.include_spectral_exact_hodge_model
        || config.include_spectral_hodge_model
        || config.include_spectral_incompressible_hodge_model
    {
        let spectral_basis =
            build_spectral_hodge_basis(topology, metric, harmonic_basis.ncols(), config)?;
        if config.include_spectral_exact_hodge_model {
            models.push(build_spectral_hodge_model_operator(
                spectral_basis.clone(),
                config,
                PlanarHolesModelKind::SpectralExactHodgeGp,
                &[HodgeBranchKind::Exact],
            )?);
        }
        if config.include_spectral_hodge_model {
            models.push(build_spectral_hodge_model_operator(
                spectral_basis.clone(),
                config,
                PlanarHolesModelKind::SpectralHodgeGp,
                &[
                    HodgeBranchKind::Exact,
                    HodgeBranchKind::Coexact,
                    HodgeBranchKind::Harmonic,
                ],
            )?);
        }
        if config.include_spectral_incompressible_hodge_model {
            models.push(build_spectral_hodge_model_operator(
                spectral_basis,
                config,
                PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
                &[HodgeBranchKind::Coexact, HodgeBranchKind::Harmonic],
            )?);
        }
    }

    let q1 = build_matern_precision_1form_for_alpha_with_coords(
        topology,
        coords,
        metric,
        hodge,
        config.alpha,
        Matern1FormConfig {
            kappa: config.nondecomposed_kappa,
            tau: config.tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    )
    .map_err(invalid_data)?;
    models.push(ModelOperators {
        model: PlanarHolesModelKind::NondecomposedFeec,
        prior_precision: q1,
        latent_to_ambient: identity_csr(topology.edges().len()),
        branch_transforms: BTreeMap::new(),
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: f64::NAN,
    });

    let q0 = build_matern_precision_0form(
        &build_laplace_beltrami_0form(topology, metric),
        Matern0FormConfig {
            kappa: config.component_kappa,
            tau: config.tau,
            mass_inverse: Matern0FormMassInverse::RowSumLumped,
        },
    );
    let component_precision = block_diag_feec_csr(&[&q0, &q0]);
    let edge_from_components = build_edge_component_operator(topology, coords);
    models.push(ModelOperators {
        model: PlanarHolesModelKind::ComponentwiseMatern,
        prior_precision: component_precision.clone(),
        latent_to_ambient: edge_from_components.clone(),
        branch_transforms: BTreeMap::new(),
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: f64::NAN,
    });
    models.push(ModelOperators {
        model: PlanarHolesModelKind::PostHocProjectedVectorMatern,
        prior_precision: component_precision,
        latent_to_ambient: &projection.total * &edge_from_components,
        branch_transforms: BTreeMap::new(),
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: f64::NAN,
    });
    if config.include_naive_euclidean_vector_matern_model {
        models.push(build_naive_euclidean_vector_matern_model_operator(
            topology,
            coords,
            config.component_kappa,
            config.tau,
        )?);
    }

    Ok(models)
}

fn build_nondecomposed_feec_model_operator(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    kappa: f64,
    tau: f64,
    alpha: MaternAlpha,
) -> Result<ModelOperators, Box<dyn Error>> {
    let q1 = build_matern_precision_1form_for_alpha_with_coords(
        topology,
        coords,
        metric,
        hodge,
        alpha,
        Matern1FormConfig {
            kappa,
            tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    )
    .map_err(invalid_data)?;
    Ok(ModelOperators {
        model: PlanarHolesModelKind::NondecomposedFeec,
        prior_precision: q1,
        latent_to_ambient: identity_csr(topology.edges().len()),
        branch_transforms: BTreeMap::new(),
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: f64::NAN,
    })
}

fn build_naive_euclidean_vector_matern_model_operator(
    topology: &Complex,
    coords: &MeshCoords,
    kappa: f64,
    tau: f64,
) -> Result<ModelOperators, Box<dyn Error>> {
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err(invalid_input("naive Euclidean kappa must be finite and positive").into());
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(invalid_input("naive Euclidean tau must be finite and positive").into());
    }
    let vertex_count = topology.vertices().len();
    let covariance_scale = 1.0 / (tau * tau);
    let mut covariance = FeecMatrix::zeros(vertex_count, vertex_count);
    for i in 0..vertex_count {
        let ci = coords.coord(i);
        for j in 0..=i {
            let cj = coords.coord(j);
            let r = (ci - cj).norm();
            // Hole-blind ambient Matern-3/2 covariance.  The only geometry used here
            // is straight Euclidean distance between vertex coordinates.
            let kr = kappa * r;
            let value = covariance_scale * (1.0 + kr) * (-kr).exp();
            covariance[(i, j)] = value;
            covariance[(j, i)] = value;
        }
    }
    let jitter = (1e-8 * covariance_scale).max(1e-12);
    for i in 0..vertex_count {
        covariance[(i, i)] += jitter;
    }
    let precision_dense = covariance.try_inverse().ok_or_else(|| {
        invalid_data(format!(
            "failed to invert naive Euclidean covariance for kappa={kappa:.6e}, tau={tau:.6e}"
        ))
    })?;
    let scalar_precision = dense_to_feec_csr(&precision_dense, 0.0);
    Ok(ModelOperators {
        model: PlanarHolesModelKind::NaiveEuclideanVectorMatern,
        prior_precision: block_diag_feec_csr(&[&scalar_precision, &scalar_precision]),
        latent_to_ambient: build_edge_component_operator(topology, coords),
        branch_transforms: BTreeMap::new(),
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: f64::NAN,
    })
}

fn build_exact_dense_exact_model_operator(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &PlanarHolesFlowConfig,
    trace_matched: bool,
) -> Result<ModelOperators, Box<dyn Error>> {
    let (raw_precision, raw_transform) =
        build_exact_dense_0form_exact_branch(topology, metric, config, 1.0)?;
    let trace_tau_scale = if trace_matched {
        let raw_energy = transformed_mass_expected_energy_from_precision(
            &raw_precision,
            &raw_transform,
            mass_1form,
        )
        .map_err(invalid_data)?;
        required_tau_scale(raw_energy, config.spectral_exact_expected_m1_energy)
    } else {
        1.0
    };
    if !trace_tau_scale.is_finite() || trace_tau_scale <= 0.0 {
        return Err(invalid_data(format!(
            "invalid exact dense trace-matched tau scale {trace_tau_scale}"
        ))
        .into());
    }
    let (precision, transform) = if trace_matched {
        build_exact_dense_0form_exact_branch(topology, metric, config, trace_tau_scale)?
    } else {
        (raw_precision, raw_transform)
    };
    let mut branch_transforms = BTreeMap::new();
    branch_transforms.insert(HodgeBranchKind::Exact, (0, transform.clone()));
    Ok(ModelOperators {
        model: if trace_matched {
            PlanarHolesModelKind::ExactDenseTraceMatchedExactHodgeMatern
        } else {
            PlanarHolesModelKind::ExactDenseExactHodgeMatern
        },
        prior_precision: precision,
        latent_to_ambient: transform,
        branch_transforms,
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: f64::NAN,
    })
}

fn build_exact_dense_0form_exact_branch(
    topology: &Complex,
    metric: &MeshLengths,
    config: &PlanarHolesFlowConfig,
    tau_scale: f64,
) -> Result<(FeecCsr, FeecCsr), Box<dyn Error>> {
    build_exact_branch_with_mass_inverse(topology, metric, config, None, tau_scale)
}

fn build_exact_branch_with_mass_inverse(
    topology: &Complex,
    metric: &MeshLengths,
    config: &PlanarHolesFlowConfig,
    mass_inverse_override: Option<&FeecCsr>,
    tau_scale: f64,
) -> Result<(FeecCsr, FeecCsr), Box<dyn Error>> {
    if !tau_scale.is_finite() || tau_scale <= 0.0 {
        return Err(invalid_data(format!("invalid exact tau scale {tau_scale}")).into());
    }
    let laplace = build_laplace_beltrami_0form(topology, metric);
    let system = build_matern_system_matrix_0form(&laplace, config.exact_kappa);
    let mass_inverse = match mass_inverse_override {
        Some(inverse) => inverse.clone(),
        None => build_exact_dense_mass_inverse_0form(&laplace.mass, 1e-14).map_err(invalid_data)?,
    };
    let precision_full = spectrum_matched_potential_precision(
        &system,
        &mass_inverse,
        config.alpha,
        config.exact_kappa,
        config.tau * config.exact_tau_scale * tau_scale,
    )
    .map_err(invalid_data)?;
    let transform_full = build_exact_1form_transform(topology);
    let (precision, transform) =
        anchor_exact_0form_branch(topology, &precision_full, &transform_full)?;
    Ok((precision, transform))
}

fn anchor_exact_0form_branch(
    topology: &Complex,
    precision_full: &FeecCsr,
    transform_full: &FeecCsr,
) -> Result<(FeecCsr, FeecCsr), Box<dyn Error>> {
    if precision_full.nrows() != precision_full.ncols() {
        return Err(invalid_data(format!(
            "exact branch source precision must be square, got {}x{}",
            precision_full.nrows(),
            precision_full.ncols()
        ))
        .into());
    }
    let anchors = connected_component_vertex_anchors_local(topology);
    if anchors.is_empty() {
        return Err(invalid_data("exact branch needs at least one vertex anchor").into());
    }
    let mut is_anchor = vec![false; precision_full.nrows()];
    for anchor in anchors {
        if anchor >= is_anchor.len() {
            return Err(invalid_data(format!(
                "exact branch anchor {anchor} is out of bounds for {} vertices",
                is_anchor.len()
            ))
            .into());
        }
        is_anchor[anchor] = true;
    }
    let kept = (0..precision_full.nrows())
        .filter(|index| !is_anchor[*index])
        .collect::<Vec<_>>();
    let mut selector = FeecCoo::new(precision_full.nrows(), kept.len());
    for (col, row) in kept.iter().copied().enumerate() {
        selector.push(row, col, 1.0);
    }
    let selector = FeecCsr::from(&selector);
    let middle = precision_full * &selector;
    let precision = selector.transpose() * &middle;
    let transform = transform_full * &selector;
    Ok((precision, transform))
}

fn connected_component_vertex_anchors_local(topology: &Complex) -> Vec<usize> {
    let mut parent = (0..topology.vertices().len()).collect::<Vec<_>>();
    for edge in topology.edges().handle_iter() {
        union_vertices(&mut parent, edge.vertices[0], edge.vertices[1]);
    }
    let mut anchors = BTreeMap::<usize, usize>::new();
    for vertex in 0..topology.vertices().len() {
        let root = find_vertex_root(&mut parent, vertex);
        anchors
            .entry(root)
            .and_modify(|anchor| *anchor = (*anchor).min(vertex))
            .or_insert(vertex);
    }
    anchors.into_values().collect()
}

fn build_exact_mass_incompressible_model_operator(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    harmonic_basis: &FeecMatrix,
    config: &PlanarHolesFlowConfig,
    trace_matched: bool,
) -> Result<ModelOperators, Box<dyn Error>> {
    let (raw_precision, raw_transform) =
        build_sparse_lower_coexact_branch(topology, coords, metric, mass_1form, config, 1.0)?;
    let trace_tau_scale = if trace_matched {
        let raw_energy = transformed_mass_expected_energy_from_precision(
            &raw_precision,
            &raw_transform,
            mass_1form,
        )
        .map_err(invalid_data)?;
        required_tau_scale(raw_energy, config.spectral_coexact_expected_m1_energy)
    } else {
        1.0
    };
    if !trace_tau_scale.is_finite() || trace_tau_scale <= 0.0 {
        return Err(invalid_data(format!(
            "invalid sparse-lower trace-matched tau scale {trace_tau_scale}"
        ))
        .into());
    }
    let (coexact_precision, coexact_transform) = if trace_matched {
        build_sparse_lower_coexact_branch(
            topology,
            coords,
            metric,
            mass_1form,
            config,
            trace_tau_scale,
        )?
    } else {
        (raw_precision, raw_transform)
    };
    let harmonic_transform = dense_to_feec_csr(harmonic_basis, 0.0);
    let harmonic_precision = scale_matrix(
        &identity_csr(harmonic_basis.ncols()),
        config.harmonic_precision,
    );
    let coexact_latent_dim = coexact_transform.ncols();
    let precision = block_diag_feec_csr(&[&coexact_precision, &harmonic_precision]);
    let latent_to_ambient =
        hstack_feec_csr(&[&coexact_transform, &harmonic_transform]).map_err(invalid_data)?;
    let mut branch_transforms = BTreeMap::new();
    branch_transforms.insert(HodgeBranchKind::Coexact, (0, coexact_transform));
    branch_transforms.insert(
        HodgeBranchKind::Harmonic,
        (coexact_latent_dim, harmonic_transform),
    );
    Ok(ModelOperators {
        model: if trace_matched {
            PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
        } else {
            PlanarHolesModelKind::ExactMassIncompressibleHodgeMatern
        },
        prior_precision: precision,
        latent_to_ambient,
        branch_transforms,
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: config.coexact_tau_scale * trace_tau_scale,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_sparse_lower_coexact_branch(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &PlanarHolesFlowConfig,
    tau_scale: f64,
) -> Result<(FeecCsr, FeecCsr), Box<dyn Error>> {
    let coexact_transform = build_exact_mass_coexact_1form_transform(topology, metric, mass_1form)
        .map_err(invalid_data)?;
    let hodge_2form = build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
        topology,
        coords,
        metric,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
    )
    .map_err(invalid_data)?;
    let coexact_system = build_matern_system_matrix_2form(&hodge_2form, config.coexact_kappa);
    let coexact_mass_inverse = build_matern_mass_inverse_2form_with_coords(
        topology,
        coords,
        metric,
        &hodge_2form.mass_u,
        Matern2FormMassInverse::default(),
    )
    .map_err(invalid_data)?;
    let coexact_precision_full = spectrum_matched_potential_precision(
        &coexact_system,
        &coexact_mass_inverse,
        config.alpha,
        config.coexact_kappa,
        config.tau * config.coexact_tau_scale * tau_scale,
    )
    .map_err(invalid_data)?;
    let (coexact_precision, coexact_transform, _) = anchor_hodge_potential_branch(
        topology,
        metric,
        2,
        &coexact_precision_full,
        &coexact_transform,
        HodgeBranchKind::Coexact,
    )
    .map_err(invalid_data)?;
    Ok((coexact_precision, coexact_transform))
}

#[allow(clippy::too_many_arguments)]
fn build_exact_lower_incompressible_model_operator(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    exact_lower_mass_inverse: &FeecCsr,
    harmonic_basis: &FeecMatrix,
    config: &PlanarHolesFlowConfig,
    trace_matched: bool,
) -> Result<ModelOperators, Box<dyn Error>> {
    let (raw_precision, raw_transform) = build_exact_lower_coexact_branch(
        topology,
        coords,
        metric,
        mass_1form,
        exact_lower_mass_inverse,
        config,
        1.0,
    )?;
    let trace_tau_scale = if trace_matched {
        let raw_energy = transformed_mass_expected_energy_from_precision(
            &raw_precision,
            &raw_transform,
            mass_1form,
        )
        .map_err(invalid_data)?;
        required_tau_scale(raw_energy, config.spectral_coexact_expected_m1_energy)
    } else {
        1.0
    };
    if !trace_tau_scale.is_finite() || trace_tau_scale <= 0.0 {
        return Err(invalid_data(format!(
            "invalid exact-lower trace-matched tau scale {trace_tau_scale}"
        ))
        .into());
    }
    let (coexact_precision, coexact_transform) = if trace_matched {
        build_exact_lower_coexact_branch(
            topology,
            coords,
            metric,
            mass_1form,
            exact_lower_mass_inverse,
            config,
            trace_tau_scale,
        )?
    } else {
        (raw_precision, raw_transform)
    };
    let harmonic_transform = dense_to_feec_csr(harmonic_basis, 0.0);
    let harmonic_precision = scale_matrix(
        &identity_csr(harmonic_basis.ncols()),
        config.harmonic_precision,
    );
    let coexact_latent_dim = coexact_transform.ncols();
    let precision = block_diag_feec_csr(&[&coexact_precision, &harmonic_precision]);
    let latent_to_ambient =
        hstack_feec_csr(&[&coexact_transform, &harmonic_transform]).map_err(invalid_data)?;
    let mut branch_transforms = BTreeMap::new();
    branch_transforms.insert(HodgeBranchKind::Coexact, (0, coexact_transform));
    branch_transforms.insert(
        HodgeBranchKind::Harmonic,
        (coexact_latent_dim, harmonic_transform),
    );
    Ok(ModelOperators {
        model: if trace_matched {
            PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
        } else {
            PlanarHolesModelKind::ExactLowerIncompressibleHodgeMatern
        },
        prior_precision: precision,
        latent_to_ambient,
        branch_transforms,
        spectral_branch_stats: BTreeMap::new(),
        coexact_tau_scale: config.coexact_tau_scale * trace_tau_scale,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_exact_lower_coexact_branch(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    exact_lower_mass_inverse: &FeecCsr,
    config: &PlanarHolesFlowConfig,
    tau_scale: f64,
) -> Result<(FeecCsr, FeecCsr), Box<dyn Error>> {
    let coexact_transform = build_exact_mass_coexact_1form_transform(topology, metric, mass_1form)
        .map_err(invalid_data)?;
    let hodge_2form = build_hodge_laplacian_2form_with_lower_mass_inverse_matrix(
        topology,
        metric,
        exact_lower_mass_inverse,
    )
    .map_err(invalid_data)?;
    let coexact_system = build_matern_system_matrix_2form(&hodge_2form, config.coexact_kappa);
    let coexact_mass_inverse = build_matern_mass_inverse_2form_with_coords(
        topology,
        coords,
        metric,
        &hodge_2form.mass_u,
        Matern2FormMassInverse::default(),
    )
    .map_err(invalid_data)?;
    let coexact_precision_full = spectrum_matched_potential_precision(
        &coexact_system,
        &coexact_mass_inverse,
        config.alpha,
        config.coexact_kappa,
        config.tau * config.coexact_tau_scale * tau_scale,
    )
    .map_err(invalid_data)?;
    let (coexact_precision, coexact_transform, _) = anchor_hodge_potential_branch(
        topology,
        metric,
        2,
        &coexact_precision_full,
        &coexact_transform,
        HodgeBranchKind::Coexact,
    )
    .map_err(invalid_data)?;
    Ok((coexact_precision, coexact_transform))
}

fn build_spectral_hodge_basis(
    topology: &Complex,
    metric: &MeshLengths,
    harmonic_dim: usize,
    config: &PlanarHolesFlowConfig,
) -> Result<HodgeDecomposedBasis, Box<dyn Error>> {
    let mut options = HodgeBuildOptions {
        harmonic_dim,
        exact_mode_count: config.spectral_exact_mode_count,
        coexact_mode_count: config.spectral_coexact_mode_count,
        ..HodgeBuildOptions::new(harmonic_dim)
    };
    options.boundary =
        spectral_boundary_condition_spec(topology, config.spectral_boundary_condition);
    HodgeDecomposedBasis::build(topology, metric, 1, options)
        .map_err(|err| invalid_data(err.to_string()).into())
}

fn spectral_boundary_condition_spec(
    topology: &Complex,
    boundary_condition: PlanarHolesSpectralBoundaryCondition,
) -> Option<BoundaryConditionSpec> {
    match boundary_condition {
        PlanarHolesSpectralBoundaryCondition::Free => None,
        PlanarHolesSpectralBoundaryCondition::StrongBoundaryOneForms => {
            Some(BoundaryConditionSpec::new().with_strong_dofs(1, boundary_edge_indices(topology)))
        }
    }
}

fn build_spectral_hodge_model_operator(
    basis: HodgeDecomposedBasis,
    config: &PlanarHolesFlowConfig,
    model: PlanarHolesModelKind,
    branches: &[HodgeBranchKind],
) -> Result<ModelOperators, Box<dyn Error>> {
    let exact_modes = if branches.contains(&HodgeBranchKind::Exact) {
        config.spectral_exact_mode_count
    } else {
        0
    };
    let coexact_modes = if branches.contains(&HodgeBranchKind::Coexact) {
        config.spectral_coexact_mode_count
    } else {
        0
    };
    let harmonic_modes = if branches.contains(&HodgeBranchKind::Harmonic) {
        config.spectral_harmonic_mode_count
    } else {
        0
    };
    let gp = HodgeCompositionalGp::from_hodge_decomposition(
        basis,
        HodgeCompositionalConfig {
            alpha: config.alpha.as_u32() as f64,
            exact: spectral_branch_config(
                config.exact_kappa,
                config.tau * config.exact_tau_scale,
                exact_modes,
                config.spectral_exact_expected_m1_energy,
                config,
            ),
            coexact: spectral_branch_config(
                config.coexact_kappa,
                config.tau * config.coexact_tau_scale,
                coexact_modes,
                config.spectral_coexact_expected_m1_energy,
                config,
            ),
            harmonic: spectral_branch_config(
                1.0,
                config.harmonic_precision.sqrt(),
                harmonic_modes,
                config.spectral_harmonic_expected_m1_energy,
                config,
            ),
        },
    )
    .map_err(|err| invalid_data(err.to_string()))?;
    let latent_to_ambient = faer_mat_to_feec_csr(gp.combined_feature_matrix(), 1e-13);
    let prior_precision = identity_csr(latent_to_ambient.ncols());
    let mut branch_transforms = BTreeMap::new();
    let mut spectral_branch_stats = BTreeMap::new();
    let mut offset = 0usize;
    for branch in [
        HodgeBranchKind::Exact,
        HodgeBranchKind::Coexact,
        HodgeBranchKind::Harmonic,
    ] {
        let features = gp.branch_feature_matrix(branch);
        let cols = features.ncols();
        if branches.contains(&branch) && cols > 0 {
            branch_transforms.insert(branch, (offset, faer_mat_to_feec_csr(features, 1e-13)));
            spectral_branch_stats.insert(branch, gp.branch_feature_stats(branch));
        }
        offset += cols;
    }

    Ok(ModelOperators {
        model,
        prior_precision,
        latent_to_ambient,
        branch_transforms,
        spectral_branch_stats,
        coexact_tau_scale: config.coexact_tau_scale,
    })
}

fn spectral_branch_config(
    kappa: f64,
    tau: f64,
    mode_count: usize,
    expected_m1_energy: f64,
    config: &PlanarHolesFlowConfig,
) -> SpectralHodgeBranchConfig {
    let mut branch = SpectralHodgeBranchConfig {
        kappa,
        tau,
        mode_count,
        energy_normalization: HodgeBranchEnergyNormalization::None,
    };
    if config.spectral_branch_energy_normalization {
        branch.energy_normalization =
            HodgeBranchEnergyNormalization::ExpectedMassEnergy(expected_m1_energy);
    }
    branch
}

fn faer_mat_to_feec_csr(mat: &Mat<f64>, drop_tolerance: f64) -> FeecCsr {
    let dense = FeecMatrix::from_fn(mat.nrows(), mat.ncols(), |row, col| mat[(row, col)]);
    dense_to_feec_csr(&dense, drop_tolerance)
}

fn compute_spectral_diagnostics(
    models: &[ModelOperators],
    truth: &TruthFields,
    mass_1form: &FeecCsr,
) -> Result<
    (
        Vec<PlanarHolesSpectralBranchDiagnostic>,
        Vec<PlanarHolesSpectralEnergyDiagnostic>,
    ),
    Box<dyn Error>,
> {
    let mut branch_rows = Vec::new();
    let mut energy_rows = Vec::new();
    for model in models {
        if model.spectral_branch_stats.is_empty() {
            continue;
        }
        for branch in [
            HodgeBranchKind::Exact,
            HodgeBranchKind::Coexact,
            HodgeBranchKind::Harmonic,
        ] {
            let Some((_, transform)) = model.branch_transforms.get(&branch) else {
                continue;
            };
            let Some(stats) = model.spectral_branch_stats.get(&branch).copied() else {
                continue;
            };
            let truth_branch = truth_branch(truth, branch);
            let (branch_diagnostic, mut branch_energy_rows) = spectral_projection_diagnostic(
                model.model,
                branch,
                transform,
                stats,
                truth_branch,
                mass_1form,
            )?;
            branch_rows.push(branch_diagnostic);
            energy_rows.append(&mut branch_energy_rows);
        }
    }
    Ok((branch_rows, energy_rows))
}

fn truth_branch(truth: &TruthFields, branch: HodgeBranchKind) -> &FeecVector {
    match branch {
        HodgeBranchKind::Exact => &truth.exact,
        HodgeBranchKind::Coexact => &truth.coexact,
        HodgeBranchKind::Harmonic => &truth.harmonic,
    }
}

fn spectral_projection_diagnostic(
    model: PlanarHolesModelKind,
    branch: HodgeBranchKind,
    transform: &FeecCsr,
    stats: HodgeBranchFeatureStats,
    truth_branch: &FeecVector,
    mass_1form: &FeecCsr,
) -> Result<
    (
        PlanarHolesSpectralBranchDiagnostic,
        Vec<PlanarHolesSpectralEnergyDiagnostic>,
    ),
    Box<dyn Error>,
> {
    if transform.nrows() != truth_branch.len() {
        return Err(invalid_data(format!(
            "spectral transform rows {} do not match truth length {}",
            transform.nrows(),
            truth_branch.len()
        ))
        .into());
    }
    let truth_mass_norm = mass_norm(truth_branch, mass_1form);
    let truth_energy = truth_mass_norm * truth_mass_norm;
    let mut projection = FeecVector::zeros(truth_branch.len());
    let mut cumulative_projected_energy = 0.0;
    let mut cumulative_prior_energy = 0.0;
    let mut projected_truth_mahalanobis_sq = 0.0;
    let mut energy_rows = Vec::with_capacity(transform.ncols());
    for col in 0..transform.ncols() {
        let feature = sparse_column_to_feec_vector(transform, col);
        let feature_energy = bilinear_form_sparse(mass_1form, &feature, &feature).max(0.0);
        cumulative_prior_energy += feature_energy;
        if feature_energy > EPS {
            let inner = bilinear_form_sparse(mass_1form, &feature, truth_branch);
            let latent = inner / feature_energy;
            projection += feature.scale(latent);
            cumulative_projected_energy += inner * inner / feature_energy;
            projected_truth_mahalanobis_sq += latent * latent;
        }
        energy_rows.push(PlanarHolesSpectralEnergyDiagnostic {
            model,
            branch,
            mode_index: col,
            cumulative_projected_energy,
            cumulative_projected_energy_fraction: relative_or_nan(
                cumulative_projected_energy,
                truth_energy,
            ),
            cumulative_prior_energy,
            cumulative_prior_energy_fraction: relative_or_nan(
                cumulative_prior_energy,
                stats.expected_m1_energy,
            ),
        });
    }
    let projected_truth_mass_norm = mass_norm(&projection, mass_1form);
    let error = &projection - truth_branch;
    Ok((
        PlanarHolesSpectralBranchDiagnostic {
            model,
            branch,
            requested_mode_count: stats.requested_mode_count,
            actual_mode_count: stats.actual_mode_count,
            unnormalized_expected_m1_energy: stats.unnormalized_expected_m1_energy,
            target_expected_m1_energy: stats.target_expected_m1_energy.unwrap_or(f64::NAN),
            normalization_scale: stats.normalization_scale,
            expected_m1_energy: stats.expected_m1_energy,
            truth_mass_norm,
            projected_truth_mass_norm,
            projection_relative_error: relative_or_nan(
                mass_norm(&error, mass_1form),
                truth_mass_norm,
            ),
            projected_truth_mahalanobis_norm: projected_truth_mahalanobis_sq.sqrt(),
        },
        energy_rows,
    ))
}

fn sparse_column_to_feec_vector(matrix: &FeecCsr, column: usize) -> FeecVector {
    let mut vector = FeecVector::zeros(matrix.nrows());
    for (row, col, value) in matrix.triplet_iter() {
        if col == column {
            vector[row] += *value;
        }
    }
    vector
}

fn branch_transforms(
    prior: &SparseAnchorHodge1FormPrior,
) -> BTreeMap<HodgeBranchKind, (usize, FeecCsr)> {
    prior
        .branches
        .iter()
        .map(|branch| (branch.kind, (branch.offset, branch.transform.clone())))
        .collect()
}

struct TruthFields {
    exact: FeecVector,
    coexact: FeecVector,
    harmonic: FeecVector,
    mixed: FeecVector,
}

fn build_truth(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    hodge_prior: &SparseAnchorHodge1FormPrior,
    config: &PlanarHolesFlowConfig,
) -> Result<TruthFields, Box<dyn Error>> {
    let exact_prior = build_sparse_anchor_hodge_1form_prior_with_coords(
        topology,
        coords,
        metric,
        SparseAnchorHodge1FormPriorConfig {
            branches: vec![HodgeBranchKind::Exact],
            exact: SparseAnchorBranchConfig {
                kappa: config.exact_kappa,
                tau: config.tau * config.exact_tau_scale,
                alpha: config.alpha,
            },
            ..SparseAnchorHodge1FormPriorConfig::default()
        },
    )
    .map_err(invalid_data)?;
    let coexact_prior = build_sparse_anchor_hodge_1form_prior_with_coords(
        topology,
        coords,
        metric,
        SparseAnchorHodge1FormPriorConfig {
            branches: vec![HodgeBranchKind::Coexact],
            coexact: SparseAnchorBranchConfig {
                kappa: config.coexact_kappa,
                tau: config.tau * config.coexact_tau_scale,
                alpha: config.alpha,
            },
            ..SparseAnchorHodge1FormPriorConfig::default()
        },
    )
    .map_err(invalid_data)?;
    let spectral_truth_branches = if config.exact_truth_source
        == PlanarHolesExactTruthSource::SpectralGp
        || config.coexact_truth_source == PlanarHolesCoexactTruthSource::SpectralGp
        || config.harmonic_truth_source == PlanarHolesHarmonicTruthSource::SpectralGp
    {
        Some(sample_spectral_truth_branches(
            topology,
            metric,
            hodge_prior.harmonic_basis.ncols(),
            config,
        )?)
    } else {
        None
    };

    let exact = if config.exact_truth_mass_norm <= 0.0 {
        FeecVector::zeros(mass_1form.nrows())
    } else {
        let exact_sample = match config.exact_truth_source {
            PlanarHolesExactTruthSource::SparseAnchor => {
                sample_prior_ambient(&exact_prior, config.rng_seed + 101)?
            }
            PlanarHolesExactTruthSource::ExactDenseGmrf => {
                sample_exact_dense_exact_truth(topology, metric, mass_1form, config)?
            }
            PlanarHolesExactTruthSource::SpectralGp => spectral_truth_branches
                .as_ref()
                .and_then(|branches| branches.get(&HodgeBranchKind::Exact))
                .cloned()
                .ok_or_else(|| invalid_data("spectral exact truth branch was not sampled"))?,
            PlanarHolesExactTruthSource::AnalyticPotential => {
                sample_analytic_exact_potential_truth(topology, coords)
            }
        };
        scale_truth_branch(
            &exact_sample,
            mass_1form,
            config.exact_truth_mass_norm,
            config.truth_scaling,
        )?
    };
    let coexact = if config.coexact_truth_mass_norm <= 0.0 {
        FeecVector::zeros(mass_1form.nrows())
    } else {
        let coexact_sample = match config.coexact_truth_source {
            PlanarHolesCoexactTruthSource::SparseAnchor => {
                sample_prior_ambient(&coexact_prior, config.rng_seed + 202)?
            }
            PlanarHolesCoexactTruthSource::ExactMassCoexact => {
                sample_exact_mass_coexact_truth(topology, coords, metric, mass_1form, config)?
            }
            PlanarHolesCoexactTruthSource::ExactLowerMassCoexact => {
                sample_exact_lower_mass_coexact_truth(topology, coords, metric, mass_1form, config)?
            }
            PlanarHolesCoexactTruthSource::SpectralGp => spectral_truth_branches
                .as_ref()
                .and_then(|branches| branches.get(&HodgeBranchKind::Coexact))
                .cloned()
                .ok_or_else(|| invalid_data("spectral coexact truth branch was not sampled"))?,
            PlanarHolesCoexactTruthSource::DirichletStreamfunction => {
                sample_dirichlet_streamfunction_truth(
                    topology,
                    coords,
                    metric,
                    mass_1form,
                    &hodge_prior.harmonic_basis,
                    config,
                )?
            }
        };
        scale_truth_branch(
            &coexact_sample,
            mass_1form,
            config.coexact_truth_mass_norm,
            config.truth_scaling,
        )?
    };
    let harmonic = if config.harmonic_truth_mass_norm <= 0.0 {
        FeecVector::zeros(mass_1form.nrows())
    } else {
        let harmonic_sample = match config.harmonic_truth_source {
            PlanarHolesHarmonicTruthSource::CanonicalFixed => {
                let coeffs = FeecVector::from_column_slice(&HARMONIC_COEFFICIENTS);
                &hodge_prior.harmonic_basis * coeffs
            }
            PlanarHolesHarmonicTruthSource::SpectralGp => spectral_truth_branches
                .as_ref()
                .and_then(|branches| branches.get(&HodgeBranchKind::Harmonic))
                .cloned()
                .ok_or_else(|| invalid_data("spectral harmonic truth branch was not sampled"))?,
        };
        scale_truth_branch(
            &harmonic_sample,
            mass_1form,
            config.harmonic_truth_mass_norm,
            config.truth_scaling,
        )?
    };
    let mixed = &(&exact + &coexact) + &harmonic;
    if matches!(
        config.coexact_truth_source,
        PlanarHolesCoexactTruthSource::ExactMassCoexact
            | PlanarHolesCoexactTruthSource::DirichletStreamfunction
    ) && config.exact_truth_mass_norm == 0.0
    {
        validate_discrete_incompressible_truth(topology, metric, mass_1form, &mixed)?;
    }

    Ok(TruthFields {
        exact,
        coexact,
        harmonic,
        mixed,
    })
}

fn sample_spectral_truth_branches(
    topology: &Complex,
    metric: &MeshLengths,
    harmonic_dim: usize,
    config: &PlanarHolesFlowConfig,
) -> Result<BTreeMap<HodgeBranchKind, FeecVector>, Box<dyn Error>> {
    let basis = build_spectral_hodge_basis(topology, metric, harmonic_dim, config)?;
    let gp = HodgeCompositionalGp::from_hodge_decomposition(
        basis,
        HodgeCompositionalConfig {
            alpha: config.alpha.as_u32() as f64,
            exact: spectral_branch_config(
                config.exact_kappa,
                config.tau * config.exact_tau_scale,
                config.spectral_exact_mode_count,
                config.spectral_exact_expected_m1_energy,
                config,
            ),
            coexact: spectral_branch_config(
                config.coexact_kappa,
                config.tau * config.coexact_tau_scale,
                config.spectral_coexact_mode_count,
                config.spectral_coexact_expected_m1_energy,
                config,
            ),
            harmonic: spectral_branch_config(
                1.0,
                config.harmonic_precision.sqrt(),
                config.spectral_harmonic_mode_count,
                config.spectral_harmonic_expected_m1_energy,
                config,
            ),
        },
    )
    .map_err(|err| invalid_data(err.to_string()))?;
    let mut branches = BTreeMap::new();
    for (branch, seed) in [
        (HodgeBranchKind::Exact, config.rng_seed + 1101),
        (HodgeBranchKind::Coexact, config.rng_seed + 1202),
        (HodgeBranchKind::Harmonic, config.rng_seed + 1303),
    ] {
        branches.insert(
            branch,
            sample_spectral_feature_matrix(gp.branch_feature_matrix(branch), seed),
        );
    }
    Ok(branches)
}

fn sample_spectral_feature_matrix(features: &Mat<f64>, seed: u64) -> FeecVector {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut sample = FeecVector::zeros(features.nrows());
    for col in 0..features.ncols() {
        let z = rng.sample::<f64, _>(StandardNormal);
        for row in 0..features.nrows() {
            sample[row] += features[(row, col)] * z;
        }
    }
    sample
}

fn sample_exact_dense_exact_truth(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &PlanarHolesFlowConfig,
) -> Result<FeecVector, Box<dyn Error>> {
    let (raw_precision, raw_transform) =
        build_exact_dense_0form_exact_branch(topology, metric, config, 1.0)?;
    let trace_tau_scale = if config.spectral_branch_energy_normalization {
        let raw_energy = transformed_mass_expected_energy_from_precision(
            &raw_precision,
            &raw_transform,
            mass_1form,
        )
        .map_err(invalid_data)?;
        required_tau_scale(raw_energy, config.spectral_exact_expected_m1_energy)
    } else {
        1.0
    };
    let (precision, transform) = if config.spectral_branch_energy_normalization {
        build_exact_dense_0form_exact_branch(topology, metric, config, trace_tau_scale)?
    } else {
        (raw_precision, raw_transform)
    };
    let source = sample_zero_mean_precision(&precision, config.rng_seed + 1501)?;
    Ok(&transform * &source)
}

fn sample_analytic_exact_potential_truth(topology: &Complex, coords: &MeshCoords) -> FeecVector {
    let mut values = FeecVector::zeros(topology.vertices().len());
    for vertex in topology.vertices().handle_iter() {
        let index = vertex.kidx();
        let point = coords.coord(index);
        let x = point[0];
        let y = point[1];
        values[index] = (TWO_PI * x).sin() * (std::f64::consts::PI * y).cos()
            + 0.35 * (3.0 * std::f64::consts::PI * x + 0.2).cos() * (TWO_PI * y).sin()
            + 0.20 * (std::f64::consts::PI * (x + y)).sin();
    }
    let transform = build_exact_1form_transform(topology);
    &transform * &values
}

fn sample_exact_mass_coexact_truth(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &PlanarHolesFlowConfig,
) -> Result<FeecVector, Box<dyn Error>> {
    let hodge_2form = build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
        topology,
        coords,
        metric,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
    )
    .map_err(invalid_data)?;
    let precision = build_matern_precision_2form_for_alpha_with_coords(
        topology,
        coords,
        metric,
        &hodge_2form,
        config.alpha,
        Matern2FormConfig {
            kappa: config.coexact_kappa,
            tau: config.tau * config.coexact_tau_scale,
            mass_inverse: Matern2FormMassInverse::default(),
        },
    )
    .map_err(invalid_data)?;
    let source = sample_zero_mean_precision(&precision, config.rng_seed + 202)?;
    let transform = build_exact_mass_coexact_1form_transform(topology, metric, mass_1form)
        .map_err(invalid_data)?;
    Ok(&transform * &source)
}

fn sample_exact_lower_mass_coexact_truth(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    config: &PlanarHolesFlowConfig,
) -> Result<FeecVector, Box<dyn Error>> {
    let exact_lower_mass_inverse =
        build_exact_dense_mass_inverse_1form(mass_1form, 1e-14).map_err(invalid_data)?;
    let (raw_precision, raw_transform) = build_exact_lower_coexact_branch(
        topology,
        coords,
        metric,
        mass_1form,
        &exact_lower_mass_inverse,
        config,
        1.0,
    )?;
    let trace_tau_scale = if config.spectral_branch_energy_normalization {
        let raw_energy = transformed_mass_expected_energy_from_precision(
            &raw_precision,
            &raw_transform,
            mass_1form,
        )
        .map_err(invalid_data)?;
        required_tau_scale(raw_energy, config.spectral_coexact_expected_m1_energy)
    } else {
        1.0
    };
    let (precision, transform) = if config.spectral_branch_energy_normalization {
        build_exact_lower_coexact_branch(
            topology,
            coords,
            metric,
            mass_1form,
            &exact_lower_mass_inverse,
            config,
            trace_tau_scale,
        )?
    } else {
        (raw_precision, raw_transform)
    };
    let source = sample_zero_mean_precision(&precision, config.rng_seed + 2602)?;
    Ok(&transform * &source)
}

fn sample_dirichlet_streamfunction_truth(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    _harmonic_basis: &FeecMatrix,
    config: &PlanarHolesFlowConfig,
) -> Result<FeecVector, Box<dyn Error>> {
    let boundary_vertices = boundary_vertex_indices(topology);
    let holes = default_holes();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.rng_seed + 2402);
    let modes = [
        (1usize, 1usize),
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (1, 3),
        (3, 2),
        (2, 3),
    ];
    let coefficients = modes
        .iter()
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect::<Vec<_>>();
    let mut psi = vec![0.0; topology.vertices().len()];
    for vertex in topology.vertices().handle_iter() {
        let index = vertex.kidx();
        if boundary_vertices.contains(&index) {
            continue;
        }
        let point = coords.coord(index);
        let x = point[0];
        let y = point[1];
        let outer_envelope = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
        let hole_envelope = holes.iter().fold(1.0_f64, |acc, hole| {
            let dx = x - hole.center[0];
            let dy = y - hole.center[1];
            let signed_distance = (dx * dx + dy * dy).sqrt() - hole.radius;
            acc * smoothstep((signed_distance / 0.10).clamp(0.0, 1.0))
        });
        let value = modes
            .iter()
            .zip(coefficients.iter())
            .map(|((kx, ky), coefficient)| {
                coefficient
                    * (std::f64::consts::PI * *kx as f64 * x).sin()
                    * (std::f64::consts::PI * *ky as f64 * y).sin()
            })
            .sum::<f64>();
        psi[index] = outer_envelope * hole_envelope * value;
    }

    let raw = streamfunction_edge_integrals(topology, coords, &psi)?;
    let transform = build_exact_mass_coexact_1form_transform(topology, metric, mass_1form)
        .map_err(invalid_data)?;
    let projection =
        build_mass_projection_operator_1form(&transform, mass_1form, 1e-10, "streamfunction")
            .map_err(invalid_data)?;
    Ok(&projection * &raw)
}

fn smoothstep(value: f64) -> f64 {
    value * value * (3.0 - 2.0 * value)
}

fn streamfunction_edge_integrals(
    topology: &Complex,
    coords: &MeshCoords,
    psi: &[f64],
) -> Result<FeecVector, Box<dyn Error>> {
    let edge_lookup = edge_lookup(topology);
    let edge_count = topology.edges().len();
    let mut values = vec![0.0; edge_count];
    let mut weights = vec![0.0; edge_count];
    for cell in topology.cells().handle_iter() {
        let vertices = &cell.vertices;
        if vertices.len() != 3 {
            continue;
        }
        let v0 = vertices[0];
        let v1 = vertices[1];
        let v2 = vertices[2];
        let p0 = coords.coord(v0);
        let p1 = coords.coord(v1);
        let p2 = coords.coord(v2);
        let x0 = p0[0];
        let y0 = p0[1];
        let x1 = p1[0];
        let y1 = p1[1];
        let x2 = p2[0];
        let y2 = p2[1];
        let twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if twice_area.abs() <= EPS {
            continue;
        }
        let area = 0.5 * twice_area.abs();
        let grad0 = [(y1 - y2) / twice_area, (x2 - x1) / twice_area];
        let grad1 = [(y2 - y0) / twice_area, (x0 - x2) / twice_area];
        let grad2 = [(y0 - y1) / twice_area, (x1 - x0) / twice_area];
        let grad_x = psi[v0] * grad0[0] + psi[v1] * grad1[0] + psi[v2] * grad2[0];
        let grad_y = psi[v0] * grad0[1] + psi[v1] * grad1[1] + psi[v2] * grad2[1];
        let ux = -grad_y;
        let uy = grad_x;
        for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
            let Some((edge_index, edge_a, edge_b)) = edge_lookup.get(&(a.min(b), a.max(b))) else {
                return Err(invalid_data(format!("triangle edge ({a},{b}) is missing")).into());
            };
            let ca = coords.coord(*edge_a);
            let cb = coords.coord(*edge_b);
            let integral = ux * (cb[0] - ca[0]) + uy * (cb[1] - ca[1]);
            values[*edge_index] += area * integral;
            weights[*edge_index] += area;
        }
    }
    for edge in 0..edge_count {
        if weights[edge] > EPS {
            values[edge] /= weights[edge];
        }
    }
    Ok(FeecVector::from_vec(values))
}

fn edge_lookup(topology: &Complex) -> BTreeMap<(usize, usize), (usize, usize, usize)> {
    let mut lookup = BTreeMap::new();
    for edge in topology.edges().handle_iter() {
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        lookup.insert((a.min(b), a.max(b)), (edge.kidx(), a, b));
    }
    lookup
}

fn validate_discrete_incompressible_truth(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    truth: &FeecVector,
) -> Result<(), Box<dyn Error>> {
    let truth_norm = mass_norm(truth, mass_1form);
    let mass_0form = de_rham::mass_matrix_form(topology, metric, 0).map_err(invalid_data)?;
    let truth_delta = de_rham::codifferential(topology, metric, 1, truth).map_err(invalid_data)?;
    let leakage = relative_or_nan(mass_norm(&truth_delta, &mass_0form), truth_norm);
    if !leakage.is_finite() || leakage > INCOMPRESSIBLE_TRUTH_CODIFFERENTIAL_LEAKAGE_TOL {
        return Err(invalid_data(format!(
            "exact-mass incompressible truth has codifferential leakage {leakage:.3e}, tolerance {:.3e}",
            INCOMPRESSIBLE_TRUTH_CODIFFERENTIAL_LEAKAGE_TOL
        ))
        .into());
    }
    Ok(())
}

fn sample_prior_ambient(
    prior: &SparseAnchorHodge1FormPrior,
    seed: u64,
) -> Result<FeecVector, Box<dyn Error>> {
    let sample = sample_zero_mean_precision(&prior.precision, seed)?;
    Ok(&prior.latent_to_ambient * &sample)
}

fn scale_truth_branch(
    field: &FeecVector,
    mass: &FeecCsr,
    target_norm: f64,
    scaling: PlanarHolesTruthScaling,
) -> Result<FeecVector, Box<dyn Error>> {
    if target_norm == 0.0 {
        return Ok(FeecVector::zeros(field.len()));
    }
    match scaling {
        PlanarHolesTruthScaling::MassNormTargets => scale_to_mass_norm(field, mass, target_norm),
        PlanarHolesTruthScaling::RawPriorSamples => Ok(field.clone()),
    }
}

fn sample_zero_mean_precision(
    precision_matrix: &FeecCsr,
    seed: u64,
) -> Result<FeecVector, Box<dyn Error>> {
    let precision = feec_csr_to_gmrf(precision_matrix);
    let factor = precision.cholesky_sqrt_lower_with_ordering(CholeskyOrdering::Identity)?;
    let mut gmrf = Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision)?
        .with_precision_sqrt(factor);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Ok(gmrf_vec_to_feec(&gmrf.sample(&mut rng)?))
}

fn scale_to_mass_norm(
    field: &FeecVector,
    mass: &FeecCsr,
    target_norm: f64,
) -> Result<FeecVector, Box<dyn Error>> {
    if target_norm == 0.0 {
        return Ok(FeecVector::zeros(field.len()));
    }
    let norm = mass_norm(field, mass);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid_data("cannot normalize a zero or non-finite field").into());
    }
    Ok(field * (target_norm / norm))
}

fn run_scenario(
    scenario: PlanarHolesObservationScenario,
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    truth: &TruthFields,
    train_cycle_observation_matrix: &FeecCsr,
    heldout_cycle_observation_matrix: &FeecCsr,
    cycles: &[PlanarHoleCycle],
    training: &[PlanarHolesObservation],
    heldout: &[PlanarHolesObservation],
    model_operators: &[ModelOperators],
    compute_field_coverage: bool,
) -> Result<PlanarHolesScenarioResult, Box<dyn Error>> {
    let (training_matrix, training_values) =
        scaled_observation_system(training, topology.edges().len());
    let heldout_matrix = observation_matrix(heldout, topology.edges().len(), false);

    let mut metrics = Vec::new();
    let mut period_summaries = Vec::new();
    let mut loop_functional_summaries = Vec::new();
    let mut field_coverage_summaries = Vec::new();
    let mut branch_recovery_summaries = Vec::new();
    let mut heldout_predictions = Vec::new();
    let mut posteriors = Vec::new();

    for operators in model_operators {
        let conditioned =
            condition_model(operators, &training_matrix, &training_values).map_err(|err| {
                invalid_data(format!(
                    "conditioning model {} failed: {err}",
                    operators.model.as_str()
                ))
            })?;
        let model_metrics = compute_metrics(
            scenario,
            conditioned.model,
            topology,
            metric,
            mass_1form,
            truth,
            train_cycle_observation_matrix,
            heldout_cycle_observation_matrix,
            &heldout_matrix,
            heldout,
            training.len(),
            &conditioned,
        )
        .map_err(|err| {
            invalid_data(format!(
                "metrics for model {} failed: {err}",
                conditioned.model.as_str()
            ))
        })?;
        period_summaries.extend(
            compute_period_summaries(
                scenario,
                conditioned.model,
                cycles,
                &truth.mixed,
                heldout_cycle_observation_matrix,
                &conditioned,
            )
            .map_err(|err| {
                invalid_data(format!(
                    "period summaries for model {} failed: {err}",
                    conditioned.model.as_str()
                ))
            })?,
        );
        loop_functional_summaries.extend(
            compute_loop_functional_summaries(
                scenario,
                conditioned.model,
                cycles,
                truth,
                train_cycle_observation_matrix,
                heldout_cycle_observation_matrix,
                &conditioned,
            )
            .map_err(|err| {
                invalid_data(format!(
                    "loop functional summaries for model {} failed: {err}",
                    conditioned.model.as_str()
                ))
            })?,
        );
        heldout_predictions.extend(
            compute_heldout_predictions(
                scenario,
                conditioned.model,
                heldout,
                &heldout_matrix,
                &conditioned,
            )
            .map_err(|err| {
                invalid_data(format!(
                    "heldout predictions for model {} failed: {err}",
                    conditioned.model.as_str()
                ))
            })?,
        );
        let field_coverage = if compute_field_coverage {
            compute_field_coverage_diagnostics(
                scenario,
                conditioned.model,
                mass_1form,
                &truth.mixed,
                heldout,
                &conditioned,
            )
            .map_err(|err| {
                invalid_data(format!(
                    "field coverage for model {} failed: {err}",
                    conditioned.model.as_str()
                ))
            })?
        } else {
            empty_field_coverage_diagnostics(
                scenario,
                conditioned.model,
                mass_1form,
                &truth.mixed,
                heldout,
                &conditioned.posterior_mean,
            )
        };
        field_coverage_summaries.extend(field_coverage.summaries);
        branch_recovery_summaries.extend(compute_branch_recovery_summaries(
            scenario,
            conditioned.model,
            mass_1form,
            truth,
            &conditioned,
        ));
        posteriors.push(PlanarHolesModelPosterior {
            scenario,
            model: conditioned.model,
            posterior_mean: conditioned.posterior_mean,
            branch_means: conditioned.branch_means,
            field_coverage: field_coverage.diagnostics,
        });
        metrics.push(model_metrics);
    }

    Ok(PlanarHolesScenarioResult {
        scenario,
        observation_count: training.len(),
        metrics,
        period_summaries,
        loop_functional_summaries,
        field_coverage_summaries,
        branch_recovery_summaries,
        heldout_predictions,
        posteriors,
    })
}

fn condition_model(
    operators: &ModelOperators,
    scaled_observation_matrix: &FeecCsr,
    scaled_observations: &FeecVector,
) -> Result<ConditionedModel, Box<dyn Error>> {
    let latent_observation = scaled_observation_matrix * &operators.latent_to_ambient;
    let (posterior_precision, information) = apply_gaussian_observations(
        &feec_csr_to_gmrf(&operators.prior_precision),
        &feec_csr_to_gmrf(&latent_observation),
        &feec_vec_to_gmrf(scaled_observations),
        None,
        1.0,
    );
    let posterior = Gmrf::from_information_and_precision(information, posterior_precision.clone())?;
    let latent_mean = gmrf_vec_to_feec(posterior.mean());
    let posterior_mean = &operators.latent_to_ambient * &latent_mean;
    let branch_means = operators
        .branch_transforms
        .iter()
        .map(|(kind, (offset, transform))| {
            let latent = FeecVector::from_vec(
                latent_mean.as_slice()[*offset..*offset + transform.ncols()].to_vec(),
            );
            (*kind, transform * &latent)
        })
        .collect();

    Ok(ConditionedModel {
        model: operators.model,
        coexact_tau_scale: operators.coexact_tau_scale,
        posterior_precision,
        latent_to_ambient: operators.latent_to_ambient.clone(),
        posterior_mean,
        branch_means,
        branch_transforms: operators.branch_transforms.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_metrics(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    truth: &TruthFields,
    train_cycle_observation_matrix: &FeecCsr,
    heldout_cycle_observation_matrix: &FeecCsr,
    heldout_matrix: &FeecCsr,
    heldout: &[PlanarHolesObservation],
    observation_count: usize,
    conditioned: &ConditionedModel,
) -> Result<PlanarHolesModelMetrics, Box<dyn Error>> {
    let error = &conditioned.posterior_mean - &truth.mixed;
    let truth_l2_norm = mass_norm(&truth.mixed, mass_1form);
    let l2_error_absolute = mass_norm(&error, mass_1form);
    let l2_error = relative_or_nan(l2_error_absolute, truth_l2_norm);

    let posterior_d = de_rham::derivative(topology, 1, &conditioned.posterior_mean);
    let truth_d = de_rham::derivative(topology, 1, &truth.mixed);
    let d_error = &posterior_d - &truth_d;
    let mass_2form = de_rham::mass_matrix_form(topology, metric, 2).map_err(invalid_data)?;
    let exterior_derivative_error_absolute = mass_norm(&d_error, &mass_2form);
    let exterior_derivative_truth_norm = mass_norm(&truth_d, &mass_2form);
    let exterior_derivative_error = relative_or_nan(
        exterior_derivative_error_absolute,
        exterior_derivative_truth_norm,
    );

    let posterior_delta = de_rham::codifferential(topology, metric, 1, &conditioned.posterior_mean)
        .map_err(invalid_data)?;
    let truth_delta =
        de_rham::codifferential(topology, metric, 1, &truth.mixed).map_err(invalid_data)?;
    let delta_error = &posterior_delta - &truth_delta;
    let mass_0form = de_rham::mass_matrix_form(topology, metric, 0).map_err(invalid_data)?;
    let codifferential_error_absolute = mass_norm(&delta_error, &mass_0form);
    let codifferential_truth_norm = mass_norm(&truth_delta, &mass_0form);
    let codifferential_error_relative =
        relative_or_nan(codifferential_error_absolute, codifferential_truth_norm);
    let codifferential_leakage =
        relative_or_nan(mass_norm(&posterior_delta, &mass_0form), truth_l2_norm);
    let (codifferential_error, codifferential_metric_kind) =
        if codifferential_truth_norm > RELATIVE_DENOM_EPS * truth_l2_norm.max(1.0) {
            (
                codifferential_error_relative,
                PlanarHolesCodifferentialMetricKind::RelativeError,
            )
        } else {
            (
                codifferential_leakage,
                PlanarHolesCodifferentialMetricKind::Leakage,
            )
        };

    let truth_periods = heldout_cycle_observation_matrix * &truth.mixed;
    let posterior_periods = heldout_cycle_observation_matrix * &conditioned.posterior_mean;
    let period_errors = &posterior_periods - &truth_periods;
    let relative_circulation_error = relative_or_nan(period_errors.norm(), truth_periods.norm());
    let annular_observation_matrix = difference_csr(
        heldout_cycle_observation_matrix,
        train_cycle_observation_matrix,
    )?;
    let truth_annular = &annular_observation_matrix * &truth.mixed;
    let posterior_annular = &annular_observation_matrix * &conditioned.posterior_mean;
    let relative_total_annular_error = relative_or_nan(
        (&posterior_annular - &truth_annular).norm(),
        truth_annular.norm(),
    );
    let relative_harmonic_period_error = branch_relative_error(
        train_cycle_observation_matrix,
        &truth.harmonic,
        HodgeBranchKind::Harmonic,
        conditioned,
    );
    let relative_coexact_annular_error = branch_relative_error(
        &annular_observation_matrix,
        &truth.coexact,
        HodgeBranchKind::Coexact,
        conditioned,
    );
    let heldout_predictions =
        compute_heldout_predictions(scenario, model, heldout, heldout_matrix, conditioned)?;
    let heldout_nlpd = mean(heldout_predictions.iter().map(|row| row.nlpd));
    let heldout_local_nlpd = mean(
        heldout_predictions
            .iter()
            .filter(|row| row.kind == PlanarHolesObservationKind::HeldoutLocal)
            .map(|row| row.nlpd),
    );
    let heldout_loop_nlpd = mean(
        heldout_predictions
            .iter()
            .filter(|row| row.kind == PlanarHolesObservationKind::HeldoutLoop)
            .map(|row| row.nlpd),
    );
    let heldout_harmonic_period_nlpd = mean(
        heldout_predictions
            .iter()
            .filter(|row| row.kind == PlanarHolesObservationKind::HeldoutHarmonicPeriod)
            .map(|row| row.nlpd),
    );

    Ok(PlanarHolesModelMetrics {
        scenario,
        model,
        observation_count,
        l2_error_absolute,
        l2_error,
        heldout_nlpd,
        heldout_local_nlpd,
        heldout_loop_nlpd,
        heldout_harmonic_period_nlpd,
        exterior_derivative_error_absolute,
        exterior_derivative_truth_norm,
        exterior_derivative_error,
        codifferential_error_absolute,
        codifferential_truth_norm,
        codifferential_error_relative,
        codifferential_leakage,
        codifferential_error,
        codifferential_metric_kind,
        relative_circulation_error,
        relative_harmonic_period_error,
        relative_coexact_annular_error,
        relative_total_annular_error,
        mean_abs_circulation_error: mean(period_errors.iter().map(|value| value.abs())),
        max_abs_circulation_error: period_errors
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max),
    })
}

fn branch_relative_error(
    observation_matrix: &FeecCsr,
    truth_branch: &FeecVector,
    branch: HodgeBranchKind,
    conditioned: &ConditionedModel,
) -> f64 {
    conditioned
        .branch_means
        .get(&branch)
        .map(|posterior_branch| {
            let truth_values = observation_matrix * truth_branch;
            let posterior_values = observation_matrix * posterior_branch;
            relative_or_nan(
                (&posterior_values - &truth_values).norm(),
                truth_values.norm(),
            )
        })
        .unwrap_or(f64::NAN)
}

fn branch_recovery_pair(
    branch: HodgeBranchKind,
    mass_1form: &FeecCsr,
    truth: &TruthFields,
    conditioned: &ConditionedModel,
) -> (f64, f64) {
    let Some(posterior_branch) = conditioned.branch_means.get(&branch) else {
        return (f64::NAN, f64::NAN);
    };
    let truth_branch = truth_branch_field(truth, branch);
    let truth_norm = mass_norm(truth_branch, mass_1form);
    let posterior_norm = mass_norm(posterior_branch, mass_1form);
    let error = posterior_branch - truth_branch;
    let relative_error = relative_or_nan(mass_norm(&error, mass_1form), truth_norm);
    let correlation = if truth_norm > RELATIVE_DENOM_EPS && posterior_norm > RELATIVE_DENOM_EPS {
        bilinear_form_sparse(mass_1form, truth_branch, posterior_branch)
            / (truth_norm * posterior_norm)
    } else {
        f64::NAN
    };
    (relative_error, correlation)
}

fn heldout_kind_mean_nlpd(
    predictions: &[PlanarHolesHeldoutPrediction],
    kind: PlanarHolesObservationKind,
) -> f64 {
    mean(
        predictions
            .iter()
            .filter(|prediction| prediction.kind == kind)
            .map(|prediction| prediction.nlpd),
    )
}

fn heldout_kind_relative_error(
    predictions: &[PlanarHolesHeldoutPrediction],
    kind: PlanarHolesObservationKind,
) -> f64 {
    let truth = predictions
        .iter()
        .filter(|prediction| prediction.kind == kind)
        .map(|prediction| prediction.truth_value)
        .collect::<Vec<_>>();
    let residual = predictions
        .iter()
        .filter(|prediction| prediction.kind == kind)
        .map(|prediction| prediction.predictive_mean - prediction.truth_value)
        .collect::<Vec<_>>();
    relative_or_nan(
        residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt(),
        truth.iter().map(|value| value * value).sum::<f64>().sqrt(),
    )
}

#[derive(Debug, Clone, Copy)]
struct PredictionKindStats {
    count: usize,
    mean_nlpd: f64,
    relative_error: f64,
    coverage_95: f64,
    mean_abs_z: f64,
}

fn prediction_kind_stats(
    predictions: &[PlanarHolesHeldoutPrediction],
    kind: PlanarHolesObservationKind,
) -> PredictionKindStats {
    prediction_kind_stats_with_variance_multiplier(predictions, kind, 1.0)
}

fn prediction_kind_stats_with_variance_multiplier(
    predictions: &[PlanarHolesHeldoutPrediction],
    kind: PlanarHolesObservationKind,
    variance_multiplier: f64,
) -> PredictionKindStats {
    let subset = predictions
        .iter()
        .filter(|prediction| prediction.kind == kind)
        .collect::<Vec<_>>();
    prediction_subset_stats(&subset, variance_multiplier)
}

fn prediction_subset_stats(
    subset: &[&PlanarHolesHeldoutPrediction],
    variance_multiplier: f64,
) -> PredictionKindStats {
    let count = subset.len();
    if count == 0 {
        return PredictionKindStats {
            count: 0,
            mean_nlpd: f64::NAN,
            relative_error: f64::NAN,
            coverage_95: f64::NAN,
            mean_abs_z: f64::NAN,
        };
    }
    let variance_multiplier = variance_multiplier.clamp(1e-3, 1e3);
    let residuals = subset
        .iter()
        .map(|prediction| prediction.predictive_mean - prediction.truth_value)
        .collect::<Vec<_>>();
    let truth_norm = subset
        .iter()
        .map(|prediction| prediction.truth_value * prediction.truth_value)
        .sum::<f64>()
        .sqrt();
    let residual_norm = residuals
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let z_scores = subset
        .iter()
        .zip(residuals.iter())
        .map(|(prediction, residual)| {
            residual.abs()
                / (prediction.predictive_variance.max(FIELD_VARIANCE_FLOOR) * variance_multiplier)
                    .sqrt()
        })
        .collect::<Vec<_>>();
    PredictionKindStats {
        count,
        mean_nlpd: mean(subset.iter().map(|prediction| {
            let variance =
                prediction.predictive_variance.max(FIELD_VARIANCE_FLOOR) * variance_multiplier;
            let residual = prediction.observed_value - prediction.predictive_mean;
            0.5 * ((TWO_PI * variance).ln() + residual * residual / variance)
        })),
        relative_error: relative_or_nan(residual_norm, truth_norm),
        coverage_95: mean(z_scores.iter().map(|z| (*z <= PERIOD_Z) as usize as f64)),
        mean_abs_z: mean(z_scores.iter().copied()),
    }
}

fn fit_predictive_variance_multiplier(predictions: &[PlanarHolesHeldoutPrediction]) -> f64 {
    if predictions.is_empty() {
        return 1.0;
    }
    let multiplier = mean(predictions.iter().map(|prediction| {
        let residual = prediction.observed_value - prediction.predictive_mean;
        residual * residual / prediction.predictive_variance.max(FIELD_VARIANCE_FLOOR)
    }));
    if multiplier.is_finite() {
        multiplier.clamp(1e-3, 1e3)
    } else {
        1.0
    }
}

fn calibration_rows_for_predictions(
    model: PlanarHolesModelKind,
    variance_multiplier: f64,
    predictions: &[PlanarHolesHeldoutPrediction],
) -> Vec<PlanarHolesTopologyVsNaiveGpCalibrationRow> {
    let mut rows = Vec::new();
    let all = predictions.iter().collect::<Vec<_>>();
    rows.push(calibration_row_from_subset(
        model,
        "all".to_string(),
        variance_multiplier,
        &all,
    ));
    let kinds = predictions
        .iter()
        .map(|prediction| prediction.kind)
        .collect::<BTreeSet<_>>();
    for kind in kinds {
        let subset = predictions
            .iter()
            .filter(|prediction| prediction.kind == kind)
            .collect::<Vec<_>>();
        rows.push(calibration_row_from_subset(
            model,
            kind.as_str().to_string(),
            variance_multiplier,
            &subset,
        ));
    }
    rows
}

fn calibration_multiplier_for_model(
    rows: &[PlanarHolesTopologyVsNaiveGpCalibrationRow],
    model: PlanarHolesModelKind,
) -> f64 {
    rows.iter()
        .find(|row| row.model == model && row.kind == "all")
        .map(|row| row.variance_multiplier)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0)
}

fn calibration_row_from_subset(
    model: PlanarHolesModelKind,
    kind: String,
    variance_multiplier: f64,
    subset: &[&PlanarHolesHeldoutPrediction],
) -> PlanarHolesTopologyVsNaiveGpCalibrationRow {
    let raw = prediction_subset_stats(subset, 1.0);
    let calibrated = prediction_subset_stats(subset, variance_multiplier);
    PlanarHolesTopologyVsNaiveGpCalibrationRow {
        model,
        kind,
        count: raw.count,
        variance_multiplier,
        raw_nlpd: raw.mean_nlpd,
        calibrated_nlpd: calibrated.mean_nlpd,
        raw_coverage_95: raw.coverage_95,
        calibrated_coverage_95: calibrated.coverage_95,
        raw_mean_abs_z: raw.mean_abs_z,
        calibrated_mean_abs_z: calibrated.mean_abs_z,
        relative_error: raw.relative_error,
    }
}

fn compute_branch_recovery_summaries(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    mass_1form: &FeecCsr,
    truth: &TruthFields,
    conditioned: &ConditionedModel,
) -> Vec<PlanarHolesBranchRecoverySummary> {
    conditioned
        .branch_means
        .iter()
        .map(|(branch, posterior)| {
            let truth_branch = truth_branch_field(truth, *branch);
            let error = posterior - truth_branch;
            let truth_mass_norm = mass_norm(truth_branch, mass_1form);
            let posterior_mass_norm = mass_norm(posterior, mass_1form);
            let error_mass_norm = mass_norm(&error, mass_1form);
            let mass_correlation = if truth_mass_norm > RELATIVE_DENOM_EPS
                && posterior_mass_norm > RELATIVE_DENOM_EPS
            {
                bilinear_form_sparse(mass_1form, truth_branch, posterior)
                    / (truth_mass_norm * posterior_mass_norm)
            } else {
                f64::NAN
            };
            PlanarHolesBranchRecoverySummary {
                scenario,
                model,
                branch: *branch,
                truth_mass_norm,
                posterior_mass_norm,
                error_mass_norm,
                relative_error: relative_or_nan(error_mass_norm, truth_mass_norm),
                mass_correlation,
            }
        })
        .collect()
}

fn truth_branch_field(truth: &TruthFields, branch: HodgeBranchKind) -> &FeecVector {
    match branch {
        HodgeBranchKind::Exact => &truth.exact,
        HodgeBranchKind::Coexact => &truth.coexact,
        HodgeBranchKind::Harmonic => &truth.harmonic,
    }
}

fn compute_loop_functional_summaries(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    cycles: &[PlanarHoleCycle],
    truth: &TruthFields,
    train_cycle_observation_matrix: &FeecCsr,
    heldout_cycle_observation_matrix: &FeecCsr,
    conditioned: &ConditionedModel,
) -> Result<Vec<PlanarHolesLoopFunctionalSummary>, Box<dyn Error>> {
    let annular_observation_matrix = difference_csr(
        heldout_cycle_observation_matrix,
        train_cycle_observation_matrix,
    )?;
    let mut rows = Vec::new();

    rows.extend(loop_functional_rows_for_total(
        scenario,
        model,
        PlanarHolesLoopFunctionalKind::TotalAnnular,
        cycles,
        &annular_observation_matrix,
        &truth.mixed,
        conditioned,
    )?);
    rows.extend(loop_functional_rows_for_branch(
        scenario,
        model,
        PlanarHolesLoopFunctionalKind::HarmonicPeriod,
        cycles,
        train_cycle_observation_matrix,
        &truth.harmonic,
        HodgeBranchKind::Harmonic,
        conditioned,
    )?);
    rows.extend(loop_functional_rows_for_branch(
        scenario,
        model,
        PlanarHolesLoopFunctionalKind::CoexactAnnular,
        cycles,
        &annular_observation_matrix,
        &truth.coexact,
        HodgeBranchKind::Coexact,
        conditioned,
    )?);

    Ok(rows)
}

fn loop_functional_rows_for_total(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    functional: PlanarHolesLoopFunctionalKind,
    cycles: &[PlanarHoleCycle],
    observation_matrix: &FeecCsr,
    truth: &FeecVector,
    conditioned: &ConditionedModel,
) -> Result<Vec<PlanarHolesLoopFunctionalSummary>, Box<dyn Error>> {
    let operator = observation_matrix * &conditioned.latent_to_ambient;
    let variances = exact_transformed_variances(&conditioned.posterior_precision, &operator)?;
    let truth_values = observation_matrix * truth;
    let posterior_means = observation_matrix * &conditioned.posterior_mean;
    Ok(loop_functional_rows_from_values(
        scenario,
        model,
        functional,
        cycles,
        &truth_values,
        &posterior_means,
        &variances,
    ))
}

#[allow(clippy::too_many_arguments)]
fn loop_functional_rows_for_branch(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    functional: PlanarHolesLoopFunctionalKind,
    cycles: &[PlanarHoleCycle],
    observation_matrix: &FeecCsr,
    truth: &FeecVector,
    branch: HodgeBranchKind,
    conditioned: &ConditionedModel,
) -> Result<Vec<PlanarHolesLoopFunctionalSummary>, Box<dyn Error>> {
    let Some(branch_mean) = conditioned.branch_means.get(&branch) else {
        return Ok(Vec::new());
    };
    let Some((offset, branch_transform)) = conditioned.branch_transforms.get(&branch) else {
        return Ok(Vec::new());
    };
    let operator = shifted_branch_observation_operator(
        observation_matrix,
        branch_transform,
        *offset,
        conditioned.posterior_precision.nrows(),
    )?;
    let variances = exact_transformed_variances(&conditioned.posterior_precision, &operator)?;
    let truth_values = observation_matrix * truth;
    let posterior_means = observation_matrix * branch_mean;
    Ok(loop_functional_rows_from_values(
        scenario,
        model,
        functional,
        cycles,
        &truth_values,
        &posterior_means,
        &variances,
    ))
}

fn loop_functional_rows_from_values(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    functional: PlanarHolesLoopFunctionalKind,
    cycles: &[PlanarHoleCycle],
    truth_values: &FeecVector,
    posterior_means: &FeecVector,
    variances: &FeecVector,
) -> Vec<PlanarHolesLoopFunctionalSummary> {
    cycles
        .iter()
        .enumerate()
        .map(|(index, cycle)| {
            let std = variances[index].max(0.0).sqrt();
            let mean = posterior_means[index];
            let residual = mean - truth_values[index];
            let posterior_lower_95 = mean - PERIOD_Z * std;
            let posterior_upper_95 = mean + PERIOD_Z * std;
            PlanarHolesLoopFunctionalSummary {
                scenario,
                model,
                functional,
                hole_index: cycle.hole_index,
                hole_name: cycle.name.clone(),
                truth_value: truth_values[index],
                posterior_mean: mean,
                posterior_std: std,
                posterior_lower_95,
                posterior_upper_95,
                residual,
                abs_residual_over_std: relative_or_nan(residual.abs(), std),
                covered_95: truth_values[index] >= posterior_lower_95
                    && truth_values[index] <= posterior_upper_95,
            }
        })
        .collect()
}

fn compute_period_summaries(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    cycles: &[PlanarHoleCycle],
    truth: &FeecVector,
    cycle_observation_matrix: &FeecCsr,
    conditioned: &ConditionedModel,
) -> Result<Vec<PlanarHolesPeriodSummary>, Box<dyn Error>> {
    let operator = cycle_observation_matrix * &conditioned.latent_to_ambient;
    let variances = exact_transformed_variances(&conditioned.posterior_precision, &operator)?;
    let truth_periods = cycle_observation_matrix * truth;
    let posterior_periods = cycle_observation_matrix * &conditioned.posterior_mean;
    Ok(cycles
        .iter()
        .enumerate()
        .map(|(index, cycle)| {
            let std = variances[index].max(0.0).sqrt();
            let mean = posterior_periods[index];
            let residual = mean - truth_periods[index];
            let posterior_lower_95 = mean - PERIOD_Z * std;
            let posterior_upper_95 = mean + PERIOD_Z * std;
            PlanarHolesPeriodSummary {
                scenario,
                model,
                hole_index: cycle.hole_index,
                hole_name: cycle.name.clone(),
                truth_period: truth_periods[index],
                posterior_mean: mean,
                posterior_std: std,
                posterior_lower_95,
                posterior_upper_95,
                residual,
                abs_residual_over_std: relative_or_nan(residual.abs(), std),
                covered_95: truth_periods[index] >= posterior_lower_95
                    && truth_periods[index] <= posterior_upper_95,
            }
        })
        .collect())
}

fn compute_heldout_predictions(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    heldout: &[PlanarHolesObservation],
    heldout_matrix: &FeecCsr,
    conditioned: &ConditionedModel,
) -> Result<Vec<PlanarHolesHeldoutPrediction>, Box<dyn Error>> {
    let operator = heldout_matrix * &conditioned.latent_to_ambient;
    let variances = exact_transformed_variances(&conditioned.posterior_precision, &operator)?;
    let means = heldout_matrix * &conditioned.posterior_mean;
    Ok(heldout
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            let predictive_variance = variances[index].max(0.0) + observation.noise_variance;
            let residual = observation.observed_value - means[index];
            let nlpd = 0.5
                * ((TWO_PI * predictive_variance).ln() + residual * residual / predictive_variance);
            PlanarHolesHeldoutPrediction {
                scenario,
                model,
                kind: observation.kind,
                label: observation.label.clone(),
                observed_value: observation.observed_value,
                truth_value: observation.truth_value,
                predictive_mean: means[index],
                predictive_variance,
                nlpd,
            }
        })
        .collect())
}

fn compute_field_coverage_diagnostics(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    mass_1form: &FeecCsr,
    truth: &FeecVector,
    heldout: &[PlanarHolesObservation],
    conditioned: &ConditionedModel,
) -> Result<ComputedFieldCoverage, Box<dyn Error>> {
    let edge_count = truth.len();
    if conditioned.posterior_mean.len() != edge_count {
        return Err(invalid_data(format!(
            "posterior mean length {} does not match truth length {edge_count}",
            conditioned.posterior_mean.len()
        ))
        .into());
    }
    let variances = exact_transformed_variances(
        &conditioned.posterior_precision,
        &conditioned.latent_to_ambient,
    )?;
    if variances.len() != edge_count {
        return Err(invalid_data(format!(
            "field variance length {} does not match truth length {edge_count}",
            variances.len()
        ))
        .into());
    }

    let mut posterior_std = Vec::with_capacity(edge_count);
    let mut abs_z_score = Vec::with_capacity(edge_count);
    let mut covered_95 = Vec::with_capacity(edge_count);
    let mut posterior_mean_error = Vec::with_capacity(edge_count);

    for edge in 0..edge_count {
        let variance = variances[edge].max(0.0);
        let std = variance.sqrt();
        let error = conditioned.posterior_mean[edge] - truth[edge];
        let z_std = variance.max(FIELD_VARIANCE_FLOOR).sqrt();
        posterior_std.push(std);
        abs_z_score.push(error.abs() / z_std);
        covered_95.push((error.abs() <= FIELD_COVERAGE_Z * std) as u8 as f64);
        posterior_mean_error.push(error);
    }

    let mass_weights = positive_diagonal(mass_1form, edge_count);
    let all_edges = (0..edge_count).collect::<Vec<_>>();
    let heldout_local_edges = heldout_local_edge_indices(heldout);
    let summaries = vec![
        summarize_field_coverage_subset(
            scenario,
            model,
            PlanarHolesFieldCoverageSubset::AllEdges,
            &all_edges,
            &mass_weights,
            &posterior_std,
            &abs_z_score,
            &covered_95,
            &posterior_mean_error,
        ),
        summarize_field_coverage_subset(
            scenario,
            model,
            PlanarHolesFieldCoverageSubset::HeldoutLocalEdges,
            &heldout_local_edges,
            &mass_weights,
            &posterior_std,
            &abs_z_score,
            &covered_95,
            &posterior_mean_error,
        ),
    ];

    Ok(ComputedFieldCoverage {
        summaries,
        diagnostics: PlanarHolesFieldCoverageDiagnostics {
            posterior_std: FeecVector::from_vec(posterior_std),
            abs_z_score: FeecVector::from_vec(abs_z_score),
            covered_95: FeecVector::from_vec(covered_95),
            posterior_mean_error: FeecVector::from_vec(posterior_mean_error),
        },
    })
}

fn empty_field_coverage_diagnostics(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    mass_1form: &FeecCsr,
    truth: &FeecVector,
    heldout: &[PlanarHolesObservation],
    posterior_mean: &FeecVector,
) -> ComputedFieldCoverage {
    let edge_count = truth.len();
    let posterior_mean_error = FeecVector::from_iterator(
        edge_count,
        (0..edge_count).map(|edge| posterior_mean[edge] - truth[edge]),
    );
    let nan_values = FeecVector::from_element(edge_count, f64::NAN);
    let mass_weights = positive_diagonal(mass_1form, edge_count);
    let all_edges = (0..edge_count).collect::<Vec<_>>();
    let heldout_local_edges = heldout_local_edge_indices(heldout);
    let summaries = vec![
        skipped_field_coverage_summary(
            scenario,
            model,
            PlanarHolesFieldCoverageSubset::AllEdges,
            &all_edges,
            &mass_weights,
        ),
        skipped_field_coverage_summary(
            scenario,
            model,
            PlanarHolesFieldCoverageSubset::HeldoutLocalEdges,
            &heldout_local_edges,
            &mass_weights,
        ),
    ];
    ComputedFieldCoverage {
        summaries,
        diagnostics: PlanarHolesFieldCoverageDiagnostics {
            posterior_std: nan_values.clone(),
            abs_z_score: nan_values.clone(),
            covered_95: nan_values,
            posterior_mean_error,
        },
    }
}

fn skipped_field_coverage_summary(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    subset: PlanarHolesFieldCoverageSubset,
    indices: &[usize],
    mass_weights: &[f64],
) -> PlanarHolesFieldCoverageSummary {
    PlanarHolesFieldCoverageSummary {
        scenario,
        model,
        subset,
        edge_count: indices.len(),
        weight_sum: indices.iter().map(|edge| mass_weights[*edge]).sum(),
        coverage_95: f64::NAN,
        mass_weighted_coverage_95: f64::NAN,
        mean_abs_z: f64::NAN,
        rms_z: f64::NAN,
        p95_abs_z: f64::NAN,
        mean_posterior_std: f64::NAN,
        mass_weighted_mean_posterior_std: f64::NAN,
        latent_nlpd: f64::NAN,
    }
}

#[allow(clippy::too_many_arguments)]
fn summarize_field_coverage_subset(
    scenario: PlanarHolesObservationScenario,
    model: PlanarHolesModelKind,
    subset: PlanarHolesFieldCoverageSubset,
    indices: &[usize],
    mass_weights: &[f64],
    posterior_std: &[f64],
    abs_z_score: &[f64],
    covered_95: &[f64],
    posterior_mean_error: &[f64],
) -> PlanarHolesFieldCoverageSummary {
    let latent_nlpd = mean(indices.iter().map(|edge| {
        let variance = (posterior_std[*edge] * posterior_std[*edge]).max(FIELD_VARIANCE_FLOOR);
        let error = posterior_mean_error[*edge];
        0.5 * ((TWO_PI * variance).ln() + error * error / variance)
    }));
    PlanarHolesFieldCoverageSummary {
        scenario,
        model,
        subset,
        edge_count: indices.len(),
        weight_sum: indices.iter().map(|edge| mass_weights[*edge]).sum(),
        coverage_95: mean(indices.iter().map(|edge| covered_95[*edge])),
        mass_weighted_coverage_95: weighted_mean(indices, mass_weights, covered_95),
        mean_abs_z: mean(indices.iter().map(|edge| abs_z_score[*edge])),
        rms_z: mean(indices.iter().map(|edge| {
            let z = abs_z_score[*edge];
            z * z
        }))
        .sqrt(),
        p95_abs_z: percentile(indices.iter().map(|edge| abs_z_score[*edge]), 0.95),
        mean_posterior_std: mean(indices.iter().map(|edge| posterior_std[*edge])),
        mass_weighted_mean_posterior_std: weighted_mean(indices, mass_weights, posterior_std),
        latent_nlpd,
    }
}

fn heldout_local_edge_indices(heldout: &[PlanarHolesObservation]) -> Vec<usize> {
    let mut edges = BTreeSet::new();
    for observation in heldout
        .iter()
        .filter(|observation| observation.kind == PlanarHolesObservationKind::HeldoutLocal)
    {
        for (edge, _) in &observation.entries {
            edges.insert(*edge);
        }
    }
    edges.into_iter().collect()
}

fn positive_diagonal(matrix: &FeecCsr, dimension: usize) -> Vec<f64> {
    let mut diagonal = vec![0.0; dimension];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col && row < dimension {
            diagonal[row] += *value;
        }
    }
    diagonal
        .into_iter()
        .map(|value| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                0.0
            }
        })
        .collect()
}

fn weighted_mean(indices: &[usize], weights: &[f64], values: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for edge in indices {
        let weight = weights[*edge];
        if weight.is_finite() && weight > 0.0 {
            numerator += weight * values[*edge];
            denominator += weight;
        }
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn percentile(values: impl Iterator<Item = f64>, probability: f64) -> f64 {
    let mut finite = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    let probability = probability.clamp(0.0, 1.0);
    let index = ((finite.len() - 1) as f64 * probability).ceil() as usize;
    finite[index]
}

fn exact_transformed_variances(
    precision: &GmrfSparseMatrix,
    operator: &FeecCsr,
) -> Result<FeecVector, Box<dyn Error>> {
    let sparse_operator = sparse_row_operator_from_feec_csr(operator).map_err(invalid_data)?;
    let constraints = GmrfDenseMatrix::zeros(0, precision.nrows());
    let mut gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())?;
    let decomposition =
        gmrf.exact_transformed_variance_decomposition(&sparse_operator, &constraints)?;
    Ok(gmrf_vec_to_feec(&decomposition.constrained_diag))
}

fn build_edge_component_operator(topology: &Complex, coords: &MeshCoords) -> FeecCsr {
    let vertex_count = topology.vertices().len();
    let mut coo = FeecCoo::new(topology.edges().len(), 2 * vertex_count);
    for edge in topology.edges().handle_iter() {
        let row = edge.kidx();
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        let ca = coords.coord(a);
        let cb = coords.coord(b);
        let dx = cb[0] - ca[0];
        let dy = cb[1] - ca[1];
        coo.push(row, a, 0.5 * dx);
        coo.push(row, b, 0.5 * dx);
        coo.push(row, vertex_count + a, 0.5 * dy);
        coo.push(row, vertex_count + b, 0.5 * dy);
    }
    FeecCsr::from(&coo)
}

fn build_hole_cycle_observation_matrix(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    radius_offset: f64,
    angle_offset: f64,
    family: &str,
) -> Result<(FeecCsr, Vec<PlanarHoleCycle>), io::Error> {
    let adjacency = build_edge_adjacency(topology, coords);
    let mut rows = Vec::with_capacity(holes.len());
    let mut cycles = Vec::with_capacity(holes.len());
    for (hole_index, hole) in holes.iter().enumerate() {
        let radius = hole.radius + radius_offset;
        let target_points = circle_points(hole.center, radius, 16, angle_offset);
        let target_vertices = target_points
            .iter()
            .map(|point| de_rham::nearest_vertex(coords, *point))
            .collect::<Vec<_>>();
        let (row, path_vertices) =
            build_cycle_row(topology, &adjacency, &target_vertices).map_err(invalid_data)?;
        let closure_residual_l1 = cycle_closure_residual(topology, &row);
        if closure_residual_l1 > 1e-10 {
            return Err(invalid_data(format!(
                "hole cycle {} is not closed; l1 residual={closure_residual_l1}",
                hole.name
            )));
        }
        cycles.push(PlanarHoleCycle {
            family: family.to_string(),
            hole_index,
            name: format!("{family}_{}", hole.name),
            radius,
            angle_offset,
            target_vertices,
            path_vertices,
            edge_count: row.len(),
            closure_residual_l1,
        });
        rows.push(row);
    }

    let mut coo = FeecCoo::new(rows.len(), topology.edges().len());
    for (row_index, row) in rows.iter().enumerate() {
        for (edge_index, weight) in row {
            coo.push(row_index, *edge_index, *weight);
        }
    }
    Ok((FeecCsr::from(&coo), cycles))
}

fn build_hole_cycle_families(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    specs: &[(f64, f64, &str)],
) -> Result<Vec<(FeecCsr, Vec<PlanarHoleCycle>)>, io::Error> {
    specs
        .iter()
        .map(|(radius_offset, angle_offset, family)| {
            build_hole_cycle_observation_matrix(
                topology,
                coords,
                holes,
                *radius_offset,
                *angle_offset,
                family,
            )
        })
        .collect()
}

fn circle_points(center: [f64; 2], radius: f64, count: usize, angle_offset: f64) -> Vec<[f64; 3]> {
    (0..count)
        .map(|index| {
            let angle = angle_offset + TWO_PI * index as f64 / count as f64;
            [
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
                0.0,
            ]
        })
        .collect()
}

fn matching_cycle_rows_identical(train: &FeecCsr, heldout: &FeecCsr) -> bool {
    if train.nrows() != heldout.nrows() || train.ncols() != heldout.ncols() {
        return false;
    }
    let train_rows = csr_rows(train);
    let heldout_rows = csr_rows(heldout);
    train_rows
        .iter()
        .zip(heldout_rows.iter())
        .all(|(left, right)| sparse_rows_approximately_equal(left, right, 1e-12))
}

fn sparse_rows_approximately_equal(
    left: &[(usize, f64)],
    right: &[(usize, f64)],
    tolerance: f64,
) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|((left_col, left_value), (right_col, right_value))| {
            left_col == right_col && (left_value - right_value).abs() <= tolerance
        })
}

fn build_edge_adjacency(topology: &Complex, coords: &MeshCoords) -> Vec<Vec<EdgeStep>> {
    let mut adjacency = vec![Vec::new(); topology.vertices().len()];
    for edge in topology.edges().handle_iter() {
        let a = edge.vertices[0];
        let b = edge.vertices[1];
        let length = (coords.coord(a) - coords.coord(b)).norm();
        adjacency[a].push(EdgeStep {
            to: b,
            edge_index: edge.kidx(),
            sign: 1.0,
            length,
        });
        adjacency[b].push(EdgeStep {
            to: a,
            edge_index: edge.kidx(),
            sign: -1.0,
            length,
        });
    }
    adjacency
}

fn build_cycle_row(
    topology: &Complex,
    adjacency: &[Vec<EdgeStep>],
    target_vertices: &[usize],
) -> Result<(Vec<(usize, f64)>, Vec<usize>), String> {
    let mut vertices = target_vertices.to_vec();
    vertices.dedup();
    if vertices.len() < 2 {
        return Err("cycle requires at least two distinct target vertices".to_string());
    }

    let mut row = BTreeMap::<usize, f64>::new();
    let mut path_vertices = vec![vertices[0]];
    for index in 0..vertices.len() {
        let start = vertices[index];
        let goal = vertices[(index + 1) % vertices.len()];
        let path = shortest_path(adjacency, start, goal)?;
        for (edge_index, sign) in path.edges {
            *row.entry(edge_index).or_insert(0.0) += sign;
        }
        path_vertices.extend(path.vertices.into_iter().skip(1));
    }
    if path_vertices.last().copied() != Some(path_vertices[0]) {
        path_vertices.push(path_vertices[0]);
    }

    let entries = row
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect::<Vec<_>>();
    if entries
        .iter()
        .any(|(edge_index, _)| *edge_index >= topology.edges().len())
    {
        return Err("cycle row contains an invalid edge index".to_string());
    }
    Ok((entries, path_vertices))
}

fn shortest_path(
    adjacency: &[Vec<EdgeStep>],
    start: usize,
    goal: usize,
) -> Result<ShortestPath, String> {
    if start == goal {
        return Ok(ShortestPath {
            vertices: vec![start],
            edges: Vec::new(),
        });
    }
    let mut dist = vec![f64::INFINITY; adjacency.len()];
    let mut prev = vec![None::<(usize, usize, f64)>; adjacency.len()];
    let mut heap = BinaryHeap::new();
    dist[start] = 0.0;
    heap.push(QueueState {
        cost: 0.0,
        vertex: start,
    });

    while let Some(QueueState { cost, vertex }) = heap.pop() {
        if vertex == goal {
            break;
        }
        if cost > dist[vertex] + EPS {
            continue;
        }
        for step in &adjacency[vertex] {
            let next_cost = cost + step.length;
            if next_cost + EPS < dist[step.to] {
                dist[step.to] = next_cost;
                prev[step.to] = Some((vertex, step.edge_index, step.sign));
                heap.push(QueueState {
                    cost: next_cost,
                    vertex: step.to,
                });
            }
        }
    }

    if !dist[goal].is_finite() {
        return Err(format!("no path found from vertex {start} to {goal}"));
    }
    let mut current = goal;
    let mut reversed_vertices = vec![goal];
    let mut reversed_edges = Vec::new();
    while current != start {
        let Some((parent, edge_index, sign)) = prev[current] else {
            return Err(format!("failed to reconstruct path from {start} to {goal}"));
        };
        reversed_edges.push((edge_index, sign));
        current = parent;
        reversed_vertices.push(current);
    }
    reversed_vertices.reverse();
    reversed_edges.reverse();
    Ok(ShortestPath {
        vertices: reversed_vertices,
        edges: reversed_edges,
    })
}

fn build_open_path_row(
    topology: &Complex,
    adjacency: &[Vec<EdgeStep>],
    target_vertices: &[usize],
) -> Result<(Vec<(usize, f64)>, Vec<usize>), String> {
    let mut vertices = target_vertices.to_vec();
    vertices.dedup();
    if vertices.len() < 2 {
        return Err("open path requires at least two distinct target vertices".to_string());
    }

    let mut row = BTreeMap::<usize, f64>::new();
    let mut path_vertices = vec![vertices[0]];
    for pair in vertices.windows(2) {
        let path = shortest_path(adjacency, pair[0], pair[1])?;
        for (edge_index, sign) in path.edges {
            *row.entry(edge_index).or_insert(0.0) += sign;
        }
        path_vertices.extend(path.vertices.into_iter().skip(1));
    }
    let entries = row
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect::<Vec<_>>();
    if entries
        .iter()
        .any(|(edge_index, _)| *edge_index >= topology.edges().len())
    {
        return Err("open path row contains an invalid edge index".to_string());
    }
    Ok((entries, path_vertices))
}

fn combine_sensor_rows(weighted_rows: &[(&[(usize, f64)], f64)]) -> Vec<(usize, f64)> {
    let mut row = BTreeMap::<usize, f64>::new();
    for (entries, scale) in weighted_rows {
        for (edge, weight) in *entries {
            *row.entry(*edge).or_insert(0.0) += scale * weight;
        }
    }
    row.into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect()
}

fn cycle_closure_residual(topology: &Complex, row: &[(usize, f64)]) -> f64 {
    let mut balance = vec![0.0; topology.vertices().len()];
    for (edge_index, weight) in row {
        let edge = topology.edges().handle_by_kidx(*edge_index);
        balance[edge.vertices[0]] -= *weight;
        balance[edge.vertices[1]] += *weight;
    }
    balance.iter().map(|value| value.abs()).sum()
}

fn select_observation_edges(
    topology: &Complex,
    coords: &MeshCoords,
    cycle_edges: &BTreeSet<usize>,
    excluded_edges: &BTreeSet<usize>,
    count: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<usize> {
    let mut candidates = topology
        .edges()
        .handle_iter()
        .filter(|edge| {
            !cycle_edges.contains(&edge.kidx())
                && !excluded_edges.contains(&edge.kidx())
                && edge.cocells().count() == 2
                && interior_edge_barycenter(coords, edge.vertices[0], edge.vertices[1])
        })
        .map(|edge| edge.kidx())
        .collect::<Vec<_>>();
    candidates.shuffle(rng);
    candidates.truncate(count.min(candidates.len()));
    candidates.sort_unstable();
    candidates
}

fn select_observation_edges_by_barycenter<P>(
    topology: &Complex,
    coords: &MeshCoords,
    cycle_edges: &BTreeSet<usize>,
    excluded_edges: &BTreeSet<usize>,
    count: usize,
    rng: &mut rand::rngs::StdRng,
    predicate: P,
) -> Vec<usize>
where
    P: Fn([f64; 2]) -> bool,
{
    let mut candidates = topology
        .edges()
        .handle_iter()
        .filter(|edge| {
            if cycle_edges.contains(&edge.kidx())
                || excluded_edges.contains(&edge.kidx())
                || edge.cocells().count() != 2
                || !interior_edge_barycenter(coords, edge.vertices[0], edge.vertices[1])
            {
                return false;
            }
            let a = coords.coord(edge.vertices[0]);
            let b = coords.coord(edge.vertices[1]);
            predicate([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
        })
        .map(|edge| edge.kidx())
        .collect::<Vec<_>>();
    candidates.shuffle(rng);
    candidates.truncate(count.min(candidates.len()));
    candidates.sort_unstable();
    candidates
}

fn select_fraction_observation_edges(
    topology: &Complex,
    coords: &MeshCoords,
    cycle_edges: &BTreeSet<usize>,
    excluded_edges: &BTreeSet<usize>,
    fraction: f64,
    rng: &mut rand::rngs::StdRng,
) -> Vec<usize> {
    let mut candidates = topology
        .edges()
        .handle_iter()
        .filter(|edge| {
            !cycle_edges.contains(&edge.kidx())
                && !excluded_edges.contains(&edge.kidx())
                && edge.cocells().count() == 2
                && interior_edge_barycenter(coords, edge.vertices[0], edge.vertices[1])
        })
        .map(|edge| edge.kidx())
        .collect::<Vec<_>>();
    candidates.shuffle(rng);
    let count = ((candidates.len() as f64) * fraction.clamp(0.0, 1.0)).round() as usize;
    candidates.truncate(count.max(1).min(candidates.len()));
    candidates.sort_unstable();
    candidates
}

fn interior_edge_barycenter(coords: &MeshCoords, a_vertex: usize, b_vertex: usize) -> bool {
    let a = coords.coord(a_vertex);
    let b = coords.coord(b_vertex);
    let x = 0.5 * (a[0] + b[0]);
    let y = 0.5 * (a[1] + b[1]);
    x > 0.08 && x < 0.92 && y > 0.08 && y < 0.92
}

fn edge_observations(
    kind: PlanarHolesObservationKind,
    prefix: &str,
    edges: &[usize],
    truth: &FeecVector,
    noise_variance: f64,
    sample_noise: bool,
    rng: &mut rand::rngs::StdRng,
) -> Vec<PlanarHolesObservation> {
    edges
        .iter()
        .copied()
        .enumerate()
        .map(|(index, edge)| {
            let truth_value = truth[edge];
            let noise = if sample_noise {
                rng.sample::<f64, _>(StandardNormal) * noise_variance.sqrt()
            } else {
                0.0
            };
            PlanarHolesObservation {
                kind,
                label: format!("{prefix}_{index}_edge_{edge}"),
                entries: vec![(edge, 1.0)],
                observed_value: truth_value + noise,
                truth_value,
                noise_variance,
            }
        })
        .collect()
}

fn loop_observations(
    kind: PlanarHolesObservationKind,
    prefix: &str,
    cycle_matrix: &FeecCsr,
    cycles: &[PlanarHoleCycle],
    truth: &FeecVector,
    noise_variance: f64,
    sample_noise: bool,
    rng: &mut rand::rngs::StdRng,
) -> Vec<PlanarHolesObservation> {
    let rows = csr_rows(cycle_matrix);
    let truth_values = cycle_matrix * truth;
    rows.into_iter()
        .enumerate()
        .map(|(index, entries)| {
            let noise = if sample_noise {
                rng.sample::<f64, _>(StandardNormal) * noise_variance.sqrt()
            } else {
                0.0
            };
            PlanarHolesObservation {
                kind,
                label: format!("{prefix}_{}", cycles[index].name),
                entries,
                observed_value: truth_values[index] + noise,
                truth_value: truth_values[index],
                noise_variance,
            }
        })
        .collect()
}

fn row_observations(
    kind: PlanarHolesObservationKind,
    rows: &[SensorRow],
    truth: &FeecVector,
    noise_variance: f64,
    sample_noise: bool,
    rng: &mut rand::rngs::StdRng,
) -> Vec<PlanarHolesObservation> {
    rows.iter()
        .map(|row| {
            let truth_value = row
                .entries
                .iter()
                .map(|(edge, weight)| weight * truth[*edge])
                .sum::<f64>();
            let noise = if sample_noise {
                rng.sample::<f64, _>(StandardNormal) * noise_variance.sqrt()
            } else {
                0.0
            };
            PlanarHolesObservation {
                kind,
                label: row.label.clone(),
                entries: row.entries.clone(),
                observed_value: truth_value + noise,
                truth_value,
                noise_variance,
            }
        })
        .collect()
}

fn build_contractible_loop_rows(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    radii: &[f64],
    count: usize,
    seed: u64,
    prefix: &str,
) -> Vec<SensorRow> {
    let adjacency = build_edge_adjacency(topology, coords);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut candidates = Vec::new();
    for &radius in radii {
        for ix in 2..18 {
            for iy in 2..18 {
                let center = [ix as f64 / 19.0, iy as f64 / 19.0];
                if valid_contractible_loop_center(center, radius, holes) {
                    candidates.push((center, radius));
                }
            }
        }
    }
    candidates.shuffle(&mut rng);
    let mut rows = Vec::new();
    let mut signatures = BTreeSet::new();
    for (center, radius) in candidates {
        if rows.len() >= count {
            break;
        }
        let angle_offset = rng.gen::<f64>() * TWO_PI;
        let target_points = circle_points(center, radius, 12, angle_offset);
        let target_vertices = target_points
            .iter()
            .map(|point| de_rham::nearest_vertex(coords, *point))
            .collect::<Vec<_>>();
        let Ok((entries, _)) = build_cycle_row(topology, &adjacency, &target_vertices) else {
            continue;
        };
        if entries.is_empty() || cycle_closure_residual(topology, &entries) > 1e-10 {
            continue;
        }
        let signature = entries
            .iter()
            .map(|(edge, weight)| (*edge, weight.signum() as i8))
            .collect::<Vec<_>>();
        if !signatures.insert(signature) {
            continue;
        }
        rows.push(SensorRow {
            label: format!(
                "{prefix}_{}_r{:.3}_x{:.3}_y{:.3}",
                rows.len(),
                radius,
                center[0],
                center[1]
            ),
            entries,
        });
    }
    rows
}

fn valid_contractible_loop_center(center: [f64; 2], radius: f64, holes: &[PlanarHoleSpec]) -> bool {
    let margin = 0.045;
    if center[0] < radius + margin
        || center[0] > 1.0 - radius - margin
        || center[1] < radius + margin
        || center[1] > 1.0 - radius - margin
    {
        return false;
    }
    holes.iter().all(|hole| {
        let dx = center[0] - hole.center[0];
        let dy = center[1] - hole.center[1];
        (dx * dx + dy * dy).sqrt() > hole.radius + radius + margin
    })
}

fn build_long_path_rows(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    count: usize,
    seed: u64,
    prefix: &str,
) -> Vec<SensorRow> {
    let adjacency = build_edge_adjacency(topology, coords);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut points = Vec::new();
    for ix in 2..18 {
        for iy in 2..18 {
            let point = [ix as f64 / 19.0, iy as f64 / 19.0];
            if valid_long_path_endpoint(point, holes) {
                points.push(point);
            }
        }
    }
    points.shuffle(&mut rng);
    let mut pairs = Vec::new();
    for (i, start) in points.iter().enumerate() {
        for end in points.iter().skip(i + 1) {
            let dx = start[0] - end[0];
            let dy = start[1] - end[1];
            if (dx * dx + dy * dy).sqrt() >= 0.55 {
                pairs.push((*start, *end));
            }
        }
    }
    pairs.shuffle(&mut rng);
    let mut rows = Vec::new();
    let mut signatures = BTreeSet::new();
    for (start_point, end_point) in pairs {
        if rows.len() >= count {
            break;
        }
        let start = de_rham::nearest_vertex(coords, [start_point[0], start_point[1], 0.0]);
        let end = de_rham::nearest_vertex(coords, [end_point[0], end_point[1], 0.0]);
        let Ok(path) = shortest_path(&adjacency, start, end) else {
            continue;
        };
        if path.edges.len() < 8 {
            continue;
        }
        let mut row = BTreeMap::<usize, f64>::new();
        for (edge, sign) in path.edges {
            *row.entry(edge).or_insert(0.0) += sign;
        }
        let entries = row
            .into_iter()
            .filter(|(_, weight)| weight.abs() > EPS)
            .collect::<Vec<_>>();
        let signature = entries
            .iter()
            .map(|(edge, weight)| (*edge, weight.signum() as i8))
            .collect::<Vec<_>>();
        if entries.is_empty() || !signatures.insert(signature) {
            continue;
        }
        rows.push(SensorRow {
            label: format!(
                "{prefix}_{}_x{:.3}_y{:.3}_to_x{:.3}_y{:.3}",
                rows.len(),
                start_point[0],
                start_point[1],
                end_point[0],
                end_point[1]
            ),
            entries,
        });
    }
    rows
}

fn build_same_side_long_path_rows(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    x_range: (f64, f64),
    count: usize,
    seed: u64,
    prefix: &str,
) -> Vec<SensorRow> {
    build_filtered_long_path_rows(
        topology,
        coords,
        holes,
        count,
        seed,
        prefix,
        |point| point[0] >= x_range.0 && point[0] <= x_range.1,
        |start, end| {
            let dy = start[1] - end[1];
            dy.abs() >= 0.35
        },
    )
}

fn build_cross_barrier_path_rows(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    count: usize,
    seed: u64,
    prefix: &str,
) -> Vec<SensorRow> {
    build_filtered_long_path_rows(
        topology,
        coords,
        holes,
        count,
        seed,
        prefix,
        |point| point[0] <= 0.30 || point[0] >= 0.70,
        |start, end| {
            let crosses =
                (start[0] <= 0.30 && end[0] >= 0.70) || (start[0] >= 0.70 && end[0] <= 0.30);
            let dy = (start[1] - end[1]).abs();
            crosses && dy <= 0.55
        },
    )
}

fn build_filtered_long_path_rows<P, Q>(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    count: usize,
    seed: u64,
    prefix: &str,
    point_filter: P,
    pair_filter: Q,
) -> Vec<SensorRow>
where
    P: Fn([f64; 2]) -> bool,
    Q: Fn([f64; 2], [f64; 2]) -> bool,
{
    let adjacency = build_edge_adjacency(topology, coords);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut points = Vec::new();
    for ix in 2..24 {
        for iy in 2..24 {
            let point = [ix as f64 / 25.0, iy as f64 / 25.0];
            if valid_long_path_endpoint(point, holes) && point_filter(point) {
                points.push(point);
            }
        }
    }
    points.shuffle(&mut rng);
    let mut pairs = Vec::new();
    for (i, start) in points.iter().enumerate() {
        for end in points.iter().skip(i + 1) {
            if pair_filter(*start, *end) {
                pairs.push((*start, *end));
            }
        }
    }
    pairs.shuffle(&mut rng);
    let mut rows = Vec::new();
    let mut signatures = BTreeSet::new();
    for (start_point, end_point) in pairs {
        if rows.len() >= count {
            break;
        }
        let start = de_rham::nearest_vertex(coords, [start_point[0], start_point[1], 0.0]);
        let end = de_rham::nearest_vertex(coords, [end_point[0], end_point[1], 0.0]);
        let Ok(path) = shortest_path(&adjacency, start, end) else {
            continue;
        };
        if path.edges.len() < 8 {
            continue;
        }
        let mut row = BTreeMap::<usize, f64>::new();
        for (edge, sign) in path.edges {
            *row.entry(edge).or_insert(0.0) += sign;
        }
        let entries = row
            .into_iter()
            .filter(|(_, weight)| weight.abs() > EPS)
            .collect::<Vec<_>>();
        let signature = entries
            .iter()
            .map(|(edge, weight)| (*edge, weight.signum() as i8))
            .collect::<Vec<_>>();
        if entries.is_empty() || !signatures.insert(signature) {
            continue;
        }
        rows.push(SensorRow {
            label: format!(
                "{prefix}_{}_x{:.3}_y{:.3}_to_x{:.3}_y{:.3}",
                rows.len(),
                start_point[0],
                start_point[1],
                end_point[0],
                end_point[1]
            ),
            entries,
        });
    }
    rows
}

fn build_path_homology_pairs(
    topology: &Complex,
    coords: &MeshCoords,
    holes: &[PlanarHoleSpec],
    radius_offset: f64,
    angle_offset: f64,
    family: &str,
) -> Result<Vec<PathHomologyPair>, io::Error> {
    let adjacency = build_edge_adjacency(topology, coords);
    let mut pairs = Vec::new();
    for (hole_index, hole) in holes.iter().enumerate() {
        let radius = hole.radius + radius_offset;
        let upper_angles = [
            std::f64::consts::PI + angle_offset,
            0.75 * std::f64::consts::PI + angle_offset,
            0.50 * std::f64::consts::PI + angle_offset,
            0.25 * std::f64::consts::PI + angle_offset,
            angle_offset,
        ];
        let lower_angles = [
            std::f64::consts::PI + angle_offset,
            1.25 * std::f64::consts::PI + angle_offset,
            1.50 * std::f64::consts::PI + angle_offset,
            1.75 * std::f64::consts::PI + angle_offset,
            angle_offset,
        ];
        let angle_to_point = |angle: f64| {
            [
                hole.center[0] + radius * angle.cos(),
                hole.center[1] + radius * angle.sin(),
                0.0,
            ]
        };
        let upper_vertices = upper_angles
            .iter()
            .map(|angle| de_rham::nearest_vertex(coords, angle_to_point(*angle)))
            .collect::<Vec<_>>();
        let lower_vertices = lower_angles
            .iter()
            .map(|angle| de_rham::nearest_vertex(coords, angle_to_point(*angle)))
            .collect::<Vec<_>>();
        if upper_vertices.first() != lower_vertices.first()
            || upper_vertices.last() != lower_vertices.last()
        {
            return Err(invalid_data(format!(
                "path homology pair for {} does not share endpoints",
                hole.name
            )));
        }
        let (upper_entries, _) =
            build_open_path_row(topology, &adjacency, &upper_vertices).map_err(invalid_data)?;
        let (lower_entries, _) =
            build_open_path_row(topology, &adjacency, &lower_vertices).map_err(invalid_data)?;
        let contrast_entries =
            combine_sensor_rows(&[(&upper_entries, 1.0), (&lower_entries, -1.0)]);
        let closure_residual_l1 = cycle_closure_residual(topology, &contrast_entries);
        if closure_residual_l1 > 1e-10 {
            return Err(invalid_data(format!(
                "path contrast for {} is not closed; l1 residual={closure_residual_l1}",
                hole.name
            )));
        }
        pairs.push(PathHomologyPair {
            family: family.to_string(),
            hole_index,
            hole_name: hole.name.clone(),
            upper: SensorRow {
                label: format!("{family}_{}_upper", hole.name),
                entries: upper_entries,
            },
            lower: SensorRow {
                label: format!("{family}_{}_lower", hole.name),
                entries: lower_entries,
            },
            contrast: SensorRow {
                label: format!("{family}_{}_contrast", hole.name),
                entries: contrast_entries,
            },
            shared_start_vertex: *upper_vertices.first().unwrap_or(&0),
            shared_end_vertex: *upper_vertices.last().unwrap_or(&0),
            closure_residual_l1,
        });
    }
    Ok(pairs)
}

fn path_pair_rows<'a>(pairs: &'a [PathHomologyPair]) -> impl Iterator<Item = &'a SensorRow> + 'a {
    pairs
        .iter()
        .flat_map(|pair| [&pair.upper, &pair.lower].into_iter())
}

fn sensor_rows_to_csr(rows: &[SensorRow], column_count: usize) -> FeecCsr {
    let mut coo = FeecCoo::new(rows.len(), column_count);
    for (row_index, row) in rows.iter().enumerate() {
        for (edge, weight) in &row.entries {
            coo.push(row_index, *edge, *weight);
        }
    }
    FeecCsr::from(&coo)
}

fn edge_set_from_sensor_rows(rows: &[SensorRow]) -> BTreeSet<usize> {
    rows.iter()
        .flat_map(|row| row.entries.iter().map(|(edge, _)| *edge))
        .collect()
}

fn valid_long_path_endpoint(point: [f64; 2], holes: &[PlanarHoleSpec]) -> bool {
    if point[0] < 0.08 || point[0] > 0.92 || point[1] < 0.08 || point[1] > 0.92 {
        return false;
    }
    holes.iter().all(|hole| {
        let dx = point[0] - hole.center[0];
        let dy = point[1] - hole.center[1];
        (dx * dx + dy * dy).sqrt() > hole.radius + 0.08
    })
}

fn scaled_observation_system(
    observations: &[PlanarHolesObservation],
    ambient_dim: usize,
) -> (FeecCsr, FeecVector) {
    let matrix = observation_matrix(observations, ambient_dim, true);
    let values = FeecVector::from_iterator(
        observations.len(),
        observations
            .iter()
            .map(|observation| observation.observed_value / observation.noise_variance.sqrt()),
    );
    (matrix, values)
}

fn observation_matrix(
    observations: &[PlanarHolesObservation],
    ambient_dim: usize,
    whiten: bool,
) -> FeecCsr {
    let mut coo = FeecCoo::new(observations.len(), ambient_dim);
    for (row, observation) in observations.iter().enumerate() {
        let scale = if whiten {
            observation.noise_variance.sqrt().recip()
        } else {
            1.0
        };
        for (col, value) in &observation.entries {
            coo.push(row, *col, *value * scale);
        }
    }
    FeecCsr::from(&coo)
}

fn observation_edge_set(matrix: &FeecCsr) -> BTreeSet<usize> {
    matrix
        .triplet_iter()
        .map(|(_, col, _)| col)
        .collect::<BTreeSet<_>>()
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

fn difference_csr(lhs: &FeecCsr, rhs: &FeecCsr) -> io::Result<FeecCsr> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err(invalid_data(format!(
            "cannot subtract matrices with shapes {}x{} and {}x{}",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols()
        )));
    }
    let mut rows = vec![BTreeMap::<usize, f64>::new(); lhs.nrows()];
    add_scaled_csr_to_rows(&mut rows, lhs, 1.0);
    add_scaled_csr_to_rows(&mut rows, rhs, -1.0);
    let mut coo = FeecCoo::new(lhs.nrows(), lhs.ncols());
    for (row, entries) in rows.iter().enumerate() {
        for (col, value) in entries {
            if value.abs() > EPS {
                coo.push(row, *col, *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

fn add_scaled_csr_to_rows(rows: &mut [BTreeMap<usize, f64>], matrix: &FeecCsr, scale: f64) {
    for (row, col, value) in matrix.triplet_iter() {
        *rows[row].entry(col).or_insert(0.0) += scale * *value;
    }
}

fn shifted_branch_observation_operator(
    observation_matrix: &FeecCsr,
    branch_transform: &FeecCsr,
    branch_offset: usize,
    latent_dim: usize,
) -> io::Result<FeecCsr> {
    let branch_operator = observation_matrix * branch_transform;
    if branch_offset + branch_operator.ncols() > latent_dim {
        return Err(invalid_data(format!(
            "branch operator with offset {branch_offset} and width {} exceeds latent dimension {latent_dim}",
            branch_operator.ncols()
        )));
    }
    let mut coo = FeecCoo::new(branch_operator.nrows(), latent_dim);
    for (row, col, value) in branch_operator.triplet_iter() {
        coo.push(row, branch_offset + col, *value);
    }
    Ok(FeecCsr::from(&coo))
}

fn identity_csr(dimension: usize) -> FeecCsr {
    let mut coo = FeecCoo::new(dimension, dimension);
    for index in 0..dimension {
        coo.push(index, index, 1.0);
    }
    FeecCsr::from(&coo)
}

fn mass_norm(field: &FeecVector, mass: &FeecCsr) -> f64 {
    bilinear_form_sparse(mass, field, field).max(0.0).sqrt()
}

fn codifferential_leakage(
    topology: &Complex,
    metric: &MeshLengths,
    mass_1form: &FeecCsr,
    field: &FeecVector,
) -> Result<f64, Box<dyn Error>> {
    let mass_0form = de_rham::mass_matrix_form(topology, metric, 0).map_err(invalid_data)?;
    let delta = de_rham::codifferential(topology, metric, 1, field).map_err(invalid_data)?;
    Ok(relative_or_nan(
        mass_norm(&delta, &mass_0form),
        mass_norm(field, mass_1form),
    ))
}

fn boundary_lumped_energy_fraction(
    topology: &Complex,
    mass_1form: &FeecCsr,
    field: &FeecVector,
) -> f64 {
    let mass_weights = positive_diagonal(mass_1form, field.len());
    let boundary_edges = boundary_edge_indices(topology);
    let mut boundary_energy = 0.0;
    let mut total_energy = 0.0;
    for edge in 0..field.len() {
        let energy = mass_weights[edge] * field[edge] * field[edge];
        total_energy += energy;
        if boundary_edges.contains(&edge) {
            boundary_energy += energy;
        }
    }
    relative_or_nan(boundary_energy, total_energy)
}

fn operator_mass_norm(transform: &FeecCsr, mass: &FeecCsr) -> f64 {
    let weighted = mass * transform;
    let gram = transform.transpose() * &weighted;
    trace_sparse(&gram).max(0.0).sqrt()
}

fn codifferential_operator_norm(
    mass_1form: &FeecCsr,
    mass_0form: &FeecCsr,
    exact_transform: &FeecCsr,
    coexact_transform: &FeecCsr,
) -> Result<f64, Box<dyn Error>> {
    let weighted = mass_1form * coexact_transform;
    let weak_codifferential = exact_transform.transpose() * &weighted;
    weak_dual_operator_norm(&weak_codifferential, mass_0form)
}

fn weak_dual_operator_norm(weak_operator: &FeecCsr, mass: &FeecCsr) -> Result<f64, Box<dyn Error>> {
    let factor = feec_csr_to_gmrf(mass)
        .cholesky_sqrt_lower()
        .map_err(|err| invalid_data(format!("failed to factor mass matrix: {err}")))?;
    let dense = FeecMatrix::from(weak_operator);
    let mut norm_squared = 0.0;
    for col in 0..dense.ncols() {
        let rhs = FeecVector::from_iterator(
            dense.nrows(),
            (0..dense.nrows()).map(|row| dense[(row, col)]),
        );
        let solution = factor
            .solve(&feec_vec_to_gmrf(&rhs))
            .map_err(|err| invalid_data(format!("failed to apply inverse mass: {err}")))?;
        let solution = gmrf_vec_to_feec(&solution);
        norm_squared += rhs.dot(&solution);
    }
    Ok(norm_squared.max(0.0).sqrt())
}

fn mass_orthogonality_ratio(left: &FeecCsr, right: &FeecCsr, mass: &FeecCsr) -> f64 {
    let weighted_right = mass * right;
    let cross = left.transpose() * &weighted_right;
    let numerator = frobenius_norm_sparse(&cross);
    let denominator = operator_mass_norm(left, mass) * operator_mass_norm(right, mass);
    relative_or_nan(numerator, denominator)
}

#[derive(Debug, Clone)]
struct OperatorSubspaceSummary {
    left_rank: usize,
    right_rank: usize,
    min_cosine: f64,
    mean_cosine: f64,
    max_angle_degrees: f64,
}

fn mass_principal_angle_summary(
    left: &FeecCsr,
    right: &FeecCsr,
    mass: &FeecCsr,
) -> OperatorSubspaceSummary {
    let left_gram = mass_cross_gram_dense(left, left, mass);
    let right_gram = mass_cross_gram_dense(right, right, mass);
    let cross_gram = mass_cross_gram_dense(left, right, mass);
    let (left_inverse_sqrt, left_rank) = inverse_sqrt_coefficients(left_gram);
    let (right_inverse_sqrt, right_rank) = inverse_sqrt_coefficients(right_gram);
    if left_rank == 0 || right_rank == 0 {
        return OperatorSubspaceSummary {
            left_rank,
            right_rank,
            min_cosine: f64::NAN,
            mean_cosine: f64::NAN,
            max_angle_degrees: f64::NAN,
        };
    }
    let cosine_matrix = left_inverse_sqrt.transpose() * cross_gram * right_inverse_sqrt;
    let svd = cosine_matrix.svd(false, false);
    let common_rank = left_rank.min(right_rank);
    let mut cosines = svd
        .singular_values
        .iter()
        .take(common_rank)
        .map(|value| value.abs().clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    if left_rank != right_rank {
        cosines.extend(std::iter::repeat(0.0).take(left_rank.max(right_rank) - common_rank));
    }
    let min_cosine = cosines
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let mean_cosine = mean(cosines.iter().copied());
    let max_angle_degrees = min_cosine.acos() * 180.0 / std::f64::consts::PI;
    OperatorSubspaceSummary {
        left_rank,
        right_rank,
        min_cosine,
        mean_cosine,
        max_angle_degrees,
    }
}

fn mass_cross_gram_dense(left: &FeecCsr, right: &FeecCsr, mass: &FeecCsr) -> FeecMatrix {
    let weighted_right = mass * right;
    let gram = left.transpose() * &weighted_right;
    FeecMatrix::from(&gram)
}

fn inverse_sqrt_coefficients(gram: FeecMatrix) -> (FeecMatrix, usize) {
    let symmetric = symmetrize_dense(&gram);
    let eigen = symmetric.symmetric_eigen();
    let max_eigenvalue = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let tolerance = (SUBSPACE_EIGEN_TOLERANCE * max_eigenvalue).max(1e-14);
    let kept = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value > tolerance).then_some((index, *value)))
        .collect::<Vec<_>>();
    let mut coefficients = FeecMatrix::zeros(gram.ncols(), kept.len());
    for (out_col, (eigen_col, eigenvalue)) in kept.iter().copied().enumerate() {
        let scale = eigenvalue.sqrt().recip();
        for row in 0..gram.ncols() {
            coefficients[(row, out_col)] = eigen.eigenvectors[(row, eigen_col)] * scale;
        }
    }
    let rank = kept.len();
    (coefficients, rank)
}

fn symmetrize_dense(matrix: &FeecMatrix) -> FeecMatrix {
    let mut symmetric = matrix.clone();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            symmetric[(row, col)] = 0.5 * (matrix[(row, col)] + matrix[(col, row)]);
        }
    }
    symmetric
}

fn trace_sparse(matrix: &FeecCsr) -> f64 {
    matrix
        .triplet_iter()
        .filter_map(|(row, col, value)| (row == col).then_some(*value))
        .sum()
}

fn frobenius_norm_sparse(matrix: &FeecCsr) -> f64 {
    matrix
        .triplet_iter()
        .map(|(_, _, value)| *value * *value)
        .sum::<f64>()
        .sqrt()
}

fn relative_or_nan(numerator: f64, denominator: f64) -> f64 {
    if denominator.is_finite() && denominator.abs() > RELATIVE_DENOM_EPS {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn write_topology_summary(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "vertices,edges,faces,euler_characteristic,b0,b1,b2,boundary_edges,train_cycle_harmonic_pairing_rank,heldout_cycle_harmonic_pairing_rank"
    )?;
    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{}",
        result.topology_summary.vertex_count,
        result.topology_summary.edge_count,
        result.topology_summary.face_count,
        result.topology_summary.euler_characteristic,
        result.topology_summary.b0,
        result.topology_summary.b1,
        result.topology_summary.b2,
        result.topology_summary.boundary_edge_count,
        result.cycle_harmonic_pairing_rank,
        result.heldout_cycle_harmonic_pairing_rank
    )
}

fn write_metrics_summary(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,observation_count,l2_error,l2_error_absolute,heldout_nlpd,heldout_local_nlpd,heldout_loop_nlpd,heldout_harmonic_period_nlpd,exterior_derivative_error,exterior_derivative_error_absolute,exterior_derivative_truth_norm,codifferential_error,codifferential_metric_kind,codifferential_error_relative,codifferential_leakage,codifferential_error_absolute,codifferential_truth_norm,relative_circulation_error,relative_harmonic_period_error,relative_coexact_annular_error,relative_total_annular_error,mean_abs_circulation_error,max_abs_circulation_error"
    )?;
    for scenario in &result.scenarios {
        for metric in &scenario.metrics {
            writeln!(
                writer,
                "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                metric.scenario.as_str(),
                metric.model.as_str(),
                metric.observation_count,
                metric.l2_error,
                metric.l2_error_absolute,
                metric.heldout_nlpd,
                metric.heldout_local_nlpd,
                metric.heldout_loop_nlpd,
                metric.heldout_harmonic_period_nlpd,
                metric.exterior_derivative_error,
                metric.exterior_derivative_error_absolute,
                metric.exterior_derivative_truth_norm,
                metric.codifferential_error,
                metric.codifferential_metric_kind.as_str(),
                metric.codifferential_error_relative,
                metric.codifferential_leakage,
                metric.codifferential_error_absolute,
                metric.codifferential_truth_norm,
                metric.relative_circulation_error,
                metric.relative_harmonic_period_error,
                metric.relative_coexact_annular_error,
                metric.relative_total_annular_error,
                metric.mean_abs_circulation_error,
                metric.max_abs_circulation_error
            )?;
        }
    }
    Ok(())
}

fn write_period_summary(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,hole_index,hole_name,truth_period,posterior_mean,posterior_std,posterior_lower_95,posterior_upper_95,residual,abs_residual_over_std,covered_95"
    )?;
    for scenario in &result.scenarios {
        for period in &scenario.period_summaries {
            writeln!(
                writer,
                "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{}",
                period.scenario.as_str(),
                period.model.as_str(),
                period.hole_index,
                period.hole_name,
                period.truth_period,
                period.posterior_mean,
                period.posterior_std,
                period.posterior_lower_95,
                period.posterior_upper_95,
                period.residual,
                period.abs_residual_over_std,
                period.covered_95
            )?;
        }
    }
    Ok(())
}

fn write_loop_functional_summary(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,functional,hole_index,hole_name,truth_value,posterior_mean,posterior_std,posterior_lower_95,posterior_upper_95,residual,abs_residual_over_std,covered_95"
    )?;
    for scenario in &result.scenarios {
        for summary in &scenario.loop_functional_summaries {
            writeln!(
                writer,
                "{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{}",
                summary.scenario.as_str(),
                summary.model.as_str(),
                summary.functional.as_str(),
                summary.hole_index,
                summary.hole_name,
                summary.truth_value,
                summary.posterior_mean,
                summary.posterior_std,
                summary.posterior_lower_95,
                summary.posterior_upper_95,
                summary.residual,
                summary.abs_residual_over_std,
                summary.covered_95
            )?;
        }
    }
    Ok(())
}

fn write_field_coverage_summary(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,subset,edge_count,weight_sum,coverage_95,mass_weighted_coverage_95,mean_abs_z,rms_z,p95_abs_z,mean_posterior_std,mass_weighted_mean_posterior_std,latent_nlpd"
    )?;
    for scenario in &result.scenarios {
        for summary in &scenario.field_coverage_summaries {
            writeln!(
                writer,
                "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                summary.scenario.as_str(),
                summary.model.as_str(),
                summary.subset.as_str(),
                summary.edge_count,
                summary.weight_sum,
                summary.coverage_95,
                summary.mass_weighted_coverage_95,
                summary.mean_abs_z,
                summary.rms_z,
                summary.p95_abs_z,
                summary.mean_posterior_std,
                summary.mass_weighted_mean_posterior_std,
                summary.latent_nlpd
            )?;
        }
    }
    Ok(())
}

fn write_branch_recovery_summary(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,branch,truth_mass_norm,posterior_mass_norm,error_mass_norm,relative_error,mass_correlation"
    )?;
    for scenario in &result.scenarios {
        for summary in &scenario.branch_recovery_summaries {
            writeln!(
                writer,
                "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12}",
                summary.scenario.as_str(),
                summary.model.as_str(),
                summary.branch.as_str(),
                summary.truth_mass_norm,
                summary.posterior_mass_norm,
                summary.error_mass_norm,
                summary.relative_error,
                summary.mass_correlation
            )?;
        }
    }
    Ok(())
}

fn write_spectral_branch_diagnostics(
    result: &PlanarHolesFlowResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,branch,requested_mode_count,actual_mode_count,unnormalized_expected_m1_energy_sq,target_expected_m1_energy_sq,normalization_scale,expected_m1_energy_sq,truth_mass_norm,projected_truth_mass_norm,projection_relative_error,projected_truth_mahalanobis_norm"
    )?;
    for row in &result.spectral_branch_diagnostics {
        writeln!(
            writer,
            "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.model.as_str(),
            row.branch.as_str(),
            row.requested_mode_count,
            row.actual_mode_count,
            row.unnormalized_expected_m1_energy,
            row.target_expected_m1_energy,
            row.normalization_scale,
            row.expected_m1_energy,
            row.truth_mass_norm,
            row.projected_truth_mass_norm,
            row.projection_relative_error,
            row.projected_truth_mahalanobis_norm
        )?;
    }
    Ok(())
}

fn write_spectral_energy_diagnostics(
    result: &PlanarHolesFlowResult,
    path: PathBuf,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,branch,mode_index,cumulative_projected_energy,cumulative_projected_energy_fraction,cumulative_prior_energy,cumulative_prior_energy_fraction"
    )?;
    for row in &result.spectral_energy_diagnostics {
        writeln!(
            writer,
            "{},{},{},{:.12},{:.12},{:.12},{:.12}",
            row.model.as_str(),
            row.branch.as_str(),
            row.mode_index,
            row.cumulative_projected_energy,
            row.cumulative_projected_energy_fraction,
            row.cumulative_prior_energy,
            row.cumulative_prior_energy_fraction
        )?;
    }
    Ok(())
}

fn write_boundary_diagnostics(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mass = result_metric_mass(result);
    let mass_weights = positive_diagonal(&mass, result.truth.len());
    let boundary_edges = result
        .topology
        .boundary_subcomplex_simplices(1)
        .into_iter()
        .map(|simplex| simplex.kidx)
        .collect::<BTreeSet<_>>();
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,field,boundary_edge_count,boundary_lumped_energy,total_lumped_energy,boundary_lumped_energy_fraction"
    )?;
    for (field, values) in [
        ("truth", &result.truth),
        ("truth_exact", &result.truth_exact),
        ("truth_coexact", &result.truth_coexact),
        ("truth_harmonic", &result.truth_harmonic),
    ] {
        write_boundary_diagnostic_row(
            &mut writer,
            "truth",
            "truth",
            field,
            &boundary_edges,
            &mass_weights,
            values,
        )?;
    }
    for scenario in &result.scenarios {
        for posterior in &scenario.posteriors {
            write_boundary_diagnostic_row(
                &mut writer,
                scenario.scenario.as_str(),
                posterior.model.as_str(),
                "posterior_mean",
                &boundary_edges,
                &mass_weights,
                &posterior.posterior_mean,
            )?;
        }
    }
    Ok(())
}

fn write_boundary_diagnostic_row(
    writer: &mut BufWriter<File>,
    scenario: &str,
    model: &str,
    field: &str,
    boundary_edges: &BTreeSet<usize>,
    mass_weights: &[f64],
    values: &FeecVector,
) -> io::Result<()> {
    let mut boundary_energy = 0.0;
    let mut total_energy = 0.0;
    for edge in 0..values.len() {
        let energy = mass_weights[edge] * values[edge] * values[edge];
        total_energy += energy;
        if boundary_edges.contains(&edge) {
            boundary_energy += energy;
        }
    }
    writeln!(
        writer,
        "{},{},{},{},{:.12},{:.12},{:.12}",
        scenario,
        model,
        field,
        boundary_edges.len(),
        boundary_energy,
        total_energy,
        relative_or_nan(boundary_energy, total_energy)
    )
}

fn write_heldout_predictions(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,kind,label,observed_value,truth_value,predictive_mean,predictive_variance,nlpd"
    )?;
    for scenario in &result.scenarios {
        for row in &scenario.heldout_predictions {
            writeln!(
                writer,
                "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12}",
                row.scenario.as_str(),
                row.model.as_str(),
                row.kind.as_str(),
                row.label,
                row.observed_value,
                row.truth_value,
                row.predictive_mean,
                row.predictive_variance,
                row.nlpd
            )?;
        }
    }
    Ok(())
}

fn write_branch_diagnostics(result: &PlanarHolesFlowResult, path: PathBuf) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "branch,mass_norm")?;
    writeln!(
        writer,
        "truth_exact,{:.12}",
        mass_norm(&result.truth_exact, &result_metric_mass(result))
    )?;
    writeln!(
        writer,
        "truth_coexact,{:.12}",
        mass_norm(&result.truth_coexact, &result_metric_mass(result))
    )?;
    writeln!(
        writer,
        "truth_harmonic,{:.12}",
        mass_norm(&result.truth_harmonic, &result_metric_mass(result))
    )?;
    for scenario in &result.scenarios {
        for posterior in &scenario.posteriors {
            for (branch, mean) in &posterior.branch_means {
                writeln!(
                    writer,
                    "{}_{}_{},{:.12}",
                    scenario.scenario.as_str(),
                    posterior.model.as_str(),
                    branch.as_str(),
                    mass_norm(mean, &result_metric_mass(result))
                )?;
            }
        }
    }
    Ok(())
}

fn result_metric_mass(result: &PlanarHolesFlowResult) -> FeecCsr {
    feg_infer::prior::matern::one_form::build_hodge_laplacian_1form(
        &result.topology,
        &result.metric,
    )
    .mass_u
}

fn write_truth_outputs(result: &PlanarHolesFlowResult, out_dir: &Path) -> io::Result<()> {
    visual_output::write_1form_vector_field(
        out_dir.join("truth_total.vtu"),
        &result.coords,
        &result.topology,
        &Cochain::new(1, result.truth.clone()),
        "truth_total",
    )?;
    visual_output::write_1form_vector_field(
        out_dir.join("truth_exact.vtu"),
        &result.coords,
        &result.topology,
        &Cochain::new(1, result.truth_exact.clone()),
        "truth_exact",
    )?;
    visual_output::write_1form_vector_field(
        out_dir.join("truth_coexact.vtu"),
        &result.coords,
        &result.topology,
        &Cochain::new(1, result.truth_coexact.clone()),
        "truth_coexact",
    )?;
    visual_output::write_1form_vector_field(
        out_dir.join("truth_harmonic.vtu"),
        &result.coords,
        &result.topology,
        &Cochain::new(1, result.truth_harmonic.clone()),
        "truth_harmonic",
    )
}

fn write_observation_outputs(result: &PlanarHolesFlowResult, out_dir: &Path) -> io::Result<()> {
    let mut local = FeecVector::zeros(result.topology.edges().len());
    for observation in &result.local_observations {
        for (edge, _) in &observation.entries {
            local[*edge] = 1.0;
        }
    }
    visual_output::write_1cochain_fields(
        out_dir.join("local_observations.vtu"),
        &result.coords,
        &result.topology,
        &[("local_observation_edge", &Cochain::new(1, local))],
    )?;
    let paths = result
        .cycles
        .iter()
        .map(|cycle| cycle.path_vertices.as_slice())
        .collect::<Vec<_>>();
    let cycle_index = result
        .cycles
        .iter()
        .map(|cycle| cycle.hole_index as f64)
        .collect::<Vec<_>>();
    visual_output::write_polyline_fields(
        out_dir.join("cycle_paths.vtu"),
        "planar holes training cycle paths",
        &result.coords,
        &paths,
        &[("hole_index", cycle_index.as_slice())],
    )?;
    let heldout_paths = result
        .heldout_cycles
        .iter()
        .map(|cycle| cycle.path_vertices.as_slice())
        .collect::<Vec<_>>();
    let heldout_cycle_index = result
        .heldout_cycles
        .iter()
        .map(|cycle| cycle.hole_index as f64)
        .collect::<Vec<_>>();
    visual_output::write_polyline_fields(
        out_dir.join("heldout_cycle_paths.vtu"),
        "planar holes held-out cycle paths",
        &result.coords,
        &heldout_paths,
        &[("hole_index", heldout_cycle_index.as_slice())],
    )
}

fn write_scenario_outputs(
    result: &PlanarHolesFlowResult,
    scenario: &PlanarHolesScenarioResult,
    out_dir: &Path,
) -> io::Result<()> {
    let scenario_dir = out_dir.join(scenario.scenario.as_str());
    fs::create_dir_all(&scenario_dir)?;
    for posterior in &scenario.posteriors {
        let model_dir = scenario_dir.join(posterior.model.as_str());
        fs::create_dir_all(&model_dir)?;
        visual_output::write_1form_vector_field(
            model_dir.join("posterior_mean.vtu"),
            &result.coords,
            &result.topology,
            &Cochain::new(1, posterior.posterior_mean.clone()),
            "posterior_mean",
        )?;
        let posterior_std = Cochain::new(1, posterior.field_coverage.posterior_std.clone());
        let abs_z_score = Cochain::new(1, posterior.field_coverage.abs_z_score.clone());
        let covered_95 = Cochain::new(1, posterior.field_coverage.covered_95.clone());
        let posterior_mean_error =
            Cochain::new(1, posterior.field_coverage.posterior_mean_error.clone());
        visual_output::write_1cochain_fields(
            model_dir.join("field_coverage_diagnostics.vtu"),
            &result.coords,
            &result.topology,
            &[
                ("posterior_std", &posterior_std),
                ("abs_z_score", &abs_z_score),
                ("covered_95", &covered_95),
                ("posterior_mean_error", &posterior_mean_error),
            ],
        )?;
        for (branch, mean) in &posterior.branch_means {
            visual_output::write_1form_vector_field(
                model_dir.join(format!("{}_posterior_mean.vtu", branch.as_str())),
                &result.coords,
                &result.topology,
                &Cochain::new(1, mean.clone()),
                &format!("{}_posterior_mean", branch.as_str()),
            )?;
        }
    }

    let hodge_periods = scenario
        .period_summaries
        .iter()
        .filter(|summary| summary.model == PlanarHolesModelKind::HodgeMatern)
        .collect::<Vec<_>>();
    if hodge_periods.len() == result.cycles.len() {
        let paths = result
            .heldout_cycles
            .iter()
            .map(|cycle| cycle.path_vertices.as_slice())
            .collect::<Vec<_>>();
        let std = hodge_periods
            .iter()
            .map(|summary| summary.posterior_std)
            .collect::<Vec<_>>();
        visual_output::write_polyline_fields(
            scenario_dir.join("hodge_harmonic_circulation_uncertainty.vtu"),
            "planar holes Hodge circulation uncertainty",
            &result.coords,
            &paths,
            &[("posterior_period_std", std.as_slice())],
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
mod tests {
    use super::*;

    #[cfg(feature = "heavy-tests")]
    fn gmsh_available() -> bool {
        Command::new("gmsh").arg("-version").output().is_ok()
    }

    #[test]
    fn planar_holes_rejects_nonpositive_branch_tau_scales() {
        let mut config = PlanarHolesFlowConfig::default();
        config.exact_tau_scale = 0.0;
        assert!(validate_config(&config).is_err());

        let mut config = PlanarHolesFlowConfig::default();
        config.coexact_tau_scale = -1.0;
        assert!(validate_config(&config).is_err());
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_coexact_transform_diagnostics_detect_sparse_defect() {
        if !gmsh_available() {
            eprintln!("skipping planar holes coexact transform test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_coexact_transform_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_coexact_transform_test.msh"),
            geo_path: out_dir.join("planar_holes_coexact_transform_test.geo"),
            force_mesh: true,
            mesh_size: 0.18,
            local_observation_count: 8,
            heldout_local_count: 4,
            ..PlanarHolesFlowConfig::default()
        };

        let all_diagnostics = run_planar_holes_coexact_transform_diagnostics_all(&config)
            .expect("coexact transform diagnostics should run on a small mesh");
        assert_eq!(all_diagnostics.len(), 2);
        assert!(all_diagnostics.iter().all(|diagnostics| {
            !diagnostics.sparse_inverse.is_empty()
                && diagnostics
                    .sparse_coexact_codifferential_leakage
                    .is_finite()
                && diagnostics
                    .sparse_exact_branch_mass_orthogonality
                    .is_finite()
                && diagnostics
                    .sparse_vs_exact_mass_transform_relative_m1_error
                    .is_finite()
                && diagnostics.sparse_coexact_rank > 0
                && diagnostics.sparse_coexact_rank == diagnostics.exact_mass_coexact_rank
                && (0.0..=1.0).contains(&diagnostics.principal_cosine_min)
                && (0.0..=1.0).contains(&diagnostics.principal_cosine_mean)
                && diagnostics.principal_angle_max_degrees.is_finite()
        }));
        let diagnostics = all_diagnostics
            .iter()
            .find(|diagnostics| diagnostics.sparse_inverse == "nc1_projected_sparse_inverse")
            .expect("NC1 projected diagnostic should be present");
        assert!(diagnostics
            .sparse_coexact_codifferential_leakage
            .is_finite());
        assert!(diagnostics
            .sparse_exact_branch_mass_orthogonality
            .is_finite());
        assert!(diagnostics
            .sparse_vs_exact_mass_transform_relative_m1_error
            .is_finite());
        assert!(
            diagnostics.exact_mass_coexact_codifferential_leakage < 1e-8,
            "exact-mass coexact transform should be co-closed, got {:.3e}",
            diagnostics.exact_mass_coexact_codifferential_leakage
        );
        assert!(
            diagnostics.exact_mass_exact_branch_mass_orthogonality < 1e-8,
            "exact-mass coexact transform should be M1-orthogonal to exact branch, got {:.3e}",
            diagnostics.exact_mass_exact_branch_mass_orthogonality
        );
        assert!(diagnostics.sparse_coexact_rank > 0);
        assert_eq!(
            diagnostics.sparse_coexact_rank,
            diagnostics.exact_mass_coexact_rank
        );
        assert!((0.0..=1.0).contains(&diagnostics.principal_cosine_min));
        assert!((0.0..=1.0).contains(&diagnostics.principal_cosine_mean));
        assert!(diagnostics.principal_angle_max_degrees.is_finite());

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_flow_outputs_topological_comparison() {
        if !gmsh_available() {
            eprintln!("skipping planar holes flow test because gmsh is unavailable");
            return;
        }
        let out_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/planar_holes_flow_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_test.msh"),
            geo_path: out_dir.join("planar_holes_test.geo"),
            force_mesh: true,
            mesh_size: 0.17,
            local_observation_count: 8,
            heldout_local_count: 4,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("planar holes Hodge flow should run on small mesh");
        assert_eq!(result.topology_summary.b1, 3);
        assert_eq!(result.cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.heldout_cycle_harmonic_pairing_rank, 3);
        let train_pairing = &result.cycle_observation_matrix * &result.harmonic_basis;
        for row in 0..train_pairing.nrows() {
            for col in 0..train_pairing.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (train_pairing[(row, col)] - expected).abs() <= 1e-6,
                    "canonical train cycle-harmonic pairing mismatch at ({row},{col})"
                );
            }
        }
        assert!(!matching_cycle_rows_identical(
            &result.cycle_observation_matrix,
            &result.heldout_cycle_observation_matrix
        ));
        assert!(result
            .cycles
            .iter()
            .chain(result.heldout_cycles.iter())
            .all(|cycle| cycle.closure_residual_l1 < 1e-10));
        let mut cycle_edges = observation_edge_set(&result.cycle_observation_matrix);
        cycle_edges.extend(observation_edge_set(
            &result.heldout_cycle_observation_matrix,
        ));
        assert!(result.local_observations.iter().all(|observation| {
            observation
                .entries
                .iter()
                .all(|(edge, _)| !cycle_edges.contains(edge))
        }));
        assert!(result
            .heldout_observations
            .iter()
            .any(|observation| { observation.kind == PlanarHolesObservationKind::HeldoutLoop }));
        assert!(result.heldout_observations.iter().any(|observation| {
            observation.kind == PlanarHolesObservationKind::HeldoutHarmonicPeriod
        }));
        assert_eq!(result.scenarios.len(), 3);
        for scenario in &result.scenarios {
            assert_eq!(scenario.metrics.len(), 5);
            for metric in &scenario.metrics {
                assert!(metric.l2_error.is_finite());
                assert!(metric.l2_error_absolute.is_finite());
                assert!(metric.heldout_nlpd.is_finite());
                assert!(metric.heldout_local_nlpd.is_finite());
                assert!(metric.heldout_loop_nlpd.is_finite());
                assert!(metric.heldout_harmonic_period_nlpd.is_finite());
                assert!(metric.exterior_derivative_error.is_finite());
                assert!(metric.exterior_derivative_error_absolute.is_finite());
                assert!(metric.codifferential_error.is_finite());
                assert!(metric.codifferential_error_absolute.is_finite());
                assert!(metric.relative_circulation_error.is_finite());
                assert!(metric.relative_total_annular_error.is_finite());
                if matches!(
                    metric.model,
                    PlanarHolesModelKind::HodgeMatern
                        | PlanarHolesModelKind::IncompressibleHodgeMatern
                ) {
                    assert!(metric.relative_harmonic_period_error.is_finite());
                    assert!(metric.relative_coexact_annular_error.is_finite());
                }
                assert!(metric.mean_abs_circulation_error.is_finite());
                assert!(metric.max_abs_circulation_error.is_finite());
            }
            assert_eq!(scenario.period_summaries.len(), 15);
            assert!(scenario
                .period_summaries
                .iter()
                .all(|summary| summary.posterior_std.is_finite()));
            assert!(
                scenario
                    .loop_functional_summaries
                    .iter()
                    .any(|summary| summary.functional
                        == PlanarHolesLoopFunctionalKind::HarmonicPeriod)
            );
            assert!(
                scenario
                    .loop_functional_summaries
                    .iter()
                    .any(|summary| summary.functional
                        == PlanarHolesLoopFunctionalKind::CoexactAnnular)
            );
            assert!(scenario
                .loop_functional_summaries
                .iter()
                .all(|summary| summary.posterior_std.is_finite()));
            assert_eq!(scenario.field_coverage_summaries.len(), 10);
            for summary in &scenario.field_coverage_summaries {
                assert!(summary.edge_count > 0);
                assert!(summary.weight_sum.is_finite());
                assert!(summary.coverage_95.is_finite());
                assert!((0.0..=1.0).contains(&summary.coverage_95));
                assert!(summary.mass_weighted_coverage_95.is_finite());
                assert!((0.0..=1.0).contains(&summary.mass_weighted_coverage_95));
                assert!(summary.mean_abs_z.is_finite());
                assert!(summary.rms_z.is_finite());
                assert!(summary.p95_abs_z.is_finite());
                assert!(summary.mean_posterior_std.is_finite());
                assert!(summary.mass_weighted_mean_posterior_std.is_finite());
                assert!(summary.latent_nlpd.is_finite());
                if summary.subset == PlanarHolesFieldCoverageSubset::HeldoutLocalEdges {
                    assert_eq!(summary.edge_count, config.heldout_local_count);
                }
            }
            assert_eq!(scenario.branch_recovery_summaries.len(), 5);
            assert!(scenario.branch_recovery_summaries.iter().all(|summary| {
                summary.truth_mass_norm.is_finite()
                    && summary.posterior_mass_norm.is_finite()
                    && summary.error_mass_norm.is_finite()
                    && (summary.truth_mass_norm <= RELATIVE_DENOM_EPS
                        || summary.relative_error.is_finite())
                    && (summary.truth_mass_norm <= RELATIVE_DENOM_EPS
                        || summary.posterior_mass_norm <= RELATIVE_DENOM_EPS
                        || summary.mass_correlation.is_finite())
            }));
        }

        let local_only = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalOnly)
            .unwrap();
        let local_plus = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalPlusLoops)
            .unwrap();
        let local_plus_harmonic = result
            .scenarios
            .iter()
            .find(|scenario| {
                scenario.scenario == PlanarHolesObservationScenario::LocalPlusHarmonicPeriods
            })
            .unwrap();
        let local_only_std = mean(local_only.period_summaries.iter().filter_map(|summary| {
            (summary.model == PlanarHolesModelKind::HodgeMatern).then_some(summary.posterior_std)
        }));
        let local_plus_std = mean(local_plus.period_summaries.iter().filter_map(|summary| {
            (summary.model == PlanarHolesModelKind::HodgeMatern).then_some(summary.posterior_std)
        }));
        let local_plus_harmonic_std = mean(local_plus_harmonic.period_summaries.iter().filter_map(
            |summary| {
                (summary.model == PlanarHolesModelKind::HodgeMatern)
                    .then_some(summary.posterior_std)
            },
        ));
        assert!(
            local_plus_std < local_only_std,
            "loop observations should reduce Hodge period uncertainty"
        );
        assert!(
            local_plus_harmonic_std < local_only_std,
            "harmonic period observations should reduce Hodge period uncertainty"
        );

        write_planar_holes_hodge_flow_outputs(&result, &out_dir)
            .expect("planar holes outputs should write");
        assert!(out_dir.join("metrics_summary.csv").is_file());
        assert!(out_dir.join("period_summary.csv").is_file());
        assert!(out_dir.join("loop_functional_summary.csv").is_file());
        assert!(out_dir.join("field_coverage_summary.csv").is_file());
        assert!(out_dir.join("branch_recovery_summary.csv").is_file());
        assert!(out_dir.join("heldout_predictions.csv").is_file());
        assert!(out_dir.join("topology_summary.csv").is_file());
        let metrics_csv =
            fs::read_to_string(out_dir.join("metrics_summary.csv")).expect("metrics csv readable");
        assert!(metrics_csv.contains("l2_error_absolute"));
        assert!(metrics_csv.contains("heldout_local_nlpd"));
        assert!(metrics_csv.contains("heldout_loop_nlpd"));
        assert!(metrics_csv.contains("heldout_harmonic_period_nlpd"));
        assert!(metrics_csv.contains("codifferential_metric_kind"));
        assert!(metrics_csv.contains("relative_harmonic_period_error"));
        let loop_csv = fs::read_to_string(out_dir.join("loop_functional_summary.csv"))
            .expect("loop functional csv readable");
        assert!(loop_csv.contains("harmonic_period"));
        assert!(loop_csv.contains("coexact_annular"));
        let field_csv = fs::read_to_string(out_dir.join("field_coverage_summary.csv"))
            .expect("field coverage csv readable");
        assert!(field_csv.contains("mass_weighted_coverage_95"));
        assert!(field_csv.contains("heldout_local_edges"));
        let branch_recovery_csv = fs::read_to_string(out_dir.join("branch_recovery_summary.csv"))
            .expect("branch recovery csv readable");
        assert!(branch_recovery_csv.contains("coexact"));
        assert!(branch_recovery_csv.contains("mass_correlation"));
        assert!(out_dir.join("truth_total.vtu").is_file());
        assert!(out_dir.join("cycle_paths.vtu").is_file());
        assert!(out_dir.join("heldout_cycle_paths.vtu").is_file());
        assert!(out_dir
            .join("local_plus_loops/feec_hodge_matern/posterior_mean.vtu")
            .is_file());
        assert!(out_dir
            .join("local_plus_loops/feec_hodge_matern/field_coverage_diagnostics.vtu")
            .is_file());
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_spectral_hodge_models_produce_finite_metrics() {
        if !gmsh_available() {
            eprintln!("skipping planar holes spectral test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_spectral_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_spectral_test.msh"),
            geo_path: out_dir.join("planar_holes_spectral_test.geo"),
            force_mesh: true,
            mesh_size: 0.18,
            local_observation_count: 8,
            heldout_local_count: 4,
            include_spectral_hodge_model: true,
            include_spectral_incompressible_hodge_model: true,
            spectral_exact_mode_count: 8,
            spectral_coexact_mode_count: 8,
            spectral_harmonic_mode_count: 3,
            ..PlanarHolesFlowConfig::default()
        };

        let result =
            run_planar_holes_hodge_flow(&config).expect("spectral planar holes run should succeed");
        assert!(!result.spectral_branch_diagnostics.is_empty());
        assert!(!result.spectral_energy_diagnostics.is_empty());
        assert!(result.spectral_branch_diagnostics.iter().all(|row| {
            row.actual_mode_count > 0
                && row.expected_m1_energy.is_finite()
                && row.projection_relative_error.is_finite()
        }));
        for scenario in &result.scenarios {
            assert_eq!(scenario.metrics.len(), 7);
            for model in [
                PlanarHolesModelKind::SpectralHodgeGp,
                PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
            ] {
                let metric = scenario
                    .metrics
                    .iter()
                    .find(|metric| metric.model == model)
                    .expect("spectral model metric should be present");
                assert!(metric.l2_error.is_finite());
                assert!(metric.heldout_local_nlpd.is_finite());
                assert!(metric.codifferential_error.is_finite());
                assert!(metric.relative_circulation_error.is_finite());
                assert!(metric.relative_harmonic_period_error.is_finite());
                assert!(metric.relative_coexact_annular_error.is_finite());
                let field = scenario
                    .field_coverage_summaries
                    .iter()
                    .find(|summary| {
                        summary.model == model
                            && summary.subset == PlanarHolesFieldCoverageSubset::AllEdges
                    })
                    .expect("spectral field coverage summary should be present");
                assert!(field.coverage_95.is_finite());
                assert!((0.0..=1.0).contains(&field.coverage_95));
            }
        }

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_normalized_spectral_dense_oracle_recovers_matched_coexact_truth() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes matched spectral oracle test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_spectral_oracle_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_spectral_oracle_test.msh"),
            geo_path: out_dir.join("planar_holes_spectral_oracle_test.geo"),
            force_mesh: true,
            mesh_size: 0.18,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::SpectralGp,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::SpectralGp,
            truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
            local_observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
            include_incompressible_hodge_model: false,
            include_spectral_hodge_model: false,
            include_spectral_incompressible_hodge_model: true,
            spectral_branch_energy_normalization: true,
            spectral_exact_mode_count: 10,
            spectral_coexact_mode_count: 10,
            spectral_harmonic_mode_count: 3,
            local_observation_count: 1,
            heldout_local_count: 1,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("matched normalized spectral oracle should run");
        write_planar_holes_hodge_flow_outputs(&result, &out_dir)
            .expect("spectral oracle outputs should write");
        let local_only = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalOnly)
            .unwrap();
        let spectral = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::SpectralIncompressibleHodgeGp)
            .unwrap();
        assert!(
            spectral.l2_error < 1e-4,
            "matched dense normalized spectral oracle E1 too large: {:.3e}",
            spectral.l2_error
        );
        let coexact_diag = result
            .spectral_branch_diagnostics
            .iter()
            .find(|row| {
                row.model == PlanarHolesModelKind::SpectralIncompressibleHodgeGp
                    && row.branch == HodgeBranchKind::Coexact
            })
            .expect("coexact spectral diagnostics should be present");
        assert!(coexact_diag.projection_relative_error < 1e-10);
        assert!((coexact_diag.expected_m1_energy - 1.0).abs() < 1e-10);
        let spectral_csv = fs::read_to_string(out_dir.join("spectral_branch_diagnostics.csv"))
            .expect("spectral diagnostics csv should be readable");
        assert!(spectral_csv.contains("projection_relative_error"));
        assert!(out_dir.join("boundary_diagnostics.csv").is_file());

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_streamfunction_truth_is_discrete_coclosed_after_projection() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes streamfunction truth test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_streamfunction_truth_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_streamfunction_test.msh"),
            geo_path: out_dir.join("planar_holes_streamfunction_test.geo"),
            force_mesh: true,
            mesh_size: 0.18,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
            local_observation_count: 8,
            heldout_local_count: 4,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("streamfunction truth should run on small mesh");
        let leakage = codifferential_leakage(
            &result.topology,
            &result.metric,
            &result_metric_mass(&result),
            &result.truth,
        )
        .expect("streamfunction leakage should compute");
        assert!(
            leakage < 1e-6,
            "projected streamfunction truth should be co-closed; leakage={leakage:.3e}"
        );
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_spectral_truth_compatibility_rows_are_finite_and_monotone() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes spectral compatibility test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_spectral_compatibility_test");
        let _ = fs::remove_dir_all(&out_dir);
        let base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_spectral_compatibility_test.msh"),
            geo_path: out_dir.join("planar_holes_spectral_compatibility_test.geo"),
            force_mesh: true,
            mesh_size: 0.20,
            ..PlanarHolesFlowConfig::default()
        };
        let config = PlanarHolesSpectralTruthCompatibilityConfig {
            base,
            mode_stages: vec![4, 8],
            include_max_available: false,
            spectral_truth_mode_count: 8,
            families: vec![
                PlanarHolesSpectralTruthFamily::SpectralCoexact,
                PlanarHolesSpectralTruthFamily::SparseAnchorCoexact,
                PlanarHolesSpectralTruthFamily::ExactMassCoexactKappa8,
                PlanarHolesSpectralTruthFamily::StreamfunctionDirichlet,
            ],
            boundary_conditions: vec![PlanarHolesSpectralBoundaryCondition::Free],
        };
        let rows = run_planar_holes_spectral_truth_compatibility(&config)
            .expect("spectral truth compatibility should run");
        assert_eq!(rows.len(), 8);
        assert!(rows.iter().all(|row| {
            row.actual_mode_count > 0
                && row.truth_mass_norm.is_finite()
                && row.projected_truth_mass_norm.is_finite()
                && row.projected_energy_fraction.is_finite()
                && row.projection_relative_error.is_finite()
                && row.projected_truth_mahalanobis_norm.is_finite()
                && row.codifferential_leakage.is_finite()
                && row.boundary_lumped_energy_fraction.is_finite()
                && row.expected_m1_energy_sq.is_finite()
        }));
        for family in [
            PlanarHolesSpectralTruthFamily::SpectralCoexact,
            PlanarHolesSpectralTruthFamily::SparseAnchorCoexact,
        ] {
            let low = rows
                .iter()
                .find(|row| row.truth_family == family && row.requested_mode_count == 4)
                .unwrap();
            let high = rows
                .iter()
                .find(|row| row.truth_family == family && row.requested_mode_count == 8)
                .unwrap();
            assert!(
                high.projection_relative_error <= low.projection_relative_error + 1e-8,
                "{} projection error should not increase: low={} high={}",
                family.as_str(),
                low.projection_relative_error,
                high.projection_relative_error
            );
        }
        let csv_path = out_dir.join("spectral_truth_compatibility.csv");
        write_planar_holes_spectral_truth_compatibility(&rows, &csv_path)
            .expect("compatibility csv should write");
        let csv = fs::read_to_string(csv_path).expect("compatibility csv should be readable");
        assert!(csv.contains("expected_m1_energy_sq"));
        assert!(csv.contains("streamfunction_dirichlet"));
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_prior_equivalence_reports_trace_matched_covariances() {
        if !gmsh_available() {
            eprintln!("skipping planar holes prior equivalence test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_prior_equivalence_test");
        let _ = fs::remove_dir_all(&out_dir);
        let base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_prior_equivalence_test.msh"),
            geo_path: out_dir.join("planar_holes_prior_equivalence_test.geo"),
            force_mesh: true,
            mesh_size: 0.24,
            ..PlanarHolesFlowConfig::default()
        };
        let config = PlanarHolesPriorEquivalenceConfig {
            base,
            mode_counts: vec![4, 8],
            include_max_available: false,
            include_exact_lower_mass: true,
            dense_eigen_dimension_cap: 10_000,
            ..PlanarHolesPriorEquivalenceConfig::default()
        };
        let rows = run_planar_holes_prior_equivalence(&config)
            .expect("prior equivalence diagnostic should run");
        assert_eq!(rows.len(), 16);
        assert!(rows.iter().all(|row| {
            row.actual_mode_count > 0
                && row.spectral_expected_m1_energy.is_finite()
                && row.gmrf_expected_m1_energy.is_finite()
                && row.required_tau_scale_to_match_spectral_trace.is_finite()
                && row.m1_frobenius_relative_error.is_finite()
                && row.source_eigen_relative_l2_error.is_finite()
        }));
        for row in rows.iter().filter(|row| {
            matches!(
                row.gmrf_variant,
                PlanarHolesPriorEquivalenceGmrfVariant::SparseLowerTraceMatched
                    | PlanarHolesPriorEquivalenceGmrfVariant::ExactLowerTraceMatched
            )
        }) {
            assert!(
                row.trace_relative_error < 1e-8,
                "trace-matched row should match spectral trace: {:?}",
                row
            );
        }
        assert!(rows.iter().any(|row| {
            row.gmrf_variant == PlanarHolesPriorEquivalenceGmrfVariant::ExactLower
                && row.source_eigen_count > 0
        }));
        let csv_path = out_dir.join("prior_equivalence.csv");
        write_planar_holes_prior_equivalence(&rows, &csv_path)
            .expect("prior equivalence csv should write");
        let csv = fs::read_to_string(csv_path).expect("prior equivalence csv should be readable");
        assert!(csv.contains("m1_frobenius_relative_error"));
        assert!(csv.contains("exact_mass_gmrf_exact_lower_trace_matched"));
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_exact_prior_equivalence_reports_trace_matched_covariances() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes exact prior equivalence test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_exact_prior_equivalence_test");
        let _ = fs::remove_dir_all(&out_dir);
        let base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_exact_prior_equivalence_test.msh"),
            geo_path: out_dir.join("planar_holes_exact_prior_equivalence_test.geo"),
            force_mesh: true,
            mesh_size: 0.24,
            ..PlanarHolesFlowConfig::default()
        };
        let config = PlanarHolesExactPriorEquivalenceConfig {
            base,
            mode_counts: vec![4, 8],
            include_max_available: false,
            dense_eigen_dimension_cap: 10_000,
            ..PlanarHolesExactPriorEquivalenceConfig::default()
        };
        let rows = run_planar_holes_exact_prior_equivalence(&config)
            .expect("exact prior equivalence diagnostic should run");
        assert_eq!(rows.len(), 24);
        assert!(rows.iter().all(|row| {
            row.actual_mode_count > 0
                && row.spectral_expected_m1_energy.is_finite()
                && row.gmrf_expected_m1_energy.is_finite()
                && row.required_tau_scale_to_match_spectral_trace.is_finite()
                && row.m1_frobenius_relative_error.is_finite()
                && row.source_eigen_relative_l2_error.is_finite()
        }));
        for row in rows.iter().filter(|row| {
            matches!(
                row.gmrf_variant,
                PlanarHolesPriorEquivalenceGmrfVariant::ExactLumpedTraceMatched
                    | PlanarHolesPriorEquivalenceGmrfVariant::ExactDenseTraceMatched
                    | PlanarHolesPriorEquivalenceGmrfVariant::ExactOrdinaryPotentialDenseTraceMatched
            )
        }) {
            assert!(
                row.trace_relative_error < 1e-8,
                "trace-matched exact row should match spectral trace: {:?}",
                row
            );
        }
        assert!(rows.iter().any(|row| {
            row.gmrf_variant == PlanarHolesPriorEquivalenceGmrfVariant::ExactDense
                && row.source_eigen_count > 0
        }));
        let csv_path = out_dir.join("exact_prior_equivalence.csv");
        write_planar_holes_prior_equivalence(&rows, &csv_path)
            .expect("exact prior equivalence csv should write");
        let csv =
            fs::read_to_string(csv_path).expect("exact prior equivalence csv should be readable");
        assert!(csv.contains("exact_gmrf_dense_0form_trace_matched"));
        assert!(csv.contains("exact_ordinary_potential_dense_0form"));
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_sensor_design_sweep_produces_finite_rows() {
        if !gmsh_available() {
            eprintln!("skipping planar holes sensor sweep test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_sensor_sweep_test");
        let _ = fs::remove_dir_all(&out_dir);
        let base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_sensor_sweep_test.msh"),
            geo_path: out_dir.join("planar_holes_sensor_sweep_test.geo"),
            force_mesh: true,
            mesh_size: 0.20,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.7,
            coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
            sample_observation_noise: false,
            include_incompressible_hodge_model: true,
            include_exact_mass_incompressible_hodge_model: true,
            include_sparse_lower_trace_matched_incompressible_hodge_model: true,
            include_exact_lower_incompressible_hodge_model: true,
            include_exact_lower_trace_matched_incompressible_hodge_model: true,
            include_spectral_hodge_model: false,
            include_spectral_incompressible_hodge_model: true,
            spectral_branch_energy_normalization: true,
            spectral_exact_mode_count: 6,
            spectral_coexact_mode_count: 6,
            spectral_harmonic_mode_count: 3,
            spectral_exact_expected_m1_energy: 1.0,
            spectral_coexact_expected_m1_energy: 1.0,
            spectral_harmonic_expected_m1_energy: 0.49,
            ..PlanarHolesFlowConfig::default()
        };
        let config = PlanarHolesSensorDesignSweepConfig {
            base,
            total_observation_budget: 18,
            heldout_local_count: 4,
            heldout_interior_loop_count: 3,
            heldout_long_path_count: 3,
            designs: vec![
                PlanarHolesSensorDesignKind::SparseEdges,
                PlanarHolesSensorDesignKind::Hybrid,
            ],
            ..PlanarHolesSensorDesignSweepConfig::default()
        };
        let rows = run_planar_holes_sensor_design_sweep(&config)
            .expect("sensor design sweep should run on a small mesh");
        assert!(!rows.is_empty());
        assert!(rows
            .iter()
            .any(|row| row.model == PlanarHolesModelKind::ExactMassIncompressibleHodgeMatern));
        assert!(rows.iter().any(|row| {
            row.model == PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
        }));
        assert!(rows
            .iter()
            .any(|row| { row.model == PlanarHolesModelKind::ExactLowerIncompressibleHodgeMatern }));
        assert!(rows.iter().any(|row| {
            row.model == PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
        }));
        assert!(rows.iter().all(|row| {
            row.observation_count > 0
                && row.l2_error.is_finite()
                && row.heldout_local_nlpd.is_finite()
                && row.heldout_interior_loop_nlpd.is_finite()
                && row.heldout_long_path_nlpd.is_finite()
                && row.codifferential_leakage.is_finite()
                && row.all_edge_coverage_95.is_finite()
                && row.heldout_edge_coverage_95.is_finite()
        }));
        let csv_path = out_dir.join("sensor_design_sweep.csv");
        write_planar_holes_sensor_design_sweep(&rows, &csv_path)
            .expect("sensor sweep csv should write");
        let csv = fs::read_to_string(csv_path).expect("sensor sweep csv should be readable");
        assert!(csv.contains("edges_small_interior_loops") || csv.contains("hybrid"));
        assert!(csv.contains("model_coexact_tau_scale"));
        assert!(csv.contains("feec_sparse_lower_trace_matched_coexact_harmonic_matern"));
        assert!(csv.contains("feec_exact_lower_coexact_harmonic_matern"));
        assert!(csv.contains("feec_exact_lower_trace_matched_coexact_harmonic_matern"));
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_topology_vs_naive_gp_produces_finite_outputs() {
        if !gmsh_available() {
            eprintln!("skipping planar holes topology-vs-naive test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_topology_vs_naive_test");
        let _ = fs::remove_dir_all(&out_dir);
        let mut base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_topology_vs_naive_test.msh"),
            geo_path: out_dir.join("planar_holes_topology_vs_naive_test.geo"),
            force_mesh: true,
            mesh_size: 0.22,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.7,
            coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            sample_observation_noise: false,
            ..PlanarHolesFlowConfig::default()
        };
        base.heldout_loop_radius_offset = 0.075;
        let config = PlanarHolesTopologyVsNaiveGpConfig {
            base,
            total_observation_budget: 18,
            validation_local_count: 4,
            validation_interior_loop_count: 2,
            validation_long_path_count: 2,
            heldout_local_count: 4,
            heldout_interior_loop_count: 2,
            heldout_long_path_count: 2,
            hodge_kappas: vec![2.0],
            hodge_taus: vec![1.0],
            naive_kappas: vec![2.0],
            naive_taus: vec![1.0],
            observation_variance_scales: vec![1.0],
            ..PlanarHolesTopologyVsNaiveGpConfig::default()
        };
        let result = run_planar_holes_topology_vs_naive_gp(&config)
            .expect("topology-vs-naive experiment should run on a small mesh");
        assert_eq!(result.topology_summary.b1, 3);
        assert_eq!(result.train_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.validation_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.heldout_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.rows.len(), 2);
        assert!(result
            .rows
            .iter()
            .any(|row| row.model == PlanarHolesModelKind::NondecomposedFeec));
        assert!(result
            .rows
            .iter()
            .any(|row| row.model == PlanarHolesModelKind::NaiveEuclideanVectorMatern));
        assert!(result.rows.iter().all(|row| {
            row.validation_nlpd.is_finite()
                && row.l2_error.is_finite()
                && row.heldout_nlpd.is_finite()
                && row.heldout_local_nlpd.is_finite()
                && row.heldout_loop_nlpd.is_finite()
                && row.heldout_interior_loop_nlpd.is_finite()
                && row.heldout_long_path_nlpd.is_finite()
                && row.exterior_derivative_error.is_finite()
                && row.codifferential_leakage.is_finite()
                && (0.0..=1.0).contains(&row.all_edge_coverage_95)
                && (0.0..=1.0).contains(&row.heldout_edge_coverage_95)
        }));
        assert!(result.tuning_rows.iter().all(|row| {
            row.validation_nlpd.is_finite()
                && row.kappa.is_finite()
                && row.tau.is_finite()
                && row.observation_variance_scale.is_finite()
        }));
        assert!(!result.calibration_rows.is_empty());
        assert!(result.calibration_rows.iter().all(|row| {
            row.variance_multiplier.is_finite()
                && row.variance_multiplier > 0.0
                && row.raw_nlpd.is_finite()
                && row.calibrated_nlpd.is_finite()
        }));
        write_planar_holes_topology_vs_naive_gp_outputs(&result, &out_dir)
            .expect("topology-vs-naive outputs should write");
        for file_name in [
            "metrics_summary.csv",
            "validation_summary.csv",
            "calibration_summary.csv",
            "heldout_predictions.csv",
            "field_coverage_summary.csv",
            "topology_summary.csv",
        ] {
            assert!(
                out_dir.join(file_name).is_file(),
                "{file_name} should be written"
            );
        }
        let metrics_csv =
            fs::read_to_string(out_dir.join("metrics_summary.csv")).expect("csv readable");
        assert!(metrics_csv.contains("naive_euclidean_vector_matern"));
        assert!(metrics_csv.contains("nondecomposed_feec_hodge_matern"));
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_barrier_challenge_produces_finite_outputs() {
        if !gmsh_available() {
            eprintln!("skipping planar holes barrier challenge test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_barrier_challenge_test");
        let _ = fs::remove_dir_all(&out_dir);
        let mut base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_barrier_challenge_test.msh"),
            geo_path: out_dir.join("planar_holes_barrier_challenge_test.geo"),
            force_mesh: true,
            mesh_size: 0.09,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 2.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            sample_observation_noise: false,
            ..PlanarHolesFlowConfig::default()
        };
        base.heldout_loop_radius_offset = 0.025;
        base.spectral_branch_energy_normalization = true;
        base.spectral_coexact_expected_m1_energy = 1.0;
        base.spectral_harmonic_expected_m1_energy = 4.0;
        base.spectral_harmonic_mode_count = 3;
        base.spectral_coexact_mode_count = 8;
        let config = PlanarHolesTopologyVsNaiveGpConfig {
            base,
            total_observation_budget: 12,
            validation_local_count: 4,
            validation_interior_loop_count: 1,
            validation_long_path_count: 2,
            heldout_local_count: 4,
            heldout_interior_loop_count: 1,
            heldout_long_path_count: 2,
            hodge_kappas: vec![3.0],
            hodge_taus: vec![0.05],
            naive_kappas: vec![6.0],
            naive_taus: vec![1.0],
            observation_variance_scales: vec![1.0],
            ..PlanarHolesTopologyVsNaiveGpConfig::default()
        };
        let result = run_planar_holes_barrier_topology_vs_naive_gp(&config)
            .expect("barrier challenge should run on a small mesh");
        assert_eq!(result.base.topology_summary.b1, 3);
        assert_eq!(result.base.train_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.base.validation_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.base.heldout_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.rows.len(), 5);
        assert!(result.base.rows.iter().any(|row| row.model
            == PlanarHolesModelKind::SpectralIncompressibleHodgeGp
            && row.spectral_coexact_actual_modes > 0
            && row.spectral_harmonic_actual_modes == 3));
        assert!(result.base.rows.iter().any(|row| row.model
            == PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
            && row.model_coexact_tau_scale.is_finite()
            && row.model_coexact_tau_scale > 0.0));
        assert!(!result.base.calibration_rows.is_empty());
        assert!(result.base.calibration_rows.iter().all(|row| {
            row.variance_multiplier.is_finite()
                && row.variance_multiplier > 0.0
                && row.raw_nlpd.is_finite()
                && row.calibrated_nlpd.is_finite()
                && (0.0..=1.0).contains(&row.raw_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_coverage_95)
        }));
        assert!(result.rows.iter().all(|row| {
            row.train_left_edge_count > 0
                && row.heldout_right_edge_count > 0
                && row.cross_barrier_local_nlpd.is_finite()
                && row.calibrated_cross_barrier_local_nlpd.is_finite()
                && row.cross_barrier_local_relative_error.is_finite()
                && row.barrier_long_path_nlpd.is_finite()
                && row.calibrated_barrier_long_path_nlpd.is_finite()
                && row.barrier_long_path_relative_error.is_finite()
                && row.hole_loop_nlpd.is_finite()
                && row.calibrated_hole_loop_nlpd.is_finite()
                && row.hole_loop_relative_error.is_finite()
                && (0.0..=1.0).contains(&row.cross_barrier_local_coverage_95)
                && (0.0..=1.0).contains(&row.barrier_long_path_coverage_95)
                && (0.0..=1.0).contains(&row.hole_loop_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_cross_barrier_local_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_barrier_long_path_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_hole_loop_coverage_95)
        }));
        write_planar_holes_barrier_topology_vs_naive_gp_outputs(&result, &out_dir)
            .expect("barrier challenge outputs should write");
        for file_name in [
            "metrics_summary.csv",
            "validation_summary.csv",
            "calibration_summary.csv",
            "heldout_predictions.csv",
            "field_coverage_summary.csv",
            "topology_summary.csv",
            "barrier_summary.csv",
        ] {
            assert!(
                out_dir.join(file_name).is_file(),
                "{file_name} should be written"
            );
        }
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_path_homology_challenge_produces_finite_outputs() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes path-homology challenge test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_path_homology_challenge_test");
        let _ = fs::remove_dir_all(&out_dir);
        let mut base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_path_homology_challenge_test.msh"),
            geo_path: out_dir.join("planar_holes_path_homology_challenge_test.geo"),
            force_mesh: true,
            mesh_size: 0.12,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 2.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            sample_observation_noise: false,
            ..PlanarHolesFlowConfig::default()
        };
        base.spectral_branch_energy_normalization = true;
        base.spectral_coexact_expected_m1_energy = 1.0;
        base.spectral_harmonic_expected_m1_energy = 4.0;
        base.spectral_harmonic_mode_count = 3;
        base.spectral_coexact_mode_count = 8;
        let config = PlanarHolesTopologyVsNaiveGpConfig {
            base,
            total_observation_budget: 12,
            validation_local_count: 4,
            validation_interior_loop_count: 1,
            validation_long_path_count: 1,
            heldout_local_count: 4,
            heldout_interior_loop_count: 1,
            heldout_long_path_count: 1,
            hodge_kappas: vec![3.0],
            hodge_taus: vec![0.05],
            naive_kappas: vec![6.0],
            naive_taus: vec![1.0],
            observation_variance_scales: vec![1.0],
            ..PlanarHolesTopologyVsNaiveGpConfig::default()
        };
        let result = run_planar_holes_path_homology_vs_naive_gp(&config)
            .expect("path-homology challenge should run on a small mesh");
        assert_eq!(result.base.topology_summary.b1, 3);
        assert_eq!(result.base.train_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.base.validation_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.base.heldout_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.path_contrast_harmonic_pairing_rank, 3);
        assert_eq!(result.rows.len(), 5);
        assert!(result.base.rows.iter().any(|row| row.model
            == PlanarHolesModelKind::SpectralIncompressibleHodgeGp
            && row.spectral_coexact_actual_modes > 0
            && row.spectral_harmonic_actual_modes == 3));
        assert!(result.base.rows.iter().any(|row| row.model
            == PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
            && row.model_coexact_tau_scale.is_finite()
            && row.model_coexact_tau_scale > 0.0));
        assert!(!result.base.calibration_rows.is_empty());
        assert!(result.base.calibration_rows.iter().all(|row| {
            row.variance_multiplier.is_finite()
                && row.variance_multiplier > 0.0
                && row.raw_nlpd.is_finite()
                && row.calibrated_nlpd.is_finite()
                && (0.0..=1.0).contains(&row.raw_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_coverage_95)
        }));
        assert!(result.rows.iter().all(|row| {
            row.train_path_pair_count == 3
                && row.validation_path_pair_count == 3
                && row.heldout_path_pair_count == 6
                && row.path_integral_nlpd.is_finite()
                && row.calibrated_path_integral_nlpd.is_finite()
                && row.path_integral_relative_error.is_finite()
                && row.path_contrast_nlpd.is_finite()
                && row.calibrated_path_contrast_nlpd.is_finite()
                && row.path_contrast_relative_error.is_finite()
                && row.path_contrast_mean_abs_z.is_finite()
                && row.calibrated_path_contrast_mean_abs_z.is_finite()
                && row.hole_loop_nlpd.is_finite()
                && row.calibrated_hole_loop_nlpd.is_finite()
                && row.hole_loop_relative_error.is_finite()
                && (0.0..=1.0).contains(&row.path_integral_coverage_95)
                && (0.0..=1.0).contains(&row.path_contrast_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_path_integral_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_path_contrast_coverage_95)
                && (0.0..=1.0).contains(&row.calibrated_hole_loop_coverage_95)
        }));
        let mesh_bytes = fs::read(&config.base.mesh_path).expect("mesh should exist");
        let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
        let pairs = build_path_homology_pairs(
            &topology,
            &coords,
            &default_holes(),
            0.035,
            std::f64::consts::PI / 5.0,
            "test_path_family",
        )
        .expect("path pairs should build");
        assert!(pairs.iter().all(|pair| {
            pair.shared_start_vertex != pair.shared_end_vertex
                && !sparse_rows_approximately_equal(&pair.upper.entries, &pair.lower.entries, 1e-12)
                && pair.closure_residual_l1 <= 1e-10
        }));
        write_planar_holes_path_homology_vs_naive_gp_outputs(&result, &out_dir)
            .expect("path-homology challenge outputs should write");
        for file_name in [
            "metrics_summary.csv",
            "validation_summary.csv",
            "calibration_summary.csv",
            "heldout_predictions.csv",
            "field_coverage_summary.csv",
            "topology_summary.csv",
            "path_homology_summary.csv",
        ] {
            assert!(
                out_dir.join(file_name).is_file(),
                "{file_name} should be written"
            );
        }
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_path_homology_vector_fields_produces_visual_outputs() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes path-homology vector-field test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_path_homology_vector_fields_test");
        let _ = fs::remove_dir_all(&out_dir);
        let mut base = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_path_homology_vector_fields_test.msh"),
            geo_path: out_dir.join("planar_holes_path_homology_vector_fields_test.geo"),
            force_mesh: true,
            mesh_size: 0.12,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 2.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            sample_observation_noise: false,
            ..PlanarHolesFlowConfig::default()
        };
        base.spectral_branch_energy_normalization = true;
        base.spectral_coexact_expected_m1_energy = 1.0;
        base.spectral_harmonic_expected_m1_energy = 4.0;
        base.spectral_harmonic_mode_count = 3;
        base.spectral_coexact_mode_count = 8;
        let config = PlanarHolesTopologyVsNaiveGpConfig {
            base,
            total_observation_budget: 12,
            validation_local_count: 4,
            validation_interior_loop_count: 1,
            validation_long_path_count: 1,
            heldout_local_count: 4,
            heldout_interior_loop_count: 1,
            heldout_long_path_count: 1,
            hodge_kappas: vec![3.0],
            hodge_taus: vec![0.05],
            naive_kappas: vec![6.0],
            naive_taus: vec![1.0],
            observation_variance_scales: vec![1.0],
            ..PlanarHolesTopologyVsNaiveGpConfig::default()
        };
        let result = run_planar_holes_path_homology_vector_fields(&config)
            .expect("path-homology vector-field figure should run on a small mesh");
        assert_eq!(result.topology_summary.b1, 3);
        assert_eq!(result.train_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.validation_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.heldout_cycle_harmonic_pairing_rank, 3);
        assert_eq!(result.path_contrast_harmonic_pairing_rank, 3);
        assert_eq!(result.rows.len(), 5);
        for expected in [
            PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
            PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern,
            PlanarHolesModelKind::IncompressibleHodgeMatern,
            PlanarHolesModelKind::NondecomposedFeec,
            PlanarHolesModelKind::NaiveEuclideanVectorMatern,
        ] {
            assert!(
                result.rows.iter().any(|row| row.model == expected),
                "missing vector-field row for {}",
                expected.as_str()
            );
        }
        assert!(result.truth_vtu_path.is_file());
        for row in &result.rows {
            assert!(row.posterior_vtu_path.is_file());
            assert!(row.l2_error.is_finite());
            assert!(row.heldout_nlpd.is_finite());
            assert!(row.calibrated_heldout_nlpd.is_finite());
            assert!(row.path_integral_relative_error.is_finite());
            assert!(row.path_contrast_relative_error.is_finite());
            assert!(row.hole_loop_relative_error.is_finite());
            assert!(row.codifferential_leakage.is_finite());
            assert!((0.0..=1.0).contains(&row.all_edge_coverage_95));
            assert!((0.0..=1.0).contains(&row.heldout_edge_coverage_95));
        }
        assert!(result.summary_csv_path.is_file());
        let summary_csv =
            fs::read_to_string(&result.summary_csv_path).expect("summary csv readable");
        assert!(summary_csv.contains("spectral_coexact_harmonic_gp"));
        assert!(summary_csv.contains("feec_exact_lower_trace_matched_coexact_harmonic_matern"));
        let png_bytes = fs::read(&result.figure_png_path).expect("png readable");
        assert!(png_bytes.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']));
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_incompressible_truth_is_discrete_coclosed() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes incompressible truth test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_incompressible_truth_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_incompressible_test.msh"),
            geo_path: out_dir.join("planar_holes_incompressible_test.geo"),
            force_mesh: true,
            mesh_size: 0.18,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.7,
            coexact_truth_source: PlanarHolesCoexactTruthSource::ExactMassCoexact,
            include_incompressible_hodge_model: true,
            local_observation_count: 8,
            heldout_local_count: 4,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("planar holes incompressible truth flow should run on small mesh");
        let mass_1form = de_rham::mass_matrix_form(&result.topology, &result.metric, 1).unwrap();
        let mass_0form = de_rham::mass_matrix_form(&result.topology, &result.metric, 0).unwrap();
        let truth_delta =
            de_rham::codifferential(&result.topology, &result.metric, 1, &result.truth).unwrap();
        let truth_leakage = relative_or_nan(
            mass_norm(&truth_delta, &mass_0form),
            mass_norm(&result.truth, &mass_1form),
        );
        assert!(
            truth_leakage <= INCOMPRESSIBLE_TRUTH_CODIFFERENTIAL_LEAKAGE_TOL,
            "exact-mass incompressible truth leakage {truth_leakage:.3e} exceeded tolerance"
        );

        for scenario in &result.scenarios {
            assert_eq!(scenario.metrics.len(), 5);
            for metric in &scenario.metrics {
                assert!(metric.l2_error.is_finite());
                assert!(metric.heldout_nlpd.is_finite());
                assert!(metric.heldout_harmonic_period_nlpd.is_finite());
                assert!(metric.codifferential_error.is_finite());
                assert!(metric.relative_total_annular_error.is_finite());
                assert_eq!(
                    metric.codifferential_metric_kind,
                    PlanarHolesCodifferentialMetricKind::Leakage
                );
            }
            assert!(
                scenario
                    .loop_functional_summaries
                    .iter()
                    .any(|summary| summary.functional
                        == PlanarHolesLoopFunctionalKind::HarmonicPeriod)
            );
            assert!(
                scenario
                    .loop_functional_summaries
                    .iter()
                    .any(|summary| summary.functional
                        == PlanarHolesLoopFunctionalKind::CoexactAnnular)
            );
            assert_eq!(scenario.field_coverage_summaries.len(), 10);
            assert!(scenario.field_coverage_summaries.iter().all(|summary| {
                summary.coverage_95.is_finite()
                    && (0.0..=1.0).contains(&summary.coverage_95)
                    && summary.mass_weighted_coverage_95.is_finite()
                    && (0.0..=1.0).contains(&summary.mass_weighted_coverage_95)
                    && summary.mean_abs_z.is_finite()
                    && summary.p95_abs_z.is_finite()
                    && summary.latent_nlpd.is_finite()
            }));
            assert_eq!(scenario.branch_recovery_summaries.len(), 5);
            assert!(scenario.branch_recovery_summaries.iter().all(|summary| {
                summary.truth_mass_norm.is_finite()
                    && summary.posterior_mass_norm.is_finite()
                    && summary.error_mass_norm.is_finite()
                    && (summary.truth_mass_norm <= RELATIVE_DENOM_EPS
                        || summary.relative_error.is_finite())
                    && (summary.truth_mass_norm <= RELATIVE_DENOM_EPS
                        || summary.posterior_mass_norm <= RELATIVE_DENOM_EPS
                        || summary.mass_correlation.is_finite())
            }));
        }

        let local_only = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalOnly)
            .unwrap();
        let local_plus = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalPlusLoops)
            .unwrap();
        let local_only_std = mean(local_only.period_summaries.iter().filter_map(|summary| {
            (summary.model == PlanarHolesModelKind::IncompressibleHodgeMatern)
                .then_some(summary.posterior_std)
        }));
        let local_plus_std = mean(local_plus.period_summaries.iter().filter_map(|summary| {
            (summary.model == PlanarHolesModelKind::IncompressibleHodgeMatern)
                .then_some(summary.posterior_std)
        }));
        assert!(
            local_plus_std < local_only_std,
            "loop observations should reduce incompressible Hodge period uncertainty"
        );

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_matched_dense_oracle_recovers_pure_coexact_truth() {
        if !gmsh_available() {
            eprintln!("skipping planar holes oracle test because gmsh is unavailable");
            return;
        }
        let out_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/planar_holes_oracle_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_oracle_test.msh"),
            geo_path: out_dir.join("planar_holes_oracle_test.geo"),
            force_mesh: true,
            mesh_size: 0.17,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::SparseAnchor,
            truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
            local_observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-8,
            loop_noise_variance: 1e-8,
            include_incompressible_hodge_model: true,
            local_observation_count: 1,
            heldout_local_count: 1,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("matched dense pure-coexact oracle should run");
        let local_only = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalOnly)
            .unwrap();
        let hodge = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::HodgeMatern)
            .unwrap();
        let incompressible = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::IncompressibleHodgeMatern)
            .unwrap();
        let nondecomposed = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::NondecomposedFeec)
            .unwrap();
        assert!(
            hodge.l2_error < 1e-2,
            "matched dense Hodge oracle E1 too large: {:.3e}",
            hodge.l2_error
        );
        assert!(
            incompressible.l2_error < 1e-2,
            "matched dense coexact-harmonic oracle E1 too large: {:.3e}",
            incompressible.l2_error
        );
        assert!(
            nondecomposed.l2_error < 1e-2,
            "matched dense nondecomposed oracle E1 too large: {:.3e}",
            nondecomposed.l2_error
        );
        for model in [
            PlanarHolesModelKind::HodgeMatern,
            PlanarHolesModelKind::IncompressibleHodgeMatern,
        ] {
            let coexact = local_only
                .branch_recovery_summaries
                .iter()
                .find(|summary| {
                    summary.model == model && summary.branch == HodgeBranchKind::Coexact
                })
                .unwrap();
            assert!(
                coexact.relative_error < 1e-2,
                "{model:?} coexact branch relative error too large: {:.3e}",
                coexact.relative_error
            );
        }

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_exact_mass_coexact_prior_dense_oracle_recovers_exact_mass_truth() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes exact-mass coexact prior oracle test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_exact_mass_coexact_prior_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_exact_mass_coexact_prior_test.msh"),
            geo_path: out_dir.join("planar_holes_exact_mass_coexact_prior_test.geo"),
            force_mesh: true,
            mesh_size: 0.17,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::ExactMassCoexact,
            truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
            local_observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-8,
            loop_noise_variance: 1e-8,
            include_incompressible_hodge_model: true,
            include_exact_mass_incompressible_hodge_model: true,
            include_sparse_lower_trace_matched_incompressible_hodge_model: true,
            local_observation_count: 1,
            heldout_local_count: 1,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("exact-mass coexact dense oracle should run");
        let local_only = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalOnly)
            .unwrap();
        let exact_mass = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::ExactMassIncompressibleHodgeMatern)
            .unwrap();
        let sparse_trace_matched = local_only
            .metrics
            .iter()
            .find(|metric| {
                metric.model
                    == PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
            })
            .unwrap();
        assert!(
            exact_mass.l2_error < 1e-2,
            "exact-mass coexact prior dense oracle E1 too large: {:.3e}",
            exact_mass.l2_error
        );
        assert!(
            sparse_trace_matched.l2_error < 1e-2,
            "trace-matched sparse-lower coexact prior dense oracle E1 too large: {:.3e}",
            sparse_trace_matched.l2_error
        );
        assert!(
            exact_mass.codifferential_leakage < 1e-8,
            "exact-mass coexact prior leakage too large: {:.3e}",
            exact_mass.codifferential_leakage
        );
        let coexact = local_only
            .branch_recovery_summaries
            .iter()
            .find(|summary| {
                summary.model == PlanarHolesModelKind::ExactMassIncompressibleHodgeMatern
                    && summary.branch == HodgeBranchKind::Coexact
            })
            .unwrap();
        assert!(
            coexact.relative_error < 1e-2,
            "exact-mass coexact branch relative error too large: {:.3e}",
            coexact.relative_error
        );

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_exact_lower_coexact_prior_dense_oracle_recovers_exact_lower_truth() {
        if !gmsh_available() {
            eprintln!(
                "skipping planar holes exact-lower coexact prior oracle test because gmsh is unavailable"
            );
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_exact_lower_coexact_prior_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_exact_lower_coexact_prior_test.msh"),
            geo_path: out_dir.join("planar_holes_exact_lower_coexact_prior_test.geo"),
            force_mesh: true,
            mesh_size: 0.17,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::ExactLowerMassCoexact,
            truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
            local_observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-8,
            loop_noise_variance: 1e-8,
            include_incompressible_hodge_model: false,
            include_exact_mass_incompressible_hodge_model: false,
            include_exact_lower_incompressible_hodge_model: true,
            include_exact_lower_trace_matched_incompressible_hodge_model: true,
            spectral_branch_energy_normalization: true,
            spectral_coexact_expected_m1_energy: 1.0,
            local_observation_count: 1,
            heldout_local_count: 1,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("exact-lower coexact dense oracle should run");
        let local_only = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalOnly)
            .unwrap();
        let exact_lower = local_only
            .metrics
            .iter()
            .find(|metric| {
                metric.model == PlanarHolesModelKind::ExactLowerIncompressibleHodgeMatern
            })
            .unwrap();
        let trace_matched = local_only
            .metrics
            .iter()
            .find(|metric| {
                metric.model
                    == PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
            })
            .unwrap();
        assert!(
            exact_lower.l2_error < 1e-2,
            "raw exact-lower coexact prior dense oracle E1 too large: {:.3e}",
            exact_lower.l2_error
        );
        assert!(
            trace_matched.l2_error < 1e-2,
            "trace-matched exact-lower coexact prior dense oracle E1 too large: {:.3e}",
            trace_matched.l2_error
        );
        assert!(
            trace_matched.codifferential_leakage < 1e-8,
            "trace-matched exact-lower coexact prior leakage too large: {:.3e}",
            trace_matched.codifferential_leakage
        );
        let trace_matched_coexact = local_only
            .branch_recovery_summaries
            .iter()
            .find(|summary| {
                summary.model
                    == PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
                    && summary.branch == HodgeBranchKind::Coexact
            })
            .unwrap();
        assert!(
            trace_matched_coexact.relative_error < 1e-2,
            "trace-matched exact-lower coexact branch relative error too large: {:.3e}",
            trace_matched_coexact.relative_error
        );

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn planar_holes_matched_dense_oracle_recovers_pure_exact_truth() {
        if !gmsh_available() {
            eprintln!("skipping planar holes exact oracle test because gmsh is unavailable");
            return;
        }
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/planar_holes_exact_oracle_test");
        let _ = fs::remove_dir_all(&out_dir);
        let config = PlanarHolesFlowConfig {
            output_dir: out_dir.clone(),
            mesh_path: out_dir.join("planar_holes_exact_oracle_test.msh"),
            geo_path: out_dir.join("planar_holes_exact_oracle_test.geo"),
            force_mesh: true,
            mesh_size: 0.17,
            exact_truth_mass_norm: 1.0,
            coexact_truth_mass_norm: 0.0,
            harmonic_truth_mass_norm: 0.0,
            truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
            local_observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-8,
            loop_noise_variance: 1e-8,
            include_exact_hodge_model: true,
            include_incompressible_hodge_model: false,
            local_observation_count: 1,
            heldout_local_count: 1,
            ..PlanarHolesFlowConfig::default()
        };

        let result = run_planar_holes_hodge_flow(&config)
            .expect("matched dense pure-exact oracle should run");
        let local_only = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario == PlanarHolesObservationScenario::LocalOnly)
            .unwrap();
        let exact_model = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::ExactHodgeMatern)
            .unwrap();
        let hodge = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::HodgeMatern)
            .unwrap();
        let nondecomposed = local_only
            .metrics
            .iter()
            .find(|metric| metric.model == PlanarHolesModelKind::NondecomposedFeec)
            .unwrap();
        assert!(
            exact_model.l2_error < 1e-2,
            "matched dense exact-only oracle E1 too large: {:.3e}",
            exact_model.l2_error
        );
        assert!(
            hodge.l2_error < 1e-2,
            "matched dense full Hodge exact oracle E1 too large: {:.3e}",
            hodge.l2_error
        );
        assert!(
            nondecomposed.l2_error < 1e-2,
            "matched dense nondecomposed exact oracle E1 too large: {:.3e}",
            nondecomposed.l2_error
        );

        let mass_1form = de_rham::mass_matrix_form(&result.topology, &result.metric, 1).unwrap();
        let truth_norm = mass_norm(&result.truth, &mass_1form);
        assert!(
            exact_model.exterior_derivative_error_absolute / truth_norm < 1e-8,
            "exact-only posterior should have negligible d leakage, got {:.3e}",
            exact_model.exterior_derivative_error_absolute / truth_norm
        );
        let exact_branch = local_only
            .branch_recovery_summaries
            .iter()
            .find(|summary| {
                summary.model == PlanarHolesModelKind::ExactHodgeMatern
                    && summary.branch == HodgeBranchKind::Exact
            })
            .unwrap();
        assert!(
            exact_branch.relative_error < 1e-2,
            "exact-only branch relative error too large: {:.3e}",
            exact_branch.relative_error
        );

        let _ = fs::remove_dir_all(&out_dir);
    }
}
