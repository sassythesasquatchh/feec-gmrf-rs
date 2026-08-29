//! Reusable C51 accuracy--resource benchmark harness.
//!
//! The smoke profile is deliberately small. It checks the archive schema and
//! compares three precision construction routes and a bounded spectral
//! reference on a three-dimensional
//! Whitney 1-form problem:
//!
//! - `FM-DIRECT`: exact eliminated full-metric precision;
//! - `FM-AUG`: the same finite Gaussian law on a sparse constrained support;
//! - `WHITNEY-COMPACT`: the projected-inverse Whitney approximation.
//!
//! Full-rank spectral rows represent the same finite Gaussian law as
//! `FM-DIRECT`; truncated spectral rows are approximation baselines and are
//! never described as the same law. Tiny timings validate instrumentation
//! only and are not evidence of a performance advantage.

use crate::c51_magnetic_sensor_design::{
    assemble_c51_active_flux_cochain_map, assemble_c51_flux_cochain_split,
    assemble_c51_surface_flux_split, assemble_juq_performance_observation_prefix, held_out_kernels,
    selection_kernels, C51_FIXED_OBSERVATION_RADIUS, C51_FLUX_COCHAIN_DESIGN_ID,
    C51_FLUX_COCHAIN_HELD_OUT_COUNT, C51_FLUX_COCHAIN_SELECTION_COUNT,
    C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT, C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID,
    C51_SURFACE_FLUX_QOI_DESIGN_ID,
};
use common::linalg::nalgebra::CsrMatrix as FeecCsrMatrix;
use feec_gmrf::prelude::{
    calibrate_prior_to_physical_rms, AdjacentMaternPriorBuilder, CholeskyOrderingKind,
    EssentialBoundaryConditions, EssentialBoundaryElimination, ExactFormOperators,
    ExactMaternPriorBuilder, ExactObservationCovarianceBlocks,
    ExactObservationCovarianceProvenance, ExactObservationPosterior, FeecGmrfError, GaussianNoise,
    GaussianPrior, LinearGaussianModelBuilder, LinearMap, LinearObservation, MagneticFieldMaps3d,
    MaternAlpha, MaternParameters, MaternPriorBuilder, ResourceMeasurement, Result, SparseMat,
    SupportedMaternPrior, SupportedMaternPriorBuilder, VarianceMethod,
};
#[cfg(feature = "external-solvers")]
use feec_gmrf::spectral::{
    exact_requested_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative,
    partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative,
    PartialMixedGeneralizedEigensolverOptions,
};
use feg_gp::{
    dense_reference_generalized_hodge_eigenbasis, DenseReferenceEigensolverOptions, EigenBasis,
    LowRankConditioningStrategy, SpectralMaternConfig, SpectralMaternGp,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::simplex::SimplexHandleExt,
    topology::complex::Complex,
};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::collections::BTreeSet;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

const C51_CALIBRATION_SELECTION_OUTPUT_COUNT: usize = 9;

pub const C51_SCHEMA_VERSION: &str = "c51-accuracy-resource-v7";
pub const C51_RESOURCE_SCHEMA_VERSION: &str = "c51-method-resource-v5";
pub const C51_STUDY_ID: &str = "c51/accuracy-resource";

const EXACT_REFERENCE_MAXIMUM_DIMENSION: usize = 1_024;
const UPDATE_NNZ_PENDING_REASON: &str =
    "supported-law observation-update nnz is not yet exposed by the root API";
const KKT_FILL_UNAVAILABLE_REASON: &str =
    "faer 0.21.9 does not expose sparse LU numeric factor storage";
const SPECTRAL_SPARSE_PRECISION_REASON: &str =
    "a spectral feature model does not assemble a sparse coefficient precision";
const MIXED_MARGINAL_NNZ_UNAVAILABLE_REASON: &str =
    "the sparse mixed shell applies the marginal weak Hodge operator without materializing it";
const RESOURCE_ACCURACY_UNAVAILABLE_REASON: &str =
    "resource-only method process does not construct an accuracy reference";
const EXACT_REQUESTED_MIXED_RESIDUAL_TOLERANCE: f64 = 1.0e-8;
const EXACT_REQUESTED_MIXED_EIGENSOLVER_TOLERANCE: f64 = 1.0e-10;
const EXACT_REQUESTED_MIXED_MASS_ORTHOGONALITY_TOLERANCE: f64 = 1.0e-8;
const EXACT_REQUESTED_MIXED_EIGENVALUE_NEGATIVE_TOLERANCE: f64 = 1.0e-10;
pub const C51_CLAIM_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT: f64 = 0.05;
pub const C51_SENSITIVITY_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT: f64 = 0.02;
pub const C51_CONTEXT_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT: f64 = 0.10;

// Frozen from clean direct-cochain accuracy archives at parent `5eef534`.
const C51_INTEGRAL_FLUX_EXACT_TAU_N8: f64 = 17.652_314_257_032_09;
const C51_INTEGRAL_FLUX_EXACT_TAU_N10: f64 = 20.453_086_423_392_97;
const C51_INTEGRAL_FLUX_COMPACT_TRACE_TAU_N8: f64 = 22.641_722_540_142_85;
const C51_INTEGRAL_FLUX_COMPACT_TRACE_TAU_N10: f64 = 26.051_269_480_098_004;
const C51_INTEGRAL_FLUX_POSTERIOR_SCALE_N8: f64 = 0.911_588_783_318_580_1;
const C51_INTEGRAL_FLUX_POSTERIOR_SCALE_N10: f64 = 0.808_522_246_426_985_3;
const C51_INTEGRAL_FLUX_TARGET_POSTERIOR_RMS_N8: f64 = 9.619_759_447_761_718e-4;
const C51_INTEGRAL_FLUX_TARGET_POSTERIOR_RMS_N10: f64 = 1.008_755_891_127_561_6e-3;

/// A resource or accuracy quantity with explicit availability semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum C51Measurement<T> {
    Measured(T),
    NotRecorded,
    Unavailable { reason: String },
}

impl<T> C51Measurement<T> {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    fn status(&self) -> &'static str {
        match self {
            Self::Measured(_) => "measured",
            Self::NotRecorded => "not_recorded",
            Self::Unavailable { .. } => "unavailable",
        }
    }

    fn value_string(&self) -> String
    where
        T: Display,
    {
        match self {
            Self::Measured(value) => value.to_string(),
            Self::NotRecorded | Self::Unavailable { .. } => String::new(),
        }
    }

    fn reason(&self) -> &str {
        match self {
            Self::Unavailable { reason } => reason,
            Self::Measured(_) | Self::NotRecorded => "",
        }
    }
}

/// The two comparison questions retained in every archive manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C51ComparisonFrame {
    /// Compare numerical representations of the identical finite Gaussian law.
    SameDiscreteLawRepresentation,
    /// Compare methods after imposing one common physical-QoI accuracy target.
    MatchedPhysicalQoiMethod,
}

impl C51ComparisonFrame {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SameDiscreteLawRepresentation => "same_discrete_law_representation",
            Self::MatchedPhysicalQoiMethod => "matched_physical_qoi_method",
        }
    }
}

/// Stable method identifiers used by CSV rows and later production adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C51MethodId {
    FullMetricDirect,
    FullMetricAugmented,
    WhitneyCompact,
    WhitneyCompactCalibrated,
    WhitneyCompactIntegralFluxCalibrated,
    ExactObservationSpaceGp,
    SpectralFullRank,
    SpectralLowRank,
}

impl C51MethodId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullMetricDirect => "fm_direct",
            Self::FullMetricAugmented => "fm_aug",
            Self::WhitneyCompact => "whitney_compact",
            Self::WhitneyCompactCalibrated => "whitney_compact_calibrated",
            Self::WhitneyCompactIntegralFluxCalibrated => {
                "whitney_compact_integral_flux_calibrated"
            }
            Self::ExactObservationSpaceGp => "exact_observation_gp",
            Self::SpectralFullRank => "spectral_full_rank",
            Self::SpectralLowRank => "spectral_low_rank",
        }
    }
}

/// Physical-B observation design used by a C51 invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C51PhysicalObservationDesign {
    /// Legacy smoke design selecting scalar rows from the cellwise
    /// reconstruction. Its physical footprint shrinks under refinement.
    LegacyCellRows,
    /// Frozen, nested Wendland-C2 volume sensors with fixed physical radius.
    /// The scalar count is independent of the latent dimension.
    JuqFixedFootprintPrefix,
    /// Deterministic prefix of exact face-integral rows of the active `D1`
    /// magnetic flux cochain.
    FluxCochainHashedPrefix,
}

impl C51PhysicalObservationDesign {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LegacyCellRows => "legacy_cellwise_b_rows",
            Self::JuqFixedFootprintPrefix => C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID,
            Self::FluxCochainHashedPrefix => C51_FLUX_COCHAIN_DESIGN_ID,
        }
    }
}

/// Physical output bank used for method accuracy and covariance queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C51PhysicalOutputDesign {
    /// Three central-region volume-average components retained by smoke.
    LegacyCentralRegionAverage,
    /// The independent selection and held-out fixed-footprint banks used by
    /// the spectral-rank study (nine scalar outputs per bank).
    RankStudySelectionAndHeldOut,
    /// Six calibration and six held-out mid-plane quadrant flux integrals.
    MidplaneQuadrantSurfaceFluxSplit,
}

impl C51PhysicalOutputDesign {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LegacyCentralRegionAverage => "legacy_central_region_volume_average_3",
            Self::RankStudySelectionAndHeldOut => {
                "rank_study_selection_and_held_out_wendland_c2_18"
            }
            Self::MidplaneQuadrantSurfaceFluxSplit => C51_SURFACE_FLUX_QOI_DESIGN_ID,
        }
    }
}

/// Observation locality is an explicit study axis, never pooled after a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C51ObservationClass {
    LocalPhysicalB,
    LocalFluxCochain,
    LocalSelectors,
    GlobalRow,
}

impl C51ObservationClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalPhysicalB => "local_physical_b",
            Self::LocalFluxCochain => "local_flux_cochain",
            Self::LocalSelectors => "local_selectors",
            Self::GlobalRow => "global_row",
        }
    }
}

/// Requested sensor count for a method-isolated resource process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum C51ResourceSensorTarget {
    Count(usize),
    Density(f64),
}

impl C51ResourceSensorTarget {
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Count(_) => "count",
            Self::Density(_) => "density",
        }
    }

    pub fn value(self) -> f64 {
        match self {
            Self::Count(count) => count as f64,
            Self::Density(density) => density,
        }
    }
}

/// Process phase whose peak memory has exactly one method owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C51ResourcePhase {
    ColdSetup,
    /// Common claim-bearing physical-output workload: replacement-data output
    /// means plus one full physical-output posterior covariance, action, and
    /// diagonal. No latent covariance query or state sampling is included.
    QueryPhysicalUq,
    QueryFullUq,
}

impl C51ResourcePhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ColdSetup => "cold_setup",
            Self::QueryPhysicalUq => "query_physical_uq",
            Self::QueryFullUq => "query_full_uq",
        }
    }
}

/// Exactly one resource workload selected for one fresh process.
#[derive(Debug, Clone, PartialEq)]
pub struct C51ResourceSelection {
    pub refinement: usize,
    pub observation_class: C51ObservationClass,
    pub sensor_target: C51ResourceSensorTarget,
    pub repeated_data_count: usize,
    pub method: C51MethodId,
    pub spectral_rank: Option<usize>,
    pub phase: C51ResourcePhase,
}

impl C51ResourceSelection {
    fn validate(&self) -> Result<()> {
        if self.refinement < 2 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 resource refinement must be at least two".to_string(),
            ));
        }
        match self.sensor_target {
            C51ResourceSensorTarget::Count(0) => {
                return Err(FeecGmrfError::InvalidParameter(
                    "C51 resource sensor count must be positive".to_string(),
                ))
            }
            C51ResourceSensorTarget::Density(density)
                if !density.is_finite() || density <= 0.0 || density > 1.0 =>
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "C51 resource sensor density must lie in (0, 1]".to_string(),
                ))
            }
            _ => {}
        }
        if self.observation_class == C51ObservationClass::GlobalRow
            && self.sensor_target != C51ResourceSensorTarget::Count(1)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "the C51 global-row control has exactly one observation".to_string(),
            ));
        }
        if self.repeated_data_count == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 resource repeated-data count must be positive".to_string(),
            ));
        }
        match (self.method, self.spectral_rank) {
            (C51MethodId::SpectralFullRank, None) => {}
            (C51MethodId::SpectralLowRank, Some(rank)) if rank > 0 => {}
            (C51MethodId::SpectralFullRank | C51MethodId::SpectralLowRank, _) => {
                return Err(FeecGmrfError::InvalidParameter(
                    "full-rank spectral selection requires no rank; low-rank spectral selection requires one positive rank"
                        .to_string(),
                ))
            }
            (_, None) => {}
            (_, Some(_)) => {
                return Err(FeecGmrfError::InvalidParameter(
                    "only a low-rank spectral resource selection accepts an explicit rank"
                        .to_string(),
                ))
            }
        }
        Ok(())
    }
}

/// Scientific constants plus one method-owned resource selection.
#[derive(Debug, Clone, PartialEq)]
pub struct C51ResourceConfig {
    pub selection: C51ResourceSelection,
    pub physical_observation_design: C51PhysicalObservationDesign,
    pub physical_output_design: C51PhysicalOutputDesign,
    pub spatial_dimension: usize,
    pub form_degree: usize,
    pub alpha: MaternAlpha,
    pub kappa: f64,
    pub tau: f64,
    pub observation_noise_variance: f64,
    pub physical_qoi_region_lower: [f64; 3],
    pub physical_qoi_region_upper: [f64; 3],
    pub spectral_reference_maximum_dimension: usize,
    pub spectral_operator_materialization_maximum_dimension: usize,
    pub posterior_sample_count: usize,
    pub seed: u64,
    /// Frozen exact-FEEC prior RMS on the nine-output selection bank. Required
    /// only by the physically calibrated projected-Whitney resource method.
    pub projected_calibration_target_prior_rms: Option<f64>,
    /// Frozen coherent posterior-QoI precision scale for the direct integral
    /// flux study. It is applied after the submitted trace normalization.
    pub projected_integral_flux_posterior_precision_scale: Option<f64>,
    /// Selection-bank posterior RMS that produced the frozen precision scale.
    pub projected_integral_flux_target_posterior_rms: Option<f64>,
}

impl C51ResourceConfig {
    /// Reuse the smoke scientific constants while selecting exactly one row.
    pub fn smoke(selection: C51ResourceSelection) -> Self {
        let smoke = C51BenchmarkConfig::smoke();
        Self {
            selection,
            physical_observation_design: smoke.physical_observation_design,
            physical_output_design: smoke.physical_output_design,
            spatial_dimension: smoke.spatial_dimension,
            form_degree: smoke.form_degree,
            alpha: smoke.alpha,
            kappa: smoke.kappa,
            tau: smoke.tau,
            observation_noise_variance: smoke.observation_noise_variance,
            physical_qoi_region_lower: smoke.physical_qoi_region_lower,
            physical_qoi_region_upper: smoke.physical_qoi_region_upper,
            spectral_reference_maximum_dimension: smoke.spectral_reference_maximum_dimension,
            spectral_operator_materialization_maximum_dimension: smoke
                .spectral_operator_materialization_maximum_dimension,
            posterior_sample_count: smoke.posterior_sample_count,
            seed: smoke.seed,
            projected_calibration_target_prior_rms: None,
            projected_integral_flux_posterior_precision_scale: None,
            projected_integral_flux_target_posterior_rms: None,
        }
    }

    /// Resource configuration for the claim-bearing JUQ performance design.
    ///
    /// Unlike [`Self::smoke`], this selects the fixed-footprint Wendland sensor
    /// prefix and can therefore request observation counts independently of
    /// the latent state dimension.
    pub fn juq_performance(selection: C51ResourceSelection) -> Self {
        let performance = C51BenchmarkConfig::juq_performance();
        Self {
            selection,
            physical_observation_design: performance.physical_observation_design,
            physical_output_design: performance.physical_output_design,
            spatial_dimension: performance.spatial_dimension,
            form_degree: performance.form_degree,
            alpha: performance.alpha,
            kappa: performance.kappa,
            tau: performance.tau,
            observation_noise_variance: performance.observation_noise_variance,
            physical_qoi_region_lower: performance.physical_qoi_region_lower,
            physical_qoi_region_upper: performance.physical_qoi_region_upper,
            spectral_reference_maximum_dimension: performance.spectral_reference_maximum_dimension,
            spectral_operator_materialization_maximum_dimension: performance
                .spectral_operator_materialization_maximum_dimension,
            posterior_sample_count: performance.posterior_sample_count,
            seed: performance.seed,
            projected_calibration_target_prior_rms: None,
            projected_integral_flux_posterior_precision_scale: None,
            projected_integral_flux_target_posterior_rms: None,
        }
    }

    /// Claim-bearing exact-cochain resource configuration.
    ///
    /// This freezes the observation density, submitted magnetic trace
    /// normalization, and any method-specific amplitude selected in the clean
    /// direct-cochain accuracy archive. The offline calibration search is not
    /// repeated inside timed method processes.
    pub fn juq_integral_flux_performance(selection: C51ResourceSelection) -> Result<Self> {
        let (expected_observations, exact_tau, compact_tau, posterior_scale, target_rms) =
            match selection.refinement {
                8 => (
                    4_096,
                    C51_INTEGRAL_FLUX_EXACT_TAU_N8,
                    C51_INTEGRAL_FLUX_COMPACT_TRACE_TAU_N8,
                    C51_INTEGRAL_FLUX_POSTERIOR_SCALE_N8,
                    C51_INTEGRAL_FLUX_TARGET_POSTERIOR_RMS_N8,
                ),
                10 => (
                    8_192,
                    C51_INTEGRAL_FLUX_EXACT_TAU_N10,
                    C51_INTEGRAL_FLUX_COMPACT_TRACE_TAU_N10,
                    C51_INTEGRAL_FLUX_POSTERIOR_SCALE_N10,
                    C51_INTEGRAL_FLUX_TARGET_POSTERIOR_RMS_N10,
                ),
                refinement => {
                    return Err(FeecGmrfError::InvalidParameter(format!(
                        "the frozen integral-flux resource profile has no refinement {refinement}"
                    )))
                }
            };
        if selection.observation_class != C51ObservationClass::LocalFluxCochain
            || selection.sensor_target != C51ResourceSensorTarget::Count(expected_observations)
        {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "the refinement-{} integral-flux profile requires exactly {} local flux-cochain observations",
                selection.refinement, expected_observations
            )));
        }
        let (tau, frozen_scale, frozen_target) = match selection.method {
            C51MethodId::WhitneyCompact => (compact_tau, None, None),
            C51MethodId::WhitneyCompactIntegralFluxCalibrated => {
                (compact_tau, Some(posterior_scale), Some(target_rms))
            }
            C51MethodId::WhitneyCompactCalibrated => {
                return Err(FeecGmrfError::InvalidParameter(
                    "the legacy prior-RMS calibrated compact method is not part of the direct integral-flux profile"
                        .to_string(),
                ))
            }
            _ => (exact_tau, None, None),
        };
        let mut config = Self::juq_performance(selection);
        config.physical_observation_design = C51PhysicalObservationDesign::FluxCochainHashedPrefix;
        config.physical_output_design = C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit;
        config.tau = tau;
        config.observation_noise_variance = 2.5e-7;
        config.projected_integral_flux_posterior_precision_scale = frozen_scale;
        config.projected_integral_flux_target_posterior_rms = frozen_target;
        config.validate()?;
        Ok(config)
    }

    fn benchmark_config(&self) -> C51BenchmarkConfig {
        let (local_sensor_counts, local_sensor_density_targets) = match self.selection.sensor_target
        {
            C51ResourceSensorTarget::Count(count) => (vec![count], Vec::new()),
            C51ResourceSensorTarget::Density(density) => (Vec::new(), vec![density]),
        };
        C51BenchmarkConfig {
            refinements: vec![self.selection.refinement],
            physical_observation_design: self.physical_observation_design,
            physical_output_design: self.physical_output_design,
            spatial_dimension: self.spatial_dimension,
            form_degree: self.form_degree,
            alpha: self.alpha,
            kappa: self.kappa,
            tau: self.tau,
            observation_noise_variance: self.observation_noise_variance,
            local_sensor_counts,
            local_sensor_density_targets,
            repeated_data_counts: vec![self.selection.repeated_data_count],
            include_global_row: self.selection.observation_class == C51ObservationClass::GlobalRow,
            include_local_selectors: self.selection.observation_class
                == C51ObservationClass::LocalSelectors,
            physical_qoi_region_lower: self.physical_qoi_region_lower,
            physical_qoi_region_upper: self.physical_qoi_region_upper,
            spectral_low_ranks: self.selection.spectral_rank.into_iter().collect(),
            spectral_rank_fractions: Vec::new(),
            spectral_reference_maximum_dimension: self.spectral_reference_maximum_dimension,
            spectral_operator_materialization_maximum_dimension: self
                .spectral_operator_materialization_maximum_dimension,
            posterior_sample_count: self.posterior_sample_count,
            seed: self.seed,
        }
    }

    fn validate(&self) -> Result<()> {
        self.selection.validate()?;
        match (
            self.selection.method,
            self.projected_calibration_target_prior_rms,
            self.projected_integral_flux_posterior_precision_scale,
            self.projected_integral_flux_target_posterior_rms,
        ) {
            (C51MethodId::WhitneyCompactCalibrated, Some(value), None, None)
                if value.is_finite() && value > 0.0 => {}
            (
                C51MethodId::WhitneyCompactIntegralFluxCalibrated,
                None,
                Some(scale),
                Some(target),
            ) if scale.is_finite() && scale > 0.0 && target.is_finite() && target > 0.0 => {}
            (C51MethodId::WhitneyCompactCalibrated, _, _, _) => {
                return Err(FeecGmrfError::InvalidParameter(
                    "the calibrated projected-Whitney method requires a positive finite frozen target prior RMS"
                        .to_string(),
                ));
            }
            (C51MethodId::WhitneyCompactIntegralFluxCalibrated, _, _, _) => {
                return Err(FeecGmrfError::InvalidParameter(
                    "the calibrated integral-flux method requires a positive frozen posterior precision scale and target RMS"
                        .to_string(),
                ));
            }
            (_, None, None, None) => {}
            _ => {
                return Err(FeecGmrfError::InvalidParameter(
                    "projected calibration metadata is valid only for its matching calibrated method"
                        .to_string(),
                ));
            }
        }
        if self.selection.method == C51MethodId::ExactObservationSpaceGp
            && !matches!(
                self.selection.observation_class,
                C51ObservationClass::LocalPhysicalB | C51ObservationClass::LocalFluxCochain
            )
        {
            return Err(FeecGmrfError::InvalidParameter(
                "the exact observation-space GP resource method requires a common local physical design"
                    .to_string(),
            ));
        }
        self.benchmark_config().validate()
    }
}

/// Hardware fields are intentionally supplied by the fresh-process runner.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C51HardwareMetadata {
    pub host: Option<String>,
    pub cpu_model: Option<String>,
    pub logical_cores: Option<usize>,
    pub total_memory_bytes: Option<u64>,
    pub operating_system: Option<String>,
    pub kernel: Option<String>,
    pub thread_affinity: Option<String>,
}

/// Reproducibility metadata written without inferring unavailable repository data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C51RunMetadata {
    pub profile: String,
    pub command: String,
    pub root_sha: Option<String>,
    pub feec_sha: Option<String>,
    pub gmrf_sha: Option<String>,
    pub root_dirty: Option<bool>,
    pub feec_dirty: Option<bool>,
    pub gmrf_dirty: Option<bool>,
    pub rustc_version: Option<String>,
    pub compiler_profile: String,
    pub sparse_backend: String,
    pub hardware: C51HardwareMetadata,
}

impl C51RunMetadata {
    pub fn smoke(command: impl Into<String>) -> Self {
        Self {
            profile: "smoke".to_string(),
            command: command.into(),
            root_sha: None,
            feec_sha: None,
            gmrf_sha: None,
            root_dirty: None,
            feec_dirty: None,
            gmrf_dirty: None,
            rustc_version: None,
            compiler_profile: "release".to_string(),
            sparse_backend: "faer-0.21".to_string(),
            hardware: C51HardwareMetadata::default(),
        }
    }

    /// Capture root/submodule Git state and the Rust compiler for a real run.
    pub fn from_repository(command: impl Into<String>, repository: &Path) -> Result<Self> {
        let mut metadata = Self::smoke(command);
        metadata.root_sha = Some(c51_command_output(
            Command::new("git")
                .arg("-C")
                .arg(repository)
                .args(["rev-parse", "HEAD"]),
            "root Git SHA",
        )?);
        metadata.root_dirty = Some(
            !c51_command_output(
                Command::new("git")
                    .arg("-C")
                    .arg(repository)
                    .args(["status", "--porcelain"]),
                "root Git status",
            )?
            .is_empty(),
        );
        for (path, sha, dirty, label) in [
            (
                repository.join("feec"),
                &mut metadata.feec_sha,
                &mut metadata.feec_dirty,
                "FEEC",
            ),
            (
                repository.join("gmrf-rs"),
                &mut metadata.gmrf_sha,
                &mut metadata.gmrf_dirty,
                "GMRF",
            ),
        ] {
            *sha = Some(c51_command_output(
                Command::new("git")
                    .arg("-C")
                    .arg(&path)
                    .args(["rev-parse", "HEAD"]),
                &format!("{label} Git SHA"),
            )?);
            *dirty = Some(
                !c51_command_output(
                    Command::new("git")
                        .arg("-C")
                        .arg(&path)
                        .args(["status", "--porcelain"]),
                    &format!("{label} Git status"),
                )?
                .is_empty(),
            );
        }
        metadata.rustc_version = Some(c51_command_output(
            Command::new("rustc").args(["--version", "--verbose"]),
            "rustc version",
        )?);
        Ok(metadata)
    }
}

fn c51_command_output(command: &mut Command, label: &str) -> Result<String> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(FeecGmrfError::Inference(format!(
            "could not capture {label}: status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| {
            FeecGmrfError::Inference(format!("could not decode {label} as UTF-8: {error}"))
        })
}

/// Frozen mathematical and workload axes for one benchmark invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct C51BenchmarkConfig {
    pub refinements: Vec<usize>,
    pub physical_observation_design: C51PhysicalObservationDesign,
    pub physical_output_design: C51PhysicalOutputDesign,
    pub spatial_dimension: usize,
    pub form_degree: usize,
    pub alpha: MaternAlpha,
    pub kappa: f64,
    pub tau: f64,
    pub observation_noise_variance: f64,
    /// Local bounded-support sensor counts.
    pub local_sensor_counts: Vec<usize>,
    /// Local sensor targets as fractions of the active state dimension.
    pub local_sensor_density_targets: Vec<f64>,
    /// Number of replacement data vectors solved against a fixed design.
    pub repeated_data_counts: Vec<usize>,
    /// Add one dense, mesh-global observation row as a locality control.
    pub include_global_row: bool,
    /// Retain coefficient selectors as a nonphysical locality control.
    pub include_local_selectors: bool,
    /// Fixed physical region for the three volume-average magnetic QoIs.
    pub physical_qoi_region_lower: [f64; 3],
    pub physical_qoi_region_upper: [f64; 3],
    /// Requested truncated spectral ranks. Full rank is always added separately.
    pub spectral_low_ranks: Vec<usize>,
    /// Additional ranks derived as fractions of the active state dimension.
    pub spectral_rank_fractions: Vec<f64>,
    /// Explicit ceiling for the dense generalized eigensolver.
    pub spectral_reference_maximum_dimension: usize,
    /// Explicit ceiling for materializing the exact marginal spectral operator
    /// in bounded dense-reference routes. The mixed spectral-rank pilot does
    /// not consult this cap.
    pub spectral_operator_materialization_maximum_dimension: usize,
    /// Posterior samples included in each available full-UQ row.
    pub posterior_sample_count: usize,
    pub seed: u64,
}

impl C51BenchmarkConfig {
    /// Cheap 3D 1-form profile exercising local/global and amortization axes.
    pub fn smoke() -> Self {
        Self {
            refinements: vec![2],
            physical_observation_design: C51PhysicalObservationDesign::LegacyCellRows,
            physical_output_design: C51PhysicalOutputDesign::LegacyCentralRegionAverage,
            spatial_dimension: 3,
            form_degree: 1,
            alpha: MaternAlpha::Two,
            kappa: 4.0,
            tau: 1.0,
            observation_noise_variance: 0.05,
            local_sensor_counts: vec![32],
            local_sensor_density_targets: vec![0.02, 0.10],
            repeated_data_counts: vec![1, 2],
            include_global_row: true,
            include_local_selectors: true,
            physical_qoi_region_lower: [0.25; 3],
            physical_qoi_region_upper: [0.75; 3],
            spectral_low_ranks: vec![8],
            spectral_rank_fractions: Vec::new(),
            spectral_reference_maximum_dimension: 512,
            spectral_operator_materialization_maximum_dimension: 512,
            posterior_sample_count: 4,
            seed: 0xC510_2026,
        }
    }

    /// Claim-bearing resource profile with fixed-footprint physical-B sensors.
    ///
    /// This constructor is intentionally distinct from smoke. It never falls
    /// back to shrinking cell rows, and its requested observation counts are
    /// absolute scalar counts rather than fractions of `N`.
    pub fn juq_performance() -> Self {
        Self {
            refinements: vec![8, 10],
            physical_observation_design: C51PhysicalObservationDesign::JuqFixedFootprintPrefix,
            physical_output_design: C51PhysicalOutputDesign::RankStudySelectionAndHeldOut,
            local_sensor_counts: vec![32, 128, 512, 1_024, 2_048, 4_096, 5_184],
            local_sensor_density_targets: Vec::new(),
            repeated_data_counts: vec![1, 10],
            include_global_row: false,
            include_local_selectors: false,
            spectral_low_ranks: vec![2_000],
            spectral_rank_fractions: Vec::new(),
            spectral_reference_maximum_dimension: 512,
            spectral_operator_materialization_maximum_dimension: 6_000,
            posterior_sample_count: 4,
            seed: 0xC513_2026,
            ..Self::smoke()
        }
    }

    /// Bounded discovery ladder. It is not a preregistered evidence profile.
    pub fn reference_ladder() -> Self {
        Self {
            refinements: vec![2, 3, 4],
            local_sensor_counts: vec![32],
            local_sensor_density_targets: vec![0.02, 0.10],
            repeated_data_counts: vec![1, 10, 100],
            spectral_low_ranks: vec![8, 16, 32, 64],
            spectral_rank_fractions: Vec::new(),
            spectral_reference_maximum_dimension: 1_024,
            spectral_operator_materialization_maximum_dimension: 1_024,
            posterior_sample_count: 8,
            seed: 0xC511_2026,
            ..Self::smoke()
        }
    }

    /// Discovery-only large-rank pilot using one shared maximum partial basis.
    ///
    /// This profile is deliberately not a performance profile: one shared
    /// maximum basis is charged and archived, but excluded from
    /// method-performance comparisons. The basis is computed through the
    /// sparse mixed shell without materializing the k>0 marginal operator.
    pub fn spectral_rank_pilot() -> Self {
        Self {
            refinements: vec![8, 10],
            local_sensor_counts: vec![32],
            local_sensor_density_targets: vec![0.02, 0.10],
            repeated_data_counts: vec![1, 10],
            include_global_row: false,
            include_local_selectors: false,
            spectral_low_ranks: vec![32, 64, 128, 256, 512, 1_024, 2_000, 2_048, 4_096],
            spectral_rank_fractions: vec![0.01, 0.025, 0.05, 0.10, 0.25, 0.50],
            spectral_reference_maximum_dimension: 512,
            spectral_operator_materialization_maximum_dimension: 6_000,
            posterior_sample_count: 1,
            seed: 0xC512_2026,
            ..Self::smoke()
        }
    }

    fn validate(&self) -> Result<()> {
        if self.refinements.is_empty() || self.refinements.iter().any(|value| *value < 2) {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 refinements must contain values of at least two".to_string(),
            ));
        }
        if self.spatial_dimension != 3 || self.form_degree != 1 {
            return Err(FeecGmrfError::InvalidParameter(
                "the current C51 physical discriminator is the 3D Whitney 1-form A-to-B chain"
                    .to_string(),
            ));
        }
        if self.local_sensor_counts.is_empty() && self.local_sensor_density_targets.is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 requires at least one local sensor count or density".to_string(),
            ));
        }
        if self.local_sensor_counts.contains(&0) {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 local sensor counts must be positive".to_string(),
            ));
        }
        if self.physical_observation_design == C51PhysicalObservationDesign::JuqFixedFootprintPrefix
        {
            if self.physical_output_design != C51PhysicalOutputDesign::RankStudySelectionAndHeldOut
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "the JUQ fixed-footprint observation design requires the frozen rank-study selection and held-out output banks"
                        .to_string(),
                ));
            }
            if self.include_global_row
                || self.include_local_selectors
                || !self.local_sensor_density_targets.is_empty()
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "the JUQ fixed-footprint profile accepts only absolute local physical-B counts"
                        .to_string(),
                ));
            }
            if self
                .local_sensor_counts
                .iter()
                .any(|count| *count > C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT)
            {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "the JUQ fixed-footprint sensor capacity is {C51_JUQ_PERFORMANCE_SCALAR_OBSERVATION_COUNT} scalar observations"
                )));
            }
        }
        if self.physical_observation_design == C51PhysicalObservationDesign::LegacyCellRows
            && self.physical_output_design != C51PhysicalOutputDesign::LegacyCentralRegionAverage
        {
            return Err(FeecGmrfError::InvalidParameter(
                "legacy smoke observations require the legacy central-region physical output"
                    .to_string(),
            ));
        }
        if self.physical_observation_design == C51PhysicalObservationDesign::FluxCochainHashedPrefix
        {
            if self.physical_output_design
                != C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit
                || self.include_global_row
                || self.include_local_selectors
                || !self.local_sensor_density_targets.is_empty()
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "the direct flux-cochain profile requires exact surface-flux outputs and absolute local flux counts only"
                        .to_string(),
                ));
            }
        } else if self.physical_output_design
            == C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit
        {
            return Err(FeecGmrfError::InvalidParameter(
                "the integral surface-flux output design requires direct flux-cochain observations"
                    .to_string(),
            ));
        }
        if self
            .local_sensor_density_targets
            .iter()
            .any(|density| !density.is_finite() || *density <= 0.0 || *density > 1.0)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 local sensor densities must lie in (0, 1]".to_string(),
            ));
        }
        if self.repeated_data_counts.is_empty() || self.repeated_data_counts.contains(&0) {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 requires at least one positive repeated-data count".to_string(),
            ));
        }
        if self.spectral_low_ranks.contains(&0) {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 spectral low-rank grid may contain only positive ranks".to_string(),
            ));
        }
        if self
            .spectral_rank_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || *fraction <= 0.0 || *fraction >= 1.0)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 spectral rank fractions must lie in (0, 1)".to_string(),
            ));
        }
        if self.spectral_reference_maximum_dimension == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 spectral reference dimension cap must be positive".to_string(),
            ));
        }
        if self.spectral_operator_materialization_maximum_dimension == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 spectral operator materialization cap must be positive".to_string(),
            ));
        }
        if self.posterior_sample_count == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 posterior sample count must be positive".to_string(),
            ));
        }
        if self
            .physical_qoi_region_lower
            .iter()
            .zip(self.physical_qoi_region_upper)
            .any(|(lower, upper)| {
                !lower.is_finite()
                    || !upper.is_finite()
                    || *lower < 0.0
                    || upper > 1.0
                    || *lower >= upper
            })
        {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 physical-QoI region bounds must define a nonempty subset of the unit cube"
                    .to_string(),
            ));
        }
        MaternParameters::new(self.alpha, self.kappa, self.tau)?;
        GaussianNoise::variance(self.observation_noise_variance)?;
        Ok(())
    }
}

/// One method/observation/amortization row in the benchmark archive.
#[derive(Debug, Clone, PartialEq)]
pub struct C51BenchmarkRow {
    pub method: C51MethodId,
    pub available: bool,
    pub unavailable_reason: String,
    pub discrete_law_id: String,
    pub same_discrete_law_eligible: bool,
    pub matched_physical_qoi_eligible: bool,
    /// Whether the row executed the common physical-output-only UQ workload
    /// used for claim-bearing method comparisons. This is independent of the
    /// row's separately validated accuracy qualification.
    pub common_physical_uq_workload_eligible: bool,
    pub full_uq_workload_eligible: bool,
    /// Whether the recorded construction/setup timings are eligible for a
    /// method-performance comparison. Shared maximum-rank discovery bases are
    /// explicitly ineligible even though their cost is archived.
    pub method_performance_timing_eligible: bool,
    /// Empty for eligible rows; otherwise the first failed timing invariant.
    pub method_performance_timing_ineligibility_reason: String,
    /// Construction route for a spectral basis, or `not_applicable`.
    pub spectral_basis_source: String,
    /// Eigensolver backend for a spectral basis, or `not_applicable`.
    pub spectral_eigensolver_backend: String,
    /// External helper workspace retained for diagnosis, when applicable.
    pub spectral_helper_workspace: String,
    /// External helper binary used for the solve, when applicable.
    pub spectral_helper_binary: String,
    /// Whether the spectral solve reused the FEEC assembly already charged in
    /// `feec_assembly_ns`, or explicitly reassembled compatible mixed blocks.
    pub spectral_feec_assembly_reuse: String,
    pub spectral_partial_which: String,
    pub spectral_partial_mass_solve: String,
    pub refinement: usize,
    pub spatial_dimension: usize,
    pub form_degree: usize,
    pub alpha: u32,
    pub kappa: f64,
    pub tau: f64,
    pub amplitude_calibration_policy: String,
    pub amplitude_calibration_target_prior_rms: C51Measurement<f64>,
    pub amplitude_calibration_uncalibrated_prior_rms: C51Measurement<f64>,
    pub amplitude_calibration_precision_scale: C51Measurement<f64>,
    pub amplitude_calibration_tau_multiplier: C51Measurement<f64>,
    pub amplitude_calibration_ns: C51Measurement<u128>,
    pub observation_class: C51ObservationClass,
    /// Immutable geometry/design identifier for the observation rows.
    pub observation_design_id: String,
    pub observation_fixed_radius: C51Measurement<f64>,
    pub observation_vector_center_count: C51Measurement<usize>,
    pub observation_component_x_count: C51Measurement<usize>,
    pub observation_component_y_count: C51Measurement<usize>,
    pub observation_component_z_count: C51Measurement<usize>,
    pub physical_qoi_design_id: String,
    pub physical_qoi_output_count: usize,
    pub physical_qoi_selected_cell_count: usize,
    pub physical_qoi_selected_volume_fraction: f64,
    pub sensor_count: usize,
    pub sensor_density: f64,
    pub repeated_data_count: usize,
    pub posterior_sample_count: usize,
    pub seed: u64,
    pub latent_dimension: usize,
    pub ambient_dimension: usize,
    pub constraint_dimension: usize,
    pub observation_nnz: usize,
    pub prior_precision_nnz: C51Measurement<usize>,
    pub observation_update_nnz: C51Measurement<usize>,
    pub posterior_precision_nnz: C51Measurement<usize>,
    pub ambient_energy_nnz: C51Measurement<usize>,
    pub constraint_nnz: C51Measurement<usize>,
    pub kkt_input_nnz: C51Measurement<usize>,
    pub factor_storage_nnz: C51Measurement<usize>,
    pub certificate_factor_storage_nnz: C51Measurement<usize>,
    pub fill_ratio: C51Measurement<f64>,
    /// Dense exact-GP diagnostics. These are deliberately separate from sparse
    /// factor nonzeros and spectral feature storage.
    pub exact_gp_dense_factor_dimension: C51Measurement<usize>,
    pub exact_gp_dense_factor_allocated_values: C51Measurement<usize>,
    pub exact_gp_dense_factor_triangular_values: C51Measurement<usize>,
    pub exact_gp_output_observation_covariance_values: C51Measurement<usize>,
    pub exact_gp_posterior_output_covariance_values: C51Measurement<usize>,
    pub exact_gp_persistent_storage_values: C51Measurement<usize>,
    pub exact_gp_posterior_covariance_max_abs_skew: C51Measurement<f64>,
    pub exact_gp_prior_covariance_raw_max_relative_skew: C51Measurement<f64>,
    pub exact_gp_prior_covariance_projection_frobenius_correction: C51Measurement<f64>,
    pub exact_gp_prior_covariance_audit_residual_aware: C51Measurement<bool>,
    pub exact_gp_prior_covariance_audit_policy: String,
    pub exact_gp_prior_covariance_audit_status: String,
    pub exact_gp_prior_covariance_audit_residual_status: String,
    pub spectral_rank: C51Measurement<usize>,
    pub spectral_basis_storage_values: C51Measurement<usize>,
    pub spectral_gp_storage_values: C51Measurement<usize>,
    pub spectral_prepared_storage_values: C51Measurement<usize>,
    pub spectral_dense_factor_values: C51Measurement<usize>,
    pub spectral_state_feature_values: C51Measurement<usize>,
    pub spectral_observation_feature_values: C51Measurement<usize>,
    pub spectral_sampling_normal_dimension: C51Measurement<usize>,
    pub spectral_rank_fraction: C51Measurement<f64>,
    pub spectral_shared_maximum_basis_rank: C51Measurement<usize>,
    pub spectral_partial_requested_rank: C51Measurement<usize>,
    pub spectral_partial_converged_rank: C51Measurement<usize>,
    pub spectral_partial_retained_rank: C51Measurement<usize>,
    pub spectral_partial_l_nnz: C51Measurement<usize>,
    pub spectral_partial_d_nnz: C51Measurement<usize>,
    pub spectral_partial_c_nnz: C51Measurement<usize>,
    pub spectral_partial_mkm1_nnz: C51Measurement<usize>,
    pub spectral_partial_mk_nnz: C51Measurement<usize>,
    pub spectral_partial_mass_nnz: C51Measurement<usize>,
    pub spectral_partial_weak_hodge_laplacian_nnz: C51Measurement<usize>,
    pub spectral_partial_input_nnz: C51Measurement<usize>,
    pub spectral_partial_basis_storage_bytes: C51Measurement<usize>,
    pub spectral_partial_clamped_negative_eigenvalues: C51Measurement<usize>,
    pub spectral_partial_maximum_relative_mixed_residual: C51Measurement<f64>,
    pub spectral_partial_maximum_mass_orthogonality_defect: C51Measurement<f64>,
    pub spectral_partial_mass_orthogonality_tolerance: C51Measurement<f64>,
    pub spectral_partial_eigenvalue_negative_tolerance: C51Measurement<f64>,
    pub spectral_partial_eigensolver_tolerance: C51Measurement<f64>,
    pub spectral_partial_residual_validation_ns: C51Measurement<u128>,
    pub solver: String,
    pub ordering: String,
    pub feec_assembly_ns: C51Measurement<u128>,
    pub prior_build_ns: C51Measurement<u128>,
    /// FEEC supported-law factorization plus requested Kyy/Kgy/Kgg actions.
    pub exact_gp_covariance_construction_ns: C51Measurement<u128>,
    /// Dense observation-space Cholesky and posterior output covariance.
    pub exact_gp_dense_conditioning_ns: C51Measurement<u128>,
    /// FEEC assembly, supported-law construction, covariance blocks, and dense
    /// conditioning charged together.
    pub exact_gp_end_to_end_setup_ns: C51Measurement<u128>,
    pub spectral_operator_recovery_ns: C51Measurement<u128>,
    pub spectral_eigensolve_ns: C51Measurement<u128>,
    pub spectral_gp_build_ns: C51Measurement<u128>,
    pub observation_update_assembly_ns: C51Measurement<u128>,
    pub observation_update_and_factor_ns: C51Measurement<u128>,
    pub symbolic_analysis_ns: C51Measurement<u128>,
    pub numeric_factor_ns: C51Measurement<u128>,
    pub combined_factor_setup_ns: C51Measurement<u128>,
    pub repeated_data_ns: C51Measurement<u128>,
    pub covariance_action_ns: C51Measurement<u128>,
    pub physical_qoi_ns: C51Measurement<u128>,
    pub sampling_ns: C51Measurement<u128>,
    pub variance_estimation_ns: C51Measurement<u128>,
    pub factor_external_peak_rss_bytes: C51Measurement<u64>,
    pub query_external_peak_rss_bytes: C51Measurement<u64>,
    pub posterior_mean_relative_error: C51Measurement<f64>,
    pub covariance_action_relative_error: C51Measurement<f64>,
    pub physical_covariance_action_relative_error: C51Measurement<f64>,
    /// Error in the common output-space action `G C_post G^T w`.
    pub physical_output_covariance_action_relative_error: C51Measurement<f64>,
    pub physical_variance_relative_error: C51Measurement<f64>,
    pub physical_transformed_variance_relative_error: C51Measurement<f64>,
    pub physical_mean_relative_error: C51Measurement<f64>,
    /// Minimum raw transformed variance before any downstream presentation policy.
    pub physical_transformed_variance_min_raw: C51Measurement<f64>,
    /// Count of raw transformed variances below zero; values are never clamped here.
    pub physical_transformed_variance_negative_count: C51Measurement<usize>,
}

impl C51BenchmarkRow {
    /// Attach the peak RSS measured by a fresh-process factor/setup monitor.
    pub fn record_factor_external_peak_rss(&mut self, bytes: u64) {
        self.factor_external_peak_rss_bytes = C51Measurement::Measured(bytes);
    }

    /// Attach the peak RSS measured by a fresh-process query monitor.
    pub fn record_query_external_peak_rss(&mut self, bytes: u64) {
        self.query_external_peak_rss_bytes = C51Measurement::Measured(bytes);
    }
}

/// Validated run result with stable CSV and manifest writers.
#[derive(Debug, Clone, PartialEq)]
pub struct C51BenchmarkArchive {
    pub metadata: C51RunMetadata,
    pub config: C51BenchmarkConfig,
    pub comparison_frames: [C51ComparisonFrame; 2],
    pub rows: Vec<C51BenchmarkRow>,
}

impl C51BenchmarkArchive {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.rows.is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 archive has no rows".to_string(),
            ));
        }
        for row in &self.rows {
            if row.method_performance_timing_eligible
                != row
                    .method_performance_timing_ineligibility_reason
                    .is_empty()
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "C51 timing eligibility and its ineligibility reason disagree".to_string(),
                ));
            }
            if !row.available || row.latent_dimension == 0 || row.observation_nnz == 0 {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "available C51 method {} has an incomplete row",
                    row.method.as_str()
                )));
            }
            match row.physical_qoi_design_id.as_str() {
                "legacy_central_region_volume_average_3"
                    if row.physical_qoi_output_count == 3
                        && row.physical_qoi_selected_cell_count > 0
                        && row.physical_qoi_selected_volume_fraction.is_finite()
                        && row.physical_qoi_selected_volume_fraction > 0.0
                        && row.physical_qoi_selected_volume_fraction < 1.0 => {}
                "rank_study_selection_and_held_out_wendland_c2_18"
                    if row.physical_qoi_output_count == 18
                        && row.physical_qoi_selected_cell_count == 0
                        && row.physical_qoi_selected_volume_fraction == 0.0 => {}
                _ => {
                    return Err(FeecGmrfError::InvalidParameter(
                        "C51 physical-QoI design metadata is missing or inconsistent".to_string(),
                    ))
                }
            }
            if row.observation_design_id == C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID
                && (!matches!(row.observation_fixed_radius, C51Measurement::Measured(value) if value == C51_FIXED_OBSERVATION_RADIUS)
                    || !matches!(
                        row.observation_vector_center_count,
                        C51Measurement::Measured(_)
                    )
                    || !matches!(
                        row.observation_component_x_count,
                        C51Measurement::Measured(_)
                    )
                    || !matches!(
                        row.observation_component_y_count,
                        C51Measurement::Measured(_)
                    )
                    || !matches!(
                        row.observation_component_z_count,
                        C51Measurement::Measured(_)
                    ))
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "C51 JUQ fixed-footprint observation metadata is incomplete".to_string(),
                ));
            }
            if row.method == C51MethodId::FullMetricAugmented {
                if !matches!(row.factor_storage_nnz, C51Measurement::Unavailable { .. })
                    || !matches!(row.kkt_input_nnz, C51Measurement::Measured(_))
                {
                    return Err(FeecGmrfError::InvalidParameter(
                        "FM-AUG must distinguish KKT input nnz from unavailable LU fill"
                            .to_string(),
                    ));
                }
            } else if row.method == C51MethodId::ExactObservationSpaceGp {
                if !row.same_discrete_law_eligible
                    || exact_gp_maximum_physical_error(row) > 1.0e-8
                    || !matches!(row.factor_storage_nnz, C51Measurement::Unavailable { .. })
                    || !matches!(
                        row.exact_gp_dense_factor_dimension,
                        C51Measurement::Measured(value) if value == row.sensor_count
                    )
                    || !matches!(
                        row.exact_gp_dense_factor_allocated_values,
                        C51Measurement::Measured(value)
                            if value == row.sensor_count * row.sensor_count
                    )
                    || !matches!(
                        row.exact_gp_persistent_storage_values,
                        C51Measurement::Measured(_)
                    )
                    || !matches!(
                        row.exact_gp_prior_covariance_raw_max_relative_skew,
                        C51Measurement::Measured(_)
                    )
                    || !matches!(row.sampling_ns, C51Measurement::Unavailable { .. })
                {
                    return Err(FeecGmrfError::InvalidParameter(
                        "exact observation-space GP row is missing dense storage or covariance-audit diagnostics"
                            .to_string(),
                    ));
                }
            } else if matches!(
                row.method,
                C51MethodId::SpectralFullRank | C51MethodId::SpectralLowRank
            ) {
                if !matches!(
                    row.spectral_prepared_storage_values,
                    C51Measurement::Measured(_)
                ) || !matches!(row.sampling_ns, C51Measurement::Measured(_))
                    || !matches!(row.variance_estimation_ns, C51Measurement::Measured(_))
                {
                    return Err(FeecGmrfError::InvalidParameter(format!(
                        "spectral method {} is missing prepared storage or full-UQ work",
                        row.method.as_str()
                    )));
                }
                if row.method == C51MethodId::SpectralFullRank
                    && (!row.same_discrete_law_eligible || full_rank_maximum_error(row) > 1.0e-8)
                {
                    return Err(FeecGmrfError::InvalidParameter(
                        "full-rank spectral reference does not reproduce the exact finite law"
                            .to_string(),
                    ));
                }
                if row.method == C51MethodId::SpectralLowRank && row.same_discrete_law_eligible {
                    return Err(FeecGmrfError::InvalidParameter(
                        "a truncated spectral row cannot be labeled as the same finite law"
                            .to_string(),
                    ));
                }
                if !matches!(row.spectral_rank_fraction, C51Measurement::Measured(_)) {
                    return Err(FeecGmrfError::InvalidParameter(
                        "a spectral C51 row must record its actual rank fraction".to_string(),
                    ));
                }
                if row.spectral_basis_source == "partial_shared_maximum_mixed_shell"
                    && (row.method_performance_timing_eligible
                        || !matches!(
                            row.spectral_partial_converged_rank,
                            C51Measurement::Measured(_)
                        )
                        || !matches!(row.spectral_partial_mass_nnz, C51Measurement::Measured(_))
                        || !matches!(row.spectral_partial_input_nnz, C51Measurement::Measured(_))
                        || !matches!(
                            &row.spectral_partial_weak_hodge_laplacian_nnz,
                            C51Measurement::Unavailable { reason }
                                if reason == MIXED_MARGINAL_NNZ_UNAVAILABLE_REASON
                        ))
                {
                    return Err(FeecGmrfError::InvalidParameter(
                        "a shared-maximum partial spectral row must archive solver diagnostics and be timing-ineligible"
                            .to_string(),
                    ));
                }
                if row.spectral_basis_source == "partial_exact_requested_mixed_shell"
                    && (!matches!(
                        row.spectral_partial_retained_rank,
                        C51Measurement::Measured(_)
                    ) || !matches!(
                        row.spectral_partial_maximum_relative_mixed_residual,
                        C51Measurement::Measured(_)
                    ) || !matches!(
                        row.spectral_partial_maximum_mass_orthogonality_defect,
                        C51Measurement::Measured(_)
                    ) || [
                        &row.spectral_partial_l_nnz,
                        &row.spectral_partial_d_nnz,
                        &row.spectral_partial_c_nnz,
                        &row.spectral_partial_mkm1_nnz,
                        &row.spectral_partial_mk_nnz,
                    ]
                    .into_iter()
                    .any(|measurement| !matches!(measurement, C51Measurement::Measured(_)))
                        || row.spectral_feec_assembly_reuse == "unclassified")
                {
                    return Err(FeecGmrfError::InvalidParameter(
                        "an exact-request mixed spectral row must archive rank, certificate, and assembly diagnostics"
                            .to_string(),
                    ));
                }
            } else {
                if !matches!(row.factor_storage_nnz, C51Measurement::Measured(_)) {
                    return Err(FeecGmrfError::InvalidParameter(format!(
                        "SPD method {} must record actual factor storage",
                        row.method.as_str()
                    )));
                }
                if !matches!(row.observation_update_nnz, C51Measurement::Measured(_))
                    || !matches!(row.sampling_ns, C51Measurement::Measured(_))
                    || !matches!(row.variance_estimation_ns, C51Measurement::Measured(_))
                {
                    return Err(FeecGmrfError::InvalidParameter(format!(
                        "SPD method {} is missing update, sample, or transformed-variance work",
                        row.method.as_str()
                    )));
                }
            }
        }
        if !self.rows.iter().any(|row| {
            row.available && row.observation_class == C51ObservationClass::LocalPhysicalB
        }) {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 archive is missing the claim-bearing local physical-B observation".to_string(),
            ));
        }
        Ok(())
    }

    pub fn write_directory(&self, directory: &Path) -> Result<()> {
        self.validate()?;
        fs::create_dir_all(directory)?;
        self.write_csv(&directory.join("rows.csv"))?;
        self.write_manifest(&directory.join("manifest.yaml"))?;
        Ok(())
    }

    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "{}", csv_header())?;
        for row in &self.rows {
            writeln!(writer, "{}", row_to_csv(row))?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_manifest(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version: {}",
            yaml_scalar(C51_SCHEMA_VERSION)
        )?;
        writeln!(writer, "study_id: {}", yaml_scalar(C51_STUDY_ID))?;
        writeln!(writer, "profile: {}", yaml_scalar(&self.metadata.profile))?;
        writeln!(
            writer,
            "compiler_profile: {}",
            yaml_scalar(&self.metadata.compiler_profile)
        )?;
        writeln!(
            writer,
            "sparse_backend: {}",
            yaml_scalar(&self.metadata.sparse_backend)
        )?;
        writeln!(writer, "command: {}", yaml_scalar(&self.metadata.command))?;
        write_yaml_option(&mut writer, "root_sha", self.metadata.root_sha.as_deref())?;
        write_yaml_option(&mut writer, "feec_sha", self.metadata.feec_sha.as_deref())?;
        write_yaml_option(&mut writer, "gmrf_sha", self.metadata.gmrf_sha.as_deref())?;
        write_yaml_bool_option(&mut writer, "root_dirty", self.metadata.root_dirty)?;
        write_yaml_bool_option(&mut writer, "feec_dirty", self.metadata.feec_dirty)?;
        write_yaml_bool_option(&mut writer, "gmrf_dirty", self.metadata.gmrf_dirty)?;
        write_yaml_option(
            &mut writer,
            "rustc_version",
            self.metadata.rustc_version.as_deref(),
        )?;
        writeln!(writer, "comparison_frames:")?;
        for frame in self.comparison_frames {
            writeln!(writer, "  - {}", yaml_scalar(frame.as_str()))?;
        }
        writeln!(writer, "study_status: discovery_in_process")?;
        writeln!(writer, "physical_covariance_error_envelopes:")?;
        writeln!(
            writer,
            "  claim: {}",
            C51_CLAIM_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT
        )?;
        writeln!(
            writer,
            "  sensitivity: {}",
            C51_SENSITIVITY_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT
        )?;
        writeln!(
            writer,
            "  context: {}",
            C51_CONTEXT_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT
        )?;
        writeln!(writer, "axes:")?;
        writeln!(
            writer,
            "  physical_observation_design: {}",
            yaml_scalar(self.config.physical_observation_design.as_str())
        )?;
        writeln!(
            writer,
            "  physical_output_design: {}",
            yaml_scalar(self.config.physical_output_design.as_str())
        )?;
        if self.config.physical_observation_design
            == C51PhysicalObservationDesign::FluxCochainHashedPrefix
        {
            writeln!(writer, "  cochain_row_semantics: exact_integral_flux_dofs")?;
            writeln!(writer, "  physical_chain: A_active_to_D1_to_B_flux_cochain")?;
            writeln!(writer, "  cell_barycenter_reconstruction_used: false")?;
            writeln!(
                writer,
                "  prior_normalization: frozen_thesis_domain_averaged_M2_D1_Qinv_D1T"
            )?;
            writeln!(writer, "  normalization_source_parent_commit: 5eef534")?;
        } else {
            writeln!(writer, "  cochain_row_semantics: null")?;
            writeln!(writer, "  physical_chain: reconstructed_physical_outputs")?;
            writeln!(writer, "  cell_barycenter_reconstruction_used: true")?;
            writeln!(writer, "  prior_normalization: profile_defined")?;
            writeln!(writer, "  normalization_source_parent_commit: null")?;
        }
        writeln!(
            writer,
            "  refinements: {}",
            yaml_usize_list(&self.config.refinements)
        )?;
        writeln!(
            writer,
            "  local_sensor_counts: {}",
            yaml_usize_list(&self.config.local_sensor_counts)
        )?;
        writeln!(
            writer,
            "  local_sensor_density_targets: {}",
            yaml_f64_list(&self.config.local_sensor_density_targets)
        )?;
        writeln!(
            writer,
            "  repeated_data_counts: {}",
            yaml_usize_list(&self.config.repeated_data_counts)
        )?;
        writeln!(
            writer,
            "  include_global_row: {}",
            self.config.include_global_row
        )?;
        writeln!(
            writer,
            "  include_local_selectors: {}",
            self.config.include_local_selectors
        )?;
        writeln!(
            writer,
            "  physical_qoi_region_lower: {}",
            yaml_f64_list(&self.config.physical_qoi_region_lower)
        )?;
        writeln!(
            writer,
            "  physical_qoi_region_upper: {}",
            yaml_f64_list(&self.config.physical_qoi_region_upper)
        )?;
        writeln!(
            writer,
            "  spectral_low_ranks: {}",
            yaml_usize_list(&self.config.spectral_low_ranks)
        )?;
        writeln!(
            writer,
            "  spectral_rank_fractions: {}",
            yaml_f64_list(&self.config.spectral_rank_fractions)
        )?;
        writeln!(
            writer,
            "  spectral_reference_maximum_dimension: {}",
            self.config.spectral_reference_maximum_dimension
        )?;
        writeln!(
            writer,
            "  spectral_operator_materialization_maximum_dimension: {}",
            self.config
                .spectral_operator_materialization_maximum_dimension
        )?;
        writeln!(
            writer,
            "  posterior_sample_count: {}",
            self.config.posterior_sample_count
        )?;
        let mut observation_classes = vec!["local_physical_b"];
        if self.config.include_local_selectors {
            observation_classes.push("local_selectors");
        }
        if self.config.include_global_row {
            observation_classes.push("global_row");
        }
        writeln!(
            writer,
            "  observation_classes: [{}]",
            observation_classes
                .into_iter()
                .map(yaml_scalar)
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(writer, "timing_interpretation:")?;
        writeln!(
            writer,
            "  shared_maximum_partial_basis: discovery_only_not_method_performance"
        )?;
        writeln!(
            writer,
            "  exact_marginal_operator_materialization: dense_reference_routes_only"
        )?;
        writeln!(writer, "spectral_operator_route:")?;
        writeln!(
            writer,
            "  pilot: sparse_mixed_shell_without_marginal_materialization"
        )?;
        writeln!(
            writer,
            "  scalable_production_status: exact_requested_mixed_resource_route_available"
        )?;
        writeln!(
            writer,
            "  exact_requested_residual_tolerance: {}",
            EXACT_REQUESTED_MIXED_RESIDUAL_TOLERANCE
        )?;
        writeln!(
            writer,
            "  exact_requested_eigensolver_tolerance: {}",
            EXACT_REQUESTED_MIXED_EIGENSOLVER_TOLERANCE
        )?;
        writeln!(
            writer,
            "  exact_requested_mass_orthogonality_tolerance: {}",
            EXACT_REQUESTED_MIXED_MASS_ORTHOGONALITY_TOLERANCE
        )?;
        writeln!(
            writer,
            "  exact_requested_eigenvalue_negative_tolerance: {}",
            EXACT_REQUESTED_MIXED_EIGENVALUE_NEGATIVE_TOLERANCE
        )?;
        writeln!(
            writer,
            "  mixed_adapter_assembly_reuse: false_reassembly_is_charged_and_archived"
        )?;
        writeln!(writer, "  seed: {}", self.config.seed)?;
        writeln!(writer, "hardware:")?;
        write_yaml_nested_option(&mut writer, "host", self.metadata.hardware.host.as_deref())?;
        write_yaml_nested_option(
            &mut writer,
            "cpu_model",
            self.metadata.hardware.cpu_model.as_deref(),
        )?;
        write_yaml_nested_usize_option(
            &mut writer,
            "logical_cores",
            self.metadata.hardware.logical_cores,
        )?;
        write_yaml_nested_u64_option(
            &mut writer,
            "total_memory_bytes",
            self.metadata.hardware.total_memory_bytes,
        )?;
        write_yaml_nested_option(
            &mut writer,
            "operating_system",
            self.metadata.hardware.operating_system.as_deref(),
        )?;
        write_yaml_nested_option(
            &mut writer,
            "kernel",
            self.metadata.hardware.kernel.as_deref(),
        )?;
        write_yaml_nested_option(
            &mut writer,
            "thread_affinity",
            self.metadata.hardware.thread_affinity.as_deref(),
        )?;
        writeln!(writer, "rows_csv: rows.csv")?;
        writeln!(writer, "row_count: {}", self.rows.len())?;
        writer.flush()?;
        Ok(())
    }
}

/// One-row archive for a method- and phase-isolated fresh process.
#[derive(Debug, Clone, PartialEq)]
pub struct C51ResourceArchive {
    pub metadata: C51RunMetadata,
    pub config: C51ResourceConfig,
    pub row: C51BenchmarkRow,
}

impl C51ResourceArchive {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        let selection = &self.config.selection;
        if self.row.method != selection.method
            || self.row.refinement != selection.refinement
            || self.row.observation_class != selection.observation_class
            || self.row.repeated_data_count != selection.repeated_data_count
        {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 resource row does not match its single method/observation/workload selection"
                    .to_string(),
            ));
        }
        let expected_sensor_count = match selection.sensor_target {
            C51ResourceSensorTarget::Count(count) => count,
            C51ResourceSensorTarget::Density(density) => {
                (density * self.row.latent_dimension as f64).ceil() as usize
            }
        };
        if self.row.sensor_count != expected_sensor_count {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "C51 resource sensor target resolves to {expected_sensor_count}, but the selected observation has {} rows",
                self.row.sensor_count
            )));
        }
        if self.config.physical_observation_design
            == C51PhysicalObservationDesign::JuqFixedFootprintPrefix
            && (self.row.observation_design_id != C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID
                || self.row.physical_qoi_design_id
                    != C51PhysicalOutputDesign::RankStudySelectionAndHeldOut.as_str()
                || self.row.physical_qoi_output_count != 18)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 JUQ resource row did not retain the frozen observation/output designs"
                    .to_string(),
            ));
        }
        if self.config.physical_observation_design
            == C51PhysicalObservationDesign::FluxCochainHashedPrefix
            && (self.row.observation_design_id != C51_FLUX_COCHAIN_DESIGN_ID
                || self.row.physical_qoi_design_id != C51_SURFACE_FLUX_QOI_DESIGN_ID
                || self.row.physical_qoi_output_count != 12)
        {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 integral-flux resource row did not retain the frozen cochain observation/output designs"
                    .to_string(),
            ));
        }
        if selection.method == C51MethodId::WhitneyCompactCalibrated {
            let measured = |measurement: &C51Measurement<f64>| matches!(measurement, C51Measurement::Measured(value) if value.is_finite() && *value > 0.0);
            if self.row.amplitude_calibration_policy
                != "exact_feec_prior_selection_bank_physical_rms_before_conditioning"
                || !measured(&self.row.amplitude_calibration_target_prior_rms)
                || !measured(&self.row.amplitude_calibration_uncalibrated_prior_rms)
                || !measured(&self.row.amplitude_calibration_precision_scale)
                || !measured(&self.row.amplitude_calibration_tau_multiplier)
                || !matches!(&self.row.amplitude_calibration_ns, C51Measurement::Measured(value) if *value > 0)
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "C51 calibrated projected resource row is missing prior calibration provenance"
                        .to_string(),
                ));
            }
            if let (C51Measurement::Measured(scale), C51Measurement::Measured(tau_multiplier)) = (
                &self.row.amplitude_calibration_precision_scale,
                &self.row.amplitude_calibration_tau_multiplier,
            ) {
                if (tau_multiplier.powi(2) - *scale).abs() > 1.0e-12 * *scale {
                    return Err(FeecGmrfError::InvalidParameter(
                        "C51 calibrated projected precision and tau scales disagree".to_string(),
                    ));
                }
            }
        } else if selection.method == C51MethodId::WhitneyCompactIntegralFluxCalibrated {
            let measured = |measurement: &C51Measurement<f64>| matches!(measurement, C51Measurement::Measured(value) if value.is_finite() && *value > 0.0);
            if self.row.amplitude_calibration_policy
                != "frozen_exact_reference_posterior_surface_flux_rms_by_prior_precision"
                || self.row.amplitude_calibration_target_prior_rms != C51Measurement::NotRecorded
                || self.row.amplitude_calibration_uncalibrated_prior_rms
                    != C51Measurement::NotRecorded
                || !measured(&self.row.amplitude_calibration_precision_scale)
                || !measured(&self.row.amplitude_calibration_tau_multiplier)
                || !matches!(
                    &self.row.amplitude_calibration_ns,
                    C51Measurement::Unavailable { .. }
                )
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "C51 integral-flux calibrated resource row is missing frozen calibration provenance"
                        .to_string(),
                ));
            }
            if let (C51Measurement::Measured(scale), C51Measurement::Measured(tau_multiplier)) = (
                &self.row.amplitude_calibration_precision_scale,
                &self.row.amplitude_calibration_tau_multiplier,
            ) {
                if (tau_multiplier.powi(2) - *scale).abs() > 1.0e-12 * *scale {
                    return Err(FeecGmrfError::InvalidParameter(
                        "C51 integral-flux calibrated precision and tau scales disagree"
                            .to_string(),
                    ));
                }
                let expected = self
                    .config
                    .projected_integral_flux_posterior_precision_scale
                    .expect("validated calibrated integral-flux scale");
                if (*scale - expected).abs() > 1.0e-15 * expected {
                    return Err(FeecGmrfError::InvalidParameter(
                        "C51 integral-flux resource row did not apply its frozen precision scale"
                            .to_string(),
                    ));
                }
            }
        } else if self.row.amplitude_calibration_policy != "none"
            || self.row.amplitude_calibration_target_prior_rms != C51Measurement::NotRecorded
            || self.row.amplitude_calibration_uncalibrated_prior_rms != C51Measurement::NotRecorded
            || self.row.amplitude_calibration_precision_scale != C51Measurement::NotRecorded
            || self.row.amplitude_calibration_tau_multiplier != C51Measurement::NotRecorded
            || self.row.amplitude_calibration_ns != C51Measurement::NotRecorded
        {
            return Err(FeecGmrfError::InvalidParameter(
                "an uncalibrated C51 resource row contains amplitude calibration metadata"
                    .to_string(),
            ));
        }
        if selection.method == C51MethodId::ExactObservationSpaceGp
            && (!matches!(
                self.row.exact_gp_dense_factor_dimension,
                C51Measurement::Measured(value) if value == self.row.sensor_count
            ) || !matches!(
                self.row.exact_gp_persistent_storage_values,
                C51Measurement::Measured(_)
            ) || !matches!(
                self.row.factor_storage_nnz,
                C51Measurement::Unavailable { .. }
            ))
        {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 exact observation-space GP resource row is missing dense diagnostics"
                    .to_string(),
            ));
        }
        if selection.method == C51MethodId::SpectralLowRank {
            let selected_rank = selection
                .spectral_rank
                .expect("validated low-rank resource selection");
            if self.row.spectral_basis_source != "partial_exact_requested_mixed_shell"
                || self.row.spectral_rank != C51Measurement::Measured(selected_rank)
                || self.row.spectral_partial_requested_rank
                    != C51Measurement::Measured(selected_rank)
                || self.row.spectral_partial_retained_rank
                    != C51Measurement::Measured(selected_rank)
                || self.row.spectral_shared_maximum_basis_rank != C51Measurement::NotRecorded
                || [
                    &self.row.spectral_partial_l_nnz,
                    &self.row.spectral_partial_d_nnz,
                    &self.row.spectral_partial_c_nnz,
                    &self.row.spectral_partial_mkm1_nnz,
                    &self.row.spectral_partial_mk_nnz,
                ]
                .into_iter()
                .any(|measurement| !matches!(measurement, C51Measurement::Measured(_)))
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "a low-rank C51 resource row must come from one exact-request mixed solve at the selected rank"
                        .to_string(),
                ));
            }
        }
        for measurement in [
            &self.row.posterior_mean_relative_error,
            &self.row.covariance_action_relative_error,
            &self.row.physical_covariance_action_relative_error,
            &self.row.physical_output_covariance_action_relative_error,
            &self.row.physical_variance_relative_error,
            &self.row.physical_transformed_variance_relative_error,
            &self.row.physical_mean_relative_error,
        ] {
            if !matches!(measurement, C51Measurement::Unavailable { reason } if reason == RESOURCE_ACCURACY_UNAVAILABLE_REASON)
            {
                return Err(FeecGmrfError::InvalidParameter(
                    "C51 resource-only rows must mark every accuracy field unavailable".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn record_external_peak_rss(&mut self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 external peak RSS must be positive".to_string(),
            ));
        }
        match self.config.selection.phase {
            C51ResourcePhase::ColdSetup => self.row.record_factor_external_peak_rss(bytes),
            C51ResourcePhase::QueryPhysicalUq | C51ResourcePhase::QueryFullUq => {
                self.row.record_query_external_peak_rss(bytes)
            }
        }
        Ok(())
    }

    pub fn write_directory(&self, directory: &Path) -> Result<()> {
        self.validate()?;
        fs::create_dir_all(directory)?;
        self.write_csv(&directory.join("rows.csv"))?;
        self.write_manifest(&directory.join("manifest.yaml"))?;
        Ok(())
    }

    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "resource_scope,workload_phase,{}", csv_header())?;
        writeln!(
            writer,
            "method_phase,{},{}",
            self.config.selection.phase.as_str(),
            row_to_csv(&self.row)
        )?;
        writer.flush()?;
        Ok(())
    }

    pub fn write_manifest(&self, path: &Path) -> Result<()> {
        let selection = &self.config.selection;
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version: {}",
            yaml_scalar(C51_RESOURCE_SCHEMA_VERSION)
        )?;
        writeln!(writer, "study_id: {}", yaml_scalar(C51_STUDY_ID))?;
        writeln!(writer, "profile: {}", yaml_scalar(&self.metadata.profile))?;
        writeln!(writer, "resource_scope: method_phase")?;
        writeln!(writer, "workload_phase: {}", selection.phase.as_str())?;
        writeln!(writer, "selection:")?;
        writeln!(writer, "  refinement: {}", selection.refinement)?;
        writeln!(writer, "  method: {}", selection.method.as_str())?;
        writeln!(
            writer,
            "  observation_class: {}",
            selection.observation_class.as_str()
        )?;
        writeln!(
            writer,
            "  physical_observation_design: {}",
            yaml_scalar(self.config.physical_observation_design.as_str())
        )?;
        writeln!(
            writer,
            "  physical_output_design: {}",
            yaml_scalar(self.config.physical_output_design.as_str())
        )?;
        writeln!(
            writer,
            "  sensor_target_kind: {}",
            selection.sensor_target.kind()
        )?;
        writeln!(
            writer,
            "  sensor_target_value: {}",
            selection.sensor_target.value()
        )?;
        writeln!(writer, "  resolved_sensor_count: {}", self.row.sensor_count)?;
        writeln!(
            writer,
            "  resolved_sensor_density: {}",
            self.row.sensor_density
        )?;
        writeln!(
            writer,
            "  repeated_data_count: {}",
            selection.repeated_data_count
        )?;
        match selection.spectral_rank {
            Some(rank) => writeln!(writer, "  spectral_rank: {rank}")?,
            None => writeln!(writer, "  spectral_rank: null")?,
        }
        writeln!(writer, "amplitude_calibration:")?;
        writeln!(
            writer,
            "  policy: {}",
            yaml_scalar(&self.row.amplitude_calibration_policy)
        )?;
        match self.config.projected_calibration_target_prior_rms {
            Some(value) => writeln!(writer, "  frozen_target_prior_rms: {value:.17e}")?,
            None => writeln!(writer, "  frozen_target_prior_rms: null")?,
        }
        match self.config.projected_integral_flux_target_posterior_rms {
            Some(value) => writeln!(writer, "  frozen_target_posterior_rms: {value:.17e}")?,
            None => writeln!(writer, "  frozen_target_posterior_rms: null")?,
        }
        match self
            .config
            .projected_integral_flux_posterior_precision_scale
        {
            Some(value) => writeln!(writer, "  frozen_posterior_precision_scale: {value:.17e}")?,
            None => writeln!(writer, "  frozen_posterior_precision_scale: null")?,
        }
        if selection.method == C51MethodId::WhitneyCompactIntegralFluxCalibrated {
            writeln!(
                writer,
                "  stage: prior_precision_scaled_before_conditioning_from_frozen_posterior_target"
            )?;
            writeln!(writer, "  source_parent_commit: 5eef534")?;
        } else {
            writeln!(writer, "  stage: prior_before_conditioning")?;
            writeln!(writer, "  source_parent_commit: null")?;
        }
        writeln!(writer, "  calibration_bank: selection")?;
        writeln!(writer, "  held_out_used_for_calibration: false")?;
        writeln!(
            writer,
            "  cost_included_in_prior_build_ns: {}",
            selection.method != C51MethodId::WhitneyCompactIntegralFluxCalibrated
        )?;
        writeln!(
            writer,
            "accuracy_status: {}",
            yaml_scalar(RESOURCE_ACCURACY_UNAVAILABLE_REASON)
        )?;
        writeln!(writer, "spectral_route:")?;
        writeln!(
            writer,
            "  basis_source: {}",
            yaml_scalar(&self.row.spectral_basis_source)
        )?;
        writeln!(
            writer,
            "  feec_assembly_reuse: {}",
            yaml_scalar(&self.row.spectral_feec_assembly_reuse)
        )?;
        writeln!(
            writer,
            "  timing_eligible: {}",
            self.row.method_performance_timing_eligible
        )?;
        writeln!(
            writer,
            "  timing_ineligibility_reason: {}",
            yaml_scalar(&self.row.method_performance_timing_ineligibility_reason)
        )?;
        writeln!(
            writer,
            "  mixed_residual_tolerance: {}",
            EXACT_REQUESTED_MIXED_RESIDUAL_TOLERANCE
        )?;
        writeln!(
            writer,
            "  eigensolver_tolerance: {}",
            EXACT_REQUESTED_MIXED_EIGENSOLVER_TOLERANCE
        )?;
        writeln!(
            writer,
            "  mass_orthogonality_tolerance: {}",
            EXACT_REQUESTED_MIXED_MASS_ORTHOGONALITY_TOLERANCE
        )?;
        writeln!(
            writer,
            "  eigenvalue_negative_tolerance: {}",
            EXACT_REQUESTED_MIXED_EIGENVALUE_NEGATIVE_TOLERANCE
        )?;
        writeln!(writer, "rows_csv: rows.csv")?;
        writeln!(writer, "row_count: 1")?;
        writer.flush()?;
        Ok(())
    }
}

/// Execute the cheap benchmark through the root `feec_gmrf` public API.
pub fn run_c51_benchmark(
    config: &C51BenchmarkConfig,
    metadata: C51RunMetadata,
) -> Result<C51BenchmarkArchive> {
    config.validate()?;
    let mut rows = Vec::new();
    for &refinement in &config.refinements {
        rows.extend(run_refinement(config, refinement)?);
    }
    let archive = C51BenchmarkArchive {
        metadata,
        config: config.clone(),
        comparison_frames: [
            C51ComparisonFrame::SameDiscreteLawRepresentation,
            C51ComparisonFrame::MatchedPhysicalQoiMethod,
        ],
        rows,
    };
    archive.validate()?;
    Ok(archive)
}

/// Accuracy-only hook for the common JUQ fixed-footprint observation and
/// physical-output designs.
///
/// This route deliberately omits eigensolvers and the bounded dense eliminated
/// prior. The exact-supported law is the reference for both WHITNEY-COMPACT
/// and the exact observation-space GP. It is the qualification route that must
/// accompany performance rows, whose accuracy fields are intentionally
/// unavailable.
pub fn run_c51_juq_accuracy(
    config: &C51BenchmarkConfig,
    metadata: C51RunMetadata,
) -> Result<C51BenchmarkArchive> {
    config.validate()?;
    if config.physical_observation_design != C51PhysicalObservationDesign::JuqFixedFootprintPrefix
        || config.physical_output_design != C51PhysicalOutputDesign::RankStudySelectionAndHeldOut
    {
        return Err(FeecGmrfError::InvalidParameter(
            "C51 JUQ accuracy requires the frozen fixed-footprint observation and rank-study output designs"
                .to_string(),
        ));
    }
    let mut rows = Vec::new();
    for &refinement in &config.refinements {
        let assembled = assemble_refinement(config, refinement)?;
        let parameters = MaternParameters::new(config.alpha, config.kappa, config.tau)?;
        let priors = build_scalable_priors(
            config,
            &assembled.topology,
            &assembled.metric,
            &assembled.operators,
            parameters,
        )?;
        for design in assembled.designs {
            if design.class != C51ObservationClass::LocalPhysicalB {
                continue;
            }
            for &repeated_data_count in &config.repeated_data_counts {
                let (supported_row, reference) = run_supported_method(
                    &priors.supported,
                    priors.supported_build,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    None,
                )?;
                rows.push(supported_row);
                rows.push(
                    run_spd_method(
                        C51MethodId::WhitneyCompact,
                        &priors.compact,
                        priors.compact_build,
                        assembled.feec_assembly,
                        config,
                        refinement,
                        &design,
                        repeated_data_count,
                        &assembled.problem,
                        Some(&reference),
                    )?
                    .0,
                );
                rows.push(run_exact_observation_gp_method(
                    &priors.supported,
                    priors.supported_build,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    C51ResourcePhase::QueryFullUq,
                    Some(&reference),
                )?);
            }
        }
    }
    let archive = C51BenchmarkArchive {
        metadata,
        config: config.clone(),
        comparison_frames: [
            C51ComparisonFrame::SameDiscreteLawRepresentation,
            C51ComparisonFrame::MatchedPhysicalQoiMethod,
        ],
        rows,
    };
    archive.validate()?;
    Ok(archive)
}

/// Execute exactly one selected scientific method and phase.
///
/// Unlike the accuracy archive, this function never constructs an exact
/// comparison model or any unselected method. Accuracy fields are therefore
/// typed unavailable. This is the only entry point intended for external
/// fresh-process peak-RSS attribution.
pub fn run_c51_resource(
    config: &C51ResourceConfig,
    metadata: C51RunMetadata,
) -> Result<C51ResourceArchive> {
    config.validate()?;
    let benchmark = config.benchmark_config();
    let selection = &config.selection;
    if matches!(
        selection.method,
        C51MethodId::WhitneyCompact
            | C51MethodId::WhitneyCompactCalibrated
            | C51MethodId::WhitneyCompactIntegralFluxCalibrated
    ) {
        let mut row = run_isolated_compact_resource(
            &benchmark,
            selection,
            config.projected_calibration_target_prior_rms,
            config.projected_integral_flux_posterior_precision_scale,
            config.projected_integral_flux_target_posterior_rms,
        )?;
        mark_resource_accuracy_unavailable(&mut row);
        let archive = C51ResourceArchive {
            metadata,
            config: config.clone(),
            row,
        };
        archive.validate()?;
        return Ok(archive);
    }
    let mut assembled = assemble_refinement(&benchmark, selection.refinement)?;
    let matching = assembled
        .designs
        .drain(..)
        .filter(|design| design.class == selection.observation_class)
        .collect::<Vec<_>>();
    let [design] = matching.as_slice() else {
        return Err(FeecGmrfError::Inference(format!(
            "C51 resource selection resolved to {} observation designs instead of one",
            matching.len()
        )));
    };
    let parameters = MaternParameters::new(config.alpha, config.kappa, config.tau)?;
    let mut row = match selection.method {
        C51MethodId::FullMetricDirect => {
            let (prior, prior_build) = build_exact_prior(&assembled.operators, parameters)?;
            run_selected_spd_resource(
                C51MethodId::FullMetricDirect,
                &prior,
                prior_build,
                assembled.feec_assembly,
                &benchmark,
                design,
                &assembled.problem,
                selection,
            )?
        }
        C51MethodId::WhitneyCompact
        | C51MethodId::WhitneyCompactCalibrated
        | C51MethodId::WhitneyCompactIntegralFluxCalibrated => {
            unreachable!("handled by isolated compact route")
        }
        C51MethodId::FullMetricAugmented => {
            let (prior, prior_build) = build_supported_prior(&assembled.operators, parameters)?;
            run_selected_supported_resource(
                &prior,
                prior_build,
                assembled.feec_assembly,
                &benchmark,
                design,
                &assembled.problem,
                selection,
            )?
        }
        C51MethodId::ExactObservationSpaceGp => {
            let (prior, prior_build) = build_supported_prior(&assembled.operators, parameters)?;
            run_exact_observation_gp_method(
                &prior,
                prior_build,
                assembled.feec_assembly,
                &benchmark,
                selection.refinement,
                design,
                selection.repeated_data_count,
                &assembled.problem,
                selection.phase,
                None,
            )?
        }
        C51MethodId::SpectralFullRank => {
            let spectral = build_spectral_reference(&benchmark, &assembled.operators)?;
            run_selected_spectral_resource(
                selection.method,
                spectral.basis.len(),
                &spectral,
                assembled.feec_assembly,
                &benchmark,
                design,
                &assembled.problem,
                selection,
            )?
        }
        C51MethodId::SpectralLowRank => {
            #[cfg(feature = "external-solvers")]
            let rank = selection
                .spectral_rank
                .expect("validated low-rank resource selection");
            #[cfg(feature = "external-solvers")]
            let spectral = build_exact_requested_mixed_spectral_reference(
                &benchmark,
                &assembled.topology,
                &assembled.metric,
                rank,
            )?;
            #[cfg(not(feature = "external-solvers"))]
            return Err(FeecGmrfError::InvalidParameter(
                "a C51 low-rank spectral resource row requires the external-solvers feature; the resource path has no dense or shared-basis fallback"
                    .to_string(),
            ));
            #[cfg(feature = "external-solvers")]
            run_selected_spectral_resource(
                selection.method,
                rank,
                &spectral,
                assembled.feec_assembly,
                &benchmark,
                design,
                &assembled.problem,
                selection,
            )?
        }
    };
    mark_resource_accuracy_unavailable(&mut row);
    let archive = C51ResourceArchive {
        metadata,
        config: config.clone(),
        row,
    };
    archive.validate()?;
    Ok(archive)
}

/// Execute the discovery-only partial spectral-rank pilot.
///
/// This route requires PETSc/SLEPc. It uses the exact supported full-metric
/// posterior as the reference and never substitutes the projected Whitney
/// operator. The shared maximum basis is computed through the sparse mixed
/// Hodge shell without materializing the exact marginal Hodge operator. Its
/// construction timing is archived but is not eligible for method-performance
/// claims.
#[cfg(feature = "external-solvers")]
pub fn run_c51_spectral_rank_pilot(
    config: &C51BenchmarkConfig,
    metadata: C51RunMetadata,
) -> Result<C51BenchmarkArchive> {
    config.validate()?;
    if config.alpha != MaternAlpha::Two {
        return Err(FeecGmrfError::InvalidParameter(
            "the C51 spectral-rank pilot is frozen to alpha=2".to_string(),
        ));
    }
    let mut rows = Vec::new();
    for &refinement in &config.refinements {
        rows.extend(run_partial_spectral_refinement(config, refinement)?);
    }
    let archive = C51BenchmarkArchive {
        metadata,
        config: config.clone(),
        comparison_frames: [
            C51ComparisonFrame::SameDiscreteLawRepresentation,
            C51ComparisonFrame::MatchedPhysicalQoiMethod,
        ],
        rows,
    };
    archive.validate()?;
    Ok(archive)
}

/// Typed failure for callers that select the pilot without external solvers.
#[cfg(not(feature = "external-solvers"))]
pub fn run_c51_spectral_rank_pilot(
    _config: &C51BenchmarkConfig,
    _metadata: C51RunMetadata,
) -> Result<C51BenchmarkArchive> {
    Err(FeecGmrfError::Unsupported(
        "the C51 spectral_rank_pilot profile requires the feg-case-studies external-solvers feature and a working PETSc/SLEPc helper"
            .to_string(),
    ))
}

struct BuiltPriors {
    exact: GaussianPrior,
    compact: GaussianPrior,
    supported: SupportedMaternPrior,
    exact_build: Duration,
    compact_build: Duration,
    supported_build: Duration,
}

struct BuiltScalablePriors {
    compact: GaussianPrior,
    supported: SupportedMaternPrior,
    compact_build: Duration,
    supported_build: Duration,
}

struct ProblemMaps {
    physical_qoi: LinearMap,
    physical_qoi_probe: Vec<f64>,
    physical_qoi_design_id: String,
    latent_probe: Vec<f64>,
    physical_qoi_selected_cell_count: usize,
    physical_qoi_selected_volume_fraction: f64,
}

struct SpectralReference {
    basis: EigenBasis,
    operator_recovery: Duration,
    eigensolve: Duration,
    basis_storage_values: usize,
    source: SpectralBasisSource,
}

struct ExactSpectralOperators {
    mass: FeecCsrMatrix,
    weak_hodge_laplacian: FeecCsrMatrix,
    recovery: Duration,
}

#[derive(Clone)]
enum SpectralBasisSource {
    DenseChecked,
    #[cfg(feature = "external-solvers")]
    PartialSharedMaximumMixed(PartialBasisDiagnostics),
    #[cfg(feature = "external-solvers")]
    PartialExactRequestedMixed(PartialBasisDiagnostics),
}

#[cfg(feature = "external-solvers")]
#[derive(Clone)]
struct PartialBasisDiagnostics {
    requested_rank: usize,
    converged_rank: usize,
    retained_rank: usize,
    which: String,
    mass_solve: String,
    l_nonzeros: usize,
    d_nonzeros: usize,
    c_nonzeros: usize,
    mkm1_nonzeros: usize,
    mk_nonzeros: usize,
    mass_nonzeros: usize,
    input_nonzeros: usize,
    basis_storage_bytes: usize,
    clamped_negative_eigenvalues: usize,
    maximum_relative_mixed_residual: f64,
    maximum_mass_orthogonality_defect: f64,
    mass_orthogonality_tolerance: f64,
    eigenvalue_negative_tolerance: f64,
    eigensolver_tolerance: f64,
    residual_validation: Duration,
    feec_assembly_reused: bool,
    workspace: String,
    binary: String,
}

#[derive(Clone)]
struct ObservationDesign {
    class: C51ObservationClass,
    operator: LinearMap,
    sensor_count: usize,
    design_id: String,
    fixed_radius: C51Measurement<f64>,
    vector_center_count: C51Measurement<usize>,
    component_counts: [C51Measurement<usize>; 3],
}

struct ReferenceAnswers {
    mean: Vec<f64>,
    covariance_action: Vec<f64>,
    physical_covariance_action: Vec<f64>,
    physical_output_covariance_action: Vec<f64>,
    physical_variance: f64,
    physical_transformed_variances: Vec<f64>,
    physical_mean: Vec<f64>,
}

struct AccuracyMeasurements {
    posterior_mean: C51Measurement<f64>,
    covariance_action: C51Measurement<f64>,
    physical_covariance_action: C51Measurement<f64>,
    physical_output_covariance_action: C51Measurement<f64>,
    physical_variance: C51Measurement<f64>,
    physical_transformed_variance: C51Measurement<f64>,
    physical_mean: C51Measurement<f64>,
}

struct AssembledRefinement {
    topology: Complex,
    metric: manifold::geometry::metric::mesh::MeshLengths,
    operators: ExactFormOperators,
    problem: ProblemMaps,
    designs: Vec<ObservationDesign>,
    feec_assembly: Duration,
}

fn run_refinement(config: &C51BenchmarkConfig, refinement: usize) -> Result<Vec<C51BenchmarkRow>> {
    let assembled = assemble_refinement(config, refinement)?;
    let parameters = MaternParameters::new(config.alpha, config.kappa, config.tau)?;
    let priors = build_priors(
        config,
        &assembled.topology,
        &assembled.metric,
        &assembled.operators,
        parameters,
    )?;
    let spectral = build_spectral_reference(config, &assembled.operators)?;
    let mut rows = Vec::new();
    for design in assembled.designs {
        for &repeated_data_count in &config.repeated_data_counts {
            let exact = run_spd_method(
                C51MethodId::FullMetricDirect,
                &priors.exact,
                priors.exact_build,
                assembled.feec_assembly,
                config,
                refinement,
                &design,
                repeated_data_count,
                &assembled.problem,
                None,
            )?;
            let reference = exact.1;
            rows.push(exact.0);
            if design.class == C51ObservationClass::LocalPhysicalB {
                rows.push(run_exact_observation_gp_method(
                    &priors.supported,
                    priors.supported_build,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    C51ResourcePhase::QueryFullUq,
                    Some(&reference),
                )?);
            }
            rows.push(
                run_supported_method(
                    &priors.supported,
                    priors.supported_build,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    Some(&reference),
                )?
                .0,
            );
            rows.push(
                run_spd_method(
                    C51MethodId::WhitneyCompact,
                    &priors.compact,
                    priors.compact_build,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    Some(&reference),
                )?
                .0,
            );
            rows.push(run_spectral_method(
                C51MethodId::SpectralFullRank,
                spectral.basis.len(),
                &spectral,
                assembled.feec_assembly,
                config,
                refinement,
                &design,
                repeated_data_count,
                &assembled.problem,
                Some(&reference),
            )?);
            for rank in spectral_low_ranks(config, spectral.basis.len()) {
                rows.push(run_spectral_method(
                    C51MethodId::SpectralLowRank,
                    rank,
                    &spectral,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    Some(&reference),
                )?);
            }
        }
    }
    Ok(rows)
}

#[cfg(feature = "external-solvers")]
fn run_partial_spectral_refinement(
    config: &C51BenchmarkConfig,
    refinement: usize,
) -> Result<Vec<C51BenchmarkRow>> {
    let assembled = assemble_refinement(config, refinement)?;
    let parameters = MaternParameters::new(config.alpha, config.kappa, config.tau)?;
    let priors = build_scalable_priors(
        config,
        &assembled.topology,
        &assembled.metric,
        &assembled.operators,
        parameters,
    )?;
    let dimension = assembled.operators.state().dimension();
    let (ranks, maximum_rank) = partial_spectral_rank_plan(config, dimension)?;
    let spectral = build_partial_spectral_reference(
        config,
        &assembled.topology,
        &assembled.metric,
        maximum_rank,
    )?;
    let mut rows = Vec::new();
    for design in assembled.designs {
        if design.class != C51ObservationClass::LocalPhysicalB {
            continue;
        }
        for &repeated_data_count in &config.repeated_data_counts {
            let (supported_row, reference) = run_supported_method(
                &priors.supported,
                priors.supported_build,
                assembled.feec_assembly,
                config,
                refinement,
                &design,
                repeated_data_count,
                &assembled.problem,
                None,
            )?;
            rows.push(supported_row);
            rows.push(
                run_spd_method(
                    C51MethodId::WhitneyCompact,
                    &priors.compact,
                    priors.compact_build,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    Some(&reference),
                )?
                .0,
            );
            for &rank in &ranks {
                rows.push(run_spectral_method(
                    C51MethodId::SpectralLowRank,
                    rank,
                    &spectral,
                    assembled.feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &assembled.problem,
                    Some(&reference),
                )?);
            }
        }
    }
    Ok(rows)
}

fn assemble_refinement(
    config: &C51BenchmarkConfig,
    refinement: usize,
) -> Result<AssembledRefinement> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let assembly_start = Instant::now();
    let operators = ExactFormOperators::from_feec_homogeneous_full_relative(
        &topology,
        &metric,
        config.form_degree,
    )?;
    let elimination =
        EssentialBoundaryElimination::from_layout(operators.state().boundary_layout().clone())?;
    let (problem, designs) = assemble_problem_maps(
        config,
        &topology,
        &coordinates,
        &elimination,
        operators.state().dimension(),
    )?;
    let feec_assembly = assembly_start.elapsed();
    Ok(AssembledRefinement {
        topology,
        metric,
        operators,
        problem,
        designs,
        feec_assembly,
    })
}

fn assemble_problem_maps(
    config: &C51BenchmarkConfig,
    topology: &Complex,
    coordinates: &manifold::geometry::coord::mesh::MeshCoords,
    elimination: &EssentialBoundaryElimination,
    state_dimension: usize,
) -> Result<(ProblemMaps, Vec<ObservationDesign>)> {
    let direct_flux = if config.physical_observation_design
        == C51PhysicalObservationDesign::FluxCochainHashedPrefix
        || config.physical_output_design
            == C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit
    {
        let flux_elimination = EssentialBoundaryConditions::homogeneous_on_boundary(topology, 2)?
            .eliminate(topology.nsimplices(2))?;
        let active_flux =
            assemble_c51_active_flux_cochain_map(topology, elimination, &flux_elimination)?;
        Some((active_flux, flux_elimination))
    } else {
        None
    };
    let (physical_qoi, legacy_observation, selected_cell_count, selected_volume_fraction) =
        match config.physical_output_design {
            C51PhysicalOutputDesign::LegacyCentralRegionAverage => {
                let (cells, volume_fraction) = physical_qoi_region_cells(
                    topology,
                    coordinates,
                    config.physical_qoi_region_lower,
                    config.physical_qoi_region_upper,
                )?;
                let maps = MagneticFieldMaps3d::from_feec_with_volume_average_cells(
                    topology,
                    coordinates,
                    &cells,
                )?;
                let components = maps
                    .volume_average_components
                    .iter()
                    .map(|map| {
                        elimination
                            .eliminate_map(map, &[0.0])
                            .map(|eliminated| eliminated.reduced().clone())
                    })
                    .collect::<Result<Vec<_>>>()?;
                (
                    LinearMap::stack(&components)?,
                    Some(maps.magnetic_field.map().clone()),
                    cells.len(),
                    volume_fraction,
                )
            }
            C51PhysicalOutputDesign::RankStudySelectionAndHeldOut => {
                let mut kernels = selection_kernels(C51_FIXED_OBSERVATION_RADIUS)?;
                kernels.extend(held_out_kernels(C51_FIXED_OBSERVATION_RADIUS)?);
                let bank = feec_gmrf::prelude::MagneticVolumeSensorBank3d::from_feec(
                    topology,
                    coordinates,
                    &kernels,
                )?;
                let map = elimination
                    .eliminate_map(
                        bank.map().map(),
                        &vec![0.0; bank.map().map().output_dimension()],
                    )?
                    .reduced()
                    .clone();
                (map, None, 0, 0.0)
            }
            C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit => {
                let (active_flux, flux_elimination) = direct_flux.as_ref().ok_or_else(|| {
                    FeecGmrfError::Inference(
                        "C51 integral surface QoIs are missing the active flux-cochain map"
                            .to_string(),
                    )
                })?;
                let split = assemble_c51_surface_flux_split(
                    topology,
                    coordinates,
                    flux_elimination,
                    active_flux,
                )?;
                (split.stacked()?, None, 0, 0.0)
            }
        };
    let physical_qoi_probe = physical_qoi_probe(physical_qoi.output_dimension(), config.seed);
    let problem = ProblemMaps {
        physical_qoi,
        physical_qoi_probe,
        physical_qoi_design_id: config.physical_output_design.as_str().to_string(),
        latent_probe: deterministic_probe(state_dimension, config.seed),
        physical_qoi_selected_cell_count: selected_cell_count,
        physical_qoi_selected_volume_fraction: selected_volume_fraction,
    };
    let designs = observation_designs(
        config,
        topology,
        coordinates,
        elimination,
        state_dimension,
        legacy_observation.as_ref(),
        direct_flux.as_ref().map(|(map, _)| map),
    )?;
    Ok((problem, designs))
}

fn build_priors(
    config: &C51BenchmarkConfig,
    topology: &manifold::topology::complex::Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    operators: &ExactFormOperators,
    parameters: MaternParameters,
) -> Result<BuiltPriors> {
    let (exact, exact_build) = build_exact_prior(operators, parameters)?;

    let scalable = build_scalable_priors(config, topology, metric, operators, parameters)?;
    let BuiltScalablePriors {
        compact,
        supported,
        compact_build,
        supported_build,
    } = scalable;
    if compact.dimension() != exact.dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "compact dimension {} differs from exact reference dimension {}",
            compact.dimension(),
            exact.dimension()
        )));
    }
    Ok(BuiltPriors {
        exact,
        compact,
        supported,
        exact_build,
        compact_build,
        supported_build,
    })
}

fn build_scalable_priors(
    config: &C51BenchmarkConfig,
    topology: &manifold::topology::complex::Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    operators: &ExactFormOperators,
    parameters: MaternParameters,
) -> Result<BuiltScalablePriors> {
    let (supported, supported_build) = build_supported_prior(operators, parameters)?;
    let (compact, compact_build) = build_compact_prior(config, topology, metric, parameters)?;
    if compact.dimension() != supported.state_dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "compact dimension {} differs from exact supported dimension {}",
            compact.dimension(),
            supported.state_dimension()
        )));
    }
    Ok(BuiltScalablePriors {
        compact,
        supported,
        compact_build,
        supported_build,
    })
}

fn build_exact_prior(
    operators: &ExactFormOperators,
    parameters: MaternParameters,
) -> Result<(GaussianPrior, Duration)> {
    let start = Instant::now();
    let prior = ExactMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .maximum_materialized_dimension(EXACT_REFERENCE_MAXIMUM_DIMENSION)?
        .build()?;
    Ok((prior, start.elapsed()))
}

fn build_supported_prior(
    operators: &ExactFormOperators,
    parameters: MaternParameters,
) -> Result<(SupportedMaternPrior, Duration)> {
    let start = Instant::now();
    let prior = SupportedMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .build()?;
    Ok((prior, start.elapsed()))
}

fn build_compact_prior(
    config: &C51BenchmarkConfig,
    topology: &manifold::topology::complex::Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    parameters: MaternParameters,
) -> Result<(GaussianPrior, Duration)> {
    let start = Instant::now();
    let prior = if config.physical_observation_design
        == C51PhysicalObservationDesign::FluxCochainHashedPrefix
    {
        AdjacentMaternPriorBuilder::from_feec_homogeneous_full_relative_with_projected_inverses(
            topology,
            metric,
            config.form_degree,
        )?
        .parameters(parameters)
        .build()?
    } else {
        MaternPriorBuilder::from_feec_whitney_projected(topology, metric, config.form_degree)?
            .parameters(parameters)
            .essential_boundary_conditions(EssentialBoundaryConditions::homogeneous_on_boundary(
                topology,
                config.form_degree,
            )?)
            .build()?
    };
    Ok((prior, start.elapsed()))
}

/// Recover the exact weak Hodge--Laplacian from the canonical alpha-one
/// system, then solve the checked generalized eigenproblem on that identical
/// active space. This is intentionally bounded dense reference work.
fn build_spectral_reference(
    config: &C51BenchmarkConfig,
    operators: &ExactFormOperators,
) -> Result<SpectralReference> {
    let dimension = operators.state().dimension();
    if dimension > config.spectral_reference_maximum_dimension {
        return Err(FeecGmrfError::ExactReferenceSizeExceeded {
            dimension,
            maximum: config.spectral_reference_maximum_dimension,
        });
    }
    let exact_operators = recover_exact_spectral_operators(
        config,
        operators,
        config.spectral_reference_maximum_dimension,
    )?;
    let eigensolve_start = Instant::now();
    let basis = dense_reference_generalized_hodge_eigenbasis(
        &exact_operators.mass,
        &exact_operators.weak_hodge_laplacian,
        DenseReferenceEigensolverOptions::new(config.spectral_reference_maximum_dimension),
    )
    .map_err(spectral_error)?;
    let eigensolve = eigensolve_start.elapsed();
    if basis.nrows() != dimension || basis.len() != dimension {
        return Err(FeecGmrfError::Dimension(format!(
            "C51 full spectral basis is {}x{} for active dimension {dimension}",
            basis.nrows(),
            basis.len()
        )));
    }
    Ok(SpectralReference {
        basis,
        operator_recovery: exact_operators.recovery,
        eigensolve,
        basis_storage_values: dimension * dimension + dimension,
        source: SpectralBasisSource::DenseChecked,
    })
}

fn recover_exact_spectral_operators(
    config: &C51BenchmarkConfig,
    operators: &ExactFormOperators,
    maximum_dimension: usize,
) -> Result<ExactSpectralOperators> {
    let dimension = operators.state().dimension();
    if dimension > maximum_dimension {
        return Err(FeecGmrfError::ExactReferenceSizeExceeded {
            dimension,
            maximum: maximum_dimension,
        });
    }
    let recovery_start = Instant::now();
    let alpha_one = ExactMaternPriorBuilder::from_operators(operators.clone())
        .parameters(MaternParameters::new(MaternAlpha::One, config.kappa, 1.0)?)
        .maximum_materialized_dimension(maximum_dimension)?
        .build()?;
    let mut weak_hodge_laplacian = SparseMat::new(dimension, dimension);
    for (row, col, value) in alpha_one.precision().triplet_iter() {
        weak_hodge_laplacian.push(row, col, value);
    }
    for (row, col, value) in operators.state().mass().triplet_iter() {
        weak_hodge_laplacian.push(row, col, -config.kappa.powi(2) * value);
    }
    Ok(ExactSpectralOperators {
        mass: feg_infer::sparse::core_triplet_to_feec_csr(operators.state().mass()),
        weak_hodge_laplacian: feg_infer::sparse::core_triplet_to_feec_csr(&weak_hodge_laplacian),
        recovery: recovery_start.elapsed(),
    })
}

#[cfg(feature = "external-solvers")]
fn build_partial_spectral_reference(
    config: &C51BenchmarkConfig,
    topology: &Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    maximum_rank: usize,
) -> Result<SpectralReference> {
    let eigensolve_start = Instant::now();
    let partial = partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
        topology,
        metric,
        config.form_degree,
        PartialMixedGeneralizedEigensolverOptions::new(maximum_rank),
    )
    .map_err(spectral_error)?;
    let eigensolve = eigensolve_start.elapsed();
    let diagnostics = partial.diagnostics();
    let archived = PartialBasisDiagnostics {
        requested_rank: diagnostics.requested_rank,
        converged_rank: diagnostics.converged_rank,
        retained_rank: diagnostics.retained_rank,
        which: format!("{:?}", diagnostics.which).to_ascii_lowercase(),
        mass_solve: format!("{:?}", diagnostics.mass_solve).to_ascii_lowercase(),
        l_nonzeros: diagnostics.l_nonzeros,
        d_nonzeros: diagnostics.d_nonzeros,
        c_nonzeros: diagnostics.c_nonzeros,
        mkm1_nonzeros: diagnostics.mkm1_nonzeros,
        mk_nonzeros: diagnostics.mk_nonzeros,
        mass_nonzeros: diagnostics.mk_nonzeros,
        input_nonzeros: diagnostics.input_nonzeros,
        basis_storage_bytes: diagnostics.basis_storage_bytes,
        clamped_negative_eigenvalues: diagnostics.clamped_negative_eigenvalues,
        maximum_relative_mixed_residual: diagnostics.maximum_relative_mixed_residual,
        maximum_mass_orthogonality_defect: diagnostics.maximum_mass_orthogonality_defect,
        mass_orthogonality_tolerance: diagnostics.mass_orthogonality_tolerance,
        eigenvalue_negative_tolerance: diagnostics.eigenvalue_negative_tolerance,
        eigensolver_tolerance: diagnostics.eigensolver_tolerance,
        residual_validation: diagnostics.residual_validation_duration,
        feec_assembly_reused: false,
        workspace: diagnostics.workspace.display().to_string(),
        binary: diagnostics.binary.display().to_string(),
    };
    let basis_storage_values = diagnostics
        .basis_storage_entries
        .checked_add(diagnostics.retained_rank)
        .ok_or_else(|| {
            FeecGmrfError::Dimension("C51 spectral basis storage count overflow".to_string())
        })?;
    Ok(SpectralReference {
        basis: partial.into_basis(),
        operator_recovery: Duration::ZERO,
        eigensolve,
        basis_storage_values,
        source: SpectralBasisSource::PartialSharedMaximumMixed(archived),
    })
}

#[cfg(feature = "external-solvers")]
fn build_exact_requested_mixed_spectral_reference(
    config: &C51BenchmarkConfig,
    topology: &Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    requested_rank: usize,
) -> Result<SpectralReference> {
    let eigensolve_start = Instant::now();
    let partial =
        exact_requested_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
            topology,
            metric,
            config.form_degree,
            PartialMixedGeneralizedEigensolverOptions::new(requested_rank),
        )
        .map_err(spectral_error)?;
    let eigensolve = eigensolve_start.elapsed();
    let diagnostics = partial.diagnostics();
    let archived = PartialBasisDiagnostics {
        requested_rank: diagnostics.requested_rank,
        converged_rank: diagnostics.converged_rank,
        retained_rank: diagnostics.retained_rank,
        which: format!("{:?}", diagnostics.which).to_ascii_lowercase(),
        mass_solve: format!("{:?}", diagnostics.mass_solve).to_ascii_lowercase(),
        l_nonzeros: diagnostics.l_nonzeros,
        d_nonzeros: diagnostics.d_nonzeros,
        c_nonzeros: diagnostics.c_nonzeros,
        mkm1_nonzeros: diagnostics.mkm1_nonzeros,
        mk_nonzeros: diagnostics.mk_nonzeros,
        mass_nonzeros: diagnostics.mk_nonzeros,
        input_nonzeros: diagnostics.input_nonzeros,
        basis_storage_bytes: diagnostics.basis_storage_bytes,
        clamped_negative_eigenvalues: diagnostics.clamped_negative_eigenvalues,
        maximum_relative_mixed_residual: diagnostics.maximum_relative_mixed_residual,
        maximum_mass_orthogonality_defect: diagnostics.maximum_mass_orthogonality_defect,
        mass_orthogonality_tolerance: diagnostics.mass_orthogonality_tolerance,
        eigenvalue_negative_tolerance: diagnostics.eigenvalue_negative_tolerance,
        eigensolver_tolerance: diagnostics.eigensolver_tolerance,
        residual_validation: diagnostics.residual_validation_duration,
        // The convenience adapter currently reassembles full-relative layouts
        // and mixed Galerkin blocks. The elapsed eigensolve adapter time and
        // this flag make that duplicate work visible rather than subtracting it.
        feec_assembly_reused: false,
        workspace: diagnostics.workspace.display().to_string(),
        binary: diagnostics.binary.display().to_string(),
    };
    let basis_storage_values = diagnostics
        .basis_storage_entries
        .checked_add(diagnostics.retained_rank)
        .ok_or_else(|| {
            FeecGmrfError::Dimension("C51 spectral basis storage count overflow".to_string())
        })?;
    Ok(SpectralReference {
        basis: partial.into_basis(),
        operator_recovery: Duration::ZERO,
        eigensolve,
        basis_storage_values,
        source: SpectralBasisSource::PartialExactRequestedMixed(archived),
    })
}

fn apply_spectral_source_metadata(
    row: &mut C51BenchmarkRow,
    source: &SpectralReference,
    selected_rank: usize,
) {
    match &source.source {
        SpectralBasisSource::DenseChecked => {
            row.method_performance_timing_eligible = source.basis.len() == selected_rank;
            row.method_performance_timing_ineligibility_reason =
                if row.method_performance_timing_eligible {
                    String::new()
                } else {
                    "dense checked basis rank does not match the selected rank".to_string()
                };
            row.spectral_basis_source = "dense_full_reference".to_string();
            row.spectral_eigensolver_backend = "dense_checked_generalized_eigensolver".to_string();
            row.spectral_feec_assembly_reuse = "recovered_from_exact_form_operators".to_string();
        }
        #[cfg(feature = "external-solvers")]
        SpectralBasisSource::PartialSharedMaximumMixed(partial) => {
            row.method_performance_timing_eligible = false;
            row.method_performance_timing_ineligibility_reason =
                "one shared maximum-rank discovery basis cannot support method-owned timing"
                    .to_string();
            apply_partial_spectral_diagnostics(row, partial, "partial_shared_maximum_mixed_shell");
            row.spectral_shared_maximum_basis_rank =
                C51Measurement::Measured(partial.retained_rank);
        }
        #[cfg(feature = "external-solvers")]
        SpectralBasisSource::PartialExactRequestedMixed(partial) => {
            apply_partial_spectral_diagnostics(row, partial, "partial_exact_requested_mixed_shell");
            let failure = exact_requested_mixed_timing_ineligibility_reason(
                partial,
                selected_rank,
                source.basis.len(),
            );
            row.method_performance_timing_eligible = failure.is_none();
            row.method_performance_timing_ineligibility_reason = failure.unwrap_or_default();
        }
    }
}

#[cfg(feature = "external-solvers")]
fn apply_partial_spectral_diagnostics(
    row: &mut C51BenchmarkRow,
    partial: &PartialBasisDiagnostics,
    source_label: &str,
) {
    row.spectral_basis_source = source_label.to_string();
    row.spectral_eigensolver_backend =
        "petsc_slepc_mixed_reduced_generalized_eigensolver".to_string();
    row.spectral_helper_workspace = partial.workspace.clone();
    row.spectral_helper_binary = partial.binary.clone();
    row.spectral_feec_assembly_reuse = if partial.feec_assembly_reused {
        "reused".to_string()
    } else {
        "not_reused_reassembled_full_relative_layouts_and_mixed_blocks_charged_in_eigensolve"
            .to_string()
    };
    row.spectral_partial_which = partial.which.clone();
    row.spectral_partial_mass_solve = partial.mass_solve.clone();
    row.spectral_partial_requested_rank = C51Measurement::Measured(partial.requested_rank);
    row.spectral_partial_converged_rank = C51Measurement::Measured(partial.converged_rank);
    row.spectral_partial_retained_rank = C51Measurement::Measured(partial.retained_rank);
    row.spectral_partial_l_nnz = C51Measurement::Measured(partial.l_nonzeros);
    row.spectral_partial_d_nnz = C51Measurement::Measured(partial.d_nonzeros);
    row.spectral_partial_c_nnz = C51Measurement::Measured(partial.c_nonzeros);
    row.spectral_partial_mkm1_nnz = C51Measurement::Measured(partial.mkm1_nonzeros);
    row.spectral_partial_mk_nnz = C51Measurement::Measured(partial.mk_nonzeros);
    row.spectral_partial_mass_nnz = C51Measurement::Measured(partial.mass_nonzeros);
    row.spectral_partial_weak_hodge_laplacian_nnz =
        C51Measurement::unavailable(MIXED_MARGINAL_NNZ_UNAVAILABLE_REASON);
    row.spectral_partial_input_nnz = C51Measurement::Measured(partial.input_nonzeros);
    row.spectral_partial_basis_storage_bytes =
        C51Measurement::Measured(partial.basis_storage_bytes);
    row.spectral_partial_clamped_negative_eigenvalues =
        C51Measurement::Measured(partial.clamped_negative_eigenvalues);
    row.spectral_partial_maximum_relative_mixed_residual =
        C51Measurement::Measured(partial.maximum_relative_mixed_residual);
    row.spectral_partial_maximum_mass_orthogonality_defect =
        C51Measurement::Measured(partial.maximum_mass_orthogonality_defect);
    row.spectral_partial_mass_orthogonality_tolerance =
        C51Measurement::Measured(partial.mass_orthogonality_tolerance);
    row.spectral_partial_eigenvalue_negative_tolerance =
        C51Measurement::Measured(partial.eigenvalue_negative_tolerance);
    row.spectral_partial_eigensolver_tolerance =
        C51Measurement::Measured(partial.eigensolver_tolerance);
    row.spectral_partial_residual_validation_ns = duration_measurement(partial.residual_validation);
}

#[cfg(feature = "external-solvers")]
fn exact_requested_mixed_timing_ineligibility_reason(
    partial: &PartialBasisDiagnostics,
    selected_rank: usize,
    basis_rank: usize,
) -> Option<String> {
    if partial.requested_rank != selected_rank
        || partial.retained_rank != selected_rank
        || basis_rank != selected_rank
        || partial.converged_rank < selected_rank
    {
        return Some(format!(
            "rank invariant failed: selected={selected_rank}, requested={}, converged={}, retained={}, basis={basis_rank}",
            partial.requested_rank, partial.converged_rank, partial.retained_rank
        ));
    }
    if partial.which != "smallest" {
        return Some(format!(
            "eigensolver requested {:?} modes instead of smallest",
            partial.which
        ));
    }
    if partial.mass_solve != "direct" {
        return Some(format!(
            "lower-mass solve policy {:?} is not the frozen direct policy",
            partial.mass_solve
        ));
    }
    if partial.eigensolver_tolerance != EXACT_REQUESTED_MIXED_EIGENSOLVER_TOLERANCE {
        return Some(format!(
            "eigensolver tolerance {} does not match the frozen tolerance {}",
            partial.eigensolver_tolerance, EXACT_REQUESTED_MIXED_EIGENSOLVER_TOLERANCE
        ));
    }
    if partial.mass_orthogonality_tolerance != EXACT_REQUESTED_MIXED_MASS_ORTHOGONALITY_TOLERANCE {
        return Some(format!(
            "mass-orthogonality tolerance {} does not match the frozen tolerance {}",
            partial.mass_orthogonality_tolerance,
            EXACT_REQUESTED_MIXED_MASS_ORTHOGONALITY_TOLERANCE
        ));
    }
    if partial.eigenvalue_negative_tolerance != EXACT_REQUESTED_MIXED_EIGENVALUE_NEGATIVE_TOLERANCE
    {
        return Some(format!(
            "negative-eigenvalue tolerance {} does not match the frozen tolerance {}",
            partial.eigenvalue_negative_tolerance,
            EXACT_REQUESTED_MIXED_EIGENVALUE_NEGATIVE_TOLERANCE
        ));
    }
    if !partial.maximum_relative_mixed_residual.is_finite()
        || partial.maximum_relative_mixed_residual > EXACT_REQUESTED_MIXED_RESIDUAL_TOLERANCE
    {
        return Some(format!(
            "maximum mixed residual {} exceeds the frozen tolerance {}",
            partial.maximum_relative_mixed_residual, EXACT_REQUESTED_MIXED_RESIDUAL_TOLERANCE
        ));
    }
    if !partial.maximum_mass_orthogonality_defect.is_finite()
        || partial.maximum_mass_orthogonality_defect > partial.mass_orthogonality_tolerance
    {
        return Some(format!(
            "mass-orthogonality defect {} exceeds tolerance {}",
            partial.maximum_mass_orthogonality_defect, partial.mass_orthogonality_tolerance
        ));
    }
    None
}

#[cfg(feature = "external-solvers")]
fn partial_spectral_rank_plan(
    config: &C51BenchmarkConfig,
    dimension: usize,
) -> Result<(Vec<usize>, usize)> {
    let ranks = spectral_low_ranks(config, dimension);
    let maximum_rank = ranks.last().copied().ok_or_else(|| {
        FeecGmrfError::InvalidParameter(format!(
            "the C51 spectral-rank grid has no rank below active dimension {dimension}"
        ))
    })?;
    Ok((ranks, maximum_rank))
}

fn spectral_low_ranks(config: &C51BenchmarkConfig, full_rank: usize) -> Vec<usize> {
    let mut ranks = config
        .spectral_low_ranks
        .iter()
        .copied()
        .filter(|rank| *rank < full_rank)
        .collect::<BTreeSet<_>>();
    for fraction in &config.spectral_rank_fractions {
        let rank = (*fraction * full_rank as f64).ceil() as usize;
        if rank > 0 && rank < full_rank {
            ranks.insert(rank);
        }
    }
    ranks.into_iter().collect()
}

fn run_isolated_compact_resource(
    config: &C51BenchmarkConfig,
    selection: &C51ResourceSelection,
    calibration_target_prior_rms: Option<f64>,
    integral_flux_posterior_precision_scale: Option<f64>,
    integral_flux_target_posterior_rms: Option<f64>,
) -> Result<C51BenchmarkRow> {
    let mesh_start = Instant::now();
    let mesh = CartesianMeshInfo::new_unit_scaled(3, selection.refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let mesh_assembly = mesh_start.elapsed();
    let parameters = MaternParameters::new(config.alpha, config.kappa, config.tau)?;
    let (raw_prior, raw_prior_build) = build_compact_prior(config, &topology, &metric, parameters)?;
    let elimination = raw_prior.boundary_elimination().ok_or_else(|| {
        FeecGmrfError::Inference(
            "C51 compact resource prior is missing its boundary elimination".to_string(),
        )
    })?;
    let maps_start = Instant::now();
    let (problem, mut designs) = assemble_problem_maps(
        config,
        &topology,
        &coordinates,
        elimination,
        raw_prior.dimension(),
    )?;
    let feec_assembly = mesh_assembly + maps_start.elapsed();
    let matching = designs
        .drain(..)
        .filter(|design| design.class == selection.observation_class)
        .collect::<Vec<_>>();
    let [design] = matching.as_slice() else {
        return Err(FeecGmrfError::Inference(format!(
            "C51 compact resource selection resolved to {} observation designs instead of one",
            matching.len()
        )));
    };
    let (prior, prior_build, calibration, frozen_integral_flux_calibration) = match selection.method
    {
        C51MethodId::WhitneyCompact => (raw_prior, raw_prior_build, None, None),
        C51MethodId::WhitneyCompactCalibrated => {
            let target_rms = calibration_target_prior_rms.ok_or_else(|| {
                FeecGmrfError::InvalidParameter(
                    "C51 calibrated projected prior is missing its frozen target RMS".to_string(),
                )
            })?;
            if problem.physical_qoi.output_dimension() != 2 * C51_CALIBRATION_SELECTION_OUTPUT_COUNT
            {
                return Err(FeecGmrfError::Dimension(format!(
                    "C51 calibrated output bank has {} rows instead of {}",
                    problem.physical_qoi.output_dimension(),
                    2 * C51_CALIBRATION_SELECTION_OUTPUT_COUNT
                )));
            }
            let calibration_map = problem
                .physical_qoi
                .select_outputs(&(0..C51_CALIBRATION_SELECTION_OUTPUT_COUNT).collect::<Vec<_>>())?;
            let start = Instant::now();
            let (prior, report) =
                calibrate_prior_to_physical_rms(&raw_prior, &calibration_map, target_rms)?;
            let elapsed = start.elapsed();
            (
                prior,
                raw_prior_build + elapsed,
                Some((report, elapsed)),
                None,
            )
        }
        C51MethodId::WhitneyCompactIntegralFluxCalibrated => {
            let scale = integral_flux_posterior_precision_scale.ok_or_else(|| {
                FeecGmrfError::InvalidParameter(
                    "C51 calibrated integral-flux prior is missing its frozen precision scale"
                        .to_string(),
                )
            })?;
            let target = integral_flux_target_posterior_rms.ok_or_else(|| {
                FeecGmrfError::InvalidParameter(
                    "C51 calibrated integral-flux prior is missing its frozen posterior RMS target"
                        .to_string(),
                )
            })?;
            let start = Instant::now();
            let prior = raw_prior.with_precision_scale(scale)?;
            let elapsed = start.elapsed();
            (
                prior,
                raw_prior_build + elapsed,
                None,
                Some((scale, target)),
            )
        }
        _ => unreachable!("isolated compact route accepts only projected methods"),
    };
    let mut row = run_selected_spd_resource(
        selection.method,
        &prior,
        prior_build,
        feec_assembly,
        config,
        design,
        &problem,
        selection,
    )?;
    if let Some((report, elapsed)) = calibration {
        row.amplitude_calibration_policy =
            "exact_feec_prior_selection_bank_physical_rms_before_conditioning".to_string();
        row.amplitude_calibration_target_prior_rms = C51Measurement::Measured(report.target_rms);
        row.amplitude_calibration_uncalibrated_prior_rms =
            C51Measurement::Measured(report.uncalibrated_rms);
        row.amplitude_calibration_precision_scale =
            C51Measurement::Measured(report.precision_scale);
        row.amplitude_calibration_tau_multiplier =
            C51Measurement::Measured(report.tau_multiplier());
        row.amplitude_calibration_ns = duration_measurement(elapsed);
    }
    if let Some((scale, _target)) = frozen_integral_flux_calibration {
        row.amplitude_calibration_policy =
            "frozen_exact_reference_posterior_surface_flux_rms_by_prior_precision".to_string();
        row.amplitude_calibration_precision_scale = C51Measurement::Measured(scale);
        row.amplitude_calibration_tau_multiplier = C51Measurement::Measured(scale.sqrt());
        row.amplitude_calibration_ns = C51Measurement::unavailable(
            "offline selection-bank calibration search is reported by the accuracy archive and excluded from timed posterior construction",
        );
    }
    Ok(row)
}

#[allow(clippy::too_many_arguments)]
fn base_resource_row(
    method: C51MethodId,
    config: &C51BenchmarkConfig,
    selection: &C51ResourceSelection,
    design: &ObservationDesign,
    problem: &ProblemMaps,
    latent_dimension: usize,
    ambient_dimension: usize,
    constraint_dimension: usize,
) -> C51BenchmarkRow {
    let accuracy_unavailable = || C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
    C51BenchmarkRow {
        method,
        available: true,
        unavailable_reason: String::new(),
        discrete_law_id: match method {
            C51MethodId::FullMetricDirect
            | C51MethodId::FullMetricAugmented
            | C51MethodId::ExactObservationSpaceGp
            | C51MethodId::SpectralFullRank => "exact_full_metric_whitney",
            C51MethodId::WhitneyCompact => "projected_inverse_whitney",
            C51MethodId::WhitneyCompactCalibrated => {
                "projected_inverse_whitney_physical_rms_calibrated"
            }
            C51MethodId::WhitneyCompactIntegralFluxCalibrated => {
                "projected_inverse_whitney_integral_flux_posterior_rms_calibrated"
            }
            C51MethodId::SpectralLowRank => "truncated_exact_full_metric_whitney_spectral",
        }
        .to_string(),
        same_discrete_law_eligible: matches!(
            method,
            C51MethodId::FullMetricDirect
                | C51MethodId::FullMetricAugmented
                | C51MethodId::ExactObservationSpaceGp
                | C51MethodId::SpectralFullRank
        ),
        matched_physical_qoi_eligible: matches!(
            method,
            C51MethodId::WhitneyCompactCalibrated
                | C51MethodId::WhitneyCompactIntegralFluxCalibrated
        ),
        common_physical_uq_workload_eligible: matches!(
            selection.phase,
            C51ResourcePhase::ColdSetup | C51ResourcePhase::QueryPhysicalUq
        ) && matches!(
            (
                config.physical_observation_design,
                config.physical_output_design,
            ),
            (
                C51PhysicalObservationDesign::JuqFixedFootprintPrefix,
                C51PhysicalOutputDesign::RankStudySelectionAndHeldOut,
            ) | (
                C51PhysicalObservationDesign::FluxCochainHashedPrefix,
                C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit,
            )
        ),
        full_uq_workload_eligible: selection.phase == C51ResourcePhase::QueryFullUq
            && !matches!(
                method,
                C51MethodId::FullMetricAugmented | C51MethodId::ExactObservationSpaceGp
            ),
        method_performance_timing_eligible: true,
        method_performance_timing_ineligibility_reason: String::new(),
        spectral_basis_source: "not_applicable".to_string(),
        spectral_eigensolver_backend: "not_applicable".to_string(),
        spectral_helper_workspace: String::new(),
        spectral_helper_binary: String::new(),
        spectral_feec_assembly_reuse: "not_applicable".to_string(),
        spectral_partial_which: "not_applicable".to_string(),
        spectral_partial_mass_solve: "not_applicable".to_string(),
        refinement: selection.refinement,
        spatial_dimension: config.spatial_dimension,
        form_degree: config.form_degree,
        alpha: config.alpha.as_u32(),
        kappa: config.kappa,
        tau: config.tau,
        amplitude_calibration_policy: "none".to_string(),
        amplitude_calibration_target_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_uncalibrated_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_precision_scale: C51Measurement::NotRecorded,
        amplitude_calibration_tau_multiplier: C51Measurement::NotRecorded,
        amplitude_calibration_ns: C51Measurement::NotRecorded,
        observation_class: design.class,
        observation_design_id: design.design_id.clone(),
        observation_fixed_radius: design.fixed_radius.clone(),
        observation_vector_center_count: design.vector_center_count.clone(),
        observation_component_x_count: design.component_counts[0].clone(),
        observation_component_y_count: design.component_counts[1].clone(),
        observation_component_z_count: design.component_counts[2].clone(),
        physical_qoi_design_id: problem.physical_qoi_design_id.clone(),
        physical_qoi_output_count: problem.physical_qoi.output_dimension(),
        physical_qoi_selected_cell_count: problem.physical_qoi_selected_cell_count,
        physical_qoi_selected_volume_fraction: problem.physical_qoi_selected_volume_fraction,
        sensor_count: design.sensor_count,
        sensor_density: design.sensor_count as f64 / latent_dimension as f64,
        repeated_data_count: selection.repeated_data_count,
        posterior_sample_count: config.posterior_sample_count,
        seed: config.seed,
        latent_dimension,
        ambient_dimension,
        constraint_dimension,
        observation_nnz: design.operator.matrix().nnz(),
        prior_precision_nnz: resource_phase_not_executed(),
        observation_update_nnz: resource_phase_not_executed(),
        posterior_precision_nnz: resource_phase_not_executed(),
        ambient_energy_nnz: resource_phase_not_executed(),
        constraint_nnz: resource_phase_not_executed(),
        kkt_input_nnz: resource_phase_not_executed(),
        factor_storage_nnz: resource_phase_not_executed(),
        certificate_factor_storage_nnz: resource_phase_not_executed(),
        fill_ratio: resource_phase_not_executed(),
        exact_gp_dense_factor_dimension: C51Measurement::NotRecorded,
        exact_gp_dense_factor_allocated_values: C51Measurement::NotRecorded,
        exact_gp_dense_factor_triangular_values: C51Measurement::NotRecorded,
        exact_gp_output_observation_covariance_values: C51Measurement::NotRecorded,
        exact_gp_posterior_output_covariance_values: C51Measurement::NotRecorded,
        exact_gp_persistent_storage_values: C51Measurement::NotRecorded,
        exact_gp_posterior_covariance_max_abs_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_raw_max_relative_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_projection_frobenius_correction: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_residual_aware: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_policy: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_status: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_residual_status: "not_applicable".to_string(),
        spectral_rank: C51Measurement::NotRecorded,
        spectral_basis_storage_values: C51Measurement::NotRecorded,
        spectral_gp_storage_values: C51Measurement::NotRecorded,
        spectral_prepared_storage_values: C51Measurement::NotRecorded,
        spectral_dense_factor_values: C51Measurement::NotRecorded,
        spectral_state_feature_values: C51Measurement::NotRecorded,
        spectral_observation_feature_values: C51Measurement::NotRecorded,
        spectral_sampling_normal_dimension: C51Measurement::NotRecorded,
        spectral_rank_fraction: C51Measurement::NotRecorded,
        spectral_shared_maximum_basis_rank: C51Measurement::NotRecorded,
        spectral_partial_requested_rank: C51Measurement::NotRecorded,
        spectral_partial_converged_rank: C51Measurement::NotRecorded,
        spectral_partial_retained_rank: C51Measurement::NotRecorded,
        spectral_partial_l_nnz: C51Measurement::NotRecorded,
        spectral_partial_d_nnz: C51Measurement::NotRecorded,
        spectral_partial_c_nnz: C51Measurement::NotRecorded,
        spectral_partial_mkm1_nnz: C51Measurement::NotRecorded,
        spectral_partial_mk_nnz: C51Measurement::NotRecorded,
        spectral_partial_mass_nnz: C51Measurement::NotRecorded,
        spectral_partial_weak_hodge_laplacian_nnz: C51Measurement::NotRecorded,
        spectral_partial_input_nnz: C51Measurement::NotRecorded,
        spectral_partial_basis_storage_bytes: C51Measurement::NotRecorded,
        spectral_partial_clamped_negative_eigenvalues: C51Measurement::NotRecorded,
        spectral_partial_maximum_relative_mixed_residual: C51Measurement::NotRecorded,
        spectral_partial_maximum_mass_orthogonality_defect: C51Measurement::NotRecorded,
        spectral_partial_mass_orthogonality_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigenvalue_negative_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigensolver_tolerance: C51Measurement::NotRecorded,
        spectral_partial_residual_validation_ns: C51Measurement::NotRecorded,
        solver: "not_executed".to_string(),
        ordering: "not_applicable".to_string(),
        feec_assembly_ns: resource_phase_not_executed(),
        prior_build_ns: resource_phase_not_executed(),
        exact_gp_covariance_construction_ns: C51Measurement::NotRecorded,
        exact_gp_dense_conditioning_ns: C51Measurement::NotRecorded,
        exact_gp_end_to_end_setup_ns: C51Measurement::NotRecorded,
        spectral_operator_recovery_ns: C51Measurement::NotRecorded,
        spectral_eigensolve_ns: C51Measurement::NotRecorded,
        spectral_gp_build_ns: C51Measurement::NotRecorded,
        observation_update_assembly_ns: resource_phase_not_executed(),
        observation_update_and_factor_ns: resource_phase_not_executed(),
        symbolic_analysis_ns: resource_phase_not_executed(),
        numeric_factor_ns: resource_phase_not_executed(),
        combined_factor_setup_ns: resource_phase_not_executed(),
        repeated_data_ns: resource_phase_not_executed(),
        covariance_action_ns: resource_phase_not_executed(),
        physical_qoi_ns: resource_phase_not_executed(),
        sampling_ns: resource_phase_not_executed(),
        variance_estimation_ns: resource_phase_not_executed(),
        factor_external_peak_rss_bytes: C51Measurement::NotRecorded,
        query_external_peak_rss_bytes: C51Measurement::NotRecorded,
        posterior_mean_relative_error: accuracy_unavailable(),
        covariance_action_relative_error: accuracy_unavailable(),
        physical_covariance_action_relative_error: accuracy_unavailable(),
        physical_output_covariance_action_relative_error: accuracy_unavailable(),
        physical_variance_relative_error: accuracy_unavailable(),
        physical_transformed_variance_relative_error: accuracy_unavailable(),
        physical_mean_relative_error: accuracy_unavailable(),
        physical_transformed_variance_min_raw: resource_phase_not_executed(),
        physical_transformed_variance_negative_count: resource_phase_not_executed(),
    }
}

fn resource_phase_not_executed<T>() -> C51Measurement<T> {
    C51Measurement::unavailable("not executed in selected resource phase")
}

fn mark_resource_accuracy_unavailable(row: &mut C51BenchmarkRow) {
    row.matched_physical_qoi_eligible = false;
    row.posterior_mean_relative_error =
        C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
    row.covariance_action_relative_error =
        C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
    row.physical_covariance_action_relative_error =
        C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
    row.physical_output_covariance_action_relative_error =
        C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
    row.physical_variance_relative_error =
        C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
    row.physical_transformed_variance_relative_error =
        C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
    row.physical_mean_relative_error =
        C51Measurement::unavailable(RESOURCE_ACCURACY_UNAVAILABLE_REASON);
}

#[allow(clippy::too_many_arguments)]
fn run_selected_spd_resource(
    method: C51MethodId,
    prior: &GaussianPrior,
    prior_build: Duration,
    feec_assembly: Duration,
    config: &C51BenchmarkConfig,
    design: &ObservationDesign,
    problem: &ProblemMaps,
    selection: &C51ResourceSelection,
) -> Result<C51BenchmarkRow> {
    if selection.phase == C51ResourcePhase::QueryFullUq {
        return run_spd_method(
            method,
            prior,
            prior_build,
            feec_assembly,
            config,
            selection.refinement,
            design,
            selection.repeated_data_count,
            problem,
            None,
        )
        .map(|result| result.0);
    }
    let values = repeated_observation_values(design.sensor_count, 1, config.seed);
    let observation = LinearObservation::new(
        design.operator.clone(),
        values[0].clone(),
        GaussianNoise::variance(config.observation_noise_variance)?,
    )?;
    let setup_start = Instant::now();
    let (prepared, resource_report) = LinearGaussianModelBuilder::new(prior.clone())
        .observe(observation)?
        .prepare_with_resource_report()?;
    let setup_elapsed = setup_start.elapsed();
    let factor = prepared.factorization_diagnostics();
    let conditioning = prepared.conditioning_diagnostics();
    let observation_update = conditioning.observation_terms.first().ok_or_else(|| {
        FeecGmrfError::Inference(
            "C51 prepared resource model is missing observation diagnostics".to_string(),
        )
    })?;
    let mut row = base_resource_row(
        method,
        config,
        selection,
        design,
        problem,
        prior.dimension(),
        prior.dimension(),
        0,
    );
    row.observation_nnz = observation_update.operator_nonzeros;
    row.prior_precision_nnz = C51Measurement::Measured(conditioning.prior_precision_nonzeros);
    row.observation_update_nnz =
        C51Measurement::Measured(observation_update.precision_update_nonzeros);
    row.posterior_precision_nnz =
        C51Measurement::Measured(conditioning.posterior_precision_nonzeros);
    row.factor_storage_nnz = usize_resource_measurement(&resource_report.factor_storage_nonzeros);
    row.fill_ratio = C51Measurement::Measured(factor.fill_ratio);
    row.solver = "sparse_cholesky".to_string();
    row.ordering = ordering_name(factor.ordering).to_string();
    row.feec_assembly_ns = duration_measurement(feec_assembly);
    row.prior_build_ns = duration_measurement(prior_build);
    row.observation_update_assembly_ns =
        duration_resource_measurement(&resource_report.assembly.wall_time);
    row.observation_update_and_factor_ns = duration_measurement(setup_elapsed);
    row.symbolic_analysis_ns = duration_resource_measurement(&resource_report.analysis.wall_time);
    row.numeric_factor_ns =
        duration_resource_measurement(&resource_report.numeric_factorization.wall_time);
    row.combined_factor_setup_ns =
        duration_resource_measurement(&resource_report.combined_factor_setup.wall_time);
    if selection.phase == C51ResourcePhase::QueryPhysicalUq {
        let values = repeated_observation_values(
            design.sensor_count,
            selection.repeated_data_count,
            config.seed,
        );
        let repeated_start = Instant::now();
        let mut final_posterior = None;
        for (index, data) in values.iter().enumerate() {
            if index + 1 == values.len() {
                let posterior =
                    prepared.condition_with_observation_values(std::slice::from_ref(data))?;
                std::hint::black_box(posterior.pushforward_mean(&problem.physical_qoi)?);
                final_posterior = Some(posterior);
            } else {
                let mean =
                    prepared.latent_mean_with_observation_values(std::slice::from_ref(data))?;
                std::hint::black_box(problem.physical_qoi.apply(&mean)?);
            }
        }
        row.repeated_data_ns = duration_measurement(repeated_start.elapsed());
        let mut posterior =
            final_posterior.expect("positive replacement-data count produces a posterior");
        let physical_start = Instant::now();
        let covariance = posterior.pushforward_covariance(&problem.physical_qoi)?;
        let covariance_action = dense_rows_times_vector(
            &covariance,
            &problem.physical_qoi_probe,
            "C51 compact physical covariance",
        )?;
        std::hint::black_box(&covariance_action);
        std::hint::black_box(dot(&problem.physical_qoi_probe, &covariance_action)?);
        row.physical_qoi_ns = duration_measurement(physical_start.elapsed());
        let variance_start = Instant::now();
        let variances = dense_rows_diagonal(&covariance, "C51 compact physical covariance")?;
        row.variance_estimation_ns = duration_measurement(variance_start.elapsed());
        row.covariance_action_ns = C51Measurement::unavailable(
            "the common physical-UQ workload excludes latent covariance actions",
        );
        row.sampling_ns = C51Measurement::unavailable(
            "the common physical-UQ workload excludes latent state sampling",
        );
        row.physical_transformed_variance_min_raw =
            C51Measurement::Measured(minimum_raw(&variances));
        row.physical_transformed_variance_negative_count =
            C51Measurement::Measured(negative_count(&variances));
    }
    Ok(row)
}

#[allow(clippy::too_many_arguments)]
fn run_selected_supported_resource(
    prior: &SupportedMaternPrior,
    prior_build: Duration,
    feec_assembly: Duration,
    config: &C51BenchmarkConfig,
    design: &ObservationDesign,
    problem: &ProblemMaps,
    selection: &C51ResourceSelection,
) -> Result<C51BenchmarkRow> {
    if selection.phase == C51ResourcePhase::QueryFullUq {
        return run_supported_method(
            prior,
            prior_build,
            feec_assembly,
            config,
            selection.refinement,
            design,
            selection.repeated_data_count,
            problem,
            None,
        )
        .map(|result| result.0);
    }
    let values = repeated_observation_values(design.sensor_count, 1, config.seed);
    let observation = LinearObservation::new(
        design.operator.clone(),
        values[0].clone(),
        GaussianNoise::variance(config.observation_noise_variance)?,
    )?;
    let setup_start = Instant::now();
    let prepared = prior.prepare(&[observation], &[])?;
    let setup_elapsed = setup_start.elapsed();
    let diagnostics = prepared.diagnostics();
    let mut row = base_resource_row(
        C51MethodId::FullMetricAugmented,
        config,
        selection,
        design,
        problem,
        prior.state_dimension(),
        diagnostics.ambient_dimension,
        diagnostics.constraint_dimension,
    );
    row.prior_precision_nnz = C51Measurement::unavailable(
        "FM-AUG is a constrained energy, not an eliminated SPD precision",
    );
    row.observation_update_nnz = C51Measurement::unavailable(UPDATE_NNZ_PENDING_REASON);
    row.posterior_precision_nnz = C51Measurement::unavailable(
        "FM-AUG is a constrained energy, not an eliminated SPD precision",
    );
    row.ambient_energy_nnz = C51Measurement::Measured(diagnostics.ambient_energy_nnz);
    row.constraint_nnz = C51Measurement::Measured(diagnostics.constraint_nnz);
    row.kkt_input_nnz = C51Measurement::Measured(diagnostics.kkt_nnz);
    row.factor_storage_nnz = C51Measurement::unavailable(KKT_FILL_UNAVAILABLE_REASON);
    row.certificate_factor_storage_nnz = C51Measurement::Measured(
        diagnostics.constraint_gram_factor_nnz + diagnostics.supported_energy_factor_nnz,
    );
    row.fill_ratio = C51Measurement::unavailable(KKT_FILL_UNAVAILABLE_REASON);
    row.solver = "sparse_lu_kkt".to_string();
    row.ordering = "backend_partial_pivoting".to_string();
    row.feec_assembly_ns = duration_measurement(feec_assembly);
    row.prior_build_ns = duration_measurement(prior_build);
    row.observation_update_assembly_ns = C51Measurement::unavailable(
        "the supported-law API currently exposes combined observation assembly and KKT setup",
    );
    row.observation_update_and_factor_ns = duration_measurement(setup_elapsed);
    row.symbolic_analysis_ns = C51Measurement::unavailable(
        "the sparse LU backend does not expose separate symbolic analysis",
    );
    row.numeric_factor_ns = C51Measurement::unavailable(
        "the sparse LU backend does not expose separate numeric factorization",
    );
    row.combined_factor_setup_ns = duration_measurement(setup_elapsed);
    if selection.phase == C51ResourcePhase::QueryPhysicalUq {
        let values = repeated_observation_values(
            design.sensor_count,
            selection.repeated_data_count,
            config.seed,
        );
        let repeated_start = Instant::now();
        let mut final_posterior = None;
        for (index, data) in values.iter().enumerate() {
            if index + 1 == values.len() {
                let posterior =
                    prepared.condition_with_observation_values(std::slice::from_ref(data))?;
                std::hint::black_box(problem.physical_qoi.apply(posterior.mean())?);
                final_posterior = Some(posterior);
            } else {
                let mean =
                    prepared.latent_mean_with_observation_values(std::slice::from_ref(data))?;
                std::hint::black_box(problem.physical_qoi.apply(&mean)?);
            }
        }
        row.repeated_data_ns = duration_measurement(repeated_start.elapsed());
        let posterior =
            final_posterior.expect("positive replacement-data count produces a posterior");
        let physical_start = Instant::now();
        let covariance = posterior.pushforward_covariance(&problem.physical_qoi)?;
        let covariance_action = dense_rows_times_vector(
            &covariance,
            &problem.physical_qoi_probe,
            "C51 supported physical covariance",
        )?;
        std::hint::black_box(&covariance_action);
        std::hint::black_box(dot(&problem.physical_qoi_probe, &covariance_action)?);
        row.physical_qoi_ns = duration_measurement(physical_start.elapsed());
        let variance_start = Instant::now();
        let variances = dense_rows_diagonal(&covariance, "C51 supported physical covariance")?;
        row.variance_estimation_ns = duration_measurement(variance_start.elapsed());
        row.covariance_action_ns = C51Measurement::unavailable(
            "the common physical-UQ workload excludes latent covariance actions",
        );
        row.sampling_ns = C51Measurement::unavailable(
            "the common physical-UQ workload excludes latent state sampling",
        );
        row.physical_transformed_variance_min_raw =
            C51Measurement::Measured(minimum_raw(&variances));
        row.physical_transformed_variance_negative_count =
            C51Measurement::Measured(negative_count(&variances));
        row.common_physical_uq_workload_eligible = true;
        row.full_uq_workload_eligible = false;
    }
    Ok(row)
}

#[allow(clippy::too_many_arguments)]
fn run_selected_spectral_resource(
    method: C51MethodId,
    rank: usize,
    source: &SpectralReference,
    feec_assembly: Duration,
    config: &C51BenchmarkConfig,
    design: &ObservationDesign,
    problem: &ProblemMaps,
    selection: &C51ResourceSelection,
) -> Result<C51BenchmarkRow> {
    if selection.phase == C51ResourcePhase::QueryFullUq {
        let row = run_spectral_method(
            method,
            rank,
            source,
            feec_assembly,
            config,
            selection.refinement,
            design,
            selection.repeated_data_count,
            problem,
            None,
        )?;
        return Ok(row);
    }
    let gp_start = Instant::now();
    let gp = SpectralMaternGp::from_eigenbasis(
        source.basis.clone(),
        SpectralMaternConfig {
            kappa: config.kappa,
            alpha: f64::from(config.alpha.as_u32()),
            tau: config.tau,
            k: rank,
        },
    )
    .map_err(spectral_error)?;
    let gp_build = gp_start.elapsed();
    let observation = feg_infer::sparse::core_triplet_to_feec_csr(design.operator.matrix());
    let physical_qoi = feg_infer::sparse::core_triplet_to_feec_csr(problem.physical_qoi.matrix());
    let setup_start = Instant::now();
    let prepared = gp
        .prepare_linear_observations(&observation, config.observation_noise_variance)
        .map_err(spectral_error)?;
    let prepared_output = prepared
        .prepare_output(&physical_qoi)
        .map_err(spectral_error)?;
    let setup_elapsed = setup_start.elapsed();
    let diagnostics = prepared.diagnostics();
    let output_diagnostics = prepared_output.diagnostics();
    let mut row = base_resource_row(
        method,
        config,
        selection,
        design,
        problem,
        diagnostics.latent_dimension,
        diagnostics.ambient_dimension,
        0,
    );
    apply_spectral_source_metadata(&mut row, source, rank);
    row.spectral_rank = C51Measurement::Measured(rank);
    row.spectral_basis_storage_values = C51Measurement::Measured(source.basis_storage_values);
    row.spectral_gp_storage_values =
        C51Measurement::Measured(diagnostics.ambient_dimension * rank + 2 * rank);
    row.spectral_prepared_storage_values =
        C51Measurement::Measured(output_diagnostics.combined_persistent_f64_entries);
    row.spectral_dense_factor_values =
        C51Measurement::Measured(diagnostics.dense_factor_allocated_entries);
    row.spectral_state_feature_values = C51Measurement::Measured(diagnostics.state_feature_entries);
    row.spectral_observation_feature_values =
        C51Measurement::Measured(diagnostics.observation_feature_entries);
    row.spectral_sampling_normal_dimension =
        C51Measurement::Measured(diagnostics.sampling_standard_normal_dimension);
    row.spectral_rank_fraction =
        C51Measurement::Measured(rank as f64 / diagnostics.ambient_dimension as f64);
    row.solver = match diagnostics.conditioning_strategy {
        LowRankConditioningStrategy::ObservationSpace => {
            "spectral_observation_space_dense_cholesky"
        }
        LowRankConditioningStrategy::LatentSpace => "spectral_latent_space_dense_cholesky",
    }
    .to_string();
    row.ordering = "not_applicable".to_string();
    row.feec_assembly_ns = duration_measurement(feec_assembly);
    row.prior_build_ns =
        duration_measurement(source.operator_recovery + source.eigensolve + gp_build);
    row.spectral_operator_recovery_ns = duration_measurement(source.operator_recovery);
    row.spectral_eigensolve_ns = duration_measurement(source.eigensolve);
    row.spectral_gp_build_ns = duration_measurement(gp_build);
    row.observation_update_assembly_ns = C51Measurement::unavailable(
        "prepared spectral API exposes combined feature projection and dense factor setup",
    );
    row.observation_update_and_factor_ns = duration_measurement(setup_elapsed);
    row.symbolic_analysis_ns = C51Measurement::unavailable(
        "symbolic sparse analysis is not applicable to dense spectral conditioning",
    );
    row.numeric_factor_ns =
        C51Measurement::unavailable("prepared spectral API exposes combined dense setup time");
    row.combined_factor_setup_ns = duration_measurement(setup_elapsed);
    if selection.phase == C51ResourcePhase::QueryPhysicalUq {
        let values = repeated_observation_values(
            design.sensor_count,
            selection.repeated_data_count,
            config.seed,
        );
        let repeated_start = Instant::now();
        for data in &values {
            std::hint::black_box(prepared_output.output_mean(data).map_err(spectral_error)?);
        }
        row.repeated_data_ns = duration_measurement(repeated_start.elapsed());
        let physical_start = Instant::now();
        let covariance = prepared_output.covariance().map_err(spectral_error)?;
        let covariance_action = (0..covariance.nrows())
            .map(|row_index| {
                (0..covariance.ncols())
                    .map(|column_index| {
                        covariance[(row_index, column_index)]
                            * problem.physical_qoi_probe[column_index]
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        std::hint::black_box(&covariance_action);
        std::hint::black_box(dot(&problem.physical_qoi_probe, &covariance_action)?);
        row.physical_qoi_ns = duration_measurement(physical_start.elapsed());
        let variance_start = Instant::now();
        let variances = (0..covariance.nrows())
            .map(|index| covariance[(index, index)])
            .collect::<Vec<_>>();
        row.variance_estimation_ns = duration_measurement(variance_start.elapsed());
        row.covariance_action_ns = C51Measurement::unavailable(
            "the common physical-UQ workload excludes latent covariance actions",
        );
        row.sampling_ns = C51Measurement::unavailable(
            "the common physical-UQ workload excludes latent state sampling",
        );
        row.physical_transformed_variance_min_raw =
            C51Measurement::Measured(minimum_raw(&variances));
        row.physical_transformed_variance_negative_count =
            C51Measurement::Measured(negative_count(&variances));
    }
    Ok(row)
}

// These arguments are the frozen method/problem/observation axes of one row;
// bundling them into an opaque context would make accidental cross-row reuse
// harder to audit in this experimental benchmark.
#[allow(clippy::too_many_arguments)]
fn run_spd_method(
    method: C51MethodId,
    prior: &GaussianPrior,
    prior_build: Duration,
    feec_assembly: Duration,
    config: &C51BenchmarkConfig,
    refinement: usize,
    design: &ObservationDesign,
    repeated_data_count: usize,
    problem: &ProblemMaps,
    reference: Option<&ReferenceAnswers>,
) -> Result<(C51BenchmarkRow, ReferenceAnswers)> {
    let values = repeated_observation_values(design.sensor_count, repeated_data_count, config.seed);
    let observation = LinearObservation::new(
        design.operator.clone(),
        values[0].clone(),
        GaussianNoise::variance(config.observation_noise_variance)?,
    )?;
    let setup_start = Instant::now();
    let (prepared, resource_report) = LinearGaussianModelBuilder::new(prior.clone())
        .observe(observation)?
        .prepare_with_resource_report()?;
    let setup_elapsed = setup_start.elapsed();
    let factor = prepared.factorization_diagnostics();
    let conditioning = prepared.conditioning_diagnostics();
    let observation_update = conditioning.observation_terms.first().ok_or_else(|| {
        FeecGmrfError::Inference(
            "C51 prepared model is missing observation diagnostics".to_string(),
        )
    })?;

    let repeated_start = Instant::now();
    let mut posterior = None;
    for (index, data) in values.iter().enumerate() {
        if index + 1 == values.len() {
            posterior =
                Some(prepared.condition_with_observation_values(std::slice::from_ref(data))?);
        } else {
            let _ = prepared.latent_mean_with_observation_values(std::slice::from_ref(data))?;
        }
    }
    let repeated_elapsed = repeated_start.elapsed();
    let mut posterior = posterior.expect("positive repeated-data count produces a posterior");
    let query_start = Instant::now();
    let covariance_action = posterior.covariance_action(&problem.latent_probe)?;
    let covariance_elapsed = query_start.elapsed();
    let physical_rhs = problem
        .physical_qoi
        .apply_transpose(&problem.physical_qoi_probe)?;
    let physical_start = Instant::now();
    let physical_covariance_action = posterior.covariance_action(&physical_rhs)?;
    let physical_output_covariance_action =
        problem.physical_qoi.apply(&physical_covariance_action)?;
    let physical_variance = dot(&physical_rhs, &physical_covariance_action)?;
    let physical_mean = problem.physical_qoi.apply(posterior.mean())?;
    let physical_elapsed = physical_start.elapsed();
    let variance_start = Instant::now();
    let physical_transformed_variances = posterior
        .pushforward_variance_estimate(&problem.physical_qoi, VarianceMethod::Exact)?
        .values;
    let variance_elapsed = variance_start.elapsed();
    let sample_start = Instant::now();
    let mut rng = StdRng::seed_from_u64(config.seed ^ method_seed(method));
    for _ in 0..config.posterior_sample_count {
        let _ = posterior.sample(&mut rng)?;
    }
    let sample_elapsed = sample_start.elapsed();
    let answers = ReferenceAnswers {
        mean: posterior.mean().to_vec(),
        covariance_action,
        physical_covariance_action,
        physical_output_covariance_action,
        physical_variance,
        physical_transformed_variances,
        physical_mean,
    };
    let errors = errors_against(&answers, reference)?;
    let row = C51BenchmarkRow {
        method,
        available: true,
        unavailable_reason: String::new(),
        discrete_law_id: if method == C51MethodId::FullMetricDirect {
            "exact_full_metric_whitney".to_string()
        } else {
            "projected_inverse_whitney".to_string()
        },
        same_discrete_law_eligible: method == C51MethodId::FullMetricDirect,
        matched_physical_qoi_eligible: method == C51MethodId::FullMetricDirect
            || physical_covariance_accuracy_qualified(&errors),
        common_physical_uq_workload_eligible: false,
        full_uq_workload_eligible: true,
        method_performance_timing_eligible: true,
        method_performance_timing_ineligibility_reason: String::new(),
        spectral_basis_source: "not_applicable".to_string(),
        spectral_eigensolver_backend: "not_applicable".to_string(),
        spectral_helper_workspace: String::new(),
        spectral_helper_binary: String::new(),
        spectral_feec_assembly_reuse: "not_applicable".to_string(),
        spectral_partial_which: "not_applicable".to_string(),
        spectral_partial_mass_solve: "not_applicable".to_string(),
        refinement,
        spatial_dimension: config.spatial_dimension,
        form_degree: config.form_degree,
        alpha: config.alpha.as_u32(),
        kappa: config.kappa,
        tau: config.tau,
        amplitude_calibration_policy: "none".to_string(),
        amplitude_calibration_target_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_uncalibrated_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_precision_scale: C51Measurement::NotRecorded,
        amplitude_calibration_tau_multiplier: C51Measurement::NotRecorded,
        amplitude_calibration_ns: C51Measurement::NotRecorded,
        observation_class: design.class,
        observation_design_id: design.design_id.clone(),
        observation_fixed_radius: design.fixed_radius.clone(),
        observation_vector_center_count: design.vector_center_count.clone(),
        observation_component_x_count: design.component_counts[0].clone(),
        observation_component_y_count: design.component_counts[1].clone(),
        observation_component_z_count: design.component_counts[2].clone(),
        physical_qoi_design_id: problem.physical_qoi_design_id.clone(),
        physical_qoi_output_count: problem.physical_qoi.output_dimension(),
        physical_qoi_selected_cell_count: problem.physical_qoi_selected_cell_count,
        physical_qoi_selected_volume_fraction: problem.physical_qoi_selected_volume_fraction,
        sensor_count: design.sensor_count,
        sensor_density: design.sensor_count as f64 / prior.dimension() as f64,
        repeated_data_count,
        posterior_sample_count: config.posterior_sample_count,
        seed: config.seed,
        latent_dimension: prior.dimension(),
        ambient_dimension: prior.dimension(),
        constraint_dimension: 0,
        observation_nnz: observation_update.operator_nonzeros,
        prior_precision_nnz: C51Measurement::Measured(conditioning.prior_precision_nonzeros),
        observation_update_nnz: C51Measurement::Measured(
            observation_update.precision_update_nonzeros,
        ),
        posterior_precision_nnz: C51Measurement::Measured(
            conditioning.posterior_precision_nonzeros,
        ),
        ambient_energy_nnz: C51Measurement::NotRecorded,
        constraint_nnz: C51Measurement::NotRecorded,
        kkt_input_nnz: C51Measurement::NotRecorded,
        factor_storage_nnz: usize_resource_measurement(&resource_report.factor_storage_nonzeros),
        certificate_factor_storage_nnz: C51Measurement::NotRecorded,
        fill_ratio: C51Measurement::Measured(factor.fill_ratio),
        exact_gp_dense_factor_dimension: C51Measurement::NotRecorded,
        exact_gp_dense_factor_allocated_values: C51Measurement::NotRecorded,
        exact_gp_dense_factor_triangular_values: C51Measurement::NotRecorded,
        exact_gp_output_observation_covariance_values: C51Measurement::NotRecorded,
        exact_gp_posterior_output_covariance_values: C51Measurement::NotRecorded,
        exact_gp_persistent_storage_values: C51Measurement::NotRecorded,
        exact_gp_posterior_covariance_max_abs_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_raw_max_relative_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_projection_frobenius_correction: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_residual_aware: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_policy: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_status: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_residual_status: "not_applicable".to_string(),
        spectral_rank: C51Measurement::NotRecorded,
        spectral_basis_storage_values: C51Measurement::NotRecorded,
        spectral_gp_storage_values: C51Measurement::NotRecorded,
        spectral_prepared_storage_values: C51Measurement::NotRecorded,
        spectral_dense_factor_values: C51Measurement::NotRecorded,
        spectral_state_feature_values: C51Measurement::NotRecorded,
        spectral_observation_feature_values: C51Measurement::NotRecorded,
        spectral_sampling_normal_dimension: C51Measurement::NotRecorded,
        spectral_rank_fraction: C51Measurement::NotRecorded,
        spectral_shared_maximum_basis_rank: C51Measurement::NotRecorded,
        spectral_partial_requested_rank: C51Measurement::NotRecorded,
        spectral_partial_converged_rank: C51Measurement::NotRecorded,
        spectral_partial_retained_rank: C51Measurement::NotRecorded,
        spectral_partial_l_nnz: C51Measurement::NotRecorded,
        spectral_partial_d_nnz: C51Measurement::NotRecorded,
        spectral_partial_c_nnz: C51Measurement::NotRecorded,
        spectral_partial_mkm1_nnz: C51Measurement::NotRecorded,
        spectral_partial_mk_nnz: C51Measurement::NotRecorded,
        spectral_partial_mass_nnz: C51Measurement::NotRecorded,
        spectral_partial_weak_hodge_laplacian_nnz: C51Measurement::NotRecorded,
        spectral_partial_input_nnz: C51Measurement::NotRecorded,
        spectral_partial_basis_storage_bytes: C51Measurement::NotRecorded,
        spectral_partial_clamped_negative_eigenvalues: C51Measurement::NotRecorded,
        spectral_partial_maximum_relative_mixed_residual: C51Measurement::NotRecorded,
        spectral_partial_maximum_mass_orthogonality_defect: C51Measurement::NotRecorded,
        spectral_partial_mass_orthogonality_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigenvalue_negative_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigensolver_tolerance: C51Measurement::NotRecorded,
        spectral_partial_residual_validation_ns: C51Measurement::NotRecorded,
        solver: "sparse_cholesky".to_string(),
        ordering: ordering_name(factor.ordering).to_string(),
        feec_assembly_ns: duration_measurement(feec_assembly),
        prior_build_ns: duration_measurement(prior_build),
        exact_gp_covariance_construction_ns: C51Measurement::NotRecorded,
        exact_gp_dense_conditioning_ns: C51Measurement::NotRecorded,
        exact_gp_end_to_end_setup_ns: C51Measurement::NotRecorded,
        spectral_operator_recovery_ns: C51Measurement::NotRecorded,
        spectral_eigensolve_ns: C51Measurement::NotRecorded,
        spectral_gp_build_ns: C51Measurement::NotRecorded,
        observation_update_assembly_ns: duration_resource_measurement(
            &resource_report.assembly.wall_time,
        ),
        observation_update_and_factor_ns: duration_measurement(setup_elapsed),
        symbolic_analysis_ns: duration_resource_measurement(&resource_report.analysis.wall_time),
        numeric_factor_ns: duration_resource_measurement(
            &resource_report.numeric_factorization.wall_time,
        ),
        combined_factor_setup_ns: duration_resource_measurement(
            &resource_report.combined_factor_setup.wall_time,
        ),
        repeated_data_ns: duration_measurement(repeated_elapsed),
        covariance_action_ns: duration_measurement(covariance_elapsed),
        physical_qoi_ns: duration_measurement(physical_elapsed),
        sampling_ns: duration_measurement(sample_elapsed),
        variance_estimation_ns: duration_measurement(variance_elapsed),
        factor_external_peak_rss_bytes: C51Measurement::NotRecorded,
        query_external_peak_rss_bytes: C51Measurement::NotRecorded,
        posterior_mean_relative_error: errors.posterior_mean,
        covariance_action_relative_error: errors.covariance_action,
        physical_covariance_action_relative_error: errors.physical_covariance_action,
        physical_output_covariance_action_relative_error: errors.physical_output_covariance_action,
        physical_variance_relative_error: errors.physical_variance,
        physical_transformed_variance_relative_error: errors.physical_transformed_variance,
        physical_mean_relative_error: errors.physical_mean,
        physical_transformed_variance_min_raw: C51Measurement::Measured(minimum_raw(
            &answers.physical_transformed_variances,
        )),
        physical_transformed_variance_negative_count: C51Measurement::Measured(negative_count(
            &answers.physical_transformed_variances,
        )),
    };
    Ok((row, answers))
}

// Keep the exact-supported row signature parallel to the SPD row so the
// comparator receives the identical frozen problem and observation axes.
#[allow(clippy::too_many_arguments)]
fn run_supported_method(
    prior: &SupportedMaternPrior,
    prior_build: Duration,
    feec_assembly: Duration,
    config: &C51BenchmarkConfig,
    refinement: usize,
    design: &ObservationDesign,
    repeated_data_count: usize,
    problem: &ProblemMaps,
    reference: Option<&ReferenceAnswers>,
) -> Result<(C51BenchmarkRow, ReferenceAnswers)> {
    let values = repeated_observation_values(design.sensor_count, repeated_data_count, config.seed);
    let first_observation = LinearObservation::new(
        design.operator.clone(),
        values[0].clone(),
        GaussianNoise::variance(config.observation_noise_variance)?,
    )?;
    let setup_start = Instant::now();
    let prepared = prior.prepare(&[first_observation], &[])?;
    let setup_elapsed = setup_start.elapsed();
    let repeated_start = Instant::now();
    let mut posterior = None;
    for (index, data) in values.iter().enumerate() {
        if index + 1 == values.len() {
            posterior =
                Some(prepared.condition_with_observation_values(std::slice::from_ref(data))?);
        } else {
            let _ = prepared.latent_mean_with_observation_values(std::slice::from_ref(data))?;
        }
    }
    let repeated_elapsed = repeated_start.elapsed();
    let posterior = posterior.expect("positive repeated-data count produces a posterior");
    let diagnostics = posterior.diagnostics();
    let query_start = Instant::now();
    let covariance_action = posterior.covariance_action(&problem.latent_probe)?;
    let covariance_elapsed = query_start.elapsed();
    let physical_rhs = problem
        .physical_qoi
        .apply_transpose(&problem.physical_qoi_probe)?;
    let physical_start = Instant::now();
    let physical_covariance_action = posterior.covariance_action(&physical_rhs)?;
    let physical_output_covariance_action =
        problem.physical_qoi.apply(&physical_covariance_action)?;
    let physical_variance = dot(&physical_rhs, &physical_covariance_action)?;
    let physical_mean = problem.physical_qoi.apply(posterior.mean())?;
    let physical_elapsed = physical_start.elapsed();
    let variance_start = Instant::now();
    let physical_transformed_variances = posterior.pushforward_variances(&problem.physical_qoi)?;
    let variance_elapsed = variance_start.elapsed();
    let answers = ReferenceAnswers {
        mean: posterior.mean().to_vec(),
        covariance_action,
        physical_covariance_action,
        physical_output_covariance_action,
        physical_variance,
        physical_transformed_variances,
        physical_mean,
    };
    let errors = errors_against(&answers, reference)?;
    let row = C51BenchmarkRow {
        method: C51MethodId::FullMetricAugmented,
        available: true,
        unavailable_reason: String::new(),
        discrete_law_id: "exact_full_metric_whitney".to_string(),
        same_discrete_law_eligible: true,
        matched_physical_qoi_eligible: true,
        common_physical_uq_workload_eligible: false,
        full_uq_workload_eligible: false,
        method_performance_timing_eligible: true,
        method_performance_timing_ineligibility_reason: String::new(),
        spectral_basis_source: "not_applicable".to_string(),
        spectral_eigensolver_backend: "not_applicable".to_string(),
        spectral_helper_workspace: String::new(),
        spectral_helper_binary: String::new(),
        spectral_feec_assembly_reuse: "not_applicable".to_string(),
        spectral_partial_which: "not_applicable".to_string(),
        spectral_partial_mass_solve: "not_applicable".to_string(),
        refinement,
        spatial_dimension: config.spatial_dimension,
        form_degree: config.form_degree,
        alpha: config.alpha.as_u32(),
        kappa: config.kappa,
        tau: config.tau,
        amplitude_calibration_policy: "none".to_string(),
        amplitude_calibration_target_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_uncalibrated_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_precision_scale: C51Measurement::NotRecorded,
        amplitude_calibration_tau_multiplier: C51Measurement::NotRecorded,
        amplitude_calibration_ns: C51Measurement::NotRecorded,
        observation_class: design.class,
        observation_design_id: design.design_id.clone(),
        observation_fixed_radius: design.fixed_radius.clone(),
        observation_vector_center_count: design.vector_center_count.clone(),
        observation_component_x_count: design.component_counts[0].clone(),
        observation_component_y_count: design.component_counts[1].clone(),
        observation_component_z_count: design.component_counts[2].clone(),
        physical_qoi_design_id: problem.physical_qoi_design_id.clone(),
        physical_qoi_output_count: problem.physical_qoi.output_dimension(),
        physical_qoi_selected_cell_count: problem.physical_qoi_selected_cell_count,
        physical_qoi_selected_volume_fraction: problem.physical_qoi_selected_volume_fraction,
        sensor_count: design.sensor_count,
        sensor_density: design.sensor_count as f64 / prior.state_dimension() as f64,
        repeated_data_count,
        posterior_sample_count: config.posterior_sample_count,
        seed: config.seed,
        latent_dimension: prior.state_dimension(),
        ambient_dimension: diagnostics.ambient_dimension,
        constraint_dimension: diagnostics.constraint_dimension,
        observation_nnz: design.operator.matrix().nnz(),
        prior_precision_nnz: C51Measurement::unavailable(
            "FM-AUG is a constrained energy, not an eliminated SPD precision",
        ),
        observation_update_nnz: C51Measurement::unavailable(UPDATE_NNZ_PENDING_REASON),
        posterior_precision_nnz: C51Measurement::unavailable(
            "FM-AUG is a constrained energy, not an eliminated SPD precision",
        ),
        ambient_energy_nnz: C51Measurement::Measured(diagnostics.ambient_energy_nnz),
        constraint_nnz: C51Measurement::Measured(diagnostics.constraint_nnz),
        kkt_input_nnz: C51Measurement::Measured(diagnostics.kkt_nnz),
        factor_storage_nnz: C51Measurement::unavailable(KKT_FILL_UNAVAILABLE_REASON),
        certificate_factor_storage_nnz: C51Measurement::Measured(
            diagnostics.constraint_gram_factor_nnz + diagnostics.supported_energy_factor_nnz,
        ),
        fill_ratio: C51Measurement::unavailable(KKT_FILL_UNAVAILABLE_REASON),
        exact_gp_dense_factor_dimension: C51Measurement::NotRecorded,
        exact_gp_dense_factor_allocated_values: C51Measurement::NotRecorded,
        exact_gp_dense_factor_triangular_values: C51Measurement::NotRecorded,
        exact_gp_output_observation_covariance_values: C51Measurement::NotRecorded,
        exact_gp_posterior_output_covariance_values: C51Measurement::NotRecorded,
        exact_gp_persistent_storage_values: C51Measurement::NotRecorded,
        exact_gp_posterior_covariance_max_abs_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_raw_max_relative_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_projection_frobenius_correction: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_residual_aware: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_policy: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_status: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_residual_status: "not_applicable".to_string(),
        spectral_rank: C51Measurement::NotRecorded,
        spectral_basis_storage_values: C51Measurement::NotRecorded,
        spectral_gp_storage_values: C51Measurement::NotRecorded,
        spectral_prepared_storage_values: C51Measurement::NotRecorded,
        spectral_dense_factor_values: C51Measurement::NotRecorded,
        spectral_state_feature_values: C51Measurement::NotRecorded,
        spectral_observation_feature_values: C51Measurement::NotRecorded,
        spectral_sampling_normal_dimension: C51Measurement::NotRecorded,
        spectral_rank_fraction: C51Measurement::NotRecorded,
        spectral_shared_maximum_basis_rank: C51Measurement::NotRecorded,
        spectral_partial_requested_rank: C51Measurement::NotRecorded,
        spectral_partial_converged_rank: C51Measurement::NotRecorded,
        spectral_partial_retained_rank: C51Measurement::NotRecorded,
        spectral_partial_l_nnz: C51Measurement::NotRecorded,
        spectral_partial_d_nnz: C51Measurement::NotRecorded,
        spectral_partial_c_nnz: C51Measurement::NotRecorded,
        spectral_partial_mkm1_nnz: C51Measurement::NotRecorded,
        spectral_partial_mk_nnz: C51Measurement::NotRecorded,
        spectral_partial_mass_nnz: C51Measurement::NotRecorded,
        spectral_partial_weak_hodge_laplacian_nnz: C51Measurement::NotRecorded,
        spectral_partial_input_nnz: C51Measurement::NotRecorded,
        spectral_partial_basis_storage_bytes: C51Measurement::NotRecorded,
        spectral_partial_clamped_negative_eigenvalues: C51Measurement::NotRecorded,
        spectral_partial_maximum_relative_mixed_residual: C51Measurement::NotRecorded,
        spectral_partial_maximum_mass_orthogonality_defect: C51Measurement::NotRecorded,
        spectral_partial_mass_orthogonality_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigenvalue_negative_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigensolver_tolerance: C51Measurement::NotRecorded,
        spectral_partial_residual_validation_ns: C51Measurement::NotRecorded,
        solver: "sparse_lu_kkt".to_string(),
        ordering: "backend_partial_pivoting".to_string(),
        feec_assembly_ns: duration_measurement(feec_assembly),
        prior_build_ns: duration_measurement(prior_build),
        exact_gp_covariance_construction_ns: C51Measurement::NotRecorded,
        exact_gp_dense_conditioning_ns: C51Measurement::NotRecorded,
        exact_gp_end_to_end_setup_ns: C51Measurement::NotRecorded,
        spectral_operator_recovery_ns: C51Measurement::NotRecorded,
        spectral_eigensolve_ns: C51Measurement::NotRecorded,
        spectral_gp_build_ns: C51Measurement::NotRecorded,
        observation_update_assembly_ns: C51Measurement::unavailable(
            "the supported-law API currently exposes combined observation assembly and KKT setup",
        ),
        observation_update_and_factor_ns: duration_measurement(setup_elapsed),
        symbolic_analysis_ns: C51Measurement::unavailable(
            "the sparse LU backend does not expose separate symbolic analysis",
        ),
        numeric_factor_ns: C51Measurement::unavailable(
            "the sparse LU backend does not expose separate numeric factorization",
        ),
        combined_factor_setup_ns: duration_measurement(setup_elapsed),
        repeated_data_ns: duration_measurement(repeated_elapsed),
        covariance_action_ns: duration_measurement(covariance_elapsed),
        physical_qoi_ns: duration_measurement(physical_elapsed),
        sampling_ns: C51Measurement::unavailable(
            "support-aware square-root sampling is not implemented",
        ),
        variance_estimation_ns: duration_measurement(variance_elapsed),
        factor_external_peak_rss_bytes: C51Measurement::NotRecorded,
        query_external_peak_rss_bytes: C51Measurement::NotRecorded,
        posterior_mean_relative_error: errors.posterior_mean,
        covariance_action_relative_error: errors.covariance_action,
        physical_covariance_action_relative_error: errors.physical_covariance_action,
        physical_output_covariance_action_relative_error: errors.physical_output_covariance_action,
        physical_variance_relative_error: errors.physical_variance,
        physical_transformed_variance_relative_error: errors.physical_transformed_variance,
        physical_mean_relative_error: errors.physical_mean,
        physical_transformed_variance_min_raw: C51Measurement::Measured(minimum_raw(
            &answers.physical_transformed_variances,
        )),
        physical_transformed_variance_negative_count: C51Measurement::Measured(negative_count(
            &answers.physical_transformed_variances,
        )),
    };
    Ok((row, answers))
}

// This exact-GP comparator deliberately owns only observation/output-space
// queries. It never materializes a latent covariance or labels any dense
// quantity as a sparse precision/factor count.
#[allow(clippy::too_many_arguments)]
fn run_exact_observation_gp_method(
    prior: &SupportedMaternPrior,
    prior_build: Duration,
    feec_assembly: Duration,
    config: &C51BenchmarkConfig,
    refinement: usize,
    design: &ObservationDesign,
    repeated_data_count: usize,
    problem: &ProblemMaps,
    phase: C51ResourcePhase,
    reference: Option<&ReferenceAnswers>,
) -> Result<C51BenchmarkRow> {
    if !matches!(
        design.class,
        C51ObservationClass::LocalPhysicalB | C51ObservationClass::LocalFluxCochain
    ) {
        return Err(FeecGmrfError::InvalidParameter(
            "the exact observation-space GP comparator requires local physical observations"
                .to_string(),
        ));
    }
    let covariance_start = Instant::now();
    let factor = prior.factor()?;
    let factor_diagnostics = factor.diagnostics();
    let stacked_dimension = design
        .sensor_count
        .checked_add(problem.physical_qoi.output_dimension())
        .ok_or_else(|| {
            FeecGmrfError::Dimension(
                "C51 exact-GP stacked covariance dimension overflowed usize".to_string(),
            )
        })?;
    let blocks = ExactObservationCovarianceBlocks::from_supported_matern_factor(
        &factor,
        &design.operator,
        &problem.physical_qoi,
        stacked_dimension,
    )?;
    let audit = match blocks.provenance() {
        ExactObservationCovarianceProvenance::SupportedMaternStacked { audit } => audit.clone(),
        ExactObservationCovarianceProvenance::CallerSuppliedExactlySymmetric => {
            return Err(FeecGmrfError::Inference(
                "C51 exact-GP supported covariance lost its raw-skew audit provenance".to_string(),
            ))
        }
    };
    let covariance_construction = covariance_start.elapsed();
    let dense_start = Instant::now();
    let posterior = ExactObservationPosterior::from_prior_covariance_blocks(
        blocks,
        config.observation_noise_variance,
    )?;
    let dense_conditioning = dense_start.elapsed();
    let diagnostics = posterior.diagnostics();

    let synthetic_selection = C51ResourceSelection {
        refinement,
        observation_class: design.class,
        sensor_target: C51ResourceSensorTarget::Count(design.sensor_count),
        repeated_data_count,
        method: C51MethodId::ExactObservationSpaceGp,
        spectral_rank: None,
        phase,
    };
    let mut row = base_resource_row(
        C51MethodId::ExactObservationSpaceGp,
        config,
        &synthetic_selection,
        design,
        problem,
        prior.state_dimension(),
        factor_diagnostics.ambient_dimension,
        factor_diagnostics.constraint_dimension,
    );
    row.same_discrete_law_eligible = true;
    row.matched_physical_qoi_eligible = phase == C51ResourcePhase::QueryFullUq;
    row.common_physical_uq_workload_eligible = matches!(
        phase,
        C51ResourcePhase::ColdSetup | C51ResourcePhase::QueryPhysicalUq
    ) && config.physical_observation_design
        == C51PhysicalObservationDesign::JuqFixedFootprintPrefix
        && config.physical_output_design == C51PhysicalOutputDesign::RankStudySelectionAndHeldOut;
    row.full_uq_workload_eligible = false;
    row.prior_precision_nnz = C51Measurement::unavailable(
        "exact observation-space GP consumes covariance blocks, not a sparse precision",
    );
    row.observation_update_nnz = C51Measurement::unavailable(
        "exact observation-space GP applies dense covariance conditioning, not a sparse update",
    );
    row.posterior_precision_nnz = C51Measurement::unavailable(
        "exact observation-space GP does not assemble a latent posterior precision",
    );
    row.factor_storage_nnz = C51Measurement::unavailable(
        "exact observation-space GP uses a dense observation factor; see exact_gp_dense fields",
    );
    row.fill_ratio = C51Measurement::unavailable(
        "sparse fill is not applicable to exact dense observation-space conditioning",
    );
    row.exact_gp_dense_factor_dimension =
        C51Measurement::Measured(diagnostics.dense_factor_dimension);
    row.exact_gp_dense_factor_allocated_values =
        C51Measurement::Measured(diagnostics.dense_factor_allocated_entries);
    row.exact_gp_dense_factor_triangular_values =
        C51Measurement::Measured(diagnostics.dense_factor_triangular_entries);
    row.exact_gp_output_observation_covariance_values =
        C51Measurement::Measured(diagnostics.output_observation_covariance_entries);
    row.exact_gp_posterior_output_covariance_values =
        C51Measurement::Measured(diagnostics.posterior_output_covariance_entries);
    row.exact_gp_persistent_storage_values =
        C51Measurement::Measured(diagnostics.persistent_f64_entries);
    row.exact_gp_posterior_covariance_max_abs_skew =
        C51Measurement::Measured(diagnostics.posterior_output_covariance_max_abs_skew);
    row.exact_gp_prior_covariance_raw_max_relative_skew =
        C51Measurement::Measured(audit.raw_maximum_relative_skew);
    row.exact_gp_prior_covariance_projection_frobenius_correction =
        C51Measurement::Measured(audit.projection_frobenius_correction);
    row.exact_gp_prior_covariance_audit_residual_aware =
        C51Measurement::Measured(audit.residual_aware);
    row.exact_gp_prior_covariance_audit_policy = audit.policy_version.to_string();
    row.exact_gp_prior_covariance_audit_status = audit.status.to_string();
    row.exact_gp_prior_covariance_audit_residual_status = audit.residual_status.to_string();
    row.solver = "exact_observation_space_dense_cholesky".to_string();
    row.ordering = "not_applicable".to_string();
    row.feec_assembly_ns = duration_measurement(feec_assembly);
    row.prior_build_ns = duration_measurement(prior_build);
    row.exact_gp_covariance_construction_ns = duration_measurement(covariance_construction);
    row.exact_gp_dense_conditioning_ns = duration_measurement(dense_conditioning);
    row.exact_gp_end_to_end_setup_ns = duration_measurement(
        feec_assembly + prior_build + covariance_construction + dense_conditioning,
    );
    row.observation_update_assembly_ns =
        C51Measurement::unavailable("exact GP covariance-block construction is charged separately");
    row.observation_update_and_factor_ns = duration_measurement(dense_conditioning);
    row.symbolic_analysis_ns = C51Measurement::unavailable(
        "symbolic sparse analysis is not applicable to a dense observation Cholesky",
    );
    row.numeric_factor_ns = C51Measurement::unavailable(
        "exact GP backend reports combined dense Cholesky and posterior-output covariance setup",
    );
    row.combined_factor_setup_ns =
        duration_measurement(covariance_construction + dense_conditioning);
    if phase == C51ResourcePhase::ColdSetup {
        return Ok(row);
    }

    let values = repeated_observation_values(design.sensor_count, repeated_data_count, config.seed);
    let repeated_start = Instant::now();
    let mut physical_mean = Vec::new();
    for data in &values {
        physical_mean = posterior.output_mean(data)?;
    }
    let repeated_elapsed = repeated_start.elapsed();
    let physical_start = Instant::now();
    let physical_output_covariance_action =
        posterior.posterior_output_covariance_action(&problem.physical_qoi_probe)?;
    let physical_variance = dot(
        &problem.physical_qoi_probe,
        &physical_output_covariance_action,
    )?;
    let physical_elapsed = physical_start.elapsed();
    let variance_start = Instant::now();
    let covariance = posterior.posterior_output_covariance();
    let physical_transformed_variances = (0..problem.physical_qoi.output_dimension())
        .map(|index| covariance[index][index])
        .collect::<Vec<_>>();
    let variance_elapsed = variance_start.elapsed();

    row.repeated_data_ns = duration_measurement(repeated_elapsed);
    row.covariance_action_ns = C51Measurement::unavailable(
        "exact observation-space GP does not construct a latent covariance action",
    );
    row.physical_qoi_ns = duration_measurement(physical_elapsed);
    row.sampling_ns = C51Measurement::unavailable(
        "the exact observation-space facade exposes output moments, not latent sampling",
    );
    row.variance_estimation_ns = duration_measurement(variance_elapsed);
    row.physical_transformed_variance_min_raw =
        C51Measurement::Measured(minimum_raw(&physical_transformed_variances));
    row.physical_transformed_variance_negative_count =
        C51Measurement::Measured(negative_count(&physical_transformed_variances));
    row.posterior_mean_relative_error = C51Measurement::unavailable(
        "exact observation-space GP does not construct a latent posterior mean",
    );
    row.covariance_action_relative_error = C51Measurement::unavailable(
        "exact observation-space GP does not construct a latent covariance action",
    );
    row.physical_covariance_action_relative_error = C51Measurement::unavailable(
        "exact observation-space GP does not construct the latent C G^T action",
    );
    if let Some(reference) = reference {
        row.physical_output_covariance_action_relative_error =
            C51Measurement::Measured(relative_l2(
                &physical_output_covariance_action,
                &reference.physical_output_covariance_action,
            )?);
        row.physical_variance_relative_error = C51Measurement::Measured(relative_scalar(
            physical_variance,
            reference.physical_variance,
        ));
        row.physical_transformed_variance_relative_error = C51Measurement::Measured(relative_l2(
            &physical_transformed_variances,
            &reference.physical_transformed_variances,
        )?);
        row.physical_mean_relative_error =
            C51Measurement::Measured(relative_l2(&physical_mean, &reference.physical_mean)?);
    } else {
        row.physical_output_covariance_action_relative_error = C51Measurement::Measured(0.0);
        row.physical_variance_relative_error = C51Measurement::Measured(0.0);
        row.physical_transformed_variance_relative_error = C51Measurement::Measured(0.0);
        row.physical_mean_relative_error = C51Measurement::Measured(0.0);
    }
    Ok(row)
}

#[allow(clippy::too_many_arguments)]
fn run_spectral_method(
    method: C51MethodId,
    rank: usize,
    source: &SpectralReference,
    feec_assembly: Duration,
    config: &C51BenchmarkConfig,
    refinement: usize,
    design: &ObservationDesign,
    repeated_data_count: usize,
    problem: &ProblemMaps,
    reference: Option<&ReferenceAnswers>,
) -> Result<C51BenchmarkRow> {
    if !matches!(
        method,
        C51MethodId::SpectralFullRank | C51MethodId::SpectralLowRank
    ) {
        return Err(FeecGmrfError::InvalidParameter(
            "run_spectral_method requires a spectral method identifier".to_string(),
        ));
    }
    let full_rank = source.basis.len();
    if rank == 0
        || rank > full_rank
        || (method == C51MethodId::SpectralFullRank && rank != full_rank)
    {
        return Err(FeecGmrfError::InvalidParameter(format!(
            "invalid C51 spectral rank {rank} for method {} and full rank {full_rank}",
            method.as_str()
        )));
    }
    let gp_start = Instant::now();
    let gp = SpectralMaternGp::from_eigenbasis(
        source.basis.clone(),
        SpectralMaternConfig {
            kappa: config.kappa,
            alpha: f64::from(config.alpha.as_u32()),
            tau: config.tau,
            k: rank,
        },
    )
    .map_err(spectral_error)?;
    let gp_build = gp_start.elapsed();
    let observation = feg_infer::sparse::core_triplet_to_feec_csr(design.operator.matrix());
    let physical_qoi = feg_infer::sparse::core_triplet_to_feec_csr(problem.physical_qoi.matrix());
    let setup_start = Instant::now();
    let prepared = gp
        .prepare_linear_observations(&observation, config.observation_noise_variance)
        .map_err(spectral_error)?;
    let setup_elapsed = setup_start.elapsed();
    let diagnostics = prepared.diagnostics();
    let values = repeated_observation_values(design.sensor_count, repeated_data_count, config.seed);
    let repeated_start = Instant::now();
    let mut mean = Vec::new();
    for data in &values {
        mean = prepared.state_mean(data).map_err(spectral_error)?;
    }
    let repeated_elapsed = repeated_start.elapsed();
    let final_values = values
        .last()
        .expect("positive repeated-data count has final values");
    let covariance_start = Instant::now();
    let covariance_action = prepared
        .state_covariance_action(&problem.latent_probe)
        .map_err(spectral_error)?;
    let covariance_elapsed = covariance_start.elapsed();
    let physical_rhs = problem
        .physical_qoi
        .apply_transpose(&problem.physical_qoi_probe)?;
    let physical_start = Instant::now();
    let physical_covariance_action = prepared
        .state_covariance_action(&physical_rhs)
        .map_err(spectral_error)?;
    let physical_output_action = prepared
        .output_covariance_action(&physical_qoi, &problem.physical_qoi_probe)
        .map_err(spectral_error)?;
    let physical_variance = dot(&problem.physical_qoi_probe, &physical_output_action)?;
    let full_output = prepared
        .output_moments_with_covariance(&physical_qoi, final_values)
        .map_err(spectral_error)?;
    let physical_elapsed = physical_start.elapsed();
    let variance_start = Instant::now();
    let output = prepared
        .output_moments(&physical_qoi, final_values)
        .map_err(spectral_error)?;
    let variance_elapsed = variance_start.elapsed();
    if full_output.covariance.nrows() != problem.physical_qoi.output_dimension()
        || full_output.covariance.ncols() != problem.physical_qoi.output_dimension()
        || relative_l2(&full_output.mean, &output.mean)? > 1.0e-12
        || relative_l2(
            &mean,
            &prepared.state_mean(final_values).map_err(spectral_error)?,
        )? > 1.0e-12
    {
        return Err(FeecGmrfError::Inference(
            "C51 prepared spectral output/state mean routes disagree".to_string(),
        ));
    }
    let sample_start = Instant::now();
    let mut rng = StdRng::seed_from_u64(config.seed ^ method_seed(method) ^ rank as u64);
    for _ in 0..config.posterior_sample_count {
        let standard_normal = (0..diagnostics.sampling_standard_normal_dimension)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect::<Vec<f64>>();
        prepared
            .state_sample_from_standard_normal(final_values, &standard_normal)
            .map_err(spectral_error)?;
    }
    let sampling_elapsed = sample_start.elapsed();
    let answers = ReferenceAnswers {
        mean,
        covariance_action,
        physical_covariance_action,
        physical_output_covariance_action: physical_output_action,
        physical_variance,
        physical_transformed_variances: output.variance,
        physical_mean: output.mean,
    };
    let errors = errors_against(&answers, reference)?;
    let mut row = C51BenchmarkRow {
        method,
        available: true,
        unavailable_reason: String::new(),
        discrete_law_id: if method == C51MethodId::SpectralFullRank {
            "exact_full_metric_whitney".to_string()
        } else {
            "truncated_exact_full_metric_whitney_spectral".to_string()
        },
        same_discrete_law_eligible: method == C51MethodId::SpectralFullRank,
        matched_physical_qoi_eligible: method == C51MethodId::SpectralFullRank
            || physical_covariance_accuracy_qualified(&errors),
        common_physical_uq_workload_eligible: false,
        full_uq_workload_eligible: true,
        method_performance_timing_eligible: false,
        method_performance_timing_ineligibility_reason:
            "spectral source metadata has not been classified".to_string(),
        spectral_basis_source: "unclassified".to_string(),
        spectral_eigensolver_backend: "unclassified".to_string(),
        spectral_helper_workspace: String::new(),
        spectral_helper_binary: String::new(),
        spectral_feec_assembly_reuse: "unclassified".to_string(),
        spectral_partial_which: "not_recorded".to_string(),
        spectral_partial_mass_solve: "not_recorded".to_string(),
        refinement,
        spatial_dimension: config.spatial_dimension,
        form_degree: config.form_degree,
        alpha: config.alpha.as_u32(),
        kappa: config.kappa,
        tau: config.tau,
        amplitude_calibration_policy: "none".to_string(),
        amplitude_calibration_target_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_uncalibrated_prior_rms: C51Measurement::NotRecorded,
        amplitude_calibration_precision_scale: C51Measurement::NotRecorded,
        amplitude_calibration_tau_multiplier: C51Measurement::NotRecorded,
        amplitude_calibration_ns: C51Measurement::NotRecorded,
        observation_class: design.class,
        observation_design_id: design.design_id.clone(),
        observation_fixed_radius: design.fixed_radius.clone(),
        observation_vector_center_count: design.vector_center_count.clone(),
        observation_component_x_count: design.component_counts[0].clone(),
        observation_component_y_count: design.component_counts[1].clone(),
        observation_component_z_count: design.component_counts[2].clone(),
        physical_qoi_design_id: problem.physical_qoi_design_id.clone(),
        physical_qoi_output_count: problem.physical_qoi.output_dimension(),
        physical_qoi_selected_cell_count: problem.physical_qoi_selected_cell_count,
        physical_qoi_selected_volume_fraction: problem.physical_qoi_selected_volume_fraction,
        sensor_count: design.sensor_count,
        sensor_density: design.sensor_count as f64 / diagnostics.ambient_dimension as f64,
        repeated_data_count,
        posterior_sample_count: config.posterior_sample_count,
        seed: config.seed,
        latent_dimension: diagnostics.latent_dimension,
        ambient_dimension: diagnostics.ambient_dimension,
        constraint_dimension: 0,
        observation_nnz: diagnostics.observation_operator_nonzeros,
        prior_precision_nnz: C51Measurement::unavailable(SPECTRAL_SPARSE_PRECISION_REASON),
        observation_update_nnz: C51Measurement::unavailable(SPECTRAL_SPARSE_PRECISION_REASON),
        posterior_precision_nnz: C51Measurement::unavailable(SPECTRAL_SPARSE_PRECISION_REASON),
        ambient_energy_nnz: C51Measurement::NotRecorded,
        constraint_nnz: C51Measurement::NotRecorded,
        kkt_input_nnz: C51Measurement::NotRecorded,
        factor_storage_nnz: C51Measurement::unavailable(
            "spectral conditioning uses a dense low-rank factor; see spectral storage fields",
        ),
        certificate_factor_storage_nnz: C51Measurement::NotRecorded,
        fill_ratio: C51Measurement::unavailable(
            "sparse Cholesky fill is not applicable to spectral conditioning",
        ),
        exact_gp_dense_factor_dimension: C51Measurement::NotRecorded,
        exact_gp_dense_factor_allocated_values: C51Measurement::NotRecorded,
        exact_gp_dense_factor_triangular_values: C51Measurement::NotRecorded,
        exact_gp_output_observation_covariance_values: C51Measurement::NotRecorded,
        exact_gp_posterior_output_covariance_values: C51Measurement::NotRecorded,
        exact_gp_persistent_storage_values: C51Measurement::NotRecorded,
        exact_gp_posterior_covariance_max_abs_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_raw_max_relative_skew: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_projection_frobenius_correction: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_residual_aware: C51Measurement::NotRecorded,
        exact_gp_prior_covariance_audit_policy: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_status: "not_applicable".to_string(),
        exact_gp_prior_covariance_audit_residual_status: "not_applicable".to_string(),
        spectral_rank: C51Measurement::Measured(rank),
        spectral_basis_storage_values: C51Measurement::Measured(source.basis_storage_values),
        spectral_gp_storage_values: C51Measurement::Measured(
            diagnostics.ambient_dimension * rank + 2 * rank,
        ),
        spectral_prepared_storage_values: C51Measurement::Measured(
            diagnostics.persistent_f64_entries,
        ),
        spectral_dense_factor_values: C51Measurement::Measured(
            diagnostics.dense_factor_allocated_entries,
        ),
        spectral_state_feature_values: C51Measurement::Measured(diagnostics.state_feature_entries),
        spectral_observation_feature_values: C51Measurement::Measured(
            diagnostics.observation_feature_entries,
        ),
        spectral_sampling_normal_dimension: C51Measurement::Measured(
            diagnostics.sampling_standard_normal_dimension,
        ),
        spectral_rank_fraction: C51Measurement::Measured(
            rank as f64 / diagnostics.ambient_dimension as f64,
        ),
        spectral_shared_maximum_basis_rank: C51Measurement::NotRecorded,
        spectral_partial_requested_rank: C51Measurement::NotRecorded,
        spectral_partial_converged_rank: C51Measurement::NotRecorded,
        spectral_partial_retained_rank: C51Measurement::NotRecorded,
        spectral_partial_l_nnz: C51Measurement::NotRecorded,
        spectral_partial_d_nnz: C51Measurement::NotRecorded,
        spectral_partial_c_nnz: C51Measurement::NotRecorded,
        spectral_partial_mkm1_nnz: C51Measurement::NotRecorded,
        spectral_partial_mk_nnz: C51Measurement::NotRecorded,
        spectral_partial_mass_nnz: C51Measurement::NotRecorded,
        spectral_partial_weak_hodge_laplacian_nnz: C51Measurement::NotRecorded,
        spectral_partial_input_nnz: C51Measurement::NotRecorded,
        spectral_partial_basis_storage_bytes: C51Measurement::NotRecorded,
        spectral_partial_clamped_negative_eigenvalues: C51Measurement::NotRecorded,
        spectral_partial_maximum_relative_mixed_residual: C51Measurement::NotRecorded,
        spectral_partial_maximum_mass_orthogonality_defect: C51Measurement::NotRecorded,
        spectral_partial_mass_orthogonality_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigenvalue_negative_tolerance: C51Measurement::NotRecorded,
        spectral_partial_eigensolver_tolerance: C51Measurement::NotRecorded,
        spectral_partial_residual_validation_ns: C51Measurement::NotRecorded,
        solver: match diagnostics.conditioning_strategy {
            LowRankConditioningStrategy::ObservationSpace => {
                "spectral_observation_space_dense_cholesky"
            }
            LowRankConditioningStrategy::LatentSpace => "spectral_latent_space_dense_cholesky",
        }
        .to_string(),
        ordering: "not_applicable".to_string(),
        feec_assembly_ns: duration_measurement(feec_assembly),
        prior_build_ns: duration_measurement(
            source.operator_recovery + source.eigensolve + gp_build,
        ),
        exact_gp_covariance_construction_ns: C51Measurement::NotRecorded,
        exact_gp_dense_conditioning_ns: C51Measurement::NotRecorded,
        exact_gp_end_to_end_setup_ns: C51Measurement::NotRecorded,
        spectral_operator_recovery_ns: duration_measurement(source.operator_recovery),
        spectral_eigensolve_ns: duration_measurement(source.eigensolve),
        spectral_gp_build_ns: duration_measurement(gp_build),
        observation_update_assembly_ns: C51Measurement::unavailable(
            "prepared spectral API exposes combined feature projection and dense factor setup",
        ),
        observation_update_and_factor_ns: duration_measurement(setup_elapsed),
        symbolic_analysis_ns: C51Measurement::unavailable(
            "symbolic sparse analysis is not applicable to dense spectral conditioning",
        ),
        numeric_factor_ns: C51Measurement::unavailable(
            "prepared spectral API exposes combined dense setup time",
        ),
        combined_factor_setup_ns: duration_measurement(setup_elapsed),
        repeated_data_ns: duration_measurement(repeated_elapsed),
        covariance_action_ns: duration_measurement(covariance_elapsed),
        physical_qoi_ns: duration_measurement(physical_elapsed),
        sampling_ns: duration_measurement(sampling_elapsed),
        variance_estimation_ns: duration_measurement(variance_elapsed),
        factor_external_peak_rss_bytes: C51Measurement::NotRecorded,
        query_external_peak_rss_bytes: C51Measurement::NotRecorded,
        posterior_mean_relative_error: errors.posterior_mean,
        covariance_action_relative_error: errors.covariance_action,
        physical_covariance_action_relative_error: errors.physical_covariance_action,
        physical_output_covariance_action_relative_error: errors.physical_output_covariance_action,
        physical_variance_relative_error: errors.physical_variance,
        physical_transformed_variance_relative_error: errors.physical_transformed_variance,
        physical_mean_relative_error: errors.physical_mean,
        physical_transformed_variance_min_raw: C51Measurement::Measured(minimum_raw(
            &answers.physical_transformed_variances,
        )),
        physical_transformed_variance_negative_count: C51Measurement::Measured(negative_count(
            &answers.physical_transformed_variances,
        )),
    };
    apply_spectral_source_metadata(&mut row, source, rank);
    Ok(row)
}

fn observation_designs(
    config: &C51BenchmarkConfig,
    topology: &Complex,
    coordinates: &manifold::geometry::coord::mesh::MeshCoords,
    elimination: &EssentialBoundaryElimination,
    dimension: usize,
    legacy_full_physical_observation: Option<&LinearMap>,
    direct_flux_observation: Option<&LinearMap>,
) -> Result<Vec<ObservationDesign>> {
    let mut designs = Vec::new();
    for requested in local_sensor_targets(config, dimension) {
        let sensor_count = requested.min(dimension);
        match config.physical_observation_design {
            C51PhysicalObservationDesign::LegacyCellRows => {
                let legacy_full_physical_observation = legacy_full_physical_observation
                    .ok_or_else(|| {
                        FeecGmrfError::Inference(
                            "legacy C51 observation design is missing its cellwise B map"
                                .to_string(),
                        )
                    })?;
                let physical_observation = elimination
                    .eliminate_map(
                        legacy_full_physical_observation,
                        &vec![0.0; legacy_full_physical_observation.output_dimension()],
                    )?
                    .reduced()
                    .clone();
                let physical_sensor_count = requested.min(physical_observation.output_dimension());
                let physical_rows = evenly_spaced_indices(
                    physical_observation.output_dimension(),
                    physical_sensor_count,
                );
                designs.push(ObservationDesign {
                    class: C51ObservationClass::LocalPhysicalB,
                    operator: physical_observation.select_outputs(&physical_rows)?,
                    sensor_count: physical_sensor_count,
                    design_id: C51PhysicalObservationDesign::LegacyCellRows
                        .as_str()
                        .to_string(),
                    fixed_radius: C51Measurement::unavailable(
                        "legacy cell-row footprint shrinks with the mesh",
                    ),
                    vector_center_count: C51Measurement::NotRecorded,
                    component_counts: std::array::from_fn(|_| C51Measurement::NotRecorded),
                });
            }
            C51PhysicalObservationDesign::JuqFixedFootprintPrefix => {
                let prefix =
                    assemble_juq_performance_observation_prefix(topology, coordinates, requested)?;
                let eliminated = elimination.eliminate_map(
                    prefix.scalar_map(),
                    &vec![0.0; prefix.scalar_map().output_dimension()],
                )?;
                let metadata = prefix.metadata();
                designs.push(ObservationDesign {
                    class: C51ObservationClass::LocalPhysicalB,
                    operator: eliminated.reduced().clone(),
                    sensor_count: metadata.scalar_observation_count,
                    design_id: C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID.to_string(),
                    fixed_radius: C51Measurement::Measured(C51_FIXED_OBSERVATION_RADIUS),
                    vector_center_count: C51Measurement::Measured(metadata.vector_center_count),
                    component_counts: metadata.component_counts.map(C51Measurement::Measured),
                });
            }
            C51PhysicalObservationDesign::FluxCochainHashedPrefix => {
                let flux = direct_flux_observation.ok_or_else(|| {
                    FeecGmrfError::Inference(
                        "C51 direct observation design is missing its active flux-cochain map"
                            .to_string(),
                    )
                })?;
                let split = assemble_c51_flux_cochain_split(
                    flux,
                    requested,
                    C51_FLUX_COCHAIN_SELECTION_COUNT,
                    C51_FLUX_COCHAIN_HELD_OUT_COUNT,
                )?;
                designs.push(ObservationDesign {
                    class: C51ObservationClass::LocalFluxCochain,
                    operator: split.observation().clone(),
                    sensor_count: requested,
                    design_id: C51_FLUX_COCHAIN_DESIGN_ID.to_string(),
                    fixed_radius: C51Measurement::unavailable(
                        "integral face-flux cochains have mesh-face support rather than a reconstruction kernel radius",
                    ),
                    vector_center_count: C51Measurement::NotRecorded,
                    component_counts: std::array::from_fn(|_| C51Measurement::NotRecorded),
                });
            }
        }
        if config.include_local_selectors {
            let indices = evenly_spaced_indices(dimension, sensor_count);
            designs.push(ObservationDesign {
                class: C51ObservationClass::LocalSelectors,
                operator: LinearMap::selector(dimension, &indices)?,
                sensor_count,
                design_id: "coefficient_selectors".to_string(),
                fixed_radius: C51Measurement::unavailable(
                    "coefficient selectors have no fixed physical radius",
                ),
                vector_center_count: C51Measurement::NotRecorded,
                component_counts: std::array::from_fn(|_| C51Measurement::NotRecorded),
            });
        }
    }
    if config.include_global_row {
        let scale = 1.0 / (dimension as f64).sqrt();
        let entries = (0..dimension)
            .map(|index| {
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                (index, sign * scale)
            })
            .collect::<Vec<_>>();
        designs.push(ObservationDesign {
            class: C51ObservationClass::GlobalRow,
            operator: LinearMap::weighted_row(dimension, &entries)?,
            sensor_count: 1,
            design_id: "dense_alternating_global_row".to_string(),
            fixed_radius: C51Measurement::unavailable(
                "global control row has no bounded physical radius",
            ),
            vector_center_count: C51Measurement::NotRecorded,
            component_counts: std::array::from_fn(|_| C51Measurement::NotRecorded),
        });
    }
    Ok(designs)
}

fn local_sensor_targets(config: &C51BenchmarkConfig, dimension: usize) -> Vec<usize> {
    let mut targets = config
        .local_sensor_counts
        .iter()
        .map(|count| match config.physical_observation_design {
            C51PhysicalObservationDesign::LegacyCellRows => (*count).min(dimension),
            C51PhysicalObservationDesign::JuqFixedFootprintPrefix
            | C51PhysicalObservationDesign::FluxCochainHashedPrefix => *count,
        })
        .collect::<BTreeSet<_>>();
    for density in &config.local_sensor_density_targets {
        let count = (*density * dimension as f64).ceil() as usize;
        targets.insert(count.clamp(1, dimension));
    }
    targets.into_iter().collect()
}

fn physical_qoi_region_cells(
    topology: &Complex,
    coordinates: &manifold::geometry::coord::mesh::MeshCoords,
    lower: [f64; 3],
    upper: [f64; 3],
) -> Result<(Vec<usize>, f64)> {
    let mut selected = Vec::new();
    let mut selected_volume = 0.0;
    let mut total_volume = 0.0;
    for cell in topology.cells().handle_iter() {
        let simplex = cell.coord_simplex(coordinates);
        let volume = simplex.vol();
        let barycenter = simplex.barycenter();
        if !volume.is_finite() || volume <= 0.0 {
            return Err(FeecGmrfError::Assembly(
                "C51 mesh contains a non-positive or non-finite cell volume".to_string(),
            ));
        }
        total_volume += volume;
        if (0..3).all(|axis| barycenter[axis] >= lower[axis] && barycenter[axis] <= upper[axis]) {
            selected.push(cell.kidx());
            selected_volume += volume;
        }
    }
    if selected.is_empty() || !total_volume.is_finite() || total_volume <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "C51 physical-QoI region selects no cells or has invalid total volume".to_string(),
        ));
    }
    Ok((selected, selected_volume / total_volume))
}

fn evenly_spaced_indices(dimension: usize, count: usize) -> Vec<usize> {
    if count == 1 {
        return vec![dimension / 2];
    }
    (0..count)
        .map(|index| index * (dimension - 1) / (count - 1))
        .collect()
}

fn repeated_observation_values(count: usize, repeats: usize, seed: u64) -> Vec<Vec<f64>> {
    (0..repeats)
        .map(|repeat| {
            (0..count)
                .map(|index| {
                    let coordinate = (index + 1) as f64;
                    let phase = (repeat + 1) as f64 + (seed % 997) as f64 / 997.0;
                    (0.37 * coordinate + 0.11 * phase).sin()
                })
                .collect()
        })
        .collect()
}

fn deterministic_probe(dimension: usize, seed: u64) -> Vec<f64> {
    (0..dimension)
        .map(|index| {
            let coordinate = index as f64 + 1.0;
            let phase = (seed % 991) as f64 / 991.0;
            (0.73 * coordinate + phase).sin() + (0.19 * coordinate - phase).cos()
        })
        .collect()
}

fn physical_qoi_probe(dimension: usize, seed: u64) -> Vec<f64> {
    if dimension == 3 {
        vec![1.0, -0.5, 0.25]
    } else {
        let mut probe = deterministic_probe(dimension, seed ^ 0xB0_0A71);
        let norm = probe.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm > 0.0 {
            for value in &mut probe {
                *value /= norm;
            }
        }
        probe
    }
}

fn errors_against(
    answers: &ReferenceAnswers,
    reference: Option<&ReferenceAnswers>,
) -> Result<AccuracyMeasurements> {
    let Some(reference) = reference else {
        return Ok(AccuracyMeasurements {
            posterior_mean: C51Measurement::Measured(0.0),
            covariance_action: C51Measurement::Measured(0.0),
            physical_covariance_action: C51Measurement::Measured(0.0),
            physical_output_covariance_action: C51Measurement::Measured(0.0),
            physical_variance: C51Measurement::Measured(0.0),
            physical_transformed_variance: C51Measurement::Measured(0.0),
            physical_mean: C51Measurement::Measured(0.0),
        });
    };
    Ok(AccuracyMeasurements {
        posterior_mean: C51Measurement::Measured(relative_l2(&answers.mean, &reference.mean)?),
        covariance_action: C51Measurement::Measured(relative_l2(
            &answers.covariance_action,
            &reference.covariance_action,
        )?),
        physical_covariance_action: C51Measurement::Measured(relative_l2(
            &answers.physical_covariance_action,
            &reference.physical_covariance_action,
        )?),
        physical_output_covariance_action: C51Measurement::Measured(relative_l2(
            &answers.physical_output_covariance_action,
            &reference.physical_output_covariance_action,
        )?),
        physical_variance: C51Measurement::Measured(relative_scalar(
            answers.physical_variance,
            reference.physical_variance,
        )),
        physical_transformed_variance: C51Measurement::Measured(relative_l2(
            &answers.physical_transformed_variances,
            &reference.physical_transformed_variances,
        )?),
        physical_mean: C51Measurement::Measured(relative_l2(
            &answers.physical_mean,
            &reference.physical_mean,
        )?),
    })
}

fn physical_covariance_accuracy_qualified(errors: &AccuracyMeasurements) -> bool {
    [
        &errors.physical_output_covariance_action,
        &errors.physical_variance,
        &errors.physical_transformed_variance,
    ]
    .into_iter()
    .all(|measurement| {
        matches!(measurement, C51Measurement::Measured(value)
            if value.is_finite()
                && *value <= C51_CLAIM_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT)
    })
}

fn relative_l2(values: &[f64], reference: &[f64]) -> Result<f64> {
    if values.len() != reference.len() {
        return Err(FeecGmrfError::Dimension(
            "C51 comparison vectors have different lengths".to_string(),
        ));
    }
    let numerator = values
        .iter()
        .zip(reference)
        .map(|(value, reference)| (value - reference).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    Ok(numerator / denominator.max(f64::MIN_POSITIVE))
}

fn relative_scalar(value: f64, reference: f64) -> f64 {
    (value - reference).abs() / reference.abs().max(f64::MIN_POSITIVE)
}

fn minimum_raw(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn negative_count(values: &[f64]) -> usize {
    values.iter().filter(|value| **value < 0.0).count()
}

fn dense_rows_times_vector(matrix: &[Vec<f64>], vector: &[f64], label: &str) -> Result<Vec<f64>> {
    if matrix.len() != vector.len() || matrix.iter().any(|row| row.len() != vector.len()) {
        return Err(FeecGmrfError::Dimension(format!(
            "{label} must be square with dimension {}, got {} rows",
            vector.len(),
            matrix.len()
        )));
    }
    Ok(matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(entry, value)| entry * value)
                .sum()
        })
        .collect())
}

fn dense_rows_diagonal(matrix: &[Vec<f64>], label: &str) -> Result<Vec<f64>> {
    if matrix.iter().any(|row| row.len() != matrix.len()) {
        return Err(FeecGmrfError::Dimension(format!(
            "{label} must be square, got {} rows with inconsistent columns",
            matrix.len()
        )));
    }
    Ok(matrix
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .collect())
}

fn dot(lhs: &[f64], rhs: &[f64]) -> Result<f64> {
    if lhs.len() != rhs.len() {
        return Err(FeecGmrfError::Dimension(
            "C51 functional and covariance action have different lengths".to_string(),
        ));
    }
    Ok(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs * rhs).sum())
}

fn ordering_name(ordering: CholeskyOrderingKind) -> &'static str {
    match ordering {
        CholeskyOrderingKind::ApproximateMinimumDegree => "amd",
        CholeskyOrderingKind::Identity => "identity",
        CholeskyOrderingKind::Custom => "custom",
    }
}

fn method_seed(method: C51MethodId) -> u64 {
    match method {
        C51MethodId::FullMetricDirect => 0xD1CE_C700,
        C51MethodId::FullMetricAugmented => 0xA090_0000,
        C51MethodId::WhitneyCompact => 0xC0A0_AC70,
        C51MethodId::WhitneyCompactCalibrated => 0xCA1B_AA7E,
        C51MethodId::WhitneyCompactIntegralFluxCalibrated => 0xF10C_CA1B,
        C51MethodId::ExactObservationSpaceGp => 0xEAC7_600D,
        C51MethodId::SpectralFullRank => 0x5EEC_F011,
        C51MethodId::SpectralLowRank => 0x5EEC_10A0,
    }
}

fn spectral_error(error: impl Display) -> FeecGmrfError {
    FeecGmrfError::Inference(format!("C51 spectral reference failed: {error}"))
}

fn measured_or_infinite(measurement: &C51Measurement<f64>) -> f64 {
    match measurement {
        C51Measurement::Measured(value) => *value,
        C51Measurement::NotRecorded | C51Measurement::Unavailable { .. } => f64::INFINITY,
    }
}

fn full_rank_maximum_error(row: &C51BenchmarkRow) -> f64 {
    [
        &row.posterior_mean_relative_error,
        &row.covariance_action_relative_error,
        &row.physical_covariance_action_relative_error,
        &row.physical_output_covariance_action_relative_error,
        &row.physical_variance_relative_error,
        &row.physical_transformed_variance_relative_error,
        &row.physical_mean_relative_error,
    ]
    .into_iter()
    .map(measured_or_infinite)
    .fold(0.0, f64::max)
}

fn exact_gp_maximum_physical_error(row: &C51BenchmarkRow) -> f64 {
    [
        &row.physical_output_covariance_action_relative_error,
        &row.physical_variance_relative_error,
        &row.physical_transformed_variance_relative_error,
        &row.physical_mean_relative_error,
    ]
    .into_iter()
    .map(measured_or_infinite)
    .fold(0.0, f64::max)
}

fn duration_measurement(duration: Duration) -> C51Measurement<u128> {
    C51Measurement::Measured(duration.as_nanos())
}

fn duration_resource_measurement(
    measurement: &ResourceMeasurement<Duration>,
) -> C51Measurement<u128> {
    match measurement {
        ResourceMeasurement::Measured(duration) => duration_measurement(*duration),
        ResourceMeasurement::NotRecorded => C51Measurement::NotRecorded,
        ResourceMeasurement::Unavailable(reason) => C51Measurement::unavailable(*reason),
    }
}

fn usize_resource_measurement(measurement: &ResourceMeasurement<usize>) -> C51Measurement<usize> {
    match measurement {
        ResourceMeasurement::Measured(value) => C51Measurement::Measured(*value),
        ResourceMeasurement::NotRecorded => C51Measurement::NotRecorded,
        ResourceMeasurement::Unavailable(reason) => C51Measurement::unavailable(*reason),
    }
}

fn csv_header() -> String {
    let scalar_fields = [
        "schema_version",
        "study_id",
        "method",
        "available",
        "unavailable_reason",
        "discrete_law_id",
        "same_discrete_law_eligible",
        "matched_physical_qoi_eligible",
        "common_physical_uq_workload_eligible",
        "full_uq_workload_eligible",
        "method_performance_timing_eligible",
        "method_performance_timing_ineligibility_reason",
        "spectral_basis_source",
        "spectral_eigensolver_backend",
        "spectral_helper_workspace",
        "spectral_helper_binary",
        "spectral_feec_assembly_reuse",
        "spectral_partial_which",
        "spectral_partial_mass_solve",
        "refinement",
        "spatial_dimension",
        "form_degree",
        "alpha",
        "kappa",
        "tau",
        "amplitude_calibration_policy",
        "observation_class",
        "observation_design_id",
        "physical_qoi_design_id",
        "physical_qoi_output_count",
        "physical_qoi_selected_cell_count",
        "physical_qoi_selected_volume_fraction",
        "sensor_count",
        "sensor_density",
        "repeated_data_count",
        "posterior_sample_count",
        "seed",
        "latent_dimension",
        "ambient_dimension",
        "constraint_dimension",
        "observation_nnz",
        "solver",
        "ordering",
        "exact_gp_prior_covariance_audit_policy",
        "exact_gp_prior_covariance_audit_status",
        "exact_gp_prior_covariance_audit_residual_status",
    ];
    let measurement_fields = measurement_field_names();
    scalar_fields
        .into_iter()
        .map(str::to_string)
        .chain(measurement_fields.into_iter().flat_map(|name| {
            [
                format!("{name}_status"),
                format!("{name}_value"),
                format!("{name}_reason"),
            ]
        }))
        .collect::<Vec<_>>()
        .join(",")
}

fn measurement_field_names() -> [&'static str; 87] {
    [
        "observation_fixed_radius",
        "observation_vector_center_count",
        "observation_component_x_count",
        "observation_component_y_count",
        "observation_component_z_count",
        "amplitude_calibration_target_prior_rms",
        "amplitude_calibration_uncalibrated_prior_rms",
        "amplitude_calibration_precision_scale",
        "amplitude_calibration_tau_multiplier",
        "amplitude_calibration_ns",
        "prior_precision_nnz",
        "observation_update_nnz",
        "posterior_precision_nnz",
        "ambient_energy_nnz",
        "constraint_nnz",
        "kkt_input_nnz",
        "factor_storage_nnz",
        "certificate_factor_storage_nnz",
        "fill_ratio",
        "exact_gp_dense_factor_dimension",
        "exact_gp_dense_factor_allocated_values",
        "exact_gp_dense_factor_triangular_values",
        "exact_gp_output_observation_covariance_values",
        "exact_gp_posterior_output_covariance_values",
        "exact_gp_persistent_storage_values",
        "exact_gp_posterior_covariance_max_abs_skew",
        "exact_gp_prior_covariance_raw_max_relative_skew",
        "exact_gp_prior_covariance_projection_frobenius_correction",
        "exact_gp_prior_covariance_audit_residual_aware",
        "spectral_rank",
        "spectral_basis_storage_values",
        "spectral_gp_storage_values",
        "spectral_prepared_storage_values",
        "spectral_dense_factor_values",
        "spectral_state_feature_values",
        "spectral_observation_feature_values",
        "spectral_sampling_normal_dimension",
        "spectral_rank_fraction",
        "spectral_shared_maximum_basis_rank",
        "spectral_partial_requested_rank",
        "spectral_partial_converged_rank",
        "spectral_partial_retained_rank",
        "spectral_partial_l_nnz",
        "spectral_partial_d_nnz",
        "spectral_partial_c_nnz",
        "spectral_partial_mkm1_nnz",
        "spectral_partial_mk_nnz",
        "spectral_partial_mass_nnz",
        "spectral_partial_weak_hodge_laplacian_nnz",
        "spectral_partial_input_nnz",
        "spectral_partial_basis_storage_bytes",
        "spectral_partial_clamped_negative_eigenvalues",
        "spectral_partial_maximum_relative_mixed_residual",
        "spectral_partial_maximum_mass_orthogonality_defect",
        "spectral_partial_mass_orthogonality_tolerance",
        "spectral_partial_eigenvalue_negative_tolerance",
        "spectral_partial_eigensolver_tolerance",
        "spectral_partial_residual_validation_ns",
        "feec_assembly_ns",
        "prior_build_ns",
        "exact_gp_covariance_construction_ns",
        "exact_gp_dense_conditioning_ns",
        "exact_gp_end_to_end_setup_ns",
        "spectral_operator_recovery_ns",
        "spectral_eigensolve_ns",
        "spectral_gp_build_ns",
        "observation_update_assembly_ns",
        "observation_update_and_factor_ns",
        "symbolic_analysis_ns",
        "numeric_factor_ns",
        "combined_factor_setup_ns",
        "repeated_data_ns",
        "covariance_action_ns",
        "physical_qoi_ns",
        "sampling_ns",
        "variance_estimation_ns",
        "factor_external_peak_rss_bytes",
        "query_external_peak_rss_bytes",
        "posterior_mean_relative_error",
        "covariance_action_relative_error",
        "physical_covariance_action_relative_error",
        "physical_output_covariance_action_relative_error",
        "physical_variance_relative_error",
        "physical_transformed_variance_relative_error",
        "physical_mean_relative_error",
        "physical_transformed_variance_min_raw",
        "physical_transformed_variance_negative_count",
    ]
}

fn row_to_csv(row: &C51BenchmarkRow) -> String {
    let mut fields = vec![
        C51_SCHEMA_VERSION.to_string(),
        C51_STUDY_ID.to_string(),
        row.method.as_str().to_string(),
        row.available.to_string(),
        row.unavailable_reason.clone(),
        row.discrete_law_id.clone(),
        row.same_discrete_law_eligible.to_string(),
        row.matched_physical_qoi_eligible.to_string(),
        row.common_physical_uq_workload_eligible.to_string(),
        row.full_uq_workload_eligible.to_string(),
        row.method_performance_timing_eligible.to_string(),
        row.method_performance_timing_ineligibility_reason.clone(),
        row.spectral_basis_source.clone(),
        row.spectral_eigensolver_backend.clone(),
        row.spectral_helper_workspace.clone(),
        row.spectral_helper_binary.clone(),
        row.spectral_feec_assembly_reuse.clone(),
        row.spectral_partial_which.clone(),
        row.spectral_partial_mass_solve.clone(),
        row.refinement.to_string(),
        row.spatial_dimension.to_string(),
        row.form_degree.to_string(),
        row.alpha.to_string(),
        row.kappa.to_string(),
        row.tau.to_string(),
        row.amplitude_calibration_policy.clone(),
        row.observation_class.as_str().to_string(),
        row.observation_design_id.clone(),
        row.physical_qoi_design_id.clone(),
        row.physical_qoi_output_count.to_string(),
        row.physical_qoi_selected_cell_count.to_string(),
        row.physical_qoi_selected_volume_fraction.to_string(),
        row.sensor_count.to_string(),
        row.sensor_density.to_string(),
        row.repeated_data_count.to_string(),
        row.posterior_sample_count.to_string(),
        row.seed.to_string(),
        row.latent_dimension.to_string(),
        row.ambient_dimension.to_string(),
        row.constraint_dimension.to_string(),
        row.observation_nnz.to_string(),
        row.solver.clone(),
        row.ordering.clone(),
        row.exact_gp_prior_covariance_audit_policy.clone(),
        row.exact_gp_prior_covariance_audit_status.clone(),
        row.exact_gp_prior_covariance_audit_residual_status.clone(),
    ];
    push_measurement(&mut fields, &row.observation_fixed_radius);
    push_measurement(&mut fields, &row.observation_vector_center_count);
    push_measurement(&mut fields, &row.observation_component_x_count);
    push_measurement(&mut fields, &row.observation_component_y_count);
    push_measurement(&mut fields, &row.observation_component_z_count);
    push_measurement(&mut fields, &row.amplitude_calibration_target_prior_rms);
    push_measurement(
        &mut fields,
        &row.amplitude_calibration_uncalibrated_prior_rms,
    );
    push_measurement(&mut fields, &row.amplitude_calibration_precision_scale);
    push_measurement(&mut fields, &row.amplitude_calibration_tau_multiplier);
    push_measurement(&mut fields, &row.amplitude_calibration_ns);
    push_measurement(&mut fields, &row.prior_precision_nnz);
    push_measurement(&mut fields, &row.observation_update_nnz);
    push_measurement(&mut fields, &row.posterior_precision_nnz);
    push_measurement(&mut fields, &row.ambient_energy_nnz);
    push_measurement(&mut fields, &row.constraint_nnz);
    push_measurement(&mut fields, &row.kkt_input_nnz);
    push_measurement(&mut fields, &row.factor_storage_nnz);
    push_measurement(&mut fields, &row.certificate_factor_storage_nnz);
    push_measurement(&mut fields, &row.fill_ratio);
    push_measurement(&mut fields, &row.exact_gp_dense_factor_dimension);
    push_measurement(&mut fields, &row.exact_gp_dense_factor_allocated_values);
    push_measurement(&mut fields, &row.exact_gp_dense_factor_triangular_values);
    push_measurement(
        &mut fields,
        &row.exact_gp_output_observation_covariance_values,
    );
    push_measurement(
        &mut fields,
        &row.exact_gp_posterior_output_covariance_values,
    );
    push_measurement(&mut fields, &row.exact_gp_persistent_storage_values);
    push_measurement(&mut fields, &row.exact_gp_posterior_covariance_max_abs_skew);
    push_measurement(
        &mut fields,
        &row.exact_gp_prior_covariance_raw_max_relative_skew,
    );
    push_measurement(
        &mut fields,
        &row.exact_gp_prior_covariance_projection_frobenius_correction,
    );
    push_measurement(
        &mut fields,
        &row.exact_gp_prior_covariance_audit_residual_aware,
    );
    push_measurement(&mut fields, &row.spectral_rank);
    push_measurement(&mut fields, &row.spectral_basis_storage_values);
    push_measurement(&mut fields, &row.spectral_gp_storage_values);
    push_measurement(&mut fields, &row.spectral_prepared_storage_values);
    push_measurement(&mut fields, &row.spectral_dense_factor_values);
    push_measurement(&mut fields, &row.spectral_state_feature_values);
    push_measurement(&mut fields, &row.spectral_observation_feature_values);
    push_measurement(&mut fields, &row.spectral_sampling_normal_dimension);
    push_measurement(&mut fields, &row.spectral_rank_fraction);
    push_measurement(&mut fields, &row.spectral_shared_maximum_basis_rank);
    push_measurement(&mut fields, &row.spectral_partial_requested_rank);
    push_measurement(&mut fields, &row.spectral_partial_converged_rank);
    push_measurement(&mut fields, &row.spectral_partial_retained_rank);
    push_measurement(&mut fields, &row.spectral_partial_l_nnz);
    push_measurement(&mut fields, &row.spectral_partial_d_nnz);
    push_measurement(&mut fields, &row.spectral_partial_c_nnz);
    push_measurement(&mut fields, &row.spectral_partial_mkm1_nnz);
    push_measurement(&mut fields, &row.spectral_partial_mk_nnz);
    push_measurement(&mut fields, &row.spectral_partial_mass_nnz);
    push_measurement(&mut fields, &row.spectral_partial_weak_hodge_laplacian_nnz);
    push_measurement(&mut fields, &row.spectral_partial_input_nnz);
    push_measurement(&mut fields, &row.spectral_partial_basis_storage_bytes);
    push_measurement(
        &mut fields,
        &row.spectral_partial_clamped_negative_eigenvalues,
    );
    push_measurement(
        &mut fields,
        &row.spectral_partial_maximum_relative_mixed_residual,
    );
    push_measurement(
        &mut fields,
        &row.spectral_partial_maximum_mass_orthogonality_defect,
    );
    push_measurement(
        &mut fields,
        &row.spectral_partial_mass_orthogonality_tolerance,
    );
    push_measurement(
        &mut fields,
        &row.spectral_partial_eigenvalue_negative_tolerance,
    );
    push_measurement(&mut fields, &row.spectral_partial_eigensolver_tolerance);
    push_measurement(&mut fields, &row.spectral_partial_residual_validation_ns);
    push_measurement(&mut fields, &row.feec_assembly_ns);
    push_measurement(&mut fields, &row.prior_build_ns);
    push_measurement(&mut fields, &row.exact_gp_covariance_construction_ns);
    push_measurement(&mut fields, &row.exact_gp_dense_conditioning_ns);
    push_measurement(&mut fields, &row.exact_gp_end_to_end_setup_ns);
    push_measurement(&mut fields, &row.spectral_operator_recovery_ns);
    push_measurement(&mut fields, &row.spectral_eigensolve_ns);
    push_measurement(&mut fields, &row.spectral_gp_build_ns);
    push_measurement(&mut fields, &row.observation_update_assembly_ns);
    push_measurement(&mut fields, &row.observation_update_and_factor_ns);
    push_measurement(&mut fields, &row.symbolic_analysis_ns);
    push_measurement(&mut fields, &row.numeric_factor_ns);
    push_measurement(&mut fields, &row.combined_factor_setup_ns);
    push_measurement(&mut fields, &row.repeated_data_ns);
    push_measurement(&mut fields, &row.covariance_action_ns);
    push_measurement(&mut fields, &row.physical_qoi_ns);
    push_measurement(&mut fields, &row.sampling_ns);
    push_measurement(&mut fields, &row.variance_estimation_ns);
    push_measurement(&mut fields, &row.factor_external_peak_rss_bytes);
    push_measurement(&mut fields, &row.query_external_peak_rss_bytes);
    push_measurement(&mut fields, &row.posterior_mean_relative_error);
    push_measurement(&mut fields, &row.covariance_action_relative_error);
    push_measurement(&mut fields, &row.physical_covariance_action_relative_error);
    push_measurement(
        &mut fields,
        &row.physical_output_covariance_action_relative_error,
    );
    push_measurement(&mut fields, &row.physical_variance_relative_error);
    push_measurement(
        &mut fields,
        &row.physical_transformed_variance_relative_error,
    );
    push_measurement(&mut fields, &row.physical_mean_relative_error);
    push_measurement(&mut fields, &row.physical_transformed_variance_min_raw);
    push_measurement(
        &mut fields,
        &row.physical_transformed_variance_negative_count,
    );
    fields
        .into_iter()
        .map(|field| csv_escape(&field))
        .collect::<Vec<_>>()
        .join(",")
}

fn push_measurement<T: Display>(fields: &mut Vec<String>, measurement: &C51Measurement<T>) {
    fields.push(measurement.status().to_string());
    fields.push(measurement.value_string());
    fields.push(measurement.reason().to_string());
}

fn csv_escape(value: &str) -> String {
    if value
        .chars()
        .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn yaml_scalar(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn yaml_usize_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn yaml_f64_list(values: &[f64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn write_yaml_option(writer: &mut impl Write, key: &str, value: Option<&str>) -> Result<()> {
    match value {
        Some(value) => writeln!(writer, "{key}: {}", yaml_scalar(value))?,
        None => writeln!(writer, "{key}: null")?,
    }
    Ok(())
}

fn write_yaml_bool_option(writer: &mut impl Write, key: &str, value: Option<bool>) -> Result<()> {
    match value {
        Some(value) => writeln!(writer, "{key}: {value}")?,
        None => writeln!(writer, "{key}: null")?,
    }
    Ok(())
}

fn write_yaml_nested_option(writer: &mut impl Write, key: &str, value: Option<&str>) -> Result<()> {
    match value {
        Some(value) => writeln!(writer, "  {key}: {}", yaml_scalar(value))?,
        None => writeln!(writer, "  {key}: null")?,
    }
    Ok(())
}

fn write_yaml_nested_usize_option(
    writer: &mut impl Write,
    key: &str,
    value: Option<usize>,
) -> Result<()> {
    match value {
        Some(value) => writeln!(writer, "  {key}: {value}")?,
        None => writeln!(writer, "  {key}: null")?,
    }
    Ok(())
}

fn write_yaml_nested_u64_option(
    writer: &mut impl Write,
    key: &str,
    value: Option<u64>,
) -> Result<()> {
    match value {
        Some(value) => writeln!(writer, "  {key}: {value}")?,
        None => writeln!(writer, "  {key}: null")?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_archive_covers_methods_locality_and_amortization() {
        let config = C51BenchmarkConfig::smoke();
        let archive = run_c51_benchmark(&config, C51RunMetadata::smoke("test"))
            .expect("C51 smoke benchmark should run");
        let dimension = archive.rows[0].ambient_dimension;
        let local_class_count = 1 + usize::from(config.include_local_selectors);
        let designs = local_class_count * local_sensor_targets(&config, dimension).len()
            + usize::from(config.include_global_row);
        let method_count = 4 + spectral_low_ranks(&config, dimension).len();
        let exact_gp_rows =
            local_sensor_targets(&config, dimension).len() * config.repeated_data_counts.len();
        assert_eq!(
            archive.rows.len(),
            designs * config.repeated_data_counts.len() * method_count + exact_gp_rows
        );
        assert!(archive.rows.iter().any(|row| {
            row.observation_class == C51ObservationClass::LocalPhysicalB
                && row.repeated_data_count == 2
        }));
        assert!(archive
            .rows
            .iter()
            .any(|row| row.observation_class == C51ObservationClass::LocalSelectors));
        assert!(archive
            .rows
            .iter()
            .any(|row| row.observation_class == C51ObservationClass::GlobalRow));
        for row in &archive.rows {
            match row.method {
                C51MethodId::FullMetricDirect => {
                    assert!(row.same_discrete_law_eligible);
                    assert!(row.full_uq_workload_eligible);
                    assert_eq!(
                        row.covariance_action_relative_error,
                        C51Measurement::Measured(0.0)
                    );
                    assert!(matches!(
                        row.factor_storage_nnz,
                        C51Measurement::Measured(_)
                    ));
                    assert!(matches!(row.sampling_ns, C51Measurement::Measured(_)));
                    assert!(matches!(
                        row.variance_estimation_ns,
                        C51Measurement::Measured(_)
                    ));
                    assert!(matches!(
                        row.observation_update_nnz,
                        C51Measurement::Measured(_)
                    ));
                    assert!(
                        measured_f64(&row.physical_transformed_variance_min_raw) > 1.0e-16,
                        "central-subdomain magnetic QoI must not collapse under boundary conditions"
                    );
                    assert!(row.physical_qoi_selected_cell_count > 0);
                    assert!(row.physical_qoi_selected_volume_fraction < 1.0);
                }
                C51MethodId::FullMetricAugmented => {
                    assert!(row.same_discrete_law_eligible);
                    assert!(!row.full_uq_workload_eligible);
                    assert!(matches!(
                        row.factor_storage_nnz,
                        C51Measurement::Unavailable { .. }
                    ));
                    assert!(measured_f64(&row.covariance_action_relative_error) <= 1.0e-9);
                    assert!(measured_f64(&row.physical_variance_relative_error) <= 1.0e-9);
                }
                C51MethodId::WhitneyCompact
                | C51MethodId::WhitneyCompactCalibrated
                | C51MethodId::WhitneyCompactIntegralFluxCalibrated => {
                    assert!(!row.same_discrete_law_eligible);
                    assert!(row.full_uq_workload_eligible);
                    assert!(measured_f64(&row.physical_variance_relative_error).is_finite());
                }
                C51MethodId::ExactObservationSpaceGp => {
                    assert!(row.same_discrete_law_eligible);
                    assert!(!row.full_uq_workload_eligible);
                    assert!(matches!(
                        row.factor_storage_nnz,
                        C51Measurement::Unavailable { .. }
                    ));
                    assert_eq!(
                        row.exact_gp_dense_factor_dimension,
                        C51Measurement::Measured(row.sensor_count)
                    );
                    assert!(exact_gp_maximum_physical_error(row) <= 1.0e-8);
                }
                C51MethodId::SpectralFullRank => {
                    assert!(row.available);
                    assert!(row.same_discrete_law_eligible);
                    assert!(row.full_uq_workload_eligible);
                    assert!(measured_f64(&row.covariance_action_relative_error) <= 1.0e-8);
                    assert!(measured_f64(&row.physical_variance_relative_error) <= 1.0e-8);
                    assert!(matches!(
                        row.spectral_prepared_storage_values,
                        C51Measurement::Measured(_)
                    ));
                }
                C51MethodId::SpectralLowRank => {
                    assert!(row.available);
                    assert!(!row.same_discrete_law_eligible);
                    assert!(row.full_uq_workload_eligible);
                    assert!(measured_f64(&row.physical_variance_relative_error).is_finite());
                }
            }
        }
    }

    #[test]
    fn smoke_writes_stable_csv_and_manifest_with_null_hardware() {
        let archive =
            run_c51_benchmark(&C51BenchmarkConfig::smoke(), C51RunMetadata::smoke("test")).unwrap();
        let stamp = std::process::id();
        let directory = std::env::temp_dir().join(format!("c51_benchmark_{stamp}"));
        if directory.exists() {
            fs::remove_dir_all(&directory).unwrap();
        }
        archive.write_directory(&directory).unwrap();
        let csv = fs::read_to_string(directory.join("rows.csv")).unwrap();
        let manifest = fs::read_to_string(directory.join("manifest.yaml")).unwrap();
        assert!(csv.starts_with("schema_version,study_id,method,available"));
        assert!(csv.contains("fm_direct"));
        assert!(csv.contains("spectral_full_rank,true"));
        assert!(csv.contains("spectral_low_rank,true"));
        assert!(manifest.contains("same_discrete_law_representation"));
        assert!(manifest.contains("matched_physical_qoi_method"));
        assert!(manifest.contains("cpu_model: null"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn reference_ladder_is_bounded_and_adds_full_rank_at_runtime() {
        let config = C51BenchmarkConfig::reference_ladder();
        config.validate().unwrap();
        assert_eq!(config.refinements, vec![2, 3, 4]);
        assert_eq!(config.repeated_data_counts, vec![1, 10, 100]);
        assert_eq!(config.local_sensor_counts, vec![32]);
        assert_eq!(config.local_sensor_density_targets, vec![0.02, 0.10]);
        assert!(!config.spectral_low_ranks.contains(&0));
        assert_eq!(spectral_low_ranks(&config, 16), vec![8]);
    }

    #[test]
    fn juq_performance_profile_freezes_independent_m_and_rank_study_outputs() {
        let config = C51BenchmarkConfig::juq_performance();
        config.validate().unwrap();
        assert_eq!(
            config.local_sensor_counts,
            vec![32, 128, 512, 1_024, 2_048, 4_096, 5_184]
        );
        assert!(config.local_sensor_density_targets.is_empty());
        assert_eq!(
            config.physical_observation_design,
            C51PhysicalObservationDesign::JuqFixedFootprintPrefix
        );
        assert_eq!(
            config.physical_output_design,
            C51PhysicalOutputDesign::RankStudySelectionAndHeldOut
        );
    }

    #[test]
    fn perf_zero_exact_gp_and_compact_share_frozen_physical_designs() {
        let mut config = C51BenchmarkConfig::juq_performance();
        // n=2 is too coarse for every frozen radius-0.16 kernel to intersect
        // the cell quadrature. n=4 is still bounded (N=316) and exercises both
        // independent nine-output banks without changing their footprints.
        config.refinements = vec![4];
        config.local_sensor_counts = vec![4];
        config.repeated_data_counts = vec![1];
        config.posterior_sample_count = 1;
        let archive = run_c51_juq_accuracy(&config, C51RunMetadata::smoke("perf-zero"))
            .expect("bounded PERF-0 accuracy route should run");
        assert_eq!(archive.rows.len(), 3);
        for row in &archive.rows {
            assert_eq!(
                row.observation_design_id,
                C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID
            );
            assert_eq!(row.physical_qoi_output_count, 18);
            assert_eq!(
                row.physical_qoi_design_id,
                C51PhysicalOutputDesign::RankStudySelectionAndHeldOut.as_str()
            );
            assert_eq!(
                row.observation_fixed_radius,
                C51Measurement::Measured(C51_FIXED_OBSERVATION_RADIUS)
            );
        }
        let exact_gp = archive
            .rows
            .iter()
            .find(|row| row.method == C51MethodId::ExactObservationSpaceGp)
            .unwrap();
        assert!(exact_gp_maximum_physical_error(exact_gp) <= 1.0e-8);
        assert_eq!(
            exact_gp.exact_gp_dense_factor_allocated_values,
            C51Measurement::Measured(16)
        );
        let compact = archive
            .rows
            .iter()
            .find(|row| row.method == C51MethodId::WhitneyCompact)
            .unwrap();
        let compact_error = measured_f64(&compact.physical_output_covariance_action_relative_error)
            .max(measured_f64(&compact.physical_variance_relative_error))
            .max(measured_f64(
                &compact.physical_transformed_variance_relative_error,
            ));
        assert_eq!(
            compact.matched_physical_qoi_eligible,
            compact_error <= C51_CLAIM_PHYSICAL_COVARIANCE_RELATIVE_ERROR_LIMIT
        );
        let directory =
            std::env::temp_dir().join(format!("c51-perf0-archive-{}", std::process::id()));
        if directory.exists() {
            fs::remove_dir_all(&directory).unwrap();
        }
        archive.write_directory(&directory).unwrap();
        crate::c51_spectral_covariance_rank::validate_c51_perf0_archive(&directory).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn spectral_rank_pilot_filters_and_deduplicates_fixed_and_fractional_ranks() {
        let config = C51BenchmarkConfig::spectral_rank_pilot();
        config.validate().unwrap();
        assert_eq!(config.refinements, vec![8, 10]);
        assert_eq!(config.repeated_data_counts, vec![1, 10]);
        assert!(!config.include_local_selectors);
        assert!(!config.include_global_row);
        let ranks = spectral_low_ranks(&config, 2_500);
        assert!(ranks.windows(2).all(|window| window[0] < window[1]));
        assert!(ranks.contains(&25));
        assert!(ranks.contains(&63));
        assert!(ranks.contains(&1_250));
        assert!(ranks.contains(&2_000));
        assert!(ranks.contains(&2_048));
        assert!(!ranks.contains(&2_500));
        assert!(!ranks.contains(&4_096));
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    fn spectral_rank_pilot_n10_plan_does_not_consult_marginal_materialization_cap() {
        let mut config = C51BenchmarkConfig::spectral_rank_pilot();
        // The current n=10 full-relative Whitney 1-form space has N=6130,
        // which exceeded the former exact-marginal materialization cap.
        let n10_dimension = 6_130;
        assert!(n10_dimension > config.spectral_operator_materialization_maximum_dimension);
        config.spectral_operator_materialization_maximum_dimension = 1;

        let (ranks, maximum_rank) = partial_spectral_rank_plan(&config, n10_dimension).unwrap();
        assert_eq!(maximum_rank, 4_096);
        assert!(ranks.contains(&2_000));
        assert!(ranks.contains(&2_048));
        assert!(ranks.contains(&4_096));
    }

    #[cfg(not(feature = "external-solvers"))]
    #[test]
    fn spectral_rank_pilot_fails_typed_without_external_solvers() {
        let error = run_c51_spectral_rank_pilot(
            &C51BenchmarkConfig::spectral_rank_pilot(),
            C51RunMetadata::smoke("test"),
        )
        .unwrap_err();
        assert!(matches!(error, FeecGmrfError::Unsupported(_)));
        assert!(error.to_string().contains("PETSc/SLEPc"));
    }

    #[test]
    fn compact_resource_process_produces_only_one_compact_setup_row() {
        let selection = C51ResourceSelection {
            refinement: 2,
            observation_class: C51ObservationClass::LocalPhysicalB,
            sensor_target: C51ResourceSensorTarget::Count(4),
            repeated_data_count: 1,
            method: C51MethodId::WhitneyCompact,
            spectral_rank: None,
            phase: C51ResourcePhase::ColdSetup,
        };
        let mut config = C51ResourceConfig::smoke(selection);
        // A spectral construction would fail this cap. The isolated compact
        // route neither assembles ExactFormOperators nor constructs a basis.
        config.spectral_reference_maximum_dimension = 1;
        let mut archive = run_c51_resource(&config, C51RunMetadata::smoke("resource-test"))
            .expect("isolated compact setup should not construct spectral or exact models");
        assert_eq!(archive.row.method, C51MethodId::WhitneyCompact);
        assert_eq!(
            archive.row.observation_class,
            C51ObservationClass::LocalPhysicalB
        );
        assert!(matches!(
            archive.row.factor_storage_nnz,
            C51Measurement::Measured(_)
        ));
        assert!(matches!(
            archive.row.spectral_rank,
            C51Measurement::NotRecorded
        ));
        assert!(matches!(
            archive.row.sampling_ns,
            C51Measurement::Unavailable { .. }
        ));
        assert!(matches!(
            archive.row.posterior_mean_relative_error,
            C51Measurement::Unavailable { ref reason }
                if reason == RESOURCE_ACCURACY_UNAVAILABLE_REASON
        ));
        archive.record_external_peak_rss(12_345).unwrap();
        assert_eq!(
            archive.row.factor_external_peak_rss_bytes,
            C51Measurement::Measured(12_345)
        );
        assert_eq!(
            archive.row.query_external_peak_rss_bytes,
            C51Measurement::NotRecorded
        );
    }

    #[test]
    fn calibrated_compact_resource_uses_root_physical_rms_pipeline_before_conditioning() {
        let selection = C51ResourceSelection {
            refinement: 4,
            observation_class: C51ObservationClass::LocalPhysicalB,
            sensor_target: C51ResourceSensorTarget::Count(4),
            repeated_data_count: 1,
            method: C51MethodId::WhitneyCompactCalibrated,
            spectral_rank: None,
            phase: C51ResourcePhase::ColdSetup,
        };
        let mut config = C51ResourceConfig::juq_performance(selection);
        config.projected_calibration_target_prior_rms = Some(0.5);
        let archive = run_c51_resource(&config, C51RunMetadata::smoke("calibrated-resource"))
            .expect("calibrated projected setup should use the canonical root pipeline");
        assert_eq!(archive.row.method, C51MethodId::WhitneyCompactCalibrated);
        assert!(!archive.row.matched_physical_qoi_eligible);
        assert_eq!(
            archive.row.amplitude_calibration_target_prior_rms,
            C51Measurement::Measured(0.5)
        );
        let scale = measured_f64(&archive.row.amplitude_calibration_precision_scale);
        let tau_multiplier = measured_f64(&archive.row.amplitude_calibration_tau_multiplier);
        assert!((tau_multiplier.powi(2) - scale).abs() <= 1.0e-12 * scale);
        assert!(matches!(
            archive.row.amplitude_calibration_ns,
            C51Measurement::Measured(value) if value > 0
        ));
    }

    #[test]
    fn integral_flux_resource_profile_freezes_method_specific_amplitudes() {
        let selection = |refinement, count, method| C51ResourceSelection {
            refinement,
            observation_class: C51ObservationClass::LocalFluxCochain,
            sensor_target: C51ResourceSensorTarget::Count(count),
            repeated_data_count: 1,
            method,
            spectral_rank: None,
            phase: C51ResourcePhase::ColdSetup,
        };
        let exact = C51ResourceConfig::juq_integral_flux_performance(selection(
            8,
            4_096,
            C51MethodId::FullMetricAugmented,
        ))
        .unwrap();
        assert_eq!(exact.tau, C51_INTEGRAL_FLUX_EXACT_TAU_N8);
        assert_eq!(
            exact.physical_output_design,
            C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit
        );

        let compact = C51ResourceConfig::juq_integral_flux_performance(selection(
            10,
            8_192,
            C51MethodId::WhitneyCompactIntegralFluxCalibrated,
        ))
        .unwrap();
        assert_eq!(compact.tau, C51_INTEGRAL_FLUX_COMPACT_TRACE_TAU_N10);
        assert_eq!(
            compact.projected_integral_flux_posterior_precision_scale,
            Some(C51_INTEGRAL_FLUX_POSTERIOR_SCALE_N10)
        );
        assert!(C51ResourceConfig::juq_integral_flux_performance(selection(
            8,
            4_095,
            C51MethodId::WhitneyCompact,
        ))
        .is_err());
    }

    #[test]
    fn bounded_integral_flux_resource_uses_no_reconstruction() {
        let selection = C51ResourceSelection {
            refinement: 4,
            observation_class: C51ObservationClass::LocalFluxCochain,
            sensor_target: C51ResourceSensorTarget::Count(32),
            repeated_data_count: 1,
            method: C51MethodId::WhitneyCompactIntegralFluxCalibrated,
            spectral_rank: None,
            phase: C51ResourcePhase::ColdSetup,
        };
        let mut config = C51ResourceConfig::juq_performance(selection);
        config.physical_observation_design = C51PhysicalObservationDesign::FluxCochainHashedPrefix;
        config.physical_output_design = C51PhysicalOutputDesign::MidplaneQuadrantSurfaceFluxSplit;
        config.observation_noise_variance = 2.5e-7;
        config.projected_integral_flux_posterior_precision_scale = Some(0.9);
        config.projected_integral_flux_target_posterior_rms = Some(1.0e-3);
        let archive = run_c51_resource(&config, C51RunMetadata::smoke("integral-flux-resource"))
            .expect("bounded direct-cochain resource row should run");
        assert_eq!(
            archive.row.observation_design_id,
            C51_FLUX_COCHAIN_DESIGN_ID
        );
        assert_eq!(
            archive.row.physical_qoi_design_id,
            C51_SURFACE_FLUX_QOI_DESIGN_ID
        );
        assert_eq!(archive.row.physical_qoi_output_count, 12);
        assert_eq!(
            archive.row.amplitude_calibration_precision_scale,
            C51Measurement::Measured(0.9)
        );
        assert!(matches!(
            archive.row.observation_fixed_radius,
            C51Measurement::Unavailable { .. }
        ));
    }

    #[test]
    fn resource_density_is_resolved_and_query_phase_executes_full_uq() {
        let selection = C51ResourceSelection {
            refinement: 2,
            observation_class: C51ObservationClass::LocalSelectors,
            sensor_target: C51ResourceSensorTarget::Density(0.5),
            repeated_data_count: 2,
            method: C51MethodId::WhitneyCompact,
            spectral_rank: None,
            phase: C51ResourcePhase::QueryFullUq,
        };
        let config = C51ResourceConfig::smoke(selection);
        let archive = run_c51_resource(&config, C51RunMetadata::smoke("resource-test"))
            .expect("isolated compact full-UQ resource row should run");
        assert_eq!(
            archive.row.sensor_count,
            (0.5 * archive.row.latent_dimension as f64).ceil() as usize
        );
        assert_eq!(archive.row.repeated_data_count, 2);
        assert!(matches!(
            archive.row.sampling_ns,
            C51Measurement::Measured(_)
        ));
        assert!(matches!(
            archive.row.variance_estimation_ns,
            C51Measurement::Measured(_)
        ));
    }

    #[test]
    fn common_physical_query_excludes_latent_work_for_compact_and_exact_gp() {
        for method in [
            C51MethodId::FullMetricAugmented,
            C51MethodId::WhitneyCompact,
            C51MethodId::ExactObservationSpaceGp,
        ] {
            let selection = C51ResourceSelection {
                refinement: 4,
                observation_class: C51ObservationClass::LocalPhysicalB,
                sensor_target: C51ResourceSensorTarget::Count(4),
                repeated_data_count: 2,
                method,
                spectral_rank: None,
                phase: C51ResourcePhase::QueryPhysicalUq,
            };
            let archive = run_c51_resource(
                &C51ResourceConfig::juq_performance(selection),
                C51RunMetadata::smoke("common-physical-query-test"),
            )
            .unwrap_or_else(|error| panic!("common physical query {method:?} failed: {error}"));
            assert!(archive.row.common_physical_uq_workload_eligible);
            assert!(!archive.row.full_uq_workload_eligible);
            assert!(matches!(
                archive.row.repeated_data_ns,
                C51Measurement::Measured(_)
            ));
            assert!(matches!(
                archive.row.physical_qoi_ns,
                C51Measurement::Measured(_)
            ));
            assert!(matches!(
                archive.row.variance_estimation_ns,
                C51Measurement::Measured(_)
            ));
            assert!(matches!(
                archive.row.covariance_action_ns,
                C51Measurement::Unavailable { .. }
            ));
            assert!(matches!(
                archive.row.sampling_ns,
                C51Measurement::Unavailable { .. }
            ));
        }
    }

    #[test]
    fn every_method_has_a_single_row_cold_setup_resource_route() {
        for (method, spectral_rank) in [
            (C51MethodId::FullMetricDirect, None),
            (C51MethodId::FullMetricAugmented, None),
            (C51MethodId::WhitneyCompact, None),
            (C51MethodId::ExactObservationSpaceGp, None),
            (C51MethodId::SpectralFullRank, None),
        ] {
            let selection = C51ResourceSelection {
                refinement: 2,
                observation_class: C51ObservationClass::LocalPhysicalB,
                sensor_target: C51ResourceSensorTarget::Count(4),
                repeated_data_count: 1,
                method,
                spectral_rank,
                phase: C51ResourcePhase::ColdSetup,
            };
            let archive = run_c51_resource(
                &C51ResourceConfig::smoke(selection),
                C51RunMetadata::smoke("resource-test"),
            )
            .unwrap_or_else(|error| panic!("resource route {method:?} failed: {error}"));
            assert_eq!(archive.row.method, method);
            assert_eq!(archive.config.selection.phase, C51ResourcePhase::ColdSetup);
        }
    }

    #[test]
    fn juq_augmented_cold_setup_is_common_physical_uq_eligible() {
        let selection = C51ResourceSelection {
            refinement: 8,
            observation_class: C51ObservationClass::LocalPhysicalB,
            sensor_target: C51ResourceSensorTarget::Count(4),
            repeated_data_count: 1,
            method: C51MethodId::FullMetricAugmented,
            spectral_rank: None,
            phase: C51ResourcePhase::ColdSetup,
        };
        let archive = run_c51_resource(
            &C51ResourceConfig::juq_performance(selection),
            C51RunMetadata::smoke("juq-augmented-cold-eligibility-test"),
        )
        .unwrap();
        assert!(archive.row.common_physical_uq_workload_eligible);
        assert!(archive.row.method_performance_timing_eligible);
        assert_eq!(
            archive.row.observation_design_id,
            C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID
        );
        assert_eq!(
            archive.row.physical_qoi_design_id,
            C51PhysicalOutputDesign::RankStudySelectionAndHeldOut.as_str()
        );
    }

    #[cfg(not(feature = "external-solvers"))]
    #[test]
    fn low_rank_resource_fails_closed_without_sparse_mixed_solver() {
        let selection = C51ResourceSelection {
            refinement: 2,
            observation_class: C51ObservationClass::LocalPhysicalB,
            sensor_target: C51ResourceSensorTarget::Count(4),
            repeated_data_count: 1,
            method: C51MethodId::SpectralLowRank,
            spectral_rank: Some(4),
            phase: C51ResourcePhase::ColdSetup,
        };
        let error = run_c51_resource(
            &C51ResourceConfig::smoke(selection),
            C51RunMetadata::smoke("resource-test"),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requires the external-solvers feature"));
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    fn exact_requested_source_classification_fails_closed() {
        let valid = PartialBasisDiagnostics {
            requested_rank: 4,
            converged_rank: 5,
            retained_rank: 4,
            which: "smallest".to_string(),
            mass_solve: "direct".to_string(),
            l_nonzeros: 3,
            d_nonzeros: 5,
            c_nonzeros: 5,
            mkm1_nonzeros: 10,
            mk_nonzeros: 10,
            mass_nonzeros: 10,
            input_nonzeros: 50,
            basis_storage_bytes: 128,
            clamped_negative_eigenvalues: 0,
            maximum_relative_mixed_residual: 1.0e-11,
            maximum_mass_orthogonality_defect: 1.0e-12,
            mass_orthogonality_tolerance: 1.0e-8,
            eigenvalue_negative_tolerance: 1.0e-10,
            eigensolver_tolerance: 1.0e-10,
            residual_validation: Duration::from_nanos(1),
            feec_assembly_reused: false,
            workspace: "workspace".to_string(),
            binary: "binary".to_string(),
        };
        assert!(exact_requested_mixed_timing_ineligibility_reason(&valid, 4, 4).is_none());

        let mut mismatched_rank = valid.clone();
        mismatched_rank.retained_rank = 3;
        assert!(
            exact_requested_mixed_timing_ineligibility_reason(&mismatched_rank, 4, 4).is_some()
        );

        let mut wrong_source_target = valid.clone();
        wrong_source_target.which = "largest".to_string();
        assert!(
            exact_requested_mixed_timing_ineligibility_reason(&wrong_source_target, 4, 4).is_some()
        );

        let mut failed_residual = valid.clone();
        failed_residual.maximum_relative_mixed_residual =
            EXACT_REQUESTED_MIXED_RESIDUAL_TOLERANCE * 2.0;
        assert!(
            exact_requested_mixed_timing_ineligibility_reason(&failed_residual, 4, 4).is_some()
        );

        let mut relaxed_orthogonality = valid.clone();
        relaxed_orthogonality.mass_orthogonality_tolerance = 1.0e-6;
        assert!(
            exact_requested_mixed_timing_ineligibility_reason(&relaxed_orthogonality, 4, 4)
                .is_some()
        );

        let mut relaxed_negative_eigenvalue = valid;
        relaxed_negative_eigenvalue.eigenvalue_negative_tolerance = 1.0e-6;
        assert!(exact_requested_mixed_timing_ineligibility_reason(
            &relaxed_negative_eigenvalue,
            4,
            4
        )
        .is_some());
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    #[ignore = "requires an installed PETSc/SLEPc helper; run explicitly as a bounded exact-request integration test"]
    fn bounded_exact_requested_resource_archives_certified_source() {
        let selection = C51ResourceSelection {
            refinement: 2,
            observation_class: C51ObservationClass::LocalPhysicalB,
            sensor_target: C51ResourceSensorTarget::Count(4),
            repeated_data_count: 1,
            method: C51MethodId::SpectralLowRank,
            spectral_rank: Some(4),
            phase: C51ResourcePhase::ColdSetup,
        };
        let archive = run_c51_resource(
            &C51ResourceConfig::smoke(selection),
            C51RunMetadata::smoke("exact-request-resource-test"),
        )
        .unwrap();
        assert_eq!(
            archive.row.spectral_basis_source,
            "partial_exact_requested_mixed_shell"
        );
        assert_eq!(
            archive.row.spectral_partial_requested_rank,
            C51Measurement::Measured(4)
        );
        assert_eq!(
            archive.row.spectral_partial_retained_rank,
            C51Measurement::Measured(4)
        );
        let mixed_block_nnz = [
            &archive.row.spectral_partial_l_nnz,
            &archive.row.spectral_partial_d_nnz,
            &archive.row.spectral_partial_c_nnz,
            &archive.row.spectral_partial_mkm1_nnz,
            &archive.row.spectral_partial_mk_nnz,
        ]
        .map(|measurement| match measurement {
            C51Measurement::Measured(value) => *value,
            other => panic!("expected measured mixed-block nnz, got {other:?}"),
        });
        assert!(mixed_block_nnz.iter().all(|value| *value > 0));
        assert_eq!(
            archive.row.spectral_partial_input_nnz,
            C51Measurement::Measured(mixed_block_nnz.into_iter().sum())
        );
        assert_eq!(
            archive.row.spectral_partial_mass_nnz,
            archive.row.spectral_partial_mk_nnz
        );
        assert!(matches!(
            archive.row.spectral_partial_maximum_relative_mixed_residual,
            C51Measurement::Measured(value) if value <= EXACT_REQUESTED_MIXED_RESIDUAL_TOLERANCE
        ));
        assert!(archive.row.method_performance_timing_eligible);
        assert!(archive
            .row
            .method_performance_timing_ineligibility_reason
            .is_empty());
        assert_ne!(archive.row.spectral_feec_assembly_reuse, "reused");
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    #[ignore = "requires an installed PETSc/SLEPc helper; run explicitly as a bounded adapter pilot"]
    fn bounded_partial_spectral_rank_pilot_archives_helper_diagnostics() {
        let mut config = C51BenchmarkConfig::spectral_rank_pilot();
        config.refinements = vec![2];
        config.local_sensor_counts = vec![4];
        config.local_sensor_density_targets = vec![0.5];
        config.repeated_data_counts = vec![1];
        config.spectral_low_ranks = vec![2, 4];
        config.spectral_rank_fractions.clear();
        config.spectral_operator_materialization_maximum_dimension = 1;
        let archive =
            run_c51_spectral_rank_pilot(&config, C51RunMetadata::smoke("bounded-partial-pilot"))
                .unwrap();
        assert!(archive.rows.iter().any(|row| {
            row.method == C51MethodId::SpectralLowRank
                && !row.method_performance_timing_eligible
                && row.spectral_basis_source == "partial_shared_maximum_mixed_shell"
                && row.spectral_eigensolver_backend
                    == "petsc_slepc_mixed_reduced_generalized_eigensolver"
                && matches!(
                    row.spectral_partial_converged_rank,
                    C51Measurement::Measured(_)
                )
                && matches!(
                    &row.spectral_partial_weak_hodge_laplacian_nnz,
                    C51Measurement::Unavailable { reason }
                        if reason == MIXED_MARGINAL_NNZ_UNAVAILABLE_REASON
                )
        }));
        assert!(archive.rows.iter().any(|row| {
            row.method == C51MethodId::WhitneyCompact
                && matches!(
                    &row.physical_variance_relative_error,
                    C51Measurement::Measured(value) if value.is_finite()
                )
        }));
    }

    fn measured_f64(measurement: &C51Measurement<f64>) -> f64 {
        match measurement {
            C51Measurement::Measured(value) => *value,
            other => panic!("expected measured f64, got {other:?}"),
        }
    }
}
