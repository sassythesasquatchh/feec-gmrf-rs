//! Reusable C51 accuracy--resource benchmark harness.
//!
//! The smoke profile is deliberately small. It checks the archive schema and
//! compares three production construction routes on a three-dimensional
//! Whitney 1-form problem:
//!
//! - `FM-DIRECT`: exact eliminated full-metric precision;
//! - `FM-AUG`: the same finite Gaussian law on a sparse constrained support;
//! - `WHITNEY-COMPACT`: the projected-inverse Whitney approximation.
//!
//! A spectral row is always present. Until the frozen generalized-eigenbasis
//! adapter is available it is recorded as typed `unavailable`; it is never
//! silently skipped or assigned zero cost. Tiny timings validate
//! instrumentation only and are not evidence of a performance advantage.

use feec_gmrf::prelude::{
    CholeskyOrderingKind, EssentialBoundaryConditions, EssentialBoundaryElimination,
    ExactFormOperators, ExactMaternPriorBuilder, FeecGmrfError, GaussianNoise, GaussianPrior,
    LinearGaussianModelBuilder, LinearMap, LinearObservation, MagneticFieldMaps3d, MaternAlpha,
    MaternParameters, MaternPriorBuilder, ResourceMeasurement, Result, SupportedMaternPrior,
    SupportedMaternPriorBuilder, VarianceMethod,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use rand::{rngs::StdRng, SeedableRng};
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};

pub const C51_SCHEMA_VERSION: &str = "c51-accuracy-resource-v1";
pub const C51_STUDY_ID: &str = "c51/accuracy-resource";

const EXACT_REFERENCE_MAXIMUM_DIMENSION: usize = 1_024;
const PHYSICAL_QOI_OUTPUT_COUNT: usize = 3;
const UPDATE_NNZ_PENDING_REASON: &str =
    "supported-law observation-update nnz is not yet exposed by the root API";
const SPECTRAL_PENDING_REASON: &str =
    "the frozen generalized-eigenbasis spectral adapter is not available in this smoke harness";
const KKT_FILL_UNAVAILABLE_REASON: &str =
    "faer 0.21.9 does not expose sparse LU numeric factor storage";

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
    Spectral,
}

impl C51MethodId {
    pub const ALL: [Self; 4] = [
        Self::FullMetricDirect,
        Self::FullMetricAugmented,
        Self::WhitneyCompact,
        Self::Spectral,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullMetricDirect => "fm_direct",
            Self::FullMetricAugmented => "fm_aug",
            Self::WhitneyCompact => "whitney_compact",
            Self::Spectral => "spectral_r",
        }
    }
}

/// Observation locality is an explicit study axis, never pooled after a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C51ObservationClass {
    LocalPhysicalB,
    LocalSelectors,
    GlobalRow,
}

impl C51ObservationClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalPhysicalB => "local_physical_b",
            Self::LocalSelectors => "local_selectors",
            Self::GlobalRow => "global_row",
        }
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
}

/// Frozen mathematical and workload axes for one benchmark invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct C51BenchmarkConfig {
    pub refinements: Vec<usize>,
    pub spatial_dimension: usize,
    pub form_degree: usize,
    pub alpha: MaternAlpha,
    pub kappa: f64,
    pub tau: f64,
    pub observation_noise_variance: f64,
    /// Local bounded-support sensor counts.
    pub local_sensor_counts: Vec<usize>,
    /// Number of replacement data vectors solved against a fixed design.
    pub repeated_data_counts: Vec<usize>,
    /// Add one dense, mesh-global observation row as a locality control.
    pub include_global_row: bool,
    /// Requested spectral rank. The smoke row is typed unavailable until its adapter lands.
    pub spectral_rank: usize,
    /// Posterior samples included in each available full-UQ row.
    pub posterior_sample_count: usize,
    pub seed: u64,
}

impl C51BenchmarkConfig {
    /// Cheap 3D 1-form profile exercising local/global and amortization axes.
    pub fn smoke() -> Self {
        Self {
            refinements: vec![2],
            spatial_dimension: 3,
            form_degree: 1,
            alpha: MaternAlpha::Two,
            kappa: 4.0,
            tau: 1.0,
            observation_noise_variance: 0.05,
            local_sensor_counts: vec![2, 4],
            repeated_data_counts: vec![1, 2],
            include_global_row: true,
            spectral_rank: 8,
            posterior_sample_count: 4,
            seed: 0xC510_2026,
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
        if self.local_sensor_counts.is_empty() || self.local_sensor_counts.contains(&0) {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 requires at least one positive local sensor count".to_string(),
            ));
        }
        if self.repeated_data_counts.is_empty() || self.repeated_data_counts.contains(&0) {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 requires at least one positive repeated-data count".to_string(),
            ));
        }
        if self.spectral_rank == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 spectral rank must be positive".to_string(),
            ));
        }
        if self.posterior_sample_count == 0 {
            return Err(FeecGmrfError::InvalidParameter(
                "C51 posterior sample count must be positive".to_string(),
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
    pub full_uq_workload_eligible: bool,
    pub refinement: usize,
    pub spatial_dimension: usize,
    pub form_degree: usize,
    pub alpha: u32,
    pub kappa: f64,
    pub tau: f64,
    pub observation_class: C51ObservationClass,
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
    pub spectral_rank: C51Measurement<usize>,
    pub spectral_basis_storage_values: C51Measurement<usize>,
    pub solver: String,
    pub ordering: String,
    pub feec_assembly_ns: C51Measurement<u128>,
    pub prior_build_ns: C51Measurement<u128>,
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
    pub physical_variance_relative_error: C51Measurement<f64>,
    pub physical_transformed_variance_relative_error: C51Measurement<f64>,
    pub physical_mean_relative_error: C51Measurement<f64>,
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
            if row.method == C51MethodId::Spectral {
                if row.available || row.unavailable_reason.is_empty() {
                    return Err(FeecGmrfError::InvalidParameter(
                        "unimplemented spectral rows must be explicitly unavailable".to_string(),
                    ));
                }
                continue;
            }
            if !row.available || row.latent_dimension == 0 || row.observation_nnz == 0 {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "available C51 method {} has an incomplete row",
                    row.method.as_str()
                )));
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
        writeln!(writer, "axes:")?;
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
            "  repeated_data_counts: {}",
            yaml_usize_list(&self.config.repeated_data_counts)
        )?;
        writeln!(
            writer,
            "  include_global_row: {}",
            self.config.include_global_row
        )?;
        writeln!(writer, "  spectral_rank: {}", self.config.spectral_rank)?;
        writeln!(
            writer,
            "  posterior_sample_count: {}",
            self.config.posterior_sample_count
        )?;
        writeln!(
            writer,
            "  observation_classes: [\"local_physical_b\", \"local_selectors\", \"global_row\"]"
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

struct BuiltPriors {
    exact: GaussianPrior,
    compact: GaussianPrior,
    supported: SupportedMaternPrior,
    exact_build: Duration,
    compact_build: Duration,
    supported_build: Duration,
}

struct ProblemMaps {
    physical_qoi: LinearMap,
    physical_observation: LinearMap,
    latent_probe: Vec<f64>,
}

#[derive(Clone)]
struct ObservationDesign {
    class: C51ObservationClass,
    operator: LinearMap,
    sensor_count: usize,
}

struct ReferenceAnswers {
    mean: Vec<f64>,
    covariance_action: Vec<f64>,
    physical_covariance_action: Vec<f64>,
    physical_variance: f64,
    physical_transformed_variances: Vec<f64>,
    physical_mean: Vec<f64>,
}

struct AccuracyMeasurements {
    posterior_mean: C51Measurement<f64>,
    covariance_action: C51Measurement<f64>,
    physical_covariance_action: C51Measurement<f64>,
    physical_variance: C51Measurement<f64>,
    physical_transformed_variance: C51Measurement<f64>,
    physical_mean: C51Measurement<f64>,
}

fn run_refinement(config: &C51BenchmarkConfig, refinement: usize) -> Result<Vec<C51BenchmarkRow>> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let assembly_start = Instant::now();
    let operators = ExactFormOperators::from_feec_homogeneous_full_relative(
        &topology,
        &metric,
        config.form_degree,
    )?;
    let physical_maps = MagneticFieldMaps3d::from_feec(&topology, &coordinates)?;
    let elimination =
        EssentialBoundaryElimination::from_layout(operators.state().boundary_layout().clone())?;
    let physical_observation = elimination
        .eliminate_map(
            physical_maps.magnetic_field.map(),
            &vec![0.0; physical_maps.magnetic_field.map().output_dimension()],
        )?
        .reduced()
        .clone();
    let physical_qoi = select_physical_qoi(&physical_observation)?;
    let feec_assembly = assembly_start.elapsed();
    let parameters = MaternParameters::new(config.alpha, config.kappa, config.tau)?;
    let priors = build_priors(config, &topology, &metric, &operators, parameters)?;
    let problem = ProblemMaps {
        physical_qoi,
        physical_observation,
        latent_probe: deterministic_probe(operators.state().dimension(), config.seed),
    };
    let designs = observation_designs(
        config,
        operators.state().dimension(),
        &problem.physical_observation,
    )?;
    let mut rows = Vec::new();
    for design in designs {
        for &repeated_data_count in &config.repeated_data_counts {
            let exact = run_spd_method(
                C51MethodId::FullMetricDirect,
                &priors.exact,
                priors.exact_build,
                feec_assembly,
                config,
                refinement,
                &design,
                repeated_data_count,
                &problem,
                None,
            )?;
            let reference = exact.1;
            rows.push(exact.0);
            rows.push(run_supported_method(
                &priors.supported,
                priors.supported_build,
                feec_assembly,
                config,
                refinement,
                &design,
                repeated_data_count,
                &problem,
                &reference,
            )?);
            rows.push(
                run_spd_method(
                    C51MethodId::WhitneyCompact,
                    &priors.compact,
                    priors.compact_build,
                    feec_assembly,
                    config,
                    refinement,
                    &design,
                    repeated_data_count,
                    &problem,
                    Some(&reference),
                )?
                .0,
            );
            rows.push(unavailable_spectral_row(
                config,
                refinement,
                &design,
                repeated_data_count,
                operators.state().dimension(),
                feec_assembly,
            ));
        }
    }
    Ok(rows)
}

fn build_priors(
    config: &C51BenchmarkConfig,
    topology: &manifold::topology::complex::Complex,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    operators: &ExactFormOperators,
    parameters: MaternParameters,
) -> Result<BuiltPriors> {
    let start = Instant::now();
    let exact = ExactMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .maximum_materialized_dimension(EXACT_REFERENCE_MAXIMUM_DIMENSION)?
        .build()?;
    let exact_build = start.elapsed();

    let start = Instant::now();
    let supported = SupportedMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .build()?;
    let supported_build = start.elapsed();

    let start = Instant::now();
    let compact =
        MaternPriorBuilder::from_feec_whitney_projected(topology, metric, config.form_degree)?
            .parameters(parameters)
            .essential_boundary_conditions(EssentialBoundaryConditions::homogeneous_on_boundary(
                topology,
                config.form_degree,
            )?)
            .build()?;
    let compact_build = start.elapsed();
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
    let physical_rhs = problem.physical_qoi.apply_transpose(&[1.0, -0.5, 0.25])?;
    let physical_start = Instant::now();
    let physical_covariance_action = posterior.covariance_action(&physical_rhs)?;
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
        let _ = posterior.sample_cochain(&mut rng)?;
    }
    let sample_elapsed = sample_start.elapsed();
    let answers = ReferenceAnswers {
        mean: posterior.mean().to_vec(),
        covariance_action,
        physical_covariance_action,
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
        matched_physical_qoi_eligible: true,
        full_uq_workload_eligible: true,
        refinement,
        spatial_dimension: config.spatial_dimension,
        form_degree: config.form_degree,
        alpha: config.alpha.as_u32(),
        kappa: config.kappa,
        tau: config.tau,
        observation_class: design.class,
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
        spectral_rank: C51Measurement::NotRecorded,
        spectral_basis_storage_values: C51Measurement::NotRecorded,
        solver: "sparse_cholesky".to_string(),
        ordering: ordering_name(factor.ordering).to_string(),
        feec_assembly_ns: duration_measurement(feec_assembly),
        prior_build_ns: duration_measurement(prior_build),
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
        physical_variance_relative_error: errors.physical_variance,
        physical_transformed_variance_relative_error: errors.physical_transformed_variance,
        physical_mean_relative_error: errors.physical_mean,
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
    reference: &ReferenceAnswers,
) -> Result<C51BenchmarkRow> {
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
    let physical_rhs = problem.physical_qoi.apply_transpose(&[1.0, -0.5, 0.25])?;
    let physical_start = Instant::now();
    let physical_covariance_action = posterior.covariance_action(&physical_rhs)?;
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
        physical_variance,
        physical_transformed_variances,
        physical_mean,
    };
    let errors = errors_against(&answers, Some(reference))?;
    Ok(C51BenchmarkRow {
        method: C51MethodId::FullMetricAugmented,
        available: true,
        unavailable_reason: String::new(),
        discrete_law_id: "exact_full_metric_whitney".to_string(),
        same_discrete_law_eligible: true,
        matched_physical_qoi_eligible: true,
        full_uq_workload_eligible: false,
        refinement,
        spatial_dimension: config.spatial_dimension,
        form_degree: config.form_degree,
        alpha: config.alpha.as_u32(),
        kappa: config.kappa,
        tau: config.tau,
        observation_class: design.class,
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
        spectral_rank: C51Measurement::NotRecorded,
        spectral_basis_storage_values: C51Measurement::NotRecorded,
        solver: "sparse_lu_kkt".to_string(),
        ordering: "backend_partial_pivoting".to_string(),
        feec_assembly_ns: duration_measurement(feec_assembly),
        prior_build_ns: duration_measurement(prior_build),
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
        physical_variance_relative_error: errors.physical_variance,
        physical_transformed_variance_relative_error: errors.physical_transformed_variance,
        physical_mean_relative_error: errors.physical_mean,
    })
}

fn unavailable_spectral_row(
    config: &C51BenchmarkConfig,
    refinement: usize,
    design: &ObservationDesign,
    repeated_data_count: usize,
    latent_dimension: usize,
    feec_assembly: Duration,
) -> C51BenchmarkRow {
    let unavailable_usize = || C51Measurement::unavailable(SPECTRAL_PENDING_REASON);
    let unavailable_u128 = || C51Measurement::unavailable(SPECTRAL_PENDING_REASON);
    let unavailable_u64 = || C51Measurement::unavailable(SPECTRAL_PENDING_REASON);
    let unavailable_f64 = || C51Measurement::unavailable(SPECTRAL_PENDING_REASON);
    C51BenchmarkRow {
        method: C51MethodId::Spectral,
        available: false,
        unavailable_reason: SPECTRAL_PENDING_REASON.to_string(),
        discrete_law_id: "exact_full_metric_whitney_truncated_spectral_target".to_string(),
        same_discrete_law_eligible: true,
        matched_physical_qoi_eligible: true,
        full_uq_workload_eligible: false,
        refinement,
        spatial_dimension: config.spatial_dimension,
        form_degree: config.form_degree,
        alpha: config.alpha.as_u32(),
        kappa: config.kappa,
        tau: config.tau,
        observation_class: design.class,
        sensor_count: design.sensor_count,
        sensor_density: design.sensor_count as f64 / latent_dimension as f64,
        repeated_data_count,
        posterior_sample_count: config.posterior_sample_count,
        seed: config.seed,
        latent_dimension,
        ambient_dimension: latent_dimension,
        constraint_dimension: 0,
        observation_nnz: design.operator.matrix().nnz(),
        prior_precision_nnz: unavailable_usize(),
        observation_update_nnz: unavailable_usize(),
        posterior_precision_nnz: unavailable_usize(),
        ambient_energy_nnz: unavailable_usize(),
        constraint_nnz: unavailable_usize(),
        kkt_input_nnz: C51Measurement::NotRecorded,
        factor_storage_nnz: unavailable_usize(),
        certificate_factor_storage_nnz: C51Measurement::NotRecorded,
        fill_ratio: unavailable_f64(),
        spectral_rank: C51Measurement::Measured(config.spectral_rank.min(latent_dimension)),
        spectral_basis_storage_values: unavailable_usize(),
        solver: "spectral_unavailable".to_string(),
        ordering: "not_applicable".to_string(),
        feec_assembly_ns: duration_measurement(feec_assembly),
        prior_build_ns: unavailable_u128(),
        observation_update_assembly_ns: unavailable_u128(),
        observation_update_and_factor_ns: unavailable_u128(),
        symbolic_analysis_ns: unavailable_u128(),
        numeric_factor_ns: unavailable_u128(),
        combined_factor_setup_ns: unavailable_u128(),
        repeated_data_ns: unavailable_u128(),
        covariance_action_ns: unavailable_u128(),
        physical_qoi_ns: unavailable_u128(),
        sampling_ns: unavailable_u128(),
        variance_estimation_ns: unavailable_u128(),
        factor_external_peak_rss_bytes: unavailable_u64(),
        query_external_peak_rss_bytes: unavailable_u64(),
        posterior_mean_relative_error: unavailable_f64(),
        covariance_action_relative_error: unavailable_f64(),
        physical_covariance_action_relative_error: unavailable_f64(),
        physical_variance_relative_error: unavailable_f64(),
        physical_transformed_variance_relative_error: unavailable_f64(),
        physical_mean_relative_error: unavailable_f64(),
    }
}

fn observation_designs(
    config: &C51BenchmarkConfig,
    dimension: usize,
    physical_observation: &LinearMap,
) -> Result<Vec<ObservationDesign>> {
    let mut designs = Vec::new();
    for &requested in &config.local_sensor_counts {
        let sensor_count = requested.min(dimension);
        let physical_sensor_count = requested.min(physical_observation.output_dimension());
        let physical_rows = evenly_spaced_indices(
            physical_observation.output_dimension(),
            physical_sensor_count,
        );
        designs.push(ObservationDesign {
            class: C51ObservationClass::LocalPhysicalB,
            operator: physical_observation.select_outputs(&physical_rows)?,
            sensor_count: physical_sensor_count,
        });
        let indices = evenly_spaced_indices(dimension, sensor_count);
        designs.push(ObservationDesign {
            class: C51ObservationClass::LocalSelectors,
            operator: LinearMap::selector(dimension, &indices)?,
            sensor_count,
        });
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
        });
    }
    Ok(designs)
}

fn select_physical_qoi(map: &LinearMap) -> Result<LinearMap> {
    if map.output_dimension() < PHYSICAL_QOI_OUTPUT_COUNT {
        return Err(FeecGmrfError::Dimension(
            "the canonical magnetic map has fewer than three outputs".to_string(),
        ));
    }
    let last = map.output_dimension() - 1;
    map.select_outputs(&[0, last / 2, last])
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

fn errors_against(
    answers: &ReferenceAnswers,
    reference: Option<&ReferenceAnswers>,
) -> Result<AccuracyMeasurements> {
    let Some(reference) = reference else {
        return Ok(AccuracyMeasurements {
            posterior_mean: C51Measurement::Measured(0.0),
            covariance_action: C51Measurement::Measured(0.0),
            physical_covariance_action: C51Measurement::Measured(0.0),
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
        C51MethodId::Spectral => 0x5EEC_7000,
    }
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
        "full_uq_workload_eligible",
        "refinement",
        "spatial_dimension",
        "form_degree",
        "alpha",
        "kappa",
        "tau",
        "observation_class",
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

fn measurement_field_names() -> [&'static str; 31] {
    [
        "prior_precision_nnz",
        "observation_update_nnz",
        "posterior_precision_nnz",
        "ambient_energy_nnz",
        "constraint_nnz",
        "kkt_input_nnz",
        "factor_storage_nnz",
        "certificate_factor_storage_nnz",
        "fill_ratio",
        "spectral_rank",
        "spectral_basis_storage_values",
        "feec_assembly_ns",
        "prior_build_ns",
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
        "physical_variance_relative_error",
        "physical_transformed_variance_relative_error",
        "physical_mean_relative_error",
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
        row.full_uq_workload_eligible.to_string(),
        row.refinement.to_string(),
        row.spatial_dimension.to_string(),
        row.form_degree.to_string(),
        row.alpha.to_string(),
        row.kappa.to_string(),
        row.tau.to_string(),
        row.observation_class.as_str().to_string(),
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
    ];
    push_measurement(&mut fields, &row.prior_precision_nnz);
    push_measurement(&mut fields, &row.observation_update_nnz);
    push_measurement(&mut fields, &row.posterior_precision_nnz);
    push_measurement(&mut fields, &row.ambient_energy_nnz);
    push_measurement(&mut fields, &row.constraint_nnz);
    push_measurement(&mut fields, &row.kkt_input_nnz);
    push_measurement(&mut fields, &row.factor_storage_nnz);
    push_measurement(&mut fields, &row.certificate_factor_storage_nnz);
    push_measurement(&mut fields, &row.fill_ratio);
    push_measurement(&mut fields, &row.spectral_rank);
    push_measurement(&mut fields, &row.spectral_basis_storage_values);
    push_measurement(&mut fields, &row.feec_assembly_ns);
    push_measurement(&mut fields, &row.prior_build_ns);
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
    push_measurement(&mut fields, &row.physical_variance_relative_error);
    push_measurement(
        &mut fields,
        &row.physical_transformed_variance_relative_error,
    );
    push_measurement(&mut fields, &row.physical_mean_relative_error);
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
        let designs = 2 * config.local_sensor_counts.len() + usize::from(config.include_global_row);
        assert_eq!(
            archive.rows.len(),
            designs * config.repeated_data_counts.len() * C51MethodId::ALL.len()
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
                C51MethodId::WhitneyCompact => {
                    assert!(!row.same_discrete_law_eligible);
                    assert!(row.full_uq_workload_eligible);
                    assert!(measured_f64(&row.physical_variance_relative_error).is_finite());
                }
                C51MethodId::Spectral => {
                    assert!(!row.available);
                    assert!(!row.unavailable_reason.is_empty());
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
        assert!(csv.contains("spectral_r,false"));
        assert!(manifest.contains("same_discrete_law_representation"));
        assert!(manifest.contains("matched_physical_qoi_method"));
        assert!(manifest.contains("cpu_model: null"));
        fs::remove_dir_all(directory).unwrap();
    }

    fn measured_f64(measurement: &C51Measurement<f64>) -> f64 {
        match measurement {
            C51Measurement::Measured(value) => *value,
            other => panic!("expected measured f64, got {other:?}"),
        }
    }
}
