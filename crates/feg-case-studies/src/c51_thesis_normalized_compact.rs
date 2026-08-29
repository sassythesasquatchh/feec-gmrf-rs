//! C51 projected-Whitney accuracy under the thesis magnetic trace convention.
//!
//! Both the exact supported FEEC law and the compact projected-Whitney law are
//! independently scaled to the same domain-wide RMS of `B = D1 A` in the
//! degree-2 FEEC mass norm. Sensor rows are used only for conditioning and
//! held-out evaluation; they never define the amplitude.

use crate::c51_benchmark::C51RunMetadata;
use crate::c51_magnetic_sensor_design::{
    assemble_juq_performance_observation_prefix, held_out_kernels, selection_kernels,
    C51_FIXED_OBSERVATION_RADIUS, C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID,
};
use crate::c51_spectral_covariance_rank::{
    covariance_error_metrics, C51CovarianceErrorMetrics, C51_PERFORMANCE_OUTPUT_DESIGN_ID,
};
use feec_gmrf::prelude::{
    pullback_quadratic_form, EssentialBoundaryConditions, ExactFormOperators, GaussianNoise,
    HutchinsonVarianceConfig, LinearGaussianModelBuilder, LinearObservation,
    MagneticVolumeSensorBank3d, MaternAlpha, MaternParameters, MaternPriorBuilder, Result,
    SupportedMaternPriorBuilder, VarianceMethod,
};
use manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::simplex::SimplexHandleExt};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION: &str = "c51-thesis-normalized-compact-v1";
pub const C51_THESIS_NORMALIZED_COMPACT_STUDY_ID: &str = "c51/thesis-normalized-compact-accuracy";
pub const C51_THESIS_NORMALIZED_COMPACT_PROFILE_N8: &str = "thesis-magnetic-trace-normalized-v1-n8";
pub const C51_THESIS_NORMALIZED_COMPACT_PROFILE_N10: &str =
    "thesis-magnetic-trace-normalized-v1-n10";
pub const C51_THESIS_TARGET_B_RMS_TESLA: f64 = 0.10;
pub const C51_THESIS_TRACE_PROBES: usize = 512;
pub const C51_THESIS_TRACE_BATCHES: usize = 8;
pub const C51_THESIS_TRACE_SEED: u64 = 0xC51B_7ACE;
pub const C51_THESIS_OBSERVATION_COUNT: usize = 4_096;

#[derive(Debug, Clone, PartialEq)]
pub struct C51ThesisNormalizedCompactConfig {
    pub profile: String,
    pub refinement: usize,
    pub alpha: MaternAlpha,
    pub kappa: f64,
    pub base_tau: f64,
    pub target_b_rms_tesla: f64,
    pub observation_noise_variance: f64,
    pub observation_count: usize,
    pub trace_probes: usize,
    pub trace_batches: usize,
    pub trace_seed: u64,
}

impl C51ThesisNormalizedCompactConfig {
    pub fn publication(refinement: usize) -> Result<Self> {
        let profile = match refinement {
            8 => C51_THESIS_NORMALIZED_COMPACT_PROFILE_N8,
            10 => C51_THESIS_NORMALIZED_COMPACT_PROFILE_N10,
            _ => {
                return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
                    "C51 thesis-normalized refinement must be 8 or 10".to_string(),
                ))
            }
        };
        Ok(Self {
            profile: profile.to_string(),
            refinement,
            alpha: MaternAlpha::Two,
            kappa: 4.0,
            base_tau: 1.0,
            target_b_rms_tesla: C51_THESIS_TARGET_B_RMS_TESLA,
            observation_noise_variance: 0.05,
            observation_count: C51_THESIS_OBSERVATION_COUNT,
            trace_probes: C51_THESIS_TRACE_PROBES,
            trace_batches: C51_THESIS_TRACE_BATCHES,
            trace_seed: C51_THESIS_TRACE_SEED,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C51ThesisTraceCalibration {
    pub raw_b_rms: f64,
    pub target_b_rms: f64,
    pub precision_scale: f64,
    pub tau_multiplier: f64,
    pub normalized_b_rms: f64,
    pub weighted_trace_relative_standard_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51ThesisNormalizedCompactReport {
    pub metadata: C51RunMetadata,
    pub config: C51ThesisNormalizedCompactConfig,
    pub state_dimension: usize,
    pub full_cochain_dimension: usize,
    pub magnetic_output_dimension: usize,
    pub domain_volume: f64,
    pub exact: C51ThesisTraceCalibration,
    pub compact: C51ThesisTraceCalibration,
    pub selection: C51CovarianceErrorMetrics,
    pub held_out: C51CovarianceErrorMetrics,
}

pub fn run_c51_thesis_normalized_compact(
    config: &C51ThesisNormalizedCompactConfig,
    metadata: C51RunMetadata,
) -> Result<C51ThesisNormalizedCompactReport> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, config.refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let operators = ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1)?;
    let parameters = MaternParameters::new(config.alpha, config.kappa, config.base_tau)?;
    let raw_exact = SupportedMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .build()?;
    let raw_compact = MaternPriorBuilder::from_feec_whitney_projected(&topology, &metric, 1)?
        .parameters(parameters)
        .essential_boundary_conditions(EssentialBoundaryConditions::homogeneous_on_boundary(
            &topology, 1,
        )?)
        .build()?;
    if raw_exact.state_dimension() != raw_compact.dimension() {
        return Err(feec_gmrf::FeecGmrfError::Dimension(format!(
            "exact state dimension {} differs from compact dimension {}",
            raw_exact.state_dimension(),
            raw_compact.dimension()
        )));
    }

    let outgoing = operators.outgoing().ok_or_else(|| {
        feec_gmrf::FeecGmrfError::Dimension(
            "C51 1-form calibration requires the outgoing D1 operator and degree-2 mass"
                .to_string(),
        )
    })?;
    let magnetic_energy =
        pullback_quadratic_form(outgoing.active_derivative(), outgoing.upper().mass())?;
    let magnetic_output_dimension = outgoing.upper().dimension();
    let domain_volume = topology
        .cells()
        .handle_iter()
        .map(|cell| cell.coord_simplex(&coordinates).vol())
        .sum::<f64>();
    if !domain_volume.is_finite() || domain_volume <= 0.0 {
        return Err(feec_gmrf::FeecGmrfError::Assembly(
            "C51 mesh has non-positive or non-finite volume".to_string(),
        ));
    }
    let trace_method = VarianceMethod::Hutchinson(HutchinsonVarianceConfig::new(
        config.trace_probes,
        config.trace_batches,
        config.trace_seed,
    )?);
    let exact_trace = raw_exact.weighted_covariance_trace(&magnetic_energy, trace_method)?;
    let compact_trace = raw_compact
        .factor()?
        .weighted_covariance_trace(&magnetic_energy, trace_method)?;
    let exact_calibration = trace_calibration(
        exact_trace.value,
        exact_trace.relative_standard_error,
        config.target_b_rms_tesla,
        domain_volume,
    )?;
    let compact_calibration = trace_calibration(
        compact_trace.value,
        compact_trace.relative_standard_error,
        config.target_b_rms_tesla,
        domain_volume,
    )?;

    let exact = SupportedMaternPriorBuilder::from_operators(operators)
        .parameters(MaternParameters::new(
            config.alpha,
            config.kappa,
            config.base_tau * exact_calibration.tau_multiplier,
        )?)
        .build()?;
    let compact = raw_compact.with_precision_scale(compact_calibration.precision_scale)?;

    let observations = assemble_juq_performance_observation_prefix(
        &topology,
        &coordinates,
        config.observation_count,
    )?;
    let observation = LinearObservation::new(
        observations.scalar_map().clone(),
        vec![0.0; config.observation_count],
        GaussianNoise::variance(config.observation_noise_variance)?,
    )?;
    let selection = MagneticVolumeSensorBank3d::from_feec(
        &topology,
        &coordinates,
        &selection_kernels(C51_FIXED_OBSERVATION_RADIUS)?,
    )?;
    let held_out = MagneticVolumeSensorBank3d::from_feec(
        &topology,
        &coordinates,
        &held_out_kernels(C51_FIXED_OBSERVATION_RADIUS)?,
    )?;

    let exact_posterior = exact.condition(std::slice::from_ref(&observation), &[])?;
    let mut compact_posterior = LinearGaussianModelBuilder::new(compact)
        .observe(observation)?
        .condition()?;
    let exact_selection = exact_posterior.pushforward_covariance(selection.map().map())?;
    let exact_held_out = exact_posterior.pushforward_covariance(held_out.map().map())?;
    let compact_selection = compact_posterior.pushforward_covariance(selection.map().map())?;
    let compact_held_out = compact_posterior.pushforward_covariance(held_out.map().map())?;

    Ok(C51ThesisNormalizedCompactReport {
        metadata,
        config: config.clone(),
        state_dimension: raw_exact.state_dimension(),
        full_cochain_dimension: raw_exact.cochain_dimension(),
        magnetic_output_dimension,
        domain_volume,
        exact: exact_calibration,
        compact: compact_calibration,
        selection: covariance_error_metrics(&compact_selection, &exact_selection)?,
        held_out: covariance_error_metrics(&compact_held_out, &exact_held_out)?,
    })
}

impl C51ThesisNormalizedCompactReport {
    pub fn write_directory(&self, directory: &Path) -> Result<()> {
        fs::create_dir_all(directory)?;
        self.write_csv(&directory.join("summary.csv"))?;
        self.write_manifest(&directory.join("manifest.yaml"))
    }

    fn write_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "schema_version,profile,refinement,state_dimension,observation_count,target_b_rms_tesla,exact_raw_b_rms,exact_precision_scale,exact_tau_multiplier,exact_trace_relative_standard_error,compact_raw_b_rms,compact_precision_scale,compact_tau_multiplier,compact_trace_relative_standard_error,selection_e_f,selection_e_2,selection_e_diag,selection_e_dir,selection_e_corr,selection_e_primary,heldout_e_f,heldout_e_2,heldout_e_diag,heldout_e_dir,heldout_e_corr,heldout_e_primary")?;
        writeln!(writer, "{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
            self.config.profile,
            self.config.refinement,
            self.state_dimension,
            self.config.observation_count,
            self.config.target_b_rms_tesla,
            self.exact.raw_b_rms,
            self.exact.precision_scale,
            self.exact.tau_multiplier,
            option_float(self.exact.weighted_trace_relative_standard_error),
            self.compact.raw_b_rms,
            self.compact.precision_scale,
            self.compact.tau_multiplier,
            option_float(self.compact.weighted_trace_relative_standard_error),
            self.selection.relative_frobenius,
            self.selection.relative_spectral,
            self.selection.maximum_relative_diagonal,
            self.selection.maximum_relative_directional,
            self.selection.relative_correlation_frobenius,
            self.selection.primary_envelope(),
            self.held_out.relative_frobenius,
            self.held_out.relative_spectral,
            self.held_out.maximum_relative_diagonal,
            self.held_out.maximum_relative_directional,
            self.held_out.relative_correlation_frobenius,
            self.held_out.primary_envelope(),
        )?;
        writer.flush()?;
        Ok(())
    }

    fn write_manifest(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version: {C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION}"
        )?;
        writeln!(writer, "study_id: {C51_THESIS_NORMALIZED_COMPACT_STUDY_ID}")?;
        writeln!(writer, "profile: {}", self.config.profile)?;
        writeln!(writer, "normalization:")?;
        writeln!(
            writer,
            "  policy: thesis_domain_averaged_feec_magnetic_trace"
        )?;
        writeln!(writer, "  equation: volume_inverse_trace_M2_D1_Qinv_D1T")?;
        writeln!(writer, "  stage: prior_before_conditioning")?;
        writeln!(
            writer,
            "  target_b_rms_tesla: {:.17e}",
            self.config.target_b_rms_tesla
        )?;
        writeln!(writer, "  sensor_rows_used_for_normalization: false")?;
        writeln!(writer, "  heldout_rows_used_for_normalization: false")?;
        writeln!(writer, "  estimator: deterministic_batched_hutchinson")?;
        writeln!(writer, "  probes: {}", self.config.trace_probes)?;
        writeln!(writer, "  batches: {}", self.config.trace_batches)?;
        writeln!(writer, "  seed: {}", self.config.trace_seed)?;
        writeln!(
            writer,
            "observation_design_id: {C51_JUQ_PERFORMANCE_SENSOR_DESIGN_ID}"
        )?;
        writeln!(
            writer,
            "physical_qoi_design_id: {C51_PERFORMANCE_OUTPUT_DESIGN_ID}"
        )?;
        writeln!(
            writer,
            "observation_count: {}",
            self.config.observation_count
        )?;
        writeln!(
            writer,
            "root_sha: {}",
            option_yaml(self.metadata.root_sha.as_deref())
        )?;
        writeln!(
            writer,
            "feec_sha: {}",
            option_yaml(self.metadata.feec_sha.as_deref())
        )?;
        writeln!(
            writer,
            "gmrf_sha: {}",
            option_yaml(self.metadata.gmrf_sha.as_deref())
        )?;
        writeln!(
            writer,
            "root_dirty: {}",
            option_bool(self.metadata.root_dirty)
        )?;
        writeln!(
            writer,
            "feec_dirty: {}",
            option_bool(self.metadata.feec_dirty)
        )?;
        writeln!(
            writer,
            "gmrf_dirty: {}",
            option_bool(self.metadata.gmrf_dirty)
        )?;
        writeln!(writer, "command: {:?}", self.metadata.command)?;
        writer.flush()?;
        Ok(())
    }
}

fn trace_calibration(
    weighted_trace: f64,
    relative_standard_error: Option<f64>,
    target_b_rms: f64,
    domain_volume: f64,
) -> Result<C51ThesisTraceCalibration> {
    if !weighted_trace.is_finite() || weighted_trace <= 0.0 {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 magnetic weighted prior trace is non-positive or non-finite".to_string(),
        ));
    }
    if !domain_volume.is_finite() || domain_volume <= 0.0 {
        return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
            "C51 magnetic trace calibration requires positive finite domain volume".to_string(),
        ));
    }
    let precision_scale = weighted_trace / (domain_volume * target_b_rms.powi(2));
    Ok(C51ThesisTraceCalibration {
        raw_b_rms: (weighted_trace / domain_volume).sqrt(),
        target_b_rms,
        precision_scale,
        tau_multiplier: precision_scale.sqrt(),
        normalized_b_rms: (weighted_trace / (domain_volume * precision_scale)).sqrt(),
        weighted_trace_relative_standard_error: relative_standard_error,
    })
}

fn option_float(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.17e}"))
        .unwrap_or_default()
}

fn option_yaml(value: Option<&str>) -> String {
    value
        .map(|value| format!("{value:?}"))
        .unwrap_or_else(|| "null".to_string())
}

fn option_bool(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_calibration_hits_target_by_precision_scaling() {
        let calibration = trace_calibration(0.36, Some(0.01), 0.10, 1.0).unwrap();
        assert!((calibration.precision_scale - 36.0).abs() < 1.0e-12);
        assert!((calibration.tau_multiplier - 6.0).abs() < 1.0e-12);
        assert!((calibration.normalized_b_rms - 0.10).abs() < 1.0e-12);
    }
}
