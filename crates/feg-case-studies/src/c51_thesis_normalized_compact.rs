//! C51 projected-Whitney accuracy under the thesis magnetic trace convention.
//!
//! Both the exact supported FEEC law and the compact projected-Whitney law are
//! independently scaled to the same domain-wide RMS of `B = D1 A` in the
//! degree-2 FEEC mass norm. Training rows, local diagnostics, and fixed surface
//! flux QoIs pair directly with the integral 2-cochain; none uses pointwise or
//! cell-barycentre reconstruction, and none defines the amplitude.

use crate::c51_benchmark::C51RunMetadata;
use crate::c51_magnetic_sensor_design::{
    assemble_c51_flux_cochain_split, C51_FLUX_COCHAIN_DESIGN_ID, C51_FLUX_COCHAIN_DESIGN_SEED,
};
use crate::c51_spectral_covariance_rank::{covariance_error_metrics, C51CovarianceErrorMetrics};
use feec_gmrf::prelude::{
    axis_aligned_flux_patch_cochain_map_3d, pullback_quadratic_form, AxisAlignedFluxPatch3d,
    EssentialBoundaryConditions, EssentialBoundaryElimination, ExactFormOperators, GaussianNoise,
    HutchinsonVarianceConfig, LinearGaussianModelBuilder, LinearMap, LinearObservation,
    MaternAlpha, MaternParameters, MaternPriorBuilder, OutgoingFormOperator, Result,
    SupportedMaternPriorBuilder, VarianceMethod,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::{coord::mesh::MeshCoords, coord::simplex::SimplexHandleExt},
    topology::complex::Complex,
};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION: &str =
    "c51-thesis-normalized-flux-cochain-v4";
pub const C51_THESIS_NORMALIZED_COMPACT_STUDY_ID: &str =
    "c51/thesis-normalized-flux-cochain-accuracy";
pub const C51_THESIS_NORMALIZED_COMPACT_PROFILE_N8: &str =
    "thesis-magnetic-trace-normalized-flux-cochain-v4-n8";
pub const C51_THESIS_NORMALIZED_COMPACT_PROFILE_N10: &str =
    "thesis-magnetic-trace-normalized-flux-cochain-v4-n10";
pub const C51_SURFACE_FLUX_QOI_DESIGN_ID: &str = "c51-midplane-quadrant-fluxes-v1";
pub const C51_THESIS_TARGET_B_RMS_TESLA: f64 = 0.10;
pub const C51_FLUX_NOISE_STANDARD_DEVIATION_WEBER: f64 = 5.0e-4;
pub const C51_THESIS_TRACE_PROBES: usize = 512;
pub const C51_THESIS_TRACE_BATCHES: usize = 8;
pub const C51_THESIS_TRACE_SEED: u64 = 0xC51B_7ACE;
pub const C51_THESIS_OBSERVATION_COUNT: usize = 4_096;
pub const C51_THESIS_SELECTION_FLUX_COUNT: usize = 64;
pub const C51_THESIS_HELD_OUT_FLUX_COUNT: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct C51ThesisNormalizedCompactConfig {
    pub profile: String,
    pub refinement: usize,
    pub alpha: MaternAlpha,
    pub kappa: f64,
    pub base_tau: f64,
    pub target_b_rms_tesla: f64,
    pub observation_noise_variance_weber_squared: f64,
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
            observation_noise_variance_weber_squared: C51_FLUX_NOISE_STANDARD_DEVIATION_WEBER
                .powi(2),
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
    pub local_prior_selection: C51CovarianceErrorMetrics,
    pub local_prior_held_out: C51CovarianceErrorMetrics,
    pub local_posterior_selection: C51CovarianceErrorMetrics,
    pub local_posterior_held_out: C51CovarianceErrorMetrics,
    pub surface_prior_selection: C51CovarianceErrorMetrics,
    pub surface_prior_held_out: C51CovarianceErrorMetrics,
    pub surface_posterior_selection: C51CovarianceErrorMetrics,
    pub surface_posterior_held_out: C51CovarianceErrorMetrics,
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
    let flux_cochain = outgoing.active_derivative().clone();
    let magnetic_energy = pullback_quadratic_form(&flux_cochain, outgoing.upper().mass())?;
    let (surface_selection, surface_held_out) =
        c51_surface_flux_qois(&topology, &coordinates, outgoing, &flux_cochain)?;
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

    let flux_split = assemble_c51_flux_cochain_split(
        &flux_cochain,
        config.observation_count,
        C51_THESIS_SELECTION_FLUX_COUNT,
        C51_THESIS_HELD_OUT_FLUX_COUNT,
    )?;
    let observation = LinearObservation::new(
        flux_split.observation().clone(),
        vec![0.0; config.observation_count],
        GaussianNoise::variance(config.observation_noise_variance_weber_squared)?,
    )?;

    let exact_prior = exact.factor()?;
    let mut compact_prior = compact.factor()?;
    let exact_prior_selection = exact_prior.pushforward_covariance(flux_split.selection())?;
    let exact_prior_held_out = exact_prior.pushforward_covariance(flux_split.held_out())?;
    let compact_prior_selection = compact_prior.pushforward_covariance(flux_split.selection())?;
    let compact_prior_held_out = compact_prior.pushforward_covariance(flux_split.held_out())?;
    let exact_surface_prior_selection = exact_prior.pushforward_covariance(&surface_selection)?;
    let exact_surface_prior_held_out = exact_prior.pushforward_covariance(&surface_held_out)?;
    let compact_surface_prior_selection =
        compact_prior.pushforward_covariance(&surface_selection)?;
    let compact_surface_prior_held_out = compact_prior.pushforward_covariance(&surface_held_out)?;

    let exact_posterior = exact.condition(std::slice::from_ref(&observation), &[])?;
    let mut compact_posterior = LinearGaussianModelBuilder::new(compact)
        .observe(observation)?
        .condition()?;
    let exact_selection = exact_posterior.pushforward_covariance(flux_split.selection())?;
    let exact_held_out = exact_posterior.pushforward_covariance(flux_split.held_out())?;
    let compact_selection = compact_posterior.pushforward_covariance(flux_split.selection())?;
    let compact_held_out = compact_posterior.pushforward_covariance(flux_split.held_out())?;
    let exact_surface_selection = exact_posterior.pushforward_covariance(&surface_selection)?;
    let exact_surface_held_out = exact_posterior.pushforward_covariance(&surface_held_out)?;
    let compact_surface_selection = compact_posterior.pushforward_covariance(&surface_selection)?;
    let compact_surface_held_out = compact_posterior.pushforward_covariance(&surface_held_out)?;

    Ok(C51ThesisNormalizedCompactReport {
        metadata,
        config: config.clone(),
        state_dimension: raw_exact.state_dimension(),
        full_cochain_dimension: raw_exact.cochain_dimension(),
        magnetic_output_dimension,
        domain_volume,
        exact: exact_calibration,
        compact: compact_calibration,
        local_prior_selection: covariance_error_metrics(
            &compact_prior_selection,
            &exact_prior_selection,
        )?,
        local_prior_held_out: covariance_error_metrics(
            &compact_prior_held_out,
            &exact_prior_held_out,
        )?,
        local_posterior_selection: covariance_error_metrics(&compact_selection, &exact_selection)?,
        local_posterior_held_out: covariance_error_metrics(&compact_held_out, &exact_held_out)?,
        surface_prior_selection: covariance_error_metrics(
            &compact_surface_prior_selection,
            &exact_surface_prior_selection,
        )?,
        surface_prior_held_out: covariance_error_metrics(
            &compact_surface_prior_held_out,
            &exact_surface_prior_held_out,
        )?,
        surface_posterior_selection: covariance_error_metrics(
            &compact_surface_selection,
            &exact_surface_selection,
        )?,
        surface_posterior_held_out: covariance_error_metrics(
            &compact_surface_held_out,
            &exact_surface_held_out,
        )?,
    })
}

impl C51ThesisNormalizedCompactReport {
    pub fn write_directory(&self, directory: &Path) -> Result<()> {
        fs::create_dir_all(directory)?;
        self.write_summary_csv(&directory.join("summary.csv"))?;
        self.write_error_csv(
            &directory.join("local_flux_cochain_errors.csv"),
            "local_face_flux_cochains",
            [
                ("prior", "selection", &self.local_prior_selection),
                ("prior", "heldout", &self.local_prior_held_out),
                ("posterior", "selection", &self.local_posterior_selection),
                ("posterior", "heldout", &self.local_posterior_held_out),
            ],
        )?;
        self.write_error_csv(
            &directory.join("surface_flux_errors.csv"),
            "midplane_quadrant_surface_fluxes",
            [
                ("prior", "selection", &self.surface_prior_selection),
                ("prior", "heldout", &self.surface_prior_held_out),
                ("posterior", "selection", &self.surface_posterior_selection),
                ("posterior", "heldout", &self.surface_posterior_held_out),
            ],
        )?;
        self.write_manifest(&directory.join("manifest.yaml"))
    }

    fn write_summary_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "schema_version,profile,refinement,state_dimension,observation_count,selection_flux_count,heldout_flux_count,observation_noise_variance_weber_squared,target_b_rms_tesla,exact_raw_b_rms,exact_precision_scale,exact_tau_multiplier,exact_trace_relative_standard_error,compact_raw_b_rms,compact_precision_scale,compact_tau_multiplier,compact_trace_relative_standard_error,local_prior_heldout_primary,local_posterior_heldout_primary,surface_prior_heldout_primary,surface_posterior_heldout_primary")?;
        writeln!(writer, "{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e}",
            C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
            self.config.profile,
            self.config.refinement,
            self.state_dimension,
            self.config.observation_count,
            C51_THESIS_SELECTION_FLUX_COUNT,
            C51_THESIS_HELD_OUT_FLUX_COUNT,
            self.config.observation_noise_variance_weber_squared,
            self.config.target_b_rms_tesla,
            self.exact.raw_b_rms,
            self.exact.precision_scale,
            self.exact.tau_multiplier,
            option_float(self.exact.weighted_trace_relative_standard_error),
            self.compact.raw_b_rms,
            self.compact.precision_scale,
            self.compact.tau_multiplier,
            option_float(self.compact.weighted_trace_relative_standard_error),
            self.local_prior_held_out.primary_envelope(),
            self.local_posterior_held_out.primary_envelope(),
            self.surface_prior_held_out.primary_envelope(),
            self.surface_posterior_held_out.primary_envelope(),
        )?;
        writer.flush()?;
        Ok(())
    }

    fn write_error_csv(
        &self,
        path: &Path,
        qoi_family: &str,
        rows: [(&str, &str, &C51CovarianceErrorMetrics); 4],
    ) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version,profile,qoi_family,stage,split,e_f,e_2,e_diag,e_dir,e_corr,e_primary"
        )?;
        for (stage, split, metrics) in rows {
            writeln!(
                writer,
                "{},{},{},{},{},{}",
                C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
                self.config.profile,
                qoi_family,
                stage,
                split,
                metrics_csv(metrics),
            )?;
        }
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
            "observation_design_id: {C51_FLUX_COCHAIN_DESIGN_ID}"
        )?;
        writeln!(writer, "local_qoi_design_id: {C51_FLUX_COCHAIN_DESIGN_ID}")?;
        writeln!(
            writer,
            "surface_qoi_design_id: {C51_SURFACE_FLUX_QOI_DESIGN_ID}"
        )?;
        writeln!(writer, "cochain_row_semantics: exact_integral_flux_dofs")?;
        writeln!(writer, "cell_barycenter_reconstruction_used: false")?;
        writeln!(writer, "row_order_seed: {C51_FLUX_COCHAIN_DESIGN_SEED}")?;
        writeln!(writer, "state_dimension: {}", self.state_dimension)?;
        writeln!(
            writer,
            "full_cochain_dimension: {}",
            self.full_cochain_dimension
        )?;
        writeln!(
            writer,
            "magnetic_output_dimension: {}",
            self.magnetic_output_dimension
        )?;
        writeln!(
            writer,
            "observation_count: {}",
            self.config.observation_count
        )?;
        writeln!(
            writer,
            "selection_flux_count: {C51_THESIS_SELECTION_FLUX_COUNT}"
        )?;
        writeln!(
            writer,
            "heldout_flux_count: {C51_THESIS_HELD_OUT_FLUX_COUNT}"
        )?;
        writeln!(
            writer,
            "observation_noise_variance_weber_squared: {:.17e}",
            self.config.observation_noise_variance_weber_squared
        )?;
        writeln!(writer, "errors:")?;
        writeln!(
            writer,
            "  local_prior_selection_primary: {:.17e}",
            self.local_prior_selection.primary_envelope()
        )?;
        writeln!(
            writer,
            "  local_prior_heldout_primary: {:.17e}",
            self.local_prior_held_out.primary_envelope()
        )?;
        writeln!(
            writer,
            "  local_posterior_selection_primary: {:.17e}",
            self.local_posterior_selection.primary_envelope()
        )?;
        writeln!(
            writer,
            "  local_posterior_heldout_primary: {:.17e}",
            self.local_posterior_held_out.primary_envelope()
        )?;
        writeln!(
            writer,
            "  surface_prior_selection_primary: {:.17e}",
            self.surface_prior_selection.primary_envelope()
        )?;
        writeln!(
            writer,
            "  surface_prior_heldout_primary: {:.17e}",
            self.surface_prior_held_out.primary_envelope()
        )?;
        writeln!(
            writer,
            "  surface_posterior_selection_primary: {:.17e}",
            self.surface_posterior_selection.primary_envelope()
        )?;
        writeln!(
            writer,
            "  surface_posterior_heldout_primary: {:.17e}",
            self.surface_posterior_held_out.primary_envelope()
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

fn c51_surface_flux_qois(
    topology: &Complex,
    coordinates: &MeshCoords,
    outgoing: &OutgoingFormOperator,
    flux_cochain: &LinearMap,
) -> Result<(LinearMap, LinearMap)> {
    let mut patches = Vec::with_capacity(12);
    for normal_axis in 0..3 {
        for first_half in 0..2 {
            for second_half in 0..2 {
                patches.push(AxisAlignedFluxPatch3d::new(
                    normal_axis,
                    0.5,
                    [
                        [0.5 * first_half as f64, 0.5 * (first_half + 1) as f64],
                        [0.5 * second_half as f64, 0.5 * (second_half + 1) as f64],
                    ],
                )?);
            }
        }
    }
    let full_patch_map = axis_aligned_flux_patch_cochain_map_3d(topology, coordinates, &patches)?;
    let eliminated =
        EssentialBoundaryElimination::from_layout(outgoing.upper().boundary_layout().clone())?
            .eliminate_map(&full_patch_map, &vec![0.0; patches.len()])?;
    if eliminated
        .fixed_offset()
        .iter()
        .any(|offset| *offset != 0.0)
    {
        return Err(feec_gmrf::FeecGmrfError::Assembly(
            "C51 homogeneous surface-flux QoIs acquired a nonzero boundary offset".to_string(),
        ));
    }
    let state_patch_map = eliminated.reduced().compose(flux_cochain)?;
    let selection = state_patch_map.select_outputs(&[0, 2, 4, 6, 8, 10])?;
    let held_out = state_patch_map.select_outputs(&[1, 3, 5, 7, 9, 11])?;
    Ok((selection, held_out))
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

fn metrics_csv(metrics: &C51CovarianceErrorMetrics) -> String {
    format!(
        "{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
        metrics.relative_frobenius,
        metrics.relative_spectral,
        metrics.maximum_relative_diagonal,
        metrics.maximum_relative_directional,
        metrics.relative_correlation_frobenius,
        metrics.primary_envelope(),
    )
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

    #[test]
    fn surface_flux_qois_compose_exact_face_integrals_with_active_d1() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
        let (topology, coordinates) = mesh.compute_coord_complex();
        let metric = coordinates.to_edge_lengths(&topology);
        let operators =
            ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1).unwrap();
        let outgoing = operators.outgoing().unwrap();
        let flux_cochain = outgoing.active_derivative().clone();
        let (selection, held_out) =
            c51_surface_flux_qois(&topology, &coordinates, outgoing, &flux_cochain).unwrap();

        assert_eq!(selection.output_dimension(), 6);
        assert_eq!(held_out.output_dimension(), 6);
        assert_eq!(selection.input_dimension(), flux_cochain.input_dimension());
        assert_eq!(held_out.input_dimension(), flux_cochain.input_dimension());
    }
}
