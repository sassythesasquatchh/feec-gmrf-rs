//! C51 projected-Whitney accuracy under the thesis magnetic trace convention.
//!
//! Both the exact supported FEEC law and the compact projected-Whitney law are
//! independently scaled to the same domain-wide RMS of `B = D1 A` in the
//! degree-2 FEEC mass norm. Training rows, local diagnostics, and fixed surface
//! flux QoIs pair directly with the integral 2-cochain; none uses pointwise or
//! cell-barycentre reconstruction. This primary development path deliberately
//! applies no output-family-specific posterior amplitude calibration.

use crate::c51_magnetic_sensor_design::{
    assemble_c51_flux_cochain_split, assemble_c51_surface_flux_split,
    C51_FLUX_COCHAIN_HELD_OUT_COUNT, C51_FLUX_COCHAIN_SELECTION_COUNT,
};
use common::linalg::nalgebra::{Matrix, Vector as FeecVector};
use feec_gmrf::prelude::{
    pullback_quadratic_form, AdjacentMaternPriorBuilder, EssentialBoundaryElimination,
    ExactFormOperators, GaussianNoise, HutchinsonVarianceConfig, LinearGaussianModelBuilder,
    LinearMap, LinearObservation, LinearPdeSystem, MaternAlpha, MaternParameters, PdeResidualNoise,
    Result, SparseMat, SupportedMaternPriorBuilder, VarianceMethod,
};
use formoniq::{
    problems::reduced_linear::build_reduced_hodge_laplace_1form_system,
    reduction::{EssentialBoundarySpec, PrescribedDof},
};
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::simplex::SimplexHandleExt,
    topology::complex::Complex,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION: &str =
    "c51-development-thesis-normalized-flux-cochain-v8";
pub const C51_THESIS_TARGET_B_RMS_TESLA: f64 = 0.10;
pub const C51_FLUX_NOISE_STANDARD_DEVIATION_WEBER: f64 = 5.0e-4;
pub const C51_THESIS_TRACE_PROBES: usize = 512;
pub const C51_THESIS_TRACE_BATCHES: usize = 8;
pub const C51_THESIS_TRACE_SEED: u64 = 0xC51B_7ACE;
pub const C51_THESIS_SELECTION_FLUX_COUNT: usize = C51_FLUX_COCHAIN_SELECTION_COUNT;
pub const C51_THESIS_HELD_OUT_FLUX_COUNT: usize = C51_FLUX_COCHAIN_HELD_OUT_COUNT;
const C51_VECTOR_OUTPUT_DIMENSION: usize = 9;
const C51_REFERENCE_VARIANCE_RELATIVE_FLOOR: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C51CovarianceErrorMetrics {
    pub relative_frobenius: f64,
    pub relative_spectral: f64,
    pub maximum_relative_diagonal: f64,
    pub maximum_relative_directional: f64,
    pub relative_correlation_frobenius: f64,
}

impl C51CovarianceErrorMetrics {
    pub fn primary_envelope(self) -> f64 {
        self.relative_frobenius
            .max(self.maximum_relative_diagonal)
            .max(self.maximum_relative_directional)
    }
}

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
    pub pde_residual_standard_deviations: Vec<f64>,
}

impl C51ThesisNormalizedCompactConfig {
    pub fn development(refinement: usize, observation_count: usize) -> Result<Self> {
        if refinement < 2 {
            return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
                "C51 development refinement must be at least two".to_string(),
            ));
        }
        if observation_count == 0 {
            return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
                "C51 development observation count must be positive".to_string(),
            ));
        }
        Ok(Self {
            profile: format!("development-thesis-normalized-n{refinement}-m{observation_count}"),
            refinement,
            alpha: MaternAlpha::Two,
            kappa: 4.0,
            base_tau: 1.0,
            target_b_rms_tesla: C51_THESIS_TARGET_B_RMS_TESLA,
            observation_noise_variance_weber_squared: C51_FLUX_NOISE_STANDARD_DEVIATION_WEBER
                .powi(2),
            observation_count,
            trace_probes: C51_THESIS_TRACE_PROBES,
            trace_batches: C51_THESIS_TRACE_BATCHES,
            trace_seed: C51_THESIS_TRACE_SEED,
            pde_residual_standard_deviations: vec![1.0e-1, 1.0e-3, 1.0e-5],
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
    pub weak_residual_manufactured_deterministic_relative_l2_error: f64,
    pub weak_residual_rows: Vec<C51WeakResidualDevelopmentRow>,
    pub latent_field_wide: C51FieldWideVarianceReport,
    pub flux_field_wide: C51FieldWideVarianceReport,
    pub covariance_action_errors: C51FieldWideCovarianceActionErrors,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C51WeakResidualDevelopmentRow {
    pub residual_standard_deviation: f64,
    pub exact_augmented_deterministic_relative_l2_error: f64,
    pub compact_deterministic_relative_l2_error: f64,
    pub compact_exact_mean_relative_l2_error: f64,
    pub exact_augmented_residual_dual_norm: f64,
    pub compact_residual_dual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct C51FieldWideVarianceReport {
    pub output_kind: String,
    pub variance_units: String,
    pub rows: Vec<C51FieldWideVarianceRow>,
    pub prior_variance_error: C51FieldWideErrorSummary,
    pub posterior_variance_error: C51FieldWideErrorSummary,
    pub posterior_prior_ratio_error: C51FieldWideErrorSummary,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C51FieldWideVarianceRow {
    pub index: usize,
    pub exact_prior_variance: f64,
    pub compact_prior_variance: f64,
    pub exact_posterior_variance: f64,
    pub compact_posterior_variance: f64,
    pub exact_posterior_prior_ratio: f64,
    pub compact_posterior_prior_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C51FieldWideErrorSummary {
    pub count: usize,
    pub relative_error_floor: f64,
    pub near_zero_reference_count: usize,
    pub median_relative_error: f64,
    pub percentile_90_relative_error: f64,
    pub percentile_99_relative_error: f64,
    pub maximum_relative_error: f64,
    pub median_absolute_error: f64,
    pub percentile_90_absolute_error: f64,
    pub percentile_99_absolute_error: f64,
    pub maximum_absolute_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C51FieldWideCovarianceActionErrors {
    pub seed: u64,
    pub probe_count: usize,
    pub latent_prior_maximum_relative_mass_error: f64,
    pub latent_posterior_maximum_relative_mass_error: f64,
    pub flux_prior_maximum_relative_energy_error: f64,
    pub flux_posterior_maximum_relative_energy_error: f64,
}

pub fn run_c51_thesis_normalized_compact(
    config: &C51ThesisNormalizedCompactConfig,
) -> Result<C51ThesisNormalizedCompactReport> {
    if config.pde_residual_standard_deviations.is_empty()
        || config
            .pde_residual_standard_deviations
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(feec_gmrf::FeecGmrfError::InvalidParameter(
            "C51 weak-residual standard deviations must be non-empty, finite, and positive"
                .to_string(),
        ));
    }
    let mesh = CartesianMeshInfo::new_unit_scaled(3, config.refinement, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let operators = ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1)?;
    let parameters = MaternParameters::new(config.alpha, config.kappa, config.base_tau)?;
    let raw_exact = SupportedMaternPriorBuilder::from_operators(operators.clone())
        .parameters(parameters)
        .build()?;
    let raw_compact =
        AdjacentMaternPriorBuilder::from_feec_homogeneous_full_relative_with_projected_inverses(
            &topology, &metric, 1,
        )?
        .parameters(parameters)
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
    let state_mass = operators.state().mass().clone();
    let flux_mass = outgoing.upper().mass().clone();
    let flux_cochain = outgoing.active_derivative().clone();
    let magnetic_energy = pullback_quadratic_form(&flux_cochain, outgoing.upper().mass())?;
    let flux_elimination =
        EssentialBoundaryElimination::from_layout(outgoing.upper().boundary_layout().clone())?;
    let surface_split =
        assemble_c51_surface_flux_split(&topology, &coordinates, &flux_elimination, &flux_cochain)?;
    let surface_selection = surface_split.selection().clone();
    let surface_held_out = surface_split.held_out().clone();
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
    let mut compact_posterior = LinearGaussianModelBuilder::new(compact.clone())
        .observe(observation.clone())?
        .condition()?;
    let exact_selection = exact_posterior.pushforward_covariance(flux_split.selection())?;
    let exact_held_out = exact_posterior.pushforward_covariance(flux_split.held_out())?;
    let compact_selection = compact_posterior.pushforward_covariance(flux_split.selection())?;
    let compact_held_out = compact_posterior.pushforward_covariance(flux_split.held_out())?;
    let exact_surface_selection = exact_posterior.pushforward_covariance(&surface_selection)?;
    let exact_surface_held_out = exact_posterior.pushforward_covariance(&surface_held_out)?;
    let compact_surface_selection = compact_posterior.pushforward_covariance(&surface_selection)?;
    let compact_surface_held_out = compact_posterior.pushforward_covariance(&surface_held_out)?;

    let identity = LinearMap::identity(raw_exact.state_dimension());
    let exact_latent_prior_variances = exact_prior.pushforward_variances(&identity)?;
    let compact_latent_prior_variances = compact_prior
        .pushforward_variance_estimate(&identity, VarianceMethod::Exact)?
        .values;
    let exact_latent_posterior_variances = exact_posterior.pushforward_variances(&identity)?;
    let compact_latent_posterior_variances = compact_posterior
        .pushforward_variance_estimate(&identity, VarianceMethod::Exact)?
        .values;
    let latent_field_wide = build_field_wide_variance_report(
        "active_1form_cochain",
        "active_1form_coefficient_squared",
        &exact_latent_prior_variances,
        &compact_latent_prior_variances,
        &exact_latent_posterior_variances,
        &compact_latent_posterior_variances,
    )?;

    let exact_flux_prior_variances = exact_prior.pushforward_variances(&flux_cochain)?;
    let compact_flux_prior_variances = compact_prior
        .pushforward_variance_estimate(&flux_cochain, VarianceMethod::Exact)?
        .values;
    let exact_flux_posterior_variances = exact_posterior.pushforward_variances(&flux_cochain)?;
    let compact_flux_posterior_variances = compact_posterior
        .pushforward_variance_estimate(&flux_cochain, VarianceMethod::Exact)?
        .values;
    let flux_field_wide = build_field_wide_variance_report(
        "active_d1_flux_cochain",
        "weber_squared",
        &exact_flux_prior_variances,
        &compact_flux_prior_variances,
        &exact_flux_posterior_variances,
        &compact_flux_posterior_variances,
    )?;

    let covariance_action_probe_count = 4;
    let covariance_action_seed = config.trace_seed ^ 0xF13D_A11C;
    let mut covariance_rng = StdRng::seed_from_u64(covariance_action_seed);
    let mut latent_prior_maximum_relative_mass_error = 0.0_f64;
    let mut latent_posterior_maximum_relative_mass_error = 0.0_f64;
    let mut flux_prior_maximum_relative_energy_error = 0.0_f64;
    let mut flux_posterior_maximum_relative_energy_error = 0.0_f64;
    for _ in 0..covariance_action_probe_count {
        let latent_probe = rademacher_probe(raw_exact.state_dimension(), &mut covariance_rng);
        let exact_prior_action = exact_prior.covariance_action(&latent_probe)?;
        let compact_prior_action = compact_prior.covariance_action(&latent_probe)?;
        latent_prior_maximum_relative_mass_error = latent_prior_maximum_relative_mass_error.max(
            metric_relative_vector_error(&compact_prior_action, &exact_prior_action, &state_mass)?,
        );
        let exact_posterior_action = exact_posterior.covariance_action(&latent_probe)?;
        let compact_posterior_action = compact_posterior.covariance_action(&latent_probe)?;
        latent_posterior_maximum_relative_mass_error = latent_posterior_maximum_relative_mass_error
            .max(metric_relative_vector_error(
                &compact_posterior_action,
                &exact_posterior_action,
                &state_mass,
            )?);

        let flux_probe = rademacher_probe(flux_cochain.output_dimension(), &mut covariance_rng);
        let flux_rhs = flux_cochain.apply_transpose(&flux_probe)?;
        let exact_prior_flux_action =
            flux_cochain.apply(&exact_prior.covariance_action(&flux_rhs)?)?;
        let compact_prior_flux_action =
            flux_cochain.apply(&compact_prior.covariance_action(&flux_rhs)?)?;
        flux_prior_maximum_relative_energy_error =
            flux_prior_maximum_relative_energy_error.max(metric_relative_vector_error(
                &compact_prior_flux_action,
                &exact_prior_flux_action,
                &flux_mass,
            )?);
        let exact_posterior_flux_action =
            flux_cochain.apply(&exact_posterior.covariance_action(&flux_rhs)?)?;
        let compact_posterior_flux_action =
            flux_cochain.apply(&compact_posterior.covariance_action(&flux_rhs)?)?;
        flux_posterior_maximum_relative_energy_error = flux_posterior_maximum_relative_energy_error
            .max(metric_relative_vector_error(
                &compact_posterior_flux_action,
                &exact_posterior_flux_action,
                &flux_mass,
            )?);
    }
    let covariance_action_errors = C51FieldWideCovarianceActionErrors {
        seed: covariance_action_seed,
        probe_count: covariance_action_probe_count,
        latent_prior_maximum_relative_mass_error,
        latent_posterior_maximum_relative_mass_error,
        flux_prior_maximum_relative_energy_error,
        flux_posterior_maximum_relative_energy_error,
    };

    let boundary = homogeneous_full_relative_boundary(&topology);
    let pde_assembly = build_reduced_hodge_laplace_1form_system(&topology, &metric, &boundary)
        .map_err(feec_gmrf::FeecGmrfError::Assembly)?;
    if pde_assembly.state_dimension() != raw_exact.state_dimension() {
        return Err(feec_gmrf::FeecGmrfError::Dimension(format!(
            "C51 weak PDE dimension {} differs from prior state dimension {}",
            pde_assembly.state_dimension(),
            raw_exact.state_dimension()
        )));
    }
    let manufactured_state = deterministic_manufactured_state(pde_assembly.state_dimension());
    let rhs = &pde_assembly.operator * FeecVector::from_vec(manufactured_state.clone())
        + &pde_assembly.residual_bias;
    let pde_system = LinearPdeSystem::from_reduced_assembly(pde_assembly)?
        .with_form_space(1, 3)?
        .with_right_hand_side(rhs.as_slice())?;
    let deterministic = pde_system.solve_deterministic()?;
    let weak_residual_manufactured_deterministic_relative_l2_error =
        pde_system.relative_state_l2_error(deterministic.active(), &manufactured_state)?;
    let mut weak_residual_rows = Vec::with_capacity(config.pde_residual_standard_deviations.len());
    for &residual_standard_deviation in &config.pde_residual_standard_deviations {
        let noise =
            PdeResidualNoise::mass_weighted_l2_standard_deviation(residual_standard_deviation)?;
        let observation = pde_system.weak_residual_observation(noise)?;
        let exact_pde_posterior = exact.condition(std::slice::from_ref(&observation), &[])?;
        let compact_pde_posterior = LinearGaussianModelBuilder::new(compact.clone())
            .observe(observation.clone())?
            .condition()?;
        weak_residual_rows.push(C51WeakResidualDevelopmentRow {
            residual_standard_deviation,
            exact_augmented_deterministic_relative_l2_error: pde_system
                .relative_state_l2_error(exact_pde_posterior.mean(), deterministic.active())?,
            compact_deterministic_relative_l2_error: pde_system
                .relative_state_l2_error(compact_pde_posterior.mean(), deterministic.active())?,
            compact_exact_mean_relative_l2_error: pde_system.relative_state_l2_error(
                compact_pde_posterior.mean(),
                exact_pde_posterior.mean(),
            )?,
            exact_augmented_residual_dual_norm: residual_dual_norm(
                &pde_system,
                exact_pde_posterior.mean(),
            )?,
            compact_residual_dual_norm: residual_dual_norm(
                &pde_system,
                compact_pde_posterior.mean(),
            )?,
        });
    }

    Ok(C51ThesisNormalizedCompactReport {
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
        weak_residual_manufactured_deterministic_relative_l2_error,
        weak_residual_rows,
        latent_field_wide,
        flux_field_wide,
        covariance_action_errors,
    })
}

impl C51ThesisNormalizedCompactReport {
    pub fn write_directory(&self, directory: &Path) -> Result<()> {
        fs::create_dir_all(directory)?;
        self.write_summary_csv(&directory.join("summary.csv"))?;
        self.write_error_csv(
            &directory.join("local_flux_cochain_errors.csv"),
            "local_face_flux_cochains",
            &[
                ("prior", "selection", &self.local_prior_selection),
                ("prior", "heldout", &self.local_prior_held_out),
                ("posterior", "selection", &self.local_posterior_selection),
                ("posterior", "heldout", &self.local_posterior_held_out),
            ],
        )?;
        self.write_error_csv(
            &directory.join("surface_flux_errors.csv"),
            "midplane_quadrant_surface_fluxes",
            &[
                ("prior", "selection", &self.surface_prior_selection),
                ("prior", "heldout", &self.surface_prior_held_out),
                ("posterior", "selection", &self.surface_posterior_selection),
                ("posterior", "heldout", &self.surface_posterior_held_out),
            ],
        )?;
        self.write_weak_residual_csv(&directory.join("weak_residual_pde.csv"))?;
        self.write_field_wide_variances_csv(&directory.join("field_wide_variances.csv"))?;
        self.write_field_wide_summary_csv(&directory.join("field_wide_accuracy_summary.csv"))?;
        self.write_covariance_action_errors_csv(
            &directory.join("field_wide_covariance_actions.csv"),
        )?;
        Ok(())
    }

    fn write_summary_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "schema_version,profile,refinement,state_dimension,observation_count,observation_to_state_ratio,selection_flux_count,heldout_flux_count,observation_noise_variance_weber_squared,target_b_rms_tesla,exact_raw_b_rms,exact_normalized_b_rms,exact_precision_scale,exact_tau_multiplier,exact_trace_relative_standard_error,compact_raw_b_rms,compact_normalized_b_rms,compact_precision_scale,compact_tau_multiplier,compact_trace_relative_standard_error,local_prior_heldout_primary,local_posterior_heldout_primary,surface_prior_heldout_primary,surface_posterior_heldout_primary")?;
        writeln!(writer, "{},{},{},{},{},{:.17e},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e}",
            C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
            self.config.profile,
            self.config.refinement,
            self.state_dimension,
            self.config.observation_count,
            self.config.observation_count as f64 / self.state_dimension as f64,
            C51_THESIS_SELECTION_FLUX_COUNT,
            C51_THESIS_HELD_OUT_FLUX_COUNT,
            self.config.observation_noise_variance_weber_squared,
            self.config.target_b_rms_tesla,
            self.exact.raw_b_rms,
            self.exact.normalized_b_rms,
            self.exact.precision_scale,
            self.exact.tau_multiplier,
            option_float(self.exact.weighted_trace_relative_standard_error),
            self.compact.raw_b_rms,
            self.compact.normalized_b_rms,
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
        rows: &[(&str, &str, &C51CovarianceErrorMetrics)],
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

    fn write_weak_residual_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version,profile,refinement,state_dimension,pde_operator,test_space_dual_weight,manufactured_deterministic_relative_l2_error,residual_standard_deviation,exact_augmented_deterministic_relative_l2_error,compact_deterministic_relative_l2_error,compact_exact_mean_relative_l2_error,exact_augmented_residual_dual_norm,compact_residual_dual_norm"
        )?;
        for row in &self.weak_residual_rows {
            writeln!(
                writer,
                "{},{},{},{},reduced_hodge_laplace_1form,sigma^-2_projected_sparse_M_test^-1,{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
                self.config.profile,
                self.config.refinement,
                self.state_dimension,
                self.weak_residual_manufactured_deterministic_relative_l2_error,
                row.residual_standard_deviation,
                row.exact_augmented_deterministic_relative_l2_error,
                row.compact_deterministic_relative_l2_error,
                row.compact_exact_mean_relative_l2_error,
                row.exact_augmented_residual_dual_norm,
                row.compact_residual_dual_norm,
            )?;
        }
        writer.flush()?;
        Ok(())
    }

    fn write_field_wide_variances_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version,profile,refinement,output_kind,variance_units,index,exact_prior_variance,compact_prior_variance,exact_posterior_variance,compact_posterior_variance,exact_posterior_prior_ratio,compact_posterior_prior_ratio"
        )?;
        for report in [&self.latent_field_wide, &self.flux_field_wide] {
            for row in &report.rows {
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                    C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
                    self.config.profile,
                    self.config.refinement,
                    report.output_kind,
                    report.variance_units,
                    row.index,
                    row.exact_prior_variance,
                    row.compact_prior_variance,
                    row.exact_posterior_variance,
                    row.compact_posterior_variance,
                    row.exact_posterior_prior_ratio,
                    row.compact_posterior_prior_ratio,
                )?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_field_wide_summary_csv(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version,profile,refinement,output_kind,variance_units,quantity,count,relative_error_floor,near_zero_reference_count,median_relative_error,percentile_90_relative_error,percentile_99_relative_error,maximum_relative_error,median_absolute_error,percentile_90_absolute_error,percentile_99_absolute_error,maximum_absolute_error"
        )?;
        for report in [&self.latent_field_wide, &self.flux_field_wide] {
            for (quantity, summary) in [
                ("prior_variance", &report.prior_variance_error),
                ("posterior_variance", &report.posterior_variance_error),
                (
                    "posterior_prior_variance_ratio",
                    &report.posterior_prior_ratio_error,
                ),
            ] {
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                    C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
                    self.config.profile,
                    self.config.refinement,
                    report.output_kind,
                    report.variance_units,
                    quantity,
                    summary.count,
                    summary.relative_error_floor,
                    summary.near_zero_reference_count,
                    summary.median_relative_error,
                    summary.percentile_90_relative_error,
                    summary.percentile_99_relative_error,
                    summary.maximum_relative_error,
                    summary.median_absolute_error,
                    summary.percentile_90_absolute_error,
                    summary.percentile_99_absolute_error,
                    summary.maximum_absolute_error,
                )?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_covariance_action_errors_csv(&self, path: &Path) -> Result<()> {
        let errors = self.covariance_action_errors;
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "schema_version,profile,refinement,seed,probe_count,latent_prior_maximum_relative_mass_error,latent_posterior_maximum_relative_mass_error,flux_prior_maximum_relative_energy_error,flux_posterior_maximum_relative_energy_error"
        )?;
        writeln!(
            writer,
            "{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e}",
            C51_THESIS_NORMALIZED_COMPACT_SCHEMA_VERSION,
            self.config.profile,
            self.config.refinement,
            errors.seed,
            errors.probe_count,
            errors.latent_prior_maximum_relative_mass_error,
            errors.latent_posterior_maximum_relative_mass_error,
            errors.flux_prior_maximum_relative_energy_error,
            errors.flux_posterior_maximum_relative_energy_error,
        )?;
        writer.flush()?;
        Ok(())
    }
}

fn build_field_wide_variance_report(
    output_kind: &str,
    variance_units: &str,
    exact_prior: &[f64],
    compact_prior: &[f64],
    exact_posterior: &[f64],
    compact_posterior: &[f64],
) -> Result<C51FieldWideVarianceReport> {
    let dimension = exact_prior.len();
    if dimension == 0
        || compact_prior.len() != dimension
        || exact_posterior.len() != dimension
        || compact_posterior.len() != dimension
    {
        return Err(feec_gmrf::FeecGmrfError::Dimension(
            "C51 field-wide variance vectors must be non-empty and aligned".to_string(),
        ));
    }
    if exact_prior
        .iter()
        .chain(compact_prior)
        .any(|value| !value.is_finite() || *value <= 0.0)
        || exact_posterior
            .iter()
            .chain(compact_posterior)
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 exact field-wide variances must be finite with proper positive priors and non-negative posteriors"
                .to_string(),
        ));
    }
    let exact_ratio = exact_posterior
        .iter()
        .zip(exact_prior)
        .map(|(posterior, prior)| posterior / prior)
        .collect::<Vec<_>>();
    let compact_ratio = compact_posterior
        .iter()
        .zip(compact_prior)
        .map(|(posterior, prior)| posterior / prior)
        .collect::<Vec<_>>();
    let rows = (0..dimension)
        .map(|index| C51FieldWideVarianceRow {
            index,
            exact_prior_variance: exact_prior[index],
            compact_prior_variance: compact_prior[index],
            exact_posterior_variance: exact_posterior[index],
            compact_posterior_variance: compact_posterior[index],
            exact_posterior_prior_ratio: exact_ratio[index],
            compact_posterior_prior_ratio: compact_ratio[index],
        })
        .collect();
    Ok(C51FieldWideVarianceReport {
        output_kind: output_kind.to_string(),
        variance_units: variance_units.to_string(),
        rows,
        prior_variance_error: field_wide_error_summary(compact_prior, exact_prior)?,
        posterior_variance_error: field_wide_error_summary(compact_posterior, exact_posterior)?,
        posterior_prior_ratio_error: field_wide_error_summary(&compact_ratio, &exact_ratio)?,
    })
}

pub(crate) fn field_wide_error_summary(
    candidate: &[f64],
    reference: &[f64],
) -> Result<C51FieldWideErrorSummary> {
    if candidate.is_empty() || candidate.len() != reference.len() {
        return Err(feec_gmrf::FeecGmrfError::Dimension(
            "C51 field-wide error vectors must be non-empty and aligned".to_string(),
        ));
    }
    let maximum_reference = reference
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let relative_error_floor = (1.0e-12 * maximum_reference).max(f64::MIN_POSITIVE);
    let near_zero_reference_count = reference
        .iter()
        .filter(|value| value.abs() < relative_error_floor)
        .count();
    let mut absolute_errors = candidate
        .iter()
        .zip(reference)
        .map(|(candidate, reference)| (candidate - reference).abs())
        .collect::<Vec<_>>();
    let mut relative_errors = absolute_errors
        .iter()
        .zip(reference)
        .map(|(absolute, reference)| absolute / reference.abs().max(relative_error_floor))
        .collect::<Vec<_>>();
    absolute_errors.sort_by(f64::total_cmp);
    relative_errors.sort_by(f64::total_cmp);
    Ok(C51FieldWideErrorSummary {
        count: candidate.len(),
        relative_error_floor,
        near_zero_reference_count,
        median_relative_error: sorted_percentile(&relative_errors, 0.50),
        percentile_90_relative_error: sorted_percentile(&relative_errors, 0.90),
        percentile_99_relative_error: sorted_percentile(&relative_errors, 0.99),
        maximum_relative_error: *relative_errors.last().expect("non-empty error vector"),
        median_absolute_error: sorted_percentile(&absolute_errors, 0.50),
        percentile_90_absolute_error: sorted_percentile(&absolute_errors, 0.90),
        percentile_99_absolute_error: sorted_percentile(&absolute_errors, 0.99),
        maximum_absolute_error: *absolute_errors.last().expect("non-empty error vector"),
    })
}

fn sorted_percentile(sorted: &[f64], probability: f64) -> f64 {
    let index = (probability * (sorted.len() - 1) as f64).ceil() as usize;
    sorted[index]
}

pub(crate) fn rademacher_probe<R: Rng + ?Sized>(dimension: usize, rng: &mut R) -> Vec<f64> {
    let normalization = (dimension as f64).sqrt();
    (0..dimension)
        .map(|_| (if rng.gen::<bool>() { 1.0 } else { -1.0 }) / normalization)
        .collect()
}

pub(crate) fn metric_relative_vector_error(
    candidate: &[f64],
    reference: &[f64],
    metric: &SparseMat,
) -> Result<f64> {
    if candidate.len() != reference.len()
        || metric.nrows() != reference.len()
        || metric.ncols() != reference.len()
    {
        return Err(feec_gmrf::FeecGmrfError::Dimension(
            "C51 metric covariance-action comparison dimensions do not align".to_string(),
        ));
    }
    let difference = candidate
        .iter()
        .zip(reference)
        .map(|(candidate, reference)| candidate - reference)
        .collect::<Vec<_>>();
    let numerator = metric_vector_norm(&difference, metric)?;
    let denominator = metric_vector_norm(reference, metric)?;
    if denominator == 0.0 {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 exact covariance action has zero metric norm".to_string(),
        ));
    }
    Ok(numerator / denominator)
}

fn metric_vector_norm(vector: &[f64], metric: &SparseMat) -> Result<f64> {
    let weighted = metric
        .apply_checked(vector)
        .map_err(feec_gmrf::FeecGmrfError::Dimension)?;
    let squared = vector
        .iter()
        .zip(weighted)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if !squared.is_finite() || squared < -1.0e-12 {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 covariance action produced an invalid metric norm".to_string(),
        ));
    }
    Ok(squared.max(0.0).sqrt())
}

pub(crate) fn homogeneous_full_relative_boundary(topology: &Complex) -> EssentialBoundarySpec {
    EssentialBoundarySpec {
        state: homogeneous_boundary_dofs(topology, 1),
        auxiliary: homogeneous_boundary_dofs(topology, 0),
    }
}

fn homogeneous_boundary_dofs(topology: &Complex, grade: usize) -> Vec<PrescribedDof> {
    topology
        .boundary_subcomplex_simplices(grade)
        .into_iter()
        .map(|simplex| PrescribedDof {
            index: simplex.kidx,
            value: 0.0,
        })
        .collect()
}

pub(crate) fn deterministic_manufactured_state(dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|index| {
            let coordinate = index as f64 + 1.0;
            (0.173 * coordinate).sin() + 0.35 * (0.071 * coordinate).cos()
        })
        .collect()
}

pub(crate) fn residual_dual_norm(system: &LinearPdeSystem, state: &[f64]) -> Result<f64> {
    let residual = system.residual_quantity("weak_residual")?;
    let mut values = residual.operator().apply(state)?;
    for (value, bias) in values.iter_mut().zip(residual.bias()) {
        *value += bias;
    }
    let inverse = system
        .assembly()
        .state_mass_inverse
        .as_ref()
        .ok_or_else(|| {
            feec_gmrf::FeecGmrfError::Inference(
                "C51 weak-residual dual norm requires the test-mass inverse".to_string(),
            )
        })?;
    let weighted = LinearMap::from_feec_csr(inverse)?.apply(&values)?;
    let squared = values
        .iter()
        .zip(weighted)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if !squared.is_finite() || squared < -1.0e-12 {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 weak-residual dual norm is non-finite or negative".to_string(),
        ));
    }
    Ok(squared.max(0.0).sqrt())
}

pub(crate) fn trace_calibration(
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

pub fn covariance_error_metrics(
    approximation: &[Vec<f64>],
    reference: &[Vec<f64>],
) -> Result<C51CovarianceErrorMetrics> {
    validate_covariance_rows(approximation)?;
    validate_covariance_rows(reference)?;
    if approximation.len() != reference.len() {
        return Err(feec_gmrf::FeecGmrfError::Dimension(
            "C51 covariance matrices have different dimensions".to_string(),
        ));
    }
    let maximum_reference_variance = reference
        .iter()
        .enumerate()
        .map(|(index, row)| row[index])
        .fold(0.0, f64::max);
    if !maximum_reference_variance.is_finite() || maximum_reference_variance <= 0.0 {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 reference covariance has no positive finite diagonal".to_string(),
        ));
    }
    for (index, row) in reference.iter().enumerate() {
        if row[index] <= C51_REFERENCE_VARIANCE_RELATIVE_FLOOR * maximum_reference_variance {
            return Err(feec_gmrf::FeecGmrfError::Inference(format!(
                "C51 reference variance {index} is materially near zero"
            )));
        }
    }
    let dimension = reference.len();
    let reference_matrix = covariance_rows_to_matrix(reference);
    let approximation_matrix = covariance_rows_to_matrix(approximation);
    let difference = &approximation_matrix - &reference_matrix;
    let reference_frobenius = reference_matrix.norm();
    let reference_spectral = covariance_spectral_norm(&reference_matrix);
    if reference_frobenius == 0.0 || reference_spectral == 0.0 {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 covariance reference norm is zero".to_string(),
        ));
    }
    let maximum_relative_diagonal = (0..dimension)
        .map(|index| {
            (approximation[index][index] - reference[index][index]).abs() / reference[index][index]
        })
        .fold(0.0, f64::max);
    let maximum_relative_directional = covariance_metric_directions(dimension)
        .into_iter()
        .map(|direction| {
            let reference_value = covariance_quadratic(reference, &direction);
            if !reference_value.is_finite() || reference_value <= 0.0 {
                return Err(feec_gmrf::FeecGmrfError::Inference(
                    "C51 directional reference covariance is non-positive".to_string(),
                ));
            }
            Ok(
                (covariance_quadratic(approximation, &direction) - reference_value).abs()
                    / reference_value,
            )
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(0.0, f64::max);
    let reference_correlation = covariance_correlation_matrix(reference)?;
    let approximation_correlation = covariance_correlation_matrix(approximation)?;
    let correlation_difference = &approximation_correlation - &reference_correlation;
    let metrics = C51CovarianceErrorMetrics {
        relative_frobenius: difference.norm() / reference_frobenius,
        relative_spectral: covariance_spectral_norm(&difference) / reference_spectral,
        maximum_relative_diagonal,
        maximum_relative_directional,
        relative_correlation_frobenius: correlation_difference.norm()
            / reference_correlation.norm(),
    };
    if [
        metrics.relative_frobenius,
        metrics.relative_spectral,
        metrics.maximum_relative_diagonal,
        metrics.maximum_relative_directional,
        metrics.relative_correlation_frobenius,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(feec_gmrf::FeecGmrfError::Inference(
            "C51 covariance metric is non-finite or negative".to_string(),
        ));
    }
    Ok(metrics)
}

fn validate_covariance_rows(covariance: &[Vec<f64>]) -> Result<()> {
    if covariance.is_empty()
        || covariance
            .iter()
            .any(|row| row.len() != covariance.len() || row.iter().any(|value| !value.is_finite()))
    {
        return Err(feec_gmrf::FeecGmrfError::Dimension(
            "C51 covariance must be nonempty, square, and finite".to_string(),
        ));
    }
    Ok(())
}

fn covariance_metric_directions(dimension: usize) -> Vec<Vec<f64>> {
    let mut directions = (0..dimension)
        .map(|index| {
            let mut direction = vec![0.0; dimension];
            direction[index] = 1.0;
            direction
        })
        .collect::<Vec<_>>();
    if dimension == C51_VECTOR_OUTPUT_DIMENSION {
        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            for sign in [-1.0, 1.0] {
                let mut direction = vec![0.0; dimension];
                for kernel in 0..3 {
                    direction[3 * kernel + left] = 1.0;
                    direction[3 * kernel + right] = sign;
                }
                let norm = direction
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                for value in &mut direction {
                    *value /= norm;
                }
                directions.push(direction);
            }
        }
    }
    directions
}

fn covariance_quadratic(matrix: &[Vec<f64>], direction: &[f64]) -> f64 {
    direction
        .iter()
        .enumerate()
        .map(|(row, left)| {
            direction
                .iter()
                .enumerate()
                .map(|(column, right)| left * matrix[row][column] * right)
                .sum::<f64>()
        })
        .sum()
}

fn covariance_correlation_matrix(covariance: &[Vec<f64>]) -> Result<Matrix> {
    let dimension = covariance.len();
    let mut correlation = Matrix::zeros(dimension, dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            if covariance[row][row] <= 0.0 || covariance[column][column] <= 0.0 {
                return Err(feec_gmrf::FeecGmrfError::Inference(
                    "C51 covariance correlation requires positive diagonal".to_string(),
                ));
            }
            correlation[(row, column)] = covariance[row][column]
                / (covariance[row][row] * covariance[column][column]).sqrt();
        }
    }
    Ok(correlation)
}

fn covariance_rows_to_matrix(rows: &[Vec<f64>]) -> Matrix {
    Matrix::from_fn(rows.len(), rows.len(), |row, column| rows[row][column])
}

fn covariance_spectral_norm(matrix: &Matrix) -> f64 {
    matrix
        .clone()
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .fold(0.0, f64::max)
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
    fn development_config_exposes_mesh_and_observation_workload() {
        let config = C51ThesisNormalizedCompactConfig::development(4, 128).unwrap();
        assert_eq!(config.refinement, 4);
        assert_eq!(config.observation_count, 128);
        assert_eq!(config.target_b_rms_tesla, C51_THESIS_TARGET_B_RMS_TESLA);
        assert!(config.profile.contains("development"));
        assert!(C51ThesisNormalizedCompactConfig::development(1, 128).is_err());
        assert!(C51ThesisNormalizedCompactConfig::development(4, 0).is_err());
    }

    #[test]
    fn weak_pde_uses_feec_projected_sparse_mass_inverse() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 4, 1.0);
        let (topology, coordinates) = mesh.compute_coord_complex();
        let metric = coordinates.to_edge_lengths(&topology);
        let boundary = homogeneous_full_relative_boundary(&topology);
        let assembly = build_reduced_hodge_laplace_1form_system(&topology, &metric, &boundary)
            .expect("FEEC should assemble the projected sparse inverse");
        let inverse = assembly
            .state_mass_inverse
            .as_ref()
            .expect("the reduced 1-form system should expose its FEEC inverse");
        assert_eq!(inverse.nrows(), assembly.state_dimension());
        assert_eq!(inverse.ncols(), assembly.state_dimension());
        assert!(
            inverse.nnz() < inverse.nrows() * inverse.ncols(),
            "the C51 path must not replace FEEC's sparse inverse with a dense matrix"
        );
    }

    #[test]
    fn field_wide_error_summary_reports_requested_quantiles_and_floor() {
        let summary = field_wide_error_summary(&[1.1, 0.8, 3.0], &[1.0, 1.0, 2.0]).unwrap();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.near_zero_reference_count, 0);
        assert!((summary.relative_error_floor - 2.0e-12).abs() < 1.0e-24);
        assert!((summary.median_relative_error - 0.2).abs() < 1.0e-12);
        assert!((summary.percentile_90_relative_error - 0.5).abs() < 1.0e-12);
        assert!((summary.percentile_99_relative_error - 0.5).abs() < 1.0e-12);
        assert!((summary.maximum_relative_error - 0.5).abs() < 1.0e-12);
        assert!((summary.maximum_absolute_error - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn weak_residual_small_noise_converges_and_exact_representations_agree() {
        let mut config = C51ThesisNormalizedCompactConfig::development(4, 32).unwrap();
        config.trace_probes = 32;
        config.trace_batches = 4;
        config.pde_residual_standard_deviations = vec![1.0e-1, 1.0e-3, 1.0e-5];
        let report = run_c51_thesis_normalized_compact(&config).unwrap();
        assert!(report.weak_residual_manufactured_deterministic_relative_l2_error < 1.0e-10);
        assert_eq!(report.weak_residual_rows.len(), 3);
        let first = report.weak_residual_rows.first().unwrap();
        let last = report.weak_residual_rows.last().unwrap();
        assert!(
            last.exact_augmented_deterministic_relative_l2_error
                < first.exact_augmented_deterministic_relative_l2_error
        );
        assert!(
            last.compact_deterministic_relative_l2_error
                < first.compact_deterministic_relative_l2_error
        );
        assert_eq!(report.latent_field_wide.rows.len(), report.state_dimension);
        assert_eq!(
            report.flux_field_wide.rows.len(),
            report.magnetic_output_dimension
        );
        for field in [&report.latent_field_wide, &report.flux_field_wide] {
            assert!(field
                .rows
                .iter()
                .all(|row| row.exact_posterior_prior_ratio >= 0.0
                    && row.exact_posterior_prior_ratio <= 1.0 + 1.0e-8
                    && row.compact_posterior_prior_ratio >= 0.0
                    && row.compact_posterior_prior_ratio <= 1.0 + 1.0e-8));
            assert!(field
                .posterior_prior_ratio_error
                .maximum_relative_error
                .is_finite());
        }
        assert!(report
            .covariance_action_errors
            .flux_posterior_maximum_relative_energy_error
            .is_finite());

        let directory = std::env::temp_dir().join(format!(
            "c51_weak_residual_development_{}",
            std::process::id()
        ));
        if directory.exists() {
            fs::remove_dir_all(&directory).unwrap();
        }
        report.write_directory(&directory).unwrap();
        assert!(directory.join("weak_residual_pde.csv").is_file());
        assert!(directory.join("field_wide_variances.csv").is_file());
        assert!(directory.join("field_wide_accuracy_summary.csv").is_file());
        assert!(directory
            .join("field_wide_covariance_actions.csv")
            .is_file());
        assert!(!directory.join("manifest.yaml").exists());
        fs::remove_dir_all(directory).unwrap();
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
        let flux_elimination =
            EssentialBoundaryElimination::from_layout(outgoing.upper().boundary_layout().clone())
                .unwrap();
        let split = assemble_c51_surface_flux_split(
            &topology,
            &coordinates,
            &flux_elimination,
            &flux_cochain,
        )
        .unwrap();
        let selection = split.selection();
        let held_out = split.held_out();

        assert_eq!(selection.output_dimension(), 6);
        assert_eq!(held_out.output_dimension(), 6);
        assert_eq!(selection.input_dimension(), flux_cochain.input_dimension());
        assert_eq!(held_out.input_dimension(), flux_cochain.input_dimension());
    }
}
