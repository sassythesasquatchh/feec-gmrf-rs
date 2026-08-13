use feg_case_studies::toroidal_inductor::{
    toroidal_exact_b_field_recovery_observation_indices, toroidal_exact_b_precision_scale_audit,
    ToroidalExactBBaseConfig, ToroidalExactBNondimensionalizationMode,
    ToroidalExactBObservationMode, ToroidalExactBRecoveryConfig, ToroidalExactBReferenceMode,
};
use feg_infer::linear_pde::LinearPdePrecisionPolicy;
use std::error::Error;

const OBSERVATION_NOISE_STD: f64 = 1.0e-9;
const FIELD_COVERAGE_AZIMUTH_COUNT: usize = 16;

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "case,term,joint_dimension,state_dimension,source_dimension,training_rows,heldout_rows,nondimensionalization,state_scale_min,state_scale_max,source_scale_min,source_scale_max,scaled_posterior_max_abs_diagonal,scaled_posterior_diagonal_ratio,nonzero_diagonal_entries,min_positive_diagonal,max_abs_diagonal,diagonal_sum,max_index,max_block,max_local_index"
    );
    for training_count in [6usize, 12, 24, 36] {
        let mut config = base_config();
        config.observation_index_override =
            Some(toroidal_exact_b_field_recovery_observation_indices(
                FIELD_COVERAGE_AZIMUTH_COUNT,
                training_count,
            )?);
        let audit = toroidal_exact_b_precision_scale_audit(&config)?;
        for term in audit.terms {
            println!(
                "training_rows={},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{},{},{}",
                training_count,
                term.term,
                audit.joint_dimension,
                audit.state_dimension,
                audit.source_dimension,
                audit.training_rows,
                audit.heldout_rows,
                audit.nondimensionalization.label(),
                audit.state_scale_min,
                audit.state_scale_max,
                audit.source_scale_min,
                audit.source_scale_max,
                audit.scaled_posterior_max_abs_diagonal,
                audit.scaled_posterior_diagonal_ratio,
                term.nonzero_diagonal_entries,
                term.min_positive_diagonal,
                term.max_abs_diagonal,
                term.diagonal_sum,
                term.max_index,
                term.max_block,
                term.max_local_index
            );
        }
    }
    Ok(())
}

fn base_config() -> ToroidalExactBRecoveryConfig {
    ToroidalExactBRecoveryConfig {
        reference_mode: ToroidalExactBReferenceMode::PerturbedSource,
        observation_mode: ToroidalExactBObservationMode::SurfaceFluxes,
        source_deltas: [0.0, 0.15, -0.10, 0.05],
        source_prior_std: 0.25,
        reference_pde_variance: Some(1.0e-8),
        prior_kappa: 1.0,
        prior_tau: 1.0e-6,
        observation_noise_std: OBSERVATION_NOISE_STD,
        synthetic_observation_noise_seed: Some(0x5EED_F10C_2026),
        heldout_count: 128,
        observation_index_override: None,
        surface_flux_azimuth_count: FIELD_COVERAGE_AZIMUTH_COUNT,
        nondimensionalization: ToroidalExactBNondimensionalizationMode::PdeColumnNorm,
        output_dir: None,
        write_outputs: false,
        base: ToroidalExactBBaseConfig {
            pde_variance: 3.0e-8,
            precision_policy: LinearPdePrecisionPolicy::DiagonalEquilibrated {
                max_relative_asymmetry: 1.0e-10,
            },
            ..ToroidalExactBBaseConfig::default()
        },
        ..ToroidalExactBRecoveryConfig::default()
    }
}
