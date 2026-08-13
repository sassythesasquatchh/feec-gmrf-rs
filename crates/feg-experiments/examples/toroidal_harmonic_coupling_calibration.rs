use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_harmonic_coupling_calibration, ToroidalHarmonicBConfig,
    ToroidalHarmonicCouplingCalibrationConfig,
};
use feg_infer::linear_pde::{
    LinearPdePrecisionPolicy, LinearPdeUqSolverConfig, LinearPdeVarianceConfig,
    LinearPdeVarianceMode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ToroidalHarmonicCouplingCalibrationConfig {
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_harmonic_coupling_calibration",
        )),
        toroidal: ToroidalHarmonicBConfig {
            output_dir: None,
            include_full_field_variance_maps: false,
            use_mass_weighted_pde_residual: true,
            normalize_mass_weighted_pde_residual: true,
            sparse_prediction_training_fraction: 0.10,
            sparse_prediction_noise_seed: 20260514,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::Hutchinson,
                    num_variance_probes: 96,
                    variance_batch_count: 4,
                    rng_seed: 20260518,
                    local_rb_block_size: 16,
                },
                precision_policy: LinearPdePrecisionPolicy::default(),
                log_diagnostics: true,
            },
            ..ToroidalHarmonicBConfig::default()
        },
        drive_currents: vec![0.6, 0.8, 1.0, 1.2, 1.4],
        coupling_prior_std_scale: 5.0,
    };

    let result = run_toroidal_harmonic_coupling_calibration(&config)?;
    println!("Toroidal harmonic-coupling calibration");
    println!(
        "betti={:?} harmonic_2_dim={} c_H_truth={:.6e} drives={:?}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.source_harmonic_kappa,
        result.drive_currents
    );
    for stage in &result.stages {
        println!(
            "{}: train={} heldout={} c_H={:.6e} sd={:.3e} err={:.3e} rmse={:.3e} coverage={}/{} fill={:.3}x",
            stage.summary.stage,
            stage.summary.training_rows,
            stage.summary.heldout_rows,
            stage.summary.coupling_posterior_mean,
            stage.summary.coupling_posterior_variance.max(0.0).sqrt(),
            stage.summary.coupling_abs_error,
            stage.summary.heldout_rmse,
            stage.summary.heldout_covered95,
            stage.summary.heldout_rows,
            stage.summary.posterior_fill_in
        );
    }
    if let Some(out_dir) = &config.output_dir {
        println!("wrote outputs to {}", out_dir.display());
    }
    Ok(())
}
