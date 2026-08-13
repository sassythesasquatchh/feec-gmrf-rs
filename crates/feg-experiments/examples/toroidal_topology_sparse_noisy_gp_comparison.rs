use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_topology_sparse_noisy_gp_comparison, ToroidalGpBaselineConfig,
    ToroidalHarmonicBConfig,
};
use feg_infer::linear_pde::{LinearPdeVarianceConfig, LinearPdeVarianceMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = ToroidalGpBaselineConfig {
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_topology_sparse_noisy_gp_comparison",
        )),
        toroidal: ToroidalHarmonicBConfig {
            output_dir: None,
            source_alpha_true: 1.15,
            include_full_field_variance_maps: false,
            use_mass_weighted_pde_residual: true,
            normalize_mass_weighted_pde_residual: true,
            sparse_prediction_training_fraction: 0.10,
            sparse_prediction_noise_seed: 20260514,
            ..ToroidalHarmonicBConfig::default()
        },
        ..ToroidalGpBaselineConfig::default()
    };
    config.toroidal.solver.variance = LinearPdeVarianceConfig {
        mode: LinearPdeVarianceMode::ExactSolves,
        num_variance_probes: 32,
        variance_batch_count: 4,
        rng_seed: 397,
        local_rb_block_size: 16,
    };

    let result = run_toroidal_topology_sparse_noisy_gp_comparison(&config)?;

    println!("Toroidal sparse/noisy GP fairness comparison");
    println!(
        "betti={:?} harmonic_2_dim={} c_H={:.6e} source_truth={:.3}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.source_harmonic_kappa,
        config.toroidal.source_alpha_true
    );
    for row in result
        .metrics
        .iter()
        .filter(|row| row.sensor_family == "all")
    {
        println!(
            "{} {}: train={} heldout={} rmse={:.3e} nlpd={:.3e} coverage={}/{} max|z|={:.3} pde={}",
            row.model,
            row.stage,
            row.training_rows,
            row.heldout_rows,
            row.rmse,
            row.nlpd,
            row.covered95,
            row.heldout_rows,
            row.max_abs_standardized_residual,
            row.pde_residual_used
        );
    }
    if let Some(out_dir) = &config.output_dir {
        println!("wrote outputs to {}", out_dir.display());
    }
    Ok(())
}
