use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_topology_sparse_noisy_prediction, ToroidalHarmonicBConfig,
};
use feg_infer::linear_pde::{LinearPdeVarianceConfig, LinearPdeVarianceMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = ToroidalHarmonicBConfig {
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_topology_sparse_noisy_prediction",
        )),
        include_full_field_variance_maps: false,
        use_mass_weighted_pde_residual: true,
        normalize_mass_weighted_pde_residual: true,
        source_alpha_true: 1.0,
        sparse_prediction_training_fraction: 0.10,
        sparse_prediction_noise_seed: 20260514,
        ..ToroidalHarmonicBConfig::default()
    };
    config.solver.variance = LinearPdeVarianceConfig {
        mode: LinearPdeVarianceMode::ExactSolves,
        num_variance_probes: 32,
        variance_batch_count: 4,
        rng_seed: 197,
        local_rb_block_size: 16,
    };
    let result = run_toroidal_topology_sparse_noisy_prediction(&config)?;

    println!("Toroidal topology sparse noisy prediction");
    println!(
        "betti={:?} harmonic_2_dim={} c_H={:.6e} source_truth={:.3}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.source_harmonic_kappa,
        config.source_alpha_true
    );
    for stage in &result.stages {
        let factor = stage.solve.debug.posterior_factorization;
        let qoi = |name: &str| {
            stage
                .pushforward_qois
                .iter()
                .find(|row| row.qoi == name)
                .map(|row| {
                    format!(
                        "mean={:.6e} sd={:.3e} ratio={:.3e}",
                        row.mean, row.sd, row.variance_ratio
                    )
                })
                .unwrap_or_else(|| "missing".to_string())
        };
        let covered = stage
            .heldout_predictions
            .iter()
            .filter(|row| row.covered95)
            .count();
        let total = stage.heldout_predictions.len();
        let max_z = stage
            .heldout_predictions
            .iter()
            .map(|row| row.standardized_residual.abs())
            .fold(0.0_f64, f64::max);
        println!(
            "{}: s({}) eta_H({}) I_gamma({}) heldout={}/{} max|z|={:.3} posterior_factor_nnz={} fill={:.3}x factor_mib={:.3}",
            stage.summary.stage,
            qoi("qoi::s"),
            qoi("qoi::eta_H"),
            qoi("qoi::I_gamma"),
            covered,
            total,
            max_z,
            factor.factor_nnz,
            factor.fill_in_ratio_vs_lower_triangle,
            factor.factor_numeric_values_mib
        );
    }
    if let Some(out_dir) = &config.output_dir {
        println!("wrote outputs to {}", out_dir.display());
    }
    Ok(())
}
