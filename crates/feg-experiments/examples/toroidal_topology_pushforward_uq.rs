use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_topology_pushforward_uq, ToroidalHarmonicBConfig,
};
use feg_infer::linear_pde::{LinearPdeVarianceConfig, LinearPdeVarianceMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = ToroidalHarmonicBConfig {
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_topology_pushforward_uq",
        )),
        include_full_field_variance_maps: false,
        use_mass_weighted_pde_residual: true,
        normalize_mass_weighted_pde_residual: true,
        ..ToroidalHarmonicBConfig::default()
    };
    config.solver.variance = LinearPdeVarianceConfig {
        mode: LinearPdeVarianceMode::ExactSolves,
        num_variance_probes: 32,
        variance_batch_count: 4,
        rng_seed: 97,
        local_rb_block_size: 16,
    };
    let result = run_toroidal_topology_pushforward_uq(&config)?;

    println!("Topology-aware magnetostatic pushforward UQ");
    println!(
        "betti={:?} harmonic_2_dim={} c_H={:.6e} linked_current_true={:.6e}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.source_harmonic_kappa,
        result.topology_summary.linked_current_true
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
        println!(
            "{}: s({}) beta_H({}) eta_H({}) Phi_link({}) I_gamma({}) posterior_precision_nnz={} posterior_factor_nnz={} fill={:.3}x factor_mib={:.3}",
            stage.summary.stage,
            qoi("qoi::s"),
            qoi("qoi::beta_H"),
            qoi("qoi::eta_H"),
            qoi("qoi::Phi_link"),
            qoi("qoi::I_gamma"),
            factor.matrix_nnz,
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
