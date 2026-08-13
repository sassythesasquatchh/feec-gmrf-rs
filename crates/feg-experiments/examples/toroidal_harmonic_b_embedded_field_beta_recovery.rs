use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_harmonic_b_embedded_field_beta_recovery, ToroidalHarmonicBConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ToroidalHarmonicBConfig {
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_harmonic_b_embedded_field_recovery",
        )),
        include_full_field_variance_maps: false,
        ..ToroidalHarmonicBConfig::default()
    };
    let result = run_toroidal_harmonic_b_embedded_field_beta_recovery(&config)?;

    println!("Toroidal embedded-field beta recovery");
    println!(
        "betti={:?} harmonic_2_dim={} beta_true={:.6e} deterministic_harmonic_projection_rel={:.3e}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.beta_true,
        result
            .topology_summary
            .deterministic_harmonic_projection_relative
    );
    for stage in &result.stages {
        let posterior_factor = stage.solve.debug.posterior_factorization;
        println!(
            "{}: hall_rmse={:.3e} flux_rmse={:.3e} beta_mean={} beta_var={} posterior_precision_nnz={} posterior_factor_nnz={} fill={:.3}x",
            stage.summary.stage,
            stage.summary.hall_rmse,
            stage.summary.flux_rmse,
            stage
                .summary
                .beta_posterior_mean
                .map(|value| format!("{value:.6e}"))
                .unwrap_or_else(|| "fixed".to_string()),
            stage
                .summary
                .beta_posterior_variance
                .map(|value| format!("{value:.6e}"))
                .unwrap_or_else(|| "fixed".to_string()),
            posterior_factor.matrix_nnz,
            posterior_factor.factor_nnz,
            posterior_factor.fill_in_ratio_vs_lower_triangle
        );
    }
    if let Some(out_dir) = &config.output_dir {
        println!("wrote outputs to {}", out_dir.display());
    }
    Ok(())
}
