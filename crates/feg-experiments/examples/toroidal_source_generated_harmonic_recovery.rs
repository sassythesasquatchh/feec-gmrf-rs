use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_source_generated_harmonic_recovery, ToroidalHarmonicBConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ToroidalHarmonicBConfig {
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_source_generated_harmonic_recovery",
        )),
        use_mass_weighted_pde_residual: true,
        normalize_mass_weighted_pde_residual: true,
        include_full_field_variance_maps: false,
        ..ToroidalHarmonicBConfig::default()
    };
    let result = run_toroidal_source_generated_harmonic_recovery(&config)?;

    println!("Toroidal source-generated harmonic recovery");
    println!(
        "betti={:?} harmonic_2_dim={} kappa={:.6e} linked_current_unit={:.6e} true_projection={:.6e}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.source_harmonic_kappa,
        result.topology_summary.linked_current_unit,
        result.topology_summary.source_harmonic_projection_true
    );
    for stage in &result.stages {
        let posterior_factor = stage.solve.debug.posterior_factorization;
        println!(
            "{}: hall_rmse={:.3e} flux_rmse={:.3e} alpha_mean={} alpha_var={} posterior_precision_nnz={} posterior_factor_nnz={} fill={:.3}x",
            stage.summary.stage,
            stage.summary.hall_rmse,
            stage.summary.flux_rmse,
            stage
                .summary
                .alpha_posterior_mean
                .map(|value| format!("{value:.6e}"))
                .unwrap_or_else(|| "fixed".to_string()),
            stage
                .summary
                .alpha_posterior_variance
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
