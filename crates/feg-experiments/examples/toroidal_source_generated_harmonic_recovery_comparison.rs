use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_source_generated_harmonic_recovery, ToroidalHarmonicBConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for (
        label,
        use_mass_weighted_pde_residual,
        normalize_mass_weighted_pde_residual,
        mass_weighted_pde_precision_scale,
    ) in [
        ("euclidean", false, false, 1.0),
        ("mass_weighted", true, false, 1.0),
        ("mass_diag_normalized", true, true, 1.0),
    ] {
        let mut config = ToroidalHarmonicBConfig {
            output_dir: Some(PathBuf::from(format!(
                "out/examples/toroidal_source_generated_harmonic_recovery_{label}"
            ))),
            use_mass_weighted_pde_residual,
            normalize_mass_weighted_pde_residual,
            mass_weighted_pde_precision_scale,
            include_full_field_variance_maps: false,
            ..ToroidalHarmonicBConfig::default()
        };
        config.solver.log_diagnostics = false;
        let result = run_toroidal_source_generated_harmonic_recovery(&config)?;
        println!("source-generated harmonic recovery ({label})");
        for stage in &result.stages {
            println!(
                "{}: alpha_mean={} alpha_var={} pde_residual={:.3e} hall_rmse={:.3e} flux_rmse={:.3e}",
                stage.summary.stage,
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
                stage.summary.pde_residual_norm,
                stage.summary.hall_rmse,
                stage.summary.flux_rmse
            );
        }
    }
    Ok(())
}
