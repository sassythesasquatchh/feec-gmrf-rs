use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_topology_source_template_gp_baseline, ToroidalGpBaselineConfig,
    ToroidalHarmonicBConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ToroidalGpBaselineConfig {
        output_dir: Some(PathBuf::from(
            "out/examples/toroidal_topology_source_template_gp_baseline",
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

    let result = run_toroidal_topology_source_template_gp_baseline(&config)?;
    println!("Toroidal source-template GP baseline");
    println!(
        "betti={:?} harmonic_2_dim={} c_H={:.6e} source_truth={:.3}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.source_harmonic_kappa,
        config.toroidal.source_alpha_true
    );
    for stage in &result.stages {
        println!(
            "{} {}: train={} heldout={} s={:.6} sd={:.3e} rmse={:.3e} nlpd={:.3e} coverage={}/{} max|z|={:.3}",
            stage.summary.model,
            stage.summary.stage,
            stage.summary.training_rows,
            stage.summary.heldout_rows,
            stage.summary.source_posterior_mean,
            stage.summary.source_posterior_variance.max(0.0).sqrt(),
            stage.summary.heldout_rmse,
            stage.summary.heldout_nlpd,
            stage.summary.heldout_covered95,
            stage.summary.heldout_rows,
            stage.summary.heldout_max_abs_standardized_residual
        );
    }
    if let Some(out_dir) = &config.output_dir {
        println!("wrote outputs to {}", out_dir.display());
    }
    Ok(())
}
