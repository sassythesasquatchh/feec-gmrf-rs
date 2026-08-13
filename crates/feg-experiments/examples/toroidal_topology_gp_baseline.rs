use std::path::PathBuf;

use feg_case_studies::toroidal_harmonic_b::{
    run_toroidal_topology_gp_baseline, ToroidalGpBaselineConfig, ToroidalHarmonicBConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ToroidalGpBaselineConfig {
        output_dir: Some(PathBuf::from("out/examples/toroidal_topology_gp_baseline")),
        toroidal: ToroidalHarmonicBConfig {
            source_alpha_true: 1.15,
            ..ToroidalHarmonicBConfig::default()
        },
        ..ToroidalGpBaselineConfig::default()
    };
    let result = run_toroidal_topology_gp_baseline(&config)?;

    println!("Naive independent-output GP baseline for toroidal pushforward UQ");
    println!(
        "betti={:?} harmonic_2_dim={} c_H={:.6e}",
        result.topology_summary.betti_numbers,
        result.topology_summary.harmonic_2_dimension,
        result.topology_summary.source_harmonic_kappa
    );
    for stage in &result.stages {
        println!(
            "{} matched {}: rows={} ell={:.3e} signal_var={:.3e} heldout_rmse={:.3e} heldout_nlpd={:.3e} coverage={}/{} max|z|={:.3}",
            stage.summary.stage,
            stage.summary.matched_feec_stage,
            stage.summary.training_rows,
            stage.summary.length_scale,
            stage.summary.signal_variance,
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
