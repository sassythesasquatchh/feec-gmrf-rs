use feg_case_studies::annulus_h_formulation::{
    run_annulus_h_formulation, write_annulus_h_formulation_outputs, AnnulusHFormulationConfig,
};
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/annulus_h_formulation");
    let mut config = AnnulusHFormulationConfig::default();
    config.output_dir = output_dir.clone();
    config.mesh_path = output_dir.join("annulus_h_formulation.msh");
    config.geo_path = output_dir.join("annulus_h_formulation.geo");
    config.mesh_size = env_f64("ANNULUS_H_MESH_SIZE", config.mesh_size);
    config.noise_trial_count = env_usize("ANNULUS_H_TRIALS", config.noise_trial_count);
    config.residual_count = env_usize("ANNULUS_H_RESIDUALS", config.residual_count);
    config.heldout_loop_count = env_usize("ANNULUS_H_HELDOUT_LOOPS", config.heldout_loop_count);

    println!("2D annulus H-formulation FEEC-GMRF benchmark");
    println!(
        "output_dir={} mesh_size={} trials={} lines={} residuals={} heldout_loops={}",
        output_dir.display(),
        config.mesh_size,
        config.noise_trial_count,
        config.line_tangential_count + config.line_radial_count + config.line_random_count,
        config.residual_count,
        config.heldout_loop_count
    );

    let result = run_annulus_h_formulation(&config)?;
    write_annulus_h_formulation_outputs(&result, &config)?;

    println!(
        "topology: V={} E={} F={} b1={} reference_period_truth={:.6} closure_max={:.3e}",
        result.topology.vertex_count,
        result.topology.edge_count,
        result.topology.face_count,
        result.topology.harmonic_1_dimension,
        result.topology.reference_period_truth,
        result.topology.truth_closure_max_abs
    );
    for row in result
        .summary_rows
        .iter()
        .filter(|row| row.regime.as_str().starts_with("D_"))
    {
        println!(
            "D model={} line_rmse={:.3e} residual_rmse={:.3e} circ_rmse={:.3e} coverage95={:.3} topo_spread={:.3e} q_mean={:.3e} q_std={:.3e} selected_total_seconds={:.3} prior_density={:.3e} posterior_factor_nnz={:.3e}",
            row.model.as_str(),
            row.rmse_line,
            row.residual_rmse,
            row.circ_rmse,
            row.coverage_95,
            row.topo_spread,
            row.q_mean,
            row.q_std,
            row.selected_total_seconds,
            row.prior_precision_density,
            row.posterior_factor_nnz
        );
    }
    println!(
        "wrote topology_summary.csv, metrics_summary.csv, trial_metrics.csv, heldout_predictions.csv, hyperparameter_tuning.csv, plots, and VTU fields in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(default)
}
