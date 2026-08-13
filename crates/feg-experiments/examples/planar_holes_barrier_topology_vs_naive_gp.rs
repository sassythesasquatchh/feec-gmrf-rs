use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_barrier_topology_vs_naive_gp,
    write_planar_holes_barrier_topology_vs_naive_gp_outputs, PlanarHolesTopologyVsNaiveGpConfig,
};
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_barrier_topology_vs_naive_gp");

    let mut config = PlanarHolesTopologyVsNaiveGpConfig::default();
    config.base.output_dir = output_dir.clone();
    config.base.mesh_path = output_dir.join("planar_holes_barrier_topology_vs_naive_gp.msh");
    config.base.geo_path = output_dir.join("planar_holes_barrier_topology_vs_naive_gp.geo");
    config.base.force_mesh = true;
    config.base.mesh_size = env_f64("PLANAR_HOLES_BARRIER_MESH_SIZE", 0.025);
    config.base.coexact_truth_mass_norm = 1.0;
    config.base.harmonic_truth_mass_norm = 2.0;
    config.base.spectral_branch_energy_normalization = true;
    config.base.spectral_coexact_expected_m1_energy = 1.0;
    config.base.spectral_harmonic_expected_m1_energy = 4.0;
    config.base.spectral_harmonic_mode_count = 3;
    config.base.spectral_coexact_mode_count =
        env_usize("PLANAR_HOLES_BARRIER_SPECTRAL_COEXACT_MODES", 384);
    config.total_observation_budget = env_usize("PLANAR_HOLES_BARRIER_TRAIN_LOCAL", 120);
    config.validation_local_count = env_usize("PLANAR_HOLES_BARRIER_VALIDATION_LOCAL", 30);
    config.validation_long_path_count = env_usize("PLANAR_HOLES_BARRIER_VALIDATION_PATHS", 40);
    config.heldout_local_count = env_usize("PLANAR_HOLES_BARRIER_HELDOUT_LOCAL", 80);
    config.heldout_long_path_count = env_usize("PLANAR_HOLES_BARRIER_HELDOUT_PATHS", 80);

    println!("Planar holes barrier topology-aware sparse Hodge vs naive Euclidean GP");
    println!("output_dir={}", output_dir.display());
    println!(
        "mesh_size={} train_left_edges={} validation_left_edges={} heldout_right_edges={} heldout_paths={}",
        config.base.mesh_size,
        config.total_observation_budget,
        config.validation_local_count,
        config.heldout_local_count,
        config.heldout_long_path_count
    );

    let result = run_planar_holes_barrier_topology_vs_naive_gp(&config)?;
    write_planar_holes_barrier_topology_vs_naive_gp_outputs(&result, &output_dir)?;
    println!(
        "topology: V={} E={} F={} b1={} train_rank={} validation_rank={} heldout_rank={}",
        result.base.topology_summary.vertex_count,
        result.base.topology_summary.edge_count,
        result.base.topology_summary.face_count,
        result.base.topology_summary.b1,
        result.base.train_cycle_harmonic_pairing_rank,
        result.base.validation_cycle_harmonic_pairing_rank,
        result.base.heldout_cycle_harmonic_pairing_rank
    );
    for row in &result.rows {
        println!(
            "model={} cross_local_nlpd={:.4e}/{:.4e} cross_local_err={:.4e} barrier_path_nlpd={:.4e}/{:.4e} barrier_path_err={:.4e} loop_nlpd={:.4e}/{:.4e} loop_err={:.4e} cov={:.3}/{:.3}",
            row.model.as_str(),
            row.cross_barrier_local_nlpd,
            row.calibrated_cross_barrier_local_nlpd,
            row.cross_barrier_local_relative_error,
            row.barrier_long_path_nlpd,
            row.calibrated_barrier_long_path_nlpd,
            row.barrier_long_path_relative_error,
            row.hole_loop_nlpd,
            row.calibrated_hole_loop_nlpd,
            row.hole_loop_relative_error,
            row.cross_barrier_local_coverage_95,
            row.calibrated_cross_barrier_local_coverage_95,
        );
    }
    println!(
        "wrote metrics_summary.csv, validation_summary.csv, calibration_summary.csv, heldout_predictions.csv, field_coverage_summary.csv, topology_summary.csv, barrier_summary.csv in {:.3}s",
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
