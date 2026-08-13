use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_topology_vs_naive_gp, write_planar_holes_topology_vs_naive_gp_outputs,
    PlanarHolesTopologyVsNaiveGpConfig,
};
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_topology_vs_naive_gp");

    let mut config = PlanarHolesTopologyVsNaiveGpConfig::default();
    config.base.output_dir = output_dir.clone();
    config.base.mesh_path = output_dir.join("planar_holes_topology_vs_naive_gp.msh");
    config.base.geo_path = output_dir.join("planar_holes_topology_vs_naive_gp.geo");
    config.base.force_mesh = true;
    config.base.mesh_size = env_f64("PLANAR_HOLES_TOPOLOGY_NAIVE_MESH_SIZE", 0.045);
    config.base.coexact_truth_mass_norm = env_f64(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_COEXACT_TRUTH_NORM",
        config.base.coexact_truth_mass_norm,
    );
    config.base.harmonic_truth_mass_norm = env_f64(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_HARMONIC_TRUTH_NORM",
        config.base.harmonic_truth_mass_norm,
    );
    config.total_observation_budget = env_usize(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_BUDGET",
        config.total_observation_budget,
    );
    config.validation_local_count = env_usize(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_VALIDATION_LOCAL",
        config.validation_local_count,
    );
    config.validation_interior_loop_count = env_usize(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_VALIDATION_LOOPS",
        config.validation_interior_loop_count,
    );
    config.validation_long_path_count = env_usize(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_VALIDATION_PATHS",
        config.validation_long_path_count,
    );
    config.heldout_local_count = env_usize(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_HELDOUT_LOCAL",
        config.heldout_local_count,
    );
    config.heldout_interior_loop_count = env_usize(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_HELDOUT_LOOPS",
        config.heldout_interior_loop_count,
    );
    config.heldout_long_path_count = env_usize(
        "PLANAR_HOLES_TOPOLOGY_NAIVE_HELDOUT_PATHS",
        config.heldout_long_path_count,
    );
    if let Some(values) = env_f64_list("PLANAR_HOLES_TOPOLOGY_NAIVE_HODGE_KAPPAS") {
        config.hodge_kappas = values;
    }
    if let Some(values) = env_f64_list("PLANAR_HOLES_TOPOLOGY_NAIVE_HODGE_TAUS") {
        config.hodge_taus = values;
    }
    if let Some(values) = env_f64_list("PLANAR_HOLES_TOPOLOGY_NAIVE_BASELINE_KAPPAS") {
        config.naive_kappas = values;
    }
    if let Some(values) = env_f64_list("PLANAR_HOLES_TOPOLOGY_NAIVE_BASELINE_TAUS") {
        config.naive_taus = values;
    }
    if let Some(values) = env_f64_list("PLANAR_HOLES_TOPOLOGY_NAIVE_VARIANCE_SCALES") {
        config.observation_variance_scales = values;
    }

    println!("Planar holes topology-aware sparse Hodge vs naive Euclidean GP");
    println!("output_dir={}", output_dir.display());
    println!(
        "mesh_size={} truth_norms=(coexact {}, harmonic {}) budget={} validation=({}, {}, {}) heldout=({}, {}, {})",
        config.base.mesh_size,
        config.base.coexact_truth_mass_norm,
        config.base.harmonic_truth_mass_norm,
        config.total_observation_budget,
        config.validation_local_count,
        config.validation_interior_loop_count,
        config.validation_long_path_count,
        config.heldout_local_count,
        config.heldout_interior_loop_count,
        config.heldout_long_path_count
    );
    println!(
        "hodge_kappas={:?} hodge_taus={:?} naive_kappas={:?} naive_taus={:?} variance_scales={:?}",
        config.hodge_kappas,
        config.hodge_taus,
        config.naive_kappas,
        config.naive_taus,
        config.observation_variance_scales
    );

    let result = run_planar_holes_topology_vs_naive_gp(&config)?;
    write_planar_holes_topology_vs_naive_gp_outputs(&result, &output_dir)?;

    println!(
        "topology: V={} E={} F={} b1={} train_rank={} validation_rank={} heldout_rank={}",
        result.topology_summary.vertex_count,
        result.topology_summary.edge_count,
        result.topology_summary.face_count,
        result.topology_summary.b1,
        result.train_cycle_harmonic_pairing_rank,
        result.validation_cycle_harmonic_pairing_rank,
        result.heldout_cycle_harmonic_pairing_rank
    );
    for row in &result.rows {
        println!(
            "model={} kappa={:.4} tau={:.4} var_scale={:.4} validation_nlpd={:.4e} E1={:.4e} heldout_nlpd={:.4e} local_nlpd={:.4e} loop_nlpd={:.4e} interior_loop_err={:.4e} long_path_err={:.4e} d_err={:.4e} delta_leak={:.4e} coverage={:.3}",
            row.model.as_str(),
            row.selected_kappa,
            row.selected_tau,
            row.selected_observation_variance_scale,
            row.validation_nlpd,
            row.l2_error,
            row.heldout_nlpd,
            row.heldout_local_nlpd,
            row.heldout_loop_nlpd,
            row.heldout_interior_loop_relative_error,
            row.heldout_long_path_relative_error,
            row.exterior_derivative_error,
            row.codifferential_leakage,
            row.all_edge_coverage_95
        );
    }
    println!(
        "wrote metrics_summary.csv, validation_summary.csv, heldout_predictions.csv, field_coverage_summary.csv, topology_summary.csv in {:.3}s",
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

fn env_f64_list(name: &str) -> Option<Vec<f64>> {
    let values = env::var(name).ok()?;
    let parsed = values
        .split(',')
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    (!parsed.is_empty()).then_some(parsed)
}
