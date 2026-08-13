use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_path_homology_vector_fields, PlanarHolesTopologyVsNaiveGpConfig,
};
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_path_homology_vector_fields");

    let mut config = PlanarHolesTopologyVsNaiveGpConfig::default();
    config.base.output_dir = output_dir.clone();
    config.base.mesh_path = output_dir.join("planar_holes_path_homology_vector_fields.msh");
    config.base.geo_path = output_dir.join("planar_holes_path_homology_vector_fields.geo");
    config.base.force_mesh = true;
    config.base.mesh_size = env_f64("PLANAR_HOLES_VECTOR_FIELDS_MESH_SIZE", 0.045);
    config.base.coexact_truth_mass_norm = 1.0;
    config.base.harmonic_truth_mass_norm = 2.0;
    config.base.spectral_branch_energy_normalization = true;
    config.base.spectral_coexact_expected_m1_energy = 1.0;
    config.base.spectral_harmonic_expected_m1_energy = 4.0;
    config.base.spectral_harmonic_mode_count = 3;
    config.base.spectral_coexact_mode_count =
        env_usize("PLANAR_HOLES_VECTOR_FIELDS_SPECTRAL_COEXACT_MODES", 512);
    config.total_observation_budget = env_usize("PLANAR_HOLES_VECTOR_FIELDS_TRAIN_LOCAL", 120);
    config.validation_local_count = env_usize("PLANAR_HOLES_VECTOR_FIELDS_VALIDATION_LOCAL", 30);
    config.heldout_local_count = env_usize("PLANAR_HOLES_VECTOR_FIELDS_HELDOUT_LOCAL", 40);

    println!("Planar holes path-homology posterior mean vector fields");
    println!("output_dir={}", output_dir.display());
    println!(
        "mesh_size={} spectral_coexact_modes={} train_local={} validation_local={} heldout_local={}",
        config.base.mesh_size,
        config.base.spectral_coexact_mode_count,
        config.total_observation_budget,
        config.validation_local_count,
        config.heldout_local_count
    );

    let result = run_planar_holes_path_homology_vector_fields(&config)?;
    println!(
        "topology: V={} E={} F={} b1={} train_rank={} validation_rank={} heldout_rank={} contrast_rank={}",
        result.topology_summary.vertex_count,
        result.topology_summary.edge_count,
        result.topology_summary.face_count,
        result.topology_summary.b1,
        result.train_cycle_harmonic_pairing_rank,
        result.validation_cycle_harmonic_pairing_rank,
        result.heldout_cycle_harmonic_pairing_rank,
        result.path_contrast_harmonic_pairing_rank
    );
    for row in &result.rows {
        println!(
            "model={} E1={:.4e} nlpd={:.4e}/{:.4e} path_err={:.4e} contrast_err={:.4e} loop_err={:.4e} leakage={:.4e}",
            row.model.as_str(),
            row.l2_error,
            row.heldout_nlpd,
            row.calibrated_heldout_nlpd,
            row.path_integral_relative_error,
            row.path_contrast_relative_error,
            row.hole_loop_relative_error,
            row.codifferential_leakage
        );
    }
    println!("wrote {}", result.truth_vtu_path.display());
    println!("wrote {}", result.figure_png_path.display());
    println!("wrote {}", result.summary_csv_path.display());
    println!("completed in {:.3}s", start.elapsed().as_secs_f64());
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
