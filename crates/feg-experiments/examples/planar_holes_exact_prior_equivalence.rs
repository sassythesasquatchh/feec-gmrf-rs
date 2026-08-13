use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_exact_prior_equivalence, write_planar_holes_prior_equivalence,
    PlanarHolesExactPriorEquivalenceConfig, PlanarHolesFlowConfig,
};
use std::{env, fs, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_exact_prior_equivalence");
    fs::create_dir_all(&output_dir)?;

    let mesh_size = env::var("PLANAR_HOLES_EXACT_PRIOR_EQUIV_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.14);
    let mode_counts = env::var("PLANAR_HOLES_EXACT_PRIOR_EQUIV_MODES")
        .ok()
        .map(|value| parse_mode_counts(&value))
        .unwrap_or_else(|| vec![32, 64, 128]);
    let include_max_available = env::var("PLANAR_HOLES_EXACT_PRIOR_EQUIV_MAX")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let dense_eigen_dimension_cap = env::var("PLANAR_HOLES_EXACT_PRIOR_EQUIV_EIGEN_CAP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(600);

    let base = PlanarHolesFlowConfig {
        output_dir: output_dir.clone(),
        mesh_path: output_dir.join("planar_holes_exact_prior_equivalence.msh"),
        geo_path: output_dir.join("planar_holes_exact_prior_equivalence.geo"),
        force_mesh: true,
        mesh_size,
        ..PlanarHolesFlowConfig::default()
    };
    let config = PlanarHolesExactPriorEquivalenceConfig {
        base,
        mode_counts,
        include_max_available,
        dense_eigen_dimension_cap,
        ..PlanarHolesExactPriorEquivalenceConfig::default()
    };

    println!("Planar holes exact-branch prior equivalence");
    println!("output_dir={}", output_dir.display());
    println!("mesh_size={mesh_size}");
    println!("mode_counts={:?}", config.mode_counts);
    println!("include_max_available={}", config.include_max_available);

    let rows = run_planar_holes_exact_prior_equivalence(&config)?;
    write_planar_holes_prior_equivalence(&rows, output_dir.join("exact_prior_equivalence.csv"))?;

    let max_requested = rows
        .iter()
        .map(|row| row.requested_mode_count)
        .max()
        .unwrap_or(0);
    for row in rows.iter().filter(|row| {
        row.spectral_reference.as_str() == "spectral_energy_normalized"
            && row.requested_mode_count == max_requested
    }) {
        println!(
            "modes={} actual={} variant={} trace={:.4e} tau={:.4e} req_tau={:.4e} frob={:.4e} eig_rel={:.4e}",
            row.requested_mode_count,
            row.actual_mode_count,
            row.gmrf_variant.as_str(),
            row.gmrf_expected_m1_energy,
            row.gmrf_tau_scale,
            row.required_tau_scale_to_match_spectral_trace,
            row.m1_frobenius_relative_error,
            row.source_eigen_relative_l2_error
        );
    }
    println!(
        "rows={} runtime={:.3}s",
        rows.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn parse_mode_counts(value: &str) -> Vec<usize> {
    value
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}
