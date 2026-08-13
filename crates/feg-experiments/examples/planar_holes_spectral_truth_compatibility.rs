use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_spectral_truth_compatibility, write_planar_holes_spectral_truth_compatibility,
    PlanarHolesFlowConfig, PlanarHolesSpectralBoundaryCondition,
    PlanarHolesSpectralTruthCompatibilityConfig,
};
use std::{env, fs, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_spectral_truth_compatibility");
    fs::create_dir_all(&output_dir)?;
    let mesh_size = env::var("PLANAR_HOLES_SPECTRAL_COMPAT_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.045);
    let mode_stages = env::var("PLANAR_HOLES_COMPAT_MODE_STAGES")
        .ok()
        .map(|value| parse_mode_stages(&value))
        .unwrap_or_else(|| vec![32, 64, 128, 256, 512, 1000]);
    let include_strong_boundary = env::var("PLANAR_HOLES_COMPAT_STRONG_BOUNDARY")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let boundary_conditions = if include_strong_boundary {
        vec![
            PlanarHolesSpectralBoundaryCondition::Free,
            PlanarHolesSpectralBoundaryCondition::StrongBoundaryOneForms,
        ]
    } else {
        vec![PlanarHolesSpectralBoundaryCondition::Free]
    };

    let base = PlanarHolesFlowConfig {
        output_dir: output_dir.clone(),
        mesh_path: output_dir.join("planar_holes_spectral_compatibility.msh"),
        geo_path: output_dir.join("planar_holes_spectral_compatibility.geo"),
        force_mesh: true,
        mesh_size,
        spectral_branch_energy_normalization: true,
        spectral_exact_expected_m1_energy: 1.0,
        spectral_coexact_expected_m1_energy: 1.0,
        spectral_harmonic_expected_m1_energy: 0.49,
        ..PlanarHolesFlowConfig::default()
    };
    let config = PlanarHolesSpectralTruthCompatibilityConfig {
        base,
        mode_stages,
        boundary_conditions,
        ..PlanarHolesSpectralTruthCompatibilityConfig::default()
    };

    println!("Planar holes spectral truth compatibility");
    println!("output_dir={}", output_dir.display());
    println!("mesh_size={mesh_size}");
    println!("mode_stages={:?}", config.mode_stages);
    let rows = run_planar_holes_spectral_truth_compatibility(&config)?;
    write_planar_holes_spectral_truth_compatibility(
        &rows,
        output_dir.join("spectral_truth_compatibility.csv"),
    )?;
    let largest_configured_stage = *config.mode_stages.last().unwrap_or(&0);
    let largest_requested_stage = rows
        .iter()
        .map(|row| row.requested_mode_count)
        .max()
        .unwrap_or(largest_configured_stage);
    for row in rows.iter().filter(|row| {
        row.boundary_condition == PlanarHolesSpectralBoundaryCondition::Free
            && (row.requested_mode_count == largest_configured_stage
                || row.requested_mode_count == largest_requested_stage)
    }) {
        println!(
            "truth={} boundary={} modes={} actual={} proj_err={:.4e} energy={:.4e} leakage={:.4e} boundary_frac={:.4e}",
            row.truth_family.as_str(),
            row.boundary_condition.as_str(),
            row.requested_mode_count,
            row.actual_mode_count,
            row.projection_relative_error,
            row.projected_energy_fraction,
            row.codifferential_leakage,
            row.boundary_lumped_energy_fraction
        );
    }
    println!(
        "rows={} runtime={:.3}s",
        rows.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn parse_mode_stages(value: &str) -> Vec<usize> {
    let modes = value
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|mode| *mode > 0)
        .collect::<Vec<_>>();
    if modes.is_empty() {
        vec![32, 64, 128, 256, 512, 1000]
    } else {
        modes
    }
}
