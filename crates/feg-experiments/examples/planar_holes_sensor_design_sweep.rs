use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_sensor_design_sweep, write_planar_holes_sensor_design_sweep,
    PlanarHolesCoexactTruthSource, PlanarHolesFlowConfig, PlanarHolesHarmonicTruthSource,
    PlanarHolesModelKind, PlanarHolesSensorDesignKind, PlanarHolesSensorDesignSweepConfig,
    PlanarHolesTruthScaling,
};
use std::{env, fs, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_sensor_design_sweep");
    fs::create_dir_all(&output_dir)?;
    let mode_count = env::var("PLANAR_HOLES_SENSOR_SWEEP_MODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    let budget = env::var("PLANAR_HOLES_SENSOR_SWEEP_BUDGET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(600);
    let mesh_size = env::var("PLANAR_HOLES_SENSOR_SWEEP_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.045);
    let sample_noise = !env::var("PLANAR_HOLES_SENSOR_SWEEP_NOISELESS")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));

    let base = PlanarHolesFlowConfig {
        output_dir: output_dir.clone(),
        mesh_path: output_dir.join("planar_holes_sensor_sweep.msh"),
        geo_path: output_dir.join("planar_holes_sensor_sweep.geo"),
        force_mesh: true,
        mesh_size,
        exact_truth_mass_norm: 0.0,
        coexact_truth_mass_norm: 1.0,
        harmonic_truth_mass_norm: 0.7,
        coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
        harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
        truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
        sample_observation_noise: sample_noise,
        include_incompressible_hodge_model: true,
        include_exact_mass_incompressible_hodge_model: true,
        include_sparse_lower_trace_matched_incompressible_hodge_model: true,
        include_exact_lower_incompressible_hodge_model: true,
        include_exact_lower_trace_matched_incompressible_hodge_model: true,
        include_spectral_hodge_model: true,
        include_spectral_incompressible_hodge_model: true,
        compute_field_coverage: false,
        spectral_branch_energy_normalization: true,
        spectral_exact_mode_count: mode_count,
        spectral_coexact_mode_count: mode_count,
        spectral_harmonic_mode_count: 3,
        spectral_exact_expected_m1_energy: 1.0,
        spectral_coexact_expected_m1_energy: 1.0,
        spectral_harmonic_expected_m1_energy: 0.49,
        ..PlanarHolesFlowConfig::default()
    };
    let config = PlanarHolesSensorDesignSweepConfig {
        base,
        total_observation_budget: budget,
        designs: PlanarHolesSensorDesignKind::all().to_vec(),
        ..PlanarHolesSensorDesignSweepConfig::default()
    };

    println!("Planar holes sensor-design sweep");
    println!("output_dir={}", output_dir.display());
    println!("mesh_size={mesh_size} mode_count={mode_count} budget={budget} noise={sample_noise}");
    let rows = run_planar_holes_sensor_design_sweep(&config)?;
    write_planar_holes_sensor_design_sweep(&rows, output_dir.join("sensor_design_sweep.csv"))?;
    for design in PlanarHolesSensorDesignKind::all() {
        for model in [
            PlanarHolesModelKind::IncompressibleHodgeMatern,
            PlanarHolesModelKind::ExactMassIncompressibleHodgeMatern,
            PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern,
            PlanarHolesModelKind::ExactLowerIncompressibleHodgeMatern,
            PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern,
            PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
            PlanarHolesModelKind::NondecomposedFeec,
            PlanarHolesModelKind::ComponentwiseMatern,
        ] {
            if let Some(row) = rows
                .iter()
                .find(|row| row.design == design && row.model == model)
            {
                println!(
                    "design={} model={} tau_scale={:.4e} obs={} E1={:.4e} local_nlpd={:.4e} int_loop_err={:.4e} long_err={:.4e} harm_err={:.4e} leakage={:.4e}",
                    design.as_str(),
                    model.as_str(),
                    row.model_coexact_tau_scale,
                    row.observation_count,
                    row.l2_error,
                    row.heldout_local_nlpd,
                    row.heldout_interior_loop_relative_error,
                    row.heldout_long_path_relative_error,
                    row.heldout_harmonic_period_relative_error,
                    row.codifferential_leakage
                );
            }
        }
    }
    println!(
        "rows={} runtime={:.3}s",
        rows.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
