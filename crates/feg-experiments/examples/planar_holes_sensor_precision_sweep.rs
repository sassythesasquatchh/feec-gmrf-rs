use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_sensor_design_sweep, PlanarHolesCoexactTruthSource, PlanarHolesFlowConfig,
    PlanarHolesHarmonicTruthSource, PlanarHolesModelKind, PlanarHolesSensorDesignKind,
    PlanarHolesSensorDesignSweepConfig, PlanarHolesSensorDesignSweepRow, PlanarHolesTruthScaling,
};
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_sensor_precision_sweep");
    fs::create_dir_all(&output_dir)?;

    let mode_count = env::var("PLANAR_HOLES_PRECISION_SWEEP_MODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    let budget = env::var("PLANAR_HOLES_PRECISION_SWEEP_BUDGET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(600);
    let mesh_size = env::var("PLANAR_HOLES_PRECISION_SWEEP_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.045);
    let variance_scales = env::var("PLANAR_HOLES_PRECISION_SWEEP_SCALES")
        .ok()
        .map(|value| parse_f64_list(&value))
        .unwrap_or_else(|| vec![100.0, 10.0, 1.0, 0.1, 0.01, 0.001]);
    let designs = env::var("PLANAR_HOLES_PRECISION_SWEEP_DESIGNS")
        .ok()
        .map(|value| parse_designs(&value))
        .unwrap_or_else(|| vec![PlanarHolesSensorDesignKind::Hybrid]);

    let base_local_variance = 1e-4;
    let base_loop_variance = 1e-5;
    let base_interior_loop_variance = 1e-5;
    let base_long_path_variance = 1e-4;

    let mut writer = BufWriter::new(File::create(output_dir.join("sensor_precision_sweep.csv"))?);
    writeln!(
        writer,
        "variance_scale,local_noise_variance,loop_noise_variance,interior_loop_noise_variance,long_path_noise_variance,design,model,model_coexact_tau_scale,observation_count,edge_observation_count,interior_loop_observation_count,long_path_observation_count,harmonic_period_observation_count,l2_error,heldout_local_nlpd,heldout_interior_loop_nlpd,heldout_long_path_nlpd,heldout_harmonic_period_nlpd,heldout_interior_loop_relative_error,heldout_long_path_relative_error,heldout_harmonic_period_relative_error,codifferential_leakage,coexact_relative_error,coexact_mass_correlation,harmonic_relative_error,harmonic_mass_correlation,all_edge_coverage_95,heldout_edge_coverage_95"
    )?;

    println!("Planar holes sensor precision sweep");
    println!("output_dir={}", output_dir.display());
    println!("mesh_size={mesh_size} mode_count={mode_count} budget={budget}");
    println!(
        "designs={:?}",
        designs
            .iter()
            .map(|design| design.as_str())
            .collect::<Vec<_>>()
    );
    println!("variance_scales={variance_scales:?}");
    println!(
        "base variances: local={base_local_variance} loop={base_loop_variance} interior_loop={base_interior_loop_variance} long_path={base_long_path_variance}"
    );

    for (scale_index, variance_scale) in variance_scales.iter().copied().enumerate() {
        if !variance_scale.is_finite() || variance_scale <= 0.0 {
            return Err(
                format!("variance scale must be finite and positive: {variance_scale}").into(),
            );
        }
        let scale_start = Instant::now();
        let scale_dir = output_dir.join(format!("scale_{variance_scale:.0e}"));
        let base = PlanarHolesFlowConfig {
            output_dir: scale_dir,
            mesh_path: output_dir.join("planar_holes_sensor_precision_sweep.msh"),
            geo_path: output_dir.join("planar_holes_sensor_precision_sweep.geo"),
            force_mesh: scale_index == 0,
            mesh_size,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.7,
            coexact_truth_source: PlanarHolesCoexactTruthSource::DirichletStreamfunction,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            sample_observation_noise: false,
            local_noise_variance: base_local_variance * variance_scale,
            loop_noise_variance: base_loop_variance * variance_scale,
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
            interior_loop_noise_variance: base_interior_loop_variance * variance_scale,
            long_path_noise_variance: base_long_path_variance * variance_scale,
            designs: designs.clone(),
            ..PlanarHolesSensorDesignSweepConfig::default()
        };
        let rows = run_planar_holes_sensor_design_sweep(&config)?;
        for row in &rows {
            write_precision_row(
                &mut writer,
                variance_scale,
                base_local_variance * variance_scale,
                base_loop_variance * variance_scale,
                base_interior_loop_variance * variance_scale,
                base_long_path_variance * variance_scale,
                row,
            )?;
        }
        writer.flush()?;
        print_scale_summary(variance_scale, &rows);
        println!(
            "completed variance_scale={variance_scale:.3e} in {:.2}s",
            scale_start.elapsed().as_secs_f64()
        );
    }

    println!("total runtime: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn write_precision_row(
    writer: &mut BufWriter<File>,
    variance_scale: f64,
    local_noise_variance: f64,
    loop_noise_variance: f64,
    interior_loop_noise_variance: f64,
    long_path_noise_variance: f64,
    row: &PlanarHolesSensorDesignSweepRow,
) -> std::io::Result<()> {
    writeln!(
        writer,
        "{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{:.12},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
        variance_scale,
        local_noise_variance,
        loop_noise_variance,
        interior_loop_noise_variance,
        long_path_noise_variance,
        row.design.as_str(),
        row.model.as_str(),
        row.model_coexact_tau_scale,
        row.observation_count,
        row.edge_observation_count,
        row.interior_loop_observation_count,
        row.long_path_observation_count,
        row.harmonic_period_observation_count,
        row.l2_error,
        row.heldout_local_nlpd,
        row.heldout_interior_loop_nlpd,
        row.heldout_long_path_nlpd,
        row.heldout_harmonic_period_nlpd,
        row.heldout_interior_loop_relative_error,
        row.heldout_long_path_relative_error,
        row.heldout_harmonic_period_relative_error,
        row.codifferential_leakage,
        row.coexact_relative_error,
        row.coexact_mass_correlation,
        row.harmonic_relative_error,
        row.harmonic_mass_correlation,
        row.all_edge_coverage_95,
        row.heldout_edge_coverage_95
    )
}

fn print_scale_summary(variance_scale: f64, rows: &[PlanarHolesSensorDesignSweepRow]) {
    for design in [
        PlanarHolesSensorDesignKind::SparseEdges,
        PlanarHolesSensorDesignKind::Hybrid,
    ] {
        for row in rows.iter().filter(|row| row.design == design) {
            if matches!(
                row.model,
                PlanarHolesModelKind::IncompressibleHodgeMatern
                    | PlanarHolesModelKind::ExactMassIncompressibleHodgeMatern
                    | PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
                    | PlanarHolesModelKind::ExactLowerIncompressibleHodgeMatern
                    | PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
                    | PlanarHolesModelKind::SpectralIncompressibleHodgeGp
                    | PlanarHolesModelKind::NondecomposedFeec
            ) {
                println!(
                    "scale={variance_scale:.3e} design={} model={} tau_scale={:.4e} E1={:.4e} local_nlpd={:.4e} int_loop_err={:.4e} long_err={:.4e} harm_err={:.4e} leakage={:.4e}",
                    design.as_str(),
                    row.model.as_str(),
                    row.model_coexact_tau_scale,
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
}

fn parse_f64_list(value: &str) -> Vec<f64> {
    let parsed = value
        .split(',')
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if parsed.is_empty() {
        vec![100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    } else {
        parsed
    }
}

fn parse_designs(value: &str) -> Vec<PlanarHolesSensorDesignKind> {
    let mut designs = Vec::new();
    for part in value.split(',').map(str::trim) {
        let design = match part {
            "sparse_edges" => Some(PlanarHolesSensorDesignKind::SparseEdges),
            "edges_hole_periods" => Some(PlanarHolesSensorDesignKind::EdgesHolePeriods),
            "edges_small_interior_loops" => {
                Some(PlanarHolesSensorDesignKind::EdgesSmallInteriorLoops)
            }
            "edges_multiscale_interior_loops" => {
                Some(PlanarHolesSensorDesignKind::EdgesMultiscaleInteriorLoops)
            }
            "edges_long_paths" => Some(PlanarHolesSensorDesignKind::EdgesLongPaths),
            "hybrid_edges_loops_paths_periods" | "hybrid" => {
                Some(PlanarHolesSensorDesignKind::Hybrid)
            }
            _ => None,
        };
        if let Some(design) = design {
            designs.push(design);
        }
    }
    if designs.is_empty() {
        vec![PlanarHolesSensorDesignKind::Hybrid]
    } else {
        designs
    }
}
