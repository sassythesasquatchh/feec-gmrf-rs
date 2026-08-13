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
    let output_dir = manifest_dir.join("../../out/planar_holes_exact_mass_coexact_prior_isolation");
    fs::create_dir_all(&output_dir)?;

    let mode_count = env::var("PLANAR_HOLES_EXACT_MASS_ISOLATION_MODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    let budget = env::var("PLANAR_HOLES_EXACT_MASS_ISOLATION_BUDGET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(600);
    let mesh_size = env::var("PLANAR_HOLES_EXACT_MASS_ISOLATION_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.045);
    let variance_scales = env::var("PLANAR_HOLES_EXACT_MASS_ISOLATION_SCALES")
        .ok()
        .map(|value| parse_f64_list(&value))
        .unwrap_or_else(|| vec![10.0, 1.0, 0.1, 0.01]);
    let truth_sources = env::var("PLANAR_HOLES_EXACT_MASS_ISOLATION_TRUTHS")
        .ok()
        .map(|value| parse_truth_sources(&value))
        .unwrap_or_else(|| {
            vec![
                PlanarHolesCoexactTruthSource::DirichletStreamfunction,
                PlanarHolesCoexactTruthSource::ExactLowerMassCoexact,
                PlanarHolesCoexactTruthSource::ExactMassCoexact,
            ]
        });
    let compute_field_coverage = env::var("PLANAR_HOLES_EXACT_MASS_ISOLATION_FIELD_COVERAGE")
        .map(|value| value != "0" && value.to_lowercase() != "false")
        .unwrap_or(true);

    let base_local_variance = 1e-4;
    let base_loop_variance = 1e-5;
    let base_interior_loop_variance = 1e-5;
    let base_long_path_variance = 1e-4;

    let csv_path = output_dir.join("exact_mass_coexact_prior_isolation.csv");
    let mut writer = BufWriter::new(File::create(&csv_path)?);
    writeln!(
        writer,
        "truth_source,variance_scale,local_noise_variance,loop_noise_variance,interior_loop_noise_variance,long_path_noise_variance,design,model,model_coexact_tau_scale,observation_count,edge_observation_count,interior_loop_observation_count,long_path_observation_count,harmonic_period_observation_count,l2_error,heldout_local_nlpd,heldout_interior_loop_nlpd,heldout_long_path_nlpd,heldout_harmonic_period_nlpd,heldout_interior_loop_relative_error,heldout_long_path_relative_error,heldout_harmonic_period_relative_error,codifferential_leakage,coexact_relative_error,coexact_mass_correlation,harmonic_relative_error,harmonic_mass_correlation,all_edge_coverage_95,heldout_edge_coverage_95"
    )?;

    println!("Planar holes exact-mass coexact prior isolation");
    println!("output_dir={}", output_dir.display());
    println!(
        "mesh_size={mesh_size} mode_count={mode_count} budget={budget} field_coverage={compute_field_coverage}"
    );
    println!("variance_scales={variance_scales:?}");
    println!(
        "truth_sources={:?}",
        truth_sources
            .iter()
            .map(|source| source.as_str())
            .collect::<Vec<_>>()
    );

    for (truth_index, truth_source) in truth_sources.iter().copied().enumerate() {
        for (scale_index, variance_scale) in variance_scales.iter().copied().enumerate() {
            if !variance_scale.is_finite() || variance_scale <= 0.0 {
                return Err(format!(
                    "variance scale must be finite and positive: {variance_scale}"
                )
                .into());
            }
            let run_start = Instant::now();
            let run_dir = output_dir.join(format!(
                "{}_scale_{variance_scale:.0e}",
                truth_source.as_str()
            ));
            let base = PlanarHolesFlowConfig {
                output_dir: run_dir,
                mesh_path: output_dir.join("planar_holes_exact_mass_coexact_prior_isolation.msh"),
                geo_path: output_dir.join("planar_holes_exact_mass_coexact_prior_isolation.geo"),
                force_mesh: truth_index == 0 && scale_index == 0,
                mesh_size,
                exact_truth_mass_norm: 0.0,
                coexact_truth_mass_norm: 1.0,
                harmonic_truth_mass_norm: 0.7,
                coexact_truth_source: truth_source,
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
                include_spectral_hodge_model: false,
                include_spectral_incompressible_hodge_model: true,
                compute_field_coverage,
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
                designs: vec![PlanarHolesSensorDesignKind::Hybrid],
                ..PlanarHolesSensorDesignSweepConfig::default()
            };
            let rows = run_planar_holes_sensor_design_sweep(&config)?;
            for row in rows.iter().filter(|row| keep_model(row.model)) {
                write_isolation_row(
                    &mut writer,
                    truth_source,
                    variance_scale,
                    base_local_variance * variance_scale,
                    base_loop_variance * variance_scale,
                    base_interior_loop_variance * variance_scale,
                    base_long_path_variance * variance_scale,
                    row,
                )?;
            }
            writer.flush()?;
            print_run_summary(truth_source, variance_scale, &rows);
            println!(
                "completed truth={} variance_scale={variance_scale:.3e} in {:.2}s",
                truth_source.as_str(),
                run_start.elapsed().as_secs_f64()
            );
        }
    }

    println!("wrote {}", csv_path.display());
    println!("total runtime: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn keep_model(model: PlanarHolesModelKind) -> bool {
    matches!(
        model,
        PlanarHolesModelKind::IncompressibleHodgeMatern
            | PlanarHolesModelKind::ExactMassIncompressibleHodgeMatern
            | PlanarHolesModelKind::SparseLowerTraceMatchedIncompressibleHodgeMatern
            | PlanarHolesModelKind::ExactLowerIncompressibleHodgeMatern
            | PlanarHolesModelKind::ExactLowerTraceMatchedIncompressibleHodgeMatern
            | PlanarHolesModelKind::SpectralIncompressibleHodgeGp
            | PlanarHolesModelKind::NondecomposedFeec
    )
}

#[allow(clippy::too_many_arguments)]
fn write_isolation_row(
    writer: &mut BufWriter<File>,
    truth_source: PlanarHolesCoexactTruthSource,
    variance_scale: f64,
    local_noise_variance: f64,
    loop_noise_variance: f64,
    interior_loop_noise_variance: f64,
    long_path_noise_variance: f64,
    row: &PlanarHolesSensorDesignSweepRow,
) -> std::io::Result<()> {
    writeln!(
        writer,
        "{},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{:.12},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
        truth_source.as_str(),
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

fn print_run_summary(
    truth_source: PlanarHolesCoexactTruthSource,
    variance_scale: f64,
    rows: &[PlanarHolesSensorDesignSweepRow],
) {
    for row in rows.iter().filter(|row| keep_model(row.model)) {
        println!(
            "truth={} scale={variance_scale:.3e} model={} tau_scale={:.4e} E1={:.4e} local_nlpd={:.4e} coexact_err={:.4e} leakage={:.4e} all_cov={:.3}",
            truth_source.as_str(),
            row.model.as_str(),
            row.model_coexact_tau_scale,
            row.l2_error,
            row.heldout_local_nlpd,
            row.coexact_relative_error,
            row.codifferential_leakage,
            row.all_edge_coverage_95
        );
    }
}

fn parse_f64_list(value: &str) -> Vec<f64> {
    let parsed = value
        .split(',')
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if parsed.is_empty() {
        vec![10.0, 1.0, 0.1, 0.01]
    } else {
        parsed
    }
}

fn parse_truth_sources(value: &str) -> Vec<PlanarHolesCoexactTruthSource> {
    let mut sources = Vec::new();
    for part in value.split(',').map(str::trim) {
        let source = match part {
            "dirichlet_streamfunction" | "streamfunction" => {
                Some(PlanarHolesCoexactTruthSource::DirichletStreamfunction)
            }
            "exact_mass_coexact" | "exact_mass" => {
                Some(PlanarHolesCoexactTruthSource::ExactMassCoexact)
            }
            "exact_lower_mass_coexact" | "exact_lower" | "exact_lower_mass" => {
                Some(PlanarHolesCoexactTruthSource::ExactLowerMassCoexact)
            }
            "sparse_anchor" | "sparse" => Some(PlanarHolesCoexactTruthSource::SparseAnchor),
            "spectral_gp" | "spectral" => Some(PlanarHolesCoexactTruthSource::SpectralGp),
            _ => None,
        };
        if let Some(source) = source {
            sources.push(source);
        }
    }
    if sources.is_empty() {
        vec![PlanarHolesCoexactTruthSource::DirichletStreamfunction]
    } else {
        sources
    }
}
