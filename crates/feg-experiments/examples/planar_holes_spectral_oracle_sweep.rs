use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_hodge_flow, PlanarHolesCoexactTruthSource, PlanarHolesFieldCoverageSubset,
    PlanarHolesFlowConfig, PlanarHolesHarmonicTruthSource, PlanarHolesLocalObservationDesign,
    PlanarHolesModelKind, PlanarHolesScenarioResult, PlanarHolesTruthScaling,
};
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Debug, Clone, Copy)]
struct OracleCase {
    name: &'static str,
    harmonic_truth_mass_norm: f64,
    observation_design: PlanarHolesLocalObservationDesign,
    sample_observation_noise: bool,
    local_noise_variance: f64,
    loop_noise_variance: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_spectral_oracle_sweep");
    fs::create_dir_all(&output_dir)?;
    let modes = env::var("PLANAR_HOLES_SPECTRAL_MODE_STAGES")
        .ok()
        .map(|value| parse_mode_stages(&value))
        .unwrap_or_else(|| vec![192, 384, 768, 1000]);
    let mesh_size = env::var("PLANAR_HOLES_SPECTRAL_SWEEP_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.045);
    let cases = [
        OracleCase {
            name: "pure_coexact_sparse_noisy",
            harmonic_truth_mass_norm: 0.0,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
        OracleCase {
            name: "pure_coexact_sparse_noiseless",
            harmonic_truth_mass_norm: 0.0,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
        OracleCase {
            name: "pure_coexact_half_noiseless",
            harmonic_truth_mass_norm: 0.0,
            observation_design: PlanarHolesLocalObservationDesign::HalfInteriorEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
        OracleCase {
            name: "pure_coexact_dense_noiseless",
            harmonic_truth_mass_norm: 0.0,
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
        OracleCase {
            name: "coexact_harmonic_dense_noiseless",
            harmonic_truth_mass_norm: 0.7,
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
        OracleCase {
            name: "coexact_harmonic_dense_noisy",
            harmonic_truth_mass_norm: 0.7,
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
    ];
    let mut writer = BufWriter::new(File::create(output_dir.join("spectral_oracle_sweep.csv"))?);
    writeln!(
        writer,
        "mode_count,case,scenario,model,observation_design,sample_observation_noise,observation_count,l2_error,heldout_local_nlpd,codifferential_leakage,relative_harmonic_period_error,field_coverage_95,field_p95_abs_z,coexact_actual_modes,coexact_expected_energy,coexact_projection_error,coexact_mahalanobis"
    )?;

    println!("Planar holes normalized spectral oracle sweep");
    println!("output_dir={}", output_dir.display());
    println!("mesh_size={mesh_size}");
    println!("mode_stages={modes:?}");

    for (mode_index, mode_count) in modes.iter().copied().enumerate() {
        for (case_index, case) in cases.iter().enumerate() {
            let case_start = Instant::now();
            let case_dir = output_dir.join(format!("k{mode_count}_{}", case.name));
            let mesh_path = output_dir.join(format!("planar_holes_spectral_k{mode_count}.msh"));
            let geo_path = output_dir.join(format!("planar_holes_spectral_k{mode_count}.geo"));
            let config = PlanarHolesFlowConfig {
                output_dir: case_dir.clone(),
                mesh_path,
                geo_path,
                force_mesh: case_index == 0,
                mesh_size,
                exact_truth_mass_norm: 0.0,
                coexact_truth_mass_norm: 1.0,
                harmonic_truth_mass_norm: case.harmonic_truth_mass_norm,
                coexact_truth_source: PlanarHolesCoexactTruthSource::SpectralGp,
                harmonic_truth_source: PlanarHolesHarmonicTruthSource::SpectralGp,
                truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
                local_observation_design: case.observation_design,
                sample_observation_noise: case.sample_observation_noise,
                local_noise_variance: case.local_noise_variance,
                loop_noise_variance: case.loop_noise_variance,
                include_incompressible_hodge_model: false,
                include_spectral_hodge_model: false,
                include_spectral_incompressible_hodge_model: true,
                spectral_branch_energy_normalization: true,
                spectral_exact_mode_count: mode_count,
                spectral_coexact_mode_count: mode_count,
                spectral_harmonic_mode_count: 3,
                spectral_exact_expected_m1_energy: 1.0,
                spectral_coexact_expected_m1_energy: 1.0,
                spectral_harmonic_expected_m1_energy: case.harmonic_truth_mass_norm
                    * case.harmonic_truth_mass_norm,
                ..PlanarHolesFlowConfig::default()
            };
            let result = run_planar_holes_hodge_flow(&config)?;
            let coexact_diag = result
                .spectral_branch_diagnostics
                .iter()
                .find(|row| {
                    row.model == PlanarHolesModelKind::SpectralIncompressibleHodgeGp
                        && row.branch.as_str() == "coexact"
                })
                .cloned();
            for scenario in &result.scenarios {
                if let Some(metric) = scenario.metrics.iter().find(|metric| {
                    metric.model == PlanarHolesModelKind::SpectralIncompressibleHodgeGp
                }) {
                    let field = field_summary(
                        scenario,
                        PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
                        PlanarHolesFieldCoverageSubset::AllEdges,
                    );
                    writeln!(
                        writer,
                        "{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12}",
                        mode_count,
                        case.name,
                        scenario.scenario.as_str(),
                        metric.model.as_str(),
                        case.observation_design.as_str(),
                        case.sample_observation_noise,
                        metric.observation_count,
                        metric.l2_error,
                        metric.heldout_local_nlpd,
                        metric.codifferential_leakage,
                        metric.relative_harmonic_period_error,
                        field.map(|row| row.coverage_95).unwrap_or(f64::NAN),
                        field.map(|row| row.p95_abs_z).unwrap_or(f64::NAN),
                        coexact_diag
                            .as_ref()
                            .map(|row| row.actual_mode_count)
                            .unwrap_or(0),
                        coexact_diag
                            .as_ref()
                            .map(|row| row.expected_m1_energy)
                            .unwrap_or(f64::NAN),
                        coexact_diag
                            .as_ref()
                            .map(|row| row.projection_relative_error)
                            .unwrap_or(f64::NAN),
                        coexact_diag
                            .as_ref()
                            .map(|row| row.projected_truth_mahalanobis_norm)
                            .unwrap_or(f64::NAN),
                    )?;
                    println!(
                        "mode={} case={} scenario={} E1={:.4e} leakage={:.4e} proj_err={:.4e} actual_modes={} runtime={:.2}s",
                        mode_count,
                        case.name,
                        scenario.scenario.as_str(),
                        metric.l2_error,
                        metric.codifferential_leakage,
                        coexact_diag
                            .as_ref()
                            .map(|row| row.projection_relative_error)
                            .unwrap_or(f64::NAN),
                        coexact_diag
                            .as_ref()
                            .map(|row| row.actual_mode_count)
                            .unwrap_or(0),
                        case_start.elapsed().as_secs_f64()
                    );
                }
            }
        }
        writer.flush()?;
        println!(
            "completed mode stage {} of {} in {:.2}s",
            mode_index + 1,
            modes.len(),
            start.elapsed().as_secs_f64()
        );
    }
    println!("total runtime: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn parse_mode_stages(value: &str) -> Vec<usize> {
    let modes = value
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|mode| *mode > 0)
        .collect::<Vec<_>>();
    if modes.is_empty() {
        vec![192, 384, 768, 1000]
    } else {
        modes
    }
}

fn field_summary(
    scenario: &PlanarHolesScenarioResult,
    model: PlanarHolesModelKind,
    subset: PlanarHolesFieldCoverageSubset,
) -> Option<&feg_case_studies::planar_holes_hodge_flow::PlanarHolesFieldCoverageSummary> {
    scenario
        .field_coverage_summaries
        .iter()
        .find(|summary| summary.model == model && summary.subset == subset)
}
