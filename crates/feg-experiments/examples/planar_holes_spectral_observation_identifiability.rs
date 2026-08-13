use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_hodge_flow, PlanarHolesBranchRecoverySummary, PlanarHolesCoexactTruthSource,
    PlanarHolesFlowConfig, PlanarHolesHarmonicTruthSource, PlanarHolesLocalObservationDesign,
    PlanarHolesModelKind, PlanarHolesScenarioResult, PlanarHolesTruthScaling,
};
use feg_core::HodgeBranchKind;
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Debug, Clone, Copy)]
struct IdentifiabilityCase {
    name: &'static str,
    observation_design: PlanarHolesLocalObservationDesign,
    sample_observation_noise: bool,
    local_noise_variance: f64,
    loop_noise_variance: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir =
        manifest_dir.join("../../out/planar_holes_spectral_observation_identifiability");
    fs::create_dir_all(&output_dir)?;
    let mode_count = env::var("PLANAR_HOLES_IDENTIFIABILITY_MODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    let mesh_size = env::var("PLANAR_HOLES_IDENTIFIABILITY_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.045);
    let cases = [
        IdentifiabilityCase {
            name: "600_sparse_noisy",
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
        IdentifiabilityCase {
            name: "600_sparse_noiseless",
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
        IdentifiabilityCase {
            name: "half_interior_noiseless",
            observation_design: PlanarHolesLocalObservationDesign::HalfInteriorEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
        IdentifiabilityCase {
            name: "half_interior_noisy",
            observation_design: PlanarHolesLocalObservationDesign::HalfInteriorEdges,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
        IdentifiabilityCase {
            name: "all_edges_noiseless",
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
        IdentifiabilityCase {
            name: "all_edges_noisy",
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
    ];
    let mut writer = BufWriter::new(File::create(
        output_dir.join("observation_identifiability.csv"),
    )?);
    writeln!(
        writer,
        "case,scenario,model,observation_design,sample_observation_noise,observation_count,l2_error,heldout_local_nlpd,codifferential_leakage,relative_harmonic_period_error,relative_total_annular_error,coexact_truth_norm,coexact_posterior_norm,coexact_relative_error,coexact_mass_correlation,harmonic_truth_norm,harmonic_posterior_norm,harmonic_relative_error,harmonic_mass_correlation"
    )?;

    println!("Planar holes spectral observation identifiability");
    println!("output_dir={}", output_dir.display());
    println!("mesh_size={mesh_size} mode_count={mode_count}");
    for (case_index, case) in cases.iter().enumerate() {
        let case_start = Instant::now();
        let case_dir = output_dir.join(case.name);
        let config = PlanarHolesFlowConfig {
            output_dir: case_dir,
            mesh_path: output_dir.join("planar_holes_spectral_identifiability.msh"),
            geo_path: output_dir.join("planar_holes_spectral_identifiability.geo"),
            force_mesh: case_index == 0,
            mesh_size,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.7,
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
        let result = run_planar_holes_hodge_flow(&config)?;
        for scenario in &result.scenarios {
            if let Some(metric) = scenario
                .metrics
                .iter()
                .find(|metric| metric.model == PlanarHolesModelKind::SpectralIncompressibleHodgeGp)
            {
                let coexact = branch_summary(
                    scenario,
                    PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
                    HodgeBranchKind::Coexact,
                );
                let harmonic = branch_summary(
                    scenario,
                    PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
                    HodgeBranchKind::Harmonic,
                );
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
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
                    metric.relative_total_annular_error,
                    coexact.map(|row| row.truth_mass_norm).unwrap_or(f64::NAN),
                    coexact.map(|row| row.posterior_mass_norm).unwrap_or(f64::NAN),
                    coexact.map(|row| row.relative_error).unwrap_or(f64::NAN),
                    coexact.map(|row| row.mass_correlation).unwrap_or(f64::NAN),
                    harmonic.map(|row| row.truth_mass_norm).unwrap_or(f64::NAN),
                    harmonic.map(|row| row.posterior_mass_norm).unwrap_or(f64::NAN),
                    harmonic.map(|row| row.relative_error).unwrap_or(f64::NAN),
                    harmonic.map(|row| row.mass_correlation).unwrap_or(f64::NAN)
                )?;
                println!(
                    "case={} scenario={} obs={} E1={:.4e} local_nlpd={:.4e} loop_err={:.4e} leakage={:.4e} runtime={:.2}s",
                    case.name,
                    scenario.scenario.as_str(),
                    metric.observation_count,
                    metric.l2_error,
                    metric.heldout_local_nlpd,
                    metric.relative_harmonic_period_error,
                    metric.codifferential_leakage,
                    case_start.elapsed().as_secs_f64()
                );
            }
        }
        writer.flush()?;
    }
    println!("total runtime: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn branch_summary(
    scenario: &PlanarHolesScenarioResult,
    model: PlanarHolesModelKind,
    branch: HodgeBranchKind,
) -> Option<&PlanarHolesBranchRecoverySummary> {
    scenario
        .branch_recovery_summaries
        .iter()
        .find(|summary| summary.model == model && summary.branch == branch)
}
