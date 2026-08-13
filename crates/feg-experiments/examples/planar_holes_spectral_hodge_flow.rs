use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_hodge_flow, write_planar_holes_hodge_flow_outputs,
    PlanarHolesCoexactTruthSource, PlanarHolesFieldCoverageSubset, PlanarHolesFlowConfig,
    PlanarHolesHarmonicTruthSource, PlanarHolesLocalObservationDesign, PlanarHolesModelKind,
    PlanarHolesScenarioResult, PlanarHolesTruthScaling,
};
use std::{env, path::PathBuf, time::Instant};

#[derive(Debug, Clone, Copy)]
struct SpectralCase {
    name: &'static str,
    exact_truth_mass_norm: f64,
    coexact_truth_mass_norm: f64,
    harmonic_truth_mass_norm: f64,
    coexact_truth_source: PlanarHolesCoexactTruthSource,
    harmonic_truth_source: PlanarHolesHarmonicTruthSource,
    truth_scaling: PlanarHolesTruthScaling,
    observation_design: PlanarHolesLocalObservationDesign,
    sample_observation_noise: bool,
    local_noise_variance: f64,
    loop_noise_variance: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_spectral_hodge_flow");
    let mesh_path = output_dir.join("planar_holes_spectral.msh");
    let geo_path = output_dir.join("planar_holes_spectral.geo");
    let mode_count = env::var("PLANAR_HOLES_SPECTRAL_MODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(192);
    let fine_mesh = env::var("PLANAR_HOLES_SPECTRAL_FINE")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    let mesh_size = if fine_mesh { 0.03 } else { 0.045 };
    let cases = [
        SpectralCase {
            name: "mixed_sparse_truth",
            exact_truth_mass_norm: 1.0,
            coexact_truth_mass_norm: 0.8,
            harmonic_truth_mass_norm: 0.7,
            coexact_truth_source: PlanarHolesCoexactTruthSource::SparseAnchor,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
        SpectralCase {
            name: "incompressible_exact_mass_truth",
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.7,
            coexact_truth_source: PlanarHolesCoexactTruthSource::ExactMassCoexact,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::CanonicalFixed,
            truth_scaling: PlanarHolesTruthScaling::MassNormTargets,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
        SpectralCase {
            name: "matched_spectral_coexact_dense_noiseless",
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.0,
            coexact_truth_source: PlanarHolesCoexactTruthSource::SpectralGp,
            harmonic_truth_source: PlanarHolesHarmonicTruthSource::SpectralGp,
            truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: 1e-10,
            loop_noise_variance: 1e-10,
        },
    ];

    println!("Planar holes spectral Hodge GP comparison");
    println!("output_dir={}", output_dir.display());
    println!("spectral_exact_modes={mode_count} spectral_coexact_modes={mode_count}");
    println!("mesh_size={mesh_size}");

    for (index, case) in cases.iter().enumerate() {
        let case_start = Instant::now();
        let case_dir = output_dir.join(case.name);
        let config = PlanarHolesFlowConfig {
            output_dir: case_dir.clone(),
            mesh_path: mesh_path.clone(),
            geo_path: geo_path.clone(),
            force_mesh: index == 0,
            mesh_size,
            exact_truth_mass_norm: case.exact_truth_mass_norm,
            coexact_truth_mass_norm: case.coexact_truth_mass_norm,
            harmonic_truth_mass_norm: case.harmonic_truth_mass_norm,
            coexact_truth_source: case.coexact_truth_source,
            harmonic_truth_source: case.harmonic_truth_source,
            truth_scaling: case.truth_scaling,
            local_observation_design: case.observation_design,
            sample_observation_noise: case.sample_observation_noise,
            local_noise_variance: case.local_noise_variance,
            loop_noise_variance: case.loop_noise_variance,
            include_incompressible_hodge_model: true,
            include_spectral_hodge_model: true,
            include_spectral_incompressible_hodge_model: true,
            spectral_exact_mode_count: mode_count,
            spectral_coexact_mode_count: mode_count,
            spectral_harmonic_mode_count: 3,
            spectral_branch_energy_normalization: true,
            spectral_exact_expected_m1_energy: case.exact_truth_mass_norm
                * case.exact_truth_mass_norm,
            spectral_coexact_expected_m1_energy: case.coexact_truth_mass_norm
                * case.coexact_truth_mass_norm,
            spectral_harmonic_expected_m1_energy: case.harmonic_truth_mass_norm
                * case.harmonic_truth_mass_norm,
            ..PlanarHolesFlowConfig::default()
        };
        let result = run_planar_holes_hodge_flow(&config)?;
        write_planar_holes_hodge_flow_outputs(&result, &config.output_dir)?;
        println!(
            "case={} topology: vertices={} edges={} faces={} b1={} train_rank={} heldout_rank={}",
            case.name,
            result.topology_summary.vertex_count,
            result.topology_summary.edge_count,
            result.topology_summary.face_count,
            result.topology_summary.b1,
            result.cycle_harmonic_pairing_rank,
            result.heldout_cycle_harmonic_pairing_rank
        );
        for scenario in &result.scenarios {
            for model in [
                PlanarHolesModelKind::HodgeMatern,
                PlanarHolesModelKind::IncompressibleHodgeMatern,
                PlanarHolesModelKind::SpectralHodgeGp,
                PlanarHolesModelKind::SpectralIncompressibleHodgeGp,
                PlanarHolesModelKind::NondecomposedFeec,
            ] {
                if let Some(metric) = scenario.metrics.iter().find(|metric| metric.model == model) {
                    let all_field =
                        field_summary(scenario, model, PlanarHolesFieldCoverageSubset::AllEdges);
                    println!(
                        "case={} scenario={} model={} rel_l2={:.4e} local_nlpd={:.4e} loop_nlpd={:.4e} harmonic_nlpd={:.4e} delta_metric={:.4e} delta_kind={} rel_loop={:.4e} rel_harmonic_period={:.4e} rel_coexact_annular={:.4e} field_cov95={:.3} field_p95_abs_z={:.3e}",
                        case.name,
                        scenario.scenario.as_str(),
                        model.as_str(),
                        metric.l2_error,
                        metric.heldout_local_nlpd,
                        metric.heldout_loop_nlpd,
                        metric.heldout_harmonic_period_nlpd,
                        metric.codifferential_error,
                        metric.codifferential_metric_kind.as_str(),
                        metric.relative_circulation_error,
                        metric.relative_harmonic_period_error,
                        metric.relative_coexact_annular_error,
                        all_field
                            .map(|summary| summary.coverage_95)
                            .unwrap_or(f64::NAN),
                        all_field
                            .map(|summary| summary.p95_abs_z)
                            .unwrap_or(f64::NAN)
                    );
                }
            }
        }
        for row in &result.spectral_branch_diagnostics {
            println!(
                "case={} spectral_diag model={} branch={} requested={} actual={} expected_energy={:.4e} target_energy={:.4e} projection_rel_error={:.4e} projected_mahalanobis={:.4e}",
                case.name,
                row.model.as_str(),
                row.branch.as_str(),
                row.requested_mode_count,
                row.actual_mode_count,
                row.expected_m1_energy,
                row.target_expected_m1_energy,
                row.projection_relative_error,
                row.projected_truth_mahalanobis_norm
            );
        }
        println!(
            "case={} output_dir={} runtime={:.3}s",
            case.name,
            case_dir.display(),
            case_start.elapsed().as_secs_f64()
        );
    }

    println!("total runtime: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
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
