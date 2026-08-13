use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_hodge_flow, PlanarHolesBranchRecoverySummary, PlanarHolesCoexactTruthSource,
    PlanarHolesFieldCoverageSubset, PlanarHolesFieldCoverageSummary, PlanarHolesFlowConfig,
    PlanarHolesModelKind, PlanarHolesScenarioResult,
};
use feg_core::HodgeBranchKind;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const COEXACT_TAU_SCALES: [f64; 4] = [1.0, 0.3, 0.1, 0.03];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_coexact_bias_sweep");
    fs::create_dir_all(&output_dir)?;
    let mesh_path = output_dir.join("planar_holes_coexact_bias_sweep.msh");
    let geo_path = output_dir.join("planar_holes_coexact_bias_sweep.geo");
    let csv_path = output_dir.join("coexact_bias_sweep.csv");
    let mut writer = BufWriter::new(File::create(&csv_path)?);
    writeln!(
        writer,
        "coexact_tau_scale,scenario,model,l2_error,heldout_local_nlpd,codifferential_leakage,all_edge_coverage_95,heldout_edge_coverage_95,coexact_truth_norm,coexact_posterior_norm,coexact_relative_branch_error,coexact_mass_correlation,relative_harmonic_period_error"
    )?;

    println!("Planar holes coexact-bias scaling sweep");
    println!("output_dir={}", output_dir.display());
    for (index, coexact_tau_scale) in COEXACT_TAU_SCALES.iter().copied().enumerate() {
        let run_start = Instant::now();
        let scale_label = scale_label(coexact_tau_scale);
        let config = PlanarHolesFlowConfig {
            output_dir: output_dir.join(format!("coexact_tau_scale_{scale_label}")),
            mesh_path: mesh_path.clone(),
            geo_path: geo_path.clone(),
            force_mesh: index == 0,
            exact_truth_mass_norm: 0.0,
            coexact_truth_mass_norm: 1.0,
            harmonic_truth_mass_norm: 0.7,
            coexact_truth_source: PlanarHolesCoexactTruthSource::ExactMassCoexact,
            include_incompressible_hodge_model: true,
            exact_tau_scale: 1.0,
            coexact_tau_scale,
            local_observation_count: 600,
            heldout_local_count: 200,
            ..PlanarHolesFlowConfig::default()
        };
        let result = run_planar_holes_hodge_flow(&config)?;
        println!(
            "scale={coexact_tau_scale:.3} topology: vertices={} edges={} faces={} b1={}",
            result.topology_summary.vertex_count,
            result.topology_summary.edge_count,
            result.topology_summary.face_count,
            result.topology_summary.b1
        );
        for scenario in &result.scenarios {
            for metric in &scenario.metrics {
                let all_field = field_summary(
                    scenario,
                    metric.model,
                    PlanarHolesFieldCoverageSubset::AllEdges,
                );
                let heldout_field = field_summary(
                    scenario,
                    metric.model,
                    PlanarHolesFieldCoverageSubset::HeldoutLocalEdges,
                );
                let coexact = branch_summary(scenario, metric.model, HodgeBranchKind::Coexact);
                writeln!(
                    writer,
                    "{:.12},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                    coexact_tau_scale,
                    metric.scenario.as_str(),
                    metric.model.as_str(),
                    metric.l2_error,
                    metric.heldout_local_nlpd,
                    metric.codifferential_leakage,
                    all_field
                        .map(|summary| summary.coverage_95)
                        .unwrap_or(f64::NAN),
                    heldout_field
                        .map(|summary| summary.coverage_95)
                        .unwrap_or(f64::NAN),
                    coexact
                        .map(|summary| summary.truth_mass_norm)
                        .unwrap_or(f64::NAN),
                    coexact
                        .map(|summary| summary.posterior_mass_norm)
                        .unwrap_or(f64::NAN),
                    coexact
                        .map(|summary| summary.relative_error)
                        .unwrap_or(f64::NAN),
                    coexact
                        .map(|summary| summary.mass_correlation)
                        .unwrap_or(f64::NAN),
                    metric.relative_harmonic_period_error,
                )?;
                if matches!(
                    metric.model,
                    PlanarHolesModelKind::HodgeMatern
                        | PlanarHolesModelKind::IncompressibleHodgeMatern
                ) {
                    println!(
                        "scale={coexact_tau_scale:.3} scenario={} model={} E1={:.4e} leakage={:.4e} coexact_norm={:.4e} coexact_rel_err={:.4e} coexact_corr={:.4e} field_cov={:.3}",
                        metric.scenario.as_str(),
                        metric.model.as_str(),
                        metric.l2_error,
                        metric.codifferential_leakage,
                        coexact
                            .map(|summary| summary.posterior_mass_norm)
                            .unwrap_or(f64::NAN),
                        coexact
                            .map(|summary| summary.relative_error)
                            .unwrap_or(f64::NAN),
                        coexact
                            .map(|summary| summary.mass_correlation)
                            .unwrap_or(f64::NAN),
                        all_field
                            .map(|summary| summary.coverage_95)
                            .unwrap_or(f64::NAN),
                    );
                }
            }
        }
        println!(
            "scale={coexact_tau_scale:.3} runtime={:.3}s",
            run_start.elapsed().as_secs_f64()
        );
    }
    writer.flush()?;
    println!("csv={}", csv_path.display());
    println!("total runtime: {:.3}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn field_summary(
    scenario: &PlanarHolesScenarioResult,
    model: PlanarHolesModelKind,
    subset: PlanarHolesFieldCoverageSubset,
) -> Option<&PlanarHolesFieldCoverageSummary> {
    scenario
        .field_coverage_summaries
        .iter()
        .find(|summary| summary.model == model && summary.subset == subset)
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

fn scale_label(scale: f64) -> String {
    format!("{scale:.2}").replace('.', "p")
}
