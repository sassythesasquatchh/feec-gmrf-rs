use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_hodge_flow, write_planar_holes_hodge_flow_outputs,
    PlanarHolesFieldCoverageSubset, PlanarHolesFlowConfig, PlanarHolesModelKind,
    PlanarHolesScenarioResult,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let config = PlanarHolesFlowConfig::default();
    let result = run_planar_holes_hodge_flow(&config)?;
    write_planar_holes_hodge_flow_outputs(&result, &config.output_dir)?;

    println!("Planar holes Hodge-GMRF flow experiment");
    println!("mesh={}", config.mesh_path.display());
    println!("output_dir={}", config.output_dir.display());
    println!(
        "topology: vertices={} edges={} faces={} chi={} b0={} b1={} b2={} train_cycle_rank={} heldout_cycle_rank={}",
        result.topology_summary.vertex_count,
        result.topology_summary.edge_count,
        result.topology_summary.face_count,
        result.topology_summary.euler_characteristic,
        result.topology_summary.b0,
        result.topology_summary.b1,
        result.topology_summary.b2,
        result.cycle_harmonic_pairing_rank,
        result.heldout_cycle_harmonic_pairing_rank
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
            println!(
                "scenario={} model={} rel_l2={:.4e} nlpd={:.4e} loop_nlpd={:.4e} harmonic_period_nlpd={:.4e} rel_total_loop_error={:.4e} rel_harmonic_period_error={:.4e} rel_coexact_annular_error={:.4e} rel_total_annular_error={:.4e} field_cov95={:.3} field_mw_cov95={:.3} heldout_field_cov95={:.3} field_mean_abs_z={:.3e} field_p95_abs_z={:.3e}",
                scenario.scenario.as_str(),
                metric.model.as_str(),
                metric.l2_error,
                metric.heldout_nlpd,
                metric.heldout_loop_nlpd,
                metric.heldout_harmonic_period_nlpd,
                metric.relative_circulation_error,
                metric.relative_harmonic_period_error,
                metric.relative_coexact_annular_error,
                metric.relative_total_annular_error,
                all_field
                    .map(|summary| summary.coverage_95)
                    .unwrap_or(f64::NAN),
                all_field
                    .map(|summary| summary.mass_weighted_coverage_95)
                    .unwrap_or(f64::NAN),
                heldout_field
                    .map(|summary| summary.coverage_95)
                    .unwrap_or(f64::NAN),
                all_field
                    .map(|summary| summary.mean_abs_z)
                    .unwrap_or(f64::NAN),
                all_field
                    .map(|summary| summary.p95_abs_z)
                    .unwrap_or(f64::NAN)
            );
        }
        let hodge_std = mean(scenario.period_summaries.iter().filter_map(|summary| {
            (summary.model == PlanarHolesModelKind::HodgeMatern).then_some(summary.posterior_std)
        }));
        println!(
            "scenario={} hodge_mean_period_std={:.4e}",
            scenario.scenario.as_str(),
            hodge_std
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

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}
