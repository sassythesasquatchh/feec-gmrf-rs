use common::linalg::nalgebra::{bilinear_form_sparse, Vector as FeecVector};
use feg_case_studies::planar_holes_hodge_flow::{
    run_planar_holes_hodge_flow, PlanarHolesBranchRecoverySummary, PlanarHolesExactTruthSource,
    PlanarHolesFieldCoverageSubset, PlanarHolesFieldCoverageSummary, PlanarHolesFlowConfig,
    PlanarHolesLocalObservationDesign, PlanarHolesModelKind, PlanarHolesScenarioResult,
    PlanarHolesTruthScaling,
};
use feg_core::HodgeBranchKind;
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const NEAR_NOISELESS_VARIANCE: f64 = 1e-8;

#[derive(Debug, Clone, Copy)]
struct OracleCase {
    name: &'static str,
    truth_source: PlanarHolesExactTruthSource,
    observation_design: PlanarHolesLocalObservationDesign,
    sample_observation_noise: bool,
    local_noise_variance: f64,
    loop_noise_variance: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("../../out/planar_holes_exact_oracle");
    fs::create_dir_all(&output_dir)?;
    let mesh_path = output_dir.join("planar_holes_exact_oracle.msh");
    let geo_path = output_dir.join("planar_holes_exact_oracle.geo");
    let csv_path = output_dir.join("exact_oracle_summary.csv");
    let mode_count = env::var("PLANAR_HOLES_EXACT_ORACLE_MODES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    let mesh_size = env::var("PLANAR_HOLES_EXACT_ORACLE_MESH_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.045);
    let local_observation_count = env::var("PLANAR_HOLES_EXACT_ORACLE_LOCAL_OBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200);
    let heldout_local_count = env::var("PLANAR_HOLES_EXACT_ORACLE_HELDOUT_OBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    let cases = [
        OracleCase {
            name: "matched_sparse_noisy",
            truth_source: PlanarHolesExactTruthSource::SparseAnchor,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: true,
            local_noise_variance: 1e-4,
            loop_noise_variance: 1e-5,
        },
        OracleCase {
            name: "matched_sparse_noiseless",
            truth_source: PlanarHolesExactTruthSource::SparseAnchor,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: false,
            local_noise_variance: NEAR_NOISELESS_VARIANCE,
            loop_noise_variance: NEAR_NOISELESS_VARIANCE,
        },
        OracleCase {
            name: "matched_dense_noiseless",
            truth_source: PlanarHolesExactTruthSource::SparseAnchor,
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: NEAR_NOISELESS_VARIANCE,
            loop_noise_variance: NEAR_NOISELESS_VARIANCE,
        },
        OracleCase {
            name: "exact_dense_sparse_noiseless",
            truth_source: PlanarHolesExactTruthSource::ExactDenseGmrf,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: false,
            local_noise_variance: NEAR_NOISELESS_VARIANCE,
            loop_noise_variance: NEAR_NOISELESS_VARIANCE,
        },
        OracleCase {
            name: "exact_dense_dense_noiseless",
            truth_source: PlanarHolesExactTruthSource::ExactDenseGmrf,
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: NEAR_NOISELESS_VARIANCE,
            loop_noise_variance: NEAR_NOISELESS_VARIANCE,
        },
        OracleCase {
            name: "spectral_sparse_noiseless",
            truth_source: PlanarHolesExactTruthSource::SpectralGp,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: false,
            local_noise_variance: NEAR_NOISELESS_VARIANCE,
            loop_noise_variance: NEAR_NOISELESS_VARIANCE,
        },
        OracleCase {
            name: "spectral_dense_noiseless",
            truth_source: PlanarHolesExactTruthSource::SpectralGp,
            observation_design: PlanarHolesLocalObservationDesign::AllEdges,
            sample_observation_noise: false,
            local_noise_variance: NEAR_NOISELESS_VARIANCE,
            loop_noise_variance: NEAR_NOISELESS_VARIANCE,
        },
        OracleCase {
            name: "analytic_sparse_noiseless",
            truth_source: PlanarHolesExactTruthSource::AnalyticPotential,
            observation_design: PlanarHolesLocalObservationDesign::SparseInterior,
            sample_observation_noise: false,
            local_noise_variance: NEAR_NOISELESS_VARIANCE,
            loop_noise_variance: NEAR_NOISELESS_VARIANCE,
        },
    ];

    let mut writer = BufWriter::new(File::create(&csv_path)?);
    writeln!(
        writer,
        "case,truth_source,truth_scaling,observation_design,sample_observation_noise,scenario,model,observation_count,l2_error,heldout_local_nlpd,exterior_derivative_error_absolute,exterior_derivative_leakage,all_edge_coverage_95,exact_truth_norm,exact_posterior_norm,exact_relative_branch_error,exact_mass_correlation"
    )?;

    println!("Planar holes pure-exact oracle experiments");
    println!("output_dir={}", output_dir.display());
    println!(
        "mesh_size={mesh_size} spectral/exact mode_count={mode_count} sparse_local_obs={local_observation_count} heldout_local_obs={heldout_local_count}"
    );
    for (index, case) in cases.iter().enumerate() {
        let case_start = Instant::now();
        let config = PlanarHolesFlowConfig {
            output_dir: output_dir.join(case.name),
            mesh_path: mesh_path.clone(),
            geo_path: geo_path.clone(),
            force_mesh: index == 0,
            mesh_size,
            exact_truth_mass_norm: 1.0,
            coexact_truth_mass_norm: 0.0,
            harmonic_truth_mass_norm: 0.0,
            exact_truth_source: case.truth_source,
            truth_scaling: PlanarHolesTruthScaling::RawPriorSamples,
            local_observation_design: case.observation_design,
            sample_observation_noise: case.sample_observation_noise,
            local_noise_variance: case.local_noise_variance,
            loop_noise_variance: case.loop_noise_variance,
            local_observation_count,
            heldout_local_count,
            include_exact_hodge_model: true,
            include_exact_dense_exact_hodge_model: true,
            include_exact_dense_trace_matched_exact_hodge_model: true,
            include_incompressible_hodge_model: false,
            include_spectral_exact_hodge_model: true,
            spectral_branch_energy_normalization: true,
            spectral_exact_mode_count: mode_count,
            spectral_exact_expected_m1_energy: 1.0,
            ..PlanarHolesFlowConfig::default()
        };
        let result = run_planar_holes_hodge_flow(&config)?;
        let truth_norm = mass_norm_from_result(&result.truth, &result);
        println!(
            "case={} topology: vertices={} edges={} faces={} b1={} truth_norm={:.4e} truth_exact_norm={:.4e}",
            case.name,
            result.topology_summary.vertex_count,
            result.topology_summary.edge_count,
            result.topology_summary.face_count,
            result.topology_summary.b1,
            truth_norm,
            mass_norm_from_result(&result.truth_exact, &result),
        );
        for scenario in &result.scenarios {
            for metric in &scenario.metrics {
                let all_field = field_summary(
                    scenario,
                    metric.model,
                    PlanarHolesFieldCoverageSubset::AllEdges,
                );
                let exact = branch_summary(scenario, metric.model, HodgeBranchKind::Exact);
                let exterior_leakage =
                    metric.exterior_derivative_error_absolute / truth_norm.max(f64::MIN_POSITIVE);
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                    case.name,
                    case.truth_source.as_str(),
                    PlanarHolesTruthScaling::RawPriorSamples.as_str(),
                    case.observation_design.as_str(),
                    case.sample_observation_noise,
                    metric.scenario.as_str(),
                    metric.model.as_str(),
                    metric.observation_count,
                    metric.l2_error,
                    metric.heldout_local_nlpd,
                    metric.exterior_derivative_error_absolute,
                    exterior_leakage,
                    all_field
                        .map(|summary| summary.coverage_95)
                        .unwrap_or(f64::NAN),
                    exact
                        .map(|summary| summary.truth_mass_norm)
                        .unwrap_or(f64::NAN),
                    exact
                        .map(|summary| summary.posterior_mass_norm)
                        .unwrap_or(f64::NAN),
                    exact
                        .map(|summary| summary.relative_error)
                        .unwrap_or(f64::NAN),
                    exact
                        .map(|summary| summary.mass_correlation)
                        .unwrap_or(f64::NAN),
                )?;
                if matches!(
                    metric.model,
                    PlanarHolesModelKind::HodgeMatern
                        | PlanarHolesModelKind::ExactHodgeMatern
                        | PlanarHolesModelKind::ExactDenseTraceMatchedExactHodgeMatern
                        | PlanarHolesModelKind::SpectralExactHodgeGp
                        | PlanarHolesModelKind::NondecomposedFeec
                ) {
                    println!(
                        "case={} scenario={} model={} obs={} E1={:.4e} d_leakage={:.4e} field_cov={:.3} exact_norm={:.4e} exact_rel_err={:.4e} exact_corr={:.4e}",
                        case.name,
                        metric.scenario.as_str(),
                        metric.model.as_str(),
                        metric.observation_count,
                        metric.l2_error,
                        exterior_leakage,
                        all_field
                            .map(|summary| summary.coverage_95)
                            .unwrap_or(f64::NAN),
                        exact
                            .map(|summary| summary.posterior_mass_norm)
                            .unwrap_or(f64::NAN),
                        exact
                            .map(|summary| summary.relative_error)
                            .unwrap_or(f64::NAN),
                        exact
                            .map(|summary| summary.mass_correlation)
                            .unwrap_or(f64::NAN),
                    );
                }
            }
        }
        println!(
            "case={} runtime={:.3}s",
            case.name,
            case_start.elapsed().as_secs_f64()
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

fn mass_norm_from_result(
    field: &FeecVector,
    result: &feg_case_studies::planar_holes_hodge_flow::PlanarHolesFlowResult,
) -> f64 {
    let mass = feg_infer::prior::matern::one_form::build_hodge_laplacian_1form(
        &result.topology,
        &result.metric,
    )
    .mass_u;
    bilinear_form_sparse(&mass, field, field).max(0.0).sqrt()
}
