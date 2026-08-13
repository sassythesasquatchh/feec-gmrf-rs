#![cfg(feature = "experimental")]
#![cfg(feature = "heavy-tests")]

use feg_case_studies::weighted_hodge_matern_isolation::{
    compute_weighted_hodge_matern_isolation_report, write_weighted_hodge_matern_isolation_outputs,
    VarianceNormalization, WeightedHodgeMaternIsolationConfig, WeightedMassInverseKind,
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn weighted_hodge_matern_isolation_level2_builds_finite_distributions() {
    let report =
        compute_weighted_hodge_matern_isolation_report(WeightedHodgeMaternIsolationConfig {
            level: 2,
            contrasts: vec![1.0, 100.0],
            kappa_factors: vec![1.0, 10.0],
            max_exact_dofs: 500,
            ..WeightedHodgeMaternIsolationConfig::default()
        })
        .expect("2D weighted Hodge Matern isolation report should build");

    assert!(report.active_edge_count >= 3);
    assert!(report.unit_weight_diagnostics.operator_max_abs_difference <= 1e-10);
    assert!(report.unit_weight_diagnostics.mass_max_abs_difference <= 1e-10);
    assert!(
        report
            .unit_weight_diagnostics
            .projected_inverse_max_abs_difference
            <= 1e-10
    );

    for scenario in &report.scenarios {
        assert_eq!(scenario.strategies.len(), 8);
        assert_eq!(scenario.probes.len(), 3);
        for strategy in &scenario.strategies {
            assert!(
                strategy.kind == WeightedMassInverseKind::ProjectedWeightedSparseInverse
                    || strategy.kind == WeightedMassInverseKind::ExactConsistentWeightedMass
                    || strategy.kind == WeightedMassInverseKind::SplitUnweightedGraph
                    || strategy.kind
                        == WeightedMassInverseKind::SplitUnweightedProjectedSparseInverse
            );
            assert!(strategy.precision_eigen.lambda_min > 0.0);
            assert!(strategy.precision_eigen.lambda_max >= strategy.precision_eigen.lambda_min);
            assert!(strategy.precision_eigen.condition_number.is_finite());
            assert!(strategy.factor_nnz > 0);
            assert!(strategy.posterior_factor_nnz > 0);
            assert!(strategy.prior_variance_stats.min > 0.0);
            assert!(strategy.posterior_variance_stats.min > 0.0);
            assert!(strategy
                .edge_results
                .iter()
                .all(|edge| edge.prior_variance.is_finite()
                    && edge.posterior_variance.is_finite()
                    && edge.prior_variance > 0.0
                    && edge.posterior_variance >= 0.0
                    && edge.posterior_variance <= edge.prior_variance * (1.0 + 1e-8)));
            assert!(
                strategy.mean_probe_variance_ratio < strategy.median_unobserved_variance_ratio,
                "directly observed probes should shrink more than the median unobserved edge"
            );
        }
    }
}

#[test]
fn weighted_hodge_matern_trace_matching_anchors_to_unit_exact_baseline() {
    let report =
        compute_weighted_hodge_matern_isolation_report(WeightedHodgeMaternIsolationConfig {
            level: 2,
            contrasts: vec![1.0],
            kappa_factors: vec![10.0],
            max_exact_dofs: 500,
            ..WeightedHodgeMaternIsolationConfig::default()
        })
        .expect("2D weighted Hodge Matern isolation report should build");

    let scenario = report
        .scenarios
        .iter()
        .find(|scenario| scenario.contrast == 1.0 && scenario.kappa_factor == 10.0)
        .expect("unit-contrast scenario should exist");
    let exact_trace = scenario
        .strategies
        .iter()
        .find(|strategy| {
            strategy.kind == WeightedMassInverseKind::ExactConsistentWeightedMass
                && strategy.normalization == VarianceNormalization::TraceMatchedBaseline
        })
        .expect("exact trace-matched strategy should exist");
    assert!(
        (exact_trace.prior_variance_stats.mean - scenario.baseline_mean_prior_variance).abs()
            <= 1e-10 * scenario.baseline_mean_prior_variance.max(1.0)
    );
    assert!((exact_trace.tau - 1.0).abs() <= 1e-10);
}

#[test]
fn weighted_hodge_matern_isolation_writes_expected_csvs() {
    let report =
        compute_weighted_hodge_matern_isolation_report(WeightedHodgeMaternIsolationConfig {
            level: 2,
            contrasts: vec![1.0],
            kappa_factors: vec![1.0],
            max_exact_dofs: 500,
            ..WeightedHodgeMaternIsolationConfig::default()
        })
        .expect("2D weighted Hodge Matern isolation report should build");
    let out_dir = temp_output_dir();
    let _ = fs::remove_dir_all(&out_dir);

    write_weighted_hodge_matern_isolation_outputs(&report, &out_dir)
        .expect("CSV outputs should write");

    let summary = fs::read_to_string(out_dir.join("summary.csv")).expect("summary should exist");
    assert!(summary.contains("precision_condition_number"));
    assert!(summary.contains("projected_weighted_sparse_inverse"));
    assert!(summary.contains("exact_consistent_weighted_mass"));
    assert!(summary.contains("split_unweighted_graph"));
    assert!(summary.contains("split_unweighted_projected_sparse_inverse"));
    assert!(!summary.contains("row_sum"));

    let edge_csv = fs::read_to_string(out_dir.join("edge_variances_level_2.csv"))
        .expect("edge CSV should exist");
    assert!(edge_csv.contains("prior_log_delta_vs_exact"));

    let probe_csv =
        fs::read_to_string(out_dir.join("probe_posterior.csv")).expect("probe CSV should exist");
    assert!(probe_csv.contains("weighted_side_center"));
    assert!(probe_csv.contains("interface"));

    let _ = fs::remove_dir_all(out_dir);
}

fn temp_output_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after UNIX epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("weighted_hodge_matern_isolation_{nanos}"))
}
