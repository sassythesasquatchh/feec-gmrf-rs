#![cfg_attr(not(feature = "heavy-tests"), allow(dead_code, unused_imports))]

use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use feg_case_studies::cube_mass_inverse_variance::{
    compute_matern_1form_cube_mass_inverse_variance_report, symmetric_extreme_eigenvalues,
    write_matern_1form_cube_mass_inverse_variance_outputs, CubeEdgeSubset, CubeMassInverseKind,
    CubeMassInverseVarianceConfig,
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "heavy-tests")]
#[test]
fn cube_mass_inverse_variance_report_builds_all_strategies() {
    let report =
        compute_matern_1form_cube_mass_inverse_variance_report(CubeMassInverseVarianceConfig {
            levels: vec![2],
            max_consistent_dofs: 200,
            barycentric_stabilization_factors: vec![0.5, 1.0, 2.0],
            ..CubeMassInverseVarianceConfig::default()
        })
        .expect("cube mass-inverse variance report should build");

    assert_eq!(report.levels.len(), 1);
    let level = &report.levels[0];
    assert_eq!(level.level, 2);
    assert_eq!(level.edge_count, level.edge_geometry.len());
    assert!(level.edge_geometry.iter().any(|edge| edge.is_boundary));
    assert!(level.edge_geometry.iter().any(|edge| !edge.is_boundary));
    assert_eq!(level.strategies.len(), 8);
    let interior_edge_count = level
        .edge_geometry
        .iter()
        .filter(|edge| !edge.is_boundary)
        .count();
    assert!(interior_edge_count > 0);

    for kind in CubeMassInverseKind::all() {
        let strategies = level
            .strategies
            .iter()
            .filter(|strategy| strategy.kind == kind)
            .collect::<Vec<_>>();
        assert!(
            !strategies.is_empty(),
            "missing strategy family {}",
            kind.label()
        );
        let strategy = strategies[0];
        assert_eq!(strategy.variances.len(), level.edge_count);
        assert!(strategy.mass_inverse_nnz > 0);
        assert!(strategy.mass_inverse_density > 0.0);
        assert!(strategy.mass_inverse_eigen.lambda_min > 0.0);
        assert!(strategy.mass_inverse_eigen.lambda_max >= strategy.mass_inverse_eigen.lambda_min);
        assert!(strategy.mass_inverse_eigen.condition_number >= 1.0);
        assert!(strategy.consistency_error.is_finite());
        assert!(strategy.consistency_error >= 0.0);
        assert!(strategy.precision_nnz > 0);
        assert!(strategy.precision_density > 0.0);
        assert!(strategy.precision_lower_nnz > 0);
        assert!(strategy.factor_nnz > 0);
        assert!(strategy.factor_density > 0.0);
        assert!(strategy.fill_in_ratio > 0.0);
        assert!(strategy.variance_stats.min > 0.0);
        assert!(strategy.variance_stats.mean.is_finite());
        assert!(strategy.variance_stats.median.is_finite());
        assert!(strategy.variance_stats.max >= strategy.variance_stats.min);
        assert!(strategy
            .variances
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
        assert_eq!(strategy.subset_summaries.len(), 1);
        assert_eq!(
            strategy.subset_summaries[0].subset,
            CubeEdgeSubset::Interior
        );
        assert_eq!(strategy.subset_summaries[0].edge_count, interior_edge_count);
    }
    assert_eq!(
        level
            .strategies
            .iter()
            .filter(|strategy| {
                strategy.kind == CubeMassInverseKind::BarycentricDualSparseInverse
                    && !strategy.is_oracle_calibrated
            })
            .count(),
        3
    );
    let row_sum = level
        .strategies
        .iter()
        .find(|strategy| strategy.kind == CubeMassInverseKind::RowSumLumped)
        .expect("row-sum strategy should be present");
    let mass_diagonal = level
        .strategies
        .iter()
        .find(|strategy| strategy.kind == CubeMassInverseKind::MassDiagonalInverse)
        .expect("mass-diagonal strategy should be present");
    assert_eq!(mass_diagonal.label, "mass_diagonal_inverse");
    assert_eq!(mass_diagonal.mass_inverse_nnz, level.edge_count);
    assert!(
        max_abs_vector_diff(&row_sum.variances, &mass_diagonal.variances) > 1e-10,
        "mass-diagonal candidate should differ from row-sum on the cube mesh"
    );
    let oracle = level
        .strategies
        .iter()
        .find(|strategy| strategy.is_oracle_calibrated)
        .expect("oracle-calibrated barycentric row should be present");
    assert_eq!(
        oracle.kind,
        CubeMassInverseKind::BarycentricDualSparseInverse
    );
    assert!(oracle.barycentric_stabilization_factor.is_some());

    let exact = level
        .strategies
        .iter()
        .find(|strategy| strategy.kind == CubeMassInverseKind::ExactConsistentInverse)
        .expect("exact strategy should be present");
    assert_eq!(
        exact.comparison_to_consistent.rms_log_delta_vs_consistent,
        0.0
    );
    assert_eq!(
        exact
            .comparison_to_consistent
            .max_abs_relative_delta_vs_consistent,
        0.0
    );
    assert!(exact.consistency_error <= 1e-8);
}

#[cfg(feature = "heavy-tests")]
#[test]
fn cube_mass_inverse_variance_writes_summary_and_edge_csvs() {
    let report =
        compute_matern_1form_cube_mass_inverse_variance_report(CubeMassInverseVarianceConfig {
            levels: vec![2],
            max_consistent_dofs: 200,
            barycentric_stabilization_factors: vec![1.0],
            include_barycentric_oracle: false,
            ..CubeMassInverseVarianceConfig::default()
        })
        .expect("cube mass-inverse variance report should build");
    let out_dir = temp_output_dir();
    let _ = fs::remove_dir_all(&out_dir);

    write_matern_1form_cube_mass_inverse_variance_outputs(&report, &out_dir)
        .expect("CSV outputs should write");

    let summary = fs::read_to_string(out_dir.join("summary.csv")).expect("summary should exist");
    assert!(summary.contains("interior_rms_log_delta_vs_consistent"));
    assert!(summary.contains("mass_inverse_lambda_min"));
    assert!(summary.contains("consistency_error"));
    assert!(summary.contains("fill_in_ratio"));
    let expected_summary_rows = 1 + report.levels[0].strategies.len();
    assert_eq!(summary.lines().count(), expected_summary_rows);

    let fit_summary =
        fs::read_to_string(out_dir.join("fit_summary.csv")).expect("fit summary should exist");
    assert!(fit_summary.contains("consistency_error"));
    assert!(fit_summary.contains("interior_rms_log_delta_vs_consistent"));

    let edge_csv = fs::read_to_string(out_dir.join("interior_edge_variances_level_2.csv"))
        .expect("edge CSV should exist");
    assert!(edge_csv.contains("is_boundary_edge"));
    assert!(edge_csv.contains("row_sum_lumped_variance"));
    assert!(edge_csv.contains("mass_diagonal_inverse_variance"));
    assert!(edge_csv.contains("exact_consistent_inverse_relative_delta_vs_consistent"));
    let interior_edge_count = report.levels[0]
        .edge_geometry
        .iter()
        .filter(|edge| !edge.is_boundary)
        .count();
    assert_eq!(edge_csv.lines().count(), 1 + interior_edge_count);
    assert!(edge_csv
        .lines()
        .skip(1)
        .all(|line| line.ends_with(",false") || line.contains(",false,")));

    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
fn cube_mass_inverse_variance_rejects_consistent_dof_limit() {
    let err =
        compute_matern_1form_cube_mass_inverse_variance_report(CubeMassInverseVarianceConfig {
            levels: vec![2],
            max_consistent_dofs: 10,
            ..CubeMassInverseVarianceConfig::default()
        })
        .expect_err("small consistent dof limit should reject level 2");

    assert!(err.contains("max_consistent_dofs"));
}

#[test]
fn symmetric_extreme_eigenvalues_reports_known_diagonal_matrix() {
    let mut coo = FeecCoo::new(3, 3);
    coo.push(0, 0, 2.0);
    coo.push(1, 1, 5.0);
    coo.push(2, 2, 10.0);
    let matrix = FeecCsr::from(&coo);

    let stats = symmetric_extreme_eigenvalues(&matrix).expect("diagonal SPD matrix should work");

    assert!((stats.lambda_min - 2.0).abs() <= 1e-12);
    assert!((stats.lambda_max - 10.0).abs() <= 1e-12);
    assert!((stats.condition_number - 5.0).abs() <= 1e-12);
}

fn max_abs_vector_diff(lhs: &FeecVector, rhs: &FeecVector) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn temp_output_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cube_mass_inverse_variance_{}_{}",
        std::process::id(),
        stamp
    ))
}
