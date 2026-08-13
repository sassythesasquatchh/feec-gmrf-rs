#![cfg_attr(
    not(any(feature = "heavy-tests", feature = "external-reference-tests")),
    allow(dead_code, unused_imports)
)]

use feg_case_studies::cube_zero_form_kernel_validation::{
    compute_cube_zero_form_kernel_validation_report, default_sensor_points,
    deterministic_sensor_indices, fixed_probe_indices_for_mesh, fixed_probe_points,
    interior_vertex_indices, petsc_solver_available,
    write_cube_zero_form_kernel_validation_outputs, CubeZeroFormKernelValidationConfig,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "heavy-tests")]
#[test]
fn cube_zero_form_validation_smoke_writes_outputs_without_spectral() {
    let report =
        compute_cube_zero_form_kernel_validation_report(CubeZeroFormKernelValidationConfig {
            levels: vec![12, 24],
            include_spectral: false,
            spectral_sweep_ks: vec![4],
            max_correlation_pairs: 128,
            correlation_bin_count: 8,
            ..CubeZeroFormKernelValidationConfig::default()
        })
        .expect("cube zero-form validation should run");

    assert_eq!(report.levels.len(), 2);
    let level = &report.levels[0];
    assert_eq!(level.level, 12);
    assert_eq!(level.eval_indices.len(), 343);
    assert_eq!(level.observation_indices.len(), 27);
    assert_eq!(level.reference_variances.len(), level.eval_indices.len());
    assert_eq!(level.gmrf_variances.len(), level.eval_indices.len());
    assert_eq!(
        level.gmrf_calibrated_variances.len(),
        level.eval_indices.len()
    );
    assert!(level.diagnostics.h_over_range.is_finite());
    assert!(level
        .diagnostics
        .gmrf_tau_calibration_multiplier
        .is_finite());
    assert!(level.diagnostics.gmrf_tau_calibration_multiplier > 0.0);
    assert!(level.spectral_variances.is_none());
    assert!(!level.correlation_pairs.is_empty());
    assert!(!level.correlation_bins.is_empty());
    assert!(level
        .reference_variances
        .iter()
        .chain(level.gmrf_variances.iter())
        .chain(level.gmrf_calibrated_variances.iter())
        .all(|value| value.is_finite() && *value >= 0.0));
    assert!(level
        .metrics
        .iter()
        .any(|metric| metric.method == "gmrf" && metric.variance_rmse.is_finite()));
    assert!(level
        .metrics
        .iter()
        .any(|metric| { metric.method == "gmrf_calibrated" && metric.variance_rmse.is_finite() }));

    let out_dir = temp_output_dir();
    let _ = fs::remove_dir_all(&out_dir);
    write_cube_zero_form_kernel_validation_outputs(&report, &out_dir)
        .expect("validation outputs should write");

    assert!(out_dir.join("hyperparameters.csv").exists());
    assert!(out_dir.join("calibration_diagnostics.csv").exists());
    assert!(out_dir.join("variance_rmse.csv").exists());
    assert!(out_dir.join("interior_variances_level_12.csv").exists());
    assert!(out_dir.join("correlation_pairs_level_12.csv").exists());
    assert!(out_dir.join("correlation_bins_level_12.csv").exists());
    assert!(out_dir
        .join("figures/correlation_gmrf_level_12.svg")
        .exists());

    let variance_rmse =
        fs::read_to_string(out_dir.join("variance_rmse.csv")).expect("rmse csv should read");
    assert!(variance_rmse.contains("euclidean_reference"));
    assert!(variance_rmse.contains("gmrf"));
    assert!(variance_rmse.contains("gmrf_calibrated"));

    let _ = fs::remove_dir_all(out_dir);
}

#[cfg(feature = "external-reference-tests")]
#[test]
fn cube_zero_form_spectral_smoke_skips_without_petsc() {
    if !petsc_solver_available() {
        eprintln!("skipping spectral smoke test: PETSc eigen solver binary unavailable");
        return;
    }

    let report =
        compute_cube_zero_form_kernel_validation_report(CubeZeroFormKernelValidationConfig {
            levels: vec![12],
            spectral_k: 16,
            spectral_sweep_level: 12,
            spectral_sweep_ks: vec![8, 16],
            max_correlation_pairs: 64,
            correlation_bin_count: 6,
            include_spectral: true,
            require_spectral: true,
            ..CubeZeroFormKernelValidationConfig::default()
        })
        .expect("spectral cube zero-form validation should run when PETSc is available");

    let level = &report.levels[0];
    assert!(level.spectral_variances.is_some());
    assert!(level
        .metrics
        .iter()
        .any(|metric| metric.method == "spectral" && metric.variance_rmse.is_finite()));
    assert_eq!(report.spectral_sweep.len(), 2);
}

#[test]
fn fixed_probe_grid_matches_large_mesh_design() {
    let probes = fixed_probe_points(12, 0.25, &default_sensor_points()).expect("fixed probes");
    assert_eq!(probes.len(), 343);
    for sensor in default_sensor_points() {
        assert!(probes.iter().any(|probe| {
            probe
                .iter()
                .zip(sensor.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1e-12)
        }));
    }

    for level in [12, 24, 36, 48] {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, level, 1.0);
        let (_topology, coords) = mesh.compute_coord_complex();
        let mapped = fixed_probe_indices_for_mesh(&coords, level, &probes)
            .expect("fixed probes should align on default levels");
        assert_eq!(mapped.len(), 343);
        assert_eq!(
            mapped
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            343
        );
    }
}

#[test]
fn non_aligned_probe_grid_fails_clearly() {
    let probes = fixed_probe_points(12, 0.25, &default_sensor_points()).expect("fixed probes");
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 10, 1.0);
    let (_topology, coords) = mesh.compute_coord_complex();
    let err = fixed_probe_indices_for_mesh(&coords, 10, &probes)
        .expect_err("level 10 should not align with level-12 probes");
    assert!(err.contains("not aligned"));
}

#[test]
fn deterministic_cube_interior_and_sensor_indices_match_legacy_selector() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 8, 1.0);
    let (_topology, coords) = mesh.compute_coord_complex();

    let interior = interior_vertex_indices(&coords, 0.15).expect("interior vertices");
    let sensors =
        deterministic_sensor_indices(&coords, &default_sensor_points(), 0.15).expect("sensors");

    assert_eq!(interior.len(), 125);
    assert_eq!(sensors.len(), 27);
    assert!(sensors.iter().all(|idx| interior.contains(idx)));
}

fn temp_output_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "cube_zero_form_kernel_validation_{}_{}",
        std::process::id(),
        stamp
    ))
}
