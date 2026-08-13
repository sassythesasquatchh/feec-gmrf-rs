#![cfg(feature = "heavy-tests")]

use feg_case_studies::torus::posterior_residual_weight::{
    default_torus_shell_mesh_path, run_torus_1form_pde_posterior_mean_weight_sweep,
    write_torus_1form_pde_posterior_mean_weight_sweep_outputs, Torus1FormPdeMeshLevel,
    Torus1FormPdePosteriorMeanWeightRow, Torus1FormPdePosteriorMeanWeightSweepConfig,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const EXPECTED_DETAIL_HEADER: &str = "resolution,weight,noise_variance,edge_dofs,h,kappa,tau,posterior_deterministic_l2_error,posterior_deterministic_relative_l2_error,posterior_continuum_l2_error,posterior_continuum_relative_l2_error,deterministic_continuum_l2_error,deterministic_continuum_relative_l2_error,posterior_residual_norm,posterior_relative_residual_norm,wall_seconds";
const EXPECTED_SUMMARY_HEADER: &str = "resolution,edge_dofs,h,kappa,tau,min_weight,max_weight,posterior_deterministic_relative_l2_slope,posterior_relative_residual_slope,high_weight_posterior_deterministic_l2_error,high_weight_posterior_deterministic_relative_l2_error,high_weight_posterior_continuum_l2_error,high_weight_posterior_continuum_relative_l2_error,deterministic_continuum_l2_error,deterministic_continuum_relative_l2_error,high_weight_posterior_residual_norm,high_weight_posterior_relative_residual_norm,total_wall_seconds";

fn test_config() -> Torus1FormPdePosteriorMeanWeightSweepConfig {
    let mut config = Torus1FormPdePosteriorMeanWeightSweepConfig::default();
    config.mesh_levels = (0..=1)
        .map(|resolution| Torus1FormPdeMeshLevel {
            resolution,
            mesh_path: default_torus_shell_mesh_path(resolution),
        })
        .collect();
    config.weights = vec![1e2, 1e8];
    config
}

fn unique_output_dir() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "torus_1form_pde_posterior_mean_convergence_{}_{}",
        std::process::id(),
        unique
    ))
}

fn assert_row_is_finite(row: &Torus1FormPdePosteriorMeanWeightRow) {
    assert!(row.weight.is_finite());
    assert!(row.noise_variance.is_finite());
    assert!(row.h.is_finite());
    assert!(row.kappa.is_finite());
    assert!(row.tau.is_finite());
    assert!(row.posterior_deterministic_l2_error.is_finite());
    assert!(row.posterior_deterministic_relative_l2_error.is_finite());
    assert!(row.posterior_continuum_l2_error.is_finite());
    assert!(row.posterior_continuum_relative_l2_error.is_finite());
    assert!(row.deterministic_continuum_l2_error.is_finite());
    assert!(row.deterministic_continuum_relative_l2_error.is_finite());
    assert!(row.posterior_residual_norm.is_finite());
    assert!(row.posterior_relative_residual_norm.is_finite());
    assert!(row.wall_seconds.is_finite());
}

fn nonincreasing(next: f64, previous: f64) -> bool {
    next <= previous * (1.0 + 1e-8) + 1e-14
}

#[test]
fn torus_1form_pde_posterior_mean_convergence_weight_sweep_reduces_error_by_level() {
    let config = test_config();
    let result = run_torus_1form_pde_posterior_mean_weight_sweep(&config)
        .expect("posterior mean weight sweep should succeed");

    assert_eq!(
        result.rows.len(),
        config.mesh_levels.len() * config.weights.len()
    );
    assert_eq!(result.summaries.len(), config.mesh_levels.len());

    for (level_index, mesh_level) in config.mesh_levels.iter().enumerate() {
        let start = level_index * config.weights.len();
        let rows = &result.rows[start..start + config.weights.len()];
        assert!(rows
            .iter()
            .all(|row| row.resolution == mesh_level.resolution));

        for (row, expected_weight) in rows.iter().zip(config.weights.iter()) {
            assert_eq!(row.weight, *expected_weight);
            assert_eq!(row.noise_variance, 1.0 / expected_weight);
            assert_eq!(row.kappa, config.kappa);
            assert_eq!(row.tau, config.tau);
            assert!(row.edge_dofs > 0);
            assert_row_is_finite(row);
        }

        let first = rows.first().expect("at least one row");
        let last = rows.last().expect("at least one row");
        assert!(
            last.posterior_deterministic_l2_error < first.posterior_deterministic_l2_error,
            "high-weight posterior mean should be closer to the deterministic reference"
        );
        assert!(
            last.posterior_residual_norm < first.posterior_residual_norm,
            "high-weight posterior mean should fit the PDE residual more closely"
        );
        assert!(
            (last.posterior_continuum_l2_error - last.deterministic_continuum_l2_error).abs()
                < 1e-3,
            "high-weight continuum error should track deterministic reference error"
        );
    }

    let level0_high = result.rows[config.weights.len() - 1].posterior_continuum_l2_error;
    let level1_high = result.rows[2 * config.weights.len() - 1].posterior_continuum_l2_error;
    assert!(
        level1_high < level0_high,
        "high-weight posterior continuum error should decrease under mesh refinement"
    );

    for summary in &result.summaries {
        assert!(summary
            .posterior_deterministic_relative_l2_slope
            .is_finite());
        assert!(summary.posterior_relative_residual_slope.is_finite());
        assert!(summary.posterior_deterministic_relative_l2_slope < 0.0);
        assert!(summary.posterior_relative_residual_slope < 0.0);
    }
}

#[test]
fn torus_1form_pde_posterior_mean_convergence_weight_sweep_sorts_weights() {
    let mut config = test_config();
    config.mesh_levels = vec![Torus1FormPdeMeshLevel {
        resolution: 0,
        mesh_path: default_torus_shell_mesh_path(0),
    }];
    config.weights = vec![1e8, 1e2];
    let result = run_torus_1form_pde_posterior_mean_weight_sweep(&config)
        .expect("posterior mean weight sweep should succeed");

    assert_eq!(result.rows.len(), 2);
    assert_eq!(result.rows[0].weight, 1e2);
    assert_eq!(result.rows[1].weight, 1e8);
    assert!(nonincreasing(
        result.rows[1].posterior_deterministic_l2_error,
        result.rows[0].posterior_deterministic_l2_error
    ));
}

#[test]
fn torus_1form_pde_posterior_mean_convergence_weight_sweep_writes_csvs() {
    let config = test_config();
    let result = run_torus_1form_pde_posterior_mean_weight_sweep(&config)
        .expect("posterior mean weight sweep should succeed");
    let out_dir = unique_output_dir();
    let _ = fs::remove_dir_all(&out_dir);

    write_torus_1form_pde_posterior_mean_weight_sweep_outputs(&result, &out_dir)
        .expect("posterior mean weight sweep output should write");

    let detail_csv_path = out_dir.join("posterior_mean_weight_sweep.csv");
    assert!(
        detail_csv_path.is_file(),
        "missing CSV output: {detail_csv_path:?}"
    );
    let detail_csv = fs::read_to_string(detail_csv_path).expect("CSV should be readable");
    let mut detail_lines = detail_csv.lines();
    assert_eq!(detail_lines.next(), Some(EXPECTED_DETAIL_HEADER));
    assert_eq!(detail_lines.count(), result.rows.len());

    let summary_csv_path = out_dir.join("posterior_mean_weight_sweep_summary.csv");
    assert!(
        summary_csv_path.is_file(),
        "missing summary CSV output: {summary_csv_path:?}"
    );
    let summary_csv = fs::read_to_string(summary_csv_path).expect("summary CSV should be readable");
    let mut summary_lines = summary_csv.lines();
    assert_eq!(summary_lines.next(), Some(EXPECTED_SUMMARY_HEADER));
    assert_eq!(summary_lines.count(), result.summaries.len());

    let _ = fs::remove_dir_all(&out_dir);
}
