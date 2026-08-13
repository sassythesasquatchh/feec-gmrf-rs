//! Period-variance diagnostics for the nondecomposed Hodge--Matern 1-form prior.
//!
//! This module keeps the topology/period-observability experiment separate from
//! sparse-anchor Hodge branch validation. The latent variable is the ambient 1-form
//! cochain itself, with the ordinary Hodge--Matern precision.

use crate::{
    genus2_1form_hodge_conditioning::{
        build_cycle_observation_matrix, default_genus2_torus_mesh_path, validate_genus2_topology,
        Genus2TopologySummary,
    },
    genus2_topological_inverse::select_local_observation_edges,
};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use feg_infer::{
    prior::matern::one_form::{
        build_hodge_laplacian_1form, build_matern_precision_1form,
        MaternConfig as Matern1FormConfig, MaternMassInverse as Matern1FormMassInverse,
    },
    sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf, sparse_row_operator_from_feec_csr},
};
use gmrf_core::{observation::apply_gaussian_observations, Gmrf};
use manifold::io::gmsh::gmsh2coord_complex;
use std::{fs, path::PathBuf};

const EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct Genus2NondecomposedPeriodVarianceConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub local_observation_count: usize,
    pub local_noise_std: f64,
    pub cycle_noise_std: f64,
}

impl Default for Genus2NondecomposedPeriodVarianceConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_genus2_torus_mesh_path(),
            kappa: 4.0,
            tau: 1.0,
            local_observation_count: 24,
            local_noise_std: 1e-2,
            cycle_noise_std: 1e-4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genus2PeriodObservationScenario {
    LocalOnly,
    LocalPlusOneCycle,
    LocalPlusTwoCycles,
    LocalPlusFourCycles,
}

impl Genus2PeriodObservationScenario {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LocalOnly => "local_only",
            Self::LocalPlusOneCycle => "local_plus_1_cycle",
            Self::LocalPlusTwoCycles => "local_plus_2_cycles",
            Self::LocalPlusFourCycles => "local_plus_4_cycles",
        }
    }

    pub fn observed_cycle_count(self) -> usize {
        match self {
            Self::LocalOnly => 0,
            Self::LocalPlusOneCycle => 1,
            Self::LocalPlusTwoCycles => 2,
            Self::LocalPlusFourCycles => 4,
        }
    }

    fn all() -> [Self; 4] {
        [
            Self::LocalOnly,
            Self::LocalPlusOneCycle,
            Self::LocalPlusTwoCycles,
            Self::LocalPlusFourCycles,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Genus2PeriodVarianceRow {
    pub scenario: Genus2PeriodObservationScenario,
    pub local_observation_count: usize,
    pub cycle_observation_count: usize,
    pub cycle_index: usize,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct Genus2NondecomposedPeriodVarianceReport {
    pub topology_summary: Genus2TopologySummary,
    pub kappa: f64,
    pub tau: f64,
    pub local_noise_std: f64,
    pub cycle_noise_std: f64,
    pub rows: Vec<Genus2PeriodVarianceRow>,
}

impl Genus2NondecomposedPeriodVarianceReport {
    pub fn mean_ratio(&self, scenario: Genus2PeriodObservationScenario) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for row in self.rows.iter().filter(|row| row.scenario == scenario) {
            sum += row.variance_ratio;
            count += 1;
        }
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    }
}

#[derive(Clone)]
struct ObservationRow {
    entries: Vec<(usize, f64)>,
    noise_std: f64,
}

pub fn compute_genus2_nondecomposed_period_variance(
    config: Genus2NondecomposedPeriodVarianceConfig,
) -> Result<Genus2NondecomposedPeriodVarianceReport, String> {
    validate_config(&config)?;

    let mesh_bytes = fs::read(&config.mesh_path)
        .map_err(|err| format!("failed to read {}: {err}", config.mesh_path.display()))?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = validate_genus2_topology(&topology).map_err(|err| err.to_string())?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        Matern1FormConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    );
    let (cycle_selector, _) =
        build_cycle_observation_matrix(&topology, &coords).map_err(|err| err.to_string())?;
    let cycle_operator = sparse_row_operator_from_feec_csr(&cycle_selector)?;
    let mut prior = Gmrf::from_mean_and_precision(
        gmrf_core::types::Vector::zeros(precision.nrows()),
        feec_csr_to_gmrf(&precision),
    )
    .map_err(|err| err.to_string())?;
    let prior_variance = prior
        .exact_transformed_variance_decomposition(
            &cycle_operator,
            &gmrf_core::types::DenseMatrix::zeros(0, precision.nrows()),
        )
        .map_err(|err| err.to_string())?
        .constrained_diag;

    let local_edges = select_local_observation_edges(
        &topology,
        &coords,
        &cycle_selector,
        config.local_observation_count,
    );
    let local_rows = local_edges
        .iter()
        .copied()
        .map(|edge_index| ObservationRow {
            entries: vec![(edge_index, 1.0)],
            noise_std: config.local_noise_std,
        })
        .collect::<Vec<_>>();
    let cycle_rows = sparse_rows(&cycle_selector)
        .into_iter()
        .map(|entries| ObservationRow {
            entries,
            noise_std: config.cycle_noise_std,
        })
        .collect::<Vec<_>>();

    let mut rows = Vec::new();
    for scenario in Genus2PeriodObservationScenario::all() {
        let observation_rows = scenario_rows(scenario, &local_rows, &cycle_rows);
        let observation_matrix = scaled_observation_matrix(&observation_rows, precision.nrows())?;
        let observations = FeecVector::zeros(observation_matrix.nrows());
        let (posterior_precision, information) = apply_gaussian_observations(
            &feec_csr_to_gmrf(&precision),
            &feec_csr_to_gmrf(&observation_matrix),
            &feec_vec_to_gmrf(&observations),
            None,
            1.0,
        );
        let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)
            .map_err(|err| err.to_string())?;
        let posterior_variance = posterior
            .exact_transformed_variance_decomposition(
                &cycle_operator,
                &gmrf_core::types::DenseMatrix::zeros(0, precision.nrows()),
            )
            .map_err(|err| err.to_string())?
            .constrained_diag;
        for cycle_index in 0..cycle_selector.nrows() {
            let ratio = posterior_variance[cycle_index] / prior_variance[cycle_index].max(EPS);
            rows.push(Genus2PeriodVarianceRow {
                scenario,
                local_observation_count: local_rows.len(),
                cycle_observation_count: scenario.observed_cycle_count(),
                cycle_index,
                prior_variance: prior_variance[cycle_index],
                posterior_variance: posterior_variance[cycle_index],
                variance_ratio: ratio,
            });
        }
    }

    Ok(Genus2NondecomposedPeriodVarianceReport {
        topology_summary,
        kappa: config.kappa,
        tau: config.tau,
        local_noise_std: config.local_noise_std,
        cycle_noise_std: config.cycle_noise_std,
        rows,
    })
}

fn scenario_rows(
    scenario: Genus2PeriodObservationScenario,
    local_rows: &[ObservationRow],
    cycle_rows: &[ObservationRow],
) -> Vec<ObservationRow> {
    let mut rows = local_rows.to_vec();
    rows.extend(
        cycle_rows
            .iter()
            .take(scenario.observed_cycle_count())
            .cloned(),
    );
    rows
}

fn sparse_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if value.abs() > EPS {
            rows[row].push((col, *value));
        }
    }
    rows
}

fn scaled_observation_matrix(rows: &[ObservationRow], state_dim: usize) -> Result<FeecCsr, String> {
    let mut coo = FeecCoo::new(rows.len(), state_dim);
    for (row_index, row) in rows.iter().enumerate() {
        if !row.noise_std.is_finite() || row.noise_std <= 0.0 {
            return Err(
                "observation noise standard deviation must be finite and positive".to_string(),
            );
        }
        for (col, value) in &row.entries {
            coo.push(row_index, *col, *value / row.noise_std);
        }
    }
    Ok(FeecCsr::from(&coo))
}

fn validate_config(config: &Genus2NondecomposedPeriodVarianceConfig) -> Result<(), String> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if config.local_observation_count == 0 {
        return Err("local observation count must be positive".to_string());
    }
    if !config.local_noise_std.is_finite() || config.local_noise_std <= 0.0 {
        return Err("local noise standard deviation must be finite and positive".to_string());
    }
    if !config.cycle_noise_std.is_finite() || config.cycle_noise_std <= 0.0 {
        return Err("cycle noise standard deviation must be finite and positive".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::lock_feec_harmonic_tests;

    #[test]
    fn genus2_nondecomposed_period_variance_responds_to_cycle_observations() {
        let _lock = lock_feec_harmonic_tests();
        let report = compute_genus2_nondecomposed_period_variance(
            Genus2NondecomposedPeriodVarianceConfig::default(),
        )
        .expect("nondecomposed genus-2 period variance should run");

        assert_eq!(report.topology_summary.b1, 4);
        assert_eq!(report.rows.len(), 16);
        let local_only = report.mean_ratio(Genus2PeriodObservationScenario::LocalOnly);
        let all_cycles = report.mean_ratio(Genus2PeriodObservationScenario::LocalPlusFourCycles);
        eprintln!("local_only_mean_ratio={local_only:.6e} all_cycles_mean_ratio={all_cycles:.6e}");
        assert!(local_only.is_finite());
        assert!(all_cycles.is_finite());
        assert!(all_cycles < local_only);
        assert!(all_cycles < 1e-5);
    }
}
