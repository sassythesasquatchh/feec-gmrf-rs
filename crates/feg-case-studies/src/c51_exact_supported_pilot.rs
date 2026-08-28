//! Cheap C51 correctness pilot for the two exact Matérn representations.
//!
//! This experiment compares the eliminated full-metric reference with the
//! sparse supported graph on tiny Cartesian meshes. Timings are retained only
//! to validate like-for-like build, factor, and covariance-action
//! instrumentation for E19. These tiny, single-run timings are not performance
//! claims.

use feec_gmrf::prelude::{
    ExactFormOperators, ExactMaternPriorBuilder, FeecGmrfError, MaternAlpha, MaternParameters,
    Result, SupportedMaternPriorBuilder,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use std::time::{Duration, Instant};

/// Small meshes and direct Matérn parameters used by the pilot.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactSupportedPilotConfig {
    /// Intrinsic Cartesian dimensions to exercise.
    pub spatial_dimensions: Vec<usize>,
    /// Number of Cartesian cells along each coordinate direction.
    pub cells_per_axis: usize,
    pub kappa: f64,
    pub tau: f64,
}

impl ExactSupportedPilotConfig {
    /// Cheap default covering every valid form degree in dimensions 1, 2, and 3.
    pub fn tiny() -> Self {
        Self {
            spatial_dimensions: vec![1, 2, 3],
            cells_per_axis: 2,
            kappa: 1.25,
            tau: 0.8,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.spatial_dimensions.is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "the C51 pilot requires at least one spatial dimension".to_string(),
            ));
        }
        if self
            .spatial_dimensions
            .iter()
            .any(|dimension| !(1..=3).contains(dimension))
        {
            return Err(FeecGmrfError::InvalidParameter(
                "the C51 pilot supports Cartesian dimensions 1, 2, and 3".to_string(),
            ));
        }
        if self.cells_per_axis < 2 {
            return Err(FeecGmrfError::InvalidParameter(
                "homogeneous full-relative pilot meshes require at least two cells per axis"
                    .to_string(),
            ));
        }
        MaternParameters::new(MaternAlpha::One, self.kappa, self.tau)?;
        Ok(())
    }
}

impl Default for ExactSupportedPilotConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

/// One correctness and instrumentation row for a dimension, degree, and alpha.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactSupportedPilotRow {
    pub spatial_dimension: usize,
    pub cells_per_axis: usize,
    pub form_degree: usize,
    pub alpha: MaternAlpha,
    pub state_dimension: usize,
    pub lower_adjacent_dimension: usize,
    pub upper_adjacent_dimension: usize,
    pub adjacent_dimension: usize,
    pub ambient_dimension: usize,
    pub support_dimension: usize,
    pub constraint_dimension: usize,
    pub eliminated_precision_nnz: usize,
    pub supported_energy_nnz: usize,
    pub supported_constraint_nnz: usize,
    /// Nonzeros in the assembled KKT input, not in an LU factor.
    pub supported_kkt_input_nnz: usize,
    pub constraint_gram_certificate_relative_residual: Option<f64>,
    pub supported_energy_certificate_relative_residual: f64,
    pub covariance_action_absolute_error: f64,
    pub covariance_action_relative_error: f64,
    pub feec_operator_build_wall_time: Duration,
    pub eliminated_build_wall_time: Duration,
    pub eliminated_factor_wall_time: Duration,
    /// One covariance action for the shared deterministic probe.
    pub eliminated_query_wall_time: Duration,
    pub supported_build_wall_time: Duration,
    pub supported_factor_wall_time: Duration,
    /// One covariance action for the same deterministic probe.
    pub supported_query_wall_time: Duration,
    /// Both queries are covariance actions, but these are unreplicated tiny-pilot timings.
    pub query_wall_times_are_comparable: bool,
}

/// Run the tiny correctness pilot without registering a maintained study.
pub fn run_exact_supported_pilot(
    config: &ExactSupportedPilotConfig,
) -> Result<Vec<ExactSupportedPilotRow>> {
    config.validate()?;
    let mut rows = Vec::new();

    for &spatial_dimension in &config.spatial_dimensions {
        let mesh =
            CartesianMeshInfo::new_unit_scaled(spatial_dimension, config.cells_per_axis, 1.0);
        let (topology, coordinates) = mesh.compute_coord_complex();
        let metric = coordinates.to_edge_lengths(&topology);

        for form_degree in 0..=spatial_dimension {
            let operator_start = Instant::now();
            let operators = ExactFormOperators::from_feec_homogeneous_full_relative(
                &topology,
                &metric,
                form_degree,
            )?;
            let feec_operator_build_wall_time = operator_start.elapsed();
            let state_dimension = operators.state().dimension();
            let lower_adjacent_dimension = operators
                .incoming()
                .map_or(0, |incoming| incoming.lower().dimension());
            let upper_adjacent_dimension = operators
                .outgoing()
                .map_or(0, |outgoing| outgoing.upper().dimension());
            let case = PilotCase {
                spatial_dimension,
                config,
                form_degree,
                operators: &operators,
                feec_operator_build_wall_time,
                state_dimension,
                lower_adjacent_dimension,
                upper_adjacent_dimension,
            };

            for alpha in [MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three] {
                rows.push(run_row(&case, alpha)?);
            }
        }
    }
    Ok(rows)
}

struct PilotCase<'a> {
    spatial_dimension: usize,
    config: &'a ExactSupportedPilotConfig,
    form_degree: usize,
    feec_operator_build_wall_time: Duration,
    state_dimension: usize,
    lower_adjacent_dimension: usize,
    upper_adjacent_dimension: usize,
    operators: &'a ExactFormOperators,
}

fn run_row(case: &PilotCase<'_>, alpha: MaternAlpha) -> Result<ExactSupportedPilotRow> {
    let parameters = MaternParameters::new(alpha, case.config.kappa, case.config.tau)?;

    let eliminated_build_start = Instant::now();
    let eliminated = ExactMaternPriorBuilder::from_operators(case.operators.clone())
        .parameters(parameters)
        .build()?;
    let eliminated_build_wall_time = eliminated_build_start.elapsed();
    let eliminated_precision_nnz = eliminated.precision().nnz();

    let eliminated_factor_start = Instant::now();
    let mut eliminated_factor = eliminated.factor()?;
    let eliminated_factor_wall_time = eliminated_factor_start.elapsed();

    let supported_build_start = Instant::now();
    let supported = SupportedMaternPriorBuilder::from_operators(case.operators.clone())
        .parameters(parameters)
        .build()?;
    let supported_build_wall_time = supported_build_start.elapsed();

    let supported_factor_start = Instant::now();
    let supported_factor = supported.factor()?;
    let supported_factor_wall_time = supported_factor_start.elapsed();
    let diagnostics = supported_factor.diagnostics();

    let rhs = deterministic_probe(case.state_dimension);
    let eliminated_query_start = Instant::now();
    let eliminated_action = eliminated_factor.covariance_action(&rhs)?;
    let eliminated_query_wall_time = eliminated_query_start.elapsed();

    let supported_query_start = Instant::now();
    let supported_action = supported_factor.covariance_action(&rhs)?;
    let supported_query_wall_time = supported_query_start.elapsed();
    let covariance_action_absolute_error = l2_difference(&eliminated_action, &supported_action)?;
    let covariance_action_relative_error =
        covariance_action_absolute_error / l2_norm(&eliminated_action).max(f64::MIN_POSITIVE);

    Ok(ExactSupportedPilotRow {
        spatial_dimension: case.spatial_dimension,
        cells_per_axis: case.config.cells_per_axis,
        form_degree: case.form_degree,
        alpha,
        state_dimension: case.state_dimension,
        lower_adjacent_dimension: case.lower_adjacent_dimension,
        upper_adjacent_dimension: case.upper_adjacent_dimension,
        adjacent_dimension: case.lower_adjacent_dimension + case.upper_adjacent_dimension,
        ambient_dimension: diagnostics.ambient_dimension,
        support_dimension: diagnostics.support_dimension,
        constraint_dimension: diagnostics.constraint_dimension,
        eliminated_precision_nnz,
        supported_energy_nnz: diagnostics.ambient_energy_nnz,
        supported_constraint_nnz: diagnostics.constraint_nnz,
        supported_kkt_input_nnz: diagnostics.kkt_nnz,
        constraint_gram_certificate_relative_residual: diagnostics
            .certificate_residuals
            .constraint_gram_relative_residual,
        supported_energy_certificate_relative_residual: diagnostics
            .certificate_residuals
            .supported_energy_relative_residual,
        covariance_action_absolute_error,
        covariance_action_relative_error,
        feec_operator_build_wall_time: case.feec_operator_build_wall_time,
        eliminated_build_wall_time,
        eliminated_factor_wall_time,
        eliminated_query_wall_time,
        supported_build_wall_time,
        supported_factor_wall_time,
        supported_query_wall_time,
        query_wall_times_are_comparable: true,
    })
}

fn deterministic_probe(dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|index| {
            let coordinate = index as f64 + 1.0;
            (0.73 * coordinate).sin() + (0.19 * coordinate).cos()
        })
        .collect()
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn l2_difference(lhs: &[f64], rhs: &[f64]) -> Result<f64> {
    if lhs.len() != rhs.len() {
        return Err(FeecGmrfError::Dimension(
            "pilot covariance actions have different dimensions".to_string(),
        ));
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_pilot_agrees_at_backward_error_scale() {
        let config = ExactSupportedPilotConfig::tiny();
        let rows = run_exact_supported_pilot(&config).expect("tiny C51 pilot should run");
        let expected_rows = config
            .spatial_dimensions
            .iter()
            .map(|dimension| dimension + 1)
            .sum::<usize>()
            * 3;
        assert_eq!(rows.len(), expected_rows);

        for row in rows {
            assert!(row.state_dimension > 0);
            assert_eq!(row.support_dimension, row.state_dimension);
            assert_eq!(
                row.ambient_dimension - row.constraint_dimension,
                row.support_dimension
            );
            assert!(row.eliminated_precision_nnz > 0);
            assert!(row.supported_energy_nnz > 0);
            assert!(row.supported_kkt_input_nnz > 0);
            assert!(row
                .constraint_gram_certificate_relative_residual
                .unwrap_or(0.0)
                .is_finite());
            assert!(row
                .supported_energy_certificate_relative_residual
                .is_finite());
            assert!(row.covariance_action_absolute_error.is_finite());
            assert!(row.covariance_action_relative_error.is_finite());
            assert!(
                row.constraint_gram_certificate_relative_residual
                    .unwrap_or(0.0)
                    <= 1.0e-10
            );
            assert!(row.supported_energy_certificate_relative_residual <= 1.0e-10);
            assert!(
                row.covariance_action_relative_error <= 1.0e-9,
                "dimension {}, degree {}, alpha {:?}: relative covariance-action error {}",
                row.spatial_dimension,
                row.form_degree,
                row.alpha,
                row.covariance_action_relative_error
            );
            assert!(row.query_wall_times_are_comparable);
            for elapsed in [
                row.feec_operator_build_wall_time,
                row.eliminated_build_wall_time,
                row.eliminated_factor_wall_time,
                row.eliminated_query_wall_time,
                row.supported_build_wall_time,
                row.supported_factor_wall_time,
                row.supported_query_wall_time,
            ] {
                assert!(
                    !elapsed.is_zero(),
                    "each pilot stage must record a nonzero wall time"
                );
            }
        }
    }
}
