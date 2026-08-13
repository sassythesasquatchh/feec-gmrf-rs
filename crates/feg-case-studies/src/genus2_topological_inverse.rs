use crate::genus2_1form_hodge_conditioning::{
    build_cycle_observation_matrix, build_truth, default_genus2_torus_mesh_path,
    validate_genus2_topology, Genus2CycleObservation, Genus2TopologySummary,
};
#[cfg(test)]
use crate::test_util::lock_feec_harmonic_tests;
use crate::visual_output;
use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use ddf::cochain::Cochain;
use feg_infer::conditioning::hodge_1form::{
    build_coexact_1form_transform, build_exact_1form_transform,
    build_harmonic_restricted_precision, compute_harmonic_basis_1form,
    mass_orthonormalize_harmonic_basis_1form,
};
use feg_infer::prior::matern::one_form::{
    build_hodge_laplacian_1form, build_matern_precision_1form,
    build_reconstructed_barycenter_field_operator, MaternConfig as Matern1FormConfig,
    MaternMassInverse as Matern1FormMassInverse, ReconstructedBarycenterFieldOperator,
};
use feg_infer::prior::matern::two_form::{
    build_hodge_laplacian_2form, build_matern_precision_2form, MaternConfig as Matern2FormConfig,
    MaternMassInverse as Matern2FormMassInverse,
};
use feg_infer::prior::matern::zero_form::{
    build_laplace_beltrami_0form, build_matern_precision_0form, MaternConfig as Matern0FormConfig,
    MaternMassInverse as Matern0FormMassInverse,
};
use feg_infer::sparse::{
    dense_to_feec_csr, feec_csr_to_dense, feec_csr_to_gmrf, gmrf_vec_to_feec,
    sparse_row_operator_from_feec_csr_with_tolerance,
};
use formoniq::io::sample_1form_cell_vectors;
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector};
use gmrf_core::{ConstrainedPrecisionSolver, Gmrf, SparseRowOperator};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    io::gmsh::gmsh2coord_complex,
    topology::complex::Complex,
};
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const EPS: f64 = 1e-12;
const PERIOD_Z: f64 = 1.96;

#[derive(Debug, Clone)]
pub struct Genus2TopologicalInverseConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub harmonic_dim: usize,
    pub local_observation_count: usize,
    pub local_noise_std: f64,
    pub loop_noise_std: f64,
    pub posterior_sample_count: usize,
    pub surface_vector_variance_probe_count: usize,
    pub rng_seed: u64,
}

impl Default for Genus2TopologicalInverseConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_genus2_torus_mesh_path(),
            kappa: 4.0,
            tau: 1.0,
            harmonic_dim: 4,
            local_observation_count: 24,
            local_noise_std: 1e-2,
            loop_noise_std: 1e-4,
            posterior_sample_count: 2,
            surface_vector_variance_probe_count: 64,
            rng_seed: 202,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Genus2ObservationScenario {
    LocalOnly,
    LocalPlus1Loop,
    LocalPlus2Loops,
    LocalPlusAll4Loops,
}

impl Genus2ObservationScenario {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LocalOnly => "local_only",
            Self::LocalPlus1Loop => "local_plus_1_loop",
            Self::LocalPlus2Loops => "local_plus_2_loops",
            Self::LocalPlusAll4Loops => "local_plus_all_4_loops",
        }
    }

    pub fn loop_count(self) -> usize {
        match self {
            Self::LocalOnly => 0,
            Self::LocalPlus1Loop => 1,
            Self::LocalPlus2Loops => 2,
            Self::LocalPlusAll4Loops => 4,
        }
    }

    fn all() -> [Self; 4] {
        [
            Self::LocalOnly,
            Self::LocalPlus1Loop,
            Self::LocalPlus2Loops,
            Self::LocalPlusAll4Loops,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Genus2PeriodSummary {
    pub scenario: Genus2ObservationScenario,
    pub model: String,
    pub cycle_index: usize,
    pub cycle_name: String,
    pub truth_period: f64,
    pub posterior_mean: f64,
    pub posterior_std: f64,
    pub posterior_lower_95: f64,
    pub posterior_upper_95: f64,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
    pub residual: f64,
    pub observed_in_scenario: bool,
}

#[derive(Debug, Clone)]
pub struct Genus2LocalObservation {
    pub observation_index: usize,
    pub edge_index: usize,
    pub observed_value: f64,
    pub noise_std: f64,
}

#[derive(Debug, Clone)]
pub struct Genus2TopologicalInverseResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub topology_summary: Genus2TopologySummary,
    pub truth: FeecVector,
    pub harmonic_basis: FeecMatrix,
    pub cycle_observation_matrix: FeecCsr,
    pub cycle_observations: Vec<Genus2CycleObservation>,
    pub cycle_harmonic_pairing: FeecMatrix,
    pub local_observations: Vec<Genus2LocalObservation>,
    pub scenarios: Vec<Genus2ScenarioResult>,
    pub harmonic_dense_prior: Vec<Genus2HarmonicDensePriorScenarioResult>,
    pub kappa0_constrained_periods: Genus2Kappa0ConstrainedPeriodDiagnostic,
}

#[derive(Debug, Clone)]
pub struct Genus2ScenarioResult {
    pub scenario: Genus2ObservationScenario,
    pub observation_count: usize,
    pub joint: Genus2JointPosteriorResult,
    pub scalar_baseline: Genus2ScalarBaselineResult,
    pub period_summaries: Vec<Genus2PeriodSummary>,
}

#[derive(Debug, Clone)]
pub struct Genus2JointPosteriorResult {
    pub posterior_mean_total: FeecVector,
    pub posterior_mean_exact: FeecVector,
    pub posterior_mean_coexact: FeecVector,
    pub posterior_mean_harmonic: FeecVector,
    pub posterior_samples: Vec<Genus2JointPosteriorSample>,
    pub surface_vector_variance: Genus2JointSurfaceVectorVarianceFields,
    pub period_variance_by_branch: Genus2JointPeriodVarianceFields,
    pub period_prior_variance: FeecVector,
    pub period_posterior_variance: FeecVector,
    pub period_posterior_mean: FeecVector,
}

#[derive(Debug, Clone)]
pub struct Genus2JointPeriodVarianceFields {
    pub total: Genus2PeriodVarianceField,
    pub exact: Genus2PeriodVarianceField,
    pub coexact: Genus2PeriodVarianceField,
    pub harmonic: Genus2PeriodVarianceField,
}

#[derive(Debug, Clone)]
pub struct Genus2PeriodVarianceField {
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub variance_ratio: FeecVector,
}

#[derive(Debug, Clone)]
pub struct Genus2JointPosteriorSample {
    pub exact: FeecVector,
    pub coexact: FeecVector,
    pub harmonic: FeecVector,
}

#[derive(Debug, Clone)]
pub struct Genus2JointSurfaceVectorVarianceFields {
    pub total: Genus2SurfaceVectorVarianceFields,
    pub exact: Genus2SurfaceVectorVarianceFields,
    pub coexact: Genus2SurfaceVectorVarianceFields,
    pub harmonic: Genus2SurfaceVectorVarianceFields,
}

#[derive(Debug, Clone)]
pub struct Genus2SurfaceVectorVarianceFields {
    pub prior_components: Vec<FeecVector>,
    pub posterior_components: Vec<FeecVector>,
    pub prior_trace: FeecVector,
    pub posterior_trace: FeecVector,
    pub trace_ratio: FeecVector,
}

impl Genus2SurfaceVectorVarianceFields {
    pub fn prior_vtk_vectors(&self) -> Vec<[f64; 3]> {
        components_to_vtk_vectors(&self.prior_components)
    }

    pub fn posterior_vtk_vectors(&self) -> Vec<[f64; 3]> {
        components_to_vtk_vectors(&self.posterior_components)
    }
}

#[derive(Debug, Clone)]
pub struct Genus2ScalarBaselineResult {
    pub period_summaries: Vec<Genus2PeriodSummary>,
}

#[derive(Debug, Clone)]
pub struct Genus2HarmonicDensePriorScenarioResult {
    pub scenario: Genus2ObservationScenario,
    pub observation_count: usize,
    pub period_summaries: Vec<Genus2PeriodSummary>,
}

#[derive(Debug, Clone)]
pub struct Genus2Kappa0ConstrainedPeriodDiagnostic {
    pub treatment: String,
    pub constraint_rank: usize,
    pub constrained_period_variance: FeecVector,
}

#[derive(Debug, Clone, Copy)]
enum SurfaceVectorFieldKind {
    Total,
    Exact,
    Coexact,
    Harmonic,
}

#[derive(Debug, Clone)]
struct PreparedJointOperators {
    exact_transform: FeecCsr,
    coexact_transform: FeecCsr,
    harmonic_transform: FeecMatrix,
    barycenter_operator: ReconstructedBarycenterFieldOperator,
    joint_precision: FeecCsr,
    exact_precision: FeecCsr,
    exact_dim: usize,
    coexact_dim: usize,
    harmonic_dim: usize,
}

#[derive(Debug, Clone)]
struct ObservationRow {
    noise_std: f64,
    entries: Vec<(usize, f64)>,
}

pub fn run_genus2_topological_inverse_problem(
    config: &Genus2TopologicalInverseConfig,
) -> Result<Genus2TopologicalInverseResult, Box<dyn Error>> {
    validate_config(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let topology_summary = validate_genus2_topology(&topology)?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw =
        compute_harmonic_basis_1form(&topology, &metric, config.harmonic_dim, None)
            .map_err(invalid_data)?;
    let harmonic_basis =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)
            .map_err(invalid_data)?;
    let truth = build_truth(&topology, &coords, &hodge.mass_u, &harmonic_basis)?;
    let (cycle_observation_matrix, cycle_observations) =
        build_cycle_observation_matrix(&topology, &coords)?;
    let cycle_harmonic_pairing = &cycle_observation_matrix * &harmonic_basis;
    if cycle_harmonic_pairing.rank(1e-8) != config.harmonic_dim {
        return Err(invalid_data("cycle-harmonic pairing must have rank 4").into());
    }

    let operators =
        build_joint_operators(config, &topology, &coords, &metric, &hodge, &harmonic_basis)?;
    let local_edges = select_local_observation_edges(
        &topology,
        &coords,
        &cycle_observation_matrix,
        config.local_observation_count,
    );
    let local_observations = local_edges
        .iter()
        .copied()
        .enumerate()
        .map(|(observation_index, edge_index)| Genus2LocalObservation {
            observation_index,
            edge_index,
            observed_value: truth[edge_index],
            noise_std: config.local_noise_std,
        })
        .collect::<Vec<_>>();
    let local_rows = local_observations
        .iter()
        .map(|observation| ObservationRow {
            noise_std: observation.noise_std,
            entries: vec![(observation.edge_index, 1.0)],
        })
        .collect::<Vec<_>>();
    let cycle_rows = cycle_rows(&cycle_observation_matrix)
        .into_iter()
        .map(|entries| ObservationRow {
            noise_std: config.loop_noise_std,
            entries,
        })
        .collect::<Vec<_>>();

    let mut scenarios = Vec::new();
    for scenario in Genus2ObservationScenario::all() {
        let rows = scenario_rows(scenario, &local_rows, &cycle_rows);
        scenarios.push(run_scenario(
            config,
            scenario,
            &topology,
            &truth,
            &cycle_observations,
            &cycle_observation_matrix,
            &operators,
            &rows,
        )?);
    }
    let harmonic_dense_prior = run_harmonic_dense_prior_scenarios(
        &truth,
        &cycle_observations,
        &cycle_observation_matrix,
        &harmonic_basis,
        &local_rows,
        &cycle_rows,
    )?;
    let kappa0_constrained_periods =
        build_kappa0_constrained_period_diagnostic(&cycle_harmonic_pairing)?;

    Ok(Genus2TopologicalInverseResult {
        topology,
        coords,
        topology_summary,
        truth,
        harmonic_basis,
        cycle_observation_matrix,
        cycle_observations,
        cycle_harmonic_pairing,
        local_observations,
        scenarios,
        harmonic_dense_prior,
        kappa0_constrained_periods,
    })
}

pub fn write_genus2_topological_inverse_outputs(
    result: &Genus2TopologicalInverseResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_experiment4_summary(result, out_dir.join("experiment4_summary.txt"))?;
    write_period_posterior_summary(result, out_dir.join("period_posterior_summary.csv"))?;
    write_period_branch_variance_summary(
        result,
        out_dir.join("period_branch_variance_summary.csv"),
    )?;
    write_surface_branch_variance_summary(
        result,
        out_dir.join("surface_branch_variance_summary.csv"),
    )?;
    write_harmonic_dense_prior_summary(result, out_dir.join("harmonic_dense_prior_summary.csv"))?;
    write_kappa0_constrained_period_summary(
        result,
        out_dir.join("kappa0_constrained_period_summary.csv"),
    )?;
    write_sensitivity_summary(result, out_dir.join("sensitivity_summary.csv"))?;
    write_local_observations_vtu(result, out_dir.join("local_observations.vtu"))?;
    for scenario in &result.scenarios {
        write_scenario_outputs(result, scenario, out_dir)?;
    }
    Ok(())
}

fn validate_config(config: &Genus2TopologicalInverseConfig) -> io::Result<()> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive"));
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive"));
    }
    if config.harmonic_dim != 4 {
        return Err(invalid_input(
            "the genus-2 inverse problem expects harmonic_dim = 4",
        ));
    }
    if config.local_observation_count == 0 {
        return Err(invalid_input("local_observation_count must be positive"));
    }
    if !config.local_noise_std.is_finite() || config.local_noise_std <= 0.0 {
        return Err(invalid_input("local_noise_std must be finite and positive"));
    }
    if !config.loop_noise_std.is_finite() || config.loop_noise_std <= 0.0 {
        return Err(invalid_input("loop_noise_std must be finite and positive"));
    }
    if config.surface_vector_variance_probe_count == 0 {
        return Err(invalid_input(
            "surface_vector_variance_probe_count must be positive",
        ));
    }
    Ok(())
}

fn build_joint_operators(
    config: &Genus2TopologicalInverseConfig,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &feg_infer::prior::matern::one_form::HodgeLaplacian1Form,
    harmonic_basis: &FeecMatrix,
) -> Result<PreparedJointOperators, Box<dyn Error>> {
    let exact_transform = build_exact_1form_transform(topology);
    let coexact_transform = build_coexact_1form_transform(topology, metric, &hodge.mass_u);
    let barycenter_operator =
        build_reconstructed_barycenter_field_operator(topology, coords).map_err(invalid_data)?;
    let q0 = build_matern_precision_0form(
        &build_laplace_beltrami_0form(topology, metric),
        Matern0FormConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: Matern0FormMassInverse::RowSumLumped,
        },
    );
    let q2 = build_matern_precision_2form(
        topology,
        metric,
        &build_hodge_laplacian_2form(topology, metric).map_err(invalid_data)?,
        Matern2FormConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: Matern2FormMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
        },
    )
    .map_err(invalid_data)?;
    let q1 = build_matern_precision_1form(
        topology,
        metric,
        hodge,
        Matern1FormConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    );
    let qh = build_harmonic_restricted_precision(&q1, harmonic_basis);
    let joint_precision = block_diag(&[&q0, &q2, &qh]);

    Ok(PreparedJointOperators {
        exact_transform,
        coexact_transform,
        harmonic_transform: harmonic_basis.clone(),
        barycenter_operator,
        joint_precision,
        exact_precision: q0,
        exact_dim: topology.vertices().len(),
        coexact_dim: topology.cells().len(),
        harmonic_dim: harmonic_basis.ncols(),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_scenario(
    config: &Genus2TopologicalInverseConfig,
    scenario: Genus2ObservationScenario,
    topology: &Complex,
    truth: &FeecVector,
    cycle_observations: &[Genus2CycleObservation],
    cycle_observation_matrix: &FeecCsr,
    operators: &PreparedJointOperators,
    rows: &[ObservationRow],
) -> Result<Genus2ScenarioResult, Box<dyn Error>> {
    let edge_observation_matrix = scaled_edge_observation_matrix(rows, topology.edges().len());
    let observations = &edge_observation_matrix * truth;
    let joint_observation_matrix = joint_observation_matrix(&edge_observation_matrix, operators);
    let (posterior_precision, information) = apply_gaussian_observations(
        &feec_csr_to_gmrf(&operators.joint_precision),
        &feec_csr_to_gmrf(&joint_observation_matrix),
        &feec_vec_to_gmrf_local(&observations),
        None,
        1.0,
    );
    let mut posterior =
        Gmrf::from_information_and_precision(information, posterior_precision.clone())?;
    let posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let joint = build_joint_result(
        config,
        scenario,
        &mut posterior,
        &operators.joint_precision,
        &posterior_mean,
        cycle_observation_matrix,
        operators,
    )?;
    let scalar_baseline = run_scalar_baseline(
        scenario,
        truth,
        cycle_observations,
        cycle_observation_matrix,
        &edge_observation_matrix,
        operators,
    )?;
    let period_summaries = build_period_summaries(
        scenario,
        "joint_hodge",
        truth,
        cycle_observations,
        cycle_observation_matrix,
        &joint.period_posterior_mean,
        &joint.period_prior_variance,
        &joint.period_posterior_variance,
    );

    Ok(Genus2ScenarioResult {
        scenario,
        observation_count: rows.len(),
        joint,
        scalar_baseline,
        period_summaries,
    })
}

fn build_joint_result(
    config: &Genus2TopologicalInverseConfig,
    scenario: Genus2ObservationScenario,
    posterior: &mut Gmrf,
    prior_precision: &FeecCsr,
    posterior_mean: &FeecVector,
    cycle_observation_matrix: &FeecCsr,
    operators: &PreparedJointOperators,
) -> Result<Genus2JointPosteriorResult, Box<dyn Error>> {
    let (exact_latent, coexact_latent, harmonic_latent) =
        split_joint_latent(posterior_mean, operators);
    let posterior_mean_exact = &operators.exact_transform * &exact_latent;
    let posterior_mean_coexact = &operators.coexact_transform * &coexact_latent;
    let posterior_mean_harmonic = &operators.harmonic_transform * &harmonic_latent;
    let posterior_mean_total =
        &(&posterior_mean_exact + &posterior_mean_coexact) + &posterior_mean_harmonic;

    let period_operator = joint_observation_matrix(cycle_observation_matrix, operators);
    let mut prior = Gmrf::from_mean_and_precision(
        GmrfVector::zeros(prior_precision.nrows()),
        feec_csr_to_gmrf(prior_precision),
    )?;
    let period_prior_variance = exact_transformed_variances(&mut prior, &period_operator)?;
    let period_posterior_variance = exact_transformed_variances(posterior, &period_operator)?;
    let period_posterior_mean = &period_operator * posterior_mean;
    let period_variance_by_branch = build_joint_period_variance_fields(
        &mut prior,
        posterior,
        cycle_observation_matrix,
        operators,
    )?;
    let surface_vector_variance = build_joint_surface_vector_variance_fields(
        config, scenario, &mut prior, posterior, operators,
    )?;
    let posterior_samples =
        sample_joint_posterior_components(config, scenario, posterior, operators)?;

    Ok(Genus2JointPosteriorResult {
        posterior_mean_total,
        posterior_mean_exact,
        posterior_mean_coexact,
        posterior_mean_harmonic,
        posterior_samples,
        surface_vector_variance,
        period_variance_by_branch,
        period_prior_variance,
        period_posterior_variance,
        period_posterior_mean,
    })
}

fn build_joint_period_variance_fields(
    prior: &mut Gmrf,
    posterior: &mut Gmrf,
    cycle_observation_matrix: &FeecCsr,
    operators: &PreparedJointOperators,
) -> Result<Genus2JointPeriodVarianceFields, Box<dyn Error>> {
    Ok(Genus2JointPeriodVarianceFields {
        total: build_period_variance_field(
            prior,
            posterior,
            &period_operator_for_kind(
                SurfaceVectorFieldKind::Total,
                cycle_observation_matrix,
                operators,
            ),
        )?,
        exact: build_period_variance_field(
            prior,
            posterior,
            &period_operator_for_kind(
                SurfaceVectorFieldKind::Exact,
                cycle_observation_matrix,
                operators,
            ),
        )?,
        coexact: build_period_variance_field(
            prior,
            posterior,
            &period_operator_for_kind(
                SurfaceVectorFieldKind::Coexact,
                cycle_observation_matrix,
                operators,
            ),
        )?,
        harmonic: build_period_variance_field(
            prior,
            posterior,
            &period_operator_for_kind(
                SurfaceVectorFieldKind::Harmonic,
                cycle_observation_matrix,
                operators,
            ),
        )?,
    })
}

fn build_period_variance_field(
    prior: &mut Gmrf,
    posterior: &mut Gmrf,
    operator: &FeecCsr,
) -> Result<Genus2PeriodVarianceField, Box<dyn Error>> {
    let prior_variance = exact_transformed_variances(prior, operator)?;
    let posterior_variance = exact_transformed_variances(posterior, operator)?;
    let variance_ratio = ratio_vector(&posterior_variance, &prior_variance);
    Ok(Genus2PeriodVarianceField {
        prior_variance,
        posterior_variance,
        variance_ratio,
    })
}

fn run_scalar_baseline(
    scenario: Genus2ObservationScenario,
    truth: &FeecVector,
    cycle_observations: &[Genus2CycleObservation],
    cycle_observation_matrix: &FeecCsr,
    edge_observation_matrix: &FeecCsr,
    operators: &PreparedJointOperators,
) -> Result<Genus2ScalarBaselineResult, Box<dyn Error>> {
    let observations = edge_observation_matrix * truth;
    let exact_observation_matrix = edge_observation_matrix * &operators.exact_transform;
    let (posterior_precision, information) = apply_gaussian_observations(
        &feec_csr_to_gmrf(&operators.exact_precision),
        &feec_csr_to_gmrf(&exact_observation_matrix),
        &feec_vec_to_gmrf_local(&observations),
        None,
        1.0,
    );
    let posterior = Gmrf::from_information_and_precision(information, posterior_precision)?;
    let _exact_posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let zero_mean = FeecVector::zeros(cycle_observations.len());
    let zero_variance = FeecVector::zeros(cycle_observations.len());
    let period_summaries = build_period_summaries(
        scenario,
        "scalar_potential",
        truth,
        cycle_observations,
        cycle_observation_matrix,
        &zero_mean,
        &zero_variance,
        &zero_variance,
    );
    Ok(Genus2ScalarBaselineResult { period_summaries })
}

fn run_harmonic_dense_prior_scenarios(
    truth: &FeecVector,
    cycle_observations: &[Genus2CycleObservation],
    cycle_observation_matrix: &FeecCsr,
    harmonic_basis: &FeecMatrix,
    local_rows: &[ObservationRow],
    cycle_rows: &[ObservationRow],
) -> Result<Vec<Genus2HarmonicDensePriorScenarioResult>, Box<dyn Error>> {
    let mut scenarios = Vec::new();
    for scenario in Genus2ObservationScenario::all() {
        let rows = scenario_rows(scenario, local_rows, cycle_rows);
        scenarios.push(run_harmonic_dense_prior_scenario(
            scenario,
            truth,
            cycle_observations,
            cycle_observation_matrix,
            harmonic_basis,
            &rows,
        )?);
    }
    Ok(scenarios)
}

fn run_harmonic_dense_prior_scenario(
    scenario: Genus2ObservationScenario,
    truth: &FeecVector,
    cycle_observations: &[Genus2CycleObservation],
    cycle_observation_matrix: &FeecCsr,
    harmonic_basis: &FeecMatrix,
    rows: &[ObservationRow],
) -> Result<Genus2HarmonicDensePriorScenarioResult, Box<dyn Error>> {
    let edge_observation_matrix = scaled_edge_observation_matrix(rows, truth.len());
    let observations = &edge_observation_matrix * truth;
    let harmonic_observation_matrix = dense_to_feec_csr(
        &(feec_csr_to_dense(&edge_observation_matrix) * harmonic_basis),
        EPS,
    );
    let harmonic_prior_precision = identity_feec_csr(harmonic_basis.ncols());
    let (posterior_precision, information) = apply_gaussian_observations(
        &feec_csr_to_gmrf(&harmonic_prior_precision),
        &feec_csr_to_gmrf(&harmonic_observation_matrix),
        &feec_vec_to_gmrf_local(&observations),
        None,
        1.0,
    );
    let mut prior = Gmrf::from_mean_and_precision(
        GmrfVector::zeros(harmonic_basis.ncols()),
        feec_csr_to_gmrf(&harmonic_prior_precision),
    )?;
    let mut posterior =
        Gmrf::from_information_and_precision(information, posterior_precision.clone())?;
    let posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let period_operator = dense_to_feec_csr(
        &(feec_csr_to_dense(cycle_observation_matrix) * harmonic_basis),
        EPS,
    );
    let period_prior_variance = exact_transformed_variances(&mut prior, &period_operator)?;
    let period_posterior_variance = exact_transformed_variances(&mut posterior, &period_operator)?;
    let period_posterior_mean = &period_operator * &posterior_mean;
    let period_summaries = build_period_summaries(
        scenario,
        "dense_harmonic",
        truth,
        cycle_observations,
        cycle_observation_matrix,
        &period_posterior_mean,
        &period_prior_variance,
        &period_posterior_variance,
    );

    Ok(Genus2HarmonicDensePriorScenarioResult {
        scenario,
        observation_count: rows.len(),
        period_summaries,
    })
}

fn build_kappa0_constrained_period_diagnostic(
    cycle_harmonic_pairing: &FeecMatrix,
) -> Result<Genus2Kappa0ConstrainedPeriodDiagnostic, Box<dyn Error>> {
    let harmonic_dim = cycle_harmonic_pairing.ncols();
    let zero_precision = dense_to_feec_csr(&FeecMatrix::zeros(harmonic_dim, harmonic_dim), EPS);
    let constraints = feec_dense_to_gmrf_dense(cycle_harmonic_pairing);
    let solver = ConstrainedPrecisionSolver::new(&feec_csr_to_gmrf(&zero_precision), &constraints)?;
    let period_operator = sparse_row_operator_from_feec_csr_with_tolerance(
        &dense_to_feec_csr(cycle_harmonic_pairing, EPS),
        EPS,
    )?;
    let constrained_period_variance =
        gmrf_vec_to_feec(&solver.exact_transformed_variances(&period_operator)?);

    Ok(Genus2Kappa0ConstrainedPeriodDiagnostic {
        treatment: "kappa0_improper_harmonic_fully_period_constrained".to_string(),
        constraint_rank: cycle_harmonic_pairing.rank(1e-8),
        constrained_period_variance,
    })
}

fn sample_joint_posterior_components(
    config: &Genus2TopologicalInverseConfig,
    scenario: Genus2ObservationScenario,
    posterior: &mut Gmrf,
    operators: &PreparedJointOperators,
) -> Result<Vec<Genus2JointPosteriorSample>, Box<dyn Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(
        config
            .rng_seed
            .wrapping_add(0x7A5A_0000)
            .wrapping_add(scenario.loop_count() as u64),
    );
    let mut samples = Vec::with_capacity(config.posterior_sample_count);
    for _ in 0..config.posterior_sample_count {
        let sample = gmrf_vec_to_feec(&posterior.sample(&mut rng)?);
        let (exact_latent, coexact_latent, harmonic_latent) =
            split_joint_latent(&sample, operators);
        samples.push(Genus2JointPosteriorSample {
            exact: &operators.exact_transform * &exact_latent,
            coexact: &operators.coexact_transform * &coexact_latent,
            harmonic: &operators.harmonic_transform * &harmonic_latent,
        });
    }
    Ok(samples)
}

fn build_joint_surface_vector_variance_fields(
    config: &Genus2TopologicalInverseConfig,
    scenario: Genus2ObservationScenario,
    prior: &mut Gmrf,
    posterior: &mut Gmrf,
    operators: &PreparedJointOperators,
) -> Result<Genus2JointSurfaceVectorVarianceFields, Box<dyn Error>> {
    Ok(Genus2JointSurfaceVectorVarianceFields {
        total: build_surface_vector_variance_fields(
            config,
            scenario,
            SurfaceVectorFieldKind::Total,
            prior,
            posterior,
            operators,
        )?,
        exact: build_surface_vector_variance_fields(
            config,
            scenario,
            SurfaceVectorFieldKind::Exact,
            prior,
            posterior,
            operators,
        )?,
        coexact: build_surface_vector_variance_fields(
            config,
            scenario,
            SurfaceVectorFieldKind::Coexact,
            prior,
            posterior,
            operators,
        )?,
        harmonic: build_surface_vector_variance_fields(
            config,
            scenario,
            SurfaceVectorFieldKind::Harmonic,
            prior,
            posterior,
            operators,
        )?,
    })
}

fn build_surface_vector_variance_fields(
    config: &Genus2TopologicalInverseConfig,
    scenario: Genus2ObservationScenario,
    kind: SurfaceVectorFieldKind,
    prior: &mut Gmrf,
    posterior: &mut Gmrf,
    operators: &PreparedJointOperators,
) -> Result<Genus2SurfaceVectorVarianceFields, Box<dyn Error>> {
    let operator = surface_vector_row_operator(kind, operators)?;
    let empty_constraints = GmrfDenseMatrix::zeros(0, operator.ncols);
    let seed = config
        .rng_seed
        .wrapping_add(0x5A17_0000)
        .wrapping_add((scenario.loop_count() as u64) << 8)
        .wrapping_add(surface_vector_field_seed_offset(kind));
    let mut prior_rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut posterior_rng = rand::rngs::StdRng::seed_from_u64(seed);
    let prior_stacked = prior
        .hutchinson_transformed_variance_decomposition(
            &operator,
            &empty_constraints,
            config.surface_vector_variance_probe_count,
            &mut prior_rng,
        )?
        .constrained_diag;
    let posterior_stacked = posterior
        .hutchinson_transformed_variance_decomposition(
            &operator,
            &empty_constraints,
            config.surface_vector_variance_probe_count,
            &mut posterior_rng,
        )?
        .constrained_diag;
    Ok(build_surface_vector_variance_from_stacked(
        &prior_stacked,
        &posterior_stacked,
        operators.barycenter_operator.ambient_dim(),
    ))
}

fn surface_vector_row_operator(
    kind: SurfaceVectorFieldKind,
    operators: &PreparedJointOperators,
) -> Result<SparseRowOperator, Box<dyn Error>> {
    let edge_to_joint = edge_to_joint_transform(kind, operators);
    let edge_to_joint = sparse_row_operator_from_feec_csr_with_tolerance(&edge_to_joint, EPS)?;
    let mut component_operators = Vec::with_capacity(operators.barycenter_operator.ambient_dim());
    for component_index in 0..operators.barycenter_operator.ambient_dim() {
        let rows = operators
            .barycenter_operator
            .component_rows(component_index)
            .ok_or_else(|| invalid_data(format!("missing barycenter component {component_index}")))?
            .to_vec();
        let barycenter_component = SparseRowOperator::new(operators.exact_transform.nrows(), rows)?;
        component_operators.push(SparseRowOperator::compose(
            &barycenter_component,
            &edge_to_joint,
        )?);
    }
    let refs = component_operators.iter().collect::<Vec<_>>();
    Ok(SparseRowOperator::stack(&refs)?)
}

fn period_operator_for_kind(
    kind: SurfaceVectorFieldKind,
    cycle_observation_matrix: &FeecCsr,
    operators: &PreparedJointOperators,
) -> FeecCsr {
    cycle_observation_matrix * &edge_to_joint_transform(kind, operators)
}

fn edge_to_joint_transform(
    kind: SurfaceVectorFieldKind,
    operators: &PreparedJointOperators,
) -> FeecCsr {
    let edge_count = operators.exact_transform.nrows();
    let joint_dim = operators.exact_dim + operators.coexact_dim + operators.harmonic_dim;
    let mut coo = FeecCoo::new(edge_count, joint_dim);
    if matches!(
        kind,
        SurfaceVectorFieldKind::Total | SurfaceVectorFieldKind::Exact
    ) {
        for (row, col, value) in operators.exact_transform.triplet_iter() {
            coo.push(row, col, *value);
        }
    }
    if matches!(
        kind,
        SurfaceVectorFieldKind::Total | SurfaceVectorFieldKind::Coexact
    ) {
        for (row, col, value) in operators.coexact_transform.triplet_iter() {
            coo.push(row, operators.exact_dim + col, *value);
        }
    }
    if matches!(
        kind,
        SurfaceVectorFieldKind::Total | SurfaceVectorFieldKind::Harmonic
    ) {
        let offset = operators.exact_dim + operators.coexact_dim;
        for row in 0..operators.harmonic_transform.nrows() {
            for col in 0..operators.harmonic_transform.ncols() {
                let value = operators.harmonic_transform[(row, col)];
                if value.abs() > EPS {
                    coo.push(row, offset + col, value);
                }
            }
        }
    }
    FeecCsr::from(&coo)
}

fn build_surface_vector_variance_from_stacked(
    prior_stacked: &GmrfVector,
    posterior_stacked: &GmrfVector,
    ambient_dim: usize,
) -> Genus2SurfaceVectorVarianceFields {
    let cell_count = if ambient_dim == 0 {
        0
    } else {
        prior_stacked.len() / ambient_dim
    };
    let prior_components = split_stacked_components(prior_stacked, ambient_dim, cell_count);
    let posterior_components = split_stacked_components(posterior_stacked, ambient_dim, cell_count);
    let prior_trace = sum_component_vectors(&prior_components, cell_count);
    let posterior_trace = sum_component_vectors(&posterior_components, cell_count);
    let trace_ratio = ratio_vector(&posterior_trace, &prior_trace);
    Genus2SurfaceVectorVarianceFields {
        prior_components,
        posterior_components,
        prior_trace,
        posterior_trace,
        trace_ratio,
    }
}

fn split_stacked_components(
    stacked: &GmrfVector,
    ambient_dim: usize,
    cell_count: usize,
) -> Vec<FeecVector> {
    (0..ambient_dim)
        .map(|component_index| {
            FeecVector::from_iterator(
                cell_count,
                (0..cell_count)
                    .map(|cell_index| stacked[component_index * cell_count + cell_index]),
            )
        })
        .collect()
}

fn build_period_summaries(
    scenario: Genus2ObservationScenario,
    model: &str,
    truth: &FeecVector,
    cycle_observations: &[Genus2CycleObservation],
    cycle_observation_matrix: &FeecCsr,
    posterior_mean: &FeecVector,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
) -> Vec<Genus2PeriodSummary> {
    let truth_periods = cycle_observation_matrix * truth;
    cycle_observations
        .iter()
        .enumerate()
        .map(|(cycle_index, cycle)| {
            let posterior_std = posterior_variance[cycle_index].max(0.0).sqrt();
            let mean = posterior_mean[cycle_index];
            Genus2PeriodSummary {
                scenario,
                model: model.to_string(),
                cycle_index,
                cycle_name: cycle.name.clone(),
                truth_period: truth_periods[cycle_index],
                posterior_mean: mean,
                posterior_std,
                posterior_lower_95: mean - PERIOD_Z * posterior_std,
                posterior_upper_95: mean + PERIOD_Z * posterior_std,
                prior_variance: prior_variance[cycle_index],
                posterior_variance: posterior_variance[cycle_index],
                variance_ratio: safe_ratio(
                    posterior_variance[cycle_index],
                    prior_variance[cycle_index],
                ),
                residual: mean - truth_periods[cycle_index],
                observed_in_scenario: cycle_index < scenario.loop_count(),
            }
        })
        .collect()
}

fn scenario_rows(
    scenario: Genus2ObservationScenario,
    local_rows: &[ObservationRow],
    cycle_rows: &[ObservationRow],
) -> Vec<ObservationRow> {
    let mut rows = local_rows.to_vec();
    rows.extend(cycle_rows.iter().take(scenario.loop_count()).cloned());
    rows
}

fn scaled_edge_observation_matrix(rows: &[ObservationRow], edge_count: usize) -> FeecCsr {
    let mut coo = FeecCoo::new(rows.len(), edge_count);
    for (row_index, row) in rows.iter().enumerate() {
        for (edge_index, value) in &row.entries {
            coo.push(row_index, *edge_index, *value / row.noise_std);
        }
    }
    FeecCsr::from(&coo)
}

fn joint_observation_matrix(
    edge_observation_matrix: &FeecCsr,
    operators: &PreparedJointOperators,
) -> FeecCsr {
    let exact = edge_observation_matrix * &operators.exact_transform;
    let coexact = edge_observation_matrix * &operators.coexact_transform;
    let harmonic_dense = feec_csr_to_dense(edge_observation_matrix) * &operators.harmonic_transform;
    hstack3(&exact, &coexact, &dense_to_feec_csr(&harmonic_dense, EPS))
}

fn split_joint_latent(
    latent: &FeecVector,
    operators: &PreparedJointOperators,
) -> (FeecVector, FeecVector, FeecVector) {
    let exact = FeecVector::from_iterator(
        operators.exact_dim,
        (0..operators.exact_dim).map(|i| latent[i]),
    );
    let coexact_offset = operators.exact_dim;
    let coexact = FeecVector::from_iterator(
        operators.coexact_dim,
        (0..operators.coexact_dim).map(|i| latent[coexact_offset + i]),
    );
    let harmonic_offset = operators.exact_dim + operators.coexact_dim;
    let harmonic = FeecVector::from_iterator(
        operators.harmonic_dim,
        (0..operators.harmonic_dim).map(|i| latent[harmonic_offset + i]),
    );
    (exact, coexact, harmonic)
}

fn block_diag(blocks: &[&FeecCsr]) -> FeecCsr {
    let total_rows = blocks.iter().map(|block| block.nrows()).sum();
    let total_cols = blocks.iter().map(|block| block.ncols()).sum();
    let mut coo = FeecCoo::new(total_rows, total_cols);
    let mut row_offset = 0;
    let mut col_offset = 0;
    for block in blocks {
        for (row, col, value) in block.triplet_iter() {
            coo.push(row + row_offset, col + col_offset, *value);
        }
        row_offset += block.nrows();
        col_offset += block.ncols();
    }
    FeecCsr::from(&coo)
}

fn identity_feec_csr(dim: usize) -> FeecCsr {
    let mut coo = FeecCoo::new(dim, dim);
    for i in 0..dim {
        coo.push(i, i, 1.0);
    }
    FeecCsr::from(&coo)
}

fn hstack3(lhs: &FeecCsr, mid: &FeecCsr, rhs: &FeecCsr) -> FeecCsr {
    assert_eq!(lhs.nrows(), mid.nrows());
    assert_eq!(lhs.nrows(), rhs.nrows());
    let mut coo = FeecCoo::new(lhs.nrows(), lhs.ncols() + mid.ncols() + rhs.ncols());
    for (row, col, value) in lhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    for (row, col, value) in mid.triplet_iter() {
        coo.push(row, col + lhs.ncols(), *value);
    }
    for (row, col, value) in rhs.triplet_iter() {
        coo.push(row, col + lhs.ncols() + mid.ncols(), *value);
    }
    FeecCsr::from(&coo)
}

fn cycle_rows(cycle_observation_matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    (0..cycle_observation_matrix.nrows())
        .map(|row| sparse_row_entries(cycle_observation_matrix, row))
        .collect()
}

fn sparse_row_entries(matrix: &FeecCsr, row_index: usize) -> Vec<(usize, f64)> {
    let mut entries = Vec::new();
    for (row, col, value) in matrix.triplet_iter() {
        if row == row_index && value.abs() > EPS {
            entries.push((col, *value));
        }
    }
    entries
}

pub(crate) fn select_local_observation_edges(
    topology: &Complex,
    coords: &MeshCoords,
    cycle_observation_matrix: &FeecCsr,
    count: usize,
) -> Vec<usize> {
    let cycle_edges = cycle_edge_set(cycle_observation_matrix);
    let cycle_centers = cycle_edges
        .iter()
        .map(|edge_index| edge_midpoint(topology, coords, *edge_index))
        .collect::<Vec<_>>();
    let mut candidates = topology
        .edges()
        .handle_iter()
        .filter_map(|edge| {
            let edge_index = edge.kidx();
            if cycle_edges.contains(&edge_index) {
                None
            } else {
                let center = edge_midpoint(topology, coords, edge_index);
                let cycle_distance = cycle_centers
                    .iter()
                    .map(|other| point_distance(&center, other))
                    .fold(f64::INFINITY, f64::min);
                Some((edge_index, center, cycle_distance))
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let pool_len = candidates.len().min((count * 8).max(count));
    candidates.truncate(pool_len);
    if candidates.len() <= count {
        return candidates
            .into_iter()
            .map(|candidate| candidate.0)
            .collect();
    }

    let mut selected = vec![candidates[0]];
    while selected.len() < count {
        let next = *candidates
            .iter()
            .filter(|candidate| !selected.iter().any(|chosen| chosen.0 == candidate.0))
            .max_by(|a, b| {
                let a_dist = selected
                    .iter()
                    .map(|chosen| point_distance(&a.1, &chosen.1))
                    .fold(f64::INFINITY, f64::min);
                let b_dist = selected
                    .iter()
                    .map(|chosen| point_distance(&b.1, &chosen.1))
                    .fold(f64::INFINITY, f64::min);
                a_dist
                    .partial_cmp(&b_dist)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
                    .then_with(|| b.0.cmp(&a.0))
            })
            .expect("candidate pool should not be exhausted");
        selected.push(next);
    }
    selected.into_iter().map(|candidate| candidate.0).collect()
}

fn cycle_edge_set(cycle_observation_matrix: &FeecCsr) -> BTreeSet<usize> {
    cycle_observation_matrix
        .triplet_iter()
        .filter_map(|(_, col, value)| (value.abs() > EPS).then_some(col))
        .collect()
}

fn edge_midpoint(topology: &Complex, coords: &MeshCoords, edge_index: usize) -> [f64; 3] {
    let edge = topology.edges().handle_by_kidx(edge_index);
    let a = coords.coord(edge.vertices[0]);
    let b = coords.coord(edge.vertices[1]);
    [
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        if coords.dim() > 2 {
            0.5 * (a[2] + b[2])
        } else {
            0.0
        },
    ]
}

fn point_distance(lhs: &[f64; 3], rhs: &[f64; 3]) -> f64 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    let dz = lhs[2] - rhs[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn vector_magnitudes(vectors: &[[f64; 3]]) -> FeecVector {
    FeecVector::from_iterator(
        vectors.len(),
        vectors
            .iter()
            .map(|[x, y, z]| (x * x + y * y + z * z).sqrt()),
    )
}

fn components_to_vtk_vectors(components: &[FeecVector]) -> Vec<[f64; 3]> {
    let cell_count = components.first().map_or(0, FeecVector::len);
    (0..cell_count)
        .map(|cell_index| {
            [
                components
                    .first()
                    .map_or(0.0, |component| component[cell_index]),
                components
                    .get(1)
                    .map_or(0.0, |component| component[cell_index]),
                components
                    .get(2)
                    .map_or(0.0, |component| component[cell_index]),
            ]
        })
        .collect()
}

fn branch_period_variance_entries(
    fields: &Genus2JointPeriodVarianceFields,
) -> [(&'static str, &Genus2PeriodVarianceField); 4] {
    [
        ("total", &fields.total),
        ("exact", &fields.exact),
        ("coexact", &fields.coexact),
        ("harmonic", &fields.harmonic),
    ]
}

fn surface_variance_entries(
    fields: &Genus2JointSurfaceVectorVarianceFields,
) -> [(&'static str, &Genus2SurfaceVectorVarianceFields); 4] {
    [
        ("total", &fields.total),
        ("exact", &fields.exact),
        ("coexact", &fields.coexact),
        ("harmonic", &fields.harmonic),
    ]
}

fn sum_component_vectors(components: &[FeecVector], length: usize) -> FeecVector {
    let mut sum = FeecVector::zeros(length);
    for component in components {
        sum += component;
    }
    sum
}

fn ratio_vector(numerator: &FeecVector, denominator: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(
        numerator.len(),
        numerator
            .iter()
            .zip(denominator.iter())
            .map(|(num, den)| safe_ratio(*num, *den)),
    )
}

fn vector_sum(values: &FeecVector) -> f64 {
    values.iter().sum()
}

fn vector_mean(values: &FeecVector) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        vector_sum(values) / values.len() as f64
    }
}

fn vector_min(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn vector_max(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn mean_iter(values: impl Iterator<Item = f64>) -> f64 {
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

fn surface_vector_field_seed_offset(kind: SurfaceVectorFieldKind) -> u64 {
    match kind {
        SurfaceVectorFieldKind::Total => 0,
        SurfaceVectorFieldKind::Exact => 1,
        SurfaceVectorFieldKind::Coexact => 2,
        SurfaceVectorFieldKind::Harmonic => 3,
    }
}

fn exact_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &FeecCsr,
) -> Result<FeecVector, Box<dyn Error>> {
    let rows = (0..operator.nrows())
        .map(|row| sparse_row_entries(operator, row))
        .collect::<Vec<_>>();
    let mut variances = FeecVector::zeros(operator.nrows());
    for (row_index, row) in rows.iter().enumerate() {
        let mut rhs = GmrfVector::zeros(operator.ncols());
        for (col, value) in row {
            rhs[*col] = *value;
        }
        let solved = gmrf.solve_precision(&rhs)?;
        let variance = row
            .iter()
            .map(|(col, value)| *value * solved[*col])
            .sum::<f64>();
        variances[row_index] = if variance > -1e-10 {
            variance.max(0.0)
        } else {
            return Err(format!("negative transformed variance {variance}").into());
        };
    }
    Ok(variances)
}

fn feec_vec_to_gmrf_local(vector: &FeecVector) -> GmrfVector {
    GmrfVector::from_vec(vector.iter().copied().collect())
}

fn feec_dense_to_gmrf_dense(matrix: &FeecMatrix) -> GmrfDenseMatrix {
    GmrfDenseMatrix::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

fn write_period_posterior_summary(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,cycle_index,cycle_name,observed_in_scenario,truth_period,posterior_mean,posterior_std,posterior_lower_95,posterior_upper_95,prior_variance,posterior_variance,variance_ratio,residual"
    )?;
    for scenario in &result.scenarios {
        for summary in scenario
            .period_summaries
            .iter()
            .chain(scenario.scalar_baseline.period_summaries.iter())
        {
            write_period_summary_row(&mut writer, summary)?;
        }
    }
    Ok(())
}

fn write_period_branch_variance_summary(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,branch,cycle_index,cycle_name,observed_in_scenario,prior_variance,posterior_variance,variance_ratio"
    )?;
    for scenario in &result.scenarios {
        for (branch, variance) in
            branch_period_variance_entries(&scenario.joint.period_variance_by_branch)
        {
            for (cycle_index, cycle) in result.cycle_observations.iter().enumerate() {
                writeln!(
                    writer,
                    "{},{},{},{},{},{:.12},{:.12},{:.12}",
                    scenario.scenario.as_str(),
                    branch,
                    cycle_index,
                    cycle.name,
                    cycle_index < scenario.scenario.loop_count(),
                    variance.prior_variance[cycle_index],
                    variance.posterior_variance[cycle_index],
                    variance.variance_ratio[cycle_index]
                )?;
            }
        }
    }
    Ok(())
}

fn write_surface_branch_variance_summary(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,branch,prior_trace_sum,posterior_trace_sum,prior_trace_mean,posterior_trace_mean,trace_ratio_mean,trace_ratio_min,trace_ratio_max"
    )?;
    for scenario in &result.scenarios {
        for (branch, variance) in surface_variance_entries(&scenario.joint.surface_vector_variance)
        {
            writeln!(
                writer,
                "{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                scenario.scenario.as_str(),
                branch,
                vector_sum(&variance.prior_trace),
                vector_sum(&variance.posterior_trace),
                vector_mean(&variance.prior_trace),
                vector_mean(&variance.posterior_trace),
                vector_mean(&variance.trace_ratio),
                vector_min(&variance.trace_ratio),
                vector_max(&variance.trace_ratio),
            )?;
        }
    }
    Ok(())
}

fn write_harmonic_dense_prior_summary(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,observation_count,loop_observation_count,cycle_index,cycle_name,observed_in_scenario,truth_period,posterior_mean,prior_variance,posterior_variance,variance_ratio"
    )?;
    for scenario in &result.harmonic_dense_prior {
        for summary in &scenario.period_summaries {
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12}",
                scenario.scenario.as_str(),
                summary.model,
                scenario.observation_count,
                scenario.scenario.loop_count(),
                summary.cycle_index,
                summary.cycle_name,
                summary.observed_in_scenario,
                summary.truth_period,
                summary.posterior_mean,
                summary.prior_variance,
                summary.posterior_variance,
                summary.variance_ratio
            )?;
        }
    }
    Ok(())
}

fn write_kappa0_constrained_period_summary(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "treatment,constraint_rank,cycle_index,cycle_name,prior_variance_policy,constrained_posterior_variance"
    )?;
    for (cycle_index, cycle) in result.cycle_observations.iter().enumerate() {
        writeln!(
            writer,
            "{},{},{},{},improper_harmonic_prior,{:.12}",
            result.kappa0_constrained_periods.treatment,
            result.kappa0_constrained_periods.constraint_rank,
            cycle_index,
            cycle.name,
            result
                .kappa0_constrained_periods
                .constrained_period_variance[cycle_index]
        )?;
    }
    Ok(())
}

fn write_experiment4_summary(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "Experiment 4: topology and harmonic-mode test")?;
    writeln!(writer, "mesh_topology=genus2_closed_surface")?;
    writeln!(writer, "b0={}", result.topology_summary.b0)?;
    writeln!(writer, "b1={}", result.topology_summary.b1)?;
    writeln!(writer, "b2={}", result.topology_summary.b2)?;
    writeln!(
        writer,
        "cycle_harmonic_pairing_rank={}",
        result.cycle_harmonic_pairing.rank(1e-8)
    )?;
    writeln!(
        writer,
        "local_observation_count={}",
        result.local_observations.len()
    )?;
    for scenario in &result.scenarios {
        writeln!(
            writer,
            "joint_hodge_{}_mean_period_variance_ratio={:.12}",
            scenario.scenario.as_str(),
            mean_iter(
                scenario
                    .period_summaries
                    .iter()
                    .map(|summary| summary.variance_ratio)
            )
        )?;
    }
    for scenario in &result.harmonic_dense_prior {
        writeln!(
            writer,
            "dense_harmonic_{}_mean_period_variance_ratio={:.12}",
            scenario.scenario.as_str(),
            mean_iter(
                scenario
                    .period_summaries
                    .iter()
                    .map(|summary| summary.variance_ratio)
            )
        )?;
    }
    writeln!(
        writer,
        "kappa0_treatment={}",
        result.kappa0_constrained_periods.treatment
    )?;
    writeln!(
        writer,
        "kappa0_constrained_max_period_variance={:.12}",
        vector_max(
            &result
                .kappa0_constrained_periods
                .constrained_period_variance
        )
    )?;
    Ok(())
}

fn write_sensitivity_summary(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,loop_observation_count,cycle_index,cycle_name,observed_in_scenario,prior_variance,posterior_variance,variance_ratio"
    )?;
    for scenario in &result.scenarios {
        for summary in &scenario.period_summaries {
            writeln!(
                writer,
                "{},{},{},{},{},{:.12},{:.12},{:.12}",
                scenario.scenario.as_str(),
                scenario.scenario.loop_count(),
                summary.cycle_index,
                summary.cycle_name,
                summary.observed_in_scenario,
                summary.prior_variance,
                summary.posterior_variance,
                summary.variance_ratio
            )?;
        }
    }
    Ok(())
}

fn write_period_summary_row(
    writer: &mut impl Write,
    summary: &Genus2PeriodSummary,
) -> io::Result<()> {
    writeln!(
        writer,
        "{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
        summary.scenario.as_str(),
        summary.model,
        summary.cycle_index,
        summary.cycle_name,
        summary.observed_in_scenario,
        summary.truth_period,
        summary.posterior_mean,
        summary.posterior_std,
        summary.posterior_lower_95,
        summary.posterior_upper_95,
        summary.prior_variance,
        summary.posterior_variance,
        summary.variance_ratio,
        summary.residual
    )
}

fn write_scenario_outputs(
    result: &Genus2TopologicalInverseResult,
    scenario: &Genus2ScenarioResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let scenario_dir = out_dir.join("scenarios").join(scenario.scenario.as_str());
    let joint_dir = scenario_dir.join("joint");
    fs::create_dir_all(&joint_dir)?;
    write_cycle_paths_vtu(result, scenario, scenario_dir.join("cycle_paths.vtu"))?;
    write_joint_period_summary(scenario, scenario_dir.join("joint_period_summary.csv"))?;
    write_scalar_baseline_summary(
        scenario,
        scenario_dir.join("scalar_potential_baseline_summary.csv"),
    )?;
    write_surface_vector(
        result,
        &scenario.joint.posterior_mean_total,
        "total_posterior_mean_surface_vector",
        Some(&scenario.joint.surface_vector_variance.total),
        joint_dir.join("total_posterior_mean_surface_vector.vtu"),
    )?;
    write_surface_vector(
        result,
        &scenario.joint.posterior_mean_exact,
        "exact_posterior_mean_surface_vector",
        Some(&scenario.joint.surface_vector_variance.exact),
        joint_dir.join("exact_posterior_mean_surface_vector.vtu"),
    )?;
    write_surface_vector(
        result,
        &scenario.joint.posterior_mean_coexact,
        "coexact_posterior_mean_surface_vector",
        Some(&scenario.joint.surface_vector_variance.coexact),
        joint_dir.join("coexact_posterior_mean_surface_vector.vtu"),
    )?;
    write_surface_vector(
        result,
        &scenario.joint.posterior_mean_harmonic,
        "harmonic_posterior_mean_surface_vector",
        Some(&scenario.joint.surface_vector_variance.harmonic),
        joint_dir.join("harmonic_posterior_mean_surface_vector.vtu"),
    )?;
    for (sample_index, sample) in scenario.joint.posterior_samples.iter().enumerate() {
        write_surface_vector(
            result,
            &sample.exact,
            "exact_posterior_sample_surface_vector",
            None,
            joint_dir.join(format!("exact_posterior_sample_{sample_index}.vtu")),
        )?;
        write_surface_vector(
            result,
            &sample.coexact,
            "coexact_posterior_sample_surface_vector",
            None,
            joint_dir.join(format!("coexact_posterior_sample_{sample_index}.vtu")),
        )?;
        write_surface_vector(
            result,
            &sample.harmonic,
            "harmonic_posterior_sample_surface_vector",
            None,
            joint_dir.join(format!("harmonic_posterior_sample_{sample_index}.vtu")),
        )?;
    }
    Ok(())
}

fn write_surface_vector(
    result: &Genus2TopologicalInverseResult,
    field: &FeecVector,
    vector_name: &str,
    variance: Option<&Genus2SurfaceVectorVarianceFields>,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let cochain = Cochain::new(1, field.clone());
    let vectors = sample_1form_cell_vectors(&result.coords, &result.topology, &cochain)?;
    let mut vector_fields = vec![(vector_name, vectors.as_slice())];
    let mut scalar_storage = Vec::<(&'static str, FeecVector)>::new();
    let posterior_variance_vectors = variance.map(|variance| variance.posterior_vtk_vectors());
    let prior_variance_vectors = variance.map(|variance| variance.prior_vtk_vectors());
    if let Some(variance) = variance {
        scalar_storage.push(("magnitude", vector_magnitudes(&vectors)));
        scalar_storage.push(("marginal_variance", variance.posterior_trace.clone()));
        scalar_storage.push((
            "marginal_std",
            variance.posterior_trace.map(|value| value.max(0.0).sqrt()),
        ));
        scalar_storage.push(("prior_marginal_variance", variance.prior_trace.clone()));
        scalar_storage.push(("marginal_variance_ratio", variance.trace_ratio.clone()));
    }
    if let Some(posterior_variance_vectors) = posterior_variance_vectors.as_ref() {
        vector_fields.push((
            "posterior_directional_variance",
            posterior_variance_vectors.as_slice(),
        ));
    }
    if let Some(prior_variance_vectors) = prior_variance_vectors.as_ref() {
        vector_fields.push((
            "prior_directional_variance",
            prior_variance_vectors.as_slice(),
        ));
    }
    let scalar_fields = scalar_storage
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect::<Vec<_>>();
    visual_output::write_top_cell_fields(
        path,
        &result.coords,
        &result.topology,
        vector_fields.as_slice(),
        scalar_fields.as_slice(),
    )?;
    Ok(())
}

fn write_joint_period_summary(
    scenario: &Genus2ScenarioResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,cycle_index,cycle_name,observed_in_scenario,truth_period,posterior_mean,posterior_std,posterior_lower_95,posterior_upper_95,prior_variance,posterior_variance,variance_ratio,residual"
    )?;
    for summary in &scenario.period_summaries {
        write_period_summary_row(&mut writer, summary)?;
    }
    Ok(())
}

fn write_scalar_baseline_summary(
    scenario: &Genus2ScenarioResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "scenario,model,cycle_index,cycle_name,observed_in_scenario,truth_period,posterior_mean,posterior_std,posterior_lower_95,posterior_upper_95,prior_variance,posterior_variance,variance_ratio,residual"
    )?;
    for summary in &scenario.scalar_baseline.period_summaries {
        write_period_summary_row(&mut writer, summary)?;
    }
    Ok(())
}

fn write_cycle_paths_vtu(
    result: &Genus2TopologicalInverseResult,
    scenario: &Genus2ScenarioResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let path_refs = result
        .cycle_observations
        .iter()
        .map(|cycle| cycle.path_vertices.as_slice())
        .collect::<Vec<_>>();
    let cycle_index = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.cycle_index as f64)
        .collect::<Vec<_>>();
    let observed_in_scenario = scenario
        .period_summaries
        .iter()
        .map(|summary| {
            if summary.observed_in_scenario {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let truth_period = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.truth_period)
        .collect::<Vec<_>>();
    let joint_posterior_mean = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.posterior_mean)
        .collect::<Vec<_>>();
    let joint_posterior_std = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.posterior_std)
        .collect::<Vec<_>>();
    let joint_posterior_lower_95 = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.posterior_lower_95)
        .collect::<Vec<_>>();
    let joint_posterior_upper_95 = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.posterior_upper_95)
        .collect::<Vec<_>>();
    let joint_prior_variance = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.prior_variance)
        .collect::<Vec<_>>();
    let joint_posterior_variance = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.posterior_variance)
        .collect::<Vec<_>>();
    let joint_posterior_prior_variance_ratio = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.variance_ratio)
        .collect::<Vec<_>>();
    let joint_variance_ratio = scenario
        .period_summaries
        .iter()
        .map(|summary| summary.variance_ratio)
        .collect::<Vec<_>>();

    visual_output::write_polyline_fields(
        path,
        "genus-2 topological inverse cycle paths",
        &result.coords,
        &path_refs,
        &[
            ("cycle_index", cycle_index.as_slice()),
            ("observed_in_scenario", observed_in_scenario.as_slice()),
            ("truth_period", truth_period.as_slice()),
            ("joint_posterior_mean", joint_posterior_mean.as_slice()),
            ("joint_posterior_std", joint_posterior_std.as_slice()),
            (
                "joint_posterior_lower_95",
                joint_posterior_lower_95.as_slice(),
            ),
            (
                "joint_posterior_upper_95",
                joint_posterior_upper_95.as_slice(),
            ),
            ("joint_prior_variance", joint_prior_variance.as_slice()),
            (
                "joint_posterior_variance",
                joint_posterior_variance.as_slice(),
            ),
            (
                "joint_posterior_prior_variance_ratio",
                joint_posterior_prior_variance_ratio.as_slice(),
            ),
            ("joint_variance_ratio", joint_variance_ratio.as_slice()),
        ],
    )
}

fn write_local_observations_vtu(
    result: &Genus2TopologicalInverseResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let paths = result
        .local_observations
        .iter()
        .map(|observation| {
            result
                .topology
                .edges()
                .handle_by_kidx(observation.edge_index)
                .vertices
                .to_vec()
        })
        .collect::<Vec<_>>();
    let path_refs = paths.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let observation_index = result
        .local_observations
        .iter()
        .map(|observation| observation.observation_index as f64)
        .collect::<Vec<_>>();
    let edge_index = result
        .local_observations
        .iter()
        .map(|observation| observation.edge_index as f64)
        .collect::<Vec<_>>();
    let observed_value = result
        .local_observations
        .iter()
        .map(|observation| observation.observed_value)
        .collect::<Vec<_>>();
    let noise_std = result
        .local_observations
        .iter()
        .map(|observation| observation.noise_std)
        .collect::<Vec<_>>();

    visual_output::write_polyline_fields(
        path,
        "genus-2 local tangent observations",
        &result.coords,
        &path_refs,
        &[
            ("observation_index", observation_index.as_slice()),
            ("edge_index", edge_index.as_slice()),
            ("observed_value", observed_value.as_slice()),
            ("noise_std", noise_std.as_slice()),
        ],
    )
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= EPS {
        0.0
    } else {
        numerator / denominator
    }
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_string())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn genus2_topological_inverse_outputs_topological_period_story() {
        let _lock = lock_feec_harmonic_tests();
        let mut config = Genus2TopologicalInverseConfig::default();
        config.posterior_sample_count = 1;
        config.surface_vector_variance_probe_count = 8;

        let result = run_genus2_topological_inverse_problem(&config)
            .expect("genus-2 topological inverse problem should run");

        assert_eq!(result.topology_summary.b0, 1);
        assert_eq!(result.topology_summary.b1, 4);
        assert_eq!(result.topology_summary.b2, 1);
        assert_eq!(result.cycle_harmonic_pairing.rank(1e-8), 4);
        assert_eq!(
            result.local_observations.len(),
            config.local_observation_count
        );
        let cycle_edges = cycle_edge_set(&result.cycle_observation_matrix);
        let mut local_edges = BTreeSet::new();
        for observation in &result.local_observations {
            assert!(!cycle_edges.contains(&observation.edge_index));
            assert!(local_edges.insert(observation.edge_index));
            assert!(observation.observed_value.is_finite());
        }
        for cycle in &result.cycle_observations {
            assert!(cycle.closure_residual_l1 <= 1e-10);
        }

        let local_only = scenario_result(&result, Genus2ObservationScenario::LocalOnly);
        let all_loops = scenario_result(&result, Genus2ObservationScenario::LocalPlusAll4Loops);
        let scenario_sequence = [
            Genus2ObservationScenario::LocalOnly,
            Genus2ObservationScenario::LocalPlus1Loop,
            Genus2ObservationScenario::LocalPlus2Loops,
            Genus2ObservationScenario::LocalPlusAll4Loops,
        ];
        for observed_cycle_index in 0..result.cycle_observations.len() {
            let mut previous_variance =
                local_only.joint.period_posterior_variance[observed_cycle_index];
            for scenario in scenario_sequence {
                if scenario.loop_count() < observed_cycle_index + 1 {
                    continue;
                }
                let variance = scenario_result(&result, scenario)
                    .joint
                    .period_posterior_variance[observed_cycle_index];
                assert!(
                    variance <= previous_variance + 1e-10,
                    "observed period variance should decrease monotonically for cycle {observed_cycle_index}"
                );
                previous_variance = variance;
            }
        }
        for cycle_index in 0..result.cycle_observations.len() {
            assert!(
                all_loops.joint.period_posterior_variance[cycle_index]
                    <= local_only.joint.period_posterior_variance[cycle_index] + 1e-10
            );
            assert!(
                all_loops.joint.period_posterior_variance[cycle_index]
                    <= 0.1 * local_only.joint.period_posterior_variance[cycle_index].max(EPS),
                "all loop observations should strongly reduce period variance for cycle {cycle_index}"
            );
        }
        for scenario in &result.scenarios {
            for (_, branch) in
                branch_period_variance_entries(&scenario.joint.period_variance_by_branch)
            {
                assert_eq!(branch.prior_variance.len(), result.cycle_observations.len());
                assert_eq!(
                    branch.posterior_variance.len(),
                    result.cycle_observations.len()
                );
                assert!(branch.prior_variance.iter().all(|value| value.is_finite()));
                assert!(branch
                    .posterior_variance
                    .iter()
                    .all(|value| value.is_finite()));
                assert!(branch.variance_ratio.iter().all(|value| value.is_finite()));
            }
        }
        assert!(
            vector_sum(
                &local_only
                    .joint
                    .period_variance_by_branch
                    .harmonic
                    .posterior_variance
            ) > EPS,
            "local observations should leave finite harmonic period uncertainty"
        );

        let dense_local = harmonic_dense_result(&result, Genus2ObservationScenario::LocalOnly);
        let dense_all =
            harmonic_dense_result(&result, Genus2ObservationScenario::LocalPlusAll4Loops);
        for cycle_index in 0..result.cycle_observations.len() {
            assert!(
                dense_all.period_summaries[cycle_index].posterior_variance
                    <= 0.01
                        * dense_local.period_summaries[cycle_index]
                            .posterior_variance
                            .max(EPS),
                "dense harmonic period observations should strongly reduce cycle {cycle_index}"
            );
        }
        assert_eq!(result.kappa0_constrained_periods.constraint_rank, 4);
        assert!(
            vector_max(
                &result
                    .kappa0_constrained_periods
                    .constrained_period_variance
            ) <= 1e-10,
            "fully constrained kappa=0 harmonic periods should have zero constrained variance"
        );
        for scenario in &result.scenarios {
            for summary in &scenario.scalar_baseline.period_summaries {
                assert_eq!(summary.posterior_mean, 0.0);
                assert_eq!(summary.posterior_variance, 0.0);
            }
        }
        assert!(all_loops
            .scalar_baseline
            .period_summaries
            .iter()
            .any(|summary| summary.residual.abs() > 1e-3));

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!("genus2_topological_inverse_{stamp}"));
        write_genus2_topological_inverse_outputs(&result, &out_dir).expect("outputs should write");
        assert!(out_dir.join("experiment4_summary.txt").is_file());
        assert!(out_dir.join("period_posterior_summary.csv").is_file());
        assert!(out_dir.join("period_branch_variance_summary.csv").is_file());
        assert!(out_dir
            .join("surface_branch_variance_summary.csv")
            .is_file());
        assert!(out_dir.join("harmonic_dense_prior_summary.csv").is_file());
        assert!(out_dir
            .join("kappa0_constrained_period_summary.csv")
            .is_file());
        assert!(out_dir.join("sensitivity_summary.csv").is_file());
        assert!(out_dir.join("local_observations.vtu").is_file());
        let experiment4_summary = fs::read_to_string(out_dir.join("experiment4_summary.txt"))
            .expect("Experiment 4 summary should read");
        assert!(experiment4_summary.contains("joint_hodge_local_plus_all_4_loops"));
        assert!(experiment4_summary.contains("dense_harmonic_local_plus_all_4_loops"));
        assert!(experiment4_summary.contains("kappa0_treatment="));
        let branch_summary = fs::read_to_string(out_dir.join("period_branch_variance_summary.csv"))
            .expect("period branch summary should read");
        assert!(branch_summary.contains("scenario,branch,cycle_index"));
        assert!(branch_summary.contains("local_plus_all_4_loops,harmonic"));
        let dense_summary = fs::read_to_string(out_dir.join("harmonic_dense_prior_summary.csv"))
            .expect("dense harmonic summary should read");
        assert!(dense_summary.contains("dense_harmonic"));
        let kappa0_summary =
            fs::read_to_string(out_dir.join("kappa0_constrained_period_summary.csv"))
                .expect("kappa0 constrained summary should read");
        assert!(kappa0_summary.contains("improper_harmonic_prior"));
        for scenario in Genus2ObservationScenario::all() {
            let scenario_dir = out_dir.join("scenarios").join(scenario.as_str());
            assert!(scenario_dir.join("cycle_paths.vtu").is_file());
            let joint_summary = scenario_dir.join("joint_period_summary.csv");
            assert!(joint_summary.is_file());
            let joint_summary_text =
                fs::read_to_string(&joint_summary).expect("joint period summary should read");
            assert!(joint_summary_text.contains("variance_ratio"));
            assert!(scenario_dir
                .join("scalar_potential_baseline_summary.csv")
                .is_file());
            let joint_dir = scenario_dir.join("joint");
            assert!(joint_dir
                .join("total_posterior_mean_surface_vector.vtu")
                .is_file());
            assert!(joint_dir
                .join("exact_posterior_mean_surface_vector.vtu")
                .is_file());
            assert!(joint_dir
                .join("coexact_posterior_mean_surface_vector.vtu")
                .is_file());
            assert!(joint_dir
                .join("harmonic_posterior_mean_surface_vector.vtu")
                .is_file());
            assert!(joint_dir.join("exact_posterior_sample_0.vtu").is_file());
            assert!(joint_dir.join("coexact_posterior_sample_0.vtu").is_file());
            assert!(joint_dir.join("harmonic_posterior_sample_0.vtu").is_file());
        }
        let total_surface = fs::read_to_string(
            out_dir
                .join("scenarios")
                .join(Genus2ObservationScenario::LocalPlusAll4Loops.as_str())
                .join("joint")
                .join("total_posterior_mean_surface_vector.vtu"),
        )
        .expect("surface vector output should read");
        assert!(total_surface.contains("Name=\"posterior_directional_variance\""));
        assert!(total_surface.contains("Name=\"prior_directional_variance\""));
        assert!(total_surface.contains("Name=\"marginal_variance\""));
        assert!(total_surface.contains("Name=\"prior_marginal_variance\""));
        assert!(total_surface.contains("Name=\"marginal_variance_ratio\""));
        let cycle_paths = fs::read_to_string(
            out_dir
                .join("scenarios")
                .join(Genus2ObservationScenario::LocalPlusAll4Loops.as_str())
                .join("cycle_paths.vtu"),
        )
        .expect("cycle paths should read");
        assert!(cycle_paths.contains("Name=\"joint_posterior_lower_95\""));
        assert!(cycle_paths.contains("Name=\"joint_posterior_upper_95\""));
        assert!(cycle_paths.contains("Name=\"joint_prior_variance\""));
        assert!(cycle_paths.contains("Name=\"joint_posterior_variance\""));
        assert!(cycle_paths.contains("Name=\"joint_posterior_prior_variance_ratio\""));
        fs::remove_dir_all(out_dir).expect("temporary output directory should clean up");
    }

    fn scenario_result(
        result: &Genus2TopologicalInverseResult,
        scenario: Genus2ObservationScenario,
    ) -> &Genus2ScenarioResult {
        result
            .scenarios
            .iter()
            .find(|entry| entry.scenario == scenario)
            .expect("scenario should be present")
    }

    fn harmonic_dense_result(
        result: &Genus2TopologicalInverseResult,
        scenario: Genus2ObservationScenario,
    ) -> &Genus2HarmonicDensePriorScenarioResult {
        result
            .harmonic_dense_prior
            .iter()
            .find(|entry| entry.scenario == scenario)
            .expect("dense harmonic scenario should be present")
    }
}
