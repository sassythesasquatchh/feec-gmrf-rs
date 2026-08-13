use feg_core::{GaussianPriorSpec, SparseTripletMatrix};
use feg_infer::adapters::FeecResidualAdapter;
use feg_infer::linear_pde::{
    LinearPdeDerivedQuantitySpec, LinearPdeVarianceConfig, LinearPdeVarianceMode,
};
use feg_infer::nonlinear::{
    solve_nonlinear_laplace, GaussNewtonConfig, GaussianNoiseModel, NonlinearLaplaceProblem,
    NonlinearLaplaceResult, NonlinearResidualTerm,
};
use feg_infer::physical::build_reduced_magnetic_flux_density_operator_3d;
use feg_infer::prior::matern::one_form::{
    build_reduced_linear_proxy_matern_alpha2_prior, ReducedLinearProxyMaternAlpha2Config,
};
use feg_infer::sparse::core_triplet_to_gmrf as sparse_from_core;
use feg_infer::sparse::{feec_csr_to_core_triplet, sparse_row_operator_to_triplet};
use formoniq::{
    problems::{
        eddy_current::{
            assemble_reduced_screened_eddy_current_sinusoidal_source,
            build_reduced_nonlinear_screened_eddy_current_1form, LocalEddyCurrentResidualProbe3d,
            NonlinearEddyCurrentReluctivityLaw, NonlinearScreenedEddyCurrentAssemblyConfig,
            ReducedNonlinearScreenedEddyCurrent1Form,
        },
        residual::ResidualModel as FeecResidualModel,
    },
    reduction::{EssentialBoundarySpec, PrescribedDof},
};
use gmrf_core::{SparseRowOperator, Vector as GmrfVector};
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::mesh::MeshCoords,
    topology::complex::Complex,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EddyCurrentPriorMode {
    WeakDiagonal,
    LinearProxyMaternAlpha2,
}

impl EddyCurrentPriorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WeakDiagonal => "weak_diagonal",
            Self::LinearProxyMaternAlpha2 => "linear_proxy_matern_alpha2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EddyCurrentObservationMode {
    PriorOnly,
    WeakFull,
    LocalFull,
    LocalUniform,
    LocalLeverage,
}

impl EddyCurrentObservationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PriorOnly => "prior_only",
            Self::WeakFull => "weak_full",
            Self::LocalFull => "local_full",
            Self::LocalUniform => "local_uniform",
            Self::LocalLeverage => "local_leverage",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearEddyCurrentComparisonConfig {
    pub mesh_level: usize,
    pub nu0: f64,
    pub beta: f64,
    pub sigma: f64,
    pub source_amplitude: f64,
    pub weak_diagonal_precision: f64,
    pub linear_proxy_tau: f64,
    pub residual_variance: f64,
    pub probe_variance: Option<f64>,
    pub probe_noise_relative_scale: f64,
    pub budget_fractions: Vec<f64>,
    pub uniform_seed: u64,
    pub max_iterations: usize,
    pub ngsolve_python: PathBuf,
    pub ngsolve_src: PathBuf,
    pub ngsolve_output_dir: PathBuf,
    pub ngsolve_order: usize,
    pub ngsolve_curve_order: usize,
    pub ngsolve_maxh: f64,
    pub ngsolve_newton_tolerance: f64,
    pub ngsolve_max_iterations: usize,
    pub ngsolve_write_vtu: bool,
    pub check_reference_adequacy: bool,
    pub adequacy_ngsolve_order: usize,
    pub adequacy_ngsolve_maxh: f64,
}

impl Default for NonlinearEddyCurrentComparisonConfig {
    fn default() -> Self {
        let ngsolve_root = default_ngsolve_root();
        Self {
            mesh_level: 2,
            nu0: 1.0,
            beta: 0.75,
            sigma: 0.5,
            source_amplitude: 1.0,
            weak_diagonal_precision: 1e-10,
            linear_proxy_tau: 1.0,
            residual_variance: 1e-8,
            probe_variance: None,
            probe_noise_relative_scale: 1e-2,
            budget_fractions: vec![
                0.0,
                1.0 / 64.0,
                1.0 / 32.0,
                1.0 / 16.0,
                1.0 / 8.0,
                1.0 / 4.0,
                1.0 / 2.0,
                1.0,
            ],
            uniform_seed: 7,
            max_iterations: 25,
            ngsolve_python: ngsolve_root.join(".venv/bin/python"),
            ngsolve_src: ngsolve_root.join("src"),
            ngsolve_output_dir: PathBuf::from("target/nonlinear_eddy_current_ngsolve_reference"),
            ngsolve_order: 3,
            ngsolve_curve_order: 1,
            ngsolve_maxh: 0.25,
            ngsolve_newton_tolerance: 1e-10,
            ngsolve_max_iterations: 30,
            ngsolve_write_vtu: false,
            check_reference_adequacy: false,
            adequacy_ngsolve_order: 4,
            adequacy_ngsolve_maxh: 0.18,
        }
    }
}

fn default_ngsolve_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../ngsolve")
}

#[derive(Debug, Clone)]
pub struct EddyCurrentReferenceReport {
    pub active_dofs: usize,
    pub cells: usize,
    pub boundary_edges: usize,
    pub source_norm: f64,
    pub linear_mean_norm: f64,
    pub calibrated_probe_variance: f64,
    pub ngsolve_order: usize,
    pub ngsolve_maxh: f64,
    pub ngsolve_converged: bool,
    pub ngsolve_iterations: usize,
    pub ngsolve_sample_count: usize,
    pub ngsolve_cell_b_norm: f64,
    pub weak_reference_residual_norm: f64,
    pub weak_reference_iterations: usize,
    pub weak_reference_converged: bool,
    pub weak_reference_cell_b_relative_error: f64,
    pub reference_adequacy_cell_b_relative_error: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct EddyCurrentComparisonRow {
    pub prior_mode: EddyCurrentPriorMode,
    pub observation_mode: EddyCurrentObservationMode,
    pub probe_count: usize,
    pub total_cells: usize,
    pub residual_rows: usize,
    pub final_weak_residual_norm: f64,
    pub ngsolve_cell_a_relative_error: f64,
    pub ngsolve_cell_b_relative_error: f64,
    pub ngsolve_sensor_b_rmse: f64,
    pub ngsolve_sensor_b_relative_rmse: f64,
    pub residual_probe_weighted_norm: f64,
    pub iterations: usize,
    pub damping_count: usize,
    pub posterior_factorizes: bool,
    pub final_factor_nnz: usize,
    pub selected_b_variance_min: f64,
    pub selected_b_variance_max: f64,
    pub sensor_standardized_error_max: f64,
    pub sensor_coverage_2sigma: f64,
    pub success: bool,
    pub failure: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NonlinearEddyCurrentComparisonReport {
    pub config: NonlinearEddyCurrentComparisonConfig,
    pub reference: EddyCurrentReferenceReport,
    pub rows: Vec<EddyCurrentComparisonRow>,
}

struct EddyCurrentSetup {
    topology: Complex,
    coords: MeshCoords,
    model: ReducedNonlinearScreenedEddyCurrent1Form,
    source: Vec<f64>,
    linear_mean: Vec<f64>,
    a_operator: SparseTripletMatrix,
    b_operator: SparseTripletMatrix,
    leverage_scores: Vec<f64>,
    reference: NgsolveReferenceSamples,
    sensor_cells: Vec<usize>,
    calibrated_probe_variance: f64,
    reference_adequacy_cell_b_relative_error: Option<f64>,
    boundary_edges: usize,
}

#[derive(Debug, Clone)]
struct NgsolveRunSummary {
    converged: bool,
    newton_iterations: usize,
    sample_count: usize,
}

#[derive(Debug, Clone)]
struct NgsolveReferenceSamples {
    cell_a: Vec<[f64; 3]>,
    cell_b: Vec<[f64; 3]>,
    sensor_b: Vec<[f64; 3]>,
    summary: NgsolveRunSummary,
}

pub fn run_nonlinear_eddy_current_weiland_comparison_experiment(
    config: NonlinearEddyCurrentComparisonConfig,
) -> Result<NonlinearEddyCurrentComparisonReport, String> {
    validate_config(&config)?;
    let setup = build_setup(&config)?;
    let physical_prior = build_prior(
        &setup,
        EddyCurrentPriorMode::LinearProxyMaternAlpha2,
        &config,
    )?;
    let weak_reference = solve_with_weak_residual(
        &setup.model,
        physical_prior.clone(),
        &setup.linear_mean,
        config.residual_variance,
        config.max_iterations,
        &setup.b_operator,
        &setup.sensor_cells,
    )?;
    let reference_map = weak_reference.map.clone();
    let reference_b = apply_triplet(&setup.b_operator, &reference_map)?;
    let weak_reference_cell_b_relative_error =
        relative_error_flat(&reference_b, &flatten3(&setup.reference.cell_b));
    let weak_reference_residual_norm = weak_residual_norm(&setup.model, &reference_map)?;

    let reference = EddyCurrentReferenceReport {
        active_dofs: setup.model.reduced_dimension(),
        cells: setup.model.num_elements(),
        boundary_edges: setup.boundary_edges,
        source_norm: l2_norm(&setup.source),
        linear_mean_norm: l2_norm(&setup.linear_mean),
        calibrated_probe_variance: setup.calibrated_probe_variance,
        ngsolve_order: config.ngsolve_order,
        ngsolve_maxh: config.ngsolve_maxh,
        ngsolve_converged: setup.reference.summary.converged,
        ngsolve_iterations: setup.reference.summary.newton_iterations,
        ngsolve_sample_count: setup.reference.summary.sample_count,
        ngsolve_cell_b_norm: l2_norm(&flatten3(&setup.reference.cell_b)),
        weak_reference_residual_norm,
        weak_reference_iterations: weak_reference.history.len(),
        weak_reference_converged: weak_reference.converged,
        weak_reference_cell_b_relative_error,
        reference_adequacy_cell_b_relative_error: setup.reference_adequacy_cell_b_relative_error,
    };

    let mut rows = Vec::new();
    for prior_mode in [
        EddyCurrentPriorMode::WeakDiagonal,
        EddyCurrentPriorMode::LinearProxyMaternAlpha2,
    ] {
        let prior = build_prior(&setup, prior_mode, &config)?;
        rows.push(run_row(
            &setup,
            prior_mode,
            EddyCurrentObservationMode::PriorOnly,
            0,
            prior.clone(),
            &reference_map,
            &config,
        ));

        let full_probe_count = setup.model.num_elements();
        rows.push(run_row(
            &setup,
            prior_mode,
            EddyCurrentObservationMode::LocalFull,
            full_probe_count,
            prior.clone(),
            &reference_map,
            &config,
        ));

        for count in budget_counts(&config.budget_fractions, setup.model.num_elements()) {
            if count == 0 || count == setup.model.num_elements() {
                continue;
            }
            rows.push(run_row(
                &setup,
                prior_mode,
                EddyCurrentObservationMode::LocalLeverage,
                count,
                prior.clone(),
                &reference_map,
                &config,
            ));
            rows.push(run_row(
                &setup,
                prior_mode,
                EddyCurrentObservationMode::LocalUniform,
                count,
                prior.clone(),
                &reference_map,
                &config,
            ));
        }
    }

    Ok(NonlinearEddyCurrentComparisonReport {
        config,
        reference,
        rows,
    })
}

fn build_setup(config: &NonlinearEddyCurrentComparisonConfig) -> Result<EddyCurrentSetup, String> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, config.mesh_level, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let boundary_edges = topology
        .boundary_subcomplex_simplices(1)
        .into_iter()
        .map(|simplex| simplex.kidx)
        .collect::<Vec<_>>();
    let boundary = EssentialBoundarySpec {
        state: boundary_edges
            .iter()
            .copied()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
        auxiliary: Vec::new(),
    };
    let source_free = build_reduced_nonlinear_screened_eddy_current_1form(
        &topology,
        &coords,
        None,
        NonlinearScreenedEddyCurrentAssemblyConfig::new(
            NonlinearEddyCurrentReluctivityLaw::new(config.nu0, config.beta)?,
            config.sigma,
            boundary.clone(),
        ),
    )?;
    if source_free.reduced_dimension() == 0 {
        return Err(
            "screened eddy-current comparison needs at least one active interior edge".to_string(),
        );
    }
    let source = assemble_reduced_screened_eddy_current_sinusoidal_source(
        &topology,
        &coords,
        None,
        &boundary,
        config.source_amplitude,
    )?;
    let model = source_free.with_source(source.clone())?;
    let beta_zero_model = build_reduced_nonlinear_screened_eddy_current_1form(
        &topology,
        &coords,
        None,
        NonlinearScreenedEddyCurrentAssemblyConfig::new(
            NonlinearEddyCurrentReluctivityLaw::new(config.nu0, 0.0)?,
            config.sigma,
            boundary,
        )
        .with_source(source.clone()),
    )?;
    let linear_rhs = source
        .iter()
        .zip(model.beta_zero_bias().iter())
        .map(|(source, bias)| source - bias)
        .collect::<Vec<_>>();
    let beta_zero_operator = feec_csr_to_core_triplet(model.beta_zero_operator());
    let linear_mean = solve_spd_triplet(&beta_zero_operator, &linear_rhs)?;
    let a_operator = feec_csr_to_core_triplet(&model.cell_vector_potential_operator());
    let b_operator = sparse_row_operator_to_triplet(
        &build_reduced_magnetic_flux_density_operator_3d(&topology, &coords, model.layout())?,
    )?;
    let leverage_scores = local_probe_leverage_scores(&beta_zero_model, &linear_mean)?;
    let sensor_cells = deterministic_sensor_cells(model.num_elements());
    let query_points = reference_query_points(&topology, &coords, &sensor_cells)?;
    let reference = run_ngsolve_reference(config, &query_points, &config.ngsolve_output_dir)?;
    let calibrated_probe_variance = match config.probe_variance {
        Some(variance) => variance,
        None => calibrate_probe_variance(&model, &linear_mean, config.probe_noise_relative_scale)?,
    };
    let reference_adequacy_cell_b_relative_error = if config.check_reference_adequacy {
        let adequacy_dir = config.ngsolve_output_dir.join("adequacy");
        let mut adequacy_config = config.clone();
        adequacy_config.ngsolve_order = config.adequacy_ngsolve_order;
        adequacy_config.ngsolve_maxh = config.adequacy_ngsolve_maxh;
        let adequacy = run_ngsolve_reference(&adequacy_config, &query_points, &adequacy_dir)?;
        Some(relative_error3(&adequacy.cell_b, &reference.cell_b))
    } else {
        None
    };

    Ok(EddyCurrentSetup {
        topology,
        coords,
        model,
        source,
        linear_mean,
        a_operator,
        b_operator,
        leverage_scores,
        reference,
        sensor_cells,
        calibrated_probe_variance,
        reference_adequacy_cell_b_relative_error,
        boundary_edges: boundary_edges.len(),
    })
}

fn build_prior(
    setup: &EddyCurrentSetup,
    prior_mode: EddyCurrentPriorMode,
    config: &NonlinearEddyCurrentComparisonConfig,
) -> Result<GaussianPriorSpec, String> {
    match prior_mode {
        EddyCurrentPriorMode::WeakDiagonal => Ok(diagonal_prior(
            setup.model.reduced_dimension(),
            vec![0.0; setup.model.reduced_dimension()],
            config.weak_diagonal_precision,
        )),
        EddyCurrentPriorMode::LinearProxyMaternAlpha2 => {
            let prior = build_reduced_linear_proxy_matern_alpha2_prior(
                &setup.topology,
                &setup.coords,
                setup.model.layout(),
                &feec_csr_to_core_triplet(setup.model.beta_zero_operator()),
                setup.linear_mean.clone(),
                ReducedLinearProxyMaternAlpha2Config {
                    kappa: 0.0,
                    tau: config.linear_proxy_tau,
                    allow_kappa_fallback: false,
                    ..ReducedLinearProxyMaternAlpha2Config::default()
                },
            )?;
            Ok(prior.spec)
        }
    }
}

fn run_row(
    setup: &EddyCurrentSetup,
    prior_mode: EddyCurrentPriorMode,
    observation_mode: EddyCurrentObservationMode,
    probe_count: usize,
    prior: GaussianPriorSpec,
    _reference_map: &[f64],
    config: &NonlinearEddyCurrentComparisonConfig,
) -> EddyCurrentComparisonRow {
    let total_cells = setup.model.num_elements();
    let result = match observation_mode {
        EddyCurrentObservationMode::PriorOnly => solve_prior_only(
            prior,
            config.max_iterations,
            &setup.b_operator,
            &setup.sensor_cells,
        ),
        EddyCurrentObservationMode::WeakFull => solve_with_weak_residual(
            &setup.model,
            prior,
            &setup.linear_mean,
            config.residual_variance,
            config.max_iterations,
            &setup.b_operator,
            &setup.sensor_cells,
        )
        .map(|result| (result, Vec::new(), 0.0)),
        EddyCurrentObservationMode::LocalFull
        | EddyCurrentObservationMode::LocalUniform
        | EddyCurrentObservationMode::LocalLeverage => {
            let selected_cells = select_cells(setup, observation_mode, probe_count, config);
            solve_with_local_probe(
                &setup.model,
                selected_cells,
                prior,
                &setup.linear_mean,
                setup.calibrated_probe_variance,
                config.max_iterations,
                &setup.b_operator,
                &setup.sensor_cells,
            )
        }
    };

    match result {
        Ok((result, selected_cells, probe_weighted_norm)) => {
            let map = result.map.clone();
            let b = apply_triplet(&setup.b_operator, &map).unwrap_or_default();
            let a = apply_triplet(&setup.a_operator, &map).unwrap_or_default();
            let weak_residual_norm = weak_residual_norm(&setup.model, &map).unwrap_or(f64::NAN);
            let (variance_min, variance_max) = selected_b_variance_range(&result);
            let sensor_metrics =
                sensor_b_metrics(&b, &setup.reference.sensor_b, &setup.sensor_cells, &result);
            EddyCurrentComparisonRow {
                prior_mode,
                observation_mode,
                probe_count: selected_cells.len(),
                total_cells,
                residual_rows: 3 * selected_cells.len(),
                final_weak_residual_norm: weak_residual_norm,
                ngsolve_cell_a_relative_error: relative_error_flat(
                    &a,
                    &flatten3(&setup.reference.cell_a),
                ),
                ngsolve_cell_b_relative_error: relative_error_flat(
                    &b,
                    &flatten3(&setup.reference.cell_b),
                ),
                ngsolve_sensor_b_rmse: sensor_metrics.rmse,
                ngsolve_sensor_b_relative_rmse: sensor_metrics.relative_rmse,
                residual_probe_weighted_norm: probe_weighted_norm,
                iterations: result.history.len(),
                damping_count: result
                    .history
                    .iter()
                    .filter(|entry| entry.alpha < 1.0 - 1e-12)
                    .count(),
                posterior_factorizes: sparse_from_core(&result.posterior_precision)
                    .cholesky_sqrt_lower()
                    .is_ok(),
                final_factor_nnz: result.final_factorization.nnz,
                selected_b_variance_min: variance_min,
                selected_b_variance_max: variance_max,
                sensor_standardized_error_max: sensor_metrics.max_standardized_error,
                sensor_coverage_2sigma: sensor_metrics.coverage_2sigma,
                success: true,
                failure: None,
            }
        }
        Err(error) => EddyCurrentComparisonRow {
            prior_mode,
            observation_mode,
            probe_count,
            total_cells,
            residual_rows: 3 * probe_count,
            final_weak_residual_norm: f64::NAN,
            ngsolve_cell_a_relative_error: f64::NAN,
            ngsolve_cell_b_relative_error: f64::NAN,
            ngsolve_sensor_b_rmse: f64::NAN,
            ngsolve_sensor_b_relative_rmse: f64::NAN,
            residual_probe_weighted_norm: f64::NAN,
            iterations: 0,
            damping_count: 0,
            posterior_factorizes: false,
            final_factor_nnz: 0,
            selected_b_variance_min: f64::NAN,
            selected_b_variance_max: f64::NAN,
            sensor_standardized_error_max: f64::NAN,
            sensor_coverage_2sigma: f64::NAN,
            success: false,
            failure: Some(error),
        },
    }
}

fn solve_prior_only(
    prior: GaussianPriorSpec,
    max_iterations: usize,
    b_operator: &SparseTripletMatrix,
    sensor_cells: &[usize],
) -> Result<(NonlinearLaplaceResult, Vec<usize>, f64), String> {
    let initial_guess = prior.mean.clone();
    let problem = NonlinearLaplaceProblem {
        prior,
        residual_terms: Vec::new(),
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: selected_b_derived_quantities(b_operator, sensor_cells)?,
    };
    let result = solve_nonlinear_laplace(
        &problem,
        &solver_config(Some(initial_guess), max_iterations),
    )?;
    Ok((result, Vec::new(), 0.0))
}

fn solve_with_weak_residual(
    model: &ReducedNonlinearScreenedEddyCurrent1Form,
    prior: GaussianPriorSpec,
    initial_guess: &[f64],
    variance: f64,
    max_iterations: usize,
    b_operator: &SparseTripletMatrix,
    sensor_cells: &[usize],
) -> Result<NonlinearLaplaceResult, String> {
    let adapter = FeecResidualAdapter::new(model);
    let problem = NonlinearLaplaceProblem {
        prior,
        residual_terms: vec![NonlinearResidualTerm::zero(
            "nonlinear_screened_eddy_current_weak",
            &adapter,
            GaussianNoiseModel::ScalarVariance(variance),
        )],
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: selected_b_derived_quantities(b_operator, sensor_cells)?,
    };
    solve_nonlinear_laplace(
        &problem,
        &solver_config(Some(initial_guess.to_vec()), max_iterations),
    )
}

fn solve_with_local_probe(
    model: &ReducedNonlinearScreenedEddyCurrent1Form,
    selected_cells: Vec<usize>,
    prior: GaussianPriorSpec,
    initial_guess: &[f64],
    variance: f64,
    max_iterations: usize,
    b_operator: &SparseTripletMatrix,
    sensor_cells: &[usize],
) -> Result<(NonlinearLaplaceResult, Vec<usize>, f64), String> {
    let probe = LocalEddyCurrentResidualProbe3d::from_model(model, selected_cells.clone())?;
    let probe_adapter = FeecResidualAdapter::new(&probe);
    let precision = local_probe_precision(model, &selected_cells, variance)?;
    let problem = NonlinearLaplaceProblem {
        prior,
        residual_terms: vec![NonlinearResidualTerm::zero(
            "nonlinear_screened_eddy_current_local_probe",
            &probe_adapter,
            GaussianNoiseModel::Precision(precision.clone()),
        )],
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: selected_b_derived_quantities(b_operator, sensor_cells)?,
    };
    let result = solve_nonlinear_laplace(
        &problem,
        &solver_config(Some(initial_guess.to_vec()), max_iterations),
    )?;
    let evaluation = probe.residual_and_jacobian(&result.map)?;
    let weighted_norm = weighted_norm_diag(evaluation.residual.as_slice(), &precision);
    Ok((result, selected_cells, weighted_norm))
}

fn solver_config(initial_guess: Option<Vec<f64>>, max_iterations: usize) -> GaussNewtonConfig {
    GaussNewtonConfig {
        initial_guess,
        max_iterations,
        step_tolerance: 1e-10,
        gradient_tolerance: 1e-9,
        variance: LinearPdeVarianceConfig {
            mode: LinearPdeVarianceMode::ExactSolves,
            ..LinearPdeVarianceConfig::default()
        },
        ..GaussNewtonConfig::default()
    }
}

fn selected_b_derived_quantities(
    b_operator: &SparseTripletMatrix,
    sensor_cells: &[usize],
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    if b_operator.nrows() == 0 {
        return Ok(Vec::new());
    }
    let cell_count = b_operator.nrows() / 3;
    let mut cells = sensor_cells
        .iter()
        .copied()
        .filter(|cell| *cell < cell_count)
        .collect::<Vec<_>>();
    if cells.is_empty() {
        cells = deterministic_sensor_cells(cell_count);
    }
    cells.sort_unstable();
    cells.dedup();
    let rows = sparse_rows(b_operator);
    let selected_rows = cells
        .into_iter()
        .flat_map(|cell| [3 * cell, 3 * cell + 1, 3 * cell + 2])
        .filter(|row| *row < rows.len())
        .map(|row| rows[row].clone())
        .collect::<Vec<_>>();
    Ok(vec![LinearPdeDerivedQuantitySpec {
        name: "selected_cell_B".to_string(),
        operator: SparseRowOperator::new(b_operator.ncols(), selected_rows)
            .map_err(|err| err.to_string())?,
    }])
}

fn selected_b_variance_range(result: &NonlinearLaplaceResult) -> (f64, f64) {
    let Some(variance) = result.derived_variances.get("selected_cell_B") else {
        return (f64::NAN, f64::NAN);
    };
    let min = variance
        .posterior_variance
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = variance
        .posterior_variance
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn local_probe_precision(
    _model: &ReducedNonlinearScreenedEddyCurrent1Form,
    selected_cells: &[usize],
    variance: f64,
) -> Result<SparseTripletMatrix, String> {
    if selected_cells.is_empty() {
        return Err("local probe precision requires at least one selected cell".to_string());
    }
    if !variance.is_finite() || variance <= 0.0 {
        return Err("local probe variance must be finite and positive".to_string());
    }
    let mut precision =
        SparseTripletMatrix::new(3 * selected_cells.len(), 3 * selected_cells.len());
    let weight = 1.0 / variance;
    for probe_index in 0..selected_cells.len() {
        for component in 0..3 {
            let row = 3 * probe_index + component;
            precision.push(row, row, weight);
        }
    }
    Ok(precision)
}

fn select_cells(
    setup: &EddyCurrentSetup,
    mode: EddyCurrentObservationMode,
    count: usize,
    config: &NonlinearEddyCurrentComparisonConfig,
) -> Vec<usize> {
    let total = setup.model.num_elements();
    let count = count.min(total);
    match mode {
        EddyCurrentObservationMode::LocalFull => (0..total).collect(),
        EddyCurrentObservationMode::LocalUniform => {
            let mut cells = (0..total).collect::<Vec<_>>();
            let mut rng = StdRng::seed_from_u64(config.uniform_seed + count as u64);
            cells.shuffle(&mut rng);
            cells.truncate(count);
            cells.sort_unstable();
            cells
        }
        EddyCurrentObservationMode::LocalLeverage => {
            let mut cells = (0..total).collect::<Vec<_>>();
            cells.sort_by(|a, b| {
                setup.leverage_scores[*b]
                    .partial_cmp(&setup.leverage_scores[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(b))
            });
            cells.truncate(count);
            cells.sort_unstable();
            cells
        }
        EddyCurrentObservationMode::PriorOnly | EddyCurrentObservationMode::WeakFull => Vec::new(),
    }
}

fn local_probe_leverage_scores(
    beta_zero_model: &ReducedNonlinearScreenedEddyCurrent1Form,
    state: &[f64],
) -> Result<Vec<f64>, String> {
    let all_cells = (0..beta_zero_model.num_elements()).collect::<Vec<_>>();
    let probe = LocalEddyCurrentResidualProbe3d::from_model(beta_zero_model, all_cells)?;
    let evaluation = probe.residual_and_jacobian(state)?;
    let mut scores = vec![0.0; beta_zero_model.num_elements()];
    for (row, _, value) in evaluation.jacobian.triplet_iter() {
        scores[row / 3] += value * value;
    }
    Ok(scores)
}

#[derive(Debug, Clone)]
struct ReferenceQueryPoint {
    id: String,
    kind: String,
    index: usize,
    point: [f64; 3],
}

fn reference_query_points(
    topology: &Complex,
    coords: &MeshCoords,
    sensor_cells: &[usize],
) -> Result<Vec<ReferenceQueryPoint>, String> {
    let cell_points = cell_barycenters(topology, coords)?;
    let mut points = Vec::with_capacity(cell_points.len() + sensor_cells.len());
    for (index, point) in cell_points.iter().copied().enumerate() {
        points.push(ReferenceQueryPoint {
            id: format!("cell_{index}"),
            kind: "cell".to_string(),
            index,
            point,
        });
    }
    for (sensor_index, &cell) in sensor_cells.iter().enumerate() {
        let point = *cell_points.get(cell).ok_or_else(|| {
            format!(
                "sensor cell {cell} is outside cell count {}",
                cell_points.len()
            )
        })?;
        points.push(ReferenceQueryPoint {
            id: format!("sensor_{sensor_index}_cell_{cell}"),
            kind: "sensor".to_string(),
            index: sensor_index,
            point,
        });
    }
    Ok(points)
}

fn cell_barycenters(topology: &Complex, coords: &MeshCoords) -> Result<Vec<[f64; 3]>, String> {
    topology
        .cells()
        .handle_iter()
        .map(|cell| {
            if cell.vertices.is_empty() {
                return Err("cell has no vertices".to_string());
            }
            let mut point = [0.0; 3];
            for vertex in &cell.vertices {
                let coord = coords.coord(*vertex);
                point[0] += coord[0];
                point[1] += coord[1];
                point[2] += coord[2];
            }
            let scale = 1.0 / cell.vertices.len() as f64;
            point[0] *= scale;
            point[1] *= scale;
            point[2] *= scale;
            Ok(point)
        })
        .collect()
}

fn deterministic_sensor_cells(cell_count: usize) -> Vec<usize> {
    if cell_count == 0 {
        return Vec::new();
    }
    let mut cells = vec![0, cell_count / 2, cell_count.saturating_sub(1)];
    cells.sort_unstable();
    cells.dedup();
    cells
}

fn run_ngsolve_reference(
    config: &NonlinearEddyCurrentComparisonConfig,
    query_points: &[ReferenceQueryPoint],
    output_dir: &Path,
) -> Result<NgsolveReferenceSamples, String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create NGSolve reference output directory {}: {err}",
            output_dir.display()
        )
    })?;
    let query_csv = output_dir.join("query_points.csv");
    write_reference_query_csv(&query_csv, query_points)?;

    let mut command = Command::new(&config.ngsolve_python);
    command
        .arg("-m")
        .arg("screened_eddy_reference")
        .arg("--order")
        .arg(config.ngsolve_order.to_string())
        .arg("--curve-order")
        .arg(config.ngsolve_curve_order.to_string())
        .arg("--maxh")
        .arg(format!("{:.16e}", config.ngsolve_maxh))
        .arg("--nu0")
        .arg(format!("{:.16e}", config.nu0))
        .arg("--beta")
        .arg(format!("{:.16e}", config.beta))
        .arg("--sigma")
        .arg(format!("{:.16e}", config.sigma))
        .arg("--source-amplitude")
        .arg(format!("{:.16e}", config.source_amplitude))
        .arg("--newton-tol")
        .arg(format!("{:.16e}", config.ngsolve_newton_tolerance))
        .arg("--max-iterations")
        .arg(config.ngsolve_max_iterations.to_string())
        .arg("--query-csv")
        .arg(&query_csv)
        .arg("--output-dir")
        .arg(output_dir);
    if !config.ngsolve_write_vtu {
        command.arg("--no-vtu");
    }
    command.env("PYTHONPATH", &config.ngsolve_src);

    let output = command.output().map_err(|err| {
        format!(
            "failed to run NGSolve reference with {}: {err}",
            config.ngsolve_python.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "NGSolve reference failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let summary = read_ngsolve_summary(&output_dir.join("reference_summary.json"))?;
    read_ngsolve_samples(
        &output_dir.join("reference_samples.csv"),
        query_points,
        summary,
    )
}

fn write_reference_query_csv(path: &Path, points: &[ReferenceQueryPoint]) -> Result<(), String> {
    let file = File::create(path).map_err(|err| {
        format!(
            "failed to create reference query CSV {}: {err}",
            path.display()
        )
    })?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "id,kind,index,x,y,z").map_err(|err| err.to_string())?;
    for point in points {
        writeln!(
            writer,
            "{},{},{},{:.16e},{:.16e},{:.16e}",
            point.id, point.kind, point.index, point.point[0], point.point[1], point.point[2]
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn read_ngsolve_summary(path: &Path) -> Result<NgsolveRunSummary, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read NGSolve summary {}: {err}", path.display()))?;
    Ok(NgsolveRunSummary {
        converged: json_bool(&text, "converged")
            .ok_or_else(|| "NGSolve summary missing boolean `converged`".to_string())?,
        newton_iterations: json_usize(&text, "newton_iterations")
            .ok_or_else(|| "NGSolve summary missing integer `newton_iterations`".to_string())?,
        sample_count: json_usize(&text, "sample_count")
            .ok_or_else(|| "NGSolve summary missing integer `sample_count`".to_string())?,
    })
}

fn read_ngsolve_samples(
    path: &Path,
    query_points: &[ReferenceQueryPoint],
    summary: NgsolveRunSummary,
) -> Result<NgsolveReferenceSamples, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read NGSolve samples {}: {err}", path.display()))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| "NGSolve sample CSV is empty".to_string())?;
    let columns = header
        .split(',')
        .enumerate()
        .map(|(index, name)| (name.to_string(), index))
        .collect::<BTreeMap<_, _>>();
    for required in [
        "kind", "index", "valid", "A_x", "A_y", "A_z", "B_x", "B_y", "B_z",
    ] {
        if !columns.contains_key(required) {
            return Err(format!("NGSolve sample CSV missing column `{required}`"));
        }
    }

    let cell_count = query_points
        .iter()
        .filter(|point| point.kind == "cell")
        .count();
    let sensor_count = query_points
        .iter()
        .filter(|point| point.kind == "sensor")
        .count();
    let mut cell_a = vec![[f64::NAN; 3]; cell_count];
    let mut cell_b = vec![[f64::NAN; 3]; cell_count];
    let mut sensor_a = vec![[f64::NAN; 3]; sensor_count];
    let mut sensor_b = vec![[f64::NAN; 3]; sensor_count];

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split(',').collect::<Vec<_>>();
        let kind = csv_field(&fields, &columns, "kind")?;
        let index = csv_field(&fields, &columns, "index")?
            .parse::<usize>()
            .map_err(|_| "NGSolve sample index is not an integer".to_string())?;
        let valid = csv_field(&fields, &columns, "valid")?
            .parse::<usize>()
            .map_err(|_| "NGSolve sample valid flag is not an integer".to_string())?;
        if valid != 1 {
            return Err(format!(
                "NGSolve reference sample `{kind}` index {index} is invalid"
            ));
        }
        let a = [
            csv_f64(&fields, &columns, "A_x")?,
            csv_f64(&fields, &columns, "A_y")?,
            csv_f64(&fields, &columns, "A_z")?,
        ];
        let b = [
            csv_f64(&fields, &columns, "B_x")?,
            csv_f64(&fields, &columns, "B_y")?,
            csv_f64(&fields, &columns, "B_z")?,
        ];
        match kind {
            "cell" => {
                if index >= cell_a.len() {
                    return Err(format!("NGSolve cell sample index {index} is out of range"));
                }
                cell_a[index] = a;
                cell_b[index] = b;
            }
            "sensor" => {
                if index >= sensor_a.len() {
                    return Err(format!(
                        "NGSolve sensor sample index {index} is out of range"
                    ));
                }
                sensor_a[index] = a;
                sensor_b[index] = b;
            }
            other => {
                return Err(format!("unsupported NGSolve sample kind `{other}`"));
            }
        }
    }

    if cell_a
        .iter()
        .chain(sensor_a.iter())
        .chain(cell_b.iter())
        .chain(sensor_b.iter())
        .flat_map(|value| value.iter())
        .any(|value| !value.is_finite())
    {
        return Err("NGSolve sample CSV did not provide all requested finite samples".to_string());
    }

    Ok(NgsolveReferenceSamples {
        cell_a,
        cell_b,
        sensor_b,
        summary,
    })
}

fn csv_field<'a>(
    fields: &'a [&'a str],
    columns: &BTreeMap<String, usize>,
    name: &str,
) -> Result<&'a str, String> {
    let index = columns
        .get(name)
        .copied()
        .ok_or_else(|| format!("CSV column `{name}` is missing"))?;
    fields
        .get(index)
        .copied()
        .ok_or_else(|| format!("CSV row is missing column `{name}`"))
}

fn csv_f64(fields: &[&str], columns: &BTreeMap<String, usize>, name: &str) -> Result<f64, String> {
    let value = csv_field(fields, columns, name)?
        .parse::<f64>()
        .map_err(|_| format!("CSV column `{name}` is not a finite float"))?;
    if !value.is_finite() {
        return Err(format!("CSV column `{name}` is not finite"));
    }
    Ok(value)
}

fn json_bool(text: &str, key: &str) -> Option<bool> {
    let value = json_raw_value(text, key)?;
    if value.starts_with("true") {
        Some(true)
    } else if value.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn json_usize(text: &str, key: &str) -> Option<usize> {
    json_raw_value(text, key)?
        .split([',', '\n', '}'])
        .next()?
        .trim()
        .parse()
        .ok()
}

fn json_raw_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let key_pattern = format!("\"{key}\"");
    let start = text.find(&key_pattern)?;
    let after_key = &text[start + key_pattern.len()..];
    let colon = after_key.find(':')?;
    Some(after_key[colon + 1..].trim_start())
}

fn calibrate_probe_variance(
    model: &ReducedNonlinearScreenedEddyCurrent1Form,
    state: &[f64],
    relative_scale: f64,
) -> Result<f64, String> {
    if !relative_scale.is_finite() || relative_scale <= 0.0 {
        return Err("probe noise relative scale must be finite and positive".to_string());
    }
    let all_cells = (0..model.num_elements()).collect::<Vec<_>>();
    let probe = LocalEddyCurrentResidualProbe3d::from_model(model, all_cells)?;
    let residual = probe.residual_and_jacobian(state)?.residual;
    let rms = rms(residual.as_slice()).max(1e-12);
    Ok((relative_scale * rms).powi(2))
}

#[derive(Debug, Clone, Copy)]
struct SensorMetrics {
    rmse: f64,
    relative_rmse: f64,
    max_standardized_error: f64,
    coverage_2sigma: f64,
}

fn sensor_b_metrics(
    cell_b: &[f64],
    reference_sensor_b: &[[f64; 3]],
    sensor_cells: &[usize],
    result: &NonlinearLaplaceResult,
) -> SensorMetrics {
    let mut prediction = Vec::new();
    let mut reference = Vec::new();
    for (sensor_index, &cell) in sensor_cells.iter().enumerate() {
        if 3 * cell + 2 >= cell_b.len() || sensor_index >= reference_sensor_b.len() {
            continue;
        }
        prediction.extend_from_slice(&cell_b[3 * cell..3 * cell + 3]);
        reference.extend_from_slice(&reference_sensor_b[sensor_index]);
    }
    let rmse = rmse_pair(&prediction, &reference);
    let relative_rmse = rmse / rms(&reference).max(1e-15);
    let Some(variance) = result.derived_variances.get("selected_cell_B") else {
        return SensorMetrics {
            rmse,
            relative_rmse,
            max_standardized_error: f64::NAN,
            coverage_2sigma: f64::NAN,
        };
    };
    let mut covered = 0usize;
    let mut total = 0usize;
    let mut max_standardized_error = 0.0f64;
    for ((predicted, truth), var) in prediction
        .iter()
        .zip(reference.iter())
        .zip(variance.posterior_variance.iter())
    {
        let sd = var.max(0.0).sqrt();
        if sd > 0.0 {
            let standardized = (predicted - truth).abs() / sd;
            max_standardized_error = max_standardized_error.max(standardized);
            if standardized <= 2.0 {
                covered += 1;
            }
            total += 1;
        }
    }
    SensorMetrics {
        rmse,
        relative_rmse,
        max_standardized_error: if total == 0 {
            f64::NAN
        } else {
            max_standardized_error
        },
        coverage_2sigma: if total == 0 {
            f64::NAN
        } else {
            covered as f64 / total as f64
        },
    }
}

fn budget_counts(fractions: &[f64], total: usize) -> Vec<usize> {
    let mut counts = fractions
        .iter()
        .map(|fraction| {
            if *fraction <= 0.0 {
                0
            } else {
                ((*fraction * total as f64).ceil() as usize).clamp(1, total)
            }
        })
        .collect::<Vec<_>>();
    counts.sort_unstable();
    counts.dedup();
    counts
}

fn diagonal_prior(dimension: usize, mean: Vec<f64>, precision: f64) -> GaussianPriorSpec {
    let mut matrix = SparseTripletMatrix::new(dimension, dimension);
    for index in 0..dimension {
        matrix.push(index, index, precision);
    }
    GaussianPriorSpec {
        mean,
        precision: matrix,
    }
}

fn solve_spd_triplet(matrix: &SparseTripletMatrix, rhs: &[f64]) -> Result<Vec<f64>, String> {
    let factor = sparse_from_core(matrix)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor screened eddy-current linear proxy: {err}"))?;
    factor
        .solve(&GmrfVector::from_vec(rhs.to_vec()))
        .map(|solution| solution.as_slice().to_vec())
        .map_err(|err| format!("failed to solve screened eddy-current linear proxy: {err}"))
}

fn weak_residual_norm(
    model: &ReducedNonlinearScreenedEddyCurrent1Form,
    state: &[f64],
) -> Result<f64, String> {
    model
        .residual_and_jacobian(state)
        .map(|evaluation| l2_norm(evaluation.residual.as_slice()))
}

fn weighted_norm_diag(values: &[f64], precision: &SparseTripletMatrix) -> f64 {
    let mut squared = 0.0;
    for (row, col, value) in precision.triplet_iter() {
        if row == col {
            squared += value * values[row] * values[row];
        }
    }
    squared.sqrt()
}

fn apply_triplet(matrix: &SparseTripletMatrix, vector: &[f64]) -> Result<Vec<f64>, String> {
    if matrix.ncols() != vector.len() {
        return Err(format!(
            "sparse matrix-vector dimension mismatch: matrix has {} columns but vector has length {}",
            matrix.ncols(),
            vector.len()
        ));
    }
    let mut out = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        out[row] += value * vector[col];
    }
    Ok(out)
}

fn sparse_rows(matrix: &SparseTripletMatrix) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        rows[row].push((col, value));
    }
    rows
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn rmse_pair(values: &[f64], reference: &[f64]) -> f64 {
    if values.len() != reference.len() || values.is_empty() {
        return f64::NAN;
    }
    (values
        .iter()
        .zip(reference.iter())
        .map(|(value, reference)| (value - reference).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn flatten3(values: &[[f64; 3]]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| value.iter().copied())
        .collect()
}

fn relative_error3(values: &[[f64; 3]], reference: &[[f64; 3]]) -> f64 {
    relative_error_flat(&flatten3(values), &flatten3(reference))
}

fn relative_error_flat(values: &[f64], reference: &[f64]) -> f64 {
    if values.len() != reference.len() {
        return f64::NAN;
    }
    let error = values
        .iter()
        .zip(reference.iter())
        .map(|(value, reference)| (value - reference).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = l2_norm(reference).max(1e-15);
    error / norm
}

fn validate_config(config: &NonlinearEddyCurrentComparisonConfig) -> Result<(), String> {
    if config.mesh_level == 0 {
        return Err(
            "nonlinear eddy-current comparison mesh_level must be at least one".to_string(),
        );
    }
    if !config.nu0.is_finite() || config.nu0 <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison nu0 must be finite and positive".to_string(),
        );
    }
    if !config.beta.is_finite() || config.beta < 0.0 {
        return Err(
            "nonlinear eddy-current comparison beta must be finite and nonnegative".to_string(),
        );
    }
    if !config.sigma.is_finite() || config.sigma <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison sigma must be finite and positive".to_string(),
        );
    }
    if !config.source_amplitude.is_finite() || config.source_amplitude <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison source_amplitude must be finite and positive"
                .to_string(),
        );
    }
    if !config.weak_diagonal_precision.is_finite() || config.weak_diagonal_precision <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison weak_diagonal_precision must be finite and positive"
                .to_string(),
        );
    }
    if !config.linear_proxy_tau.is_finite() || config.linear_proxy_tau <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison linear_proxy_tau must be finite and positive"
                .to_string(),
        );
    }
    if !config.residual_variance.is_finite() || config.residual_variance <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison residual_variance must be finite and positive"
                .to_string(),
        );
    }
    if let Some(probe_variance) = config.probe_variance {
        if !probe_variance.is_finite() || probe_variance <= 0.0 {
            return Err(
                "nonlinear eddy-current comparison probe_variance must be finite and positive"
                    .to_string(),
            );
        }
    }
    if !config.probe_noise_relative_scale.is_finite() || config.probe_noise_relative_scale <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison probe_noise_relative_scale must be finite and positive"
                .to_string(),
        );
    }
    if config.max_iterations == 0 {
        return Err(
            "nonlinear eddy-current comparison max_iterations must be positive".to_string(),
        );
    }
    if config
        .budget_fractions
        .iter()
        .any(|fraction| !fraction.is_finite() || *fraction < 0.0 || *fraction > 1.0)
    {
        return Err(
            "nonlinear eddy-current comparison budget fractions must be finite and in [0, 1]"
                .to_string(),
        );
    }
    if config.ngsolve_order == 0 || config.ngsolve_curve_order == 0 {
        return Err(
            "nonlinear eddy-current comparison NGSolve orders must be positive".to_string(),
        );
    }
    if !config.ngsolve_maxh.is_finite() || config.ngsolve_maxh <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison ngsolve_maxh must be finite and positive"
                .to_string(),
        );
    }
    if !config.ngsolve_newton_tolerance.is_finite() || config.ngsolve_newton_tolerance <= 0.0 {
        return Err(
            "nonlinear eddy-current comparison ngsolve_newton_tolerance must be finite and positive"
                .to_string(),
        );
    }
    if config.ngsolve_max_iterations == 0 {
        return Err(
            "nonlinear eddy-current comparison ngsolve_max_iterations must be positive".to_string(),
        );
    }
    if config.check_reference_adequacy {
        if config.adequacy_ngsolve_order == 0 {
            return Err(
                "nonlinear eddy-current comparison adequacy_ngsolve_order must be positive"
                    .to_string(),
            );
        }
        if !config.adequacy_ngsolve_maxh.is_finite() || config.adequacy_ngsolve_maxh <= 0.0 {
            return Err(
                "nonlinear eddy-current comparison adequacy_ngsolve_maxh must be finite and positive"
                    .to_string(),
            );
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;

    fn smoke_config(name: &str) -> NonlinearEddyCurrentComparisonConfig {
        let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../target/nonlinear_eddy_current_ngsolve_reference_tests/{name}"
        ));
        NonlinearEddyCurrentComparisonConfig {
            mesh_level: 2,
            budget_fractions: vec![0.0, 0.25, 0.5, 1.0],
            max_iterations: 12,
            ngsolve_order: 1,
            ngsolve_maxh: 0.5,
            ngsolve_newton_tolerance: 1e-8,
            ngsolve_output_dir: output_dir,
            ..NonlinearEddyCurrentComparisonConfig::default()
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_eddy_current_full_weak_solve_converges() {
        let setup = build_setup(&smoke_config("full_weak")).expect("setup should build");
        let config = smoke_config("full_weak_solve");
        let prior = build_prior(
            &setup,
            EddyCurrentPriorMode::LinearProxyMaternAlpha2,
            &config,
        )
        .unwrap();
        let result = solve_with_weak_residual(
            &setup.model,
            prior,
            &setup.linear_mean,
            config.residual_variance,
            config.max_iterations,
            &setup.b_operator,
            &setup.sensor_cells,
        )
        .expect("full weak residual solve should converge");
        assert!(result.converged);
        assert!(weak_residual_norm(&setup.model, &result.map).unwrap() < 1e-6);
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_eddy_current_full_local_probes_track_weak_reference() {
        let config = smoke_config("full_local");
        let report = run_nonlinear_eddy_current_weiland_comparison_experiment(config)
            .expect("comparison should run");
        let full_local = report
            .rows
            .iter()
            .find(|row| {
                row.prior_mode == EddyCurrentPriorMode::LinearProxyMaternAlpha2
                    && row.observation_mode == EddyCurrentObservationMode::LocalFull
            })
            .expect("full local row should be present");
        assert!(full_local.success, "{:?}", full_local.failure);
        assert!(
            full_local.ngsolve_cell_b_relative_error.is_finite(),
            "full local probes should report a finite NGSolve B error"
        );
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_eddy_current_sparse_probe_comparison_records_finite_metrics() {
        let report = run_nonlinear_eddy_current_weiland_comparison_experiment(smoke_config(
            "sparse_metrics",
        ))
        .expect("comparison should run");
        let sparse_rows = report
            .rows
            .iter()
            .filter(|row| {
                matches!(
                    row.observation_mode,
                    EddyCurrentObservationMode::LocalLeverage
                        | EddyCurrentObservationMode::LocalUniform
                )
            })
            .collect::<Vec<_>>();
        assert!(!sparse_rows.is_empty());
        for row in sparse_rows {
            assert!(row.success, "{:?}", row.failure);
            assert!(row.final_weak_residual_norm.is_finite());
            assert!(row.ngsolve_cell_a_relative_error.is_finite());
            assert!(row.ngsolve_cell_b_relative_error.is_finite());
            assert!(row.ngsolve_sensor_b_rmse.is_finite());
            assert!(row.ngsolve_sensor_b_relative_rmse.is_finite());
            assert!(row.selected_b_variance_min.is_finite());
            assert!(row.selected_b_variance_max.is_finite());
        }
    }

    #[cfg(feature = "heavy-tests")]
    #[test]
    fn nonlinear_eddy_current_physical_proxy_plus_residual_improves_over_prior_only() {
        let report = run_nonlinear_eddy_current_weiland_comparison_experiment(smoke_config(
            "proxy_improves",
        ))
        .expect("comparison should run");
        let prior_only = report
            .rows
            .iter()
            .find(|row| {
                row.prior_mode == EddyCurrentPriorMode::LinearProxyMaternAlpha2
                    && row.observation_mode == EddyCurrentObservationMode::PriorOnly
            })
            .expect("prior-only row should be present");
        let moderate = report
            .rows
            .iter()
            .find(|row| {
                row.prior_mode == EddyCurrentPriorMode::LinearProxyMaternAlpha2
                    && row.observation_mode == EddyCurrentObservationMode::LocalLeverage
                    && row.probe_count >= row.total_cells / 2
                    && row.probe_count < row.total_cells
            })
            .expect("moderate leverage row should be present");
        assert!(moderate.success, "{:?}", moderate.failure);
        assert!(
            moderate.final_weak_residual_norm < prior_only.final_weak_residual_norm,
            "physical proxy plus residual probes should reduce the true weak residual"
        );
    }
}
