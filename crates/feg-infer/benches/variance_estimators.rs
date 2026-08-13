use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::ManifoldComplexExt;
use feg_infer::metis_ordering::metis_nested_dissection_ordering;
use feg_infer::prior::matern::zero_form::{
    build_laplace_beltrami_0form, build_matern_precision_0form, feec_csr_to_gmrf, feec_vec_to_gmrf,
    MaternConfig, MaternMassInverse,
};
use feg_infer::sparse::sparse_row_operator_from_feec_csr;
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::{
    estimate_hutchinson_transformed_variances, estimate_hutchinson_variances,
    estimate_local_rbmc_transformed_variances, estimate_local_rbmc_variances,
    estimate_monte_carlo_transformed_variances, estimate_monte_carlo_variances,
    selected_inverse_diag_with_diagnostics, selected_inverse_transformed_diag, BlockId,
    CholeskyOrdering, Gmrf, LatentBlockMode, Permutation, PermutedIndex, ProbeDistribution,
    SelectedInverseDiagnostics, SparseRowOperator, VarianceEstimate, Vector,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use std::collections::BTreeSet;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

const SAMPLE_COUNTS_MC: &[usize] = &[8, 16, 32, 64, 128, 256, 512];
const SAMPLE_COUNTS_HUTCHINSON: &[usize] = &[8, 16, 32, 64, 128, 256, 512];
const SAMPLE_COUNTS_LOCAL_RB: &[usize] = &[8, 16, 32, 64, 128, 256];
const BATCH_COUNT: usize = 8;
const TARGET_RELATIVE_L2_ERROR: f64 = 0.10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchmarkOrdering {
    Amd,
    Identity,
    Metis,
}

#[derive(Clone)]
struct BenchmarkRow {
    target: &'static str,
    method: String,
    sample_count: usize,
    factorization_time: Duration,
    estimator_time: Duration,
    relative_l2_error: f64,
    median_pointwise_relative_error: f64,
    p95_pointwise_relative_error: f64,
    num_negative: usize,
    min_value: f64,
    batch_relative_standard_error: f64,
    selected_requested_pairs: usize,
    selected_closure_pairs: usize,
    selected_factor_pairs: usize,
    selected_closure_over_factor: f64,
    selected_closure_limit: usize,
    status: String,
}

impl BenchmarkRow {
    fn total_time(&self) -> Duration {
        self.factorization_time + self.estimator_time
    }

    fn with_selected_diagnostics(mut self, diagnostics: &SelectedInverseDiagnostics) -> Self {
        self.selected_requested_pairs = diagnostics.requested_pairs;
        self.selected_closure_pairs = diagnostics.closure_pairs;
        self.selected_factor_pairs = diagnostics.factor_pairs;
        self.selected_closure_over_factor = diagnostics.closure_over_factor;
        self.selected_closure_limit = diagnostics.closure_limit;
        self
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("variance estimator benchmark failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let problem = build_uq_problem()?;
    let ordering_request =
        env_benchmark_ordering("VARIANCE_BENCH_ORDERING", BenchmarkOrdering::Amd)?;
    let ordering_start = Instant::now();
    let Some((ordering_label, ordering)) =
        resolve_benchmark_ordering(ordering_request, &problem.posterior_precision)?
    else {
        println!("ordering=metis");
        println!("metis=unavailable");
        return Ok(());
    };
    let ordering_time = ordering_start.elapsed();
    let factor_start = Instant::now();
    let factor = problem
        .posterior_precision
        .cholesky_sqrt_lower_with_ordering(ordering)
        .map_err(|err| err.to_string())?;
    let factorization_time = factor_start.elapsed();
    println!("ordering={ordering_label}");
    println!("ordering_time_seconds={:.6}", ordering_time.as_secs_f64());
    println!(
        "variance estimator benchmark: vertices={} edges={} precision_nnz={} factor_nnz={}",
        factor.dimension(),
        problem.d0.nrows(),
        problem.posterior_precision.nnz(),
        factor.nnz()
    );
    println!(
        "factorization_time_seconds={:.6}",
        factorization_time.as_secs_f64()
    );
    if env_bool("VARIANCE_BENCH_FILL_ONLY", false)? {
        return Ok(());
    }
    let mut posterior = Gmrf::from_information_and_precision_with_sqrt(
        problem.information,
        problem.posterior_precision,
        factor,
    )
    .map_err(|err| err.to_string())?;

    let mut rows = Vec::new();
    let skip_exact = env_bool("VARIANCE_BENCH_SKIP_EXACT", false)?;
    let skip_selected = env_bool("VARIANCE_BENCH_SKIP_SELECTED", false)?;
    let (latent_reference, transformed_reference) = if skip_exact {
        println!("exact_reference=skipped");
        (None, None)
    } else {
        let exact_latent_start = Instant::now();
        let latent_reference = exact_repeated_latent_variances(
            posterior
                .precision_factor()
                .ok_or_else(|| "posterior factor missing".to_string())?,
        )?;
        let exact_latent_time = exact_latent_start.elapsed();
        let exact_transformed_start = Instant::now();
        let transformed_reference = exact_repeated_transformed_variances(
            posterior
                .precision_factor()
                .ok_or_else(|| "posterior factor missing".to_string())?,
            &problem.d0,
        )?;
        let exact_transformed_time = exact_transformed_start.elapsed();

        rows.push(row_from_values(VarianceRowInput {
            target: "latent",
            method: "exact-repeated-solves".to_string(),
            sample_count: 0,
            factorization_time,
            estimator_time: exact_latent_time,
            reference: Some(&latent_reference),
            values: &latent_reference,
            relative_standard_error: None,
            status: "ok".to_string(),
        }));
        rows.push(row_from_values(VarianceRowInput {
            target: "d0",
            method: "exact-repeated-solves".to_string(),
            sample_count: 0,
            factorization_time,
            estimator_time: exact_transformed_time,
            reference: Some(&transformed_reference),
            values: &transformed_reference,
            relative_standard_error: None,
            status: "ok".to_string(),
        }));
        (Some(latent_reference), Some(transformed_reference))
    };

    if skip_selected {
        println!("selected_inverse=skipped");
    } else {
        let selected_start = Instant::now();
        let selected_latent = selected_inverse_diag_with_diagnostics(
            posterior
                .precision_factor()
                .ok_or_else(|| "posterior factor missing".to_string())?,
        )
        .map_err(|err| err.to_string())?;
        rows.push(
            row_from_estimate(
                "latent",
                "selected-inverse".to_string(),
                factorization_time,
                selected_start.elapsed(),
                latent_reference.as_ref(),
                &selected_latent.estimate,
                "ok".to_string(),
            )
            .with_selected_diagnostics(&selected_latent.diagnostics),
        );

        let selected_transformed_start = Instant::now();
        let selected_transformed = selected_inverse_transformed_diag(
            posterior
                .precision_factor()
                .ok_or_else(|| "posterior factor missing".to_string())?,
            &problem.d0,
        )
        .map_err(|err| err.to_string())?;
        let selected_transformed_time = selected_transformed_start.elapsed();
        if let Some(estimate) = selected_transformed.estimate {
            rows.push(
                row_from_estimate(
                    "d0",
                    "selected-inverse".to_string(),
                    factorization_time,
                    selected_transformed_time,
                    transformed_reference.as_ref(),
                    &estimate,
                    "ok".to_string(),
                )
                .with_selected_diagnostics(&selected_transformed.diagnostics),
            );
        } else {
            rows.push(
                unavailable_row(
                    "d0",
                    "selected-inverse".to_string(),
                    0,
                    factorization_time,
                    selected_transformed_time,
                    format!(
                        "closure-too-large requested={} closure={} limit={}",
                        selected_transformed.diagnostics.requested_pairs,
                        selected_transformed.diagnostics.closure_pairs,
                        selected_transformed.diagnostics.closure_limit
                    ),
                )
                .with_selected_diagnostics(&selected_transformed.diagnostics),
            );
        }
    }

    for &samples in SAMPLE_COUNTS_MC {
        let start = Instant::now();
        let estimate = estimate_monte_carlo_variances(
            &mut posterior,
            samples,
            BATCH_COUNT,
            1_000 + samples as u64,
        )
        .map_err(|err| err.to_string())?;
        rows.push(row_from_estimate(
            "latent",
            "monte-carlo".to_string(),
            factorization_time,
            start.elapsed(),
            latent_reference.as_ref(),
            &estimate,
            "ok".to_string(),
        ));

        let start = Instant::now();
        let estimate = estimate_monte_carlo_transformed_variances(
            &mut posterior,
            &problem.d0,
            samples,
            BATCH_COUNT,
            2_000 + samples as u64,
        )
        .map_err(|err| err.to_string())?;
        rows.push(row_from_estimate(
            "d0",
            "monte-carlo".to_string(),
            factorization_time,
            start.elapsed(),
            transformed_reference.as_ref(),
            &estimate,
            "ok".to_string(),
        ));
    }

    for &samples in SAMPLE_COUNTS_HUTCHINSON {
        let start = Instant::now();
        let estimate = estimate_hutchinson_variances(
            &mut posterior,
            samples,
            BATCH_COUNT,
            3_000 + samples as u64,
            ProbeDistribution::Rademacher,
        )
        .map_err(|err| err.to_string())?;
        rows.push(row_from_estimate(
            "latent",
            "hutchinson".to_string(),
            factorization_time,
            start.elapsed(),
            latent_reference.as_ref(),
            &estimate,
            "ok".to_string(),
        ));

        let start = Instant::now();
        let estimate = estimate_hutchinson_transformed_variances(
            &mut posterior,
            &problem.d0,
            samples,
            BATCH_COUNT,
            4_000 + samples as u64,
            ProbeDistribution::Rademacher,
        )
        .map_err(|err| err.to_string())?;
        rows.push(row_from_estimate(
            "d0",
            "hutchinson".to_string(),
            factorization_time,
            start.elapsed(),
            transformed_reference.as_ref(),
            &estimate,
            "ok".to_string(),
        ));
    }

    let precision = posterior
        .precision_matrix()
        .ok_or_else(|| "posterior precision missing".to_string())?;
    let factor = posterior
        .precision_factor()
        .ok_or_else(|| "posterior factor missing".to_string())?;
    let local_rb_block_size = env_usize("VARIANCE_BENCH_LOCAL_RB_BLOCK_SIZE", 16)?;
    let latent_blocks = LatentBlockMode::ContiguousPermuted {
        block_size: local_rb_block_size,
    };
    let use_row_blocks = std::env::var("VARIANCE_BENCH_LOCAL_RB_ROW_BLOCKS")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let transformed_blocks = if use_row_blocks {
        row_support_blocks(factor, &problem.d0)?
    } else {
        support_patch_blocks(factor, &problem.d0, local_rb_block_size)?
    };
    for &samples in SAMPLE_COUNTS_LOCAL_RB {
        let start = Instant::now();
        let estimate = estimate_local_rbmc_variances(
            precision,
            factor,
            &latent_blocks,
            samples,
            BATCH_COUNT,
            5_000 + samples as u64,
        )
        .map_err(|err| err.to_string())?;
        rows.push(row_from_estimate(
            "latent",
            "local-rbmc".to_string(),
            factorization_time,
            start.elapsed(),
            latent_reference.as_ref(),
            &estimate.estimate,
            "ok".to_string(),
        ));

        let start = Instant::now();
        let estimate = estimate_local_rbmc_transformed_variances(
            precision,
            factor,
            &problem.d0,
            &transformed_blocks,
            samples,
            BATCH_COUNT,
            6_000 + samples as u64,
        )
        .map_err(|err| err.to_string())?;
        rows.push(row_from_estimate(
            "d0",
            "local-rbmc".to_string(),
            factorization_time,
            start.elapsed(),
            transformed_reference.as_ref(),
            &estimate.estimate,
            "ok".to_string(),
        ));
    }

    print_table(&rows);
    print_thresholds(&rows);
    write_csv(&rows)?;
    Ok(())
}

struct BuiltProblem {
    posterior_precision: gmrf_core::SparseMatrix,
    information: Vector,
    d0: SparseRowOperator,
}

fn exact_repeated_latent_variances(
    factor: &gmrf_core::SparseCholeskyFactor,
) -> Result<Vector, String> {
    let n = factor.dimension();
    let mut values = Vector::zeros(n);
    for index in 0..n {
        let mut rhs = Vector::zeros(n);
        rhs[index] = 1.0;
        factor
            .solve_in_place(&mut rhs)
            .map_err(|err| err.to_string())?;
        values[index] = rhs[index];
    }
    Ok(values)
}

fn exact_repeated_transformed_variances(
    factor: &gmrf_core::SparseCholeskyFactor,
    operator: &SparseRowOperator,
) -> Result<Vector, String> {
    let mut values = Vector::zeros(operator.nrows());
    for row_index in 0..operator.nrows() {
        let rhs = operator
            .row_as_vector(row_index)
            .map_err(|err| err.to_string())?;
        let solved = factor.solve(&rhs).map_err(|err| err.to_string())?;
        values[row_index] = rhs.dot(&solved);
    }
    Ok(values)
}

fn build_uq_problem() -> Result<BuiltProblem, String> {
    let dim = env_usize("VARIANCE_BENCH_DIM", 3)?;
    let cells_axis = env_usize("VARIANCE_BENCH_CELLS_AXIS", 8)?;
    let mesh = CartesianMeshInfo::new_unit_scaled(dim, cells_axis, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let laplace = build_laplace_beltrami_0form(&topology, &metric);
    let prior_precision = build_matern_precision_0form(
        &laplace,
        MaternConfig {
            kappa: 1.5,
            tau: 1.0,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );
    let ndofs = laplace.mass.nrows();
    let truth = FeecVector::from_iterator(
        ndofs,
        (0..ndofs).map(|index| {
            let t = index as f64 / ndofs.max(1) as f64;
            (2.0 * std::f64::consts::PI * t).sin() + 0.25 * (7.0 * t).cos()
        }),
    );
    let rhs = &laplace.laplacian * &truth;

    let h_gmrf = feec_csr_to_gmrf(&laplace.laplacian);
    let y_gmrf = feec_vec_to_gmrf(&rhs);
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);
    let (posterior_precision, information) =
        apply_gaussian_observations(&q_prior_gmrf, &h_gmrf, &y_gmrf, None, 2.5e-3);

    let d0 = sparse_row_operator_from_feec_csr(&FeecCsr::from(
        &topology.exterior_derivative_operator(0),
    ))?;
    Ok(BuiltProblem {
        posterior_precision,
        information,
        d0,
    })
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|err| format!("invalid {name}={value:?}: {err}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(format!("invalid {name}: {err}")),
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool, String> {
    match std::env::var(name) {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!("invalid {name}={value:?}: expected boolean")),
        },
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(format!("invalid {name}: {err}")),
    }
}

fn env_benchmark_ordering(
    name: &str,
    default: BenchmarkOrdering,
) -> Result<BenchmarkOrdering, String> {
    match std::env::var(name) {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "amd" => Ok(BenchmarkOrdering::Amd),
            "identity" | "natural" => Ok(BenchmarkOrdering::Identity),
            "metis" | "ndmetis" | "nested-dissection" => Ok(BenchmarkOrdering::Metis),
            _ => Err(format!(
                "invalid {name}={value:?}: expected amd, identity, or metis"
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(format!("invalid {name}: {err}")),
    }
}

fn resolve_benchmark_ordering(
    request: BenchmarkOrdering,
    precision: &gmrf_core::SparseMatrix,
) -> Result<Option<(&'static str, CholeskyOrdering)>, String> {
    match request {
        BenchmarkOrdering::Amd => Ok(Some(("amd", CholeskyOrdering::Amd))),
        BenchmarkOrdering::Identity => Ok(Some(("identity", CholeskyOrdering::Identity))),
        BenchmarkOrdering::Metis => metis_nested_dissection_ordering(precision).map(|ordering| {
            ordering.map(|permutation| ("metis", CholeskyOrdering::Custom(permutation)))
        }),
    }
}

fn support_patch_blocks(
    factor: &gmrf_core::SparseCholeskyFactor,
    operator: &SparseRowOperator,
    target_block_size: usize,
) -> Result<LatentBlockMode, String> {
    let permutation = factor.permutation();
    if operator.nrows() == 0 {
        return Ok(LatentBlockMode::Explicit {
            blocks: Vec::new(),
            row_assignments: Some(Vec::new()),
        });
    }
    if operator.ncols == 0 {
        return Err("cannot build local RB patches for an operator with zero columns".to_string());
    }

    let row_supports = operator
        .rows
        .iter()
        .map(|row| unique_row_support(row))
        .collect::<Vec<_>>();
    let max_support_size = row_supports
        .iter()
        .map(|support| support.len())
        .max()
        .unwrap_or(1);
    let target_block_size = target_block_size.max(max_support_size).max(1);
    let adjacency = support_graph_adjacency(operator.ncols, &row_supports);

    let mut blocks = Vec::new();
    let mut assignments = vec![BlockId(usize::MAX); operator.nrows()];
    let mut assigned = vec![false; operator.nrows()];

    while let Some(seed_row) = assigned.iter().position(|is_assigned| !*is_assigned) {
        let mut patch = BTreeSet::new();
        if row_supports[seed_row].is_empty() {
            patch.insert(0);
        } else {
            patch.extend(row_supports[seed_row].iter().copied());
        }
        grow_support_patch(&mut patch, &adjacency, target_block_size);

        let block_id = BlockId(blocks.len());
        let mut assigned_any = false;
        for (row_index, support) in row_supports.iter().enumerate() {
            if assigned[row_index] {
                continue;
            }
            let contained =
                support.is_empty() || support.iter().all(|column| patch.contains(column));
            if contained {
                assigned[row_index] = true;
                assignments[row_index] = block_id;
                assigned_any = true;
            }
        }

        if !assigned_any {
            return Err(
                "failed to assign a transformed local RB row to a support patch".to_string(),
            );
        }

        let mut block = permuted_block_from_original_patch(&permutation, patch)?;
        block.sort_unstable();
        block.dedup();
        blocks.push(block);
    }

    Ok(LatentBlockMode::Explicit {
        blocks,
        row_assignments: Some(assignments),
    })
}

fn row_support_blocks(
    factor: &gmrf_core::SparseCholeskyFactor,
    operator: &SparseRowOperator,
) -> Result<LatentBlockMode, String> {
    let permutation = factor.permutation();
    if operator.nrows() == 0 {
        return Ok(LatentBlockMode::Explicit {
            blocks: Vec::new(),
            row_assignments: Some(Vec::new()),
        });
    }
    if operator.ncols == 0 {
        return Err(
            "cannot build local RB row-support blocks for an operator with zero columns"
                .to_string(),
        );
    }

    let mut blocks = Vec::with_capacity(operator.nrows());
    let mut assignments = Vec::with_capacity(operator.nrows());
    for (row_index, row) in operator.rows.iter().enumerate() {
        let original_patch = unique_row_support(row).into_iter().collect::<BTreeSet<_>>();
        let mut block = permuted_block_from_original_patch(&permutation, original_patch)?;
        if block.is_empty() {
            block.push(PermutedIndex(0));
        }
        block.sort_unstable();
        block.dedup();
        blocks.push(block);
        assignments.push(BlockId(row_index));
    }

    Ok(LatentBlockMode::Explicit {
        blocks,
        row_assignments: Some(assignments),
    })
}

fn permuted_block_from_original_patch(
    permutation: &Permutation,
    patch: BTreeSet<usize>,
) -> Result<Vec<PermutedIndex>, String> {
    patch
        .into_iter()
        .map(|original| {
            permutation
                .orig_to_perm
                .get(original)
                .copied()
                .map(PermutedIndex)
                .ok_or_else(|| {
                    format!("operator column {original} is outside the Cholesky permutation domain")
                })
        })
        .collect()
}

fn unique_row_support(row: &[(usize, f64)]) -> Vec<usize> {
    let mut support = row.iter().map(|(column, _)| *column).collect::<Vec<_>>();
    support.sort_unstable();
    support.dedup();
    support
}

fn support_graph_adjacency(ncols: usize, row_supports: &[Vec<usize>]) -> Vec<BTreeSet<usize>> {
    let mut adjacency = vec![BTreeSet::new(); ncols];
    for support in row_supports {
        for &column in support {
            adjacency[column].extend(
                support
                    .iter()
                    .copied()
                    .filter(|neighbor| *neighbor != column),
            );
        }
    }
    adjacency
}

fn grow_support_patch(
    patch: &mut BTreeSet<usize>,
    adjacency: &[BTreeSet<usize>],
    target_block_size: usize,
) {
    let mut frontier = patch.iter().copied().collect::<BTreeSet<_>>();
    while patch.len() < target_block_size && !frontier.is_empty() {
        let mut next_frontier = BTreeSet::new();
        for column in &frontier {
            for neighbor in &adjacency[*column] {
                if patch.insert(*neighbor) {
                    next_frontier.insert(*neighbor);
                }
            }
        }
        frontier = next_frontier;
    }
}

fn row_from_estimate(
    target: &'static str,
    method: String,
    factorization_time: Duration,
    estimator_time: Duration,
    reference: Option<&Vector>,
    estimate: &VarianceEstimate,
    status: String,
) -> BenchmarkRow {
    row_from_values(VarianceRowInput {
        target,
        method,
        sample_count: estimate.sample_count,
        factorization_time,
        estimator_time,
        reference,
        values: &estimate.values,
        relative_standard_error: estimate.relative_standard_error.as_ref(),
        status,
    })
}

struct VarianceRowInput<'a> {
    target: &'static str,
    method: String,
    sample_count: usize,
    factorization_time: Duration,
    estimator_time: Duration,
    reference: Option<&'a Vector>,
    values: &'a Vector,
    relative_standard_error: Option<&'a Vector>,
    status: String,
}

fn row_from_values(input: VarianceRowInput<'_>) -> BenchmarkRow {
    let VarianceRowInput {
        target,
        method,
        sample_count,
        factorization_time,
        estimator_time,
        reference,
        values,
        relative_standard_error,
        status,
    } = input;
    let (relative_l2_error, median_pointwise_relative_error, p95_pointwise_relative_error) =
        if let Some(reference) = reference {
            let errors = pointwise_relative_errors(reference, values);
            (
                relative_l2_error(reference, values),
                quantile(errors.clone(), 0.5),
                quantile(errors, 0.95),
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
    BenchmarkRow {
        target,
        method,
        sample_count,
        factorization_time,
        estimator_time,
        relative_l2_error,
        median_pointwise_relative_error,
        p95_pointwise_relative_error,
        num_negative: values.iter().filter(|value| **value < 0.0).count(),
        min_value: values.iter().copied().fold(f64::INFINITY, f64::min),
        batch_relative_standard_error: relative_standard_error
            .map(|stderr| {
                quantile(
                    stderr
                        .iter()
                        .copied()
                        .filter(|value| value.is_finite())
                        .collect(),
                    0.5,
                )
            })
            .unwrap_or(f64::NAN),
        selected_requested_pairs: 0,
        selected_closure_pairs: 0,
        selected_factor_pairs: 0,
        selected_closure_over_factor: f64::NAN,
        selected_closure_limit: 0,
        status,
    }
}

fn unavailable_row(
    target: &'static str,
    method: String,
    sample_count: usize,
    factorization_time: Duration,
    estimator_time: Duration,
    status: String,
) -> BenchmarkRow {
    BenchmarkRow {
        target,
        method,
        sample_count,
        factorization_time,
        estimator_time,
        relative_l2_error: f64::NAN,
        median_pointwise_relative_error: f64::NAN,
        p95_pointwise_relative_error: f64::NAN,
        num_negative: 0,
        min_value: f64::NAN,
        batch_relative_standard_error: f64::NAN,
        selected_requested_pairs: 0,
        selected_closure_pairs: 0,
        selected_factor_pairs: 0,
        selected_closure_over_factor: f64::NAN,
        selected_closure_limit: 0,
        status,
    }
}

fn relative_l2_error(reference: &Vector, values: &Vector) -> f64 {
    let numerator = reference
        .iter()
        .zip(values.iter())
        .map(|(reference, value)| {
            let diff = value - reference;
            diff * diff
        })
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    numerator / denominator.max(1e-14)
}

fn pointwise_relative_errors(reference: &Vector, values: &Vector) -> Vec<f64> {
    reference
        .iter()
        .zip(values.iter())
        .map(|(reference, value)| (value - reference).abs() / reference.abs().max(1e-14))
        .collect()
}

fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let index = ((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn print_table(rows: &[BenchmarkRow]) {
    println!(
        "{:<7} {:<24} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>5} {:>12} {:>10} {:>10} {:>10} {:>10} status",
        "target",
        "method",
        "samples",
        "est_s",
        "total_s",
        "rel_l2",
        "med_rel",
        "p95_rel",
        "neg",
        "batch_rse",
        "req_pairs",
        "closure",
        "factor",
        "clos/fact"
    );
    for row in rows {
        println!(
            "{:<7} {:<24} {:>7} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>5} {:>12.4} {:>10} {:>10} {:>10} {:>10.3} {}",
            row.target,
            row.method,
            row.sample_count,
            row.estimator_time.as_secs_f64(),
            row.total_time().as_secs_f64(),
            row.relative_l2_error,
            row.median_pointwise_relative_error,
            row.p95_pointwise_relative_error,
            row.num_negative,
            row.batch_relative_standard_error,
            row.selected_requested_pairs,
            row.selected_closure_pairs,
            row.selected_factor_pairs,
            row.selected_closure_over_factor,
            row.status
        );
    }
}

fn print_thresholds(rows: &[BenchmarkRow]) {
    if !rows
        .iter()
        .any(|row| row.relative_l2_error.is_finite() && row.method != "exact-repeated-solves")
    {
        println!("relative-L2 thresholds unavailable because exact reference was skipped");
        return;
    }
    println!(
        "smallest sample/probe count reaching relative L2 <= {:.2}",
        TARGET_RELATIVE_L2_ERROR
    );
    for target in ["latent", "d0"] {
        for method in ["monte-carlo", "hutchinson", "local-rbmc"] {
            let reached = rows
                .iter()
                .filter(|row| row.target == target && row.method == method)
                .filter(|row| row.relative_l2_error <= TARGET_RELATIVE_L2_ERROR)
                .map(|row| row.sample_count)
                .min();
            match reached {
                Some(samples) => println!("{target:<7} {method:<16} {samples}"),
                None => println!("{target:<7} {method:<16} not reached"),
            }
        }
    }
}

fn write_csv(rows: &[BenchmarkRow]) -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target")
        .join("variance_estimators.csv");
    let parent = path
        .parent()
        .ok_or_else(|| "CSV output path has no parent".to_string())?;
    create_dir_all(parent).map_err(|err| err.to_string())?;
    let mut file = File::create(&path).map_err(|err| err.to_string())?;
    writeln!(
        file,
        "target,method,sample_count,factorization_seconds,estimator_seconds,total_seconds,relative_l2_error,median_pointwise_relative_error,p95_pointwise_relative_error,num_negative,min_value,batch_relative_standard_error,selected_requested_pairs,selected_closure_pairs,selected_factor_pairs,selected_closure_over_factor,selected_closure_limit,status"
    )
    .map_err(|err| err.to_string())?;
    for row in rows {
        writeln!(
            file,
            "{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{:.9},{:.9},{},{},{},{:.9},{},{}",
            row.target,
            row.method,
            row.sample_count,
            row.factorization_time.as_secs_f64(),
            row.estimator_time.as_secs_f64(),
            row.total_time().as_secs_f64(),
            row.relative_l2_error,
            row.median_pointwise_relative_error,
            row.p95_pointwise_relative_error,
            row.num_negative,
            row.min_value,
            row.batch_relative_standard_error,
            row.selected_requested_pairs,
            row.selected_closure_pairs,
            row.selected_factor_pairs,
            row.selected_closure_over_factor,
            row.selected_closure_limit,
            row.status.replace(',', ";")
        )
        .map_err(|err| err.to_string())?;
    }
    println!("wrote {}", path.display());
    Ok(())
}
