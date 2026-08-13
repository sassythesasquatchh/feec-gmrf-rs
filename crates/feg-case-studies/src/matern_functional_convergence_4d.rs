//! Convergence experiment for fixed integral-functional Matern variances on 4D cube meshes.

use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::field::ExteriorField;
use feg_infer::{
    prior::matern::{
        generic::{build_matern_precision_form, MaternConfig as GenericMaternConfig},
        MaternAlpha,
    },
    sparse::feec_csr_to_gmrf,
};
use gmrf_core::{
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf, SparseRowOperator,
};
use manifold::{
    gen::cartesian::{cartesian_index2linear_index, CartesianMeshInfo},
    geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
    topology::{complex::Complex, handle::SimplexHandle},
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const DEFAULT_LEVELS: [usize; 3] = [3, 4, 5];
const DEFAULT_A: f64 = 0.25;
const DEFAULT_B: f64 = 0.75;
const DEFAULT_FUNCTIONAL_QUADRATURE_FACTOR: usize = 2;
const EPS: f64 = 1e-12;
const COORD_TOL: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct FunctionalConvergence4dConfig {
    pub levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub functional_quadrature_factor: usize,
}

impl Default for FunctionalConvergence4dConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            levels: DEFAULT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/matern_functional_convergence_4d"),
            kappa: 4.0,
            tau: 1.0,
            functional_quadrature_factor: DEFAULT_FUNCTIONAL_QUADRATURE_FACTOR,
        }
    }
}

impl FunctionalConvergence4dConfig {
    /// Cheap deterministic configuration intended for continuous integration.
    pub fn smoke(output_dir: PathBuf) -> Self {
        Self {
            levels: vec![1],
            output_dir,
            functional_quadrature_factor: 1,
            ..Self::default()
        }
    }

    /// Immutable configuration used by the submitted thesis.
    pub fn thesis_submitted(output_dir: PathBuf) -> Self {
        Self {
            levels: DEFAULT_LEVELS.to_vec(),
            output_dir,
            kappa: 4.0,
            tau: 1.0,
            functional_quadrature_factor: DEFAULT_FUNCTIONAL_QUADRATURE_FACTOR,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FixedObservable4d {
    pub grade: usize,
    pub label: &'static str,
    pub row: Vec<(usize, f64)>,
    pub measure: f64,
}

#[derive(Debug, Clone)]
pub struct VarianceRow4d {
    pub n: usize,
    pub h: f64,
    pub grade: usize,
    pub alpha: MaternAlpha,
    pub dofs: usize,
    pub precision_nnz: usize,
    pub precision_density: f64,
    pub precision_lower_nnz: usize,
    pub cholesky_factor_nnz: usize,
    pub cholesky_factor_density: f64,
    pub fill_in_ratio: f64,
    pub factor_seconds: f64,
    pub variance_seconds: f64,
    pub support_entries: usize,
    pub raw_variance: f64,
    pub normalized_variance: f64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct FitSummaryRow4d {
    pub grade: usize,
    pub alpha: MaternAlpha,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct StageReport4d {
    pub label: String,
    pub completed_levels: Vec<usize>,
    pub next_level: Option<usize>,
    pub decision: String,
    pub rows: usize,
    pub fit_rows: usize,
}

#[derive(Debug, Clone)]
pub struct FunctionalConvergence4dResult {
    pub rows: Vec<VarianceRow4d>,
    pub fit_summaries: Vec<FitSummaryRow4d>,
    pub stage_reports: Vec<StageReport4d>,
}

pub fn default_levels() -> Vec<usize> {
    DEFAULT_LEVELS.to_vec()
}

pub fn run_functional_convergence_4d_experiment(
    config: &FunctionalConvergence4dConfig,
) -> Result<FunctionalConvergence4dResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut rows = Vec::new();
    let mut stage_reports = Vec::new();
    for (level_index, &n) in config.levels.iter().enumerate() {
        eprintln!("[matern_functional_convergence_4d] mesh n={n}");
        run_level(n, config, &mut rows)?;

        let fit_summaries = fit_summaries(&rows);
        stage_reports.push(StageReport4d {
            label: format!("level_{n}_complete"),
            completed_levels: config.levels[..=level_index].to_vec(),
            next_level: config.levels.get(level_index + 1).copied(),
            decision: "flushed after mesh level".to_string(),
            rows: rows.len(),
            fit_rows: fit_summaries.len(),
        });
        write_outputs(config, &rows, &fit_summaries, &stage_reports)?;
    }

    let fit_summaries = fit_summaries(&rows);

    Ok(FunctionalConvergence4dResult {
        rows,
        fit_summaries,
        stage_reports,
    })
}

fn run_level(
    n: usize,
    config: &FunctionalConvergence4dConfig,
    rows: &mut Vec<VarianceRow4d>,
) -> Result<(), Box<dyn Error>> {
    run_level_subset(
        n,
        config,
        rows,
        0..=4,
        &[MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three],
    )
}

fn run_level_subset(
    n: usize,
    config: &FunctionalConvergence4dConfig,
    rows: &mut Vec<VarianceRow4d>,
    grades: impl IntoIterator<Item = usize>,
    alphas: &[MaternAlpha],
) -> Result<(), Box<dyn Error>> {
    let mesh = CartesianMeshInfo::new_unit_scaled(4, n, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let quadrature_points_per_axis = config.functional_quadrature_factor.saturating_mul(n).max(1);

    for grade in grades {
        let observable =
            fixed_simplex_observable_4d(&topology, &coords, n, grade, quadrature_points_per_axis)?;
        for &alpha in alphas {
            let precision = build_matern_precision_form(
                &topology,
                &metric,
                grade,
                alpha,
                GenericMaternConfig {
                    kappa: config.kappa,
                    tau: config.tau,
                },
            )?;
            let precision_nnz = precision.nnz();
            let precision_lower_nnz = lower_triangle_nnz(&precision);
            let dimension = topology.nsimplices(grade);
            let variance_report =
                exact_observable_variance(&precision, dimension, &observable.row)?;
            let normalized_variance =
                variance_report.raw_variance / observable.measure.max(EPS).powi(2);
            rows.push(VarianceRow4d {
                n,
                h: 1.0 / n as f64,
                grade,
                alpha,
                dofs: precision.nrows(),
                precision_nnz,
                precision_density: matrix_density(precision_nnz, dimension, dimension),
                precision_lower_nnz,
                cholesky_factor_nnz: variance_report.cholesky_factor_nnz,
                cholesky_factor_density: triangular_density(
                    variance_report.cholesky_factor_nnz,
                    dimension,
                ),
                fill_in_ratio: variance_report.cholesky_factor_nnz as f64
                    / precision_lower_nnz.max(1) as f64,
                factor_seconds: variance_report.factor_seconds,
                variance_seconds: variance_report.variance_seconds,
                support_entries: observable.row.len(),
                raw_variance: variance_report.raw_variance,
                normalized_variance,
                status: status_for(grade, alpha),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ExactVarianceReport {
    raw_variance: f64,
    cholesky_factor_nnz: usize,
    factor_seconds: f64,
    variance_seconds: f64,
}

fn exact_observable_variance(
    precision: &FeecCsr<f64>,
    dimension: usize,
    row: &[(usize, f64)],
) -> Result<ExactVarianceReport, String> {
    let precision_gmrf = feec_csr_to_gmrf(precision);
    let factor_start = Instant::now();
    let factor = precision_gmrf
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision: {err}"))?;
    let factor_seconds = factor_start.elapsed().as_secs_f64();
    let cholesky_factor_nnz = factor.nnz();
    let mut gmrf = Gmrf::from_mean_and_precision(GmrfVector::zeros(dimension), precision_gmrf)
        .map_err(|err| format!("failed to build prior GMRF: {err}"))?
        .with_precision_sqrt(factor);
    let operator = SparseRowOperator::new(dimension, vec![row.to_vec()])
        .map_err(|err| format!("failed to build observable operator: {err}"))?;
    let constraints = GmrfDenseMatrix::zeros(0, dimension);
    let variance_start = Instant::now();
    let variance = gmrf
        .exact_transformed_variance_decomposition(&operator, &constraints)
        .map_err(|err| format!("failed to compute transformed variance: {err}"))?;
    Ok(ExactVarianceReport {
        raw_variance: variance.unconstrained_diag[0],
        cholesky_factor_nnz,
        factor_seconds,
        variance_seconds: variance_start.elapsed().as_secs_f64(),
    })
}

pub fn fixed_simplex_observable_4d(
    topology: &Complex,
    coords: &MeshCoords,
    mesh_level: usize,
    grade: usize,
    quadrature_points_per_axis: usize,
) -> Result<FixedObservable4d, String> {
    if grade > 4 {
        return Err(format!("unsupported observable grade {grade}"));
    }

    let vertices = embedded_simplex_vertices(grade);
    let samples = standard_simplex_midpoint_samples(grade, quadrature_points_per_axis);
    let mut accum = BTreeMap::<usize, f64>::new();

    for (reference_point, weight) in samples {
        let point = embedded_simplex_point(&vertices, &reference_point);
        let point_vec = vec4(point);
        let cell = locate_cartesian_kuhn_cell(topology, coords, mesh_level, &point_vec)?;
        let cell_coords = cell.coord_simplex(coords);
        let local = cell_coords.global2local(point_vec.as_view());
        let tangent_local = embedded_tangent_local(&cell_coords.inv_linear_transform(), &vertices);

        for dof_simp in cell.mesh_subsimps(grade) {
            let local_dof = dof_simp.relative_to(&cell);
            let local_value =
                WhitneyLsf::standard(topology.dim(), local_dof).at_point(local.as_view());
            let coefficient = if grade == 0 {
                local_value.coeffs()[0]
            } else {
                local_value.precompose_form(&tangent_local).coeffs()[0]
            };
            if coefficient.abs() > EPS {
                *accum.entry(dof_simp.kidx()).or_insert(0.0) += weight * coefficient;
            }
        }
    }

    let row = map_to_row(accum);
    if row.is_empty() {
        return Err(format!(
            "fixed {} observable selected no mesh simplices",
            observable_label(grade)
        ));
    }
    Ok(FixedObservable4d {
        grade,
        label: observable_label(grade),
        row,
        measure: simplex_measure(grade),
    })
}

fn observable_label(grade: usize) -> &'static str {
    match grade {
        0 => "point",
        1 => "line",
        2 => "triangle",
        3 => "tetrahedron",
        4 => "4-simplex",
        _ => "unsupported",
    }
}

fn embedded_simplex_vertices(grade: usize) -> Vec<[f64; 4]> {
    (0..=grade)
        .map(|vertex| {
            let mut point = [DEFAULT_A; 4];
            for component in point.iter_mut().take(vertex) {
                *component = DEFAULT_B;
            }
            point
        })
        .collect()
}

fn embedded_simplex_point(vertices: &[[f64; 4]], reference_point: &[f64]) -> [f64; 4] {
    let mut point = vertices[0];
    for (axis, &coordinate) in reference_point.iter().enumerate() {
        let edge = sub4(vertices[axis + 1], vertices[0]);
        for component in 0..4 {
            point[component] += coordinate * edge[component];
        }
    }
    point
}

fn embedded_tangent_local(inv_cell_transform: &FeecMatrix, vertices: &[[f64; 4]]) -> FeecMatrix {
    let grade = vertices.len() - 1;
    let mut physical_tangent = FeecMatrix::zeros(4, grade);
    for axis in 0..grade {
        let edge = sub4(vertices[axis + 1], vertices[0]);
        for component in 0..4 {
            physical_tangent[(component, axis)] = edge[component];
        }
    }
    inv_cell_transform * physical_tangent
}

fn standard_simplex_midpoint_samples(dim: usize, points_per_axis: usize) -> Vec<(Vec<f64>, f64)> {
    if dim == 0 {
        return vec![(Vec::new(), 1.0)];
    }

    let points_per_axis = points_per_axis.max(1);
    let sample_count = points_per_axis.pow(dim as u32);
    let mut samples = Vec::with_capacity(sample_count);
    for linear_index in 0..sample_count {
        let mut remaining_index = linear_index;
        let mut cube_point = vec![0.0; dim];
        for coordinate in cube_point.iter_mut() {
            let index = remaining_index % points_per_axis;
            remaining_index /= points_per_axis;
            *coordinate = (index as f64 + 0.5) / points_per_axis as f64;
        }

        let mut reference_point = vec![0.0; dim];
        let mut remaining = 1.0;
        for (reference, &cube_coordinate) in reference_point.iter_mut().zip(&cube_point) {
            *reference = remaining * cube_coordinate;
            remaining *= 1.0 - cube_coordinate;
        }

        let mut jacobian = 1.0;
        for (axis, &cube_coordinate) in cube_point.iter().enumerate().take(dim.saturating_sub(1)) {
            jacobian *= (1.0 - cube_coordinate).powi((dim - axis - 1) as i32);
        }
        samples.push((
            reference_point,
            jacobian / (points_per_axis as f64).powi(dim as i32),
        ));
    }

    let target_volume = 1.0 / factorial_usize(dim) as f64;
    let total_weight = samples.iter().map(|(_, weight)| *weight).sum::<f64>();
    for (_, weight) in &mut samples {
        *weight *= target_volume / total_weight.max(EPS);
    }
    samples
}

fn locate_cartesian_kuhn_cell<'a>(
    topology: &'a Complex,
    coords: &MeshCoords,
    mesh_level: usize,
    point: &FeecVector,
) -> Result<SimplexHandle<'a>, String> {
    if topology.dim() == 4 && coords.dim() == 4 && mesh_level > 0 {
        let mut cartesian_index = FeecVector::<usize>::zeros(4);
        let mut fractional = [0.0; 4];
        for component in 0..4 {
            let scaled = point[component] * mesh_level as f64;
            if scaled < -COORD_TOL || scaled > mesh_level as f64 + COORD_TOL {
                return Err(format!(
                    "quadrature point lies outside unit cube: {point:?}"
                ));
            }
            let clamped = scaled.clamp(0.0, mesh_level as f64);
            let cell_index = clamped.floor().min((mesh_level - 1) as f64) as usize;
            cartesian_index[component] = cell_index;
            fractional[component] = clamped - cell_index as f64;
        }

        let mut permutation = [0usize, 1, 2, 3];
        permutation.sort_by(|&lhs, &rhs| {
            fractional[rhs]
                .partial_cmp(&fractional[lhs])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| lhs.cmp(&rhs))
        });
        let box_index = cartesian_index2linear_index(cartesian_index, mesh_level);
        let cell_index =
            box_index * factorial_usize(4) + permutation_rank_lexicographic(&permutation);
        if cell_index < topology.cells().len() {
            let cell = topology.cells().handle_by_kidx(cell_index);
            if cell_contains_point_tolerant(cell, coords, point) {
                return Ok(cell);
            }
        }
    }

    topology
        .cells()
        .handle_iter()
        .find(|&cell| cell_contains_point_tolerant(cell, coords, point))
        .ok_or_else(|| format!("failed to locate quadrature point in mesh: {point:?}"))
}

fn cell_contains_point_tolerant(
    cell: SimplexHandle<'_>,
    coords: &MeshCoords,
    point: &FeecVector,
) -> bool {
    let bary = cell.coord_simplex(coords).global2bary(point.as_view());
    (bary.sum() - 1.0).abs() <= 1e-8
        && bary
            .iter()
            .all(|&coordinate| (-1e-9..=1.0 + 1e-9).contains(&coordinate))
}

fn permutation_rank_lexicographic(permutation: &[usize]) -> usize {
    let mut remaining = (0..permutation.len()).collect::<Vec<_>>();
    let mut rank = 0;
    for (position, value) in permutation.iter().enumerate() {
        let index = remaining
            .iter()
            .position(|candidate| candidate == value)
            .expect("permutation entries must be valid");
        rank += index * factorial_usize(permutation.len() - position - 1);
        remaining.remove(index);
    }
    rank
}

fn map_to_row(accum: BTreeMap<usize, f64>) -> Vec<(usize, f64)> {
    accum
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect()
}

fn vec4(point: [f64; 4]) -> FeecVector {
    FeecVector::from_vec(point.to_vec())
}

fn simplex_measure(grade: usize) -> f64 {
    (DEFAULT_B - DEFAULT_A).powi(grade as i32) / factorial_usize(grade) as f64
}

fn factorial_usize(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

fn lower_triangle_nnz(precision: &FeecCsr<f64>) -> usize {
    precision
        .triplet_iter()
        .filter(|(row, col, value)| row >= col && value.abs() > 0.0)
        .count()
}

fn matrix_density(nnzs: usize, rows: usize, cols: usize) -> f64 {
    nnzs as f64 / rows.saturating_mul(cols).max(1) as f64
}

fn triangular_density(nnzs: usize, dimension: usize) -> f64 {
    let entries = dimension.saturating_mul(dimension + 1) / 2;
    nnzs as f64 / entries.max(1) as f64
}

fn status_for(grade: usize, alpha: MaternAlpha) -> String {
    let codimension = 4_i32 - grade as i32;
    let smoothing = 2_i32 * alpha.as_u32() as i32;
    if smoothing < codimension {
        format!("expected_h^-{}_divergence", codimension - smoothing)
    } else if smoothing == codimension {
        "expected_log_borderline".to_string()
    } else {
        "expected_convergent".to_string()
    }
}

fn fit_summaries(rows: &[VarianceRow4d]) -> Vec<FitSummaryRow4d> {
    let mut grouped: BTreeMap<(usize, MaternAlpha), Vec<&VarianceRow4d>> = BTreeMap::new();
    for row in rows {
        grouped.entry((row.grade, row.alpha)).or_default().push(row);
    }

    let mut summaries = Vec::new();
    for ((grade, alpha), mut group) in grouped {
        group.sort_by_key(|row| row.n);
        if group.len() < 2 {
            continue;
        }

        let codimension = 4_i32 - grade as i32;
        let smoothing = 2_i32 * alpha.as_u32() as i32;
        if smoothing < codimension {
            let exponent = codimension - smoothing;
            let xs = group
                .iter()
                .map(|row| (row.n as f64).ln())
                .collect::<Vec<_>>();
            let ys = group
                .iter()
                .map(|row| row.raw_variance.max(EPS).ln())
                .collect::<Vec<_>>();
            summaries.push(FitSummaryRow4d {
                grade,
                alpha,
                diagnostic: "loglog_slope_vs_n".to_string(),
                value: linear_slope(&xs, &ys),
                expected: format!("near +{exponent}"),
                status: "divergence_diagnostic".to_string(),
            });
        } else if smoothing == codimension {
            let xs = group
                .iter()
                .map(|row| (row.n as f64).ln())
                .collect::<Vec<_>>();
            let ys = group.iter().map(|row| row.raw_variance).collect::<Vec<_>>();
            summaries.push(FitSummaryRow4d {
                grade,
                alpha,
                diagnostic: "slope_vs_log_n".to_string(),
                value: linear_slope(&xs, &ys),
                expected: "positive".to_string(),
                status: "borderline_log_diagnostic".to_string(),
            });
        } else {
            let prev = group[group.len() - 2].raw_variance;
            let last = group[group.len() - 1].raw_variance;
            summaries.push(FitSummaryRow4d {
                grade,
                alpha,
                diagnostic: "finest_relative_change".to_string(),
                value: (last - prev).abs() / last.abs().max(EPS),
                expected: "small under convergence".to_string(),
                status: "convergence_diagnostic".to_string(),
            });
        }
    }
    summaries
}

fn linear_slope(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let numerator = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    let denominator = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();
    numerator / denominator.max(EPS)
}

fn write_outputs(
    config: &FunctionalConvergence4dConfig,
    rows: &[VarianceRow4d],
    fit_summaries: &[FitSummaryRow4d],
    stage_reports: &[StageReport4d],
) -> io::Result<()> {
    write_variance_csv(&config.output_dir.join("functional_variance.csv"), rows)?;
    write_fit_summary_csv(&config.output_dir.join("fit_summary.csv"), fit_summaries)?;
    write_stage_report_csv(&config.output_dir.join("stage_report.csv"), stage_reports)?;
    write_readme(&config.output_dir.join("README.md"), config)?;
    Ok(())
}

pub fn write_variance_csv(path: &Path, rows: &[VarianceRow4d]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,h,grade,alpha,dofs,precision_nnz,precision_density,precision_lower_nnz,cholesky_factor_nnz,cholesky_factor_density,fill_in_ratio,factor_seconds,variance_seconds,support_entries,raw_variance,normalized_variance,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17},{},{},{},{},{:.17e},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{}",
            row.n,
            row.h,
            row.grade,
            row.alpha.as_u32(),
            row.dofs,
            row.precision_nnz,
            row.precision_density,
            row.precision_lower_nnz,
            row.cholesky_factor_nnz,
            row.cholesky_factor_density,
            row.fill_in_ratio,
            row.factor_seconds,
            row.variance_seconds,
            row.support_entries,
            row.raw_variance,
            row.normalized_variance,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_fit_summary_csv(path: &Path, rows: &[FitSummaryRow4d]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "grade,alpha,diagnostic,value,expected,status")?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{:.17e},{},{}",
            row.grade,
            row.alpha.as_u32(),
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_stage_report_csv(path: &Path, rows: &[StageReport4d]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "label,completed_levels,next_level,decision,rows,fit_rows"
    )?;
    for row in rows {
        let completed = row
            .completed_levels
            .iter()
            .map(|level| level.to_string())
            .collect::<Vec<_>>()
            .join("|");
        let next = row
            .next_level
            .map(|level| level.to_string())
            .unwrap_or_default();
        writeln!(
            writer,
            "{},{},{},{},{},{}",
            row.label, completed, next, row.decision, row.rows, row.fit_rows
        )?;
    }
    Ok(())
}

pub fn write_readme(path: &Path, config: &FunctionalConvergence4dConfig) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# 4D Hodge-Matern Functional Variance Convergence")?;
    writeln!(writer)?;
    writeln!(writer, "- levels: {:?}", config.levels)?;
    writeln!(writer, "- kappa: {:.17}", config.kappa)?;
    writeln!(writer, "- tau: {:.17}", config.tau)?;
    writeln!(writer, "- fixed coordinates: a={DEFAULT_A}, b={DEFAULT_B}")?;
    writeln!(
        writer,
        "- quadrature points per axis: mesh level * {}",
        config.functional_quadrature_factor
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "| alpha | k=0 point | k=1 line | k=2 area | k=3 volume | k=4 hypervolume |"
    )?;
    writeln!(writer, "|---|---|---|---|---|---|")?;
    writeln!(
        writer,
        "| 1 | diverges like h^-2 | diverges like h^-1 | borderline log h^-1 | converges | converges |"
    )?;
    writeln!(
        writer,
        "| 2 | borderline log h^-1 | converges | converges | converges | converges |"
    )?;
    writeln!(
        writer,
        "| 3 | converges | converges | converges | converges | converges |"
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "The primary CSV reports raw integral variances, precision/factor densities, Cholesky fill-in, factorization timings, and exact transformed-variance timings. Normalized variances divide by the fixed observable measure squared."
    )?;
    Ok(())
}

fn validate_config(config: &FunctionalConvergence4dConfig) -> Result<(), String> {
    if config.levels.is_empty() {
        return Err("at least one mesh level is required".to_string());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if config.functional_quadrature_factor == 0 {
        return Err("functional_quadrature_factor must be positive".to_string());
    }
    for &level in &config.levels {
        if level == 0 {
            return Err(format!(
                "mesh level {level} is invalid; levels must be positive"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
fn constant_form_cochain(topology: &Complex, coords: &MeshCoords, grade: usize) -> FeecVector {
    FeecVector::from_iterator(
        topology.nsimplices(grade),
        topology.skeleton(grade).handle_iter().map(|simplex| {
            let points = simplex
                .vertices
                .iter()
                .map(|&vertex| coord4(coords, vertex))
                .collect::<Vec<_>>();
            match grade {
                0 => 1.0,
                1 => points[1][0] - points[0][0],
                2 => signed_xy_area(points[0], points[1], points[2]),
                3 => signed_volume_xyz(points[0], points[1], points[2], points[3]),
                4 => signed_hypervolume(points[0], points[1], points[2], points[3], points[4]),
                _ => 0.0,
            }
        }),
    )
}

#[cfg(test)]
fn apply_row(row: &[(usize, f64)], values: &FeecVector) -> f64 {
    row.iter().map(|(col, weight)| *weight * values[*col]).sum()
}

#[cfg(test)]
fn coord4(coords: &MeshCoords, vertex: usize) -> [f64; 4] {
    let coord = coords.coord(vertex);
    [
        coord[0],
        if coords.dim() > 1 { coord[1] } else { 0.0 },
        if coords.dim() > 2 { coord[2] } else { 0.0 },
        if coords.dim() > 3 { coord[3] } else { 0.0 },
    ]
}

#[cfg(test)]
fn signed_xy_area(a: [f64; 4], b: [f64; 4], c: [f64; 4]) -> f64 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

#[cfg(test)]
fn signed_volume_xyz(a: [f64; 4], b: [f64; 4], c: [f64; 4], d: [f64; 4]) -> f64 {
    let ab = sub4(b, a);
    let ac = sub4(c, a);
    let ad = sub4(d, a);
    det3([
        [ab[0], ac[0], ad[0]],
        [ab[1], ac[1], ad[1]],
        [ab[2], ac[2], ad[2]],
    ]) / 6.0
}

#[cfg(test)]
fn signed_hypervolume(a: [f64; 4], b: [f64; 4], c: [f64; 4], d: [f64; 4], e: [f64; 4]) -> f64 {
    let ab = sub4(b, a);
    let ac = sub4(c, a);
    let ad = sub4(d, a);
    let ae = sub4(e, a);
    det4([
        [ab[0], ac[0], ad[0], ae[0]],
        [ab[1], ac[1], ad[1], ae[1]],
        [ab[2], ac[2], ad[2], ae[2]],
        [ab[3], ac[3], ad[3], ae[3]],
    ]) / 24.0
}

fn sub4(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[cfg(test)]
fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[cfg(test)]
fn det4(m: [[f64; 4]; 4]) -> f64 {
    let mut det = 0.0;
    for col in 0..4 {
        let mut minor = [[0.0; 3]; 3];
        for row in 1..4 {
            let mut minor_col = 0;
            for (candidate_col, &value) in m[row].iter().enumerate() {
                if candidate_col == col {
                    continue;
                }
                minor[row - 1][minor_col] = value;
                minor_col += 1;
            }
        }
        let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * m[0][col] * det3(minor);
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold::geometry::metric::mesh::MeshLengths;

    fn mesh(level: usize) -> (Complex, MeshCoords, MeshLengths) {
        let mesh = CartesianMeshInfo::new_unit_scaled(4, level, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        (topology, coords, metric)
    }

    #[test]
    fn matern_functional_convergence_4d_observable_supports_are_nonempty() {
        let level = 3;
        let (topology, coords, _metric) = mesh(level);
        for grade in 0..=4 {
            let observable = fixed_simplex_observable_4d(&topology, &coords, level, grade, 8)
                .expect("fixed observable should build");
            assert_eq!(observable.grade, grade);
            assert!(!observable.row.is_empty());
        }
    }

    #[test]
    fn matern_functional_convergence_4d_observables_integrate_constant_forms() {
        let level = 3;
        let (topology, coords, _metric) = mesh(level);
        let expected = [
            1.0,
            DEFAULT_B - DEFAULT_A,
            (DEFAULT_B - DEFAULT_A).powi(2) / 2.0,
            (DEFAULT_B - DEFAULT_A).powi(3) / 6.0,
            (DEFAULT_B - DEFAULT_A).powi(4) / 24.0,
        ];

        for (grade, &expected_value) in expected.iter().enumerate() {
            let observable = fixed_simplex_observable_4d(&topology, &coords, level, grade, 8)
                .expect("fixed observable should build");
            let values = constant_form_cochain(&topology, &coords, grade);
            let actual = apply_row(&observable.row, &values);
            assert!(
                (actual - expected_value).abs() <= 1e-12,
                "grade {grade}: expected {expected_value}, got {actual}"
            );
        }
    }

    #[test]
    fn matern_functional_convergence_4d_exact_variance_smoke_is_finite() {
        let config = FunctionalConvergence4dConfig {
            levels: vec![4],
            output_dir: std::env::temp_dir().join(format!(
                "matern_functional_convergence_4d_test_{}",
                std::process::id()
            )),
            kappa: 4.0,
            tau: 1.0,
            functional_quadrature_factor: 2,
        };
        let mut rows = Vec::new();
        run_level_subset(4, &config, &mut rows, [0], &[MaternAlpha::Three])
            .expect("representative 4D exact variance should run");

        assert_eq!(rows.len(), 1);
        assert!(rows.iter().all(|row| {
            row.raw_variance.is_finite()
                && row.raw_variance >= 0.0
                && row.normalized_variance.is_finite()
                && row.support_entries > 0
                && row.dofs > 0
        }));
    }
}
