//! Submitted scalar 4D Matérn study with exact nested point, line, and area averages.

use common::linalg::nalgebra::Vector as FeecVector;
use feec_gmrf::prelude::{
    DerivedQuantity, GaussianPrior, LinearGaussianModelBuilder, LinearMap, MaternAlpha,
    MaternParameters, MaternPriorBuilder, SparseMat,
};
use manifold::gen::cartesian::{cartesian_index2linear_index, CartesianMeshInfo};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const DEFAULT_LEVELS: [usize; 3] = [4, 8, 12];
const THESIS_SUBMITTED_LEVELS: [usize; 4] = [4, 8, 12, 16];
const DEFAULT_COORD_NUMERATORS: [usize; 3] = [1, 2, 3];
const COMMON_DENOMINATOR: usize = 4;
const LINE_MEASURE: f64 = 0.5;
const AREA_MEASURE: f64 = 0.125;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct ScalarBorderline4dConfig {
    pub levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
}

impl Default for ScalarBorderline4dConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            levels: DEFAULT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/matern_scalar_borderline_4d"),
            kappa: 4.0,
            tau: 1.0,
        }
    }
}

impl ScalarBorderline4dConfig {
    /// Cheap profile for registry and CI validation.
    pub fn smoke(output_dir: PathBuf) -> Self {
        Self {
            levels: vec![4],
            output_dir,
            kappa: 4.0,
            tau: 1.0,
        }
    }

    /// Immutable configuration used for the submitted thesis artifacts.
    pub fn thesis_submitted(output_dir: PathBuf) -> Self {
        Self {
            levels: THESIS_SUBMITTED_LEVELS.to_vec(),
            output_dir,
            kappa: 4.0,
            tau: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScalarFunctionalKind {
    AreaAverage,
    LineAverage,
    Point,
}

impl ScalarFunctionalKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::LineAverage => "line_average",
            Self::AreaAverage => "area_average",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalarFunctional4d {
    pub kind: ScalarFunctionalKind,
    pub id: String,
    pub row: Vec<(usize, f64)>,
    pub geometric_measure: f64,
}

#[derive(Debug, Clone)]
pub struct ScalarVarianceRow4d {
    pub n: usize,
    pub h: f64,
    pub alpha: MaternAlpha,
    pub functional_kind: ScalarFunctionalKind,
    pub functional_id: String,
    pub dofs: usize,
    pub precision_nnz: usize,
    pub precision_density: f64,
    pub precision_lower_nnz: usize,
    pub cholesky_factor_nnz: usize,
    pub cholesky_factor_density: f64,
    pub fill_in_ratio: f64,
    pub factor_seconds: f64,
    pub variance_seconds_total: f64,
    pub support_entries: usize,
    pub geometric_measure: f64,
    pub average_variance: f64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct ScalarSummaryRow4d {
    pub n: usize,
    pub alpha: MaternAlpha,
    pub functional_kind: ScalarFunctionalKind,
    pub count: usize,
    pub mean_variance: f64,
    pub median_variance: f64,
    pub min_variance: f64,
    pub max_variance: f64,
    pub stddev_variance: f64,
}

#[derive(Debug, Clone)]
pub struct ScalarFitSummaryRow4d {
    pub alpha: MaternAlpha,
    pub functional_kind: ScalarFunctionalKind,
    pub statistic: String,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct ScalarStageReport4d {
    pub label: String,
    pub completed_levels: Vec<usize>,
    pub next_level: Option<usize>,
    pub decision: String,
    pub variance_rows: usize,
    pub summary_rows: usize,
    pub fit_rows: usize,
}

#[derive(Debug, Clone)]
pub struct ScalarBorderline4dResult {
    pub rows: Vec<ScalarVarianceRow4d>,
    pub summaries: Vec<ScalarSummaryRow4d>,
    pub fit_summaries: Vec<ScalarFitSummaryRow4d>,
    pub stage_reports: Vec<ScalarStageReport4d>,
}

pub fn default_levels() -> Vec<usize> {
    DEFAULT_LEVELS.to_vec()
}

pub fn run_scalar_borderline_4d_experiment(
    config: &ScalarBorderline4dConfig,
) -> Result<ScalarBorderline4dResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut rows = Vec::new();
    let mut stage_reports = Vec::new();
    for (level_index, &n) in config.levels.iter().enumerate() {
        eprintln!("[matern_scalar_borderline_4d] mesh n={n}");
        run_level(n, config, &mut rows)?;

        let summaries = summary_rows(&rows);
        let fit_summaries = fit_summary_rows(&summaries);
        stage_reports.push(ScalarStageReport4d {
            label: format!("level_{n}_complete"),
            completed_levels: config.levels[..=level_index].to_vec(),
            next_level: config.levels.get(level_index + 1).copied(),
            decision: "flushed after mesh level".to_string(),
            variance_rows: rows.len(),
            summary_rows: summaries.len(),
            fit_rows: fit_summaries.len(),
        });
        write_outputs(config, &rows, &summaries, &fit_summaries, &stage_reports)?;
    }

    let summaries = summary_rows(&rows);
    let fit_summaries = fit_summary_rows(&summaries);
    Ok(ScalarBorderline4dResult {
        rows,
        summaries,
        fit_summaries,
        stage_reports,
    })
}

fn run_level(
    n: usize,
    config: &ScalarBorderline4dConfig,
    rows: &mut Vec<ScalarVarianceRow4d>,
) -> Result<(), Box<dyn Error>> {
    let mesh = CartesianMeshInfo::new_unit_scaled(4, n, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let functionals = build_exact_nested_scalar_functionals(n)?;

    for alpha in [MaternAlpha::Two, MaternAlpha::Three] {
        let prior = MaternPriorBuilder::from_feec(&topology, &metric, 0)
            .map_err(|error| error.to_string())?
            .parameters(
                MaternParameters::new(alpha, config.kappa, config.tau)
                    .map_err(|error| error.to_string())?,
            )
            .build()
            .map_err(|error| error.to_string())?;
        let precision_nnz = prior.precision().nnz();
        let precision_lower_nnz = lower_triangle_nnz(prior.precision());
        let dimension = topology.nsimplices(0);
        let report = exact_functional_variances(prior, dimension, &functionals)?;

        for (functional, average_variance) in functionals.iter().zip(report.variances.iter()) {
            rows.push(ScalarVarianceRow4d {
                n,
                h: 1.0 / n as f64,
                alpha,
                functional_kind: functional.kind,
                functional_id: functional.id.clone(),
                dofs: dimension,
                precision_nnz,
                precision_density: matrix_density(precision_nnz, dimension, dimension),
                precision_lower_nnz,
                cholesky_factor_nnz: report.cholesky_factor_nnz,
                cholesky_factor_density: triangular_density(report.cholesky_factor_nnz, dimension),
                fill_in_ratio: report.cholesky_factor_nnz as f64
                    / precision_lower_nnz.max(1) as f64,
                factor_seconds: report.factor_seconds,
                variance_seconds_total: report.variance_seconds,
                support_entries: functional.row.len(),
                geometric_measure: functional.geometric_measure,
                average_variance: *average_variance,
                status: status_for(functional.kind, alpha).to_string(),
            });
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct ExactVarianceBatchReport {
    variances: Vec<f64>,
    cholesky_factor_nnz: usize,
    factor_seconds: f64,
    variance_seconds: f64,
}

fn exact_functional_variances(
    prior: GaussianPrior,
    dimension: usize,
    functionals: &[ScalarFunctional4d],
) -> Result<ExactVarianceBatchReport, String> {
    if prior.dimension() != dimension {
        return Err(format!(
            "scalar prior dimension {} does not match functional dimension {dimension}",
            prior.dimension()
        ));
    }
    let operator = LinearMap::new(
        SparseMat::from_rows(
            dimension,
            &functionals
                .iter()
                .map(|functional| functional.row.clone())
                .collect::<Vec<_>>(),
        )
        .map_err(|error| format!("failed to build scalar functional operator: {error}"))?,
    )
    .map_err(|error| format!("failed to build scalar functional operator: {error}"))?;
    let derived = DerivedQuantity::new("scalar-functionals", operator)
        .map_err(|error| format!("failed to build scalar functional output: {error}"))?;
    let factor_start = Instant::now();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .derive(derived)
        .map_err(|error| format!("failed to register scalar functionals: {error}"))?
        .condition()
        .map_err(|error| format!("failed to factor scalar prior precision: {error}"))?;
    let factor_seconds = factor_start.elapsed().as_secs_f64();
    let cholesky_factor_nnz = posterior
        .precision_factor()
        .ok_or_else(|| "scalar posterior did not retain its precision factor".to_string())?
        .nnz();
    let variance_start = Instant::now();
    let variances = posterior
        .derived_variances("scalar-functionals")
        .map_err(|error| format!("failed to compute scalar transformed variances: {error}"))?;
    Ok(ExactVarianceBatchReport {
        variances,
        cholesky_factor_nnz,
        factor_seconds,
        variance_seconds: variance_start.elapsed().as_secs_f64(),
    })
}

pub fn build_exact_nested_scalar_functionals(
    level: usize,
) -> Result<Vec<ScalarFunctional4d>, String> {
    validate_level(level)?;
    let mut functionals = Vec::new();
    functionals.extend(area_average_functionals(level));
    functionals.extend(line_average_functionals(level));
    functionals.extend(point_functionals(level));
    Ok(functionals)
}

fn point_functionals(level: usize) -> Vec<ScalarFunctional4d> {
    let mut functionals = Vec::new();
    let mut id = 0;
    for x0 in DEFAULT_COORD_NUMERATORS {
        for x1 in DEFAULT_COORD_NUMERATORS {
            for x2 in DEFAULT_COORD_NUMERATORS {
                for x3 in DEFAULT_COORD_NUMERATORS {
                    let indices = [
                        coord_index(level, x0),
                        coord_index(level, x1),
                        coord_index(level, x2),
                        coord_index(level, x3),
                    ];
                    functionals.push(ScalarFunctional4d {
                        kind: ScalarFunctionalKind::Point,
                        id: format!("point_{id:03}_q{x0}{x1}{x2}{x3}"),
                        row: vec![(vertex_index(level, indices), 1.0)],
                        geometric_measure: 1.0,
                    });
                    id += 1;
                }
            }
        }
    }
    functionals
}

fn line_average_functionals(level: usize) -> Vec<ScalarFunctional4d> {
    let mut functionals = Vec::new();
    let mut id = 0;
    let start = coord_index(level, 1);
    let end = coord_index(level, 3);
    let edge_weight = (1.0 / level as f64) / (2.0 * LINE_MEASURE);

    for direction in 0..4 {
        let fixed_axes = complement_axes3(direction);
        for fixed0 in DEFAULT_COORD_NUMERATORS {
            for fixed1 in DEFAULT_COORD_NUMERATORS {
                for fixed2 in DEFAULT_COORD_NUMERATORS {
                    let fixed = [fixed0, fixed1, fixed2];
                    let mut accum = BTreeMap::<usize, f64>::new();
                    for index in start..end {
                        let mut left = [0usize; 4];
                        let mut right = [0usize; 4];
                        left[direction] = index;
                        right[direction] = index + 1;
                        for (axis_index, &axis) in fixed_axes.iter().enumerate() {
                            let coordinate = coord_index(level, fixed[axis_index]);
                            left[axis] = coordinate;
                            right[axis] = coordinate;
                        }
                        *accum.entry(vertex_index(level, left)).or_insert(0.0) += edge_weight;
                        *accum.entry(vertex_index(level, right)).or_insert(0.0) += edge_weight;
                    }
                    functionals.push(ScalarFunctional4d {
                        kind: ScalarFunctionalKind::LineAverage,
                        id: format!(
                            "line_{id:03}_dir{direction}_fixed{}{}{}",
                            fixed0, fixed1, fixed2
                        ),
                        row: map_to_row(accum),
                        geometric_measure: LINE_MEASURE,
                    });
                    id += 1;
                }
            }
        }
    }
    functionals
}

fn area_average_functionals(level: usize) -> Vec<ScalarFunctional4d> {
    let mut functionals = Vec::new();
    let mut id = 0;
    let start = coord_index(level, 1);
    let end = coord_index(level, 3);
    let small_area = 0.5 * (1.0 / level as f64).powi(2);
    let vertex_weight = small_area / (3.0 * AREA_MEASURE);

    for first_axis in 0..4 {
        for second_axis in (first_axis + 1)..4 {
            let fixed_axes = complement_axes2(first_axis, second_axis);
            for fixed0 in DEFAULT_COORD_NUMERATORS {
                for fixed1 in DEFAULT_COORD_NUMERATORS {
                    let patch = AreaFunctionalPatch {
                        level,
                        varying_axes: [first_axis, second_axis],
                        fixed_axes,
                        fixed_coordinates: [fixed0, fixed1],
                        vertex_weight,
                    };
                    for orientation in [AreaOrientation::Lower, AreaOrientation::Upper] {
                        let mut accum = BTreeMap::<usize, f64>::new();
                        for first in start..end {
                            for second in start..end {
                                let first_offset = first - start;
                                let second_offset = second - start;
                                match orientation {
                                    AreaOrientation::Lower if first_offset > second_offset => {
                                        patch.add_triangle_pair(first, second, &mut accum);
                                    }
                                    AreaOrientation::Lower if first_offset == second_offset => {
                                        patch.add_triangle(
                                            first,
                                            second,
                                            AreaOrientation::Lower,
                                            &mut accum,
                                        );
                                    }
                                    AreaOrientation::Upper if second_offset > first_offset => {
                                        patch.add_triangle_pair(first, second, &mut accum);
                                    }
                                    AreaOrientation::Upper if second_offset == first_offset => {
                                        patch.add_triangle(
                                            first,
                                            second,
                                            AreaOrientation::Upper,
                                            &mut accum,
                                        );
                                    }
                                    _ => {}
                                }
                            }
                        }
                        functionals.push(ScalarFunctional4d {
                            kind: ScalarFunctionalKind::AreaAverage,
                            id: format!(
                                "area_{id:03}_axes{first_axis}{second_axis}_fixed{}{}_{}",
                                fixed0,
                                fixed1,
                                orientation.as_str()
                            ),
                            row: map_to_row(accum),
                            geometric_measure: AREA_MEASURE,
                        });
                        id += 1;
                    }
                }
            }
        }
    }
    functionals
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AreaOrientation {
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
struct AreaFunctionalPatch {
    level: usize,
    varying_axes: [usize; 2],
    fixed_axes: [usize; 2],
    fixed_coordinates: [usize; 2],
    vertex_weight: f64,
}

impl AreaFunctionalPatch {
    fn add_triangle_pair(self, first: usize, second: usize, accum: &mut BTreeMap<usize, f64>) {
        self.add_triangle(first, second, AreaOrientation::Lower, accum);
        self.add_triangle(first, second, AreaOrientation::Upper, accum);
    }

    fn add_triangle(
        self,
        first: usize,
        second: usize,
        orientation: AreaOrientation,
        accum: &mut BTreeMap<usize, f64>,
    ) {
        let vertices = match orientation {
            AreaOrientation::Lower => [
                self.vertex_indices(first, second),
                self.vertex_indices(first + 1, second),
                self.vertex_indices(first + 1, second + 1),
            ],
            AreaOrientation::Upper => [
                self.vertex_indices(first, second),
                self.vertex_indices(first, second + 1),
                self.vertex_indices(first + 1, second + 1),
            ],
        };
        for indices in vertices {
            *accum
                .entry(vertex_index(self.level, indices))
                .or_insert(0.0) += self.vertex_weight;
        }
    }

    fn vertex_indices(self, first: usize, second: usize) -> [usize; 4] {
        let mut indices = [0usize; 4];
        indices[self.varying_axes[0]] = first;
        indices[self.varying_axes[1]] = second;
        for (axis_index, &axis) in self.fixed_axes.iter().enumerate() {
            indices[axis] = coord_index(self.level, self.fixed_coordinates[axis_index]);
        }
        indices
    }
}

impl AreaOrientation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Lower => "lower",
            Self::Upper => "upper",
        }
    }
}

fn complement_axes3(excluded: usize) -> [usize; 3] {
    let axes = (0..4).filter(|axis| *axis != excluded).collect::<Vec<_>>();
    [axes[0], axes[1], axes[2]]
}

fn complement_axes2(first_excluded: usize, second_excluded: usize) -> [usize; 2] {
    let axes = (0..4)
        .filter(|axis| *axis != first_excluded && *axis != second_excluded)
        .collect::<Vec<_>>();
    [axes[0], axes[1]]
}

fn coord_index(level: usize, numerator: usize) -> usize {
    level * numerator / COMMON_DENOMINATOR
}

fn vertex_index(level: usize, indices: [usize; 4]) -> usize {
    let mut vector = FeecVector::<usize>::zeros(4);
    for axis in 0..4 {
        vector[axis] = indices[axis];
    }
    cartesian_index2linear_index(vector, level + 1)
}

fn map_to_row(accum: BTreeMap<usize, f64>) -> Vec<(usize, f64)> {
    accum
        .into_iter()
        .filter(|(_, value)| value.abs() > EPS)
        .collect()
}

fn summary_rows(rows: &[ScalarVarianceRow4d]) -> Vec<ScalarSummaryRow4d> {
    let mut grouped: BTreeMap<(usize, MaternAlpha, ScalarFunctionalKind), Vec<f64>> =
        BTreeMap::new();
    for row in rows {
        grouped
            .entry((row.n, row.alpha, row.functional_kind))
            .or_default()
            .push(row.average_variance);
    }

    grouped
        .into_iter()
        .map(|((n, alpha, functional_kind), values)| ScalarSummaryRow4d {
            n,
            alpha,
            functional_kind,
            count: values.len(),
            mean_variance: mean(&values),
            median_variance: median(values.clone()),
            min_variance: values.iter().copied().fold(f64::INFINITY, f64::min),
            max_variance: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            stddev_variance: stddev(&values),
        })
        .collect()
}

fn fit_summary_rows(summaries: &[ScalarSummaryRow4d]) -> Vec<ScalarFitSummaryRow4d> {
    let mut grouped: BTreeMap<(MaternAlpha, ScalarFunctionalKind), Vec<&ScalarSummaryRow4d>> =
        BTreeMap::new();
    for row in summaries {
        grouped
            .entry((row.alpha, row.functional_kind))
            .or_default()
            .push(row);
    }

    let mut fits = Vec::new();
    for ((alpha, functional_kind), mut group) in grouped {
        group.sort_by_key(|row| row.n);
        if group.len() < 2 {
            continue;
        }
        let values = group
            .iter()
            .map(|row| row.median_variance)
            .collect::<Vec<_>>();
        let ns = group.iter().map(|row| row.n as f64).collect::<Vec<_>>();
        let (diagnostic, value, expected, status) =
            if alpha == MaternAlpha::Two && functional_kind == ScalarFunctionalKind::Point {
                (
                    "median_slope_vs_log_n".to_string(),
                    linear_slope(&ns.iter().map(|n| n.ln()).collect::<Vec<_>>(), &values),
                    "positive for log divergence".to_string(),
                    "borderline_point_diagnostic".to_string(),
                )
            } else {
                let prev = values[values.len() - 2];
                let last = values[values.len() - 1];
                (
                    "median_finest_relative_change".to_string(),
                    (last - prev).abs() / last.abs().max(EPS),
                    "small under convergence".to_string(),
                    "convergence_diagnostic".to_string(),
                )
            };
        fits.push(ScalarFitSummaryRow4d {
            alpha,
            functional_kind,
            statistic: "median_average_variance".to_string(),
            diagnostic,
            value,
            expected,
            status,
        });
    }
    fits
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn stddev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mean = mean(values);
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn linear_slope(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let x_mean = mean(xs);
    let y_mean = mean(ys);
    let numerator = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    let denominator = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();
    numerator / denominator.max(EPS)
}

fn lower_triangle_nnz(precision: &SparseMat) -> usize {
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

fn status_for(kind: ScalarFunctionalKind, alpha: MaternAlpha) -> &'static str {
    match (kind, alpha) {
        (ScalarFunctionalKind::Point, MaternAlpha::Two) => "expected_log_divergent",
        (ScalarFunctionalKind::Point, MaternAlpha::Three) => "expected_convergent",
        (ScalarFunctionalKind::LineAverage, MaternAlpha::Two)
        | (ScalarFunctionalKind::LineAverage, MaternAlpha::Three) => "expected_convergent",
        (ScalarFunctionalKind::AreaAverage, MaternAlpha::Two)
        | (ScalarFunctionalKind::AreaAverage, MaternAlpha::Three) => "expected_convergent",
        _ => "not_tested",
    }
}

fn write_outputs(
    config: &ScalarBorderline4dConfig,
    rows: &[ScalarVarianceRow4d],
    summaries: &[ScalarSummaryRow4d],
    fit_summaries: &[ScalarFitSummaryRow4d],
    stage_reports: &[ScalarStageReport4d],
) -> io::Result<()> {
    write_variance_csv(&config.output_dir.join("functional_variance.csv"), rows)?;
    write_summary_csv(&config.output_dir.join("summary.csv"), summaries)?;
    write_fit_summary_csv(&config.output_dir.join("fit_summary.csv"), fit_summaries)?;
    write_stage_report_csv(&config.output_dir.join("stage_report.csv"), stage_reports)?;
    write_readme(&config.output_dir.join("README.md"), config)?;
    Ok(())
}

pub fn write_variance_csv(path: &Path, rows: &[ScalarVarianceRow4d]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,h,alpha,functional_kind,functional_id,dofs,precision_nnz,precision_density,precision_lower_nnz,cholesky_factor_nnz,cholesky_factor_density,fill_in_ratio,factor_seconds,variance_seconds_total,support_entries,geometric_measure,average_variance,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17},{},{},{},{},{},{:.17e},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{}",
            row.n,
            row.h,
            row.alpha.as_u32(),
            row.functional_kind.as_str(),
            row.functional_id,
            row.dofs,
            row.precision_nnz,
            row.precision_density,
            row.precision_lower_nnz,
            row.cholesky_factor_nnz,
            row.cholesky_factor_density,
            row.fill_in_ratio,
            row.factor_seconds,
            row.variance_seconds_total,
            row.support_entries,
            row.geometric_measure,
            row.average_variance,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_summary_csv(path: &Path, rows: &[ScalarSummaryRow4d]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,alpha,functional_kind,count,mean_variance,median_variance,min_variance,max_variance,stddev_variance"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.n,
            row.alpha.as_u32(),
            row.functional_kind.as_str(),
            row.count,
            row.mean_variance,
            row.median_variance,
            row.min_variance,
            row.max_variance,
            row.stddev_variance
        )?;
    }
    Ok(())
}

pub fn write_fit_summary_csv(path: &Path, rows: &[ScalarFitSummaryRow4d]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "alpha,functional_kind,statistic,diagnostic,value,expected,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17e},{},{}",
            row.alpha.as_u32(),
            row.functional_kind.as_str(),
            row.statistic,
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_stage_report_csv(path: &Path, rows: &[ScalarStageReport4d]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "label,completed_levels,next_level,decision,variance_rows,summary_rows,fit_rows"
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
            "{},{},{},{},{},{},{}",
            row.label,
            completed,
            next,
            row.decision,
            row.variance_rows,
            row.summary_rows,
            row.fit_rows
        )?;
    }
    Ok(())
}

pub fn write_readme(path: &Path, config: &ScalarBorderline4dConfig) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# Scalar 4D Matern Borderline Experiment")?;
    writeln!(writer)?;
    writeln!(writer, "- levels: {:?}", config.levels)?;
    writeln!(writer, "- alphas: [2, 3]")?;
    writeln!(writer, "- kappa: {:.17}", config.kappa)?;
    writeln!(writer, "- tau: {:.17}", config.tau)?;
    writeln!(
        writer,
        "- exact common-grid coordinates: {{1/4, 1/2, 3/4}}^4"
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "The functional ensemble has 81 point evaluations, 108 line averages, and 108 area averages. Line and area rows are normalized averages, so a constant scalar field equal to one evaluates to one for every functional."
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "Expected behavior: alpha=2 point variances are log divergent in 4D; alpha=2 line and area averages converge; alpha=3 point variances converge."
    )?;
    Ok(())
}

fn validate_config(config: &ScalarBorderline4dConfig) -> Result<(), String> {
    if config.levels.is_empty() {
        return Err("at least one mesh level is required".to_string());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    for &level in &config.levels {
        validate_level(level)?;
    }
    Ok(())
}

fn validate_level(level: usize) -> Result<(), String> {
    if level == 0 || level % COMMON_DENOMINATOR != 0 {
        return Err(format!(
            "mesh level {level} is invalid; levels must be positive multiples of {COMMON_DENOMINATOR}"
        ));
    }
    Ok(())
}

#[cfg(test)]
fn apply_row(row: &[(usize, f64)], values: &[f64]) -> f64 {
    row.iter().map(|(col, weight)| *weight * values[*col]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_borderline_4d_functional_counts_match_exact_nested_ensemble() {
        let functionals =
            build_exact_nested_scalar_functionals(4).expect("functional ensemble should build");
        assert_eq!(
            functionals
                .iter()
                .filter(|functional| functional.kind == ScalarFunctionalKind::Point)
                .count(),
            81
        );
        assert_eq!(
            functionals
                .iter()
                .filter(|functional| functional.kind == ScalarFunctionalKind::LineAverage)
                .count(),
            108
        );
        assert_eq!(
            functionals
                .iter()
                .filter(|functional| functional.kind == ScalarFunctionalKind::AreaAverage)
                .count(),
            108
        );
    }

    #[test]
    fn scalar_borderline_4d_exact_average_rows_preserve_constants() {
        let level = 4;
        let functionals =
            build_exact_nested_scalar_functionals(level).expect("functional ensemble should build");
        let values = vec![1.0; (level + 1).pow(4)];
        for functional in &functionals {
            let actual = apply_row(&functional.row, &values);
            assert!(
                (actual - 1.0).abs() <= 1e-12,
                "{} expected constant average 1, got {actual}",
                functional.id
            );
        }
    }

    #[test]
    fn scalar_borderline_4d_rows_are_valid_on_requested_levels() {
        for level in [4, 8, 12] {
            let dimension = (level + 1usize).pow(4);
            let functionals = build_exact_nested_scalar_functionals(level)
                .expect("functional ensemble should build");
            for functional in &functionals {
                assert!(!functional.row.is_empty());
                assert!(functional
                    .row
                    .iter()
                    .all(|(column, value)| *column < dimension && value.is_finite()));
            }
        }
    }

    #[test]
    fn scalar_borderline_4d_profiles_preserve_submitted_levels() {
        let smoke = ScalarBorderline4dConfig::smoke(PathBuf::from("smoke"));
        let thesis = ScalarBorderline4dConfig::thesis_submitted(PathBuf::from("thesis"));
        assert_eq!(smoke.levels, vec![4]);
        assert_eq!(thesis.levels, vec![4, 8, 12, 16]);
        assert_eq!((thesis.kappa, thesis.tau), (4.0, 1.0));
    }

    #[test]
    fn scalar_borderline_4d_exact_point_variance_smoke_is_finite() {
        let level = 4;
        let mesh = CartesianMeshInfo::new_unit_scaled(4, level, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let prior = MaternPriorBuilder::from_feec(&topology, &metric, 0)
            .expect("scalar prior builder should initialize")
            .parameters(
                MaternParameters::new(MaternAlpha::Three, 4.0, 1.0)
                    .expect("scalar Matérn parameters should validate"),
            )
            .build()
            .expect("scalar alpha=3 precision should build");
        let point_functionals = build_exact_nested_scalar_functionals(level)
            .expect("functional ensemble should build")
            .into_iter()
            .filter(|functional| functional.kind == ScalarFunctionalKind::Point)
            .collect::<Vec<_>>();
        let report = exact_functional_variances(prior, topology.nsimplices(0), &point_functionals)
            .expect("exact scalar point variances should run");
        assert_eq!(report.variances.len(), 81);
        assert!(report
            .variances
            .iter()
            .all(|variance| variance.is_finite() && *variance >= 0.0));
    }
}
