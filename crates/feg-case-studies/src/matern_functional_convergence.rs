//! Convergence experiment for fixed integral-functional Matérn variances on 3D cube meshes.

#[cfg(test)]
use common::linalg::nalgebra::Vector as FeecVector;
use feg_infer::{
    prior::matern::{
        one_form::{
            build_hodge_laplacian_1form, build_matern_precision_1form_for_alpha,
            build_matern_precision_1form_for_alpha_with_coords, MaternConfig as Matern1FormConfig,
            MaternMassInverse as Matern1FormMassInverse,
        },
        three_form::{
            build_hodge_laplacian_3form,
            build_hodge_laplacian_3form_with_lower_mass_inverse_coords,
            build_matern_precision_3form_for_alpha, MaternConfig as Matern3FormConfig,
        },
        two_form::{
            build_hodge_laplacian_2form,
            build_hodge_laplacian_2form_with_lower_mass_inverse_coords,
            build_matern_precision_2form_for_alpha,
            build_matern_precision_2form_for_alpha_with_coords, MaternConfig as Matern2FormConfig,
            MaternMassInverse as Matern2FormMassInverse,
        },
        zero_form::{
            build_laplace_beltrami_0form, build_matern_precision_0form_for_alpha,
            MaternConfig as Matern0FormConfig, MaternMassInverse as Matern0FormMassInverse,
        },
        MaternAlpha,
    },
    sparse::feec_csr_to_gmrf,
};
use gmrf_core::{
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf, SparseRowOperator,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

const DEFAULT_LEVELS: [usize; 6] = [4, 8, 12, 16, 20, 24];
const DEFAULT_A: f64 = 0.25;
const DEFAULT_B: f64 = 0.75;
const DEFAULT_COORD_NUMERATORS: [usize; 3] = [1, 2, 3];
const COMMON_DENOMINATOR: usize = 4;
const EPS: f64 = 1e-12;
const COORD_TOL: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct FunctionalConvergenceConfig {
    pub levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
}

impl Default for FunctionalConvergenceConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            levels: DEFAULT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/matern_functional_convergence"),
            kappa: 4.0,
            tau: 1.0,
        }
    }
}

impl FunctionalConvergenceConfig {
    /// Cheap deterministic configuration intended for continuous integration.
    pub fn smoke(output_dir: PathBuf) -> Self {
        Self {
            levels: vec![4],
            output_dir,
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SparseInverseFamily {
    ScalarRowSum,
    Projected,
    BarycentricDual,
}

impl SparseInverseFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ScalarRowSum => "scalar_row_sum",
            Self::Projected => "projected",
            Self::BarycentricDual => "barycentric_dual",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FixedObservable {
    pub grade: usize,
    pub id: String,
    pub label: &'static str,
    pub row: Vec<(usize, f64)>,
    pub measure: f64,
}

#[derive(Debug, Clone)]
pub struct VarianceRow {
    pub n: usize,
    pub h: f64,
    pub grade: usize,
    pub alpha: MaternAlpha,
    pub inverse_family: SparseInverseFamily,
    pub functional_id: String,
    pub functional_label: &'static str,
    pub dofs: usize,
    pub precision_nnz: usize,
    pub support_entries: usize,
    pub geometric_measure: f64,
    pub raw_variance: f64,
    pub normalized_variance: f64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct FunctionalSummaryRow {
    pub n: usize,
    pub grade: usize,
    pub alpha: MaternAlpha,
    pub inverse_family: SparseInverseFamily,
    pub functional_label: &'static str,
    pub count: usize,
    pub mean_raw_variance: f64,
    pub median_raw_variance: f64,
    pub min_raw_variance: f64,
    pub max_raw_variance: f64,
    pub stddev_raw_variance: f64,
    pub median_normalized_variance: f64,
}

#[derive(Debug, Clone)]
pub struct FitSummaryRow {
    pub grade: usize,
    pub alpha: MaternAlpha,
    pub inverse_family: SparseInverseFamily,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct FunctionalConvergenceResult {
    pub rows: Vec<VarianceRow>,
    pub summaries: Vec<FunctionalSummaryRow>,
    pub fit_summaries: Vec<FitSummaryRow>,
}

pub fn default_levels() -> Vec<usize> {
    DEFAULT_LEVELS.to_vec()
}

pub fn run_functional_convergence_experiment(
    config: &FunctionalConvergenceConfig,
) -> Result<FunctionalConvergenceResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut rows = Vec::new();
    for &n in &config.levels {
        eprintln!("[matern_functional_convergence] mesh n={n}");
        let mesh = CartesianMeshInfo::new_unit_scaled(3, n, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        for grade in 0..=3 {
            let observables = fixed_simplex_observables(&topology, &coords, grade)?;
            for alpha in [MaternAlpha::One, MaternAlpha::Two] {
                for family in families_for_grade(grade) {
                    let precision = build_prior_precision(
                        &topology,
                        &coords,
                        &metric,
                        grade,
                        alpha,
                        family,
                        config.kappa,
                        config.tau,
                    )?;
                    let raw_variances = exact_observable_variances(
                        &precision,
                        topology.nsimplices(grade),
                        &observables,
                    )?;
                    for (observable, raw_variance) in observables.iter().zip(raw_variances) {
                        let normalized_variance =
                            raw_variance / observable.measure.max(EPS).powi(2);
                        rows.push(VarianceRow {
                            n,
                            h: 1.0 / n as f64,
                            grade,
                            alpha,
                            inverse_family: family,
                            functional_id: observable.id.clone(),
                            functional_label: observable.label,
                            dofs: precision.nrows(),
                            precision_nnz: precision.nnz(),
                            support_entries: observable.row.len(),
                            geometric_measure: observable.measure,
                            raw_variance,
                            normalized_variance,
                            status: status_for(grade, alpha, family).to_string(),
                        });
                    }
                }
            }
        }
    }

    let summaries = summary_rows(&rows);
    let fit_summaries = fit_summaries(&summaries);
    write_variance_csv(&config.output_dir.join("functional_variance.csv"), &rows)?;
    write_summary_csv(&config.output_dir.join("summary.csv"), &summaries)?;
    write_fit_summary_csv(&config.output_dir.join("fit_summary.csv"), &fit_summaries)?;
    write_readme(&config.output_dir.join("README.md"), config)?;

    Ok(FunctionalConvergenceResult {
        rows,
        summaries,
        fit_summaries,
    })
}

// This adapter selects mathematically distinct, recorded inverse-mass policies for
// each form degree. Cross-degree recurrence and maintained 3D smoke tests cover it.
#[allow(clippy::too_many_arguments)]
fn build_prior_precision(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    grade: usize,
    alpha: MaternAlpha,
    family: SparseInverseFamily,
    kappa: f64,
    tau: f64,
) -> Result<common::linalg::nalgebra::CsrMatrix<f64>, String> {
    match grade {
        0 => {
            let laplace = build_laplace_beltrami_0form(topology, metric);
            Ok(build_matern_precision_0form_for_alpha(
                &laplace,
                alpha,
                Matern0FormConfig {
                    kappa,
                    tau,
                    mass_inverse: Matern0FormMassInverse::RowSumLumped,
                },
            ))
        }
        1 => {
            let hodge = build_hodge_laplacian_1form(topology, metric);
            let config = Matern1FormConfig {
                kappa,
                tau,
                mass_inverse: one_form_inverse_family(family)?,
            };
            match family {
                SparseInverseFamily::BarycentricDual => {
                    build_matern_precision_1form_for_alpha_with_coords(
                        topology, coords, metric, &hodge, alpha, config,
                    )
                }
                _ => Ok(build_matern_precision_1form_for_alpha(
                    topology, metric, &hodge, alpha, config,
                )),
            }
        }
        2 => {
            let lower_inverse = one_form_inverse_family(family)?;
            let hodge = match family {
                SparseInverseFamily::BarycentricDual => {
                    build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
                        topology,
                        coords,
                        metric,
                        lower_inverse,
                    )?
                }
                _ => build_hodge_laplacian_2form(topology, metric)?,
            };
            let config = Matern2FormConfig {
                kappa,
                tau,
                mass_inverse: two_form_inverse_family(family)?,
            };
            match family {
                SparseInverseFamily::BarycentricDual => {
                    build_matern_precision_2form_for_alpha_with_coords(
                        topology, coords, metric, &hodge, alpha, config,
                    )
                }
                _ => {
                    build_matern_precision_2form_for_alpha(topology, metric, &hodge, alpha, config)
                }
            }
        }
        3 => {
            let lower_inverse = two_form_inverse_family(family)?;
            let hodge = match family {
                SparseInverseFamily::BarycentricDual => {
                    build_hodge_laplacian_3form_with_lower_mass_inverse_coords(
                        topology,
                        coords,
                        metric,
                        lower_inverse,
                    )?
                }
                _ => build_hodge_laplacian_3form(topology, metric)?,
            };
            Ok(build_matern_precision_3form_for_alpha(
                &hodge,
                alpha,
                Matern3FormConfig { kappa, tau },
            ))
        }
        _ => Err(format!("unsupported form grade {grade}")),
    }
}

fn exact_observable_variances(
    precision: &common::linalg::nalgebra::CsrMatrix<f64>,
    dimension: usize,
    observables: &[FixedObservable],
) -> Result<Vec<f64>, String> {
    let precision_gmrf = feec_csr_to_gmrf(precision);
    let factor = precision_gmrf
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision: {err}"))?;
    let mut gmrf = Gmrf::from_mean_and_precision(GmrfVector::zeros(dimension), precision_gmrf)
        .map_err(|err| format!("failed to build prior GMRF: {err}"))?
        .with_precision_sqrt(factor);
    let operator = SparseRowOperator::new(
        dimension,
        observables
            .iter()
            .map(|observable| observable.row.clone())
            .collect(),
    )
    .map_err(|err| format!("failed to build observable operator: {err}"))?;
    let constraints = GmrfDenseMatrix::zeros(0, dimension);
    let variance = gmrf
        .exact_transformed_variance_decomposition(&operator, &constraints)
        .map_err(|err| format!("failed to compute transformed variance: {err}"))?;
    Ok((0..observables.len())
        .map(|index| variance.unconstrained_diag[index])
        .collect())
}

pub fn fixed_simplex_observable(
    topology: &Complex,
    coords: &MeshCoords,
    grade: usize,
) -> Result<FixedObservable, String> {
    fixed_simplex_observables(topology, coords, grade)?
        .into_iter()
        .next()
        .ok_or_else(|| format!("no observables were built for grade {grade}"))
}

pub fn fixed_simplex_observables(
    topology: &Complex,
    coords: &MeshCoords,
    grade: usize,
) -> Result<Vec<FixedObservable>, String> {
    match grade {
        0 => point_observables(coords),
        1 => line_observables(topology, coords),
        2 => surface_observables(topology, coords),
        3 => volume_observables(topology, coords),
        _ => Err(format!("unsupported observable grade {grade}")),
    }
}

fn point_observables(coords: &MeshCoords) -> Result<Vec<FixedObservable>, String> {
    let mut observables = Vec::new();
    let mut id = 0;
    for x in DEFAULT_COORD_NUMERATORS {
        for y in DEFAULT_COORD_NUMERATORS {
            for z in DEFAULT_COORD_NUMERATORS {
                let target = [coord_value(x), coord_value(y), coord_value(z)];
                let vertex = (0..coords.nvertices())
                    .find(|&vertex| close_point(coord3(coords, vertex), target))
                    .ok_or_else(|| {
                        format!("fixed point {target:?} does not coincide with a mesh vertex")
                    })?;
                observables.push(FixedObservable {
                    grade: 0,
                    id: format!("point_{id:03}_q{x}{y}{z}"),
                    label: "point",
                    row: vec![(vertex, 1.0)],
                    measure: 1.0,
                });
                id += 1;
            }
        }
    }
    Ok(observables)
}

fn line_observables(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<FixedObservable>, String> {
    let mut observables = Vec::new();
    let mut id = 0;
    for direction in 0..3 {
        let fixed_axes = complement_axis1(direction);
        for fixed0 in DEFAULT_COORD_NUMERATORS {
            for fixed1 in DEFAULT_COORD_NUMERATORS {
                let mut fixed = [0.0; 3];
                fixed[fixed_axes[0]] = coord_value(fixed0);
                fixed[fixed_axes[1]] = coord_value(fixed1);
                let row = line_row(topology, coords, direction, fixed);
                observables.push(require_nonempty(
                    row,
                    1,
                    format!("line_{id:03}_dir{direction}_fixed{fixed0}{fixed1}"),
                    "line",
                    DEFAULT_B - DEFAULT_A,
                )?);
                id += 1;
            }
        }
    }
    Ok(observables)
}

fn line_row(
    topology: &Complex,
    coords: &MeshCoords,
    direction: usize,
    fixed: [f64; 3],
) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for edge in topology.skeleton(1).handle_iter() {
        let p0 = coord3(coords, edge.vertices[0]);
        let p1 = coord3(coords, edge.vertices[1]);
        if (0..3)
            .filter(|axis| *axis != direction)
            .any(|axis| !close(p0[axis], fixed[axis]) || !close(p1[axis], fixed[axis]))
        {
            continue;
        }
        if !in_closed_interval(p0[direction], DEFAULT_A, DEFAULT_B)
            || !in_closed_interval(p1[direction], DEFAULT_A, DEFAULT_B)
        {
            continue;
        }
        let delta = p1[direction] - p0[direction];
        if delta.abs() <= COORD_TOL {
            continue;
        }
        row.push((edge.kidx(), delta.signum()));
    }
    row
}

fn surface_observables(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<FixedObservable>, String> {
    let mut observables = Vec::new();
    let mut id = 0;
    for first_axis in 0..3 {
        for second_axis in (first_axis + 1)..3 {
            let fixed_axis = complement_axis2(first_axis, second_axis);
            for fixed in DEFAULT_COORD_NUMERATORS {
                let row = surface_row(
                    topology,
                    coords,
                    first_axis,
                    second_axis,
                    fixed_axis,
                    coord_value(fixed),
                );
                observables.push(require_nonempty(
                    row,
                    2,
                    format!("surface_{id:03}_axes{first_axis}{second_axis}_fixed{fixed}"),
                    "surface",
                    (DEFAULT_B - DEFAULT_A).powi(2),
                )?);
                id += 1;
            }
        }
    }
    Ok(observables)
}

fn surface_row(
    topology: &Complex,
    coords: &MeshCoords,
    first_axis: usize,
    second_axis: usize,
    fixed_axis: usize,
    fixed_value: f64,
) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for face in topology.skeleton(2).handle_iter() {
        let points = face
            .vertices
            .iter()
            .map(|&vertex| coord3(coords, vertex))
            .collect::<Vec<_>>();
        if points
            .iter()
            .any(|point| !close(point[fixed_axis], fixed_value))
        {
            continue;
        }
        if points.iter().any(|point| {
            !in_closed_interval(point[first_axis], DEFAULT_A, DEFAULT_B)
                || !in_closed_interval(point[second_axis], DEFAULT_A, DEFAULT_B)
        }) {
            continue;
        }
        let area = signed_projected_area(points[0], points[1], points[2], first_axis, second_axis);
        if area.abs() <= COORD_TOL {
            continue;
        }
        row.push((face.kidx(), area.signum()));
    }
    row
}

fn volume_observables(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<FixedObservable>, String> {
    let mut observables = Vec::new();
    let mut id = 0;
    for x0 in [0, 2] {
        for y0 in [0, 2] {
            for z0 in [0, 2] {
                let min = [coord_value(x0), coord_value(y0), coord_value(z0)];
                let max = [
                    coord_value(x0 + 2),
                    coord_value(y0 + 2),
                    coord_value(z0 + 2),
                ];
                let row = volume_row(topology, coords, min, max);
                observables.push(require_nonempty(
                    row,
                    3,
                    format!("volume_{id:03}_box{x0}{y0}{z0}"),
                    "volume",
                    (DEFAULT_B - DEFAULT_A).powi(3),
                )?);
                id += 1;
            }
        }
    }
    Ok(observables)
}

fn volume_row(
    topology: &Complex,
    coords: &MeshCoords,
    min: [f64; 3],
    max: [f64; 3],
) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for cell in topology.cells().handle_iter() {
        let points = cell
            .vertices
            .iter()
            .map(|&vertex| coord3(coords, vertex))
            .collect::<Vec<_>>();
        if points
            .iter()
            .any(|point| (0..3).any(|axis| !in_closed_interval(point[axis], min[axis], max[axis])))
        {
            continue;
        }
        let volume = signed_volume(points[0], points[1], points[2], points[3]);
        if volume.abs() <= COORD_TOL {
            continue;
        }
        row.push((cell.kidx(), volume.signum()));
    }
    row
}

fn require_nonempty(
    row: Vec<(usize, f64)>,
    grade: usize,
    id: String,
    label: &'static str,
    measure: f64,
) -> Result<FixedObservable, String> {
    if row.is_empty() {
        return Err(format!(
            "fixed {label} observable selected no mesh simplices"
        ));
    }
    Ok(FixedObservable {
        grade,
        id,
        label,
        row,
        measure,
    })
}

fn families_for_grade(grade: usize) -> Vec<SparseInverseFamily> {
    if grade == 0 {
        vec![SparseInverseFamily::ScalarRowSum]
    } else {
        vec![
            SparseInverseFamily::Projected,
            SparseInverseFamily::BarycentricDual,
        ]
    }
}

fn one_form_inverse_family(family: SparseInverseFamily) -> Result<Matern1FormMassInverse, String> {
    match family {
        SparseInverseFamily::ScalarRowSum => Ok(Matern1FormMassInverse::RowSumLumped),
        SparseInverseFamily::Projected => Ok(Matern1FormMassInverse::Nc1ProjectedSparseInverse),
        SparseInverseFamily::BarycentricDual => {
            Ok(Matern1FormMassInverse::BarycentricDualSparseInverse)
        }
    }
}

fn two_form_inverse_family(family: SparseInverseFamily) -> Result<Matern2FormMassInverse, String> {
    match family {
        SparseInverseFamily::ScalarRowSum | SparseInverseFamily::Projected => {
            Ok(Matern2FormMassInverse::ExactTopDegreeDiagonalOrProjectedNc2)
        }
        SparseInverseFamily::BarycentricDual => {
            Ok(Matern2FormMassInverse::BarycentricDualSparseInverse)
        }
    }
}

fn status_for(grade: usize, alpha: MaternAlpha, family: SparseInverseFamily) -> &'static str {
    match (grade, alpha, family) {
        (0, MaternAlpha::One, _) => "expected_h_inverse_divergence",
        (1, MaternAlpha::One, SparseInverseFamily::BarycentricDual) => {
            "expected_log_borderline_no_load_inverse_difference"
        }
        (1, MaternAlpha::One, _) => "expected_log_borderline",
        _ => "expected_convergent",
    }
}

fn summary_rows(rows: &[VarianceRow]) -> Vec<FunctionalSummaryRow> {
    let mut grouped: BTreeMap<
        (usize, usize, MaternAlpha, SparseInverseFamily, &'static str),
        Vec<&VarianceRow>,
    > = BTreeMap::new();
    for row in rows {
        grouped
            .entry((
                row.n,
                row.grade,
                row.alpha,
                row.inverse_family,
                row.functional_label,
            ))
            .or_default()
            .push(row);
    }

    grouped
        .into_iter()
        .map(
            |((n, grade, alpha, inverse_family, functional_label), group)| {
                let raw = group.iter().map(|row| row.raw_variance).collect::<Vec<_>>();
                let normalized = group
                    .iter()
                    .map(|row| row.normalized_variance)
                    .collect::<Vec<_>>();
                FunctionalSummaryRow {
                    n,
                    grade,
                    alpha,
                    inverse_family,
                    functional_label,
                    count: group.len(),
                    mean_raw_variance: mean(&raw),
                    median_raw_variance: median(raw.clone()),
                    min_raw_variance: raw.iter().copied().fold(f64::INFINITY, f64::min),
                    max_raw_variance: raw.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    stddev_raw_variance: stddev(&raw),
                    median_normalized_variance: median(normalized),
                }
            },
        )
        .collect()
}

fn fit_summaries(rows: &[FunctionalSummaryRow]) -> Vec<FitSummaryRow> {
    let mut grouped: BTreeMap<
        (usize, MaternAlpha, SparseInverseFamily),
        Vec<&FunctionalSummaryRow>,
    > = BTreeMap::new();
    for row in rows {
        grouped
            .entry((row.grade, row.alpha, row.inverse_family))
            .or_default()
            .push(row);
    }

    let mut summaries = Vec::new();
    for ((grade, alpha, family), mut group) in grouped {
        group.sort_by_key(|row| row.n);
        if group.len() < 2 {
            continue;
        }
        if grade == 0 && alpha == MaternAlpha::One {
            let xs = group
                .iter()
                .map(|row| (row.n as f64).ln())
                .collect::<Vec<_>>();
            let ys = group
                .iter()
                .map(|row| row.median_raw_variance.max(EPS).ln())
                .collect::<Vec<_>>();
            summaries.push(FitSummaryRow {
                grade,
                alpha,
                inverse_family: family,
                diagnostic: "loglog_slope_vs_n".to_string(),
                value: linear_slope(&xs, &ys),
                expected: "near +1".to_string(),
                status: "divergence_diagnostic".to_string(),
            });
        } else if grade == 1 && alpha == MaternAlpha::One {
            let xs = group
                .iter()
                .map(|row| (row.n as f64).ln())
                .collect::<Vec<_>>();
            let ys = group
                .iter()
                .map(|row| row.median_raw_variance)
                .collect::<Vec<_>>();
            summaries.push(FitSummaryRow {
                grade,
                alpha,
                inverse_family: family,
                diagnostic: "slope_vs_log_n".to_string(),
                value: linear_slope(&xs, &ys),
                expected: "positive".to_string(),
                status: "borderline_log_diagnostic".to_string(),
            });
        } else {
            let prev = group[group.len() - 2].median_raw_variance;
            let last = group[group.len() - 1].median_raw_variance;
            summaries.push(FitSummaryRow {
                grade,
                alpha,
                inverse_family: family,
                diagnostic: "finest_relative_change".to_string(),
                value: (last - prev).abs() / last.abs().max(EPS),
                expected: "small under convergence".to_string(),
                status: "convergence_diagnostic".to_string(),
            });
        }
    }
    summaries
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

pub fn write_variance_csv(path: &Path, rows: &[VarianceRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,h,grade,alpha,inverse_family,functional_label,functional_id,dofs,precision_nnz,support_entries,geometric_measure,raw_variance,normalized_variance,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{}",
            row.n,
            row.h,
            row.grade,
            row.alpha.as_u32(),
            row.inverse_family.as_str(),
            row.functional_label,
            row.functional_id,
            row.dofs,
            row.precision_nnz,
            row.support_entries,
            row.geometric_measure,
            row.raw_variance,
            row.normalized_variance,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_summary_csv(path: &Path, rows: &[FunctionalSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,grade,alpha,inverse_family,functional_label,count,mean_raw_variance,median_raw_variance,min_raw_variance,max_raw_variance,stddev_raw_variance,median_normalized_variance"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.n,
            row.grade,
            row.alpha.as_u32(),
            row.inverse_family.as_str(),
            row.functional_label,
            row.count,
            row.mean_raw_variance,
            row.median_raw_variance,
            row.min_raw_variance,
            row.max_raw_variance,
            row.stddev_raw_variance,
            row.median_normalized_variance
        )?;
    }
    Ok(())
}

pub fn write_fit_summary_csv(path: &Path, rows: &[FitSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "grade,alpha,inverse_family,diagnostic,value,expected,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17e},{},{}",
            row.grade,
            row.alpha.as_u32(),
            row.inverse_family.as_str(),
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_readme(path: &Path, config: &FunctionalConvergenceConfig) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# 3D Hodge-Matern Functional Variance Convergence")?;
    writeln!(writer)?;
    writeln!(writer, "- levels: {:?}", config.levels)?;
    writeln!(writer, "- kappa: {:.17}", config.kappa)?;
    writeln!(writer, "- tau: {:.17}", config.tau)?;
    writeln!(writer, "- fixed coordinates: a={DEFAULT_A}, b={DEFAULT_B}")?;
    writeln!(
        writer,
        "- functional ensemble: 27 points, 27 line integrals, 9 surface integrals, and 8 volume integrals"
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "| alpha | k=0 point | k=1 line | k=2 area | k=3 volume |"
    )?;
    writeln!(writer, "|---|---|---|---|---|")?;
    writeln!(
        writer,
        "| 1 | diverges like h^-1 | borderline log h^-1 | converges | converges |"
    )?;
    writeln!(
        writer,
        "| 2 | converges | converges | converges | converges |"
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "The primary CSV reports raw integral variances for every functional. The summary CSV reports ensemble statistics by mesh level, form degree, alpha, and sparse inverse family. Normalized variances divide by the fixed observable measure squared."
    )?;
    Ok(())
}

fn validate_config(config: &FunctionalConvergenceConfig) -> Result<(), String> {
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
        if level == 0 || level % 4 != 0 {
            return Err(format!(
                "mesh level {level} is invalid; levels must be positive multiples of 4"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
fn constant_form_cochain_for_observable(
    topology: &Complex,
    coords: &MeshCoords,
    observable: &FixedObservable,
) -> FeecVector {
    let line_direction = || parse_digit_after(&observable.id, "_dir");
    let surface_axes = || {
        let offset = observable
            .id
            .find("_axes")
            .map(|index| index + "_axes".len())
            .expect("surface observable id contains axes");
        let bytes = observable.id.as_bytes();
        (
            (bytes[offset] - b'0') as usize,
            (bytes[offset + 1] - b'0') as usize,
        )
    };

    FeecVector::from_iterator(
        topology.nsimplices(observable.grade),
        topology
            .skeleton(observable.grade)
            .handle_iter()
            .map(|simplex| {
                let points = simplex
                    .vertices
                    .iter()
                    .map(|&vertex| coord3(coords, vertex))
                    .collect::<Vec<_>>();
                match observable.grade {
                    0 => 1.0,
                    1 => {
                        let direction = line_direction();
                        points[1][direction] - points[0][direction]
                    }
                    2 => {
                        let (first_axis, second_axis) = surface_axes();
                        signed_projected_area(
                            points[0],
                            points[1],
                            points[2],
                            first_axis,
                            second_axis,
                        )
                    }
                    3 => signed_volume(points[0], points[1], points[2], points[3]),
                    _ => 0.0,
                }
            }),
    )
}

#[cfg(test)]
fn parse_digit_after(value: &str, marker: &str) -> usize {
    let offset = value
        .find(marker)
        .map(|index| index + marker.len())
        .expect("observable id contains marker");
    (value.as_bytes()[offset] - b'0') as usize
}

#[cfg(test)]
fn apply_row(row: &[(usize, f64)], values: &FeecVector) -> f64 {
    row.iter().map(|(col, weight)| *weight * values[*col]).sum()
}

fn coord3(coords: &MeshCoords, vertex: usize) -> [f64; 3] {
    let coord = coords.coord(vertex);
    [
        coord[0],
        if coords.dim() > 1 { coord[1] } else { 0.0 },
        if coords.dim() > 2 { coord[2] } else { 0.0 },
    ]
}

fn close(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= COORD_TOL
}

fn close_point(lhs: [f64; 3], rhs: [f64; 3]) -> bool {
    close(lhs[0], rhs[0]) && close(lhs[1], rhs[1]) && close(lhs[2], rhs[2])
}

fn in_closed_interval(value: f64, min: f64, max: f64) -> bool {
    value >= min - COORD_TOL && value <= max + COORD_TOL
}

fn coord_value(numerator: usize) -> f64 {
    numerator as f64 / COMMON_DENOMINATOR as f64
}

fn complement_axis1(axis: usize) -> [usize; 2] {
    let axes = (0..3)
        .filter(|candidate| *candidate != axis)
        .collect::<Vec<_>>();
    [axes[0], axes[1]]
}

fn complement_axis2(first_axis: usize, second_axis: usize) -> usize {
    (0..3)
        .find(|axis| *axis != first_axis && *axis != second_axis)
        .expect("two distinct axes in 3D have one complement")
}

fn signed_projected_area(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    first_axis: usize,
    second_axis: usize,
) -> f64 {
    0.5 * ((b[first_axis] - a[first_axis]) * (c[second_axis] - a[second_axis])
        - (b[second_axis] - a[second_axis]) * (c[first_axis] - a[first_axis]))
}

fn signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ad = sub(d, a);
    dot(ab, cross(ac, ad)) / 6.0
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(level: usize) -> (Complex, MeshCoords, MeshLengths) {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, level, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        (topology, coords, metric)
    }

    #[test]
    fn matern_functional_convergence_observable_supports_are_nonempty() {
        for level in [4, 8] {
            let (topology, coords, _metric) = mesh(level);
            let expected_counts = [27, 27, 9, 8];
            for (grade, &expected_count) in expected_counts.iter().enumerate() {
                let observables = fixed_simplex_observables(&topology, &coords, grade)
                    .expect("fixed observables should build");
                assert_eq!(observables.len(), expected_count);
                assert!(observables
                    .iter()
                    .all(|observable| observable.grade == grade && !observable.row.is_empty()));
            }
        }
    }

    #[test]
    fn matern_functional_convergence_observables_integrate_constant_forms() {
        let (topology, coords, _metric) = mesh(8);

        for grade in 0..=3 {
            let observables = fixed_simplex_observables(&topology, &coords, grade)
                .expect("fixed observables should build");
            for observable in &observables {
                let values = constant_form_cochain_for_observable(&topology, &coords, observable);
                let actual = apply_row(&observable.row, &values);
                assert!(
                    (actual - observable.measure).abs() <= 1e-12,
                    "{} expected {}, got {actual}",
                    observable.id,
                    observable.measure
                );
            }
        }
    }

    #[test]
    fn matern_functional_convergence_tiny_sweep_produces_finite_variances() {
        let config = FunctionalConvergenceConfig {
            levels: vec![4, 8],
            output_dir: std::env::temp_dir().join(format!(
                "matern_functional_convergence_test_{}",
                std::process::id()
            )),
            kappa: 4.0,
            tau: 1.0,
        };
        let result = run_functional_convergence_experiment(&config)
            .expect("tiny functional convergence sweep should run");

        let expected_rows_per_level = 2 * (27 + 2 * 27 + 2 * 9 + 2 * 8);
        assert_eq!(
            result.rows.len(),
            config.levels.len() * expected_rows_per_level
        );
        assert!(result.rows.iter().all(|row| {
            row.raw_variance.is_finite()
                && row.raw_variance >= 0.0
                && row.normalized_variance.is_finite()
                && row.support_entries > 0
                && row.dofs > 0
        }));
        assert!(!result.summaries.is_empty());
        assert!(!result.fit_summaries.is_empty());
    }
}
