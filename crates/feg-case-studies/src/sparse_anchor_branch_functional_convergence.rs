//! Functional variance convergence for 3D Hodge-Matern and decomposed branches.
//!
//! This experiment compares the full 1-form Hodge-Matern baseline,
//! form-spectrum and potential-spectrum decomposed branch priors, and
//! pushforward diagnostics. It uses native 1-form line integrals and
//! barycenter-reconstructed point component/trace pushforwards.

use crate::matern_functional_convergence::{fixed_simplex_observables, FixedObservable};
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use feg_core::HodgeBranchKind;
use feg_infer::{
    prior::{
        hodge_matern::{
            build_hodge_matern_1form_prior_with_coords, HodgeMatern1FormPrior,
            HodgeMatern1FormPriorConfig, HodgeMaternBranchConfig, HodgeMaternSpectrum,
        },
        matern::{
            one_form::{
                build_hodge_laplacian_1form, build_matern_precision_1form_for_alpha_with_coords,
                build_reconstructed_barycenter_field_operator, MaternConfig as Matern1FormConfig,
                MaternMassInverse as Matern1FormMassInverse,
            },
            MaternAlpha,
        },
    },
    sparse::{feec_csr_to_gmrf, sparse_row_operator_from_feec_csr},
};
use gmrf_core::{
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf, SparseRowOperator,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::{
        coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
        metric::mesh::MeshLengths,
    },
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
const DEFAULT_COORD_NUMERATORS: [usize; 3] = [1, 2, 3];
const COMMON_DENOMINATOR: usize = 4;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct BranchFunctionalConvergenceConfig {
    pub levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
}

impl Default for BranchFunctionalConvergenceConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            levels: DEFAULT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/sparse_anchor_branch_functional_convergence"),
            kappa: 4.0,
            tau: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BranchFunctionalModel {
    BaselineHodgeMatern,
    FormMatern,
    PotentialMatern,
}

impl BranchFunctionalModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BaselineHodgeMatern => "baseline_hodge_matern",
            Self::FormMatern => "form_matern",
            Self::PotentialMatern => "potential_matern",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BranchFunctionalComponent {
    Full1Form,
    Exact,
    Coexact,
    JointTotal,
}

impl BranchFunctionalComponent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full1Form => "full_1form",
            Self::Exact => "exact",
            Self::Coexact => "coexact",
            Self::JointTotal => "joint_total",
        }
    }

    fn branches(self) -> Result<Vec<HodgeBranchKind>, String> {
        match self {
            Self::Exact => Ok(vec![HodgeBranchKind::Exact]),
            Self::Coexact => Ok(vec![HodgeBranchKind::Coexact]),
            Self::JointTotal => Ok(vec![HodgeBranchKind::Exact, HodgeBranchKind::Coexact]),
            Self::Full1Form => Err(
                "full 1-form component is only valid for the baseline Hodge-Matern model"
                    .to_string(),
            ),
        }
    }
}

const REPORT_MODELS: [BranchFunctionalModel; 3] = [
    BranchFunctionalModel::BaselineHodgeMatern,
    BranchFunctionalModel::FormMatern,
    BranchFunctionalModel::PotentialMatern,
];
const FULL_1FORM_COMPONENTS: [BranchFunctionalComponent; 1] =
    [BranchFunctionalComponent::Full1Form];
const BRANCH_COMPONENTS: [BranchFunctionalComponent; 2] = [
    BranchFunctionalComponent::Exact,
    BranchFunctionalComponent::Coexact,
];

fn components_for_model(model: BranchFunctionalModel) -> &'static [BranchFunctionalComponent] {
    match model {
        BranchFunctionalModel::BaselineHodgeMatern => &FULL_1FORM_COMPONENTS,
        BranchFunctionalModel::FormMatern | BranchFunctionalModel::PotentialMatern => {
            &BRANCH_COMPONENTS
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BranchObservableKind {
    Line,
    PointComponent,
    PointTrace,
}

impl BranchObservableKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::PointComponent => "point_component",
            Self::PointTrace => "point_trace",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BranchFunctionalVarianceRow {
    pub n: usize,
    pub h: f64,
    pub alpha: MaternAlpha,
    pub model: BranchFunctionalModel,
    pub component: BranchFunctionalComponent,
    pub observable_kind: BranchObservableKind,
    pub functional_id: String,
    pub latent_dofs: usize,
    pub precision_nnz: usize,
    pub transform_nnz: usize,
    pub support_entries: usize,
    pub geometric_measure: f64,
    pub raw_variance: f64,
    pub normalized_variance: f64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct BranchFunctionalSummaryRow {
    pub n: usize,
    pub model: BranchFunctionalModel,
    pub component: BranchFunctionalComponent,
    pub observable_kind: BranchObservableKind,
    pub count: usize,
    pub mean_raw_variance: f64,
    pub median_raw_variance: f64,
    pub min_raw_variance: f64,
    pub max_raw_variance: f64,
    pub stddev_raw_variance: f64,
    pub median_normalized_variance: f64,
}

#[derive(Debug, Clone)]
pub struct BranchFunctionalFitSummaryRow {
    pub model: BranchFunctionalModel,
    pub component: BranchFunctionalComponent,
    pub observable_kind: BranchObservableKind,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct BranchFunctionalConvergenceResult {
    pub rows: Vec<BranchFunctionalVarianceRow>,
    pub summaries: Vec<BranchFunctionalSummaryRow>,
    pub fit_summaries: Vec<BranchFunctionalFitSummaryRow>,
}

struct MeshData {
    topology: Complex,
    coords: MeshCoords,
    metric: MeshLengths,
}

struct BuiltPrior {
    latent_dimension: usize,
    precision: FeecCsr,
    latent_to_ambient: Option<FeecCsr>,
}

impl BuiltPrior {
    fn from_ambient_precision(precision: FeecCsr) -> Self {
        Self {
            latent_dimension: precision.nrows(),
            precision,
            latent_to_ambient: None,
        }
    }

    fn from_hodge_matern(prior: HodgeMatern1FormPrior) -> Self {
        Self {
            latent_dimension: prior.latent_dimension(),
            precision: prior.precision,
            latent_to_ambient: Some(prior.latent_to_ambient),
        }
    }

    fn latent_dimension(&self) -> usize {
        self.latent_dimension
    }

    fn transform_nnz(&self) -> usize {
        self.latent_to_ambient
            .as_ref()
            .map_or(self.latent_dimension, |matrix| matrix.nnz())
    }
}

#[derive(Debug, Clone)]
struct PointComponentObservable {
    point_id: String,
    component_index: usize,
    row: Vec<(usize, f64)>,
}

pub fn run_sparse_anchor_branch_functional_convergence(
    config: &BranchFunctionalConvergenceConfig,
) -> Result<BranchFunctionalConvergenceResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut rows = Vec::new();
    for &n in &config.levels {
        eprintln!("[sparse_anchor_branch_functional_convergence] mesh n={n}");
        let mesh = build_mesh(n);
        let line_observables = fixed_simplex_observables(&mesh.topology, &mesh.coords, 1)?;
        let line_operator =
            sparse_operator_from_fixed_observables(mesh.topology.edges().len(), &line_observables)?;
        let point_observables = point_component_observables(&mesh.topology, &mesh.coords)?;
        let point_operator = SparseRowOperator::new(
            mesh.topology.edges().len(),
            point_observables
                .iter()
                .map(|observable| observable.row.clone())
                .collect(),
        )
        .map_err(|err| err.to_string())?;
        let stacked_operator = SparseRowOperator::stack(&[&line_operator, &point_operator])
            .map_err(|err| err.to_string())?;

        for model in REPORT_MODELS {
            for &component in components_for_model(model) {
                let prior = build_prior(&mesh, config, model, component)?;
                let variances = transformed_variances(&prior, &stacked_operator)?;
                let line_count = line_observables.len();
                let point_component_count = point_observables.len();
                let line_variances = &variances[..line_count];
                let point_component_variances =
                    &variances[line_count..line_count + point_component_count];

                rows.extend(line_rows(
                    n,
                    model,
                    component,
                    &prior,
                    &line_observables,
                    line_variances,
                ));
                rows.extend(point_component_rows(
                    n,
                    model,
                    component,
                    &prior,
                    &point_observables,
                    point_component_variances,
                ));
                rows.extend(point_trace_rows(
                    n,
                    model,
                    component,
                    &prior,
                    &point_observables,
                    point_component_variances,
                ));
            }
        }
    }

    let summaries = summary_rows(&rows);
    let fit_summaries = fit_summary_rows(&summaries);
    write_variance_csv(&config.output_dir.join("functional_variance.csv"), &rows)?;
    write_summary_csv(&config.output_dir.join("summary.csv"), &summaries)?;
    write_fit_summary_csv(&config.output_dir.join("fit_summary.csv"), &fit_summaries)?;
    write_readme(&config.output_dir.join("README.md"), config)?;

    Ok(BranchFunctionalConvergenceResult {
        rows,
        summaries,
        fit_summaries,
    })
}

fn build_mesh(n: usize) -> MeshData {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, n, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    MeshData {
        topology,
        coords,
        metric,
    }
}

fn build_prior(
    mesh: &MeshData,
    config: &BranchFunctionalConvergenceConfig,
    model: BranchFunctionalModel,
    component: BranchFunctionalComponent,
) -> Result<BuiltPrior, String> {
    let branch_config = HodgeMaternBranchConfig {
        kappa: config.kappa,
        tau: config.tau,
        alpha: MaternAlpha::Two,
    };
    match model {
        BranchFunctionalModel::BaselineHodgeMatern => {
            if component != BranchFunctionalComponent::Full1Form {
                return Err(format!(
                    "baseline Hodge-Matern model requires component {}, got {}",
                    BranchFunctionalComponent::Full1Form.as_str(),
                    component.as_str()
                ));
            }
            let hodge = build_hodge_laplacian_1form(&mesh.topology, &mesh.metric);
            let precision = build_matern_precision_1form_for_alpha_with_coords(
                &mesh.topology,
                &mesh.coords,
                &mesh.metric,
                &hodge,
                MaternAlpha::Two,
                Matern1FormConfig {
                    kappa: config.kappa,
                    tau: config.tau,
                    mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
                },
            )?;
            Ok(BuiltPrior::from_ambient_precision(precision))
        }
        BranchFunctionalModel::FormMatern | BranchFunctionalModel::PotentialMatern => {
            let spectrum = match model {
                BranchFunctionalModel::FormMatern => HodgeMaternSpectrum::Form,
                BranchFunctionalModel::PotentialMatern => HodgeMaternSpectrum::Potential,
                BranchFunctionalModel::BaselineHodgeMatern => unreachable!(),
            };
            build_hodge_matern_1form_prior_with_coords(
                &mesh.topology,
                &mesh.coords,
                &mesh.metric,
                spectrum,
                HodgeMatern1FormPriorConfig {
                    branches: component.branches()?,
                    exact: branch_config,
                    coexact: branch_config,
                    harmonic_dim: Some(0),
                    ..HodgeMatern1FormPriorConfig::default()
                },
            )
            .map(BuiltPrior::from_hodge_matern)
        }
    }
}

fn sparse_operator_from_fixed_observables(
    ncols: usize,
    observables: &[FixedObservable],
) -> Result<SparseRowOperator, String> {
    SparseRowOperator::new(
        ncols,
        observables
            .iter()
            .map(|observable| observable.row.clone())
            .collect(),
    )
    .map_err(|err| err.to_string())
}

fn point_component_observables(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<PointComponentObservable>, String> {
    let reconstruction = build_reconstructed_barycenter_field_operator(topology, coords)?;
    let mut observables = Vec::new();
    let mut point_index = 0;
    for x in DEFAULT_COORD_NUMERATORS {
        for y in DEFAULT_COORD_NUMERATORS {
            for z in DEFAULT_COORD_NUMERATORS {
                let target = [coord_value(x), coord_value(y), coord_value(z)];
                let cell_index = nearest_cell_barycenter(topology, coords, target)?;
                let point_id = format!("point_{point_index:03}_q{x}{y}{z}");
                for component_index in 0..reconstruction.component_count() {
                    let component_rows = reconstruction
                        .component_rows(component_index)
                        .ok_or_else(|| format!("missing point component {component_index}"))?;
                    let row = component_rows
                        .get(cell_index)
                        .ok_or_else(|| {
                            format!(
                                "nearest cell index {cell_index} is outside reconstruction rows"
                            )
                        })?
                        .clone();
                    observables.push(PointComponentObservable {
                        point_id: point_id.clone(),
                        component_index,
                        row,
                    });
                }
                point_index += 1;
            }
        }
    }
    Ok(observables)
}

fn nearest_cell_barycenter(
    topology: &Complex,
    coords: &MeshCoords,
    target: [f64; 3],
) -> Result<usize, String> {
    let mut best = None;
    for cell in topology.cells().handle_iter() {
        let barycenter = cell.coord_simplex(coords).barycenter();
        let point = [barycenter[0], barycenter[1], barycenter[2]];
        let distance2 = squared_distance(point, target);
        match best {
            Some((_, best_distance2)) if distance2 >= best_distance2 => {}
            _ => best = Some((cell.kidx(), distance2)),
        }
    }
    best.map(|(cell_index, _)| cell_index)
        .ok_or_else(|| "mesh has no top-dimensional cells".to_string())
}

fn transformed_variances(
    prior: &BuiltPrior,
    ambient_operator: &SparseRowOperator,
) -> Result<Vec<f64>, String> {
    let latent_operator = match &prior.latent_to_ambient {
        Some(latent_to_ambient) => {
            let transform_operator = sparse_row_operator_from_feec_csr(latent_to_ambient)?;
            SparseRowOperator::compose(ambient_operator, &transform_operator)
                .map_err(|err| err.to_string())?
        }
        None => {
            if ambient_operator.ncols != prior.latent_dimension {
                return Err(format!(
                    "ambient operator has {} columns but baseline prior has {} dofs",
                    ambient_operator.ncols, prior.latent_dimension
                ));
            }
            ambient_operator.clone()
        }
    };
    let precision = feec_csr_to_gmrf(&prior.precision);
    let factor = precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor branch precision: {err}"))?;
    let mut gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(prior.latent_dimension()), precision)
            .map_err(|err| format!("failed to build branch GMRF: {err}"))?
            .with_precision_sqrt(factor);
    let constraints = GmrfDenseMatrix::zeros(0, prior.latent_dimension());
    let variance = gmrf
        .exact_transformed_variance_decomposition(&latent_operator, &constraints)
        .map_err(|err| format!("failed to compute transformed variances: {err}"))?;
    Ok(variance.unconstrained_diag.iter().copied().collect())
}

fn line_rows(
    n: usize,
    model: BranchFunctionalModel,
    component: BranchFunctionalComponent,
    prior: &BuiltPrior,
    observables: &[FixedObservable],
    variances: &[f64],
) -> Vec<BranchFunctionalVarianceRow> {
    observables
        .iter()
        .zip(variances.iter().copied())
        .map(|(observable, variance)| {
            variance_row(
                n,
                model,
                component,
                BranchObservableKind::Line,
                observable.id.clone(),
                prior,
                observable.row.len(),
                observable.measure,
                variance,
            )
        })
        .collect()
}

fn point_component_rows(
    n: usize,
    model: BranchFunctionalModel,
    component: BranchFunctionalComponent,
    prior: &BuiltPrior,
    observables: &[PointComponentObservable],
    variances: &[f64],
) -> Vec<BranchFunctionalVarianceRow> {
    observables
        .iter()
        .zip(variances.iter().copied())
        .map(|(observable, variance)| {
            variance_row(
                n,
                model,
                component,
                BranchObservableKind::PointComponent,
                format!(
                    "{}_component{}",
                    observable.point_id, observable.component_index
                ),
                prior,
                observable.row.len(),
                1.0,
                variance,
            )
        })
        .collect()
}

fn point_trace_rows(
    n: usize,
    model: BranchFunctionalModel,
    component: BranchFunctionalComponent,
    prior: &BuiltPrior,
    observables: &[PointComponentObservable],
    variances: &[f64],
) -> Vec<BranchFunctionalVarianceRow> {
    let mut grouped = BTreeMap::<String, (f64, usize)>::new();
    for (observable, variance) in observables.iter().zip(variances.iter().copied()) {
        let entry = grouped
            .entry(observable.point_id.clone())
            .or_insert((0.0, 0));
        entry.0 += variance;
        entry.1 += observable.row.len();
    }
    grouped
        .into_iter()
        .map(|(point_id, (variance, support_entries))| {
            variance_row(
                n,
                model,
                component,
                BranchObservableKind::PointTrace,
                point_id,
                prior,
                support_entries,
                1.0,
                variance,
            )
        })
        .collect()
}

fn variance_row(
    n: usize,
    model: BranchFunctionalModel,
    component: BranchFunctionalComponent,
    observable_kind: BranchObservableKind,
    functional_id: String,
    prior: &BuiltPrior,
    support_entries: usize,
    geometric_measure: f64,
    raw_variance: f64,
) -> BranchFunctionalVarianceRow {
    BranchFunctionalVarianceRow {
        n,
        h: 1.0 / n as f64,
        alpha: MaternAlpha::Two,
        model,
        component,
        observable_kind,
        functional_id,
        latent_dofs: prior.latent_dimension(),
        precision_nnz: prior.precision.nnz(),
        transform_nnz: prior.transform_nnz(),
        support_entries,
        geometric_measure,
        raw_variance,
        normalized_variance: raw_variance / geometric_measure.max(EPS).powi(2),
        status: status_for(model, observable_kind).to_string(),
    }
}

fn status_for(model: BranchFunctionalModel, observable_kind: BranchObservableKind) -> &'static str {
    match (model, observable_kind) {
        (BranchFunctionalModel::BaselineHodgeMatern, _) => "expected_convergent",
        (BranchFunctionalModel::FormMatern, _) => "expected_convergent",
        (BranchFunctionalModel::PotentialMatern, BranchObservableKind::Line) => {
            "expected_log_borderline_growth"
        }
        (BranchFunctionalModel::PotentialMatern, _) => "expected_point_growth",
    }
}

fn summary_rows(rows: &[BranchFunctionalVarianceRow]) -> Vec<BranchFunctionalSummaryRow> {
    let mut grouped = BTreeMap::<
        (
            usize,
            BranchFunctionalModel,
            BranchFunctionalComponent,
            BranchObservableKind,
        ),
        Vec<&BranchFunctionalVarianceRow>,
    >::new();
    for row in rows {
        grouped
            .entry((row.n, row.model, row.component, row.observable_kind))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .map(|((n, model, component, observable_kind), group)| {
            let raw = group.iter().map(|row| row.raw_variance).collect::<Vec<_>>();
            let normalized = group
                .iter()
                .map(|row| row.normalized_variance)
                .collect::<Vec<_>>();
            BranchFunctionalSummaryRow {
                n,
                model,
                component,
                observable_kind,
                count: group.len(),
                mean_raw_variance: mean(&raw),
                median_raw_variance: median(raw.clone()),
                min_raw_variance: raw.iter().copied().fold(f64::INFINITY, f64::min),
                max_raw_variance: raw.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                stddev_raw_variance: stddev(&raw),
                median_normalized_variance: median(normalized),
            }
        })
        .collect()
}

fn fit_summary_rows(rows: &[BranchFunctionalSummaryRow]) -> Vec<BranchFunctionalFitSummaryRow> {
    let mut grouped = BTreeMap::<
        (
            BranchFunctionalModel,
            BranchFunctionalComponent,
            BranchObservableKind,
        ),
        Vec<&BranchFunctionalSummaryRow>,
    >::new();
    for row in rows {
        grouped
            .entry((row.model, row.component, row.observable_kind))
            .or_default()
            .push(row);
    }

    let mut summaries = Vec::new();
    for ((model, component, observable_kind), mut group) in grouped {
        group.sort_by_key(|row| row.n);
        if group.len() < 2 {
            continue;
        }
        let (diagnostic, value, expected, status) = match (model, observable_kind) {
            (BranchFunctionalModel::PotentialMatern, BranchObservableKind::Line) => {
                let xs = group
                    .iter()
                    .map(|row| (row.n as f64).ln())
                    .collect::<Vec<_>>();
                let ys = group
                    .iter()
                    .map(|row| row.median_raw_variance)
                    .collect::<Vec<_>>();
                (
                    "slope_vs_log_n",
                    linear_slope(&xs, &ys),
                    "positive",
                    "borderline_log_diagnostic",
                )
            }
            (BranchFunctionalModel::PotentialMatern, _) => {
                let xs = group
                    .iter()
                    .map(|row| (row.n as f64).ln())
                    .collect::<Vec<_>>();
                let ys = group
                    .iter()
                    .map(|row| row.median_raw_variance.max(EPS).ln())
                    .collect::<Vec<_>>();
                (
                    "loglog_slope_vs_n",
                    linear_slope(&xs, &ys),
                    "positive, near +1 for point trace",
                    "point_growth_diagnostic",
                )
            }
            _ => {
                let prev = group[group.len() - 2].median_raw_variance;
                let last = group[group.len() - 1].median_raw_variance;
                (
                    "finest_relative_change",
                    (last - prev).abs() / last.abs().max(EPS),
                    "small under convergence",
                    "convergence_diagnostic",
                )
            }
        };
        summaries.push(BranchFunctionalFitSummaryRow {
            model,
            component,
            observable_kind,
            diagnostic: diagnostic.to_string(),
            value,
            expected: expected.to_string(),
            status: status.to_string(),
        });
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

pub fn write_variance_csv(path: &Path, rows: &[BranchFunctionalVarianceRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,h,alpha,model,component,observable_kind,functional_id,latent_dofs,precision_nnz,transform_nnz,support_entries,geometric_measure,raw_variance,normalized_variance,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{}",
            row.n,
            row.h,
            row.alpha.as_u32(),
            row.model.as_str(),
            row.component.as_str(),
            row.observable_kind.as_str(),
            row.functional_id,
            row.latent_dofs,
            row.precision_nnz,
            row.transform_nnz,
            row.support_entries,
            row.geometric_measure,
            row.raw_variance,
            row.normalized_variance,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_summary_csv(path: &Path, rows: &[BranchFunctionalSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,model,component,observable_kind,count,mean_raw_variance,median_raw_variance,min_raw_variance,max_raw_variance,stddev_raw_variance,median_normalized_variance"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.n,
            row.model.as_str(),
            row.component.as_str(),
            row.observable_kind.as_str(),
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

pub fn write_fit_summary_csv(
    path: &Path,
    rows: &[BranchFunctionalFitSummaryRow],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,component,observable_kind,diagnostic,value,expected,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17e},{},{}",
            row.model.as_str(),
            row.component.as_str(),
            row.observable_kind.as_str(),
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_readme(path: &Path, config: &BranchFunctionalConvergenceConfig) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# Hodge-Decomposed Cube Mesh-Invariance Report")?;
    writeln!(writer)?;
    writeln!(writer, "- levels: {:?}", config.levels)?;
    writeln!(writer, "- kappa: {:.17}", config.kappa)?;
    writeln!(writer, "- tau: {:.17}", config.tau)?;
    writeln!(writer, "- alpha: 2")?;
    writeln!(
        writer,
        "- observables: 27 fixed line integrals, 81 Whitney 1-form barycenter component rows, and 27 barycenter trace rows"
    )?;
    writeln!(
        writer,
        "- models: baseline_hodge_matern full_1form, form_matern exact/coexact, and potential_matern exact/coexact"
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "All variances are exact sparse-solve transformed variances of sparse pushforward rows. No dense edge-space covariance matrix is formed."
    )?;
    Ok(())
}

fn validate_config(config: &BranchFunctionalConvergenceConfig) -> Result<(), String> {
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

fn coord_value(numerator: usize) -> f64 {
    numerator as f64 / COMMON_DENOMINATOR as f64
}

fn squared_distance(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    (lhs[0] - rhs[0]).powi(2) + (lhs[1] - rhs[1]).powi(2) + (lhs[2] - rhs[2]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(name: &str) -> BranchFunctionalConvergenceConfig {
        BranchFunctionalConvergenceConfig {
            levels: vec![4],
            output_dir: std::env::temp_dir().join(format!("{name}_{}", std::process::id())),
            kappa: 4.0,
            tau: 1.0,
        }
    }

    #[test]
    fn sparse_anchor_branch_functional_convergence_tiny_sweep_is_finite() {
        let result = run_sparse_anchor_branch_functional_convergence(&tiny_config(
            "sparse_anchor_branch_functional_convergence_test",
        ))
        .expect("tiny branch functional convergence sweep should run");

        let expected_rows_per_level = 5 * (27 + 81 + 27);
        assert_eq!(result.rows.len(), expected_rows_per_level);
        assert!(result.rows.iter().all(|row| {
            row.raw_variance.is_finite()
                && row.raw_variance >= 0.0
                && row.normalized_variance.is_finite()
                && row.latent_dofs > 0
                && row.precision_nnz > 0
                && row.transform_nnz > 0
        }));
        let expected_cases = [
            (
                BranchFunctionalModel::BaselineHodgeMatern,
                BranchFunctionalComponent::Full1Form,
            ),
            (
                BranchFunctionalModel::FormMatern,
                BranchFunctionalComponent::Exact,
            ),
            (
                BranchFunctionalModel::FormMatern,
                BranchFunctionalComponent::Coexact,
            ),
            (
                BranchFunctionalModel::PotentialMatern,
                BranchFunctionalComponent::Exact,
            ),
            (
                BranchFunctionalModel::PotentialMatern,
                BranchFunctionalComponent::Coexact,
            ),
        ];
        for (model, component) in expected_cases {
            assert!(
                result
                    .rows
                    .iter()
                    .any(|row| row.model == model && row.component == component),
                "missing rows for {} / {}",
                model.as_str(),
                component.as_str()
            );
        }
        assert!(result.rows.iter().all(|row| match row.model {
            BranchFunctionalModel::BaselineHodgeMatern =>
                row.component == BranchFunctionalComponent::Full1Form,
            BranchFunctionalModel::FormMatern | BranchFunctionalModel::PotentialMatern => matches!(
                row.component,
                BranchFunctionalComponent::Exact | BranchFunctionalComponent::Coexact
            ),
        }));
        assert_eq!(result.summaries.len(), 5 * 3);
        assert!(result.fit_summaries.is_empty());
    }

    #[test]
    fn sparse_anchor_branch_functional_convergence_fit_summaries_are_finite() {
        let mut summaries = Vec::new();
        for model in REPORT_MODELS {
            for &component in components_for_model(model) {
                for observable_kind in [
                    BranchObservableKind::Line,
                    BranchObservableKind::PointComponent,
                    BranchObservableKind::PointTrace,
                ] {
                    for (n, scale) in [(4, 1.0), (8, 1.2)] {
                        summaries.push(BranchFunctionalSummaryRow {
                            n,
                            model,
                            component,
                            observable_kind,
                            count: 3,
                            mean_raw_variance: scale,
                            median_raw_variance: scale,
                            min_raw_variance: scale,
                            max_raw_variance: scale,
                            stddev_raw_variance: 0.0,
                            median_normalized_variance: scale,
                        });
                    }
                }
            }
        }

        let fit = fit_summary_rows(&summaries);
        assert_eq!(fit.len(), 5 * 3);
        assert!(fit.iter().all(|row| row.value.is_finite()));
    }
}
