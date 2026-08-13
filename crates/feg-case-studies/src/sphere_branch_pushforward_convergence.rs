//! Analytic convergence study for exact/coexact 1-form pushforwards on S^2.
//!
//! The experiment compares ordinary potential priors with the spectrum-matched
//! sparse-anchor correction for alpha=2.  Outputs are ambient vector component
//! covariances reconstructed at selected triangle barycenters.

use crate::sphere_sparse_anchor_kernel_validation::{
    analytic_branch_covariance, analytic_joint_one_form_covariance,
};
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use feg_core::HodgeBranchKind;
use feg_infer::{
    prior::{
        matern::{one_form::build_reconstructed_barycenter_field_operator, MaternAlpha},
        sparse_anchor_hodge::{
            build_ordinary_potential_hodge_1form_prior_with_coords,
            build_sparse_anchor_hodge_1form_prior_with_coords,
            OrdinaryPotentialHodge1FormPriorConfig, SparseAnchorBranchConfig,
            SparseAnchorHodge1FormPriorConfig,
        },
    },
    sparse::{feec_csr_to_gmrf, sparse_row_operator_from_feec_csr},
};
use gmrf_core::{
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf, SparseRowOperator,
};
use manifold::{
    dim3::mesh_sphere_surface,
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

const DEFAULT_REFINEMENT_LEVELS: [usize; 4] = [1, 2, 3, 4];
const DEFAULT_ANALYTIC_LMAX: usize = 50;
const DEFAULT_MAX_CELLS: usize = 24;
const EPS: f64 = 1e-14;

#[derive(Debug, Clone)]
pub struct SphereBranchPushforwardConvergenceConfig {
    pub refinement_levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub analytic_lmax: usize,
    pub max_cells: usize,
}

impl Default for SphereBranchPushforwardConvergenceConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            refinement_levels: DEFAULT_REFINEMENT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/sphere_branch_pushforward_convergence"),
            kappa: 1.0,
            tau: 1.0,
            analytic_lmax: DEFAULT_ANALYTIC_LMAX,
            max_cells: DEFAULT_MAX_CELLS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SphereBranchPushforwardModel {
    OrdinaryPotential,
    SpectrallyCorrected,
}

impl SphereBranchPushforwardModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OrdinaryPotential => "ordinary_potential",
            Self::SpectrallyCorrected => "spectrally_corrected",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SphereBranchPushforwardComponent {
    Exact,
    Coexact,
    Joint,
}

impl SphereBranchPushforwardComponent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Coexact => "coexact",
            Self::Joint => "joint",
        }
    }

    fn branches(self) -> Vec<HodgeBranchKind> {
        match self {
            Self::Exact => vec![HodgeBranchKind::Exact],
            Self::Coexact => vec![HodgeBranchKind::Coexact],
            Self::Joint => vec![HodgeBranchKind::Exact, HodgeBranchKind::Coexact],
        }
    }
}

const REPORT_MODELS: [SphereBranchPushforwardModel; 2] = [
    SphereBranchPushforwardModel::OrdinaryPotential,
    SphereBranchPushforwardModel::SpectrallyCorrected,
];
const REPORT_COMPONENTS: [SphereBranchPushforwardComponent; 3] = [
    SphereBranchPushforwardComponent::Exact,
    SphereBranchPushforwardComponent::Coexact,
    SphereBranchPushforwardComponent::Joint,
];

#[derive(Debug, Clone)]
pub struct SphereBranchComponentVarianceRow {
    pub refinement_level: usize,
    pub h: f64,
    pub model: SphereBranchPushforwardModel,
    pub component: SphereBranchPushforwardComponent,
    pub point_index: usize,
    pub cell_index: usize,
    pub component_index: usize,
    pub functional_id: String,
    pub model_variance: f64,
    pub analytic_variance: f64,
    pub absolute_error: f64,
    pub relative_error: f64,
}

#[derive(Debug, Clone)]
pub struct SphereBranchCovarianceSummaryRow {
    pub refinement_level: usize,
    pub h: f64,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub cell_count: usize,
    pub selected_cell_count: usize,
    pub model: SphereBranchPushforwardModel,
    pub component: SphereBranchPushforwardComponent,
    pub latent_dofs: usize,
    pub precision_nnz: usize,
    pub covariance_dimension: usize,
    pub relative_frobenius_error: f64,
    pub diagonal_relative_l2_error: f64,
    pub best_scalar_rescaled_relative_frobenius_error: f64,
    pub best_scalar: f64,
    pub model_trace: f64,
    pub analytic_trace: f64,
    pub trace_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct SphereBranchFitSummaryRow {
    pub model: SphereBranchPushforwardModel,
    pub component: SphereBranchPushforwardComponent,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct SphereBranchPushforwardConvergenceResult {
    pub component_variances: Vec<SphereBranchComponentVarianceRow>,
    pub covariance_summaries: Vec<SphereBranchCovarianceSummaryRow>,
    pub fit_summaries: Vec<SphereBranchFitSummaryRow>,
}

struct SphereLevelData {
    topology: Complex,
    coords: MeshCoords,
    metric: MeshLengths,
    selected_cells: Vec<usize>,
    points: Vec<[f64; 3]>,
    operator: SparseRowOperator,
}

struct ModelCovariance {
    latent_dofs: usize,
    precision_nnz: usize,
    covariance: GmrfDenseMatrix,
}

pub fn run_sphere_branch_pushforward_convergence(
    config: &SphereBranchPushforwardConvergenceConfig,
) -> Result<SphereBranchPushforwardConvergenceResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut component_variances = Vec::new();
    let mut covariance_summaries = Vec::new();
    for &refinement_level in &config.refinement_levels {
        eprintln!("[sphere_branch_pushforward_convergence] level={refinement_level}");
        let level = build_level(refinement_level, config.max_cells)?;
        for model in REPORT_MODELS {
            for component in REPORT_COMPONENTS {
                let model_covariance = compute_model_covariance(model, component, &level, config)?;
                let analytic_covariance = analytic_covariance_for_component(
                    component,
                    &level.points,
                    config.kappa,
                    config.tau,
                    config.analytic_lmax,
                )?;
                component_variances.extend(component_variance_rows(
                    refinement_level,
                    &level,
                    model,
                    component,
                    &model_covariance.covariance,
                    &analytic_covariance,
                ));
                covariance_summaries.push(covariance_summary_row(
                    refinement_level,
                    &level,
                    model,
                    component,
                    &model_covariance,
                    &analytic_covariance,
                )?);
            }
        }
    }

    let fit_summaries = fit_summary_rows(&covariance_summaries);
    write_component_variance_csv(
        &config.output_dir.join("component_variance.csv"),
        &component_variances,
    )?;
    write_covariance_summary_csv(
        &config.output_dir.join("covariance_summary.csv"),
        &covariance_summaries,
    )?;
    write_fit_summary_csv(&config.output_dir.join("fit_summary.csv"), &fit_summaries)?;

    Ok(SphereBranchPushforwardConvergenceResult {
        component_variances,
        covariance_summaries,
        fit_summaries,
    })
}

fn build_level(refinement_level: usize, max_cells: usize) -> Result<SphereLevelData, String> {
    let surface = mesh_sphere_surface(refinement_level);
    let (topology, coords) = surface.into_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let selected_cells = evenly_spaced_indices(topology.cells().len(), max_cells);
    let points = normalized_cell_barycenters(&topology, &coords, &selected_cells)?;
    let operator = stacked_barycenter_reconstruction_operator(&topology, &coords, &selected_cells)?;
    Ok(SphereLevelData {
        topology,
        coords,
        metric,
        selected_cells,
        points,
        operator,
    })
}

fn compute_model_covariance(
    model: SphereBranchPushforwardModel,
    component: SphereBranchPushforwardComponent,
    level: &SphereLevelData,
    config: &SphereBranchPushforwardConvergenceConfig,
) -> Result<ModelCovariance, String> {
    let branches = component.branches();
    let branch_config = SparseAnchorBranchConfig {
        kappa: config.kappa,
        tau: config.tau,
        alpha: MaternAlpha::Two,
    };
    let prior = match model {
        SphereBranchPushforwardModel::OrdinaryPotential => {
            build_ordinary_potential_hodge_1form_prior_with_coords(
                &level.topology,
                &level.coords,
                &level.metric,
                OrdinaryPotentialHodge1FormPriorConfig {
                    branches,
                    exact: branch_config,
                    coexact: branch_config,
                    ..OrdinaryPotentialHodge1FormPriorConfig::default()
                },
            )
        }
        SphereBranchPushforwardModel::SpectrallyCorrected => {
            build_sparse_anchor_hodge_1form_prior_with_coords(
                &level.topology,
                &level.coords,
                &level.metric,
                SparseAnchorHodge1FormPriorConfig {
                    branches,
                    exact: branch_config,
                    coexact: branch_config,
                    harmonic_dim: Some(0),
                    ..SparseAnchorHodge1FormPriorConfig::default()
                },
            )
        }
    }?;
    let transform_operator = sparse_row_operator_from_feec_csr(&prior.latent_to_ambient)?;
    let latent_operator = SparseRowOperator::compose(&level.operator, &transform_operator)
        .map_err(|err| err.to_string())?;
    let covariance = transformed_covariance(&prior.precision, &latent_operator)?;
    Ok(ModelCovariance {
        latent_dofs: prior.latent_dimension(),
        precision_nnz: prior.precision.nnz(),
        covariance,
    })
}

fn transformed_covariance(
    precision: &FeecCsr,
    operator: &SparseRowOperator,
) -> Result<GmrfDenseMatrix, String> {
    let mut gmrf = Gmrf::from_mean_and_precision(
        GmrfVector::zeros(precision.nrows()),
        feec_csr_to_gmrf(precision),
    )
    .map_err(|err| err.to_string())?;
    gmrf.exact_transformed_covariance(operator)
        .map_err(|err| err.to_string())
}

fn analytic_covariance_for_component(
    component: SphereBranchPushforwardComponent,
    points: &[[f64; 3]],
    kappa: f64,
    tau: f64,
    lmax: usize,
) -> Result<GmrfDenseMatrix, String> {
    match component {
        SphereBranchPushforwardComponent::Exact => analytic_branch_covariance(
            points,
            HodgeBranchKind::Exact,
            kappa,
            tau,
            MaternAlpha::Two,
            lmax,
        ),
        SphereBranchPushforwardComponent::Coexact => analytic_branch_covariance(
            points,
            HodgeBranchKind::Coexact,
            kappa,
            tau,
            MaternAlpha::Two,
            lmax,
        ),
        SphereBranchPushforwardComponent::Joint => {
            analytic_joint_one_form_covariance(points, kappa, tau, MaternAlpha::Two, lmax)
        }
    }
}

fn component_variance_rows(
    refinement_level: usize,
    level: &SphereLevelData,
    model: SphereBranchPushforwardModel,
    component: SphereBranchPushforwardComponent,
    covariance: &GmrfDenseMatrix,
    analytic: &GmrfDenseMatrix,
) -> Vec<SphereBranchComponentVarianceRow> {
    let point_count = level.selected_cells.len();
    let mut rows = Vec::with_capacity(covariance.nrows().min(analytic.nrows()));
    for row in 0..covariance.nrows().min(analytic.nrows()) {
        let component_index = row / point_count;
        let point_index = row % point_count;
        let model_variance = covariance[(row, row)];
        let analytic_variance = analytic[(row, row)];
        rows.push(SphereBranchComponentVarianceRow {
            refinement_level,
            h: level.metric.mesh_width_max(),
            model,
            component,
            point_index,
            cell_index: level.selected_cells[point_index],
            component_index,
            functional_id: format!(
                "point_{point_index:03}_cell{}_component{}",
                level.selected_cells[point_index], component_index
            ),
            model_variance,
            analytic_variance,
            absolute_error: (model_variance - analytic_variance).abs(),
            relative_error: (model_variance - analytic_variance).abs()
                / analytic_variance.abs().max(EPS),
        });
    }
    rows
}

fn covariance_summary_row(
    refinement_level: usize,
    level: &SphereLevelData,
    model: SphereBranchPushforwardModel,
    component: SphereBranchPushforwardComponent,
    model_covariance: &ModelCovariance,
    analytic: &GmrfDenseMatrix,
) -> Result<SphereBranchCovarianceSummaryRow, String> {
    let covariance = &model_covariance.covariance;
    if covariance.nrows() != analytic.nrows() || covariance.ncols() != analytic.ncols() {
        return Err(format!(
            "covariance shape mismatch: model {}x{}, analytic {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            analytic.nrows(),
            analytic.ncols()
        ));
    }

    let diff = subtract_dense(covariance, analytic)?;
    let best_scalar = best_scalar_fit(covariance, analytic);
    let scaled_diff = subtract_dense(&scale_dense(covariance, best_scalar), analytic)?;
    let model_diag = diagonal(covariance);
    let analytic_diag = diagonal(analytic);
    let diagonal_error = model_diag
        .iter()
        .zip(analytic_diag.iter())
        .map(|(model_value, analytic_value)| (model_value - analytic_value).powi(2))
        .sum::<f64>()
        .sqrt();
    let model_trace = model_diag.iter().sum::<f64>();
    let analytic_trace = analytic_diag.iter().sum::<f64>();

    Ok(SphereBranchCovarianceSummaryRow {
        refinement_level,
        h: level.metric.mesh_width_max(),
        vertex_count: level.topology.vertices().len(),
        edge_count: level.topology.edges().len(),
        cell_count: level.topology.cells().len(),
        selected_cell_count: level.selected_cells.len(),
        model,
        component,
        latent_dofs: model_covariance.latent_dofs,
        precision_nnz: model_covariance.precision_nnz,
        covariance_dimension: covariance.nrows(),
        relative_frobenius_error: frobenius_norm(&diff) / frobenius_norm(analytic).max(EPS),
        diagonal_relative_l2_error: diagonal_error / vector_norm(&analytic_diag).max(EPS),
        best_scalar_rescaled_relative_frobenius_error: frobenius_norm(&scaled_diff)
            / frobenius_norm(analytic).max(EPS),
        best_scalar,
        model_trace,
        analytic_trace,
        trace_ratio: model_trace / analytic_trace.abs().max(EPS),
    })
}

fn fit_summary_rows(rows: &[SphereBranchCovarianceSummaryRow]) -> Vec<SphereBranchFitSummaryRow> {
    let mut grouped = BTreeMap::<
        (
            SphereBranchPushforwardModel,
            SphereBranchPushforwardComponent,
        ),
        Vec<&SphereBranchCovarianceSummaryRow>,
    >::new();
    for row in rows {
        grouped
            .entry((row.model, row.component))
            .or_default()
            .push(row);
    }

    let mut summaries = Vec::new();
    for ((model, component), mut group) in grouped {
        group.sort_by(|lhs, rhs| {
            lhs.h
                .partial_cmp(&rhs.h)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if group.len() < 2 {
            continue;
        }
        for (diagnostic, values, expected) in [
            (
                "loglog_slope_relative_frobenius_vs_h",
                group
                    .iter()
                    .map(|row| row.relative_frobenius_error)
                    .collect::<Vec<_>>(),
                "positive under convergence",
            ),
            (
                "loglog_slope_diagonal_relative_l2_vs_h",
                group
                    .iter()
                    .map(|row| row.diagonal_relative_l2_error)
                    .collect::<Vec<_>>(),
                "positive under diagonal convergence",
            ),
        ] {
            let hs = group.iter().map(|row| row.h).collect::<Vec<_>>();
            summaries.push(SphereBranchFitSummaryRow {
                model,
                component,
                diagnostic: diagnostic.to_string(),
                value: loglog_slope(&hs, &values),
                expected: expected.to_string(),
                status: fit_status(model).to_string(),
            });
        }
    }
    summaries
}

fn fit_status(model: SphereBranchPushforwardModel) -> &'static str {
    match model {
        SphereBranchPushforwardModel::SpectrallyCorrected => "expected_convergent",
        SphereBranchPushforwardModel::OrdinaryPotential => "diagnostic_baseline",
    }
}

fn stacked_barycenter_reconstruction_operator(
    topology: &Complex,
    coords: &MeshCoords,
    selected_cells: &[usize],
) -> Result<SparseRowOperator, String> {
    let reconstruction = build_reconstructed_barycenter_field_operator(topology, coords)?;
    let mut rows = Vec::with_capacity(reconstruction.component_count() * selected_cells.len());
    for component_index in 0..reconstruction.component_count() {
        let component_rows = reconstruction
            .component_rows(component_index)
            .ok_or_else(|| format!("missing barycenter component {component_index}"))?;
        for &cell_index in selected_cells {
            let row = component_rows
                .get(cell_index)
                .ok_or_else(|| format!("selected cell {cell_index} is out of bounds"))?;
            rows.push(row.clone());
        }
    }
    SparseRowOperator::new(topology.edges().len(), rows).map_err(|err| err.to_string())
}

fn normalized_cell_barycenters(
    topology: &Complex,
    coords: &MeshCoords,
    selected_cells: &[usize],
) -> Result<Vec<[f64; 3]>, String> {
    let cells = topology.cells().handle_iter().collect::<Vec<_>>();
    let mut points = Vec::with_capacity(selected_cells.len());
    for &cell_index in selected_cells {
        let cell = cells
            .get(cell_index)
            .ok_or_else(|| format!("selected cell {cell_index} is out of bounds"))?;
        let barycenter = cell.coord_simplex(coords).barycenter();
        points.push(normalize3([barycenter[0], barycenter[1], barycenter[2]])?);
    }
    Ok(points)
}

fn evenly_spaced_indices(len: usize, max_count: usize) -> Vec<usize> {
    if len <= max_count {
        return (0..len).collect();
    }
    (0..max_count).map(|i| i * len / max_count).collect()
}

fn normalize3(point: [f64; 3]) -> Result<[f64; 3], String> {
    let norm = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    if norm <= EPS {
        return Err("cannot normalize zero barycenter".to_string());
    }
    Ok([point[0] / norm, point[1] / norm, point[2] / norm])
}

fn diagonal(matrix: &GmrfDenseMatrix) -> Vec<f64> {
    (0..matrix.nrows().min(matrix.ncols()))
        .map(|index| matrix[(index, index)])
        .collect()
}

fn frobenius_norm(matrix: &GmrfDenseMatrix) -> f64 {
    let mut sum = 0.0;
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            sum += matrix[(row, col)] * matrix[(row, col)];
        }
    }
    sum.sqrt()
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn best_scalar_fit(source: &GmrfDenseMatrix, target: &GmrfDenseMatrix) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in 0..source.nrows() {
        for col in 0..source.ncols() {
            numerator += source[(row, col)] * target[(row, col)];
            denominator += source[(row, col)] * source[(row, col)];
        }
    }
    numerator / denominator.max(EPS)
}

fn subtract_dense(lhs: &GmrfDenseMatrix, rhs: &GmrfDenseMatrix) -> Result<GmrfDenseMatrix, String> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err("dense matrix shapes must match".to_string());
    }
    Ok(GmrfDenseMatrix::from_fn(
        lhs.nrows(),
        lhs.ncols(),
        |i, j| lhs[(i, j)] - rhs[(i, j)],
    ))
}

fn scale_dense(matrix: &GmrfDenseMatrix, scale: f64) -> GmrfDenseMatrix {
    GmrfDenseMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        scale * matrix[(i, j)]
    })
}

fn loglog_slope(hs: &[f64], values: &[f64]) -> f64 {
    if hs.len() != values.len() || hs.len() < 2 {
        return f64::NAN;
    }
    let xs = hs.iter().map(|h| h.max(EPS).ln()).collect::<Vec<_>>();
    let ys = values
        .iter()
        .map(|value| value.abs().max(EPS).ln())
        .collect::<Vec<_>>();
    linear_slope(&xs, &ys)
}

fn linear_slope(xs: &[f64], ys: &[f64]) -> f64 {
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

pub fn write_component_variance_csv(
    path: &Path,
    rows: &[SphereBranchComponentVarianceRow],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "refinement_level,h,model,component,point_index,cell_index,component_index,functional_id,model_variance,analytic_variance,absolute_error,relative_error"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.refinement_level,
            row.h,
            row.model.as_str(),
            row.component.as_str(),
            row.point_index,
            row.cell_index,
            row.component_index,
            row.functional_id,
            row.model_variance,
            row.analytic_variance,
            row.absolute_error,
            row.relative_error
        )?;
    }
    Ok(())
}

pub fn write_covariance_summary_csv(
    path: &Path,
    rows: &[SphereBranchCovarianceSummaryRow],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "refinement_level,h,vertex_count,edge_count,cell_count,selected_cell_count,model,component,latent_dofs,precision_nnz,covariance_dimension,relative_frobenius_error,diagonal_relative_l2_error,best_scalar_rescaled_relative_frobenius_error,best_scalar,model_trace,analytic_trace,trace_ratio"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.refinement_level,
            row.h,
            row.vertex_count,
            row.edge_count,
            row.cell_count,
            row.selected_cell_count,
            row.model.as_str(),
            row.component.as_str(),
            row.latent_dofs,
            row.precision_nnz,
            row.covariance_dimension,
            row.relative_frobenius_error,
            row.diagonal_relative_l2_error,
            row.best_scalar_rescaled_relative_frobenius_error,
            row.best_scalar,
            row.model_trace,
            row.analytic_trace,
            row.trace_ratio
        )?;
    }
    Ok(())
}

pub fn write_fit_summary_csv(path: &Path, rows: &[SphereBranchFitSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "model,component,diagnostic,value,expected,status")?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{:.17e},{},{}",
            row.model.as_str(),
            row.component.as_str(),
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

fn validate_config(config: &SphereBranchPushforwardConvergenceConfig) -> Result<(), String> {
    if config.refinement_levels.is_empty() {
        return Err("at least one sphere refinement level is required".to_string());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if config.analytic_lmax == 0 {
        return Err("analytic_lmax must be positive".to_string());
    }
    if config.max_cells == 0 {
        return Err("max_cells must be positive".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn sphere_branch_pushforward_convergence_corrected_errors_decrease() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        let output_dir =
            std::env::temp_dir().join(format!("sphere_branch_pushforward_convergence_{stamp}"));
        let config = SphereBranchPushforwardConvergenceConfig {
            refinement_levels: vec![1, 2, 3],
            output_dir: output_dir.clone(),
            analytic_lmax: 20,
            max_cells: 8,
            ..SphereBranchPushforwardConvergenceConfig::default()
        };
        let result = run_sphere_branch_pushforward_convergence(&config)
            .expect("sphere branch pushforward convergence should run");

        assert_eq!(result.covariance_summaries.len(), 3 * 2 * 3);
        assert!(result.covariance_summaries.iter().all(|row| {
            row.relative_frobenius_error.is_finite()
                && row.diagonal_relative_l2_error.is_finite()
                && row
                    .best_scalar_rescaled_relative_frobenius_error
                    .is_finite()
                && row.trace_ratio.is_finite()
        }));
        assert!(result.component_variances.iter().all(|row| {
            row.model_variance.is_finite()
                && row.analytic_variance.is_finite()
                && row.relative_error.is_finite()
        }));
        assert!(!result.fit_summaries.is_empty());
        assert!(output_dir.join("component_variance.csv").exists());
        assert!(output_dir.join("covariance_summary.csv").exists());
        assert!(output_dir.join("fit_summary.csv").exists());

        for component in [
            SphereBranchPushforwardComponent::Exact,
            SphereBranchPushforwardComponent::Coexact,
        ] {
            let errors = corrected_errors(&result.covariance_summaries, component);
            assert_eq!(errors.len(), 3);
            assert!(
                errors[1] < errors[0],
                "{} corrected error should decrease from level 1 to 2: {:?}",
                component.as_str(),
                errors
            );
            assert!(
                errors[2] < errors[1],
                "{} corrected error should decrease from level 2 to 3: {:?}",
                component.as_str(),
                errors
            );
        }

        let _ = fs::remove_dir_all(output_dir);
    }

    fn corrected_errors(
        rows: &[SphereBranchCovarianceSummaryRow],
        component: SphereBranchPushforwardComponent,
    ) -> Vec<f64> {
        let mut rows = rows
            .iter()
            .filter(|row| {
                row.model == SphereBranchPushforwardModel::SpectrallyCorrected
                    && row.component == component
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.refinement_level);
        rows.iter()
            .map(|row| row.relative_frobenius_error)
            .collect()
    }
}
