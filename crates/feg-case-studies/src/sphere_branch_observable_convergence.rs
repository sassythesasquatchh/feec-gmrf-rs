//! Variance convergence for exact/coexact smooth 1-form observables on S^2.
//!
//! This experiment tests fixed real vector-spherical-harmonic observables against
//! alpha=2 ordinary-potential and spectrum-matched sparse-anchor branch priors.
//! It computes only transformed marginal variances, not full covariance matrices.

use crate::sphere_sparse_anchor_kernel_validation::analytic_branch_covariance;
use common::linalg::nalgebra::Vector as FeecVector;
use exterior::field::DiffFormClosure;
use feec_gmrf::prelude::{
    HodgeBranchKind, HodgeOneFormPrior, HodgeOneFormPriorBuilder, LinearMap,
    OrdinaryPotentialHodge1FormPriorConfig, SparseAnchorBranchConfig,
    SparseAnchorHodge1FormPriorConfig, VarianceMethod,
};
use feg_infer::prior::matern::{
    one_form::build_reconstructed_barycenter_field_operator, MaternAlpha,
};
use formoniq::{assemble::assemble_galvec, operators::SourceElVec};
use manifold::{
    dim3::mesh_sphere_surface,
    geometry::{
        coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexHandleExt},
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
    time::Instant,
};

const DEFAULT_LEVELS: [usize; 5] = [1, 2, 3, 4, 5];
const DEFAULT_LMAX: usize = 3;
const DEFAULT_POINTWISE_ANALYTIC_LMAX: usize = 400;
const SPARSE_ROW_TOLERANCE: f64 = 1e-14;
const EPS: f64 = 1e-14;
const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct SphereBranchObservableConvergenceConfig {
    pub levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub lmax: usize,
    pub pointwise_analytic_lmax: usize,
}

impl Default for SphereBranchObservableConvergenceConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            levels: DEFAULT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/sphere_branch_observable_convergence"),
            kappa: 1.0,
            tau: 1.0,
            lmax: DEFAULT_LMAX,
            pointwise_analytic_lmax: DEFAULT_POINTWISE_ANALYTIC_LMAX,
        }
    }
}

impl SphereBranchObservableConvergenceConfig {
    /// Cheap deterministic configuration intended for continuous integration.
    pub fn smoke() -> Self {
        Self {
            levels: vec![1],
            lmax: 1,
            pointwise_analytic_lmax: 8,
            ..Self::default()
        }
    }

    /// Immutable configuration used by the submitted thesis.
    ///
    /// The submitted observable study includes refinement level 6; the
    /// interactive default stops at level 5 to keep ad-hoc runs manageable.
    pub fn thesis_submitted() -> Self {
        Self {
            levels: (1..=6).collect(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SphereBranchObservableModel {
    OrdinaryPotential,
    SpectrallyCorrected,
}

impl SphereBranchObservableModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OrdinaryPotential => "ordinary_potential",
            Self::SpectrallyCorrected => "spectrally_corrected",
        }
    }
}

const REPORT_MODELS: [SphereBranchObservableModel; 2] = [
    SphereBranchObservableModel::OrdinaryPotential,
    SphereBranchObservableModel::SpectrallyCorrected,
];
const REPORT_BRANCHES: [HodgeBranchKind; 2] = [HodgeBranchKind::Exact, HodgeBranchKind::Coexact];

#[derive(Debug, Clone)]
pub struct SphereObservableVarianceRow {
    pub refinement_level: usize,
    pub h: f64,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub cell_count: usize,
    pub model: SphereBranchObservableModel,
    pub prior_branch: HodgeBranchKind,
    pub observable_branch: HodgeBranchKind,
    pub kind: &'static str,
    pub ell: usize,
    pub m: isize,
    pub observable_id: String,
    pub latent_dofs: usize,
    pub precision_nnz: usize,
    pub factor_nnz: usize,
    pub row_nnz: usize,
    pub model_variance: f64,
    pub analytic_variance: f64,
    pub target_scale: f64,
    pub absolute_error: f64,
    pub scaled_error: f64,
    pub build_seconds: f64,
    pub factor_seconds: f64,
    pub variance_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct SphereObservableSummaryRow {
    pub refinement_level: usize,
    pub h: f64,
    pub model: SphereBranchObservableModel,
    pub prior_branch: HodgeBranchKind,
    pub observable_branch: HodgeBranchKind,
    pub kind: &'static str,
    pub count: usize,
    pub mean_model_variance: f64,
    pub median_model_variance: f64,
    pub rms_scaled_error: f64,
    pub median_scaled_error: f64,
    pub max_scaled_error: f64,
}

#[derive(Debug, Clone)]
pub struct SphereObservableFitSummaryRow {
    pub model: SphereBranchObservableModel,
    pub prior_branch: HodgeBranchKind,
    pub observable_branch: HodgeBranchKind,
    pub kind: &'static str,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct SpherePointwiseVarianceRow {
    pub refinement_level: usize,
    pub h: f64,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub cell_count: usize,
    pub model: SphereBranchObservableModel,
    pub prior_branch: HodgeBranchKind,
    pub point_index: usize,
    pub cell_index: usize,
    pub component_index: usize,
    pub component_name: &'static str,
    pub target_x: f64,
    pub target_y: f64,
    pub target_z: f64,
    pub eval_x: f64,
    pub eval_y: f64,
    pub eval_z: f64,
    pub direction_error: f64,
    pub latent_dofs: usize,
    pub precision_nnz: usize,
    pub factor_nnz: usize,
    pub row_nnz: usize,
    pub model_variance: f64,
    pub analytic_variance: f64,
    pub absolute_error: f64,
    pub relative_error: f64,
    pub build_seconds: f64,
    pub factor_seconds: f64,
    pub variance_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct SpherePointwiseSummaryRow {
    pub refinement_level: usize,
    pub h: f64,
    pub model: SphereBranchObservableModel,
    pub prior_branch: HodgeBranchKind,
    pub component_family: &'static str,
    pub count: usize,
    pub mean_model_variance: f64,
    pub median_model_variance: f64,
    pub rms_relative_error: f64,
    pub median_relative_error: f64,
    pub max_relative_error: f64,
}

#[derive(Debug, Clone)]
pub struct SpherePointwiseFitSummaryRow {
    pub model: SphereBranchObservableModel,
    pub prior_branch: HodgeBranchKind,
    pub component_family: &'static str,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct SphereBranchObservableConvergenceResult {
    pub variance_rows: Vec<SphereObservableVarianceRow>,
    pub summary_rows: Vec<SphereObservableSummaryRow>,
    pub fit_summary_rows: Vec<SphereObservableFitSummaryRow>,
    pub pointwise_rows: Vec<SpherePointwiseVarianceRow>,
    pub pointwise_summary_rows: Vec<SpherePointwiseSummaryRow>,
    pub pointwise_fit_summary_rows: Vec<SpherePointwiseFitSummaryRow>,
}

struct MeshData {
    topology: Complex,
    coords: MeshCoords,
    metric: MeshLengths,
}

struct PriorVarianceReport {
    latent_dofs: usize,
    precision_nnz: usize,
    factor_nnz: usize,
    smooth_variances: Vec<f64>,
    pointwise_variances: Vec<f64>,
    build_seconds: f64,
    factor_seconds: f64,
    variance_seconds: f64,
}

struct ObservableOperator {
    operator: LinearMap,
    row_nnz: Vec<usize>,
}

struct PointwiseOperator {
    operator: LinearMap,
    row_nnz: Vec<usize>,
    observations: Vec<PointwiseObservation>,
}

#[derive(Debug, Clone)]
struct PointwiseObservation {
    point_index: usize,
    cell_index: usize,
    component_index: usize,
    component_name: &'static str,
    target: [f64; 3],
    eval_point: [f64; 3],
    direction_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SphereObservable {
    branch: HodgeBranchKind,
    ell: usize,
    m: isize,
}

impl SphereObservable {
    fn id(self) -> String {
        format!("{}_l{}_m{}", self.branch.as_str(), self.ell, self.m)
    }

    fn lambda(self) -> f64 {
        lambda_for_ell(self.ell)
    }

    fn source(self) -> DiffFormClosure {
        DiffFormClosure::one_form(
            move |point| {
                let unit = normalize3([point[0], point[1], point[2]]);
                let value = self.vector_proxy(unit);
                FeecVector::from_vec(vec![value[0], value[1], value[2]])
            },
            3,
        )
    }

    fn vector_proxy(self, unit: [f64; 3]) -> [f64; 3] {
        let (value, grad) = real_spherical_harmonic_value_grad(self.ell, self.m, unit);
        let lambda = self.lambda();
        let surface_grad = sub3(grad, scale3(unit, self.ell as f64 * value));
        let exact = scale3(surface_grad, lambda.sqrt().recip());
        match self.branch {
            HodgeBranchKind::Exact => exact,
            HodgeBranchKind::Coexact => cross3(unit, exact),
            HodgeBranchKind::Harmonic => [0.0, 0.0, 0.0],
        }
    }
}

pub fn run_sphere_branch_observable_convergence(
    config: &SphereBranchObservableConvergenceConfig,
) -> Result<SphereBranchObservableConvergenceResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let observables = observables(config.lmax);
    let mut variance_rows = Vec::new();
    let mut pointwise_rows = Vec::new();
    for &level in &config.levels {
        eprintln!("[sphere_branch_observable_convergence] level={level}");
        let mesh = build_mesh(level);
        let smooth_operator = observable_operator(&mesh, &observables)?;
        let pointwise_operator = pointwise_operator(&mesh)?;
        for model in REPORT_MODELS {
            for prior_branch in REPORT_BRANCHES {
                eprintln!(
                    "[sphere_branch_observable_convergence] level={level} model={} branch={}",
                    model.as_str(),
                    prior_branch.as_str()
                );
                let report = compute_prior_variances(
                    &mesh,
                    config,
                    model,
                    prior_branch,
                    &smooth_operator.operator,
                    &pointwise_operator.operator,
                )?;
                eprintln!(
                    "[sphere_branch_observable_convergence] level={level} model={} branch={} latent_dofs={} factor_nnz={} build={:.3}s factor={:.3}s variance={:.3}s",
                    model.as_str(),
                    prior_branch.as_str(),
                    report.latent_dofs,
                    report.factor_nnz,
                    report.build_seconds,
                    report.factor_seconds,
                    report.variance_seconds
                );
                variance_rows.extend(variance_rows_for_prior(
                    level,
                    &mesh,
                    config,
                    model,
                    prior_branch,
                    &observables,
                    &smooth_operator.row_nnz,
                    &report,
                ));
                pointwise_rows.extend(pointwise_rows_for_prior(
                    level,
                    &mesh,
                    config,
                    model,
                    prior_branch,
                    &pointwise_operator,
                    &report,
                )?);
            }
        }
    }

    let summary_rows = summary_rows(&variance_rows);
    let fit_summary_rows = fit_summary_rows(&summary_rows);
    let pointwise_summary_rows = pointwise_summary_rows(&pointwise_rows);
    let pointwise_fit_summary_rows = pointwise_fit_summary_rows(&pointwise_summary_rows);
    write_observable_variance_csv(
        &config.output_dir.join("observable_variance.csv"),
        &variance_rows,
    )?;
    write_summary_csv(&config.output_dir.join("summary.csv"), &summary_rows)?;
    write_fit_summary_csv(
        &config.output_dir.join("fit_summary.csv"),
        &fit_summary_rows,
    )?;
    write_pointwise_variance_csv(
        &config.output_dir.join("pointwise_variance.csv"),
        &pointwise_rows,
    )?;
    write_pointwise_summary_csv(
        &config.output_dir.join("pointwise_summary.csv"),
        &pointwise_summary_rows,
    )?;
    write_pointwise_fit_summary_csv(
        &config.output_dir.join("pointwise_fit_summary.csv"),
        &pointwise_fit_summary_rows,
    )?;
    write_readme(&config.output_dir.join("README.md"), config)?;

    Ok(SphereBranchObservableConvergenceResult {
        variance_rows,
        summary_rows,
        fit_summary_rows,
        pointwise_rows,
        pointwise_summary_rows,
        pointwise_fit_summary_rows,
    })
}

fn build_mesh(level: usize) -> MeshData {
    let surface = mesh_sphere_surface(level);
    let (topology, coords) = surface.into_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    MeshData {
        topology,
        coords,
        metric,
    }
}

fn observables(lmax: usize) -> Vec<SphereObservable> {
    let mut out = Vec::new();
    for ell in 1..=lmax {
        for m in -(ell as isize)..=(ell as isize) {
            for branch in REPORT_BRANCHES {
                out.push(SphereObservable { branch, ell, m });
            }
        }
    }
    out
}

fn observable_operator(
    mesh: &MeshData,
    observables: &[SphereObservable],
) -> Result<ObservableOperator, String> {
    let qr = Some(SimplexQuadRule::order3(mesh.topology.dim()));
    let rows = observables
        .iter()
        .copied()
        .map(|observable| {
            let source = observable.source();
            let load = assemble_galvec(
                &mesh.topology,
                &mesh.metric,
                SourceElVec::new(&source, &mesh.coords, qr.clone()),
            );
            vector_to_sparse_row(&load, SPARSE_ROW_TOLERANCE)
        })
        .collect::<Vec<_>>();
    let row_nnz = rows.iter().map(Vec::len).collect::<Vec<_>>();
    Ok(ObservableOperator {
        operator: LinearMap::weighted_rows(mesh.topology.edges().len(), &rows)
            .map_err(|err| err.to_string())?,
        row_nnz,
    })
}

fn pointwise_operator(mesh: &MeshData) -> Result<PointwiseOperator, String> {
    let directions = fixed_pointwise_directions();
    let selected = select_nearest_unique_cells(mesh, &directions)?;
    let reconstruction =
        build_reconstructed_barycenter_field_operator(&mesh.topology, &mesh.coords)?;
    if reconstruction.ambient_dim() != 3 {
        return Err(format!(
            "expected 3 ambient reconstruction components, got {}",
            reconstruction.ambient_dim()
        ));
    }

    let point_count = selected.len();
    let mut rows = Vec::with_capacity(3 * point_count);
    let mut observations = Vec::with_capacity(3 * point_count);
    for component_index in 0..3 {
        let component_rows = reconstruction
            .component_rows(component_index)
            .ok_or_else(|| format!("missing barycenter component {component_index}"))?;
        for selected_point in &selected {
            let row = component_rows
                .get(selected_point.cell_index)
                .ok_or_else(|| {
                    format!(
                        "selected cell {} is out of bounds for barycenter reconstruction",
                        selected_point.cell_index
                    )
                })?
                .clone();
            rows.push(row);
            observations.push(PointwiseObservation {
                point_index: selected_point.point_index,
                cell_index: selected_point.cell_index,
                component_index,
                component_name: component_name(component_index)?,
                target: selected_point.target,
                eval_point: selected_point.eval_point,
                direction_error: selected_point.direction_error,
            });
        }
    }
    let row_nnz = rows.iter().map(Vec::len).collect::<Vec<_>>();
    Ok(PointwiseOperator {
        operator: LinearMap::weighted_rows(mesh.topology.edges().len(), &rows)
            .map_err(|err| err.to_string())?,
        row_nnz,
        observations,
    })
}

#[derive(Debug, Clone)]
struct SelectedPointwiseCell {
    point_index: usize,
    cell_index: usize,
    target: [f64; 3],
    eval_point: [f64; 3],
    direction_error: f64,
}

fn fixed_pointwise_directions() -> Vec<[f64; 3]> {
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    let signs = [-1.0, 1.0];
    let mut directions = Vec::with_capacity(8);
    for sx in signs {
        for sy in signs {
            for sz in signs {
                directions.push([sx * inv_sqrt3, sy * inv_sqrt3, sz * inv_sqrt3]);
            }
        }
    }
    directions
}

fn select_nearest_unique_cells(
    mesh: &MeshData,
    directions: &[[f64; 3]],
) -> Result<Vec<SelectedPointwiseCell>, String> {
    if mesh.topology.cells().len() < directions.len() {
        return Err(format!(
            "need at least {} cells to select unique pointwise barycenters, got {}",
            directions.len(),
            mesh.topology.cells().len()
        ));
    }

    let cells = mesh.topology.cells().handle_iter().collect::<Vec<_>>();
    let barycenters = cells
        .iter()
        .map(|cell| {
            let barycenter = cell.coord_simplex(&mesh.coords).barycenter();
            normalize3([barycenter[0], barycenter[1], barycenter[2]])
        })
        .collect::<Vec<_>>();

    let mut used = vec![false; barycenters.len()];
    let mut selected = Vec::with_capacity(directions.len());
    for (point_index, &target) in directions.iter().enumerate() {
        let (cell_index, eval_point, distance_squared) = barycenters
            .iter()
            .copied()
            .enumerate()
            .filter(|(cell_index, point)| {
                !used[*cell_index] && point.iter().all(|value| value.is_finite())
            })
            .map(|(cell_index, point)| {
                let delta = sub3(point, target);
                (cell_index, point, dot3(delta, delta))
            })
            .min_by(|lhs, rhs| {
                lhs.2
                    .partial_cmp(&rhs.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| {
                format!("failed to select a finite unique cell for pointwise point {point_index}")
            })?;
        used[cell_index] = true;
        selected.push(SelectedPointwiseCell {
            point_index,
            cell_index,
            target,
            eval_point,
            direction_error: distance_squared.sqrt(),
        });
    }
    Ok(selected)
}

fn component_name(component_index: usize) -> Result<&'static str, String> {
    match component_index {
        0 => Ok("x"),
        1 => Ok("y"),
        2 => Ok("z"),
        _ => Err(format!(
            "unsupported ambient component index {component_index}"
        )),
    }
}

fn compute_prior_variances(
    mesh: &MeshData,
    config: &SphereBranchObservableConvergenceConfig,
    model: SphereBranchObservableModel,
    prior_branch: HodgeBranchKind,
    smooth_operator: &LinearMap,
    pointwise_operator: &LinearMap,
) -> Result<PriorVarianceReport, String> {
    let started = Instant::now();
    let prior = build_prior(mesh, config, model, prior_branch)?;
    let build_seconds = started.elapsed().as_secs_f64();

    let smooth_latent_operator = smooth_operator
        .compose(prior.ambient_map())
        .map_err(|err| err.to_string())?;
    let pointwise_latent_operator = pointwise_operator
        .compose(prior.ambient_map())
        .map_err(|err| err.to_string())?;
    let stacked_operator = LinearMap::stack(&[smooth_latent_operator, pointwise_latent_operator])
        .map_err(|err| err.to_string())?;
    let factor_started = Instant::now();
    let mut factored = prior
        .latent_prior()
        .factor()
        .map_err(|err| format!("failed to factor precision: {err}"))?;
    let factor_seconds = factor_started.elapsed().as_secs_f64();
    let factor_nnz = factored
        .factorization_diagnostics()
        .map_err(|err| err.to_string())?
        .factor_nonzeros;
    let variance_started = Instant::now();
    let estimate = factored
        .pushforward_variance_estimate(&stacked_operator, VarianceMethod::Exact)
        .map_err(|err| format!("failed to compute transformed variances: {err}"))?;
    let smooth_count = smooth_operator.output_dimension();
    let pointwise_count = pointwise_operator.output_dimension();
    let values = estimate.values.as_slice();
    if values.len() != smooth_count + pointwise_count {
        return Err(format!(
            "variance solve returned {} rows for {} smooth + {} pointwise rows",
            values.len(),
            smooth_count,
            pointwise_count
        ));
    }
    Ok(PriorVarianceReport {
        latent_dofs: prior.latent_dimension(),
        precision_nnz: prior.latent_prior().precision().nnz(),
        factor_nnz,
        smooth_variances: values[..smooth_count].to_vec(),
        pointwise_variances: values[smooth_count..].to_vec(),
        build_seconds,
        factor_seconds,
        variance_seconds: variance_started.elapsed().as_secs_f64(),
    })
}

fn build_prior(
    mesh: &MeshData,
    config: &SphereBranchObservableConvergenceConfig,
    model: SphereBranchObservableModel,
    prior_branch: HodgeBranchKind,
) -> Result<HodgeOneFormPrior, String> {
    let branch_config = SparseAnchorBranchConfig {
        kappa: config.kappa,
        tau: config.tau,
        alpha: MaternAlpha::Two,
    };
    let branches = vec![prior_branch];
    let builder = match model {
        SphereBranchObservableModel::OrdinaryPotential => {
            HodgeOneFormPriorBuilder::ordinary_potential(
                &mesh.topology,
                &mesh.metric,
                OrdinaryPotentialHodge1FormPriorConfig {
                    branches,
                    exact: branch_config,
                    coexact: branch_config,
                    ..OrdinaryPotentialHodge1FormPriorConfig::default()
                },
            )
        }
        SphereBranchObservableModel::SpectrallyCorrected => {
            HodgeOneFormPriorBuilder::sparse_anchor(
                &mesh.topology,
                &mesh.metric,
                SparseAnchorHodge1FormPriorConfig {
                    branches,
                    exact: branch_config,
                    coexact: branch_config,
                    harmonic_dim: Some(0),
                    ..SparseAnchorHodge1FormPriorConfig::default()
                },
            )
        }
    };
    builder
        .with_coords(&mesh.coords)
        .build()
        .map_err(|err| err.to_string())
}

// Each row records mesh, prior branch, observable support, and analytic reference
// independently. The sphere-observable smoke profile validates these artifacts.
#[allow(clippy::too_many_arguments)]
fn variance_rows_for_prior(
    refinement_level: usize,
    mesh: &MeshData,
    config: &SphereBranchObservableConvergenceConfig,
    model: SphereBranchObservableModel,
    prior_branch: HodgeBranchKind,
    observables: &[SphereObservable],
    row_nnz: &[usize],
    report: &PriorVarianceReport,
) -> Vec<SphereObservableVarianceRow> {
    observables
        .iter()
        .enumerate()
        .map(|(index, observable)| {
            let matched = prior_branch == observable.branch;
            let analytic_matched =
                analytic_matched_variance(observable.ell, config.kappa, config.tau);
            let analytic_variance =
                analytic_target_variance(prior_branch, *observable, config.kappa, config.tau);
            let model_variance = report.smooth_variances[index];
            let absolute_error = (model_variance - analytic_variance).abs();
            let target_scale = analytic_matched.max(EPS);
            SphereObservableVarianceRow {
                refinement_level,
                h: mesh.metric.mesh_width_max(),
                vertex_count: mesh.topology.vertices().len(),
                edge_count: mesh.topology.edges().len(),
                cell_count: mesh.topology.cells().len(),
                model,
                prior_branch,
                observable_branch: observable.branch,
                kind: if matched { "matched" } else { "leakage" },
                ell: observable.ell,
                m: observable.m,
                observable_id: observable.id(),
                latent_dofs: report.latent_dofs,
                precision_nnz: report.precision_nnz,
                factor_nnz: report.factor_nnz,
                row_nnz: row_nnz[index],
                model_variance,
                analytic_variance,
                target_scale,
                absolute_error,
                scaled_error: absolute_error / target_scale,
                build_seconds: report.build_seconds,
                factor_seconds: report.factor_seconds,
                variance_seconds: report.variance_seconds,
            }
        })
        .collect()
}

fn pointwise_rows_for_prior(
    refinement_level: usize,
    mesh: &MeshData,
    config: &SphereBranchObservableConvergenceConfig,
    model: SphereBranchObservableModel,
    prior_branch: HodgeBranchKind,
    pointwise_operator: &PointwiseOperator,
    report: &PriorVarianceReport,
) -> Result<Vec<SpherePointwiseVarianceRow>, String> {
    let point_count = fixed_pointwise_directions().len();
    if pointwise_operator.observations.len() != 3 * point_count {
        return Err(format!(
            "expected {} pointwise observations, got {}",
            3 * point_count,
            pointwise_operator.observations.len()
        ));
    }
    if report.pointwise_variances.len() != pointwise_operator.observations.len() {
        return Err(format!(
            "pointwise variance count {} does not match observation count {}",
            report.pointwise_variances.len(),
            pointwise_operator.observations.len()
        ));
    }

    let eval_points = pointwise_operator.observations[..point_count]
        .iter()
        .map(|observation| observation.eval_point)
        .collect::<Vec<_>>();
    let analytic = analytic_branch_covariance(
        &eval_points,
        prior_branch,
        config.kappa,
        config.tau,
        MaternAlpha::Two,
        config.pointwise_analytic_lmax,
    )?;

    pointwise_operator
        .observations
        .iter()
        .enumerate()
        .map(|(row_index, observation)| {
            let analytic_variance = analytic[(row_index, row_index)];
            if !analytic_variance.is_finite() || analytic_variance < -EPS {
                return Err(format!(
                    "non-finite or negative analytic pointwise variance at row {row_index}: {analytic_variance}"
                ));
            }
            let model_variance = report.pointwise_variances[row_index];
            let absolute_error = (model_variance - analytic_variance).abs();
            Ok(SpherePointwiseVarianceRow {
                refinement_level,
                h: mesh.metric.mesh_width_max(),
                vertex_count: mesh.topology.vertices().len(),
                edge_count: mesh.topology.edges().len(),
                cell_count: mesh.topology.cells().len(),
                model,
                prior_branch,
                point_index: observation.point_index,
                cell_index: observation.cell_index,
                component_index: observation.component_index,
                component_name: observation.component_name,
                target_x: observation.target[0],
                target_y: observation.target[1],
                target_z: observation.target[2],
                eval_x: observation.eval_point[0],
                eval_y: observation.eval_point[1],
                eval_z: observation.eval_point[2],
                direction_error: observation.direction_error,
                latent_dofs: report.latent_dofs,
                precision_nnz: report.precision_nnz,
                factor_nnz: report.factor_nnz,
                row_nnz: pointwise_operator.row_nnz[row_index],
                model_variance,
                analytic_variance,
                absolute_error,
                relative_error: absolute_error / analytic_variance.abs().max(EPS),
                build_seconds: report.build_seconds,
                factor_seconds: report.factor_seconds,
                variance_seconds: report.variance_seconds,
            })
        })
        .collect()
}

fn analytic_target_variance(
    prior_branch: HodgeBranchKind,
    observable: SphereObservable,
    kappa: f64,
    tau: f64,
) -> f64 {
    if prior_branch == observable.branch {
        analytic_matched_variance(observable.ell, kappa, tau)
    } else {
        0.0
    }
}

fn analytic_matched_variance(ell: usize, kappa: f64, tau: f64) -> f64 {
    tau.powi(-2) * (kappa * kappa + lambda_for_ell(ell)).powi(-(MaternAlpha::Two.as_u32() as i32))
}

fn lambda_for_ell(ell: usize) -> f64 {
    ell as f64 * (ell as f64 + 1.0)
}

fn vector_to_sparse_row(vector: &FeecVector, tolerance: f64) -> Vec<(usize, f64)> {
    vector
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            if value.abs() > tolerance {
                Some((index, *value))
            } else {
                None
            }
        })
        .collect()
}

fn summary_rows(rows: &[SphereObservableVarianceRow]) -> Vec<SphereObservableSummaryRow> {
    let mut grouped = BTreeMap::<
        (
            usize,
            SphereBranchObservableModel,
            HodgeBranchKind,
            HodgeBranchKind,
            &'static str,
        ),
        Vec<&SphereObservableVarianceRow>,
    >::new();
    for row in rows {
        grouped
            .entry((
                row.refinement_level,
                row.model,
                row.prior_branch,
                row.observable_branch,
                row.kind,
            ))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .map(
            |((refinement_level, model, prior_branch, observable_branch, kind), group)| {
                let variances = group
                    .iter()
                    .map(|row| row.model_variance)
                    .collect::<Vec<_>>();
                let scaled_errors = group.iter().map(|row| row.scaled_error).collect::<Vec<_>>();
                SphereObservableSummaryRow {
                    refinement_level,
                    h: group[0].h,
                    model,
                    prior_branch,
                    observable_branch,
                    kind,
                    count: group.len(),
                    mean_model_variance: mean(&variances),
                    median_model_variance: median(variances),
                    rms_scaled_error: rms(&scaled_errors),
                    median_scaled_error: median(scaled_errors.clone()),
                    max_scaled_error: scaled_errors
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                }
            },
        )
        .collect()
}

fn fit_summary_rows(rows: &[SphereObservableSummaryRow]) -> Vec<SphereObservableFitSummaryRow> {
    let mut grouped = BTreeMap::<
        (
            SphereBranchObservableModel,
            HodgeBranchKind,
            HodgeBranchKind,
            &'static str,
        ),
        Vec<&SphereObservableSummaryRow>,
    >::new();
    for row in rows {
        grouped
            .entry((row.model, row.prior_branch, row.observable_branch, row.kind))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for ((model, prior_branch, observable_branch, kind), mut group) in grouped {
        group.sort_by(|lhs, rhs| {
            lhs.h
                .partial_cmp(&rhs.h)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if group.len() < 2 {
            continue;
        }
        let hs = group.iter().map(|row| row.h).collect::<Vec<_>>();
        let rms = group
            .iter()
            .map(|row| row.rms_scaled_error)
            .collect::<Vec<_>>();
        let finest = group
            .iter()
            .min_by(|lhs, rhs| {
                lhs.h
                    .partial_cmp(&rhs.h)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|row| row.rms_scaled_error)
            .unwrap_or(f64::NAN);
        out.push(SphereObservableFitSummaryRow {
            model,
            prior_branch,
            observable_branch,
            kind,
            diagnostic: if kind == "matched" {
                "loglog_slope_matched_rms_relative_error_vs_h".to_string()
            } else {
                "loglog_slope_leakage_rms_ratio_vs_h".to_string()
            },
            value: loglog_slope(&hs, &rms),
            expected: if kind == "matched" {
                "positive under variance convergence".to_string()
            } else {
                "positive if leakage decays; finest value should be small".to_string()
            },
            status: if kind == "matched" {
                "variance_convergence_diagnostic".to_string()
            } else {
                "leakage_diagnostic".to_string()
            },
        });
        out.push(SphereObservableFitSummaryRow {
            model,
            prior_branch,
            observable_branch,
            kind,
            diagnostic: if kind == "matched" {
                "finest_rms_relative_error".to_string()
            } else {
                "finest_rms_leakage_ratio".to_string()
            },
            value: finest,
            expected: "small on the finest mesh".to_string(),
            status: if kind == "matched" {
                "variance_convergence_diagnostic".to_string()
            } else {
                "leakage_diagnostic".to_string()
            },
        });
    }
    out
}

fn pointwise_summary_rows(rows: &[SpherePointwiseVarianceRow]) -> Vec<SpherePointwiseSummaryRow> {
    let mut grouped = BTreeMap::<
        (
            usize,
            SphereBranchObservableModel,
            HodgeBranchKind,
            &'static str,
        ),
        Vec<&SpherePointwiseVarianceRow>,
    >::new();
    for row in rows {
        grouped
            .entry((
                row.refinement_level,
                row.model,
                row.prior_branch,
                row.component_name,
            ))
            .or_default()
            .push(row);
        grouped
            .entry((row.refinement_level, row.model, row.prior_branch, "all"))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .map(
            |((refinement_level, model, prior_branch, component_family), group)| {
                let variances = group
                    .iter()
                    .map(|row| row.model_variance)
                    .collect::<Vec<_>>();
                let relative_errors = group
                    .iter()
                    .map(|row| row.relative_error)
                    .collect::<Vec<_>>();
                SpherePointwiseSummaryRow {
                    refinement_level,
                    h: group[0].h,
                    model,
                    prior_branch,
                    component_family,
                    count: group.len(),
                    mean_model_variance: mean(&variances),
                    median_model_variance: median(variances),
                    rms_relative_error: rms(&relative_errors),
                    median_relative_error: median(relative_errors.clone()),
                    max_relative_error: relative_errors
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                }
            },
        )
        .collect()
}

fn pointwise_fit_summary_rows(
    rows: &[SpherePointwiseSummaryRow],
) -> Vec<SpherePointwiseFitSummaryRow> {
    let mut grouped = BTreeMap::<
        (SphereBranchObservableModel, HodgeBranchKind, &'static str),
        Vec<&SpherePointwiseSummaryRow>,
    >::new();
    for row in rows {
        grouped
            .entry((row.model, row.prior_branch, row.component_family))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for ((model, prior_branch, component_family), mut group) in grouped {
        group.sort_by(|lhs, rhs| {
            lhs.h
                .partial_cmp(&rhs.h)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if group.len() < 2 {
            continue;
        }
        let hs = group.iter().map(|row| row.h).collect::<Vec<_>>();
        let rms_errors = group
            .iter()
            .map(|row| row.rms_relative_error)
            .collect::<Vec<_>>();
        let finest = group
            .iter()
            .min_by(|lhs, rhs| {
                lhs.h
                    .partial_cmp(&rhs.h)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|row| row.rms_relative_error)
            .unwrap_or(f64::NAN);
        out.push(SpherePointwiseFitSummaryRow {
            model,
            prior_branch,
            component_family,
            diagnostic: "loglog_slope_pointwise_rms_relative_error_vs_h".to_string(),
            value: loglog_slope(&hs, &rms_errors),
            expected: "positive if local ambient branch variance converges; may be slower than smooth observables"
                .to_string(),
            status: "pointwise_local_field_diagnostic".to_string(),
        });
        out.push(SpherePointwiseFitSummaryRow {
            model,
            prior_branch,
            component_family,
            diagnostic: "finest_pointwise_rms_relative_error".to_string(),
            value: finest,
            expected:
                "small on the finest mesh, subject to local-field and analytic-truncation error"
                    .to_string(),
            status: "pointwise_local_field_diagnostic".to_string(),
        });
    }
    out
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
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

fn write_observable_variance_csv(
    path: &Path,
    rows: &[SphereObservableVarianceRow],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "refinement_level,h,vertex_count,edge_count,cell_count,model,prior_branch,observable_branch,kind,ell,m,observable_id,latent_dofs,precision_nnz,factor_nnz,row_nnz,model_variance,analytic_variance,target_scale,absolute_error,scaled_error,build_seconds,factor_seconds,variance_seconds"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.6},{:.6},{:.6}",
            row.refinement_level,
            row.h,
            row.vertex_count,
            row.edge_count,
            row.cell_count,
            row.model.as_str(),
            row.prior_branch.as_str(),
            row.observable_branch.as_str(),
            row.kind,
            row.ell,
            row.m,
            row.observable_id,
            row.latent_dofs,
            row.precision_nnz,
            row.factor_nnz,
            row.row_nnz,
            row.model_variance,
            row.analytic_variance,
            row.target_scale,
            row.absolute_error,
            row.scaled_error,
            row.build_seconds,
            row.factor_seconds,
            row.variance_seconds
        )?;
    }
    Ok(())
}

fn write_summary_csv(path: &Path, rows: &[SphereObservableSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "refinement_level,h,model,prior_branch,observable_branch,kind,count,mean_model_variance,median_model_variance,rms_scaled_error,median_scaled_error,max_scaled_error"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.refinement_level,
            row.h,
            row.model.as_str(),
            row.prior_branch.as_str(),
            row.observable_branch.as_str(),
            row.kind,
            row.count,
            row.mean_model_variance,
            row.median_model_variance,
            row.rms_scaled_error,
            row.median_scaled_error,
            row.max_scaled_error
        )?;
    }
    Ok(())
}

fn write_fit_summary_csv(path: &Path, rows: &[SphereObservableFitSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,prior_branch,observable_branch,kind,diagnostic,value,expected,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{:.17e},{},{}",
            row.model.as_str(),
            row.prior_branch.as_str(),
            row.observable_branch.as_str(),
            row.kind,
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

fn write_pointwise_variance_csv(
    path: &Path,
    rows: &[SpherePointwiseVarianceRow],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "refinement_level,h,vertex_count,edge_count,cell_count,model,prior_branch,point_index,cell_index,component_index,component_name,target_x,target_y,target_z,eval_x,eval_y,eval_z,direction_error,latent_dofs,precision_nnz,factor_nnz,row_nnz,model_variance,analytic_variance,absolute_error,relative_error,build_seconds,factor_seconds,variance_seconds"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.6},{:.6},{:.6}",
            row.refinement_level,
            row.h,
            row.vertex_count,
            row.edge_count,
            row.cell_count,
            row.model.as_str(),
            row.prior_branch.as_str(),
            row.point_index,
            row.cell_index,
            row.component_index,
            row.component_name,
            row.target_x,
            row.target_y,
            row.target_z,
            row.eval_x,
            row.eval_y,
            row.eval_z,
            row.direction_error,
            row.latent_dofs,
            row.precision_nnz,
            row.factor_nnz,
            row.row_nnz,
            row.model_variance,
            row.analytic_variance,
            row.absolute_error,
            row.relative_error,
            row.build_seconds,
            row.factor_seconds,
            row.variance_seconds
        )?;
    }
    Ok(())
}

fn write_pointwise_summary_csv(path: &Path, rows: &[SpherePointwiseSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "refinement_level,h,model,prior_branch,component_family,count,mean_model_variance,median_model_variance,rms_relative_error,median_relative_error,max_relative_error"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.refinement_level,
            row.h,
            row.model.as_str(),
            row.prior_branch.as_str(),
            row.component_family,
            row.count,
            row.mean_model_variance,
            row.median_model_variance,
            row.rms_relative_error,
            row.median_relative_error,
            row.max_relative_error
        )?;
    }
    Ok(())
}

fn write_pointwise_fit_summary_csv(
    path: &Path,
    rows: &[SpherePointwiseFitSummaryRow],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,prior_branch,component_family,diagnostic,value,expected,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17e},{},{}",
            row.model.as_str(),
            row.prior_branch.as_str(),
            row.component_family,
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

fn write_readme(path: &Path, config: &SphereBranchObservableConvergenceConfig) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# Sphere Exact/Coexact Observable Convergence")?;
    writeln!(writer)?;
    writeln!(writer, "- levels: {:?}", config.levels)?;
    writeln!(writer, "- kappa: {:.17}", config.kappa)?;
    writeln!(writer, "- tau: {:.17}", config.tau)?;
    writeln!(writer, "- alpha: 2")?;
    writeln!(writer, "- lmax: {}", config.lmax)?;
    writeln!(
        writer,
        "- pointwise_analytic_lmax: {}",
        config.pointwise_analytic_lmax
    )?;
    writeln!(
        writer,
        "- observables: normalized real vector spherical harmonics dY/sqrt(lambda) and star dY/sqrt(lambda)"
    )?;
    writeln!(
        writer,
        "- computation: transformed marginal variances by sparse Cholesky factorization plus RHS solves; no full transformed covariance"
    )?;
    writeln!(
        writer,
        "- timings: build_seconds includes branch-prior construction and internal validation; factor_seconds is the experiment factorization reused for variance RHS solves"
    )?;
    writeln!(
        writer,
        "- matched rows compare exact-on-exact and coexact-on-coexact variances to analytic alpha=2 branch variances"
    )?;
    writeln!(
        writer,
        "- leakage rows report exact observables under coexact priors and coexact observables under exact priors, scaled by the matched analytic variance"
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Output Files")?;
    writeln!(
        writer,
        "- observable_variance.csv, summary.csv, fit_summary.csv: thesis-primary smooth exact/coexact mode observables with analytic variance targets"
    )?;
    writeln!(
        writer,
        "- pointwise_variance.csv, pointwise_summary.csv, pointwise_fit_summary.csv: ambient x/y/z Whitney-reconstructed barycenter field-value diagnostics"
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Pointwise Diagnostic")?;
    writeln!(
        writer,
        "- requested points: the 8 normalized cube-corner directions (+/-1,+/-1,+/-1)/sqrt(3)"
    )?;
    writeln!(
        writer,
        "- evaluation points: nearest unique triangle barycenters on each mesh, normalized before analytic evaluation"
    )?;
    writeln!(
        writer,
        "- analytic pointwise targets: diagonal ambient-component variances from the matching exact or coexact vector-spherical-harmonic branch kernel"
    )?;
    writeln!(
        writer,
        "- interpretation: pointwise ambient values are a harder local-field diagnostic, not a pure exact/coexact observable family and not a replacement for the smooth-mode convergence result"
    )?;
    Ok(())
}

fn validate_config(config: &SphereBranchObservableConvergenceConfig) -> Result<(), String> {
    if config.levels.is_empty() {
        return Err("at least one sphere refinement level is required".to_string());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if config.lmax == 0 || config.lmax > 3 {
        return Err("lmax must be in 1..=3 for the built-in real harmonics".to_string());
    }
    if config.pointwise_analytic_lmax == 0 {
        return Err("pointwise_analytic_lmax must be positive".to_string());
    }
    Ok(())
}

fn real_spherical_harmonic_value_grad(ell: usize, m: isize, unit: [f64; 3]) -> (f64, [f64; 3]) {
    let [x, y, z] = unit;
    match (ell, m) {
        (1, -1) => {
            let c = (3.0 / FOUR_PI).sqrt();
            (c * y, [0.0, c, 0.0])
        }
        (1, 0) => {
            let c = (3.0 / FOUR_PI).sqrt();
            (c * z, [0.0, 0.0, c])
        }
        (1, 1) => {
            let c = (3.0 / FOUR_PI).sqrt();
            (c * x, [c, 0.0, 0.0])
        }
        (2, -2) => {
            let c = (15.0 / FOUR_PI).sqrt();
            (c * x * y, [c * y, c * x, 0.0])
        }
        (2, -1) => {
            let c = (15.0 / FOUR_PI).sqrt();
            (c * y * z, [0.0, c * z, c * y])
        }
        (2, 0) => {
            let c = (5.0 / (16.0 * std::f64::consts::PI)).sqrt();
            (
                c * (2.0 * z * z - x * x - y * y),
                [-2.0 * c * x, -2.0 * c * y, 4.0 * c * z],
            )
        }
        (2, 1) => {
            let c = (15.0 / FOUR_PI).sqrt();
            (c * x * z, [c * z, 0.0, c * x])
        }
        (2, 2) => {
            let c = (15.0 / (16.0 * std::f64::consts::PI)).sqrt();
            (c * (x * x - y * y), [2.0 * c * x, -2.0 * c * y, 0.0])
        }
        (3, -3) => {
            let c = 0.25 * (35.0 / (2.0 * std::f64::consts::PI)).sqrt();
            (
                c * y * (3.0 * x * x - y * y),
                [6.0 * c * x * y, c * (3.0 * x * x - 3.0 * y * y), 0.0],
            )
        }
        (3, -2) => {
            let c = 0.5 * (105.0 / std::f64::consts::PI).sqrt();
            (c * x * y * z, [c * y * z, c * x * z, c * x * y])
        }
        (3, -1) => {
            let c = 0.25 * (21.0 / (2.0 * std::f64::consts::PI)).sqrt();
            let q = 4.0 * z * z - x * x - y * y;
            (
                c * y * q,
                [
                    -2.0 * c * x * y,
                    c * (4.0 * z * z - x * x - 3.0 * y * y),
                    8.0 * c * y * z,
                ],
            )
        }
        (3, 0) => {
            let c = 0.25 * (7.0 / std::f64::consts::PI).sqrt();
            (
                c * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
                [
                    -6.0 * c * x * z,
                    -6.0 * c * y * z,
                    c * (6.0 * z * z - 3.0 * x * x - 3.0 * y * y),
                ],
            )
        }
        (3, 1) => {
            let c = 0.25 * (21.0 / (2.0 * std::f64::consts::PI)).sqrt();
            let q = 4.0 * z * z - x * x - y * y;
            (
                c * x * q,
                [
                    c * (4.0 * z * z - 3.0 * x * x - y * y),
                    -2.0 * c * x * y,
                    8.0 * c * x * z,
                ],
            )
        }
        (3, 2) => {
            let c = 0.25 * (105.0 / std::f64::consts::PI).sqrt();
            (
                c * z * (x * x - y * y),
                [2.0 * c * x * z, -2.0 * c * y * z, c * (x * x - y * y)],
            )
        }
        (3, 3) => {
            let c = 0.25 * (35.0 / (2.0 * std::f64::consts::PI)).sqrt();
            (
                c * x * (x * x - 3.0 * y * y),
                [c * (3.0 * x * x - 3.0 * y * y), -6.0 * c * x * y, 0.0],
            )
        }
        _ => panic!("unsupported real spherical harmonic ell={ell}, m={m}"),
    }
}

fn normalize3(point: [f64; 3]) -> [f64; 3] {
    let norm = dot3(point, point).sqrt();
    if norm <= EPS {
        return [1.0, 0.0, 0.0];
    }
    [point[0] / norm, point[1] / norm, point[2] / norm]
}

fn dot3(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn cross3(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn sub3(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

fn scale3(value: [f64; 3], scale: f64) -> [f64; 3] {
    [scale * value[0], scale * value[1], scale * value[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeSet,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn exact_and_coexact_vector_proxies_are_tangent() {
        let points = [
            normalize3([0.2, -0.4, 0.8]),
            normalize3([-0.7, 0.1, 0.3]),
            normalize3([0.5, 0.6, -0.2]),
        ];
        for observable in observables(3) {
            for point in points {
                let proxy = observable.vector_proxy(point);
                assert!(
                    dot3(point, proxy).abs() <= 1e-12,
                    "{} proxy should be tangent",
                    observable.id()
                );
                assert!(proxy.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn vector_proxy_modes_have_unit_l2_norm() {
        for observable in observables(3) {
            let integral = fibonacci_sphere_points(4096)
                .into_iter()
                .map(|point| {
                    dot3(
                        observable.vector_proxy(point),
                        observable.vector_proxy(point),
                    )
                })
                .sum::<f64>()
                * FOUR_PI
                / 4096.0;
            assert!(
                (integral - 1.0).abs() <= 8e-3,
                "{} approximate L2 norm should be one, got {integral:.6e}",
                observable.id()
            );
        }
    }

    #[test]
    fn analytic_matched_variance_uses_alpha_two_eigenvalue() {
        let kappa: f64 = 1.5;
        let tau: f64 = 2.0;
        for ell in 1..=3 {
            let lambda = lambda_for_ell(ell);
            let expected = tau.powi(-2) * (kappa * kappa + lambda).powi(-2);
            assert!((analytic_matched_variance(ell, kappa, tau) - expected).abs() <= 1e-14);
        }
    }

    #[test]
    fn analytic_cross_branch_target_is_zero() {
        let exact = SphereObservable {
            branch: HodgeBranchKind::Exact,
            ell: 2,
            m: 1,
        };
        let coexact = SphereObservable {
            branch: HodgeBranchKind::Coexact,
            ell: 2,
            m: 1,
        };
        assert_eq!(
            analytic_target_variance(HodgeBranchKind::Coexact, exact, 1.0, 1.0),
            0.0
        );
        assert_eq!(
            analytic_target_variance(HodgeBranchKind::Exact, coexact, 1.0, 1.0),
            0.0
        );
    }

    #[test]
    fn fixed_pointwise_directions_are_unit_vectors() {
        let directions = fixed_pointwise_directions();
        assert_eq!(directions.len(), 8);
        for direction in directions {
            assert!(direction.iter().all(|value| value.is_finite()));
            assert!((dot3(direction, direction) - 1.0).abs() <= 1e-14);
        }
    }

    #[test]
    fn pointwise_operator_selects_unique_cells_and_has_expected_rows() {
        let mesh = build_mesh(1);
        let operator = pointwise_operator(&mesh).expect("pointwise operator should build");
        let point_count = fixed_pointwise_directions().len();
        assert_eq!(operator.operator.output_dimension(), 3 * point_count);
        assert_eq!(
            operator.operator.input_dimension(),
            mesh.topology.edges().len()
        );
        assert_eq!(operator.row_nnz.len(), 3 * point_count);
        assert_eq!(operator.observations.len(), 3 * point_count);
        assert!(operator
            .operator
            .matrix()
            .triplet_iter()
            .all(|(_, _, value)| value.is_finite()));

        let unique_cells = operator
            .observations
            .iter()
            .filter(|observation| observation.component_index == 0)
            .map(|observation| observation.cell_index)
            .collect::<BTreeSet<_>>();
        assert_eq!(unique_cells.len(), point_count);
    }

    #[test]
    fn analytic_pointwise_branch_variances_are_finite_nonnegative() {
        let mesh = build_mesh(1);
        let operator = pointwise_operator(&mesh).expect("pointwise operator should build");
        let point_count = fixed_pointwise_directions().len();
        let eval_points = operator.observations[..point_count]
            .iter()
            .map(|observation| observation.eval_point)
            .collect::<Vec<_>>();
        for branch in REPORT_BRANCHES {
            let covariance =
                analytic_branch_covariance(&eval_points, branch, 1.0, 1.0, MaternAlpha::Two, 40)
                    .expect("analytic pointwise covariance should build");
            for index in 0..covariance.nrows() {
                let value = covariance[(index, index)];
                assert!(
                    value.is_finite() && value >= 0.0,
                    "{} diagonal entry {index} should be finite and nonnegative, got {value:.6e}",
                    branch.as_str()
                );
            }
        }
    }

    #[test]
    fn tiny_sphere_branch_observable_convergence_runs() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        let output_dir =
            std::env::temp_dir().join(format!("sphere_branch_observable_convergence_{stamp}"));
        let config = SphereBranchObservableConvergenceConfig {
            levels: vec![1],
            output_dir: output_dir.clone(),
            lmax: 1,
            ..SphereBranchObservableConvergenceConfig::default()
        };
        let result = run_sphere_branch_observable_convergence(&config)
            .expect("tiny sphere observable convergence run should complete");
        assert_eq!(result.variance_rows.len(), 2 * 2 * 2 * 3);
        assert_eq!(result.pointwise_rows.len(), 2 * 2 * 8 * 3);
        assert!(result.variance_rows.iter().all(|row| {
            row.model_variance.is_finite()
                && row.model_variance >= 0.0
                && row.analytic_variance.is_finite()
                && row.scaled_error.is_finite()
                && row.latent_dofs > 0
                && row.precision_nnz > 0
                && row.factor_nnz > 0
        }));
        assert!(result.pointwise_rows.iter().all(|row| {
            row.model_variance.is_finite()
                && row.model_variance >= 0.0
                && row.analytic_variance.is_finite()
                && row.analytic_variance >= 0.0
                && row.relative_error.is_finite()
                && row.direction_error.is_finite()
                && row.latent_dofs > 0
                && row.precision_nnz > 0
                && row.factor_nnz > 0
        }));
        for model in REPORT_MODELS {
            for prior_branch in REPORT_BRANCHES {
                for observable_branch in REPORT_BRANCHES {
                    assert!(result.variance_rows.iter().any(|row| {
                        row.model == model
                            && row.prior_branch == prior_branch
                            && row.observable_branch == observable_branch
                    }));
                }
            }
        }
        assert!(output_dir.join("observable_variance.csv").exists());
        assert!(output_dir.join("summary.csv").exists());
        assert!(output_dir.join("fit_summary.csv").exists());
        assert!(output_dir.join("pointwise_variance.csv").exists());
        assert!(output_dir.join("pointwise_summary.csv").exists());
        assert!(output_dir.join("pointwise_fit_summary.csv").exists());
        assert!(output_dir.join("README.md").exists());
        let _ = fs::remove_dir_all(output_dir);
    }

    fn fibonacci_sphere_points(count: usize) -> Vec<[f64; 3]> {
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..count)
            .map(|index| {
                let z = 1.0 - 2.0 * (index as f64 + 0.5) / count as f64;
                let radius = (1.0 - z * z).sqrt();
                let theta = golden * index as f64;
                [radius * theta.cos(), radius * theta.sin(), z]
            })
            .collect()
    }
}
