//! Spectral-reference validation for 1-form Matern covariances on the sphere.
//!
//! This experiment compares full row-sum/NC1 lumped and Whitney projected sparse
//! inverse alpha=2 priors against the analytic vector-spherical-harmonic
//! covariance on the unit sphere.  Unlike the cube NC1 diagnostic, the reference
//! is not the finest computed mesh.

use crate::{
    nc1_matern_convergence::build_full_nc1_matern_precision_alpha_two,
    sphere_sparse_anchor_kernel_validation::analytic_joint_one_form_covariance,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::{exterior_dim, ExteriorElement};
use feg_infer::{
    prior::matern::{
        one_form::{
            build_hodge_laplacian_1form, build_matern_precision_1form_for_alpha,
            build_reconstructed_barycenter_field_operator, MaternConfig as WhitneyMaternConfig,
            MaternMassInverse,
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
    dim3::mesh_sphere_surface,
    geometry::{
        coord::{
            mesh::MeshCoords,
            simplex::{barycenter_local, SimplexHandleExt},
            CoordRef,
        },
        metric::mesh::MeshLengths,
    },
    topology::{complex::Complex, simplex::standard_subsimps},
};
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const DEFAULT_REFINEMENT_LEVELS: [usize; 4] = [1, 2, 3, 4];
const DEFAULT_ANALYTIC_LMAX: usize = 35;
const DEFAULT_MAX_CELLS: usize = 32;
const RECONSTRUCTION_TOLERANCE: f64 = 1e-14;
const EPS: f64 = 1e-14;

#[derive(Debug, Clone)]
pub struct SphereNc1MaternSpectralReferenceConfig {
    pub refinement_levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub analytic_lmax: usize,
    pub max_cells: usize,
}

impl Default for SphereNc1MaternSpectralReferenceConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            refinement_levels: DEFAULT_REFINEMENT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/sphere_nc1_matern_spectral_reference"),
            kappa: 1.0,
            tau: 1.0,
            analytic_lmax: DEFAULT_ANALYTIC_LMAX,
            max_cells: DEFAULT_MAX_CELLS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SphereNc1MaternModel {
    FullNc1Lumped,
    WhitneyProjected,
}

impl SphereNc1MaternModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullNc1Lumped => "full_nc1_row_sum_vertex_lumped",
            Self::WhitneyProjected => "whitney_projected_sparse_inverse",
        }
    }

    pub fn status(self) -> &'static str {
        match self {
            Self::FullNc1Lumped => "analytic_reference",
            Self::WhitneyProjected => "comparison_only",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SphereNc1MaternSpectralReferenceRow {
    pub refinement_level: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub cell_count: usize,
    pub selected_cell_count: usize,
    pub covariance_dimension: usize,
    pub model: SphereNc1MaternModel,
    pub state_dofs: usize,
    pub precision_nnz: usize,
    pub factor_nnz: usize,
    pub relative_frobenius_error: f64,
    pub diagonal_relative_l2_error: f64,
    pub best_scalar_rescaled_relative_frobenius_error: f64,
    pub best_scalar: f64,
    pub model_trace: f64,
    pub analytic_trace: f64,
    pub same_level_relative_gap_vs_full_nc1: f64,
    pub build_seconds: f64,
    pub covariance_seconds: f64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct SphereNc1MaternSpectralReferenceResult {
    pub rows: Vec<SphereNc1MaternSpectralReferenceRow>,
}

struct ModelComputation {
    model: SphereNc1MaternModel,
    state_dofs: usize,
    precision_nnz: usize,
    factor_nnz: usize,
    covariance: GmrfDenseMatrix,
    build_seconds: f64,
    covariance_seconds: f64,
}

struct CovarianceMetrics {
    relative_frobenius_error: f64,
    diagonal_relative_l2_error: f64,
    best_scalar_rescaled_relative_frobenius_error: f64,
    best_scalar: f64,
    model_trace: f64,
    analytic_trace: f64,
}

pub fn run_sphere_nc1_matern_spectral_reference(
    config: &SphereNc1MaternSpectralReferenceConfig,
) -> Result<SphereNc1MaternSpectralReferenceResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut rows = Vec::new();
    for &refinement_level in &config.refinement_levels {
        eprintln!("[sphere_nc1_matern_spectral_reference] level={refinement_level}");
        rows.extend(compute_level_rows(refinement_level, config)?);
    }

    write_summary_csv(&config.output_dir.join("summary.csv"), &rows)?;
    write_readme(&config.output_dir.join("README.md"), config)?;
    Ok(SphereNc1MaternSpectralReferenceResult { rows })
}

fn compute_level_rows(
    refinement_level: usize,
    config: &SphereNc1MaternSpectralReferenceConfig,
) -> Result<Vec<SphereNc1MaternSpectralReferenceRow>, String> {
    let surface = mesh_sphere_surface(refinement_level);
    let (topology, coords) = surface.into_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let selected_cells = evenly_spaced_indices(topology.cells().len(), config.max_cells);
    let points = normalized_cell_barycenters(&topology, &coords, &selected_cells)?;
    let analytic = analytic_joint_one_form_covariance(
        &points,
        config.kappa,
        config.tau,
        MaternAlpha::Two,
        config.analytic_lmax,
    )?;

    let full = compute_model(
        SphereNc1MaternModel::FullNc1Lumped,
        &topology,
        &coords,
        &metric,
        &selected_cells,
        config,
    )?;
    let whitney = compute_model(
        SphereNc1MaternModel::WhitneyProjected,
        &topology,
        &coords,
        &metric,
        &selected_cells,
        config,
    )?;
    let whitney_gap = relative_frobenius_difference(&whitney.covariance, &full.covariance)?;

    let mut rows = Vec::with_capacity(2);
    rows.push(build_row(
        refinement_level,
        &topology,
        selected_cells.len(),
        &analytic,
        &full,
        0.0,
    )?);
    rows.push(build_row(
        refinement_level,
        &topology,
        selected_cells.len(),
        &analytic,
        &whitney,
        whitney_gap,
    )?);
    Ok(rows)
}

fn compute_model(
    model: SphereNc1MaternModel,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    selected_cells: &[usize],
    config: &SphereNc1MaternSpectralReferenceConfig,
) -> Result<ModelComputation, String> {
    let build_start = Instant::now();
    let (precision, operator) = match model {
        SphereNc1MaternModel::FullNc1Lumped => {
            let precision = build_full_nc1_matern_precision_alpha_two(
                topology,
                metric,
                config.kappa,
                config.tau,
            )?;
            let operator =
                stacked_nc1_barycenter_reconstruction_operator(topology, coords, selected_cells)?;
            (precision, operator)
        }
        SphereNc1MaternModel::WhitneyProjected => {
            let hodge = build_hodge_laplacian_1form(topology, metric);
            let precision = build_matern_precision_1form_for_alpha(
                topology,
                metric,
                &hodge,
                MaternAlpha::Two,
                WhitneyMaternConfig {
                    kappa: config.kappa,
                    tau: config.tau,
                    mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
                },
            );
            let operator = stacked_whitney_barycenter_reconstruction_operator(
                topology,
                coords,
                selected_cells,
            )?;
            (precision, operator)
        }
    };
    let build_seconds = build_start.elapsed().as_secs_f64();

    let precision_nnz = precision.nnz();
    let state_dofs = precision.nrows();
    let covariance_start = Instant::now();
    let (covariance, factor_nnz) = exact_transformed_covariance(&precision, &operator)?;
    let covariance_seconds = covariance_start.elapsed().as_secs_f64();

    Ok(ModelComputation {
        model,
        state_dofs,
        precision_nnz,
        factor_nnz,
        covariance,
        build_seconds,
        covariance_seconds,
    })
}

fn build_row(
    refinement_level: usize,
    topology: &Complex,
    selected_cell_count: usize,
    analytic: &GmrfDenseMatrix,
    computation: &ModelComputation,
    same_level_relative_gap_vs_full_nc1: f64,
) -> Result<SphereNc1MaternSpectralReferenceRow, String> {
    let metrics = compare_to_analytic(&computation.covariance, analytic)?;
    Ok(SphereNc1MaternSpectralReferenceRow {
        refinement_level,
        vertex_count: topology.vertices().len(),
        edge_count: topology.edges().len(),
        cell_count: topology.cells().len(),
        selected_cell_count,
        covariance_dimension: computation.covariance.nrows(),
        model: computation.model,
        state_dofs: computation.state_dofs,
        precision_nnz: computation.precision_nnz,
        factor_nnz: computation.factor_nnz,
        relative_frobenius_error: metrics.relative_frobenius_error,
        diagonal_relative_l2_error: metrics.diagonal_relative_l2_error,
        best_scalar_rescaled_relative_frobenius_error: metrics
            .best_scalar_rescaled_relative_frobenius_error,
        best_scalar: metrics.best_scalar,
        model_trace: metrics.model_trace,
        analytic_trace: metrics.analytic_trace,
        same_level_relative_gap_vs_full_nc1,
        build_seconds: computation.build_seconds,
        covariance_seconds: computation.covariance_seconds,
        status: computation.model.status().to_string(),
    })
}

fn stacked_whitney_barycenter_reconstruction_operator(
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

fn stacked_nc1_barycenter_reconstruction_operator(
    topology: &Complex,
    coords: &MeshCoords,
    selected_cells: &[usize],
) -> Result<SparseRowOperator, String> {
    let topo_dim = topology.dim();
    if topo_dim < 1 {
        return Err("topology dimension must be at least 1".to_string());
    }
    let ambient_dim = coords.dim();
    let bary_local = barycenter_local(topo_dim);
    let cells = topology.cells().handle_iter().collect::<Vec<_>>();
    let mut component_rows = vec![Vec::with_capacity(selected_cells.len()); ambient_dim];

    for &cell_index in selected_cells {
        let cell = cells
            .get(cell_index)
            .ok_or_else(|| format!("selected cell {cell_index} is out of bounds"))?;
        let cell_coords = cell.coord_simplex(coords);
        let local_edges = cell.mesh_subsimps(1).collect::<Vec<_>>();
        let basis = nc1_local_basis_coeffs(topo_dim, bary_local.as_view());
        let mut rows_for_cell = vec![Vec::new(); ambient_dim];

        for (local_edge_index, edge) in local_edges.iter().enumerate() {
            for slot in 0..2 {
                let local_col = 2 * local_edge_index + slot;
                let local_form =
                    ExteriorElement::new(basis.column(local_col).into_owned(), topo_dim, 1);
                let ambient_value = cell_coords.lift_form(&local_form).into_grade1();
                let global_col = nc1_global_dof(edge.kidx(), slot);
                for component_index in 0..ambient_dim {
                    let coefficient = ambient_value[component_index];
                    if coefficient.abs() > RECONSTRUCTION_TOLERANCE {
                        rows_for_cell[component_index].push((global_col, coefficient));
                    }
                }
            }
        }

        for component_index in 0..ambient_dim {
            component_rows[component_index]
                .push(std::mem::take(&mut rows_for_cell[component_index]));
        }
    }

    let rows = component_rows.into_iter().flatten().collect::<Vec<_>>();
    SparseRowOperator::new(nc1_dof_count(topology), rows).map_err(|err| err.to_string())
}

fn nc1_local_basis_coeffs(dim: usize, coord: CoordRef) -> FeecMatrix {
    let mut barys = Vec::with_capacity(dim + 1);
    barys.push(1.0 - coord.iter().sum::<f64>());
    barys.extend(coord.iter().copied());

    let local_edges = standard_subsimps(dim, 1).collect::<Vec<_>>();
    let mut coeffs = FeecMatrix::zeros(exterior_dim(dim, 1), 2 * local_edges.len());
    for (edge_index, edge) in local_edges.into_iter().enumerate() {
        let lsf = WhitneyLsf::standard(dim, edge.clone());
        for slot in 0..2 {
            let sign = if slot == 0 { 1.0 } else { -1.0 };
            let vertex = edge[slot];
            let column = lsf.wedge_term(slot).into_coeffs() * (sign * barys[vertex]);
            coeffs.set_column(2 * edge_index + slot, &column);
        }
    }
    coeffs
}

fn nc1_dof_count(topology: &Complex) -> usize {
    2 * topology.nsimplices(1)
}

fn nc1_global_dof(edge_kidx: usize, slot: usize) -> usize {
    2 * edge_kidx + slot
}

fn exact_transformed_covariance(
    precision: &FeecCsr,
    operator: &SparseRowOperator,
) -> Result<(GmrfDenseMatrix, usize), String> {
    let gmrf_precision = feec_csr_to_gmrf(precision);
    let factor = gmrf_precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor precision: {err}"))?;
    let factor_nnz = factor.nnz();
    let mut gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), gmrf_precision)
            .map_err(|err| format!("failed to build GMRF: {err}"))?
            .with_precision_sqrt(factor);
    let covariance = gmrf
        .exact_transformed_covariance(operator)
        .map_err(|err| format!("failed to compute transformed covariance: {err}"))?;
    Ok((covariance, factor_nnz))
}

fn compare_to_analytic(
    model: &GmrfDenseMatrix,
    analytic: &GmrfDenseMatrix,
) -> Result<CovarianceMetrics, String> {
    ensure_same_shape(model, analytic)?;
    let diff = dense_difference(model, analytic);
    let analytic_norm = frobenius_norm(analytic).max(EPS);
    let best_scalar = best_scalar_fit(model, analytic);
    let scaled_diff = dense_scaled_difference(model, best_scalar, analytic);
    let (model_trace, analytic_trace, diag_diff_norm, analytic_diag_norm) =
        diagonal_stats(model, analytic);
    Ok(CovarianceMetrics {
        relative_frobenius_error: frobenius_norm(&diff) / analytic_norm,
        diagonal_relative_l2_error: diag_diff_norm / analytic_diag_norm.max(EPS),
        best_scalar_rescaled_relative_frobenius_error: frobenius_norm(&scaled_diff) / analytic_norm,
        best_scalar,
        model_trace,
        analytic_trace,
    })
}

fn relative_frobenius_difference(
    lhs: &GmrfDenseMatrix,
    rhs: &GmrfDenseMatrix,
) -> Result<f64, String> {
    ensure_same_shape(lhs, rhs)?;
    Ok(frobenius_norm(&dense_difference(lhs, rhs)) / frobenius_norm(rhs).max(EPS))
}

fn ensure_same_shape(lhs: &GmrfDenseMatrix, rhs: &GmrfDenseMatrix) -> Result<(), String> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err(format!(
            "covariance shape mismatch: lhs {}x{}, rhs {}x{}",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    Ok(())
}

fn dense_difference(lhs: &GmrfDenseMatrix, rhs: &GmrfDenseMatrix) -> GmrfDenseMatrix {
    GmrfDenseMatrix::from_fn(lhs.nrows(), lhs.ncols(), |i, j| lhs[(i, j)] - rhs[(i, j)])
}

fn dense_scaled_difference(
    lhs: &GmrfDenseMatrix,
    lhs_scale: f64,
    rhs: &GmrfDenseMatrix,
) -> GmrfDenseMatrix {
    GmrfDenseMatrix::from_fn(lhs.nrows(), lhs.ncols(), |i, j| {
        lhs_scale * lhs[(i, j)] - rhs[(i, j)]
    })
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

fn diagonal_stats(model: &GmrfDenseMatrix, analytic: &GmrfDenseMatrix) -> (f64, f64, f64, f64) {
    let mut model_trace = 0.0;
    let mut analytic_trace = 0.0;
    let mut diff_norm2 = 0.0;
    let mut analytic_norm2 = 0.0;
    for index in 0..model.nrows().min(model.ncols()) {
        let model_value = model[(index, index)];
        let analytic_value = analytic[(index, index)];
        model_trace += model_value;
        analytic_trace += analytic_value;
        diff_norm2 += (model_value - analytic_value).powi(2);
        analytic_norm2 += analytic_value.powi(2);
    }
    (
        model_trace,
        analytic_trace,
        diff_norm2.sqrt(),
        analytic_norm2.sqrt(),
    )
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

fn normalize3(point: [f64; 3]) -> Result<[f64; 3], String> {
    let norm = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    if norm <= EPS {
        return Err("cannot normalize zero barycenter".to_string());
    }
    Ok([point[0] / norm, point[1] / norm, point[2] / norm])
}

fn evenly_spaced_indices(len: usize, max_count: usize) -> Vec<usize> {
    if len <= max_count {
        return (0..len).collect();
    }
    (0..max_count).map(|i| i * len / max_count).collect()
}

fn validate_config(config: &SphereNc1MaternSpectralReferenceConfig) -> Result<(), String> {
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

pub fn write_summary_csv(
    path: &Path,
    rows: &[SphereNc1MaternSpectralReferenceRow],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "refinement_level,vertex_count,edge_count,cell_count,selected_cell_count,covariance_dimension,model,state_dofs,precision_nnz,factor_nnz,relative_frobenius_error,diagonal_relative_l2_error,best_scalar_rescaled_relative_frobenius_error,best_scalar,model_trace,analytic_trace,same_level_relative_gap_vs_full_nc1,build_seconds,covariance_seconds,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            row.refinement_level,
            row.vertex_count,
            row.edge_count,
            row.cell_count,
            row.selected_cell_count,
            row.covariance_dimension,
            row.model.as_str(),
            row.state_dofs,
            row.precision_nnz,
            row.factor_nnz,
            row.relative_frobenius_error,
            row.diagonal_relative_l2_error,
            row.best_scalar_rescaled_relative_frobenius_error,
            row.best_scalar,
            row.model_trace,
            row.analytic_trace,
            row.same_level_relative_gap_vs_full_nc1,
            row.build_seconds,
            row.covariance_seconds,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_readme(
    path: &Path,
    config: &SphereNc1MaternSpectralReferenceConfig,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# Sphere NC1 1-form Matern spectral reference")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "- refinement levels: {:?}",
        config.refinement_levels
    )?;
    writeln!(writer, "- kappa: {:.17}", config.kappa)?;
    writeln!(writer, "- tau: {:.17}", config.tau)?;
    writeln!(writer, "- alpha: 2")?;
    writeln!(writer, "- analytic lmax: {}", config.analytic_lmax)?;
    writeln!(
        writer,
        "- max selected cells per level: {}",
        config.max_cells
    )?;
    writeln!(
        writer,
        "- reference: exact plus coexact vector-spherical-harmonic covariance on the unit sphere"
    )?;
    writeln!(
        writer,
        "- models: full NC1 row-sum/vertex-lumped prior and Whitney projected sparse inverse prior"
    )?;
    writeln!(
        writer,
        "- observables: ambient tangent vector components reconstructed at selected triangle barycenters"
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn sphere_nc1_matern_spectral_reference_smoke_runs() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        let output_dir =
            std::env::temp_dir().join(format!("sphere_nc1_matern_spectral_reference_{stamp}"));
        let config = SphereNc1MaternSpectralReferenceConfig {
            refinement_levels: vec![0],
            output_dir: output_dir.clone(),
            kappa: 1.0,
            tau: 1.0,
            analytic_lmax: 8,
            max_cells: 4,
        };
        let result = run_sphere_nc1_matern_spectral_reference(&config)
            .expect("small sphere spectral-reference experiment should run");
        assert_eq!(result.rows.len(), 2);
        assert!(result.rows.iter().all(|row| {
            row.relative_frobenius_error.is_finite()
                && row.diagonal_relative_l2_error.is_finite()
                && row
                    .best_scalar_rescaled_relative_frobenius_error
                    .is_finite()
                && row.same_level_relative_gap_vs_full_nc1.is_finite()
        }));
        assert!(output_dir.join("summary.csv").exists());
        let _ = fs::remove_dir_all(output_dir);
    }
}
