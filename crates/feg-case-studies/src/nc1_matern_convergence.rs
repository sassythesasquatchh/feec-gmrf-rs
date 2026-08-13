//! Covariance convergence experiment for the proof-compliant NC1 1-form Matern prior.
//!
//! The experiment compares two alpha=2 priors on uniform tetrahedral cube meshes:
//!
//! - a full first-order `P1 Lambda^1`/NC1 state with row-sum P1 scalar codifferential
//!   lumping and vertex-lumped NC1 load covariance;
//! - the practical Whitney 1-form construction using the projected NC1 sparse inverse.
//!
//! The reported quantities are covariance matrices of fixed smooth L2 observables.
//! Since the priors are centred Gaussian, convergence of these finite covariance
//! matrices is the numerical counterpart of the cylindrical convergence statement.

use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use ddf::{whitney::lsf::WhitneyLsf, CoordSimplexExt};
use exterior::{
    exterior_dim,
    field::{DiffFormClosure, ExteriorField},
    term::multi_gramian,
};
use feg_infer::{
    prior::matern::{
        build_lindgren_precision_from_system,
        one_form::{
            build_hodge_laplacian_1form, build_matern_precision_1form_for_alpha,
            MaternConfig as WhitneyMaternConfig, MaternMassInverse,
        },
        MaternAlpha,
    },
    sparse::{add_sparse, diag_matrix, feec_csr_to_gmrf, invert_diag, lumped_diag, scale_matrix},
};
use formoniq::{
    assemble::{
        assemble_galmat, assemble_galvec, assemble_nc1_lumped_mass_inverse_galmat,
        assemble_nc1_mass_galmat,
    },
    operators::{HodgeMassElmat, SourceElVec},
};
use gmrf_core::{
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf, SparseRowOperator,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::{
        coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexCoords, CoordRef},
        metric::{mesh::MeshLengths, simplex::SimplexLengths},
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

const DEFAULT_LEVELS: [usize; 5] = [2, 4, 6, 8, 10];
const SPARSE_ROW_TOLERANCE: f64 = 1e-14;
const EPS: f64 = 1e-14;

#[derive(Debug, Clone)]
pub struct Nc1MaternConvergenceConfig {
    pub levels: Vec<usize>,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub tau: f64,
}

impl Default for Nc1MaternConvergenceConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            levels: DEFAULT_LEVELS.to_vec(),
            output_dir: manifest_dir.join("../../out/nc1_matern_convergence"),
            kappa: 4.0,
            tau: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OneFormConvergenceModel {
    FullNc1Lumped,
    WhitneyProjected,
}

impl OneFormConvergenceModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullNc1Lumped => "full_nc1_row_sum_vertex_lumped",
            Self::WhitneyProjected => "whitney_projected_sparse_inverse",
        }
    }

    pub fn status(self) -> &'static str {
        match self {
            Self::FullNc1Lumped => "proof_compliant",
            Self::WhitneyProjected => "comparison_only",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SmoothOneFormObservable {
    ConstantX,
    ConstantY,
    ConstantZ,
    LinearGradient,
    QuadraticMix,
    CurlLike,
}

impl SmoothOneFormObservable {
    pub fn all() -> &'static [Self] {
        &[
            Self::ConstantX,
            Self::ConstantY,
            Self::ConstantZ,
            Self::LinearGradient,
            Self::QuadraticMix,
            Self::CurlLike,
        ]
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::ConstantX => "constant_x",
            Self::ConstantY => "constant_y",
            Self::ConstantZ => "constant_z",
            Self::LinearGradient => "linear_gradient",
            Self::QuadraticMix => "quadratic_mix",
            Self::CurlLike => "curl_like",
        }
    }

    fn source(self) -> DiffFormClosure {
        DiffFormClosure::one_form(
            move |p| match self {
                Self::ConstantX => FeecVector::from_vec(vec![1.0, 0.0, 0.0]),
                Self::ConstantY => FeecVector::from_vec(vec![0.0, 1.0, 0.0]),
                Self::ConstantZ => FeecVector::from_vec(vec![0.0, 0.0, 1.0]),
                Self::LinearGradient => FeecVector::from_vec(vec![
                    1.0 + 2.0 * p[0],
                    -0.25 + 2.0 * p[1],
                    0.5 + 2.0 * p[2],
                ]),
                Self::QuadraticMix => FeecVector::from_vec(vec![
                    1.0 + p[0] + p[1] * p[2],
                    -0.5 + 0.25 * p[0] + p[1] * p[1],
                    0.75 + p[2] + p[0] * p[1],
                ]),
                Self::CurlLike => FeecVector::from_vec(vec![p[1] - p[2], p[2] - p[0], p[0] - p[1]]),
            },
            3,
        )
    }
}

#[derive(Debug, Clone)]
pub struct Nc1MaternSummaryRow {
    pub n: usize,
    pub h: f64,
    pub model: OneFormConvergenceModel,
    pub state_dofs: usize,
    pub precision_nnz: usize,
    pub factor_nnz: usize,
    pub covariance_frobenius_norm: f64,
    pub reference_relative_frobenius_error: f64,
    pub same_level_relative_gap_vs_full_nc1: f64,
    pub build_seconds: f64,
    pub covariance_seconds: f64,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct Nc1MaternCovarianceEntryRow {
    pub n: usize,
    pub h: f64,
    pub model: OneFormConvergenceModel,
    pub observable_row: String,
    pub observable_col: String,
    pub covariance: f64,
    pub reference_covariance: f64,
    pub same_level_full_nc1_covariance: f64,
    pub absolute_reference_error: f64,
    pub relative_reference_error: f64,
}

#[derive(Debug, Clone)]
pub struct Nc1MaternFitSummaryRow {
    pub model: OneFormConvergenceModel,
    pub diagnostic: String,
    pub value: f64,
    pub expected: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct Nc1MaternConvergenceResult {
    pub summary_rows: Vec<Nc1MaternSummaryRow>,
    pub covariance_rows: Vec<Nc1MaternCovarianceEntryRow>,
    pub fit_summary_rows: Vec<Nc1MaternFitSummaryRow>,
}

struct ModelComputation {
    n: usize,
    model: OneFormConvergenceModel,
    state_dofs: usize,
    precision_nnz: usize,
    factor_nnz: usize,
    covariance: GmrfDenseMatrix,
    build_seconds: f64,
    covariance_seconds: f64,
}

pub fn run_nc1_matern_convergence_experiment(
    config: &Nc1MaternConvergenceConfig,
) -> Result<Nc1MaternConvergenceResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mut computations = Vec::new();
    for &n in &config.levels {
        eprintln!("[nc1_matern_convergence] mesh n={n}");
        computations.push(compute_model(
            n,
            OneFormConvergenceModel::FullNc1Lumped,
            config,
        )?);
        computations.push(compute_model(
            n,
            OneFormConvergenceModel::WhitneyProjected,
            config,
        )?);
    }

    let result = build_result(&computations)?;
    write_summary_csv(&config.output_dir.join("summary.csv"), &result.summary_rows)?;
    write_covariance_csv(
        &config.output_dir.join("covariance_entries.csv"),
        &result.covariance_rows,
    )?;
    write_fit_summary_csv(
        &config.output_dir.join("fit_summary.csv"),
        &result.fit_summary_rows,
    )?;
    write_readme(&config.output_dir.join("README.md"), config)?;

    Ok(result)
}

fn compute_model(
    n: usize,
    model: OneFormConvergenceModel,
    config: &Nc1MaternConvergenceConfig,
) -> Result<ModelComputation, String> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, n, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let build_start = Instant::now();
    let (precision, rows) = match model {
        OneFormConvergenceModel::FullNc1Lumped => {
            let precision = build_full_nc1_matern_precision(&topology, &coords, &metric, config)?;
            let rows = nc1_observable_rows(&topology, &coords, &metric)?;
            (precision, rows)
        }
        OneFormConvergenceModel::WhitneyProjected => {
            let hodge = build_hodge_laplacian_1form(&topology, &metric);
            let precision = build_matern_precision_1form_for_alpha(
                &topology,
                &metric,
                &hodge,
                MaternAlpha::Two,
                WhitneyMaternConfig {
                    kappa: config.kappa,
                    tau: config.tau,
                    mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
                },
            );
            let rows = whitney_observable_rows(&topology, &coords, &metric)?;
            (precision, rows)
        }
    };
    let build_seconds = build_start.elapsed().as_secs_f64();

    let precision_nnz = precision.nnz();
    let state_dofs = precision.nrows();
    let covariance_start = Instant::now();
    let (covariance, factor_nnz) = exact_transformed_covariance(&precision, rows)?;
    let covariance_seconds = covariance_start.elapsed().as_secs_f64();

    Ok(ModelComputation {
        n,
        model,
        state_dofs,
        precision_nnz,
        factor_nnz,
        covariance,
        build_seconds,
        covariance_seconds,
    })
}

fn build_full_nc1_matern_precision(
    topology: &Complex,
    _coords: &MeshCoords,
    metric: &MeshLengths,
    config: &Nc1MaternConvergenceConfig,
) -> Result<FeecCsr, String> {
    build_full_nc1_matern_precision_alpha_two(topology, metric, config.kappa, config.tau)
}

pub(crate) fn build_full_nc1_matern_precision_alpha_two(
    topology: &Complex,
    metric: &MeshLengths,
    kappa: f64,
    tau: f64,
) -> Result<FeecCsr, String> {
    let system = build_full_nc1_system_matrix(topology, metric, kappa)?;
    let load_inverse = FeecCsr::from(&assemble_nc1_lumped_mass_inverse_galmat(topology, metric));
    if load_inverse.nrows() != system.nrows() || load_inverse.ncols() != system.ncols() {
        return Err(format!(
            "NC1 load inverse dimensions {}x{} do not match system {}x{}",
            load_inverse.nrows(),
            load_inverse.ncols(),
            system.nrows(),
            system.ncols()
        ));
    }
    Ok(build_lindgren_precision_from_system(
        &system,
        &load_inverse,
        MaternAlpha::Two,
        tau,
    ))
}

fn build_full_nc1_system_matrix(
    topology: &Complex,
    metric: &MeshLengths,
    kappa: f64,
) -> Result<FeecCsr, String> {
    if topology.dim() < 2 {
        return Err(format!(
            "full NC1 convergence experiment expects dimension at least 2, got dimension {}",
            topology.dim()
        ));
    }
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }

    let mass = FeecCsr::from(&assemble_nc1_mass_galmat(topology, metric));
    let d_stiffness = assemble_nc1_exterior_derivative_stiffness(topology, metric)?;
    let codiff_stiffness = assemble_nc1_codifferential_stiffness(topology, metric)?;
    let shifted_mass = scale_matrix(&mass, kappa * kappa);
    let system = add_sparse(&add_sparse(&shifted_mass, &d_stiffness), &codiff_stiffness);
    Ok(system)
}

fn assemble_nc1_codifferential_stiffness(
    topology: &Complex,
    metric: &MeshLengths,
) -> Result<FeecCsr, String> {
    let scalar_mass = FeecCsr::from(&assemble_galmat(
        topology,
        metric,
        HodgeMassElmat::new(topology.dim(), 0),
    ));
    let scalar_mass_inverse = diag_matrix(&invert_diag(&lumped_diag(&scalar_mass)));
    let coupling = assemble_scalar_nc1_codifferential_coupling(topology, metric)?;
    let weighted = &scalar_mass_inverse * &coupling;
    let coupling_t = coupling.transpose();
    Ok(&coupling_t * &weighted)
}

fn assemble_scalar_nc1_codifferential_coupling(
    topology: &Complex,
    metric: &MeshLengths,
) -> Result<FeecCsr, String> {
    let dim = topology.dim();
    let ndofs_nc1 = nc1_dof_count(topology);
    let mut coo = FeecCoo::new(topology.nsimplices(0), ndofs_nc1);
    let qr = SimplexQuadRule::order3(dim);
    let reference = SimplexCoords::standard(dim);
    let scalar_diffs = reference.difbarys_ext();

    for cell in topology.cells().handle_iter() {
        let geo = metric.simplex_lengths(cell);
        let local = local_scalar_nc1_codifferential_coupling(dim, &geo, &qr, &scalar_diffs)?;
        let local_edges = cell.mesh_subsimps(1).collect::<Vec<_>>();
        for local_vertex in 0..=dim {
            let row = cell[local_vertex];
            for (local_edge_index, edge) in local_edges.iter().enumerate() {
                for slot in 0..2 {
                    let local_col = 2 * local_edge_index + slot;
                    let value = local[(local_vertex, local_col)];
                    if value != 0.0 {
                        coo.push(row, nc1_global_dof(edge.kidx(), slot), value);
                    }
                }
            }
        }
    }

    Ok(FeecCsr::from(&coo))
}

fn local_scalar_nc1_codifferential_coupling(
    dim: usize,
    geo: &SimplexLengths,
    qr: &SimplexQuadRule,
    scalar_diffs: &[exterior::MultiForm],
) -> Result<FeecMatrix, String> {
    let local_edges = standard_subsimps(dim, 1).collect::<Vec<_>>();
    let nlocal_nc1 = 2 * local_edges.len();
    let inner = multi_gramian(&geo.to_metric_tensor().inverse(), 1);
    let local = qr.integrate_local(
        &|point: CoordRef| {
            let basis = nc1_local_basis_coeffs(dim, point);
            let mut values = FeecMatrix::zeros(dim + 1, nlocal_nc1);
            for vertex in 0..=dim {
                let scalar_diff = scalar_diffs[vertex].coeffs();
                for col in 0..nlocal_nc1 {
                    values[(vertex, col)] =
                        inner.inner(scalar_diff, &basis.column(col).into_owned());
                }
            }
            values
        },
        geo.vol(),
    );

    if local.nrows() != dim + 1 || local.ncols() != nlocal_nc1 {
        return Err("unexpected local scalar/NC1 coupling dimensions".to_string());
    }
    Ok(local)
}

fn assemble_nc1_exterior_derivative_stiffness(
    topology: &Complex,
    metric: &MeshLengths,
) -> Result<FeecCsr, String> {
    let dim = topology.dim();
    let ndofs = nc1_dof_count(topology);
    let mut coo = FeecCoo::new(ndofs, ndofs);

    for cell in topology.cells().handle_iter() {
        let geo = metric.simplex_lengths(cell);
        let local = local_nc1_exterior_derivative_stiffness(dim, &geo)?;
        let local_edges = cell.mesh_subsimps(1).collect::<Vec<_>>();
        for (local_i, edge_i) in local_edges.iter().enumerate() {
            for slot_i in 0..2 {
                let row = nc1_global_dof(edge_i.kidx(), slot_i);
                let i = 2 * local_i + slot_i;
                for (local_j, edge_j) in local_edges.iter().enumerate() {
                    for slot_j in 0..2 {
                        let value = local[(i, 2 * local_j + slot_j)];
                        if value != 0.0 {
                            coo.push(row, nc1_global_dof(edge_j.kidx(), slot_j), value);
                        }
                    }
                }
            }
        }
    }

    Ok(FeecCsr::from(&coo))
}

fn local_nc1_exterior_derivative_stiffness(
    dim: usize,
    geo: &SimplexLengths,
) -> Result<FeecMatrix, String> {
    if dim < 2 {
        return Err("NC1 derivative stiffness requires dimension at least 2".to_string());
    }
    let derivs = nc1_local_derivative_coeffs(dim);
    let inner = multi_gramian(&geo.to_metric_tensor().inverse(), 2);
    Ok(geo.vol() * inner.inner_mat(&derivs, &derivs))
}

fn nc1_observable_rows(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
) -> Result<Vec<Vec<(usize, f64)>>, String> {
    SmoothOneFormObservable::all()
        .iter()
        .copied()
        .map(|observable| {
            let source = observable.source();
            let load = assemble_nc1_source_load(topology, coords, metric, &source)?;
            Ok(vector_to_sparse_row(&load, SPARSE_ROW_TOLERANCE))
        })
        .collect()
}

fn whitney_observable_rows(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
) -> Result<Vec<Vec<(usize, f64)>>, String> {
    SmoothOneFormObservable::all()
        .iter()
        .copied()
        .map(|observable| {
            let source = observable.source();
            let load = assemble_galvec(topology, metric, SourceElVec::new(&source, coords, None));
            Ok(vector_to_sparse_row(&load, SPARSE_ROW_TOLERANCE))
        })
        .collect()
}

fn assemble_nc1_source_load(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    source: &DiffFormClosure,
) -> Result<FeecVector, String> {
    if source.grade() != 1 {
        return Err(format!(
            "NC1 source load expects a 1-form source, got grade {}",
            source.grade()
        ));
    }
    if source.dim_ambient() != coords.dim() {
        return Err(format!(
            "source ambient dimension {} does not match coordinate dimension {}",
            source.dim_ambient(),
            coords.dim()
        ));
    }

    let dim = topology.dim();
    let ndofs = nc1_dof_count(topology);
    let qr = SimplexQuadRule::order3(dim);
    let mut load = FeecVector::zeros(ndofs);

    for cell in topology.cells().handle_iter() {
        let geo = metric.simplex_lengths(cell);
        let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
        let inner = multi_gramian(&geo.to_metric_tensor().inverse(), 1);
        let local_edges = cell.mesh_subsimps(1).collect::<Vec<_>>();
        let nlocal = 2 * local_edges.len();
        let local_values = qr.integrate_local(
            &|point: CoordRef| {
                let global = cell_coords.local2global(point);
                let reference_source = cell_coords.pullback_form(&source.at_point(&global));
                let basis = nc1_local_basis_coeffs(dim, point);
                let mut values = FeecVector::zeros(nlocal);
                for col in 0..nlocal {
                    values[col] =
                        inner.inner(reference_source.coeffs(), &basis.column(col).into_owned());
                }
                values
            },
            geo.vol(),
        );

        for (local_edge_index, edge) in local_edges.iter().enumerate() {
            for slot in 0..2 {
                load[nc1_global_dof(edge.kidx(), slot)] +=
                    local_values[2 * local_edge_index + slot];
            }
        }
    }

    Ok(load)
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

fn nc1_local_derivative_coeffs(dim: usize) -> FeecMatrix {
    let local_edges = standard_subsimps(dim, 1).collect::<Vec<_>>();
    let mut coeffs = FeecMatrix::zeros(exterior_dim(dim, 2), 2 * local_edges.len());
    for (edge_index, edge) in local_edges.into_iter().enumerate() {
        let deriv = WhitneyLsf::standard(dim, edge).dif().into_coeffs() * 0.5;
        coeffs.set_column(2 * edge_index, &deriv);
        coeffs.set_column(2 * edge_index + 1, &deriv);
    }
    coeffs
}

fn exact_transformed_covariance(
    precision: &FeecCsr,
    rows: Vec<Vec<(usize, f64)>>,
) -> Result<(GmrfDenseMatrix, usize), String> {
    let gmrf_precision = feec_csr_to_gmrf(precision);
    let factor = gmrf_precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor precision: {err}"))?;
    let factor_nnz = factor.nnz();
    let operator = SparseRowOperator::new(precision.nrows(), rows)
        .map_err(|err| format!("failed to build observable operator: {err}"))?;
    let mut gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), gmrf_precision)
            .map_err(|err| format!("failed to build GMRF: {err}"))?
            .with_precision_sqrt(factor);
    let covariance = gmrf
        .exact_transformed_covariance(&operator)
        .map_err(|err| format!("failed to compute transformed covariance: {err}"))?;
    Ok((covariance, factor_nnz))
}

fn build_result(computations: &[ModelComputation]) -> Result<Nc1MaternConvergenceResult, String> {
    let observables = SmoothOneFormObservable::all();
    let reference = computations
        .iter()
        .filter(|entry| entry.model == OneFormConvergenceModel::FullNc1Lumped)
        .max_by_key(|entry| entry.n)
        .ok_or_else(|| "missing full-NC1 reference covariance".to_string())?;

    let mut summary_rows = Vec::new();
    let mut covariance_rows = Vec::new();
    for computation in computations {
        let same_level_full = computations
            .iter()
            .find(|entry| {
                entry.n == computation.n && entry.model == OneFormConvergenceModel::FullNc1Lumped
            })
            .ok_or_else(|| {
                format!(
                    "missing same-level full-NC1 covariance for n={}",
                    computation.n
                )
            })?;

        let h = 1.0 / computation.n as f64;
        let relative_reference_error =
            relative_frobenius_difference(&computation.covariance, &reference.covariance);
        let same_level_gap =
            relative_frobenius_difference(&computation.covariance, &same_level_full.covariance);
        summary_rows.push(Nc1MaternSummaryRow {
            n: computation.n,
            h,
            model: computation.model,
            state_dofs: computation.state_dofs,
            precision_nnz: computation.precision_nnz,
            factor_nnz: computation.factor_nnz,
            covariance_frobenius_norm: frobenius_norm(&computation.covariance),
            reference_relative_frobenius_error: relative_reference_error,
            same_level_relative_gap_vs_full_nc1: same_level_gap,
            build_seconds: computation.build_seconds,
            covariance_seconds: computation.covariance_seconds,
            status: computation.model.status().to_string(),
        });

        for i in 0..observables.len() {
            for j in 0..observables.len() {
                let covariance = computation.covariance[(i, j)];
                let reference_covariance = reference.covariance[(i, j)];
                let same_level_full_covariance = same_level_full.covariance[(i, j)];
                covariance_rows.push(Nc1MaternCovarianceEntryRow {
                    n: computation.n,
                    h,
                    model: computation.model,
                    observable_row: observables[i].id().to_string(),
                    observable_col: observables[j].id().to_string(),
                    covariance,
                    reference_covariance,
                    same_level_full_nc1_covariance: same_level_full_covariance,
                    absolute_reference_error: (covariance - reference_covariance).abs(),
                    relative_reference_error: (covariance - reference_covariance).abs()
                        / reference_covariance.abs().max(EPS),
                });
            }
        }
    }

    let fit_summary_rows = fit_summary_rows(&summary_rows);
    Ok(Nc1MaternConvergenceResult {
        summary_rows,
        covariance_rows,
        fit_summary_rows,
    })
}

fn fit_summary_rows(rows: &[Nc1MaternSummaryRow]) -> Vec<Nc1MaternFitSummaryRow> {
    [
        OneFormConvergenceModel::FullNc1Lumped,
        OneFormConvergenceModel::WhitneyProjected,
    ]
    .into_iter()
    .flat_map(|model| {
        let mut model_rows = rows
            .iter()
            .filter(|row| row.model == model)
            .collect::<Vec<_>>();
        model_rows.sort_by_key(|row| row.n);
        if model_rows.len() < 2 {
            return Vec::new();
        }

        let mut out = Vec::new();
        let xs = model_rows
            .iter()
            .filter(|row| row.reference_relative_frobenius_error > EPS)
            .map(|row| row.h.ln())
            .collect::<Vec<_>>();
        let ys = model_rows
            .iter()
            .filter(|row| row.reference_relative_frobenius_error > EPS)
            .map(|row| row.reference_relative_frobenius_error.ln())
            .collect::<Vec<_>>();
        if xs.len() >= 2 {
            out.push(Nc1MaternFitSummaryRow {
                model,
                diagnostic: "loglog_reference_error_slope_vs_h".to_string(),
                value: linear_slope(&xs, &ys),
                expected: "positive for covariance convergence".to_string(),
                status: model.status().to_string(),
            });
        }

        let finest = model_rows[model_rows.len() - 1];
        out.push(Nc1MaternFitSummaryRow {
            model,
            diagnostic: "finest_same_level_relative_gap_vs_full_nc1".to_string(),
            value: finest.same_level_relative_gap_vs_full_nc1,
            expected: "small if Whitney sparse approximation tracks the NC1 scheme".to_string(),
            status: model.status().to_string(),
        });
        out
    })
    .collect()
}

fn vector_to_sparse_row(vector: &FeecVector, tolerance: f64) -> Vec<(usize, f64)> {
    vector
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| (value.abs() > tolerance).then_some((index, value)))
        .collect()
}

fn nc1_dof_count(topology: &Complex) -> usize {
    2 * topology.nsimplices(1)
}

fn nc1_global_dof(edge_kidx: usize, slot: usize) -> usize {
    2 * edge_kidx + slot
}

fn frobenius_norm(matrix: &GmrfDenseMatrix) -> f64 {
    let mut sum = 0.0;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            sum += matrix[(i, j)].powi(2);
        }
    }
    sum.sqrt()
}

fn relative_frobenius_difference(lhs: &GmrfDenseMatrix, rhs: &GmrfDenseMatrix) -> f64 {
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    let mut sum = 0.0;
    for i in 0..lhs.nrows() {
        for j in 0..lhs.ncols() {
            sum += (lhs[(i, j)] - rhs[(i, j)]).powi(2);
        }
    }
    sum.sqrt() / frobenius_norm(rhs).max(EPS)
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

fn validate_config(config: &Nc1MaternConvergenceConfig) -> Result<(), String> {
    if config.levels.is_empty() {
        return Err("at least one mesh level is required".to_string());
    }
    if config.levels.contains(&0) {
        return Err("mesh levels must be positive".to_string());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    Ok(())
}

pub fn write_summary_csv(path: &Path, rows: &[Nc1MaternSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,h,model,state_dofs,precision_nnz,factor_nnz,covariance_frobenius_norm,reference_relative_frobenius_error,same_level_relative_gap_vs_full_nc1,build_seconds,covariance_seconds,status"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            row.n,
            row.h,
            row.model.as_str(),
            row.state_dofs,
            row.precision_nnz,
            row.factor_nnz,
            row.covariance_frobenius_norm,
            row.reference_relative_frobenius_error,
            row.same_level_relative_gap_vs_full_nc1,
            row.build_seconds,
            row.covariance_seconds,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_covariance_csv(path: &Path, rows: &[Nc1MaternCovarianceEntryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "n,h,model,observable_row,observable_col,covariance,reference_covariance,same_level_full_nc1_covariance,absolute_reference_error,relative_reference_error"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.17e},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            row.n,
            row.h,
            row.model.as_str(),
            row.observable_row,
            row.observable_col,
            row.covariance,
            row.reference_covariance,
            row.same_level_full_nc1_covariance,
            row.absolute_reference_error,
            row.relative_reference_error
        )?;
    }
    Ok(())
}

pub fn write_fit_summary_csv(path: &Path, rows: &[Nc1MaternFitSummaryRow]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "model,diagnostic,value,expected,status")?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{:.17e},{},{}",
            row.model.as_str(),
            row.diagnostic,
            row.value,
            row.expected,
            row.status
        )?;
    }
    Ok(())
}

pub fn write_readme(path: &Path, config: &Nc1MaternConvergenceConfig) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# NC1 1-form Matern covariance convergence")?;
    writeln!(writer)?;
    writeln!(writer, "- levels: {:?}", config.levels)?;
    writeln!(writer, "- kappa: {:.17}", config.kappa)?;
    writeln!(writer, "- tau: {:.17}", config.tau)?;
    writeln!(
        writer,
        "- models: full NC1 row-sum/vertex-lumped proof-compliant prior and Whitney projected sparse inverse prior"
    )?;
    writeln!(
        writer,
        "- observables: fixed smooth polynomial L2 1-form test functionals"
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "`summary.csv` reports covariance-matrix convergence diagnostics. `covariance_entries.csv` reports every entry of the fixed observable covariance matrix."
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn nc1_basis_splits_whitney_basis() {
        let dim = 3;
        let point = FeecVector::from_vec(vec![0.2, 0.3, 0.1]);
        let basis = nc1_local_basis_coeffs(dim, point.as_view());
        let edges = standard_subsimps(dim, 1).collect::<Vec<_>>();
        for (edge_index, edge) in edges.into_iter().enumerate() {
            let whitney = WhitneyLsf::standard(dim, edge).at_point(point.as_view());
            let split = basis.column(2 * edge_index) + basis.column(2 * edge_index + 1);
            assert!(
                (&split - whitney.coeffs()).norm() <= 1e-12,
                "NC1 split basis should sum to the Whitney basis"
            );
        }
    }

    #[test]
    fn small_nc1_convergence_experiment_runs() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        let output_dir = std::env::temp_dir().join(format!("nc1_matern_convergence_{stamp}"));
        let config = Nc1MaternConvergenceConfig {
            levels: vec![1, 2],
            output_dir: output_dir.clone(),
            kappa: 2.0,
            tau: 1.0,
        };
        let result = run_nc1_matern_convergence_experiment(&config)
            .expect("small NC1 convergence experiment should run");
        assert_eq!(result.summary_rows.len(), 4);
        assert_eq!(
            result.covariance_rows.len(),
            4 * SmoothOneFormObservable::all().len() * SmoothOneFormObservable::all().len()
        );
        assert!(result.summary_rows.iter().all(|row| {
            row.covariance_frobenius_norm.is_finite()
                && row.reference_relative_frobenius_error.is_finite()
                && row.same_level_relative_gap_vs_full_nc1.is_finite()
        }));
        assert!(output_dir.join("summary.csv").exists());
        assert!(output_dir.join("covariance_entries.csv").exists());
        let _ = fs::remove_dir_all(output_dir);
    }
}
