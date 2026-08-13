use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::cochain::cochain_projection;
use exterior::field::DiffFormClosure;
use feg_infer::{
    prior::matern::{
        one_form::{
            build_hodge_laplacian_1form, build_matern_mass_inverse_1form,
            build_matern_precision_1form_with_mass_inverse_for_alpha, HodgeLaplacian1Form,
            MaternMassInverse,
        },
        MaternAlpha,
    },
    sparse::{
        dense_to_feec_csr, diag_matrix, feec_csr_to_dense, feec_csr_to_gmrf, gmrf_vec_to_feec,
        invert_diag, matrix_diag,
    },
};
use formoniq::assemble::{
    assemble_barycentric_dual_1form_sparse_inverse_galmat, BarycentricDualSparseInverseConfig,
};
use gmrf_core::{
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::{
        coord::mesh::MeshCoords, coord::quadrature::SimplexQuadRule, metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
    time::Instant,
};

const DEFAULT_DROP_TOLERANCE: f64 = 1e-14;
const DEFAULT_MAX_CONSISTENT_DOFS: usize = 5_000;

fn default_barycentric_stabilization_factors() -> Vec<f64> {
    vec![0.25, 0.5, 1.0, 2.0, 4.0]
}

#[derive(Debug, Clone)]
pub struct CubeMassInverseVarianceConfig {
    pub levels: Vec<usize>,
    pub kappa: f64,
    pub tau: f64,
    pub drop_tolerance: f64,
    pub max_consistent_dofs: usize,
    pub barycentric_stabilization_factors: Vec<f64>,
    pub include_barycentric_oracle: bool,
}

impl Default for CubeMassInverseVarianceConfig {
    fn default() -> Self {
        Self {
            levels: vec![2, 4, 6, 8],
            kappa: 4.0,
            tau: 1.0,
            drop_tolerance: DEFAULT_DROP_TOLERANCE,
            max_consistent_dofs: DEFAULT_MAX_CONSISTENT_DOFS,
            barycentric_stabilization_factors: default_barycentric_stabilization_factors(),
            include_barycentric_oracle: true,
        }
    }
}

impl CubeMassInverseVarianceConfig {
    fn validate(&self) -> Result<(), String> {
        if self.levels.is_empty() {
            return Err("at least one cube mesh level is required".to_string());
        }
        if self.levels.contains(&0) {
            return Err("cube mesh levels must be positive".to_string());
        }
        if !self.kappa.is_finite() || self.kappa <= 0.0 {
            return Err("kappa must be finite and positive".to_string());
        }
        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err("tau must be finite and positive".to_string());
        }
        if !self.drop_tolerance.is_finite() || self.drop_tolerance < 0.0 {
            return Err("drop_tolerance must be finite and nonnegative".to_string());
        }
        if self.max_consistent_dofs == 0 {
            return Err("max_consistent_dofs must be positive".to_string());
        }
        if self.barycentric_stabilization_factors.is_empty() {
            return Err("at least one barycentric stabilization factor is required".to_string());
        }
        for &factor in &self.barycentric_stabilization_factors {
            if !factor.is_finite() || factor <= 0.0 {
                return Err(
                    "barycentric stabilization factors must be finite and positive".to_string(),
                );
            }
        }
        let mut labels = Vec::with_capacity(self.barycentric_stabilization_factors.len());
        for &factor in &self.barycentric_stabilization_factors {
            let label = barycentric_strategy_label(factor);
            if labels.iter().any(|existing| existing == &label) {
                return Err(format!(
                    "barycentric stabilization factor {factor} produces duplicate strategy label {label}"
                ));
            }
            labels.push(label);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CubeMassInverseKind {
    RowSumLumped,
    MassDiagonalInverse,
    Nc1ProjectedSparseInverse,
    BarycentricDualSparseInverse,
    ExactConsistentInverse,
}

impl CubeMassInverseKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::RowSumLumped => "row_sum_lumped",
            Self::MassDiagonalInverse => "mass_diagonal_inverse",
            Self::Nc1ProjectedSparseInverse => "nc1_projected_sparse_inverse",
            Self::BarycentricDualSparseInverse => "barycentric_dual_sparse_inverse",
            Self::ExactConsistentInverse => "exact_consistent_inverse",
        }
    }

    pub fn all() -> [Self; 5] {
        [
            Self::RowSumLumped,
            Self::MassDiagonalInverse,
            Self::Nc1ProjectedSparseInverse,
            Self::BarycentricDualSparseInverse,
            Self::ExactConsistentInverse,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct CubeMassInverseVarianceReport {
    pub config: CubeMassInverseVarianceConfig,
    pub levels: Vec<CubeMassInverseLevelReport>,
}

#[derive(Debug, Clone)]
pub struct CubeMassInverseLevelReport {
    pub level: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub tetrahedron_count: usize,
    pub edge_geometry: Vec<CubeEdgeGeometry>,
    pub strategies: Vec<CubeMassInverseStrategyReport>,
}

#[derive(Debug, Clone)]
pub struct CubeEdgeGeometry {
    pub edge_index: usize,
    pub vertices: [usize; 2],
    pub midpoint: [f64; 3],
    pub length: f64,
    pub is_boundary: bool,
}

#[derive(Debug, Clone)]
pub struct CubeMassInverseStrategyReport {
    pub kind: CubeMassInverseKind,
    pub label: String,
    pub barycentric_stabilization_factor: Option<f64>,
    pub is_oracle_calibrated: bool,
    pub mass_inverse_nnz: usize,
    pub mass_inverse_density: f64,
    pub mass_inverse_eigen: CubeEigenStats,
    pub consistency_error: f64,
    pub precision_nnz: usize,
    pub precision_density: f64,
    pub precision_lower_nnz: usize,
    pub factor_nnz: usize,
    pub factor_density: f64,
    pub fill_in_ratio: f64,
    pub timings: CubeMassInverseTimings,
    pub variance_stats: CubeVarianceStats,
    pub comparison_to_consistent: CubeVarianceComparison,
    pub subset_summaries: Vec<CubeVarianceSubsetSummary>,
    pub variances: FeecVector,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CubeMassInverseTimings {
    pub mass_inverse_seconds: f64,
    pub precision_seconds: f64,
    pub factor_seconds: f64,
    pub variance_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CubeVarianceStats {
    pub min: f64,
    pub mean: f64,
    pub median: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CubeVarianceComparison {
    pub rms_log_delta_vs_consistent: f64,
    pub max_abs_log_delta_vs_consistent: f64,
    pub rms_relative_delta_vs_consistent: f64,
    pub max_abs_relative_delta_vs_consistent: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CubeEigenStats {
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub condition_number: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeEdgeSubset {
    All,
    Interior,
    Boundary,
}

impl CubeEdgeSubset {
    pub fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Interior => "interior",
            Self::Boundary => "boundary",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CubeVarianceSubsetSummary {
    pub subset: CubeEdgeSubset,
    pub edge_count: usize,
    pub variance_stats: CubeVarianceStats,
    pub comparison_to_consistent: CubeVarianceComparison,
}

pub fn compute_matern_1form_cube_mass_inverse_variance_report(
    config: CubeMassInverseVarianceConfig,
) -> Result<CubeMassInverseVarianceReport, String> {
    config.validate()?;

    let mut levels = Vec::with_capacity(config.levels.len());
    for &level in &config.levels {
        levels.push(compute_level_report(level, &config)?);
    }

    Ok(CubeMassInverseVarianceReport { config, levels })
}

pub fn write_matern_1form_cube_mass_inverse_variance_outputs(
    report: &CubeMassInverseVarianceReport,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_summary_csv(report, &out_dir.join("summary.csv"))?;
    write_fit_summary_csv(report, &out_dir.join("fit_summary.csv"))?;
    for level in &report.levels {
        write_interior_edge_variance_csv(
            level,
            &out_dir.join(format!("interior_edge_variances_level_{}.csv", level.level)),
        )?;
    }
    Ok(())
}

fn compute_level_report(
    level: usize,
    config: &CubeMassInverseVarianceConfig,
) -> Result<CubeMassInverseLevelReport, String> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, level, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let edge_count = topology.nsimplices(1);
    if edge_count > config.max_consistent_dofs {
        return Err(format!(
            "cube level {level} has {edge_count} edge dofs, exceeding max_consistent_dofs {}; rerun with a larger --max-consistent-dofs to include this level",
            config.max_consistent_dofs
        ));
    }

    let metric = coords.to_edge_lengths(&topology);
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let edge_geometry = cube_edge_geometry(&topology, &coords)?;
    let interior_indices = interior_edge_indices(&edge_geometry);
    if interior_indices.is_empty() {
        return Err(format!(
            "cube level {level} has no interior edge dofs for interior-only variance reporting"
        ));
    }
    let consistency_cochain = smooth_polynomial_one_form_cochain(&topology, &coords);

    let requests = strategy_requests(config);
    let mut strategies =
        Vec::with_capacity(requests.len() + usize::from(config.include_barycentric_oracle));
    for request in requests {
        strategies.push(compute_strategy_report(
            request,
            &topology,
            &coords,
            &metric,
            &hodge,
            &consistency_cochain,
            config,
        )?);
    }

    let consistent = strategies
        .iter()
        .find(|strategy| strategy.kind == CubeMassInverseKind::ExactConsistentInverse)
        .ok_or_else(|| "missing exact-consistent strategy report".to_string())?
        .variances
        .clone();
    for strategy in &mut strategies {
        strategy.variance_stats =
            variance_stats_for_indices(&strategy.variances, &interior_indices)?;
        strategy.comparison_to_consistent =
            compare_variances_for_indices(&strategy.variances, &consistent, &interior_indices)?;
        strategy.subset_summaries = summarize_interior_strategy_subset(
            &strategy.variances,
            &consistent,
            &interior_indices,
        )?;
    }

    if config.include_barycentric_oracle {
        let oracle = oracle_barycentric_strategy(&strategies)?;
        strategies.push(oracle);
    }

    Ok(CubeMassInverseLevelReport {
        level,
        vertex_count: topology.nsimplices(0),
        edge_count,
        face_count: topology.nsimplices(2),
        tetrahedron_count: topology.nsimplices(3),
        edge_geometry,
        strategies,
    })
}

#[derive(Debug, Clone)]
struct CubeMassInverseStrategyRequest {
    kind: CubeMassInverseKind,
    label: String,
    barycentric_stabilization_factor: Option<f64>,
}

fn strategy_requests(
    config: &CubeMassInverseVarianceConfig,
) -> Vec<CubeMassInverseStrategyRequest> {
    let mut requests = Vec::with_capacity(4 + config.barycentric_stabilization_factors.len());
    requests.push(CubeMassInverseStrategyRequest {
        kind: CubeMassInverseKind::RowSumLumped,
        label: CubeMassInverseKind::RowSumLumped.label().to_string(),
        barycentric_stabilization_factor: None,
    });
    requests.push(CubeMassInverseStrategyRequest {
        kind: CubeMassInverseKind::MassDiagonalInverse,
        label: CubeMassInverseKind::MassDiagonalInverse.label().to_string(),
        barycentric_stabilization_factor: None,
    });
    requests.push(CubeMassInverseStrategyRequest {
        kind: CubeMassInverseKind::Nc1ProjectedSparseInverse,
        label: CubeMassInverseKind::Nc1ProjectedSparseInverse
            .label()
            .to_string(),
        barycentric_stabilization_factor: None,
    });
    for &factor in &config.barycentric_stabilization_factors {
        requests.push(CubeMassInverseStrategyRequest {
            kind: CubeMassInverseKind::BarycentricDualSparseInverse,
            label: barycentric_strategy_label(factor),
            barycentric_stabilization_factor: Some(factor),
        });
    }
    requests.push(CubeMassInverseStrategyRequest {
        kind: CubeMassInverseKind::ExactConsistentInverse,
        label: CubeMassInverseKind::ExactConsistentInverse
            .label()
            .to_string(),
        barycentric_stabilization_factor: None,
    });
    requests
}

fn barycentric_strategy_label(stabilization_factor: f64) -> String {
    if stabilization_factor == 1.0 {
        CubeMassInverseKind::BarycentricDualSparseInverse
            .label()
            .to_string()
    } else {
        format!(
            "{}_stab_{}",
            CubeMassInverseKind::BarycentricDualSparseInverse.label(),
            factor_label_token(stabilization_factor)
        )
    }
}

fn factor_label_token(value: f64) -> String {
    let raw = if (1e-4..1e4).contains(&value.abs()) {
        let mut decimal = format!("{value:.12}");
        while decimal.contains('.') && decimal.ends_with('0') {
            decimal.pop();
        }
        if decimal.ends_with('.') {
            decimal.pop();
        }
        decimal
    } else {
        format!("{value:.6e}")
    };
    raw.replace('.', "p").replace('-', "m").replace('+', "")
}

fn oracle_barycentric_strategy(
    strategies: &[CubeMassInverseStrategyReport],
) -> Result<CubeMassInverseStrategyReport, String> {
    let mut best = strategies
        .iter()
        .filter(|strategy| {
            strategy.kind == CubeMassInverseKind::BarycentricDualSparseInverse
                && !strategy.is_oracle_calibrated
        })
        .min_by(|lhs, rhs| {
            lhs.comparison_to_consistent
                .rms_log_delta_vs_consistent
                .partial_cmp(&rhs.comparison_to_consistent.rms_log_delta_vs_consistent)
                .expect("finite RMS log deltas should compare")
        })
        .cloned()
        .ok_or_else(|| {
            "cannot build barycentric oracle without barycentric sweep rows".to_string()
        })?;

    best.label = format!(
        "{}_oracle",
        CubeMassInverseKind::BarycentricDualSparseInverse.label()
    );
    best.is_oracle_calibrated = true;
    Ok(best)
}

fn compute_strategy_report(
    request: CubeMassInverseStrategyRequest,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    consistency_cochain: &FeecVector,
    config: &CubeMassInverseVarianceConfig,
) -> Result<CubeMassInverseStrategyReport, String> {
    let total_start = Instant::now();

    let mass_inverse_start = Instant::now();
    let mass_inverse = build_mass_inverse(&request, topology, coords, metric, hodge, config)?;
    let mass_inverse_seconds = mass_inverse_start.elapsed().as_secs_f64();
    let mass_inverse_nnz = mass_inverse.nnz();
    let mass_inverse_density = sparse_density(&mass_inverse);
    let mass_inverse_eigen = symmetric_extreme_eigenvalues(&mass_inverse)?;
    let consistency_error =
        mass_inverse_consistency_error(&mass_inverse, &hodge.mass_u, consistency_cochain)?;

    let precision_start = Instant::now();
    let precision = build_matern_precision_1form_with_mass_inverse_for_alpha(
        hodge,
        &mass_inverse,
        MaternAlpha::Two,
        config.kappa,
        config.tau,
    );
    let precision_seconds = precision_start.elapsed().as_secs_f64();
    let precision_nnz = precision.nnz();
    let precision_density = sparse_density(&precision);
    let precision_lower_nnz = lower_triangle_nnz(&precision);

    let (variances, factor_nnz, factor_seconds, variance_seconds) =
        exact_marginal_variances(&precision)?;
    let factor_density = triangular_density(factor_nnz, precision.nrows());
    let fill_in_ratio = factor_nnz as f64 / precision_lower_nnz.max(1) as f64;

    Ok(CubeMassInverseStrategyReport {
        kind: request.kind,
        label: request.label,
        barycentric_stabilization_factor: request.barycentric_stabilization_factor,
        is_oracle_calibrated: false,
        mass_inverse_nnz,
        mass_inverse_density,
        mass_inverse_eigen,
        consistency_error,
        precision_nnz,
        precision_density,
        precision_lower_nnz,
        factor_nnz,
        factor_density,
        fill_in_ratio,
        timings: CubeMassInverseTimings {
            mass_inverse_seconds,
            precision_seconds,
            factor_seconds,
            variance_seconds,
            total_seconds: total_start.elapsed().as_secs_f64(),
        },
        variance_stats: CubeVarianceStats::default(),
        comparison_to_consistent: CubeVarianceComparison::default(),
        subset_summaries: Vec::new(),
        variances,
    })
}

fn build_mass_inverse(
    request: &CubeMassInverseStrategyRequest,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: &CubeMassInverseVarianceConfig,
) -> Result<FeecCsr, String> {
    match request.kind {
        CubeMassInverseKind::RowSumLumped => Ok(build_matern_mass_inverse_1form(
            topology,
            metric,
            &hodge.mass_u,
            MaternMassInverse::RowSumLumped,
        )),
        CubeMassInverseKind::MassDiagonalInverse => Ok(diag_matrix(&invert_diag(
            matrix_diag(&hodge.mass_u).as_slice(),
        ))),
        CubeMassInverseKind::Nc1ProjectedSparseInverse => Ok(build_matern_mass_inverse_1form(
            topology,
            metric,
            &hodge.mass_u,
            MaternMassInverse::Nc1ProjectedSparseInverse,
        )),
        CubeMassInverseKind::BarycentricDualSparseInverse => {
            let stabilization_factor =
                request.barycentric_stabilization_factor.ok_or_else(|| {
                    "barycentric-dual strategy request is missing a stabilization factor"
                        .to_string()
                })?;
            Ok(FeecCsr::from(
                &assemble_barycentric_dual_1form_sparse_inverse_galmat(
                    topology,
                    coords,
                    BarycentricDualSparseInverseConfig {
                        stabilization_factor,
                        ..BarycentricDualSparseInverseConfig::default()
                    },
                )?,
            ))
        }
        CubeMassInverseKind::ExactConsistentInverse => {
            let inverse = feec_csr_to_dense(&hodge.mass_u)
                .try_inverse()
                .ok_or_else(|| {
                    "failed to invert consistent Whitney 1-form mass matrix".to_string()
                })?;
            let inverse = (&inverse + inverse.transpose()) * 0.5;
            Ok(dense_to_feec_csr(&inverse, config.drop_tolerance))
        }
    }
}

fn exact_marginal_variances(precision: &FeecCsr) -> Result<(FeecVector, usize, f64, f64), String> {
    let gmrf_precision = feec_csr_to_gmrf(precision);
    let dim = gmrf_precision.nrows();

    let factor_start = Instant::now();
    let factor = gmrf_precision
        .cholesky_sqrt_lower()
        .map_err(|err| err.to_string())?;
    let factor_seconds = factor_start.elapsed().as_secs_f64();
    let factor_nnz = factor.nnz();

    let variance_start = Instant::now();
    let constraints = GmrfDenseMatrix::zeros(0, dim);
    let mut gmrf = Gmrf::from_mean_and_precision(GmrfVector::zeros(dim), gmrf_precision)
        .map_err(|err| err.to_string())?
        .with_precision_sqrt(factor);
    let decomposition = gmrf
        .exact_constrained_variance_decomposition(&constraints)
        .map_err(|err| err.to_string())?;
    let variance_seconds = variance_start.elapsed().as_secs_f64();

    Ok((
        gmrf_vec_to_feec(&decomposition.unconstrained_diag),
        factor_nnz,
        factor_seconds,
        variance_seconds,
    ))
}

fn cube_edge_geometry(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<CubeEdgeGeometry>, String> {
    if coords.dim() != 3 {
        return Err(format!(
            "cube edge geometry expects 3D coordinates, got dimension {}",
            coords.dim()
        ));
    }

    let boundary_edges = boundary_edge_flags(topology);
    topology
        .edges()
        .handle_iter()
        .enumerate()
        .map(|(edge_index, edge)| {
            let vertices: [usize; 2] = (*edge)
                .clone()
                .try_into()
                .map_err(|_| "failed to convert edge simplex to two vertices".to_string())?;
            let a = coords.coord(vertices[0]);
            let b = coords.coord(vertices[1]);
            let midpoint = [
                0.5 * (a[0] + b[0]),
                0.5 * (a[1] + b[1]),
                0.5 * (a[2] + b[2]),
            ];
            let length = (b - a).norm();
            Ok(CubeEdgeGeometry {
                edge_index,
                vertices,
                midpoint,
                length,
                is_boundary: boundary_edges[edge.kidx()],
            })
        })
        .collect()
}

fn boundary_edge_flags(topology: &Complex) -> Vec<bool> {
    let mut flags = vec![false; topology.nsimplices(1)];
    for facet_idx in topology.boundary_facets() {
        let facet = facet_idx.handle(topology);
        for edge in facet.mesh_subsimps(1) {
            flags[edge.kidx()] = true;
        }
    }
    flags
}

fn sparse_density(matrix: &FeecCsr) -> f64 {
    let entries = matrix.nrows() * matrix.ncols();
    if entries == 0 {
        0.0
    } else {
        matrix.nnz() as f64 / entries as f64
    }
}

fn lower_triangle_nnz(matrix: &FeecCsr) -> usize {
    matrix
        .triplet_iter()
        .filter(|(row, col, value)| row >= col && value.abs() > 0.0)
        .count()
}

fn triangular_density(nnzs: usize, dimension: usize) -> f64 {
    let entries = dimension.saturating_mul(dimension + 1) / 2;
    nnzs as f64 / entries.max(1) as f64
}

pub fn symmetric_extreme_eigenvalues(matrix: &FeecCsr) -> Result<CubeEigenStats, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "eigenvalue diagnostics require a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if matrix.nrows() == 0 {
        return Err("eigenvalue diagnostics require a non-empty matrix".to_string());
    }

    let dense = feec_csr_to_dense(matrix);
    let sym = (&dense + dense.transpose()) * 0.5;
    let eigen = sym.symmetric_eigen();
    let mut values = eigen.eigenvalues.iter().copied().collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("eigenvalue diagnostics produced a non-finite eigenvalue".to_string());
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues should compare"));
    let lambda_min = values[0];
    let lambda_max = *values.last().expect("non-empty eigenvalue vector");
    if lambda_min <= 0.0 {
        return Err(format!(
            "eigenvalue diagnostics require an SPD matrix, got lambda_min={lambda_min:.6e}"
        ));
    }

    Ok(CubeEigenStats {
        lambda_min,
        lambda_max,
        condition_number: lambda_max / lambda_min,
    })
}

fn smooth_polynomial_one_form_cochain(topology: &Complex, coords: &MeshCoords) -> FeecVector {
    let omega = DiffFormClosure::one_form(
        |p| {
            FeecVector::from_vec(vec![
                1.0 + p[0] + p[1] * p[2],
                -0.5 + 0.25 * p[0] + p[1] * p[1],
                0.75 + p[2] + p[0] * p[1],
            ])
        },
        3,
    );
    let quadrature = SimplexQuadRule::order3(1);
    cochain_projection(&omega, topology, coords, Some(&quadrature)).coeffs
}

fn mass_inverse_consistency_error(
    mass_inverse: &FeecCsr,
    mass: &FeecCsr,
    cochain: &FeecVector,
) -> Result<f64, String> {
    if mass_inverse.nrows() != mass.nrows()
        || mass_inverse.ncols() != mass.ncols()
        || mass.ncols() != cochain.len()
    {
        return Err(format!(
            "consistency dimensions do not align: E={}x{}, M={}x{}, c={}",
            mass_inverse.nrows(),
            mass_inverse.ncols(),
            mass.nrows(),
            mass.ncols(),
            cochain.len()
        ));
    }
    let norm = cochain.norm();
    if !norm.is_finite() || norm <= 0.0 {
        return Err("consistency cochain must have positive finite norm".to_string());
    }
    let mass_cochain = mass * cochain;
    let recovered = mass_inverse * &mass_cochain;
    Ok((recovered - cochain).norm() / norm)
}

fn variance_stats_for_indices(
    values: &FeecVector,
    indices: &[usize],
) -> Result<CubeVarianceStats, String> {
    if indices.is_empty() {
        return Err("cannot summarize an empty variance vector".to_string());
    }

    let mut sorted = Vec::with_capacity(indices.len());
    for &index in indices {
        let value = values[index];
        if !value.is_finite() || value <= 0.0 {
            return Err("variance vector must contain finite positive entries".to_string());
        }
        sorted.push(value);
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite values should compare"));
    let midpoint = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        0.5 * (sorted[midpoint - 1] + sorted[midpoint])
    } else {
        sorted[midpoint]
    };
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;

    Ok(CubeVarianceStats {
        min: sorted[0],
        mean,
        median,
        max: *sorted.last().expect("non-empty sorted vector"),
    })
}

fn compare_variances_for_indices(
    variances: &FeecVector,
    consistent: &FeecVector,
    indices: &[usize],
) -> Result<CubeVarianceComparison, String> {
    if variances.len() != consistent.len() {
        return Err(format!(
            "variance dimensions differ: {} vs {}",
            variances.len(),
            consistent.len()
        ));
    }
    if indices.is_empty() {
        return Err("cannot compare an empty variance subset".to_string());
    }

    let mut log_sq = 0.0;
    let mut rel_sq = 0.0;
    let mut max_abs_log = 0.0_f64;
    let mut max_abs_rel = 0.0_f64;
    for &index in indices {
        let base = consistent[index];
        let value = variances[index];
        if !base.is_finite() || !value.is_finite() || base <= 0.0 || value <= 0.0 {
            return Err("variance comparisons require finite positive values".to_string());
        }
        let log_delta = (value / base).ln();
        let relative_delta = (value - base) / base;
        log_sq += log_delta * log_delta;
        rel_sq += relative_delta * relative_delta;
        max_abs_log = max_abs_log.max(log_delta.abs());
        max_abs_rel = max_abs_rel.max(relative_delta.abs());
    }

    let n = indices.len() as f64;
    Ok(CubeVarianceComparison {
        rms_log_delta_vs_consistent: (log_sq / n).sqrt(),
        max_abs_log_delta_vs_consistent: max_abs_log,
        rms_relative_delta_vs_consistent: (rel_sq / n).sqrt(),
        max_abs_relative_delta_vs_consistent: max_abs_rel,
    })
}

fn interior_edge_indices(edge_geometry: &[CubeEdgeGeometry]) -> Vec<usize> {
    edge_geometry
        .iter()
        .filter(|edge| !edge.is_boundary)
        .map(|edge| edge.edge_index)
        .collect()
}

fn summarize_interior_strategy_subset(
    variances: &FeecVector,
    consistent: &FeecVector,
    interior_indices: &[usize],
) -> Result<Vec<CubeVarianceSubsetSummary>, String> {
    if variances.len() != consistent.len() {
        return Err(format!(
            "variance dimensions differ: {} vs {}",
            variances.len(),
            consistent.len()
        ));
    }
    Ok(vec![CubeVarianceSubsetSummary {
        subset: CubeEdgeSubset::Interior,
        edge_count: interior_indices.len(),
        variance_stats: variance_stats_for_indices(variances, interior_indices)?,
        comparison_to_consistent: compare_variances_for_indices(
            variances,
            consistent,
            interior_indices,
        )?,
    }])
}

fn write_summary_csv(report: &CubeMassInverseVarianceReport, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,h,kappa,tau,drop_tolerance,vertex_count,edge_dofs,interior_edge_count,face_count,tetrahedron_count,strategy,kind,barycentric_stabilization_factor,is_oracle_calibrated,mass_inverse_nnz,mass_inverse_density,mass_inverse_lambda_min,mass_inverse_lambda_max,mass_inverse_condition_number,consistency_error,precision_nnz,precision_density,precision_lower_nnz,factor_nnz,factor_density,fill_in_ratio,mass_inverse_seconds,precision_seconds,factor_seconds,variance_seconds,total_seconds,interior_variance_min,interior_variance_mean,interior_variance_median,interior_variance_max,interior_rms_log_delta_vs_consistent,interior_max_abs_log_delta_vs_consistent,interior_rms_relative_delta_vs_consistent,interior_max_abs_relative_delta_vs_consistent"
    )?;

    for level in &report.levels {
        let interior_edge_count = level
            .edge_geometry
            .iter()
            .filter(|edge| !edge.is_boundary)
            .count();
        for strategy in &level.strategies {
            writeln!(
                writer,
                "{},{:.16},{:.16},{:.16},{:.16},{},{},{},{},{},{},{},{},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{},{:.16},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
                level.level,
                1.0 / level.level as f64,
                report.config.kappa,
                report.config.tau,
                report.config.drop_tolerance,
                level.vertex_count,
                level.edge_count,
                interior_edge_count,
                level.face_count,
                level.tetrahedron_count,
                strategy.label.as_str(),
                strategy.kind.label(),
                optional_f64_csv(strategy.barycentric_stabilization_factor),
                strategy.is_oracle_calibrated,
                strategy.mass_inverse_nnz,
                strategy.mass_inverse_density,
                strategy.mass_inverse_eigen.lambda_min,
                strategy.mass_inverse_eigen.lambda_max,
                strategy.mass_inverse_eigen.condition_number,
                strategy.consistency_error,
                strategy.precision_nnz,
                strategy.precision_density,
                strategy.precision_lower_nnz,
                strategy.factor_nnz,
                strategy.factor_density,
                strategy.fill_in_ratio,
                strategy.timings.mass_inverse_seconds,
                strategy.timings.precision_seconds,
                strategy.timings.factor_seconds,
                strategy.timings.variance_seconds,
                strategy.timings.total_seconds,
                strategy.variance_stats.min,
                strategy.variance_stats.mean,
                strategy.variance_stats.median,
                strategy.variance_stats.max,
                strategy.comparison_to_consistent.rms_log_delta_vs_consistent,
                strategy.comparison_to_consistent.max_abs_log_delta_vs_consistent,
                strategy
                    .comparison_to_consistent
                    .rms_relative_delta_vs_consistent,
                strategy
                    .comparison_to_consistent
                    .max_abs_relative_delta_vs_consistent,
            )?;
        }
    }

    Ok(())
}

fn write_fit_summary_csv(report: &CubeMassInverseVarianceReport, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "strategy,kind,is_oracle_calibrated,diagnostic,level_count,slope_log_value_vs_log_h,finest_value,status"
    )?;

    let mut series = BTreeMap::<String, Vec<(f64, &CubeMassInverseStrategyReport)>>::new();
    for level in &report.levels {
        let h = 1.0 / level.level as f64;
        for strategy in &level.strategies {
            series
                .entry(strategy.label.clone())
                .or_default()
                .push((h, strategy));
        }
    }

    for (label, mut rows) in series {
        rows.sort_by(|lhs, rhs| {
            lhs.0
                .partial_cmp(&rhs.0)
                .expect("finite mesh widths should compare")
        });
        let strategy = rows
            .last()
            .map(|(_, strategy)| *strategy)
            .expect("series entries are non-empty");
        for (diagnostic, values) in [
            (
                "consistency_error",
                rows.iter()
                    .map(|(h, strategy)| (*h, strategy.consistency_error))
                    .collect::<Vec<_>>(),
            ),
            (
                "interior_rms_log_delta_vs_consistent",
                rows.iter()
                    .map(|(h, strategy)| {
                        (
                            *h,
                            strategy
                                .comparison_to_consistent
                                .rms_log_delta_vs_consistent,
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            (
                "interior_rms_relative_delta_vs_consistent",
                rows.iter()
                    .map(|(h, strategy)| {
                        (
                            *h,
                            strategy
                                .comparison_to_consistent
                                .rms_relative_delta_vs_consistent,
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
        ] {
            let positive = values
                .iter()
                .copied()
                .filter(|(_, value)| value.is_finite() && *value > 0.0)
                .collect::<Vec<_>>();
            let (slope, status) = if positive.len() >= 2 {
                let xs = positive.iter().map(|(h, _)| h.ln()).collect::<Vec<_>>();
                let ys = positive
                    .iter()
                    .map(|(_, value)| value.ln())
                    .collect::<Vec<_>>();
                (linear_slope(&xs, &ys), "fit")
            } else {
                (f64::NAN, "insufficient_positive_values")
            };
            let finest_value = rows
                .first()
                .map(|(_, strategy)| match diagnostic {
                    "consistency_error" => strategy.consistency_error,
                    "interior_rms_log_delta_vs_consistent" => {
                        strategy
                            .comparison_to_consistent
                            .rms_log_delta_vs_consistent
                    }
                    "interior_rms_relative_delta_vs_consistent" => {
                        strategy
                            .comparison_to_consistent
                            .rms_relative_delta_vs_consistent
                    }
                    _ => f64::NAN,
                })
                .unwrap_or(f64::NAN);
            writeln!(
                writer,
                "{},{},{},{},{},{:.16},{:.16},{}",
                label,
                strategy.kind.label(),
                strategy.is_oracle_calibrated,
                diagnostic,
                positive.len(),
                slope,
                finest_value,
                status
            )?;
        }
    }

    Ok(())
}

fn optional_f64_csv(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.16}"))
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
    if denominator <= 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}

fn write_interior_edge_variance_csv(
    level: &CubeMassInverseLevelReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    write!(
        writer,
        "edge_index,vertex_i,vertex_j,midpoint_x,midpoint_y,midpoint_z,edge_length,is_boundary_edge"
    )?;
    for strategy in &level.strategies {
        let label = &strategy.label;
        write!(
            writer,
            ",{label}_variance,{label}_variance_per_length2,{label}_log_delta_vs_consistent,{label}_relative_delta_vs_consistent"
        )?;
    }
    writeln!(writer)?;

    let consistent = level
        .strategies
        .iter()
        .find(|strategy| strategy.kind == CubeMassInverseKind::ExactConsistentInverse)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing consistent strategy"))?;

    for edge in level.edge_geometry.iter().filter(|edge| !edge.is_boundary) {
        write!(
            writer,
            "{},{},{},{:.16},{:.16},{:.16},{:.16},{}",
            edge.edge_index,
            edge.vertices[0],
            edge.vertices[1],
            edge.midpoint[0],
            edge.midpoint[1],
            edge.midpoint[2],
            edge.length,
            edge.is_boundary,
        )?;
        let base = consistent.variances[edge.edge_index];
        for strategy in &level.strategies {
            let variance = strategy.variances[edge.edge_index];
            let variance_per_length2 = variance / (edge.length * edge.length);
            let log_delta = (variance / base).ln();
            let relative_delta = (variance - base) / base;
            write!(
                writer,
                ",{:.16},{:.16},{:.16},{:.16}",
                variance, variance_per_length2, log_delta, relative_delta
            )?;
        }
        writeln!(writer)?;
    }

    Ok(())
}
