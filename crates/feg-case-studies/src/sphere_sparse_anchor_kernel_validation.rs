//! Sphere validation for the sparse-anchor decomposed Hodge-Matern prior.
//!
//! This compares FEEC/GMRF pushforward covariances against the intrinsic spherical
//! Hodge kernels implemented by Robert-Nicoud, Krause, and Borovitskiy. The
//! analytic side uses the same vector spherical harmonic addition-theorem form:
//! exact modes are \(\nabla Y_{\ell m}/\sqrt{\lambda_\ell}\), coexact modes are
//! \(\star\nabla Y_{\ell m}/\sqrt{\lambda_\ell}\), and
//! \(\lambda_\ell=\ell(\ell+1)\).

use common::linalg::nalgebra::Vector as FeecVector;
use faer::Mat;
use feg_core::HodgeBranchKind;
use feg_gp::{
    HodgeBranchConfig as SpectralHodgeBranchConfig, HodgeBuildOptions, HodgeCompositionalConfig,
    HodgeCompositionalGp, HodgeDecomposedBasis,
};
use feg_infer::{
    prior::{
        matern::{one_form::build_reconstructed_barycenter_field_operator, MaternAlpha},
        sparse_anchor_hodge::{
            build_sparse_anchor_hodge_1form_prior_with_coords, SparseAnchorBranchConfig,
            SparseAnchorHodge1FormPriorConfig,
        },
    },
    sparse::{feec_csr_to_gmrf, sparse_row_operator_from_feec_csr},
};
use gmrf_core::{
    types::{DenseMatrix, Vector as GmrfVector},
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

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct SphereSparseAnchorKernelValidationConfig {
    pub refinement_levels: Vec<usize>,
    pub kappa: f64,
    pub tau: f64,
    pub alpha: MaternAlpha,
    pub analytic_lmax: usize,
    pub spectral_lmax: usize,
    pub max_cells: usize,
}

impl Default for SphereSparseAnchorKernelValidationConfig {
    fn default() -> Self {
        Self {
            refinement_levels: vec![1, 2],
            kappa: 1.0,
            tau: 1.0,
            alpha: MaternAlpha::Two,
            analytic_lmax: 35,
            spectral_lmax: 4,
            max_cells: 24,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SphereKernelValidationMethod {
    SparseAnchor,
    Spectral,
}

impl SphereKernelValidationMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SparseAnchor => "sparse_anchor",
            Self::Spectral => "spectral",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SphereKernelValidationTarget {
    AnalyticFull,
    AnalyticTruncated,
}

impl SphereKernelValidationTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AnalyticFull => "analytic_full",
            Self::AnalyticTruncated => "analytic_truncated",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SphereSparseAnchorKernelBranchReport {
    pub method: SphereKernelValidationMethod,
    pub target: SphereKernelValidationTarget,
    pub branch: Option<HodgeBranchKind>,
    pub covariance_dimension: usize,
    pub relative_frobenius_error: f64,
    pub diagonal_relative_l2_error: f64,
    pub best_scalar_rescaled_relative_frobenius_error: f64,
    pub best_scalar: f64,
    pub model_trace: f64,
    pub analytic_trace: f64,
}

#[derive(Debug, Clone)]
pub struct SphereSparseAnchorKernelLevelReport {
    pub refinement_level: usize,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub cell_count: usize,
    pub selected_cell_count: usize,
    pub exact: SphereSparseAnchorKernelBranchReport,
    pub coexact: SphereSparseAnchorKernelBranchReport,
    pub joint: SphereSparseAnchorKernelBranchReport,
    pub spectral_exact: SphereSparseAnchorKernelBranchReport,
    pub spectral_coexact: SphereSparseAnchorKernelBranchReport,
    pub spectral_joint: SphereSparseAnchorKernelBranchReport,
}

impl SphereSparseAnchorKernelLevelReport {
    pub fn branch_reports(&self) -> [&SphereSparseAnchorKernelBranchReport; 6] {
        [
            &self.exact,
            &self.coexact,
            &self.joint,
            &self.spectral_exact,
            &self.spectral_coexact,
            &self.spectral_joint,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct SphereSparseAnchorKernelValidationReport {
    pub levels: Vec<SphereSparseAnchorKernelLevelReport>,
}

pub fn compute_sphere_sparse_anchor_kernel_validation(
    config: SphereSparseAnchorKernelValidationConfig,
) -> Result<SphereSparseAnchorKernelValidationReport, String> {
    validate_config(&config)?;
    let mut levels = Vec::with_capacity(config.refinement_levels.len());
    for refinement_level in &config.refinement_levels {
        levels.push(compute_level_report(*refinement_level, &config)?);
    }
    Ok(SphereSparseAnchorKernelValidationReport { levels })
}

fn compute_level_report(
    refinement_level: usize,
    config: &SphereSparseAnchorKernelValidationConfig,
) -> Result<SphereSparseAnchorKernelLevelReport, String> {
    let surface = mesh_sphere_surface(refinement_level);
    let (topology, coords) = surface.into_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let selected_cells = evenly_spaced_indices(topology.cells().len(), config.max_cells);
    let points = normalized_cell_barycenters(&topology, &coords, &selected_cells)?;
    let observation_operator =
        stacked_barycenter_reconstruction_operator(&topology, &coords, &selected_cells)?;

    let exact = compute_branch_report(
        HodgeBranchKind::Exact,
        &topology,
        &coords,
        &metric,
        &observation_operator,
        &points,
        config,
    )?;
    let coexact = compute_branch_report(
        HodgeBranchKind::Coexact,
        &topology,
        &coords,
        &metric,
        &observation_operator,
        &points,
        config,
    )?;
    let joint = compute_joint_report(
        &topology,
        &coords,
        &metric,
        &observation_operator,
        &points,
        config,
    )?;
    let (spectral_exact, spectral_coexact, spectral_joint) =
        compute_spectral_reports(&topology, &metric, &observation_operator, &points, config)?;

    Ok(SphereSparseAnchorKernelLevelReport {
        refinement_level,
        vertex_count: topology.vertices().len(),
        edge_count: topology.edges().len(),
        cell_count: topology.cells().len(),
        selected_cell_count: selected_cells.len(),
        exact,
        coexact,
        joint,
        spectral_exact,
        spectral_coexact,
        spectral_joint,
    })
}

fn compute_branch_report(
    branch: HodgeBranchKind,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    observation_operator: &SparseRowOperator,
    points: &[[f64; 3]],
    config: &SphereSparseAnchorKernelValidationConfig,
) -> Result<SphereSparseAnchorKernelBranchReport, String> {
    let prior = build_sparse_anchor_hodge_1form_prior_with_coords(
        topology,
        coords,
        metric,
        prior_config(config, [branch]),
    )?;
    let branch_prior = prior.branch(branch).ok_or_else(|| {
        format!(
            "sparse-anchor prior did not contain {} branch",
            branch.as_str()
        )
    })?;
    let transform_operator = sparse_row_operator_from_feec_csr(&branch_prior.transform)?;
    let latent_operator = SparseRowOperator::compose(observation_operator, &transform_operator)
        .map_err(|err| err.to_string())?;
    let gmrf_covariance = transformed_covariance(&branch_prior.precision, &latent_operator)?;
    let analytic_covariance = analytic_branch_covariance(
        points,
        branch,
        config.kappa,
        config.tau,
        config.alpha,
        config.analytic_lmax,
    )?;
    compare_covariances(
        SphereKernelValidationMethod::SparseAnchor,
        SphereKernelValidationTarget::AnalyticFull,
        Some(branch),
        &gmrf_covariance,
        &analytic_covariance,
    )
}

fn compute_joint_report(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    observation_operator: &SparseRowOperator,
    points: &[[f64; 3]],
    config: &SphereSparseAnchorKernelValidationConfig,
) -> Result<SphereSparseAnchorKernelBranchReport, String> {
    let prior = build_sparse_anchor_hodge_1form_prior_with_coords(
        topology,
        coords,
        metric,
        prior_config(config, [HodgeBranchKind::Exact, HodgeBranchKind::Coexact]),
    )?;
    let transform_operator = sparse_row_operator_from_feec_csr(&prior.latent_to_ambient)?;
    let latent_operator = SparseRowOperator::compose(observation_operator, &transform_operator)
        .map_err(|err| err.to_string())?;
    let gmrf_covariance = transformed_covariance(&prior.precision, &latent_operator)?;
    let exact = analytic_branch_covariance(
        points,
        HodgeBranchKind::Exact,
        config.kappa,
        config.tau,
        config.alpha,
        config.analytic_lmax,
    )?;
    let coexact = analytic_branch_covariance(
        points,
        HodgeBranchKind::Coexact,
        config.kappa,
        config.tau,
        config.alpha,
        config.analytic_lmax,
    )?;
    let analytic_covariance = add_dense(&exact, &coexact)?;
    compare_covariances(
        SphereKernelValidationMethod::SparseAnchor,
        SphereKernelValidationTarget::AnalyticFull,
        None,
        &gmrf_covariance,
        &analytic_covariance,
    )
}

fn compute_spectral_reports(
    topology: &Complex,
    metric: &MeshLengths,
    observation_operator: &SparseRowOperator,
    points: &[[f64; 3]],
    config: &SphereSparseAnchorKernelValidationConfig,
) -> Result<
    (
        SphereSparseAnchorKernelBranchReport,
        SphereSparseAnchorKernelBranchReport,
        SphereSparseAnchorKernelBranchReport,
    ),
    String,
> {
    let mode_count = spherical_branch_mode_count(config.spectral_lmax);
    let build_options = HodgeBuildOptions {
        harmonic_dim: 0,
        exact_mode_count: mode_count,
        coexact_mode_count: mode_count,
        ..HodgeBuildOptions::new(0)
    };
    let basis = HodgeDecomposedBasis::build(topology, metric, 1, build_options)
        .map_err(|err| err.to_string())?;
    let branch_config = SpectralHodgeBranchConfig {
        kappa: config.kappa,
        tau: config.tau,
        mode_count,
        ..SpectralHodgeBranchConfig::default()
    };
    let gp = HodgeCompositionalGp::from_hodge_decomposition(
        basis,
        HodgeCompositionalConfig {
            alpha: config.alpha.as_u32() as f64,
            exact: branch_config,
            coexact: branch_config,
            harmonic: SpectralHodgeBranchConfig {
                mode_count: 0,
                ..branch_config
            },
        },
    )
    .map_err(|err| err.to_string())?;

    let exact = compute_spectral_branch_report(
        &gp,
        HodgeBranchKind::Exact,
        observation_operator,
        points,
        config,
    )?;
    let coexact = compute_spectral_branch_report(
        &gp,
        HodgeBranchKind::Coexact,
        observation_operator,
        points,
        config,
    )?;
    let joint_features =
        apply_operator_to_feature_matrix(observation_operator, gp.combined_feature_matrix())?;
    let joint_covariance = covariance_from_feature_matrix(&joint_features);
    let analytic_exact = analytic_branch_covariance(
        points,
        HodgeBranchKind::Exact,
        config.kappa,
        config.tau,
        config.alpha,
        config.spectral_lmax,
    )?;
    let analytic_coexact = analytic_branch_covariance(
        points,
        HodgeBranchKind::Coexact,
        config.kappa,
        config.tau,
        config.alpha,
        config.spectral_lmax,
    )?;
    let analytic_joint = add_dense(&analytic_exact, &analytic_coexact)?;
    let joint = compare_covariances(
        SphereKernelValidationMethod::Spectral,
        SphereKernelValidationTarget::AnalyticTruncated,
        None,
        &joint_covariance,
        &analytic_joint,
    )?;
    Ok((exact, coexact, joint))
}

fn compute_spectral_branch_report(
    gp: &HodgeCompositionalGp,
    branch: HodgeBranchKind,
    observation_operator: &SparseRowOperator,
    points: &[[f64; 3]],
    config: &SphereSparseAnchorKernelValidationConfig,
) -> Result<SphereSparseAnchorKernelBranchReport, String> {
    let output_features =
        apply_operator_to_feature_matrix(observation_operator, gp.branch_feature_matrix(branch))?;
    let spectral_covariance = covariance_from_feature_matrix(&output_features);
    let analytic_covariance = analytic_branch_covariance(
        points,
        branch,
        config.kappa,
        config.tau,
        config.alpha,
        config.spectral_lmax,
    )?;
    compare_covariances(
        SphereKernelValidationMethod::Spectral,
        SphereKernelValidationTarget::AnalyticTruncated,
        Some(branch),
        &spectral_covariance,
        &analytic_covariance,
    )
}

fn transformed_covariance(
    precision: &common::linalg::nalgebra::CsrMatrix<f64>,
    operator: &SparseRowOperator,
) -> Result<DenseMatrix, String> {
    let mut gmrf = Gmrf::from_mean_and_precision(
        GmrfVector::zeros(precision.nrows()),
        feec_csr_to_gmrf(precision),
    )
    .map_err(|err| err.to_string())?;
    gmrf.exact_transformed_covariance(operator)
        .map_err(|err| err.to_string())
}

fn apply_operator_to_feature_matrix(
    operator: &SparseRowOperator,
    features: &Mat<f64>,
) -> Result<Mat<f64>, String> {
    if operator.ncols != features.nrows() {
        return Err(format!(
            "feature matrix row count {} must match operator column count {}",
            features.nrows(),
            operator.ncols
        ));
    }
    let mut output = Mat::zeros(operator.nrows(), features.ncols());
    for (out_row, row) in operator.rows.iter().enumerate() {
        for &(feature_row, weight) in row {
            for feature_col in 0..features.ncols() {
                output[(out_row, feature_col)] += weight * features[(feature_row, feature_col)];
            }
        }
    }
    Ok(output)
}

fn covariance_from_feature_matrix(features: &Mat<f64>) -> DenseMatrix {
    DenseMatrix::from_fn(features.nrows(), features.nrows(), |i, j| {
        (0..features.ncols())
            .map(|col| features[(i, col)] * features[(j, col)])
            .sum()
    })
}

fn prior_config(
    config: &SphereSparseAnchorKernelValidationConfig,
    branches: impl IntoIterator<Item = HodgeBranchKind>,
) -> SparseAnchorHodge1FormPriorConfig {
    SparseAnchorHodge1FormPriorConfig {
        branches: branches.into_iter().collect(),
        exact: branch_config(config),
        coexact: branch_config(config),
        harmonic_precision: 1.0,
        harmonic_dim: Some(0),
        ..SparseAnchorHodge1FormPriorConfig::default()
    }
}

fn branch_config(config: &SphereSparseAnchorKernelValidationConfig) -> SparseAnchorBranchConfig {
    SparseAnchorBranchConfig {
        kappa: config.kappa,
        tau: config.tau,
        alpha: config.alpha,
    }
}

fn spherical_branch_mode_count(lmax: usize) -> usize {
    lmax * (lmax + 2)
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

pub(crate) fn analytic_branch_covariance(
    points: &[[f64; 3]],
    branch: HodgeBranchKind,
    kappa: f64,
    tau: f64,
    alpha: MaternAlpha,
    lmax: usize,
) -> Result<DenseMatrix, String> {
    if branch == HodgeBranchKind::Harmonic {
        return Err("S^2 has no harmonic 1-form branch in this validation".to_string());
    }
    let point_count = points.len();
    let dimension = 3 * point_count;
    Ok(DenseMatrix::from_fn(dimension, dimension, |row, col| {
        let row_component = row / point_count;
        let row_point = row % point_count;
        let col_component = col / point_count;
        let col_point = col % point_count;
        let kernel = analytic_pair_kernel(
            points[row_point],
            points[col_point],
            branch,
            kappa,
            tau,
            alpha,
            lmax,
        );
        kernel[row_component][col_component]
    }))
}

#[cfg(feature = "experimental")]
pub(crate) fn analytic_joint_one_form_covariance(
    points: &[[f64; 3]],
    kappa: f64,
    tau: f64,
    alpha: MaternAlpha,
    lmax: usize,
) -> Result<DenseMatrix, String> {
    let exact =
        analytic_branch_covariance(points, HodgeBranchKind::Exact, kappa, tau, alpha, lmax)?;
    let coexact =
        analytic_branch_covariance(points, HodgeBranchKind::Coexact, kappa, tau, alpha, lmax)?;
    add_dense(&exact, &coexact)
}

fn analytic_pair_kernel(
    x: [f64; 3],
    y: [f64; 3],
    branch: HodgeBranchKind,
    kappa: f64,
    tau: f64,
    alpha: MaternAlpha,
    lmax: usize,
) -> [[f64; 3]; 3] {
    let t = dot3(x, y).clamp(-1.0, 1.0);
    let legendre = legendre_with_derivatives(t, lmax);
    let mut kernel = [[0.0; 3]; 3];
    for (ell, &(_, d_p, dd_p)) in legendre.iter().enumerate().take(lmax + 1).skip(1) {
        let lambda = ell as f64 * (ell as f64 + 1.0);
        let multiplier = tau.powi(-2) * (kappa * kappa + lambda).powi(-(alpha.as_u32() as i32));
        let coefficient = multiplier * (2 * ell + 1) as f64 / (FOUR_PI * lambda);
        let hessian = add3(outer3(y, x, dd_p), scale3(identity3(), d_p));
        let contribution = match branch {
            HodgeBranchKind::Exact => {
                matmul3(matmul3(tangent_projector(x), hessian), tangent_projector(y))
            }
            HodgeBranchKind::Coexact => {
                let star_x = cross_matrix(x);
                let star_y = cross_matrix(y);
                matmul3(matmul3(star_x, hessian), transpose3(star_y))
            }
            HodgeBranchKind::Harmonic => [[0.0; 3]; 3],
        };
        kernel = add3(kernel, scale3(contribution, coefficient));
    }
    kernel
}

fn legendre_with_derivatives(t: f64, lmax: usize) -> Vec<(f64, f64, f64)> {
    let mut values = vec![(0.0, 0.0, 0.0); lmax + 1];
    values[0] = (1.0, 0.0, 0.0);
    if lmax >= 1 {
        values[1] = (t, 1.0, 0.0);
    }
    for ell in 2..=lmax {
        let ell_f = ell as f64;
        let a = (2 * ell - 1) as f64;
        let b = (ell - 1) as f64;
        let (p_prev, dp_prev, ddp_prev) = values[ell - 1];
        let (p_prev2, dp_prev2, ddp_prev2) = values[ell - 2];
        values[ell] = (
            (a * t * p_prev - b * p_prev2) / ell_f,
            (a * (p_prev + t * dp_prev) - b * dp_prev2) / ell_f,
            (a * (2.0 * dp_prev + t * ddp_prev) - b * ddp_prev2) / ell_f,
        );
    }
    values
}

fn compare_covariances(
    method: SphereKernelValidationMethod,
    target: SphereKernelValidationTarget,
    branch: Option<HodgeBranchKind>,
    model: &DenseMatrix,
    analytic: &DenseMatrix,
) -> Result<SphereSparseAnchorKernelBranchReport, String> {
    if model.nrows() != analytic.nrows() || model.ncols() != analytic.ncols() {
        return Err(format!(
            "covariance shape mismatch: model {}x{}, analytic {}x{}",
            model.nrows(),
            model.ncols(),
            analytic.nrows(),
            analytic.ncols()
        ));
    }
    let analytic_norm = frobenius_norm(analytic).max(f64::EPSILON);
    let diff = subtract_dense(model, analytic)?;
    let best_scalar = best_scalar_fit(model, analytic);
    let scaled_diff = subtract_dense(&scale_dense(model, best_scalar), analytic)?;
    let model_diag = diagonal(model);
    let analytic_diag = diagonal(analytic);
    let diag_diff = FeecVector::from_iterator(
        model_diag.len(),
        model_diag
            .iter()
            .zip(analytic_diag.iter())
            .map(|(a, b)| a - b),
    );
    let analytic_diag_norm = analytic_diag.norm().max(f64::EPSILON);
    Ok(SphereSparseAnchorKernelBranchReport {
        method,
        target,
        branch,
        covariance_dimension: model.nrows(),
        relative_frobenius_error: frobenius_norm(&diff) / analytic_norm,
        diagonal_relative_l2_error: diag_diff.norm() / analytic_diag_norm,
        best_scalar_rescaled_relative_frobenius_error: frobenius_norm(&scaled_diff) / analytic_norm,
        best_scalar,
        model_trace: model_diag.iter().sum(),
        analytic_trace: analytic_diag.iter().sum(),
    })
}

fn validate_config(config: &SphereSparseAnchorKernelValidationConfig) -> Result<(), String> {
    if config.refinement_levels.is_empty() {
        return Err("at least one sphere refinement level is required".to_string());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if config.alpha == MaternAlpha::Three {
        return Err("sparse-anchor validation currently supports alpha=1 or alpha=2".to_string());
    }
    if config.analytic_lmax == 0 {
        return Err("analytic_lmax must be positive".to_string());
    }
    if config.spectral_lmax == 0 {
        return Err("spectral_lmax must be positive".to_string());
    }
    if config.max_cells == 0 {
        return Err("max_cells must be positive".to_string());
    }
    Ok(())
}

fn evenly_spaced_indices(len: usize, max_count: usize) -> Vec<usize> {
    if len <= max_count {
        return (0..len).collect();
    }
    (0..max_count).map(|i| i * len / max_count).collect()
}

fn diagonal(matrix: &DenseMatrix) -> FeecVector {
    FeecVector::from_iterator(
        matrix.nrows().min(matrix.ncols()),
        (0..matrix.nrows()).map(|i| matrix[(i, i)]),
    )
}

fn frobenius_norm(matrix: &DenseMatrix) -> f64 {
    let mut sum = 0.0;
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            sum += matrix[(row, col)] * matrix[(row, col)];
        }
    }
    sum.sqrt()
}

fn best_scalar_fit(source: &DenseMatrix, target: &DenseMatrix) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in 0..source.nrows() {
        for col in 0..source.ncols() {
            numerator += source[(row, col)] * target[(row, col)];
            denominator += source[(row, col)] * source[(row, col)];
        }
    }
    numerator / denominator.max(f64::EPSILON)
}

fn add_dense(lhs: &DenseMatrix, rhs: &DenseMatrix) -> Result<DenseMatrix, String> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err("dense matrix shapes must match".to_string());
    }
    Ok(DenseMatrix::from_fn(lhs.nrows(), lhs.ncols(), |i, j| {
        lhs[(i, j)] + rhs[(i, j)]
    }))
}

fn subtract_dense(lhs: &DenseMatrix, rhs: &DenseMatrix) -> Result<DenseMatrix, String> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err("dense matrix shapes must match".to_string());
    }
    Ok(DenseMatrix::from_fn(lhs.nrows(), lhs.ncols(), |i, j| {
        lhs[(i, j)] - rhs[(i, j)]
    }))
}

fn scale_dense(matrix: &DenseMatrix, scale: f64) -> DenseMatrix {
    DenseMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        scale * matrix[(i, j)]
    })
}

fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn tangent_projector(x: [f64; 3]) -> [[f64; 3]; 3] {
    subtract3(identity3(), outer3(x, x, 1.0))
}

fn cross_matrix(x: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]]
}

fn matmul3(lhs: [[f64; 3]; 3], rhs: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| lhs[i][k] * rhs[k][j]).sum();
        }
    }
    out
}

fn transpose3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = matrix[j][i];
        }
    }
    out
}

fn add3(lhs: [[f64; 3]; 3], rhs: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = lhs[i][j] + rhs[i][j];
        }
    }
    out
}

fn subtract3(lhs: [[f64; 3]; 3], rhs: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = lhs[i][j] - rhs[i][j];
        }
    }
    out
}

fn scale3(matrix: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = scale * matrix[i][j];
        }
    }
    out
}

fn outer3(lhs: [f64; 3], rhs: [f64; 3], scale: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = scale * lhs[i] * rhs[j];
        }
    }
    out
}

fn dot3(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn normalize3(x: [f64; 3]) -> Result<[f64; 3], String> {
    let norm = dot3(x, x).sqrt();
    if norm <= f64::EPSILON {
        return Err("cannot normalize zero barycenter".to_string());
    }
    Ok([x[0] / norm, x[1] / norm, x[2] / norm])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_exact_and_coexact_kernels_are_tangent_on_the_sphere() {
        let x = normalize3([0.2, -0.3, 0.7]).unwrap();
        let y = normalize3([-0.5, 0.4, 0.1]).unwrap();
        for branch in [HodgeBranchKind::Exact, HodgeBranchKind::Coexact] {
            let kernel = analytic_pair_kernel(x, y, branch, 1.0, 1.0, MaternAlpha::Two, 12);
            for (col, _) in kernel[0].iter().enumerate() {
                let column = [kernel[0][col], kernel[1][col], kernel[2][col]];
                assert!(dot3(x, column).abs() <= 1e-10);
            }
            for row in kernel {
                assert!(dot3(y, row).abs() <= 1e-10);
            }
        }
    }

    #[cfg(feature = "external-reference-tests")]
    #[test]
    fn sparse_anchor_sphere_kernel_validation_converges_to_analytic_kernel() {
        let report = compute_sphere_sparse_anchor_kernel_validation(
            SphereSparseAnchorKernelValidationConfig {
                refinement_levels: vec![1, 2, 3],
                analytic_lmax: 20,
                max_cells: 8,
                ..SphereSparseAnchorKernelValidationConfig::default()
            },
        )
        .expect("sphere sparse-anchor kernel validation should run");
        for level in &report.levels {
            eprintln!(
                "sphere sparse-anchor level {} errors: exact {:.3e} (scaled {:.3e}, scale {:.3e}), coexact {:.3e} (scaled {:.3e}, scale {:.3e}), joint {:.3e} (scaled {:.3e}, scale {:.3e})",
                level.refinement_level,
                level.exact.relative_frobenius_error,
                level.exact.best_scalar_rescaled_relative_frobenius_error,
                level.exact.best_scalar,
                level.coexact.relative_frobenius_error,
                level.coexact.best_scalar_rescaled_relative_frobenius_error,
                level.coexact.best_scalar,
                level.joint.relative_frobenius_error,
                level.joint.best_scalar_rescaled_relative_frobenius_error,
                level.joint.best_scalar
            );
            for branch in level.branch_reports() {
                assert!(branch.relative_frobenius_error.is_finite());
                assert!(branch.diagonal_relative_l2_error.is_finite());
                assert!(branch
                    .best_scalar_rescaled_relative_frobenius_error
                    .is_finite());
                assert!(branch.model_trace.is_finite());
                assert!(branch.analytic_trace.is_finite());
            }
        }
        let coarse = &report.levels[0];
        let middle = &report.levels[1];
        let fine = &report.levels[2];
        assert!(middle.exact.relative_frobenius_error < coarse.exact.relative_frobenius_error);
        assert!(fine.exact.relative_frobenius_error < middle.exact.relative_frobenius_error);
        assert!(middle.coexact.relative_frobenius_error < coarse.coexact.relative_frobenius_error);
        assert!(fine.coexact.relative_frobenius_error < middle.coexact.relative_frobenius_error);
        assert!(middle.joint.relative_frobenius_error < coarse.joint.relative_frobenius_error);
        assert!(fine.joint.relative_frobenius_error < middle.joint.relative_frobenius_error);
        assert!(fine.exact.relative_frobenius_error <= 5e-2);
        assert!(fine.coexact.relative_frobenius_error <= 3e-2);
        assert!(fine.joint.relative_frobenius_error <= 4e-2);
        assert!(fine.exact.diagonal_relative_l2_error <= 6e-2);
        assert!(fine.coexact.diagonal_relative_l2_error <= 2e-2);
        assert!(fine.joint.diagonal_relative_l2_error <= 4e-2);
    }

    #[cfg(feature = "external-reference-tests")]
    #[test]
    fn spectral_hodge_kernel_validation_converges_to_matched_truncation() {
        let report = compute_sphere_sparse_anchor_kernel_validation(
            SphereSparseAnchorKernelValidationConfig {
                refinement_levels: vec![1, 2, 3],
                analytic_lmax: 20,
                spectral_lmax: 4,
                max_cells: 8,
                ..SphereSparseAnchorKernelValidationConfig::default()
            },
        )
        .expect("sphere spectral Hodge kernel validation should run");
        for level in &report.levels {
            eprintln!(
                "sphere spectral level {} errors: exact {:.3e}, coexact {:.3e}, joint {:.3e}",
                level.refinement_level,
                level.spectral_exact.relative_frobenius_error,
                level.spectral_coexact.relative_frobenius_error,
                level.spectral_joint.relative_frobenius_error,
            );
            for branch in [
                &level.spectral_exact,
                &level.spectral_coexact,
                &level.spectral_joint,
            ] {
                assert_eq!(branch.method, SphereKernelValidationMethod::Spectral);
                assert_eq!(
                    branch.target,
                    SphereKernelValidationTarget::AnalyticTruncated
                );
                assert!(branch.relative_frobenius_error.is_finite());
                assert!(branch.diagonal_relative_l2_error.is_finite());
                assert!(branch.model_trace.is_finite());
                assert!(branch.analytic_trace.is_finite());
            }
        }
        let coarse = &report.levels[0];
        let middle = &report.levels[1];
        let fine = &report.levels[2];
        assert!(
            middle.spectral_exact.relative_frobenius_error
                < coarse.spectral_exact.relative_frobenius_error
        );
        assert!(
            fine.spectral_exact.relative_frobenius_error
                < middle.spectral_exact.relative_frobenius_error
        );
        assert!(
            middle.spectral_coexact.relative_frobenius_error
                < coarse.spectral_coexact.relative_frobenius_error
        );
        assert!(
            fine.spectral_coexact.relative_frobenius_error
                < middle.spectral_coexact.relative_frobenius_error
        );
        assert!(
            middle.spectral_joint.relative_frobenius_error
                < coarse.spectral_joint.relative_frobenius_error
        );
        assert!(
            fine.spectral_joint.relative_frobenius_error
                < middle.spectral_joint.relative_frobenius_error
        );
        assert!(fine.spectral_exact.relative_frobenius_error <= 6e-2);
        assert!(fine.spectral_coexact.relative_frobenius_error <= 6e-2);
        assert!(fine.spectral_joint.relative_frobenius_error <= 6e-2);
        assert!(fine.spectral_exact.diagonal_relative_l2_error <= 8e-2);
        assert!(fine.spectral_coexact.diagonal_relative_l2_error <= 8e-2);
        assert!(fine.spectral_joint.diagonal_relative_l2_error <= 8e-2);
    }
}
