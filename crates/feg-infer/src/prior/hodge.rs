use crate::conditioning::linear::{
    DerivedOperator, DerivedOperatorSet, DerivedVarianceMode, HutchinsonConfig,
    LinearGaussianConditioningProblem, LinearGaussianConditioningResult,
};
use crate::linear_pde::{
    solve_linear_pde_uq_with_config, LinearPdeDerivedMarginalResult, LinearPdeDerivedQuantitySpec,
    LinearPdeUqProblem, LinearPdeUqResult, LinearPdeUqSolverConfig,
};
use crate::prior::matern::one_form::{
    build_hodge_laplacian_1form, build_matern_mass_inverse_1form,
    build_matern_mass_inverse_1form_with_coords, build_matern_precision_1form,
    build_matern_precision_1form_with_coords, HodgeLaplacian1Form,
    MaternConfig as Matern1FormConfig, MaternMassInverse as Matern1FormMassInverse,
};
use crate::prior::matern::two_form::{
    build_hodge_laplacian_2form, build_hodge_laplacian_2form_with_lower_mass_inverse_coords,
    build_matern_precision_2form, build_matern_precision_2form_with_coords,
    MaternConfig as Matern2FormConfig, MaternMassInverse as Matern2FormMassInverse,
};
use crate::prior::matern::zero_form::{
    build_laplace_beltrami_0form, build_matern_precision_0form, MaternConfig as Matern0FormConfig,
    MaternMassInverse as Matern0FormMassInverse,
};
use crate::sparse::{
    block_diag_feec_csr, core_triplet_to_feec_csr, dense_to_feec_csr, feec_csr_to_core_triplet,
    feec_csr_to_dense, feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec, hstack_feec_csr,
    identity_feec_csr, restrict_columns_and_fold_fixed, restrict_rows_with_layout,
    sparse_row_operator_from_feec_csr,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::ManifoldComplexExt;
use feg_core::{GaussianPriorSpec, HodgeBranchKind, LinearGaussianMeasurementSpec};
use formoniq::problems::hodge_laplace::{solve_hodge_laplace_harmonics_with_galmats, MixedGalmats};
use formoniq::problems::reduced_linear::ReducedLinearPdeAssembly;
use formoniq::reduction::DofLayout;
use gmrf_core::{types::DenseMatrix as GmrfDenseMatrix, SparseRowOperator};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

pub const HODGE_AMBIENT_DERIVED_NAME: &str = "hodge_ambient";
const DEFAULT_HUTCHINSON_PROBES: usize = 64;
const DEFAULT_HUTCHINSON_BATCHES: usize = 4;
const DEFAULT_PROJECTION_RIDGE: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct Hodge1FormPriorConfig {
    pub kappa: f64,
    pub tau: f64,
    pub branches: Vec<HodgeBranchKind>,
    pub harmonic_dim: usize,
    pub harmonic_basis_override: Option<FeecMatrix>,
    pub zero_form_mass_inverse: Matern0FormMassInverse,
    pub one_form_mass_inverse: Matern1FormMassInverse,
    pub two_form_mass_inverse: Matern2FormMassInverse,
}

impl Default for Hodge1FormPriorConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            tau: 1.0,
            branches: HodgeBranchKind::ALL.to_vec(),
            harmonic_dim: 0,
            harmonic_basis_override: None,
            zero_form_mass_inverse: Matern0FormMassInverse::default(),
            one_form_mass_inverse: Matern1FormMassInverse::default(),
            two_form_mass_inverse: Matern2FormMassInverse::default(),
        }
    }
}

impl Hodge1FormPriorConfig {
    pub fn all(kappa: f64, tau: f64, harmonic_dim: usize) -> Self {
        Self {
            kappa,
            tau,
            harmonic_dim,
            ..Self::default()
        }
    }

    pub fn selected(
        kappa: f64,
        tau: f64,
        branches: impl IntoIterator<Item = HodgeBranchKind>,
        harmonic_dim: usize,
    ) -> Self {
        Self {
            kappa,
            tau,
            branches: branches.into_iter().collect(),
            harmonic_dim,
            ..Self::default()
        }
    }

    pub fn branch(kappa: f64, tau: f64, branch: HodgeBranchKind, harmonic_dim: usize) -> Self {
        Self::selected(kappa, tau, [branch], harmonic_dim)
    }
}

#[derive(Debug, Clone)]
pub struct Hodge1FormBranchPrior {
    pub kind: HodgeBranchKind,
    pub offset: usize,
    pub latent_dimension: usize,
    pub ambient_dimension: usize,
    pub precision: FeecCsr,
    pub transform: FeecCsr,
}

impl Hodge1FormBranchPrior {
    pub fn latent_range(&self) -> Range<usize> {
        self.offset..self.offset + self.latent_dimension
    }
}

#[derive(Debug, Clone)]
pub struct Hodge1FormDecomposedPrior {
    pub ambient_dimension: usize,
    pub precision: FeecCsr,
    pub latent_mean: FeecVector,
    pub latent_to_ambient: FeecCsr,
    pub harmonic_basis: FeecMatrix,
    pub mass_u: FeecCsr,
    pub branches: Vec<Hodge1FormBranchPrior>,
}

#[derive(Debug, Clone)]
pub struct HodgeProjectionOperatorConfig {
    pub harmonic_dim: Option<usize>,
    pub harmonic_basis_override: Option<FeecMatrix>,
    pub ridge: f64,
}

impl Default for HodgeProjectionOperatorConfig {
    fn default() -> Self {
        Self {
            harmonic_dim: None,
            harmonic_basis_override: None,
            ridge: DEFAULT_PROJECTION_RIDGE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HodgeProjectionOperator {
    pub exact: FeecCsr,
    pub coexact: FeecCsr,
    pub harmonic: FeecCsr,
    pub total: FeecCsr,
    pub harmonic_basis: FeecMatrix,
    pub mass_u: FeecCsr,
}

impl HodgeProjectionOperator {
    pub fn ambient_dimension(&self) -> usize {
        self.mass_u.nrows()
    }

    pub fn apply_exact(&self, values: &FeecVector) -> FeecVector {
        &self.exact * values
    }

    pub fn apply_coexact(&self, values: &FeecVector) -> FeecVector {
        &self.coexact * values
    }

    pub fn apply_harmonic(&self, values: &FeecVector) -> FeecVector {
        &self.harmonic * values
    }

    pub fn apply_total(&self, values: &FeecVector) -> FeecVector {
        &self.total * values
    }

    pub fn total_operator(&self) -> Result<SparseRowOperator, String> {
        sparse_row_operator_from_feec_csr(&self.total)
    }

    pub fn harmonic_operator(&self) -> Result<SparseRowOperator, String> {
        sparse_row_operator_from_feec_csr(&self.harmonic)
    }
}

impl Hodge1FormDecomposedPrior {
    pub fn latent_dimension(&self) -> usize {
        self.latent_mean.len()
    }

    pub fn branch(&self, kind: HodgeBranchKind) -> Option<&Hodge1FormBranchPrior> {
        self.branches.iter().find(|branch| branch.kind == kind)
    }

    pub fn branch_operator(&self, kind: HodgeBranchKind) -> Result<SparseRowOperator, String> {
        let branch = self
            .branch(kind)
            .ok_or_else(|| format!("Hodge prior does not contain {} branch", kind.as_str()))?;
        offset_transform_operator(&branch.transform, branch.offset, self.latent_dimension())
    }

    pub fn ambient_operator(&self) -> Result<SparseRowOperator, String> {
        sparse_row_operator_from_feec_csr(&self.latent_to_ambient)
    }

    pub fn gaussian_prior_spec(&self) -> GaussianPriorSpec {
        GaussianPriorSpec {
            mean: self.latent_mean.iter().copied().collect(),
            precision: feec_csr_to_core_triplet(&self.precision),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HodgeLinearConditioningResult {
    pub latent: LinearGaussianConditioningResult,
    pub posterior_mean: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub branch_posterior_means: BTreeMap<HodgeBranchKind, FeecVector>,
    pub branch_prior_variances: BTreeMap<HodgeBranchKind, FeecVector>,
    pub branch_posterior_variances: BTreeMap<HodgeBranchKind, FeecVector>,
}

#[derive(Debug, Clone)]
pub struct HodgeLinearPdeUqResult {
    pub latent: LinearPdeUqResult,
    pub posterior_mean: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub branch_posterior_means: BTreeMap<HodgeBranchKind, FeecVector>,
    pub branch_variances: BTreeMap<HodgeBranchKind, LinearPdeDerivedMarginalResult>,
}

pub fn build_exact_1form_transform(topology: &Complex) -> FeecCsr {
    FeecCsr::from(&topology.exterior_derivative_operator(0))
}

pub fn build_coexact_1form_transform(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u_1form: &FeecCsr,
) -> FeecCsr {
    build_coexact_1form_transform_with_optional_coords(
        topology,
        None,
        metric,
        mass_u_1form,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
    )
    .unwrap_or_else(|err| panic!("{err}"))
}

pub fn build_coexact_1form_transform_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_u_1form: &FeecCsr,
    mass_inverse: Matern1FormMassInverse,
) -> Result<FeecCsr, String> {
    build_coexact_1form_transform_with_optional_coords(
        topology,
        Some(coords),
        metric,
        mass_u_1form,
        mass_inverse,
    )
}

fn build_coexact_1form_transform_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    mass_u_1form: &FeecCsr,
    mass_inverse: Matern1FormMassInverse,
) -> Result<FeecCsr, String> {
    let galmats = MixedGalmats::compute(topology, metric, 2);
    let codif_u = FeecCsr::from(galmats.codif_u());
    let mass_inverse_1form = if let Some(coords) = coords {
        build_matern_mass_inverse_1form_with_coords(
            topology,
            coords,
            metric,
            mass_u_1form,
            mass_inverse,
        )?
    } else if mass_inverse == Matern1FormMassInverse::BarycentricDualSparseInverse {
        return Err("barycentric-dual coexact 1-form transform requires MeshCoords".to_string());
    } else {
        build_matern_mass_inverse_1form(topology, metric, mass_u_1form, mass_inverse)
    };
    Ok(&mass_inverse_1form * &codif_u)
}

pub fn build_exact_mass_coexact_1form_transform(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u_1form: &FeecCsr,
) -> Result<FeecCsr, String> {
    if topology.dim() < 2 {
        return Err("exact-mass coexact 1-form transform requires a 2D mesh".to_string());
    }
    if mass_u_1form.nrows() != mass_u_1form.ncols() {
        return Err("1-form mass matrix must be square".to_string());
    }
    if mass_u_1form.nrows() != topology.nsimplices(1) {
        return Err(format!(
            "1-form mass matrix has {} rows but topology has {} edges",
            mass_u_1form.nrows(),
            topology.nsimplices(1)
        ));
    }

    let galmats = MixedGalmats::compute(topology, metric, 2);
    let codif_u = FeecCsr::from(galmats.codif_u());
    if codif_u.nrows() != mass_u_1form.nrows() {
        return Err(format!(
            "2-form codifferential has {} rows but 1-form mass has {} rows",
            codif_u.nrows(),
            mass_u_1form.nrows()
        ));
    }

    let factor = feec_csr_to_gmrf(mass_u_1form)
        .cholesky_sqrt_lower()
        .map_err(|err| {
            format!("failed to factor 1-form mass for exact coexact transform: {err}")
        })?;
    let mut rhs_columns = (0..codif_u.ncols())
        .map(|_| FeecVector::zeros(codif_u.nrows()))
        .collect::<Vec<_>>();
    for (row, col, value) in codif_u.triplet_iter() {
        rhs_columns[col][row] += *value;
    }

    let mut transform = FeecMatrix::zeros(codif_u.nrows(), codif_u.ncols());
    for (col, rhs) in rhs_columns.iter().enumerate() {
        let solution = factor.solve(&feec_vec_to_gmrf(rhs)).map_err(|err| {
            format!("failed to solve exact coexact transform column {col}: {err}")
        })?;
        let solution = gmrf_vec_to_feec(&solution);
        for row in 0..solution.len() {
            transform[(row, col)] = solution[row];
        }
    }

    Ok(dense_to_feec_csr(&transform, 1e-13))
}

pub fn transformed_mass_expected_energy_from_precision(
    precision: &FeecCsr,
    transform: &FeecCsr,
    mass: &FeecCsr,
) -> Result<f64, String> {
    if precision.nrows() != precision.ncols() {
        return Err(format!(
            "precision must be square, got {}x{}",
            precision.nrows(),
            precision.ncols()
        ));
    }
    if transform.ncols() != precision.nrows() {
        return Err(format!(
            "transform has {} columns but precision dimension is {}",
            transform.ncols(),
            precision.nrows()
        ));
    }
    if mass.nrows() != transform.nrows() || mass.ncols() != transform.nrows() {
        return Err(format!(
            "mass matrix {}x{} does not match transform row count {}",
            mass.nrows(),
            mass.ncols(),
            transform.nrows()
        ));
    }

    let weighted_transform = mass * transform;
    let gram = transform.transpose() * &weighted_transform;
    let mut rhs = GmrfDenseMatrix::zeros(precision.nrows(), precision.ncols());
    for (row, col, value) in gram.triplet_iter() {
        rhs[(row, col)] += *value;
    }
    let factor = feec_csr_to_gmrf(precision)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor precision for mass-energy trace: {err}"))?;
    factor
        .solve_dense_in_place(&mut rhs)
        .map_err(|err| format!("failed to solve precision for mass-energy trace: {err}"))?;
    Ok((0..precision.nrows())
        .map(|index| rhs[(index, index)])
        .sum())
}

pub fn compute_harmonic_basis_1form(
    topology: &Complex,
    metric: &MeshLengths,
    harmonic_dim: usize,
    harmonic_basis_override: Option<&FeecMatrix>,
) -> Result<FeecMatrix, String> {
    if let Some(basis) = harmonic_basis_override {
        if basis.nrows() != topology.skeleton(1).len() {
            return Err(format!(
                "harmonic basis row count {} does not match 1-form dimension {}",
                basis.nrows(),
                topology.skeleton(1).len()
            ));
        }
        if basis.ncols() != harmonic_dim {
            return Err(format!(
                "harmonic basis column count {} does not match requested harmonic dimension {}",
                basis.ncols(),
                harmonic_dim
            ));
        }
        return Ok(basis.clone());
    }

    let galmats = MixedGalmats::compute(topology, metric, 1);
    Ok(solve_hodge_laplace_harmonics_with_galmats(
        topology,
        &galmats,
        1,
        harmonic_dim,
        None,
        None,
    ))
}

pub fn mass_orthonormalize_harmonic_basis_1form(
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<FeecMatrix, String> {
    if harmonic_basis.ncols() == 0 {
        return Ok(FeecMatrix::zeros(harmonic_basis.nrows(), 0));
    }

    let mut columns = Vec::with_capacity(harmonic_basis.ncols());
    for j in 0..harmonic_basis.ncols() {
        let mut column = harmonic_basis.column(j).into_owned();
        for previous in &columns {
            let coeff = mass_inner_product(previous, &column, mass_u);
            column -= previous * coeff;
        }

        let norm_sq = mass_inner_product(&column, &column, mass_u);
        if !norm_sq.is_finite() || norm_sq <= 1e-12 {
            return Err(format!(
                "harmonic basis column {j} became singular during mass orthonormalization"
            ));
        }
        column /= norm_sq.sqrt();
        columns.push(column);
    }

    Ok(FeecMatrix::from_columns(&columns))
}

pub fn build_harmonic_restricted_precision(q1: &FeecCsr, harmonic_basis: &FeecMatrix) -> FeecCsr {
    if harmonic_basis.ncols() == 0 {
        return FeecCsr::from(&common::linalg::nalgebra::CooMatrix::new(0, 0));
    }

    let q1_dense = feec_csr_to_dense(q1);
    let reduced = harmonic_basis.transpose() * q1_dense * harmonic_basis;
    dense_to_feec_csr(&reduced, 0.0)
}

pub fn build_hodge_projection_operator_1form(
    topology: &Complex,
    metric: &MeshLengths,
    config: HodgeProjectionOperatorConfig,
) -> Result<HodgeProjectionOperator, String> {
    let hodge = build_hodge_laplacian_1form(topology, metric);
    let ambient_dimension = hodge.mass_u.nrows();
    let harmonic_dim = config
        .harmonic_dim
        .unwrap_or_else(|| topology.homology_dim(1));
    let harmonic_basis = if harmonic_dim == 0 {
        FeecMatrix::zeros(ambient_dimension, 0)
    } else {
        let basis = compute_harmonic_basis_1form(
            topology,
            metric,
            harmonic_dim,
            config.harmonic_basis_override.as_ref(),
        )?;
        mass_orthonormalize_harmonic_basis_1form(&basis, &hodge.mass_u)?
    };

    build_hodge_projection_operator_1form_with_basis(
        topology,
        metric,
        &hodge.mass_u,
        harmonic_basis,
        config.ridge,
    )
}

pub fn build_hodge_projection_operator_1form_with_basis(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    harmonic_basis: FeecMatrix,
    ridge: f64,
) -> Result<HodgeProjectionOperator, String> {
    let ambient_dimension = mass_u.nrows();
    if mass_u.ncols() != ambient_dimension {
        return Err("1-form mass matrix must be square".to_string());
    }
    if harmonic_basis.nrows() != ambient_dimension {
        return Err(format!(
            "harmonic basis row count {} does not match 1-form dimension {}",
            harmonic_basis.nrows(),
            ambient_dimension
        ));
    }

    let exact_transform = build_exact_1form_transform(topology);
    let coexact_transform = build_coexact_1form_transform(topology, metric, mass_u);
    let harmonic_transform = dense_to_feec_csr(&harmonic_basis, 0.0);

    let exact = mass_projection_matrix(&exact_transform, mass_u, ridge, "exact")?;
    let coexact = mass_projection_matrix(&coexact_transform, mass_u, ridge, "coexact")?;
    let harmonic = mass_projection_matrix(&harmonic_transform, mass_u, ridge, "harmonic")?;
    let combined_transform =
        hstack_feec_csr(&[&exact_transform, &coexact_transform, &harmonic_transform])?;
    let total = mass_projection_matrix(&combined_transform, mass_u, ridge, "total")?;

    Ok(HodgeProjectionOperator {
        exact,
        coexact,
        harmonic,
        total,
        harmonic_basis,
        mass_u: mass_u.clone(),
    })
}

pub fn build_mass_projection_operator_1form(
    transform: &FeecCsr,
    mass_u: &FeecCsr,
    ridge: f64,
    label: &str,
) -> Result<FeecCsr, String> {
    mass_projection_matrix(transform, mass_u, ridge, label)
}

pub fn build_hodge_1form_decomposed_prior(
    topology: &Complex,
    metric: &MeshLengths,
    config: Hodge1FormPriorConfig,
) -> Result<Hodge1FormDecomposedPrior, String> {
    build_hodge_1form_decomposed_prior_with_optional_coords(topology, None, metric, config)
}

pub fn build_hodge_1form_decomposed_prior_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    config: Hodge1FormPriorConfig,
) -> Result<Hodge1FormDecomposedPrior, String> {
    build_hodge_1form_decomposed_prior_with_optional_coords(topology, Some(coords), metric, config)
}

fn build_hodge_1form_decomposed_prior_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    config: Hodge1FormPriorConfig,
) -> Result<Hodge1FormDecomposedPrior, String> {
    validate_config(&config)?;

    let hodge = build_hodge_laplacian_1form(topology, metric);
    let ambient_dimension = hodge.mass_u.nrows();
    let harmonic_basis = if config.branches.contains(&HodgeBranchKind::Harmonic) {
        let basis = compute_harmonic_basis_1form(
            topology,
            metric,
            config.harmonic_dim,
            config.harmonic_basis_override.as_ref(),
        )?;
        mass_orthonormalize_harmonic_basis_1form(&basis, &hodge.mass_u)?
    } else {
        FeecMatrix::zeros(ambient_dimension, 0)
    };

    let mut branches = Vec::with_capacity(config.branches.len());
    let mut offset = 0;
    let assembly = HodgeBranchPriorAssembly {
        topology,
        coords,
        metric,
        hodge: &hodge,
        harmonic_basis: &harmonic_basis,
        config: &config,
    };
    for kind in &config.branches {
        let mut branch = build_branch_prior(*kind, offset, &assembly)?;
        offset += branch.latent_dimension;
        branch.offset = offset - branch.latent_dimension;
        branches.push(branch);
    }

    let precision_blocks = branches
        .iter()
        .map(|branch| &branch.precision)
        .collect::<Vec<_>>();
    let transform_blocks = branches
        .iter()
        .map(|branch| &branch.transform)
        .collect::<Vec<_>>();
    let precision = block_diag_feec_csr(&precision_blocks);
    let latent_to_ambient = hstack_feec_csr(&transform_blocks)?;
    let latent_mean = FeecVector::zeros(offset);

    Ok(Hodge1FormDecomposedPrior {
        ambient_dimension,
        precision,
        latent_mean,
        latent_to_ambient,
        harmonic_basis,
        mass_u: hodge.mass_u,
        branches,
    })
}

pub fn build_hodge_1form_branch_prior(
    topology: &Complex,
    metric: &MeshLengths,
    kind: HodgeBranchKind,
    config: Hodge1FormPriorConfig,
) -> Result<Hodge1FormBranchPrior, String> {
    let prior = build_hodge_1form_decomposed_prior(
        topology,
        metric,
        Hodge1FormPriorConfig {
            branches: vec![kind],
            ..config
        },
    )?;
    prior
        .branches
        .into_iter()
        .next()
        .ok_or_else(|| "single-branch Hodge prior unexpectedly contained no branches".to_string())
}

pub fn build_hodge_linear_conditioning_problem(
    prior: &Hodge1FormDecomposedPrior,
    observation_matrix: &FeecCsr,
    observations: &FeecVector,
    noise_variance: f64,
    hutchinson: HutchinsonConfig,
) -> Result<LinearGaussianConditioningProblem, String> {
    if observation_matrix.ncols() != prior.ambient_dimension {
        return Err(format!(
            "observation matrix column count {} must match Hodge prior ambient dimension {}",
            observation_matrix.ncols(),
            prior.ambient_dimension
        ));
    }
    if observation_matrix.nrows() != observations.len() {
        return Err(format!(
            "observation vector length {} must match observation row count {}",
            observations.len(),
            observation_matrix.nrows()
        ));
    }

    let latent_observation_matrix = observation_matrix * &prior.latent_to_ambient;
    Ok(LinearGaussianConditioningProblem {
        prior_precision: feec_csr_to_gmrf(&prior.precision),
        observation_operator: feec_csr_to_gmrf(&latent_observation_matrix),
        observations: feec_vec_to_gmrf(observations),
        noise_variance,
        harmonic_subspace: None,
        derived_operators: hodge_derived_operators(prior, DerivedVarianceMode::Exact)?,
        hutchinson,
    })
}

pub fn solve_hodge_linear_conditioning(
    prior: &Hodge1FormDecomposedPrior,
    observation_matrix: &FeecCsr,
    observations: &FeecVector,
    noise_variance: f64,
    hutchinson: HutchinsonConfig,
) -> Result<HodgeLinearConditioningResult, String> {
    let latent = build_hodge_linear_conditioning_problem(
        prior,
        observation_matrix,
        observations,
        noise_variance,
        hutchinson,
    )?
    .solve()
    .map_err(|err| err.to_string())?;
    let posterior_mean = &prior.latent_to_ambient * &gmrf_vec_to_feec(&latent.posterior_mean);
    let prior_variance =
        derived_variance(&latent.derived_prior_variances, HODGE_AMBIENT_DERIVED_NAME)?;
    let posterior_variance = derived_variance(
        &latent.derived_posterior_variances,
        HODGE_AMBIENT_DERIVED_NAME,
    )?;
    let branch_posterior_means = branch_posterior_means(prior, latent.posterior_mean.as_slice());
    let branch_prior_variances =
        branch_conditioning_variances(prior, &latent.derived_prior_variances)?;
    let branch_posterior_variances =
        branch_conditioning_variances(prior, &latent.derived_posterior_variances)?;

    Ok(HodgeLinearConditioningResult {
        latent,
        posterior_mean,
        prior_variance,
        posterior_variance,
        branch_posterior_means,
        branch_prior_variances,
        branch_posterior_variances,
    })
}

pub fn project_linear_pde_problem_to_hodge_latents(
    prior: &Hodge1FormDecomposedPrior,
    problem: &LinearPdeUqProblem,
) -> Result<LinearPdeUqProblem, String> {
    if problem.system.layout.full_dimension != prior.ambient_dimension {
        return Err(format!(
            "linear PDE layout full dimension {} must match Hodge prior ambient dimension {}",
            problem.system.layout.full_dimension, prior.ambient_dimension
        ));
    }

    let reduced_transform =
        restrict_rows_with_layout(&prior.latent_to_ambient, &problem.system.layout)?;
    let projected_operator = &problem.system.operator * &reduced_transform;
    let physical_measurements = transform_measurements_to_latents(
        &problem.physical_measurements,
        &problem.system.layout,
        &reduced_transform,
    )?;
    let derived_quantities =
        transform_derived_quantities_to_latents(prior, &problem.derived_quantities)?;
    let latent_dimension = prior.latent_dimension();

    Ok(LinearPdeUqProblem {
        state_prior: prior.gaussian_prior_spec(),
        system: ReducedLinearPdeAssembly {
            operator: projected_operator,
            residual_bias: problem.system.residual_bias.clone(),
            state_mass: identity_feec_csr(latent_dimension, 1.0),
            state_mass_inverse: Some(identity_feec_csr(latent_dimension, 1.0)),
            layout: DofLayout::identity(latent_dimension),
            forcing_operator: problem.system.forcing_operator.clone(),
            neumann_operator: problem.system.neumann_operator.clone(),
        },
        uncertain_inputs: problem.uncertain_inputs.clone(),
        physical_measurements,
        joint_measurements: Vec::new(),
        derived_quantities,
        joint_derived_quantities: Vec::new(),
        pde_variance: problem.pde_variance,
        pde_precision: problem.pde_precision.clone(),
    })
}

pub fn solve_hodge_linear_pde_uq_with_config(
    prior: &Hodge1FormDecomposedPrior,
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
) -> Result<HodgeLinearPdeUqResult, String> {
    let projected = project_linear_pde_problem_to_hodge_latents(prior, problem)?;
    let latent = solve_linear_pde_uq_with_config(&projected, config)?;
    let posterior_mean = &prior.latent_to_ambient * &latent.posterior_mean;
    let ambient_variance = latent
        .derived_variances
        .get(HODGE_AMBIENT_DERIVED_NAME)
        .ok_or_else(|| "Hodge ambient derived variance was not produced".to_string())?;
    let prior_variance = ambient_variance.prior_variance.clone();
    let posterior_variance = ambient_variance.posterior_variance.clone();
    let branch_posterior_means = branch_posterior_means(prior, latent.posterior_mean.as_slice());
    let mut branch_variances = BTreeMap::new();
    for branch in &prior.branches {
        let name = branch_derived_name(branch.kind);
        let variance = latent.derived_variances.get(&name).ok_or_else(|| {
            format!(
                "Hodge {} branch variance was not produced",
                branch.kind.as_str()
            )
        })?;
        branch_variances.insert(branch.kind, variance.clone());
    }

    Ok(HodgeLinearPdeUqResult {
        latent,
        posterior_mean,
        prior_variance,
        posterior_variance,
        branch_posterior_means,
        branch_variances,
    })
}

pub fn default_hodge_hutchinson_config() -> HutchinsonConfig {
    HutchinsonConfig {
        num_probes: DEFAULT_HUTCHINSON_PROBES,
        batch_count: DEFAULT_HUTCHINSON_BATCHES,
        rng_seed: 0,
    }
}

struct HodgeBranchPriorAssembly<'a> {
    topology: &'a Complex,
    coords: Option<&'a MeshCoords>,
    metric: &'a MeshLengths,
    hodge: &'a HodgeLaplacian1Form,
    harmonic_basis: &'a FeecMatrix,
    config: &'a Hodge1FormPriorConfig,
}

fn build_branch_prior(
    kind: HodgeBranchKind,
    offset: usize,
    assembly: &HodgeBranchPriorAssembly<'_>,
) -> Result<Hodge1FormBranchPrior, String> {
    let HodgeBranchPriorAssembly {
        topology,
        coords,
        metric,
        hodge,
        harmonic_basis,
        config,
    } = *assembly;
    let (transform, precision) = match kind {
        HodgeBranchKind::Exact => {
            let transform = build_exact_1form_transform(topology);
            let precision = build_matern_precision_0form(
                &build_laplace_beltrami_0form(topology, metric),
                Matern0FormConfig {
                    kappa: config.kappa,
                    tau: config.tau,
                    mass_inverse: config.zero_form_mass_inverse,
                },
            );
            (transform, precision)
        }
        HodgeBranchKind::Coexact => {
            if topology.dim() < 2 {
                return Err(
                    "coexact 1-form priors require topological dimension at least 2".to_string(),
                );
            }
            let transform = build_coexact_1form_transform_with_optional_coords(
                topology,
                coords,
                metric,
                &hodge.mass_u,
                config.one_form_mass_inverse,
            )?;
            let hodge_2form = if let Some(coords) = coords {
                build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
                    topology,
                    coords,
                    metric,
                    config.one_form_mass_inverse,
                )?
            } else {
                build_hodge_laplacian_2form(topology, metric)?
            };
            let precision_config = Matern2FormConfig {
                kappa: config.kappa,
                tau: config.tau,
                mass_inverse: config.two_form_mass_inverse,
            };
            let precision = if let Some(coords) = coords {
                build_matern_precision_2form_with_coords(
                    topology,
                    coords,
                    metric,
                    &hodge_2form,
                    precision_config,
                )?
            } else {
                build_matern_precision_2form(topology, metric, &hodge_2form, precision_config)?
            };
            (transform, precision)
        }
        HodgeBranchKind::Harmonic => {
            let transform = dense_to_feec_csr(harmonic_basis, 0.0);
            let precision_config = Matern1FormConfig {
                kappa: config.kappa,
                tau: config.tau,
                mass_inverse: config.one_form_mass_inverse,
            };
            let q1 = if let Some(coords) = coords {
                build_matern_precision_1form_with_coords(
                    topology,
                    coords,
                    metric,
                    hodge,
                    precision_config,
                )?
            } else if config.one_form_mass_inverse
                == Matern1FormMassInverse::BarycentricDualSparseInverse
            {
                return Err(
                    "barycentric-dual harmonic 1-form precision requires MeshCoords".to_string(),
                );
            } else {
                build_matern_precision_1form(topology, metric, hodge, precision_config)
            };
            let precision = build_harmonic_restricted_precision(&q1, harmonic_basis);
            (transform, precision)
        }
    };

    if transform.nrows() != hodge.mass_u.nrows() {
        return Err(format!(
            "{} transform row count {} must match 1-form dimension {}",
            kind.as_str(),
            transform.nrows(),
            hodge.mass_u.nrows()
        ));
    }
    if precision.nrows() != precision.ncols() {
        return Err(format!("{} precision must be square", kind.as_str()));
    }
    if transform.ncols() != precision.nrows() {
        return Err(format!(
            "{} transform column count {} must match precision dimension {}",
            kind.as_str(),
            transform.ncols(),
            precision.nrows()
        ));
    }

    Ok(Hodge1FormBranchPrior {
        kind,
        offset,
        latent_dimension: precision.nrows(),
        ambient_dimension: transform.nrows(),
        precision,
        transform,
    })
}

fn validate_config(config: &Hodge1FormPriorConfig) -> Result<(), String> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("Hodge prior kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("Hodge prior tau must be finite and positive".to_string());
    }
    if config.branches.is_empty() {
        return Err("Hodge prior must contain at least one branch".to_string());
    }
    let mut seen = BTreeSet::new();
    for branch in &config.branches {
        if !seen.insert(*branch) {
            return Err(format!(
                "Hodge prior branch `{}` was requested more than once",
                branch.as_str()
            ));
        }
    }
    Ok(())
}

fn mass_projection_matrix(
    transform: &FeecCsr,
    mass: &FeecCsr,
    ridge: f64,
    label: &str,
) -> Result<FeecCsr, String> {
    if transform.nrows() != mass.nrows() || mass.nrows() != mass.ncols() {
        return Err(format!(
            "{label} projection dimensions do not align: transform={}x{}, mass={}x{}",
            transform.nrows(),
            transform.ncols(),
            mass.nrows(),
            mass.ncols()
        ));
    }
    if transform.ncols() == 0 {
        return Ok(FeecCsr::from(&common::linalg::nalgebra::CooMatrix::new(
            mass.nrows(),
            mass.nrows(),
        )));
    }

    let basis = mass_orthonormalize_column_space(transform, mass, ridge, label)?;
    if basis.ncols() == 0 {
        return Ok(FeecCsr::from(&common::linalg::nalgebra::CooMatrix::new(
            mass.nrows(),
            mass.nrows(),
        )));
    }
    let mass_dense = feec_csr_to_dense(mass);
    let projection = &basis * basis.transpose() * mass_dense;
    Ok(dense_to_feec_csr(&projection, 1e-13))
}

fn mass_orthonormalize_column_space(
    transform: &FeecCsr,
    mass_u: &FeecCsr,
    tolerance: f64,
    label: &str,
) -> Result<FeecMatrix, String> {
    let transform_dense = feec_csr_to_dense(transform);
    let tol = if tolerance.is_finite() && tolerance > 0.0 {
        tolerance
    } else {
        1e-12
    };
    let mut columns = Vec::new();
    for j in 0..transform_dense.ncols() {
        let mut column = transform_dense.column(j).into_owned();
        for previous in &columns {
            let coeff = mass_inner_product(previous, &column, mass_u);
            column -= previous * coeff;
        }

        let norm_sq = mass_inner_product(&column, &column, mass_u);
        if !norm_sq.is_finite() {
            return Err(format!(
                "{label} projection column {j} produced a non-finite mass norm"
            ));
        }
        if norm_sq <= tol {
            continue;
        }
        column /= norm_sq.sqrt();
        columns.push(column);
    }
    if columns.is_empty() {
        Ok(FeecMatrix::zeros(transform.nrows(), 0))
    } else {
        Ok(FeecMatrix::from_columns(&columns))
    }
}

fn mass_inner_product(lhs: &FeecVector, rhs: &FeecVector, mass_u: &FeecCsr) -> f64 {
    let weighted_rhs = mass_u * rhs;
    lhs.dot(&weighted_rhs)
}

fn hodge_derived_operators(
    prior: &Hodge1FormDecomposedPrior,
    variance_mode: DerivedVarianceMode,
) -> Result<DerivedOperatorSet, String> {
    let mut derived = DerivedOperatorSet::new();
    derived.insert(
        HODGE_AMBIENT_DERIVED_NAME.to_string(),
        DerivedOperator {
            operator: prior.ambient_operator()?,
            variance_mode,
        },
    );
    for branch in &prior.branches {
        derived.insert(
            branch_derived_name(branch.kind),
            DerivedOperator {
                operator: prior.branch_operator(branch.kind)?,
                variance_mode,
            },
        );
    }
    Ok(derived)
}

fn transform_measurements_to_latents(
    measurements: &[LinearGaussianMeasurementSpec],
    layout: &DofLayout,
    reduced_transform: &FeecCsr,
) -> Result<Vec<LinearGaussianMeasurementSpec>, String> {
    measurements
        .iter()
        .map(|measurement| {
            let operator = core_triplet_to_feec_csr(&measurement.operator);
            let bias = FeecVector::from_vec(measurement.bias.clone());
            let (reduced_operator, reduced_bias) =
                restrict_columns_and_fold_fixed(&operator, &bias, layout)?;
            let latent_operator = &reduced_operator * reduced_transform;
            Ok(LinearGaussianMeasurementSpec {
                name: measurement.name.clone(),
                operator: feec_csr_to_core_triplet(&latent_operator),
                observations: measurement.observations.clone(),
                bias: reduced_bias.iter().copied().collect(),
                variance: measurement.variance,
            })
        })
        .collect()
}

fn transform_derived_quantities_to_latents(
    prior: &Hodge1FormDecomposedPrior,
    derived: &[LinearPdeDerivedQuantitySpec],
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    let mut transformed = Vec::with_capacity(1 + prior.branches.len() + derived.len());
    transformed.push(LinearPdeDerivedQuantitySpec {
        name: HODGE_AMBIENT_DERIVED_NAME.to_string(),
        operator: prior.ambient_operator()?,
    });
    for branch in &prior.branches {
        transformed.push(LinearPdeDerivedQuantitySpec {
            name: branch_derived_name(branch.kind),
            operator: prior.branch_operator(branch.kind)?,
        });
    }
    for quantity in derived {
        transformed.push(LinearPdeDerivedQuantitySpec {
            name: quantity.name.clone(),
            operator: compose_sparse_row_operator_with_csr(
                &quantity.operator,
                &prior.latent_to_ambient,
            )?,
        });
    }
    Ok(transformed)
}

fn branch_posterior_means(
    prior: &Hodge1FormDecomposedPrior,
    latent_mean: &[f64],
) -> BTreeMap<HodgeBranchKind, FeecVector> {
    prior
        .branches
        .iter()
        .map(|branch| {
            let range = branch.latent_range();
            let latent = FeecVector::from_vec(latent_mean[range].to_vec());
            (branch.kind, &branch.transform * &latent)
        })
        .collect()
}

fn branch_conditioning_variances(
    prior: &Hodge1FormDecomposedPrior,
    variances: &BTreeMap<String, gmrf_core::TransformedVarianceDecomposition>,
) -> Result<BTreeMap<HodgeBranchKind, FeecVector>, String> {
    let mut out = BTreeMap::new();
    for branch in &prior.branches {
        out.insert(
            branch.kind,
            derived_variance(variances, &branch_derived_name(branch.kind))?,
        );
    }
    Ok(out)
}

fn derived_variance(
    variances: &BTreeMap<String, gmrf_core::TransformedVarianceDecomposition>,
    name: &str,
) -> Result<FeecVector, String> {
    variances
        .get(name)
        .map(|decomposition| gmrf_vec_to_feec(&decomposition.constrained_diag))
        .ok_or_else(|| format!("derived variance `{name}` was not produced"))
}

fn branch_derived_name(kind: HodgeBranchKind) -> String {
    format!("hodge_{}", kind.as_str())
}

fn offset_transform_operator(
    transform: &FeecCsr,
    offset: usize,
    ncols: usize,
) -> Result<SparseRowOperator, String> {
    if offset + transform.ncols() > ncols {
        return Err("branch transform offset exceeds joint latent dimension".to_string());
    }
    let mut rows = vec![Vec::new(); transform.nrows()];
    for (row, col, value) in transform.triplet_iter() {
        if *value != 0.0 {
            rows[row].push((offset + col, *value));
        }
    }
    SparseRowOperator::new(ncols, rows).map_err(|err| err.to_string())
}

fn compose_sparse_row_operator_with_csr(
    lhs: &SparseRowOperator,
    rhs: &FeecCsr,
) -> Result<SparseRowOperator, String> {
    if lhs.ncols != rhs.nrows() {
        return Err(format!(
            "operator column count {} must match transform row count {}",
            lhs.ncols,
            rhs.nrows()
        ));
    }
    let rhs_rows = csr_rows(rhs);
    let mut composed = Vec::with_capacity(lhs.nrows());
    for row in &lhs.rows {
        let mut accum = BTreeMap::<usize, f64>::new();
        for (mid, lhs_value) in row {
            for (col, rhs_value) in &rhs_rows[*mid] {
                *accum.entry(*col).or_insert(0.0) += *lhs_value * *rhs_value;
            }
        }
        composed.push(
            accum
                .into_iter()
                .filter(|(_, value)| *value != 0.0)
                .collect(),
        );
    }
    SparseRowOperator::new(rhs.ncols(), composed).map_err(|err| err.to_string())
}

fn csr_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if *value != 0.0 {
            rows[row].push((col, *value));
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::identity_triplet_matrix;
    use common::linalg::nalgebra::CooMatrix as FeecCoo;
    #[cfg(feature = "external-solver-tests")]
    use manifold::io::gmsh::gmsh2coord_complex;
    use manifold::{dim3::mesh_sphere_surface, gen::cartesian::CartesianMeshInfo};
    #[cfg(feature = "external-solver-tests")]
    use std::{fs, path::PathBuf};

    #[cfg(feature = "external-solver-tests")]
    fn default_torus_shell_resolution_1_mesh_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell_resolution_1.msh")
    }

    fn selector_from_indices(dimension: usize, indices: &[usize]) -> FeecCsr {
        let mut coo = FeecCoo::new(indices.len(), dimension);
        for (row, col) in indices.iter().copied().enumerate() {
            coo.push(row, col, 1.0);
        }
        FeecCsr::from(&coo)
    }

    fn max_abs_dense(matrix: &FeecMatrix) -> f64 {
        matrix
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    fn max_abs_csr(matrix: &FeecCsr) -> f64 {
        matrix
            .triplet_iter()
            .map(|(_, _, value)| value.abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn transformed_mass_expected_energy_matches_dense_trace_formula() {
        let mut precision_coo = FeecCoo::new(2, 2);
        precision_coo.push(0, 0, 2.0);
        precision_coo.push(1, 1, 4.0);
        let precision = FeecCsr::from(&precision_coo);

        let mut transform_coo = FeecCoo::new(2, 2);
        transform_coo.push(0, 0, 1.0);
        transform_coo.push(1, 1, 1.0);
        let transform = FeecCsr::from(&transform_coo);

        let mut mass_coo = FeecCoo::new(2, 2);
        mass_coo.push(0, 0, 3.0);
        mass_coo.push(1, 1, 5.0);
        let mass = FeecCsr::from(&mass_coo);

        let expected = 3.0 / 2.0 + 5.0 / 4.0;
        let actual = transformed_mass_expected_energy_from_precision(&precision, &transform, &mass)
            .expect("trace diagnostic should compute");
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected:.12e}, got {actual:.12e}"
        );
    }

    #[test]
    fn exact_mass_coexact_transform_is_diagnostic_coclosed() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        let transform = build_exact_mass_coexact_1form_transform(&topology, &metric, &hodge.mass_u)
            .expect("exact-mass coexact transform should build");

        assert_eq!(transform.nrows(), topology.nsimplices(1));
        assert_eq!(transform.ncols(), topology.nsimplices(2));

        let d0 = FeecCsr::from(&topology.exterior_derivative_operator(0));
        let weighted_transform = &hodge.mass_u * &transform;
        let coclosed_residual = d0.transpose() * &weighted_transform;
        let relative_residual =
            max_abs_csr(&coclosed_residual) / max_abs_csr(&weighted_transform).max(1e-12);

        assert!(
            relative_residual <= 1e-10,
            "exact-mass coexact transform should be coclosed, relative residual={relative_residual:.3e}"
        );
    }

    #[test]
    fn hodge_projection_operator_is_idempotent_and_reconstructs_components() {
        let surface = mesh_sphere_surface(0);
        let (topology, coords) = surface.into_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let projection = build_hodge_projection_operator_1form(
            &topology,
            &metric,
            HodgeProjectionOperatorConfig {
                harmonic_dim: Some(0),
                ..HodgeProjectionOperatorConfig::default()
            },
        )
        .expect("projection operator should build");

        let total = feec_csr_to_dense(&projection.total);
        let idempotence_error = &total * &total - &total;
        let idempotence_max = max_abs_dense(&idempotence_error);
        assert!(
            idempotence_max <= 1e-6,
            "projection should be approximately idempotent, max abs error={idempotence_max:e}"
        );

        let values = FeecVector::from_iterator(
            topology.edges().len(),
            (0..topology.edges().len()).map(|index| {
                let x = index as f64 + 1.0;
                (0.37 * x).sin() + 0.25 * (0.11 * x).cos()
            }),
        );
        let exact = projection.apply_exact(&values);
        let coexact = projection.apply_coexact(&values);
        let harmonic = projection.apply_harmonic(&values);
        let reconstructed = &(&exact + &coexact) + &harmonic;
        let total_applied = projection.apply_total(&values);
        assert!((&reconstructed - &total_applied).norm() <= 1e-10);

        let cross = mass_inner_product(&exact, &coexact, &projection.mass_u);
        let denom = mass_inner_product(&exact, &exact, &projection.mass_u)
            .abs()
            .sqrt()
            * mass_inner_product(&coexact, &coexact, &projection.mass_u)
                .abs()
                .sqrt();
        let relative_cross = cross.abs() / denom.max(1e-12);
        assert!(
            relative_cross <= 1e-6,
            "exact and coexact projections should be mass orthogonal, relative cross={relative_cross:e}"
        );
    }

    #[test]
    fn hodge_projection_operator_preserves_harmonic_basis_vectors() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let mut harmonic_basis = FeecMatrix::zeros(topology.edges().len(), 1);
        harmonic_basis[(0, 0)] = 1.0;
        let projection = build_hodge_projection_operator_1form(
            &topology,
            &metric,
            HodgeProjectionOperatorConfig {
                harmonic_dim: Some(1),
                harmonic_basis_override: Some(harmonic_basis),
                ..HodgeProjectionOperatorConfig::default()
            },
        )
        .expect("projection operator should build");
        assert_eq!(projection.harmonic_basis.ncols(), 1);

        let harmonic = projection.harmonic_basis.column(0).into_owned();
        let projected = projection.apply_harmonic(&harmonic);
        let relative_error = (&projected - &harmonic).norm() / harmonic.norm().max(1e-12);
        assert!(
            relative_error <= 1e-6,
            "harmonic projection should preserve harmonic basis vectors, got {relative_error:.3e}"
        );
    }

    #[test]
    fn exact_and_coexact_prior_builds_joint_sparse_blocks() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let prior = build_hodge_1form_decomposed_prior(
            &topology,
            &metric,
            Hodge1FormPriorConfig::selected(
                1.5,
                1.0,
                [HodgeBranchKind::Exact, HodgeBranchKind::Coexact],
                0,
            ),
        )
        .expect("exact/coexact prior should build");

        assert_eq!(prior.branches.len(), 2);
        assert_eq!(prior.ambient_dimension, topology.edges().len());
        assert_eq!(prior.latent_to_ambient.nrows(), topology.edges().len());
        assert_eq!(prior.precision.nrows(), prior.latent_dimension());
        assert_eq!(prior.precision.ncols(), prior.latent_dimension());
        assert_eq!(prior.branch(HodgeBranchKind::Exact).unwrap().offset, 0);
        assert_eq!(
            prior.branch(HodgeBranchKind::Coexact).unwrap().offset,
            prior
                .branch(HodgeBranchKind::Exact)
                .unwrap()
                .latent_dimension
        );
        feec_csr_to_gmrf(&prior.precision)
            .cholesky_sqrt_lower()
            .expect("joint branch precision should factorize");
    }

    #[test]
    fn barycentric_dual_hodge_prior_builds_exact_coexact_and_harmonic_branches() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let harmonic_basis = FeecMatrix::zeros(topology.edges().len(), 0);

        let prior = build_hodge_1form_decomposed_prior_with_coords(
            &topology,
            &coords,
            &metric,
            Hodge1FormPriorConfig {
                kappa: 1.25,
                tau: 1.0,
                branches: HodgeBranchKind::ALL.to_vec(),
                harmonic_dim: 0,
                harmonic_basis_override: Some(harmonic_basis),
                zero_form_mass_inverse: Matern0FormMassInverse::default(),
                one_form_mass_inverse: Matern1FormMassInverse::BarycentricDualSparseInverse,
                two_form_mass_inverse: Matern2FormMassInverse::BarycentricDualSparseInverse,
            },
        )
        .expect("coordinate-aware barycentric-dual Hodge prior should build");

        assert_eq!(prior.branches.len(), HodgeBranchKind::ALL.len());
        assert_eq!(prior.ambient_dimension, topology.edges().len());
        assert!(prior.branch(HodgeBranchKind::Exact).is_some());
        assert!(prior.branch(HodgeBranchKind::Coexact).is_some());
        assert!(prior.branch(HodgeBranchKind::Harmonic).is_some());
        feec_csr_to_gmrf(&prior.precision)
            .cholesky_sqrt_lower()
            .expect("barycentric-dual Hodge prior precision should factorize");
    }

    #[test]
    fn barycentric_dual_hodge_prior_without_coords_is_rejected() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let err = build_hodge_1form_decomposed_prior(
            &topology,
            &metric,
            Hodge1FormPriorConfig {
                kappa: 1.25,
                tau: 1.0,
                branches: vec![HodgeBranchKind::Coexact],
                harmonic_dim: 0,
                harmonic_basis_override: None,
                zero_form_mass_inverse: Matern0FormMassInverse::default(),
                one_form_mass_inverse: Matern1FormMassInverse::BarycentricDualSparseInverse,
                two_form_mass_inverse: Matern2FormMassInverse::default(),
            },
        )
        .expect_err("barycentric-dual Hodge prior should require coordinates");

        assert!(err.contains("requires MeshCoords"));
    }

    #[test]
    fn duplicate_branch_requests_are_rejected() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let err = build_hodge_1form_decomposed_prior(
            &topology,
            &metric,
            Hodge1FormPriorConfig::selected(
                1.0,
                1.0,
                [HodgeBranchKind::Exact, HodgeBranchKind::Exact],
                0,
            ),
        )
        .expect_err("duplicate branches should fail");

        assert!(err.contains("requested more than once"));
    }

    #[test]
    fn linear_conditioning_adapter_returns_ambient_fields() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let prior = build_hodge_1form_decomposed_prior(
            &topology,
            &metric,
            Hodge1FormPriorConfig::branch(1.5, 1.0, HodgeBranchKind::Exact, 0),
        )
        .expect("exact prior should build");
        let phi = FeecVector::from_iterator(
            topology.vertices().len(),
            (0..topology.vertices().len()).map(|i| (i as f64 * 0.2).sin()),
        );
        let truth = &prior.branch(HodgeBranchKind::Exact).unwrap().transform * &phi;
        let observations = selector_from_indices(truth.len(), &[0, 1, 2]) * &truth;
        let result = solve_hodge_linear_conditioning(
            &prior,
            &selector_from_indices(truth.len(), &[0, 1, 2]),
            &observations,
            1e-6,
            default_hodge_hutchinson_config(),
        )
        .expect("Hodge linear conditioning should solve");

        assert_eq!(result.posterior_mean.len(), prior.ambient_dimension);
        assert_eq!(result.posterior_variance.len(), prior.ambient_dimension);
        assert!(result.posterior_mean.iter().all(|value| value.is_finite()));
        assert!(result
            .posterior_variance
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .branch_posterior_means
            .contains_key(&HodgeBranchKind::Exact));
    }

    #[test]
    fn linear_pde_adapter_pushes_posterior_variance_to_ambient_space() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let prior = build_hodge_1form_decomposed_prior(
            &topology,
            &metric,
            Hodge1FormPriorConfig::branch(1.5, 1.0, HodgeBranchKind::Exact, 0),
        )
        .expect("exact prior should build");
        let ambient_dim = prior.ambient_dimension;
        let system = ReducedLinearPdeAssembly {
            operator: identity_feec_csr(ambient_dim, 1.0),
            residual_bias: FeecVector::zeros(ambient_dim),
            state_mass: identity_feec_csr(ambient_dim, 1.0),
            state_mass_inverse: Some(identity_feec_csr(ambient_dim, 1.0)),
            layout: DofLayout::identity(ambient_dim),
            forcing_operator: identity_feec_csr(ambient_dim, -1.0),
            neumann_operator: identity_feec_csr(ambient_dim, -1.0),
        };
        let problem = LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; ambient_dim],
                precision: identity_triplet_matrix(ambient_dim, 1.0),
            },
            system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-3),
            pde_precision: None,
        };

        let result = solve_hodge_linear_pde_uq_with_config(
            &prior,
            &problem,
            &LinearPdeUqSolverConfig::default(),
        )
        .expect("Hodge latent PDE problem should solve");

        assert_eq!(result.posterior_mean.len(), ambient_dim);
        assert_eq!(result.posterior_variance.len(), ambient_dim);
        assert!(result.posterior_mean.iter().all(|value| value.is_finite()));
        assert!(result
            .posterior_variance
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .branch_variances
            .contains_key(&HodgeBranchKind::Exact));
    }

    #[test]
    #[cfg(feature = "external-solver-tests")]
    fn harmonic_branch_builds_requested_torus_dimension() {
        let _lock = crate::conditioning::hodge_1form::lock_feec_harmonic_tests();
        let mesh_bytes = fs::read(default_torus_shell_resolution_1_mesh_path())
            .expect("torus mesh should be readable");
        let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
        let metric = coords.to_edge_lengths(&topology);

        let prior = build_hodge_1form_decomposed_prior(
            &topology,
            &metric,
            Hodge1FormPriorConfig::branch(2.0, 1.0, HodgeBranchKind::Harmonic, 2),
        )
        .expect("harmonic prior should build");

        let harmonic = prior.branch(HodgeBranchKind::Harmonic).unwrap();
        assert_eq!(harmonic.latent_dimension, 2);
        assert_eq!(prior.harmonic_basis.ncols(), 2);
        assert_eq!(prior.latent_to_ambient.ncols(), 2);
        feec_csr_to_gmrf(&prior.precision)
            .cholesky_sqrt_lower()
            .expect("harmonic restricted precision should factorize");
    }
}
