use super::build_lindgren_precision_from_system;
use crate::sparse::{
    add_sparse, core_triplet_to_feec_csr, diag_matrix, feec_csr_to_core_triplet, invert_diag,
    lumped_diag, restrict_square_with_layout, scale_matrix, sparse_row_operator_apply_feec,
    symmetrize_feec_csr,
};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::field::ExteriorField;
use feg_core::{GaussianPriorSpec, SparseTripletMatrix};
use formoniq::reduction::DofLayout;
use formoniq::{
    assemble::{
        assemble_barycentric_dual_1form_sparse_inverse_galmat,
        assemble_whitney_projected_sparse_inverse_galmat, BarycentricDualSparseInverseConfig,
    },
    problems::hodge_laplace::MixedGalmats,
};
use gmrf_core::{
    estimate_transformed_mc_variances, GmrfError, SparseRowOperator, Vector as GmrfVector,
};
use manifold::{
    geometry::{
        coord::{
            mesh::MeshCoords,
            simplex::{barycenter_local, SimplexHandleExt},
        },
        metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
};

pub use super::MaternAlpha;
pub use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf};

const RECONSTRUCTION_EPS: f64 = 1e-12;

type SparseRowLinearOperator = SparseRowOperator;

pub struct HodgeLaplacian1Form {
    pub mass_u: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone, Copy)]
pub struct ReducedLinearProxyMaternAlpha2Config {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: MaternMassInverse,
    pub allow_kappa_fallback: bool,
}

impl Default for ReducedLinearProxyMaternAlpha2Config {
    fn default() -> Self {
        Self {
            kappa: 0.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
            allow_kappa_fallback: true,
        }
    }
}

pub struct ReducedLinearProxyMaternAlpha2Prior {
    pub spec: GaussianPriorSpec,
    pub kappa: f64,
    pub tau: f64,
    pub kappa_fallback_used: bool,
}

pub fn default_reduced_linear_proxy_matern_kappa(_coords: &MeshCoords) -> f64 {
    1.0
}

pub fn inverse_domain_diameter_matern_kappa(coords: &MeshCoords) -> f64 {
    1.0 / bounding_box_diameter(coords).max(1e-12)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaternMassInverse {
    RowSumLumped,
    #[default]
    Nc1ProjectedSparseInverse,
    BarycentricDualSparseInverse,
}

#[derive(Debug, Clone, Copy)]
pub struct MaternConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: MaternMassInverse,
}

#[derive(Debug, Clone)]
pub struct ReconstructedBarycenterFieldOperator {
    ambient_dim: usize,
    component_operators: Vec<SparseRowLinearOperator>,
}

#[derive(Debug, Clone)]
pub struct ReconstructedBarycenterField {
    ambient_dim: usize,
    component_values: Vec<FeecVector>,
}

impl ReconstructedBarycenterFieldOperator {
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    pub fn component_count(&self) -> usize {
        self.component_operators.len()
    }

    pub fn cell_count(&self) -> usize {
        self.component_operators
            .first()
            .map_or(0, SparseRowLinearOperator::nrows)
    }

    pub fn component_rows(&self, component_index: usize) -> Option<&[Vec<(usize, f64)>]> {
        self.component_operators
            .get(component_index)
            .map(|operator| operator.rows.as_slice())
    }

    pub fn apply_to_slice(&self, input: &[f64]) -> Result<ReconstructedBarycenterField, String> {
        let input = FeecVector::from_vec(input.to_vec());
        let mut component_values = Vec::with_capacity(self.component_operators.len());
        for operator in &self.component_operators {
            component_values.push(sparse_row_operator_apply_feec(operator, &input)?);
        }
        ReconstructedBarycenterField::from_components(component_values)
    }
}

pub fn estimate_reconstructed_barycenter_field_component_variances<R, F>(
    operator: &ReconstructedBarycenterFieldOperator,
    mean: &GmrfVector,
    num_samples: usize,
    rng: &mut R,
    mut sample_draw: F,
) -> Result<ReconstructedBarycenterField, String>
where
    R: rand::Rng + ?Sized,
    F: FnMut(&mut R) -> Result<GmrfVector, GmrfError>,
{
    let component_operators = operator.component_operators.iter().collect::<Vec<_>>();
    let stacked_operator =
        SparseRowOperator::stack(&component_operators).map_err(|err| err.to_string())?;
    let stacked_variances =
        estimate_transformed_mc_variances(&stacked_operator, mean, num_samples, rng, |rng| {
            sample_draw(rng)
        })
        .map_err(|err| err.to_string())?;

    let cell_count = operator.cell_count();
    let mut component_values = Vec::with_capacity(operator.component_count());
    for component_index in 0..operator.component_count() {
        let offset = component_index * cell_count;
        component_values.push(FeecVector::from_iterator(
            cell_count,
            (0..cell_count).map(|cell_index| stacked_variances[offset + cell_index]),
        ));
    }

    ReconstructedBarycenterField::from_components(component_values)
}

impl ReconstructedBarycenterField {
    pub fn from_components(component_values: Vec<FeecVector>) -> Result<Self, String> {
        let Some(first) = component_values.first() else {
            return Err("at least one ambient component is required".to_string());
        };
        let cell_count = first.len();
        if component_values
            .iter()
            .any(|component| component.len() != cell_count)
        {
            return Err("all ambient components must have the same cell count".to_string());
        }

        Ok(Self {
            ambient_dim: component_values.len(),
            component_values,
        })
    }

    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    pub fn cell_count(&self) -> usize {
        self.component_values
            .first()
            .map_or(0, |component| component.len())
    }

    pub fn components(&self) -> &[FeecVector] {
        &self.component_values
    }

    pub fn component(&self, component_index: usize) -> Option<&FeecVector> {
        self.component_values.get(component_index)
    }

    pub fn trace(&self) -> FeecVector {
        let mut trace = FeecVector::zeros(self.cell_count());
        for component in &self.component_values {
            trace += component;
        }
        trace
    }

    pub fn vtk_vectors(&self) -> Vec<[f64; 3]> {
        (0..self.cell_count())
            .map(|cell_index| {
                [
                    self.component_values
                        .first()
                        .map_or(0.0, |component| component[cell_index]),
                    self.component_values
                        .get(1)
                        .map_or(0.0, |component| component[cell_index]),
                    self.component_values
                        .get(2)
                        .map_or(0.0, |component| component[cell_index]),
                ]
            })
            .collect()
    }
}

pub fn build_hodge_laplacian_1form(
    topology: &Complex,
    metric: &MeshLengths,
) -> HodgeLaplacian1Form {
    let galmats = MixedGalmats::compute(topology, metric, 1);
    build_hodge_laplacian_1form_from_galmats(&galmats)
}

pub fn build_hodge_laplacian_1form_from_galmats(galmats: &MixedGalmats) -> HodgeLaplacian1Form {
    let mass_u = galmats.mass_u_csr();
    let laplacian = galmats.hodge_laplacian_schur_complement_lumped();
    HodgeLaplacian1Form { mass_u, laplacian }
}

pub fn build_matern_system_matrix_1form(hodge: &HodgeLaplacian1Form, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&hodge.laplacian, &scale_matrix(&hodge.mass_u, kappa2))
}

pub fn build_matern_mass_inverse_1form(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> FeecCsr {
    build_matern_mass_inverse_1form_with_optional_coords(topology, None, metric, mass_u, strategy)
        .unwrap_or_else(|err| panic!("{err}"))
}

pub fn build_matern_mass_inverse_1form_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecCsr, String> {
    build_matern_mass_inverse_1form_with_optional_coords(
        topology,
        Some(coords),
        metric,
        mass_u,
        strategy,
    )
}

pub fn build_exact_dense_mass_inverse_1form(
    mass_u: &FeecCsr,
    drop_tolerance: f64,
) -> Result<FeecCsr, String> {
    if mass_u.nrows() != mass_u.ncols() {
        return Err(format!(
            "exact 1-form mass inverse requires a square matrix, got {}x{}",
            mass_u.nrows(),
            mass_u.ncols()
        ));
    }
    let factor = feec_csr_to_gmrf(mass_u)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor 1-form mass matrix: {err}"))?;
    let n = mass_u.nrows();
    let tolerance = drop_tolerance.abs();
    let mut inverse = FeecCoo::new(n, n);
    for col in 0..n {
        let mut rhs = GmrfVector::zeros(n);
        rhs[col] = 1.0;
        let solution = factor.solve(&rhs).map_err(|err| {
            format!("failed to solve exact 1-form mass inverse column {col}: {err}")
        })?;
        for (row, value) in solution.iter().copied().enumerate() {
            if value.abs() > tolerance {
                inverse.push(row, col, value);
            }
        }
    }
    Ok(FeecCsr::from(&inverse))
}

fn build_matern_mass_inverse_1form_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecCsr, String> {
    match strategy {
        MaternMassInverse::RowSumLumped => Ok(diag_matrix(&invert_diag(&lumped_diag(mass_u)))),
        MaternMassInverse::Nc1ProjectedSparseInverse => {
            let projected = assemble_whitney_projected_sparse_inverse_galmat(topology, metric);
            let projected = FeecCsr::from(&projected);
            validate_mass_inverse_dimensions(
                &projected,
                mass_u,
                "projected 1-form sparse inverse",
            )?;
            Ok(projected)
        }
        MaternMassInverse::BarycentricDualSparseInverse => {
            let coords = coords.ok_or_else(|| {
                "barycentric-dual 1-form sparse inverse requires MeshCoords; use the coordinate-aware checked builder".to_string()
            })?;
            let barycentric = assemble_barycentric_dual_1form_sparse_inverse_galmat(
                topology,
                coords,
                BarycentricDualSparseInverseConfig::default(),
            )?;
            let barycentric = FeecCsr::from(&barycentric);
            validate_mass_inverse_dimensions(
                &barycentric,
                mass_u,
                "barycentric-dual 1-form sparse inverse",
            )?;
            Ok(barycentric)
        }
    }
}

fn validate_mass_inverse_dimensions(
    inverse: &FeecCsr,
    mass: &FeecCsr,
    name: &str,
) -> Result<(), String> {
    if inverse.nrows() != mass.nrows() || inverse.ncols() != mass.ncols() {
        return Err(format!(
            "{name} dimensions {}x{} do not match 1-form mass {}x{}",
            inverse.nrows(),
            inverse.ncols(),
            mass.nrows(),
            mass.ncols()
        ));
    }
    Ok(())
}

pub fn build_matern_precision_1form_with_mass_inverse(
    hodge: &HodgeLaplacian1Form,
    mass_inverse: &FeecCsr,
    kappa: f64,
    tau: f64,
) -> FeecCsr {
    build_matern_precision_1form_with_mass_inverse_for_alpha(
        hodge,
        mass_inverse,
        MaternAlpha::Two,
        kappa,
        tau,
    )
}

pub fn build_matern_precision_1form_with_mass_inverse_for_alpha(
    hodge: &HodgeLaplacian1Form,
    mass_inverse: &FeecCsr,
    alpha: MaternAlpha,
    kappa: f64,
    tau: f64,
) -> FeecCsr {
    let a = build_matern_system_matrix_1form(hodge, kappa);
    build_lindgren_precision_from_system(&a, mass_inverse, alpha, tau)
}

pub fn build_split_graph_matern_precision_1form_with_mass_inverse_for_alpha(
    hodge: &HodgeLaplacian1Form,
    mass_inverse: &FeecCsr,
    groups: &[Vec<usize>],
    alpha: MaternAlpha,
    kappa: f64,
    tau: f64,
) -> Result<FeecCsr, String> {
    validate_split_graph_groups(hodge.mass_u.nrows(), groups)?;
    validate_mass_inverse_dimensions(
        mass_inverse,
        &hodge.mass_u,
        "split-graph 1-form mass inverse",
    )?;
    if hodge.laplacian.nrows() != hodge.mass_u.nrows()
        || hodge.laplacian.ncols() != hodge.mass_u.ncols()
    {
        return Err(format!(
            "split-graph 1-form laplacian dimensions {}x{} must match mass dimensions {}x{}",
            hodge.laplacian.nrows(),
            hodge.laplacian.ncols(),
            hodge.mass_u.nrows(),
            hodge.mass_u.ncols()
        ));
    }

    let mut blocks = Vec::with_capacity(groups.len());
    for group in groups {
        let block_hodge = HodgeLaplacian1Form {
            mass_u: restrict_square_by_indices(&hodge.mass_u, group)?,
            laplacian: restrict_square_by_indices(&hodge.laplacian, group)?,
        };
        let block_mass_inverse = restrict_square_by_indices(mass_inverse, group)?;
        blocks.push(build_matern_precision_1form_with_mass_inverse_for_alpha(
            &block_hodge,
            &block_mass_inverse,
            alpha,
            kappa,
            tau,
        ));
    }

    scatter_square_blocks(hodge.mass_u.nrows(), groups, &blocks)
}

pub fn build_matern_precision_1form(
    topology: &Complex,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
) -> FeecCsr {
    build_matern_precision_1form_for_alpha(topology, metric, hodge, MaternAlpha::Two, config)
}

pub fn build_matern_precision_1form_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    build_matern_precision_1form_for_alpha_with_coords(
        topology,
        coords,
        metric,
        hodge,
        MaternAlpha::Two,
        config,
    )
}

pub fn build_matern_precision_1form_for_alpha(
    topology: &Complex,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> FeecCsr {
    let mass_inverse =
        build_matern_mass_inverse_1form(topology, metric, &hodge.mass_u, config.mass_inverse);
    build_matern_precision_1form_with_mass_inverse_for_alpha(
        hodge,
        &mass_inverse,
        alpha,
        config.kappa,
        config.tau,
    )
}

pub fn build_matern_precision_1form_for_alpha_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    let mass_inverse = build_matern_mass_inverse_1form_with_coords(
        topology,
        coords,
        metric,
        &hodge.mass_u,
        config.mass_inverse,
    )?;
    Ok(build_matern_precision_1form_with_mass_inverse_for_alpha(
        hodge,
        &mass_inverse,
        alpha,
        config.kappa,
        config.tau,
    ))
}

fn validate_split_graph_groups(dimension: usize, groups: &[Vec<usize>]) -> Result<(), String> {
    if dimension == 0 {
        return Err("split-graph 1-form prior requires a nonempty matrix".to_string());
    }
    if groups.is_empty() {
        return Err("split-graph 1-form prior requires at least one group".to_string());
    }
    let mut seen = vec![false; dimension];
    for (group_index, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(format!("split-graph group {group_index} is empty"));
        }
        for &index in group {
            if index >= dimension {
                return Err(format!(
                    "split-graph index {index} is outside matrix dimension {dimension}"
                ));
            }
            if seen[index] {
                return Err(format!(
                    "split-graph index {index} appears in multiple groups"
                ));
            }
            seen[index] = true;
        }
    }
    if let Some(index) = seen.iter().position(|value| !*value) {
        return Err(format!(
            "split-graph index {index} is not covered by any group"
        ));
    }
    Ok(())
}

fn restrict_square_by_indices(matrix: &FeecCsr, indices: &[usize]) -> Result<FeecCsr, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "split-graph restriction requires a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut map = vec![None; matrix.nrows()];
    for (local, &index) in indices.iter().enumerate() {
        if index >= matrix.nrows() {
            return Err(format!(
                "split-graph index {index} is outside matrix dimension {}",
                matrix.nrows()
            ));
        }
        if map[index].replace(local).is_some() {
            return Err(format!("split-graph index {index} appears more than once"));
        }
    }
    let mut coo = FeecCoo::new(indices.len(), indices.len());
    for (row, col, value) in matrix.triplet_iter() {
        if let (Some(local_row), Some(local_col)) = (map[row], map[col]) {
            if *value != 0.0 {
                coo.push(local_row, local_col, *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

fn scatter_square_blocks(
    dimension: usize,
    groups: &[Vec<usize>],
    blocks: &[FeecCsr],
) -> Result<FeecCsr, String> {
    if groups.len() != blocks.len() {
        return Err("split-graph block count does not match group count".to_string());
    }
    let mut coo = FeecCoo::new(dimension, dimension);
    for (group, block) in groups.iter().zip(blocks) {
        if block.nrows() != group.len() || block.ncols() != group.len() {
            return Err(format!(
                "split-graph block dimensions {}x{} do not match group size {}",
                block.nrows(),
                block.ncols(),
                group.len()
            ));
        }
        for (row, col, value) in block.triplet_iter() {
            if *value != 0.0 {
                coo.push(group[row], group[col], *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

pub fn build_reduced_linear_proxy_matern_alpha2_prior(
    topology: &Complex,
    coords: &MeshCoords,
    layout: &DofLayout,
    linear_proxy: &SparseTripletMatrix,
    mean: Vec<f64>,
    config: ReducedLinearProxyMaternAlpha2Config,
) -> Result<ReducedLinearProxyMaternAlpha2Prior, String> {
    if !config.kappa.is_finite() || config.kappa < 0.0 {
        return Err("linear proxy kappa must be finite and nonnegative".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("linear proxy tau must be finite and positive".to_string());
    }
    if mean.len() != layout.reduced_dimension() {
        return Err(format!(
            "linear proxy prior mean length {} must match reduced dimension {}",
            mean.len(),
            layout.reduced_dimension()
        ));
    }
    if linear_proxy.nrows() != layout.reduced_dimension()
        || linear_proxy.ncols() != layout.reduced_dimension()
    {
        return Err(format!(
            "linear proxy dimensions {}x{} must match reduced dimension {}",
            linear_proxy.nrows(),
            linear_proxy.ncols(),
            layout.reduced_dimension()
        ));
    }

    let metric = coords.to_edge_lengths(topology);
    let full_hodge = build_hodge_laplacian_1form(topology, &metric);
    let full_mass_inverse = build_matern_mass_inverse_1form_with_coords(
        topology,
        coords,
        &metric,
        &full_hodge.mass_u,
        config.mass_inverse,
    )?;
    let hodge = HodgeLaplacian1Form {
        mass_u: restrict_square_with_layout(&full_hodge.mass_u, layout)?,
        laplacian: core_triplet_to_feec_csr(linear_proxy),
    };
    let mass_inverse = restrict_square_with_layout(&full_mass_inverse, layout)?;

    let mut kappas = vec![config.kappa];
    if config.allow_kappa_fallback {
        let h = bounding_box_diameter(coords).max(1e-12);
        let mut candidate = (1e-6 / h).max(config.kappa.max(0.0));
        while candidate < 1.0 {
            if !kappas
                .iter()
                .any(|kappa| (*kappa - candidate).abs() <= 1e-14 * candidate.max(1.0))
            {
                kappas.push(candidate);
            }
            candidate *= 10.0;
        }
        if !kappas.iter().any(|kappa| (*kappa - 1.0).abs() <= 1e-14) {
            kappas.push(1.0);
        }
    }

    let mut last_error = "precision matrix is not positive definite".to_string();
    for (attempt, kappa) in kappas.into_iter().enumerate() {
        let precision = build_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            MaternAlpha::Two,
            kappa,
            config.tau,
        );
        let precision = symmetrize_feec_csr(&precision);
        if !precision
            .triplet_iter()
            .all(|(_, _, value)| value.is_finite())
        {
            last_error = "linear proxy alpha-2 prior contains non-finite entries".to_string();
            continue;
        }
        match feec_csr_to_gmrf(&precision).cholesky_sqrt_lower() {
            Ok(_) => {
                return Ok(ReducedLinearProxyMaternAlpha2Prior {
                    spec: GaussianPriorSpec {
                        mean,
                        precision: feec_csr_to_core_triplet(&precision),
                    },
                    kappa,
                    tau: config.tau,
                    kappa_fallback_used: attempt > 0,
                });
            }
            Err(err) => {
                last_error = err.to_string();
            }
        }
    }

    Err(format!(
        "linear proxy alpha-2 prior did not factorize for requested kappa candidates: {last_error}"
    ))
}

pub fn build_reduced_linear_proxy_matern_alpha2_prior_from_reduced_matrices(
    linear_proxy: &SparseTripletMatrix,
    mass: &SparseTripletMatrix,
    mass_inverse: &SparseTripletMatrix,
    mean: Vec<f64>,
    config: ReducedLinearProxyMaternAlpha2Config,
) -> Result<ReducedLinearProxyMaternAlpha2Prior, String> {
    if !config.kappa.is_finite() || config.kappa < 0.0 {
        return Err("linear proxy kappa must be finite and nonnegative".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("linear proxy tau must be finite and positive".to_string());
    }
    let dimension = linear_proxy.nrows();
    if linear_proxy.ncols() != dimension {
        return Err(format!(
            "linear proxy must be square, got {}x{}",
            linear_proxy.nrows(),
            linear_proxy.ncols()
        ));
    }
    if mean.len() != dimension {
        return Err(format!(
            "linear proxy prior mean length {} must match reduced dimension {}",
            mean.len(),
            dimension
        ));
    }
    if mass.nrows() != dimension || mass.ncols() != dimension {
        return Err(format!(
            "linear proxy mass dimensions {}x{} must match reduced dimension {}",
            mass.nrows(),
            mass.ncols(),
            dimension
        ));
    }
    if mass_inverse.nrows() != dimension || mass_inverse.ncols() != dimension {
        return Err(format!(
            "linear proxy mass inverse dimensions {}x{} must match reduced dimension {}",
            mass_inverse.nrows(),
            mass_inverse.ncols(),
            dimension
        ));
    }

    let hodge = HodgeLaplacian1Form {
        mass_u: core_triplet_to_feec_csr(mass),
        laplacian: core_triplet_to_feec_csr(linear_proxy),
    };
    let mass_inverse = core_triplet_to_feec_csr(mass_inverse);
    let mut kappas = vec![config.kappa];
    if config.allow_kappa_fallback {
        let mut candidate = config.kappa.max(1e-6);
        while candidate < 1.0 {
            candidate *= 10.0;
            kappas.push(candidate);
        }
        if !kappas.iter().any(|kappa| (*kappa - 1.0).abs() <= 1e-14) {
            kappas.push(1.0);
        }
    }

    let mut last_error = "precision matrix is not positive definite".to_string();
    for (attempt, kappa) in kappas.into_iter().enumerate() {
        let precision = build_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            MaternAlpha::Two,
            kappa,
            config.tau,
        );
        let precision = symmetrize_feec_csr(&precision);
        if !precision
            .triplet_iter()
            .all(|(_, _, value)| value.is_finite())
        {
            last_error = "linear proxy alpha-2 prior contains non-finite entries".to_string();
            continue;
        }
        match feec_csr_to_gmrf(&precision).cholesky_sqrt_lower() {
            Ok(_) => {
                return Ok(ReducedLinearProxyMaternAlpha2Prior {
                    spec: GaussianPriorSpec {
                        mean,
                        precision: feec_csr_to_core_triplet(&precision),
                    },
                    kappa,
                    tau: config.tau,
                    kappa_fallback_used: attempt > 0,
                });
            }
            Err(err) => last_error = err.to_string(),
        }
    }

    Err(format!(
        "linear proxy alpha-2 prior did not factorize for requested kappa candidates: {last_error}"
    ))
}

fn bounding_box_diameter(coords: &MeshCoords) -> f64 {
    if coords.nvertices() == 0 {
        return 1.0;
    }
    let first = coords.coord(0);
    let mut min = vec![0.0; coords.dim()];
    let mut max = vec![0.0; coords.dim()];
    for d in 0..coords.dim() {
        min[d] = first[d];
        max[d] = first[d];
    }
    for vertex in 1..coords.nvertices() {
        let point = coords.coord(vertex);
        for d in 0..coords.dim() {
            min[d] = min[d].min(point[d]);
            max[d] = max[d].max(point[d]);
        }
    }
    min.iter()
        .zip(max.iter())
        .map(|(min, max)| (max - min).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn build_reconstructed_barycenter_field_operator(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<ReconstructedBarycenterFieldOperator, String> {
    let topo_dim = topology.dim();
    if topo_dim == 0 {
        return Err("topology dimension must be at least 1 to reconstruct a 1-form".to_string());
    }
    if topo_dim > coords.dim() {
        return Err(format!(
            "invalid mesh dimensions: topology dim {} > coordinate dim {}",
            topo_dim,
            coords.dim()
        ));
    }

    let cell_skeleton = topology.skeleton(topo_dim);
    let bary_local = barycenter_local(topo_dim);
    let edge_count = topology.skeleton(1).len();
    let ambient_dim = coords.dim();
    let mut component_rows = vec![Vec::with_capacity(cell_skeleton.len()); ambient_dim];

    for cell in cell_skeleton.handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let mut cell_component_rows = vec![Vec::new(); ambient_dim];

        for dof_simp in cell.mesh_subsimps(1) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let ambient_value = cell_coords
                .lift_form(&lsf.at_point(&bary_local))
                .into_grade1();

            for component_index in 0..ambient_dim {
                let coefficient = ambient_value[component_index];
                if coefficient.abs() > RECONSTRUCTION_EPS {
                    cell_component_rows[component_index].push((dof_simp.kidx(), coefficient));
                }
            }
        }

        for component_index in 0..ambient_dim {
            component_rows[component_index]
                .push(std::mem::take(&mut cell_component_rows[component_index]));
        }
    }

    let component_operators = component_rows
        .into_iter()
        .map(|rows| SparseRowLinearOperator::new(edge_count, rows).map_err(|err| err.to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ReconstructedBarycenterFieldOperator {
        ambient_dim,
        component_operators,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddf::cochain::Cochain;
    use formoniq::io::write_1form_vector_field_vtk;
    use manifold::gen::cartesian::CartesianMeshInfo;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn diagonal_entries(mat: &FeecCsr) -> Vec<f64> {
        let mut diag = vec![0.0; mat.nrows()];
        for (row, col, value) in mat.triplet_iter() {
            if row == col {
                diag[row] += *value;
            }
        }
        diag
    }

    fn max_abs_entry_diff(lhs: &FeecCsr, rhs: &FeecCsr) -> f64 {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());

        let mut entries = HashMap::new();
        for (row, col, value) in lhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) += *value;
        }
        for (row, col, value) in rhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) -= *value;
        }
        entries
            .values()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    }

    fn parse_vtk_vectors(content: &str, field_name: &str, count: usize) -> Vec<[f64; 3]> {
        let header = format!("VECTORS {field_name} double");
        let start = content
            .lines()
            .position(|line| line.trim() == header)
            .expect("vector header should be present")
            + 1;
        content
            .lines()
            .skip(start)
            .take(count)
            .map(|line| {
                let values = line
                    .split_whitespace()
                    .map(|token| token.parse::<f64>().expect("vector component should parse"))
                    .collect::<Vec<_>>();
                assert_eq!(values.len(), 3);
                [values[0], values[1], values[2]]
            })
            .collect()
    }

    fn constant_covector_cochain(
        topology: &Complex,
        coords: &MeshCoords,
        component_index: usize,
    ) -> FeecVector {
        let mut coefficients = FeecVector::zeros(topology.skeleton(1).len());
        for edge in topology.skeleton(1).handle_iter() {
            let tail = coords.coord(edge.vertices[0]);
            let head = coords.coord(edge.vertices[1]);
            coefficients[edge.kidx()] = head[component_index] - tail[component_index];
        }
        coefficients
    }

    #[test]
    fn hodge_laplacian_1form_dimensions() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        assert!(hodge.mass_u.nrows() > 0);
        assert_eq!(hodge.mass_u.nrows(), hodge.mass_u.ncols());
        assert_eq!(hodge.laplacian.nrows(), hodge.mass_u.nrows());
        assert_eq!(hodge.laplacian.ncols(), hodge.mass_u.ncols());
    }

    #[test]
    fn exact_dense_mass_inverse_1form_inverts_mass_matrix() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        let inverse = build_exact_dense_mass_inverse_1form(&hodge.mass_u, 0.0)
            .expect("exact 1-form mass inverse should build");
        let product = &hodge.mass_u * &inverse;
        let identity = diag_matrix(&vec![1.0; hodge.mass_u.nrows()]);

        assert!(max_abs_entry_diff(&product, &identity) < 1e-10);
    }

    #[test]
    fn matern_precision_has_positive_diagonal_with_row_sum_lumping() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let precision = build_matern_precision_1form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );

        assert!(diagonal_entries(&precision).iter().all(|v| *v > 0.0));
    }

    #[test]
    fn matern_precision_projected_sparse_inverse_differs_from_row_sum() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        let row_sum = build_matern_precision_1form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let projected = build_matern_precision_1form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
            },
        );

        assert_eq!(row_sum.nrows(), projected.nrows());
        assert_eq!(row_sum.ncols(), projected.ncols());
        assert!(diagonal_entries(&projected).iter().all(|v| *v > 0.0));

        let row_sum_gmrf = feec_csr_to_gmrf(&row_sum);
        let projected_gmrf = feec_csr_to_gmrf(&projected);
        row_sum_gmrf
            .cholesky_sqrt_lower()
            .expect("row-sum precision should factorize");
        projected_gmrf
            .cholesky_sqrt_lower()
            .expect("projected precision should factorize");

        assert!(max_abs_entry_diff(&row_sum, &projected) > 1e-9);
    }

    #[test]
    fn reduced_linear_proxy_alpha2_prior_builds_with_projected_sparse_inverse() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let active_dofs = (0..hodge.laplacian.nrows())
            .filter(|index| index % 2 == 0)
            .collect::<Vec<_>>();
        let prescribed_dofs = (0..hodge.laplacian.nrows())
            .filter(|index| index % 2 != 0)
            .map(|index| formoniq::reduction::PrescribedDof { index, value: 0.0 })
            .collect();
        let layout = DofLayout::new(hodge.laplacian.nrows(), active_dofs, prescribed_dofs);
        let reduced_proxy = restrict_square_with_layout(&hodge.laplacian, &layout)
            .expect("test proxy should reduce to active dofs");
        let mean = vec![0.0; layout.reduced_dimension()];

        let prior = build_reduced_linear_proxy_matern_alpha2_prior(
            &topology,
            &coords,
            &layout,
            &feec_csr_to_core_triplet(&reduced_proxy),
            mean,
            ReducedLinearProxyMaternAlpha2Config {
                kappa: 0.25,
                tau: 0.75,
                mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
                allow_kappa_fallback: false,
            },
        )
        .expect("reduced alpha-2 linear proxy prior should build");

        assert_eq!(prior.kappa, 0.25);
        assert!(!prior.kappa_fallback_used);
        assert_eq!(prior.spec.precision.nrows(), layout.reduced_dimension());
        assert_eq!(prior.spec.precision.ncols(), layout.reduced_dimension());
        assert_eq!(prior.spec.mean.len(), layout.reduced_dimension());
        feec_csr_to_gmrf(&core_triplet_to_feec_csr(&prior.spec.precision))
            .cholesky_sqrt_lower()
            .expect("reduced projected sparse inverse prior should factorize");
    }

    #[test]
    fn matern_precision_barycentric_dual_sparse_inverse_factorizes_and_differs_from_projected() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        let projected = build_matern_precision_1form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
            },
        );
        let barycentric = build_matern_precision_1form_with_coords(
            &topology,
            &coords,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::BarycentricDualSparseInverse,
            },
        )
        .expect("barycentric-dual 1-form precision should build with coordinates");

        assert_eq!(projected.nrows(), barycentric.nrows());
        assert_eq!(projected.ncols(), barycentric.ncols());
        assert!(diagonal_entries(&barycentric).iter().all(|v| *v > 0.0));
        feec_csr_to_gmrf(&barycentric)
            .cholesky_sqrt_lower()
            .expect("barycentric-dual 1-form precision should factorize");
        assert!(max_abs_entry_diff(&projected, &barycentric) > 1e-9);
    }

    #[test]
    fn alpha_one_precision_is_scaled_whittle_system_matrix() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let kappa = 1.5;
        let tau = 0.75;
        let mass_inverse = build_matern_mass_inverse_1form(
            &topology,
            &metric,
            &hodge.mass_u,
            MaternMassInverse::RowSumLumped,
        );

        let precision = build_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            MaternAlpha::One,
            kappa,
            tau,
        );
        let expected = scale_matrix(&build_matern_system_matrix_1form(&hodge, kappa), tau * tau);

        assert!(max_abs_entry_diff(&precision, &expected) <= 1e-12);
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("alpha=1 Whittle system precision should factorize");
    }

    #[test]
    fn alpha_two_precision_matches_legacy_builder() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let config = MaternConfig {
            kappa: 1.5,
            tau: 0.75,
            mass_inverse: MaternMassInverse::RowSumLumped,
        };

        let legacy = build_matern_precision_1form(&topology, &metric, &hodge, config);
        let alpha_two = build_matern_precision_1form_for_alpha(
            &topology,
            &metric,
            &hodge,
            MaternAlpha::Two,
            config,
        );

        assert!(max_abs_entry_diff(&legacy, &alpha_two) <= 1e-12);
    }

    #[test]
    fn precision_from_supplied_mass_inverse_matches_enum_path() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let config = MaternConfig {
            kappa: 1.5,
            tau: 0.75,
            mass_inverse: MaternMassInverse::RowSumLumped,
        };

        let from_enum = build_matern_precision_1form(&topology, &metric, &hodge, config);
        let mass_inverse = build_matern_mass_inverse_1form(
            &topology,
            &metric,
            &hodge.mass_u,
            MaternMassInverse::RowSumLumped,
        );
        let from_supplied = build_matern_precision_1form_with_mass_inverse(
            &hodge,
            &mass_inverse,
            config.kappa,
            config.tau,
        );

        assert!(max_abs_entry_diff(&from_enum, &from_supplied) <= 1e-12);
    }

    #[test]
    fn split_graph_precision_drops_cross_block_couplings_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let mass_inverse = build_matern_mass_inverse_1form(
            &topology,
            &metric,
            &hodge.mass_u,
            MaternMassInverse::RowSumLumped,
        );
        let dimension = hodge.mass_u.nrows();
        let groups = vec![
            (0..dimension)
                .filter(|index| index % 2 == 0)
                .collect::<Vec<_>>(),
            (0..dimension)
                .filter(|index| index % 2 == 1)
                .collect::<Vec<_>>(),
        ];

        let precision = build_split_graph_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            &groups,
            MaternAlpha::Two,
            1.0,
            0.5,
        )
        .expect("split-graph precision should build");
        let group_index = split_graph_test_group_index(dimension, &groups);

        assert_eq!(precision.nrows(), dimension);
        assert_eq!(precision.ncols(), dimension);
        for (row, col, value) in precision.triplet_iter() {
            if group_index[row] != group_index[col] {
                assert_eq!(*value, 0.0);
            }
        }
        feec_csr_to_gmrf(&symmetrize_feec_csr(&precision))
            .cholesky_sqrt_lower()
            .expect("split-graph precision should factorize");
    }

    #[test]
    fn split_graph_precision_rejects_non_partition_groups() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let mass_inverse = build_matern_mass_inverse_1form(
            &topology,
            &metric,
            &hodge.mass_u,
            MaternMassInverse::RowSumLumped,
        );

        let duplicate = build_split_graph_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            &[vec![0], vec![0]],
            MaternAlpha::Two,
            1.0,
            1.0,
        )
        .unwrap_err();
        assert!(duplicate.contains("appears"));

        let uncovered = build_split_graph_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            &[vec![0]],
            MaternAlpha::Two,
            1.0,
            1.0,
        )
        .unwrap_err();
        assert!(uncovered.contains("not covered"));
    }

    #[test]
    fn hodge_laplacian_1form_torus_mesh_dimensions() {
        let mesh_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../meshes/torus_shell_resolution_1.msh");
        let mesh_bytes = std::fs::read(mesh_path).expect("Failed to read torus mesh");
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        assert_eq!(topology.dim(), 2);
        assert_eq!(coords.dim(), 3);
        assert!(hodge.mass_u.nrows() > 0);
        assert_eq!(hodge.mass_u.nrows(), hodge.mass_u.ncols());
        assert_eq!(hodge.laplacian.nrows(), hodge.mass_u.nrows());
        assert_eq!(hodge.laplacian.ncols(), hodge.mass_u.ncols());
    }

    #[test]
    fn reconstructed_barycenter_field_trace_sums_component_fields() {
        let field = ReconstructedBarycenterField::from_components(vec![
            FeecVector::from_vec(vec![1.0, 2.0]),
            FeecVector::from_vec(vec![0.5, 1.5]),
            FeecVector::from_vec(vec![2.0, 3.0]),
        ])
        .expect("field components should be compatible");

        let trace = field.trace();
        assert_eq!(field.ambient_dim(), 3);
        assert_eq!(field.cell_count(), 2);
        assert!((trace[0] - 3.5).abs() < 1e-12);
        assert!((trace[1] - 6.5).abs() < 1e-12);
    }

    #[test]
    fn reconstructed_barycenter_operator_lifts_full_dimensional_constant_covectors() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let operator = build_reconstructed_barycenter_field_operator(&topology, &coords)
            .expect("reconstruction operator should build");

        assert_eq!(topology.dim(), 3);
        assert_eq!(coords.dim(), 3);
        assert_eq!(operator.ambient_dim(), 3);
        assert_eq!(operator.cell_count(), topology.cells().len());

        for covector_component in 0..3 {
            let coefficients = constant_covector_cochain(&topology, &coords, covector_component);
            let field = operator
                .apply_to_slice(coefficients.as_slice())
                .expect("operator application should succeed");

            for output_component in 0..3 {
                let expected = if output_component == covector_component {
                    1.0
                } else {
                    0.0
                };
                let values = field
                    .component(output_component)
                    .expect("component should exist");
                for cell_index in 0..field.cell_count() {
                    assert!(
                        (values[cell_index] - expected).abs() < 1e-10,
                        "component {output_component}, cell {cell_index}: expected {expected}, got {}",
                        values[cell_index]
                    );
                }
            }
        }
    }

    #[test]
    fn reconstructed_barycenter_operator_matches_formoniq_vector_writer_on_torus() {
        let mesh_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../meshes/torus_shell_resolution_1.msh");
        let mesh_bytes = std::fs::read(mesh_path).expect("failed to read torus mesh");
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);

        let operator = build_reconstructed_barycenter_field_operator(&topology, &coords)
            .expect("reconstruction operator should build");
        assert_eq!(operator.ambient_dim(), coords.dim());
        assert_eq!(operator.cell_count(), topology.cells().len());

        let edge_count = topology.skeleton(1).len();
        let coefficients =
            FeecVector::from_iterator(edge_count, (0..edge_count).map(|i| ((i + 1) as f64).sin()));
        let field = operator
            .apply_to_slice(coefficients.as_slice())
            .expect("operator application should succeed");
        let expected_vectors = field.vtk_vectors();

        let cochain = Cochain::new(1, coefficients);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "reconstructed_barycenter_operator_matches_formoniq_{stamp}.vtk"
        ));
        write_1form_vector_field_vtk(&path, &coords, &topology, &cochain, "embedded")
            .expect("vector field VTK should write");

        let content = std::fs::read_to_string(&path).expect("VTK should be readable");
        let _ = std::fs::remove_file(&path);
        let actual_vectors = parse_vtk_vectors(&content, "embedded", topology.cells().len());

        assert_eq!(expected_vectors.len(), actual_vectors.len());
        for (expected, actual) in expected_vectors.iter().zip(actual_vectors.iter()) {
            for component in 0..3 {
                assert!(
                    (expected[component] - actual[component]).abs() < 1e-10,
                    "component mismatch: expected {} got {}",
                    expected[component],
                    actual[component]
                );
            }
        }
    }

    #[test]
    fn nonlinear_team13_linear_proxy_prior_factorizes_with_inverse_diameter_kappa() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let layout = DofLayout::identity(topology.nsimplices(1));
        let kappa = inverse_domain_diameter_matern_kappa(&coords);
        let prior = build_reduced_linear_proxy_matern_alpha2_prior(
            &topology,
            &coords,
            &layout,
            &crate::sparse::feec_csr_to_core_triplet(&hodge.laplacian),
            vec![0.0; layout.reduced_dimension()],
            ReducedLinearProxyMaternAlpha2Config {
                kappa,
                tau: 1.0,
                allow_kappa_fallback: false,
                ..ReducedLinearProxyMaternAlpha2Config::default()
            },
        )
        .expect("TEAM13-style alpha-2 linear proxy prior should factorize");

        assert_eq!(prior.spec.mean.len(), layout.reduced_dimension());
        assert!((prior.kappa - kappa).abs() <= 1e-14);
        assert!(!prior.kappa_fallback_used);
        assert!(prior.spec.precision.nnz() > 0);
    }

    #[test]
    fn nonlinear_team13_reduced_matrix_linear_proxy_prior_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let mass_inverse = build_matern_mass_inverse_1form_with_coords(
            &topology,
            &coords,
            &metric,
            &hodge.mass_u,
            MaternMassInverse::Nc1ProjectedSparseInverse,
        )
        .expect("projected sparse inverse should build");
        let dimension = hodge.mass_u.nrows();
        let prior = build_reduced_linear_proxy_matern_alpha2_prior_from_reduced_matrices(
            &crate::sparse::feec_csr_to_core_triplet(&hodge.laplacian),
            &crate::sparse::feec_csr_to_core_triplet(&hodge.mass_u),
            &crate::sparse::feec_csr_to_core_triplet(&mass_inverse),
            vec![0.0; dimension],
            ReducedLinearProxyMaternAlpha2Config {
                kappa: 10.0,
                tau: 1e-6,
                allow_kappa_fallback: false,
                ..ReducedLinearProxyMaternAlpha2Config::default()
            },
        )
        .expect("reduced-matrix alpha-2 linear proxy prior should factorize");

        assert_eq!(prior.spec.mean.len(), dimension);
        assert_eq!(prior.kappa, 10.0);
        assert_eq!(prior.tau, 1e-6);
        assert!(!prior.kappa_fallback_used);
        assert!(prior.spec.precision.nnz() > 0);
    }

    fn split_graph_test_group_index(dimension: usize, groups: &[Vec<usize>]) -> Vec<usize> {
        let mut group_index = vec![usize::MAX; dimension];
        for (group, indices) in groups.iter().enumerate() {
            for &index in indices {
                group_index[index] = group;
            }
        }
        group_index
    }
}
