//! Exact Matérn laws represented by sparse energies on constrained supports.
//!
//! These types preserve the local FEEC blocks and use a certified indefinite
//! KKT solve. The KKT matrix is not exposed as an ordinary Gaussian precision.

use super::{
    assemble_exact_matern_blocks, exact_matern_metadata, ExactMaternBlocks, InverseUse,
    MassInverseKind, MaternParameters, MaternPriorMetadata, MaternRepresentation,
};
use crate::boundary::{EliminatedLinearMap, EssentialBoundaryElimination};
use crate::model::{GaussianNoise, LinearConstraint, LinearObservation};
use crate::operator::{BoundaryLayout, ExactFormOperators, FormDegree, LinearMap, SparseMat};
use crate::{FeecGmrfError, Result};
use feg_core::SparseTriplet;
use gmrf_core::{
    linear_observation_update, LinearObservationTerm, SelectedCoordinates, SparseRowOperator,
    SupportedGaussianCertificateResiduals, SupportedGaussianFactor, SupportedGaussianSystem,
    Vector,
};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};
use std::ops::Range;

/// Ambient block ordering of an exact supported Matérn construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportedMaternLayout {
    state: Range<usize>,
    lower_adjoint_auxiliaries: Vec<Range<usize>>,
    recurrence_states: Vec<Range<usize>>,
    ambient_dimension: usize,
    constraint_dimension: usize,
}

impl SupportedMaternLayout {
    pub fn state(&self) -> Range<usize> {
        self.state.clone()
    }

    pub fn lower_adjoint_auxiliaries(&self) -> &[Range<usize>] {
        &self.lower_adjoint_auxiliaries
    }

    pub fn recurrence_states(&self) -> &[Range<usize>] {
        &self.recurrence_states
    }

    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    pub fn constraint_dimension(&self) -> usize {
        self.constraint_dimension
    }

    pub fn support_dimension(&self) -> usize {
        self.ambient_dimension - self.constraint_dimension
    }
}

/// Setup and solve diagnostics for a certified supported law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportedMaternDiagnostics {
    pub ambient_dimension: usize,
    pub support_dimension: usize,
    pub constraint_dimension: usize,
    pub ambient_energy_nnz: usize,
    pub constraint_nnz: usize,
    pub constraint_gram_factor_nnz: usize,
    pub supported_energy_factor_nnz: usize,
    pub kkt_nnz: usize,
    pub certificate_residuals: SupportedGaussianCertificateResiduals,
}

/// Geometry-independent builder for the exact supported FEEC Matérn law.
pub struct SupportedMaternPriorBuilder {
    operators: ExactFormOperators,
    parameters: MaternParameters,
}

impl SupportedMaternPriorBuilder {
    pub fn from_operators(operators: ExactFormOperators) -> Self {
        Self {
            operators,
            parameters: MaternParameters::default(),
        }
    }

    /// Assemble compatible lowest-order Whitney spaces on the homogeneous
    /// full-relative subcomplex before constructing sparse graph constraints.
    pub fn from_feec_homogeneous_full_relative(
        topology: &Complex,
        metric: &MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        Ok(Self::from_operators(
            ExactFormOperators::from_feec_homogeneous_full_relative(topology, metric, degree)?,
        ))
    }

    pub fn parameters(mut self, parameters: MaternParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Build and certify the exact sparse supported representation.
    pub fn build(self) -> Result<SupportedMaternPrior> {
        let blocks = assemble_exact_matern_blocks(&self.operators, self.parameters.kappa)?;
        let (energy, constraints, layout) =
            assemble_supported_matern_graph(&blocks, self.parameters)?;
        let system = SupportedGaussianSystem::new(
            feg_infer::sparse::core_triplet_to_gmrf(&energy),
            feg_infer::sparse::core_triplet_to_gmrf(&constraints),
        )?;
        if system.support_dim() != self.operators.state().dimension() {
            return Err(FeecGmrfError::Assembly(format!(
                "supported Matérn graph has support dimension {}, expected state dimension {}",
                system.support_dim(),
                self.operators.state().dimension()
            )));
        }
        let metadata = exact_supported_metadata(&self.operators, self.parameters);
        Ok(SupportedMaternPrior {
            system,
            layout,
            state_layout: self.operators.state().boundary_layout().clone(),
            form_degree: self.operators.degree(),
            metadata,
        })
    }
}

/// Certified exact Matérn probability law on a sparse constrained support.
#[derive(Clone, Debug)]
pub struct SupportedMaternPrior {
    system: SupportedGaussianSystem,
    layout: SupportedMaternLayout,
    state_layout: BoundaryLayout,
    form_degree: FormDegree,
    metadata: MaternPriorMetadata,
}

impl SupportedMaternPrior {
    pub fn state_dimension(&self) -> usize {
        self.layout.state.len()
    }

    pub fn cochain_dimension(&self) -> usize {
        self.state_layout.full_dimension()
    }

    pub fn form_degree(&self) -> FormDegree {
        self.form_degree
    }

    pub fn layout(&self) -> &SupportedMaternLayout {
        &self.layout
    }

    pub fn matern_metadata(&self) -> &MaternPriorMetadata {
        &self.metadata
    }

    /// Factor the prior support equations without adding observations.
    pub fn factor(&self) -> Result<SupportedMaternPosterior> {
        self.finish_factorization(
            self.system.factorize()?,
            Vector::zeros(self.layout.ambient_dimension),
            self.system.energy().nnz(),
            self.system.constraints().nnz(),
        )
    }

    /// Condition state-coordinate observations and append exact state constraints.
    pub fn condition(
        &self,
        observations: &[LinearObservation],
        constraints: &[LinearConstraint],
    ) -> Result<SupportedMaternPosterior> {
        let ambient_dimension = self.layout.ambient_dimension;
        let mut energy = self.system.energy().clone();
        let mut information = Vector::zeros(ambient_dimension);
        for observation in observations {
            let eliminated = resolve_state_map(
                &self.state_layout,
                self.state_dimension(),
                &observation.operator,
                &observation.bias,
            )?;
            let ambient_map = embed_state_map(eliminated.reduced(), ambient_dimension)?;
            let matrix = feg_infer::sparse::core_triplet_to_gmrf(&ambient_map);
            let values = Vector::from_vec(observation.values.clone());
            let bias = Vector::from_vec(eliminated.fixed_offset().to_vec());
            let (update, update_information) = match &observation.noise {
                GaussianNoise::ScalarVariance(variance) => {
                    linear_observation_update(&LinearObservationTerm::scalar_variance(
                        &matrix,
                        &values,
                        Some(&bias),
                        *variance,
                    ))
                }
                GaussianNoise::Precision(precision) => {
                    let precision = feg_infer::sparse::core_triplet_to_gmrf(precision);
                    linear_observation_update(&LinearObservationTerm::precision(
                        &matrix,
                        &values,
                        Some(&bias),
                        &precision,
                    ))
                }
            };
            energy = gmrf_core::add_sparse(&energy, &update);
            information += update_information;
        }

        let (support_constraints, affine_rhs) = append_state_constraints(
            self.system.constraints(),
            self.system.affine_rhs(),
            &self.state_layout,
            self.state_dimension(),
            ambient_dimension,
            constraints,
        )?;
        let posterior_system =
            SupportedGaussianSystem::new_affine(energy, support_constraints, affine_rhs)?;
        let energy_nnz = posterior_system.energy().nnz();
        let constraint_nnz = posterior_system.constraints().nnz();
        self.finish_factorization(
            posterior_system.factorize()?,
            information,
            energy_nnz,
            constraint_nnz,
        )
    }

    fn finish_factorization(
        &self,
        factor: SupportedGaussianFactor,
        information: Vector,
        energy_nnz: usize,
        constraint_nnz: usize,
    ) -> Result<SupportedMaternPosterior> {
        let ambient_dimension = factor.ambient_dim();
        let selection = SelectedCoordinates::prefix(ambient_dimension, self.state_dimension())?;
        let mean = factor.selected_mean(&selection, &information)?;
        let certificate = factor.certificate();
        let diagnostics = SupportedMaternDiagnostics {
            ambient_dimension,
            support_dimension: factor.support_dim(),
            constraint_dimension: factor.constraint_dim(),
            ambient_energy_nnz: energy_nnz,
            constraint_nnz,
            constraint_gram_factor_nnz: certificate.constraint_gram_factor_nnz(),
            supported_energy_factor_nnz: certificate.supported_energy_factor_nnz(),
            kkt_nnz: factor.kkt_nnz(),
            certificate_residuals: certificate.residuals(),
        };
        Ok(SupportedMaternPosterior {
            factor,
            selection,
            mean: mean.iter().copied().collect(),
            state_layout: self.state_layout.clone(),
            form_degree: self.form_degree,
            metadata: self.metadata.clone(),
            diagnostics,
        })
    }
}

/// Factored prior or linearly conditioned posterior of an exact supported law.
#[derive(Debug)]
pub struct SupportedMaternPosterior {
    factor: SupportedGaussianFactor,
    selection: SelectedCoordinates,
    mean: Vec<f64>,
    state_layout: BoundaryLayout,
    form_degree: FormDegree,
    metadata: MaternPriorMetadata,
    diagnostics: SupportedMaternDiagnostics,
}

impl SupportedMaternPosterior {
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn cochain_mean(&self) -> Result<Vec<f64>> {
        let elimination = EssentialBoundaryElimination::from_layout(self.state_layout.clone())?;
        elimination.lift_state(&self.mean)
    }

    pub fn form_degree(&self) -> FormDegree {
        self.form_degree
    }

    pub fn matern_metadata(&self) -> &MaternPriorMetadata {
        &self.metadata
    }

    pub fn diagnostics(&self) -> SupportedMaternDiagnostics {
        self.diagnostics
    }

    pub fn covariance_action(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        if rhs.len() != self.mean.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "selected covariance right-hand side length {} does not match state dimension {}",
                rhs.len(),
                self.mean.len()
            )));
        }
        let action = self
            .factor
            .selected_covariance_action(&self.selection, &Vector::from_vec(rhs.to_vec()))?;
        Ok(action.iter().copied().collect())
    }

    /// Exact variances of an active- or full-cochain linear map.
    pub fn pushforward_variances(&self, map: &LinearMap) -> Result<Vec<f64>> {
        let eliminated = resolve_state_map(
            &self.state_layout,
            self.mean.len(),
            map,
            &vec![0.0; map.output_dimension()],
        )?;
        let ambient_map = embed_state_map(eliminated.reduced(), self.factor.ambient_dim())?;
        let operator = SparseRowOperator::from_sparse_matrix(
            &feg_infer::sparse::core_triplet_to_gmrf(&ambient_map),
        )?;
        let variances = self.factor.exact_transformed_variances(&operator)?;
        Ok(variances.iter().copied().collect())
    }

    /// Sampling is not available until a support-aware square root is implemented.
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<f64>> {
        self.factor
            .sample(rng)
            .map(|sample| {
                self.selection
                    .indices()
                    .iter()
                    .map(|index| sample[*index])
                    .collect()
            })
            .map_err(Into::into)
    }
}

fn exact_supported_metadata(
    operators: &ExactFormOperators,
    parameters: MaternParameters,
) -> MaternPriorMetadata {
    exact_matern_metadata(
        operators,
        parameters,
        MaternRepresentation::ExactSupported,
        if operators.incoming().is_some() {
            InverseUse::Applied(MassInverseKind::ConstrainedSupport)
        } else {
            InverseUse::NotApplicable
        },
        if parameters.alpha == feg_core::MaternAlpha::One {
            InverseUse::NotUsed
        } else {
            InverseUse::Applied(MassInverseKind::ConstrainedSupport)
        },
    )
}

fn assemble_supported_matern_graph(
    blocks: &ExactMaternBlocks,
    parameters: MaternParameters,
) -> Result<(SparseMat, SparseMat, SupportedMaternLayout)> {
    let n = blocks.mass.nrows();
    let p = blocks.lower.as_ref().map_or(0, |lower| lower.mass.nrows());
    let tau2 = parameters.tau.powi(2);
    let (ambient, constraint_dimension, lower_ranges, recurrence_ranges) = match parameters.alpha {
        feg_core::MaternAlpha::One => (
            n + p,
            p,
            (p > 0).then_some(n..n + p).into_iter().collect(),
            Vec::new(),
        ),
        feg_core::MaternAlpha::Two => (
            2 * n + p,
            n + p,
            (p > 0).then_some(n..n + p).into_iter().collect(),
            std::iter::once(n + p..2 * n + p).collect(),
        ),
        feg_core::MaternAlpha::Three => (
            2 * n + 2 * p,
            n + 2 * p,
            if p > 0 {
                vec![n..n + p, 2 * n + p..2 * n + 2 * p]
            } else {
                Vec::new()
            },
            std::iter::once(n + p..2 * n + p).collect(),
        ),
    };
    let layout = SupportedMaternLayout {
        state: 0..n,
        lower_adjoint_auxiliaries: lower_ranges,
        recurrence_states: recurrence_ranges,
        ambient_dimension: ambient,
        constraint_dimension,
    };
    let mut energy = Vec::new();
    let mut constraints = Vec::new();
    let lower = blocks.lower.as_ref();

    match parameters.alpha {
        feg_core::MaternAlpha::One => {
            push_block(&mut energy, 0, 0, &blocks.local_system, tau2);
            if let Some(lower) = lower {
                push_block(&mut energy, n, n, &lower.mass, tau2);
                push_block(&mut constraints, 0, 0, &lower.coupling.transpose(), -1.0);
                push_block(&mut constraints, 0, n, &lower.mass, 1.0);
            }
        }
        feg_core::MaternAlpha::Two => {
            let y = n + p;
            push_block(&mut energy, y, y, &blocks.mass, tau2);
            if let Some(lower) = lower {
                push_block(&mut constraints, 0, 0, &lower.coupling.transpose(), -1.0);
                push_block(&mut constraints, 0, n, &lower.mass, 1.0);
            }
            push_block(&mut constraints, p, 0, &blocks.local_system, -1.0);
            if let Some(lower) = lower {
                push_block(&mut constraints, p, n, &lower.coupling, -1.0);
            }
            push_block(&mut constraints, p, y, &blocks.mass, 1.0);
        }
        feg_core::MaternAlpha::Three => {
            let y = n + p;
            let sigma_one = 2 * n + p;
            push_block(&mut energy, y, y, &blocks.local_system, tau2);
            if let Some(lower) = lower {
                push_block(&mut energy, sigma_one, sigma_one, &lower.mass, tau2);
                push_block(&mut constraints, 0, 0, &lower.coupling.transpose(), -1.0);
                push_block(&mut constraints, 0, n, &lower.mass, 1.0);
            }
            push_block(&mut constraints, p, 0, &blocks.local_system, -1.0);
            if let Some(lower) = lower {
                push_block(&mut constraints, p, n, &lower.coupling, -1.0);
            }
            push_block(&mut constraints, p, y, &blocks.mass, 1.0);
            if let Some(lower) = lower {
                push_block(
                    &mut constraints,
                    p + n,
                    y,
                    &lower.coupling.transpose(),
                    -1.0,
                );
                push_block(&mut constraints, p + n, sigma_one, &lower.mass, 1.0);
            }
        }
    }
    Ok((
        SparseMat::from_triplets(ambient, ambient, energy),
        SparseMat::from_triplets(constraint_dimension, ambient, constraints),
        layout,
    ))
}

fn push_block(
    output: &mut Vec<SparseTriplet>,
    row_offset: usize,
    column_offset: usize,
    block: &SparseMat,
    scale: f64,
) {
    output.extend(
        block
            .triplet_iter()
            .filter(|(_, _, value)| *value != 0.0)
            .map(|(row, column, value)| SparseTriplet {
                row: row_offset + row,
                col: column_offset + column,
                value: scale * value,
            }),
    );
}

fn resolve_state_map(
    layout: &BoundaryLayout,
    state_dimension: usize,
    map: &LinearMap,
    bias: &[f64],
) -> Result<EliminatedLinearMap> {
    if map.input_dimension() == state_dimension {
        if bias.len() != map.output_dimension() {
            return Err(FeecGmrfError::Dimension(
                "map bias length must match map output dimension".to_string(),
            ));
        }
        return Ok(EliminatedLinearMap::from_active(map.clone(), bias.to_vec()));
    }
    if map.input_dimension() != layout.full_dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "map input dimension {} matches neither active dimension {state_dimension} nor full cochain dimension {}",
            map.input_dimension(),
            layout.full_dimension()
        )));
    }
    EssentialBoundaryElimination::from_layout(layout.clone())?.eliminate_map(map, bias)
}

fn embed_state_map(map: &LinearMap, ambient_dimension: usize) -> Result<SparseMat> {
    if map.input_dimension() > ambient_dimension {
        return Err(FeecGmrfError::Dimension(
            "state map cannot be embedded into a smaller ambient space".to_string(),
        ));
    }
    Ok(SparseMat::from_triplets(
        map.output_dimension(),
        ambient_dimension,
        map.matrix()
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet { row, col, value }),
    ))
}

fn append_state_constraints(
    base: &gmrf_core::SparseMatrix,
    base_rhs: &Vector,
    layout: &BoundaryLayout,
    state_dimension: usize,
    ambient_dimension: usize,
    additions: &[LinearConstraint],
) -> Result<(gmrf_core::SparseMatrix, Vector)> {
    let added_rows = additions
        .iter()
        .map(|constraint| constraint.operator.output_dimension())
        .sum::<usize>();
    if added_rows == 0 {
        return Ok((base.clone(), base_rhs.clone()));
    }
    let mut matrix = SparseMat::new(base.nrows() + added_rows, ambient_dimension);
    for (row, col, value) in base.triplet_iter() {
        matrix.push(row, col, *value);
    }
    let mut rhs = base_rhs.iter().copied().collect::<Vec<_>>();
    let mut row_offset = base.nrows();
    for constraint in additions {
        let zero_bias = vec![0.0; constraint.operator.output_dimension()];
        let eliminated =
            resolve_state_map(layout, state_dimension, &constraint.operator, &zero_bias)?;
        let embedded = embed_state_map(eliminated.reduced(), ambient_dimension)?;
        for (row, col, value) in embedded.triplet_iter() {
            matrix.push(row_offset + row, col, value);
        }
        rhs.extend(
            constraint
                .target
                .iter()
                .zip(eliminated.fixed_offset())
                .map(|(target, offset)| target - offset),
        );
        row_offset += constraint.operator.output_dimension();
    }
    Ok((
        feg_infer::sparse::core_triplet_to_gmrf(&matrix),
        Vector::from_vec(rhs),
    ))
}
