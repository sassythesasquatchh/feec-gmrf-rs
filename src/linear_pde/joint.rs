//! Independent state/input priors and observations on their joint coordinates.
//!
//! This is a typed facade over the existing sparse Gaussian conditioner. Means
//! stay in physical coordinates, including nonzero input means; no implicit
//! centering or conditioning on a boundary lifting is performed.

use crate::infer::Posterior;
use crate::model::{DerivedQuantity, GaussianNoise, LinearGaussianModelBuilder, LinearObservation};
use crate::operator::{LinearMap, SparseMat};
use crate::prior::GaussianPrior;
use crate::{FeecGmrfError, Result};

/// Concatenate maps acting on disjoint independent latent blocks.
pub fn joint_block_map(blocks: &[&LinearMap]) -> Result<LinearMap> {
    let Some(first) = blocks.first() else {
        return Err(FeecGmrfError::Dimension(
            "joint map needs at least one block".into(),
        ));
    };
    let rows = first.output_dimension();
    if blocks.iter().any(|b| b.output_dimension() != rows) {
        return Err(FeecGmrfError::Dimension(
            "joint map blocks must have equal output dimensions".into(),
        ));
    }
    let mut matrix = SparseMat::new(rows, blocks.iter().map(|b| b.input_dimension()).sum());
    let mut offset = 0;
    for block in blocks {
        for (r, c, v) in block.matrix().triplet_iter() {
            matrix.push(r, c + offset, v);
        }
        offset += block.input_dimension();
    }
    LinearMap::new(matrix)
}

/// Joint linear PDE model with an independent Gaussian state and uncertain input.
///
/// Boundary elimination belongs to the state prior and the caller's maps. The
/// returned posterior has coordinates `[state, input]`, not a full FEEC cochain.
/// Use an explicit total-field map such as `[P, G]` for physical outputs.
pub struct JointLinearPdeModelBuilder {
    model: LinearGaussianModelBuilder,
    state_dimension: usize,
    input_dimension: usize,
}

impl JointLinearPdeModelBuilder {
    pub fn new(state: &GaussianPrior, input: &GaussianPrior) -> Result<Self> {
        let state_precision = feg_infer::sparse::core_triplet_to_feec_csr(state.precision());
        let input_precision = feg_infer::sparse::core_triplet_to_feec_csr(input.precision());
        let precision =
            feg_infer::sparse::block_diag_feec_csr(&[&state_precision, &input_precision]);
        let mean = state.mean().iter().chain(input.mean()).copied().collect();
        let prior = GaussianPrior::new(
            mean,
            feg_infer::sparse::feec_csr_to_core_triplet(&precision),
        )?;
        Ok(Self {
            model: LinearGaussianModelBuilder::new(prior),
            state_dimension: state.dimension(),
            input_dimension: input.dimension(),
        })
    }

    pub fn joint_map(&self, state: &LinearMap, input: &LinearMap) -> Result<LinearMap> {
        if state.input_dimension() != self.state_dimension
            || input.input_dimension() != self.input_dimension
        {
            return Err(FeecGmrfError::Dimension(
                "state/input map does not match its prior block".into(),
            ));
        }
        joint_block_map(&[state, input])
    }

    /// Observe `state_map*z + input_map*xi`, with arbitrary correlated noise.
    pub fn observe(
        mut self,
        state_map: &LinearMap,
        input_map: &LinearMap,
        values: Vec<f64>,
        noise: GaussianNoise,
    ) -> Result<Self> {
        let map = self.joint_map(state_map, input_map)?;
        self.model = self
            .model
            .observe(LinearObservation::new(map, values, noise)?)?;
        Ok(self)
    }

    /// Add finite-precision weak PDE information `0 = A*z + B*xi + eps`.
    pub fn observe_pde(
        self,
        state: &LinearMap,
        input: &LinearMap,
        noise: GaussianNoise,
    ) -> Result<Self> {
        self.observe(state, input, vec![0.0; state.output_dimension()], noise)
    }

    pub fn derive(
        mut self,
        name: impl Into<String>,
        state: &LinearMap,
        input: &LinearMap,
    ) -> Result<Self> {
        let map = self.joint_map(state, input)?;
        self.model = self.model.derive(DerivedQuantity::new(name, map)?)?;
        Ok(self)
    }

    pub fn condition(self) -> Result<Posterior> {
        self.model.condition()
    }
}
