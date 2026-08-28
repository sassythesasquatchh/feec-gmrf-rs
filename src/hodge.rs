//! First-class root façade for sparse 1-form Hodge-decomposed Gaussian priors.

use crate::infer::{Posterior, VarianceEstimate, VarianceMethod};
use crate::model::{
    DerivedQuantity, LinearConstraint, LinearGaussianModelBuilder, LinearObservation,
};
use crate::operator::LinearMap;
use crate::physical::PhysicalMap;
use crate::prior::GaussianPrior;
use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use rand::Rng;
use std::collections::BTreeMap;

pub use feg_core::HodgeBranchKind;
pub use feg_infer::prior::hodge_matern::{
    HodgeMatern1FormPriorConfig, HodgeMaternBranchConfig, HodgeMaternSpectrum,
};
pub use feg_infer::prior::matern::{
    one_form::MaternMassInverse as HodgeOneFormMassInverse,
    two_form::MaternMassInverse as HodgeTwoFormMassInverse,
    zero_form::MaternMassInverse as HodgeZeroFormMassInverse,
};

enum HodgePriorConstruction {
    PotentialMatern(HodgeMatern1FormPriorConfig),
    FormMatern(HodgeMatern1FormPriorConfig),
}

/// Builder selecting one canonical sparse 1-form Hodge prior construction.
pub struct HodgeOneFormPriorBuilder<'a> {
    topology: &'a Complex,
    coords: Option<&'a MeshCoords>,
    metric: &'a MeshLengths,
    construction: HodgePriorConstruction,
}

impl<'a> HodgeOneFormPriorBuilder<'a> {
    /// Put the requested Matérn spectra on the exact/coexact potentials.
    ///
    /// If a potential eigenmode has Hodge--Laplacian eigenvalue `lambda`,
    /// applying `d` or `delta` contributes a factor `lambda` to the ambient
    /// 1-form covariance. Consequently an alpha-`a` potential spectrum gives
    /// an ambient branch spectrum proportional to
    /// `lambda * (kappa^2 + lambda)^(-a)`. Alpha one, two, and three are
    /// supported.
    pub fn potential_matern(
        topology: &'a Complex,
        metric: &'a MeshLengths,
        config: HodgeMatern1FormPriorConfig,
    ) -> Self {
        Self {
            topology,
            coords: None,
            metric,
            construction: HodgePriorConstruction::PotentialMatern(config),
        }
    }

    /// Put the requested Matérn spectra on the synthesized 1-form branches.
    ///
    /// The latent potential precision compensates for the spectral factor
    /// introduced by `d` or `delta`, so an ambient branch eigenmode has
    /// covariance proportional to `(kappa^2 + lambda)^(-a)`. Sparse gauges
    /// are selected internally only to make the potential representation
    /// proper; they are not part of the statistical model's public identity.
    /// Alpha one and two are supported; alpha three is rejected during
    /// validation because its spectrum-matched sparse precision is not yet
    /// implemented.
    pub fn form_matern(
        topology: &'a Complex,
        metric: &'a MeshLengths,
        config: HodgeMatern1FormPriorConfig,
    ) -> Self {
        Self {
            topology,
            coords: None,
            metric,
            construction: HodgePriorConstruction::FormMatern(config),
        }
    }

    /// Supply mesh coordinates required by coordinate-aware mass inverses.
    pub fn with_coords(mut self, coords: &'a MeshCoords) -> Self {
        self.coords = Some(coords);
        self
    }

    /// Assemble the selected canonical lower-layer construction and wrap it.
    pub fn build(self) -> Result<HodgeOneFormPrior> {
        let (spectrum, config) = match self.construction {
            HodgePriorConstruction::PotentialMatern(config) => {
                (HodgeMaternSpectrum::Potential, config)
            }
            HodgePriorConstruction::FormMatern(config) => (HodgeMaternSpectrum::Form, config),
        };
        let lower = match self.coords {
            Some(coords) => {
                feg_infer::prior::hodge_matern::build_hodge_matern_1form_prior_with_coords(
                    self.topology,
                    coords,
                    self.metric,
                    spectrum,
                    config,
                )
            }
            None => feg_infer::prior::hodge_matern::build_hodge_matern_1form_prior(
                self.topology,
                self.metric,
                spectrum,
                config,
            ),
        }
        .map_err(FeecGmrfError::Assembly)?;
        wrap_hodge_matern_prior(lower)
    }
}

/// Latent Gaussian prior plus its ambient 1-form and branch synthesis maps.
#[derive(Debug, Clone)]
pub struct HodgeOneFormPrior {
    latent_prior: GaussianPrior,
    ambient_map: LinearMap,
    branch_maps: BTreeMap<HodgeBranchKind, LinearMap>,
}

impl HodgeOneFormPrior {
    fn from_parts(
        spec: feg_core::GaussianPriorSpec,
        latent_to_ambient: &FeecCsr,
        branch_maps: BTreeMap<HodgeBranchKind, LinearMap>,
    ) -> Result<Self> {
        Ok(Self {
            latent_prior: GaussianPrior::new(spec.mean, spec.precision)?,
            ambient_map: LinearMap::from_feec_csr(latent_to_ambient)?,
            branch_maps,
        })
    }

    pub fn latent_prior(&self) -> &GaussianPrior {
        &self.latent_prior
    }

    pub fn latent_dimension(&self) -> usize {
        self.latent_prior.dimension()
    }

    pub fn ambient_dimension(&self) -> usize {
        self.ambient_map.output_dimension()
    }

    pub fn ambient_map(&self) -> &LinearMap {
        &self.ambient_map
    }

    pub fn branch_map(&self, branch: HodgeBranchKind) -> Option<&LinearMap> {
        self.branch_maps.get(&branch)
    }

    pub fn branches(&self) -> impl Iterator<Item = HodgeBranchKind> + '_ {
        self.branch_maps.keys().copied()
    }

    pub fn model_builder(self) -> HodgeLinearGaussianModelBuilder {
        HodgeLinearGaussianModelBuilder::new(self)
    }
}

/// Linear Gaussian composition in the ambient 1-form space of a Hodge prior.
pub struct HodgeLinearGaussianModelBuilder {
    ambient_map: LinearMap,
    branch_maps: BTreeMap<HodgeBranchKind, LinearMap>,
    inner: LinearGaussianModelBuilder,
}

impl HodgeLinearGaussianModelBuilder {
    pub fn new(prior: HodgeOneFormPrior) -> Self {
        Self {
            ambient_map: prior.ambient_map,
            branch_maps: prior.branch_maps,
            inner: LinearGaussianModelBuilder::new(prior.latent_prior),
        }
    }

    /// Observe an affine quantity defined on ambient 1-form coefficients.
    pub fn observe_ambient(mut self, observation: LinearObservation) -> Result<Self> {
        validate_ambient_input(&observation.operator, &self.ambient_map)?;
        let operator = observation.operator.compose(&self.ambient_map)?;
        self.inner = self.inner.observe(LinearObservation::with_bias(
            operator,
            observation.values,
            observation.bias,
            observation.noise,
        )?)?;
        Ok(self)
    }

    /// Constrain an affine quantity defined on ambient 1-form coefficients.
    pub fn constrain_ambient(mut self, constraint: LinearConstraint) -> Result<Self> {
        validate_ambient_input(&constraint.operator, &self.ambient_map)?;
        self.inner = self.inner.constrain(LinearConstraint::new(
            constraint.operator.compose(&self.ambient_map)?,
            constraint.target,
        )?)?;
        Ok(self)
    }

    /// Register a named output defined on ambient 1-form coefficients.
    pub fn derive_ambient(mut self, quantity: DerivedQuantity) -> Result<Self> {
        validate_ambient_input(&quantity.operator, &self.ambient_map)?;
        self.inner = self.inner.derive(DerivedQuantity::with_bias(
            quantity.name,
            quantity.operator.compose(&self.ambient_map)?,
            quantity.bias,
        )?)?;
        Ok(self)
    }

    /// Register a physical map defined on ambient 1-form coefficients.
    pub fn derive_physical(self, physical: PhysicalMap) -> Result<Self> {
        self.derive_ambient(physical.into_derived_quantity()?)
    }

    pub fn condition(self) -> Result<HodgePosterior> {
        Ok(HodgePosterior {
            latent: self.inner.condition()?,
            ambient_map: self.ambient_map,
            branch_maps: self.branch_maps,
        })
    }
}

/// Posterior views in latent, ambient, and branch-specific coordinates.
pub struct HodgePosterior {
    latent: Posterior,
    ambient_map: LinearMap,
    branch_maps: BTreeMap<HodgeBranchKind, LinearMap>,
}

impl HodgePosterior {
    pub fn latent(&self) -> &Posterior {
        &self.latent
    }

    pub fn latent_mut(&mut self) -> &mut Posterior {
        &mut self.latent
    }

    pub fn into_latent(self) -> Posterior {
        self.latent
    }

    pub fn ambient_mean(&self) -> Result<Vec<f64>> {
        self.latent.pushforward_mean(&self.ambient_map)
    }

    pub fn ambient_variance_estimate(
        &mut self,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        self.latent
            .pushforward_variance_estimate(&self.ambient_map, method)
    }

    pub fn branch_mean(&self, branch: HodgeBranchKind) -> Result<Vec<f64>> {
        self.latent.pushforward_mean(self.branch_map(branch)?)
    }

    pub fn branch_variance_estimate(
        &mut self,
        branch: HodgeBranchKind,
        method: VarianceMethod,
    ) -> Result<VarianceEstimate> {
        let map = self.branch_map(branch)?.clone();
        self.latent.pushforward_variance_estimate(&map, method)
    }

    pub fn sample_ambient<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Vec<f64>> {
        let sample = self.latent.sample(rng)?;
        self.ambient_map.apply(&sample)
    }

    fn branch_map(&self, branch: HodgeBranchKind) -> Result<&LinearMap> {
        self.branch_maps.get(&branch).ok_or_else(|| {
            FeecGmrfError::InvalidParameter(format!(
                "Hodge prior does not contain the {} branch",
                branch.as_str()
            ))
        })
    }
}

fn wrap_hodge_matern_prior(
    lower: feg_infer::prior::hodge_matern::HodgeMatern1FormPrior,
) -> Result<HodgeOneFormPrior> {
    let branches = lower
        .branches
        .iter()
        .map(|branch| {
            Ok((
                branch.kind,
                offset_transform(&branch.transform, branch.offset, lower.latent_dimension())?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    HodgeOneFormPrior::from_parts(
        lower.gaussian_prior_spec(),
        &lower.latent_to_ambient,
        branches,
    )
}

fn offset_transform(
    transform: &FeecCsr,
    column_offset: usize,
    latent_dimension: usize,
) -> Result<LinearMap> {
    if column_offset + transform.ncols() > latent_dimension {
        return Err(FeecGmrfError::Dimension(
            "Hodge branch transform lies outside the joint latent space".to_string(),
        ));
    }
    let mut rows = vec![Vec::new(); transform.nrows()];
    for (row, column, value) in transform.triplet_iter() {
        rows[row].push((column_offset + column, *value));
    }
    LinearMap::weighted_rows(latent_dimension, &rows)
}

fn validate_ambient_input(operator: &LinearMap, ambient_map: &LinearMap) -> Result<()> {
    if operator.input_dimension() != ambient_map.output_dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "ambient operator input dimension {} does not match Hodge ambient dimension {}",
            operator.input_dimension(),
            ambient_map.output_dimension()
        )));
    }
    Ok(())
}
