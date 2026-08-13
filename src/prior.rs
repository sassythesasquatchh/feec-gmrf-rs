//! Geometry-independent Gaussian prior construction.

use crate::boundary::{
    EliminatedLinearMap, EssentialBoundaryConditions, EssentialBoundaryElimination,
};
use crate::operator::{add_sparse, FormDegree, FormOperators, LinearMap, SparseMat};
use crate::{FeecGmrfError, Result};
pub use feg_core::MaternAlpha;
use feg_core::SparseTriplet;
use gmrf_core::{Gmrf, Vector};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

/// Provenance for converting user-facing Matérn parameters to `alpha/kappa/tau`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaternParameterConvention {
    /// Parameters were supplied directly as `alpha`, `kappa`, and `tau`.
    DirectAlphaKappaTau,
    /// `kappa` was computed using `range = sqrt(8 nu) / kappa`.
    PracticalRange {
        practical_range: f64,
        intrinsic_dimension: usize,
    },
}

/// Canonical Matérn parameters used by the discrete precision recurrence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaternParameters {
    /// Lindgren/Whittle recurrence exponent.
    pub alpha: MaternAlpha,
    /// Inverse correlation-length parameter.
    pub kappa: f64,
    /// Precision amplitude; the final precision is scaled by `tau^2`.
    pub tau: f64,
    /// Convention used to obtain the canonical parameters.
    pub convention: MaternParameterConvention,
}

impl MaternParameters {
    /// Construct validated `alpha`, `kappa`, and `tau` parameters.
    pub fn new(alpha: MaternAlpha, kappa: f64, tau: f64) -> Result<Self> {
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Matérn kappa must be finite and non-negative".to_string(),
            ));
        }
        if !tau.is_finite() || tau <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Matérn tau must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            kappa,
            tau,
            convention: MaternParameterConvention::DirectAlphaKappaTau,
        })
    }

    /// Construct parameters from the practical-range convention
    /// `range = sqrt(8 nu) / kappa`, with `nu = alpha - dimension / 2`.
    ///
    /// The `tau` amplitude remains explicit because converting marginal
    /// variance to `tau` depends on the discretization and normalization.
    pub fn from_practical_range(
        alpha: MaternAlpha,
        practical_range: f64,
        intrinsic_dimension: usize,
        tau: f64,
    ) -> Result<Self> {
        if !practical_range.is_finite() || practical_range <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "Matérn practical range must be finite and positive".to_string(),
            ));
        }
        let nu = alpha.as_u32() as f64 - intrinsic_dimension as f64 / 2.0;
        if nu <= 0.0 {
            return Err(FeecGmrfError::Unsupported(format!(
                "the practical-range convention requires nu > 0, got {nu}"
            )));
        }
        let mut parameters = Self::new(alpha, (8.0 * nu).sqrt() / practical_range, tau)?;
        parameters.convention = MaternParameterConvention::PracticalRange {
            practical_range,
            intrinsic_dimension,
        };
        Ok(parameters)
    }
}

impl Default for MaternParameters {
    fn default() -> Self {
        Self {
            alpha: MaternAlpha::Two,
            kappa: 1.0,
            tau: 1.0,
            convention: MaternParameterConvention::DirectAlphaKappaTau,
        }
    }
}

/// Policy used to approximate or supply the form mass inverse.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MassInversePolicy {
    /// Invert the row-sum lumped mass diagonal.
    RowSumLumped,
    /// Invert only the assembled diagonal. Appropriate for diagonal top-degree masses.
    Diagonal,
    /// Use the canonical Whitney projected/top-degree inverse supplied by FEEC.
    #[default]
    Projected,
    /// Use an explicitly supplied inverse.
    Provided(SparseMat),
}

/// Optional deterministic scaling applied after Matérn construction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PriorNormalization {
    /// Preserve the precision implied by `tau`.
    #[default]
    None,
    /// Rescale the precision so its matrix trace equals the supplied value.
    TargetPrecisionTrace(f64),
}

/// An assembled Gaussian prior in precision form.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPrior {
    mean: Vec<f64>,
    precision: SparseMat,
    form_degree: Option<FormDegree>,
    boundary_elimination: Option<EssentialBoundaryElimination>,
}

impl GaussianPrior {
    /// Construct a prior from an explicit mean and precision.
    pub fn new(mean: Vec<f64>, precision: SparseMat) -> Result<Self> {
        if precision.nrows() != precision.ncols() || mean.len() != precision.nrows() {
            return Err(FeecGmrfError::Dimension(format!(
                "prior mean length {} and precision shape {}x{} do not match",
                mean.len(),
                precision.nrows(),
                precision.ncols()
            )));
        }
        if !mean.iter().all(|value| value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "prior mean must contain only finite values".to_string(),
            ));
        }
        Ok(Self {
            mean,
            precision,
            form_degree: None,
            boundary_elimination: None,
        })
    }

    pub(crate) fn with_form_degree(mut self, degree: FormDegree) -> Self {
        self.form_degree = Some(degree);
        self
    }

    pub(crate) fn with_boundary_elimination(
        mut self,
        elimination: EssentialBoundaryElimination,
    ) -> Result<Self> {
        if elimination.active_dimension() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "boundary active dimension {} does not match prior dimension {}",
                elimination.active_dimension(),
                self.dimension()
            )));
        }
        self.boundary_elimination = Some(elimination);
        Ok(self)
    }

    /// Prior dimension.
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Dimension of the complete FEEC cochain before essential-boundary elimination.
    pub fn cochain_dimension(&self) -> usize {
        self.boundary_elimination.as_ref().map_or_else(
            || self.dimension(),
            EssentialBoundaryElimination::full_dimension,
        )
    }

    /// Prior mean in active coordinates.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Prior mean in full cochain ordering, including prescribed values.
    pub fn cochain_mean(&self) -> Result<Vec<f64>> {
        match &self.boundary_elimination {
            Some(elimination) => elimination.lift_state(&self.mean),
            None => Ok(self.mean.clone()),
        }
    }

    /// Prior precision in active coordinates.
    pub fn precision(&self) -> &SparseMat {
        &self.precision
    }

    /// FEEC form degree, when the prior was assembled from form operators.
    pub fn form_degree(&self) -> Option<FormDegree> {
        self.form_degree
    }

    /// Essential-boundary elimination carried by this prior, when present.
    pub fn boundary_elimination(&self) -> Option<&EssentialBoundaryElimination> {
        self.boundary_elimination.as_ref()
    }

    /// Condition an arbitrary full-space Gaussian prior on prescribed
    /// essential-boundary coefficients.
    ///
    /// Matérn users should normally prefer
    /// [`MaternPriorBuilder::essential_boundary_conditions`], which reduces
    /// the FEEC form space before applying the mass-inverse recurrence. This
    /// method instead computes the exact conditional distribution associated
    /// with an already assembled full precision.
    pub fn condition_on_essential_boundary(
        &self,
        conditions: EssentialBoundaryConditions,
    ) -> Result<Self> {
        if self.boundary_elimination.is_some() {
            return Err(FeecGmrfError::InvalidParameter(
                "prior already carries essential-boundary elimination".to_string(),
            ));
        }
        let elimination = conditions.eliminate(self.dimension())?;
        let reduced_precision = elimination.reduce_square(&self.precision)?;
        let centered = self
            .mean
            .iter()
            .zip(elimination.prescribed_values())
            .map(|(mean, prescribed)| mean - prescribed)
            .collect::<Vec<_>>();
        let full_information = self
            .precision
            .apply_checked(&centered)
            .map_err(FeecGmrfError::Dimension)?;
        let information = elimination.reduce_state(&full_information)?;
        let gmrf = Gmrf::from_information_and_precision(
            Vector::from_vec(information),
            crate::infer::gmrf_sparse(&reduced_precision),
        )?;
        let mean = gmrf.mean().iter().copied().collect();
        let mut prior = Self::new(mean, reduced_precision)?;
        prior.form_degree = self.form_degree;
        prior.with_boundary_elimination(elimination)
    }

    /// Eliminate a full-cochain map, or validate an already-active map.
    pub fn eliminate_map(&self, map: &LinearMap, bias: &[f64]) -> Result<EliminatedLinearMap> {
        if let Some(elimination) = &self.boundary_elimination {
            if map.input_dimension() == elimination.full_dimension() {
                return elimination.eliminate_map(map, bias);
            }
        }
        if map.input_dimension() != self.dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "map input dimension {} matches neither active dimension {} nor full cochain dimension {}",
                map.input_dimension(),
                self.dimension(),
                self.cochain_dimension()
            )));
        }
        if bias.len() != map.output_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "map bias length {} does not match output dimension {}",
                bias.len(),
                map.output_dimension()
            )));
        }
        Ok(EliminatedLinearMap::from_active(map.clone(), bias.to_vec()))
    }

    /// Return the same Gaussian location with its precision multiplied by a
    /// positive scalar. Consequently, every covariance is divided by `scale`.
    pub fn with_precision_scale(&self, scale: f64) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(FeecGmrfError::InvalidParameter(
                "precision scale must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            mean: self.mean.clone(),
            precision: self.precision.scaled(scale),
            form_degree: self.form_degree,
            boundary_elimination: self.boundary_elimination.clone(),
        })
    }
}

enum MaternSource<'a> {
    Operators(Box<FormOperators>),
    Feec {
        topology: &'a Complex,
        metric: &'a MeshLengths,
        degree: FormDegree,
    },
}

/// Builder for Matérn GMRFs on arbitrary FEEC form spaces.
pub struct MaternPriorBuilder<'a> {
    source: MaternSource<'a>,
    parameters: MaternParameters,
    mass_inverse: MassInversePolicy,
    normalization: PriorNormalization,
    mean: Option<Vec<f64>>,
    essential_boundary_conditions: Option<EssentialBoundaryConditions>,
}

impl MaternPriorBuilder<'static> {
    /// Start from user-supplied weak mass and Hodge--Laplacian operators.
    pub fn from_operators(operators: FormOperators) -> Self {
        Self {
            source: MaternSource::Operators(Box::new(operators)),
            parameters: MaternParameters::default(),
            mass_inverse: MassInversePolicy::RowSumLumped,
            normalization: PriorNormalization::None,
            mean: None,
            essential_boundary_conditions: None,
        }
    }
}

impl<'a> MaternPriorBuilder<'a> {
    /// Start from a FEEC topology, metric, and form degree.
    pub fn from_feec(
        topology: &'a Complex,
        metric: &'a MeshLengths,
        degree: usize,
    ) -> Result<Self> {
        let degree = FormDegree::new(degree, topology.dim())?;
        Ok(Self {
            source: MaternSource::Feec {
                topology,
                metric,
                degree,
            },
            parameters: MaternParameters::default(),
            mass_inverse: MassInversePolicy::Projected,
            normalization: PriorNormalization::None,
            mean: None,
            essential_boundary_conditions: None,
        })
    }

    /// Set canonical Matérn parameters.
    pub fn parameters(mut self, parameters: MaternParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set the mass-inverse policy.
    pub fn mass_inverse(mut self, policy: MassInversePolicy) -> Self {
        self.mass_inverse = policy;
        self
    }

    /// Set a deterministic post-construction normalization.
    pub fn normalization(mut self, normalization: PriorNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set a non-zero prior mean. For a boundary-aware builder this may use
    /// active ordering or full cochain ordering. A supplied full mean must
    /// agree with every prescribed essential value.
    pub fn mean(mut self, mean: Vec<f64>) -> Self {
        self.mean = Some(mean);
        self
    }

    /// Eliminate prescribed essential-boundary coefficients before applying
    /// the Matérn recurrence.
    pub fn essential_boundary_conditions(
        mut self,
        conditions: EssentialBoundaryConditions,
    ) -> Self {
        self.essential_boundary_conditions = Some(conditions);
        self
    }

    /// Assemble the prior precision through the canonical grade-independent recurrence.
    pub fn build(self) -> Result<GaussianPrior> {
        let (mut operators, mut projected_inverse) = match self.source {
            MaternSource::Operators(operators) => (*operators, None),
            MaternSource::Feec {
                topology,
                metric,
                degree,
            } => {
                let hodge = feg_infer::prior::matern::generic::build_hodge_laplacian_form(
                    topology,
                    metric,
                    degree.get(),
                )
                .map_err(FeecGmrfError::Assembly)?;
                let mass = feg_infer::sparse::feec_csr_to_core_triplet(&hodge.mass_u);
                let laplacian = feg_infer::sparse::feec_csr_to_core_triplet(&hodge.laplacian);
                let inverse =
                    feg_infer::prior::matern::generic::build_projected_or_top_degree_mass_inverse(
                        topology,
                        metric,
                        degree.get(),
                        &hodge.mass_u,
                    )
                    .map_err(FeecGmrfError::Assembly)?;
                (
                    FormOperators::new(degree, topology.dim(), mass, laplacian)?,
                    Some(feg_infer::sparse::feec_csr_to_core_triplet(&inverse)),
                )
            }
        };

        let original_dimension = operators.dimension();
        let embedded_layout = operators.boundary_layout().clone();
        let already_reduced = embedded_layout.full_dimension() != operators.dimension()
            || !embedded_layout.fixed_dofs().is_empty();
        let mut boundary_elimination = if already_reduced {
            Some(EssentialBoundaryElimination::from_layout(embedded_layout)?)
        } else {
            None
        };
        if let Some(conditions) = self.essential_boundary_conditions {
            if already_reduced {
                return Err(FeecGmrfError::InvalidParameter(
                    "cannot apply essential boundary conditions to operators that are already boundary-reduced"
                        .to_string(),
                ));
            }
            let elimination = conditions.eliminate(original_dimension)?;
            let reduced_mass = elimination.reduce_square(operators.mass())?;
            let reduced_laplacian = elimination.reduce_square(operators.hodge_laplacian())?;
            operators = FormOperators::new(
                operators.degree(),
                operators.complex_dimension(),
                reduced_mass,
                reduced_laplacian,
            )?
            .with_boundary_layout(elimination.layout().clone())?;
            projected_inverse = projected_inverse
                .as_ref()
                .map(|inverse| elimination.reduce_square(inverse))
                .transpose()?;
            boundary_elimination = Some(elimination);
        }

        let mass_inverse = match self.mass_inverse {
            MassInversePolicy::RowSumLumped => row_sum_inverse(operators.mass())?,
            MassInversePolicy::Diagonal => diagonal_inverse(operators.mass())?,
            MassInversePolicy::Projected => projected_inverse.ok_or_else(|| {
                FeecGmrfError::Unsupported(
                    "projected mass inversion requires the FEEC constructor or an explicit inverse"
                        .to_string(),
                )
            })?,
            MassInversePolicy::Provided(inverse) => {
                if inverse.nrows() == operators.dimension()
                    && inverse.ncols() == operators.dimension()
                {
                    inverse
                } else if let Some(elimination) = &boundary_elimination {
                    elimination.reduce_square(&inverse)?
                } else {
                    return Err(FeecGmrfError::Dimension(
                        "provided mass inverse does not match the form-space dimension".to_string(),
                    ));
                }
            }
        };

        let system = add_sparse(
            operators.hodge_laplacian(),
            &operators.mass().scaled(self.parameters.kappa.powi(2)),
        )?;
        let mut precision = matern_recurrence(
            &system,
            &mass_inverse,
            self.parameters.alpha,
            self.parameters.tau,
        )?;
        apply_normalization(&mut precision, self.normalization)?;
        let mean = resolve_reduced_mean(
            self.mean,
            operators.dimension(),
            boundary_elimination.as_ref(),
        )?;
        let prior = GaussianPrior::new(mean, precision)?.with_form_degree(operators.degree());
        match boundary_elimination {
            Some(elimination) => prior.with_boundary_elimination(elimination),
            None => Ok(prior),
        }
    }
}

fn resolve_reduced_mean(
    mean: Option<Vec<f64>>,
    active_dimension: usize,
    elimination: Option<&EssentialBoundaryElimination>,
) -> Result<Vec<f64>> {
    let Some(mean) = mean else {
        return Ok(vec![0.0; active_dimension]);
    };
    let Some(elimination) = elimination else {
        if mean.len() != active_dimension {
            return Err(FeecGmrfError::Dimension(format!(
                "prior mean length {} does not match dimension {active_dimension}",
                mean.len()
            )));
        }
        return Ok(mean);
    };
    if mean.len() == active_dimension {
        return Ok(mean);
    }
    if mean.len() != elimination.full_dimension() {
        return Err(FeecGmrfError::Dimension(format!(
            "boundary-aware prior mean length {} matches neither active dimension {} nor full dimension {}",
            mean.len(),
            active_dimension,
            elimination.full_dimension()
        )));
    }
    for &(index, prescribed) in elimination.layout().fixed_dofs() {
        let supplied = mean[index];
        let tolerance = 1.0e-12 * (1.0 + prescribed.abs());
        if (supplied - prescribed).abs() > tolerance {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "prior mean at prescribed dof {index} is {supplied}, expected {prescribed}"
            )));
        }
    }
    elimination.reduce_state(&mean)
}

/// Apply the canonical Matérn precision recurrence to an assembled system and mass inverse.
pub fn matern_recurrence(
    system: &SparseMat,
    mass_inverse: &SparseMat,
    alpha: MaternAlpha,
    tau: f64,
) -> Result<SparseMat> {
    if system.nrows() != system.ncols()
        || mass_inverse.nrows() != system.nrows()
        || mass_inverse.ncols() != system.ncols()
    {
        return Err(FeecGmrfError::Dimension(
            "Matérn system and mass inverse must be square and share a dimension".to_string(),
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "Matérn tau must be finite and positive".to_string(),
        ));
    }
    let system = feg_infer::sparse::core_triplet_to_feec_csr(system);
    let mass_inverse = feg_infer::sparse::core_triplet_to_feec_csr(mass_inverse);
    let precision = feg_infer::prior::matern::build_lindgren_precision_from_system(
        &system,
        &mass_inverse,
        alpha,
        tau,
    );
    Ok(feg_infer::sparse::feec_csr_to_core_triplet(&precision))
}

fn diagonal_inverse(mass: &SparseMat) -> Result<SparseMat> {
    let mut diagonal = vec![0.0; mass.nrows()];
    for (row, col, value) in mass.triplet_iter() {
        if row == col {
            diagonal[row] += value;
        }
    }
    inverse_diagonal(&diagonal, "mass diagonal")
}

fn row_sum_inverse(mass: &SparseMat) -> Result<SparseMat> {
    let mut sums = vec![0.0; mass.nrows()];
    for (row, _, value) in mass.triplet_iter() {
        sums[row] += value;
    }
    inverse_diagonal(&sums, "row-sum lumped mass")
}

fn inverse_diagonal(values: &[f64], name: &str) -> Result<SparseMat> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(FeecGmrfError::InvalidParameter(format!(
            "{name} entries must be finite and positive"
        )));
    }
    Ok(SparseMat::from_triplets(
        values.len(),
        values.len(),
        values
            .iter()
            .enumerate()
            .map(|(index, value)| SparseTriplet {
                row: index,
                col: index,
                value: value.recip(),
            }),
    ))
}

fn apply_normalization(precision: &mut SparseMat, normalization: PriorNormalization) -> Result<()> {
    let PriorNormalization::TargetPrecisionTrace(target) = normalization else {
        return Ok(());
    };
    if !target.is_finite() || target <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "target precision trace must be finite and positive".to_string(),
        ));
    }
    let trace = precision
        .triplet_iter()
        .filter(|(row, col, _)| row == col)
        .map(|(_, _, value)| value)
        .sum::<f64>();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(FeecGmrfError::InvalidParameter(
            "assembled precision trace must be finite and positive".to_string(),
        ));
    }
    *precision = precision.scaled(target / trace);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrence_is_independent_of_form_degree() {
        let a = SparseMat::diagonal(2, 2.0);
        let m_inv = SparseMat::diagonal(2, 0.5);
        let q1 = matern_recurrence(&a, &m_inv, MaternAlpha::One, 1.0).unwrap();
        let q2 = matern_recurrence(&a, &m_inv, MaternAlpha::Two, 1.0).unwrap();
        let q3 = matern_recurrence(&a, &m_inv, MaternAlpha::Three, 1.0).unwrap();
        assert_eq!(q1.apply_checked(&[1.0, 1.0]).unwrap(), vec![2.0, 2.0]);
        assert_eq!(q2.apply_checked(&[1.0, 1.0]).unwrap(), vec![2.0, 2.0]);
        assert_eq!(q3.apply_checked(&[1.0, 1.0]).unwrap(), vec![2.0, 2.0]);
    }
}
