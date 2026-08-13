//! Backend-neutral contracts for FEEC/GMRF model composition.

mod contracts;

pub use contracts::*;

/// Integer exponent in the Lindgren/Whittle Matérn precision recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum MaternAlpha {
    One,
    #[default]
    Two,
    Three,
}

impl MaternAlpha {
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }

    pub fn whittle_smoothness_2d(self) -> f64 {
        self.as_u32() as f64 - 1.0
    }

    pub fn diagnostic_range_2d(self, kappa: f64) -> f64 {
        let nu = self.whittle_smoothness_2d();
        if nu > 0.0 {
            (8.0 * nu).sqrt() / kappa
        } else {
            1.0 / kappa
        }
    }
}

impl TryFrom<u32> for MaternAlpha {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            _ => Err(format!(
                "unsupported Matérn alpha {value}; expected 1, 2, or 3"
            )),
        }
    }
}

impl std::str::FromStr for MaternAlpha {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let alpha = value
            .parse::<u32>()
            .map_err(|_| format!("invalid Matérn alpha `{value}`; expected 1, 2, or 3"))?;
        Self::try_from(alpha)
    }
}

/// Branch of a discrete Hodge decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HodgeBranchKind {
    Exact,
    Coexact,
    Harmonic,
}

impl HodgeBranchKind {
    pub const ALL: [Self; 3] = [Self::Exact, Self::Coexact, Self::Harmonic];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Coexact => "coexact",
            Self::Harmonic => "harmonic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationPreference {
    Auto,
    ForceCollapsed,
    ForceLatent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPriorSpec {
    pub mean: Vec<f64>,
    pub precision: SparseTripletMatrix,
}

impl GaussianPriorSpec {
    pub fn dimension(&self) -> usize {
        self.precision.nrows()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.precision.nrows() != self.precision.ncols() {
            return Err("Gaussian prior precision must be square".to_string());
        }
        if self.mean.len() != self.precision.nrows() {
            return Err(format!(
                "Gaussian prior mean length {} must match precision dimension {}",
                self.mean.len(),
                self.precision.nrows()
            ));
        }
        if !self.mean.iter().all(|value| value.is_finite()) {
            return Err("Gaussian prior mean must contain only finite values".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearUncertainInputSpec {
    pub name: String,
    pub operator: SparseTripletMatrix,
    pub prior: GaussianPriorSpec,
    pub preference: RepresentationPreference,
    pub collapsed_precision: Option<SparseTripletMatrix>,
}

impl LinearUncertainInputSpec {
    pub fn validate(&self, residual_dimension: usize) -> Result<(), String> {
        self.prior.validate()?;
        if self.operator.nrows() != residual_dimension {
            return Err(format!(
                "uncertain input `{}` operator row count {} must match residual dimension {}",
                self.name,
                self.operator.nrows(),
                residual_dimension
            ));
        }
        if self.operator.ncols() != self.prior.dimension() {
            return Err(format!(
                "uncertain input `{}` operator column count {} must match prior dimension {}",
                self.name,
                self.operator.ncols(),
                self.prior.dimension()
            ));
        }
        if let Some(collapsed) = &self.collapsed_precision {
            if collapsed.nrows() != residual_dimension || collapsed.ncols() != residual_dimension {
                return Err(format!(
                    "uncertain input `{}` collapsed precision must be {}x{}, got {}x{}",
                    self.name,
                    residual_dimension,
                    residual_dimension,
                    collapsed.nrows(),
                    collapsed.ncols()
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionWeightedGaussianMeasurementSpec {
    pub name: String,
    pub operator: SparseTripletMatrix,
    pub observations: Vec<f64>,
    pub bias: Vec<f64>,
    pub precision: SparseTripletMatrix,
}

impl PrecisionWeightedGaussianMeasurementSpec {
    pub fn validate(&self, state_dimension: usize) -> Result<(), String> {
        if self.operator.ncols() != state_dimension {
            return Err(format!(
                "precision-weighted measurement `{}` operator column count {} must match state dimension {}",
                self.name,
                self.operator.ncols(),
                state_dimension
            ));
        }
        if self.operator.nrows() != self.observations.len()
            || self.operator.nrows() != self.bias.len()
        {
            return Err(format!(
                "precision-weighted measurement `{}` row, observation, and bias counts must match",
                self.name
            ));
        }
        if self.precision.nrows() != self.operator.nrows()
            || self.precision.ncols() != self.operator.nrows()
        {
            return Err(format!(
                "precision-weighted measurement `{}` precision must be {}x{}, got {}x{}",
                self.name,
                self.operator.nrows(),
                self.operator.nrows(),
                self.precision.nrows(),
                self.precision.ncols()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_validation_rejects_inconsistent_dimensions() {
        let prior = GaussianPriorSpec {
            mean: vec![0.0],
            precision: SparseTripletMatrix::diagonal(2, 1.0),
        };
        assert!(prior.validate().is_err());
    }

    #[test]
    fn hodge_branch_names_are_stable() {
        assert_eq!(
            HodgeBranchKind::ALL.map(HodgeBranchKind::as_str),
            ["exact", "coexact", "harmonic"]
        );
    }
}
