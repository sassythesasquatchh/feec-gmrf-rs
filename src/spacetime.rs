//! Geometry-independent spatiotemporal precision construction.

use crate::operator::SparseMat;
use crate::{FeecGmrfError, Result};
use common::linalg::nalgebra::{CooMatrix, CsrMatrix};
use feg_infer::sparse::{core_triplet_to_feec_csr, feec_csr_to_gmrf};

/// Time discretization used to compose spatial FEEC operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeDiscretization {
    /// First-order implicit Euler transition.
    ImplicitEuler,
}

/// A spatiotemporal GMRF precision and its resolved time grid.
#[derive(Debug, Clone)]
pub struct SpacetimePrior {
    times: Vec<f64>,
    precision: gmrf_core::BlockTridiagonalPrecision,
}

impl SpacetimePrior {
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn precision(&self) -> &gmrf_core::BlockTridiagonalPrecision {
        &self.precision
    }
}

/// Builder for a structured GMRF from deterministic spatial operators,
/// an initial-state precision, and a process precision.
pub struct SpacetimePriorBuilder {
    mass: CsrMatrix,
    drift: CsrMatrix,
    initial_precision: CsrMatrix,
    process_precision: CsrMatrix,
    times: Vec<f64>,
    discretization: TimeDiscretization,
}

impl SpacetimePriorBuilder {
    pub fn from_operators(
        mass: SparseMat,
        drift: SparseMat,
        initial_precision: SparseMat,
        process_precision: SparseMat,
    ) -> Result<Self> {
        let dimension = mass.nrows();
        if [&mass, &drift, &initial_precision, &process_precision]
            .iter()
            .any(|matrix| matrix.nrows() != dimension || matrix.ncols() != dimension)
        {
            return Err(FeecGmrfError::Dimension(
                "spatiotemporal operators must share one square dimension".to_string(),
            ));
        }
        Ok(Self {
            mass: core_triplet_to_feec_csr(&mass),
            drift: core_triplet_to_feec_csr(&drift),
            initial_precision: core_triplet_to_feec_csr(&initial_precision),
            process_precision: core_triplet_to_feec_csr(&process_precision),
            times: Vec::new(),
            discretization: TimeDiscretization::ImplicitEuler,
        })
    }

    pub fn times(mut self, times: Vec<f64>) -> Self {
        self.times = times;
        self
    }

    pub fn discretization(mut self, discretization: TimeDiscretization) -> Self {
        self.discretization = discretization;
        self
    }

    pub fn build(self) -> Result<SpacetimePrior> {
        match self.discretization {
            TimeDiscretization::ImplicitEuler => {}
        }
        if self.times.is_empty()
            || !self.times.iter().all(|time| time.is_finite())
            || self.times.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(FeecGmrfError::InvalidParameter(
                "time grid must be finite, nonempty, and strictly increasing".to_string(),
            ));
        }
        let dimension = self.mass.nrows();
        let mut diagonal = vec![zero_matrix(dimension); self.times.len()];
        let mut lower = Vec::with_capacity(self.times.len().saturating_sub(1));
        diagonal[0] = add_sparse(&diagonal[0], &self.initial_precision);
        let mass_t = self.mass.transpose();
        for pair in self.times.windows(2) {
            let dt = pair[1] - pair[0];
            let inv_dt = 1.0 / dt;
            let g = add_sparse(&self.mass, &scale_matrix(&self.drift, dt));
            let step = lower.len();
            diagonal[step] = add_sparse(
                &diagonal[step],
                &scale_matrix(&(&mass_t * &self.process_precision * &self.mass), inv_dt),
            );
            diagonal[step + 1] = add_sparse(
                &diagonal[step + 1],
                &scale_matrix(&(&g.transpose() * &self.process_precision * &g), inv_dt),
            );
            lower.push(scale_matrix(
                &(&g.transpose() * &self.process_precision * &self.mass),
                -inv_dt,
            ));
        }
        let precision = gmrf_core::BlockTridiagonalPrecision::new(
            diagonal.iter().map(feec_csr_to_gmrf).collect(),
            lower.iter().map(feec_csr_to_gmrf).collect(),
        )
        .map_err(|error| FeecGmrfError::Assembly(error.to_string()))?;
        Ok(SpacetimePrior {
            times: self.times,
            precision,
        })
    }
}

fn zero_matrix(dimension: usize) -> CsrMatrix {
    CsrMatrix::from(&CooMatrix::new(dimension, dimension))
}

fn add_sparse(lhs: &CsrMatrix, rhs: &CsrMatrix) -> CsrMatrix {
    let mut coo = CooMatrix::from(lhs);
    for (row, col, value) in rhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    CsrMatrix::from(&coo)
}

fn scale_matrix(matrix: &CsrMatrix, scale: f64) -> CsrMatrix {
    let mut coo = CooMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        let value = *value * scale;
        if value != 0.0 {
            coo.push(row, col, value);
        }
    }
    CsrMatrix::from(&coo)
}
