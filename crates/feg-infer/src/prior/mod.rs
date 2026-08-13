pub mod exact_potential;
pub mod hodge;
pub mod matern;
pub mod spacetime;
pub mod sparse_anchor_hodge;
pub mod trace_normalization;

use crate::sparse::identity_triplet_matrix;
use feg_core::GaussianPriorSpec;

pub fn zero_mean_diagonal_prior(dimension: usize, precision: f64) -> GaussianPriorSpec {
    GaussianPriorSpec {
        mean: vec![0.0; dimension],
        precision: identity_triplet_matrix(dimension, precision),
    }
}
