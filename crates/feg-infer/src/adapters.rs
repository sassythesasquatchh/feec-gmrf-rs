//! Conversions from FEEC assembly results to statistical model specifications.

use feg_core::{NonlinearResidualEvaluation, NonlinearResidualModel};
use formoniq::problems::residual::{ResidualEvaluation, ResidualModel};

use crate::sparse::feec_csr_to_core_triplet;

/// Adapt a native FEEC residual/Jacobian assembler to the shared nonlinear
/// inference contract.
#[derive(Debug, Clone, Copy)]
pub struct FeecResidualAdapter<'a, M: ?Sized> {
    model: &'a M,
}

impl<'a, M: ResidualModel + ?Sized> FeecResidualAdapter<'a, M> {
    /// Borrow a native FEEC residual model.
    pub fn new(model: &'a M) -> Self {
        Self { model }
    }

    /// Borrow the underlying FEEC model.
    pub fn inner(&self) -> &'a M {
        self.model
    }
}

impl<M: ResidualModel + ?Sized> NonlinearResidualModel for FeecResidualAdapter<'_, M> {
    fn state_dimension(&self) -> usize {
        self.model.state_dimension()
    }

    fn residual_dimension(&self) -> usize {
        self.model.residual_dimension()
    }

    fn residual_and_jacobian(&self, state: &[f64]) -> Result<NonlinearResidualEvaluation, String> {
        let ResidualEvaluation { residual, jacobian } = self.model.residual_and_jacobian(state)?;
        Ok(NonlinearResidualEvaluation {
            residual: residual.as_slice().to_vec(),
            jacobian: feec_csr_to_core_triplet(&jacobian),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Vector};

    struct NativeIdentity;

    impl ResidualModel for NativeIdentity {
        fn state_dimension(&self) -> usize {
            1
        }

        fn residual_dimension(&self) -> usize {
            1
        }

        fn residual_and_jacobian(&self, state: &[f64]) -> Result<ResidualEvaluation, String> {
            let mut jacobian = CooMatrix::new(1, 1);
            jacobian.push(0, 0, 1.0);
            Ok(ResidualEvaluation {
                residual: Vector::from_column_slice(state),
                jacobian: CsrMatrix::from(&jacobian),
            })
        }
    }

    #[test]
    fn preserves_native_residual_and_jacobian() {
        let adapter = FeecResidualAdapter::new(&NativeIdentity);
        let evaluation = NonlinearResidualModel::residual_and_jacobian(&adapter, &[2.0]).unwrap();
        assert_eq!(evaluation.residual, vec![2.0]);
        assert_eq!(
            evaluation.jacobian.triplet_iter().collect::<Vec<_>>(),
            vec![(0, 0, 1.0)]
        );
    }
}
