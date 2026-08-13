//! Essential-boundary elimination for full FEEC cochain spaces.
//!
//! Homogeneous and prescribed essential values use the same resolved
//! elimination. Internally, a full cochain is represented as `x = P z + g`,
//! where `z` contains the active coefficients. Both boundary cases use
//! `EssentialBoundaryConditions`.

use crate::operator::{BoundaryLayout, LinearMap, SparseMat};
use crate::{FeecGmrfError, Result};
use feg_core::{NonlinearResidualEvaluation, NonlinearResidualModel, SparseTriplet};
use formoniq::reduction::{DofLayout, PrescribedDof};
use manifold::topology::complex::Complex;
use std::collections::{BTreeMap, BTreeSet};

/// User-supplied essential values on selected full-space coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct EssentialBoundaryConditions {
    fixed_dofs: Vec<(usize, f64)>,
}

impl EssentialBoundaryConditions {
    /// Prescribe zero on the selected full-space coefficients.
    pub fn homogeneous(dofs: impl IntoIterator<Item = usize>) -> Result<Self> {
        let dofs = dofs.into_iter().collect::<Vec<_>>();
        Self::prescribed(dofs.clone(), vec![0.0; dofs.len()])
    }

    /// Prescribe one value for each selected full-space coefficient.
    pub fn prescribed(dofs: Vec<usize>, values: Vec<f64>) -> Result<Self> {
        if dofs.len() != values.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "essential-boundary dof count {} does not match value count {}",
                dofs.len(),
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FeecGmrfError::InvalidParameter(
                "essential-boundary values must be finite".to_string(),
            ));
        }
        let mut fixed = BTreeMap::new();
        for (dof, value) in dofs.into_iter().zip(values) {
            if fixed.insert(dof, value).is_some() {
                return Err(FeecGmrfError::InvalidParameter(format!(
                    "essential-boundary dof {dof} is prescribed more than once"
                )));
            }
        }
        Ok(Self {
            fixed_dofs: fixed.into_iter().collect(),
        })
    }

    /// Prescribe zero on every degree-`degree` simplex in the topological boundary.
    pub fn homogeneous_on_boundary(topology: &Complex, degree: usize) -> Result<Self> {
        Self::homogeneous(Self::topological_boundary_dofs(topology, degree)?)
    }

    /// Prescribe values, in topological boundary order, on every
    /// degree-`degree` simplex in the topological boundary.
    pub fn prescribed_on_boundary(
        topology: &Complex,
        degree: usize,
        values: Vec<f64>,
    ) -> Result<Self> {
        Self::prescribed(Self::topological_boundary_dofs(topology, degree)?, values)
    }

    /// Full-space coefficient indices on the topological boundary for one form degree.
    pub fn topological_boundary_dofs(topology: &Complex, degree: usize) -> Result<Vec<usize>> {
        if degree > topology.dim() {
            return Err(FeecGmrfError::Dimension(format!(
                "boundary form degree {degree} exceeds complex dimension {}",
                topology.dim()
            )));
        }
        let mut dofs = topology
            .boundary_subcomplex_simplices(degree)
            .into_iter()
            .map(|simplex| simplex.kidx)
            .collect::<Vec<_>>();
        dofs.sort_unstable();
        dofs.dedup();
        Ok(dofs)
    }

    /// Prescribed full-space coefficient indices and values.
    pub fn fixed_dofs(&self) -> &[(usize, f64)] {
        &self.fixed_dofs
    }

    /// Resolve active coefficients and the full-to-reduced transformation.
    pub fn eliminate(&self, full_dimension: usize) -> Result<EssentialBoundaryElimination> {
        if let Some((index, _)) = self
            .fixed_dofs
            .iter()
            .find(|(index, _)| *index >= full_dimension)
        {
            return Err(FeecGmrfError::Dimension(format!(
                "essential-boundary dof {index} is outside full dimension {full_dimension}"
            )));
        }
        let fixed = self
            .fixed_dofs
            .iter()
            .map(|(index, _)| *index)
            .collect::<BTreeSet<_>>();
        let active = (0..full_dimension)
            .filter(|index| !fixed.contains(index))
            .collect::<Vec<_>>();
        EssentialBoundaryElimination::from_layout(BoundaryLayout::new(
            full_dimension,
            active,
            self.fixed_dofs.clone(),
        )?)
    }
}

/// A linear map after fixed full-space columns have been eliminated.
///
/// Applying a full map `H` to `x = P z + g` gives
/// `H x + b = reduced * z + fixed_offset`.
#[derive(Debug, Clone, PartialEq)]
pub struct EliminatedLinearMap {
    reduced: LinearMap,
    fixed_offset: Vec<f64>,
}

impl EliminatedLinearMap {
    pub(crate) fn from_active(reduced: LinearMap, fixed_offset: Vec<f64>) -> Self {
        Self {
            reduced,
            fixed_offset,
        }
    }

    /// Reduced linear operator acting on active coefficients.
    pub fn reduced(&self) -> &LinearMap {
        &self.reduced
    }

    /// Output offset contributed by prescribed coefficients and input bias.
    pub fn fixed_offset(&self) -> &[f64] {
        &self.fixed_offset
    }

    /// Apply the eliminated map, including its fixed-value offset.
    pub fn apply(&self, active: &[f64]) -> Result<Vec<f64>> {
        let mut output = self.reduced.apply(active)?;
        for (value, offset) in output.iter_mut().zip(&self.fixed_offset) {
            *value += offset;
        }
        Ok(output)
    }

    pub(crate) fn into_parts(self) -> (LinearMap, Vec<f64>) {
        (self.reduced, self.fixed_offset)
    }
}

/// Resolved elimination of essential-boundary coefficients.
///
/// This structure is used unchanged for homogeneous and non-homogeneous
/// essential conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct EssentialBoundaryElimination {
    layout: BoundaryLayout,
    active_to_full: LinearMap,
    prescribed_values: Vec<f64>,
}

/// A full-cochain residual model evaluated in active coordinates after
/// essential-boundary elimination.
pub struct EliminatedResidualModel<'a> {
    full_model: &'a dyn NonlinearResidualModel,
    elimination: EssentialBoundaryElimination,
}

impl<'a> EliminatedResidualModel<'a> {
    /// Wrap a residual model whose state uses the full cochain ordering.
    pub fn new(
        full_model: &'a dyn NonlinearResidualModel,
        elimination: EssentialBoundaryElimination,
    ) -> Result<Self> {
        if full_model.state_dimension() != elimination.full_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "residual state dimension {} does not match full cochain dimension {}",
                full_model.state_dimension(),
                elimination.full_dimension()
            )));
        }
        Ok(Self {
            full_model,
            elimination,
        })
    }

    /// Boundary elimination applied by this residual adapter.
    pub fn elimination(&self) -> &EssentialBoundaryElimination {
        &self.elimination
    }
}

impl NonlinearResidualModel for EliminatedResidualModel<'_> {
    fn state_dimension(&self) -> usize {
        self.elimination.active_dimension()
    }

    fn residual_dimension(&self) -> usize {
        self.full_model.residual_dimension()
    }

    fn residual(&self, active: &[f64]) -> std::result::Result<Vec<f64>, String> {
        let full = self
            .elimination
            .lift_state(active)
            .map_err(|error| error.to_string())?;
        self.full_model.residual(&full)
    }

    fn residual_and_jacobian(
        &self,
        active: &[f64],
    ) -> std::result::Result<NonlinearResidualEvaluation, String> {
        let full = self
            .elimination
            .lift_state(active)
            .map_err(|error| error.to_string())?;
        let evaluation = self.full_model.residual_and_jacobian(&full)?;
        evaluation.validate(self.residual_dimension(), self.elimination.full_dimension())?;
        let jacobian = self
            .elimination
            .eliminate_sparse_columns(&evaluation.jacobian)
            .map_err(|error| error.to_string())?;
        Ok(NonlinearResidualEvaluation {
            residual: evaluation.residual,
            jacobian,
        })
    }
}

impl EssentialBoundaryElimination {
    /// Construct an elimination from a complete active/fixed layout.
    pub fn from_layout(layout: BoundaryLayout) -> Result<Self> {
        if layout.active_dofs().is_empty() {
            return Err(FeecGmrfError::InvalidParameter(
                "essential-boundary elimination must leave at least one active coefficient"
                    .to_string(),
            ));
        }
        if layout.active_dofs().len() + layout.fixed_dofs().len() != layout.full_dimension() {
            return Err(FeecGmrfError::InvalidParameter(
                "every full-space coefficient must be either active or prescribed".to_string(),
            ));
        }
        let mut covered = BTreeSet::new();
        covered.extend(layout.active_dofs().iter().copied());
        covered.extend(layout.fixed_dofs().iter().map(|(index, _)| *index));
        if covered.len() != layout.full_dimension() {
            return Err(FeecGmrfError::InvalidParameter(
                "essential-boundary layout does not cover the full coefficient space".to_string(),
            ));
        }

        let active_to_full = LinearMap::new(SparseMat::from_triplets(
            layout.full_dimension(),
            layout.active_dofs().len(),
            layout
                .active_dofs()
                .iter()
                .copied()
                .enumerate()
                .map(|(active, full)| SparseTriplet {
                    row: full,
                    col: active,
                    value: 1.0,
                }),
        ))?;
        let mut prescribed_values = vec![0.0; layout.full_dimension()];
        for &(index, value) in layout.fixed_dofs() {
            prescribed_values[index] = value;
        }
        Ok(Self {
            layout,
            active_to_full,
            prescribed_values,
        })
    }

    /// Full/reduced coefficient layout.
    pub fn layout(&self) -> &BoundaryLayout {
        &self.layout
    }

    /// Sparse lift `P` in `x = P z + g`.
    pub fn active_to_full(&self) -> &LinearMap {
        &self.active_to_full
    }

    /// Full-space vector `g` of prescribed values.
    pub fn prescribed_values(&self) -> &[f64] {
        &self.prescribed_values
    }

    /// Dimension of the unconstrained coefficient vector.
    pub fn active_dimension(&self) -> usize {
        self.layout.active_dofs().len()
    }

    /// Dimension of the complete FEEC cochain.
    pub fn full_dimension(&self) -> usize {
        self.layout.full_dimension()
    }

    /// Whether all prescribed values are zero.
    pub fn is_homogeneous(&self) -> bool {
        self.layout
            .fixed_dofs()
            .iter()
            .all(|(_, value)| *value == 0.0)
    }

    /// Select the active coefficients from a full cochain.
    pub fn reduce_state(&self, full: &[f64]) -> Result<Vec<f64>> {
        if full.len() != self.full_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "full cochain length {} does not match boundary dimension {}",
                full.len(),
                self.full_dimension()
            )));
        }
        Ok(self
            .layout
            .active_dofs()
            .iter()
            .map(|index| full[*index])
            .collect())
    }

    /// Lift active coefficients and insert all prescribed values.
    pub fn lift_state(&self, active: &[f64]) -> Result<Vec<f64>> {
        if active.len() != self.active_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "active cochain length {} does not match reduced dimension {}",
                active.len(),
                self.active_dimension()
            )));
        }
        let mut full = self.prescribed_values.clone();
        for (active_index, full_index) in self.layout.active_dofs().iter().copied().enumerate() {
            full[full_index] = active[active_index];
        }
        Ok(full)
    }

    /// Lift active marginal variances; prescribed coefficients receive zero variance.
    pub fn lift_variances(&self, active: &[f64]) -> Result<Vec<f64>> {
        if active.len() != self.active_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "active variance count {} does not match reduced dimension {}",
                active.len(),
                self.active_dimension()
            )));
        }
        let mut full = vec![0.0; self.full_dimension()];
        for (active_index, full_index) in self.layout.active_dofs().iter().copied().enumerate() {
            full[full_index] = active[active_index];
        }
        Ok(full)
    }

    /// Restrict a full square operator to active rows and columns.
    pub fn reduce_square(&self, full: &SparseMat) -> Result<SparseMat> {
        if full.nrows() != self.full_dimension() || full.ncols() != self.full_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "full square operator is {}x{}, expected {}x{}",
                full.nrows(),
                full.ncols(),
                self.full_dimension(),
                self.full_dimension()
            )));
        }
        feg_infer::sparse::select_square_triplet_rows_cols(full, self.layout.active_dofs())
            .map_err(FeecGmrfError::Dimension)
    }

    /// Eliminate prescribed columns from a full-space map and fold their values into the bias.
    pub fn eliminate_map(&self, full: &LinearMap, bias: &[f64]) -> Result<EliminatedLinearMap> {
        if full.input_dimension() != self.full_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "full map input dimension {} does not match boundary dimension {}",
                full.input_dimension(),
                self.full_dimension()
            )));
        }
        if bias.len() != full.output_dimension() {
            return Err(FeecGmrfError::Dimension(format!(
                "map bias length {} does not match output dimension {}",
                bias.len(),
                full.output_dimension()
            )));
        }
        let (reduced, fixed_offset) = feg_infer::sparse::restrict_triplet_columns_and_fold_fixed(
            full.matrix(),
            bias,
            &self.state_layout(),
        )
        .map_err(FeecGmrfError::Dimension)?;
        Ok(EliminatedLinearMap {
            reduced: LinearMap::new(reduced)?,
            fixed_offset,
        })
    }

    fn eliminate_sparse_columns(&self, full: &SparseMat) -> Result<SparseMat> {
        let zero_bias = vec![0.0; full.nrows()];
        feg_infer::sparse::restrict_triplet_columns_and_fold_fixed(
            full,
            &zero_bias,
            &self.state_layout(),
        )
        .map(|(reduced, _)| reduced)
        .map_err(FeecGmrfError::Dimension)
    }

    fn state_layout(&self) -> DofLayout {
        DofLayout::new(
            self.full_dimension(),
            self.layout.active_dofs().to_vec(),
            self.layout
                .fixed_dofs()
                .iter()
                .map(|(index, value)| PrescribedDof {
                    index: *index,
                    value: *value,
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct QuadraticFullResidual;

    impl NonlinearResidualModel for QuadraticFullResidual {
        fn state_dimension(&self) -> usize {
            3
        }

        fn residual_dimension(&self) -> usize {
            1
        }

        fn residual_and_jacobian(
            &self,
            state: &[f64],
        ) -> std::result::Result<NonlinearResidualEvaluation, String> {
            Ok(NonlinearResidualEvaluation {
                residual: vec![state[0] * state[0] + state[1] - state[2]],
                jacobian: SparseMat::from_rows(
                    3,
                    &[vec![(0, 2.0 * state[0]), (1, 1.0), (2, -1.0)]],
                )?,
            })
        }
    }

    #[test]
    fn homogeneous_and_prescribed_conditions_share_one_elimination() {
        let homogeneous = EssentialBoundaryConditions::homogeneous([0, 2])
            .unwrap()
            .eliminate(4)
            .unwrap();
        assert!(homogeneous.is_homogeneous());
        assert_eq!(
            homogeneous.lift_state(&[3.0, 4.0]).unwrap(),
            [0.0, 3.0, 0.0, 4.0]
        );

        let prescribed = EssentialBoundaryConditions::prescribed(vec![0, 2], vec![1.5, -2.0])
            .unwrap()
            .eliminate(4)
            .unwrap();
        assert!(!prescribed.is_homogeneous());
        assert_eq!(
            prescribed.lift_state(&[3.0, 4.0]).unwrap(),
            [1.5, 3.0, -2.0, 4.0]
        );
        assert_eq!(
            prescribed.lift_variances(&[0.2, 0.3]).unwrap(),
            [0.0, 0.2, 0.0, 0.3]
        );
    }

    #[test]
    fn map_elimination_folds_prescribed_values_into_offset() {
        let elimination = EssentialBoundaryConditions::prescribed(vec![1], vec![4.0])
            .unwrap()
            .eliminate(3)
            .unwrap();
        let full = LinearMap::new(
            SparseMat::from_rows(3, &[vec![(0, 2.0), (1, -1.0), (2, 3.0)]]).unwrap(),
        )
        .unwrap();
        let eliminated = elimination.eliminate_map(&full, &[0.5]).unwrap();
        assert_eq!(eliminated.apply(&[1.0, 2.0]).unwrap(), [4.5]);
        assert_eq!(eliminated.fixed_offset(), [-3.5]);
    }

    #[test]
    fn eliminated_nonlinear_jacobian_matches_finite_differences() {
        let elimination = EssentialBoundaryConditions::prescribed(vec![1], vec![2.0])
            .unwrap()
            .eliminate(3)
            .unwrap();
        let model = EliminatedResidualModel::new(&QuadraticFullResidual, elimination).unwrap();
        let state = [3.0, 4.0];
        let evaluation = model.residual_and_jacobian(&state).unwrap();
        assert_eq!(evaluation.residual, [7.0]);

        let epsilon = 1.0e-6;
        for column in 0..state.len() {
            let mut plus = state;
            let mut minus = state;
            plus[column] += epsilon;
            minus[column] -= epsilon;
            let finite_difference = (model.residual(&plus).unwrap()[0]
                - model.residual(&minus).unwrap()[0])
                / (2.0 * epsilon);
            let jacobian = evaluation
                .jacobian
                .triplet_iter()
                .find_map(|(row, col, value)| (row == 0 && col == column).then_some(value))
                .unwrap();
            assert!((finite_difference - jacobian).abs() < 1.0e-8);
        }
    }

    #[test]
    fn invalid_boundary_declarations_fail_before_assembly() {
        assert!(EssentialBoundaryConditions::prescribed(vec![0], vec![]).is_err());
        assert!(EssentialBoundaryConditions::prescribed(vec![1, 1], vec![0.0, 2.0]).is_err());
        assert!(EssentialBoundaryConditions::prescribed(vec![0], vec![f64::NAN]).is_err());
        assert!(EssentialBoundaryConditions::homogeneous([3])
            .unwrap()
            .eliminate(3)
            .is_err());
        assert!(EssentialBoundaryConditions::homogeneous([0, 1])
            .unwrap()
            .eliminate(2)
            .is_err());
    }
}
