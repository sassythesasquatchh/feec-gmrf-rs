use crate::sparse::{core_triplet_to_feec_csr, feec_csr_to_gmrf, restrict_columns_and_fold_fixed};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use feg_core::LinearGaussianMeasurementSpec;
use formoniq::reduction::DofLayout;
use gmrf_core::{StackedObservationSystem, TimeStackedObservationBuilder};

#[derive(Debug, Clone)]
pub struct SpacetimeLinearObservationBuilder {
    inner: TimeStackedObservationBuilder,
    layout: DofLayout,
}

impl SpacetimeLinearObservationBuilder {
    pub fn new(slice_count: usize, layout: &DofLayout) -> Self {
        Self {
            inner: TimeStackedObservationBuilder::new(slice_count, layout.reduced_dimension()),
            layout: layout.clone(),
        }
    }

    pub fn push_slice_block_from_full(
        &mut self,
        slice_index: usize,
        full_block: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        let (reduced, reduced_bias) =
            restrict_columns_and_fold_fixed(full_block, bias, &self.layout)?;
        self.inner
            .push_slice_block(
                slice_index,
                &feec_csr_to_gmrf(&reduced),
                observations.as_slice(),
                reduced_bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn push_slice_block_from_reduced(
        &mut self,
        slice_index: usize,
        reduced_block: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        self.inner
            .push_slice_block(
                slice_index,
                &feec_csr_to_gmrf(reduced_block),
                observations.as_slice(),
                bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn push_transition_block_from_full(
        &mut self,
        left_slice: usize,
        left_full: &FeecCsr,
        right_full: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        let (left_reduced, left_bias) =
            restrict_columns_and_fold_fixed(left_full, bias, &self.layout)?;
        let (right_reduced, final_bias) =
            restrict_columns_and_fold_fixed(right_full, &left_bias, &self.layout)?;
        self.inner
            .push_transition_block(
                left_slice,
                &feec_csr_to_gmrf(&left_reduced),
                &feec_csr_to_gmrf(&right_reduced),
                observations.as_slice(),
                final_bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn push_transition_block_from_reduced(
        &mut self,
        left_slice: usize,
        left_reduced: &FeecCsr,
        right_reduced: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        self.inner
            .push_transition_block(
                left_slice,
                &feec_csr_to_gmrf(left_reduced),
                &feec_csr_to_gmrf(right_reduced),
                observations.as_slice(),
                bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn add_soft_boundary_observations(
        &mut self,
        slice_count: usize,
        measurements: &[LinearGaussianMeasurementSpec],
    ) -> Result<(), String> {
        for time_index in 0..slice_count {
            for measurement in measurements {
                self.push_soft_measurement(time_index, measurement)?;
            }
        }
        Ok(())
    }

    pub fn finish(self) -> StackedObservationSystem {
        self.inner.finish()
    }

    fn push_soft_measurement(
        &mut self,
        slice_index: usize,
        measurement: &LinearGaussianMeasurementSpec,
    ) -> Result<(), String> {
        let block = core_triplet_to_feec_csr(&measurement.operator);
        let observations = FeecVector::from_vec(measurement.observations.clone());
        let bias = FeecVector::from_vec(measurement.bias.clone());
        self.push_slice_block_from_full(
            slice_index,
            &block,
            &observations,
            &bias,
            measurement.variance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::CooMatrix as FeecCoo;
    use feg_core::{SparseTriplet, SparseTripletMatrix};
    use formoniq::reduction::PrescribedDof;

    #[test]
    fn hard_fixed_columns_are_folded_into_bias() {
        let mut block = FeecCoo::new(1, 3);
        block.push(0, 0, 2.0);
        block.push(0, 1, -1.0);
        block.push(0, 2, 3.0);
        let block = FeecCsr::from(&block);
        let bias = FeecVector::from_vec(vec![0.5]);
        let layout = DofLayout::from_parts(
            3,
            vec![0, 2],
            vec![PrescribedDof {
                index: 1,
                value: 4.0,
            }],
        )
        .unwrap();

        let (reduced, folded_bias) =
            restrict_columns_and_fold_fixed(&block, &bias, &layout).unwrap();

        let mut rows = reduced
            .triplet_iter()
            .map(|(r, c, v)| (r, c, *v))
            .collect::<Vec<_>>();
        rows.sort_by_key(|(r, c, _)| (*r, *c));
        assert_eq!(rows, vec![(0, 0, 2.0), (0, 1, 3.0)]);
        assert!((folded_bias[0] + 3.5).abs() < 1e-12);
    }

    #[test]
    fn soft_observations_are_added_for_each_time_slice() {
        let layout = DofLayout::identity(2);
        let measurement = LinearGaussianMeasurementSpec {
            name: "soft".to_string(),
            operator: SparseTripletMatrix::from_triplets(
                1,
                2,
                [SparseTriplet {
                    row: 0,
                    col: 1,
                    value: 1.0,
                }],
            ),
            observations: vec![2.0],
            bias: vec![0.0],
            variance: 0.25,
        };

        let mut builder = SpacetimeLinearObservationBuilder::new(3, &layout);
        builder
            .add_soft_boundary_observations(3, &[measurement])
            .unwrap();
        let system = builder.finish();
        assert_eq!(system.matrix.nrows(), 3);
        assert_eq!(system.matrix.ncols(), 6);
        assert_eq!(system.observations.as_slice(), &[4.0, 4.0, 4.0]);
    }
}
