use feg_core::{
    BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, LinearGaussianMeasurementSpec,
    SparseTriplet, SparseTripletMatrix,
};
use formoniq::reduction::{EssentialBoundarySpec, PrescribedDof};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptedBoundarySpec {
    pub essential: EssentialBoundarySpec,
    pub soft_state_measurements: Vec<LinearGaussianMeasurementSpec>,
}

pub fn adapt_boundary_spec(
    boundary: &BoundarySpec,
    state_dimension: usize,
    auxiliary_dimension: usize,
) -> Result<AdaptedBoundarySpec, String> {
    let mut hard_state = Vec::new();
    let mut hard_auxiliary = Vec::new();
    let mut soft_state_measurements = Vec::new();
    let mut hard_state_indices = BTreeSet::new();
    let mut hard_auxiliary_indices = BTreeSet::new();
    let mut soft_state_indices = BTreeSet::new();

    for region in &boundary.state_regions {
        validate_region(region, state_dimension)?;
        match region.treatment {
            BoundaryTreatment::HardEssential => collect_hard(
                region,
                &mut hard_state_indices,
                &soft_state_indices,
                &mut hard_state,
            )?,
            BoundaryTreatment::SoftEssential { variance } => {
                if !variance.is_finite() || variance <= 0.0 {
                    return Err(format!(
                        "soft-essential region `{}` variance must be finite and positive",
                        region.name
                    ));
                }
                for &dof in &region.dofs {
                    if hard_state_indices.contains(&dof) {
                        return Err(format!(
                            "state dof {dof} appears in both hard and soft boundary regions"
                        ));
                    }
                    if !soft_state_indices.insert(dof) {
                        return Err(format!("state dof {dof} appears in multiple soft regions"));
                    }
                }
                soft_state_measurements.push(LinearGaussianMeasurementSpec {
                    name: region.name.clone(),
                    operator: SparseTripletMatrix::from_triplets(
                        region.dofs.len(),
                        state_dimension,
                        region
                            .dofs
                            .iter()
                            .enumerate()
                            .map(|(row, &col)| SparseTriplet {
                                row,
                                col,
                                value: 1.0,
                            }),
                    ),
                    observations: region.values.clone(),
                    bias: vec![0.0; region.dofs.len()],
                    variance,
                });
            }
            BoundaryTreatment::Natural => {}
        }
    }

    for region in &boundary.auxiliary_regions {
        validate_region(region, auxiliary_dimension)?;
        match region.treatment {
            BoundaryTreatment::HardEssential => collect_hard(
                region,
                &mut hard_auxiliary_indices,
                &BTreeSet::new(),
                &mut hard_auxiliary,
            )?,
            BoundaryTreatment::SoftEssential { .. } => {
                return Err(format!(
                    "soft auxiliary boundary region `{}` is not supported",
                    region.name
                ));
            }
            BoundaryTreatment::Natural => {}
        }
    }

    Ok(AdaptedBoundarySpec {
        essential: EssentialBoundarySpec {
            state: hard_state,
            auxiliary: hard_auxiliary,
        },
        soft_state_measurements,
    })
}

fn validate_region(region: &BoundaryRegionSpec, dimension: usize) -> Result<(), String> {
    if region.dofs.len() != region.values.len() {
        return Err(format!(
            "boundary region `{}` has mismatched dof/value counts",
            region.name
        ));
    }
    for (&dof, &value) in region.dofs.iter().zip(&region.values) {
        if dof >= dimension {
            return Err(format!(
                "boundary region `{}` references dof {dof} outside dimension {dimension}",
                region.name
            ));
        }
        if !value.is_finite() {
            return Err(format!(
                "boundary region `{}` has non-finite value for dof {dof}",
                region.name
            ));
        }
    }
    Ok(())
}

fn collect_hard(
    region: &BoundaryRegionSpec,
    hard_indices: &mut BTreeSet<usize>,
    soft_indices: &BTreeSet<usize>,
    prescribed: &mut Vec<PrescribedDof>,
) -> Result<(), String> {
    for (&index, &value) in region.dofs.iter().zip(&region.values) {
        if soft_indices.contains(&index) {
            return Err(format!(
                "dof {index} appears in both hard and soft boundary regions"
            ));
        }
        if !hard_indices.insert(index) {
            return Err(format!(
                "dof {index} appears in multiple hard boundary regions"
            ));
        }
        prescribed.push(PrescribedDof { index, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_hard_soft_and_natural_regions() {
        let boundary = BoundarySpec::default()
            .with_state_region(BoundaryRegionSpec::new(
                "hard",
                vec![0],
                vec![1.0],
                BoundaryTreatment::HardEssential,
            ))
            .with_state_region(BoundaryRegionSpec::new(
                "soft",
                vec![1],
                vec![2.0],
                BoundaryTreatment::SoftEssential { variance: 0.25 },
            ))
            .with_state_region(BoundaryRegionSpec::new(
                "natural",
                vec![2],
                vec![0.0],
                BoundaryTreatment::Natural,
            ));
        let adapted = adapt_boundary_spec(&boundary, 3, 0).unwrap();
        assert_eq!(adapted.essential.state.len(), 1);
        assert_eq!(adapted.soft_state_measurements.len(), 1);
    }

    #[test]
    fn rejects_overlap_and_soft_auxiliary() {
        let overlap = BoundarySpec::default()
            .with_state_region(BoundaryRegionSpec::new(
                "hard",
                vec![0],
                vec![0.0],
                BoundaryTreatment::HardEssential,
            ))
            .with_state_region(BoundaryRegionSpec::new(
                "soft",
                vec![0],
                vec![0.0],
                BoundaryTreatment::SoftEssential { variance: 1.0 },
            ));
        assert!(adapt_boundary_spec(&overlap, 1, 0).is_err());

        let auxiliary = BoundarySpec::default().with_auxiliary_region(BoundaryRegionSpec::new(
            "soft-aux",
            vec![0],
            vec![0.0],
            BoundaryTreatment::SoftEssential { variance: 1.0 },
        ));
        assert!(adapt_boundary_spec(&auxiliary, 0, 1).is_err());
    }

    #[test]
    fn rejects_duplicate_hard_and_soft_state_dofs() {
        let duplicate_hard = BoundarySpec::default()
            .with_state_region(BoundaryRegionSpec::new(
                "hard-a",
                vec![0],
                vec![0.0],
                BoundaryTreatment::HardEssential,
            ))
            .with_state_region(BoundaryRegionSpec::new(
                "hard-b",
                vec![0],
                vec![0.0],
                BoundaryTreatment::HardEssential,
            ));
        assert!(adapt_boundary_spec(&duplicate_hard, 1, 0).is_err());

        let duplicate_soft = BoundarySpec::default()
            .with_state_region(BoundaryRegionSpec::new(
                "soft-a",
                vec![0],
                vec![0.0],
                BoundaryTreatment::SoftEssential { variance: 1.0 },
            ))
            .with_state_region(BoundaryRegionSpec::new(
                "soft-b",
                vec![0],
                vec![0.0],
                BoundaryTreatment::SoftEssential { variance: 1.0 },
            ));
        assert!(adapt_boundary_spec(&duplicate_soft, 1, 0).is_err());
    }
}
