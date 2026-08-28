use feec_gmrf::prelude::*;
use manifold::gen::cartesian::CartesianMeshInfo;

fn two_by_two_exact_operators() -> ExactFormOperators {
    let identity = BoundaryLayout::identity;
    let lower = FormSpaceMetric::new(
        FormDegree::new(0, 2).unwrap(),
        SparseMat::diagonal(1, 4.0),
        identity(1),
    )
    .unwrap();
    let state = FormSpaceMetric::new(
        FormDegree::new(1, 2).unwrap(),
        SparseMat::from_rows(2, &[vec![(0, 2.0)], vec![(1, 3.0)]]).unwrap(),
        identity(2),
    )
    .unwrap();
    let upper = FormSpaceMetric::new(
        FormDegree::new(2, 2).unwrap(),
        SparseMat::diagonal(1, 5.0),
        identity(1),
    )
    .unwrap();
    let incoming = IncomingFormOperator::new(
        lower,
        LinearMap::new(SparseMat::from_rows(1, &[vec![(0, 1.0)], vec![(0, 2.0)]]).unwrap())
            .unwrap(),
        None,
    )
    .unwrap();
    let outgoing = OutgoingFormOperator::new(
        upper,
        LinearMap::new(SparseMat::from_rows(2, &[vec![(0, 2.0), (1, -1.0)]]).unwrap()).unwrap(),
        None,
    )
    .unwrap();
    ExactFormOperators::new(
        2,
        state,
        Some(incoming),
        Some(outgoing),
        ExactBoundaryDomain::UserSuppliedCompatibleSubcomplex {
            name: "two-by-two oracle".to_string(),
        },
        "independent exact fixture",
    )
    .unwrap()
}

fn dense_entries(matrix: &SparseMat) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; matrix.ncols()]; matrix.nrows()];
    for (row, column, value) in matrix.triplet_iter() {
        dense[row][column] += value;
    }
    dense
}

fn assert_dense_close(actual: &SparseMat, expected: &[[f64; 2]; 2]) {
    let actual = dense_entries(actual);
    for row in 0..2 {
        for column in 0..2 {
            let scale = expected[row][column].abs().max(1.0);
            assert!(
                (actual[row][column] - expected[row][column]).abs() <= 1.0e-11 * scale,
                "entry ({row}, {column}) was {}, expected {}",
                actual[row][column],
                expected[row][column]
            );
        }
    }
}

#[test]
fn exact_adjacent_builder_matches_small_closed_form_for_all_supported_alphas() {
    let expected = [
        [[92.0, -28.0], [-28.0, 68.0]],
        [[3370.0 / 3.0, -1442.0 / 3.0], [-1442.0 / 3.0, 1450.0 / 3.0]],
        [
            [126359.0 / 9.0, -59899.0 / 9.0],
            [-59899.0 / 9.0, 39791.0 / 9.0],
        ],
    ];
    for (alpha, expected) in [MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three]
        .into_iter()
        .zip(expected)
    {
        let prior = ExactMaternPriorBuilder::from_operators(two_by_two_exact_operators())
            .parameters(MaternParameters::new(alpha, 1.0, 2.0).unwrap())
            .build()
            .unwrap();
        assert_dense_close(prior.precision(), &expected);
        let metadata = prior.matern_metadata().unwrap();
        assert_eq!(
            metadata.representation(),
            MaternRepresentation::ExactEliminatedReference
        );
        assert_eq!(
            metadata.operator_source(),
            &MaternOperatorSource::UserSuppliedAdjacent {
                label: "independent exact fixture".to_string()
            }
        );
        assert_eq!(metadata.adjacent_dimensions(), [Some(1), Some(2), Some(1)]);
        assert_eq!(
            metadata.lower_adjoint_inverse(),
            InverseUse::Applied(MassInverseKind::ExactMaterialized)
        );
        let expected_recurrence = if alpha == MaternAlpha::One {
            InverseUse::NotUsed
        } else {
            InverseUse::Applied(MassInverseKind::ExactMaterialized)
        };
        assert_eq!(metadata.recurrence_inverse(), expected_recurrence);
    }
}

#[test]
fn feec_exact_builder_records_adjacent_full_relative_layouts() {
    let mesh = CartesianMeshInfo::new_unit(2, 2);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let prior = ExactMaternPriorBuilder::from_feec_homogeneous_full_relative(&topology, &metric, 1)
        .unwrap()
        .parameters(MaternParameters::new(MaternAlpha::One, 1.5, 1.0).unwrap())
        .build()
        .unwrap();

    assert!(prior.cochain_dimension() > prior.dimension());
    let metadata = prior.matern_metadata().unwrap();
    assert_eq!(
        metadata.representation(),
        MaternRepresentation::ExactEliminatedReference
    );
    assert_eq!(
        metadata.operator_source(),
        &MaternOperatorSource::FeecWhitney
    );
    assert_eq!(metadata.recurrence_inverse(), InverseUse::NotUsed);
    let MaternBoundarySemantics::AssembledHomogeneousFullRelative { layouts } = metadata.boundary()
    else {
        panic!("FEEC exact builder must report full-relative adjacent assembly");
    };
    assert!(layouts.lower().is_some());
    assert!(!layouts.state().fixed_dofs().is_empty());
    assert!(layouts.upper().is_some());
}

#[test]
fn exact_builder_enforces_materialization_guard_without_fallback() {
    let error = ExactMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(MaternParameters::new(MaternAlpha::Two, 1.0, 1.0).unwrap())
        .maximum_materialized_dimension(1)
        .unwrap()
        .build()
        .unwrap_err();
    assert!(matches!(
        error,
        FeecGmrfError::ExactReferenceSizeExceeded {
            dimension: 2,
            maximum: 1
        }
    ));
}

#[test]
fn exact_operator_bundle_rejects_a_noncomplex_pair() {
    let identity = BoundaryLayout::identity;
    let lower = FormSpaceMetric::new(
        FormDegree::new(0, 2).unwrap(),
        SparseMat::diagonal(1, 1.0),
        identity(1),
    )
    .unwrap();
    let state = FormSpaceMetric::new(
        FormDegree::new(1, 2).unwrap(),
        SparseMat::diagonal(2, 1.0),
        identity(2),
    )
    .unwrap();
    let upper = FormSpaceMetric::new(
        FormDegree::new(2, 2).unwrap(),
        SparseMat::diagonal(1, 1.0),
        identity(1),
    )
    .unwrap();
    let incoming = IncomingFormOperator::new(
        lower,
        LinearMap::new(SparseMat::from_rows(1, &[vec![(0, 1.0)], vec![(0, 1.0)]]).unwrap())
            .unwrap(),
        None,
    )
    .unwrap();
    let outgoing = OutgoingFormOperator::new(
        upper,
        LinearMap::new(SparseMat::from_rows(2, &[vec![(0, 1.0), (1, 1.0)]]).unwrap()).unwrap(),
        None,
    )
    .unwrap();
    assert!(ExactFormOperators::new(
        2,
        state,
        Some(incoming),
        Some(outgoing),
        ExactBoundaryDomain::UserSuppliedCompatibleSubcomplex {
            name: "invalid".to_string()
        },
        "invalid fixture"
    )
    .is_err());
}

#[test]
fn caller_supplied_operators_cannot_claim_feec_assembled_boundary_provenance() {
    let state = FormSpaceMetric::new(
        FormDegree::new(0, 0).unwrap(),
        SparseMat::diagonal(1, 1.0),
        BoundaryLayout::identity(1),
    )
    .unwrap();
    assert!(matches!(
        ExactFormOperators::new(
            0,
            state,
            None,
            None,
            ExactBoundaryDomain::HomogeneousFullRelative,
            "spoofed provenance"
        ),
        Err(FeecGmrfError::InvalidParameter(_))
    ));
}
