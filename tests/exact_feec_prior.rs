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

fn assert_matrices_close(actual: &SparseMat, expected: &SparseMat) {
    let actual = dense_entries(actual);
    let expected = dense_entries(expected);
    assert_eq!(actual.len(), expected.len());
    for (actual_row, expected_row) in actual.iter().zip(&expected) {
        assert_eq!(actual_row.len(), expected_row.len());
        for (&actual, &expected) in actual_row.iter().zip(expected_row) {
            let scale = expected.abs().max(1.0);
            assert!((actual - expected).abs() <= 1.0e-11 * scale);
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
fn adjacent_builder_selects_lower_and_recurrence_inverse_sites_independently() {
    let parameters = MaternParameters::new(MaternAlpha::Two, 1.0, 2.0).unwrap();
    let exact = ExactMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(parameters)
        .build()
        .unwrap();
    let exact_lower = SparseMat::diagonal(1, 0.25);
    let exact_state = SparseMat::from_rows(2, &[vec![(0, 0.5)], vec![(1, 1.0 / 3.0)]]).unwrap();

    let lower_provided = AdjacentMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(parameters)
        .lower_adjoint_inverse(EliminatedMassInversePolicy::Provided(exact_lower.clone()))
        .build()
        .unwrap();
    let recurrence_provided =
        AdjacentMaternPriorBuilder::from_operators(two_by_two_exact_operators())
            .parameters(parameters)
            .recurrence_inverse(EliminatedMassInversePolicy::Provided(exact_state.clone()))
            .build()
            .unwrap();
    let both_provided = AdjacentMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(parameters)
        .lower_adjoint_inverse(EliminatedMassInversePolicy::Provided(exact_lower))
        .recurrence_inverse(EliminatedMassInversePolicy::Provided(exact_state))
        .build()
        .unwrap();

    for prior in [&lower_provided, &recurrence_provided, &both_provided] {
        assert_matrices_close(prior.precision(), exact.precision());
    }
    let lower_metadata = lower_provided.matern_metadata().unwrap();
    assert_eq!(
        lower_metadata.representation(),
        MaternRepresentation::MixedAdjacentEliminatedApproximation
    );
    assert_eq!(
        lower_metadata.lower_adjoint_inverse(),
        InverseUse::Applied(MassInverseKind::Provided)
    );
    assert_eq!(
        lower_metadata.recurrence_inverse(),
        InverseUse::Applied(MassInverseKind::ExactMaterialized)
    );
    let recurrence_metadata = recurrence_provided.matern_metadata().unwrap();
    assert_eq!(
        recurrence_metadata.lower_adjoint_inverse(),
        InverseUse::Applied(MassInverseKind::ExactMaterialized)
    );
    assert_eq!(
        recurrence_metadata.recurrence_inverse(),
        InverseUse::Applied(MassInverseKind::Provided)
    );
    assert_eq!(
        both_provided.matern_metadata().unwrap().representation(),
        MaternRepresentation::ProvidedAdjacentEliminatedApproximation
    );
}

#[test]
fn adjacent_builder_validates_each_provided_inverse_against_its_metric_space() {
    let parameters = MaternParameters::new(MaternAlpha::Two, 1.0, 1.0).unwrap();
    let lower_error = AdjacentMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(parameters)
        .lower_adjoint_inverse(EliminatedMassInversePolicy::Provided(SparseMat::diagonal(
            2, 1.0,
        )))
        .build()
        .unwrap_err();
    assert!(matches!(lower_error, FeecGmrfError::Dimension(_)));

    let recurrence_error = AdjacentMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(parameters)
        .recurrence_inverse(EliminatedMassInversePolicy::Provided(SparseMat::diagonal(
            1, 1.0,
        )))
        .build()
        .unwrap_err();
    assert!(matches!(recurrence_error, FeecGmrfError::Dimension(_)));
}

#[test]
fn boundary_first_projected_constructor_builds_full_relative_one_form_prior() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let prior =
        AdjacentMaternPriorBuilder::from_feec_homogeneous_full_relative_with_projected_inverses(
            &topology, &metric, 1,
        )
        .unwrap()
        .parameters(MaternParameters::new(MaternAlpha::Two, 4.0, 1.0).unwrap())
        .build()
        .unwrap();

    assert!(prior.cochain_dimension() > prior.dimension());
    let metadata = prior.matern_metadata().unwrap();
    assert_eq!(
        metadata.representation(),
        MaternRepresentation::ProvidedAdjacentEliminatedApproximation
    );
    assert_eq!(
        metadata.lower_adjoint_inverse(),
        InverseUse::Applied(MassInverseKind::Provided)
    );
    assert_eq!(
        metadata.recurrence_inverse(),
        InverseUse::Applied(MassInverseKind::Provided)
    );
    prior.factor().unwrap();
}

#[test]
fn adjacent_builder_materialization_guard_applies_only_to_exact_sites() {
    let provided_lower = AdjacentMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(MaternParameters::new(MaternAlpha::One, 1.0, 1.0).unwrap())
        .lower_adjoint_inverse(EliminatedMassInversePolicy::Provided(SparseMat::diagonal(
            1, 0.25,
        )))
        .maximum_materialized_dimension(1)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(provided_lower.dimension(), 2);

    let guarded_recurrence =
        AdjacentMaternPriorBuilder::from_operators(two_by_two_exact_operators())
            .parameters(MaternParameters::new(MaternAlpha::Two, 1.0, 1.0).unwrap())
            .lower_adjoint_inverse(EliminatedMassInversePolicy::Provided(SparseMat::diagonal(
                1, 0.25,
            )))
            .maximum_materialized_dimension(1)
            .unwrap()
            .build()
            .unwrap_err();
    assert!(matches!(
        guarded_recurrence,
        FeecGmrfError::ExactReferenceSizeExceeded {
            dimension: 2,
            maximum: 1
        }
    ));
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
