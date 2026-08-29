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
            name: "two-by-two supported oracle".to_string(),
        },
        "independent supported-law fixture",
    )
    .unwrap()
}

fn assert_scalar_close(actual: f64, expected: f64, context: &str) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        (actual - expected).abs() <= 1.0e-10 * scale,
        "{context}: actual {actual:.16e}, expected {expected:.16e}"
    );
}

fn assert_vector_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_scalar_close(*actual, *expected, &format!("{context}[{index}]"));
    }
}

fn assert_covariance_close(actual: &[Vec<f64>], expected: &[Vec<f64>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: row mismatch");
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context}: column mismatch in row {row}"
        );
        for (column, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_scalar_close(*actual, *expected, &format!("{context}[{row}, {column}]"));
        }
    }
}

#[test]
fn exact_supported_law_matches_exact_elimination_for_all_supported_alphas() {
    for alpha in [MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three] {
        let parameters = MaternParameters::new(alpha, 1.3, 0.7).unwrap();
        let operators = two_by_two_exact_operators();
        let exact = ExactMaternPriorBuilder::from_operators(operators.clone())
            .parameters(parameters)
            .build()
            .unwrap();
        let supported = SupportedMaternPriorBuilder::from_operators(operators)
            .parameters(parameters)
            .build()
            .unwrap();

        assert_eq!(supported.state_dimension(), 2);
        assert_eq!(supported.cochain_dimension(), 2);
        assert_eq!(supported.layout().support_dimension(), 2);
        assert_eq!(supported.layout().state(), 0..2);
        let expected_lower_auxiliaries = if alpha == MaternAlpha::Three { 2 } else { 1 };
        assert_eq!(
            supported.layout().lower_adjoint_auxiliaries().len(),
            expected_lower_auxiliaries
        );
        let expected_recurrence_states = usize::from(alpha != MaternAlpha::One);
        assert_eq!(
            supported.layout().recurrence_states().len(),
            expected_recurrence_states
        );

        let metadata = supported.matern_metadata();
        assert_eq!(
            metadata.representation(),
            MaternRepresentation::ExactSupported
        );
        assert_eq!(
            metadata.lower_adjoint_inverse(),
            InverseUse::Applied(MassInverseKind::ConstrainedSupport)
        );
        assert_eq!(
            metadata.recurrence_inverse(),
            if alpha == MaternAlpha::One {
                InverseUse::NotUsed
            } else {
                InverseUse::Applied(MassInverseKind::ConstrainedSupport)
            }
        );

        let mut exact_factor = exact.factor().unwrap();
        let supported_factor = supported.factor().unwrap();
        let diagnostics = supported_factor.diagnostics();
        assert_eq!(
            diagnostics.supported_energy_factor_nnz,
            diagnostics.supported_energy_factor.lower_factor_nnz
        );
        if let Some(constraint_factor) = diagnostics.constraint_gram_factor {
            assert_eq!(
                diagnostics.constraint_gram_factor_nnz,
                constraint_factor.lower_factor_nnz
            );
        }
        assert!(matches!(
            supported_factor.kkt_factor_nonzeros(),
            Err(FeecGmrfError::Unsupported(_))
        ));
        assert_vector_close(
            supported_factor.mean(),
            exact_factor.mean(),
            &format!("alpha {alpha:?} prior mean"),
        );
        let exact_covariance = exact_factor
            .pushforward_covariance(&LinearMap::identity(2))
            .unwrap();
        assert_covariance_close(
            &supported
                .pushforward_covariance(&LinearMap::identity(2))
                .unwrap(),
            &exact_covariance,
            &format!("alpha {alpha:?} prior convenience covariance"),
        );
        let audited_supported_covariance = supported_factor
            .pushforward_covariance_audited(&LinearMap::identity(2))
            .unwrap();
        assert_eq!(
            audited_supported_covariance.symmetric[0][1],
            audited_supported_covariance.symmetric[1][0]
        );
        assert!(!audited_supported_covariance.audit.residual_aware);
        assert_covariance_close(
            &audited_supported_covariance.symmetric,
            &exact_covariance,
            &format!("alpha {alpha:?} prior covariance"),
        );

        let observation = LinearObservation::new(
            LinearMap::weighted_row(2, &[(0, 1.0), (1, -0.5)]).unwrap(),
            vec![0.8],
            GaussianNoise::variance(0.6).unwrap(),
        )
        .unwrap();
        let mut exact_observed = LinearGaussianModelBuilder::new(exact.clone())
            .observe(observation.clone())
            .unwrap()
            .condition()
            .unwrap();
        let supported_observed = supported
            .condition(std::slice::from_ref(&observation), &[])
            .unwrap();
        assert_vector_close(
            supported_observed.mean(),
            exact_observed.mean(),
            &format!("alpha {alpha:?} observed mean"),
        );
        assert_covariance_close(
            &supported_observed
                .pushforward_covariance(&LinearMap::identity(2))
                .unwrap(),
            &exact_observed
                .pushforward_covariance(&LinearMap::identity(2))
                .unwrap(),
            &format!("alpha {alpha:?} observed covariance"),
        );

        let prepared_supported = supported.prepare(&[observation], &[]).unwrap();
        assert_eq!(prepared_supported.observation_row_counts(), [1]);
        assert_vector_close(
            prepared_supported.condition().unwrap().mean(),
            supported_observed.mean(),
            &format!("alpha {alpha:?} prepared default mean"),
        );
        let replacement_values = vec![vec![-0.35]];
        let replacement_observation = LinearObservation::new(
            LinearMap::weighted_row(2, &[(0, 1.0), (1, -0.5)]).unwrap(),
            replacement_values[0].clone(),
            GaussianNoise::variance(0.6).unwrap(),
        )
        .unwrap();
        let direct_replacement = supported
            .condition(&[replacement_observation], &[])
            .unwrap();
        assert_vector_close(
            &prepared_supported
                .latent_mean_with_observation_values(&replacement_values)
                .unwrap(),
            direct_replacement.mean(),
            &format!("alpha {alpha:?} prepared replacement mean"),
        );
        assert_vector_close(
            prepared_supported
                .condition_with_observation_values(&replacement_values)
                .unwrap()
                .mean(),
            direct_replacement.mean(),
            &format!("alpha {alpha:?} prepared replacement posterior"),
        );

        let constrained_observation = LinearObservation::new(
            LinearMap::weighted_row(2, &[(0, 1.0), (1, -0.5)]).unwrap(),
            vec![0.8],
            GaussianNoise::variance(0.6).unwrap(),
        )
        .unwrap();
        let constraint = LinearConstraint::new(
            LinearMap::weighted_row(2, &[(0, 1.0), (1, 1.0)]).unwrap(),
            vec![0.25],
        )
        .unwrap();
        let mut exact_constrained = LinearGaussianModelBuilder::new(exact)
            .observe(constrained_observation.clone())
            .unwrap()
            .constrain(constraint.clone())
            .unwrap()
            .condition()
            .unwrap();
        let supported_constrained = supported
            .condition(&[constrained_observation], &[constraint])
            .unwrap();
        assert_vector_close(
            supported_constrained.mean(),
            exact_constrained.mean(),
            &format!("alpha {alpha:?} constrained mean"),
        );
        assert_covariance_close(
            &supported_constrained
                .pushforward_covariance(&LinearMap::identity(2))
                .unwrap(),
            &exact_constrained
                .pushforward_covariance(&LinearMap::identity(2))
                .unwrap(),
            &format!("alpha {alpha:?} constrained covariance"),
        );

        let quantity = LinearMap::weighted_row(2, &[(0, 1.25), (1, -0.75)]).unwrap();
        let supported_variance = supported_constrained
            .pushforward_variances(&quantity)
            .unwrap();
        let exact_variance = exact_constrained.pushforward_covariance(&quantity).unwrap();
        assert_eq!(supported_variance.len(), 1);
        assert_scalar_close(
            supported_variance[0],
            exact_variance[0][0],
            &format!("alpha {alpha:?} constrained transformed variance"),
        );
        let outputs =
            LinearMap::weighted_rows(2, &[vec![(0, 1.25), (1, -0.75)], vec![(0, -0.4), (1, 0.9)]])
                .unwrap();
        assert_covariance_close(
            &supported_constrained
                .pushforward_covariance(&outputs)
                .unwrap(),
            &exact_constrained.pushforward_covariance(&outputs).unwrap(),
            &format!("alpha {alpha:?} constrained output covariance"),
        );
    }
}

#[test]
fn supported_weighted_physical_trace_matches_dense_reference_and_hutchinson() {
    let supported = SupportedMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(MaternParameters::new(MaternAlpha::Two, 1.3, 0.7).unwrap())
        .build()
        .unwrap();
    let outputs =
        LinearMap::weighted_rows(2, &[vec![(0, 1.25), (1, -0.75)], vec![(0, -0.4), (1, 0.9)]])
            .unwrap();
    let weights = [0.25, 0.75];
    let covariance = supported.pushforward_covariance(&outputs).unwrap();
    let expected = weights
        .iter()
        .enumerate()
        .map(|(index, weight)| weight * covariance[index][index])
        .sum::<f64>();

    let exact = supported
        .pushforward_weighted_variance_estimate(&outputs, &weights, VarianceMethod::Exact)
        .unwrap();
    assert_scalar_close(
        exact.weighted_trace,
        expected,
        "exact weighted physical trace",
    );

    let hutchinson = supported
        .pushforward_weighted_variance_estimate(
            &outputs,
            &weights,
            VarianceMethod::Hutchinson(HutchinsonVarianceConfig::new(4_096, 16, 74).unwrap()),
        )
        .unwrap();
    assert!(hutchinson.weighted_trace_relative_standard_error.is_some());
    assert!(
        (hutchinson.weighted_trace - expected).abs() <= 0.05 * expected,
        "Hutchinson weighted trace {} differs from dense reference {}",
        hutchinson.weighted_trace,
        expected
    );

    let state_weight =
        SparseMat::from_rows(2, &[vec![(0, 2.0), (1, 0.25)], vec![(0, 0.25), (1, 3.0)]]).unwrap();
    let state_covariance = supported
        .pushforward_covariance(&LinearMap::identity(2))
        .unwrap();
    let state_expected = state_weight
        .triplet_iter()
        .map(|(row, column, value)| value * state_covariance[column][row])
        .sum::<f64>();
    let state_exact = supported
        .weighted_covariance_trace(&state_weight, VarianceMethod::Exact)
        .unwrap();
    assert_scalar_close(
        state_exact.value,
        state_expected,
        "exact supported sparse weighted trace",
    );
    let state_hutchinson = supported
        .weighted_covariance_trace(
            &state_weight,
            VarianceMethod::Hutchinson(HutchinsonVarianceConfig::new(16_384, 16, 91).unwrap()),
        )
        .unwrap();
    assert!(
        (state_hutchinson.value - state_expected).abs() <= 0.05 * state_expected,
        "Hutchinson supported weighted trace {} differs from dense reference {}",
        state_hutchinson.value,
        state_expected
    );
}

#[test]
fn supported_feec_prior_preserves_full_relative_boundary_layout() {
    let mesh = CartesianMeshInfo::new_unit(2, 2);
    let (topology, coordinates) = mesh.compute_coord_complex();
    let metric = coordinates.to_edge_lengths(&topology);
    let prior =
        SupportedMaternPriorBuilder::from_feec_homogeneous_full_relative(&topology, &metric, 1)
            .unwrap()
            .parameters(MaternParameters::new(MaternAlpha::Two, 1.5, 1.0).unwrap())
            .build()
            .unwrap();

    assert!(prior.cochain_dimension() > prior.state_dimension());
    assert_eq!(
        prior.matern_metadata().operator_source(),
        &MaternOperatorSource::FeecWhitney
    );
    let MaternBoundarySemantics::AssembledHomogeneousFullRelative { layouts } =
        prior.matern_metadata().boundary()
    else {
        panic!("supported FEEC prior must report its full-relative assembly domain");
    };
    assert!(!layouts.state().fixed_dofs().is_empty());

    let posterior = prior.factor().unwrap();
    assert_eq!(
        posterior.cochain_mean().unwrap().len(),
        prior.cochain_dimension()
    );
    let cochain_variances = posterior
        .pushforward_variances(&LinearMap::identity(prior.cochain_dimension()))
        .unwrap();
    let audited_full_covariance = posterior
        .pushforward_covariance_audited(&LinearMap::identity(prior.cochain_dimension()))
        .unwrap();
    let full_covariance = audited_full_covariance.symmetric;
    assert!(full_covariance
        .iter()
        .enumerate()
        .all(|(row, values)| values
            .iter()
            .enumerate()
            .all(|(column, value)| *value == full_covariance[column][row])));
    for (fixed, _) in layouts.state().fixed_dofs() {
        assert_eq!(cochain_variances[*fixed], 0.0);
        assert!(full_covariance[*fixed].iter().all(|value| *value == 0.0));
        assert!(full_covariance.iter().all(|row| row[*fixed] == 0.0));
    }
    let active_covariance = posterior
        .pushforward_covariance(&LinearMap::identity(prior.state_dimension()))
        .unwrap();
    for (active_row, &full_row) in layouts.state().active_dofs().iter().enumerate() {
        for (active_column, &full_column) in layouts.state().active_dofs().iter().enumerate() {
            assert_scalar_close(
                full_covariance[full_row][full_column],
                active_covariance[active_row][active_column],
                "full-cochain boundary-eliminated covariance",
            );
        }
    }
}
