use feec_gmrf::prelude::*;
use nalgebra::{DMatrix, DVector};

fn two_by_two_exact_operators() -> ExactFormOperators {
    let lower = FormSpaceMetric::new(
        FormDegree::new(0, 2).unwrap(),
        SparseMat::diagonal(1, 4.0),
        BoundaryLayout::identity(1),
    )
    .unwrap();
    let state = FormSpaceMetric::new(
        FormDegree::new(1, 2).unwrap(),
        SparseMat::from_rows(2, &[vec![(0, 2.0)], vec![(1, 3.0)]]).unwrap(),
        BoundaryLayout::identity(2),
    )
    .unwrap();
    let upper = FormSpaceMetric::new(
        FormDegree::new(2, 2).unwrap(),
        SparseMat::diagonal(1, 5.0),
        BoundaryLayout::identity(1),
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
            name: "exact observation-space oracle".to_string(),
        },
        "independent exact observation-space fixture",
    )
    .unwrap()
}

fn rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}

fn assert_matrix_close(actual: &[Vec<f64>], expected: &DMatrix<f64>) {
    assert_eq!(actual.len(), expected.nrows());
    for (row, values) in actual.iter().enumerate() {
        assert_eq!(values.len(), expected.ncols());
        for (column, value) in values.iter().enumerate() {
            let scale = value.abs().max(expected[(row, column)].abs()).max(1.0);
            assert!((value - expected[(row, column)]).abs() <= 1.0e-12 * scale);
        }
    }
}

#[test]
fn operator_level_blocks_match_an_independent_dense_reference() {
    let kyy = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.5]);
    let kgy = DMatrix::from_row_slice(2, 2, &[1.0, 0.2, -0.3, 0.8]);
    let kgg = DMatrix::from_row_slice(2, 2, &[1.4, 0.1, 0.1, 1.1]);
    let noise_variance = 0.25;
    let observations = DVector::from_vec(vec![0.7, -0.2]);
    let noisy_kyy = &kyy + DMatrix::identity(2, 2) * noise_variance;
    let factor = noisy_kyy.cholesky().unwrap();
    let expected_mean = &kgy * factor.solve(&observations);
    let expected_covariance = &kgg - &kgy * factor.solve(&kgy.transpose());

    let blocks = ExactObservationCovarianceBlocks::new(rows(&kyy), rows(&kgy), rows(&kgg)).unwrap();
    assert_eq!(
        blocks.provenance(),
        &ExactObservationCovarianceProvenance::CallerSuppliedExactlySymmetric
    );
    let posterior =
        ExactObservationPosterior::from_prior_covariance_blocks(blocks, noise_variance).unwrap();
    let actual_mean = posterior.output_mean(observations.as_slice()).unwrap();
    for (actual, expected) in actual_mean.iter().zip(expected_mean.iter()) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_matrix_close(
        &posterior.posterior_output_covariance(),
        &expected_covariance,
    );
    let action = posterior
        .posterior_output_covariance_action(&[0.25, -0.75])
        .unwrap();
    let expected_action = expected_covariance * DVector::from_vec(vec![0.25, -0.75]);
    for (actual, expected) in action.iter().zip(expected_action.iter()) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    let diagnostics = posterior.diagnostics();
    assert_eq!(diagnostics.dense_factor_dimension, 2);
    assert_eq!(diagnostics.dense_factor_allocated_entries, 4);
    assert_eq!(diagnostics.persistent_f64_entries, 12);
}

#[test]
fn operator_level_blocks_accept_correlated_noise_covariance() {
    let kyy = DMatrix::from_row_slice(2, 2, &[1.4, 0.2, 0.2, 1.1]);
    let kgy = DMatrix::from_row_slice(2, 2, &[0.6, -0.1, 0.3, 0.8]);
    let kgg = DMatrix::from_row_slice(2, 2, &[1.2, 0.15, 0.15, 0.9]);
    let noise = DMatrix::from_row_slice(2, 2, &[0.3, 0.1, 0.1, 0.2]);
    let observations = DVector::from_vec(vec![0.4, -0.7]);
    let factor = (&kyy + &noise).cholesky().unwrap();
    let expected_mean = &kgy * factor.solve(&observations);
    let expected_covariance = &kgg - &kgy * factor.solve(&kgy.transpose());

    let blocks = ExactObservationCovarianceBlocks::new(rows(&kyy), rows(&kgy), rows(&kgg)).unwrap();
    let posterior = ExactObservationPosterior::from_prior_covariance_blocks_with_noise_covariance(
        blocks,
        rows(&noise),
    )
    .unwrap();
    let actual_mean = posterior.output_mean(observations.as_slice()).unwrap();
    for (actual, expected) in actual_mean.iter().zip(expected_mean.iter()) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_matrix_close(
        &posterior.posterior_output_covariance(),
        &expected_covariance,
    );
}

#[test]
fn caller_blocks_are_never_silently_symmetrized() {
    let error = ExactObservationCovarianceBlocks::new(
        vec![vec![1.0, 1.0e-15], vec![0.0, 1.0]],
        vec![vec![0.5, 0.25]],
        vec![vec![1.0]],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        FeecGmrfError::MatrixNotSymmetric { tolerance: 0.0, .. }
    ));
}

#[test]
fn supported_matern_adapter_matches_direct_precision_conditioning() {
    let parameters = MaternParameters::new(MaternAlpha::Two, 1.3, 0.7).unwrap();
    let exact = ExactMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(parameters)
        .build()
        .unwrap();
    let supported = SupportedMaternPriorBuilder::from_operators(two_by_two_exact_operators())
        .parameters(parameters)
        .build()
        .unwrap();
    let observation_map = LinearMap::weighted_row(2, &[(0, 1.0), (1, -0.5)]).unwrap();
    let output_map = LinearMap::new(
        SparseMat::from_rows(2, &[vec![(0, 0.75), (1, 0.2)], vec![(1, 1.0)]]).unwrap(),
    )
    .unwrap();
    let noise_variance = 0.6;
    let observation_value = vec![0.8];

    let exact_observation = ExactObservationPosterior::from_supported_matern_prior(
        &supported,
        &observation_map,
        &output_map,
        noise_variance,
        3,
    )
    .unwrap();
    let audit = match exact_observation.covariance_provenance() {
        ExactObservationCovarianceProvenance::SupportedMaternStacked { audit } => audit,
        ExactObservationCovarianceProvenance::CallerSuppliedExactlySymmetric => {
            panic!("supported adapter lost its audit provenance")
        }
    };
    assert_eq!(audit.dimension, 3);
    assert_eq!(
        audit.status,
        "accepted_interim_scale_relative_and_projected"
    );
    assert!(!audit.residual_aware);

    let observation = LinearObservation::new(
        observation_map.clone(),
        observation_value.clone(),
        GaussianNoise::variance(noise_variance).unwrap(),
    )
    .unwrap();
    let mut direct = LinearGaussianModelBuilder::new(exact)
        .observe(observation)
        .unwrap()
        .condition()
        .unwrap();
    let expected_mean = output_map.apply(direct.mean()).unwrap();
    let actual_mean = exact_observation.output_mean(&observation_value).unwrap();
    for (actual, expected) in actual_mean.iter().zip(expected_mean) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
    let expected_covariance = direct.pushforward_covariance(&output_map).unwrap();
    let actual_covariance = exact_observation.posterior_output_covariance();
    for (actual, expected) in actual_covariance
        .iter()
        .flatten()
        .zip(expected_covariance.iter().flatten())
    {
        assert!((actual - expected).abs() <= 1.0e-10);
    }

    let diagonal_blocks = ExactObservationDiagonalBlocks::from_supported_matern_factor(
        &supported.factor().unwrap(),
        &observation_map,
        &output_map,
        1,
        2,
    )
    .unwrap();
    let diagonal = ExactObservationDiagonalPosterior::from_blocks_with_noise_covariance(
        diagonal_blocks,
        vec![vec![noise_variance]],
    )
    .unwrap();
    let diagonal_mean = diagonal.output_mean(&observation_value).unwrap();
    for (actual, expected) in diagonal_mean.iter().zip(actual_mean) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
    for (index, variance) in diagonal.posterior_output_variances().iter().enumerate() {
        assert!((variance - actual_covariance[index][index]).abs() <= 1.0e-10);
    }
    let input = vec![0.3, -0.8];
    let prior_action = supported
        .factor()
        .unwrap()
        .covariance_action(&output_map.apply_transpose(&input).unwrap())
        .and_then(|action| output_map.apply(&action))
        .unwrap();
    let diagonal_action = diagonal
        .posterior_output_covariance_action_from_prior_action(&input, &prior_action)
        .unwrap();
    let full_action = exact_observation
        .posterior_output_covariance_action(&input)
        .unwrap();
    for (actual, expected) in diagonal_action.iter().zip(full_action) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }

    let guarded = ExactObservationPosterior::from_supported_matern_prior(
        &supported,
        &LinearMap::selector(2, &[0, 1]).unwrap(),
        &LinearMap::identity(2),
        noise_variance,
        3,
    );
    assert!(matches!(
        guarded,
        Err(FeecGmrfError::DenseDiagnosticSizeExceeded {
            dimension: 4,
            maximum: 3
        })
    ));
}
