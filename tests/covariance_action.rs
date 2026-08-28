use feec_gmrf::prelude::*;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "entry {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn factored_prior_covariance_action_matches_exact_inverse() {
    let precision =
        SparseMat::from_rows(2, &[vec![(0, 4.0), (1, 1.0)], vec![(0, 1.0), (1, 3.0)]]).unwrap();
    let prior = GaussianPrior::new(vec![0.0, 0.0], precision).unwrap();
    let mut factored = prior.factor().unwrap();

    let action = factored.covariance_action(&[2.0, -1.0]).unwrap();

    assert_close(&action, &[7.0 / 11.0, -6.0 / 11.0]);
}

#[test]
fn factored_prior_reports_factor_and_repeated_solve_resources() {
    let precision =
        SparseMat::from_rows(2, &[vec![(0, 4.0), (1, 1.0)], vec![(0, 1.0), (1, 3.0)]]).unwrap();
    let prior = GaussianPrior::new(vec![0.25, -0.5], precision).unwrap();
    let (factored, mut resources) = prior.factor_with_resource_report().unwrap();

    assert_eq!(factored.mean(), &[0.25, -0.5]);
    assert_eq!(resources.solver, SparseDirectSolverKind::Cholesky);
    assert_eq!(resources.dimension, 2);
    assert_eq!(resources.coefficient_nnz, 4);
    assert!(matches!(
        resources.factor_storage_nonzeros,
        ResourceMeasurement::Measured(_)
    ));
    assert!(matches!(
        resources.analysis.wall_time,
        ResourceMeasurement::Measured(_)
    ));
    assert!(matches!(
        resources.numeric_factorization.wall_time,
        ResourceMeasurement::Measured(_)
    ));
    assert_eq!(resources.solve.calls, 0);

    let action = factored
        .covariance_action_with_resource_report(&[2.0, -1.0], &mut resources)
        .unwrap();
    assert_close(&action, &[7.0 / 11.0, -6.0 / 11.0]);
    assert_eq!(resources.solve.calls, 1);
    assert_eq!(resources.solve.right_hand_sides, 1);
    assert!(matches!(
        resources.solve.wall_time,
        ResourceMeasurement::Measured(_)
    ));
}

#[test]
fn posterior_covariance_action_includes_observations() {
    let prior = GaussianPrior::new(vec![0.0, 0.0], SparseMat::diagonal(2, 1.0)).unwrap();
    let observation =
        LinearObservation::at_indices(2, &[0], vec![7.0], GaussianNoise::variance(1.0).unwrap())
            .unwrap();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .observe(observation)
        .unwrap()
        .condition()
        .unwrap();

    let action = posterior.covariance_action(&[2.0, 3.0]).unwrap();

    assert_close(&action, &[1.0, 3.0]);
}

#[test]
fn posterior_covariance_action_respects_hard_constraints() {
    let prior = GaussianPrior::new(vec![0.0, 0.0], SparseMat::diagonal(2, 1.0)).unwrap();
    let sum =
        LinearMap::new(SparseMat::from_rows(2, &[vec![(0, 1.0), (1, 1.0)]]).unwrap()).unwrap();
    let constraint = LinearConstraint::new(sum, vec![4.0]).unwrap();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .constrain(constraint)
        .unwrap()
        .condition()
        .unwrap();

    let action = posterior.covariance_action(&[2.0, -1.0]).unwrap();

    assert_close(posterior.mean(), &[2.0, 2.0]);
    assert_close(&action, &[1.5, -1.5]);
    assert!((action[0] + action[1]).abs() < 1.0e-12);
}

#[test]
fn public_covariance_actions_return_typed_dimension_errors() {
    let prior = GaussianPrior::new(vec![0.0, 0.0], SparseMat::diagonal(2, 1.0)).unwrap();
    let mut factored = prior.clone().factor().unwrap();
    let mut posterior = LinearGaussianModelBuilder::new(prior).condition().unwrap();

    assert!(matches!(
        factored.covariance_action(&[1.0]),
        Err(FeecGmrfError::Dimension(_))
    ));
    assert!(matches!(
        posterior.covariance_action(&[1.0, 2.0, 3.0]),
        Err(FeecGmrfError::Dimension(_))
    ));
}
