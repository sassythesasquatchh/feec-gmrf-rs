use feec_gmrf::prelude::*;
use manifold::gen::cartesian::CartesianMeshInfo;

#[test]
fn compatible_modal_precision_gives_matern_derivative_spectrum() {
    let mass = SparseMat::from_rows(2, &[vec![(0, 2.0)], vec![(1, 3.0)]]).unwrap();
    let laplacian = SparseMat::from_rows(2, &[vec![(0, 4.0)], vec![(1, 15.0)]]).unwrap();
    let inverse = SparseMat::from_rows(2, &[vec![(0, 0.5)], vec![(1, 1.0 / 3.0)]]).unwrap();
    let kappa = 1.2_f64;
    let tau = 0.7_f64;
    let prior = MaternPriorBuilder::from_operators(
        FormOperators::new(FormDegree::new(1, 3).unwrap(), 3, mass, laplacian).unwrap(),
    )
    .mass_inverse(MassInversePolicy::Provided(inverse))
    .parameters(MaternParameters::new(MaternAlpha::Two, kappa, tau).unwrap())
    .build_spectrum_matched_potential()
    .unwrap();

    // In mass-normalized Hodge coordinates d multiplies a positive mode by
    // sqrt(lambda), so its variance must be 1/[tau² (kappa²+lambda)²].
    let mut posterior = LinearGaussianModelBuilder::new(prior).condition().unwrap();
    let derivative = LinearMap::weighted_rows(
        2,
        &[
            vec![(0, (2.0_f64 * 2.0).sqrt())],
            vec![(1, (3.0_f64 * 5.0).sqrt())],
        ],
    )
    .unwrap();
    let covariance = posterior.pushforward_covariance(&derivative).unwrap();
    for (i, lambda) in [2.0_f64, 5.0].iter().enumerate() {
        let expected = 1.0 / (tau.powi(2) * (kappa.powi(2) + lambda).powi(2));
        assert!((covariance[i][i] - expected).abs() < 1e-13);
    }
    assert!(covariance[0][1].abs() < 1e-13);
}

#[test]
fn relative_prior_regularizes_gradients_without_a_hard_gauge() {
    let (topology, coords) = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let prior = MaternPriorBuilder::from_feec_homogeneous_full_relative_with_projected_inverses(
        &topology, &metric, 1,
    )
    .unwrap()
    .build_spectrum_matched_potential()
    .unwrap();
    let relative =
        formoniq::relative::assemble_whitney_full_relative_operators(&topology, &metric, 1)
            .unwrap();
    let gradient = LinearMap::from_feec_csr(relative.incoming().unwrap().active()).unwrap();
    let phi = vec![1.0; gradient.input_dimension()];
    let z = gradient.apply(&phi).unwrap();
    assert!(z.iter().any(|value| value.abs() > 0.5));
    let curl = LinearMap::from_feec_csr(relative.outgoing().unwrap().active()).unwrap();
    assert!(curl
        .apply(&z)
        .unwrap()
        .iter()
        .all(|value| value.abs() < 1e-12));
    assert!(prior.mahalanobis_distance(&z).unwrap() > 0.0);
    let boundary = prior.boundary_elimination().unwrap();
    let q = boundary.lift_state(&z).unwrap();
    for &(index, _) in boundary.layout().fixed_dofs() {
        assert_eq!(q[index], 0.0);
    }
}

#[test]
fn spectrum_matched_prior_rejects_harmonic_and_unsupported_parameters() {
    for (laplacian, alpha, kappa) in [
        (0.0, MaternAlpha::Two, 1.0),
        (1.0, MaternAlpha::Three, 1.0),
        (1.0, MaternAlpha::Two, 0.0),
    ] {
        let operators = FormOperators::new(
            FormDegree::new(1, 3).unwrap(),
            3,
            SparseMat::diagonal(1, 1.0),
            SparseMat::diagonal(1, laplacian),
        )
        .unwrap();
        assert!(MaternPriorBuilder::from_operators(operators)
            .parameters(MaternParameters::new(alpha, kappa, 1.0).unwrap())
            .build_spectrum_matched_potential()
            .is_err());
    }
}
