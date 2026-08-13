use feec_gmrf::prelude::*;
use feec_gmrf::prior::matern_recurrence;
use feg_infer::prior::matern::generic::{
    build_hodge_laplacian_form, build_matern_system_matrix_form,
    build_projected_or_top_degree_mass_inverse,
};
use manifold::gen::cartesian::CartesianMeshInfo;
use rand::{rngs::StdRng, SeedableRng};

struct FullSumResidual;

impl ResidualModel for FullSumResidual {
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
            residual: vec![state.iter().sum()],
            jacobian: SparseMat::from_rows(3, &[vec![(0, 1.0), (1, 1.0), (2, 1.0)]])?,
        })
    }
}

#[test]
fn canonical_matern_recurrence_matches_legacy_for_all_supported_degrees() {
    for intrinsic_dimension in [2, 3, 4] {
        let mesh = CartesianMeshInfo::new_unit_scaled(intrinsic_dimension, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        for degree in 0..=intrinsic_dimension {
            let hodge = build_hodge_laplacian_form(&topology, &metric, degree).unwrap();
            let system = build_matern_system_matrix_form(&hodge, 1.25);
            let inverse = build_projected_or_top_degree_mass_inverse(
                &topology,
                &metric,
                degree,
                &hodge.mass_u,
            )
            .unwrap();
            let system_contract = feg_infer::sparse::feec_csr_to_core_triplet(&system);
            let inverse_contract = feg_infer::sparse::feec_csr_to_core_triplet(&inverse);

            for (alpha, legacy_alpha) in [
                (MaternAlpha::One, feg_infer::prior::matern::MaternAlpha::One),
                (MaternAlpha::Two, feg_infer::prior::matern::MaternAlpha::Two),
                (
                    MaternAlpha::Three,
                    feg_infer::prior::matern::MaternAlpha::Three,
                ),
            ] {
                let canonical =
                    matern_recurrence(&system_contract, &inverse_contract, alpha, 0.75).unwrap();
                let legacy = feg_infer::prior::matern::build_lindgren_precision_from_system(
                    &system,
                    &inverse,
                    legacy_alpha,
                    0.75,
                );
                let legacy = feg_infer::sparse::feec_csr_to_core_triplet(&legacy);
                assert_sparse_close(&canonical, &legacy, 1e-10);
            }
        }
    }
}

#[test]
fn downstream_user_can_build_condition_and_push_forward_a_form_prior() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let prior = MaternPriorBuilder::from_feec(&topology, &metric, 1)
        .unwrap()
        .parameters(MaternParameters::new(MaternAlpha::Two, 1.0, 1.0).unwrap())
        .build()
        .unwrap();
    let dimension = prior.dimension();
    let observation_map =
        LinearMap::new(SparseMat::from_rows(dimension, &[vec![(0, 1.0)]]).unwrap()).unwrap();
    let observation = LinearObservation::new(
        observation_map.clone(),
        vec![0.25],
        GaussianNoise::variance(0.01).unwrap(),
    )
    .unwrap();
    let derived = DerivedQuantity::new("sensor", observation_map).unwrap();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .observe(observation)
        .unwrap()
        .derive(derived)
        .unwrap()
        .condition()
        .unwrap();

    assert_eq!(posterior.mean().len(), dimension);
    assert_eq!(posterior.derived_mean("sensor").unwrap().len(), 1);
    assert_eq!(posterior.derived_variances("sensor").unwrap().len(), 1);
    let covariance = posterior.derived_covariance("sensor").unwrap();
    assert_eq!(covariance.len(), 1);
    assert_eq!(covariance[0].len(), 1);
    assert!(covariance[0][0].is_finite() && covariance[0][0] > 0.0);
}

#[test]
fn practical_range_parameters_record_the_conversion_convention() {
    let parameters = MaternParameters::from_practical_range(MaternAlpha::Two, 0.4, 2, 1.5)
        .expect("valid practical-range parameters");
    assert_eq!(parameters.kappa, 8.0_f64.sqrt() / 0.4);
    assert_eq!(
        parameters.convention,
        MaternParameterConvention::PracticalRange {
            practical_range: 0.4,
            intrinsic_dimension: 2,
        }
    );
}

#[test]
fn hard_constraints_are_applied_by_generic_gmrf_algebra() {
    let prior = GaussianPrior::new(vec![0.0, 0.0], SparseMat::diagonal(2, 1.0)).unwrap();
    let sum =
        LinearMap::new(SparseMat::from_rows(2, &[vec![(0, 1.0), (1, 1.0)]]).unwrap()).unwrap();
    let constraint = LinearConstraint::new(sum, vec![2.0]).unwrap();
    let posterior = LinearGaussianModelBuilder::new(prior)
        .constrain(constraint)
        .unwrap()
        .condition()
        .unwrap();
    assert!((posterior.mean()[0] - 1.0).abs() < 1e-12);
    assert!((posterior.mean()[1] - 1.0).abs() < 1e-12);
}

#[test]
fn operator_level_spacetime_builder_produces_expected_block_dimension() {
    let prior = SpacetimePriorBuilder::from_operators(
        SparseMat::diagonal(2, 1.0),
        SparseMat::diagonal(2, 0.5),
        SparseMat::diagonal(2, 1.0),
        SparseMat::diagonal(2, 1.0),
    )
    .unwrap()
    .times(vec![0.0, 0.1, 0.25])
    .build()
    .unwrap();
    assert_eq!(prior.precision().block_count(), 3);
    assert_eq!(prior.precision().dimension(), 6);
}

#[test]
fn prescribed_essential_values_propagate_through_linear_uq() {
    let degree = FormDegree::new(0, 1).unwrap();
    let prior = MaternPriorBuilder::from_operators(
        FormOperators::new(degree, 1, SparseMat::diagonal(3, 1.0), SparseMat::new(3, 3)).unwrap(),
    )
    .parameters(MaternParameters::new(MaternAlpha::One, 1.0, 1.0).unwrap())
    .essential_boundary_conditions(
        EssentialBoundaryConditions::prescribed(vec![1], vec![2.0]).unwrap(),
    )
    .build()
    .unwrap();

    assert_eq!(prior.dimension(), 2);
    assert_eq!(prior.cochain_dimension(), 3);

    let full_map =
        LinearMap::new(SparseMat::from_rows(3, &[vec![(0, 1.0), (1, 3.0), (2, 2.0)]]).unwrap())
            .unwrap();
    let observation = LinearObservation::new(
        full_map.clone(),
        vec![10.0],
        GaussianNoise::variance(1.0).unwrap(),
    )
    .unwrap();
    let derived = DerivedQuantity::new("full_qoi", full_map).unwrap();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .observe(observation)
        .unwrap()
        .derive(derived)
        .unwrap()
        .condition()
        .unwrap();

    assert_vector_close(posterior.latent_mean(), &[2.0 / 3.0, 4.0 / 3.0], 1e-12);
    assert_vector_close(
        posterior.cochain_mean(),
        &[2.0 / 3.0, 2.0, 4.0 / 3.0],
        1e-12,
    );
    assert_vector_close(
        &posterior.derived_mean("full_qoi").unwrap(),
        &[28.0 / 3.0],
        1e-12,
    );
    assert_vector_close(
        &posterior.cochain_variances().unwrap(),
        &[5.0 / 6.0, 0.0, 1.0 / 3.0],
        1e-12,
    );
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..4 {
        assert_eq!(posterior.sample_cochain(&mut rng).unwrap()[1], 2.0);
    }
}

#[test]
fn full_space_constraints_account_for_prescribed_values() {
    let prior = GaussianPrior::new(vec![0.0; 3], SparseMat::diagonal(3, 1.0))
        .unwrap()
        .condition_on_essential_boundary(
            EssentialBoundaryConditions::prescribed(vec![1], vec![2.0]).unwrap(),
        )
        .unwrap();
    let full_sum =
        LinearMap::new(SparseMat::from_rows(3, &[vec![(0, 1.0), (1, 1.0)]]).unwrap()).unwrap();
    let constraint = LinearConstraint::new(full_sum, vec![3.0]).unwrap();
    let posterior = LinearGaussianModelBuilder::new(prior)
        .constrain(constraint)
        .unwrap()
        .condition()
        .unwrap();
    assert_vector_close(posterior.cochain_mean(), &[1.0, 2.0, 0.0], 1e-12);
}

#[test]
fn boundary_only_constraints_are_removed_or_rejected() {
    let prior = GaussianPrior::new(vec![0.0; 2], SparseMat::diagonal(2, 1.0))
        .unwrap()
        .condition_on_essential_boundary(
            EssentialBoundaryConditions::prescribed(vec![1], vec![2.0]).unwrap(),
        )
        .unwrap();
    let fixed_selector =
        LinearMap::new(SparseMat::from_rows(2, &[vec![(1, 1.0)]]).unwrap()).unwrap();

    let redundant = LinearConstraint::new(fixed_selector.clone(), vec![2.0]).unwrap();
    let posterior = LinearGaussianModelBuilder::new(prior.clone())
        .constrain(redundant)
        .unwrap()
        .condition()
        .unwrap();
    assert_eq!(posterior.cochain_mean(), [0.0, 2.0]);

    let inconsistent = LinearConstraint::new(fixed_selector, vec![3.0]).unwrap();
    assert!(LinearGaussianModelBuilder::new(prior)
        .constrain(inconsistent)
        .is_err());
}

#[test]
fn arbitrary_gaussian_boundary_elimination_uses_conditional_mean() {
    let precision =
        SparseMat::from_rows(2, &[vec![(0, 2.0), (1, 1.0)], vec![(0, 1.0), (1, 2.0)]]).unwrap();
    let prior = GaussianPrior::new(vec![0.0, 0.0], precision)
        .unwrap()
        .condition_on_essential_boundary(
            EssentialBoundaryConditions::prescribed(vec![1], vec![3.0]).unwrap(),
        )
        .unwrap();
    assert_vector_close(prior.mean(), &[-1.5], 1e-12);
    assert_eq!(prior.precision(), &SparseMat::diagonal(1, 2.0));
}

#[test]
fn essential_elimination_precedes_the_matern_mass_recurrence() {
    let mass = SparseMat::from_rows(
        3,
        &[
            vec![(0, 2.0), (1, 1.0)],
            vec![(0, 1.0), (1, 2.0), (2, 1.0)],
            vec![(1, 1.0), (2, 2.0)],
        ],
    )
    .unwrap();
    let operators = FormOperators::new(
        FormDegree::new(0, 1).unwrap(),
        1,
        mass,
        SparseMat::new(3, 3),
    )
    .unwrap();
    let prior = MaternPriorBuilder::from_operators(operators)
        .parameters(MaternParameters::new(MaternAlpha::Two, 1.0, 1.0).unwrap())
        .mass_inverse(MassInversePolicy::RowSumLumped)
        .essential_boundary_conditions(
            EssentialBoundaryConditions::prescribed(vec![1], vec![5.0]).unwrap(),
        )
        .build()
        .unwrap();

    assert_eq!(prior.precision(), &SparseMat::diagonal(2, 2.0));
    assert_eq!(
        prior
            .boundary_elimination()
            .unwrap()
            .lift_state(prior.mean())
            .unwrap(),
        [0.0, 5.0, 0.0]
    );
}

#[test]
fn pre_reduced_form_operators_retain_their_boundary_layout() {
    let layout = BoundaryLayout::new(3, vec![0, 2], vec![(1, -1.25)]).unwrap();
    let operators = FormOperators::new(
        FormDegree::new(1, 2).unwrap(),
        2,
        SparseMat::diagonal(2, 1.0),
        SparseMat::diagonal(2, 0.5),
    )
    .unwrap()
    .with_boundary_layout(layout)
    .unwrap();
    let prior = MaternPriorBuilder::from_operators(operators)
        .build()
        .unwrap();
    assert_eq!(prior.dimension(), 2);
    assert_eq!(prior.cochain_dimension(), 3);
    assert_vector_close(&prior.cochain_mean().unwrap(), &[0.0, -1.25, 0.0], 1e-12);
}

#[test]
fn feec_matern_boundary_path_builds_across_degrees_and_alphas() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    for degree in 0..=2 {
        for alpha in [MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three] {
            let prior = MaternPriorBuilder::from_feec(&topology, &metric, degree)
                .unwrap()
                .parameters(MaternParameters::new(alpha, 1.0, 1.0).unwrap())
                .essential_boundary_conditions(
                    EssentialBoundaryConditions::homogeneous([0]).unwrap(),
                )
                .build()
                .unwrap();
            assert_eq!(prior.cochain_dimension(), prior.dimension() + 1);
            LinearGaussianModelBuilder::new(prior)
                .condition()
                .expect("boundary-reduced Matérn precision should factorize");
        }
    }
}

#[test]
fn nonlinear_facade_eliminates_full_cochain_residuals() {
    let prior = GaussianPrior::new(vec![0.0; 3], SparseMat::diagonal(3, 1.0))
        .unwrap()
        .condition_on_essential_boundary(
            EssentialBoundaryConditions::prescribed(vec![1], vec![2.0]).unwrap(),
        )
        .unwrap();
    let term = NonlinearResidualTerm::zero(
        "full_sum",
        &FullSumResidual,
        GaussianNoise::variance(1.0).unwrap(),
    )
    .unwrap();
    let config = feec_gmrf::model::nonlinear::GaussNewtonConfig {
        linear_solve: feec_gmrf::model::nonlinear::GaussNewtonLinearSolve::DirectCholesky,
        ..Default::default()
    };
    let posterior = NonlinearLaplaceModelBuilder::new(prior)
        .config(config)
        .residual(term)
        .unwrap()
        .solve()
        .unwrap();

    assert!(posterior.converged);
    assert_vector_close(
        posterior.cochain_map(),
        &[-2.0 / 3.0, 2.0, -2.0 / 3.0],
        1e-10,
    );
    assert_vector_close(
        posterior.cochain_variances(),
        &[2.0 / 3.0, 0.0, 2.0 / 3.0],
        1e-10,
    );
}

fn assert_sparse_close(lhs: &SparseMat, rhs: &SparseMat, tolerance: f64) {
    assert_eq!((lhs.nrows(), lhs.ncols()), (rhs.nrows(), rhs.ncols()));
    let mut dense = vec![0.0; lhs.nrows() * lhs.ncols()];
    for (row, col, value) in lhs.triplet_iter() {
        dense[row * lhs.ncols() + col] += value;
    }
    for (row, col, value) in rhs.triplet_iter() {
        dense[row * lhs.ncols() + col] -= value;
    }
    let scale = dense
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    assert!(dense.iter().all(|value| value.abs() <= tolerance * scale));
}

fn assert_vector_close(lhs: &[f64], rhs: &[f64], tolerance: f64) {
    assert_eq!(lhs.len(), rhs.len());
    assert!(lhs
        .iter()
        .zip(rhs)
        .all(|(left, right)| (left - right).abs() <= tolerance));
}
