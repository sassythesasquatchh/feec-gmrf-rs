use feg_core::{BoundaryRegionSpec, BoundarySpec, BoundaryTreatment};
use feg_infer::{
    model::spacetime::SpacetimeLinearObservationBuilder,
    prior::spacetime::{
        build_0form_spacetime_prior, build_1form_spacetime_prior,
        build_2form_spacetime_prior_with_coords, Hodge1PriorConfig, Hodge2PriorConfig,
        ScalarPriorConfig, SpacetimePriorConfig,
    },
};
use formoniq::problems::reduced_linear::MassInverseApproximation;
use manifold::gen::cartesian::CartesianMeshInfo;

fn times() -> SpacetimePriorConfig {
    SpacetimePriorConfig {
        times: vec![0.0, 0.1, 0.3],
    }
}

fn assert_factorizable(prior: &feg_infer::prior::spacetime::SpacetimePrior) {
    assert_eq!(prior.precision.block_count(), 3);
    assert_eq!(prior.precision.dimension(), 3 * prior.state_dimension());
    prior
        .precision
        .to_sparse()
        .cholesky_sqrt_lower()
        .expect("spacetime precision should factorize");
}

#[test]
fn zero_form_spacetime_prior_factorizes() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
        "hard",
        vec![0],
        vec![0.0],
        BoundaryTreatment::HardEssential,
    ));
    let prior = build_0form_spacetime_prior(
        &topology,
        &metric,
        &boundary,
        ScalarPriorConfig {
            kappa: 1.0,
            tau: 1.0,
        },
        &times(),
    )
    .unwrap();
    assert_factorizable(&prior);
}

#[test]
fn one_form_spacetime_prior_factorizes_with_separate_inverse_choices() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let prior = build_1form_spacetime_prior(
        &topology,
        &metric,
        &BoundarySpec::default(),
        Hodge1PriorConfig {
            kappa: 1.0,
            tau: 1.0,
            lower_mass_inverse: MassInverseApproximation::RowSumLumped,
            state_mass_inverse: MassInverseApproximation::WhitneyProjected,
        },
        &times(),
    )
    .unwrap();
    assert_eq!(prior.state_dimension(), topology.nsimplices(1));
    assert_factorizable(&prior);
}

#[test]
fn projected_two_form_3d_uses_lower_one_form_inverse_and_factorizes() {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let prior = build_2form_spacetime_prior_with_coords(
        &topology,
        &coords,
        &metric,
        &BoundarySpec::default(),
        Hodge2PriorConfig {
            kappa: 0.8,
            tau: 1.2,
            lower_mass_inverse: MassInverseApproximation::WhitneyProjected,
            state_mass_inverse: MassInverseApproximation::WhitneyProjected,
        },
        &times(),
    )
    .unwrap();
    assert_eq!(prior.state_dimension(), topology.nsimplices(2));
    assert_factorizable(&prior);
}

#[test]
fn soft_boundaries_become_time_stacked_observations() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
        "soft",
        vec![0],
        vec![2.0],
        BoundaryTreatment::SoftEssential { variance: 0.25 },
    ));
    let prior = build_0form_spacetime_prior(
        &topology,
        &metric,
        &boundary,
        ScalarPriorConfig {
            kappa: 1.0,
            tau: 1.0,
        },
        &times(),
    )
    .unwrap();
    let mut builder = SpacetimeLinearObservationBuilder::new(3, prior.layout());
    builder
        .add_soft_boundary_observations(3, prior.soft_observations())
        .unwrap();
    let observations = builder.finish();
    assert_eq!(observations.matrix.nrows(), 3);
    assert_eq!(observations.matrix.ncols(), 3 * prior.state_dimension());
}

#[test]
fn invalid_matern_parameters_are_rejected() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    for spatial in [
        ScalarPriorConfig {
            kappa: -1.0,
            tau: 1.0,
        },
        ScalarPriorConfig {
            kappa: 1.0,
            tau: 0.0,
        },
    ] {
        assert!(build_0form_spacetime_prior(
            &topology,
            &metric,
            &BoundarySpec::default(),
            spatial,
            &times(),
        )
        .is_err());
    }
}
