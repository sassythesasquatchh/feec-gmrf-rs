use feg_core::{
    BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, GaussianPriorSpec, SparseTripletMatrix,
};
use feg_infer::adapters::FeecResidualAdapter;
use feg_infer::boundary::adapt_boundary_spec;
use feg_infer::linear_pde::{
    LinearPdeDerivedQuantitySpec, LinearPdeVarianceConfig, LinearPdeVarianceMode,
};
use feg_infer::nonlinear::{
    solve_nonlinear_laplace, GaussNewtonConfig, GaussianNoiseModel, NonlinearLaplaceProblem,
    NonlinearResidualTerm,
};
use feg_infer::physical::{
    build_reduced_magnetic_flux_density_operator_3d,
    build_reduced_scalar_az_magnetic_flux_density_operator_2d,
};
use feg_infer::sparse::core_triplet_to_gmrf as sparse_from_core;
use formoniq::problems::nonlinear_magnetostatic::{
    build_reduced_scalar_az_magnetostatic_2d, build_reduced_vector_potential_magnetostatic_3d,
    NonlinearMagnetostaticAssemblyConfig, NonlinearReluctivityLaw, ReducedScalarAzMagnetostatic2d,
    ReducedVectorPotentialMagnetostatic3d,
};
use formoniq::problems::residual::ResidualModel as FeecResidualModel;
use manifold::gen::cartesian::CartesianMeshInfo;
use manifold::geometry::coord::mesh::MeshCoords;
use manifold::topology::complex::Complex;
use std::f64::consts::PI;

fn diagonal_prior(dimension: usize, precision: f64) -> GaussianPriorSpec {
    let mut matrix = SparseTripletMatrix::new(dimension, dimension);
    for index in 0..dimension {
        matrix.push(index, index, precision);
    }
    GaussianPriorSpec {
        mean: vec![0.0; dimension],
        precision: matrix,
    }
}

fn manufactured_model() -> (
    Complex,
    MeshCoords,
    ReducedScalarAzMagnetostatic2d,
    Vec<f64>,
) {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 3, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let boundary_vertices = topology
        .boundary_subcomplex_simplices(0)
        .into_iter()
        .map(|simplex| simplex.kidx)
        .collect::<Vec<_>>();
    let boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
        "outer",
        boundary_vertices.clone(),
        vec![0.0; boundary_vertices.len()],
        BoundaryTreatment::HardEssential,
    ));
    let material = NonlinearReluctivityLaw::new(1.1, 0.25).unwrap();
    let boundary = adapt_boundary_spec(&boundary, topology.nsimplices(0), 0)
        .unwrap()
        .essential;
    let source_free = build_reduced_scalar_az_magnetostatic_2d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(material, boundary),
    )
    .unwrap();
    let truth = source_free
        .layout()
        .active_dofs
        .iter()
        .map(|&vertex| {
            let point = coords.coord(vertex);
            0.4 * (PI * point[0]).sin() * (PI * point[1]).sin()
        })
        .collect::<Vec<_>>();
    let source = source_free.manufactured_source(&truth).unwrap();
    let model = source_free.with_source(source).unwrap();
    (topology, coords, model, truth)
}

fn manufactured_model_3d() -> (
    Complex,
    MeshCoords,
    ReducedVectorPotentialMagnetostatic3d,
    Vec<f64>,
) {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let material = NonlinearReluctivityLaw::new(1.0, 0.25).unwrap();
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(
            material,
            adapt_boundary_spec(&BoundarySpec::default(), topology.nsimplices(1), 0)
                .unwrap()
                .essential,
        ),
    )
    .unwrap();
    let truth = smooth_edge_truth_3d(&source_free, &topology, &coords, 0.2);
    let source = source_free.manufactured_source(&truth).unwrap();
    let model = source_free.with_source(source).unwrap();
    (topology, coords, model, truth)
}

fn smooth_edge_truth_3d(
    model: &ReducedVectorPotentialMagnetostatic3d,
    topology: &Complex,
    coords: &MeshCoords,
    scale: f64,
) -> Vec<f64> {
    let mut full = vec![0.0; topology.nsimplices(1)];
    for edge in topology.edges().handle_iter() {
        let [v0, v1]: [usize; 2] = edge.vertices.clone().try_into().unwrap();
        let p0 = coords.coord(v0);
        let p1 = coords.coord(v1);
        let midpoint = [
            0.5 * (p0[0] + p1[0]),
            0.5 * (p0[1] + p1[1]),
            0.5 * (p0[2] + p1[2]),
        ];
        let tangent = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let vector_potential = [
            scale * (PI * midpoint[1]).sin() * (PI * midpoint[2]).sin(),
            scale * (PI * midpoint[2]).sin() * (PI * midpoint[0]).sin(),
            scale * (PI * midpoint[0]).sin() * (PI * midpoint[1]).sin(),
        ];
        full[edge.kidx()] = vector_potential[0] * tangent[0]
            + vector_potential[1] * tangent[1]
            + vector_potential[2] * tangent[2];
    }
    model
        .layout()
        .active_dofs
        .iter()
        .map(|&edge| full[edge])
        .collect()
}

fn l2_norm(values: impl AsRef<[f64]>) -> f64 {
    values
        .as_ref()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

#[test]
fn nonlinear_magnetostatic_feec_model_solves_with_laplace_precision_and_flux_variance() {
    let (topology, coords, model, truth) = manufactured_model();
    let dimension = model.reduced_dimension();
    let initial_guess = vec![0.0; dimension];
    let initial_residual = model
        .residual_and_jacobian(&initial_guess)
        .unwrap()
        .residual;
    let initial_residual_norm = l2_norm(&initial_residual);
    let flux_operator = build_reduced_scalar_az_magnetic_flux_density_operator_2d(
        &topology,
        &coords,
        model.layout(),
    )
    .unwrap();

    let residual_model = FeecResidualAdapter::new(&model);
    let problem = NonlinearLaplaceProblem {
        prior: diagonal_prior(dimension, 1e-8),
        residual_terms: vec![NonlinearResidualTerm::zero(
            "scalar_Az_magnetostatic_2d",
            &residual_model,
            GaussianNoiseModel::ScalarVariance(1e-8),
        )],
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: vec![LinearPdeDerivedQuantitySpec {
            name: "cell_B".to_string(),
            operator: flux_operator,
        }],
    };

    let result = solve_nonlinear_laplace(
        &problem,
        &GaussNewtonConfig {
            initial_guess: Some(initial_guess),
            max_iterations: 30,
            step_tolerance: 1e-10,
            gradient_tolerance: 1e-9,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::Exact,
                ..LinearPdeVarianceConfig::default()
            },
            ..GaussNewtonConfig::default()
        },
    )
    .expect("FEEC nonlinear magnetostatic problem should solve");

    assert!(result.converged);
    let final_residual = &result.final_residuals[0].residual;
    assert!(
        l2_norm(final_residual) <= 1e-5 * initial_residual_norm,
        "nonlinear solve did not sufficiently reduce the FEEC residual"
    );
    let map_error = result
        .map
        .iter()
        .zip(truth.iter())
        .map(|(estimate, exact)| (estimate - exact).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        map_error <= 1e-4 * l2_norm(&truth).max(1.0),
        "MAP error {map_error} was too large for the manufactured magnetostatic solution"
    );

    sparse_from_core(&result.posterior_precision)
        .cholesky_sqrt_lower()
        .expect("final Laplace precision should factorize");
    assert_eq!(result.posterior_precision.nrows(), dimension);
    assert_eq!(result.posterior_precision.ncols(), dimension);
    assert!(result
        .posterior_variance
        .iter()
        .all(|value| value.is_finite()));

    let flux_variance = result
        .derived_variances
        .get("cell_B")
        .expect("cell flux derived covariance should be reported");
    assert_eq!(
        flux_variance.posterior_variance.len(),
        2 * model.num_elements()
    );
    assert!(flux_variance
        .posterior_variance
        .iter()
        .all(|value| value.is_finite() && *value >= -1e-12));
}

#[test]
fn nonlinear_magnetostatic_3d_feec_model_solves_with_laplace_precision_and_flux_variance() {
    let (topology, coords, model, truth) = manufactured_model_3d();
    let dimension = model.reduced_dimension();
    let initial_guess = vec![0.0; dimension];
    let initial_residual = model
        .residual_and_jacobian(&initial_guess)
        .unwrap()
        .residual;
    let initial_residual_norm = l2_norm(&initial_residual);
    let flux_operator =
        build_reduced_magnetic_flux_density_operator_3d(&topology, &coords, model.layout())
            .unwrap();

    let residual_model = FeecResidualAdapter::new(&model);
    let problem = NonlinearLaplaceProblem {
        prior: diagonal_prior(dimension, 1e-12),
        residual_terms: vec![NonlinearResidualTerm::zero(
            "vector_potential_magnetostatic_3d",
            &residual_model,
            GaussianNoiseModel::ScalarVariance(1e-8),
        )],
        linear_measurements: Vec::new(),
        precision_weighted_measurements: Vec::new(),
        derived_quantities: vec![LinearPdeDerivedQuantitySpec {
            name: "cell_B_3d".to_string(),
            operator: flux_operator,
        }],
    };

    let result = solve_nonlinear_laplace(
        &problem,
        &GaussNewtonConfig {
            initial_guess: Some(initial_guess),
            max_iterations: 30,
            step_tolerance: 1e-10,
            gradient_tolerance: 1e-9,
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::Exact,
                ..LinearPdeVarianceConfig::default()
            },
            ..GaussNewtonConfig::default()
        },
    )
    .expect("3D FEEC nonlinear magnetostatic problem should solve");

    assert!(result.converged);
    let final_residual = &result.final_residuals[0].residual;
    assert!(
        l2_norm(final_residual) <= 1e-5 * initial_residual_norm,
        "3D nonlinear solve did not sufficiently reduce the FEEC residual"
    );
    let map_error = result
        .map
        .iter()
        .zip(truth.iter())
        .map(|(estimate, exact)| (estimate - exact).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        map_error <= 1e-4 * l2_norm(&truth).max(1.0),
        "3D MAP error {map_error} was too large for the manufactured magnetostatic solution"
    );

    sparse_from_core(&result.posterior_precision)
        .cholesky_sqrt_lower()
        .expect("3D final Laplace precision should factorize");
    assert_eq!(result.posterior_precision.nrows(), dimension);
    assert_eq!(result.posterior_precision.ncols(), dimension);
    assert!(result
        .posterior_variance
        .iter()
        .all(|value| value.is_finite()));

    let flux_variance = result
        .derived_variances
        .get("cell_B_3d")
        .expect("3D cell flux derived covariance should be reported");
    assert_eq!(
        flux_variance.posterior_variance.len(),
        3 * topology.nsimplices(3)
    );
    assert!(flux_variance
        .posterior_variance
        .iter()
        .all(|value| value.is_finite() && *value >= -1e-12));
}
