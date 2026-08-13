use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::ManifoldComplexExt;
use feg_core::{GaussianPriorSpec, SparseTriplet, SparseTripletMatrix};
use feg_infer::linear_pde::{
    solve_linear_pde_uq_with_config, LinearPdeDerivedQuantitySpec, LinearPdePrecisionPolicy,
    LinearPdeUqProblem, LinearPdeUqSolverConfig, LinearPdeVarianceConfig, LinearPdeVarianceMode,
};
use feg_infer::prior::matern::one_form::{
    build_hodge_laplacian_1form, build_matern_precision_1form, feec_csr_to_gmrf, feec_vec_to_gmrf,
    MaternConfig, MaternMassInverse,
};
use feg_infer::prior::matern::zero_form::{
    build_laplace_beltrami_0form, build_matern_precision_0form,
    feec_csr_to_gmrf as feec_csr_to_gmrf_0form, feec_vec_to_gmrf as feec_vec_to_gmrf_0form,
    LaplaceBeltrami0Form, MaternConfig as Matern0Config, MaternMassInverse as Matern0MassInverse,
};
use feg_infer::sparse::sparse_row_operator_from_feec_csr;
use formoniq::problems::reduced_linear::build_reduced_laplace_beltrami_system;
use formoniq::reduction::EssentialBoundarySpec;
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::Gmrf;
use manifold::gen::cartesian::CartesianMeshInfo;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn to_core_triplets(matrix: &FeecCsr) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col,
                value: *value,
            }),
    )
}

// Smoke test for FEEC -> GMRF integration.
// This exercises: FEEC assembly, conversion to GMRF sparse types, Gaussian conditioning,
// and sampling from the resulting posterior.
#[test]
fn feec_gmrf_pipeline_samples() {
    // 1) Build a tiny mesh and FEEC operators.
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let prior_precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 2.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
        },
    );

    // 2) Create a synthetic "true" field and its PDE observation.
    let ndofs = hodge.mass_u.nrows();
    let u_true = FeecVector::from_iterator(ndofs, (0..ndofs).map(|i| (i as f64 * 0.11).cos()));
    let rhs = &hodge.laplacian * &u_true;

    // 3) Convert FEEC matrices/vectors into GMRF types and condition on observations.
    let h_gmrf = feec_csr_to_gmrf(&hodge.laplacian);
    let y_gmrf = feec_vec_to_gmrf(&rhs);
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);

    let noise_variance = 1e-3;
    let (posterior_precision, information) =
        apply_gaussian_observations(&q_prior_gmrf, &h_gmrf, &y_gmrf, None, noise_variance);

    // 4) Build the posterior and draw a sample to verify basic sanity.
    let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)
        .expect("posterior should build");

    let mut rng = StdRng::seed_from_u64(42);
    let sample = posterior.sample(&mut rng).expect("sample should succeed");

    // 5) Sanity checks: dimension and finite values.
    assert_eq!(sample.len(), ndofs);
    assert!(sample.iter().all(|v| v.is_finite()));
}

#[test]
fn feec_gmrf_pipeline_samples_for_0forms() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let laplace = build_laplace_beltrami_0form(&topology, &metric);
    let prior_precision = build_matern_precision_0form(
        &laplace,
        Matern0Config {
            kappa: 2.0,
            tau: 1.0,
            mass_inverse: Matern0MassInverse::RowSumLumped,
        },
    );

    let ndofs = laplace.mass.nrows();
    let u_true = FeecVector::from_iterator(ndofs, (0..ndofs).map(|i| (i as f64 * 0.17).sin()));
    let rhs = &laplace.laplacian * &u_true;

    let h_gmrf = feec_csr_to_gmrf_0form(&laplace.laplacian);
    let y_gmrf = feec_vec_to_gmrf_0form(&rhs);
    let q_prior_gmrf = feec_csr_to_gmrf_0form(&prior_precision);

    let noise_variance = 1e-3;
    let (posterior_precision, information) =
        apply_gaussian_observations(&q_prior_gmrf, &h_gmrf, &y_gmrf, None, noise_variance);

    let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)
        .expect("posterior should build");

    let mut rng = StdRng::seed_from_u64(7);
    let sample = posterior.sample(&mut rng).expect("sample should succeed");

    assert_eq!(sample.len(), ndofs);
    assert!(sample.iter().all(|v| v.is_finite()));
}

#[test]
fn feec_d0_variances_run_with_support_aware_local_rb_blocks() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let system = build_reduced_laplace_beltrami_system(
        &topology,
        &metric,
        &EssentialBoundarySpec::default(),
    )
    .expect("0-form linear PDE system should assemble");
    let prior_precision = build_matern_precision_0form(
        &LaplaceBeltrami0Form {
            laplacian: system.operator.clone(),
            mass: system.state_mass.clone(),
        },
        Matern0Config {
            kappa: 1.0,
            tau: 1.0,
            mass_inverse: Matern0MassInverse::RowSumLumped,
        },
    );
    let d0 = sparse_row_operator_from_feec_csr(&FeecCsr::from(
        &topology.exterior_derivative_operator(0),
    ))
    .expect("D0 should convert to sparse row operator");

    let result = solve_linear_pde_uq_with_config(
        &LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; system.state_dimension()],
                precision: to_core_triplets(&prior_precision),
            },
            system,
            uncertain_inputs: Vec::new(),
            joint_measurements: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "d0".to_string(),
                operator: d0,
            }],
            joint_derived_quantities: Vec::new(),
            pde_variance: Some(1e-4),
            pde_precision: None,
        },
        &LinearPdeUqSolverConfig {
            variance: LinearPdeVarianceConfig {
                mode: LinearPdeVarianceMode::LocalRbmc,
                num_variance_probes: 12,
                variance_batch_count: 3,
                rng_seed: 123,
                local_rb_block_size: 2,
            },
            precision_policy: LinearPdePrecisionPolicy::default(),
            log_diagnostics: false,
        },
    )
    .expect("support-aware transformed local RB solve should succeed");

    let d0_variance = result
        .derived_variances
        .get("d0")
        .expect("D0 derived variance should be present");
    assert_eq!(d0_variance.posterior_variance.len(), topology.nsimplices(1));
    assert!(d0_variance
        .posterior_variance
        .iter()
        .all(|value| value.is_finite()));
}
