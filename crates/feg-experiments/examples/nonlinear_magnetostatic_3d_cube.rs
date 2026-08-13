use feg_core::{BoundarySpec, GaussianPriorSpec, NonlinearResidualModel, SparseTripletMatrix};
use feg_infer::linear_pde::{
    LinearPdeDerivedQuantitySpec, LinearPdeVarianceConfig, LinearPdeVarianceMode,
};
use feg_infer::nonlinear::{
    solve_nonlinear_laplace, GaussNewtonConfig, GaussianNoiseModel, NonlinearLaplaceProblem,
    NonlinearResidualTerm,
};
use feg_infer::physical::build_reduced_magnetic_flux_density_operator_3d;
use feg_infer::sparse::core_triplet_to_gmrf as sparse_from_core;
use formoniq::problems::nonlinear_magnetostatic::{
    build_reduced_vector_potential_magnetostatic_3d, NonlinearMagnetostaticAssemblyConfig,
    NonlinearReluctivityLaw, ReducedVectorPotentialMagnetostatic3d,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::mesh::MeshCoords,
    topology::complex::Complex,
};
use std::f64::consts::PI;

fn main() -> Result<(), String> {
    let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let material = NonlinearReluctivityLaw::new(1.0, 0.25)?;
    let source_free = build_reduced_vector_potential_magnetostatic_3d(
        &topology,
        &coords,
        NonlinearMagnetostaticAssemblyConfig::new(material, BoundarySpec::default()),
    )?;
    let truth = smooth_edge_truth_3d(&source_free, &topology, &coords, 0.2);
    let source = source_free.manufactured_source(&truth)?;
    let model = source_free.with_source(source)?;
    let dimension = model.reduced_dimension();
    let initial_guess = vec![0.0; dimension];
    let initial_residual = model.residual_and_jacobian(&initial_guess)?.residual;
    let initial_residual_norm = l2_norm(&initial_residual);
    let flux_operator =
        build_reduced_magnetic_flux_density_operator_3d(&topology, &coords, model.layout())?;

    let problem = NonlinearLaplaceProblem {
        prior: diagonal_prior(dimension, 1e-12),
        residual_terms: vec![NonlinearResidualTerm::zero(
            "vector_potential_magnetostatic_3d",
            &model,
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
    )?;

    let final_residual_norm = l2_norm(&result.final_residuals[0].residual);
    let truth_norm = l2_norm(&truth);
    let map_error = result
        .map
        .iter()
        .zip(truth.iter())
        .map(|(estimate, exact)| (estimate - exact).powi(2))
        .sum::<f64>()
        .sqrt();
    let posterior_factorizes = sparse_from_core(&result.posterior_precision)
        .cholesky_sqrt_lower()
        .is_ok();
    let flux_variance = result
        .derived_variances
        .get("cell_B_3d")
        .ok_or_else(|| "missing 3D cell B variance report".to_string())?;
    let min_flux_var = flux_variance
        .posterior_variance
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_flux_var = flux_variance
        .posterior_variance
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("3D nonlinear magnetostatic cube");
    println!("active_dofs={dimension}");
    println!("vertices={}", topology.nsimplices(0));
    println!("edges={}", topology.nsimplices(1));
    println!("cells={}", topology.nsimplices(3));
    println!("boundary_edge_dofs={}", model.boundary_edge_dofs().len());
    println!("gauge_edge_dofs={}", model.gauge_edge_dofs().len());
    println!("converged={}", result.converged);
    println!("iterations={}", result.history.len());
    println!("initial_residual_norm={initial_residual_norm:.16e}");
    println!("final_residual_norm={final_residual_norm:.16e}");
    println!(
        "residual_reduction={:.16e}",
        final_residual_norm / initial_residual_norm
    );
    println!("truth_norm={truth_norm:.16e}");
    println!("map_error={map_error:.16e}");
    println!("relative_map_error={:.16e}", map_error / truth_norm);
    println!(
        "posterior_precision_nnz={}",
        result.posterior_precision.nnz()
    );
    println!("posterior_factorizes={posterior_factorizes}");
    println!("latent_variance_len={}", result.posterior_variance.len());
    println!(
        "flux_variance_len={}",
        flux_variance.posterior_variance.len()
    );
    println!("flux_variance_min={min_flux_var:.16e}");
    println!("flux_variance_max={max_flux_var:.16e}");
    for entry in &result.history {
        println!(
            "iter={} objective={:.16e} trial={:.16e} grad={:.16e} step={:.16e} alpha={:.16e} weighted_residual={:.16e}",
            entry.iteration,
            entry.objective,
            entry.trial_objective,
            entry.gradient_norm,
            entry.step_norm,
            entry.alpha,
            entry.residual_norm
        );
    }

    Ok(())
}

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

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
