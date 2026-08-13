//! Electromagnetic UQ on a unit cube with mixed boundary conditions.
//!
//! The unknown vector potential `A` is a Whitney 1-form. The code assembles the
//! boundary-reduced Hodge--Laplacian system, solves it once as a deterministic
//! FEEC problem, and then uses the same system in two Gaussian models. Magnetic
//! field outputs follow the FEEC map
//! `A --D1--> flux cochain --Whitney reconstruction--> cellwise B`.
//!
//! The manufactured fields are
//! `A = (0, 0, c x y^2)` and `B = curl(A) = (2 c x y, -c y^2, 0)` on the unit
//! cube. Essential conditions are imposed on the coordinate-zero faces and
//! natural conditions on the coordinate-one faces.
//!
//! The two models make different prior assumptions:
//!
//! - Model A uses `A = A_det + delta`, where `delta` has a smooth Matérn prior.
//!   Its covariance is calibrated to 0.50 T B-RMS and conditioned on the flux
//!   sensor.
//! - Model B uses the weak proper prior `Q0 = lambda M1`, then observes
//!   `0 = K A + boundary_bias - rhs + epsilon_pde` with precision
//!   `sigma_pde^-2 M1^-1`, before observing the same flux sensor.
//!
//! The sensor value is the analytic continuum flux through `x=1`. Model B's
//! finite weak-residual variance represents discretization error and permits a
//! small departure from the fixed-mesh FEEC solution.
//!
//! The five scalar engineering QoIs use exact posterior covariance. Full-field
//! variances use a seeded Monte Carlo estimate to avoid one exact solve per
//! coefficient. The runtime report compares each posterior with the deterministic
//! FEEC solution, shows the sensor update and residual diagnostic, and reports
//! uncertainty for the field and engineering QoIs.
//!
//! Model A illustrates how one flux sensor constrains an informative smooth
//! prior while leaving uncertainty in other field directions. For Model B, the
//! PDE-conditioned mean should first track the deterministic FEEC solution; the
//! sensor then resolves the continuum-to-discrete flux difference within the
//! uncertainty assigned to the weak residual. The final checks cover these
//! qualitative outcomes and the Monte Carlo variance estimates.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example em_1form_uq
//! ```
//!
//! The mesh is fixed at 6x6x6 cells. ParaView output is written to
//! `out/em_1form_uq/`.

use common::linalg::nalgebra::{CsrMatrix, Vector as FeecVector};
use ddf::cochain::{cochain_projection, Cochain};
use exterior::{field::DiffFormClosure, ExteriorElement};
use feec_gmrf::prelude::*;
use formoniq::{
    assemble::{
        self, assemble_boundary_integral_term, assemble_galvec,
        assemble_whitney_projected_sparse_inverse_galmat,
    },
    io::{write_cochain_vtu_fields, write_top_cell_vtu_fields},
    operators::SourceElVec,
    problems::{
        hodge_laplace::MixedGalmats,
        reduced_linear::{
            build_reduced_hodge_laplace_1form_system_with_galmats,
            reduce_reduced_hodge_laplace_1form_rhs_with_galmats,
        },
    },
    reduction::{EssentialBoundarySpec, PrescribedDof},
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt, CoordRef},
    topology::{complex::Complex, handle::KSimplexIdx},
};
use std::{collections::HashSet, error::Error, fs, path::Path, time::Instant};

const QOI_NAMES: [&str; 5] = ["flux_x1", "flux_y1", "mean_bx", "mean_by", "mean_bz"];
// Keep the mesh and uncertainty scales fixed so the numerical checks remain
// reproducible.
const CELLS_PER_AXIS: usize = 6;
const SENSOR_STANDARD_DEVIATION_WB: f64 = 0.005;
const PDE_STANDARD_DEVIATION: f64 = 0.03;
const MATERN_B_RMS_TARGET_T: f64 = 0.50;
const L2_WHITE_DETERMINISTIC_DISTANCE: f64 = 1.0;
const OUTPUT_ROOT: &str = "out/em_1form_uq";

type ExampleResult<T> = std::result::Result<T, Box<dyn Error>>;

/// Mesh, assembled system, deterministic solution, and projected exact field.
struct ElectromagneticProblem {
    c: f64,
    topology: Complex,
    coords: MeshCoords,
    system: LinearPdeSystem,
    deterministic: DeterministicLinearPdeSolution,
    truth_a: Cochain,
}

/// FEEC pushforwards used by the sensor and the reported quantities.
struct PhysicalOutputs {
    magnetic: MagneticFieldMaps3d,
    flux_x1: LinearMap,
    qois: LinearMap,
}

/// Continuum truth and fixed-mesh deterministic reference values.
struct ReferenceData {
    truth_qois: [f64; 5],
    deterministic_qois: Vec<f64>,
    deterministic_b: Vec<f64>,
}

/// Data shared by the console and VTU output routines.
struct ReportContext<'a> {
    problem: &'a ElectromagneticProblem,
    outputs: &'a PhysicalOutputs,
    reference: &'a ReferenceData,
    output_root: &'a Path,
}

fn main() -> ExampleResult<()> {
    let started = Instant::now();
    let problem = assemble_manufactured_problem()?;
    let outputs = build_physical_outputs(&problem.topology, &problem.coords)?;
    let reference = ReferenceData {
        truth_qois: [
            problem.c,
            -problem.c,
            problem.c / 2.0,
            -problem.c / 3.0,
            0.0,
        ],
        deterministic_qois: outputs.qois.apply(problem.deterministic.cochain())?,
        deterministic_b: outputs
            .magnetic
            .magnetic_field
            .apply(problem.deterministic.cochain())?,
    };

    // Use the continuum flux as the datum. Before assimilating it in Model B,
    // we check it against the PDE-only predictive distribution.
    let sensor = LinearObservation::new(
        outputs.flux_x1.clone(),
        vec![reference.truth_qois[0]],
        GaussianNoise::standard_deviation(SENSOR_STANDARD_DEVIATION_WB)?,
    )?;

    // Model A puts smooth Matérn increments around the deterministic FEEC
    // solution. The B-RMS calibration preserves that mean.
    let matern_uncalibrated = problem
        .system
        .matern_prior_builder()?
        .parameters(MaternParameters::from_practical_range(
            MaternAlpha::Two,
            0.5,
            3,
            1.0,
        )?)
        .mean(problem.deterministic.active().to_vec())
        .build()?;
    let (matern_prior, matern_calibration) = calibrate_prior_to_weighted_physical_rms(
        &matern_uncalibrated,
        outputs.magnetic.magnetic_field.map(),
        &outputs.magnetic.vector_rms_weights,
        MATERN_B_RMS_TARGET_T,
    )?;
    let matern_flux_std = prior_standard_deviation(&matern_prior, &outputs.flux_x1)?;

    // Model B uses Q = lambda M1, which is white in the FEEC L2 metric. The PDE
    // likelihood below supplies the spatial structure and nonzero mean.
    let l2_white_uncalibrated = problem.system.l2_white_noise_prior(1.0)?;
    let (l2_white_prior, l2_white_calibration) = l2_white_uncalibrated
        .calibrate_to_mahalanobis_distance(
            problem.deterministic.active(),
            L2_WHITE_DETERMINISTIC_DISTANCE,
        )?;
    let l2_white_flux_std = prior_standard_deviation(&l2_white_prior, &outputs.flux_x1)?;
    let l2_white_b_rms = outputs.magnetic.vector_rms_standard_deviation(
        &l2_white_prior.pushforward_variances(outputs.magnetic.magnetic_field.map())?,
    )?;

    print_setup(
        &problem,
        &reference,
        &matern_calibration,
        matern_flux_std,
        &l2_white_calibration,
        l2_white_flux_std,
        l2_white_b_rms,
    );

    let output_root = Path::new(OUTPUT_ROOT);
    fs::create_dir_all(output_root)?;
    let context = ReportContext {
        problem: &problem,
        outputs: &outputs,
        reference: &reference,
        output_root,
    };

    let model_started = Instant::now();
    let matern_posterior = register_outputs(
        LinearPdeModelBuilder::new(matern_prior, &problem.system)?.observe(sensor.clone())?,
        &problem.system,
        &outputs,
    )?
    .condition()?;
    report_and_write(
        "matern_sensor_only",
        matern_posterior,
        matern_flux_std,
        &context,
    )?;
    println!("  Matérn model elapsed: {:.3?}", model_started.elapsed());

    let model_started = Instant::now();
    let pde_noise = PdeResidualNoise::mass_weighted_l2_standard_deviation(PDE_STANDARD_DEVIATION)?;

    // Inspect the PDE-only distribution before adding the sensor.
    let physics_only = register_outputs(
        LinearPdeModelBuilder::new(l2_white_prior.clone(), &problem.system)?
            .observe_weak_residual(pde_noise)?,
        &problem.system,
        &outputs,
    )?
    .condition()?;
    let physics_only_flux_std = report_physics_only(physics_only, &context)?;

    let l2_white_posterior = register_outputs(
        LinearPdeModelBuilder::new(l2_white_prior, &problem.system)?
            .observe_weak_residual(pde_noise)?
            .observe(sensor)?,
        &problem.system,
        &outputs,
    )?
    .condition()?;
    report_and_write(
        "l2_white_pde_sensor",
        l2_white_posterior,
        physics_only_flux_std,
        &context,
    )?;
    println!(
        "  L2-white/PDE model elapsed: {:.3?}",
        model_started.elapsed()
    );
    println!("  total elapsed: {:.3?}", started.elapsed());
    println!("  wrote VTU output under {}", output_root.display());
    Ok(())
}

/// FEEC assembly for the manufactured mixed-boundary problem.
///
/// Assemble the FEEC system consumed by the statistical setup in `main`.
fn assemble_manufactured_problem() -> ExampleResult<ElectromagneticProblem> {
    let c = 0.50 * (45.0_f64 / 29.0).sqrt();
    let mesh = CartesianMeshInfo::new_unit_scaled(3, CELLS_PER_AXIS, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let a_exact = DiffFormClosure::one_form(
        move |p| FeecVector::from_column_slice(&[0.0, 0.0, c * p[0] * p[1] * p[1]]),
        3,
    );
    let source = DiffFormClosure::one_form(
        move |p| FeecVector::from_column_slice(&[0.0, 0.0, -2.0 * c * p[0]]),
        3,
    );
    let solution_neumann = DiffFormClosure::new(
        Box::new(move |p| {
            ExteriorElement::new(
                FeecVector::from_column_slice(&[2.0 * c * p[0] * p[1], -c * p[1] * p[1], 0.0]),
                3,
                1,
            )
        }),
        3,
        1,
    );
    let sigma_neumann = DiffFormClosure::new(
        Box::new(move |p| {
            ExteriorElement::new(
                FeecVector::from_column_slice(&[-c * p[0] * p[1] * p[1], 0.0, 0.0]),
                3,
                2,
            )
        }),
        3,
        2,
    );

    // Essential conditions apply on x=0, y=0, and z=0. The complementary
    // coordinate-one faces carry the assembled natural data below.
    let strong = |p: CoordRef| p[0].abs() < 1.0e-12 || p[1].abs() < 1.0e-12 || p[2].abs() < 1.0e-12;
    let weak = |p: CoordRef| !strong(p);
    let state_fixed = assemble::boundary_simplices_where_barycenter(&topology, &coords, 1, strong)
        .into_iter()
        .map(|index| PrescribedDof { index, value: 0.0 })
        .collect::<Vec<_>>();
    let auxiliary_fixed =
        assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, strong)
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect::<Vec<_>>();
    let boundary = EssentialBoundarySpec::default()
        .with_state(state_fixed)
        .with_auxiliary(auxiliary_fixed);
    let weak_faces = assemble::boundary_simplices_where_barycenter(&topology, &coords, 2, weak)
        .into_iter()
        .collect::<HashSet<_>>();
    let weak_face = |index: KSimplexIdx| weak_faces.contains(&index);

    let galmats = MixedGalmats::compute(&topology, &metric, 1);
    let projected_inverse = CsrMatrix::from(&assemble_whitney_projected_sparse_inverse_galmat(
        &topology, &metric,
    ));
    let assembly = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &boundary,
        &projected_inverse,
    )?;
    let source_rhs = assemble_galvec(&topology, &metric, SourceElVec::new(&source, &coords, None));
    let solution_neumann_rhs =
        assemble_boundary_integral_term(&topology, &coords, 1, &solution_neumann, None, &weak_face);
    let sigma_neumann_rhs =
        assemble_boundary_integral_term(&topology, &coords, 0, &sigma_neumann, None, &weak_face);
    // Reduce the volume source and both natural boundary terms into the active
    // 1-form ordering used by `LinearPdeSystem`.
    let reduced_rhs = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        &galmats,
        &boundary,
        &sigma_neumann_rhs,
        &(source_rhs + solution_neumann_rhs),
    )?;
    let system = LinearPdeSystem::from_reduced_assembly(assembly)?
        .with_form_space(1, 3)?
        .with_right_hand_side(reduced_rhs.as_slice())?;
    let deterministic = system.solve_deterministic()?;
    let truth_a = cochain_projection(&a_exact, &topology, &coords, None);
    Ok(ElectromagneticProblem {
        c,
        topology,
        coords,
        system,
        deterministic,
        truth_a,
    })
}

/// Build orientation-aware physical maps from the full FEEC cochain ordering.
fn build_physical_outputs(topology: &Complex, coords: &MeshCoords) -> Result<PhysicalOutputs> {
    let magnetic = MagneticFieldMaps3d::from_feec(topology, coords)?;
    let x1_faces = boundary_faces_on(topology, coords, |p| (p[0] - 1.0).abs() < 1.0e-12);
    let y1_faces = boundary_faces_on(topology, coords, |p| (p[1] - 1.0).abs() < 1.0e-12);
    let flux_x1 = outward_boundary_flux_map_3d(topology, coords, &x1_faces)?;
    let flux_y1 = outward_boundary_flux_map_3d(topology, coords, &y1_faces)?;
    let qois = LinearMap::stack(&[
        flux_x1.clone(),
        flux_y1,
        magnetic.volume_average_components[0].clone(),
        magnetic.volume_average_components[1].clone(),
        magnetic.volume_average_components[2].clone(),
    ])?;
    Ok(PhysicalOutputs {
        magnetic,
        flux_x1,
        qois,
    })
}

fn print_setup(
    problem: &ElectromagneticProblem,
    reference: &ReferenceData,
    matern_calibration: &PhysicalRmsCalibration,
    matern_flux_std: f64,
    l2_white_calibration: &PriorMahalanobisCalibration,
    l2_white_flux_std: f64,
    l2_white_b_rms: f64,
) {
    println!("3D electromagnetic 1-form UQ workflow");
    println!(
        "  mesh: {CELLS_PER_AXIS}^3 cubes, {} tetrahedra, {} full / {} active edge coefficients",
        problem.topology.nsimplices(3),
        problem.system.cochain_dimension(),
        problem.system.state_dimension()
    );
    println!(
        "  analytic flux sensor std: {:.6} Wb; Model B weak-residual discrepancy std: {:.6}",
        SENSOR_STANDARD_DEVIATION_WB, PDE_STANDARD_DEVIATION,
    );
    println!(
        "  Matérn calibration: {:.6} T -> {:.6} T, precision scale {:.6e}",
        matern_calibration.uncalibrated_rms,
        matern_calibration.target_rms,
        matern_calibration.precision_scale,
    );
    println!(
        "    prior x=1 flux std: {:.6} Wb ({:.3} sensor standard deviations)",
        matern_flux_std,
        matern_flux_std / SENSOR_STANDARD_DEVIATION_WB,
    );
    println!(
        "  L2-white regularizer scaling: deterministic distance {:.6} -> {:.6}, precision scale {:.6e}",
        l2_white_calibration.uncalibrated_distance,
        l2_white_calibration.target_distance,
        l2_white_calibration.precision_scale,
    );
    println!(
        "    raw prior x=1 flux std: {:.6} Wb ({:.3} sensor standard deviations)",
        l2_white_flux_std,
        l2_white_flux_std / SENSOR_STANDARD_DEVIATION_WB,
    );
    println!(
        "    raw prior volume-weighted B RMS std: {:.6} T",
        l2_white_b_rms,
    );

    println!("  deterministic FEEC QoIs:");
    for (index, name) in QOI_NAMES.iter().enumerate() {
        println!(
            "    {:>8}: truth={:+.6} deterministic={:+.6} error={:+.6}",
            name,
            reference.truth_qois[index],
            reference.deterministic_qois[index],
            reference.deterministic_qois[index] - reference.truth_qois[index],
        );
    }
    println!(
        "    x=1 deterministic flux error: {:+.3} sensor standard deviations",
        (reference.deterministic_qois[0] - reference.truth_qois[0]) / SENSOR_STANDARD_DEVIATION_WB,
    );
}

/// Register outputs once so every model returns the ordinary root `Posterior`
/// with the same names and coordinate conventions.
fn register_outputs<'a>(
    builder: LinearPdeModelBuilder<'a>,
    system: &'a LinearPdeSystem,
    outputs: &PhysicalOutputs,
) -> Result<LinearPdeModelBuilder<'a>> {
    let builder = builder
        .derive(DerivedQuantity::new(
            "d1a_flux_cochain",
            outputs.magnetic.flux_cochain.clone(),
        )?)?
        .derive_physical(PhysicalMap::new(
            "magnetic_field",
            outputs.magnetic.magnetic_field.map().clone(),
        )?)?
        .derive(DerivedQuantity::new(
            "engineering_qois",
            outputs.qois.clone(),
        )?)?
        .derive(system.residual_quantity("weak_pde_residual")?)?;
    Ok(builder)
}

fn prior_standard_deviation(prior: &GaussianPrior, map: &LinearMap) -> Result<f64> {
    Ok(prior.pushforward_variances(map)?[0].max(0.0).sqrt())
}

fn report_and_write(
    model_name: &str,
    mut posterior: Posterior,
    pre_sensor_flux_std: f64,
    context: &ReportContext<'_>,
) -> ExampleResult<()> {
    let system = &context.problem.system;
    let magnetic = &context.outputs.magnetic;
    let topology = &context.problem.topology;
    let coords = &context.problem.coords;
    let truth_a = &context.problem.truth_a;
    let c = context.reference.truth_qois[0];
    let truth_qois = &context.reference.truth_qois;
    let deterministic = &context.problem.deterministic;
    let deterministic_qois = &context.reference.deterministic_qois;
    let deterministic_b = &context.reference.deterministic_b;
    // Full-field marginal variances use reproducible batched Monte Carlo. The
    // scalar QoI covariance queried below remains exact.
    let mc = VarianceMethod::MonteCarlo(MonteCarloVarianceConfig::new(1024, 8, 42)?);
    let a_variance = posterior.cochain_variance_estimate(mc)?;
    let d1_variance = posterior.derived_variance_estimate("d1a_flux_cochain", mc)?;
    let b_variance = posterior.derived_variance_estimate("magnetic_field", mc)?;
    let d1_mean = posterior.derived_mean("d1a_flux_cochain")?;
    let b_mean = posterior.derived_mean("magnetic_field")?;
    let residual = posterior.derived_mean("weak_pde_residual")?;
    let qoi_mean = posterior.derived_mean("engineering_qois")?;
    let qoi_covariance = posterior.derived_covariance("engineering_qois")?;
    let posterior_b_rms = magnetic.vector_rms_standard_deviation(&b_variance.values)?;
    let state_deterministic_gap =
        system.relative_state_l2_error(posterior.mean(), deterministic.active())?;
    let b_deterministic_gap = magnetic.relative_vector_rms_error(&b_mean, deterministic_b)?;

    let residual_dual_norm = residual_dual_norm(system, &residual)?;

    println!("\n  {model_name}");
    println!(
        "    mass-weighted weak residual diagnostic dual norm: {:.6e}",
        residual_dual_norm,
    );
    println!("    posterior mean vs deterministic FEEC solution:");
    println!(
        "      relative state M1-L2 gap: {:.6e}",
        state_deterministic_gap,
    );
    println!(
        "      relative reconstructed-B RMS gap: {:.6e}",
        b_deterministic_gap,
    );
    println!(
        "    x=1 sensor mismatch: {:+.3} declared noise standard deviations",
        (qoi_mean[0] - c) / SENSOR_STANDARD_DEVIATION_WB,
    );
    println!(
        "    x=1 uncertainty: pre-sensor std={:.6} Wb, posterior std={:.6} Wb, variance reduction={:.1}%",
        pre_sensor_flux_std,
        qoi_covariance[0][0].max(0.0).sqrt(),
        100.0 * (1.0 - qoi_covariance[0][0] / pre_sensor_flux_std.powi(2)),
    );
    println!(
        "    posterior volume-weighted B RMS std (MC): {:.6} T",
        posterior_b_rms,
    );
    println!("    engineering QoIs:");
    for (index, name) in QOI_NAMES.iter().enumerate() {
        let standard_deviation = qoi_covariance[index][index].max(0.0).sqrt();
        println!(
            "      {:>8}: truth={:+.6} deterministic={:+.6} mean={:+.6} std={:.6} truth_z={:+.3} deterministic_z={:+.3}",
            name,
            truth_qois[index],
            deterministic_qois[index],
            qoi_mean[index],
            standard_deviation,
            (qoi_mean[index] - truth_qois[index]) / standard_deviation,
            (qoi_mean[index] - deterministic_qois[index]) / standard_deviation,
        );
    }
    println!("    posterior QoI correlation (order {:?}):", QOI_NAMES);
    for row in 0..QOI_NAMES.len() {
        for column in 0..QOI_NAMES.len() {
            let denominator = (qoi_covariance[row][row] * qoi_covariance[column][column]).sqrt();
            let correlation = if denominator > 0.0 {
                qoi_covariance[row][column] / denominator
            } else {
                0.0
            };
            print!(" {correlation:+.4}");
        }
        println!();
    }
    println!(
        "    MC max relative SE: A={:.3}, D1A={:.3}, B={:.3}",
        max_relative_se(&a_variance),
        max_relative_se(&d1_variance),
        max_relative_se(&b_variance),
    );
    let sensor_residual = ((qoi_mean[0] - c) / SENSOR_STANDARD_DEVIATION_WB).abs();
    let variance_reduction = 1.0 - qoi_covariance[0][0] / pre_sensor_flux_std.powi(2);
    let max_truth_z = (0..QOI_NAMES.len())
        .map(|index| {
            ((qoi_mean[index] - truth_qois[index]) / qoi_covariance[index][index].max(0.0).sqrt())
                .abs()
        })
        .fold(0.0, f64::max);
    validate_conditioned_exemplar(
        sensor_residual,
        variance_reduction,
        max_truth_z,
        state_deterministic_gap,
        b_deterministic_gap,
    )?;
    println!(
        "    exemplar checks: PASS (sensor fit, variance reduction, truth coverage, deterministic-reference gaps)"
    );

    let output_dir = context.output_root.join(model_name);
    fs::create_dir_all(&output_dir)?;
    let a_mean = Cochain::new(1, FeecVector::from_column_slice(posterior.cochain_mean()));
    let a_var = Cochain::new(1, FeecVector::from_column_slice(&a_variance.values));
    let a_std = Cochain::new(
        1,
        FeecVector::from_iterator(
            a_variance.values.len(),
            a_variance.values.iter().map(|value| value.max(0.0).sqrt()),
        ),
    );
    write_cochain_vtu_fields(
        output_dir.join("a_1form.vtu"),
        coords,
        topology,
        1,
        &[
            ("truth_projection", truth_a),
            ("posterior_mean", &a_mean),
            ("posterior_variance_mc", &a_var),
            ("posterior_std_mc", &a_std),
        ],
    )?;

    let full_d1 = feg_infer::physical::build_exterior_derivative_1_operator(topology)?;
    let truth_d1_values = full_d1.apply(&gmrf_core::Vector::from_iterator(
        truth_a.coeffs.len(),
        truth_a.coeffs.iter().copied(),
    ))?;
    let truth_d1 = Cochain::new(
        2,
        FeecVector::from_iterator(truth_d1_values.len(), truth_d1_values.iter().copied()),
    );
    let d1_mean = Cochain::new(2, FeecVector::from_column_slice(&d1_mean));
    let d1_var = Cochain::new(2, FeecVector::from_column_slice(&d1_variance.values));
    let d1_std = Cochain::new(
        2,
        FeecVector::from_iterator(
            d1_variance.values.len(),
            d1_variance.values.iter().map(|value| value.max(0.0).sqrt()),
        ),
    );
    write_cochain_vtu_fields(
        output_dir.join("d1a_2form.vtu"),
        coords,
        topology,
        2,
        &[
            ("truth", &truth_d1),
            ("posterior_mean", &d1_mean),
            ("posterior_variance_mc", &d1_var),
            ("posterior_std_mc", &d1_std),
        ],
    )?;

    let truth_b = topology
        .cells()
        .handle_iter()
        .map(|cell| {
            let point = cell.coord_simplex(coords).barycenter();
            [2.0 * c * point[0] * point[1], -c * point[1] * point[1], 0.0]
        })
        .collect::<Vec<_>>();
    let posterior_b = interleaved_vectors(&b_mean);
    let component_variances = split_interleaved_components(&b_variance.values);
    let component_stds = component_variances.clone().map(|values| {
        values
            .iter()
            .map(|value| value.max(0.0).sqrt())
            .collect::<Vec<_>>()
    });
    write_top_cell_vtu_fields(
        output_dir.join("b_cellwise.vtu"),
        coords,
        topology,
        &[("truth", &truth_b), ("posterior_mean", &posterior_b)],
        &[
            ("variance_bx_mc", &component_variances[0]),
            ("variance_by_mc", &component_variances[1]),
            ("variance_bz_mc", &component_variances[2]),
            ("std_bx_mc", &component_stds[0]),
            ("std_by_mc", &component_stds[1]),
            ("std_bz_mc", &component_stds[2]),
        ],
    )?;
    Ok(())
}

fn report_physics_only(
    mut posterior: Posterior,
    context: &ReportContext<'_>,
) -> ExampleResult<f64> {
    let system = &context.problem.system;
    let magnetic = &context.outputs.magnetic;
    let deterministic = &context.problem.deterministic;
    let deterministic_qois = &context.reference.deterministic_qois;
    let deterministic_b = &context.reference.deterministic_b;
    let truth_qois = &context.reference.truth_qois;
    let qoi_mean = posterior.derived_mean("engineering_qois")?;
    let covariance = posterior.derived_covariance("engineering_qois")?;
    let residual = posterior.derived_mean("weak_pde_residual")?;
    let b_mean = posterior.derived_mean("magnetic_field")?;
    let flux_std = covariance[0][0].max(0.0).sqrt();
    let predictive_std = (covariance[0][0] + SENSOR_STANDARD_DEVIATION_WB.powi(2)).sqrt();
    let state_deterministic_gap =
        system.relative_state_l2_error(posterior.mean(), deterministic.active())?;
    let b_deterministic_gap = magnetic.relative_vector_rms_error(&b_mean, deterministic_b)?;
    println!("\n  l2_white_pde physics-only checkpoint");
    println!(
        "    mass-weighted weak residual diagnostic dual norm: {:.6e}",
        residual_dual_norm(system, &residual)?,
    );
    println!(
        "    mean vs deterministic FEEC: relative state M1-L2 gap={:.6e}, relative B-RMS gap={:.6e}",
        state_deterministic_gap,
        b_deterministic_gap,
    );
    println!(
        "    x=1 prediction: deterministic={:+.6}, analytic sensor={:+.6}, mean={:+.6}, latent std={:.6}, analytic error={:+.3} sensor SD, predictive z={:+.3}",
        deterministic_qois[0],
        truth_qois[0],
        qoi_mean[0],
        flux_std,
        (qoi_mean[0] - truth_qois[0]) / SENSOR_STANDARD_DEVIATION_WB,
        (qoi_mean[0] - truth_qois[0]) / predictive_std,
    );
    let predictive_z = ((qoi_mean[0] - truth_qois[0]) / predictive_std).abs();
    validate_physics_only_exemplar(predictive_z, state_deterministic_gap, b_deterministic_gap)?;
    println!("    exemplar checks: PASS (sensor consistency and deterministic-solution recovery)");
    Ok(flux_std)
}

fn validate_conditioned_exemplar(
    sensor_residual: f64,
    variance_reduction: f64,
    max_truth_z: f64,
    state_deterministic_gap: f64,
    b_deterministic_gap: f64,
) -> std::result::Result<(), Box<dyn Error>> {
    if sensor_residual > 1.0
        || variance_reduction < 0.5
        || max_truth_z > 2.0
        || state_deterministic_gap > 0.05
        || b_deterministic_gap > 0.05
    {
        return Err(format!(
            "conditioned exemplar failed: sensor residual {sensor_residual:.3}, variance reduction {variance_reduction:.3}, maximum QoI |z| {max_truth_z:.3}, state gap {state_deterministic_gap:.3}, B gap {b_deterministic_gap:.3}"
        )
        .into());
    }
    Ok(())
}

fn validate_physics_only_exemplar(
    predictive_z: f64,
    state_deterministic_gap: f64,
    b_deterministic_gap: f64,
) -> std::result::Result<(), Box<dyn Error>> {
    if predictive_z > 2.0 || state_deterministic_gap > 1.0e-3 || b_deterministic_gap > 1.0e-3 {
        return Err(format!(
            "physics-only exemplar failed: sensor innovation |z| {predictive_z:.3}, state gap {state_deterministic_gap:.3e}, B gap {b_deterministic_gap:.3e}"
        )
        .into());
    }
    Ok(())
}

fn residual_dual_norm(
    system: &LinearPdeSystem,
    residual: &[f64],
) -> std::result::Result<f64, Box<dyn Error>> {
    let mass_inverse = system
        .assembly()
        .state_mass_inverse
        .as_ref()
        .ok_or("electromagnetic system is missing its state-mass inverse")?;
    let weighted_residual =
        LinearMap::new(feg_infer::sparse::feec_csr_to_core_triplet(mass_inverse))?
            .apply(residual)?;
    Ok(residual
        .iter()
        .zip(weighted_residual)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        .max(0.0)
        .sqrt())
}

fn boundary_faces_on(
    topology: &Complex,
    coords: &MeshCoords,
    predicate: impl Fn(CoordRef) -> bool + Sync,
) -> Vec<usize> {
    assemble::boundary_simplices_where_barycenter(topology, coords, 2, predicate)
}

fn interleaved_vectors(values: &[f64]) -> Vec<[f64; 3]> {
    values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect()
}

fn split_interleaved_components(values: &[f64]) -> [Vec<f64>; 3] {
    let mut components = [Vec::new(), Vec::new(), Vec::new()];
    for chunk in values.chunks_exact(3) {
        for component in 0..3 {
            components[component].push(chunk[component]);
        }
    }
    components
}

fn max_relative_se(estimate: &VarianceEstimate) -> f64 {
    estimate
        .relative_standard_error
        .as_ref()
        .map(|values| values.iter().copied().fold(0.0, f64::max))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditioned_exemplar_gate_enforces_all_uq_outcomes() {
        assert!(validate_conditioned_exemplar(0.5, 0.8, 1.5, 0.02, 0.03).is_ok());
        assert!(validate_conditioned_exemplar(1.1, 0.8, 1.5, 0.02, 0.03).is_err());
        assert!(validate_conditioned_exemplar(0.5, 0.4, 1.5, 0.02, 0.03).is_err());
        assert!(validate_conditioned_exemplar(0.5, 0.8, 2.1, 0.02, 0.03).is_err());
        assert!(validate_conditioned_exemplar(0.5, 0.8, 1.5, 0.06, 0.03).is_err());
        assert!(validate_conditioned_exemplar(0.5, 0.8, 1.5, 0.02, 0.06).is_err());
    }

    #[test]
    fn physics_only_gate_requires_discrete_recovery_and_sensor_consistency() {
        assert!(validate_physics_only_exemplar(1.5, 5.0e-4, 4.0e-4).is_ok());
        assert!(validate_physics_only_exemplar(2.1, 5.0e-4, 4.0e-4).is_err());
        assert!(validate_physics_only_exemplar(1.5, 2.0e-3, 4.0e-4).is_err());
        assert!(validate_physics_only_exemplar(1.5, 5.0e-4, 2.0e-3).is_err());
    }
}
