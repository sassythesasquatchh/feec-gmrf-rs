//! Development measurement of one combined one-form PDE/data posterior.
//!
//! The common likelihood uses FEEC's projected sparse dual residual metric.
//! Its reduced Hodge PDE operator eliminates the auxiliary zero-form with
//! the existing assembler's row-lumped inverse mass, for every method.
//! `supported` is exact with respect to its full-metric prior and this common
//! likelihood; it is not a claim that the residual metric is the exact inverse.
//! The timed sparse workloads construct no full field covariance or dense mass
//! inverse. The explicitly bounded `bounds` diagnostic exports state covariance
//! only at refinements 4 and 6, after selected-UQ timing has stopped.
//! The guarded GP baseline constructs dense observation and noise covariances.

use crate::c51_magnetic_sensor_design::{
    assemble_c51_flux_cochain_split, assemble_c51_surface_flux_split,
};
use crate::c51_thesis_normalized_compact::{
    homogeneous_full_relative_boundary, residual_dual_norm, trace_calibration,
};
use common::linalg::nalgebra::Vector;
use ddf::cochain::cochain_projection;
use exterior::field::DiffFormClosure;
use faer::{linalg::solvers::Solve, Mat, Side};
use feec_gmrf::infer::{FactoredGaussianPrior, PreparedLinearGaussianModel};
use feec_gmrf::prelude::*;
use feec_gmrf::prior::supported::{
    PreparedSupportedMaternModel, SupportedMaternPosterior, SupportedMaternPrior,
};
use feg_gp::{
    partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative,
    PartialMixedGeneralizedEigensolverOptions, SpectralMaternConfig, SpectralMaternGp,
};
use formoniq::problems::reduced_linear::build_reduced_hodge_laplace_1form_system;
use formoniq::{assemble::assemble_galvec, operators::SourceElVec};
use manifold::gen::cartesian::CartesianMeshInfo;
use manifold::geometry::coord::{mesh::MeshCoords, quadrature::SimplexQuadRule};
use manifold::geometry::metric::mesh::MeshLengths;
use manifold::topology::complex::Complex;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

const SEED: u64 = 0xC51_1F0A;
const TRACE_PROBES: usize = 512;
const MC_SAMPLES: usize = 2048;
const REPLACEMENT_DATASETS: usize = 20;
const SENSOR_SIGMA: f64 = 5.0e-4;
const TARGET_B_RMS: f64 = 0.1;
const LOCAL_WIDTH: f64 = 0.15;
const LOCAL_CENTERS: [[f64; 3]; 2] = [[0.30, 0.40, 0.55], [0.65, 0.55, 0.35]];

fn simpson_unit(f: impl Fn(f64) -> f64) -> f64 {
    // Independent one-dimensional quadrature for separable analytic truths.
    const STEPS: usize = 1024;
    let mut sum = f(0.0) + f(1.0);
    for i in 1..STEPS {
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(i as f64 / STEPS as f64);
    }
    sum / (3 * STEPS) as f64
}

fn gaussian_moments(center: f64) -> [f64; 3] {
    let weight = |x: f64| (-0.5 * ((x - center) / LOCAL_WIDTH).powi(2)).exp();
    let z = simpson_unit(weight);
    [
        z,
        simpson_unit(|x| weight(x) * (std::f64::consts::PI * x).sin()) / z,
        simpson_unit(|x| weight(x) * (std::f64::consts::PI * x).cos()) / z,
    ]
}

fn local_output_truth() -> Vec<f64> {
    let mut one_form = Vec::new();
    let mut curl = Vec::new();
    for center in LOCAL_CENTERS {
        let moments = center.map(gaussian_moments);
        for axis in 0..3 {
            let j = (axis + 1) % 3;
            let k = (axis + 2) % 3;
            one_form.push(continuum_amplitude() * moments[j][1] * moments[k][1]);
            curl.push(
                continuum_amplitude()
                    * std::f64::consts::PI
                    * moments[axis][1]
                    * (moments[j][2] - moments[k][2]),
            );
        }
    }
    one_form.extend(curl);
    one_form
}

fn local_output_maps(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    operators: &ExactFormOperators,
) -> Result<LinearMap> {
    let outgoing = operators
        .outgoing()
        .expect("3D one-form has outgoing derivative");
    let mut maps = Vec::new();
    // First six rows average A; last six average its curl. For curl, assemble
    // a test two-form then compose with D1, preserving D1 D0 = 0 exactly.
    for grade in [1, 2] {
        let layout = if grade == 1 {
            operators.state().boundary_layout()
        } else {
            outgoing.upper().boundary_layout()
        };
        for center in LOCAL_CENTERS {
            let normalization: f64 = center.map(|x| gaussian_moments(x)[0]).iter().product();
            for axis in 0..3 {
                let source = DiffFormClosure::new(
                    Box::new(move |p| {
                        let distance: f64 = (0..3)
                            .map(|i| ((p[i] - center[i]) / LOCAL_WIDTH).powi(2))
                            .sum();
                        let weight = (-0.5 * distance).exp() / normalization;
                        let mut coefficients = Vector::zeros(3);
                        if grade == 1 {
                            coefficients[axis] = weight;
                        } else {
                            // dx^dy, dx^dz, dy^dz correspond to (Bz,-By,Bx).
                            coefficients[2 - axis] = if axis == 1 { -weight } else { weight };
                        }
                        exterior::ExteriorElement::new(coefficients, 3, grade)
                    }),
                    3,
                    grade,
                );
                let load = assemble_galvec(
                    topology,
                    metric,
                    SourceElVec::new(&source, coords, Some(SimplexQuadRule::order3(3))),
                );
                let weights: Vec<_> = layout
                    .active_dofs()
                    .iter()
                    .enumerate()
                    .map(|(active, &full)| (active, load[full]))
                    .collect();
                let map = LinearMap::weighted_row(layout.active_dofs().len(), &weights)?;
                maps.push(if grade == 1 {
                    map
                } else {
                    map.compose(outgoing.active_derivative())?
                });
            }
        }
    }
    LinearMap::stack(&maps)
}

fn continuum_amplitude() -> f64 {
    // Integral |curl A|^2 on the unit cube is 3*pi^2/2 before scaling.
    TARGET_B_RMS / (std::f64::consts::PI * 1.5_f64.sqrt())
}

fn continuum_surface_truth() -> Vec<f64> {
    let a = continuum_amplitude();
    // Positive normals, with cyclic tangential coordinates (y,z), (z,x),
    // (x,y), as documented by AxisAlignedFluxPatch3d; then even/odd split.
    let patches = [0.0, a, -a, 0.0, 0.0, a, -a, 0.0, 0.0, a, -a, 0.0];
    [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
        .iter()
        .map(|&i| patches[i])
        .collect()
}

fn relative_vector_error(candidate: &[f64], reference: &[f64]) -> f64 {
    (candidate
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / reference.iter().map(|v| v * v).sum::<f64>())
    .sqrt()
}

enum Prior {
    Compact(GaussianPrior),
    Supported(SupportedMaternPrior),
}

enum Factor {
    Compact(FactoredGaussianPrior),
    Supported(SupportedMaternPosterior),
}

enum Prepared {
    Compact(PreparedLinearGaussianModel),
    Supported(PreparedSupportedMaternModel),
}

enum Conditional {
    Compact(Posterior),
    Supported(SupportedMaternPosterior),
}

impl Prior {
    fn factor(&self) -> Result<Factor> {
        match self {
            Self::Compact(p) => Ok(Factor::Compact(p.factor()?)),
            Self::Supported(p) => Ok(Factor::Supported(p.factor()?)),
        }
    }
}

impl Factor {
    fn trace(&self, weight: &SparseMat) -> Result<WeightedCovarianceTraceEstimate> {
        let method =
            VarianceMethod::Hutchinson(HutchinsonVarianceConfig::new(TRACE_PROBES, 8, SEED)?);
        match self {
            Self::Compact(p) => p.weighted_covariance_trace(weight, method),
            Self::Supported(p) => p.weighted_covariance_trace(weight, method),
        }
    }

    fn variances(&mut self, map: &LinearMap, method: VarianceMethod) -> Result<VarianceEstimate> {
        match self {
            Self::Compact(p) => p.pushforward_variance_estimate(map, method),
            Self::Supported(p) => Ok(p
                .pushforward_weighted_variance_estimate(
                    map,
                    &vec![1.0; map.output_dimension()],
                    method,
                )?
                .variances),
        }
    }
}

impl Prepared {
    fn condition(&self) -> Result<Conditional> {
        match self {
            Self::Compact(p) => Ok(Conditional::Compact(p.condition()?)),
            Self::Supported(p) => Ok(Conditional::Supported(p.condition()?)),
        }
    }

    fn replacement_mean(&self, values: &[Vec<f64>]) -> Result<Vec<f64>> {
        match self {
            Self::Compact(p) => p.latent_mean_with_observation_values(values),
            Self::Supported(p) => p.latent_mean_with_observation_values(values),
        }
    }
}

impl Conditional {
    fn mean(&self) -> &[f64] {
        match self {
            Self::Compact(p) => p.mean(),
            Self::Supported(p) => p.mean(),
        }
    }

    fn covariance(&mut self, map: &LinearMap) -> Result<Vec<Vec<f64>>> {
        match self {
            Self::Compact(p) => p.pushforward_covariance(map),
            Self::Supported(p) => p.pushforward_covariance(map),
        }
    }

    fn diagnostic_covariance(&mut self, map: &LinearMap) -> Result<Vec<Vec<f64>>> {
        match self {
            Self::Compact(p) => p.pushforward_covariance(map),
            Self::Supported(p) => p.pushforward_covariance_raw(map),
        }
    }

    fn action(&mut self, rhs: &[f64]) -> Result<Vec<f64>> {
        match self {
            Self::Compact(p) => p.covariance_action(rhs),
            Self::Supported(p) => p.covariance_action(rhs),
        }
    }

    fn variances(&mut self, map: &LinearMap, method: VarianceMethod) -> Result<VarianceEstimate> {
        match self {
            Self::Compact(p) => p.pushforward_variance_estimate(map, method),
            Self::Supported(p) => Ok(p
                .pushforward_weighted_variance_estimate(
                    map,
                    &vec![1.0; map.output_dimension()],
                    method,
                )?
                .variances),
        }
    }
}

fn timed<T>(
    metrics: &mut BTreeMap<String, f64>,
    name: &str,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let start = Instant::now();
    let result = f()?;
    let seconds = start.elapsed().as_secs_f64();
    metrics.insert(format!("{name}_s"), seconds);
    eprintln!("{name}: {seconds:.6} s");
    Ok(result)
}

/// Run one method in one process. All outputs are ordinary development CSVs.
pub fn run(
    method: &str,
    refinement: usize,
    pde_sigma: f64,
    field_mode: &str,
    output: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_with_precision_scale(method, refinement, pde_sigma, field_mode, output, None)
}

/// A supplied scale freezes the physical prior parameter before timing. Spectral
/// truncations use the full-metric reference scale, unchanged with rank.
pub fn run_with_precision_scale(
    method: &str,
    refinement: usize,
    pde_sigma: f64,
    field_mode: &str,
    output: &Path,
    fixed_precision_scale: Option<f64>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_with_workload(
        method,
        refinement,
        pde_sigma,
        field_mode,
        output,
        fixed_precision_scale,
        false,
    )
}

/// `continuum` uses analytic forcing, mesh-independent amplitude, six fixed
/// surface sensors and analytic surface truths. It requires a fixed prior scale.
#[allow(clippy::too_many_arguments)]
pub fn run_with_workload(
    method: &str,
    refinement: usize,
    pde_sigma: f64,
    field_mode: &str,
    output: &Path,
    fixed_precision_scale: Option<f64>,
    continuum: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run_with_observables(
        method,
        refinement,
        pde_sigma,
        field_mode,
        output,
        fixed_precision_scale,
        continuum,
        false,
    )
}

/// Local outputs append fixed physical Gaussian averages of A and curl A.
#[allow(clippy::too_many_arguments)]
pub fn run_with_observables(
    method: &str,
    refinement: usize,
    pde_sigma: f64,
    field_mode: &str,
    output: &Path,
    fixed_precision_scale: Option<f64>,
    continuum: bool,
    local_outputs: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let spectral_rank = method
        .strip_prefix("spectral-")
        .and_then(|s| s.parse::<usize>().ok());
    if (!matches!(method, "compact" | "supported" | "gp" | "pde") && spectral_rank.is_none())
        || !matches!(field_mode, "exact" | "mc" | "none" | "bounds")
        || (local_outputs && !continuum)
        || (field_mode == "bounds"
            && (!continuum || refinement > 6 || !matches!(method, "compact" | "supported")))
        || refinement < 4
        || refinement % 2 != 0
        || !pde_sigma.is_finite()
        || pde_sigma <= 0.0
        || (method == "supported" && field_mode == "mc")
        || (method == "gp" && (field_mode != "none" || refinement > 8))
        || fixed_precision_scale.is_some_and(|s| !s.is_finite() || s <= 0.0)
        || (continuum && method != "pde" && fixed_precision_scale.is_none())
        || (method == "pde" && (!continuum || field_mode != "none"))
        || (spectral_rank.is_some()
            && (field_mode != "none"
                || refinement > 12
                || (refinement > 8 && spectral_rank.unwrap() > 256)
                || fixed_precision_scale.is_none()))
    {
        return Err(FeecGmrfError::InvalidParameter(
            "use compact/supported/gp/spectral-RANK/pde, even refinement >=4, positive scales; mc requires compact; gp requires none and refinement <=8; spectral requires none, refinement <=12 and rank <=256 above refinement 8; spectral and continuum UQ require a fixed scale; pde requires continuum and none"
                .into(),
        )
        .into());
    }
    fs::create_dir_all(output)?;
    let start = Instant::now();
    let mut metrics = BTreeMap::new();
    let (
        topology,
        coords,
        metric,
        operators,
        system,
        truth,
        flux,
        surface,
        sensors,
        sensor_truth,
        sensor_values,
    ) = timed(&mut metrics, "geometry_pde_maps", || {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, refinement, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let operators =
            ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1)?;
        let outgoing = operators.outgoing().expect("one-form in 3D has D1");
        let flux = outgoing.active_derivative().clone();
        let elimination =
            EssentialBoundaryElimination::from_layout(outgoing.upper().boundary_layout().clone())?;
        let surface_split =
            assemble_c51_surface_flux_split(&topology, &coords, &elimination, &flux)?;
        let global_surface = surface_split.stacked()?;
        let surface = if local_outputs {
            let local_start = Instant::now();
            let local = local_output_maps(&topology, &coords, &metric, &operators)?;
            eprintln!(
                "local_output_maps: {:.6} s",
                local_start.elapsed().as_secs_f64()
            );
            LinearMap::stack(&[global_surface, local])?
        } else {
            global_surface
        };
        let n = operators.state().dimension();
        let sensors = if continuum {
            surface_split.selection().clone()
        } else {
            assemble_c51_flux_cochain_split(&flux, n / 4, 64, 64)?
                .observation()
                .clone()
        };
        // Smooth tangentially homogeneous one-form, projected by FEEC edge integrals.
        // The discrete workload uses f_h = R_h a_h. The continuum workload
        // independently assembles the forcing below, enabling a spatial-error test.
        let smooth = DiffFormClosure::one_form(
            |p| {
                let [x, y, z] = [p[0], p[1], p[2]].map(|v| (std::f64::consts::PI * v).sin());
                Vector::from_vec(vec![y * z, z * x, x * y])
            },
            3,
        );
        let edge_quadrature = SimplexQuadRule::order3(1);
        let full_truth = cochain_projection(
            &smooth,
            &topology,
            &coords,
            continuum.then_some(&edge_quadrature),
        );
        let mut truth: Vec<_> = operators
            .state()
            .boundary_layout()
            .active_dofs()
            .iter()
            .map(|&i| full_truth.coeffs[i])
            .collect();
        let b = flux.apply(&truth)?;
        let mb = outgoing
            .upper()
            .mass()
            .apply_checked(&b)
            .map_err(FeecGmrfError::Dimension)?;
        let rms = b.iter().zip(mb).map(|(a, b)| a * b).sum::<f64>().sqrt();
        for value in &mut truth {
            *value *= if continuum {
                continuum_amplitude()
            } else {
                TARGET_B_RMS / rms
            };
        }
        let assembly = build_reduced_hodge_laplace_1form_system(
            &topology,
            &metric,
            &homogeneous_full_relative_boundary(&topology),
        )
        .map_err(FeecGmrfError::Assembly)?;
        if assembly.layout.active_dofs != operators.state().boundary_layout().active_dofs() {
            return Err(FeecGmrfError::Dimension(
                "PDE and prior state orderings differ".into(),
            ));
        }
        let rhs = if continuum {
            // -Delta A = 2*pi^2 A, div A = 0. Assemble (f, v_h) from
            // continuum forcing rather than applying the discrete operator.
            let source = DiffFormClosure::one_form(
                |p| {
                    let [x, y, z] = [p[0], p[1], p[2]].map(|v| (std::f64::consts::PI * v).sin());
                    Vector::from_vec(vec![y * z, z * x, x * y])
                        * (2.0 * std::f64::consts::PI.powi(2) * continuum_amplitude())
                },
                3,
            );
            let full_load = assemble_galvec(
                &topology,
                &metric,
                SourceElVec::new(&source, &coords, Some(SimplexQuadRule::order3(3))),
            );
            Vector::from_iterator(n, assembly.layout.active_dofs.iter().map(|&i| full_load[i]))
        } else {
            &assembly.operator * Vector::from_vec(truth.clone()) + &assembly.residual_bias
        };
        let system = LinearPdeSystem::from_reduced_assembly(assembly)?
            .with_form_space(1, 3)?
            .with_right_hand_side(rhs.as_slice())?;
        let mut rng = StdRng::seed_from_u64(SEED);
        let sensor_truth = if continuum {
            continuum_surface_truth()[..6].to_vec()
        } else {
            sensors.apply(&truth)?
        };
        let sensor_values: Vec<f64> = sensor_truth
            .clone()
            .into_iter()
            .map(|v| {
                v + SENSOR_SIGMA
                    * <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
            })
            .collect();
        Ok((
            topology,
            coords,
            metric,
            operators,
            system,
            truth,
            flux,
            surface,
            sensors,
            sensor_truth,
            sensor_values,
        ))
    })?;
    let n = system.state_dimension();
    metrics.insert("state_dimension".into(), n as f64);
    metrics.insert("pde_rows".into(), n as f64);
    metrics.insert("sensor_rows".into(), sensors.output_dimension() as f64);
    metrics.insert("flux_rows".into(), flux.output_dimension() as f64);
    metrics.insert(
        "local_outputs".into(),
        if local_outputs { 1.0 } else { 0.0 },
    );
    metrics.insert("output_rows".into(), surface.output_dimension() as f64);
    if local_outputs {
        metrics.insert("local_width".into(), LOCAL_WIDTH);
    }
    metrics.insert("pde_sigma".into(), pde_sigma);
    metrics.insert(
        "trace_probes".into(),
        if fixed_precision_scale.is_some() || method == "pde" {
            0.0
        } else {
            TRACE_PROBES as f64
        },
    );
    metrics.insert(
        "continuum_forcing".into(),
        if continuum { 1.0 } else { 0.0 },
    );
    let surface_truth = if continuum {
        let mut values = continuum_surface_truth();
        if local_outputs {
            values.extend(local_output_truth());
        }
        values
    } else {
        surface.apply(&truth)?
    };
    if continuum {
        metrics.insert(
            "surface_interpolation_relative_error".into(),
            relative_vector_error(&surface.apply(&truth)?[..12], &surface_truth[..12]),
        );
        if local_outputs {
            let interpolated = surface.apply(&truth)?;
            metrics.insert(
                "local_one_form_interpolation_error".into(),
                relative_vector_error(&interpolated[12..18], &surface_truth[12..18]),
            );
            metrics.insert(
                "local_curl_interpolation_error".into(),
                relative_vector_error(&interpolated[18..24], &surface_truth[18..24]),
            );
        }
    }
    if method == "pde" {
        let solution = timed(&mut metrics, "deterministic_solve", || {
            system.solve_deterministic()
        })?;
        let mean = surface.apply(solution.active())?;
        metrics.insert(
            "surface_relative_error".into(),
            relative_vector_error(&mean[..12], &surface_truth[..12]),
        );
        if local_outputs {
            metrics.insert(
                "local_one_form_relative_error".into(),
                relative_vector_error(&mean[12..18], &surface_truth[12..18]),
            );
            metrics.insert(
                "local_curl_relative_error".into(),
                relative_vector_error(&mean[18..24], &surface_truth[18..24]),
            );
        }
        metrics.insert(
            "state_interpolant_relative_l2_error".into(),
            system.relative_state_l2_error(solution.active(), &truth)?,
        );
        metrics.insert(
            "residual_dual_norm".into(),
            residual_dual_norm(&system, solution.active())?,
        );
        metrics.insert("total_compute_s".into(), start.elapsed().as_secs_f64());
        write_metrics(output, method, refinement, field_mode, &metrics)?;
        let mut csv = csv::Writer::from_path(output.join("deterministic_surface.csv"))?;
        csv.write_record(["row", "mean", "truth"])?;
        for i in 0..mean.len() {
            csv.serialize((i, mean[i], surface_truth[i]))?;
        }
        csv.flush()?;
        return Ok(());
    }
    if let Some(rank) = spectral_rank {
        if rank == 0 || rank > n {
            return Err(
                FeecGmrfError::InvalidParameter("spectral rank must be in 1..=N".into()).into(),
            );
        }
        let scale = fixed_precision_scale.unwrap();
        metrics.insert("precision_scale".into(), scale);
        let partial = timed(&mut metrics, "spectral_eigenbasis", || {
            partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
                &topology,
                &metric,
                1,
                PartialMixedGeneralizedEigensolverOptions::new(rank),
            )
            .map_err(|e| FeecGmrfError::Inference(e.to_string()))
        })?;
        metrics.insert("spectral_rank".into(), rank as f64);
        metrics.insert(
            "eigenbasis_relative_residual".into(),
            partial.diagnostics().maximum_relative_mixed_residual,
        );
        metrics.insert(
            "eigenbasis_mass_orthogonality_defect".into(),
            partial.diagnostics().maximum_mass_orthogonality_defect,
        );
        let features = timed(&mut metrics, "spectral_features", || {
            SpectralMaternGp::from_eigenbasis(
                partial.into_basis(),
                SpectralMaternConfig {
                    kappa: 4.0,
                    alpha: 2.0,
                    tau: scale.sqrt(),
                    k: rank,
                },
            )
            .map(|gp| gp.feature_matrix())
            .map_err(|e| FeecGmrfError::Inference(e.to_string()))
        })?;
        let residual = system.residual_quantity("pde")?;
        let inverse =
            LinearMap::from_feec_csr(system.assembly().state_mass_inverse.as_ref().unwrap())?;
        let values: Vec<_> = residual.bias().iter().map(|b| -b).collect();
        let prepared = timed(&mut metrics, "spectral_prepare", || {
            WeightedSpectralOutput::new(
                &features,
                residual.operator(),
                &inverse,
                &values,
                pde_sigma,
                &sensors,
                &surface,
            )
        })?;
        let mean = timed(&mut metrics, "posterior_mean", || {
            prepared.mean(&sensor_values)
        })?;
        let covariance = timed(&mut metrics, "surface_covariance", || prepared.covariance())?;
        metrics.insert("selected_uq_total_s".into(), start.elapsed().as_secs_f64());
        metrics.insert("fixed_precision_scale".into(), 1.0);
        let replacements = replacement_sensor_values(&sensor_truth);
        let replacement_means = timed(&mut metrics, "replacement_means", || {
            replacements
                .iter()
                .map(|values| prepared.mean(values))
                .collect::<Result<Vec<_>>>()
        })?;
        metrics.insert("replacement_datasets".into(), REPLACEMENT_DATASETS as f64);
        if continuum {
            cached_sensor_update_check(
                &mut metrics,
                &mean,
                &covariance,
                &sensor_values,
                &replacements,
                &replacement_means,
            )?;
        }
        metrics.insert("total_compute_s".into(), start.elapsed().as_secs_f64());
        write_metrics(output, method, refinement, field_mode, &metrics)?;
        write_surface(output, &mean, &surface_truth, &covariance)?;
        write_replacement_means(output, &replacement_means)?;
        return Ok(());
    }
    let prior = if let Some(scale) = fixed_precision_scale {
        metrics.insert("precision_scale".into(), scale);
        metrics.insert("fixed_precision_scale".into(), 1.0);
        timed(&mut metrics, "prior_construction", || {
            match method {
            "compact" => Ok(Prior::Compact(
                AdjacentMaternPriorBuilder::from_feec_homogeneous_full_relative_with_projected_inverses(&topology, &metric, 1)?
                    .parameters(MaternParameters::new(MaternAlpha::Two, 4.0, 1.0)?).build()?
                    .with_precision_scale(scale)?,
            )),
            _ => Ok(Prior::Supported(SupportedMaternPriorBuilder::from_operators(operators)
                .parameters(MaternParameters::new(MaternAlpha::Two, 4.0, scale.sqrt())?).build()?)),
        }
        })?
    } else {
        let energy = pullback_quadratic_form(&flux, operators.outgoing().unwrap().upper().mass())?;
        let parameters = MaternParameters::new(MaternAlpha::Two, 4.0, 1.0)?;
        let raw = timed(&mut metrics, "prior_construction", || {
            match method {
        "compact" => Ok(Prior::Compact(AdjacentMaternPriorBuilder::from_feec_homogeneous_full_relative_with_projected_inverses(&topology, &metric, 1)?.parameters(parameters).build()?)),
        _ => Ok(Prior::Supported(SupportedMaternPriorBuilder::from_operators(operators.clone()).parameters(parameters).build()?)),
    }
        })?;
        let raw_factor = timed(&mut metrics, "prior_factor", || raw.factor())?;
        let trace = timed(&mut metrics, "prior_trace", || raw_factor.trace(&energy))?;
        let calibration = trace_calibration(
            trace.value,
            trace.relative_standard_error,
            TARGET_B_RMS,
            1.0,
        )?;
        metrics.insert("precision_scale".into(), calibration.precision_scale);
        metrics.insert(
            "trace_relative_standard_error".into(),
            trace.relative_standard_error.unwrap_or(f64::NAN),
        );
        drop(raw_factor);
        timed(
            &mut metrics,
            "normalized_prior_construction",
            || match raw {
                Prior::Compact(p) => Ok(Prior::Compact(
                    p.with_precision_scale(calibration.precision_scale)?,
                )),
                Prior::Supported(_) => Ok(Prior::Supported(
                    SupportedMaternPriorBuilder::from_operators(operators)
                        .parameters(MaternParameters::new(
                            MaternAlpha::Two,
                            4.0,
                            calibration.tau_multiplier,
                        )?)
                        .build()?,
                )),
            },
        )?
    };
    let observations = timed(&mut metrics, "likelihood", || {
        Ok(vec![
            system.weak_residual_observation(
                PdeResidualNoise::mass_weighted_l2_standard_deviation(pde_sigma)?,
            )?,
            LinearObservation::new(
                sensors.clone(),
                sensor_values.clone(),
                GaussianNoise::standard_deviation(SENSOR_SIGMA)?,
            )?,
        ])
    })?;
    if method == "gp" {
        // This guarded covariance-form reference alone constructs dense
        // observation covariance and noise covariance. The sparse methods above
        // and below never enter this branch.
        let Prior::Supported(exact) = &prior else {
            unreachable!()
        };
        let residual = system.residual_quantity("pde")?;
        let observed = LinearMap::stack(&[residual.operator().clone(), sensors.clone()])?;
        let mut values: Vec<_> = residual.bias().iter().map(|b| -b).collect();
        values.extend_from_slice(&sensor_values);
        let m = observed.output_dimension();
        let factor = timed(&mut metrics, "gp_normalized_prior_factor", || {
            exact.factor()
        })?;
        let blocks = timed(&mut metrics, "gp_covariance_blocks", || {
            ExactObservationCovarianceBlocks::from_supported_matern_factor(
                &factor, &observed, &surface, 4096,
            )
        })?;
        let (noise, skew) = timed(&mut metrics, "gp_noise_covariance", || {
            let inverse =
                LinearMap::from_feec_csr(system.assembly().state_mass_inverse.as_ref().unwrap())?;
            let noise_prior = GaussianPrior::new(
                vec![0.0; n],
                inverse.into_matrix().scaled(1.0 / pde_sigma.powi(2)),
            )?;
            let covariance = noise_prior
                .factor()?
                .pushforward_covariance(&LinearMap::identity(n))?;
            let scale = covariance
                .iter()
                .flatten()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max);
            let mut max_skew = 0.0_f64;
            let mut noise = vec![vec![0.0; m]; m];
            for i in 0..n {
                for j in 0..=i {
                    max_skew = max_skew.max((covariance[i][j] - covariance[j][i]).abs());
                    let value = 0.5 * (covariance[i][j] + covariance[j][i]);
                    noise[i][j] = value;
                    noise[j][i] = value;
                }
            }
            if max_skew > 1.0e-10 * scale {
                return Err(FeecGmrfError::Inference(
                    "GP noise covariance has material numerical skew".into(),
                ));
            }
            for (i, row) in noise.iter_mut().enumerate().skip(n) {
                row[i] = SENSOR_SIGMA.powi(2);
            }
            Ok((noise, max_skew / scale))
        })?;
        metrics.insert("gp_noise_covariance_relative_skew".into(), skew);
        let gp = timed(&mut metrics, "gp_condition_and_covariance", || {
            ExactObservationPosterior::from_prior_covariance_blocks_with_noise_covariance(
                blocks, noise,
            )
        })?;
        let mean = timed(&mut metrics, "posterior_mean", || gp.output_mean(&values))?;
        let covariance = gp.posterior_output_covariance();
        metrics.insert("selected_uq_total_s".into(), start.elapsed().as_secs_f64());
        metrics.insert("dense_observation_factor_dimension".into(), m as f64);
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        let replacements: Vec<Vec<f64>> = (0..REPLACEMENT_DATASETS)
            .map(|_| {
                let mut v = values[..n].to_vec();
                v.extend(sensor_truth.iter().copied().map(|v| {
                    v + SENSOR_SIGMA
                        * <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                }));
                v
            })
            .collect();
        let replacement_means = timed(&mut metrics, "replacement_means", || {
            replacements
                .iter()
                .map(|v| gp.output_mean(v))
                .collect::<Result<Vec<_>>>()
        })?;
        metrics.insert("replacement_datasets".into(), REPLACEMENT_DATASETS as f64);
        metrics.insert(
            "replacement_checksum".into(),
            replacement_means.iter().flatten().sum(),
        );
        if continuum {
            cached_sensor_update_check(
                &mut metrics,
                &mean,
                &covariance,
                &sensor_values,
                &replacement_sensor_values(&sensor_truth),
                &replacement_means,
            )?;
        }
        metrics.insert("total_compute_s".into(), start.elapsed().as_secs_f64());
        write_metrics(output, method, refinement, field_mode, &metrics)?;
        write_surface(output, &mean, &surface_truth, &covariance)?;
        write_replacement_means(output, &replacement_means)?;
        eprintln!(
            "gp, N={n}: selected PDE UQ {:.3} s",
            metrics["selected_uq_total_s"]
        );
        return Ok(());
    }
    let (prepared, resources) = timed(&mut metrics, "posterior_prepare", || match &prior {
        Prior::Compact(p) => {
            let mut builder = LinearGaussianModelBuilder::new(p.clone());
            for observation in &observations {
                builder = builder.observe(observation.clone())?;
            }
            let (prepared, resources) = builder.prepare_with_resource_report()?;
            Ok((Prepared::Compact(prepared), Some(resources)))
        }
        Prior::Supported(p) => Ok((Prepared::Supported(p.prepare(&observations, &[])?), None)),
    })?;
    if let Some(report) = resources {
        for (name, phase) in [
            ("posterior_update_assembly_s", report.assembly),
            ("posterior_symbolic_s", report.analysis),
            ("posterior_numeric_s", report.numeric_factorization),
        ] {
            if let Some(duration) = phase.wall_time.measured() {
                metrics.insert(name.into(), duration.as_secs_f64());
            }
        }
    }
    match &prepared {
        Prepared::Compact(p) => {
            let d = p.factorization_diagnostics();
            metrics.insert(
                "posterior_precision_nnz".into(),
                d.precision_nonzeros as f64,
            );
            metrics.insert("posterior_factor_nnz".into(), d.factor_nonzeros as f64);
            metrics.insert("posterior_fill_ratio".into(), d.fill_ratio);
        }
        Prepared::Supported(p) => {
            let d = p.diagnostics();
            metrics.insert("kkt_input_nnz".into(), d.kkt_nnz as f64);
            metrics.insert("augmented_dimension".into(), d.ambient_dimension as f64);
        }
    }
    let mut posterior = timed(&mut metrics, "posterior_mean", || prepared.condition())?;
    let surface_mean = surface.apply(posterior.mean())?;
    let surface_covariance = timed(&mut metrics, "surface_covariance", || {
        posterior.covariance(&surface)
    })?;
    metrics.insert("selected_uq_total_s".into(), start.elapsed().as_secs_f64());
    metrics.insert(
        "mean_relative_discrete_l2_error".into(),
        system.relative_state_l2_error(posterior.mean(), &truth)?,
    );
    metrics.insert(
        "residual_dual_norm".into(),
        residual_dual_norm(&system, posterior.mean())?,
    );

    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let replacement_values: Vec<Vec<Vec<f64>>> = (0..REPLACEMENT_DATASETS)
        .map(|_| {
            let pde = vec![0.0; n];
            let sensor: Vec<_> = sensor_truth
                .iter()
                .copied()
                .map(|v| {
                    v + SENSOR_SIGMA
                        * <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                })
                .collect();
            vec![pde, sensor]
        })
        .collect();
    let replacement_means = timed(&mut metrics, "replacement_means", || {
        replacement_values
            .iter()
            .map(|values| surface.apply(&prepared.replacement_mean(values)?))
            .collect::<Result<Vec<_>>>()
    })?;
    metrics.insert("replacement_datasets".into(), REPLACEMENT_DATASETS as f64);
    metrics.insert(
        "replacement_checksum".into(),
        replacement_means.iter().flatten().sum(),
    );
    if continuum {
        cached_sensor_update_check(
            &mut metrics,
            &surface_mean,
            &surface_covariance,
            &sensor_values,
            &replacement_sensor_values(&sensor_truth),
            &replacement_means,
        )?;
        write_replacement_means(output, &replacement_means)?;
        if field_mode == "bounds" {
            // Bounded validation only: full state covariance is never requested
            // in the performance workload. No dense matrices above n=6.
            let covariance = timed(&mut metrics, "diagnostic_state_covariance", || {
                posterior.diagnostic_covariance(&LinearMap::identity(n))
            })?;
            let mut csv = csv::Writer::from_path(output.join("diagnostic_state_covariance.csv"))?;
            for row in covariance {
                csv.serialize(row)?;
            }
            csv.flush()?;
            let mut dense_map = vec![vec![0.0; n]; surface.output_dimension()];
            for (i, j, value) in surface.matrix().triplet_iter() {
                dense_map[i][j] += value;
            }
            let mut csv = csv::Writer::from_path(output.join("diagnostic_output_map.csv"))?;
            for row in dense_map {
                csv.serialize(row)?;
            }
            csv.flush()?;
        }
        if matches!(field_mode, "none" | "bounds") {
            metrics.insert("total_compute_s".into(), start.elapsed().as_secs_f64());
            write_metrics(output, method, refinement, field_mode, &metrics)?;
            write_surface(output, &surface_mean, &surface_truth, &surface_covariance)?;
            return Ok(());
        }
    }
    drop(replacement_values);
    // Four full-state covariance actions are common to both representations.
    let actions = timed(&mut metrics, "covariance_actions", || {
        let mut actions = Vec::new();
        for j in 0..4 {
            let rhs: Vec<_> = (0..n)
                .map(|i| (((i + 1) * (j + 1)) as f64 * 0.173).sin())
                .collect();
            actions.push(posterior.action(&rhs)?);
        }
        Ok(actions)
    })?;
    let field_map = LinearMap::stack(&[LinearMap::identity(n), flux.clone()])?;
    let field_mean = field_map.apply(posterior.mean())?;
    let field_estimates = if field_mode != "none" {
        let estimator = if field_mode == "exact" {
            VarianceMethod::Exact
        } else {
            VarianceMethod::MonteCarlo(MonteCarloVarianceConfig::new(MC_SAMPLES, 8, SEED + 2)?)
        };
        let post = timed(&mut metrics, "field_posterior_variance", || {
            posterior.variances(&field_map, estimator)
        })?;
        let mut prior_factor = timed(&mut metrics, "normalized_prior_factor_for_ratios", || {
            prior.factor()
        })?;
        let pre = timed(&mut metrics, "field_prior_variance", || {
            prior_factor.variances(&field_map, estimator)
        })?;
        metrics.insert("variance_samples".into(), post.sample_count as f64);
        Some((pre, post))
    } else {
        None
    };
    // Supported sampling is unavailable; it is not assigned a zero cost.
    if let Conditional::Compact(p) = &mut posterior {
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        let checksum = timed(&mut metrics, "samples_100", || {
            let mut sum = 0.0;
            for _ in 0..100 {
                sum += surface.apply(&p.sample(&mut rng)?)?.iter().sum::<f64>();
            }
            Ok(sum)
        })?;
        metrics.insert("sample_checksum".into(), checksum);
    }
    metrics.insert("total_compute_s".into(), start.elapsed().as_secs_f64());
    write_metrics(output, method, refinement, field_mode, &metrics)?;
    write_surface(output, &surface_mean, &surface_truth, &surface_covariance)?;
    let mut csv = csv::Writer::from_path(output.join("field.csv"))?;
    csv.write_record([
        "kind",
        "index",
        "mean",
        "prior_variance",
        "posterior_variance",
        "variance_ratio",
        "posterior_estimator_se",
    ])?;
    for (i, mean) in field_mean.iter().enumerate() {
        let mut row = vec![
            if i < n { "one_form" } else { "flux" }.to_string(),
            (if i < n { i } else { i - n }).to_string(),
            format!("{mean:.17e}"),
        ];
        if let Some((pre, post)) = &field_estimates {
            row.extend(
                [
                    pre.values[i],
                    post.values[i],
                    post.values[i] / pre.values[i],
                    post.batch_standard_error.as_ref().map_or(0.0, |se| se[i]),
                ]
                .map(|v| format!("{v:.17e}")),
            );
        } else {
            row.extend(vec![String::new(); 4]);
        }
        csv.write_record(row)?;
    }
    csv.flush()?;
    let mut csv = csv::Writer::from_path(output.join("covariance_actions.csv"))?;
    for action in actions {
        csv.serialize(action)?;
    }
    csv.flush()?;
    drop(coords);
    eprintln!(
        "{method}, N={n}: selected PDE UQ {:.3} s; complete requested work {:.3} s",
        metrics["selected_uq_total_s"], metrics["total_compute_s"]
    );
    Ok(())
}

fn sparse_features(map: &LinearMap, features: &Mat<f64>) -> Mat<f64> {
    let mut product = Mat::zeros(map.output_dimension(), features.ncols());
    // Column-major dense output; sparse map application costs O(nnz(map) r).
    for column in 0..features.ncols() {
        for (row, inner, value) in map.matrix().triplet_iter() {
            product[(row, column)] += value * features[(inner, column)];
        }
    }
    product
}

/// Exact conditioning within the retained feature space, with the SAME sparse
/// PDE noise precision E/sigma^2 as the other methods. If a=F z, z~N(0,I),
/// the latent precision is I+(KF)'E(KF)/sigma^2+(HF)'(HF)/sensor_sigma^2.
/// Only r-by-r and requested-output covariances are dense; E is never inverted.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn weighted_spectral_outputs(
    features: &Mat<f64>,
    pde: &LinearMap,
    residual_metric: &LinearMap,
    pde_values: &[f64],
    pde_sigma: f64,
    sensors: &LinearMap,
    sensor_values: &[f64],
    output: &LinearMap,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let prepared = WeightedSpectralOutput::new(
        features,
        pde,
        residual_metric,
        pde_values,
        pde_sigma,
        sensors,
        output,
    )?;
    Ok((prepared.mean(sensor_values)?, prepared.covariance()?))
}

struct WeightedSpectralOutput {
    factor: faer::linalg::solvers::Llt<f64>,
    outputs: Mat<f64>,
    sensor_features: Mat<f64>,
    fixed_pde_rhs: Mat<f64>,
}

impl WeightedSpectralOutput {
    #[allow(clippy::too_many_arguments)]
    fn new(
        features: &Mat<f64>,
        pde: &LinearMap,
        residual_metric: &LinearMap,
        pde_values: &[f64],
        pde_sigma: f64,
        sensors: &LinearMap,
        output: &LinearMap,
    ) -> Result<Self> {
        let mut pde_features = sparse_features(pde, features);
        let mut sensor_features = sparse_features(sensors, features);
        for j in 0..features.ncols() {
            for i in 0..pde_features.nrows() {
                pde_features[(i, j)] /= pde_sigma;
            }
            for i in 0..sensor_features.nrows() {
                sensor_features[(i, j)] /= SENSOR_SIGMA;
            }
        }
        let weighted_pde = sparse_features(residual_metric, &pde_features);
        let mut precision = pde_features.transpose() * &weighted_pde
            + sensor_features.transpose() * &sensor_features;
        for i in 0..features.ncols() {
            precision[(i, i)] += 1.0;
        }
        let factor = precision
            .llt(Side::Lower)
            .map_err(|e| FeecGmrfError::Inference(format!("spectral latent precision: {e:?}")))?;
        let weighted_values = residual_metric.apply(pde_values)?;
        let pde_rhs = Mat::from_fn(pde_values.len(), 1, |i, _| weighted_values[i] / pde_sigma);
        let outputs = sparse_features(output, features);
        let fixed_pde_rhs = pde_features.transpose() * &pde_rhs;
        Ok(Self {
            factor,
            outputs,
            sensor_features,
            fixed_pde_rhs,
        })
    }

    fn mean(&self, sensor_values: &[f64]) -> Result<Vec<f64>> {
        let sensor_rhs = Mat::from_fn(sensor_values.len(), 1, |i, _| {
            sensor_values[i] / SENSOR_SIGMA
        });
        let mut latent_mean = &self.fixed_pde_rhs + self.sensor_features.transpose() * &sensor_rhs;
        self.factor.solve_in_place(latent_mean.as_mut());
        let mean = &self.outputs * &latent_mean;
        Ok((0..mean.nrows()).map(|i| mean[(i, 0)]).collect())
    }

    fn covariance(&self) -> Result<Vec<Vec<f64>>> {
        let mut solved = self.outputs.transpose().to_owned();
        self.factor.solve_in_place(solved.as_mut());
        let covariance = &self.outputs * &solved;
        Ok((0..covariance.nrows())
            .map(|i| {
                (0..covariance.ncols())
                    .map(|j| covariance[(i, j)])
                    .collect()
            })
            .collect())
    }
}

fn replacement_sensor_values(sensor_truth: &[f64]) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    (0..REPLACEMENT_DATASETS)
        .map(|_| {
            sensor_truth
                .iter()
                .map(|&v| {
                    v + SENSOR_SIGMA
                        * <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                })
                .collect()
        })
        .collect()
}

fn cached_sensor_update_check(
    metrics: &mut BTreeMap<String, f64>,
    mean: &[f64],
    covariance: &[Vec<f64>],
    initial: &[f64],
    replacement: &[Vec<f64>],
    direct_means: &[Vec<f64>],
) -> Result<()> {
    // Here sensors are exactly the first six output rows. Therefore the
    // cached data-to-output gain is C_G[:,0..6] / sensor_sigma^2 for ALL methods.
    let cached = timed(metrics, "cached_replacement_means", || {
        Ok(replacement
            .iter()
            .map(|data| {
                mean.iter()
                    .enumerate()
                    .map(|(i, &mu)| {
                        mu + data
                            .iter()
                            .zip(initial)
                            .enumerate()
                            .map(|(j, (new, old))| {
                                covariance[i][j] * (new - old) / SENSOR_SIGMA.powi(2)
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>())
    })?;
    let error = cached
        .iter()
        .zip(direct_means)
        .map(|(a, b)| relative_vector_error(a, b))
        .fold(0.0, f64::max);
    metrics.insert("cached_update_maximum_relative_error".into(), error);
    if error > 1e-8 {
        return Err(FeecGmrfError::Inference(format!(
            "cached output update discrepancy {error}"
        )));
    }
    Ok(())
}

fn write_replacement_means(
    output: &Path,
    means: &[Vec<f64>],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut csv = csv::Writer::from_path(output.join("replacement_surface_means.csv"))?;
    for mean in means {
        csv.serialize(mean)?;
    }
    csv.flush()?;
    Ok(())
}

fn write_metrics(
    output: &Path,
    method: &str,
    refinement: usize,
    field_mode: &str,
    metrics: &BTreeMap<String, f64>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut csv = csv::Writer::from_path(output.join("metrics.csv"))?;
    csv.write_record(["method", "refinement", "field_mode", "quantity", "value"])?;
    for (key, value) in metrics {
        csv.write_record([
            method,
            &refinement.to_string(),
            field_mode,
            key,
            &format!("{value:.17e}"),
        ])?;
    }
    csv.flush()?;
    Ok(())
}

fn write_surface(
    output: &Path,
    mean: &[f64],
    truth: &[f64],
    covariance: &[Vec<f64>],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut csv = csv::Writer::from_path(output.join("surface.csv"))?;
    csv.write_record(["row", "mean", "truth", "variance"])?;
    for i in 0..mean.len() {
        csv.serialize((i, mean[i], truth[i], covariance[i][i]))?;
    }
    csv.flush()?;
    let mut csv = csv::Writer::from_path(output.join("surface_covariance.csv"))?;
    for row in covariance {
        csv.serialize(row)?;
    }
    csv.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_maps_detect_gradients_and_have_convergent_analytic_orientation() {
        let mut interpolation_errors = Vec::new();
        for refinement in [4, 8] {
            let (topology, coords) =
                CartesianMeshInfo::new_unit_scaled(3, refinement, 1.0).compute_coord_complex();
            let metric = coords.to_edge_lengths(&topology);
            let operators =
                ExactFormOperators::from_feec_homogeneous_full_relative(&topology, &metric, 1)
                    .unwrap();
            let maps = local_output_maps(&topology, &coords, &metric, &operators).unwrap();
            let incoming = operators.incoming().unwrap();
            let scalar: Vec<_> = (0..incoming.lower().dimension())
                .map(|i| (i as f64 * 0.73).sin())
                .collect();
            let gradient = incoming.active_derivative().apply(&scalar).unwrap();
            let observed_gradient = maps.apply(&gradient).unwrap();
            assert!(observed_gradient[..6].iter().any(|v| v.abs() > 1e-4));
            assert!(observed_gradient[6..].iter().all(|v| v.abs() < 1e-10));
            let smooth = DiffFormClosure::one_form(
                |p| {
                    let s: Vec<_> = p.iter().map(|x| (std::f64::consts::PI * x).sin()).collect();
                    Vector::from_vec(vec![s[1] * s[2], s[2] * s[0], s[0] * s[1]])
                        * continuum_amplitude()
                },
                3,
            );
            let full = cochain_projection(
                &smooth,
                &topology,
                &coords,
                Some(&SimplexQuadRule::order3(1)),
            );
            let active: Vec<_> = operators
                .state()
                .boundary_layout()
                .active_dofs()
                .iter()
                .map(|&i| full.coeffs[i])
                .collect();
            let values = maps.apply(&active).unwrap();
            let truth = local_output_truth();
            let errors = [
                relative_vector_error(&values[..6], &truth[..6]),
                relative_vector_error(&values[6..], &truth[6..]),
            ];
            assert!(
                errors.iter().all(|&e| e < 0.25),
                "local orientation/interpolation: {errors:?}"
            );
            interpolation_errors.push(errors);
        }
        for i in 0..2 {
            assert!(
                interpolation_errors[1][i] < 0.7 * interpolation_errors[0][i],
                "{interpolation_errors:?}"
            );
        }
    }

    #[test]
    fn local_supported_and_gp_agree_for_unobserved_one_form_and_curl_outputs() {
        let root = std::env::temp_dir().join(format!("c51-local-gaussian-{}", std::process::id()));
        for method in ["supported", "gp"] {
            run_with_observables(
                method,
                4,
                0.1,
                "none",
                &root.join(method),
                Some(206.6615202953504),
                true,
                true,
            )
            .unwrap();
        }
        for file in ["surface_covariance.csv", "replacement_surface_means.csv"] {
            let read = |method: &str| -> Vec<f64> {
                csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_path(root.join(method).join(file))
                    .unwrap()
                    .records()
                    .flat_map(|r| {
                        r.unwrap()
                            .iter()
                            .map(|v| v.parse::<f64>().unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };
            assert!(
                relative_vector_error(&read("gp"), &read("supported")) < 1e-8,
                "{file}"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn continuum_forcing_has_nonzero_decreasing_pde_flux_error() {
        let root = std::env::temp_dir().join(format!("c51-continuum-pde-{}", std::process::id()));
        let mut errors = Vec::new();
        for refinement in [4, 8] {
            let directory = root.join(format!("n{refinement}"));
            run_with_workload("pde", refinement, 0.1, "none", &directory, None, true).unwrap();
            let data: BTreeMap<String, f64> = csv::Reader::from_path(directory.join("metrics.csv"))
                .unwrap()
                .records()
                .map(|r| {
                    let r = r.unwrap();
                    (r[3].to_owned(), r[4].parse().unwrap())
                })
                .collect();
            assert!(data["surface_interpolation_relative_error"] < 1e-10);
            assert!(data["surface_relative_error"] > 1e-4); // Detect discrete-manufacturing tautologies.
            errors.push(data["surface_relative_error"]);
        }
        assert!(errors[0] < 0.05);
        assert!(errors[1] < 0.35 * errors[0], "{errors:?}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn continuum_supported_and_gp_agree_including_replacement_data() {
        let root =
            std::env::temp_dir().join(format!("c51-continuum-gaussian-{}", std::process::id()));
        for method in ["supported", "gp"] {
            run_with_workload(
                method,
                4,
                0.1,
                "none",
                &root.join(method),
                Some(206.6615202953504),
                true,
            )
            .unwrap();
        }
        for file in ["surface_covariance.csv", "replacement_surface_means.csv"] {
            let read = |method: &str| -> Vec<f64> {
                csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_path(root.join(method).join(file))
                    .unwrap()
                    .records()
                    .flat_map(|r| {
                        r.unwrap()
                            .iter()
                            .map(|v| v.parse::<f64>().unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };
            assert!(
                relative_vector_error(&read("gp"), &read("supported")) < 1e-8,
                "{file}"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn weighted_spectral_outputs_match_covariance_conditioning_with_correlated_noise() {
        let features = faer::mat![[1.0, 0.3, -0.2], [0.1, 0.7, 0.4]];
        let pde =
            LinearMap::weighted_rows(2, &[vec![(0, 2.0), (1, -0.5)], vec![(0, 0.1), (1, 1.0)]])
                .unwrap();
        let metric =
            LinearMap::weighted_rows(2, &[vec![(0, 2.0), (1, 0.6)], vec![(0, 0.6), (1, 1.5)]])
                .unwrap();
        let sensors =
            LinearMap::weighted_row(2, &[(0, SENSOR_SIGMA), (1, -0.4 * SENSOR_SIGMA)]).unwrap();
        let output = LinearMap::identity(2);
        let values = [0.2, -0.7, 0.3 * SENSOR_SIGMA];
        let (mean, covariance) = weighted_spectral_outputs(
            &features,
            &pde,
            &metric,
            &values[..2],
            0.8,
            &sensors,
            &values[2..],
            &output,
        )
        .unwrap();
        // Independent observation-space conditioning, including an analytic
        // inverse of the non-diagonal residual metric. This catches using E
        // as a covariance, dropping off-diagonals, and applying sigma once.
        let observed = sparse_features(&LinearMap::stack(&[pde, sensors]).unwrap(), &features);
        let rows = |matrix: Mat<f64>| -> Vec<Vec<f64>> {
            (0..matrix.nrows())
                .map(|i| (0..matrix.ncols()).map(|j| matrix[(i, j)]).collect())
                .collect()
        };
        let blocks = ExactObservationCovarianceBlocks::new(
            rows(&observed * observed.transpose()),
            rows(&features * observed.transpose()),
            rows(&features * features.transpose()),
        )
        .unwrap();
        let noise = vec![
            vec![0.64 * 1.5 / 2.64, -0.64 * 0.6 / 2.64, 0.0],
            vec![-0.64 * 0.6 / 2.64, 0.64 * 2.0 / 2.64, 0.0],
            vec![0.0, 0.0, SENSOR_SIGMA.powi(2)],
        ];
        let gp = ExactObservationPosterior::from_prior_covariance_blocks_with_noise_covariance(
            blocks, noise,
        )
        .unwrap();
        let reference_mean = gp.output_mean(&values).unwrap();
        let reference_covariance = gp.posterior_output_covariance();
        for i in 0..2 {
            assert!((mean[i] - reference_mean[i]).abs() < 1.0e-10);
            for j in 0..2 {
                assert!((covariance[i][j] - reference_covariance[i][j]).abs() < 1.0e-10);
            }
        }
    }

    #[test]
    fn combined_one_form_pde_gp_matches_supported_mean_and_covariance() {
        let directory =
            std::env::temp_dir().join(format!("c51-combined-pde-{}", std::process::id()));
        let supported = directory.join("supported");
        let gp = directory.join("gp");
        run("supported", 4, 0.1, "none", &supported).unwrap();
        run("gp", 4, 0.1, "none", &gp).unwrap();
        let read = |path: &Path, headers: bool| -> Vec<Vec<f64>> {
            csv::ReaderBuilder::new()
                .has_headers(headers)
                .from_path(path)
                .unwrap()
                .records()
                .map(|row| row.unwrap().iter().map(|v| v.parse().unwrap()).collect())
                .collect()
        };
        for file in ["surface.csv", "surface_covariance.csv"] {
            let reference = read(&supported.join(file), file == "surface.csv");
            let candidate = read(&gp.join(file), file == "surface.csv");
            let mut error = 0.0;
            let mut scale = 0.0;
            for (x, y) in candidate.iter().zip(&reference) {
                assert_eq!(x.len(), y.len());
                // In surface.csv only compare mean and variance, not row IDs or truth.
                let indices: Vec<_> = if file == "surface.csv" {
                    vec![1, 3]
                } else {
                    (0..x.len()).collect()
                };
                for i in indices {
                    error += (x[i] - y[i]).powi(2);
                    scale += y[i].powi(2);
                }
            }
            assert!(
                (error / scale).sqrt() < 1.0e-8,
                "{file}: relative discrepancy {}",
                (error / scale).sqrt()
            );
        }
        fs::remove_dir_all(directory).unwrap();
    }
}
