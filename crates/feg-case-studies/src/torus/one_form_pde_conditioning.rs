use crate::torus::diagnostics::{
    build_analytic_torus_harmonic_basis, build_harmonic_orthogonality_constraints,
    infer_torus_radii,
};
use crate::torus::one_form_conditioning::{
    SurfaceVectorVarianceMode, Torus1FormAmbientVarianceFields, Torus1FormVarianceComponentFields,
    Torus1FormVarianceFieldSet,
};
use crate::visual_output;
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::cochain::{cochain_projection, Cochain};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::field::ExteriorField;
use feg_infer::conditioning::linear::{
    DerivedOperator, DerivedOperatorSet, DerivedVarianceMode, HarmonicSubspace, HutchinsonConfig,
    LinearGaussianConditioningProblem,
};
use feg_infer::prior::matern::convert_whittle_params_to_matern;
use feg_infer::prior::matern::one_form::{
    build_hodge_laplacian_1form, build_matern_precision_1form, build_matern_system_matrix_1form,
    feec_csr_to_gmrf, feec_vec_to_gmrf, HodgeLaplacian1Form, MaternConfig, MaternMassInverse,
};
use feg_infer::sparse::gmrf_vec_to_feec;
use formoniq::fe::{fe_l2_error, l2_norm};
use formoniq::io::sample_1form_cell_vectors;
use formoniq::torus_convergence::build_torus_reference_fields;
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::{
    DenseMatrix as GmrfDenseMatrix, SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector,
};
use gmrf_core::{
    clip_vector_to_prior as gmrf_clip_vector_to_prior,
    estimate_batched_transformed_hutchinson_decomposition,
    estimate_batched_transformed_hutchinson_with_solve, ConstrainedPrecisionSolver, Gmrf,
    GmrfError, ProbeBatchConfig, SparseRowOperator, TransformedVarianceDecomposition,
    VarianceFloor,
};
use manifold::{
    geometry::{
        coord::{
            mesh::MeshCoords,
            simplex::{barycenter_local, SimplexHandleExt},
        },
        metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const EPS: f64 = 1e-12;
const DEFAULT_NUM_VARIANCE_PROBES: usize = 256;
const DEFAULT_VARIANCE_BATCH_COUNT: usize = 8;
const DEFAULT_RNG_SEED: u64 = 13;
const SMOOTHING_BANDWIDTH_SCALE: f64 = 0.5;
const SMOOTHING_CUTOFF_SCALE: f64 = 1.0;

pub use crate::torus::posterior_residual_weight::{
    default_torus_pde_posterior_mean_weight_mesh_levels, default_torus_shell_mesh_path,
    run_torus_1form_pde_posterior_mean_weight_sweep,
    write_torus_1form_pde_posterior_mean_weight_sweep_outputs, Torus1FormPdeMeshLevel,
    Torus1FormPdePosteriorMeanWeightRow, Torus1FormPdePosteriorMeanWeightSummaryRow,
    Torus1FormPdePosteriorMeanWeightSweepConfig, Torus1FormPdePosteriorMeanWeightSweepResult,
};

#[derive(Debug, Clone)]
pub struct Torus1FormPdeConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_variance_probes: usize,
    pub variance_batch_count: usize,
    pub rng_seed: u64,
}

impl Default for Torus1FormPdeConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_torus_shell_resolution_1_mesh_path(),
            kappa: 4.0,
            tau: 1.0,
            noise_variance: 1e-8,
            surface_vector_variance_mode: SurfaceVectorVarianceMode::Exact,
            num_variance_probes: DEFAULT_NUM_VARIANCE_PROBES,
            variance_batch_count: DEFAULT_VARIANCE_BATCH_COUNT,
            rng_seed: DEFAULT_RNG_SEED,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeVarianceFields {
    pub reconstructed: Torus1FormVarianceComponentFields,
    pub surface_vector: Torus1FormAmbientVarianceFields,
    pub smoothed: Torus1FormVarianceComponentFields,
    pub circulation: Torus1FormVarianceFieldSet,
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeConditioningResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub edge_theta: FeecVector,
    pub edge_phi: FeecVector,
    pub toroidal_alignment_sq: FeecVector,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_variance_probes: usize,
    pub variance_batch_count: usize,
    pub rng_seed: u64,
    pub effective_range: f64,
    pub truth: FeecVector,
    pub rhs: FeecVector,
    pub posterior_mean: FeecVector,
    pub posterior_rhs: FeecVector,
    pub pde_residual: FeecVector,
    pub absolute_mean_error: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub variance_reduction: FeecVector,
    pub variance_ratio: FeecVector,
    pub harmonic_free_truth: FeecVector,
    pub harmonic_free_posterior_mean: FeecVector,
    pub harmonic_free_absolute_mean_error: FeecVector,
    pub harmonic_free_prior_variance: FeecVector,
    pub harmonic_free_posterior_variance: FeecVector,
    pub harmonic_free_variance_reduction: FeecVector,
    pub harmonic_free_variance_ratio: FeecVector,
    pub harmonic_coefficients_truth: [f64; 2],
    pub harmonic_coefficients_posterior_mean: [f64; 2],
    pub posterior_deterministic_l2_error: f64,
    pub l2_error: f64,
    pub hd_error: f64,
    pub truth_residual_norm: f64,
    pub truth_relative_residual_norm: f64,
    pub posterior_residual_norm: f64,
    pub posterior_relative_residual_norm: f64,
    pub variance_fields: Torus1FormPdeVarianceFields,
}

pub(crate) struct PreparedTorus1FormPdeProblem {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub metric: MeshLengths,
    pub edge_geometry: TorusEdgeGeometry,
    pub cell_geometry: TorusCellGeometry,
    pub hodge: HodgeLaplacian1Form,
    pub system_matrix: FeecCsr,
    pub harmonic_basis_orthonormal: FeecMatrix,
    pub harmonic_constraints: GmrfDenseMatrix,
    pub truth: FeecVector,
    pub rhs: FeecVector,
    pub reconstructed_stacked_operator: SparseRowLinearOperator,
    pub surface_vector_stacked_operator: SparseRowLinearOperator,
    pub smoothed_stacked_operator: SparseRowLinearOperator,
    pub circulation_operator: SparseRowLinearOperator,
    pub effective_range: f64,
}

pub(crate) type SparseRowLinearOperator = SparseRowOperator;

pub(crate) struct TorusEdgeGeometry {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    pub toroidal_alignment_sq: Vec<f64>,
}

pub(crate) struct TorusCellGeometry {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
}

pub(crate) struct HutchinsonVarianceEstimates {
    pub unconstrained: GmrfVector,
    pub harmonic_free: GmrfVector,
}

pub(crate) struct HutchinsonWorkspace {
    gmrf: Gmrf,
    harmonic_constraints: GmrfDenseMatrix,
}

pub(crate) struct Torus1FormVarianceComponentEstimates {
    toroidal: HutchinsonVarianceEstimates,
    poloidal: HutchinsonVarianceEstimates,
}

pub(crate) struct Torus1FormAmbientVarianceEstimates {
    x: HutchinsonVarianceEstimates,
    y: HutchinsonVarianceEstimates,
    z: HutchinsonVarianceEstimates,
}

pub fn default_torus_shell_resolution_1_mesh_path() -> PathBuf {
    default_torus_shell_mesh_path(1)
}

pub fn run_torus_1form_pde_conditioning(
    config: &Torus1FormPdeConditioningConfig,
) -> Result<Torus1FormPdeConditioningResult, Box<dyn Error>> {
    validate_config(config)?;
    let prepared = prepare_torus_1form_pde_problem(config)?;
    run_prepared_torus_1form_pde_conditioning(&prepared, config)
}

pub fn write_torus_1form_pde_conditioning_outputs(
    result: &Torus1FormPdeConditioningResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    write_overall_summary(result, out_dir)?;
    write_edge_fields_vtu(result, out_dir)?;
    write_edge_csv(result, out_dir)?;
    write_surface_vector_vtu(result, out_dir)?;
    write_variance_field_vtus(result, out_dir)?;

    Ok(())
}

pub(crate) fn prepare_torus_1form_pde_problem(
    config: &Torus1FormPdeConditioningConfig,
) -> Result<PreparedTorus1FormPdeProblem, Box<dyn Error>> {
    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let edge_geometry = build_torus_edge_geometry(&topology, &coords)?;
    let cell_geometry = build_torus_cell_geometry(
        &topology,
        &coords,
        edge_geometry.major_radius,
        edge_geometry.minor_radius,
    )
    .map_err(invalid_data)?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let system_matrix = build_matern_system_matrix_1form(&hodge, config.kappa);

    let harmonic_basis =
        build_analytic_torus_harmonic_basis(&topology, &coords, &metric).map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis(&harmonic_basis, &hodge.mass_u).map_err(invalid_data)?;
    let harmonic_constraints =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u)
            .map_err(invalid_data)?;

    let (u_exact, _dif_solution_exact) = build_torus_reference_fields();
    let truth_cochain = cochain_projection(&u_exact, &topology, &coords, None);
    let truth = truth_cochain.coeffs.clone();
    let rhs = &system_matrix * &truth;

    let smoothing_bandwidth = SMOOTHING_BANDWIDTH_SCALE
        * convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2).2;
    let smoothing_cutoff = SMOOTHING_CUTOFF_SCALE
        * convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2).2;
    let (_nu, _variance, effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);

    let toroidal_operator =
        build_reconstructed_component_operator(&topology, &coords, &cell_geometry, true)
            .map_err(invalid_data)?;
    let poloidal_operator =
        build_reconstructed_component_operator(&topology, &coords, &cell_geometry, false)
            .map_err(invalid_data)?;
    let surface_x_operator =
        build_embedded_component_operator(&topology, &coords, 0).map_err(invalid_data)?;
    let surface_y_operator =
        build_embedded_component_operator(&topology, &coords, 1).map_err(invalid_data)?;
    let surface_z_operator =
        build_embedded_component_operator(&topology, &coords, 2).map_err(invalid_data)?;
    let reconstructed_stacked_operator =
        SparseRowLinearOperator::stack(&[&toroidal_operator, &poloidal_operator])
            .map_err(|err| invalid_data(err.to_string()))?;
    let surface_vector_stacked_operator = SparseRowLinearOperator::stack(&[
        &surface_x_operator,
        &surface_y_operator,
        &surface_z_operator,
    ])
    .map_err(|err| invalid_data(err.to_string()))?;
    let smoothing_operator =
        build_gaussian_smoothing_operator(&cell_geometry, smoothing_bandwidth, smoothing_cutoff)
            .map_err(invalid_data)?;
    let smoothed_toroidal_operator =
        SparseRowLinearOperator::compose(&smoothing_operator, &toroidal_operator)
            .map_err(|err| invalid_data(err.to_string()))?;
    let smoothed_poloidal_operator =
        SparseRowLinearOperator::compose(&smoothing_operator, &poloidal_operator)
            .map_err(|err| invalid_data(err.to_string()))?;
    let smoothed_stacked_operator =
        SparseRowLinearOperator::stack(&[&smoothed_toroidal_operator, &smoothed_poloidal_operator])
            .map_err(|err| invalid_data(err.to_string()))?;
    let circulation_operator =
        build_local_circulation_operator(&topology, hodge.mass_u.nrows()).map_err(invalid_data)?;

    Ok(PreparedTorus1FormPdeProblem {
        topology,
        coords,
        metric,
        edge_geometry,
        cell_geometry,
        hodge,
        system_matrix,
        harmonic_basis_orthonormal,
        harmonic_constraints,
        truth,
        rhs,
        reconstructed_stacked_operator,
        surface_vector_stacked_operator,
        smoothed_stacked_operator,
        circulation_operator,
        effective_range,
    })
}

pub(crate) fn run_prepared_torus_1form_pde_conditioning(
    prepared: &PreparedTorus1FormPdeProblem,
    config: &Torus1FormPdeConditioningConfig,
) -> Result<Torus1FormPdeConditioningResult, Box<dyn Error>> {
    let prior_precision = build_matern_precision_1form(
        &prepared.topology,
        &prepared.metric,
        &prepared.hodge,
        MaternConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
        },
    );
    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let mut derived_operators = DerivedOperatorSet::new();
    derived_operators.insert(
        "reconstructed".to_string(),
        DerivedOperator {
            operator: prepared.reconstructed_stacked_operator.clone(),
            variance_mode: DerivedVarianceMode::Hutchinson,
        },
    );
    derived_operators.insert(
        "surface_vector".to_string(),
        DerivedOperator {
            operator: prepared.surface_vector_stacked_operator.clone(),
            variance_mode: match config.surface_vector_variance_mode {
                SurfaceVectorVarianceMode::Exact => DerivedVarianceMode::Exact,
                SurfaceVectorVarianceMode::Hutchinson
                | SurfaceVectorVarianceMode::HutchinsonStabilized => {
                    DerivedVarianceMode::Hutchinson
                }
            },
        },
    );
    derived_operators.insert(
        "smoothed".to_string(),
        DerivedOperator {
            operator: prepared.smoothed_stacked_operator.clone(),
            variance_mode: DerivedVarianceMode::Hutchinson,
        },
    );
    derived_operators.insert(
        "circulation".to_string(),
        DerivedOperator {
            operator: prepared.circulation_operator.clone(),
            variance_mode: DerivedVarianceMode::Hutchinson,
        },
    );

    let conditioning = LinearGaussianConditioningProblem {
        prior_precision: q_prior,
        observation_operator: feec_csr_to_gmrf(&prepared.system_matrix),
        observations: feec_vec_to_gmrf(&prepared.rhs),
        noise_variance: config.noise_variance,
        harmonic_subspace: Some(HarmonicSubspace {
            basis: prepared.harmonic_basis_orthonormal.clone(),
            constraints: prepared.harmonic_constraints.clone(),
            projector: None,
        }),
        derived_operators,
        hutchinson: HutchinsonConfig {
            num_probes: config.num_variance_probes,
            batch_count: config.variance_batch_count,
            rng_seed: config.rng_seed,
        },
    }
    .solve()?;

    let posterior_mean = gmrf_vec_to_feec(&conditioning.posterior_mean);
    let cell_count = prepared.cell_geometry.theta.len();
    let reconstructed_prior = split_component_estimates(
        decomposition_to_estimates(get_derived_decomposition(
            &conditioning,
            "reconstructed",
            true,
        )?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let reconstructed_posterior = split_component_estimates(
        decomposition_to_estimates(get_derived_decomposition(
            &conditioning,
            "reconstructed",
            false,
        )?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let surface_vector_prior = split_ambient_estimates(
        decomposition_to_estimates(get_derived_decomposition(
            &conditioning,
            "surface_vector",
            true,
        )?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let surface_vector_posterior_raw = split_ambient_estimates(
        decomposition_to_estimates(get_derived_decomposition(
            &conditioning,
            "surface_vector",
            false,
        )?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let surface_vector_posterior =
        if config.surface_vector_variance_mode == SurfaceVectorVarianceMode::HutchinsonStabilized {
            clip_hutchinson_posterior_to_prior(&surface_vector_prior, &surface_vector_posterior_raw)
        } else {
            surface_vector_posterior_raw
        };
    let smoothed_prior = split_component_estimates(
        decomposition_to_estimates(get_derived_decomposition(&conditioning, "smoothed", true)?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let smoothed_posterior = split_component_estimates(
        decomposition_to_estimates(get_derived_decomposition(&conditioning, "smoothed", false)?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let circulation_prior = decomposition_to_estimates(get_derived_decomposition(
        &conditioning,
        "circulation",
        true,
    )?);
    let circulation_posterior = decomposition_to_estimates(get_derived_decomposition(
        &conditioning,
        "circulation",
        false,
    )?);

    assemble_torus_1form_pde_conditioning_result(
        prepared,
        config,
        posterior_mean,
        &conditioning.prior_latent_variance.unconstrained_diag,
        &conditioning.posterior_latent_variance.unconstrained_diag,
        &conditioning.prior_latent_variance.constrained_diag,
        &conditioning.posterior_latent_variance.constrained_diag,
        &reconstructed_prior,
        &reconstructed_posterior,
        &surface_vector_prior,
        &surface_vector_posterior,
        &smoothed_prior,
        &smoothed_posterior,
        &circulation_prior,
        &circulation_posterior,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_torus_1form_pde_conditioning_result(
    prepared: &PreparedTorus1FormPdeProblem,
    config: &Torus1FormPdeConditioningConfig,
    posterior_mean: FeecVector,
    prior_variance_diag: &GmrfVector,
    posterior_variance_diag: &GmrfVector,
    prior_harmonic_free_variance_diag: &GmrfVector,
    posterior_harmonic_free_variance_diag: &GmrfVector,
    reconstructed_prior: &Torus1FormVarianceComponentEstimates,
    reconstructed_posterior: &Torus1FormVarianceComponentEstimates,
    surface_vector_prior: &Torus1FormAmbientVarianceEstimates,
    surface_vector_posterior: &Torus1FormAmbientVarianceEstimates,
    smoothed_prior: &Torus1FormVarianceComponentEstimates,
    smoothed_posterior: &Torus1FormVarianceComponentEstimates,
    circulation_prior: &HutchinsonVarianceEstimates,
    circulation_posterior: &HutchinsonVarianceEstimates,
) -> Result<Torus1FormPdeConditioningResult, Box<dyn Error>> {
    let absolute_mean_error = absolute_difference(&posterior_mean, &prepared.truth);
    let prior_variance = gmrf_vec_to_feec(prior_variance_diag);
    let posterior_variance = gmrf_vec_to_feec(posterior_variance_diag);
    let variance_reduction = &prior_variance - &posterior_variance;
    let variance_ratio = ratio_vector(&posterior_variance, &prior_variance);

    let harmonic_free_truth = remove_harmonic_content(
        &prepared.truth,
        &prepared.harmonic_basis_orthonormal,
        &prepared.hodge.mass_u,
    );
    let harmonic_free_posterior_mean = remove_harmonic_content(
        &posterior_mean,
        &prepared.harmonic_basis_orthonormal,
        &prepared.hodge.mass_u,
    );
    let harmonic_free_absolute_mean_error =
        absolute_difference(&harmonic_free_posterior_mean, &harmonic_free_truth);
    let harmonic_free_prior_variance = gmrf_vec_to_feec(prior_harmonic_free_variance_diag);
    let harmonic_free_posterior_variance = gmrf_vec_to_feec(posterior_harmonic_free_variance_diag);
    let harmonic_free_variance_reduction =
        &harmonic_free_prior_variance - &harmonic_free_posterior_variance;
    let harmonic_free_variance_ratio = ratio_vector(
        &harmonic_free_posterior_variance,
        &harmonic_free_prior_variance,
    );

    let harmonic_coefficients_truth = harmonic_coefficients(
        &prepared.truth,
        &prepared.harmonic_basis_orthonormal,
        &prepared.hodge.mass_u,
    )
    .map_err(invalid_data)?;
    let harmonic_coefficients_posterior_mean = harmonic_coefficients(
        &posterior_mean,
        &prepared.harmonic_basis_orthonormal,
        &prepared.hodge.mass_u,
    )
    .map_err(invalid_data)?;

    let (u_exact, dif_solution_exact) = build_torus_reference_fields();
    let posterior_mean_cochain = Cochain::new(1, posterior_mean.clone());
    let posterior_dif = posterior_mean_cochain.dif(&prepared.topology);
    let deterministic_solution = Cochain::new(1, prepared.truth.clone());
    let posterior_deterministic_l2_error = l2_norm(
        &(posterior_mean_cochain.clone() - deterministic_solution),
        &prepared.topology,
        &prepared.metric,
    );
    let l2_error = fe_l2_error(
        &posterior_mean_cochain,
        &u_exact,
        &prepared.topology,
        &prepared.coords,
    );
    let hd_error = fe_l2_error(
        &posterior_dif,
        &dif_solution_exact,
        &prepared.topology,
        &prepared.coords,
    );

    let truth_rhs = &prepared.system_matrix * &prepared.truth;
    let posterior_rhs = &prepared.system_matrix * &posterior_mean;
    let truth_residual = &truth_rhs - &prepared.rhs;
    let pde_residual = &posterior_rhs - &prepared.rhs;
    let rhs_norm = prepared.rhs.norm().max(EPS);

    let variance_fields = Torus1FormPdeVarianceFields {
        reconstructed: build_component_field_set(reconstructed_prior, reconstructed_posterior),
        surface_vector: build_ambient_field_set(surface_vector_prior, surface_vector_posterior),
        smoothed: build_component_field_set(smoothed_prior, smoothed_posterior),
        circulation: build_variance_field_set(circulation_prior, circulation_posterior),
    };

    Ok(Torus1FormPdeConditioningResult {
        topology: prepared.topology.clone(),
        coords: prepared.coords.clone(),
        edge_theta: FeecVector::from_vec(prepared.edge_geometry.theta.clone()),
        edge_phi: FeecVector::from_vec(prepared.edge_geometry.phi.clone()),
        toroidal_alignment_sq: FeecVector::from_vec(
            prepared.edge_geometry.toroidal_alignment_sq.clone(),
        ),
        major_radius: prepared.edge_geometry.major_radius,
        minor_radius: prepared.edge_geometry.minor_radius,
        surface_vector_variance_mode: config.surface_vector_variance_mode,
        num_variance_probes: config.num_variance_probes,
        variance_batch_count: config.variance_batch_count,
        rng_seed: config.rng_seed,
        effective_range: prepared.effective_range,
        truth: prepared.truth.clone(),
        rhs: prepared.rhs.clone(),
        posterior_mean,
        posterior_rhs,
        pde_residual: pde_residual.clone(),
        absolute_mean_error,
        prior_variance,
        posterior_variance,
        variance_reduction,
        variance_ratio,
        harmonic_free_truth,
        harmonic_free_posterior_mean,
        harmonic_free_absolute_mean_error,
        harmonic_free_prior_variance,
        harmonic_free_posterior_variance,
        harmonic_free_variance_reduction,
        harmonic_free_variance_ratio,
        harmonic_coefficients_truth,
        harmonic_coefficients_posterior_mean,
        posterior_deterministic_l2_error,
        l2_error,
        hd_error,
        truth_residual_norm: truth_residual.norm(),
        truth_relative_residual_norm: truth_residual.norm() / rhs_norm,
        posterior_residual_norm: pde_residual.norm(),
        posterior_relative_residual_norm: pde_residual.norm() / rhs_norm,
        variance_fields,
    })
}

fn get_derived_decomposition<'a>(
    conditioning: &'a feg_infer::conditioning::linear::LinearGaussianConditioningResult,
    name: &str,
    prior: bool,
) -> Result<&'a TransformedVarianceDecomposition, Box<dyn Error>> {
    let map = if prior {
        &conditioning.derived_prior_variances
    } else {
        &conditioning.derived_posterior_variances
    };
    map.get(name).ok_or_else(|| {
        invalid_data(format!("missing derived variance decomposition `{name}`")).into()
    })
}

fn decomposition_to_estimates(
    decomposition: &TransformedVarianceDecomposition,
) -> HutchinsonVarianceEstimates {
    HutchinsonVarianceEstimates {
        unconstrained: decomposition.unconstrained_diag.clone(),
        harmonic_free: decomposition.constrained_diag.clone(),
    }
}

pub(crate) fn validate_config(
    config: &Torus1FormPdeConditioningConfig,
) -> Result<(), Box<dyn Error>> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive").into());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive").into());
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err(invalid_input("noise_variance must be finite and positive").into());
    }
    if config.num_variance_probes == 0 {
        return Err(invalid_input("num_variance_probes must be >= 1").into());
    }
    if config.variance_batch_count == 0 {
        return Err(invalid_input("variance_batch_count must be >= 1").into());
    }
    Ok(())
}

pub(crate) fn build_hutchinson_workspace(
    precision: &GmrfSparseMatrix,
    harmonic_constraints: &GmrfDenseMatrix,
) -> Result<HutchinsonWorkspace, Box<dyn Error>> {
    let q_factor = precision.cholesky_sqrt_lower()?;
    let gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())?
            .with_precision_sqrt(q_factor);
    Ok(HutchinsonWorkspace {
        gmrf,
        harmonic_constraints: harmonic_constraints.clone(),
    })
}

pub(crate) fn exact_transformed_variances(
    workspace: &mut HutchinsonWorkspace,
    operator: &SparseRowLinearOperator,
) -> Result<HutchinsonVarianceEstimates, GmrfError> {
    let decomposition = workspace
        .gmrf
        .exact_transformed_variance_decomposition(operator, &workspace.harmonic_constraints)?;
    Ok(HutchinsonVarianceEstimates {
        unconstrained: decomposition.unconstrained_diag,
        harmonic_free: decomposition.constrained_diag,
    })
}

pub(crate) fn estimate_transformed_hutchinson_variances(
    workspace: &mut HutchinsonWorkspace,
    operator: &SparseRowLinearOperator,
    num_variance_probes: usize,
    variance_batch_count: usize,
    rng_seed: u64,
) -> Result<HutchinsonVarianceEstimates, Box<dyn Error>> {
    let estimate = estimate_batched_transformed_hutchinson_decomposition(
        &mut workspace.gmrf,
        operator,
        &workspace.harmonic_constraints,
        ProbeBatchConfig {
            num_probes: num_variance_probes,
            batch_count: variance_batch_count,
            rng_seed,
        },
        VarianceFloor::PositiveMean { scale: 1e-12 },
    )?;
    Ok(HutchinsonVarianceEstimates {
        unconstrained: estimate.decomposition.unconstrained_diag,
        harmonic_free: estimate.decomposition.constrained_diag,
    })
}

pub(crate) fn split_component_estimates(
    stacked: HutchinsonVarianceEstimates,
    cell_count: usize,
) -> Result<Torus1FormVarianceComponentEstimates, String> {
    if stacked.unconstrained.len() != 2 * cell_count
        || stacked.harmonic_free.len() != 2 * cell_count
    {
        return Err(
            "stacked component estimates must contain toroidal and poloidal blocks".to_string(),
        );
    }

    Ok(Torus1FormVarianceComponentEstimates {
        toroidal: HutchinsonVarianceEstimates {
            unconstrained: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.unconstrained[i]),
            ),
            harmonic_free: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.harmonic_free[i]),
            ),
        },
        poloidal: HutchinsonVarianceEstimates {
            unconstrained: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.unconstrained[cell_count + i]),
            ),
            harmonic_free: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.harmonic_free[cell_count + i]),
            ),
        },
    })
}

pub(crate) fn split_ambient_estimates(
    stacked: HutchinsonVarianceEstimates,
    cell_count: usize,
) -> Result<Torus1FormAmbientVarianceEstimates, String> {
    if stacked.unconstrained.len() != 3 * cell_count
        || stacked.harmonic_free.len() != 3 * cell_count
    {
        return Err(
            "stacked ambient estimates must contain x, y, and z component blocks".to_string(),
        );
    }

    let block = |values: &GmrfVector, block_index: usize| {
        GmrfVector::from_iterator(
            cell_count,
            (0..cell_count).map(|i| values[block_index * cell_count + i]),
        )
    };

    Ok(Torus1FormAmbientVarianceEstimates {
        x: HutchinsonVarianceEstimates {
            unconstrained: block(&stacked.unconstrained, 0),
            harmonic_free: block(&stacked.harmonic_free, 0),
        },
        y: HutchinsonVarianceEstimates {
            unconstrained: block(&stacked.unconstrained, 1),
            harmonic_free: block(&stacked.harmonic_free, 1),
        },
        z: HutchinsonVarianceEstimates {
            unconstrained: block(&stacked.unconstrained, 2),
            harmonic_free: block(&stacked.harmonic_free, 2),
        },
    })
}

pub(crate) fn clip_hutchinson_posterior_to_prior(
    prior: &Torus1FormAmbientVarianceEstimates,
    posterior: &Torus1FormAmbientVarianceEstimates,
) -> Torus1FormAmbientVarianceEstimates {
    Torus1FormAmbientVarianceEstimates {
        x: clip_hutchinson_estimates_to_prior(&prior.x, &posterior.x),
        y: clip_hutchinson_estimates_to_prior(&prior.y, &posterior.y),
        z: clip_hutchinson_estimates_to_prior(&prior.z, &posterior.z),
    }
}

fn clip_hutchinson_estimates_to_prior(
    prior: &HutchinsonVarianceEstimates,
    posterior: &HutchinsonVarianceEstimates,
) -> HutchinsonVarianceEstimates {
    HutchinsonVarianceEstimates {
        unconstrained: clip_vector_to_prior(&prior.unconstrained, &posterior.unconstrained),
        harmonic_free: clip_vector_to_prior(&prior.harmonic_free, &posterior.harmonic_free),
    }
}

fn clip_vector_to_prior(prior: &GmrfVector, posterior: &GmrfVector) -> GmrfVector {
    gmrf_clip_vector_to_prior(prior, posterior)
        .expect("prior and posterior variance estimates must have matching dimensions")
}

fn build_torus_edge_geometry(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<TorusEdgeGeometry, io::Error> {
    let (major_radius, minor_radius) = infer_torus_radii(coords).map_err(invalid_data)?;
    let edge_skeleton = topology.skeleton(1);

    let mut theta = Vec::with_capacity(edge_skeleton.len());
    let mut phi = Vec::with_capacity(edge_skeleton.len());
    let mut toroidal_alignment_sq = Vec::with_capacity(edge_skeleton.len());

    for edge in edge_skeleton.handle_iter() {
        let v0 = coords.coord(edge.vertices[0]);
        let v1 = coords.coord(edge.vertices[1]);
        let midpoint = (v0 + v1) / 2.0;
        let rho = (midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1])
            .sqrt()
            .max(EPS);
        let midpoint_theta = midpoint[2].atan2(rho - major_radius);
        let midpoint_phi = midpoint[1].atan2(midpoint[0]);

        let tangent = v1 - v0;
        let tangent_norm = tangent.norm();
        let alignment_sq = if tangent_norm <= EPS {
            0.0
        } else {
            let e_phi =
                FeecVector::from_column_slice(&[-midpoint_phi.sin(), midpoint_phi.cos(), 0.0]);
            let unit_tangent = tangent / tangent_norm;
            unit_tangent.dot(&e_phi).powi(2).clamp(0.0, 1.0)
        };

        theta.push(midpoint_theta);
        phi.push(midpoint_phi);
        toroidal_alignment_sq.push(alignment_sq);
    }

    Ok(TorusEdgeGeometry {
        major_radius,
        minor_radius,
        theta,
        phi,
        toroidal_alignment_sq,
    })
}

fn build_torus_cell_geometry(
    topology: &Complex,
    coords: &MeshCoords,
    major_radius: f64,
    minor_radius: f64,
) -> Result<TorusCellGeometry, String> {
    let mut theta = Vec::with_capacity(topology.cells().len());
    let mut phi = Vec::with_capacity(topology.cells().len());

    for cell in topology.cells().handle_iter() {
        let barycenter = cell.coord_simplex(coords).barycenter();
        let x = barycenter[0];
        let y = barycenter[1];
        let z = barycenter[2];
        let rho = (x * x + y * y).sqrt().max(EPS);
        theta.push(z.atan2(rho - major_radius));
        phi.push(y.atan2(x));
    }

    Ok(TorusCellGeometry {
        major_radius,
        minor_radius,
        theta,
        phi,
    })
}

fn build_reconstructed_component_operator(
    topology: &Complex,
    coords: &MeshCoords,
    cell_geometry: &TorusCellGeometry,
    toroidal_component: bool,
) -> Result<SparseRowLinearOperator, String> {
    let topo_dim = topology.dim();
    let cell_skeleton = topology.skeleton(topo_dim);
    let bary_local = barycenter_local(topo_dim);
    let mut rows = Vec::with_capacity(cell_skeleton.len());

    for (cell_index, cell) in cell_skeleton.handle_iter().enumerate() {
        let theta = cell_geometry.theta[cell_index];
        let phi = cell_geometry.phi[cell_index];
        let direction = if toroidal_component {
            [-phi.sin(), phi.cos(), 0.0]
        } else {
            [
                -theta.sin() * phi.cos(),
                -theta.sin() * phi.sin(),
                theta.cos(),
            ]
        };

        let cell_coords = cell.coord_simplex(coords);
        let jacobian_pinv = cell_coords.inv_linear_transform();
        let mut row = Vec::new();
        for dof_simp in cell.mesh_subsimps(1) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let local_value = lsf.at_point(&bary_local).into_grade1();
            let ambient_value = if topo_dim == coords.dim() {
                local_value
            } else {
                jacobian_pinv.transpose() * local_value
            };
            let coefficient = ambient_value[0] * direction[0]
                + ambient_value[1] * direction[1]
                + ambient_value[2] * direction[2];
            if coefficient.abs() > EPS {
                row.push((dof_simp.kidx(), coefficient));
            }
        }
        rows.push(row);
    }

    SparseRowLinearOperator::new(topology.skeleton(1).len(), rows).map_err(|err| err.to_string())
}

fn build_embedded_component_operator(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowLinearOperator, String> {
    if component_index >= coords.dim() {
        return Err(format!(
            "ambient component index {} is out of range for coordinate dimension {}",
            component_index,
            coords.dim()
        ));
    }

    let topo_dim = topology.dim();
    let cell_skeleton = topology.skeleton(topo_dim);
    let bary_local = barycenter_local(topo_dim);
    let mut rows = Vec::with_capacity(cell_skeleton.len());

    for cell in cell_skeleton.handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let jacobian_pinv = cell_coords.inv_linear_transform();
        let mut row = Vec::new();
        for dof_simp in cell.mesh_subsimps(1) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let local_value = lsf.at_point(&bary_local).into_grade1();
            let ambient_value = if topo_dim == coords.dim() {
                local_value
            } else {
                jacobian_pinv.transpose() * local_value
            };
            let coefficient = ambient_value[component_index];
            if coefficient.abs() > EPS {
                row.push((dof_simp.kidx(), coefficient));
            }
        }
        rows.push(row);
    }

    SparseRowLinearOperator::new(topology.skeleton(1).len(), rows).map_err(|err| err.to_string())
}

fn build_gaussian_smoothing_operator(
    cell_geometry: &TorusCellGeometry,
    bandwidth: f64,
    cutoff: f64,
) -> Result<SparseRowLinearOperator, String> {
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err("smoothing bandwidth must be finite and positive".to_string());
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err("smoothing cutoff must be finite and positive".to_string());
    }

    let mut rows = Vec::with_capacity(cell_geometry.theta.len());
    for row_index in 0..cell_geometry.theta.len() {
        let mut row = Vec::new();
        let mut weight_sum = 0.0;
        for col_index in 0..cell_geometry.theta.len() {
            let distance = intrinsic_torus_distance(
                cell_geometry.major_radius,
                cell_geometry.minor_radius,
                cell_geometry.theta[row_index],
                cell_geometry.phi[row_index],
                cell_geometry.theta[col_index],
                cell_geometry.phi[col_index],
            );
            if distance > cutoff {
                continue;
            }
            let weight = (-0.5 * (distance / bandwidth).powi(2)).exp();
            if weight <= EPS {
                continue;
            }
            row.push((col_index, weight));
            weight_sum += weight;
        }

        if weight_sum <= EPS {
            row.push((row_index, 1.0));
            weight_sum = 1.0;
        }
        for (_, weight) in row.iter_mut() {
            *weight /= weight_sum;
        }
        rows.push(row);
    }

    SparseRowLinearOperator::new(cell_geometry.theta.len(), rows).map_err(|err| err.to_string())
}

fn build_local_circulation_operator(
    topology: &Complex,
    edge_count: usize,
) -> Result<SparseRowLinearOperator, String> {
    let boundary = topology.boundary_operator(topology.dim());
    let mut rows = vec![Vec::new(); topology.cells().len()];
    for (edge_index, cell_index, value) in boundary.triplet_iter() {
        if edge_index >= edge_count || cell_index >= rows.len() {
            return Err("invalid face-boundary incidence entry".to_string());
        }
        if value.abs() > EPS {
            rows[cell_index].push((edge_index, *value));
        }
    }
    SparseRowLinearOperator::new(edge_count, rows).map_err(|err| err.to_string())
}

fn mass_orthonormalize_harmonic_basis(
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<FeecMatrix, String> {
    if harmonic_basis.ncols() == 0 {
        return Err("harmonic basis must contain at least one column".to_string());
    }

    let mut columns = Vec::with_capacity(harmonic_basis.ncols());
    for j in 0..harmonic_basis.ncols() {
        let mut column = harmonic_basis.column(j).into_owned();
        for previous in &columns {
            let coeff = mass_inner_product(previous, &column, mass_u);
            column -= previous * coeff;
        }

        let norm_sq = mass_inner_product(&column, &column, mass_u);
        if !norm_sq.is_finite() || norm_sq <= EPS {
            return Err(format!(
                "harmonic basis column {j} became singular during orthonormalization"
            ));
        }
        column /= norm_sq.sqrt();
        columns.push(column);
    }

    Ok(FeecMatrix::from_columns(&columns))
}

fn harmonic_coefficients(
    field: &FeecVector,
    harmonic_basis_orthonormal: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<[f64; 2], String> {
    if harmonic_basis_orthonormal.ncols() != 2 {
        return Err(format!(
            "expected exactly two harmonic basis vectors on the torus, found {}",
            harmonic_basis_orthonormal.ncols()
        ));
    }

    Ok([
        mass_inner_product(
            &harmonic_basis_orthonormal.column(0).into_owned(),
            field,
            mass_u,
        ),
        mass_inner_product(
            &harmonic_basis_orthonormal.column(1).into_owned(),
            field,
            mass_u,
        ),
    ])
}

fn remove_harmonic_content(
    field: &FeecVector,
    harmonic_basis_orthonormal: &FeecMatrix,
    mass_u: &FeecCsr,
) -> FeecVector {
    let mut harmonic_free = field.clone();
    for j in 0..harmonic_basis_orthonormal.ncols() {
        let basis_col = harmonic_basis_orthonormal.column(j).into_owned();
        let coeff = mass_inner_product(&basis_col, field, mass_u);
        harmonic_free -= basis_col.scale(coeff);
    }
    harmonic_free
}

fn mass_inner_product(lhs: &FeecVector, rhs: &FeecVector, mass_u: &FeecCsr) -> f64 {
    let weighted_rhs = mass_u * rhs;
    lhs.dot(&weighted_rhs)
}

pub(crate) fn build_variance_field_set(
    prior: &HutchinsonVarianceEstimates,
    posterior: &HutchinsonVarianceEstimates,
) -> Torus1FormVarianceFieldSet {
    let prior = gmrf_vec_to_feec(&prior.unconstrained);
    let posterior = gmrf_vec_to_feec(&posterior.unconstrained);
    let ratio = ratio_vector(&posterior, &prior);
    Torus1FormVarianceFieldSet {
        prior,
        posterior,
        ratio,
    }
}

pub(crate) fn build_component_field_set(
    prior: &Torus1FormVarianceComponentEstimates,
    posterior: &Torus1FormVarianceComponentEstimates,
) -> Torus1FormVarianceComponentFields {
    let toroidal = build_variance_field_set(&prior.toroidal, &posterior.toroidal);
    let poloidal = build_variance_field_set(&prior.poloidal, &posterior.poloidal);
    let trace_prior = &toroidal.prior + &poloidal.prior;
    let trace_posterior = &toroidal.posterior + &poloidal.posterior;
    let trace_ratio = ratio_vector(&trace_posterior, &trace_prior);
    Torus1FormVarianceComponentFields {
        toroidal,
        poloidal,
        trace: Torus1FormVarianceFieldSet {
            prior: trace_prior,
            posterior: trace_posterior,
            ratio: trace_ratio,
        },
    }
}

pub(crate) fn build_ambient_field_set(
    prior: &Torus1FormAmbientVarianceEstimates,
    posterior: &Torus1FormAmbientVarianceEstimates,
) -> Torus1FormAmbientVarianceFields {
    let x = build_variance_field_set(&prior.x, &posterior.x);
    let y = build_variance_field_set(&prior.y, &posterior.y);
    let z = build_variance_field_set(&prior.z, &posterior.z);
    let trace_prior = &x.prior + &y.prior + &z.prior;
    let trace_posterior = &x.posterior + &y.posterior + &z.posterior;
    let trace_ratio = ratio_vector(&trace_posterior, &trace_prior);

    Torus1FormAmbientVarianceFields {
        x,
        y,
        z,
        trace: Torus1FormVarianceFieldSet {
            prior: trace_prior,
            posterior: trace_posterior,
            ratio: trace_ratio,
        },
    }
}

fn ratio_vector(numerator: &FeecVector, denominator: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(
        numerator.len(),
        (0..numerator.len()).map(|i| safe_ratio(numerator[i], denominator[i])),
    )
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= EPS {
        0.0
    } else {
        numerator / denominator
    }
}

fn absolute_difference(lhs: &FeecVector, rhs: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(lhs.len(), (0..lhs.len()).map(|i| (lhs[i] - rhs[i]).abs()))
}

fn intrinsic_torus_distance(
    major_radius: f64,
    minor_radius: f64,
    theta: f64,
    phi: f64,
    theta_ref: f64,
    phi_ref: f64,
) -> f64 {
    let delta_theta = wrap_angle_difference(theta, theta_ref);
    let delta_phi = wrap_angle_difference(phi, phi_ref);
    let phi_scale = major_radius + minor_radius * ((theta + theta_ref) * 0.5).cos();
    ((minor_radius * delta_theta).powi(2) + (phi_scale * delta_phi).powi(2)).sqrt()
}

fn wrap_angle_difference(angle: f64, reference: f64) -> f64 {
    let mut delta = angle - reference;
    while delta <= -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    delta
}

fn mean(values: &FeecVector) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_value(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn write_overall_summary(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("summary.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Torus 1-form Matérn PDE conditioning")?;
    writeln!(writer, "major_radius={}", result.major_radius)?;
    writeln!(writer, "minor_radius={}", result.minor_radius)?;
    writeln!(writer, "num_variance_probes={}", result.num_variance_probes)?;
    writeln!(
        writer,
        "variance_batch_count={}",
        result.variance_batch_count
    )?;
    writeln!(writer, "rng_seed={}", result.rng_seed)?;
    writeln!(
        writer,
        "surface_vector_variance_mode={}",
        result.surface_vector_variance_mode.as_str()
    )?;
    writeln!(writer, "effective_range={}", result.effective_range)?;
    writeln!(
        writer,
        "harmonic_coefficients_truth={},{}",
        result.harmonic_coefficients_truth[0], result.harmonic_coefficients_truth[1]
    )?;
    writeln!(
        writer,
        "harmonic_coefficients_posterior_mean={},{}",
        result.harmonic_coefficients_posterior_mean[0],
        result.harmonic_coefficients_posterior_mean[1]
    )?;
    writeln!(
        writer,
        "posterior_deterministic_l2_error={}",
        result.posterior_deterministic_l2_error
    )?;
    writeln!(writer, "l2_error={}", result.l2_error)?;
    writeln!(writer, "hd_error={}", result.hd_error)?;
    writeln!(writer, "truth_residual_norm={}", result.truth_residual_norm)?;
    writeln!(
        writer,
        "truth_relative_residual_norm={}",
        result.truth_relative_residual_norm
    )?;
    writeln!(
        writer,
        "posterior_residual_norm={}",
        result.posterior_residual_norm
    )?;
    writeln!(
        writer,
        "posterior_relative_residual_norm={}",
        result.posterior_relative_residual_norm
    )?;
    writeln!(
        writer,
        "edge_mean_abs_error={}",
        mean(&result.absolute_mean_error)
    )?;
    writeln!(
        writer,
        "edge_max_abs_error={}",
        max_value(&result.absolute_mean_error)
    )?;
    writeln!(
        writer,
        "edge_variance_ratio_mean={}",
        mean(&result.variance_ratio)
    )?;
    writeln!(
        writer,
        "harmonic_free_edge_variance_ratio_mean={}",
        mean(&result.harmonic_free_variance_ratio)
    )?;
    writeln!(
        writer,
        "surface_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.surface_vector.trace.ratio)
    )?;
    writeln!(
        writer,
        "reconstructed_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.reconstructed.trace.ratio)
    )?;
    writeln!(
        writer,
        "smoothed_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.smoothed.trace.ratio)
    )?;
    writeln!(
        writer,
        "circulation_variance_ratio_mean={}",
        mean(&result.variance_fields.circulation.ratio)
    )?;
    Ok(())
}

fn write_edge_fields_vtu(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let truth = Cochain::new(1, result.truth.clone());
    let rhs = Cochain::new(1, result.rhs.clone());
    let posterior_mean = Cochain::new(1, result.posterior_mean.clone());
    let posterior_rhs = Cochain::new(1, result.posterior_rhs.clone());
    let pde_residual = Cochain::new(1, result.pde_residual.clone());
    let absolute_mean_error = Cochain::new(1, result.absolute_mean_error.clone());
    let prior_variance = Cochain::new(1, result.prior_variance.clone());
    let posterior_variance = Cochain::new(1, result.posterior_variance.clone());
    let variance_reduction = Cochain::new(1, result.variance_reduction.clone());
    let variance_ratio = Cochain::new(1, result.variance_ratio.clone());
    let harmonic_free_truth = Cochain::new(1, result.harmonic_free_truth.clone());
    let harmonic_free_posterior_mean = Cochain::new(1, result.harmonic_free_posterior_mean.clone());
    let harmonic_free_absolute_mean_error =
        Cochain::new(1, result.harmonic_free_absolute_mean_error.clone());
    let harmonic_free_prior_variance = Cochain::new(1, result.harmonic_free_prior_variance.clone());
    let harmonic_free_posterior_variance =
        Cochain::new(1, result.harmonic_free_posterior_variance.clone());
    let harmonic_free_variance_reduction =
        Cochain::new(1, result.harmonic_free_variance_reduction.clone());
    let harmonic_free_variance_ratio = Cochain::new(1, result.harmonic_free_variance_ratio.clone());
    let edge_theta = Cochain::new(1, result.edge_theta.clone());
    let edge_phi = Cochain::new(1, result.edge_phi.clone());
    let toroidal_alignment_sq = Cochain::new(1, result.toroidal_alignment_sq.clone());

    visual_output::write_1cochain_fields(
        out_dir.join("fields.vtu"),
        &result.coords,
        &result.topology,
        &[
            ("truth", &truth),
            ("rhs", &rhs),
            ("posterior_mean", &posterior_mean),
            ("posterior_rhs", &posterior_rhs),
            ("pde_residual", &pde_residual),
            ("absolute_mean_error", &absolute_mean_error),
            ("prior_variance", &prior_variance),
            ("posterior_variance", &posterior_variance),
            ("variance_reduction", &variance_reduction),
            ("variance_ratio", &variance_ratio),
            ("harmonic_free_truth", &harmonic_free_truth),
            (
                "harmonic_free_posterior_mean",
                &harmonic_free_posterior_mean,
            ),
            (
                "harmonic_free_absolute_mean_error",
                &harmonic_free_absolute_mean_error,
            ),
            (
                "harmonic_free_prior_variance",
                &harmonic_free_prior_variance,
            ),
            (
                "harmonic_free_posterior_variance",
                &harmonic_free_posterior_variance,
            ),
            (
                "harmonic_free_variance_reduction",
                &harmonic_free_variance_reduction,
            ),
            (
                "harmonic_free_variance_ratio",
                &harmonic_free_variance_ratio,
            ),
            ("edge_theta", &edge_theta),
            ("edge_phi", &edge_phi),
            ("toroidal_alignment_sq", &toroidal_alignment_sq),
        ],
    )?;

    visual_output::write_1form_vector_proxy_fields(
        out_dir.join("posterior_mean_vector.vtu"),
        &result.coords,
        &result.topology,
        "posterior_mean_vector",
        &posterior_mean,
        &[
            ("truth", &truth),
            ("absolute_mean_error", &absolute_mean_error),
            ("posterior_variance", &posterior_variance),
            ("pde_residual", &pde_residual),
        ],
    )?;

    Ok(())
}

fn write_surface_vector_vtu(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let posterior_mean = Cochain::new(1, result.posterior_mean.clone());
    let posterior_mean_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &posterior_mean)?;
    let posterior_mean_magnitude = vector_magnitudes(&posterior_mean_vectors);
    let surface = &result.variance_fields.surface_vector;
    let posterior_variance_vectors = ambient_variance_vectors(surface, false);
    let prior_variance_vectors = ambient_variance_vectors(surface, true);
    let posterior_marginal_std = surface.trace.posterior.map(|value| value.max(0.0).sqrt());

    visual_output::write_top_cell_fields(
        out_dir.join("posterior_mean_surface_vector.vtu"),
        &result.coords,
        &result.topology,
        &[
            (
                "posterior_mean_surface_vector",
                posterior_mean_vectors.as_slice(),
            ),
            (
                "posterior_directional_variance",
                posterior_variance_vectors.as_slice(),
            ),
            (
                "prior_directional_variance",
                prior_variance_vectors.as_slice(),
            ),
        ],
        &[
            ("magnitude", posterior_mean_magnitude.as_slice()),
            ("marginal_variance", surface.trace.posterior.as_slice()),
            ("marginal_std", posterior_marginal_std.as_slice()),
            ("prior_marginal_variance", surface.trace.prior.as_slice()),
            ("marginal_variance_ratio", surface.trace.ratio.as_slice()),
        ],
    )?;

    Ok(())
}

fn write_edge_csv(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("edge_fields.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "edge_index,theta,phi,toroidal_alignment_sq,truth,rhs,posterior_mean,posterior_rhs,pde_residual,absolute_mean_error,prior_variance,posterior_variance,variance_reduction,variance_ratio,harmonic_free_truth,harmonic_free_posterior_mean,harmonic_free_absolute_mean_error,harmonic_free_prior_variance,harmonic_free_posterior_variance,harmonic_free_variance_reduction,harmonic_free_variance_ratio"
    )?;

    for edge_index in 0..result.truth.len() {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            edge_index,
            result.edge_theta[edge_index],
            result.edge_phi[edge_index],
            result.toroidal_alignment_sq[edge_index],
            result.truth[edge_index],
            result.rhs[edge_index],
            result.posterior_mean[edge_index],
            result.posterior_rhs[edge_index],
            result.pde_residual[edge_index],
            result.absolute_mean_error[edge_index],
            result.prior_variance[edge_index],
            result.posterior_variance[edge_index],
            result.variance_reduction[edge_index],
            result.variance_ratio[edge_index],
            result.harmonic_free_truth[edge_index],
            result.harmonic_free_posterior_mean[edge_index],
            result.harmonic_free_absolute_mean_error[edge_index],
            result.harmonic_free_prior_variance[edge_index],
            result.harmonic_free_posterior_variance[edge_index],
            result.harmonic_free_variance_reduction[edge_index],
            result.harmonic_free_variance_ratio[edge_index],
        )?;
    }

    Ok(())
}

fn write_variance_field_vtus(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    visual_output::write_top_cell_scalar_fields(
        out_dir.join("reconstructed_component_variance.vtu"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_var_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .ratio
                    .as_slice(),
            ),
            (
                "prior_var_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .ratio
                    .as_slice(),
            ),
            (
                "trace_prior",
                result.variance_fields.reconstructed.trace.prior.as_slice(),
            ),
            (
                "trace_post",
                result
                    .variance_fields
                    .reconstructed
                    .trace
                    .posterior
                    .as_slice(),
            ),
            (
                "trace_ratio",
                result.variance_fields.reconstructed.trace.ratio.as_slice(),
            ),
        ],
    )?;
    visual_output::write_top_cell_scalar_fields(
        out_dir.join("smoothed_component_variance.vtu"),
        &result.coords,
        &result.topology,
        &[
            (
                "smoothed_prior_toroidal",
                result.variance_fields.smoothed.toroidal.prior.as_slice(),
            ),
            (
                "smoothed_post_toroidal",
                result
                    .variance_fields
                    .smoothed
                    .toroidal
                    .posterior
                    .as_slice(),
            ),
            (
                "smoothed_ratio_toroidal",
                result.variance_fields.smoothed.toroidal.ratio.as_slice(),
            ),
            (
                "smoothed_prior_poloidal",
                result.variance_fields.smoothed.poloidal.prior.as_slice(),
            ),
            (
                "smoothed_post_poloidal",
                result
                    .variance_fields
                    .smoothed
                    .poloidal
                    .posterior
                    .as_slice(),
            ),
            (
                "smoothed_ratio_poloidal",
                result.variance_fields.smoothed.poloidal.ratio.as_slice(),
            ),
            (
                "smoothed_trace_prior",
                result.variance_fields.smoothed.trace.prior.as_slice(),
            ),
            (
                "smoothed_trace_post",
                result.variance_fields.smoothed.trace.posterior.as_slice(),
            ),
            (
                "smoothed_trace_ratio",
                result.variance_fields.smoothed.trace.ratio.as_slice(),
            ),
        ],
    )?;
    visual_output::write_top_cell_scalar_fields(
        out_dir.join("circulation_variance.vtu"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_circulation",
                result.variance_fields.circulation.prior.as_slice(),
            ),
            (
                "post_circulation",
                result.variance_fields.circulation.posterior.as_slice(),
            ),
            (
                "ratio_circulation",
                result.variance_fields.circulation.ratio.as_slice(),
            ),
        ],
    )?;
    Ok(())
}

fn ambient_variance_vectors(
    surface: &Torus1FormAmbientVarianceFields,
    use_prior: bool,
) -> Vec<[f64; 3]> {
    let x = if use_prior {
        &surface.x.prior
    } else {
        &surface.x.posterior
    };
    let y = if use_prior {
        &surface.y.prior
    } else {
        &surface.y.posterior
    };
    let z = if use_prior {
        &surface.z.prior
    } else {
        &surface.z.posterior
    };
    (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect()
}

fn vector_magnitudes(vectors: &[[f64; 3]]) -> Vec<f64> {
    vectors
        .iter()
        .map(|[x, y, z]| (x * x + y * y + z * z).sqrt())
        .collect()
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeConditioningKappa0Config {
    pub mesh_path: PathBuf,
    pub tau: f64,
    pub noise_variance: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_variance_probes: usize,
    pub variance_batch_count: usize,
    pub rng_seed: u64,
}

impl Default for Torus1FormPdeConditioningKappa0Config {
    fn default() -> Self {
        Self {
            mesh_path: default_torus_shell_resolution_1_mesh_path(),
            tau: 1.0,
            noise_variance: 1e-8,
            surface_vector_variance_mode: SurfaceVectorVarianceMode::HutchinsonStabilized,
            num_variance_probes: DEFAULT_NUM_VARIANCE_PROBES,
            variance_batch_count: DEFAULT_VARIANCE_BATCH_COUNT,
            rng_seed: DEFAULT_RNG_SEED,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeConditioningKappa0VarianceFields {
    pub reconstructed: Torus1FormVarianceComponentFields,
    pub surface_vector: Torus1FormAmbientVarianceFields,
    pub circulation: Torus1FormVarianceFieldSet,
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeConditioningKappa0Result {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub edge_theta: FeecVector,
    pub edge_phi: FeecVector,
    pub toroidal_alignment_sq: FeecVector,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_variance_probes: usize,
    pub variance_batch_count: usize,
    pub rng_seed: u64,
    pub truth: FeecVector,
    pub rhs: FeecVector,
    pub posterior_mean: FeecVector,
    pub posterior_rhs: FeecVector,
    pub pde_residual: FeecVector,
    pub absolute_mean_error: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub variance_reduction: FeecVector,
    pub variance_ratio: FeecVector,
    pub harmonic_coefficients_truth: [f64; 2],
    pub harmonic_coefficients_posterior_mean: [f64; 2],
    pub posterior_deterministic_l2_error: f64,
    pub truth_residual_norm: f64,
    pub truth_relative_residual_norm: f64,
    pub posterior_residual_norm: f64,
    pub posterior_relative_residual_norm: f64,
    pub variance_fields: Torus1FormPdeConditioningKappa0VarianceFields,
}

pub fn run_torus_1form_pde_conditioning_kappa0(
    config: &Torus1FormPdeConditioningKappa0Config,
) -> Result<Torus1FormPdeConditioningKappa0Result, Box<dyn Error>> {
    validate_kappa0_pde_config(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let edge_geometry = build_torus_edge_geometry(&topology, &coords)?;
    let cell_geometry = build_torus_cell_geometry(
        &topology,
        &coords,
        edge_geometry.major_radius,
        edge_geometry.minor_radius,
    )
    .map_err(invalid_data)?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let system_matrix = build_matern_system_matrix_1form(&hodge, 0.0);

    let harmonic_basis =
        build_analytic_torus_harmonic_basis(&topology, &coords, &metric).map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis(&harmonic_basis, &hodge.mass_u).map_err(invalid_data)?;
    let harmonic_constraints =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u)
            .map_err(invalid_data)?;

    let (u_exact, _dif_solution_exact) = build_torus_reference_fields();
    let truth_full = cochain_projection(&u_exact, &topology, &coords, None).coeffs;
    let truth = remove_harmonic_content(&truth_full, &harmonic_basis_orthonormal, &hodge.mass_u);
    let rhs = &system_matrix * &truth;

    let toroidal_operator =
        build_reconstructed_component_operator(&topology, &coords, &cell_geometry, true)
            .map_err(invalid_data)?;
    let poloidal_operator =
        build_reconstructed_component_operator(&topology, &coords, &cell_geometry, false)
            .map_err(invalid_data)?;
    let surface_x_operator =
        build_embedded_component_operator(&topology, &coords, 0).map_err(invalid_data)?;
    let surface_y_operator =
        build_embedded_component_operator(&topology, &coords, 1).map_err(invalid_data)?;
    let surface_z_operator =
        build_embedded_component_operator(&topology, &coords, 2).map_err(invalid_data)?;
    let reconstructed_stacked_operator =
        SparseRowLinearOperator::stack(&[&toroidal_operator, &poloidal_operator])
            .map_err(|err| invalid_data(err.to_string()))?;
    let surface_vector_stacked_operator = SparseRowLinearOperator::stack(&[
        &surface_x_operator,
        &surface_y_operator,
        &surface_z_operator,
    ])
    .map_err(|err| invalid_data(err.to_string()))?;
    let circulation_operator =
        build_local_circulation_operator(&topology, hodge.mass_u.nrows()).map_err(invalid_data)?;

    let prior_precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 0.0,
            tau: config.tau,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
        },
    );
    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let observation_matrix = feec_csr_to_gmrf(&system_matrix);
    let observations = feec_vec_to_gmrf(&rhs);
    let (posterior_precision, information) = apply_gaussian_observations(
        &q_prior,
        &observation_matrix,
        &observations,
        None,
        config.noise_variance,
    );

    let identity_operator = SparseRowLinearOperator::identity(hodge.mass_u.nrows());
    let prior_solver = ConstrainedPrecisionSolver::new(&q_prior, &harmonic_constraints)?;
    let prior_latent_variances =
        kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
            &prior_solver,
            &identity_operator,
            config.num_variance_probes,
            config.variance_batch_count,
            config.rng_seed.wrapping_add(0x0800),
        )?);
    let cell_count = cell_geometry.theta.len();
    let reconstructed_prior = split_component_estimates(
        kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
            &prior_solver,
            &reconstructed_stacked_operator,
            config.num_variance_probes,
            config.variance_batch_count,
            config.rng_seed.wrapping_add(0x1000),
        )?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let surface_vector_prior_estimates = match config.surface_vector_variance_mode {
        SurfaceVectorVarianceMode::Exact => kappa0_estimates(exact_kappa0_transformed_variances(
            &prior_solver,
            &surface_vector_stacked_operator,
        )?),
        SurfaceVectorVarianceMode::Hutchinson | SurfaceVectorVarianceMode::HutchinsonStabilized => {
            kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
                &prior_solver,
                &surface_vector_stacked_operator,
                config.num_variance_probes,
                config.variance_batch_count,
                config.rng_seed.wrapping_add(0x1800),
            )?)
        }
    };
    let surface_vector_prior = split_ambient_estimates(surface_vector_prior_estimates, cell_count)
        .map_err(invalid_data)?;
    let circulation_prior = kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
        &prior_solver,
        &circulation_operator,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed.wrapping_add(0x3000),
    )?);

    let posterior_solver =
        ConstrainedPrecisionSolver::new(&posterior_precision, &harmonic_constraints)?;
    let posterior_latent_variances =
        kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
            &posterior_solver,
            &identity_operator,
            config.num_variance_probes,
            config.variance_batch_count,
            config.rng_seed.wrapping_add(0x2800),
        )?);
    let reconstructed_posterior = split_component_estimates(
        kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
            &posterior_solver,
            &reconstructed_stacked_operator,
            config.num_variance_probes,
            config.variance_batch_count,
            config.rng_seed.wrapping_add(0x1000),
        )?),
        cell_count,
    )
    .map_err(invalid_data)?;
    let surface_vector_posterior_estimates = match config.surface_vector_variance_mode {
        SurfaceVectorVarianceMode::Exact => kappa0_estimates(exact_kappa0_transformed_variances(
            &posterior_solver,
            &surface_vector_stacked_operator,
        )?),
        SurfaceVectorVarianceMode::Hutchinson | SurfaceVectorVarianceMode::HutchinsonStabilized => {
            kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
                &posterior_solver,
                &surface_vector_stacked_operator,
                config.num_variance_probes,
                config.variance_batch_count,
                config.rng_seed.wrapping_add(0x1800),
            )?)
        }
    };
    let surface_vector_posterior =
        if config.surface_vector_variance_mode == SurfaceVectorVarianceMode::HutchinsonStabilized {
            clip_hutchinson_posterior_to_prior(
                &surface_vector_prior,
                &split_ambient_estimates(surface_vector_posterior_estimates, cell_count)
                    .map_err(invalid_data)?,
            )
        } else {
            split_ambient_estimates(surface_vector_posterior_estimates, cell_count)
                .map_err(invalid_data)?
        };
    let circulation_posterior = kappa0_estimates(estimate_kappa0_transformed_hutchinson_variances(
        &posterior_solver,
        &circulation_operator,
        config.num_variance_probes,
        config.variance_batch_count,
        config.rng_seed.wrapping_add(0x3000),
    )?);

    let posterior_mean = gmrf_vec_to_feec(&posterior_solver.solve_mean(&information)?);

    let posterior_rhs = &system_matrix * &posterior_mean;
    let pde_residual = &posterior_rhs - &rhs;
    let absolute_mean_error = absolute_difference(&posterior_mean, &truth);
    let prior_variance = gmrf_vec_to_feec(&prior_latent_variances.harmonic_free);
    let posterior_variance = gmrf_vec_to_feec(&posterior_latent_variances.harmonic_free);
    let variance_reduction = &prior_variance - &posterior_variance;
    let variance_ratio = ratio_vector(&posterior_variance, &prior_variance);
    let harmonic_coefficients_truth =
        harmonic_coefficients(&truth, &harmonic_basis_orthonormal, &hodge.mass_u)
            .map_err(invalid_data)?;
    let harmonic_coefficients_posterior_mean =
        harmonic_coefficients(&posterior_mean, &harmonic_basis_orthonormal, &hodge.mass_u)
            .map_err(invalid_data)?;
    let posterior_mean_cochain = Cochain::new(1, posterior_mean.clone());
    let deterministic_solution = Cochain::new(1, truth.clone());
    let posterior_deterministic_l2_error = l2_norm(
        &(posterior_mean_cochain - deterministic_solution),
        &topology,
        &metric,
    );
    let rhs_norm = rhs.norm().max(EPS);

    Ok(Torus1FormPdeConditioningKappa0Result {
        topology,
        coords,
        edge_theta: FeecVector::from_vec(edge_geometry.theta.clone()),
        edge_phi: FeecVector::from_vec(edge_geometry.phi.clone()),
        toroidal_alignment_sq: FeecVector::from_vec(edge_geometry.toroidal_alignment_sq.clone()),
        major_radius: edge_geometry.major_radius,
        minor_radius: edge_geometry.minor_radius,
        surface_vector_variance_mode: config.surface_vector_variance_mode,
        num_variance_probes: config.num_variance_probes,
        variance_batch_count: config.variance_batch_count,
        rng_seed: config.rng_seed,
        truth,
        rhs,
        posterior_mean,
        posterior_rhs,
        pde_residual: pde_residual.clone(),
        absolute_mean_error,
        prior_variance,
        posterior_variance,
        variance_reduction,
        variance_ratio,
        harmonic_coefficients_truth,
        harmonic_coefficients_posterior_mean,
        posterior_deterministic_l2_error,
        truth_residual_norm: 0.0,
        truth_relative_residual_norm: 0.0,
        posterior_residual_norm: pde_residual.norm(),
        posterior_relative_residual_norm: pde_residual.norm() / rhs_norm,
        variance_fields: Torus1FormPdeConditioningKappa0VarianceFields {
            reconstructed: build_component_field_set(
                &reconstructed_prior,
                &reconstructed_posterior,
            ),
            surface_vector: build_ambient_field_set(
                &surface_vector_prior,
                &surface_vector_posterior,
            ),
            circulation: build_variance_field_set(&circulation_prior, &circulation_posterior),
        },
    })
}

pub fn write_torus_1form_pde_conditioning_kappa0_outputs(
    result: &Torus1FormPdeConditioningKappa0Result,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    write_kappa0_pde_summary(result, out_dir)?;
    write_kappa0_pde_edge_fields_vtu(result, out_dir)?;
    write_kappa0_pde_edge_csv(result, out_dir)?;
    write_kappa0_pde_surface_vector_vtu(result, out_dir)?;
    write_kappa0_pde_variance_field_vtus(result, out_dir)?;
    crate::torus::kappa0_support::write_surface_vector_stats(
        &result.coords,
        &result.topology,
        &result.posterior_mean,
        &result.truth,
        &result.variance_fields.surface_vector,
        out_dir,
        &out_dir.join("summary.txt"),
    )?;

    Ok(())
}

fn validate_kappa0_pde_config(
    config: &Torus1FormPdeConditioningKappa0Config,
) -> Result<(), Box<dyn Error>> {
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive").into());
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err(invalid_input("noise_variance must be finite and positive").into());
    }
    if config.num_variance_probes == 0 {
        return Err(invalid_input("num_variance_probes must be >= 1").into());
    }
    if config.variance_batch_count == 0 {
        return Err(invalid_input("variance_batch_count must be >= 1").into());
    }
    Ok(())
}

fn kappa0_estimates(constrained: GmrfVector) -> HutchinsonVarianceEstimates {
    HutchinsonVarianceEstimates {
        unconstrained: constrained.clone(),
        harmonic_free: constrained,
    }
}

fn exact_kappa0_transformed_variances(
    solver: &ConstrainedPrecisionSolver,
    operator: &SparseRowLinearOperator,
) -> Result<GmrfVector, GmrfError> {
    solver.exact_transformed_variances(operator)
}

fn estimate_kappa0_transformed_hutchinson_variances(
    solver: &ConstrainedPrecisionSolver,
    operator: &SparseRowLinearOperator,
    num_variance_probes: usize,
    variance_batch_count: usize,
    rng_seed: u64,
) -> Result<GmrfVector, Box<dyn Error>> {
    estimate_batched_transformed_hutchinson_with_solve(
        operator,
        ProbeBatchConfig {
            num_probes: num_variance_probes,
            batch_count: variance_batch_count,
            rng_seed,
        },
        VarianceFloor::PositiveMean { scale: 1e-12 },
        |rhs| solver.solve_covariance_action(rhs),
    )
    .map(|estimate| estimate.estimate)
    .map_err(|err| err.into())
}

fn write_kappa0_pde_summary(
    result: &Torus1FormPdeConditioningKappa0Result,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("summary.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "Torus 1-form Matérn PDE conditioning (kappa=0, harmonic-free)"
    )?;
    writeln!(writer, "major_radius={}", result.major_radius)?;
    writeln!(writer, "minor_radius={}", result.minor_radius)?;
    writeln!(writer, "num_variance_probes={}", result.num_variance_probes)?;
    writeln!(
        writer,
        "variance_batch_count={}",
        result.variance_batch_count
    )?;
    writeln!(writer, "rng_seed={}", result.rng_seed)?;
    writeln!(
        writer,
        "surface_vector_variance_mode={}",
        result.surface_vector_variance_mode.as_str()
    )?;
    writeln!(
        writer,
        "harmonic_coefficients_truth={},{}",
        result.harmonic_coefficients_truth[0], result.harmonic_coefficients_truth[1]
    )?;
    writeln!(
        writer,
        "harmonic_coefficients_posterior_mean={},{}",
        result.harmonic_coefficients_posterior_mean[0],
        result.harmonic_coefficients_posterior_mean[1]
    )?;
    writeln!(
        writer,
        "posterior_deterministic_l2_error={}",
        result.posterior_deterministic_l2_error
    )?;
    writeln!(writer, "truth_residual_norm={}", result.truth_residual_norm)?;
    writeln!(
        writer,
        "truth_relative_residual_norm={}",
        result.truth_relative_residual_norm
    )?;
    writeln!(
        writer,
        "posterior_residual_norm={}",
        result.posterior_residual_norm
    )?;
    writeln!(
        writer,
        "posterior_relative_residual_norm={}",
        result.posterior_relative_residual_norm
    )?;
    writeln!(
        writer,
        "edge_mean_abs_error={}",
        mean(&result.absolute_mean_error)
    )?;
    writeln!(
        writer,
        "edge_max_abs_error={}",
        max_value(&result.absolute_mean_error)
    )?;
    writeln!(
        writer,
        "edge_variance_ratio_mean={}",
        mean(&result.variance_ratio)
    )?;
    writeln!(
        writer,
        "surface_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.surface_vector.trace.ratio)
    )?;
    writeln!(
        writer,
        "reconstructed_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.reconstructed.trace.ratio)
    )?;
    writeln!(
        writer,
        "circulation_variance_ratio_mean={}",
        mean(&result.variance_fields.circulation.ratio)
    )?;
    Ok(())
}

fn write_kappa0_pde_edge_fields_vtu(
    result: &Torus1FormPdeConditioningKappa0Result,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let truth = Cochain::new(1, result.truth.clone());
    let rhs = Cochain::new(1, result.rhs.clone());
    let posterior_mean = Cochain::new(1, result.posterior_mean.clone());
    let posterior_rhs = Cochain::new(1, result.posterior_rhs.clone());
    let pde_residual = Cochain::new(1, result.pde_residual.clone());
    let absolute_mean_error = Cochain::new(1, result.absolute_mean_error.clone());
    let prior_variance = Cochain::new(1, result.prior_variance.clone());
    let posterior_variance = Cochain::new(1, result.posterior_variance.clone());
    let variance_reduction = Cochain::new(1, result.variance_reduction.clone());
    let variance_ratio = Cochain::new(1, result.variance_ratio.clone());
    let edge_theta = Cochain::new(1, result.edge_theta.clone());
    let edge_phi = Cochain::new(1, result.edge_phi.clone());
    let toroidal_alignment_sq = Cochain::new(1, result.toroidal_alignment_sq.clone());

    visual_output::write_1cochain_fields(
        out_dir.join("fields.vtu"),
        &result.coords,
        &result.topology,
        &[
            ("truth", &truth),
            ("rhs", &rhs),
            ("posterior_mean", &posterior_mean),
            ("posterior_rhs", &posterior_rhs),
            ("pde_residual", &pde_residual),
            ("absolute_mean_error", &absolute_mean_error),
            ("prior_variance", &prior_variance),
            ("posterior_variance", &posterior_variance),
            ("variance_reduction", &variance_reduction),
            ("variance_ratio", &variance_ratio),
            ("edge_theta", &edge_theta),
            ("edge_phi", &edge_phi),
            ("toroidal_alignment_sq", &toroidal_alignment_sq),
        ],
    )?;
    visual_output::write_1form_vector_proxy_fields(
        out_dir.join("posterior_mean_vector.vtu"),
        &result.coords,
        &result.topology,
        "posterior_mean_vector",
        &posterior_mean,
        &[
            ("truth", &truth),
            ("absolute_mean_error", &absolute_mean_error),
            ("posterior_variance", &posterior_variance),
            ("pde_residual", &pde_residual),
        ],
    )?;
    Ok(())
}

fn write_kappa0_pde_surface_vector_vtu(
    result: &Torus1FormPdeConditioningKappa0Result,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let posterior_mean = Cochain::new(1, result.posterior_mean.clone());
    let posterior_mean_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &posterior_mean)?;
    let posterior_mean_magnitude = vector_magnitudes(&posterior_mean_vectors);
    let surface = &result.variance_fields.surface_vector;
    let posterior_variance_vectors = ambient_variance_vectors(surface, false);
    let prior_variance_vectors = ambient_variance_vectors(surface, true);
    let posterior_marginal_std = surface.trace.posterior.map(|value| value.max(0.0).sqrt());

    visual_output::write_top_cell_fields(
        out_dir.join("posterior_mean_surface_vector.vtu"),
        &result.coords,
        &result.topology,
        &[
            (
                "posterior_mean_surface_vector",
                posterior_mean_vectors.as_slice(),
            ),
            (
                "posterior_directional_variance",
                posterior_variance_vectors.as_slice(),
            ),
            (
                "prior_directional_variance",
                prior_variance_vectors.as_slice(),
            ),
        ],
        &[
            ("magnitude", posterior_mean_magnitude.as_slice()),
            ("marginal_variance", surface.trace.posterior.as_slice()),
            ("marginal_std", posterior_marginal_std.as_slice()),
            ("prior_marginal_variance", surface.trace.prior.as_slice()),
            ("marginal_variance_ratio", surface.trace.ratio.as_slice()),
        ],
    )?;
    Ok(())
}

fn write_kappa0_pde_edge_csv(
    result: &Torus1FormPdeConditioningKappa0Result,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("edge_fields.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "edge_index,theta,phi,toroidal_alignment_sq,truth,rhs,posterior_mean,posterior_rhs,pde_residual,absolute_mean_error,prior_variance,posterior_variance,variance_reduction,variance_ratio"
    )?;

    for edge_index in 0..result.truth.len() {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            edge_index,
            result.edge_theta[edge_index],
            result.edge_phi[edge_index],
            result.toroidal_alignment_sq[edge_index],
            result.truth[edge_index],
            result.rhs[edge_index],
            result.posterior_mean[edge_index],
            result.posterior_rhs[edge_index],
            result.pde_residual[edge_index],
            result.absolute_mean_error[edge_index],
            result.prior_variance[edge_index],
            result.posterior_variance[edge_index],
            result.variance_reduction[edge_index],
            result.variance_ratio[edge_index],
        )?;
    }

    Ok(())
}

fn write_kappa0_pde_variance_field_vtus(
    result: &Torus1FormPdeConditioningKappa0Result,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    visual_output::write_top_cell_scalar_fields(
        out_dir.join("reconstructed_component_variance.vtu"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_var_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .ratio
                    .as_slice(),
            ),
            (
                "prior_var_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .ratio
                    .as_slice(),
            ),
            (
                "trace_prior",
                result.variance_fields.reconstructed.trace.prior.as_slice(),
            ),
            (
                "trace_post",
                result
                    .variance_fields
                    .reconstructed
                    .trace
                    .posterior
                    .as_slice(),
            ),
            (
                "trace_ratio",
                result.variance_fields.reconstructed.trace.ratio.as_slice(),
            ),
        ],
    )?;
    visual_output::write_top_cell_scalar_fields(
        out_dir.join("circulation_variance.vtu"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_circulation",
                result.variance_fields.circulation.prior.as_slice(),
            ),
            (
                "post_circulation",
                result.variance_fields.circulation.posterior.as_slice(),
            ),
            (
                "ratio_circulation",
                result.variance_fields.circulation.ratio.as_slice(),
            ),
        ],
    )?;
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

pub(crate) fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
