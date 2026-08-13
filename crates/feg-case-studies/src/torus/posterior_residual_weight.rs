//! Posterior convergence as the torus PDE residual precision tends to infinity.
//!
//! This is the maintained implementation behind the submitted residual-weight
//! figure and table. Geometry and manufactured-field assembly stay in the
//! case-study layer; Matérn construction and Gaussian conditioning delegate to
//! their canonical integration and GMRF implementations.

use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::cochain::{cochain_projection, Cochain};
use feg_infer::prior::matern::one_form::{
    build_hodge_laplacian_1form, build_matern_precision_1form, build_matern_system_matrix_1form,
    feec_csr_to_gmrf, HodgeLaplacian1Form, MaternConfig, MaternMassInverse,
};
use feg_infer::sparse::gmrf_vec_to_feec;
use formoniq::fe::{fe_l2_error, l2_norm};
use formoniq::torus_convergence::build_torus_reference_fields;
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::Gmrf;
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const EPS: f64 = 1e-12;
const SUBMITTED_WEIGHTS: [f64; 6] = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12];

#[derive(Debug, Clone)]
pub struct Torus1FormPdeMeshLevel {
    pub resolution: usize,
    pub mesh_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdePosteriorMeanWeightSweepConfig {
    pub mesh_levels: Vec<Torus1FormPdeMeshLevel>,
    pub kappa: f64,
    pub tau: f64,
    pub weights: Vec<f64>,
}

impl Default for Torus1FormPdePosteriorMeanWeightSweepConfig {
    fn default() -> Self {
        Self::thesis_submitted()
    }
}

impl Torus1FormPdePosteriorMeanWeightSweepConfig {
    /// Cheap profile covering weak and tight residual precisions on one mesh.
    pub fn smoke() -> Self {
        Self {
            mesh_levels: vec![Torus1FormPdeMeshLevel {
                resolution: 0,
                mesh_path: default_torus_shell_mesh_path(0),
            }],
            kappa: 4.0,
            tau: 1.0,
            weights: vec![1e2, 1e8],
        }
    }

    /// Immutable configuration used for the submitted thesis figure and table.
    pub fn thesis_submitted() -> Self {
        Self {
            mesh_levels: default_torus_pde_posterior_mean_weight_mesh_levels(),
            kappa: 4.0,
            tau: 1.0,
            weights: SUBMITTED_WEIGHTS.to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Torus1FormPdePosteriorMeanWeightRow {
    pub resolution: usize,
    pub weight: f64,
    pub noise_variance: f64,
    pub edge_dofs: usize,
    pub h: f64,
    pub kappa: f64,
    pub tau: f64,
    pub posterior_deterministic_l2_error: f64,
    pub posterior_deterministic_relative_l2_error: f64,
    pub posterior_continuum_l2_error: f64,
    pub posterior_continuum_relative_l2_error: f64,
    pub deterministic_continuum_l2_error: f64,
    pub deterministic_continuum_relative_l2_error: f64,
    pub posterior_residual_norm: f64,
    pub posterior_relative_residual_norm: f64,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Torus1FormPdePosteriorMeanWeightSummaryRow {
    pub resolution: usize,
    pub edge_dofs: usize,
    pub h: f64,
    pub kappa: f64,
    pub tau: f64,
    pub min_weight: f64,
    pub max_weight: f64,
    pub posterior_deterministic_relative_l2_slope: f64,
    pub posterior_relative_residual_slope: f64,
    pub high_weight_posterior_deterministic_l2_error: f64,
    pub high_weight_posterior_deterministic_relative_l2_error: f64,
    pub high_weight_posterior_continuum_l2_error: f64,
    pub high_weight_posterior_continuum_relative_l2_error: f64,
    pub deterministic_continuum_l2_error: f64,
    pub deterministic_continuum_relative_l2_error: f64,
    pub high_weight_posterior_residual_norm: f64,
    pub high_weight_posterior_relative_residual_norm: f64,
    pub total_wall_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Torus1FormPdePosteriorMeanWeightSweepResult {
    pub rows: Vec<Torus1FormPdePosteriorMeanWeightRow>,
    pub summaries: Vec<Torus1FormPdePosteriorMeanWeightSummaryRow>,
}

struct PreparedTorus1FormPdeMeanSweepProblem {
    topology: Complex,
    coords: MeshCoords,
    metric: MeshLengths,
    hodge: HodgeLaplacian1Form,
    system_matrix: FeecCsr,
    truth: FeecVector,
    rhs: FeecVector,
    deterministic_l2_norm: f64,
    continuum_l2_norm: f64,
    deterministic_continuum_l2_error: f64,
    deterministic_continuum_relative_l2_error: f64,
}

pub fn default_torus_shell_mesh_path(resolution: usize) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../meshes/torus_shell_resolution_{resolution}.msh"
    ))
}

pub fn default_torus_pde_posterior_mean_weight_mesh_levels() -> Vec<Torus1FormPdeMeshLevel> {
    (0..=3)
        .map(|resolution| Torus1FormPdeMeshLevel {
            resolution,
            mesh_path: default_torus_shell_mesh_path(resolution),
        })
        .collect()
}

pub fn run_torus_1form_pde_posterior_mean_weight_sweep(
    config: &Torus1FormPdePosteriorMeanWeightSweepConfig,
) -> Result<Torus1FormPdePosteriorMeanWeightSweepResult, Box<dyn Error>> {
    validate_config(config)?;
    let mut weights = config.weights.clone();
    weights.sort_by(f64::total_cmp);

    let mut rows = Vec::with_capacity(config.mesh_levels.len() * weights.len());
    for mesh_level in &config.mesh_levels {
        let prepared = prepare_problem(&mesh_level.mesh_path, config.kappa)?;
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
        let observation_operator = feec_csr_to_gmrf(&prepared.system_matrix);
        let observations = feg_infer::prior::matern::one_form::feec_vec_to_gmrf(&prepared.rhs);
        let edge_dofs = prepared.truth.len();
        let h = prepared.metric.mesh_width_max();
        let rhs_norm = prepared.rhs.norm().max(EPS);

        for &weight in &weights {
            let row_start = Instant::now();
            let noise_variance = 1.0 / weight;
            let (posterior_precision, information) = apply_gaussian_observations(
                &q_prior,
                &observation_operator,
                &observations,
                None,
                noise_variance,
            );
            let posterior_factor = posterior_precision.cholesky_sqrt_lower()?;
            let posterior = Gmrf::from_information_and_precision_with_sqrt(
                information,
                posterior_precision,
                posterior_factor,
            )?;
            let posterior_mean = gmrf_vec_to_feec(posterior.mean());
            let posterior_mean_cochain = Cochain::new(1, posterior_mean.clone());
            let deterministic_solution = Cochain::new(1, prepared.truth.clone());
            let posterior_deterministic_l2_error = l2_norm(
                &(posterior_mean_cochain.clone() - deterministic_solution),
                &prepared.topology,
                &prepared.metric,
            );
            let (u_exact, _) = build_torus_reference_fields();
            let posterior_continuum_l2_error = fe_l2_error(
                &posterior_mean_cochain,
                &u_exact,
                &prepared.topology,
                &prepared.coords,
            );
            let posterior_rhs = &prepared.system_matrix * &posterior_mean;
            let posterior_residual_norm = (&posterior_rhs - &prepared.rhs).norm();

            rows.push(Torus1FormPdePosteriorMeanWeightRow {
                resolution: mesh_level.resolution,
                weight,
                noise_variance,
                edge_dofs,
                h,
                kappa: config.kappa,
                tau: config.tau,
                posterior_deterministic_l2_error,
                posterior_deterministic_relative_l2_error: posterior_deterministic_l2_error
                    / prepared.deterministic_l2_norm,
                posterior_continuum_l2_error,
                posterior_continuum_relative_l2_error: posterior_continuum_l2_error
                    / prepared.continuum_l2_norm,
                deterministic_continuum_l2_error: prepared.deterministic_continuum_l2_error,
                deterministic_continuum_relative_l2_error: prepared
                    .deterministic_continuum_relative_l2_error,
                posterior_residual_norm,
                posterior_relative_residual_norm: posterior_residual_norm / rhs_norm,
                wall_seconds: row_start.elapsed().as_secs_f64(),
            });
        }
    }

    let summaries = summarize_rows(&rows);
    Ok(Torus1FormPdePosteriorMeanWeightSweepResult { rows, summaries })
}

pub fn write_torus_1form_pde_posterior_mean_weight_sweep_outputs(
    result: &Torus1FormPdePosteriorMeanWeightSweepResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_detail_csv(result, &out_dir.join("posterior_mean_weight_sweep.csv"))?;
    write_summary_csv(
        result,
        &out_dir.join("posterior_mean_weight_sweep_summary.csv"),
    )?;
    Ok(())
}

fn prepare_problem(
    mesh_path: &Path,
    kappa: f64,
) -> Result<PreparedTorus1FormPdeMeanSweepProblem, Box<dyn Error>> {
    let mesh_bytes = fs::read(mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let system_matrix = build_matern_system_matrix_1form(&hodge, kappa);
    let (u_exact, _) = build_torus_reference_fields();
    let truth_cochain = cochain_projection(&u_exact, &topology, &coords, None);
    let truth = truth_cochain.coeffs.clone();
    let rhs = &system_matrix * &truth;
    let deterministic_l2_norm = l2_norm(&truth_cochain, &topology, &metric).max(EPS);
    let zero_cochain = Cochain::new(1, FeecVector::zeros(truth.len()));
    let continuum_l2_norm = fe_l2_error(&zero_cochain, &u_exact, &topology, &coords).max(EPS);
    let deterministic_continuum_l2_error =
        fe_l2_error(&truth_cochain, &u_exact, &topology, &coords);
    let deterministic_continuum_relative_l2_error =
        deterministic_continuum_l2_error / continuum_l2_norm;
    Ok(PreparedTorus1FormPdeMeanSweepProblem {
        topology,
        coords,
        metric,
        hodge,
        system_matrix,
        truth,
        rhs,
        deterministic_l2_norm,
        continuum_l2_norm,
        deterministic_continuum_l2_error,
        deterministic_continuum_relative_l2_error,
    })
}

fn validate_config(
    config: &Torus1FormPdePosteriorMeanWeightSweepConfig,
) -> Result<(), Box<dyn Error>> {
    if config.mesh_levels.is_empty() {
        return Err(invalid_input("mesh_levels must contain at least one mesh").into());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive").into());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive").into());
    }
    if config.weights.is_empty() {
        return Err(invalid_input("weights must contain at least one value").into());
    }
    for (index, &weight) in config.weights.iter().enumerate() {
        if !weight.is_finite() || weight <= 0.0 || !(1.0 / weight).is_finite() {
            return Err(invalid_input(format!(
                "weights[{index}] must be finite, positive, and have a finite reciprocal"
            ))
            .into());
        }
    }
    Ok(())
}

fn summarize_rows(
    rows: &[Torus1FormPdePosteriorMeanWeightRow],
) -> Vec<Torus1FormPdePosteriorMeanWeightSummaryRow> {
    let mut summaries = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let resolution = rows[start].resolution;
        let end = rows[start..]
            .iter()
            .position(|row| row.resolution != resolution)
            .map(|offset| start + offset)
            .unwrap_or(rows.len());
        let level_rows = &rows[start..end];
        if let (Some(first), Some(last)) = (level_rows.first(), level_rows.last()) {
            summaries.push(Torus1FormPdePosteriorMeanWeightSummaryRow {
                resolution,
                edge_dofs: first.edge_dofs,
                h: first.h,
                kappa: first.kappa,
                tau: first.tau,
                min_weight: first.weight,
                max_weight: last.weight,
                posterior_deterministic_relative_l2_slope: log10_weight_slope(level_rows, |row| {
                    row.posterior_deterministic_relative_l2_error
                }),
                posterior_relative_residual_slope: log10_weight_slope(level_rows, |row| {
                    row.posterior_relative_residual_norm
                }),
                high_weight_posterior_deterministic_l2_error: last.posterior_deterministic_l2_error,
                high_weight_posterior_deterministic_relative_l2_error: last
                    .posterior_deterministic_relative_l2_error,
                high_weight_posterior_continuum_l2_error: last.posterior_continuum_l2_error,
                high_weight_posterior_continuum_relative_l2_error: last
                    .posterior_continuum_relative_l2_error,
                deterministic_continuum_l2_error: last.deterministic_continuum_l2_error,
                deterministic_continuum_relative_l2_error: last
                    .deterministic_continuum_relative_l2_error,
                high_weight_posterior_residual_norm: last.posterior_residual_norm,
                high_weight_posterior_relative_residual_norm: last.posterior_relative_residual_norm,
                total_wall_seconds: level_rows.iter().map(|row| row.wall_seconds).sum(),
            });
        }
        start = end;
    }
    summaries
}

fn log10_weight_slope<F>(rows: &[Torus1FormPdePosteriorMeanWeightRow], value: F) -> f64
where
    F: Fn(&Torus1FormPdePosteriorMeanWeightRow) -> f64,
{
    let pairs = rows
        .iter()
        .filter_map(|row| {
            let y = value(row);
            (row.weight > 0.0 && y > 0.0 && row.weight.is_finite() && y.is_finite())
                .then_some((row.weight.log10(), y.log10()))
        })
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return f64::NAN;
    }
    let x_mean = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let y_mean = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pairs.len() as f64;
    let denominator = pairs
        .iter()
        .map(|(x, _)| (*x - x_mean).powi(2))
        .sum::<f64>();
    if denominator <= 0.0 {
        return f64::NAN;
    }
    pairs
        .iter()
        .map(|(x, y)| (*x - x_mean) * (*y - y_mean))
        .sum::<f64>()
        / denominator
}

fn write_detail_csv(
    result: &Torus1FormPdePosteriorMeanWeightSweepResult,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "resolution,weight,noise_variance,edge_dofs,h,kappa,tau,posterior_deterministic_l2_error,posterior_deterministic_relative_l2_error,posterior_continuum_l2_error,posterior_continuum_relative_l2_error,deterministic_continuum_l2_error,deterministic_continuum_relative_l2_error,posterior_residual_norm,posterior_relative_residual_norm,wall_seconds"
    )?;
    for row in &result.rows {
        writeln!(
            writer,
            "{},{:.12e},{:.12e},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.resolution,
            row.weight,
            row.noise_variance,
            row.edge_dofs,
            row.h,
            row.kappa,
            row.tau,
            row.posterior_deterministic_l2_error,
            row.posterior_deterministic_relative_l2_error,
            row.posterior_continuum_l2_error,
            row.posterior_continuum_relative_l2_error,
            row.deterministic_continuum_l2_error,
            row.deterministic_continuum_relative_l2_error,
            row.posterior_residual_norm,
            row.posterior_relative_residual_norm,
            row.wall_seconds,
        )?;
    }
    Ok(())
}

fn write_summary_csv(
    result: &Torus1FormPdePosteriorMeanWeightSweepResult,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "resolution,edge_dofs,h,kappa,tau,min_weight,max_weight,posterior_deterministic_relative_l2_slope,posterior_relative_residual_slope,high_weight_posterior_deterministic_l2_error,high_weight_posterior_deterministic_relative_l2_error,high_weight_posterior_continuum_l2_error,high_weight_posterior_continuum_relative_l2_error,deterministic_continuum_l2_error,deterministic_continuum_relative_l2_error,high_weight_posterior_residual_norm,high_weight_posterior_relative_residual_norm,total_wall_seconds"
    )?;
    for row in &result.summaries {
        writeln!(
            writer,
            "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.resolution,
            row.edge_dofs,
            row.h,
            row.kappa,
            row.tau,
            row.min_weight,
            row.max_weight,
            row.posterior_deterministic_relative_l2_slope,
            row.posterior_relative_residual_slope,
            row.high_weight_posterior_deterministic_l2_error,
            row.high_weight_posterior_deterministic_relative_l2_error,
            row.high_weight_posterior_continuum_l2_error,
            row.high_weight_posterior_continuum_relative_l2_error,
            row.deterministic_continuum_l2_error,
            row.deterministic_continuum_relative_l2_error,
            row.high_weight_posterior_residual_norm,
            row.high_weight_posterior_relative_residual_norm,
            row.total_wall_seconds,
        )?;
    }
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_preserve_submitted_inputs() {
        let smoke = Torus1FormPdePosteriorMeanWeightSweepConfig::smoke();
        let thesis = Torus1FormPdePosteriorMeanWeightSweepConfig::thesis_submitted();
        assert_eq!(
            smoke
                .mesh_levels
                .iter()
                .map(|level| level.resolution)
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(smoke.weights, vec![1e2, 1e8]);
        assert_eq!(
            thesis
                .mesh_levels
                .iter()
                .map(|level| level.resolution)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(thesis.weights, SUBMITTED_WEIGHTS);
        assert_eq!((thesis.kappa, thesis.tau), (4.0, 1.0));
    }
}
