//! End-to-end de Rham ladder experiment on one 3D handlebody mesh.
//!
//! The reusable FEEC/GMRF machinery lives in [`crate::de_rham`]. This module only
//! fixes the demonstration geometry, grade labels, natural observation supports,
//! and output layout.

use crate::de_rham::{
    betti_numbers, boundary_face_integral_support, build_matern_precision_form, codifferential,
    condition_on_observations, derivative, distance, dot, edge_path_integral_support,
    estimate_marginal_variance, hodge_project, median_positive, nearest_vertex,
    sample_zero_mean_precision, squared_distance, sub, supports_to_operator,
    top_cell_region_integral_support, volume_average_0form_support, FormMassInverse,
    FormMaternConfig, HodgeProjectionConfig, ObservationOperator, VarianceProbeConfig,
    WeightedSimplexSupport,
};
use crate::visual_output;
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::cochain::Cochain;
use manifold::{
    geometry::coord::mesh::MeshCoords, io::gmsh::gmsh2coord_complex, topology::complex::Complex,
};
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

const EXPECTED_BETTI: [usize; 4] = [1, 3, 2, 0];

#[derive(Debug, Clone)]
pub struct DerhamLadderConfig {
    pub mesh_path: PathBuf,
    pub output_dir: PathBuf,
    pub kappa: f64,
    pub observation_noise_variance: f64,
    pub sample_seed: u64,
    pub variance_probe_count: usize,
    pub variance_seed: u64,
    pub hodge_projection_ridge: f64,
}

impl Default for DerhamLadderConfig {
    fn default() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            mesh_path: manifest_dir.join("../../meshes/derham_ladder_handlebody.msh"),
            output_dir: manifest_dir.join("../../out/derham_ladder"),
            kappa: 4.0,
            observation_noise_variance: 1e-8,
            sample_seed: 20240517,
            variance_probe_count: 16,
            variance_seed: 314159,
            hodge_projection_ridge: 1e-9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeshSummary {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub cells: usize,
    pub betti: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct GradeSummary {
    pub grade: usize,
    pub interpretation: &'static str,
    pub dimension: usize,
    pub precision_nnz: usize,
    pub posterior_precision_nnz: usize,
    pub tau: f64,
    pub prior_median_variance: f64,
    pub posterior_median_variance: f64,
    pub observation_count: usize,
    pub max_abs_observation_residual: f64,
    pub hodge_reconstruction_error: f64,
    pub hodge_orthogonality_error: f64,
}

#[derive(Debug, Clone)]
pub struct DerhamLadderResult {
    pub mesh: MeshSummary,
    pub grades: Vec<GradeSummary>,
}

#[derive(Debug, Clone, Copy)]
struct TunnelSpec {
    label: &'static str,
    axis: Axis,
    center: [f64; 3],
    radius: f64,
    loop_radius: f64,
}

#[derive(Debug, Clone, Copy)]
struct CavitySpec {
    label: &'static str,
    center: [f64; 3],
    radius: f64,
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

const TUNNELS: [TunnelSpec; 3] = [
    TunnelSpec {
        label: "x_tunnel",
        axis: Axis::X,
        center: [0.0, -0.65, -0.65],
        radius: 0.18,
        loop_radius: 0.34,
    },
    TunnelSpec {
        label: "y_tunnel",
        axis: Axis::Y,
        center: [0.65, 0.0, 0.0],
        radius: 0.17,
        loop_radius: 0.32,
    },
    TunnelSpec {
        label: "z_tunnel",
        axis: Axis::Z,
        center: [-0.55, 0.65, 0.0],
        radius: 0.16,
        loop_radius: 0.31,
    },
];

const CAVITIES: [CavitySpec; 2] = [
    CavitySpec {
        label: "upper_cavity",
        center: [0.15, 0.0, 0.60],
        radius: 0.18,
    },
    CavitySpec {
        label: "inner_cavity",
        center: [-0.45, -0.25, 0.15],
        radius: 0.17,
    },
];

pub fn run_derham_ladder_experiment(
    config: &DerhamLadderConfig,
) -> Result<DerhamLadderResult, Box<dyn Error>> {
    validate_config(config)?;
    fs::create_dir_all(&config.output_dir)?;

    let mesh_bytes = fs::read(&config.mesh_path).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!(
                "failed to read mesh {}; generate it with `gmsh -3 geometries/derham_ladder_handlebody.geo -format msh41 -o meshes/derham_ladder_handlebody.msh`: {err}",
                config.mesh_path.display()
            ),
        )
    })?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let betti = betti_numbers(&topology);
    if betti.as_slice() != EXPECTED_BETTI {
        return Err(format!(
            "unexpected handlebody Betti numbers: got {:?}, expected {:?}",
            betti, EXPECTED_BETTI
        )
        .into());
    }

    let mesh_summary = MeshSummary {
        vertices: topology.nsimplices(0),
        edges: topology.nsimplices(1),
        faces: topology.nsimplices(2),
        cells: topology.nsimplices(3),
        betti,
    };
    write_mesh_summary(&mesh_summary, &config.output_dir.join("mesh_summary.txt"))?;
    write_figure_layout(&config.output_dir.join("figure_layout.md"))?;

    let mut grade_summaries = Vec::new();
    for grade in 0..=topology.dim() {
        eprintln!("[derham_ladder] running grade k={grade}");
        let grade_summary = run_grade(
            config,
            &topology,
            &coords,
            &metric,
            grade,
            &config.output_dir,
        )?;
        grade_summaries.push(grade_summary);
    }
    write_grade_summary(
        &grade_summaries,
        &config.output_dir.join("grade_summary.csv"),
    )?;

    Ok(DerhamLadderResult {
        mesh: mesh_summary,
        grades: grade_summaries,
    })
}

fn run_grade(
    config: &DerhamLadderConfig,
    topology: &Complex,
    coords: &MeshCoords,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    grade: usize,
    output_dir: &Path,
) -> Result<GradeSummary, Box<dyn Error>> {
    let grade_dir = output_dir.join(format!("k{grade}"));
    fs::create_dir_all(&grade_dir)?;

    let base_config = FormMaternConfig {
        kappa: config.kappa,
        tau: 1.0,
        mass_inverse: FormMassInverse::Diagonal,
    };
    let base_system = build_matern_precision_form(topology, metric, grade, base_config)?;
    let base_variance = estimate_marginal_variance(
        &base_system.precision,
        VarianceProbeConfig {
            probe_count: config.variance_probe_count,
            seed: config.variance_seed + grade as u64,
        },
    )?;
    let base_median_variance = median_positive(&base_variance).unwrap_or(1.0);
    let tau = base_median_variance.sqrt().max(1e-8);
    let system = build_matern_precision_form(
        topology,
        metric,
        grade,
        FormMaternConfig { tau, ..base_config },
    )?;

    let prior_sample =
        sample_zero_mean_precision(&system.precision, config.sample_seed + grade as u64)?;
    let observations = build_grade_observations(topology, coords, grade)?;
    let conditioning = condition_on_observations(
        &system.precision,
        &observations.matrix,
        &prior_sample,
        config.observation_noise_variance,
        VarianceProbeConfig {
            probe_count: config.variance_probe_count,
            seed: config.variance_seed + 101 + grade as u64,
        },
    )?;

    let d_prior = derivative(topology, grade, &prior_sample);
    let delta_prior = codifferential(topology, metric, grade, &prior_sample)?;
    let hodge = hodge_project(
        topology,
        metric,
        grade,
        &prior_sample,
        HodgeProjectionConfig {
            ridge: config.hodge_projection_ridge,
        },
    )?;

    let marker = observation_marker(system.precision.nrows(), &observations.matrix);
    write_grade_outputs(
        &grade_dir,
        coords,
        topology,
        grade,
        &prior_sample,
        &conditioning.posterior_mean,
        &conditioning.posterior_variance,
        &marker,
        &hodge,
        if grade < topology.dim() {
            Some(&d_prior)
        } else {
            None
        },
        if grade > 0 { Some(&delta_prior) } else { None },
    )?;
    write_observations_csv(
        &grade_dir.join("observations.csv"),
        &observations.labels,
        &conditioning.observations,
        &conditioning.posterior_observations,
        &conditioning.observation_residual,
    )?;

    let posterior_median_variance =
        median_positive(&conditioning.posterior_variance).unwrap_or(0.0);
    Ok(GradeSummary {
        grade,
        interpretation: grade_interpretation(grade),
        dimension: system.precision.nrows(),
        precision_nnz: system.precision.nnz(),
        posterior_precision_nnz: conditioning.posterior_precision.nnz(),
        tau,
        prior_median_variance: median_positive(&conditioning.prior_variance).unwrap_or(0.0),
        posterior_median_variance,
        observation_count: observations.matrix.nrows(),
        max_abs_observation_residual: max_abs(&conditioning.observation_residual),
        hodge_reconstruction_error: hodge.reconstruction_error,
        hodge_orthogonality_error: hodge.orthogonality_error,
    })
}

fn build_grade_observations(
    topology: &Complex,
    coords: &MeshCoords,
    grade: usize,
) -> Result<ObservationOperator, Box<dyn Error>> {
    let dimension = topology.nsimplices(grade);
    let supports = match grade {
        0 => build_0form_observations(topology, coords)?,
        1 => build_1form_observations(topology, coords)?,
        2 => build_2form_observations(topology, coords)?,
        3 => build_3form_observations(topology, coords)?,
        _ => return Err(format!("unsupported grade {grade}").into()),
    };
    Ok(supports_to_operator(dimension, &supports))
}

fn build_0form_observations(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<WeightedSimplexSupport>, Box<dyn Error>> {
    let points = [
        ("point_west", [-0.82, 0.72, 0.72]),
        ("point_east", [0.82, -0.82, 0.68]),
        ("point_core", [0.18, 0.20, -0.18]),
    ];
    let mut supports = points
        .iter()
        .map(|(label, point)| {
            WeightedSimplexSupport::new(*label, vec![(nearest_vertex(coords, *point), 1.0)])
        })
        .collect::<Vec<_>>();
    supports.push(volume_average_0form_support(
        topology,
        coords,
        "average_left_half",
        |bary| bary[0] < 0.0,
    )?);
    supports.push(volume_average_0form_support(
        topology,
        coords,
        "average_right_half",
        |bary| bary[0] >= 0.0,
    )?);
    Ok(supports)
}

fn build_1form_observations(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<WeightedSimplexSupport>, Box<dyn Error>> {
    TUNNELS
        .iter()
        .map(|tunnel| {
            let points = tunnel_loop_points(*tunnel, 18);
            edge_path_integral_support(
                topology,
                coords,
                format!("line_integral_{}", tunnel.label),
                &points,
                true,
            )
            .map_err(|err| err.into())
        })
        .collect()
}

fn build_2form_observations(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<WeightedSimplexSupport>, Box<dyn Error>> {
    let mut supports = Vec::new();
    for cavity in CAVITIES {
        let label = format!("flux_{}", cavity.label);
        supports.push(boundary_face_integral_support(
            topology,
            coords,
            label,
            move |bary| (distance(bary, cavity.center) - cavity.radius).abs() < 0.11,
            move |bary, normal| dot(normal, sub(bary, cavity.center)),
        )?);
    }
    for tunnel in TUNNELS {
        let label = format!("flux_wall_patch_{}", tunnel.label);
        supports.push(boundary_face_integral_support(
            topology,
            coords,
            label,
            move |bary| {
                let (axis_coord, radial) = tunnel_axis_and_radial(tunnel, bary);
                axis_coord.abs() < 0.45 && (radial - tunnel.radius).abs() < 0.11
            },
            move |bary, normal| dot(normal, tunnel_radial_vector(tunnel, bary)),
        )?);
    }
    Ok(supports)
}

fn build_3form_observations(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<WeightedSimplexSupport>, Box<dyn Error>> {
    let regions = [
        ("volume_northwest", [-0.55, 0.35, 0.35], 0.42),
        ("volume_southeast", [0.55, -0.35, 0.35], 0.40),
        ("volume_lower_core", [0.00, 0.10, -0.45], 0.45),
    ];
    regions
        .into_iter()
        .map(|(label, center, radius)| {
            top_cell_region_integral_support(topology, coords, label, move |bary| {
                squared_distance(bary, center) < radius * radius
            })
            .map_err(|err| err.into())
        })
        .collect()
}

fn tunnel_loop_points(tunnel: TunnelSpec, count: usize) -> Vec<[f64; 3]> {
    (0..count)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (count as f64);
            let c = theta.cos() * tunnel.loop_radius;
            let s = theta.sin() * tunnel.loop_radius;
            match tunnel.axis {
                Axis::X => [0.0, tunnel.center[1] + c, tunnel.center[2] + s],
                Axis::Y => [tunnel.center[0] + c, 0.0, tunnel.center[2] + s],
                Axis::Z => [tunnel.center[0] + c, tunnel.center[1] + s, 0.0],
            }
        })
        .collect()
}

fn tunnel_axis_and_radial(tunnel: TunnelSpec, point: [f64; 3]) -> (f64, f64) {
    match tunnel.axis {
        Axis::X => {
            let dy = point[1] - tunnel.center[1];
            let dz = point[2] - tunnel.center[2];
            (point[0], (dy * dy + dz * dz).sqrt())
        }
        Axis::Y => {
            let dx = point[0] - tunnel.center[0];
            let dz = point[2] - tunnel.center[2];
            (point[1], (dx * dx + dz * dz).sqrt())
        }
        Axis::Z => {
            let dx = point[0] - tunnel.center[0];
            let dy = point[1] - tunnel.center[1];
            (point[2], (dx * dx + dy * dy).sqrt())
        }
    }
}

fn tunnel_radial_vector(tunnel: TunnelSpec, point: [f64; 3]) -> [f64; 3] {
    match tunnel.axis {
        Axis::X => [
            0.0,
            point[1] - tunnel.center[1],
            point[2] - tunnel.center[2],
        ],
        Axis::Y => [
            point[0] - tunnel.center[0],
            0.0,
            point[2] - tunnel.center[2],
        ],
        Axis::Z => [
            point[0] - tunnel.center[0],
            point[1] - tunnel.center[1],
            0.0,
        ],
    }
}

fn observation_marker(dimension: usize, operator: &FeecCsr) -> FeecVector {
    let mut marker = FeecVector::zeros(dimension);
    for (_row, col, value) in operator.triplet_iter() {
        marker[col] += value.abs();
    }
    let max = marker.iter().fold(0.0_f64, |acc, value| acc.max(*value));
    if max > 0.0 {
        marker /= max;
    }
    marker
}

fn write_grade_outputs(
    grade_dir: &Path,
    coords: &MeshCoords,
    topology: &Complex,
    grade: usize,
    prior_sample: &FeecVector,
    posterior_mean: &FeecVector,
    posterior_variance: &FeecVector,
    observation_marker: &FeecVector,
    hodge: &crate::de_rham::HodgeProjection,
    derivative: Option<&FeecVector>,
    codifferential: Option<&FeecVector>,
) -> io::Result<()> {
    write_form_field(
        grade_dir.join("prior_sample.vtu"),
        coords,
        topology,
        grade,
        prior_sample,
        "prior_sample",
    )?;
    write_form_field(
        grade_dir.join("posterior_mean.vtu"),
        coords,
        topology,
        grade,
        posterior_mean,
        "posterior_mean",
    )?;
    write_form_field(
        grade_dir.join("posterior_variance.vtu"),
        coords,
        topology,
        grade,
        posterior_variance,
        "posterior_variance",
    )?;
    write_form_field(
        grade_dir.join("observation_marker.vtu"),
        coords,
        topology,
        grade,
        observation_marker,
        "observation_marker",
    )?;
    write_form_field(
        grade_dir.join("hodge_exact.vtu"),
        coords,
        topology,
        grade,
        &hodge.exact,
        "hodge_exact",
    )?;
    write_form_field(
        grade_dir.join("hodge_coexact.vtu"),
        coords,
        topology,
        grade,
        &hodge.coexact,
        "hodge_coexact",
    )?;
    write_form_field(
        grade_dir.join("hodge_harmonic.vtu"),
        coords,
        topology,
        grade,
        &hodge.harmonic,
        "hodge_harmonic",
    )?;
    if let Some(values) = derivative {
        write_form_field(
            grade_dir.join("d_omega.vtu"),
            coords,
            topology,
            grade + 1,
            values,
            "d_omega",
        )?;
    } else {
        fs::write(
            grade_dir.join("d_omega_endpoint_zero.txt"),
            "top-degree derivative is zero\n",
        )?;
    }
    if let Some(values) = codifferential {
        write_form_field(
            grade_dir.join("delta_omega.vtu"),
            coords,
            topology,
            grade - 1,
            values,
            "delta_omega",
        )?;
    } else {
        fs::write(
            grade_dir.join("delta_omega_endpoint_zero.txt"),
            "0-form codifferential is zero\n",
        )?;
    }
    Ok(())
}

fn write_form_field(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    grade: usize,
    values: &FeecVector,
    name: &str,
) -> io::Result<()> {
    let path = path.as_ref();
    let cochain = Cochain::new(grade, values.clone());
    visual_output::write_cochain(path, coords, topology, &cochain, name)?;
    if grade == 1 {
        visual_output::write_1form_vector_field(
            path.with_file_name(format!("{name}_vector.vtu")),
            coords,
            topology,
            &cochain,
            name,
        )?;
    } else if grade == 2 && topology.dim() == 3 && coords.dim() == 3 {
        visual_output::write_2form_vector_field(
            path.with_file_name(format!("{name}_vector.vtu")),
            coords,
            topology,
            &cochain,
            name,
        )?;
    }
    Ok(())
}

fn write_observations_csv(
    path: &Path,
    labels: &[String],
    observations: &FeecVector,
    posterior_observations: &FeecVector,
    residual: &FeecVector,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "label,observation,posterior_observation,residual")?;
    for i in 0..observations.len() {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12}",
            labels[i], observations[i], posterior_observations[i], residual[i]
        )?;
    }
    Ok(())
}

fn write_mesh_summary(summary: &MeshSummary, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "vertices={}", summary.vertices)?;
    writeln!(writer, "edges={}", summary.edges)?;
    writeln!(writer, "faces={}", summary.faces)?;
    writeln!(writer, "cells={}", summary.cells)?;
    writeln!(writer, "betti={:?}", summary.betti)?;
    Ok(())
}

fn write_grade_summary(summaries: &[GradeSummary], path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "grade,interpretation,dimension,precision_nnz,posterior_precision_nnz,tau,prior_median_variance,posterior_median_variance,observation_count,max_abs_observation_residual,hodge_reconstruction_error,hodge_orthogonality_error"
    )?;
    for summary in summaries {
        writeln!(
            writer,
            "{},{},{},{},{},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12}",
            summary.grade,
            summary.interpretation,
            summary.dimension,
            summary.precision_nnz,
            summary.posterior_precision_nnz,
            summary.tau,
            summary.prior_median_variance,
            summary.posterior_median_variance,
            summary.observation_count,
            summary.max_abs_observation_residual,
            summary.hodge_reconstruction_error,
            summary.hodge_orthogonality_error
        )?;
    }
    Ok(())
}

fn write_figure_layout(path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# De Rham Ladder Figure Layout")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "Columns: `k=0 scalar`, `k=1 circulation`, `k=2 flux`, `k=3 density`."
    )?;
    writeln!(
        writer,
        "Rows: prior sample, derivative/codifferential, Hodge exact/coexact/harmonic components, posterior mean, posterior variance, observation marker."
    )?;
    Ok(())
}

fn grade_interpretation(grade: usize) -> &'static str {
    match grade {
        0 => "scalar potential / temperature / pressure",
        1 => "circulation / work / field intensity",
        2 => "flux / current density / magnetic flux density",
        3 => "mass / charge / source density",
        _ => "unsupported",
    }
}

fn validate_config(config: &DerhamLadderConfig) -> Result<(), Box<dyn Error>> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".into());
    }
    if !config.observation_noise_variance.is_finite() || config.observation_noise_variance <= 0.0 {
        return Err("observation noise variance must be finite and positive".into());
    }
    if config.variance_probe_count == 0 {
        return Err("variance probe count must be positive".into());
    }
    if !config.hodge_projection_ridge.is_finite() || config.hodge_projection_ridge < 0.0 {
        return Err("Hodge projection ridge must be finite and nonnegative".into());
    }
    Ok(())
}

fn max_abs(values: &FeecVector) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

#[allow(dead_code)]
fn dense_row_operator(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> FeecCsr {
    let mut coo = FeecCoo::new(rows, cols);
    for &(row, col, value) in entries {
        coo.push(row, col, value);
    }
    FeecCsr::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_marker_marks_operator_support() {
        let mut coo = FeecCoo::new(2, 4);
        coo.push(0, 1, -2.0);
        coo.push(1, 3, 1.0);
        let marker = observation_marker(4, &FeecCsr::from(&coo));
        assert_eq!(marker[0], 0.0);
        assert_eq!(marker[1], 1.0);
        assert_eq!(marker[2], 0.0);
        assert_eq!(marker[3], 0.5);
    }

    #[test]
    fn tunnel_loop_points_have_requested_count() {
        let points = tunnel_loop_points(TUNNELS[0], 12);
        assert_eq!(points.len(), 12);
        assert!(points.iter().all(|point| point[0] == 0.0));
    }
}
