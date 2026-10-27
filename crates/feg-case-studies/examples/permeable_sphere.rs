//! Linear permeable sphere with an uncertain excitation.
//! Usage: permeable_sphere mesh.msh output_dir

use feec_gmrf::prelude::*;
use manifold::{
    geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
    topology::complex::Complex,
};
use nalgebra::Vector3;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::{error::Error, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mesh_path = args
        .next()
        .ok_or("usage: permeable_sphere mesh.msh output_dir")?;
    let output = PathBuf::from(args.next().ok_or("missing output directory")?);

    // Inputs: radius must match the mesh; field values use Tesla.
    let sphere = SphereReference::new(1.0, 10.0)?;
    let excitation_mean = [0.0, 0.0, 1.0];
    let excitation_std = [0.03_f64, 0.03, 0.05];
    let truth = [0.025, -0.015, 1.04];
    let sensor_sigma = 0.05;
    let pde_sigma = 0.003;
    let seed = 130013;

    let mesh = TaggedTetrahedralMesh::from_gmsh_bytes(&std::fs::read(mesh_path)?)?;
    let (topology, coords) = (&mesh.topology, &mesh.coords);
    if mesh.cell_tags.iter().any(|tag| *tag != 1 && *tag != 2)
        || !mesh.cell_tags.contains(&1)
        || !mesh.cell_tags.contains(&2)
    {
        return Err("expected material tags sphere=1 and air=2".into());
    }
    let metric = coords.to_edge_lengths(topology);
    let reluctivities = mesh
        .cell_tags
        .iter()
        .map(|tag| if *tag == 1 { 1.0 / sphere.mu_r } else { 1.0 })
        .collect::<Vec<_>>();
    let system =
        LinearMagnetostaticSystem::from_cell_reluctivities(topology, &metric, &reluctivities)?;

    // Cartesian three-component air sensors. Mesh refinement uses these same positions.
    let sensors = [
        [1.42, 0.27, 0.36],
        [-1.53, 0.31, -0.22],
        [0.24, 1.47, 0.41],
        [-0.37, -1.61, 0.19],
        [0.29, -0.23, 1.54],
        [-0.21, 0.38, -1.43],
    ]
    .map(|x| MagneticPointSensor3d::cartesian((Vector3::from(x) * sphere.radius).into()))
    .into_iter()
    .collect::<feec_gmrf::Result<Vec<_>>>()?;
    for sensor in &sensors {
        let x = common::linalg::nalgebra::Vector::from_column_slice(&sensor.position());
        let cell = coords
            .find_cell_containing(topology, x.as_view())
            .ok_or("sensor is outside mesh")?;
        if mesh.cell_tags[cell.kidx()] != 2 {
            return Err("sensors must lie in air".into());
        }
    }
    let sensor_map = magnetic_point_measurement_map_3d(topology, coords, &sensors)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let observations = sensors
        .iter()
        .flat_map(|s| sphere.flux_density(s.position(), truth))
        .map(|value| {
            let noise: f64 = StandardNormal.sample(&mut rng);
            value + sensor_sigma * noise
        })
        .collect();

    // Embedding inserts zeros on boundary edges; lifting supplies the total boundary values.
    let correction_embedding = system.boundary.active_to_full();
    let excitation_lifting = lifting_map(topology, coords, sphere)?;
    let total_potential_map = joint_block_map(&[correction_embedding, &excitation_lifting])?;

    // Corrected alpha=2 prior: Matérn spectrum on the magnetic correction d(Pz).
    let field_prior =
        MaternPriorBuilder::from_feec_homogeneous_full_relative_with_projected_inverses(
            topology, &metric, 1,
        )?
        .parameters(MaternParameters::new(MaternAlpha::Two, 1.0, 0.001)?)
        .build_spectrum_matched_potential()?;
    let prior_boundary = field_prior
        .boundary_elimination()
        .ok_or("missing prior boundary layout")?;
    if prior_boundary.layout().active_dofs() != system.boundary.layout().active_dofs() {
        return Err("prior and PDE boundary layouts differ".into());
    }
    let mut input_precision = SparseMat::new(3, 3);
    for (i, sigma) in excitation_std.iter().enumerate() {
        input_precision.push(i, i, 1.0 / sigma.powi(2));
    }
    let input_prior = GaussianPrior::new(excitation_mean.to_vec(), input_precision)?;
    let mut posterior = JointLinearPdeModelBuilder::new(&field_prior, &input_prior)?
        .observe_pde(
            &system.correction_operator,
            &system.lifting_residual(&excitation_lifting)?,
            system.residual_noise(pde_sigma)?,
        )?
        .observe(
            &sensor_map.compose(correction_embedding)?,
            &sensor_map.compose(&excitation_lifting)?,
            observations,
            GaussianNoise::standard_deviation(sensor_sigma)?,
        )?
        .condition()?;

    let n = field_prior.dimension();
    let excitation_map = LinearMap::selector(n + 3, &[n, n + 1, n + 2])?;
    let excitation = posterior.pushforward_mean(&excitation_map)?;
    let excitation_covariance = posterior.pushforward_covariance(&excitation_map)?;
    let xi = Vector3::from_column_slice(&excitation);
    let direction: [f64; 3] = (xi / xi.norm()).into();
    println!("truth: {truth:?}; posterior mean: {xi:?}");
    println!("magnitude: {}; direction: {:?}", xi.norm(), direction);

    // The joint pushforward includes correction/excitation cross covariance.
    let magnetic = MagneticFieldMaps3d::from_feec(topology, coords)?;
    let field_map = magnetic
        .magnetic_field
        .map()
        .compose(&total_potential_map)?;
    let field_mean = posterior.pushforward_mean(&field_map)?;
    let variance = posterior.pushforward_variance_estimate(
        &field_map,
        VarianceMethod::MonteCarlo(MonteCarloVarianceConfig::new(512, 8, seed + 1)?),
    )?;
    let field_std = variance
        .values
        .iter()
        .map(|v| v.max(0.0).sqrt())
        .collect::<Vec<_>>();
    let reference = topology
        .cells()
        .handle_iter()
        .flat_map(|cell| {
            let x = cell.coord_simplex(coords).barycenter();
            sphere.flux_density([x[0], x[1], x[2]], truth)
        })
        .collect::<Vec<_>>();
    let error = field_mean
        .iter()
        .zip(&reference)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();

    std::fs::create_dir_all(&output)?;
    let mut fields = TopCellVtuBuilder::new();
    for (name, values) in [
        ("B_reference", &reference),
        ("B_posterior_mean", &field_mean),
        ("B_posterior_std_mc", &field_std),
        ("B_posterior_error", &error),
    ] {
        fields.add_flat_vector(name, values, VectorLayout3::Interleaved)?;
    }
    fields.add_scalar(
        "material_tag",
        mesh.cell_tags.iter().map(|tag| *tag as f64).collect(),
    )?;
    fields.write(output.join("fields.vtu"), coords, topology)?;
    std::fs::write(
        output.join("excitation.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "truth": truth, "posterior_mean": excitation, "posterior_covariance": excitation_covariance,
            "magnitude_of_mean": xi.norm(), "direction_of_mean": direction
        }))?,
    )?;
    println!("ParaView fields: {}", output.join("fields.vtu").display());
    Ok(())
}

/// Exact solution in an unbounded exterior, for any positive linear mu_r.
#[derive(Debug, Clone, Copy)]
struct SphereReference {
    radius: f64,
    mu_r: f64,
}
impl SphereReference {
    fn new(radius: f64, mu_r: f64) -> Result<Self> {
        if !radius.is_finite() || radius <= 0.0 || !mu_r.is_finite() || mu_r <= 0.0 {
            return Err("sphere radius and relative permeability must be positive".into());
        }
        Ok(Self { radius, mu_r })
    }
    fn beta(self) -> f64 {
        (self.mu_r - 1.0) / (self.mu_r + 2.0)
    }
    fn interior_factor(self) -> f64 {
        3.0 * self.mu_r / (self.mu_r + 2.0)
    }
    #[cfg(test)]
    fn potential(self, x: [f64; 3], xi: [f64; 3]) -> [f64; 3] {
        let x = Vector3::from(x);
        let gamma = if x.norm() <= self.radius {
            1.0
        } else {
            (self.radius / x.norm()).powi(3)
        };
        ((0.5 + self.beta() * gamma) * Vector3::from(xi).cross(&x)).into()
    }
    fn flux_density(self, x: [f64; 3], xi: [f64; 3]) -> [f64; 3] {
        let x = Vector3::from(x);
        let xi = Vector3::from(xi);
        if x.norm() <= self.radius {
            return (self.interior_factor() * xi).into();
        }
        let n = x / x.norm();
        (xi + self.beta() * (self.radius / x.norm()).powi(3) * (3.0 * n * n.dot(&xi) - xi)).into()
    }
    // Smooth extension of the exact outer trace; only the incident field near the sphere.
    fn lifting(self, x: [f64; 3], xi: [f64; 3]) -> [f64; 3] {
        let x = Vector3::from(x);
        let r = x.norm();
        let s = ((r / self.radius - 1.25) / 0.75).clamp(0.0, 1.0);
        let chi = s.powi(3) * (10.0 - 15.0 * s + 6.0 * s * s);
        let dipole = if r <= 1.25 * self.radius {
            0.0
        } else {
            chi * self.beta() * (self.radius / r).powi(3)
        };
        ((0.5 + dipole) * Vector3::from(xi).cross(&x)).into()
    }
}

fn lifting_map(
    topology: &Complex,
    coords: &MeshCoords,
    reference: SphereReference,
) -> Result<LinearMap> {
    for edge in EssentialBoundaryConditions::topological_boundary_dofs(topology, 1)? {
        for &vertex in &topology.skeleton(1).simplex_by_kidx(edge).vertices {
            if coords.coord(vertex).norm() < 2.01 * reference.radius {
                return Err("outer boundary must lie beyond the lifting cutoff".into());
            }
        }
    }
    let subdivisions = 32;
    let mut matrix = SparseMat::new(topology.nsimplices(1), 3);
    for edge in topology.skeleton(1).handle_iter() {
        let v0 = coords.coord(edge.vertices[0]);
        let v1 = coords.coord(edge.vertices[1]);
        let x0 = Vector3::new(v0[0], v0[1], v0[2]);
        let d = Vector3::new(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]);
        for k in 0..3 {
            let mut xi = [0.0; 3];
            xi[k] = 1.0;
            let mut integral = 0.0;
            for j in 0..subdivisions {
                for g in [-1.0_f64 / 3.0_f64.sqrt(), 1.0_f64 / 3.0_f64.sqrt()] {
                    let t = (j as f64 + 0.5 * (1.0 + g)) / subdivisions as f64;
                    let potential = Vector3::from(reference.lifting((x0 + d * t).into(), xi));
                    integral += potential.dot(&d) / (2 * subdivisions) as f64;
                }
            }
            if integral != 0.0 {
                matrix.push(edge.kidx(), k, integral);
            }
        }
    }
    Ok(LinearMap::new(matrix)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_satisfies_material_interface_and_demagnetization() {
        let xi = Vector3::new(0.23, -0.17, 1.07);
        let n = Vector3::new(1.0, 2.0, -3.0).normalize();
        for mu in [0.1, 1.0, 10.0, 1000.0] {
            let sphere = SphereReference::new(0.7, mu).unwrap();
            let inside = xi * sphere.interior_factor();
            let outside = xi + sphere.beta() * (3.0 * n * n.dot(&xi) - xi);
            assert!(n.dot(&(inside - outside)).abs() < 1e-12);
            assert!(n.cross(&(inside / mu - outside)).norm() < 1e-12);
            assert!((inside / 3.0 + inside * (2.0 / (3.0 * mu)) - xi).norm() < 1e-12);
            let x = n * 0.7;
            assert!(
                (x.cross(&xi) * (0.5 * sphere.interior_factor() - 0.5 - sphere.beta())).norm()
                    < 1e-12
            );
        }
    }

    #[test]
    fn reference_curl_and_divergence_match_closed_form() {
        let sphere = SphereReference::new(1.0, 10.0).unwrap();
        let xi = [0.12, -0.2, 0.93];
        for point in [[0.2, -0.3, 0.1], [1.4, -0.2, 0.35]] {
            let eps = 1e-5;
            let mut derivative = [Vector3::zeros(); 3];
            let mut divergence = 0.0;
            for i in 0..3 {
                let mut plus = point;
                plus[i] += eps;
                let mut minus = point;
                minus[i] -= eps;
                derivative[i] = (Vector3::from(sphere.potential(plus, xi))
                    - Vector3::from(sphere.potential(minus, xi)))
                    / (2.0 * eps);
                divergence += (sphere.flux_density(plus, xi)[i]
                    - sphere.flux_density(minus, xi)[i])
                    / (2.0 * eps);
            }
            let curl = Vector3::new(
                derivative[1][2] - derivative[2][1],
                derivative[2][0] - derivative[0][2],
                derivative[0][1] - derivative[1][0],
            );
            assert!((curl - Vector3::from(sphere.flux_density(point, xi))).norm() < 1e-8);
            assert!(divergence.abs() < 1e-8);
        }
    }

    #[test]
    fn lifting_has_exact_outer_trace_and_incident_interior() {
        let sphere = SphereReference::new(1.0, 10.0).unwrap();
        let xi = [0.1, 0.2, 1.0];
        for point in [[3.0, 0.3, -0.7], [-0.2, -3.0, 0.6]] {
            assert!(
                (Vector3::from(sphere.lifting(point, xi))
                    - Vector3::from(sphere.potential(point, xi)))
                .norm()
                    < 1e-14
            );
        }
        let point = [0.2, 0.3, 0.4];
        let incident = Vector3::from(xi).cross(&Vector3::from(point)) * 0.5;
        assert!((Vector3::from(sphere.lifting(point, xi)) - incident).norm() < 1e-14);
        assert!(
            (Vector3::from(sphere.lifting(point, xi)) - Vector3::from(sphere.potential(point, xi)))
                .norm()
                > 0.1
        );
        let vacuum = SphereReference::new(1.0, 1.0).unwrap();
        assert_eq!(vacuum.flux_density([2.0, 0.1, 0.3], xi), xi);
    }
}
