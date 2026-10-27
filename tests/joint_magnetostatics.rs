use ddf::ManifoldComplexExt;
use feec_gmrf::prelude::*;
use manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::simplex::SimplexHandleExt};

fn map(rows: &[&[f64]]) -> LinearMap {
    LinearMap::weighted_rows(
        rows[0].len(),
        &rows
            .iter()
            .map(|row| row.iter().copied().enumerate().collect())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn cochain_basis_projection_preserves_oriented_edge_integrals() {
    use common::linalg::nalgebra::Vector;
    use exterior::field::DiffFormClosure;
    use manifold::geometry::coord::quadrature::SimplexQuadRule;

    let (topology, coords) = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0).compute_coord_complex();
    let basis: [_; 3] = std::array::from_fn(|axis| {
        DiffFormClosure::one_form(
            move |_| {
                let mut value = Vector::zeros(3);
                value[axis] = 1.0;
                value
            },
            3,
        )
    });
    let rule = SimplexQuadRule::segment_gauss_legendre(32);
    let projection = cochain_projection_map(&topology, &coords, &basis, Some(&rule)).unwrap();
    let input = [0.2, -0.4, 0.9];
    let actual = projection.apply(&input).unwrap();
    for (edge, value) in topology.skeleton(1).handle_iter().zip(actual) {
        let a = coords.coord(edge.vertices[0]);
        let b = coords.coord(edge.vertices[1]);
        let expected: f64 = (0..3).map(|i| input[i] * (b[i] - a[i])).sum();
        assert!((value - expected).abs() < 1e-13);
    }

    // Fundamental theorem for the non-polynomial exact 1-form d(exp(x)).
    let exact = [DiffFormClosure::one_form(
        |x| Vector::from_column_slice(&[x[0].exp(), 0.0, 0.0]),
        3,
    )];
    let projected = cochain_projection_map(&topology, &coords, &exact, Some(&rule))
        .unwrap()
        .apply(&[1.0])
        .unwrap();
    for (edge, value) in topology.skeleton(1).handle_iter().zip(projected) {
        let a = coords.coord(edge.vertices[0]);
        let b = coords.coord(edge.vertices[1]);
        assert!((value - (b[0].exp() - a[0].exp())).abs() < 1e-10);
    }
}

#[test]
fn cochain_basis_projection_rejects_incompatible_fields_and_quadrature() {
    use common::linalg::nalgebra::Vector;
    use exterior::field::DiffFormClosure;
    use manifold::geometry::coord::quadrature::SimplexQuadRule;

    let (topology, coords) = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0).compute_coord_complex();
    let basis = [
        DiffFormClosure::scalar(|_| 1.0, 3),
        DiffFormClosure::one_form(|_| Vector::zeros(3), 3),
    ];
    assert!(cochain_projection_map(&topology, &coords, &basis, None).is_err());
    assert!(cochain_projection_map(
        &topology,
        &coords,
        &basis[1..],
        Some(&SimplexQuadRule::barycentric(2)),
    )
    .is_err());
    assert!(cochain_projection_map(&topology, &coords, &basis[..0], None).is_err());
}

#[test]
fn joint_model_preserves_input_prior_and_matches_dense_coupled_gaussian() {
    let state = GaussianPrior::new(vec![0.2], SparseMat::diagonal(1, 4.0)).unwrap();
    let input = GaussianPrior::new(vec![1.3], SparseMat::diagonal(1, 9.0)).unwrap();
    let mut independent = JointLinearPdeModelBuilder::new(&state, &input)
        .unwrap()
        .condition()
        .unwrap();
    assert_eq!(independent.latent_mean(), &[0.2, 1.3]);
    let independent_cov = independent
        .pushforward_covariance(&LinearMap::identity(2))
        .unwrap();
    assert!((independent_cov[1][1] - 1.0 / 9.0).abs() < 1e-13);
    assert!(independent_cov[0][1].abs() < 1e-13);
    let mut posterior = JointLinearPdeModelBuilder::new(&state, &input)
        .unwrap()
        .observe_pde(
            &map(&[&[2.0]]),
            &map(&[&[-3.0]]),
            GaussianNoise::variance(0.7).unwrap(),
        )
        .unwrap()
        .observe(
            &map(&[&[1.0]]),
            &map(&[&[2.0]]),
            vec![2.7],
            GaussianNoise::variance(0.11).unwrap(),
        )
        .unwrap()
        .condition()
        .unwrap();
    let prior_q = nalgebra::Matrix2::new(4.0, 0.0, 0.0, 9.0);
    let a = nalgebra::Vector2::new(2.0, -3.0);
    let h = nalgebra::Vector2::new(1.0, 2.0);
    let q = prior_q + a * a.transpose() / 0.7 + h * h.transpose() / 0.11;
    let cov = q.try_inverse().unwrap();
    let expected = cov * (prior_q * nalgebra::Vector2::new(0.2, 1.3) + h * (2.7 / 0.11));
    for i in 0..2 {
        assert!((posterior.latent_mean()[i] - expected[i]).abs() < 1e-12);
    }
    let actual = posterior
        .pushforward_covariance(&LinearMap::identity(2))
        .unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!((actual[i][j] - cov[(i, j)]).abs() < 1e-12);
        }
    }
    // Total q=Pz+Gxi includes posterior state-input covariance.
    let total = joint_block_map(&[&map(&[&[1.0], &[0.0]]), &map(&[&[2.0], &[3.0]])]).unwrap();
    let total_cov = posterior.pushforward_covariance(&total).unwrap();
    let t = nalgebra::Matrix2::new(1.0, 2.0, 0.0, 3.0);
    let expected_total = t * cov * t.transpose();
    assert!((total_cov[0][0] - expected_total[(0, 0)]).abs() < 1e-12);
    assert!((total_cov[1][1] - 9.0 * cov[(1, 1)]).abs() < 1e-12);
    assert!(total_cov[1][1] > 0.0); // zero correction trace is not zero total variance
    assert!((total_cov[0][0] - (cov[(0, 0)] + 4.0 * cov[(1, 1)])).abs() > 1e-3);
}

#[test]
fn material_curl_and_point_maps_preserve_gauge_and_uniform_fields() {
    let (topology, coords) = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let nu = (0..topology.nsimplices(3))
        .map(|i| if i % 2 == 0 { 0.1 } else { 1.0 })
        .collect::<Vec<_>>();
    let system =
        LinearMagnetostaticSystem::from_cell_reluctivities(&topology, &metric, &nu).unwrap();
    let d0 = LinearMap::from_feec_csr(&common::linalg::nalgebra::CsrMatrix::from(
        &topology.exterior_derivative_operator(0),
    ))
    .unwrap();
    let phi = (0..topology.nsimplices(0))
        .map(|i| (i as f64).sin())
        .collect::<Vec<_>>();
    let gradient = d0.apply(&phi).unwrap();
    let kg = system.full_operator.apply(&gradient).unwrap();
    assert!(kg.iter().all(|x| x.abs() < 1e-12));
    let b = [0.2, -0.4, 0.9];
    let q = topology
        .skeleton(1)
        .handle_iter()
        .map(|edge| {
            let a = coords.coord(edge.vertices[0]);
            let c = coords.coord(edge.vertices[1]);
            0.5 * (b[0] * (a[1] * c[2] - a[2] * c[1])
                + b[1] * (a[2] * c[0] - a[0] * c[2])
                + b[2] * (a[0] * c[1] - a[1] * c[0]))
        })
        .collect::<Vec<_>>();
    let cell = topology.cells().handle_iter().nth(3).unwrap();
    let x = cell.coord_simplex(&coords).barycenter();
    let sensor = MagneticPointSensor3d::new(
        [x[0], x[1], x[2]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap();
    let h = magnetic_point_measurement_map_3d(&topology, &coords, &[sensor]).unwrap();
    let actual = h.apply(&q).unwrap();
    for (a, e) in actual.iter().zip([-0.4, -0.2, 0.9]) {
        assert!((a - e).abs() < 1e-12);
    }
    assert!(h.apply(&gradient).unwrap().iter().all(|x| x.abs() < 1e-12));
    assert!(MagneticPointSensor3d::new(
        [0.0; 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    .is_err());
    assert!(magnetic_point_measurement_map_3d(
        &topology,
        &coords,
        &[MagneticPointSensor3d::cartesian([10.0; 3]).unwrap()]
    )
    .is_err());
    // Material assembly is linear in per-cell reluctivity.
    let scaled = LinearMagnetostaticSystem::from_cell_reluctivities(
        &topology,
        &metric,
        &nu.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
    )
    .unwrap();
    let kq = system.full_operator.apply(&q).unwrap();
    let twice = scaled.full_operator.apply(&q).unwrap();
    for (a, b) in twice.iter().zip(kq) {
        assert!((a - 2.0 * b).abs() < 1e-12);
    }
}

#[test]
fn tagged_gmsh_import_uses_physical_tags_and_explicit_shuffled_node_tags() {
    let fixture = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Entities\n0 0 0 1\n7 0 0 0 1 1 1 1 42 0\n$EndEntities\n$Nodes\n1 4 10 40\n3 7 0 4\n40 10 30 20\n0 0 1\n0 0 0\n0 1 0\n1 0 0\n$EndNodes\n$Elements\n1 1 1 1\n3 7 4 1\n1 10 20 30 40\n$EndElements\n";
    let mesh = TaggedTetrahedralMesh::from_gmsh_bytes(fixture).unwrap();
    assert_eq!(mesh.cell_tags, vec![42]);
    assert_eq!(mesh.coords.coord(0).as_slice(), &[0.0, 0.0, 0.0]);
    assert_eq!(mesh.coords.coord(1).as_slice(), &[1.0, 0.0, 0.0]);
    assert_eq!(mesh.coords.coord(3).as_slice(), &[0.0, 0.0, 1.0]);
    assert!(TaggedTetrahedralMesh::from_gmsh_bytes(b"bad mesh").is_err());
}

fn uniform_field_lifting(
    topology: &manifold::topology::complex::Complex,
    coords: &manifold::geometry::coord::mesh::MeshCoords,
) -> LinearMap {
    // The exact edge integral of A=.5*(xi cross x) is
    // .5*xi dot (x_start cross x_end), in canonical edge orientation.
    let rows = topology
        .skeleton(1)
        .handle_iter()
        .map(|edge| {
            let a = coords.coord(edge.vertices[0]);
            let b = coords.coord(edge.vertices[1]);
            vec![
                (0, 0.5 * (a[1] * b[2] - a[2] * b[1])),
                (1, 0.5 * (a[2] * b[0] - a[0] * b[2])),
                (2, 0.5 * (a[0] * b[1] - a[1] * b[0])),
            ]
        })
        .collect::<Vec<_>>();
    LinearMap::weighted_rows(3, &rows).unwrap()
}

#[test]
fn vacuum_deterministic_control_keeps_uniform_field_and_meets_original_residual_tolerance() {
    let (topology, coords) = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let system = LinearMagnetostaticSystem::from_cell_reluctivities(
        &topology,
        &metric,
        &vec![1.0; topology.nsimplices(3)],
    )
    .unwrap();
    let lifting = uniform_field_lifting(&topology, &coords);
    let tolerance = 1.0e-10;
    let xi = [0.25, -0.17, 1.04];
    let original_lifting = lifting.apply(&xi).unwrap();
    let lifted_residual = system
        .lifting_residual(&lifting)
        .unwrap()
        .apply(&xi)
        .unwrap();
    let initial_residual_norm = lifted_residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(initial_residual_norm <= tolerance);
    let solved = system
        .solve_lifted_deterministic(&lifting, &xi, tolerance)
        .unwrap();
    for (actual, expected) in solved.iter().zip(&original_lifting) {
        assert!((actual - expected).abs() < tolerance);
    }
    let residual = system
        .free_test_selector
        .apply(&system.full_operator.apply(&solved).unwrap())
        .unwrap();
    let residual_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(residual_norm <= tolerance, "residual={residual_norm:e}");
    let magnetic = MagneticFieldMaps3d::from_feec(&topology, &coords).unwrap();
    let field = magnetic.magnetic_field.map().apply(&solved).unwrap();
    for actual in field.chunks_exact(3) {
        for k in 0..3 {
            assert!((actual[k] - xi[k]).abs() < 1e-12);
        }
    }
}

#[test]
fn heterogeneous_curl_energy_matches_volume_integral_of_reconstructed_field() {
    let (topology, coords) = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let reluctivities = (0..topology.nsimplices(3))
        .map(|i| if i % 3 == 0 { 0.07 } else { 1.3 })
        .collect::<Vec<_>>();
    let system =
        LinearMagnetostaticSystem::from_cell_reluctivities(&topology, &metric, &reluctivities)
            .unwrap();
    let q = (0..topology.nsimplices(1))
        .map(|i| (0.37 * i as f64).sin() + (0.13 * i as f64).cos())
        .collect::<Vec<_>>();
    let kq = system.full_operator.apply(&q).unwrap();
    let assembled_energy: f64 = q.iter().zip(&kq).map(|(a, b)| a * b).sum();
    let magnetic = MagneticFieldMaps3d::from_feec(&topology, &coords).unwrap();
    let field = magnetic.magnetic_field.map().apply(&q).unwrap();
    // d of lowest-order Whitney 1-forms is constant on each affine cell, so
    // this volume integral is exact, without numerical quadrature.
    let reconstructed_energy: f64 = topology
        .cells()
        .handle_iter()
        .map(|cell| {
            let c = cell.kidx();
            reluctivities[c]
                * cell.coord_simplex(&coords).vol()
                * field[3 * c..3 * c + 3].iter().map(|b| b * b).sum::<f64>()
        })
        .sum();
    assert!(assembled_energy > 0.0);
    assert!(
        (assembled_energy - reconstructed_energy).abs() <= 1e-12 * reconstructed_energy,
        "assembled={assembled_energy:e}, reconstructed={reconstructed_energy:e}"
    );
}

#[test]
fn magnetostatic_assembly_and_controls_reject_invalid_parameters() {
    let (topology, coords) = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let weights = vec![1.0; topology.nsimplices(3)];
    assert!(LinearMagnetostaticSystem::from_cell_reluctivities(&topology, &metric, &[]).is_err());
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut bad_weights = weights.clone();
        bad_weights[0] = invalid;
        assert!(LinearMagnetostaticSystem::from_cell_reluctivities(
            &topology,
            &metric,
            &bad_weights
        )
        .is_err());
    }
    let system =
        LinearMagnetostaticSystem::from_cell_reluctivities(&topology, &metric, &weights).unwrap();
    let lifting = uniform_field_lifting(&topology, &coords);
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(system
            .solve_lifted_deterministic(&lifting, &[0.0, 0.0, 1.0], invalid)
            .is_err());
        assert!(system.residual_noise(invalid).is_err());
    }
    assert!(system
        .solve_lifted_deterministic(&lifting, &[0.0, 1.0], 1e-10)
        .is_err());
    for invalid in [f64::NAN, f64::INFINITY] {
        assert!(system
            .solve_lifted_deterministic(&lifting, &[invalid, 0.0, 1.0], 1e-10)
            .is_err());
    }
    let wrong_space = LinearMap::identity(topology.nsimplices(1) - 1);
    assert!(system.lifting_residual(&wrong_space).is_err());
}
