use crate::prior::matern::{
    build_lindgren_precision_from_system,
    one_form::{
        build_matern_mass_inverse_1form, build_matern_mass_inverse_1form_with_coords,
        MaternMassInverse as Matern1FormMassInverse,
    },
};
use crate::sparse::{add_sparse, diag_matrix, invert_diag, matrix_diag, scale_matrix};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr};
use formoniq::{
    assemble::{
        assemble_barycentric_dual_2form_sparse_inverse_galmat,
        assemble_whitney_2form_projected_sparse_inverse_galmat, BarycentricDualSparseInverseConfig,
    },
    problems::hodge_laplace::MixedGalmats,
};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};

pub use super::MaternAlpha;
pub use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf};

pub struct HodgeLaplacian2Form {
    pub mass_u: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaternMassInverse {
    #[default]
    ExactTopDegreeDiagonalOrProjectedNc2,
    BarycentricDualSparseInverse,
}

#[derive(Debug, Clone, Copy)]
pub struct MaternConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: MaternMassInverse,
}

pub fn build_hodge_laplacian_2form(
    topology: &Complex,
    metric: &MeshLengths,
) -> Result<HodgeLaplacian2Form, String> {
    let galmats = MixedGalmats::compute(topology, metric, 2);
    build_hodge_laplacian_2form_from_galmats(topology, metric, &galmats)
}

pub fn build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    lower_mass_inverse: Matern1FormMassInverse,
) -> Result<HodgeLaplacian2Form, String> {
    let galmats = MixedGalmats::compute(topology, metric, 2);
    build_hodge_laplacian_2form_from_galmats_with_optional_coords(
        topology,
        Some(coords),
        metric,
        &galmats,
        lower_mass_inverse,
    )
}

pub fn build_hodge_laplacian_2form_with_lower_mass_inverse_matrix(
    topology: &Complex,
    metric: &MeshLengths,
    lower_mass_inverse: &FeecCsr,
) -> Result<HodgeLaplacian2Form, String> {
    let galmats = MixedGalmats::compute(topology, metric, 2);
    let mass_u = galmats.mass_u_csr();
    let codifdif_u = if galmats.codifdif_u().nrows() == 0 {
        FeecCsr::from(&FeecCoo::new(mass_u.nrows(), mass_u.ncols()))
    } else {
        FeecCsr::from(galmats.codifdif_u())
    };

    let laplacian = if galmats.mass_sigma().nrows() == 0 {
        codifdif_u
    } else {
        let mass_sigma = FeecCsr::from(galmats.mass_sigma());
        if lower_mass_inverse.nrows() != mass_sigma.nrows()
            || lower_mass_inverse.ncols() != mass_sigma.ncols()
        {
            return Err(format!(
                "lower mass inverse dimensions {}x{} do not match sigma mass {}x{}",
                lower_mass_inverse.nrows(),
                lower_mass_inverse.ncols(),
                mass_sigma.nrows(),
                mass_sigma.ncols()
            ));
        }
        let dif_sigma = FeecCsr::from(galmats.dif_sigma());
        let codif_u = FeecCsr::from(galmats.codif_u());
        let schur_mid = &dif_sigma * lower_mass_inverse;
        let schur = schur_mid * &codif_u;
        add_sparse(&codifdif_u, &schur)
    };

    Ok(HodgeLaplacian2Form { mass_u, laplacian })
}

pub fn build_hodge_laplacian_2form_from_galmats(
    topology: &Complex,
    metric: &MeshLengths,
    galmats: &MixedGalmats,
) -> Result<HodgeLaplacian2Form, String> {
    build_hodge_laplacian_2form_from_galmats_with_optional_coords(
        topology,
        None,
        metric,
        galmats,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
    )
}

fn build_hodge_laplacian_2form_from_galmats_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    galmats: &MixedGalmats,
    lower_mass_inverse: Matern1FormMassInverse,
) -> Result<HodgeLaplacian2Form, String> {
    let mass_u = galmats.mass_u_csr();
    let codifdif_u = if galmats.codifdif_u().nrows() == 0 {
        FeecCsr::from(&FeecCoo::new(mass_u.nrows(), mass_u.ncols()))
    } else {
        FeecCsr::from(galmats.codifdif_u())
    };

    let laplacian = if galmats.mass_sigma().nrows() == 0 {
        codifdif_u
    } else {
        let mass_sigma = FeecCsr::from(galmats.mass_sigma());
        let sigma_inverse = if let Some(coords) = coords {
            build_matern_mass_inverse_1form_with_coords(
                topology,
                coords,
                metric,
                &mass_sigma,
                lower_mass_inverse,
            )?
        } else {
            build_matern_mass_inverse_1form(topology, metric, &mass_sigma, lower_mass_inverse)
        };
        let dif_sigma = FeecCsr::from(galmats.dif_sigma());
        let codif_u = FeecCsr::from(galmats.codif_u());
        let schur_mid = &dif_sigma * &sigma_inverse;
        let schur = schur_mid * &codif_u;
        add_sparse(&codifdif_u, &schur)
    };

    Ok(HodgeLaplacian2Form { mass_u, laplacian })
}

pub fn build_matern_system_matrix_2form(hodge: &HodgeLaplacian2Form, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&hodge.laplacian, &scale_matrix(&hodge.mass_u, kappa2))
}

pub fn build_matern_mass_inverse_2form(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecCsr, String> {
    build_matern_mass_inverse_2form_with_optional_coords(topology, None, metric, mass_u, strategy)
}

pub fn build_matern_mass_inverse_2form_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecCsr, String> {
    build_matern_mass_inverse_2form_with_optional_coords(
        topology,
        Some(coords),
        metric,
        mass_u,
        strategy,
    )
}

fn build_matern_mass_inverse_2form_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecCsr, String> {
    match strategy {
        MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2 => match topology.dim() {
            2 => Ok(diag_matrix(&invert_diag(&matrix_diag(mass_u)))),
            3 => {
                let projected =
                    assemble_whitney_2form_projected_sparse_inverse_galmat(topology, metric);
                let projected = FeecCsr::from(&projected);
                if projected.nrows() != mass_u.nrows() || projected.ncols() != mass_u.ncols() {
                    return Err(format!(
                        "projected 2-form sparse inverse dimensions {}x{} do not match 2-form mass {}x{}",
                        projected.nrows(),
                        projected.ncols(),
                        mass_u.nrows(),
                        mass_u.ncols()
                    ));
                }
                Ok(projected)
            }
            dim => Err(format!(
                "2-form Matérn mass inverse is only implemented for intrinsic mesh dimensions 2 and 3, got {dim}"
            )),
        },
        MaternMassInverse::BarycentricDualSparseInverse => {
            let coords = coords.ok_or_else(|| {
                "barycentric-dual 2-form sparse inverse requires MeshCoords; use the coordinate-aware checked builder".to_string()
            })?;
            let barycentric = assemble_barycentric_dual_2form_sparse_inverse_galmat(
                topology,
                coords,
                BarycentricDualSparseInverseConfig::default(),
            )?;
            let barycentric = FeecCsr::from(&barycentric);
            if barycentric.nrows() != mass_u.nrows() || barycentric.ncols() != mass_u.ncols() {
                return Err(format!(
                    "barycentric-dual 2-form sparse inverse dimensions {}x{} do not match 2-form mass {}x{}",
                    barycentric.nrows(),
                    barycentric.ncols(),
                    mass_u.nrows(),
                    mass_u.ncols()
                ));
            }
            Ok(barycentric)
        }
    }
}

pub fn build_matern_precision_2form_with_mass_inverse(
    hodge: &HodgeLaplacian2Form,
    mass_inverse: &FeecCsr,
    kappa: f64,
    tau: f64,
) -> FeecCsr {
    build_matern_precision_2form_with_mass_inverse_for_alpha(
        hodge,
        mass_inverse,
        MaternAlpha::Two,
        kappa,
        tau,
    )
}

pub fn build_matern_precision_2form_with_mass_inverse_for_alpha(
    hodge: &HodgeLaplacian2Form,
    mass_inverse: &FeecCsr,
    alpha: MaternAlpha,
    kappa: f64,
    tau: f64,
) -> FeecCsr {
    let a = build_matern_system_matrix_2form(hodge, kappa);
    build_lindgren_precision_from_system(&a, mass_inverse, alpha, tau)
}

pub fn build_matern_precision_2form(
    topology: &Complex,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian2Form,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    build_matern_precision_2form_for_alpha(topology, metric, hodge, MaternAlpha::Two, config)
}

pub fn build_matern_precision_2form_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian2Form,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    build_matern_precision_2form_for_alpha_with_coords(
        topology,
        coords,
        metric,
        hodge,
        MaternAlpha::Two,
        config,
    )
}

pub fn build_matern_precision_2form_for_alpha(
    topology: &Complex,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian2Form,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    let mass_inverse =
        build_matern_mass_inverse_2form(topology, metric, &hodge.mass_u, config.mass_inverse)?;
    Ok(build_matern_precision_2form_with_mass_inverse_for_alpha(
        hodge,
        &mass_inverse,
        alpha,
        config.kappa,
        config.tau,
    ))
}

pub fn build_matern_precision_2form_for_alpha_with_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian2Form,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    let mass_inverse = build_matern_mass_inverse_2form_with_coords(
        topology,
        coords,
        metric,
        &hodge.mass_u,
        config.mass_inverse,
    )?;
    Ok(build_matern_precision_2form_with_mass_inverse_for_alpha(
        hodge,
        &mass_inverse,
        alpha,
        config.kappa,
        config.tau,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn diagonal_entries(mat: &FeecCsr) -> Vec<f64> {
        let mut diag = vec![0.0; mat.nrows()];
        for (row, col, value) in mat.triplet_iter() {
            if row == col {
                diag[row] += *value;
            }
        }
        diag
    }

    fn max_abs_entry_diff(lhs: &FeecCsr, rhs: &FeecCsr) -> f64 {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());

        let mut entries = std::collections::HashMap::new();
        for (row, col, value) in lhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) += *value;
        }
        for (row, col, value) in rhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) -= *value;
        }
        entries
            .values()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn matern_precision_2form_top_degree_has_positive_diagonal_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("2-form Hodge Laplacian should assemble");
        let precision = build_matern_precision_2form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
            },
        )
        .expect("2-form precision should build on a surface mesh");

        assert!(diagonal_entries(&precision)
            .iter()
            .all(|value| *value > 0.0));
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("2-form top-degree precision should factorize");
    }

    #[test]
    fn matern_precision_2form_3d_has_positive_diagonal_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("3d 2-form Hodge Laplacian should assemble");
        let precision = build_matern_precision_2form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
            },
        )
        .expect("3d 2-form precision should build");

        assert!(diagonal_entries(&precision)
            .iter()
            .all(|value| *value > 0.0));
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("3d 2-form precision should factorize");
    }

    #[test]
    fn matern_precision_2form_barycentric_dual_sparse_inverse_factorizes_and_differs_from_projected(
    ) {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let projected_hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("projected 2-form Hodge Laplacian should assemble");
        let projected = build_matern_precision_2form(
            &topology,
            &metric,
            &projected_hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
            },
        )
        .expect("projected 2-form precision should build");

        let barycentric_hodge = build_hodge_laplacian_2form_with_lower_mass_inverse_coords(
            &topology,
            &coords,
            &metric,
            Matern1FormMassInverse::BarycentricDualSparseInverse,
        )
        .expect("barycentric lower 1-form inverse should build");
        let barycentric = build_matern_precision_2form_with_coords(
            &topology,
            &coords,
            &metric,
            &barycentric_hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::BarycentricDualSparseInverse,
            },
        )
        .expect("barycentric-dual 2-form precision should build");

        assert_eq!(projected.nrows(), barycentric.nrows());
        assert_eq!(projected.ncols(), barycentric.ncols());
        assert!(diagonal_entries(&barycentric)
            .iter()
            .all(|value| *value > 0.0));
        feec_csr_to_gmrf(&barycentric)
            .cholesky_sqrt_lower()
            .expect("barycentric-dual 2-form precision should factorize");
        assert!(max_abs_entry_diff(&projected, &barycentric) > 1e-9);
    }

    #[test]
    fn hodge_laplacian_2form_accepts_supplied_exact_lower_mass_inverse() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let projected_hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("projected 2-form Hodge Laplacian should assemble");
        let lower_hodge =
            crate::prior::matern::one_form::build_hodge_laplacian_1form(&topology, &metric);
        let exact_lower_inverse =
            crate::prior::matern::one_form::build_exact_dense_mass_inverse_1form(
                &lower_hodge.mass_u,
                0.0,
            )
            .expect("exact lower 1-form inverse should build");
        let exact_lower_hodge = build_hodge_laplacian_2form_with_lower_mass_inverse_matrix(
            &topology,
            &metric,
            &exact_lower_inverse,
        )
        .expect("exact-lower 2-form Hodge Laplacian should assemble");

        assert_eq!(
            projected_hodge.mass_u.nrows(),
            exact_lower_hodge.mass_u.nrows()
        );
        assert_eq!(
            projected_hodge.laplacian.nrows(),
            exact_lower_hodge.laplacian.nrows()
        );
        assert!(
            max_abs_entry_diff(&projected_hodge.laplacian, &exact_lower_hodge.laplacian) > 1e-9
        );

        let precision = build_matern_precision_2form_with_coords(
            &topology,
            &coords,
            &metric,
            &exact_lower_hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
            },
        )
        .expect("exact-lower 2-form precision should build");
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("exact-lower 2-form precision should factorize");
    }

    #[test]
    fn barycentric_dual_2form_mass_inverse_requires_coordinates() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("3d 2-form Hodge Laplacian should assemble");

        let err = build_matern_mass_inverse_2form(
            &topology,
            &metric,
            &hodge.mass_u,
            MaternMassInverse::BarycentricDualSparseInverse,
        )
        .expect_err("barycentric-dual 2-form inverse should require coordinates");

        assert!(err.contains("requires MeshCoords"));
    }

    #[test]
    fn alpha_one_precision_2form_is_scaled_whittle_system_matrix() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("2-form Hodge Laplacian should assemble");
        let kappa = 1.25;
        let tau = 0.75;
        let precision = build_matern_precision_2form_for_alpha(
            &topology,
            &metric,
            &hodge,
            MaternAlpha::One,
            MaternConfig {
                kappa,
                tau,
                mass_inverse: MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
            },
        )
        .expect("2-form alpha=1 precision should build");
        let expected = scale_matrix(&build_matern_system_matrix_2form(&hodge, kappa), tau * tau);

        assert_eq!(precision.nrows(), expected.nrows());
        assert_eq!(precision.ncols(), expected.ncols());
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("2-form alpha=1 precision should factorize");
    }
}
