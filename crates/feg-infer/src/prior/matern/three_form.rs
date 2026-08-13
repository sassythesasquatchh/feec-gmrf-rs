use crate::prior::matern::{
    build_lindgren_precision_from_system,
    two_form::{
        build_matern_mass_inverse_2form, build_matern_mass_inverse_2form_with_coords,
        MaternMassInverse as Matern2FormMassInverse,
    },
};
use crate::sparse::{add_sparse, diag_matrix, invert_diag, matrix_diag, scale_matrix};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr};
use formoniq::problems::hodge_laplace::MixedGalmats;
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};

pub use super::MaternAlpha;
pub use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf};

pub struct HodgeLaplacian3Form {
    pub mass_u: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone, Copy)]
pub struct MaternConfig {
    pub kappa: f64,
    pub tau: f64,
}

impl Default for MaternConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            tau: 1.0,
        }
    }
}

pub fn build_hodge_laplacian_3form(
    topology: &Complex,
    metric: &MeshLengths,
) -> Result<HodgeLaplacian3Form, String> {
    let galmats = MixedGalmats::compute(topology, metric, 3);
    build_hodge_laplacian_3form_from_galmats(topology, metric, &galmats)
}

pub fn build_hodge_laplacian_3form_with_lower_mass_inverse_coords(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    lower_mass_inverse: Matern2FormMassInverse,
) -> Result<HodgeLaplacian3Form, String> {
    let galmats = MixedGalmats::compute(topology, metric, 3);
    build_hodge_laplacian_3form_from_galmats_with_optional_coords(
        topology,
        Some(coords),
        metric,
        &galmats,
        lower_mass_inverse,
    )
}

pub fn build_hodge_laplacian_3form_from_galmats(
    topology: &Complex,
    metric: &MeshLengths,
    galmats: &MixedGalmats,
) -> Result<HodgeLaplacian3Form, String> {
    build_hodge_laplacian_3form_from_galmats_with_optional_coords(
        topology,
        None,
        metric,
        galmats,
        Matern2FormMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
    )
}

fn build_hodge_laplacian_3form_from_galmats_with_optional_coords(
    topology: &Complex,
    coords: Option<&MeshCoords>,
    metric: &MeshLengths,
    galmats: &MixedGalmats,
    lower_mass_inverse: Matern2FormMassInverse,
) -> Result<HodgeLaplacian3Form, String> {
    if topology.dim() != 3 {
        return Err(format!(
            "3-form Matérn prior requires a 3D topology, got {}D",
            topology.dim()
        ));
    }

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
            build_matern_mass_inverse_2form_with_coords(
                topology,
                coords,
                metric,
                &mass_sigma,
                lower_mass_inverse,
            )?
        } else {
            build_matern_mass_inverse_2form(topology, metric, &mass_sigma, lower_mass_inverse)?
        };
        let dif_sigma = FeecCsr::from(galmats.dif_sigma());
        let codif_u = FeecCsr::from(galmats.codif_u());
        let schur_mid = &dif_sigma * &sigma_inverse;
        let schur = schur_mid * &codif_u;
        add_sparse(&codifdif_u, &schur)
    };

    Ok(HodgeLaplacian3Form { mass_u, laplacian })
}

pub fn build_matern_system_matrix_3form(hodge: &HodgeLaplacian3Form, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&hodge.laplacian, &scale_matrix(&hodge.mass_u, kappa2))
}

pub fn build_matern_mass_inverse_3form(mass_u: &FeecCsr) -> FeecCsr {
    diag_matrix(&invert_diag(&matrix_diag(mass_u)))
}

pub fn build_matern_precision_3form(hodge: &HodgeLaplacian3Form, config: MaternConfig) -> FeecCsr {
    build_matern_precision_3form_for_alpha(hodge, MaternAlpha::Two, config)
}

pub fn build_matern_precision_3form_for_alpha(
    hodge: &HodgeLaplacian3Form,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> FeecCsr {
    let a = build_matern_system_matrix_3form(hodge, config.kappa);
    let mass_inverse = build_matern_mass_inverse_3form(&hodge.mass_u);
    build_lindgren_precision_from_system(&a, &mass_inverse, alpha, config.tau)
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

    fn max_off_diagonal_abs(mat: &FeecCsr) -> f64 {
        mat.triplet_iter()
            .filter(|(row, col, _)| row != col)
            .map(|(_, _, value)| value.abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn three_form_alpha_one_and_two_factorize_with_projected_lower_inverse() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_3form(&topology, &metric)
            .expect("3-form Hodge Laplacian should assemble");

        for alpha in [MaternAlpha::One, MaternAlpha::Two] {
            let precision = build_matern_precision_3form_for_alpha(
                &hodge,
                alpha,
                MaternConfig {
                    kappa: 1.25,
                    tau: 1.0,
                },
            );
            assert_eq!(precision.nrows(), topology.nsimplices(3));
            assert_eq!(precision.ncols(), topology.nsimplices(3));
            assert!(diagonal_entries(&precision)
                .iter()
                .all(|value| *value > 0.0));
            feec_csr_to_gmrf(&precision)
                .cholesky_sqrt_lower()
                .expect("3-form precision should factorize");
        }
    }

    #[test]
    fn three_form_barycentric_lower_inverse_builds_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_3form_with_lower_mass_inverse_coords(
            &topology,
            &coords,
            &metric,
            Matern2FormMassInverse::BarycentricDualSparseInverse,
        )
        .expect("3-form Hodge Laplacian should assemble with barycentric lower inverse");
        let precision = build_matern_precision_3form_for_alpha(
            &hodge,
            MaternAlpha::Two,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
            },
        );

        assert_eq!(precision.nrows(), topology.nsimplices(3));
        assert_eq!(precision.ncols(), topology.nsimplices(3));
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("barycentric-lower 3-form precision should factorize");
    }

    #[test]
    fn three_form_top_degree_mass_is_diagonal() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_3form(&topology, &metric)
            .expect("3-form Hodge Laplacian should assemble");

        assert!(max_off_diagonal_abs(&hodge.mass_u) <= 1e-12);
        assert!(diagonal_entries(&hodge.mass_u)
            .iter()
            .all(|value| *value > 0.0));
    }
}
