use super::{build_lindgren_precision_from_system, MaternAlpha};
use crate::sparse::{add_sparse, diag_matrix, invert_diag, matrix_diag, scale_matrix};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr};
use formoniq::{
    assemble::assemble_whitney_projected_sparse_inverse_galmat_for_grade,
    problems::hodge_laplace::MixedGalmats,
};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

pub struct HodgeLaplacianForm {
    pub grade: usize,
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

pub fn build_hodge_laplacian_form(
    topology: &Complex,
    metric: &MeshLengths,
    grade: usize,
) -> Result<HodgeLaplacianForm, String> {
    if grade > topology.dim() {
        return Err(format!(
            "form grade {grade} exceeds topology dimension {}",
            topology.dim()
        ));
    }
    let galmats = MixedGalmats::compute(topology, metric, grade);
    build_hodge_laplacian_form_from_galmats(topology, metric, grade, &galmats)
}

pub fn build_hodge_laplacian_form_from_galmats(
    topology: &Complex,
    metric: &MeshLengths,
    grade: usize,
    galmats: &MixedGalmats,
) -> Result<HodgeLaplacianForm, String> {
    if grade > topology.dim() {
        return Err(format!(
            "form grade {grade} exceeds topology dimension {}",
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
        let sigma_inverse =
            build_projected_or_top_degree_mass_inverse(topology, metric, grade - 1, &mass_sigma)?;
        let dif_sigma = FeecCsr::from(galmats.dif_sigma());
        let codif_u = FeecCsr::from(galmats.codif_u());
        let schur_mid = &dif_sigma * &sigma_inverse;
        let schur = schur_mid * &codif_u;
        add_sparse(&codifdif_u, &schur)
    };

    Ok(HodgeLaplacianForm {
        grade,
        mass_u,
        laplacian,
    })
}

pub fn build_matern_system_matrix_form(hodge: &HodgeLaplacianForm, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&hodge.laplacian, &scale_matrix(&hodge.mass_u, kappa2))
}

pub fn build_projected_or_top_degree_mass_inverse(
    topology: &Complex,
    metric: &MeshLengths,
    grade: usize,
    mass: &FeecCsr,
) -> Result<FeecCsr, String> {
    if grade > topology.dim() {
        return Err(format!(
            "mass inverse grade {grade} exceeds topology dimension {}",
            topology.dim()
        ));
    }

    let inverse = if grade == topology.dim() {
        diag_matrix(&invert_diag(&matrix_diag(mass)))
    } else {
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_for_grade(
            topology, metric, grade,
        ))
    };

    if inverse.nrows() != mass.nrows() || inverse.ncols() != mass.ncols() {
        return Err(format!(
            "mass inverse dimensions {}x{} do not match grade-{grade} mass {}x{}",
            inverse.nrows(),
            inverse.ncols(),
            mass.nrows(),
            mass.ncols()
        ));
    }
    Ok(inverse)
}

pub fn build_matern_precision_form_for_alpha(
    topology: &Complex,
    metric: &MeshLengths,
    hodge: &HodgeLaplacianForm,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    let system = build_matern_system_matrix_form(hodge, config.kappa);
    if alpha == MaternAlpha::One {
        return Ok(build_lindgren_precision_from_system(
            &system,
            &hodge.mass_u,
            alpha,
            config.tau,
        ));
    }

    let mass_inverse =
        build_projected_or_top_degree_mass_inverse(topology, metric, hodge.grade, &hodge.mass_u)?;
    Ok(build_lindgren_precision_from_system(
        &system,
        &mass_inverse,
        alpha,
        config.tau,
    ))
}

pub fn build_matern_precision_form(
    topology: &Complex,
    metric: &MeshLengths,
    grade: usize,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    let hodge = build_hodge_laplacian_form(topology, metric, grade)?;
    build_matern_precision_form_for_alpha(topology, metric, &hodge, alpha, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::feec_csr_to_gmrf;
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

    #[test]
    fn projected_mass_inverse_dimensions_match_4d_forms() {
        let mesh = CartesianMeshInfo::new_unit_scaled(4, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        for grade in 0..topology.dim() {
            let hodge = build_hodge_laplacian_form(&topology, &metric, grade)
                .expect("4D Hodge Laplacian should assemble");
            let inverse = build_projected_or_top_degree_mass_inverse(
                &topology,
                &metric,
                grade,
                &hodge.mass_u,
            )
            .expect("projected inverse should assemble");
            assert_eq!(inverse.nrows(), hodge.mass_u.nrows());
            assert_eq!(inverse.ncols(), hodge.mass_u.ncols());
        }
    }

    #[test]
    fn four_dimensional_alpha_precisions_factorize() {
        let mesh = CartesianMeshInfo::new_unit_scaled(4, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        for grade in 0..=topology.dim() {
            let hodge = build_hodge_laplacian_form(&topology, &metric, grade)
                .expect("4D Hodge Laplacian should assemble");
            for alpha in [MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three] {
                let precision = build_matern_precision_form_for_alpha(
                    &topology,
                    &metric,
                    &hodge,
                    alpha,
                    MaternConfig {
                        kappa: 1.25,
                        tau: 1.0,
                    },
                )
                .expect("4D precision should build");
                assert_eq!(precision.nrows(), topology.nsimplices(grade));
                assert_eq!(precision.ncols(), topology.nsimplices(grade));
                assert!(diagonal_entries(&precision)
                    .iter()
                    .all(|value| *value > 0.0));
                feec_csr_to_gmrf(&precision)
                    .cholesky_sqrt_lower()
                    .expect("4D precision should factorize");
            }
        }
    }
}
