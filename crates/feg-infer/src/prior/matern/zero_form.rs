use super::build_lindgren_precision_from_system;
use crate::sparse::{add_sparse, diag_matrix, invert_diag, lumped_diag, scale_matrix};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr};
use formoniq::problems::laplace_beltrami::LaplaceBeltramiGalmats;
use gmrf_core::Vector as GmrfVector;
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

pub use super::MaternAlpha;
pub use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf};

pub struct LaplaceBeltrami0Form {
    pub mass: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaternMassInverse {
    #[default]
    RowSumLumped,
}

#[derive(Debug, Clone, Copy)]
pub struct MaternConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: MaternMassInverse,
}

pub fn build_laplace_beltrami_0form(
    topology: &Complex,
    metric: &MeshLengths,
) -> LaplaceBeltrami0Form {
    let galmats = LaplaceBeltramiGalmats::compute(topology, metric);
    build_laplace_beltrami_0form_from_galmats(&galmats)
}

pub fn build_laplace_beltrami_0form_from_galmats(
    galmats: &LaplaceBeltramiGalmats,
) -> LaplaceBeltrami0Form {
    LaplaceBeltrami0Form {
        mass: galmats.mass_csr(),
        laplacian: galmats.stiffness_csr(),
    }
}

pub fn build_matern_system_matrix_0form(laplace: &LaplaceBeltrami0Form, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&laplace.laplacian, &scale_matrix(&laplace.mass, kappa2))
}

pub fn build_matern_mass_inverse_0form(mass: &FeecCsr, strategy: MaternMassInverse) -> FeecCsr {
    match strategy {
        MaternMassInverse::RowSumLumped => diag_matrix(&invert_diag(&lumped_diag(mass))),
    }
}

pub fn build_exact_dense_mass_inverse_0form(
    mass: &FeecCsr,
    drop_tolerance: f64,
) -> Result<FeecCsr, String> {
    if mass.nrows() != mass.ncols() {
        return Err(format!(
            "exact 0-form mass inverse requires a square matrix, got {}x{}",
            mass.nrows(),
            mass.ncols()
        ));
    }
    let factor = feec_csr_to_gmrf(mass)
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor 0-form mass matrix: {err}"))?;
    let n = mass.nrows();
    let tolerance = drop_tolerance.abs();
    let mut inverse = FeecCoo::new(n, n);
    for col in 0..n {
        let mut rhs = GmrfVector::zeros(n);
        rhs[col] = 1.0;
        let solution = factor.solve(&rhs).map_err(|err| {
            format!("failed to solve exact 0-form mass inverse column {col}: {err}")
        })?;
        for (row, value) in solution.iter().copied().enumerate() {
            if value.abs() > tolerance {
                inverse.push(row, col, value);
            }
        }
    }
    Ok(FeecCsr::from(&inverse))
}

pub fn build_matern_precision_0form(
    laplace: &LaplaceBeltrami0Form,
    config: MaternConfig,
) -> FeecCsr {
    build_matern_precision_0form_for_alpha(laplace, MaternAlpha::Two, config)
}

pub fn build_matern_precision_0form_for_alpha(
    laplace: &LaplaceBeltrami0Form,
    alpha: MaternAlpha,
    config: MaternConfig,
) -> FeecCsr {
    let a = build_matern_system_matrix_0form(laplace, config.kappa);
    let mass_inverse = build_matern_mass_inverse_0form(&laplace.mass, config.mass_inverse);
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

    #[test]
    fn exact_dense_mass_inverse_0form_inverts_mass_matrix() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let laplace = build_laplace_beltrami_0form(&topology, &metric);
        let inverse = build_exact_dense_mass_inverse_0form(&laplace.mass, 0.0)
            .expect("0-form mass inverse should factorize");
        let product = &laplace.mass * &inverse;
        for row in 0..product.nrows() {
            for col in 0..product.ncols() {
                let mut value = 0.0;
                for (i, j, entry) in product.triplet_iter() {
                    if i == row && j == col {
                        value += *entry;
                    }
                }
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-9,
                    "M M^-1 mismatch at ({row},{col}): got {value}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn laplace_beltrami_0form_dimensions() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let laplace = build_laplace_beltrami_0form(&topology, &metric);

        assert!(laplace.mass.nrows() > 0);
        assert_eq!(laplace.mass.nrows(), laplace.mass.ncols());
        assert_eq!(laplace.laplacian.nrows(), laplace.mass.nrows());
        assert_eq!(laplace.laplacian.ncols(), laplace.mass.ncols());
    }

    #[test]
    fn matern_precision_0form_has_positive_diagonal_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let laplace = build_laplace_beltrami_0form(&topology, &metric);
        let precision = build_matern_precision_0form(
            &laplace,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );

        assert!(diagonal_entries(&precision).iter().all(|v| *v > 0.0));
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("0-form precision should factorize");
    }

    #[test]
    fn alpha_one_precision_0form_is_scaled_whittle_system_matrix() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let laplace = build_laplace_beltrami_0form(&topology, &metric);
        let kappa = 1.5;
        let tau = 0.75;
        let precision = build_matern_precision_0form_for_alpha(
            &laplace,
            MaternAlpha::One,
            MaternConfig {
                kappa,
                tau,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let expected = scale_matrix(
            &build_matern_system_matrix_0form(&laplace, kappa),
            tau * tau,
        );

        assert_eq!(precision.nrows(), expected.nrows());
        assert_eq!(precision.ncols(), expected.ncols());
        for (row, col, expected_value) in expected.triplet_iter() {
            let actual = precision
                .triplet_iter()
                .find_map(|(r, c, value)| (r == row && c == col).then_some(*value));
            assert!((actual.unwrap_or(0.0) - expected_value).abs() <= 1e-12);
        }
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("0-form alpha=1 precision should factorize");
    }
}
