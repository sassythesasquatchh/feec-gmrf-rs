use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::cochain::Cochain;
use exterior::field::DiffFormClosure;
use feg_case_studies::visual_output;
use feg_infer::prior::matern::one_form::{
    build_matern_precision_1form, feec_csr_to_gmrf, feec_vec_to_gmrf, HodgeLaplacian1Form,
    MaternConfig, MaternMassInverse,
};
use feg_infer::prior::matern::convert_whittle_params_to_matern;
use formoniq::{
    assemble::{self, assemble_galvec},
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::hodge_laplace,
};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::Vector as GmrfVector;
use gmrf_core::Gmrf;
use manifold::geometry::coord::CoordRef;
use manifold::topology::handle::KSimplexIdx;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "out/magnetostatic_posterior";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    let target_air_cell_size = 0.25;
    let mesh_path = "meshes/toroidal_inductor.msh";
    let mesh_bytes = fs::read(mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);

    let dim = 3;

    let kappa = 10.;
    let tau = 1.;
    let alpha = 2.;
    let (_, _, effective_range) = convert_whittle_params_to_matern(alpha, tau, kappa, dim);

    println!("Effective range: {}", effective_range);

    let mu_0 = 4e-7 * std::f64::consts::PI;
    let mu_0_inverse = 1.0 / mu_0;
    let inverse_permeability = InnerProductWeightClosure::new(move |_| mu_0_inverse);

    // Magnetostatic setup copied from formoniq/examples/magnetostatic.rs.
    let major_radius = 2.0;
    let core_minor_radius = 0.60;
    let coil_minor_radius = 0.85;
    let box_half_length = 6.0;

    // Homogeneous essential BCs near the outer boundary.
    let strong_dof_predicate = |p: CoordRef| {
        let d = (box_half_length - p[0].abs())
            .min(box_half_length - p[1].abs())
            .min(box_half_length - p[2].abs());
        d < 5.0 * 2.0 * target_air_cell_size
    };

    let strong_k_dofs =
        assemble::boundary_simplices_where_barycenter(&topology, &coords, 1, strong_dof_predicate)
            .into_iter()
            .collect::<HashSet<usize>>();

    let strong_k_minus_one_dofs =
        assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, strong_dof_predicate)
            .into_iter()
            .collect::<HashSet<usize>>();

    let strong_k_dof_predicate = |sidx: KSimplexIdx| strong_k_dofs.contains(&sidx);
    let strong_k_minus_one_dof_predicate =
        |sidx: KSimplexIdx| strong_k_minus_one_dofs.contains(&sidx);
    let homogeneous_bc_data = |_idx: KSimplexIdx| 0.0;

    let j0: f64 = 1.0;
    let sigma: f64 = 0.18;
    let eps: f64 = 0.03;
    let current_density = DiffFormClosure::one_form(
        move |p| {
            let x = p[0];
            let y = p[1];
            let z = p[2];

            let rho = (x * x + y * y).sqrt();
            if rho < 1e-12 {
                return FeecVector::from_column_slice(&[0.0, 0.0, 0.0]);
            }

            let s = ((rho - major_radius).powi(2) + z * z).sqrt();
            let smoothstep = |t: f64| t * t * (3.0 - 2.0 * t);

            let inner = core_minor_radius + eps;
            let outer = coil_minor_radius - eps;
            let tin = ((s - inner) / eps).clamp(0.0, 1.0);
            let tout = ((outer - s) / eps).clamp(0.0, 1.0);
            let cutoff = smoothstep(tin) * smoothstep(tout);

            let s0 = 0.5 * (core_minor_radius + coil_minor_radius);
            let gauss = (-((s - s0) * (s - s0)) / (sigma * sigma)).exp();

            let amp = mu_0 * j0 * gauss * cutoff;
            FeecVector::from_column_slice(&[amp * (-y / rho), amp * (x / rho), 0.0])
        },
        3,
    );

    let source_galvec = assemble_galvec(
        &topology,
        &metric,
        SourceElVec::new_weighted(&current_density, &coords, None, &inverse_permeability),
    );

    // This problem has one harmonic 1-form.
    let homology_dim = 1;
    let (_sigma_sol, u_sol, _harmonics) =
        hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
            &topology,
            &metric,
            None,
            source_galvec.clone(),
            1,
            homology_dim,
            &coords,
            None,
            &inverse_permeability,
            &strong_k_dof_predicate,
            &homogeneous_bc_data,
            &strong_k_minus_one_dof_predicate,
            &homogeneous_bc_data,
        );

    // Build BC-aware reduced operators for inference (same BCs as FEEC solve).
    let galmats = hodge_laplace::MixedGalmats::compute_weighted(
        &topology,
        &metric,
        1,
        &coords,
        None,
        &inverse_permeability,
    );
    let (laplacian_reduced, mass_u_reduced) =
        reduced_hodge_laplacian_1form_with_bc(&galmats, &strong_k_minus_one_dofs, &strong_k_dofs);

    let mut strong_k_dofs_sorted = strong_k_dofs.iter().copied().collect::<Vec<_>>();
    strong_k_dofs_sorted.sort_unstable();
    let mut source_reduced = source_galvec.clone();
    assemble::drop_dofs_galvec(&strong_k_dofs_sorted, &mut source_reduced);

    let hodge = HodgeLaplacian1Form {
        mass_u: mass_u_reduced,
        laplacian: laplacian_reduced.clone(),
    };
    let prior_precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        MaternConfig {
            kappa,
            tau,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );
    let prior_precision = stabilize_feec_precision_for_cholesky("prior", prior_precision);

    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let observation_matrix = feec_csr_to_gmrf(&laplacian_reduced);
    let observations = feec_vec_to_gmrf(&source_reduced);
    let ndofs = laplacian_reduced.nrows();

    let q_prior_factor = q_prior.cholesky_sqrt_lower()?;
    let mut prior = Gmrf::from_mean_and_precision(GmrfVector::zeros(ndofs), q_prior.clone())?
        .with_precision_sqrt(q_prior_factor);

    let noise_variance = 1e-8;
    let (posterior_precision, information) = apply_gaussian_observations(
        &q_prior,
        &observation_matrix,
        &observations,
        None,
        noise_variance,
    );
    let q_factor = posterior_precision.cholesky_sqrt_lower()?;
    let mut posterior =
        Gmrf::from_information_and_precision_with_sqrt(information, posterior_precision, q_factor)?;

    let num_mc_samples = 64;
    let mut prior_rng = StdRng::seed_from_u64(11);
    let prior_variances_reduced = prior.mc_variances(num_mc_samples, &mut prior_rng)?;
    let mut posterior_rng = StdRng::seed_from_u64(17);
    let posterior_variances_reduced = posterior.mc_variances(num_mc_samples, &mut posterior_rng)?;

    let posterior_mean_reduced = gmrf_vec_to_feec(posterior.mean());
    let prior_mean_reduced = gmrf_vec_to_feec(prior.mean());
    let prior_var_reduced = gmrf_vec_to_feec(&prior_variances_reduced);
    let posterior_var_reduced = gmrf_vec_to_feec(&posterior_variances_reduced);

    let posterior_mean_full =
        reintroduce_homogeneous_dofs(posterior_mean_reduced.clone(), &strong_k_dofs_sorted);
    let prior_mean_full = reintroduce_homogeneous_dofs(prior_mean_reduced, &strong_k_dofs_sorted);
    let prior_var_full = reintroduce_homogeneous_dofs(prior_var_reduced, &strong_k_dofs_sorted);
    let posterior_var_full =
        reintroduce_homogeneous_dofs(posterior_var_reduced, &strong_k_dofs_sorted);

    let mut u_sol_reduced = u_sol.coeffs.clone();
    assemble::drop_dofs_galvec(&strong_k_dofs_sorted, &mut u_sol_reduced);
    let rel_l2 =
        (&posterior_mean_reduced - &u_sol_reduced).norm() / u_sol_reduced.norm().max(1e-12);

    let prior_mean_cochain = Cochain::new(1, prior_mean_full);
    let prior_var_cochain = Cochain::new(1, prior_var_full);
    let posterior_mean_cochain = Cochain::new(1, posterior_mean_full);
    let posterior_var_cochain = Cochain::new(1, posterior_var_full);
    let posterior_diff_cochain = Cochain::new(1, &posterior_mean_cochain.coeffs - &u_sol.coeffs);

    visual_output::write_cochain(
        format!("{out_dir}/magnetic_vector_potential_feec.vtu"),
        &coords,
        &topology,
        &u_sol,
        "A_feec",
    )?;
    visual_output::write_1form_vector_field(
        format!("{out_dir}/magnetic_vector_potential_feec_vector.vtu"),
        &coords,
        &topology,
        &u_sol,
        "A_feec_vec",
    )?;

    let b_feec = u_sol.dif(&topology);
    visual_output::write_cochain(
        format!("{out_dir}/magnetic_flux_density_feec.vtu"),
        &coords,
        &topology,
        &b_feec,
        "B_feec",
    )?;
    visual_output::write_2form_vector_field(
        format!("{out_dir}/magnetic_flux_density_feec_vector.vtu"),
        &coords,
        &topology,
        &b_feec,
        "B_feec_vec",
    )?;

    visual_output::write_1cochain_fields(
        format!("{out_dir}/prior_mean.vtu"),
        &coords,
        &topology,
        &[
            ("prior_mean", &prior_mean_cochain),
            ("prior_variance_mc", &prior_var_cochain),
        ],
    )?;
    visual_output::write_1form_vector_field(
        format!("{out_dir}/prior_mean_vector_field.vtu"),
        &coords,
        &topology,
        &prior_mean_cochain,
        "prior_mean_vector_field",
    )?;

    visual_output::write_1cochain_fields(
        format!("{out_dir}/posterior_mean.vtu"),
        &coords,
        &topology,
        &[
            ("posterior_mean", &posterior_mean_cochain),
            ("posterior_variance_mc", &posterior_var_cochain),
        ],
    )?;
    visual_output::write_1form_vector_field(
        format!("{out_dir}/posterior_mean_vector_field.vtu"),
        &coords,
        &topology,
        &posterior_mean_cochain,
        "posterior_mean_vector_field",
    )?;
    visual_output::write_cochain(
        format!("{out_dir}/posterior_mean_minus_feec.vtu"),
        &coords,
        &topology,
        &posterior_diff_cochain,
        "posterior_mean_minus_feec",
    )?;

    let b_posterior = posterior_mean_cochain.dif(&topology);
    visual_output::write_cochain(
        format!("{out_dir}/magnetic_flux_density_posterior.vtu"),
        &coords,
        &topology,
        &b_posterior,
        "B_posterior",
    )?;
    visual_output::write_2form_vector_field(
        format!("{out_dir}/magnetic_flux_density_posterior_vector.vtu"),
        &coords,
        &topology,
        &b_posterior,
        "B_posterior_vec",
    )?;

    println!("Loaded mesh from {mesh_path}");
    println!(
        "Strong BC dofs: sigma(k-1)={} u(k)={}",
        strong_k_minus_one_dofs.len(),
        strong_k_dofs.len()
    );
    println!("Reduced 1-form dofs for inference: {ndofs}");
    println!("Posterior mean reduced-space relative L2 error vs FEEC: {rel_l2:.3e}");
    println!("Homology dimension used in FEEC solve: {homology_dim}");
    println!("Wrote VTU outputs to {out_dir}");
    Ok(())
}

fn reduced_hodge_laplacian_1form_with_bc(
    galmats: &hodge_laplace::MixedGalmats,
    strong_k_minus_one_dofs: &HashSet<usize>,
    strong_k_dofs: &HashSet<usize>,
) -> (FeecCsr, FeecCsr) {
    let mut mass_sigma = galmats.mass_sigma().clone();
    let mut dif_sigma = galmats.dif_sigma().clone();
    let mut codif_u = galmats.codif_u().clone();
    let mut codifdif_u = galmats.codifdif_u().clone();
    let mut mass_u = galmats.mass_u().clone();

    assemble::drop_dofs_galmat(strong_k_minus_one_dofs, &mut mass_sigma);
    assemble::drop_dofs_rectangular_galmat(strong_k_dofs, strong_k_minus_one_dofs, &mut dif_sigma);
    assemble::drop_dofs_rectangular_galmat(strong_k_minus_one_dofs, strong_k_dofs, &mut codif_u);
    assemble::drop_dofs_galmat(strong_k_dofs, &mut codifdif_u);
    assemble::drop_dofs_galmat(strong_k_dofs, &mut mass_u);

    let mass_sigma = FeecCsr::from(&mass_sigma);
    let dif_sigma = FeecCsr::from(&dif_sigma);
    let codif_u = FeecCsr::from(&codif_u);
    let codifdif_u = FeecCsr::from(&codifdif_u);
    let mass_u = FeecCsr::from(&mass_u);

    if mass_sigma.nrows() == 0 {
        return (codifdif_u, mass_u);
    }

    // Lumped Schur complement on reduced spaces:
    // L = codifdif_u + dif_sigma * M_sigma^{-1} * codif_u
    let mass_sigma_inv_diag = invert_diag(&lumped_diag(&mass_sigma));
    let codif_u_scaled = scale_rows(&codif_u, &mass_sigma_inv_diag);
    let schur = &dif_sigma * &codif_u_scaled;
    let laplacian = add_sparse(&codifdif_u, &schur);
    (laplacian, mass_u)
}

fn reintroduce_homogeneous_dofs(mut reduced: FeecVector, prescribed_dofs: &[usize]) -> FeecVector {
    let dof_coeffs = prescribed_dofs
        .iter()
        .copied()
        .map(|idx| (idx, 0.0))
        .collect::<Vec<_>>();
    assemble::reintroduce_non_homogenous_dofs_galsols(&dof_coeffs, &mut reduced);
    reduced
}

fn gmrf_vec_to_feec(vec: &gmrf_core::types::Vector) -> FeecVector {
    FeecVector::from_vec(vec.iter().copied().collect())
}

fn lumped_diag(mat: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; mat.nrows()];
    for (row, _col, value) in mat.triplet_iter() {
        diag[row] += *value;
    }
    diag
}

fn invert_diag(diag: &[f64]) -> Vec<f64> {
    let eps = 1e-12;
    diag.iter()
        .map(|v| if v.abs() < eps { 0.0 } else { 1.0 / v })
        .collect()
}

fn scale_rows(mat: &FeecCsr, row_scales: &[f64]) -> FeecCsr {
    assert_eq!(mat.nrows(), row_scales.len());
    let mut coo = FeecCoo::new(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        let scaled = *value * row_scales[row];
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

fn add_sparse(a: &FeecCsr, b: &FeecCsr) -> FeecCsr {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());

    let mut coo = FeecCoo::new(a.nrows(), a.ncols());
    for (row, col, value) in a.triplet_iter() {
        coo.push(row, col, *value);
    }
    for (row, col, value) in b.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
}

fn stabilize_feec_precision_for_cholesky(label: &str, precision: FeecCsr) -> FeecCsr {
    if feec_csr_to_gmrf(&precision).cholesky_sqrt_lower().is_ok() {
        return precision;
    }

    let precision = symmetrize(&precision);
    if feec_csr_to_gmrf(&precision).cholesky_sqrt_lower().is_ok() {
        println!("{label} precision stabilized by symmetrization");
        return precision;
    }

    let diagonal = matrix_diag(&precision);
    let min_diag = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_diag = diagonal.iter().copied().map(f64::abs).fold(1.0, f64::max);
    let mut shift = if min_diag <= 0.0 {
        -min_diag + 1e-8 * max_abs_diag
    } else {
        1e-12 * max_abs_diag
    }
    .max(1e-10);

    for _ in 0..10 {
        let shifted = add_diagonal_shift(&precision, shift);
        if feec_csr_to_gmrf(&shifted).cholesky_sqrt_lower().is_ok() {
            println!("{label} precision stabilized by diagonal shift {shift:.6e}");
            return shifted;
        }
        shift *= 10.0;
    }

    println!("{label} precision required fallback diagonal shift {shift:.6e}");
    add_diagonal_shift(&precision, shift)
}

fn symmetrize(matrix: &FeecCsr) -> FeecCsr {
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            coo.push(row, col, *value);
        } else {
            coo.push(row, col, 0.5 * *value);
            coo.push(col, row, 0.5 * *value);
        }
    }
    FeecCsr::from(&coo)
}

fn matrix_diag(matrix: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diag[row] += *value;
        }
    }
    diag
}

fn add_diagonal_shift(matrix: &FeecCsr, shift: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for index in 0..matrix.nrows() {
        coo.push(index, index, shift);
    }
    FeecCsr::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reintroduce_homogeneous_dofs_inserts_zeros_at_fixed_indices() {
        let reduced = FeecVector::from_vec(vec![1.0, 2.0, 3.0]);
        let fixed = vec![1usize, 3usize];
        let full = reintroduce_homogeneous_dofs(reduced, &fixed);

        assert_eq!(full.len(), 5);
        assert_eq!(full[0], 1.0);
        assert_eq!(full[1], 0.0);
        assert_eq!(full[2], 2.0);
        assert_eq!(full[3], 0.0);
        assert_eq!(full[4], 3.0);
    }
}
