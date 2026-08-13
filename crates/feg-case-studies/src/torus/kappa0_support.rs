use crate::torus::one_form_conditioning::Torus1FormAmbientVarianceFields;
use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use formoniq::io::sample_1form_cell_vectors;
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

#[cfg(test)]
use gmrf_core::types::{
    CooMatrix as GmrfCooMatrix, DenseMatrix as GmrfDenseMatrix, SparseMatrix as GmrfSparseMatrix,
    Vector as GmrfVector,
};
#[cfg(test)]
use gmrf_core::GmrfError;

#[cfg(test)]
type ConstrainedKktSolver = gmrf_core::ConstrainedPrecisionSolver;

#[derive(Debug, Clone)]
pub(crate) struct SurfaceVectorSummaryStats {
    pub posterior_mean_magnitude_mean: f64,
    pub posterior_mean_magnitude_max: f64,
    pub vector_error_magnitude_mean: f64,
    pub vector_error_magnitude_max: f64,
    pub marginal_variance_mean: f64,
    pub marginal_variance_max: f64,
    pub marginal_variance_ratio_mean: f64,
    pub marginal_variance_ratio_min: f64,
    pub marginal_variance_ratio_max: f64,
    pub prior_directional_variance_mean: [f64; 3],
    pub posterior_directional_variance_mean: [f64; 3],
}

pub(crate) fn write_surface_vector_stats(
    coords: &MeshCoords,
    topology: &Complex,
    posterior_mean: &FeecVector,
    truth: &FeecVector,
    surface: &Torus1FormAmbientVarianceFields,
    out_dir: &Path,
    summary_path: &Path,
) -> Result<SurfaceVectorSummaryStats, Box<dyn Error>> {
    let posterior_cochain = Cochain::new(1, posterior_mean.clone());
    let truth_cochain = Cochain::new(1, truth.clone());
    let posterior_vectors = sample_1form_cell_vectors(coords, topology, &posterior_cochain)?;
    let truth_vectors = sample_1form_cell_vectors(coords, topology, &truth_cochain)?;
    let posterior_magnitudes = vector_magnitudes(&posterior_vectors);
    let mut error_vectors = Vec::with_capacity(posterior_vectors.len());
    for i in 0..posterior_vectors.len() {
        error_vectors.push([
            posterior_vectors[i][0] - truth_vectors[i][0],
            posterior_vectors[i][1] - truth_vectors[i][1],
            posterior_vectors[i][2] - truth_vectors[i][2],
        ]);
    }
    let error_magnitudes = vector_magnitudes(&error_vectors);

    let csv_path = out_dir.join("surface_vector_stats.csv");
    let file = File::create(csv_path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "cell_index,posterior_mean_x,posterior_mean_y,posterior_mean_z,posterior_mean_magnitude,truth_x,truth_y,truth_z,error_x,error_y,error_z,error_magnitude,prior_marginal_variance,posterior_marginal_variance,marginal_variance_ratio,prior_directional_variance_x,prior_directional_variance_y,prior_directional_variance_z,posterior_directional_variance_x,posterior_directional_variance_y,posterior_directional_variance_z"
    )?;
    for i in 0..posterior_vectors.len() {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            i,
            posterior_vectors[i][0],
            posterior_vectors[i][1],
            posterior_vectors[i][2],
            posterior_magnitudes[i],
            truth_vectors[i][0],
            truth_vectors[i][1],
            truth_vectors[i][2],
            error_vectors[i][0],
            error_vectors[i][1],
            error_vectors[i][2],
            error_magnitudes[i],
            surface.trace.prior[i],
            surface.trace.posterior[i],
            surface.trace.ratio[i],
            surface.x.prior[i],
            surface.y.prior[i],
            surface.z.prior[i],
            surface.x.posterior[i],
            surface.y.posterior[i],
            surface.z.posterior[i],
        )?;
    }

    let stats = SurfaceVectorSummaryStats {
        posterior_mean_magnitude_mean: mean_f64(&posterior_magnitudes),
        posterior_mean_magnitude_max: max_f64(&posterior_magnitudes),
        vector_error_magnitude_mean: mean_f64(&error_magnitudes),
        vector_error_magnitude_max: max_f64(&error_magnitudes),
        marginal_variance_mean: mean_vector(&surface.trace.posterior),
        marginal_variance_max: max_vector(&surface.trace.posterior),
        marginal_variance_ratio_mean: mean_vector(&surface.trace.ratio),
        marginal_variance_ratio_min: min_vector(&surface.trace.ratio),
        marginal_variance_ratio_max: max_vector(&surface.trace.ratio),
        prior_directional_variance_mean: [
            mean_vector(&surface.x.prior),
            mean_vector(&surface.y.prior),
            mean_vector(&surface.z.prior),
        ],
        posterior_directional_variance_mean: [
            mean_vector(&surface.x.posterior),
            mean_vector(&surface.y.posterior),
            mean_vector(&surface.z.posterior),
        ],
    };

    let mut summary = OpenOptions::new().append(true).open(summary_path)?;
    writeln!(
        summary,
        "surface_vector_posterior_mean_magnitude_mean={}",
        stats.posterior_mean_magnitude_mean
    )?;
    writeln!(
        summary,
        "surface_vector_posterior_mean_magnitude_max={}",
        stats.posterior_mean_magnitude_max
    )?;
    writeln!(
        summary,
        "surface_vector_error_magnitude_mean={}",
        stats.vector_error_magnitude_mean
    )?;
    writeln!(
        summary,
        "surface_vector_error_magnitude_max={}",
        stats.vector_error_magnitude_max
    )?;
    writeln!(
        summary,
        "surface_vector_marginal_variance_mean={}",
        stats.marginal_variance_mean
    )?;
    writeln!(
        summary,
        "surface_vector_marginal_variance_max={}",
        stats.marginal_variance_max
    )?;
    writeln!(
        summary,
        "surface_vector_marginal_variance_ratio_mean={}",
        stats.marginal_variance_ratio_mean
    )?;
    writeln!(
        summary,
        "surface_vector_marginal_variance_ratio_min={}",
        stats.marginal_variance_ratio_min
    )?;
    writeln!(
        summary,
        "surface_vector_marginal_variance_ratio_max={}",
        stats.marginal_variance_ratio_max
    )?;
    writeln!(
        summary,
        "surface_vector_prior_directional_variance_mean={},{},{}",
        stats.prior_directional_variance_mean[0],
        stats.prior_directional_variance_mean[1],
        stats.prior_directional_variance_mean[2]
    )?;
    writeln!(
        summary,
        "surface_vector_posterior_directional_variance_mean={},{},{}",
        stats.posterior_directional_variance_mean[0],
        stats.posterior_directional_variance_mean[1],
        stats.posterior_directional_variance_mean[2]
    )?;

    Ok(stats)
}

fn vector_magnitudes(vectors: &[[f64; 3]]) -> Vec<f64> {
    vectors
        .iter()
        .map(|[x, y, z]| (x * x + y * y + z * z).sqrt())
        .collect()
}

fn mean_vector(values: &FeecVector) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_vector(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn min_vector(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_core::solver::{IterativeMethod, PreconditionerKind, SolverAlgorithm, SolverConfig};
    use gmrf_core::{Gmrf, LinearOperator, SparseRowOperator};

    const SOLVER_TOLERANCE: f64 = 1e-10;

    #[derive(Clone)]
    struct RegularizedSparsePrecisionOperator {
        base_precision: GmrfSparseMatrix,
        constraint_matrix: GmrfDenseMatrix,
        alpha: f64,
    }

    impl RegularizedSparsePrecisionOperator {
        fn new(
            base_precision: GmrfSparseMatrix,
            constraint_matrix: GmrfDenseMatrix,
            alpha: f64,
        ) -> Result<Self, GmrfError> {
            if base_precision.nrows() != base_precision.ncols() {
                return Err(GmrfError::DimensionMismatch(
                    "regularized precision must be square",
                ));
            }
            if constraint_matrix.ncols() != base_precision.nrows() {
                return Err(GmrfError::DimensionMismatch(
                    "constraint matrix columns must match precision dimension",
                ));
            }
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(GmrfError::DimensionMismatch(
                    "regularization weight must be finite and positive",
                ));
            }
            Ok(Self {
                base_precision,
                constraint_matrix,
                alpha,
            })
        }
    }

    impl LinearOperator for RegularizedSparsePrecisionOperator {
        fn dimension(&self) -> usize {
            self.base_precision.nrows()
        }

        fn apply(&self, x: &GmrfVector) -> Result<GmrfVector, GmrfError> {
            if x.len() != self.dimension() {
                return Err(GmrfError::DimensionMismatch(
                    "operator input length must match precision dimension",
                ));
            }

            let mut out = self.base_precision.mul_vec(x);
            if self.constraint_matrix.nrows() > 0 {
                let constrained = dense_matvec(&self.constraint_matrix, x);
                let penalty = dense_transpose_matvec(&self.constraint_matrix, &constrained);
                out += self.alpha * penalty;
            }
            Ok(out)
        }
    }

    fn kappa0_solver_config(dimension: usize) -> SolverConfig {
        SolverConfig {
            algorithm: SolverAlgorithm::Iterative(IterativeMethod::ConjugateGradient),
            tolerance: SOLVER_TOLERANCE,
            max_iterations: (8 * dimension.max(1)).max(1024),
            preconditioner: PreconditionerKind::None,
        }
    }

    fn zero_mean_gmrf_with_operator(operator: RegularizedSparsePrecisionOperator) -> Gmrf {
        let dimension = operator.dimension();
        Gmrf::from_operator(GmrfVector::zeros(dimension), Box::new(operator))
            .with_solver_config(kappa0_solver_config(dimension))
    }

    fn gmrf_from_information_with_operator(
        information: GmrfVector,
        operator: RegularizedSparsePrecisionOperator,
    ) -> Result<Gmrf, GmrfError> {
        if information.len() != operator.dimension() {
            return Err(GmrfError::DimensionMismatch(
                "information vector length must match precision dimension",
            ));
        }

        let dimension = operator.dimension();
        let mut solver_gmrf = zero_mean_gmrf_with_operator(operator.clone());
        let mean = solver_gmrf.solve_precision(&information)?;
        Ok(Gmrf::from_operator(mean, Box::new(operator))
            .with_solver_config(kappa0_solver_config(dimension)))
    }

    fn constrained_mean_regularized(
        gmrf: &mut Gmrf,
        constraint_matrix: &GmrfDenseMatrix,
        constraint_rhs: &GmrfVector,
    ) -> Result<GmrfVector, GmrfError> {
        gmrf.constrained_mean(constraint_matrix, constraint_rhs)
    }

    fn dense_matvec(matrix: &GmrfDenseMatrix, vector: &GmrfVector) -> GmrfVector {
        let mut out = GmrfVector::zeros(matrix.nrows());
        for (j, col) in matrix.as_ref().col_iter().enumerate() {
            let xj = vector[j];
            if xj == 0.0 {
                continue;
            }
            let col = col
                .try_as_col_major()
                .expect("dense matrix is column-major");
            for (i, value) in col.as_slice().iter().enumerate() {
                out[i] += *value * xj;
            }
        }
        out
    }

    fn dense_transpose_matvec(matrix: &GmrfDenseMatrix, vector: &GmrfVector) -> GmrfVector {
        let mut out = GmrfVector::zeros(matrix.ncols());
        for (j, col) in matrix.as_ref().col_iter().enumerate() {
            let col = col
                .try_as_col_major()
                .expect("dense matrix is column-major");
            out[j] = col
                .as_slice()
                .iter()
                .enumerate()
                .map(|(i, value)| *value * vector[i])
                .sum::<f64>();
        }
        out
    }

    fn diagonal_sparse(diagonal: &[f64]) -> GmrfSparseMatrix {
        let mut coo = GmrfCooMatrix::new(diagonal.len(), diagonal.len());
        for (i, value) in diagonal.iter().copied().enumerate() {
            if value != 0.0 {
                coo.push(i, i, value);
            }
        }
        GmrfSparseMatrix::from(&coo)
    }

    fn constrained_identity_diag_regularized(
        base_precision: &GmrfSparseMatrix,
        constraints: &GmrfDenseMatrix,
        alpha: f64,
    ) -> GmrfVector {
        let operator = RegularizedSparsePrecisionOperator::new(
            base_precision.clone(),
            constraints.clone(),
            alpha,
        )
        .expect("operator should build");
        let dim = base_precision.nrows();
        let mut gmrf = zero_mean_gmrf_with_operator(operator);
        gmrf.exact_transformed_variance_decomposition(
            &SparseRowOperator::identity(dim),
            constraints,
        )
        .expect("regularized constrained variance should compute")
        .constrained_diag
    }

    fn constrained_identity_diag_kkt(
        solver: &ConstrainedKktSolver,
        latent_dim: usize,
    ) -> GmrfVector {
        let mut diagonal = GmrfVector::zeros(latent_dim);
        for i in 0..latent_dim {
            let mut rhs = GmrfVector::zeros(latent_dim);
            rhs[i] = 1.0;
            let solved = solver
                .solve_covariance_action(&rhs)
                .expect("KKT solve should succeed");
            diagonal[i] = solved[i].max(0.0);
        }
        diagonal
    }

    #[test]
    fn kkt_constrained_mean_matches_regularized_reference() {
        let base_precision = diagonal_sparse(&[2.0, 0.0]);
        let constraints = GmrfDenseMatrix::from_fn(1, 2, |_, col| if col == 1 { 1.0 } else { 0.0 });
        let constraint_rhs = GmrfVector::zeros(1);
        let information = GmrfVector::from_vec(vec![1.0, 3.0]);

        let mut regularized = gmrf_from_information_with_operator(
            information.clone(),
            RegularizedSparsePrecisionOperator::new(
                base_precision.clone(),
                constraints.clone(),
                1.0,
            )
            .expect("operator should build"),
        )
        .expect("regularized posterior should build");
        let mean_regularized =
            constrained_mean_regularized(&mut regularized, &constraints, &constraint_rhs).unwrap();

        let solver = ConstrainedKktSolver::new(&base_precision, &constraints)
            .expect("KKT solver should factorize");
        let mean_kkt = solver.solve_mean(&information).unwrap();

        assert!((mean_kkt - mean_regularized).norm() < 1e-8);
    }

    #[test]
    fn kkt_covariance_action_matches_regularized_reference() {
        let base_precision = diagonal_sparse(&[2.0, 0.0]);
        let constraints = GmrfDenseMatrix::from_fn(1, 2, |_, col| if col == 1 { 1.0 } else { 0.0 });

        let solver = ConstrainedKktSolver::new(&base_precision, &constraints)
            .expect("KKT solver should factorize");
        let diag_kkt = constrained_identity_diag_kkt(&solver, 2);
        let diag_regularized =
            constrained_identity_diag_regularized(&base_precision, &constraints, 1.0);

        assert!((diag_kkt - diag_regularized).norm() < 1e-8);
    }

    #[test]
    fn kkt_solver_handles_two_constraints() {
        let base_precision = diagonal_sparse(&[2.0, 3.0, 0.0, 0.0]);
        let constraints = GmrfDenseMatrix::from_fn(2, 4, |row, col| match (row, col) {
            (0, 2) => 1.0,
            (1, 3) => 1.0,
            _ => 0.0,
        });
        let information = GmrfVector::from_vec(vec![1.0, -2.0, 5.0, -7.0]);

        let solver = ConstrainedKktSolver::new(&base_precision, &constraints)
            .expect("KKT solver should factorize");
        let mean = solver.solve_mean(&information).unwrap();
        let diag = constrained_identity_diag_kkt(&solver, 4);

        assert!((mean[0] - 0.5).abs() < 1e-10);
        assert!((mean[1] + 2.0 / 3.0).abs() < 1e-10);
        assert!(mean[2].abs() < 1e-10);
        assert!(mean[3].abs() < 1e-10);
        assert!((diag[0] - 0.5).abs() < 1e-10);
        assert!((diag[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!(diag[2].abs() < 1e-10);
        assert!(diag[3].abs() < 1e-10);
    }
}
