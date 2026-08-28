//! Spectral Gaussian process utilities based on Hodge Laplacian eigenfunctions.

#[cfg(feature = "external-solvers")]
mod hodge;

use common::linalg::nalgebra::quadratic_form_sparse;
use common::linalg::nalgebra::{CsrMatrix, Matrix as NaMatrix, Vector as NaVector};
#[cfg(feature = "external-solvers")]
use common::linalg::petsc::{GhiepReducedSolve, GhiepWhich};
#[cfg(feature = "external-solvers")]
use exterior::ExteriorGrade;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::Solve;
use faer::{Accum, Idx, Mat, Par, Side, Unbind};
#[cfg(feature = "external-solvers")]
use formoniq::{
    assemble,
    operators::HodgeMassElmat,
    problems::{hodge_laplace, laplace_beltrami},
};
use libm::tgamma;
#[cfg(feature = "external-solvers")]
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};
use std::cmp::Ordering;
use thiserror::Error;

#[cfg(feature = "external-solvers")]
pub use hodge::*;

pub const DEFAULT_K: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct SpectralMaternConfig {
    pub kappa: f64,
    pub alpha: f64,
    pub tau: f64,
    pub k: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EuclideanMaternConfig {
    pub kappa: f64,
    pub nu: f64,
    pub variance: f64,
}

impl Default for SpectralMaternConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: DEFAULT_K,
        }
    }
}

impl Default for EuclideanMaternConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            nu: 1.5,
            variance: 1.0,
        }
    }
}

#[derive(Debug, Error)]
pub enum GpError {
    #[error("k must be positive")]
    InvalidK,
    #[error("kappa must be positive")]
    InvalidKappa,
    #[error("nu must be positive")]
    InvalidNu,
    #[error("tau must be positive")]
    InvalidTau,
    #[error("alpha must be positive")]
    InvalidAlpha,
    #[error("variance must be non-negative")]
    InvalidVariance,
    #[error("expected mass energy must be finite and non-negative")]
    InvalidExpectedMassEnergy,
    #[error("points slice is empty")]
    EmptyPoints,
    #[error("points must have at least one dimension")]
    EmptyPointDimension,
    #[error("point dimension mismatch: expected {expected}, got {got}")]
    PointDimensionMismatch { expected: usize, got: usize },
    #[error("distance must be finite and non-negative")]
    InvalidDistance,
    #[error("unsupported nu {nu}; supported values are 0.5, 1.0, 1.5, 2.5")]
    UnsupportedNu { nu: f64 },
    #[error("eigenvalues length {values} does not match eigenvector columns {cols}")]
    EigenDimensionMismatch { values: usize, cols: usize },
    #[error("eigenvectors must have at least one row")]
    EmptyEigenvectors,
    #[error("non-finite eigenvalue {value}")]
    NonFiniteEigenvalue { value: f64 },
    #[error("invalid eigenvalue {value}: kappa^2 + lambda must be positive")]
    InvalidEigenvalue { value: f64 },
    #[error("invalid eigenvector mass norm {value}")]
    InvalidEigenvectorNorm { value: f64 },
    #[error(
        "mass matrix shape {rows}x{cols} does not match eigenvector row dimension {dimension}"
    )]
    EigenvectorMassDimensionMismatch {
        rows: usize,
        cols: usize,
        dimension: usize,
    },
    #[error("mass-orthogonality tolerance must be finite and non-negative")]
    InvalidMassOrthogonalityTolerance,
    #[error(
        "generalized eigenvectors {left} and {right} have mass inner product {value}, exceeding tolerance {tolerance}"
    )]
    EigenvectorsNotMassOrthogonal {
        left: usize,
        right: usize,
        value: f64,
        tolerance: f64,
    },
    #[error("covariance must be square, got {rows}x{cols}")]
    InvalidCovarianceShape { rows: usize, cols: usize },
    #[error("observation indices length {indices} does not match values length {values}")]
    ObservationCountMismatch { indices: usize, values: usize },
    #[error("observation index {index} out of bounds for size {n}")]
    ObservationIndexOutOfBounds { index: usize, n: usize },
    #[error("observation index {index} is duplicated")]
    ObservationIndexRepeated { index: usize },
    #[error("noise variance must be finite and non-negative")]
    InvalidNoiseVariance,
    #[error("covariance is not positive definite (cholesky failed)")]
    CholeskyFailed,
    #[error("standard normal length {got} does not match basis size {expected}")]
    SampleDimensionMismatch { expected: usize, got: usize },
    #[error("observation matrix column count {got} does not match ambient dimension {expected}")]
    ObservationMatrixDimensionMismatch { expected: usize, got: usize },
    #[error(
        "observation matrix row count {rows} does not match observation value length {values}"
    )]
    ObservationValueDimensionMismatch { rows: usize, values: usize },
    #[error("invalid strong dof index {index} for dimension {dimension}")]
    InvalidStrongDofIndex { index: usize, dimension: usize },
    #[error("full vector length {got} does not match full dimension {expected}")]
    FullDimensionMismatch { expected: usize, got: usize },
    #[error("reduced vector length {got} does not match reduced dimension {expected}")]
    ReducedDimensionMismatch { expected: usize, got: usize },
    #[error("requested harmonic dimension {requested} exceeds reduced space dimension {reduced_dimension}")]
    InvalidHarmonicDimension {
        requested: usize,
        reduced_dimension: usize,
    },
}

#[derive(Debug, Clone)]
pub struct EigenBasis {
    eigenvalues: Vec<f64>,
    eigenvectors: Mat<f64>,
}

impl EigenBasis {
    pub fn new(eigenvalues: Vec<f64>, eigenvectors: Mat<f64>) -> Result<Self, GpError> {
        if eigenvalues.is_empty() {
            return Err(GpError::InvalidK);
        }
        if eigenvectors.nrows() == 0 {
            return Err(GpError::EmptyEigenvectors);
        }
        if eigenvalues.len() != eigenvectors.ncols() {
            return Err(GpError::EigenDimensionMismatch {
                values: eigenvalues.len(),
                cols: eigenvectors.ncols(),
            });
        }
        if let Some(&value) = eigenvalues.iter().find(|value| !value.is_finite()) {
            return Err(GpError::NonFiniteEigenvalue { value });
        }
        Ok(Self {
            eigenvalues,
            eigenvectors,
        })
    }

    pub fn from_hodge_laplace_eigenpairs(
        eigenvalues: NaVector,
        eigenvectors: NaMatrix,
    ) -> Result<Self, GpError> {
        // This compatibility constructor assumes the caller has already
        // normalized the columns in the relevant mass inner product.  New
        // generalized-eigenproblem callers should prefer
        // `from_generalized_hodge_laplace_eigenpairs`.
        let values = eigenvalues.iter().copied().collect::<Vec<_>>();
        let vectors = nalgebra_matrix_to_faer(&eigenvectors);
        Self::new(values, vectors)
    }

    /// Construct a basis from a symmetric generalized eigenproblem
    /// `L v = lambda M v` using the coefficient-space mass matrix `M`.
    ///
    /// Each eigenvector is normalized in the mass inner product and the
    /// resulting columns are checked for mutual mass orthogonality.  This is
    /// the normalization under which a full-rank spectral Matérn covariance
    /// equals the covariance obtained by inverting the corresponding FEEC
    /// Matérn precision recurrence.
    pub fn from_generalized_hodge_laplace_eigenpairs(
        eigenvalues: NaVector,
        mut eigenvectors: NaMatrix,
        mass: &CsrMatrix,
        orthogonality_tolerance: f64,
    ) -> Result<Self, GpError> {
        if mass.nrows() != eigenvectors.nrows() || mass.ncols() != eigenvectors.nrows() {
            return Err(GpError::EigenvectorMassDimensionMismatch {
                rows: mass.nrows(),
                cols: mass.ncols(),
                dimension: eigenvectors.nrows(),
            });
        }
        if !orthogonality_tolerance.is_finite() || orthogonality_tolerance < 0.0 {
            return Err(GpError::InvalidMassOrthogonalityTolerance);
        }
        mass_normalize_eigenvectors(mass, &mut eigenvectors)?;
        validate_mass_orthogonality(mass, &eigenvectors, orthogonality_tolerance)?;
        Self::from_hodge_laplace_eigenpairs(eigenvalues, eigenvectors)
    }

    pub fn nrows(&self) -> usize {
        self.eigenvectors.nrows()
    }

    pub fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    pub fn is_empty(&self) -> bool {
        self.eigenvalues.is_empty()
    }

    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> &Mat<f64> {
        &self.eigenvectors
    }

    pub fn truncate_largest(&self, k: usize) -> Result<Self, GpError> {
        self.truncate(k, TruncationOrder::Largest)
    }

    pub fn truncate_smallest(&self, k: usize) -> Result<Self, GpError> {
        self.truncate(k, TruncationOrder::Smallest)
    }

    fn truncate(&self, k: usize, order: TruncationOrder) -> Result<Self, GpError> {
        if k == 0 {
            return Err(GpError::InvalidK);
        }
        let keep = k.min(self.eigenvalues.len());
        let mut indices: Vec<usize> = (0..self.eigenvalues.len()).collect();
        indices.sort_by(|&a, &b| match order {
            TruncationOrder::Largest => self.eigenvalues[b]
                .partial_cmp(&self.eigenvalues[a])
                .unwrap_or(Ordering::Equal),
            TruncationOrder::Smallest => self.eigenvalues[a]
                .partial_cmp(&self.eigenvalues[b])
                .unwrap_or(Ordering::Equal),
        });

        let mut values = Vec::with_capacity(keep);
        let mut vectors = Mat::zeros(self.eigenvectors.nrows(), keep);

        for (new_col, &old_col) in indices.iter().take(keep).enumerate() {
            let val = self.eigenvalues[old_col];
            if !val.is_finite() {
                return Err(GpError::NonFiniteEigenvalue { value: val });
            }
            values.push(val);
            for row in 0..self.eigenvectors.nrows() {
                vectors[(row, new_col)] = self.eigenvectors[(row, old_col)];
            }
        }

        Ok(Self {
            eigenvalues: values,
            eigenvectors: vectors,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum TruncationOrder {
    Largest,
    Smallest,
}

#[derive(Debug, Clone)]
pub struct SpectralMaternGp {
    basis: EigenBasis,
    config: SpectralMaternConfig,
    weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ConditionedGp {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ConditionedFullGp {
    pub mean: Vec<f64>,
    pub covariance: Mat<f64>,
}

/// Dense system used for exact conditioning of a low-rank spectral model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowRankConditioningStrategy {
    /// Factor the observation covariance, with dimension equal to the number
    /// of observations.
    ObservationSpace,
    /// Factor the posterior latent precision, with dimension equal to the
    /// number of retained spectral modes.
    LatentSpace,
}

/// Select the smaller exact dense conditioning system when positive noise
/// permits both algebraically equivalent formulations.
pub fn low_rank_conditioning_strategy(
    observation_count: usize,
    latent_dimension: usize,
    noise_variance: f64,
) -> LowRankConditioningStrategy {
    if noise_variance > 0.0 && latent_dimension < observation_count {
        LowRankConditioningStrategy::LatentSpace
    } else {
        LowRankConditioningStrategy::ObservationSpace
    }
}

impl SpectralMaternGp {
    pub fn from_eigenbasis(
        basis: EigenBasis,
        config: SpectralMaternConfig,
    ) -> Result<Self, GpError> {
        validate_config(config)?;
        // Matérn covariance weights decrease monotonically with the unshifted
        // Hodge--Laplacian eigenvalue. A rank-k approximation therefore retains
        // the smallest eigenvalues (equivalently, the largest covariance weights).
        let basis = basis.truncate_smallest(config.k)?;
        Self::from_truncated_basis(basis, config)
    }

    fn from_truncated_basis(
        basis: EigenBasis,
        config: SpectralMaternConfig,
    ) -> Result<Self, GpError> {
        let mut weights = Vec::with_capacity(basis.len());
        for &lambda in basis.eigenvalues() {
            weights.push(matern_weight(lambda, config)?);
        }
        Ok(Self {
            basis,
            config,
            weights,
        })
    }

    #[cfg(feature = "external-solvers")]
    pub fn from_hodge_laplace(
        topology: &Complex,
        geometry: &MeshLengths,
        grade: ExteriorGrade,
        config: SpectralMaternConfig,
    ) -> Result<Self, GpError> {
        validate_config(config)?;

        let (eigenvals, mut eigenvecs) = if grade == 0 {
            let (eigenvalues, eigen_us) = laplace_beltrami::solve_laplace_beltrami_evp_as_matrix(
                topology, geometry, config.k,
            );
            (eigenvalues, eigen_us)
        } else {
            let (eigenvals, _eigen_sigmas, eigen_us) =
                hodge_laplace::solve_hodge_laplace_evp_config(
                    topology,
                    geometry,
                    grade,
                    config.k,
                    GhiepWhich::Smallest,
                    GhiepReducedSolve::Direct,
                );
            (eigenvals, eigen_us)
        };
        // TODO put inside feec library
        let mass_galmat = assemble::assemble_galmat(
            topology,
            geometry,
            HodgeMassElmat::new(topology.dim(), grade),
        );
        let mass = CsrMatrix::from(&mass_galmat);
        mass_normalize_eigenvectors(&mass, &mut eigenvecs)?;

        let basis = EigenBasis::from_hodge_laplace_eigenpairs(eigenvals, eigenvecs)?;
        Self::from_eigenbasis(basis, config)
    }

    pub fn k(&self) -> usize {
        self.basis.len()
    }

    pub fn eigenvalues(&self) -> &[f64] {
        self.basis.eigenvalues()
    }

    pub fn eigenvectors(&self) -> &Mat<f64> {
        self.basis.eigenvectors()
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn covariance_matrix(&self) -> Mat<f64> {
        let n = self.basis.eigenvectors.nrows();
        let k = self.basis.eigenvectors.ncols();
        let mut scaled = self.basis.eigenvectors.clone();
        for col in 0..k {
            let weight = self.weights[col];
            if weight == 1.0 {
                continue;
            }
            for row in 0..n {
                scaled[(row, col)] *= weight;
            }
        }

        let mut cov = Mat::zeros(n, n);
        matmul(
            &mut cov,
            Accum::Replace,
            scaled.as_ref(),
            self.basis.eigenvectors.as_ref().transpose(),
            1.0,
            Par::Seq,
        );
        cov
    }

    pub fn covariance_entry(&self, i: usize, j: usize) -> f64 {
        let mut acc = 0.0;
        for (col, &weight) in self.weights.iter().enumerate() {
            acc += weight * self.basis.eigenvectors[(i, col)] * self.basis.eigenvectors[(j, col)];
        }
        acc
    }

    pub fn config(&self) -> SpectralMaternConfig {
        self.config
    }

    pub fn feature_matrix(&self) -> Mat<f64> {
        weighted_feature_matrix(self.basis.eigenvectors(), &self.weights)
    }

    pub fn prior_variance(&self) -> Vec<f64> {
        diagonal_from_feature_matrix(&self.feature_matrix())
    }

    pub fn sample_from_standard_normal(&self, z: &[f64]) -> Result<Vec<f64>, GpError> {
        let k = self.weights.len();
        if z.len() != k {
            return Err(GpError::SampleDimensionMismatch {
                expected: k,
                got: z.len(),
            });
        }
        let n = self.basis.eigenvectors.nrows();
        let mut sample = vec![0.0; n];
        for (col, &weight) in self.weights.iter().enumerate() {
            let scale = weight.max(0.0).sqrt();
            if scale == 0.0 {
                continue;
            }
            let zc = z[col];
            for (row, value) in sample.iter_mut().enumerate() {
                *value += scale * self.basis.eigenvectors[(row, col)] * zc;
            }
        }
        Ok(sample)
    }

    pub fn condition_linear_observations(
        &self,
        observation_matrix: &CsrMatrix,
        observation_values: &[f64],
        noise_variance: f64,
    ) -> Result<ConditionedGp, GpError> {
        condition_low_rank_factors(
            &self.feature_matrix(),
            observation_matrix,
            observation_values,
            noise_variance,
        )
    }

    /// Condition the spectral model and report moments after a sparse linear
    /// output map, without materializing an ambient dense covariance.
    ///
    /// This is the appropriate path for reconstructed fields and engineering
    /// quantities of interest.  It applies the same map to the retained
    /// features used by the observation update.
    pub fn condition_linear_observations_at_outputs(
        &self,
        observation_matrix: &CsrMatrix,
        observation_values: &[f64],
        noise_variance: f64,
        output_matrix: &CsrMatrix,
    ) -> Result<ConditionedGp, GpError> {
        let features = self.feature_matrix();
        let output_features = sparse_dense_product(output_matrix, &features)?;
        condition_low_rank_factors_at_outputs(
            &features,
            &output_features,
            observation_matrix,
            observation_values,
            noise_variance,
        )
    }

    pub fn condition_linear_observations_with_covariance(
        &self,
        observation_matrix: &CsrMatrix,
        observation_values: &[f64],
        noise_variance: f64,
    ) -> Result<ConditionedFullGp, GpError> {
        condition_low_rank_factors_with_covariance(
            &self.feature_matrix(),
            observation_matrix,
            observation_values,
            noise_variance,
        )
    }
}

pub fn matern_covariance_euclidean(
    distance: f64,
    config: EuclideanMaternConfig,
) -> Result<f64, GpError> {
    validate_euclidean_config(config)?;
    if !distance.is_finite() || distance < 0.0 {
        return Err(GpError::InvalidDistance);
    }
    if distance == 0.0 {
        return Ok(config.variance);
    }
    let x = config.kappa * distance;
    if let Some(val) = half_integer_kernel(x, config.variance, config.nu) {
        return Ok(val);
    }
    if (config.nu - 1.0).abs() <= 1e-12 {
        let prefactor = 2.0_f64.powf(1.0 - config.nu) / tgamma(config.nu);
        let value = config.variance * prefactor * x.powf(config.nu) * bessel_k1(x);
        return Ok(value);
    }
    Err(GpError::UnsupportedNu { nu: config.nu })
}

pub fn matern_covariance_matrix_euclidean(
    points: &[Vec<f64>],
    config: EuclideanMaternConfig,
) -> Result<Mat<f64>, GpError> {
    validate_euclidean_config(config)?;
    if points.is_empty() {
        return Err(GpError::EmptyPoints);
    }
    let dim = points[0].len();
    if dim == 0 {
        return Err(GpError::EmptyPointDimension);
    }
    if !supports_matern_nu(config.nu) {
        return Err(GpError::UnsupportedNu { nu: config.nu });
    }
    if points[0].iter().any(|v| !v.is_finite()) {
        return Err(GpError::InvalidDistance);
    }
    for point in points.iter().skip(1) {
        if point.len() != dim {
            return Err(GpError::PointDimensionMismatch {
                expected: dim,
                got: point.len(),
            });
        }
        if point.iter().any(|v| !v.is_finite()) {
            return Err(GpError::InvalidDistance);
        }
    }

    let n = points.len();
    let mut cov = Mat::zeros(n, n);
    for i in 0..n {
        cov[(i, i)] = config.variance;
        for j in (i + 1)..n {
            let dist = euclidean_distance(&points[i], &points[j])?;
            let x = config.kappa * dist;
            let value = if let Some(val) = half_integer_kernel(x, config.variance, config.nu) {
                val
            } else if (config.nu - 1.0).abs() <= 1e-12 {
                let prefactor = 2.0_f64.powf(1.0 - config.nu) / tgamma(config.nu);
                config.variance * prefactor * x.powf(config.nu) * bessel_k1(x)
            } else {
                return Err(GpError::UnsupportedNu { nu: config.nu });
            };
            cov[(i, j)] = value;
            cov[(j, i)] = value;
        }
    }
    Ok(cov)
}

pub fn condition_full_covariance(
    cov: &Mat<f64>,
    obs_indices: &[usize],
    obs_values: &[f64],
    noise_variance: f64,
) -> Result<ConditionedGp, GpError> {
    if cov.nrows() != cov.ncols() {
        return Err(GpError::InvalidCovarianceShape {
            rows: cov.nrows(),
            cols: cov.ncols(),
        });
    }
    if obs_indices.len() != obs_values.len() {
        return Err(GpError::ObservationCountMismatch {
            indices: obs_indices.len(),
            values: obs_values.len(),
        });
    }
    if !noise_variance.is_finite() || noise_variance < 0.0 {
        return Err(GpError::InvalidNoiseVariance);
    }

    let n = cov.nrows();
    let m = obs_indices.len();
    let mut seen = vec![false; n];
    for &idx in obs_indices {
        if idx >= n {
            return Err(GpError::ObservationIndexOutOfBounds { index: idx, n });
        }
        if seen[idx] {
            return Err(GpError::ObservationIndexRepeated { index: idx });
        }
        seen[idx] = true;
    }

    let observations = Mat::from_fn(m, 1, |i, _| obs_values[i.unbound()]);

    let mut k_oo = Mat::from_fn(m, m, |i, j| {
        let row = obs_indices[i.unbound()];
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });
    for i in 0..m {
        let ii = idx(i);
        k_oo[(ii, ii)] += noise_variance;
    }

    let chol = k_oo.llt(Side::Lower).map_err(|_| GpError::CholeskyFailed)?;
    let mut alpha = observations.clone();
    chol.solve_in_place(alpha.as_mut());

    let k_star_o = Mat::from_fn(n, m, |i, j| {
        let row = i.unbound();
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });

    let mut posterior_mean = Mat::zeros(n, 1);
    matmul(
        &mut posterior_mean,
        Accum::Replace,
        k_star_o.as_ref(),
        alpha.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut w = k_star_o.transpose().to_owned();
    chol.solve_in_place(w.as_mut());

    let mut posterior_var = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..m {
            acc += k_star_o[(idx(i), idx(j))] * w[(idx(j), idx(i))];
        }
        let prior_var = cov[(idx(i), idx(i))];
        let var = (prior_var - acc).max(0.0);
        posterior_var.push(var);
    }

    Ok(ConditionedGp {
        mean: mat_col_to_vec(&posterior_mean),
        variance: posterior_var,
    })
}

pub fn condition_full_covariance_with_covariance(
    cov: &Mat<f64>,
    obs_indices: &[usize],
    obs_values: &[f64],
    noise_variance: f64,
) -> Result<ConditionedFullGp, GpError> {
    if cov.nrows() != cov.ncols() {
        return Err(GpError::InvalidCovarianceShape {
            rows: cov.nrows(),
            cols: cov.ncols(),
        });
    }
    if obs_indices.len() != obs_values.len() {
        return Err(GpError::ObservationCountMismatch {
            indices: obs_indices.len(),
            values: obs_values.len(),
        });
    }
    if !noise_variance.is_finite() || noise_variance < 0.0 {
        return Err(GpError::InvalidNoiseVariance);
    }

    let n = cov.nrows();
    let m = obs_indices.len();
    let mut seen = vec![false; n];
    for &idx in obs_indices {
        if idx >= n {
            return Err(GpError::ObservationIndexOutOfBounds { index: idx, n });
        }
        if seen[idx] {
            return Err(GpError::ObservationIndexRepeated { index: idx });
        }
        seen[idx] = true;
    }

    let observations = Mat::from_fn(m, 1, |i, _| obs_values[i.unbound()]);

    let mut k_oo = Mat::from_fn(m, m, |i, j| {
        let row = obs_indices[i.unbound()];
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });
    for i in 0..m {
        let ii = idx(i);
        k_oo[(ii, ii)] += noise_variance;
    }

    let chol = k_oo.llt(Side::Lower).map_err(|_| GpError::CholeskyFailed)?;
    let mut alpha = observations.clone();
    chol.solve_in_place(alpha.as_mut());

    let k_star_o = Mat::from_fn(n, m, |i, j| {
        let row = i.unbound();
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });

    let mut posterior_mean = Mat::zeros(n, 1);
    matmul(
        &mut posterior_mean,
        Accum::Replace,
        k_star_o.as_ref(),
        alpha.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut w = k_star_o.transpose().to_owned();
    chol.solve_in_place(w.as_mut());

    let mut correction = Mat::zeros(n, n);
    matmul(
        &mut correction,
        Accum::Replace,
        k_star_o.as_ref(),
        w.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut posterior_cov = cov.clone();
    for i in 0..n {
        for j in 0..n {
            posterior_cov[(idx(i), idx(j))] -= correction[(idx(i), idx(j))];
        }
    }

    Ok(ConditionedFullGp {
        mean: mat_col_to_vec(&posterior_mean),
        covariance: posterior_cov,
    })
}

pub(crate) fn weighted_feature_matrix(eigenvectors: &Mat<f64>, weights: &[f64]) -> Mat<f64> {
    let mut features = eigenvectors.clone();
    for (col, &weight) in weights.iter().enumerate() {
        let scale = weight.max(0.0).sqrt();
        for row in 0..features.nrows() {
            features[(row, col)] *= scale;
        }
    }
    features
}

pub(crate) fn diagonal_from_feature_matrix(features: &Mat<f64>) -> Vec<f64> {
    let mut diagonal = Vec::with_capacity(features.nrows());
    for row in 0..features.nrows() {
        let mut value = 0.0;
        for col in 0..features.ncols() {
            let entry = features[(row, col)];
            value += entry * entry;
        }
        diagonal.push(value.max(0.0));
    }
    diagonal
}

pub(crate) fn condition_low_rank_factors(
    features: &Mat<f64>,
    observation_matrix: &CsrMatrix,
    observation_values: &[f64],
    noise_variance: f64,
) -> Result<ConditionedGp, GpError> {
    condition_low_rank_factors_at_outputs(
        features,
        features,
        observation_matrix,
        observation_values,
        noise_variance,
    )
}

pub(crate) fn condition_low_rank_factors_at_outputs(
    prior_features: &Mat<f64>,
    output_features: &Mat<f64>,
    observation_matrix: &CsrMatrix,
    observation_values: &[f64],
    noise_variance: f64,
) -> Result<ConditionedGp, GpError> {
    let conditioned = condition_low_rank_factors_impl(
        prior_features,
        output_features,
        observation_matrix,
        observation_values,
        noise_variance,
        false,
    )?;
    Ok(ConditionedGp {
        mean: conditioned.mean,
        variance: conditioned.variance,
    })
}

pub(crate) fn condition_low_rank_factors_with_covariance(
    features: &Mat<f64>,
    observation_matrix: &CsrMatrix,
    observation_values: &[f64],
    noise_variance: f64,
) -> Result<ConditionedFullGp, GpError> {
    let conditioned = condition_low_rank_factors_impl(
        features,
        features,
        observation_matrix,
        observation_values,
        noise_variance,
        true,
    )?;
    let covariance = conditioned
        .covariance
        .expect("full covariance requested must produce covariance");
    Ok(ConditionedFullGp {
        mean: conditioned.mean,
        covariance,
    })
}

struct LowRankConditioningResult {
    mean: Vec<f64>,
    variance: Vec<f64>,
    covariance: Option<Mat<f64>>,
}

fn condition_low_rank_factors_impl(
    prior_features: &Mat<f64>,
    output_features: &Mat<f64>,
    observation_matrix: &CsrMatrix,
    observation_values: &[f64],
    noise_variance: f64,
    need_covariance: bool,
) -> Result<LowRankConditioningResult, GpError> {
    validate_linear_observations(
        prior_features.nrows(),
        observation_matrix,
        observation_values,
        noise_variance,
    )?;
    if output_features.ncols() != prior_features.ncols() {
        return Err(GpError::EigenDimensionMismatch {
            values: prior_features.ncols(),
            cols: output_features.ncols(),
        });
    }

    let n = output_features.nrows();
    let latent_dim = prior_features.ncols();
    let observation_count = observation_matrix.nrows();

    if observation_count == 0 {
        let covariance = if need_covariance {
            Some(full_covariance_from_feature_matrix(output_features))
        } else {
            None
        };
        return Ok(LowRankConditioningResult {
            mean: vec![0.0; n],
            variance: diagonal_from_feature_matrix(output_features),
            covariance,
        });
    }

    let obs_features = sparse_dense_product(observation_matrix, prior_features)?;
    let observations = Mat::from_fn(observation_count, 1, |row, _| {
        observation_values[row.unbound()]
    });

    let (feature_rhs, latent_covariance) =
        match low_rank_conditioning_strategy(observation_count, latent_dim, noise_variance) {
            LowRankConditioningStrategy::ObservationSpace => {
                condition_in_observation_space(&obs_features, &observations, noise_variance)?
            }
            LowRankConditioningStrategy::LatentSpace => {
                condition_in_latent_space(&obs_features, &observations, noise_variance)?
            }
        };

    let mut posterior_mean = Mat::zeros(n, 1);
    matmul(
        &mut posterior_mean,
        Accum::Replace,
        output_features.as_ref(),
        feature_rhs.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut feature_latent_covariance = Mat::zeros(n, latent_dim);
    matmul(
        &mut feature_latent_covariance,
        Accum::Replace,
        output_features.as_ref(),
        latent_covariance.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut variance = Vec::with_capacity(n);
    for row in 0..n {
        let mut value = 0.0;
        for col in 0..latent_dim {
            value += feature_latent_covariance[(idx(row), idx(col))]
                * output_features[(idx(row), idx(col))];
        }
        variance.push(value.max(0.0));
    }

    let covariance = if need_covariance {
        let mut posterior_covariance = Mat::zeros(n, n);
        matmul(
            &mut posterior_covariance,
            Accum::Replace,
            feature_latent_covariance.as_ref(),
            output_features.as_ref().transpose(),
            1.0,
            Par::Seq,
        );
        Some(posterior_covariance)
    } else {
        None
    };

    Ok(LowRankConditioningResult {
        mean: mat_col_to_vec(&posterior_mean),
        variance,
        covariance,
    })
}

fn condition_in_observation_space(
    observation_features: &Mat<f64>,
    observations: &Mat<f64>,
    noise_variance: f64,
) -> Result<(Mat<f64>, Mat<f64>), GpError> {
    let observation_count = observation_features.nrows();
    let latent_dimension = observation_features.ncols();
    let mut gram = Mat::zeros(observation_count, observation_count);
    matmul(
        &mut gram,
        Accum::Replace,
        observation_features.as_ref(),
        observation_features.as_ref().transpose(),
        1.0,
        Par::Seq,
    );
    for row in 0..observation_count {
        gram[(idx(row), idx(row))] += noise_variance;
    }
    let chol = gram.llt(Side::Lower).map_err(|_| GpError::CholeskyFailed)?;

    let mut solved_observations = observations.clone();
    chol.solve_in_place(solved_observations.as_mut());
    let mut posterior_latent_mean = Mat::zeros(latent_dimension, 1);
    matmul(
        &mut posterior_latent_mean,
        Accum::Replace,
        observation_features.as_ref().transpose(),
        solved_observations.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut solved_observation_features = observation_features.clone();
    chol.solve_in_place(solved_observation_features.as_mut());
    let mut correction = Mat::zeros(latent_dimension, latent_dimension);
    matmul(
        &mut correction,
        Accum::Replace,
        observation_features.as_ref().transpose(),
        solved_observation_features.as_ref(),
        1.0,
        Par::Seq,
    );
    let latent_covariance = Mat::from_fn(latent_dimension, latent_dimension, |row, col| {
        f64::from(row == col) - correction[(row, col)]
    });
    Ok((posterior_latent_mean, latent_covariance))
}

fn condition_in_latent_space(
    observation_features: &Mat<f64>,
    observations: &Mat<f64>,
    noise_variance: f64,
) -> Result<(Mat<f64>, Mat<f64>), GpError> {
    debug_assert!(noise_variance > 0.0);
    let latent_dimension = observation_features.ncols();
    let mut posterior_latent_system = Mat::zeros(latent_dimension, latent_dimension);
    matmul(
        &mut posterior_latent_system,
        Accum::Replace,
        observation_features.as_ref().transpose(),
        observation_features.as_ref(),
        1.0,
        Par::Seq,
    );
    for index in 0..latent_dimension {
        posterior_latent_system[(idx(index), idx(index))] += noise_variance;
    }
    let chol = posterior_latent_system
        .llt(Side::Lower)
        .map_err(|_| GpError::CholeskyFailed)?;

    let mut posterior_latent_mean = Mat::zeros(latent_dimension, 1);
    matmul(
        &mut posterior_latent_mean,
        Accum::Replace,
        observation_features.as_ref().transpose(),
        observations.as_ref(),
        1.0,
        Par::Seq,
    );
    chol.solve_in_place(posterior_latent_mean.as_mut());

    let mut latent_covariance = Mat::identity(latent_dimension, latent_dimension);
    chol.solve_in_place(latent_covariance.as_mut());
    for row in 0..latent_dimension {
        for col in 0..latent_dimension {
            latent_covariance[(idx(row), idx(col))] *= noise_variance;
        }
    }
    Ok((posterior_latent_mean, latent_covariance))
}

pub(crate) fn full_covariance_from_feature_matrix(features: &Mat<f64>) -> Mat<f64> {
    let n = features.nrows();
    let mut covariance = Mat::zeros(n, n);
    matmul(
        &mut covariance,
        Accum::Replace,
        features.as_ref(),
        features.as_ref().transpose(),
        1.0,
        Par::Seq,
    );
    covariance
}

fn validate_linear_observations(
    ambient_dimension: usize,
    observation_matrix: &CsrMatrix,
    observation_values: &[f64],
    noise_variance: f64,
) -> Result<(), GpError> {
    if observation_matrix.ncols() != ambient_dimension {
        return Err(GpError::ObservationMatrixDimensionMismatch {
            expected: ambient_dimension,
            got: observation_matrix.ncols(),
        });
    }
    if observation_matrix.nrows() != observation_values.len() {
        return Err(GpError::ObservationValueDimensionMismatch {
            rows: observation_matrix.nrows(),
            values: observation_values.len(),
        });
    }
    if !noise_variance.is_finite() || noise_variance < 0.0 {
        return Err(GpError::InvalidNoiseVariance);
    }
    Ok(())
}

pub(crate) fn sparse_dense_product(lhs: &CsrMatrix, rhs: &Mat<f64>) -> Result<Mat<f64>, GpError> {
    if lhs.ncols() != rhs.nrows() {
        return Err(GpError::ObservationMatrixDimensionMismatch {
            expected: rhs.nrows(),
            got: lhs.ncols(),
        });
    }

    let mut product = Mat::zeros(lhs.nrows(), rhs.ncols());
    for (row, col, value) in lhs.triplet_iter() {
        for dense_col in 0..rhs.ncols() {
            product[(row, dense_col)] += *value * rhs[(col, dense_col)];
        }
    }
    Ok(product)
}

fn validate_config(config: SpectralMaternConfig) -> Result<(), GpError> {
    if config.k == 0 {
        return Err(GpError::InvalidK);
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(GpError::InvalidKappa);
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 {
        return Err(GpError::InvalidAlpha);
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(GpError::InvalidTau);
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 {
        return Err(GpError::InvalidAlpha);
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(GpError::InvalidTau);
    }
    Ok(())
}

fn validate_euclidean_config(config: EuclideanMaternConfig) -> Result<(), GpError> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(GpError::InvalidKappa);
    }
    if !config.nu.is_finite() || config.nu <= 0.0 {
        return Err(GpError::InvalidNu);
    }
    if !config.variance.is_finite() || config.variance < 0.0 {
        return Err(GpError::InvalidVariance);
    }
    Ok(())
}

fn matern_weight(lambda: f64, config: SpectralMaternConfig) -> Result<f64, GpError> {
    if !lambda.is_finite() {
        return Err(GpError::NonFiniteEigenvalue { value: lambda });
    }
    let base = config.kappa * config.kappa + lambda;
    if !base.is_finite() || base <= 0.0 {
        return Err(GpError::InvalidEigenvalue { value: lambda });
    }
    let inverse_tau = 1. / config.tau;
    Ok(inverse_tau * inverse_tau * base.powf(-config.alpha))
}

fn half_integer_kernel(x: f64, variance: f64, nu: f64) -> Option<f64> {
    let tol = 1e-12;
    let exp_term = (-x).exp();
    if (nu - 0.5).abs() <= tol {
        return Some(variance * exp_term);
    }
    if (nu - 1.5).abs() <= tol {
        return Some(variance * (1.0 + x) * exp_term);
    }
    if (nu - 2.5).abs() <= tol {
        return Some(variance * (1.0 + x + x * x / 3.0) * exp_term);
    }
    None
}

fn supports_matern_nu(nu: f64) -> bool {
    (nu - 0.5).abs() <= 1e-12
        || (nu - 1.0).abs() <= 1e-12
        || (nu - 1.5).abs() <= 1e-12
        || (nu - 2.5).abs() <= 1e-12
}

fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let mut out = if ax <= 3.75 {
        let t = ax / 3.75;
        let t2 = t * t;
        ax * (0.5
            + t2 * (0.87890594
                + t2 * (0.51498869
                    + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (-0.03988024
                    + t * (-0.00362018
                        + t * (0.00163801
                            + t * (-0.01031555
                                + t * (0.02282967
                                    + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059))))))))
    };
    if x < 0.0 {
        out = -out;
    }
    out
}

fn bessel_k1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        let t = x * x / 4.0;
        (x / 2.0).ln() * bessel_i1(x)
            + (1.0 / x)
                * (1.0
                    + t * (0.15443144
                        + t * (-0.67278579
                            + t * (-0.18156897
                                + t * (-0.01919402 + t * (-0.00110404 + t * (-0.00004686)))))))
    } else {
        let t = 2.0 / x;
        (-(x)).exp() / x.sqrt()
            * (1.25331414
                + t * (0.23498619
                    + t * (-0.03655620
                        + t * (0.01504268
                            + t * (-0.00780353 + t * (0.00325614 - t * 0.00068245))))))
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64, GpError> {
    if a.len() != b.len() {
        return Err(GpError::PointDimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(GpError::EmptyPointDimension);
    }
    let mut sum = 0.0;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let diff = av - bv;
        sum += diff * diff;
    }
    Ok(sum.sqrt())
}

fn nalgebra_matrix_to_faer(matrix: &NaMatrix) -> Mat<f64> {
    let mut out = Mat::zeros(matrix.nrows(), matrix.ncols());
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            out[(i, j)] = matrix[(i, j)];
        }
    }
    out
}

fn mass_normalize_eigenvectors(
    mass: &CsrMatrix,
    eigenvectors: &mut NaMatrix,
) -> Result<(), GpError> {
    for col in 0..eigenvectors.ncols() {
        let vec = eigenvectors.column(col).into_owned();
        let norm2 = quadratic_form_sparse(mass, &vec);
        if !norm2.is_finite() || norm2 <= 0.0 {
            return Err(GpError::InvalidEigenvectorNorm { value: norm2 });
        }
        let scale = 1.0 / norm2.sqrt();
        for row in 0..eigenvectors.nrows() {
            eigenvectors[(row, col)] *= scale;
        }
    }
    Ok(())
}

fn validate_mass_orthogonality(
    mass: &CsrMatrix,
    eigenvectors: &NaMatrix,
    tolerance: f64,
) -> Result<(), GpError> {
    for left in 0..eigenvectors.ncols() {
        for right in (left + 1)..eigenvectors.ncols() {
            let mut value = 0.0;
            for (row, col, mass_value) in mass.triplet_iter() {
                value += mass_value * eigenvectors[(row, left)] * eigenvectors[(col, right)];
            }
            if !value.is_finite() || value.abs() > tolerance {
                return Err(GpError::EigenvectorsNotMassOrthogonal {
                    left,
                    right,
                    value,
                    tolerance,
                });
            }
        }
    }
    Ok(())
}

#[inline]
fn idx(i: usize) -> Idx<usize> {
    unsafe { Idx::<usize>::new_unbound(i) }
}

fn mat_col_to_vec(col: &Mat<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(col.nrows());
    for i in 0..col.nrows() {
        out.push(col[(idx(i), idx(0))]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64, tol: f64) {
        let diff = (left - right).abs();
        assert!(diff <= tol, "expected {left} to be within {tol} of {right}");
    }

    fn assert_matrix_close(left: &Mat<f64>, right: &Mat<f64>, tol: f64) {
        assert_eq!(left.nrows(), right.nrows());
        assert_eq!(left.ncols(), right.ncols());
        for row in 0..left.nrows() {
            for col in 0..left.ncols() {
                assert_close(left[(row, col)], right[(row, col)], tol);
            }
        }
    }

    fn dense_product(left: &Mat<f64>, right: &Mat<f64>) -> Mat<f64> {
        assert_eq!(left.ncols(), right.nrows());
        let mut product = Mat::zeros(left.nrows(), right.ncols());
        matmul(
            &mut product,
            Accum::Replace,
            left.as_ref(),
            right.as_ref(),
            1.0,
            Par::Seq,
        );
        product
    }

    fn dense_transpose(matrix: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
            matrix[(col, row)]
        })
    }

    fn spd_inverse(matrix: &Mat<f64>) -> Mat<f64> {
        let factor = matrix
            .llt(Side::Lower)
            .expect("test matrix should be positive definite");
        let mut inverse = Mat::identity(matrix.nrows(), matrix.ncols());
        factor.solve_in_place(inverse.as_mut());
        inverse
    }

    #[test]
    fn covariance_matches_weighted_identity_basis() {
        let mut eigenvectors = Mat::zeros(2, 2);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        let eigenvalues = vec![4.0, 9.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 2,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();
        let cov = gp.covariance_matrix();

        let w0 = matern_weight(4.0, config).unwrap();
        let w1 = matern_weight(9.0, config).unwrap();
        assert_close(cov[(0, 0)], w0, 1e-12);
        assert_close(cov[(1, 1)], w1, 1e-12);
        assert_close(cov[(0, 1)], 0.0, 1e-12);
        assert_close(cov[(1, 0)], 0.0, 1e-12);
    }

    #[test]
    fn matern_truncation_keeps_smallest_unshifted_eigenvalues() {
        let mut eigenvectors = Mat::zeros(3, 3);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        eigenvectors[(2, 2)] = 1.0;
        let eigenvalues = vec![1.0, 10.0, 5.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 2,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();

        assert_eq!(gp.k(), 2);
        assert!(gp.eigenvalues()[0] <= gp.eigenvalues()[1]);
        assert_close(gp.eigenvalues()[0], 1.0, 1e-12);
        assert_close(gp.eigenvalues()[1], 5.0, 1e-12);
    }

    #[test]
    fn eigenbasis_rejects_nonfinite_values_before_ordering() {
        let eigenvectors = Mat::identity(3, 3);
        let error = EigenBasis::new(vec![1.0, f64::NAN, 2.0], eigenvectors).unwrap_err();
        assert!(matches!(error, GpError::NonFiniteEigenvalue { value } if value.is_nan()));
    }

    #[test]
    fn full_rank_generalized_spectrum_matches_matern_precision_for_alpha_one_to_three() {
        use common::linalg::nalgebra::CooMatrix;

        // Choose a nonorthogonal eigenvector matrix V, then construct
        // M = V^{-T} V^{-1} and L = V^{-T} Lambda V^{-1}. Consequently,
        // V^T M V = I and L V = M V Lambda exactly up to roundoff.
        let eigenvectors = Mat::from_fn(3, 3, |row, col| {
            const VALUES: [[f64; 3]; 3] = [[1.0, 0.5, 0.0], [0.0, 1.0, 1.0 / 3.0], [0.0, 0.0, 1.0]];
            VALUES[row][col]
        });
        let eigenvector_inverse = Mat::from_fn(3, 3, |row, col| {
            const VALUES: [[f64; 3]; 3] = [
                [1.0, -0.5, 1.0 / 6.0],
                [0.0, 1.0, -1.0 / 3.0],
                [0.0, 0.0, 1.0],
            ];
            VALUES[row][col]
        });
        let eigenvalues = [0.25, 2.0, 7.0];
        let inverse_transpose = dense_transpose(&eigenvector_inverse);
        let mass = dense_product(&inverse_transpose, &eigenvector_inverse);
        let mut sparse_mass = CooMatrix::new(3, 3);
        for row in 0..3 {
            for col in 0..3 {
                if mass[(row, col)] != 0.0 {
                    sparse_mass.push(row, col, mass[(row, col)]);
                }
            }
        }
        let sparse_mass = CsrMatrix::from(&sparse_mass);
        let weighted_inverse = Mat::from_fn(3, 3, |row, col| {
            eigenvalues[row] * eigenvector_inverse[(row, col)]
        });
        let laplacian = dense_product(&inverse_transpose, &weighted_inverse);

        assert!(
            mass[(0, 1)].abs() > 0.1,
            "mass fixture must be non-diagonal"
        );
        assert!(
            eigenvectors[(0, 1)].abs() > 0.1,
            "basis fixture must be non-diagonal"
        );
        let gram = dense_product(
            &dense_product(&dense_transpose(&eigenvectors), &mass),
            &eigenvectors,
        );
        assert_matrix_close(&gram, &Mat::identity(3, 3), 2e-14);
        let generalized_left = dense_product(&laplacian, &eigenvectors);
        let mass_times_basis = dense_product(&mass, &eigenvectors);
        let generalized_right = Mat::from_fn(3, 3, |row, col| {
            mass_times_basis[(row, col)] * eigenvalues[col]
        });
        assert_matrix_close(&generalized_left, &generalized_right, 2e-14);

        let kappa = 1.3;
        let tau = 0.7;
        let system = Mat::from_fn(3, 3, |row, col| {
            kappa * kappa * mass[(row, col)] + laplacian[(row, col)]
        });
        let mass_inverse = spd_inverse(&mass);

        // Present the eigenpairs out of order so this also exercises the
        // constructor's smallest-eigenvalue ordering at full rank.
        let permutation = [2, 0, 1];
        let unordered_values = permutation
            .iter()
            .map(|&col| eigenvalues[col])
            .collect::<Vec<_>>();
        let unordered_vectors =
            Mat::from_fn(3, 3, |row, col| eigenvectors[(row, permutation[col])]);

        let mut unscaled_precision = system.clone();
        for alpha in 1..=3 {
            if alpha > 1 {
                unscaled_precision =
                    dense_product(&dense_product(&unscaled_precision, &mass_inverse), &system);
            }
            let precision =
                Mat::from_fn(3, 3, |row, col| tau * tau * unscaled_precision[(row, col)]);
            let precision_covariance = spd_inverse(&precision);

            let basis = EigenBasis::from_generalized_hodge_laplace_eigenpairs(
                NaVector::from_vec(unordered_values.clone()),
                NaMatrix::from_fn(3, 3, |row, col| unordered_vectors[(row, col)]),
                &sparse_mass,
                2.0e-14,
            )
            .expect("mass-orthonormal generalized eigenbasis should be valid");
            let spectral = SpectralMaternGp::from_eigenbasis(
                basis,
                SpectralMaternConfig {
                    kappa,
                    alpha: f64::from(alpha),
                    tau,
                    k: 3,
                },
            )
            .expect("full-rank spectral Matérn model should be valid");

            assert_eq!(spectral.eigenvalues(), &eigenvalues);
            assert_matrix_close(&spectral.covariance_matrix(), &precision_covariance, 2e-12);
        }
    }

    #[test]
    fn conditioning_matches_small_covariance() {
        let mut cov = Mat::zeros(2, 2);
        cov[(idx(0), idx(0))] = 1.0;
        cov[(idx(1), idx(1))] = 1.0;
        cov[(idx(0), idx(1))] = 0.5;
        cov[(idx(1), idx(0))] = 0.5;

        let obs_indices = vec![0];
        let obs_values = vec![1.0];
        let conditioned = condition_full_covariance(&cov, &obs_indices, &obs_values, 0.0).unwrap();

        assert_close(conditioned.mean[0], 1.0, 1e-12);
        assert_close(conditioned.mean[1], 0.5, 1e-12);
        assert_close(conditioned.variance[0], 0.0, 1e-12);
        assert_close(conditioned.variance[1], 0.75, 1e-12);
    }

    #[test]
    fn sample_from_standard_normal_matches_identity_basis() {
        let mut eigenvectors = Mat::zeros(2, 2);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        let eigenvalues = vec![3.0, 1.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 2,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();

        let z = vec![2.0, -4.0];
        let sample = gp.sample_from_standard_normal(&z).unwrap();

        let weight_for_first_coordinate = matern_weight(3.0, config).unwrap();
        let weight_for_second_coordinate = matern_weight(1.0, config).unwrap();
        assert_eq!(gp.eigenvalues(), &[1.0, 3.0]);
        assert_close(sample[0], weight_for_first_coordinate.sqrt() * z[1], 1e-12);
        assert_close(sample[1], weight_for_second_coordinate.sqrt() * z[0], 1e-12);
    }

    #[test]
    fn sample_from_standard_normal_rejects_mismatch() {
        let mut eigenvectors = Mat::zeros(2, 1);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 0)] = 1.0;
        let eigenvalues = vec![1.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 1,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();

        let err = gp.sample_from_standard_normal(&[0.0, 1.0]).unwrap_err();
        match err {
            GpError::SampleDimensionMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn euclidean_matern_half_integer_matches_closed_form() {
        let config = EuclideanMaternConfig {
            kappa: 1.3,
            nu: 0.5,
            variance: 2.5,
        };
        let r = 0.7;
        let expected = config.variance * (-config.kappa * r).exp();
        let got = matern_covariance_euclidean(r, config).unwrap();
        assert_close(got, expected, 1e-12);

        let config = EuclideanMaternConfig {
            kappa: 0.8,
            nu: 1.5,
            variance: 1.2,
        };
        let r = 1.1;
        let x = config.kappa * r;
        let expected = config.variance * (1.0 + x) * (-x).exp();
        let got = matern_covariance_euclidean(r, config).unwrap();
        assert_close(got, expected, 1e-12);
    }

    #[test]
    fn euclidean_covariance_matrix_is_symmetric() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let config = EuclideanMaternConfig {
            kappa: 1.0,
            nu: 2.5,
            variance: 0.7,
        };
        let cov = matern_covariance_matrix_euclidean(&points, config).unwrap();
        assert_eq!(cov.nrows(), 3);
        assert_eq!(cov.ncols(), 3);
        assert_close(cov[(0, 0)], config.variance, 1e-12);
        assert_close(cov[(1, 1)], config.variance, 1e-12);
        assert_close(cov[(2, 2)], config.variance, 1e-12);
        assert_close(cov[(0, 1)], cov[(1, 0)], 1e-12);
        assert_close(cov[(0, 2)], cov[(2, 0)], 1e-12);
        assert_close(cov[(1, 2)], cov[(2, 1)], 1e-12);
    }

    #[test]
    fn conditioning_full_covariance_produces_full_posterior() {
        let mut cov = Mat::zeros(2, 2);
        cov[(idx(0), idx(0))] = 1.0;
        cov[(idx(1), idx(1))] = 1.0;
        cov[(idx(0), idx(1))] = 0.5;
        cov[(idx(1), idx(0))] = 0.5;

        let obs_indices = vec![0];
        let obs_values = vec![1.0];
        let conditioned =
            condition_full_covariance_with_covariance(&cov, &obs_indices, &obs_values, 0.0)
                .unwrap();

        assert_close(conditioned.mean[0], 1.0, 1e-12);
        assert_close(conditioned.mean[1], 0.5, 1e-12);
        assert_close(conditioned.covariance[(idx(0), idx(0))], 0.0, 1e-12);
        assert_close(conditioned.covariance[(idx(1), idx(1))], 0.75, 1e-12);
    }

    #[test]
    fn euclidean_matern_rejects_unsupported_nu() {
        let config = EuclideanMaternConfig {
            kappa: 1.0,
            nu: 0.7,
            variance: 1.0,
        };
        let err = matern_covariance_euclidean(0.3, config).unwrap_err();
        match err {
            GpError::UnsupportedNu { nu } => assert_close(nu, 0.7, 1e-12),
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn euclidean_matern_nu_one_uses_bessel_k1() {
        let config = EuclideanMaternConfig {
            kappa: 1.0,
            nu: 1.0,
            variance: 2.0,
        };
        let r = 1.0;
        let expected = 2.0 * 1.0 * bessel_k1(1.0);
        let got = matern_covariance_euclidean(r, config).unwrap();
        assert_close(got, expected, 1e-6);
    }

    #[test]
    fn mass_normalization_scales_columns_to_unit_mass() {
        use common::linalg::nalgebra::{CooMatrix, Vector as NaVector};

        let mass =
            CooMatrix::try_from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![2.0, 8.0]).unwrap();
        let mass = CsrMatrix::from(&mass);

        let mut eigenvectors = NaMatrix::from_column_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        mass_normalize_eigenvectors(&mass, &mut eigenvectors).unwrap();

        for col in 0..eigenvectors.ncols() {
            let vec = NaVector::from_column_slice(eigenvectors.column(col).as_slice());
            let norm2 = quadratic_form_sparse(&mass, &vec);
            assert_close(norm2, 1.0, 1e-12);
        }
    }

    #[test]
    fn generalized_basis_constructor_normalizes_and_checks_mass_orthogonality() {
        use common::linalg::nalgebra::CooMatrix;

        let mass =
            CooMatrix::try_from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![2.0, 8.0]).unwrap();
        let mass = CsrMatrix::from(&mass);
        let eigenvalues = NaVector::from_vec(vec![1.0, 3.0]);
        let eigenvectors = NaMatrix::identity(2, 2);
        let basis = EigenBasis::from_generalized_hodge_laplace_eigenpairs(
            eigenvalues,
            eigenvectors,
            &mass,
            1.0e-12,
        )
        .unwrap();
        assert_close(basis.eigenvectors()[(0, 0)], 1.0 / 2.0_f64.sqrt(), 1e-12);
        assert_close(basis.eigenvectors()[(1, 1)], 1.0 / 8.0_f64.sqrt(), 1e-12);

        let nonorthogonal = NaMatrix::from_fn(2, 2, |row, col| {
            const VALUES: [[f64; 2]; 2] = [[1.0, 1.0], [0.0, 1.0]];
            VALUES[row][col]
        });
        let error = EigenBasis::from_generalized_hodge_laplace_eigenpairs(
            NaVector::from_vec(vec![1.0, 3.0]),
            nonorthogonal,
            &mass,
            1.0e-12,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::EigenvectorsNotMassOrthogonal { .. }
        ));
    }

    #[test]
    fn smallest_truncation_keeps_smallest_eigenvalues() {
        let mut eigenvectors = Mat::zeros(3, 3);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        eigenvectors[(2, 2)] = 1.0;
        let basis = EigenBasis::new(vec![3.0, 1.0, 2.0], eigenvectors).unwrap();

        let truncated = basis.truncate_smallest(2).unwrap();
        assert_eq!(truncated.eigenvalues(), &[1.0, 2.0]);
    }

    #[test]
    fn low_rank_linear_conditioning_matches_dense_conditioning() {
        use common::linalg::nalgebra::CooMatrix;

        let mut eigenvectors = Mat::zeros(2, 2);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        let eigenvalues = vec![3.0, 1.0];
        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 2,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();

        let mut observation_matrix = CooMatrix::new(1, 2);
        observation_matrix.push(0, 0, 1.0);
        observation_matrix.push(0, 1, 0.5);
        let observation_matrix = CsrMatrix::from(&observation_matrix);
        let observation_values = vec![1.25];
        let dense_covariance = gp.covariance_matrix();
        let noise_variance = 1e-9;

        let low_rank = gp
            .condition_linear_observations(&observation_matrix, &observation_values, noise_variance)
            .unwrap();

        let a0 = 1.0_f64;
        let a1 = 0.5_f64;
        let s = dense_covariance[(idx(0), idx(0))] * a0 * a0
            + 2.0 * dense_covariance[(idx(0), idx(1))] * a0 * a1
            + dense_covariance[(idx(1), idx(1))] * a1 * a1
            + noise_variance;
        let gain0 =
            (dense_covariance[(idx(0), idx(0))] * a0 + dense_covariance[(idx(0), idx(1))] * a1) / s;
        let gain1 =
            (dense_covariance[(idx(1), idx(0))] * a0 + dense_covariance[(idx(1), idx(1))] * a1) / s;
        let dense_mean0 = gain0 * observation_values[0];
        let dense_mean1 = gain1 * observation_values[0];

        assert_close(low_rank.mean[0], dense_mean0, 1e-8);
        assert_close(low_rank.mean[1], dense_mean1, 1e-8);
        assert!(low_rank.variance.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn conditioning_selects_smaller_exact_dense_system() {
        assert_eq!(
            low_rank_conditioning_strategy(2, 8, 1.0e-3),
            LowRankConditioningStrategy::ObservationSpace
        );
        assert_eq!(
            low_rank_conditioning_strategy(8, 2, 1.0e-3),
            LowRankConditioningStrategy::LatentSpace
        );
        assert_eq!(
            low_rank_conditioning_strategy(8, 2, 0.0),
            LowRankConditioningStrategy::ObservationSpace
        );
    }

    #[test]
    fn latent_and_observation_space_conditioning_are_equivalent() {
        let observation_features = Mat::from_fn(4, 2, |row, col| {
            const VALUES: [[f64; 2]; 4] = [[1.0, 0.2], [0.3, 1.1], [0.7, -0.4], [1.2, 0.5]];
            VALUES[row][col]
        });
        let observations = Mat::from_fn(4, 1, |row, _| [0.4, -0.2, 1.1, 0.7][row.unbound()]);
        let noise_variance = 0.15;

        let (observation_mean, observation_covariance) =
            condition_in_observation_space(&observation_features, &observations, noise_variance)
                .unwrap();
        let (latent_mean, latent_covariance) =
            condition_in_latent_space(&observation_features, &observations, noise_variance)
                .unwrap();
        assert_matrix_close(&latent_mean, &observation_mean, 2.0e-13);
        assert_matrix_close(&latent_covariance, &observation_covariance, 2.0e-13);
    }

    #[test]
    fn transformed_output_conditioning_matches_full_posterior_pushforward() {
        use common::linalg::nalgebra::CooMatrix;

        let basis = EigenBasis::new(
            vec![0.5, 2.0],
            Mat::from_fn(3, 2, |row, col| {
                const VALUES: [[f64; 2]; 3] = [[1.0, 0.0], [0.2, 0.8], [0.0, 1.1]];
                VALUES[row][col]
            }),
        )
        .unwrap();
        let gp = SpectralMaternGp::from_eigenbasis(
            basis,
            SpectralMaternConfig {
                kappa: 1.2,
                alpha: 2.0,
                tau: 0.9,
                k: 2,
            },
        )
        .unwrap();

        let mut observation = CooMatrix::new(2, 3);
        observation.push(0, 0, 1.0);
        observation.push(1, 1, 0.5);
        observation.push(1, 2, 1.0);
        let observation = CsrMatrix::from(&observation);
        let values = [0.3, -0.7];
        let noise_variance = 0.2;

        let mut output = CooMatrix::new(2, 3);
        output.push(0, 0, 1.0);
        output.push(0, 2, -0.4);
        output.push(1, 1, 2.0);
        let output = CsrMatrix::from(&output);

        let ambient = gp
            .condition_linear_observations_with_covariance(&observation, &values, noise_variance)
            .unwrap();
        let transformed = gp
            .condition_linear_observations_at_outputs(
                &observation,
                &values,
                noise_variance,
                &output,
            )
            .unwrap();

        for output_row in 0..output.nrows() {
            let mut expected_mean = 0.0;
            for (row, col, value) in output.triplet_iter() {
                if row == output_row {
                    expected_mean += value * ambient.mean[col];
                }
            }
            assert_close(transformed.mean[output_row], expected_mean, 2.0e-13);

            let mut expected_variance = 0.0;
            for (left_row, left_col, left_value) in output.triplet_iter() {
                if left_row != output_row {
                    continue;
                }
                for (right_row, right_col, right_value) in output.triplet_iter() {
                    if right_row == output_row {
                        expected_variance += left_value
                            * ambient.covariance[(idx(left_col), idx(right_col))]
                            * right_value;
                    }
                }
            }
            assert_close(transformed.variance[output_row], expected_variance, 2.0e-13);
        }
    }
}
