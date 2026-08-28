//! Spectral Gaussian process utilities based on Hodge Laplacian eigenfunctions.

#[cfg(feature = "external-solvers")]
mod hodge;

#[cfg(feature = "external-solvers")]
use common::linalg::nalgebra::CooMatrix;
use common::linalg::nalgebra::{CsrMatrix, Matrix as NaMatrix, Vector as NaVector};
#[cfg(feature = "external-solvers")]
use common::linalg::petsc::{
    try_petsc_ghep_reduced_with_which, try_petsc_ghiep, GhepReducedOperators, GhiepReducedSolve,
    GhiepWhich, PetscError, PetscGeneralizedEigenDiagnostics,
    PetscReducedGeneralizedEigenDiagnostics,
};
#[cfg(feature = "external-solvers")]
use exterior::ExteriorGrade;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Llt, Solve};
use faer::{Accum, Idx, Mat, MatRef, Par, Side, Unbind};
#[cfg(feature = "external-solvers")]
use formoniq::{
    assemble,
    operators::HodgeMassElmat,
    problems::{hodge_laplace, laplace_beltrami},
    reduction::DofLayout,
};
use libm::tgamma;
#[cfg(feature = "external-solvers")]
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};
use std::cmp::Ordering;
#[cfg(feature = "external-solvers")]
use std::path::PathBuf;
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
    #[cfg(feature = "external-solvers")]
    #[error("partial generalized eigensolver rank must be in 1..={dimension}, got {requested}")]
    InvalidPartialGeneralizedEigenbasisRank { requested: usize, dimension: usize },
    #[cfg(feature = "external-solvers")]
    #[error(
        "partial generalized eigenproblem requires matching square matrices, got mass {mass_rows}x{mass_cols} and Laplacian {laplacian_rows}x{laplacian_cols}"
    )]
    PartialGeneralizedEigenbasisOperatorDimensionMismatch {
        mass_rows: usize,
        mass_cols: usize,
        laplacian_rows: usize,
        laplacian_cols: usize,
    },
    #[cfg(feature = "external-solvers")]
    #[error("eigenvalue negative tolerance must be finite and non-negative")]
    InvalidEigenvalueNegativeTolerance,
    #[cfg(feature = "external-solvers")]
    #[error("generalized eigenvalue {index} is negative beyond tolerance {tolerance}: {value}")]
    NegativeGeneralizedEigenvalue {
        index: usize,
        value: f64,
        tolerance: f64,
    },
    #[cfg(feature = "external-solvers")]
    #[error("partial eigenbasis resource diagnostic overflow for {quantity}")]
    PartialEigenbasisResourceOverflow { quantity: &'static str },
    #[cfg(feature = "external-solvers")]
    #[error("mixed generalized Hodge eigenbasis requires form degree k > 0, got {degree}")]
    MixedGeneralizedEigenbasisRequiresPositiveDegree { degree: usize },
    #[cfg(feature = "external-solvers")]
    #[error("invalid {space} FEEC boundary layout: {message}")]
    InvalidMixedGeneralizedEigenbasisLayout {
        space: &'static str,
        message: String,
    },
    #[cfg(feature = "external-solvers")]
    #[error(
        "mixed generalized Hodge eigenbasis requires homogeneous boundary data; {space} dof {index} has value {value}"
    )]
    NonHomogeneousMixedGeneralizedEigenbasisLayout {
        space: &'static str,
        index: usize,
        value: f64,
    },
    #[cfg(feature = "external-solvers")]
    #[error(
        "mixed generalized Hodge block {operator} has shape {rows}x{cols}, expected {expected_rows}x{expected_cols} from the supplied full layouts"
    )]
    MixedGeneralizedEigenbasisOperatorDimensionMismatch {
        operator: &'static str,
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: usize,
    },
    #[cfg(feature = "external-solvers")]
    #[error("mixed generalized Hodge eigenbasis has empty active {space} space")]
    EmptyMixedGeneralizedEigenbasisSpace { space: &'static str },
    #[cfg(feature = "external-solvers")]
    #[error("failed to assemble homogeneous full-relative FEEC operators: {message}")]
    MixedGeneralizedEigenbasisAssembly { message: String },
    #[cfg(feature = "external-solvers")]
    #[error(transparent)]
    ExternalGeneralizedEigenproblem(#[from] PetscError),
    #[error(
        "generalized eigenvectors {left} and {right} have mass inner product {value}, exceeding tolerance {tolerance}"
    )]
    EigenvectorsNotMassOrthogonal {
        left: usize,
        right: usize,
        value: f64,
        tolerance: f64,
    },
    #[error("dense reference eigensolver maximum dimension must be positive")]
    InvalidDenseReferenceMaximumDimension,
    #[error(
        "dense reference eigensolver dimension {dimension} exceeds the explicit maximum {maximum}"
    )]
    DenseReferenceDimensionLimitExceeded { dimension: usize, maximum: usize },
    #[error(
        "dense reference generalized eigenproblem requires matching square matrices, got mass {mass_rows}x{mass_cols} and Laplacian {laplacian_rows}x{laplacian_cols}"
    )]
    DenseReferenceOperatorDimensionMismatch {
        mass_rows: usize,
        mass_cols: usize,
        laplacian_rows: usize,
        laplacian_cols: usize,
    },
    #[error("dense reference {operator} entry ({row}, {col}) is non-finite: {value}")]
    NonFiniteDenseReferenceOperatorEntry {
        operator: &'static str,
        row: usize,
        col: usize,
        value: f64,
    },
    #[error("dense reference symmetry tolerance must be finite and non-negative")]
    InvalidDenseReferenceSymmetryTolerance,
    #[error(
        "dense reference {operator} is not symmetric at ({row}, {col}): {left} versus {right} exceeds tolerance {tolerance}"
    )]
    DenseReferenceOperatorNotSymmetric {
        operator: &'static str,
        row: usize,
        col: usize,
        left: f64,
        right: f64,
        tolerance: f64,
    },
    #[error("dense reference mass matrix is not positive definite")]
    DenseReferenceMassNotPositiveDefinite,
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
    #[error("prepared spectral posterior requires finite positive noise variance")]
    NonPositivePreparedNoiseVariance,
    #[error("prepared spectral posterior requires at least one observation row")]
    EmptyPreparedObservationSystem,
    #[error("covariance is not positive definite (cholesky failed)")]
    CholeskyFailed,
    #[error("standard normal length {got} does not match basis size {expected}")]
    SampleDimensionMismatch { expected: usize, got: usize },
    #[error("latent vector length {got} does not match latent dimension {expected}")]
    LatentVectorDimensionMismatch { expected: usize, got: usize },
    #[error("state vector length {got} does not match ambient dimension {expected}")]
    StateVectorDimensionMismatch { expected: usize, got: usize },
    #[error("output vector length {got} does not match output dimension {expected}")]
    OutputVectorDimensionMismatch { expected: usize, got: usize },
    #[error("observation matrix column count {got} does not match ambient dimension {expected}")]
    ObservationMatrixDimensionMismatch { expected: usize, got: usize },
    #[error(
        "observation matrix row count {rows} does not match observation value length {values}"
    )]
    ObservationValueDimensionMismatch { rows: usize, values: usize },
    #[error("observation value {index} is non-finite: {value}")]
    NonFiniteObservationValue { index: usize, value: f64 },
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

/// Validation policy for a production partial generalized eigensolve.
///
/// The adapter solves the boundary-reduced FEEC problem
/// `L v = lambda M v` on the supplied sparse operators. It never substitutes
/// a dense eigensolver when the PETSc/SLEPc helper is unavailable.
#[cfg(feature = "external-solvers")]
#[derive(Debug, Clone, Copy)]
pub struct PartialGeneralizedEigensolverOptions {
    pub requested_rank: usize,
    pub eigenvalue_negative_tolerance: f64,
    pub mass_orthogonality_tolerance: f64,
}

#[cfg(feature = "external-solvers")]
impl PartialGeneralizedEigensolverOptions {
    /// Construct options for the requested number of smallest eigenpairs.
    pub fn new(requested_rank: usize) -> Self {
        Self {
            requested_rank,
            eigenvalue_negative_tolerance: 1.0e-10,
            mass_orthogonality_tolerance: 1.0e-8,
        }
    }
}

/// Auditable resource and external-helper metadata for a partial basis.
#[cfg(feature = "external-solvers")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialGeneralizedEigenbasisDiagnostics {
    pub dimension: usize,
    pub requested_rank: usize,
    pub converged_rank: usize,
    pub retained_rank: usize,
    pub mass_nonzeros: usize,
    pub weak_hodge_laplacian_nonzeros: usize,
    pub input_nonzeros: usize,
    pub basis_storage_entries: usize,
    pub basis_storage_bytes: usize,
    pub clamped_negative_eigenvalues: usize,
    pub workspace: PathBuf,
    pub binary: PathBuf,
}

/// A validated mass-orthonormal basis and the diagnostics for its helper run.
#[cfg(feature = "external-solvers")]
#[derive(Debug, Clone)]
pub struct PartialGeneralizedEigenbasis {
    basis: EigenBasis,
    diagnostics: PartialGeneralizedEigenbasisDiagnostics,
}

/// Policy for a sparse mixed Hodge eigensolve.
///
/// The external helper applies the lower-codifferential contribution through
/// a solve with `M_(k-1)` and never materializes the dense marginal weak Hodge
/// Laplacian. Only the smallest modes are retained.
#[cfg(feature = "external-solvers")]
#[derive(Debug, Clone, Copy)]
pub struct PartialMixedGeneralizedEigensolverOptions {
    pub requested_rank: usize,
    pub eigenvalue_negative_tolerance: f64,
    pub mass_orthogonality_tolerance: f64,
    pub mass_solve: GhiepReducedSolve,
}

#[cfg(feature = "external-solvers")]
impl PartialMixedGeneralizedEigensolverOptions {
    /// Construct a smallest-eigenmode solve using the helper's direct lower-mass policy.
    pub fn new(requested_rank: usize) -> Self {
        Self {
            requested_rank,
            eigenvalue_negative_tolerance: 1.0e-10,
            mass_orthogonality_tolerance: 1.0e-8,
            mass_solve: GhiepReducedSolve::Direct,
        }
    }
}

/// Auditable dimensions, sparsity, storage, and process provenance for one
/// sparse mixed Hodge eigensolve.
#[cfg(feature = "external-solvers")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialMixedGeneralizedEigenbasisDiagnostics {
    pub sigma_dimension: usize,
    pub u_dimension: usize,
    pub requested_rank: usize,
    pub converged_rank: usize,
    pub retained_rank: usize,
    pub l_nonzeros: usize,
    pub d_nonzeros: usize,
    pub c_nonzeros: usize,
    pub mkm1_nonzeros: usize,
    pub mk_nonzeros: usize,
    pub input_nonzeros: usize,
    pub which: GhiepWhich,
    pub mass_solve: GhiepReducedSolve,
    pub basis_storage_entries: usize,
    pub basis_storage_bytes: usize,
    pub clamped_negative_eigenvalues: usize,
    pub workspace: PathBuf,
    pub binary: PathBuf,
}

/// A checked mass-orthonormal basis on the active degree-`k` cochain space.
#[cfg(feature = "external-solvers")]
#[derive(Debug, Clone)]
pub struct PartialMixedGeneralizedEigenbasis {
    basis: EigenBasis,
    diagnostics: PartialMixedGeneralizedEigenbasisDiagnostics,
}

#[cfg(feature = "external-solvers")]
impl PartialMixedGeneralizedEigenbasis {
    pub fn basis(&self) -> &EigenBasis {
        &self.basis
    }

    pub fn diagnostics(&self) -> &PartialMixedGeneralizedEigenbasisDiagnostics {
        &self.diagnostics
    }

    pub fn into_basis(self) -> EigenBasis {
        self.basis
    }
}

#[cfg(feature = "external-solvers")]
impl PartialGeneralizedEigenbasis {
    pub fn basis(&self) -> &EigenBasis {
        &self.basis
    }

    pub fn diagnostics(&self) -> &PartialGeneralizedEigenbasisDiagnostics {
        &self.diagnostics
    }

    pub fn into_basis(self) -> EigenBasis {
        self.basis
    }
}

/// Explicit resource and validation limits for the dense reference
/// generalized Hodge eigensolver.
///
/// The maximum dimension is intentionally supplied by every caller.  This
/// solver materializes several dense square matrices and is suitable only for
/// smoke tests and small reference problems, not production scaling studies.
#[derive(Debug, Clone, Copy)]
pub struct DenseReferenceEigensolverOptions {
    pub maximum_dimension: usize,
    pub symmetry_tolerance: f64,
    pub mass_orthogonality_tolerance: f64,
}

impl DenseReferenceEigensolverOptions {
    /// Construct options with an explicit problem-size ceiling.
    pub fn new(maximum_dimension: usize) -> Self {
        Self {
            maximum_dimension,
            symmetry_tolerance: 1.0e-11,
            mass_orthogonality_tolerance: 1.0e-9,
        }
    }
}

/// Solve `L v = lambda M v` by a dense Cholesky whitening of `M`.
///
/// Both sparse inputs must describe the same boundary-reduced coefficient
/// space.  The routine is a correctness/reference adapter only: it converts
/// both operators to dense matrices, performs a dense Cholesky factorization,
/// and computes the complete spectrum.  Every caller must state a maximum
/// accepted dimension through `options`.
pub fn dense_reference_generalized_hodge_eigenbasis(
    mass: &CsrMatrix,
    weak_hodge_laplacian: &CsrMatrix,
    options: DenseReferenceEigensolverOptions,
) -> Result<EigenBasis, GpError> {
    validate_dense_reference_inputs(mass, weak_hodge_laplacian, options)?;
    let dimension = mass.nrows();
    let mass_dense = dense_reference_matrix(mass, "mass")?;
    let laplacian_dense = dense_reference_matrix(weak_hodge_laplacian, "Hodge Laplacian")?;
    validate_dense_reference_symmetry(&mass_dense, "mass", options.symmetry_tolerance)?;
    validate_dense_reference_symmetry(
        &laplacian_dense,
        "Hodge Laplacian",
        options.symmetry_tolerance,
    )?;

    let mass_cholesky = mass_dense
        .cholesky()
        .ok_or(GpError::DenseReferenceMassNotPositiveDefinite)?;
    let mass_lower_inverse = mass_cholesky
        .l()
        .try_inverse()
        .ok_or(GpError::DenseReferenceMassNotPositiveDefinite)?;
    let whitened = &mass_lower_inverse * laplacian_dense * mass_lower_inverse.transpose();
    let whitened = 0.5 * (&whitened + whitened.transpose());
    let eigen = whitened.symmetric_eigen();
    let generalized_vectors = mass_lower_inverse.transpose() * eigen.eigenvectors;

    EigenBasis::from_generalized_hodge_laplace_eigenpairs(
        eigen.eigenvalues,
        generalized_vectors,
        mass,
        options.mass_orthogonality_tolerance,
    )?
    .truncate_smallest(dimension)
}

/// Compute the smallest generalized Hodge--Laplacian modes with PETSc/SLEPc.
///
/// `mass` and `weak_hodge_laplacian` must be the exact same boundary-reduced
/// operators used to define the discrete law. The external helper may report
/// more modes than requested; this adapter records the actual converged count,
/// mass-normalizes and checks all returned modes through [`EigenBasis`], then
/// sorts and retains exactly the requested number of smallest modes. Returning
/// fewer than the requested number is treated as invalid helper output rather
/// than silently changing the approximation rank.
#[cfg(feature = "external-solvers")]
pub fn partial_generalized_hodge_eigenbasis(
    mass: &CsrMatrix,
    weak_hodge_laplacian: &CsrMatrix,
    options: PartialGeneralizedEigensolverOptions,
) -> Result<PartialGeneralizedEigenbasis, GpError> {
    validate_partial_generalized_eigenbasis_inputs(mass, weak_hodge_laplacian, options)?;

    let result = try_petsc_ghiep(weak_hodge_laplacian, mass, options.requested_rank)?;
    let PetscGeneralizedEigenDiagnostics {
        dimension,
        requested_rank,
        converged_rank,
        lhs_nonzeros,
        rhs_nonzeros,
        workspace,
        binary,
    } = result.diagnostics;
    if converged_rank < requested_rank {
        return Err(GpError::ExternalGeneralizedEigenproblem(
            PetscError::InvalidHelperOutput {
                binary,
                workspace,
                detail: format!(
                    "the helper converged {converged_rank} eigenpairs, fewer than the requested {requested_rank}"
                ),
            },
        ));
    }

    let mut eigenvalues = result.eigenvalues;
    let clamped_negative_eigenvalues = validate_and_clamp_generalized_eigenvalues(
        &mut eigenvalues,
        options.eigenvalue_negative_tolerance,
    )?;
    let basis = EigenBasis::from_generalized_hodge_laplace_eigenpairs(
        eigenvalues,
        result.eigenvectors,
        mass,
        options.mass_orthogonality_tolerance,
    )?
    .truncate_smallest(requested_rank)?;

    let input_nonzeros = lhs_nonzeros.checked_add(rhs_nonzeros).ok_or(
        GpError::PartialEigenbasisResourceOverflow {
            quantity: "input nonzero count",
        },
    )?;
    let basis_storage_entries =
        dimension
            .checked_mul(basis.len())
            .ok_or(GpError::PartialEigenbasisResourceOverflow {
                quantity: "basis entry count",
            })?;
    let basis_storage_bytes = basis_storage_entries
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(GpError::PartialEigenbasisResourceOverflow {
            quantity: "basis byte count",
        })?;

    Ok(PartialGeneralizedEigenbasis {
        diagnostics: PartialGeneralizedEigenbasisDiagnostics {
            dimension,
            requested_rank,
            converged_rank,
            retained_rank: basis.len(),
            mass_nonzeros: rhs_nonzeros,
            weak_hodge_laplacian_nonzeros: lhs_nonzeros,
            input_nonzeros,
            basis_storage_entries,
            basis_storage_bytes,
            clamped_negative_eigenvalues,
            workspace,
            binary,
        },
        basis,
    })
}

/// Compute the smallest generalized Hodge--Laplacian modes from FEEC mixed blocks.
///
/// `sigma_layout` and `u_layout` must be the homogeneous layouts for the
/// degree-`k-1` and degree-`k` spaces, respectively. The implementation
/// restricts every [`hodge_laplace::MixedGalmats`] block with those same active
/// index sets before invoking the sparse PETSc/SLEPc shell operator
/// `L + D M_(k-1)^(-1) C`. It does not assemble the marginal operator and does
/// not use a dense fallback.
#[cfg(feature = "external-solvers")]
pub fn partial_mixed_generalized_hodge_eigenbasis(
    galmats: &hodge_laplace::MixedGalmats,
    sigma_layout: &DofLayout,
    u_layout: &DofLayout,
    options: PartialMixedGeneralizedEigensolverOptions,
) -> Result<PartialMixedGeneralizedEigenbasis, GpError> {
    validate_partial_mixed_options(options, u_layout.reduced_dimension())?;
    validate_homogeneous_layout("degree-k-1", sigma_layout)?;
    validate_homogeneous_layout("degree-k", u_layout)?;
    if sigma_layout.reduced_dimension() == 0 {
        return Err(GpError::EmptyMixedGeneralizedEigenbasisSpace {
            space: "degree-k-1",
        });
    }
    if u_layout.reduced_dimension() == 0 {
        return Err(GpError::EmptyMixedGeneralizedEigenbasisSpace { space: "degree-k" });
    }

    let mass_sigma_full = CsrMatrix::from(galmats.mass_sigma());
    let dif_sigma_full = CsrMatrix::from(galmats.dif_sigma());
    let codif_u_full = CsrMatrix::from(galmats.codif_u());
    let codifdif_u_full = if galmats.codifdif_u().nrows() == 0 {
        CsrMatrix::zeros(u_layout.full_dimension, u_layout.full_dimension)
    } else {
        CsrMatrix::from(galmats.codifdif_u())
    };
    let mass_u_full = CsrMatrix::from(galmats.mass_u());

    validate_mixed_full_shape(
        "M_(k-1)",
        &mass_sigma_full,
        sigma_layout.full_dimension,
        sigma_layout.full_dimension,
    )?;
    validate_mixed_full_shape(
        "D",
        &dif_sigma_full,
        u_layout.full_dimension,
        sigma_layout.full_dimension,
    )?;
    validate_mixed_full_shape(
        "C",
        &codif_u_full,
        sigma_layout.full_dimension,
        u_layout.full_dimension,
    )?;
    validate_mixed_full_shape(
        "L",
        &codifdif_u_full,
        u_layout.full_dimension,
        u_layout.full_dimension,
    )?;
    validate_mixed_full_shape(
        "M_k",
        &mass_u_full,
        u_layout.full_dimension,
        u_layout.full_dimension,
    )?;

    let mkm1 = restrict_mixed_operator(&mass_sigma_full, sigma_layout, sigma_layout);
    let d = restrict_mixed_operator(&dif_sigma_full, u_layout, sigma_layout);
    let c = restrict_mixed_operator(&codif_u_full, sigma_layout, u_layout);
    let l = restrict_mixed_operator(&codifdif_u_full, u_layout, u_layout);
    let mk = restrict_mixed_operator(&mass_u_full, u_layout, u_layout);
    let result = try_petsc_ghep_reduced_with_which(
        GhepReducedOperators {
            l: &l,
            d: &d,
            c: &c,
            mkm1: &mkm1,
            mk: &mk,
        },
        options.requested_rank,
        GhiepWhich::Smallest,
        options.mass_solve,
    )?;
    let PetscReducedGeneralizedEigenDiagnostics {
        sigma_dimension,
        u_dimension,
        requested_rank,
        converged_rank,
        l_nonzeros,
        d_nonzeros,
        c_nonzeros,
        mkm1_nonzeros,
        mk_nonzeros,
        input_nonzeros,
        which,
        mass_solve,
        workspace,
        binary,
    } = result.diagnostics;

    let mut eigenvalues = result.eigenvalues;
    let clamped_negative_eigenvalues = validate_and_clamp_generalized_eigenvalues(
        &mut eigenvalues,
        options.eigenvalue_negative_tolerance,
    )?;
    let basis = EigenBasis::from_generalized_hodge_laplace_eigenpairs(
        eigenvalues,
        result.u_eigenvectors,
        &mk,
        options.mass_orthogonality_tolerance,
    )?
    .truncate_smallest(options.requested_rank)?;
    let basis_storage_entries =
        u_dimension
            .checked_mul(basis.len())
            .ok_or(GpError::PartialEigenbasisResourceOverflow {
                quantity: "mixed basis entry count",
            })?;
    let basis_storage_bytes = basis_storage_entries
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(GpError::PartialEigenbasisResourceOverflow {
            quantity: "mixed basis byte count",
        })?;

    Ok(PartialMixedGeneralizedEigenbasis {
        diagnostics: PartialMixedGeneralizedEigenbasisDiagnostics {
            sigma_dimension,
            u_dimension,
            requested_rank,
            converged_rank,
            retained_rank: basis.len(),
            l_nonzeros,
            d_nonzeros,
            c_nonzeros,
            mkm1_nonzeros,
            mk_nonzeros,
            input_nonzeros,
            which,
            mass_solve,
            basis_storage_entries,
            basis_storage_bytes,
            clamped_negative_eigenvalues,
            workspace,
            binary,
        },
        basis,
    })
}

/// Assemble homogeneous full-relative degree-`k-1` and degree-`k` layouts,
/// then solve the sparse mixed generalized eigenproblem. The root package's
/// `ExactFormOperators` constructor uses the same FEEC assembly entry point.
///
/// The degree must be positive. This convenience adapter deliberately calls
/// `formoniq::relative::assemble_whitney_full_relative_operators`, so its
/// active ordering is the FEEC ordering rather than an independently inferred
/// boundary convention.
#[cfg(feature = "external-solvers")]
pub fn partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
    topology: &Complex,
    geometry: &MeshLengths,
    degree: usize,
    options: PartialMixedGeneralizedEigensolverOptions,
) -> Result<PartialMixedGeneralizedEigenbasis, GpError> {
    if degree == 0 {
        return Err(GpError::MixedGeneralizedEigenbasisRequiresPositiveDegree { degree });
    }
    let relative =
        formoniq::relative::assemble_whitney_full_relative_operators(topology, geometry, degree)
            .map_err(|error| GpError::MixedGeneralizedEigenbasisAssembly {
                message: error.to_string(),
            })?;
    let lower = relative
        .lower()
        .ok_or(GpError::MixedGeneralizedEigenbasisRequiresPositiveDegree { degree })?;
    let galmats = hodge_laplace::MixedGalmats::compute(topology, geometry, degree);
    partial_mixed_generalized_hodge_eigenbasis(
        &galmats,
        lower.layout(),
        relative.state().layout(),
        options,
    )
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
        mass_normalize_and_validate_eigenvectors(mass, &mut eigenvectors, orthogonality_tolerance)?;
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

/// Persistent dimensions and floating-point storage owned by a prepared
/// spectral posterior.
///
/// These counts exclude the eigensolver, the source `SpectralMaternGp` and its
/// eigenbasis if the caller retains them, and temporary mean/output workspaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedSpectralPosteriorDiagnostics {
    pub ambient_dimension: usize,
    pub latent_dimension: usize,
    pub observation_count: usize,
    pub observation_operator_nonzeros: usize,
    pub conditioning_strategy: LowRankConditioningStrategy,
    pub dense_factor_dimension: usize,
    /// Allocated entries in the factor's dense square backing matrix.
    pub dense_factor_allocated_entries: usize,
    /// Mathematically active entries in its lower triangle.
    pub dense_factor_triangular_entries: usize,
    pub state_feature_entries: usize,
    pub observation_feature_entries: usize,
    pub persistent_f64_entries: usize,
    pub sampling_standard_normal_dimension: usize,
    pub noise_variance: f64,
}

/// A fixed-operator spectral posterior that reuses one adaptive dense factor.
///
/// The object stores an `N x r` feature matrix, its `m x r` observation
/// projection, and one dense Cholesky factor of dimension `min(m, r)` for
/// positive scalar observation noise.  It never constructs an `N x N`
/// covariance.  Replacement observation values reuse the same factor.
#[derive(Debug, Clone)]
pub struct PreparedSpectralPosterior {
    state_features: Mat<f64>,
    observation_features: Mat<f64>,
    noise_variance: f64,
    strategy: LowRankConditioningStrategy,
    factor: Llt<f64>,
    observation_operator_nonzeros: usize,
}

impl PreparedSpectralPosterior {
    fn new(
        state_features: Mat<f64>,
        observation_matrix: &CsrMatrix,
        noise_variance: f64,
    ) -> Result<Self, GpError> {
        if !noise_variance.is_finite() || noise_variance <= 0.0 {
            return Err(GpError::NonPositivePreparedNoiseVariance);
        }
        if observation_matrix.ncols() != state_features.nrows() {
            return Err(GpError::ObservationMatrixDimensionMismatch {
                expected: state_features.nrows(),
                got: observation_matrix.ncols(),
            });
        }
        if observation_matrix.nrows() == 0 {
            return Err(GpError::EmptyPreparedObservationSystem);
        }
        let observation_features = sparse_dense_product(observation_matrix, &state_features)?;
        let strategy = low_rank_conditioning_strategy(
            observation_features.nrows(),
            observation_features.ncols(),
            noise_variance,
        );
        let system = conditioning_system(&observation_features, noise_variance, strategy);
        let factor = system
            .llt(Side::Lower)
            .map_err(|_| GpError::CholeskyFailed)?;
        Ok(Self {
            state_features,
            observation_features,
            noise_variance,
            strategy,
            factor,
            observation_operator_nonzeros: observation_matrix.triplet_iter().count(),
        })
    }

    pub fn ambient_dimension(&self) -> usize {
        self.state_features.nrows()
    }

    pub fn latent_dimension(&self) -> usize {
        self.state_features.ncols()
    }

    pub fn observation_count(&self) -> usize {
        self.observation_features.nrows()
    }

    pub fn conditioning_strategy(&self) -> LowRankConditioningStrategy {
        self.strategy
    }

    pub fn diagnostics(&self) -> PreparedSpectralPosteriorDiagnostics {
        let factor_dimension = self.factor.L().nrows();
        let state_feature_entries = self.ambient_dimension() * self.latent_dimension();
        let observation_feature_entries = self.observation_count() * self.latent_dimension();
        let dense_factor_allocated_entries = factor_dimension * factor_dimension;
        PreparedSpectralPosteriorDiagnostics {
            ambient_dimension: self.ambient_dimension(),
            latent_dimension: self.latent_dimension(),
            observation_count: self.observation_count(),
            observation_operator_nonzeros: self.observation_operator_nonzeros,
            conditioning_strategy: self.strategy,
            dense_factor_dimension: factor_dimension,
            dense_factor_allocated_entries,
            dense_factor_triangular_entries: factor_dimension * (factor_dimension + 1) / 2,
            state_feature_entries,
            observation_feature_entries,
            persistent_f64_entries: state_feature_entries
                + observation_feature_entries
                + dense_factor_allocated_entries,
            sampling_standard_normal_dimension: match self.strategy {
                LowRankConditioningStrategy::ObservationSpace => {
                    self.latent_dimension() + self.observation_count()
                }
                LowRankConditioningStrategy::LatentSpace => self.latent_dimension(),
            },
            noise_variance: self.noise_variance,
        }
    }

    /// Posterior latent mean for replacement observation values.
    pub fn latent_mean(&self, observation_values: &[f64]) -> Result<Vec<f64>, GpError> {
        self.validate_observation_values(observation_values)?;
        let values = Mat::from_fn(self.observation_count(), 1, |row, _| {
            observation_values[row.unbound()]
        });
        let mean = match self.strategy {
            LowRankConditioningStrategy::ObservationSpace => {
                let mut solved = values;
                self.factor.solve_in_place(solved.as_mut());
                dense_transpose_times_matrix(&self.observation_features, &solved)
            }
            LowRankConditioningStrategy::LatentSpace => {
                let mut rhs = dense_transpose_times_matrix(&self.observation_features, &values);
                self.factor.solve_in_place(rhs.as_mut());
                rhs
            }
        };
        Ok(mat_col_to_vec(&mean))
    }

    /// Posterior coefficient-space mean for replacement observation values.
    pub fn state_mean(&self, observation_values: &[f64]) -> Result<Vec<f64>, GpError> {
        let latent = self.latent_mean(observation_values)?;
        Ok(dense_matrix_times_vector(&self.state_features, &latent))
    }

    /// Apply the posterior latent covariance without materializing it.
    pub fn latent_covariance_action(&self, input: &[f64]) -> Result<Vec<f64>, GpError> {
        self.validate_latent_vector(input)?;
        match self.strategy {
            LowRankConditioningStrategy::ObservationSpace => {
                let projected = dense_matrix_times_vector(&self.observation_features, input);
                let mut solved = Mat::from_fn(self.observation_count(), 1, |row, _| {
                    projected[row.unbound()]
                });
                self.factor.solve_in_place(solved.as_mut());
                let correction = dense_transpose_times_matrix(&self.observation_features, &solved);
                Ok(input
                    .iter()
                    .zip(mat_col_to_vec(&correction))
                    .map(|(prior, removed)| prior - removed)
                    .collect())
            }
            LowRankConditioningStrategy::LatentSpace => {
                let mut solved =
                    Mat::from_fn(self.latent_dimension(), 1, |row, _| input[row.unbound()]);
                self.factor.solve_in_place(solved.as_mut());
                Ok(mat_col_to_vec(&solved)
                    .into_iter()
                    .map(|value| self.noise_variance * value)
                    .collect())
            }
        }
    }

    /// Apply the posterior coefficient covariance `F C_z F^T`.
    pub fn state_covariance_action(&self, input: &[f64]) -> Result<Vec<f64>, GpError> {
        if input.len() != self.ambient_dimension() {
            return Err(GpError::StateVectorDimensionMismatch {
                expected: self.ambient_dimension(),
                got: input.len(),
            });
        }
        let latent_input = dense_transpose_times_vector(&self.state_features, input);
        let latent_output = self.latent_covariance_action(&latent_input)?;
        Ok(dense_matrix_times_vector(
            &self.state_features,
            &latent_output,
        ))
    }

    /// Posterior moments after a sparse physical/output map.
    pub fn output_moments(
        &self,
        output_matrix: &CsrMatrix,
        observation_values: &[f64],
    ) -> Result<ConditionedGp, GpError> {
        let output_features = sparse_dense_product(output_matrix, &self.state_features)?;
        let latent_mean = self.latent_mean(observation_values)?;
        let mean = dense_matrix_times_vector(&output_features, &latent_mean);
        let variance = self.feature_variances(&output_features)?;
        Ok(ConditionedGp { mean, variance })
    }

    /// Posterior moments and full output covariance after a sparse output map.
    ///
    /// Only the requested output covariance is materialized; the ambient
    /// coefficient covariance is not.
    pub fn output_moments_with_covariance(
        &self,
        output_matrix: &CsrMatrix,
        observation_values: &[f64],
    ) -> Result<ConditionedFullGp, GpError> {
        let output_features = sparse_dense_product(output_matrix, &self.state_features)?;
        let latent_mean = self.latent_mean(observation_values)?;
        let mean = dense_matrix_times_vector(&output_features, &latent_mean);
        let covariance = self.feature_covariance(&output_features)?;
        Ok(ConditionedFullGp { mean, covariance })
    }

    /// Apply the posterior covariance in a requested sparse output space.
    pub fn output_covariance_action(
        &self,
        output_matrix: &CsrMatrix,
        input: &[f64],
    ) -> Result<Vec<f64>, GpError> {
        let output_features = sparse_dense_product(output_matrix, &self.state_features)?;
        if input.len() != output_features.nrows() {
            return Err(GpError::OutputVectorDimensionMismatch {
                expected: output_features.nrows(),
                got: input.len(),
            });
        }
        let latent_input = dense_transpose_times_vector(&output_features, input);
        let latent_output = self.latent_covariance_action(&latent_input)?;
        Ok(dense_matrix_times_vector(&output_features, &latent_output))
    }

    /// Draw a latent posterior sample from independent standard normals.
    pub fn latent_sample_from_standard_normal(
        &self,
        observation_values: &[f64],
        standard_normal: &[f64],
    ) -> Result<Vec<f64>, GpError> {
        self.validate_observation_values(observation_values)?;
        let expected = self.diagnostics().sampling_standard_normal_dimension;
        if standard_normal.len() != expected {
            return Err(GpError::SampleDimensionMismatch {
                expected,
                got: standard_normal.len(),
            });
        }
        match self.strategy {
            LowRankConditioningStrategy::ObservationSpace => {
                let latent_dimension = self.latent_dimension();
                let prior_latent = &standard_normal[..latent_dimension];
                let noise_standard = &standard_normal[latent_dimension..];
                let synthetic = dense_matrix_times_vector(&self.observation_features, prior_latent);
                let residual = observation_values
                    .iter()
                    .zip(synthetic)
                    .zip(noise_standard)
                    .map(|((observed, predicted), noise)| {
                        observed - predicted - self.noise_variance.sqrt() * noise
                    })
                    .collect::<Vec<_>>();
                let mut solved = Mat::from_fn(self.observation_count(), 1, |row, _| {
                    residual[row.unbound()]
                });
                self.factor.solve_in_place(solved.as_mut());
                let correction = mat_col_to_vec(&dense_transpose_times_matrix(
                    &self.observation_features,
                    &solved,
                ));
                Ok(prior_latent
                    .iter()
                    .zip(correction)
                    .map(|(prior, update)| prior + update)
                    .collect())
            }
            LowRankConditioningStrategy::LatentSpace => {
                let mean = self.latent_mean(observation_values)?;
                let innovation = solve_lower_transpose(self.factor.L(), standard_normal);
                Ok(mean
                    .into_iter()
                    .zip(innovation)
                    .map(|(mean, innovation)| mean + self.noise_variance.sqrt() * innovation)
                    .collect())
            }
        }
    }

    /// Draw an ambient coefficient posterior sample without an ambient
    /// covariance matrix.
    pub fn state_sample_from_standard_normal(
        &self,
        observation_values: &[f64],
        standard_normal: &[f64],
    ) -> Result<Vec<f64>, GpError> {
        let latent =
            self.latent_sample_from_standard_normal(observation_values, standard_normal)?;
        Ok(dense_matrix_times_vector(&self.state_features, &latent))
    }

    /// Draw a posterior sample directly in a sparse physical/output space.
    pub fn output_sample_from_standard_normal(
        &self,
        output_matrix: &CsrMatrix,
        observation_values: &[f64],
        standard_normal: &[f64],
    ) -> Result<Vec<f64>, GpError> {
        let latent =
            self.latent_sample_from_standard_normal(observation_values, standard_normal)?;
        let output_features = sparse_dense_product(output_matrix, &self.state_features)?;
        Ok(dense_matrix_times_vector(&output_features, &latent))
    }

    fn validate_observation_values(&self, values: &[f64]) -> Result<(), GpError> {
        if values.len() != self.observation_count() {
            return Err(GpError::ObservationValueDimensionMismatch {
                rows: self.observation_count(),
                values: values.len(),
            });
        }
        if let Some((index, &value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(GpError::NonFiniteObservationValue { index, value });
        }
        Ok(())
    }

    fn validate_latent_vector(&self, input: &[f64]) -> Result<(), GpError> {
        if input.len() != self.latent_dimension() {
            return Err(GpError::LatentVectorDimensionMismatch {
                expected: self.latent_dimension(),
                got: input.len(),
            });
        }
        Ok(())
    }

    fn feature_variances(&self, features: &Mat<f64>) -> Result<Vec<f64>, GpError> {
        let mut variances = Vec::with_capacity(features.nrows());
        for row in 0..features.nrows() {
            let feature = (0..features.ncols())
                .map(|col| features[(row, col)])
                .collect::<Vec<_>>();
            let covariance_feature = self.latent_covariance_action(&feature)?;
            // Preserve the computed quadratic form, including a possible tiny
            // negative roundoff residual, so callers can archive and apply
            // their explicit uncertainty-stabilization policy.  This prepared
            // comparator must not silently hide material instability.
            variances.push(
                feature
                    .iter()
                    .zip(covariance_feature)
                    .map(|(left, right)| left * right)
                    .sum::<f64>(),
            );
        }
        Ok(variances)
    }

    fn feature_covariance(&self, features: &Mat<f64>) -> Result<Mat<f64>, GpError> {
        let covariance_actions = (0..features.nrows())
            .map(|row| {
                let feature = (0..features.ncols())
                    .map(|col| features[(row, col)])
                    .collect::<Vec<_>>();
                self.latent_covariance_action(&feature)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Mat::from_fn(
            features.nrows(),
            features.nrows(),
            |row, col| {
                let right = row.unbound();
                let left = col.unbound();
                (0..features.ncols())
                    .map(|latent| features[(left, latent)] * covariance_actions[right][latent])
                    .sum()
            },
        ))
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

    /// Prior covariance after a sparse output map.
    ///
    /// Only the requested output covariance is materialized. The ambient
    /// coefficient covariance is never constructed.
    pub fn prior_output_covariance(&self, output_matrix: &CsrMatrix) -> Result<Mat<f64>, GpError> {
        let output_features = sparse_dense_product(output_matrix, &self.feature_matrix())?;
        Ok(full_covariance_from_feature_matrix(&output_features))
    }

    /// Apply the prior covariance in a requested sparse output space.
    pub fn prior_output_covariance_action(
        &self,
        output_matrix: &CsrMatrix,
        input: &[f64],
    ) -> Result<Vec<f64>, GpError> {
        let output_features = sparse_dense_product(output_matrix, &self.feature_matrix())?;
        if input.len() != output_features.nrows() {
            return Err(GpError::OutputVectorDimensionMismatch {
                expected: output_features.nrows(),
                got: input.len(),
            });
        }
        let latent_input = dense_transpose_times_vector(&output_features, input);
        Ok(dense_matrix_times_vector(&output_features, &latent_input))
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

    /// Prepare a reusable posterior for fixed sparse observations and finite
    /// positive scalar noise.  Observation values are supplied later and may
    /// be replaced without refactorization.
    pub fn prepare_linear_observations(
        &self,
        observation_matrix: &CsrMatrix,
        noise_variance: f64,
    ) -> Result<PreparedSpectralPosterior, GpError> {
        PreparedSpectralPosterior::new(self.feature_matrix(), observation_matrix, noise_variance)
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

fn conditioning_system(
    observation_features: &Mat<f64>,
    noise_variance: f64,
    strategy: LowRankConditioningStrategy,
) -> Mat<f64> {
    let dimension = match strategy {
        LowRankConditioningStrategy::ObservationSpace => observation_features.nrows(),
        LowRankConditioningStrategy::LatentSpace => observation_features.ncols(),
    };
    let mut system = Mat::zeros(dimension, dimension);
    match strategy {
        LowRankConditioningStrategy::ObservationSpace => matmul(
            &mut system,
            Accum::Replace,
            observation_features.as_ref(),
            observation_features.as_ref().transpose(),
            1.0,
            Par::Seq,
        ),
        LowRankConditioningStrategy::LatentSpace => matmul(
            &mut system,
            Accum::Replace,
            observation_features.as_ref().transpose(),
            observation_features.as_ref(),
            1.0,
            Par::Seq,
        ),
    }
    for index in 0..dimension {
        system[(idx(index), idx(index))] += noise_variance;
    }
    system
}

fn dense_matrix_times_vector(matrix: &Mat<f64>, vector: &[f64]) -> Vec<f64> {
    debug_assert_eq!(matrix.ncols(), vector.len());
    (0..matrix.nrows())
        .map(|row| {
            vector
                .iter()
                .enumerate()
                .map(|(col, value)| matrix[(row, col)] * value)
                .sum()
        })
        .collect()
}

fn dense_transpose_times_vector(matrix: &Mat<f64>, vector: &[f64]) -> Vec<f64> {
    debug_assert_eq!(matrix.nrows(), vector.len());
    (0..matrix.ncols())
        .map(|col| {
            vector
                .iter()
                .enumerate()
                .map(|(row, value)| matrix[(row, col)] * value)
                .sum()
        })
        .collect()
}

fn dense_transpose_times_matrix(left: &Mat<f64>, right: &Mat<f64>) -> Mat<f64> {
    debug_assert_eq!(left.nrows(), right.nrows());
    let mut product = Mat::zeros(left.ncols(), right.ncols());
    matmul(
        &mut product,
        Accum::Replace,
        left.as_ref().transpose(),
        right.as_ref(),
        1.0,
        Par::Seq,
    );
    product
}

fn solve_lower_transpose(lower: MatRef<'_, f64>, rhs: &[f64]) -> Vec<f64> {
    debug_assert_eq!(lower.nrows(), lower.ncols());
    debug_assert_eq!(lower.nrows(), rhs.len());
    let mut solution = rhs.to_vec();
    for row in (0..lower.nrows()).rev() {
        let mut value = solution[row];
        for col in (row + 1)..lower.ncols() {
            value -= lower[(col, row)] * solution[col];
        }
        solution[row] = value / lower[(row, row)];
    }
    solution
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

/// Mass-normalize all columns and validate their mutual mass orthogonality.
///
/// This forms `M V` once in `O(nnz(M) r)`, then forms the dense Gram matrix
/// `V^T M V` with the optimized dense matrix product used elsewhere in this
/// crate.  The former scalar-pair implementation traversed every mass nonzero
/// for every pair of modes, `O(nnz(M) r^2)`.  Peak workspace here is one
/// explicit `N x r` product and one `r x r` Gram matrix; no copy of `V` is
/// made for the dense product.
fn mass_normalize_and_validate_eigenvectors(
    mass: &CsrMatrix,
    eigenvectors: &mut NaMatrix,
    tolerance: f64,
) -> Result<(), GpError> {
    let row_count = eigenvectors.nrows();
    let rank = eigenvectors.ncols();
    let mass_times_eigenvectors = apply_mass_to_eigenvectors(mass, eigenvectors);
    let eigenvectors_ref = MatRef::from_column_major_slice(
        eigenvectors.as_slice(),
        eigenvectors.nrows(),
        eigenvectors.ncols(),
    );
    let mass_times_ref = MatRef::from_column_major_slice(
        mass_times_eigenvectors.as_slice(),
        mass_times_eigenvectors.nrows(),
        mass_times_eigenvectors.ncols(),
    );
    let mut gram = Mat::zeros(rank, rank);
    matmul(
        &mut gram,
        Accum::Replace,
        eigenvectors_ref.transpose(),
        mass_times_ref,
        1.0,
        Par::Seq,
    );

    let mut scales = Vec::with_capacity(rank);
    for col in 0..rank {
        let norm2 = gram[(idx(col), idx(col))];
        if !norm2.is_finite() || norm2 <= 0.0 {
            return Err(GpError::InvalidEigenvectorNorm { value: norm2 });
        }
        scales.push(1.0 / norm2.sqrt());
    }

    for (col, &scale) in scales.iter().enumerate() {
        for row in 0..row_count {
            eigenvectors[(row, col)] *= scale;
        }
    }

    for left in 0..rank {
        for right in (left + 1)..rank {
            let value = gram[(idx(left), idx(right))] * scales[left] * scales[right];
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

/// Preserve the normalize-only semantics of the legacy mesh constructor while
/// applying `M` to all columns in one sparse--dense product.
#[cfg(any(feature = "external-solvers", test))]
fn mass_normalize_eigenvectors(
    mass: &CsrMatrix,
    eigenvectors: &mut NaMatrix,
) -> Result<(), GpError> {
    let mass_times_eigenvectors = apply_mass_to_eigenvectors(mass, eigenvectors);
    for col in 0..eigenvectors.ncols() {
        let norm2 = eigenvectors
            .column(col)
            .dot(&mass_times_eigenvectors.column(col));
        if !norm2.is_finite() || norm2 <= 0.0 {
            return Err(GpError::InvalidEigenvectorNorm { value: norm2 });
        }
        eigenvectors.column_mut(col).scale_mut(1.0 / norm2.sqrt());
    }
    Ok(())
}

fn apply_mass_to_eigenvectors(mass: &CsrMatrix, eigenvectors: &NaMatrix) -> NaMatrix {
    mass * eigenvectors
}

#[cfg(test)]
fn scalar_mass_normalize_and_validate_reference(
    mass: &CsrMatrix,
    eigenvectors: &mut NaMatrix,
    tolerance: f64,
) -> Result<(), GpError> {
    use common::linalg::nalgebra::quadratic_form_sparse;

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

#[cfg(feature = "external-solvers")]
fn validate_partial_generalized_eigenbasis_inputs(
    mass: &CsrMatrix,
    weak_hodge_laplacian: &CsrMatrix,
    options: PartialGeneralizedEigensolverOptions,
) -> Result<(), GpError> {
    let dimension = mass.nrows();
    if dimension == 0
        || mass.ncols() != dimension
        || weak_hodge_laplacian.nrows() != dimension
        || weak_hodge_laplacian.ncols() != dimension
    {
        return Err(
            GpError::PartialGeneralizedEigenbasisOperatorDimensionMismatch {
                mass_rows: mass.nrows(),
                mass_cols: mass.ncols(),
                laplacian_rows: weak_hodge_laplacian.nrows(),
                laplacian_cols: weak_hodge_laplacian.ncols(),
            },
        );
    }
    if options.requested_rank == 0 || options.requested_rank > dimension {
        return Err(GpError::InvalidPartialGeneralizedEigenbasisRank {
            requested: options.requested_rank,
            dimension,
        });
    }
    if !options.eigenvalue_negative_tolerance.is_finite()
        || options.eigenvalue_negative_tolerance < 0.0
    {
        return Err(GpError::InvalidEigenvalueNegativeTolerance);
    }
    if !options.mass_orthogonality_tolerance.is_finite()
        || options.mass_orthogonality_tolerance < 0.0
    {
        return Err(GpError::InvalidMassOrthogonalityTolerance);
    }
    Ok(())
}

#[cfg(feature = "external-solvers")]
fn validate_partial_mixed_options(
    options: PartialMixedGeneralizedEigensolverOptions,
    u_dimension: usize,
) -> Result<(), GpError> {
    if options.requested_rank == 0 || options.requested_rank > u_dimension {
        return Err(GpError::InvalidPartialGeneralizedEigenbasisRank {
            requested: options.requested_rank,
            dimension: u_dimension,
        });
    }
    if !options.eigenvalue_negative_tolerance.is_finite()
        || options.eigenvalue_negative_tolerance < 0.0
    {
        return Err(GpError::InvalidEigenvalueNegativeTolerance);
    }
    if !options.mass_orthogonality_tolerance.is_finite()
        || options.mass_orthogonality_tolerance < 0.0
    {
        return Err(GpError::InvalidMassOrthogonalityTolerance);
    }
    Ok(())
}

#[cfg(feature = "external-solvers")]
fn validate_homogeneous_layout(space: &'static str, layout: &DofLayout) -> Result<(), GpError> {
    DofLayout::from_parts(
        layout.full_dimension,
        layout.active_dofs.clone(),
        layout.prescribed_dofs.clone(),
    )
    .map_err(|message| GpError::InvalidMixedGeneralizedEigenbasisLayout { space, message })?;
    if let Some(dof) = layout.prescribed_dofs.iter().find(|dof| dof.value != 0.0) {
        return Err(GpError::NonHomogeneousMixedGeneralizedEigenbasisLayout {
            space,
            index: dof.index,
            value: dof.value,
        });
    }
    Ok(())
}

#[cfg(feature = "external-solvers")]
fn validate_mixed_full_shape(
    operator: &'static str,
    matrix: &CsrMatrix,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), GpError> {
    if matrix.nrows() != expected_rows || matrix.ncols() != expected_cols {
        return Err(
            GpError::MixedGeneralizedEigenbasisOperatorDimensionMismatch {
                operator,
                rows: matrix.nrows(),
                cols: matrix.ncols(),
                expected_rows,
                expected_cols,
            },
        );
    }
    Ok(())
}

#[cfg(feature = "external-solvers")]
fn restrict_mixed_operator(full: &CsrMatrix, rows: &DofLayout, columns: &DofLayout) -> CsrMatrix {
    let row_map = active_index_map(rows);
    let column_map = active_index_map(columns);
    let mut reduced = CooMatrix::new(rows.reduced_dimension(), columns.reduced_dimension());
    for (row, column, value) in full.triplet_iter() {
        if let (Some(reduced_row), Some(reduced_column)) = (row_map[row], column_map[column]) {
            reduced.push(reduced_row, reduced_column, *value);
        }
    }
    CsrMatrix::from(&reduced)
}

#[cfg(feature = "external-solvers")]
fn active_index_map(layout: &DofLayout) -> Vec<Option<usize>> {
    let mut map = vec![None; layout.full_dimension];
    for (reduced, &full) in layout.active_dofs.iter().enumerate() {
        map[full] = Some(reduced);
    }
    map
}

#[cfg(feature = "external-solvers")]
fn validate_and_clamp_generalized_eigenvalues(
    eigenvalues: &mut NaVector,
    negative_tolerance: f64,
) -> Result<usize, GpError> {
    let mut clamped = 0;
    for (index, value) in eigenvalues.iter_mut().enumerate() {
        if !value.is_finite() {
            return Err(GpError::NonFiniteEigenvalue { value: *value });
        }
        if *value < -negative_tolerance {
            return Err(GpError::NegativeGeneralizedEigenvalue {
                index,
                value: *value,
                tolerance: negative_tolerance,
            });
        }
        if *value < 0.0 {
            *value = 0.0;
            clamped += 1;
        }
    }
    Ok(clamped)
}

fn validate_dense_reference_inputs(
    mass: &CsrMatrix,
    weak_hodge_laplacian: &CsrMatrix,
    options: DenseReferenceEigensolverOptions,
) -> Result<(), GpError> {
    if options.maximum_dimension == 0 {
        return Err(GpError::InvalidDenseReferenceMaximumDimension);
    }
    if !options.symmetry_tolerance.is_finite() || options.symmetry_tolerance < 0.0 {
        return Err(GpError::InvalidDenseReferenceSymmetryTolerance);
    }
    if !options.mass_orthogonality_tolerance.is_finite()
        || options.mass_orthogonality_tolerance < 0.0
    {
        return Err(GpError::InvalidMassOrthogonalityTolerance);
    }
    let dimension = mass.nrows();
    if dimension == 0
        || mass.ncols() != dimension
        || weak_hodge_laplacian.nrows() != dimension
        || weak_hodge_laplacian.ncols() != dimension
    {
        return Err(GpError::DenseReferenceOperatorDimensionMismatch {
            mass_rows: mass.nrows(),
            mass_cols: mass.ncols(),
            laplacian_rows: weak_hodge_laplacian.nrows(),
            laplacian_cols: weak_hodge_laplacian.ncols(),
        });
    }
    if dimension > options.maximum_dimension {
        return Err(GpError::DenseReferenceDimensionLimitExceeded {
            dimension,
            maximum: options.maximum_dimension,
        });
    }
    Ok(())
}

fn dense_reference_matrix(matrix: &CsrMatrix, operator: &'static str) -> Result<NaMatrix, GpError> {
    let mut dense = NaMatrix::zeros(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if !value.is_finite() {
            return Err(GpError::NonFiniteDenseReferenceOperatorEntry {
                operator,
                row,
                col,
                value: *value,
            });
        }
        dense[(row, col)] += *value;
    }
    Ok(dense)
}

fn validate_dense_reference_symmetry(
    matrix: &NaMatrix,
    operator: &'static str,
    tolerance: f64,
) -> Result<(), GpError> {
    for row in 0..matrix.nrows() {
        for col in (row + 1)..matrix.ncols() {
            let left = matrix[(row, col)];
            let right = matrix[(col, row)];
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > tolerance * scale {
                return Err(GpError::DenseReferenceOperatorNotSymmetric {
                    operator,
                    row,
                    col,
                    left,
                    right,
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
        let mut sparse_laplacian = CooMatrix::new(3, 3);
        for row in 0..3 {
            for col in 0..3 {
                if laplacian[(row, col)] != 0.0 {
                    sparse_laplacian.push(row, col, laplacian[(row, col)]);
                }
            }
        }
        let sparse_laplacian = CsrMatrix::from(&sparse_laplacian);

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

        let basis = dense_reference_generalized_hodge_eigenbasis(
            &sparse_mass,
            &sparse_laplacian,
            DenseReferenceEigensolverOptions::new(3),
        )
        .expect("bounded dense generalized reference solve should succeed");
        let solved_left = dense_product(&laplacian, basis.eigenvectors());
        let solved_mass_basis = dense_product(&mass, basis.eigenvectors());
        let solved_right = Mat::from_fn(3, 3, |row, col| {
            solved_mass_basis[(row, col)] * basis.eigenvalues()[col]
        });
        assert_matrix_close(&solved_left, &solved_right, 2e-12);
        assert_eq!(basis.eigenvalues().len(), 3);

        let kappa = 1.3;
        let tau = 0.7;
        let system = Mat::from_fn(3, 3, |row, col| {
            kappa * kappa * mass[(row, col)] + laplacian[(row, col)]
        });
        let mass_inverse = spd_inverse(&mass);

        let mut unscaled_precision = system.clone();
        for alpha in 1..=3 {
            if alpha > 1 {
                unscaled_precision =
                    dense_product(&dense_product(&unscaled_precision, &mass_inverse), &system);
            }
            let precision =
                Mat::from_fn(3, 3, |row, col| tau * tau * unscaled_precision[(row, col)]);
            let precision_covariance = spd_inverse(&precision);

            let spectral = SpectralMaternGp::from_eigenbasis(
                basis.clone(),
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
        use common::linalg::nalgebra::{quadratic_form_sparse, CooMatrix, Vector as NaVector};

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
    fn optimized_mass_validation_matches_scalar_reference_on_non_diagonal_mass() {
        use common::linalg::nalgebra::CooMatrix;

        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, 0.25, 0.25, 3.0, -0.4, -0.4, 4.0];
        let mass =
            CsrMatrix::from(&CooMatrix::try_from_triplets(3, 3, rows, cols, values).unwrap());
        let mass_dense =
            NaMatrix::from_row_slice(3, 3, &[2.0, 0.25, 0.0, 0.25, 3.0, -0.4, 0.0, -0.4, 4.0]);
        let lower = mass_dense.cholesky().unwrap().l();
        let mut input = lower.transpose().try_inverse().unwrap();
        for (col, scale) in [2.0, 0.5, 4.0].into_iter().enumerate() {
            input.column_mut(col).scale_mut(scale);
        }

        let mut optimized = input.clone();
        let mut scalar = input;
        mass_normalize_and_validate_eigenvectors(&mass, &mut optimized, 1.0e-12).unwrap();
        scalar_mass_normalize_and_validate_reference(&mass, &mut scalar, 1.0e-12).unwrap();

        for row in 0..optimized.nrows() {
            for col in 0..optimized.ncols() {
                assert_close(optimized[(row, col)], scalar[(row, col)], 2.0e-15);
            }
        }
    }

    #[test]
    fn optimized_mass_validation_preserves_first_failure_and_tolerance() {
        use common::linalg::nalgebra::CooMatrix;

        let mass = CsrMatrix::from(
            &CooMatrix::try_from_triplets(
                3,
                3,
                vec![0, 0, 1, 1, 1, 2, 2],
                vec![0, 1, 0, 1, 2, 1, 2],
                vec![2.0, 0.25, 0.25, 3.0, -0.4, -0.4, 4.0],
            )
            .unwrap(),
        );
        let input =
            NaMatrix::from_column_slice(3, 3, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let tolerance = 1.0e-12;
        let mut optimized = input.clone();
        let mut scalar = input;
        let optimized_error =
            mass_normalize_and_validate_eigenvectors(&mass, &mut optimized, tolerance).unwrap_err();
        let scalar_error =
            scalar_mass_normalize_and_validate_reference(&mass, &mut scalar, tolerance)
                .unwrap_err();

        match (optimized_error, scalar_error) {
            (
                GpError::EigenvectorsNotMassOrthogonal {
                    left: optimized_left,
                    right: optimized_right,
                    value: optimized_value,
                    tolerance: optimized_tolerance,
                },
                GpError::EigenvectorsNotMassOrthogonal {
                    left: scalar_left,
                    right: scalar_right,
                    value: scalar_value,
                    tolerance: scalar_tolerance,
                },
            ) => {
                assert_eq!(
                    (optimized_left, optimized_right),
                    (scalar_left, scalar_right)
                );
                assert_eq!(optimized_tolerance, scalar_tolerance);
                assert_close(optimized_value, scalar_value, 2.0e-15);
            }
            errors => panic!("unexpected errors: {errors:?}"),
        }
    }

    #[test]
    fn optimized_mass_validation_rejects_zero_norm_column() {
        use common::linalg::nalgebra::CooMatrix;

        let mass = CsrMatrix::from(
            &CooMatrix::try_from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![2.0, 3.0]).unwrap(),
        );
        let mut eigenvectors = NaMatrix::from_column_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let error = mass_normalize_and_validate_eigenvectors(&mass, &mut eigenvectors, 1.0e-12)
            .unwrap_err();
        assert!(matches!(
            error,
            GpError::InvalidEigenvectorNorm { value: 0.0 }
        ));
    }

    #[test]
    #[ignore = "release-only comparison of optimized and scalar-reference validation"]
    fn mass_validation_microbenchmark() {
        use common::linalg::nalgebra::CooMatrix;
        use std::time::Instant;

        let dimension = 512;
        let rank = 192;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();
        let mut dense_mass = NaMatrix::zeros(dimension, dimension);
        for row in 0..dimension {
            for offset in -5_isize..=5 {
                let col = row as isize + offset;
                if !(0..dimension as isize).contains(&col) {
                    continue;
                }
                let col = col as usize;
                let value = if offset == 0 {
                    4.0
                } else {
                    -0.15 / offset.unsigned_abs() as f64
                };
                rows.push(row);
                cols.push(col);
                values.push(value);
                dense_mass[(row, col)] = value;
            }
        }
        let mass = CsrMatrix::from(
            &CooMatrix::try_from_triplets(dimension, dimension, rows, cols, values).unwrap(),
        );
        let lower = dense_mass.cholesky().unwrap().l();
        let selectors = NaMatrix::from_fn(
            dimension,
            rank,
            |row, col| if row == col { 1.0 } else { 0.0 },
        );
        let input = lower.transpose().lu().solve(&selectors).unwrap();

        let mut optimized = input.clone();
        let optimized_start = Instant::now();
        mass_normalize_and_validate_eigenvectors(&mass, &mut optimized, 1.0e-10).unwrap();
        let optimized_elapsed = optimized_start.elapsed();

        let mut scalar = input;
        let scalar_start = Instant::now();
        scalar_mass_normalize_and_validate_reference(&mass, &mut scalar, 1.0e-10).unwrap();
        let scalar_elapsed = scalar_start.elapsed();
        for row in 0..dimension {
            for col in 0..rank {
                assert_close(optimized[(row, col)], scalar[(row, col)], 1.0e-12);
            }
        }
        eprintln!(
            "mass validation N={dimension} r={rank}: optimized={optimized_elapsed:?}, scalar={scalar_elapsed:?}, speedup={:.3}",
            scalar_elapsed.as_secs_f64() / optimized_elapsed.as_secs_f64()
        );
    }

    #[test]
    #[ignore = "bounded rank-2048 workspace check; run explicitly in release mode"]
    fn mass_validation_rank_2048_resource_check() {
        use std::time::Instant;

        let rank = 2048;
        let mass = CsrMatrix::identity(rank);
        let mut eigenvectors = NaMatrix::identity(rank, rank);
        let workspace_entries = rank * rank + rank * rank;
        let workspace_bytes = workspace_entries * std::mem::size_of::<f64>();
        assert_eq!(workspace_bytes, 64 * 1024 * 1024);

        let start = Instant::now();
        mass_normalize_and_validate_eigenvectors(&mass, &mut eigenvectors, 0.0).unwrap();
        eprintln!(
            "mass validation N={rank} r={rank}: elapsed={:?}, explicit workspace={} MiB",
            start.elapsed(),
            workspace_bytes / (1024 * 1024)
        );
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    fn partial_eigensolver_validation_rejects_rank_dimensions_and_tolerances() {
        use common::linalg::nalgebra::CooMatrix;

        let mut identity = CooMatrix::new(2, 2);
        identity.push(0, 0, 1.0);
        identity.push(1, 1, 1.0);
        let identity = CsrMatrix::from(&identity);

        for requested_rank in [0, 3] {
            let error = validate_partial_generalized_eigenbasis_inputs(
                &identity,
                &identity,
                PartialGeneralizedEigensolverOptions::new(requested_rank),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                GpError::InvalidPartialGeneralizedEigenbasisRank {
                    requested,
                    dimension: 2
                } if requested == requested_rank
            ));
        }

        let mut rectangular = CooMatrix::new(2, 3);
        rectangular.push(0, 0, 1.0);
        rectangular.push(1, 1, 1.0);
        let error = validate_partial_generalized_eigenbasis_inputs(
            &identity,
            &CsrMatrix::from(&rectangular),
            PartialGeneralizedEigensolverOptions::new(1),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::PartialGeneralizedEigenbasisOperatorDimensionMismatch { .. }
        ));

        let mut invalid = PartialGeneralizedEigensolverOptions::new(1);
        invalid.eigenvalue_negative_tolerance = f64::NAN;
        assert!(matches!(
            validate_partial_generalized_eigenbasis_inputs(&identity, &identity, invalid),
            Err(GpError::InvalidEigenvalueNegativeTolerance)
        ));
        let mut invalid = PartialGeneralizedEigensolverOptions::new(1);
        invalid.mass_orthogonality_tolerance = -1.0;
        assert!(matches!(
            validate_partial_generalized_eigenbasis_inputs(&identity, &identity, invalid),
            Err(GpError::InvalidMassOrthogonalityTolerance)
        ));
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    fn partial_eigenvalue_validation_clamps_roundoff_but_rejects_invalid_values() {
        let mut values = NaVector::from_vec(vec![2.0, -5.0e-11, 1.0]);
        assert_eq!(
            validate_and_clamp_generalized_eigenvalues(&mut values, 1.0e-10).unwrap(),
            1
        );
        assert_eq!(values.as_slice(), &[2.0, 0.0, 1.0]);

        let mut values = NaVector::from_vec(vec![-1.0e-4]);
        assert!(matches!(
            validate_and_clamp_generalized_eigenvalues(&mut values, 1.0e-10),
            Err(GpError::NegativeGeneralizedEigenvalue { index: 0, .. })
        ));

        let mut values = NaVector::from_vec(vec![f64::INFINITY]);
        assert!(matches!(
            validate_and_clamp_generalized_eigenvalues(&mut values, 1.0e-10),
            Err(GpError::NonFiniteEigenvalue { .. })
        ));
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    fn mixed_eigensolver_validates_exact_layouts_before_launch() {
        use formoniq::reduction::PrescribedDof;
        use manifold::gen::cartesian::CartesianMeshInfo;

        let mesh = CartesianMeshInfo::new_unit(2, 2);
        let (topology, coordinates) = mesh.compute_coord_complex();
        let geometry = coordinates.to_edge_lengths(&topology);
        let relative =
            formoniq::relative::assemble_whitney_full_relative_operators(&topology, &geometry, 1)
                .unwrap();
        let galmats = hodge_laplace::MixedGalmats::compute(&topology, &geometry, 1);
        let lower_layout = relative.lower().unwrap().layout();
        let state_layout = relative.state().layout();

        let error = partial_mixed_generalized_hodge_eigenbasis(
            &galmats,
            lower_layout,
            state_layout,
            PartialMixedGeneralizedEigensolverOptions::new(0),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::InvalidPartialGeneralizedEigenbasisRank { requested: 0, .. }
        ));

        let mut nonhomogeneous = lower_layout.clone();
        nonhomogeneous.prescribed_dofs[0] = PrescribedDof {
            index: nonhomogeneous.prescribed_dofs[0].index,
            value: 1.0,
        };
        let error = partial_mixed_generalized_hodge_eigenbasis(
            &galmats,
            &nonhomogeneous,
            state_layout,
            PartialMixedGeneralizedEigensolverOptions::new(1),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::NonHomogeneousMixedGeneralizedEigenbasisLayout { .. }
        ));

        let wrong_state = DofLayout::identity(state_layout.full_dimension + 1);
        let error = partial_mixed_generalized_hodge_eigenbasis(
            &galmats,
            lower_layout,
            &wrong_state,
            PartialMixedGeneralizedEigensolverOptions::new(1),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::MixedGeneralizedEigenbasisOperatorDimensionMismatch { operator: "D", .. }
        ));
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    fn mixed_eigensolver_requires_positive_degree() {
        use manifold::gen::cartesian::CartesianMeshInfo;

        let mesh = CartesianMeshInfo::new_unit(2, 1);
        let (topology, coordinates) = mesh.compute_coord_complex();
        let geometry = coordinates.to_edge_lengths(&topology);
        let error = partial_mixed_generalized_hodge_eigenbasis_from_feec_homogeneous_full_relative(
            &topology,
            &geometry,
            0,
            PartialMixedGeneralizedEigensolverOptions::new(1),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::MixedGeneralizedEigenbasisRequiresPositiveDegree { degree: 0 }
        ));
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    #[ignore = "requires built PETSc/SLEPc reduced Hodge helper"]
    fn mixed_eigensolver_matches_explicit_full_relative_marginal_reference() {
        use manifold::gen::cartesian::CartesianMeshInfo;

        let mesh = CartesianMeshInfo::new_unit(2, 2);
        let (topology, coordinates) = mesh.compute_coord_complex();
        let geometry = coordinates.to_edge_lengths(&topology);
        let relative =
            formoniq::relative::assemble_whitney_full_relative_operators(&topology, &geometry, 1)
                .unwrap();
        let lower_layout = relative.lower().unwrap().layout();
        let state_layout = relative.state().layout();
        let galmats = hodge_laplace::MixedGalmats::compute(&topology, &geometry, 1);
        let rank = state_layout.reduced_dimension();
        let mixed = partial_mixed_generalized_hodge_eigenbasis(
            &galmats,
            lower_layout,
            state_layout,
            PartialMixedGeneralizedEigensolverOptions::new(rank),
        )
        .expect("bounded mixed helper solve should succeed");

        let full_mkm1 = CsrMatrix::from(galmats.mass_sigma());
        let full_d = CsrMatrix::from(galmats.dif_sigma());
        let full_c = CsrMatrix::from(galmats.codif_u());
        let full_l = CsrMatrix::from(galmats.codifdif_u());
        let full_mk = CsrMatrix::from(galmats.mass_u());
        let mkm1 = restrict_mixed_operator(&full_mkm1, lower_layout, lower_layout);
        let d = restrict_mixed_operator(&full_d, state_layout, lower_layout);
        let c = restrict_mixed_operator(&full_c, lower_layout, state_layout);
        let l = restrict_mixed_operator(&full_l, state_layout, state_layout);
        let mk = restrict_mixed_operator(&full_mk, state_layout, state_layout);
        let mkm1_dense = NaMatrix::from(&mkm1);
        let marginal_dense = NaMatrix::from(&l)
            + NaMatrix::from(&d)
                * mkm1_dense
                    .try_inverse()
                    .expect("small lower mass should be invertible")
                * NaMatrix::from(&c);
        let mut marginal_coo = CooMatrix::new(rank, rank);
        for row in 0..rank {
            for column in 0..rank {
                let value = marginal_dense[(row, column)];
                if value != 0.0 {
                    marginal_coo.push(row, column, value);
                }
            }
        }
        let reference = dense_reference_generalized_hodge_eigenbasis(
            &mk,
            &CsrMatrix::from(&marginal_coo),
            DenseReferenceEigensolverOptions::new(rank),
        )
        .expect("small explicit marginal reference should succeed");

        for (left, right) in mixed
            .basis()
            .eigenvalues()
            .iter()
            .zip(reference.eigenvalues())
        {
            assert_close(*left, *right, 2.0e-8);
        }
        let mass_dense = NaMatrix::from(&mk);
        let mixed_vectors = NaMatrix::from_fn(rank, rank, |row, column| {
            mixed.basis().eigenvectors()[(row, column)]
        });
        let reference_vectors = NaMatrix::from_fn(rank, rank, |row, column| {
            reference.eigenvectors()[(row, column)]
        });
        let mixed_gram = mixed_vectors.transpose() * &mass_dense * &mixed_vectors;
        assert!((&mixed_gram - NaMatrix::identity(rank, rank)).norm() <= 2.0e-8);
        let cross_gram = reference_vectors.transpose() * mass_dense * mixed_vectors;
        let singular_values = cross_gram.svd(false, false).singular_values;
        assert!(singular_values
            .iter()
            .all(|value| (*value - 1.0).abs() <= 2.0e-8));

        assert_eq!(
            mixed.diagnostics().sigma_dimension,
            lower_layout.reduced_dimension()
        );
        assert_eq!(mixed.diagnostics().u_dimension, rank);
        assert_eq!(mixed.diagnostics().retained_rank, rank);
        assert_eq!(mixed.diagnostics().which, GhiepWhich::Smallest);
        assert_eq!(mixed.diagnostics().mass_solve, GhiepReducedSolve::Direct);
        assert_eq!(mixed.diagnostics().basis_storage_entries, rank * rank);
        eprintln!(
            "retained reduced Hodge helper workspace for inspection: {:?}",
            mixed.diagnostics().workspace
        );
    }

    #[cfg(feature = "external-solvers")]
    #[test]
    #[ignore = "requires built PETSc/SLEPc ghiep helper"]
    fn partial_eigensolver_solves_and_truncates_bounded_diagonal_problem() {
        use common::linalg::nalgebra::CooMatrix;

        let diagonal = |values: &[f64]| {
            let mut matrix = CooMatrix::new(values.len(), values.len());
            for (index, value) in values.iter().copied().enumerate() {
                matrix.push(index, index, value);
            }
            CsrMatrix::from(&matrix)
        };
        let mass_diagonal = [2.0, 4.0, 8.0];
        let mass = diagonal(&mass_diagonal);
        let laplacian = diagonal(&[8.0, 4.0, 16.0]);
        let result = partial_generalized_hodge_eigenbasis(
            &mass,
            &laplacian,
            PartialGeneralizedEigensolverOptions::new(2),
        )
        .expect("bounded helper solve should succeed");

        assert_close(result.basis().eigenvalues()[0], 1.0, 1.0e-8);
        assert_close(result.basis().eigenvalues()[1], 2.0, 1.0e-8);
        assert_eq!(result.diagnostics().dimension, 3);
        assert_eq!(result.diagnostics().requested_rank, 2);
        assert!(result.diagnostics().converged_rank >= 2);
        assert_eq!(result.diagnostics().retained_rank, 2);
        assert_eq!(result.diagnostics().mass_nonzeros, 3);
        assert_eq!(result.diagnostics().weak_hodge_laplacian_nonzeros, 3);
        assert_eq!(result.diagnostics().input_nonzeros, 6);
        assert_eq!(result.diagnostics().basis_storage_entries, 6);
        assert_eq!(result.diagnostics().basis_storage_bytes, 48);
        for left in 0..result.basis().len() {
            for right in 0..result.basis().len() {
                let mass_inner_product = mass_diagonal
                    .iter()
                    .enumerate()
                    .map(|(row, mass_value)| {
                        mass_value
                            * result.basis().eigenvectors()[(row, left)]
                            * result.basis().eigenvectors()[(row, right)]
                    })
                    .sum::<f64>();
                assert_close(
                    mass_inner_product,
                    if left == right { 1.0 } else { 0.0 },
                    1.0e-8,
                );
            }
        }
        eprintln!(
            "retained PETSc helper workspace for inspection: {:?}",
            result.diagnostics().workspace
        );
    }

    #[test]
    fn dense_reference_eigensolver_enforces_limits_symmetry_and_spd_mass() {
        use common::linalg::nalgebra::CooMatrix;

        let mut identity = CooMatrix::new(2, 2);
        identity.push(0, 0, 1.0);
        identity.push(1, 1, 1.0);
        let identity = CsrMatrix::from(&identity);

        let error = dense_reference_generalized_hodge_eigenbasis(
            &identity,
            &identity,
            DenseReferenceEigensolverOptions::new(1),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::DenseReferenceDimensionLimitExceeded {
                dimension: 2,
                maximum: 1
            }
        ));

        let mut nonsymmetric = CooMatrix::new(2, 2);
        nonsymmetric.push(0, 0, 1.0);
        nonsymmetric.push(0, 1, 0.25);
        nonsymmetric.push(1, 1, 2.0);
        let error = dense_reference_generalized_hodge_eigenbasis(
            &identity,
            &CsrMatrix::from(&nonsymmetric),
            DenseReferenceEigensolverOptions::new(2),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::DenseReferenceOperatorNotSymmetric {
                operator: "Hodge Laplacian",
                ..
            }
        ));

        let mut indefinite = CooMatrix::new(2, 2);
        indefinite.push(0, 0, 1.0);
        indefinite.push(1, 1, -1.0);
        let error = dense_reference_generalized_hodge_eigenbasis(
            &CsrMatrix::from(&indefinite),
            &identity,
            DenseReferenceEigensolverOptions::new(2),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GpError::DenseReferenceMassNotPositiveDefinite
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

    fn check_prepared_posterior_against_dense_reference(
        observation_indices: &[usize],
        expected_strategy: LowRankConditioningStrategy,
    ) {
        use common::linalg::nalgebra::CooMatrix;

        let basis = EigenBasis::new(
            vec![0.4, 1.7, 4.2],
            Mat::from_fn(4, 3, |row, col| {
                const VALUES: [[f64; 3]; 4] = [
                    [1.0, 0.0, 0.2],
                    [0.1, 0.8, -0.3],
                    [0.0, 0.4, 1.1],
                    [0.7, -0.2, 0.5],
                ];
                VALUES[row][col]
            }),
        )
        .unwrap();
        let gp = SpectralMaternGp::from_eigenbasis(
            basis,
            SpectralMaternConfig {
                kappa: 1.1,
                alpha: 2.0,
                tau: 0.8,
                k: 3,
            },
        )
        .unwrap();
        let mut observation = CooMatrix::new(observation_indices.len(), 4);
        for (row, &col) in observation_indices.iter().enumerate() {
            observation.push(row, col, 1.0);
        }
        let observation = CsrMatrix::from(&observation);
        let noise_variance = 0.17;
        let prepared = gp
            .prepare_linear_observations(&observation, noise_variance)
            .unwrap();
        assert_eq!(prepared.conditioning_strategy(), expected_strategy);

        let values = (0..observation_indices.len())
            .map(|index| 0.25 - 0.3 * index as f64)
            .collect::<Vec<_>>();
        let covariance = gp.covariance_matrix();
        let dense = condition_full_covariance_with_covariance(
            &covariance,
            observation_indices,
            &values,
            noise_variance,
        )
        .unwrap();
        let prepared_mean = prepared.state_mean(&values).unwrap();
        for (actual, expected) in prepared_mean.iter().zip(&dense.mean) {
            assert_close(*actual, *expected, 3.0e-13);
        }

        let state_input = [0.3, -0.6, 0.2, 1.1];
        let action = prepared.state_covariance_action(&state_input).unwrap();
        let expected_action = (0..4)
            .map(|row| {
                state_input
                    .iter()
                    .enumerate()
                    .map(|(col, value)| dense.covariance[(idx(row), idx(col))] * value)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        for (actual, expected) in action.iter().zip(expected_action) {
            assert_close(*actual, expected, 4.0e-13);
        }

        let mut output = CooMatrix::new(2, 4);
        output.push(0, 0, 1.0);
        output.push(0, 2, -0.4);
        output.push(1, 1, 0.3);
        output.push(1, 3, 1.2);
        let output = CsrMatrix::from(&output);
        let prior_output_covariance = gp.prior_output_covariance(&output).unwrap();
        let expected_prior_output_covariance = sparse_covariance_pushforward(&output, &covariance);
        assert_matrix_close(
            &prior_output_covariance,
            &expected_prior_output_covariance,
            5.0e-13,
        );
        let prior_output_input = [0.4, -0.9];
        let prior_output_action = gp
            .prior_output_covariance_action(&output, &prior_output_input)
            .unwrap();
        for row in 0..2 {
            let expected = (0..2)
                .map(|col| {
                    expected_prior_output_covariance[(idx(row), idx(col))] * prior_output_input[col]
                })
                .sum::<f64>();
            assert_close(prior_output_action[row], expected, 5.0e-13);
        }
        let output_moments = prepared
            .output_moments_with_covariance(&output, &values)
            .unwrap();
        let expected_output_mean = sparse_vector_product(&output, &dense.mean);
        for (actual, expected) in output_moments.mean.iter().zip(expected_output_mean) {
            assert_close(*actual, expected, 4.0e-13);
        }
        let expected_output_covariance = sparse_covariance_pushforward(&output, &dense.covariance);
        assert_matrix_close(
            &output_moments.covariance,
            &expected_output_covariance,
            5.0e-13,
        );
        let diagonal_only = prepared.output_moments(&output, &values).unwrap();
        for index in 0..2 {
            assert_close(
                diagonal_only.variance[index],
                expected_output_covariance[(idx(index), idx(index))],
                5.0e-13,
            );
        }
        let output_input = [0.4, -0.9];
        let output_action = prepared
            .output_covariance_action(&output, &output_input)
            .unwrap();
        for row in 0..2 {
            let expected = (0..2)
                .map(|col| expected_output_covariance[(idx(row), idx(col))] * output_input[col])
                .sum::<f64>();
            assert_close(output_action[row], expected, 5.0e-13);
        }

        let replacement_values = values
            .iter()
            .enumerate()
            .map(|(index, value)| value + 0.1 * (index + 1) as f64)
            .collect::<Vec<_>>();
        let replacement_dense = condition_full_covariance_with_covariance(
            &covariance,
            observation_indices,
            &replacement_values,
            noise_variance,
        )
        .unwrap();
        let replacement_mean = prepared.state_mean(&replacement_values).unwrap();
        for (actual, expected) in replacement_mean.iter().zip(replacement_dense.mean) {
            assert_close(*actual, expected, 4.0e-13);
        }

        let diagnostics = prepared.diagnostics();
        assert_eq!(diagnostics.ambient_dimension, 4);
        assert_eq!(diagnostics.latent_dimension, 3);
        assert_eq!(diagnostics.observation_count, observation_indices.len());
        assert_eq!(
            diagnostics.dense_factor_dimension,
            observation_indices.len().min(3)
        );
        assert_eq!(
            diagnostics.persistent_f64_entries,
            diagnostics.state_feature_entries
                + diagnostics.observation_feature_entries
                + diagnostics.dense_factor_allocated_entries
        );
        let zero_standard_normal = vec![0.0; diagnostics.sampling_standard_normal_dimension];
        let zero_sample = prepared
            .state_sample_from_standard_normal(&values, &zero_standard_normal)
            .unwrap();
        for (sample, mean) in zero_sample.iter().zip(prepared_mean) {
            assert_close(*sample, mean, 4.0e-13);
        }
        let zero_output_sample = prepared
            .output_sample_from_standard_normal(&output, &values, &zero_standard_normal)
            .unwrap();
        for (sample, mean) in zero_output_sample.iter().zip(output_moments.mean) {
            assert_close(*sample, mean, 4.0e-13);
        }

        // The sample is an affine map of the documented independent standard
        // normals.  Summing outer products of its responses to the canonical
        // normal basis therefore reconstructs its covariance exactly.
        let latent_mean = prepared.latent_mean(&values).unwrap();
        let mut sample_covariance = Mat::zeros(3, 3);
        for normal_index in 0..diagnostics.sampling_standard_normal_dimension {
            let mut standard_normal = vec![0.0; diagnostics.sampling_standard_normal_dimension];
            standard_normal[normal_index] = 1.0;
            let sample = prepared
                .latent_sample_from_standard_normal(&values, &standard_normal)
                .unwrap();
            for row in 0..3 {
                for col in 0..3 {
                    sample_covariance[(idx(row), idx(col))] +=
                        (sample[row] - latent_mean[row]) * (sample[col] - latent_mean[col]);
                }
            }
        }
        for col in 0..3 {
            let mut unit = vec![0.0; 3];
            unit[col] = 1.0;
            let expected_column = prepared.latent_covariance_action(&unit).unwrap();
            for row in 0..3 {
                assert_close(
                    sample_covariance[(idx(row), idx(col))],
                    expected_column[row],
                    8.0e-13,
                );
            }
        }
    }

    fn sparse_vector_product(operator: &CsrMatrix, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; operator.nrows()];
        for (row, col, value) in operator.triplet_iter() {
            output[row] += value * input[col];
        }
        output
    }

    fn sparse_covariance_pushforward(operator: &CsrMatrix, covariance: &Mat<f64>) -> Mat<f64> {
        Mat::from_fn(operator.nrows(), operator.nrows(), |left, right| {
            let mut value = 0.0;
            for (left_row, left_col, left_value) in operator.triplet_iter() {
                if left_row != left {
                    continue;
                }
                for (right_row, right_col, right_value) in operator.triplet_iter() {
                    if right_row == right {
                        value +=
                            left_value * covariance[(idx(left_col), idx(right_col))] * right_value;
                    }
                }
            }
            value
        })
    }

    #[test]
    fn prepared_posterior_matches_dense_reference_in_observation_space() {
        check_prepared_posterior_against_dense_reference(
            &[0, 2],
            LowRankConditioningStrategy::ObservationSpace,
        );
    }

    #[test]
    fn prepared_posterior_matches_dense_reference_in_latent_space() {
        check_prepared_posterior_against_dense_reference(
            &[0, 1, 2, 3],
            LowRankConditioningStrategy::LatentSpace,
        );
    }

    #[test]
    fn prepared_posterior_rejects_nonpositive_noise_and_wrong_dimensions() {
        use common::linalg::nalgebra::CooMatrix;

        let gp = SpectralMaternGp::from_eigenbasis(
            EigenBasis::new(vec![1.0], Mat::ones(2, 1)).unwrap(),
            SpectralMaternConfig {
                k: 1,
                ..SpectralMaternConfig::default()
            },
        )
        .unwrap();
        let mut observation = CooMatrix::new(1, 2);
        observation.push(0, 0, 1.0);
        let observation = CsrMatrix::from(&observation);
        let error = gp
            .prepare_linear_observations(&observation, 0.0)
            .unwrap_err();
        assert!(matches!(error, GpError::NonPositivePreparedNoiseVariance));

        let prepared = gp.prepare_linear_observations(&observation, 0.1).unwrap();
        assert!(matches!(
            prepared.state_mean(&[]).unwrap_err(),
            GpError::ObservationValueDimensionMismatch { rows: 1, values: 0 }
        ));
        assert!(matches!(
            prepared.state_mean(&[f64::NAN]).unwrap_err(),
            GpError::NonFiniteObservationValue { index: 0, value } if value.is_nan()
        ));
        assert!(matches!(
            prepared.latent_covariance_action(&[0.0, 1.0]).unwrap_err(),
            GpError::LatentVectorDimensionMismatch {
                expected: 1,
                got: 2
            }
        ));
    }
}
