use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use gmrf_core::types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector};

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SummaryStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
}

impl SummaryStats {
    pub fn ratio(&self) -> f64 {
        if self.min.abs() <= EPS {
            f64::INFINITY
        } else {
            self.max / self.min
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LinearEqualityConstraints<'a> {
    pub matrix: &'a GmrfDenseMatrix,
    pub rhs: &'a GmrfVector,
}

pub fn summarize_vector(vec: &FeecVector) -> Option<SummaryStats> {
    if vec.is_empty() {
        return None;
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &value in vec.iter() {
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    let mean = sum / vec.len() as f64;
    let variance = vec
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / vec.len() as f64;
    Some(SummaryStats {
        min,
        max,
        mean,
        std: variance.sqrt(),
    })
}

pub fn pearson_correlation(lhs: &FeecVector, rhs: &FeecVector) -> Option<f64> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return None;
    }
    let lhs_mean = lhs.iter().sum::<f64>() / lhs.len() as f64;
    let rhs_mean = rhs.iter().sum::<f64>() / rhs.len() as f64;
    let mut cov = 0.0;
    let mut lhs_var = 0.0;
    let mut rhs_var = 0.0;
    for i in 0..lhs.len() {
        let lx = lhs[i] - lhs_mean;
        let ry = rhs[i] - rhs_mean;
        cov += lx * ry;
        lhs_var += lx * lx;
        rhs_var += ry * ry;
    }
    if lhs_var <= EPS || rhs_var <= EPS {
        None
    } else {
        Some(cov / (lhs_var.sqrt() * rhs_var.sqrt()))
    }
}

pub fn build_harmonic_orthogonality_constraints(
    harmonic_basis: &FeecMatrix,
    mass: &FeecCsr,
) -> Result<GmrfDenseMatrix, String> {
    if harmonic_basis.nrows() != mass.nrows() {
        return Err(format!(
            "harmonic basis row count {} does not match mass rows {}",
            harmonic_basis.nrows(),
            mass.nrows()
        ));
    }
    if mass.ncols() != mass.nrows() {
        return Err("mass matrix must be square".to_string());
    }
    let mass_harmonics = mass * harmonic_basis;
    Ok(GmrfDenseMatrix::from_fn(
        harmonic_basis.ncols(),
        harmonic_basis.nrows(),
        |row, col| mass_harmonics[(col, row)],
    ))
}

pub fn matrix_diag(mat: &FeecCsr) -> FeecVector {
    let mut diag = FeecVector::zeros(mat.nrows());
    for (row, col, value) in mat.triplet_iter() {
        if row == col {
            diag[row] += *value;
        }
    }
    diag
}

pub fn lumped_diag(mat: &FeecCsr) -> FeecVector {
    let mut diag = FeecVector::zeros(mat.nrows());
    for (row, _col, value) in mat.triplet_iter() {
        diag[row] += *value;
    }
    diag
}
