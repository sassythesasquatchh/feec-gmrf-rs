use crate::sparse::scale_matrix;
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
pub use feg_core::MaternAlpha;

/// Convert Whittle `alpha/tau/kappa` parameters to smoothness, marginal
/// variance, and the practical-range convention `sqrt(8 nu) / kappa`.
pub fn convert_whittle_params_to_matern(
    alpha: f64,
    tau: f64,
    kappa: f64,
    dimension: usize,
) -> (f64, f64, f64) {
    let nu = alpha - dimension as f64 / 2.0;
    let variance = libm::tgamma(nu)
        / (tau
            * tau
            * libm::tgamma(alpha)
            * (4.0 * std::f64::consts::PI).powf(dimension as f64 / 2.0)
            * kappa.powf(2.0 * nu));
    let effective_range = (8.0 * nu).sqrt() / kappa;
    (nu, variance, effective_range)
}

pub mod generic;
pub mod one_form;
pub mod three_form;
pub mod two_form;
pub mod zero_form;

pub fn build_lindgren_precision_from_system(
    system: &FeecCsr,
    mass_inverse: &FeecCsr,
    alpha: MaternAlpha,
    tau: f64,
) -> FeecCsr {
    let timing_enabled = std::env::var_os("FEG_MATERN_TIMINGS").is_some();
    let mut precision = match alpha {
        MaternAlpha::One => system.clone(),
        MaternAlpha::Two => {
            let started = std::time::Instant::now();
            let middle = mass_inverse * system;
            if timing_enabled {
                eprintln!(
                    "[matern_precision] alpha=2 mass_inverse*system nnz={} elapsed={:.3}s",
                    middle.nnz(),
                    started.elapsed().as_secs_f64()
                );
            }
            let started = std::time::Instant::now();
            let precision = system * &middle;
            if timing_enabled {
                eprintln!(
                    "[matern_precision] alpha=2 system*middle nnz={} elapsed={:.3}s",
                    precision.nnz(),
                    started.elapsed().as_secs_f64()
                );
            }
            precision
        }
        MaternAlpha::Three => {
            let started = std::time::Instant::now();
            let alpha_two_middle = mass_inverse * system;
            if timing_enabled {
                eprintln!(
                    "[matern_precision] alpha=3 mass_inverse*system nnz={} elapsed={:.3}s",
                    alpha_two_middle.nnz(),
                    started.elapsed().as_secs_f64()
                );
            }
            let started = std::time::Instant::now();
            let alpha_two = system * &alpha_two_middle;
            if timing_enabled {
                eprintln!(
                    "[matern_precision] alpha=3 system*alpha_two_middle nnz={} elapsed={:.3}s",
                    alpha_two.nnz(),
                    started.elapsed().as_secs_f64()
                );
            }
            let started = std::time::Instant::now();
            let alpha_three_middle = mass_inverse * &alpha_two;
            if timing_enabled {
                eprintln!(
                    "[matern_precision] alpha=3 mass_inverse*alpha_two nnz={} elapsed={:.3}s",
                    alpha_three_middle.nnz(),
                    started.elapsed().as_secs_f64()
                );
            }
            let started = std::time::Instant::now();
            let precision = system * &alpha_three_middle;
            if timing_enabled {
                eprintln!(
                    "[matern_precision] alpha=3 system*alpha_three_middle nnz={} elapsed={:.3}s",
                    precision.nnz(),
                    started.elapsed().as_secs_f64()
                );
            }
            precision
        }
    };
    if (tau - 1.0).abs() > f64::EPSILON {
        precision = scale_matrix(&precision, tau * tau);
    }
    precision
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matern_alpha_accepts_lindgren_alpha_three() {
        assert_eq!(MaternAlpha::try_from(3).unwrap(), MaternAlpha::Three);
        assert_eq!("3".parse::<MaternAlpha>().unwrap(), MaternAlpha::Three);
        assert_eq!(MaternAlpha::Three.as_u32(), 3);
    }
}
