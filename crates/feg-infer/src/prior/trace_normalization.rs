use crate::sparse::scale_matrix;
use common::linalg::nalgebra::CsrMatrix as FeecCsr;

/// Scaling needed to match a target weighted covariance trace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceNormalization {
    pub raw_trace: f64,
    pub target_trace: f64,
    pub precision_scale: f64,
    pub tau_multiplier: f64,
}

impl TraceNormalization {
    pub fn normalized_trace(self) -> f64 {
        self.raw_trace / self.precision_scale
    }

    pub fn normalized_mean_trace_variance(self, domain_measure: f64) -> Result<f64, String> {
        validate_positive_finite(domain_measure, "domain_measure")?;
        Ok(self.normalized_trace() / domain_measure)
    }

    pub fn scale_precision(self, precision: &FeecCsr) -> FeecCsr {
        scale_matrix(precision, self.precision_scale)
    }
}

pub fn trace_normalization_from_target_trace(
    raw_trace: f64,
    target_trace: f64,
) -> Result<TraceNormalization, String> {
    validate_positive_finite(raw_trace, "raw_trace")?;
    validate_positive_finite(target_trace, "target_trace")?;
    let precision_scale = raw_trace / target_trace;
    validate_positive_finite(precision_scale, "precision_scale")?;
    Ok(TraceNormalization {
        raw_trace,
        target_trace,
        precision_scale,
        tau_multiplier: precision_scale.sqrt(),
    })
}

pub fn trace_normalization_from_mean_variance(
    raw_trace: f64,
    target_mean_trace_variance: f64,
    domain_measure: f64,
) -> Result<TraceNormalization, String> {
    validate_positive_finite(target_mean_trace_variance, "target_mean_trace_variance")?;
    validate_positive_finite(domain_measure, "domain_measure")?;
    trace_normalization_from_target_trace(raw_trace, target_mean_trace_variance * domain_measure)
}

pub fn scale_precision_to_trace(precision: &FeecCsr, normalization: TraceNormalization) -> FeecCsr {
    normalization.scale_precision(precision)
}

fn validate_positive_finite(value: f64, name: &str) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be finite and positive, got {value}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::CooMatrix as FeecCoo;

    #[test]
    fn trace_normalization_matches_target_mean_variance() {
        let normalization = trace_normalization_from_mean_variance(8.0, 1.0, 2.0)
            .expect("normalization should build");
        assert_eq!(normalization.raw_trace, 8.0);
        assert_eq!(normalization.target_trace, 2.0);
        assert_eq!(normalization.precision_scale, 4.0);
        assert_eq!(normalization.tau_multiplier, 2.0);
        assert!(
            (normalization
                .normalized_mean_trace_variance(2.0)
                .expect("domain measure is valid")
                - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn trace_normalization_scales_precision() {
        let normalization =
            trace_normalization_from_target_trace(9.0, 3.0).expect("normalization should build");
        let mut coo = FeecCoo::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 5.0);
        let scaled = normalization.scale_precision(&FeecCsr::from(&coo));
        let entries = scaled.triplet_iter().collect::<Vec<_>>();
        assert!(entries
            .iter()
            .any(|(row, col, value)| *row == 0 && *col == 0 && (**value - 6.0).abs() < 1e-12));
        assert!(entries
            .iter()
            .any(|(row, col, value)| *row == 1 && *col == 1 && (**value - 15.0).abs() < 1e-12));
    }

    #[test]
    fn trace_normalization_rejects_invalid_inputs() {
        assert!(trace_normalization_from_target_trace(0.0, 1.0).is_err());
        assert!(trace_normalization_from_target_trace(1.0, -1.0).is_err());
        assert!(trace_normalization_from_mean_variance(1.0, 0.0, 1.0).is_err());
        assert!(trace_normalization_from_mean_variance(1.0, 1.0, f64::NAN).is_err());
    }
}
