use super::PosteriorReport;
use crate::infer::VarianceEstimator;
use crate::Result;
use std::io::Write;

/// Controls deterministic, bounded console rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsoleReportOptions {
    pub precision: usize,
    pub max_rows: usize,
    pub include_covariance: bool,
    pub include_correlation: bool,
}

impl Default for ConsoleReportOptions {
    fn default() -> Self {
        Self {
            precision: 6,
            max_rows: 24,
            include_covariance: false,
            include_correlation: false,
        }
    }
}

/// Print aggregate field summaries and bounded QoI/prediction rows.
pub fn write_console_report<W: Write>(
    writer: &mut W,
    report: &PosteriorReport,
    options: &ConsoleReportOptions,
) -> Result<()> {
    let precision = options.precision;
    writeln!(writer, "Posterior report")?;
    for metric in &report.metrics {
        writeln!(
            writer,
            "metric {} ({}): {:.*} {}",
            metric.id, metric.label, precision, metric.value, metric.unit
        )?;
    }
    for field in &report.fields {
        let (mean_min, mean_max) = range(&field.mean);
        let (sd_min, sd_max) = range(&field.standard_deviations);
        writeln!(
            writer,
            "field {} ({}): n={}, mean=[{:.*}, {:.*}], std=[{:.*}, {:.*}] {}",
            field.id,
            field.label,
            field.mean.len(),
            precision,
            mean_min,
            precision,
            mean_max,
            precision,
            sd_min,
            precision,
            sd_max,
            field.unit
        )?;
        writeln!(
            writer,
            "  variance: estimator={}, samples={}, batches={:?}, negative_raw={}, minimum_raw={:.*}",
            estimator_name(field.variance.estimator),
            field.variance.sample_count,
            field.variance.batch_sizes,
            field.variance.negative_count,
            precision,
            field.variance.minimum_value
        )?;
    }
    for qoi in &report.qois {
        writeln!(writer, "qoi {} ({}):", qoi.id, qoi.label)?;
        for index in 0..qoi.mean.len().min(options.max_rows) {
            writeln!(
                writer,
                "  {}: mean={:.*}, std={:.*} {}",
                qoi.labels[index],
                precision,
                qoi.mean[index],
                precision,
                qoi.standard_deviations[index],
                qoi.units[index]
            )?;
        }
        if qoi.mean.len() > options.max_rows {
            writeln!(
                writer,
                "  ... {} rows omitted",
                qoi.mean.len() - options.max_rows
            )?;
        }
        if options.include_covariance {
            write_matrix(writer, "covariance", &qoi.covariance, precision)?;
        }
        if options.include_correlation {
            write_matrix(writer, "correlation", &qoi.correlation, precision)?;
        }
    }
    for prediction in &report.predictions {
        writeln!(
            writer,
            "prediction {} ({}): rmse={:.*}, nlpd={:.*}, coverage={:.*}",
            prediction.id,
            prediction.label,
            precision,
            prediction.diagnostics.rmse,
            precision,
            prediction.diagnostics.mean_negative_log_predictive_density,
            precision,
            prediction.diagnostics.empirical_coverage
        )?;
        for index in 0..prediction.predictive_means.len().min(options.max_rows) {
            writeln!(
                writer,
                "  {}: observed={:.*}, mean={:.*}, pred_std={:.*}, z={:.*} {}",
                prediction.labels[index],
                precision,
                prediction.observations[index],
                precision,
                prediction.predictive_means[index],
                precision,
                prediction.diagnostics.predictive_standard_deviations[index],
                precision,
                prediction.diagnostics.standardized_residuals[index],
                prediction.units[index]
            )?;
        }
        if prediction.predictive_means.len() > options.max_rows {
            writeln!(
                writer,
                "  ... {} rows omitted",
                prediction.predictive_means.len() - options.max_rows
            )?;
        }
    }
    if let Some(factorization) = report.factorization {
        writeln!(
            writer,
            "factorization: dimension={}, precision_nnz={}, factor_nnz={}, fill_ratio={:.*}",
            factorization.dimension,
            factorization.precision_nonzeros,
            factorization.factor_nonzeros,
            precision,
            factorization.fill_ratio
        )?;
    }
    Ok(())
}

fn range(values: &[f64]) -> (f64, f64) {
    values.iter().copied().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    )
}

fn write_matrix<W: Write>(
    writer: &mut W,
    name: &str,
    matrix: &[Vec<f64>],
    precision: usize,
) -> Result<()> {
    writeln!(writer, "  {name}:")?;
    for row in matrix {
        let values = row
            .iter()
            .map(|value| format!("{value:.precision$}"))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(writer, "    {values}")?;
    }
    Ok(())
}

fn estimator_name(estimator: VarianceEstimator) -> &'static str {
    match estimator {
        VarianceEstimator::Exact => "exact",
        VarianceEstimator::MonteCarlo => "monte_carlo",
        VarianceEstimator::Hutchinson => "hutchinson",
    }
}
