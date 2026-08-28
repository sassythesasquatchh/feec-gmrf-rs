use super::{PosteriorReport, PredictionReport};
use crate::infer::VarianceEstimator;
use crate::{FeecGmrfError, Result};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// One validated cell in a report table.
#[derive(Debug, Clone, PartialEq)]
pub enum ReportCell {
    Text(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Missing,
}

/// A named rectangular table with stable column ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportTable {
    id: String,
    columns: Vec<String>,
    rows: Vec<Vec<ReportCell>>,
}

impl ReportTable {
    pub fn new(id: impl Into<String>, columns: Vec<String>) -> Result<Self> {
        let id = id.into();
        super::validate_artifact_id(&id)?;
        if columns.is_empty() || columns.iter().any(|column| column.trim().is_empty()) {
            return Err(FeecGmrfError::InvalidParameter(
                "report table columns must be non-empty".to_string(),
            ));
        }
        let unique = columns.iter().collect::<BTreeSet<_>>();
        if unique.len() != columns.len() {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "report table `{id}` has duplicate columns"
            )));
        }
        Ok(Self {
            id,
            columns,
            rows: Vec::new(),
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    pub fn rows(&self) -> &[Vec<ReportCell>] {
        &self.rows
    }

    pub fn push_row(&mut self, row: Vec<ReportCell>) -> Result<()> {
        if row.len() != self.columns.len() {
            return Err(FeecGmrfError::Dimension(format!(
                "table `{}` row width {} does not match column count {}",
                self.id,
                row.len(),
                self.columns.len()
            )));
        }
        validate_cells(&row)?;
        self.rows.push(row);
        Ok(())
    }

    pub fn with_row(mut self, row: Vec<ReportCell>) -> Result<Self> {
        self.push_row(row)?;
        Ok(self)
    }

    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(path, self)
    }
}

/// Write one report table with RFC-compatible CSV quoting.
pub fn write_csv(path: impl AsRef<Path>, table: &ReportTable) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(csv_error)?;
    writer.write_record(&table.columns).map_err(csv_error)?;
    for row in &table.rows {
        validate_cells(row)?;
        let record = row.iter().map(cell_text).collect::<Vec<_>>();
        writer.write_record(record).map_err(csv_error)?;
    }
    writer.flush()?;
    Ok(())
}

/// Write each table to `<directory>/<table-id>.csv`.
pub fn write_csv_directory(directory: impl AsRef<Path>, tables: &[ReportTable]) -> Result<()> {
    let directory = directory.as_ref();
    fs::create_dir_all(directory)?;
    let mut ids = BTreeSet::new();
    for table in tables {
        if !ids.insert(table.id()) {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "duplicate table ID `{}`",
                table.id()
            )));
        }
        write_csv(directory.join(format!("{}.csv", table.id())), table)?;
    }
    Ok(())
}

impl PosteriorReport {
    /// Materialize the standard metric, field, QoI, matrix, prediction, and
    /// factorization tables.
    pub fn tables(&self) -> Result<Vec<ReportTable>> {
        let mut tables = vec![metric_table(self)?];
        for field in &self.fields {
            let mut table = ReportTable::new(
                format!("{}_field", field.id),
                strings(&[
                    "index",
                    "mean",
                    "variance",
                    "standard_deviation",
                    "truth",
                    "error",
                    "z_score",
                    "reference",
                    "reference_error",
                    "reference_z_score",
                    "baseline_variance",
                    "variance_reduction",
                    "batch_standard_error",
                    "relative_standard_error",
                ]),
            )?;
            for index in 0..field.mean.len() {
                table.push_row(vec![
                    integer(index)?,
                    ReportCell::Float(field.mean[index]),
                    ReportCell::Float(field.variance.values[index]),
                    ReportCell::Float(field.standard_deviations[index]),
                    optional_value(field.truth.as_deref(), index),
                    optional_value(field.errors.as_deref(), index),
                    optional_nested(field.z_scores.as_deref(), index),
                    optional_value(field.reference.as_deref(), index),
                    optional_value(field.reference_errors.as_deref(), index),
                    optional_nested(field.reference_z_scores.as_deref(), index),
                    optional_value(field.baseline_variances.as_deref(), index),
                    optional_nested(field.variance_reductions.as_deref(), index),
                    optional_value(field.variance.batch_standard_error.as_deref(), index),
                    optional_value(field.variance.relative_standard_error.as_deref(), index),
                ])?;
            }
            tables.push(table);

            let mut estimator = ReportTable::new(
                format!("{}_estimator", field.id),
                strings(&[
                    "estimator",
                    "sample_count",
                    "batch_sizes",
                    "negative_count",
                    "minimum_raw_variance",
                ]),
            )?;
            estimator.push_row(vec![
                ReportCell::Text(estimator_name(field.variance.estimator).to_string()),
                integer(field.variance.sample_count)?,
                ReportCell::Text(
                    field
                        .variance
                        .batch_sizes
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(";"),
                ),
                integer(field.variance.negative_count)?,
                ReportCell::Float(field.variance.minimum_value),
            ])?;
            tables.push(estimator);
        }

        for qoi in &self.qois {
            let mut table = ReportTable::new(
                format!("{}_qoi", qoi.id),
                strings(&[
                    "index",
                    "label",
                    "unit",
                    "mean",
                    "standard_deviation",
                    "truth",
                    "z_score",
                    "reference",
                    "reference_z_score",
                    "baseline_variance",
                    "variance_reduction",
                ]),
            )?;
            for index in 0..qoi.mean.len() {
                table.push_row(vec![
                    integer(index)?,
                    ReportCell::Text(qoi.labels[index].clone()),
                    ReportCell::Text(qoi.units[index].clone()),
                    ReportCell::Float(qoi.mean[index]),
                    ReportCell::Float(qoi.standard_deviations[index]),
                    optional_value(qoi.truth.as_deref(), index),
                    optional_nested(qoi.z_scores.as_deref(), index),
                    optional_value(qoi.reference.as_deref(), index),
                    optional_nested(qoi.reference_z_scores.as_deref(), index),
                    optional_value(qoi.baseline_variances.as_deref(), index),
                    optional_nested(qoi.variance_reductions.as_deref(), index),
                ])?;
            }
            tables.push(table);
            tables.push(matrix_table(
                &format!("{}_covariance", qoi.id),
                &qoi.labels,
                &qoi.covariance,
            )?);
            tables.push(matrix_table(
                &format!("{}_correlation", qoi.id),
                &qoi.labels,
                &qoi.correlation,
            )?);
        }

        for prediction in &self.predictions {
            tables.push(prediction_table(prediction)?);
            tables.push(prediction_summary_table(prediction)?);
        }

        if let Some(factorization) = self.factorization {
            let mut table = ReportTable::new(
                "factorization",
                strings(&[
                    "dimension",
                    "precision_nonzeros",
                    "factor_nonzeros",
                    "fill_ratio",
                ]),
            )?;
            table.push_row(vec![
                integer(factorization.dimension)?,
                integer(factorization.precision_nonzeros)?,
                integer(factorization.factor_nonzeros)?,
                ReportCell::Float(factorization.fill_ratio),
            ])?;
            tables.push(table);
        }
        Ok(tables)
    }
}

fn metric_table(report: &PosteriorReport) -> Result<ReportTable> {
    let mut table = ReportTable::new("metrics", strings(&["id", "label", "unit", "value"]))?;
    for metric in &report.metrics {
        table.push_row(vec![
            ReportCell::Text(metric.id.clone()),
            ReportCell::Text(metric.label.clone()),
            ReportCell::Text(metric.unit.clone()),
            ReportCell::Float(metric.value),
        ])?;
    }
    Ok(table)
}

fn matrix_table(id: &str, labels: &[String], matrix: &[Vec<f64>]) -> Result<ReportTable> {
    let mut table = ReportTable::new(
        id,
        strings(&[
            "row_index",
            "row_label",
            "column_index",
            "column_label",
            "value",
        ]),
    )?;
    for (row_index, row) in matrix.iter().enumerate() {
        for (column_index, value) in row.iter().enumerate() {
            table.push_row(vec![
                integer(row_index)?,
                ReportCell::Text(labels[row_index].clone()),
                integer(column_index)?,
                ReportCell::Text(labels[column_index].clone()),
                ReportCell::Float(*value),
            ])?;
        }
    }
    Ok(table)
}

fn prediction_table(prediction: &PredictionReport) -> Result<ReportTable> {
    let mut table = ReportTable::new(
        format!("{}_prediction", prediction.id),
        strings(&[
            "index",
            "label",
            "unit",
            "observation",
            "predictive_mean",
            "latent_variance",
            "observation_variance",
            "predictive_standard_deviation",
            "residual",
            "standardized_residual",
            "covered",
        ]),
    )?;
    let width = prediction.diagnostics.interval_standard_deviations;
    for index in 0..prediction.predictive_means.len() {
        let standardized = prediction.diagnostics.standardized_residuals[index];
        table.push_row(vec![
            integer(index)?,
            ReportCell::Text(prediction.labels[index].clone()),
            ReportCell::Text(prediction.units[index].clone()),
            ReportCell::Float(prediction.observations[index]),
            ReportCell::Float(prediction.predictive_means[index]),
            ReportCell::Float(prediction.latent_variance.values[index]),
            ReportCell::Float(prediction.observation_variances[index]),
            ReportCell::Float(prediction.diagnostics.predictive_standard_deviations[index]),
            ReportCell::Float(prediction.diagnostics.residuals[index]),
            ReportCell::Float(standardized),
            ReportCell::Boolean(standardized.abs() <= width),
        ])?;
    }
    Ok(table)
}

fn prediction_summary_table(prediction: &PredictionReport) -> Result<ReportTable> {
    let mut table = ReportTable::new(
        format!("{}_prediction_summary", prediction.id),
        strings(&["metric", "value"]),
    )?;
    for (name, value) in [
        ("rmse", prediction.diagnostics.rmse),
        (
            "mean_negative_log_predictive_density",
            prediction.diagnostics.mean_negative_log_predictive_density,
        ),
        (
            "interval_standard_deviations",
            prediction.diagnostics.interval_standard_deviations,
        ),
        (
            "empirical_coverage",
            prediction.diagnostics.empirical_coverage,
        ),
    ] {
        table.push_row(vec![
            ReportCell::Text(name.to_string()),
            ReportCell::Float(value),
        ])?;
    }
    Ok(table)
}

fn validate_cells(cells: &[ReportCell]) -> Result<()> {
    if cells
        .iter()
        .any(|cell| matches!(cell, ReportCell::Float(value) if !value.is_finite()))
    {
        return Err(FeecGmrfError::InvalidParameter(
            "report table Float cells must be finite; use Missing or explicit Text for absent or non-finite data"
                .to_string(),
        ));
    }
    Ok(())
}

fn cell_text(cell: &ReportCell) -> String {
    match cell {
        ReportCell::Text(value) => value.clone(),
        ReportCell::Integer(value) => value.to_string(),
        ReportCell::Float(value) => value.to_string(),
        ReportCell::Boolean(value) => value.to_string(),
        ReportCell::Missing => String::new(),
    }
}

fn optional_value(values: Option<&[f64]>, index: usize) -> ReportCell {
    values
        .map(|values| ReportCell::Float(values[index]))
        .unwrap_or(ReportCell::Missing)
}

fn optional_nested(values: Option<&[Option<f64>]>, index: usize) -> ReportCell {
    values
        .and_then(|values| values[index])
        .map(ReportCell::Float)
        .unwrap_or(ReportCell::Missing)
}

fn integer(value: usize) -> Result<ReportCell> {
    i64::try_from(value)
        .map(ReportCell::Integer)
        .map_err(|_| FeecGmrfError::InvalidParameter("integer exceeds CSV range".to_string()))
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn estimator_name(estimator: VarianceEstimator) -> &'static str {
    match estimator {
        VarianceEstimator::Exact => "exact",
        VarianceEstimator::MonteCarlo => "monte_carlo",
        VarianceEstimator::Hutchinson => "hutchinson",
    }
}

fn csv_error(error: csv::Error) -> FeecGmrfError {
    FeecGmrfError::Io(std::io::Error::other(error))
}
