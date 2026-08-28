//! Validated posterior summaries and scientific artifact renderers.
//!
//! Reports query latent, full-cochain, derived, or mapped quantities from a
//! posterior. Applications supply the physical interpretation, units,
//! references, and selected output artifacts.

mod console;
mod table;
mod vtu;

pub use console::{write_console_report, ConsoleReportOptions};
pub use table::{write_csv, write_csv_directory, ReportCell, ReportTable};
pub use vtu::{CochainVtuBuilder, TopCellVtuBuilder, VectorLayout3};

use crate::diagnostics::{gaussian_predictive_diagnostics, GaussianPredictiveDiagnostics};
use crate::infer::{FactorizationDiagnostics, Posterior, VarianceEstimate, VarianceMethod};
use crate::operator::LinearMap;
use crate::{FeecGmrfError, Result};
use std::collections::BTreeSet;

/// A finite scalar attached to a report.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportMetric {
    pub id: String,
    pub label: String,
    pub unit: String,
    pub value: f64,
}

impl ReportMetric {
    pub fn new(id: impl Into<String>, label: impl Into<String>, value: f64) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            unit: String::new(),
            value,
        }
    }

    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }
}

#[derive(Debug, Clone)]
enum ReportMap {
    Latent,
    Cochain,
    Derived(String),
    AdHoc(LinearMap),
}

/// Request for a latent, full-cochain, named-derived, or ad-hoc mapped field.
#[derive(Debug, Clone)]
pub struct FieldRequest {
    id: String,
    label: String,
    unit: String,
    source: ReportMap,
    variance_method: VarianceMethod,
    truth: Option<Vec<f64>>,
    reference: Option<Vec<f64>>,
    baseline_variances: Option<Vec<f64>>,
}

impl FieldRequest {
    pub fn latent(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self::new(id, label, ReportMap::Latent)
    }

    pub fn cochain(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self::new(id, label, ReportMap::Cochain)
    }

    pub fn derived(
        id: impl Into<String>,
        label: impl Into<String>,
        output_name: impl Into<String>,
    ) -> Self {
        Self::new(id, label, ReportMap::Derived(output_name.into()))
    }

    pub fn mapped(id: impl Into<String>, label: impl Into<String>, map: LinearMap) -> Self {
        Self::new(id, label, ReportMap::AdHoc(map))
    }

    fn new(id: impl Into<String>, label: impl Into<String>, source: ReportMap) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            unit: String::new(),
            source,
            variance_method: VarianceMethod::Exact,
            truth: None,
            reference: None,
            baseline_variances: None,
        }
    }

    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }

    pub fn variance_method(mut self, method: VarianceMethod) -> Self {
        self.variance_method = method;
        self
    }

    pub fn truth(mut self, values: Vec<f64>) -> Self {
        self.truth = Some(values);
        self
    }

    pub fn reference(mut self, values: Vec<f64>) -> Self {
        self.reference = Some(values);
        self
    }

    pub fn baseline_variances(mut self, values: Vec<f64>) -> Self {
        self.baseline_variances = Some(values);
        self
    }
}

/// Request for a small exact-covariance block.
#[derive(Debug, Clone)]
pub struct QoiRequest {
    id: String,
    label: String,
    labels: Vec<String>,
    units: Vec<String>,
    source: ReportMap,
    truth: Option<Vec<f64>>,
    reference: Option<Vec<f64>>,
    baseline_variances: Option<Vec<f64>>,
}

impl QoiRequest {
    pub fn derived(
        id: impl Into<String>,
        label: impl Into<String>,
        output_name: impl Into<String>,
        labels: Vec<String>,
    ) -> Self {
        Self::new(id, label, ReportMap::Derived(output_name.into()), labels)
    }

    pub fn mapped(
        id: impl Into<String>,
        label: impl Into<String>,
        map: LinearMap,
        labels: Vec<String>,
    ) -> Self {
        Self::new(id, label, ReportMap::AdHoc(map), labels)
    }

    fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        source: ReportMap,
        labels: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            units: vec![String::new(); labels.len()],
            labels,
            source,
            truth: None,
            reference: None,
            baseline_variances: None,
        }
    }

    pub fn units(mut self, units: Vec<String>) -> Self {
        self.units = units;
        self
    }

    pub fn truth(mut self, values: Vec<f64>) -> Self {
        self.truth = Some(values);
        self
    }

    pub fn reference(mut self, values: Vec<f64>) -> Self {
        self.reference = Some(values);
        self
    }

    pub fn baseline_variances(mut self, values: Vec<f64>) -> Self {
        self.baseline_variances = Some(values);
        self
    }
}

/// Request for diagnostics of independent Gaussian held-out observations.
#[derive(Debug, Clone)]
pub struct PredictionRequest {
    id: String,
    label: String,
    labels: Vec<String>,
    units: Vec<String>,
    source: ReportMap,
    observations: Vec<f64>,
    observation_variances: Vec<f64>,
    variance_method: VarianceMethod,
    interval_standard_deviations: f64,
}

impl PredictionRequest {
    pub fn derived(
        id: impl Into<String>,
        label: impl Into<String>,
        output_name: impl Into<String>,
        labels: Vec<String>,
        observations: Vec<f64>,
        observation_variances: Vec<f64>,
    ) -> Self {
        Self::new(
            id,
            label,
            ReportMap::Derived(output_name.into()),
            labels,
            observations,
            observation_variances,
        )
    }

    pub fn mapped(
        id: impl Into<String>,
        label: impl Into<String>,
        map: LinearMap,
        labels: Vec<String>,
        observations: Vec<f64>,
        observation_variances: Vec<f64>,
    ) -> Self {
        Self::new(
            id,
            label,
            ReportMap::AdHoc(map),
            labels,
            observations,
            observation_variances,
        )
    }

    fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        source: ReportMap,
        labels: Vec<String>,
        observations: Vec<f64>,
        observation_variances: Vec<f64>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            units: vec![String::new(); labels.len()],
            labels,
            source,
            observations,
            observation_variances,
            variance_method: VarianceMethod::Exact,
            interval_standard_deviations: 1.96,
        }
    }

    pub fn units(mut self, units: Vec<String>) -> Self {
        self.units = units;
        self
    }

    pub fn variance_method(mut self, method: VarianceMethod) -> Self {
        self.variance_method = method;
        self
    }

    pub fn interval_standard_deviations(mut self, value: f64) -> Self {
        self.interval_standard_deviations = value;
        self
    }
}

/// Posterior summary for one vector-valued field.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldReport {
    pub id: String,
    pub label: String,
    pub unit: String,
    pub mean: Vec<f64>,
    pub variance: VarianceEstimate,
    pub standard_deviations: Vec<f64>,
    pub truth: Option<Vec<f64>>,
    pub errors: Option<Vec<f64>>,
    pub z_scores: Option<Vec<Option<f64>>>,
    pub reference: Option<Vec<f64>>,
    pub reference_errors: Option<Vec<f64>>,
    pub reference_z_scores: Option<Vec<Option<f64>>>,
    pub baseline_variances: Option<Vec<f64>>,
    pub variance_reductions: Option<Vec<Option<f64>>>,
}

impl FieldReport {
    /// Root-mean-square error against the requested truth values.
    pub fn truth_rmse(&self) -> Option<f64> {
        self.errors.as_ref().map(|errors| {
            (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt()
        })
    }

    /// Fraction of truth values inside the requested pointwise Gaussian band.
    pub fn truth_coverage(&self, standard_deviations: f64) -> Option<f64> {
        if !standard_deviations.is_finite() || standard_deviations < 0.0 {
            return None;
        }
        self.errors.as_ref().map(|errors| {
            errors
                .iter()
                .zip(&self.standard_deviations)
                .filter(|(error, std)| error.abs() <= standard_deviations * **std)
                .count() as f64
                / errors.len() as f64
        })
    }

    /// Largest reported relative standard error, or zero for exact estimates.
    pub fn max_relative_standard_error(&self) -> f64 {
        self.variance
            .relative_standard_error
            .as_ref()
            .map(|values| values.iter().copied().fold(0.0, f64::max))
            .unwrap_or(0.0)
    }
}

/// Exact posterior summary for a small quantity-of-interest block.
#[derive(Debug, Clone, PartialEq)]
pub struct QoiReport {
    pub id: String,
    pub label: String,
    pub labels: Vec<String>,
    pub units: Vec<String>,
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub correlation: Vec<Vec<f64>>,
    pub standard_deviations: Vec<f64>,
    pub truth: Option<Vec<f64>>,
    pub z_scores: Option<Vec<Option<f64>>>,
    pub reference: Option<Vec<f64>>,
    pub reference_z_scores: Option<Vec<Option<f64>>>,
    pub baseline_variances: Option<Vec<f64>>,
    pub variance_reductions: Option<Vec<Option<f64>>>,
}

/// Posterior predictive summary for independent Gaussian observations.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionReport {
    pub id: String,
    pub label: String,
    pub labels: Vec<String>,
    pub units: Vec<String>,
    pub observations: Vec<f64>,
    pub predictive_means: Vec<f64>,
    pub latent_variance: VarianceEstimate,
    pub observation_variances: Vec<f64>,
    pub diagnostics: GaussianPredictiveDiagnostics,
}

/// Fully validated reusable posterior report.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorReport {
    pub fields: Vec<FieldReport>,
    pub qois: Vec<QoiReport>,
    pub predictions: Vec<PredictionReport>,
    pub metrics: Vec<ReportMetric>,
    pub factorization: Option<FactorizationDiagnostics>,
}

impl PosteriorReport {
    pub fn field(&self, id: &str) -> Option<&FieldReport> {
        self.fields.iter().find(|field| field.id == id)
    }

    pub fn qoi(&self, id: &str) -> Option<&QoiReport> {
        self.qois.iter().find(|qoi| qoi.id == id)
    }

    pub fn prediction(&self, id: &str) -> Option<&PredictionReport> {
        self.predictions
            .iter()
            .find(|prediction| prediction.id == id)
    }

    /// Add a validated caller-owned scientific metric after report extraction.
    pub fn push_metric(&mut self, metric: ReportMetric) -> Result<()> {
        validate_artifact_id(&metric.id)?;
        if metric.label.trim().is_empty() || !metric.value.is_finite() {
            return Err(FeecGmrfError::InvalidParameter(
                "report metrics require a non-empty label and finite value".to_string(),
            ));
        }
        let duplicate = self.fields.iter().any(|item| item.id == metric.id)
            || self.qois.iter().any(|item| item.id == metric.id)
            || self.predictions.iter().any(|item| item.id == metric.id)
            || self.metrics.iter().any(|item| item.id == metric.id);
        if duplicate {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "duplicate report artifact ID `{}`",
                metric.id
            )));
        }
        self.metrics.push(metric);
        Ok(())
    }
}

/// Builds a report directly from a solved posterior.
pub struct PosteriorReportBuilder<'a> {
    posterior: &'a mut Posterior,
    fields: Vec<FieldRequest>,
    qois: Vec<QoiRequest>,
    predictions: Vec<PredictionRequest>,
    metrics: Vec<ReportMetric>,
    include_factorization: bool,
}

impl<'a> PosteriorReportBuilder<'a> {
    pub fn new(posterior: &'a mut Posterior) -> Self {
        Self {
            posterior,
            fields: Vec::new(),
            qois: Vec::new(),
            predictions: Vec::new(),
            metrics: Vec::new(),
            include_factorization: false,
        }
    }

    pub fn field(mut self, request: FieldRequest) -> Self {
        self.fields.push(request);
        self
    }

    pub fn qoi(mut self, request: QoiRequest) -> Self {
        self.qois.push(request);
        self
    }

    pub fn prediction(mut self, request: PredictionRequest) -> Self {
        self.predictions.push(request);
        self
    }

    pub fn metric(mut self, metric: ReportMetric) -> Self {
        self.metrics.push(metric);
        self
    }

    pub fn include_factorization_diagnostics(mut self, include: bool) -> Self {
        self.include_factorization = include;
        self
    }

    pub fn build(self) -> Result<PosteriorReport> {
        validate_ids_and_metrics(&self.fields, &self.qois, &self.predictions, &self.metrics)?;
        let PosteriorReportBuilder {
            posterior,
            fields,
            qois,
            predictions,
            metrics,
            include_factorization,
        } = self;

        let mut field_reports = Vec::with_capacity(fields.len());
        for request in fields {
            field_reports.push(build_field(posterior, request)?);
        }
        let mut qoi_reports = Vec::with_capacity(qois.len());
        for request in qois {
            qoi_reports.push(build_qoi(posterior, request)?);
        }
        let mut prediction_reports = Vec::with_capacity(predictions.len());
        for request in predictions {
            prediction_reports.push(build_prediction(posterior, request)?);
        }
        let factorization = include_factorization
            .then(|| posterior.factorization_diagnostics())
            .transpose()?;
        Ok(PosteriorReport {
            fields: field_reports,
            qois: qoi_reports,
            predictions: prediction_reports,
            metrics,
            factorization,
        })
    }
}

fn build_field(posterior: &mut Posterior, request: FieldRequest) -> Result<FieldReport> {
    let (mean, variance) = mean_and_variance(posterior, &request.source, request.variance_method)?;
    validate_finite_vector("field mean", &mean)?;
    if mean.is_empty() {
        return Err(FeecGmrfError::InvalidParameter(
            "field requests must have at least one output".to_string(),
        ));
    }
    validate_estimate(&variance, mean.len())?;
    validate_optional_vector("field truth", &request.truth, mean.len(), false)?;
    validate_optional_vector("field reference", &request.reference, mean.len(), false)?;
    validate_optional_vector(
        "field baseline variances",
        &request.baseline_variances,
        mean.len(),
        true,
    )?;
    let standard_deviations = standard_deviations(&variance.values);
    let (errors, z_scores) =
        errors_and_z_scores(&mean, request.truth.as_deref(), &standard_deviations);
    let (reference_errors, reference_z_scores) =
        errors_and_z_scores(&mean, request.reference.as_deref(), &standard_deviations);
    let variance_reductions =
        variance_reductions(&variance.values, request.baseline_variances.as_deref());
    Ok(FieldReport {
        id: request.id,
        label: request.label,
        unit: request.unit,
        mean,
        variance,
        standard_deviations,
        truth: request.truth,
        errors,
        z_scores,
        reference: request.reference,
        reference_errors,
        reference_z_scores,
        baseline_variances: request.baseline_variances,
        variance_reductions,
    })
}

fn build_qoi(posterior: &mut Posterior, request: QoiRequest) -> Result<QoiReport> {
    if request.labels.is_empty() || request.labels.iter().any(|label| label.is_empty()) {
        return Err(FeecGmrfError::InvalidParameter(
            "QoI labels must be non-empty".to_string(),
        ));
    }
    let mean = mean_for(posterior, &request.source)?;
    let covariance = covariance_for(posterior, &request.source)?;
    let dimension = mean.len();
    if request.labels.len() != dimension || request.units.len() != dimension {
        return Err(FeecGmrfError::Dimension(format!(
            "QoI labels/units must match output dimension {dimension}"
        )));
    }
    validate_finite_vector("QoI mean", &mean)?;
    validate_optional_vector("QoI truth", &request.truth, dimension, false)?;
    validate_optional_vector("QoI reference", &request.reference, dimension, false)?;
    validate_optional_vector(
        "QoI baseline variances",
        &request.baseline_variances,
        dimension,
        true,
    )?;
    let correlation = gmrf_core::covariance_to_correlation(&covariance)?;
    let variances = (0..dimension)
        .map(|index| covariance[index][index])
        .collect::<Vec<_>>();
    let standard_deviations = standard_deviations(&variances);
    let (_, z_scores) = errors_and_z_scores(&mean, request.truth.as_deref(), &standard_deviations);
    let (_, reference_z_scores) =
        errors_and_z_scores(&mean, request.reference.as_deref(), &standard_deviations);
    let variance_reductions =
        variance_reductions(&variances, request.baseline_variances.as_deref());
    Ok(QoiReport {
        id: request.id,
        label: request.label,
        labels: request.labels,
        units: request.units,
        mean,
        covariance,
        correlation,
        standard_deviations,
        truth: request.truth,
        z_scores,
        reference: request.reference,
        reference_z_scores,
        baseline_variances: request.baseline_variances,
        variance_reductions,
    })
}

fn build_prediction(
    posterior: &mut Posterior,
    request: PredictionRequest,
) -> Result<PredictionReport> {
    let (predictive_means, latent_variance) =
        mean_and_variance(posterior, &request.source, request.variance_method)?;
    let dimension = predictive_means.len();
    if request.labels.len() != dimension
        || request.units.len() != dimension
        || request.observations.len() != dimension
        || request.observation_variances.len() != dimension
    {
        return Err(FeecGmrfError::Dimension(format!(
            "prediction labels, units, observations, and variances must match output dimension {dimension}"
        )));
    }
    if request.labels.iter().any(|label| label.trim().is_empty()) {
        return Err(FeecGmrfError::InvalidParameter(
            "prediction labels must be non-empty".to_string(),
        ));
    }
    validate_estimate(&latent_variance, dimension)?;
    let stabilized = latent_variance
        .values
        .iter()
        .map(|variance| variance.max(0.0))
        .collect::<Vec<_>>();
    let diagnostics = gaussian_predictive_diagnostics(
        &request.observations,
        &predictive_means,
        &stabilized,
        &request.observation_variances,
        request.interval_standard_deviations,
    )?;
    Ok(PredictionReport {
        id: request.id,
        label: request.label,
        labels: request.labels,
        units: request.units,
        observations: request.observations,
        predictive_means,
        latent_variance,
        observation_variances: request.observation_variances,
        diagnostics,
    })
}

fn mean_and_variance(
    posterior: &mut Posterior,
    source: &ReportMap,
    method: VarianceMethod,
) -> Result<(Vec<f64>, VarianceEstimate)> {
    let mean = mean_for(posterior, source)?;
    let variance = match source {
        ReportMap::Latent => posterior.latent_variance_estimate(method)?,
        ReportMap::Cochain => posterior.cochain_variance_estimate(method)?,
        ReportMap::Derived(name) => posterior.derived_variance_estimate(name, method)?,
        ReportMap::AdHoc(map) => posterior.pushforward_variance_estimate(map, method)?,
    };
    Ok((mean, variance))
}

fn mean_for(posterior: &Posterior, source: &ReportMap) -> Result<Vec<f64>> {
    match source {
        ReportMap::Latent => Ok(posterior.latent_mean().to_vec()),
        ReportMap::Cochain => Ok(posterior.cochain_mean().to_vec()),
        ReportMap::Derived(name) => posterior.derived_mean(name),
        ReportMap::AdHoc(map) => posterior.pushforward_mean(map),
    }
}

fn covariance_for(posterior: &mut Posterior, source: &ReportMap) -> Result<Vec<Vec<f64>>> {
    match source {
        ReportMap::Derived(name) => posterior.derived_covariance(name),
        ReportMap::AdHoc(map) => posterior.pushforward_covariance(map),
        ReportMap::Latent => {
            posterior.pushforward_covariance(&LinearMap::identity(posterior.mean().len()))
        }
        ReportMap::Cochain => {
            posterior.pushforward_covariance(&LinearMap::identity(posterior.cochain_mean().len()))
        }
    }
}

fn standard_deviations(variances: &[f64]) -> Vec<f64> {
    variances
        .iter()
        .map(|variance| variance.max(0.0).sqrt())
        .collect()
}

fn errors_and_z_scores(
    mean: &[f64],
    target: Option<&[f64]>,
    standard_deviations: &[f64],
) -> (Option<Vec<f64>>, Option<Vec<Option<f64>>>) {
    let Some(target) = target else {
        return (None, None);
    };
    let errors = mean
        .iter()
        .zip(target)
        .map(|(mean, target)| mean - target)
        .collect::<Vec<_>>();
    let z_scores = errors
        .iter()
        .zip(standard_deviations)
        .map(|(error, standard_deviation)| {
            (*standard_deviation > 0.0).then_some(error / standard_deviation)
        })
        .collect();
    (Some(errors), Some(z_scores))
}

fn variance_reductions(variances: &[f64], baseline: Option<&[f64]>) -> Option<Vec<Option<f64>>> {
    baseline.map(|baseline| {
        variances
            .iter()
            .zip(baseline)
            .map(|(posterior, baseline)| {
                (*baseline > 0.0).then_some(1.0 - posterior.max(0.0) / baseline)
            })
            .collect()
    })
}

fn validate_ids_and_metrics(
    fields: &[FieldRequest],
    qois: &[QoiRequest],
    predictions: &[PredictionRequest],
    metrics: &[ReportMetric],
) -> Result<()> {
    let mut ids = BTreeSet::new();
    for (id, label) in fields
        .iter()
        .map(|request| (&request.id, &request.label))
        .chain(qois.iter().map(|request| (&request.id, &request.label)))
        .chain(
            predictions
                .iter()
                .map(|request| (&request.id, &request.label)),
        )
        .chain(metrics.iter().map(|metric| (&metric.id, &metric.label)))
    {
        validate_artifact_id(id)?;
        if label.trim().is_empty() {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "report item `{id}` must have a non-empty label"
            )));
        }
        if !ids.insert(id.as_str()) {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "duplicate report artifact ID `{id}`"
            )));
        }
    }
    if metrics.iter().any(|metric| !metric.value.is_finite()) {
        return Err(FeecGmrfError::InvalidParameter(
            "report metrics must be finite".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_artifact_id(id: &str) -> Result<()> {
    let mut chars = id.chars();
    let first_ok = chars
        .next()
        .is_some_and(|character| character.is_ascii_lowercase() || character.is_ascii_digit());
    if !first_ok
        || !chars.all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || character == '_'
                || character == '-'
        })
    {
        return Err(FeecGmrfError::InvalidParameter(format!(
            "artifact ID `{id}` must use lowercase ASCII letters, digits, `_`, or `-` and start with a letter or digit"
        )));
    }
    Ok(())
}

fn validate_estimate(estimate: &VarianceEstimate, dimension: usize) -> Result<()> {
    if estimate.values.len() != dimension {
        return Err(FeecGmrfError::Dimension(format!(
            "variance estimate length {} does not match mean length {dimension}",
            estimate.values.len()
        )));
    }
    validate_finite_vector("variance estimate", &estimate.values)?;
    if !estimate.minimum_value.is_finite() {
        return Err(FeecGmrfError::InvalidParameter(
            "variance-estimator minimum must be finite".to_string(),
        ));
    }
    validate_optional_vector(
        "variance batch standard errors",
        &estimate.batch_standard_error,
        dimension,
        true,
    )?;
    validate_optional_vector(
        "variance relative standard errors",
        &estimate.relative_standard_error,
        dimension,
        true,
    )?;
    Ok(())
}

fn validate_optional_vector(
    name: &str,
    values: &Option<Vec<f64>>,
    dimension: usize,
    non_negative: bool,
) -> Result<()> {
    if let Some(values) = values {
        if values.len() != dimension {
            return Err(FeecGmrfError::Dimension(format!(
                "{name} length {} does not match output dimension {dimension}",
                values.len()
            )));
        }
        validate_finite_vector(name, values)?;
        if non_negative && values.iter().any(|value| *value < 0.0) {
            return Err(FeecGmrfError::InvalidParameter(format!(
                "{name} must be non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_finite_vector(name: &str, values: &[f64]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FeecGmrfError::InvalidParameter(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
