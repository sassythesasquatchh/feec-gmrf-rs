use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use feg_infer::{
    prior::{
        matern::{
            one_form::{
                build_hodge_laplacian_1form, build_matern_precision_1form_for_alpha,
                build_reconstructed_barycenter_field_operator, MaternConfig as Matern1FormConfig,
                MaternMassInverse as Matern1FormMassInverse,
            },
            zero_form::{
                build_laplace_beltrami_0form, build_matern_precision_0form_for_alpha,
                MaternConfig as Matern0FormConfig, MaternMassInverse as Matern0FormMassInverse,
            },
            MaternAlpha,
        },
        trace_normalization::{trace_normalization_from_mean_variance, TraceNormalization},
    },
    sparse::{feec_csr_to_gmrf, symmetrize_feec_csr},
};
use gmrf_core::{
    estimate_hutchinson_transformed_variances, estimate_hutchinson_variances,
    estimate_hutchinson_weighted_covariance_trace,
    estimate_hutchinson_weighted_transformed_covariance_trace, exact_solve_diag,
    exact_solve_transformed_diag, exact_weighted_covariance_trace,
    exact_weighted_transformed_covariance_trace, Gmrf, ProbeBatchConfig, ProbeDistribution,
    SparseCholeskyFactor, SparseMatrix as GmrfSparse, SparseRowOperator, Vector as GmrfVector,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
    topology::complex::Complex,
};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
};

const DEFAULT_PRACTICAL_RANGE: f64 = 0.20;

#[derive(Debug, Clone)]
pub struct MaternTraceNormalizationConfig {
    pub levels: Vec<usize>,
    pub alphas: Vec<MaternAlpha>,
    pub kappa: f64,
    pub target_mean_trace_variance: f64,
    pub exact_max_dofs: usize,
    pub hutchinson_probes: usize,
    pub hutchinson_batches: usize,
    pub rng_seed: u64,
}

impl Default for MaternTraceNormalizationConfig {
    fn default() -> Self {
        Self {
            levels: vec![8, 16, 32, 64],
            alphas: vec![MaternAlpha::One, MaternAlpha::Two, MaternAlpha::Three],
            kappa: (8.0_f64).sqrt() / DEFAULT_PRACTICAL_RANGE,
            target_mean_trace_variance: 1.0,
            exact_max_dofs: 400,
            hutchinson_probes: 128,
            hutchinson_batches: 8,
            rng_seed: 0x51A7_7ACE,
        }
    }
}

impl MaternTraceNormalizationConfig {
    /// Cheap deterministic configuration intended for continuous integration.
    pub fn smoke() -> Self {
        Self {
            levels: vec![2],
            alphas: vec![MaternAlpha::Two],
            exact_max_dofs: 128,
            hutchinson_probes: 8,
            hutchinson_batches: 2,
            ..Self::default()
        }
    }

    /// Immutable configuration used by the submitted thesis.
    pub fn thesis_submitted() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct MaternTraceNormalizationReport {
    pub config: MaternTraceNormalizationConfig,
    pub rows: Vec<MaternTraceNormalizationRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaternTraceStage {
    Scalar0Form,
    Reconstructed1Form,
}

impl MaternTraceStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Scalar0Form => "scalar_0form",
            Self::Reconstructed1Form => "reconstructed_1form",
        }
    }

    fn seed_offset(self) -> u64 {
        match self {
            Self::Scalar0Form => 10_000,
            Self::Reconstructed1Form => 20_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaternTraceNormalizationRow {
    pub stage: MaternTraceStage,
    pub level: usize,
    pub alpha: MaternAlpha,
    pub kappa: f64,
    pub ndofs: usize,
    pub output_dim: usize,
    pub domain_measure: f64,
    pub normalization_source: String,
    pub raw_exact_trace: Option<f64>,
    pub raw_exact_mean_trace_variance: Option<f64>,
    pub raw_hutchinson_trace: f64,
    pub raw_hutchinson_relative_standard_error: Option<f64>,
    pub tau_multiplier: f64,
    pub precision_scale: f64,
    pub normalized_exact_trace: Option<f64>,
    pub normalized_exact_mean_trace_variance: Option<f64>,
    pub normalized_hutchinson_trace: f64,
    pub normalized_hutchinson_mean_trace_variance: f64,
    pub normalized_hutchinson_relative_standard_error: Option<f64>,
    pub raw_marginal_stats: VarianceStats,
    pub normalized_marginal_stats: VarianceStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VarianceStats {
    pub min: f64,
    pub mean: f64,
    pub median: f64,
    pub max: f64,
}

impl VarianceStats {
    fn from_slice(values: &[f64]) -> Result<Self, String> {
        if values.is_empty() {
            return Err("variance stats require at least one value".to_string());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("variance stats contain non-finite values".to_string());
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            let hi = sorted.len() / 2;
            0.5 * (sorted[hi - 1] + sorted[hi])
        } else {
            sorted[sorted.len() / 2]
        };
        Ok(Self {
            min: sorted[0],
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
            median,
            max: *sorted.last().expect("nonempty"),
        })
    }

    fn scaled(self, scale: f64) -> Self {
        Self {
            min: self.min * scale,
            mean: self.mean * scale,
            median: self.median * scale,
            max: self.max * scale,
        }
    }
}

pub fn compute_matern_trace_normalization_report(
    config: MaternTraceNormalizationConfig,
) -> Result<MaternTraceNormalizationReport, String> {
    validate_config(&config)?;
    let mut rows = Vec::new();

    for &level in &config.levels {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, level, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let cell_volumes = cell_volumes(&topology, &coords)?;
        let domain_measure = cell_volumes.iter().sum::<f64>();

        let laplace = build_laplace_beltrami_0form(&topology, &metric);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let reconstruction = build_reconstructed_barycenter_field_operator(&topology, &coords)?;
        let reconstruction_operator = stacked_reconstruction_operator(&topology, &reconstruction)?;
        let reconstruction_weights = stacked_component_cell_weights(coords.dim(), &cell_volumes);

        for &alpha in &config.alphas {
            let zero_precision = symmetrize_feec_csr(&build_matern_precision_0form_for_alpha(
                &laplace,
                alpha,
                Matern0FormConfig {
                    kappa: config.kappa,
                    tau: 1.0,
                    mass_inverse: Matern0FormMassInverse::RowSumLumped,
                },
            ));
            rows.push(compute_sparse_weight_row(
                &config,
                MaternTraceStage::Scalar0Form,
                level,
                alpha,
                config.kappa,
                domain_measure,
                &zero_precision,
                &laplace.mass,
            )?);

            let one_precision = symmetrize_feec_csr(&build_matern_precision_1form_for_alpha(
                &topology,
                &metric,
                &hodge,
                alpha,
                Matern1FormConfig {
                    kappa: config.kappa,
                    tau: 1.0,
                    mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
                },
            ));
            rows.push(compute_transformed_weight_row(
                &config,
                MaternTraceStage::Reconstructed1Form,
                level,
                alpha,
                config.kappa,
                domain_measure,
                &one_precision,
                &reconstruction_operator,
                &reconstruction_weights,
            )?);
        }
    }

    Ok(MaternTraceNormalizationReport { config, rows })
}

pub fn write_matern_trace_normalization_outputs(
    report: &MaternTraceNormalizationReport,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_summary_csv(report, &out_dir.join("summary.csv"))
}

fn validate_config(config: &MaternTraceNormalizationConfig) -> Result<(), String> {
    if config.levels.is_empty() {
        return Err("at least one mesh level is required".to_string());
    }
    if config.levels.contains(&0) {
        return Err("mesh levels must be positive".to_string());
    }
    if config.alphas.is_empty() {
        return Err("at least one Matérn alpha is required".to_string());
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.target_mean_trace_variance.is_finite() || config.target_mean_trace_variance <= 0.0 {
        return Err("target_mean_trace_variance must be finite and positive".to_string());
    }
    if config.hutchinson_probes == 0 {
        return Err("hutchinson_probes must be positive".to_string());
    }
    if config.hutchinson_batches == 0 {
        return Err("hutchinson_batches must be positive".to_string());
    }
    Ok(())
}

// Trace-normalization rows retain stage, discretization, physical measure, and the
// selected weight operator explicitly. The maintained trace smoke study covers it.
#[allow(clippy::too_many_arguments)]
fn compute_sparse_weight_row(
    config: &MaternTraceNormalizationConfig,
    stage: MaternTraceStage,
    level: usize,
    alpha: MaternAlpha,
    kappa: f64,
    domain_measure: f64,
    raw_precision: &FeecCsr,
    weight: &FeecCsr,
) -> Result<MaternTraceNormalizationRow, String> {
    let weight = feec_csr_to_gmrf(weight);
    let (raw_q, raw_factor) = factorize(raw_precision)?;
    let row_seed = row_seed(config.rng_seed, stage, level, alpha);
    let raw_exact = if raw_factor.dimension() <= config.exact_max_dofs {
        Some(exact_weighted_covariance_trace(&raw_factor, &weight).map_err(|err| err.to_string())?)
    } else {
        None
    };
    let raw_hutchinson = estimate_hutchinson_weighted_covariance_trace(
        &raw_factor,
        &weight,
        probe_config(config, row_seed),
        ProbeDistribution::Rademacher,
    )
    .map_err(|err| err.to_string())?;

    let raw_trace_for_normalization = raw_exact
        .as_ref()
        .map_or(raw_hutchinson.value, |estimate| estimate.value);
    let normalization = trace_normalization_from_mean_variance(
        raw_trace_for_normalization,
        config.target_mean_trace_variance,
        domain_measure,
    )?;
    let normalized_precision = normalization.scale_precision(raw_precision);
    let (_normalized_q, normalized_factor) = factorize(&normalized_precision)?;
    let normalized_exact = if raw_exact.is_some() {
        Some(
            exact_weighted_covariance_trace(&normalized_factor, &weight)
                .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let normalized_hutchinson = estimate_hutchinson_weighted_covariance_trace(
        &normalized_factor,
        &weight,
        probe_config(config, row_seed),
        ProbeDistribution::Rademacher,
    )
    .map_err(|err| err.to_string())?;

    let raw_marginal_stats = sparse_marginal_stats(
        &raw_q,
        &raw_factor,
        raw_exact.is_some(),
        config,
        row_seed.wrapping_add(1),
    )?;
    finish_row(
        stage,
        level,
        alpha,
        kappa,
        raw_factor.dimension(),
        raw_factor.dimension(),
        domain_measure,
        raw_exact,
        raw_hutchinson,
        normalization,
        normalized_exact,
        normalized_hutchinson,
        raw_marginal_stats,
    )
}

// Transformed trace rows retain the physical pushforward and output weights as
// separate recorded operators. The maintained trace smoke study covers this route.
#[allow(clippy::too_many_arguments)]
fn compute_transformed_weight_row(
    config: &MaternTraceNormalizationConfig,
    stage: MaternTraceStage,
    level: usize,
    alpha: MaternAlpha,
    kappa: f64,
    domain_measure: f64,
    raw_precision: &FeecCsr,
    operator: &SparseRowOperator,
    output_weights: &GmrfVector,
) -> Result<MaternTraceNormalizationRow, String> {
    let (raw_q, raw_factor) = factorize(raw_precision)?;
    let row_seed = row_seed(config.rng_seed, stage, level, alpha);
    let raw_exact = if raw_factor.dimension() <= config.exact_max_dofs {
        Some(
            exact_weighted_transformed_covariance_trace(&raw_factor, operator, output_weights)
                .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let raw_hutchinson = estimate_hutchinson_weighted_transformed_covariance_trace(
        &raw_factor,
        operator,
        output_weights,
        probe_config(config, row_seed),
        ProbeDistribution::Rademacher,
    )
    .map_err(|err| err.to_string())?;

    let raw_trace_for_normalization = raw_exact
        .as_ref()
        .map_or(raw_hutchinson.value, |estimate| estimate.value);
    let normalization = trace_normalization_from_mean_variance(
        raw_trace_for_normalization,
        config.target_mean_trace_variance,
        domain_measure,
    )?;
    let normalized_precision = normalization.scale_precision(raw_precision);
    let (_normalized_q, normalized_factor) = factorize(&normalized_precision)?;
    let normalized_exact = if raw_exact.is_some() {
        Some(
            exact_weighted_transformed_covariance_trace(
                &normalized_factor,
                operator,
                output_weights,
            )
            .map_err(|err| err.to_string())?,
        )
    } else {
        None
    };
    let normalized_hutchinson = estimate_hutchinson_weighted_transformed_covariance_trace(
        &normalized_factor,
        operator,
        output_weights,
        probe_config(config, row_seed),
        ProbeDistribution::Rademacher,
    )
    .map_err(|err| err.to_string())?;

    let raw_marginal_stats = transformed_marginal_stats(
        &raw_q,
        &raw_factor,
        operator,
        raw_exact.is_some(),
        config,
        row_seed.wrapping_add(1),
    )?;
    finish_row(
        stage,
        level,
        alpha,
        kappa,
        raw_factor.dimension(),
        operator.nrows(),
        domain_measure,
        raw_exact,
        raw_hutchinson,
        normalization,
        normalized_exact,
        normalized_hutchinson,
        raw_marginal_stats,
    )
}

// Final report construction records exact and stochastic estimates side by side;
// keeping these fields explicit prevents confusing reference and estimated values.
#[allow(clippy::too_many_arguments)]
fn finish_row(
    stage: MaternTraceStage,
    level: usize,
    alpha: MaternAlpha,
    kappa: f64,
    ndofs: usize,
    output_dim: usize,
    domain_measure: f64,
    raw_exact: Option<gmrf_core::WeightedTraceEstimate>,
    raw_hutchinson: gmrf_core::WeightedTraceEstimate,
    normalization: TraceNormalization,
    normalized_exact: Option<gmrf_core::WeightedTraceEstimate>,
    normalized_hutchinson: gmrf_core::WeightedTraceEstimate,
    raw_marginal_stats: VarianceStats,
) -> Result<MaternTraceNormalizationRow, String> {
    let normalized_marginal_stats = raw_marginal_stats.scaled(1.0 / normalization.precision_scale);
    Ok(MaternTraceNormalizationRow {
        stage,
        level,
        alpha,
        kappa,
        ndofs,
        output_dim,
        domain_measure,
        normalization_source: if raw_exact.is_some() {
            "exact".to_string()
        } else {
            "hutchinson".to_string()
        },
        raw_exact_trace: raw_exact.as_ref().map(|estimate| estimate.value),
        raw_exact_mean_trace_variance: raw_exact
            .as_ref()
            .map(|estimate| estimate.value / domain_measure),
        raw_hutchinson_trace: raw_hutchinson.value,
        raw_hutchinson_relative_standard_error: raw_hutchinson.relative_standard_error,
        tau_multiplier: normalization.tau_multiplier,
        precision_scale: normalization.precision_scale,
        normalized_exact_trace: normalized_exact.as_ref().map(|estimate| estimate.value),
        normalized_exact_mean_trace_variance: normalized_exact
            .as_ref()
            .map(|estimate| estimate.value / domain_measure),
        normalized_hutchinson_trace: normalized_hutchinson.value,
        normalized_hutchinson_mean_trace_variance: normalized_hutchinson.value / domain_measure,
        normalized_hutchinson_relative_standard_error: normalized_hutchinson
            .relative_standard_error,
        raw_marginal_stats,
        normalized_marginal_stats,
    })
}

fn factorize(precision: &FeecCsr) -> Result<(GmrfSparse, SparseCholeskyFactor), String> {
    let q = feec_csr_to_gmrf(precision);
    let factor = q
        .cholesky_sqrt_lower()
        .map_err(|err| format!("precision factorization failed: {err}"))?;
    Ok((q, factor))
}

fn sparse_marginal_stats(
    precision: &GmrfSparse,
    factor: &SparseCholeskyFactor,
    use_exact: bool,
    config: &MaternTraceNormalizationConfig,
    seed: u64,
) -> Result<VarianceStats, String> {
    let variances = if use_exact {
        exact_solve_diag(factor)
            .map_err(|err| err.to_string())?
            .values
    } else {
        let mut gmrf =
            Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())
                .map_err(|err| err.to_string())?;
        estimate_hutchinson_variances(
            &mut gmrf,
            config.hutchinson_probes,
            config.hutchinson_batches,
            seed,
            ProbeDistribution::Rademacher,
        )
        .map_err(|err| err.to_string())?
        .values
    };
    VarianceStats::from_slice(variances.as_slice())
}

fn transformed_marginal_stats(
    precision: &GmrfSparse,
    factor: &SparseCholeskyFactor,
    operator: &SparseRowOperator,
    use_exact: bool,
    config: &MaternTraceNormalizationConfig,
    seed: u64,
) -> Result<VarianceStats, String> {
    let variances = if use_exact {
        exact_solve_transformed_diag(factor, operator)
            .map_err(|err| err.to_string())?
            .values
    } else {
        let mut gmrf =
            Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())
                .map_err(|err| err.to_string())?;
        estimate_hutchinson_transformed_variances(
            &mut gmrf,
            operator,
            config.hutchinson_probes,
            config.hutchinson_batches,
            seed,
            ProbeDistribution::Rademacher,
        )
        .map_err(|err| err.to_string())?
        .values
    };
    VarianceStats::from_slice(variances.as_slice())
}

fn probe_config(config: &MaternTraceNormalizationConfig, rng_seed: u64) -> ProbeBatchConfig {
    ProbeBatchConfig {
        num_probes: config.hutchinson_probes,
        batch_count: config.hutchinson_batches,
        rng_seed,
    }
}

fn row_seed(base: u64, stage: MaternTraceStage, level: usize, alpha: MaternAlpha) -> u64 {
    base.wrapping_add(stage.seed_offset())
        .wrapping_add((level as u64).wrapping_mul(1_000))
        .wrapping_add(alpha.as_u32() as u64)
}

fn stacked_reconstruction_operator(
    topology: &Complex,
    reconstruction: &feg_infer::prior::matern::one_form::ReconstructedBarycenterFieldOperator,
) -> Result<SparseRowOperator, String> {
    let mut rows =
        Vec::with_capacity(reconstruction.component_count() * reconstruction.cell_count());
    for component_index in 0..reconstruction.component_count() {
        let component_rows = reconstruction
            .component_rows(component_index)
            .ok_or_else(|| format!("missing reconstructed component {component_index}"))?;
        rows.extend(component_rows.iter().cloned());
    }
    SparseRowOperator::new(topology.edges().len(), rows).map_err(|err| err.to_string())
}

fn stacked_component_cell_weights(ambient_dim: usize, cell_volumes: &[f64]) -> GmrfVector {
    GmrfVector::from_iterator(
        ambient_dim * cell_volumes.len(),
        (0..ambient_dim).flat_map(|_| cell_volumes.iter().copied()),
    )
}

fn cell_volumes(topology: &Complex, coords: &MeshCoords) -> Result<Vec<f64>, String> {
    let mut volumes = Vec::with_capacity(topology.cells().len());
    for cell in topology.cells().handle_iter() {
        let volume = cell.coord_simplex(coords).vol();
        if !volume.is_finite() || volume <= 0.0 {
            return Err(format!(
                "cell {} has non-positive finite volume {volume}",
                cell.kidx()
            ));
        }
        volumes.push(volume);
    }
    Ok(volumes)
}

fn write_summary_csv(report: &MaternTraceNormalizationReport, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "stage,level,alpha,kappa,ndofs,output_dim,domain_measure,normalization_source,raw_exact_trace,raw_exact_mean_trace_variance,raw_hutchinson_trace,raw_hutchinson_relative_standard_error,tau_multiplier,precision_scale,normalized_exact_trace,normalized_exact_mean_trace_variance,normalized_hutchinson_trace,normalized_hutchinson_mean_trace_variance,normalized_hutchinson_relative_standard_error,raw_variance_min,raw_variance_mean,raw_variance_median,raw_variance_max,normalized_variance_min,normalized_variance_mean,normalized_variance_median,normalized_variance_max"
    )?;
    for row in &report.rows {
        writeln!(
            writer,
            "{},{},{},{:.16e},{},{},{:.16e},{},{},{},{:.16e},{},{:.16e},{:.16e},{},{},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            row.stage.label(),
            row.level,
            row.alpha.as_u32(),
            row.kappa,
            row.ndofs,
            row.output_dim,
            row.domain_measure,
            row.normalization_source,
            optional_f64(row.raw_exact_trace),
            optional_f64(row.raw_exact_mean_trace_variance),
            row.raw_hutchinson_trace,
            optional_f64(row.raw_hutchinson_relative_standard_error),
            row.tau_multiplier,
            row.precision_scale,
            optional_f64(row.normalized_exact_trace),
            optional_f64(row.normalized_exact_mean_trace_variance),
            row.normalized_hutchinson_trace,
            row.normalized_hutchinson_mean_trace_variance,
            optional_f64(row.normalized_hutchinson_relative_standard_error),
            row.raw_marginal_stats.min,
            row.raw_marginal_stats.mean,
            row.raw_marginal_stats.median,
            row.raw_marginal_stats.max,
            row.normalized_marginal_stats.min,
            row.normalized_marginal_stats.mean,
            row.normalized_marginal_stats.median,
            row.normalized_marginal_stats.max
        )?;
    }
    writer.flush()
}

fn optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.16e}"))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matern_trace_normalization_2d_exact_rows_match_unit_mean_variance() {
        let report = compute_matern_trace_normalization_report(MaternTraceNormalizationConfig {
            levels: vec![2],
            exact_max_dofs: 10_000,
            hutchinson_probes: 8,
            hutchinson_batches: 2,
            ..MaternTraceNormalizationConfig::default()
        })
        .expect("trace-normalization report should build");

        assert_eq!(report.rows.len(), 6);
        for row in &report.rows {
            assert_eq!(row.normalization_source, "exact");
            let exact_mean = row
                .normalized_exact_mean_trace_variance
                .expect("exact normalized trace should be present");
            assert!(
                (exact_mean - 1.0).abs() < 1e-8,
                "{} alpha={} normalized exact mean trace variance = {exact_mean}",
                row.stage.label(),
                row.alpha.as_u32()
            );
            assert!(row.raw_hutchinson_trace.is_finite());
            assert!(row.normalized_hutchinson_mean_trace_variance.is_finite());
            assert!(row.tau_multiplier.is_finite() && row.tau_multiplier > 0.0);
            assert!(row.normalized_marginal_stats.mean.is_finite());
        }
    }
}
