use crate::team13::{NU_AIR, NU_IRON};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use feg_infer::{
    prior::matern::{
        one_form::{build_matern_precision_1form_with_mass_inverse_for_alpha, HodgeLaplacian1Form},
        MaternAlpha,
    },
    sparse::{
        dense_to_feec_csr, feec_csr_to_dense, feec_csr_to_gmrf, scale_matrix, symmetrize_feec_csr,
    },
};
use formoniq::{
    assemble,
    operators::InnerProductWeightClosure,
    problems::reduced_linear::{
        build_reduced_hodge_laplace_1form_system,
        build_reduced_weighted_hodge_laplace_1form_system, ReducedLinearPdeAssembly,
    },
    reduction::{EssentialBoundarySpec, PrescribedDof},
};
use gmrf_core::{
    observation::{apply_gaussian_observations, observation_selector},
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf,
};
use manifold::{
    gen::cartesian::CartesianMeshInfo,
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
};

const DEFAULT_DROP_TOLERANCE: f64 = 1e-14;
const DEFAULT_MAX_EXACT_DOFS: usize = 2_000;
const INTERFACE_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct WeightedHodgeMaternIsolationConfig {
    pub level: usize,
    pub contrasts: Vec<f64>,
    pub kappa_factors: Vec<f64>,
    pub max_exact_dofs: usize,
    pub drop_tolerance: f64,
}

impl Default for WeightedHodgeMaternIsolationConfig {
    fn default() -> Self {
        Self {
            level: 4,
            contrasts: vec![1.0, 1e2, NU_AIR / NU_IRON, 1e4, 1e6],
            kappa_factors: vec![1.0, 10.0, 100.0],
            max_exact_dofs: DEFAULT_MAX_EXACT_DOFS,
            drop_tolerance: DEFAULT_DROP_TOLERANCE,
        }
    }
}

impl WeightedHodgeMaternIsolationConfig {
    fn validate(&self) -> Result<(), String> {
        if self.level == 0 {
            return Err("split-square level must be positive".to_string());
        }
        if self.contrasts.is_empty() {
            return Err("at least one material contrast is required".to_string());
        }
        if self
            .contrasts
            .iter()
            .any(|contrast| !contrast.is_finite() || *contrast <= 0.0)
        {
            return Err("material contrasts must be finite and positive".to_string());
        }
        if self.kappa_factors.is_empty() {
            return Err("at least one kappa factor is required".to_string());
        }
        if self
            .kappa_factors
            .iter()
            .any(|factor| !factor.is_finite() || *factor <= 0.0)
        {
            return Err("kappa factors must be finite and positive".to_string());
        }
        if self.max_exact_dofs == 0 {
            return Err("max_exact_dofs must be positive".to_string());
        }
        if !self.drop_tolerance.is_finite() || self.drop_tolerance < 0.0 {
            return Err("drop_tolerance must be finite and nonnegative".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WeightedHodgeMaternIsolationReport {
    pub config: WeightedHodgeMaternIsolationConfig,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub active_edge_count: usize,
    pub triangle_count: usize,
    pub inverse_domain_diameter: f64,
    pub unit_weight_diagnostics: UnitWeightAssemblyDiagnostics,
    pub active_edges: Vec<ActiveEdgeGeometry>,
    pub scenarios: Vec<WeightedHodgeMaternScenarioReport>,
}

#[derive(Debug, Clone, Copy)]
pub struct UnitWeightAssemblyDiagnostics {
    pub operator_max_abs_difference: f64,
    pub mass_max_abs_difference: f64,
    pub projected_inverse_max_abs_difference: f64,
}

#[derive(Debug, Clone)]
pub struct WeightedHodgeMaternScenarioReport {
    pub contrast: f64,
    pub kappa_factor: f64,
    pub kappa: f64,
    pub baseline_mean_prior_variance: f64,
    pub observation_variance: f64,
    pub probes: Vec<EdgeProbe>,
    pub strategies: Vec<WeightedHodgeMaternStrategyReport>,
}

#[derive(Debug, Clone)]
pub struct WeightedHodgeMaternStrategyReport {
    pub kind: WeightedMassInverseKind,
    pub normalization: VarianceNormalization,
    pub tau: f64,
    pub mass_stats: MatrixEntryStats,
    pub mass_inverse_stats: MatrixEntryStats,
    pub laplacian_stats: MatrixEntryStats,
    pub precision_stats: MatrixEntryStats,
    pub precision_eigen: EigenStats,
    pub precision_lower_nnz: usize,
    pub factor_nnz: usize,
    pub posterior_factor_nnz: usize,
    pub fill_in_ratio: f64,
    pub prior_variance_stats: VarianceStats,
    pub posterior_variance_stats: VarianceStats,
    pub region_summaries: Vec<RegionVarianceSummary>,
    pub probe_results: Vec<EdgeProbePosterior>,
    pub mean_probe_variance_ratio: f64,
    pub median_unobserved_variance_ratio: f64,
    pub edge_results: Vec<EdgeVarianceResult>,
}

#[derive(Debug, Clone)]
pub struct ActiveEdgeGeometry {
    pub reduced_index: usize,
    pub full_edge_index: usize,
    pub vertices: [usize; 2],
    pub midpoint: [f64; 2],
    pub length: f64,
    pub region: MaterialRegion,
}

#[derive(Debug, Clone)]
pub struct EdgeProbe {
    pub kind: EdgeProbeKind,
    pub reduced_index: usize,
    pub full_edge_index: usize,
    pub region: MaterialRegion,
}

#[derive(Debug, Clone)]
pub struct EdgeProbePosterior {
    pub kind: EdgeProbeKind,
    pub reduced_index: usize,
    pub full_edge_index: usize,
    pub region: MaterialRegion,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct EdgeVarianceResult {
    pub reduced_index: usize,
    pub full_edge_index: usize,
    pub region: MaterialRegion,
    pub prior_variance: f64,
    pub posterior_variance: f64,
    pub variance_ratio: f64,
    pub prior_log_delta_vs_exact: Option<f64>,
    pub posterior_log_delta_vs_exact: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RegionVarianceSummary {
    pub region: MaterialRegion,
    pub edge_count: usize,
    pub prior_mean: f64,
    pub posterior_mean: f64,
    pub variance_ratio_mean: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WeightedMassInverseKind {
    ProjectedWeightedSparseInverse,
    ExactConsistentWeightedMass,
    SplitUnweightedGraph,
    SplitUnweightedProjectedSparseInverse,
}

impl WeightedMassInverseKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::ProjectedWeightedSparseInverse => "projected_weighted_sparse_inverse",
            Self::ExactConsistentWeightedMass => "exact_consistent_weighted_mass",
            Self::SplitUnweightedGraph => "split_unweighted_graph",
            Self::SplitUnweightedProjectedSparseInverse => {
                "split_unweighted_projected_sparse_inverse"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VarianceNormalization {
    RawTauOne,
    TraceMatchedBaseline,
}

impl VarianceNormalization {
    pub fn label(self) -> &'static str {
        match self {
            Self::RawTauOne => "raw_tau_1",
            Self::TraceMatchedBaseline => "trace_matched_baseline",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaterialRegion {
    WeightedSide,
    Interface,
    UnitSide,
}

impl MaterialRegion {
    pub fn label(self) -> &'static str {
        match self {
            Self::WeightedSide => "weighted_side",
            Self::Interface => "interface",
            Self::UnitSide => "unit_side",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeProbeKind {
    WeightedSideCenter,
    UnitSideCenter,
    Interface,
}

impl EdgeProbeKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::WeightedSideCenter => "weighted_side_center",
            Self::UnitSideCenter => "unit_side_center",
            Self::Interface => "interface",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MatrixEntryStats {
    pub nnz: usize,
    pub density: f64,
    pub min_abs_nonzero: f64,
    pub max_abs: f64,
    pub abs_entry_ratio: f64,
    pub asymmetry_max_abs: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EigenStats {
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub condition_number: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VarianceStats {
    pub min: f64,
    pub mean: f64,
    pub median: f64,
    pub max: f64,
}

pub fn compute_weighted_hodge_matern_isolation_report(
    config: WeightedHodgeMaternIsolationConfig,
) -> Result<WeightedHodgeMaternIsolationReport, String> {
    config.validate()?;

    let mesh = CartesianMeshInfo::new_unit_scaled(2, config.level, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let boundary = split_square_outer_boundary(&topology, &coords);
    let inverse_domain_diameter = inverse_domain_diameter(&coords);
    let unit_weight_diagnostics =
        compute_unit_weight_assembly_diagnostics(&topology, &coords, &metric, &boundary)?;

    let baseline_system =
        build_weighted_split_square_system(&topology, &coords, &metric, &boundary, 1.0)?;
    let active_edges = active_edge_geometry(&topology, &coords, &baseline_system)?;
    let probes = select_edge_probes(&active_edges)?;
    let baseline_by_kappa =
        compute_baseline_mean_variances(&config, &baseline_system, inverse_domain_diameter)?;

    let mut scenarios = Vec::new();
    for &contrast in &config.contrasts {
        let system =
            build_weighted_split_square_system(&topology, &coords, &metric, &boundary, contrast)?;
        if system.layout.active_dofs != baseline_system.layout.active_dofs {
            return Err("split-square active edge layout changed across contrasts".to_string());
        }
        for &kappa_factor in &config.kappa_factors {
            let kappa = kappa_factor * inverse_domain_diameter;
            let baseline_mean_prior_variance = *baseline_by_kappa
                .get(&factor_key(kappa_factor))
                .ok_or_else(|| {
                    format!("missing baseline variance for kappa factor {kappa_factor}")
                })?;
            let observation_variance = 0.01 * baseline_mean_prior_variance;
            let strategies = compute_scenario_strategies(
                &config,
                &system,
                &baseline_system,
                &active_edges,
                &probes,
                kappa,
                baseline_mean_prior_variance,
                observation_variance,
            )?;
            scenarios.push(WeightedHodgeMaternScenarioReport {
                contrast,
                kappa_factor,
                kappa,
                baseline_mean_prior_variance,
                observation_variance,
                probes: probes.clone(),
                strategies,
            });
        }
    }

    Ok(WeightedHodgeMaternIsolationReport {
        config,
        vertex_count: topology.nsimplices(0),
        edge_count: topology.nsimplices(1),
        active_edge_count: active_edges.len(),
        triangle_count: topology.nsimplices(2),
        inverse_domain_diameter,
        unit_weight_diagnostics,
        active_edges,
        scenarios,
    })
}

pub fn write_weighted_hodge_matern_isolation_outputs(
    report: &WeightedHodgeMaternIsolationReport,
    out_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_summary_csv(report, &out_dir.join("summary.csv"))?;
    write_edge_variance_csv(
        report,
        &out_dir.join(format!("edge_variances_level_{}.csv", report.config.level)),
    )?;
    write_probe_posterior_csv(report, &out_dir.join("probe_posterior.csv"))?;
    Ok(())
}

fn compute_baseline_mean_variances(
    config: &WeightedHodgeMaternIsolationConfig,
    system: &ReducedLinearPdeAssembly,
    inverse_domain_diameter: f64,
) -> Result<BTreeMap<String, f64>, String> {
    let mass = system.state_mass.clone();
    let laplacian = system.operator.clone();
    let exact_inverse = exact_consistent_mass_inverse(&mass, config)?;
    let hodge = HodgeLaplacian1Form {
        mass_u: mass,
        laplacian,
    };

    let mut baseline = BTreeMap::new();
    for &kappa_factor in &config.kappa_factors {
        let kappa = kappa_factor * inverse_domain_diameter;
        let precision =
            symmetrize_feec_csr(&build_matern_precision_1form_with_mass_inverse_for_alpha(
                &hodge,
                &exact_inverse,
                MaternAlpha::Two,
                kappa,
                1.0,
            ));
        let diagnostics = exact_variance_diagnostics(&precision)?;
        baseline.insert(factor_key(kappa_factor), diagnostics.variance_stats.mean);
    }
    Ok(baseline)
}

fn compute_scenario_strategies(
    config: &WeightedHodgeMaternIsolationConfig,
    system: &ReducedLinearPdeAssembly,
    unweighted_system: &ReducedLinearPdeAssembly,
    active_edges: &[ActiveEdgeGeometry],
    probes: &[EdgeProbe],
    kappa: f64,
    baseline_mean_prior_variance: f64,
    observation_variance: f64,
) -> Result<Vec<WeightedHodgeMaternStrategyReport>, String> {
    let mass = system.state_mass.clone();
    let laplacian = system.operator.clone();
    let mass_stats = matrix_entry_stats(&mass);
    let laplacian_stats = matrix_entry_stats(&laplacian);
    let hodge = HodgeLaplacian1Form {
        mass_u: mass.clone(),
        laplacian: laplacian.clone(),
    };

    let mut mass_inverses = Vec::new();
    let projected = system
        .state_mass_inverse
        .as_ref()
        .ok_or_else(|| {
            "weighted split-square system is missing projected sparse inverse".to_string()
        })?
        .clone();
    mass_inverses.push((
        WeightedMassInverseKind::ProjectedWeightedSparseInverse,
        projected,
    ));
    if mass.nrows() <= config.max_exact_dofs {
        mass_inverses.push((
            WeightedMassInverseKind::ExactConsistentWeightedMass,
            exact_consistent_mass_inverse(&mass, config)?,
        ));
    }

    let mut raw_reports = Vec::new();
    for (kind, mass_inverse) in mass_inverses {
        let pre_sym_precision = build_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            MaternAlpha::Two,
            kappa,
            1.0,
        );
        let raw_precision = symmetrize_feec_csr(&pre_sym_precision);
        let raw_report = compute_strategy_report(
            kind,
            VarianceNormalization::RawTauOne,
            1.0,
            &mass,
            mass_stats,
            &mass_inverse,
            laplacian_stats,
            &raw_precision,
            matrix_asymmetry_max_abs(&pre_sym_precision),
            active_edges,
            probes,
            observation_variance,
        )?;
        let tau = (raw_report.prior_variance_stats.mean / baseline_mean_prior_variance).sqrt();
        if !tau.is_finite() || tau <= 0.0 {
            return Err(format!(
                "invalid trace-matching tau {tau} for {}",
                kind.label()
            ));
        }
        let trace_precision = scale_matrix(&raw_precision, tau * tau);
        let trace_report = compute_strategy_report(
            kind,
            VarianceNormalization::TraceMatchedBaseline,
            tau,
            &mass,
            mass_stats,
            &mass_inverse,
            laplacian_stats,
            &trace_precision,
            tau * tau * matrix_asymmetry_max_abs(&pre_sym_precision),
            active_edges,
            probes,
            observation_variance,
        )?;
        raw_reports.push(raw_report);
        raw_reports.push(trace_report);
    }

    for source in [
        SplitUnweightedMassInverseSource::ExactConsistentMass,
        SplitUnweightedMassInverseSource::ProjectedSparseInverse,
    ] {
        let kind = source.report_kind();
        let split_components = build_split_unweighted_graph_components(
            config,
            unweighted_system,
            active_edges,
            kappa,
            source,
        )?;
        let split_raw_report = compute_strategy_report(
            kind,
            VarianceNormalization::RawTauOne,
            1.0,
            &split_components.mass,
            matrix_entry_stats(&split_components.mass),
            &split_components.mass_inverse,
            matrix_entry_stats(&split_components.laplacian),
            &split_components.precision,
            split_components.pre_sym_asymmetry,
            active_edges,
            probes,
            observation_variance,
        )?;
        let split_tau =
            (split_raw_report.prior_variance_stats.mean / baseline_mean_prior_variance).sqrt();
        if !split_tau.is_finite() || split_tau <= 0.0 {
            return Err(format!(
                "invalid trace-matching tau {split_tau} for {}",
                kind.label()
            ));
        }
        let split_trace_precision =
            scale_matrix(&split_components.precision, split_tau * split_tau);
        let split_trace_report = compute_strategy_report(
            kind,
            VarianceNormalization::TraceMatchedBaseline,
            split_tau,
            &split_components.mass,
            matrix_entry_stats(&split_components.mass),
            &split_components.mass_inverse,
            matrix_entry_stats(&split_components.laplacian),
            &split_trace_precision,
            split_tau * split_tau * split_components.pre_sym_asymmetry,
            active_edges,
            probes,
            observation_variance,
        )?;
        raw_reports.push(split_raw_report);
        raw_reports.push(split_trace_report);
    }

    attach_exact_log_deltas(&mut raw_reports)?;
    Ok(raw_reports)
}

fn compute_strategy_report(
    kind: WeightedMassInverseKind,
    normalization: VarianceNormalization,
    tau: f64,
    _mass: &FeecCsr,
    mass_stats: MatrixEntryStats,
    mass_inverse: &FeecCsr,
    laplacian_stats: MatrixEntryStats,
    precision: &FeecCsr,
    precision_pre_sym_asymmetry: f64,
    active_edges: &[ActiveEdgeGeometry],
    probes: &[EdgeProbe],
    observation_variance: f64,
) -> Result<WeightedHodgeMaternStrategyReport, String> {
    let mass_inverse_stats = matrix_entry_stats(mass_inverse);
    let mut precision_stats = matrix_entry_stats(precision);
    precision_stats.asymmetry_max_abs = precision_pre_sym_asymmetry;
    let precision_eigen = symmetric_extreme_eigenvalues(precision)?;
    let precision_lower_nnz = lower_triangle_nnz(precision);
    let diagnostics =
        exact_prior_posterior_variance_diagnostics(precision, probes, observation_variance)?;
    let region_summaries = region_variance_summaries(
        active_edges,
        &diagnostics.prior_variance,
        &diagnostics.posterior_variance,
    )?;
    let edge_results = active_edges
        .iter()
        .map(|edge| {
            let prior = diagnostics.prior_variance[edge.reduced_index];
            let posterior = diagnostics.posterior_variance[edge.reduced_index];
            EdgeVarianceResult {
                reduced_index: edge.reduced_index,
                full_edge_index: edge.full_edge_index,
                region: edge.region,
                prior_variance: prior,
                posterior_variance: posterior,
                variance_ratio: posterior / prior,
                prior_log_delta_vs_exact: None,
                posterior_log_delta_vs_exact: None,
            }
        })
        .collect::<Vec<_>>();
    let probe_results = probes
        .iter()
        .map(|probe| {
            let prior = diagnostics.prior_variance[probe.reduced_index];
            let posterior = diagnostics.posterior_variance[probe.reduced_index];
            EdgeProbePosterior {
                kind: probe.kind,
                reduced_index: probe.reduced_index,
                full_edge_index: probe.full_edge_index,
                region: probe.region,
                prior_variance: prior,
                posterior_variance: posterior,
                variance_ratio: posterior / prior,
            }
        })
        .collect::<Vec<_>>();
    let probe_indices = probes
        .iter()
        .map(|probe| probe.reduced_index)
        .collect::<Vec<_>>();
    let mean_probe_variance_ratio = mean(
        probe_results
            .iter()
            .map(|probe| probe.variance_ratio)
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    let median_unobserved_variance_ratio = median(
        edge_results
            .iter()
            .filter(|edge| !probe_indices.contains(&edge.reduced_index))
            .map(|edge| edge.variance_ratio)
            .collect::<Vec<_>>(),
    )?;

    Ok(WeightedHodgeMaternStrategyReport {
        kind,
        normalization,
        tau,
        mass_stats,
        mass_inverse_stats,
        laplacian_stats,
        precision_stats,
        precision_eigen,
        precision_lower_nnz,
        factor_nnz: diagnostics.factor_nnz,
        posterior_factor_nnz: diagnostics.posterior_factor_nnz,
        fill_in_ratio: diagnostics.factor_nnz as f64 / precision_lower_nnz.max(1) as f64,
        prior_variance_stats: diagnostics.prior_stats,
        posterior_variance_stats: diagnostics.posterior_stats,
        region_summaries,
        probe_results,
        mean_probe_variance_ratio,
        median_unobserved_variance_ratio,
        edge_results,
    })
}

fn attach_exact_log_deltas(
    reports: &mut [WeightedHodgeMaternStrategyReport],
) -> Result<(), String> {
    let exact_by_normalization = reports
        .iter()
        .filter(|report| report.kind == WeightedMassInverseKind::ExactConsistentWeightedMass)
        .map(|report| (report.normalization, report.edge_results.clone()))
        .collect::<BTreeMap<_, _>>();

    for report in reports {
        let Some(exact_edges) = exact_by_normalization.get(&report.normalization) else {
            continue;
        };
        if exact_edges.len() != report.edge_results.len() {
            return Err("exact comparison edge counts do not match".to_string());
        }
        for (edge, exact) in report.edge_results.iter_mut().zip(exact_edges) {
            if edge.reduced_index != exact.reduced_index {
                return Err("exact comparison edge ordering does not match".to_string());
            }
            edge.prior_log_delta_vs_exact = Some((edge.prior_variance / exact.prior_variance).ln());
            edge.posterior_log_delta_vs_exact =
                Some((edge.posterior_variance / exact.posterior_variance).ln());
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct SplitUnweightedGraphComponents {
    mass: FeecCsr,
    mass_inverse: FeecCsr,
    laplacian: FeecCsr,
    precision: FeecCsr,
    pre_sym_asymmetry: f64,
}

#[derive(Debug, Clone, Copy)]
enum SplitUnweightedMassInverseSource {
    ExactConsistentMass,
    ProjectedSparseInverse,
}

impl SplitUnweightedMassInverseSource {
    fn report_kind(self) -> WeightedMassInverseKind {
        match self {
            Self::ExactConsistentMass => WeightedMassInverseKind::SplitUnweightedGraph,
            Self::ProjectedSparseInverse => {
                WeightedMassInverseKind::SplitUnweightedProjectedSparseInverse
            }
        }
    }
}

fn build_split_unweighted_graph_components(
    config: &WeightedHodgeMaternIsolationConfig,
    system: &ReducedLinearPdeAssembly,
    active_edges: &[ActiveEdgeGeometry],
    kappa: f64,
    mass_inverse_source: SplitUnweightedMassInverseSource,
) -> Result<SplitUnweightedGraphComponents, String> {
    let full_mass = system.state_mass.clone();
    let full_laplacian = system.operator.clone();
    let full_projected_inverse = match mass_inverse_source {
        SplitUnweightedMassInverseSource::ExactConsistentMass => None,
        SplitUnweightedMassInverseSource::ProjectedSparseInverse => Some(
            system
                .state_mass_inverse
                .as_ref()
                .ok_or_else(|| {
                    "unweighted split-square system is missing projected sparse inverse".to_string()
                })?
                .clone(),
        ),
    };
    let groups = split_graph_groups(active_edges)?;
    let dimension = full_mass.nrows();
    let mut mass_blocks = Vec::with_capacity(groups.len());
    let mut mass_inverse_blocks = Vec::with_capacity(groups.len());
    let mut laplacian_blocks = Vec::with_capacity(groups.len());
    let mut precision_blocks = Vec::with_capacity(groups.len());
    let mut pre_symmetry_blocks = Vec::with_capacity(groups.len());

    for group in &groups {
        let mass = restrict_square_by_indices(&full_mass, group)?;
        let laplacian = restrict_square_by_indices(&full_laplacian, group)?;
        let mass_inverse = match mass_inverse_source {
            SplitUnweightedMassInverseSource::ExactConsistentMass => {
                exact_consistent_mass_inverse(&mass, config)?
            }
            SplitUnweightedMassInverseSource::ProjectedSparseInverse => restrict_square_by_indices(
                full_projected_inverse
                    .as_ref()
                    .expect("projected inverse is present for projected split source"),
                group,
            )?,
        };
        let hodge = HodgeLaplacian1Form {
            mass_u: mass.clone(),
            laplacian: laplacian.clone(),
        };
        let pre_sym_precision = build_matern_precision_1form_with_mass_inverse_for_alpha(
            &hodge,
            &mass_inverse,
            MaternAlpha::Two,
            kappa,
            1.0,
        );
        pre_symmetry_blocks.push(matrix_asymmetry_max_abs(&pre_sym_precision));
        precision_blocks.push(symmetrize_feec_csr(&pre_sym_precision));
        mass_blocks.push(mass);
        mass_inverse_blocks.push(mass_inverse);
        laplacian_blocks.push(laplacian);
    }

    Ok(SplitUnweightedGraphComponents {
        mass: scatter_square_blocks(dimension, &groups, &mass_blocks)?,
        mass_inverse: scatter_square_blocks(dimension, &groups, &mass_inverse_blocks)?,
        laplacian: scatter_square_blocks(dimension, &groups, &laplacian_blocks)?,
        precision: scatter_square_blocks(dimension, &groups, &precision_blocks)?,
        pre_sym_asymmetry: pre_symmetry_blocks
            .into_iter()
            .fold(0.0_f64, |acc, value| acc.max(value)),
    })
}

fn split_graph_groups(active_edges: &[ActiveEdgeGeometry]) -> Result<Vec<Vec<usize>>, String> {
    let mut weighted_side = Vec::new();
    let mut unit_side = Vec::new();
    for edge in active_edges {
        if edge.midpoint[0] < 0.5 {
            weighted_side.push(edge.reduced_index);
        } else {
            unit_side.push(edge.reduced_index);
        }
    }
    if weighted_side.is_empty() || unit_side.is_empty() {
        return Err(
            "split graph prior requires non-empty weighted and unit side groups".to_string(),
        );
    }
    Ok(vec![weighted_side, unit_side])
}

fn restrict_square_by_indices(matrix: &FeecCsr, indices: &[usize]) -> Result<FeecCsr, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "split graph restriction requires a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut map = BTreeMap::new();
    for (local, index) in indices.iter().copied().enumerate() {
        if index >= matrix.nrows() {
            return Err(format!(
                "split graph index {index} is outside matrix dimension {}",
                matrix.nrows()
            ));
        }
        if map.insert(index, local).is_some() {
            return Err(format!("split graph index {index} appears more than once"));
        }
    }
    let mut coo = FeecCoo::new(indices.len(), indices.len());
    for (row, col, value) in matrix.triplet_iter() {
        if let (Some(local_row), Some(local_col)) = (map.get(&row), map.get(&col)) {
            if *value != 0.0 {
                coo.push(*local_row, *local_col, *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

fn scatter_square_blocks(
    dimension: usize,
    groups: &[Vec<usize>],
    blocks: &[FeecCsr],
) -> Result<FeecCsr, String> {
    if groups.len() != blocks.len() {
        return Err("split graph block count does not match group count".to_string());
    }
    let mut coo = FeecCoo::new(dimension, dimension);
    for (group, block) in groups.iter().zip(blocks) {
        if block.nrows() != group.len() || block.ncols() != group.len() {
            return Err(format!(
                "split graph block dimensions {}x{} do not match group size {}",
                block.nrows(),
                block.ncols(),
                group.len()
            ));
        }
        for (row, col, value) in block.triplet_iter() {
            if *value != 0.0 {
                coo.push(group[row], group[col], *value);
            }
        }
    }
    Ok(FeecCsr::from(&coo))
}

#[derive(Debug, Clone)]
struct PriorPosteriorVarianceDiagnostics {
    prior_variance: FeecVector,
    posterior_variance: FeecVector,
    prior_stats: VarianceStats,
    posterior_stats: VarianceStats,
    factor_nnz: usize,
    posterior_factor_nnz: usize,
}

#[derive(Debug, Clone)]
struct VarianceDiagnostics {
    variance_stats: VarianceStats,
}

fn exact_prior_posterior_variance_diagnostics(
    precision: &FeecCsr,
    probes: &[EdgeProbe],
    observation_variance: f64,
) -> Result<PriorPosteriorVarianceDiagnostics, String> {
    if !observation_variance.is_finite() || observation_variance <= 0.0 {
        return Err("observation variance must be finite and positive".to_string());
    }
    let gmrf_precision = feec_csr_to_gmrf(precision);
    let prior_factor = gmrf_precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision: {err}"))?;
    let factor_nnz = prior_factor.nnz();
    let prior_variance = exact_variances_from_factor(gmrf_precision.clone(), prior_factor)?;

    let probe_indices = probes
        .iter()
        .map(|probe| probe.reduced_index)
        .collect::<Vec<_>>();
    let selector = observation_selector(precision.nrows(), &probe_indices);
    let observations = GmrfVector::zeros(probe_indices.len());
    let (posterior_precision, _) = apply_gaussian_observations(
        &gmrf_precision,
        &selector,
        &observations,
        None,
        observation_variance,
    );
    let posterior_factor = posterior_precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor posterior precision: {err}"))?;
    let posterior_factor_nnz = posterior_factor.nnz();
    let posterior_variance = exact_variances_from_factor(posterior_precision, posterior_factor)?;

    Ok(PriorPosteriorVarianceDiagnostics {
        prior_stats: variance_stats(prior_variance.iter().copied())?,
        posterior_stats: variance_stats(posterior_variance.iter().copied())?,
        prior_variance,
        posterior_variance,
        factor_nnz,
        posterior_factor_nnz,
    })
}

fn exact_variance_diagnostics(precision: &FeecCsr) -> Result<VarianceDiagnostics, String> {
    let gmrf_precision = feec_csr_to_gmrf(precision);
    let factor = gmrf_precision
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor baseline precision: {err}"))?;
    let variance = exact_variances_from_factor(gmrf_precision, factor)?;
    Ok(VarianceDiagnostics {
        variance_stats: variance_stats(variance.iter().copied())?,
    })
}

fn exact_variances_from_factor(
    precision: gmrf_core::types::SparseMatrix,
    factor: gmrf_core::types::SparseCholeskyFactor,
) -> Result<FeecVector, String> {
    let dim = precision.nrows();
    let constraints = GmrfDenseMatrix::zeros(0, dim);
    let mut gmrf = Gmrf::from_mean_and_precision(GmrfVector::zeros(dim), precision)
        .map_err(|err| err.to_string())?
        .with_precision_sqrt(factor);
    let decomposition = gmrf
        .exact_constrained_variance_decomposition(&constraints)
        .map_err(|err| err.to_string())?;
    Ok(FeecVector::from_iterator(
        decomposition.unconstrained_diag.len(),
        decomposition.unconstrained_diag.iter().copied(),
    ))
}

fn exact_consistent_mass_inverse(
    mass: &FeecCsr,
    config: &WeightedHodgeMaternIsolationConfig,
) -> Result<FeecCsr, String> {
    if mass.nrows() > config.max_exact_dofs {
        return Err(format!(
            "exact weighted mass inverse requires {} dofs, exceeding max_exact_dofs {}",
            mass.nrows(),
            config.max_exact_dofs
        ));
    }
    let inverse = feec_csr_to_dense(mass)
        .try_inverse()
        .ok_or_else(|| "failed to invert reduced weighted 1-form mass matrix".to_string())?;
    let inverse = (&inverse + inverse.transpose()) * 0.5;
    Ok(dense_to_feec_csr(&inverse, config.drop_tolerance))
}

fn build_weighted_split_square_system(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    boundary: &EssentialBoundarySpec,
    contrast: f64,
) -> Result<ReducedLinearPdeAssembly, String> {
    let weight =
        InnerProductWeightClosure::new(move |point| if point[0] < 0.5 { contrast } else { 1.0 });
    build_reduced_weighted_hodge_laplace_1form_system(
        topology, metric, coords, None, &weight, boundary,
    )
}

fn split_square_outer_boundary(topology: &Complex, coords: &MeshCoords) -> EssentialBoundarySpec {
    let state_dofs = sorted_boundary_dofs(topology, coords, 1);
    let auxiliary_dofs = sorted_boundary_dofs(topology, coords, 0);
    EssentialBoundarySpec {
        state: state_dofs
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
        auxiliary: auxiliary_dofs
            .into_iter()
            .map(|index| PrescribedDof { index, value: 0.0 })
            .collect(),
    }
}

fn sorted_boundary_dofs(topology: &Complex, coords: &MeshCoords, dim: usize) -> Vec<usize> {
    let mut dofs = assemble::boundary_simplices_where_barycenter(topology, coords, dim, |point| {
        near(point[0], 0.0) || near(point[0], 1.0) || near(point[1], 0.0) || near(point[1], 1.0)
    });
    dofs.sort_unstable();
    dofs
}

fn compute_unit_weight_assembly_diagnostics(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    boundary: &EssentialBoundarySpec,
) -> Result<UnitWeightAssemblyDiagnostics, String> {
    let unweighted = build_reduced_hodge_laplace_1form_system(topology, metric, boundary)?;
    let weighted = build_weighted_split_square_system(topology, coords, metric, boundary, 1.0)?;
    let unweighted_inverse = unweighted
        .state_mass_inverse
        .as_ref()
        .ok_or_else(|| "unweighted 1-form system is missing mass inverse".to_string())?;
    let weighted_inverse = weighted
        .state_mass_inverse
        .as_ref()
        .ok_or_else(|| "weighted unit 1-form system is missing mass inverse".to_string())?;
    Ok(UnitWeightAssemblyDiagnostics {
        operator_max_abs_difference: triplet_max_abs_difference(
            &unweighted.operator,
            &weighted.operator,
        )?,
        mass_max_abs_difference: triplet_max_abs_difference(
            &unweighted.state_mass,
            &weighted.state_mass,
        )?,
        projected_inverse_max_abs_difference: triplet_max_abs_difference(
            unweighted_inverse,
            weighted_inverse,
        )?,
    })
}

fn triplet_max_abs_difference(lhs: &FeecCsr, rhs: &FeecCsr) -> Result<f64, String> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err(format!(
            "cannot compare matrix dimensions {}x{} and {}x{}",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    let lhs = feec_csr_to_dense(lhs);
    let rhs = feec_csr_to_dense(rhs);
    Ok((lhs - rhs)
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs())))
}

fn active_edge_geometry(
    topology: &Complex,
    coords: &MeshCoords,
    system: &ReducedLinearPdeAssembly,
) -> Result<Vec<ActiveEdgeGeometry>, String> {
    let all_edges = full_edge_geometry(topology, coords)?;
    system
        .layout
        .active_dofs
        .iter()
        .copied()
        .enumerate()
        .map(|(reduced_index, full_edge_index)| {
            let mut edge = all_edges.get(full_edge_index).cloned().ok_or_else(|| {
                format!("active edge {full_edge_index} is outside edge geometry table")
            })?;
            edge.reduced_index = reduced_index;
            Ok(edge)
        })
        .collect()
}

fn full_edge_geometry(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<ActiveEdgeGeometry>, String> {
    topology
        .edges()
        .handle_iter()
        .enumerate()
        .map(|(edge_index, edge)| {
            let vertices: [usize; 2] = (*edge)
                .clone()
                .try_into()
                .map_err(|_| "failed to convert edge simplex to two vertices".to_string())?;
            let a = coords.coord(vertices[0]);
            let b = coords.coord(vertices[1]);
            let midpoint = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
            let length = (b - a).norm();
            Ok(ActiveEdgeGeometry {
                reduced_index: edge_index,
                full_edge_index: edge.kidx(),
                vertices,
                midpoint,
                length,
                region: material_region(midpoint[0]),
            })
        })
        .collect()
}

fn select_edge_probes(active_edges: &[ActiveEdgeGeometry]) -> Result<Vec<EdgeProbe>, String> {
    if active_edges.len() < 3 {
        return Err("at least three active edges are required for probe selection".to_string());
    }
    let targets = [
        (EdgeProbeKind::WeightedSideCenter, [0.25, 0.5]),
        (EdgeProbeKind::UnitSideCenter, [0.75, 0.5]),
        (EdgeProbeKind::Interface, [0.5, 0.5]),
    ];
    let mut selected = Vec::with_capacity(targets.len());
    for (kind, target) in targets {
        let edge = active_edges
            .iter()
            .filter(|edge| {
                !selected
                    .iter()
                    .any(|probe: &EdgeProbe| probe.reduced_index == edge.reduced_index)
            })
            .min_by(|lhs, rhs| {
                squared_distance(lhs.midpoint, target)
                    .partial_cmp(&squared_distance(rhs.midpoint, target))
                    .expect("finite distances should compare")
            })
            .ok_or_else(|| format!("failed to select {} edge probe", kind.label()))?;
        selected.push(EdgeProbe {
            kind,
            reduced_index: edge.reduced_index,
            full_edge_index: edge.full_edge_index,
            region: edge.region,
        });
    }
    Ok(selected)
}

fn material_region(x: f64) -> MaterialRegion {
    if (x - 0.5).abs() <= INTERFACE_TOLERANCE {
        MaterialRegion::Interface
    } else if x < 0.5 {
        MaterialRegion::WeightedSide
    } else {
        MaterialRegion::UnitSide
    }
}

fn region_variance_summaries(
    edges: &[ActiveEdgeGeometry],
    prior: &FeecVector,
    posterior: &FeecVector,
) -> Result<Vec<RegionVarianceSummary>, String> {
    [
        MaterialRegion::WeightedSide,
        MaterialRegion::Interface,
        MaterialRegion::UnitSide,
    ]
    .into_iter()
    .map(|region| {
        let indices = edges
            .iter()
            .filter(|edge| edge.region == region)
            .map(|edge| edge.reduced_index)
            .collect::<Vec<_>>();
        if indices.is_empty() {
            return Ok(RegionVarianceSummary {
                region,
                edge_count: 0,
                prior_mean: f64::NAN,
                posterior_mean: f64::NAN,
                variance_ratio_mean: f64::NAN,
            });
        }
        let prior_values = indices
            .iter()
            .map(|&index| prior[index])
            .collect::<Vec<_>>();
        let posterior_values = indices
            .iter()
            .map(|&index| posterior[index])
            .collect::<Vec<_>>();
        let ratios = indices
            .iter()
            .map(|&index| posterior[index] / prior[index])
            .collect::<Vec<_>>();
        Ok(RegionVarianceSummary {
            region,
            edge_count: indices.len(),
            prior_mean: mean(&prior_values)?,
            posterior_mean: mean(&posterior_values)?,
            variance_ratio_mean: mean(&ratios)?,
        })
    })
    .collect()
}

fn matrix_entry_stats(matrix: &FeecCsr) -> MatrixEntryStats {
    let mut min_abs_nonzero = f64::INFINITY;
    let mut max_abs = 0.0_f64;
    for (_row, _col, value) in matrix.triplet_iter() {
        let abs = value.abs();
        if abs > 0.0 {
            min_abs_nonzero = min_abs_nonzero.min(abs);
            max_abs = max_abs.max(abs);
        }
    }
    if !min_abs_nonzero.is_finite() {
        min_abs_nonzero = 0.0;
    }
    MatrixEntryStats {
        nnz: matrix.nnz(),
        density: sparse_density(matrix),
        min_abs_nonzero,
        max_abs,
        abs_entry_ratio: if min_abs_nonzero > 0.0 {
            max_abs / min_abs_nonzero
        } else {
            f64::INFINITY
        },
        asymmetry_max_abs: matrix_asymmetry_max_abs(matrix),
    }
}

fn matrix_asymmetry_max_abs(matrix: &FeecCsr) -> f64 {
    if matrix.nrows() != matrix.ncols() {
        return f64::NAN;
    }
    let dense = feec_csr_to_dense(matrix);
    let diff = dense.clone() - dense.transpose();
    diff.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn symmetric_extreme_eigenvalues(matrix: &FeecCsr) -> Result<EigenStats, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "eigenvalue diagnostics require a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if matrix.nrows() == 0 {
        return Err("eigenvalue diagnostics require a non-empty matrix".to_string());
    }
    let dense = feec_csr_to_dense(matrix);
    let symmetric = (&dense + dense.transpose()) * 0.5;
    let eigen = symmetric.symmetric_eigen();
    let mut values = eigen.eigenvalues.iter().copied().collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("eigenvalue diagnostics produced a non-finite eigenvalue".to_string());
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues should compare"));
    let lambda_min = values[0];
    let lambda_max = *values.last().expect("non-empty eigenvalue vector");
    if lambda_min <= 0.0 {
        return Err(format!(
            "eigenvalue diagnostics require an SPD matrix, got lambda_min={lambda_min:.6e}"
        ));
    }
    Ok(EigenStats {
        lambda_min,
        lambda_max,
        condition_number: lambda_max / lambda_min,
    })
}

fn variance_stats(values: impl Iterator<Item = f64>) -> Result<VarianceStats, String> {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return Err("cannot summarize an empty variance vector".to_string());
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("variance summaries require finite positive values".to_string());
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite values should compare"));
    let min = values[0];
    let max = *values.last().expect("non-empty values");
    Ok(VarianceStats {
        min,
        mean: mean(&values)?,
        median: median(values)?,
        max,
    })
}

fn mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot average an empty vector".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("mean input contains non-finite values".to_string());
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn median(mut values: Vec<f64>) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot compute median of an empty vector".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("median input contains non-finite values".to_string());
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite values should compare"));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Ok(0.5 * (values[mid - 1] + values[mid]))
    } else {
        Ok(values[mid])
    }
}

fn sparse_density(matrix: &FeecCsr) -> f64 {
    let entries = matrix.nrows() * matrix.ncols();
    if entries == 0 {
        0.0
    } else {
        matrix.nnz() as f64 / entries as f64
    }
}

fn lower_triangle_nnz(matrix: &FeecCsr) -> usize {
    matrix
        .triplet_iter()
        .filter(|(row, col, value)| row >= col && value.abs() > 0.0)
        .count()
}

fn inverse_domain_diameter(coords: &MeshCoords) -> f64 {
    if coords.nvertices() == 0 {
        return 1.0;
    }
    let first = coords.coord(0);
    let mut min = vec![0.0; coords.dim()];
    let mut max = vec![0.0; coords.dim()];
    for d in 0..coords.dim() {
        min[d] = first[d];
        max[d] = first[d];
    }
    for vertex in 1..coords.nvertices() {
        let coord = coords.coord(vertex);
        for d in 0..coords.dim() {
            min[d] = min[d].min(coord[d]);
            max[d] = max[d].max(coord[d]);
        }
    }
    let diameter = min
        .iter()
        .zip(max.iter())
        .map(|(lo, hi)| (hi - lo).powi(2))
        .sum::<f64>()
        .sqrt();
    1.0 / diameter.max(1e-12)
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10
}

fn squared_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn factor_key(value: f64) -> String {
    format!("{value:.12e}")
}

fn write_summary_csv(report: &WeightedHodgeMaternIsolationReport, path: &Path) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,vertices,edges,active_edges,triangles,inverse_domain_diameter,unit_operator_max_abs_diff,unit_mass_max_abs_diff,unit_projected_inverse_max_abs_diff,contrast,kappa_factor,kappa,baseline_mean_prior_variance,observation_variance,strategy,normalization,tau,mass_nnz,mass_min_abs,mass_max_abs,mass_abs_entry_ratio,mass_asymmetry,mass_inverse_nnz,mass_inverse_min_abs,mass_inverse_max_abs,mass_inverse_abs_entry_ratio,mass_inverse_asymmetry,laplacian_nnz,laplacian_min_abs,laplacian_max_abs,laplacian_abs_entry_ratio,laplacian_asymmetry,precision_nnz,precision_min_abs,precision_max_abs,precision_abs_entry_ratio,precision_pre_sym_asymmetry,precision_lambda_min,precision_lambda_max,precision_condition_number,precision_lower_nnz,factor_nnz,posterior_factor_nnz,fill_in_ratio,prior_variance_min,prior_variance_mean,prior_variance_median,prior_variance_max,posterior_variance_min,posterior_variance_mean,posterior_variance_median,posterior_variance_max,mean_probe_variance_ratio,median_unobserved_variance_ratio,weighted_side_edge_count,weighted_side_prior_mean,weighted_side_posterior_mean,weighted_side_variance_ratio_mean,interface_edge_count,interface_prior_mean,interface_posterior_mean,interface_variance_ratio_mean,unit_side_edge_count,unit_side_prior_mean,unit_side_posterior_mean,unit_side_variance_ratio_mean"
    )?;
    for scenario in &report.scenarios {
        for strategy in &scenario.strategies {
            let weighted = region_summary(strategy, MaterialRegion::WeightedSide);
            let interface = region_summary(strategy, MaterialRegion::Interface);
            let unit = region_summary(strategy, MaterialRegion::UnitSide);
            let fields = vec![
                report.config.level.to_string(),
                report.vertex_count.to_string(),
                report.edge_count.to_string(),
                report.active_edge_count.to_string(),
                report.triangle_count.to_string(),
                f64_csv(report.inverse_domain_diameter),
                f64_csv(report.unit_weight_diagnostics.operator_max_abs_difference),
                f64_csv(report.unit_weight_diagnostics.mass_max_abs_difference),
                f64_csv(
                    report
                        .unit_weight_diagnostics
                        .projected_inverse_max_abs_difference,
                ),
                f64_csv(scenario.contrast),
                f64_csv(scenario.kappa_factor),
                f64_csv(scenario.kappa),
                f64_csv(scenario.baseline_mean_prior_variance),
                f64_csv(scenario.observation_variance),
                strategy.kind.label().to_string(),
                strategy.normalization.label().to_string(),
                f64_csv(strategy.tau),
                strategy.mass_stats.nnz.to_string(),
                f64_csv(strategy.mass_stats.min_abs_nonzero),
                f64_csv(strategy.mass_stats.max_abs),
                f64_csv(strategy.mass_stats.abs_entry_ratio),
                f64_csv(strategy.mass_stats.asymmetry_max_abs),
                strategy.mass_inverse_stats.nnz.to_string(),
                f64_csv(strategy.mass_inverse_stats.min_abs_nonzero),
                f64_csv(strategy.mass_inverse_stats.max_abs),
                f64_csv(strategy.mass_inverse_stats.abs_entry_ratio),
                f64_csv(strategy.mass_inverse_stats.asymmetry_max_abs),
                strategy.laplacian_stats.nnz.to_string(),
                f64_csv(strategy.laplacian_stats.min_abs_nonzero),
                f64_csv(strategy.laplacian_stats.max_abs),
                f64_csv(strategy.laplacian_stats.abs_entry_ratio),
                f64_csv(strategy.laplacian_stats.asymmetry_max_abs),
                strategy.precision_stats.nnz.to_string(),
                f64_csv(strategy.precision_stats.min_abs_nonzero),
                f64_csv(strategy.precision_stats.max_abs),
                f64_csv(strategy.precision_stats.abs_entry_ratio),
                f64_csv(strategy.precision_stats.asymmetry_max_abs),
                f64_csv(strategy.precision_eigen.lambda_min),
                f64_csv(strategy.precision_eigen.lambda_max),
                f64_csv(strategy.precision_eigen.condition_number),
                strategy.precision_lower_nnz.to_string(),
                strategy.factor_nnz.to_string(),
                strategy.posterior_factor_nnz.to_string(),
                f64_csv(strategy.fill_in_ratio),
                f64_csv(strategy.prior_variance_stats.min),
                f64_csv(strategy.prior_variance_stats.mean),
                f64_csv(strategy.prior_variance_stats.median),
                f64_csv(strategy.prior_variance_stats.max),
                f64_csv(strategy.posterior_variance_stats.min),
                f64_csv(strategy.posterior_variance_stats.mean),
                f64_csv(strategy.posterior_variance_stats.median),
                f64_csv(strategy.posterior_variance_stats.max),
                f64_csv(strategy.mean_probe_variance_ratio),
                f64_csv(strategy.median_unobserved_variance_ratio),
                weighted.edge_count.to_string(),
                f64_csv(weighted.prior_mean),
                f64_csv(weighted.posterior_mean),
                f64_csv(weighted.variance_ratio_mean),
                interface.edge_count.to_string(),
                f64_csv(interface.prior_mean),
                f64_csv(interface.posterior_mean),
                f64_csv(interface.variance_ratio_mean),
                unit.edge_count.to_string(),
                f64_csv(unit.prior_mean),
                f64_csv(unit.posterior_mean),
                f64_csv(unit.variance_ratio_mean),
            ];
            writeln!(writer, "{}", fields.join(","))?;
        }
    }
    Ok(())
}

fn write_edge_variance_csv(
    report: &WeightedHodgeMaternIsolationReport,
    path: &Path,
) -> io::Result<()> {
    let geometry = report
        .active_edges
        .iter()
        .map(|edge| (edge.reduced_index, edge))
        .collect::<BTreeMap<_, _>>();
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,contrast,kappa_factor,kappa,strategy,normalization,tau,reduced_edge_index,full_edge_index,vertex_a,vertex_b,midpoint_x,midpoint_y,edge_length,material_region,prior_variance,posterior_variance,variance_ratio,prior_log_delta_vs_exact,posterior_log_delta_vs_exact"
    )?;
    for scenario in &report.scenarios {
        for strategy in &scenario.strategies {
            for edge in &strategy.edge_results {
                let Some(geom) = geometry.get(&edge.reduced_index) else {
                    continue;
                };
                writeln!(
                    writer,
                    "{},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{},{},{},{},{:.16e},{:.16e},{:.16e},{},{:.16e},{:.16e},{:.16e},{},{}",
                    report.config.level,
                    scenario.contrast,
                    scenario.kappa_factor,
                    scenario.kappa,
                    strategy.kind.label(),
                    strategy.normalization.label(),
                    strategy.tau,
                    edge.reduced_index,
                    edge.full_edge_index,
                    geom.vertices[0],
                    geom.vertices[1],
                    geom.midpoint[0],
                    geom.midpoint[1],
                    geom.length,
                    edge.region.label(),
                    edge.prior_variance,
                    edge.posterior_variance,
                    edge.variance_ratio,
                    optional_f64(edge.prior_log_delta_vs_exact),
                    optional_f64(edge.posterior_log_delta_vs_exact)
                )?;
            }
        }
    }
    Ok(())
}

fn write_probe_posterior_csv(
    report: &WeightedHodgeMaternIsolationReport,
    path: &Path,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "level,contrast,kappa_factor,kappa,baseline_mean_prior_variance,observation_variance,strategy,normalization,tau,probe,reduced_edge_index,full_edge_index,material_region,prior_variance,posterior_variance,variance_ratio"
    )?;
    for scenario in &report.scenarios {
        for strategy in &scenario.strategies {
            for probe in &strategy.probe_results {
                writeln!(
                    writer,
                    "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{:.16e},{},{},{},{},{:.16e},{:.16e},{:.16e}",
                    report.config.level,
                    scenario.contrast,
                    scenario.kappa_factor,
                    scenario.kappa,
                    scenario.baseline_mean_prior_variance,
                    scenario.observation_variance,
                    strategy.kind.label(),
                    strategy.normalization.label(),
                    strategy.tau,
                    probe.kind.label(),
                    probe.reduced_index,
                    probe.full_edge_index,
                    probe.region.label(),
                    probe.prior_variance,
                    probe.posterior_variance,
                    probe.variance_ratio
                )?;
            }
        }
    }
    Ok(())
}

fn region_summary(
    strategy: &WeightedHodgeMaternStrategyReport,
    region: MaterialRegion,
) -> RegionVarianceSummary {
    strategy
        .region_summaries
        .iter()
        .find(|summary| summary.region == region)
        .cloned()
        .unwrap_or(RegionVarianceSummary {
            region,
            edge_count: 0,
            prior_mean: f64::NAN,
            posterior_mean: f64::NAN,
            variance_ratio_mean: f64::NAN,
        })
}

fn optional_f64(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.16e}"))
}

fn f64_csv(value: f64) -> String {
    format!("{value:.16e}")
}
