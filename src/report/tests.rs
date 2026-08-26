use super::*;
use crate::boundary::EssentialBoundaryConditions;
use crate::infer::{HutchinsonVarianceConfig, MonteCarloVarianceConfig, VarianceEstimator};
use crate::model::{DerivedQuantity, LinearConstraint, LinearGaussianModelBuilder};
use crate::operator::{LinearMap, SparseMat};
use crate::prior::GaussianPrior;
use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use manifold::gen::cartesian::CartesianMeshInfo;
use std::fs;

fn basic_posterior() -> Posterior {
    let prior = GaussianPrior::new(
        vec![1.0, -2.0],
        SparseMat::from_rows(2, &[vec![(0, 2.0)], vec![(1, 4.0)]]).unwrap(),
    )
    .unwrap();
    let qoi = LinearMap::weighted_rows(2, &[vec![(0, 1.0), (1, 1.0)], vec![(0, 1.0), (1, -1.0)]])
        .unwrap();
    LinearGaussianModelBuilder::new(prior)
        .derive(DerivedQuantity::new("qoi", qoi).unwrap())
        .unwrap()
        .condition()
        .unwrap()
}

#[test]
fn builder_reports_latent_cochain_derived_and_ad_hoc_fields() {
    let mut posterior = basic_posterior();
    let map = LinearMap::selector(2, &[1]).unwrap();
    let report = PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::latent("latent", "Latent").truth(vec![1.0, -2.0]))
        .field(FieldRequest::cochain("cochain", "Cochain"))
        .field(FieldRequest::derived("derived", "Derived", "qoi"))
        .field(FieldRequest::mapped("mapped", "Mapped", map))
        .include_factorization_diagnostics(true)
        .build()
        .unwrap();
    assert!((report.field("latent").unwrap().variance.values[0] - 0.5).abs() < 1.0e-12);
    assert!((report.field("latent").unwrap().variance.values[1] - 0.25).abs() < 1.0e-12);
    assert_eq!(report.field("cochain").unwrap().mean, [1.0, -2.0]);
    assert_eq!(report.field("derived").unwrap().mean, [-1.0, 3.0]);
    assert_eq!(report.field("mapped").unwrap().mean, [-2.0]);
    assert!(report.factorization.is_some());
}

#[test]
fn boundary_lifting_inserts_prescribed_mean_and_zero_variance() {
    let prior = GaussianPrior::new(vec![0.0; 3], SparseMat::diagonal(3, 1.0))
        .unwrap()
        .condition_on_essential_boundary(
            EssentialBoundaryConditions::prescribed(vec![1], vec![3.0]).unwrap(),
        )
        .unwrap();
    let mut posterior = LinearGaussianModelBuilder::new(prior).condition().unwrap();
    let report = PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::cochain("full", "Full cochain"))
        .build()
        .unwrap();
    let field = report.field("full").unwrap();
    assert_eq!(field.mean, [0.0, 3.0, 0.0]);
    assert_eq!(field.variance.values, [1.0, 0.0, 1.0]);
}

#[test]
fn qoi_uses_exact_constrained_covariance_and_correlation() {
    let prior = GaussianPrior::new(vec![0.0; 2], SparseMat::diagonal(2, 1.0)).unwrap();
    let identity = LinearMap::identity(2);
    let constraint = LinearConstraint::new(
        LinearMap::weighted_row(2, &[(0, 1.0), (1, 1.0)]).unwrap(),
        vec![0.0],
    )
    .unwrap();
    let mut posterior = LinearGaussianModelBuilder::new(prior)
        .derive(DerivedQuantity::new("state", identity).unwrap())
        .unwrap()
        .constrain(constraint)
        .unwrap()
        .condition()
        .unwrap();
    let report = PosteriorReportBuilder::new(&mut posterior)
        .qoi(QoiRequest::derived(
            "state_qoi",
            "State QoI",
            "state",
            vec!["x".into(), "y".into()],
        ))
        .build()
        .unwrap();
    let qoi = report.qoi("state_qoi").unwrap();
    assert!((qoi.covariance[0][0] - 0.5).abs() < 1.0e-12);
    assert!((qoi.covariance[0][1] + 0.5).abs() < 1.0e-12);
    assert!((qoi.correlation[0][1] + 1.0).abs() < 1.0e-12);
}

#[test]
fn exact_mc_hutchinson_and_auto_metadata_are_preserved() {
    let mut posterior = basic_posterior();
    let hutchinson = HutchinsonVarianceConfig::new(16, 4, 9).unwrap();
    let report = PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::latent("exact", "Exact"))
        .field(
            FieldRequest::latent("mc", "MC").variance_method(VarianceMethod::MonteCarlo(
                MonteCarloVarianceConfig::new(128, 4, 7).unwrap(),
            )),
        )
        .field(
            FieldRequest::latent("hutchinson", "Hutchinson")
                .variance_method(VarianceMethod::Hutchinson(hutchinson)),
        )
        .field(
            FieldRequest::latent("automatic", "Automatic")
                .variance_method(VarianceMethod::auto(2, hutchinson).unwrap()),
        )
        .field(
            FieldRequest::latent("automatic_hutchinson", "Automatic Hutchinson")
                .variance_method(VarianceMethod::auto(1, hutchinson).unwrap()),
        )
        .build()
        .unwrap();
    assert_eq!(
        report.field("exact").unwrap().variance.estimator,
        VarianceEstimator::Exact
    );
    assert_eq!(
        report.field("mc").unwrap().variance.estimator,
        VarianceEstimator::MonteCarlo
    );
    assert_eq!(report.field("mc").unwrap().variance.sample_count, 128);
    assert_eq!(
        report.field("hutchinson").unwrap().variance.estimator,
        VarianceEstimator::Hutchinson
    );
    assert_eq!(
        report.field("automatic").unwrap().variance.estimator,
        VarianceEstimator::Exact
    );
    assert_eq!(
        report
            .field("automatic_hutchinson")
            .unwrap()
            .variance
            .estimator,
        VarianceEstimator::Hutchinson
    );
}

#[test]
fn prediction_reuses_generic_gaussian_diagnostics() {
    let mut posterior = basic_posterior();
    let report = PosteriorReportBuilder::new(&mut posterior)
        .prediction(PredictionRequest::mapped(
            "held_out",
            "Held out",
            LinearMap::identity(2),
            vec!["a".into(), "b".into()],
            vec![1.5, -2.0],
            vec![0.5, 0.75],
        ))
        .build()
        .unwrap();
    let prediction = report.prediction("held_out").unwrap();
    assert_eq!(prediction.diagnostics.residuals, [0.5, 0.0]);
    assert!(prediction.diagnostics.rmse > 0.0);
}

#[test]
fn standard_deviations_clamp_negative_estimates_without_hiding_raw_values() {
    assert_eq!(
        super::standard_deviations(&[-0.25, 0.0, 4.0]),
        [0.0, 0.0, 2.0]
    );
}

#[test]
fn build_rejects_duplicate_ids_dimensions_missing_outputs_and_nonfinite_metrics() {
    let mut posterior = basic_posterior();
    assert!(PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::latent("same", "A"))
        .metric(ReportMetric::new("same", "B", 1.0))
        .build()
        .is_err());
    assert!(PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::latent("bad_truth", "Bad").truth(vec![0.0]))
        .build()
        .is_err());
    assert!(PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::latent("bad_reference", "Bad").reference(vec![0.0]))
        .build()
        .is_err());
    assert!(PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::latent("bad_baseline", "Bad").baseline_variances(vec![1.0]))
        .build()
        .is_err());
    assert!(PosteriorReportBuilder::new(&mut posterior)
        .field(FieldRequest::derived(
            "missing",
            "Missing",
            "does-not-exist"
        ))
        .build()
        .is_err());
    assert!(PosteriorReportBuilder::new(&mut posterior)
        .metric(ReportMetric::new("bad", "Bad", f64::NAN))
        .build()
        .is_err());
}

#[test]
fn console_output_is_deterministic() {
    let report = PosteriorReport {
        fields: vec![],
        qois: vec![],
        predictions: vec![],
        metrics: vec![ReportMetric::new("rmse", "RMSE", 1.25).unit("m")],
        factorization: None,
    };
    let mut output = Vec::new();
    write_console_report(
        &mut output,
        &report,
        &ConsoleReportOptions {
            precision: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(output).unwrap(),
        "Posterior report\nmetric rmse (RMSE): 1.25 m\n"
    );
}

#[test]
fn report_table_validates_width_finiteness_and_csv_quoting() {
    let mut table = ReportTable::new(
        "quoted",
        vec!["text".into(), "value".into(), "missing".into()],
    )
    .unwrap();
    table
        .push_row(vec![
            ReportCell::Text("comma, quote \" tab\t newline\n".into()),
            ReportCell::Float(2.5),
            ReportCell::Missing,
        ])
        .unwrap();
    assert!(table.push_row(vec![ReportCell::Missing]).is_err());
    assert!(ReportTable::new("duplicates", vec!["x".into(), "x".into()]).is_err());
    assert!(ReportTable::new("nonfinite", vec!["x".into()])
        .unwrap()
        .push_row(vec![ReportCell::Float(f64::INFINITY)])
        .is_err());

    let path = temporary_path("quoted.csv");
    table.write_csv(&path).unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("text,value,missing\n"));
    assert!(contents.contains("\"comma, quote \"\" tab\t newline\n\""));
    let _ = fs::remove_file(path);
}

#[test]
fn vtu_builders_validate_layouts_and_emit_named_fields() {
    assert_eq!(
        VectorLayout3::ComponentMajor
            .to_vectors(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap(),
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    );
    assert!(VectorLayout3::Interleaved.to_vectors(&[1.0, 2.0]).is_err());

    let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let vertex_count = coords.nvertices();
    let mut cochains = CochainVtuBuilder::new(0);
    cochains
        .add_values("posterior_mean", vec![1.0; vertex_count])
        .unwrap();
    assert!(cochains.add_values("bad", vec![0.0]).is_err());
    assert!(CochainVtuBuilder::new(0)
        .add_cochain(
            "wrong_degree",
            Cochain::new(1, FeecVector::from_vec(vec![0.0])),
        )
        .is_err());
    let path = temporary_path("cochain.vtu");
    cochains.write(&path, &coords, &topology).unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    assert!(contents.contains("posterior_mean"));
    let _ = fs::remove_file(path);

    let top_count = topology.skeleton(2).len();
    let mut cells = TopCellVtuBuilder::new();
    cells
        .add_scalar("variance", vec![0.25; top_count])
        .unwrap()
        .add_vector("field", vec![[1.0, 2.0, 3.0]; top_count])
        .unwrap();
    assert!(cells.add_scalar("wrong_size", vec![0.0]).is_err());
    let path = temporary_path("top-cell.vtu");
    cells.write(&path, &coords, &topology).unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    assert!(contents.contains("variance"));
    assert!(contents.contains("field"));
    let _ = fs::remove_file(path);
}

fn temporary_path(name: &str) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("feec_gmrf_report_{stamp}_{name}"))
}
