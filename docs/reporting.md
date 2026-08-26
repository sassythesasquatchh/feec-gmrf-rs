# Posterior reporting and artifacts

`feec_gmrf::report` carries the compact root API through posterior extraction,
console summaries, validated CSV tables, and VTU field bundles. It does not
decide which quantities are scientifically meaningful: callers still define
truth/reference fields, units, held-out data, metrics, artifact selection, and
acceptance thresholds.

## From a model to a report

A solved `Posterior` can be queried by latent ordering, full FEEC cochain
ordering, a named derived output registered on the model, or an ad-hoc
`LinearMap`:

```rust
use feec_gmrf::prelude::*;

fn summarize(
    posterior: &mut Posterior,
    sensor: LinearMap,
    observations: Vec<f64>,
    observation_variances: Vec<f64>,
) -> Result<PosteriorReport> {
    let labels = (0..observations.len())
        .map(|index| format!("sensor_{index}"))
        .collect();

    PosteriorReportBuilder::new(posterior)
        .field(
            FieldRequest::cochain("state", "Full FEEC state")
                .unit("state units")
                .variance_method(VarianceMethod::Exact),
        )
        .field(
            FieldRequest::derived("flux", "Flux cochain", "d1a_flux_cochain")
                .unit("Wb"),
        )
        .qoi(
            QoiRequest::derived(
                "engineering",
                "Engineering QoIs",
                "engineering_qois",
                vec!["flux_x1".into(), "mean_bx".into()],
            )
            .units(vec!["Wb".into(), "T".into()]),
        )
        .prediction(
            PredictionRequest::mapped(
                "held_out",
                "Held-out sensors",
                sensor,
                labels,
                observations,
                observation_variances,
            )
            .units(vec!["T".into(); 2]),
        )
        .include_factorization_diagnostics(true)
        .build()
}
```

Every artifact ID is separate from its human-readable label. IDs use stable
lowercase ASCII letters, digits, `_`, and `-`; `build()` rejects duplicate IDs,
unknown named outputs, non-finite values, and every label/unit/truth/reference/
baseline dimension mismatch.

`FieldReport` retains the complete `VarianceEstimate`, including estimator,
sample count, batch sizes, batch and relative standard errors, raw negative
count, and raw minimum. Its standard deviations use
`sqrt(max(variance, 0))`; the raw estimator diagnostics are not discarded.
Optional truth and reference vectors produce errors and optional z-scores.
Zero posterior standard deviation produces a missing z-score instead of an
infinite table value. Optional baseline variances produce pointwise variance
reductions when the baseline is positive.

`QoiReport` always obtains a small exact covariance block from the posterior.
The covariance-to-correlation conversion is validated in `gmrf-core`: matrices
must be finite, square, and symmetric; materially negative diagonals are errors;
and zero-variance rows and columns have zero correlation.

`PredictionReport` uses the canonical independent-Gaussian predictive
diagnostics. It records observations, predictive means, raw latent variance
metadata, observation variances, residuals, predictive standard deviations,
standardized residuals, RMSE, mean NLPD, and empirical coverage. Pass/fail
thresholds remain in the calling workflow.

## Console and canonical CSV output

```rust
let options = ConsoleReportOptions {
    precision: 6,
    max_rows: 12,
    include_covariance: false,
    include_correlation: true,
};
write_console_report(&mut std::io::stdout(), &report, &options)?;

let tables = report.tables()?;
write_csv_directory("out/my_workflow", &tables)?;
```

Console rendering prints ranges and estimator diagnostics for mesh-sized
fields; it never dumps complete arrays. QoI and prediction rows are bounded by
`max_rows`, and matrix output is opt-in.

Canonical tables use long, stable schemas:

- `metrics.csv`: `id,label,unit,value`;
- `<id>_field.csv` and `<id>_estimator.csv`: pointwise field values and
  estimator metadata;
- `<id>_qoi.csv`, `<id>_covariance.csv`, and `<id>_correlation.csv`;
- `<id>_prediction.csv` and `<id>_prediction_summary.csv`;
- `factorization.csv`, when requested.

Studies can create narrower domain tables with `ReportTable`. Columns are
unique and ordered, every row has the declared width, and cells are typed as
`Text`, `Integer`, `Float`, `Boolean`, or `Missing`. `Float` rejects NaN and
infinity. Use `Missing` for absent data; use explicit `Text("NaN")` only when a
study intentionally records a non-finite scientific result. CSV encoding uses
the `csv` crate, including correct quoting for commas, quotes, tabs, and
newlines.

## VTU bundles

`CochainVtuBuilder` writes named scalar arrays at a declared form degree and can
add a `FieldReport` directly as mean, variance, and standard-deviation arrays:

```rust
let state = report.field("state").expect("requested field");
let mut vtu = CochainVtuBuilder::new(0);
vtu.add_field_report("posterior", state)?;
vtu.write("out/my_workflow/state.vtu", &coords, &topology)?;
```

`TopCellVtuBuilder` combines named scalar and three-vector fields. Flattened
vectors must declare `VectorLayout3::Interleaved` or
`VectorLayout3::ComponentMajor`; conversion, component splitting, dimensions,
finite values, and duplicate names are validated. Actual XML VTU encoding
continues to use the canonical FEEC writers. Physical FEEC reconstruction is
still performed by the established `feg-infer` maps before values reach the
artifact builder.

## Workflows in this release

- `minimal_0form` uses field and prediction requests for sensors, a named
  transect, an ad-hoc query, and exact/Monte Carlo estimator comparison. It
  writes canonical tables plus `sensors.csv`, `transect.csv`, `query.csv`,
  `estimator_comparison.csv`, and `posterior_0form.vtu` under
  `out/examples/minimal_0form`.
- `em_1form_uq` uses report requests for A, D1A, reconstructed B, engineering
  QoIs, and the flux prediction. It writes canonical CSVs and A/D1A/B VTU
  bundles in each model directory under `out/em_1form_uq`.
- Matérn trace normalization and magnetic physical calibration retain their
  typed scientific report rows but use `ReportTable` for their output schemas.

Mixed state/source inference and nonlinear Laplace inference deliberately do
not have automatic report adapters in this release. Those workflows can use
`ReportTable` and the VTU builders now, while typed adapters from
`LinearPdeUqResult` and `NonlinearPosterior` require a separate API design.
