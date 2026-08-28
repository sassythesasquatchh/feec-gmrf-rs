# Posterior reporting

The `feec_gmrf::report` module turns posterior queries into validated,
structured results. It supports console summaries, CSV tables, predictive
diagnostics, exact covariance blocks for small quantities of interest, and VTU
fields for FEEC meshes.

The reporting types do not decide which quantities are scientifically
meaningful. Applications supply names, units, truth or reference values,
physical maps, held-out measurements, and acceptance criteria.

## Requesting posterior fields

A `PosteriorReportBuilder` can query:

- latent active coefficients;
- the complete FEEC cochain, including prescribed boundary values;
- a derived quantity registered while building the model;
- an ad hoc `LinearMap`.

Each request selects an uncertainty method independently:

```rust
use feec_gmrf::prelude::*;

fn summarize(
    posterior: &mut Posterior,
    held_out_map: LinearMap,
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
            FieldRequest::derived(
                "flux",
                "Magnetic flux cochain",
                "d1a_flux_cochain",
            )
            .unit("Wb"),
        )
        .qoi(
            QoiRequest::derived(
                "engineering",
                "Engineering quantities",
                "engineering_qois",
                vec!["flux_x1".into(), "mean_bx".into()],
            )
            .units(vec!["Wb".into(), "T".into()]),
        )
        .prediction(
            PredictionRequest::mapped(
                "held_out",
                "Held-out sensors",
                held_out_map,
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

Artifact IDs are machine-readable keys and are separate from labels. IDs use
lowercase ASCII letters, digits, `_`, and `-`. Construction fails on duplicate
IDs, unknown derived quantities, non-finite data, or inconsistent dimensions.

## Field uncertainty

`FieldReport` contains the posterior mean, marginal variances, standard
deviations, and the complete `VarianceEstimate`. Stochastic estimates retain:

- estimator family and seed;
- sample or probe count;
- batch sizes;
- batch and relative standard errors;
- the number and minimum of raw negative estimates;
- the variance-floor policy.

Reported standard deviations use `sqrt(max(variance, 0))`, while the raw
diagnostics remain available for judging whether stabilization is material.

Optional truth values produce errors and pointwise z-scores. A coefficient with
zero posterior variance has no finite z-score. Optional baseline variances
produce pointwise variance reductions wherever the baseline is positive.

## Quantities of interest

`QoiRequest` is intended for small output vectors such as integrated fluxes,
circulations, or volume-averaged field components. The report obtains their
exact transformed covariance matrix from the posterior factorization,
including any hard-constraint correction.

Covariance-to-correlation conversion checks that the matrix is finite, square,
and symmetric. A materially negative diagonal is an error. A zero-variance
quantity has zero correlation with every quantity, including itself.

## Predictive diagnostics

`PredictionRequest` compares held-out observations with a Gaussian predictive
distribution. The result contains:

- predictive mean and latent variance;
- observation variance;
- residual and standardized residual;
- predictive standard deviation;
- root-mean-square error;
- mean negative log predictive density;
- empirical interval coverage.

These diagnostics currently assume independent Gaussian held-out errors.
Correlated observation blocks can still be queried as fields or quantities of
interest, but a joint Mahalanobis score is not presently part of
`PredictionReport`.

## Console output and tables

Console summaries are bounded so that a mesh-sized field does not print every
coefficient:

```rust
let options = ConsoleReportOptions {
    precision: 6,
    max_rows: 12,
    include_covariance: false,
    include_correlation: true,
};
write_console_report(&mut std::io::stdout(), &report, &options)?;
```

The standard CSV tables are written with:

```rust
let tables = report.tables()?;
write_csv_directory("out/my_workflow", &tables)?;
```

The table set includes:

- `metrics.csv`;
- pointwise field and estimator tables;
- quantity-of-interest values, covariance, and correlation;
- prediction rows and prediction summaries;
- optional factorization diagnostics.

Applications can construct additional domain-specific tables with
`ReportTable`. Columns are ordered and unique, and each row must have the
declared width. Cells are typed as `Text`, `Integer`, `Float`, `Boolean`, or
`Missing`. A floating-point cell rejects NaN and infinity. CSV escaping is
handled by the `csv` crate.

## FEEC VTU fields

`CochainVtuBuilder` writes scalar arrays associated with one form degree:

```rust
let state = report.field("state").expect("requested field");
let mut vtu = CochainVtuBuilder::new(0);
vtu.add_field_report("posterior", state)?;
vtu.write("out/state.vtu", &coords, &topology)?;
```

Adding a `FieldReport` writes mean, variance, and standard-deviation arrays.
The array length must match the number of simplices for the selected form
degree.

`TopCellVtuBuilder` writes scalar or three-component fields on top-dimensional
cells. A flattened vector must declare `VectorLayout3::Interleaved` or
`VectorLayout3::ComponentMajor`; layout conversion and all dimensions are
validated.

Physical reconstruction occurs before reporting. For example, the magnetic
field values supplied to `TopCellVtuBuilder` come from the explicit
`A -> D1 A -> B` map.

## Current result types

`PosteriorReportBuilder` operates directly on the public `Posterior` returned by
linear Gaussian and reduced linear PDE model builders. The lower-level mixed
state/source result and `NonlinearPosterior` expose their means, variances, and
derived quantities through different structures. Those workflows can write
typed `ReportTable` data and VTU fields directly; there is not yet a single
automatic report constructor for them.

The two introductory examples show the complete reporting path:

- `minimal_0form` reports sensors, a transect, an ad hoc query, exact and Monte
  Carlo uncertainty, and a 0-form VTU field.
- `em_1form_uq` reports the vector potential, flux cochain, reconstructed
  magnetic field, engineering covariance, predictive diagnostics, and
  separate A, D1A, and B VTU files.
