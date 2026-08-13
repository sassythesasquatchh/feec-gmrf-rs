# FEG Case Studies

This crate contains the maintained thesis workflows used for publication and
report reproducibility. They are available through
`feg_case_studies::publication`.

The registry-driven `feg-study` CLI runs the report workflows. Research
prototypes and earlier entrypoints live in `feg-experiments`.

## Publication workflows

- Chapter 7 validation: cube mass-inverse variance, Matérn trace
  normalization, Matérn functional convergence, sphere branch observables, and
  torus 1-form PDE workflows.
- Chapter 8 electromagnetic UQ: magnetic physical calibration, magnetic prior
  UQ comparison, annular H-formulation, and exact toroidal B/source workflows.

`feg-infer`, `feg-gp`, and `gmrf-core` provide the reusable FEEC/GMRF
operations. The case-study modules handle geometry, configuration, reporting,
and thesis-specific artifact generation.

Studies may use public `feg-infer` and `gmrf-core` APIs for specialist
diagnostics or lower-level control. External applications should use the root
`feec-gmrf` API. Shared operations belong in their lower-layer implementation.

## Experimental workflows

Root-level modules gated by the `experimental` feature are historical or
exploratory and are re-exported through `feg-experiments`. The publication
guarantee covers the workflows listed above.

## Test Tiers

The default test suite is intended to be fast and deterministic:

```sh
rtk cargo test -p feg-case-studies --release
```

Generated-mesh, nonlinear, publication-scale, output-writing, and large
workflow regression tests are opt-in:

```sh
rtk cargo test -p feg-case-studies --release --features heavy-tests --lib -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_team13 -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_toroidal -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_planar_holes -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features heavy-tests --test heavy_magnetic_uq -- --test-threads=1
```

Tests that need external reference tools or large reference artifacts use the
external tier:

```sh
rtk cargo test -p feg-case-studies --release --features external-reference-tests --test external_references -- --test-threads=1
rtk cargo test -p feg-case-studies --release --features external-reference-tests --lib -- --test-threads=1
```

Most heavy assertions remain as feature-gated unit tests beside the workflow
code, where they reuse the private setup helpers. New workflow-scale tests
should use `#[cfg(feature = "heavy-tests")]`;
tests requiring PETSc, NGSolve exports, or other external reference artifacts
should use `#[cfg(feature = "external-reference-tests")]`.
