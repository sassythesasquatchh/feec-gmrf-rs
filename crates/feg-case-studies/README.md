# FEEC–GMRF case studies

This crate contains maintained numerical studies for Gaussian fields on
differential-form spaces. The studies exercise convergence, topology,
physical calibration, electromagnetic inverse problems, and uncertainty
quantification at reproducible parameter settings.

## Scientific groups

The `publication` module groups the maintained studies into:

- **Matérn and FEEC validation**: scalar kernels, mass-inverse effects,
  functional convergence, trace normalization, and form-degree dependence;
- **Hodge and topology**: exact, coexact, and harmonic branch observables on
  spheres, tori, multiply connected planar domains, and genus-two meshes;
- **electromagnetic uncertainty**: magnetic-field calibration, prior
  comparison, annular H-formulations, TEAM benchmarks, and toroidal
  state/source recovery.

Study modules define the geometry, physical parameters, measurement design,
reference values, and scientific artifacts. FEEC assembly is provided by
`formoniq`; Gaussian conditioning and uncertainty calculations are provided by
`gmrf-core` and `feg-infer`.

Nonlinear magnetostatic and eddy-current investigations are available among
the exploratory programs.

## Running studies

The registry-driven command lists the available studies and profiles:

```text
cargo run --release -p feg-cli --bin feg-study -- list
cargo run --release -p feg-cli --bin feg-study -- describe STUDY_ID
cargo run --release -p feg-cli --bin feg-study -- \
  run STUDY_ID --profile smoke --output out/STUDY_ID
```

`smoke` profiles are small deterministic calculations. `thesis-submitted`
profiles retain the numerical settings associated with the submitted thesis
and may take much longer. A study descriptor reports whether Gmsh, PETSc,
SLEPc, NGSolve, or an external fixture is required.

## Tests

The portable suite is:

```text
cargo test -p feg-case-studies --release
```

Generated-mesh, nonlinear, large-output, and full-workflow checks are enabled
separately:

```text
cargo test -p feg-case-studies --release \
  --features heavy-tests --lib -- --test-threads=1
cargo test -p feg-case-studies --release \
  --features heavy-tests --test heavy_team13 -- --test-threads=1
cargo test -p feg-case-studies --release \
  --features heavy-tests --test heavy_toroidal -- --test-threads=1
cargo test -p feg-case-studies --release \
  --features heavy-tests --test heavy_planar_holes -- --test-threads=1
cargo test -p feg-case-studies --release \
  --features heavy-tests --test heavy_magnetic_uq -- --test-threads=1
```

Comparisons that require external software or reference data use:

```text
cargo test -p feg-case-studies --release \
  --features external-reference-tests --test external_references \
  -- --test-threads=1
```

Exploratory programs that are not part of the maintained study registry are in
`feg-experiments`.
