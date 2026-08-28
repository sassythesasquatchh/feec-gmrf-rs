# FEEC–GMRF experiments

This crate contains exploratory numerical programs for new priors, diagnostics,
observation designs, geometries, and solver strategies. Programs are kept as
Rust examples so that each experiment states its complete configuration and
can be rerun directly.

List the available programs with:

```text
cargo run --release -p feg-experiments --example EXAMPLE_NAME
```

The crate is excluded from the default workspace members because some examples
have large meshes, external-solver requirements, or long runtimes. Compile and
test it explicitly with:

```text
cargo check --release -p feg-experiments --all-targets
cargo test --release -p feg-experiments --all-targets
```

Studies with fixed configurations and run manifests are exposed through
`feg-study` and documented in the case-study crate.
