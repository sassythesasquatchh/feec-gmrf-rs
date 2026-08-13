# FEEC–GMRF experiments

This default-off crate is the home of exploratory and historical workflows.
Its `examples/` directory preserves the former case-study entrypoints without
making them part of the supported Cargo target surface. Reusable mathematics
must live in `feec`, `gmrf`, or the root `feec-gmrf` API; maintained report
workflows are invoked through `feg-study`.

The library surface is compiled and tested in a separate CI job. An
experimental entrypoint is promoted only after its reusable operations have
moved to the owning lower layer and it has smoke/publication profiles plus a
stable registry entry.
