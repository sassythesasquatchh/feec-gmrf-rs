# Third-party code and lineage

## formoniq / FEEC submodule

The `feec` submodule retains the `formoniq` crate name and is an attributed
thesis fork of Luis Wirth's implementation. Its recorded upstream baseline is
commit `65b98b55f3fee1c28bc37acb68981a3f3bd63e9e` (the “bsc-thesis version”).
Upstream and fork code are available under MIT or Apache-2.0; both license texts
are included in the submodule.

## GMRF submodule

The `gmrf-rs` submodule contains the standalone `gmrf-core` crate from the
clean-history `gmrf-rs` repository. Its `UPSTREAM.md` records the exact import
commit from the earlier Rust port, whose recorded divergence point is Tim
Weiland's commit `6cfba1e6ab8a71b49eabd5a0545e7eb72eae940a`. It is distributed
under MIT and retains upstream attribution and the original license text.

## Rust dependencies

Rust dependency names, versions, declared license expressions, sources, and
repositories are generated from all three locked workspaces with `cargo
metadata`. The generated inventory belongs in the versioned release archive,
not in source control, because exact transitive versions are fixed by the three
lockfiles.
