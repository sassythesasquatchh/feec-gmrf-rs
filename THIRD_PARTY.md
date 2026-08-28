# Acknowledgements and third-party licenses

## FEEC implementation

The `formoniq` finite element code in `feec/` is derived from Luis Wirth's
[formoniq project](https://github.com/luiswirth/formoniq), developed in
connection with his ETH Zürich bachelor's thesis on coordinate-free Whitney
finite element exterior calculus.

The FEEC workspace retains the upstream MIT/Apache-2.0 license choice. Both
license texts are included in that workspace. Patrick Dowd and contributors
developed the boundary-aware assembly, electromagnetic residuals and
Jacobians, spatiotemporal operators, reconstruction maps, and integration
interfaces used by FEEC–GMRF.

## GMRF implementation

The `gmrf-core` code in `gmrf-rs/` builds on Gaussian Markov random field work
by Tim Weiland. Patrick Dowd and contributors developed the Rust integration,
sparse conditioning, constraint, sampling, covariance, and uncertainty
functionality used here.

The GMRF workspace is distributed under the MIT license and includes its
license text.

## Rust dependencies

The exact transitive Rust dependencies are fixed by the lockfiles in the
parent, FEEC, and GMRF workspaces. Their names, versions, sources, and declared
licenses can be obtained with `cargo metadata` or a dependency-license tool.
