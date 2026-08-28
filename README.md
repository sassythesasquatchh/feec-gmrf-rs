# FEEC–GMRF

`feec-gmrf` is a Rust library for Gaussian random fields represented by finite
element differential forms. It combines finite element exterior calculus
(FEEC) with sparse Gaussian Markov random field (GMRF) inference, so that the
geometry, topology, boundary conditions, and differential structure of a field
remain explicit throughout prior construction, conditioning, and uncertainty
quantification.

The library supports scalar fields, vector and flux fields, Hodge-decomposed
models, probabilistic PDE formulations, nonlinear Laplace approximations, and
spatiotemporal models. The same sparse operators used to state the model are
also used to derive physical quantities and their posterior uncertainty.

## Why combine FEEC and GMRFs?

A differential \(k\)-form is represented discretely by a \(k\)-cochain: one
coefficient for each degree-\(k\) simplex or finite element degree of freedom.
FEEC supplies compatible discrete spaces and exterior derivatives

```text
C^0 --D0--> C^1 --D1--> C^2 --D2--> C^3,
```

with \(D_{k+1}D_k=0\). Mass matrices \(M_k\) encode the metric and the finite
element inner product. Together with boundary conditions, these operators
produce weak Hodge–Laplacians and the discrete analogues of grad, curl, and
div.

A GMRF is specified by a sparse precision matrix \(Q\), rather than a dense
covariance matrix. Local finite element operators naturally lead to sparse
precisions. Sparse factorization and covariance actions then make it possible
to condition fields, impose exact constraints, draw samples, and quantify
uncertainty without constructing a dense field covariance.

This pairing is useful when a statistical model must respect curved and non-trivial geometries, and when the field is best modelled without reference to a pariticular coordinate system. It is therefore very well suited to uncertainty quantification in electromagnetic simulation. 

## Mathematical model

### Matérn fields on finite element forms

For form degree \(k\), let \(M_k\) be the FEEC mass matrix and \(L_k\) the weak
Hodge–Laplacian. Define

```text
A_k = kappa^2 M_k + L_k.
```

For the supported integer orders, the discrete precision recurrence is

```text
Q_1 = tau^2 A_k
Q_2 = tau^2 A_k M_k^-1 A_k
Q_3 = tau^2 A_k M_k^-1 A_k M_k^-1 A_k.
```

`MaternPriorBuilder` assembles these models for 0-, 1-, 2-, and 3-form
spaces, with explicit policies for the mass inverse and essential boundary
conditions. The parameter `kappa` controls correlation length, `tau` controls
precision amplitude, and `alpha` controls spectral decay and therefore smoothness of the random field.

For k-forms (where 0 < k < n), a Hodge-decomposed model can place separate priors on exact,
coexact, and harmonic components. The two non-harmonic constructions differ in
where the requested spectrum is defined:

- a **potential-spectrum** prior gives the latent k-1 or k+1-form potential a
  Matérn spectrum; applying \(d\) or \(\delta\) contributes an additional
  eigenvalue factor to the synthesized k-form covariance;
- a **form-spectrum** prior compensates for that factor so the resulting
  exact or coexact 1-form has the requested Matérn spectrum.

A **form-spectrum** Hodge-decomposed model is spectrally equivalent to a non-decomposed model defined on the k-form space, whereas
a **potential-spectrum** model is not.

### Observations, constraints, and posterior precision

A linear observation has the form

```text
y = H x + b + epsilon,       epsilon ~ N(0, R).
```

For a prior \(x\sim N(\mu,Q^{-1})\), conditioning adds

```text
Q_posterior = Q + H^T R^-1 H
eta_posterior = Q mu + H^T R^-1 (y - b).
```

The observation map may select coefficients, integrate a field, compose FEEC
operators, or reconstruct a physical quantity. Multiple observations can have
independent, heteroscedastic, or correlated Gaussian noise. Exact linear
constraints \(Cx=d\) are imposed through constrained sparse solves, including
their covariance correction.

Essential boundary values use the affine representation \(x=Pz+g\). Priors,
observations, constraints, nonlinear residuals, and derived quantities are
pulled back to the active coefficients \(z\), and posterior results are lifted
to the complete cochain ordering with zero variance at prescribed
coefficients.

### PDE information

The library uses PDE operators in two distinct ways:

- **Weak-residual conditioning** treats an assembled residual
  \(r(x)=Ax+c\) as a noisy observation of zero. Its contribution to the
  posterior precision is \(A^TWA\), and it can be combined with a different
  prior and with physical measurements.
- **PDE-induced prior construction** defines the prior itself by
  \(Q=A^TWA\). This is appropriate when the residual norm expresses the desired
  Gaussian regularity.

`LinearPdeSystem::matern_prior_builder` is a convenience for the special case
where the reduced PDE operator is a self-adjoint, non-negative elliptic
generator compatible with the supplied state mass matrix. It interprets that
operator as \(L_k\) in the Matérn recurrence. General affine PDE systems should
use `pde_induced_prior` or an independently constructed prior plus a weak
residual term.


### Physical fields and quantities of interest

Physical outputs are explicit linear maps. In three-dimensional
electromagnetism, magnetic flux density is constructed as

```text
vector-potential 1-cochain --D1--> flux 2-cochain
    --Whitney reconstruction / vector proxy--> physical B.
```

The same chain is used for prior calibration, observations, posterior means,
variances, samples, and reported engineering quantities. This keeps orientation
and metric choices visible and avoids assigning physical units to an unrelated
coefficient-space norm.

## Supported workflows

| Workflow | Main API |
|---|---|
| Matérn priors on differential forms | `MaternPriorBuilder`, `MaternParameters` |
| Hodge-decomposed 1-form priors | `HodgeOneFormPriorBuilder` |
| Linear Gaussian conditioning | `LinearGaussianModelBuilder` |
| Reduced linear PDE models | `LinearPdeSystem`, `LinearPdeModelBuilder` |
| Nonlinear MAP and Laplace approximation | `NonlinearLaplaceModelBuilder` |
| Spatiotemporal precisions | `SpacetimePriorBuilder` |
| Physical field maps and calibration | `PhysicalMap`, `MagneticFieldMaps3d` |
| Exact and stochastic uncertainty | `VarianceMethod`, `Posterior` |
| Tables, diagnostics, and VTU fields | `PosteriorReportBuilder` |

Mixed state/source models use the lower-level `feg-infer` model-composition
types because their block layouts and source eliminations are problem-specific.
They still use the same `gmrf-core` conditioning, constraint, and uncertainty
algorithms.

## Quick start

The repository uses Git submodules for the FEEC and GMRF libraries:

```text
git clone --recursive https://github.com/sassythesasquatchh/feec-gmrf-rs.git
cd feec-gmrf-rs
cargo run --release --example minimal_0form
cargo run --release --example em_1form_uq
```

The default library and both introductory examples use the in-process sparse
solver and do not require PETSc, SLEPc, or NGSolve.

A linear conditioning problem can be assembled directly from a prior and a
sparse observation map:

```rust
use feec_gmrf::prelude::*;

fn condition(
    prior: GaussianPrior,
    sensor_map: LinearMap,
    observations: Vec<f64>,
) -> Result<Posterior> {
    let noise = GaussianNoise::standard_deviation(0.05)?;
    let observation =
        LinearObservation::new(sensor_map, observations, noise)?;

    LinearGaussianModelBuilder::new(prior)
        .observe(observation)?
        .condition()
}
```

The runnable examples provide the complete FEEC assembly around this model:

- [`minimal_0form`](examples/minimal_0form.rs) constructs and physically
  calibrates a scalar Matérn field on a triangulated square, assimilates three
  noisy measurements, compares exact and Monte Carlo variances, and writes CSV
  and VTU results.
- [`em_1form_uq`](examples/em_1form_uq.rs) constructs a mixed-boundary
  electromagnetic 1-form problem, derives magnetic flux and reconstructed
  \(B\), compares an observation-only Matérn model with weak-PDE-residual
  conditioning, and reports engineering quantities of interest.

## Installation and optional solvers

Rust 1.80 or newer is required. A pinned Git dependency can be used because the
workspace crates are not published to crates.io:

```toml
[dependencies]
feec-gmrf = { git = "https://github.com/sassythesasquatchh/feec-gmrf-rs", tag = "v0.1.0" }
```

The `spectral` feature enables low-rank and eigensolver-based Gaussian-process
models. The `external-solvers` feature enables workflows that call PETSc or
SLEPc helper programs. See [Installation and first run](docs/getting-started.md)
for the optional native solver stack and platform-specific commands.

## Source tree

- `src/` contains the public `feec_gmrf` API.
- `feec/` contains FEEC topology, finite element assembly, boundary
  reduction, PDE residuals, and reconstruction.
- `gmrf-rs/` contains sparse Gaussian precision algebra, conditioning,
  constraints, solves, sampling, and variance estimation.
- `crates/feg-infer/` composes FEEC operators with statistical models.
- `crates/feg-case-studies/` contains maintained scientific studies.
- `crates/feg-experiments/` contains exploratory numerical investigations.
- `crates/feg-cli/` provides the `feg-study` command.

Further reading:

- [Mathematical and software architecture](docs/architecture.md)
- [Posterior reporting](docs/reporting.md)
- [Study reproduction](docs/reproduction.md)
- [Scientific input data](docs/assets.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

Generate the complete API reference with:

```text
cargo doc --release --workspace --exclude feg-experiments --no-deps --open
```

## Scientific references

- Douglas N. Arnold, Richard S. Falk, and Ragnar Winther,
  [“Finite element exterior calculus: from Hodge theory to numerical stability”](https://www.ams.org/bull/2010-47-02/S0273-0979-10-01278-4/S0273-0979-10-01278-4.pdf),
  *Bulletin of the American Mathematical Society* 47 (2010), 281–354.
- Finn Lindgren, Håvard Rue, and Johan Lindström,
  [“An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach”](https://academic.oup.com/jrsssb/article-abstract/73/4/423/7034732?login=false),
  *Journal of the Royal Statistical Society: Series B* 73 (2011), 423–498.
- Håvard Rue and Leonhard Held,
  [*Gaussian Markov Random Fields: Theory and Applications*](https://www.routledge.com/Gaussian-Markov-Random-Fields-Theory-and-Applications/Rue-Held/p/book/9781032477909),
  Chapman & Hall/CRC, 2005.

## Acknowledgements and license

This software accompanies Patrick Dowd's master's thesis. The FEEC
implementation is derived from Luis Wirth's
[`formoniq`](https://github.com/luiswirth/formoniq), and the GMRF work builds
on code by Tim Weiland. Detailed acknowledgements and license terms are in
[THIRD_PARTY.md](THIRD_PARTY.md).

The integration code is distributed under the MIT license. The FEEC submodule
retains its MIT/Apache-2.0 license choice, and the GMRF submodule is distributed
under the MIT license.
