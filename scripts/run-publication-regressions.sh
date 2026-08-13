#!/usr/bin/env bash
set -euo pipefail

output_root=${1:?usage: run-publication-regressions.sh OUTPUT_DIRECTORY}
mkdir -p "$output_root"

study_ids=(
  hodge-laplacian/cube
  hodge-laplacian/torus
  hodge-inverse/sparse
  matern/scalar
  matern/trace-normalization
  matern/marginal-variance-3d
  matern/marginal-variance-4d
  hodge/sphere-observables
  hodge/torus-residual-weight
  magnetic/calibration
  magnetic/prior-mismatch
  annulus/h-formulation
  toroidal-b/canonical
  toroidal-b/source-noise
  toroidal-b/coverage
)

for study_id in "${study_ids[@]}"; do
  study_dir="$output_root/${study_id//\//__}"
  cargo run --release -p feg-cli --bin feg-study -- \
    run "$study_id" --profile thesis-submitted --output "$study_dir"
  cargo run --release -p feg-cli --bin feg-study -- \
    verify "$study_dir" --against thesis-submitted
done
