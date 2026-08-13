#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO_TARGET_DIR="$repository_root/target/external-consumer" \
cargo check \
  --release \
  --manifest-path "$repository_root/tests/external-consumer/Cargo.toml"
