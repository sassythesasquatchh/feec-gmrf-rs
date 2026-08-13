#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <version-tag> <output-directory>" >&2
  exit 2
fi

version="$1"
output_directory="$2"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "version must be a tag such as v0.1.0" >&2
  exit 2
fi

if [[ -n "$(git -C "$repository_root" status --porcelain --ignore-submodules=none)" ]]; then
  echo "release archives require a clean root and recursive submodule checkout" >&2
  exit 1
fi

for repository in "$repository_root" "$repository_root/feec" "$repository_root/gmrf-rs"; do
  if [[ ! -d "$repository/.git" && ! -f "$repository/.git" ]]; then
    echo "missing initialized repository: $repository" >&2
    exit 1
  fi
  actual_tag="$(git -C "$repository" describe --tags --exact-match HEAD 2>/dev/null || true)"
  if [[ "$actual_tag" != "$version" ]]; then
    echo "$repository is at tag '$actual_tag', expected '$version'" >&2
    exit 1
  fi
done

mkdir -p "$output_directory"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT
archive_root="$temporary_directory/feec-gmrf-${version#v}"
mkdir -p "$archive_root"

git -C "$repository_root" archive HEAD | tar -xf - -C "$archive_root"
rm -rf "$archive_root/feec" "$archive_root/gmrf-rs"
mkdir -p "$archive_root/feec" "$archive_root/gmrf-rs"
git -C "$repository_root/feec" archive HEAD | tar -xf - -C "$archive_root/feec"
git -C "$repository_root/gmrf-rs" archive HEAD | tar -xf - -C "$archive_root/gmrf-rs"

forbidden_paths=(
  "AGENTS.md"
  "thesis-plan"
  "python"
  "petsc-solver/in"
  "docs/masters-thesis-contributions-summary.tex"
  "docs/thesis-work-codebase-map.md"
  "scripts/thesis_experiment_sweep.py"
  "scripts/render_report_figures.py"
  "scripts/render_torus_pde_weight_convergence.py"
  "feec/.envrc"
  "feec/.vscode"
  "feec/flake.lock"
  "feec/flake.nix"
  "feec/out"
  "feec/plot"
)
for relative_path in "${forbidden_paths[@]}"; do
  if [[ -e "$archive_root/$relative_path" ]]; then
    echo "forbidden release path: $relative_path" >&2
    exit 1
  fi
done

if find "$archive_root" -name AGENTS.md -print | grep -q .; then
  echo "release archive contains agent-only maintenance instructions" >&2
  find "$archive_root" -name AGENTS.md -print >&2
  exit 1
fi

workstation_user="patrick"
tool_cache_name="codex"
if grep -R -I -n -E "(/Users/${workstation_user}|\\.cache/${tool_cache_name})" "$archive_root"; then
  echo "release archive contains a workstation-specific path" >&2
  exit 1
fi

documentation_target="$temporary_directory/documentation-target"
CARGO_TARGET_DIR="$documentation_target" RUSTDOCFLAGS="-D warnings" cargo doc \
  --release \
  --workspace \
  --exclude feg-experiments \
  --no-deps \
  --manifest-path "$repository_root/Cargo.toml"
mkdir -p "$archive_root/generated-documentation"
cp -R "$documentation_target/doc/." "$archive_root/generated-documentation/"

documentation_path="$output_directory/feec-gmrf-${version#v}-documentation.tar.gz"
tar -czf "$documentation_path" -C "$documentation_target" doc

license_inventory="$output_directory/feec-gmrf-${version#v}-license-inventory.txt"
root_metadata="$temporary_directory/root-metadata.json"
feec_metadata="$temporary_directory/feec-metadata.json"
gmrf_metadata="$temporary_directory/gmrf-metadata.json"
cargo metadata --locked --format-version 1 --manifest-path "$repository_root/Cargo.toml" > "$root_metadata"
cargo metadata --locked --format-version 1 --manifest-path "$repository_root/feec/Cargo.toml" > "$feec_metadata"
cargo metadata --locked --format-version 1 --manifest-path "$repository_root/gmrf-rs/Cargo.toml" > "$gmrf_metadata"
python3 "$repository_root/scripts/generate-license-inventory.py" \
  "$archive_root" \
  "$license_inventory" \
  "$root_metadata" \
  "$feec_metadata" \
  "$gmrf_metadata"

{
  echo "root $(git -C "$repository_root" rev-parse HEAD)"
  echo "feec $(git -C "$repository_root/feec" rev-parse HEAD)"
  echo "gmrf-rs $(git -C "$repository_root/gmrf-rs" rev-parse HEAD)"
} > "$archive_root/REPOSITORY-SHAS"

archive_path="$output_directory/feec-gmrf-${version#v}-source-with-submodules.tar.gz"
tar -czf "$archive_path" -C "$temporary_directory" "$(basename "$archive_root")"
shasum -a 256 "$archive_path" > "$archive_path.sha256"
shasum -a 256 "$documentation_path" > "$documentation_path.sha256"
shasum -a 256 "$license_inventory" > "$license_inventory.sha256"

echo "$archive_path"
echo "$archive_path.sha256"
echo "$documentation_path"
echo "$documentation_path.sha256"
echo "$license_inventory"
echo "$license_inventory.sha256"
