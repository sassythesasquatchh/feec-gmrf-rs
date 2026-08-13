#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repository_root/scripts/petsc-environment.sh"

missing=0
for command in gmsh make python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "missing required publication command: $command" >&2
    missing=1
  fi
done

if ! resolve_petsc_environment || ! validate_petsc_environment; then
  missing=1
else
  print_petsc_environment
fi

if command -v python3 >/dev/null 2>&1 && ! python3 -c 'import ngsolve' >/dev/null 2>&1; then
  echo "missing required Python NGSolve module" >&2
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  echo "publication prerequisites are incomplete; no study was skipped" >&2
  exit 1
fi

echo "publication prerequisites are available"
