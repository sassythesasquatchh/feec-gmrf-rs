#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

missing=0
if ! bash "$repository_root/scripts/check-native-prerequisites.sh"; then
  missing=1
fi

if command -v python3 >/dev/null 2>&1 && ! python3 -c 'import ngsolve' >/dev/null 2>&1; then
  echo "missing optional publication-reference Python NGSolve module" >&2
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  echo "publication prerequisites are incomplete; no study was skipped" >&2
  exit 1
fi

echo "publication prerequisites are available"
