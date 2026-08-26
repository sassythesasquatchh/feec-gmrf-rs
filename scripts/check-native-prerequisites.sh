#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repository_root/scripts/petsc-environment.sh"

missing=0
required_commands=(git make cmake pkg-config gmsh python3 flex)
case "$(uname -s)" in
  Darwin)
    required_commands+=(clang clang++ gfortran)
    ;;
  Linux)
    required_commands+=(gcc g++ gfortran)
    ;;
  *)
    echo "unsupported native-install platform: $(uname -s)" >&2
    missing=1
    ;;
esac

for command in "${required_commands[@]}"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "missing required native command: $command" >&2
    missing=1
  fi
done

if ! resolve_petsc_environment || ! validate_petsc_environment; then
  missing=1
else
  print_petsc_environment

  if [[ "$PETSC_SELECTION" == "repository-local installation" ]]; then
    for project in petsc slepc; do
      if [[ "$project" == "petsc" ]]; then
        project_directory="$PETSC_DIR"
        expected_tag="v$FEG_PETSC_VERSION"
      else
        project_directory="$SLEPC_DIR"
        expected_tag="v$FEG_SLEPC_VERSION"
      fi
      actual_tag=$(git -C "$project_directory" describe --tags --exact-match HEAD 2>/dev/null || true)
      if [[ "$actual_tag" != "$expected_tag" ]]; then
        echo "repository-local $project must be at $expected_tag, found ${actual_tag:-an untagged checkout}" >&2
        missing=1
      fi
    done
  fi

  petsc_binary_directory="$PETSC_DIR${PETSC_ARCH:+/$PETSC_ARCH}/bin"
  for command in mpicc mpiexec; do
    if ! command -v "$command" >/dev/null 2>&1 \
      && [[ ! -x "$petsc_binary_directory/$command" ]]; then
      echo "missing MPI command supplied by PETSc: $command" >&2
      missing=1
    fi
  done
fi

if [[ "$missing" -ne 0 ]]; then
  echo "native prerequisites are incomplete; no workflow was skipped" >&2
  exit 1
fi

echo "native prerequisites are available"
