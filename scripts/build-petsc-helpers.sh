#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repository_root/scripts/petsc-environment.sh"

resolve_petsc_environment
validate_petsc_environment
prepare_compiler_library_path
print_petsc_environment

solver_directory="$repository_root/feec/petsc-solver"
make -C "$solver_directory" clean
make -C "$solver_directory" all

for helper in ghiep.out ghep_reduced.out hils.out; do
  if [[ ! -x "$solver_directory/$helper" ]]; then
    echo "PETSc/SLEPc helper was not built: $solver_directory/$helper" >&2
    exit 1
  fi
done

echo "PETSc/SLEPc helpers are ready in $solver_directory"
