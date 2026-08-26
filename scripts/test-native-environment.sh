#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_script="$repository_root/scripts/petsc-environment.sh"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT

fail() {
  echo "native environment test failed: $*" >&2
  exit 1
}

assert_contains() {
  local text=$1
  local expected=$2
  if [[ "$text" != *"$expected"* ]]; then
    fail "expected output to contain '$expected', found: $text"
  fi
}

make_fake_install() {
  local root=$1
  local arch=${2:-}
  local petsc_include
  if [[ -n "$arch" ]]; then
    petsc_include="$root/petsc/$arch/include"
  else
    petsc_include="$root/petsc/include"
  fi
  mkdir -p "$petsc_include" "$root/slepc/lib/slepc/conf"
  printf '#define PETSC_HAVE_MUMPS 1\n' > "$petsc_include/petscconf.h"
  : > "$root/slepc/lib/slepc/conf/slepc_common"
}

run_environment() {
  env -u PETSC_DIR -u PETSC_ARCH -u SLEPC_DIR \
    FEG_NATIVE_ROOT="$1" \
    PATH="$2" \
    bash -c 'set -e; source "$1"; resolve_petsc_environment; validate_petsc_environment; print_petsc_environment' \
    _ "$environment_script"
}

fake_bin="$temporary_directory/bin"
pkg_root="$temporary_directory/pkg"
mkdir -p "$fake_bin"
make_fake_install "$pkg_root" ""
cat > "$fake_bin/pkg-config" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  --exists)
    exit 0
    ;;
  --variable=prefix)
    case "$2" in
      PETSc|petsc) printf '%s\n' "$FAKE_PKG_ROOT/petsc" ;;
      SLEPc|slepc) printf '%s\n' "$FAKE_PKG_ROOT/slepc" ;;
      *) exit 1 ;;
    esac
    ;;
  --variable=includedir)
    printf '%s\n' "$FAKE_PKG_ROOT/petsc/include"
    ;;
  *)
    exit 1
    ;;
esac
EOF
chmod +x "$fake_bin/pkg-config"
test_path="$fake_bin:/usr/bin:/bin"
export FAKE_PKG_ROOT="$pkg_root"

explicit_root="$temporary_directory/explicit"
local_root="$temporary_directory/local"
make_fake_install "$explicit_root" "explicit-arch"
make_fake_install "$local_root" "arch-feec-mumps-opt"
explicit_output=$(env \
  PETSC_DIR="$explicit_root/petsc" \
  PETSC_ARCH=explicit-arch \
  SLEPC_DIR="$explicit_root/slepc" \
  FEG_NATIVE_ROOT="$local_root" \
  PATH="$test_path" \
  bash -c 'set -e; source "$1"; resolve_petsc_environment; validate_petsc_environment; print_petsc_environment' \
  _ "$environment_script")
assert_contains "$explicit_output" "PETSc/SLEPc selection: explicit environment"
assert_contains "$explicit_output" "PETSC_ARCH=explicit-arch"

local_output=$(run_environment "$local_root" "$test_path")
assert_contains "$local_output" "PETSc/SLEPc selection: repository-local installation"
assert_contains "$local_output" "PETSC_ARCH=arch-feec-mumps-opt"

empty_native="$temporary_directory/empty-native"
mkdir -p "$empty_native"
pkg_output=$(run_environment "$empty_native" "$test_path")
assert_contains "$pkg_output" "PETSc/SLEPc selection: pkg-config"
assert_contains "$pkg_output" "PETSC_DIR=$pkg_root/petsc"

partial_root="$temporary_directory/partial"
mkdir -p "$partial_root/petsc"
set +e
partial_output=$(env -u PETSC_DIR -u PETSC_ARCH -u SLEPC_DIR \
  FEG_NATIVE_ROOT="$partial_root" PATH="$test_path" \
  bash -c 'set -e; source "$1"; resolve_petsc_environment; echo "$PETSC_SELECTION"; validate_petsc_environment' \
  _ "$environment_script" 2>&1)
partial_status=$?
set -e
[[ "$partial_status" -ne 0 ]] || fail "partial local installation was accepted"
assert_contains "$partial_output" "repository-local installation"
assert_contains "$partial_output" "SLEPC_DIR does not exist"

no_mumps_root="$temporary_directory/no-mumps"
make_fake_install "$no_mumps_root" "arch-feec-mumps-opt"
printf '#define PETSC_HAVE_MUMPS 0\n' \
  > "$no_mumps_root/petsc/arch-feec-mumps-opt/include/petscconf.h"
set +e
no_mumps_output=$(run_environment "$no_mumps_root" "$test_path" 2>&1)
no_mumps_status=$?
set -e
[[ "$no_mumps_status" -ne 0 ]] || fail "PETSc without MUMPS was accepted"
assert_contains "$no_mumps_output" "lacks required MUMPS support"

missing_slepc_root="$temporary_directory/missing-slepc-config"
make_fake_install "$missing_slepc_root" "arch-feec-mumps-opt"
rm "$missing_slepc_root/slepc/lib/slepc/conf/slepc_common"
set +e
missing_slepc_output=$(run_environment "$missing_slepc_root" "$test_path" 2>&1)
missing_slepc_status=$?
set -e
[[ "$missing_slepc_status" -ne 0 ]] || fail "SLEPc without make configuration was accepted"
assert_contains "$missing_slepc_output" "SLEPc make configuration is missing"

echo "native environment selection tests passed"
