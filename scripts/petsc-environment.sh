#!/usr/bin/env bash

# Shared PETSc/SLEPc selection for publication checks and helper builds.
# Explicit environment variables select an in-place or prefix installation;
# pkg-config is used only when no explicit PETSc installation was requested.

select_pkg_config_package() {
  local candidate
  for candidate in "$@"; do
    if pkg-config --exists "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_petsc_environment() {
  PETSC_SELECTION=""
  PETSC_PKG_CONFIG_PACKAGE=""
  SLEPC_PKG_CONFIG_PACKAGE=""

  if [[ -n "${PETSC_DIR:-}" ]]; then
    if [[ -z "${SLEPC_DIR:-}" ]]; then
      echo "SLEPC_DIR must be set when PETSC_DIR selects an explicit installation" >&2
      return 1
    fi
    PETSC_SELECTION="explicit environment"
  else
    if [[ -n "${PETSC_ARCH:-}" || -n "${SLEPC_DIR:-}" ]]; then
      echo "PETSC_ARCH and SLEPC_DIR require an explicit PETSC_DIR" >&2
      return 1
    fi
    if ! command -v pkg-config >/dev/null 2>&1; then
      echo "pkg-config is required when PETSC_DIR and SLEPC_DIR are not set" >&2
      return 1
    fi
    PETSC_PKG_CONFIG_PACKAGE=$(select_pkg_config_package PETSc petsc) || {
      echo "missing PETSc pkg-config metadata" >&2
      return 1
    }
    SLEPC_PKG_CONFIG_PACKAGE=$(select_pkg_config_package SLEPc slepc) || {
      echo "missing SLEPc pkg-config metadata" >&2
      return 1
    }
    PETSC_DIR=$(pkg-config --variable=prefix "$PETSC_PKG_CONFIG_PACKAGE")
    SLEPC_DIR=$(pkg-config --variable=prefix "$SLEPC_PKG_CONFIG_PACKAGE")
    PETSC_SELECTION="pkg-config"
  fi

  export PETSC_DIR SLEPC_DIR
  if [[ -n "${PETSC_ARCH:-}" ]]; then
    export PETSC_ARCH
  fi
}

locate_petsc_configuration_header() {
  local candidate
  local candidates=()

  if [[ -n "${PETSC_ARCH:-}" ]]; then
    candidates+=("$PETSC_DIR/$PETSC_ARCH/include/petscconf.h")
  fi
  candidates+=("$PETSC_DIR/include/petscconf.h")
  if [[ -n "${PETSC_PKG_CONFIG_PACKAGE:-}" ]]; then
    candidates+=("$(pkg-config --variable=includedir "$PETSC_PKG_CONFIG_PACKAGE")/petscconf.h")
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "could not locate petscconf.h for PETSC_DIR=$PETSC_DIR PETSC_ARCH=${PETSC_ARCH:-<prefix install>}" >&2
  return 1
}

validate_petsc_environment() {
  local petsc_configuration_header

  if [[ ! -d "$PETSC_DIR" ]]; then
    echo "PETSC_DIR does not exist: $PETSC_DIR" >&2
    return 1
  fi
  if [[ ! -d "$SLEPC_DIR" ]]; then
    echo "SLEPC_DIR does not exist: $SLEPC_DIR" >&2
    return 1
  fi
  if [[ ! -f "$SLEPC_DIR/lib/slepc/conf/slepc_common" ]]; then
    echo "SLEPc make configuration is missing: $SLEPC_DIR/lib/slepc/conf/slepc_common" >&2
    return 1
  fi

  petsc_configuration_header=$(locate_petsc_configuration_header) || return 1
  if ! grep -Eq '^#define[[:space:]]+PETSC_HAVE_MUMPS[[:space:]]+1' \
    "$petsc_configuration_header"; then
    echo "selected PETSc configuration lacks required MUMPS support: $petsc_configuration_header" >&2
    return 1
  fi
}

add_library_path_entry() {
  local entry=$1
  if [[ -z "$entry" || ! -d "$entry" ]]; then
    return 0
  fi
  case ":${LIBRARY_PATH:-}:" in
    *":$entry:"*) ;;
    *) LIBRARY_PATH="$entry${LIBRARY_PATH:+:$LIBRARY_PATH}" ;;
  esac
}

prepare_compiler_library_path() {
  local gcc_prefix
  local gcc_library_root
  local gcc_runtime_archive

  # PETSc records compiler-library search paths at configure time. Homebrew
  # removes versioned GCC Cellar directories on upgrade, while this stable
  # opt path follows the current GCC installation. Adding it allows an older
  # valid PETSc configuration to relink applications without editing PETSc.
  if command -v brew >/dev/null 2>&1; then
    gcc_prefix=$(brew --prefix gcc 2>/dev/null || true)
    gcc_library_root="$gcc_prefix/lib/gcc/current"
    add_library_path_entry "$gcc_library_root"
    if [[ -d "$gcc_library_root/gcc" ]]; then
      gcc_runtime_archive=$(find "$gcc_library_root/gcc" -name libemutls_w.a -print -quit)
      if [[ -n "$gcc_runtime_archive" ]]; then
        add_library_path_entry "$(dirname "$gcc_runtime_archive")"
      fi
    fi
  fi

  export LIBRARY_PATH
}

print_petsc_environment() {
  echo "PETSc/SLEPc selection: $PETSC_SELECTION"
  echo "PETSC_DIR=$PETSC_DIR"
  if [[ -n "${PETSC_ARCH:-}" ]]; then
    echo "PETSC_ARCH=$PETSC_ARCH"
  fi
  echo "SLEPC_DIR=$SLEPC_DIR"
}
