#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_DEV_HUB_REPO=${VLLM_HUST_DEV_HUB_REPO:-$WORKSPACE_ROOT/vllm-hust-dev-hub}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT/vllm-hust}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
VLLM_HUST_CONDA_ENV=${VLLM_HUST_CONDA_ENV:-vllm-hust-dev}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

find_conda_for_install_only() {
  local candidate
  local candidates=(
    "${CONDA_EXE:-}"
    "$(command -v conda 2>/dev/null || true)"
    "${CI_HOME:-}/miniconda3/bin/conda"
    "${CI_HOME:-}/anaconda3/bin/conda"
    "${CI_HOME:-}/mambaforge/bin/conda"
    "${CI_HOME:-}/miniforge3/bin/conda"
    "${HOME:-}/miniconda3/bin/conda"
    "${HOME:-}/anaconda3/bin/conda"
    "${HOME:-}/mambaforge/bin/conda"
    "${HOME:-}/miniforge3/bin/conda"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]] && (unset PYTHONPATH; "$candidate" info --base >/dev/null 2>&1); then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

ensure_conda_for_install_only() {
  local conda_bin
  local conda_prefix

  if conda_bin="$(find_conda_for_install_only)"; then
    export CONDA_EXE="$conda_bin"
    export PATH="$(dirname "$conda_bin"):$PATH"
    echo "Using conda for install-only flow: $conda_bin"
    return 0
  fi

  conda_prefix="${CI_HOME:-${HOME:-/tmp}}/miniconda3"
  echo "Installing Miniconda for install-only flow: $conda_prefix"
  HOME="${CI_HOME:-${HOME:-/tmp}}" bash "$VLLM_HUST_DEV_HUB_REPO/scripts/install-miniconda.sh" --prefix "$conda_prefix" --yes

  conda_bin="$conda_prefix/bin/conda"
  if [[ ! -x "$conda_bin" ]] || ! (unset PYTHONPATH; "$conda_bin" info --base >/dev/null 2>&1); then
    echo "Miniconda installation did not produce a usable conda binary: $conda_bin" >&2
    return 2
  fi

  export CONDA_EXE="$conda_bin"
  export PATH="$(dirname "$conda_bin"):$PATH"
  echo "Using newly installed conda for install-only flow: $conda_bin"
}

ensure_conda_env_for_install_only() {
  local conda_bin
  local resolved_prefix

  ensure_conda_for_install_only
  conda_bin="${CONDA_EXE:-}"
  if [[ -z "$conda_bin" ]]; then
    conda_bin="$(command -v conda 2>/dev/null || true)"
  fi
  if [[ -z "$conda_bin" ]]; then
    echo "Could not resolve conda binary for install-only flow." >&2
    return 2
  fi

  resolved_prefix="$( (unset PYTHONPATH; "$conda_bin" env list) 2>/dev/null | awk -v env_name="${VLLM_HUST_CONDA_ENV}" '$1 == env_name {print $NF; exit}')"
  if [[ -z "$resolved_prefix" || ! -x "${resolved_prefix}/bin/python" ]]; then
    echo "Creating minimal conda env '$VLLM_HUST_CONDA_ENV' for install-only flow."
    if ! (unset PYTHONPATH; "$conda_bin" create -y -n "$VLLM_HUST_CONDA_ENV" \
      --override-channels \
      -c "https://repo.huaweicloud.com/ascend/repos/conda/" \
      -c "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/" \
      "python=$PYTHON_VERSION" pip); then
      (unset PYTHONPATH; "$conda_bin" create -y -n "$VLLM_HUST_CONDA_ENV" \
        --override-channels \
        -c "https://repo.huaweicloud.com/ascend/repos/conda/" \
        -c "conda-forge" \
        "python=$PYTHON_VERSION" pip)
    fi
    resolved_prefix="$( (unset PYTHONPATH; "$conda_bin" env list) 2>/dev/null | awk -v env_name="${VLLM_HUST_CONDA_ENV}" '$1 == env_name {print $NF; exit}')"
    if [[ -z "$resolved_prefix" || ! -x "${resolved_prefix}/bin/python" ]]; then
      echo "Failed to create conda env '$VLLM_HUST_CONDA_ENV' for install-only flow." >&2
      return 2
    fi
  else
    echo "Using existing conda env for install-only flow: $resolved_prefix"
  fi

  export CONDA_PREFIX="$resolved_prefix"
  export VLLM_HUST_CONDA_PREFIX="$resolved_prefix"
  export CONDA_DEFAULT_ENV="$VLLM_HUST_CONDA_ENV"
  export PATH="${resolved_prefix}/bin:$PATH"

  (unset PYTHONPATH; "$conda_bin" run -n "$VLLM_HUST_CONDA_ENV" python -m pip install --upgrade pip "setuptools>=77,<81" wheel "setuptools-scm>=8" "setuptools-rust")
  echo "Prepared conda env for install-only flow: $resolved_prefix"
}

ensure_conda_ld_library_path_priority() {
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${CONDA_PREFIX}/lib:"*) return 0 ;;
    esac
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    else
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
    fi
    echo "LD_LIBRARY_PATH prioritized for conda runtime libs: ${CONDA_PREFIX}/lib"
  fi
}

run_timed() {
  local label="$1"
  shift
  local start_ts
  local end_ts
  local status

  start_ts=$(date +%s)
  echo "::group::${label}"
  set +e
  "$@"
  status=$?
  set -e
  end_ts=$(date +%s)
  echo "${label} duration: $((end_ts - start_ts))s"
  echo "::endgroup::"
  return "$status"
}

install_vllm_hust_repo() {
  VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 \
    hust_run_pip install -e "$VLLM_HUST_REPO" --no-build-isolation --no-deps
}

if [[ ! -f "$VLLM_HUST_DEV_HUB_REPO/scripts/install-miniconda.sh" ]]; then
  echo "install-miniconda not found: $VLLM_HUST_DEV_HUB_REPO/scripts/install-miniconda.sh" >&2
  exit 2
fi

if [[ ! -f "$VLLM_ASCEND_HUST_REPO/scripts/install_local_ascend_plugin.sh" ]]; then
  echo "install_local_ascend_plugin not found: $VLLM_ASCEND_HUST_REPO/scripts/install_local_ascend_plugin.sh" >&2
  exit 2
fi

if [[ ! -f "$WORKSPACE_ROOT/ascend-runtime-manager/pyproject.toml" ]]; then
  echo "ascend-runtime-manager checkout not found under workspace: $WORKSPACE_ROOT/ascend-runtime-manager" >&2
  exit 2
fi

ensure_conda_env_for_install_only

# shellcheck source=/dev/null
source "$VLLM_ASCEND_HUST_REPO/scripts/hust_ascend_manager_helper.sh"

PYTHON_BIN="$(hust_resolve_python_bin)"
export VLLM_HUST_PYTHON_BIN="$PYTHON_BIN"
export PYTHON_BIN="$PYTHON_BIN"

if [[ -f "$VLLM_ASCEND_HUST_REPO/scripts/use_single_ascend_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$VLLM_ASCEND_HUST_REPO/scripts/use_single_ascend_env.sh"
fi

ensure_conda_ld_library_path_priority

export PYTHONPATH="$VLLM_HUST_REPO:$VLLM_HUST_BENCHMARK_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Using install-only workspace bootstrap:"
echo "  VLLM_HUST_PYTHON_BIN=$VLLM_HUST_PYTHON_BIN"
echo "  PYTHONPATH=$PYTHONPATH"

run_timed "install vllm-hust repo" \
  install_vllm_hust_repo

run_timed "install benchmark runtime Python deps" \
  hust_run_pip install jsonschema

run_timed "install vllm-hust-benchmark repo" \
  hust_run_pip install -e "$VLLM_HUST_BENCHMARK_REPO" --no-build-isolation --no-deps

if [[ "${PUBLISH_TO_HF:-0}" == "1" ]]; then
  run_timed "install huggingface_hub for HF publish" \
    hust_run_pip install "huggingface_hub>=0.20"
fi

run_timed "install local Ascend plugin" \
  env COMPILE_CUSTOM_KERNELS=0 \
  bash "$VLLM_ASCEND_HUST_REPO/scripts/install_local_ascend_plugin.sh" "$VLLM_ASCEND_HUST_REPO"
