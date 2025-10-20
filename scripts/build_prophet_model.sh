#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmdstan_dir="${CMDSTAN_DIR:-}"

if [[ -z "${cmdstan_dir}" ]]; then
  echo "error: CMDSTAN_DIR environment variable is not set" >&2
  echo "Set CMDSTAN_DIR to your CmdStan installation directory and rerun." >&2
  exit 1
fi

if [[ ! -d "${cmdstan_dir}" ]]; then
  echo "error: CMDSTAN_DIR='${cmdstan_dir}' is not a directory" >&2
  exit 1
fi

model_target="${repo_root}/stan/prophet_model"
model_source="${repo_root}/stan/prophet.stan"
temp_stan="${model_target}.stan"
created_temp=false

if [[ ! -f "${model_source}" ]]; then
  echo "error: Stan source not found at '${model_source}'" >&2
  exit 1
fi

if [[ ! -f "${temp_stan}" ]]; then
  created_temp=true
  ln -sf "$(basename "${model_source}")" "${temp_stan}"
fi

cleanup() {
  if [[ "${created_temp}" == true ]]; then
    rm -f "${temp_stan}"
  fi
}
trap cleanup EXIT

printf 'Building prophet_model with CmdStan at %s\n' "${cmdstan_dir}"
make -C "${cmdstan_dir}" \
  STAN_THREADS=true \
  STAN_PROFILE=true \
  CXXFLAGS+=-O3 \
  "${model_target}"

tbb_lib_dir="${cmdstan_dir}/stan/lib/stan_math/lib/tbb"
if [[ -d "${tbb_lib_dir}" ]]; then
  cp -P "${tbb_lib_dir}"/libtbb*.so* "${repo_root}/stan/" 2>/dev/null || true
  cp -P "${tbb_lib_dir}"/libtbbmalloc*.so* "${repo_root}/stan/" 2>/dev/null || true
fi

echo "Build complete -> ${model_target}"
