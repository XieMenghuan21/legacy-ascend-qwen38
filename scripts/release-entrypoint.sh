#!/usr/bin/env bash
set -euo pipefail

release_root=/opt/qwen38-910
command_name=${1:-help}
if [[ "$command_name" == help || "$command_name" == --help ]]; then
  printf '%s\n' \
    'Usage: qwen38-910 validate-operators [arguments]' \
    '       qwen38-910 benchmark --model PATH --devices IDS --context N --output-tokens N' \
    '       qwen38-910 convert-weights INPUT.gguf OUTPUT.gguf' \
    'This is an experimental short-context benchmark image, not an API server.'
  exit 0
fi
shift

case "$command_name" in
  validate-operators) target="$release_root/bin/validate-operators" ;;
  convert-weights) target="$release_root/bin/expand-weights" ;;
  benchmark) target="$release_root/bin/model-benchmark" ;;
  *) printf 'Unsupported command: %s\n' "$command_name" >&2; exit 2 ;;
esac
if [[ ! -x "$target" ]]; then
  printf 'Release artifact is missing: %s\n' "$target" >&2
  exit 1
fi

# Vendor environment scripts may reference unset variables.
set +u
source "${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
set -u
export LD_LIBRARY_PATH="$release_root/lib:${LD_LIBRARY_PATH:-}"
export GGML_CANN_GDN_910_BINARY="$release_root/kernel_meta/gdn_decode_910.o"
export GGML_CANN_GDN_910_BINARY_48="$release_root/kernel_meta/gdn_decode_910_h48.o"
export GGML_CANN_GDN_NATIVE_DIR="$release_root/kernel_meta"
export GGML_CANN_SSM_910_BINARY="$release_root/kernel_meta/ssm_conv_910_t1.o"
export GGML_CANN_RMS_910_DIR="$release_root/kernel_meta"
export GGML_CANN_ROPE_910_COMPOSITE=1
export GGML_CANN_ROPE_910_NATIVE_DIR="$release_root/kernel_meta"
export GGML_CANN_SETROWS_910_DIR="$release_root/kernel_meta"
export GGML_CANN_WEIGHT_NZ=off
unset GGML_CANN_PROFILE_SYNC

# The separately built cache fix is verified only with the documented CANN ABI.
if [[ "${ENABLE_VERIFIED_CACHE_FIX:-1}" == 1 ]]; then
  cache_library="$release_root/cache-fix/libnnopbase.so"
  if [[ ! -r "$cache_library" ]]; then
    printf 'Verified cache fix requested but absent: %s\n' "$cache_library" >&2
    exit 1
  fi
  export LD_PRELOAD="$cache_library${LD_PRELOAD:+:$LD_PRELOAD}"
  export ACLNN_CACHE_LIMIT=10000
else
  export ACLNN_CACHE_LIMIT=0
fi
exec "$target" "$@"
