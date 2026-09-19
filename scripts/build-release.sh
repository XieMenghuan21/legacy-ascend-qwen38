#!/usr/bin/env bash
# Build the pinned and patched runtime without accessing NPU hardware.
set -euo pipefail

project_root=/build/release
source_root=/build/llama.cpp
artifact_root=/opt/qwen38-910
jobs=${BUILD_JOBS:-8}
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]] || (( jobs > 32 )); then
  printf 'BUILD_JOBS must be an integer from 1 to 32\n' >&2
  exit 2
fi
cd "$project_root"

for required in patches/llama.cpp.patch tools/model-benchmark.cpp tests/validate-operators.sh; do
  if [[ ! -f "$required" ]]; then
    printf 'Assemble the release input before building: %s\n' "$required" >&2
    exit 1
  fi
done
python3 scripts/source-tree-sha256.py "$source_root" --manifest release-manifest.json

git -C "$source_root" apply --check "$project_root/patches/llama.cpp.patch"
git -C "$source_root" apply "$project_root/patches/llama.cpp.patch"
if [[ -d src/backend ]]; then
  cp -a src/backend/. "$source_root/ggml/src/ggml-cann/"
fi

cmake -S "$source_root" -B /build/llama-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CANN=ON -DSOC_TYPE=Ascend910B \
  -DCANN_INSTALL_DIR="$ASCEND_TOOLKIT_HOME" -DUSE_ACL_GRAPH=OFF \
  -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF
cmake --build /build/llama-build --target llama ggml -j "$jobs"
mkdir -p "$artifact_root/bin" "$artifact_root/lib" "$artifact_root/kernel_meta"
cp -a /build/llama-build/bin/*.so* "$artifact_root/lib/"

g++ -O3 -std=c++17 tools/model-benchmark.cpp \
  -I "$source_root/include" -I "$source_root/ggml/include" \
  -L "$artifact_root/lib" -Wl,-rpath,'$ORIGIN/../lib' \
  -lllama -lggml -lggml-base -lggml-cpu -ldl -pthread \
  -o "$artifact_root/bin/model-benchmark"
for source_file in tests/test_*.cpp; do
  [[ -f "$source_file" ]] || continue
  binary_name=$(basename "$source_file" .cpp)
  g++ -O3 -std=c++17 "$source_file" \
    -I "$source_root/include" -I "$source_root/ggml/include" \
    -L "$artifact_root/lib" -Wl,-rpath,'$ORIGIN/../lib' \
    -lllama -lggml -lggml-base -lggml-cpu -ldl -pthread \
    -o "$artifact_root/bin/$binary_name"
done
cp tests/validate-operators.sh "$artifact_root/bin/validate-operators"
chmod 0755 "$artifact_root/bin/validate-operators"
cp -a tests "$artifact_root/tests"

mkdir -p /build/kernels
cd /build/kernels
GDN_HEADS=32 python3 "$project_root/src/kernels/build_gdn.py"
GDN_HEADS=48 python3 "$project_root/src/kernels/build_gdn.py"
GDN_QK_HEADS=16 python3 "$project_root/src/kernels/build_gdn_native.py"
GDN_QK_HEADS=48 python3 "$project_root/src/kernels/build_gdn_native.py"
python3 "$project_root/src/kernels/build_ssm_conv_native.py"
python3 "$project_root/src/kernels/build_rms_910.py"
python3 "$project_root/src/kernels/build_rms_910.py" --l2-only
python3 "$project_root/src/kernels/build_setrows_decode.py"
python3 "$project_root/src/kernels/build_rope_decode.py"
cp -a kernel_meta/. "$artifact_root/kernel_meta/"

mkdir -p "$artifact_root/cache-fix"
python3 - <<'CHECK'
import hashlib
from pathlib import Path
p=Path('/build/release/artifacts/cache-fix/libnnopbase.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='344f759c370ab28b08ce38d73aa2e2b464a5ce84a139a7d726704e879570c324', 'Cache-fix binary hash mismatch'
CHECK
cp "$project_root/artifacts/cache-fix/libnnopbase.so" "$artifact_root/cache-fix/"
g++ -O3 -std=c++17 "$project_root/tools/expand_weights.cpp" -I "$source_root/include" -I "$source_root/ggml/include" -L "$artifact_root/lib" -Wl,-rpath,'$ORIGIN/../lib' -lllama -o "$artifact_root/bin/expand-weights"
cp "$project_root/release-manifest.json" "$artifact_root/"
find "$artifact_root" -type f -exec sha256sum {} + > /build/release-artifacts.sha256
