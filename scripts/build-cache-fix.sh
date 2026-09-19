#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Experimental reconstruction recipe; not a verified byte-for-byte rebuild.
# The opbase source/patch and derived library retain CANN COSL 2.0.
set -euo pipefail

if [[ "${1:-}" == --help || $# -lt 3 || $# -gt 4 ]]; then
  printf '%s\n' \
    '用法：CANN_ROOT=/path/to/cann-8.5.0 bash scripts/build-cache-fix.sh OPBASE_GIT_REPO NEW_WORK_DIR NEW_OUTPUT_DIR [JOBS=8]' \
    '仅在新目录构建候选库，不安装、不覆盖 SDK、不运行 NPU；本脚本尚未重新编译验证。'
  [[ "${1:-}" == --help ]] && exit 0
  exit 2
fi

source_repo=$1
work_dir=$2
output_dir=$3
jobs=${4:-8}
base_commit=b27e949de1ce29df366a279adf3976f2660e669c
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
patch_file="$project_dir/patches/cache-pointer-fix.patch"
: "${CANN_ROOT:?请显式设置已验证兼容的 CANN 8.5.0 根目录}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]] || (( jobs > 32 )); then
  printf 'JOBS 必须为 1 到 32 的整数\n' >&2
  exit 2
fi
if [[ "$(uname -m)" != aarch64 ]]; then
  printf '本复现脚本限定在原验证架构 aarch64 上构建\n' >&2
  exit 1
fi
for tool in git cmake c++ python3 tar nm c++filt readelf ldd sha256sum; do
  command -v "$tool" >/dev/null || { printf '缺少工具：%s\n' "$tool" >&2; exit 1; }
done
git -C "$source_repo" cat-file -e "$base_commit^{commit}"
test -f "$patch_file"
test -f "$CANN_ROOT/set_env.sh"
test -f "$CANN_ROOT/lib64/libnnopbase.so"
test -f "$CANN_ROOT/lib64/libascend_protobuf.so.3.13.0.0"

# Refuse reuse or SDK-contained destinations before creating anything.
python3 - "$source_repo" "$work_dir" "$output_dir" "$CANN_ROOT" <<'CHECK_PATHS'
import sys
from pathlib import Path
source, work, output, sdk = map(lambda value: Path(value).resolve(), sys.argv[1:])
for candidate in (work, output):
    if candidate.exists():
        raise SystemExit(f"必须使用尚不存在的新目录：{candidate}")
    if candidate == sdk or sdk in candidate.parents or candidate == source or source in candidate.parents:
        raise SystemExit("构建/输出目录不能位于 SDK 或来源仓库内")
if work == output or work in output.parents or output in work.parents:
    raise SystemExit("构建目录和输出目录须互相独立")
version_files = [sdk / 'version.info', sdk / 'version.cfg', sdk / 'ascend_toolkit_install.info',
                 sdk / 'aarch64-linux/ascend_toolkit_install.info']
version_text = str(sdk) + '\n' + '\n'.join(p.read_text(errors='ignore') for p in version_files if p.is_file())
if '8.5.0' not in version_text:
    raise SystemExit("无法从路径或安装记录确认 CANN 8.5.0，请使用其版本化根目录")
CHECK_PATHS

sdk_hash_before=$(sha256sum "$CANN_ROOT/lib64/libnnopbase.so" | cut -d ' ' -f1)
mkdir -p "$work_dir/source" "$output_dir"
work_dir=$(cd -- "$work_dir" && pwd)
output_dir=$(cd -- "$output_dir" && pwd)
git -C "$source_repo" archive "$base_commit" | tar -xf - -C "$work_dir/source"
git -C "$work_dir/source" apply --check "$patch_file"
git -C "$work_dir/source" apply "$patch_file"

# Preserve the original build's private-source adjustments; no SDK files change.
python3 - "$work_dir/source" <<'PATCH_PROTOBUF'
import sys
from pathlib import Path
root = Path(sys.argv[1])
cmake = root / 'cmake/third_party/protobuf.cmake'
text = cmake.read_text()
if text.count('patch -p1') != 4:
    raise SystemExit('protobuf.cmake 与锁定版本预期不符，停止而非猜测修改')
cmake.write_text(text.replace('patch -p1', 'git apply -p1'))
for name in ('protobuf_25.1_change_version.patch', 'protobuf-hide_absl_symbols.patch'):
    path = root / 'cmake/third_party' / name
    data = path.read_bytes()
    path.write_bytes(data if data.endswith(b'\n') else data + b'\n')
PATCH_PROTOBUF

set +u
source "$CANN_ROOT/set_env.sh"
set -u
export ASCEND_HOME_PATH="$CANN_ROOT"
unset LD_PRELOAD
export LD_LIBRARY_PATH="$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
nm -D --defined-only "$CANN_ROOT/lib64/libascend_protobuf.so.3.13.0.0" | c++filt > "$work_dir/protobuf-symbols.txt"
python3 - "$work_dir/protobuf-symbols.txt" <<'CHECK_ABI'
import sys
from pathlib import Path
symbols = Path(sys.argv[1]).read_text()
lines = [line for line in symbols.splitlines() if 'CopyToEncodedBuffer' in line]
if not any('absl::lts_ascend_private::string_view' in line for line in lines):
    raise SystemExit('SDK protobuf 未显示预期 absl::string_view ABI；不要直接套用此脚本')
CHECK_ABI

cmake -S "$work_dir/source" -B "$work_dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
  -DENABLE_UT=OFF -DENABLE_ST=OFF \
  -DCMAKE_INSTALL_PREFIX="$work_dir/private-install" 2>&1 | tee "$work_dir/configure.log"

# Build the private protoc/header dependency before adjusting the installed copy
# of its Abseil options. The CANN-provided protobuf shared library stays unchanged.
cmake --build "$work_dir/build" --target protobuf_host_build -j "$jobs" 2>&1 | tee "$work_dir/protobuf-build.log"
python3 - "$work_dir/build/protobuf_host/include/absl/base/options.h" <<'FIX_ABI'
import re, sys
from pathlib import Path
path = Path(sys.argv[1])
text, count = re.subn(r'(?m)^#define ABSL_OPTION_USE_STD_STRING_VIEW [012]$',
                      '#define ABSL_OPTION_USE_STD_STRING_VIEW 0', path.read_text())
if count != 1:
    raise SystemExit('无法唯一定位 Abseil string_view 选项，停止构建')
path.write_text(text)
FIX_ABI
cmake --build "$work_dir/build" --target nnopbase -j "$jobs" 2>&1 | tee "$work_dir/nnopbase-build.log"

built_library="$work_dir/build/src/nnopbase/libnnopbase.so"
built_tls="$work_dir/build/src/nnopbase/tls_guardian/libdummy_tls.so"
ldd -r "$built_library" > "$work_dir/ldd-rebuild.txt" 2>&1
python3 - "$work_dir/ldd-rebuild.txt" <<'CHECK_LINK'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
if 'not found' in text or 'undefined symbol' in text:
    raise SystemExit('候选库存在未解析依赖，检查 ldd-rebuild.txt；不能作为运行库')
CHECK_LINK
readelf -d "$built_library" > "$work_dir/readelf-dynamic.txt"
sdk_hash_after=$(sha256sum "$CANN_ROOT/lib64/libnnopbase.so" | cut -d ' ' -f1)
if [[ "$sdk_hash_before" != "$sdk_hash_after" ]]; then
  printf 'SDK 原库摘要发生变化，请停止使用候选结果并调查\n' >&2
  exit 1
fi
cp "$built_library" "$built_tls" "$output_dir/"
cp "$work_dir/ldd-rebuild.txt" "$work_dir/readelf-dynamic.txt" "$output_dir/"
sha256sum "$output_dir/libnnopbase.so" "$output_dir/libdummy_tls.so" > "$output_dir/SHA256SUMS"
printf '%s\n' \
  '候选库构建和动态链接检查完成；尚未执行 NPU/整模型验证。' \
  '它不是默认公开固定 hash 库。不要覆盖 artifacts/cache-fix 或 SDK。' \
  '该候选库可能带构建目录 RPATH；公开前需清理、重新校验摘要并做独立运行测试。'
