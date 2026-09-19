# syntax=docker/dockerfile:1.7
# Build context must be prepared with scripts/prepare-source.py.
# No default image: an A2 image must not be silently selected for original 910.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS build

ARG BUILD_JOBS=8
ARG CANN_ENV=/usr/local/Ascend/cann/set_env.sh
ENV CANN_ENV=${CANN_ENV}
SHELL ["/bin/bash", "-c"]
WORKDIR /build/release

# Deliberate allowlist: never COPY the old application container or all host files.
COPY build-inputs/llama.cpp/ /build/llama.cpp/
COPY patches/ /build/release/patches/
COPY src/ /build/release/src/
COPY tests/ /build/release/tests/
COPY tools/ /build/release/tools/
COPY artifacts/cache-fix/libnnopbase.so /build/release/artifacts/cache-fix/libnnopbase.so
COPY scripts/ /build/release/scripts/
COPY release-manifest.json /build/release/release-manifest.json
COPY licenses/ /build/release/licenses/
COPY THIRD_PARTY.md /build/release/THIRD_PARTY.md
COPY LICENSE /build/release/LICENSE

# This script must verify the source manifest, apply the exact backend patch,
# compile kernels for original 910 and stage binaries under /opt/qwen38-910.
# It must not access NPU hardware during the image build.
RUN test "$(uname -m)" = aarch64 \
 && test -f "$CANN_ENV" \
 && source "$CANN_ENV" \
 && command -v cmake >/dev/null \
 && command -v g++ >/dev/null \
 && command -v python3 >/dev/null \
 && python3 -c 'from tbe import tik' \
 && BUILD_JOBS="$BUILD_JOBS" bash scripts/build-release.sh \
 && test -x /opt/qwen38-910/bin/model-benchmark \
 && test -d /opt/qwen38-910/kernel_meta

# The supplied base retains its SDK/runtime license. This Dockerfile does not
# assert permission to push that base or the resulting image to a public registry.
FROM ${BASE_IMAGE} AS runtime
ARG CANN_ENV=/usr/local/Ascend/cann/set_env.sh
ENV CANN_ENV=${CANN_ENV}
COPY --from=build /opt/qwen38-910/ /opt/qwen38-910/
COPY --from=build /build/release/licenses/ /opt/qwen38-910/licenses/
COPY --from=build /build/release/THIRD_PARTY.md /opt/qwen38-910/THIRD_PARTY.md
COPY --from=build /build/release/LICENSE /opt/qwen38-910/LICENSE
COPY --from=build /build/release/release-manifest.json /opt/qwen38-910/release-manifest.json
COPY scripts/release-entrypoint.sh /usr/local/bin/qwen38-910
RUN chmod 0755 /usr/local/bin/qwen38-910
WORKDIR /opt/qwen38-910
ENTRYPOINT ["/usr/local/bin/qwen38-910"]
CMD ["help"]
LABEL org.opencontainers.image.title="Legacy Ascend Qwen3.8 runtime"
LABEL org.opencontainers.image.version="2026.09.19-experimental"
