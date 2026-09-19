# Third-party provenance and license boundaries

This file records source provenance. It does not change upstream licenses or
claim redistribution rights for artifacts that are not part of the repository.

| Component | Upstream / revision | License treatment | Intended distribution |
|---|---|---|---|
| llama.cpp / GGML | https://github.com/ggml-org/llama.cpp ; `895c045fd104ced72132160245edcd6a86e50ba0` | MIT; preserve original notices in [license copy](licenses/llama.cpp-MIT.txt) | Targeted patch and, where necessary, derived backend files |
| CANN opbase | https://gitcode.com/cann/opbase ; v8.5.0 / `b27e949de1ce29df366a279adf3976f2660e669c` | [CANN Open Software License Agreement 2.0](licenses/CANN-COSL-2.0.txt) | Minimal patch; independently built library with matching license/notices |
| Cache pointer fix | Upstream commit `f37e88e9525fac93214acd6fc652d9c69e191e02` | Same CANN license; retain author/commit attribution | Backport patch, separately identified from MIT material |
| CANN SDK, TBE and runtime | Independently supplied CANN 8.5.0 development environment | Governed by their respective Huawei terms | **Not bundled by this source release** |
| Host Ascend driver | Driver 25.5.2 in the measured host | Governed by its distribution terms | **Not bundled and not installed by this project** |
| Original TIK kernels and standalone tests | This project's original contributions | MIT; see the root LICENSE, preserving any existing upstream notices | Source files and build scripts; no implied relicensing of upstream material |
| Model weights / tokenizer | Independently obtained Qwen3.8-27B GGUF and recorded conversion | Model and quantization release terms remain applicable | **Not included**; record source and hashes |
| Container base | User-supplied compatible original-910 development image | Base image and bundled software terms apply | Build recipe only by default; do not publish the existing application container |

The CANN license contains processor-scope and notice requirements. Its text is
included in full; do not relabel the `nnopbase` patch or derived binary as MIT.
Likewise, do not apply the CANN license to the entire repository merely because
the original kernels call CANN/TIK APIs.

If the isolated cache library is published as a release artifact, keep the exact
source revision, patch, build configuration and corresponding notices alongside
it. Validate its dynamic dependencies against CANN 8.5.0. It is not a general
cross-version replacement for the installed `libnnopbase.so`.

The source release includes one checksum-pinned, separately licensed cache-fix library. Its original and sanitized hashes, build provenance and dependency notices are recorded in the manifest and docs/CACHE_FIX.md. The kernel/runtime binaries are built from source by the Dockerfile. Any public
container image must be built from an explicitly redistributable base, stripped
of credentials and business data, and verified independently from the old running
container. A successful local build does not by itself establish redistribution
permission for that base.

## Headers and generated code used by the cache backport

- nlohmann/json v3.11.3: [MIT license](licenses/nlohmann-json-MIT.txt).
- Protocol Buffers v25.1: [BSD license](licenses/protobuf-BSD.txt).
- Abseil 20230802.1: [Apache 2.0 license](licenses/abseil-Apache-2.0.txt).

These notices accompany the derived cache library. Its dynamic CANN and protobuf dependencies remain supplied by the compatible SDK image, not by this source archive.

## Public image distribution

The sanitized `2026.09.19-public` image is supplied as multipart GitHub Release assets. It retains the SDK component licenses and notices installed in the base environment; the root MIT license does not relicense those components. Historical runtime logs and user-level configuration were removed before filesystem flattening. Model weights and host drivers are not included. See docs/PUBLIC_IMAGE.md for the exact image identity and archive checksums.
