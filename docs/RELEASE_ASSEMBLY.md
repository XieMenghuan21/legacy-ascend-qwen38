# GitHub 发布与复现清单

本目录是已完成设备验收的实验项目源码包。无需上传模型、SDK、SSH 信息或原业务容器；`src/kernels` 保存原始算子生成器，`patches/llama.cpp.patch` 保存完整后端接入代码。

1. 将本目录作为仓库根目录，检查 `.gitignore`。其中单独允许公开的缓存修复库必须一起提交；体积约 4.6 MiB，不需要把模型放入 Git LFS。
2. 按 `docs/CONTAINER.md` 取得适配基础镜像并执行 `scripts/prepare-source.py`。源树严格校验，`build-inputs/` 不提交。
3. 构建镜像并运行 `validate-operators`、`benchmark --regression` 和固定 64-token 测试。若更换驱动、SDK、硬件或源码版本，重新验收，不能自动套用本次成绩。
4. `benchmarks/` 包含脱敏日志、逐请求结果和汇总。性能使用模型 decode 计时；同步 profiling 仅用于定位瓶颈。
5. 保留根 MIT、llama.cpp MIT、CANN COSL 及第三方许可。缓存修复库和补丁不改标 MIT。库来源与重建边界见 `docs/CACHE_FIX.md`。
6. 源码仓库为 [https://github.com/XieMenghuan21/legacy-ascend-qwen38](https://github.com/XieMenghuan21/legacy-ascend-qwen38)，实验版标签为 `v2026.09.19`。发布资产包含源码 ZIP 与 SHA256；清理后的完整 SDK 镜像另以 `v2026.09.19-public-image` Release 分卷发布；未推送公共容器 registry。后续版本仍应执行上述验证。

`release-manifest.json` 记录来源、源树/补丁/模型哈希；`SHA256SUMS` 覆盖项目交付文件。最终镜像验证文件在镜像构建完成后生成，因此无需强行认为镜像中提前复制的 manifest 已包含它自身 ID；外部验收 JSON 是最终镜像身份依据。

## 发布范围

- 公开：原创算子、完整后端补丁、测试程序、CPU oracle、构建/运行脚本、脱敏证据。
- 独立取得：上游 llama.cpp、CANN SDK 基础镜像、主机驱动、模型权重。
- 镜像发布：清理历史运行信息后的单层 Docker save 归档，按 Release 附件大小限制分卷；详见 PUBLIC_IMAGE.md。

README 应保留“实验版本、6 卡、ctx=256、并发=1、Q4 展开存储、存在 CPU 回退”等边界。本次结果不是通用高并发推理框架或完整模型能力评测。
