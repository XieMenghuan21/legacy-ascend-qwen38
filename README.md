# Qwen3.8-27B 在旧款昇腾 910 上的适配与算子优化

[源码仓库](https://github.com/XieMenghuan21/legacy-ascend-qwen38) · [实验版 Release](https://github.com/XieMenghuan21/legacy-ascend-qwen38/releases/tag/v2026.09.19)

在 **6 张旧款 Ascend 910** 上跑通 Qwen3.8-27B，并通过自定义算子和隔离的 CANN 缓存修复提高解码速度。项目包含完整后端补丁、TIK 算子源码、设备测试、Dockerfile、固定版本来源和脱敏性能证据。

**这是已实机验证的短上下文实验版本：ctx=256、单并发、存在 CPU 回退。权重由 Q4 GGUF 展开为 FP16 存储，仍保留 Q4 量化损失。镜像提供 CLI 测试，不是 HTTP API 服务。**

ACL 返回的 SoC 名称是 `Ascend910B`，这里指旧款 910，不应与 Atlas A2 的 `Ascend910B1/B2` 混淆。

## 实测结果

最终数字与测试协议见 [优化报告](docs/OPTIMIZATION_REPORT.md) 和 [机器可读汇总](benchmarks/results.json)。在相同 33-token 提示词、64-token 输出预算下，初始解码速度为 **3.67 token/s**，上一轮为 **4.96 token/s**；本轮加入 RMS、RoPE 和 KV 更新优化，最终达到 **5.67 token/s（累计 +54.4%）**。最终镜像以独立容器完成算子测试、四项模型回归和重复性能测试。

计时是模型 decode 阶段，不含权重加载和提示词处理。它不是并发吞吐量，也不是完整模型能力分数。

## 环境与文档

| 项目 | 实测 |
|---|---|
| CPU / 系统 | Kunpeng 920、aarch64、Ubuntu 20.04 |
| NPU | 6 × 旧款 Ascend 910，单卡 32 GiB 级别 HBM |
| 主机驱动 / 容器 CANN | 25.5.2 / 8.5.0 |
| 模型 | Qwen3.8-27B，GGUF `qwen35`，Q4 展开存储 |
| 并发 / 上下文 | 1 / 256 tokens |
| 镜像标签 | `legacy-ascend-qwen38:2026.09.19`，尚未推送公共 registry |

- [完整优化报告与效果](docs/OPTIMIZATION_REPORT.md)
- [全部参数、开关与适用范围](docs/PARAMETERS.md)
- [构建、启动、转换权重、导出镜像](docs/CONTAINER.md)
- [算子 ABI、边界与验证](docs/OPERATORS.md)
- [CANN 缓存修复来源及 ABI](docs/CACHE_FIX.md)
- [性能测试协议](docs/BENCHMARKS.md)
- [发布与复现清单](docs/RELEASE_ASSEMBLY.md)

## 快速构建

需要自行提供支持旧款 910 的 aarch64 CANN 8.5.0 开发镜像。源码包不包含 SDK、驱动或模型。

```bash
python3 scripts/prepare-source.py
export BASE_IMAGE='your-compatible-cann85-base:local'
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg BUILD_JOBS=12 -f docker/Dockerfile \
  -t legacy-ascend-qwen38:2026.09.19 .
```

实际验收命令包含 `--privileged`，适用于本次旧驱动容器环境；具体权限边界、模型挂载和六卡运行命令见 [容器说明](docs/CONTAINER.md)。基础镜像版本、源码 commit、模型及修复库哈希见 [manifest](release-manifest.json)，最终镜像 ID 见 [镜像验收记录](benchmarks/image-validation.json)。

## 源码位置

```text
src/kernels/                GDN / SSM / RMS / RoPE / SET_ROWS TIK 生成器
patches/llama.cpp.patch      完整后端适配、调度及来源标记补丁
patches/cache-pointer-fix.patch  CANN 上游缓存修复的独立移植
tests/                      C++ 后端测试、Python 数值参考
scripts/                    固定源码准备、构建、入口与缓存库处理
tools/                      模型回归/性能工具、权重展开工具
artifacts/cache-fix/         已校验的隔离修复库，独立 CANN 许可
benchmarks/                 脱敏日志、JSON 结果与最终镜像身份
docs/                       中文文档
```

后端代码集中保存在精确版本补丁中；应用补丁即可得到对应 `.cpp/.h/.inc` 文件。镜像构建自动编译算子 `.o`，放入 `/opt/qwen38-910/kernel_meta/`。

## 许可与复现边界

原创代码按 MIT 发布；llama.cpp 派生部分保留 MIT；opbase 缓存修复及其二进制保留 CANN COSL 2.0。各组件和依赖许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。固定来源的 llama.cpp 和原始算子可通过 Dockerfile 重新编译；缓存修复库采用随包固定哈希二进制，另附来源及未重新执行完整编译的候选重建脚本，不能把整个包宣传为已验证的全依赖源码从零复现。

目前没有验证长上下文、多请求并发、视觉、工具调用、官方原始 FP16 权重精度，也不保证可直接套用于 A2、910B2 或其他 CANN 版本。
