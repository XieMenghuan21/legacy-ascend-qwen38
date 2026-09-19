# 实测参数与适用范围

本次实测为 Qwen3.8-27B 文本生成，使用 6 张旧款 Ascend 910、CPU 回退与专用 decode 算子。参数记录对应本项目 `model-benchmark`；不等于 vLLM、MindIE 或标准 llama-server 的通用启动配置。

## 模型与设备

| 参数 | 实测值 |
|---|---|
| 模型架构 | GGUF `qwen35`，模型名 Qwen3.8-27B |
| 权重输入 | Unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf` |
| 运行权重 | 将该 Q4 GGUF 反量化展开为 FP16 存储，保留 Q4 量化误差 |
| 计算设备 | CANN2,CANN3,CANN4,CANN5,CANN6,CANN7 |
| 切分 | llama.cpp 默认按层切分，未设置自定义 tensor split |
| `n_gpu_layers` | 99，允许不支持算子回退 CPU |
| SoC / 架构 | 旧款 Ascend 910；ACL 返回 `Ascend910B`；aarch64 |
| 单卡 HBM | 32 GiB 级别 |
| 驱动 / 容器 CANN | 25.5.2 / 8.5.0 |
| CPU | Kunpeng 920 5250，主机共 192 核 |
| 主机内存 | 约 754 GiB |

本机有 8 张卡，实验使用其中 6 张，另外 2 张保留原有业务。并不是硬件只能双卡；本次四卡配置可以加载权重，但后续计算工作区分配失败。换旧内存分配器和尝试非均分后仍失败，不能把原因简化为“分配不均”。

## 生成参数

| 参数 | 实测值 |
|---|---|
| `--context` / `n_ctx` | 256 |
| `--output-tokens` | 固定长度性能实验 64；回归用例遇 EOS 提前停止 |
| `--threads` / `n_threads_batch` | 8 / 8 |
| `n_batch` / `n_ubatch` | 64 / 32 |
| `n_seq_max` / 请求并发 | 1 / 1 |
| 采样 | greedy，确定性选择最大 logit |
| 思考模式 | 模板中显式闭合 `<think>` 前缀 |
| Flash Attention | 关闭 |
| KV 类型 | 上游默认 FP16 |
| 性能 profiling | `GGML_CANN_PROFILE_SYNC` 不设置 |

仅验证了这个短上下文范围。CLI 虽可接受其他 context 数值，但 SET_ROWS 专用路径仅适用于 256，其他形状会回退；长上下文的显存、速度和质量没有本次验收数据。

## 运行环境变量

入口脚本自动设置下表。文件位于镜像 `/opt/qwen38-910`。

| 变量 | 值 / 用途 |
|---|---|
| `GGML_CANN_GDN_NATIVE_DIR` | `kernel_meta` 目录；原生布局 GDN |
| `GGML_CANN_GDN_910_BINARY` | `gdn_decode_910.o`；旧 32 头路径 |
| `GGML_CANN_GDN_910_BINARY_48` | `gdn_decode_910_h48.o`；48 头后备路径 |
| `GGML_CANN_SSM_910_BINARY` | `ssm_conv_910_t1.o` |
| `GGML_CANN_RMS_910_DIR` | `kernel_meta` 目录；5 个专用 RMS 形状 |
| `GGML_CANN_ROPE_910_COMPOSITE` | `1` |
| `GGML_CANN_ROPE_910_NATIVE_DIR` | `kernel_meta` 目录；T1 原生 RoPE |
| `GGML_CANN_SETROWS_910_DIR` | `kernel_meta` 目录；已标记 KV 更新 |
| `GGML_CANN_WEIGHT_NZ` | `off` |
| `ENABLE_VERIFIED_CACHE_FIX` | 默认 `1`，启用本项目已验证的隔离库 |
| `ACLNN_CACHE_LIMIT` | 修复库开启时 `10000`，关闭修复库时 `0` |
| `LD_PRELOAD` | 指向镜像内修复版 `libnnopbase.so`，不替换 SDK 原库 |

`ENABLE_VERIFIED_CACHE_FIX=0` 是回退选项；关闭后不能沿用开启状态的速度结论。修改 CANN 版本必须重新检查动态库 ABI。

## 不采用的路线

- 预先把矩阵乘的激活转为 FP16：结果相同，但本次实测没有收益，工作区也未减少。
- Weight NZ：旧款 Atlas 训练系列的接口支持受限，本机实际返回不支持；默认关闭。
- 四卡强行部署：权重装载之后的工作区分配未通过，未作为可用配置交付。
- “Q4 展开成 FP16 等于恢复原始精度”：不成立；本项目不作这种质量承诺。
