# 旧 Ascend 910 算子契约与验证边界

本文件对应 [llama.cpp 集成补丁](../patches/llama.cpp.patch)、[设备内核源码](../src/kernels/)、[测试入口](../tests/validate-operators.sh)及 [CANN 缓存修复补丁](../patches/cache-pointer-fix.patch)。形状均按 GGML 的顺序书写：`ne[0]` 是连续的最内层维度；设备 ABI 中另有说明的数组使用通常的外层到内层顺序。

本版面向运行时 SoC 名称为 **`Ascend910B` 的旧 Ascend 910**，不是 Atlas A2 的 `Ascend910B1/B2/...`。已验证环境为 ARM64、CANN 8.5.0；发布构建关闭 `USE_ACL_GRAPH`。自定义设备 ELF 使用 legacy `v100/cloud` TIK 配置，并在当前 NPU stream 上执行。ELF 字节、注册句柄及入口名称在进程生命周期内保持有效，不能在 ACL 清理之后再次使用。

**算子数值验证与整模型能力是不同的验收层次。** 本次模型权重由 Q4 GGUF 展开为 FP16 存储，保留 Q4 量化损失；通过以下测试不等于恢复官方原生 FP16 权重质量，也不证明长上下文、多请求并发或所有 Qwen3.x 型号可用。自定义算子中的 FP32 运算不代表整条模型计算图始终采用 FP32；其余 CANN 路径的内部精度应另行验证。整模型速度与运行范围见 [BENCHMARKS.md](BENCHMARKS.md)，本文件不引入新的性能数字。

## 启用变量与回退规则

下列变量应在进程启动前设置。内核配置、文件可用性或注册对象会被缓存；运行途中修改环境变量不能可靠切换版本。[发布入口](../scripts/release-entrypoint.sh)会主动设置全部已验证路径，单独关闭某项做 A/B 时，应直接调用相应测试二进制并控制其环境，不能假定外部 `unset` 会绕过入口脚本的重新赋值。

| 变量 | 含义及发布值 | 未启用或不匹配时 |
|---|---|---|
| `GGML_CANN_GDN_910_BINARY` | H32 packed GDN ELF：`gdn_decode_910.o` | 无可用对应 ELF 时不声明支持该自定义 GDN |
| `GGML_CANN_GDN_910_BINARY_48` | H48 packed GDN ELF：`gdn_decode_910_h48.o` | 同上；也是 H48 native 布局不满足时的备选 |
| `GGML_CANN_GDN_NATIVE_DIR` | 含 `gdn_native_h48_q16.o`、`gdn_native_h48_q48.o` 的目录 | 尝试对应 packed 路径；再不满足则交回原图调度 |
| `GGML_CANN_SSM_910_BINARY` | `ssm_conv_910_t1.o` 的绝对路径 | 保留原 CANN SSM_CONV |
| `GGML_CANN_RMS_910_DIR` | 五种 RMS ELF 的目录 | 形状、epsilon 或可用文件不匹配时保留原 CANN RMS |
| `GGML_CANN_ROPE_910_COMPOSITE` | 仅字符串 `1` 开启旧卡 IMRoPE 支持 | 旧卡该 ROPE 节点不由此 CANN 扩展接管，由原调度器选择支持的后端 |
| `GGML_CANN_ROPE_910_NATIVE_DIR` | 含 H4/H24 单 token RoPE ELF 的目录 | 不设置或形状不匹配时继续 composite |
| `GGML_CANN_SETROWS_910_DIR` | 含 `setrows_k_910_t1.o`、`setrows_v_910_t1.o` 的目录 | 不设置、标记或形状不匹配时保留原 CANN SET_ROWS |
| `ENABLE_VERIFIED_CACHE_FIX` | 发布入口默认 `1`，装载独立 `libnnopbase.so` | 设为 `0` 时入口使用 `ACLNN_CACHE_LIMIT=0` |
| `ACLNN_CACHE_LIMIT` | 修复库启用时入口设为 `10000` | 缓存关闭基线为 `0` |
| `GGML_CANN_WEIGHT_NZ` | 发布入口固定为 `off` | 本版不把其他权重布局作为已验证优化 |
| `GGML_CANN_PROFILE_SYNC` | 性能测试时入口移除该变量 | 逐算子同步分析不能混入正常吞吐对照 |

“回退”表示没有让不满足契约的输入进入专用内核，不是对任意输入都保证能在旧卡运行。GDN 在实际执行时对错误 SoC 或启用 ACL graph 的情况会明确终止；SSM、RMS、RoPE native、SET_ROWS 在其检查不满足时返回原路径。发布构建统一关闭 ACL graph。

文件失效与形状回退也不同：GDN/SSM 会先检查候选 ELF 可读性；RMS 缺失可选文件会回退。RoPE native、SET_ROWS 已配置目录且命中形状后，如果指定 ELF 缺失、非法或读取不完整，会终止并报错，不会悄悄换实现。成功注册或打印“launch queued”只说明已提交；必须同步执行并检查数值结果。

## Gated DeltaNet：原始布局与 packed 备选

实现文件位于补丁中的 `gdn_910.cpp/.h`；内核为 `build_gdn_native.py` 和 `build_gdn.py`。

共同前提：`GGML_OP_GATED_DELTA_NET`，全部六个输入及输出为 F32；`T=1`、`B=1`、键和值宽度均为 128；GGML 的该算子参数为 `K=1`。状态、门控和输出必须连续，Q/K/V 的每个 128 元素行必须连续。输出 value heads `Hv` 只能为 32 或 48。

| 张量 | GGML 形状 | 语义 |
|---|---|---|
| Q/K | `[128,Hq或Hk,1,1]` | 已在图中做 L2 归一化；head 数须为正且整除 `Hv` |
| V | `[128,Hv,1,1]` | 当前 token 的 value |
| g / beta | `[1,Hv,1,1]` | g 为对数衰减；beta 已经过 sigmoid |
| state | `[128,128,Hv,1]` | 物理顺序为 `[value_head,value_channel,key_channel]` |
| 返回缓冲区 | `[128*Hv,129,1,1]` | 先放 `Hv*128` 个输出值，再放更新后的完整状态 |

单个 head 的含义是先计算 `S_decay=exp(g)*S`，再计算 `delta=(v-S_decay*k)*beta`、`S_new=S_decay+delta*k^T`，最后输出 `S_new*(q/sqrt(128))`。这里的状态行是 value 维、列是 key 维；不能交换矩阵方向。输入 Q 已归一化，但内核仍按算子定义执行一次 `1/sqrt(128)` 缩放，调用者不能重复加这次缩放。

原始布局快路径仅覆盖 `Hv=48`、Q/K head 数同时为 16 或同时为 48，且 Q/K/V 行间距严格为 128 个 F32。设备 ABI 指针顺序为 `state,q,k,v,g,beta,out_state,out`，门控各为紧凑 48 个 F32。设备核直接读取原始指针，内部采用 **`head % Hq` 的循环映射**，不是把每个 Q/K head 连续复制若干次。H48 使用 24 个 block，各负责两个 head。

packed 备选支持 H32/H48，先按 GGML 的循环 head 规则补齐 Q/K，并在需要时复制跨行 stride 的向量。其 g/beta ABI 为每个 head 8 个 F32，只有首元素有效，以适应 32 字节 DMA；不能把紧凑门控直接交给 packed ELF。H32 使用 32 个 block。两套 ABI 不可互换。

不满足单 token、状态形状、类型或 stride 契约时，自定义 GDN 不声明支持。Prefill、其他维度和并发序列不由该内核处理，仍需原图调度与对应后端。`GDN_QK_HEADS=16|48` 和 `GDN_HEADS=32|48` 是生成 ELF 时的构建变量，不是运行时扩展支持范围的开关。

## 四抽头 SSM 卷积

`build_ssm_conv_native.py` 对应 `GGML_OP_SSM_CONV` 的单 token 特化：输入及权重的 GGML 形状均为 `[4,10240,1,1]`，输出为 `[10240,1,1,1]`，全部连续 F32。设备 ABI 为 `x,weight,out`，从通常的数组角度看输入为 `[channel,4]`。

每个通道执行四项乘积并按抽头 0、1、2、3 的顺序累加。32 个 block 各负责 320 个通道；F32 向量运算不引入 FP16 转换。内核只负责点积，不做 SiLU，也不更新卷积历史状态，这些仍是图中的独立操作。

`T>1`、通道数变化、非连续布局、其他精度、其他 SoC 或未配置可用 ELF 时保留原 CANN 实现。测试程序虽然也可接受 23-token fixture，但那不表示这个专用 ELF 支持 prefill；镜像的专用内核复验使用 T1。

## RMS_NORM：五个严格特化形状

数学定义为沿 `ne[0]` 计算 `y=x/sqrt(mean(x*x)+epsilon)`。该内核**不乘学习到的归一化权重**，也不执行后续门控或 L2 缩放。

| 宽度 | 行数 | 输入行间距，F32 元素 | epsilon | 用途与 ELF |
|---:|---:|---:|---:|---|
| 5120 | 1 | 5120 | `1e-6` | 主残差归一化；`rms_910_w5120_r1_s5120.o` |
| 128 | 48 | 128 | `1e-6` | GDN 输出归一化；`rms_910_w128_r48_s128.o` |
| 256 | 24 | 512 | `1e-6` | Q 与 gate 交错视图；`rms_910_w256_r24_s512.o` |
| 256 | 4 | 256 | `1e-6` | K 归一化；`rms_910_w256_r4_s256.o` |
| 128 | 16 | 128 | `1e-6/128`，即 `7.8125e-9` | GDN Q/K 的 L2 组成部分；`rms_910_w128_r16_s128_l2.o` |

设备 ABI 为 `input,output`，输入输出均为 F32，输入 `nb[0]=4`、`ne[2]=ne[3]=1`，输出连续且与输入逻辑形状相同。epsilon 按编译时 F32 值精确匹配；Q 的行间距 512 包含被跳过的 gate 数据，不能误按 256 读取。

计算采用 FP32 平方、归约、近似倒平方根及两次 Newton 修正。L2 形状使用独立 ELF，图中的后续缩放保留不变；不能用常规 `1e-6` 替代 `1e-6/128`。设备 ABI 的 block 数为 `min(rows,32)`。

构建脚本可生成更多满足其基础约束的形状，但运行时只允许上表五项；“能生成 ELF”不等于被发布版本接入或验证。其他形状、epsilon、缺失可选 ELF、类型或 SoC 不匹配时保留原 CANN RMS。

## RoPE：IMRoPE composite 与单 token 融合核

本模型的完整注意力使用 `GGML_ROPE_TYPE_IMROPE=40`：Q 为 `[256,24,T,1]`，K 为 `[256,4,T,1]`，输入输出 F32；位置为 I32 `[4*T]`，四段依次表示 t/h/w/e。`n_dims=64`、sections 为 `[11,11,10,0]`、频率基数 `1e7`、频率缩放 1、外推因子 0、attention factor 1。四轴位置不能假设相等。

旋转配对为维度 0–31 与 32–63；维度 64–255 保留原值。IMRoPE 的轴选择包括交错规则及各 section 的边界，不能用普通 RoPE 代替。原 CANN 缓存生成器继续负责位置选择、频率和 sin/cos；快路径只替换缓存生成之后的应用部分。

### Composite 路径

旧卡支持检查要求开启 `GGML_CANN_ROPE_910_COMPOSITE=1`、F32 数据、I32 位置、无单独频率因子输入、mode 40、offset 为 0。旋转维度必须为正偶数且不超过头宽，头宽不超过 896；sections 非负、总和大于零且不超过旋转维度；数据最内层 stride 为 4 字节，位置向量至少有 `4*T` 项。

其实现是四次 Mul、一次 Sub、一次 Add，以及必要的未旋转尾部 Copy。四项乘积全部读取后才写回任一半边，因此不会提前覆盖仍需读取的数据。它支持模型 prefill 和带行间 padding 的输入；上面的宽泛检查是代码边界，硬件验收范围仍以本模型形状和下文测试为准。

### T1 native 路径

先满足 composite 契约，再要求头宽 256、旋转维度 64、head 数为 4 或 24、`T=B=1`、输入输出连续且地址 32 字节对齐，sin/cos 缓存已生成。设备 ABI 为 `x,sine,cosine,output`，数据均为 F32；cache 各含 64 项，内核只读前 32 项。

`rope_910_h4_t1.o` 和 `rope_910_h24_t1.o` 分别启动 4、24 个 block，每个 block 负责一个 head。整个 256 元素行先进入片上缓冲区，保持四次乘法、再减/加的运算顺序，不改为 FMA、不降低精度；只改前 64 项，再写回整行。尾部 192 项通过数据搬运保留。Prefill、padding、其他 head 数或 native 目录未设置时继续 composite；没有绕过原 IMRoPE 位置逻辑。

## SET_ROWS：单 token KV 更新与索引来源证明

专用核将 F32 到 F16 的转换和 KV 写入合并。只允许单 stream、单 token、1024 通道、缓存长度 256 的已标记图节点；缓存长短改变后必须回退，不能沿用固定步长内核。

| 类型 | 源数据，GGML 顺序 | 索引 | F16 目标 | 设备 ABI |
|---|---|---|---|---|
| K | F32 `[1024,1,1,1]` | I64 `[1]` | `[1024,256,1,1]` | `x,index,out` |
| V | F32 `[1,1024,1,1]` | I64 `[1024]` | `[1,262144,1,1]` | `x,index,cache,out` |

全部要求连续，源和目标地址 32 字节对齐。V 的 cache/out 必须是同一存储。两核均启动 32 个 block，每个 block 独占连续 32 个通道。K 将转换后的 1024 项写入一个连续 cache 行。

### 为什么不能仅凭形状启用 V 核

原 `llama-kv-cache.cpp` 中，`cpy_v` 的 `v_trans` 分支把每个 head 元素变成单元素行；`set_input_v_idxs` 生成 `stream*kv_size*channels + channel*kv_size + slot`。在精确的单 stream/T1/1024/256 条件下，得到：

```text
K: index[0] = position
V: index[channel] = channel*256 + position，0 <= position < 256
```

普通 SET_ROWS 即使形状相同，也可能有重复索引，不能用并行块读改写。补丁因此只在上述 `cpy_k/cpy_v` 图构造分支写入私有 `op_params` 标记：K 为 `0x53524b31`，V 为 `0x53525631`。native dispatch 同时要求标记、类型、形状、stride 和缓冲区条件。标记证明了索引生成来源，不是让外部任意 scatter 使用的通行证；新增调用者不得自行伪造标记。由于标记位于模型图源码，接入时必须同时重编 `libllama` 与 `ggml-cann`。

V 内核由 index[0] 取得 position，依赖已证明的其余索引关系。它对每个通道读入包含 position 的对齐 32 字节块，仅更新其中一个 half，再写回。每核用一次 32-burst strided DMA 收集 32 个块、在片上修改后再用一次 strided DMA 写回；通道行相隔 512 字节，不同核没有共享写块，另外 15 个 half 保持不变。

两核读取 index[0] 时使用 32 字节 DMA。K 的逻辑索引只有 8 字节，封装会额外检查索引地址对齐，并确认其所在 GGML 后端缓冲区从该地址起至少还有 32 可读字节，否则回退。不能因只使用第一个 I64 就忽视 DMA 的实际读取范围。

转换使用 TIK `vec_conv` 的 `none` 模式。当前位置越界时内核不写入；正常图构造保证 slot 有效，此保护不应当被当作接受非法索引后的业务成功语义。目录未设置、来源标记不匹配或任一形状条件不满足时，保留原 CANN 的 Cast + InplaceIndexCopy 路径，包括 prefill。

## CANN 多卡执行器缓存修复

该修复不是模型 KV 缓存优化。它修复 opbase 执行器缓存中 kernel 名称指针的生命周期问题：在 `GetKernelLaunchConfig()` 返回前，将 `launchCfg_.kernelNameOfNoFatBin` 重新指向当前 `kernelNameOfNoFatBin_`。对应上游修复提交为 `f37e88e9525fac93214acd6fc652d9c69e191e02`，本版基于 opbase `b27e949de1ce29df366a279adf3976f2660e669c` 构建，来源锁定见 [release-manifest.json](../release-manifest.json)。

独立修复库通过发布进程的 `LD_PRELOAD` 加载，并启用 `ACLNN_CACHE_LIMIT=10000`；不替换系统安装目录中的 CANN 库。修复库必须匹配 SDK 的 protobuf/abseil 等 ABI，本次构建使用 `ABSL_OPTION_USE_STD_STRING_VIEW=0` 与已安装库保持一致。

编译成功或单卡矩阵乘通过都不足以验收。应确认 `ldd -r` 没有未解析符号、实际进程加载的是独立修复库，并完成多卡整模型缓存开启的连续运行。`ENABLE_VERIFIED_CACHE_FIX=0` 保留关闭 ACLNN 缓存的基线；形状特化内核与缓存修复可分别对照。

## 已完成的硬件测试与复验入口

当前镜像已在旧卡上通过 SSM、五种 RMS、32 项 RoPE、28 项 SET_ROWS，以及当前 GDN backend 测试。GDN 另有前一阶段独立原生 runner 的 **128 步递归状态硬件验证**；不要把它与镜像脚本中的单步检查混写。

| 项目 | 检查内容 | 判定与范围 |
|---|---|---|
| GDN 当前 backend 测试 | Hq/Hk=16、Hv=48、D128，连续与跨 head stride 两种布局；检查完整 output 与 state | 逐元素 `abs_error <= 2e-6 + 2e-5*abs(reference)`，拒绝非有限值；覆盖 native 与 packed 备选，当前脚本运行单步 |
| GDN 历史独立 runner | native Q16/Q48 路径的 128 步状态递推 | `native48-run.log` 记录 `verified_steps=128`；对应 `native48-verification.json`、`native16-verification.json` 的完整输出和状态检查通过。Python oracle 自检也有 128 步，但不能替代 NPU 证据 |
| SSM | T1、10240 通道、四抽头；检查完整输出 | 镜像传入最大绝对误差与 NRMSE 门槛均为 `2e-6`。测试器不传门槛时只是报告，不能据退出码宣称精度验收 |
| RMS 五形状 | 每种覆盖零、微小值、正常值、大数、负常量等 7 组输入；Q padding 放置哨兵 | 逐元素 `abs_error <= 2e-5 + 2e-5*abs(reference)`，拒绝非有限值；L2 用独立 epsilon 验证 |
| RoPE 32 项 | H4/H24 × T1/T23 × 连续/padding × 四个起始位置；四位置轴刻意不同，并重复更新同一图的位置 | 起始位置 0/17/127：max_abs≤`2e-4`、NRMSE≤`2e-5`；8191：≤`8e-3`、≤`8e-4`；所有尾部元素相等。后者门槛考虑原缓存生成与 CPU 频率递推顺序的差异，不是长上下文整模型质量保证 |
| SET_ROWS 28 项 | K/V × 14 个边界位置，每次比较完整 262144-half cache | 按 `uint16` 逐位相等；含正负零、舍入中点、half subnormal，未更新缓存覆盖全部 half 位模式，包括 NaN payload；同时验证未更新位置保留 |

RoPE 当前测试对尾部采用浮点值相等检查，并未像 SET_ROWS 那样逐位覆盖 NaN payload 或正负零；应区分“实现按原值复制”的契约与测试实际覆盖范围。测试起始位置到 8191 也不意味着本版 256-token 整模型缓存已经扩展。

镜像内统一复验入口为：

```sh
qwen38-910 validate-operators CANN4
```

其中 `CANN4` 需替换为容器实际可见、用于验证的设备编号；该命令会执行 NPU 工作。入口建立临时 fixture 并清理，脚本遇到任何失败立即停止。GDN fixture 生成器默认 `steps=1`，现有 `test_gdn_backend` 也只调用一次 GDN；增加 `--steps` 生成文件不能使此 backend 测试自动变为多步硬件测试。

单项复验可以使用 `test_gdn_backend`、`test_ssm_bench`、`test_rms_backend`、`test_rope_backend`、`test_setrows_backend`。SET_ROWS 的可选 `untagged` 参数用于强制普通索引路径对照；RoPE 在保留 composite 开关的同时移除 native 目录，可以对照融合核。硬件执行需检查自定义路径日志与数值结果，两者缺一不可。

以上通过后，仍需用固定模型权重、设备集合、上下文、提示词和输出 token 数做整模型 A/B，检查答案、持续状态及错误日志。算子测试或一次短问答不能单独证明整体提速，也不能代替原生 FP16 模型能力评测。
