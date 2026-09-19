# CANN 8.5.0 多卡执行器缓存修复：来源、ABI 与复现

本项目对 `nnopbase` 做了隔离构建，将上游的一行缓存指针修复移植到配套的
CANN 8.5.0 源码。镜像默认通过**进程级 `LD_PRELOAD`**使用随发布物固定摘要
校验的库，并设置 `ACLNN_CACHE_LIMIT=10000`。**宿主驱动和 SDK 原库均不替换。**

本文区分三个对象：已运行验证的原始隔离构建库、清理 RPATH 后的公开交付库、
以及用户按脚本重新构建的候选库。三者不能混用 SHA256；公开交付库的独立镜像复验结果另见 `benchmarks/image-validation.json`。用户自行重建的候选库仍需独立验收。

## 源码来源和实际修复

| 项目 | 值 |
|---|---|
| 官方源码 | [cann/opbase](https://gitcode.com/cann/opbase) |
| 配套基线 | `v8.5.0`，`b27e949de1ce29df366a279adf3976f2660e669c` |
| 上游修复 | [`f37e88e9525fac93214acd6fc652d9c69e191e02`](https://gitcode.com/cann/opbase/commit/f37e88e9525fac93214acd6fc652d9c69e191e02) |
| 修复标题 | `fix kernel name null issue when run from cache in invalid handle scenario` |
| 本项目补丁 | [`patches/cache-pointer-fix.patch`](../patches/cache-pointer-fix.patch) |
| 目标库 | `libnnopbase.so`，不是 `libopapi.so` |
| 许可 | [CANN Open Software License Agreement 2.0](../licenses/CANN-COSL-2.0.txt) |

移植位置为 `src/nnopbase/composite_op/aclnn_engine/rts_arg.h`：

```cpp
KernelLaunchConfig &GetKernelLaunchConfig()
{
    launchCfg_.kernelNameOfNoFatBin = kernelNameOfNoFatBin_;
    return launchCfg_;
}
```

执行器缓存对象被复制后，内部 `kernelNameOfNoFatBin` 原始指针可能仍指向旧缓冲。
当切换设备后函数句柄失效，需要重新取得内核函数时，旧指针会带来空值或乱码。
此修复每次取配置时重新绑定到当前缓存对象中的名称数组。

只移植兼容 8.5.0 的修复行，不把后续版本的完整测试/结构体定义混入旧基线，也不
直接使用 master 或新 CANN 版本的整库。`ASCEND_CACHE_PATH` 等磁盘缓存目录
选项不能修复这个进程内指针问题。

## 已有产物与验证边界

| 对象 | SHA256 / 状态 |
|---|---|
| 原始隔离构建 `libnnopbase.so` | `3a6ed3baeaed2ff69ddfc264461433352697e64c7463be52caf41a26cf4bf9da` |
| 原构建配套 `libdummy_tls.so` | `db32a75ce27aa68e5d50d668d117c5b9f9f8c2c9b5263e700608bf0961973d69` |
| 原环境 SDK `libnnopbase.so` | `7f0a0e1d437a285a593dac4cb6f394ea07a513d2988f0c93f0ed1dc7af2ef9ff`；前后保持一致 |
| 清理 RPATH 后公开库 | `344f759c370ab28b08ce38d73aa2e2b464a5ce84a139a7d726704e879570c324`；清理后镜像/运行复验以最终发布证据为准 |

原隔离构建库已完成 `ldd -r` 检查、真实多卡模型运行以及同参数 cache-off/on
对照。原库的简单小矩阵测试也能通过，因此小矩阵测试不是修复前失败/修复后成功
的充分证据；实际缓存错误的对照来自完整多卡模型。

原 ELF 的 `DT_RPATH` 含临时构建目录和末尾空项。公开交付版本已将其清理为
`/usr/local/Ascend/cann/lib64`；这是二进制分发整理，不是新增的缓存算法修改。
RPATH 清理会改变 SHA256，必须重新检查动态依赖、实际加载路径和推理结果。
公开库的最终身份以发布清单与构建脚本中的固定摘要一致为准。
清理过程保存在 [`scripts/sanitize-cache-rpath.py`](../scripts/sanitize-cache-rpath.py)。
该工具只接受上述已知摘要的原始库或清理后库，不是任意新编译 ELF 的通用编辑器。

原构建产生过配套 `libdummy_tls.so`。运行验证中也检查过使用 SDK 提供的同名
依赖的情况；不能仅因名称相同就认定任意版本可互换。最终公开镜像应保留其
实际 `ldd -r` 和 `/proc/<pid>/maps` 证据。不要擅自把 `dummy_tls` 从另一 CANN
版本拷入当前 SDK。

## 原构建环境与 ABI 事实

- 平台为 aarch64，CANN 8.5.0；原构建日志显示 GCC 12.3.1、CMake 3.27。
- CMake 使用 `Release`、`BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON`、
  `ENABLE_UT=OFF`、`ENABLE_ST=OFF`，只构建 `nnopbase` 及其必要依赖。
- `nnopbase` 按官方规则使用 C++17、`_GLIBCXX_USE_CXX11_ABI=0`，并保留
  `google=ascend_private`、`nlohmann=ascend_nlohmann` 等符号隔离定义。
- 本地生成 protobuf 25.1 / Abseil 20230802.1 的工具和头文件；实际链接的
  protobuf 共享依赖由已安装 CANN 提供，名称为
  `libascend_protobuf.so.3.13.0.0`。不能仅凭该文件名推断源码版本，也不能用
  系统 `apt`/`pip` 安装的通用 protobuf 替换它。
- 原构建的 `link.txt` 未列入静态 `.a` 库；但 protobuf 生成代码及 JSON/Abseil
  头文件中的模板/内联代码会参与编译，所以仍保留其第三方许可。

首次链接成功后，`ldd -r` 报告 `CopyToEncodedBuffer(std::string_view)` 未解析。
检查 SDK 的动态符号后确认实际导出的是
`absl::lts_ascend_private::string_view` 版本。解决办法是将隔离构建目录中的

```text
build/protobuf_host/include/absl/base/options.h
```

改为：

```cpp
#define ABSL_OPTION_USE_STD_STRING_VIEW 0
```

再重新编译 `nnopbase`。这不是修改 SDK 头文件，也不是把全项目 C++ 标准改旧。
如果重新生成或安装这份私有 protobuf 头文件，需要再次确认该宏没有被覆盖。

原容器缺少 `patch` 命令，构建时还在私有源码副本中将 `protobuf.cmake` 的
`patch -p1` 改为 `git apply -p1`，并给两个 protobuf patch 补了末尾换行。
这些只解决构建环境兼容，不改变缓存修复的数学或执行语义。

## 重新构建候选库

[`scripts/build-cache-fix.sh`](../scripts/build-cache-fix.sh) 整理了上述已观察到的
构建步骤。**本轮仅完成脚本静态检查，没有重新执行它完成一次干净编译；不能把它
描述为已经验证的自动重建流程，也不保证逐字节复现原始 SHA256。**

准备包含固定提交的官方 Git 仓库，在已提供兼容 CANN 8.5.0 开发环境的 aarch64
容器中执行。示例路径是用户自行选择的临时目录，不需要放在 SDK 内：

```bash
git clone https://gitcode.com/cann/opbase.git ./work/opbase-upstream
git -C ./work/opbase-upstream fetch origin b27e949de1ce29df366a279adf3976f2660e669c

CANN_ROOT=/usr/local/Ascend/cann-8.5.0 \
  bash scripts/build-cache-fix.sh \
  ./work/opbase-upstream ./work/cache-rebuild ./work/cache-candidate 8
```

脚本会从固定提交导出独立源码、校验补丁上下文、构建私有 protobuf 工具/头文件、
设置 Abseil ABI 选项、编译 `nnopbase` 并执行动态链接检查。上游 CMake 可能下载
固定版本的 protobuf、Abseil、JSON、Eigen 和 makeself；实际下载来源及摘要应
作为新构建的独立来源记录保存。脚本不执行 SDK 安装、不调用 NPU、不替换仓库
默认固定摘要库，也不清理或复用已有目录。

编译成功之后还需要：检查/清理 RPATH，保存新摘要，确认全部依赖解析正确，在独立
进程完成算子精度及多卡模型测试，最后再决定是否把候选库纳入另一个发布版本。
仅有 `ldd -r` 干净不能证明模型正确；也不能直接把新候选库改名覆盖已验证产物。

## 默认加载和回退

已组装镜像的入口默认使用公开固定摘要库：

```bash
export LD_PRELOAD=/opt/qwen38-910/cache-fix/libnnopbase.so
export ACLNN_CACHE_LIMIT=10000
```

若要对照关闭缓存，在启动容器时设置：

```bash
-e ENABLE_VERIFIED_CACHE_FIX=0
```

这会令本项目入口不加入该 preload 并设置 `ACLNN_CACHE_LIMIT=0`。对照进程还应
确认没有从外部继承另一份 `LD_PRELOAD`。库搜索路径和 preload 仅作用于该进程；
不修改 `/usr/local/Ascend` 中的任何库。

## 随附第三方许可

| 组件 | 实际来源/版本 | 许可文件 |
|---|---|---|
| opbase / nnopbase / dummy_tls | 官方 v8.5.0 与上述最小补丁 | [CANN COSL 2.0](../licenses/CANN-COSL-2.0.txt) |
| nlohmann JSON | 构建目录头文件确认 **3.11.3** | [MIT](../licenses/nlohmann-json-MIT.txt) |
| protobuf | 上游 CMake 指定 **25.1**，带官方命名/版本 patch | [BSD](../licenses/protobuf-BSD.txt) |
| Abseil | 上游 CMake 指定 **20230802.1**，带官方隐藏符号 patch | [Apache 2.0](../licenses/abseil-Apache-2.0.txt) |

这些许可证副本来自实际构建目录；CANN SDK 自身、动态依赖和宿主驱动仍由用户的
基础环境提供。本仓库根目录的 MIT 许可不会覆盖或改变上述组件的原始许可。
