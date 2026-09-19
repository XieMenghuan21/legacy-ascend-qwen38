# 镜像构建与运行

本项目已在 aarch64 旧款 Ascend 910 上完成独立镜像构建和硬件验收。最终镜像 ID、字节大小及验收状态记录于 `benchmarks/image-validation.json`。镜像标签是 `legacy-ascend-qwen38:2026.09.19`，尚未推送公共仓库。

## 准备与构建

需要自行提供支持旧款 910 的 CANN 8.5.0 开发基础镜像，内含 TBE/TIK、Python、CMake、Git 和 C++17 编译器。本次基础镜像使用 GCC 12.3.1。A2 专用镜像不能据名称直接替代。主机需要兼容驱动。

```bash
python3 scripts/prepare-source.py
export BASE_IMAGE='your-compatible-cann85-base:local'
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg BUILD_JOBS=12 -f docker/Dockerfile \
  -t legacy-ascend-qwen38:2026.09.19 .
```

上游源码锁定到 manifest 中的 commit，并验证完整源树摘要。准备源码脚本需要 Python 3.9 或更新版本。镜像构建不需要 NPU，不下载模型、不安装驱动。构建上下文采用白名单，源码包不含 SDK 或模型。

## 已验证的干净容器启动方式

以下是本机实际验收使用的方式。该旧驱动环境中，普通容器的设备映射尝试返回错误 87；通过 `--privileged` 的独立容器测试成功。因此目前只把这个启动方式标为已验证。它赋予容器较高权限且可见其他设备，`--devices` 只约束模型实际选卡，不是操作系统层面的设备隔离。没有要求修改宿主驱动或既有容器配置。

```bash
docker run --rm --network none --privileged --shm-size 8g \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  legacy-ascend-qwen38:2026.09.19 validate-operators CANN4
```

模型文件需是已展开的 GGUF，使用绝对路径挂载。示例使用 2 至 7 号卡，实际使用前应确认它们空闲。

```bash
export MODEL_FILE='/absolute/path/Qwen3.8-27B-from-Q4-F16.gguf'
docker run --rm --network none --privileged --shm-size 8g \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$MODEL_FILE:/models/model.gguf:ro" \
  legacy-ascend-qwen38:2026.09.19 benchmark \
  --model /models/model.gguf --devices 2,3,4,5,6,7 \
  --context 256 --output-tokens 64 --threads 8 \
  --prompt '用中文详细解释检索增强生成系统如何完成文档解析、检索、重排和答案生成。'
```

把最后一行 `--prompt ...` 改为 `--regression` 可执行四项短回归。stdout 是逐请求 JSON，stderr 是后端日志。镜像不监听 HTTP 端口；本次不涉及业务端口映射。

## 权重展开

先独立取得 manifest 指定的 Q4 GGUF，再运行：

```bash
export MODEL_DIR='/absolute/path/models'
docker run --rm --network none -v "$MODEL_DIR:/models" \
  legacy-ascend-qwen38:2026.09.19 convert-weights \
  /models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  /models/Qwen3.8-27B-from-Q4-F16.gguf
```

为输入加输出预留约 70 GiB 磁盘空间。展开仅变更存储格式；模型仍保留原 Q4 误差。转换入口在已验证镜像内完成编译，原实验转换结果的 SHA256 记录在 manifest。

## 导出与导入

```bash
docker save legacy-ascend-qwen38:2026.09.19 | gzip -1 > legacy-ascend-qwen38-2026.09.19.tar.gz
gzip -dc legacy-ascend-qwen38-2026.09.19.tar.gz | docker load
```

本次构建使用的 SDK 基础镜像需由使用者自行取得。GitHub 项目包发布源码、补丁和构建方法；将包含 SDK 的完整镜像推到公共 registry 前，需按所用基础镜像及组件条款处理分发。公共源码不提供未核实的第三方镜像下载地址。
