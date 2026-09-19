# 镜像构建与发布范围

本项目已在 aarch64 旧款 Ascend 910 上完成独立镜像构建和硬件验收。最终镜像 ID、字节大小及验收状态记录于 `benchmarks/image-validation.json`。原始实验镜像标签是 `legacy-ascend-qwen38:2026.09.19`；清理后的公开镜像为 `legacy-ascend-qwen38:2026.09.19-public`，下载与摘要见 [PUBLIC_IMAGE.md](PUBLIC_IMAGE.md)。

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

## 验收范围

本项目公开镜像用于 linux/arm64 和旧款 Ascend 910，提供 CLI 实验运行环境。镜像及源码包不含模型权重或宿主驱动。模型的 Q4 展开存储说明、具体算子条件和已验证参数保留在优化报告、PARAMETERS.md 与 OPERATORS.md。

当前验收使用 6 张旧款 910、ctx=256、单并发，并存在 CPU 回退。独立容器运行所用权限、只读挂载和镜像身份记录在机器可读验收文件中；不应将本次结果推广为其他驱动、卡型或高并发配置的验收结论。

本公开文档不提供私人部署地址、API 凭据或模型调用示例。完整镜像的版本、分卷名称和 SHA256 见 [PUBLIC_IMAGE.md](PUBLIC_IMAGE.md)。

## 许可与来源

构建时使用的 SDK 基础镜像需独立取得。公开实验镜像清除了原环境运行日志和用户级配置，保留安装组件及其许可；项目 MIT 不替代 SDK、模型或第三方依赖的原始条款。缓存修复库的来源、ABI 和候选重建边界见 CACHE_FIX.md。
