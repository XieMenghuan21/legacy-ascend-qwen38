# 公开实验镜像

完整镜像发布于 [GitHub Release v2026.09.19-public-image](https://github.com/XieMenghuan21/legacy-ascend-qwen38/releases/tag/v2026.09.19-public-image)，平台为 `linux/arm64`，用于旧款 Ascend 910（ACL SoC 为 `Ascend910B`，不等于 A2 的 B1/B2）。不包含模型权重或宿主驱动。

| 项目 | 值 |
|---|---|
| 镜像标签 | `legacy-ascend-qwen38:2026.09.19-public` |
| 镜像 ID | `sha256:cf8abd0ef1556b82c8af8b8520f3c68da0f528c0e22798facdc9d58dd52fcff7` |
| 镜像大小 | 18,847,576,764 字节 |
| 压缩归档 | `legacy-ascend-qwen38-2026.09.19-public.tar.gz` |
| 归档大小 | 7,829,988,076 字节 |
| 归档 SHA256 | `fc88ea4a0b62fd49b2efd287226977842b097a30f338bd3ca7fe6197e96e57ee` |
| 发布分卷 | 4 个，`.part00` 至 `.part03` |

GitHub Release 单文件上限为 2 GiB，因此采用分卷附件，详见 [GitHub 官方说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。Release 同时提供分卷与整包两份 SHA256 清单。先核对分卷摘要，再按编号连接所有分卷，整包摘要必须与上表一致。分卷不能单独解压或导入 Docker。

## 文件与验收

公开镜像清除了基础环境的历史运行日志、缓存、用户级配置与历史记录，并通过文件系统导出、重新导入形成单层镜像。旧镜像层未复用，已删除内容不会残留在旧层中。

76 项运行文件与链接的哈希保持一致，ENV、入口和工作目录等配置保持一致。公开镜像重新运行四项真实模型回归，全部通过。原有单算子测试记录与固定 64-token 的 5.67 token/s 结果属于原始实验镜像；本次没有为公开镜像重新宣称另一组完整性能成绩。详细证据见 [公开镜像验收记录](../benchmarks/public-image-validation.json)。

保留 SDK 原有许可与声明；原创项目的 MIT 许可不替代第三方组件许可。本页面只提供公开发布物信息，不包含任何私人部署地址、凭据或模型调用示例。
