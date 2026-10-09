# D5 验收记录

本文件区分宿主合同测试、真实 Docker 执行和真实模型调用。模拟执行器不能证明容器隔离，API 调用成功也不能证明任务成果。

## 自动验收

- 宿主：CSV 自动识别/资源限制、重复表头与原始值、TXT/MD/PDF/BM25/引用、实际双 Agent 工具轨迹、交付项、预算、恢复去重、流式正文与执行进度、产物攻击输入/分页/全量导出/失败回滚、VS Code 正式入口。
- Docker：三类核心任务与独立手算值、列顺序/新类别、280 组和大整数、报错修正、网络/密钥/输入写入限制、无限循环/OOM、取消清理、符号链接/50 MiB 超限、stdout 截断/协议伪造与无模型重放。
- CI：Ubuntu/Windows × Python 3.11/3.12 运行宿主测试；Ubuntu 构建锁定镜像并运行真实 Docker 测试及公开问卷演示。没有向 CI 提供模型密钥。

```powershell
uv run --frozen pytest -q -m "not docker"
# 先启动 Docker 并构建镜像：
$env:LABWEAVER_REQUIRE_DOCKER="1"
uv run --frozen pytest -q -m docker
```

缺少 Docker 时本地 Docker 测试明确 skip；CI 设置 REQUIRE_DOCKER=1，缺失或构建失败必须 fail。正式入口只有 `src/labweaver/app.py`，测试集中在 `tests/`，生成代码/日志/图片只在忽略的运行目录中。

## 本轮实际状态（2026-10-09）

宿主测试已运行，最后结果将在完成后更新。Docker Desktop 4.94.0 和 Microsoft WSL 3.0.1 已在本机安装，硬件虚拟化开启、Virtual Machine Platform 显示 Enabled，但 Windows 的 HypervisorPresent=false，Docker Linux 引擎报告 Virtual Machine Platform not enabled 并返回 500。因此本机暂不能构建镜像或进行真实容器验收；需要完成 Windows 虚拟化启动/重启后复测。

Ubuntu CI 容器验收与公开合成数据在线结果会分别记录，不将当前环境失败记作通过。不自动重启用户电脑。

## 验收解释

完成同时要求合法执行证据、通过校验的产物和所有记录的必需交付项；显式图表必须有真实图片。数值与口径仍用独立预期核对。引用仅证明实际检索片段与来源可追溯，不自动证明全文语义。

源码输入 SHA-256 与结果来源关联；源 CSV 不直接挂入可写容器。报告版本2不迁移旧日志，也不支持进程退出后 checkpoint 恢复。重放只重新执行保存的成功代码。
