> 历史阶段记录：本次连续统计更新已替代旧终态与功能范围；当前使用方式见 [连续统计实现](session-statistics.md)。

# Intake 入口更新记录

日期：2026-10-03。本轮实施移除独立 CSV 概览命令、统一运行配置、发布 VS Code 入口与第三天计划。RAG 尚未实现。

这是入口改造时的历史记录。当前 D3 RAG 已实现，最新结果见 [D3 验收记录](day3-results.md)。

## 行为变化

- CLI 只保留 intake；旧独立命令会被 argparse 拒绝。
- Agent 内部 `profile_csv` 和所有统计字段、CSV 单元测试保留。
- VS Code 与 CLI 共用 `labweaver.toml`，只在临时覆盖时添加参数。
- 本地已有模型配置由被 Git 忽略的 `labweaver.local.toml` 引用；公开代码默认选择配置目录下的 `.env`，或进程中的 LLM 环境变量。
- `runtime/intake.py` 共用任务读取、模型选择、运行及记录保存。离线模式不读取模型凭据。
- 直接入口按项目目录定位，不受终端 cwd 影响。F5 配置提供 Windows、Linux、macOS 的虚拟环境路径。
- README、徽章、本地 Git 远程均使用 [pypypaoying/LabWeaver](https://github.com/pypypaoying/LabWeaver)。

## 本地实际验收

环境：Windows、Python 3.11.15、uv 0.11.12。完整测试 **118 passed、28 subtests passed**。`uv build --offline` 生成 wheel 和源码包成功。

本机以前的 pytest 临时目录存在用户 ACL 问题，验收使用新的被忽略目录作为 basetemp，不改变测试内容：

```shell
uv run --frozen pytest -q -p no:cacheprovider --basetemp .cache/full-suite-20261003-v1
uv build --offline
```

该 basetemp 仅供本次本机验收记录；通常使用 `uv run --frozen pytest -q`。再次在本机运行时应选择新的可写临时目录。

| 验证路径 | 模式 | 数据 | 模型调用 | 实际工具执行 | 终态 |
| --- | --- | --- | --- | --- | --- |
| CLI，默认配置仅覆盖模式 | offline | 问卷 8 × 5 | 2 | 1 | awaiting_confirmation |
| CLI，覆盖 CSV 与任务文件 | offline | 实验 6 × 6 | 2 | 1 | awaiting_confirmation |
| 直接入口，不同 cwd，网络与凭据读取被阻断 | offline | 问卷 8 × 5 | 2 | 1 | awaiting_confirmation |
| 直接入口，不同 cwd，使用现有模型配置 | live | 问卷 8 × 5 | 2 | 1 | awaiting_confirmation |

直接入口测试和在线验收均调用 `run_labweaver.main()`，即 Run Python File 执行的主流程；没有通过 GUI 点击验证 F5。VS Code 配置完成 JSON 检查，在线运行的原始控制台输出和 JSON 留在被忽略目录，不发布。

新增验证覆盖配置优先级、来源相对路径、任务来源切换、仅配置即可运行、旧命令拒绝、编码及分隔符通过 intake 使用、缺失 CSV/任务/配置文件、冲突参数和非法配置。共享执行路径还验证：不读取无关 cwd 的 `.env`、模型构造异常不回显密钥或响应正文、空任务在构造模型前拒绝。在线问卷源文件哈希一致；CSV 工具原有成功和失败只读验证保留。

## 发布与下一轮

发布只包含源码、锁文件、公开配置、VS Code 配置、测试、文档和合成示例；不包含本地配置路径文件、凭据、真实用户数据、原始日志、虚拟环境和构建产物。

GitHub Actions 保留 Ubuntu/Windows × Python 3.11/3.12 四个环境，并显式选择对应 Python。CI 运行离线测试、默认问卷 intake 与实验覆盖示例，不提供模型密钥。每次发布结果可在 [Actions](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml) 核对。

[第三天开发计划](day3.md) 定义 Agent 主动检索本地 TXT、Markdown、文字型 PDF，使用 BM25 生成带真实出处的任务方案；这是下一轮待实现与待验收功能。多智能体留待分析与核验阶段接入。
