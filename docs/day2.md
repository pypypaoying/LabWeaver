# 第二天开发记录：LabWeaver 接入基线

日期：2026-10-02。英文名 LabWeaver，建议仓库名 `labweaver`；发布目标为 `pypypaoying/Agent-project`。

## 完成的用户流程

输入本地 CSV 与任务说明，工具读取实际文件并统计，Agent 基于结果给出任务摘要、分析方向与待确认问题，终态为 `awaiting_confirmation`。更换问卷与实验记录时，工具和 Agent 代码都无需修改。

本次使用 Deep Agents 单 Agent。应用 Harness 是执行约束与证据核验层：只开放绑定文件的 CSV 工具，限制调用预算，并校验真实执行结果。RAG、多智能体和分析工具列入后续路线，不包含空实现。

## 工程交付

- 可安装的 `src` 包、CLI、显式模型配置、依赖锁文件与包构建。
- 标准库严格 CSV 解析，逐列缺失统计、保守类型推断、有限数值摘要和最多五行样例。
- 表头按位置保留；前导零编号、中文、引号内逗号和换行可正确处理。
- 工具和模型预算、工具白名单、运行证据核验、受控失败与严格 JSON 记录。
- 两份合成示例、离线工具循环测试、跨系统与 Python 版本的 GitHub Actions。

## 实际验收

环境：Windows，Python 3.11.15，uv 0.11.12；锁定 Deep Agents 0.7.21。

`uv run --frozen pytest -q`：59 个测试、28 个子用例通过。`uv build --offline` 成功产出 wheel 与源码分发包。

### 手算与自动核对

| 数据 | 已核对内容 |
| --- | --- |
| survey.csv | 8 行、5 列；experience_years 缺失 1/8、均值 17/7；satisfaction 缺失 1/8、均值 27/7；comment 缺失 2/8；001 等编号保留为文本 |
| experiments.csv | 6 行、6 列；loss 缺失 1/6，min=0.48、max=0.8、mean=0.594；temperature_c mean=149/6；epoch_count mean=100/6 |

测试覆盖错误编码、空文件、仅表头、重复/空表头、行宽不一致、未闭合引号、非有限数字和各资源上限。均值使用浮点数整数比累加，验证极大正负数抵消后的小残差；非零数值下溢时保留文本并警告。成功与失败执行均核对源文件内容未改变。完整读取后的错误记录保留该快照哈希，超限文件不生成伪全量统计或完整哈希。

### 离线流程

确定性模型通过真实 Deep Agents 工具分发调用 `profile_csv`，接收带匹配 ID 的 `ToolMessage` 后回答。两种数据均为 2 次模型调用、1 次实际工具执行，并进入 `awaiting_confirmation`。

测试检查只绑定 `profile_csv`、调用次数、执行账本与 ID 配对，并阻断网络。未调用工具、非法工具、重复尝试、错误 ID、伪造结果、缺少最终回答、模型失败及过大工具输出都会得到失败状态。

### 真实 API

通过已有配置文件读取 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID`，使用 `ChatOpenAI` 适配兼容接口，没有复制密钥到工程。对两份合成 CSV 分别进行了真实调用：

| 数据 | 工具实际执行 | 模型调用 | 最终状态 | 失败原因 | 源文件哈希 |
| --- | --- | --- | --- | --- | --- |
| survey.csv | 1 次，成功 | 2 次 | awaiting_confirmation | 无 | 前后一致 |
| experiments.csv | 1 次，成功 | 2 次 | awaiting_confirmation | 无 | 前后一致 |

模型回答包含实际行列数、缺失概况、数值摘要和待确认问题，没有执行后续分析。该验收验证接口与工具循环可用；自然语言回答没有逐句自动核验，仍可能出现冗余问题或表述误差。原始运行记录保留在本地 `runs/`，不随仓库发布。

### 发布检查

发布内容包括工程、锁文件、测试、文档、CI 和合成样例。排除 `.env`、`.venv`、缓存、真实数据、构建产物与原始运行日志。Actions 的 Ubuntu/Windows × Python 3.11/3.12 矩阵执行离线测试与示例，结果可在仓库 Actions 查看。

## 下一天建议：带出处的任务资料接入

先让用户提供课程要求、实验说明或方法文档，并围绕“任务要求是什么、字段含义是什么、哪些方法被允许”检索。每条检索结果应保留文件、章节与片段标识；模型回答引用实际片段，资料缺失时说明无法确定。

验收至少包括可回答问题、资料无答案、相互冲突的要求及含指令文本的资料。与当前单 Agent 基线比较答案支持率、引用准确性与成本，再决定是否需要分析和核验子 Agent。

## 复现命令

```shell
uv sync --locked --python 3.11
uv run --frozen pytest -q
uv run labweaver profile --csv examples/data/survey.csv
uv run labweaver intake --csv examples/data/survey.csv --task "分析问卷数据" --offline
uv run labweaver intake --csv examples/data/experiments.csv --task-file examples/tasks/experiments.txt --offline
uv run labweaver intake --csv examples/data/survey.csv --task-file examples/tasks/survey.txt --live --env-file /path/to/existing.env
```
