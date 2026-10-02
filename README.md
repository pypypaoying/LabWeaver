# LabWeaver

[![Offline verification](https://github.com/pypypaoying/Agent-project/actions/workflows/tests.yml/badge.svg)](https://github.com/pypypaoying/Agent-project/actions/workflows/tests.yml)

**从任务说明和 CSV 开始，整理数据概览、分析方向与待确认问题。**

LabWeaver 面向需要开展课程项目、数模或初步科研的大学生与初级研究人员。用户往往已经有数据，但尚未明确字段含义、缺失口径和分析目标。本项目先帮助用户检查实际数据，再确认下一步该做什么。同一工具支持问卷、实验记录等不同表结构，无需针对每份 CSV 改代码。

英文名 **LabWeaver** 表示将任务、资料、数据和成果组织成项目；建议未来仓库名为 `labweaver`。当前 GitHub 地址保留为 [pypypaoying/Agent-project](https://github.com/pypypaoying/Agent-project)。

## 当前交付

- 独立、只读的通用 `profile_csv`：行列数、逐列缺失统计、推断类型、数值摘要和原始样例。
- 基于 Deep Agents 的单 Agent：实际调用工具后生成任务摘要、可开展的分析和待确认事项。
- 应用 Harness：文件绑定、工具白名单、调用预算、结果配对与失败处理。
- 明确区分离线模型验证与真实 API 调用；JSON 运行记录保存工具证据。

当前成功终态是 `awaiting_confirmation`，只表示项目接入完成。RAG、多智能体、清洗、图表、分析执行及 checkpoint 恢复是后续开发内容。

## 安装

需要 Python 3.11 或 3.12，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。依赖使用 `uv.lock` 固定，本地开发验证使用 Python 3.11.15。

```shell
git clone https://github.com/pypypaoying/Agent-project.git
cd Agent-project
uv sync --locked --python 3.11
uv run --frozen labweaver --version
```

## 直接概览 CSV

不需要模型或 API key：

```shell
uv run labweaver profile --csv examples/data/survey.csv
```

明确指定编码和分隔符，例如中文 GB18030、分号分隔的文件：

```shell
uv run labweaver profile --csv data/my_table.csv --encoding gb18030 --delimiter ";"
uv run labweaver profile --csv data/my_table.tsv --delimiter tab --sample-rows 0
```

默认使用 `utf-8-sig` 和逗号；不自动猜测编码、分隔符或修复异常行。

```python
from labweaver.tools import profile_csv

result = profile_csv("examples/data/survey.csv", sample_rows=5)
assert result["status"] == "completed"
print(result["row_count"], result["column_count"])  # 8 5
```

## 离线 Agent

离线模式使用确定性测试模型，通过真实 Deep Agents 流程发出 `profile_csv` 调用，接收实际工具消息后回答。它用于验证工程流程，不代表真实 LLM 的理解能力，且不会发出网络请求。

```shell
uv run labweaver intake --csv examples/data/survey.csv --task "分析问卷数据" --offline
uv run labweaver intake --csv examples/data/experiments.csv --task-file examples/tasks/experiments.txt --offline
```

更换文件与任务说明即可复用同一 Agent。CLI 将严格 JSON 输出到标准输出，将运行文件位置写到标准错误。默认记录保存到 `runs/`，可通过 `--output-dir` 修改。

## 真实模型

使用支持 Chat Completions 工具调用的 OpenAI 兼容接口。将 `.env.example` 复制为本地 `.env`，填入：

```dotenv
LLM_API_KEY=your-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL_ID=your-model-id
LLM_TIMEOUT=30
```

```shell
uv run labweaver intake --csv examples/data/survey.csv --task "分析问卷数据" --live
```

已有配置可通过 `--env-file` 读取，无需复制密钥：

```shell
uv run labweaver intake --csv examples/data/survey.csv --task-file examples/tasks/survey.txt --live --env-file /path/to/existing.env
```

环境变量优先于配置文件。每次模型请求默认超时 30 秒、禁止自动重试；每次 intake 最多 3 次模型调用、1 次工具尝试。接口不支持工具调用、连接失败或证据不匹配时返回 `error`，不会报告成功。真实模式会将任务说明、字段统计及最多五行样例发送到所配置的模型接口；`--sample-rows 0` 可关闭样例。

## CSV 规则与边界

| 项目 | 规则 |
| --- | --- |
| 记录 | 首条非空 CSV 记录作为表头；行数仅计数据记录。完全空行跳过并提示；引号内换行仍属于同一记录 |
| 缺失 | 空白单元格计为缺失；`0`、`NA`、`NULL` 保留，比例分母为全部数据记录 |
| 列名 | 原样保留，重复或空列名以从 1 开始的 `position` 区分 |
| 类型 | `numeric`、`text`、`mixed`、`empty` 均是推断；前导零和至少 16 位的整数字符串按标识符文本保留 |
| 数值 | 仅完全由有限数值和缺失组成的列计算 min/max/mean；均值分母排除缺失，使用浮点计算 |
| 非有限数值 | `NaN`、`Inf`、超出浮点范围的文本保留并警告，不产生该列数值摘要 |
| 样例 | 最多前五行，使用解析后的原始字符串列表；不会替代全量统计 |
| 仅表头 | 返回零行、空值类型及提示；无数据时缺失比例为 JSON `null` |
| 错误 | 空输入、解码失败、引号未闭合、行宽不一致或读取超限返回明确错误，不输出部分统计 |
| 资源 | 文件最多 10 MiB、数据记录最多 100,000、列最多 200；单字段最多 64 KiB UTF-8 字节且受字符上限约束 |

标准库 `csv.reader(strict=True)` 负责解析。CSV 无需符合特定领域 schema；数值类型推断不验证业务规则、实验成功、数据唯一性或统计方法适用性。

Agent 工具结果额外限制为 64 KiB UTF-8 JSON，以避免超大样例被框架转换为文件引用。超过时受控失败，可先使用 `--sample-rows 0`；独立 `profile` 命令仍保留完整概览。

## Harness 与运行证据

模型可见、可执行的工具只有绑定到本次文件的 `profile_csv`，该工具不接受路径参数。Deep Agents 自带的文件、规划与委派工具通过应用中间件过滤；即使模型自行请求其他工具，执行边界也会拒绝。

成功要求：恰好一次实际概览执行、工具调用 ID 与 `ToolMessage` 一一匹配、消息结果等于执行账本、概览完成、最终返回 AI 摘要。模型说“已检查”本身不构成执行证据。自动摘要与工具输出卸载关闭，避免绕过显式预算或丢失结果配对。

运行 JSON 包含任务、来源文件名与 SHA-256、调用轨迹、实际执行账本、最终回答、调用计数和状态。未能完整读取文件时哈希为 `null`。失败记录只保留受控错误与安全诊断。证据核验针对工具执行与运行状态，模型自然语言回答仍可能出现误差；应以 `profile` 中的实际统计为准。JSON 记录可用于排查与复核，未实现暂停后的恢复执行。

退出码：`0` 表示概览完成或 intake 等待确认；`1` 表示受控执行失败；`2` 表示 CLI 配置、任务读取或报告写入失败。

`.env`、虚拟环境、缓存、本地数据与原始运行记录均在 `.gitignore` 中。仓库只包含合成示例，CI 仅运行离线测试。

## 验收

2026-10-02 在 Python 3.11.15 下：**59 项测试、28 个子用例全部通过**，wheel 和源码包构建成功。

| 验证 | 问卷 CSV | 实验 CSV |
| --- | --- | --- |
| 数据规模 | 8 行 × 5 列 | 6 行 × 6 列 |
| 手算核对 | experience_years：17/7；satisfaction：27/7 | loss：2.97/5；temperature_c：149/6；epoch_count：100/6 |
| 离线真实工具循环 | 1 次工具、2 次模型调用 | 1 次工具、2 次模型调用 |
| 小规模真实 API | 1 次工具、2 次模型调用，`awaiting_confirmation` | 1 次工具、2 次模型调用，`awaiting_confirmation` |
| 源文件 | 执行前后哈希一致 | 执行前后哈希一致 |

测试还覆盖中文与 GB18030、引号内逗号/换行、前导零、重复表头、空输入、畸形记录、非有限数值、资源限制，以及未调用工具、非法工具、重复调用、ID 不匹配等失败路径。离线测试阻断网络，真实 API 验收单独进行；两次成功不代表所有兼容接口均可用。详情见 [第二天开发记录](docs/day2.md)。

```shell
uv run --frozen pytest -q
uv build
```

[GitHub Actions](https://github.com/pypypaoying/Agent-project/actions) 在 Ubuntu/Windows 与 Python 3.11/3.12 上安装锁定依赖并运行离线测试及示例，无需模型密钥。

## 代码阅读顺序

```text
src/labweaver/
  cli.py              # 命令、模式和运行记录
  config.py           # 显式配置与兼容模型适配
  tools/csv_profile.py # 独立的数据概览
  agent.py            # Deep Agent、Harness、证据与状态
  offline.py          # 确定性离线工具循环
  runtime/records.py  # 严格 JSON 运行记录
tests/                # 功能及失败路径
examples/             # 两种合成 CSV 与任务说明
docs/                 # 架构和每日开发记录
runs/                 # 本地运行产物，忽略提交
```

下一步优先接入用户提供的任务要求与方法资料：检索结果保留来源、章节与片段引用，评估问题是否能从资料得到支持，然后再扩展分析执行与核验子 Agent。详见 [架构与路线](docs/architecture.md)。
