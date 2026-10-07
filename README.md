# LabWeaver

[![Offline verification](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml/badge.svg)](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml)

**让 Agent 根据你的任务读取 CSV、按需检索资料，并完成可核对的只读统计。**

LabWeaver 面向持有 CSV 数据、需要概览、筛选、分组聚合或排名的用户。用户提供 CSV、可选资料和任务说明；Agent 查看真实字段，判断是否需要资料解释，执行受控统计，并在影响结果的口径不明确时提问。回答问题后，它在同一运行中继续计算，完成后可以继续追问。

英文名 **LabWeaver** 表示将任务、资料、数据和成果组织成项目。仓库：[pypypaoying/LabWeaver](https://github.com/pypypaoying/LabWeaver)。

## 当前能力

- CSV 自动识别编码与分隔符；严格读取完整文件，概览和分析共享一个只读数据快照。
- 通用只读统计：按列位置筛选、分组、计数、求和、均值、最小值、最大值、排序和前 K 项。
- 可选 Agentic RAG：TXT、Markdown、文字型 PDF，本地 BM25 检索，实际命中片段保留哈希及页码/行号。
- 连续对话：`ask_user` 通过 LangGraph interrupt 暂停，用户输入后恢复同一任务；完成后可继续提出新任务。
- Harness：绑定文件、工具白名单、执行顺序、跨恢复调用预算、调用 ID 去重、实际执行证据及引用校验。
- 输出严格 JSON、Markdown 简报与独立结果 CSV；结果表由程序根据工具结果生成，源数据不修改。

当前为单 Agent。多智能体、清洗、任意 Python/SQL 执行、建模、绘图及关闭程序后恢复尚未接入。当前进程的内存 checkpointer 支持中断恢复，JSON 文件只是运行记录。

2026-10-07 验收：369 项离线测试、28 项子测试及十二项一键成果验收通过；真实奖牌 CSV 的全部年份与 2024 年前五在线结果匹配参考答案，CSV＋PDF 完成实际检索及来源引用。详细口径与验证记录见 [本轮开发记录](docs/session-statistics.md)。

## 首次安装

需要 Python 3.11 或 3.12，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。锁文件固定依赖。

```shell
git clone https://github.com/pypypaoying/LabWeaver.git
cd LabWeaver
uv sync --locked --python 3.11
```

## 在 VS Code 直接运行

1. 用 VS Code 打开整个项目，选择项目 `.venv` 的 Python 解释器。
2. 在项目 `.env` 填写模型配置，或用被 Git 忽略的 `labweaver.local.toml` 引用已有配置文件。
3. 打开 `run_labweaver.py`，点击 Python 扩展的 **Run Python File**；也可按 F5 选择 **LabWeaver: conversation**。
4. 在集成终端输入 **CSV 路径、可选资料路径、任务说明**。多份资料用分号分隔；资料留空即可。
5. Agent 有关键问题时，直接输入回答。结果出来后可继续追问；回车或输入“退出”结束。

日常无需填写编码和分隔符。遇到不能可靠识别的格式，程序显示候选预览并请你选择；它不会忽略坏字节或将畸形多列文件当作单列继续分析。日志记录实际格式及识别方式，用户选择的候选记录为确认后的格式。

例如：

```text
任务：统计获得最多奖牌的前 5 个国家
确认回复：全部年份、按 Total 累计、保留原始 NOC
继续追问：改为 2024 年
```

任务明确时可以直接写“统计全部年份各原始 NOC 的 Total 累计值，降序取前 5”，避免多余确认。提供 PDF 后，若任务写“基于竞赛说明……”，Agent 应真实检索相关说明；仅提供 PDF 不强制检索。没有资料时，可独立完成的 CSV 统计继续执行，未知业务规则才需要确认。

`run_real_checks.py` 保留为兼容入口，使用同一连续对话流程；不再强制运行“仅 CSV”和“CSV+PDF”两遍。正常使用无需修改 Python 中的路径或任务。

控制台显示状态、解析设置、实际工具执行、问题与回答，并显示运行产物路径。记录按 `runs/<session_id>/<task_id>/` 保存；待回答事件保存 JSON，完成事件保存 JSON、Markdown，计算任务另保存结果 CSV。真实数据与原始日志被 Git 忽略。

## 运行配置与模型

`labweaver.toml` 为默认配置；VS Code 的终端输入会选择本轮文件与任务。高级配置仍可显式覆盖编码和分隔符。

```toml
[intake]
csv = "examples/data/survey.csv"
task_file = "examples/tasks/survey_rag.txt"
materials = ["examples/materials/survey/requirements.md"]
mode = "live"
encoding = "auto"
delimiter = "auto"
sample_rows = 5
output_dir = "runs"
```

`task = "比较不同部门的满意度均值"` 可代替 `task_file`，同一配置层不能同时设置两项。TOML 相对路径以文件目录为基准，CLI 覆盖路径以当前目录为基准。

配置优先级：入口覆盖 > 同目录 `labweaver.local.toml` > 公开 TOML > 内置默认值。模型配置默认读取选定配置目录的 `.env`；环境变量优先。

```dotenv
LLM_API_KEY=your-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL_ID=your-model-id
LLM_TIMEOUT=30
```

已有配置只需在本地文件引用，不复制密钥：

```toml
# labweaver.local.toml — 被 Git 忽略
[intake]
env_file = "/path/to/existing.env"
```

Windows 路径可用 `D:/your-folder/existing.env`。公开 TOML 不放 API key 或个人路径。`mode = "offline"` 使用确定性离线测试模型，不读取凭据、不发网络请求；它走真实工具流程，不能代表真实 LLM 的任务理解能力。

在线接口需要支持 Chat Completions 工具调用。每次请求默认超时 30 秒、禁止自动重试。任务、工具统计、最多五行样例及实际检索命中片段会发送到所配置接口；`sample_rows = 0` 可关闭样例。

## 云端终端或服务器

当前没有托管网页服务；这里指检出工程后运行 Python。先设置配置，通常只需：

```shell
uv run labweaver intake
uv run labweaver intake --offline
```

临时更换配置或资料时：

```shell
uv run labweaver intake --config examples/configs/experiments.toml
uv run labweaver intake --task "根据资料解释满意度字段" --material /path/to/variables.pdf
```

CLI 是单次调用入口，结果 JSON 输出到标准输出，产物路径输出到标准错误。若状态为 `awaiting_input`，使用 VS Code 对话入口回答并继续；JSON 本身无法恢复已关闭的进程。退出码：`0` completed，`1` error/cancelled，`2` 配置或读写失败，`3` awaiting_input。

## 工具与约束

`profile_csv(path, *, encoding="auto", delimiter="auto", sample_rows=5)` 仍可作为独立 Python 函数使用，无面向用户的 profile 命令。

| 项目 | 规则 |
| --- | --- |
| 编码 | BOM → 严格 UTF-8 → charset-normalizer 推断；不能可靠区分时询问，不替换坏字节 |
| 分隔符 | 逗号、分号、Tab、竖线；引号感知检测，确定格式后全量严格校验 |
| 缺失 | 空白单元格缺失；`0`、`NA`、`NULL` 保留 |
| 列 | 原始列名保留，重复/空列名用一基列位置区分 |
| 类型 | 类型只是推断；前导零及长整数字符串作为文本保留 |
| 统计 | 数值聚合排除缺失，未排除的非法数值报错；整数计数和求和保持整数精度 |
| 排名 | `top_k` 上限 100，同分按分组标签稳定排序；历史标签默认不合并 |
| 资源 | CSV 10 MiB、10 万行、200 列、单字段 64 KiB；错误不输出部分全量统计 |
| 资料 | 最多十份，单文件 5 MiB、总计 20 MiB、PDF 一百页；扫描件没有 OCR |
| 检索 | 首次搜索才建索引，每次最多三个真实片段，无匹配返回空 |
| 预算 | 每任务模型最多 12 次、概览一次、检索三次、分析四次、澄清三次；resume 不重置 |

`analyze_csv(spec)` 只接受结构化请求：`filters`、`group_by`、`metrics`、`order_by`、`top_k`。它绑定本会话快照，不接受新文件路径。只有真实分析结果才能完成计算型任务，提出分析计划不会替代计算结果。分组口径、筛选与实际行数保存在 JSON。

提供资料只让 `search_materials(query)` 可用。未提供资料时，模型看不到搜索工具。Deep Agents 自带文件、委派等工具被中间件隐藏，执行边界拒绝白名单外请求；模型不能读任意本地路径。

引用必须来自实际检索返回的片段，附原文件位置。引用校验确认出处真实，不自动证明回答的每句话都有资料支持。结果应结合 JSON 的真实表格和统计口径核对。

## Python 接口与代码结构

```python
from labweaver.agent import create_session

session = create_session("my_data.csv", model, material_paths=["instructions.pdf"])
report = session.invoke("统计前五")
if report["status"] == "awaiting_input":
    report = session.resume("全部年份、按 Total 累计、保留原始 NOC")
report = session.invoke("改为 2024 年")
```

状态为 `completed`、`awaiting_input`、`error` 或 `cancelled`。一次会话固定 thread_id；新任务重置任务预算，已有确认与对话仍保留。更换 CSV 应创建新会话。`run_intake()` 保留为单次调用包装。

```text
run_labweaver.py        # VS Code 输入、恢复与追问循环
run_real_checks.py     # 兼容入口
labweaver.toml         # 公开默认配置
src/labweaver/
  cli.py               # 单次入口与参数覆盖
  run_config.py        # TOML 分层、路径和配置校验
  config.py            # 模型配置与 ChatOpenAI 适配
  agent.py             # AgentSession、Harness、interrupt 与证据校验
  offline.py           # 不联网的确定性工具调用模型
  materials.py         # 资料解析、来源和 BM25 检索服务
  tools/               # CSV 快照、自动格式读取、概览与统计
  runtime/intake.py    # 模型选择、共享会话与报告保存
  runtime/records.py   # 严格 JSON、Markdown 与结果 CSV
```

## 如何检验

在 VS Code 的测试面板运行全部 pytest 测试；也可在终端执行：

```shell
uv run --frozen pytest -q
```

也可打开 `verify_d3.py`，点击 **Run Python File** 做一键离线验收。文件名作为兼容入口保留，现验证按需检索、澄清后实际排名、2024 追问、结果 CSV、输入哈希与零网络请求；全部完成显示 **Passed: 12/12**。

重点检查“成果”，不要只看 completed：核对 `analysis_results` 的实际表格与 `spec`、执行账本中的 `analyze_csv`、筛选年份、输入哈希、真实引用来源，以及 pending → resume → 新任务的过程。步骤和本地奖牌案例参考答案见 [连续统计验收](docs/session-statistics.md) 与 [真实数据检验](docs/real-data-checks.md)。

[GitHub Actions](https://github.com/pypypaoying/LabWeaver/actions) 在 Ubuntu/Windows × Python 3.11/3.12 四个环境验证离线流程，不提供模型密钥。历史 D2/D3 验收记录分别在 [第二天记录](docs/day2.md) 与 [D3 记录](docs/day3-results.md)，其中旧终态与功能范围已被本次更新替代。
