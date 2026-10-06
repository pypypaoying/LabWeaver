# LabWeaver

[![Offline verification](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml/badge.svg)](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml)

**结合任务说明、CSV 与项目资料，生成带出处的分析方案和待确认问题。**

LabWeaver 面向开展课程项目、数模或初步科研的大学生与初级研究人员。用户已有数据，但尚未明确字段含义、缺失口径和分析目标；Agent 先检查实际数据，再提出下一步的候选方向。同一工具适用于问卷、实验记录等不同表结构，无需为每份 CSV 改代码。

英文名 **LabWeaver** 表示将任务、资料、数据和成果组织成项目。仓库：[pypypaoying/LabWeaver](https://github.com/pypypaoying/LabWeaver)。

## 当前能力

- 单 Agent intake：先执行只读 `profile_csv`，再主动检索项目资料，输出带引用的任务方案。
- 本地 RAG：UTF-8 TXT/Markdown、文字型 PDF，BM25 与中文分词，片段保留哈希和行号或页码。
- 通用 CSV 工具：行列数、逐列缺失、推断类型、有限数值摘要和最多五行原始样例。
- 应用 Harness：文件绑定、工具白名单、先概览后检索、独立调用预算、多工具结果配对与引用校验。
- VS Code 与 CLI 共用运行配置；在线模型和确定性离线验证分别标记，保存严格 JSON 证据。

用户入口只保留 `intake`，成功终态为 `awaiting_confirmation`。D3 资料辅助规划已实现，成功时保存 JSON 和 Markdown 简报。多智能体、清洗、分析执行、图表和 checkpoint 恢复仍属后续能力。详见 [第三天实现](docs/day3.md)。

## 首次安装

需要 Python 3.11 或 3.12，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。依赖由 `uv.lock` 固定。

```shell
git clone https://github.com/pypypaoying/LabWeaver.git
cd LabWeaver
uv sync --locked --python 3.11
```

## 本地：在 VS Code 直接运行

安装依赖后，平时无需输入运行命令：

1. 用 VS Code 打开整个项目文件夹，选择项目 `.venv` 的 Python 解释器。
2. 修改根目录 `labweaver.toml` 中的 CSV、任务文件、materials 资料列表和模式。
3. 在项目 `.env` 填写模型配置，或在被 Git 忽略的 `labweaver.local.toml` 中引用已有配置文件。
4. 打开 `run_labweaver.py`，点击 Python 扩展的 **Run Python File**；也可按 F5 选择 **LabWeaver: run intake**。

公开配置的默认模式是 `live`。暂时只验证工程流程时，将 `mode` 改为 `"offline"`，不需要模型配置。

控制台显示状态、模型调用、实际工具执行、数据规模与 Agent 回答；完整记录写入 `runs/`。直接入口按自身所在项目目录读取配置，不依赖终端当前目录。F5 提供 Windows、Linux 和 macOS 的项目虚拟环境路径；已安装的 Python 扩展负责 Run Python File。

## 运行配置

`labweaver.toml` 是两种入口共用的公开默认值：

```toml
[intake]
csv = "examples/data/survey.csv"
task_file = "examples/tasks/survey_rag.txt"
materials = [
  "examples/materials/survey/requirements.md",
  "examples/materials/survey/variables.txt",
  "examples/materials/survey/methods.pdf",
]
mode = "live"
encoding = "utf-8-sig"
delimiter = ","
sample_rows = 5
output_dir = "runs"
```

任务可用 `task = "比较不同组别的测量结果"` 替代 `task_file`，同一配置层内不能同时设置两项。`materials = []` 保留 D2 的纯 CSV 流程。`run_labweaver.py` 的 `MATERIAL_PATHS = None` 继承 TOML，列表可临时覆盖，空列表关闭资料检索。中文 GB18030 CSV 可设置 `encoding = "gb18030"`；该设置不改变 TXT/MD 资料必须采用 UTF-8 的要求。分隔符可用 `";"` 或 `"tab"`，默认不猜测。

配置优先级为：**入口覆盖 > 同目录的 labweaver.local.toml > 公开 TOML > 内置默认值**。TOML 内相对路径以该文件目录为基准；CLI 覆盖路径以当前目录为基准。CLI 默认读取当前目录的 `labweaver.toml`；未找到时使用内置默认值，显式选择的配置文件不存在则报错。

模型凭据默认读取选定运行配置目录下的 `.env`，也可用 `LLM_*` 环境变量。已有配置只需引用，不复制密钥：

```toml
# labweaver.local.toml — 被 Git 忽略；此处仅存路径
[intake]
env_file = "/path/to/existing.env"
```

Windows 可写 `"D:/your-folder/existing.env"`。请将自己的路径写入本地文件；仓库中没有个人配置路径。TOML 不存 API key。相应 `.env` 内容如下（可参考 `.env.example`）：

```dotenv
LLM_API_KEY=your-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL_ID=your-model-id
LLM_TIMEOUT=30
```

环境变量优先于所选 `.env` 的同名值。离线模式不读取模型凭据。

## 云端终端或服务器：使用 CLI

这里指在服务器、云端开发环境中检出工程后运行 Python，当前项目没有托管网页服务。首次安装步骤同上，随后填写 `.env`（或设置环境变量），修改 `labweaver.toml`。在线运行只需：

```shell
uv run labweaver intake
```

无需 API 的离线流程验证：

```shell
uv run labweaver intake --offline
```

只有临时覆盖配置时才加参数，例如：

```shell
uv run labweaver intake --config /path/to/labweaver.toml
uv run labweaver intake --config examples/configs/experiments.toml
```

CLI 将结果 JSON 写到标准输出，运行记录位置写到标准错误；选项可通过 `uv run labweaver intake --help` 查看。`--material` 可重复传入以替换资料集合；`--offline` 和 `--live` 覆盖模式；`--env-file` 覆盖凭据文件路径。实验示例配置默认离线；在线使用该配置时，凭据应位于所选配置目录、由环境变量提供，或显式覆盖 env_file。

离线模式是确定性测试模型：通过真实 Deep Agents 流程调用工具，接收匹配的工具消息后回答，不发出网络请求；它不代表真实 LLM 理解能力。

在线使用支持 Chat Completions 工具调用的 OpenAI 兼容接口。每次请求默认超时 30 秒、禁止自动重试；有资料时最多 5 次模型调用、1 次 CSV 和 2 次检索，图步数 16；无资料时保持 3 次模型调用和 1 次 CSV。接口不支持工具调用、连接失败或证据不匹配返回 `error`。任务、统计、样例、资料标识与检索命中原文会发送到所配置的接口；`sample_rows = 0` 可关闭 CSV 样例。

## 不用命令行检验 D3

打开根目录 `verify_d3.py`，选择项目解释器并点击 **Run Python File**。它禁用网络、无需模型密钥，检查七个标注查询、问卷/实验/替代资料/无答案的真实 Agent 流程、文件哈希和简报保存；全部通过显示 12 项 PASS。

在线检验仍运行 `run_labweaver.py`，保持 `mode = "live"`。成功应满足 `status = "awaiting_confirmation"`、`materials_completed = true`，执行账本先 profile_csv 后 search_materials，引用 ID 来自 retrieved_chunks；控制台显示同 stem JSON 与 Markdown 路径。默认问卷 8×5，实验 6×6。

仅有正确引用 ID 不代表建议逐句正确。请检查方法是否符合资料要求；把问卷 requirements.md 换成 requirements_alternative.md，再用同一 CSV 运行，应出现总体分布目标和部门比较目标的冲突，而数据统计保持一致。

VS Code 测试面板已配置 pytest，可点击运行全部测试。完整步骤和真实验收见 [D3 结果与检验方法](docs/day3-results.md)。

## CSV 工具规则

`profile_csv(path, *, encoding="utf-8-sig", delimiter=",", sample_rows=5)` 保留为独立于模型的 Python 工具函数，由 Agent 调用；无独立概览命令。

| 项目 | 规则 |
| --- | --- |
| 记录 | 第一条非空记录为表头；仅计数据记录，完全空行跳过并提示，引号内换行仍属同一记录 |
| 缺失 | 空白单元格计缺失；`0`、`NA`、`NULL` 保留；比例分母为全部数据记录 |
| 列名 | 原样保留，重复或空列名按从 1 开始的 `position` 区分 |
| 类型 | `numeric`、`text`、`mixed`、`empty` 均为推断；前导零及至少 16 位整数字符串保留为文本 |
| 数值 | 仅有限数值及缺失组成的列计算 min/max/mean；均值排除缺失 |
| 非有限数值 | `NaN`、`Inf`、超出浮点范围的文本保留并警告，不产生该列数值摘要 |
| 样例 | 最多前五行解析后的原始字符串列表；统计始终来自全部已接受记录 |
| 仅表头 | 返回零行与提示；无数据时缺失比例为 JSON `null` |
| 错误 | 空输入、解码失败、未闭合引号、行宽不一致和超限明确报错，不返回部分统计 |
| 资源 | 文件 10 MiB、100,000 数据行、200 列，单字段 64 KiB UTF-8 字节且受字符上限约束 |

解析使用标准库 `csv.reader(strict=True)`，源文件只读。推断类型不验证业务 schema、唯一性或方法适用性。Agent 的工具结果另限制为 64 KiB UTF-8 JSON；超过时可关闭样例再运行，若列元数据仍超限则受控失败。

## Harness 与证据

模型首次只见 `profile_csv`；概览成功后有资料时才开放 `search_materials(query)`。两个工具都绑定本次输入，模型不能传文件路径。中间件过滤 Deep Agents 内置文件、规划与委派工具，执行边界拒绝白名单外请求；自动摘要和结果卸载关闭。

成功要求：恰好一次实际概览、全部工具调用 ID 与 ToolMessage/账本匹配、概览完成、存在最终回答。有资料时还要求实际检索、五部分输出和有效引用；无命中须明确资料不足。JSON 另保存资料哈希、查询、片段和引用；成功资料接入额外保存 Markdown 简报。未完整读取 CSV 时哈希为 `null`。

证据校验验证执行过程；自然语言回答仍可能有误，应以 JSON 的 `profile` 字段统计为准。记录不具备 checkpoint 恢复能力。退出码：`0` 等待确认，`1` 受控执行失败，`2` 配置、任务读取或记录写入失败。

`.env`、`labweaver.local.toml`、虚拟环境、缓存、本地真实数据和原始日志均被 Git 忽略。仓库只有合成示例，CI 不使用模型密钥。

## 验收

第二天原始基线（2026-10-02，Python 3.11.15）：59 项测试及 28 个子用例通过，wheel 和源码包构建成功。

| 原始基线验证 | 问卷 CSV | 实验 CSV |
| --- | --- | --- |
| 数据规模 | 8 行 × 5 列 | 6 行 × 6 列 |
| 手算核对 | experience_years：17/7；satisfaction：27/7 | loss：2.97/5；temperature_c：149/6；epoch_count：100/6 |
| 离线工具循环 | 1 次工具、2 次模型调用 | 1 次工具、2 次模型调用 |
| 真实 API | 1 次工具、2 次模型调用，awaiting_confirmation | 1 次工具、2 次模型调用，awaiting_confirmation |
| 源文件 | 哈希未改变 | 哈希未改变 |

原始验收见 [第二天记录](docs/day2.md)。入口改造验收（2026-10-03）118 项测试及 28 个子用例通过，包构建成功；直接入口的真实 API 验证为 2 次模型调用、1 次工具执行、`awaiting_confirmation`，源 CSV 哈希未变。详情见 [入口更新记录](docs/intake-entry-update.md)。

D3 验收（2026-10-07）：**247 项测试、28 个子用例通过**，一键离线验收 **12/12**，包构建成功。七个标注查询、四组离线接入及三组在线资料接入结果见 [D3 记录](docs/day3-results.md)。在线问卷更换资料后候选方案和冲突问题变化，数据统计与输入哈希一致；原始在线记录不发布。

```shell
uv run --frozen pytest -q
uv build
```

[GitHub Actions](https://github.com/pypypaoying/LabWeaver/actions) 保留 Ubuntu/Windows × Python 3.11/3.12 四个离线验证环境，运行测试和问卷、实验 intake 示例。

## 代码阅读

```text
run_labweaver.py       # VS Code 直接入口
verify_d3.py           # 无网络、无需密钥的 D3 验收入口
labweaver.toml        # 两种入口共用的公开配置
src/labweaver/
  cli.py              # intake 参数覆盖与 JSON 输出
  run_config.py       # TOML 分层与路径、类型校验
  config.py           # 模型凭据、超时与 ChatOpenAI 适配
  materials.py        # 资料解析、来源、BM25 检索
  tools/csv_profile.py # 独立只读的 CSV 统计
  agent.py            # Deep Agent、Harness、证据与状态
  offline.py          # 确定性离线工具循环
  runtime/intake.py   # 任务读取、模型选择、执行与保存
  runtime/records.py  # 严格 JSON 运行产物
```

下一轮在用户确认后接入受控分析执行，再评估分析和核验子 Agent 的必要性。详见 [第三天实现](docs/day3.md) 和 [架构路线](docs/architecture.md)。
