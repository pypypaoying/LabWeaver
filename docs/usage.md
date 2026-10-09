# 使用说明

## VS Code 连续对话

在项目根目录运行 `uv sync --locked --python 3.11`，选择 `.venv` 解释器。打开 `src/labweaver/app.py` 点击 **Run Python File**，或 F5 → **LabWeaver: conversation**。终端工作目录保持项目根目录；F5 已配置该目录和 `labweaver.app` 模块。

1. 输入 CSV 完整路径；回车使用配置中的公开示例。
2. 输入可选 TXT/MD/文字型 PDF 路径，多份用分号分隔；回车不提供资料。
3. 输入任务，或 `@任务TXT完整路径`；回车使用示例任务。
4. 影响统计口径的问题直接回复，恢复后完成实际计算。
5. 完成后继续追问；回车或“退出”结束，待回答时“退出”取消当前任务。

日常不询问编码、分隔符或字体。不能可靠判断 CSV 格式时展示候选预览并等待选择，确定格式后严格读取全文件。任务 TXT 为 UTF-8，最多 64 KiB。更换 CSV 重新启动会话。

控制台显示状态、主/子模型调用、实际工具次数、解析设置、回答以及产物路径。运行记录默认保存到被 Git 忽略的 `runs/`。

对话默认流式运行：先显示 `[执行]` / `[结果]` 工具进度，模型生成回答时逐步显示正文。生成中的文字待最终校验；失败时明确标明本次未完成，不把未核验文字当成交付。仅展示回答正文，不展示内部推理、原始 JSON 或子 Agent 消息。不支持 token 流的模型仍可显示工具进度，正文由接口整段返回。

分组统计默认返回全部结果，不再限制 100 行。`top_k` 省略或设为 `null` 表示全部，只有用户要求“前 N 项”时设置正整数。终端最多预览 10 行，完整表格在结果 CSV、JSON 和 Markdown 中；终端预览不改变计算或导出。大结果给模型的每条结果消息最多 64 KiB，标明 `preview_only` 和分页游标，模型可调用 `read_analysis_rows(data_id, offset, limit)` 阅读下一页，不能把预览当成完整总体求和。原始 CSV 的资源限制仍保留。

## 任务示例

问卷公开 CSV：`examples/data/survey.csv`；相关资料位于 `examples/materials/survey/`。

```text
按 department 分组计算非空 satisfaction 均值，并绘制柱状图。
将刚才的图改成横向柱状图，标题为“部门满意度”。
对 experience_years 绘制 5 箱直方图，排除空白。
```

实验公开 CSV：`examples/data/experiments.csv`；相关资料位于 `examples/materials/experiments/`。

```text
按 run_name 汇总最小 loss，只返回表格。
根据资料解释 temperature_c 的含义与单位，并标明出处。
```

折线图需要数值或年份在前的日期字段，支持 `YYYY-MM`、`YYYY-M-D`、`YYYY/M/D`，例如 `2024/12/1`。程序按真实日期排序，绘图数据保留原始标签；不会猜测日/月顺序不明确的日期。重复日期仍需明确聚合口径。例如使用自己的 `month,value` 表格，任务为“按 month 汇总 value 的均值，绘制折线图”。日期派生、重采样、多指标组合图不在当前范围。

主 Agent 按任务判断图表需求。明确要求图表时执行；简单查询或只要表格无需委派。每图一个指标，每任务最多两张。直方图使用完整数据快照而非五行样例或 Top K 表格；空白排除，非法及非有限数值报错。

直方图默认 10 箱，可指定 1–50 箱等宽分箱；区间左闭右开，最后一箱包含最大值。记录有效数、缺失数和真实边界。恒定数值以该值 ±0.5 形成范围，再按要求分箱。

## 配置

默认从工作目录载入 `labweaver.toml`。优先级为入口/CLI 覆盖 > 同目录 `labweaver.local.toml` > 公开 TOML > 内置默认值。TOML 路径相对于配置目录；CLI 覆盖路径相对于当前目录。

```toml
[intake]
csv = "examples/data/survey.csv"
task = "按部门计算满意度均值，并画柱状图"
materials = []
mode = "live"
encoding = "auto"
delimiter = "auto"
sample_rows = 5
output_dir = "runs"
```

`task_file` 可代替 `task`，同一配置层不能同时使用。环境变量优先于选定配置目录的 `.env`。已有凭据文件可在被忽略的 `labweaver.local.toml` 中配置 `env_file`，无需复制密钥。

`mode = "offline"` 使用确定性双模型演示，真实读取数据、执行工具并保存成果；不读取凭据、不联网。它支持固定演示任务，不能替代在线模型质量验证。

绘图自动选中文系统字体，Windows 可匹配微软雅黑，Ubuntu 可安装 `fonts-noto-cjk`。需要覆盖时，在本地 TOML 设置 `font_path = "D:/fonts/chinese.ttf"`，或 CLI `--font-path`。字体缺失返回明确错误；不增加日常配置问题。

在线接口须支持 Chat Completions 工具调用，默认超时 30 秒，传输失败不自动重试。模型跳过当前必需工具时可收到一次纠正请求，计入原有模型调用预算；仍失败则记录错误。工具统计、至多五行样例、资料命中片段及授权绘图数据会发送至所配置模型接口；`sample_rows = 0` 关闭概览样例。

## CLI

```shell
uv run labweaver intake --offline
uv run labweaver intake --config examples/configs/experiments.toml
uv run labweaver intake --csv my_data.csv --task "按类别汇总指标，并画柱状图" --material instructions.pdf
```

CLI 只运行一次，JSON 到标准输出，产物路径到标准错误。退出码：0 完成，1 错误/取消，2 配置或文件读写失败，3 待回答。待回答使用包内对话入口运行，关闭进程后 JSON 无法恢复会话。

## Python 与产物

```python
from labweaver.agent import create_session

session = create_session(csv_path, model, material_paths=materials,
                         visualization_model=optional_other_model)
report = session.invoke(task)
if report["status"] == "awaiting_input":
    report = session.resume(reply)
saved = session.save(output_dir="runs")
report = session.invoke("把刚才的图改成横向柱状图")
# session.cancel() 用于中止当前任务。
```

Python 调用可用 `session.invoke(task, on_event=handler)` 和 `session.resume(reply, on_event=handler)` 接收流式事件；`handler` 是接收字典的函数。事件类型为 `tool_start`、`tool_end`、`answer_delta`（`text` 是新增正文）和 `validated`（最终 `status` 和已验证 `answer`）。未传入回调保持同步返回行为，CLI 的标准输出仍为完整 JSON。

`visualization_model` 可省略，默认使用主模型。`run_intake()` 保留单次报告包装；图表导出使用会话或 CLI/runtime。

记录在 `runs/<session_id>/<task_id>/`。完成事件包含 JSON、Markdown、实际结果 CSV，以及图表 PNG、SVG、绘图数据 CSV。Markdown 嵌入 PNG；JSON 记录图表规格、来源绑定、调用证据、路径、哈希与耗时。每次保存生成新文件，不覆盖先前产物。

CSV 限 10 MiB、10 万行、200 列、单字段 64 KiB；资料最多十份、单文件 5 MiB、总计 20 MiB、PDF 一百页，扫描件无 OCR。真实源文件保持只读。
