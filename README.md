# LabWeaver

[![Offline verification](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml/badge.svg)](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml)

**提供 CSV、可选资料和任务，获得可核对的统计、图表与报告。**

分析主 Agent 理解任务、澄清计算口径、按需检索资料并完成真实统计；可视化 Agent 查看授权的统计数据，选择图型并调用受控绘图工具。宿主程序负责完整文件读取、确定性计算、预算、执行证据和产物保存。角色提示词不绑定行业、字段或固定任务。

## 安装和运行

需要 Python 3.11 或 3.12，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```shell
git clone https://github.com/pypypaoying/LabWeaver.git
cd LabWeaver
uv sync --locked --python 3.11
```

在项目 `.env` 中填写支持工具调用的 Chat Completions 接口：

```dotenv
LLM_API_KEY=your-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL_ID=your-model-id
LLM_TIMEOUT=30
```

VS Code 打开整个项目，选择 `.venv` 解释器。打开 **`src/labweaver/app.py`**，点击 **Run Python File**；或按 F5 选择 **LabWeaver: conversation**。保持终端工作目录为项目根目录，正常使用只输入 CSV 路径、可选 TXT/MD/文字型 PDF 路径和任务。多份资料用分号分隔；资料留空即可。Agent 提问时直接回答，完成后继续追问，回车或输入“退出”结束。

```text
任务：按部门计算满意度均值，并绘制柱状图。
继续追问：把刚才的图改为横向柱状图。
```

默认配置从当前工作目录的 `labweaver.toml` 读取。设置 `mode = "offline"` 可使用确定性双模型演示，不读取模型凭据或联网；离线模型验证真实工具流程，不能代表真实模型的理解能力。详细配置、中文字体和更多任务见 [使用说明](docs/usage.md)。

## 能力与产物

- 自动识别 CSV 编码和分隔符，严格读取完整文件；概览和统计共享只读快照。
- 按列位置筛选、分组、计数、求和、均值、最小值、最大值、排序和前 K 项。
- 本地 BM25 检索可选资料，引用保留文件哈希及页码或行号。
- 柱状图、折线图和直方图，每张图一个指标，每任务最多两张；保存 PNG、SVG 和绘图数据 CSV。
- 同一进程内暂停、回复、恢复和连续追问，跨恢复预算与调用 ID 去重。
- JSON 记录真实工具证据，Markdown 简报包含实际表格和图表。源 CSV 与资料保持只读。

主 Agent 按任务决定绘图需要，简单查询或明确只要表格时无需委派。直方图从完整快照计算分箱；Top K 表格和五行样例不能替代总体分布。失败会留下错误记录，只有宿主登记的真实图片才计为已交付。

暂不支持任意 Python/SQL、数据清洗、建模、网页界面、多指标组合图、日期重采样或程序退出后恢复。JSON 是运行记录，内存检查点只在当前进程有效。

## CLI 与 Python

CLI 保留单次 intake 接口：

```shell
uv run labweaver intake --offline
uv run labweaver intake --config examples/configs/experiments.toml
uv run labweaver intake --csv my_data.csv --task "按类别汇总数值，并画柱状图"
```

`awaiting_input` 的退出码为 3，需要连续回答时使用上述 VS Code 正式入口。

```python
from labweaver.agent import create_session

session = create_session("my_data.csv", model, material_paths=["instructions.pdf"])
report = session.invoke("按类别计算数值均值，并画柱状图")
if report["status"] == "awaiting_input":
    report = session.resume("排除空白，保留原始类别")
saved_report = session.save(output_dir="runs")
report = session.invoke("将刚才的图改为横向柱状图")
```

高级接口可传入独立 `visualization_model` 和 `font_path`，默认复用主模型配置。

## 验证与仓库

```shell
uv run --frozen pytest -q
```

测试全部位于 `tests/`，包含真实工具成果、暂停恢复和产物的离线验收。`examples/` 只保留问卷和实验两套公开合成演示；奖牌与检索标注等回归材料位于 `tests/fixtures/`。

仓库保留包内主程序、标准测试、三份现行文档、公开配置、依赖锁文件和 CI。个人数据、凭据、本地配置、原始日志、图片产物、缓存和一次性检查脚本不发布。同步 GitHub 前逐项检查暂存文件归属，并核对远程目录和 CI。

[架构](docs/architecture.md) · [使用](docs/usage.md) · [验收](docs/verification.md)
