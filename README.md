# AnaliyMaster

<img src="frontend/public/analyvia.svg" width="64" height="64" alt="AnaliyMaster logo" />

AnaliyMaster 是本地优先的通用数据分析助手，原名 LabWeaver。它从自然语言任务生成可核对的分析报告、结果表、图表与代码。为兼容现有安装、凭据和历史，Python 包、`labweaver` 命令、Docker 镜像与应用数据目录继续使用原标识。GitHub 仓库地址保持可用，现有 logo 沿用。

[![Verification](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml/badge.svg)](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml)

通用 CSV 代码分析助手：输入一份 CSV、可选 TXT/Markdown/文字型 PDF 和任务，Agent 编写 Python，在本地 Docker 中运行与修正，交付结果表、指标、PNG/SVG、分析代码和执行记录。源文件保持只读。

主 Agent 使用 Deep Agents 理解任务、按需检索、澄清并核对交付项；独立代码 Agent 使用 LangChain `create_agent`，通过 pandas、NumPy、Matplotlib 完成处理、计算和绘图。采用轻量计划＋ReAct 工具循环＋有限报错修正。固定统计与绘图工具已经替换。

可选分析 Skills 提供数据探索、统计、可视化与交付检查指导，通过受控工具按需读取；实际计算仍在原 Docker 沙箱完成。技能已适配公开的 Anthropic Data Analyst Plugin，来源、修改和许可随包保留。当前默认关闭，可在 `[intake]` 中设 `skills_enabled = true` 后重启应用。使用与对照测试见[Skills说明](docs/usage.md#按需分析-skills)，加载技能不等于分析已通过核验。

[Skills外部基准对照](evaluation/results/skills-dabench.md)使用官方 InfiAgent-DABench 的24题公开验证子集，每题每组3次，共144次真实在线运行。答案匹配且交付完成：关闭61/72、开启64/72；开启组token约增加47.9%，95%探索性区间覆盖零差值，尚不能确认稳定收益，因此默认仍关闭。题目来自23种不同CSV内容，并未全面检验专业报告、RAG或建模能力。[原48次合成对照](evaluation/results/skills-pilot.md)保留为集成与回归证据，不与外部基准合并。

## 本地聊天界面

浏览器应用复用现有双 Agent 和 Docker 执行。支持上传 CSV、可选资料、流式执行事件、澄清回复、真实结果与下载，并保存基础聊天历史。一个对话固定一份 CSV；首次发送后附件冻结。重启后能查看历史并继续发起新任务，不能恢复已中断的内存检查点。

先按下面的步骤准备 Docker，再一次性安装界面依赖并构建；在线模型可直接在页面的设置中配置：

```powershell
uv sync --locked --extra web
cd frontend
npm ci
npm run build
cd ..
```

构建需要 Node.js；日常使用双击仓库根目录 `start-analiymaster.cmd`，或执行 `uv run --frozen --no-sync --extra web labweaver serve --open`。重复启动验证并复用同一项目的服务；页面仅通过受保护的回环 API 访问本机数据，后端不返回 API Key。关闭浏览器页面保留后台服务；完全退出使用左下角 **退出应用**或在原启动终端按 **Ctrl+C**，有任务时先完成或停止。历史与已登记成果保留。

输入框右下角可以切换服务商和模型，默认 DeepSeek。左下角 **设置** 可填写 API Key、Base URL 和模型列表，添加其他 OpenAI Chat Completions 兼容服务商，并测试连接或从接口获取模型列表。Windows 新凭据使用当前用户的 DPAPI 加密保存；已有官方 DeepSeek `.env` 可继续通过引用使用，不复制密钥。切换在下一次任务生效，同一任务的主/代码 Agent 使用同一模型；运行中或等待回复时暂时锁定。详见[模型设置](docs/usage.md#模型选择与服务商配置)。

聊天区展示真实工具、代码尝试、核验和保存进度，右侧执行面板提供完整代码、错误和检索出处；完整 CSV 下载与表格预览分开。数据库、附件与成果保存在用户应用数据目录，不进入仓库。搜索、工作空间、独立知识库和 Tauri 安装包尚未实现。应用设计见 [Desktop](docs/desktop.md)，操作及测试见 [使用](docs/usage.md)和[验收](docs/verification.md)。

在左侧对话的 **⋯** 菜单（也可右键）重命名、置顶或永久删除。置顶随重启保留；删除前需确认，活动任务须先停止。永久删除会清理本机应用保存的消息、导入副本、归属该对话的临时快照和运行成果，用户原文件保留；文件清理失败时明确提示并支持重试。

默认以分析报告展示结论：复杂任务使用章节、自然段、重点、核心结果表及真实图表；指标以可读表格呈现，原始 JSON 可展开或下载。长数组和矩阵明确标注预览范围，完整结果仍保留。超过浏览器安全整数范围的数值使用精确文本展示，完整精度以 JSON 下载文件为准。既有历史指标无需重跑即可使用新的展示方式。

运行时显示本任务用时与当前主／代码 Agent 模型等待时间，每秒保存宿主计时事件。等待用户澄清暂停主动执行计时；完成、停止或错误后冻结，刷新和重启不会从零计算旧记录。界面执行时间包括工作进程启动和成果保存，报告的 Agent 时间只统计 Agent 运行；模型请求累计用时另列。旧版没有计时证据的历史显示“未记录”，服务意外中断只保留最后实测时间并标记未记录尾段。

## 安装与 VS Code 使用

1. 安装 Python 3.11/3.12、uv 和 Docker Desktop，启用 **Linux containers**。Windows 需要可运行的 WSL2 和硬件虚拟化；系统要求重启时先完成重启。
2. 在项目目录执行 `uv sync --locked`，VS Code 选择 `.venv` 的解释器。
3. 启动 Docker Desktop。在 VS Code 选择 **Terminal → Run Task → AnaliyMaster: Build analysis sandbox**，首次构建会下载锁定依赖和中文字体。也可一次执行：

   ```powershell
   docker build -t labweaver-python:0.3.0 sandbox
   ```

4. 将 `.env.example` 的设置填入忽略的 `.env`；也可在忽略的 `labweaver.local.toml` 中用 `env_file` 引用已有配置。使用 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID`。
5. 打开 `src/labweaver/app.py`，点击 **Run Python File** 或 F5。输入 CSV、可选资料和任务。关键歧义时输入回复，完成后可以继续追问。退出输入 `退出`。

任务示例：

> 按 department 汇总 satisfaction 的均值，空白排除，保留原始部门名称，绘制柱状图并导出完整结果。

> 将日期转为时间，按日汇总数量，计算峰值、谷值和差值，绘制趋势图；请先确认无法判断的日期格式。

文件保存到 `runs/<session>/<task>/`，包含报告 JSON/Markdown、每次尝试的 `.py`、CSV/指标 JSON、PNG/SVG 和绘图数据 CSV。模型接收分页预览，完整结果没有 100 行上限。未完成时保存已取得成果并明确缺少的交付项。

## 配置与接口

`labweaver.toml` 提供公开默认值，`labweaver.local.toml` 提供忽略的本地覆盖。代码模型输出默认 12288 tokens，可设置 `analysis_max_tokens`；执行超时默认 60 秒，可设置 `execution_timeout`。详细说明见 [使用](docs/usage.md)。

复杂任务同样核对每项实际产物；只打印日志、模型响应截断或缺少成果不能完成。代码模型的格式修正最多一次，仍占原模型预算；代码修正最多三次。浏览器服务默认单次模型等待 180 秒、累计运行 900 秒，等待用户澄清暂停累计计时；超限停止本应用的工作进程与所属容器，保留合法部分成果。代码尝试仍受独立 Docker 资源限制。

保留配置驱动的 CLI（云端需本地 Docker daemon）：

```powershell
uv run labweaver intake
uv run labweaver intake --offline --task "概览字段和缺失情况"
```

离线模式使用确定性演示模型，走真实工具流程；涉及分析时同样需要 Docker。它不提供通用自然语言理解，也不把模拟产物当作真实执行。

```python
from labweaver.agent import create_session
from labweaver.runtime.execution import ExecutionConfig
from labweaver.runtime.replay import replay_analysis

session = create_session("selected.csv", model, analysis_model=code_model,
                         material_paths=["rules.pdf"], execution_config=ExecutionConfig())
report = session.invoke("用户的分析任务")
if report["status"] == "awaiting_input":
    report = session.resume("用户确认的口径")
saved = session.save()
# 不调用模型，输入哈希相同、原镜像仍存在时重放成功代码：
replayed = replay_analysis(saved["record_path"], "selected.csv")
```

## 验证与当前边界

```powershell
uv run --frozen --extra web pytest -q -m "not docker"
# 构建镜像并启动 Docker 后：
$env:LABWEAVER_REQUIRE_DOCKER="1"
uv run --frozen --extra web pytest -q -m "docker and not live"
```

宿主测试和真实容器验收分别记录，CI 包含 Ubuntu/Windows × Python 3.11/3.12，以及 Ubuntu 的真实 Docker 测试。当前结果与本机环境状态见 [验收](docs/verification.md)，代码结构和权限见 [架构](docs/architecture.md)。

当前支持单 CSV、按需 RAG、当前进程内连续对话、数据处理/统计/绘图与脚本重放。开发中的报告版本 3 增加任务约定、稳定源记录 ID、分类/回归成果协议和独立指标复算；沙箱加入 scikit-learn 1.7.2。界面分别展示交付状态、核验覆盖与接口实际 token，未检查部分明确标注。详见[可靠分析设计与边界](docs/reliability.md)。

本轮 v2 完整评测达到数值门槛：共同任务基线31/48，新版35/48；24次建模类任务均达到预定行为（包括应澄清的情形），二分类、多分类和Ridge核心案例各3/3。总体59/72，仍有13次未达成；无需内部修正即达成次数没有提升。见[v2结果与边界](evaluation/results/v2.md)及[原版失败记录](evaluation/results/initial.md)。结果仅代表固定公开合成任务，不能外推为任意分析正确性保证。

容器关闭网络、使用非 root 和只读根目录，限制资源，不接收宿主密钥或项目目录。Docker 不可用时明确失败，不改用宿主 Python。独立核验覆盖指定样本和指标，不能证明训练全程无泄漏、模型方法或全部解释正确。多表、模型保存与再次预测、联网数据和长期 Notebook 内核留待后续。
