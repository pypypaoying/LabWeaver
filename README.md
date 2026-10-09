# LabWeaver

[![Verification](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml/badge.svg)](https://github.com/pypypaoying/LabWeaver/actions/workflows/tests.yml)

通用 CSV 代码分析助手：输入一份 CSV、可选 TXT/Markdown/文字型 PDF 和任务，Agent 编写 Python，在本地 Docker 中运行与修正，交付结果表、指标、PNG/SVG、分析代码和执行记录。源文件保持只读。

主 Agent 使用 Deep Agents 理解任务、按需检索、澄清并核对交付项；独立代码 Agent 使用 LangChain `create_agent`，通过 pandas、NumPy、Matplotlib 完成处理、计算和绘图。采用轻量计划＋ReAct 工具循环＋有限报错修正。固定统计与绘图工具已经替换。

## 安装与 VS Code 使用

1. 安装 Python 3.11/3.12、uv 和 Docker Desktop，启用 **Linux containers**。Windows 需要可运行的 WSL2 和硬件虚拟化；系统要求重启时先完成重启。
2. 在项目目录执行 `uv sync --locked`，VS Code 选择 `.venv` 的解释器。
3. 启动 Docker Desktop。在 VS Code 选择 **Terminal → Run Task → LabWeaver: Build analysis sandbox**，首次构建会下载锁定依赖和中文字体。也可一次执行：

   ```powershell
   docker build -t labweaver-python:0.2.0 sandbox
   ```

4. 将 `.env.example` 的设置填入忽略的 `.env`；也可在忽略的 `labweaver.local.toml` 中用 `env_file` 引用已有配置。使用 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID`。
5. 打开 `src/labweaver/app.py`，点击 **Run Python File** 或 F5。输入 CSV、可选资料和任务。关键歧义时输入回复，完成后可以继续追问。退出输入 `退出`。

任务示例：

> 按 department 汇总 satisfaction 的均值，空白排除，保留原始部门名称，绘制柱状图并导出完整结果。

> 将日期转为时间，按日汇总数量，计算峰值、谷值和差值，绘制趋势图；请先确认无法判断的日期格式。

文件保存到 `runs/<session>/<task>/`，包含报告 JSON/Markdown、每次尝试的 `.py`、CSV/指标 JSON、PNG/SVG 和绘图数据 CSV。模型接收分页预览，完整结果没有 100 行上限。未完成时保存已取得成果并明确缺少的交付项。

## 配置与接口

`labweaver.toml` 提供公开默认值，`labweaver.local.toml` 提供忽略的本地覆盖。代码模型输出默认 4096 tokens，可设置 `analysis_max_tokens`；执行超时默认 60 秒，可设置 `execution_timeout`。详细说明见 [使用](docs/usage.md)。

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
uv run --frozen pytest -q -m "not docker"
# 构建镜像并启动 Docker 后：
$env:LABWEAVER_REQUIRE_DOCKER="1"
uv run --frozen pytest -q -m docker
```

宿主测试和真实容器验收分别记录，CI 包含 Ubuntu/Windows × Python 3.11/3.12，以及 Ubuntu 的真实 Docker 测试。当前结果与本机环境状态见 [验收](docs/verification.md)，代码结构和权限见 [架构](docs/architecture.md)。

当前支持单 CSV、按需 RAG、当前进程内连续对话、数据处理/统计/绘图与脚本重放。容器关闭网络、使用非 root 和只读根目录，限制资源，不接收宿主密钥或项目目录。Docker 不可用时明确失败，不改用宿主 Python。执行与产物检查不自动证明所有分析口径和数值正确；验收还需独立基准。多表、机器学习、联网数据和长期 Notebook 内核留待后续。
