# 使用与检验

## Windows / VS Code

首次安装 Docker Desktop 与 WSL2、启用 Virtual Machine Platform/硬件虚拟化，按系统要求重启。确认 Docker Desktop Linux 引擎已运行，再在 VS Code 的 Terminal → Run Task 选择 **LabWeaver: Build analysis sandbox**。安装宿主依赖 `uv sync --locked` 后选择项目 `.venv` 解释器。

模型配置使用忽略的 `.env` 或 `labweaver.local.toml` 引用已有配置，例如：

```toml
[intake]
env_file = "D:/private/existing.env"
mode = "live"
analysis_max_tokens = 4096
execution_image = "labweaver-python:0.2.0"
execution_timeout = 60
```

模型/API 超时仍由 `LLM_TIMEOUT` 控制，容器执行超时单独配置。主 Agent 与代码 Agent 默认同一接口，代码 Agent 使用单独的输出 token 设置。镜像在构建时安装锁定依赖；分析过程中不联网安装。

打开 `src/labweaver/app.py`，Run Python File 或 F5。日常只输入 CSV 路径、可选资料路径（分号分隔）、任务；不要改程序里的 prompt 或资料路径。CSV 默认自动识别编码与分隔符，歧义时展示候选预览供选择。通过 TOML/CLI 仍可高级覆盖。

Agent 会显示短计划、真实代码执行与修正次数、流式回答。收到关键问题时输入确认，随后继续同一任务；完成后可修改范围、图型、标题或继续新任务。退出输入 `退出`。图表和全量结果保存在报告列出的路径，终端只预览部分行。

## 三类任务验收

可以将 `tests/fixtures/` 的合成 CSV 用于在线本机检验，也可以指定自己的数据。检验是否产生正确成果，不能只看 completed。

| CSV | 任务 | 独立期望 |
|---|---|---|
| composite.csv | 拆分组合编码，按类别/区域交叉统计数量，绘制分组柱状图 | A: East=6/West=3；B: East=5/West=1；完整交叉 CSV＋PNG/SVG |
| dates.csv | 解析日期，按日汇总数量，计算峰谷差值并绘制趋势图 | 12/1=5、12/2=4、12/3=3；差值=2；日期升序 |
| cleaning.csv | 删除完全重复行，再将数量空白填0，派生双倍数量并导出，报告前后变化 | 4→3行；去重1；填充1；双倍为4/0/8；编号001/002/003保留 |

打开 Markdown 与 PNG 检查中文、轴和图型；检查 CSV/指标 JSON 的实际值；打开每次 `.py` 与运行 JSON，核对 execute_python 工具 ID、退出码、修正错误和真实 image_id。源文件 SHA-256 应不变。

## 编程接口和重放

`create_session(csv_path, model, *, analysis_model=None, execution_config=None, material_paths=None, encoding="auto", delimiter="auto")` 返回当前进程的 `AgentSession`。`invoke(task)`、`resume(reply)` 返回报告，`cancel()` 终止容器并取消会话，`save(output_dir)` 导出最新报告并返回包含路径的字典。`run_intake()` 是单次包装，无法独自完成需回复的澄清。

```python
from labweaver.runtime.replay import replay_analysis
report = replay_analysis("runs/会话/任务/记录.json", "原始CSV路径")
print(report["status"], report["result_csv_paths"])
```

重放不调用模型，核对输入和代码 SHA-256，并使用记录中的不可变 Docker 镜像 ID；原镜像被删除时明确失败。它重新计算成功脚本，并非恢复已关闭的会话。表格/指标应与原运行一致，图像字节可因生成元数据不同而不同。

`--offline` 适合公开演示和工具流程验证，代码模型只认识有限的合成任务，不用于评价通用理解。离线统计仍真实执行容器；宿主合同测试的注入执行器只用于测试，不构成第二套生产路径。

## 常见问题

- `docker_unavailable/docker_not_ready`：安装并启动 Linux Docker 引擎。Windows 虚拟化未生效时完成重启，不能通过改用宿主 Python绕过。
- `sandbox_image_missing`：运行 VS Code 的镜像构建任务。生产执行不会自动拉取或构建镜像。
- `python_failed`：代码 Agent 根据真实 stderr 自动有限修正；连续失败则保留代码/错误，返回未完成。
- `missing_deliverables`：任务交付项尚缺真实产物，不能用计划文字代替结果。
- `execution_timeout/memory_limit/output_limit/artifact_rejected`：资源或产物协议失败，查看保存的执行记录；不静默丢行。
- 关闭程序后不能 resume；仅可通过成功代码重放或创建新会话。
