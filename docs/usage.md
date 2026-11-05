# 使用与检验

## Windows 本地聊天界面

浏览器应用需要先完成下面的 Docker 配置，模型可通过页面设置配置。首次安装额外需要 Node.js 构建前端；日常运行只使用 Python 服务和构建后的页面，不启动 Vite，也不自动安装 Docker 或分析依赖。

```powershell
uv sync --locked --extra web
cd frontend
npm ci
npm run build
cd ..
```

双击 `start-analiymaster.cmd`。也可从仓库根目录执行：

```powershell
uv run --frozen --no-sync --extra web labweaver serve --open
```

启动会检查模型配置、数据目录和本地 Docker/镜像。页面的应用访问令牌由启动链接传入后从地址栏移除；它不是模型 API Key。无需把密钥粘贴到聊天框。服务仅绑定 `127.0.0.1`，不要配置公共监听地址。

1. 新建对话，选择或拖入一份 CSV，可加 TXT/Markdown/文字型 PDF。已有无附件、消息和任务的空草稿会复用，反复点击不会堆积空对话。上传后查看预览；图片、Excel、JSON、DOCX 尚不支持分析。
2. 输入任务并发送。首次发送冻结 CSV 和资料；需要更换输入时新建对话。输入法确认不会发送，Shift+Enter 换行。
3. 等待时在聊天区查看“执行过程”：工具调用、Python 执行与修正、检索、核验和保存均来自实际事件。展开右侧执行面板可查看完整计划、代码、错误及资料来源；出现澄清问题时在当前运行提交回复。单个步骤完成不等于全部任务完成。
4. 核验完成后查看表格/图表，下载完整 CSV、PNG/SVG、代码和报告。表格显示的是分页预览，完整结果没有 100 行限制。
5. 停止后等待取消与容器清理确认；关闭或刷新浏览器不会自动重跑或取消，也不会结束后台服务。全局只有一个活动任务，等待回复时也占用该位置。
6. 服务重启后重新打开历史，继续提交新任务。旧的未完成运行会显示 interrupted，不能回复旧问题以恢复节点；可按原任务重新执行。

基础历史、导入附件和生成成果位于应用数据目录，与 CLI 的 `runs/` 分开。旧对话的新运行重建有限历史背景，再从原始 CSV 重新计算。Windows 资源管理器复制的文件未必能直接粘贴到浏览器；失败时使用拖拽或文件选择器，粘贴路径不会读取磁盘。

2026-10-12 更新增大了常用文字和图标。已运行的服务需重启并刷新页面才能使用更新后的界面和路径修复；过去失败的任务不会自动重跑，可在历史中按原任务重新执行。

在对话列表的 **⋯** 菜单或右键菜单中重命名、置顶/取消置顶、永久删除。置顶保存在 SQLite，重启后仍在顶部。删除必须明确确认，会移除该对话的消息、导入副本及全部独占运行目录（包含部分失败代码与日志），不改用户原文件。运行中/等待回复/停止中的对话须先结束任务。清理失败显示待删除状态，再次确认即可继续；它不能继续分析，也不能假报删除成功。

离线公开演示可使用 `labweaver serve --offline --open`，分析仍需要真实 Docker；确定性模型只支持有限示例。搜索、工作空间、独立知识库与桌面安装包留待 P2–P4。详细架构见 [Desktop](desktop.md)。

复杂任务默认代码输出预算 12288 tokens。`model_output_truncated` 表示超过模型输出预算；程序会在原预算内做一次格式修正，不执行半截工具参数。`missing_deliverables` 表示真实执行未交齐成果；只打印结果不算完成，代码 Agent 会收到缺失项并有限修正。聊天会展示主/代码 Agent 的真实请求开始、结束与耗时。

浏览器服务的 `WebSettings.model_timeout=180`、`task_timeout=900` 分别约束单模型等待与累计运行时间；等待人工回复暂停累计计时，恢复不重置已消耗时间。这是应用服务的进程监督，区别于 HTTP 读超时和 Docker 每次 60 秒的执行时限。超时保留合法成果并核对所属容器清理，不允许永久显示运行中。高级 Python 集成可通过 `WebSettings` 调整；GUI 尚无预算编辑页面。

### 重复启动与退出

关闭浏览器标签页仅关闭界面，Python 服务继续运行。再次双击启动器时，会验证并重新打开同一项目的已有服务，沿用该服务的访问令牌；不会新建第二个进程抢占 8765，也不会先打开无法认证的页面。运行模式不同需先退出原服务。

完全结束应用请点击左下角 **退出应用**，或在原启动终端按 **Ctrl+C**。有任务运行、等待澄清或正在取消时，先完成或停止任务再退出。退出后页面显示“应用已退出”，工作进程和监听端口关闭，历史、附件与已登记成果保留；重新启动生成新的访问令牌。

`10048/port_in_use` 表示端口已被占用。陌生服务或未更新的旧版服务不会被自动停止，也不会收到新应用令牌；这时需要退出原启动终端，或明确使用其他端口。修复之前的版本在绑定端口前就打开新令牌页面，因此重复启动会同时出现端口错误和模型列表认证失败。更新后首次使用需让原服务退出一次。过期页面会明确提示“访问令牌已失效，请用启动器重新打开应用”。

默认启动的受保护实例信息位于 `~/.labweaver/instances/`，它只保存本机应用访问资料和默认数据根绑定，不保存模型 API Key；Windows 使用当前用户 DPAPI，Unix 使用私有目录与文件权限。保留数据根绑定，使普通 Windows 启动器与打包宿主重启后沿用同一份历史。显式 `--data-dir` 的运行信息只在指定目录内，独立测试不能复用用户服务。不要将本机实例信息提交到仓库。

## 模型选择与服务商配置

输入框右下角、发送按钮左侧是模型选择器，按服务商分组展示。新用户默认 DeepSeek；点击未配置的模型会进入配置页。每次任务和历史卡片记录实际选择的服务商与模型。切换模型会在下一项任务重建 Agent 运行时，并把已核验的最近历史作为背景；CSV 和资料绑定保持当前对话的输入。运行中或等待澄清时不能切换，回复继续使用原任务模型。

1. 点击左下角 **设置**，选择 DeepSeek，或点击 **添加兼容供应商**。
2. 填写服务商名称、API 根地址 **Base URL**、**API Key**；模型列表每行一个实际模型 ID，默认模型必须包含在列表里。地址应来自服务商 API 文档，不能填写聊天网页、密钥、查询参数或用户名。首版支持 OpenAI **Chat Completions** 兼容接口，不支持原生 Anthropic 协议。
3. 点击 **保存配置**。Key 留空保留已有凭据；勾选“保存时清除现有密钥”才会移除。返回页面后不会回显密钥。
4. 可点击 **获取模型列表**或 **测试连接**。两者只向保存的接口发送专用请求，不发送 CSV、任务或资料。连接成功和工具调用验证分别展示，不代表已经证明任意分析任务可完成。接口不支持列表或返回超过 50 个模型时，手动维护需要的模型即可。
5. 返回聊天，在右下角选择模型后发送任务。**设为应用默认**影响默认选择；重新打开历史仍显示该对话上次使用的模型。

DeepSeek 预置 Base URL 为 `https://api.deepseek.com`，模型为 `deepseek-flash`、`deepseek-v4-pro`，默认前者，按[官方 API 文档](https://api-docs.deepseek.com/api/create-chat-completion/)核对。当前 ChatOpenAI 适配器不会回传 DeepSeek 的 `reasoning_content`，因此仅对官方接口的这些模型禁用 thinking，确保连续工具调用兼容；其他兼容服务商不注入此参数。[DeepSeek 工具调用要求](https://api-docs.deepseek.com/guides/thinking_mode/)

Windows 新 Key 加密保存在本机应用数据目录，只能由当前 Windows 用户解密，不进入 SQLite、聊天、执行记录或容器。非 Windows 本轮仅在后端内存保存新 Key，重启服务后需重新填写；不会退回明文文件。设置页会说明实际保存方式。浏览器只在输入和保存请求中短暂持有新 Key，不保存到 localStorage/sessionStorage，保存后清空字段。

已有后端配置继续兼容：官方 DeepSeek 配置作为预置服务商的环境引用；其他配置显示为“现有本地配置”，公开地址和模型只读。修改环境引用的接口地址后不会把原环境 Key 发往新地址。需要修改只读配置时可以新增服务商；CLI 和 VS Code 入口继续使用原 `.env`/TOML 配置，不读取桌面服务商设置。离线演示锁定选择器并显示“离线演示”，不假装调用在线模型。

## Windows / VS Code

首次安装 Docker Desktop 与 WSL2、启用 Virtual Machine Platform/硬件虚拟化，按系统要求重启。确认 Docker Desktop Linux 引擎已运行，再在 VS Code 的 Terminal → Run Task 选择 **AnaliyMaster: Build analysis sandbox**。安装宿主依赖 `uv sync --locked` 后选择项目 `.venv` 解释器。

模型配置使用忽略的 `.env` 或 `labweaver.local.toml` 引用已有配置，例如：

```toml
[intake]
env_file = "D:/private/existing.env"
mode = "live"
analysis_max_tokens = 12288
execution_image = "labweaver-python:0.3.0"
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
- Windows 打包宿主中出现 `/input/analysis.py` 不存在：曾由 AppData/TEMP 路径重定向使 Docker 挂到空目录引起。本轮已在临时目录创建后解析真实路径；更新后重启服务再重新执行，不需要放开容器输入写权限。
- `save_failed`：成果保存或数据库登记未完成，真实代码执行成功也不能标记整个任务完成。本轮同时修复了 Windows 路径重定向造成的存储根目录不一致；其他磁盘、权限或登记错误仍需查看实际失败记录。
- `missing_deliverables`：任务交付项尚缺真实产物，不能用计划文字代替结果。
- `execution_timeout/memory_limit/output_limit/artifact_rejected`：资源或产物协议失败，查看保存的执行记录；不静默丢行。
- 关闭程序后不能 resume；仅可通过成功代码重放或创建新会话。
# 可靠分析与建模开发版

使用镜像 `labweaver-python:0.3.0` 时可以请求单表分类或回归，例如“用指定特征预测目标，说明拆分方式、比较简单基线，并交付逐行测试预测、指标、图表和报告”。仍使用原对话入口，不需要新的命令模式。目标或切分口径不清楚时回答 Agent 的澄清。

任务完成后展开“任务约定”和“核验覆盖”：前者记录实际使用的目标、特征、范围与默认值；后者区分已通过、失败和未检查的项目，并可查看原始证据。旧记录没有新证据时显示未核验。`completed` 表示登记成果和当前必需检查通过，不代表训练全过程没有泄漏或所有解释均正确。

v2评测已达到数值门槛，仍有未达成案例，详见[v2完整结果](../evaluation/results/v2.md)和[设计边界](reliability.md)。测试预测表使用稳定源记录 ID；完整结果、代码和报告仍可从原下载入口获取。不要用图表或少量预览代替完整结果核对。

## 按需分析 Skills

项目内含四个经过适配的只读技能：数据探索、统计分析、可视化、交付检查。它们提供专业方法参考，不直接执行统计。主 Agent 在完成 CSV 概览后、代码 Agent 在获授权委派后，可以自行选择 `read_skill(skill_id)`。正文只在实际读取时进入模型上下文；简单概览无需加载。

主、代码 Agent 沿用原技能目录和按需读取流程。角色分流曾作为token优化候选实验，但首轮未达采纳门槛，已保守撤回；没有把它作为已验证收益发布。

当前作为可选能力，默认关闭。在项目 `labweaver.toml` 或被忽略的 `labweaver.local.toml` 的 `[intake]` 中设置：

```toml
[intake]
skills_enabled = true
```

修改后重启应用，再发起任务；VS Code 入口和 CLI 也使用此配置。直接调用 Python 时使用 `create_session(csv_path, model, skills_enabled=True)`；设为 `False` 关闭。模型配置和沙箱镜像沿用现有值，安装时不需要增加新的分析依赖。

运行过程中“已读取分析技能”表示真实工具读取。报告 JSON 的 `skills.enabled` 表示开关，`skills.loads` 记录实际角色、调用ID、技能ID和内容哈希；开启但未读取会保留空列表。加载技能不等于完成分析，结果仍须来自 Docker 和原有成果/核验流程。技能不能作为用户资料的引用出处。

技能来自固定版本的公开 Anthropic Data Analyst Plugin，已翻译、缩写并适配字段ID、现有库和 `emit_*` 协议。许可证与修改归属随 Python 包保存于 `src/labweaver/skills/`；这些说明文件不进入模型上下文。没有直接安装 Claude 插件或更换模型。

受控试验：`python evaluation/skills_pilot.py --run --output .cache/evaluation/skills-pilot-new` 执行8个独立合成任务、开关各3次共48次，须先启动Docker并配置后端 `LLM_*` 或被引用的 `.env`；脚本不自动采用页面当前选择的模型。用 `--report --output 同一目录` 查看汇总。复现新版本请使用新的空目录；同一目录仅恢复原快照、跳过已有试次，保留全部失败。它不是离线演示，不能仅凭技能被读取就断言完成率提高。

## Token 效率

常规入口在Skills关闭时自动使用单份结构化委派包、阶段性工具及建模规则、精简成果回执。Skills开启时保留基线模型上下文；两轮开发实验中的全局压缩候选未通过该模式的交付门槛，没有为了降耗强行启用。沿用原Skills开关，不需要新配置；报告的model_context_policy说明实际策略。完整数据、代码、图表、来源及核验证据保存在宿主；模型可按成果ID读取更多。显示省略或preview_only的内容不能当作完整结果。

运行JSON的每次 `token_usage` 保留接口实际返回的输入/输出/总token，并新增 `cache_usage`。供应商未返回缓存计数时为null，不能理解为零或直接换算费用。`context_bytes`仅是逻辑请求各部分的UTF-8字节数，不是精确分词或计费归因；不记录原始请求或认证信息。

开发计划、采纳门槛和后续实验见[Token效率](token-efficiency.md)。长历史压缩、脚本补丁和单Agent路由尚待独立验收。
