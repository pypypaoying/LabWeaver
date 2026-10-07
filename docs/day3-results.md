> 历史阶段记录：本次连续统计更新已替代旧终态与功能范围；当前使用方式见 [连续统计实现](session-statistics.md)。

# D3 验收结果与检验方法

验收日期：2026-10-07。Windows、Python 3.11.15；依赖锁定 pypdf 6.19.0、rank-bm25 0.2.2、jieba 0.42.1，Deep Agents 版本保持既有基线。开发实现范围见 [D3 说明](day3.md)。

## 实际结果

- 完整测试：**247 passed、28 subtests passed**。
- 可直接运行的 verify_d3.py：**12/12** 检查通过，退出 0；产生四对真实 JSON/Markdown 记录。
- 包构建：uv build --offline 成功生成 wheel 和源码包。
- 合成文字 PDF：两页，已渲染目检，真实检索可定位第 1 页和第 2 页。
- 实验 CLI：仅选择 examples/configs/experiments.toml 即可完成资料接入，生成 JSON 和 Markdown。

本机受限环境的系统临时目录存在 ACL 问题，原有 CSV 测试的 TemporaryDirectory 在该环境写入失败。本次完整回归在允许临时目录写入的执行环境中，用新 workspace basetemp 完成；未跳过 CSV 测试。分词缓存曾在系统临时目录初始化卡住，已改为当前目录 .cache/jieba 下的可写目录，并增加即时权限检查与回归测试。

## 离线验收

七个标注查询保存在 examples/retrieval_queries.json，覆盖部门问卷要求、前导零编号的 PDF 第 2 页、有序量表的 PDF 第 1 页、替代总体分布目标、实验比较方法、temperature_c 单位，以及无答案的 quasar_redshift。

另外四次 Agent 验收全部走真实 Deep Agents 流程：

| 项目 | CSV | 模型 | CSV 工具 | 检索工具 | 终态与产物 |
| --- | --- | --- | --- | --- | --- |
| 部门问卷 | 8×5 | 3 | 1 | 1 | awaiting_confirmation，JSON+MD |
| 实验记录 | 6×6 | 3 | 1 | 1 | awaiting_confirmation，JSON+MD |
| 同问卷、替代资料 | 8×5 | 3 | 1 | 1 | 统计相同、引用来源改变，JSON+MD |
| 资料无答案 | 8×5 | 3 | 1 | 1 | 明确资料不足、零引用，JSON+MD |

验收程序阻断网络，检查模型确实接收带匹配 ID 的 ToolMessage；调用、结果和实际账本相等。所有 CSV 与资料的执行前后哈希一致。离线模型验证执行流程与出处，不证明真实 LLM 的理解能力。

测试还覆盖：TXT/Markdown/PDF 来源定位、中文与英文标识符、分块重叠、空/错误编码/损坏/扫描/加密/超限资料、独立预算、并行检索、先检索后概览的拒绝、缺少检索、伪造 ID/来源/页码、畸形工具结果、被篡改的查询或 ToolMessage、资料中的指令文本、异常不回显敏感信息以及产物写入失败和重名保护。

## 小规模真实在线验收

使用既有本地模型配置，没有复制或发布密钥；通过 run_labweaver.main 执行直接入口。三组均真实调用工具并保存产物：

| 资料集合 | CSV 工具 | 检索工具 | 模型调用 | 终态 | 失败原因 | 输入哈希 |
| --- | --- | --- | --- | --- | --- | --- |
| 问卷要求、变量 TXT、方法 PDF | 1 | 2 | 3 | awaiting_confirmation | 无 | 未变 |
| 同 CSV、替代问卷要求 | 1 | 2 | 3 | awaiting_confirmation | 无 | 未变 |
| 实验要求、变量 TXT、方法 MD | 1 | 2 | 3 | awaiting_confirmation | 无 | 未变 |

同一问卷 CSV 更换要求后，候选方案由部门层面的分布比较，转为列出“用户部门比较诉求与资料总体分布要求”的冲突，以及总体频数、比例和中位数方案；引用文件改变，profile 与文件哈希保持一致。实验回答提出评估集、loss 定义、随机种子/重复运行的可比性确认，未声称已执行统计分析。

人工审阅发现初次问卷回答仍把均值列为有序量表的候选主指标，和资料约束不够协调。随后补充通用提示：工具 numeric 推断不等于连续量表，方法约束用于筛选候选方案。再次在线复验成功（3 次模型、1 次 CSV、2 次检索），问卷回答改为频数与中位数等描述，并明确不主张均值作为主指标。这是一次人工核验与修正，不宣称已实现自动语义支持证明。

原始在线 JSON、Markdown 和控制台记录在被忽略的 runs/ 与 .cache/，不随仓库发布。只验证本次兼容接口与这些小样例，不能据此保证所有任务或接口的表现。

## 用 VS Code 自己检验

### 一键离线验收

1. 在 VS Code 打开项目，选择项目 .venv 的 Python 3.11+。
2. 打开根目录 verify_d3.py，点击 Run Python File。
3. 应显示逐项 PASS，最后 Passed: 12/12；无需模型密钥。
4. 打开打印的输出目录，检查四对同 stem JSON 和 Markdown，确认正文、来源原文及行/页位置。

### 在线项目接入

1. 在 labweaver.toml 保持 mode = "live"，默认三份问卷资料已列好；.env 或本地路径配置保持已有设置。
2. 打开 run_labweaver.py，点击 Run Python File。
3. 控制台应显示 awaiting_confirmation、CSV 8×5、实际资料检索与简报位置。
4. JSON 应包含 materials_completed=true、profile_completed=true、真实 retrieval_queries、retrieved_chunks、citations；执行账本先 profile_csv 后 search_materials，检索 1–2 次，模型不超过 5 次。
5. Markdown 按五部分组织；引用 [D1-C3] 等 ID 必须对应 JSON 命中片段，其文件名、行号/页码和原文能够对照源资料。

在线调用次数可由模型决定，不能把固定“3 次模型”作为所有在线运行的成功条件。

### 换资料和项目

将 materials 的 requirements.md 改为 requirements_alternative.md，保持问卷 CSV 与任务不变后运行。预期统计不变，回答识别部门比较与总体分布要求的冲突；若仍照搬之前建议，应作为理解质量问题人工复核。

实验项目可在公开配置中改为 examples/data/experiments.csv、examples/tasks/experiments_rag.txt，并将三份材料路径改为 examples/materials/experiments 下的 requirements.md、variables.txt、methods.md。无需修改工具代码。

在已配置的 VS Code pytest 测试面板点击运行全部测试，可检验正常和失败路径。仅有绿色状态或有效引用不证明分析建议正确；仍需检查量表解释、方法条件与资料要求。

### 云端复现

```shell
uv sync --locked
uv run --frozen pytest -q
uv run --frozen python verify_d3.py
uv run --frozen labweaver intake --offline
uv run --frozen labweaver intake --config examples/configs/experiments.toml
```

上述云端验证均无需模型密钥。在线默认可用 uv run labweaver intake。GitHub Actions 保留 Ubuntu/Windows × Python 3.11/3.12 四个环境，并执行一键验收程序。
