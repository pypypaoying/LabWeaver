# 用自己的 CSV 与 PDF 做真实在线检验

此入口使用已经配置的真实模型。它依次运行相同任务、相同 CSV 与相同解析设置：第一项不提供任何资料，第二项只提供你选定的一个 PDF。自己的文件不复制进项目，记录写入被 Git 忽略的 `runs/`。

## VS Code 直接运行

1. 选择项目 `.venv` 的 Python 3.11+，保持已有 `.env` 或本地配置路径有效。
2. 打开根目录 `run_real_checks.py`，点击 **Run Python File**；也可 F5 选择 **LabWeaver: real CSV + PDF checks**。在下方集成终端输入。
3. 输入 CSV 和文字型 PDF 的完整路径。路径可以含空格，也可以带复制路径时的外层引号。
4. 输入自己的任务说明；回车采用通用数据总结。多行任务可存成 UTF-8 TXT，输入 `@D:/your-project/task.txt`，无需改 Python。
5. 输入 CSV 编码与分隔符；普通 UTF-8 逗号 CSV 两项均回车。中文旧编码可填 `gb18030`，制表符可填 `tab`。
6. 等待两次 Agent 运行，打开打印出的 JSON、Markdown 和 `comparison.json`。

任务可以写：

> 请总结 CSV 的数据规模、字段、缺失情况及工具已计算的数值摘要。对 PDF 中涉及的字段定义、单位和统计口径检索原文并给出处；没有依据的解释标为待确认。不要生成未执行的计算或研究结论。

PDF 最好是这份 CSV 的数据字典、数据集说明或项目要求。当前仅支持可提取文字的 PDF，不做 OCR；加密、损坏或超限资料会报告失败。CSV 上限 10 MiB、10 万行、200 列；PDF 单文件 5 MiB、100 页。资料检索最多两次、每次三个片段，因此任务应明确具体字段或术语。

## 检查什么

| 项目 | 预期 |
| --- | --- |
| 仅 CSV | `awaiting_confirmation`；实际执行账本只有一次 `profile_csv`；`materials=[]`、无检索和引用。控制台与 JSON 的 `final_answer` 是总结，没有资料简报 MD。 |
| CSV + PDF | `awaiting_confirmation`、`materials_completed=true`；先概览，再实际检索；有真实命中和引用，生成 JSON 与 Markdown。 |
| 统计一致 | 两份实际执行账本的 CSV 概览完全相同；资料只帮助解释，不改变解析与统计。 |
| 输入不变 | CSV、PDF 运行前后 SHA-256 一致。 |
| 出处核验 | 回答中的 `[D1-Cn]` 对应 `retrieved_chunks`，来源是所选 PDF 的实际页码；在原 PDF 中核对正文及解释是否相符。 |

输出位于 `runs/real-checks-<ID>/`，两个子目录分别是 `csv-only`、`csv-with-pdf`，比较记录是 `comparison.json`。总计五项检查均 PASS，入口返回 0；未通过返回 1，配置或读写失败返回 2。

若检索确实执行，但未命中，普通 D3 流程可能仍以“资料不足”结束并等待确认；这次“带出处总结”的检验要求至少存在实际引用，因此比较记录中的 `pdf_has_retrieved_citations` 为 FAIL。这不把无答案伪装成带出处成功。

自动检查验证执行和来源，语义质量仍需人工核对。数值摘要取自 JSON 中的实际 CSV 结果，不应把模型措辞当成新统计；当前未实现唯一值计数、分组聚合、相关性、绘图或整表逐行语义分析。源文件不修改；运行时会将任务、概览及检索片段发送到你配置的模型接口。

## 现有入口也允许自定义

`run_labweaver.py` 的任务和资料来自外部 `labweaver.toml`，其中的问卷路径只是合成样例默认值。也可以在被忽略的 `labweaver.local.toml` 的同一个 `[intake]` 表内保留原有 `env_file`，并增加：

```toml
csv = "D:/your-project/data.csv"
task = "请概括实际数据，并结合资料解释关键字段，给出出处和待确认问题。"
materials = ["D:/your-project/data-dictionary.pdf"]
mode = "live"
```

`materials=[]` 切换成仅 CSV；使用 `task_file="D:/your-project/task.txt"` 可替代 `task`，同层不能同时填写两者。配置中的 `task` 是用户消息；`agent.py` 的系统提示负责阶段边界、工具规则和证据要求。

入口开发回归（2026-10-07）：完整离线测试 250 项、28 个子用例通过。新增三项用实际 Deep Agents 循环验证用户选择的路径与任务、PDF 命中、无答案及损坏 PDF；阻断网络和模型凭据读取。这些是合成样例回归，自己的真实文件与真实模型结果需要运行上述在线入口检验。
