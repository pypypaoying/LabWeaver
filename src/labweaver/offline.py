"""Deterministic demo models. Real tools/containers execute; arbitrary NLP needs live mode."""

import json
import re
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


def _result(message):
    return ChatResult(generations=[ChatGeneration(message=message)])


class OfflineIntakeModel(BaseChatModel):
    _serial: int = PrivateAttr(default=0)
    _available_tools: list = PrivateAttr(default_factory=list)
    _bound_tool_sets: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "labweaver-offline"

    @property
    def bound_tool_sets(self):
        return self._bound_tool_sets

    def bind_tools(self, tools, **kwargs):
        names = [t.get("name") if isinstance(t, dict) else t.name for t in tools]
        self._bound_tool_sets.append(names)
        self._available_tools = names
        return self

    def get_num_tokens(self, text):
        return max(1, len(text) // 3)

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(m.content)) for m in messages)

    def _call(self, name, args):
        self._serial += 1
        return _result(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": args,
                        "id": f"offline-{self._serial}",
                        "type": "tool_call",
                    }
                ],
            )
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        start = max(i for i, m in enumerate(messages) if m.type == "human")
        task = str(messages[start].content)
        results = [m for m in messages[start:] if isinstance(m, ToolMessage)]
        names = [m.name for m in results]
        if "profile_csv" in self._available_tools:
            return self._call("profile_csv", {})
        if (
            "search_materials" in self._available_tools
            and "search_materials" not in names
            and re.search(r"资料|依据|说明|基于|规则", task)
        ):
            return self._call(
                "search_materials", {"query": "score 实验 变量 方法 要求"}
            )
        calculation = bool(re.search(r"统计|交叉|汇总|绘|画|清洗|去重|拆分|计算", task))
        if calculation and "set_task_plan" not in names:
            deliveries = [
                {"id": "table", "kind": "table", "description": "完整统计或派生数据表"}
            ]
            if re.search(r"峰|谷|差|缺失|重复|清洗|去重", task):
                deliveries.append(
                    {
                        "id": "metrics",
                        "kind": "metric",
                        "description": "计算指标与处理前后变化",
                    }
                )
            if re.search(r"图|绘|画", task):
                deliveries.append(
                    {
                        "id": "chart",
                        "kind": "figure",
                        "description": "所需图表及实际绘图数据",
                    }
                )
            return self._call("set_task_plan", {"deliverables": deliveries})
        if calculation and "delegate_analysis" not in names:
            plan = next(
                json.loads(m.content) for m in results if m.name == "set_task_plan"
            )
            return self._call(
                "delegate_analysis",
                {
                    "instruction": task,
                    "deliverable_ids": [d["id"] for d in plan["deliverables"]],
                },
            )
        profile = next(
            (
                json.loads(m.content)
                for m in messages
                if isinstance(m, ToolMessage) and m.name == "profile_csv"
            ),
            {},
        )
        text = (
            "已根据真实工具结果完成任务；统计表、指标、代码与图表见产物。"
            if calculation
            else f"已读取完整 CSV：{profile.get('row_count', 0)} 行 × {profile.get('column_count', 0)} 列。字段："
            + "、".join(c["name"] for c in profile.get("columns", []))
            + "；缺失和数值摘要见概览记录。"
        )
        for m in results:
            if m.name == "search_materials":
                hits = json.loads(m.content).get("matches", [])
                text += " " + (
                    " ".join(f"[{h['id']}]" for h in hits)
                    if hits
                    else "资料不足，无命中。"
                )
        return _result(
            AIMessage(
                content=json.dumps(
                    {
                        "task_kind": "calculation" if calculation else "summary",
                        "retrieval_reason": "按当前任务决定检索，离线演示不提供通用自然语言推理",
                        "answer": text,
                    },
                    ensure_ascii=False,
                )
            )
        )


class OfflineAnalysisModel(OfflineIntakeModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        context = json.loads(next(m.content for m in messages if m.type == "human"))
        results = [
            json.loads(m.content)
            for m in messages
            if isinstance(m, ToolMessage) and m.name == "execute_python"
        ]
        if not results:
            return self._call("execute_python", {"code": demo_code(context)})
        output = results[-1]
        return _result(
            AIMessage(
                content=json.dumps(
                    {
                        "status": "completed",
                        "artifact_ids": output.get("artifact_ids", []),
                        "question": "",
                        "summary": "确定性演示脚本已实际执行；在线模型用于通用任务。",
                    },
                    ensure_ascii=False,
                )
            )
        )


def demo_code(context):
    """Public demo scripts, not a production algorithm tool or host execution path."""
    instruction = context["instruction"]
    columns = {c["name"]: c["id"] for c in context["columns"]}
    deliveries = {d["kind"]: d["id"] for d in context["deliverables"]}
    header = "from helper import load_dataset, emit_table, emit_metric, emit_figure\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\ndf = load_dataset()\n"
    if "组合编码" in instruction:
        body = f"""parts = df[{columns["组合编码"]!r}].str.split('-', expand=True)
df['类别'], df['区域'] = parts[0], parts[1]
df['数量'] = pd.to_numeric(df[{columns["数量"]!r}], errors='raise')
result = df.pivot_table(index='类别', columns='区域', values='数量', aggfunc='sum', fill_value=0).sort_index()
plot_data = result.reset_index()
fig, ax = plt.subplots()
result.plot.bar(ax=ax, title='交叉统计', ylabel='数量', rot=0)
"""
    elif "日期" in instruction or "趋势" in instruction or "汇总" in instruction:
        date = next(columns[n] for n in columns if n in {"日期", "时间"})
        value = columns.get("数量", columns.get("包裹量"))
        frequency = "W-SUN" if "周" in instruction else "D"
        body = f"""dates = pd.to_datetime(df[{date!r}], errors='raise')
values = pd.to_numeric(df[{value!r}], errors='raise')
series = pd.Series(values.to_numpy(), index=dates).sort_index().resample({frequency!r}).sum()
plot_data = series.rename('数量').rename_axis('日期').reset_index()
plot_data['日期'] = plot_data['日期'].dt.strftime('%Y-%m-%d')
result = plot_data
metrics = {{'峰值日期': str(series.idxmax().date()), '峰值': int(series.max()), '谷值日期': str(series.idxmin().date()), '谷值': int(series.min()), '差值': int(series.max() - series.min())}}
fig, ax = plt.subplots()
ax.plot(pd.to_datetime(plot_data['日期']), plot_data['数量'], marker='o')
ax.set(title='数量趋势', xlabel='日期', ylabel='数量')
fig.autofmt_xdate()
"""
    elif "缺失" in instruction or "去重" in instruction:
        amount = columns.get("数量", columns.get("value"))
        body = f"""before = len(df)
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
empty = df[{amount!r}].str.strip().eq('')
missing = int(empty.sum())
df.loc[empty, {amount!r}] = '0'
df['数量'] = pd.to_numeric(df[{amount!r}], errors='raise')
df['双倍数量'] = df['数量'] * 2
df.columns = [next((c['name'] for c in df.attrs['columns'] if c['id'] == name), name) for name in df.columns]
result = df
metrics = {{'原始行数': before, '处理后行数': len(df), '删除重复': duplicates, '填充缺失': missing}}
"""
    elif (
        "department" in columns
        and "satisfaction" in columns
        or "部门" in columns
        and "score" in columns
        or "department" in columns
        and "score" in columns
    ):
        group = columns.get("部门", columns.get("department"))
        score = columns.get("score", columns.get("satisfaction"))
        body = f"""score = df[{score!r}].str.strip().replace('', np.nan)
df['score'] = pd.to_numeric(score, errors='raise')
result = df.groupby({group!r}, sort=True)['score'].mean().reset_index(name='平均分')
result.columns = ['部门', '平均分']
plot_data = result
fig, ax = plt.subplots()
ax.bar(result['部门'], result['平均分'])
ax.set(title='部门平均分', xlabel='部门', ylabel='平均分')
"""
    elif "run_name" in columns and "loss" in columns:
        body = f"""valid = df[{columns["loss"]!r}].str.strip().ne('')
result = pd.DataFrame({{'实验': df.loc[valid, {columns["run_name"]!r}], 'loss': pd.to_numeric(df.loc[valid, {columns["loss"]!r}], errors='raise')}})
plot_data = result
fig, ax = plt.subplots()
ax.bar(result['实验'], result['loss'])
ax.set(title='实验 loss', xlabel='实验', ylabel='loss')
fig.autofmt_xdate()
"""
    else:
        raise ValueError(
            "Offline analysis supports the documented public demos only; use live mode for other tasks"
        )
    tail = ""
    if "table" in deliveries:
        tail += f"emit_table(result.reset_index(drop=True), {deliveries['table']!r})\n"
    if "metric" in deliveries:
        tail += f"emit_metric(metrics, {deliveries['metric']!r})\n"
    if "figure" in deliveries:
        tail += f"emit_figure(fig, plot_data, {deliveries['figure']!r})\n"
    return header + body + tail
