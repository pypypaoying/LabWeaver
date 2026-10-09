"""Deterministic offline fixture model; exercises the real tools, never calls an API.

This supports demo summaries and the public medal dialogue, not arbitrary NLP.
Use a live model for general task interpretation.
"""

from __future__ import annotations
import json
import re
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class OfflineIntakeModel(BaseChatModel):
    _bound_tool_sets: list[list[str]] = PrivateAttr(default_factory=list)
    _seen_tool_results: list[str] = PrivateAttr(default_factory=list)
    _available_tools: list[str] = PrivateAttr(default_factory=list)
    _serial: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "labweaver-offline"

    @property
    def bound_tool_sets(self):
        return [list(v) for v in self._bound_tool_sets]

    @property
    def seen_tool_results(self):
        return list(self._seen_tool_results)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [v.get("name") if isinstance(v, dict) else v.name for v in tools]
        if not set(names) <= {
            "profile_csv",
            "analyze_csv",
            "search_materials",
            "ask_user",
            "prepare_distribution",
            "delegate_visualization",
            "read_analysis_rows",
        }:
            raise ValueError("Unexpected tool exposed to offline model")
        self._bound_tool_sets.append(names)
        self._available_tools = names
        return self

    def get_num_tokens(self, text):
        return max(1, (len(text) + 3) // 4)

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(m.content)) for m in messages)

    def _call(self, name, args):
        self._serial += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": name,
                                "args": args,
                                "id": f"offline-{name}-{self._serial}",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager: Any = None, **kwargs
    ):
        last_human = max(i for i, m in enumerate(messages) if m.type == "human")
        task = str(messages[last_human].content)
        tools = [m for m in messages if isinstance(m, ToolMessage)]
        current = [m for m in messages[last_human + 1 :] if isinstance(m, ToolMessage)]
        for m in tools:
            if m.tool_call_id not in self._seen_tool_results:
                self._seen_tool_results.append(m.tool_call_id)
        profiles = [json.loads(m.content) for m in tools if m.name == "profile_csv"]
        if not profiles:
            return self._call("profile_csv", {})
        profile = profiles[-1]
        if profile.get("status") != "completed":
            raise ValueError("The fixture requires a valid profile")
        searches = [
            json.loads(m.content) for m in current if m.name == "search_materials"
        ]
        needs_search = bool(
            re.search(r"资料|竞赛|说明|依据|方法|requirements|material", task, re.I)
        )
        if (
            "search_materials" in self._available_tools
            and needs_search
            and not searches
        ):
            names = " ".join(c["name"] for c in profile["columns"])
            query = (
                "quasar_redshift"
                if "quasar_redshift" in task
                else (names + " " + task)[:300]
            )
            return self._call("search_materials", {"query": query})
        positions = {c["name"]: c["position"] for c in profile["columns"]}
        is_medal = {"NOC", "Total", "Year"} <= set(positions)
        calculation = is_medal and bool(re.search(r"前|奖牌|2024|top|rank", task, re.I))
        analyses = [json.loads(m.content) for m in current if m.name == "analyze_csv"]
        distributions = [json.loads(m.content) for m in current if m.name == "prepare_distribution"]
        delegations = [json.loads(m.content) for m in current if m.name == "delegate_visualization"]
        wants_chart = bool(re.search(r"图表|柱状图|折线图|直方图|可视化|绘图|\b(chart|plot|histogram)\b", task, re.I))
        if re.search(r"(?:不要|无需|不需要|不用|不执行)[^。；\n]{0,50}(?:图|plot)|只(?:要|需).*表|(?:no|without)\s+(?:charts?|plots?)", task, re.I):
            wants_chart = False
        style_only = wants_chart and bool(re.search(r"改为|改成|修改|重新绘", task)) and not re.search(r"范围|筛选|\d{4}|前\s*\d+", task)
        if style_only and not delegations:
            previous_data = [json.loads(message.content) for message in tools if message.name in {"analyze_csv", "prepare_distribution"}]
            if previous_data:
                return self._call("delegate_visualization", {"data_id": previous_data[-1]["data_id"], "instruction": task})
        if wants_chart and not is_medal:
            numerical = [column for column in profile["columns"] if column["inferred_type"] == "numeric"]
            named = [column for column in numerical if column["name"].casefold() in task.casefold()]
            value_column = (named or numerical)[-1] if numerical else None
            if value_column is None:
                raise ValueError("This offline visualization fixture requires a numeric column.")
            if re.search(r"直方图|histogram|分布", task, re.I) and not distributions:
                match = re.search(r"(\d+)\s*(?:箱|bins)", task, re.I)
                return self._call("prepare_distribution", {"column_position": value_column["position"], "filters": [], "bins": int(match.group(1)) if match else 10})
            if not re.search(r"直方图|histogram|分布", task, re.I) and not analyses:
                groups = [column for column in profile["columns"] if column["position"] != value_column["position"]]
                group = groups[0]
                operation = "mean" if re.search(r"均值|平均|mean|average", task, re.I) else "sum"
                spec = {"filters": [], "group_by": [group["position"]], "metrics": [{"op": operation, "column": value_column["position"], "alias": "value"}], "order_by": [{"field": "value", "direction": "desc"}], "top_k": None}
                return self._call("analyze_csv", {"spec": spec})
            if not delegations:
                data = (distributions or analyses)[-1]
                return self._call("delegate_visualization", {"data_id": data["data_id"], "instruction": task})
        if calculation and not analyses:
            replies = [
                json.loads(m.content)["reply"] for m in tools if m.name == "ask_user"
            ]
            combined = task + " " + " ".join(replies)
            if not re.search(r"全部|所有|累计|2024|all years", combined, re.I):
                return self._call(
                    "ask_user",
                    {
                        "question": "请确认年份范围、奖牌指标及历史国家标签。可回复：全部年份、按 Total 累计、保留原始 NOC。"
                    },
                )
            filters = (
                [{"column": positions["Year"], "op": "eq", "value": 2024}]
                if "2024" in task
                else []
            )
            spec = {
                "filters": filters,
                "group_by": [positions["NOC"]],
                "metrics": [
                    {"op": "sum", "column": positions["Total"], "alias": "total_medals"}
                ],
                "order_by": [{"field": "total_medals", "direction": "desc"}],
                "top_k": 5,
            }
            return self._call("analyze_csv", {"spec": spec})
        if wants_chart and is_medal and analyses and not delegations:
            return self._call("delegate_visualization", {"data_id": analyses[-1]["data_id"], "instruction": task})
        answer = f"实际概览：{profile['source']['name']}，{profile['row_count']} 行、{profile['column_count']} 列；字段：{', '.join(c['name'] for c in profile['columns'])}。"
        if calculation:
            answer += "已按原始 NOC 分组，对 Total 求和，按奖牌数降序取前五。" + (
                "范围为 2024 年。" if "2024" in task else "范围为全部年份。"
            )
        if searches:
            chunks = [c for s in searches for c in s.get("matches", [])]
            for c in chunks:
                loc = c["location"]
                position = (
                    f"第{loc['page']}页"
                    if "page" in loc
                    else f"第{loc['line_start']}–{loc['line_end']}行"
                )
                answer += f"\n[{c['id']}]（{c['name']}，{position}）：{c['text'][:140].replace('[D', '［D')}"
            if not chunks:
                answer += "\n检索无命中，资料不足，不能提供资料依据。"
        answer += "\n离线模式是固定验收流程；通用任务解释需在线模型。"
        result = {
            "task_kind": "calculation" if calculation or analyses or distributions else "summary",
            "retrieval_reason": "当前任务需要资料依据"
            if searches
            else "当前任务可以由 CSV 数据独立完成",
            "answer": answer,
        }
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=json.dumps(result, ensure_ascii=False))
                )
            ]
        )


class OfflineVisualizationModel(OfflineIntakeModel):
    """Deterministic chart-demo model executing the actual isolated child graph."""

    @property
    def _llm_type(self):
        return "labweaver-offline-visualization"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [v.get("name") if isinstance(v, dict) else v.name for v in tools]
        if not set(names) <= {"get_plot_data", "render_chart"}:
            raise ValueError("Unexpected visualization tools")
        self._bound_tool_sets.append(names)
        self._available_tools = names
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        request = json.loads(next(m.content for m in messages if m.type == "human"))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        for message in results:
            if message.tool_call_id not in self._seen_tool_results:
                self._seen_tool_results.append(message.tool_call_id)
        data_messages = [json.loads(m.content) for m in results if m.name == "get_plot_data"]
        renders = [json.loads(m.content) for m in results if m.name == "render_chart"]
        if not data_messages:
            return self._call("get_plot_data", {"data_id": request["data_id"]})
        if not renders:
            data = data_messages[-1]["data"]
            instruction = request["instruction"]
            histogram = data.get("kind") == "distribution"
            line = bool(re.search(r"折线|趋势|line|trend", instruction, re.I))
            spec = {"chart_type": "histogram" if histogram else "line" if line else "bar", "title": "数值分布" if histogram else "统计结果"}
            if not histogram:
                spec.update(x=data["group_columns"][0]["field"], y=data["spec"]["metrics"][0]["alias"])
            if re.search(r"横向|horizontal", instruction, re.I):
                spec["orientation"] = "horizontal"
            return self._call("render_chart", {"data_id": request["data_id"], "spec": spec})
        if renders[-1].get("status") != "completed":
            raise ValueError("Offline chart rendering failed")
        content = {"chart_ids": [result["chart"]["chart_id"] for result in renders], "reason": "根据已登记真实数据完成图型选择和绘图。"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(content, ensure_ascii=False)))])
