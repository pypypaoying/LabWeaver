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
            "task_kind": "calculation" if calculation else "summary",
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
