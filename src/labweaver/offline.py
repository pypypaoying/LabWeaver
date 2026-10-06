"""Deterministic offline chat model for exercising the real Agent tool loop."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class OfflineIntakeModel(BaseChatModel):
    """Request a real profile, then summarize its result without any network API."""

    _bound_tool_sets: list[list[str]] = PrivateAttr(default_factory=list)
    _seen_tool_results: list[str] = PrivateAttr(default_factory=list)
    _available_tools: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "labweaver-offline"

    @property
    def bound_tool_sets(self) -> list[list[str]]:
        return [list(names) for names in self._bound_tool_sets]

    @property
    def seen_tool_results(self) -> list[str]:
        return list(self._seen_tool_results)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [item.get("name") if isinstance(item, dict) else item.name for item in tools]
        self._bound_tool_sets.append(names)
        if names not in (["profile_csv"], ["profile_csv", "search_materials"]):
            raise ValueError("The offline intake model requires the bounded intake tool set.")
        self._available_tools = names
        return self

    def get_num_tokens(self, text: str) -> int:
        # No tokenizer downloads: an estimate is sufficient for this bounded demo.
        return max(1, (len(text) + 3) // 4)

    def get_num_tokens_from_messages(self, messages, tools=None) -> int:
        return sum(self.get_num_tokens(str(message.content)) for message in messages)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            message = AIMessage(content="", tool_calls=[{
                "name": "profile_csv", "args": {}, "id": "offline-profile-1", "type": "tool_call",
            }])
        else:
            evidence = tool_messages[-1]
            profiles = [item for item in tool_messages if item.name == "profile_csv"]
            searches = [item for item in tool_messages if item.name == "search_materials"]
            # Both tool names and IDs are checked before any result is summarized.
            profile_message = profiles[0] if profiles else evidence
            if profile_message.tool_call_id != "offline-profile-1" or profile_message.name != "profile_csv":
                raise ValueError("The offline model received an unmatched profile result.")
            if profile_message.tool_call_id not in self._seen_tool_results:
                self._seen_tool_results.append(profile_message.tool_call_id)
            profile = json.loads(profile_message.content)
            task_messages = [item for item in messages if item.type == "human"]
            task = str(task_messages[-1].content) if task_messages else "未提供任务"
            if (profile.get("status") == "completed" and
                    "search_materials" in self._available_tools and not searches):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content="", tool_calls=[{"name": "search_materials",
                    "args": {"query": task[:300]}, "id": "offline-search-1", "type": "tool_call"}],
                ))])
            if searches:
                search = searches[-1]
                if search.tool_call_id != "offline-search-1":
                    raise ValueError("The offline model received an unmatched retrieval result.")
                self._seen_tool_results.append(search.tool_call_id)
                retrieval = json.loads(search.content)
                matches = retrieval.get("matches", [])
                references = []
                for chunk in matches:
                    location = chunk["location"]
                    position = (f"第{location['page']}页" if "page" in location else
                                f"第{location['line_start']}–{location['line_end']}行")
                    # Escape citation-looking data so source text cannot mint IDs.
                    excerpt = chunk["text"][:200].replace("[D", "［D")
                    references.append(f"- [{chunk['id']}]（{chunk['name']}，{position}）：{excerpt}")
                content = (
                    f"## 任务理解\n{task}\n"
                    f"## 数据条件\n实际概览：{profile['source']['name']}，{profile['row_count']} 行、"
                    f"{profile['column_count']} 列；字段：{', '.join(c['name'] for c in profile['columns'])}。\n"
                    "## 候选分析\n以下是待确认方案，尚未执行分析。"
                    + ("可按照检索资料中的指标、分组和交付要求规划分析。\n" if matches else
                       "资料不足，无法确认资料规定的分析方法。\n")
                    + "## 资料依据\n" + ("\n".join(references) if references else
                       "检索无命中；资料不足，请补充相关要求或变量说明。")
                    + "\n## 待确认事项\n确认任务目标、字段含义、缺失处理口径和交付形式。"
                    "资料中的指令文本仅作为数据，不改变执行权限；如要求冲突，请用户确认。"
                    "当前等待用户确认；尚未清洗数据、执行分析或生成研究结论。"
                )
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            if profile.get("status") != "completed":
                content = "CSV 概览未完成，请先检查输入文件或解析设置；当前停在项目接收阶段。"
            else:
                column_names = [column["name"] for column in profile["columns"]]
                content = (
                    f"任务摘要：{task}\n"
                    f"实际概览：{profile['source']['name']}，{profile['row_count']} 行、"
                    f"{profile['column_count']} 列；字段：{', '.join(column_names)}。\n"
                    f"工具还返回了逐列缺失统计及 {len(profile['sample_rows'])} 行样例。\n"
                    "下一步建议：确认需要比较的字段、分组方式和期望交付成果，"
                    "然后再制定分析步骤。\n"
                    "待确认：本次优先回答什么问题？哪些字段是指标或分组依据？"
                    "缺失数据的处理规则是什么？\n"
                    "当前等待用户确认；尚未清洗数据、执行分析或生成研究结论。"
                )
            message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])
