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
        if names != ["profile_csv"]:
            raise ValueError("The offline intake model requires only profile_csv.")
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
            if evidence.tool_call_id != "offline-profile-1" or evidence.name != "profile_csv":
                raise ValueError("The offline model received an unmatched tool result.")
            self._seen_tool_results.append(evidence.tool_call_id)
            profile = json.loads(evidence.content)
            if profile.get("status") != "completed":
                content = "CSV 概览未完成，请先检查输入文件或解析设置；当前停在项目接收阶段。"
            else:
                column_names = [column["name"] for column in profile["columns"]]
                task_messages = [message for message in messages if message.type == "human"]
                task = str(task_messages[-1].content) if task_messages else "未提供任务"
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
