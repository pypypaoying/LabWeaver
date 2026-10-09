"""Public answer/progress events from a live graph; never print reasoning tokens."""

import json
import re


def _answer_prefix(text: str) -> str:
    """Decode the available prefix of the top-level JSON answer string only."""
    text = re.sub(r"^```(?:json)?\s*", "", text.lstrip())
    if not text.startswith("{"):
        return ""
    decoder = json.JSONDecoder()
    position = 1
    try:
        while True:
            position += len(text[position:]) - len(text[position:].lstrip())
            key, position = decoder.raw_decode(text, position)
            position += len(text[position:]) - len(text[position:].lstrip())
            if text[position] != ":":
                return ""
            position += 1
            position += len(text[position:]) - len(text[position:].lstrip())
            if key == "answer":
                if text[position] != '"':
                    return ""
                start = position + 1
                position = start
                while position < len(text):
                    char = text[position]
                    if char == '"':
                        break
                    if char == "\\":
                        if position + 1 >= len(text):
                            break
                        size = 6 if text[position + 1] == "u" else 2
                        if position + size > len(text):
                            break
                        if (
                            size == 6
                            and 0xD800
                            <= int(text[position + 2 : position + 6], 16)
                            <= 0xDBFF
                        ):
                            # Do not emit an incomplete UTF-16 surrogate pair.
                            if position + 12 > len(text):
                                break
                            size = 12
                        position += size
                    else:
                        position += 1
                value = json.loads('"' + text[start:position] + '"')
                if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
                    return ""
                return value
            _, position = decoder.raw_decode(text, position)
            position += len(text[position:]) - len(text[position:].lstrip())
            if text[position] != ",":
                return ""
            position += 1
    except (ValueError, IndexError, TypeError):
        return ""


def stream_execution(graph, command, config, on_event):
    """Drive the graph exactly once, retaining interrupts and authoritative state.

    answer_delta is provisional until a subsequent validated event. No raw JSON,
    tool arguments, retrieved content, child-agent text or reasoning is emitted.
    """
    result, buffers, prefixes, starts, ends = {}, {}, {}, set(), set()
    interrupts = ()
    for event in graph.stream(
        command,
        config=config,
        stream_mode=["messages", "updates", "values", "custom"],
        version="v2",
    ):
        # Child graph progress is emitted by host tools, never from model text.
        # Its namespace can be nested even when the parent writer is captured.
        if event["ns"] and event["type"] != "custom":
            continue
        data = event["data"]
        if event["type"] == "custom":
            if isinstance(data, dict) and data.get("type") in {
                "plan",
                "execution_start",
                "execution_end",
            }:
                on_event(data)
        elif event["type"] == "values":
            result = data
        elif event["type"] == "updates":
            if "__interrupt__" in data:
                interrupts = data["__interrupt__"]
            for update in data.values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    for call in getattr(message, "tool_calls", []) or []:
                        if call["id"] not in starts:
                            starts.add(call["id"])
                            on_event(
                                {
                                    "type": "tool_start",
                                    "name": call["name"],
                                    "id": call["id"],
                                }
                            )
                    if (
                        getattr(message, "type", "") == "tool"
                        and message.tool_call_id not in ends
                    ):
                        ends.add(message.tool_call_id)
                        try:
                            status = json.loads(message.content).get(
                                "status", "unknown"
                            )
                        except (ValueError, TypeError, AttributeError):
                            status = "unknown"
                        on_event(
                            {
                                "type": "tool_end",
                                "name": message.name,
                                "id": message.tool_call_id,
                                "status": status,
                            }
                        )
        elif event["type"] == "messages":
            message, metadata = data
            namespace = metadata.get("langgraph_checkpoint_ns", "")
            if (
                metadata.get("langgraph_node") != "model"
                or not re.fullmatch(r"model:[^|]+", namespace)
                or getattr(message, "tool_calls", None)
                or getattr(message, "tool_call_chunks", None)
            ):
                continue
            key = message.id or namespace
            content = message.content
            if isinstance(content, list):
                content = "".join(
                    v.get("text", "")
                    for v in content
                    if isinstance(v, dict) and v.get("type") == "text"
                )
            if not isinstance(content, str):
                continue
            buffers[key] = buffers.get(key, "") + content
            prefix = _answer_prefix(buffers[key])
            previous = prefixes.get(key, "")
            if prefix.startswith(previous) and len(prefix) > len(previous):
                on_event(
                    {"type": "answer_delta", "id": key, "text": prefix[len(previous) :]}
                )
                prefixes[key] = prefix
    return {**result, "__interrupt__": interrupts} if interrupts else result


class ConsoleStream:
    """Print progress and answer tokens immediately, with explicit validation."""

    def __init__(self):
        self.answer = ""
        self.answer_id = None
        self.validated_answer = ""
        self.line_open = False

    def __call__(self, event):
        kind = event["type"]
        if kind == "answer_delta":
            if event["id"] != self.answer_id:
                print("\nAgent 回答（生成中，待核验）：", flush=True)
                self.answer_id, self.answer = event["id"], ""
            print(event["text"], end="", flush=True)
            self.answer += event["text"]
            self.line_open = True
            return
        if self.line_open:
            print(flush=True)
            self.line_open = False
        if kind == "plan":
            print(
                "[计划] " + "；".join(d["description"] for d in event["deliverables"]),
                flush=True,
            )
        elif kind == "completion_feedback":
            print(
                "[核验] 先前生成文本尚未通过；正在补充真实执行。"
                if event["reason"] == "missing_execution"
                else "[核验] 正在修正回答格式，保留已有成果。",
                flush=True,
            )
        elif kind == "execution_start":
            print(
                f"[代码执行] 第 {event['attempt']} 次"
                + ("（修正）" if event["attempt"] > 1 else ""),
                flush=True,
            )
        elif kind == "execution_end":
            print(f"[代码结果] {event['status']}", flush=True)
        elif kind == "tool_start":
            print(f"[执行] {event['name']}", flush=True)
        elif kind == "tool_end":
            print(f"[结果] {event['name']}：{event['status']}", flush=True)
        elif kind == "validated":
            self.validated_answer = (
                event["answer"]
                if event["status"] == "completed" and self.answer == event["answer"]
                else ""
            )
            if self.answer:
                print(
                    "[核验] 已通过执行证据校验。"
                    if event["status"] == "completed"
                    else "[核验] 本次尚未完成，以上生成文本不作为已完成结果。",
                    flush=True,
                )
