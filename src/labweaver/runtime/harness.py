"""Shared allowlist, actual execution ledger, bounded calls and replay deduplication."""

import copy
import json
import threading
import time
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from labweaver.runtime.evidence import _message_content, _trace_messages
from labweaver.runtime.token_usage import cache_measurements, context_footprint


class HarnessRejected(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ExecutionHarness(AgentMiddleware):
    def __init__(
        self,
        limits,
        *,
        available=None,
        context=None,
        model_limit=12,
        model_counter=None,
        reject_invalid=True,
        repairable_invalid=(),
        cancellation_event=None,
        progress=None,
        role="main",
    ):
        self.limits = limits
        self.available = available or (lambda: set(limits))
        self.context = context or (lambda: "")
        self.model_limit = model_limit
        self.model_counter = model_counter
        self.reject_invalid = reject_invalid
        self.repairable_invalid = frozenset(repairable_invalid)
        self.cancellation_event = cancellation_event
        self.progress = progress or (lambda event: None)
        self.role = role
        self._lock = threading.RLock()
        self._active = threading.local()
        self.new_task()

    def check_cancelled(self):
        """Observe the host signal before admitting work or charging a budget."""
        if self.cancellation_event is not None and self.cancellation_event.is_set():
            raise HarnessRejected("cancelled")

    def new_task(self):
        self.model_calls = self.tool_attempts = 0
        self.tool_counts = dict.fromkeys(self.limits, 0)
        self.execution_ledger, self.events, self.tool_exposure = [], [], []
        self._admitted, self._cached, self._advertised = {}, {}, set()
        self.failure = None
        self.model_diagnostics = []
        self.token_usage = []

    def wrap_model_call(self, request, handler):
        # A truncated JSON/tool response is not executable evidence. One bounded
        # retry may repair the format; every request consumes the same budget.
        for attempt in range(2):
            response = self._model_call(request, handler)
            problem = next((
                "model_output_truncated"
                if (getattr(message, "response_metadata", {}) or {}).get("finish_reason") == "length"
                else "invalid_tool_arguments"
                for message in response.result
                if (getattr(message, "response_metadata", {}) or {}).get("finish_reason") == "length"
                or getattr(message, "invalid_tool_calls", None)
            ), None)
            if not problem:
                self.events.extend(_trace_messages(response.result))
                return response
            self.model_diagnostics.append({"call": self.model_calls, "code": problem})
            if attempt:
                raise HarnessRejected(problem)
            self.progress({"type": "model_retry", "role": self.role,
                           "call": self.model_calls, "error_code": problem})
            request = request.override(messages=[*request.messages, HumanMessage(
                content="宿主格式反馈：上一响应超过输出预算或工具参数不是完整 JSON，未执行任何工具。"
                "请紧凑地重写完整响应；代码需保留所有授权交付项及 emit_* 调用，不能只打印结果。"
            )])

    def _model_call(self, request, handler):
        with self._lock:
            self.check_cancelled()
            if self.failure:
                raise HarnessRejected(self.failure)
            if self.model_counter:
                self.model_counter()
            if self.model_calls >= self.model_limit:
                raise HarnessRejected("model_budget_exhausted")
            self.model_calls += 1
            names = self.available()
            self._advertised = set(names)
            self.tool_exposure.append(sorted(names))
        allowed = [
            t
            for t in request.tools
            if (t.get("name") if isinstance(t, dict) else t.name) in names
        ]
        if len(allowed) != len(names):
            raise HarnessRejected("invalid_tool_configuration")
        start = time.monotonic()
        call = self.model_calls
        system_content = (_message_content(request.system_message)
                          + "\n当前开放工具：" + ", ".join(sorted(names)) + self.context())
        footprint = context_footprint(request.messages, system_content, allowed)
        self.progress({"type": "model_start", "role": self.role, "call": call,
                       "started_monotonic": start})
        try:
            response = handler(
                request.override(
                    tools=allowed,
                    system_message=SystemMessage(
                        content=system_content
                    ),
                )
            )
        except BaseException:
            self.progress({"type": "model_end", "role": self.role, "call": call,
                           "status": "error", "elapsed_seconds": round(time.monotonic() - start, 3)})
            raise
        reasons = [(getattr(m, "response_metadata", {}) or {}).get("finish_reason") for m in response.result]
        # Only normalized metadata leaves the model boundary, never raw content,
        # SDK errors, tool arguments, reasoning, or authentication information.
        reason = next((v for v in reasons if v in {"length", "stop", "tool_calls", "content_filter"}), None)
        usages = [getattr(m,"usage_metadata",None) for m in response.result]
        known = [u for u in usages if isinstance(u,dict) and all(type(u.get(k)) is int and u[k]>=0 for k in ("input_tokens","output_tokens","total_tokens"))]
        usage = {k:sum(u[k] for u in known) for k in ("input_tokens","output_tokens","total_tokens")} if known and len(known)==len(usages) else None
        self.token_usage.append({"call":call,"role":self.role,"usage":usage,
                                 "cache_usage":cache_measurements(response.result),
                                 "context_bytes":footprint})
        self.progress({"type": "model_end", "role": self.role, "call": call,
                       "status": "completed", "finish_reason": reason, "usage": usage,
                       "elapsed_seconds": round(time.monotonic() - start, 3)})
        self.check_cancelled()
        return response

    def _reject(self, call, code):
        self.failure = self.failure or code
        message = ToolMessage(
            content=json.dumps({"status": "error", "error": {"code": code}}),
            name=call.get("name"),
            tool_call_id=call.get("id") or "missing-id",
            status="error",
        )
        self.events.extend(_trace_messages([message]))
        return message

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        cid, name = call.get("id"), call.get("name")
        with self._lock:
            self.check_cancelled()
            if not isinstance(cid, str) or not cid:
                return self._reject(call, "unmatched_tool_evidence")
            if cid in self._admitted:
                if self._admitted[cid] != call:
                    return self._reject(call, "reused_tool_call_id")
                if cid in self._cached:
                    return ToolMessage(
                        content=json.dumps(
                            self._cached[cid], ensure_ascii=False, allow_nan=False
                        ),
                        name=name,
                        tool_call_id=cid,
                    )
            else:
                if name not in self.limits or name not in self._advertised:
                    return self._reject(
                        call,
                        "tool_before_profile"
                        if name == "search_materials"
                        and self._advertised == {"profile_csv"}
                        else "tool_not_allowed",
                    )
                self.tool_attempts += 1
                self.tool_counts[name] += 1
                if self.tool_counts[name] > self.limits[name]:
                    return self._reject(
                        call,
                        "retrieval_budget_exhausted"
                        if name == "search_materials"
                        else "tool_budget_exhausted",
                    )
                self._admitted[cid] = copy.deepcopy(call)
        self._active.call_id = cid
        try:
            response = handler(request)
            if isinstance(response, ToolMessage):
                if cid not in self._cached:
                    issues = []
                    schema = getattr(getattr(request, "tool", None), "args_schema", None)
                    if name in self.repairable_invalid and hasattr(schema, "model_validate"):
                        from pydantic import ValidationError
                        try:
                            schema.model_validate(call.get("args", {}))
                        except ValidationError as exc:
                            # Never echo raw SDK errors, input values or validator context.
                            issues = [{"location": list(e["loc"]), "type": e["type"], "message": e["msg"][:500]}
                                      for e in exc.errors(include_input=False, include_context=False, include_url=False)[:10]]
                    self.record(
                        {
                            "status": "error",
                            "error": {"code": "invalid_tool_arguments"},
                            **({"argument_errors": issues,
                                "repair_instruction": "按工具 JSON schema 修正字段位置和类型，重新提交完整参数；本次未执行工具，仍计入原预算。"} if issues else {}),
                        },
                        name,
                    )
                    response = ToolMessage(
                        content=json.dumps(self._cached[cid], ensure_ascii=False),
                        name=name,
                        tool_call_id=cid,
                    )
                    if self.reject_invalid and not issues:
                        self.failure = "invalid_tool_arguments"
                self.events.extend(_trace_messages([response]))
            return response
        finally:
            self._active.call_id = None

    def record(self, result, name):
        with self._lock:
            cid = getattr(self._active, "call_id", None)
            if cid not in self._admitted:
                raise HarnessRejected("unadmitted_execution")
            if cid not in self._cached:
                self.execution_ledger.append(
                    {
                        "tool_call_id": cid,
                        "name": name,
                        "arguments": copy.deepcopy(self._admitted[cid]["args"]),
                        "result": copy.deepcopy(result),
                    }
                )
                self._cached[cid] = copy.deepcopy(result)
            return result


def validate_evidence(trace, ledger, *, pending=False):
    calls = {item["id"]: item for item in trace if item["kind"] == "tool_call"}
    returns = {
        item["tool_call_id"]: item for item in trace if item["kind"] == "tool_result"
    }
    executions = {item["tool_call_id"]: item for item in ledger}
    if set(returns) - set(calls) or set(executions) - set(calls):
        return "unmatched_tool_evidence"
    if (
        len(calls) != sum(item["kind"] == "tool_call" for item in trace)
        or len(returns) != sum(item["kind"] == "tool_result" for item in trace)
        or len(executions) != len(ledger)
    ):
        return "unmatched_tool_evidence"
    for cid, call in calls.items():
        if (
            pending
            and call["name"] in {"ask_user", "profile_csv"}
            and cid not in returns
            and cid not in executions
        ):
            continue
        if cid not in returns or cid not in executions:
            return "missing_execution_evidence"
        result, actual = returns[cid], executions[cid]
        try:
            if (
                actual["name"] != call["name"]
                or result["name"] != call["name"]
                or actual["arguments"] != call["args"]
                or json.loads(result["content"]) != actual["result"]
                or result["status"] == "error"
            ):
                return "unmatched_tool_evidence"
        except (ValueError, KeyError, TypeError):
            return "invalid_tool_result"
    if set(returns) - set(calls) or set(executions) - set(calls):
        return "unmatched_tool_evidence"
    return None
