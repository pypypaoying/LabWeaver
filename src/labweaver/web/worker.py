"""Owned Agent worker. Generated Python still executes exclusively in Docker."""

import json
import os
import multiprocessing
from pathlib import Path
import threading

from .security import contains_secret, managed_path, scrub, secret_values


def agent_worker(settings, app_id, csv_path, materials, context, commands, events, cancellation):
    # The worker is backend-only. Tracing is always disabled.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if settings.runtime_dir is None:
        raise ValueError("Worker requires a host-owned runtime directory")
    work = managed_path(settings.data_dir, settings.runtime_dir.relative_to(settings.data_dir).as_posix())
    if not work.is_dir():
        raise ValueError("Worker runtime directory is missing")
    temp = work / "temp"
    temp.mkdir(exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR"):
        os.environ[name] = str(temp)
    import tempfile
    tempfile.tempdir = str(temp)
    os.chdir(work)
    secrets = secret_values(settings)
    session = None
    current = {"run_id": None, "identity": None}
    stopping = threading.Event()

    def send(kind, payload):
        events.put({"run_id": current["run_id"], "kind": kind, "payload": scrub(payload, secrets)})

    def saving_failed(exc, stage, executions):
        diagnostics = {"stage": stage, "exception_type": type(exc).__name__}
        if isinstance(exc, OSError):
            diagnostics["errno"] = exc.errno
        failed = next((record for record in reversed(executions) if record.get("status") == "error"), None)
        send("worker_error", {
            "code": "save_failed",
            "message": "执行记录或成果无法保存；真实代码执行错误保留在诊断中。",
            "exception_type": type(exc).__name__,
            "save_error": diagnostics,
            "execution_error": {
                key: failed[key]
                for key in ("id", "error", "exit_code", "stderr", "container_state", "container_removed")
                if key in failed
            } if failed else None,
        })

    def on_event(event):
        if session is not None and current["identity"] != session.task_id:
            current["identity"] = session.task_id
            send("runtime_identity", {"session_id": session.session_id, "task_id": session.task_id})
        event = dict(event)
        kind = event.pop("type")
        send(kind, event)
        if kind == "execution_end" and session is not None:
            # Persist validated partial assets before another (potentially blocking) model call.
            # This snapshot is never evidence that the whole task completed.
            from labweaver.runtime.records import save_run
            from copy import deepcopy
            assets = session.artifact_assets
            partial = {"report_version": 3, "task_spec":session.task_spec, "session_id": session.session_id, "task_id": session.task_id,
                       "task": session.task, "status": "error", "source": (session.profile_result or {}).get("source"),
                       "task_plan": deepcopy(session.deliverables), "artifacts": [deepcopy(a["metadata"]) for a in assets.values()],
                       "code_executions": deepcopy(session.analyst.attempts), "final_answer": "",
                       "error": {"code": "partial_checkpoint", "message": "已取得部分执行成果，任务尚未核验完成。"}}
            if not contains_secret(partial, secrets) and not any(contains_secret(raw, secrets) for a in assets.values() for raw in a["files"].values()):
                try:
                    path = save_run(partial, settings.data_dir / "exports", artifact_assets=assets, with_brief=False)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    saving_failed(exc, "partial_export", partial["code_executions"])
                    raise
                send("checkpoint_report", {"report": json.loads(path.read_text(encoding="utf-8"))})

    def watch_cancel():
        cancellation.wait()
        stopping.set()
        if session is not None:
            # Avoid cancel() waiting on the Agent operation lock during model IO.
            session.executor.cancel()

    threading.Thread(target=watch_cancel, daemon=True).start()
    def watch_parent():
        parent = multiprocessing.parent_process()
        if parent is None:
            return
        while not stopping.wait(0.5):
            if not parent.is_alive():
                cancellation.set()
                stopping.set()
                # A stuck model request must not outlive the owning API process.
                threading.Timer(5, lambda: os._exit(1)).start()
                if session is not None:
                    session.executor.cancel()
                break
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        from labweaver.agent import create_session
        from labweaver.config import create_model, load_config
        from labweaver.runtime.execution import ExecutionConfig
        if settings.mode == "offline":
            from labweaver.offline import OfflineIntakeModel, OfflineAnalysisModel
            main, child = OfflineIntakeModel(), OfflineAnalysisModel()
        else:
            config = settings.model_config or load_config(settings.env_file, use_default_env=False)
            main = create_model(config)
            child = create_model(config, max_tokens=settings.analysis_max_tokens)
        session = create_session(csv_path, main, material_paths=materials, analysis_model=child,
                                 history_context=context, cancellation_event=cancellation,
                                 skills_enabled=settings.skills_enabled,
                                 execution_config=ExecutionConfig(image=settings.image, timeout=settings.execution_timeout))
        while not stopping.is_set():
            command = commands.get()
            if command["op"] == "shutdown":
                break
            current["run_id"] = command["run_id"]
            session.executor.set_owner(app_id, current["run_id"], on_lifecycle=on_event)
            if command["op"] == "invoke":
                report = session.invoke(command["task"], on_event=on_event)
            elif command["op"] == "resume":
                report = session.resume(command["reply"], on_event=on_event)
            else:
                raise ValueError("Invalid worker command")
            if contains_secret(report, secrets) or any(contains_secret(raw, secrets) for a in session.artifact_assets.values() for raw in a["files"].values()):
                send("worker_error", {"code": "secret_in_output", "message": "输出包含敏感配置，已拒绝保存与交付。"})
                break
            send("saving", {"status": report["status"]})
            try:
                saved = session.save(settings.data_dir / "exports")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                saving_failed(exc, "final_export", report.get("code_executions", []))
                raise
            send("report", {"report": saved, "record_path": saved["record_path"]})
            if report["status"] in {"cancelled", "error"}:
                break
    except BaseException as exc:
        # Never pass an SDK exception, credentials or traceback into chat history.
        send("worker_error", {"code": "worker_failed", "exception_type": type(exc).__name__,
                              "message": "Agent 工作进程失败，请查看环境状态并重新执行。"})
    finally:
        if session is not None:
            session.executor.cancel()
