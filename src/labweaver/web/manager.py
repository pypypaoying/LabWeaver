"""One active task, supervised spawned runtimes, durable real events and atomic registration."""

import json
from dataclasses import replace
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from .context import history_context
from .files import MEDIA, file_view, import_file, read_registered
from .security import AppError, contains_secret, digest, identifier, managed_path, purge_managed_tree, scrub, secret_values
from .store import Store, encode, now, uid
from .worker import agent_worker
from .providers import ProviderRegistry
from .deadlines import RunDeadline

TERMINAL = {"completed", "error", "cancelled", "interrupted"}
ACTIVE = {"running", "awaiting_input", "cancelling"}


class ApplicationLock:
    """Prevent two service processes from recovering/cleaning each other's active runtime."""
    def __init__(self, root, *, filename='.application.lock'):
        root = Path(root).absolute()
        managed_path(root, filename)
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        self.root = root
        self.file = managed_path(root, filename).open("a+b")
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise AppError("already_running", "该数据目录已有应用运行，请退出原服务后重试。", 409) from None

    def close(self):
        if not self.file.closed:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()


class Runtime:
    def __init__(self, settings, app_id, conversation_id, csv_path, materials, context, target):
        ctx = mp.get_context("spawn")
        self.conversation_id, self.run_id = conversation_id, None
        self.model_config = settings.model_config
        self.selection_key = None
        # Host-generated IDs establish ownership; PIDs may be reused after a crash.
        self.workspace_relative = f"runtime/{identifier(conversation_id)}/{uid()}"
        self.workspace_root = settings.data_dir
        self.workspace_cleaned = False
        work = managed_path(settings.data_dir, self.workspace_relative)
        work.mkdir(parents=True, exist_ok=False)
        settings = replace(settings, runtime_dir=work)
        self.commands, self.events, self.cancel_signal = ctx.Queue(), ctx.Queue(), ctx.Event()
        self.process = ctx.Process(target=target, args=(settings, app_id, csv_path, materials, context,
                                                     self.commands, self.events, self.cancel_signal), daemon=False)
        self.known_containers = {}
        self.deadline = None
        self.last_timing_at = None
        self.closed = False
        try:
            self.process.start()
        except BaseException:
            self._cleanup_workspace()
            raise

    def _cleanup_workspace(self):
        try:
            purge_managed_tree(self.workspace_root, self.workspace_relative)
            self.workspace_cleaned = True
        except (OSError, AppError):
            self.workspace_cleaned = False

    def stop(self, *, force=False):
        if self.closed:
            if not self.process.is_alive():
                if self.deadline is not None:
                    self.deadline.finish_models()
                self._cleanup_workspace()
            return
        self.closed = True
        if self.process.is_alive():
            if not force:
                self.commands.put({"op": "shutdown"})
                self.process.join(timeout=1)
            if self.process.is_alive():
                if os.name == "nt":
                    try:
                        subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True, timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        self.process.terminate()
                else:
                    self.process.terminate()
                self.process.join(timeout=3)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=3)
        if not self.process.is_alive():
            if self.deadline is not None:
                self.deadline.finish_models()
            self._cleanup_workspace()


class TaskManager:
    def __init__(self, settings, *, worker_target=agent_worker):
        self.settings, self.worker_target = settings, worker_target
        self.application_lock = ApplicationLock(settings.data_dir)
        settings = replace(settings, data_dir=self.application_lock.root)
        self.store = Store(settings.data_dir)
        # Use the canonical storage root in workers, export registration and
        # their temporary directories; nominal AppData may be virtualized.
        self.settings = replace(settings, data_dir=self.store.root)
        self.providers = ProviderRegistry(self.store, self.settings)
        self.lock = threading.RLock()
        self.active = None
        self.runtime = None
        self.closed = False
        self.stopping = False
        self.secrets = tuple(set(secret_values(settings)) | set(self.providers.secret_values()))
        self._health = None
        # Recover metadata only. Never restore an interrupted graph or auto-run a task.
        with self.store.connect() as db:
            unfinished = [dict(r) for r in db.execute("SELECT * FROM runs WHERE status IN ('running','awaiting_input','cancelling')")]
        for run in unfinished:
            cleaned = self.cleanup_owned(run["id"])
            self._recover_exports(run)
            self.fail(run["id"], "interrupted", "服务曾中断；请按原任务重新执行。", status="interrupted",
                      extra={"cleanup_confirmed": cleaned})

    def emit(self, run_id, kind, payload):
        role = "analysis_agent" if kind in {"execution_start", "execution_end"} else "host" if kind in {"container_lifecycle", "runtime_identity", "saving", "artifacts_saved", "timing", "verification_start", "verification_end"} else "main_agent"
        return self.store.append_event(run_id, kind, scrub({"role": role, **payload}, self.secrets))

    def health(self, *, refresh=False):
        if self._health is None or refresh:
            catalog = self.providers.list()
            model = {"ready": self.settings.mode == "offline" or any(p["has_key"] and p["models"] for p in catalog["providers"]),
                     "default_provider_id": catalog["default_provider_id"], "default_model_id": catalog["default_model_id"]}
            if not model["ready"]:
                model["error"] = "请在设置页面配置服务商、API Key 和模型，也可沿用已有后端 .env。"
            from labweaver.runtime.execution import DockerExecutor, ExecutionConfig, ExecutionError
            try:
                executor = DockerExecutor(ExecutionConfig(image=self.settings.image))
                _, image = executor.preflight()
                docker = {"ready": True, "image_id": image}
            except ExecutionError as exc:
                docker = {"ready": False, "error": {"code": exc.code, "message": str(exc)}}
            self._health = {"mode": self.settings.mode, "model": model, "docker": docker}
        return self._health | {"active_run_id": self.active}

    def _refresh_secrets(self):
        # Retain old keys in memory for delayed events from an old worker.
        self.secrets = tuple(set(self.secrets) | set(self.providers.secret_values()))
        self._health = None

    def configure_provider(self, provider_id, payload):
        with self.lock:
            if self.active:
                raise AppError("task_active", "请完成当前任务或停止后再修改服务商配置。", 409)
            if contains_secret({k: v for k, v in payload.items() if k != "api_key"}, self.secrets):
                raise AppError("secret_in_provider", "公开服务商字段不能包含敏感配置。")
            result = self.providers.upsert(provider_id, payload)
            self._refresh_secrets()
            return result

    def configure_default(self, provider_id, model_id):
        with self.lock:
            if self.active:
                raise AppError("task_active", "请完成当前任务或停止后再切换默认模型。", 409)
            result = self.providers.set_default(provider_id, model_id)
            self._refresh_secrets()
            return result

    def _model_for_task(self, conversation_id, provider_id, model_id):
        if (provider_id is None) != (model_id is None):
            raise AppError("model_selection_invalid", "请同时指定服务商和模型。")
        if self.settings.mode == "offline":
            if provider_id is not None:
                raise AppError("offline_model_selection", "离线演示不调用在线服务商；请使用在线模式切换模型。")
            return {"provider_id": "offline", "provider_name": "离线演示", "model_id": "deterministic-demo", "base_url": ""}, None, 0
        catalog = self.providers.list()
        if provider_id is None:
            with self.store.connect() as db:
                stored = db.execute("SELECT selection_json FROM conversation_models WHERE conversation_id=?", (conversation_id,)).fetchone()
            selected = json.loads(stored[0]) if stored else None
            if selected and any(p["id"] == selected["provider_id"] for p in catalog["providers"]):
                provider_id, model_id = selected["provider_id"], selected["model_id"]
            else:
                provider_id, model_id = catalog["default_provider_id"], catalog["default_model_id"]
                default = next(p for p in catalog["providers"] if p["id"] == provider_id)
                # Backward-compatible old API clients may still rely on a non-DeepSeek .env.
                if not default["has_key"]:
                    legacy = next((p for p in catalog["providers"] if p["kind"] == "env" and p["has_key"]), None)
                    if legacy:
                        provider_id, model_id = legacy["id"], legacy["default_model_id"]
        config = self.providers.resolve(provider_id, model_id)
        provider = next(p for p in catalog["providers"] if p["id"] == provider_id)
        self.secrets = tuple(set(self.secrets) | {config.api_key})
        return {"provider_id": provider_id, "provider_name": provider["name"], "model_id": model_id, "base_url": config.base_url}, config, provider["revision"]

    def upload(self, conversation_id, name, raw):
        with self.lock:
            if contains_secret(raw, self.secrets) or contains_secret(name, self.secrets):
                raise AppError("secret_in_input", "输入包含敏感配置，拒绝保存。")
            return import_file(self.store, conversation_id, name, raw, forbidden=self.secrets)

    def remove_file(self, conversation_id, file_id):
        with self.lock, self.store.connect() as db:
            conversation = self.store.require("conversations", conversation_id, db)
            if conversation["frozen"]:
                raise AppError("inputs_frozen", "首次任务已冻结附件。", 409)
            file = self.store.require("files", file_id, db)
            if file["conversation_id"] != conversation_id or file["category"] != "input":
                raise AppError("not_authorized", "附件不属于当前对话。", 403)
            db.execute("DELETE FROM conversation_files WHERE file_id=?", (file_id,))
            db.execute("DELETE FROM files WHERE id=?", (file_id,))
        managed_path(self.store.root, file["relative_path"]).unlink(missing_ok=True)

    def start(self, conversation_id, task, request_id, context_ids=None, provider_id=None, model_id=None):
        with self.lock:
            if self.closed or self.stopping:
                raise AppError('application_stopping', '应用正在退出，请重新启动后分析。', 503)
            if contains_secret(task, self.secrets):
                raise AppError("secret_in_input", "任务包含敏感配置，拒绝保存。")
            conversation = self.store.require("conversations", conversation_id)
            with self.store.connect() as db:
                previous = db.execute("SELECT id,task FROM runs WHERE conversation_id=? AND request_id=?", (conversation_id, request_id)).fetchone()
                if previous:
                    if previous["task"] != task:
                        raise AppError("request_reused", "请求 ID 已用于不同任务。", 409)
                    if provider_id is not None or model_id is not None:
                        prior = db.execute("SELECT selection_json FROM run_models WHERE run_id=?", (previous["id"],)).fetchone()
                        selection = json.loads(prior[0]) if prior else {}
                        if selection.get("provider_id") != provider_id or selection.get("model_id") != model_id:
                            raise AppError("request_reused", "请求 ID 已用于不同模型的任务。", 409)
                    return self.run_view(previous["id"])
                bound = [dict(r) for r in db.execute("SELECT f.*,cf.role FROM files f JOIN conversation_files cf ON cf.file_id=f.id WHERE cf.conversation_id=? ORDER BY cf.position", (conversation_id,))]
            if self.active:
                raise AppError("task_active", "已有活动任务；请完成回复或停止该任务后再开始。", 409)
            csv = next((f for f in bound if f["role"] == "csv"), None)
            if csv is None:
                raise AppError("csv_required", "请先添加一份 CSV。")
            for file in bound:
                read_registered(self.store, file["id"])
            context, selected_ids = history_context(self.store, conversation_id, context_ids)
            selection, model_config, revision = self._model_for_task(conversation_id, provider_id, model_id)
            selection_key = (selection["provider_id"], selection["model_id"], revision)
            # Refresh a completed runtime only when changing conversations or selecting history explicitly.
            if self.runtime and (not self.runtime.process.is_alive() or self.runtime.conversation_id != conversation_id or context_ids is not None
                                 or getattr(self.runtime, "selection_key", None) != selection_key
                                 or getattr(self.runtime, "model_config", None) != model_config):
                self.runtime.stop()
                self.runtime = None
            id, timestamp = uid(), now()
            with self.store.connect() as db:
                db.execute("INSERT INTO runs(id,conversation_id,request_id,task,status,created_at,updated_at,context_ids) VALUES (?,?,?,?,?,?,?,?)",
                           (id, conversation_id, request_id, task, "running", timestamp, timestamp, encode(selected_ids)))
                db.execute("INSERT INTO messages(id,conversation_id,run_id,role,content,state,created_at,request_id) VALUES (?,?,?,?,?,?,?,?)",
                           (uid(), conversation_id, id, "user", task, "sent", timestamp, request_id))
                title = task[:40] if conversation["title"] == "新对话" else conversation["title"]
                db.execute("UPDATE conversations SET frozen=1,title=?,updated_at=? WHERE id=?", (title, timestamp, conversation_id))
                db.execute("INSERT INTO run_models VALUES (?,?)", (id, encode(selection)))
                db.execute("INSERT INTO conversation_models VALUES (?,?) ON CONFLICT(conversation_id) DO UPDATE SET selection_json=excluded.selection_json", (conversation_id, encode(selection)))
            self.active = id
            deadline = RunDeadline(self.settings.model_timeout, self.settings.task_timeout)
            self.emit(id, "running", {"status": "running", "history_run_ids": selected_ids,
                                      "model_selection": selection, "timing": deadline.snapshot()})
            try:
                if self.runtime is None:
                    runtime_settings = replace(self.settings, model_config=model_config)
                    self.runtime = Runtime(runtime_settings, self.store.app_id, conversation_id,
                                           managed_path(self.store.root, csv["relative_path"]),
                                           [managed_path(self.store.root, f["relative_path"]) for f in bound if f["role"] == "material"],
                                           context, self.worker_target)
                    self.runtime.selection_key = selection_key
                    self.runtime.model_config = model_config
                    self.runtime.run_id = id
                    self.runtime.deadline = deadline
                    threading.Thread(target=self._listen, args=(self.runtime,), daemon=True).start()
                self.runtime.run_id = id
                self.runtime.deadline = deadline
                self.runtime.last_timing_at = None
                self._emit_timing(self.runtime, force=True)
                self.runtime.cancel_signal.clear()
                self.runtime.commands.put({"op": "invoke", "run_id": id, "task": task})
            except Exception:
                deadline.stop()
                self.emit(id, "timing", {"timing": deadline.snapshot()})
                self.fail(id, "worker_start_failed", "无法启动 Agent 工作进程。")
            return self.run_view(id)

    def reply(self, run_id, question_id, reply, request_id):
        if contains_secret(reply, self.secrets):
            raise AppError("secret_in_input", "回复包含敏感配置，拒绝保存。")
        with self.lock:
            with self.store.connect() as db:
                run = self.store.require("runs", run_id, db)
                old = db.execute("SELECT content,question_id FROM messages WHERE run_id=? AND request_id=?", (run_id, request_id)).fetchone()
                if old:
                    if old["content"] != reply or old["question_id"] != question_id:
                        raise AppError("request_reused", "请求 ID 已用于不同回复。", 409)
                    return self.run_view(run_id)
                pending = json.loads(run["pending_json"] or "null")
                if run["status"] != "awaiting_input" or not pending or pending["id"] != question_id:
                    raise AppError("question_expired", "问题已回答或运行时已失效，不能恢复旧节点。", 409)
                if self.active != run_id or not self.runtime or not self.runtime.process.is_alive():
                    raise AppError("runtime_expired", "运行时已失效，请重新执行任务。", 409)
                db.execute("INSERT INTO messages(id,conversation_id,run_id,role,content,state,created_at,question_id,request_id) VALUES (?,?,?,?,?,?,?,?,?)",
                           (uid(), run["conversation_id"], run_id, "user", reply, "sent", now(), question_id, request_id))
                db.execute("UPDATE runs SET status='running',pending_json=NULL,updated_at=? WHERE id=?", (now(), run_id))
                db.execute("UPDATE messages SET state='answered' WHERE run_id=? AND question_id=? AND role='assistant'", (run_id, question_id))
            if getattr(self.runtime, "deadline", None):
                self.runtime.deadline.resume()
            self.emit(run_id, "running", {"status": "running", "resumed_question_id": question_id,
                                         "timing": self._timing_snapshot(run_id)})
            self._emit_timing(self.runtime, force=True)
            self.runtime.commands.put({"op": "resume", "run_id": run_id, "reply": reply})
            return self.run_view(run_id)

    def _listen(self, runtime):
        while not runtime.closed and not self.closed:
            if self._deadline_exceeded(runtime):
                break
            self._emit_timing(runtime)
            try:
                item = runtime.events.get(timeout=0.5)
            except queue.Empty:
                if not runtime.process.is_alive():
                    with self.lock:
                        if self.active == runtime.run_id:
                            if self.store.require("runs", runtime.run_id)["status"] == "cancelling":
                                break  # The cancellation supervisor owns cleanup and final state.
                            cleaned = self.cleanup_owned(runtime.run_id, runtime)
                            self.fail(runtime.run_id, "worker_crashed", "工作进程意外退出，已保留执行记录。", extra={"cleanup_confirmed": cleaned})
                    break
                continue
            run_id = item.get("run_id") or runtime.run_id
            if not run_id:
                continue
            with self.lock:
                if runtime.closed or self.closed:
                    break  # An idle conversation may have been deleted while waiting for the lock.
                run = self.store.require("runs", run_id)
                if run["status"] in TERMINAL:
                    continue
                kind, payload = item["kind"], scrub(item["payload"], self.secrets)
                if getattr(runtime, "deadline", None):
                    runtime.deadline.observe(kind, payload)
                if kind == "runtime_identity":
                    with self.store.connect() as db:
                        db.execute("UPDATE runs SET core_session_id=?,core_task_id=? WHERE id=?", (payload["session_id"], payload["task_id"], run_id))
                if kind in {"report", "checkpoint_report"}:
                    try:
                        if kind == "report" and run["status"] == "cancelling":
                            continue
                        self._register_report(run_id, payload["report"], final=kind == "report")
                        if kind == "report" and getattr(runtime, "deadline", None):
                            runtime.deadline.pause()
                    except Exception as exc:
                        diagnostics = {"save_error": {"stage": "report_registration", "exception_type": type(exc).__name__}}
                        if isinstance(exc, AppError):
                            diagnostics["save_error"]["code"] = exc.code
                        elif isinstance(exc, OSError):
                            diagnostics["save_error"]["errno"] = exc.errno
                        saved = payload.get("report")
                        if isinstance(saved, dict):
                            failed = next(
                                (record for record in reversed(saved.get("code_executions", []))
                                 if isinstance(record, dict) and record.get("status") == "error"),
                                None,
                            )
                            if failed:
                                diagnostics["execution_error"] = {
                                    key: failed[key]
                                    for key in ("id", "error", "exit_code", "stderr", "container_state", "container_removed")
                                    if key in failed
                                }
                        self.fail(run_id, "save_failed", "成果保存或数据库登记失败；真实代码执行结果保留在执行记录中，本次不能标记为完成。",
                                  extra=scrub(diagnostics, self.secrets))
                elif kind == "worker_error":
                    if run["status"] != "cancelling":
                        runtime.cancel_signal.set()
                        runtime.stop(force=True)
                        cleaned = self.cleanup_owned(run_id, runtime)
                        extra = {"exception_type": payload.get("exception_type"), "cleanup_confirmed": cleaned}
                        extra.update({key: payload[key] for key in ("save_error", "execution_error") if key in payload})
                        self.fail(run_id, payload["code"], payload["message"], extra=extra)
                else:
                    if kind == "container_lifecycle":
                        runtime.known_containers[payload["name"]] = payload
                    self.emit(run_id, kind, payload)
                    if kind == "answer_delta":
                        with self.store.connect() as db:
                            request = "draft:" + payload["id"]
                            db.execute("INSERT INTO messages(id,conversation_id,run_id,role,content,state,created_at,request_id) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_id,request_id) DO UPDATE SET content=content || excluded.content",
                                       (uid(), run["conversation_id"], run_id, "assistant", payload["text"], "provisional", now(), request))

    def _emit_timing(self, runtime, *, force=False):
        """Persist a bounded one-second heartbeat even during silent model IO."""
        with self.lock:
            deadline = getattr(runtime, "deadline", None)
            if runtime.closed or self.closed or self.active != runtime.run_id or deadline is None:
                return
            current = deadline.clock()
            previous = getattr(runtime, "last_timing_at", None)
            if not force and previous is not None and current - previous < 1:
                return
            run = self.store.require("runs", runtime.run_id)
            if run["status"] not in {"running", "cancelling"} and not force:
                return
            runtime.last_timing_at = current
            self.emit(runtime.run_id, "timing", {"timing": deadline.snapshot()})

    def _timing_snapshot(self, run_id, *, final_status=None):
        runtime = self.runtime
        deadline = getattr(runtime, "deadline", None) if runtime and runtime.run_id == run_id else None
        if deadline is not None:
            if final_status == "awaiting_input":
                deadline.pause()
            elif final_status in TERMINAL:
                deadline.stop()
            return deadline.snapshot()
        with self.store.connect() as db:
            row = db.execute("SELECT payload_json FROM run_events WHERE run_id=? AND type IN ('timing','completed','error','cancelled','interrupted','awaiting_input') ORDER BY seq DESC", (run_id,))
            for event in row:
                timing = json.loads(event["payload_json"]).get("timing")
                if isinstance(timing, dict):
                    # A stopped service cannot measure its unobserved tail. Never
                    # replace monotonic evidence with hours of wall-clock downtime.
                    if final_status in TERMINAL:
                        timing = {**timing, "active": False, "current_models": [],
                                  "interrupted_unknown_tail": final_status == "interrupted"}
                    return timing
        return None

    def _deadline_exceeded(self, runtime):
        with self.lock:
            deadline = getattr(runtime, "deadline", None)
            if self.active != runtime.run_id or not deadline:
                return False
            failure = deadline.error()
            if not failure:
                return False
            run = self.store.require("runs", runtime.run_id)
            if run["status"] != "running":
                return False
            runtime.cancel_signal.set()
            runtime.stop(force=True)
            cleaned = self.cleanup_owned(runtime.run_id, runtime)
            self.fail(runtime.run_id, failure["code"],
                      "模型请求超过等待时限，已停止本次运行并保留已有成果。"
                      if failure["code"] == "model_request_timeout"
                      else "本次任务超过运行时限，已停止并保留已有成果；可拆分任务后重新执行。",
                      extra={"deadline": failure, "cleanup_confirmed": cleaned})
            return True

    def _register_report(self, run_id, report, *, final=True):
        run = self.store.require("runs", run_id)
        canonical = json.loads(encode(report))
        if run["core_session_id"] and (run["core_session_id"] != report["session_id"] or run["core_task_id"] != report["task_id"]):
            raise ValueError("Runtime identity mismatch")
        if report.get("status") not in {"completed", "awaiting_input", "error", "cancelled"}:
            raise ValueError("Unknown core state")
        if final and report["status"] in {"error", "cancelled"} and self.runtime:
            self.runtime.cancel_signal.set()
            self.runtime.stop(force=True)
            cleaned = self.cleanup_owned(run_id, self.runtime)
            report.setdefault("error", {"code": "cancelled", "message": "任务已停止。"})
            report["error"]["cleanup_confirmed"] = cleaned
            if not cleaned:
                report["status"] = "error"
                report["error"] = {"code": "cleanup_failed", "message": "工作进程已停止，容器清理未确认。", "cleanup_confirmed": False}
        paths = set(report.get("artifact_paths", []))
        for key in ("record_path", "brief_path"):
            if report.get(key):
                paths.add(report[key])
        new_files = {}
        for path in paths:
            relative = Path(path).relative_to(self.store.root).as_posix()
            target = managed_path(self.store.root, relative)
            if not relative.startswith("exports/"):
                raise ValueError("Export outside owned directory")
            raw = target.read_bytes()
            new_files[path] = dict(id=uid(), conversation_id=run["conversation_id"], run_id=run_id,
                                   name=target.name, relative_path=relative, media_type=MEDIA.get(target.suffix, "application/octet-stream"),
                                   size=len(raw), sha256=digest(raw), preview_json=None, parse_error_json=None, category="generated")
        if report.get("record_path") not in new_files:
            raise ValueError("Report file missing")
        from labweaver.runtime.artifacts import strict_json
        record_file = managed_path(self.store.root, new_files[report["record_path"]]["relative_path"])
        if strict_json(record_file.read_bytes()) != canonical:
            raise ValueError("Report file/IPC mismatch")
        for execution in report.get("code_executions", []):
            code_file = new_files.get(execution.get("code_path"))
            if not code_file or code_file["sha256"] != execution["code_sha256"] or digest(execution["code"].encode()) != execution["code_sha256"]:
                raise ValueError("Saved code hash mismatch")
        # A core completed report is accepted only with the bound immutable input and matching asset manifests.
        with self.store.connect() as db:
            input_row = db.execute("SELECT f.sha256 FROM files f JOIN conversation_files cf ON cf.file_id=f.id WHERE cf.conversation_id=? AND cf.role='csv'", (run["conversation_id"],)).fetchone()
            bound = [dict(f) for f in db.execute("SELECT f.*,cf.role FROM files f JOIN conversation_files cf ON cf.file_id=f.id WHERE cf.conversation_id=? ORDER BY cf.position", (run["conversation_id"],))]
        for file in bound:
            read_registered(self.store, file["id"])
        materials = [f for f in bound if f["role"] == "material"]
        if final and report.get("materials"):
            sources = report["materials"]
            if len(sources) != len(materials) or any(source.get("sha256") != file["sha256"] for source, file in zip(sources, materials)):
                raise ValueError("Material provenance mismatch")
        if report["status"] == "completed" and (report.get("source") or {}).get("sha256") != input_row["sha256"]:
            raise ValueError("Input provenance mismatch")
        pending = report.get("pending_input")
        if report["status"] == "awaiting_input" and (not isinstance(pending, dict) or not pending.get("id")):
            raise ValueError("Interrupt ID missing")
        with self.store.connect() as db:
            # Pending exports are retained as evidence, but assets are registered only once per task.
            for file in new_files.values():
                prior = db.execute("SELECT id FROM files WHERE relative_path=?", (file["relative_path"],)).fetchone()
                if prior:
                    file["id"] = prior["id"]
                else:
                    db.execute("INSERT INTO files VALUES (:id,:conversation_id,:run_id,:name,:relative_path,:media_type,:size,:sha256,:preview_json,:parse_error_json,:category)", file)
            for index, execution in enumerate(report.get("code_executions", []), 1):
                execution_id = execution["id"]
                code_file = new_files.get(execution.get("code_path"))
                db.execute("INSERT INTO executions VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET code_file_id=excluded.code_file_id,record_json=excluded.record_json", (execution_id, run_id, index, code_file["id"] if code_file else None, encode(execution)))
            for meta in report.get("artifacts", []):
                existing = db.execute("SELECT id FROM artifacts WHERE run_id=? AND core_id=?", (run_id, meta["id"])).fetchone()
                if existing:
                    continue
                id = uid()
                db.execute("INSERT INTO artifacts VALUES (?,?,?,?)", (id, run_id, meta["id"], encode(meta)))
                for name, path in meta.get("paths", {}).items():
                    file = new_files[path]
                    if meta["files"][name]["sha256"] != file["sha256"] or meta["files"][name]["size"] != file["size"]:
                        raise ValueError("Artifact hash mismatch")
                    db.execute("INSERT INTO artifact_files VALUES (?,?,?)", (id, file["id"], name))
            status = report["status"] if final else run["status"]
            db.execute("UPDATE runs SET status=?,updated_at=?,report_path=?,report_json=?,error_json=?,pending_json=?,core_session_id=?,core_task_id=? WHERE id=?",
                       (status, now(), new_files[report["record_path"]]["relative_path"], encode(report), encode(report.get("error")) if final else run["error_json"],
                        encode(pending) if final else run["pending_json"], report["session_id"], report["task_id"], run_id))
            if final:
                db.execute("UPDATE messages SET state='superseded' WHERE run_id=? AND state='provisional'", (run_id,))
            if final and pending:
                db.execute("INSERT OR IGNORE INTO messages(id,conversation_id,run_id,role,content,state,created_at,question_id,request_id) VALUES (?,?,?,?,?,?,?,?,?)",
                           (uid(), run["conversation_id"], run_id, "assistant", pending["question"], "question", now(), pending["id"], "question:" + pending["id"]))
            elif final and report.get("final_answer"):
                db.execute("INSERT INTO messages(id,conversation_id,run_id,role,content,state,created_at,request_id) VALUES (?,?,?,?,?,?,?,?)",
                           (uid(), run["conversation_id"], run_id, "assistant", report["final_answer"], "validated" if report["status"] == "completed" else report["status"], now(), "final:" + uid()))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now(), run["conversation_id"]))
            self.store.append_event(run_id, report["status"] if final else "artifacts_saved",
                                    {"status": status, "pending_input": pending if final else None,
                                     "error": report.get("error") if final else None, "final_answer": report.get("final_answer", ""),
                                     "timing": self._timing_snapshot(run_id, final_status=status if final else None)}, conn=db)
        if final and report["status"] in TERMINAL:
            self.active = None
            if report["status"] != "completed" and self.runtime:
                self.runtime.stop(force=True)
                self.runtime = None

    def fail(self, run_id, code, message, *, status="error", extra=None):
        run = self.store.require("runs", run_id)
        error = {"code": code, "message": message} | (extra or {})
        if self.runtime and self.runtime.run_id == run_id:
            self.runtime.cancel_signal.set()
            self.runtime.stop(force=True)
            if "cleanup_confirmed" not in error:
                error["cleanup_confirmed"] = self.cleanup_owned(run_id, self.runtime)
            self._recover_exports(run)
        timing = self._timing_snapshot(run_id, final_status=status)
        with self.store.connect() as db:
            db.execute("UPDATE runs SET status=?,error_json=?,pending_json=NULL,updated_at=? WHERE id=?", (status, encode(error), now(), run_id))
            self.store.append_event(run_id, status, {"status": status, "error": error, "timing": timing}, conn=db)
        if self.active == run_id:
            self.active = None
            if self.runtime:
                self.runtime.stop(force=True)
                self.runtime = None

    def cancel(self, run_id):
        with self.lock:
            run = self.store.require("runs", run_id)
            if run["status"] in TERMINAL or run["status"] == "cancelling":
                return self.run_view(run_id)
            if self.active != run_id or not self.runtime:
                raise AppError("runtime_expired", "运行时已失效。", 409)
            runtime = self.runtime
            with self.store.connect() as db:
                db.execute("UPDATE runs SET status='cancelling',updated_at=? WHERE id=?", (now(), run_id))
            self.emit(run_id, "cancelling", {"status": "cancelling"})
            runtime.cancel_signal.set()
            threading.Thread(target=self._cancel_finish, args=(runtime, run_id), daemon=True).start()
            return self.run_view(run_id)

    def _cancel_finish(self, runtime, run_id):
        runtime.process.join(timeout=self.settings.cancel_grace)
        runtime.stop(force=True)
        cleaned = self.cleanup_owned(run_id, runtime)
        with self.lock:
            if self.store.require("runs", run_id)["status"] not in TERMINAL:
                self._recover_exports(self.store.require("runs", run_id))
                self.fail(run_id, "cancelled" if cleaned else "cancel_cleanup_failed",
                          "任务已停止。" if cleaned else "未能确认工作进程、临时目录或容器清理，请查看执行记录并重试。",
                          status="cancelled" if cleaned else "error", extra={"cleanup_confirmed": cleaned})

    def cleanup_owned(self, run_id, runtime=None):
        """Inspect only exact app/run labels on the pinned local daemon; never remove arbitrary workloads."""
        from labweaver.runtime.execution import DockerExecutor, docker_executable, ExecutionError
        from labweaver.runtime.execution import ExecutionConfig
        if runtime is not None and not runtime.process.is_alive() and hasattr(runtime, "_cleanup_workspace"):
            runtime._cleanup_workspace()
        if runtime is not None and (runtime.process.is_alive() or not getattr(runtime, "workspace_cleaned", True)):
            return False
        executor = DockerExecutor(ExecutionConfig(image=self.settings.image))
        try:
            endpoint = next((p.get("endpoint") for p in (runtime.known_containers.values() if runtime else []) if p.get("endpoint")), None)
            if endpoint:
                if not endpoint.startswith(("unix://", "npipe://")):
                    return False
                executor._endpoint = endpoint
                docker = docker_executable()
            else:
                docker = docker_executable()
                with self.store.connect() as db:
                    prior = db.execute("SELECT payload_json FROM run_events WHERE run_id=? AND type='container_lifecycle' ORDER BY seq DESC LIMIT 1", (run_id,)).fetchone()
                endpoint = json.loads(prior["payload_json"]).get("endpoint") if prior else None
                if not endpoint:
                    selected_context = os.environ.get("DOCKER_CONTEXT")
                    endpoint = None if selected_context else os.environ.get("DOCKER_HOST")
                    if not endpoint:
                        context = subprocess.run([docker, "context", "inspect", *([selected_context] if selected_context else []),
                                                  "--format", "{{.Endpoints.docker.Host}}"], capture_output=True, timeout=10)
                        if context.returncode:
                            raise OSError
                        endpoint = context.stdout.decode().strip()
                if not endpoint.startswith(("unix://", "npipe://")):
                    return False
                executor._endpoint = endpoint
            command = executor._docker_cli(docker)
            filters = ["--filter", f"label=labweaver.app={self.store.app_id}", "--filter", f"label=labweaver.run={run_id}"]
            def listed():
                result = subprocess.run(command + ["ps", "-aq", *filters], capture_output=True, timeout=10, env=executor._client_env())
                if result.returncode:
                    raise OSError
                return result.stdout.decode().split()
            ids = listed()
            for id in ids:
                # Re-inspect labels before removal, avoiding a filter-only authority decision.
                probe = subprocess.run(command + ["inspect", id, "--format", "{{json .Config.Labels}}"], capture_output=True, timeout=10, env=executor._client_env())
                labels = json.loads(probe.stdout)
                if labels.get("labweaver.app") != self.store.app_id or labels.get("labweaver.run") != run_id:
                    return False
                if not executor._remove(id):
                    return False
            return not listed()
        except (ExecutionError, OSError, ValueError, TypeError, subprocess.TimeoutExpired):
            # If no execution has begun and no Docker container can exist, no cleanup is required.
            with self.store.connect() as db:
                began = db.execute("SELECT 1 FROM run_events WHERE run_id=? AND type IN ('container_lifecycle','execution_start') LIMIT 1", (run_id,)).fetchone()
            return began is None

    def _recover_exports(self, run):
        """Recover only known core task folders, never resume or claim historical completion."""
        from labweaver.runtime.artifacts import strict_json, validate_png, validate_svg, csv_table
        if not run["core_session_id"] or not run["core_task_id"]:
            return
        relative = f"exports/{run['core_session_id']}/{run['core_task_id']}"
        try:
            folder = managed_path(self.store.root, relative)
            if not folder.is_dir():
                return
            candidates = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)[-40:]
            for candidate in candidates:
                raw = managed_path(self.store.root, candidate.relative_to(self.store.root).as_posix()).read_bytes()
                if len(raw) > 16 * 1024 * 1024:
                    continue
                report = strict_json(raw)
                if not isinstance(report, dict) or report.get("report_version") not in {2,3} or report.get("session_id") != run["core_session_id"] or report.get("task_id") != run["core_task_id"]:
                    continue
                if contains_secret(report, self.secrets):
                    continue
                # Recheck orphan artifact content, not just its declared manifest.
                for meta in report.get("artifacts", []):
                    for name, path in meta.get("paths", {}).items():
                        bytes = managed_path(self.store.root, Path(path).relative_to(self.store.root).as_posix()).read_bytes()
                        if digest(bytes) != meta["files"][name]["sha256"]:
                            raise ValueError("Orphan hash mismatch")
                        if name.endswith(".png"):
                            validate_png(bytes)
                        elif name.endswith(".svg"):
                            validate_svg(bytes)
                        elif name.endswith(".csv"):
                            columns, rows = csv_table(bytes)
                            if columns != meta["columns"] or len(rows) != meta["row_count"]:
                                raise ValueError("Orphan table mismatch")
                        else:
                            strict_json(bytes)
                for execution in report.get("code_executions", []):
                    if digest(execution["code"].encode()) != execution["code_sha256"]:
                        raise ValueError("Orphan code mismatch")
                self._register_report(run["id"], report, final=False)
        except (OSError, ValueError, TypeError, KeyError, AppError):
            # Invalid remnants are never exposed as valid assets.
            return

    def run_view(self, run_id):
        run = self.store.require("runs", run_id)
        report = json.loads(run["report_json"] or "{}")
        active_assets = {a["id"] for a in report.get("artifacts", [])}
        current_paths = set(report.get("artifact_paths", [])) | {
            report[k] for k in ("record_path", "brief_path") if report.get(k)
        }
        with self.store.connect() as db:
            artifacts = []
            for item in db.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,)):
                if "artifacts" in report and item["core_id"] not in active_assets:
                    continue  # Superseded partial outputs remain in execution history only.
                meta = json.loads(item["metadata_json"])
                files = [file_view(dict(f)) | {"name": f["artifact_name"]} for f in db.execute("SELECT f.*,af.name AS artifact_name FROM files f JOIN artifact_files af ON af.file_id=f.id WHERE af.artifact_id=?", (item["id"],))]
                view = {k: meta[k] for k in ("kind", "label", "row_count", "columns", "value", "preview") if k in meta} | {"id": item["id"], "files": files}
                if meta.get("kind") == "metric":
                    from labweaver.runtime.presentation import metric_display_fields
                    view.pop("value", None)
                    view.update(metric_display_fields(meta.get("value")))
                artifacts.append(view)
            downloads = [file_view(dict(f)) for f in db.execute("SELECT * FROM files WHERE run_id=? AND category='generated'", (run_id,))
                         if not current_paths or str(managed_path(self.store.root, f["relative_path"])) in current_paths]
            model = db.execute("SELECT selection_json FROM run_models WHERE run_id=?", (run_id,)).fetchone()
        return {k: run[k] for k in ("id", "conversation_id", "task", "status", "created_at", "updated_at", "core_session_id", "core_task_id")} | {
            "error": json.loads(run["error_json"] or "null"), "pending_input": json.loads(run["pending_json"] or "null"),
            "final_answer": report.get("final_answer", ""), "artifacts": artifacts, "downloads": downloads,
            "task_plan": report.get("task_plan", []), "model_calls": report.get("model_calls", 0),
            "task_spec":report.get("task_spec"), "verification":report.get("verification",[]), "token_usage":report.get("token_usage"),
            "skills":report.get("skills"),
            "analysis_model_calls": report.get("analysis_model_calls", 0), "events": self.all_events(run_id),
            "history_run_ids": json.loads(run["context_ids"]), "model_selection": json.loads(model[0]) if model else None,
            "timing": self._timing_snapshot(run_id) or report.get("timing")}

    def all_events(self, run_id):
        rows, after = [], 0
        while True:
            page = self.store.events(run_id, after)
            rows.extend(page)
            if len(page) < 500:
                return rows
            after = page[-1]["seq"]

    def conversation_view(self, conversation_id):
        conversation = self.store.require("conversations", conversation_id)
        with self.store.connect() as db:
            files = [file_view(dict(r)) for r in db.execute("SELECT f.*,cf.role FROM files f JOIN conversation_files cf ON f.id=cf.file_id WHERE cf.conversation_id=? ORDER BY cf.position", (conversation_id,))]
            messages = [dict(r) for r in db.execute("SELECT id,run_id,role,content,state,question_id,created_at FROM messages WHERE conversation_id=? ORDER BY created_at", (conversation_id,))]
            run_ids = [r["id"] for r in db.execute("SELECT id FROM runs WHERE conversation_id=? ORDER BY created_at", (conversation_id,))]
            model = db.execute("SELECT selection_json FROM conversation_models WHERE conversation_id=?", (conversation_id,)).fetchone()
        return dict(conversation=conversation, files=files, messages=messages, runs=[self.run_view(id) for id in run_ids], model_selection=json.loads(model[0]) if model else None)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            runtime, active = self.runtime, self.active
            if runtime:
                runtime.cancel_signal.set()
                runtime.stop(force=True)
            if active:
                cleaned = self.cleanup_owned(active, runtime)
                self.fail(active, "interrupted", "服务已关闭；下次创建新运行时重新执行。", status="interrupted", extra={"cleanup_confirmed": cleaned})
            self.application_lock.close()
