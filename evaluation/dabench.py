"""Pinned InfiAgent-DABench subset, paired Skills comparison, real Docker only.

No benchmark answer, scorer or extra dataset is passed to the Agent sandbox.
The original English question/constraints/format are used without method hints.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / ".cache/evaluation/dabench-source"
DEFAULT_OUTPUT = ROOT / ".cache/evaluation/skills-dabench-v2"
SEED = 20261103
DATA_REVISION = "b455d578e30fee513abd79936cbdf7a6de026cb5"
SCORER_REVISION = "3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c"
SCORER_SHA256 = "8c4cbeed80c6c95aca65f72cea3213581629e81894a0fa5a496f22726b6e4501"
OPTIONS = dict(temperature=0, main_max_tokens=4096, analysis_max_tokens=12288,
               max_retries=0, execution_timeout=60, worker_timeout=900,
               formatter_max_tokens=2048, formatter_retries=0)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    temporary.replace(path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def original_task(question):
    """Transport fields only: no new task, hints, answer keys or method defaults."""
    return "\n\n".join(question[key] for key in ("question", "constraints", "format") if question.get(key))


def choose_questions(questions, eligible, *, seed=SEED):
    """Predeclared, answer-blind, dataset-disjoint difficulty-stratified selection."""
    used, selected = set(), {"development": [], "heldout": []}
    for split, quotas in [("development", [("hard", 1), ("medium", 2), ("easy", 1)]),
                          ("heldout", [("hard", 8), ("medium", 8), ("easy", 8)])]:
        for level, count in quotas:
            pool = sorted((q for q in questions if q["level"] == level and q["file_name"] in eligible),
                          key=lambda q: digest(f"{seed}:{split}:{q['id']}".encode()))
            choices = []
            for q in pool:
                if q["file_name"] in used:
                    continue
                used.add(q["file_name"])
                choices.append(dict(id=f"DA{q['id']}", official_id=q["id"], dataset=q["file_name"],
                                    level=q["level"], concepts=q["concepts"], task=original_task(q)))
                if len(choices) == count:
                    break
            if len(choices) != count:
                raise ValueError(f"Not enough distinct compatible CSVs for {split}/{level}")
            selected[split].extend(choices)
    return selected


def prepare(data, output):
    """Freeze selection before any model run; reject changed data on resume."""
    data, output = Path(data).resolve(), Path(output).resolve()
    if (output / "suite.json").exists():
        manifest = read_json(output / "inputs-manifest.json")
        for name, sha in manifest.items():
            if digest((output / name).read_bytes()) != sha:
                raise ValueError("Benchmark input snapshot changed")
        return output
    sys.path.insert(0, str(ROOT / "src"))
    from labweaver.tools.csv_profile import CsvReadError, load_csv_snapshot
    metadata = read_json(data / "source.json")
    if metadata["revision"] != DATA_REVISION:
        raise ValueError("Unexpected upstream dataset revision")
    for entry in metadata["files"]:
        if digest((data / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError("Upstream dataset content does not match manifest")
    questions = jsonl(data / "da-dev-questions.jsonl")
    eligible, excluded = {}, {}
    for filename in sorted({q["file_name"] for q in questions}):
        if Path(filename).name != filename or not filename.lower().endswith(".csv"):
            excluded[filename] = "unsupported_filename"
            continue
        path = data / "da-dev-tables" / filename
        try:
            snapshot = load_csv_snapshot(path)
            eligible[filename] = dict(sha256=digest(path.read_bytes()), rows=len(snapshot.rows),
                                      columns=len(snapshot.headers), size=path.stat().st_size)
        except CsvReadError as exc:
            excluded[filename] = exc.code
    selection = choose_questions(questions, eligible)
    suite = dict(version=1, seed=SEED, dataset="infiagent/DABench", revision=DATA_REVISION,
                 source_questions=len(questions), source_datasets=len(eligible) + len(excluded),
                 eligible=eligible, excluded=excluded, repetitions=3,
                 selection_policy="SHA256(seed:split:question_id), quotas 4 dev and 8 per level heldout; one question per CSV; dev allocated first",
                 **selection)
    suite["questions"] = {f"DA{q['id']}": q for q in questions if any(c["official_id"] == q["id"] for c in selection["development"] + selection["heldout"])}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "suite.json", suite)
    for q in selection["development"] + selection["heldout"]:
        target = output / "inputs" / q["dataset"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data / "da-dev-tables" / q["dataset"], target)
    # Oracle is host-only; workers send only original task and selected CSV snapshot.
    oracle = output / "oracle"
    oracle.mkdir(exist_ok=True)
    shutil.copyfile(data / "da-dev-labels.jsonl", oracle / "labels.jsonl")
    scorer_raw = (data / "upstream-eval_closed_form.py").read_bytes()
    if digest(scorer_raw) != SCORER_SHA256:
        raise ValueError("Unexpected upstream scorer content")
    (oracle / "eval_closed_form.py").write_bytes(scorer_raw)
    shutil.copyfile(data / "source.json", output / "source.json")
    paths = [output / "suite.json", output / "source.json", *sorted((output / "inputs").glob("*.csv")),
             oracle / "labels.jsonl", oracle / "eval_closed_form.py"]
    write_json(output / "inputs-manifest.json", {p.relative_to(output).as_posix(): digest(p.read_bytes()) for p in paths})
    return output


def freeze_contract(output, source, model, image):
    from run import freeze_source
    frozen = freeze_source(source, output)
    files = [Path(__file__), Path(__file__).with_name("run.py"), Path(__file__).with_name("skills_pilot.py"),
             Path(__file__).with_name("answer_format.py"), Path(__file__).with_name("measurements.py"),
             Path(__file__).with_name("benchmark_statistics.py")]
    contract = dict(version=1, upstream_dataset_revision=DATA_REVISION, upstream_scorer_revision=SCORER_REVISION,
                    implementation_sha256=digest(json.dumps(read_json(output / "implementation/manifest.json"), sort_keys=True).encode()),
                    inputs_sha256=digest((output / "inputs-manifest.json").read_bytes()),
                    evaluator_files={p.name: digest(p.read_bytes()) for p in files}, model=model, image=image,
                    options=OPTIONS, arms=["off", "on"],
                    scoring="Official comparator; raw and blind model-formatted final answers separate; deterministic metric-key extraction supplemental; formatter uses same pinned provider model without tools or data",
                    context="Fresh process/session per trial; original English task; no user-reply simulation; awaiting_input is a non-completion")
    target = output / "contract.json"
    if target.exists() and read_json(target) != contract:
        raise ValueError("Benchmark contract changed; create a new batch")
    write_json(target, contract)
    return frozen, contract


def verify_contract(output, source):
    contract = read_json(output / "contract.json")
    for name, sha in contract["evaluator_files"].items():
        if digest(Path(__file__).with_name(name).read_bytes()) != sha:
            raise ValueError("Evaluator changed during batch")
    if digest((output / "inputs-manifest.json").read_bytes()) != contract["inputs_sha256"]:
        raise ValueError("Benchmark manifest changed")
    for name, sha in read_json(output / "inputs-manifest.json").items():
        if digest((output / name).read_bytes()) != sha:
            raise ValueError("Benchmark input changed")
    manifest = read_json(output / "implementation/manifest.json")
    if digest(json.dumps(manifest, sort_keys=True).encode()) != contract["implementation_sha256"]:
        raise ValueError("Implementation manifest changed")
    for name, sha in manifest.items():
        if digest((Path(source) / name).read_bytes()) != sha:
            raise ValueError("Implementation snapshot changed")
    return contract


def load_scorer(output):
    """Only author-published scorer code, pinned and hashed, never generated code."""
    path = Path(output) / "oracle/eval_closed_form.py"
    raw = path.read_bytes()
    if digest(raw) != SCORER_SHA256:
        raise ValueError("Unexpected upstream scorer content")
    names = {"extract_format", "is_equal", "evaluate_responses", "evaluate_accuracy_by_question",
             "evaluate_accuracy_by_sub_question", "evaluate_accuracy_proportional_by_sub_question_adjusted"}
    parsed = ast.parse(raw, filename=str(path))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Official scorer functions missing")
    module = types.ModuleType("pinned_infiagent_scorer")
    module.re = re
    # Execute the exact published pure functions, omitting its CLI and unrelated imports.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def final_text(report):
    """Last main-assistant answer only; never select a better earlier answer."""
    for message in reversed(report.get("messages", [])):
        if message.get("type") == "ai":
            content = message.get("content", "")
            if not isinstance(content, str):
                return ""
            try:
                value = json.loads(content)
                if isinstance(value, dict) and isinstance(value.get("answer"), str):
                    return value["answer"]
            except (ValueError, TypeError):
                pass
            return content
    return report.get("answer_text") or report.get("final_answer") or ""


def answer_keys(task_format):
    return re.findall(r"@([\w]+)\[", task_format)


def normalized_response(text, assets, keys, scorer):
    """Extract existing values only, without label access or recomputation.

    Final @key[value] always wins, including wrong values. Missing keys may be
    taken from scalar metric keys. Conflicting artifact values stay unanswered.
    Values are not rounded, lowercased, unit-converted or numerically repaired.
    """
    names, values = scorer.extract_format(text)
    observed = dict(zip(names, values))
    provenance = {key: "final_answer" for key in observed}
    candidates = defaultdict(list)
    for artifact_id, asset in assets.items():
        metadata = asset.get("metadata", {})
        if metadata.get("kind") != "metric":
            continue
        def visit(value):
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                if key in keys and isinstance(item, (str, int, float, bool)) and not isinstance(item, bool):
                    if not isinstance(item, float) or math.isfinite(item):
                        candidates[key].append((str(item), artifact_id))
                if isinstance(item, dict):
                    visit(item)
        visit(metadata.get("value"))
    for key in keys:
        if key in observed:
            continue
        matches = candidates.get(key, [])
        if matches and len({value for value, _ in matches}) == 1:
            observed[key] = matches[0][0]
            provenance[key] = "metric:" + matches[0][1]
    return "\n".join(f"@{key}[{observed[key]}]" for key in keys if key in observed) or "__NO_ANSWER__", provenance


def score_one(scorer, label, response):
    """Retain every expected key in the denominator, even an empty response."""
    # The official scorer skips missing/empty responses; use a nonempty sentinel.
    rows = scorer.evaluate_responses([label], [{"id": label["id"], "response": response or "__NO_ANSWER__"}])
    if len(rows) != 1:
        raise ValueError("Official scorer did not return one result")
    return rows[0]


def score_flags(score):
    flags = list(score["correctness"].values())
    return bool(flags) and all(flags), sum(flags), len(flags)


def verified_assets(report, assets):
    """Only host-registered outputs of successful, checked executions qualify."""
    attempts = {e["id"]: e for e in report.get("code_executions", [])}
    result = {}
    for aid, asset in assets.items():
        metadata = asset.get("metadata", {})
        execution = attempts.get(metadata.get("execution_id"), {})
        if (execution.get("status") == "completed"
                and execution.get("verification", {}).get("passed", True)
                and aid in execution.get("artifact_ids", [])):
            result[aid] = asset
    return result


def schedule(cases, repetitions):
    for repeat in range(1, repetitions + 1):
        for index, case in enumerate(cases):
            for arm in (["off", "on"] if (repeat + index) % 2 else ["on", "off"]):
                yield case, repeat, arm


def owner_id(output, case, repeat, arm):
    return f"{case}-{repeat}-{arm}-{digest(str(Path(output).resolve()).encode())[:16]}"


def cleanup_owned(output, case, repeat, arm, image):
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig, docker_executable
    executor = DockerExecutor(ExecutionConfig(image=image))
    executor.preflight()
    command = executor._docker_cli(docker_executable())
    response = subprocess.run(command + ["ps", "-aq", "--filter", "label=labweaver.app=skills-dabench",
                                        "--filter", f"label=labweaver.run={owner_id(output, case, repeat, arm)}"],
                              capture_output=True, timeout=15, env=executor._client_env(), check=True)
    for container in response.stdout.decode().split():
        subprocess.run(command + ["rm", "-f", container], capture_output=True, timeout=15,
                       check=True, env=executor._client_env())


def run_worker(args):
    output, source = Path(args.output).resolve(), Path(args.source).resolve()
    contract = verify_contract(output, source)
    sys.path.insert(0, str(source))
    os.chdir(ROOT)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from labweaver.agent import create_session
    from labweaver.config import create_model, load_config
    from labweaver.run_config import load_intake_config
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig
    from measurements import measurements
    suite = read_json(output / "suite.json")
    case = next(c for c in suite[args.split] if c["id"] == args.case)
    destination = output / args.split / "trials" / f"{args.case}-{args.repeat}-{args.arm}"
    config = load_config(load_intake_config().env_file)
    if dict(model_id=config.model_id, base_url=config.base_url, timeout=config.timeout) != contract["model"]:
        raise ValueError("Model settings changed")
    execution_config = ExecutionConfig(image=args.image, timeout=OPTIONS["execution_timeout"])
    probe = DockerExecutor(execution_config)
    probe.preflight()
    if probe.image_id != contract["image"]["id"]:
        raise ValueError("Sandbox image changed")
    path = output / "inputs" / case["dataset"]
    summary = dict(case=case["id"], dataset=case["dataset"], level=case["level"], concepts=case["concepts"],
                   repeat=args.repeat, arm=args.arm, contract_sha256=digest((output / "contract.json").read_bytes()),
                   source_sha256=digest(path.read_bytes()), split=args.split,
                   answer_correct=False, raw_answer_correct=False, delivery_completed=False, end_to_end=False,
                   correct_subanswers=0, raw_correct_subanswers=0, skill_loads=[])
    session = create_session(path, create_model(config, max_tokens=OPTIONS["main_max_tokens"]),
                             analysis_model=create_model(config, max_tokens=OPTIONS["analysis_max_tokens"]),
                             skills_enabled=args.arm == "on", execution_config=execution_config)
    session.executor.set_owner("skills-dabench", owner_id(output / args.split, args.case, args.repeat, args.arm))
    start = time.monotonic()
    def progress(event):
        public = {key: event[key] for key in ["type", "role", "call", "attempt", "status", "elapsed_seconds", "execution_id"] if key in event}
        with (destination / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(public, ensure_ascii=False) + "\n")
    try:
        report = session.invoke(case["task"], on_event=progress)
        agent_elapsed = round(time.monotonic() - start, 3)
        text = final_text(report)
        scorer = load_scorer(output)
        # Keys originate only in the public question format, not hidden labels.
        keys = answer_keys(case["task"])
        normalized, provenance = normalized_response(text, verified_assets(report, session.artifact_assets), keys, scorer)
        from answer_format import format_answer
        format_start = time.monotonic()
        try:
            formatted = format_answer(create_model(config, max_tokens=OPTIONS["formatter_max_tokens"]),
                                      question=suite["questions"][case["id"]], answer=text)
        except Exception as exc:
            formatted = dict(response="__NO_ANSWER__", called=True, usage=None, error=type(exc).__name__)
        formatted["elapsed_seconds"] = round(time.monotonic() - format_start, 3)
        label = next(label for label in jsonl(output / "oracle/labels.jsonl") if label["id"] == case["official_id"])
        raw_score = score_one(scorer, label, text)
        deterministic_score = score_one(scorer, label, normalized)
        normalized_score = score_one(scorer, label, formatted["response"])
        raw_pass, raw_count, count = score_flags(raw_score)
        passed, passed_count, count = score_flags(normalized_score)
        write_json(destination / "scoring.json", dict(raw_text=text, normalized_text=normalized,
                                                     provenance=provenance, raw=raw_score,
                                                     deterministic=deterministic_score, normalized=normalized_score,
                                                     formatter=formatted))
        saved = session.save(destination)
        unchanged = digest(path.read_bytes()) == summary["source_sha256"]
        delivery = report["status"] == "completed" and not report.get("missing_deliverables") and unchanged
        summary.update(status=report["status"], error=report.get("error"), source_unchanged=unchanged,
                       record_path=saved["record_path"], answer_correct=passed, raw_answer_correct=raw_pass,
                       delivery_completed=delivery, end_to_end=passed and delivery,
                       correct_subanswers=passed_count, raw_correct_subanswers=raw_count, total_subanswers=count,
                       raw_score=raw_score, normalized_score=normalized_score,
                       deterministic_score=deterministic_score, formatter=formatted,
                       agent_elapsed_seconds=agent_elapsed,
                       model_calls=report.get("model_calls", 0), analysis_model_calls=report.get("analysis_model_calls", 0),
                       skill_loads=report.get("skills", {}).get("loads", []),
                       missing_deliverables=report.get("missing_deliverables", []),
                       **measurements(report))
    except Exception as exc:
        summary.update(status="error", errors=[type(exc).__name__])
    finally:
        session.executor.cancel()
        summary["total_elapsed_seconds"] = round(time.monotonic() - start, 3)
        summary["elapsed_seconds"] = summary.get("agent_elapsed_seconds", summary["total_elapsed_seconds"])
        write_json(destination / "summary.json", summary)


def run_batch(args):
    output = prepare(args.data, args.output)
    sys.path.insert(0, str(Path(args.source).resolve()))
    from labweaver.config import load_config
    from labweaver.run_config import load_intake_config
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig
    from skills_pilot import claim_trial
    config = load_config(load_intake_config().env_file)
    probe = DockerExecutor(ExecutionConfig(image=args.image))
    probe.preflight()
    model = dict(model_id=config.model_id, base_url=config.base_url, timeout=config.timeout)
    frozen, _ = freeze_contract(output, args.source, model, dict(tag=args.image, id=probe.image_id))
    verify_contract(output, frozen)
    cases = read_json(output / "suite.json")[args.split]
    contract_sha = digest((output / "contract.json").read_bytes())
    repetitions = 1 if args.split == "development" else 3
    for case, repeat, arm in schedule(cases, repetitions):
        if (output / "STOP").exists():
            print("Stopped between trials; failures retained.", flush=True)
            return
        destination = output / args.split / "trials" / f"{case['id']}-{repeat}-{arm}"
        if not claim_trial(destination, case["id"], repeat, arm, contract_sha):
            continue
        print(f"{args.split} {case['id']} {case['level']} repeat={repeat} skills={arm}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output),
                   "--source", str(frozen), "--split", args.split, "--case", case["id"],
                   "--repeat", str(repeat), "--arm", arm, "--image", args.image]
        with (destination / "worker.log").open("w", encoding="utf-8") as log:
            try:
                subprocess.run(command, stdout=log, stderr=log, timeout=OPTIONS["worker_timeout"], check=False)
            except subprocess.TimeoutExpired:
                write_json(destination / "summary.json", dict(case=case["id"], dataset=case["dataset"],
                           repeat=repeat, arm=arm, contract_sha256=contract_sha, status="error",
                           answer_correct=False, raw_answer_correct=False, delivery_completed=False,
                           end_to_end=False, errors=["worker_timeout"]))
        if not (destination / "summary.json").exists():
            write_json(destination / "summary.json", dict(case=case["id"], dataset=case["dataset"],
                       repeat=repeat, arm=arm, contract_sha256=contract_sha, status="error",
                       answer_correct=False, raw_answer_correct=False, delivery_completed=False,
                       end_to_end=False, errors=["worker_failure"]))
        cleanup_owned(output / args.split, case["id"], repeat, arm, args.image)
        result = read_json(destination / "summary.json")
        print(json.dumps({key: result.get(key) for key in ["case", "repeat", "arm", "answer_correct", "delivery_completed", "end_to_end", "elapsed_seconds", "error", "errors"]}), flush=True)


def aggregate(output, split):
    from benchmark_statistics import summarize_trials
    output = Path(output)
    suite = read_json(output / "suite.json")
    contract_sha = digest((output / "contract.json").read_bytes())
    rows = []
    for path in sorted((output / split / "trials").glob("*/summary.json")):
        row = read_json(path)
        if row.get("contract_sha256") != contract_sha:
            raise ValueError("Mixed evaluation contracts")
        rows.append(row)
    result = summarize_trials(rows, planned_cases=suite[split], repetitions=1 if split == "development" else 3)
    result.update(version=1, split=split, contract_sha256=contract_sha,
                  dataset_revision=DATA_REVISION, scorer_revision=SCORER_REVISION,
                  complete=all(a["missing"] == 0 for a in result["arms"].values()))
    for arm in ["off", "on"]:
        trials = [r for r in rows if r["arm"] == arm]
        metrics = result["arms"][arm]
        metrics.update(raw_answer_correct=sum(r.get("raw_answer_correct", False) for r in trials),
                       correct_subanswers=sum(r.get("correct_subanswers", 0) for r in trials),
                       raw_correct_subanswers=sum(r.get("raw_correct_subanswers", 0) for r in trials),
                       no_internal_repair_attained=sum(r["end_to_end"] and r.get("no_internal_repair", False) for r in trials),
                       code_attempts=sum(r.get("code_attempts", 0) for r in trials))
        # Evaluate all PLANNED questions, not just workers which reached scoring.
        scorer = load_scorer(output)
        labels = {label["id"]: label for label in jsonl(output / "oracle/labels.jsonl")}
        lookup = {(r["case"], r["repeat"]): r for r in trials}
        raw_results, normalized_results = [], []
        for case in suite[split]:
            label = labels[case["official_id"]]
            for repeat in range(1, 2 if split == "development" else 4):
                row = lookup.get((case["id"], repeat), {})
                raw_results.append(row.get("raw_score") or score_one(scorer, label, ""))
                normalized_results.append(row.get("normalized_score") or score_one(scorer, label, ""))
        metrics["total_subanswers"] = sum(len(row["correctness"]) for row in normalized_results)
        def official_scores(results):
            return dict(ABQ=scorer.evaluate_accuracy_by_question(results),
                        PASQ=scorer.evaluate_accuracy_proportional_by_sub_question_adjusted(results),
                        UASQ=scorer.evaluate_accuracy_by_sub_question(results))
        metrics["official_raw"] = official_scores(raw_results)
        metrics["official_blind_reformat"] = official_scores(normalized_results)
        metrics["formatter_known_total_tokens"] = sum((r.get("formatter", {}).get("usage") or {}).get("total_tokens", 0) for r in trials)
        metrics["formatter_unknown_usage_calls"] = sum(r.get("formatter", {}).get("called", False) and not r.get("formatter", {}).get("usage") for r in trials)
        metrics["formatter_calls"] = sum(r.get("formatter", {}).get("called", False) for r in trials)
    result["by_case"] = {c["id"]: {"dataset": c["dataset"], "level": c["level"], "concepts": c["concepts"],
                           **{arm: {key: sum(bool(r.get(key)) for r in rows if r["case"] == c["id"] and r["arm"] == arm)
                                    for key in ["answer_correct", "raw_answer_correct", "delivery_completed", "end_to_end"]}
                              for arm in ["off", "on"]}} for c in suite[split]}
    write_json(output / split / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--report", action="store_true")
    action.add_argument("--worker", action="store_true")
    parser.add_argument("--data", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=ROOT / "src")
    parser.add_argument("--image", default="labweaver-python:0.3.0")
    parser.add_argument("--split", choices=["development", "heldout"], default="development")
    parser.add_argument("--case")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--arm", choices=["off", "on"])
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.data, args.output))
    elif args.run:
        run_batch(args)
    elif args.report:
        print(json.dumps(aggregate(args.output, args.split), ensure_ascii=False, indent=2))
    else:
        if args.case is None or args.repeat not in [1, 2, 3] or args.arm is None:
            parser.error("Worker requires case, repeat 1..3 and arm")
        run_worker(args)


if __name__ == "__main__":
    main()
