"""Official-benchmark transport and accounting tests; no paid calls or Docker.

The tiny scorer below is a test double with the author's public return schema.
It checks the adapter, not data analysis performance or the upstream benchmark.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCORER_DOUBLE = '''
from utils.utils import read_jsonl
raise AssertionError("The CLI and imports must not execute")

def extract_format(text):
    matches = re.findall(r"@(\\w+)\\[(.*?)\\]", text)
    return [m[0] for m in matches], [m[1] for m in matches]

def is_equal(response, label):
    if response == label:
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except (ValueError, TypeError):
        return False

def evaluate_responses(labels, responses):
    output = []
    for label in labels:
        response = next((r for r in responses if r.get("id") == label["id"] and r.get("response")), None)
        if response:
            names, values = extract_format(response["response"])
            predicted = dict(zip(names, values))
            output.append({"id": label["id"], "predicted_answers": predicted,
                "correctness": {key: is_equal(predicted.get(key), value) for key, value in label["common_answers"]}})
    return output

def evaluate_accuracy_by_question(results):
    return sum(all(r["correctness"].values()) for r in results) / len(results) if results else 0

def evaluate_accuracy_by_sub_question(results):
    total = sum(len(r["correctness"]) for r in results)
    return sum(sum(r["correctness"].values()) for r in results) / total if total else 0

def evaluate_accuracy_proportional_by_sub_question_adjusted(results):
    return sum(sum(r["correctness"].values()) / len(r["correctness"]) for r in results) / len(results) if results else 0
'''


@pytest.fixture
def benchmark(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "evaluation"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("dabench_evaluation_test", directory / "dabench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scorer(benchmark, tmp_path, monkeypatch):
    path = tmp_path / "oracle/eval_closed_form.py"
    path.parent.mkdir(parents=True)
    path.write_text(SCORER_DOUBLE, encoding="utf-8")
    monkeypatch.setattr(benchmark, "SCORER_SHA256", benchmark.digest(path.read_bytes()))
    return benchmark.load_scorer(tmp_path)


def question(index, level="easy", dataset=None):
    return {"id": index, "level": level, "file_name": dataset or f"data-{index}.csv",
            "question": "Compute the requested statistic.", "concepts": ["Summary Statistics"],
            "constraints": "Keep original records.", "format": "@result[value]"}


def questions():
    return [question(index + offset, level) for level, offset in [("easy", 0), ("medium", 20), ("hard", 40)]
            for index in range(12)]


def metric(value):
    return {"metadata": {"kind": "metric", "value": value}}


def test_transport_keeps_original_task_and_never_adds_oracle_answers(benchmark):
    source = question(1)
    source["common_answers"] = [["result", "SECRET_ORACLE_VALUE"]]
    task = benchmark.original_task(source)
    assert task == "Compute the requested statistic.\n\nKeep original records.\n\n@result[value]"
    assert "SECRET_ORACLE_VALUE" not in task
    assert benchmark.answer_keys(task) == ["result"]


def test_selection_is_answer_blind_deterministic_and_dataset_disjoint(benchmark):
    inputs = questions()
    eligible = {q["file_name"]: {} for q in inputs}
    chosen = benchmark.choose_questions(inputs, eligible)
    changed_labels = [{**q, "common_answers": [["result", "different"]]} for q in reversed(inputs)]
    assert benchmark.choose_questions(changed_labels, eligible) == chosen
    assert len(chosen["development"]) == 4
    assert len(chosen["heldout"]) == 24
    datasets = [q["dataset"] for split in chosen.values() for q in split]
    assert len(datasets) == len(set(datasets)) == 28
    assert {level: sum(q["level"] == level for q in chosen["heldout"])
            for level in ["easy", "medium", "hard"]} == {"easy": 8, "medium": 8, "hard": 8}


def test_selection_rejects_insufficient_compatible_datasets(benchmark):
    with pytest.raises(ValueError, match="Not enough"):
        benchmark.choose_questions(questions(), {})


def test_schedule_has_fresh_paired_identities_and_balanced_order(benchmark):
    cases = [{"id": "DA1"}, {"id": "DA2"}]
    scheduled = list(benchmark.schedule(cases, 3))
    identities = [(case["id"], repeat, arm) for case, repeat, arm in scheduled]
    assert len(identities) == len(set(identities)) == 12
    for case in cases:
        arms = [arm for row, _, arm in scheduled if row == case]
        assert arms.count("off") == arms.count("on") == 3
        assert arms[:2] != arms[2:4]


def test_last_answer_is_used_even_if_an_earlier_answer_was_better(benchmark):
    report = {"status": "error", "answer_text": "", "messages": [
        {"type": "ai", "content": '@result[7]'},
        {"type": "tool", "content": '@result[7]'},
        {"type": "ai", "content": json.dumps({"answer": "@result[8]"})}]}
    assert benchmark.final_text(report) == "@result[8]"
    report["messages"][-1]["content"] = "malformed @result[9]"
    assert benchmark.final_text(report) == "malformed @result[9]"


def test_scorer_loader_omits_cli_imports_and_checks_fixed_bytes(benchmark, scorer, tmp_path):
    assert scorer.is_equal("7.0000001", "7")
    assert not scorer.is_equal("7.00001", "7")
    assert not scorer.is_equal("7%", "7")
    assert scorer.evaluate_accuracy_by_question([{"correctness": {"result": False}}]) == 0
    with (tmp_path / "oracle/eval_closed_form.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# changed after pinning\n")
    with pytest.raises(ValueError, match="scorer content"):
        benchmark.load_scorer(tmp_path)


def test_empty_response_retains_all_official_subquestions_in_denominator(benchmark, scorer):
    label = {"id": 1, "common_answers": [["result", "7"], ["other", "8"]]}
    scored = benchmark.score_one(scorer, label, "")
    assert scored["correctness"] == {"result": False, "other": False}
    assert benchmark.score_flags(scored) == (False, 0, 2)
    assert benchmark.score_flags(benchmark.score_one(scorer, label, "@result[7]")) == (False, 1, 2)


def test_unlabelled_format_keys_do_not_become_extra_required_answers(benchmark, scorer):
    label = {"id": 1, "common_answers": [["result", "7"]]}
    scored = benchmark.score_one(scorer, label, "@result[7]")
    assert benchmark.score_flags(scored) == (True, 1, 1)


def test_normalization_never_replaces_a_wrong_final_answer(benchmark, scorer):
    assets = {"receipt": metric({"result": 7})}
    normalized, provenance = benchmark.normalized_response("@result[999]", assets, ["result"], scorer)
    assert normalized == "@result[999]"
    assert provenance == {"result": "final_answer"}


def test_normalization_reads_exact_existing_scalar_keys_without_recalculation(benchmark, scorer):
    assets = {"receipt": metric({"nested": {"result": 0.25}, "not_requested": 100})}
    normalized, provenance = benchmark.normalized_response("No machine-formatted answer", assets, ["result"], scorer)
    assert normalized == "@result[0.25]"
    assert provenance == {"result": "metric:receipt"}
    assert "25%" not in normalized


def test_conflicting_metric_candidates_are_not_cherry_picked(benchmark, scorer):
    assets = {"one": metric({"result": 7}), "two": metric({"result": 8})}
    assert benchmark.normalized_response("", assets, ["result"], scorer) == ("__NO_ANSWER__", {})


def test_artifact_fallback_rejects_failed_or_unregistered_executions(benchmark):
    assets = {name: {"metadata": {"execution_id": name}} for name in ["good", "failed", "unchecked", "foreign"]}
    report = {"code_executions": [
        {"id": "good", "status": "completed", "artifact_ids": ["good"], "verification": {"passed": True}},
        {"id": "failed", "status": "error", "artifact_ids": ["failed"]},
        {"id": "unchecked", "status": "completed", "artifact_ids": ["unchecked"], "verification": {"passed": False}},
        {"id": "foreign", "status": "completed", "artifact_ids": ["another-artifact"]}]}
    assert benchmark.verified_assets(report, assets) == {"good": assets["good"]}


@pytest.mark.parametrize("value", [True, None, [1, 2], float("nan"), float("inf")])
def test_normalization_rejects_non_scalar_or_nonfinite_candidates(benchmark, scorer, value):
    assert benchmark.normalized_response("", {"one": metric({"result": value})}, ["result"], scorer) == ("__NO_ANSWER__", {})


def source_fixture(benchmark, directory, monkeypatch):
    directory.mkdir()
    entries, all_questions = [], questions()
    for q in all_questions:
        path = directory / "da-dev-tables" / q["file_name"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"label,value\nfirst,1\nsecond,2\n")
        entries.append({"path": path.relative_to(directory).as_posix(), "sha256": benchmark.digest(path.read_bytes())})
    for name, contents in [("da-dev-questions.jsonl", all_questions),
                           ("da-dev-labels.jsonl", [{"id": q["id"], "common_answers": [["result", "3"]]} for q in all_questions])]:
        path = directory / name
        path.write_text("".join(json.dumps(row) + "\n" for row in contents), encoding="utf-8")
        entries.append({"path": name, "sha256": benchmark.digest(path.read_bytes())})
    raw = SCORER_DOUBLE.encode("utf-8")
    (directory / "upstream-eval_closed_form.py").write_bytes(raw)
    monkeypatch.setattr(benchmark, "SCORER_SHA256", benchmark.digest(raw))
    benchmark.write_json(directory / "source.json", {"revision": benchmark.DATA_REVISION, "files": entries})
    return directory


def test_prepare_pins_dataset_split_inputs_and_host_only_oracle(benchmark, tmp_path, monkeypatch):
    source = source_fixture(benchmark, tmp_path / "source", monkeypatch)
    output = benchmark.prepare(source, tmp_path / "batch")
    suite = benchmark.read_json(output / "suite.json")
    assert len(list((output / "inputs").glob("*.csv"))) == 28
    assert (output / "oracle/labels.jsonl").is_file()
    assert not any("common_answers" in q for q in suite["heldout"])
    assert benchmark.prepare(source, output) == output
    path = next((output / "inputs").glob("*.csv"))
    path.write_bytes(path.read_bytes() + b"third,3\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        benchmark.prepare(source, output)


def test_prepare_rejects_unpinned_source_or_scorer(benchmark, tmp_path, monkeypatch):
    source = source_fixture(benchmark, tmp_path / "source", monkeypatch)
    scorer = source / "upstream-eval_closed_form.py"
    scorer.write_text("raise RuntimeError('unreviewed')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scorer content"):
        benchmark.prepare(source, tmp_path / "batch")


def test_contract_rejects_evaluator_inputs_and_implementation_tampering(benchmark, tmp_path, monkeypatch):
    source = source_fixture(benchmark, tmp_path / "data", monkeypatch)
    output = benchmark.prepare(source, tmp_path / "batch")
    package = tmp_path / "src/labweaver"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# frozen test package\n", encoding="utf-8")
    model = {"model_id": "test", "base_url": "https://example.invalid", "timeout": 30}
    image = {"tag": "test-image", "id": "sha256:test-only"}
    frozen, contract = benchmark.freeze_contract(output, package.parent, model, image)
    assert benchmark.verify_contract(output, frozen) == contract
    with pytest.raises(ValueError, match="contract changed"):
        benchmark.freeze_contract(output, package.parent, {**model, "timeout": 60}, image)
    implementation = frozen / "labweaver/__init__.py"
    original = implementation.read_bytes()
    implementation.write_bytes(b"# different implementation\n")
    with pytest.raises(ValueError, match="Implementation snapshot"):
        benchmark.verify_contract(output, frozen)
    implementation.write_bytes(original)
    manifest = output / "inputs-manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest changed"):
        benchmark.verify_contract(output, frozen)


def test_aggregation_counts_every_planned_subanswer_even_without_worker_records(benchmark, scorer, tmp_path):
    cases = [{"id": "DA1", "official_id": 1, "dataset": "one.csv", "level": "easy", "concepts": []},
             {"id": "DA2", "official_id": 2, "dataset": "two.csv", "level": "medium", "concepts": []}]
    benchmark.write_json(tmp_path / "suite.json", {"development": cases})
    benchmark.write_json(tmp_path / "contract.json", {"version": 1})
    labels = [{"id": 1, "common_answers": [["first", "1"], ["second", "2"]]},
              {"id": 2, "common_answers": [["third", "3"]]}]
    (tmp_path / "oracle/labels.jsonl").write_text("".join(json.dumps(label) + "\n" for label in labels), encoding="utf-8")
    result = benchmark.aggregate(tmp_path, "development")
    assert not result["complete"]
    for arm in ["off", "on"]:
        assert result["arms"][arm]["missing"] == 2
        assert result["arms"][arm]["total_subanswers"] == 3
        assert result["arms"][arm]["official_raw"] == {"ABQ": 0, "PASQ": 0, "UASQ": 0}


@pytest.mark.parametrize("status", ["completed", "error"])
def test_worker_passes_only_original_task_and_separates_answer_from_delivery(benchmark, scorer, tmp_path, monkeypatch, status):
    # The fake session tests only API transport. It does not claim real execution.
    import labweaver.agent as agent
    import labweaver.config as config
    import labweaver.run_config as run_config
    import labweaver.runtime.execution as execution

    case = {"id": "DA1", "official_id": 1, "dataset": "data.csv", "level": "easy",
            "concepts": ["Summary Statistics"], "task": "Compute a value.\n@result[value]"}
    benchmark.write_json(tmp_path / "suite.json", {"development": [case], "questions": {case["id"]: {
        "question": "Compute a value.", "format": "@result[value]"}}})
    benchmark.write_json(tmp_path / "contract.json", {"version": 1})
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/data.csv").write_bytes(b"field\n1\n")
    (tmp_path / "oracle/labels.jsonl").write_text(json.dumps({"id": 1, "common_answers": [["result", "7"]]}) + "\n", encoding="utf-8")
    destination = tmp_path / "development/trials/DA1-1-on"
    destination.mkdir(parents=True)
    state = {"invoked": False, "cancelled": False}
    executor = SimpleNamespace(set_owner=lambda *args: None, cancel=lambda: state.update(cancelled=True))

    class Session:
        artifact_assets = {}

        def invoke(self, task, **kwargs):
            assert task == case["task"]
            assert "common_answers" not in task
            state["invoked"] = True
            return {"status": status, "messages": [{"type": "ai", "content": '@result[7]'}],
                    "missing_deliverables": []}

        def save(self, path):
            return {"record_path": str(path / "record.json")}

    session = Session()
    session.executor = executor

    def create(csv_path, main_model, **kwargs):
        assert csv_path == tmp_path / "inputs/data.csv"
        assert kwargs["skills_enabled"] is True
        assert "history_context" not in kwargs
        assert "material_paths" not in kwargs
        return session

    class Probe:
        image_id = "sha256:test-only"

        def __init__(self, *args, **kwargs):
            pass

        def preflight(self):
            pass

    model = SimpleNamespace(model_id="test", base_url="https://example.invalid", timeout=30)
    monkeypatch.setattr(benchmark, "verify_contract", lambda *args: {"model": vars(model), "image": {"id": Probe.image_id}})
    monkeypatch.setattr(agent, "create_session", create)
    monkeypatch.setattr(config, "load_config", lambda *args: model)
    class Formatter:
        def invoke(self, messages):
            payload = json.loads(messages[1].content)
            assert set(payload) == {"question", "requested_format", "analyst_answer"}
            assert payload["analyst_answer"] == "@result[7]"
            return SimpleNamespace(content="@result[7]", usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})

    monkeypatch.setattr(config, "create_model", lambda *args, **kwargs: Formatter())
    monkeypatch.setattr(run_config, "load_intake_config", lambda: SimpleNamespace(env_file=None))
    monkeypatch.setattr(execution, "DockerExecutor", Probe)
    original_jsonl = benchmark.jsonl

    def read_oracle(path):
        if Path(path).name == "labels.jsonl":
            assert state["invoked"], "Labels must not be read before the Agent invocation"
        return original_jsonl(path)

    monkeypatch.setattr(benchmark, "jsonl", read_oracle)
    args = SimpleNamespace(output=tmp_path, source=Path(__file__).parents[1] / "src", split="development",
                           case="DA1", repeat=1, arm="on", image="test-image")
    benchmark.run_worker(args)
    summary = benchmark.read_json(destination / "summary.json")
    assert state["cancelled"]
    assert summary["answer_correct"] is True
    assert summary["delivery_completed"] is (status == "completed")
    assert summary["end_to_end"] is (status == "completed")
    assert summary["formatter"]["usage"]["total_tokens"] == 7
    assert summary["known_total_tokens"] == 0  # Formatting is separate evaluation overhead.
