"""Source-linked briefs and failure-safe artifact pairs without model calls."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import labweaver.runtime.records as records


def _report():
    sources = [
        {"id": "D1", "name": "requirements.md", "sha256": "a" * 64, "format": "markdown"},
        {"id": "D2", "name": "methods.pdf", "sha256": "b" * 64, "format": "pdf"},
    ]
    chunks = [
        {"id": "D1-C1", "source_id": "D1", "name": "requirements.md", "sha256": "a" * 64,
         "text": "比较分组均值。", "location": {"line_start": 2, "line_end": 4}},
        {"id": "D2-C1", "source_id": "D2", "name": "methods.pdf", "sha256": "b" * 64,
         "text": "确认适用条件后再分析。", "location": {"page": 3}},
    ]
    return {
        "status": "completed", "stage": "intake", "task": "理解数据并提出方案",
        "source": {"name": "survey.csv", "sha256": "c" * 64},
        "materials_completed": True, "materials": sources,
        "retrieved_chunks": chunks, "citations": ["D1-C1", "D2-C1"],
        "final_answer": "候选分析：比较分组均值 [D1-C1]。先确认适用条件 [D2-C1]。",
    }


def test_brief_pair_has_same_stem_real_locations_and_exact_json_path(tmp_path):
    report = _report()
    before = copy.deepcopy(report)
    target = records.save_run(report, tmp_path, with_brief=True)
    saved = json.loads(target.read_text(encoding="utf-8"))
    brief = target.with_suffix(".md")
    assert saved["run_id"] == target.stem == brief.stem
    assert saved["brief_path"] == str(brief)
    text = brief.read_text(encoding="utf-8")
    assert "[D1-C1] requirements.md" in text and "第 2–4 行" in text
    assert "[D2-C1] methods.pdf" in text and "第 3 页" in text
    assert "a" * 64 in text and "比较分组均值。" in text
    assert "仅执行只读概览、筛选、聚合和排名" in text
    assert report == before


def test_default_save_remains_json_only_and_no_hit_materials_still_get_a_brief(tmp_path):
    plain = records.save_run({"status": "completed"}, tmp_path)
    assert "brief_path" not in json.loads(plain.read_text(encoding="utf-8"))
    assert not plain.with_suffix(".md").exists()
    report = _report()
    report.update(retrieved_chunks=[], citations=[], final_answer="资料不足，请补充具体要求。")
    target = records.save_run(report, tmp_path, with_brief=True)
    assert "本次没有可引用的检索片段" in target.with_suffix(".md").read_text(encoding="utf-8")


@pytest.mark.parametrize("change", [
    {"status": "error"}, {"status": "awaiting_input"}, {"citations": ["D9-C99"]},
])
def test_failed_or_untraceable_report_cannot_produce_a_success_brief(tmp_path, change):
    report = _report()
    report.update(change)
    with pytest.raises(ValueError):
        records.save_run(report, tmp_path, with_brief=True)
    assert list(tmp_path.iterdir()) == []


def test_citation_source_mismatch_is_rejected_before_creating_files(tmp_path):
    report = _report()
    report["retrieved_chunks"][0]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="registered material source"):
        records.save_run(report, tmp_path, with_brief=True)
    assert list(tmp_path.iterdir()) == []


def test_independent_markdown_writer_never_replaces_existing_file(tmp_path):
    target = tmp_path / "brief.md"
    target.write_text("keep this existing file", encoding="utf-8")
    with pytest.raises(FileExistsError):
        records.save_markdown_brief(_report(), target)
    assert target.read_text(encoding="utf-8") == "keep this existing file"


def test_json_failure_removes_only_the_brief_created_by_this_attempt(tmp_path, monkeypatch):
    original_open = Path.open

    def fail_json_open(path, mode="r", *args, **kwargs):
        if mode == "x" and path.suffix == ".json":
            raise OSError("simulated JSON storage failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_json_open)
    with pytest.raises(OSError):
        records.save_run(_report(), tmp_path, with_brief=True)
    assert list(tmp_path.iterdir()) == []


def test_preexisting_json_collision_is_preserved_and_new_brief_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "uuid4", lambda: SimpleNamespace(hex="fixed-run-id"))
    existing = tmp_path / "fixed-run-id.json"
    existing.write_text("preserve original JSON", encoding="utf-8")
    with pytest.raises(FileExistsError):
        records.save_run(_report(), tmp_path, with_brief=True)
    assert existing.read_text(encoding="utf-8") == "preserve original JSON"
    assert not existing.with_suffix(".md").exists()


def test_nonfinite_json_is_rejected_before_creating_either_artifact(tmp_path):
    report = _report()
    report["invalid"] = float("nan")
    with pytest.raises(ValueError):
        records.save_run(report, tmp_path, with_brief=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failed_suffix", [".md", ".json"])
def test_partial_stream_write_removes_both_new_artifacts(tmp_path, monkeypatch, failed_suffix):
    original_open = Path.open

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *details):
            return self.stream.__exit__(*details)

        def write(self, payload):
            self.stream.write(payload[:1])
            raise OSError("simulated failure after partial write")

    def intercept(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if mode == "x" and path.suffix == failed_suffix:
            return PartialWrite(stream)
        return stream

    monkeypatch.setattr(Path, "open", intercept)
    with pytest.raises(OSError):
        records.save_run(_report(), tmp_path, with_brief=True)
    assert list(tmp_path.iterdir()) == []


def test_completed_statistics_create_exact_integer_csv_and_session_task_artifacts(tmp_path):
    report = {
        "session_id": "session-123", "task_id": "task-456", "status": "completed",
        "task": "按 NOC 累计 Total 排名", "source": {"name": "medals.csv", "sha256": "c" * 64},
        "final_answer": "工具已计算。", "analysis_results": [{
            "status": "completed", "source": {"name": "medals.csv", "sha256": "c" * 64},
            "columns": ["NOC", "medals"], "rows": [{"NOC": "A, B", "medals": 9007199254740993}],
            "spec": {"group_by": [1], "metrics": [{"op": "sum", "column": 2, "alias": "medals"}]},
        }],
    }
    target = records.save_run(report, tmp_path, with_brief=True)
    assert target.parent == tmp_path / "session-123" / "task-456"
    document = json.loads(target.read_text(encoding="utf-8"))
    result_csv = Path(document["result_csv_paths"][0])
    with result_csv.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"NOC": "A, B", "medals": "9007199254740993"}]
    assert "9007199254740993" in target.with_suffix(".md").read_text(encoding="utf-8")
    assert "result_csv_paths" not in report


def test_invalid_statistics_source_is_rejected_before_any_write(tmp_path):
    report = {"status": "completed", "source": {"name": "A", "sha256": "a" * 64},
              "analysis_results": [{"status": "completed", "source": {"name": "B", "sha256": "b" * 64},
                                    "columns": ["value"], "rows": [{"value": 1}]}]}
    with pytest.raises(ValueError, match="bound CSV"):
        records.save_run(report, tmp_path, with_brief=True)
    assert list(tmp_path.iterdir()) == []


def test_untrusted_session_identifier_cannot_escape_output_folder(tmp_path):
    with pytest.raises(ValueError, match="safe artifact identifier"):
        records.save_run({"status": "error", "session_id": "../../outside"}, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_followup_can_cite_verified_session_evidence_without_new_search(tmp_path):
    report = _report()
    report["citation_chunks"] = report.pop("retrieved_chunks")
    report["retrieved_chunks"] = []
    target = records.save_run(report, tmp_path, with_brief=True)
    assert "[D1-C1] requirements.md" in target.with_suffix(".md").read_text(encoding="utf-8")
