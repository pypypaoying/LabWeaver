"""Exercise both personal-data cases through the real, offline Agent loop."""

import importlib.util
import json
from pathlib import Path
import shutil
import socket

import pytest

from labweaver.run_config import load_intake_config


ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("real_checks_under_test", ROOT / "run_real_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case", ["matched", "no-answer", "invalid-pdf"])
def test_real_checks_pair_uses_external_task_and_files_without_credentials_or_network(
    tmp_path, monkeypatch, case,
):
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(name, "false")

    def denied(*args, **kwargs):
        raise AssertionError("Offline verification must not access model credentials or the network")

    import labweaver.runtime.intake as runtime

    monkeypatch.setattr(runtime, "load_config", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    config = load_intake_config(ROOT / "labweaver.toml", overrides={"mode": "offline"})
    csv = tmp_path / "my real data.csv"
    pdf = tmp_path / "my references.pdf"
    shutil.copyfile(ROOT / "examples/data/survey.csv", csv)
    if case == "invalid-pdf":
        pdf.write_bytes(b"not a PDF")
    else:
        shutil.copyfile(ROOT / "examples/materials/survey/methods.pdf", pdf)
    task_file = tmp_path / "my task.txt"
    task_file.write_text(
        "quasar_redshift" if case == "no-answer" else "ordinal_survey_method_v1 请说明满意度量表与实际数据规模。",
        encoding="utf-8-sig",
    )
    runner = _runner()
    task = runner._task_text("@" + str(task_file))
    assert task == task_file.read_text(encoding="utf-8-sig")
    assert runner._selected_file('"' + str(csv) + '"', "csv") == csv
    output = tmp_path / "outputs"
    summary = runner._run_pair(config, csv, pdf, task, output)
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == summary
    assert summary["checks"]["inputs_unchanged"] is True
    assert summary["checks"]["csv_only_success"] is True
    records = [json.loads(Path(item["json_path"]).read_text(encoding="utf-8")) for item in summary["runs"]]
    first, second = records
    assert all(record["task"] == task and record["mode"] == "offline" for record in records)
    assert first["source"]["name"] == csv.name
    assert first["materials"] == [] and first["citations"] == []
    assert [entry["name"] for entry in first["execution_ledger"]] == ["profile_csv"]
    assert "brief_path" not in first
    if case == "invalid-pdf":
        assert second["status"] == "error" and second["error"]["code"] == "pdf_invalid"
        assert summary["checks"]["csv_with_pdf_success"] is False
        assert summary["passed"] is False
    else:
        assert summary["checks"]["csv_statistics_identical"] is True
        assert second["materials_completed"] is True and second["materials"][0]["name"] == pdf.name
        assert [entry["name"] for entry in second["execution_ledger"]] == ["profile_csv", "search_materials"]
        assert Path(second["brief_path"]).is_file()
        assert summary["checks"]["pdf_has_retrieved_citations"] == (case == "matched")
        assert summary["passed"] == (case == "matched")
        if case == "matched":
            assert second["citations"] and all(chunk["name"] == pdf.name for chunk in second["retrieved_chunks"])
        else:
            assert second["citations"] == [] and second["retrieved_chunks"] == []
            assert "资料不足" in second["final_answer"]
