"""Blind, read-only audit of saved DABench formatting decisions.

Only raw_text/formatter/provenance and public trial status fields are inspected.
Oracle files and raw/normalized/deterministic scoring objects are never used.
This emits review candidates; it never changes scores or calls a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re


DEFAULT_BATCH = Path(__file__).resolve().parents[1] / ".cache/evaluation/skills-dabench-v2"
KINDS = (
    "explicit_answer_changed",
    "new_numeric_value",
    "semantic_translation_review",
    "answer_passed_but_core_incomplete",
    "formatted_pass_without_explicit_raw_format",
)
SUMMARY_FIELDS = (
    "case", "dataset", "repeat", "arm", "split", "status", "error", "errors",
    "answer_correct", "raw_answer_correct", "delivery_completed", "end_to_end",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pairs(text):
    return re.findall(r"@(\w+)\[(.*?)\]", text) if isinstance(text, str) else []


def _numbers(text):
    """Literal numeric identity only: no rounding, percentage conversion or math."""
    text = text.replace("\u2212", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    found = set()
    for token in re.findall(r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?![A-Za-z0-9_]|\.\d)", text):
        try:
            value = Decimal(token)
        except InvalidOperation:
            continue
        if value.is_finite():
            found.add(value)
    return found


def inspect_trial(scoring, summary):
    """Project exact input allowlists before any review logic.

    Correctness booleans are the already-recorded outcome, not re-evaluation.
    Do not inspect scoring['raw'], scoring['normalized'] or any label data.
    """
    raw_text = scoring.get("raw_text", "")
    formatter = scoring.get("formatter", {})
    if not isinstance(raw_text, str) or not isinstance(formatter, dict):
        raise ValueError("Invalid raw answer or formatter record")
    formatted = formatter.get("response", "")
    if not isinstance(formatted, str):
        raise ValueError("Invalid formatter response")
    # A shallow projected summary deliberately excludes arbitrary nested fields.
    public_summary = {key: summary[key] for key in SUMMARY_FIELDS if key in summary}
    semantic_keys = formatter.get("semantic_review_keys", [])
    kept_keys = formatter.get("kept_original_keys", [])
    if not isinstance(semantic_keys, list) or not isinstance(kept_keys, list):
        raise ValueError("Invalid formatter review key list")
    semantic_keys = [key for key in semantic_keys if isinstance(key, str)]
    kept_keys = [key for key in kept_keys if isinstance(key, str)]
    original = dict(_pairs(raw_text))
    extracted = dict(_pairs(formatted))
    changes = []
    for key, value in original.items():
        if key in extracted and extracted[key] != value:
            changes.append({"key": key, "original": value, "formatted": extracted[key]})
        elif key in kept_keys and key not in extracted:
            changes.append({"key": key, "original": value, "formatted": None})
    original_numbers = _numbers(raw_text)
    new_values = []
    for key, value in _pairs(formatted):
        missing = _numbers(value) - original_numbers
        if missing:
            new_values.append({"key": key, "formatted": value,
                               "numbers_absent_from_original": sorted(str(v) for v in missing)})
    kinds = []
    if changes:
        kinds.append("explicit_answer_changed")
    if new_values:
        kinds.append("new_numeric_value")
    if semantic_keys:
        kinds.append("semantic_translation_review")
    if (summary.get("answer_correct") is True
            and (summary.get("status") not in {None, "completed"} or summary.get("error"))):
        kinds.append("answer_passed_but_core_incomplete")
    if (not original and summary.get("answer_correct") is True
            and summary.get("raw_answer_correct") is False):
        kinds.append("formatted_pass_without_explicit_raw_format")
    return {"kinds": kinds, "summary": public_summary, "raw_text": raw_text,
            "formatted_response": formatted, "semantic_review_keys": semantic_keys,
            "explicit_changes": changes, "new_numeric_values": new_values,
            "formatter_rejections": [
                {key: item[key] for key in ("key", "reason") if key in item}
                for item in formatter.get("rejected", []) if isinstance(item, dict)],
            "manual_review_required": bool(kinds)}


def audit_batch(batch, *, split="heldout"):
    """Audit an immutable snapshot of currently present trial files.

    Running trials may have no summary yet; those are counted, never declared
    failures or completed. Rerunning the audit does not rerun any Agent task.
    """
    batch = Path(batch).resolve()
    if not batch.is_dir():
        raise ValueError("Benchmark batch directory does not exist")
    if split not in {"development", "heldout", "all"}:
        raise ValueError("Unknown split")
    splits = ["development", "heldout"] if split == "all" else [split]
    reviews, issues = [], []
    audited, summary_missing, by_kind = 0, 0, Counter()
    for selected in splits:
        trials = batch / selected / "trials"
        for path in sorted(trials.glob("*/scoring.json")):
            # Never follow links into an oracle or outside the selected batch.
            if path.is_symlink() or path.resolve().parent.parent != trials.resolve():
                issues.append({"trial": path.parent.name, "split": selected, "error": "unexpected_path"})
                continue
            summary_path = path.with_name("summary.json")
            try:
                scoring = _read_json(path)
                if summary_path.is_file() and not summary_path.is_symlink():
                    summary = _read_json(summary_path)
                else:
                    summary = {}
                    summary_missing += 1
                if not isinstance(scoring, dict) or not isinstance(summary, dict):
                    raise ValueError("Invalid trial object")
                review = inspect_trial(scoring, summary)
            except (OSError, ValueError, TypeError) as exc:
                issues.append({"trial": path.parent.name, "split": selected, "error": type(exc).__name__})
                continue
            audited += 1
            by_kind.update(review["kinds"])
            if review["manual_review_required"]:
                reviews.append({"trial": path.parent.name, "split": selected, **review})
    result = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "split": split,
              "audited_scoring_files": audited, "scoring_without_summary": summary_missing,
              "review_trials": len(reviews), "by_kind": {key: by_kind[key] for key in KINDS},
              "issues": issues, "inputs_read_only": True, "scores_changed": False,
              "model_calls": 0, "oracle_accessed": False,
              "boundary": "Blind review candidates, not proof of semantic equivalence or a replacement score. A number's presence does not prove correct field mapping; semantic translations require human review.",
              "reviews": [{"trial": row["trial"], "split": row["split"], "kinds": row["kinds"],
                           "summary": row["summary"]} for row in reviews]}
    for name, raw in [
        ("format-audit.json", json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"),
        ("format-review.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in reviews)),
    ]:
        destination = batch / name
        if destination.is_symlink():
            raise ValueError("Refuse linked audit output")
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.is_symlink():
            raise ValueError("Refuse linked temporary output")
        temporary.write_text(raw, encoding="utf-8")
        temporary.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--split", choices=["development", "heldout", "all"], default="heldout")
    args = parser.parse_args()
    result = audit_batch(args.batch, split=args.split)
    print(json.dumps({key: result[key] for key in ["split", "audited_scoring_files", "review_trials", "by_kind", "issues"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
