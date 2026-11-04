"""Versioned, no-model rescore of existing DABench formatter responses.

Correct two transport bugs discovered on development cases: a verbatim public
format placeholder is not an analyst answer; U+2212 is the same minus sign as
ASCII '-'. No question is solved, no code or model reruns, and v2 is retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / ".cache/evaluation/skills-dabench-v2"
RULES = {
    "version": 3,
    "placeholder": "Remove only @key[value] tokens whose value exactly equals that key's placeholder in the public question.format.",
    "minus": "Replace Unicode U+2212 with ASCII '-' in the replayed analyst text and existing formatter raw_response only.",
    "replay": "Call the frozen format_answer with a local FakeModel returning only the previously recorded raw_response; no new inference.",
    "preserve_actual_answers": "Actual original @key[value] answers remain immutable, including wrong numbers.",
    "missing_response": "If a formatter raw_response is absent or empty, preserve its original recorded score/failure.",
    "scope": "Apply the same rule to every planned sample; never inspect labels while deciding the formatting rule.",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Refuse linked rescore output")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def remove_format_placeholders(answer, public_format):
    """Only exact, key-specific copies of the public template are removed."""
    placeholders = {}
    for key, value in re.findall(r"@(\w+)\[(.*?)\]", public_format):
        placeholders.setdefault(key, set()).add(value)
    removed = []

    def replace(match):
        key, value = match.groups()
        if value in placeholders.get(key, set()):
            removed.append({"key": key, "placeholder": value})
            return " "
        return match.group(0)

    return re.sub(r"@(\w+)\[(.*?)\]", replace, answer), removed


def replay_existing_response(question, scoring, *, formatter=None):
    """Formatting decisions are blind: no labels, scores, data or artifacts.

    A local fake model supplies a saved response to the exact frozen adapter.
    Its invoke method does no inference and reports no additional token usage.
    """
    original = scoring.get("formatter", {})
    if not isinstance(original, dict):
        raise ValueError("Invalid source formatter record")
    raw_response = original.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        return {"replayed": False, "reason": "no_existing_formatter_raw_response",
                "response": original.get("response") if isinstance(original.get("response"), str) else "__NO_ANSWER__",
                "removed_placeholders": [], "minus_signs_normalized": 0,
                "new_model_calls": 0, "new_tokens": 0}
    raw_answer = scoring.get("raw_text", "")
    if not isinstance(raw_answer, str):
        raise ValueError("Invalid source analyst answer")
    prepared, removed = remove_format_placeholders(raw_answer, question["format"])
    minus_count = prepared.count("\u2212") + raw_response.count("\u2212")
    prepared = prepared.replace("\u2212", "-")
    response = raw_response.replace("\u2212", "-")
    if formatter is None:
        from answer_format import format_answer
        formatter = format_answer
    calls = []

    class FakeModel:
        def invoke(self, messages):
            calls.append(1)
            # Verify the transport stays blind even if the adapter changes.
            payload = json.loads(messages[1].content)
            if set(payload) != {"question", "requested_format", "analyst_answer"}:
                raise ValueError("Unexpected formatter replay inputs")
            return SimpleNamespace(content=response, usage_metadata=None)

    replay = formatter(FakeModel(), question={"question": question["question"], "format": question["format"]}, answer=prepared)
    return {"replayed": True, "reason": "saved_response_replayed",
            "response": replay["response"], "removed_placeholders": removed,
            "minus_signs_normalized": minus_count, "replay_invocations": len(calls),
            "rejected": replay.get("rejected", []),
            "semantic_review_keys": replay.get("semantic_review_keys", []),
            "kept_original_keys": replay.get("kept_original_keys", []),
            "new_model_calls": 0, "new_tokens": 0}


def freeze_rescore_contract(batch, destination, split, source_contract):
    evaluator = Path(__file__).resolve().parent
    dependencies = ["answer_format.py", "dabench.py", "benchmark_statistics.py"]
    expected = source_contract["evaluator_files"]
    for name in dependencies:
        if digest((evaluator / name).read_bytes()) != expected[name]:
            raise ValueError("Frozen evaluator changed before rescoring")
    contract = {"version": 3, "split": split, "rules": RULES,
                "source_contract_sha256": digest((batch / "contract.json").read_bytes()),
                "source_suite_sha256": digest((batch / "suite.json").read_bytes()),
                "script_sha256": digest(Path(__file__).read_bytes()),
                "dependency_sha256": {name: expected[name] for name in dependencies},
                "new_model_calls": 0, "new_tokens": 0,
                "cost_policy": "Original Agent and formatter usage is referenced once per trial, not newly incurred; v2/v3 costs must not be added."}
    path = destination / "contract.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("Rescore rules or source contract changed; use a new rescore destination")
    write_json(path, contract)
    return contract


def _official_metrics(scorer, results):
    return {"ABQ": scorer.evaluate_accuracy_by_question(results),
            "PASQ": scorer.evaluate_accuracy_proportional_by_sub_question_adjusted(results),
            "UASQ": scorer.evaluate_accuracy_by_sub_question(results)}


def aggregate_rescore(source_rows, rescored_rows, *, cases, repetitions, labels, scorer):
    """Planned denominators and dataset-clustered CIs, with v2 kept alongside v3."""
    from benchmark_statistics import summarize_trials
    from dabench import score_one
    v2 = summarize_trials(source_rows, planned_cases=cases, repetitions=repetitions)
    v3 = summarize_trials(rescored_rows, planned_cases=cases, repetitions=repetitions)
    official = {}
    for arm in ["off", "on"]:
        original_lookup = {(r["case"], r["repeat"]): r for r in source_rows if r["arm"] == arm}
        rescore_lookup = {(r["case"], r["repeat"]): r for r in rescored_rows if r["arm"] == arm}
        before, after, raw = [], [], []
        for case in cases:
            for repeat in range(1, repetitions + 1):
                label = labels[case["official_id"]]
                original = original_lookup.get((case["id"], repeat), {})
                rescored = rescore_lookup.get((case["id"], repeat), {})
                missing = score_one(scorer, label, "")
                before.append(original.get("normalized_score") or missing)
                after.append(rescored.get("normalized_score") or missing)
                raw.append(original.get("raw_score") or missing)
        official[arm] = {"v2_original": _official_metrics(scorer, before),
                         "v3_rescored": _official_metrics(scorer, after),
                         "raw_unchanged": _official_metrics(scorer, raw),
                         "planned_subanswers": sum(len(result["correctness"]) for result in before)}
    original_by_id = {(r["case"], r["repeat"], r["arm"]): r for r in source_rows}
    changes = []
    for row in rescored_rows:
        key = row["case"], row["repeat"], row["arm"]
        original = original_by_id[key]
        if row["answer_correct"] != original["answer_correct"]:
            changes.append({"case": row["case"], "repeat": row["repeat"], "arm": row["arm"],
                            "v2_answer_correct": original["answer_correct"], "v3_answer_correct": row["answer_correct"]})
    return {"version": 3, "complete": v3["complete"], "v2_original": v2, "v3_rescored": v3,
            "official": official, "changed_trials": changes,
            "upgraded_trials": sum(row["v3_answer_correct"] for row in changes),
            "downgraded_trials": sum(not row["v3_answer_correct"] for row in changes),
            "new_model_calls": 0, "new_tokens": 0,
            "boundary": "Post-hoc transport-bug correction derived from development cases, applied uniformly to the frozen subset. No new analysis or model inference; original v2 scores are retained."}


def rescore_batch(batch, *, split="heldout", destination=None):
    """Resume safely as original trials finish; never overwrite source records."""
    from dabench import verify_contract, load_scorer, score_one, score_flags, schedule
    batch = Path(batch).resolve()
    if split not in {"development", "heldout"}:
        raise ValueError("Unknown split")
    source_contract = verify_contract(batch, batch / "implementation/src")
    suite = read_json(batch / "suite.json")
    if destination is None:
        destination = batch / ("rescored-format-v3" if split == "heldout" else "rescored-format-v3-development")
    destination = Path(destination).resolve()
    if destination.parent != batch or destination in {batch / "heldout", batch / "development", batch / "oracle"}:
        raise ValueError("Rescore destination must be a new direct child of the source batch")
    contract = freeze_rescore_contract(batch, destination, split, source_contract)
    contract_sha = digest((destination / "contract.json").read_bytes())
    repetitions = 1 if split == "development" else 3
    cases = suite[split]
    prepared = []
    for case, repeat, arm in schedule(cases, repetitions):
        name = f"{case['id']}-{repeat}-{arm}"
        source_dir = batch / split / "trials" / name
        summary_path, scoring_path = source_dir / "summary.json", source_dir / "scoring.json"
        if not summary_path.is_file():
            continue
        original = read_json(summary_path)
        identity = {"case": case["id"], "dataset": case["dataset"], "repeat": repeat,
                    "arm": arm, "contract_sha256": contract["source_contract_sha256"]}
        if any(original.get(key) != value for key, value in identity.items()):
            raise ValueError("Source trial identity or contract mismatch")
        scoring = read_json(scoring_path) if scoring_path.is_file() else {}
        provenance = {"version": 3, "rescore_contract_sha256": contract_sha,
                      "source_summary_sha256": digest(summary_path.read_bytes()),
                      "source_scoring_sha256": digest(scoring_path.read_bytes()) if scoring_path.is_file() else None,
                      "source_trial": (Path(split) / "trials" / name).as_posix()}
        target = destination / "trials" / name
        if (target / "summary.json").is_file():
            recorded_provenance = read_json(target / "provenance.json")
            if any(recorded_provenance.get(key) != value for key, value in provenance.items()):
                raise ValueError("Original trial changed after rescore; refuse mixed provenance")
            for filename in ["summary.json", "scoring.json"]:
                recorded_hash = recorded_provenance.get("output_sha256", {}).get(filename)
                if recorded_hash != digest((target / filename).read_bytes()):
                    raise ValueError("Rescored output changed; refuse cached score tampering")
            updated = read_json(target / "summary.json")
            if updated.get("rescore_contract_sha256") != contract_sha:
                raise ValueError("Mixed rescore trial contracts")
            prepared.append((case, original, scoring, updated, None, provenance, target))
            continue
        # All formatting decisions finish before labels are loaded below.
        replay = replay_existing_response(suite["questions"][case["id"]], scoring)
        prepared.append((case, original, scoring, None, replay, provenance, target))
    # Separate scoring phase: labels do not enter replay_existing_response.
    scorer = load_scorer(batch)
    labels = {label["id"]: label for label in (
        json.loads(line) for line in (batch / "oracle/labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    source_rows, rescored_rows = [], []
    for case, original, scoring, cached, replay, provenance, target in prepared:
        if cached is None:
            label = labels[case["official_id"]]
            # A partially written scoring.json cannot upgrade a worker failure.
            # Use the same recorded-summary source as the original aggregate.
            old_score = original.get("normalized_score") or score_one(scorer, label, "")
            new_score = score_one(scorer, label, replay["response"]) if replay["replayed"] else old_score
            passed, correct, total = score_flags(new_score)
            updated = copy.deepcopy(original)
            updated.update(answer_correct=passed, end_to_end=passed and original["delivery_completed"],
                           correct_subanswers=correct, total_subanswers=total, normalized_score=new_score,
                           rescore_contract_sha256=contract_sha, format_rescore=replay)
            write_json(target / "scoring.json", {"version": 3, "v2_original_score": old_score,
                       "v3_rescored_score": new_score, "format_rescore": replay,
                       "raw_text_sha256": digest(scoring.get("raw_text", "").encode("utf-8"))})
            write_json(target / "summary.json", updated)
            write_json(target / "provenance.json", {**provenance, "output_sha256": {
                name: digest((target / name).read_bytes()) for name in ["summary.json", "scoring.json"]}})
        else:
            updated = cached
        source_rows.append(original)
        rescored_rows.append(updated)
    report = aggregate_rescore(source_rows, rescored_rows, cases=cases, repetitions=repetitions,
                               labels=labels, scorer=scorer)
    report.update(split=split, rules=RULES, rescore_contract_sha256=contract_sha,
                  source_contract_sha256=contract["source_contract_sha256"],
                  original_trial_records=len(source_rows), output_directory=str(destination))
    write_json(destination / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--split", choices=["development", "heldout"], default="heldout")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    report = rescore_batch(args.batch, split=args.split, destination=args.destination)
    print(json.dumps({key: report[key] for key in ["split", "complete", "original_trial_records", "official",
                                                  "changed_trials", "new_model_calls", "new_tokens"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
