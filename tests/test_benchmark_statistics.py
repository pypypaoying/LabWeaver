"""Independent accounting checks for repeated paired benchmark runs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def stats():
    path = Path(__file__).resolve().parents[1] / "evaluation/benchmark_statistics.py"
    spec = importlib.util.spec_from_file_location("benchmark_statistics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trial(case="A", dataset="one.csv", repeat=1, arm="off", correct=True, delivered=True, **changes):
    return {
        "case": case,
        "dataset": dataset,
        "repeat": repeat,
        "arm": arm,
        "answer_correct": correct,
        "delivery_completed": delivered,
        "end_to_end": correct and delivered,
        "known_total_tokens": 100,
        "unknown_usage_calls": 0,
        "elapsed_seconds": 3,
        "skill_loads": [],
        **changes,
    }


def test_missing_trials_stay_in_planned_denominator_and_pairs(stats):
    rows = [trial(), trial(arm="on", correct=False, skill_loads=[{"skill_id": "test"}])]
    result = stats.summarize_trials(rows, planned_cases=[{"id": "A", "dataset": "one.csv"}])
    assert not result["complete"]
    assert result["planned_trials"] == 6
    off, on = result["arms"]["off"], result["arms"]["on"]
    assert off["recorded"] == 1 and off["missing"] == 2
    assert off["rates_observed"]["answer_correct"] == 1
    assert off["rates_planned"]["answer_correct"] == pytest.approx(1 / 3)
    assert on["skill_reads"] == on["trials_reading_skills"] == 1
    assert result["paired"]["answer_correct"] == dict(
        off_only=1, on_only=0, both_pass=0, both_fail=0, incomplete=2
    )
    interval = result["confidence_intervals"]["answer_correct"]
    assert interval["on_minus_off"] == interval["lower"] == interval["upper"] == -1
    assert interval["complete_pairs"] == interval["dataset_clusters"] == 1
    assert interval["excluded_incomplete_pairs"] == 2
    assert "degenerate" in interval["boundary"]
    assert result["strict_all_repeats"]["answer_correct"]["off"]["passed_cases"] == 0


def test_answer_delivery_and_end_to_end_are_separate(stats):
    rows = [trial(delivered=False), trial(arm="on", correct=False)]
    result = stats.summarize_trials(rows, planned_cases=[{"id": "A", "dataset": "one.csv"}], repetitions=1)
    assert result["complete"]
    assert result["paired"]["answer_correct"]["off_only"] == 1
    assert result["paired"]["delivery_completed"]["on_only"] == 1
    assert result["paired"]["end_to_end"]["both_fail"] == 1
    assert result["arms"]["off"]["answer_correct"] == 1
    assert result["arms"]["off"]["end_to_end"] == 0


def test_strict_all_repeats_does_not_count_one_success_as_reliable_task(stats):
    rows = [trial(repeat=n, arm=arm, correct=not (arm == "on" and n == 2))
            for n in (1, 2, 3) for arm in ("off", "on")]
    result = stats.summarize_trials(rows, planned_cases=[{"id": "A", "dataset": "one.csv"}])
    strict = result["strict_all_repeats"]["answer_correct"]
    assert strict["off"] == dict(passed_cases=1, complete_cases=1, planned_cases=1, rate_planned=1)
    assert strict["on"] == dict(passed_cases=0, complete_cases=1, planned_cases=1, rate_planned=0)


def test_cluster_bootstrap_resamples_all_repeats_and_questions_together(stats):
    # +1 across six observations on one CSV, -1 across three on the other.
    # A cluster draw therefore yields only -1, +1/3, or +1, never intermediate
    # values from independently drawing the nine correlated calls.
    cases = [{"id": "A", "dataset": "one.csv"}, {"id": "B", "dataset": "one.csv"},
             {"id": "C", "dataset": "two.csv"}]
    rows = [trial(case=c["id"], dataset=c["dataset"], repeat=n, arm=arm,
                  correct=(arm == "on") if c["id"] != "C" else (arm == "off"))
            for c in cases for n in (1, 2, 3) for arm in ("off", "on")]
    result = stats.summarize_trials(rows, planned_cases=cases, bootstrap_samples=1000)
    interval = result["confidence_intervals"]["answer_correct"]
    assert interval["dataset_clusters"] == 2
    assert interval["complete_pairs"] == 9
    assert interval["on_minus_off"] == pytest.approx(1 / 3)
    assert (interval["lower"], interval["upper"]) == (-1, 1)
    assert result["unique_cases"] == 3 and result["unique_datasets"] == 2
    assert result == stats.summarize_trials(list(reversed(rows)), planned_cases=list(reversed(cases)), bootstrap_samples=1000)


def test_unknown_cost_and_timing_are_not_reported_as_measured_zero(stats):
    rows = [trial(known_total_tokens=None, unknown_usage_calls=None,
                  elapsed_seconds=None, skill_loads=None),
            trial(arm="on", unknown_usage_calls=2, known_total_tokens=37, elapsed_seconds=8)]
    result = stats.summarize_trials(rows, planned_cases=[{"id": "A", "dataset": "one.csv"}], repetitions=1)
    off, on = result["arms"]["off"], result["arms"]["on"]
    assert off["usage_unmeasured_trials"] == off["usage_call_counts_unmeasured_trials"] == 1
    assert off["timing_unmeasured_trials"] == off["skill_reads_unmeasured_trials"] == 1
    assert off["median_seconds"] is None
    assert on["known_total_tokens"] == 37 and on["unknown_usage_calls"] == 2
    assert on["median_seconds"] == 8


def test_empty_records_have_no_observed_rate_or_invented_interval(stats):
    result = stats.summarize_trials([], planned_cases=[{"id": "A", "dataset": "one.csv"}])
    assert result["arms"]["off"]["rates_observed"]["answer_correct"] is None
    assert result["arms"]["off"]["rates_planned"]["answer_correct"] == 0
    assert result["confidence_intervals"]["answer_correct"]["lower"] is None
    assert result["confidence_intervals"]["answer_correct"]["excluded_incomplete_pairs"] == 3
    json.dumps(result, allow_nan=False)


def test_inputs_are_not_mutated(stats):
    rows, cases = [trial(), trial(arm="on")], [{"id": "A", "dataset": "one.csv"}]
    original = copy.deepcopy((rows, cases))
    stats.summarize_trials(rows, planned_cases=cases)
    assert (rows, cases) == original


@pytest.mark.parametrize("change,match", [
    ({"case": "unknown"}, "Unknown benchmark case"),
    ({"dataset": "changed.csv"}, "dataset identity"),
    ({"arm": "maybe"}, "arm"),
    ({"repeat": 4}, "Repeat"),
    ({"repeat": True}, "Repeat"),
    ({"answer_correct": 1}, "boolean"),
    ({"delivery_completed": None}, "boolean"),
    ({"end_to_end": "yes"}, "boolean"),
    ({"known_total_tokens": -1}, "nonnegative"),
    ({"known_total_tokens": True}, "nonnegative"),
    ({"unknown_usage_calls": 1.5}, "nonnegative"),
    ({"elapsed_seconds": float("nan")}, "finite"),
    ({"elapsed_seconds": -1}, "finite"),
    ({"skill_loads": {}}, "Skill loads"),
])
def test_invalid_trial_is_rejected(stats, change, match):
    with pytest.raises(ValueError, match=match):
        stats.summarize_trials([trial(**change)], planned_cases=[{"id": "A", "dataset": "one.csv"}])


def test_duplicate_trials_plans_and_impossible_success_are_rejected(stats):
    cases = [{"id": "A", "dataset": "one.csv"}]
    with pytest.raises(ValueError, match="Duplicate benchmark trial"):
        stats.summarize_trials([trial(), trial()], planned_cases=cases)
    with pytest.raises(ValueError, match="Duplicate planned case"):
        stats.summarize_trials([], planned_cases=cases + cases)
    with pytest.raises(ValueError, match="End-to-end"):
        stats.summarize_trials([trial(correct=False, end_to_end=True)], planned_cases=cases)


@pytest.mark.parametrize("settings", [
    {"repetitions": 0}, {"repetitions": True}, {"bootstrap_samples": 0},
    {"bootstrap_samples": 2.5}, {"seed": True},
])
def test_invalid_configuration_is_rejected(stats, settings):
    with pytest.raises(ValueError):
        stats.summarize_trials([], planned_cases=[], **settings)
