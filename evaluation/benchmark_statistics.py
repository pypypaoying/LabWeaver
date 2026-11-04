"""Paired benchmark accounting with dataset-clustered uncertainty.

Repeated model calls are observations, not independent benchmark problems.  All
questions and repeats sharing a CSV are resampled together.  Missing trials are
reported explicitly; they are never silently dropped from planned accounting.
This module knows neither model clients nor benchmark answers.
"""

from collections import defaultdict
import math
import random
import statistics


METRICS = ("answer_correct", "delivery_completed", "end_to_end")
ARMS = ("off", "on")


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _identity(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _nonnegative_integer(record, field):
    value = record.get(field)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{field} must be a nonnegative integer or unknown")
    return value


def _percentile(ordered, probability):
    """Linear empirical percentile, including the one-observation boundary."""
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _validate(trials, planned_cases, repetitions):
    cases = {}
    for case in planned_cases:
        if not isinstance(case, dict):
            raise ValueError("Planned cases must be dictionaries")
        case_id = _identity(case.get("id"), "Case id")
        dataset = _identity(case.get("dataset"), "Dataset id")
        if case_id in cases:
            raise ValueError("Duplicate planned case")
        cases[case_id] = dataset
    records = {}
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Trials must be dictionaries")
        case = trial.get("case")
        if not isinstance(case, str) or case not in cases:
            raise ValueError("Unknown benchmark case")
        arm, repeat = trial.get("arm"), trial.get("repeat")
        if arm not in ARMS:
            raise ValueError("Unknown benchmark arm")
        if type(repeat) is not int or not 1 <= repeat <= repetitions:
            raise ValueError("Repeat outside planned range")
        if trial.get("dataset") != cases[case]:
            raise ValueError("Mixed or changed dataset identity")
        key = (case, repeat, arm)
        if key in records:
            raise ValueError("Duplicate benchmark trial")
        for metric in METRICS:
            if type(trial.get(metric)) is not bool:
                raise ValueError(f"{metric} must be boolean")
        if trial["end_to_end"] and not (
            trial["answer_correct"] and trial["delivery_completed"]
        ):
            raise ValueError("End-to-end success requires answer and delivery success")
        _nonnegative_integer(trial, "known_total_tokens")
        _nonnegative_integer(trial, "unknown_usage_calls")
        elapsed = trial.get("elapsed_seconds")
        if elapsed is not None and (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("Elapsed seconds must be finite and nonnegative or unknown")
        if trial.get("skill_loads") is not None and not isinstance(
            trial["skill_loads"], list
        ):
            raise ValueError("Skill loads must be a list or unknown")
        records[key] = trial
    return cases, records


def _arms(cases, records, repetitions):
    result = {}
    planned = len(cases) * repetitions
    for arm in ARMS:
        observed = [record for key, record in sorted(records.items()) if key[2] == arm]
        count = len(observed)
        elapsed = [r["elapsed_seconds"] for r in observed if r.get("elapsed_seconds") is not None]
        known_usage = [r for r in observed if r.get("known_total_tokens") is not None]
        unknown_calls = [r for r in observed if r.get("unknown_usage_calls") is not None]
        skill_measured = [r for r in observed if r.get("skill_loads") is not None]
        successes = {metric: sum(r[metric] for r in observed) for metric in METRICS}
        result[arm] = {
            "planned": planned,
            "recorded": count,
            "missing": planned - count,
            **successes,
            "rates_observed": {
                metric: successes[metric] / count if count else None for metric in METRICS
            },
            "rates_planned": {
                metric: successes[metric] / planned if planned else None for metric in METRICS
            },
            "observed_rate_denominator": count,
            "planned_rate_denominator": planned,
            "planned_rate_policy": "Missing trials are unachieved; observed successes / planned trials.",
            "known_total_tokens": sum(r["known_total_tokens"] for r in known_usage),
            "usage_unmeasured_trials": count - len(known_usage),
            "unknown_usage_calls": sum(r["unknown_usage_calls"] for r in unknown_calls),
            "usage_call_counts_unmeasured_trials": count - len(unknown_calls),
            "median_seconds": statistics.median(elapsed) if elapsed else None,
            "timing_unmeasured_trials": count - len(elapsed),
            "skill_reads": sum(len(r["skill_loads"]) for r in skill_measured),
            "trials_reading_skills": sum(bool(r["skill_loads"]) for r in skill_measured),
            "skill_reads_unmeasured_trials": count - len(skill_measured),
        }
    return result


def _pairs(cases, records, repetitions):
    paired = {
        metric: dict(off_only=0, on_only=0, both_pass=0, both_fail=0, incomplete=0)
        for metric in METRICS
    }
    complete = []
    for case in sorted(cases):
        for repeat in range(1, repetitions + 1):
            off = records.get((case, repeat, "off"))
            on = records.get((case, repeat, "on"))
            if off is None or on is None:
                for metric in METRICS:
                    paired[metric]["incomplete"] += 1
                continue
            complete.append((cases[case], case, repeat, off, on))
            for metric in METRICS:
                key = (
                    "both_pass" if off[metric] and on[metric]
                    else "off_only" if off[metric]
                    else "on_only" if on[metric]
                    else "both_fail"
                )
                paired[metric][key] += 1
    return paired, complete


def _cluster_intervals(complete_pairs, planned_pairs, bootstrap_samples, seed):
    groups = defaultdict(list)
    for pair in complete_pairs:
        groups[pair[0]].append(pair)
    datasets = sorted(groups)
    totals = {
        dataset: {
            metric: sum(int(p[4][metric]) - int(p[3][metric]) for p in groups[dataset])
            for metric in METRICS
        }
        for dataset in datasets
    }
    sizes = {dataset: len(groups[dataset]) for dataset in datasets}
    count = len(complete_pairs)
    points = {
        metric: sum(totals[d][metric] for d in datasets) / count if count else None
        for metric in METRICS
    }
    samples = {metric: [] for metric in METRICS}
    if datasets:
        generator = random.Random(seed)
        for _ in range(bootstrap_samples):
            # Same sampled clusters for all metrics. Questions and repetitions
            # within a selected dataset always travel together.
            draw = generator.choices(datasets, k=len(datasets))
            denominator = sum(sizes[dataset] for dataset in draw)
            for metric in METRICS:
                samples[metric].append(sum(totals[d][metric] for d in draw) / denominator)
    result = {}
    for metric in METRICS:
        ordered = sorted(samples[metric])
        result[metric] = {
            "on_minus_off": points[metric],
            "lower": _percentile(ordered, 0.025) if ordered else None,
            "upper": _percentile(ordered, 0.975) if ordered else None,
            "confidence_level": 0.95,
            "method": "dataset_cluster_percentile_bootstrap",
            "resampling_unit": "dataset",
            "estimand": "Difference of success rates across complete case/repeat OFF-ON pairs.",
            "complete_pairs": count,
            "excluded_incomplete_pairs": planned_pairs - count,
            "dataset_clusters": len(datasets),
            "bootstrap_samples": bootstrap_samples if datasets else 0,
            "seed": seed,
            "boundary": (
                "No complete pairs; interval unavailable." if not datasets
                else "One dataset only; degenerate interval cannot measure between-dataset uncertainty."
                if len(datasets) == 1
                else "Exploratory interval on this fixed subset; repeated calls are not independent tasks."
            ),
        }
    return result


def summarize_trials(
    trials,
    *,
    planned_cases,
    repetitions=3,
    bootstrap_samples=4000,
    seed=20261103,
):
    """Account for all planned trials and compare complete OFF/ON pairs.

    ``rates_observed`` uses recorded trials, while ``rates_planned`` counts
    missing trials as unachieved. Intervals use only complete pairs, explicitly
    disclosing exclusions, and resample entire datasets. They neither certify
    significance nor estimate performance on arbitrary real-world tasks.
    Missing usage/timing measurements are counted separately, never imputed as
    measured zero. Inputs are validated and left unchanged.
    """
    _positive_integer(repetitions, "repetitions")
    _positive_integer(bootstrap_samples, "bootstrap_samples")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    cases, records = _validate(trials, planned_cases, repetitions)
    paired, complete = _pairs(cases, records, repetitions)
    strict = {}
    for metric in METRICS:
        strict[metric] = {}
        for arm in ARMS:
            complete_cases = 0
            passed_cases = 0
            for case in sorted(cases):
                runs = [records.get((case, repeat, arm)) for repeat in range(1, repetitions + 1)]
                if all(record is not None for record in runs):
                    complete_cases += 1
                    passed_cases += all(record[metric] for record in runs)
            strict[metric][arm] = {
                "passed_cases": passed_cases,
                "complete_cases": complete_cases,
                "planned_cases": len(cases),
                "rate_planned": passed_cases / len(cases) if cases else None,
            }
    planned = len(cases) * repetitions * len(ARMS)
    return {
        "version": 1,
        "complete": len(records) == planned,
        "planned_trials": planned,
        "recorded_trials": len(records),
        "unique_cases": len(cases),
        "unique_datasets": len(set(cases.values())),
        "repetitions": repetitions,
        "arms": _arms(cases, records, repetitions),
        "paired": paired,
        "confidence_intervals": _cluster_intervals(
            complete, len(cases) * repetitions, bootstrap_samples, seed
        ),
        "strict_all_repeats": strict,
        "boundary": "Subset estimates only. Skill reads are adoption, not proof of following guidance or task correctness.",
    }
