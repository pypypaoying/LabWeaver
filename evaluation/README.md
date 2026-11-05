# Task-level evaluation

The public suite has 24 tasks: 8 analysis, 8 modeling and 8 conversation/retrieval/error cases. `frozen: true` cases are holdouts: do not use their model outputs to tune prompts. Data and expected behavior are fixed before implementing reliability v3. Generated runs, predictions and scripts belong under ignored `.cache/evaluation/`.

Run the shared 16 cases three times on commit `62bb7ad`, then all 24 three times on the candidate. Record every attempt, including failures; development runs must use a separate output directory. Do not compare new modeling capabilities with the baseline's unsupported capabilities. Independent oracles check outputs, not success text. A verifier passing is not itself the task oracle.

The runner records exact data/config fingerprints (never credentials), task attainment, repairs, timings and available token usage. Unknown usage remains null. No cost estimates are invented. The final report must disclose incomplete runs and limitations.

Measurement definitions: a trial attains its task only if every turn satisfies its predefined oracle (including expected clarification/error). “First attempt” additionally requires no model-format retry, failed tool result or host completion feedback. Code repair counts use the actual `execute_python` result, including failed independent checks even when Python exited zero. Conversation trials sum calls and repairs across both turns. Delivery completeness is distinct from numeric correctness. Token totals include only reported usage; request counts without usage are separately unknown. Repeated trials are correlated; rates on this small synthetic suite do not estimate arbitrary real-world task correctness.

Re-scoring saved outputs after the dictionary-row oracle bug is done with `python evaluation/rescore.py .cache/evaluation/baseline`; it preserves the original scoring fields. It does not call any model. `summarize.py` aggregates all attempts, including failures. Frozen cases must not be used for prompt tuning.

New batches copy Python/SQL source into their ignored output directory before running. Each worker records an implementation fingerprint; resuming uses the same checked snapshot, even if the working tree changes. A different implementation requires a new output directory. Creating a `STOP` file in a batch directory stops it after the current trial finishes; it does not erase failures or declare the batch complete. `report.py` exports aggregate results only. Compare the same shared tasks and disclose partial or stopped batches separately. Local wall-clock timing is not an isolated performance benchmark.

`suite.json` preserves v1. With explicit user approval, `suite-v2.json` clarifies only the non-frozen Ridge task M3: random 80/20 regression splitting, without target stratification. All numeric oracles, eight frozen tasks and sixteen shared baseline tasks are unchanged. Do not describe the new M3 result as a same-task improvement over v1. A v2 batch uses `python evaluation/run.py --source src --phase candidate --suite evaluation/suite-v2.json --output .cache/evaluation/candidate-v2 --image labweaver-python:0.3.0`. The runner freezes both task definitions and source code per batch.

## Skills pilot

`python evaluation/skills_pilot.py --prepare` writes eight new synthetic tasks and inputs to ignored `.cache/evaluation/skills-pilot/`. `--run` evaluates Skills OFF and ON three times each (48 fresh sessions), balancing the order within case/repeat pairs. Both arms retain the same model, source snapshot, Python dependencies and budgets. Source freezing includes packaged Markdown/JSON/license resources; content, evaluator, model and immutable image fingerprints are checked on resume. All failures remain; an already-started trial without a summary is not silently submitted again. `--report` creates aggregate JSON/Markdown in the batch directory.

Independent checks cover exact full tables, metrics, plot data, and the modeling task's target/features/source identity/baseline/prediction metrics. A real skill read is measured from the execution ledger, separately from an enabled flag. Skill text is guidance, not task-success evidence. Report paired improvements AND regressions, costs and unmeasured requests. A small synthetic pilot with related repeats cannot establish general reliability or statistical significance. Do not tune old frozen tasks or pool explicit-skill development demonstrations into the comparison.

## External Skills evaluation: InfiAgent-DABench

Completed comparison: [144-trial report](results/skills-dabench.md), [aggregate JSON](results/skills-dabench.json), and [independent review / engineering findings](results/skills-dabench-review.md). The review does not change scores or tune the frozen production Agent.

The synthetic pilot remains an integration/regression suite, not representative evidence of general Skills effectiveness. The external comparison uses [the authors' dataset](https://huggingface.co/datasets/infiagent/DABench/tree/b455d578e30fee513abd79936cbdf7a6de026cb5) and [published scorer](https://github.com/InfiAgent/InfiAgent/blob/3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c/examples/DA-Agent/eval_closed_form.py), pinned to hashes. No benchmark CSVs, labels, raw runs or generated analysis scripts belong in Git.

Reproduce in a **new output directory** with the existing configured model and a prebuilt Docker image:

```powershell
uv run python evaluation/fetch_dabench.py
uv run python evaluation/dabench.py --prepare --output .cache/evaluation/dabench-new
uv run python evaluation/dabench.py --run --split development --output .cache/evaluation/dabench-new
uv run python evaluation/dabench.py --run --split heldout --output .cache/evaluation/dabench-new
uv run python evaluation/dabench.py --report --split heldout --output .cache/evaluation/dabench-new
uv run python evaluation/audit_dabench_format.py --batch .cache/evaluation/dabench-new
uv run python evaluation/rescore_dabench_format.py --batch .cache/evaluation/dabench-new
uv run python evaluation/report_dabench.py --batch .cache/evaluation/dabench-new
```

`fetch_dabench.py --offline` verifies a previously downloaded cache without network. Model runs are opt-in and incur provider usage. The comparison never executes generated code on the host. Each trial starts a fresh process/session with the same frozen implementation, model, Docker image and budgets; only Skills ON/OFF changes. The ON intervention includes the catalogue, routing instructions, read tool and skill bodies, not a pure skill-text ablation. Default Skills behavior in the application is unchanged.

The fixed dataset revision has 257 public questions referencing 52 CSVs (the download also contains 16 unreferenced CSVs). Current strict input parsing accepts all 52 referenced datasets. Answer-blind deterministic sampling selects one question per CSV: four development questions and 24 disjoint heldout questions, eight per official difficulty. Heldout questions run three times per arm (144 trials). This is a stratified **public-validation subset**, not the official hidden test or a directly comparable leaderboard result. Public data may have appeared in model training. Three repeats are not three independent questions.

The Agent receives exactly the original question, constraints and answer-format fields, plus its selected CSV snapshot. Labels and scorer remain outside its Docker sandbox. Once heldout execution starts, do not change prompts, skills, source, tasks, evaluators, model options or input bytes. Resume validates the snapshot, skips completed trials and refuses to silently resubmit an already-started trial. A `STOP` file stops between trials; missing and failed trials remain visible.

Scoring separates:

- **Raw answer score**: the last main-Agent answer, with only its JSON transport envelope removed.
- **Blind reformatted score**: one additional bounded model call, given only that answer, the original question and requested format; no CSV, artifacts, labels or tools. The formatter may copy existing answers, not solve the problem. Its model is fixed to the same configured provider/model, unlike the original benchmark's GPT-3.5 formatter. Original explicit answers cannot be changed, conflicting formatted values are omitted, and numeric values absent from the answer are rejected. Translated yes/no conclusions are flagged for semantic review. These guards do not prove every field-value mapping or translation correct.
- **Supplemental deterministic extraction**: exact public-format keys found in checked metric artifacts, never replacing an explicit final value. This is not pooled with the primary reformatted answer score.
- **Delivery and end-to-end success**: validated core completion, actual saved files and unchanged source; end-to-end additionally requires all officially labelled answers correct. Correct numeric answers can coexist with a failed product delivery.

The official functions are loaded unchanged from their pinned source using an AST function allowlist, without the upstream CLI/import side effects. String equality and numeric absolute tolerance `<1e-6` are preserved. Nonempty missing-answer sentinels keep every expected question/subquestion in the denominator; the original scorer otherwise skips empty responses. Only labelled subquestions count, even when the public format lists more keys. ABQ, PASQ and UASQ are reported separately. Pre-execution question-ambiguity notes do not waive failures or change labels.

`benchmark_statistics.py` reports paired successes/regressions, missing trials, all-three-repeat task success and exploratory 95% intervals by resampling whole CSV datasets. Agent tokens/latency and formatter evaluation overhead are separate; unknown usage stays unknown. Local elapsed times are not isolated system-performance measurements. This benchmark does not directly grade professional report quality, RAG, conversation handling, safety, or the full modeling feature set. Keep product regression and independent artifact/verifier tests alongside it.

The first format-adapter development attempt is retained at `.cache/evaluation/skills-dabench-v1/` (8 real Agent trials); its lack of format normalization was diagnosed using development outputs only. Final heldout runs use the separate v2 contract. Those initial development results and any formatter-only checks are excluded from the 144-trial comparison.

Development also exposed two representation errors: literal copies of the public `@key[placeholder]` template were mistaken for actual answers, and the official numeric parser does not accept Unicode minus signs. `rescore_dabench_format.py` applies the same explicit correction to **all** saved samples, replaying only each existing formatter response through the frozen guard. It performs no new inference, computation or rounding, preserves wrong actual answers and every original score, and writes a separate versioned contract/report under `rescored-format-v3/`. Pass `--rescore <batch>/rescored-format-v3/report.json` to the report exporter to show both versions; their token counts must not be added together.

The initial heldout batch `.cache/evaluation/skills-dabench-v2/` stopped at 38/144 following multi-hour clock jumps, model connection failures and a Docker preflight timeout. Its records remain intact. The complete comparison is collected separately under `.cache/evaluation/skills-dabench-v4/`, with byte-identical source, inputs and evaluation contract; use the first complete uninterrupted batch, never select the best-scoring batch. The local supervisor holds a temporary Windows system-awake request and stops between trials after transport failure. `--interrupted-batch <path>` discloses abandoned-batch usage separately instead of hiding it or pooling it into the Skills estimate.

CSV-name separation is not data-family separation. The selected 24 CSV paths contain 23 distinct byte contents; three heldout files and one development file belong to the Titanic family. The Agent/prompt was not tuned on development outputs, but this still limits independence. `evaluation/results/skills-dabench-data-audit.json` documents this source-only audit. Pass `--data-audit` to the report exporter for byte-content/known-family clustering sensitivity intervals alongside the original CSV-cluster interval. No selected questions or failures are removed, and the family audit is not exhaustive.

## Token-efficiency development comparison

`token_efficiency.py` compares frozen baseline/candidate packages with the same model, image and budgets. Four already-known public development tasks cover grouped statistics, daily totals, actual plotting and mixed-feature classification. Two repeats per version give 16 trials. Skills OFF and ON use separate immutable output directories; they are not pooled into an external-quality score.

Extract the baseline without changing your working checkout, then prepare a new batch:

```powershell
New-Item -ItemType Directory -Force .cache | Out-Null
git archive --format=zip --output=.cache/token-baseline.zip 4c3accf19ba5ea0df71cfc027d91a4e0ddc1b9eb src
uv run python -m zipfile -e .cache/token-baseline.zip .cache/token-baseline
uv run python evaluation/token_efficiency.py --prepare --output .cache/evaluation/token-new-off --baseline-source .cache/token-baseline/src --candidate-source src
uv run python evaluation/token_efficiency.py --run --output .cache/evaluation/token-new-off
uv run python evaluation/token_efficiency.py --report --output .cache/evaluation/token-new-off
```

Prepare a different directory with `--skills` for ON. Preparation/reporting do not call models; `--run` uses configured credentials and real local Docker. The first run freezes model/image/runtime identity. Started trials cannot silently be resubmitted; all failures, save errors and missing runs remain visible. Sources/evaluators may not change mid-batch. The independent oracle checks exported full tables, metrics, plotting data and sample/label/prediction evidence, not completion text.

Report input/output and actual cache receipts separately. Baseline logs without cache fields are unknown, not zero. Logical request bytes are not exact provider-token attribution. Local elapsed time is not a speed benchmark. The two-repeat development set is too small and familiar to establish general or statistically significant gains. Report all iterations, including rejected candidates, and use a new untouched validation set for broader claims. The development/adoption plan is in [Token efficiency](../docs/token-efficiency.md).
