"""Frozen, paired development-set token evaluation (no model calls at prepare/report).

Prepare once, run once or resume unstarted trials, then report. Default: four public
development regressions x two repeats x two source variants = 16 paid trials.
Skills is fixed for both variants; run a separate output directory for Skills ON.
Raw receipts remain in the ignored output directory. Only --report --publish exports
allowlisted aggregates to evaluation/results/token-efficiency-{off,on}.{json,md}.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from measurements import measurements
from run import freeze_source
from skills_pilot import public_suite, check_outputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / '.cache/evaluation/token-efficiency-v1'
DEFAULT_BASELINE = DEFAULT_OUTPUT / 'baseline/implementation/src'
DEFAULT_CASES = ('S1', 'S2', 'S5', 'S7')
VARIANTS = ('baseline', 'candidate')
MODEL_OPTIONS = dict(temperature=0, main_max_tokens=4096, analysis_max_tokens=12288, max_retries=0)
TRIAL_TIMEOUT = 900
BOUNDARY = ('Four public development regressions, not an untouched holdout or external benchmark. '
            'Two repeats are correlated; no statistical significance or universal improvement claimed. '
            'Independent output oracles do not prove every narrative claim or absence of training leakage.')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable protocol or contract changed; use a new output directory')
    else:
        write_json(path, value)


def evaluator_hashes():
    return {name: digest(Path(__file__).with_name(name).read_bytes())
            for name in ('token_efficiency.py', 'run.py', 'skills_pilot.py', 'measurements.py')}


def source_identity(output, variant):
    directory = Path(output) / variant / 'implementation'
    manifest = read_json(directory / 'manifest.json')
    source = directory / 'src'
    actual = {p.relative_to(source).as_posix(): digest(p.read_bytes()) for p in source.rglob('*')
              if p.is_file() and p.suffix in {'.py', '.sql', '.md', '.json', '.txt'}}
    if not manifest or actual != manifest:
        raise ValueError('Implementation snapshot changed')
    return digest(json_bytes(manifest))


def trial_schedule(cases, repetitions=2):
    for repeat in range(1, repetitions + 1):
        for index, case in enumerate(cases):
            order = VARIANTS if (repeat + index) % 2 else tuple(reversed(VARIANTS))
            for variant in order:
                yield case['id'], repeat, variant


def prepare(output, baseline_source=None, candidate_source=None, *, cases=None, repetitions=None, skills=None):
    """Freeze source and selected public inputs before any online request is possible."""
    output = Path(output).resolve()
    path = output / 'protocol.json'
    if path.exists():
        protocol = verify_prepared(output)
        if ((cases is not None and list(cases) != protocol['cases']) or
                (repetitions is not None and repetitions != protocol['repetitions']) or
                (skills is not None and skills != protocol['skills_enabled'])):
            raise ValueError('Prepared case selection, repetitions or Skills changed; use a new output directory')
        return output
    selected = list(DEFAULT_CASES if cases is None else cases)
    repetitions = 2 if repetitions is None else repetitions
    if not selected or len(selected) != len(set(selected)) or any(c not in DEFAULT_CASES for c in selected):
        raise ValueError('Select unique public development cases from S1,S2,S5,S7')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('Repetitions must be a positive integer')
    suite, inputs = public_suite()
    lookup = {case['id']: case for case in suite['cases']}
    chosen = [lookup[c] for c in selected]
    suite = dict(version=1, name='token-efficiency-development', cases=chosen,
                 repetitions=repetitions, boundary=BOUNDARY)
    contents = {'suite.json': json_bytes(suite),
                **{'inputs/' + name: inputs[name] for name in sorted({c['file'] for c in chosen})}}
    manifest = {name: digest(raw) for name, raw in contents.items()}
    for name, raw in contents.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('Task/input snapshot changed; use a new output directory')
        target.write_bytes(raw)
    immutable_json(output / 'inputs-manifest.json', manifest)
    for variant, source in [('baseline', baseline_source or DEFAULT_BASELINE),
                            ('candidate', candidate_source or ROOT / 'src')]:
        freeze_source(Path(source), output / variant)
    protocol = dict(version=1, cases=selected, repetitions=repetitions, skills_enabled=bool(skills),
                    planned_trials=len(selected) * repetitions * 2,
                    model_options=MODEL_OPTIONS, trial_timeout_seconds=TRIAL_TIMEOUT,
                    suite_sha256=manifest['suite.json'], inputs_sha256=digest(json_bytes(manifest)),
                    implementations={v: source_identity(output, v) for v in VARIANTS},
                    evaluators=evaluator_hashes(),
                    schedule=[dict(case=c, repeat=r, variant=v) for c, r, v in trial_schedule(chosen, repetitions)],
                    boundary=BOUNDARY)
    immutable_json(path, protocol)
    (output / 'protocol.sha256').write_text(digest(path.read_bytes()) + '\n', encoding='ascii')
    return output


def verify_prepared(output):
    output = Path(output)
    protocol = read_json(output / 'protocol.json')
    if digest((output / 'protocol.json').read_bytes()) != (output / 'protocol.sha256').read_text().strip():
        raise ValueError('Prepared protocol snapshot changed')
    if evaluator_hashes() != protocol['evaluators']:
        raise ValueError('Evaluator changed after preparation')
    manifest = read_json(output / 'inputs-manifest.json')
    if digest(json_bytes(manifest)) != protocol['inputs_sha256']:
        raise ValueError('Input manifest changed')
    for name, sha in manifest.items():
        if digest((output / name).read_bytes()) != sha:
            raise ValueError('Task/input snapshot changed')
    suite = read_json(output / 'suite.json')
    schedule = [dict(case=c, repeat=r, variant=v)
                for c, r, v in trial_schedule(suite['cases'], protocol['repetitions'])]
    if (digest((output / 'suite.json').read_bytes()) != protocol['suite_sha256'] or
            [c['id'] for c in suite['cases']] != protocol['cases'] or
            suite['repetitions'] != protocol['repetitions'] or
            protocol['model_options'] != MODEL_OPTIONS or protocol['trial_timeout_seconds'] != TRIAL_TIMEOUT or
            schedule != protocol['schedule'] or len(schedule) != protocol['planned_trials']):
        raise ValueError('Prepared protocol changed')
    if {v: source_identity(output, v) for v in VARIANTS} != protocol['implementations']:
        raise ValueError('Implementation identity changed')
    return protocol


def runtime_identity():
    packages = ('deepagents', 'langchain', 'langchain-core', 'langchain-openai', 'langgraph', 'openai', 'pydantic')
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=platform.python_version(), packages=versions)


def freeze_contract(output, model, image):
    output = Path(output)
    verify_prepared(output)
    contract = dict(version=1, protocol_sha256=digest((output / 'protocol.json').read_bytes()),
                    model=model, image=image, runtime=runtime_identity())
    immutable_json(output / 'contract.json', contract)
    return contract


def verify_contract(output):
    output = Path(output)
    protocol = verify_prepared(output)
    contract = read_json(output / 'contract.json')
    if (contract['protocol_sha256'] != digest((output / 'protocol.json').read_bytes()) or
            contract['runtime'] != runtime_identity()):
        raise ValueError('Runtime or protocol differs from frozen contract')
    return protocol, contract


def owner_id(output, case, repeat, variant):
    return f'{case}-{repeat}-{variant}-{digest(str(Path(output).resolve()).encode())[:16]}'


def identity(case, repeat, variant, contract_sha):
    return dict(case=case, repeat=repeat, variant=variant, contract_sha256=contract_sha)


def claim_trial(directory, case, repeat, variant, contract_sha):
    """An interrupted or failed paid attempt is never silently submitted again."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = identity(case, repeat, variant, contract_sha)
    marker, summary = directory / 'attempt.json', directory / 'summary.json'
    if marker.exists() and read_json(marker) != expected:
        raise ValueError('Trial attempt identity changed')
    if summary.exists():
        if any(read_json(summary).get(k) != value for k, value in expected.items()):
            raise ValueError('Trial summary identity changed')
        if not marker.exists():
            raise ValueError('Summary has no attempt marker')
        return False
    if marker.exists():
        raise ValueError('Previously started trial has no summary; inspect and record interruption before resuming')
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(expected, stream, sort_keys=True)
    return True


def receipt_measurements(report):
    """Known token totals are lower bounds. Unsupported cache accounting stays unknown."""
    usage = report.get('token_usage') or {}
    main = usage.get('main', [])
    child = usage.get('analysis', [])
    calls = [*main, *child]
    def valid_usage(call):
        return (isinstance(call, dict) and isinstance(call.get('usage'), dict) and
                all(type(call['usage'].get(k)) is int and call['usage'][k] >= 0
                    for k in ('input_tokens', 'output_tokens', 'total_tokens')))
    known = [c['usage'] for c in calls if valid_usage(c)]
    main_calls, child_calls = report.get('model_calls'), report.get('analysis_model_calls')
    count_known = all(type(c) is int and c >= 0 for c in (main_calls, child_calls))
    expected = main_calls + child_calls if count_known else None
    clean = {**report, 'model_calls': main_calls if count_known else 0,
             'analysis_model_calls': child_calls if count_known else 0,
             'token_usage': {role: [{'usage': c['usage'] if valid_usage(c) else None} for c in entries]
                             for role, entries in [('main', main), ('analysis', child)]}}
    result = measurements(clean)
    # Shared measurements is reused; tighten accounting so missing/invalid receipts
    # and receipts beyond the declared call count cannot make usage look complete.
    exact = count_known and len(main) == main_calls and len(child) == child_calls and len(known) == expected
    cache = [(c.get('cache_usage') or {}).get('read_tokens') for c in calls if isinstance(c, dict)]
    cache_known = [n for n in cache if type(n) is int and n >= 0]
    cache_complete = count_known and len(cache_known) == expected and len(calls) == expected
    result.update(model_calls=main_calls, analysis_model_calls=child_calls,
                  known_input_tokens=sum(u['input_tokens'] for u in known) if known or exact else None,
                  known_output_tokens=sum(u['output_tokens'] for u in known) if known or exact else None,
                  known_total_tokens=sum(u['total_tokens'] for u in known) if known or exact else None,
                  usage_receipt_calls=len(known), unknown_usage_calls=max(0, expected - len(known)) if count_known else None,
                  usage_complete=exact, usage_receipt_mismatch=len(main) != main_calls or len(child) != child_calls,
                  cached_input_tokens=sum(cache_known) if cache_complete else None,
                  known_cached_input_tokens=sum(cache_known) if cache_known or cache_complete else None,
                  cache_unknown_calls=max(0, expected - len(cache_known)) if count_known else None,
                  cache_usage_complete=cache_complete)
    result['context_bytes'] = {}
    for key in ('system_bytes', 'message_bytes', 'tool_schema_bytes'):
        values = [(c.get('context_bytes') or {}).get(key) for c in calls if isinstance(c, dict)]
        numeric = [n for n in values if type(n) is int and n >= 0]
        result['context_bytes'][key] = dict(known_sum=sum(numeric) if numeric else None,
                                          measured_calls=len(numeric), expected_calls=expected)
    return result


def cleanup_owned(output, contract, case, repeat, variant):
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig, docker_executable
    executor = DockerExecutor(ExecutionConfig(image=contract['image']['tag']))
    executor.preflight()
    if executor.image_id != contract['image']['id']:
        raise ValueError('Sandbox image changed before cleanup')
    command = executor._docker_cli(docker_executable())
    owned = subprocess.run(command + ['ps', '-aq', '--filter', 'label=labweaver.app=token-efficiency',
                                      '--filter', f'label=labweaver.run={owner_id(output, case, repeat, variant)}'],
                           capture_output=True, timeout=15, env=executor._client_env(), check=True)
    for cid in owned.stdout.decode().split():
        subprocess.run(command + ['rm', '-f', cid], capture_output=True, timeout=15,
                       env=executor._client_env(), check=True)


def run_worker(args):
    output = Path(args.output).resolve()
    protocol, contract = verify_contract(output)
    expected_sha = digest((output / 'contract.json').read_bytes())
    key = dict(case=args.case, repeat=args.repeat, variant=args.variant)
    if key not in protocol['schedule']:
        raise ValueError('Worker is outside the frozen trial schedule')
    destination = output / 'trials' / f'{args.case}-{args.repeat}-{args.variant}'
    expected = identity(args.case, args.repeat, args.variant, expected_sha)
    if read_json(destination / 'attempt.json') != expected or (destination / 'summary.json').exists():
        raise ValueError('Worker requires an uncompleted claimed trial')
    # Direct duplicate --worker invocations must also fail before a model request.
    with (destination / 'worker-started.json').open('x', encoding='utf-8') as stream:
        json.dump(expected, stream, sort_keys=True)
    start = time.monotonic()
    source = output / args.variant / 'implementation/src'
    sys.path.insert(0, str(source))
    os.chdir(ROOT)
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    summary = dict(**expected, skills_enabled=protocol['skills_enabled'], attained=False, status='error',
                   saved=False, errors=[], usage_complete=False, input_hash_unchanged=None)
    session = None
    source_path = None
    stage = 'preflight'
    try:
        from labweaver.agent import create_session
        from labweaver.config import load_config, create_model
        from labweaver.run_config import load_intake_config
        from labweaver.runtime.execution import DockerExecutor, ExecutionConfig
        config = load_config(load_intake_config().env_file)
        if dict(model_id=config.model_id, base_url=config.base_url, timeout=config.timeout) != contract['model']:
            raise ValueError('Model settings differ from frozen contract')
        execution_config = ExecutionConfig(image=contract['image']['tag'])
        probe = DockerExecutor(execution_config)
        probe.preflight()
        if probe.image_id != contract['image']['id']:
            raise ValueError('Sandbox image differs from frozen contract')
        if (output / 'STOP').exists():
            summary['errors'].append('stop_requested')
            return
        case = next(c for c in read_json(output / 'suite.json')['cases'] if c['id'] == args.case)
        source_path = output / 'inputs' / case['file']
        summary['input_sha256'] = digest(source_path.read_bytes())
        # Only task text and its CSV enter the Agent; the independent oracle stays on host.
        session = create_session(source_path, create_model(config, max_tokens=MODEL_OPTIONS['main_max_tokens']),
                                 analysis_model=create_model(config, max_tokens=MODEL_OPTIONS['analysis_max_tokens']),
                                 skills_enabled=protocol['skills_enabled'], execution_config=execution_config)
        session.executor.set_owner('token-efficiency', owner_id(output, args.case, args.repeat, args.variant))
        def progress(event):
            public = {k: event[k] for k in ('type', 'role', 'call', 'attempt', 'status', 'elapsed_seconds', 'execution_id') if k in event}
            with (destination / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(public, ensure_ascii=False) + '\n')
        stage = 'invoke'
        report = session.invoke(case['task'], on_event=progress)
        write_json(destination / 'report.json', report)
        summary.update(receipt_measurements(report))
        summary.update(status=report.get('status', 'error'), error=report.get('error'),
                       skill_load_count=len((report.get('skills') or {}).get('loads', [])))
        # Save receipts even when the independent oracle rejects the delivery.
        stage = 'save'
        session.save(destination / 'session')
        summary['saved'] = True
        stage = 'oracle'
        summary['errors'] = check_outputs(report, session.artifact_assets, case, source_path)
        summary['input_hash_unchanged'] = digest(source_path.read_bytes()) == summary['input_sha256']
        if not summary['input_hash_unchanged']:
            summary['errors'].append('source_input_changed')
        summary['attained'] = report.get('status') == 'completed' and not summary['errors']
    except Exception as exc:
        summary.update(status='error', attained=False)
        summary['errors'].append(stage + '_' + type(exc).__name__)
    finally:
        try:
            if session is not None:
                session.executor.cancel()
        except Exception as exc:
            summary.update(attained=False, status='error')
            summary['errors'].append('cleanup_' + type(exc).__name__)
        if source_path is not None and 'input_sha256' in summary:
            try:
                summary['input_hash_unchanged'] = digest(source_path.read_bytes()) == summary['input_sha256']
            except OSError:
                summary['input_hash_unchanged'] = False
            if not summary['input_hash_unchanged']:
                summary['attained'] = False
                if 'source_input_changed' not in summary['errors']:
                    summary['errors'].append('source_input_changed')
        summary['elapsed_seconds'] = round(time.monotonic() - start, 3)
        write_json(destination / 'summary.json', summary)


def run_batch(args):
    output = Path(args.output).resolve()
    protocol = verify_prepared(output)
    # Config and Docker checks make no model requests. Each paid trial uses a new worker.
    sys.path.insert(0, str(output / 'candidate/implementation/src'))
    from labweaver.config import load_config
    from labweaver.run_config import load_intake_config
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig
    config = load_config(load_intake_config().env_file)
    executor = DockerExecutor(ExecutionConfig(image=args.image))
    executor.preflight()
    contract = freeze_contract(output, dict(model_id=config.model_id, base_url=config.base_url, timeout=config.timeout),
                               dict(tag=args.image, id=executor.image_id))
    verify_contract(output)
    expected_sha = digest((output / 'contract.json').read_bytes())
    for trial in protocol['schedule']:
        if (output / 'STOP').exists():
            print('Stopped between trials; all recorded attempts retained.', flush=True)
            return
        case, repeat, variant = trial['case'], trial['repeat'], trial['variant']
        directory = output / 'trials' / f'{case}-{repeat}-{variant}'
        summary = directory / 'summary.json'
        if not claim_trial(directory, case, repeat, variant, expected_sha):
            continue
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(output),
                   '--case', case, '--repeat', str(repeat), '--variant', variant]
        print(f'{case} repeat={repeat} variant={variant} skills={protocol["skills_enabled"]}', flush=True)
        failure = None
        with (directory / 'worker.log').open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=log, timeout=TRIAL_TIMEOUT, check=False)
                if process.returncode:
                    failure = 'worker_nonzero_exit'
            except subprocess.TimeoutExpired:
                failure = 'worker_timeout'
            except OSError:
                failure = 'worker_start_failed'
        if failure or not summary.exists():
            record = read_json(summary) if summary.exists() else identity(case, repeat, variant, expected_sha)
            record.update(attained=False, saved=record.get('saved', False), status='error',
                          errors=[*record.get('errors', []), failure or 'worker_missing_summary'])
            write_json(summary, record)
            cleanup_owned(output, contract, case, repeat, variant)
        print(json.dumps({k: read_json(summary).get(k) for k in
                          ('case', 'repeat', 'variant', 'attained', 'errors', 'known_total_tokens', 'elapsed_seconds')}), flush=True)


def aggregate(output):
    """Allowlisted aggregates only; no raw tasks, responses, endpoints or local paths."""
    output = Path(output)
    protocol = verify_prepared(output)
    contract = read_json(output / 'contract.json') if (output / 'contract.json').exists() else None
    contract_sha = digest((output / 'contract.json').read_bytes()) if contract else None
    if contract and contract['protocol_sha256'] != digest((output / 'protocol.json').read_bytes()):
        raise ValueError('Contract references a different protocol')
    expected = {(t['case'], t['repeat'], t['variant']) for t in protocol['schedule']}
    rows = {}
    for path in output.glob('trials/*/summary.json'):
        row = read_json(path)
        key = row.get('case'), row.get('repeat'), row.get('variant')
        if (key not in expected or key in rows or row.get('contract_sha256') != contract_sha or
                contract_sha is None or path.parent.name != f'{key[0]}-{key[1]}-{key[2]}'):
            raise ValueError('Duplicate, unknown, or mixed-contract trial')
        if read_json(path.parent / 'attempt.json') != identity(*key, contract_sha):
            raise ValueError('Trial has no matching immutable attempt')
        if row.get('attained') and (row.get('status') != 'completed' or row.get('saved') is not True or
                                    row.get('input_hash_unchanged') is not True or row.get('errors')):
            raise ValueError('Successful trial is missing delivery/save/input evidence')
        rows[key] = row
    variants = {}
    metrics = ('known_input_tokens', 'known_output_tokens', 'known_total_tokens', 'model_calls', 'analysis_model_calls')
    planned = len(expected) // 2
    for variant in VARIANTS:
        trials = [r for key, r in rows.items() if key[2] == variant]
        known = {k: [r[k] for r in trials if type(r.get(k)) is int and r[k] >= 0] for k in metrics}
        cache_known = [r['known_cached_input_tokens'] for r in trials
                       if type(r.get('known_cached_input_tokens')) is int]
        cache_complete = len(trials) == planned and all(r.get('cache_usage_complete') is True for r in trials)
        elapsed = [r['elapsed_seconds'] for r in trials if type(r.get('elapsed_seconds')) in (int, float)]
        variants[variant] = dict(planned=planned, recorded=len(trials), missing=planned - len(trials),
                                 attained=sum(bool(r.get('attained')) for r in trials),
                                 success_rate_planned=sum(bool(r.get('attained')) for r in trials) / planned,
                                 failed=len(trials) - sum(bool(r.get('attained')) for r in trials),
                                 usage_complete_trials=sum(r.get('usage_complete') is True for r in trials),
                                 usage_unmeasured_trials=sum(type(r.get('unknown_usage_calls')) is not int for r in trials),
                                 unknown_usage_calls=sum(r['unknown_usage_calls'] for r in trials if type(r.get('unknown_usage_calls')) is int),
                                 cached_input_tokens=sum(cache_known) if cache_complete else None,
                                 known_cached_input_tokens=sum(cache_known) if cache_known else None,
                                 cache_usage_complete=cache_complete,
                                 median_seconds=statistics.median(elapsed) if elapsed else None,
                                 first_attempt_attained=sum(bool(r.get('attained')) and r.get('no_internal_repair') is True for r in trials),
                                 **{k: sum(values) if values else None for k, values in known.items()})
        variants[variant]['context_bytes'] = {}
        for key in ('system_bytes', 'message_bytes', 'tool_schema_bytes'):
            values = [(r.get('context_bytes') or {}).get(key, {}) for r in trials]
            measured = [v['known_sum'] for v in values if type(v.get('known_sum')) is int]
            variants[variant]['context_bytes'][key] = dict(
                known_sum=sum(measured) if measured else None,
                measured_calls=sum(v.get('measured_calls', 0) for v in values),
                measurement='logical_utf8_bytes_not_provider_tokens')
    paired = dict(both_passed=0, both_failed=0, candidate_only_passed=0, baseline_only_passed=0,
                  incomplete_pairs=0, fully_metered_pairs=0, both_passed_fully_metered_pairs=0)
    paired_tokens = {'all': {v: 0 for v in VARIANTS}, 'both_passed': {v: 0 for v in VARIANTS}}
    for case in protocol['cases']:
        for repeat in range(1, protocol['repetitions'] + 1):
            a, b = rows.get((case, repeat, 'baseline')), rows.get((case, repeat, 'candidate'))
            if a is None or b is None:
                paired['incomplete_pairs'] += 1
                continue
            passed = a.get('attained') and b.get('attained')
            paired['both_passed' if passed else 'candidate_only_passed' if b.get('attained') else
                   'baseline_only_passed' if a.get('attained') else 'both_failed'] += 1
            if all(r.get('usage_complete') is True and type(r.get('known_total_tokens')) is int for r in (a, b)):
                paired['fully_metered_pairs'] += 1
                for variant, row in zip(VARIANTS, (a, b)):
                    paired_tokens['all'][variant] += row['known_total_tokens']
                if passed:
                    paired['both_passed_fully_metered_pairs'] += 1
                    for variant, row in zip(VARIANTS, (a, b)):
                        paired_tokens['both_passed'][variant] += row['known_total_tokens']
    all_pairs = planned
    comparable = paired['fully_metered_pairs'] == all_pairs
    baseline_tokens = paired_tokens['all']['baseline']
    token_reduction = (1 - paired_tokens['all']['candidate'] / baseline_tokens) if comparable and baseline_tokens else None
    both_tokens = paired_tokens['both_passed']
    by_case = {case: {v: dict(planned=protocol['repetitions'],
                              recorded=sum(k[0] == case and k[2] == v for k in rows),
                              attained=sum(bool(r.get('attained')) for k, r in rows.items() if k[0] == case and k[2] == v))
                      for v in VARIANTS} for case in protocol['cases']}
    return dict(version=1, complete=len(rows) == len(expected), protocol_sha256=digest((output / 'protocol.json').read_bytes()),
                contract_sha256=contract_sha, source_sha256=protocol['implementations'],
                cases=protocol['cases'], repetitions=protocol['repetitions'], skills_enabled=protocol['skills_enabled'],
                model_options=protocol['model_options'], trial_timeout_seconds=TRIAL_TIMEOUT,
                planned_trials=protocol['planned_trials'], variants=variants, paired=paired, by_case=by_case,
                fully_metered_complete_comparison=comparable, all_planned_token_reduction_fraction=token_reduction,
                both_passed_token_reduction_fraction=(1 - both_tokens['candidate'] / both_tokens['baseline']) if both_tokens['baseline'] else None,
                both_passed_paired_tokens=both_tokens if paired['both_passed_fully_metered_pairs'] else None,
                boundary=BOUNDARY,
                accounting='All planned trials remain in success denominators. Known token totals are lower bounds when usage is incomplete; missing usage and cache hits are not zero. Both-passed token ratios describe a selected subset.')


def save_report(output, publish=False):
    output = Path(output)
    data = aggregate(output)
    lines = ['# Token 效率开发集对照', '',
             f"固定 {len(data['cases'])} 个公开开发回归案例，每题每版本 {data['repetitions']} 次，共 {data['planned_trials']} 次。Skills：{'开启' if data['skills_enabled'] else '关闭'}。完整批次：{'是' if data['complete'] else '否'}。", '',
             '| 指标 | Baseline | Candidate |', '|---|---:|---:|']
    for key, label in [('planned', '计划运行'), ('recorded', '已记录'), ('missing', '缺失'), ('attained', '任务达成'),
                       ('success_rate_planned', '达成率（分母为计划运行）'), ('failed', '已记录失败'),
                       ('known_input_tokens', '已知输入 token'), ('known_output_tokens', '已知输出 token'),
                       ('known_total_tokens', '已知总 token'), ('model_calls', '主模型调用'), ('analysis_model_calls', '子模型调用'),
                       ('unknown_usage_calls', '用量未知调用'), ('usage_unmeasured_trials', '未取得计量的运行'),
                       ('usage_complete_trials', '收据完整运行'), ('cached_input_tokens', '缓存输入 token'),
                       ('first_attempt_attained', '无需内部修正达成'), ('median_seconds', '耗时中位数（秒）')]:
        values = ['unknown' if data['variants'][v][key] is None else data['variants'][v][key] for v in VARIANTS]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    reduction = data['all_planned_token_reduction_fraction']
    lines += ['', f"完整计划的 token 降幅：{reduction:.2%}。" if reduction is not None else
              '完整计划的 token 降幅：unknown（尚有缺失试次或未知用量）。', '',
              '| 案例 | Baseline 达成/计划 | Candidate 达成/计划 |', '|---|---:|---:|']
    for case, arms in data['by_case'].items():
        lines.append(f"| {case} | {arms['baseline']['attained']}/{arms['baseline']['planned']} | {arms['candidate']['attained']}/{arms['candidate']['planned']} |")
    p = data['paired']
    lines += ['', f"配对：双方达成 {p['both_passed']}，仅 candidate 达成 {p['candidate_only_passed']}，仅 baseline 达成 {p['baseline_only_passed']}，双方失败 {p['both_failed']}，不完整 {p['incomplete_pairs']}。", '',
              '这些题是参与实现验证的公开开发集，不是未接触 holdout，也不等同于官方外部基准。两次重复相关，样本小，不能据此宣称统计显著或普遍提升。', '',
              '失败、超时和缺失始终保留在计划分母。已知 token 在用量不完整时只是下界；缓存收据不可用，不能当作零。原始付费试次不得覆盖重跑。双方达成子集的 token 比较存在选择偏差。', '']
    markdown = '\n'.join(lines)
    write_json(output / 'report.json', data)
    (output / 'report.md').write_text(markdown, encoding='utf-8')
    if publish:
        target = ROOT / 'evaluation/results'
        stem = 'token-efficiency-on' if data['skills_enabled'] else 'token-efficiency-off'
        write_json(target / (stem + '.json'), data)
        (target / (stem + '.md')).write_text(markdown, encoding='utf-8')
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare', 'run', 'report', 'worker'):
        actions.add_argument('--' + action, action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--candidate-source', type=Path)
    parser.add_argument('--cases', nargs='+', choices=DEFAULT_CASES)
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--skills', action='store_true', default=None)
    parser.add_argument('--image', default='labweaver-python:0.3.0')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--case', choices=DEFAULT_CASES)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if not args.prepare and any(x is not None for x in (args.baseline_source, args.candidate_source, args.cases, args.repetitions, args.skills)):
        parser.error('Source, case, repetition and Skills choices are only accepted by --prepare')
    if args.publish and not args.report:
        parser.error('--publish requires --report')
    if args.prepare:
        print(prepare(args.output, args.baseline_source, args.candidate_source,
                      cases=args.cases, repetitions=args.repetitions, skills=args.skills))
    elif args.run:
        run_batch(args)
    elif args.report:
        save_report(args.output, args.publish)
    else:
        if args.case is None or args.repeat is None or args.variant is None:
            parser.error('--worker requires --case --repeat --variant')
        run_worker(args)


if __name__ == '__main__':
    main()
