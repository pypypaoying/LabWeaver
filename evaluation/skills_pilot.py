"""Fixed public-synthetic Skills OFF/ON pilot; never executes generated code on host.

Prepare, run, then report with the same --output (default ignored .cache directory).
Each of eight cases runs three times per arm. Missing/failed runs stay in accounting.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / '.cache/evaluation/skills-pilot'
MODEL_OPTIONS = dict(temperature=0, main_max_tokens=4096, analysis_max_tokens=12288, max_retries=0)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\n').writerows(rows)
    return stream.getvalue().encode('utf-8')


def public_suite():
    """All task text and hand-checkable expected values fixed before the pilot."""
    grouped = [['district', 'parcel_count', 'approved'], ['Cedar', 3, 'yes'], ['Aspen', 2, 'yes'],
               ['Cedar', 5, 'no'], ['Birch', 4, 'yes'], ['Aspen', 7, 'yes'],
               ['Birch', 1, 'no'], ['Cedar', 0, 'yes'], ['Birch', 8, 'yes']]
    dates = [['observed_at', 'energy_kwh'], ['2026-04-01 12:00', 4], ['2026-04-01 08:00', 3],
             ['2026-04-04 09:00', 6], ['2026-04-02 16:00', 5], ['2026-04-04 18:00', 2]]
    cleaning = [['ticket', 'amount', 'unit_cost'], ['001', 2, 3], ['001', 2, 3], ['002', '', 4],
                ['003', 5, 2], ['004', 0, 9], ['005', -1, 2], ['006', '', 7]]
    weighted = [['price', 'volume'], [10, 2], [20, 1], [40, 1], ['', 3], [8, 0]]
    dispersion = [['reading'], *[[n] for n in [2, 4, 4, 4, 5, 5, 7, 9]], ['']]
    learning = [['sample_code', 'signal_strength', 'channel', 'decision']]
    for i in range(80):
        signal = ((i * 17) % 83) / 10
        channel = ['email', 'store', 'phone'][i % 3]
        label = 'pass' if signal + (0.8 if channel == 'store' else 0) > 4.3 else 'fail'
        learning.append([f'{i+1:04}', signal, channel, label])
    inputs = {'districts.csv': grouped, 'energy.csv': dates, 'tickets.csv': cleaning,
              'weights.csv': weighted, 'readings.csv': dispersion, 'decisions.csv': learning}
    expected_group = [['Birch', 12], ['Aspen', 9], ['Cedar', 3]]
    cases = [
        dict(id='S1', file='districts.csv', task='只保留 approved=yes，按 district 汇总 parcel_count。导出两列 district,total，按 total 降序排列，保留全部组。不要绘图。',
             oracle=dict(columns=['district', 'total'], table=expected_group)),
        dict(id='S2', file='energy.csv', task='解析 observed_at，按日汇总 energy_kwh。补齐2026-04-01至04-04每天，无记录日期填0，输出日期升序的 date,total_kwh 表；指标 JSON 保存 peak_kwh、min_kwh、difference，包含零值日期。不绘图。',
             oracle=dict(columns=['date', 'total_kwh'], table=[['2026-04-01', 7], ['2026-04-02', 5], ['2026-04-03', 0], ['2026-04-04', 8]], metric=dict(peak_kwh=8, min_kwh=0, difference=8))),
        dict(id='S3', file='tickets.csv', task='删除完全重复记录，保留首次；amount 空白填0，派生 value=amount*unit_cost。按原顺序导出 ticket,amount,value，ticket 保留前导零。指标 JSON 保存 before_rows、after_rows、removed_duplicates、filled_missing、total_value，报告处理变化。不绘图。',
             oracle=dict(columns=['ticket', 'amount', 'value'], table=[['001', 2, 6], ['002', 0, 0], ['003', 5, 10], ['004', 0, 0], ['005', -1, -2], ['006', 0, 0]],
                         metric=dict(before_rows=7, after_rows=6, removed_duplicates=1, filled_missing=2, total_value=14))),
        dict(id='S4', file='weights.csv', task='以 volume 为权重计算 price 的加权平均。排除任一字段空白的记录，权重0记录保留计数。指标 JSON 保存 weighted_mean、valid_rows、weight_sum，说明分母口径，不绘图。',
             oracle=dict(metric=dict(weighted_mean=20, valid_rows=4, weight_sum=4))),
        dict(id='S5', file='districts.csv', task='只保留 approved=yes，按 district 汇总 parcel_count，按总数降序。交付 district,total 两列表和一张柱状图；绘图数据使用相同两列、相同完整组与顺序，不额外分组。',
             oracle=dict(columns=['district', 'total'], table=expected_group, figure=True, figure_table=expected_group)),
        dict(id='S6', file='readings.csv', task='排除 reading 空白值，计算样本均值、样本方差(ddof=1)、样本标准差、均值标准误=样本标准差/sqrt(n)。JSON 指标固定名 n,mean,sample_variance,sample_std,standard_error，说明样本与总体口径区别。不绘图。',
             oracle=dict(metric=dict(n=8, mean=5, sample_variance=32/7, sample_std=math.sqrt(32/7), standard_error=math.sqrt(32/7)/math.sqrt(8)))),
        dict(id='S7', file='decisions.csv', task='用 signal_strength 和 channel 预测 decision，训练逻辑回归二分类模型。sample_code 仅作编号，排除特征；保留重复记录。seed=42，训练/测试80/20，按原始标签分层；Pipeline处理数值和类别，预处理只拟合训练集。比较训练集多数类基线，交付测试集逐行源记录ID/真实值/预测/基线预测、完整拆分元数据、分类指标、混淆矩阵图和专业报告。',
             oracle=dict(modeling='binary', target='c4', features=['c2', 'c3'], source_rows=80, test_rows=16, figure=True)),
        dict(id='S8', file='districts.csv', task='仅概述实际 CSV 的行列数、字段名称和缺失情况，不执行计算、不绘图。',
             oracle=dict(profile=[8, 3], no_execution=True)),
    ]
    return dict(version=1, name='skills-pilot', repetitions=3, cases=cases,
                boundary='Independent synthetic pilot, not universal accuracy; no changes to prior frozen tasks.'), {name: csv_bytes(rows) for name, rows in inputs.items()}


def prepare(output):
    """Immutable task/input snapshot; resume rejects any changed bytes."""
    output = Path(output).resolve()
    suite, inputs = public_suite()
    contents = {'suite.json': (json.dumps(suite, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode(),
                **{'inputs/' + name: raw for name, raw in inputs.items()}}
    expected = {name: digest(raw) for name, raw in contents.items()}
    manifest = output/'inputs-manifest.json'
    if manifest.exists():
        if read_json(manifest) != expected:
            raise ValueError('Pilot task definitions changed; use a new output directory')
        for name, sha in expected.items():
            if digest((output/name).read_bytes()) != sha:
                raise ValueError('Pilot input snapshot changed; use a new output directory')
    else:
        for name, raw in contents.items():
            path = output/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        write_json(manifest, expected)
    return output


def trial_schedule(cases, repetitions=3):
    """Balanced treatment order; each trial starts a fresh process/session."""
    for repeat in range(1, repetitions + 1):
        for index, case in enumerate(cases):
            arms = ['off', 'on'] if (repeat + index) % 2 else ['on', 'off']
            for arm in arms:
                yield case['id'], repeat, arm


def owner_id(output, case, repeat, arm):
    """Separate two pilot batches even if their case/repeat names coincide."""
    batch = digest(str(Path(output).resolve()).encode())[:16]
    return f'{case}-{repeat}-{arm}-{batch}'


def freeze_contract(output, source, model, image):
    from run import freeze_source
    frozen = freeze_source(source, output)
    manifest = read_json(output/'implementation/manifest.json')
    contract = dict(version=1, implementation_sha256=digest(json.dumps(manifest, sort_keys=True).encode()),
                    suite_sha256=digest((output/'suite.json').read_bytes()),
                    inputs_sha256=digest(json.dumps(read_json(output/'inputs-manifest.json'), sort_keys=True).encode()),
                    evaluator_sha256=digest(Path(__file__).read_bytes()),
                    common_evaluator_sha256=digest(Path(__file__).with_name('run.py').read_bytes()),
                    measurements_sha256=digest(Path(__file__).with_name('measurements.py').read_bytes()),
                    model=model, model_options=MODEL_OPTIONS, image=image)
    path = output/'contract.json'
    if path.exists() and read_json(path) != contract:
        raise ValueError('Pilot model, budget, evaluator or snapshot changed; use a new output directory')
    write_json(path, contract)
    return frozen, contract


def verify_worker_contract(output, source):
    contract = read_json(output/'contract.json')
    if digest(Path(__file__).read_bytes()) != contract['evaluator_sha256'] or digest(Path(__file__).with_name('run.py').read_bytes()) != contract['common_evaluator_sha256'] or digest(Path(__file__).with_name('measurements.py').read_bytes()) != contract['measurements_sha256']:
        raise ValueError('Pilot evaluator changed during batch')
    if digest((output/'suite.json').read_bytes()) != contract['suite_sha256']:
        raise ValueError('Pilot task snapshot changed during batch')
    for name, sha in read_json(output/'inputs-manifest.json').items():
        if digest((output/name).read_bytes()) != sha:
            raise ValueError('Pilot input snapshot changed during batch')
    manifest = read_json(output/'implementation/manifest.json')
    if digest(json.dumps(manifest, sort_keys=True).encode()) != contract['implementation_sha256']:
        raise ValueError('Pilot implementation manifest changed during batch')
    for name, sha in manifest.items():
        if digest((Path(source)/name).read_bytes()) != sha:
            raise ValueError('Pilot implementation snapshot changed during batch')
    return contract


def equal(value, expected):
    if isinstance(expected, (int, float)):
        try:
            return math.isfinite(float(value)) and math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=1e-8)
        except (ValueError, TypeError):
            return False
    return value == expected


def matrix_and_f1(actual, predicted, labels):
    matrix = [[sum(a == x and p == y for a, p in zip(actual, predicted)) for y in labels] for x in labels]
    f1s = []
    for i in range(len(labels)):
        denom = sum(matrix[i]) + sum(row[i] for row in matrix)
        if denom:
            f1s.append(2 * matrix[i][i] / denom)
    return matrix, sum(f1s)/len(f1s) if f1s else None


def check_modeling(report, assets, oracle, source_path):
    """Independent source/ID/metric check, not an import of the production verifier."""
    errors = []
    model = (report.get('task_spec') or {}).get('modeling') or {}
    required = dict(task_type='classification', target=oracle['target'], features=oracle['features'],
                    split='stratified', seed=42, test_size=.2, duplicate_policy='keep')
    if any(model.get(k) != v for k, v in required.items()):
        return ['modeling_task_contract']
    deliveries = {a['metadata']['deliverable_id']: a for a in assets.values()}
    try:
        prediction = deliveries[model['prediction_delivery']]
        predictions = list(csv.DictReader(io.StringIO(prediction['files']['table.csv'].decode('utf-8-sig'))))
        meta = deliveries[model['metadata_delivery']]['metadata']['value']
        metrics = deliveries[model['metrics_delivery']]['metadata']['value']
        source = list(csv.reader(io.StringIO(Path(source_path).read_text(encoding='utf-8'))))[1:]
        truth = {f'r{i:06}': row[3] for i, row in enumerate(source, 1)}
        train, test = meta['train_ids'], meta['test_ids']
        ids = [p['source_record_id'] for p in predictions]
        if len(source) != oracle['source_rows'] or len(predictions) != oracle['test_rows'] or len(train) != 64 or len(test) != 16:
            errors.append('modeling_sample_counts')
        if len(ids) != len(set(ids)) or set(ids) != set(test) or set(train) & set(test) or set(train + test) != set(truth) or meta['excluded_ids']:
            errors.append('modeling_sample_identity')
        training = [truth[rid] for rid in train]
        baseline = min(set(training), key=lambda label: (-training.count(label), label))
        actual = [truth[p['source_record_id']] for p in predictions]
        if any(p['y_true'] != value for p, value in zip(predictions, actual)):
            errors.append('modeling_source_truth')
        if any(p['baseline_pred'] != baseline for p in predictions):
            errors.append('modeling_train_only_baseline')
        labels = sorted(set(truth.values()))
        for key, predicted in [('model', [p['y_pred'] for p in predictions]), ('baseline', [baseline] * len(predictions))]:
            if not predicted or not set(predicted) <= set(labels):
                errors.append('modeling_prediction_domain')
                continue
            accuracy = sum(a == b for a, b in zip(actual, predicted)) / len(actual)
            matrix, f1 = matrix_and_f1(actual, predicted, labels)
            values = metrics[key]
            if not math.isclose(values['accuracy'], accuracy, rel_tol=0, abs_tol=1e-8) or not equal(values['macro_f1'], f1) or values['labels'] != labels or values['confusion_matrix'] != matrix:
                errors.append('independent_classification_metrics:' + key)
        if not report.get('verification') or any(c['status'] == 'failed' for group in report['verification'] for c in group['results']):
            errors.append('production_verification_missing_or_failed')
    except (ValueError, KeyError, TypeError, IndexError, OSError, ZeroDivisionError):
        errors.append('invalid_modeling_receipts')
    return sorted(set(errors))


def check_outputs(report, assets, case, source_path):
    from run import check
    oracle = case['oracle']
    generic = {key: value for key, value in oracle.items() if key not in {'modeling', 'figure_table'}}
    errors = check(report, assets, generic)
    if oracle.get('modeling'):
        errors.extend(check_modeling(report, assets, oracle, source_path))
    if 'figure_table' in oracle:
        expected = oracle['figure_table']
        matched = False
        for asset in assets.values():
            if asset['metadata']['kind'] != 'figure':
                continue
            rows = list(csv.reader(io.StringIO(asset['files']['data.csv'].decode('utf-8-sig'))))
            if rows and rows[0] == oracle['columns'] and len(rows)-1 == len(expected):
                matched |= all(len(r) == len(e) and all(equal(x, y) for x, y in zip(r, e)) for r, e in zip(rows[1:], expected))
        if not matched:
            errors.append('figure_data_not_full_expected_table')
    return sorted(set(errors))


def run_worker(args):
    output = Path(args.output).resolve()
    source = Path(args.source).resolve()
    contract = verify_worker_contract(output, source)
    sys.path.insert(0, str(source))
    os.chdir(ROOT)
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    from labweaver.agent import create_session
    from labweaver.config import load_config, create_model
    from labweaver.run_config import load_intake_config
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig
    from measurements import measurements
    case = next(c for c in read_json(output/'suite.json')['cases'] if c['id'] == args.case)
    destination = output/'trials'/f'{args.case}-{args.repeat}-{args.arm}'
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(load_intake_config().env_file)
    if dict(model_id=config.model_id, base_url=config.base_url, timeout=config.timeout) != contract['model']:
        raise ValueError('Pilot model settings changed before worker execution')
    execution_config = ExecutionConfig(image=args.image)
    probe = DockerExecutor(execution_config)
    probe.preflight()
    if probe.image_id != contract['image']['id']:
        raise ValueError('Pilot sandbox image changed before worker execution')
    source_path = output/'inputs'/case['file']
    summary = dict(case=args.case, repeat=args.repeat, arm=args.arm, skills_enabled=args.arm == 'on',
                   contract_sha256=digest((output/'contract.json').read_bytes()), source_sha256=digest(source_path.read_bytes()))
    session = create_session(source_path, create_model(config, max_tokens=4096),
                             analysis_model=create_model(config, max_tokens=12288), skills_enabled=args.arm == 'on',
                             execution_config=execution_config)
    session.executor.set_owner('skills-pilot', owner_id(output, args.case, args.repeat, args.arm))
    start = time.monotonic()
    def progress(event):
        public = {key: event[key] for key in ['type', 'role', 'call', 'attempt', 'status', 'elapsed_seconds', 'execution_id'] if key in event}
        with (destination/'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(public, ensure_ascii=False) + '\n')
    try:
        report = session.invoke(case['task'], on_event=progress)
        errors = check_outputs(report, session.artifact_assets, case, source_path)
        summary.update(status=report['status'], attained=not errors, errors=errors, error=report.get('error'),
                       skill_loads=report.get('skills', {}).get('loads', []),
                       model_calls=report.get('model_calls', 0), analysis_model_calls=report.get('analysis_model_calls', 0),
                       **measurements(report))
        summary['record_path'] = session.save(destination)['record_path']
        if digest(source_path.read_bytes()) != summary['source_sha256']:
            summary.update(attained=False, errors=[*errors, 'source_input_changed'])
    except Exception as exc:
        summary.update(status='error', attained=False, errors=[type(exc).__name__])
    finally:
        session.executor.cancel()
        summary['elapsed_seconds'] = round(time.monotonic() - start, 3)
        write_json(destination/'summary.json', summary)


def cleanup_owned(args, output, case, repeat, arm):
    """Only containers tagged for this pilot trial, on the prechecked endpoint."""
    sys.path.insert(0, str(Path(args.source).resolve()))
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig, docker_executable
    executor = DockerExecutor(ExecutionConfig(image=args.image))
    executor.preflight()
    command = executor._docker_cli(docker_executable())
    owned = subprocess.run(command + ['ps', '-aq', '--filter', 'label=labweaver.app=skills-pilot',
                                      '--filter', f'label=labweaver.run={owner_id(output, case, repeat, arm)}'],
                           capture_output=True, timeout=15, env=executor._client_env())
    if owned.returncode:
        raise RuntimeError('Pilot container cleanup could not be verified')
    for cid in owned.stdout.decode().split():
        subprocess.run(command + ['rm', '-f', cid], capture_output=True, timeout=15, check=True, env=executor._client_env())


def claim_trial(directory, case, repeat, arm, contract_sha):
    """A crash must not silently turn into a second paid attempt on resume."""
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(case=case, repeat=repeat, arm=arm, contract_sha256=contract_sha)
    summary = directory/'summary.json'
    if summary.exists():
        record = read_json(summary)
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError('Pilot trial identity changed; use a new output directory')
        return False
    marker = directory/'attempt.json'
    if marker.exists():
        if read_json(marker) != identity:
            raise ValueError('Pilot attempt identity changed; use a new output directory')
        raise ValueError('Previously started pilot trial has no summary; inspect/stop its worker, record interruption, then resume without rerunning it')
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(identity, stream, sort_keys=True)
    return True


def run_batch(args):
    output = prepare(args.output)
    # Local config and sandbox preflight happen before any model client is invoked.
    sys.path.insert(0, str(Path(args.source).resolve()))
    from labweaver.config import load_config
    from labweaver.run_config import load_intake_config
    from labweaver.runtime.execution import DockerExecutor, ExecutionConfig
    config = load_config(load_intake_config().env_file)
    executor = DockerExecutor(ExecutionConfig(image=args.image))
    executor.preflight()
    model = dict(model_id=config.model_id, base_url=config.base_url, timeout=config.timeout)
    frozen, _ = freeze_contract(output, args.source, model, dict(tag=args.image, id=executor.image_id))
    verify_worker_contract(output, frozen)
    expected_contract = digest((output/'contract.json').read_bytes())
    for case, repeat, arm in trial_schedule(read_json(output/'suite.json')['cases']):
        if (output/'STOP').exists():
            print('Stopped between trials; every existing result retained.', flush=True)
            return
        directory = output/'trials'/f'{case}-{repeat}-{arm}'
        summary = directory/'summary.json'
        if not claim_trial(directory, case, repeat, arm, expected_contract):
            continue
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source', str(frozen),
                   '--output', str(output), '--image', args.image, '--case', case, '--repeat', str(repeat), '--arm', arm]
        print(f'{case} repeat={repeat} skills={arm}', flush=True)
        with (directory/'worker.log').open('w', encoding='utf-8') as log:
            try:
                subprocess.run(command, stdout=log, stderr=log, timeout=900, check=False)
            except subprocess.TimeoutExpired:
                write_json(summary, dict(case=case, repeat=repeat, arm=arm, contract_sha256=expected_contract,
                                         attained=False, status='error', errors=['worker_timeout']))
                cleanup_owned(args, output, case, repeat, arm)
        if not summary.exists():
            write_json(summary, dict(case=case, repeat=repeat, arm=arm, contract_sha256=expected_contract,
                                     attained=False, status='error', errors=['worker_failure_or_timeout']))
            cleanup_owned(args, output, case, repeat, arm)
        print(json.dumps({key: read_json(summary).get(key) for key in ['case', 'repeat', 'arm', 'attained', 'errors', 'elapsed_seconds']}), flush=True)


def aggregate(output):
    output = Path(output)
    suite = read_json(output/'suite.json')
    contract_sha = digest((output/'contract.json').read_bytes()) if (output/'contract.json').exists() else None
    expected = {(case, repeat, arm) for case, repeat, arm in trial_schedule(suite['cases'])}
    rows = {}
    for path in output.glob('trials/*/summary.json'):
        record = read_json(path)
        key = (record.get('case'), record.get('repeat'), record.get('arm'))
        if key not in expected or key in rows or record.get('contract_sha256') != contract_sha:
            raise ValueError('Duplicate, unknown, or mixed-contract pilot trial')
        rows[key] = record
    arms = {}
    for arm in ['off', 'on']:
        trials = [r for key, r in rows.items() if key[2] == arm]
        elapsed = sorted(r['elapsed_seconds'] for r in trials if isinstance(r.get('elapsed_seconds'), (int, float)))
        failed = [r for r in trials if r.get('execution_failures', 0)]
        arms[arm] = dict(planned=len(expected)//2, recorded=len(trials), missing=len(expected)//2-len(trials),
                         attained=sum(bool(r.get('attained')) for r in trials),
                         first_attempt_attained=sum(bool(r.get('attained')) and r.get('no_internal_repair', False) for r in trials),
                         code_failure_trials=len(failed), repaired_and_attained=sum(bool(r.get('attained')) for r in failed),
                         known_total_tokens=sum(r.get('known_total_tokens', 0) for r in trials),
                         unknown_usage_calls=sum(r.get('unknown_usage_calls', 0) for r in trials),
                         usage_unmeasured_trials=sum('unknown_usage_calls' not in r for r in trials),
                         skill_load_count=sum(len(r.get('skill_loads', [])) for r in trials),
                         trials_loading_skills=sum(bool(r.get('skill_loads')) for r in trials),
                         median_seconds=statistics.median(elapsed) if elapsed else None)
    paired = dict(both_passed=0, both_failed=0, skills_only_passed=0, baseline_only_passed=0, incomplete_pairs=0)
    for case, repeat, arm in expected:
        if arm != 'off':
            continue
        a, b = rows.get((case, repeat, 'off')), rows.get((case, repeat, 'on'))
        if a is None or b is None:
            paired['incomplete_pairs'] += 1
        else:
            paired['both_passed' if a.get('attained') and b.get('attained') else
                   'skills_only_passed' if b.get('attained') else
                   'baseline_only_passed' if a.get('attained') else 'both_failed'] += 1
    cases = {case['id']: {arm: {'recorded': sum(k[0] == case['id'] and k[2] == arm for k in rows),
                                       'attained': sum(bool(r.get('attained')) for k, r in rows.items() if k[0] == case['id'] and k[2] == arm)}
                             for arm in ['off', 'on']} for case in suite['cases']}
    return dict(version=1, complete=len(rows) == len(expected), contract_sha256=contract_sha, arms=arms, paired=paired, by_case=cases,
                boundary='Small paired synthetic pilot; three repeats are correlated. Skills loading and code exit zero are not task success. No universal improvement or statistical significance claimed.')


def save_report(output):
    output = Path(output)
    data = aggregate(output)
    write_json(output/'report.json', data)
    lines = ['# Skills 开关对照试验', '', f"完整批次：{'是' if data['complete'] else '否'}。固定8题，每题每组3次；两组使用相同模型、数据、代码和资源预算。", '',
             '| 指标 | Skills 关闭 | Skills 开启 |', '|---|---:|---:|']
    for key, label in [('recorded', '已记录运行'), ('missing', '缺失运行'), ('attained', '任务达成'),
                       ('first_attempt_attained', '无需内部修正即达成'), ('code_failure_trials', '发生执行/核验失败'),
                       ('repaired_and_attained', '失败后修正达成'), ('known_total_tokens', '实际记录 token'),
                       ('unknown_usage_calls', '请求用量未知'), ('usage_unmeasured_trials', '未取得用量计量的运行'),
                       ('skill_load_count', '实际技能加载次数'), ('trials_loading_skills', '实际加载技能的运行'), ('median_seconds', '耗时中位数（秒）')]:
        lines.append(f"| {label} | {data['arms']['off'][key]} | {data['arms']['on'][key]} |")
    p = data['paired']
    lines += ['', f"配对结果：双方达成{p['both_passed']}，双方未达成{p['both_failed']}，仅Skills达成{p['skills_only_passed']}，仅关闭组达成{p['baseline_only_passed']}，未完成配对{p['incomplete_pairs']}。", '',
              '| 任务 | 关闭组达成/记录 | 开启组达成/记录 |', '|---|---:|---:|']
    for case, arms in data['by_case'].items():
        lines.append(f"| {case} | {arms['off']['attained']}/{arms['off']['recorded']} | {arms['on']['attained']}/{arms['on']['recorded']} |")
    lines += ['', '不因Skills未加载而删除开启组运行；两组所有失败均保留。不重新运行覆盖失败。缺失试次不能当作成功或零成本。', '',
              '独立判据检查真实表格、指标、源记录、预测与绘图数据；不能证明任意算法语义、训练全过程无泄漏或报告所有结论正确。小规模合成数据与相关重复不足以证明普遍提升或统计显著性。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--report', action='store_true')
    action.add_argument('--worker', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--source', type=Path, default=ROOT/'src')
    parser.add_argument('--image', default='labweaver-python:0.3.0')
    parser.add_argument('--case')
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--arm', choices=['off', 'on'])
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.output))
    elif args.run:
        run_batch(args)
    elif args.report:
        save_report(args.output)
    else:
        if args.case is None or args.repeat not in [1, 2, 3] or args.arm is None:
            parser.error('Worker requires --case, --repeat 1..3 and --arm')
        run_worker(args)


if __name__ == '__main__':
    main()
