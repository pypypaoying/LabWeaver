"""Offline paired evaluation contracts, receipt accounting and trial lifecycle."""
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def evaluator(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / 'evaluation'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('token_efficiency_test_eval', directory / 'token_efficiency.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources(tmp_path):
    result = []
    for name, version in [('original', 1), ('optimized', 2)]:
        source = tmp_path / name / 'src'
        package = source / 'labweaver'
        skill = package / 'skills/statistics/SKILL.md'
        skill.parent.mkdir(parents=True)
        (package / '__init__.py').write_text(f'VERSION = {version}\n', encoding='utf-8')
        (package / '.env').write_text('PRIVATE_CREDENTIAL_SENTINEL', encoding='utf-8')
        skill.write_text('Fixed reviewed guidance', encoding='utf-8')
        result.append(source)
    return result


def prepared(evaluator, tmp_path, sources, **choices):
    output = evaluator.prepare(tmp_path / 'batch', *sources, **choices)
    evaluator.freeze_contract(output, dict(model_id='test-model', base_url='https://PRIVATE_ENDPOINT.test/v1', timeout=30),
                              dict(tag='test-image', id='sha256:identity'))
    return output


def usage(input_tokens, output_tokens, **extra):
    return dict(usage=dict(input_tokens=input_tokens, output_tokens=output_tokens,
                           total_tokens=input_tokens + output_tokens), **extra)


def metered(evaluator, total):
    return evaluator.receipt_measurements(dict(model_calls=1, analysis_model_calls=0,
                                               token_usage=dict(main=[usage(total, 0)], analysis=[])))


def record_trial(evaluator, output, case, repeat, variant, attained, **extra):
    directory = output / 'trials' / f'{case}-{repeat}-{variant}'
    fingerprint = evaluator.digest((output / 'contract.json').read_bytes())
    assert evaluator.claim_trial(directory, case, repeat, variant, fingerprint)
    row = dict(**evaluator.identity(case, repeat, variant, fingerprint), attained=attained,
               status='completed' if attained else 'error', saved=attained,
               input_hash_unchanged=True, errors=[] if attained else ['independent_oracle_failed'])
    row.update(extra)
    evaluator.write_json(directory / 'summary.json', row)
    return directory / 'summary.json'


def test_default_preparation_freezes_exactly_sixteen_trials_and_both_sources(evaluator, sources, tmp_path):
    output = evaluator.prepare(tmp_path / 'batch', *sources)
    protocol = evaluator.verify_prepared(output)
    assert protocol['cases'] == ['S1', 'S2', 'S5', 'S7']
    assert protocol['planned_trials'] == len(protocol['schedule']) == 16
    assert protocol['repetitions'] == 2 and protocol['skills_enabled'] is False
    assert protocol['trial_timeout_seconds'] == 900
    assert protocol['model_options'] == dict(temperature=0, main_max_tokens=4096,
                                              analysis_max_tokens=12288, max_retries=0)
    assert protocol['implementations']['baseline'] != protocol['implementations']['candidate']
    assert set(evaluator.read_json(output / 'inputs-manifest.json')) == {
        'suite.json', 'inputs/districts.csv', 'inputs/energy.csv', 'inputs/decisions.csv'}
    assert not list(output.rglob('.env'))
    frozen = output / 'candidate/implementation/src/labweaver/__init__.py'
    original = frozen.read_bytes()
    (sources[1] / 'labweaver/__init__.py').write_text('VERSION = 99', encoding='utf-8')
    assert evaluator.prepare(output, *sources) == output
    assert frozen.read_bytes() == original


def test_schedule_alternates_variant_order_and_never_changes_trial_membership(evaluator):
    cases = [dict(id=case) for case in evaluator.DEFAULT_CASES]
    schedule = list(evaluator.trial_schedule(cases))
    assert len(schedule) == len(set(schedule)) == 16
    assert schedule[:4] == [('S1', 1, 'baseline'), ('S1', 1, 'candidate'),
                            ('S2', 1, 'candidate'), ('S2', 1, 'baseline')]
    for case in evaluator.DEFAULT_CASES:
        order = [v for c, r, v in schedule if c == case]
        assert order[:2] == list(reversed(order[2:]))
        assert order.count('baseline') == order.count('candidate') == 2


@pytest.mark.parametrize('changed', [dict(cases=['S1']), dict(repetitions=3), dict(skills=True)])
def test_prepared_choices_cannot_be_changed_on_resume(evaluator, sources, tmp_path, changed):
    output = evaluator.prepare(tmp_path / 'batch', *sources)
    with pytest.raises(ValueError, match='selection, repetitions or Skills changed'):
        evaluator.prepare(output, *sources, **changed)


@pytest.mark.parametrize('choices', [dict(cases=['S1', 'S1']), dict(cases=['S8']),
                                    dict(cases=[]), dict(repetitions=0), dict(repetitions=True)])
def test_invalid_protocol_is_rejected_before_freezing_trials(evaluator, sources, tmp_path, choices):
    with pytest.raises(ValueError):
        evaluator.prepare(tmp_path / 'batch', *sources, **choices)
    assert not list(tmp_path.rglob('attempt.json'))


@pytest.mark.parametrize('target', ['inputs/energy.csv', 'suite.json',
                                   'baseline/implementation/src/labweaver/__init__.py', 'protocol.json'])
def test_tampered_inputs_source_or_protocol_are_rejected(evaluator, sources, tmp_path, target):
    output = evaluator.prepare(tmp_path / 'batch', *sources)
    path = output / target
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'):
        evaluator.verify_prepared(output)


def test_unmanifested_source_addition_is_rejected(evaluator, sources, tmp_path):
    output = evaluator.prepare(tmp_path / 'batch', *sources)
    (output / 'candidate/implementation/src/labweaver/untracked.py').write_text('injected = True', encoding='utf-8')
    with pytest.raises(ValueError, match='snapshot changed'):
        evaluator.verify_prepared(output)


def test_model_image_runtime_and_evaluator_are_bound_to_contract(evaluator, sources, tmp_path, monkeypatch):
    output = prepared(evaluator, tmp_path, sources)
    _, contract = evaluator.verify_contract(output)
    with pytest.raises(ValueError, match='contract changed'):
        evaluator.freeze_contract(output, {**contract['model'], 'model_id': 'different-model'}, contract['image'])
    with pytest.raises(ValueError, match='contract changed'):
        evaluator.freeze_contract(output, contract['model'], {**contract['image'], 'id': 'sha256:different'})
    monkeypatch.setattr(evaluator, 'runtime_identity', lambda: {'python': 'different-runtime'})
    with pytest.raises(ValueError, match='Runtime or protocol'):
        evaluator.verify_contract(output)
    monkeypatch.setattr(evaluator, 'evaluator_hashes', lambda: {'token_efficiency.py': 'different'})
    with pytest.raises(ValueError, match='Evaluator changed'):
        evaluator.verify_prepared(output)


def test_claims_prevent_reexecution_of_interrupted_or_finished_attempts(evaluator, tmp_path):
    directory = tmp_path / 'trial'
    assert evaluator.claim_trial(directory, 'S1', 1, 'baseline', 'contract')
    with pytest.raises(ValueError, match='Previously started'):
        evaluator.claim_trial(directory, 'S1', 1, 'baseline', 'contract')
    evaluator.write_json(directory / 'summary.json', dict(**evaluator.identity('S1', 1, 'baseline', 'contract'),
                                                         attained=False, status='error'))
    assert not evaluator.claim_trial(directory, 'S1', 1, 'baseline', 'contract')
    with pytest.raises(ValueError, match='identity changed'):
        evaluator.claim_trial(directory, 'S1', 1, 'baseline', 'different-contract')
    assert evaluator.owner_id(tmp_path / 'one', 'S1', 1, 'baseline') != evaluator.owner_id(tmp_path / 'two', 'S1', 1, 'baseline')
    assert evaluator.owner_id(tmp_path, 'S1', 1, 'baseline') != evaluator.owner_id(tmp_path, 'S1', 1, 'candidate')


def test_receipts_meter_both_roles_and_leave_unsupported_cache_unknown(evaluator):
    report = dict(model_calls=2, analysis_model_calls=1,
                  token_usage=dict(main=[usage(100, 10), usage(20, 5)], analysis=[usage(50, 15)]))
    result = evaluator.receipt_measurements(report)
    assert result['known_input_tokens'] == 170
    assert result['known_output_tokens'] == 30
    assert result['known_total_tokens'] == 200
    assert result['model_calls'] == 2 and result['analysis_model_calls'] == 1
    assert result['usage_complete'] is True and result['unknown_usage_calls'] == 0
    assert result['cached_input_tokens'] is None and result['known_cached_input_tokens'] is None
    assert result['cache_unknown_calls'] == 3 and result['cache_usage_complete'] is False


def test_cache_and_context_bytes_are_only_numeric_receipts(evaluator):
    report = dict(model_calls=1, analysis_model_calls=1,
                  token_usage=dict(main=[usage(100, 10, cache_usage={'read_tokens': 40},
                                                context_bytes=dict(system_bytes=11, message_bytes=20, tool_schema_bytes=30))],
                                   analysis=[usage(50, 5, cache_usage={'read_tokens': 0},
                                                   context_bytes=dict(system_bytes=7, message_bytes='PRIVATE_SENTINEL', tool_schema_bytes=3))]))
    result = evaluator.receipt_measurements(report)
    assert result['cached_input_tokens'] == 40 and result['cache_usage_complete'] is True
    assert result['cache_unknown_calls'] == 0
    assert result['context_bytes']['system_bytes'] == dict(known_sum=18, measured_calls=2, expected_calls=2)
    assert result['context_bytes']['message_bytes'] == dict(known_sum=20, measured_calls=1, expected_calls=2)
    assert 'PRIVATE_SENTINEL' not in json.dumps(result)


@pytest.mark.parametrize('change', ['missing-receipt', 'invalid-receipt', 'extra-receipt', 'missing-count'])
def test_incomplete_or_mismatched_usage_cannot_be_complete(evaluator, change):
    report = dict(model_calls=1, analysis_model_calls=1,
                  token_usage=dict(main=[usage(100, 10)], analysis=[usage(50, 5)]))
    if change == 'missing-receipt':
        report['token_usage']['analysis'] = []
    elif change == 'invalid-receipt':
        report['token_usage']['analysis'][0]['usage']['input_tokens'] = -1
    elif change == 'extra-receipt':
        report['token_usage']['analysis'].append(usage(3, 4))
    else:
        del report['analysis_model_calls']
    result = evaluator.receipt_measurements(report)
    assert result['usage_complete'] is False
    if change == 'missing-count':
        assert result['unknown_usage_calls'] is None
    if change in {'missing-receipt', 'invalid-receipt'}:
        assert result['known_total_tokens'] == 110
        assert result['unknown_usage_calls'] == 1


def test_unknown_usage_is_not_fabricated_zero(evaluator):
    result = evaluator.receipt_measurements(dict(model_calls=1, analysis_model_calls=0))
    assert result['known_input_tokens'] is None
    assert result['known_output_tokens'] is None
    assert result['known_total_tokens'] is None
    assert result['unknown_usage_calls'] == 1 and result['usage_complete'] is False


def test_aggregate_retains_failed_and_missing_runs_in_planned_denominator(evaluator, sources, tmp_path):
    output = prepared(evaluator, tmp_path, sources)
    record_trial(evaluator, output, 'S1', 1, 'baseline', True, **metered(evaluator, 100))
    record_trial(evaluator, output, 'S1', 1, 'candidate', False, **metered(evaluator, 50))
    record_trial(evaluator, output, 'S2', 1, 'baseline', False)
    result = evaluator.aggregate(output)
    assert not result['complete'] and not result['fully_metered_complete_comparison']
    assert result['variants']['baseline']['planned'] == 8
    assert result['variants']['baseline']['recorded'] == 2
    assert result['variants']['baseline']['attained'] == 1
    assert result['variants']['baseline']['success_rate_planned'] == 1 / 8
    assert result['variants']['candidate']['failed'] == 1
    assert result['variants']['candidate']['missing'] == 7
    assert result['by_case']['S2']['baseline'] == dict(planned=2, recorded=1, attained=0)
    assert result['paired']['baseline_only_passed'] == 1
    assert result['paired']['incomplete_pairs'] == 7
    assert result['all_planned_token_reduction_fraction'] is None


def test_complete_token_comparison_includes_failures_and_does_not_filter_to_success(evaluator, sources, tmp_path):
    output = prepared(evaluator, tmp_path, sources, cases=['S1'], repetitions=2)
    for repeat in (1, 2):
        record_trial(evaluator, output, 'S1', repeat, 'baseline', True, **metered(evaluator, 100))
        record_trial(evaluator, output, 'S1', repeat, 'candidate', repeat == 1,
                     **metered(evaluator, 50 if repeat == 1 else 150))
    result = evaluator.aggregate(output)
    assert result['complete'] and result['fully_metered_complete_comparison']
    assert result['variants']['candidate']['known_total_tokens'] == 200
    assert result['variants']['candidate']['attained'] == 1
    assert result['all_planned_token_reduction_fraction'] == 0
    assert result['both_passed_token_reduction_fraction'] == 0.5
    assert result['paired']['both_passed_fully_metered_pairs'] == 1
    assert result['both_passed_paired_tokens'] == dict(baseline=100, candidate=50)


def test_complete_run_with_unknown_usage_cannot_claim_a_token_reduction(evaluator, sources, tmp_path):
    output = prepared(evaluator, tmp_path, sources, cases=['S1'], repetitions=1)
    record_trial(evaluator, output, 'S1', 1, 'baseline', True, **metered(evaluator, 100))
    record_trial(evaluator, output, 'S1', 1, 'candidate', True,
                 **evaluator.receipt_measurements(dict(model_calls=1, analysis_model_calls=0)))
    result = evaluator.aggregate(output)
    assert result['complete'] is True
    assert result['fully_metered_complete_comparison'] is False
    assert result['all_planned_token_reduction_fraction'] is None
    assert result['variants']['candidate']['known_total_tokens'] is None


@pytest.mark.parametrize('corruption', ['mixed-contract', 'wrong-directory', 'missing-attempt', 'unknown-trial'])
def test_aggregate_rejects_untraceable_or_mixed_trials(evaluator, sources, tmp_path, corruption):
    output = prepared(evaluator, tmp_path, sources, cases=['S1'], repetitions=1)
    path = record_trial(evaluator, output, 'S1', 1, 'baseline', False)
    if corruption == 'mixed-contract':
        value = evaluator.read_json(path)
        value['contract_sha256'] = 'another-contract'
        evaluator.write_json(path, value)
    elif corruption == 'wrong-directory':
        path.parent.rename(path.parent.with_name('copied-result'))
    elif corruption == 'missing-attempt':
        evaluator.write_json(path.parent / 'attempt.json', {'invalid': True})
    else:
        value = evaluator.read_json(path)
        value['case'] = 'UNKNOWN'
        evaluator.write_json(path, value)
    with pytest.raises(ValueError, match='mixed-contract|matching immutable attempt'):
        evaluator.aggregate(output)


@pytest.mark.parametrize('missing', ['status', 'saved', 'input_hash_unchanged', 'errors'])
def test_success_requires_completed_saved_unchanged_input_and_oracle(evaluator, sources, tmp_path, missing):
    output = prepared(evaluator, tmp_path, sources, cases=['S1'], repetitions=1)
    path = record_trial(evaluator, output, 'S1', 1, 'baseline', True)
    value = evaluator.read_json(path)
    value[missing] = {'status': 'error', 'saved': False, 'input_hash_unchanged': False,
                      'errors': ['oracle_failed']}[missing]
    evaluator.write_json(path, value)
    with pytest.raises(ValueError, match='missing delivery/save/input evidence'):
        evaluator.aggregate(output)


@pytest.mark.parametrize('skills', [False, True])
def test_public_report_only_exports_allowlisted_aggregates(evaluator, sources, tmp_path, monkeypatch, skills):
    output = prepared(evaluator, tmp_path, sources, cases=['S1'], repetitions=1, skills=skills)
    for variant in evaluator.VARIANTS:
        directory = record_trial(evaluator, output, 'S1', 1, variant, True, **metered(evaluator, 100),
                                 record_path='D:/PRIVATE_LOCAL_PATH/private.json',
                                 error={'message': 'PRIVATE_RESPONSE_SENTINEL'},
                                 api_key='PRIVATE_CREDENTIAL_SENTINEL').parent
        (directory / 'worker.log').write_text('PRIVATE_RAW_LOG_SENTINEL', encoding='utf-8')
    public_root = tmp_path / 'published'
    monkeypatch.setattr(evaluator, 'ROOT', public_root)
    data = evaluator.save_report(output, publish=True)
    arm = 'on' if skills else 'off'
    published_json = public_root / f'evaluation/results/token-efficiency-{arm}.json'
    published_md = public_root / f'evaluation/results/token-efficiency-{arm}.md'
    assert evaluator.read_json(published_json) == data
    combined = published_json.read_text(encoding='utf-8') + published_md.read_text(encoding='utf-8')
    for secret in ('PRIVATE_', str(tmp_path), 'base_url', 'api_key', 'record_path'):
        assert secret not in combined
    assert 'public development regressions' in combined
    assert 'holdout' in combined


def install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(__import__('sys').modules, name, module)


@pytest.mark.parametrize('save_failure', [True, False])
def test_worker_keeps_actual_receipts_but_never_passes_a_save_failure(evaluator, sources, tmp_path, monkeypatch, save_failure):
    output = prepared(evaluator, tmp_path, sources, cases=['S1'], repetitions=1)
    contract = evaluator.read_json(output / 'contract.json')
    directory = output / 'trials/S1-1-candidate'
    fingerprint = evaluator.digest((output / 'contract.json').read_bytes())
    evaluator.claim_trial(directory, 'S1', 1, 'candidate', fingerprint)
    events = []
    report = dict(status='completed', model_calls=1, analysis_model_calls=0,
                  token_usage=dict(main=[usage(90, 10)], analysis=[]))
    class Executor:
        image_id = contract['image']['id']
        def __init__(self, *args, **kwargs):
            pass
        def preflight(self):
            events.append('preflight')
        def set_owner(self, app, owner):
            events.append((app, owner))
        def cancel(self):
            events.append('cancel')
    class Session:
        executor = Executor()
        artifact_assets = {}
        def invoke(self, task, on_event):
            events.append(('invoke', task))
            return report
        def save(self, path):
            events.append('save')
            if save_failure:
                raise OSError('PRIVATE_SAVE_FAILURE')
            return {'record_path': str(path / 'record.json')}
    install_module(monkeypatch, 'labweaver.agent', create_session=lambda *a, **k: Session())
    install_module(monkeypatch, 'labweaver.config', load_config=lambda *a: SimpleNamespace(**contract['model']),
                   create_model=lambda *a, **k: object())
    install_module(monkeypatch, 'labweaver.run_config', load_intake_config=lambda: SimpleNamespace(env_file=None))
    install_module(monkeypatch, 'labweaver.runtime.execution', DockerExecutor=Executor,
                   ExecutionConfig=lambda **kwargs: kwargs)
    monkeypatch.setattr(evaluator, 'check_outputs', lambda *a: [])
    monkeypatch.chdir(tmp_path)
    evaluator.run_worker(SimpleNamespace(output=output, case='S1', repeat=1, variant='candidate'))
    summary = evaluator.read_json(directory / 'summary.json')
    assert evaluator.read_json(directory / 'report.json') == report
    assert summary['known_total_tokens'] == 100
    assert summary['input_hash_unchanged'] is True
    assert summary['saved'] is (not save_failure)
    assert summary['attained'] is (not save_failure)
    assert 'cancel' in events
    if save_failure:
        assert summary['status'] == 'error' and summary['errors'] == ['save_OSError']
        assert 'PRIVATE_SAVE_FAILURE' not in json.dumps(summary)
    before = len(events)
    with pytest.raises(ValueError, match='uncompleted claimed trial'):
        evaluator.run_worker(SimpleNamespace(output=output, case='S1', repeat=1, variant='candidate'))
    assert len(events) == before
