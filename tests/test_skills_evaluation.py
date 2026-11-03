"""A/B accounting and independent output oracles; no network or generated host code."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def pilot(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/'evaluation'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('skills_pilot_eval', directory/'skills_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_keeps_tasks_and_inputs_immutable_on_resume(pilot, tmp_path):
    pilot.prepare(tmp_path)
    original = (tmp_path/'suite.json').read_bytes()
    assert len(pilot.read_json(tmp_path/'suite.json')['cases']) == 8
    assert pilot.prepare(tmp_path) == tmp_path
    assert (tmp_path/'suite.json').read_bytes() == original
    path = tmp_path/'inputs/weights.csv'
    path.write_bytes(path.read_bytes().replace(b'10,2', b'10,7'))
    with pytest.raises(ValueError, match='snapshot changed'):
        pilot.prepare(tmp_path)


def test_balanced_order_contains_exactly_three_trials_per_arm(pilot):
    suite, _ = pilot.public_suite()
    schedule = list(pilot.trial_schedule(suite['cases']))
    assert len(schedule) == len(set(schedule)) == 48
    for index, case in enumerate(suite['cases']):
        order = [arm for cid, repeat, arm in schedule if cid == case['id']]
        assert order.count('on') == order.count('off') == 3
        assert order[:2] != order[2:4]


def test_container_ownership_is_scoped_to_batch_and_trial(pilot, tmp_path):
    first = pilot.owner_id(tmp_path/'one', 'S1', 1, 'on')
    assert first == pilot.owner_id(tmp_path/'one', 'S1', 1, 'on')
    assert first != pilot.owner_id(tmp_path/'two', 'S1', 1, 'on')
    assert first != pilot.owner_id(tmp_path/'one', 'S1', 1, 'off')
    assert len(first) <= 80


def test_aggregate_retains_losses_and_incomplete_pairs(pilot, tmp_path):
    pilot.prepare(tmp_path)
    contract = {'model': 'public-test-model'}
    pilot.write_json(tmp_path/'contract.json', contract)
    fingerprint = pilot.digest((tmp_path/'contract.json').read_bytes())
    def trial(case, repeat, arm, attained, **extra):
        pilot.write_json(tmp_path/f'trials/{case}-{repeat}-{arm}/summary.json',
                         dict(case=case, repeat=repeat, arm=arm, attained=attained,
                              contract_sha256=fingerprint, **extra))
    trial('S1', 1, 'off', True, no_internal_repair=True, elapsed_seconds=3)
    trial('S1', 1, 'on', False, skill_loads=[{'skill_id': 'data-exploration'}])
    trial('S2', 1, 'off', False)
    trial('S2', 1, 'on', True, execution_failures=1, known_total_tokens=55, unknown_usage_calls=2)
    trial('S3', 1, 'off', False)
    report = pilot.aggregate(tmp_path)
    assert not report['complete']
    assert report['paired'] == dict(both_passed=0, both_failed=0, skills_only_passed=1, baseline_only_passed=1, incomplete_pairs=22)
    assert report['arms']['on']['attained'] == 1
    assert report['arms']['on']['skill_load_count'] == 1
    assert report['arms']['on']['repaired_and_attained'] == 1
    assert report['arms']['on']['known_total_tokens'] == 55
    assert report['arms']['on']['unknown_usage_calls'] == 2
    assert report['arms']['on']['usage_unmeasured_trials'] == 1
    pilot.write_json(tmp_path/'trials/extra/summary.json',
                     dict(case='S1', repeat=1, arm='off', contract_sha256=fingerprint, attained=True))
    with pytest.raises(ValueError, match='Duplicate'):
        pilot.aggregate(tmp_path)


def test_report_rejects_cross_contract_results(pilot, tmp_path):
    pilot.prepare(tmp_path)
    pilot.write_json(tmp_path/'contract.json', {'version': 1})
    pilot.write_json(tmp_path/'trials/S1-1-on/summary.json',
                     dict(case='S1', repeat=1, arm='on', attained=True, contract_sha256='wrong'))
    with pytest.raises(ValueError, match='mixed-contract'):
        pilot.aggregate(tmp_path)


def test_resume_never_silently_reexecutes_started_or_completed_trial(pilot, tmp_path):
    assert pilot.claim_trial(tmp_path, 'S1', 1, 'on', 'contract')
    with pytest.raises(ValueError, match='Previously started'):
        pilot.claim_trial(tmp_path, 'S1', 1, 'on', 'contract')
    pilot.write_json(tmp_path/'summary.json',dict(case='S1',repeat=1,arm='on',contract_sha256='contract',attained=False))
    assert not pilot.claim_trial(tmp_path, 'S1', 1, 'on', 'contract')
    with pytest.raises(ValueError, match='identity changed'):
        pilot.claim_trial(tmp_path, 'S1', 1, 'on', 'new-contract')


def test_contract_refuses_changed_model_and_resource_tampering(pilot, tmp_path):
    output = tmp_path/'batch'
    pilot.prepare(output)
    package = tmp_path/'src/labweaver'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('',encoding='utf-8')
    (package/'SKILL.md').write_text('Reviewed skill',encoding='utf-8')
    model = dict(model_id='same-model',base_url='https://example.test/v1',timeout=30)
    image = dict(tag='public-test-image',id='sha256:fake-for-identity-tests')
    source, contract = pilot.freeze_contract(output, package.parent, model, image)
    assert pilot.verify_worker_contract(output, source) == contract
    with pytest.raises(ValueError, match='model, budget'):
        pilot.freeze_contract(output, package.parent, {**model,'model_id':'different-model'}, image)
    (source/'labweaver/SKILL.md').write_text('Changed skill',encoding='utf-8')
    with pytest.raises(ValueError, match='implementation snapshot changed'):
        pilot.verify_worker_contract(output, source)


def test_completed_text_does_not_satisfy_actual_result_oracle(pilot, tmp_path):
    suite, inputs = pilot.public_suite()
    case = suite['cases'][0]
    report = dict(status='completed')
    assert 'expected_full_table_not_found' in pilot.check_outputs(report, {}, case, tmp_path/'source.csv')
    def table(rows):
        return {'id': {'metadata': {'kind': 'table'}, 'files': {'table.csv': pilot.csv_bytes(rows)}}}
    expected = [case['oracle']['columns'], *case['oracle']['table']]
    assert pilot.check_outputs(report, table(expected), case, tmp_path/'source.csv') == []
    wrong = copy.deepcopy(expected)
    wrong[1][1] += 1
    assert pilot.check_outputs(report, table(wrong), case, tmp_path/'source.csv')
    assert pilot.check_outputs(report, table(expected[:-1]), case, tmp_path/'source.csv')


def test_figure_requires_actual_matching_full_plot_data(pilot, tmp_path):
    suite, _ = pilot.public_suite()
    case = suite['cases'][4]
    expected = [case['oracle']['columns'], *case['oracle']['table']]
    assets = {'table': {'metadata': {'kind': 'table'}, 'files': {'table.csv': pilot.csv_bytes(expected)}},
              'figure': {'metadata': {'kind': 'figure'}, 'files': {'data.csv': pilot.csv_bytes(expected)}}}
    assert pilot.check_outputs({'status': 'completed'}, assets, case, tmp_path/'source.csv') == []
    assets['figure']['files']['data.csv'] = pilot.csv_bytes(expected[:-1])
    assert 'figure_data_not_full_expected_table' in pilot.check_outputs({'status': 'completed'}, assets, case, tmp_path/'source.csv')


def test_binary_metrics_recomputed_independently_and_bound_to_task(pilot, tmp_path):
    suite, inputs = pilot.public_suite()
    case = suite['cases'][6]
    source = tmp_path/case['file']
    source.write_bytes(inputs[case['file']])
    # A source-label copying test fixture verifies the oracle only, not training.
    rows = list(pilot.csv.reader(pilot.io.StringIO(inputs[case['file']].decode())))[1:]
    train = [f'r{i:06}' for i in range(1, 65)]
    test = [f'r{i:06}' for i in range(65, 81)]
    training = [row[3] for row in rows[:64]]
    baseline = min(set(training), key=lambda label: (-training.count(label), label))
    actual = [row[3] for row in rows[64:]]
    labels = sorted(set(actual))
    metrics = {}
    for key, predicted in [('model', actual), ('baseline', [baseline]*16)]:
        matrix, f1 = pilot.matrix_and_f1(actual, predicted, labels)
        metrics[key] = dict(accuracy=sum(a == b for a,b in zip(actual,predicted))/16,
                            macro_f1=f1, labels=labels, confusion_matrix=matrix)
    spec = dict(task_type='classification', target='c4', features=['c2','c3'], split='stratified', seed=42,
                test_size=.2, duplicate_policy='keep', prediction_delivery='p', metadata_delivery='d', metrics_delivery='m')
    prediction = [['source_record_id','y_true','y_pred','baseline_pred'],
                  *[[rid,y,y,baseline] for rid,y in zip(test,actual)]]
    assets = {'p': {'metadata': {'deliverable_id':'p'}, 'files': {'table.csv':pilot.csv_bytes(prediction)}},
              'd': {'metadata': {'deliverable_id':'d','value':{'train_ids':train,'test_ids':test,'excluded_ids':[]}}},
              'm': {'metadata': {'deliverable_id':'m','value':metrics}}}
    report = {'task_spec':{'modeling':spec},'verification':[{'results':[{'status':'passed'}]}]}
    assert pilot.check_modeling(report, assets, case['oracle'], source) == []
    assets['m']['metadata']['value']['model']['accuracy'] = .9
    assert 'independent_classification_metrics:model' in pilot.check_modeling(report,assets,case['oracle'],source)
    report['task_spec']['modeling']['target'] = 'c3'
    assert pilot.check_modeling(report,assets,case['oracle'],source) == ['modeling_task_contract']
