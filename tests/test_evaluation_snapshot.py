"""Batch source isolation prevents mixed-revision benchmark claims."""
import importlib.util
import json
from pathlib import Path
import pytest

module_spec=importlib.util.spec_from_file_location('evaluation_runner',Path(__file__).resolve().parents[1]/'evaluation/run.py')
runner=importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(runner)


def test_batch_resume_uses_frozen_source_and_rejects_tampering(tmp_path):
    package=tmp_path/'src/labweaver';package.mkdir(parents=True)
    (package/'__init__.py').write_text('VERSION=1',encoding='utf-8')
    (package/'.env').write_text('PRIVATE_SENTINEL',encoding='utf-8')
    skill=package/'skills/data-exploration/SKILL.md'
    skill.parent.mkdir(parents=True)
    skill.write_text('Reviewed original guidance',encoding='utf-8')
    output=tmp_path/'run'
    frozen=runner.freeze_source(package.parent,output)
    (package/'__init__.py').write_text('VERSION=2',encoding='utf-8')
    assert runner.freeze_source(package.parent,output)==frozen
    assert (frozen/'labweaver/__init__.py').read_text()=='VERSION=1'
    assert not (frozen/'labweaver/.env').exists()
    assert (frozen/'labweaver/skills/data-exploration/SKILL.md').read_text()=='Reviewed original guidance'
    skill.write_text('Changed development guidance',encoding='utf-8')
    assert runner.freeze_source(package.parent,output)==frozen
    (frozen/'labweaver/__init__.py').write_text('VERSION=3',encoding='utf-8')
    with pytest.raises(ValueError,match='snapshot changed'):
        runner.freeze_source(package.parent,output)


def test_task_text_and_oracle_do_not_change_mid_batch(tmp_path):
    task=tmp_path/'tasks.json';task.write_text('{"cases":[{"id":"A","task":"Original"}]}',encoding='utf-8')
    output=tmp_path/'run';frozen=runner.freeze_suite(task,output)
    original=frozen.read_bytes()
    task.write_text('{"cases":[]}',encoding='utf-8')
    assert runner.freeze_suite(task,output).read_bytes()==original
    frozen.write_text('{"cases":[]}',encoding='utf-8')
    with pytest.raises(ValueError,match='task snapshot changed'):
        runner.freeze_suite(task,output)


def test_v2_only_clarifies_approved_non_frozen_regression_task():
    root=Path(__file__).resolve().parents[1]/'evaluation'
    v1=json.loads((root/'suite.json').read_text(encoding='utf-8'))
    v2=json.loads((root/'suite-v2.json').read_text(encoding='utf-8'))
    assert v1['version']==1 and v2['version']==2
    changed=[]
    for before,after in zip(v1['cases'],v2['cases'],strict=True):
        if before!=after:
            changed.append(before['id'])
            assert not before['frozen'] and before['family']=='modeling'
            assert {k:v for k,v in before.items() if k!='task'}=={k:v for k,v in after.items() if k!='task'}
            assert '回归采用随机拆分，不进行目标分层' in after['task']
    assert changed==['M3']
