"""Public-data end-to-end benchmark; each run isolated from candidate/baseline imports.

python evaluation/run.py --source .cache/baseline-62bb7ad/src --phase baseline
python evaluation/run.py --source src --phase candidate
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def freeze_source(source, output):
    """Each batch imports an immutable local copy, even if development continues."""
    destination=output.resolve()/'implementation'/'src'
    manifest=destination.parent/'manifest.json'
    if manifest.exists():
        expected=json.loads(manifest.read_text(encoding='utf-8'))
        for relative,digest in expected.items():
            if hashlib.sha256((destination/relative).read_bytes()).hexdigest()!=digest:
                raise ValueError('Batch implementation snapshot changed; use a new output directory')
        return destination
    package=Path(source).resolve()/'labweaver'
    expected={}
    for item in sorted(package.rglob('*')):
        if item.is_file() and item.suffix in {'.py','.sql','.md','.json','.txt'}:
            relative=Path('labweaver')/item.relative_to(package)
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            raw=item.read_bytes();target.write_bytes(raw)
            expected[relative.as_posix()]=hashlib.sha256(raw).hexdigest()
    if not expected: raise ValueError('No implementation to evaluate')
    manifest.write_text(json.dumps(expected,sort_keys=True,indent=2),encoding='utf-8')
    return destination


def freeze_suite(source, output):
    """Task wording and oracles belong to the batch just like its source code."""
    output.mkdir(parents=True,exist_ok=True)
    target=output.resolve()/'suite.json'
    fingerprint=output.resolve()/'suite.sha256'
    if target.exists():
        if not fingerprint.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=fingerprint.read_text().strip():
            raise ValueError('Batch task snapshot changed; use a new output directory')
        return target
    raw=Path(source).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    for prior in output.glob('*/summary.json'):
        expected=json.loads(prior.read_text(encoding='utf-8')).get('suite_sha256')
        if expected and expected!=digest:
            raise ValueError('Cannot mix task revisions in an existing batch')
    target.write_bytes(raw);fingerprint.write_text(digest+'\n',encoding='utf-8')
    return target


def check(report, assets, oracle):
    from labweaver.runtime.artifacts import csv_table
    errors = []
    if report["status"] != oracle.get("status", "completed"):
        errors.append("status:" + report["status"])
    if "table" in oracle:
        def equal(value, expected):
            if isinstance(expected, (float,int)):
                try: return math.isclose(float(value),expected, rel_tol=1e-6,abs_tol=1e-8)
                except (ValueError,TypeError): return False
            return value == expected
        matched = False
        for a in assets.values():
            if a["metadata"]["kind"] != "table": continue
            columns, rows = csv_table(a["files"]["table.csv"])
            rows = [[r[c] for c in columns] for r in rows]
            expected = oracle["table"]
            if "columns" in oracle and columns != oracle["columns"]: continue
            if len(rows)==len(expected) and all(len(r)==len(e) and all(equal(v,w) for v,w in zip(r,e)) for r,e in zip(rows,expected)):
                matched = True
        if not matched: errors.append("expected_full_table_not_found")
    if "metric" in oracle:
        metrics = [a["metadata"].get("value") for a in assets.values() if a["metadata"]["kind"]=="metric"]
        for key,value in oracle["metric"].items():
            if not any(isinstance(m,dict) and isinstance(m.get(key),(int,float)) and math.isclose(m[key],value,rel_tol=1e-6,abs_tol=1e-8) for m in metrics): errors.append("metric:"+key)
    if oracle.get("figure") and not any(a["metadata"]["kind"]=="figure" for a in assets.values()): errors.append("missing_figure")
    if oracle.get("retrieval") and report.get("retrieval_status")!="used": errors.append("missing_retrieval")
    if oracle.get("no_execution") and report.get("code_executions"): errors.append("unexpected_execution")
    if "profile" in oracle:
        p = report.get("profile_result") or {}
        if [p.get("row_count"),p.get("column_count")] != oracle["profile"]: errors.append("profile_counts")
    if oracle.get("modeling"):
        # The separately maintained modeling oracle runs outside the Agent.
        from modeling_oracle import check_modeling
        errors.extend(check_modeling(report, assets, oracle))
    return errors


def worker(args):
    os.environ['LANGSMITH_TRACING']='false'
    os.environ['LANGCHAIN_TRACING_V2']='false'
    sys.path.insert(0,str(Path(args.source).resolve()))
    from labweaver.agent import create_session
    from labweaver.config import load_config, create_model
    from labweaver.run_config import load_intake_config
    from labweaver.runtime.execution import ExecutionConfig
    spec = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    case = next(c for c in spec["cases"] if c["id"] == args.case)
    os.chdir(ROOT)
    cfg = load_intake_config()
    model = load_config(cfg.env_file)
    path = ROOT/"evaluation/fixtures"/case["file"]
    destination = Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    session = create_session(path,create_model(model),analysis_model=create_model(model,max_tokens=12288),
                             material_paths=[ROOT/"evaluation/fixtures"/p for p in case.get("materials",[])],
                             execution_config=ExecutionConfig(image=args.image))
    session.executor.set_owner("evaluation",args.case+"-"+str(args.repeat)+"-"+args.phase)
    start = time.monotonic()
    def progress(event):
        public={k:event[k] for k in ['type','role','call','attempt','status','elapsed_seconds','execution_id'] if k in event}
        with (destination/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(public,ensure_ascii=False)+'\n')
    summary = {"case":case["id"],"repeat":args.repeat,"phase":args.phase,"model":model.model_id,
               "base_url":model.base_url,"image":args.image,"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
               "suite_sha256":hashlib.sha256(Path(args.suite).read_bytes()).hexdigest()}
    source_root=Path(args.source).resolve()/"labweaver"
    hashes={p.relative_to(source_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.rglob('*.py'))}
    summary['implementation_sha256']=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    summary['model_configuration']={'temperature':0,'main_max_tokens':4096,'analysis_max_tokens':12288,'max_retries':0}
    try:
        report = session.invoke(case["task"],on_event=progress if args.phase=='development' else None)
        reports=[report]
        errors = check(report,session.artifact_assets,case["oracle"])
        saved = session.save(destination)
        summary["record_path"] = saved["record_path"]
        if case.get("followup") and report["status"]=="completed":
            report = session.invoke(case["followup"],on_event=progress if args.phase=='development' else None)
            reports.append(report)
            errors.extend(check(report,session.artifact_assets,case["followup_oracle"]))
            summary["followup_record_path"] = session.save(destination)["record_path"]
        summary.update(status=report["status"],attained=not errors,errors=errors,
                       model_calls=report.get("model_calls"),analysis_model_calls=report.get("analysis_model_calls"),
                       token_usage=report.get("token_usage"),code_attempts=len(report.get("code_executions",[])),
                       execution_failures=sum(e["status"]!="completed" for e in report.get("code_executions",[])),
                       missing_deliverables=report.get("missing_deliverables"),error=report.get("error"))
        from measurements import trial_measurements
        summary.update(trial_measurements(reports))
    except Exception as exc:
        summary.update(status="error",attained=False,errors=[type(exc).__name__])
    finally:
        session.executor.cancel()
        summary["elapsed_seconds"] = round(time.monotonic()-start,3)
        (destination/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="src"); p.add_argument("--phase",choices=["baseline","candidate","development"],required=True)
    p.add_argument("--suite",default=str(ROOT/'evaluation/suite.json'))
    p.add_argument("--image",default="labweaver-python:0.2.0"); p.add_argument("--output")
    p.add_argument("--case"); p.add_argument("--repeat",type=int,default=1); p.add_argument("--worker",action="store_true")
    args=p.parse_args()
    if args.worker: return worker(args)
    root=Path(args.output or ROOT/".cache/evaluation"/args.phase)
    source=freeze_source(args.source,root)
    suite=freeze_suite(args.suite,root)
    cases=json.loads(suite.read_text(encoding="utf-8"))["cases"]
    if args.phase=="baseline": cases=[c for c in cases if c["family"]!="modeling"]
    if args.case: cases=[c for c in cases if c["id"]==args.case]
    for n in range(1,4):
        for case in cases:
            if (root/'STOP').exists():
                print('Batch stopped between trials; existing results and implementation retained.',flush=True)
                return
            out=root/(case["id"]+f"-{n}")
            if (out/"summary.json").exists(): continue
            out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,__file__,"--worker","--source",str(source),"--phase",args.phase,
                     "--suite",str(suite),
                     "--case",case["id"],"--repeat",str(n),"--output",str(out),"--image",args.image]
            print(f"{args.phase} {case['id']} repetition {n}",flush=True)
            with (out/"worker.log").open("w",encoding="utf-8") as log:
                try: subprocess.run(command,stdout=log,stderr=log,timeout=900,check=False)
                except subprocess.TimeoutExpired:
                    # Only containers belonging to this exact benchmark run.
                    owned=subprocess.run(["docker","ps","-aq","--filter","label=labweaver.app=evaluation","--filter",f"label=labweaver.run={case['id']}-{n}-{args.phase}"],capture_output=True,text=True)
                    for cid in owned.stdout.split(): subprocess.run(["docker","rm","-f",cid],capture_output=True)
            if not (out/"summary.json").exists():
                (out/"summary.json").write_text(json.dumps({"case":case["id"],"repeat":n,"phase":args.phase,"attained":False,"errors":["worker_failure_or_timeout"]}),encoding="utf-8")
            s=json.loads((out/"summary.json").read_text(encoding="utf-8"))
            print(json.dumps({k:s.get(k) for k in ["case","repeat","attained","errors","elapsed_seconds"]}),flush=True)


if __name__=="__main__": main()
