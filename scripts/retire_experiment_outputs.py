"""Retire explicit generated-output paths after a surviving report is recorded.

Dry run by default. Never accepts source, checkpoint or original-reference paths.
The input plan contains exact absolute paths, allowed bases, protected paths and
an existing result-report file; the audit is written outside all deletion targets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def validate(plan,report):
    bases=[Path(p).resolve(strict=True) for p in plan['allowed_bases']]
    protected=[Path(p).resolve() for p in plan['protected']]
    evidence=Path(plan['result_report']).resolve(strict=True)
    output=Path(report).resolve();records=[]
    for item in plan['targets']:
        raw=Path(item)
        if not raw.is_absolute():raise ValueError('Absolute paths required')
        p=raw.resolve()
        if p!=raw or raw.is_symlink():raise ValueError('Aliased deletion path')
        if not any(p.is_relative_to(b) and p!=b for b in bases):raise ValueError('Deletion outside allowed bases')
        if any(p.is_relative_to(q) or q.is_relative_to(p) for q in protected):raise ValueError('Protected path')
        if evidence.is_relative_to(p) or output.is_relative_to(p):raise ValueError('Report must survive')
        if not p.exists():continue
        files=[p] if p.is_file() else sorted(p.rglob('*'))
        if any(f.is_symlink() for f in files):raise ValueError('Nested symlink')
        entries=[f for f in files if f.is_file()]
        records.append({'path':str(p),'files':len(entries),'bytes':sum(f.stat().st_size for f in entries)})
    return records,evidence


def retire(plan_path,report,apply=False):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8-sig'));records,evidence=validate(plan,report)
    audit={'status':'inventory','targets':records,'bytes':sum(v['bytes'] for v in records),
           'surviving_report':str(evidence),'report_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),
           'protected':plan['protected'],'policy':'User requested report-only retention; deleted structural data will require regeneration.'}
    if apply:
        for row in records:
            p=Path(row['path']);shutil.rmtree(p) if p.is_dir() else p.unlink()
        audit['status']='deleted'
    out=Path(report);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':audit['status'],'targets':len(records),'bytes':audit['bytes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--report',required=True);p.add_argument('--apply',action='store_true')
    a=p.parse_args();retire(a.plan,a.report,a.apply)
