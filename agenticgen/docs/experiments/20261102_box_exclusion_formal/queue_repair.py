"""Deploy an explicitly diagnosed code repair, bounded to two scientific attempts/teacher.

Run from the *new pushed commit*. This is not a retry daemon and never edits old claims.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import deploy


def science(root):
    path=Path(root)/'attempts.json'
    rows=deploy.read(path) if path.exists() else {}
    return {s:sum(r['count'] for r in rows.values() if r['stage']==s) for s in ('calibration','formal')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-plan',required=True);parser.add_argument('--teacher',required=True)
    parser.add_argument('--commit',required=True);parser.add_argument('--evidence',required=True)
    args=parser.parse_args()
    parent=deploy.read(args.parent_plan)
    if parent.get('engineering_repair'):
        raise ValueError('Use the original PLAN for all accounting; never restart a repair blind')
    job=next(j for j in parent['jobs'] if j['teacher']==args.teacher)
    prior=Path(job['root'])
    evidence=deploy.read(args.evidence)
    if evidence.get('classification')!='engineering' or evidence.get('fixed_commit')!=args.commit:
        raise ValueError('A concrete code-defect diagnosis and pushed fix commit are required')
    terminal=Path(parent['root'])/'job_results'/(job['id']+'.json')
    if not terminal.exists() or deploy.read(terminal)['status']!='failed_skipped':
        raise ValueError('The original job is not an archived terminal failure')
    if not (Path(parent['root'])/'SUITE_FINISHED.json').exists():
        raise ValueError('Keep healthy original jobs running; queue the repair after the original sequence')
    for proc in Path('/proc').glob('[0-9]*'):
        if proc.name==str(os.getpid()):continue
        try:command=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(v.decode(errors='ignore')==str(prior) or v.decode(errors='ignore').startswith(str(prior)+'/') for v in command):
            # An idle CPU quality worker is allowed and never killed by this launcher.
            if not any(b'agenticgen.uniguide.quality_queue' in v for v in command):
                raise ValueError('An original sampling process still belongs to the failed root')
    for name in ('.executor.lock','.campaign.lock'):
        with (prior/name).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(lock,fcntl.LOCK_UN)
    store=deploy.STORAGE/'repair_authorizations';store.mkdir(parents=True,exist_ok=True)
    ledger=store/'REPAIR_LEDGER.json'
    with (store/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=deploy.read(ledger) if ledger.exists() else []
        entries=[r for r in rows if r['teacher']==args.teacher]
        roots=[str(prior),*[r['root'] for r in entries]]
        actual=[science(p) for p in roots]
        if sum(sum(r.values())>0 for r in actual)>=2:
            raise ValueError('The single full scientific repair allowance is exhausted')
        for entry in entries:
            root=Path(entry['root'])
            if root.exists() and not (root/'SUITE_FINISHED.json').exists() and not (root/'DEPLOY_FAILED.json').exists():
                # Inspect the deployment audit as well; an active or ambiguous launch is not retried.
                audit=Path(entry['audit'])
                if not (audit/'DEPLOY_FAILED.json').exists():
                    raise ValueError('An earlier repair is still active or unverified')
        source=Path(__file__).resolve().parents[4]
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=args.commit:
            raise ValueError('Repair launcher is not running from the pushed fixed commit')
        suffix='-'+args.teacher.split('_')[0]+'-'+str(len(entries)+1)
        deploy.ROOT=Path(parent['root']+'_repair'+suffix+'-'+args.commit[:8])
        deploy.TARGETS=(args.teacher,);deploy.REPO_SUFFIX=suffix
        deploy.EXTRA_PROTECTED=(prior,)
        deploy.REPAIR_META=dict(original_plan=str(args.parent_plan),original_plan_sha256=deploy.sha(args.parent_plan),
            failed_root=str(prior),diagnosis=evidence,prior_scientific_attempts=actual,
            old_failures_and_claims_preserved=True,maximum_scientific_attempts_per_teacher=2500,
            maximum_scientific_attempts_all_teachers=5000)
        entry=dict(teacher=args.teacher,root=str(deploy.ROOT),commit=args.commit,evidence_sha256=deploy.sha(args.evidence),
                   audit=str(deploy.STORAGE/'audits'/('box-exclusion-'+args.commit[:8]+suffix)),timestamp=deploy.now())
        rows.append(entry);deploy.write(ledger,rows)
    deploy.deploy(args.commit)


if __name__=='__main__':main()
