"""Read actual processes, locks, products and budgets; never launch/retry work."""
import datetime
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path):return json.loads(path.read_text()) if path.exists() else None
def main():
    root=Path(sys.argv[1]);config=read(root/'GEOMETRY_CONFIG.json')
    if config is None:raise ValueError('Prepared root required')
    processes=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if any(x in args for x in ['run_uniguide_geometry.py','run_campaign.py','agenticgen.uniguide.quality_queue']) and 'inspect_run.py' not in args:
                processes.append(dict(pid=int(p.name),state=(p/'stat').read_text().split()[2],command=args))
        except (OSError,UnicodeError):pass
    locks={}
    for p in [Path(config['generation_context_root'])/'.executor.lock',root/'.executor.lock',root/'.campaign.lock',root/'quality_queue/.worker.lock']:
        if not p.exists():locks[str(p)]=None;continue
        with p.open('r+') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[str(p)]='free'
            except BlockingIOError:locks[str(p)]='held'
    complete=[str(p.parent.relative_to(root)) for p in root.glob('versions/GD01/*/iteration_*/*/COMPLETE.json')]
    failures=[dict(path=str(p.relative_to(root)),data=read(p)) for pattern in
        ['CAMPAIGN_FAILED.json','calibration/**/INSUFFICIENT_EVIDENCE.json','versions/**/FAILED.json','versions/**/QUALITY_FAILED.json'] for p in root.glob(pattern)]
    status=read(root/'STATUS.json');log=Path(status['log']) if status and 'log' in status else None
    progress=sorted(root.rglob('steps.json'),key=lambda p:p.stat().st_mtime)[-4:]
    info=dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),root=str(root),
        status=status,heartbeat=read(root/'HEARTBEAT.json'),claim=read(root/'CAMPAIGN_CLAIM.json'),
        processes=processes,locks=locks,attempts=read(root/'attempts.json'),
        complete_formal_groups=complete,failures=failures,
        latest_progress=[dict(path=str(p.relative_to(root)),steps=len(rows),last_step=rows[-1]['step'] if rows else None)
            for p in progress for rows in [read(p)]],
        markers={name:read(root/name) for name in ['ACCEPTANCE_FINISHED.json','CALIBRATE_FINISHED.json','FORMAL_FINISHED.json',
            'SAMPLING_FINISHED.json','CAMPAIGN_FINISHED.json','FINALIZED.json','versions/GD01/AWAITING_GPT_REVIEW.json']},
        quality={name:read(root/'quality_queue'/name) for name in ['worker.json','status.json','POLICY.json']},
        quality_request_count=len(list((root/'quality_queue/requests').glob('*.json'))),
        quality_complete=len(list(root.glob('versions/**/QUALITY_COMPLETE.json'))),
        log_tail=log.read_text(errors='replace')[-3000:] if log and log.exists() else None)
    protected=read(root/'SOURCE_PRESERVATION.json')
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    info['protected_files_changed']=[p for p,h in protected.items() if sha(p)!=h]
    info['protected_files_count']=len(protected)
    info['runtime_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[4],text=True).strip()
    out=root/'monitoring';out.mkdir(exist_ok=True)
    destination=out/(datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'.json')
    destination.write_text(json.dumps(info,ensure_ascii=True,indent=2)+'\n')
    print(json.dumps(dict(audit=str(destination),**info),ensure_ascii=True))


if __name__=='__main__':main()
