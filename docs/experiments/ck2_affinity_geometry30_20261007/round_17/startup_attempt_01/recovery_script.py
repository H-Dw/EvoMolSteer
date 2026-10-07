"""One explicitly scoped recovery; no generated structures are removed here."""
import getpass,json,paramiko,shlex
from pathlib import Path

root=next(p for p in Path(__file__).resolve().parents if (p/'src/evomolsteer').is_dir() and (p/'configs').is_dir())
e=root/'docs/experiments/ck2_affinity_geometry30_20261007/round_17/startup_attempt_01'
command="""import json,os,hashlib
from pathlib import Path
w=Path('/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_affinity_geometry30_20261007').resolve()
n=17;c='coordinate_r17_affinity30';a=w/'round17.startup_attempt_01'
assert w.name=='evomolsteer_affinity_geometry30_20261007'
assert (w/'round17.exit').read_text().strip()=='1'
assert 'Learning reference does not cover every controlled state' in (w/'round17.log').read_text()
assert not (w/'generated/results'/c).exists(), 'Generated round output must be inspected instead of retried'
assert not any((w/(c+s)).exists() for s in ('.tar.gz','.tar.gz.json'))
m=json.loads((w/'round_ready.json').read_text());assert m['round']==n and m['campaigns']==[c]
launch=json.loads((w/'round17.dispatch/launch.json').read_text());assert launch['round']==n and launch['campaign']==c
repo=Path('/root/private_data/MolSteer/EvoMolSteer');cfg=repo/'configs/experiments/ck2_affinity_geometry30_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(cfg/'backtrack_round17.json')==launch['program_sha256']
assert sha(cfg/'terminal_reference.json.gz')==launch['reference_sha256']
proc=Path('/proc')/str(launch['pid'])/'cmdline'
assert not proc.exists() or not proc.read_bytes(), 'Old inference still running'
files=['round17.dispatch','round17.log','round17.exit','cleanup_before_round17.json','cleanup_before_round17.plan.json']
assert all((w/name).exists() for name in files), 'Incomplete startup evidence'
a.mkdir()
(a/'round_ready_before.json').write_bytes((w/'round_ready.json').read_bytes())
for name in files:
    p=w/name
    assert p.parent==w
    os.replace(p,a/name)
m.update(round=16,status='ready',campaigns=['coordinate_r16_affinity30'],completed_rounds=16,cleanup_audit=str(w/'cleanup_before_round16.json'))
(w/'round_ready.json').write_text(json.dumps(m,indent=2))
report={'status':'same_round_retry_ready','round':17,'failed_inference_commit':launch['inference_commit'],
        'unchanged_program_sha256':launch['program_sha256'],'unchanged_reference_sha256':launch['reference_sha256'],
        'generated_round_output_exists':False,'structures_deleted':0,'archive_directory':str(a),
        'preserved_files':[{ 'path':str(p.relative_to(a)), 'bytes':p.stat().st_size,
          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(a.rglob('*')) if p.is_file()]}
(a/'recovery.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
"""
if __name__=='__main__':
    s=paramiko.SSHClient();s.load_system_host_keys();s.set_missing_host_key_policy(paramiko.RejectPolicy())
    s.connect('ksai.scnet.cn',port=10544,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
    try:
        _,out,err=s.exec_command('/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python -c '+shlex.quote(command),timeout=30)
        text=out.read().decode();error=err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(error)
        report=json.loads(text);(e/'remote_recovery.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        with s.open_sftp() as f:
            for item in report['preserved_files']:
                p=e/'remote'/item['path'];p.parent.mkdir(parents=True,exist_ok=True)
                f.get(report['archive_directory']+'/'+item['path'],str(p))
        p=root/'configs/experiments/ck2_affinity_geometry30_v1/campaign.json';c=json.loads(p.read_text())
        assert c['rounds_completed']==16 and c['rounds_started']==17 and c['rounds'][-1]['round']==17
        c['rounds'][-1].update(status='frozen',startup_attempts_failed=1)
        p.write_text(json.dumps(c,indent=2)+'\n',encoding='utf8');print(text)
    finally:s.close()
