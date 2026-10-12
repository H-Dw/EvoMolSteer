"""Read-only remote runtime capture; no inference or workflow mutation.

Credentials use MOLSTEER_SCNET_PASSWORD. Only inference logs and selected
process counters are collected; process environments are never read.
"""
import argparse
import gzip
import json
import shlex
import time
from pathlib import Path

from run_m2_skill_api_study import Study, shell
from evomolsteer.ablation.collection import capture_state


REMOTE = r'''
import json, os, pathlib, shutil, subprocess, time
work=pathlib.Path(WORK)
def read_json(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
def processes():
 allp={}
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   stat=(p/'stat').read_text(); tail=stat[stat.rfind(')')+2:].split()
   argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   cmd=' '.join(argv).strip()
   devices=[]
   for fd in (p/'fd').iterdir():
    try:
     link=os.readlink(fd)
     if link.startswith('/dev/dri/') and link not in devices:devices.append(link)
    except OSError:pass
   allp[int(p.name)]={'pid':int(p.name),'ppid':int(tail[1]),'state':tail[0],
     'cpu_seconds':(int(tail[11])+int(tail[12]))/os.sysconf('SC_CLK_TCK'),
     'rss_bytes':int(tail[21])*os.sysconf('SC_PAGE_SIZE'),'command':cmd,
     'is_cohort_driver':any(pathlib.Path(arg).name=='run_dependency_cohort.py' for arg in argv if len(arg)<4096),
     'wchan':(p/'wchan').read_text().strip(),'drm_devices':devices,
     'io':dict(line.split(': ',1) for line in (p/'io').read_text().splitlines())}
  except (OSError,ValueError,IndexError):pass
 selected={pid for pid,p in allp.items() if str(work) in p['command'] and p['is_cohort_driver']}
 for _ in range(5):selected|={pid for pid,p in allp.items() if p['ppid'] in selected}
 return [allp[pid] for pid in sorted(selected)]
def sample():
 gpu={}
 for card in pathlib.Path('/sys/class/drm').glob('card[0-9]*'):
  if '-' in card.name:continue
  vals={}
  for field in ['gpu_busy_percent','mem_info_vram_used','mem_info_vram_total','vendor','device']:
   try:vals[field]=(card/'device'/field).read_text().strip()
   except OSError:pass
  for hw in (card/'device/hwmon').glob('hwmon*'):
   for field in ['name','temp1_input','power1_average']:
    try:vals[field]=(hw/field).read_text().strip()
    except OSError:pass
  if vals:gpu[card.name]=vals
 progress={}
 for p in (work/'generated/results').glob('*/progress.json'):
  progress[str(p.relative_to(work))]={'mtime_unix':p.stat().st_mtime,'value':read_json(p)}
 return {'unix':time.time(),'processes':processes(),'gpu_sysfs':gpu,'progress':progress,
   'loadavg':pathlib.Path('/proc/loadavg').read_text().strip(),
   'meminfo':{k:v for k,v in (line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']}}
result={'observed_unix':time.time(),'work':str(work),'samples':[sample()],
 'cpu_count':os.cpu_count(),'statuses':{},'logs':{},'batch_artifacts':[]}
for p in sorted(work.glob('*.status.json')):result['statuses'][p.stem.removesuffix('.status')]=read_json(p)
for p in sorted(work.glob('*.log')):
 if p.stat().st_size>200000:result['logs'][p.name]={'bytes':p.stat().st_size,'tail':p.read_bytes()[-30000:].decode(errors='replace')}
 else:result['logs'][p.name]={'bytes':p.stat().st_size,'text':p.read_text(errors='replace')}
for p in sorted((work/'generated/results').rglob('*')):
 if p.is_file() and (p.name in ['manifest.json','learning_manifest.json','execution_state.npz','terminal_records.json','COMPLETE.json','runtime.json']):
  result['batch_artifacts'].append({'path':str(p.relative_to(work)),'mtime_unix':p.stat().st_mtime,'bytes':p.stat().st_size})
result['live_storage_manifests']={}
for p in (work/'generated/results').rglob('*manifest.json'):
 if p.stat().st_size<2000000:result['live_storage_manifests'][str(p.relative_to(work))]=read_json(p)
source=pathlib.Path(FLOWR)/'flowr/models/fm_pocket.py'
lines=source.read_text().splitlines();start=next((i for i,line in enumerate(lines) if 'def _generate_selective(' in line),None)
result['native_generation_source']={'path':str(source),'first_line':start+1 if start is not None else None,
 'text':'\n'.join(lines[start:start+360]) if start is not None else None}
sv=os.statvfs(work);result['free_bytes']=sv.f_bavail*sv.f_frsize
for p in ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us']:
 try:result.setdefault('cgroup_limits',{})[p]=pathlib.Path(p).read_text().strip()
 except OSError:pass
for binary in ['rocm-smi','nvidia-smi','hy-smi']:
 resolved=shutil.which(binary)
 if not resolved:
  resolved=next((str(p) for base in ['/opt/dtk','/opt/rocm'] for p in pathlib.Path(base).glob('*/'+binary)),None)
 if resolved:
  try:
   cmd=[resolved]+(['--showuse','--showmemuse','--showproductname','--json'] if binary=='rocm-smi' else [])
   proc=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
   result.setdefault('gpu_tools',{})[binary]={'exit':proc.returncode,'stdout':proc.stdout[:16000],'stderr':proc.stderr[:2000]}
  except (OSError,subprocess.TimeoutExpired) as e:result.setdefault('gpu_tools',{})[binary]={'error':type(e).__name__}
time.sleep(SAMPLE_SECONDS)
result['samples'].append(sample())
result['captured_unix']=time.time()
print(json.dumps(result))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sample-seconds', type=float, default=12)
    a = p.parse_args()
    if not 1 <= a.sample_seconds <= 20:
        p.error('sample interval must be between 1 and 20 seconds')
    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads((repo / a.config).read_text(encoding='utf-8'))
    destination = repo / a.output
    destination.mkdir(parents=True, exist_ok=True)
    raw, state = capture_state(repo / cfg['study'] / 'run_status.json')
    (destination / 'local_state_snapshot.json.gz').write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    study = Study(repo, cfg)
    code = ('WORK=' + repr(cfg['remote_work']) + '\nFLOWR=' + repr(cfg['flowr']) +
            '\nSAMPLE_SECONDS=' + repr(a.sample_seconds) + '\n' + REMOTE)
    start = time.time()
    with study.ssh() as client:
        result = json.loads(shell(client, shlex.join([cfg['remote_python'], '-c', code])))
    result['capture_roundtrip_seconds'] = time.time() - start
    serialized = (json.dumps(result, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    (destination / 'remote_runtime.json.gz').write_bytes(gzip.compress(serialized, compresslevel=6, mtime=0))
    print(json.dumps({'output':str(destination),'task':state.get('task'),'remote_samples':len(result['samples']),
        'free_GiB':result['free_bytes']/2**30,'remote_statuses':len(result['statuses'])},ensure_ascii=False))


if __name__ == '__main__':
    main()
