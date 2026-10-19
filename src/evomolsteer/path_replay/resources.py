"""Read the GPU visible to FLOWR, not unrelated host DRM devices."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def parse_vram(value,device=0):
    key='card'+str(device)
    row=value.get(key)
    if not isinstance(row,dict): raise ValueError('Target device absent from SMI result')
    found={}
    for label,number in row.items():
        text=label.lower()
        if 'vram' not in text or 'memory' not in text: continue
        factor=1024**2 if '(mib)' in text else 1 if '(b)' in text or '(bytes)' in text else None
        if factor is None: continue
        if 'total used' in text or 'used memory' in text: found['used_bytes']=int(number)*factor
        elif 'total memory' in text: found['total_bytes']=int(number)*factor
    total=found.get('total_bytes',0);used=found.get('used_bytes',-1)
    if not 0<=used<=total or total<=0: raise ValueError('Invalid/missing SMI VRAM capacity')
    return dict(found,percent=100*used/total,device=key,pci_bus=row.get('PCI Bus'),raw=row)


def visible_gpu_memory(device=0):
    candidates=['/opt/dtk/bin/rocm-smi','/opt/hyhal/bin/hy-smi',shutil.which('rocm-smi')]
    errors=[]
    for path in dict.fromkeys(p for p in candidates if p and Path(p).is_file()):
        cmd=[path,'--showmeminfo','vram','--showbus','--json']
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=30,check=True)
            return dict(parse_vram(json.loads(p.stdout),device),command=cmd,unix=time.time())
        except (OSError,ValueError,subprocess.SubprocessError) as e: errors.append(str(e))
    raise RuntimeError('Cannot verify target VRAM: '+'; '.join(errors))


def replay_processes():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try:
            args=(p/'cmdline').read_bytes().decode(errors='replace').split('\x00')
            # Inspect argv tokens, avoiding false positives from SSH bash code.
            if any(a.startswith(('evomolsteer.path_replay','evomolsteer.elite_alignment')) for a in args):
                rows.append({'pid':int(p.name),'argv':args})
        except OSError: pass
    return rows


def resource_gate(work,threshold=10.):
    memory=visible_gpu_memory();processes=replay_processes()
    available=None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): available=int(line.split()[1])*1024
    result={'memory':memory,'replay_processes':processes,'available_host_bytes':available,
            'disk_free_bytes':shutil.disk_usage(work).free,'threshold_percent':threshold}
    result['passed']=memory['percent']<threshold and not processes and available is not None and available>=96*1024**3 and result['disk_free_bytes']>=256*1024**2
    return result
