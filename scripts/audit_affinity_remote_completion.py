"""Post-campaign scoped verification/synchronization; credentials in memory."""
import argparse
import getpass
import json
import select
import shlex
import socket
import subprocess
import threading
from pathlib import Path
import paramiko

p = argparse.ArgumentParser()
p.add_argument('--sync', action='store_true')
p.add_argument('--output', required=True)
a = p.parse_args()
expected = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
ssh.connect('ksai.scnet.cn', port=10544, username='root',
    password=getpass.getpass('SSH password: '), look_for_keys=False,
    allow_agent=False, timeout=25, auth_timeout=25)
ssh.get_transport().set_keepalive(30)

def relay(channel):
    upstream = None
    try:
        upstream = socket.create_connection(('127.0.0.1', 7897), timeout=20)
        while True:
            ready, _, _ = select.select([channel, upstream], [], [], 30)
            for source, dest in ((channel, upstream), (upstream, channel)):
                if source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    dest.sendall(data)
    except OSError:
        pass
    finally:
        if upstream:
            upstream.close()
        channel.close()

def handler(channel, origin, server):
    threading.Thread(target=relay, args=(channel,), daemon=True).start()

repo = '/root/private_data/MolSteer/EvoMolSteer'
remote_code = r'''
import hashlib,json,subprocess
from pathlib import Path
flowr=Path('/root/private_data/MolSteer/flowr_root')
original=flowr/'experiments/ck2_clk3_lineage_20261003'
files=[p for p in original.rglob('*') if p.is_file()]
work=flowr/'experiments/evomolsteer_affinity_geometry30_20261007'
checkpoint=flowr/'checkpoints/flowr_root_v2.ckpt'
sha=hashlib.sha256()
with checkpoint.open('rb') as handle:
    for data in iter(lambda:handle.read(1024*1024),b''):sha.update(data)
head=subprocess.check_output(['git','-C','/root/private_data/MolSteer/EvoMolSteer','rev-parse','HEAD'],text=True).strip()
v={'remote_evomolsteer_head':head,'original_Steer_files':len(files),
   'original_Steer_bytes':sum(p.stat().st_size for p in files),
   'original_Steer_NPZ':sum(p.suffix=='.npz' for p in files),
   'original_trajectory_prefix_entries':len(list(original.rglob('trajectory*'))),
   'baseline_inventory_metric':'trajectory* entries, not NPZ suffix count',
   'checkpoint_sha256':sha.hexdigest(),
   'flowr_tracked_clean':subprocess.run(['git','-C',str(flowr),'diff','--quiet']).returncode==0
      and subprocess.run(['git','-C',str(flowr),'diff','--cached','--quiet']).returncode==0,
   'flowr_head':subprocess.check_output(['git','-C',str(flowr),'rev-parse','HEAD'],text=True).strip(),
   'generated_directory_absent':not (work/'generated').exists(),
   'remaining_transport_archives':[p.name for p in work.glob('*.tar.gz')],
   'remaining_transport_archive_sidecars':[p.name for p in work.glob('*.tar.gz.json')],
   'final_cleanup':json.loads((work/'final_remote_cleanup.json').read_text()),
   'remote_role':'Inference/export only; statistics and design executed locally'}
assert v['original_Steer_files']==2115 and v['original_Steer_bytes']==1076104210 and v['original_trajectory_prefix_entries']==69,v
assert v['checkpoint_sha256']=='f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0',v
assert v['flowr_tracked_clean'] and v['generated_directory_absent'] and not v['remaining_transport_archives'] and not v['remaining_transport_archive_sidecars'],v
print(json.dumps(v))
'''
try:
    if a.sync:
        ssh.get_transport().request_port_forward('127.0.0.1',17897,handler=handler)
    command = 'set -eu\n'
    if a.sync:
        command += 'git -C '+shlex.quote(repo)+' -c http.proxy=http://127.0.0.1:17897 pull --ff-only --quiet\n'
    command += '/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python - <<\'PY\'\n'+remote_code+'\nPY'
    _, stdout, stderr = ssh.exec_command(command, timeout=60)
    raw, errors = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(errors[-4000:])
    value=json.loads(raw)
    assert value['remote_evomolsteer_head']==expected,(expected,value['remote_evomolsteer_head'])
    value.update(local_expected_head=expected, exact_commit_match=True, synced_via_local_VPN=a.sync)
    Path(a.output).write_bytes((json.dumps(value,indent=2)+'\n').encode('utf8'))
    print(json.dumps({k:value[k] for k in ('remote_evomolsteer_head','exact_commit_match',
        'original_Steer_files','original_Steer_bytes','generated_directory_absent')}))
finally:
    ssh.close()
