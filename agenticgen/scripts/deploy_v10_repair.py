"""Local, fixed-purpose GitHub deployment and V10 failed-teacher restart.

Separate from read_remote_status. Executes the reviewed server repair, including
audited deletion of failed Top1 outputs. Normal tool approval still applies.
No arbitrary command, path, host, version, upload or credential-file interface.
"""
import argparse
import base64
import getpass
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import time

from read_remote_status import HOST, PORT, USER, HOST_KEY, ROOT, error_summary

OLD = '/opt/MolSteer/runtimes/agenticgen_d1163b7e'


def command(revision):
    if not re.fullmatch('[0-9a-f]{40}',revision): raise ValueError('Full fixed Git revision required')
    new='/opt/MolSteer/runtimes/agenticgen_'+revision[:8]
    # All executable source is read by the server from the fetched Git object.
    lines=['set -euo pipefail',
        f'git -C {OLD} fetch origin codex/plip-adaptive-schedule >/dev/null 2>&1',
        f'git -C {OLD} cat-file -e {revision}^{{commit}}',
        f'git -C {OLD} show {revision}:agenticgen/scripts/install_fixed_runtime.py | '
            f'{OLD}/agenticgen/.venv/bin/python - --commit {revision} --from-runtime {OLD}',
        'source /opt/MolSteer/scripts/scnet/activate_dtk.sh >/dev/null 2>&1',
        'unset PYTHONPATH',
        f'export AGENTICGEN_REPOSITORY={new} AGENTICGEN_PYTHON={new}/agenticgen/.venv/bin/python',
        f'exec {new}/agenticgen/.venv/bin/python {new}/agenticgen/scripts/restart_v10_output_budget.py --commit {revision}']
    return 'bash -ic '+shlex.quote('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inspect-only',action='store_true',help='Read only this fixed deployment receipts/log tails; execute nothing')
    args=parser.parse_args()
    cmd=command(args.commit)
    if not sys.stdin.isatty(): raise ValueError('Non-echoing terminal authentication required')
    import paramiko
    class PinnedHost(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self,client,hostname,key):
            actual='SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
            if actual!=HOST_KEY: raise ValueError('SSH host fingerprint mismatch')
    client=paramiko.SSHClient();client.set_missing_host_key_policy(PinnedHost())
    password=getpass.getpass('SSH password (not stored): ')
    try:
        client.connect(HOST,port=PORT,username=USER,password=password,look_for_keys=False,allow_agent=False,
                       timeout=25,auth_timeout=25,banner_timeout=25)
        del password
        channel=None
        if not args.inspect_only:
            channel=client.get_transport().open_session(timeout=25);channel.settimeout(25);channel.exec_command(cmd)
            print('Fixed V10 repair command submitted through normal SSH.',flush=True)
        pending=b'';events=[]
        while channel is not None and (not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready()):
            if channel.recv_ready():
                pending+=channel.recv(32768)
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    if line.startswith(b'AGENTICGEN_EVENT '):
                        event=json.loads(line[len(b'AGENTICGEN_EVENT '):]);events.append(event)
                        print(json.dumps(event),flush=True)
                if len(pending)>65536: pending=b''
            if channel.recv_stderr_ready(): channel.recv_stderr(32768)  # No arbitrary shell/env output.
            time.sleep(.1)
        status=channel.recv_exit_status() if channel is not None else None
        sources={};logs={};dep='/opt/MolSteer/operations/agenticgen_deploy_'+args.commit[:8]
        op=ROOT+'/operations/failed_teacher_rerun_v10_'+args.commit[:8]
        with client.open_sftp() as sftp:
            for folder,names in ((dep,('installation.json','api_preflight.json','tests_receipt.json',
                    'stage_receipt.json','finish_receipt.json','aborted_cost_summary.json','approval_receipt.json','launch_receipt.json')),
                    (op,('journal.json','aborted_resource_usage.json','deletion_manifest.json'))):
                for name in names:
                    path=folder+'/'+name
                    try:
                        if sftp.normalize(path)!=path: raise ValueError('Receipt path redirected')
                        with sftp.open(path,'rb') as handle: raw=handle.read(16*1024*1024+1)
                    except FileNotFoundError: continue
                    if len(raw)>16*1024*1024: raise ValueError('Receipt exceeds bound')
                    sources[path]={'sha256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
            if args.inspect_only:
                for name in ('worktree.log','sparse.log','checkout.log','venv.log','install.log',
                             'doctor.log','related-tests.log','native-acceptance.log','launch.log'):
                    path=dep+'/'+name
                    try:
                        if sftp.normalize(path)!=path: raise ValueError('Log path redirected')
                        with sftp.open(path,'rb') as handle:
                            size=handle.stat().st_size;handle.seek(max(0,size-16384));raw=handle.read(16384)
                    except FileNotFoundError: continue
                    logs[name]={'bytes':size,'tail':error_summary({'message':raw.decode('utf-8','replace')[-1200:]})['message']}
        receipt={'commit':args.commit,'exit_code':status,'events':events,'sources':sources,'observed_at':time.time(),
                 'source_transport':'remote GitHub fetch; no source upload','inspect_only':args.inspect_only,'logs':logs}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps({'exit_code':status,'receipt':str(args.output),'source_count':len(sources),'logs':logs}),flush=True)
        return status or 0
    finally: client.close()


if __name__=='__main__': raise SystemExit(main())
