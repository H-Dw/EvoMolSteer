"""Local, fixed-purpose GitHub deployment and V21 failed-teacher restart.

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
import subprocess
import sys
import time

from read_remote_status import HOST, PORT, USER, HOST_KEY, ROOT, error_summary, collect

PREVIOUS_COMMIT = '4c4aa68fc9a04484dcdafd47bf21073c9171af24'


def command(revision, bootstrap_oid, bootstrap_sha256):
    from deploy_reviewed_version import command as reviewed_command
    # Reuse the verified API/install prefix, replacing only the fixed final entry.
    normal = reviewed_command('v21',revision,PREVIOUS_COMMIT,bootstrap_oid,bootstrap_sha256)
    shell, option, script = shlex.split(normal)
    expected = 'start_reviewed_version.py --version v21 --commit '+revision
    if shell != 'bash' or option != '-ic' or not script.endswith(expected):
        raise ValueError('Fixed deployment entry changed; review before use')
    script = script[:-len(expected)]+'restart_v21_commitments.py --commit '+revision
    return 'bash -ic '+shlex.quote(script)


def ensure_failed_scope(sftp, revision):
    state=collect(sftp,'v21')
    phases={k:v['phase'] for k,v in state['teachers'].items()}
    if (state['launch']['commit']!=PREVIOUS_COMMIT or state['process']['exists']
            or state['completion_verified'] or state['state']!='pipeline_failed'
            or state['teachers']['5nn4_SC2_A_1017__top1']['round'] != 5
            or 'Fewer than four unexecuted proposed schedules' not in str(state['teachers']['5nn4_SC2_A_1017__top1']['error'])
            or phases!={'5nn4_SC2_A_1017__top1':'failed','5nn4_SC2_A_1017__top2':'complete'}):
        raise ValueError('V21 failure scope changed; inspect, never relaunch')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inspect-only',action='store_true',help='Read only this fixed deployment receipts/log tails; execute nothing')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2];relative='agenticgen/scripts/fetch_fixed_github_api.py'
    raw=subprocess.check_output(['git','-C',str(repo),'show',args.commit+':'+relative])
    cmd=command(args.commit,hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),hashlib.sha256(raw).hexdigest())
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
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(25)
                ensure_failed_scope(sftp,args.commit)
            channel=client.get_transport().open_session(timeout=25);channel.settimeout(25);channel.exec_command(cmd)
            print('Fixed V21 repair command submitted through normal SSH.',flush=True)
        pending=b'';events=[];stderr_tail=b'';started=time.monotonic()
        while channel is not None and (not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready()):
            if time.monotonic()-started>1500:
                channel.close();raise TimeoutError('Outcome uncertain; inspect receipts before retry')
            if channel.recv_ready():
                pending+=channel.recv(32768)
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    if line.startswith(b'AGENTICGEN_EVENT '):
                        event=json.loads(line[len(b'AGENTICGEN_EVENT '):]);events.append(event)
                        print(json.dumps(event),flush=True)
                if len(pending)>65536: pending=b''
            if channel.recv_stderr_ready(): stderr_tail=(stderr_tail+channel.recv_stderr(32768))[-16384:]
            time.sleep(.1)
        status=channel.recv_exit_status() if channel is not None else None
        errors=[error_summary({'message':line})['message'] for line in stderr_tail.decode('utf-8','replace').splitlines()
                if line.startswith(('fatal:','error:','RuntimeError:','ValueError:','FileNotFoundError:'))][-5:]
        sources={};logs={};health=None;dep='/opt/MolSteer/operations/agenticgen_deploy_'+args.commit[:8]
        op=ROOT+'/operations/failed_teacher_rerun_v21_'+args.commit[:8]
        protocol=ROOT+'/protocols/analysis_output_repair_v21_'+args.commit[:8]
        probe=ROOT+'/operations/teacher_restart_acceptance_'+args.commit[:8]
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(25)
            for folder,names in ((dep,('github_object_receipt.json','installation.json','api_preflight.json','tests_receipt.json',
                    'stage_receipt.json','native_reuse_receipt.json','finish_receipt.json','aborted_cost_summary.json','approval_receipt.json','launch_receipt.json')),
                    (op,('journal.json','aborted_resource_usage.json','deletion_manifest.json')),
                    (protocol,('config.json','PREPARED.json','ACCEPTANCE_COMPLETE.json','validation/gradient_checks.json')),
                    (probe,('results.json',))):
                for name in names:
                    path=folder+'/'+name
                    try:
                        if sftp.normalize(path)!=path: raise ValueError('Receipt path redirected')
                        with sftp.open(path,'rb') as handle: raw=handle.read(16*1024*1024+1)
                    except FileNotFoundError: continue
                    if len(raw)>16*1024*1024: raise ValueError('Receipt exceeds bound')
                    sources[path]={'sha256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
            if args.inspect_only or status:
                for name in ('worktree.log','sparse.log','checkout.log','venv.log','install.log',
                             'doctor.log','related-tests.log','native-acceptance.log','launch.log'):
                    path=dep+'/'+name
                    try:
                        if sftp.normalize(path)!=path: raise ValueError('Log path redirected')
                        with sftp.open(path,'rb') as handle:
                            size=handle.stat().st_size;handle.seek(max(0,size-16384));raw=handle.read(16384)
                    except FileNotFoundError: continue
                    logs[name]={'bytes':size,'tail':error_summary({'message':raw.decode('utf-8','replace')[-1200:]})['message']}
            if dep+'/launch_receipt.json' in sources:
                health=collect(sftp,'v21')
                if health['launch']['commit']!=args.commit:raise ValueError('Another runtime launched; inspect actual identity')
        receipt={'commit':args.commit,'exit_code':status,'events':events,'sources':sources,'observed_at':time.time(),
                 'source_transport':'official GitHub API verified fixed objects; no source upload','inspect_only':args.inspect_only,'logs':logs,
                 'errors':errors,'startup_health':health}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps({'exit_code':status,'receipt':str(args.output),'source_count':len(sources),'logs':logs,'errors':errors}),flush=True)
        return status or 0
    finally: client.close()


if __name__=='__main__': raise SystemExit(main())
