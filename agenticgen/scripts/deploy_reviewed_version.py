"""Deploy one normally reviewed version from verified official GitHub objects.

Fixed server and result root; no uploads, arbitrary shell, repair/deletion entry,
credential files or environment changes. Platform tool approval still applies.
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

from read_remote_status import HOST, PORT, USER, HOST_KEY, ROOT, collect, error_summary


def validate(version, revision):
    if not re.fullmatch(r'v(?:1[1-9]|2[0-9]|30)', version):
        raise ValueError('Normal continuation only: V11..V30')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Exact fixed commit required')


def command(version, revision, previous_commit, helper_oid, helper_sha256):
    validate(version, revision)
    if not re.fullmatch('[0-9a-f]{40}', previous_commit):
        raise ValueError('Previous installed source must be fixed')
    if not re.fullmatch('[0-9a-f]{40}', helper_oid) or not re.fullmatch('[0-9a-f]{64}', helper_sha256):
        raise ValueError('Committed fetcher digests required')
    old = '/opt/MolSteer/runtimes/agenticgen_'+previous_commit[:8]
    new = '/opt/MolSteer/runtimes/agenticgen_'+revision[:8]
    loader = """import base64,hashlib,json,sys,urllib.request
oid,sha,target=sys.argv[1:]
r=urllib.request.Request('https://api.github.com/repos/H-Dw/EvoMolSteer/git/blobs/'+oid,headers={'User-Agent':'agenticgen-fixed-commit-deployment'})
with urllib.request.urlopen(r,timeout=30) as response:d=json.load(response)
raw=base64.b64decode(d['content'])
if d['sha']!=oid or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\\0'+raw).hexdigest()!=oid:raise ValueError('Fetcher Git object mismatch')
if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Fetcher SHA256 mismatch')
sys.argv=['github-fixed-fetch','--commit',target]
exec(compile(raw,'<hash-verified GitHub fetch helper>','exec'),{'__name__':'__main__'})"""
    lines = ['set -euo pipefail',
        f'test "$(git -C {old} rev-parse HEAD)" = {previous_commit}',
        f'timeout 240 {old}/agenticgen/.venv/bin/python -c {shlex.quote(loader)} {helper_oid} {helper_sha256} {revision}',
        f'git -C {old} cat-file -e {revision}^{{commit}}',
        f'git -C {old} show {revision}:agenticgen/scripts/install_fixed_runtime.py | '
            f'{old}/agenticgen/.venv/bin/python - --commit {revision} --from-runtime {old}',
        'source /opt/MolSteer/scripts/scnet/activate_dtk.sh >/dev/null 2>&1',
        'unset PYTHONPATH',
        f'export AGENTICGEN_REPOSITORY={new} AGENTICGEN_PYTHON={new}/agenticgen/.venv/bin/python',
        f'exec {new}/agenticgen/.venv/bin/python {new}/agenticgen/scripts/start_reviewed_version.py '
            f'--version {version} --commit {revision}']
    return 'bash -ic '+shlex.quote('\n'.join(lines))


def read_source(sftp, path):
    if sftp.normalize(path) != path:
        raise ValueError('Source path redirected')
    with sftp.open(path, 'rb') as handle:
        raw = handle.read(8*1024*1024+1)
    if len(raw) > 8*1024*1024:
        raise ValueError('Source size exceeded')
    return {'sha256':hashlib.sha256(raw).hexdigest(), 'value':json.loads(raw)}


def ensure_unstarted(sftp, version):
    for path in (ROOT+'/versions/'+version, ROOT+'/FINALIZED_'+version+'.json'):
        try:
            sftp.stat(path)
        except FileNotFoundError:
            continue
        raise ValueError('Version already staged or completed; use inspect-only, never relaunch')
    previous = 'v'+str(int(version[1:])-1)
    final = read_source(sftp, ROOT+'/FINALIZED_'+previous+'.json')['value']
    launch = read_source(sftp, ROOT+'/launch.json')['value']
    if final.get('outcomes') != 1000 or final.get('rounds') != 5 or launch['version'] != previous:
        raise ValueError('Predecessor completion/launch differs')
    pid = launch['pid']
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError('Invalid worker identity')
    try:
        sftp.stat('/proc/'+str(pid))
    except FileNotFoundError:
        return launch['commit']
    raise ValueError('Previous process still exists; no deployment')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inspect-only',action='store_true',help='Read fixed deployment receipts and log tails, execute nothing')
    args=parser.parse_args();validate(args.version,args.commit)
    if not sys.stdin.isatty():raise ValueError('Non-echoing terminal authentication required')
    repo=Path(__file__).resolve().parents[2]
    helper_path='agenticgen/scripts/fetch_fixed_github_api.py'
    helper=subprocess.check_output(['git','-C',str(repo),'show',args.commit+':'+helper_path])
    helper_oid=hashlib.sha1(b'blob '+str(len(helper)).encode()+b'\0'+helper).hexdigest()
    helper_sha=hashlib.sha256(helper).hexdigest()
    import paramiko
    class PinnedHost(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self,client,hostname,key):
            actual='SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
            if actual!=HOST_KEY:raise ValueError('SSH host fingerprint mismatch')
    client=paramiko.SSHClient();client.set_missing_host_key_policy(PinnedHost())
    secret=getpass.getpass('SSH password (not stored): ')
    try:
        client.connect(HOST,port=PORT,username=USER,password=secret,look_for_keys=False,allow_agent=False,
                       timeout=25,auth_timeout=25,banner_timeout=25)
        del secret
        status=None;errors=[];channel=None
        if not args.inspect_only:
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(25)
                previous=ensure_unstarted(sftp,args.version)
            cmd=command(args.version,args.commit,previous,helper_oid,helper_sha)
            channel=client.get_transport().open_session(timeout=25);channel.settimeout(25)
            channel.exec_command(cmd)
            print('Fixed normal-version deployment submitted; official GitHub API source, no upload.',flush=True)
            stderr=b'';started=time.monotonic()
            while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                if channel.recv_ready():channel.recv(32768)  # Never display uncontrolled remote output.
                if channel.recv_stderr_ready():stderr=(stderr+channel.recv_stderr(32768))[-16384:]
                if time.monotonic()-started>1200:
                    channel.close()
                    raise TimeoutError('Deployment outcome uncertain; inspect receipts before any retry')
                time.sleep(.1)
            status=channel.recv_exit_status()
            errors=[error_summary({'message':line})['message'] for line in stderr.decode('utf8','replace').splitlines()
                    if line.startswith(('fatal:','error:','RuntimeError:','ValueError:','FileNotFoundError:'))][-5:]
        sources={};logs={};health=None
        dep='/opt/MolSteer/operations/agenticgen_deploy_'+args.commit[:8]
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(25)
            for name in ('github_object_receipt.json','installation.json','tests_receipt.json',
                         'approval_receipt.json','launch_receipt.json'):
                path=dep+'/'+name
                try:sources[path]=read_source(sftp,path)
                except FileNotFoundError:continue
            if args.inspect_only or status:
                for name in ('worktree.log','checkout.log','install.log','doctor.log','related-tests.log','launch.log'):
                    path=dep+'/'+name
                    try:
                        if sftp.normalize(path)!=path:raise ValueError('Log path redirected')
                        with sftp.open(path,'rb') as handle:
                            size=handle.stat().st_size;handle.seek(max(0,size-2048));raw=handle.read(2048)
                    except FileNotFoundError:continue
                    logs[name]=error_summary({'message':raw.decode('utf8','replace')})['message']
            if dep+'/launch_receipt.json' in sources:
                health=collect(sftp,args.version)
                if health['launch']['commit']!=args.commit:raise ValueError('Another source launched; inspect actual identity')
        receipt={'commit':args.commit,'version':args.version,'observed_at':time.time(),'exit_code':status,
            'source_transport':'official GitHub API with Git object and fetcher hash verification',
            'inspect_only':args.inspect_only,'sources':sources,'logs':logs,'errors':errors,'startup_health':health}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
        print(json.dumps({'exit_code':status,'receipt':str(args.output),'sources':len(sources),'errors':errors}),flush=True)
        return status or 0
    finally:client.close()


if __name__=='__main__':raise SystemExit(main())
