"""Deploy one reviewed version with bounded Git pack transfer and fixed checkout.

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

# Ordered, explicitly user-authorized mirrors. One invocation attempts one source.
MIRRORS = {
    'ghfast': 'https://ghfast.top/',
    'gh-proxy': 'https://gh-proxy.com/',
    'ghproxy': 'https://ghproxy.net/',
    'gh-llkk': 'https://gh.llkk.cc/',
}


def validate(version, revision):
    if not re.fullmatch(r'v(?:1[1-9]|2[0-9]|30)', version):
        raise ValueError('Normal continuation only: V11..V30')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Exact fixed commit required')


def command(version, revision, previous_commit, transport_commit, validation_hashes, source_transport="github"):
    validate(version, revision)
    if source_transport not in ("github", *MIRRORS):
        raise ValueError("Only the fixed GitHub and user-authorized mirror sources are allowed")
    for value in (previous_commit, transport_commit):
        if not re.fullmatch('[0-9a-f]{40}', value):
            raise ValueError('Scientific and transport sources must be fixed commits')
    expected = ('agenticgen/scripts/deploy_reviewed_version.py',
                'agenticgen/scripts/read_remote_status.py',
                'agenticgen/tests/unit/test_reviewed_deploy.py')
    if set(validation_hashes) != set(expected) or any(
            not re.fullmatch('[0-9a-f]{64}', x) for x in validation_hashes.values()):
        raise ValueError('Committed transport checks require exact source hashes')
    old = '/opt/MolSteer/runtimes/agenticgen_'+previous_commit[:8]
    new = '/opt/MolSteer/runtimes/agenticgen_'+revision[:8]
    dep = '/opt/MolSteer/operations/agenticgen_deploy_'+revision[:8]
    # This control snippet extracts committed files from Git; it uploads no sources.
    verification = """import hashlib,json,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
old,dep,target,transport,encoded,status,source,clone_status,import_status,clone_attempted=sys.argv[1:]
dep=Path(dep);status=int(status)
receipt={'commit':target,'transport_commit':transport,'time':time.time(),
 'source':source,'fetch_calls':int(source=='github'),'clone_calls':int(clone_attempted),
 'local_import_calls':int(import_status!='not_run'),'clone_exit_code':int(clone_status) if clone_status!='not_run' else None,
 'local_import_exit_code':int(import_status) if import_status!='not_run' else None,'rest_api_calls':0,
 'fetch_exit_code':status,'proxy_environment':'inherited; values omitted'}
(dep/'git_fetch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
if status:raise SystemExit(status)
def git(*args):return subprocess.check_output(['git','-c','remote.origin.promisor=false','-C',old,*args])
for rev in (target,transport):
 if git('rev-parse',rev+'^{commit}').decode().strip()!=rev:raise ValueError('Fixed commit missing')
 subprocess.run(['git','-C',old,'merge-base','--is-ancestor',rev,'FETCH_HEAD'],check=True)
checks=dep/('transport_'+transport[:8]);checks.mkdir(exist_ok=True)
for path,digest in json.loads(encoded).items():
 raw=git('show',transport+':'+path)
 if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Transport source hash mismatch')
 dest=checks/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
# Confirm deployment blobs exist locally before checkout; never hydrate via API.
entries=git('ls-tree','-r',target,'agenticgen/','src/evomolsteer/generation/',
 'src/evomolsteer/adaptive_schedule/vendor/').decode().splitlines()
if not entries:raise ValueError('Deployment tree missing')
objects=[line.split()[2] for line in entries]
proc=subprocess.run(['git','-c','remote.origin.promisor=false','-C',old,'cat-file','--batch-check'],
 input=('\\n'.join(objects)+'\\n').encode(),stdout=subprocess.PIPE,check=True)
if any(b' missing' in line for line in proc.stdout.splitlines()):raise ValueError('Deployment object missing after single fetch')
identity=hashlib.sha256(json.dumps(json.loads(encoded),sort_keys=True).encode()).hexdigest()
cache=dep.parent/('agenticgen_git_transport_validation_'+identity+'.json')
prior=json.loads(cache.read_text()) if cache.exists() else None
reuse=prior is not None and prior.get('passed') is True and prior.get('source_sha256')==json.loads(encoded)
if reuse:
 saved=Path(prior['junit_path'])
 reuse=saved.is_file() and hashlib.sha256(saved.read_bytes()).hexdigest()==prior['junit_sha256']
if reuse:
 validation={**prior,'commit':target,'transport_commit':transport,'reused':True,'fresh_tests_run':False}
else:
 junit=checks/'tests.xml'
 with (checks/'tests.log').open('wb') as log:
  code=subprocess.run([sys.executable,'-m','pytest','-q',str(checks/'agenticgen/tests/unit/test_reviewed_deploy.py'),
  '--junitxml='+str(junit)],cwd='/tmp',stdout=log,stderr=subprocess.STDOUT).returncode
 suites=list(ET.parse(junit).getroot().iter('testsuite')) if junit.exists() else []
 passed=code==0 and bool(suites) and sum(int(x.get('tests','0')) for x in suites)>0 and not any(
  int(x.get(k,'0')) for x in suites for k in ('failures','errors','skipped'))
 validation={'commit':target,'transport_commit':transport,'passed':passed,'exit_code':code,
  'source_sha256':json.loads(encoded),'junit_sha256':hashlib.sha256(junit.read_bytes()).hexdigest() if junit.exists() else None,
  'junit_path':str(junit),'local_tests_run':False,'reused':False,'fresh_tests_run':True,
  'remote_test_files':['agenticgen/tests/unit/test_reviewed_deploy.py']}
 if passed:cache.write_text(json.dumps(validation,indent=2)+'\\n')
(dep/'transport_validation.json').write_text(json.dumps(validation,indent=2)+'\\n')
if not validation['passed']:raise ValueError('Scoped transport checks failed; no installation or launch')
receipt.update(verified_commit=target,verified_transport_commit=transport,verified_deployment_blobs=len(objects))
(dep/'git_fetch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
"""
    encoded = shlex.quote(json.dumps(validation_hashes))
    lines = ['set -euo pipefail',
        f'test "$(git -C {old} rev-parse HEAD)" = {previous_commit}',
        'export GIT_TERMINAL_PROMPT=0 GIT_NO_LAZY_FETCH=1',
        f'mkdir -p {dep}',
        'fetch_status=0; clone_status=not_run; import_status=not_run; clone_attempted=0']
    if source_transport == 'github':
        lines += [f'timeout 120 git -c fetch.fsckObjects=true -C {old} fetch --no-tags '
            f'--no-recurse-submodules --no-filter https://github.com/H-Dw/EvoMolSteer.git '
            f'refs/heads/codex/plip-adaptive-schedule > {dep}/git-fetch.log 2>&1 || fetch_status=$?']
    else:
        # Never overwrite a previous clone or change an existing runtime origin.
        mirror = '/opt/MolSteer/source_cache/'+source_transport+'_'+transport_commit[:8]+'.git'
        mirror_url = MIRRORS[source_transport]+'https://github.com/H-Dw/EvoMolSteer.git'
        lines += [f'mkdir -p /opt/MolSteer/source_cache',
            f'if test -e {mirror}; then echo "Clone destination exists; inspect before retry" > {dep}/{source_transport}-clone.log; clone_status=73; else',
            'clone_status=0; clone_attempted=1',
            'env -u GIT_CONFIG_COUNT -u GIT_CONFIG_PARAMETERS GIT_CONFIG_NOSYSTEM=1 '
            'GIT_CONFIG_GLOBAL=/dev/null GIT_ASKPASS=/bin/false SSH_ASKPASS=/bin/false '
            'timeout 120 git -c credential.helper= -c http.extraHeader= -c transfer.fsckObjects=true '
            'clone --bare --single-branch --branch codex/plip-adaptive-schedule --no-tags '
            f'{mirror_url} {mirror} '
            f'> {dep}/{source_transport}-clone.log 2>&1 || clone_status=$?',
            'fi', 'fetch_status=$clone_status',
            'if test "$clone_status" = 0; then', 'import_status=0',
            f'timeout 120 git -c fetch.fsckObjects=true -C {old} fetch --no-tags '
            f'--no-recurse-submodules --no-filter {mirror} refs/heads/codex/plip-adaptive-schedule '
            f'> {dep}/git-local-import.log 2>&1 || import_status=$?',
            'fetch_status=$import_status', 'fi']
    lines += [f'{old}/agenticgen/.venv/bin/python -c {shlex.quote(verification)} '
        f'{old} {dep} {revision} {transport_commit} {encoded} "$fetch_status" '
        f'{source_transport} "$clone_status" "$import_status" "$clone_attempted"',
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
    parser.add_argument('--transport-commit', help='Fixed committed Git transport/check source; defaults to local HEAD. Scientific --commit stays unchanged.')
    parser.add_argument('--source-transport', choices=('github',*MIRRORS), default='github',
                        help='After GitHub failure, try authorized mirrors in listed order, once each; stop on success or validation failure.')
    parser.add_argument('--inspect-only',action='store_true',help='Read fixed deployment receipts and log tails, execute nothing')
    args=parser.parse_args();validate(args.version,args.commit)
    if not sys.stdin.isatty():raise ValueError('Non-echoing terminal authentication required')
    repo=Path(__file__).resolve().parents[2]
    transport = args.transport_commit or subprocess.check_output(
        ['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}',transport):raise ValueError('Fixed transport commit required')
    validation_hashes={}
    for path in ('agenticgen/scripts/deploy_reviewed_version.py',
                 'agenticgen/scripts/read_remote_status.py',
                 'agenticgen/tests/unit/test_reviewed_deploy.py'):
        raw=subprocess.check_output(['git','-C',str(repo),'show',transport+':'+path])
        if (repo/path).read_bytes()!=raw:raise ValueError('Local transport/check source differs from fixed commit')
        validation_hashes[path]=hashlib.sha256(raw).hexdigest()
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
            cmd=command(args.version,args.commit,previous,transport,validation_hashes,args.source_transport)
            channel=client.get_transport().open_session(timeout=25);channel.settimeout(25)
            channel.exec_command(cmd)
            print('Fixed deployment submitted; bounded '+args.source_transport+' Git transfer, no REST API or upload.',flush=True)
            stderr=b'';started=time.monotonic()
            while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                if channel.recv_ready():channel.recv(32768)  # Never display uncontrolled remote output.
                if channel.recv_stderr_ready():stderr=(stderr+channel.recv_stderr(32768))[-16384:]
                if time.monotonic()-started>1200:
                    channel.close()
                    raise TimeoutError('Deployment outcome uncertain; inspect receipts before any retry')
                time.sleep(.1)
            status=channel.recv_exit_status()
            # Keep exception summaries, not shell/profile output or environment values.
            errors=[error_summary({'message':line})['message'] for line in stderr.decode('utf8','replace').splitlines()
                    if re.match(r'^(?:fatal|error|[\w.]+(?:Error|Exception)):',line)][-5:]
        sources={};logs={};health=None
        dep='/opt/MolSteer/operations/agenticgen_deploy_'+args.commit[:8]
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(25)
            for name in ('github_object_receipt.json','installation.json','tests_receipt.json',
                         'approval_receipt.json','launch_receipt.json','fetcher_validation.json',
                         'git_fetch_receipt.json','transport_validation.json'):
                path=dep+'/'+name
                try:sources[path]=read_source(sftp,path)
                except FileNotFoundError:continue
            if args.inspect_only or status:
                for name in ('git-fetch.log',*(name+'-clone.log' for name in MIRRORS),'git-local-import.log','worktree.log','checkout.log','install.log','doctor.log','related-tests.log','launch.log'):
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
            'source_transport':args.source_transport+' bounded Git pack transfer; fixed commit and source hashes; no REST API',
            'transport_commit':transport,
            'inspect_only':args.inspect_only,'sources':sources,'logs':logs,'errors':errors,'startup_health':health}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
        print(json.dumps({'exit_code':status,'receipt':str(args.output),'sources':len(sources),'errors':errors}),flush=True)
        return status or 0
    finally:client.close()


if __name__=='__main__':raise SystemExit(main())
