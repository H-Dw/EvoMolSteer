"""Validate fixed deployment routing without SSH, credentials or remote mutations."""
import importlib.util
from pathlib import Path
import sys
import pytest

scripts=Path(__file__).parents[2]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('reviewed_deploy',scripts/'deploy_reviewed_version.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('version',['v10','v31','V11','v011','../v11','v11;id'])
def test_only_normal_versions(version):
    with pytest.raises(ValueError):module.validate(version,'a'*40)


@pytest.mark.parametrize('revision',['main','a'*8,'a'*39+';','../main'])
def test_fixed_commit_only(revision):
    with pytest.raises(ValueError):module.validate('v11',revision)


HASHES={p:'d'*64 for p in ('agenticgen/scripts/deploy_reviewed_version.py',
 'agenticgen/scripts/read_remote_status.py','agenticgen/tests/unit/test_reviewed_deploy.py')}


def test_one_git_pull_stage_and_normal_entry_only():
    cmd=module.command('v28','a'*40,'b'*40,'e'*40,HASHES)
    assert cmd.count(' fetch --no-tags ')==1
    assert 'timeout 120 git' in cmd and '--no-filter' in cmd
    assert 'api.github.com' not in cmd and 'urllib' not in cmd
    assert 'GIT_NO_LAZY_FETCH=1' in cmd and 'GIT_TERMINAL_PROMPT=0' in cmd
    assert 'start_reviewed_version.py' in cmd and '--version v28' in cmd
    assert 'install_fixed_runtime.py' in cmd and 'agenticgen_bbbbbbbb' in cmd
    assert 'merge-base' in cmd and 'FETCH_HEAD' in cmd and 'hashlib.sha256' in cmd
    assert not any(x in cmd for x in ('restart_v10','stage-', 'curl ', 'StrictHostKeyChecking=no'))


def test_transport_override_keeps_scientific_commit():
    cmd=module.command('v28','a'*40,'b'*40,'e'*40,HASHES)
    assert '--commit '+'a'*40 in cmd
    assert 'test_reviewed_deploy.py' in cmd and 'transport_validation.json' in cmd
    assert 'e'*40 in cmd and 'no installation or launch' in cmd


@pytest.mark.parametrize('old,transport,hashes',[
 ('main','e'*40,HASHES),('b'*40,'main',HASHES),('b'*40,'e'*40,{}),
 ('b'*40,'e'*40,{**HASHES,'agenticgen/scripts/deploy_reviewed_version.py':'x'*64})])
def test_source_identity_validation(old,transport,hashes):
    with pytest.raises(ValueError):module.command('v28','a'*40,old,transport,hashes)


def test_failed_single_fetch_stops_before_validation_install_and_launch(tmp_path,monkeypatch):
    import json,os,subprocess
    old=tmp_path/'runtimes/agenticgen_bbbbbbbb';dep=tmp_path/'operations/agenticgen_deploy_aaaaaaaa'
    (old/'agenticgen/.venv/bin').mkdir(parents=True)
    (old/'agenticgen/.venv/bin/python').symlink_to(sys.executable)
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'git-calls'
    fake=bindir/'git'
    fake.write_text('#!/bin/sh\n'
        'printf "%s\n" "$*" >> "$GIT_CALL_LOG"\n'
        'case "$*" in *rev-parse*) printf "'+ 'b'*40 +'\n"; exit 0;; esac\n'
        'case "$*" in *"fetch --no-tags"*) echo "fatal: simulated CONNECT 503" >&2; exit 73;; esac\n'
        'exit 99\n')
    fake.chmod(0o755)
    cmd=module.command('v28','a'*40,'b'*40,'e'*40,HASHES).replace('/opt/MolSteer',str(tmp_path))
    # Avoid loading a user profile in this isolated failure-path check.
    cmd=cmd.replace('bash -ic ','bash -c ',1)
    env=os.environ.copy();env.update(PATH=str(bindir)+os.pathsep+env['PATH'],GIT_CALL_LOG=str(calls))
    result=subprocess.run(cmd,shell=True,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert result.returncode==73
    actual=calls.read_text().splitlines()
    assert len(actual)==2 and sum('fetch --no-tags' in x for x in actual)==1
    receipt=json.loads((dep/'git_fetch_receipt.json').read_text())
    assert receipt['fetch_exit_code']==73 and receipt['fetch_calls']==1 and receipt['rest_api_calls']==0
    assert not (dep/'transport_validation.json').exists()
    assert not (tmp_path/'runtimes/agenticgen_aaaaaaaa').exists()


def test_ghfast_clone_is_isolated_and_keeps_fixed_sources():
    cmd=module.command('v28','a'*40,'b'*40,'e'*40,HASHES,'ghfast')
    assert 'https://ghfast.top/https://github.com/H-Dw/EvoMolSteer.git' in cmd
    assert cmd.count('clone --bare --single-branch')==1
    assert 'source_cache/ghfast_eeeeeeee.git' in cmd
    assert 'credential.helper= -c http.extraHeader=' in cmd
    assert 'GIT_CONFIG_GLOBAL=/dev/null' in cmd
    assert 'Clone destination exists; inspect before retry' in cmd
    assert 'git-local-import.log' in cmd and 'merge-base' in cmd
    assert '--commit '+'a'*40 in cmd
    assert 'api.github.com' not in cmd and 'fetch --no-tags --no-recurse-submodules --no-filter https://github.com' not in cmd
    with pytest.raises(ValueError,match='user-authorized'):
        module.command('v28','a'*40,'b'*40,'e'*40,HASHES,'arbitrary-proxy')


def test_failed_ghfast_clone_never_imports_or_installs(tmp_path):
    import json,os,subprocess
    old=tmp_path/'runtimes/agenticgen_bbbbbbbb';dep=tmp_path/'operations/agenticgen_deploy_aaaaaaaa'
    (old/'agenticgen/.venv/bin').mkdir(parents=True)
    (old/'agenticgen/.venv/bin/python').symlink_to(sys.executable)
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'git-calls'
    fake=bindir/'git'
    fake.write_text('#!/bin/sh\n'
        'printf "%s\n" "$*" >> "$GIT_CALL_LOG"\n'
        'case "$*" in *rev-parse*) printf "'+ 'b'*40 +'\n"; exit 0;; esac\n'
        'case "$*" in *"clone --bare"*) exit 74;; esac\n'
        'exit 99\n')
    fake.chmod(0o755)
    cmd=module.command('v28','a'*40,'b'*40,'e'*40,HASHES,'ghfast').replace('/opt/MolSteer',str(tmp_path)).replace('bash -ic ','bash -c ',1)
    env=os.environ.copy();env.update(PATH=str(bindir)+os.pathsep+env['PATH'],GIT_CALL_LOG=str(calls))
    result=subprocess.run(cmd,shell=True,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert result.returncode==74
    assert len(calls.read_text().splitlines())==2
    r=json.loads((dep/'git_fetch_receipt.json').read_text())
    assert r['clone_calls']==1 and r['fetch_calls']==0 and r['local_import_calls']==0
    assert r['clone_exit_code']==74 and r['rest_api_calls']==0
    assert not (dep/'transport_validation.json').exists()
    assert not (tmp_path/'runtimes/agenticgen_aaaaaaaa').exists()


def test_authorized_mirror_order_is_frozen():
    assert list(module.MIRRORS.values())==['https://ghfast.top/','https://gh-proxy.com/','https://ghproxy.net/','https://gh.llkk.cc/']


@pytest.mark.parametrize('name,base',list(module.MIRRORS.items()))
def test_each_mirror_has_separate_cache_and_one_clone(name,base):
    cmd=module.command('v28','a'*40,'b'*40,'e'*40,HASHES,name)
    assert base+'https://github.com/H-Dw/EvoMolSteer.git' in cmd
    assert 'source_cache/'+name+'_eeeeeeee.git' in cmd
    assert name+'-clone.log' in cmd
    assert cmd.count('clone --bare --single-branch')==1
    assert '--commit '+'a'*40 in cmd
    assert 'api.github.com' not in cmd


class SFTP:
    def __init__(self,existing=()):self.existing=set(existing)
    def stat(self,path):
        if path not in self.existing:raise FileNotFoundError(path)
        return object()


def source(path):
    if path.endswith('/launch.json'):
        return {'value':{'version':'v10','pid':101,'commit':'b'*40}}
    return {'value':{'outcomes':1000,'rounds':5}}


def test_finished_idle_predecessor_required(monkeypatch):
    monkeypatch.setattr(module,'read_source',lambda s,p:source(p))
    assert module.ensure_unstarted(SFTP(),'v11')=='b'*40
    with pytest.raises(ValueError,match='process still exists'):
        module.ensure_unstarted(SFTP(['/proc/101']),'v11')


@pytest.mark.parametrize('suffix',['/versions/v11','/FINALIZED_v11.json'])
def test_existing_version_never_relaunched(suffix):
    with pytest.raises(ValueError,match='already staged'):
        module.ensure_unstarted(SFTP([module.ROOT+suffix]),'v11')


def test_incomplete_predecessor_rejected(monkeypatch):
    monkeypatch.setattr(module,'read_source',lambda s,p:{'value':
        {'outcomes':500,'rounds':5} if 'FINALIZED' in p else source(p)['value']})
    with pytest.raises(ValueError,match='Predecessor'):
        module.ensure_unstarted(SFTP(),'v11')
