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


def test_api_first_and_normal_entry_only():
    cmd=module.command('v11','a'*40,'b'*40,'c'*40,'d'*64)
    assert 'api.github.com/repos/H-Dw/EvoMolSteer/git/blobs/' in cmd
    assert 'timeout 240' in cmd and 'hashlib.sha1' in cmd and 'hashlib.sha256' in cmd
    assert 'start_reviewed_version.py' in cmd and '--version v11' in cmd
    assert 'install_fixed_runtime.py' in cmd and 'agenticgen_bbbbbbbb' in cmd
    assert not any(x in cmd for x in ('fetch origin','restart_v10','stage-', 'curl ', 'StrictHostKeyChecking=no'))


def test_fixed_fetcher_override_keeps_scientific_commit():
    cmd=module.command('v11','a'*40,'b'*40,'c'*40,'d'*64,'e'*40,'f'*40,'1'*64)
    assert '--commit '+'a'*40 in cmd
    assert 'test_github_object_fetch.py' in cmd and 'fetcher_validation.json' in cmd
    assert 'e'*40 in cmd and 'Scoped fetcher checks failed' in cmd


def test_fetcher_override_without_validation_rejected():
    with pytest.raises(ValueError,match='committed remote checks'):
        module.command('v11','a'*40,'b'*40,'c'*40,'d'*64,'e'*40)


@pytest.mark.parametrize('old,oid,digest',[('main','c'*40,'d'*64),('b'*40,'c'*8,'d'*64),('b'*40,'c'*40,'x'*64)])
def test_source_identity_validation(old,oid,digest):
    with pytest.raises(ValueError):module.command('v11','a'*40,old,oid,digest)


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
