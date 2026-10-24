import importlib.util
from pathlib import Path
import sys
import pytest
from agenticgen.adaptive_schedule.io import read,write

SCRIPTS=Path(__file__).parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import deploy_v21_repair as deploy
import restart_v21_commitments as restart


def test_fixed_api_deploy_and_only_v21_repair_entry():
    command=deploy.command('a'*40,'b'*40,'c'*64)
    assert 'api.github.com/repos/H-Dw/EvoMolSteer/git/blobs/' in command
    assert 'timeout 240' in command and 'hashlib.sha1' in command and 'hashlib.sha256' in command
    assert 'restart_v21_commitments.py --commit '+'a'*40 in command
    assert 'start_reviewed_version.py' not in command and 'restart_v20' not in command
    assert not any(x in command for x in ('fetch origin','scp ','curl ','StrictHostKeyChecking=no'))
    for value in ('main','a'*39,'a'*40+';id'):
        with pytest.raises(ValueError):deploy.command(value,'b'*40,'c'*64)


@pytest.mark.parametrize('mutation',['none','alive','commit','completed','scope'])
def test_restart_preflight_requires_exact_failed_teacher(monkeypatch,mutation):
    state={'launch':{'commit':deploy.PREVIOUS_COMMIT},'process':{'exists':False},
           'completion_verified':False,'state':'pipeline_failed',
           'teachers':{'5nn4_SC2_A_1017__top1':{'phase':'failed','round':5,'error':{'message':'Fewer than four unexecuted proposed schedules'}},'5nn4_SC2_A_1017__top2':{'phase':'complete'}}}
    if mutation=='alive':state['process']['exists']=True
    elif mutation=='commit':state['launch']['commit']='other'
    elif mutation=='completed':state['completion_verified']=True
    elif mutation=='scope':state['teachers']['5nn4_SC2_A_1017__top2']['phase']='failed'
    monkeypatch.setattr(deploy,'collect',lambda s,v:state)
    if mutation=='none':assert deploy.ensure_failed_scope(None,'a'*40)==state
    else:
        with pytest.raises(ValueError):deploy.ensure_failed_scope(None,'a'*40)


def test_renewal_preserves_real_v20_limited_review_and_original_top2_source():
    old={'mode':'best_round_strategy','full_review_completed':False,'previous_version':'v20','next_version':'v21',
         'strategy_review_sha256':'actual-v20','active_protocol_gate':{'operation':'old'}}
    j={'failed_teachers':['top1'],'completed_teachers':['top2'],'old_deployment':{'commit':'cea3611c'},'old_binding':{}}
    result=restart.renewed_approval(old,'new','p','a','o',j)
    assert result['strategy_review_sha256']=='actual-v20' and result['full_review_completed'] is False
    assert result['preserved_teacher_source_commits']=={'top2':'cea3611c'}
    assert result['interface_repair_policy']['private_max_format_repairs']==2
    assert result['analysis_output_budget_unchanged'] and result['analyst_output_limit']==98304
    assert 'active_protocol_gate' in old and 'active_protocol_gate' not in result
    for changes in ({'next_version':'v22'},{'previous_version':'v9'},{'full_review_completed':True}):
        with pytest.raises(ValueError):restart.renewed_approval({**old,**changes},'n','p','a','o',j)


def test_costs_exclude_completed_teacher_and_deduplicate(tmp_path):
    op=tmp_path/'operations/repair';a=op/'failure_audit'
    write(op/'journal.json',{'failed_teachers':['top1'],'failed_candidate_count':500})
    for name in ('one','copy'):
        write(a/(name+'.receipt.json'),{'provider_request_id':'same','usage':{'prompt_tokens':2,'completion_tokens':3}})
    write(a/'pipeline.json',{'events':[{'teacher':t,'round':1,'kind':k,'time':v}
        for t in ('top1','top2') for k,v in [('gpu_started',10),('gpu_finished_analysis_queued',40)]]})
    result=restart.aborted_costs(op,tmp_path)
    assert result['version']=='v21' and result['deleted_candidates']==500
    assert result['provider_id_deduplicated_receipts']==1 and result['gpu_seconds']==30
    assert result['prompt_tokens']==2 and not result['official_results_included']
    write(a/'conflict.receipt.json',{'provider_request_id':'same','usage':{'completion_tokens':1}})
    with pytest.raises(ValueError,match='Conflicting'):restart.aborted_costs(op,tmp_path)


def test_inspection_does_not_execute_or_log_credentials(tmp_path,monkeypatch):
    from types import SimpleNamespace
    class SFTP:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get_channel(self):return self
        def settimeout(self,value):pass
        def normalize(self,path):return path
        def open(self,path,mode):
            assert mode=='rb' and any(x+'a'*8 in path for x in
                ('agenticgen_deploy_','failed_teacher_rerun_v21_','analysis_output_repair_v21_','teacher_restart_acceptance_'))
            raise FileNotFoundError(path)
    class Client:
        def set_missing_host_key_policy(self,p):pass
        def connect(self,*a,**k):pass
        def open_sftp(self):return SFTP()
        def close(self):pass
        def get_transport(self):raise AssertionError('Inspection cannot execute commands')
    monkeypatch.setitem(sys.modules,'paramiko',SimpleNamespace(MissingHostKeyPolicy=object,SSHClient=Client))
    monkeypatch.setattr(deploy.subprocess,'check_output',lambda *a,**k:b'synthetic fixed helper')
    monkeypatch.setattr(deploy.getpass,'getpass',lambda _: 'synthetic-test-only')
    monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    output=tmp_path/'inspection.json'
    monkeypatch.setattr(sys,'argv',['deploy','--commit','a'*40,'--inspect-only','--output',str(output)])
    assert deploy.main()==0 and read(output)['inspect_only']
    assert 'synthetic-test-only' not in output.read_text()


@pytest.mark.parametrize('case', ['manifest','ordinary','oversize','changed','grew'])
def test_bounded_manifest_inspection_hashes_full_bytes_and_rejects_unstable_sources(tmp_path,monkeypatch,case):
    import hashlib,io,json
    from types import SimpleNamespace
    raw=json.dumps({'phase':'complete','protected_sha256':{'preserved':'digest'},
                    'audit_copies':['one'],'padding':'x'*(17*1024*1024)}).encode()
    target=('agenticgen_deploy_'+ 'a'*8 +'/installation.json' if case=='ordinary'
            else 'failed_teacher_rerun_v21_'+'a'*8+'/journal.json')
    class Handle(io.BytesIO):
        calls=0
        def stat(self):
            self.calls+=1
            size=129*1024*1024 if case=='oversize' else len(raw)
            if case=='grew':size=10
            return SimpleNamespace(st_size=size,st_mtime=2 if case=='changed' and self.calls>1 else 1)
    class SFTP:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def get_channel(self):return self
        def settimeout(self,v):pass
        def normalize(self,p):return p
        def open(self,p,mode):
            assert mode=='rb'
            if p.endswith(target):return Handle(raw)
            raise FileNotFoundError(p)
    class Client:
        def set_missing_host_key_policy(self,p):pass
        def connect(self,*a,**k):pass
        def open_sftp(self):return SFTP()
        def close(self):pass
        def get_transport(self):raise AssertionError('Inspection cannot execute commands')
    monkeypatch.setitem(sys.modules,'paramiko',SimpleNamespace(MissingHostKeyPolicy=object,SSHClient=Client))
    monkeypatch.setattr(deploy.subprocess,'check_output',lambda *a,**k:b'synthetic fixed helper')
    monkeypatch.setattr(deploy.getpass,'getpass',lambda _: 'synthetic-test-only')
    monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    output=tmp_path/'inspection.json'
    monkeypatch.setattr(sys,'argv',['deploy','--commit','a'*40,'--inspect-only','--output',str(output)])
    if case!='manifest':
        with pytest.raises(ValueError):deploy.main()
        assert not output.exists()
    else:
        assert deploy.main()==0
        result=read(output);source=next(iter(result['sources'].values()))
        assert source['sha256']==hashlib.sha256(raw).hexdigest()
        assert source['bytes']==len(raw) and source['projection_only'] is True
        assert source['value']['protected_file_count']==1
        assert source['value']['audit_copy_count']==1
        assert 'padding' not in source['value'] and output.stat().st_size<4096
