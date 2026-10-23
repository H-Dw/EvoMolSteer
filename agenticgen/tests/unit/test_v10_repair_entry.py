import importlib.util
import sys
from pathlib import Path
import pytest
from agenticgen.adaptive_schedule.io import read,write

SCRIPTS=Path(__file__).parents[2]/'scripts'
def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(SCRIPTS))
    try: spec.loader.exec_module(module)
    finally: sys.path.remove(str(SCRIPTS))
    return module


def test_fixed_deployment_has_no_user_supplied_command_or_transfer():
    m=load('deploy_v10_repair');cmd=m.command('a'*40)
    assert 'git -C '+m.OLD+' fetch origin codex/plip-adaptive-schedule' in cmd
    assert 'restart_v10_output_budget.py --commit '+'a'*40 in cmd
    assert 'scp ' not in cmd and 'curl ' not in cmd
    for invalid in ('a'*39,'a'*40+';uname','$(whoami)','v10'):
        with pytest.raises(ValueError):m.command(invalid)


def test_inspect_only_opens_no_command_channel(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m=load('deploy_v10_repair')
    class SFTP:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def normalize(self,path):return path
        def open(self,path,mode):
            assert mode=='rb' and ('agenticgen_deploy_'+'a'*8 in path or 'failed_teacher_rerun_v10_'+'a'*8 in path)
            raise FileNotFoundError(path)
    class Client:
        def set_missing_host_key_policy(self,policy):pass
        def connect(self,*args,**kwargs):pass
        def open_sftp(self):return SFTP()
        def close(self):pass
        def get_transport(self):raise AssertionError('Inspection must not open an execution channel')
    monkeypatch.setitem(sys.modules,'paramiko',SimpleNamespace(MissingHostKeyPolicy=object,SSHClient=Client))
    monkeypatch.setattr(m.getpass,'getpass',lambda _: 'synthetic-test-only')
    monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    target=tmp_path/'inspection.json'
    monkeypatch.setattr(sys,'argv',['deploy','--commit','a'*40,'--inspect-only','--output',str(target)])
    assert m.main()==0 and read(target)['inspect_only'] is True
    assert 'synthetic-test-only' not in target.read_text()


def test_renewed_approval_retains_selected_strategy_and_never_claims_full_review():
    m=load('restart_v10_output_budget')
    old={'mode':'best_round_strategy','full_review_completed':False,'review_completed':True,
         'previous_version':'v9','next_version':'v10','strategy_review_sha256':'selected-actual-v9',
         'review_policy_operation':'original-adoption','coverage':[]}
    j={'failed_teachers':['top1'],'completed_teachers':['top2'],'old_deployment':{'commit':'original'}}
    new=m.renewed_approval(old,'new','prep','accept','operation',j)
    assert new['strategy_review_sha256']=='selected-actual-v9' and new['full_review_completed'] is False
    assert new['analyst_output_limit']==98304 and new['preserved_teacher_source_commit']=='original'
    assert 'review_policy_operation' in old and 'review_policy_operation' not in new
    with pytest.raises(ValueError):m.renewed_approval({**old,'next_version':'v11'},'n','p','a','o',j)
    with pytest.raises(ValueError):m.renewed_approval({**old,'full_review_completed':True},'n','p','a','o',j)


def test_failed_costs_deduplicate_and_exclude_completed_teacher_gpu(tmp_path):
    m=load('restart_v10_output_budget');op=tmp_path/'operations/repair';archive=op/'failure_audit'
    write(op/'journal.json',{'failed_teachers':['top1'],'failed_candidate_count':400})
    for name in ('one','duplicate'):
        write(archive/(name+'.receipt.json'),{'provider_request_id':'same','usage':{'prompt_tokens':3,'completion_tokens':98304}})
    write(archive/'pipeline.json',{'events':[
        {'teacher':t,'round':1,'kind':k,'time':v} for t in ('top1','top2')
        for k,v in [('gpu_started',10),('gpu_finished_analysis_queued',60)]]})
    r=m.aborted_costs(op,tmp_path)
    assert r['provider_id_deduplicated_receipts']==1 and r['gpu_seconds']==50
    assert r['completion_tokens']==98304 and not r['official_results_included']
    write(archive/'conflict.receipt.json',{'provider_request_id':'same','usage':{'completion_tokens':1}})
    with pytest.raises(ValueError,match='Conflicting'):m.aborted_costs(op,tmp_path)
