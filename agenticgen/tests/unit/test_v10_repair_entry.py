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
    assert 'timeout 120 git -C '+m.OLD in cmd and 'fetch origin codex/plip-adaptive-schedule' in cmd
    assert 'restart_v10_output_budget.py --commit '+'a'*40 in cmd
    assert 'scp ' not in cmd and 'curl ' not in cmd
    for invalid in ('a'*39,'a'*40+';uname','$(whoami)','v10'):
        with pytest.raises(ValueError):m.command(invalid)
    with pytest.raises(ValueError):m.command('a'*40,github_api=True,bootstrap_oid='../escape',bootstrap_sha256='b'*64)
    fallback=m.command('a'*40,github_api=True,bootstrap_oid='b'*40,bootstrap_sha256='c'*64)
    assert 'api.github.com/repos/H-Dw/EvoMolSteer/git/blobs/' in fallback
    assert 'sha256' in fallback and 'http_proxy' not in fallback and 'timeout 240' in fallback
    citations=m.command('a'*40,citation_repair=True)
    assert 'restart_v10_citations.py' in citations and 'agenticgen_098467f5' in citations
    assert 'restart_v10_output_budget.py' not in citations


def test_github_object_reconstruction_checks_exact_digest():
    m=load('fetch_fixed_github_api')
    raw=b'tree '+b'a'*40+b'\nauthor Tester <t@example.test> 0 +0800\ncommitter Tester <t@example.test> 0 +0800\n\nmessage\n'
    target=m.object_id('commit',raw)
    doc={'sha':target,'tree':{'sha':'a'*40},'parents':[],'message':'message',
         'author':{'name':'Tester','email':'t@example.test','date':'1970-01-01T00:00:00Z'},
         'committer':{'name':'Tester','email':'t@example.test','date':'1970-01-01T00:00:00Z'}}
    assert m.commit_bytes(doc,target)==raw
    with pytest.raises(ValueError):m.commit_bytes(doc,'b'*40)
    tree=m.tree_bytes([{'name':'z','type':'blob','mode':'100644','sha':'b'*40},
                       {'name':'a','type':'tree','mode':'040000','sha':'a'*40}])
    assert tree.startswith(b'40000 a\0'+bytes.fromhex('a'*40))


def test_inspect_only_opens_no_command_channel(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m=load('deploy_v10_repair')
    class SFTP:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def normalize(self,path):return path
        def open(self,path,mode):
            assert mode=='rb' and any(prefix+'a'*8 in path for prefix in
                ('agenticgen_deploy_','failed_teacher_rerun_v10_','analysis_output_repair_v10_','teacher_restart_acceptance_'))
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


def test_citation_renewal_preserves_original_teacher_source_and_prior_review():
    m=load('restart_v10_citations')
    old={'mode':'best_round_strategy','full_review_completed':False,'previous_version':'v9','next_version':'v10',
         'strategy_review_sha256':'actual-v9'}
    j={'failed_teachers':['top1'],'completed_teachers':['top2'],'old_deployment':{'commit':'96k-attempt'},
       'old_binding':{'preserved_teacher_provenance':{'top2':{'deployment':{'commit':'original-64k'}}}}}
    new=m.renewed_approval(old,'new','prep','accept','operation',j)
    assert new['preserved_teacher_source_commits']=={'top2':'original-64k'}
    assert new['strategy_review_sha256']=='actual-v9' and new['analysis_output_budget_unchanged']
    assert new['full_review_completed'] is False and new['citation_protocol']=='private-evidence-alias-v1'
