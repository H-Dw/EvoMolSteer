"""Best-round scope, factual selection and approval provenance, using synthetic data."""
import copy
from pathlib import Path
import subprocess

import pytest

from agenticgen.adaptive_schedule import group_review as gate
from agenticgen.adaptive_schedule import strategy_review as focused
from agenticgen.adaptive_schedule.io import read, write, sha


@pytest.fixture
def focused_transition(tmp_path, monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();root=tmp_path/'results'
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a],stderr=subprocess.STDOUT).decode().strip()
    git('init');git('config','core.autocrlf','false');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    write(root/'config.json',{'schema':'synthetic','versions':['v7','v8','v9'], 'condition_comparison':{'ranking_tolerance_A':.02}})
    write(root/'teachers.json',[{'teacher_id':'a'},{'teacher_id':'b'}])
    source_auth=Path(__file__).parents[2]/'docs/authorizations/20261023_best_round_strategy_review.json'
    write(repo/focused.AUTHORIZATION,read(source_auth))
    auth={'path':focused.AUTHORIZATION,'sha256':sha(repo/focused.AUTHORIZATION)}
    write(root/'PREPARED.json',{'master_review_policy':{'mode':focused.MODE,'authorization':auth}})
    for path in gate.SKILLS.values():
        p=repo/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('Synthetic skill\n',encoding='utf8')
    rows=[]; representatives={}
    for teacher in ['a','b']:
        local=[]
        for r in range(1,6):
            for g in range(1,5):
                c=1+r*.001+g*.0001;gap=5-r*.1+g*.01
                ticks=[r,g,5,5];identifier=f'r{r:02d}:G{g}_unselected'
                local.append({'id':identifier,'condition':c,'functional_gap':gap,'best':7.,'ticks':ticks})
                rows.append({'teacher':teacher,'round':r,'collection':f'G{g}_unselected','ticks':ticks,
                    'condition_rmsd_A_all_attempts':{'mean':c},'mean_affinity_absolute_gap':gap,'affinity':{'mean':4.,'max':7.}})
        representatives[teacher]={'primary':focused.choose(local,.02)}
    report={'rows':rows,'representatives':representatives};base=root/'versions/v7';write(base/'report.json',report)
    write(base/'binding.json',{'synthetic':True});write(base/'review/manifest.json',{'expected_sessions':52,'binding_sha256':sha(base/'binding.json'),'shards':[]})
    write(root/'FINALIZED_v7.json',{'report_sha256':sha(base/'report.json'),'review_manifest_sha256':sha(base/'review/manifest.json')})
    teachers=[]
    for t in ['a','b']:
        prefix=f'versions/v7/{t}/round_05/council/'
        files={}
        for name in ('synthesis.json','analysts/G1/hypotheses.json'):
            write(root/(prefix+name),{'synthetic':'hypothesis'});files[prefix+name]=sha(root/(prefix+name))
        teachers.append({'teacher':t,'best_round':5,'best_group':'G1','evidence_files':files,
            **{k:'Synthetic strategy judgment; no real experimental approval.' for k in
               ('hypothesis_assessment','first_principles','performance_interpretation','next_direction')}})
    review={'mode':focused.MODE,'version':'v7','report_sha256':sha(base/'report.json'),'full_review_completed':False,'teachers':teachers,
            'capability_hypotheses':[{k:'Synthetic test only.' for k in ('blind_spot','strategy','expected_behavior','next_version_test')}]}
    write(repo/'strategy.json',review);(repo/'review.md').write_text('Synthetic review\n',encoding='utf8')
    git('add','.');git('commit','-m','Synthetic focused review')
    monkeypatch.setattr(gate,'repository',lambda:repo);monkeypatch.setattr(gate,'commit',lambda:git('rev-parse','HEAD'))
    monkeypatch.setattr(focused,'commit',lambda:git('rev-parse','HEAD'))
    import agenticgen.adaptive_schedule.version_gate as control
    checks=[];monkeypatch.setattr(control,'check_control_gate',lambda *args:checks.append(args[1]))
    approval={'mode':focused.MODE,'review_completed':True,'full_review_completed':False,'coverage':[],
      'previous_version':'v7','next_version':'v8','review_commit':git('rev-parse','HEAD'),
      'finalized_sha256':sha(root/'FINALIZED_v7.json'),'manifest_sha256':sha(base/'review/manifest.json'),
      'skill_sha256':{k:sha(repo/p) for k,p in gate.SKILLS.items()},'authorization_document':auth['path'],
      'authorization_document_sha256':auth['sha256'],'review_document':'review.md','review_document_sha256':sha(repo/'review.md'),
      'strategy_review':'strategy.json','strategy_review_sha256':sha(repo/'strategy.json')}
    write(root/'versions/v8/review_approval.json',approval)
    return root,repo,approval,git,checks


def test_four_selected_contexts_pass_without_full_coverage(focused_transition):
    root,_,_,_,checks=focused_transition
    result=gate.review_check(root,'v8')
    assert result['sessions']==4 and result['full_review_completed'] is False
    assert checks==['v8']


@pytest.mark.parametrize('mutation',['missing_teacher','wrong_round','blank_strategy','missing_hypothesis'])
def test_incomplete_or_wrong_focused_review_rejected(focused_transition,mutation):
    root,repo,a,git,_=focused_transition;r=read(repo/'strategy.json')
    if mutation=='missing_teacher':r['teachers'].pop()
    if mutation=='wrong_round':r['teachers'][0]['best_round']=4
    if mutation=='blank_strategy':r['teachers'][0]['first_principles']=' '
    if mutation=='missing_hypothesis':r['teachers'][0]['evidence_files']={}
    write(repo/'strategy.json',r);git('add','.');git('commit','-m','Invalid synthetic review')
    a.update(review_commit=git('rev-parse','HEAD'),strategy_review_sha256=sha(repo/'strategy.json'))
    write(root/'versions/v8/review_approval.json',a)
    with pytest.raises(ValueError):gate.review_check(root,'v8')


@pytest.mark.parametrize('mutation',['full_claim','old_mode','source','report','authorization'])
def test_gate_preserves_scope_and_selected_provenance(focused_transition,mutation):
    root,repo,a,_,_=focused_transition
    if mutation=='full_claim':a['full_review_completed']=True
    if mutation=='old_mode':a['mode']='complete_master_review'
    if mutation=='source':write(root/'versions/v7/a/round_05/council/synthesis.json',{'changed':True})
    if mutation=='report':write(root/'versions/v7/report.json',{'changed':True})
    if mutation=='authorization':a['authorization_document_sha256']='0'*64
    write(root/'versions/v8/review_approval.json',a)
    with pytest.raises(ValueError):gate.review_check(root,'v8')


def test_winner_uses_group_band_and_ignores_selected_archive(focused_transition):
    root,_,_,_,_=focused_transition;report=read(root/'versions/v7/report.json')
    selected=copy.deepcopy(report['rows'][0]);selected.update(collection='TOP1',mean_affinity_absolute_gap=0)
    selected['condition_rmsd_A_all_attempts']['mean']=0;report['rows'].append(selected)
    assert focused.best_rounds(report,.02)['a']['round']==5
    report['representatives']['a']['primary']['selected']['id']='r01:G1_unselected'
    with pytest.raises(ValueError,match='representative'):focused.best_rounds(report,.02)
