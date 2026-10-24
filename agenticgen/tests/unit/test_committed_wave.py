"""Pure contract/provenance checks; no LLM replay or scientific outcomes."""
import copy
import pytest
from agenticgen.adaptive_schedule.io import read, write, sha
from agenticgen.adaptive_schedule.committed_wave import load_commitment, validate_commitment, reject_committed


def saved_wave(tmp_path, r):
    base=tmp_path/'v21'/'t'; output=base/f'round_{r:02d}'/'council'
    ticks=[[5,5,5,3],[5,5,3,5],[5,3,5,5],[3,5,5,5]]
    jobs=[{'ticks':t,'source_proposal_id':f'G{i+1}:H1','id':f'H{i+1}',
           'private_canary':'MUST_NOT_APPEAR'} for i,t in enumerate(ticks)]
    if r == 1:
        for j in jobs:j['source_proposal_id']='bootstrap:'+j['id']
    source={'teacher_id':'t','version':'v21','observed_round':r-1,'jobs':jobs,
            'private_canary':'MUST_NOT_APPEAR'}
    path=base/('bootstrap/hypotheses.json' if r==1 else f'round_{r-1:02d}/council/selection.json')
    write(path,source)
    if r>1:write(path.parent/'COMPLETE.json',{'selection_sha256':sha(path)})
    write(output.parent/'job.json',source)
    specs=[{'group_id':f'G{i+1}','anchor_ticks':t,
            'data':{'teacher':{'teacher_id':'t'},'observed_round':r,'sibling_result':'MUST_NOT_APPEAR'}} for i,t in enumerate(ticks)]
    return output,specs,path


@pytest.mark.parametrize('r',[1,5])
def test_projection_contains_only_prior_arrays_and_provenance(tmp_path,r):
    output,specs,path=saved_wave(tmp_path,r)
    doc=load_commitment(output,specs,'v21',10)
    assert 'MUST_NOT_APPEAR' not in str(doc)
    assert doc['source_sha256']==sha(path) and doc['selected_after_round']==r-1
    assert doc['schedules']==sorted(s['anchor_ticks'] for s in specs)
    with pytest.raises(ValueError,match='already selected'):
        reject_committed({'hypotheses':[{'ticks':specs[1]['anchor_ticks']}]},doc['schedules'])
    reject_committed({'hypotheses':[{'ticks':[5,5,5,2]}]},doc['schedules'])


@pytest.mark.parametrize('mutation',['teacher','version','round','future','source','hash','extra','missing_anchor','duplicate'])
def test_projection_rejects_scope_or_sibling_content(tmp_path,mutation):
    output,specs,_=saved_wave(tmp_path,5);doc=load_commitment(output,specs,'v21',10)
    if mutation=='teacher':doc['teacher_id']='other'
    elif mutation=='version':doc['version']='v20'
    elif mutation=='round':doc['observed_round']=4
    elif mutation=='future':doc['selected_after_round']=5
    elif mutation=='source':doc['source_relative_path']='../other.json'
    elif mutation=='hash':doc['source_sha256']='not-a-hash'
    elif mutation=='extra':doc['sibling_results']=[1]
    elif mutation=='missing_anchor':doc['schedules'][doc['schedules'].index(specs[0]['anchor_ticks'])]=[1,1,1,1]
    else:doc['schedules'][1]=doc['schedules'][0]
    with pytest.raises(ValueError):validate_commitment(doc,'t','v21',5,specs[0]['anchor_ticks'],10)


@pytest.mark.parametrize('mutation',['source_tamper','job_ticks','job_ids','private_anchor'])
def test_prior_selection_must_match_actual_wave(tmp_path,mutation):
    output,specs,path=saved_wave(tmp_path,5)
    if mutation=='source_tamper':
        source=read(path);source['jobs'][0]['ticks'][0]=6;write(path,source)
    elif mutation=='private_anchor':specs[0]['anchor_ticks']=[1,1,1,1]
    else:
        job=read(output.parent/'job.json')
        job['jobs'][0]['ticks' if mutation=='job_ticks' else 'source_proposal_id']=[1,1,1,1] if mutation=='job_ticks' else 'G2:H2'
        write(output.parent/'job.json',job)
    with pytest.raises(ValueError):load_commitment(output,specs,'v21',10)
