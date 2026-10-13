"""Scientific evaluation must preserve censoring and exact ancestry eligibility."""
import importlib.util
from pathlib import Path

import pytest

from evomolsteer.io import read_json,write_json


def evaluator():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_language_lineage_trial.py'
    spec=importlib.util.spec_from_file_location('language_evaluator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.evaluate


def fixture(tmp_path):
    trial=tmp_path/'trial';out=tmp_path/'out'
    write_json(trial/'manifest.json',{'blind_files':{}})
    write_json(trial/'blind/protocol.json',{'start_steps':[1]})
    rows=[]
    for i in range(6):
        rows.append({'step':1,'id':f'C{i}','batch':0,'slot':i,'aliases':[i],
            'n_descendants':2 if i in [0,2] else 0,'n_quality_descendants':2 if i in [0,2] else 0,
            'future_observed':i in [0,2],'has_elite_descendant':i==0,
            'quality_future_max_pic50':8.5 if i==0 else 7. if i==2 else None,
            'quality_future_mean_pic50':8.4 if i==0 else 6.9 if i==2 else None,
            'intermediate_pic50':float(i)})
    write_json(trial/'oracle/candidate_truth.json',{'candidates':rows,'elite_threshold':8.3,'strain_cap':1.})
    answer={'t':.01,'ranking':[{'rank':i+1,'candidate_id':f'C{i}'} for i in range(5)]}
    write_json(trial/'blind/answers/ranking_001.json',answer)
    return trial,out,rows,answer


def test_censoring_is_not_low_affinity(tmp_path):
    trial,out,_,_=fixture(tmp_path);evaluator()(trial,out)
    d=read_json(out/'evaluation.json');s=d['starts'][0]
    assert s['selected_observed']==2 and s['selected_censored']==3 and s['selected_elite']==1
    assert s['best_future_gap']==0
    r=d['rankings'][1]
    assert r['quality_future_max_pic50'] is None and r['future_quality_rank_among_observed'] is None
    assert s['random_five_expected_elite']==pytest.approx(5/6)


def test_unrelated_future_child_rejected(tmp_path):
    trial,out,rows,answer=fixture(tmp_path)
    child={**rows[0],'step':2,'id':'new','parent_id':'C5'};rows.append(child)
    write_json(trial/'oracle/candidate_truth.json',{'candidates':rows,'elite_threshold':8.3,'strain_cap':1.})
    write_json(trial/'blind/answers/continuation_001.json',{'nodes':[
        {'t':.01,'eligible_count':5,'ranking':answer['ranking']},
        {'t':.02,'eligible_count':1,'ranking':[{'rank':1,'candidate_id':'new'}]}]})
    with pytest.raises(ValueError,match='unrelated'):
        evaluator()(trial,out)


@pytest.mark.parametrize('bad_id',['C0','nonexistent'])
def test_duplicate_or_unknown_initial_direction_rejected(tmp_path,bad_id):
    trial,out,_,answer=fixture(tmp_path);answer['ranking'][-1]['candidate_id']=bad_id
    write_json(trial/'blind/answers/ranking_001.json',answer)
    with pytest.raises(ValueError):evaluator()(trial,out)
