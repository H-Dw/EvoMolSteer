"""Index all outcomes without hiding older failures behind a recency cutoff."""
import json
from evomolsteer.io import read_json


def load_history(repo,prior_study):
    prior_study=str(prior_study).replace('\\','/')
    state=read_json(repo/prior_study/'state.json');rows=[]
    for r in state['rounds']:
        p=r.get('program',{});m=r.get('metrics',{})
        rows.append({'campaign':prior_study,'round':r['round'],'factor':r['factor'],'status':r['status'],
            'candidate':p.get('candidate_id'),'region':p.get('atom_slots'),'operator':p.get('geometry_mode'),
            'weights':p.get('weights'),'dose':p.get('dose'),'mean':m.get('mean_all_pic50'),'best':m.get('best_qualified_pic50'),
            'n':m.get('n'),'native_options':p.get('native_options'),'window':p.get('window'),'initial_sha256':m.get('initial_sha256'),
            'realization':r.get('realization'),'rollback':r.get('rollback',False)})
    return rows


def retrieve(history,candidate=None,factor=None):
    return [r for r in history if (candidate is None or r.get('candidate')==candidate) and (factor is None or r.get('factor')==factor)]
