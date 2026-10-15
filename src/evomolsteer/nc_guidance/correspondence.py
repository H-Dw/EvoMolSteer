"""Retrospective identity diagnostics with stable generated atom slots.

Compare correspondence in the saved teacher-clock current state with a fresh
terminal geometric assignment. This is not the last active bridge assignment,
and does not infer unsaved intermediate chemical identities.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import read_json,write_json,digest
from .search import atomic_indices


def assignment(points,target,mask):
    valid=np.flatnonzero(mask)
    cost=np.square(points[valid,None]-target[None]).sum(-1)
    rr,cc=linear_sum_assignment(cost)
    if len(cc)!=len(target):raise ValueError('Insufficient active generated points')
    return valid[rr[np.argsort(cc)]]


def observations(geometry,program,teacher,labels,records):
    half=geometry['teacher_clock'];final=geometry['terminal'];slots=program['atom_slots']
    hx=np.asarray(half['coords_A']);fx=np.asarray(final['coords_A'])
    if hx.shape!=fx.shape or labels.shape!=hx.shape[:2]:raise ValueError('Generated slot continuity is unavailable')
    th=np.asarray(teacher['coords_A'])[slots];tf=np.asarray(teacher['terminal_coords_A'])[slots]
    local={s:i for i,s in enumerate(slots)};rows=[]
    for i,(h,f,a,record) in enumerate(zip(hx,fx,labels,records)):
        hm=assignment(h,th,np.asarray(half['mask'][i],bool))
        fm=assignment(f,tf,np.asarray(final['mask'][i],bool))
        for slot,element in zip(program['typed_slots'],program['typed_elements']):
            j=local[slot];u=int(hm[j]);v=int(fm[j]);desired=teacher['atom_vocabulary'][element]
            rows.append({'sample':i,'teacher_slot':slot,'desired_element':element,
                'half_generated_slot':u,'final_generated_slot':v,'correspondence_switched':u!=v,
                'terminal_correct_along_half_slot':bool(a[u]==desired) if a[u]>=0 else None,
                'terminal_correct_after_rematching':bool(a[v]==desired) if a[v]>=0 else None,
                'half_match_distance_A':float(np.linalg.norm(h[u]-th[j])),
                'tracked_terminal_distance_A':float(np.linalg.norm(f[u]-tf[j])),
                'rematched_terminal_distance_A':float(np.linalg.norm(f[v]-tf[j])),
                'pic50':record['pic50'],'qualified_structure':record['qualified_structure']})
    return rows


def run(repo,cfg,details=False):
    repo=Path(repo);study=repo/cfg['campaign_study'];state=read_json(study/'state.json')
    rows=[];sources=[]
    for entry in state['rounds']:
        if entry.get('status')=='designer_deferred':continue
        label=entry['round'];folder=repo/cfg['generated_root']/label/label
        binding=read_json(next((study/label/'candidates').glob('*/bindings.json')))
        p=read_json(repo/binding['programs'][0]['program']);t=read_json(repo/binding['teacher'])
        if not folder.exists():raise FileNotFoundError('Resolve reused generation directory explicitly: '+str(folder))
        g=read_json(folder/'geometry.json');records=read_json(folder/'records.json')
        a=atomic_indices(folder,g,t['atom_vocabulary'])
        rows.extend({'round':label,**r} for r in observations(g,p,t,a,records))
        sources.append({'round':label,'geometry_sha256':digest(folder/'geometry.json'),
            'program_sha256':digest(repo/binding['programs'][0]['program']),
            'teacher_sha256':digest(repo/binding['teacher'])})
    df=pd.DataFrame(rows);summary=[]
    for (label,element),group in df.groupby(['round','desired_element'],sort=True):
        for scope,v in [('all_attempts',group),('basic_qualified',group[group.qualified_structure])]:
            def mean(key):
                values=v[key].dropna();return float(values.astype(float).mean()) if len(values) else None
            summary.append({'round':label,'element':element,'scope':scope,'n_sites':len(v),
                'n_identity_observed':int(v.terminal_correct_along_half_slot.notna().sum()),
                'correspondence_switch_fraction':mean('correspondence_switched'),
                'terminal_accuracy_tracking_half':mean('terminal_correct_along_half_slot'),
                'terminal_accuracy_rematching':mean('terminal_correct_after_rematching'),
                'tracked_terminal_distance_A':mean('tracked_terminal_distance_A'),
                'rematched_terminal_distance_A':mean('rematched_terminal_distance_A')})
    out=study/'summary';write_json(out/'correspondence_summary.json',{
        'definition':'Hungarian assignment at saved teacher-clock current geometry vs new terminal assignment; same generated slot tracked between the two saved frames',
        'not_observed':['Actual last bridge correspondence','Intermediate atom identities when not saved'],
        'interpretation':'A switch can confound spatial identity accuracy; tracking alone does not prove the N/C reward succeeded',
        'sources':sources,'groups':summary})
    if details:df.to_parquet(out/'correspondence_details.parquet',index=False,compression=None)
    return pd.DataFrame(summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True)
    p.add_argument('--details',action='store_true');a=p.parse_args()
    print(run(a.repo,read_json(a.config),a.details).to_string(index=False))


if __name__=='__main__':main()
