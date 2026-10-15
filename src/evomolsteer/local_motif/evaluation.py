"""Frozen tail objective, paired uncertainty, local outcomes and energy audits."""
import copy
import gzip
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest


def original_reference(repo,master):
    rows=[];sources={}
    base=repo/master['raw_dataset']/'results'/master['campaign']/master['arm']
    for path in sorted(base.glob('batch_*/final_records.json')):
        sources[str(path.relative_to(repo))]=digest(path)
        for r in read_json(path):
            rows.append({'pic50':r['pic50_on_rescore'],'qualified_structure':bool(r.get('build_success') and r.get('connected') and r.get('min_protein_distance_A',0)>=.8),
                'batch':r['batch'],'slot':r['slot'],'root':str(r['batch'])+':'+str(r.get('root_slot'))})
    qualified=[r['pic50'] for r in rows if r['qualified_structure'] and r['pic50'] is not None]
    return {'records':rows,'sources':sources,'tau':float(np.quantile(qualified,.9)),
        'threshold_definition':'Frozen q90 of original basic-qualified scores: decodable, connected, minimum receptor distance >= .8 A.',
        'n':len(rows),'n_qualified':len(qualified),'best':float(max(qualified)),'mean_all_scored':float(np.mean([r['pic50'] for r in rows if r['pic50'] is not None]))}


def score_records(records,tau):
    scores=np.array([np.nan if r.get('pic50') is None else r['pic50'] for r in records]);qualified=np.array([r['qualified_structure'] for r in records])&np.isfinite(scores)
    utility=np.where(qualified,np.maximum(np.nan_to_num(scores)-tau,0),0.)
    return {'n':len(records),'n_scored':int(np.isfinite(scores).sum()),'n_qualified':int(qualified.sum()),
        'mean_all_pic50':float(np.nanmean(scores)),'best_qualified_pic50':float(scores[qualified].max()) if qualified.any() else None,
        'U_tau':float(utility.mean()),'hit_fraction':float((qualified&(scores>=tau)).mean()),
        'n_above_target':int((qualified&(scores>=8.7048063278)).sum()),'tau':tau}


def paired(a,b,tau):
    aa=read_json(a/'guidance_audit.json')['input_pairing'];bb=read_json(b/'guidance_audit.json')['input_pairing']
    for k in ['initial_sha256','initial_slots','seed','resolved_native_options']:
        if aa[k]!=bb[k]:raise ValueError('Unpaired comparison: '+k)
    if aa.get('resume')!=bb.get('resume'):raise ValueError('Continuation identity mismatch')
    ar=read_json(a/'records.json');br=read_json(b/'records.json')
    def vectors(rows):
        s=np.array([r['pic50'] for r in rows]);q=np.array([r['qualified_structure'] for r in rows]);return s,np.where(q,np.maximum(s-tau,0),0.)
    ax,au=vectors(ar);bx,bu=vectors(br)
    rng=np.random.default_rng(42);ids=rng.integers(0,len(ar),size=(5000,len(ar)))
    return {'n_pairs':len(ar),'mean_difference':float((ax-bx).mean()),'U_difference':float((au-bu).mean()),
        'mean95':np.quantile((ax-bx)[ids].mean(1),[.025,.975]).tolist(),
        'U95':np.quantile((au-bu)[ids].mean(1),[.025,.975]).tolist(),
        'scope':'Descriptive paired pool; not search-adjusted and not target generalization.'}


def local_outcomes(folder,p,t,output):
    import torch
    from .reward import MotifReward
    g=read_json(folder/'geometry.json');rows=[]
    for state_name,state in [('boundary',g['teacher_clock']),('final',g['terminal'])]:
        q=copy.deepcopy(p);q['representation']='bridge' if state_name=='boundary' else 'endpoint';reward=MotifReward(q,t)
        xyz=torch.as_tensor(state['coords_A'],dtype=torch.double);mask=torch.as_tensor(state['mask'] if 'mask' in state else g['terminal']['mask'])
        atom=np.asarray(state['atomic_indices']);logits=torch.nn.functional.one_hot(torch.as_tensor(atom),max(t['atom_vocabulary'].values())+1).double()
        value,detail=reward.evaluate(xyz,logits,mask)
        desired=np.array(reward.labels);matched=np.take_along_axis(atom,detail['assignment'].numpy(),1)
        for i in range(len(xyz)):
            error=float(detail['target_error'][i]);point=float(detail['point_rms_A'][i])
            rows.append({'slot':i,'state':state_name,'target_representation':q['representation'],'error':error,'point_rms_A':point,
                'native_element_fraction':float((matched[i]==desired).mean()),
                'geometric_attainment':bool(point<=p['base_band_A']*p['band_multiplier']) if p['operator']=='pointwise' else bool(error<1e-6)})
    f=pd.DataFrame(rows);output.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(output,index=False,compression=None)
    result={}
    for s,part in f.groupby('state'):
        result[s]={'mean_error':float(part.error.mean()),'mean_point_rms_A':float(part.point_rms_A.mean()),
            'native_element_fraction':float(part.native_element_fraction.mean()),'attainment':float(part.geometric_attainment.mean())}
    result['limits']='Geometry and native elemental identity are separate. Final evaluation uses terminal teacher targets; not a claim of certified intermediate chemical roles.'
    return result


def energy(folder,output):
    from rdkit import Chem,rdBase
    from evomolsteer.nc_guidance.energy_probe import measure
    if output.exists():return read_json(output)
    rows=[]
    for r in read_json(folder/'records.json'):
        item={'slot':r['slot'],'pic50':r['pic50'],'qualified':r['qualified_structure'],'status':'not_decodable'}
        path=folder/('%03d.sdf.gz'%r['slot'])
        if path.exists() and r['valid_connected']:
            try:
                with gzip.open(path,'rt') as f:m=Chem.MolFromMolBlock(f.read().split('$$$$')[0],sanitize=False,removeHs=False)
                item.update(measure(m))
            except Exception as e:item.update(status='error',error=type(e).__name__)
        rows.append(item)
    values=[r['heavy_relaxation_relief_after_H'] for r in rows if r.get('heavy_relaxation_relief_after_H') is not None]
    result={'rdkit':rdBase.rdkitVersion,'n':len(rows),'coverage':len(values),'median':float(np.median(values)) if values else None,
        'p90':float(np.quantile(values,.9)) if values else None,'rows':rows,'meaning':'Two-stage local MMFF94s relaxation proxy, not binding energy; no new gate.'}
    write_json(output,result);return result
