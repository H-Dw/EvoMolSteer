"""Full terminal catalogue, one-candidate-versus-rest fields and actual ancestors."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import read_json,write_json,digest
from evomolsteer.continuous.affinity_geometry import landmarks
from evomolsteer.continuous.terminal_regions import feature_names
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from evomolsteer.storage.trajectory import TrajectoryPackage


def candidate_id(batch,slot):return f'F_b{int(batch):03d}_s{int(slot):03d}'


def compact_regions(x,points,radius):
    """Exactly zero outside the same support used for teacher slot transfer."""
    displacement=x[:,:,None]-np.asarray(points)[None,None]
    mass=np.maximum(1-np.square(displacement).sum(-1)/radius**2,0)**2
    total=mass.sum(1);weight=np.divide(mass,total[:,None],out=np.zeros_like(mass),where=total[:,None]>0)
    mean=(weight[...,None]*displacement).sum(1);residual=displacement-mean[:,None]
    cov=np.einsum('bnr,bnri,bnrj->brij',weight,residual,residual)
    values=np.concatenate([mass.mean(1)[...,None],mean,
        np.stack([cov[:,:,i,j] for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]],-1)],-1)
    values[:,:,1:]=np.where((total>0)[...,None],values[:,:,1:],np.nan)
    return values.reshape(len(x),-1)


def prepare(repo,cfg):
    import torch
    repo=Path(repo);out=repo/cfg['study'];out.mkdir(parents=True,exist_ok=True)
    raw=repo/cfg['dataset'];metrics=pd.read_parquet(repo/cfg['terminal_metrics']).copy()
    if metrics.duplicated(['batch','slot']).any():raise ValueError('Duplicate final locators')
    metrics['candidate_id']=[candidate_id(b,s) for b,s in metrics[['batch','slot']].to_numpy()]
    metrics=metrics.sort_values(['pic50_on_rescore','batch','slot'],ascending=[False,True,True],kind='stable',na_position='last').reset_index(drop=True)
    metrics['rank']=np.arange(1,len(metrics)+1);metrics['processing_status']='pending'
    metrics.to_parquet(out/'ranked_finals.parquet',compression=None,index=False)
    clouds={};masks={};sources=[]
    for batch in sorted(metrics.batch.unique()):
        path=raw/'results'/cfg['campaign']/cfg['arm']/f'batch_{batch:03d}'/'final_prediction.pt.gz'
        if not path.exists():raise FileNotFoundError('Actual final artifact missing, not a reason to exclude it: '+str(path))
        with gzip.open(path,'rb') as f:state=torch.load(f,map_location='cpu',weights_only=False)
        xyz=state['coords'].detach().cpu().numpy();mask=state['mask'].detach().cpu().numpy().astype(bool)
        if xyz.ndim!=3 or xyz.shape[:2]!=mask.shape or not np.isfinite(xyz).all():raise ValueError('Invalid final coordinates')
        for slot in metrics[metrics.batch.eq(batch)].slot:
            key=candidate_id(batch,slot);clouds[key]=xyz[slot];masks[key]=mask[slot]
        sources.append({'batch':int(batch),'artifact':path.relative_to(repo).as_posix(),'sha256':digest(path),
            'coordinate_semantics':'Actual completed decoder input, world Angstrom; no SDF atom reordering'})
    keys=metrics.candidate_id.tolist();xyz=np.stack([clouds[k] for k in keys]);mask=np.stack([masks[k] for k in keys])
    if not mask.all():raise ValueError('This initial coordinate field adapter requires fixed active slots; do not silently compress slot identity')
    points,origin=landmarks(raw/'inputs'/cfg['protein'],raw/'inputs'/cfg['ligand'])
    fields=feature_names(len(points));features=compact_regions(xyz.astype(float),points,cfg['region_radius_A'])
    np.savez_compressed(out/'terminal_fields.npz',coords_A=xyz.astype(np.float32),mask=mask,features=features)
    write_json(out/'manifest.json',{'schema_version':'final-single-candidate-evidence-1.0','terminal_time':cfg['terminal_time'],
        'representation':'terminal_world_A','candidate_ids':keys,'candidate_count':len(keys),
        'region_radius_A':cfg['region_radius_A'],'landmarks_A':points,'origin_A':origin,'feature_names':fields,
        'sources':sources,'terminal_metrics_sha256':digest(repo/cfg['terminal_metrics']),
        'no_middle_head_shortlist':True,'comparison':'One actual terminal molecule versus every other terminal molecule',
        'statistics':'Descriptive one-to-rest differences and empirical ranks; not single-positive p/q tests',
        'region_kernel':'(1-distance_squared/radius_squared)_+^2; exactly zero outside transferred atom-slot support',
        'empty_regions':'Mass is zero; conditional coordinate moments are undefined, not zero geometry',
        'source_dependence':'Historical clones and shared roots are dependent; empirical rest is not independent replication'})
    write_json(out/'cursor.json',{'next_rank':1,'total':len(keys),'processed':[],'analysis_budget':cfg['analysis_limit'],'not_processed':'pending, retrievable by candidate ID'})
    return metrics


def candidate_packet(repo,cfg,rank):
    out=Path(repo)/cfg['study'];manifest=read_json(out/'manifest.json');table=pd.read_parquet(out/'ranked_finals.parquet')
    matches=np.flatnonzero(table['rank'].to_numpy()==rank)
    if len(matches)!=1:raise ValueError('Unregistered final rank')
    i=int(matches[0]);row=table.iloc[i];arrays=np.load(out/'terminal_fields.npz',allow_pickle=False)
    xyz=arrays['coords_A'][i];features=arrays['features'];rest=np.arange(len(table))!=i;valid=table.valid_connected.to_numpy(bool)&rest
    regions=[];evidence=[];radius=manifest['region_radius_A']
    for r,point in enumerate(manifest['landmarks_A']):
        slots=np.flatnonzero(np.linalg.norm(xyz-np.asarray(point),axis=1)<radius).tolist()
        regions.append({'region_id':f'R{r:02d}','center_A':point,'radius_A':radius,'atom_slots':slots,'local_usable':0<len(slots)<len(xyz)})
        for j in range(10):
            column=r*10+j;values=features[rest,column];values=values[np.isfinite(values)]
            raw=features[i,column];v=float(raw) if np.isfinite(raw) else None
            mean=float(values.mean()) if len(values) else None;sd=float(values.std()) if len(values) else None
            vv=features[valid,column];vv=vv[np.isfinite(vv)]
            evidence.append({'evidence_id':f"{row.candidate_id}/R{r:02d}/{j}",'region_id':f'R{r:02d}',
                'observable':manifest['feature_names'][column],'candidate_value':v,'rest_count':int(rest.sum()),'coordinate_supported_rest_count':len(values),
                'rest_mean':mean,'rest_sd':sd,'candidate_minus_rest':v-mean if v is not None and mean is not None else None,
                'standardized_difference':(v-mean)/sd if v is not None and sd is not None and sd>1e-12 else None,
                'empirical_percentile':float(((values<v).sum()+.5*(values==v).sum())/len(values)) if v is not None and len(values) else None,
                'valid_rest_count':int(valid.sum()),'valid_rest_mean':float(vv.mean()) if len(vv) else None,
                'units':'dimensionless compact mass' if j==0 else 'Angstrom' if j<=3 else 'Angstrom squared'})
    packet={'candidate':row.to_dict(),'terminal_time':cfg['terminal_time'],'coordinate_representation':'terminal_world_A',
        'regions':regions,'one_to_rest':evidence,'terminal_coords_A':xyz.tolist(),
        'objective':'Hypothesize why one high final-affinity molecule has useful spatial regions; not a causal or calibrated native-success conclusion',
        'comparison_coverage':{'rest_all':int(rest.sum()),'rest_valid':int(valid.sum()),'no_prior_topk':True},
        'region_kernel':manifest['region_kernel'],
        'limitations':[manifest['source_dependence'],'Anchor regions are fixed in the receptor frame; no per-molecule alignment',
            'Coordinate moments condition on nonempty support; all other candidates remain in mass and failure inventory',
            'These are untyped geometric measurements, not energies or binding contacts',
            'Final geometry hypothesis must be frozen before intermediate teacher retrieval']}
    write_json(out/'candidates'/row.candidate_id/'terminal_packet.json',packet)
    return packet


def trace_teacher(repo,cfg,packet):
    repo=Path(repo);candidate=packet['candidate'];batch=int(candidate['batch']);slot=int(candidate['slot'])
    source=repo/cfg['compact_dataset']/'results'/cfg['campaign'];path=source/cfg['arm']/f'batch_{batch:03d}'/'trajectory.h5'
    frame=read_json(source/f'frame_batch_{batch:03d}.json');com=np.asarray(frame['target_com']).reshape(-1,1,3)
    scale=read_json(source/'config.json')['coord_scale']
    with TrajectoryPackage(path) as tr:
        times=tr.read('score_time')[:,0];matches=np.flatnonzero(np.isclose(times,cfg['teacher_time'],rtol=0,atol=2e-6))
        if len(matches)!=1:raise ValueError('No exact real current-state teacher at configured clock')
        step=int(matches[0]);selected=tr.read('selected_indices').astype(int);ancestors=terminal_ancestors(selected)
        parent=int(ancestors[step,slot]);coords=tr.read('current_coords',step)[parent]*scale+com[parent]
        mask=tr.read('mask',step)[parent].astype(bool)
        if step:
            expected=tr.read('proposal_coords',step-1)[selected[step-1,parent]]
            if not np.array_equal(tr.read('current_coords',step)[parent],expected):raise ValueError('Ancestor coordinate-copy audit failed')
    if len(coords)!=len(packet['terminal_coords_A']) or not mask.all():raise ValueError('Slot identity cannot be transferred without an explicit map')
    teacher={'schema_version':'actual-current-teacher-1.0','candidate_id':candidate['candidate_id'],
        'terminal_slot':slot,'batch':batch,'step':step,'ancestor_slot':parent,'time':cfg['teacher_time'],
        'representation':'current_world_A','coords_A':coords.tolist(),'active_slots':np.flatnonzero(mask).tolist(),
        'trajectory_sha256':digest(path),'parent_copy_verified':True,
        'slot_mapping':'Identity of atom slots along this recorded lineage only; no inter-molecule atom correspondence',
        'terminal_score':float(candidate['pic50_on_rescore'])}
    write_json(repo/cfg['study']/'candidates'/candidate['candidate_id']/'teacher.json',teacher)
    return teacher
