"""Pure coordinate fields and compact score-stratified Steer teacher libraries."""
import gzip,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import spearmanr,ttest_1samp
from ..io import read_json,write_json,write_table,digest
from ..storage.trajectory import TrajectoryPackage
from .coordinate_mining import bh,curve_fit

PAIR_RADII=(1.5,2.5,3.5,5.,7.)

def landmarks(pdb,ligand,count=20):
    from rdkit import Chem
    mol=next(m for m in Chem.SDMolSupplier(str(ligand),removeHs=False) if m is not None)
    xyz=mol.GetConformer().GetPositions()
    rows=[]
    for line in Path(pdb).read_text().splitlines():
        if line[:6].strip()!='ATOM' or line[76:78].strip()=='H':continue
        p=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
        if np.linalg.norm(p[None]-xyz,axis=1).min()<=8.:rows.append(p)
    points=np.asarray(rows);chosen=[int(np.linalg.norm(points-xyz.mean(0),axis=1).argmin())]
    while len(chosen)<min(count,len(points)):
        dist=np.linalg.norm(points[:,None]-points[chosen][None],axis=2).min(1);dist[chosen]=-1
        chosen.append(int(dist.argmax()))
    return points[chosen],xyz.mean(0)

def names(points):
    return ['global_centroid_'+a for a in 'xyz']+['shape_'+a for a in ('xx','yy','zz','xy','xz','yz')]+[
        'pair_kernel_'+str(r) for r in PAIR_RADII]+[f'landmark_{i:02d}_{v}' for i in range(len(points)) for v in ('softmin','occupancy3','occupancy5')]

def numpy_geometry(x,points,origin):
    center=x.mean(1);r=x-center[:,None];cov=np.einsum('bni,bnj->bij',r,r)/x.shape[1]
    parts=[center-origin,np.stack([cov[:,i,j]*(1 if i==j else np.sqrt(2)) for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]],1)]
    d=np.sqrt(((x[:,:,None]-x[:,None])**2).sum(-1)+1e-20);edge=~np.eye(x.shape[1],dtype=bool)
    parts.append(np.stack([np.exp(-.5*((d[:,edge]-a)/.75)**2).mean(1) for a in PAIR_RADII],1))
    d=np.sqrt(((x[:,:,None]-points[None,None])**2).sum(-1)+1e-20)
    field=np.stack([-.5*(logsumexp(-d/.5,axis=1)-np.log(x.shape[1])),
                    np.exp(-.5*((d-3)/1.)**2).mean(1),np.exp(-.5*((d-5)/1.)**2).mean(1)],-1)
    parts.append(field.reshape(len(x),-1));return np.concatenate(parts,1)

def torch_geometry(x,points,origin):
    points=x.new_tensor(points);origin=x.new_tensor(origin);center=x.mean(1);r=x-center[:,None]
    cov=torch.einsum('bni,bnj->bij',r,r)/x.shape[1]
    parts=[center-origin,torch.stack([cov[:,i,j]*(1 if i==j else np.sqrt(2)) for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]],1)]
    d=((x[:,:,None]-x[:,None]).square().sum(-1)+1e-20).sqrt();edge=~torch.eye(x.shape[1],device=x.device,dtype=torch.bool)
    parts.append(torch.stack([torch.exp(-.5*((d[:,edge]-a)/.75).square()).mean(1) for a in PAIR_RADII],1))
    d=((x[:,:,None]-points[None,None]).square().sum(-1)+1e-20).sqrt()
    field=torch.stack([-.5*(torch.logsumexp(-d/.5,dim=1)-np.log(x.shape[1])),
                       torch.exp(-.5*((d-3)/1.).square()).mean(1),torch.exp(-.5*((d-5)/1.).square()).mean(1)],-1)
    parts.append(field.reshape(len(x),-1));return torch.cat(parts,1)

def mine(dataset,campaign,output,batches=range(14),window=(0.,.5),teachers_per_batch=2):
    root=Path(dataset);source=root/'results'/campaign;out=Path(output)
    if out.exists():raise FileExistsError(out)
    cfg=read_json(source/'config.json');points,origin=landmarks(root/'inputs/3PE1_protein_aligned.pdb',root/'inputs/3PE1_ligand_aligned.sdf')
    fields=names(points);nodes={};sources=[];effects=[];all_var=[]
    for b in batches:
        path=source/'single'/f'batch_{b:03d}'/'trajectory.h5'
        com=np.asarray(read_json(source/f'frame_batch_{b:03d}.json')['target_com'])[:,None]
        with TrajectoryPackage(path) as z:
            times=np.round(z.read('score_time')[:,0].astype(float),6);states=z.read('state_time')
            ids=np.flatnonzero(z.read('resampled')&(times>=window[0]-1e-6)&(states<=window[1]+1e-6))
            for i in ids:
                t=float(times[i]);x=z.read('proposal_coords',int(i)).astype(float)*cfg['coord_scale']+com
                y=z.read('predicted_coords',int(i)).astype(float)*cfg['coord_scale']+com
                score=z.read('pic50_on',int(i)).astype(float).reshape(-1)
                if not z.read('mask',int(i)).all():raise ValueError('Matched fixed active atom slots required')
                v=numpy_geometry(x,points,origin);vy=numpy_geometry(y,points,origin)
                lo,hi=np.quantile(score,[.25,.75]);low=score<=lo;high=score>=hi
                if min(low.sum(),high.sum())<3:raise ValueError('Insufficient score strata')
                # One elite per mode; add a spatially distinct second teacher.
                rank=np.argsort(-score,kind='stable');elite=[int(rank[0])]
                for candidate in rank[1:]:
                    if len(elite)>=teachers_per_batch:break
                    if min(np.linalg.norm(vy[candidate]-vy[j]) for j in elite)>1e-3:elite.append(int(candidate))
                nodes.setdefault(t,[]).append({'batch':int(b),'high':v[high].mean(0),'low':v[low].mean(0),
                    'variance':v.var(0),'endpoint_high':vy[high].mean(0),'endpoint_low':vy[low].mean(0),
                    'teachers_proposal_A':x[elite].astype(np.float32).tolist(),'teachers_endpoint_A':y[elite].astype(np.float32).tolist(),
                    'teacher_scores':score[elite].tolist(),'teacher_slots':elite,'score_high_mean':float(score[high].mean()),
                    'score_low_mean':float(score[low].mean()),'roots':int(np.unique(z.read('root_slot',int(i))).size),
                    'correlation':[float(spearmanr(v[:,j],score).statistic) if v[:,j].std()>1e-12 else 0. for j in range(v.shape[1])]})
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    times=sorted(nodes);grid=np.asarray(times)
    if len(grid)<2 or not np.isclose(grid[0],window[0]) or not np.isclose(grid[-1]+np.median(np.diff(grid)),window[1]):
        raise ValueError('Complete actual-state selection window required')
    variance=np.array([np.mean([r['variance'] for r in nodes[t]],0) for t in grid])
    scale=np.sqrt(np.trapezoid(variance,grid,axis=0)/(grid[-1]-grid[0])).clip(1e-5)
    delta=np.array([[r['high']-r['low'] for r in nodes[t]] for t in grid]).transpose(1,0,2)/scale
    corr=np.array([[r['correlation'] for r in nodes[t]] for t in grid]).transpose(1,0,2)
    integrated=np.trapezoid(delta,grid,axis=1)/(grid[-1]-grid[0]);icorr=np.trapezoid(corr,grid,axis=1)/(grid[-1]-grid[0])
    rng=np.random.default_rng(42);boot=integrated[rng.integers(0,len(integrated),(2000,len(integrated)))].mean(1)
    functions={};pvalues=[]
    for j,name in enumerate(fields):
        ci=np.quantile(boot[:,j],[.025,.975]);p=float(ttest_1samp(integrated[:,j],0).pvalue);pvalues.append(p)
        functions[name]=curve_fit(grid,delta[:,:,j],max_degree=3)
        effects.append({'feature':name,'high_low_integrated_z':float(integrated[:,j].mean()),'CI_low':float(ci[0]),'CI_high':float(ci[1]),
                        'score_correlation_integrated':float(icorr[:,j].mean()),'positive_batch_fraction':float((integrated[:,j]>0).mean()),
                        'endpoint_minus_start_delta_z':float(delta[:,-1,j].mean()-delta[:,0,j].mean()),'p':p})
    for row,q in zip(effects,bh(np.nan_to_num(pvalues,nan=1.))):row['q']=float(q)
    frames=[]
    for t in grid:
        group=nodes[t];frames.append({'time':float(t),'high_scaled':(np.array([r['high'] for r in group])/scale).tolist(),
            'low_scaled':(np.array([r['low'] for r in group])/scale).tolist(),
            'variance_scaled':(np.array([r['variance'] for r in group])/scale**2+.05).tolist(),
            'teacher_proposal_A':sum([r['teachers_proposal_A'] for r in group],[]),
            'teacher_endpoint_A':sum([r['teachers_endpoint_A'] for r in group],[]),'teacher_scores':sum([r['teacher_scores'] for r in group],[]),
            'teacher_batches':sum([[r['batch']]*len(r['teacher_scores']) for r in group],[]),
            'root_count_min':min(r['roots'] for r in group)})
    from ..generation.launcher import INPUT_FILES
    ref={'schema_version':'affinity-coordinate-library-1.0','window':list(window),'times':times,'frames':frames,
        'landmarks_A':points.tolist(),'origin_A':origin.tolist(),'features':fields,'feature_scale':scale.tolist(),
        'control_representation':'proposal','spatial_anchor':'endpoint','representation':'actual native proposal, affinity-stratified coordinate teachers',
        'feature_unit':'mixed coordinate geometry; centroid/distances A, covariance A2, densities dimensionless',
        'required_input_sha256':{name:digest(root/'inputs'/name) for name in INPUT_FILES},'sources':sources,'discovery_batches':list(batches),
        'time_alignment':'score t labels its endpoint forecast; coordinate control proposal at t+dt, only states within learned window',
        'label_semantics':'Recorded pre-selection target head only; no t=1 labels and no additional network fits or per-step head calls',
        'limitations':['Within-event score association is observational and shares the original oracle','Dependent clones are not independent samples',
                       'Landmarks are coordinate occupancy fields, not chemical interaction labels','Teacher correspondence is conditional and not a FLOWR Jacobian']}
    out.mkdir(parents=True);path=out/'geometry_reference.json.gz'
    path.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    import pandas as pd
    pd.DataFrame(effects).to_parquet(out/'whole_window_effects.parquet',compression=None,index=False)
    write_json(out/'effect_functions.json',functions)
    write_json(out/'manifest.json',{'schema_version':'affinity-coordinate-mining-1.0','reference_sha256':digest(path),'reference_bytes':path.stat().st_size,
        'features':len(fields),'nodes':len(times),'window':list(window),'sources':sources,'discovery_batches':list(batches),
        'statistics_unit':'Equal independent generation batches, not clone or particle rows','storage':'Aggregated effects, functions and compact teacher library; no expanded particle cache'})
    return ref
