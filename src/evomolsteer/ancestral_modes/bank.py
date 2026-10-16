"""Actual ancestor/prediction lookup, chemical supports, and full-window evidence."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from evomolsteer.local_motif.state_controls import full_state_controls


def lookup(mode, time, field='coords_A'):
    clocks=np.asarray(mode['path']['times']);a=np.asarray(mode['path'][field])
    if time<clocks[0]-2e-6 or time>clocks[-1]+2e-6:raise ValueError('Ancestor extrapolation forbidden')
    time=float(np.clip(time,clocks[0],clocks[-1]));j=int(np.searchsorted(clocks,time))
    if j==0:return a[0]
    if j==len(clocks):return a[-1]
    w=(time-clocks[j-1])/(clocks[j]-clocks[j-1]);return a[j-1]*(1-w)+a[j]*w


def prepare(repo,m):
    out=repo/m['campaign_study'];out.mkdir(parents=True,exist_ok=True)
    if (out/'bank.json').exists():return read_json(out/'bank.json'),read_json(out/'regions.json')
    old=repo/m['prior_study'];templates=read_json(old/'templates.json');regions=read_json(old/'evidence.json')['regions']
    report=Path(m['layout_report']);atoms=pd.read_csv(report/'all_atoms.csv.gz')
    pocket=pd.read_csv(report/'reference_inputs/inputs/pocket_atoms.csv')
    members=pd.read_csv(report/'reference_inputs/inputs/frozen_membership.csv')
    bank={};seen=set();scale=read_json(repo/m['compact_dataset']/'results'/m['campaign']/'config.json')['coord_scale']
    for tid,t in templates.items():
        if t.get('counterfactual_labels') or t['candidate_id'] in seen:continue
        seen.add(t['candidate_id']);v=copy.deepcopy(t);base=repo/m['compact_dataset']/'results'/m['campaign']
        path=base/m['arm']/f"batch_{t['batch']:03d}"/'trajectory.h5'
        com=np.asarray(read_json(base/f"frame_batch_{t['batch']:03d}.json")['target_com']).reshape(-1,1,3)
        with TrajectoryPackage(path) as tr:
            times=tr.read('score_time')[:,0];anc=terminal_ancestors(tr.read('selected_indices').astype(int))
            ids=np.flatnonzero((times>=m['learning_window'][0]-2e-6)&(times<=m['learning_window'][1]+2e-6))
            fields={'current_coords':'coords_A','predicted_coords':'endpoint_A','predicted_atomics_probs_f16':'element_probs',
                    'predicted_charges_probs_f16':'charge_probs','current_atomics':'current_atomics','current_charges':'current_charges'}
            profile={'times':times[ids].astype(float).tolist(),'ancestor_slots':anc[ids,t['terminal_slot']].tolist()}
            for source,key in fields.items():
                data=[]
                for step in ids:
                    slot=int(anc[step,t['terminal_slot']]);a=tr.read(source,int(step))[slot]
                    if source.endswith('coords'):a=a*scale+com[slot,0]
                    data.append(np.asarray(a).tolist())
                profile[key]=data
        a=atoms[atoms.node_id.eq(t['evidence_origin'])].sort_values('atom_index')
        v['terminal_charges']=[v['charge_vocabulary'][str(int(q))] for q in a.formal_charge]
        v['terminal_roles']=a[['atom_index','element','aromatic','acceptor','donor','formal_charge']].to_dict('records')
        member=members[members.node_id.eq(t['evidence_origin'])].iloc[0]
        v.update(root_id=member.root_id,smiles=member.smiles,path=profile,mode_id=tid)
        bank[tid]=v
    # Additional local support proposals have explicit chemical or interface reasons.
    extras=[]
    for tid,v in bank.items():
        x=np.asarray(v['terminal_coords_A']);roles=v['terminal_roles'];adj={int(k):vv for k,vv in v['full_adjacency'].items()}
        pxyz=pocket[['x_A','y_A','z_A']].to_numpy();dist=np.linalg.norm(x[:,None]-pxyz[None],axis=-1)
        specs=[]
        for r in roles:
            i=int(r['atom_index'])
            if r['acceptor'] or r['donor']:
                specs.append(('interface'+str(i),sorted({i,*adj[i]}),'interface_role',f"Observed {r['element']} polar site plus bonded direction neighbors"))
            if r['aromatic'] and r['element']=='N':
                specs.append(('NC_local'+str(i),sorted({i,*adj[i]}),'NC_layout','Observed aromatic N and neighboring scaffold positions; not a certified early acceptor'))
            if r['element']=='O':
                ids={i,*adj[i]};ids|={j for k in adj[i] for j in adj[k]}
                specs.append(('group'+str(i),sorted(ids),'functional_group','O-bearing group and attachment neighbors; microstate kept explicit'))
        closest=sorted(np.argsort(dist.min(1),kind='stable')[:m['small_support_points']].tolist())
        specs.append(('interface_geometry',closest,'interface_geometry','Closest observed interface points; geometric comparator without role selection'))
        inherited_count=len(read_json(repo/m['initial_program'])['atom_slots'])
        broad=sorted(np.argsort(dist.min(1),kind='stable')[:inherited_count].tolist())
        specs.append(('countmatched_reference',broad,'countmatched_reference','Proper-subset geometric comparator matched to inherited reference count, not a minimal chemical support'))
        used=set()
        for suffix,slots,family,reason in specs:
            allowed=m['max_new_local_points'] if family!='countmatched_reference' else inherited_count
            if not 2<=len(slots)<=allowed or len(slots)>=len(x) or (tuple(slots),family) in used:continue
            used.add((tuple(slots),family));center=x[slots]
            nearest=np.argsort(np.linalg.norm(center[:,None]-pxyz[None],axis=-1).min(0),kind='stable')[:4]
            sites=[{'coords_A':pxyz[j].tolist(),'id':pocket.iloc[j].residue+':'+pocket.iloc[j].atom_name,'index':int(pocket.iloc[j].pocket_index)} for j in nearest]
            trips=[[i,*adj[i][:2]] for i in slots if len(adj[i])>=2 and set(adj[i][:2])<=set(slots)]
            extras.append({'region_id':v['candidate_id']+'__v3_'+suffix,'template_id':tid,'candidate_id':v['candidate_id'],
                'atom_slots':slots,'family':family,'claim':reason,'source_score':v['terminal_score'],'pocket_sites':sites,
                'orientation_triplets':trips,'base_band_A':.35,'scale_source':'Uncalibrated exploratory geometric band; not an interaction energy'})
    for r in regions:r.setdefault('family','inherited')
    regions+=extras
    value={'schema_version':'actual-ancestor-bank.v3','learning_window':m['learning_window'],'modes':bank,
        'source_report_sha256':digest(report/'Top10_spatial_motif_mining.zh-CN.html'),
        'limits':['Finite historical selected paths; not calibrated continuation probabilities.',
                  'Terminal roles select support; they are not asserted in early states.',
                  'Element/charge prediction probabilities retain original float16 quantization.']}
    write_json(out/'bank.json',value);write_json(out/'regions.json',regions)
    write_json(out/'bank_manifest.json',{'modes':len(bank),'regions':len(regions),'bank_sha256':digest(out/'bank.json'),
        'source_trajectories':{k:v['trajectory_sha256'] for k,v in bank.items()},'root_ids':sorted({v['root_id'] for v in bank.values()})})
    return value,regions


def regional_evidence(repo,m,bank,region):
    """Selection acts on proposal states; exclude updates beyond learned state time."""
    out=repo/m['campaign_study']/'regional_evidence';path=out/(region['region_id']+'.json')
    if path.exists():return read_json(path)
    mode=bank['modes'][region['template_id']];slots=region['atom_slots'];base=repo/m['compact_dataset']/'results'/m['campaign']
    source=base/m['arm']/f"batch_{mode['batch']:03d}"/'trajectory.h5'
    com=np.asarray(read_json(base/f"frame_batch_{mode['batch']:03d}.json")['target_com']).reshape(-1,1,3)
    scale=read_json(base/'config.json')['coord_scale'];rows=[]
    with TrajectoryPackage(source) as tr:
        times=tr.read('score_time')[:,0];state_times=tr.read('state_time');anc=terminal_ancestors(tr.read('selected_indices').astype(int))
        for step,t in enumerate(times):
            if not m['learning_window'][0]-2e-6<=t<=m['learning_window'][1]+2e-6:continue
            x=tr.read('current_coords',step)*scale+com;at=tr.read('current_atomics',step);parent=int(anc[step,mode['terminal_slot']])
            unique=full_state_controls(tr,step,x,at,parent);target=lookup(mode,float(t))[slots]
            def observable(clouds):
                selected=[]
                for cloud in clouds:
                    rr,cc=linear_sum_assignment(np.square(cloud[:,None]-target).sum(-1));selected.append(cloud[rr[np.argsort(cc)]])
                z=np.asarray(selected);delta=z-target;kernel=np.exp(-np.square(delta).sum(-1)/(2*m['field_sigma_A']**2))
                return np.column_stack([kernel.mean(1),(kernel[...,None]*delta).mean(1)])
            phi=observable(x);back=phi[unique]
            row={'time':float(t),'state_time':float(state_times[step]),'focal':phi[parent].tolist(),
                 'background_mean':back.mean(0).tolist() if len(back) else [0.]*4,
                 'background_sd':back.std(0).tolist() if len(back) else [0.]*4,'unique_controls':len(unique)}
            if state_times[step]<=m['learning_window'][1]+2e-6 and tr.read('resampled',step):
                proposal=tr.read('proposal_coords',step)*scale+com;v=observable(proposal);w=tr.read('selection_probability',step).astype(float);w/=w.sum()
                row.update(native_shift=(v.mean(0)-phi.mean(0)).tolist(),expected_selection_shift=(w@v-v.mean(0)).tolist(),
                    realized_selection_shift=(v[tr.read('selected_indices',step).astype(int)].mean(0)-v.mean(0)).tolist())
            rows.append(row)
    frame=pd.DataFrame(rows);curves={}
    for key in ('focal','background_mean','expected_selection_shift','native_shift'):
        valid=[r for r in rows if key in r]
        if len(valid)<3:continue
        tt=np.asarray([r['time'] for r in valid]);yy=np.asarray([r[key] for r in valid]);coef=np.polynomial.polynomial.polyfit(tt,yy,2)
        curves[key]={'coefficients':coef.tolist(),'derivative_coefficients':(coef[1:]*np.arange(1,3)[:,None]).tolist(),
            'rmse':np.sqrt(np.square(np.polynomial.polynomial.polyval(tt,coef).T-yy).mean(0)).tolist()}
    result={'region_id':region['region_id'],'source_sha256':digest(source),'domain':m['learning_window'],'rows':rows,'full_window_fits':curves,
        'units':['dimensionless','A','A','A'],'selection_clock_note':'Weights from score_time; applied proposal at state_time. Proposal beyond learning window excluded.',
        'limits':'Descriptive single-success path against unique complete-state contemporaries; not independent time repeats or terminal failure labels.'}
    write_json(path,result);return result
