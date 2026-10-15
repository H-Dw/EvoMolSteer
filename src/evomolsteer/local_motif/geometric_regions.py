"""Discover untyped spatial-contrast patches, including nonpolar scaffold areas."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from .regional_context import context
from .regional_trajectories import field


def extend(repo,m,e,templates):
    output=repo/m['campaign_study']/'geometric_regions_v1.json'
    if output.exists():
        extra=read_json(output)
        if extra['base_evidence_sha256']!=digest(repo/m['campaign_study']/'evidence.json'):raise ValueError('Geometric registry source changed')
    else:
        source=Path(m['layout_report']);atoms=pd.read_csv(source/'all_atoms.csv.gz');pocket=pd.read_csv(source/'reference_inputs/inputs/pocket_atoms.csv')
        members=pd.read_csv(source/'reference_inputs/inputs/frozen_membership.csv');bonds=pd.read_csv(source/'reference_inputs/inputs/final_bonds.csv.gz')
        coords={node:g.sort_values('atom_index')[['x_A','y_A','z_A']].to_numpy() for node,g in atoms.groupby('node_id')}
        regions=[];cards={};catalog=[]
        existing={t['candidate_id']:{tuple(r['atom_slots']) for r in e['regions'] if r['candidate_id']==t['candidate_id']} for t in templates.values()}
        for tid,t in templates.items():
            node=t['evidence_origin'];x=np.asarray(t['terminal_coords_A']);membership=members[members.node_id.eq(node)].iloc[0]
            same=members.loc[members.smiles.eq(membership.smiles)&~members.node_id.eq(node),'node_id'].tolist()
            kind='same_graph'
            if len(same)<2:
                same=members.loc[members.root_id.eq(membership.root_id)&~members.node_id.eq(node),'node_id'].tolist();kind='same_root_different_graph_possible'
            if not same:continue
            unique={coords[n].tobytes():n for n in same if coords[n].shape==x.shape};nodes=list(unique.values())
            if not nodes:continue
            controls=np.stack([coords[n] for n in nodes]);bb=bonds[bonds.node_id.eq(node)]
            radius=float(bb.length_A.median()*m.get('geometric_radius_bond_multiple',2.))
            proposals=[]
            for center in range(len(x)):
                slots=np.flatnonzero(np.linalg.norm(x-x[center],axis=1)<=radius).tolist()
                if not 3<=len(slots)<=m['max_local_points'] or len(slots)>=len(x) or tuple(slots) in existing[t['candidate_id']]:continue
                centers=x[slots];focal=field(x[None],centers)[0];other=field(controls,centers);delta=focal-other.mean(0)
                bulk=x.mean(0)-controls.mean((0,1));adjusted=delta.copy();adjusted[1:]-=bulk
                scales=np.maximum(other.std(0),[.01,.1,.1,.1]);rank=float(np.linalg.norm(adjusted/scales))
                proposals.append({'center_slot':center,'atom_slots':slots,'rank':rank,'focal':focal.tolist(),
                    'control_mean':other.mean(0).tolist(),'delta':delta.tolist(),'bulk_translation_A':bulk.tolist(),
                    'bulk_adjusted_delta':adjusted.tolist(),'n_controls':len(nodes),'control_kind':kind})
            used=set(existing[t['candidate_id']])
            for z in sorted(proposals,key=lambda z:(-z['rank'],z['center_slot'])):
                if tuple(z['atom_slots']) in used:continue
                used.add(tuple(z['atom_slots']));slots=z['atom_slots'];rid=t['candidate_id']+'__G%02d'%z['center_slot']
                rc=context(node,slots,atoms,pocket,members);dist=np.linalg.norm(pocket[['x_A','y_A','z_A']].to_numpy()[:,None]-x[slots][None],axis=-1).min(1)
                nearest=pocket.iloc[np.argsort(dist,kind='stable')[:6]]
                r={'region_id':rid,'template_id':tid,'candidate_id':t['candidate_id'],'node_id':node,'atom_slots':slots,
                    'claim':'Untyped local spatial contrast beyond bulk translation; exploratory candidate-conditioned scaffold region',
                    'region_family':'spatial_contrast','source_score':t['terminal_score'],'n_points':len(slots),'n_total':len(x),
                    'native_elements':[t['native_elements'][i] for i in slots],'base_band_A':float(bb.length_A.median()*.25),
                    'band_basis':'Proposed geometry response calibration, not an established optimum',
                    'pocket_sites':[{'index':int(q.pocket_index),'coords_A':[q.x_A,q.y_A,q.z_A],'id':q.residue+':'+q.atom_name} for q in nearest.itertuples()],
                    'orientation_triplets':[[i,*t['full_adjacency'][str(i)][:2]] for i in slots if len(t['full_adjacency'][str(i)])>=2 and set(t['full_adjacency'][str(i)][:2])<=set(slots)][:3]}
                rc['untyped_spatial_contrast']={**z,'control_node_ids':nodes,'descriptor':['spatial_mass','signed_dx_A','signed_dy_A','signed_dz_A'],
                    'limits':'Candidate-conditioned descriptor ranking, not affinity attribution or significance. Bulk subtraction is a geometric contrast proxy; matched roots remain dependent.'}
                local_points=np.asarray(t['coords_A'])[slots];centered=local_points-local_points.mean(0);cov=centered.T@centered/len(slots)
                rc['observed_local_point_moments']={'representation':'actual_teacher_time_world_A','time':t['time'],
                    'centroid_A':local_points.mean(0).tolist(),'covariance_A2':cov.tolist(),
                    'principal_spreads_A':np.sqrt(np.maximum(np.linalg.eigvalsh(cov),0)).tolist(),
                    'limits':'Unweighted source local-point target; distinct from Gaussian whole-cloud discovery fields. Not a proven affinity mechanism.'}
                regions.append(r);cards[rid]={'context':rc,'conditional_layout':e['cards'][next(k for k in e['cards'] if k.startswith(t['candidate_id']+'__'))]['conditional_layout'],
                    'source':{'teacher_time':t['time'],'teacher_representation':t['representation'],'trajectory_sha256':t['trajectory_sha256'],'counterfactual_teacher':False}}
                catalog.append({k:r[k] for k in ['region_id','candidate_id','claim','n_points','n_total','source_score','native_elements']})
                if sum(rr['candidate_id']==t['candidate_id'] for rr in regions)>=3:break
        extra={'regions':regions,'cards':cards,'catalog':catalog,'base_evidence_sha256':digest(repo/m['campaign_study']/'evidence.json')}
        write_json(output,extra)
    e=copy.deepcopy(e);e['regions']+=extra['regions'];e['cards'].update(extra['cards']);e['catalog']+=extra['catalog']
    return e,extra['regions']
