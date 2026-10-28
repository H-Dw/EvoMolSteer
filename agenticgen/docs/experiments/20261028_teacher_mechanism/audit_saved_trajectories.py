"""Read-only saved-trajectory audit; never loads a model or generates candidates."""
from pathlib import Path
import json, gzip, hashlib, datetime
import numpy as np
import torch, h5py
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski

BASE=Path('/workspace/agenticgen_uniguide_flowr_20261026')
JOINT=Path('/workspace/agenticgen_joint_inpainting_20261027')
TIME=Path('/workspace/agenticgen_time_window_20261028')
OUT=Path('/workspace/uniguide_audits/teacher-raw999-analysis-20261028')
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def pt(p):
    with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
teachers=read(BASE/'teachers.json')
if isinstance(teachers,dict):teachers=teachers.get('teachers',list(teachers.values()))
groups=read(TIME/'RESULTS.json')['groups']
if isinstance(groups,list):groups={g['target']+'/'+g['arm']:g for g in groups}
result=dict(time_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),rdkit_version=rdBase.rdkitVersion,teachers={},groups={},traces={},source_hashes={},no_model_execution=True)
config=read(BASE/'config.json')
for teacher in teachers:
    target=teacher['target'];ids=teacher['core_sdf_atoms'];core=np.asarray(teacher['core_A']);sdf=teacher['sdf']
    mol=next(m for m in Chem.SDMolSupplier(sdf,removeHs=True) if m is not None)
    edges=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),float(b.GetBondTypeAsDouble())) for b in mol.GetBonds()]
    inside=[e for e in edges if e[0] in ids and e[1] in ids];boundary=[e for e in edges if (e[0] in ids)!=(e[1] in ids)]
    neighbors={i:set() for i in ids}
    for a,b,_ in inside:neighbors[a].add(b);neighbors[b].add(a)
    left=set(ids);components=[]
    while left:
        todo=[min(left)];visited=set()
        while todo:
            i=todo.pop()
            if i in visited:continue
            visited.add(i);todo.extend(neighbors[i]-visited)
        left-=visited;components.append(sorted(visited))
    dm=np.linalg.norm(core[:,None]-core[None,:],axis=-1);dist=dm[np.triu_indices(len(ids),1)]
    meta=read(JOINT/'features'/f'{target}.json')
    r0=read(BASE/'R0'/target/'aggregate.json')
    result['teachers'][target]=dict(node_id=teacher['node_id'],heavy_atoms=mol.GetNumHeavyAtoms(),core_atom_count=len(ids),core_fraction=len(ids)/mol.GetNumHeavyAtoms(),smiles=Chem.MolToSmiles(mol),rings=mol.GetRingInfo().NumRings(),rotatable_bonds=Lipinski.NumRotatableBonds(mol),core_sdf_atoms=ids,core_elements=meta['atomic_symbols'],core_bonds=inside,boundary_bonds=boundary,core_connected_components=components,core_singular_values_A=np.linalg.svd(core-core.mean(0),compute_uv=False).tolist(),core_pair_distance_min_A=float(dist.min()),core_pair_distance_max_A=float(dist.max()),r0={k:r0.get(k) for k in ['C','C_se','mean_pic50','built','attempts']},feature_metadata=meta)
    result['source_hashes'][sdf]=sha(sdf)
for key,g in groups.items():
    if not key.endswith('__window'):continue
    folder=Path(g['folder']);target=g['target'];teacher=next(t for t in teachers if t['target']==target)
    rows=read(folder/'steps.json');N=len(rows[0]['loss'])
    increments=np.array([r['max_atom_increment_A'] for r in rows]);switches=np.array([r['assignment_switch_fraction'] for r in rows]);loss=np.array([r['loss'] for r in rows])
    peak=np.unravel_index(np.argmax(increments),increments.shape)
    item=dict(folder=str(folder),max_increment_A=float(increments[peak]),max_increment_step=int(rows[peak[0]]['step']),max_increment_row=int(peak[1]),switch_mean_active=float(switches[[10<=r['step']<=90 for r in rows]].mean()),switch_mean_late=float(switches[[r['step']>=91 for r in rows]].mean()),per_sample_max_increment_A=increments.max(0).tolist(),step_mean_rmsd_A=np.sqrt(2*loss).mean(1).tolist(),schedule={str(r['step']):r['scale'] for r in rows if r['step'] in [10,26,51,76,91]})
    output=pt(folder/'terminal.pt.gz');xyz=output['coords'].numpy();mask=output['mask'].bool().numpy();core=np.asarray(teacher['core_A']);assign=[]
    for i in range(N):
        cost=((core[:,None]-xyz[i][None])**2).sum(-1);cost[:,~mask[i]]=np.inf
        _,a=linear_sum_assignment(cost);assign.append(a)
    assign=np.array(assign);atomic=output['atomics'].argmax(-1).numpy();charge=output['charges'].argmax(-1).numpy();bonds=output['bonds'].argmax(-1).numpy()
    meta=result['teachers'][target]['feature_metadata'];expected_bonds=np.array(meta['induced_bond_classes'])
    # All teacher core charges are neutral; infer the neutral class from an audited hard-inpainted state.
    c=pt(JOINT/'groups'/target/'chemical_raw999'/'checkpoint_100.pt.gz')
    a=c['previous_assignment'][0];neutral=c['current']['charges'][0,a].argmax(-1).unique()
    assert len(neutral)==1 and set(meta['formal_charges'])=={0}
    expected_atoms=c['current']['atomics'][0,a].argmax(-1).numpy()
    ap=atomic[np.arange(N)[:,None],assign]==expected_atoms[None]
    cp=charge[np.arange(N)[:,None],assign]==int(neutral[0])
    pairs=np.triu_indices(len(expected_atoms),1);bp=bonds[np.arange(N)[:,None],assign[:,pairs[0]],assign[:,pairs[1]]]==expected_bonds[pairs][None]
    positive=expected_bonds[pairs]>0
    item['terminal_geometric_assignment_feature_agreement']=dict(atomic_fraction=float(ap.mean()),charge_fraction=float(cp.mean()),all_pair_bond_fraction=float(bp.mean()),positive_bond_fraction=float(bp[:,positive].mean()),molecules_all_atoms=int(ap.all(1).sum()),molecules_all_core_bonds=int(bp.all(1).sum()),molecules_full_core=int((ap.all(1)&cp.all(1)&bp.all(1)).sum()),per_sample_atom_fraction=ap.mean(1).tolist(),per_sample_bond_fraction=bp.mean(1).tolist(),assignment=assign.tolist(),neutral_charge_class=int(neutral[0]))
    result['groups'][key]=item
    result['source_hashes'][str(folder/'steps.json')]=sha(folder/'steps.json');result['source_hashes'][str(folder/'terminal.pt.gz')]=sha(folder/'terminal.pt.gz')
    if target.startswith('6pyy') and ('chemical' in key or 'gradient__raw999' in key):
        track=[]
        with h5py.File(folder/'trace.h5','r') as f:
            for r in rows:
                step=r['step'];row=2;x=f['current_A'][step,row];end=f['endpoint_A'][step,row];delta=f['guidance_delta_A'][step,row];a=f['assignment'][step,row].astype(int)
                distances=np.linalg.norm(x[:,None]-x[None,:],axis=-1);np.fill_diagonal(distances,np.inf)
                enddist=np.linalg.norm(end[:,None]-end[None,:],axis=-1);np.fill_diagonal(enddist,np.inf)
                track.append(dict(step=step,time=r['time'],scale=r['scale'],coefficient=r['coefficient'][row],rmsd_A=float(np.sqrt(2*r['loss'][row])),gradient_norm=r['gradient_norm'][row],max_increment_A=r['max_atom_increment_A'][row],delta_native_ratio=r['delta_native_ratio'][row],assignment_switch_fraction=r['assignment_switch_fraction'][row],assignment=a.tolist(),current_max_distance_from_core_center_A=float(np.linalg.norm(x-core.mean(0),axis=-1).max()),current_min_pair_distance_A=float(distances.min()),predicted_min_pair_distance_A=float(enddist.min()),predicted_max_distance_from_core_center_A=float(np.linalg.norm(end-core.mean(0),axis=-1).max()),max_increment_atom=int(np.argmax(np.linalg.norm(delta,axis=-1)))))
        result['traces'][key]=track
    if (folder/'NETWORK_TIMES.json').exists():
        clock=read(folder/'NETWORK_TIMES.json');item['clock_selected']=[v for v in clock if v['phase'] in ['integration:0','integration:10','integration:49','integration:50','integration:99','endpoint_rescore']]
    if (folder/'OUTPUT_POLICY.json').exists():item['output_policy']=read(folder/'OUTPUT_POLICY.json')
result['source_hashes_unchanged_after_read']=all(sha(p)==v for p,v in result['source_hashes'].items())
native=Path('/opt/MolSteer/flowr_root/flowr/models/fm_pocket.py');builder=native.with_name('mol_builder.py')
result['native_source_sha256']={str(p):sha(p) for p in [native,builder]}
c=torch.load(config['checkpoint'],map_location='cpu',weights_only=False);h=c['hyper_parameters']
result['checkpoint']=dict(path=config['checkpoint'],sha256_bound_in_config=config['checkpoint_sha256'],relevant_hparams={k:v for k,v in h.items() if k in ['mixed_uncond_inpaint','func_group_inpainting','scaffold_inpainting','linker_inpainting','fragment_inpainting','core_inpainting','interaction_inpainting','coord_scale','remove_aromaticity','add_feats']})
p=OUT/'evidence.json';p.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(dict(path=str(p),sha256=sha(p),groups=len(result['groups']),protected_sources=len(result['source_hashes']),unchanged=result['source_hashes_unchanged_after_read'])))
