"""Local terminal chemistry, unmodified-pose geometry, energy and head-score reports."""
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem,Descriptors,QED,rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from posebusters import PoseBusters
from ..io import read_json,write_json,digest,clean
from .prototypes import measure_patch
from .window_reference import load_reference
from .terminal_statistics import converged_energy

RDLogger.DisableLog('rdApp.warning')


def energy_relaxation(mol,patch):
    """Same-graph MMFF94s local strain relief; neither binding energy nor global minimum."""
    result={'energy_status':'not_evaluated'}
    try:
        heavy=Chem.RemoveHs(Chem.Mol(mol));nh=heavy.GetNumAtoms();m=Chem.AddHs(heavy,addCoords=True)
        if not AllChem.MMFFHasAllMoleculeParams(m):return {'energy_status':'unsupported_MMFF94s'}
        props=AllChem.MMFFGetMoleculeProperties(m,mmffVariant='MMFF94s')
        ff=AllChem.MMFFGetMoleculeForceField(m,props,nonBondedThresh=100.)
        for i in range(nh):ff.AddFixedPoint(i)
        ff.Initialize();hs=int(ff.Minimize(maxIts=500));e_pose=float(ff.CalcEnergy())
        relaxed=Chem.Mol(m);full=AllChem.MMFFGetMoleculeForceField(relaxed,props,nonBondedThresh=100.)
        full.Initialize();status=int(full.Minimize(maxIts=1000));e_min=float(full.CalcEnergy())
        original=heavy.GetConformer().GetPositions();xyz=relaxed.GetConformer().GetPositions()[:nh]
        if not np.isfinite([e_pose,e_min]).all():raise ValueError('Nonfinite MMFF')
        shift=np.square(xyz-original).sum(1)
        result.update(energy_status='converged' if hs==0 and status==0 else 'not_converged',
                      mmff_H_relax_status=hs,mmff_full_relax_status=status,
                      mmff_pose_Hrelaxed_kcal_mol=e_pose,mmff_local_min_kcal_mol=e_min,
                      mmff_relief_kcal_mol=e_pose-e_min,mmff_relief_per_heavy=(e_pose-e_min)/nh,
                      relax_rms_patch_A=float(np.sqrt(shift[patch].mean())) if patch.any() else None,
                      relax_rms_surround_A=float(np.sqrt(shift[~patch].mean())) if (~patch).any() else None)
    except Exception as e:result.update(energy_status='error',energy_error=type(e).__name__+': '+str(e))
    return result


def protein_points(path):
    points=[]
    for line in Path(path).read_text().splitlines():
        if line.startswith(('ATOM  ','HETATM')) and line[76:78].strip()!='H':
            points.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return np.array(points)


def load_mols(folder):
    path=folder/'molecules_raw_decodable.sdf';mapping={}
    for mol in Chem.SDMolSupplier(str(path),sanitize=False,removeHs=False):
        if mol is not None:
            if not mol.HasProp('node_id'):raise ValueError('Unidentified molecule')
            key=mol.GetProp('node_id')
            if key in mapping:raise ValueError('Duplicate terminal slot')
            mapping[key]=mol
    return mapping


def evaluate_one(record,mol,points,reference):
    r=dict(record);r.update(valid_connected=False,sanitized=False,pb_fast_pass=False)
    if mol is None:r['failure_reason']='missing_raw_decode';return r,None
    try:
        m=Chem.Mol(mol);Chem.SanitizeMol(m);r['sanitized']=True
        r['connected']=len(Chem.GetMolFrags(m))==1
        r['radicals']=sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
        r['valid_connected']=r['connected'] and r['radicals']==0
        r['smiles']=Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True)
        r['scaffold']=MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        r['mw']=Descriptors.MolWt(m);r['qed']=QED.qed(m);r['heavy_atoms']=m.GetNumHeavyAtoms()
        if not r['valid_connected']:r['failure_reason']='disconnected_or_radical';return r,None
        heavy=Chem.RemoveHs(m);xyz=heavy.GetConformer().GetPositions()
        if not np.isfinite(xyz).all():raise ValueError('Nonfinite coordinates')
        catalog=reference['catalog'];patch_points=np.concatenate([np.asarray(v['points_A']) for v in catalog['regions'].values()])
        patch=np.linalg.norm(xyz[:,None,:]-patch_points[None],axis=-1).min(1)<=5.
        distance=np.linalg.norm(xyz[:,None,:]-points[None],axis=-1);closest=distance.min(1)
        r.update(min_protein_distance_A=float(closest.min()),severe_pairs_below_1_2A=int((distance<1.2).sum()),
                 patch_atoms=int(patch.sum()),surround_atoms=int((~patch).sum()),
                 patch_clash_atom_fraction=float((closest[patch]<1.2).mean()) if patch.any() else None,
                 surround_clash_atom_fraction=float((closest[~patch]<1.2).mean()) if (~patch).any() else None)
        labels=np.array([[catalog['atom_vocabulary'].get(a.GetSymbol(),-999) for a in heavy.GetAtoms()]])
        z=measure_patch(xyz[None],labels,np.ones(labels.shape,dtype=bool),catalog)[0]
        for i,name in enumerate(reference['features']):r[name]=float(z[i]) if np.isfinite(z[i]) else None
        if np.isfinite(z).all():
            target=reference['frames'][-1];delta=z-np.asarray(target['center_A'])
            r['terminal_patch_q_over_r2']=float(delta@np.linalg.inv(target['covariance_A2'])@delta/target['radius_squared'])
            r['terminal_patch_in_window_end_set']=r['terminal_patch_q_over_r2']<=1
        else:r['terminal_patch_in_window_end_set']=None
        r.update(energy_relaxation(heavy,patch))
        m.SetProp('_Name',r['node_id'])
        return r,m
    except Exception as e:
        r['failure_reason']=type(e).__name__+': '+str(e);return r,None


def summarize(rows):
    d=pd.DataFrame(rows);summary={'n':len(d),'valid_connected':int(d.valid_connected.sum()),'valid_rate':float(d.valid_connected.mean()),
        'pb_fast_pass':int(d.pb_fast_pass.sum()),'pb_fast_pass_rate':float(d.pb_fast_pass.mean())}
    valid=d[d.valid_connected];unique=valid.drop_duplicates('smiles')
    summary.update(unique_smiles=len(unique),unique_scaffolds=int(unique.scaffold.nunique()) if len(unique) else 0,
                   unique_yield=len(unique)/len(d),duplicates_among_valid=1-len(unique)/max(len(valid),1))
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in unique.smiles] if len(unique) else []
    similarities=[v for i in range(1,len(fps)) for v in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])]
    summary['unique_graph_diversity']=1-float(np.mean(similarities)) if similarities else None
    for group,frame in [('all',d),('valid',valid),('unique_valid',unique),('pb_fast',d[d.pb_fast_pass])]:
        for name in ['pic50_on_rescore','pic50_off_rescore','gap_rescore','qed','terminal_patch_q_over_r2','surround_clash_atom_fraction','relax_rms_surround_A']:
            values=pd.to_numeric(frame.get(name,pd.Series(dtype=float)),errors='coerce').dropna()
            summary[group+'_'+name+'_mean']=float(values.mean()) if len(values) else None
            summary[group+'_'+name+'_n']=len(values)
        for name in ['mmff_relief_kcal_mol','mmff_relief_per_heavy']:
            values=converged_energy(frame,name)
            summary[group+'_'+name+'_median']=float(values.median()) if len(values) else None
            summary[group+'_'+name+'_n']=len(values)
    summary['energy_failure_counts']=d.energy_status.fillna('not_applicable').value_counts().to_dict() if 'energy_status' in d else {}
    return summary


def evaluate_terminal(dataset,campaign,reference_path,output,arms,batches,workers=4):
    root=Path(dataset);campaign_root=root/'results'/campaign;out=Path(output)
    if not (campaign_root/'COMPLETE.json').exists():raise ValueError('Campaign incomplete')
    out.mkdir(parents=True,exist_ok=True);reference=load_reference(reference_path)
    from ..trajectory_source import pocket_input_path
    protein=pocket_input_path(root,'target_protein');points=protein_points(protein)
    sources=[];allrows=[];batch_summaries=[];pbrows=[]
    for arm in arms:
        for batch in batches:
            folder=campaign_root/arm/f'batch_{batch:03d}';records=read_json(folder/'final_records.json')
            if len({r['slot'] for r in records})!=len(records):raise ValueError('Duplicate slots')
            mols=load_mols(folder)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                evaluated=list(pool.map(lambda r:evaluate_one(r,mols.get(r['node_id']),points,reference),records))
            usable=[(i,m) for i,(r,m) in enumerate(evaluated) if m is not None]
            # Per-molecule call preserves an exact slot mapping; avoids silently dropping PB failures.
            pb=PoseBusters(config='dock_fast',max_workers=0)
            for i,m in usable:
                row=evaluated[i][0]
                try:
                    checks=pb.bust(mol_pred=m,mol_cond=str(protein),full_report=False)
                    if len(checks)!=1 or checks.shape[1]==0 or checks.isna().any().any():raise ValueError('Incomplete PoseBusters checks')
                    row['pb_fast_pass']=bool(checks.to_numpy(dtype=bool).all())
                    row['pb_fast_failed_checks']=[str(k) for k,v in checks.iloc[0].items() if not bool(v)]
                    pbrows.append({'node_id':row['node_id'],**checks.iloc[0].to_dict()})
                except Exception as e:row.update(pb_fast_pass=False,pb_error=type(e).__name__+': '+str(e))
            rows=[r for r,m in evaluated];allrows.extend(rows)
            batch_summaries.append({'arm':arm,'batch':batch,**summarize(rows)})
            for name in ['final_records.json','molecules_raw_decodable.sdf']:
                sources.append({'path':str(folder/name),'sha256':digest(folder/name)})
            print(json.dumps({'arm':arm,'batch':batch,'n':len(rows),'valid':sum(r['valid_connected'] for r in rows),'pb_fast':sum(r['pb_fast_pass'] for r in rows)}),flush=True)
    report={'schema_version':'terminal-evaluation-1.0','campaign':campaign,'evaluation_location':'local',
        'seed':read_json(campaign_root/'config.json')['experiment']['seed'],'reference_sha256':digest(reference_path),
        'structure_input':'raw decoded unmodified generated pose; common local sanitization; no pose repair before evaluation',
        'energy_definition':'MMFF94s energy decrease after same-graph local relaxation, with H-only preparation; not global strain, binding energy or free energy; nonconverged excluded with counts',
        'posebusters_definition':'dock_fast 0.6.5 structural subset, not full PB-valid; energy evaluated separately',
        'terminal_region_definition':'window-end learned endpoint acceptable set applied as a terminal diagnostic, not evidence of causality',
        'patch_surround_definition':'ligand heavy atoms within 5 A of ASN117/VAL116 receptor atoms versus complement; geometric labels, not atom identities conserved between molecules',
        'affinity_definition':'FLOWR same decoded state affinity-head rescore at .9999, not independent experimental affinity',
        'unique_policy':'First occurrence of each canonical isomeric SMILES, not highest-scoring pose; unique-yield and fingerprint diversity reported separately',
        'versions':{k:importlib.metadata.version(k) for k in ['rdkit','posebusters','numpy']},
        'batch_results':batch_summaries,'results':{a:summarize([r for r in allrows if r['arm']==a]) for a in arms},'sources':sources}
    write_json(out/'terminal_report.json',report);write_json(out/'candidate_metrics.json',allrows)
    pd.DataFrame(allrows).to_csv(out/'candidate_metrics.csv',index=False)
    pd.DataFrame(batch_summaries).to_csv(out/'batch_metrics.csv',index=False)
    pd.DataFrame(pbrows).to_csv(out/'posebusters_fast_checks.csv',index=False)
    return report
