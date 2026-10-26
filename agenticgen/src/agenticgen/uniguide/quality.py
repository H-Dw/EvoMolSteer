"""All-attempt quality accounting and recoverable structure retention."""
from pathlib import Path
from rdkit import Chem
from .io import read,write,sha
from .profiling import profile_task,quality


def assess_light(folder,teacher):
    """Cheap all-attempt statistics; PLIP and PoseBusters run in CPU workers."""
    import numpy as np
    folder=Path(folder)
    binding={'records':sha(folder/'records.json'),'sdf':sha(folder/'molecules_all_built.sdf'),
             'protein':sha(teacher['protein'])}
    dest=folder/'QUALITY_LIGHT_COMPLETE.json'
    if dest.exists():
        saved=read(dest)
        if saved['source_binding']!=binding:raise ValueError('Light quality source changed')
        return saved
    mols=built_molecules(folder/'molecules_all_built.sdf');records=read(folder/'records.json')
    protein=np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])]
        for l in Path(teacher['protein']).read_text().splitlines()
        if l.startswith(('ATOM  ','HETATM')) and l[76:78].strip().upper() not in ('H','D')])
    results=[]
    for row in records:
        mol=mols.get(row['node_id'])
        if mol is None:
            results.append({'node_id':row['node_id'],'built':False,'valid_connected':False,'severe_clash':None})
        else:
            distance=np.linalg.norm(np.array(mol.GetConformer().GetPositions())[:,None]-protein[None],axis=-1)
            results.append({'node_id':row['node_id'],'built':True,'valid_connected':len(Chem.GetMolFrags(mol))==1,
                'severe_clash':bool((distance<1.2).any()),'min_protein_distance_A':float(distance.min())})
    write(folder/'quality_light.json',results)
    receipt={'mode':'all_attempts_lightweight','attempts':len(records),'built':sum(r['built'] for r in results),
        'valid_connected':sum(r['valid_connected'] for r in results),
        'severe_clash':sum(bool(r['severe_clash']) for r in results),
        'clash_evaluable':sum(r['severe_clash'] is not None for r in results),
        'plip_posebusters':'separate asynchronous diagnostics; never infer pass from this receipt',
        'source_binding':binding,'rows_sha256':sha(folder/'quality_light.json')}
    write(dest,receipt)
    return receipt


def built_molecules(path):
    path=Path(path)
    return ({m.GetProp("node_id"):m for m in Chem.SDMolSupplier(str(path),removeHs=True) if m is not None}
            if path.stat().st_size else {})


def assess(folder,teacher):
    folder=Path(folder)
    if (folder/"QUALITY_COMPLETE.json").exists(): return read(folder/"QUALITY_COMPLETE.json")
    mols=built_molecules(folder/"molecules_all_built.sdf")
    rows=read(folder/"records.json");results=[]
    for row in rows:
        mol=mols.get(row["node_id"])
        if mol is None:
            q=quality(None,teacher["protein"]);q["severe_clash"]=None
            results.append(dict(node_id=row["node_id"],quality=q,build_success=False))
            continue
        result=profile_task(dict(node_id=row["node_id"],molblock=Chem.MolToMolBlock(mol),protein=teacher["protein"],out=str(folder/"profiles"/row["node_id"])))
        profile=read(result["profile_path"])
        ref={tuple(x) for x in teacher["plip_profile"]["fingerprints"]}
        evaluable=profile.get('plip_evaluable',True)
        found={tuple(x) for x in profile["fingerprints"]} if evaluable else set()
        result.update(build_success=True,plip_recovery=len(found&ref)/len(ref) if evaluable and ref else None)
        results.append(result)
    write(folder/"quality.json",results)
    receipt={"attempts":len(rows),"built":sum(r["build_success"] for r in results),
        "posebusters_pass":sum(bool(r["quality"].get("posebusters_pass",False)) for r in results),
        "posebusters_evaluable":sum(r['quality'].get('posebusters_evaluable',r['build_success']) for r in results),
        "posebusters_unevaluable_built":sum(r['build_success'] and not r['quality'].get('posebusters_evaluable',True) for r in results),
        "valid_connected":sum(r["quality"].get("valid_connected",False) for r in results),
        "severe_clash":sum(bool(r["quality"].get("severe_clash")) for r in results),
        "clash_evaluable":sum(r["quality"].get("severe_clash") is not None for r in results),
        "plip_evaluable":sum(r['quality'].get('plip_evaluable',r['build_success']) for r in results),
        "plip_unevaluable_built":sum(r['build_success'] and not r['quality'].get('plip_evaluable',True) for r in results),
        "plip_unevaluable_reasons":{reason:sum(r['quality'].get('plip_unevaluable_reason')==reason for r in results)
            for reason in sorted({r['quality']['plip_unevaluable_reason'] for r in results if 'plip_unevaluable_reason' in r['quality']})},
        "quality_sha256":sha(folder/"quality.json"),"records_sha256":sha(folder/"records.json"),
        "quality_implementation_sha256":{name:sha(Path(__file__).parent/name) for name in ['quality.py','profiling.py']}}
    write(folder/"QUALITY_COMPLETE.json",receipt)
    return receipt


def retention(root,teacher,groups,winner,folder):
    """Export 50 per-slot winners; preserve every source until explicitly cleaned."""
    from .protocol import choose
    root=Path(root);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    pool={}
    for group in groups:
        for row in read(root/group["id"]/"records.json"):
            key=(row["logical_seed"],row["logical_slot"])
            pool.setdefault(key,[]).append(dict(row,source_group=group["id"],id=group["id"]+"/"+row["node_id"],
                C=row["condition_rmsd_A"],gap=abs(row["pic50_on_rescore"]-teacher["pic50"]),best=row["pic50_on_rescore"],complete=True))
    top=[choose(rows) for _,rows in sorted(pool.items())]
    if len(top)!=50: raise ValueError("Per-slot retention requires all fifty logical slots")
    write(folder/"per_slot_top1.json",top)
    writer=Chem.SDWriter(str(folder/"per_slot_top1_built.sdf"))
    loaded={}
    for row in top:
        group=row["source_group"]
        if group not in loaded:
            loaded[group]=built_molecules(root/group/"molecules_all_built.sdf")
        mol=loaded[group].get(row["node_id"])
        if mol is not None:
            mol.SetProp("source_group",group);writer.write(mol)
    writer.close()
    receipt={"per_slot_entries":50,"main_representative":winner["id"],
             "representative_terminal_sha256":sha(root/winner["id"]/"terminal.pt.gz"),
             "representative_trace_sha256":sha(root/winner["id"]/"trace.h5"),
             "retention_policy":"all sources preserved; per-slot TOP1 plus full representative population identified"}
    write(folder/"RETENTION_COMPLETE.json",receipt)
    return receipt
