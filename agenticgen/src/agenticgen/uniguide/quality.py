"""All-attempt quality accounting and recoverable structure retention."""
from pathlib import Path
from rdkit import Chem
from .io import read,write,sha
from .profiling import profile_task,quality


def assess(folder,teacher):
    folder=Path(folder)
    if (folder/"QUALITY_COMPLETE.json").exists(): return read(folder/"QUALITY_COMPLETE.json")
    mols={m.GetProp("node_id"):m for m in Chem.SDMolSupplier(str(folder/"molecules_all_built.sdf"),removeHs=True) if m is not None}
    rows=read(folder/"records.json");results=[]
    for row in rows:
        mol=mols.get(row["node_id"])
        if mol is None:
            results.append(dict(node_id=row["node_id"],quality=quality(None,teacher["protein"]),build_success=False))
            continue
        result=profile_task(dict(node_id=row["node_id"],molblock=Chem.MolToMolBlock(mol),protein=teacher["protein"],out=str(folder/"profiles"/row["node_id"])))
        profile=read(result["profile_path"])
        ref={tuple(x) for x in teacher["plip_profile"]["fingerprints"]}
        found={tuple(x) for x in profile["fingerprints"]}
        result.update(build_success=True,plip_recovery=len(found&ref)/len(ref) if ref else None)
        results.append(result)
    write(folder/"quality.json",results)
    receipt={"attempts":len(rows),"built":sum(r["build_success"] for r in results),
        "posebusters_pass":sum(r["quality"].get("posebusters_pass",False) for r in results),
        "valid_connected":sum(r["quality"].get("valid_connected",False) for r in results),
        "severe_clash":sum(r["quality"].get("severe_clash",False) for r in results),
        "quality_sha256":sha(folder/"quality.json"),"records_sha256":sha(folder/"records.json")}
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
            loaded[group]={m.GetProp("node_id"):m for m in Chem.SDMolSupplier(str(root/group/"molecules_all_built.sdf")) if m is not None}
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
