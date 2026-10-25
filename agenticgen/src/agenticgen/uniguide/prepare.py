"""Exact archive extraction, PLIP-only teachers and native prior reconstruction."""
import io
import random
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from .io import read, write, sha, frozen
from .native_tools import cpu, save_pt, load_pt, instrument
from .protocol import SEEDS
from agenticgen.runtime.rng import rng_state, set_rng


def extract_target(archive, target, destination):
    prefix = f"targets/{target}/"
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.name.startswith(prefix) or not member.isfile():
                continue
            path = destination / member.name[len(prefix):]
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
            data = tf.extractfile(member).read()
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Extracted source changed")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def unique_teacher(path, node_id, heavy_atoms):
    matches = [m for m in Chem.SDMolSupplier(str(path), removeHs=True)
               if m is not None and m.HasProp("node_id") and m.GetProp("node_id") == node_id]
    if len(matches) != 1 or matches[0].GetNumHeavyAtoms() != heavy_atoms:
        raise ValueError("Teacher node_id is absent/nonunique or heavy count differs")
    return matches[0]


def prepare_teachers(config):
    from .profiling import plip_profile
    from evomolsteer.storage.trajectory import TrajectoryPackage
    root = Path(config["root"])
    frozen(root / "config.json", config)
    catalog = []
    for teacher in config["teachers"]:
        target = teacher["target"]
        source = root / "sources" / "steer" / target
        archive = Path(config["archive_root"]) / teacher["archive"]
        extract_target(archive, target, source)
        # Historical baseline is provenance, never a new zero-guidance run.
        unarchive = Path(config["unguided_archive_root"]) / teacher["archive"]
        extract_target(unarchive, target, root / "sources" / "unguided" / target)
        campaign = source / "results/single_w050"
        batch = campaign / "single" / f"batch_{teacher['batch']:03d}"
        sdf = batch / "molecules_all_built.sdf"
        if sha(sdf) != teacher["member_sha256"]:
            raise ValueError("Teacher SDF member hash differs")
        mol = unique_teacher(sdf, teacher["node_id"], teacher["heavy_atoms"])
        rows = [r for r in read(campaign / "final_records.json") if r["node_id"] == teacher["node_id"]]
        if len(rows) != 1 or abs(rows[0]["pic50_on_rescore"]-teacher["pic50"]) > 1e-7:
            raise ValueError("Teacher stored score/record binding differs")
        row = rows[0]
        folder = root / "teachers" / target
        folder.mkdir(parents=True, exist_ok=True)
        with Chem.SDWriter(str(folder / "teacher.sdf")) as writer:
            writer.write(mol)
        protein = source / "inputs/target_protein.pdb"
        profile = plip_profile(mol, protein, folder / "plip")
        core = profile["core_atoms"]
        if not core or any(mol.GetAtomWithIdx(i).GetAtomicNum() <= 1 for i in core):
            raise ValueError("Empty or non-heavy PLIP core")
        with TrajectoryPackage(batch / "terminal.h5") as package:
            native = {k:package.read(k, 0, int(row["slot"])) for k in ["coords", "atomics", "charges", "mask"]}
        cfg = read(campaign / "config.json")
        xyz = mol.GetConformer().GetPositions()
        elements = np.array([cfg["atom_vocabulary"][a.GetSymbol()] for a in mol.GetAtoms()])
        cost = ((xyz[:,None]-native["coords"][None])**2).sum(-1)
        cost += 1e6*(elements[:,None] != native["atomics"].argmax(-1)[None])
        ii,jj = linear_sum_assignment(cost)
        mapping = jj[np.argsort(ii)]
        if len(ii) != len(xyz) or cost[ii,jj].max() > 1e-6:
            raise ValueError("SDF/native teacher coordinate mapping failed")
        template = dict(teacher, source_record=row, source_archive=str(archive), archive_sha256=sha(archive),
            source_member=f"targets/{target}/results/single_w050/single/batch_{teacher['batch']:03d}/molecules_all_built.sdf",
            source_sdf=str(sdf), protein=str(protein), protein_sha256=sha(protein),
            sdf=str(folder / "teacher.sdf"), sdf_to_native=mapping.tolist(),
            core_sdf_atoms=core, core_native_slots=mapping[core].tolist(), core_A=xyz[core].tolist(),
            condition_scope="PLIP direct ligand heavy-atom union; zero covalent expansion",
            plip_profile=profile, native_config=str(campaign / "config.json"))
        frozen(folder / "template.json", template)
        catalog.append(template)
    frozen(root / "teachers.json", catalog)
    write(root / "TEACHERS_PREPARED.json", {"count":len(catalog), "templates_sha256":sha(root/"teachers.json")})
    return catalog


def load_model(config, teacher, datasets=False):
    flowr = Path(config["flowr"])
    sys.path.insert(0, str(flowr))
    import flowr as imported
    from flowr.scriptutil import load_model as native_load
    if not Path(imported.__file__).resolve().is_relative_to(flowr.resolve()):
        raise ValueError("Wrong FLOWR import")
    for name, expected in config["native_hashes"].items():
        if sha(flowr / name) != expected:
            raise ValueError("Native source changed: " + name)
    if sha(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Wrong v2 checkpoint")
    args = SimpleNamespace(**read(teacher["native_config"])["flowr_args"])
    inp = Path(teacher["protein"]).parent
    for side in ["target", "untarget"]:
        setattr(args, "pdb_file_"+side, str(inp / "target_protein.pdb"))
        setattr(args, "ligand_file_"+side, str(inp / "target_ligand.sdf"))
    args.ckpt_path = config["checkpoint"]
    args.save_dir = str(Path(config["root"]) / "runtime" / teacher["target"])
    args.seed, args.num_workers, args.batch_cost, args.sample_n_molecules_per_target = 42, 0, 20, 100
    for key in ["sample_mol_sizes", "rotation_alignment", "permutation_alignment", "anisotropic_prior", "ref_ligand_com_prior"]:
        if getattr(args, key, False):
            raise ValueError("Unapproved native prior option: "+key)
    torch.set_float32_matmul_precision("high")
    loaded = native_load(args)
    model, hp, vocab, vc, vh, va, vpa, vpr = loaded
    model = model.to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.integrator.use_sde_simulation = True
    model.integrator.coord_noise_level = .2
    if model.inpainting_mode or model.graph_inpainting or model.integrator.use_cosine_scheduler:
        raise ValueError("Unsupported inpainting/scheduler")
    if model.integrator.type_strategy != "uniform-sample" or model.integrator.bond_strategy != "uniform-sample":
        raise ValueError("Unsupported categorical strategy")
    if datasets:
        from flowr.gen import utils as util
        from flowr.gen.generate_from_pdb_selective import get_dataset
        transform, interpolant = util.load_util(args,hp,vocab,vc,vh,va)
        st,so = util.load_data_from_pdb_selective(args,remove_hs=hp["remove_hs"],remove_aromaticity=hp["remove_aromaticity"])
        dt=get_dataset(st,transform,vocab,interpolant,args,hp)
        do=get_dataset(so,transform,vocab,interpolant,args,hp)
        return model,util.get_dataloader(args,dt,interpolant),util.get_dataloader(args,do,interpolant)
    instrument(model, Path(config["root"])/"provenance/native")
    return model


def prepare_initials(config):
    from evomolsteer.storage.trajectory import TrajectoryPackage
    from .inputs import make_inputs
    root=Path(config["root"])
    for teacher in read(root/"teachers.json"):
        target=teacher["target"]
        folder=root/"initials"/target
        if (folder/"COMPLETE.json").exists():
            for b in range(5): load_pt(folder/f"batch_{b:03d}.pt.gz")
            continue
        model,dlt,dlo=load_model(config,teacher,datasets=True)
        checks=[]
        for b,(bt,bo) in enumerate(zip(dlt,dlo)):
            if b>=5: raise ValueError("Unexpected native prior batch count")
            prior,pt,po,frame=make_inputs(model,bt,bo,"cuda")
            if len(prior["coords"]) !=20 or not bool((prior["mask"].sum(-1)==teacher["heavy_atoms"]).all()):
                raise ValueError("Wrong original prior shape")
            shared=rng_state()
            verified={}
            for arm,dirname,sub in [("steer","single_w050","single"),("unguided","unguided_w050","unguided")]:
                path=root/"sources"/arm/target/"results"/dirname/sub/f"batch_{b:03d}"/"trajectory.h5"
                with TrajectoryPackage(path) as package:
                    for key in ["coords","atomics","bonds","charges","mask"]:
                        old=package.read("mask" if key=="mask" else "current_"+key,0)
                        new=prior[key].detach().cpu().numpy()
                        if key in ["atomics","bonds","charges"]: new=new.argmax(-1)
                        verified[arm+"_"+key]=bool(np.array_equal(old,new))
            if not all(verified.values()):
                write(folder/f"MISMATCH_{b}.json",verified)
                raise ValueError("Reconstructed prior differs from historical arrays")
            torch.manual_seed(SEEDS[b]); np.random.seed(SEEDS[b]); random.seed(SEEDS[b])
            com=torch.stack([torch.as_tensor(s.com) for s in pt["complex"]]).reshape(-1,3)
            state=dict(prior=cpu(prior), pocket_target=cpu({k:v for k,v in pt.items() if k!="complex"}),
                pocket_off=cpu({k:v for k,v in po.items() if k!="complex"}),target_com=com.cpu(),
                rng=rng_state(),seed=SEEDS[b],batch=b,coord_scale=float(model.coord_scale))
            save_pt(folder/f"batch_{b:03d}.pt.gz",state)
            checks.append(dict(batch=b,checks=verified,frame=frame))
            set_rng(shared)
        if len(checks)!=5: raise ValueError("Five initial batches required")
        write(folder/"COMPLETE.json",checks)
        del model,dlt,dlo
        torch.cuda.empty_cache()
    write(root/"INITIALS_VERIFIED.json",{"targets":[x["target"] for x in config["teachers"]],"native_rows":20,"selected_slots":list(range(10))})
