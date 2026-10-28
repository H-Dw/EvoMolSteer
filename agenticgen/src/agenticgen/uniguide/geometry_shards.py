"""Stable native random slots and immutable ten-attempt promotion units."""
import copy
from pathlib import Path
import h5py
import numpy as np
import torch
from rdkit import Chem
from .random_streams import NativeStreams
from .batching import take
from .chunking import concatenate
from .native_tools import load_pt, save_pt
from .io import read, write
from .protocol import SEEDS


def validate_slots(slots):
    if not isinstance(slots,list) or len(slots)!=2 or any(type(x) is not int for x in slots):
        raise ValueError("A promotion shard requires two integer native slots")
    if slots not in [[i,i+1] for i in range(0,10,2)]:
        raise ValueError("Only frozen disjoint two-slot blocks are supported")
    return slots


class SlotStreams(NativeStreams):
    def __init__(self,states,slots):
        self.slots=validate_slots(slots)
        super().__init__(states,1,2)

    def native_block(self,value,offset):
        block=value[offset:offset+2]
        native=torch.cat([block]*10,0)
        native[self.slots]=block
        return native

    def draw(self,kind,row_count,callback):
        # Native twenty-row draws, including discarded slots, consume identical RNG.
        result=super().draw(kind,row_count,lambda offset:callback(offset)[self.slots])
        self.ledger[-1]["native_logical_slots"]=self.slots
        return result


def initial_shard(config,target,slots):
    from .sampler import population
    slots=validate_slots(slots);full=population(config,target,50)
    ids=[seed*10+slot for seed in range(5) for slot in slots]
    result={key:take(full[key],ids,50) for key in ("prior","pocket_target","pocket_off","target_com")}
    return dict(result,population=10,stream_states=full["stream_states"],logical_slots=slots)


def combine_products(root,shards,folder):
    """Materialize a population view without another forward, builder or random draw."""
    folders=[Path(root)/s["id"] for s in shards];folder=Path(folder)
    records=[dict(r,source_shard=s["id"],source_tensor_row=r["tensor_row"])
        for s,path in zip(shards,folders) for r in read(path/"records.json")]
    order=sorted(range(len(records)),key=lambda i:(SEEDS.index(records[i]["logical_seed"]),records[i]["logical_slot"]))
    records=[records[i] for i in order]
    if len({(r["logical_seed"],r["logical_slot"]) for r in records})!=len(records):
        raise ValueError("Promotion attempted to count a random slot twice")
    for i,row in enumerate(records):row["tensor_row"]=i
    write(folder/"records.json",records)
    terminal=concatenate([load_pt(p/"terminal.pt.gz") for p in folders],[10]*len(folders))
    save_pt(folder/"terminal.pt.gz",take(terminal,order,len(records)))
    for name in ("molecules_all_built.sdf","molecules_raw_decodable.sdf"):
        # Copy source structures; do not rerun molecular construction during promotion.
        identity={r["node_id"]:r for r in records}
        with Chem.SDWriter(str(folder/name)) as writer:
            for source in folders:
                if (source/name).stat().st_size==0:continue
                for molecule in Chem.SDMolSupplier(str(source/name),sanitize=False,removeHs=False):
                    if molecule is not None:
                        molecule.SetProp("tensor_row",str(identity[molecule.GetProp("node_id")]["tensor_row"]))
                        writer.write(molecule)
    with h5py.File(folder/"trace.h5","w") as output:
        handles=[h5py.File(p/"trace.h5","r") for p in folders]
        try:
            for key in handles[0]:
                if any(key not in h or h[key].shape[:2]!=(100,10) for h in handles):
                    raise ValueError("Unexpected shard trace layout")
                shape=(100,len(records),*handles[0][key].shape[2:])
                data=output.create_dataset(key,shape=shape,dtype=handles[0][key].dtype,
                    chunks=(1,*shape[1:]),compression="gzip",compression_opts=1)
                for step in range(100):data[step]=np.concatenate([h[key][step] for h in handles],axis=0)[order]
        finally:
            for handle in handles:handle.close()
    steps=[read(p/"steps.json") for p in folders];combined=[]
    for index in range(100):
        row=copy.deepcopy(steps[0][index])
        for key in ("loss","assignment_switch_fraction","gradient_norm","coefficient","max_atom_increment_A","delta_native_ratio"):
            joined=[v for part in steps for v in part[index][key]]
            row[key]=[joined[i] for i in order]
        row["sampling_rng_after_sha256"]={s["id"]:part[index]["sampling_rng_after_sha256"] for s,part in zip(shards,steps)}
        combined.append(row)
    write(folder/"steps.json",combined)
    write(folder/"rng_ledger.json",{s["id"]:read(p/"rng_ledger.json") for s,p in zip(shards,folders)})
    return records
