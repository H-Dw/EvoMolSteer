"""Chemical-only state inpainting sharing the coordinate objective's correspondence.

The template graph supplies chemistry; its coordinates are never read here.
Raw output and self-conditioning remain the model's uncovered predictions.
"""
import torch
from rdkit import Chem, rdBase
from .chem_inpainting import project_chemistry, NODE_FEATURES
from .io import frozen, write, value_sha

POLICY="fixed-template-controlled-chemical-state-v1"
CHANNELS=(*NODE_FEATURES,"bonds")
STEP_FIELDS=("chemical_inpainting_applied",)+tuple(
    f"chemical_{key}_{stage}_match_fraction" for key in CHANNELS for stage in ("before","after"))


def policy(config):
    settings=config.get("chemical_inpainting",{"enabled":True})
    if not isinstance(settings,dict) or set(settings)!={"enabled"} or type(settings["enabled"]) is not bool:
        raise ValueError("chemical_inpainting accepts only the boolean enabled setting")
    from .absolute_coordinate import POLICY as absolute_policy
    from .absolute_coordinate import DYNAMIC_ASSIGNMENT
    correspondence=("current_step_endpoint_hungarian_shared_with_coordinate_loss" if config.get("atom_assignment_policy")==DYNAMIC_ASSIGNMENT else
                    "initial_endpoint_hungarian_then_frozen_per_candidate" if config.get("protocol")==absolute_policy
                    else "frozen_condition_atom_to_generated_slot")
    return dict(policy=POLICY,enabled=settings["enabled"],scope="controlled_atoms_and_induced_bonds",
        step_bounds=[10,99],stage="after_native_transition_and_coordinate_guidance",
        correspondence=correspondence,coordinates_replaced=False,
        self_conditioning_replaced=False,terminal_projection=False,
        native_categorical_rng_preserved=True)


def condition_features(model,condition):
    """Encode the optimized input graph, without loading a teacher SDF or target."""
    from flowr.util.rdkit import BOND_IDX_MAP
    atoms=condition["atoms"];index={a["id"]:i for i,a in enumerate(atoms)}
    molecule=Chem.RWMol()
    for atom in atoms:
        a=Chem.Atom(atom["element"]);a.SetFormalCharge(atom.get("formal_charge",0));molecule.AddAtom(a)
    types={"single":Chem.BondType.SINGLE,"double":Chem.BondType.DOUBLE,
        "triple":Chem.BondType.TRIPLE,"aromatic":Chem.BondType.AROMATIC}
    for bond in condition["bonds"]:
        left,right=bond["atom_ids"];molecule.AddBond(index[left],index[right],types[bond["order"]])
    molecule=molecule.GetMol();Chem.SanitizeMol(molecule)
    if model.hparams.get("remove_aromaticity",False):Chem.Kekulize(molecule,clearAromaticFlags=True)
    ids=list(condition["controlled_atom_ids"])
    selected=[molecule.GetAtomWithIdx(index[i]) for i in ids]
    features=dict(atomics=torch.tensor(model.vocab.indices_from_tokens([a.GetSymbol() for a in selected],one_hot=True)),
        charges=torch.tensor(model.vocab_charges.indices_from_tokens([a.GetFormalCharge() for a in selected],one_hot=True)))
    if model.add_feats:
        features["hybridization"]=torch.tensor(model.vocab_hybridization.indices_from_tokens(
            [a.GetHybridization() for a in selected],one_hot=True))
    bonds=torch.zeros((len(ids),len(ids)),dtype=torch.long)
    for i,left in enumerate(ids):
        for j,right in enumerate(ids):
            bond=molecule.GetBondBetweenAtoms(index[left],index[right]) if i!=j else None
            if bond is not None:bonds[i,j]=BOND_IDX_MAP[bond.GetBondType()]
    features["bonds"]=torch.nn.functional.one_hot(bonds,5)
    from .absolute_coordinate import is_absolute
    slots=[] if is_absolute(condition) else [condition["atom_to_generated_slot"][i] for i in ids]
    binding=dict(policy=POLICY,condition_sha256=condition["condition_sha256"],controlled_atom_ids=ids,
        generated_slots=slots,features={k:v.tolist() for k,v in features.items()},
        remove_aromaticity=bool(model.hparams.get("remove_aromaticity",False)),rdkit_version=rdBase.rdkitVersion,
        coordinates_replaced=False,self_conditioning_replaced=False,terminal_projection=False)
    from .absolute_coordinate import dynamic_assignment
    if dynamic_assignment(condition):binding["assignment_policy"]=condition["assignment_policy"]
    return features,slots,dict(binding,chemical_condition_sha256=value_sha(binding))


def match_fractions(state,ids,features):
    """One diagnostic per candidate, independent of physical batch composition."""
    batch=torch.arange(len(ids),device=ids.device)[:,None]
    result={}
    for key in NODE_FEATURES:
        if key in features:
            expected=features[key].to(state[key]).argmax(-1)
            result[key]=(state[key][batch,ids].argmax(-1)==expected).float().mean(-1).tolist()
    k=ids.shape[1];left,right=torch.where(~torch.eye(k,dtype=torch.bool,device=ids.device))
    if len(left):
        expected=features["bonds"].to(state["bonds"])[left,right].argmax(-1)
        result["bonds"]=(state["bonds"][batch,ids[:,left],ids[:,right]].argmax(-1)==expected).float().mean(-1).tolist()
    else:result["bonds"]=[1.]*len(ids)
    return result


class ChemicalStateInpainting:
    def configure_chemistry(self):
        self.chemical_binding=None
        if (self.sampling_policy["chemical_inpainting"]["enabled"]
                or getattr(self,"absolute_config",{}).get("prefix_inpainting",{}).get("enabled",False)):
            self.chemical_features,self.chemical_slots,self.chemical_binding=condition_features(self.model,self.geometry_condition)
            frozen(self.folder/"CHEMICAL_CONDITION.json",self.chemical_binding)

    def apply_chemical_inpainting(self,state,step):
        n=len(state["mask"]);enabled=self.sampling_policy["chemical_inpainting"]["enabled"]
        active=enabled and 10<=step<=99
        self.rows[-1]["chemical_inpainting_applied"]=[active]*n
        if not active:return state
        if getattr(self,"absolute_state",None) is not None:
            ids=torch.tensor([self.absolute_state.rows[i]["slots"] for i in range(n)],device=state["mask"].device)
        else:
            ids=torch.tensor(self.chemical_slots,device=state["mask"].device)[None].expand(n,-1)
        before=match_fractions(state,ids,self.chemical_features)
        result,_=project_chemistry(state,ids,self.chemical_features)
        after=match_fractions(result,ids,self.chemical_features)
        for stage,values in (("before",before),("after",after)):
            for key,value in values.items():self.rows[-1][f"chemical_{key}_{stage}_match_fraction"]=value
        for key in self.chemical_features:
            self.store("inpainted_chemical_"+key,step,result[key].argmax(-1))
        # Control.after writes periodic records before this post-transition hook.
        if step%10==9:
            self.trace.flush();write(self.folder/"steps.json",self.rows)
        return result
