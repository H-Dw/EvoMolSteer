"""Chemical state injection must not become coordinate or raw-output projection."""
import copy
from types import SimpleNamespace
import pytest
import torch
from rdkit import Chem
from agenticgen.uniguide.geometry_condition import prepare
from agenticgen.uniguide.geometry_sampling import GeometryControl,scale_binding
from agenticgen.uniguide.geometry_dynamics import policy as dynamics_policy
from agenticgen.uniguide.geometry_chemistry import policy,condition_features,match_fractions
from agenticgen.uniguide.native_tools import tensor_digest


class Vocabulary:
    def __init__(self,values):self.values=values
    def indices_from_tokens(self,tokens,one_hot):
        assert one_hot
        return torch.nn.functional.one_hot(torch.tensor([self.values.index(t) for t in tokens]),len(self.values)).tolist()


def fixture(tmp_path,enabled=True,controlled=('a','c')):
    features=dict(atoms=[dict(id=i,element=e,formal_charge=0) for i,e in [('a','C'),('b','C'),('c','O')]],
        bonds=[dict(atom_ids=['a','b'],order='single'),dict(atom_ids=['b','c'],order='single')],
        coordinates_angstrom=[[0,0,0],[1.5,0,0],[2.5,1,0]],controlled_atom_ids=list(controlled),
        atom_to_generated_slot={'a':3,'b':0,'c':2})
    condition=prepare(features)
    model=SimpleNamespace(hparams={},add_feats=True,vocab=Vocabulary(['C','N','O']),vocab_charges=Vocabulary([-1,0,1]),
        vocab_hybridization=Vocabulary([Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3]))
    control=object.__new__(GeometryControl)
    control.model=model;control.geometry_condition=condition;control.folder=tmp_path
    control.sampling_policy=dynamics_policy(dict(output_policy='raw999',maximum_single_atom_guidance_increment_angstrom=3.,
        chemical_inpainting={'enabled':enabled}))
    control.configure_chemistry();control.rows=[dict(step=10)];control.trace=SimpleNamespace(flush=lambda:None)
    control.store=lambda *args:None
    state=dict(coords=torch.arange(30.).reshape(2,5,3).requires_grad_(),mask=torch.tensor([[1,1,1,1,0]]*2),
        atomics=torch.eye(3)[torch.ones(2,5,dtype=torch.long)],charges=torch.eye(3)[torch.zeros(2,5,dtype=torch.long)],
        hybridization=torch.eye(3)[torch.zeros(2,5,dtype=torch.long)],
        bonds=torch.eye(5)[torch.ones(2,5,5,dtype=torch.long)],self_condition='untouched')
    return control,state,features


def test_defaults_and_reject_coordinate_options():
    assert policy({})['enabled'] and not policy({})['coordinates_replaced']
    assert not policy({'chemical_inpainting':{'enabled':False}})['enabled']
    for invalid in [None,True,{}, {'enabled':1},{'enabled':True,'coords':True}]:
        with pytest.raises(ValueError):policy({'chemical_inpainting':invalid})


def test_fixed_control_mapping_only_chemistry_and_induced_nonbonds(tmp_path):
    c,state,_=fixture(tmp_path);before=tensor_digest(state);rng=torch.get_rng_state().clone()
    c.last_assignment=torch.tensor([[0,1,2]]*2) # Must never be used for chemical selection.
    out=c.apply_chemical_inpainting(state,10)
    assert tensor_digest(state)==before and torch.equal(rng,torch.get_rng_state())
    assert out['coords'] is state['coords'] and out['mask'] is state['mask'] and out['self_condition']==state['self_condition']
    ids=torch.tensor([[3,2]]*2)
    assert all(v==[1.,1.] for v in match_fractions(out,ids,c.chemical_features).values())
    for key in ('atomics','charges','hybridization'):
        assert torch.equal(out[key][:,[0,1,4]],state[key][:,[0,1,4]])
    for i in range(5):
        for j in range(5):
            if (i,j) not in ((3,2),(2,3)):assert torch.equal(out['bonds'][:,i,j],state['bonds'][:,i,j])
    assert (out['bonds'][:,3,2].argmax(-1)==0).all() # Template nonbond is injected too.
    assert c.rows[-1]['chemical_inpainting_applied']==[True,True]


@pytest.mark.parametrize('step',[0,9,100])
def test_only_injects_after_native_steps_10_to_99(tmp_path,step):
    c,state,_=fixture(tmp_path)
    assert c.apply_chemical_inpainting(state,step) is state
    assert c.rows[-1]['chemical_inpainting_applied']==[False,False]


def test_disable_switch_and_single_control_have_finite_audits(tmp_path):
    c,state,_=fixture(tmp_path/'off',enabled=False)
    assert c.apply_chemical_inpainting(state,10) is state and c.chemical_binding is None
    c,state,_=fixture(tmp_path/'one',controlled=('a',))
    out=c.apply_chemical_inpainting(state,99)
    assert c.rows[-1]['chemical_bonds_after_match_fraction']==[1.,1.]
    assert torch.equal(out['bonds'],state['bonds'])


def test_chemistry_comes_from_optimized_graph_and_ignores_coordinates(tmp_path):
    c,_,_=fixture(tmp_path)
    changed=copy.deepcopy(c.geometry_condition);changed['atoms'][2]['formal_charge']=-1
    altered,_,_=condition_features(c.model,changed)
    assert not torch.equal(altered['charges'],c.chemical_features['charges'])
    shifted=copy.deepcopy(c.geometry_condition);shifted['reference_coordinates_angstrom']=[[9999]*3]*3
    encoded,slots,_=condition_features(c.model,shifted)
    assert slots==[3,2] and tensor_digest(encoded)==tensor_digest(c.chemical_features)


def test_resume_binds_chemistry_and_rejects_changed_features(tmp_path):
    c,_,_=fixture(tmp_path);c.scale_binding=scale_binding([0]*4,1.)
    c.resume=None
    binding=c.checkpoint_metadata()['geometry_binding']
    assert binding['chemical_binding']['generated_slots']==[3,2]
    c.resume=dict(geometry_binding=copy.deepcopy(binding))
    c.resume['geometry_binding']['chemical_binding']['generated_slots']=[0,1]
    with pytest.raises(ValueError,match='identical distance condition'):c.initialize()


def test_batch_projection_is_identical_to_separate_rows(tmp_path):
    from agenticgen.uniguide.chem_inpainting import project_chemistry
    c,state,_=fixture(tmp_path);ids=torch.tensor([[3,2]]*2)
    whole=project_chemistry(state,ids,c.chemical_features)[0]
    for i in range(2):
        part={k:v[i:i+1] if torch.is_tensor(v) else v for k,v in state.items()}
        separate=project_chemistry(part,ids[i:i+1],c.chemical_features)[0]
        for key in ['coords','mask','atomics','charges','bonds','hybridization']:
            assert torch.equal(whole[key][i:i+1],separate[key])
