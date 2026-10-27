"""Remote-only engineering checks for chemical-only projection and pilot scope."""
import importlib.util
from pathlib import Path

import pytest
import torch

from agenticgen.uniguide.chem_inpainting import project_chemistry, terminal_projection, PilotBudget
from agenticgen.uniguide.io import read,write,value_sha
from agenticgen.uniguide.native_tools import tensor_digest


def example():
    features=dict(atomics=torch.eye(4)[[1,2]],charges=torch.eye(3)[[1,0]],
                  hybridization=torch.eye(3)[[2,1]],bonds=torch.eye(5)[torch.tensor([[0,2],[2,0]])])
    state=dict(coords=torch.arange(30.).reshape(2,5,3),mask=torch.tensor([[1,1,1,1,0],[1,1,1,1,0]]),
        atomics=torch.eye(4)[torch.zeros((2,5),dtype=torch.long)],charges=torch.eye(3)[torch.zeros((2,5),dtype=torch.long)],
        hybridization=torch.eye(3)[torch.zeros((2,5),dtype=torch.long)],bonds=torch.eye(5)[torch.zeros((2,5,5),dtype=torch.long)],
        extra="native")
    return state,features,torch.tensor([[1,3],[2,0]])


def test_projection_changes_only_matched_chemical_channels_with_no_rng_or_coordinate_changes():
    state,features,ids=example()
    before=tensor_digest(state);rng=torch.get_rng_state().clone()
    result,audit=project_chemistry(state,ids,features)
    assert tensor_digest(state)==before and torch.equal(rng,torch.get_rng_state())
    assert result['coords'] is state['coords'] and result['mask'] is state['mask']
    assert audit['coordinates_replaced'] is False and set(audit['after'].values())=={1.}
    for b,slots in enumerate(ids):
        untouched=[i for i in range(5) if i not in slots]
        for key in ['atomics','charges','hybridization']:
            assert torch.equal(result[key][b,slots],features[key])
            assert torch.equal(result[key][b,untouched],state[key][b,untouched])
        assert torch.equal(result['bonds'][b],result['bonds'][b].transpose(0,1))
        for i in range(5):
            for j in range(5):
                if i==j or i in untouched or j in untouched:
                    assert torch.equal(result['bonds'][b,i,j],state['bonds'][b,i,j])


@pytest.mark.parametrize('error',['coords','duplicate','masked','vocabulary'])
def test_projection_rejects_coordinate_injection_or_inconsistent_mapping(error):
    state,features,ids=example()
    if error=='coords':features['coords']=torch.zeros(2,3)
    if error=='duplicate':ids[0]=torch.tensor([1,1])
    if error=='masked':ids[0,0]=4
    if error=='vocabulary':features['charges']=torch.eye(4)[[1,0]]
    with pytest.raises(ValueError):project_chemistry(state,ids,features)


def test_terminal_projection_retains_native_coordinates_and_preserves_raw_terminal(tmp_path):
    state,features,ids=example()
    teacher=dict(core_A=state['coords'][0,[1,3]].tolist())
    old=state['coords'].clone()
    result=terminal_projection(state,teacher,features,tmp_path)
    assert torch.equal(result['coords'],old)
    assert (tmp_path/'terminal_native_before_chemical_projection.pt.gz').exists()
    audit=read(tmp_path/'FINAL_CHEMICAL_PROJECTION.json')
    assert not audit['coordinates_replaced'] and set(audit['after'].values())=={1.}


def test_pilot_cap_charges_failures_without_touching_UG_attempts(tmp_path):
    write(tmp_path/'attempts.json',{'existing':{'count':6450}})
    original=(tmp_path/'attempts.json').read_bytes()
    b=PilotBudget(tmp_path)
    b.reserve('groups/a','formal',50,'A');b.reserve('groups/a','formal',50,'A')
    b.reserve('groups/b','formal',50,'B')
    with pytest.raises(ValueError):b.reserve('groups/c','formal',50,'C')
    with pytest.raises(ValueError):b.reserve('groups/a','formal',50,'changed')
    assert (tmp_path/'attempts.json').read_bytes()==original
    assert sum(r['count'] for r in read(tmp_path/'pilot_attempts.json').values())==100


def test_configuration_selection_uses_mean_not_single_best_and_does_not_count_repeats(tmp_path):
    script=Path(__file__).parents[2]/'scripts/run_uniguide_chem_inpainting.py'
    spec=importlib.util.spec_from_file_location('chemical_pilot',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    write(tmp_path/'teachers.json',[{'target':'a'},{'target':'b'}])
    for teacher in ['a','b']:
        for version,mean,best,schedule in [('UG01',7.,12.,[1]*4),('UG02',8.,10.,[2]*4),('UG03',8.,10.,[2]*4)]:
            key=f'versions/{version}/{teacher}/R01/increase'
            write(tmp_path/key/'aggregate.json',dict(id=key,complete=True,attempts=50,mean_pic50=mean,best=best,C=1.,gap=1.,schedule=schedule))
    _,selected=module.select_controls(tmp_path)
    assert selected['a']['group'].startswith('versions/UG02/')
    assert selected['b']['aggregate']['mean_pic50']==8.
    assert selected['a']['complete_groups']==3 and selected['a']['unique_schedules']==2
