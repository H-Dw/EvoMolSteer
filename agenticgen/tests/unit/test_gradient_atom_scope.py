import pytest
import torch
from agenticgen.uniguide.gradient_atom_scope import apply, atom_mask, policy


def graph(edges, count=6):
    labels = torch.zeros((1,count,count),dtype=torch.long)
    for a,b in edges: labels[0,a,b] = 1
    return torch.nn.functional.one_hot(labels,5).float()


@pytest.mark.parametrize('hops,expected',[(0,[0]),(1,[0,1]),(2,[0,1,2])])
def test_chain_masks_only_extra_guidance(hops,expected):
    valid=torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool)
    bonds=graph([(0,1),(2,1),(2,3),(0,4),(4,5)])
    gradient=torch.arange(18,dtype=torch.float).reshape(1,6,3)+1
    native=torch.randn_like(gradient); before=native.clone()
    state=torch.random.get_rng_state().clone()
    scoped,mask=apply(gradient,valid,torch.tensor([[0]]),bonds,policy({'guidance_atom_scope':{'mode':'condition_hops','hops':hops}}))
    assert mask.nonzero()[:,1].tolist()==expected
    assert torch.equal(scoped[mask],gradient[mask]) and torch.count_nonzero(scoped[~mask])==0
    assert torch.equal(native,before) and torch.equal((native+scoped)[~mask],native[~mask])
    assert torch.equal(state,torch.random.get_rng_state())


def test_default_is_original_gradient_without_copy():
    gradient=torch.randn(2,5,3)
    result,mask=apply(gradient,torch.ones(2,5),None,None,policy({}))
    assert result is gradient and mask is None


def test_dynamic_mapping_and_candidate_local_graph():
    valid=torch.ones(2,6,dtype=torch.bool)
    bonds=torch.cat([graph([(0,1)]),graph([(4,3),(3,2)])])
    ids=torch.tensor([[0],[4]])
    assert atom_mask(valid,ids,bonds,1).nonzero().tolist()==[[0,0],[0,1],[1,3],[1,4]]
    assert atom_mask(valid,ids,bonds,2)[1].nonzero().flatten().tolist()==[2,3,4]


def test_full_coupled_vjp_precedes_mask_and_loss_is_not_expanded():
    x=torch.tensor([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]],requires_grad=True)
    endpoint=x[:,0]+2*x[:,1]+3*x[:,2]
    loss=endpoint.square().sum()/2
    gradient=torch.autograd.grad(loss,x)[0]
    assert torch.count_nonzero(gradient[:,1:])==6
    scoped,mask=apply(gradient,torch.ones(1,3),torch.tensor([[0]]),None,
                      policy({'guidance_atom_scope':{'mode':'condition_hops','hops':0}}))
    assert torch.equal(scoped[:,0],gradient[:,0]) and torch.count_nonzero(scoped[:,1:])==0


@pytest.mark.parametrize('value',[{'mode':'condition_hops','hops':True},{'mode':'condition_hops','hops':3},
                                 {'mode':'all','hops':0},{'mode':'condition_hops','hops':-1}])
def test_invalid_policies_fail(value):
    with pytest.raises(ValueError):policy({'guidance_atom_scope':value})


@pytest.mark.parametrize('ids',[[[0,0]],[[5]],[[-1]]])
def test_bad_condition_or_padding_fails(ids):
    with pytest.raises(ValueError):atom_mask(torch.tensor([[1,1,1,1,0,0]]),torch.tensor(ids),None,0)
