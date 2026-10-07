import numpy as np
import pytest
from evomolsteer.continuous.path_graph import graph_batch, CURRENT, PROPOSAL, NATIVE, COPY


def example():
    selected=np.array([[0,0,2],[1,2,2],[0,1,2],[0,1,2]])
    roots=np.zeros_like(selected);roots[0]=np.arange(3)
    x=np.arange(4*3*2*3,dtype=float).reshape(4,3,2,3)
    proposal=x+1
    for i in range(3):
        roots[i+1]=roots[i,selected[i]]
        x[i+1]=proposal[i,selected[i]]
    return dict(selected_indices=selected,root_slot=roots,score_time=np.arange(4)/10,
                state_time=(np.arange(4)+1)/10,resampled=np.array([1,1,0,0]),
                offspring_count=np.stack([np.bincount(v,minlength=3) for v in selected]),
                current_coords=x,proposal_coords=proposal,pic50_on=np.ones((4,3))*8,
                pic50_off=np.ones((4,3))*7,selection_probability=np.ones((4,3))/3)


def test_sparse_graph_keeps_boundary_current_and_distinguishes_copies():
    nodes,edges,m=graph_batch(example(),[0,.2])
    assert len(nodes)==15 and len(edges)==12
    assert (edges.kind==NATIVE).sum()==6 and (edges.kind==COPY).sum()==6
    assert (edges.loc[edges.kind==COPY,"dt"]==0).all()
    assert not nodes.loc[nodes.score_time==.2,"learning_node"].any()
    assert set(nodes.loc[nodes.score_time==.2,"representation"])=={CURRENT}
    assert m["coordinate_copies_verified"] and m["learning_score_times"]==[0.,.1]
    # Parent proposal 0 has two children; the unselected proposal is kept.
    assert (edges.loc[edges.kind==COPY,"parent_id"]==1).sum()==2
    assert 3 in set(nodes.node_id) and 3 not in set(edges.loc[edges.kind==COPY,"parent_id"])


def test_nonzero_dynamic_window_uses_original_step_locators():
    nodes,edges,m=graph_batch(example(),[.1,.2],source_index=4,node_offset=100)
    assert m["learning_score_times"]==[.1] and nodes.source_index.eq(4).all()
    assert nodes.step.min()==1 and nodes.step.max()==2


@pytest.mark.parametrize("field",["root_slot","offspring_count","current_coords"])
def test_corrupt_genealogy_or_copy_is_rejected(field):
    a=example();a[field][1].flat[0]+=1
    with pytest.raises(ValueError):graph_batch(a,[0,.2])


def test_copy_and_native_rate_edges_are_not_interchangeable():
    nodes,edges,_=graph_batch(example(),[0,.2])
    state=nodes.set_index("node_id").state_time
    np.testing.assert_allclose(state.loc[edges.child_id].to_numpy()-state.loc[edges.parent_id].to_numpy(),edges.dt)
    assert edges.loc[edges.kind==NATIVE,"dt"].gt(0).all()
