"""Analytic geometry and state contracts of the absolute-coordinate protocol."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("rdkit")
from agenticgen.uniguide.absolute_coordinate import (
    POLICY, RMSD, MAX_ERROR, prepare, match, loss, metrics, kabsch, AssignmentState,
    CoordinateObjective,
)
from agenticgen.uniguide.io import value_sha


@pytest.fixture
def condition(tmp_path):
    pdb=tmp_path/"protein.pdb"
    points=[(0.,0.,0.),(4.,0.,0.),(0.,5.,0.),(0.,0.,6.)]
    pdb.write_text("".join(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
                            for i,(x,y,z) in enumerate(points,1)))
    return prepare(dict(atoms=[dict(id="a",element="C"),dict(id="b",element="O")],
        bonds=[dict(atom_ids=["a","b"],order="single")],controlled_atom_ids=["a","b"],
        coordinates_angstrom=[[0.,0.,0.],[1.4,0.,0.]],coordinate_frame="generation_context_angstrom"),pdb)


def test_mean_loss_exact_gradient_and_fixed_mapping(condition):
    x=torch.tensor([[[8.,8.,8.],[0.,0.,0.],[1.4,0.,0.]]],dtype=torch.float64,requires_grad=True)
    valid=torch.ones((1,3),dtype=torch.bool)
    assignment=match(x,condition,valid)
    assert assignment.tolist()==[[1,2]]
    assert loss(x,condition,valid,assignment)[0].item()==0
    shifted=x.detach().clone();shifted[0,1,1]=2;shifted.requires_grad_(True)
    value=loss(shifted,condition,valid,assignment)[0]
    assert value.item()==1
    gradient=torch.autograd.grad(value.sum(),shifted)[0]
    assert gradient[0,1,1].item()==1
    assert torch.equal(gradient[0,0],torch.zeros(3,dtype=x.dtype))
    assert metrics(shifted,condition,valid,assignment)[RMSD].item()==pytest.approx(np.sqrt(2))


def test_protein_alignment_preserves_relative_pose(condition):
    protein=np.array([[0.,0.,0.],[4.,0.,0.],[0.,5.,0.],[0.,0.,6.]])
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    translation=np.array([12.,-6.,2.])
    moved=protein@rotation+translation
    r,t,error=kabsch(moved,protein)
    assert error<1e-12 and np.linalg.det(r)>0
    c=copy.deepcopy(condition);c['protein_alignment'].update(rotation=r.tolist(),translation_angstrom=t.tolist())
    c['condition_sha256']=value_sha({k:v for k,v in c.items() if k!='condition_sha256'})
    points=np.array([[0.,0.,0.],[1.4,0.,0.]])@rotation+translation
    x=torch.tensor(points[None],dtype=torch.float64);valid=torch.ones((1,2),dtype=torch.bool)
    assert loss(x,c,valid,[[0,1]])[0].item()<1e-24
    assert loss(x+torch.tensor([0.,0.,2.]),c,valid,[[0,1]])[0].item()>1


@pytest.mark.parametrize('assignment',[[[0,0]],[[0,3]],[[-1,1]]])
def test_invalid_assignments_rejected(condition,assignment):
    with pytest.raises(ValueError):
        loss(torch.zeros(1,3,3),condition,torch.ones(1,3,dtype=torch.bool),assignment)


def test_masked_slot_rejected(condition):
    with pytest.raises(ValueError):
        loss(torch.zeros(1,3,3),condition,torch.tensor([[True,False,True]]),[[0,1]])


def test_assignment_freezes_and_checkpoint_is_verified(condition):
    c=SimpleNamespace(geometry_condition=condition,active_step=9,resume=None,
                      check_rows=lambda name,x,ids:x)
    state=AssignmentState(c)
    x=torch.tensor([[[9.,9.,9.],[0.,0.,0.],[1.4,0.,0.]]]);valid=torch.ones(1,3,dtype=torch.bool)
    state.evaluate(x,valid,[0]);assert not state.rows
    c.active_step=10;state.evaluate(x,valid,[0]);assert state.rows[0]['slots']==[1,2]
    c.active_step=11
    shuffled=x[:,[1,0,2]]
    assert state.evaluate(shuffled,valid,[0])[3].tolist()==[[1,2]]
    assert match(shuffled,condition,valid).tolist()==[[0,2]]
    c.resume=dict(absolute_assignment_state=state.rows,absolute_assignment_sha256=value_sha(state.rows))
    recovered=AssignmentState(c);recovered.restore();assert recovered.rows==state.rows
    c.resume['absolute_assignment_state']=copy.deepcopy(state.rows)
    c.resume['absolute_assignment_state'][0]['slots']=[0,2]
    with pytest.raises(ValueError):AssignmentState(c).restore()


def test_reflection_not_accepted_as_rotation():
    a=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
    r,t,error=kabsch(a*np.array([-1,1,1]),a)
    assert np.linalg.det(r)==pytest.approx(1.) and error>.1


def test_batch_reduction_is_candidate_local(condition):
    valid=torch.ones(1,2,dtype=torch.bool)
    x=torch.tensor([[[.2,.3,0.],[1.5,0.,.1]]],requires_grad=True)
    independent=torch.autograd.grad(loss(x,condition,valid,[[0,1]])[0].sum(),x)[0]
    batch=x.detach().repeat(3,1,1).requires_grad_(True)
    gradient=torch.autograd.grad(loss(batch,condition,valid.repeat(3,1),[[0,1]]*3)[0].sum(),batch)[0]
    assert torch.equal(gradient,independent.repeat(3,1,1))


def test_new_release_uses_single_objective_and_retains_learning():
    from agenticgen.uniguide.tool_runtime import assets
    files=assets({'agent_tools':{'skill_release':'2.4.0'}})
    text='\n'.join(p.read_text(encoding='utf8') for p in files.values())
    for term in ('condition_intra_weight','condition_outer_weight','interact_weigh','loss_control_id'):
        assert term not in text
    for term in ('compatible historical','counterexample','integer','JSON','Hungarian','Kabsch'):
        assert term in text


def test_absolute_primary_ranking_not_legacy_distance():
    from agenticgen.uniguide.geometry_protocol import choose
    rows=[dict(id='a',complete=True,**{RMSD:1.,MAX_ERROR:2.}),
          dict(id='b',complete=True,**{RMSD:.5,MAX_ERROR:3.})]
    assert choose(rows)['id']=='b'


def test_prepared_loss_keeps_exact_full_vjp_and_random_stream(condition):
    objective = CoordinateObjective(condition)
    x = torch.tensor([[[.2, .3, 0.], [1.5, 0., .1]]], requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    matrix = x.new_tensor([[1.2, .3, 0.], [0., .8, .1], [.2, 0., 1.1]])
    endpoint = x @ matrix + .4
    binding = objective.bind(endpoint, valid, [[0, 1]])
    before = torch.get_rng_state().clone()
    actual = objective.evaluate(endpoint, valid, binding)[0]
    frame = condition['protein_alignment']
    original = endpoint @ endpoint.new_tensor(frame['rotation']) + endpoint.new_tensor(frame['translation_angstrom'])
    expected = (original - endpoint.new_tensor([[0., 0., 0.], [1.4, 0., 0.]])[None]).square().sum((1, 2)) / 4
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
                       torch.autograd.grad(expected.sum(), x)[0])
    assert torch.equal(before, torch.get_rng_state())


def test_static_condition_and_frozen_assignment_prepared_once(condition, monkeypatch):
    import agenticgen.uniguide.absolute_coordinate as coordinates
    calls = {'condition': 0, 'assignment': 0}
    original_validate, original_bind = coordinates.validate, CoordinateObjective.bind

    def validated(value):
        calls['condition'] += 1
        return original_validate(value)

    def bound(self, *args):
        calls['assignment'] += 1
        return original_bind(self, *args)

    monkeypatch.setattr(coordinates, 'validate', validated)
    monkeypatch.setattr(CoordinateObjective, 'bind', bound)
    control = SimpleNamespace(geometry_condition=condition, active_step=10, resume=None,
                              check_rows=lambda name, value, ids: value)
    state = AssignmentState(control)
    x = torch.tensor([[[0., 0., 0.], [1.4, 0., 0.]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    first = state.evaluate(x, valid, [0])
    for step in range(11, 100):
        control.active_step = step
        current = state.evaluate(x, valid, [0])
        assert current[3] is first[3]
    assert calls == {'condition': 1, 'assignment': 1}
    assert len(state.objective._tensors) == 1
    # Frozen input is isolated from caller mutations; fresh entry still rejects them.
    condition['reference_coordinates_angstrom'][0][0] = 999.
    assert state.evaluate(x, valid, [0])[0].item() == 0.
    with pytest.raises(ValueError, match='binding changed'):
        CoordinateObjective(condition)


def test_dynamic_mask_and_finiteness_are_not_cached(condition):
    objective = CoordinateObjective(condition)
    endpoint = torch.zeros(1, 2, 3)
    valid = torch.ones(1, 2, dtype=torch.bool)
    binding = objective.bind(endpoint, valid, [[0, 1]])
    objective.evaluate(endpoint, valid, binding)
    valid[0, 1] = False
    with pytest.raises(ValueError, match='valid selection'):
        objective.evaluate(endpoint, valid, binding)
    valid[0, 1] = True
    endpoint[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        objective.evaluate(endpoint, valid, binding)


def test_resume_rebinds_and_rejects_missing_frozen_assignment(condition):
    control = SimpleNamespace(geometry_condition=condition, active_step=10, resume=None,
                              check_rows=lambda name, value, ids: value)
    state = AssignmentState(control)
    x = torch.tensor([[[0., 0., 0.], [1.4, 0., 0.]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    expected = state.evaluate(x, valid, [0])
    control.resume = dict(absolute_assignment_state=copy.deepcopy(state.rows),
                          absolute_assignment_sha256=value_sha(state.rows))
    control.active_step = 11
    recovered = AssignmentState(control)
    recovered.restore()
    actual = recovered.evaluate(x, valid, [0])
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[3], expected[3])
    assert actual[3] is not expected[3]
    recovered.rows.clear()
    with pytest.raises(ValueError, match='cannot rematch'):
        recovered.evaluate(x, valid, [0])


def test_absolute_terminal_materialization_and_receipt_reuse(condition, tmp_path, monkeypatch):
    import h5py
    from rdkit import Chem
    from agenticgen.uniguide import geometry_sampling as sampling
    from agenticgen.uniguide.io import read, write
    from agenticgen.uniguide.native_tools import save_pt

    folder = tmp_path / 'sample_shards' / 'postprocessing'
    folder.mkdir(parents=True)
    # The terminal-nearest slot is 0, while the actual frozen assignment uses slot 1.
    xyz = torch.tensor([[[0., 0., 0.], [0., 2., 0.], [1.4, 0., 0.]]]).repeat(10, 1, 1)
    output = dict(coords=xyz, mask=torch.ones(10, 3, dtype=torch.bool))
    with h5py.File(folder / 'trace.h5', 'w') as trace:
        trace['assignment'] = np.tile([1, 2], (100, 10, 1))
        trace['guidance_delta_A'] = np.zeros((100, 10, 3, 3), dtype=np.float32)
    save_pt(folder / 'terminal.pt.gz', output)
    write(folder / 'rng_ledger.json', [])
    write(folder / 'steps.json', [dict(step=i, loss=[1.] * 10, gradient_norm=[1.] * 10,
          delta_native_ratio=[0.] * 10, max_atom_increment_A=[0.] * 10,
          assignment_switch_fraction=[0.] * 10, coefficient=[1.] * 10) for i in range(100)])
    binding = dict(teacher_id='teacher', condition_sha256=condition['condition_sha256'])
    monkeypatch.setattr(sampling, 'compatibility', lambda *args: binding)

    class Builder:
        calls = 0

        def _generate_mols(self, output, sanitise=True):
            self.calls += 1
            result = []
            for i, positions in enumerate(output['coords']):
                if i == 3:
                    result.append(None)
                    continue
                mol = Chem.MolFromSmiles('CCO')
                conformer = Chem.Conformer(3)
                for j, point in enumerate(positions.tolist()):
                    conformer.SetAtomPosition(j, point)
                mol.AddConformer(conformer)
                result.append(mol)
            return result

    model = Builder()
    args = (model, dict(root=str(tmp_path)), dict(target='teacher'), condition,
            [1.] * 4, folder, 'calibration', [0, 1])
    result = sampling.execute_shard(*args, _materialized=output)
    assert result['attempts'] == 10 and result['built'] == 9
    assert result[RMSD] == pytest.approx(np.sqrt(2))
    rows = read(folder / 'records.json')
    assert all(row['condition_generated_slots'] == [1, 2] for row in rows)
    assert rows[3]['build_success'] is False and rows[3][RMSD] == pytest.approx(np.sqrt(2))
    assert read(folder / 'COORDINATE_ASSIGNMENT.json')['no_terminal_rematching'] is True
    assert sampling.validate_receipt(folder) == result
    again = sampling.execute_shard(*args, _materialized=output)
    assert again == result and model.calls == 2
    assert sum(row['count'] for row in read(tmp_path / 'attempts.json').values()) == 10
    correspondence = read(folder / 'COORDINATE_ASSIGNMENT.json')
    correspondence['assignment'][0] = [0, 2]
    write(folder / 'COORDINATE_ASSIGNMENT.json', correspondence)
    with pytest.raises(ValueError, match='source changed'):
        sampling.validate_receipt(folder)
