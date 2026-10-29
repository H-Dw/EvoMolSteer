"""No clipping/gradient rescue; robust statistics are complete-population views."""
import json
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('rdkit')
from test_absolute_coordinate_protocol import condition
from agenticgen.uniguide.absolute_coordinate import POLICY, RMSD, loss
from agenticgen.uniguide.geometry_dynamics import policy, uncapped_increment, LIMIT
from agenticgen.uniguide.geometry_sampling import GeometryControl, coordinate_population_statistics
from agenticgen.uniguide.coordinate_statistics import RANKING, SUMMARY, aggregate_fields
from agenticgen.uniguide.geometry_protocol import choose, primary_score


def test_absolute_has_no_radius_and_preserves_large_finite_vector_and_rng():
    delta=torch.tensor([[[.1,0.,0.]], [[1e20,-1e20,0.]]])
    valid=torch.ones(2,1,dtype=torch.bool)
    before=torch.get_rng_state().clone()
    result,audit=uncapped_increment(delta,valid,1.)
    assert result is delta and result[1].abs().max()>3
    assert audit['unconstrained_max_atom_increment_A'][1]>1e20
    assert 'guidance_increment_limited' not in audit
    assert torch.equal(before,torch.get_rng_state())
    config=policy(dict(protocol=POLICY,output_policy='raw999'))
    assert config['limiter']=='none' and LIMIT not in config
    control=object.__new__(GeometryControl)
    control.sampling_policy=config;control.model=SimpleNamespace(coord_scale=1.)
    assert control.constrain_guidance(delta,valid) is delta


@pytest.mark.parametrize('radius',[None,3.,100.])
def test_legacy_radius_setting_cannot_reenable_protection(radius):
    with pytest.raises(ValueError,match='uncapped'):
        policy(dict(protocol=POLICY,output_policy='raw999',**{LIMIT:radius}))


def test_nonfinite_increment_still_fails_instead_of_being_rescued():
    with pytest.raises(FloatingPointError):
        uncapped_increment(torch.full((1,1,3),float('nan')),torch.ones(1,1,dtype=torch.bool),1.)


def test_high_rmsd_candidate_keeps_full_gradient_and_healthy_gradient_unchanged(condition):
    reference=torch.tensor([[[0.,0.,0.],[1.4,0.,0.]]])
    x=torch.cat([reference+.0001,reference+10000.]).requires_grad_(True)
    valid=torch.ones(2,2,dtype=torch.bool)
    values=loss(x,condition,valid,[[0,1],[0,1]])[0]
    gradients=torch.autograd.grad(values.sum(),x)[0]
    independent=x[:1].detach().clone().requires_grad_(True)
    single=torch.autograd.grad(loss(independent,condition,valid[:1],[[0,1]])[0].sum(),independent)[0]
    assert torch.equal(gradients[:1],single)
    assert gradients[1].norm()>1000 and values[1]>1e7
    assert gradients[0].norm()>0


def records(values):
    return [dict(node_id=str(i),logical_seed=i//10,logical_slot=i%10,**{RMSD:v}) for i,v in enumerate(values)]


def test_group_loss_and_ranking_keep_raw_rows_and_all_low_samples(condition,tmp_path):
    rows=records([.0001]+[.5]*48+[100.])
    before=json.dumps(rows)
    aggregate={RMSD:sum(r[RMSD] for r in rows)/50, 'attempts':50}
    coordinate_population_statistics(tmp_path,rows,condition,{},aggregate)
    assert json.dumps(rows)==before and aggregate['attempts']==50
    assert aggregate[RMSD]>2 and aggregate[RANKING]<.5
    assert aggregate[SUMMARY]['excluded_high_indices']==[49]
    assert (tmp_path/'COORDINATE_STATISTICS.json').exists()
    a=dict(aggregate,id='a',complete=True)
    b=dict(aggregate_fields(records([.55]*50),{}),id='b',complete=True,**{RMSD:.55})
    bad=dict(aggregate_fields(records([.00001]*40+[100.]*10),{}),id='bad',complete=True,**{RMSD:20.})
    assert choose([a,b,bad])['id']=='a'
    assert primary_score(a)==aggregate[RANKING]
    with pytest.raises(ValueError,match='eligible'):choose([bad])


def test_paired_tool_uses_fixed_upper_masks_and_no_symmetric_lower_trimming():
    from agenticgen.uniguide.analysis_tools import paired_summary
    from agenticgen.uniguide.coordinate_statistics import DEFAULTS
    left=records([.5]*50);right=records([.0001]+[.4]*48+[100.])
    out=paired_summary(left,right,[RMSD],upper_tail_policy=DEFAULTS)
    assert out['population_statistics']['candidate']['excluded_high_indices']==[49]
    assert out['metrics'][RMSD]['paired_retained_count']==49
    assert out['metrics'][RMSD]['upper_tail_filtered_paired_mean_change']<-.1
    assert 'trimmed_mean_change' not in out['metrics'][RMSD]
    assert out['population_statistics']['candidate']['minimum']==.0001


def test_new_skills_and_config_bind_current_statistics_and_keep_history():
    from pathlib import Path
    from agenticgen.uniguide.tool_runtime import assets
    from agenticgen.uniguide.absolute_runtime import validate_config
    path=Path(__file__).resolve().parents[2]/'configs/uniguide_absolute_coordinate.json'
    config=validate_config(json.loads(path.read_text(encoding='utf-8')))
    assert config['protocol']==POLICY and LIMIT not in config
    assert config['agent_tools']['skill_release']=='2.5.0'
    text='\n'.join(p.read_text(encoding='utf-8') for p in assets(config).values())
    assert 'All low RMSD observations remain included' in text
    assert 'Every finite candidate retains its full coordinate gradient' in text
    assert 'counterexample' in text and 'integer' in text
    assert 'The bound objective has three template-distance losses' not in text
