import copy
import math
from pathlib import Path

import pytest

from agenticgen.uniguide import calibration_search as search
from agenticgen.uniguide.io import Budget, frozen, read, sha, value_sha, write


def row(scale, error=1., count=10, complete=True, displacement=None, target="teacher"):
    d = scale if displacement is None else displacement
    return dict(id=f"measure/{target}/{scale}/{count}", teacher_id=target,
        condition_sha256=target+"condition", compatibility_sha256="sampling", complete=complete,
        attempts=count, **{search.EFFECTIVE:[scale]*4, search.RMSD:error, search.RANKING:error},
        coordinate_loss_statistics=dict(ranking_eligible=True),
        window_diagnostics=[dict(increment_vector_rms_A=d/10,max_atom_increment_A=d) for _ in range(4)])


def curve():
    return [row(16,.71),row(64,.61),row(256,14.,displacement=1e6),
            row(1024,complete=False),row(4096,complete=False)]


def test_failure_boundary_requests_128_instead_of_aborting():
    rows=curve()
    usable,states,boundary=search.assess(rows)
    assert [search.strength(r) for r in usable]==[16,64]
    assert boundary["effective_scale"]==256
    assert search.next_probe(rows)==128
    assert states[-1]["status"]=="nonfinite_measurement"


def test_failed_128_requests_geometric_lower_midpoint_and_does_not_retry():
    rows=curve()+[row(128,complete=False)]
    proposed=search.next_probe(rows)
    assert proposed==pytest.approx(math.sqrt(64*128))
    assert proposed not in [search.strength(r) for r in rows]


def test_passing_128_explores_towards_existing_upper_boundary():
    rows=curve()+[row(128,.55)]
    assert search.next_probe(rows)==pytest.approx(math.sqrt(128*256))


def test_actual_grid_boundary_is_bounded_without_automatic_expansion():
    assert search.next_probe([row(1,3),row(2,2),row(4,1)])==math.sqrt(2*4)
    assert search.next_probe([row(1,3),row(math.sqrt(2),2),row(2,1)]) is None


def test_engineering_or_nonfinite_displacement_is_not_statistically_removed():
    with pytest.raises(ValueError,match="displacement"):
        search.assess([row(1,displacement=float('inf'))])
    values=curve();values[1][search.STATISTICS]["ranking_eligible"]=False
    assert [search.strength(r) for r in search.assess(values)[0]]==[16]


def test_populations_never_mixed_and_duplicate_configs_rejected():
    with pytest.raises(ValueError,match="populations"):
        search.assess([row(1),row(2,count=20)])
    with pytest.raises(ValueError,match="Duplicate"):
        search.assess([row(1),row(1)])


def test_integer_support_panel_has_no_unmeasured_fractional_levels():
    assert search.support_panel(128,16,128)==[32,64,128]
    assert search.support_panel(16,16,128)==[16,32,64]
    assert search.support_panel(64,64,128) is None
    rows=[row(s,abs(math.log2(s/64)),20) for s in [16,32,64,90.50966799187809,128]]
    assert [search.strength(r) for r in search.select_panel(rows)]==[16,32,64]


def test_panels_respect_observed_stability_component():
    panels=search.panel_candidates(curve()+[row(128,.55),row(math.sqrt(128*256),.54)],1)
    assert 1<=len(panels)<=2
    assert all(max(p)<=math.sqrt(128*256)+1e-9 for p in panels)
    assert all(math.isclose(p[1]/p[0],2) and math.isclose(p[2]/p[1],2) for p in panels)


def test_teacher_budget_is_enforced_and_duplicates_do_not_recharge(tmp_path):
    frozen(tmp_path/'ATTEMPT_LIMITS.json',dict(stage_caps=dict(calibration=700,formal=1500),
        per_teacher_stage_caps=dict(calibration=350)))
    write(tmp_path/'BUDGET_SCOPE.json',dict(stage='calibration',teacher_id='left'))
    budget=Budget(tmp_path);budget.reserve('one','calibration',350,'one')
    budget.reserve('one','calibration',350,'one')
    with pytest.raises(ValueError,match="Teacher"):
        budget.reserve('extra','calibration',10,'extra')
    write(tmp_path/'BUDGET_SCOPE.json',dict(stage='calibration',teacher_id='right'))
    budget.reserve('other','calibration',350,'other')
    assert sum(r['count'] for r in read(tmp_path/'attempts.json').values())==700
    with pytest.raises(ValueError):budget.reserve('extra','calibration',10,'extra')


def test_default_legacy_budget_is_unchanged(tmp_path):
    Budget(tmp_path).reserve('legacy','calibration',400,'binding')
    assert 'teacher_id' not in read(tmp_path/'attempts.json')['legacy']


def test_boundary_winner_can_be_frozen_with_three_confirmed_levels():
    pytest.importorskip('torch')
    from agenticgen.uniguide.geometry_protocol import validate_domain
    rows=[dict(row(s,1/s,50),phase='full') for s in [16,32,64]]
    domain=search.derive_domain(rows,'teacher','teachercondition','sampling')
    assert domain['calibrated_guidance_multiplier']==64
    assert domain['domain_status']=='bounded_best'
    assert domain['allowed_normalized_levels']==[-2,-1,0]
    validate_domain(domain)
    broken=copy.deepcopy(domain);broken['confirmed_effective_levels'][-1]=128
    broken['domain_sha256']=value_sha({k:v for k,v in broken.items() if k!='domain_sha256'})
    with pytest.raises(ValueError,match='Unmeasured'):validate_domain(broken)


def install_fake_sampling(monkeypatch,root,failed_teacher=None,engineering=False):
    pytest.importorskip('torch')
    from agenticgen.uniguide import adaptive_calibration as adaptive
    from agenticgen.uniguide import geometry_sampling as sampling
    def compatibility(config,teacher,condition):
        return dict(teacher_id=teacher['target'],condition_sha256=condition['condition_sha256'])
    monkeypatch.setattr(sampling,'compatibility',compatibility)
    measured={};calls=[]
    def executor(model,config,teacher,condition,schedule,folder,stage,count):
        target=teacher['target'];scale=schedule[0];calls.append((target,scale,count))
        if engineering and target==failed_teacher:raise RuntimeError('disk unavailable')
        prior=measured.get((target,scale),0)
        if count>prior:Budget(root).reserve(str(folder),stage,count-prior,value_sha([target,scale,count]))
        measured[target,scale]=count
        if target==failed_teacher or scale>=1024:raise FloatingPointError('numerical divergence')
        result=row(scale,math.log2(scale/64)**2+.1,count,target=target,
                   displacement=scale*(1e6 if scale>=256 else 1))
        result.update(id=folder.relative_to(root).as_posix(),compatibility_sha256=value_sha(compatibility(config,teacher,condition)))
        write(folder/'aggregate.json',result)
        write(folder/'COMPLETE.json',dict(files={'aggregate.json':sha(folder/'aggregate.json')}))
        return result
    return adaptive,executor,calls


def test_adaptive_run_is_teacher_local_bounded_and_frozen(tmp_path,monkeypatch):
    adaptive,executor,calls=install_fake_sampling(monkeypatch,tmp_path,failed_teacher='left')
    subjects=[dict(teacher=dict(target=t),condition=dict(condition_sha256=t+'condition')) for t in ['left','right']]
    config=dict(root=str(tmp_path),calibration=dict(search_policy=search.POLICY))
    domains=adaptive.run(config,subjects,lambda *a:None,executor)
    assert set(domains)=={'right'}
    assert read(tmp_path/'calibration/TEACHER_STATUS.json')['left']['status']=='needs_evidence'
    assert (tmp_path/'calibration/PARTIAL.json').exists()
    ledger=read(tmp_path/'attempts.json')
    assert all(sum(r['count'] for r in ledger.values() if r['teacher_id']==t)<=350 for t in ['left','right'])
    before=len(calls)
    assert adaptive.run(config,subjects,lambda *a:pytest.fail('reload'),executor)==domains
    assert len(calls)==before


def test_engineering_failure_does_not_get_treated_as_bad_guidance(tmp_path,monkeypatch):
    adaptive,executor,calls=install_fake_sampling(monkeypatch,tmp_path,failed_teacher='left',engineering=True)
    subject=dict(teacher=dict(target='left'),condition=dict(condition_sha256='leftcondition'))
    with pytest.raises(RuntimeError,match='disk unavailable'):
        adaptive.run(dict(root=str(tmp_path),calibration=dict(search_policy=search.POLICY)),[subject],lambda *a:None,executor)
    assert not list(tmp_path.rglob('INSUFFICIENT_EVIDENCE.json'))


def test_import_preserves_measurements_failures_and_counts_source_budget_once(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from agenticgen.uniguide import geometry_sampling as sampling
    from agenticgen.uniguide.calibration_import import import_measurements
    source,destination=tmp_path/'source',tmp_path/'new'
    binding=dict(teacher_id='teacher',condition_sha256='teachercondition',science='unchanged')
    monkeypatch.setattr(sampling,'compatibility',lambda *a:binding)
    rows=[];ledger={}
    for i in range(13):
        run=f'sample_shards/{i}';group=source/run
        identity=dict(compatibility=binding,population=10)
        record=dict(source_group=run,identity=identity)
        write(source/f'slot_registry/{i}.json',record)
        write(source/f'measurement_registry/{i}.json',record)
        result=row(2.**i,complete=i<11)
        result.update(id=run,compatibility_sha256=value_sha(binding))
        if result['complete']:
            write(group/'aggregate.json',result)
            write(group/'COMPLETE.json',dict(files={'aggregate.json':sha(group/'aggregate.json')}))
        else:write(group/'FAILED.json',dict(error='nonfinite',attempts=10))
        rows.append(result);ledger[run]=dict(stage='calibration',count=10,binding=str(i))
    write(source/'attempts.json',ledger);write(source/'acceptance/COMPLETE.json',dict(engineering_only=True))
    write(source/'calibration/frozen/key/coarse_results.json',rows)
    subjects=[dict(teacher=dict(target='teacher'),condition=dict(condition_sha256='teachercondition'))]
    config=dict(root=str(destination),calibration=dict(measurement_source_root=str(source)))
    before=sha(source/'attempts.json');imported=import_measurements(config,subjects)
    assert len(imported['teacher'])==13
    assert sum(x['count'] for x in read(destination/'attempts.json').values())==130
    assert sha(source/'attempts.json')==before
    assert (destination/'sample_shards/12/FAILED.json').exists()
    assert import_measurements(config,subjects)==imported
    assert sum(x['count'] for x in read(destination/'attempts.json').values())==130
    write(destination/'sample_shards/0/aggregate.json',dict(corrupted=True))
    with pytest.raises(ValueError,match='changed'):import_measurements(config,subjects)


def test_import_rejects_changed_sampling_implementation_before_claim(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from agenticgen.uniguide import geometry_sampling as sampling
    from agenticgen.uniguide.calibration_import import import_measurements
    source,destination=tmp_path/'source',tmp_path/'new'
    write(source/'acceptance/COMPLETE.json',{})
    write(source/'slot_registry/one.json',dict(source_group='one',identity=dict(
        compatibility=dict(teacher_id='teacher',science='old'))))
    monkeypatch.setattr(sampling,'compatibility',lambda *a:dict(teacher_id='teacher',science='new'))
    with pytest.raises(ValueError,match='differ'):
        import_measurements(dict(root=str(destination),calibration=dict(measurement_source_root=str(source))),
            [dict(teacher=dict(target='teacher'),condition={})])
    assert not (destination/'CALIBRATION_MEASUREMENT_IMPORT_CLAIM.json').exists()
