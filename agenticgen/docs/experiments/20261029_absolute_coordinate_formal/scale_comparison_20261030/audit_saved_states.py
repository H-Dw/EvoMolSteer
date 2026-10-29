"""Read saved trajectories and compare frozen coordinate objectives; never sample."""
import argparse
import datetime
import fcntl
import hashlib
import json
import subprocess
import types
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import torch

REPO = Path('/workspace/uniguide_runs/absolute-restart-1fa48243')
ROOT = Path('/workspace/agenticgen_absolute_coordinate_restart_20261029')
OLD = Path('/workspace/agenticgen_uniguide_flowr_20261026')
OUT = Path('/workspace/uniguide_audits/absolute-scale-comparison-20261030')
OLD_COMMIT = '930db03280c66c3f8a39f131ae01df3202544af4'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def old_module():
    text = subprocess.check_output(['git', 'show', OLD_COMMIT + ':agenticgen/src/agenticgen/uniguide/condition.py'], cwd=REPO, text=True)
    module = types.ModuleType('archived_ug04_condition')
    exec(compile(text, 'UG04_condition_930db032.py', 'exec'), module.__dict__)
    return module, text


def cpu_audit():
    from agenticgen.uniguide.absolute_coordinate import CoordinateObjective, match, RMSD
    from agenticgen.uniguide.geometry_protocol import displacement
    from agenticgen.uniguide.native_tools import load_pt
    legacy, text = old_module()
    write('archived_condition_source.json', dict(commit=OLD_COMMIT, sha256=hashlib.sha256(text.encode()).hexdigest(), source=text))
    current_config = read(ROOT / 'GEOMETRY_CONFIG.json')
    old_config = read(OLD / 'config.json')
    protected = read(ROOT / 'SOURCE_PRESERVATION.json')
    changed = [p for p, h in protected.items() if sha(p) != h]
    assert not changed
    protected_ledger = {str(p): sha(p) for p in [ROOT/'attempts.json', OLD/'attempts.json', ROOT/'FINALIZED.json']}
    current_sources = read(ROOT / 'engineering_audits/calibration_boundary_stop_20261030_0010/source_hashes.json')
    assert all(sha(REPO / p) == h for p, h in current_sources.items())
    details = []
    for subject in read(ROOT / 'SUBJECTS.json'):
        teacher, condition = subject['teacher'], subject['condition']
        ids = [a['id'] for a in condition['atoms']]
        core = [condition['reference_coordinates_angstrom'][ids.index(i)] for i in condition['controlled_atom_ids']]
        assert core == teacher['core_A']
        initial = load_pt(OLD / 'initials' / teacher['target'] / 'batch_000.pt.gz')
        details.append(dict(teacher=teacher['target'], condition_policy=condition['policy'],
            loss_reduction=condition['loss_reduction'], controlled_atom_count=len(core),
            reference_coordinates_exactly_equal=True, condition_sha256=condition['condition_sha256'],
            rotation=condition['protein_alignment']['rotation'],
            translation_angstrom=condition['protein_alignment']['translation_angstrom'],
            initial_coord_scale=initial.get('coord_scale'),
            old_ug04_manifest={k:v for k,v in read(OLD/'versions/UG04/manifest.json').items()
                              if k in ('commit','git_commit','config_sha256','source_binding','version')}))
    coarse = read(next(ROOT.glob('calibration/**/coarse_results.json')))
    condition = read(ROOT / 'SUBJECTS.json')[0]['condition']
    teacher = read(ROOT / 'SUBJECTS.json')[0]['teacher']
    objective = CoordinateObjective(condition)
    evidence = []
    trace_arrays = {}
    previous = None
    for row in coarse:
        folder = ROOT / row['sampling_provenance'][0]['source_group']
        steps = read(folder / 'steps.json')
        raw_by_step = []
        for step in steps:
            if 10 <= step['step'] <= 99:
                factors = np.asarray(step['guidance_increment_scale_factor'])
                raw = np.asarray(step['unconstrained_max_atom_increment_A'])
                raw_by_step.append(dict(step=step['step'], time=step['time'], dt=step['dt'],
                    limited_count=sum(step['guidance_increment_limited']), maximum_raw=float(raw.max()),
                    maximum_applied=max(step['max_atom_increment_A']),
                    minimum_factor=float(factors.min()), median_factor=float(np.median(factors)),
                    effective_nominal_times_factor=(factors * row['effective_guidance_scale_schedule'][0]).tolist()))
        comparisons = []
        with h5py.File(folder / 'trace.h5', 'r') as trace:
            a10 = trace['assignment'][10]
            assert np.array_equal(a10, trace['assignment'][99])
            for step in [10, 11, 25, 50, 75, 90, 98, 99]:
                endpoint = torch.from_numpy(trace['endpoint_A'][step]).requires_grad_(True)
                valid = torch.from_numpy(trace['mask'][step]).bool()
                assignment = torch.from_numpy(trace['assignment'][step]).long()
                new_value = objective.evaluate(endpoint, valid, objective.bind(endpoint, valid, assignment))[0]
                target = endpoint.detach().clone()
                target.scatter_(1, assignment[...,None].expand(-1,-1,3), endpoint.new_tensor(teacher['core_A'])[None].expand(len(endpoint),-1,-1))
                selected = torch.zeros_like(valid).scatter_(1, assignment, True)
                same_assignment_old = legacy.coordinate_loss(endpoint, target, selected)
                gnew = torch.autograd.grad(new_value.sum(), endpoint, retain_graph=True)[0]
                gold = torch.autograd.grad(same_assignment_old.sum(), endpoint, retain_graph=True)[0]
                dynamic, _, _, dynamic_assignment = legacy.loss(endpoint, teacher['core_A'], valid)
                comparisons.append(dict(step=step, same_mapping_loss_max_abs_diff=float((new_value-same_assignment_old).abs().max()),
                    same_mapping_endpoint_gradient_max_abs_diff=float((gnew-gold).abs().max()),
                    current_mapping_is_dynamic_optimum_count=int((assignment==dynamic_assignment).all(1).sum()),
                    frozen_mapping_rmsd_mean=float((2*new_value).sqrt().mean()),
                    rematched_on_identical_endpoint_rmsd_mean=float((2*dynamic).sqrt().mean())))
            if row['effective_guidance_scale_schedule'][0] in [256., 1024., 4096.]:
                trace_arrays[row['effective_guidance_scale_schedule'][0]] = {
                    k:trace[k][10:14] for k in ['input_gradient','guidance_delta_A','guided_proposal','assignment']}
        d = displacement(row)
        scale = row['effective_guidance_scale_schedule'][0]
        growth = ({k:d[k]/max(previous['displacement'][k]*(scale/previous['scale']),1e-12) for k in d}
                  if previous is not None else {})
        evidence.append(dict(actual_scale=scale, measured_mean_rmsd=row[RMSD], built=row['built'],
            group_id=row['id'], sample_folder=str(folder),
            raw_maximum=row['maximum_unconstrained_single_atom_guidance_increment_angstrom'],
            applied_maximum=row['maximum_single_atom_guidance_increment_angstrom'],
            limited_candidate_steps=row['guidance_increment_limited_candidate_steps'],
            limited_candidates=row['guidance_increment_limited_candidates'],
            risk_check_displacement=d, normalized_growth=growth, selected_step_objective_comparisons=comparisons,
            active_steps=raw_by_step))
        previous = dict(scale=scale, displacement=d)
    prefix_comparisons = []
    for lo, hi in [(256.,1024.),(1024.,4096.)]:
        prefix_comparisons.append(dict(scales=[lo,hi], steps=[10,11,12,13],
            exact={k:[bool(np.array_equal(a,b)) for a,b in zip(trace_arrays[lo][k],trace_arrays[hi][k])] for k in trace_arrays[lo]}))
    old_tables = {}
    for phase in ['coarse','fine','full']:
        rows = read(OLD / 'calibration' / (phase+'_results.json'))
        old_tables[phase] = [{k:r.get(k) for k in ['id','target','schedule','attempts','complete','C','C_se','gap','built','error','window_diagnostics']} for r in rows]
    write('legacy_calibration.json', old_tables)
    write('trajectory_audit.json', dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        runtime_clean=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)=='',
        protected_files_count=len(protected), protected_files_changed=changed,
        ledger_hashes_before=protected_ledger,
        checkpoint_sha256_old=old_config['checkpoint_sha256'],checkpoint_sha256_new=current_config['checkpoint_sha256'],
        native_hashes_old=old_config['native_hashes'],native_hashes_new=current_config['native_hashes'],
        subjects=details,coarse=evidence,prefix_scale_comparisons=prefix_comparisons,
        new_sampling_attempts=0,new_llm_calls=0,old_and_new_loss_comparison_uses_identical_saved_endpoints=True))
    print(json.dumps(dict(output=str(OUT/'trajectory_audit.json'),groups=len(evidence),subjects=details,
        max_equal_mapping_gradient_diff=max(c['same_mapping_endpoint_gradient_max_abs_diff'] for row in evidence for c in row['selected_step_objective_comparisons']),
        max_equal_mapping_loss_diff=max(c['same_mapping_loss_max_abs_diff'] for row in evidence for c in row['selected_step_objective_comparisons']),
        ledgers_unchanged=all(sha(p)==h for p,h in protected_ledger.items())),ensure_ascii=True))


def vjp_audit():
    from agenticgen.uniguide.absolute_coordinate import CoordinateObjective, match
    from agenticgen.uniguide.native_tools import load_pt, device
    from agenticgen.uniguide.prepare import load_model
    from agenticgen.uniguide.geometry_shards import initial_shard
    from agenticgen.uniguide.random_streams import capture, equal
    from agenticgen.uniguide.condition import guidance_delta
    legacy, _ = old_module()
    config = read(ROOT / 'GEOMETRY_CONFIG.json')
    config = dict(config,root=str(OUT/'engineering_model_load'))
    checks = []
    lock = (OLD/'.executor.lock').open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    for subject in read(ROOT/'SUBJECTS.json'):
        teacher, condition = subject['teacher'], subject['condition']
        model = load_model(config,teacher)
        initial = initial_shard(config,teacher['target'],[0,1])
        path = ROOT/'acceptance'/teacher['target']/'raw999_vjp_prefix/checkpoint_010.pt.gz'
        state = load_pt(path)
        curr, cond, times = (device(state[k]) for k in ['current','cond','times'])
        pocket = device(initial['pocket_target'])
        com = initial['target_com'].to('cuda')
        with torch.no_grad():
            equis,invs=model.gen.get_pocket_encoding(pocket['coords'],pocket['atom_names'],
                pocket_atom_charges=pocket['charges'].argmax(-1),pocket_bond_types=pocket['bonds'].argmax(-1),
                pocket_res_types=pocket['res_names'],pocket_atom_mask=pocket['mask'])
        before=capture()
        x=curr['coords'].detach().clone().requires_grad_(True)
        out=model(dict(curr,coords=x),pocket,times,cond_batch=cond if model.self_condition else None,
                  pocket_equis=equis,pocket_invs=invs,training=False)
        def denied(gradient):raise AssertionError('Affinity entered coordinate VJP')
        for head in out.get('affinity',{}).values():
            if torch.is_tensor(head) and head.requires_grad:head.register_hook(denied)
        pred,_=model._get_predictions(out)
        endpoint=pred['coords']*model.coord_scale+com[:,None]
        valid=curr['mask'].bool()
        assignment=match(endpoint,condition,valid)
        old_value,_,_,old_assignment=legacy.loss(endpoint,teacher['core_A'],valid)
        objective=CoordinateObjective(condition)
        new_value=objective.evaluate(endpoint,valid,objective.bind(endpoint,valid,assignment))[0]
        old_grad=torch.autograd.grad(old_value.sum(),x,retain_graph=True)[0]
        new_grad=torch.autograd.grad(new_value.sum(),x)[0]
        scale=[181.01933598375618]*len(x)
        gamma=model.integrator.noise_scheduler(times[0][0])
        old_delta, old_coeff=legacy.guidance_delta(old_grad,times[0],.00999,scale,gamma,valid)
        new_delta, new_coeff=guidance_delta(new_grad,times[0],.00999,scale,gamma,valid)
        checks.append(dict(teacher=teacher['target'],checkpoint=str(path),checkpoint_sha256=sha(path),
            coordinate_scale=float(model.coord_scale),time=float(times[0][0]),population=len(x),
            equal_initial_assignments=bool(torch.equal(assignment,old_assignment)),
            old_loss_values=old_value.tolist(),new_loss_values=new_value.tolist(),
            maximum_loss_difference=float((old_value-new_value).abs().max()),
            maximum_input_vjp_difference=float((old_grad-new_grad).abs().max()),
            input_vjp_bitwise_equal=bool(torch.equal(old_grad,new_grad)),
            old_gradient_norm=float(old_grad.norm()),new_gradient_norm=float(new_grad.norm()),
            guidance_coefficient_bitwise_equal=bool(torch.equal(old_coeff,new_coeff)),
            uncapped_guidance_max_abs_difference=float((old_delta-new_delta).abs().max()),
            model_parameters_frozen=all(not p.requires_grad and p.grad is None for p in model.parameters()),
            sampling_rng_unchanged=equal(before,capture()),affinity_gradient_entered=False,
            real_model_forward_calls=1,integrator_calls=0,scientific_candidates_generated=0))
        del model,out,pred,x,old_grad,new_grad,endpoint
        torch.cuda.empty_cache()
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    write('real_model_vjp_comparison.json',dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        old_source_commit=OLD_COMMIT,new_source_commit='1fa48243453cfdb801c6becff72d54a8734b9c10',
        comparisons=checks,check_uses_saved_prefix_not_new_sampling=True))
    print(json.dumps(dict(output=str(OUT/'real_model_vjp_comparison.json'),checks=checks),ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cpu','vjp']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1)
    (cpu_audit if args.mode=='cpu' else vjp_audit)()
