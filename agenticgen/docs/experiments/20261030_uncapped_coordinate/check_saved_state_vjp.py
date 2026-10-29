"""Saved-state, two-teacher engineering check. No sampling or LLM calls."""
import datetime
import fcntl
import hashlib
import json
import subprocess
import types
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
from agenticgen.uniguide.absolute_coordinate import prepare, CoordinateObjective, match, POLICY
from agenticgen.uniguide.geometry_sampling import GeometryControl
from agenticgen.uniguide.geometry_dynamics import policy
from agenticgen.uniguide.geometry_shards import initial_shard
from agenticgen.uniguide.prepare import load_model
from agenticgen.uniguide.native_tools import load_pt, device
from agenticgen.uniguide.random_streams import capture, equal
from agenticgen.uniguide.condition import guidance_delta

REPO=Path('/workspace/uniguide_audits/uncapped-coordinate-6fc1dd05')
AUDIT=Path('/workspace/uniguide_audits/uncapped-coordinate-validation-6fc1dd05')
ROOT=Path('/workspace/agenticgen_absolute_coordinate_restart_20261029')
OLD=Path('/workspace/agenticgen_uniguide_flowr_20261026')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    before_files={str(p):sha(p) for p in [ROOT/'attempts.json',ROOT/'FINALIZED.json',OLD/'attempts.json']}
    assert read(AUDIT/'tests.json')['returncode']==0
    code=subprocess.check_output(['git','show','930db032:agenticgen/src/agenticgen/uniguide/condition.py'],cwd=REPO,text=True)
    legacy=types.ModuleType('legacy_coordinate_condition');exec(compile(code,'legacy_coordinate_condition.py','exec'),legacy.__dict__)
    config=read(ROOT/'GEOMETRY_CONFIG.json')
    config.pop('sampling_recovery',None);config.pop('maximum_single_atom_guidance_increment_angstrom')
    config.update(protocol=POLICY,root=str(AUDIT/'model_load'),initials_root=str(OLD))
    checks=[]
    with (OLD/'.executor.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for subject in read(ROOT/'SUBJECTS.json'):
            teacher,old_condition=subject['teacher'],subject['condition']
            features={k:old_condition[k] for k in ['atoms','bonds','controlled_atom_ids','coordinate_frame']}
            features['coordinates_angstrom']=old_condition['reference_coordinates_angstrom']
            condition=prepare(features,teacher['protein'])
            model=load_model(config,teacher)
            source=ROOT/'acceptance'/teacher['target']/'raw999_vjp_prefix/checkpoint_010.pt.gz'
            saved=load_pt(source);initial=initial_shard(config,teacher['target'],[0,1])
            curr,cond,times=(device(saved[k]) for k in ['current','cond','times'])
            pocket=device(initial['pocket_target']);com=initial['target_com'].to('cuda')
            with torch.no_grad():
                equis,invs=model.gen.get_pocket_encoding(pocket['coords'],pocket['atom_names'],
                    pocket_atom_charges=pocket['charges'].argmax(-1),pocket_bond_types=pocket['bonds'].argmax(-1),
                    pocket_res_types=pocket['res_names'],pocket_atom_mask=pocket['mask'])
            before=capture();x=curr['coords'].detach().clone().requires_grad_(True)
            out=model(dict(curr,coords=x),pocket,times,cond_batch=cond if model.self_condition else None,
                pocket_equis=equis,pocket_invs=invs,training=False)
            def denied(gradient):raise AssertionError('Affinity entered coordinate gradient')
            for value in out.get('affinity',{}).values():
                if torch.is_tensor(value) and value.requires_grad:value.register_hook(denied)
            prediction,_=model._get_predictions(out)
            endpoint=prediction['coords']*model.coord_scale+com[:,None];valid=curr['mask'].bool()
            objective=CoordinateObjective(condition);assignment=objective.bind(endpoint,valid,match(endpoint,condition,valid))
            old_loss=legacy.loss(endpoint,teacher['core_A'],valid)[0]
            new_loss=objective.evaluate(endpoint,valid,assignment)[0]
            old_gradient=torch.autograd.grad(old_loss.sum(),x,retain_graph=True)[0]
            new_gradient=torch.autograd.grad(new_loss.sum(),x,retain_graph=True)[0]
            offset=torch.zeros_like(endpoint);offset[0]=10000.
            high_loss=objective.evaluate(endpoint+offset,valid,assignment)[0]
            high_gradient=torch.autograd.grad(high_loss.sum(),x)[0]
            gamma=model.integrator.noise_scheduler(times[0][0])
            delta,_=guidance_delta(new_gradient,times[0],.00999,[4096.]*len(x),gamma,valid)
            controller=object.__new__(GeometryControl);controller.model=model;controller.sampling_policy=policy(config)
            unchanged=controller.constrain_guidance(delta,valid)
            check=dict(teacher=teacher['target'],source_checkpoint=str(source),condition_is_engineering_fixture=True,
                condition_policy=condition['policy'],old_new_input_gradient_bitwise_equal=bool(torch.equal(old_gradient,new_gradient)),
                high_rmsd_loss=float(high_loss[0]),high_rmsd_input_gradient_norm=float(high_gradient[0].norm()),
                high_rmsd_gradient_not_suppressed=bool(high_gradient[0].norm()>0),
                healthy_rows_unchanged_with_high_rmsd_candidate=bool(torch.equal(high_gradient[1:],new_gradient[1:])),
                returned_increment_is_original_tensor=unchanged is delta,
                maximum_single_atom_increment_angstrom=float((delta.double()*model.coord_scale).norm(dim=-1).max()),
                limiter=controller.sampling_policy['limiter'],sampling_rng_unchanged=equal(before,capture()),
                affinity_gradient_entered=False,model_forward_calls=1,integrator_calls=0)
            assert check['old_new_input_gradient_bitwise_equal']
            assert check['high_rmsd_gradient_not_suppressed'] and check['healthy_rows_unchanged_with_high_rmsd_candidate']
            assert check['returned_increment_is_original_tensor'] and check['maximum_single_atom_increment_angstrom']>3
            assert check['sampling_rng_unchanged'] and check['limiter']=='none'
            checks.append(check)
            del model,out,prediction,x,endpoint,old_gradient,new_gradient,high_gradient,controller
            torch.cuda.empty_cache()
    assert all(sha(Path(p))==h for p,h in before_files.items())
    result=dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        checks=checks,ledger_hashes_unchanged=True,science_candidates_generated=0,llm_calls=0,
        scope='same-state full real-model VJP and uncapped increment; not full sampling acceptance')
    (AUDIT/'model_engineering.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
