"""One loaded model per condition; isolated, durable per-state inference."""
import argparse
import copy
import gzip
import hashlib
import inspect
import os
from pathlib import Path
import random
import sys
import textwrap
import time
import traceback
import types
from types import SimpleNamespace

import numpy as np
import torch

from evomolsteer.io import read_json, digest
from evomolsteer.storage.transactions import atomic_json
from .common import validate_protocol, validate_condition, ordered_attempts, read_result, result_path, binding, failure


def run(repo, config_path, work, condition_id, pilot=False):
    repo, work = Path(repo).resolve(), Path(work).resolve()
    config = read_json(config_path); validate_protocol(config)
    condition = next(x for x in config['conditions'] if x['id'] == condition_id)
    state_file = work / 'worker_state.json'
    def heartbeat(**kw):
        atomic_json(state_file, {'condition': condition_id, 'pid': os.getpid(), 'updated_unix': time.time(), **kw})
    heartbeat(phase='loading')
    program, teacher = validate_condition(repo, condition)
    sys.path.insert(0, config['flowr_root'])
    from flowr.gen.generate_from_pdb_selective import get_args
    from flowr.scriptutil import load_model
    from evomolsteer.final_guidance.native import instrument, instrument_coord_telemetry, Control
    from evomolsteer.generation.scalar_guidance import detached
    from evomolsteer.agentic_steer.native import molecule_quality
    if digest(config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('Global checkpoint checksum differs')
    argv = sys.argv
    sys.argv = ['elite-alignment', '--arch', 'pocket', '--pocket_type', 'holo', '--gpus', '1', '--num_workers', '0',
        '--batch_cost', '1', '--sample_n_molecules_per_target', '1', '--ckpt_path', config['checkpoint'],
        '--save_dir', str(work / 'methods' / condition_id), '--integration_steps', str(config['steps']),
        '--no_cat_noise_euler_guard', '--no_ligand_valence_repair']
    try:
        args = get_args()
    finally:
        sys.argv = argv
    args.seed = config['master_seed']; torch.set_float32_matmul_precision('high')
    model = load_model(args)[0].to('cuda').eval(); model.requires_grad_(False)
    if model.sc_recycle_steps != 1 or model.inpainting_mode or model.graph_inpainting or model.integrator.use_cosine_scheduler:
        raise ValueError('Unsupported native integration configuration')
    for key, value in config['native_options'].items():
        if key != 'final_corr_pred': setattr(model.integrator, key, value)
    original = model._generate
    source = textwrap.dedent(inspect.getsource(original)); changed = instrument(source)
    namespace = dict(original.__func__.__globals__)
    def forbidden(*a, **kw):
        raise RuntimeError('SMC is forbidden in an aligned guidance experiment')
    namespace.update(apply_smc_guidance=forbidden, apply_selective_smc_guidance=forbidden)
    exec(compile(changed, 'elite_aligned_native.py', 'exec'), namespace)
    model._generate = types.MethodType(namespace['_generate'], model)
    coord_source = textwrap.dedent(inspect.getsource(model.integrator.coord_step))
    coord_namespace = dict(model.integrator.coord_step.__func__.__globals__)
    exec(compile(instrument_coord_telemetry(coord_source), 'elite_coord_telemetry.py', 'exec'), coord_namespace)
    model.integrator.coord_step = types.MethodType(coord_namespace['coord_step'], model.integrator)
    provenance = {'protocol_sha256': config['protocol_sha256'], 'condition': condition,
        'checkpoint_sha256': config['checkpoint_sha256'], 'native_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'integrator_sha256': digest(inspect.getsourcefile(type(model.integrator))), 'torch': torch.__version__,
        'native_options': config['native_options'], 'no_smc': True, 'affinity_head_gradient': False,
        'model_loaded_once': True, 'batch_size': 1, 'real_FLOWR_VJP': condition['kind'] != 'unguided'}
    from subprocess import check_output
    provenance['commit'] = os.environ.get('EVOMOLSTEER_RUNTIME_COMMIT') or check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    atomic_json(work / 'methods' / condition_id / 'source_audit.json', provenance)
    try:
        from posebusters import PoseBusters
        from evomolsteer.local_motif.quality import binary_checks
        pb = PoseBusters(config='dock', max_workers=0)
    except ImportError:
        pb = None
    cache = {}; baseline_verified = (work / 'native_zero_equivalence.json').exists()
    for attempt in ordered_attempts(config['attempts']):
        if pilot and attempt['rank'] not in (1,50,100,200): continue
        existing = read_result(work, config, condition, attempt)
        if existing is not None:
            continue
        folder = result_path(work, condition, attempt).parent; folder.mkdir(parents=True, exist_ok=True)
        heartbeat(phase='inference', attempt=attempt['id'], rank=attempt['rank'], step=-1)
        started = time.time()
        try:
            key = attempt['initial_id']; spec = config['initials'][key]
            if key not in cache:
                initial = work / spec['file']
                if digest(initial) != spec['sha256']:
                    raise ValueError('Initial-state checksum differs')
                with gzip.open(initial, 'rb') as stream:
                    cache[key] = torch.load(stream, map_location='cpu', weights_only=False)
            initial = cache[key]
            from evomolsteer.agentic_steer.native import slice_tree
            prior = slice_tree(initial['prior'], 0, 1, 'cuda')
            pocket = slice_tree(initial['pocket_target'], 0, 1, 'cuda')
            com = torch.as_tensor(np.asarray(initial['target_com']).reshape(1, 3), device='cuda', dtype=torch.float32)
            pocket['complex'] = [SimpleNamespace(com=com[0].cpu().numpy())]
            if abs(model.coord_scale - initial['coord_scale']) > 1e-12:
                raise ValueError('Initial state and model coordinate scales differ')
            control_class = Control
            if condition['kind'] == 'legacy':
                from .control import LegacyControl
                control_class = LegacyControl
            elif program is not None:
                families = {'local_nc_layout': ('evomolsteer.nc_guidance.reward', 'NCControl'),
                            'local_motif_v2': ('evomolsteer.local_motif.reward', 'MotifControl'),
                            'ancestral_modes_v3': ('evomolsteer.ancestral_modes.reward', 'AncestorControl')}
                import importlib
                module, symbol = families[program['family']]
                control_class = getattr(importlib.import_module(module), symbol)
            control = control_class(model, copy.deepcopy(program), copy.deepcopy(teacher), com, .5)
            forward = control.forward
            def observed(curr, pocket_data, times, cond, equis, invs, i, dt):
                if i % 10 == 0:
                    heartbeat(phase='inference', attempt=attempt['id'], rank=attempt['rank'], step=i)
                return forward(curr, pocket_data, times, cond, equis, invs, i, dt)
            control.forward = observed; model._half_guidance = control
            def reset_rng():
                torch.manual_seed(attempt['suffix_seed']); np.random.seed(attempt['suffix_seed']); random.seed(attempt['suffix_seed'])
            kw = dict(steps=config['steps'], times=[torch.zeros(1, device='cuda') for _ in range(3)],
                      apply_guidance=False, strategy='linear', solver='euler', save_traj=False,
                      coord_noise_level=.2, final_corr_pred=True)
            reset_rng()
            if condition['kind'] == 'unguided' and not baseline_verified:
                expected = original(copy.deepcopy(prior), copy.deepcopy(pocket), **copy.deepcopy(kw))
                reset_rng()
            output = model._generate(prior, pocket, **kw)
            if condition['kind'] == 'unguided' and not baseline_verified:
                checks = {k: bool(torch.equal(output[k], expected[k])) for k in ('coords', 'atomics', 'bonds', 'charges', 'mask')}
                atomic_json(work / 'native_zero_equivalence.json', {'passed': all(checks.values()), 'fields': checks, 'attempt': attempt['id']})
                if not all(checks.values()): raise ValueError('Zero-controller native equivalence failed')
                baseline_verified = True
            active = [x['step'] for x in control.rows if x['active']]
            expected_steps = [] if program is None else [i for i in range(config['steps'])
                if i / config['steps'] >= program['window'][0] - 1e-6 and (i+1) / config['steps'] <= program['window'][1] + 1e-6]
            if active != expected_steps or len(control.rows) != config['steps']:
                raise ValueError('Actual guidance coverage differs from the frozen window')
            if program and not any(x.get('nonzero_molecules', 0) for x in control.rows):
                raise ValueError('Guidance was active but injected no displacement')
            heartbeat(phase='scoring', attempt=attempt['id'], rank=attempt['rank'], step=100)
            final = {k: v.to('cuda') for k, v in output.items() if torch.is_tensor(v) and k != 'affinity'}
            final['coords'] = (final['coords'] - com[:, None]) / model.coord_scale
            for field in ('atomics', 'bonds', 'charges'):
                final[field] = torch.nn.functional.one_hot(final[field].argmax(-1), final[field].shape[-1]).float()
            with torch.no_grad():
                score = float(model._predict_affinity(final, final, pocket,
                    [torch.full((1,), .9999, device='cuda')] * 3)['affinity']['pic50'].reshape(-1)[0])
            if not np.isfinite(score): raise ValueError('Nonfinite terminal affinity score')
            np.savez_compressed(folder / 'states.npz', half_coords_A=np.asarray(control.teacher_snapshot['coords_A'], dtype=np.float32),
                terminal_coords_A=output['coords'].cpu().numpy().astype(np.float32), mask=output['mask'].cpu().numpy().astype(bool),
                **{field: output[field].argmax(-1).cpu().numpy().astype(np.uint8) for field in ('atomics', 'bonds', 'charges')})
            atomic_json(folder / 'guidance.json', {'steps': control.rows, 'active_steps': active, 'cumulative_rms_A': float(control.total[0]),
                'initial_sha256': spec['sha256'], 'suffix_seed': attempt['suffix_seed'], 'no_smc': True, 'affinity_head_gradient': False})
            molecules = model._generate_mols(output); mol = molecules[0]
            quality = molecule_quality(mol, pocket, com[0].cpu().numpy(), model.coord_scale)
            quality['pb_status'] = 'not_decodable' if mol is None else 'unavailable'
            if mol is not None and pb is not None:
                try:
                    checks = binary_checks(pb.bust(mol, mol_cond=Path(config['protein']), full_report=False).iloc[0].to_dict())
                    quality.update(pb_status='evaluated', pb_pass=bool(checks and all(v is True for v in checks.values())),
                                   pb_tests=checks)
                except Exception as error:
                    quality.update(pb_status='error', pb_error=type(error).__name__+': '+str(error))
            if mol is not None:
                from rdkit import Chem
                with gzip.open(folder / 'terminal.sdf.gz', 'wt') as stream:
                    stream.write(Chem.MolToMolBlock(mol) + '\n$$$$\n')
            files = ['states.npz', 'guidance.json'] + (['terminal.sdf.gz'] if mol is not None else [])
            record = {'binding': binding(config, condition, attempt), 'status': 'complete', 'attempt': attempt,
                'condition': condition_id, 'pic50': score, **quality, 'seconds': time.time()-started,
                'finished_unix': time.time(), 'artifacts': {name: digest(folder / name) for name in files}}
            atomic_json(folder / 'result.json', record)
            print({'condition': condition_id, 'rank': attempt['rank'], 'pic50': score, 'valid': quality['valid_connected']}, flush=True)
        except Exception as error:
            failure(work, config, condition, attempt, traceback.format_exc())
            print({'condition': condition_id, 'rank': attempt['rank'], 'error': type(error).__name__ + ': ' + str(error)}, flush=True)
            if isinstance(error, torch.cuda.OutOfMemoryError) or any(s in str(error).lower() for s in ('device-side assert', 'illegal memory', 'hip error')):
                raise
        finally:
            torch.cuda.empty_cache()
    heartbeat(phase='complete')


def main():
    p = argparse.ArgumentParser()
    for k in ('repo', 'config', 'work', 'condition'): p.add_argument('--'+k, required=True)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    try:
        run(a.repo, a.config, a.work, a.condition, a.pilot)
    except Exception:
        atomic_json(Path(a.work) / 'worker_error.json', {'condition': a.condition, 'error': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
