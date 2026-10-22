"""Native population execution with audited common randomness and compact parents."""
import copy
import gc
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .common_random import NativeStreams, capture, equal
from .flowr import (Control, cpu, device, load_initial, load_pt, save_pt, tensor_digest)
from .grid import CHECKPOINT_STEPS
from .io import read, write, sha
from .vendor.common import EXPECTED_CHECKPOINT, EXPECTED_SAMPLER
from .vendor.reward import LocalReward
from agenticgen.runtime.rng import set_rng, rng_state


def take(value, indices, size):
    if torch.is_tensor(value):
        return value[indices].clone() if value.ndim and value.shape[0] == size else value.clone()
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {k: take(v, indices, size) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) == size:
            return type(value)(copy.deepcopy(value[i]) for i in indices)
        return type(value)(take(v, indices, size) for v in value)
    return copy.deepcopy(value)


def join(values, sizes, indices):
    """Select rows from heterogeneous full populations without inferring slot identity."""
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim and all(v.shape[0] == size for v, size in zip(values, sizes)):
            return torch.cat([v[i:i+1] for v, i in zip(values, indices)], 0)
        if any(not torch.equal(first, v) for v in values[1:]):
            raise ValueError('Non-row tensor differs in mixed-parent recovery')
        return first.clone()
    if isinstance(first, dict) or hasattr(first, 'items'):
        return {k: join([v[k] for v in values], sizes, indices) for k in first.keys()}
    if isinstance(first, (list, tuple)):
        if all(len(v) == size for v, size in zip(values, sizes)):
            return type(first)(copy.deepcopy(v[i]) for v, i in zip(values, indices))
        return type(first)(join([v[i] for v in values], sizes, indices) for i in range(len(first)))
    if any(v != first for v in values[1:]):
        raise ValueError('Non-row metadata differs in mixed-parent recovery')
    return copy.deepcopy(first)


def initial_population(config, target, groups):
    initials = [load_initial(config, target, b)[0] for b in range(5)]
    per_seed = config.get('population_size', 100)//5
    size = per_seed*5
    blocks = [initials[b] for g in range(groups) for b in range(5) for _ in range(per_seed)]
    result = dict(initials[0])
    for key in ['prior', 'pocket_target', 'pocket_off', 'target_com']:
        result[key] = join([s[key] for s in blocks], [20]*len(blocks), list(range(per_seed))*(5*groups))
    result['population_size'] = size
    result['stream_states'] = [s['rng'] for s in initials]
    return result


class WaveControl(Control):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        states = self.resume['streams'] if self.resume else self.initial['stream_states']
        self.random_streams = NativeStreams(states, self.batch_size//self.initial.get('population_size',100),
            logical_rows=self.initial.get('population_size',100)//5)

    def initialize(self, curr, cond, times, prior, pt, po, et, it, eo, io, grid):
        context = {'prior': prior, 'pt': {k: v for k, v in pt.items() if k != 'complex'},
                   'po': {k: v for k, v in po.items() if k != 'complex'}, 'com': self.com,
                   'et': et, 'it': it, 'eo': eo, 'io': io, 'grid': grid,
                   'coord_scale': float(self.model.coord_scale), 'self_condition': self.model.self_condition,
                   'integrator': self.model.integrator.hparams, 'sampler_sha256': EXPECTED_SAMPLER,
                   'checkpoint_sha256': EXPECTED_CHECKPOINT,
                   'precision': torch.get_float32_matmul_precision(), 'device': torch.cuda.get_device_name(0)}
        # Encodings can differ at float-roundoff level with batch shape. Persist
        # each physical context, but bind mixed parents to immutable raw inputs.
        self.context_sha = tensor_digest({k: take(context[k], list(range(self.initial.get('population_size',100))), self.batch_size)
                                         for k in ['prior', 'pt', 'po', 'com', 'grid', 'coord_scale',
                                                   'integrator', 'sampler_sha256', 'checkpoint_sha256', 'precision', 'device']})
        if not self.context_path.exists():
            save_pt(self.context_path, context)
        elif tensor_digest(load_pt(self.context_path)) != tensor_digest(cpu(context)):
            raise ValueError('Physical native context changed')
        if self.resume:
            r = self.resume
            if r['context_sha256'] != self.context_sha:
                raise ValueError('Mixed-parent raw context mismatch')
            if r['current']['coords'].shape[0] != self.batch_size:
                raise ValueError('Resume row layout mismatch')
            curr, cond, times = device(r['current']), device(r['cond']), device(r['times'])
            self.model.integrator._guard_warned = r['integrator']['guard_warned']
            self.last_assignment = device(r.get('previous_assignment'))
        return curr, cond, times, et, it, eo, io

    def checkpoint(self, step, curr, cond, times):
        if step not in CHECKPOINT_STEPS:
            return
        state = {'schema': 'wave-crn-v1', 'step': step, 'rows': self.batch_size,
                 'current': curr, 'cond': cond, 'times': times,
                 'streams': self.random_streams.states, 'context_sha256': self.context_sha,
                 'context_path': str(self.context_path), 'previous_assignment': self.last_assignment,
                 'rng': rng_state(),
                 'integrator': {'hparams': self.model.integrator.hparams,
                                'guard_warned': self.model.integrator._guard_warned}}
        state['rng_sha256'] = tensor_digest(state['streams'])
        save_pt(self.folder/f'checkpoint_{step:03d}.pt.gz', state)

    def after(self, curr, step):
        result = super().after(curr, step)
        self.rows[-1]['logical_streams_sha256'] = tensor_digest(self.random_streams.states)
        self.rows[-1]['random_calls'] = list(self.random_streams.ledger)
        self.random_streams.ledger.clear()
        return result


def expand_resume(state, groups):
    if state['rows'] not in (25,100):
        raise ValueError('Only a verified hundred-chain incumbent can be a parent')
    r = copy.deepcopy(state)
    for key in ['current', 'cond', 'times', 'previous_assignment']:
        r[key] = take(state[key], list(range(state['rows']))*groups, state['rows'])
    r['rows'] = state['rows']*groups
    return r


def mix_checkpoint(parent, wave, winners):
    if parent['step'] != wave['step'] or parent['context_sha256'] != wave['context_sha256']:
        raise ValueError('Checkpoint context/time differs')
    if parent['rng_sha256'] != wave['rng_sha256'] or parent['integrator'] != wave['integrator']:
        raise ValueError('Logical streams/integrator differ across parents')
    state = copy.deepcopy(parent)
    sources = [wave if w is not None else parent for w in winners]
    indices = [w if w is not None else i for i, w in enumerate(winners)]
    for key in ['current', 'cond', 'times', 'previous_assignment']:
        state[key] = join([s[key] for s in sources], [s['rows'] for s in sources], indices)
    state['rows'] = len(winners)
    return state


def selected_document(row, index, size):
    return {k: v[index] if isinstance(v, list) and len(v) == size else v for k, v in row.items()}


def finish_nodes(model, initial, template, folder, records):
    output, state = load_pt(folder/'terminal.pt.gz'), load_pt(folder/'checkpoint_100.pt.gz')
    xyz = output['coords']
    current = state['current']['coords'] * model.coord_scale + initial['target_com'][:, None]
    reward = LocalReward(template, 'C', 'endpoint')
    value, _, assignment, _ = reward.loss(current, xyz, {}, output['mask'].bool())
    ids = assignment[:, reward.core_columns]
    row = torch.arange(len(records))[:, None]
    nodes = read(folder/'nodes.json')
    nodes['100'] = {'step': 100, 'state_kind': 'actual_terminal', 'local_loss': value.tolist(),
        'local_current_A': current[row, ids].tolist(), 'local_endpoint_A': xyz[row, ids].tolist(),
        'assignment': assignment.tolist(), 'pic50_on_rescore': [r.get('pic50_on_rescore') for r in records]}
    write(folder/'nodes.json', nodes)


def execute_wave(model, config, template, schedules, folder, context_path, parent=None, start_step=0):
    from agenticgen.runtime.scoring import final_metrics
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    population = config.get('population_size',100)
    size, groups = len(schedules), len(schedules)//population
    if size not in (population, population*4):
        raise ValueError('Formal execution must contain one or four configured populations')
    binding = {'schedules': schedules, 'start_step': start_step,
               'parent_manifest': sha(parent/'COMPLETE.json') if parent else None,
               'teacher_id': template['teacher_id'], 'groups': groups, 'rows': size, 'population_size': population}
    if (folder/'manifest.json').exists() and read(folder/'manifest.json') != binding:
        raise ValueError('Wave identity changed')
    write(folder/'manifest.json', binding)
    if (folder/'COMPLETE.json').exists():
        done = read(folder/'COMPLETE.json')
        if done['records_sha256'] != sha(folder/'final_records.json'):
            raise ValueError('Completed wave records changed')
        return read(folder/'final_records.json')
    initial = initial_population(config, template['target'], groups)
    prior, pt, po = (device(initial[k]) for k in ['prior', 'pocket_target', 'pocket_off'])
    for pocket in [pt, po]:
        pocket['complex'] = [SimpleNamespace(com=x.numpy()) for x in initial['target_com']]
    started = time.time()
    terminal = folder/'terminal.pt.gz'
    if not terminal.exists():
        resume = expand_resume(load_pt(parent/f'checkpoint_{start_step:03d}.pt.gz'), groups) if parent else None
        from .search_policy import upper_tick
        control = WaveControl(model, template, initial, schedules, folder, context_path, resume, tick_upper=upper_tick(config))
        model._adaptive = control
        set_rng(initial['rng'])
        torch.cuda.reset_peak_memory_stats()
        try:
            output = model._generate_selective(prior=prior, pocket_data_target=pt, pocket_data_untarget=po,
                steps=100, times=[torch.zeros(size, device='cuda') for _ in range(3)],
                apply_guidance=False, coord_noise_level=.2)
            save_pt(terminal, output)
            write(folder/'SAMPLING_COMPLETE.json', {'seconds': time.time()-started,
                'start_step': start_step, 'steps': 100-start_step, 'rows': size,
                'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
                'streams_sha256': tensor_digest(control.random_streams.states), 'affinity_vjp': False})
        finally:
            control.close()
            del model._adaptive
            model.integrator._adaptive_observer = None
    output = load_pt(terminal)
    if not (folder/'EVALUATION_COMPLETE.json').exists():
        trace = SimpleNamespace(pt=pt, po=po, batch=0, seed=42, arm=folder.name,
                                roots=np.arange(size), parents=np.arange(size), events=list(range(100)))
        external = capture()
        try:
            records = final_metrics(model, output, trace, folder)
        finally:
            from .common_random import restore
            restore(external)
        for i, r in enumerate(records):
            r.update(candidate_id=f'b{i%population//(population//5):03d}:s{i%(population//5):03d}', logical_batch=i%population//(population//5),
                     logical_slot=i%(population//5), hypothesis=i//population, physical_slot=i)
        write(folder/'final_records.json', records)
        write(folder/'EVALUATION_COMPLETE.json', {'attempts': len(records), 'time': time.time()})
    records = read(folder/'final_records.json')
    if len(records) != size:
        raise ValueError('Missing native attempts')
    finish_nodes(model, initial, template, folder, records)
    write(folder/'COMPLETE.json', {'records_sha256': sha(folder/'final_records.json'), 'attempts': size,
                                 'seconds_this_invocation': time.time()-started})
    del prior, pt, po, output
    gc.collect()
    return records
