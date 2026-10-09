import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('scale_campaign', ROOT / 'scripts/run_guidance_scale_campaign.py')
scale = importlib.util.module_from_spec(loader)
loader.loader.exec_module(scale)


@pytest.fixture
def spec():
    return json.loads((ROOT / 'configs/experiments/flowcompat_scale1000_v1/spec.json').read_text())


def test_frozen_sample_size_and_both_reward_bindings(spec):
    scale.validate_spec(spec)
    assert spec['n_per_arm'] == 1000 and spec['batch'] == 50
    for cohort in spec['cohorts']:
        assert scale.sha(ROOT / cohort['program']) == cohort['program_sha256']
        assert scale.sha(ROOT / cohort['reference']) == cohort['reference_sha256']
    assert min(spec['batch_indices']) > 54


@pytest.mark.parametrize('field,value', [('n_per_arm', 999), ('n_per_arm', True),
    ('batch', 0), ('batch_indices', [55]*20), ('batch_indices', list(range(55,74))),
    ('seed', 43), ('steps', 50)])
def test_bad_counts_random_streams_or_grid_rejected(spec, field, value):
    spec[field] = value
    with pytest.raises(ValueError):
        scale.validate_spec(spec)


def test_no_selection_arm_can_be_submitted(spec):
    spec['cohorts'][0]['arms'] = 'single,gradient'
    with pytest.raises(ValueError, match='resampling'):
        scale.validate_spec(spec)


def test_generation_reuses_frozen_interface_and_full_inference(spec, tmp_path):
    args = scale.generation_command(ROOT, tmp_path/'flowr', tmp_path/'work', spec, spec['cohorts'][0], tmp_path/'checkpoint')
    values = {args[i]: args[i+1] for i in range(3, len(args)-1, 2)}
    assert values['--n'] == '1000' and values['--steps'] == '100'
    assert values['--batch'] == '50' and values['--arms'] == 'gradient'
    assert values['--batch-indices'] == ','.join(map(str, range(55,75)))
    assert args[-1] == '--export-terminal'
    assert args[2].endswith('generate_flowcompat_v2_flowr.py')


def test_terminal_view_requires_complete_exact_coverage(spec):
    view = {'batches': [{'batch_path': f'gradient/batch_{i:03d}', 'steps': 100,
        'initial_state_signature': str(i)} for i in spec['batch_indices']]}
    assert len(scale.validate_execution_view(view, spec, 'gradient')) == 20
    bad = copy.deepcopy(view)
    bad['batches'].pop()
    with pytest.raises(ValueError, match='coverage'):
        scale.validate_execution_view(bad, spec, 'gradient')
    bad = copy.deepcopy(view)
    bad['batches'][0]['steps'] = 50
    with pytest.raises(ValueError, match='inference'):
        scale.validate_execution_view(bad, spec, 'gradient')


@pytest.mark.parametrize('corruption', ['resampled', 'selected', 'truncated'])
def test_real_selection_records_must_be_identity(spec, corruption):
    state = {'resampled': np.zeros(100, np.uint8), 'selected_indices': np.tile(np.arange(50), (100, 1))}
    scale.validate_selection_state(state, spec)
    if corruption == 'resampled':
        state['resampled'][2] = 1
    elif corruption == 'selected':
        state['selected_indices'][2,3] = 4
    else:
        state['resampled'] = state['resampled'][:-1]
    with pytest.raises(ValueError, match='selection'):
        scale.validate_selection_state(state, spec)


def test_three_arm_pairing_rejects_a_different_initial_state(spec):
    left = {'initial_state_signatures': {f'gradient/{i}': str(i) for i in spec['batch_indices']}}
    right = {'initial_state_signatures': {f'{arm}/{i}': str(i) for arm in ['gradient','unguided'] for i in spec['batch_indices']}}
    scale.validate_pairing(left, right, spec['batch_indices'])
    right['initial_state_signatures']['unguided/60'] = 'different'
    with pytest.raises(ValueError, match='not paired'):
        scale.validate_pairing(left, right, spec['batch_indices'])


def test_terminal_seeds_survive_byte_for_byte_before_retirement(tmp_path):
    run, work = tmp_path/'run', tmp_path/'work'
    pieces = [b'first molecule\n$$$$\n', b'second molecule\n$$$$\n']
    for batch, content in zip([55,56], pieces):
        path = run / f'gradient/batch_{batch:03d}'
        path.mkdir(parents=True)
        (path / 'molecules_all_built.sdf').write_bytes(content)
    (run / 'final_records.json').write_text('[{"build_success": false}]')
    result = scale.retain_terminal(run, work, {'name': 'R11', 'arms': 'gradient'})
    assert Path(result['gradient']['path']).read_bytes() == b''.join(pieces)
    assert (work / 'seed_structures/R11/final_records.json').read_bytes() == (run / 'final_records.json').read_bytes()


def test_launch_detaches_and_refuses_duplicate_submission(spec, tmp_path, monkeypatch):
    work, repo, flowr = tmp_path/'work', tmp_path/'repo', tmp_path/'flowr'
    config = tmp_path/'spec.json'
    config.write_text(json.dumps(spec))
    monkeypatch.setattr(scale, 'validate_inputs', lambda *a: (spec, Path('checkpoint')))
    monkeypatch.setattr(scale.subprocess, 'check_output', lambda *a, **kw: 'commit123\n')
    calls = []
    class Process:
        pid = 12345
    def spawn(*a, **kw):
        calls.append((a,kw))
        assert not (work/'launch.json').exists()
        return Process()
    monkeypatch.setattr(scale.subprocess, 'Popen', spawn)
    result = scale.launch(repo, flowr, work, config)
    assert result['pid'] == 12345 and result['monitor'] == 'none'
    assert calls[0][1]['start_new_session'] and calls[0][1]['stdin'] == scale.subprocess.DEVNULL
    assert scale.read(work/'launch.json') == scale.read(work/'status.json')
    with pytest.raises(FileExistsError):
        scale.launch(repo, flowr, work, config)
    assert len(calls) == 1
