"""Read saved states and compare old/prepared VJPs; never integrate or call an LLM."""
import fcntl
import gc
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from agenticgen.uniguide.absolute_coordinate import CoordinateObjective
from agenticgen.uniguide.geometry_shards import initial_shard
from agenticgen.uniguide.io import read, sha, write
from agenticgen.uniguide.native_tools import device, load_pt
from agenticgen.uniguide.prepare import load_model
from agenticgen.uniguide.random_streams import capture, equal


SOURCE = Path('/workspace/agenticgen_absolute_coordinate_20261029')
BASELINE = Path('/workspace/uniguide_runs/absolute-coordinate-4d4d12ef')
OUTPUT = Path('/workspace/uniguide_audits/validation-trim-821bf3c4')


def check_teacher(subject, old):
    teacher, condition = subject['teacher'], subject['condition']
    config = read(SOURCE / 'GEOMETRY_CONFIG.json')
    # Native instrumentation writes only into this separate engineering directory.
    config['root'] = str(OUTPUT / 'model_check')
    initial = initial_shard(config, teacher['target'], [0, 1])
    checkpoint = SOURCE / 'acceptance' / teacher['target'] / 'raw999_vjp_prefix/checkpoint_010.pt.gz'
    state = load_pt(checkpoint)
    model = load_model(config, teacher)
    current, conditioning, times = (device(state[key]) for key in ('current', 'cond', 'times'))
    pocket = device(initial['pocket_target'])
    center = initial['target_com'].to('cuda')
    with torch.no_grad():
        equis, invs = model.gen.get_pocket_encoding(
            pocket['coords'], pocket['atom_names'],
            pocket_atom_charges=pocket['charges'].argmax(-1),
            pocket_bond_types=pocket['bonds'].argmax(-1),
            pocket_res_types=pocket['res_names'], pocket_atom_mask=pocket['mask'],
        )
    before = capture()
    x = current['coords'].detach().clone().requires_grad_(True)
    prediction = model(
        dict(current, coords=x), pocket, times,
        cond_batch=conditioning if model.self_condition else None,
        pocket_equis=equis, pocket_invs=invs, training=False,
    )
    hooks = []

    def denied(gradient):
        raise AssertionError('Affinity head entered coordinate VJP')

    for head in prediction.get('affinity', {}).values():
        if torch.is_tensor(head) and head.requires_grad:
            hooks.append(head.register_hook(denied))
    decoded, _ = model._get_predictions(prediction)
    endpoint = decoded['coords'] * model.coord_scale + center[:, None]
    valid = current['mask'].bool()
    slots = old.match(endpoint, condition, valid)
    expected = old.loss(endpoint, condition, valid, slots)[0]
    objective = CoordinateObjective(condition)
    binding = objective.bind(endpoint, valid, slots)
    actual = objective.evaluate(endpoint, valid, binding)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), x, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
    assert torch.equal(expected, actual), 'Loss changed'
    assert torch.equal(expected_gradient, actual_gradient), 'Full native-coordinate VJP changed'
    assert torch.isfinite(actual_gradient).all() and actual_gradient.norm() > 1e-8
    assert equal(before, capture()), 'Prepared objective consumed RNG'
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    for hook in hooks:
        hook.remove()
    return dict(
        teacher_id=teacher['target'], source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=sha(checkpoint), condition_sha256=condition['condition_sha256'],
        loss_bitwise_equal=True, full_coordinate_vjp_bitwise_equal=True,
        maximum_gradient_difference=float((expected_gradient - actual_gradient).abs().max()),
        input_gradient_norm=float(actual_gradient.norm()), affinity_gradient_entered=False,
        sampling_rng_unchanged=True, integration_steps_executed=0, new_scientific_candidates=0,
    )


def main():
    baseline_path = BASELINE / 'agenticgen/src/agenticgen/uniguide/absolute_coordinate.py'
    spec = importlib.util.spec_from_file_location('agenticgen.uniguide._reference_coordinate', baseline_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    # Cooperate with every other GPU executor; never interrupt another task.
    with (Path('/workspace/agenticgen_uniguide_flowr_20261026') / '.executor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks = []
        for subject in read(SOURCE / 'SUBJECTS.json'):
            checks.append(check_teacher(subject, old))
            gc.collect()
            torch.cuda.empty_cache()
    result = dict(
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        baseline_source_sha256=sha(baseline_path), checks=checks,
        experiment_restarted=False, llm_calls=0,
    )
    write(OUTPUT / 'gpu_contract.json', result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
