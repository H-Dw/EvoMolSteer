#!/usr/bin/env python3
"""Compare batched decoder/VJP with the original twenty-slot operation.

Read-only saved-state diagnostic, no integration, API calls or new candidates.
An identical RNG digest alone does not establish identical random draws or heads.
"""
import argparse
import gc
from pathlib import Path


def main():
    import torch
    from benchmark_plip_batches import join_blocks
    from evomolsteer.adaptive_schedule.flowr import load_model, load_initial, load_pt, device, detach, tensor_digest
    from evomolsteer.adaptive_schedule.io import read, write, commit
    from evomolsteer.adaptive_schedule.vendor.reward import LocalReward
    from evomolsteer.generation.rng import rng_state
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new audit directory')
    a.output.mkdir(parents=True)
    cfg = read(a.source_root/'config.json')
    teachers = [t for t in read(a.source_root/'teachers.json') if t['rank'] in (1, 2)]
    model = load_model(cfg, teachers[0]['target'], a.output)
    initials = [load_initial(cfg, teachers[0]['target'], b)[0] for b in range(5)]
    folders = []
    for b in range(5):
        h = read(a.source_root/'versions/v1'/teachers[0]['teacher_id']/f'batch_{b:03d}/histories.json')
        folders.append(a.source_root / h[0]['history'][0]['branch'])
    rows = []
    for step in [0, 50, 90]:
        states = [load_pt(f/f'checkpoint_{step:03d}.pt.gz') for f in folders]
        for teacher in teachers:
            reward = LocalReward(teacher, 'C', 'endpoint')
            def predict(blocks):
                state = {k: join_blocks([states[b][k] for b in blocks], torch) for k in ['current', 'cond']}
                curr, cond = device(state['current']), device(state['cond'])
                pocket = device(join_blocks([initials[b]['pocket_target'] for b in blocks], torch))
                com = device(join_blocks([initials[b]['target_com'] for b in blocks], torch))
                times = [torch.cat([states[b]['times'][j] for b in blocks]).to('cuda') for j in range(3)]
                before = tensor_digest(rng_state())
                with torch.no_grad():
                    equis, invs = model.gen.get_pocket_encoding(pocket['coords'], pocket['atom_names'],
                        pocket_atom_charges=pocket['charges'].argmax(-1), pocket_bond_types=pocket['bonds'].argmax(-1),
                        pocket_res_types=pocket['res_names'], pocket_atom_mask=pocket['mask'])
                with torch.enable_grad():
                    x = curr['coords'].detach().clone().requires_grad_(True)
                    out = model(dict(curr, coords=x), pocket, times, cond_batch=detach(cond) if model.self_condition else None,
                                pocket_equis=equis, pocket_invs=invs, training=False)
                    pred, _ = model._get_predictions(out)
                    loss, _, assignment, _ = reward.loss(x*model.coord_scale+com[:,None],
                        pred['coords']*model.coord_scale+com[:,None], out, curr['mask'].bool())
                    grad = torch.autograd.grad(-loss.sum(), x)[0]
                assert tensor_digest(rng_state()) == before
                return {**{k: out[k].detach().cpu() for k in ['coords', 'atomics', 'bonds', 'charges']},
                        'gradient': grad.detach().cpu(), 'assignment': assignment.cpu()}
            reference = [predict([b]) for b in range(5)]
            for size in [40, 60, 80, 100]:
                actual = predict(list(range(size//20)))
                expected = {k: torch.cat([r[k] for r in reference[:size//20]]) for k in actual}
                rows.append({'teacher': teacher['teacher_id'], 'step': step, 'physical_batch': size,
                    'bitwise_equal': {k: torch.equal(actual[k], expected[k]) for k in actual},
                    'max_abs_difference': {k: float((actual[k]-expected[k]).abs().max()) for k in actual}})
                write(a.output/'head_equivalence.json', rows)
            gc.collect()
            torch.cuda.empty_cache()
    write(a.output/'COMPLETE.json', {'commit': commit(), 'cases': len(rows), 'native_rng_consumed': False,
            'all_bitwise_equal': all(all(r['bitwise_equal'].values()) for r in rows),
            'scope': 'saved Top1 R0 states at 0/50/90, evaluated with both local teacher losses; no native transitions'})


if __name__ == '__main__':
    main()
