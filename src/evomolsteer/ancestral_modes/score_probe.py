"""Terminal-only FLOWR affinity counterfactuals; never generate or differentiate."""
import argparse
import gzip
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.final_guidance.native import tensor_tree


def run(repo, config, flowr_root):
    sys.path.insert(0, str(Path(flowr_root).resolve()))
    from flowr.gen.generate_from_pdb_selective import get_args
    from flowr.scriptutil import load_model
    repo = Path(repo); cfg = read_json(config); out = Path(cfg['remote_output']); out.mkdir(parents=True, exist_ok=True)
    inputs = repo / cfg['inputs']; data = np.load(inputs, allow_pickle=False)
    if digest(inputs) != cfg['inputs_sha256']: raise ValueError('Counterfactual input payload altered')
    if digest(cfg['checkpoint']) != cfg['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    n = len(data['A_coords_A']); slot = cfg['focal_slot']
    oldargv = sys.argv
    sys.argv = ['score-probe', '--arch', 'pocket', '--pocket_type', 'holo', '--gpus', '1', '--num_workers', '0',
                '--batch_cost', str(n), '--sample_n_molecules_per_target', str(n), '--ckpt_path', cfg['checkpoint'],
                '--save_dir', str(out), '--integration_steps', '100', '--no_cat_noise_euler_guard', '--no_ligand_valence_repair']
    try: args = get_args()
    finally: sys.argv = oldargv
    args.seed = cfg['seed']; torch.set_float32_matmul_precision('high')
    model = load_model(args)[0].to('cuda').eval(); model.requires_grad_(False)
    if model.add_feats: raise ValueError('Hybridization inputs were not preserved; cannot reconstruct this checkpoint')
    with gzip.open(cfg['initial_state'], 'rb') as f: init = torch.load(f, map_location='cpu', weights_only=False)
    if digest(cfg['initial_state']) != cfg['initial_sha256']: raise ValueError('Initial pocket context differs')
    b = len(init['prior']['coords']); device = torch.device('cuda')
    pocket = tensor_tree(init['pocket_target'], n, b, device)
    com = torch.as_tensor(np.asarray(init['target_com'])[:n].reshape(n, 3), device=device, dtype=torch.float32)
    pocket['complex'] = [SimpleNamespace(com=c.cpu().numpy()) for c in com]
    output = []
    for name, base, swapped_bonds in [('A_coords_A_bonds', 'A', None), ('B_coords_B_bonds', 'B', None),
                                      ('A_coords_B_bonds', 'A', 'B'), ('B_coords_A_bonds', 'B', 'A')]:
        final = {'coords': (torch.as_tensor(data[base+'_coords_A'], device=device)-com[:, None])/model.coord_scale,
                 'mask': torch.as_tensor(data[base+'_mask'], device=device)}
        for key in ('atomics', 'bonds', 'charges'):
            values = data[base+'_'+key].copy()
            if key == 'bonds' and swapped_bonds: values[slot] = data[swapped_bonds+'_'+key][slot]
            labels = torch.as_tensor(values.astype(np.int64), device=device)
            final[key] = torch.nn.functional.one_hot(labels, init['prior'][key].shape[-1]).float()
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
        with torch.no_grad():
            scores = model._predict_affinity(final, final, pocket,
                [torch.full((n,), .9999, device=device)]*3)['affinity']['pic50'].reshape(-1).cpu().tolist()
        output.append({'variant': name, 'focal_pic50': scores[slot], 'all_scores': scores})
    result = {'rows': output, 'focal_slot': slot, 'n_context': n, 'checkpoint_sha256': cfg['checkpoint_sha256'],
              'inputs_sha256': digest(inputs), 'score_method': 'Same native hard-category terminal _predict_affinity at t=.9999; two self-conditioning forward passes.',
              'new_generated_molecules': 0, 'affinity_gradients': False,
              'scope': 'Factorial coordinate/bond-representation scoring on existing endpoints; not experimental affinity or a physical energy intervention.'}
    write_json(out/'results.json', result); return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', required=True); p.add_argument('--config', required=True)
    p.add_argument('--flowr-root', required=True); a = p.parse_args()
    r = run(a.repo, a.config, a.flowr_root)
    print({v['variant']: v['focal_pic50'] for v in r['rows']}, flush=True)


if __name__ == '__main__': main()
