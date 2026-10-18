"""Independently trace elite terminal seats to exact saved initial tensors.

This reads existing data only. Initial-state equality, within-new-run noise
pairing, and historical path replay are deliberately separate audit claims.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import torch

from evomolsteer.agentic_steer.native import slice_tree
from evomolsteer.io import digest, read_json, write_table
from evomolsteer.storage.arrays import array_digest, byte_equal
from evomolsteer.storage.transactions import atomic_json
from .common import binding, validate_protocol


def parent_path(selected, final_slot):
    """Row k maps each post-selection slot to its pre-selection parent at k."""
    selected = np.asarray(selected)
    if selected.ndim != 2 or selected.dtype.kind not in 'iu':
        raise ValueError('Expected an integer steps-by-population ancestry array')
    if np.any(selected < 0) or np.any(selected >= selected.shape[1]):
        raise ValueError('Parent index outside population')
    if not 0 <= final_slot < selected.shape[1]:
        raise ValueError('Final slot outside population')
    before = np.empty(len(selected), dtype=np.int64)
    after = np.empty(len(selected), dtype=np.int64)
    slot = int(final_slot)
    for step in range(len(selected)-1, -1, -1):
        after[step] = slot
        slot = int(selected[step, slot])
        before[step] = slot
    return before, after


def tree_identity(value):
    """Portable exact tensor bytes, shapes and metadata; not pickle bytes."""
    def canonical(x):
        if torch.is_tensor(x):
            return {'tensor': array_digest(x.detach().cpu().numpy())}
        if isinstance(x, np.ndarray):
            return {'array': array_digest(x)}
        if isinstance(x, dict):
            return {'dict': {k: canonical(v) for k, v in sorted(x.items())}}
        if isinstance(x, (list, tuple)):
            return {type(x).__name__: [canonical(v) for v in x]}
        if x is None or isinstance(x, (str, bool, int, float)):
            return x
        raise TypeError('Unexpected input metadata: '+type(x).__name__)
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_state(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=False)


def source_method(source, method):
    text = Path(source).read_text(encoding='utf-8')
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == method)
    start = min([node.lineno]+[d.lineno for d in node.decorator_list])
    return textwrap.dedent(''.join(text.splitlines(keepends=True)[start-1:node.end_lineno]))


def run(repo, config_path, work, output, upstream=None):
    repo, work, output = Path(repo).resolve(), Path(work).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_json(config_path); validate_protocol(cfg)
    raw = repo/'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/single'
    ranked = [r for f in sorted(raw.glob('batch_*/final_records.json')) for r in read_json(f)]
    ranked.sort(key=lambda r: (-r['pic50_on_rescore'], r['batch'], r['slot']))
    if len(ranked) != 1000:
        raise ValueError('Incomplete ranking universe')
    for source, expected in cfg['ranking_sources'].items():
        if digest(repo/source) != expected:
            raise ValueError('Ranking source changed: '+source)
    batches, initials, roots, paths, clone_examples = {}, {}, [], [], []
    for attempt, historical in zip(cfg['attempts'], ranked[:len(cfg['attempts'])], strict=True):
        batch, slot = attempt['batch'], attempt['final_slot']
        if (batch, slot, attempt['historical_pic50']) != (
                historical['batch'], historical['slot'], historical['pic50_on_rescore']):
            raise ValueError('Elite ranking identity differs')
        if batch not in batches:
            folder = raw/f'batch_{batch:03d}'
            with np.load(folder/'trajectory.npz', allow_pickle=False) as z:
                tr = {k: z[k] for k in ['selected_indices', 'root_slot', 'resampled',
                    'current_coords', 'proposal_coords', 'current_atomics', 'proposal_atomics',
                    'current_bonds', 'proposal_bonds', 'current_charges', 'proposal_charges', 'mask']}
            init = load_state(folder/'initial_state.pt.gz')
            restart = load_state(folder/'restart_step_000.pt.gz')
            if tree_identity(init['prior']) != tree_identity(restart['current']):
                raise ValueError('Saved prior differs from actual step-zero current')
            for field in ['coords', 'atomics', 'bonds', 'charges', 'mask']:
                start = init['prior'][field].numpy()
                if field in ['atomics', 'bonds', 'charges']: start = start.argmax(-1).astype(np.uint8)
                if field == 'mask': start = start.astype(np.uint8)
                name = 'mask' if field == 'mask' else 'current_'+field
                if not byte_equal(start, tr[name][0]):
                    raise ValueError('Trajectory t=0 differs from saved prior: '+field)
            for field in ['coords', 'atomics', 'bonds', 'charges']:
                for k, ids in enumerate(tr['selected_indices'][:-1]):
                    if not byte_equal(tr['current_'+field][k+1], tr['proposal_'+field][k][ids]):
                        raise ValueError('Broken saved parent link: '+field)
            batches[batch] = init, tr
        init, tr = batches[batch]
        before, after = parent_path(tr['selected_indices'], slot)
        root = int(before[0])
        if root != attempt['root_slot'] or root != historical['root_slot']:
            raise ValueError('Independently traced root differs')
        if not np.all(tr['root_slot'][np.arange(len(before)), before] == root):
            raise ValueError('Forward root labels differ from reverse ancestry')
        expected_cohorts = [k for k, s in cfg['cohorts'].items() if attempt['rank'] <= s['n']]
        if attempt['cohorts'] != expected_cohorts:
            raise ValueError('Cohort has non-elite seats')
        key = attempt['initial_id']; spec = cfg['initials'][key]
        if spec['batch'] != batch or spec['root_slot'] != root:
            raise ValueError('Initial input lookup differs from true ancestor')
        if key not in initials:
            source = repo/spec['source']; exported = work/spec['file']
            if digest(source) != spec['source_sha256'] or digest(exported) != spec['sha256']:
                raise ValueError('Source/exported initial file checksum differs')
            exported_state = load_state(exported)
            expected_state = {k: slice_tree(init[k], root, len(init['prior']['coords']), 'cpu')
                              for k in ['prior', 'pocket_target']}
            expected_state.update(target_com=np.asarray(init['target_com'])[root:root+1], coord_scale=init['coord_scale'])
            if tree_identity(exported_state) != tree_identity(expected_state):
                raise ValueError('Exported initial tensors or pocket differ from ancestor')
            initials[key] = tree_identity(exported_state)
            roots.append({'initial_id':key, 'batch':batch, 'root_slot':root,
                'exported_sha256':spec['sha256'], 'state_identity':initials[key],
                **{f'{k}_sha256':array_digest(exported_state['prior'][k].numpy())
                   for k in ['coords','atomics','bonds','charges','mask']},
                'pocket_sha256':tree_identity(exported_state['pocket_target']),
                'historical_rng_available': 'rng' in init,
                'historical_rng_in_export': 'rng' in exported_state, 'exact_tensor_match':True})
        paths.append({**attempt, 'cohorts':','.join(attempt['cohorts']),
            'root_independently_traced':root, 'initial_tensor_match':True,
            'input_state_identity':initials[key],
            'changed_slot_transitions':int(np.sum(before != after))})
        if attempt['rank'] == 1:
            lineage = []
            for k, parent in enumerate(before):
                children = np.flatnonzero(tr['selected_indices'][k] == parent)
                lineage.append({'step':k,'score_time':k/100, 'state_after_time':(k+1)/100,
                    'parent_slot':int(parent),'descendant_slot':int(after[k]),
                    'offspring_of_this_parent':len(children),'resampled':bool(tr['resampled'][k]),
                    'same_root_population_before':int(np.sum(tr['root_slot'][k] == root))})
                if len(children)>1 and k+1<len(before):
                    current = tr['current_coords'][k+1,children]
                    proposals = tr['proposal_coords'][k+1,children]
                    rms = np.sqrt(np.square(proposals-proposals[0]).sum(-1).mean(-1))*init['coord_scale']
                    clone_examples.append({'parent_step':k,'parent_slot':int(parent),
                        'child_slots':children.tolist(),'copied_coordinates_identical':bool(np.all(current == current[0])),
                        'next_proposal_max_rms_A':float(rms.max())})
            write_table(output/'best_original_lineage.csv', pd.DataFrame(lineage))
            atomic_json(output/'best_original_clone_examples.json',clone_examples)
    collection = read_json(repo/'docs/experiments'/cfg['name']/'latest_collection.json')
    receipt_paths = collection['snapshot']['receipt_paths']
    attempts = {a['id']:a for a in cfg['attempts']}
    conditions = {c['id']:c for c in cfg['conditions']}
    receipts, evidence, native_hash = [], [], None
    if upstream:
        native_hash = hashlib.sha256(source_method(upstream, '_generate').encode()).hexdigest()
    for name in receipt_paths:
        record = read_json(work/name)
        condition = conditions[record['condition']]; attempt = attempts[record['attempt']['id']]
        if record['attempt'] != attempt or record['binding'] != binding(cfg, condition, attempt):
            raise ValueError('Result does not use the registered elite attempt')
        if record['status'] != 'complete':continue
        folder = (work/name).parent
        for artifact, expected in record['artifacts'].items():
            if digest(folder/artifact) != expected: raise ValueError('Changed artifact: '+artifact)
        guidance = read_json(folder/'guidance.json');spec = cfg['initials'][attempt['initial_id']]
        if guidance['initial_sha256'] != spec['sha256'] or guidance['suffix_seed'] != attempt['suffix_seed']:
            raise ValueError('Receipt initial/noise pairing mismatch')
        receipts.append({**attempt, 'cohorts':','.join(attempt['cohorts']),
            'condition':condition['id'],'input_sha256':guidance['initial_sha256'],
            'registered_input_exact_match':True,'new_suffix_paired':True,'historical_noise_replay':False,
            'guidance_sha256':record['artifacts']['guidance.json']})
    for cid, condition in conditions.items():
        audit_path = work/'methods'/cid/'source_audit.json'
        audit = read_json(audit_path) if audit_path.exists() else None
        if audit and native_hash and audit['native_source_sha256'] != native_hash:
            raise ValueError('Loaded upstream generation source differs: '+cid)
        selected = [r for r in receipts if r['condition']==cid]
        evidence.append({'condition':cid,'complete':len(selected),
            **{k+'_complete':sum(r['rank']<=s['n'] for r in selected) for k,s in cfg['cohorts'].items()},
            'registered_initials_exact':True,'completed_receipts_match':True if selected else None,
            'runtime_source_hash_checked':bool(audit and native_hash),
            'historical_noise_replayed':False,'extra_random_initials':0})
    for name, rows in [('initial_roots',roots),('elite_attempts',paths),('receipt_inputs',receipts),('conditions',evidence)]:
        write_table(output/(name+'.csv'), pd.DataFrame(rows))
    result = {'protocol_sha256':cfg['protocol_sha256'],'snapshot_sha256':collection['snapshot']['sha256'],
        'collected_unix':collection['collected_unix'],'n_ranked':len(ranked),'n_elite_seats':len(paths),
        'n_distinct_initial_roots':len(roots),'n_verified_completed_receipts':len(receipts),
        'cohorts':cfg['cohorts'],'all_exported_tensors_exact':True,'all_reverse_ancestry_matches':True,
        'source_native_sha256':native_hash,'source_hashes':{k:digest(Path(__file__).with_name(k))
            for k in ['worker.py','prepare.py','common.py']},
        'entry_state_evidence':'Exact exported tensors + registered load code/source + per-result input-file SHA. No separate per-attempt pre-loop tensor dump was saved.',
        'historical_path_replay':False,'new_model_calls':0,'new_inference':False,
        'cohort_reuse':'One 200-seat union per method; top05/top10 are exact nested 50/100-seat views, never additional random starts.'}
    atomic_json(output/'audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.')
    p.add_argument('--config',required=True)
    p.add_argument('--work',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--upstream',help='Retrieved remote fm_pocket.py; verifies completed workers used the inspected source')
    a=p.parse_args(); print(json.dumps(run(a.repo,a.config,a.work,a.output,a.upstream),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
