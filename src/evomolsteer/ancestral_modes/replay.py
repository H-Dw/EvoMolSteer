"""Post-campaign exact-input replay; never redesign or update search state."""
import argparse
import copy
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json, write_json
from evomolsteer.partial_search.engine import Remote
from evomolsteer.local_motif.evaluation import paired, score_records
from .research import execute, cleanup_runtime, commit


def run(repo, config, source_round):
    repo = Path(repo).resolve(); master = read_json(config); study = repo / master['campaign_study']
    original_state = (study / 'state.json').read_bytes()
    source = study / source_round
    p = read_json(source / 'selected_program.json')
    bank = read_json(source / 'candidates/registered/teacher.json')
    source_results = repo / master['generated_root'] / source_round / source_round
    pairing = read_json(source_results / 'guidance_audit.json')['input_pairing']
    if pairing['seed'] != master['seed'] or pairing['initial_slots'] != list(range(master['screen_samples'])):
        raise ValueError('Replay requires the recorded full batch and seed, not a truncated sample')
    child = copy.deepcopy(master)
    child['campaign_study'] += '/mechanism_followup'
    child['config_dir'] += '/mechanism_followup'
    child['generated_root'] += '/mechanism_followup'
    child['remote_campaign'] += '/mechanism_followup'
    folder = repo / child['campaign_study']; folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'exact_replay.json'
    if done.exists(): return read_json(done)
    write_json(folder / 'protocol.json', {'purpose': 'Distinguish exact-input repeatability from new-prior transfer.',
        'source_round': source_round, 'source_input_pairing': pairing, 'redesign': False,
        'new_optimization_rounds': 0, 'additional_terminal_attempts': master['screen_samples'],
        'window': p['window'], 'guidance_program_changed': False, 'preserve_campaign_state': True})
    remote = Remote()
    try:
        result, sha = execute(repo, child, p, bank, 'D01_' + source_round + '_EXACT_REPLAY', remote,
                              batch=0, n=master['screen_samples'], include_unguided=False)
        target = result / ('D01_' + source_round + '_EXACT_REPLAY')
        comparison = paired(target, source_results, master.get('tau', read_json(study / 'state.json')['original']['tau']))
        a = read_json(source_results / 'records.json'); b = read_json(target / 'records.json')
        delta = np.asarray([y['pic50'] - x['pic50'] for x, y in zip(a, b)])
        top = max((r for r in a if r['qualified_structure']), key=lambda r: r['pic50'])
        top2 = max((r for r in b if r['qualified_structure']), key=lambda r: r['pic50'])
        report = {'source_round': source_round, 'inference_commit': sha, 'additional_attempts': len(b),
            'new_optimization_rounds': 0, 'n': len(b), 'paired_protocol': comparison,
            'max_abs_score_difference': float(np.abs(delta).max()),
            'mean_abs_score_difference': float(np.abs(delta).mean()),
            'scores_bitwise_equal': bool(np.array_equal(delta, np.zeros_like(delta))),
            'canonical_smiles_equal': sum(x['smiles'] == y['smiles'] for x, y in zip(a, b)),
            'structure_qualification_equal': sum(x['qualified_structure'] == y['qualified_structure'] for x, y in zip(a, b)),
            'source_best': top, 'replay_best': top2, 'result_folder': target.relative_to(repo).as_posix(),
            'scope': 'Exactly the same saved priors, B50, seed42, checkpoint and native protocol; current operative-equivalent controller. No new LLM design, SMC or affinity gradient.'}
        if (study / 'state.json').read_bytes() != original_state: raise ValueError('Search state unexpectedly changed')
        write_json(done, report); cleanup_runtime(remote, child)
        commit(repo, child, 'diagnostic(ancestral-v3): record exact-input R02 repeatability without redesign')
        return report
    finally: remote.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', default='.'); p.add_argument('--config', required=True)
    p.add_argument('--source-round', default='R02_v3'); a = p.parse_args()
    r = run(a.repo, a.config, a.source_round)
    print({k: r[k] for k in ('n', 'max_abs_score_difference', 'scores_bitwise_equal', 'source_best', 'replay_best')})


if __name__ == '__main__': main()
