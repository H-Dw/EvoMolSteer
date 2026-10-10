"""Local deterministic preparation; no credentials or remote source uploads."""
import argparse
import copy
import gzip
import json
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.continuous.outcome_agents import run_tool

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/experiments/terminal_outcome15_20261010'
CFG = ROOT/'configs/experiments/terminal_outcome15_v1'
BASES = {
    'R26': ROOT/'docs/experiments/flowcompat_scale1000_20261009/completed_20261009_1836/controls_snapshot/inference_config/reward_program.json',
    'R11': ROOT/'configs/experiments/flowcompat30_v1/round11.json'}
REFERENCES = {'R26': 'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz',
              'R11': 'configs/experiments/flowcompat30_v1/branch_reference.json.gz'}


def plan(kind, mode='mean', base='R26', branch='instantaneous', budget=2, tail_weight=.5):
    out = DOC/'mining'/kind
    inputs = [str(ROOT/'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/config.json'),
              str(ROOT/'docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv'), str(ROOT/REFERENCES[base])]
    for batch in range(14):
        folder = ROOT/f'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050'
        inputs += [str(folder/f'single/batch_{batch:03d}/trajectory.npz'), str(folder/f'frame_batch_{batch:03d}.json')]
    arguments = ['--dataset', str(ROOT/'data/raw/ck2_clk3_lineage_20261003'), '--campaign', 'main1000_w050',
        '--metrics', inputs[1], '--baseline', inputs[2], '--output', str(out), '--mode', mode,
        '--branch-mode', branch, '--budget', str(budget), '--tail-weight', str(tail_weight),
        '--score-start', '0.0', '--score-end', '0.5']
    p = DOC/'tool_plans'/(kind+'.json')
    write_json(p, {'tool_id': 'terminal_outcome', 'arguments': arguments, 'input_files': inputs,
                  'output_files': [str(out/n) for n in ['evidence.json', 'manifest.json', 'reference.json.gz',
                    'ancestor_outcomes.parquet', 'batch_event_effects.parquet', 'event_support.csv', 'teacher_coverage.csv', 'trends.json']]})
    return p


def publish_reference(kind):
    source = DOC/'mining'/kind/'reference.json.gz'
    target = CFG/'references'/(kind+'.json.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def boundary_control(base, number):
    kind = base+'_instant_051'
    reference = publish_reference(kind)
    p = copy.deepcopy(read_json(BASES[base]))
    for key in ['flowcompat_binding', 'agent_binding', 'flowcompat_supplemental_binding']:
        p.pop(key, None)
    ref = json.loads(gzip.decompress(reference.read_bytes()))
    p.update(program_id=f'{base}_matched_state_boundary', round=number, seed=42,
        window=ref['window'], score_window=ref['score_window'], reference_sha256=digest(reference),
        generation_interface='flowcompat_v2', target_definition='instantaneous_boundary_only_control',
        control_origin={'parent_program_sha256': digest(BASES[base]), 'change': 'Last observed score event included; original first 50 teachers unchanged'})
    target = CFG/f'round{number:02d}_{base}.json'
    write_json(target, p)
    job = {'cohort': base, 'campaign': f'outcome_r{number:02d}_{base}', 'program': target.relative_to(ROOT).as_posix(),
           'reference': reference.relative_to(ROOT).as_posix(), 'n': 100, 'batches': [131, 132],
           'arms': 'gradient,unguided' if number == 1 else 'gradient'}
    write_json(CFG/f'round{number:02d}.jobs.json', {'round': number, 'seed': 42, 'jobs': [job]})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=['init', 'tool', 'publish', 'control'], required=True)
    p.add_argument('--kind'); p.add_argument('--mode', default='mean'); p.add_argument('--base', default='R26')
    p.add_argument('--branch', default='instantaneous'); p.add_argument('--budget', type=int, default=2)
    p.add_argument('--tail-weight', type=float, default=.5); p.add_argument('--round', type=int)
    a = p.parse_args()
    if a.action == 'init':
        write_json(DOC/'protocol.json', {'status': 'registered', 'maximum_rounds': 15, 'seed': 42,
            'score_window': [0., .5], 'state_boundary': 'Derived from final observed selection proposal clock',
            'development_batches': [131, 132], 'confirmation_batches': [133, 134, 135, 136],
            'priority': 'Decoded final predicted affinity first; validity and converged MMFF strain secondary',
            'primary': 'Paired all-slot affinity change and valid-affinity mean; report both',
            'tail_threshold': 8.258901977539063, 'discovery_sources': list(range(14)),
            'sequence': ['Matched state-boundary controls', 'Final label source', 'Matched ancestor geometry extension',
                         'Descendant distribution extension', 'One-axis calibration', 'Frozen independent confirmation'],
            'rollback': 'Keep R11/R26 controls; failed candidate does not overwrite retained parent',
            'retention': 'Per-round metrics, instructions, responses, receipts, config and evidence; retire own generated data before next round',
            'simulation_model': 'gpt-6-luna', 'limitations': 'One target, adaptive small development panel; historical Steer unpaired'})
        print(plan('mean'))
    elif a.action == 'tool':
        path = plan(a.kind, a.mode, a.base, a.branch, a.budget, a.tail_weight)
        run_tool(path, DOC/'tool_receipts'/(a.kind+'.json'))
    elif a.action == 'publish':
        print(publish_reference(a.kind))
    else:
        boundary_control(a.base, a.round)
