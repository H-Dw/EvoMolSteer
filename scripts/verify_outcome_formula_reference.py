import gzip
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evomolsteer.io import digest, read_json, write_json
from scripts.prepare_outcome_confirmation import runtime_program

cfg = ROOT/'configs/experiments/terminal_outcome15_v1'
doc = ROOT/'docs/experiments/terminal_outcome15_20261010'
lock = read_json(cfg/'confirmation.freeze.json')
reference_path = cfg/'references/family_tail025_shrink2_v1.json.gz'
import json
reference = json.loads(gzip.decompress(reference_path.read_bytes()))
errors = []
for frame in reference['frames']:
    for score, outcome in zip(frame['teacher_scores'], frame['teacher_outcomes'], strict=True):
        expected = outcome['alias_family_credit']['terminal_mean'] + .25*outcome['tail_regularization']['regularized_fraction']
        errors.append(abs(score-expected))
checks = {'runtime': {}, 'sources': {}, 'reference': digest(reference_path) == lock['reference_sha256']}
for number in (14, 15):
    path = cfg/f'round{number:02d}_R11_tail_shrink2_frozen.json'
    checks['runtime'][str(number)] = runtime_program(read_json(path)) == lock['runtime_program']
for name, sha in lock['source_sha256'].items():
    checks['sources'][name] = digest(ROOT/'src/evomolsteer/generation'/name) == sha
interpretation = read_json(doc/'validation_interpretation_TO14.json')
summary = read_json(doc/'round14/summary.json')
for item in interpretation['source_files']:
    if digest(ROOT/item['path']) != item['sha256']:
        raise ValueError('Analyst read-only source binding changed')
facts = interpretation['confirmed_observations']
metric_mappings = {'candidate_metrics_from_TO14_summary': ('R11_tail_shrink2_frozen', 'gradient'),
    'same_round_R11_051_metrics': ('R11_051', 'gradient'),
    'same_round_R26_051_metrics': ('R26_051', 'gradient'),
    'within_job_unguided_metrics': ('R11_tail_shrink2_frozen', 'unguided')}
checked_metrics = 0
for key, (cohort, arm) in metric_mappings.items():
    for field, value in facts[key].items():
        if summary['results'][cohort][arm][field] != value:
            raise ValueError('Analyst interpreted metric differs from actual result')
        checked_metrics += 1
for key in ('paired_candidate_vs_within_job_native', 'paired_candidate_vs_same_round_R11_051', 'paired_candidate_vs_same_round_R26_051'):
    value = facts[key]
    source = summary['paired_comparisons'][value['comparison_key']]
    if any(source[field] != v for field, v in value.items() if field != 'comparison_key'):
        raise ValueError('Analyst interpreted pair differs from actual result')
if not (all(checks['runtime'].values()) and all(checks['sources'].values()) and checks['reference'] and errors and max(errors) < 1e-12):
    raise ValueError('Reference utility or frozen implementation mismatch')
write_json(doc/'formula_correction_verification.json', {
    'teacher_records_checked': len(errors), 'maximum_teacher_utility_error': max(errors),
    'checks': checks, 'corrections': ['Explicit utility rather than mean-only prior', 'Sum of step RMS path budget',
        'Enumerated observational branch support', 'Included active flowcompat_v2 adapter'],
    'reward_document_sha256': digest(doc/'reward_formula.zh-CN.md'),
    'auditor_report_sha256': digest(doc/'formula_audit_luna.json'), 'verification_script_sha256': digest(__file__),
    'Analyst_TO14_metrics_checked': checked_metrics, 'Analyst_TO14_paired_groups_checked': 3,
    'Analyst_TO14_source_files_checked': len(interpretation['source_files']),
    'scope': 'Documentation correction only; frozen generation and reference unchanged; no efficacy inference'})
print({'teacher_records': len(errors), 'max_error': max(errors), 'checks': checks})
