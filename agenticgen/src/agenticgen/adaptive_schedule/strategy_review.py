"""Small best-round strategy review; performance is computed, judgment is authored."""
from pathlib import Path
import re

from .io import read, sha, commit
from .selection import choose

MODE = 'best_round_strategy'
AUTHORIZATION = 'agenticgen/docs/authorizations/20261023_best_round_strategy_review.json'


def authorization(path=AUTHORIZATION, digest=None):
    from .group_review import committed_file
    if path != AUTHORIZATION or not digest:
        raise ValueError('Focused review requires the explicit committed user authorization')
    doc = read(committed_file(path, digest, commit()))
    if (doc.get('authority') != 'human_user' or doc.get('review_mode') != MODE
            or doc.get('full_review_prohibited') is not True
            or doc.get('first_next_version') != 'v8' or doc.get('last_version') != 'v30'):
        raise ValueError('Focused review authorization scope differs')
    return {'path': path, 'sha256': digest}


def best_rounds(report, tolerance):
    """Rank shared schedules only. Never substitute per-chain selected archives."""
    teachers = sorted(report['representatives'])
    result = {}
    for teacher in teachers:
        rows = [r for r in report['rows'] if r['teacher'] == teacher
                and re.fullmatch(r'G[1-4]_unselected', r['collection'])]
        ranked = choose([{'id': f"r{r['round']:02d}:{r['collection']}",
                          'condition': r['condition_rmsd_A_all_attempts']['mean'],
                          'functional_gap': r['mean_affinity_absolute_gap'],
                          'best': r['affinity']['max'], 'ticks': r['ticks']}
                         for r in rows], tolerance)
        if ranked['status'] != 'complete':
            raise ValueError('Complete shared-schedule performance required')
        winner = ranked['selected']; round_id, group = winner['id'].split(':')
        expected = report['representatives'][teacher]['primary']['selected']
        if winner != expected:
            raise ValueError('Report representative does not match frozen ranking')
        result[teacher] = {'round': int(round_id[1:]), 'group': group.removesuffix('_unselected'),
                           'selected': winner, 'minimum_condition_A': ranked['minimum_condition_A'],
                           'band_upper_A': ranked['minimum_condition_A'] + tolerance}
    if not result:
        raise ValueError('No teacher performance')
    return result


def performance_context(root, version):
    """Read aggregate performance, not every controller conversation."""
    from .protocol_state import load_config
    root = Path(root); base = root/'versions'/version
    finalized = read(root/f'FINALIZED_{version}.json'); path = base/'report.json'
    if finalized['report_sha256'] != sha(path):
        raise ValueError('Finalized performance report changed')
    config = load_config(root)
    tolerance = config['condition_comparison']['ranking_tolerance_A']
    report = read(path); selected = best_rounds(report, tolerance)
    if set(selected) != {t['teacher_id'] for t in read(root/'teachers.json')}:
        raise ValueError('Teacher performance incomplete')
    return {'version': version, 'report_sha256': sha(path), 'teachers': selected,
            'round_performance': [
                {k: row[k] for k in ('teacher', 'round', 'collection', 'ticks',
                 'condition_rmsd_A_all_attempts', 'affinity', 'mean_affinity_absolute_gap')}
                for row in report['rows'] if re.fullmatch(r'G[1-4]_unselected', row['collection'])]}


def check(root, previous, next_version, approval):
    from .group_review import committed_file
    from .protocol_state import prepared_path
    root = Path(root); policy = read(prepared_path(root)).get('master_review_policy', {})
    if not 8 <= int(next_version[1:]) <= 30 or policy.get('mode') != MODE:
        raise ValueError('Best-round review policy has not been adopted')
    auth = authorization(approval.get('authorization_document'), approval.get('authorization_document_sha256'))
    if policy.get('authorization') != auth:
        raise ValueError('Approval authorization differs from the active review policy')
    if (approval.get('review_completed') is not True or approval.get('full_review_completed') is not False
            or approval.get('coverage')):
        raise ValueError('Focused review must not claim full-session coverage')
    document = committed_file(approval['strategy_review'], approval['strategy_review_sha256'], commit())
    review = read(document); context = performance_context(root, previous)
    if (review.get('mode') != MODE or review.get('version') != previous
            or review.get('report_sha256') != context['report_sha256']
            or review.get('full_review_completed') is not False):
        raise ValueError('Strategy review does not bind current performance and limited scope')
    actual = {}
    for item in review.get('teachers', []):
        teacher = item['teacher']; target = context['teachers'].get(teacher)
        if teacher in actual or target is None or item.get('best_round') != target['round'] or item.get('best_group') != target['group']:
            raise ValueError('Review must inspect exactly the selected best round for each teacher')
        for key in ('hypothesis_assessment', 'first_principles', 'performance_interpretation', 'next_direction'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError('Authored strategy assessment is required: '+key)
        evidence = item.get('evidence_files', {})
        prefix = f"versions/{previous}/{teacher}/round_{target['round']:02d}/council/"
        required = {prefix+'synthesis.json', prefix+f"analysts/{target['group']}/hypotheses.json"}
        if not required <= set(evidence):
            raise ValueError('Best-round hypotheses and council decision must be inspected')
        for name, digest in evidence.items():
            p = (root/name).resolve()
            if not name.startswith(prefix) or not p.is_relative_to(root.resolve()) or sha(p) != digest:
                raise ValueError('Selected best-round evidence changed or escaped scope')
        actual[teacher] = item
    if set(actual) != set(context['teachers']):
        raise ValueError('Missing teacher strategy review')
    hypotheses = review.get('capability_hypotheses', [])
    if not hypotheses or len(hypotheses) > 3 or any(
            any(not isinstance(h.get(k), str) or not h[k].strip()
                for k in ('blind_spot', 'strategy', 'expected_behavior', 'next_version_test')) for h in hypotheses):
        raise ValueError('One to three testable capability hypotheses required')
    from .version_gate import check_control_gate
    check_control_gate(root, next_version, approval)
    return {'approved': True, 'previous': previous, 'next': next_version, 'mode': MODE,
            'teachers': len(actual), 'sessions': 2*len(actual), 'full_review_completed': False,
            'scope': 'winning-group hypothesis and council strategy in each teacher best round'}
