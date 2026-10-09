"""Read-only audit of a running or finished target campaign, including archives.

Uses one forward tar stream for results after byte-verifying each archive. No
extraction, inference, progress reset, source retirement or energy evaluation.
"""
import argparse
from collections import Counter
import datetime
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

from evomolsteer.storage.target_archive import verify_target_archive
from evomolsteer.generation.steer_launcher import verify_dataset


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def statistics_of(values):
    values = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return {'n': 0}
    def quantile(p):
        at = (len(values) - 1) * p
        lo = math.floor(at); hi = math.ceil(at)
        return values[lo] + (values[hi] - values[lo]) * (at - lo)
    return {'n': len(values), 'mean': statistics.fmean(values), 'median': statistics.median(values),
            'minimum': values[0], 'maximum': values[-1], 'p10': quantile(.1), 'p90': quantile(.9)}


def summarize_records(target_id, rows):
    built = [r for r in rows if r.get('build_success')]
    usable = [r for r in built if r.get('smiles') and not r.get('failure_reason')]
    connected = [r for r in usable if r.get('connected') is True]
    geometry = [r for r in usable if isinstance(r.get('pairs_below_1_2A'), (int, float))]
    return {'target_id': target_id, 'slots': len(rows), 'built': len(built),
            'build_failed': len(rows) - len(built), 'usable_records': len(usable),
            'connected_usable': len(connected),
            'unique_built_smiles': len({r['smiles'] for r in built if r.get('smiles')}),
            'geometry_records': len(geometry),
            'records_with_pairs_below_1_2A': sum(r['pairs_below_1_2A'] > 0 for r in geometry),
            'affinity_all_slots_upstream': statistics_of([r.get('pic50_on_upstream') for r in rows]),
            'affinity_usable_upstream': statistics_of([r.get('pic50_on_upstream') for r in usable]),
            'affinity_usable_rescore': statistics_of([r.get('pic50_on_rescore') for r in usable]),
            'qed_usable': statistics_of([r.get('qed') for r in usable]),
            'energy_status': 'not_evaluated_by_generation_pipeline'}


def documents(root, state, wanted, *, archive=None):
    if archive is None:
        for key in wanted:
            record = state['targets'][wanted[key]]
            folder = root / record['dataset']
            for path in folder.rglob('*.json'):
                relative = path.relative_to(folder).as_posix()
                if interesting(relative, state['config']['campaign']):
                    yield key, relative, read(path)
        return
    with tarfile.open(root / archive['path'], 'r|gz') as stream:
        for member in stream:
            parts = member.name.split('/', 2)
            if len(parts) != 3 or parts[0] != 'targets' or parts[1] not in wanted:
                continue
            if interesting(parts[2], state['config']['campaign']):
                if not member.isfile() or member.size > 8 * 1024 * 1024:
                    raise ValueError('Invalid result member: ' + member.name)
                yield parts[1], parts[2], json.load(stream.extractfile(member))


def interesting(name, campaign):
    return (name == 'learning_dataset_manifest.json'
            or name in ('results/' + campaign + '/runtime.json', 'results/' + campaign + '/final_records.json')
            or name.endswith('/learning_manifest.json') or name.endswith('/events.json'))


def audit(root):
    root = Path(root).resolve()
    state = read(root / 'campaign_state.json'); cfg = state['config']
    report = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'output': str(root), 'campaign_status': state['status'],
              'scope': 'snapshot; campaign may continue while audit reads completed immutable targets',
              'counts': dict(Counter(r['state'] for r in state['targets'].values())),
              'archives': [], 'targets': [], 'failures': [], 'issues': []}
    row_by_id = {row['target_id']: row for row in state['catalog']}
    for identifier, record in state['targets'].items():
        if record['state'] != 'failed':
            continue
        error_path = root / 'generation_progress' / (row_by_id[identifier]['key'] + '.error')
        text = error_path.read_text(encoding='utf-8', errors='replace') if error_path.exists() else ''
        oom = re.findall(r'^.*(?:OutOfMemoryError|out of memory).*$' , text, re.M | re.I)
        report['failures'].append({'target_id': identifier, 'type': 'OOM' if oom else 'other',
            'error_file': str(error_path), 'native_oom_lines': oom,
            'failure_in_pocket_encoding': 'get_pocket_encoding' in text})
    group_documents = []
    for archive in state['archives']:
        print(json.dumps({'event': 'verifying_archive', 'path': archive['path']}), flush=True)
        verification = verify_target_archive(root / archive['path'], archive['metadata'])
        expected_ids = set(archive['target_ids'])
        actual_ids = {r['target_id'] for r in verification['manifest']['targets'].values()}
        if actual_ids != expected_ids:
            raise ValueError('Archived target IDs differ from campaign state')
        remaining = [path for path in archive['directories'].values() if (root / path).exists()]
        report['archives'].append({'path': archive['path'], **{k:v for k,v in verification.items() if k != 'manifest'},
                                  'remaining_raw_directories': remaining, 'retirement': archive['retirement']})
        group_documents += list(documents(root, state, {k: r['target_id'] for k,r in verification['manifest']['targets'].items()}, archive=archive))
        print(json.dumps({'event': 'archive_verified', 'path': archive['path'], 'files': verification['files']}), flush=True)
    raw_ids = {row_by_id[k]['key']: k for k,r in state['targets'].items() if r['state'] == 'complete'}
    for identifier in raw_ids.values():
        record = state['targets'][identifier]
        verification = verify_dataset(root / record['dataset'], cfg['campaign'], arms=('single',), samples=cfg['samples'])
        print(json.dumps({'event': 'raw_target_verified', 'target_id': identifier, 'batches': len(verification['batches'])}), flush=True)
    group_documents += list(documents(root, state, raw_ids))
    grouped = {}
    for key, name, value in group_documents:
        if name in grouped.setdefault(key, {}):
            raise ValueError('Duplicate target evidence: ' + key + '/' + name)
        grouped[key][name] = value
    for identifier, record in state['targets'].items():
        if record['state'] not in ('complete', 'archived'):
            continue
        key = row_by_id[identifier]['key']; docs = grouped.get(key, {})
        prefix = 'results/' + cfg['campaign'] + '/'
        rows = docs[prefix + 'final_records.json']; manifest = docs['learning_dataset_manifest.json']
        runtime = docs[prefix + 'runtime.json']
        batches = [v for k,v in docs.items() if k.endswith('/learning_manifest.json')]
        events = [v for k,v in docs.items() if k.endswith('/events.json')]
        summary = summarize_records(identifier, rows)
        summary.update(batch_count=len(batches), window_events_per_batch=sorted({m['window_events'] for m in batches}),
                       steps_per_batch=sorted({m['total_integration_steps'] for m in batches}),
                       particles_per_batch=sorted({m['particles'] for m in batches}),
                       actual_selected_events=sum(sum(bool(e['resampled']) for e in es) for es in events),
                       checkpoint_sha256=manifest['checkpoint_sha256'], native_device=runtime['device'],
                       model_code_commit=runtime.get('evomolsteer_commit'))
        if len(rows) != cfg['samples'] or len(batches) != cfg['samples'] // cfg['batch_size']:
            raise ValueError('Incomplete target slots/batches: ' + identifier)
        if manifest['state'] != 'complete' or manifest['verification'].get('verified') is not True:
            raise ValueError('Target has no completed native verification: ' + identifier)
        if runtime.get('gradient_guidance') is not False or not (runtime.get('hip') or runtime.get('cuda')):
            raise ValueError('Target lacks expected native GPU Steer mode: ' + identifier)
        if any(m['particles'] != cfg['batch_size'] or m['total_integration_steps'] != cfg['steps'] for m in batches):
            raise ValueError('Target particle/step scope differs: ' + identifier)
        for m in batches:
            expected = set(m['selection_steps'])
            if m['window'] != [cfg['window_start'], cfg['window_end']] or m['state'] != 'complete':
                raise ValueError('Incomplete or inconsistent scoring window: ' + identifier)
            event_name = next(k for k,v in docs.items() if v is m).replace('learning_manifest.json', 'events.json')
            es = docs[event_name]
            if len(es) != cfg['steps'] or {e['step'] for e in es if e['resampled']} != expected:
                raise ValueError('Actual resampling schedule differs: ' + identifier)
        for field, native in [('attempted_slots','slots'), ('built_slots','built'), ('unique_built_smiles','unique_built_smiles')]:
            if record['summary'][field] != summary[native]:
                raise ValueError('Campaign summary differs from native records: ' + identifier)
        if not (root / 'generation_progress' / (key + '.finished')).exists():
            raise ValueError('Completed target has no finished marker: ' + identifier)
        report['targets'].append(summary)
    report['totals'] = {k: sum(r[k] for r in report['targets']) for k in
        ('slots','built','build_failed','usable_records','connected_usable','unique_built_smiles',
         'geometry_records','records_with_pairs_below_1_2A','actual_selected_events')}
    report['totals'].update(successful_targets=len(report['targets']),
        failed_targets=len(report['failures']), failure_types=dict(Counter(r['type'] for r in report['failures'])),
        archive_bytes=sum(a['archive_bytes'] for a in report['archives']),
        archived_source_bytes=sum(a['source_bytes'] for a in report['archives']))
    report['unique_count_scope'] = 'sum of target-specific unique SMILES counts; neither global deduplication nor unique 3D conformations'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    print('AUDIT_REPORT_BEGIN', flush=True)
    result = audit(args.dataset)
    print('AUDIT_JSON_BEGIN', flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
