"""Combine completed local-feature pilots without recounting reused controls."""
import argparse
from pathlib import Path

from evomolsteer.io import digest, read_json
from evomolsteer.storage.transactions import atomic_json
from .analysis import distribution, paired_cluster, write_csv
from .common import batch_path, groups_for, identity, read_batch
from .local_research import load


def source_identity(audit):
    # Reuse annotations document copying, not a change in numerical execution.
    # Keep every device/library/source/config field in the comparison.
    canonical = {k: v for k, v in audit.items() if k != 'reused_current_from'}
    if 'condition' in canonical:
        canonical['condition'] = {k: v for k, v in canonical['condition'].items() if k != 'reuse_current_from'}
    return identity(canonical)


def merge_condition(datasets, signatures, name, records, signature):
    if name in datasets:
        if signatures[name] != signature or datasets[name] != records:
            raise ValueError('Repeated condition has different results or execution evidence: ' + name)
        return
    datasets[name], signatures[name] = records, signature


def summarize(repo, inputs, output):
    repo, output = Path(repo), Path(output)
    datasets, signatures, sources, responses = {}, {}, [], {}
    common_ids = protocol = scope = None
    for manifest, work in inputs:
        manifest, work = repo / manifest, repo / work
        cfg, study, _ = load(repo, manifest)
        ids = [i for g in groups_for(cfg, study['conditions'][0]) for i in g['ids']]
        current_scope = study.get('comparison_scope', 'strict_historical_replay')
        if protocol is not None and (cfg['protocol_sha256'] != protocol or ids != common_ids or current_scope != scope):
            raise ValueError('Studies do not share the same ordered historical cohort and comparison scope')
        common_ids, protocol, scope = ids, cfg['protocol_sha256'], current_scope
        sources.append({'manifest': str(manifest.relative_to(repo)), 'sha256': digest(manifest),
            'work': str(work.relative_to(repo)), 'research_sha256': study['research_sha256'],
            'deployment': read_json(work / 'deployment.json')})
        result = read_json(work / 'analysis/summary.json')
        if result['research_sha256'] != study['research_sha256']:
            raise ValueError('Analysis belongs to a different study manifest')
        for condition in study['conditions']:
            records, artifacts = [], []
            if [i for g in groups_for(cfg, condition) for i in g['ids']] != common_ids:
                raise ValueError('Every reported condition must use the identical ordered historical cohort')
            for group in groups_for(cfg, condition):
                receipt = read_batch(work, cfg, condition, group)
                if receipt is None or receipt['status'] != 'complete':
                    raise ValueError('Cannot summarize an unfinished condition: ' + condition['id'])
                records.extend(receipt['records'])
                artifacts.append(receipt['artifacts'])
            audit = work / 'methods' / condition['id'] / 'source_audit.json'
            signature = identity({'protocol': protocol, 'groups': ids, 'artifacts': artifacts,
                'source_audit_identity': source_identity(read_json(audit)),
                'execution': {k: condition.get(k) for k in ('kind', 'program_sha256', 'reference_sha256', 'window')}})
            merge_condition(datasets, signatures, condition['id'], records, signature)
        for row in result['responses']:
            key = (row['condition'], row['group'])
            if key in responses and responses[key] != row:
                raise ValueError('Reused control has different measured response')
            responses[key] = row
    if not inputs:
        raise ValueError('At least one completed study is required')
    original = [dict(a, pic50=a['historical_pic50'], valid_connected=a['historical_valid_connected'])
        for a in cfg['attempts'] if a['id'] in set(common_ids)]
    datasets['matched_original_steer'] = original
    performance = [{'condition': name, **distribution(rows, len(common_ids), cfg['tail_threshold'])}
        for name, rows in datasets.items()]
    paired = [{'condition': name, 'reference': reference, **paired_cluster(rows, baseline, 10000, 42)}
        for name, rows in datasets.items() for reference, baseline in datasets.items()
        if name != reference]
    all_original = [dict(a, pic50=a['historical_pic50'], valid_connected=a['historical_valid_connected'])
        for a in cfg['attempts']]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'performance.csv', performance)
    write_csv(output / 'paired.csv', paired)
    write_csv(output / 'response.csv', list(responses.values()))
    summary = {'comparison_scope': scope, 'base_protocol_sha256': protocol,
        'unique_historical_path_count': len(common_ids), 'unique_current_inference_count': sum(len(v) for k, v in datasets.items() if k != 'matched_original_steer'),
        'cohort_ids_sha256': identity(common_ids), 'sources': sources, 'performance': performance,
        'paired': paired, 'responses': list(responses.values()),
        'frozen_high_score_threshold': cfg['tail_threshold'],
        'historical_global_separate_cohort': distribution(all_original, len(all_original), cfg['tail_threshold']),
        'limits': 'Reused controls are counted once. All current conditions share the same selected historical paths. This is not a random-start generalization test or a passed cross-GPU exact replay.'}
    atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--input', nargs=2, action='append', required=True, metavar=('MANIFEST', 'WORK'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.repo.resolve(), args.input, args.output)
    print({'conditions': len(result['performance']), 'unique_current_inferences': result['unique_current_inference_count'],
        'unique_historical_paths': result['unique_historical_path_count']})


if __name__ == '__main__':
    main()
