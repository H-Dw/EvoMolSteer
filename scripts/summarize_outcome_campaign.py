"""Rebuild compact campaign comparisons from retained, verified round reports."""
import argparse
from pathlib import Path

import pandas as pd

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.generation.path_evaluation import paired_effect, summarize_tail

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'docs/experiments/terminal_outcome15_20261010'


def summarize(directory, output, allow_incomplete=False):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    protocol = read_json(directory / 'protocol.json')
    maximum = protocol['maximum_rounds']
    reports = []
    rows = []
    for number in range(1, maximum + 1):
        folder = directory / f'round{number:02d}'
        path = folder / 'summary.json'
        if not path.is_file():
            if allow_incomplete:
                break
            raise ValueError(f'Completed round {number} is missing')
        report = read_json(path)
        retained = read_json(folder / 'retention.json')
        if report['status'] != 'complete' or retained['status'] != 'complete':
            raise ValueError('Incomplete scientific report')
        for record in retained['files']:
            file = (folder / record['path']).resolve()
            if not file.is_relative_to(folder) or digest(file) != record['sha256']:
                raise ValueError('Retained report changed or escaped its round')
        reports.append({'round': number, 'summary_sha256': digest(path),
                        'retention_sha256': digest(folder / 'retention.json')})
        for cohort, arms in report['results'].items():
            for arm, metrics in arms.items():
                row = {'round': number, 'cohort': cohort, 'arm': arm, **metrics}
                for label in ('native', 'R11_051', 'R26_051', 'immediate_parent',
                              'within_job_native'):
                    value = report['paired_comparisons'].get(cohort + '/' + label)
                    if value is not None and arm == 'gradient':
                        row['paired_delta_' + label] = value['paired_mean_pic50']
                rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'round_metrics.csv', index=False)
    result = {'completed_rounds': len(reports), 'maximum_rounds': maximum,
              'protocol_sha256': digest(directory / 'protocol.json'), 'reports': reports,
              'development_warning': 'Repeated development particles are adaptive trials, not independent pooled samples',
              'historical_steer': read_json(directory / 'round01/summary.json')['historical_steer_unpaired']}
    if len(reports) == maximum:
        # The campaign reserves the final two rounds for a frozen new-batch panel.
        tables = {'candidate': [], 'native': [], 'R11_051': [], 'R26_051': []}
        references, runtime_sources = set(), []
        for number in (maximum - 1, maximum):
            folder = directory / f'round{number:02d}'
            report = read_json(folder / 'summary.json')
            candidate_names = [k for k in report['results'] if k not in ('R11_051', 'R26_051')]
            if len(candidate_names) != 1:
                raise ValueError('One frozen confirmation candidate required')
            candidate = candidate_names[0]
            table = pd.read_csv(folder / 'results' / candidate / 'candidate_metrics.csv')
            tables['candidate'].append(table[table.arm.eq('gradient')])
            tables['native'].append(table[table.arm.eq('unguided')])
            references.add(report['label_response'][candidate]['reference_sha256'])
            runtime_sources.append(read_json(folder / 'results' / candidate / 'execution_report.json')['preflight'])
            for label in ('R11_051', 'R26_051'):
                table = pd.read_csv(folder / 'results' / label / 'candidate_metrics.csv')
                tables[label].append(table[table.arm.eq('gradient')])
        if len(references) != 1:
            raise ValueError('Confirmation reference changed')
        joined = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
        expected = protocol['confirmation_batches']
        for table in joined.values():
            if sorted(table.batch.unique()) != expected or table.duplicated(['batch', 'slot']).any():
                raise ValueError('Confirmation batches overlap or differ from protocol')
        result['frozen_confirmation'] = {
            'reference_sha256': list(references)[0],
            'metrics': {k: summarize_tail(v, protocol['tail_threshold']) for k, v in joined.items()},
            'paired': {k: {**paired_effect(joined['candidate'], joined[k]),
                          'limitation': 'Four independent batch units at fixed global seed; conditional test for this pocket, not generalization'}
                       for k in ('native', 'R11_051', 'R26_051')},
            'runtime_preflight': runtime_sources}
    write_json(output / 'campaign_summary.json', result)
    print({'completed': len(reports), 'maximum': maximum, 'output': str(output)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', default=str(DEFAULT))
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    summarize(args.reports, args.output, args.allow_incomplete)
