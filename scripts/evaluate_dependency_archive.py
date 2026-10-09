"""Locally verify and evaluate a completed dependency-ablation transport.

Never accepts a reward library as the terminal diagnostic reference. Scientific
analysis stays local; remote generation archives are immutable and checksummed.
Existing successful reports can be reused only after verifying their sources.
"""
import argparse
from pathlib import Path
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.storage.generation_archive import verify_generation_archive
from evomolsteer.generation.terminal_evaluation import evaluate_terminal
from audit_dependency_execution import audit


def run(name, archive, metadata, restored, reports, output, reference, workers):
    reports.mkdir(parents=True, exist_ok=True)
    verification = reports / (name + '.archive_verification.json')
    if restored.exists():
        record = read_json(verification)
        if record['archive_sha256'] != digest(archive):
            raise ValueError('Previously restored archive changed')
    else:
        write_json(verification, verify_generation_archive(archive, metadata, restored))
    root = restored / 'results' / name
    write_json(reports / (name + '.execution_audit.json'), audit(root))
    manifest = read_json(root / 'EVALUATION_VIEW.json')
    arms = sorted({x['batch_path'].split('/')[0] for x in manifest['batches']})
    batches = sorted({int(x['batch_path'].split('_')[-1]) for x in manifest['batches']})
    provenance = {'archive_sha256': digest(archive), 'reference_sha256': digest(reference),
                  'evaluation_source_sha256': digest(Path(__file__).resolve().parents[1] /
                      'src/evomolsteer/generation/terminal_evaluation.py')}
    prior = output / 'dependency_evaluation_sources.json'
    if (output / 'terminal_report.json').exists():
        if read_json(prior) != provenance:
            raise ValueError('Existing evaluation sources differ; choose a new output')
    else:
        evaluate_terminal(restored, name, reference, output, arms, batches, workers)
        write_json(prior, provenance)
    report = read_json(output / 'terminal_report.json')
    write_json(reports / (name + '.terminal_summary.json'),
               {'sources': provenance, 'results': report['results']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--transport', required=True)
    p.add_argument('--reports', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args(); transport = Path(a.transport)
    run(a.name, transport / (a.name + '.tar.gz'), transport / (a.name + '.tar.gz.json'),
        transport / ('restored' + a.name), Path(a.reports), Path(a.output), Path(a.reference), a.workers)
