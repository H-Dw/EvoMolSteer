"""Retain small scientific reports and provenance before retiring raw outputs."""
import argparse
from pathlib import Path
import shutil
from evomolsteer.io import read_json, digest
from evomolsteer.generation.prototypes import write_json


FILES = ['terminal_report.json', 'candidate_metrics.csv', 'candidate_metrics.json',
         'batch_metrics.csv', 'posebusters_fast_checks.csv', 'execution_report.json',
         'regional_time_metrics.csv', 'regional_window_trends.csv', 'window/report.json']


def preserve(results, dataset, campaign, output):
    results, dataset, output = map(lambda p: Path(p).resolve(), [results, dataset, output])
    root = dataset/'results'/campaign
    if output.exists():
        raise FileExistsError('Use a fresh immutable report directory')
    if not all((results/p).is_file() for p in FILES):
        raise ValueError('Scientific evaluation and execution audit must be complete')
    cfg = read_json(root/'config.json')
    if read_json(root/'COMPLETE.json')['status'] != 'complete':
        raise ValueError('Inference incomplete')
    execution = read_json(results/'execution_report.json')
    if execution['outside_window_injection'] or not execution['no_particle_resampling']:
        raise ValueError('Execution contract failed')
    rows = read_json(results/'candidate_metrics.json')
    if len(rows) != cfg['experiment']['n']*len(cfg['experiment']['arms'].split(',')):
        raise ValueError('Failed candidates must also be preserved')
    copied = []
    extras=[p for p in ['coordinate_audit.json','coordinate_time_metrics.csv','coordinate_time_rates.csv','coordinate_injection_motion.csv','coordinate_dose_time.csv','guidance_response.json'] if (results/p).is_file()]
    for source, relative in [(results/p, p) for p in FILES+extras] + [
            (root/p, 'inference_config/'+p) for p in ['config.json', 'reward_program.json', 'COMPLETE.json']]:
        dest = output/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if digest(source) != digest(dest):
            raise ValueError('Report copy checksum mismatch')
        copied.append({'path': relative, 'bytes': dest.stat().st_size, 'sha256': digest(dest)})
    report = {'campaign': campaign, 'candidate_rows': len(rows), 'files': copied,
              'bytes': sum(r['bytes'] for r in copied), 'inference_commit': execution['code_commit'],
              'raw_trajectory_hashes': execution['sources'],
              'retention': 'Reports/config only; no structures or coordinate trajectories. Recreate retired samples using frozen source/input/checkpoint/seed. Duplicate caches and transport archives are disposable.'}
    write_json(output/'retention.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['results', 'dataset', 'campaign', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args(); report = preserve(a.results, a.dataset, a.campaign, a.output)
    print({key: report[key] for key in ['campaign', 'candidate_rows', 'bytes']})
