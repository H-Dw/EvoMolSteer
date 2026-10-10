"""Remote inference and lossless export only; all scientific analysis is local."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from evomolsteer.io import digest, read_json, write_json


def run(manifest, work, flowr):
    repo = Path(__file__).resolve().parents[1]
    plan = read_json(manifest)
    work, flowr = Path(work), Path(flowr)
    number = plan['round']
    status = 1
    try:
        if not 1 <= number <= 15 or plan['seed'] != 42:
            raise ValueError('Fifteen-round task contract')
        for job in plan['jobs']:
            campaign = job['campaign']
            program, reference = repo/job['program'], repo/job['reference']
            p = read_json(program)
            if p['reference_sha256'] != digest(reference) or len(job['batches']) != job['n']//50:
                raise ValueError('Committed reference/population contract')
            before = work/(campaign+'.source.before.json')
            after = work/(campaign+'.source.after.json')
            source_args = ['--input-dataset', str(flowr)]
            for relative in ['flowr/models/fm_pocket.py', 'flowr/models/integrator.py', 'flowr/models/pocket.py', 'flowr/models/pocket_util.py']:
                source_args += ['--relative-source', relative]
            subprocess.check_call([sys.executable, 'scripts/attest_flowr_sources.py', *source_args, '--output-record', str(before)])
            subprocess.check_call([sys.executable, '-u', 'scripts/generate_flowcompat_v2_flowr.py',
                '--flowr-root', str(flowr), '--input-dataset', str(flowr/'experiments/ck2_clk3_lineage_20261003'),
                '--root', str(work/'generated'), '--checkpoint', str(flowr/'checkpoints/flowr_root_v2.ckpt'),
                '--steps', '100', '--program', str(program), '--reference', str(reference), '--campaign', campaign,
                '--n', str(job['n']), '--batch', '50', '--seed', '42', '--arms', job['arms'],
                '--batch-indices', ','.join(map(str, job['batches'])), '--export-terminal'])
            subprocess.check_call([sys.executable, 'scripts/attest_flowr_sources.py', *source_args,
                '--reference-record', str(before), '--output-record', str(after)])
            subprocess.check_call([sys.executable, 'scripts/bind_flowr_source_attestation.py', '--dataset', str(work/'generated'),
                '--campaign', campaign, '--before-record', str(before), '--after-record', str(after)])
            subprocess.check_call([sys.executable, 'scripts/archive_generation.py', '--dataset', str(work/'generated'),
                '--campaign', campaign, '--output', str(work/'archives'/(campaign+'.tar.gz')), '--evaluation-only'])
        status = 0
    finally:
        (work/f'round{number:02d}.exit').write_text(str(status)+'\n')
    return status


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['manifest', 'work', 'flowr-root']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    sys.exit(run(a.manifest, a.work, a.flowr_root))
