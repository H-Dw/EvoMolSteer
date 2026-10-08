"""Compare tracked inference algorithms across frozen experimental commits."""
import argparse
import hashlib
import subprocess
from pathlib import Path

from evomolsteer.io import write_json


DEFAULT_PATHS = [
    'src/evomolsteer/generation/controller.py',
    'src/evomolsteer/generation/endpoint_controller.py',
    'src/evomolsteer/generation/endpoint_reward.py',
    'src/evomolsteer/generation/affinity_geometry_reward.py',
    'src/evomolsteer/generation/dynamic_region_reward.py',
    'src/evomolsteer/generation/coordinate_contrast.py',
    'src/evomolsteer/generation/local_reward.py',
    'src/evomolsteer/generation/multistage_reward.py',
    'src/evomolsteer/continuous/affinity_geometry.py',
]


def audit(repo, commits, paths, output):
    repo = Path(repo).resolve(); hashes = {}
    for commit in commits:
        resolved = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', '--verify', commit+'^{commit}'], text=True).strip()
        hashes[resolved] = {}
        for path in paths:
            value = subprocess.check_output(['git', '-C', str(repo), 'show', resolved+':'+path])
            hashes[resolved][path] = hashlib.sha256(value).hexdigest()
    unchanged = {p: len({h[p] for h in hashes.values()}) == 1 for p in paths}
    result = {'schema_version': 'frozen-inference-code-audit-1.0', 'commits': hashes,
              'unchanged_by_module': unchanged, 'all_compared_algorithms_unchanged': all(unchanged.values()),
              'semantics': 'Exact committed source-byte comparison; references, program parameters and real execution still require separate validation.'}
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='.')
    p.add_argument('--commits', nargs='+', required=True)
    p.add_argument('--paths', nargs='+', default=DEFAULT_PATHS)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = audit(a.repo, a.commits, a.paths, a.output)
    print({'all_compared_algorithms_unchanged': result['all_compared_algorithms_unchanged']})
