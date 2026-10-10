"""Prepare explicit SFTP inputs, or verify all received campaign archives.

No credentials, network calls, remote commands or deletion are performed here.
The transport plan can be passed to the authenticated remote session interface.
"""
import argparse
from pathlib import Path, PurePosixPath
import shutil

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.storage.generation_archive import verify_generation_archive

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/experiments/terminal_outcome15_20261010'
CFG = ROOT/'configs/experiments/terminal_outcome15_v1'
DEFAULT_REMOTE = '/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_terminal_outcome15_20261010'


def collect(number, action, remote=DEFAULT_REMOTE):
    if not 1 <= number <= 15:
        raise ValueError('Fifteen-round campaign required')
    plan = read_json(CFG/f'round{number:02d}.jobs.json')
    if plan['round'] != number:
        raise ValueError('Round manifest mismatch')
    local = ROOT/f'test/terminal_relabel15/round{number:02d}'
    transport = DOC/f'round{number:02d}/transport'
    pairs = []
    views = []
    for key in ('launch', 'cleanup'):
        name = f'round{number:02d}.{key}.json'
        pairs.append([str(PurePosixPath(remote)/name), str(transport/name)])
    for job in plan['jobs']:
        name = job['campaign']
        archive = local/(name+'.tar.gz')
        meta = transport/(name+'.archive.json')
        pairs += [[str(PurePosixPath(remote)/'archives'/(name+'.tar.gz')), str(archive)],
                  [str(PurePosixPath(remote)/'archives'/(name+'.tar.gz.json')), str(meta)]]
        for clock in ('before', 'after'):
            source_name = name+f'.source.{clock}.json'
            pairs.append([str(PurePosixPath(remote)/source_name), str(transport/source_name)])
        if action == 'verify':
            view = local/'verified'/name
            result = verify_generation_archive(archive, meta, view)
            if result['campaign'] != name:
                raise ValueError('Received archive campaign mismatch')
            write_json(transport/(name+'.verification.json'), result)
            views.append((name, view))
    if action == 'verify':
        destination = local/'data'
        stage = local/'data.partial'
        if destination.exists() or stage.exists():
            raise FileExistsError('A new aggregate dataset is required')
        stage.mkdir(parents=True)
        for name, view in views:
            for file in (view/'inputs').rglob('*'):
                if file.is_file():
                    target = stage/'inputs'/file.relative_to(view/'inputs')
                    if target.exists() and digest(target) != digest(file):
                        raise ValueError('Campaign controls have different input files')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copyfile(file, target)
            shutil.copytree(view/'results'/name, stage/'results'/name)
        stage.rename(destination)
    target = transport/'download_plan.json'
    payload = {'downloads': pairs}
    if action == 'plan':
        if target.exists() and read_json(target) != payload:
            raise ValueError('Existing transport plan differs')
        write_json(target, payload)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--action', choices=['plan', 'verify'], required=True)
    parser.add_argument('--remote-work', default=DEFAULT_REMOTE)
    args = parser.parse_args()
    print(collect(args.round, args.action, args.remote_work))
