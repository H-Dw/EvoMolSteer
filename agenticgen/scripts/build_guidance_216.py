"""Check (default) or synchronize the unreleased 2.16 skill/runtime payload.

Run --sync only while preparing this release. Published releases must be immutable;
subsequent edits need another version. This script never touches earlier releases.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build(sync=False):
    root = Path(__file__).resolve().parents[1]/'src/agenticgen/resources/skill_releases/guidance/2.16.0'
    runtime = root/'runtime/linear_absolute_coordinate'
    entries = {'analysis':'jump-guidance-history', 'proposal':'jump-guidance-proposal',
               'computation':'jump-guidance-evidence-computation'}
    copies = {root/'skills'/skill/'SKILL.md':runtime/(role+'.md') for role,skill in entries.items()}
    copies[root/'skills/jump-guidance-proposal/references/search-policy.md'] = runtime/'references/search-policy.md'
    for source,dest in copies.items():
        data = source.read_bytes()
        if sync:
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(data)
        elif not dest.exists() or dest.read_bytes() != data:
            raise ValueError('Runtime copy differs from skill source: '+str(dest))
    manifest_path = root/'RELEASE.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hashes = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*')) if p.is_file() and p != manifest_path}
    if sync:
        manifest['files_sha256'] = hashes
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    elif hashes != manifest['files_sha256']:
        raise ValueError('Release manifest differs from the skill/runtime payload')
    print('2.16.0: three synchronized modules and supporting references verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync',action='store_true')
    build(parser.parse_args().sync)
