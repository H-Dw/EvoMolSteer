#!/usr/bin/env python3
"""Run four isolated analysis sessions and one synthesis session. No GPU sampling."""
import argparse
import json
from pathlib import Path

from evomolsteer.adaptive_schedule.io import read, defaults
from evomolsteer.adaptive_schedule.council import analyze_four_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True,
                   help='Four group evidence paths, anchors and optional separate history paths')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--published-memories', type=Path)
    args = p.parse_args()
    manifest = read(args.manifest)
    specs = []
    for row in manifest['groups']:
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else args.manifest.parent/path
        specs.append({'group_id': row['group_id'], 'data': read(resolve(row['evidence'])),
                      'anchor_ticks': row['anchor_ticks'],
                      'group_history': read(resolve(row['history'])) if row.get('history') else []})
    result = analyze_four_groups(specs, args.output, defaults(),
        published_reflections=read(args.published_memories) if args.published_memories else None)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
