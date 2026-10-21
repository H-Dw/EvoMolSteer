#!/usr/bin/env python3
"""Export frozen population evidence, list/call tools, or obtain teacher-level hypotheses.

This command never starts formal generation. DeepSeek reads DEEPSEEK_API_KEY
and the inherited proxy environment through the existing audited transport.
"""
import argparse
import json
from pathlib import Path

from evomolsteer.adaptive_schedule.io import read, defaults
from evomolsteer.adaptive_schedule.group_tools import export_legacy_population, PopulationTools
from evomolsteer.adaptive_schedule.group_agent import capacity_count, plan_population
from evomolsteer.adaptive_schedule.cohort import TEACHERS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--source-root', type=Path, required=True)
    export.add_argument('--teacher', required=True)
    export.add_argument('--version', choices=['v1', 'v2'], default='v1')
    export.add_argument('--observed-round', type=int, required=True)
    export.add_argument('--output', type=Path, required=True)
    for name in ['tools', 'plan']:
        cmd = sub.add_parser(name)
        cmd.add_argument('--evidence', type=Path, required=True)
        cmd.add_argument('--output', type=Path, required=True)
        if name == 'tools':
            cmd.add_argument('--name')
            cmd.add_argument('--arguments', default='{}')
        else:
            cmd.add_argument('--capacity', type=Path, required=True)
            cmd.add_argument('--anchor-ticks', type=int, nargs=4, default=[5]*4)
            cmd.add_argument('--group-history', type=Path)
    args = p.parse_args()
    if args.command == 'export':
        data = export_legacy_population(args.source_root, args.teacher, args.version, args.observed_round, args.output)
        result = {'candidates': len(data['chains']), 'teacher': data['teacher']['teacher_id'], 'output': str(args.output)}
    elif args.command == 'tools':
        tools = PopulationTools(read(args.evidence), args.output)
        result = tools.call(args.name, json.loads(args.arguments)) if args.name else tools.catalog()
    else:
        # Capacity selection follows the user's 8 -> 4 -> 3 -> 2 policy.
        expected = [f'{t["target"]}__top{t["rank"]}' for t in TEACHERS]
        count = capacity_count(args.capacity, expected)
        result = plan_population(read(args.evidence), args.output, defaults(), hypotheses=count,
            anchor=args.anchor_ticks, group_history=read(args.group_history) if args.group_history else None)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
