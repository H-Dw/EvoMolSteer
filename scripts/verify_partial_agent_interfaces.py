"""Verify factual selection metadata and coupled real APIs on frozen coordinates."""
import argparse
from pathlib import Path

from evomolsteer.io import read_json, write_json
from evomolsteer.partial_search.agents import analyze, signature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    p.add_argument('--config',required=True);p.add_argument('--round',default='R13');a=p.parse_args()
    repo=Path(a.repo).resolve();master=read_json(repo/a.config)
    cfg=read_json(repo/'configs/experiments/partial_region_search_20261014'/(a.round+'.json'))
    folder=repo/cfg['study']/'candidates'/cfg['candidate_id'];original=read_json(folder/'Analyst.json')
    cfg.update(study=master['campaign_study']+'/interface_replay_v3',analyst_skill=master['analyst_skill'],
        designer_skill=master['designer_skill'],frozen_selection={k:original[k] for k in ('region_ids','atom_slots','point_importance')})
    programs=analyze(repo,cfg,{'verification_only':True,'no_generation':True,
        'instruction':'Preserve the frozen checkpoint, audit its evidence rather than calling it the strongest. Check explicit catalogue eligibility and state a genuinely prospective falsifier.'})
    expected=[read_json(repo/b['program']) for b in read_json(folder/'bindings.json')['programs']]
    unchanged=[signature(p) for p in programs]==[signature(p) for p in expected]
    if not unchanged:raise ValueError('Frozen interface verification changed the actual reward')
    write_json(repo/cfg['study']/'result.json',{'verification_kind':'Real-API request/extraction verification, outside optimization rounds',
        'frozen_checkpoint':a.round,'unchanged_numerical_reward':unchanged,'additional_generation_attempts':0})
    print({'unchanged_numerical_reward':unchanged,'additional_generation_attempts':0})


if __name__=='__main__':main()
