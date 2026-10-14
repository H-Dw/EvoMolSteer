"""Explicit real-API region selection followed by declarative compilation."""
import argparse
from pathlib import Path
from evomolsteer.continuous.terminal_region_agents import run_roles,compile_programs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--mining',required=True);p.add_argument('--calls',required=True);p.add_argument('--programs',required=True)
    p.add_argument('--base-program',default='configs/experiments/steer_dependency_v1/D2_without_trajectory_static.json')
    a=p.parse_args();run_roles(a.repo,a.mining,a.calls)
    print(compile_programs(a.repo,a.mining,Path(a.calls)/'selection.json',a.programs,a.base_program))
