"""Build immutable, source-verified coordinate definitions for role inputs."""
import argparse
from pathlib import Path
from evomolsteer.ablation.observables import build

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--study',type=Path,required=True)
    args=p.parse_args();print(build(args.repo,args.study))
