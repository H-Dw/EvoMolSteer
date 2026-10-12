"""Summarize a read-only runtime capture using the retained API/phase receipts."""
import argparse
import json
from pathlib import Path
from evomolsteer.ablation.timing import analyze

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--capture',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    print(json.dumps(analyze(repo,repo/a.study,repo/a.capture,repo/a.output),ensure_ascii=False,indent=2))
