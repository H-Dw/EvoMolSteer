"""Prepare compact terminal-ranked regional evidence without any model call."""
import argparse
import json
from evomolsteer.continuous.terminal_regions import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','campaign','metrics','baseline','config','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.dataset,a.campaign,a.metrics,a.baseline,a.config,a.output),ensure_ascii=False))
