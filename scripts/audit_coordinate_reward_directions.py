import argparse
from evomolsteer.continuous.reward_direction_diagnostics import audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Offline temperature direction sensitivity on original native control trajectories')
    for name in ('dataset','campaign','program','reference','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--temperatures',default='.05,1.');p.add_argument('--batches',default='0,1')
    a=p.parse_args();result=audit(a.dataset,a.campaign,a.program,a.reference,a.output,
        tuple(float(v) for v in a.temperatures.split(',')),tuple(int(v) for v in a.batches.split(',')))
    print(result['summary'])
