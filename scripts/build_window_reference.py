import argparse
from evomolsteer.generation.window_reference import build_reference

p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
p.add_argument('--batches',required=True,help='Comma-separated discovery batch ids')
p.add_argument('--output',required=True);p.add_argument('--window-start',type=float);p.add_argument('--window-end',type=float)
p.add_argument('--representatives',type=int,default=2);a=p.parse_args()
r=build_reference(a.dataset,a.campaign,[int(s) for s in a.batches.split(',')],a.output,a.window_start,a.window_end,a.representatives)
print('Built actual-state reference:',r['window'],len(r['frames']),'representatives')
