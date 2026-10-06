import argparse
from evomolsteer.generation.motif_reward import build
p=argparse.ArgumentParser()
for key in ('dataset','campaign','mining','output'):p.add_argument('--'+key,required=True)
p.add_argument('--channels',default='all')
p.add_argument('--target',choices=('instantaneous','boundary_survival'),default='instantaneous')
p.add_argument('--strict-scale',action='store_true')
a=p.parse_args();build(a.dataset,a.campaign,a.mining,a.output,tuple(a.channels.split(',')),a.target,a.strict_scale)
