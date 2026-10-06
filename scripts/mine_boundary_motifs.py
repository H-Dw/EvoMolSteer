import argparse
from evomolsteer.continuous.boundary_motifs import mine
p=argparse.ArgumentParser()
for key in ('dataset','campaign','mining','output'):p.add_argument('--'+key,required=True)
p.add_argument('--channels',default='all,NOS_C')
a=p.parse_args();mine(a.dataset,a.campaign,a.mining,a.output,tuple(a.channels.split(',')))
