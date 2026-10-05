import argparse
from evomolsteer.generation.local_reference import build_local_reference

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);p.add_argument('--central-mass',type=float,default=.5)
    a=p.parse_args();r=build_local_reference(a.dataset,a.campaign,a.analysis,a.output,a.central_mass)
    print({'window':r['window'],'times':len(r['times'])})
