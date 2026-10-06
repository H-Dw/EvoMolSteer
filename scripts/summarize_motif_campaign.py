import argparse
from evomolsteer.generation.motif_reporting import report
p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
p.add_argument('--require-complete',action='store_true')
a=p.parse_args();report(a.evidence,a.output,a.require_complete)
