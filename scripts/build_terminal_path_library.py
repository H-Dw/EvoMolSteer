import argparse
from evomolsteer.continuous.terminal_path_library import build_library
from evomolsteer.io import write_json
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['dataset','graph','credit','metrics','baseline','output','report']:p.add_argument('--'+key,required=True)
    p.add_argument('--budget',type=int,default=2)
    a=p.parse_args();r=build_library(a.dataset,a.graph,a.credit,a.metrics,a.baseline,a.output,a.budget)
    write_json(a.report,r);print(r)
