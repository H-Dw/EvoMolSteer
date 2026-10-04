"""Evaluate a frozen temporal feature curve and its analytic time derivatives."""
import argparse
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_table,write_json,digest
from evomolsteer.continuous.functional import evaluate_frozen


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--functions',required=True)
    p.add_argument('--model-id',required=True);p.add_argument('--times',nargs='+',type=float,required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    model=read_json(a.functions)[a.model_id];t=np.asarray(a.times)
    write_table(a.output,pd.DataFrame({'time':t,'value':evaluate_frozen(model,t),
        'time_derivative':evaluate_frozen(model,t,1),'time_second_derivative':evaluate_frozen(model,t,2)}))
    write_json(a.output+'.json',{'model':model,'source_sha256':digest(a.functions),
        'meaning':'Temporal descriptive curve. These derivatives are NOT spatial reward gradients.'})
