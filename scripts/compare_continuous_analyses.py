"""Numerical cross-host equivalence of all primary continuous result tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    left=Path(a.reference);right=Path(a.candidate);records=[]
    for f in sorted(left.glob('*/continuous/**/*')):
        if f.suffix not in ['.csv','.parquet']:continue
        other=right/f.relative_to(left)
        load=pd.read_parquet if f.suffix=='.parquet' else pd.read_csv
        x=load(f);y=load(other)
        equal=list(x.columns)==list(y.columns) and x.shape==y.shape
        maxdiff=0.;bad=[]
        if equal:
            for col in x:
                if pd.api.types.is_numeric_dtype(x[col]):
                    xx=x[col].to_numpy(float); yy=y[col].to_numpy(float)
                    good=np.allclose(xx,yy,rtol=1e-7,atol=1e-9,equal_nan=True)
                    finite=np.isfinite(xx)&np.isfinite(yy)
                    if finite.any():maxdiff=max(maxdiff,float(np.max(np.abs(xx[finite]-yy[finite]))))
                else:good=x[col].fillna('').equals(y[col].fillna(''))
                if not good:bad.append(col)
        records.append({'file':str(f.relative_to(left)),'shape_equal':equal,'max_abs_difference':maxdiff,'different_columns':bad})
    passed=bool(records) and all(r['shape_equal'] and not r['different_columns'] for r in records)
    write_json(a.output,{'passed':passed,'rtol':1e-7,'atol':1e-9,'tables':records})
    print('Tables compared:',len(records),'all equivalent:',passed)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
