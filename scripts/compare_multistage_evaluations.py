"""Compare all deterministic result tables produced on two hosts."""
import argparse
from pathlib import Path
import pandas as pd

from evomolsteer.io import digest, write_json


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tolerance',type=float,default=1e-10)
    args=parser.parse_args()
    if not 0<=args.tolerance<1e-3:raise ValueError('Invalid comparison tolerance')
    reference,candidate=Path(args.reference),Path(args.candidate)
    names=lambda root:{p.name for p in root.iterdir() if p.suffix in ('.csv','.parquet')}
    if names(reference)!=names(candidate) or not names(reference):
        raise ValueError('Both evaluations must contain the same nonempty set of result tables')
    rows=[]
    for name in sorted(names(reference)):
        read=pd.read_parquet if name.endswith('.parquet') else pd.read_csv
        left,right=read(reference/name),read(candidate/name)
        pd.testing.assert_frame_equal(left,right,check_dtype=False,check_exact=False,
                                      rtol=args.tolerance,atol=args.tolerance)
        rows.append({'file':name,'rows':len(left),'columns':len(left.columns),
            'reference_sha256':digest(reference/name),'candidate_sha256':digest(candidate/name),
            'numeric_equivalence':True})
    write_json(args.output,{'status':'passed','tables':len(rows),'relative_and_absolute_tolerance':args.tolerance,
        'reference':str(reference.resolve()),'candidate':str(candidate.resolve()),'files':rows})
    print('Verified',len(rows),'equivalent tables')
