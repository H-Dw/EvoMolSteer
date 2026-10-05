"""Local paired round ledger; preserve rejections and never retune confirmation."""
import argparse
import json
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest


def compare(root,output):
    root=Path(root);paths=sorted(root.glob('round_*/report.json'))
    reports=[read_json(p) for p in paths]
    baseline=reports[0];native=sorted([r for r in baseline['batch_results'] if r['arm']=='unguided'],key=lambda x:x['batch'])
    incumbent=np.array([r['symmetric_shape_A'] for r in native]);incumbent_name='native';records=[]
    native_mean=float(incumbent.mean())
    for path,r in zip(paths,reports):
        rows=sorted([v for v in r['batch_results'] if v['arm']=='gradient'],key=lambda x:x['batch'])
        confirmation=r['seed']!=baseline['seed']
        if not confirmation:
            if len(rows)!=len(native):raise ValueError('Adaptation batch mismatch')
            for row in rows:
                key=str(row['batch'])
                if r['initial_state_signatures']['gradient/'+key]!=baseline['initial_state_signatures']['unguided/'+key]:raise ValueError('Unpaired initial state')
            values=np.array([v['symmetric_shape_A'] for v in rows])
            gain=1-values.mean()/incumbent.mean()
            accepted=bool(gain>=.02 and (values<incumbent).all())
            record={'campaign':r['campaign'],'role':'adaptation','primary':float(values.mean()),
                    'improvement_vs_native_percent':float(100*(1-values.mean()/native_mean)),
                    'comparator':incumbent_name,'improvement_vs_incumbent_percent':float(100*gain),
                    'batch_differences_vs_incumbent_A':(values-incumbent).tolist(),'accepted':accepted,
                    'report_sha256':digest(path)}
            if accepted:incumbent=values;incumbent_name=r['campaign']
        else:
            record={'campaign':r['campaign'],'role':'frozen_new_seed_confirmation','report_sha256':digest(path),'not_used_for_selection':True}
        records.append(record)
    result={'selection_rule':'At least 2% primary mean improvement and every paired adaptation batch improves.',
            'accepted_incumbent':incumbent_name,'records':records,'n_optimization_rounds_observed':len(records)}
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(compare(a.results,a.output),indent=2))
