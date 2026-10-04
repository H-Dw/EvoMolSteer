"""Small auditable hypothesis tables and frozen-target replication, no refitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,write_table,digest
from evomolsteer.statistics import selection_moments


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--program',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.analysis);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    program=read_json(a.program);cfg=read_json(root/'config.json');arm=cfg['evidence_arm']
    regions=('ck2:A:GLU114:','ck2:A:HIS115:','ck2:A:VAL116:','ck2:A:ASN117:','ck2:A:ASN118:',
             'ck2:A:LYS64:','ck2:A:VAL65:','ck2:A:VAL66:','ck2:A:MET163:','ligand::radius_gyration')
    tables=[];sources={}
    for method in ['enrichment','differential','trends','pca']:
        for filename in ['effects.csv','effect_rates.csv']:
            path=root/'discovery'/method/filename;d=pd.read_csv(path)
            sources[f'{method}/{filename}']=digest(path)
            d=d[d.arm==arm].copy();d['method']=method;d['statistic']=filename.removesuffix('.csv')
            if method!='pca':d=d[d.feature.str.startswith(regions)]
            tables.append(d)
    write_table(out/'regional_effects_and_rates.csv',pd.concat(tables,ignore_index=True))
    e=pd.read_csv(root/'discovery/trends/effects.csv')
    e=e[(e.arm==arm)&(e.representation=='predicted_endpoint')&(e.contrast=='expected_selection_shift')]
    counts=e.assign(significant=e.q_value<.05).groupby('stage').significant.sum().to_dict()
    ck2=e[e.feature.str.startswith('ck2:')]
    ck2_counts=ck2.assign(significant=ck2.q_value<.05).groupby('stage').significant.sum().to_dict()
    fields=['split','batch','stage','representation','arm','step','score_time','probability','offspring_count']
    features=sorted({t['feature'] for t in program['terms']})
    d=pd.read_parquet(root/'features.parquet',columns=fields+features,
        filters=[('arm','==',arm),('representation','==','predicted_endpoint')])
    rows=[]
    for term in program['terms']:
        # Scores are rounded only for membership in the originally declared stage.
        times=d.score_time.round(6)
        sub=d[(times>=term['stage_start'])&(times<term['stage_end'])]
        for (split,batch,step),g in sub.groupby(['split','batch','step'],sort=True):
            m=selection_moments(g[[term['feature']]].to_numpy(),g.probability,g.offspring_count)
            rows.append({'term_id':term['id'],'split':split,'batch':batch,'step':step,
                         'score_time':g.score_time.iloc[0],**{k:float(v[0]) for k,v in m.items()}})
    events=pd.DataFrame(rows);write_table(out/'frozen_target_events.csv',events)
    metrics=['expected_selection_shift','realized_selection_shift','selection_noise_shift','population_mean']
    batch=events.groupby(['term_id','split','batch'],as_index=False)[metrics].mean()
    write_table(out/'frozen_target_batches.csv',batch)
    summary=[]
    for (term,split),g in batch.groupby(['term_id','split'],sort=True):
        summary.append({'term_id':term,'split':split,'n_batches':len(g),
            'expected_shift':g.expected_selection_shift.mean(),'negative_batches':int((g.expected_selection_shift<0).sum()),
            'realized_shift':g.realized_selection_shift.mean(),
            'interpretation':'descriptive replication with frozen term/stage; no new p/q values or parameter updates'})
    write_json(out/'hypothesis_summary.json',{'program_sha256':digest(a.program),'source_hashes':sources,
         'discovery_significant_endpoint_features_by_stage':counts,
         'discovery_significant_ck2_endpoint_features_by_stage':ck2_counts,'frozen_target_replication':summary,
         'notes':['Each event compares candidates in the same population; equal event then batch weights.',
                  'Validation and heldout have only 3 batches each: direction replication only.',
                  'No terminal outcome or non-selection state is read. Reward parameters are never changed.']})
    print(summary)


if __name__=='__main__':main()
