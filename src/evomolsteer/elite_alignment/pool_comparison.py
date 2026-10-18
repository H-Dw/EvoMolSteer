"""Describe historical ordinary-prior pools versus exact elite-root reruns.

Cross-pool differences are not a randomized test of elite-root selection:
population size, suffix streams and batch shape differ. Within each historical
pool, controls are matched by batch and slot before computing guidance gains.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import read_json, digest, write_table
from evomolsteer.storage.transactions import atomic_json
from .analysis import distribution
from .common import validate_protocol


def unique_rows(files):
    rows=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    for _,g in rows.groupby(['batch','slot']):
        if g.pic50_on_rescore.max()-g.pic50_on_rescore.min()>1e-6:
            raise ValueError('Conflicting historical duplicate scores')
    return rows.drop_duplicates(['batch','slot']).copy()


def run(repo,config,work,output):
    repo,work,output=Path(repo).resolve(),Path(work).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    cfg=read_json(config);validate_protocol(cfg)
    collection=read_json(repo/'docs/experiments'/cfg['name']/'latest_collection.json')
    records=[read_json(work/p) for p in collection['snapshot']['receipt_paths']]
    current=pd.DataFrame([{'condition':r['condition'],'rank':r['attempt']['rank'],
        'pic50':r['pic50'],'valid':r['valid_connected'],'qualified':r['qualified_structure']}
        for r in records if r['status']=='complete'])
    metrics=[]
    for cohort,spec in cfg['cohorts'].items():
        for condition in cfg['conditions']:
            g=current[(current.condition==condition['id'])&(current['rank']<=spec['n'])]
            metrics.append({'cohort':cohort,'condition':condition['id'],
                'complete':len(g)==spec['n'],**distribution(g.pic50,g.valid,g.qualified,
                    spec['n'],cfg['tail_threshold'],cfg['steer_best'])})
    metrics=pd.DataFrame(metrics)
    tools=repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/transport_records'
    temporal=repo/'docs/experiments/temporal_guidance_20261013/transport_records'
    regions=repo/'docs/experiments/m2_terminal_regions_20261014/evaluated'
    input_paths=set()
    def read(files):
        input_paths.update(files);return unique_rows(files)
    base_file=tools/'DSF_c62e296818e0_b6376_controls_indexed_batch.unguided.metrics.csv'
    baseline=read([base_file])
    histories={}
    for cid,sig in [('M2','c62e296818e0'),('blend_0p5','ab41341eeb9b'),
                    ('blend_0p25','6d42e0d667e4'),('linear_decay','aa26106a518f')]:
        histories[cid]=read(sorted(tools.glob(f'DSF_{sig}_b6376*.gradient.metrics.csv')))
    for cid,sig in [('S3','3255dfbc980c'),('C3','20b868bcf02b'),('C2','572c774209da'),
                    ('S1','4edf30248b26'),('W1','e8a8ad052151'),('W3','1f217373730e')]:
        histories[cid]=read(sorted(temporal.glob(f'DSF_{sig}_b63*.gradient.metrics.csv')))
    for cid,folder in [('top10_global_0','M2REG_terminal_topk_global_0_0p5'),
                       ('top10_global_0p1','M2REG_terminal_topk_global_0p1_0p5'),
                       ('top10_local_0p1','M2REG_terminal_topk_local_0p1_0p5')]:
        histories[cid]=read([regions/folder/'candidate_metrics.csv'])
    new=metrics[metrics.cohort=='top20'].set_index('condition')
    new_base=float(new.loc['unguided','mean']); comparisons=[]
    for cid,history in histories.items():
        matched=history.merge(baseline[['batch','slot','pic50_on_rescore']],
            on=['batch','slot'],suffixes=('','_native'),validate='one_to_one')
        if len(matched)!=len(history):raise ValueError('Historical control coverage differs: '+cid)
        old_mean=float(matched.pic50_on_rescore.mean())
        old_base=float(matched.pic50_on_rescore_native.mean())
        old_gain=old_mean-old_base
        row=new.loc[cid]
        comparisons.append({'condition':cid,'old_n':len(matched),'old_mean':old_mean,
            'old_native_mean':old_base,'old_gain':old_gain,'old_best':float(matched.pic50_on_rescore.max()),
            'new_n':int(row.n_scored),'new_complete':bool(row.complete),'new_mean':row['mean'],
            'new_native_mean':new_base,'new_gain':float(row['mean']-new_base),
            'new_best':row['best_valid'],'absolute_mean_change':float(row['mean']-old_mean),
            'guidance_gain_change':float(row['mean']-new_base-old_gain)})
    comparisons=pd.DataFrame(comparisons)
    complete=comparisons[comparisons.new_complete]
    controls=[]
    frozen_path=repo/'docs/experiments/m2_tools_full_drop_review_20261012_update_01/program_quality.csv'
    input_paths.add(frozen_path)
    frozen=pd.read_csv(frozen_path).set_index('program')
    for key,label in [('c62e296818e05c479dc68bee5d34f24ef0871cab3bc3e6204f442a9b6dfe40cd','gradient'),
                      ('unguided','unguided')]:
        r=frozen.loc[key]
        controls.append({'pool':'frozen_20261012_report_'+label,'n':int(r.attempts),
            'mean':float(r.mean_pic50),'best':float(r.max_pic50)})
    for arm in ['gradient','unguided']:
        files=sorted(tools.glob(f'DSF_c62e296818e0_b*_controls_indexed_batch.{arm}.metrics.csv'))
        pool=read(files)
        controls.append({'pool':'historical_all_'+arm,'n':len(pool),'mean':float(pool.pic50_on_rescore.mean()),
            'best':float(pool.loc[pool.valid_connected,'pic50_on_rescore'].max())})
    for cohort,spec in cfg['cohorts'].items():
        for cid in ['unguided','M2']:
            r=metrics[(metrics.cohort==cohort)&(metrics.condition==cid)].iloc[0]
            controls.append({'pool':cohort+'_'+cid,'n':int(r.n_scored),'mean':float(r['mean']),'best':float(r.best_valid)})
    for name,frame in [('current_performance',metrics),('historical_pool_comparison',comparisons),
                       ('M2_pool_sizes',pd.DataFrame(controls))]:
        write_table(output/(name+'.csv'),frame)
    result={'snapshot_sha256':collection['snapshot']['sha256'],'collected_unix':collection['collected_unix'],
        'complete_guided_conditions_comparable':len(complete),
        'absolute_mean_lower_than_old_pool':int((complete.absolute_mean_change<0).sum()),
        'within_pool_guidance_gain_lower_than_old':int((complete.guidance_gain_change<0).sum()),
        'historical_files_sha256':{p.relative_to(repo).as_posix():digest(p) for p in sorted(input_paths)},
        'interpretation':'Descriptive cross-pool comparison. No causal estimate of elite-root selection: batching and suffix RNG also changed.'}
    atomic_json(output/'pool_comparison_audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.');p.add_argument('--config',required=True)
    p.add_argument('--work',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.repo,a.config,a.work,a.output)
    print({k:v for k,v in r.items() if k!='historical_files_sha256'})


if __name__=='__main__':main()
