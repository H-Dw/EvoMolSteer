"""Rebuild compact execution/feature-response evidence from completed conditions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from evomolsteer.io import read_json,write_json,digest
from .verify import observe


FEATURES=['initial_target_error_A','last_target_error_A','boundary_posthoc_local_rms_A',
    'terminal_posthoc_local_rms_A','actual_cumulative_step_rms_A',
    'boundary_tracked_terminal_element_fraction','terminal_tracked_terminal_element_fraction']


def analyze(repo,config,output):
    repo=Path(repo);m=read_json(config);study=repo/m['campaign_study'];state=read_json(study/'state.json')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];sources={};executions=[]
    for condition in state['rounds']:
        label=condition['round'];tables=[]
        for f in sorted((study/label).glob('outcomes_*/per_seed.parquet')):
            sources[f.relative_to(repo).as_posix()]=digest(f);tables.append(pd.read_parquet(f))
        for f in condition['result_folders']:
            destination=output/'execution'/(label+'__'+Path(f).name+'__'+Path(f).parent.name+'.json')
            executions.append({'round':label,**observe(repo/f,destination)})
        if not tables:continue
        table=pd.concat(tables,ignore_index=True)
        row={'round':label,'phase':condition['phase'],'parent':condition['parent'],'n':len(table)}
        for feature in FEATURES:
            valid=table[[feature,'pic50']].replace([np.inf,-np.inf],np.nan).dropna()
            row[feature+'_vs_pic50_spearman']=float(spearmanr(valid[feature],valid.pic50).statistic) if valid[feature].nunique()>1 and valid.pic50.nunique()>1 else None
            row[feature+'_mean']=float(valid[feature].mean()) if len(valid) else None
        rows.append(row)
    result={'sources':sources,'rows':rows,'execution_summary':[{
        'round':e['round'],'program_id':e['program_id'],'passed':e['passed'],'active_steps':e['active_steps'],
        'nonzero_steps':e['nonzero_steps'],'mode_switches':e['mode_switches'],'routing_VJPs':e['routing_VJPs'],'components':e['components']} for e in executions],
        'limits':['Within-condition descriptive correlations; no significance or causal routing rule inferred.',
            'Policy-relative target errors from different clocks are different observables.',
            'Confirmation batches concatenated only for descriptive correlations; formal intervals use independent batch units.',
            'Old header real_generator_vjp can overstate pure current-state derivatives; per-step derivative_scope is authoritative.']}
    write_json(output/'diagnostics.json',result)
    pd.DataFrame(rows).to_csv(output/'feature_correlations.csv',index=False)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(Path(a.repo).resolve(),a.config,a.output);print({'conditions':len(r['rows']),'verified_executions':len(r['execution_summary'])})


if __name__=='__main__':main()
