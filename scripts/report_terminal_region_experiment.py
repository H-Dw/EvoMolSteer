"""Compare locally evaluated regional cohorts with reused frozen controls."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.ablation.statistics import summarize,compare


def report(repo,study,output):
    repo=repo.resolve();study=study.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    bank=read_json(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json')
    threshold=read_json(repo/'configs/experiments/m2_terminal_regions_20261014/analysis.json')['evaluation']['elite_threshold_pic50']
    controls={name:pd.read_csv(repo/bank['result']['outcomes'][arm]['metrics']) for name,arm in [('M2','gradient'),('unguided','unguided')]}
    historical_path=repo/'docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv'
    historical=pd.read_csv(historical_path)
    candidates={}
    for path in sorted((study/'evaluated').glob('M2REG_terminal_*/candidate_metrics.csv')):
        candidates[path.parent.name]=pd.read_csv(path)
    rows=[];comparisons={}
    frames=dict(controls,**candidates,historical_Steer=historical)
    for name,frame in frames.items():
        summary=summarize([frame],threshold)['policy_mean']
        rows.append({'condition':name,'n':len(frame),**summary,'p99_pic50':float(frame.pic50_on_rescore.quantile(.99)),
            'best_valid_pic50':float(frame.loc[frame.valid_connected,'pic50_on_rescore'].max())})
    for name,frame in candidates.items():
        comparisons[name]={k:compare([frame],[v],family=max(1,len(candidates)*2)) for k,v in controls.items()}
    names=list(candidates)
    if len(names)==4:
        local={name:frame for name,frame in candidates.items() if '_local_' in name}
        for name,frame in local.items():
            other=name.replace('_local_','_global_')
            comparisons[name]['local_minus_same_window_global']=compare([frame],[candidates[other]],family=4)
        for mode in ['local','global']:
            early=next(n for n in names if f'_{mode}_0_0p5' in n)
            late=next(n for n in names if f'_{mode}_0p1_0p5' in n)
            comparisons[late]['delayed_minus_full']=compare([candidates[late]],[candidates[early]],family=4)
    pd.DataFrame(rows).to_csv(output/'performance.csv',index=False)
    write_json(output/'comparison.json',{'comparisons':comparisons,'summary':rows,
        'historical_steer':'Unpaired reference with different search budget and dependent descendants',
        'threshold':threshold,'candidate_programs':len(candidates),'source_control_manifest_sha256':digest(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json'),
        'historical_metrics_sha256':digest(historical_path),'evaluation':'All attempts, failures retained; final predicted pIC50, not measured affinity'})
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.',type=Path)
    p.add_argument('--study',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(report(a.repo,a.study,a.output))
