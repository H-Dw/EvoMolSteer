import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json
from evomolsteer.generation.window_summary import summarize_curves

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.report_directory);r=read_json(root/'execution_report.json')
    sizes={(v['arm'],v['batch']):v['n'] for v in r['batch_results']}
    summary=summarize_curves(pd.read_csv(root/'regional_time_metrics.csv'),r['window'],sizes)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);summary.to_csv(out,index=False)
    print(summary[['arm','batch','mean_deficit_time_mean','availability_mean_over_time']].to_string(index=False))
