"""Refresh observed paired-batch rates without refitting frozen functions."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,write_table,digest
from evomolsteer.continuous.summary import KEYS,FIT_METRICS,point_curve
from evomolsteer.continuous.events import METRICS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True);a=p.parse_args()
    root=Path(a.analysis);record={}
    for path in sorted(root.glob('*/continuous/*/batch_event_curves.parquet')):
        method=path.parent.name;events=pd.read_parquet(path);rows=[]
        for key,g in events.groupby(KEYS,sort=True):
            times=sorted(g.time.unique())
            for metric in METRICS.get(method,FIT_METRICS[method]):
                if metric not in events:continue
                matrix=g.pivot(index='batch',columns='time',values=metric).reindex(columns=times).to_numpy()
                rows.append(point_curve(dict(zip(KEYS,key)),metric,times,matrix))
        target=path.parent/'point_curves.parquet';old=digest(target)
        write_table(target,pd.concat(rows,ignore_index=True))
        record[str(target.relative_to(root))]={'source_sha256':digest(path),'old_sha256':old,'sha256':digest(target)}
    write_json(root/'paired_rate_refresh.json',{'reason':'Adjacent event rates use the same finite batch pairs; includes paired coverage/SE',
        'script_sha256':digest(__file__),'summary_source_sha256':digest(Path(__file__).resolve().parents[1]/'src/evomolsteer/continuous/summary.py'),
        'refit':False,'files':record})
    print('Refreshed paired rates:',len(record))


if __name__=='__main__':main()
