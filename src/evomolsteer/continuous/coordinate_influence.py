"""Attribute a whole-window effect to its observed quadrature nodes, without bins."""
from pathlib import Path
import numpy as np
import pandas as pd
from .functional import quadrature
from ..io import read_json,write_json,write_table,digest


def analyze(mining,output,metrics=('selection_shift','partial_affinity_correlation','lag_partial_gain_correlation')):
    source,out=Path(mining),Path(output)
    if out.exists():raise FileExistsError(out)
    m=read_json(source/'manifest.json');times=np.asarray(m['times'])
    d=pd.read_parquet(source/'batch_coordinate_statistics.parquet',columns=['batch','time','feature',*metrics])
    rows=[]
    for split,batches in m['splits'].items():
        for feature,g in d[d.batch.isin(batches)].groupby('feature',sort=True):
            for metric in metrics:
                grid=times[:-1] if metric.startswith('lag_') else times
                values=g.pivot(index='batch',columns='time',values=metric).reindex(columns=grid).to_numpy()
                w=quadrature(grid);finite=np.isfinite(values);coverage=finite@w
                valid=coverage>=.25 if metric.startswith('lag_') else finite.all(1)
                if valid.sum()<2:continue
                contributions=np.where(finite[valid],values[valid],0)*w/coverage[valid,None]
                node=contributions.mean(0);mass=np.abs(node);total=mass.sum();j=int(mass.argmax())
                rows.append({'split':split,'feature':feature,'metric':metric,'n_batches':int(valid.sum()),
                    'signed_window_effect':float(node.sum()),'absolute_contribution_sum':float(total),
                    'first_time':float(grid[0]),'first_node_contribution':float(node[0]),
                    'first_fraction_absolute_contributions':float(mass[0]/total) if total>1e-15 else None,
                    'dominant_time':float(grid[j]),'dominant_fraction_absolute_contributions':float(mass[j]/total) if total>1e-15 else None,
                    'effective_time_nodes':float(total**2/(mass@mass)) if total>1e-15 else None,
                    'sum_of_all_other_node_contributions':float(node[1:].sum()),
                    'mean_time_coverage':float(coverage[valid].mean())})
    write_table(out/'whole_window_node_influence.csv',rows)
    write_json(out/'manifest.json',{'window':m['window'],'source_manifest_sha256':digest(source/'manifest.json'),
        'metrics':list(metrics),'n_rows':len(rows),'per_particle_detail':False,
        'interpretation':'Decomposition of the original full-window statistic, not subinterval hypothesis tests. Time nodes are not independent replicates; SDE startup can dominate measured native velocity.'})

