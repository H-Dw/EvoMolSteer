"""Continuous-window descriptive integrals; never subdivide or refit rewards."""
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


def summarize_curves(curves, window, batch_sizes):
    a,b=map(float,window)
    if not b>a:raise ValueError('Positive learning window required')
    d=curves[curves.time.between(a-1e-6,b+1e-6)];rows=[]
    for (arm,batch),g in d.groupby(['arm','batch'],sort=True):
        g=g.sort_values('time');t=g.time.to_numpy(float)
        if len(t)<2 or (np.diff(t)<=0).any() or abs(t[0]-a)>2e-6 or abs(t[-1]-b)>2e-6:
            raise ValueError('Complete unique continuous-window observations required')
        n=batch_sizes[(arm,int(batch))];available=g.n_available.to_numpy(float)/n
        if (available<0).any() or (available>1).any():raise ValueError('Invalid availability denominator')
        row={'arm':arm,'batch':int(batch),'window_start':a,'window_end':b,'nodes':len(t),
             'candidate_n':n,'availability_mean_over_time':float(trapezoid(available,t)/(b-a)),
             'availability_min':float(available.min()),'interpretation':'Descriptive window integral, not a new reward or independent efficacy test'}
        for field in ['mean_deficit','inside_fraction_available']:
            value=g[field].to_numpy(float);finite=np.isfinite(value)
            row[field+'_missing_nodes']=int((~finite).sum())
            row[field+'_time_mean']=float(trapezoid(value,t)/(b-a)) if finite.all() else None
            row[field+'_endpoint_change']=float(value[-1]-value[0]) if finite[[0,-1]].all() else None
        value=g.inside_fraction_available.to_numpy(float)*available
        row['inside_all_slot_time_mean']=float(trapezoid(value,t)/(b-a)) if np.isfinite(value).all() else None
        rows.append(row)
    return pd.DataFrame(rows)
