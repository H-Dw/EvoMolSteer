"""Exact finite-population neutral ancestry reference, not simulated affinity."""
import argparse,math
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest


def occupancy_transition(n):
    """k independent uniform parental draws occupy j parents: S(k,j)(n)_j/n^k."""
    stirling=[[0]*(n+1) for _ in range(n+1)];stirling[0][0]=1
    for k in range(1,n+1):
        for j in range(1,k+1):stirling[k][j]=j*stirling[k-1][j]+stirling[k-1][j-1]
    transition=np.zeros((n+1,n+1));transition[0,0]=1
    falling=1
    for j in range(1,n+1):
        falling*=n-j+1
        for k in range(j,n+1):transition[k,j]=stirling[k][j]*falling/n**k
    if not np.allclose(transition.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Neutral parent mass lost')
    return transition


def analyze(events,population,output):
    events,output=Path(events),Path(output);d=pd.read_parquet(events);grid=np.sort(d.time.unique());batches=d.batch.nunique()
    transition=occupancy_transition(population);p=np.zeros(population+1);p[population]=1;records=[];endpoints=[]
    for step,t in enumerate(grid):
        p=p@transition;observed=d[d.time.eq(t)].surviving_roots
        if len(observed)!=batches:raise ValueError('Complete batch ancestry panel required')
        aggregate=np.array([1.])
        for _ in range(batches):aggregate=np.convolve(aggregate,p)
        cdf=np.cumsum(aggregate);lo=np.searchsorted(cdf,.025)/batches;hi=np.searchsorted(cdf,.975)/batches
        record={'step':step,'score_time':float(t),'observed_mean_roots':float(observed.mean()),
          'neutral_mean_roots':float(p@np.arange(population+1)),'neutral_panel_low95':lo,'neutral_panel_high95':hi}
        records.append(record)
        if step in [0,len(grid)-1]:
            endpoints.append({**record,'one_sided_null_probability':float(cdf[int(observed.sum())])})
    pd.DataFrame(records).to_parquet(output.with_suffix('.parquet'),compression=None,index=False)
    result={'population':population,'independent_batches':batches,'events_sha256':digest(events),'steps':len(grid),
      'method':'Exact uniform Wright-Fisher ancestry occupancy Markov distribution; independent-panel convolution, no Monte Carlo.',
      'endpoints':endpoints,'limitations':['Idealized neutral ancestry only, not a matched FLOWR outcome or physical evolution model.',
        'Recorded head scores and selection share a selector; accelerated root loss does not identify coordinate advantages or future affinity.',
        'Teacher prior ESS and inference without resampling are different objects; do not reuse this null as their performance measure.']}
    write_json(output.with_suffix('.json'),result);print(result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--events',required=True);p.add_argument('--population',required=True,type=int);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.events,a.population,a.output)
