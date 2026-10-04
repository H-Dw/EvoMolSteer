"""Within-event evidence on all observed times, and retained-window ancestry."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..scope import load
from ..statistics import weighted_mean, selection_moments
from ..io import write_json, write_table
from .lineage import window_copy_counts, diagnostics

METRICS = {
    'trends':['expected_selection_shift','realized_selection_shift','selection_noise_shift',
              'population_mean','expected_selected_mean','window_ancestor_mean',
              'window_ancestor_shift','population_minus_background'],
    'enrichment':['expected_low_mass_shift','realized_low_mass_shift','ancestor_low_mass_shift',
                  'expected_high_mass_shift','realized_high_mass_shift','ancestor_high_mass_shift',
                  'retained_vs_extinct_low_log_odds','retained_vs_extinct_high_log_odds'],
    'differential':['selected_minus_rejected','same_parent_selected_minus_rejected','retained_minus_extinct'],
    'dynamics':['population_change_rate','selection_component_rate','transmission_component_rate',
                'cumulative_realized_selection','cumulative_transmission']}


def qualified_mean(x, weights, minimum):
    w=np.asarray(weights,float)
    if w.sum()<=0: return np.full(x.shape[1],np.nan)
    w=w/w.sum(); m,coverage=weighted_mean(x,w)
    return np.where(coverage>=minimum,m,np.nan)


def _table(g, features, **values):
    return pd.DataFrame({'representation':g.representation.iloc[0], 'arm':g.arm.iloc[0],
        'batch':int(g.batch.iloc[0]), 'step':int(g.step.iloc[0]),
        'time':float(g.score_time.iloc[0]),'feature':features, **values})


def build_events(analysis, split='discovery'):
    analysis=Path(analysis); df,cfg,cat,scope=load(analysis,split)
    features=list(cat['features']); minimum=cfg['minimum_feature_fraction']
    ref=df if split=='discovery' else load(analysis,'discovery')[0]
    ref=ref[ref.arm==scope['background_arm']]
    thresholds={tail:ref.groupby(['representation','step'],sort=True)[features].quantile(q)
                for tail,q in [('low',.25),('high',.75)]}
    dest=analysis/split/'continuous';dest.mkdir(parents=True,exist_ok=True)
    if split=='discovery':
        for tail,table in thresholds.items():
            write_table(dest/f'{tail}_thresholds.parquet',table.reset_index())
    backgrounds={(rep,int(batch),int(step)):g[features].mean().to_numpy()
        for (rep,batch,step),g in df[df.arm==scope['background_arm']].groupby(['representation','batch','step'],sort=True)}
    rows={name:[] for name in METRICS}; audit=[]; lineage_labels=[]
    for (rep,arm,batch),group in df[df.resampled].groupby(['representation','arm','batch'],sort=True):
        groups=[g.sort_values('slot').reset_index(drop=True) for _,g in group.groupby('step',sort=True)]
        if [int(g.step.iloc[0]) for g in groups] != scope['steps']:
            raise ValueError('Incomplete window for a branch population')
        copies=window_copy_counts(groups)
        cumulative_selection=np.zeros(len(features));cumulative_transmission=np.zeros(len(features))
        previous=None
        for i,g in enumerate(groups):
            x=g[features].to_numpy(float); count=g.offspring_count.to_numpy(float)
            p=g.probability.to_numpy(float,copy=True);p/=p.sum(); descendants=copies[i]
            keep=descendants>0
            moments=selection_moments(x,p,count,minimum)
            ancestor=qualified_mean(x,descendants,minimum)
            base=np.where(moments['valid_fraction']>=minimum,moments['population_mean'],np.nan)
            background=backgrounds[(rep,int(batch),int(g.step.iloc[0]))]
            rows['trends'].append(_table(g,features,
                expected_selection_shift=moments['expected_selection_shift'],
                realized_selection_shift=moments['realized_selection_shift'],
                selection_noise_shift=moments['selection_noise_shift'], population_mean=base,
                expected_selected_mean=base+moments['expected_selection_shift'],
                window_ancestor_mean=ancestor,window_ancestor_shift=ancestor-base,
                population_minus_background=base-background,
                valid_fraction=moments['valid_fraction']))
            # Binary survival and descendant multiplicity answer different questions.
            selected=g.selected.to_numpy(bool)
            delta=qualified_mean(x,selected,minimum)-qualified_mean(x,~selected,minimum)
            survived=qualified_mean(x,keep,minimum)-qualified_mean(x,~keep,minimum)
            sibling=[]
            if i:
                for _,s in g.groupby('parent_node_id',sort=True):
                    ss=s.selected.to_numpy(bool)
                    if ss.any() and (~ss).any():
                        xx=s[features].to_numpy(float)
                        sibling.append(qualified_mean(xx,ss,minimum)-qualified_mean(xx,~ss,minimum))
            sib=np.full(len(features),np.nan)
            if sibling:
                arr=np.asarray(sibling);valid=np.isfinite(arr);den=valid.sum(0)
                sib=np.divide(np.where(valid,arr,0).sum(0),den,out=sib,where=den>0)
            rows['differential'].append(_table(g,features,selected_minus_rejected=delta,
                same_parent_selected_minus_rejected=sib,retained_minus_extinct=survived,
                n_selected=int(selected.sum()),n_retained=int(keep.sum()),n_sibling_parents=len(sibling)))
            enrich={}
            for tail in ['low','high']:
                cutoff=thresholds[tail].loc[(rep,int(g.step.iloc[0]))].to_numpy()
                finite=np.isfinite(x)&np.isfinite(cutoff)
                # Inclusive quantiles cause trivial all-one enrichment for ties.
                indicator=np.where(finite,x<cutoff if tail=='low' else x>cutoff,np.nan)
                uniform=qualified_mean(indicator,np.ones(len(x)),minimum)
                for name,weight in [('expected',p),('realized',count),('ancestor',descendants)]:
                    enrich[f'{name}_{tail}_mass_shift']=qualified_mean(indicator,weight,minimum)-uniform
                yes=indicator==1; no=indicator==0
                a=(yes&keep[:,None]).sum(0); b=(no&keep[:,None]).sum(0)
                c=(yes&~keep[:,None]).sum(0); d=(no&~keep[:,None]).sum(0)
                eligible=((a+b)>0)&((c+d)>0)&(finite.mean(0)>=minimum)
                enrich[f'retained_vs_extinct_{tail}_log_odds']=np.where(eligible,
                    np.log((a+.5)*(d+.5)/((b+.5)*(c+.5))),np.nan)
            rows['enrichment'].append(_table(g,features,**enrich))
            dynamics={name:np.full(len(features),np.nan) for name in METRICS['dynamics']}
            if previous is not None:
                prev,px,pm=previous; parent=pd.Index(prev.node_id).get_indexer(g.parent_node_id)
                if (parent<0).any(): raise ValueError('Missing parent edge')
                dt=float(g.score_time.iloc[0]-prev.score_time.iloc[0])
                # Complete-feature identity only: missingness may change denominators.
                complete=np.isfinite(px).all(0)&np.isfinite(x).all(0)
                transmission=(x-px[parent]).mean(0)
                shift=pm['realized_selection_shift']
                residual=base-pm['population_mean']-shift-transmission
                if np.any(np.abs(residual[complete])>1e-9): raise ValueError('Selection/transmission identity failed')
                cumulative_selection+=np.where(complete,shift,np.nan)
                cumulative_transmission+=np.where(complete,transmission,np.nan)
                dynamics.update(population_change_rate=np.where(complete,(base-pm['population_mean'])/dt,np.nan),
                    selection_component_rate=np.where(complete,shift/dt,np.nan),
                    transmission_component_rate=np.where(complete,transmission/dt,np.nan))
            dynamics['cumulative_realized_selection']=cumulative_selection.copy()
            dynamics['cumulative_transmission']=cumulative_transmission.copy()
            rows['dynamics'].append(_table(g,features,**dynamics))
            if rep==cfg['analysis_representations'][0]:
                audit.append(diagnostics(g,descendants))
                label=g[['node_id','parent_node_id','arm','batch','step','slot','score_time','root_id','offspring_count','probability','pic50_on']].copy()
                label['window_end_copies']=descendants
                lineage_labels.append(label)
            previous=(g,x,moments)
    for method,parts in rows.items():
        write_table(dest/method/'batch_event_curves.parquet',pd.concat(parts,ignore_index=True))
    write_table(dest/'lineage/diagnostics.csv',audit)
    write_table(dest/'lineage/window_ancestry.parquet',pd.concat(lineage_labels,ignore_index=True))
    write_json(dest/'scope.json',{'schema_version':'continuous-3.0','scope':scope,
        'no_time_bins':True,'end_label':'Copies immediately after the last actual selection; never final t=1 outcomes',
        'retrospective_warning':'Window-survivor features condition on future selections; not independent causal evidence.',
        'energy_status':cat.get('energy_status',{}),'n_features':len(features),'split':split})
    return dest
