"""Compact per-seed formation/retention, routing and actual-dose diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import read_json,write_json,digest
from .bank import lookup


def analyze(folder,bank,output):
    folder=Path(folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit=read_json(folder/'guidance_audit.json');p=audit['program']
    if not p or p['family']!='ancestral_modes_v3':raise ValueError('Expected an ancestral-mode guided outcome')
    manifest={'audit_sha256':digest(folder/'guidance_audit.json'),'geometry_sha256':digest(folder/'geometry.json'),
        'records_sha256':digest(folder/'records.json')}
    result_path=output/'summary.json'
    if result_path.exists():
        old=read_json(result_path)
        if old['sources']!=manifest:raise ValueError('Compact outcome cache input changed')
        return old
    geometry=read_json(folder/'geometry.json');records=read_json(folder/'records.json')
    active=[r for r in audit['steps'] if r['active']];last=active[-1];n=len(records);rows=[];mode_counts={};curves=[]
    categories=np.load(folder/'categorical_states.npz',allow_pickle=False)
    mode_matrix=np.asarray([r['mode_indices'] for r in active]);dose=np.asarray([r['injected_rms_A_per_seed'] for r in active])
    errors=np.asarray([r['target_error_per_seed_A'] for r in active]);times=np.asarray([r['time'] for r in active])
    for i in range(n):
        mid=int(last['mode_indices'][i]);s=p['mode_specs'][mid];m=bank['modes'][s['template_id']];slots=s['atom_slots']
        followed=np.asarray(last['actual_local_assignments'][i],dtype=int)
        labels=np.asarray([m['atom_vocabulary'][m['native_elements'][j]] for j in slots])
        mode_counts[s['template_id']]=mode_counts.get(s['template_id'],0)+1
        row={'slot':i,'pic50':records[i]['pic50'],'qualified':records[i]['qualified_structure'],'mode_id':s['template_id'],
            'support_count':len(slots),'mode_switches':int((mode_matrix[1:,i]!=mode_matrix[:-1,i]).sum()),
            'actual_cumulative_step_rms_A':float(dose[:,i].sum()),'actual_max_step_rms_A':float(dose[:,i].max()),
            'initial_target_error_A':float(errors[0,i]),'last_target_error_A':float(errors[-1,i])}
        for name,key in [('boundary','teacher_clock'),('terminal','terminal')]:
            state=geometry[key];xyz=np.asarray(state['coords_A'])[i];atomic=np.asarray(state['atomic_indices'])[i]
            desired=lookup(m,p['learning_window'][1])[slots] if name=='boundary' else np.asarray(m['terminal_coords_A'])[slots]
            rr,cc=linear_sum_assignment(np.square(xyz[:,None]-desired).sum(-1));ids=rr[np.argsort(cc)]
            row[name+'_posthoc_local_rms_A']=float(np.sqrt(np.square(xyz[ids]-desired).sum(-1).mean()))
            row[name+'_geometry_matched_terminal_element_fraction']=float((atomic[ids]==labels).mean())
            row[name+'_tracked_terminal_element_fraction']=float((atomic[followed]==labels).mean())
            nc=np.isin(labels,[m['atom_vocabulary']['N'],m['atom_vocabulary']['C']])
            if nc.any():row[name+'_tracked_NC_joint_fraction']=float((atomic[followed][nc]==labels[nc]).mean())
            if name=='boundary':
                nearest=int(np.abs(np.asarray(m['path']['times'])-p['learning_window'][1]).argmin())
                actual=np.asarray(m['path']['current_atomics'][nearest])[slots]
                row['boundary_tracked_actual_ancestor_element_fraction']=float((atomic[followed]==actual).mean())
            charge_key=('boundary' if name=='boundary' else 'terminal')+'_charges'
            if charge_key in categories:
                expected=np.asarray(m['path']['current_charges'][-1])[slots] if name=='boundary' else np.asarray(m['terminal_charges'])[slots]
                row[name+'_tracked_source_charge_fraction']=float((categories[charge_key][i,followed]==expected).mean())
        rows.append(row)
        if len(times)>=3:
            coef=np.polynomial.polynomial.polyfit(times,errors[:,i],2)
            curves.append({'slot':i,'observable':'distance_to_current_policy_target','domain':[float(times.min()),float(times.max())],
                'control_window':p['window'],
                'coefficients_A_per_time_power':coef.tolist(),'derivative_coefficients':(coef[1:]*[1,2]).tolist(),
                'rmse_A':float(np.sqrt(np.square(np.polynomial.polynomial.polyval(times,coef)-errors[:,i]).mean()))})
    table=pd.DataFrame(rows);table.to_parquet(output/'per_seed.parquet',compression=None,index=False)
    numeric=table.drop(columns=['slot']).select_dtypes(include=[np.number]);counts=np.asarray(list(mode_counts.values()))/n
    by_mode=[]
    for mode,g in table.groupby('mode_id'):
        by_mode.append({'mode_id':mode,'n':len(g),'mean_pic50':float(g.pic50.mean()),'best_pic50':float(g.pic50.max()),
            'qualified':int(g.qualified.sum()),'mean_dose_A':float(g.actual_cumulative_step_rms_A.mean())})
    result={'sources':manifest,'n':n,'mode_occupancy':mode_counts,'mode_entropy_nats':float(-(counts*np.log(counts)).sum()),
        'per_mode_outcomes':by_mode,'means':{k:float(v) for k,v in numeric.mean().items()},'full_window_error_fits':curves,
        'limits':['Tracked slot identity and geometry-only posthoc identity are separate.',
                  'Element fraction is not a verified H-bond/functional-group role.',
                  'Moving-target error is policy-relative, not fixed-space physical velocity.',
                  'Cumulative dose is sum of per-step RMS, not endpoint displacement.',
                  'Within-pool mode associations do not establish causal mode value.']}
    write_json(result_path,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--generated',required=True);p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=analyze(a.generated,read_json(a.bank),a.output);print({'n':v['n'],'modes':v['mode_occupancy']})
