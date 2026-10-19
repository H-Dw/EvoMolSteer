"""Whole-window local enrichment of saved endpoint forecasts, not raw states."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from evomolsteer.local_motif.state_controls import full_state_controls
from .bank import lookup


def calibration_summary(profile,floors):
    rows=[r for r in profile['rows'] if r['unique_controls']>0];times=np.array([r['time'] for r in rows])
    mu=np.asarray([r['focal'] for r in rows]);bg=np.asarray([r['background_mean'] for r in rows])
    sd=np.maximum(np.asarray([r['background_sd'] for r in rows]),floors)
    strength=np.sqrt(np.square((mu-bg)/sd).mean(1));gain=strength/np.sqrt(1+strength**2)
    ids=[0,len(times)//2,-1];coef=np.polynomial.polynomial.polyfit(times,strength,2)
    return {'representation':profile.get('feature_representation','current'),'density_scale_floor':floors,
        'contrast_min':float(strength.min()),'contrast_max':float(strength.max()),
        'gain_first_mid_last':gain[ids].tolist(),'times_first_mid_last':times[ids].tolist(),
        'full_window_contrast_coefficients':coef.tolist(),'derivative_coefficients':(coef[1:]*[1,2]).tolist(),
        'meaning':'Standardized descriptive separation; neither calibrated future-success probability nor significance confidence.'}


def evidence(repo,m,bank,region):
    mode=bank['modes'][region['template_id']];slots=region['atom_slots'];base=repo/m['compact_dataset']/'results'/m['campaign']
    source=base/m['arm']/f"batch_{mode['batch']:03d}"/'trajectory.h5'
    from .evidence_cache import cache_path
    path,cache_key=cache_path(repo,m,mode,region,source,'endpoint_prediction',__file__)
    if path.exists():
        cached=read_json(path)
        if cached.get('cache_key')!=cache_key:raise ValueError('Forecast evidence cache binding changed')
        return cached
    com=np.asarray(read_json(base/f"frame_batch_{mode['batch']:03d}.json")['target_com']).reshape(-1,1,3)
    scale=read_json(base/'config.json')['coord_scale'];centers=np.asarray(mode['terminal_coords_A'])[slots];rows=[]
    with TrajectoryPackage(source) as tr:
        times=tr.read('score_time')[:,0];state_times=tr.read('state_time');anc=terminal_ancestors(tr.read('selected_indices').astype(int))
        for step,t in enumerate(times):
            if not m['learning_window'][0]-2e-6<=t<=m['learning_window'][1]+2e-6:continue
            current=tr.read('current_coords',step)*scale+com
            unique=full_state_controls(tr,step,current,tr.read('current_atomics',step),int(anc[step,mode['terminal_slot']]))
            clouds=tr.read('predicted_coords',step)*scale+com;target=lookup(mode,float(t),'endpoint_A')[slots];values=[]
            for cloud in clouds:
                rr,cc=linear_sum_assignment(np.square(cloud[:,None]-target).sum(-1));delta=cloud[rr[np.argsort(cc)]]-centers
                kernel=np.exp(-np.square(delta).sum(-1)/(2*m['field_sigma_A']**2))
                values.append(np.r_[kernel.mean(),(kernel[:,None]*delta).mean(0)])
            phi=np.asarray(values);parent=int(anc[step,mode['terminal_slot']]);bg=phi[unique]
            row={'time':float(t),'proposal_state_time':float(state_times[step]),'focal':phi[parent].tolist(),
                'background_mean':bg.mean(0).tolist() if len(bg) else [None]*4,
                'background_sd':bg.std(0).tolist() if len(bg) else [None]*4,'unique_controls':len(unique)}
            if state_times[step]<=m['learning_window'][1]+2e-6 and tr.read('resampled',step):
                w=tr.read('selection_probability',step).astype(float);w/=w.sum()
                row.update(expected_selection_prediction_shift=(w@phi-phi.mean(0)).tolist(),
                    realized_selection_prediction_shift=(phi[tr.read('selected_indices',step).astype(int)].mean(0)-phi.mean(0)).tolist())
            rows.append(row)
    fits={}
    for key in ('focal','background_mean','expected_selection_prediction_shift'):
        valid=[r for r in rows if key in r and all(v is not None for v in r[key])]
        if len(valid)<3:continue
        tt=np.asarray([r['time'] for r in valid]);yy=np.asarray([r[key] for r in valid]);c=np.polynomial.polynomial.polyfit(tt,yy,2)
        fits[key]={'coefficients':c.tolist(),'derivative_coefficients':(c[1:]*np.arange(1,3)[:,None]).tolist(),
            'rmse':np.sqrt(np.square(np.polynomial.polynomial.polyval(tt,c).T-yy).mean(0)).tolist()}
    result={'schema_version':2,'cache_key':cache_key,'cache_reference':path.relative_to(repo).as_posix(),
        'feature_representation':'endpoint_prediction','region_id':region['region_id'],
        'source_sha256':digest(source),'domain':m['learning_window'],'feature_centers_A':centers.tolist(),
        'rows':rows,'full_window_fits':fits,'focal_feature_range':np.ptp(np.asarray([r['focal'] for r in rows]),axis=0).tolist(),
        'feature_definition':'Kernel occupancy and signed displacement to fixed terminal centers; matching and coordinates both use the endpoint prediction made at this same forward time.',
        'units':['dimensionless','A','A','A'],
        'limits':['Unique full CURRENT states define comparison units, not distinct projections or independent final outcomes.',
            'Prediction enrichment is weighted by subsequent proposal selection; it is NOT proposal geometry enrichment or native physical motion.',
            'No extra model inference and no calibrated future-success probability.']}
    write_json(path,result);return result


def main():
    import argparse
    from pathlib import Path
    from .bank import regional_evidence
    parser=argparse.ArgumentParser();parser.add_argument('--repo',default='.');parser.add_argument('--config',required=True)
    parser.add_argument('--region',required=True);parser.add_argument('--representation',choices=['current','endpoint_prediction'],required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--density-floor',type=float,nargs=4,help='Four feature scales; defaults to the frozen reference program, not a new fitted value.')
    args=parser.parse_args();repo=Path(args.repo).resolve();m=read_json(args.config);root=repo/m['campaign_study'];bank=read_json(root/'bank.json')
    region=next(r for r in read_json(root/'regions.json') if r['region_id']==args.region)
    profile=(evidence if args.representation=='endpoint_prediction' else regional_evidence)(repo,m,bank,region)
    floors=args.density_floor or read_json(root/'state.json')['reference']['program']['density_scale_floor']
    write_json(Path(args.output),{'region_id':args.region,'profile':profile,'calibration':calibration_summary(profile,floors)})


if __name__=='__main__':main()
