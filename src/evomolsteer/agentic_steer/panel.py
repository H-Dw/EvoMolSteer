"""Prepare blinded state panels and terminal-conditioned historical contrasts.

Immutable coordinate locators replace copied trajectories. Extinction remains
censored. Discovery terminal labels never enter exploration candidate cards.
"""
from pathlib import Path
import hashlib
import json
import inspect

import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.continuous.affinity_geometry import landmarks, names, numpy_geometry
from evomolsteer.continuous.elite_path_credit import terminal_ancestors


def selected_features(all_names, landmark_indices):
    return [x for x in all_names if x.startswith(('global_', 'shape_', 'pair_'))
            or any(x.startswith(f'landmark_{j:02d}_') for j in landmark_indices)]


def load_metrics(path):
    f = pd.read_parquet(path)
    return f[f.arm.eq('single')].copy() if 'arm' in f else f


def candidate_indices(features, scores, count):
    """No terminal label: cover current score ranks, then geometric diversity."""
    rank = np.argsort(-scores, kind='stable')
    chosen = list(dict.fromkeys([int(rank[0]), int(rank[len(rank)//2]), int(rank[-1])]))
    scale = np.std(features, axis=0).clip(1e-5)
    z = (features-features.mean(0))/scale
    while len(chosen) < count:
        distance = np.linalg.norm(z[:, None]-z[chosen][None], axis=2).min(1)
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    return chosen[:count]


def make_panel(repo, config_path):
    repo = Path(repo).resolve(); cfg = read_json(config_path)
    out = repo/cfg['study']; out.mkdir(parents=True, exist_ok=True)
    raw = repo/cfg['dataset']; compact = repo/cfg['compact_dataset']
    campaign = cfg['campaign']; step = cfg['start_step']
    source = compact/'results'/campaign
    input_dir = raw/'inputs'
    points, origin = landmarks(input_dir/cfg['protein'], input_dir/cfg['ligand'])
    fields = selected_features(names(points), cfg['landmark_indices'])
    columns = [names(points).index(x) for x in fields]
    metrics = load_metrics(repo/cfg['terminal_metrics'])
    valid = metrics.valid_connected.astype(bool)
    discovery = metrics[metrics.batch.isin(cfg['discovery_batches']) & valid]
    threshold = float(discovery.pic50_on_rescore.quantile(cfg['elite_quantile']))
    candidate_rows, contrasts, locators, sources, paths = [], [], {}, {}, []
    for batch in cfg['discovery_batches']+cfg['exploration_batches']:
        path = source/'single'/f'batch_{batch:03d}'/'trajectory.h5'
        frame = read_json(source/f'frame_batch_{batch:03d}.json')
        com = np.asarray(frame['target_com'])[:, None]
        scale = read_json(source/'config.json')['coord_scale']
        with TrajectoryPackage(path) as tr:
            times = tr.read('score_time')[:, 0]
            if not cfg['window'][0]-1e-6 <= times[step] <= cfg['window'][1]+1e-6:
                raise ValueError('Start node outside declared observation support')
            y = tr.read('predicted_coords', step)*scale+com
            x = tr.read('current_coords', step)*scale+com
            previous = max(0, step-1)
            yp = tr.read('predicted_coords', previous)*scale+com
            dt = float(times[step]-times[previous])
            fy = numpy_geometry(y, points, origin)[:, columns]
            fx = numpy_geometry(x, points, origin)[:, columns]
            # The previous forecast must follow the actual selected parent,
            # not the same array slot after resampling.
            parents = tr.read('selected_indices', previous).astype(int)
            derivative = (fy-numpy_geometry(yp, points, origin)[parents][:, columns])/dt if dt else np.zeros_like(fy)
            head = tr.read('pic50_on', step).reshape(-1)
            roots = tr.read('root_slot', step)
            if batch in cfg['discovery_batches']:
                ancestor = terminal_ancestors(tr.read('selected_indices'))
                group = metrics[metrics.batch.eq(batch)]
                quality = dict(zip(group.slot.astype(int), group.valid_connected.astype(bool)))
                score = dict(zip(group.slot.astype(int), group.pic50_on_rescore.astype(float)))
                observed = {int(a):[] for a in ancestor[step]}
                for final_slot,a in enumerate(ancestor[step]):
                    if quality.get(final_slot,False):observed[int(a)].append(score[final_slot])
                high = [a for a, values in observed.items() if values and max(values)>=threshold]
                ordinary = [a for a, values in observed.items() if values and max(values)<threshold]
                if high and ordinary:
                    contrasts.append({'source_id':f'discovery/{batch}', 'batch':batch,
                        'elite_ancestors':len(high),'ordinary_ancestors':len(ordinary),
                        'features':dict(zip(fields, (fy[high].mean(0)-fy[ordinary].mean(0)).tolist())),
                        'signed_rate_contrast':dict(zip(fields,(derivative[high].mean(0)-derivative[ordinary].mean(0)).tolist()))})
                # Multiple true terminal paths, including measured ordinary
                # descendants. Select distinct current ancestors within each
                # stratum, rather than cloning one terminal winner's path.
                for label, candidates in [('elite',high),('ordinary',ordinary)]:
                    chosen=[]
                    for final_slot in sorted(score,key=lambda s:-score[s]):
                        a=int(ancestor[step,final_slot])
                        if a not in candidates or a in chosen or not quality.get(final_slot,False):continue
                        chosen.append(a)
                        profile=[]
                        for s in sorted(set(round(t*cfg['steps']) for t in np.linspace(*cfg['window'],6))):
                            if not cfg['window'][0]-1e-6<=times[s]<=cfg['window'][1]+1e-6:continue
                            slot_s=int(ancestor[s,final_slot])
                            yz=tr.read('predicted_coords',s)[slot_s:slot_s+1]*scale+com[slot_s:slot_s+1]
                            profile.append({'time':float(times[s]),'features':dict(zip(fields,map(float,numpy_geometry(yz,points,origin)[0,columns])))})
                        paths.append({'source_id':f'path/{batch}/{final_slot}','stratum':label,
                            'final_pic50':score[final_slot],'observed_valid_terminal':True,
                            'current_ancestor':a,'profiles':profile})
                        if len(chosen)>=2:break
                continue
            for slot in candidate_indices(fy,head,cfg['candidates_per_batch']):
                node=f'N{len(candidate_rows):03d}'
                vector=dict(zip(fields,map(float,fy[slot])))
                vector.update({'rate/'+k:float(v) for k,v in zip(fields,derivative[slot])})
                row={'node_id':node,'source_id':f'candidate/{node}','state_time':float(times[step]),
                    'joint_head_pic50':float(head[slot]),'features':vector,
                    'current_Rg_A':float(np.sqrt(np.square(x[slot]-x[slot].mean(0)).sum(1).mean())),
                    'forecast_Rg_A':float(np.sqrt(np.square(y[slot]-y[slot].mean(0)).sum(1).mean())),
                    'future_label':'unknown; deliberately withheld',
                    'forecast_hard_atom_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(tr.read('predicted_atomics',step)[slot],return_counts=True))}}
                candidate_rows.append(row)
                restart=raw/'results'/campaign/'single'/f'batch_{batch:03d}'/f'restart_step_{step:03d}.pt.gz'
                initial=restart.parent/'initial_state.pt.gz'
                if not restart.exists() or not initial.exists():raise FileNotFoundError('Complete restart state required')
                locators[node]={'batch':int(batch),'slot':int(slot),'step':step,'representation':'current',
                    'root_slot':int(roots[slot]),'restart_sha256':digest(restart),'initial_sha256':digest(initial),
                    'expected_forecast_A':y[slot].tolist(),'expected_joint_head':float(head[slot])}
                sources[node]={'trajectory_sha256':digest(path),'restart_sha256':digest(restart)}
    standard_fields=fields+['rate/'+k for k in fields]
    matrix=np.asarray([[r['features'][k] for k in standard_fields] for r in candidate_rows])
    center=matrix.mean(0); sd=matrix.std(0).clip(1e-5)
    for r in candidate_rows:
        r['standardized_features']={k:float((r['features'][k]-m)/s) for k,m,s in zip(standard_fields,center,sd)}
    region_atoms=[]
    for j in cfg['landmark_indices']:
        best=None
        for line in (input_dir/cfg['protein']).read_text().splitlines():
            if line.startswith('ATOM'):
                position=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]); distance=float(np.linalg.norm(position-points[j]))
                if best is None or distance<best[0]:best=(distance,line[12:16].strip(),line[17:20].strip(),line[21:26].strip())
        region_atoms.append({'landmark':j,'position_A':points[j].tolist(),'nearest_receptor_atom':best[1:],
            'interpretation':'Geometric anchor; atom name alone does not establish a binding interaction'})
    packet={'schema_version':'agentic-steer-panel-1.0','objective':{'primary':'Final predicted target pIC50',
        'qualified_elite':'valid connected, no severe receptor overlap, and pic50 >= frozen discovery quantile','threshold':threshold,
        'secondary':['strain','structural validity'],'binding_truth':'Model predictions, not experimental affinities'},
        'window':cfg['window'],'score_time':float(times[step]),'candidate_cards':candidate_rows,
        'discovery_contrasts':contrasts,'terminal_paths':paths,'pocket_context':region_atoms,
        'selection_code':inspect.getsource(candidate_indices),
        'definitions':{'forecast':'Joint model endpoint prediction, not decoded final molecule',
            'rate':'Signed forecast feature difference along selected parent divided by actual dt',
            'censoring':'No observed descendant is unknown, never a failed final affinity label',
            'scale':'Feature-standardization uses only the current candidate panel, without terminal labels',
            'selection_code':'Initial numeric baseline sorts current head; adaptive baseline uses measured native continuation results.'},
        'limitations':['Sparse Steer-conditioned historical ancestry; not calibrated native success probabilities',
            'Exploration batches used in earlier project work; new suffix RNG outcomes are not yet observed',
            'Finite trial budget cannot establish an optimal direction',
            'Current and forecast features must not be mixed as the same physical representation']}
    write_json(out/'panel.json',packet);write_json(out/'locators.json',locators)
    write_json(out/'input_manifest.json',{'config_sha256':digest(config_path),'sources':sources,
        'terminal_metrics_sha256':digest(repo/cfg['terminal_metrics']),
        'exploration_terminal_labels_in_packet':False,'candidate_count':len(candidate_rows)})
    return packet


def trial_seed(master, node, replicate):
    return int.from_bytes(hashlib.sha256(f'{master}/{node}/{replicate}'.encode()).digest()[:4],'big')%(2**31-1)


def make_job(repo, cfg, requests, destination):
    study=Path(repo)/cfg['study']; locators=read_json(study/'locators.json')
    jobs=[]
    for node,replicate in requests:
        if node not in locators or not 0<=replicate<cfg['max_replicates']:raise ValueError('Unregistered trial')
        jobs.append({'trial_id':f'{node}_r{replicate:02d}','node_id':node,'replicate':replicate,
            'rng_seed':trial_seed(cfg['seed'],node,replicate),**locators[node]})
    if len({j['trial_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate trial identity')
    write_json(destination,{'schema_version':'agentic-native-suffix-1.0','steps':cfg['steps'],
        'master_seed':cfg['seed'],'checkpoint':cfg['remote_checkpoint'],'flowr_root':cfg['remote_flowr_root'],
        'dataset':cfg['remote_dataset'],'campaign':cfg['campaign'],'output':cfg['remote_output'],
        'coord_noise_level':cfg['coord_noise_level'],
        'inputs':{k:cfg[k] for k in ['protein','ligand','off_protein','off_ligand']},
        'expected_forecast_rms_tolerance_A':.002,
        'threshold':read_json(study/'panel.json')['objective']['threshold'],'trials':jobs})
    return jobs
