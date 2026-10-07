"""Build compact, identity-preserving teachers from decoded terminal paths."""
import copy, gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory
from ..generation.window_reference import load_reference
from .elite_path_credit import terminal_ancestors


def choose_paths(metrics, budget=2):
    if not 1 <= budget <= 50 or metrics.valid_connected.dtype != bool:
        raise ValueError('Finite path budget and boolean validity required')
    m=metrics[metrics.valid_connected & np.isfinite(metrics.pic50_on_rescore)]
    return m.sort_values(['pic50_on_rescore','slot'],ascending=[False,True],kind='stable').drop_duplicates('smiles').head(budget)


def path_teachers(ancestors, chosen, step):
    groups={}
    for row in chosen.itertuples(index=False):
        slot=int(ancestors[step,int(row.slot)])
        item=groups.setdefault(slot,{'slot':slot,'score':float(row.pic50_on_rescore),'terminal_slots':[]})
        item['score']=max(item['score'],float(row.pic50_on_rescore));item['terminal_slots'].append(int(row.slot))
    return list(groups.values())


def build_library(dataset, graph, credit, metrics, baseline, output, budget=2):
    root,graph,credit,out=map(Path,(dataset,graph,credit,output))
    if out.exists():raise FileExistsError(out)
    gm,cm=read_json(graph/'manifest.json'),read_json(credit/'manifest.json')
    if cm['source_graph_manifest_sha256']!=digest(graph/'manifest.json') or cm['terminal_metrics_sha256']!=digest(metrics):
        raise ValueError('Credit provenance mismatch')
    ref=copy.deepcopy(load_reference(baseline))
    if ref['window']!=gm['window']:raise ValueError('Learning support mismatch')
    nodes=pd.read_parquet(graph/'nodes.parquet');m=pd.read_csv(metrics)
    cfg=read_json(root/'results'/gm['campaign']/'config.json');frames=ref['frames']
    for f in frames:
        for key in list(f):
            if key.startswith('teacher_'):del f[key]
        for key in ['teacher_endpoint_A','teacher_scores','teacher_batches','teacher_node_ids','teacher_path_ids']:
            f[key]=[]
    final_paths=[]
    for source in gm['sources']:
        path=root/source['path'];batch=source['batch']
        if digest(path)!=source['sha256']:raise ValueError('Trajectory provenance mismatch')
        chosen=choose_paths(m[m.batch==batch],budget)
        if not len(chosen):raise ValueError('No valid terminal path in donor batch')
        final_paths.extend([{'batch':batch,'slot':int(r.slot),'pic50':float(r.pic50_on_rescore),'smiles':r.smiles} for r in chosen.itertuples()])
        loc=nodes[(nodes.source_index==source['source_index'])&(nodes.representation==0)&nodes.learning_node]
        com=np.asarray(read_json(path.parent.parent.parent/f'frame_batch_{batch:03d}.json')['target_com'],float)
        with open_trajectory(path) as a:
            ancestors=terminal_ancestors(a['selected_indices'])
            for f in frames:
                at=loc[np.isclose(loc.state_time,f['time'],atol=2e-6,rtol=0)]
                if not len(at):raise ValueError('Missing exact learned teacher clock')
                step=int(at.iloc[0].step)
                cloud=np.asarray(a['predicted_coords'][step],float)*cfg['coord_scale']+com[:,None]
                if not np.asarray(a['mask'][step]).all():raise ValueError('Fixed active point clouds required')
                for teacher in path_teachers(ancestors,chosen,step):
                    slot=teacher['slot']
                    f['teacher_endpoint_A'].append(cloud[slot].tolist())
                    f['teacher_scores'].append(teacher['score']);f['teacher_batches'].append(batch)
                    f['teacher_node_ids'].append(int(at[at.slot==slot].iloc[0].node_id))
                    f['teacher_path_ids'].append([f'{batch}:{v}' for v in teacher['terminal_slots']])
    for f in frames:
        b=np.asarray(f['teacher_batches'])
        f['teacher_base_log_weight']=[float(-np.log((b==v).sum())) for v in b]
    ref.update(reference_variant='decoded-terminal-path-library-1.0',allowed_reward_views=['endpoint_pointcloud'],
        sources=gm['sources'],teacher_library={'graph_sha256':digest(graph/'manifest.json'),'credit_sha256':digest(credit/'manifest.json')},
        terminal_paths=final_paths,label_semantics=cm['label_semantics'],path_clock='Native endpoint forecast at current source state_time',
        terminal_threshold_pic50=cm['threshold_pic50'],future_semantics=cm['future_semantics'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    report={'schema_version':'decoded-terminal-path-library-build-1.0','window':gm['window'],
        'terminal_paths':len(final_paths),'unique_terminal_graphs':len(set(v['smiles'] for v in final_paths)),
        'teachers_per_time':[len(f['teacher_scores']) for f in frames],
        'teacher_score_min':min(s for f in frames for s in f['teacher_scores']),
        'teacher_score_max':max(s for f in frames for s in f['teacher_scores']),
        'reference_sha256':digest(out),'reference_bytes':out.stat().st_size,'source_code_sha256':digest(__file__),
        'change':'Terminal path labels and exact IDs only; executable endpoint pointcloud formula unchanged',
        'limitations':['Observed Steer future, not native value','All teacher paths preselected from donor panel','No claim of improvement before prospective paired inference']}
    return report
