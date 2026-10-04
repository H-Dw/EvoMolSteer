"""Extract actual selection events and time-matched background, without outcomes."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .io import read_json,write_json,write_table,digest
from .geometry import build_catalog,measure
from .scope import SCOPE,REPRESENTATIONS,discover_scope,stage_for
from .output_policy import parquet_options,register_feature_cache,extraction_representations
from .trajectory_source import trajectory_paths,open_trajectory,analysis_arrays,validate_input_bundle


def split_for(batch,cfg):
    for name in ('discovery','validation','heldout'):
        if batch in cfg[name+'_batches']:
            return name
    raise ValueError(f'Batch {batch} has no configured split')


def ingest(root,output,cfg):
    if cfg.get('analysis_scope')!=SCOPE:
        raise ValueError('Only selection-window extraction is supported')
    extraction_representations(cfg)
    if cfg.get('ingest_workers',1)>1 and 'include_batches' not in cfg:
        from .parallel_ingest import run
        return run(root,output,cfg)
    root,output = Path(root),Path(output)
    validate_input_bundle(root,cfg)
    camp = root/'results'/cfg['campaign']
    scope = discover_scope(camp,cfg)
    steps = np.asarray(scope['steps'],int)
    config = read_json(camp/'config.json')
    catalog = build_catalog(root,cfg,config['atom_vocabulary'])
    output.mkdir(parents=True,exist_ok=True)
    writer = None
    edges,events,audits,priors,files = [],[],[],{},{}
    sources = {str(p.relative_to(root)):digest(p) for p in sorted((root/'inputs').glob('*')) if p.is_file()}
    if (root/'input_bundle_manifest.json').exists():
        sources['input_bundle_manifest.json'] = digest(root/'input_bundle_manifest.json')
    sources[str((camp/'config.json').relative_to(root))] = digest(camp/'config.json')
    for path in trajectory_paths(camp):
        bdir = path.parent
        batch,arm = int(bdir.name.split('_')[1]),bdir.parent.name
        if arm not in cfg['selection_arms']+[cfg['background_arm']] or not (bdir/'COMPLETE.json').exists():
            continue
        if cfg.get('include_batches') is not None and batch not in cfg['include_batches']:
            continue
        with open_trajectory(path) as archive:
            z = analysis_arrays(archive)
        T,B = z['pic50_on'].shape
        N = z['predicted_coords'].shape[2]
        times = z['score_time'][:,0].astype(float)
        ids = z['selected_indices'].astype(int)
        counts,probs = z['offspring_count'],z['selection_probability']
        if not np.all(np.diff(times)>0) or not np.all(z['step_size']>0):
            raise ValueError('Nonmonotonic time grid')
        if np.any(ids<0) or np.any(ids>=B):
            raise ValueError('Invalid selected indices')
        if not np.allclose(times[steps],scope['score_times'],rtol=0,atol=1e-7):
            raise ValueError('Background and selection times disagree')
        actual = np.flatnonzero(z['resampled'])
        if arm in cfg['selection_arms'] and not np.array_equal(actual,steps):
            raise ValueError('Selection schedule changed')
        if arm==cfg['background_arm'] and len(actual):
            raise ValueError('Background resampled')
        if batch in priors and not np.array_equal(priors[batch],z['current_coords'][0]):
            raise ValueError('Initial priors are not matched across arms')
        priors[batch] = z['current_coords'][0].copy()
        frame_path = camp/f'frame_batch_{batch:03d}.json'
        com = np.array(read_json(frame_path)['target_com']).reshape(B,3)
        sources[str(frame_path.relative_to(root))] = digest(frame_path)
        split = split_for(batch,cfg)
        tag = f'{cfg["campaign"]}:{arm}_b{batch:03d}'
        node = lambda t,s:f'{tag}_t{t:03d}_s{s:03d}'
        common = []
        stages = stage_for(times[steps],scope['stage_edges'])
        for position,t in enumerate(steps):
            if not np.array_equal(counts[t],np.bincount(ids[t],minlength=B)):
                raise ValueError('Offspring counts disagree')
            for weight in ('selection_probability','weight_on','weight_off'):
                if not np.isfinite(z[weight][t]).all() or np.any(z[weight][t]<0) or not np.isclose(z[weight][t].sum(),1):
                    raise ValueError('Invalid probability')
            if t<T-1:
                for key in ('coords','atomics','bonds','charges'):
                    if not np.array_equal(z['current_'+key][t+1],z['proposal_'+key][t,ids[t]]):
                        raise ValueError('State ancestry mismatch: '+key)
                if not np.array_equal(z['parent_slot'][t+1],ids[t]) or not np.array_equal(z['root_slot'][t+1],z['root_slot'][t,ids[t]]):
                    raise ValueError('Root/parent mismatch')
            for s in range(B):
                common.append({'node_id':node(t,s),
                    'parent_node_id':node(t-1,int(z['parent_slot'][t,s])) if t and t-1 in steps else '',
                    'campaign':cfg['campaign'],'arm':arm,'batch':batch,'split':split,
                    'root_id':f'{cfg["campaign"]}:b{batch:03d}:root{int(z["root_slot"][t,s]):03d}',
                    'step':int(t),'slot':s,'score_time':times[t],
                    'state_time':float(z['state_time'][t]),'dt':float(z['step_size'][t]),
                    'stage':int(stages[position]),'resampled':bool(z['resampled'][t]),
                    'selected':bool(counts[t,s]>0),'offspring_count':int(counts[t,s]),
                    'probability':float(probs[t,s]),'weight_on':float(z['weight_on'][t,s]),
                    'weight_off':float(z['weight_off'][t,s]),'pic50_on':float(z['pic50_on'][t,s]),
                    'pic50_off':float(z['pic50_off'][t,s])})
            events.append({'arm':arm,'batch':batch,'split':split,'step':int(t),
                'stage':int(stages[position]),'score_time':times[t],'state_time':float(z['state_time'][t]),
                'resampled':bool(z['resampled'][t]),'population':B,
                'n_selected':int((counts[t]>0).sum()),'n_rejected':int((counts[t]==0).sum()),
                'n_roots':len(np.unique(z['root_slot'][t])),'ess':float(1/np.square(probs[t]).sum())})
            if t+1 in steps:
                for s in range(B):
                    edges.append({'source':node(t,int(ids[t,s])),'target':node(t+1,s),
                        'arm':arm,'batch':batch,'step':int(t),'stage':int(stages[position]),'dt':times[t+1]-times[t]})
        for representation in cfg['analysis_representations']:
            prefix = REPRESENTATIONS[representation]
            xyz = z[prefix+'_coords'][steps].astype(float)*config['coord_scale']+com[None,:,None,:]
            values = measure(xyz.reshape(-1,N,3),z[prefix+'_atomics'][steps].reshape(-1,N),z['mask'][steps].reshape(-1,N),catalog)
            df = pd.DataFrame(common)
            df['representation'] = representation
            df['geometry_time'] = df.state_time if prefix=='proposal' else df.score_time
            table = pa.Table.from_pandas(pd.concat([df,pd.DataFrame(values)],axis=1),preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output/'features.parquet.partial',table.schema,**parquet_options(cfg,table.schema))
            writer.write_table(table)
        files[str(path.relative_to(root))] = digest(path)
        audits.append({'arm':arm,'batch':batch,'T':T,'B':B,'analyzed_steps':steps.tolist(),
                       'resampling_times':times[actual].tolist(),'ancestry':'passed'})
    if writer is None:
        raise ValueError('No completed batches')
    writer.close()
    (output/'features.parquet.partial').replace(output/'features.parquet')
    write_table(output/'edges.parquet',edges)
    write_table(output/'selection_events.parquet',events)
    write_json(output/'selection_scope.json',scope)
    write_json(output/'feature_catalog.json',catalog)
    write_json(output/'config.json',cfg)
    write_json(output/'ingest_manifest.json',{'source_root':str(root.resolve()),'campaign':cfg['campaign'],
        'analysis_scope':SCOPE,'trajectory_sha256':files,'source_sha256':sources,'audits':audits,
        'config_sha256':digest(output/'config.json'),'notes':[
            'No terminal, descendant, post-window or counterfactual outcomes consumed.',
            'Background includes the same event steps but is never labeled a selected branch.',
            'Proposal geometry is after integration; scope follows the score/selection event.']})
    register_feature_cache(output,cfg)
    return audits
