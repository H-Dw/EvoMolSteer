"""Independent batch shards; bounded workers and canonical deterministic merge."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from .io import read_json,write_json,write_table,digest
from .output_policy import policy,parquet_options,register_feature_cache,clean_generated_shards
from .trajectory_source import trajectory_paths

def extract_batch(root,destination,cfg,batch):
    from .ingest import ingest
    shard_cfg={**cfg,'ingest_workers':1,'include_batches':[batch]}
    ingest(root,destination,shard_cfg)
    return batch

def run(root,output,cfg):
    root,output=Path(root).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    campaign=root/'results'/cfg['campaign']
    batches=sorted({int(p.parent.name.split('_')[1]) for p in trajectory_paths(campaign) if (p.parent/'COMPLETE.json').exists()})
    if not batches:raise ValueError('No completed batches')
    workers=min(int(cfg['ingest_workers']),len(batches))
    shards={b:output/'_shards'/f'batch_{b:03d}' for b in batches}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(extract_batch,str(root),str(shards[b]),cfg,b) for b in batches]
        for n,future in enumerate(as_completed(tasks),1):
            batch=future.result();print(f'Audited batch {batch:03d}; {n}/{len(batches)} independent batches complete',flush=True)
    # Each shard validates matched initial priors across all arms of its batch.
    catalogs=[read_json(shards[b]/'feature_catalog.json') for b in batches]
    if any(c!=catalogs[0] for c in catalogs[1:]):raise ValueError('Feature catalogs differ across shards')
    entries=[];readers={b:pq.ParquetFile(shards[b]/'features.parquet') for b in batches}
    order={'current_state':0,'predicted_endpoint':1,'proposal_state':2}
    for b,reader in readers.items():
        names=reader.schema_arrow.names
        for i in range(reader.num_row_groups):
            meta=reader.metadata.row_group(i)
            arm=meta.column(names.index('arm')).statistics.min
            rep=meta.column(names.index('representation')).statistics.min
            entries.append((arm,b,order[rep],i))
    partial=output/'features.parquet.partial'
    with pq.ParquetWriter(partial,readers[batches[0]].schema_arrow,**parquet_options(cfg,readers[batches[0]].schema_arrow)) as writer:
        for arm,b,rep,i in sorted(entries):writer.write_table(readers[b].read_row_group(i))
    partial.replace(output/'features.parquet')
    expected_rows=sum(reader.metadata.num_rows for reader in readers.values())
    with pq.ParquetFile(output/'features.parquet') as merged_reader:
        if merged_reader.metadata.num_rows!=expected_rows:raise ValueError('Merged feature row count mismatch')
    for reader in readers.values():reader.close()
    for name,keys in [('edges',['arm','batch','step','target']),('selection_events',['arm','batch','step'])]:
        frame=pd.concat([pd.read_parquet(shards[b]/(name+'.parquet')) for b in batches],ignore_index=True)
        write_table(output/(name+'.parquet'),frame.sort_values(keys,kind='stable').reset_index(drop=True))
    manifests=[read_json(shards[b]/'ingest_manifest.json') for b in batches]
    scopes=[read_json(shards[b]/'selection_scope.json') for b in batches]
    if any(s!=scopes[0] for s in scopes[1:]):raise ValueError('Selection scopes differ across shards')
    write_json(output/'selection_scope.json',scopes[0])
    merged={**manifests[0]}
    for key in ['trajectory_sha256','source_sha256']:merged[key]={k:v for m in manifests for k,v in m[key].items()}
    merged['audits']=sorted([r for m in manifests for r in m['audits']],key=lambda r:(r['arm'],r['batch']))
    write_json(output/'config.json',cfg);merged['config_sha256']=digest(output/'config.json')
    keep=policy(cfg).keep_worker_shards
    shard_hashes={str(p.relative_to(output)):digest(p) for p in sorted((output/'_shards').rglob('*')) if p.is_file()}
    merged['ingestion']={'workers':workers,'merge_order':'arm,batch,current/predicted/proposal,step,slot','shards_preserved':keep,'shard_sha256':shard_hashes}
    write_json(output/'feature_catalog.json',catalogs[0]);write_json(output/'ingest_manifest.json',merged)
    register_feature_cache(output,cfg)
    if not keep:clean_generated_shards(output,shard_hashes)
    return merged['audits']
