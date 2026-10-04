"""Import verified existing feature rows to evaluate a new output policy.

Avoids recalculating geometry for the large policy trial. The independent real
fixture run exercises extraction itself. Never changes the source analysis.
"""
import argparse
from pathlib import Path
import shutil
import pyarrow.parquet as pq
from evomolsteer.io import digest,read_json,write_json
from evomolsteer.output_policy import extraction_representations,parquet_options,register_feature_cache


def prepare(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    cfg=read_json(source/'config.json')
    if cfg.get('analysis_scope')!='actual_resampling_events':
        raise ValueError('This importer only accepts the selection-window pipeline')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    cfg['output_policy']={'profile':'compact','retain_feature_cache':False,'keep_worker_shards':False,
                          'write_node_details':False,'write_step_details':False}
    selected=extraction_representations(cfg)
    reader=pq.ParquetFile(source/'features.parquet');groups=[]
    partial=output/'features.parquet.partial'
    with pq.ParquetWriter(partial,reader.schema_arrow,**parquet_options(cfg,reader.schema_arrow)) as writer:
        for i in range(reader.num_row_groups):
            reps=reader.read_row_group(i,columns=['representation'])['representation'].unique().to_pylist()
            if len(reps)!=1:raise ValueError('Expected one representation per row group')
            if reps[0] in selected:
                writer.write_table(reader.read_row_group(i));groups.append(i)
    partial.replace(output/'features.parquet')
    with pq.ParquetFile(output/'features.parquet') as returned:
        for j,i in enumerate(groups):
            if not reader.read_row_group(i).equals(returned.read_row_group(j),check_metadata=True):
                raise AssertionError('Imported source values/schema differ')
        rows=returned.metadata.num_rows
    sources={'features.parquet':digest(source/'features.parquet')}
    for name in ['edges.parquet','selection_events.parquet','selection_scope.json','feature_catalog.json']:
        shutil.copyfile(source/name,output/name);sources[name]=digest(source/name)
    write_json(output/'config.json',cfg)
    manifest=read_json(source/'ingest_manifest.json')
    manifest['config_sha256']=digest(output/'config.json')
    manifest['feature_cache_import']={'source':str(source),'sha256':sources,'source_row_groups':groups,
                                      'verified_rows':rows,'values_and_schema_equal':True,
                                      'note':'Geometry reused without refitting or rounding; new policy applied only to output retention'}
    write_json(output/'ingest_manifest.json',manifest)
    register_feature_cache(output,cfg)
    print('Imported verified rows:',rows,'cache bytes:',(output/'features.parquet').stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.source,a.output)
