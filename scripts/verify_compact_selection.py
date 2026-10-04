"""Same-scope equivalence and storage measurements for the selection pipeline."""
import argparse
from pathlib import Path
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from evomolsteer.io import digest,read_json,write_json
from evomolsteer.diagnostics import ArrayAudit


def normalized_bundle(value):
    if isinstance(value,dict):
        return {k:normalized_bundle(v) for k,v in value.items() if k not in ('dataset_id','path','full_catalog_path')}
    if isinstance(value,list):
        return [normalized_bundle(v) for v in value]
    return value


def total(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def verify(baseline,compact):
    baseline,compact=Path(baseline),Path(compact)
    if read_json(baseline/'selection_scope.json')!=read_json(compact/'selection_scope.json'):
        raise AssertionError('Scientific scopes differ')
    if read_json(baseline/'feature_catalog.json')!=read_json(compact/'feature_catalog.json'):
        raise AssertionError('Feature catalogs differ')
    files,batches=[],[]
    for method in ['enrichment','differential','trends','pca']:
        old,new=baseline/'discovery'/method,compact/'discovery'/method
        for name in ['effects.csv','effect_rates.csv']:
            if digest(old/name)!=digest(new/name):raise AssertionError('Changed statistic '+method+'/'+name)
            files.append(method+'/'+name)
        for stem in ['batch_effects','batch_effect_rates']:
            path=new/(stem+'.parquet')
            with pq.ParquetFile(path) as pf:
                schema=pf.schema_arrow
                if any(pa.types.is_floating(f.type) and f.type!=pa.float64() for f in schema):
                    raise AssertionError('Reduced floating precision')
                if any(pf.metadata.row_group(r).column(c).compression!='UNCOMPRESSED'
                       for r in range(pf.num_row_groups) for c in range(len(schema))):
                    raise AssertionError('Batch tables should be uncompressed')
                table=pf.read().to_pandas()
            serialized=table.to_csv(index=False,float_format='%.12g',lineterminator='\n').encode('utf-8')
            if hashlib.sha256(serialized).hexdigest()!=digest(old/(stem+'.csv')):
                raise AssertionError('Changed batch table at original CSV precision: '+method+'/'+stem)
            batches.append({'method':method,'table':stem,'rows':len(table),'float64':True,'compression':'none',
                            'original_csv_serialization_identical':True})
        for name in ['events.parquet','event_rates.parquet','batch_effects.csv','batch_effect_rates.csv']:
            if (new/name).exists():raise AssertionError('Unexpected verbose output: '+str(new/name))
    for name in ['enrichment/thresholds.csv','pca/basis.json','pca/loadings.csv','pca/explained_variance.csv']:
        if digest(baseline/'discovery'/name)!=digest(compact/'discovery'/name):raise AssertionError('Changed '+name)
        files.append(name)
    old_bundle,new_bundle=[read_json(p/'agents/evidence_bundle.json') for p in (baseline,compact)]
    if normalized_bundle(old_bundle)!=normalized_bundle(new_bundle):
        raise AssertionError('LLM evidence changed beyond source paths/dataset provenance ID')
    if digest(baseline/'agents/selection_feature_profiles.json')!=digest(compact/'agents/selection_feature_profiles.json'):
        raise AssertionError('On-demand event profiles differ')
    edge_checks=[]
    for method,filename,keys in [('trends','parent_child_rates.parquet',['representation','arm','batch']),
                                  ('pca','parent_child_velocity.parquet',['representation'])]:
        if (compact/'discovery'/method/filename).exists():raise AssertionError('Unexpected node detail')
        diag=read_json(compact/'discovery'/method/'parent_child_diagnostics.json')
        frame=pd.read_parquet(baseline/'discovery'/method/filename,columns=keys+diag['columns'])
        audit=ArrayAudit(diag['columns'])
        for label,g in frame.groupby(keys,sort=True):
            label=list(label) if isinstance(label,tuple) else label
            if method=='pca':label=label[0] if isinstance(label,list) else label
            audit.update(label,g[diag['columns']].to_numpy())
        path=compact/'verification'/('baseline_'+method+'_edge_diagnostics.json')
        audit.save(path)
        if read_json(path)!=diag:raise AssertionError('Parent-child computation changed')
        edge_checks.append({'method':method,'rows':diag['rows'],'partition_hash':diag['partition_hash'],'exact_arrays_match':True})
        del frame
    before,after=total(baseline/'discovery'),total(compact/'discovery')
    manifest=read_json(compact/'feature_cache_manifest.json')
    result={'same_scope':True,'byte_identical_formal_files':files,'batch_tables':batches,'edge_calculations':edge_checks,
        'evidence_entries':len(new_bundle['evidence']),'target_prototypes':len(new_bundle['targets']),
        'event_profiles':len(new_bundle['event_dynamics']),'llm_evidence_exact_except_provenance':True,
        'formal_before_bytes':before,'formal_after_bytes':after,'formal_reduction_percent':100*(1-after/before),
        'cache_bytes':manifest['bytes'],'cache_sha256':manifest['sha256'],
        'scope':new_bundle['scope'],'passed':True}
    write_json(compact/'verification/output_equivalence.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--compact',required=True)
    a=p.parse_args();r=verify(a.baseline,a.compact)
    print('Formal files:',len(r['byte_identical_formal_files']),'batch tables:',len(r['batch_tables']))
    print('Formal bytes:',r['formal_before_bytes'],'->',r['formal_after_bytes'])
    print('LLM evidence / prototypes / profiles:',r['evidence_entries'],r['target_prototypes'],r['event_profiles'])
