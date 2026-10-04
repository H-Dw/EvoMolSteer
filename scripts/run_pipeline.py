"""Sequential reproducible pipeline; LLM transport is an explicit separate step."""
import argparse
import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.ingest import ingest
from evomolsteer.evidence import build_bundle
from evomolsteer.agents import export_request
from evomolsteer.output_policy import policy,retire_feature_cache

if __name__=='__main__':
    project=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(project/'configs/default.json'))
    p.add_argument('--skip-ingest',action='store_true')
    a=p.parse_args();out=Path(a.output);cfg=read_json(a.config)
    if cfg.get('time_analysis')=='continuous_window' and not a.skip_ingest and (out/'config.json').exists():
        raise FileExistsError('Continuous analysis already extracted here; use a fresh output or --skip-ingest with the same extraction config')
    source={str(f.relative_to(project)):digest(f) for folder in ['src','scripts','skills','prompts','schemas','configs']
            for f in sorted((project/folder).rglob('*')) if f.is_file() and '__pycache__' not in str(f)}
    runtime={'python':platform.python_version(),'platform':platform.platform(),
             'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','pyarrow','torch','jsonschema','httpx']}}
    write_json(out/'run_provenance.json',{'code_sha256':source,'runtime':runtime,'config':cfg,'status':'running'})
    def record_failure(kind,value,traceback):
        write_json(out/'run_provenance.json',{'code_sha256':source,'runtime':runtime,'config':cfg,
            'status':'failed','error':str(value)})
        sys.__excepthook__(kind,value,traceback)
    sys.excepthook=record_failure
    if a.skip_ingest:
        manifest=read_json(out/'ingest_manifest.json')
        old_cfg=read_json(out/'config.json')
        if policy(old_cfg).profile!=policy(cfg).profile:
            raise ValueError('Changing the cache encoding profile requires a fresh output directory')
        geometry_keys=['analysis_scope','campaign','analysis_representations','selection_arms','background_arm','stage_width',
                       'time_analysis','feature_schema','feature_pockets','bond_order_by_label',
                       'pocket_radius_A','softmin_temperature_A','contact_midpoint_A','contact_width_A',
                       'discovery_batches','validation_batches','heldout_batches','include_batches']
        if Path(manifest['source_root']).resolve()!=Path(a.root).resolve() or any(old_cfg.get(k)!=cfg.get(k) for k in geometry_keys):
            raise ValueError('Existing extraction does not match root/extraction config')
        if old_cfg!=cfg:
            manifest['previous_extraction_config_sha256']=manifest['config_sha256']
            write_json(out/'config.json',cfg)
            manifest['config_sha256']=digest(out/'config.json')
            manifest['analysis_config_updated_without_reextracting']=True
            write_json(out/'ingest_manifest.json',manifest)
    else:
        print('Extracting actual selection events and matched controls',flush=True);ingest(a.root,out,cfg)
    if cfg.get('time_analysis')=='continuous_window':
        from evomolsteer.continuous.pipeline import run
        run(out)
    else:
        for name in ['enrichment','differential','trends','pca']:
            print('Analyzing '+name,flush=True);importlib.import_module('evomolsteer.'+name).run(out,'discovery')
        print('Building discovery evidence and Analyst request',flush=True)
        build_bundle(out);export_request(out,'Analyst')
        from evomolsteer.reporting import summarize
        summarize(out)
    retired=retire_feature_cache(out,cfg)
    write_json(out/'output_policy_report.json',{'policy':vars(policy(cfg)),'feature_cache':retired,
        'complete_statistics_retained':True,'significance_filter_applied':False,
        'node_detail_regeneration':'Use audit output_policy in a separate output directory with the same scientific config',
        'feature_cache_regeneration':'scripts/materialize_feature_cache.py --analysis THIS_RUN'})
    write_json(out/'run_provenance.json',{'code_sha256':source,'runtime':runtime,'config':cfg,'status':'complete',
        'output_sha256':{str(f.relative_to(out)):digest(f) for split in ['discovery','validation','heldout','agents']
                         for f in sorted((out/split).rglob('*')) if f.is_file()}})
    print('Complete; no external LLM API called',flush=True)
