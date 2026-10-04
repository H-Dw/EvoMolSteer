"""Freeze the two declared CK2 hypotheses from discovery evidence, without fitting."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.analysis);out=Path(a.output)
    # Never open validation/heldout caches here.
    bundle=read_json(root/'agents/evidence_bundle.json')
    assert bundle['split']=='discovery' and bundle['evidence_arm']=='single'
    specifications=[('region','ck2:A:VAL116::distance_softmin',0),('compact','ligand::radius_gyration',1)]
    terms=[]
    for name,feature,stage in specifications:
        target=next(t for t in bundle['targets'] if t['feature']==feature and t['stage']==stage)
        terms.append({'id':name,'feature':feature,'operator':'decrease_until_target',
            'target':target['weighted_median'],'scale':target['scale'],'weight':1.,
            'stage_start':target['stage_start'],'stage_end':target['stage_end'],'gate_width':.02,
            'target_id':target['target_id'],'evidence_ids':target['matched_evidence_ids'],
            'interpretation':'Stop attraction at the discovery selected median; this is not a physical optimum'})
    catalog=read_json(root/'feature_catalog.json')
    regions={catalog['features'][t['feature']]['region'] for t in terms}
    catalog['features']={t['feature']:catalog['features'][t['feature']] for t in terms}
    catalog['regions']={k:v for k,v in catalog['regions'].items() if k in regions}
    inputs=Path(read_json(root/'ingest_manifest.json')['source_root'])/'inputs'
    from evomolsteer.generation.launcher import INPUT_FILES
    program={'schema_version':'live-regional-1.0','program_id':'ck2_discovery_regional_v1',
        'representation':'predicted_endpoint_world_A','terms':terms,
        'source_dataset_id':bundle['dataset_id'],'source_request_sha256':digest(root/'agents/Analyst.request.json'),
        'source_catalog_sha256':digest(root/'feature_catalog.json'),
        'source_catalog_hash_meaning':'Complete original analysis catalog, before runtime region/feature trimming',
        'required_input_sha256':{name:digest(inputs/name) for name in INPUT_FILES},
        'parameter_policy':'Discovery-only fixed medians/IQR-derived scales; no validation or final outcomes',
        'hypothesis_status':'selection-associated geometry; affinity benefit unvalidated',
        'gradient_semantics':'sum of independent per-particle rewards; no mean scaling with batch size',
        'penalty':'u=max((z-target)/scale,0); phi=u^2/2 if u<=1 else u-1/2; R=-gate*phi'}
    write_json(out/'reward_program.json',program);write_json(out/'reward_catalog.json',catalog)
    write_json(out/'frozen_targets.json',bundle['targets'])
    print('Frozen discovery-only terms:',[t['id'] for t in terms])


if __name__=='__main__':main()
