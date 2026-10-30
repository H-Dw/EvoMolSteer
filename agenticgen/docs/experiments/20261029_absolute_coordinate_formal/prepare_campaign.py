"""Create isolated absolute-coordinate inputs and a protected campaign root."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
from agenticgen.uniguide.io import read, sha, frozen, write
from agenticgen.uniguide.absolute_runtime import prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--previous-config',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--measurement-source-root',type=Path)
    args=parser.parse_args()
    old=read(args.previous_config)
    config=read(args.repo/'agenticgen/configs/uniguide_absolute_coordinate.json')
    if args.measurement_source_root:
        from agenticgen.uniguide.calibration_search import POLICY
        if config.get('calibration',{}).get('search_policy')!=POLICY:
            raise ValueError('Measurement import requires adaptive calibration')
        config['calibration']['measurement_source_root']=str(args.measurement_source_root.resolve())
    config['plip_dependencies']=old['plip_dependencies']
    expected={'6pyy_H7S_A_402':'single_b004_final_015','3w54_RNB_A_504':'single_b000_final_003'}
    inputs=Path(str(args.repo)+'-inputs');inputs.mkdir(exist_ok=True)
    previous={x['teacher_id']:x for x in old['conditions']};sources={}
    for item in config['conditions']:
        target=item['teacher_id'];source=Path(previous[target]['features_path']);value=read(source)
        # Preserve original atoms, graph and absolute reference; no old distance
        # weights, protein pairs or supplied generated-slot indexing is migrated.
        features={k:value[k] for k in ('atoms','bonds','controlled_atom_ids','coordinates_angstrom','coordinate_frame')}
        if len(features['controlled_atom_ids'])!=9:raise ValueError('Original nine PLIP condition atoms required')
        destination=inputs/(target+'.json');frozen(destination,features);item['features_path']=str(destination)
        sources[str(source)]=sha(source)
    catalog={t['target']:t for t in read(Path(config['generation_context_root'])/'teachers.json')}
    if any(catalog[t]['node_id']!=node for t,node in expected.items()):raise ValueError('Teacher changed')
    frozen(args.config,config)
    subjects=prepare(config);root=Path(config['root'])
    previous_root=Path(old['root']);protected=dict(read(previous_root/'SOURCE_PRESERVATION.json'))
    for p in previous_root.rglob('*'):
        if p.is_file() and 'monitoring' not in p.relative_to(previous_root).parts and p.name not in ('.executor.lock','.campaign.lock','.worker.lock'):
            protected[str(p)]=sha(p)
    protected.update(sources)
    frozen(root/'SOURCE_PRESERVATION.json',protected)
    commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
    audit=dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),repo=str(args.repo),commit=commit,
               config=str(args.config),config_sha256=sha(args.config),root=str(root),
               source_preservation_count=len(protected),source_features_sha256=sources,
               protocol=config['protocol'],skills=config['agent_tools']['skill_release'],rounds_per_teacher=config['rounds'],
               teachers=[dict(target=s['teacher']['target'],node_id=s['teacher']['node_id'],
                  condition_sha256=s['condition']['condition_sha256'],
                  condition_atoms=len(s['condition']['controlled_atom_ids']),
                  protein_anchor_count=len(s['condition']['protein_alignment']['atom_identities']),
                  protein_alignment=s['condition']['protein_alignment']['fitted_protein_rmsd_angstrom']) for s in subjects])
    frozen(root/'PREPARED_AUDIT.json',audit)
    print(json.dumps(audit,ensure_ascii=True))


if __name__=='__main__':main()
