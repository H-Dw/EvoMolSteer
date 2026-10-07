"""Compile the literal Agent response only after its behavioral checks pass."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.continuous.regional_design import audit_regional_behavior,compile_regional
from evomolsteer.generation.window_reference import load_reference

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    root=Path(a.repo);e=root/'docs/experiments/ck2_affinity_geometry30_20261007';folder=e/'regional_skill_test'
    prior=e/'regional_mining_v1/region_prior.json';ref=root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz'
    audit=audit_regional_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-regional-pointcloud.md',folder/'input.json',folder/'prompt.txt',folder/'response.json',prior,folder/'behavior_audit.json')
    program=compile_regional(read_json(folder/'response.json'),read_json(root/'configs/experiments/ck2_affinity_geometry30_v1/backtrack_round07.json'),
        load_reference(ref),digest(ref),read_json(prior),digest(prior),prior.relative_to(root).as_posix())
    write_json(folder/'compiled_design.json',program);print({'behavior_verified':audit['passed'],'selected_region':audit['selected_region'],'family':program['reward_view']})

if __name__=='__main__':main()
