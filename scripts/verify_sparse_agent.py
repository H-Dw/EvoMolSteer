"""Bind a literal independent Agent response to the recorded coordinate prior."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.window_reference import load_reference
from evomolsteer.continuous.sparse_design import audit_sparse_behavior,compile_sparse


def verify(root):
    evidence=root/'docs/experiments/ck2_affinity_geometry30_20261007'
    folder=evidence/'sparse_skill_test_v2';prior=evidence/'endpoint_mining/sparse_coordinate_prior_v2.json'
    reference=root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz'
    audit=audit_sparse_behavior(root/'skills/affinity-sparse-coordinate/SKILL.md',folder/'input.json',folder/'prompt.txt',
                               folder/'response.json',prior,folder/'behavior_audit.json')
    program=compile_sparse(read_json(folder/'response.json'),read_json(evidence/'endpoint_skill_test/compiled_design.json'),
                           load_reference(reference),digest(reference),read_json(prior),digest(prior))
    program['sparse_designer_response_sha256']=digest(folder/'response.json')
    program['sparse_behavior_audit_sha256']=digest(folder/'behavior_audit.json')
    write_json(folder/'compiled_design.json',program)
    return {'audit_passed':audit['sparse_response_verified'],'mode':program['geometry_direction_mode'],
            'compiled_design_sha256':digest(folder/'compiled_design.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    print(verify(parser.parse_args().repo))
