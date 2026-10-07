"""Verify observable decisions in a literal Skill/API simulation before activation."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.window_reference import load_reference
from evomolsteer.continuous.persistent_design import audit_persistent_behavior,compile_persistent


def verify(root):
    e=root/'docs/experiments/ck2_affinity_geometry30_20261007';folder=e/'persistent_skill_test'
    prior=e/'endpoint_mining/sparse_coordinate_prior_v2.json';history=e/'kinematics_mining_v3/kinematics_prior.json'
    reference=root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz'
    audit=audit_persistent_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-persistent-coordinate.md',folder/'input.json',folder/'prompt.txt',
                folder/'response.json',prior,history,folder/'behavior_audit.json')
    program=compile_persistent(read_json(folder/'response.json'),read_json(e/'sparse_skill_test_v2/compiled_design.json'),
        load_reference(reference),digest(reference),read_json(prior),digest(prior),read_json(history),digest(history),history.relative_to(root).as_posix())
    program.update(persistent_designer_response_sha256=digest(folder/'response.json'),persistent_behavior_audit_sha256=digest(folder/'behavior_audit.json'))
    write_json(folder/'compiled_design.json',program)
    return dict(passed=audit['persistent_response_verified'],compiled_design_sha256=digest(folder/'compiled_design.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);print(verify(parser.parse_args().repo))
