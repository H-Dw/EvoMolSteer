"""Prepare compact static-mixture inputs and an ordered proposed test matrix."""
import argparse
import copy
import gzip
import json
from pathlib import Path
from evomolsteer.io import read_json, write_json, digest


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def prepare(base_path, structure_path, teacher_path, out):
    base, structure, teacher = read_json(base_path), load(structure_path), load(teacher_path)
    if structure['required_input_sha256'] != teacher['required_input_sha256']:
        raise ValueError('Source coordinate frames / input dataset differ')
    if structure['window'] != teacher['window'] or base['window'] != teacher['window']:
        raise ValueError('Explicit common execution support required')
    if teacher.get('dependency_ablation') != 'static':
        raise ValueError('Only a declared fixed teacher bank is accepted')
    frame = teacher['frames'][-1]
    points = frame['teacher_endpoint_A']
    if any(f['teacher_endpoint_A'] != points for f in teacher['frames']):
        raise ValueError('Temporal teacher changes remain in the static source')
    out.mkdir(parents=True, exist_ok=False)
    reference = {k: copy.deepcopy(structure[k]) for k in
                 ['window', 'times', 'required_input_sha256', 'representation', 'bound_ligand_A', 'protein_points_A']}
    reference.update(schema_version='static-hybrid-coordinate-library-1.0', control_representation='proposal',
                     teacher_bank_A=points, teacher_bank_source_time=frame['time'],
                     steer_sources=[{'path': str(teacher_path), 'sha256': digest(teacher_path)}],
                     original_structure_source={'path': str(structure_path), 'sha256': digest(structure_path)},
                     affinity_head_gradient=False, teacher_scores_used=False,
                     scope='Selected Steer coordinates plus original bound pose; NOT teacher-free')
    ref_path = out / 'reference.json.gz'
    ref_path.write_bytes(gzip.compress(json.dumps(reference, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    common = copy.deepcopy(base)
    for key in ['derivation', 'geometry_block_weights', 'robust_delta', 'target_definition',
                'teacher_score_beta', 'endpoint_compiled_baseline_sha256', 'endpoint_designer_response_sha256',
                'geometry_review_sha256', 'skill_behavior_audit_sha256', 'skill_sha256',
                'round', 'contrast_bound_nats', 'core_radius_A', 'motif_components']:
        common.pop(key, None)
    common.update(reward_view='endpoint_static_hybrid', reference_sha256=digest(ref_path),
                  native_rms_ratio=.33, time_ramp_power=0, pointcloud_delta_A=1,
                  teacher_neighbors=4, teacher_endpoint_temperature_A2=4,
                  contact_weight=0, contact_phase_power=0, repulsion_weight=0,
                  base_program_sha256=digest(base_path),
                  evidence_role='Root-proposed numerical module combination; not a new Analyst/Designer response')
    proposals = []
    for name, rho in [('H25', .25), ('H50', .5), ('H75', .75)]:
        program = {**common, 'program_id': 'static_combo_' + name, 'original_pose_prior_mass': rho}
        write_json(out / (name + '.json'), program)
        proposals.append({'name': name, 'original_pose_prior_mass': rho,
                          'program_sha256': digest(out / (name + '.json'))})
    late = {**common, 'program_id': 'static_combo_H50_late_contact', 'original_pose_prior_mass': .5,
            'contact_weight': .05, 'contact_phase_power': 2}
    write_json(out / 'H50_late_contact.json', late)
    write_json(out / 'protocol.json', {
        'schema_version': 'static-combination-proposal-1.0', 'status': 'prepared_not_inference_validated',
        'proposals': proposals, 'reference_sha256': digest(ref_path),
        'window': common['window'], 'master_seed': 42,
        'screen_batches': [83, 84], 'confirmation_batches': list(range(85, 91)),
        'screen_n_per_arm': 100, 'confirmation_n_per_arm': 300, 'batch_size': 50, 'steps': 100,
        'baseline_programs': ['Frozen I3 original pose', 'Frozen M2 static teacher', 'Frozen R26', 'Native'],
        'ordered_tests': ['First validate reward mathematics and endpoint limits',
                          'Then H25/H50/H75 separately, equal dose/count against frozen controls',
                          'Only if H50 improves affinity, compare H50_late_contact against H50',
                          'Confirm the selected program on previously unused batches before promotion'],
        'primary': 'All-attempt final FLOWR predicted pIC50; strain secondary',
        'required_diagnostics': ['Actual dose and gradient continuity', 'Endpoint cancellation ratio and mixture posterior ESS',
                                 'PB fast and local MMFF strain', 'High-affinity tail yield', 'Unique graphs and surrounding relaxation'],
        'limits': 'Prior mass is not physical force share. A mixed field may cancel or lose diversity. Late contact is an unvalidated schedule hypothesis; no fitted temporal mechanism claim.',
        'new_generation_submitted': False})
    return {'reference_bytes': ref_path.stat().st_size, 'source_bytes': teacher_path.stat().st_size,
            'teacher_clouds': len(points), 'reference': str(ref_path)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-program', required=True); p.add_argument('--structure-reference', required=True)
    p.add_argument('--static-teacher-reference', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(Path(a.base_program), Path(a.structure_reference),
                             Path(a.static_teacher_reference), Path(a.output))))
