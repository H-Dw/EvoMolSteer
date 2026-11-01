"""Independently verify every delivered transform and its reported residual."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    references = json.loads((HERE/'reference_coordinates.json').read_text(encoding='utf-8'))
    source_files = json.loads((HERE/'raw/source_preservation.json').read_text(encoding='utf-8'))
    assert references['teachers_sha256'] == source_files[references['teachers_path']]['sha256']
    manifest = json.loads((HERE/'computation_manifest.json').read_text(encoding='utf-8'))
    for name in ['alignment_results.json','aligned_candidates.jsonl.gz','source_preservation.json','validation.json']:
        path = HERE/'raw'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['results/'+name]['sha256']
    assert hashlib.sha256((HERE/'align_condition_results.py').read_bytes()).hexdigest() == manifest['align_condition_results.py']['sha256']
    count, identities, max_error = 0, set(), 0.
    with gzip.open(HERE/'raw/aligned_candidates.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['experiment'],row['teacher'],row['physical_id'])
            assert key not in identities
            identities.add(key)
            p = np.asarray(row['matched_coordinates_angstrom'])
            q = np.asarray(references['reference_coordinates_angstrom'][row['teacher']])
            r, t = np.asarray(row['rotation_row_vectors']), np.asarray(row['translation_angstrom'])
            assert p.shape == q.shape == (9,3) and len(set(row['condition_slots'])) == 9
            assert abs(np.linalg.det(r)-1.) < 1e-10
            assert np.max(np.abs(r.T @ r - np.eye(3))) < 1e-10
            assert np.max(np.abs(p @ r + t - row['aligned_condition_coordinates_angstrom'])) < 1e-12
            residual = float(np.sqrt(np.mean(np.sum((p @ r + t - q)**2,axis=1))))
            max_error = max(max_error,abs(residual-row['aligned_rmsd']))
            assert abs(residual-row['aligned_rmsd']) < 1e-12
            assert row['aligned_rmsd'] <= row['raw_rmsd'] + 1e-10
            count += 1
    assert count == 19135
    result = dict(verified_candidate_transforms=count, all_candidates_distinct_within_experiment=True,
                  max_reconstructed_rmsd_error=max_error, references_match_protected_teacher_catalog=True,
                  downloaded_machine_results_sha256_verified=True,
                  executed_script_matches_delivered_script=True)
    (HERE/'local_validation.json').write_bytes((json.dumps(result,ensure_ascii=True,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result,ensure_ascii=True))


if __name__ == '__main__':
    main()
