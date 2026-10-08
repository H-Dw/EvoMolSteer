"""Map empirical spatial landmarks to the exact input receptor atoms.

This is coordinate annotation, not an interaction or energy assignment. Residue
identifiers are data-derived and never enter generic Skills or reward gates.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from evomolsteer.io import digest, write_json


def annotate(reference, receptor, output):
    reference, receptor = Path(reference), Path(receptor)
    ref = json.loads(gzip.decompress(reference.read_bytes()))
    expected = ref['required_input_sha256'].get(receptor.name)
    if expected is None or digest(receptor) != expected:
        raise ValueError('Exact reference receptor required')
    atoms = []
    for line in receptor.read_text().splitlines():
        if line.startswith('ATOM') and line[76:78].strip() != 'H':
            atoms.append({'atom_serial': int(line[6:11]), 'atom': line[12:16].strip(),
                          'residue': line[17:20].strip(), 'chain': line[21:22],
                          'residue_number': line[22:27].strip(), 'element': line[76:78].strip(),
                          'xyz_A': [float(line[s:s+8]) for s in [30, 38, 46]]})
    xyz = np.asarray([a['xyz_A'] for a in atoms])
    chosen = {r['region_index'] for r in ref['dynamic_cohort']['regions']}
    rows = []
    for i, point in enumerate(ref['landmarks_A']):
        dist = np.linalg.norm(xyz - point, axis=1)
        nearest = int(dist.argmin())
        rows.append({'region_index': i, 'selected_in_reference': i in chosen, 'xyz_A': point,
                     'nearest_receptor_atom': atoms[nearest], 'matching_distance_A': float(dist[nearest])})
    result = {'schema_version': 'reward-landmark-annotation-1.0',
              'reference_sha256': digest(reference), 'receptor_sha256': digest(receptor),
              'annotations': rows,
              'semantics': 'Nearest receptor atom in the frozen coordinate frame; does not prove chemical contact or energy benefit.'}
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--receptor', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = annotate(a.reference, a.receptor, a.output)
    print(json.dumps([r for r in result['annotations'] if r['selected_in_reference']]))
