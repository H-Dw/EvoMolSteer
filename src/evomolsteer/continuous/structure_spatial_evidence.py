"""Compact deterministic original-structure geometry, without selection labels."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

def spatial_evidence(protein_path, ligand, crop_radius=12, maximum_regions=10):
    atoms, residues = [], []
    for line in Path(protein_path).read_text().splitlines():
        if line.startswith(('ATOM  ', 'HETATM')) and line[76:78].strip() != 'H':
            atoms.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            residues.append(':'.join([line[21:22].strip(), line[22:27].strip(), line[17:20].strip()]))
    points = np.asarray(atoms); residues = np.asarray(residues)
    included = np.linalg.norm(points - np.asarray(ligand).mean(0), axis=1) <= crop_radius
    points, residues = points[included], residues[included]
    distances = cdist(np.asarray(ligand), points)
    regional = []
    for residue in sorted(set(residues)):
        ids = residues == residue; d = distances[:, ids]
        regional.append({'residue_id': residue, 'center_A': points[ids].mean(0).tolist(),
            'protein_heavy_atoms': int(ids.sum()), 'ligand_atoms_within_4_5A': int((d.min(1) < 4.5).sum()),
            'minimum_center_distance_A': float(d.min()),
            'proxy_contact_sum': float(np.exp(-.5*((d-3.6)/.6)**2).sum())})
    regional.sort(key=lambda r: (-r['ligand_atoms_within_4_5A'], r['residue_id']))
    nearest = distances.min(1)
    pair = cdist(points, points); np.fill_diagonal(pair, np.inf)
    return [{'evidence_id': 'structure:original_contacts', 'kind': 'original_ligand_geometry', 'data': {
        'regions': regional[:maximum_regions], 'total_regions': len(regional),
        'nearest_protein_A_quantiles': np.quantile(nearest, [0, .25, .5, .75, 1]).tolist(),
        'contact_definition': 'Untyped centre distances and Gaussian shell; no interaction/energy labels',
        'interpretation': 'Observed bound pose geometry only; no generated-state enrichment or target optimum'}},
        {'evidence_id': 'structure:protein_crop', 'kind': 'receptor_geometry', 'data': {
        'crop_centroid_A': points.mean(0).tolist(), 'crop_radius_A': crop_radius,
        'protein_heavy_atoms': len(points), 'nearest_protein_neighbor_A_quantiles': np.quantile(pair.min(1), [0, .5, 1]).tolist(),
        'origin': 'Original receptor crop around the standard FLOWR input site'}}]
