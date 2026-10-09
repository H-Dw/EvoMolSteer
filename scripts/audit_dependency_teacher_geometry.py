"""Summarize forecast-teacher geometry over the observed window without fitting.

This is a post-design explanatory audit. Its output is not part of the isolated
ablation Agent inputs. Teacher-to-template distance is a geometry diagnostic,
not a causal score explanation or a physical binding-energy estimate.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import write_json, digest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', required=True); p.add_argument('--structure', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    with gzip.open(a.teacher, 'rt', encoding='utf-8') as f: teacher = json.load(f)
    with gzip.open(a.structure, 'rt', encoding='utf-8') as f: structure = json.load(f)
    ligand = np.asarray(structure['bound_ligand_A']); rows = []
    for time, frame in zip(teacher['times'], teacher['frames']):
        clouds = np.asarray(frame['teacher_endpoint_A']); distances, radii = [], []
        for cloud in clouds:
            costs = ((cloud[:, None] - ligand[None])**2).sum(-1)
            i, j = linear_sum_assignment(costs)
            distances.append(float(np.sqrt(costs[i, j].mean())))
            radii.append(float(np.sqrt(((cloud - cloud.mean(0))**2).sum(1).mean())))
        rows.append({'time': time, 'teachers': len(clouds),
            'template_assignment_rms_A_mean': float(np.mean(distances)),
            'template_assignment_rms_A_p90': float(np.quantile(distances, .9)),
            'teacher_gyration_radius_A_mean': float(np.mean(radii)),
            'forecast_teacher_score_mean': float(np.mean(frame['teacher_scores']))})
    write_json(a.output, {'schema_version': 'teacher-geometry-audit-1.0',
        'teacher_sha256': digest(a.teacher), 'structure_sha256': digest(a.structure),
        'original_ligand_gyration_radius_A': float(np.sqrt(((ligand-ligand.mean(0))**2).sum(1).mean())),
        'rows': rows, 'not_supplied_to_design_agents': True,
        'limits': 'Selected endpoint forecasts only; temporal trends cannot establish that the dynamic target caused a generation-quality change'})
