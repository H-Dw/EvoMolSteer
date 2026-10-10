"""Final-family-credit branch directions at matched instantaneous affinity.

Point selection and final-family priors stay frozen. Only observed conditional
natural branch modes change; no field is fitted to unobserved extinct futures.
"""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import clean, digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths
from .branch_mutation import compose_ancestors, conditional_teacher_contrast
from .outcome_alias_credit import pooled_outcome


def build(dataset, campaign, labels, metrics, evidence, output, tolerance=.25):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    packet = read_json(evidence)
    if not np.isfinite(tolerance) or tolerance < 0 or digest(metrics) != packet['metrics_sha256']:
        raise ValueError('Matched tolerance and common final evaluation required')
    ref = copy.deepcopy(json.loads(gzip.decompress(Path(packet['reference_path']).read_bytes())))
    all_labels, all_metrics = pd.read_parquet(labels), pd.read_csv(metrics)
    source = Path(dataset)/'results'/campaign
    cfg = read_json(source/'config.json')
    paths = {int(p.parent.name.split('_')[1]):p for p in trajectory_paths(source) if p.parent.parent.name == 'single'}
    diagnostics = []
    for batch in packet['donor_batches']:
        if digest(paths[batch]) != next(v['sha256'] for v in ref['sources'] if v['batch'] == batch):
            raise ValueError('Original natural branch source changed')
        batch_metrics = all_metrics[all_metrics.batch == batch]
        com = np.array(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:, None]
        with open_trajectory(paths[batch]) as tr:
            selected = np.asarray(tr['selected_indices'])
            for frame in ref['frames']:
                lf = all_labels[(all_labels.batch == batch)&(np.abs(all_labels.score_time-frame['time']) <= 2e-6)].sort_values('slot').reset_index(drop=True)
                step = int(lf.step.iloc[0])
                if step < 2:
                    continue
                xyz = np.asarray(tr['predicted_coords'][step], float)*cfg['coord_scale']+com
                online = np.asarray(tr['pic50_on'][step], float)
                parents = selected[step-1]
                ancestors = compose_ancestors(parents, selected[step-2])
                quality = np.full(len(parents), np.nan)
                for parent in np.unique(parents):
                    slot = int(np.flatnonzero(parents == parent)[0])
                    if not lf.loc[parents == parent, 'valid_n'].sum():
                        continue
                    credit = pooled_outcome(lf, parents, slot, batch_metrics, packet['threshold_pic50'])
                    quality[parents == parent] = credit['terminal_mean']
                for k in np.flatnonzero(np.array(frame['teacher_batches']) == batch):
                    slot = frame['teacher_slots'][k]
                    anchor = np.array(frame['teacher_endpoint_A'][k])
                    if not np.allclose(anchor, xyz[slot], rtol=0, atol=1e-5):
                        raise ValueError('Teacher is not original observed endpoint')
                    eligible = np.isfinite(quality)&(np.abs(online-online[slot]) <= tolerance)
                    contrast = conditional_teacher_contrast(anchor, quality[slot], xyz[eligible], quality[eligible],
                        parents[eligible], ancestors[eligible], int(parents[slot]), int(ancestors[slot]))
                    frame['teacher_contrast_direction_unit'][k] = contrast['direction_unit'].tolist()
                    frame['teacher_contrast_confidence'][k] = contrast['confidence']
                    frame['teacher_contrast_atom_weight'][k] = contrast['atom_weight'].tolist()
                    frame['teacher_contrast_provenance'][k] = {'lag2_observed': True,
                        'raw_direction_RMS_A': contrast['raw_rms_A'], 'common_grandparent_slot': int(ancestors[slot]),
                        'immediate_parent_slot': int(parents[slot]), 'lower_immediate_parent_slots': contrast['lower_parent_ids'],
                        'distinct_observed_mutations': contrast['distinct_observed_mutations'],
                        'score_label': 'decoded_final_copy_family_mean', 'instant_score_tolerance': tolerance}
                    diagnostics.append({'batch': batch, 'time': frame['time'], 'teacher': int(k),
                        'raw_RMS_A': contrast['raw_rms_A'], 'confidence': contrast['confidence'],
                        'distinct_mutations': contrast['distinct_observed_mutations'],
                        'lower_branch_n': len(contrast['lower_parent_ids'])})
    ref['teacher_selection']['branch_mode'] = 'decoded_final_copy_family_matched'
    ref['teacher_selection']['instant_score_tolerance'] = tolerance
    out.mkdir(parents=True)
    pd.DataFrame(diagnostics).to_csv(out/'branch_support.csv', index=False)
    path = out/'reference.json.gz'
    path.write_bytes(gzip.compress(json.dumps(clean(ref), separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    packet['evidence_items'].append({'id': 'outcome/matched_reward_modes', 'teacher_events': len(diagnostics),
        'qualified_teacher_events': sum(v['raw_RMS_A'] > 0 and v['confidence'] > 0 for v in diagnostics),
        'online_tolerance': tolerance, 'point_selection_and_priors_unchanged': True,
        'direction_label': 'Observed exact-copy family mean final affinity, conditioned on common grandparent and similar online score'})
    packet.update(reference_path=str(path.resolve()), reference_sha256=digest(path), branch_mode='decoded_final_copy_family_matched',
                  matched_reference_source_sha256=digest(__file__))
    write_json(out/'evidence.json', packet)
    return packet
