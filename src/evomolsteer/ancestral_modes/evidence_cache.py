"""Content-bound caches; a changed support/window cannot reuse an old curve."""
import hashlib
import json
from pathlib import Path

from evomolsteer.io import digest


def cache_path(repo, master, mode, region, source, kind, implementation):
    base = Path(source).parent.parent.parent
    # The coordinate conversion and lineage mapping are part of the measurement.
    frame = base / ('frame_batch_%03d.json' % mode['batch'])
    config = base / 'config.json'
    binding = {'kind': kind, 'source_sha256': digest(source), 'frame_sha256': digest(frame),
               'dataset_config_sha256': digest(config), 'learning_window': master['learning_window'],
               'field_sigma_A': master['field_sigma_A'], 'region_id': region['region_id'],
               'atom_slots': region['atom_slots'], 'terminal_slot': mode['terminal_slot'],
               'terminal_coords_A': mode['terminal_coords_A'], 'teacher_path': mode['path'],
               'implementation_sha256': digest(Path(implementation))}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    folder = 'regional_evidence_v3' if kind == 'current' else 'regional_forecast_evidence_v2'
    path = repo / master['campaign_study'] / folder / (region['region_id'] + '_' + key[:16] + '.json')
    return path, key
