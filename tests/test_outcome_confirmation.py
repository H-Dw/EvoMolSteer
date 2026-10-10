import copy
import pytest
from evomolsteer.io import digest, write_json
from scripts import prepare_outcome_confirmation as confirmation


def test_confirmation_allows_metadata_revision_but_rejects_retuning(tmp_path, monkeypatch):
    cfg = tmp_path/'configs'
    cfg.mkdir()
    monkeypatch.setattr(confirmation, 'ROOT', tmp_path)
    monkeypatch.setattr(confirmation, 'CFG', cfg)
    monkeypatch.setattr(confirmation, 'SOURCES', [])
    ref = cfg/'candidate.gz'
    ref.write_bytes(b'frozen teacher library')
    program = {'round': 14, 'program_id': 'candidate14', 'window': [0, .51],
        'score_window': [0, .5], 'reference_sha256': digest(ref), 'teacher_score_beta': 2,
        'outcome_binding': {'updates': {}, 'base_program_sha256': 'development-parent'}}
    for name, old in [('R11', 'round02'), ('R26', 'round01')]:
        reference = cfg/'references'/(name+'_instant_051.json.gz')
        reference.parent.mkdir(exist_ok=True)
        reference.write_bytes(name.encode())
        control = {**program, 'reference_sha256': digest(reference)}
        write_json(cfg/(old+'_'+name+'.json'), control)
    def candidate(number, value):
        path = cfg/f'candidate{number}.json'
        write_json(path, value)
        write_json(cfg/f'round{number:02d}.jobs.json', {'round': number, 'jobs': [{
            'program': str(path.relative_to(tmp_path)), 'reference': str(ref.relative_to(tmp_path)),
            'batches': [133, 134] if number == 14 else [135, 136], 'arms': 'gradient'}]})
        return path
    confirmation.prepare(14, candidate(14, program))
    changed = copy.deepcopy(program)
    changed.update(round=15, program_id='candidate15', derivation={'new_request': 'metadata'})
    changed['teacher_score_beta'] = 3
    path = candidate(15, changed)
    with pytest.raises(ValueError, match='Frozen candidate'):
        confirmation.prepare(15, path)
    changed['teacher_score_beta'] = 2
    write_json(path, changed)
    confirmation.prepare(15, path)
