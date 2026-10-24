"""Pre-sampling public assignments only; never publish sibling observations."""
from pathlib import Path
from .contracts import validate_ticks
from .io import read, sha

PROTOCOL = 'prior_selected_schedules_v1'


def validate_commitment(document, teacher, version, observed_round, anchor, tick_upper):
    if document is None:
        return []  # Bootstrap has no preselected wave.
    keys = {'protocol', 'teacher_id', 'version', 'observed_round', 'selected_after_round',
            'source_relative_path', 'source_sha256', 'schedules'}
    if set(document) != keys or document['protocol'] != PROTOCOL:
        raise ValueError('Only the committed schedule projection is permitted')
    if (document['teacher_id'] != teacher or document['version'] != version or
            document['observed_round'] != observed_round or
            document['selected_after_round'] != observed_round-1 or observed_round < 1):
        raise ValueError('Committed schedule teacher/version/round mismatch')
    expected = ('bootstrap/hypotheses.json' if observed_round == 1 else
                f'round_{observed_round-1:02d}/council/selection.json')
    digest = document['source_sha256']
    if (document['source_relative_path'] != expected or not isinstance(digest, str) or
            len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Committed schedule source mismatch')
    ticks = [validate_ticks(t, tick_upper) for t in document['schedules']]
    if len(ticks) != 4 or len({tuple(t) for t in ticks}) != 4 or anchor not in ticks:
        raise ValueError('Four distinct committed assignments including the private anchor are required')
    return ticks


def load_commitment(output, specs, version, tick_upper):
    """Verify the saved inference job against its prior public selection before projection."""
    output = Path(output)
    teacher = specs[0]['data']['teacher']['teacher_id']
    r = specs[0]['data']['observed_round']
    teacher_folder = output.parent.parent
    relative = 'bootstrap/hypotheses.json' if r == 1 else f'round_{r-1:02d}/council/selection.json'
    path = teacher_folder/relative
    source = read(path)
    if (source.get('teacher_id') != teacher or source.get('observed_round') != r-1 or
            (r == 1 and source.get('version') != version)):
        raise ValueError('Prior selection belongs to a different task')
    if r > 1 and read(path.parent/'COMPLETE.json')['selection_sha256'] != sha(path):
        raise ValueError('Prior selected schedules changed after publication')
    jobs = source['jobs']
    expected_ids = [('bootstrap:'+j['id']) if r == 1 else j['source_proposal_id'] for j in jobs]
    saved = read(output.parent/'job.json')
    if (saved.get('teacher_id') != teacher or saved.get('observed_round') != r-1 or
            [j['ticks'] for j in saved['jobs']] != [j['ticks'] for j in jobs] or
            [j['source_proposal_id'] for j in saved['jobs']] != expected_ids or
            [s['anchor_ticks'] for s in specs] != [j['ticks'] for j in jobs]):
        raise ValueError('Sampled job differs from the prior committed selection')
    doc = {'protocol': PROTOCOL, 'teacher_id': teacher, 'version': version,
           'observed_round': r, 'selected_after_round': r-1,
           'source_relative_path': relative, 'source_sha256': sha(path),
           'schedules': sorted([j['ticks'] for j in jobs])}
    for spec in specs:
        validate_commitment(doc, teacher, version, r, spec['anchor_ticks'], tick_upper)
    return doc


def reject_committed(message, schedules):
    reserved = {tuple(t) for t in schedules}
    if any(tuple(h['ticks']) in reserved for h in message['hypotheses']):
        raise ValueError('Schedule was already selected before this wave and has now been sampled. '
                         'Use the supplied committed schedule exclusion list and choose a new legal proposal yourself; '
                         'no sibling result is available in this private session.')
