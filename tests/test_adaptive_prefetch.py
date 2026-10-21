import copy
import threading
import time
from types import SimpleNamespace

import pytest

from evomolsteer.adaptive_schedule import prefetch
from evomolsteer.adaptive_schedule.io import defaults, read, write


def inputs(teacher='top1'):
    chains = [{'slot': s, 'best_id': 'r00', 'history': [{'id': 'r00'}]} for s in range(20)]
    return chains, {'decision_round': 1, 'teacher_id': teacher}


def test_two_teachers_share_global_http_limit_and_preserve_slot_order(tmp_path):
    active = peak = 0
    guard = threading.Lock()
    def proposer(root, cfg, template, slot, history, best, skill, output, population):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(.003 * (1 + slot % 3))
        with guard:
            active -= 1
        return {'slot': slot, 'teacher': template['teacher_id']}
    pool = prefetch.ProposalPool(4)
    try:
        batches = []
        for teacher in ['top1', 'top2']:
            chains, population = inputs(teacher)
            folder = tmp_path / teacher
            bundle = pool.submit(tmp_path, defaults(), {'teacher_id': teacher}, chains, tmp_path/'skill',
                                 folder, population, proposer)
            batches.append((bundle, chains, population, folder))
        for bundle, chains, population, folder in batches:
            actions = prefetch.collect(bundle, chains, population, folder)
            assert [a['slot'] for a in actions] == list(range(20))
            assert {a['teacher'] for a in actions} == {population['teacher_id']}
        assert 1 < peak <= 4
    finally:
        pool.close()


def test_changed_history_or_population_cannot_consume_prefetched_decisions(tmp_path):
    chains, population = inputs()
    pool = prefetch.ProposalPool(1)
    try:
        bundle = pool.submit(tmp_path, defaults(), {'teacher_id': 'top1'}, chains, tmp_path/'skill',
                             tmp_path/'calls', population, lambda *a, **k: {})
        changed = copy.deepcopy(chains)
        changed[0]['best_id'] = 'r01'
        with pytest.raises(ValueError, match='changed'):
            prefetch.collect(bundle, changed, population, tmp_path/'calls')
        with pytest.raises(ValueError, match='changed'):
            prefetch.collect(bundle, chains, dict(population, decision_round=2), tmp_path/'calls')
    finally:
        pool.close()


def test_interleaving_preserves_all_hundred_chain_round_barriers(tmp_path, monkeypatch):
    import sys
    from evomolsteer.adaptive_schedule import runner
    config = defaults()
    config['rounds'] = 2
    config['retention']['mode'] = 'all'
    teachers = [{'teacher_id': t, 'target': 'target', 'source_record': {'pic50_on_rescore': 7}}
                for t in ['top1', 'top2']]
    score = {'pic50_on_rescore': 4, 'build_success': True, 'connected': True, 'failure_reason': ''}
    for arm in ['steer', 'unguided']:
        write(tmp_path/'baselines'/f'target__{arm}.json', [score]*100)
    monkeypatch.setitem(sys.modules, 'evomolsteer.adaptive_schedule.flowr', SimpleNamespace(load_model=lambda *a: object()))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    queued, executed = [], []
    def submit(self, root, cfg, t, chains, skill, folder, population):
        r = population['decision_round']
        assert population['previous_unselected_round']['attempts'] == 100
        assert all(len(c['history']) == r for c in chains)
        queued.append((r, t['teacher_id'], folder.parent.name))
        return {'teacher': t['teacher_id'], 'round': r}
    monkeypatch.setattr(prefetch.ProposalPool, 'submit', submit)
    def batch_run(model, root, cfg, t, batch, version, skill, fast, round_no, population, proposals=None):
        assert threading.current_thread() is threading.main_thread()
        folder = root/'versions'/version/t['teacher_id']/f'batch_{batch:03d}'
        if round_no:
            assert proposals == {'teacher': t['teacher_id'], 'round': round_no}
            assert len(queued) >= 2  # Next teacher's HTTP work precedes this GPU job.
            chains = read(folder/'histories.json')
        else:
            chains = [{'slot': s, 'best_id': 'r00', 'history': []} for s in range(20)]
        for c in chains:
            c['history'].append({'id': f'r{round_no:02d}', 'record': score})
        write(folder/'histories.json', chains)
        executed.append((round_no, batch, t['teacher_id']))
    monkeypatch.setattr(runner, 'run_batch', batch_run)
    assert prefetch.interleaved_run(tmp_path, config, teachers, 'v1', tmp_path/'skill', True) == []
    assert executed == [(r, b, t['teacher_id']) for r in range(3) for b in range(5) for t in teachers]
    # All committed work is skipped on resume; no additional HTTP or GPU attempt.
    before = len(queued), len(executed)
    prefetch.interleaved_run(tmp_path, config, teachers, 'v1', tmp_path/'skill', True)
    assert (len(queued), len(executed)) == before
