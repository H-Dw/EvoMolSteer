"""Saved sampling is postprocessed once; charged reservations are aliases, not refills."""
import copy

import h5py
import numpy as np
import pytest
import torch

from agenticgen.uniguide import absolute_recovery as recovery
from agenticgen.uniguide import absolute_runtime, sampler
from agenticgen.uniguide.geometry_protocol import EFFECTIVE
from agenticgen.uniguide.io import read, sha, value_sha, write
from agenticgen.uniguide.native_tools import save_pt
from test_absolute_coordinate_protocol import condition


def binding(condition, version):
    return dict(policy=recovery.POLICY, teacher_id='first', condition_sha256=condition['condition_sha256'],
                scientific_sources={name: version for name in recovery.STATIC_REFACTOR_SOURCES})


def test_only_explicit_static_source_revisions_can_transfer(condition):
    before, after = binding(condition, 'old'), binding(condition, 'new')
    changes = {name: dict(previous='old', current='new') for name in recovery.STATIC_REFACTOR_SOURCES}
    recovery.assert_sampling_equivalence(before, after, changes)
    altered = copy.deepcopy(after);altered['teacher_id'] = 'second'
    with pytest.raises(ValueError, match='settings'):
        recovery.assert_sampling_equivalence(before, altered, changes)
    altered = copy.deepcopy(after);altered['scientific_sources']['sampler'] = 'unexpected'
    with pytest.raises(ValueError, match='dependency set'):
        recovery.assert_sampling_equivalence(before, altered, changes)
    with pytest.raises(ValueError, match='unaudited'):
        recovery.assert_sampling_equivalence(before, after, {})


def fixture(tmp_path, condition, monkeypatch):
    old, new = tmp_path/'source', tmp_path/'new'
    old.mkdir();new.mkdir()
    config = dict(root=str(new), protocol=recovery.POLICY)
    previous, current = binding(condition, 'old'), binding(condition, 'new')
    subjects = [dict(teacher=dict(target=t), condition=condition) for t in ('first', 'second')]
    write(old/'SUBJECTS.json', subjects);write(new/'SUBJECTS.json', subjects)
    ledger = {}
    for scale in (1., 2., 4.):
        name = 'sample_shards/old'+str(int(scale))
        identity = dict(compatibility=previous, **{EFFECTIVE:[scale]*4}, population=10, logical_slots=[0, 1])
        write(old/name/'manifest.json', dict(id=name, identity=identity))
        ledger[name] = dict(stage='calibration', count=10, binding=value_sha(identity))
    write(old/'attempts.json', ledger)
    batch = old/'candidate_batches/saved';batch.mkdir(parents=True)
    xyz = torch.tensor([[[0., 0., 0.], [0., 2., 0.], [1.4, 0., 0.]]]).repeat(20, 1, 1)
    save_pt(batch/'terminal.pt.gz', dict(coords=xyz, mask=torch.ones(20, 3, dtype=torch.bool)))
    with h5py.File(batch/'trace.h5', 'w') as trace:
        trace['assignment'] = np.tile([1, 2], (100, 20, 1))
        trace['endpoint_A'] = np.tile(xyz.numpy()[None], (100, 1, 1, 1))
        trace['guidance_delta_A'] = np.zeros((100, 20, 3, 3), dtype=np.float32)
    write(batch/'steps.json', [dict(step=i, loss=[1.]*20, gradient_norm=[1.]*20,
          coefficient=[1.]*20, max_atom_increment_A=[0.]*20, delta_native_ratio=[0.]*20,
          assignment_switch_fraction=[0.]*20, candidate_rng_after_sha256=['fixture']*2,
          failed_units={}) for i in range(100)])
    write(batch/'rng_ledger.json', [[], []])
    write(batch/'FAILED.json', dict(type='NameError', error="name 'h5py' is not defined"))
    write(batch/'candidates_batch.json', dict(teacher_id='first', condition_sha256=condition['condition_sha256'],
          candidates=[dict(source_group='sample_shards/old'+str(int(scale)),
          logical_seed=seed, logical_slot=slot, **{EFFECTIVE:[scale]*4})
          for scale in (1., 2.) for seed in recovery.SEEDS for slot in (0, 1)]))
    proof = tmp_path/'proof.json'
    write(proof, dict(teacher_vjp=[dict(teacher_id=t, condition_sha256=condition['condition_sha256'],
         loss_bitwise_equal=True, full_coordinate_vjp_bitwise_equal=True,
         sampling_rng_unchanged=True, affinity_gradient_entered=False) for t in ('first', 'second')]))
    original_hashes = {str(p.relative_to(old)):sha(p) for p in old.rglob('*') if p.is_file()}
    auth = dict(policy=recovery.RECOVERY_POLICY, source_root=str(old), target_root=str(new),
        source_commit='original-fixed-commit', target_config_sha256=value_sha(config),
        source_artifact_hashes=original_hashes, source_attempts_sha256=sha(old/'attempts.json'),
        equivalence_evidence=str(proof), equivalence_evidence_sha256=sha(proof),
        source_changes={name:dict(previous='old', current='new') for name in recovery.STATIC_REFACTOR_SOURCES},
        source_batch=str(batch.relative_to(old)), expected_reserved_attempts=30, expected_saved_candidates=20)
    path = tmp_path/'authorization.json';write(path, auth)
    config['sampling_recovery'] = dict(authorization_path=str(path), authorization_sha256=sha(path))
    monkeypatch.setattr(recovery, 'compatibility', lambda *args:current)
    monkeypatch.setattr(recovery, 'calibration_plan', lambda *args:dict(coarse_actual_scales=[1., 2., 4.]))
    monkeypatch.setattr(absolute_runtime, 'require_acceptance', lambda config:None)
    monkeypatch.setattr(sampler, 'sample', lambda *args, **kwargs:pytest.fail('Recovery must never sample'))
    return old, new, config, auth, original_hashes


def test_preflight_is_readonly_and_transfer_never_samples_or_reserves_twice(tmp_path, condition, monkeypatch):
    old, new, config, auth, hashes = fixture(tmp_path, condition, monkeypatch)
    summary = recovery.preflight(config)
    assert summary['reserved_attempts'] == 30 and summary['saved_candidates'] == 20
    assert summary['pending_reserved_candidates'] == 10 and not (new/'attempts.json').exists()
    loads = []

    class Builder:
        def _generate_mols(self, output, sanitise=True):
            return [None]*len(output['coords'])

    def loader(config, teacher):
        loads.append(teacher['target'])
        return Builder()

    result = recovery.run(config, loader)
    assert result['new_scientific_attempts'] == 0 and len(result['materialized_receipts']) == 2
    assert loads == ['first']
    assert sum(v['count'] for v in read(new/'attempts.json').values()) == 30
    assert len(list((new/'slot_registry').glob('*.json'))) == 2
    assert all(sha(old/name) == digest for name, digest in hashes.items())
    with pytest.raises(RuntimeError, match='already claimed'):
        recovery.run(config, loader)
    # A pending reservation is still unused; its first claim does not add budget.
    pending = [a for a in result['reservation_aliases']
               if not (new/a['current_id']/'COMPLETE.json').exists()]
    assert len(pending) == 1
    for receipt in result['materialized_receipts']:
        provenance = read(new/receipt['id']/'BATCH_PROVENANCE.json')['sampling_recovery']
        assert provenance['sampling_was_not_reexecuted'] and provenance['reservation_alias']


def test_changed_saved_artifact_is_not_silently_regenerated(tmp_path, condition, monkeypatch):
    old, new, config, auth, hashes = fixture(tmp_path, condition, monkeypatch)
    path = old/'candidate_batches/saved/FAILED.json'
    write(path, dict(type='FloatingPointError'))
    with pytest.raises(ValueError, match='artifact changed'):
        recovery.preflight(config)
    assert not (new/'RECOVERY_CLAIM.json').exists() and not (new/'attempts.json').exists()


def test_duplicate_reservation_aliases_are_rejected():
    identity = {EFFECTIVE:[1.]*4, 'logical_slots':[0,1]}
    previous = {'a':dict(stage='calibration',count=10,binding=value_sha(identity))}
    unit = dict(identity=identity, id='new', fingerprint='new')
    with pytest.raises(ValueError, match='population differs'):
        recovery.reservation_aliases(previous, {'a':dict(identity=identity)}, [unit, unit])
