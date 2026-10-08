import io
import json
import tarfile

from evomolsteer.generation.campaign_health import check_campaign


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture(tmp_path):
    root = tmp_path / 'output'
    cfg = {'samples': 1000, 'flowr_root': str(tmp_path / 'flowr'),
           'campaign': 'single_w050', 'window_start': 0., 'window_end': .5}
    record = {'state': 'complete', 'dataset': 'targets/target',
              'attempts': [{'state': 'complete'}],
              'summary': {'attempted_slots': 1000, 'built_slots': 850}}
    state = {'config': cfg, 'status': 'running', 'archives': [],
             'catalog': [{'target_id': 'target', 'key': 'target'}],
             'targets': {'target': record}}
    manifest = {'state': 'complete', 'verification': {'verified': True, 'records': 1000},
                'checkpoint_sha256': 'model_hash', 'flowr_sampler_sha256': 'sampler_hash',
                'command': ['python', '-m', 'evomolsteer.generation.steer_worker'],
                'gradient_guidance': False}
    runtime = {'device': 'fixture GPU', 'hip': '6.3', 'torch': '2.9',
               'flowr_package': str(tmp_path / 'flowr/flowr/__init__.py'),
               'gradient_guidance': False, 'selection_window': [0., .5]}
    save(root / 'campaign_state.json', state)
    marker = root / 'generation_progress/target.finished'
    marker.parent.mkdir(); marker.touch()
    save(root / 'targets/target/learning_dataset_manifest.json', manifest)
    save(root / 'targets/target/results/single_w050/runtime.json', runtime)
    return root, state, manifest, runtime


def test_pending_does_not_claim_gpu_validation(tmp_path):
    root = tmp_path / 'output'; root.mkdir()
    result = check_campaign(root)
    assert result['status'] == 'pending' and not result['normal_target_verified']


def test_normal_requires_native_completion_and_gpu_provenance(tmp_path):
    root, _, _, _ = fixture(tmp_path)
    result = check_campaign(root)
    assert result['status'] == 'normal' and result['native_target']['attempted_slots'] == 1000
    (root / 'targets/target/results/single_w050/runtime.json').unlink()
    result = check_campaign(root)
    assert result['status'] == 'attention' and not result['normal_target_verified']


def test_finished_marker_alone_never_proves_success(tmp_path):
    root, state, _, _ = fixture(tmp_path)
    state['targets']['target']['state'] = 'running'
    save(root / 'campaign_state.json', state)
    result = check_campaign(root)
    assert result['status'] == 'pending' and not result['normal_target_verified']


def test_oom_prevents_normal_even_after_target_completion(tmp_path):
    root, _, _, _ = fixture(tmp_path)
    log = tmp_path / 'controller.log'
    log.write_text('inference started\nHIP out of memory. Tried to allocate 2 GiB\n')
    result = check_campaign(root, controller_log=log)
    assert result['normal_target_verified'] and result['status'] == 'attention'
    assert result['oom'][0]['line'] == 2


def test_zero_valid_molecules_and_conflicting_markers_reject_success(tmp_path):
    root, state, _, _ = fixture(tmp_path)
    state['targets']['target']['summary']['built_slots'] = 0
    save(root / 'campaign_state.json', state)
    assert check_campaign(root)['status'] == 'attention'
    state['targets']['target']['summary']['built_slots'] = 850
    save(root / 'campaign_state.json', state)
    (root / 'generation_progress/target.error').write_text('Concurrent execution')
    assert check_campaign(root)['status'] == 'attention'


def test_retired_target_is_checked_without_extracting_archive(tmp_path):
    root, state, manifest, runtime = fixture(tmp_path)
    directory = root / 'targets/target'
    for path in (directory / 'learning_dataset_manifest.json', directory / 'results/single_w050/runtime.json'):
        path.unlink()
    archive = root / 'archives/targets_0000.tar.gz'
    archive.parent.mkdir()
    with tarfile.open(archive, 'w:gz') as stream:
        for name, doc in [('learning_dataset_manifest.json', manifest), ('results/single_w050/runtime.json', runtime)]:
            data = json.dumps(doc).encode()
            member = tarfile.TarInfo('targets/target/' + name); member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    state['targets']['target'].update(state='archived', archive='archives/targets_0000.tar.gz')
    state['archives'] = [{'path': 'archives/targets_0000.tar.gz', 'target_ids': ['target'],
                          'directories': {'target': 'targets/target'}, 'metadata': {'verified': True},
                          'retirement': {'source_deleted': True}}]
    save(root / 'campaign_state.json', state)
    result = check_campaign(root)
    assert result['status'] == 'normal' and result['native_target']['source'] == 'verified_archive'
    assert not (directory / 'learning_dataset_manifest.json').exists()
