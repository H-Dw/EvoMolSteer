"""Verify the downloaded archive chain without extracting or executing its members."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone, timedelta
import collections, hashlib, json, os, shutil, tarfile

BASE = Path(__file__).resolve().parent
OLD = BASE/'downloads/20261102_0534'
NEW = BASE/'downloads/20261102_0850'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write('\n')


def safe_name(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts
    leaf = path.name.lower()
    assert 'launcher.log' not in leaf
    assert leaf != 'core' and not leaf.startswith(('core.', 'core_'))
    assert not leaf.endswith(('.ckpt', '.safetensors'))
    assert leaf not in ('pytorch_model.bin', 'model.pt', 'weights.pt')


def verify(archive, expected_sha, expected_manifest):
    before = archive.stat()
    digest = sha(archive)
    assert digest == expected_sha
    verified, links, saved_manifest = {}, 0, None
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            safe_name(member.name)
            if member.name == 'MANIFEST.json':
                assert saved_manifest is None and member.isfile()
                saved_manifest = json.load(tar.extractfile(member))
                continue
            assert member.name not in verified
            if member.islnk():
                safe_name(member.linkname)
                assert member.linkname in verified
                verified[member.name] = verified[member.linkname]
                links += 1
            else:
                assert member.isfile(), 'Unexpected non-file archive member'
                h, count = hashlib.sha256(), 0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(8<<20), b''):
                        h.update(block)
                        count += len(block)
                verified[member.name] = (count, h.hexdigest())
    assert saved_manifest == expected_manifest
    assert len(verified) == len(expected_manifest)
    for row in expected_manifest:
        assert verified[row['path']] == (row['bytes'], row['sha256']), row['path']
    after = archive.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return dict(timestamp=now(), archive=str(archive), bytes=after.st_size, sha256=digest,
                members=len(verified), hardlinks=links,
                hdf5_members=sum(name.endswith('.h5') for name in verified),
                all_members_verified_locally=True, verified_locally=True,
                no_extraction=True, launcher_core_model_weights_included=False)


def main():
    first_index = read(OLD/'INDEX_LOCALLY_VERIFIED.json')
    assert first_index['verified_locally']
    first_manifest = read(OLD/'MANIFEST.json')
    first = verify(OLD/'completed_increment.tar.gz', first_index['previous_archive_sha256'], first_manifest)
    assert first['bytes'] == 363777660 and first['members'] == 2527
    assert first_index['files'] == {row['path']:row['sha256'] for row in first_manifest}
    print(json.dumps(dict(stage='first_archive_all_members_verified', receipt=first)), flush=True)
    receipt = read(NEW/'COLLECT_COMPLETE.json')
    final_manifest = read(NEW/'MANIFEST.json')
    final_path = NEW/'completed_increment.tar.gz'
    partial = NEW/'completed_increment.tar.gz.part'
    source = final_path if final_path.exists() else partial
    assert source.resolve().is_relative_to(BASE.resolve())
    assert source.stat().st_size == receipt['bytes']
    final = verify(source, receipt['sha256'], final_manifest)
    assert final['members'] == receipt['members'] == 7208
    assert final['hdf5_members'] == receipt['hdf5_members'] == 305
    references = read(NEW/'PREVIOUS_ARCHIVE_REFERENCES.json')
    assert references['previous_index_sha256'] == sha(OLD/'INDEX_LOCALLY_VERIFIED.json')
    assert references['previous_archive_sha256'] == first['sha256']
    for item in references['files']:
        assert first_index['files'][item['path']] == item['sha256']
    if source == partial:
        assert final_path.resolve().is_relative_to(BASE.resolve()) and not final_path.exists()
        os.rename(partial, final_path)
    final['archive'] = str(final_path)
    combined = dict(first_index['files'])
    collisions = []
    for item in final_manifest:
        previous = combined.get(item['path'])
        if previous and previous != item['sha256']:
            collisions.append(dict(path=item['path'], previous_sha256=previous, current_sha256=item['sha256']))
        combined[item['path']] = item['sha256']
    coverage = read(NEW/'SUPERSESSION_COVERAGE.json')
    assert coverage['covering_archive_sha256'] == final['sha256']
    aliases, resolved = [], []
    for proof in coverage['proof']:
        assert not proof['uncovered']
        for item in proof['locally_verified_metadata_aliases']:
            matches = [str(Path(path).resolve()) for path in item['local_aliases']
                       if Path(path).is_file() and sha(Path(path)) == item['sha256']]
            assert matches
            combined[item['source_path']] = item['sha256']
            aliases.append(dict(remote_source_path=item['source_path'], sha256=item['sha256'], local_paths=matches))
        resolved.append(dict(older_archive=proof['older_archive'], coverage_verified_locally=True,
                             redundant_raw_download_skipped=True, covering_archive_sha256=final['sha256']))
    artifacts = dict(timestamp=now(), fixed_scientific_commit=receipt['fixed_commit'],
        plan_sha256=receipt['plan_sha256'], first_archive=first, final_archive=final,
        previous_index_sha256=sha(OLD/'INDEX_LOCALLY_VERIFIED.json'),
        referenced_verified_files=len(references['files']), aliases=aliases,
        superseded_pending_archives=resolved, overwritten_version_references=collisions,
        current_logical_files=len(combined), all_archive_members_and_references_verified_locally=True,
        local_free_bytes=shutil.disk_usage(BASE).free, new_sampling=0, new_model_forwards=0, new_llm_calls=0,
        remote_originals_deleted=False, moved_to_private_data=False)
    chains = [dict(first_index['archive_chain'][0], local_archive=first['archive'], bytes=first['bytes'],
                   members=first['members'], all_members_verified_locally=True),
              dict(archive=receipt['archive'], local_archive=final['archive'], sha256=final['sha256'],
                   bytes=final['bytes'], members=final['members'], verified_locally=True,
                   all_members_verified_locally=True, references_archive_sha256=first['sha256'])]
    index = dict(timestamp=now(), verified_locally=True, previous_archive_sha256=final['sha256'],
                 files=dict(sorted(combined.items())), archive_chain=chains,
                 local_metadata_aliases=aliases, overwritten_version_references=collisions,
                 verification_receipt_sha256=None)
    write(NEW/'LOCAL_ARCHIVES_VERIFIED.json', artifacts)
    index['verification_receipt_sha256'] = sha(NEW/'LOCAL_ARCHIVES_VERIFIED.json')
    write(NEW/'INDEX_LOCALLY_VERIFIED.json', index)
    print(json.dumps(dict(stage='final_archive_chain_verified', receipt=artifacts,
                          index_sha256=sha(NEW/'INDEX_LOCALLY_VERIFIED.json'))), flush=True)


if __name__ == '__main__':
    main()
