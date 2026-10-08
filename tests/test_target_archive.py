import io
import json
import tarfile

import pytest

from evomolsteer.io import digest
from evomolsteer.storage.target_archive import archive_target_directories,verify_target_archive,restore_target,retire_archived_directories,MANIFEST


def sources(tmp_path):
    roots={key:tmp_path/'working'/key for key in ('target_a','target_b')}
    for i,p in enumerate(roots.values()):
        (p/'inputs').mkdir(parents=True);(p/'inputs/reference.sdf').write_bytes(b'original coordinates\n'*100)
        (p/'scores.json').write_text(json.dumps({'affinity':8+i,'failed_slots':[1]}))
    return roots


def test_group_roundtrip_is_deterministic_and_preserves_every_byte(tmp_path):
    roots=sources(tmp_path)
    a=tmp_path/'a.tar.gz';b=tmp_path/'b.tar.gz'
    meta=archive_target_directories(roots,a);again=archive_target_directories(roots,b)
    assert digest(a)==digest(b) and meta['target_count']==2 and meta['source_deleted'] is False
    assert meta['saved_payload_bytes']==meta['source_bytes']-meta['archive_bytes']
    assert meta['sidecar_bytes']==(tmp_path/'a.tar.gz.json').stat().st_size
    verify_target_archive(a,meta)
    output=tmp_path/'restored';restore_target(a,'target_b',output,meta)
    for p in roots['target_b'].rglob('*'):
        if p.is_file():assert (output/p.relative_to(roots['target_b'])).read_bytes()==p.read_bytes()
    assert not (output/'target_a').exists()
    with pytest.raises(FileExistsError):restore_target(a,'target_b',output)


def test_retirement_checks_all_sources_and_can_resume_after_interruption(tmp_path):
    roots=sources(tmp_path);a=tmp_path/'a.tar.gz';meta=archive_target_directories(roots,a)
    outsider=tmp_path/'protected';outsider.mkdir();(outsider/'source.pdb').write_text('protected')
    with pytest.raises(ValueError):retire_archived_directories(a,meta,{'target_a':outsider},owned_root=tmp_path/'working')
    (roots['target_b']/'new_unarchived.txt').write_text('must preserve')
    with pytest.raises(ValueError):retire_archived_directories(a,meta,roots,owned_root=tmp_path/'working')
    assert (roots['target_a']/'scores.json').is_file()
    (roots['target_b']/'new_unarchived.txt').unlink()
    (roots['target_a']/'scores.json').unlink()  # simulate an interrupted verified retirement
    result=retire_archived_directories(a,meta,roots,owned_root=tmp_path/'working')
    assert result['source_deleted'] and all(not p.exists() for p in roots.values())
    restore_target(a,'target_a',tmp_path/'recover')
    assert json.loads((tmp_path/'recover/scores.json').read_text())['failed_slots']==[1]
    assert (outsider/'source.pdb').read_text()=='protected'


@pytest.mark.parametrize('change',['payload','transport','symlink'])
def test_changed_archive_or_source_is_rejected_before_retirement(tmp_path,change):
    roots=sources(tmp_path);a=tmp_path/'a.tar.gz';meta=archive_target_directories(roots,a)
    if change=='payload':
        (roots['target_b']/'scores.json').write_text('changed')
        with pytest.raises(ValueError):retire_archived_directories(a,meta,roots,owned_root=tmp_path/'working')
    elif change=='transport':
        meta['archive_sha256']='0'*64
        with pytest.raises(ValueError):verify_target_archive(a,meta)
    else:
        bad=tmp_path/'link.tar.gz'
        with tarfile.open(bad,'w:gz') as f:
            member=tarfile.TarInfo('targets/target_a/link');member.type=tarfile.SYMTYPE;member.linkname='../../outside'
            f.addfile(member)
        with pytest.raises(ValueError):verify_target_archive(bad)
    assert (roots['target_a']/'scores.json').is_file()


@pytest.mark.parametrize('member',['../outside','targets/a/../../outside','C:/outside','targets/a/duplicate'])
def test_unsafe_or_undeclared_archive_members_fail(tmp_path,member):
    p=tmp_path/'bad.tar.gz'
    manifest={'format':'evomolsteer.steer_target_archive.v1','complete':True,'targets':{'a':{'target_id':'a'}},'files':{}}
    with tarfile.open(p,'w:gz') as f:
        data=json.dumps(manifest).encode();m=tarfile.TarInfo(MANIFEST);m.size=len(data);f.addfile(m,io.BytesIO(data))
        data=b'bad';m=tarfile.TarInfo(member);m.size=len(data);f.addfile(m,io.BytesIO(data))
    with pytest.raises(ValueError):verify_target_archive(p)
