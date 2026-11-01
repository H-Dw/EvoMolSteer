"""User-authorized rebuild: verify a result-only archive and retain it on workspace."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
from datetime import datetime
from zoneinfo import ZoneInfo


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def pack(plan, check_clock=True):
    stamp=datetime.now(ZoneInfo('Asia/Shanghai'))
    if check_clock and stamp < datetime.fromisoformat(plan['activate_after']):
        raise ValueError('This archival task may activate only at/after Shanghai 08:00')
    staging=Path(plan['staging']).resolve();staging.mkdir(parents=True,exist_ok=True)
    if check_clock:
        claim=staging/'ARCHIVE_CLAIM.json'
        with claim.open('x',encoding='utf-8') as stream:
            json.dump({'timestamp':stamp.isoformat(),'plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
                       'no_automatic_duplicate_pack':True},stream,ensure_ascii=True)
    name='uniguide_all_experiment_results_'+stamp.strftime('%Y%m%d_%H%M%S')
    archive=staging/(name+'.tar.gz');manifest={};links={};skipped=[]
    assert not archive.exists()
    roots=[Path(r).resolve() for r in plan['roots']]
    assert len(roots)==len(set(roots)) and len({r.name for r in roots})==len(roots)
    assert all(r.is_dir() and not staging.is_relative_to(r) for r in roots)
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for root in roots:
            for path in sorted(root.rglob('*')):
                rel=path.relative_to(root)
                if path.is_symlink():
                    skipped.append({'path':str(path),'reason':'external_link_not_dereferenced','target':os.readlink(path)});continue
                if not path.is_file():continue
                backup=path.name.endswith('.tar.gz') and ('monitoring' in rel.parts or path.name.startswith(('backup','closed_artifacts')))
                weight=(str(path.resolve()) in plan.get('model_weight_paths',[]) or path.suffix in {'.ckpt','.safetensors'}
                        or path.name.startswith(('pytorch_model','model_weights')))
                if weight or backup or 'launcher.log' in path.name or re.fullmatch(r'core(?:[._-]\d+)?',path.name):
                    skipped.append({'path':str(path),'reason':'model_weight' if weight else 'existing_backup' if backup else 'restricted_log_or_core'});continue
                before=path.stat();digest=sha(path);member=root.name+'/'+rel.as_posix()
                info=tar.gettarinfo(str(path),arcname=member)
                key=(digest,before.st_size,info.mode)
                if key in links:
                    info.type=tarfile.LNKTYPE;info.linkname=links[key];info.size=0;tar.addfile(info)
                else:
                    with path.open('rb') as stream:tar.addfile(info,stream)
                    links[key]=member
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or sha(path)!=digest:
                    raise RuntimeError('Source changed during archive; retain staging and do not move: '+str(path))
                manifest[member]={'sha256':digest,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,
                                  'content_hardlink_to':info.linkname if info.islnk() else None}
        inventory={'timestamp':stamp.isoformat(),'timezone':'Asia/Shanghai','files':manifest,'excluded':skipped,
                   'model_weights_included':False,'generated_tensor_checkpoints_included':True,
                   'identical_content_stored_as_standard_tar_hardlinks':True,'original_results_deleted':False}
        raw=(json.dumps(inventory,ensure_ascii=True,indent=2)+'\n').encode()
        info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    seen={}
    with tarfile.open(archive,'r|gz') as tar:
        for item in tar:
            if item.name=='ARCHIVE_MANIFEST.json':
                assert json.loads(tar.extractfile(item).read())==inventory;continue
            expected=manifest[item.name]
            if item.islnk():
                assert item.linkname in seen
                digest,size=seen[item.linkname]
            else:
                assert item.isfile();h=hashlib.sha256();size=0;stream=tar.extractfile(item)
                for block in iter(lambda:stream.read(8<<20),b''):h.update(block);size+=len(block)
                digest=h.hexdigest()
            assert digest==expected['sha256'] and size==expected['bytes']
            seen[item.name]=(digest,size)
    assert set(seen)==set(manifest)
    digest=sha(archive);size=archive.stat().st_size
    receipt={'timestamp':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'archive':str(archive),
             'sha256':digest,'bytes':size,'files':len(manifest),'all_member_hashes_verified':True,
             'model_weights_included':False,'original_results_deleted':False,'moved_to_persistent':False,
             'status':'complete_workspace_only','destination_policy':'leave_verified_archive_in_workspace'}
    (staging/(name+'_RECEIPT.json')).write_text(json.dumps(receipt,ensure_ascii=True,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    print(json.dumps(pack(json.loads(Path(parser.parse_args().plan).read_text())),ensure_ascii=True),flush=True)
