"""Review-only migration preserves completed versions and still requires native acceptance."""
import shutil
from pathlib import Path

import pytest

from test_budget_extension import boundary, accept
from agenticgen.runtime import review_policy as p
from agenticgen.runtime import extend_budget as budget
from agenticgen.adaptive_schedule.io import read, write, sha


@pytest.fixture
def review_boundary(boundary, monkeypatch):
    root,old,progress,auth=boundary
    c=read(old/'config.json');c.update(versions=[f'v{i}' for i in range(6,31)],total_result_budget=25050)
    write(old/'config.json',c)
    prior=read(old/'PREPARED.json');prior['config_sha256']=sha(old/'config.json')
    write(old/'PREPARED.json',prior);write(old/'ACCEPTANCE_COMPLETE.json',prior)
    write(root/'ACTIVE_PROTOCOL.json',{'directory':'protocols/old','prepared_sha256':sha(old/'PREPARED.json')})
    shutil.copytree(root/'versions/v6',root/'versions/v7')
    f=read(root/'FINALIZED_v6.json');f['version']='v7';write(root/'FINALIZED_v7.json',f)
    for name in ('idle_locks','no_worker','assert_committed_runtime','committed','commit'):
        monkeypatch.setattr(p,name,getattr(budget,name))
    fp={**prior['source_fingerprint'],'adaptive_schedule/group_review.py':'changed','runtime/review_policy.py':'new'}
    monkeypatch.setattr(p,'source_fingerprint',lambda:fp)
    monkeypatch.setattr(p,'authorization',lambda path,digest:{'path':path,'sha256':digest})
    monkeypatch.setattr(p,'performance_context',lambda *a:{'teachers':{'a':{'round':2},'b':{'round':4}}})
    def native(root):
        folder=p.protocol_dir(root);a=read(folder/'ACCEPTANCE_COMPLETE.json')
        if a['source_fingerprint']!=fp:raise ValueError('Fresh native acceptance required')
    monkeypatch.setattr(p,'native_ready',native)
    return root,old,auth,fp


def test_stage_and_finish_preserve_old_provenance(review_boundary):
    root,old,auth,fp=review_boundary
    before=budget.inventory(root/'versions');old_before=budget.inventory(old)
    op=Path(p.stage(root,'v7',auth)['operation']);folder=p.protocol_dir(root)
    assert budget.inventory(root/'versions')==before and budget.inventory(old)==old_before
    assert read(folder/'config.json')==read(old/'config.json')
    with pytest.raises(FileNotFoundError):p.finish(root,op)
    write(folder/'ACCEPTANCE_COMPLETE.json',{'source_fingerprint':fp})
    assert p.finish(root,op)['launch_approved'] is False
    assert p.verify_ready(root,'v8')['phase']=='accepted_for_review_check'
    assert p.verify_ready(root,'v30')['next_version']=='v8'
    with pytest.raises(ValueError):p.verify_ready(root,'v31')
    with pytest.raises(ValueError):p.stage(root,'v7',auth)


@pytest.mark.parametrize('mutation',['worker','future','source','result','quality'])
def test_unsafe_adoption_does_not_change_active_pointer(review_boundary,monkeypatch,mutation):
    root,old,auth,fp=review_boundary;pointer=(root/'ACTIVE_PROTOCOL.json').read_bytes()
    if mutation=='worker':monkeypatch.setattr(p,'no_worker',lambda r:(_ for _ in ()).throw(ValueError('worker alive')))
    if mutation=='future':write(root/'versions/v8/binding.json',{})
    if mutation=='source':monkeypatch.setattr(p,'source_fingerprint',lambda:{**fp,'adaptive_schedule/wave_native.py':'changed'})
    if mutation=='result':write(root/'versions/v7/report.json',{'changed':True})
    if mutation=='quality':(root/'versions/v7/a/round_05/wave/QUALITY_COMPLETE.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):p.stage(root,'v7',auth)
    assert (root/'ACTIVE_PROTOCOL.json').read_bytes()==pointer


def test_changed_completed_result_rejects_finish(review_boundary):
    root,old,auth,fp=review_boundary;op=Path(p.stage(root,'v7',auth)['operation'])
    write(p.protocol_dir(root)/'ACCEPTANCE_COMPLETE.json',{'source_fingerprint':fp})
    write(root/'versions/v7/report.json',{'changed':True})
    with pytest.raises(ValueError,match='Protected artifact'):p.finish(root,op)
