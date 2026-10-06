import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('dispatch_coordinate_round',Path(__file__).parents[1]/'scripts/dispatch_coordinate_round.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_remote_dispatch_reuses_immutable_identity_without_duplicate_inference(tmp_path,monkeypatch):
    repo=tmp_path/'repo';work=tmp_path/'work';(repo/'configs').mkdir(parents=True);work.mkdir()
    program=repo/'configs/p.json';reference=repo/'configs/r.gz';program.write_text('{}');reference.write_bytes(b'reference')
    calls=[]
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'commit\n')
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:(calls.append(a),SimpleNamespace(pid=12345))[1])
    args=(repo,work,16,'coordinate_r16_backtrack',program,reference,'docs/previous.json')
    first=module.dispatch(*args);(work/'round16.exit').write_text('0')
    assert module.dispatch(*args)==first and len(calls)==1
    reference.write_bytes(b'changed')
    with pytest.raises(ValueError,match='different immutable'):module.dispatch(*args)
    assert len(calls)==1


def test_incomplete_claim_is_not_automatically_relaunched(tmp_path):
    repo=tmp_path/'repo';work=tmp_path/'work';(repo/'configs').mkdir(parents=True);work.mkdir()
    p=repo/'configs/p.json';r=repo/'configs/r.gz';p.write_text('{}');r.write_bytes(b'reference')
    (work/'round16.dispatch').mkdir()
    with pytest.raises(RuntimeError,match='Incomplete dispatch'):
        module.dispatch(repo,work,16,'coordinate_r16_backtrack',p,r,'docs/previous.json')
