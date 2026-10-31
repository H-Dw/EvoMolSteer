import importlib.util
from pathlib import Path
import tarfile
import pytest

path=Path(__file__).resolve().parents[2]/'docs/experiments/20261101_gradient_scope/archive_results.py'
spec=importlib.util.spec_from_file_location('scope_archive',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_archive_retains_result_states_and_deduplicates_content(tmp_path):
    root=tmp_path/'experiment';root.mkdir();dest=tmp_path/'durable';dest.mkdir()
    for name in ['trace.h5','terminal.pt.gz','checkpoint_010.pt.gz','same_trace.h5']:(root/name).write_bytes(b'result-data')
    for name in ['flowr.ckpt','launcher.log','api_probe.launcher.log','core.123']:(root/name).write_bytes(b'excluded')
    r=module.pack({'roots':[str(root)],'staging':str(tmp_path/'stage'),'destination':str(dest)},check_clock=False)
    assert r['moved_to_persistent'] and r['all_member_hashes_verified']
    assert (root/'trace.h5').read_bytes()==b'result-data'
    with tarfile.open(r['archive']) as archive:
        names=archive.getnames()
        assert set(names)=={'ARCHIVE_MANIFEST.json',*[root.name+'/'+n for n in ['trace.h5','terminal.pt.gz','checkpoint_010.pt.gz','same_trace.h5']]}
        assert any(m.islnk() for m in archive)


def test_early_activation_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='08:00'):
        module.pack({'activate_after':'2999-01-01T08:00:00+08:00'})
