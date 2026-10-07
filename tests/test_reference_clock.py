import gzip
import json
from types import SimpleNamespace
import numpy as np
import pytest
from evomolsteer.generation.window_reference import reference_node_time
from evomolsteer.generation.window_controller import WindowExtension
from evomolsteer.io import digest


def test_score_forecast_clock_differs_from_actual_state_clock():
    for value in ('proposal','predicted_endpoint'):
        assert reference_node_time(dict(schema_version='affinity-endpoint-library-1.0',control_representation=value),.49,.5)==.49
    assert reference_node_time(dict(control_representation='proposal'),.49,.5)==.49
    assert reference_node_time({},.49,.5)==.5
    with pytest.raises(ValueError):reference_node_time(dict(schema_version='affinity-endpoint-library-1.0',control_representation='current'),.49,.5)
    with pytest.raises(ValueError):reference_node_time(dict(control_representation='unknown'),.49,.5)


@pytest.mark.parametrize('representation',['proposal','predicted_endpoint'])
def test_real_prepare_checks_all_fifty_pre_native_nodes(tmp_path,monkeypatch,representation):
    from evomolsteer.generation import launcher
    monkeypatch.setattr(launcher,'INPUT_FILES',())
    ref=dict(schema_version='affinity-endpoint-library-1.0',control_representation=representation,
        window=[0.,.5],times=(np.arange(50)/100).tolist(),required_input_sha256={})
    path=tmp_path/'reference.json.gz';path.write_bytes(gzip.compress(json.dumps(ref).encode(),mtime=0))
    program=tmp_path/'program.json';program.write_text(json.dumps({'reference_sha256':digest(path)}))
    inp=tmp_path/'input';inp.mkdir()
    opt=SimpleNamespace(n=50,batch=50,steps=100,arms='unguided,gradient,gradient_zero',
        program=str(program),reference=str(path),window_start=None,window=None,
        input_dataset=str(inp),root=str(tmp_path/'work'),seed=42)
    WindowExtension().prepare(opt,tmp_path/'not_created')
    assert opt.window_start==0. and opt.window==.5
    # A genuinely missing score node remains rejected, rather than loosening
    # the learned-window check to make an invalid reference launch.
    ref['times']=ref['times'][:-1];path.write_bytes(gzip.compress(json.dumps(ref).encode(),mtime=0))
    program.write_text(json.dumps({'reference_sha256':digest(path)}))
    with pytest.raises(ValueError,match='does not cover every controlled state'):
        WindowExtension().prepare(opt,tmp_path/'still_not_created')
