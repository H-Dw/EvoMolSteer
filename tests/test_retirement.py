import importlib.util
from pathlib import Path
import pytest
import json

spec=importlib.util.spec_from_file_location('retirement',Path(__file__).parents[1]/'scripts/retire_experiment_outputs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_retirement_protects_reference_report_and_scope(tmp_path):
    base=tmp_path/'outputs';base.mkdir();target=base/'old';target.mkdir();(target/'data').write_text('trajectory')
    reference=base/'original';reference.mkdir();report=tmp_path/'report.json';report.write_text('{}')
    plan={'allowed_bases':[str(base)],'protected':[str(reference)],'result_report':str(report),'targets':[str(target)]}
    rows,_=module.validate(plan,tmp_path/'audit.json');assert rows[0]['bytes']==10
    for bad in [reference,base,tmp_path]:
        with pytest.raises(ValueError):module.validate(dict(plan,targets=[str(bad)]),tmp_path/'audit.json')
    with pytest.raises(ValueError):module.validate(plan,target/'audit.json')


def test_inference_gate_rejects_old_outputs_and_wrong_frozen_program(tmp_path):
    spec=importlib.util.spec_from_file_location('ready',Path(__file__).parents[1]/'scripts/check_terminal_round_ready.py')
    ready=importlib.util.module_from_spec(spec);spec.loader.exec_module(ready)
    original=tmp_path/'original';original.mkdir()
    report=tmp_path/'report.json';report.write_text('{}')
    old=tmp_path/'old_run';audit=tmp_path/'audit.json'
    audit.write_text(json.dumps({'status':'deleted','targets':[{'path':str(old)}], 'bytes':123,'surviving_report':str(report)}))
    manifest=tmp_path/'ready.json';manifest.write_text(json.dumps({'master_seed':42,'round':3,'maximum_rounds':5,
        'campaigns':['r3'],'cleanup_audit':str(audit),'protected_inputs':[str(original)],'generated_root':str(tmp_path/'new')}))
    program=tmp_path/'program.json';program.write_text(json.dumps({'seed':42,'round':2}))
    with pytest.raises(ValueError,match='frozen round'):ready.check(manifest,'r3',program)
    program.write_text(json.dumps({'seed':42,'round':3}))
    assert ready.check(manifest,'r3',program)['ready']
    old.mkdir()
    with pytest.raises(ValueError,match='still exists'):ready.check(manifest,'r3',program)
