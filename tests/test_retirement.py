import importlib.util
from pathlib import Path
import pytest

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
