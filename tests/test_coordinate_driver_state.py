import importlib.util
from pathlib import Path
import pytest
from evomolsteer.io import read_json
from evomolsteer.generation.prototypes import write_json

spec=importlib.util.spec_from_file_location('resume_coordinate_campaign',Path(__file__).parents[1]/'scripts/resume_coordinate_campaign.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('status',['running','completed','frozen'])
def test_stage_recovery_and_user_pause_never_launch_an_extra_round(tmp_path,status):
    driver=module.Driver.__new__(module.Driver);driver.state=tmp_path/'state';driver.state.mkdir()
    driver.evidence=tmp_path/'evidence';driver.campaign=tmp_path/'campaign.json'
    write_json(driver.campaign,{'maximum_rounds':30,'master_seed':42,'rounds':[{'round':15,'status':status,'campaign':'test'}]})
    value={'round':15,'shape_improvement_fraction':.01,'all_head_change_vs_native':0.,'MMFF_relief_relative_change':0.,'surround_RMS_improvement_fraction':0.}
    write_json(driver.evidence/'round_15.outcome.json',value)
    events=[];launches=[]
    def pause():write_json(driver.state/'USER_STOP.json',{'requested_by':'user','action':'pause'})
    def finish(row):pause();return value
    def launch(row):
        launches.append(row['round']);c=read_json(driver.campaign);c['rounds'][-1]['status']='running';write_json(driver.campaign,c)
    driver.finish=finish;driver.cleanup_local=lambda *args:pause();driver.launch_frozen=launch
    driver.event=lambda stage,**kwargs:events.append(stage)
    driver.start_next=lambda *args:pytest.fail('User pause must prevent the next round')
    driver.run()
    assert events==['stopped_by_user'] and launches==([15] if status=='frozen' else [])
