import pytest
from evomolsteer.continuous.coordinate_agents import validate,DESIGNER


def test_design_contract_prohibits_unprovided_view_window_and_evidence():
    bundle={'window':[.1,.4],'evidence':[{'evidence_id':'e','feature':'r::all::spread'}],
            'features':{'r::all::spread':{'region':'r'}}}
    d={'schema_version':'coordinate-1.0','agent':'Designer','design_status':'exploratory','architecture':'spread_upper',
       'regions':['r'],'channel':'all','window':[.1,.4],'native_rms_ratio':.2,'mixture_temperature':.25,'robust_delta':1.,
       'rationale':'Association hypothesis','evidence_ids':['e'],'limitations':['Not causal']}
    assert validate(d,DESIGNER,bundle)
    for field,value in [('window',[0.,.5]),('regions',['unmeasured']),('evidence_ids',['unknown'])]:
        with pytest.raises(ValueError):validate({**d,field:value},DESIGNER,bundle)
    bundle.update(control_representation='proposal',features={'r::all::proposal_spread':{'region':'r'}})
    assert validate(d,DESIGNER,bundle)
    bundle['features']={'r::all::spread':{'region':'r'}}
    with pytest.raises(ValueError):validate(d,DESIGNER,bundle)
