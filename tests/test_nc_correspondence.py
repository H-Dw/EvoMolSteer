import numpy as np
from evomolsteer.nc_guidance.correspondence import observations


def test_reassignment_is_distinguished_from_final_identity_on_tracked_half_atom():
    source=[[0.,0.,0.],[1.,0.,0.],[4.,0.,0.]]
    g={'teacher_clock':{'coords_A':[source],'mask':[[1,1,1]]},
       'terminal':{'coords_A':[[source[1],source[0],source[2]]],'mask':[[1,1,1]]}}
    p={'atom_slots':[0,1],'typed_slots':[0],'typed_elements':['N']}
    t={'coords_A':source,'terminal_coords_A':source,'atom_vocabulary':{'C':3,'N':4}}
    rows=observations(g,p,t,np.array([[3,4,3]]),[{'pic50':8.,'qualified_structure':True}])
    r=rows[0]
    assert r['half_generated_slot']==0 and r['final_generated_slot']==1
    assert r['correspondence_switched'] and not r['terminal_correct_along_half_slot']
    assert r['terminal_correct_after_rematching'] and r['rematched_terminal_distance_A']==0.


def test_missing_final_type_is_not_imputed_as_failure_or_success():
    x=[[0.,0.,0.],[1.,0.,0.],[4.,0.,0.]];frame={'coords_A':[x],'mask':[[1,1,1]]}
    g={'teacher_clock':frame,'terminal':frame};p={'atom_slots':[0,1],'typed_slots':[0],'typed_elements':['N']}
    t={'coords_A':x,'terminal_coords_A':x,'atom_vocabulary':{'C':3,'N':4}}
    r=observations(g,p,t,np.array([[-1,4,3]]),[{'pic50':7.,'qualified_structure':False}])[0]
    assert r['terminal_correct_along_half_slot'] is None and r['terminal_correct_after_rematching'] is None
    assert not r['correspondence_switched'] and not r['qualified_structure']
