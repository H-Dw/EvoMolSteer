import numpy as np
import pytest
from evomolsteer.continuous.coordinate_mining import boundary_descendants


def test_ancestry_is_preselection_mass_and_excludes_outside_edge():
    selected=np.array([[0,0,2],[0,2,2],[1,1,1]],int)
    ids,mass=boundary_descendants([True]*3,[.25,.5,.75],selected,[0,.5])
    assert ids.tolist()==[0,1]
    assert mass.tolist()==[[1,0,2],[1,0,2]]
    assert np.all(mass.sum(1)==3)
    modified=selected.copy();modified[-1]=[0,0,0]
    np.testing.assert_array_equal(mass,boundary_descendants([True]*3,[.25,.5,.75],modified,[0,.5])[1])


def test_dynamic_boundary_and_malformed_ancestry():
    s=np.array([[0,0],[0,1],[1,1]],int)
    ids,mass=boundary_descendants([True]*3,[.11,.21,.31],s,[.1,.31])
    assert ids.tolist()==[0,1,2] and mass.tolist()==[[2,0],[0,2],[0,2]]
    with pytest.raises(ValueError):boundary_descendants([True,False,True],[.1,.2,.3],s,[0,.3])
    with pytest.raises(ValueError):boundary_descendants([True]*3,[.1,.2,.3],s.astype(float),[0,.3])
