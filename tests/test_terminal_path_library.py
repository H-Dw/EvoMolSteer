import numpy as np
import pandas as pd
import pytest
from evomolsteer.continuous.terminal_path_library import choose_paths,path_teachers
def test_graph_duplicates_and_invalids_do_not_fill_archive():
    m=pd.DataFrame({'slot':[0,1,2,3],'smiles':['CC','CC','CN','CO'],
        'pic50_on_rescore':[8.7,8.6,9.,8.3],'valid_connected':[True,True,False,True]})
    assert choose_paths(m).slot.tolist()==[0,3]
def test_shared_ancestor_is_one_teacher_with_multiple_paths():
    m=pd.DataFrame({'slot':[0,1],'pic50_on_rescore':[8.7,8.4]})
    a=np.array([[2,2],[0,1]])
    assert path_teachers(a,m,0)==[{'slot':2,'score':8.7,'terminal_slots':[0,1]}]
    assert len(path_teachers(a,m,1))==2
def test_invalid_budget_rejected():
    with pytest.raises(ValueError):choose_paths(pd.DataFrame(),0)
