import numpy as np
import pandas as pd
import pytest
from evomolsteer.generation.affinity_reporting import batch_inference,paired_head

def test_batch_bootstrap_and_exact_symmetry_null():
    a=batch_inference([.1]*6)
    assert a['batch_bootstrap_CI95']==pytest.approx([.1,.1])
    assert a['symmetric_null_exact_signflip_p']==pytest.approx(2/64)

def test_paired_reports_preserve_negative_changes_and_all_attempts():
    a=pd.DataFrame({'arm':['gradient']*2,'batch':[0,0],'seed':[42,42],'slot':[0,1],'pic50_on_rescore':[7.,8.]})
    n=a.copy();n['arm']='unguided';n['pic50_on_rescore']=[8.,7.]
    p=paired_head(a,n)
    np.testing.assert_array_equal(p.delta,[-1,1])
    with pytest.raises(ValueError):paired_head(a,n.iloc[:1])
