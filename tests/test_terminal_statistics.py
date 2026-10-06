import numpy as np
import pandas as pd
from evomolsteer.generation.terminal_statistics import converged_energy, finite_values


def test_finite_failed_energy_is_excluded_and_coverage_is_not_all_attempts():
    d=pd.DataFrame({'energy_status':['converged','not_converged','error','converged','converged'],
        'mmff_relief_per_heavy':[1.,999.,100.,np.inf,np.nan],
        'relax_rms_surround_A':[.2,.3,np.nan,.4,np.inf]})
    e=converged_energy(d)
    assert e.tolist()==[1.] and len(e)/len(d)==.2
    assert finite_values(d,'relax_rms_surround_A').tolist()==[.2,.3,.4]


def test_missing_status_is_not_silently_assumed_converged():
    d=pd.DataFrame({'mmff_relief_per_heavy':[1.]})
    assert converged_energy(d).empty
