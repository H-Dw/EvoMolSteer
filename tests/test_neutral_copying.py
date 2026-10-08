import importlib.util
from pathlib import Path
import numpy as np


def test_exact_uniform_parent_occupancy_matches_analytic_first_step():
    source=Path(__file__).resolve().parents[1]/'scripts/analyze_neutral_copying.py'
    spec=importlib.util.spec_from_file_location('neutral_copying_under_test',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for n in [2,5,50]:
        transition=module.occupancy_transition(n)
        assert np.allclose(transition.sum(1),1)
        assert np.isclose(transition[n]@np.arange(n+1),n*(1-(1-1/n)**n))
        assert transition[1,1]==1 and (transition[np.triu_indices(n+1,1)]==0).all()
