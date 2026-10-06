"""Shared physical-statistic coverage: convergence and finite values are explicit."""
import numpy as np
import pandas as pd


def finite_values(frame, name):
    values = pd.to_numeric(frame.get(name, pd.Series(index=frame.index, dtype=float)), errors='coerce')
    return values[np.isfinite(values)]


def converged_energy(frame, name='mmff_relief_per_heavy'):
    if 'energy_status' not in frame:
        return pd.Series(dtype=float)
    return finite_values(frame.loc[frame.energy_status.eq('converged')], name)
