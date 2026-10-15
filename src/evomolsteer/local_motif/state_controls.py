"""Full saved molecular-state clone keys for local trajectory evidence."""
import numpy as np
from evomolsteer.nc_guidance.window_features import unique_state_controls


def full_state_controls(trajectory,step,coordinates,atomics,parent):
    n=len(coordinates)
    categories=[np.asarray(atomics).reshape(n,-1)]
    for key in ('current_bonds','current_charges'):
        categories.append(np.asarray(trajectory.read(key,step)).reshape(n,-1))
    return unique_state_controls(coordinates,np.concatenate(categories,axis=1),parent)
