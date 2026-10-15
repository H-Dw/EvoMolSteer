"""Reduced measured gradient telemetry, not stored full Jacobians."""
import numpy as np
from evomolsteer.io import read_json


def summarize(folder):
    audit=read_json(folder/'guidance_audit.json');rows=[r for r in audit['steps'] if r['active']]
    keys=sorted({k for r in rows for k,v in r.items() if isinstance(v,(float,int)) and any(s in k for s in ['rms','cosine','clip','_error','gain','switch'])})
    return {'program':audit['program']['program_id'] if audit['program'] else None,
        'active_updates':len(rows),'nonzero_updates':sum(r.get('nonzero_molecules',0)>0 for r in rows),
        'statistics':{k:{'first':float(v[0]),'last':float(v[-1]),'mean':float(np.mean(v)),'max':float(np.max(v))}
            for k in keys if (v:=[r[k] for r in rows if isinstance(r.get(k),(float,int))])},
        'meaning':'Measured current-program response; no extra inference, no terminal-affinity gradient.'}
