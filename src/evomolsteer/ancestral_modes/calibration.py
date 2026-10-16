"""Reference-path relative-field calibration from measured coordinate VJPs."""
import numpy as np
from evomolsteer.io import read_json,digest


def field_options(audit_path,targets,weight_bounds):
    audit=read_json(audit_path);old=float(audit['program']['typed_weight'])
    ratios=np.asarray([v for r in audit['steps'] for v in r.get('typed_gradient_fraction_of_total',[])],dtype=float)
    ratios=ratios[np.isfinite(ratios)&(ratios>0)]
    if old<=0 or len(ratios)==0:return []
    observed=float(np.median(ratios));result=[]
    for target in targets:
        value=float(np.clip(old*target/observed,*weight_bounds))
        if np.isclose(value,old):continue
        result.append({'old_weight':old,'new_weight':value,'target_reference_ratio':target,'observed_reference_median_ratio':observed,
            'measured_probe_values':len(ratios),'source_audit_sha256':digest(audit_path),'weight_bounds':weight_bounds,
            'prescription':'old_weight * target_ratio / median_reference_component_to_total_ratio, clipped to declared bounds',
            'limit':'First-order reference-path calibration only. Total gradient and new trajectory change with weight, so actual new ratio must be measured; no affinity gradient or extra model call.'})
    return result
