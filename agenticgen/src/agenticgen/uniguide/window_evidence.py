"""Read-only window coverage and observed update magnitudes, never mechanism claims."""
from pathlib import Path
import statistics
from . import linear_scale as scale
from .box_search import grids
from .io import read,sha,value_sha


def context(config,history,domain,best):
    actual=best[scale.EFFECTIVE]
    coverage=[]
    for i,grid in enumerate(grids(domain)):
        same=[r for r in history if all(r[scale.EFFECTIVE][j]==actual[j] for j in range(len(actual)) if j!=i)]
        levels=sorted({r[scale.EFFECTIVE][i] for r in same})
        coverage.append(dict(window_id=i,allowed=domain['guidance_box_mask'][i],declared_grid=grid,
            observed_levels_any_background=sorted({r[scale.EFFECTIVE][i] for r in history}),
            current_background_other_scales=[v for j,v in enumerate(actual) if j!=i],
            observed_levels_at_current_background=levels,
            unmeasured_levels_at_current_background=[v for v in grid if v not in levels],
            source_ids_at_current_background=[r['id'] for r in same],
            optimum_proved=False,boundary_unresolved=actual[i] in (grid[0],grid[-1])))
    folder=Path(config['root'])/best['id']
    diagnostics=dict(source_id=best['id'],status='unavailable',interpretation='observation only; not a surrogate for terminal performance')
    if (folder/'steps.json').exists() and (folder/'COMPLETE.json').exists():
        from .geometry_sampling import validate_receipt
        validate_receipt(folder)
        rows=read(folder/'steps.json')
        if rows and all(k in rows[0] for k in ('delta_native_ratio','max_atom_increment_A','assignment_switch_fraction')):
            windows=[]
            for i,(lo,hi) in enumerate(domain['guidance_windows']):
                active=[r for r in rows if max(1,lo)<=r['step']<=min(98,hi)]
                result=dict(window_id=i,effective_steps=[max(1,lo),min(98,hi)])
                for key in ('delta_native_ratio','max_atom_increment_A','assignment_switch_fraction','gradient_norm'):
                    values=[v for r in active for v in r.get(key,[])]
                    if values:
                        result[key]=dict(median=statistics.median(values),maximum=max(values),observations=len(values))
                windows.append(result)
            diagnostics.update(status='observed',steps_sha256=sha(folder/'steps.json'),windows=windows)
    body=dict(coverage=coverage,incumbent_update_diagnostics=diagnostics,
        evidence_scope='complete saved profiles and incumbent steps only; different backgrounds are not per-window causal evidence',
        large_update_is_not_failure=True,small_update_is_not_futility=True,
        no_new_sampling=True,no_gradient_clipping=True)
    return dict(body,context_sha256=value_sha(body))
