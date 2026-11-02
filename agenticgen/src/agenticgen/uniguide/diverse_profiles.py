"""Finite coverage suggestions and five executed hypotheses in the expanded space."""
import itertools
from . import linear_scale as scale
from .box_search import grids, distance, changes, classify, validate_profile
from .io import value_sha


def resolve(action, anchors, domain, seen, failures):
    if action.get('anchor_id') not in anchors:
        raise ValueError('Unknown compatible measured anchor')
    actual=action.get('target_scale_schedule')
    validate_profile(actual,domain)
    if any(v for v,on in zip(actual,domain['guidance_box_mask']) if not on):
        raise ValueError('An excluded box cannot be re-enabled')
    blocked=[*seen,*[r for r in failures if r.get('failure') or r.get('error') or
        r.get('evidence_status') in ('statistics_ineligible','measured_displacement_growth')]]
    if any(actual==r.get(scale.EFFECTIVE) for r in blocked):
        raise ValueError('Profile is already measured, reserved or failed')
    target=action.get('optimization_target',{})
    incumbent=anchors.get(target.get('incumbent_source_id'))
    if (incumbent is None or target.get('calibration_source_id')!=domain['initial_source']
            or target.get('purpose')!='seek_potential_global_improvement'):
        raise ValueError('Action must bind the real calibration root and current best')
    best=incumbent[scale.EFFECTIVE]
    tier=classify(actual,best,domain)
    if tier!=action.get('exploration_tier'):
        raise ValueError('Exploration tier differs from actual grid-index distance')
    if max(changes(actual,best,domain))>domain['control_settings']['maximum_window_tick_change']:
        raise ValueError('Per-window exploration distance exceeded')
    return dict(action,**scale.binding(scale.encode(actual,domain),domain),direction='mixed',
        total_absolute_tick_change=distance(actual,best,domain),distance_unit='per_window_grid_index',
        task_type='optimization',domain_sha256=domain['domain_sha256'],calibration_key=domain['calibration_key'],
        executor_resolved=True,agent_action_sha256=value_sha(action),stability_guarantee=False)


def bank(history,domain,best,failures=()):
    """Use window coverage and mixed profiles, rather than one local L1 ball."""
    active=[i for i,on in enumerate(domain['guidance_box_mask']) if on]
    blocked={tuple(r[scale.EFFECTIVE]) for r in [*history,*[r for r in failures if r.get('failure') or r.get('error') or
        r.get('evidence_status') in ('statistics_ineligible','measured_displacement_growth')]] if scale.EFFECTIVE in r}
    values=grids(domain)
    candidates={};indices={};digests={}
    def add(profile,opportunity):
        if tuple(profile) in blocked or profile==best:return
        try:tier=classify(profile,best,domain)
        except ValueError:return
        key=tuple(profile)
        indices[key]=tuple(grid.index(v) for grid,v in zip(values,profile));digests[key]=value_sha(profile)
        candidates.setdefault(key,dict(target_scale_schedule=profile,exploration_tier=tier,
            distance_from_global_best_ticks=distance(profile,best,domain),distance_unit='per_window_grid_index',
            search_opportunity=opportunity,performance=None,evidence_status='unmeasured',new_attempts=0))
    # All one-window legal doses remain visible, including weak first-box doses and frontiers.
    for i in active:
        for v in values[i]:
            p=list(best);p[i]=v;add(p,'early_dose' if i==0 and v<=20 else 'window_coverage')
    # Two-window jumps include the far edges, not just +/-20 around the incumbent.
    def representatives(i):
        grid=values[i];index=grid.index(best[i]);last=len(grid)-1
        positions={0,last,index,max(0,index-1),min(last,index+1),last//4,last//2,3*last//4}
        return sorted({grid[k] for k in positions}|{v for v in (1,2,5,200,250,320,400) if v in grid})
    for left,right in itertools.combinations(active,2):
        for a,b in itertools.product(representatives(left),representatives(right)):
            if a==best[left] or b==best[right]:continue
            p=list(best);p[left],p[right]=a,b
            add(p,'joint_frontier' if a>200 or b>200 else 'joint_reallocation')
    # Distinct full shapes in declared index space; no inference or mechanism scores.
    for pattern in ('ascending','descending','alternating','balanced_low','balanced_high'):
        for shift in (0,1,2):
            p=list(best)
            for k,i in enumerate(active):
                n=len(values[i])-1
                fraction= ((k+1)/(len(active)+1) if pattern=='ascending' else
                    (len(active)-k)/(len(active)+1) if pattern=='descending' else
                    (.15 if (k+shift)%2 else .85) if pattern=='alternating' else
                    .15+.1*shift if pattern=='balanced_low' else .65+.1*shift)
                p[i]=values[i][min(n,round(fraction*n))]
            add(p,'diverse_complete_profile')
    # 24 suggestions: reserve visibility for each window, each intensity and global shapes.
    result=[]
    def choose(options):
        options=[p for p in options if p not in result]
        if not options:return
        def separation(r,q):
            return sum(abs(a-b) for a,b in zip(indices[tuple(r['target_scale_schedule'])],indices[tuple(q['target_scale_schedule'])]))
        pick=max(options,key=lambda r:(min((separation(r,q) for q in result),default=0),digests[tuple(r['target_scale_schedule'])]))
        result.append(pick)
    all_rows=list(candidates.values())
    if 0 in active:
        for dose in (1,2,5):choose([r for r in all_rows if r['target_scale_schedule']==[dose,*best[1:]]])
    for i in sorted(active,key=lambda j:len({r[scale.EFFECTIVE][j] for r in history if scale.EFFECTIVE in r})):
        for weak in (True,False):
            choose([r for r in all_rows if r['target_scale_schedule'][i]!=best[i]
                and sum(a!=b for a,b in zip(r['target_scale_schedule'],best))==1
                and ((r['target_scale_schedule'][i]<best[i])==weak)])
    for tier in ('focused','expanded','bold'):
        choose([r for r in all_rows if r['exploration_tier']==tier])
    for opportunity in ('early_dose','joint_frontier','diverse_complete_profile'):
        for _ in range(2):choose([r for r in all_rows if r['search_opportunity']==opportunity])
    return result[:24]


def require_portfolio(actions,domain,best,suggestions,remaining,*,pairwise=True):
    if len(actions)>5:raise ValueError('At most five executable hypotheses per round')
    profiles=[a['target_scale_schedule'] for a in actions]
    if len({tuple(p) for p in profiles})!=len(profiles):raise ValueError('Exact duplicate hypotheses')
    active=sum(domain['guidance_box_mask'])
    if remaining>=50 and active>=2 and any(r['exploration_tier']=='bold' for r in suggestions):
        if not any(classify(p,best,domain)=='bold' and sum(a!=b for a,b in zip(p,best))>=2 for p in profiles):
            raise ValueError('Include a genuinely distinct multi-window bold opportunity')
    if pairwise and len(profiles)>=3 and active>=2:
        changed={tuple(i for i,(a,b) in enumerate(zip(p,best)) if a!=b) for p in profiles}
        if len(changed)<min(3,len(profiles)):
            raise ValueError('The portfolio repeats the same changed-window subset')
    # A same-background dose bracket is deliberately legal. Exact duplicates stay forbidden.
    # This rule never weakens failure identities, gate validation, or cumulative admission.
