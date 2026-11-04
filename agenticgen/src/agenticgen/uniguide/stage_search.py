"""Stage-based search metadata; never changes sampling identity or dynamics."""
import itertools
from .io import value_sha
from .linear_scale import EFFECTIVE
from .condition_evaluation import score as primary_score

POLICY = "stage-forward-backtracking-v2"
STAGES = ("early", "middle", "late")

def enabled(value):
    return value.get("path_search_policy") == POLICY or value.get("agent_tools", {}).get("skill_release") == "2.15.0"

def validate_binding(config, domain):
    if enabled(config) != enabled(domain):
        raise ValueError('Stage search release and frozen domain policy differ')


def partition_schema(domain):
    n = len(domain['search_grids'])
    properties = {s:dict(type='array',minItems=1,uniqueItems=True,items=dict(type='integer',minimum=0,maximum=n-1)) for s in STAGES}
    properties['reason'] = dict(type='string',minLength=1,maxLength=3000)
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

def validate_partition(partition, domain):
    import jsonschema
    try: jsonschema.validate(partition, partition_schema(domain))
    except jsonschema.ValidationError as error: raise ValueError(error.message) from error
    active = [i for i,on in enumerate(domain.get('guidance_box_mask',[True]*len(domain['search_grids']))) if on]
    if len(active)<3: raise ValueError('Three control stages require at least three controllable windows')
    if sum((partition[s] for s in STAGES),[]) != active:
        raise ValueError('Stages must be consecutive, disjoint, ordered and cover every controllable window')
    if len(active)>=4 and len(partition['early'])<2:
        raise ValueError('Early stage must contain at least two controllable windows')
    return partition

def focus_schema(anchors):
    props=dict(stage=dict(enum=list(STAGES)),operation=dict(enum=['start','advance','refine','backtrack']),source_id=dict(enum=list(anchors)))
    return dict(type='object',properties=props,required=list(props),additionalProperties=False)

def validate_focus(focus, context):
    previous=context.get('previous_focus')
    if previous is None:
        if focus['stage']!='early' or focus['operation']!='start': raise ValueError('Start in Early Control Stage')
    else:
        a,b=STAGES.index(previous['stage']),STAGES.index(focus['stage'])
        expected='advance' if b>a else 'backtrack' if b<a else 'refine'
        if focus['operation']!=expected: raise ValueError('Stage operation differs from focus transition')

def values(schedule, indices): return tuple(schedule[i] for i in indices)

def context(history, rounds, domain, incumbent, partition=None):
    partition=partition or next((r['stage_partition'] for r in rounds if r.get('stage_partition')),None)
    previous=next((r['search_focus'] for r in reversed(rounds) if r.get('search_focus')),None)
    result=dict(policy=POLICY,first_round=not rounds,previous_focus=previous,stage_partition=partition,
                stage_partition_sha256=value_sha(partition) if partition else None,prefix_families=[],highlighted_prefixes=[],
                pending_early_revisit=None,all_measured_anchors_available=True)
    if partition is None: return result
    validate_partition(partition,domain)
    for r in rounds:
        if r.get('stage_partition')!=partition: raise ValueError('Persisted stage partition changed')
        if r.get('stage_partition_sha256',value_sha(partition))!=value_sha(partition):
            raise ValueError('Persisted stage partition hash changed')
    prefix=partition['early']+partition['middle'];late=partition['late']
    by_id={r['id']:r for r in history}; families={}; adopted=[]; sequence=0
    formal_ids={g['id'] for r in rounds for g in r.get('groups',[])}
    def record(row):
        key=values(row[EFFECTIVE],prefix)
        f=families.setdefault(key,dict(prefix=list(key),source_ids=[],suffix_measurements=[],best_measured_mean=float('inf'),
            best_source_id=None,credit=1,paused_at=None))
        f['source_ids'].append(row['id'])
        f['suffix_measurements'].append(dict(suffix=list(values(row[EFFECTIVE],late)),source_id=row['id'],mean=primary_score(row),sequence=sequence))
        improved=primary_score(row)<f['best_measured_mean']
        if improved: f.update(best_measured_mean=primary_score(row),best_source_id=row['id'],credit=1,paused_at=None)
        return f,improved
    for row in history:
        if row['id'] not in formal_ids: record(row)
    for r in rounds:
        prior_prefixes=set(families)
        learning=r.get('checkpoint_learning',{})
        entering=by_id.get(learning.get('best_checkpoint_id'))
        if entering is None: raise ValueError('Stage replay requires the recorded entering incumbent')
        old_prefix=values(entering[EFFECTIVE],prefix)
        measured={g.get('selected_proposal',g.get('proposal',{})).get('id'):g for g in r.get('groups',[])}
        for action in r.get('stage_actions',[]):
            sequence+=1
            key=values(action['target_scale_schedule'],prefix)
            side=key in prior_prefixes and key!=old_prefix
            row=measured.get(action['id'])
            if row:
                f,improved=record(row)
                if side: f.update(credit=1 if improved else 0,paused_at=None if improved else sequence)
            elif side: families[key].update(credit=0,paused_at=sequence)
        sequence+=1  # Adoption follows the ordered attempts, including failures.
        candidates=[entering,*r.get('groups',[])]
        winner=min(candidates,key=primary_score)
        if values(winner[EFFECTIVE],late)!=values(entering[EFFECTIVE],late):
            adopted.append(dict(source_id=winner['id'],suffix=list(values(winner[EFFECTIVE],late)),sequence=sequence))
            if r is rounds[-1]: result['pending_early_revisit']=dict(source_id=winner['id'],suffix=adopted[-1]['suffix'],trigger_round=r['iteration'])
    current=values(incumbent[EFFECTIVE],prefix)
    for key,f in families.items():
        f['measured_suffixes']=len({tuple(v['suffix']) for v in f['suffix_measurements']})
        f['reactivation_options']=[]
        if not f['credit']:
            paused=f['paused_at']
            for e in adopted:
                if e['sequence']>paused: f['reactivation_options'].append(dict(suffix=e['suffix'],evidence_ids=[e['source_id']]))
            for other in families.values():
                for new in other['suffix_measurements']:
                    if new['sequence']<=paused: continue
                    for old in other['suffix_measurements']:
                        if old['sequence']<new['sequence'] and new['mean']<old['mean'] and new['suffix']!=old['suffix']:
                            f['reactivation_options'].append(dict(suffix=new['suffix'],evidence_ids=[old['source_id'],new['source_id']]))
        f['is_incumbent_prefix']=key==current
    representatives=sorted(families.values(),key=lambda f:(f['best_measured_mean'],f['best_source_id']))
    eligible=[f for f in representatives if f['credit'] or f['reactivation_options']]
    highlights=[families[current]] if current in families else []
    alternatives=[f for f in eligible if f not in highlights]
    if alternatives: highlights.append(alternatives.pop(0))
    if alternatives: highlights.append(min(alternatives,key=lambda f:(f['measured_suffixes'],f['best_measured_mean'],f['best_source_id'])))
    result.update(prefix_families=representatives,highlighted_prefixes=highlights,adopted_suffixes=adopted)
    return result

def lookup(actual, history, failures=()):
    for row in history:
        if row.get(EFFECTIVE)==actual:
            if not row.get('complete') or row.get('attempts')!=50: raise ValueError('Duplicate or incomplete profile is not a reusable completed population50')
            score=primary_score(row)
            return dict(status='measured',source_id=row['id'],mean_condition_aligned_rmsd_angstrom=score,required_new_attempts=0)
    for row in failures:
        if row.get(EFFECTIVE)==actual:
            return dict(status=('ineligible' if row.get('evidence_status') else 'reserved' if row.get('failure',{}).get('error')=='interrupted' else 'failed'),source_id=row.get('id'),required_new_attempts=None)
    return dict(status='unmeasured',required_new_attempts=50)

def audit(actions, history, domain, failures, remaining, learning, focus, proposed_partition=None):
    if len(actions)>domain['control_settings'].get('maximum_selected',5): raise ValueError('Too many stage proposals')
    if len({tuple(a['target_scale_schedule']) for a in actions})!=len(actions): raise ValueError('Exact duplicate stage hypotheses')
    state=learning['search_path_context']; saved=state.get('stage_partition')
    if saved is not None and proposed_partition is not None: raise ValueError('Stage partition is immutable after the first accepted decision')
    partition=validate_partition(saved or proposed_partition,domain)
    anchors={r['id']:r for r in history}
    import jsonschema
    try: jsonschema.validate(focus,focus_schema(anchors))
    except jsonschema.ValidationError as error: raise ValueError(error.message) from error
    validate_focus(focus,state)
    if focus['source_id'] not in anchors: raise ValueError('Unknown stage focus source')
    prefix=partition['early']+partition['middle'];late=partition['late'];early=partition['early']
    incumbent=anchors[learning['best_checkpoint_id']]
    if state['first_round']:
        state=context(history,[],domain,incumbent,partition=partition)
    families={tuple(f['prefix']):f for f in state['prefix_families']}
    seen_prefixes=set();blocked={tuple(r[EFFECTIVE]) for r in [*history,*failures] if EFFECTIVE in r}
    def branch_allowed(a):
        key=values(a['target_scale_schedule'],prefix);f=families.get(key)
        if not f or f['is_incumbent_prefix']: return True
        if f['credit']: return True
        return any(values(a['target_scale_schedule'],late)==tuple(o['suffix']) and set(o['evidence_ids'])<=set(a['evidence_ids']) for o in f['reactivation_options'])
    signatures=set();singles=set();focus_hit=False;revisit=False
    for a in actions:
        target=a['target_scale_schedule'];parent=anchors[a['anchor_id']][EFFECTIVE]
        changed=[i for i,(x,y) in enumerate(zip(target,parent)) if x!=y]
        signature=tuple((target[i]>parent[i])-(target[i]<parent[i]) for i in early)
        if any(signature): signatures.add(signature)
        if len(changed)==1 and changed[0] in early: singles.add(changed[0] if len(early)>1 else signature)
        focus_hit |= bool(set(changed)&set(partition[focus['stage']]))
        revisit |= a['anchor_id']==incumbent['id'] and values(target,late)==values(incumbent[EFFECTIVE],late) and any(signature)
        key=values(target,prefix);f=families.get(key)
        if f and not f['is_incumbent_prefix']:
            if key in seen_prefixes: raise ValueError('At most one suffix adaptation per historical nonwinner prefix in a batch')
            if not branch_allowed(a): raise ValueError('Paused prefix requires new relevant measured suffix evidence')
            seen_prefixes.add(key)
    # Exact availability in a product region: subtract blocked points and paused
    # prefix slices. The released grid distance bound spans the complete domain.
    active=set(prefix+late)
    grids=[g if i in active else [incumbent[EFFECTIVE][i]] for i,g in enumerate(domain['search_grids'])]
    import math
    if domain['control_settings']['maximum_total_tick_change']<sum(len(g)-1 for g in grids): raise ValueError('Stage availability requires the complete grid distance bound')
    def available(options):
        total=math.prod(map(len,options))
        if not total:return False
        constrained=set()
        for key,f in families.items():
            if f['is_incumbent_prefix'] or not all(v in options[i] for i,v in zip(prefix,key)):continue
            constrained.add(key)
            count=math.prod(len(options[i]) for i in late)
            total-=count
            matching=[p for p in blocked if values(p,prefix)==key and all(v in g for v,g in zip(p,options))]
            if f['credit']:
                available_suffixes=count-len(matching)
            else:
                supported={tuple(o['suffix']) for o in f['reactivation_options'] if all(v in options[i] for i,v in zip(late,o['suffix']))}
                available_suffixes=len(supported-{values(p,late) for p in matching})
            total+=min(1,max(0,available_suffixes))
        total-=sum(values(point,prefix) not in constrained and all(v in g for v,g in zip(point,options)) for point in blocked)
        return max(0,total)
    available_signatures=set();available_singles=set();can_focus=False;can_revisit=False
    for row in history:
        parent=row[EFFECTIVE]
        for sig in itertools.product((-1,0,1),repeat=len(early)):
            if not any(sig) or sig in available_signatures:continue
            opts=[list(g) for g in grids]
            for i,sign in zip(early,sig):opts[i]=[v for v in grids[i] if (v>parent[i])-(v<parent[i])==sign]
            if available(opts):available_signatures.add(sig)
        for i in early:
            for sign in (-1,1):
                opts=[[v] for v in parent];opts[i]=[v for v in grids[i] if (v>parent[i])-(v<parent[i])==sign]
                if available(opts):available_singles.add(i if len(early)>1 else (sign,))
        for i in partition[focus['stage']]:
            opts=[list(g) for g in grids];opts[i]=[v for v in grids[i] if v!=parent[i]]
            can_focus |= bool(available(opts))
    for i in early:
        opts=[list(g) for g in grids]
        for k in late:opts[k]=[incumbent[EFFECTIVE][k]]
        opts[i]=[v for v in grids[i] if v!=incumbent[EFFECTIVE][i]]
        can_revisit |= bool(available(opts))
    capacity=remaining//50
    required=min(2,capacity,len(available_signatures),available(grids))
    required_singles=min(2,capacity,len(available_singles),available(grids)) if state['first_round'] else 0
    if len(signatures)<required:raise ValueError('Need two distinct parent-relative Early stage directions when feasible')
    if len(singles)<required_singles:raise ValueError('First round requires independent single-window Early probes')
    if actions and can_focus and not focus_hit:raise ValueError('Declared stage focus must be changed by a proposal')
    pending=bool(state.get('pending_early_revisit'))
    if pending and capacity and can_revisit and not revisit:raise ValueError('New Late background requires an Early revision anchored at the current best')
    return dict(stage_partition=partition,stage_partition_sha256=value_sha(partition),required_early_directions=required,
        observed_early_directions=len(signatures),required_initial_single_window_probes=required_singles,
        focus_changed=focus_hit,early_revisit_required=pending and capacity>0 and can_revisit,early_revisit_proposed=revisit,
        availability=dict(early_signatures=len(available_signatures),single_window_directions=len(available_singles),focus=can_focus,revisit=can_revisit),
        exemption_reason='budget_or_exact_legal_availability' if required<2 or (pending and not(capacity and can_revisit)) else None)


def freeze_partition(root, domain, partition):
    from pathlib import Path
    from .io import frozen
    validate_partition(partition, domain)
    frozen(Path(root)/'stage_partitions'/(domain['teacher_id']+'.json'),
           dict(stage_partition=partition,stage_partition_sha256=value_sha(partition),domain_sha256=domain['domain_sha256']))

def restore_partition(root, domain, state):
    from pathlib import Path
    from .io import read
    path=Path(root)/'stage_partitions'/(domain['teacher_id']+'.json')
    if path.exists():
        saved=read(path);partition=validate_partition(saved['stage_partition'],domain)
        if saved['domain_sha256']!=domain['domain_sha256'] or saved['stage_partition_sha256']!=value_sha(partition):
            raise ValueError('Frozen stage mapping identity changed')
        if state['stage_partition'] is not None and state['stage_partition']!=partition:raise ValueError('Stage replay differs from frozen mapping')
        state.update(stage_partition=partition,stage_partition_sha256=value_sha(partition))
    return state
