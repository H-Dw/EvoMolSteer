"""Released early-first path context and deterministic portfolio checks."""
from collections import defaultdict
from .io import value_sha

RELEASE = "2.14.0"
POLICY = "early-forward-backtracking-v1"


def enabled(value):
    return (value.get("agent_tools", {}).get("skill_release") in (RELEASE, "2.15.0", "2.16.0")
            or value.get("path_search_policy") in (POLICY, "stage-forward-backtracking-v2"))


def validate_config(config):
    from .guidance_windows import SIX, windows
    from .box_search import grids
    from .condition_evaluation import CONFIG, POLICY as evaluation
    from .path_calibration import POLICY as calibration
    if config.get('calibration') != dict(search_policy=calibration):
        raise ValueError('Path search requires its deterministic fresh calibration policy')
    if config.get('checkpoint_learning') != dict(enabled=True) or config.get('prefix_inpainting') != dict(enabled=False):
        raise ValueError('Path search enables measured backtracking and retains the unprojected early trajectory')
    if config.get(CONFIG) != dict(policy=evaluation):
        raise ValueError("Path search requires explicit condition-only fitted mean evaluation")
    if config.get("box_exclusion", {"enabled": False}) != {"enabled": False}:
        raise ValueError("Path search disables box Selection")
    if "guidance_box_mask" in config and config["guidance_box_mask"] != [True] * 6:
        raise ValueError("All six path-search boxes must remain available")
    if windows(config) != [list(w) for w in SIX] or config["rounds"] != (10 if config["agent_tools"]["skill_release"]=="2.16.0" else 5):
        raise ValueError("Path search requires six windows and five rounds")
    settings = config["guidance_control"]
    required = dict(maximum_attempts_per_teacher=500, maximum_proposals=5, maximum_selected=5,
                    search_minimum=0, search_maximum=400, guidance_scale_grid_unit=1,
                    maximum_window_tick_change=43, maximum_total_tick_change=243)
    if any(settings[k] != v for k, v in required.items()) or settings["teacher_overrides"]:
        raise ValueError("Path search budget/grid differs from the released protocol")
    expected = [[0, 1, 2, 5, *range(10, 401, 10)], *[list(range(0, 401, 10)) for _ in range(5)]]
    if grids(config) != expected or config["box_search"]["teacher_window_grids"]:
        raise ValueError("Path search requires its six declared window grids")
    if config["box_search"]["early_guidance_shape"] != "constant":
        raise ValueError("The released path-search first-window shape is constant")
    return config


def validate_domain(domain):
    from .guidance_windows import SIX, windows
    from .condition_evaluation import CONFIG, POLICY as evaluation
    expected = [[0,1,2,5,*range(10,401,10)], *[list(range(0,401,10)) for _ in range(5)]]
    settings = domain['control_settings']
    if (windows(domain) != [list(w) for w in SIX] or domain.get(CONFIG) != dict(policy=evaluation)
            or domain.get('search_grids') != expected or domain.get('guidance_box_mask') != [True]*6
            or domain.get('box_exclusion_policy') or domain.get('selection_sha256')
            or domain.get('calibrated_guidance_reference_scale') != 0
            or domain.get('guidance_scale_grid_unit') != 1
            or settings['maximum_total_tick_change'] != 243 or settings['maximum_window_tick_change'] != 43):
        raise ValueError('Path domain must bind six open windows, full grids, and condition-only evaluation')


def context(history, completed_rounds):
    from .linear_scale import EFFECTIVE
    from .geometry_protocol import primary_score
    families = defaultdict(list)
    for row in history:
        families[tuple(row[EFFECTIVE][:4])].append(row)
    representatives = []
    for prefix, rows in families.items():
        best = min(rows, key=lambda r: (primary_score(r), r["id"]))
        representatives.append(dict(prefix=list(prefix), source_ids=[r["id"] for r in rows],
            measured_suffixes=len({tuple(r[EFFECTIVE][4:]) for r in rows}),
            best_source_id=best["id"], best_measured_mean=primary_score(best)))
    representatives.sort(key=lambda r: (r["best_measured_mean"], r["best_source_id"]))
    previous = next((r["search_focus"] for r in reversed(completed_rounds) if r.get("search_focus")), None)
    return dict(policy=POLICY, previous_focus=previous, first_round=previous is None,
                recommended_initial_window_ids=[0, 1], prefix_families=representatives,
                highlighted_prefixes=representatives[:3], all_measured_anchors_available=True,
                interpretation="Measured complete profiles; no intermediate inference state is reused")


def focus_schema(anchors):
    def obj(p): return dict(type="object", properties=p, required=list(p), additionalProperties=False)
    return obj(dict(window_id=dict(type="integer", minimum=0, maximum=5),
                    operation=dict(enum=["start", "advance", "refine", "backtrack"]),
                    source_id=dict(type="string", enum=list(anchors))))


def validate_focus(focus, path_context):
    previous = path_context["previous_focus"]
    index, operation = focus["window_id"], focus["operation"]
    if previous is None:
        if index not in (0, 1) or operation != "start":
            raise ValueError("Start path exploration at B1 or B2")
    else:
        old = previous["window_id"]
        expected = "advance" if index > old else "backtrack" if index < old else "refine"
        if operation != expected:
            raise ValueError("Path operation must agree with the recorded focus transition")


def early_signature(actual, parent, domain):
    return tuple((grid.index(a) > grid.index(b)) - (grid.index(a) < grid.index(b))
                 for a, b, grid in zip(actual[:4], parent[:4], domain["search_grids"][:4]))


def audit_early(actions, history, domain, failures, remaining):
    """Require two different signed supports; never count inherited prefix changes."""
    from .linear_scale import EFFECTIVE
    anchors = {r["id"]: r for r in history}
    blocked = {tuple(r[EFFECTIVE]) for r in [*history, *failures] if EFFECTIVE in r}
    signatures = {early_signature(a["target_scale_schedule"], anchors[a["anchor_id"]][EFFECTIVE], domain)
                  for a in actions}
    signatures.discard((0, 0, 0, 0))
    # The released distance caps span the full Cartesian grid. Count unused profiles
    # in each parent's signed early region exactly; do not infer exhaustion from a bank.
    import itertools
    import math
    from .geometry_protocol import choose
    grids = domain['search_grids']
    best = choose(history)[EFFECTIVE]
    blocked.add(tuple(best))
    blocked = {p for p in blocked if len(p) == len(grids) and all(v in g for v,g in zip(p,grids))}
    diameter = sum(len(g)-1 for g in grids)
    if domain['control_settings']['maximum_total_tick_change'] < diameter:
        raise ValueError('Exact early availability requires the released full-grid distance cap')
    available_profiles = math.prod(map(len, grids)) - len(blocked)
    available_signatures, witnesses = set(), []
    for row in history:
        parent = row[EFFECTIVE]
        for signature in itertools.product((-1,0,1), repeat=4):
            if not any(signature) or signature in available_signatures: continue
            options = [[v for v in g if (v > b)-(v < b) == sign]
                       for g,b,sign in zip(grids[:4],parent[:4],signature)] + grids[4:]
            count = math.prod(map(len, options))
            if not count: continue
            blocked_count = sum(all(v in g for v,g in zip(p,options)) for p in blocked)
            if count <= blocked_count: continue
            available_signatures.add(signature)
            if len(witnesses) < 2:
                candidate = next(list(p) for p in itertools.product(*options) if p not in blocked)
                witnesses.append(dict(anchor_id=row['id'], target_scale_schedule=candidate,
                                      signed_direction=list(signature)))
    required = min(2, remaining // 50, available_profiles, len(available_signatures))
    if len(signatures) < required:
        raise ValueError("Need at least %d distinct parent-relative early signed directions; found %d. "
                         "Available witnesses: %s" % (required, len(signatures), witnesses))
    reasons = []
    if remaining < 100: reasons.append('remaining_budget_below_100')
    if min(available_profiles,len(available_signatures)) < 2: reasons.append('insufficient_legal_early_candidates')
    return dict(required_early_directions=required, observed_early_directions=len(signatures),
                signed_directions=[list(s) for s in sorted(signatures)],
                reduction_reason=';'.join(reasons) if reasons else None,
                remaining_formal_attempts=remaining, available_profile_count=available_profiles,
                available_signed_direction_count=len(available_signatures),
                availability_witnesses=witnesses, source_history_sha256=value_sha(history))


def verified_evidence_receipts(folder, history, domain):
    """Recheck persisted calculation identities before execution, without a model call."""
    from pathlib import Path
    from .io import read
    folder = Path(folder)/'calculations'
    if not folder.exists(): return set()
    binding = read(folder/'BINDING.json')
    if binding.get('history_sha256') != value_sha(history) or binding.get('domain_sha256') != value_sha(domain):
        raise ValueError('Evidence registry no longer binds the submitted history/domain')
    observed = set()
    for path in folder.glob('ev_*.json'):
        receipt = read(path); source = receipt.get('binding', {})
        if source.get('registry_sha256') != value_sha(binding) or path.stem != 'ev_'+value_sha(source)[:20]:
            raise ValueError('Observed evidence receipt identity changed')
        observed.add(path.stem)
    return observed


def record_numerical_failure(config, subject, job, error):
    """Keep a failed whole population addressable after a batch numerical error.

    Native batches already reserve all candidate slots before inference. Register
    the failed population view without charging or replacing those slots again.
    """
    from pathlib import Path
    from .io import read, write, frozen
    from .geometry_sampling import compatibility, scale_binding, candidate_identity, reserve_group
    from .geometry_protocol import sampling_reference
    from .box_search import early_shape
    root, folder = Path(config['root']), Path(job['folder'])
    binding = compatibility(config,subject['teacher'],subject['condition'])
    scales = scale_binding(job['schedule'],sampling_reference(job.get('domain')),
        job.get('enabled',True),subject['condition'],job.get('loss_weights'),
        window_bounds=config['windows'],early_shape=early_shape(config))
    count = job.get('count',50)
    identity = dict(compatibility=binding,**candidate_identity(scales),
                    population=count,logical_slots_per_seed=count//5)
    identifier = folder.relative_to(root).as_posix()
    entry = root/'measurement_registry'/(value_sha(identity)+'.json')
    if entry.exists():
        saved = read(entry)
        if saved['identity'] != identity or (root/saved['source_group']/'COMPLETE.json').exists():
            raise ValueError('Numerical failure conflicts with an existing complete population')
    else:
        reserve_group(root,identifier,job['stage'],count,identity,charge=False)
    frozen(folder/'manifest.json',dict(id=identifier,identity=identity))
    write(folder/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),
          requested_population=count,failed_attempts_remain_charged=True,no_retry=True,no_replacement=True))
