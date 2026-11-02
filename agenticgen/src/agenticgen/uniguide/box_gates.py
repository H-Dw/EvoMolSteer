"""Optional time-box gates: native-only means no gradient AND no chemical overlay."""
from .guidance_windows import windows, SIX

POLICY = "negative-box-exclusion-v1"


def enabled(config):
    return bool(config.get("box_exclusion", {}).get("enabled", False))


def mask(config):
    value = config.get("guidance_box_mask", [True] * len(windows(config)))
    if not isinstance(value, list) or len(value) != len(windows(config)) or any(type(x) is not bool for x in value):
        raise ValueError("One explicit boolean per guidance box required")
    if not enabled(config) and "guidance_box_mask" in config:
        raise ValueError("Box gates require the opt-in exclusion protocol")
    return list(value)


def configure(config):
    from .box_search import enabled as expanded_search, RELEASE
    if 'box_search' in config and not expanded_search(config):
        raise ValueError('Window-specific search settings require the explicit new protocol')
    if not enabled(config):
        if 'guidance_box_mask' in config:
            raise ValueError('Box gates require the opt-in exclusion protocol')
        if config.get('agent_tools',{}).get('skill_release') in ('2.11.0', RELEASE):
            raise ValueError('Box-selection skills require explicit negative box exclusion')
        return config
    new_search=expanded_search(config)
    from .box_search import POLICY as new_policy, grids
    if config["box_exclusion"] != dict(enabled=True, policy=new_policy if new_search else POLICY):
        raise ValueError("Unknown box exclusion settings")
    if windows(config) != [list(p) for p in SIX] or config["rounds"] != 5:
        raise ValueError("Box exclusion requires six boxes and exactly five rounds")
    if config["agent_tools"]["skill_release"] != (RELEASE if new_search else "2.11.0"):
        raise ValueError("Box exclusion and its immutable skill release must use the same protocol")
    if config["guidance_control"]["maximum_attempts_per_teacher"] != 500:
        raise ValueError("Box pre-experiment cap remains 500")
    settings=config['guidance_control']
    if new_search:
        variants=[grids(config),*config['box_search']['teacher_window_grids'].values()]
        if (settings['search_minimum']!=0 or settings['guidance_scale_grid_unit']!=1 or settings['teacher_overrides']
                or settings['maximum_proposals']!=5 or settings['maximum_selected']!=5
                or settings['maximum_window_tick_change'] < max(len(g)-1 for v in variants for g in v)
                or settings['maximum_total_tick_change'] < max(sum(len(g)-1 for g in v) for v in variants)):
            raise ValueError('Expanded search needs five fully executable branches and a fully reachable declared grid')
        if config['guidance_exploration']['bold_maximum_total_ticks'] != settings['maximum_total_tick_change']:
            raise ValueError('Expanded jump distances differ from admission')
        from .box_exclusion import EXPANDED_BACKGROUNDS, profiles
        from .box_search import validate_profile
        for teacher, backgrounds in EXPANDED_BACKGROUNDS.items():
            domain={'search_grids':grids(config,teacher)}
            for spec in profiles(teacher,backgrounds).values():
                validate_profile(spec['actual'],domain)
        mask(config)
        return config
    if any(settings[name]!=value for name,value in dict(search_minimum=0,search_maximum=200,
        guidance_scale_grid_unit=10,maximum_window_tick_change=12,maximum_total_tick_change=12).items()) or settings['teacher_overrides']:
        raise ValueError('Box exclusion requires the declared absolute 0..200 grid and twelve-tick search distance')
    if [config['guidance_exploration'][name+'_maximum_total_ticks'] for name in ('focused','expanded','bold')]!=[2,6,12]:
        raise ValueError('Box exclusion requires focused/expanded/bold ranges 1..2/3..6/7..12')
    mask(config)
    return config


def step_enabled(config, step):
    if not enabled(config):
        return True
    if not 1 <= step <= 98:
        return False
    return next(mask(config)[i] for i, (lo, hi) in enumerate(windows(config)) if lo <= step <= hi)


def apply(values, step, config):
    return values if step_enabled(config, step) else [0.] * len(values)


def validate_profile(config, actual):
    if enabled(config) and any(v != 0 for v, on in zip(actual, mask(config)) if not on):
        raise ValueError("Native-only boxes must explicitly request zero strength")
