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
    if not enabled(config):
        if 'guidance_box_mask' in config:
            raise ValueError('Box gates require the opt-in exclusion protocol')
        if config.get('agent_tools',{}).get('skill_release')=='2.11.0':
            raise ValueError('Skills 2.11.0 require explicit negative box exclusion')
        return config
    if config["box_exclusion"] != dict(enabled=True, policy=POLICY):
        raise ValueError("Unknown box exclusion settings")
    if windows(config) != [list(p) for p in SIX] or config["rounds"] != 5:
        raise ValueError("Box exclusion requires six boxes and exactly five rounds")
    if config["agent_tools"]["skill_release"] != "2.11.0":
        raise ValueError("Box exclusion requires skills 2.11.0")
    if config["guidance_control"]["maximum_attempts_per_teacher"] != 500:
        raise ValueError("Box pre-experiment cap remains 500")
    settings=config['guidance_control']
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
