"""Opt-in expanded, fine-early search space; old frozen protocols stay unchanged."""
import math

POLICY = "negative-box-exclusion-v2"
SEARCH_POLICY = "coverage-aware-five-branch-v1"
RELEASE = "2.12.0"
SPLIT_RELEASE = "2.13.0"


def enabled(value):
    from .path_search import enabled as path_search
    return (path_search(value) or value.get("box_exclusion", {}).get("policy") == POLICY
            or value.get("box_exclusion_policy") == POLICY)


def configure(config):
    if not enabled(config):
        if "box_search" in config:
            raise ValueError("Expanded box search requires its explicit new protocol")
        return config
    maximum = config["guidance_control"].get("search_maximum")
    from .guidance_windows import windows
    count=len(windows(config))
    if type(maximum) is not int or maximum<10 or maximum%10:
        raise ValueError('Declare a finite expanded search maximum on the later-window grid')
    defaults = dict(policy=SEARCH_POLICY,
        window_grids=[[0,1,2,5,*range(10,maximum+1,10)],
                      *[list(range(0,maximum+1,10)) for _ in range(count-1)]],
        teacher_window_grids={}, early_guidance_shape="constant")
    settings = dict(defaults, **config.get("box_search", {}))
    if set(settings) != set(defaults) or settings["policy"] != SEARCH_POLICY:
        raise ValueError("Unknown expanded box search settings")
    if settings["early_guidance_shape"] not in ("constant", "linear_ramp"):
        raise ValueError("Unknown first-box dose shape")
    if not isinstance(settings["teacher_window_grids"], dict):
        raise ValueError("Teacher grids must be explicit mappings")
    for target in [None, *settings["teacher_window_grids"]]:
        validate_grids(settings["teacher_window_grids"].get(target, settings["window_grids"]), maximum,count)
    return dict(config, box_search=settings)


def validate_grids(values, maximum, count=6):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("One explicit grid per declared guidance window required")
    for grid in values:
        if (not isinstance(grid, list) or len(grid) < 2
                or any(type(x) is not int or not 0 <= x <= maximum for x in grid)
                or grid != sorted(set(grid)) or grid[0] != 0):
            raise ValueError("Window grids require unique increasing nonnegative integer scales, including zero")


def grids(value, teacher=None):
    if "search_grids" in value:
        return value["search_grids"]
    settings = value["box_search"]
    return settings["teacher_window_grids"].get(teacher, settings["window_grids"])


def validate_profile(actual, domain):
    if (not isinstance(actual, (list, tuple)) or len(actual) != len(grids(domain))
            or any(type(x) is not int or x not in grid for x,grid in zip(actual,grids(domain)))):
        raise ValueError("Actual integer scales must belong to each declared window grid")


def changes(left, right, domain):
    validate_profile(left, domain); validate_profile(right, domain)
    return [abs(grid.index(a)-grid.index(b)) for a,b,grid in zip(left,right,grids(domain))]


def distance(left, right, domain):
    return sum(changes(left,right,domain))


def classify(actual, best, domain):
    amount = distance(actual,best,domain)
    settings = domain["exploration_policy"]
    if amount <= 0 or amount > settings["bold_maximum_total_ticks"]:
        raise ValueError("Profile exceeds the declared grid-index distance")
    return next(tier for tier in ("focused","expanded","bold")
                if amount <= settings[tier+"_maximum_total_ticks"])


def early_shape(config):
    if not enabled(config):
        return None
    shape = config["box_search"]["early_guidance_shape"]
    return dict(policy=shape, window=[0,10], effective_steps=[1,10],
                integer_rounding="nearest_half_up", gradient_normalization=False)


def apply_shape(schedule, shape):
    if shape is None:
        return schedule
    if (not isinstance(shape,dict) or shape.get('policy') not in ('constant','linear_ramp')
            or shape != dict(policy=shape.get('policy'),window=[0,10],effective_steps=[1,10],
                integer_rounding='nearest_half_up',gradient_normalization=False)):
        raise ValueError('Unrecognized frozen early dose shape')
    result = list(schedule)
    if shape["policy"] == "linear_ramp":
        endpoint = result[10]
        for step in range(1,11):
            result[step] = math.floor(endpoint*step/10+.5)
    result[0] = result[99] = 0
    return result
