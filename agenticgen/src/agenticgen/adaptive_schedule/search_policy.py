"""Version-frozen range, coarse probe scale and fine grid are distinct controls."""
import copy
import math
from .io import read, sha, write


def policy(upper=1, coarse=None):
    return {'upper_amplitude': upper, 'coarse_probe_scale': (.2 if upper==1 else upper/10) if coarse is None else coarse,
            'fine_resolution': .1, 'max_range_probes': 1 if upper > 1 else 0,
            'max_changed_intervals': 2}


def validate_policy(value, version, previous=None):
    n = int(version.removeprefix('v')); upper = value['upper_amplitude']
    if n not in range(6, 26) or type(upper) is not int or not (upper == 1 or upper >= 5 and upper % 5 == 0):
        raise ValueError('Authorized versions V6..V25; range upper is 1 or a positive multiple of 5')
    if n <= 10 and upper != 1:
        raise ValueError('Range expansion is unavailable before full V10 review')
    if value['fine_resolution'] != .1 or value['max_changed_intervals'] != 2 or value['max_range_probes'] != (1 if upper > 1 else 0):
        raise ValueError('Fine grid, interval count and range-probe quota are fixed')
    scale = value['coarse_probe_scale']
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or not .1 <= scale <= upper or not math.isclose(scale*10, round(scale*10)):
        raise ValueError('Coarse scale must remain on the tenth grid, inside the range')
    if previous:
        old = previous['upper_amplitude']; next_upper = 5 if old == 1 else old + 5
        if upper > next_upper:
            raise ValueError('At most one authorized range level per version')
    return value


def for_config(config):
    return config.get('active_search_policy', policy())


def upper_tick(config):
    return int(for_config(config)['upper_amplitude']*10)


def validate_probe_quota(proposals, selected=None, active=None):
    active = active or policy()
    items = proposals if selected is None else [proposals[x] for x in selected]
    if sum(p.get('search_mode', 'local') == 'range_probe' for p in items) > active['max_range_probes']:
        raise ValueError('At most one range probe in the executed four; none before range expansion')


def load_version_policy(root, version):
    def read_frozen(name):
        folder=root/'versions'/name
        value=read(folder/'search_policy.json')
        if (folder/'binding.json').exists():
            from .io import value_sha
            if read(folder/'binding.json').get('search_policy_sha256') != value_sha(value):
                raise ValueError('Previously bound version search policy changed')
        return value
    path = root/'versions'/version/'search_policy.json'
    previous = None
    if int(version[1:]) > 6:
        prev = root/'versions'/f'v{int(version[1:])-1}'/'search_policy.json'
        if not prev.exists():
            raise ValueError('Previous frozen version range policy missing')
        previous = read_frozen(f'v{int(version[1:])-1}')
    if not path.exists() and (path.parent/'binding.json').exists():
        raise ValueError('Bound version search policy was removed')
    value = read_frozen(version) if path.exists() else copy.deepcopy(previous or policy())
    return validate_policy(value, version, previous)


def effective_config(root, version, config, freeze=False):
    result = copy.deepcopy(config)
    value = load_version_policy(root, version)
    result['active_search_policy'] = value
    result['tick_bounds'] = [0, int(value['upper_amplitude']*10)]
    if freeze:
        write(root/'versions'/version/'search_policy.json', value)
    return result
