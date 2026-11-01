"""Explicit endpoint semantics for the optional zero-start raw999 schedule."""

POLICY = "native-zero-uncovered-last-v1"


def policy(config):
    value = config.get("guidance_endpoints")
    if value is None:
        return None
    if value != {"policy": POLICY}:
        raise ValueError("Unknown gradient guidance endpoint policy")
    return dict(policy=POLICY, gradient_update_steps=[1,98],
                zero_step="native_transition_only; no VJP or extra gradient update at t=0",
                last_step="capture_uncovered_prediction; no VJP or extra gradient update",
                native_network_times_unchanged=True, native_transition_unchanged=True,
                extra_terminal_forward=False, coordinate_projection=False)


def apply(values, step, config):
    return [0.0]*len(values) if policy(config) is not None and step in (0,99) else values
