"""Three jump-intensity hypotheses, independent of parameter direction."""

POLICY = "three-intensity-jump-search-v1"
TIERS = ("focused", "expanded", "bold")


def configure(config):
    maximum = config["guidance_control"]["maximum_total_tick_change"]
    defaults = dict(
        policy=POLICY,
        focused_maximum_total_ticks=2,
        expanded_maximum_total_ticks=6 if maximum > 6 else 4,
        bold_maximum_total_ticks=maximum,
    )
    settings = dict(defaults, **config.get("guidance_exploration", {}))
    if set(settings) != set(defaults) or settings["policy"] != POLICY:
        raise ValueError("Unknown jump exploration policy")
    bounds = [settings[name + "_maximum_total_ticks"] for name in TIERS]
    if any(type(value) is not int for value in bounds) or not (
        1 <= bounds[0] < bounds[1] < bounds[2] and bounds[2] == maximum
    ):
        raise ValueError(
            "Three ordered integer intensity ranges must fit the action budget"
        )
    return dict(config, guidance_exploration=settings)


def classify(updates, settings):
    distance = sum(abs(update["step"]) for update in updates)
    for tier in TIERS:
        if 0 < distance <= settings[tier + "_maximum_total_ticks"]:
            return tier
    raise ValueError("Action does not fit a declared jump intensity")


def describe(settings, unit):
    lower = 1
    result = []
    for tier in TIERS:
        upper = settings[tier + "_maximum_total_ticks"]
        result.append(
            dict(
                exploration_tier=tier,
                total_absolute_tick_change=[lower, upper],
                total_absolute_strength_change=[lower * unit, upper * unit],
                allowed_directions=["increase", "decrease", "mixed"],
                interpretation="Exploration distance only; no causal or ablation hypothesis required.",
            )
        )
        lower = upper + 1
    return result
