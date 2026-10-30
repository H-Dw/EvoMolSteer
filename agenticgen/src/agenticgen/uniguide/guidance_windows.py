"""One window contract for actions, execution, diagnostics and calibration."""

DEFAULT = ((10, 25), (26, 50), (51, 75), (76, 99))
FIVE = ((10, 20), (21, 30), (31, 40), (41, 50), (51, 99))


def windows(value=None):
    source = value or {}
    result = source.get("guidance_windows", source.get("windows", DEFAULT))
    if not isinstance(result, (list, tuple)) or not 1 <= len(result) <= 10:
        raise ValueError("One to ten consecutive guidance windows required")
    expected = 10
    output = []
    for pair in result:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(i) is not int for i in pair)
        ):
            raise ValueError("Integer closed window bounds required")
        lo, hi = pair
        if lo != expected or hi < lo or hi > 99:
            raise ValueError("Windows must cover 10..99 exactly once without gaps")
        output.append([lo, hi])
        expected = hi + 1
    if expected != 100:
        raise ValueError("Guidance must end at step 99")
    return output


def expand(schedule, value=None):
    import math

    bounds = windows(value)
    if len(schedule) != len(bounds) or any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in schedule
    ):
        raise ValueError("Finite nonnegative strengths must match the window count")
    result = [0.0] * 100
    for strength, (lo, hi) in zip(schedule, bounds):
        result[lo : hi + 1] = [strength] * (hi - lo + 1)
    return result
