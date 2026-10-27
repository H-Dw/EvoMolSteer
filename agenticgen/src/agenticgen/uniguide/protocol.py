"""Frozen schedules, log-space actions and explicit selection semantics."""
import math

SEEDS = [42, 100045, 200048, 300051, 400054]
WINDOWS = [(10, 25), (26, 50), (51, 75), (76, 90)]
BRANCHES = ["increase", "decrease", "redistribute"]


def expand(schedule):
    if len(schedule) != 4 or any(not math.isfinite(x) or x <= 0 for x in schedule):
        raise ValueError("Four positive actual scales required")
    return [next((schedule[w] for w, (lo, hi) in enumerate(WINDOWS) if lo <= step <= hi), 0.)
            for step in range(100)]


def lattice_index(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Scale must be positive and finite")
    j = 2 * math.log2(value)
    if abs(j - round(j)) > 1e-8:
        raise ValueError("Scale is outside the 2^(j/2) lattice; no rounding")
    return round(j)


def validate_action(proposal, anchors, domain, seen):
    if proposal["anchor_id"] not in anchors:
        raise ValueError("Action anchor must have measured results")
    before = anchors[proposal["anchor_id"]]["schedule"]
    after = proposal["schedule"]
    if len(after) != 4 or proposal["mode"] not in ("coarse", "fine"):
        raise ValueError("Invalid schedule/mode")
    old = [lattice_index(x) for x in before]
    new = [lattice_index(x) for x in after]
    lo, hi = domain["min_index"], domain["max_index"]
    if any(j < lo or j > hi for j in new):
        raise ValueError("Scale outside this teacher's calibrated domain")
    changes = [b-a for a,b in zip(old,new) if b != a]
    size = 2 if proposal["mode"] == "coarse" else 1
    if not 1 <= len(changes) <= 2 or any(abs(x) != size for x in changes):
        raise ValueError("Change one/two windows by one declared log-space step")
    branch = proposal["branch"]
    if branch == "increase" and not all(x > 0 for x in changes):
        raise ValueError("Increase may only increase")
    if branch == "decrease" and not all(x < 0 for x in changes):
        raise ValueError("Decrease may only decrease")
    if branch == "redistribute" and (len(changes) != 2 or sum(changes) != 0):
        raise ValueError("Redistribute requires two opposite log steps")
    if branch not in BRANCHES or tuple(new) in {tuple(lattice_index(x) for x in s) for s in seen}:
        raise ValueError("Invalid branch or duplicate schedule")
    return after


def choose(rows, band=.02):
    if not rows or any(not x.get("complete") for x in rows):
        raise ValueError("Selection requires complete all-attempt metrics")
    low = min(r["C"] for r in rows)
    return min((r for r in rows if r["C"] <= low + band), key=lambda r: (r["gap"], r["C"], -r["best"], r["id"]))


def regressed(previous, current):
    dc = current["C"] - previous["C"]
    return current["gap"] > previous["gap"] if abs(dc) <= .02 else dc > .02


def coarse_scales():
    return sorted({float(m * 10.**k) for k in range(-6, 6) for m in (1,3)} | {50., 1e6})
