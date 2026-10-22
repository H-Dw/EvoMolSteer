"""Inclusive native step labels requested for the four amplitude controls.

Time labels use k/100, not comparisons against accumulated float timesteps.
Checkpoints are saved before update k. Thus interval 2 starts from state 26,
while state 25 is also retained as the requested diagnostic control boundary.
"""
INTERVALS = ((10, 25), (26, 50), (51, 75), (76, 90))
CONTROL_NODES = (25, 50, 75)
CHECKPOINT_STEPS = (0, 10, 25, 26, 50, 51, 75, 76, 90, 91, 100)
HEAD_STEPS = CHECKPOINT_STEPS[:-1]
DIAGNOSTIC_INTERVALS = ((0, 9), *INTERVALS, (91, 99))
INITIAL_TICKS = (5, 5, 5, 5)


def description():
    return [{"index": i, "start_step": a, "end_step_inclusive": b,
             "time_labels": [a / 100, b / 100], "updates": b - a + 1}
            for i, (a, b) in enumerate(INTERVALS)]
