"""Deterministic trace audit; compact batch/node summaries, no model calls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import digest, write_json
from .policy import clock_grid, validate_policy


def audit_delivery(traces, program, output, steps):
    if not traces:
        raise ValueError("At least one completed trace required")
    a, b = program["window"]
    policy = program.get("temporal_policy")
    observed = clock_grid([a, b], steps)
    if policy:
        validate_policy(policy, [a, b], observed, steps)
    summaries, sources, controlled = [], {}, []
    for value in traces:
        path = Path(value)
        sources[str(path.resolve())] = digest(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != steps or [r["step"] for r in rows] != list(range(steps)):
            raise ValueError("Incomplete, repeated or unordered integration trace")
        batch = int(path.parent.name.split("_")[-1])
        n_controlled = 0
        for i, r in enumerate(rows):
            if not np.isclose(r["score_time"], i / steps, atol=2e-6, rtol=0) or not np.isclose(r["state_time"], (i + 1) / steps, atol=2e-6, rtol=0):
                raise ValueError("Score/state clocks differ from the frozen grid")
            if r["production_target_forward_calls"] != 1 or r["particle_resampled"] or not r["affinity_outputs_detached"]:
                raise ValueError("Extra production forward, particle selection or non-detached head")
            eligible = r["score_time"] >= a - 1e-6 and r["state_time"] <= b + 1e-6
            expected = eligible and r["arm"] == "gradient"
            if r["active"] != expected or r["reward_evaluated"] != (eligible and r["arm"] != "unguided"):
                raise ValueError("Control/reward did not respect learned support")
            injection = np.asarray(r["injection_l2_A"], float)
            if not np.isfinite(injection).all() or np.any(injection < 0) or not expected and np.any(injection != 0):
                raise ValueError("Nonfinite or inactive-window injection")
            if not expected:
                continue
            j = int(np.abs(observed - r["score_time"]).argmin())
            actual_policy = {}
            for field, key, default in [("temporal_dose_factor", "dose_factors", 1.),
                                         ("temporal_teacher_blend", "teacher_blend", 0.),
                                         ("temporal_delta_A", "delta_A", 1.)]:
                if policy and field not in r:
                    raise ValueError("Scheduled adapter did not expose its operative policy")
                values = np.asarray(r.get(field, [default]), float)
                target = policy[key][j] if policy else default
                if not np.isfinite(values).all() or not np.allclose(values, target, atol=2e-6, rtol=0):
                    raise ValueError("Executed temporal policy differs from its compiled table")
                actual_policy[key] = float(values.mean())
            def array(name, default=None):
                values = np.asarray(r[name] if name in r else default, float)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite trace statistic: " + name)
                return values
            requested, delivered, scale = [array(k) for k in ["requested_rms_A", "injection_rms_A", "calibration_rms_A"]]
            if requested.shape != delivered.shape or requested.shape != scale.shape:
                raise ValueError("Mismatched per-seed dose arrays")
            g = array("gradient_norm")
            row = {"batch": batch, "score_time": float(r["score_time"]), "state_time": float(r["state_time"]),
                   **actual_policy, "seeds": len(delivered),
                   "requested_rms_mean_A": float(requested.mean()), "delivered_rms_mean_A": float(delivered.mean()),
                   "delivered_rms_p95_A": float(np.quantile(delivered, .95)),
                   "calibration_rms_mean_A": float(scale.mean()),
                   "delivered_ratio_mean": float(np.divide(delivered, scale, out=np.zeros_like(delivered), where=scale > 0).mean()),
                   "zero_calibration_fraction": float((scale == 0).mean()),
                   "cap_triggered_fraction": float((array("cap_factor") < 1 - 1e-6).mean()),
                   "backtracked_fraction": float((array("backtrack_factor") < 1 - 1e-6).mean()),
                   "zero_injection_fraction": float((delivered == 0).mean()),
                   "gradient_l2_mean": float(g.mean()), "zero_gradient_fraction": float((g <= 1e-8).mean()),
                   "gradient_lag_cosine_mean": float(array("flowcompat_gradient_lag_cosine").mean()) if "flowcompat_gradient_lag_cosine" in r else None,
                   "first_order_reward_change_mean": float(array("first_order_reward_change").mean())}
            summaries.append(row)
            n_controlled += 1
        controlled.append({"batch": batch, "controlled_steps": n_controlled, "native_steps": steps - n_controlled})
    if len({c["batch"] for c in controlled}) != len(controlled):
        raise ValueError("Duplicate batch traces are not independent observations")
    frame = pd.DataFrame(summaries)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "batch_node_delivery.parquet", compression=None, index=False)
    report = {"passed": True, "trace_sources_sha256": sources, "window": [a, b], "steps": steps,
              "batches": controlled, "model_calls_from_this_tool": 0,
              "independent_unit": "batch; per-seed values summarized without erasing original trace",
              "no_time_bins": True,
              "limitations": ["First-order reward change is not a refreshed endpoint score.",
                              "Smooth weights do not prove smooth teacher identities or gradients.",
                              "This trace audit does not establish improved affinity, rare-mode coverage or neural-module causation."]}
    write_json(out / "delivery_audit.json", report)
    return report
