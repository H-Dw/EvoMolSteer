"""Finite differences on saved R0 states; no candidate sampling or RNG changes."""
import argparse
from pathlib import Path

import torch

from evomolsteer.adaptive_schedule.io import DEFAULT_ROOT, read, write, source_fingerprint
from evomolsteer.adaptive_schedule.flowr import load_model, load_pt, device, tensor_digest
from evomolsteer.adaptive_schedule.vendor.reward import LocalReward, displacement
from evomolsteer.generation.rng import rng_state, set_rng


def check(root):
    config = read(root / "config.json")
    evidence = []
    for teacher in read(root / "teachers.json"):
        if teacher["rank"] != 1:
            continue
        model = load_model(config, teacher["target"], root)
        folder = root / "branches" / teacher["teacher_id"] / "batch_000/r0"
        saved = load_pt(folder / "checkpoint_050.pt.gz")
        context = device(load_pt(saved["context_path"]))
        current, times, cond = device(saved["current"]), device(saved["times"]), device(saved["cond"])
        com, scale = context["com"][:, None], model.coord_scale
        reward = LocalReward(teacher, "C", "endpoint")
        set_rng(saved["rng"])
        before = tensor_digest(rng_state())
        affinity_grad = []

        def objective(x, assignment=None):
            output = model(dict(current, coords=x), context["pt"], times,
                           cond_batch=cond if model.self_condition else None,
                           pocket_equis=context["et"], pocket_invs=context["it"], training=False)
            for v in output.get("affinity", {}).values():
                if torch.is_tensor(v) and v.requires_grad:
                    v.register_hook(lambda g: affinity_grad.append(g))
            pred, _ = model._get_predictions(output)
            value, _, ids, _ = reward.loss(x * scale + com, pred["coords"] * scale + com,
                                         output, current["mask"].bool(), assignment)
            return -value, ids

        x = current["coords"].detach().clone().requires_grad_(True)
        value, assignment = objective(x)
        grad = torch.autograd.grad(value.sum(), x)[0].detach()
        # Use two largest components in different molecules to avoid cancellation.
        for slot in [0, 1]:
            flat = int(grad[slot].abs().argmax())
            atom, axis = divmod(flat, 3)
            analytical = float(grad[slot, atom, axis])
            differences = []
            for epsilon in [.03, .01, .003, .001]:
                positive, negative = x.detach().clone(), x.detach().clone()
                positive[slot, atom, axis] += epsilon
                negative[slot, atom, axis] -= epsilon
                with torch.no_grad():
                    p, _ = objective(positive, assignment)
                    n, _ = objective(negative, assignment)
                # Subtract only the perturbed molecule's reward. Subtracting two
                # batch sums loses the tiny response in unrelated molecules' loss.
                numerical = float((p[slot].double() - n[slot].double()) / (2 * epsilon))
                relative = abs(numerical - analytical) / max(abs(analytical), 1e-5)
                differences.append({"epsilon_native": epsilon, "numerical": numerical, "relative_error": relative})
            passed = abs(analytical) > 1e-6 and min(d["relative_error"] for d in differences) < .03
            evidence.append({"teacher": teacher["teacher_id"], "slot": slot, "atom": atom,
                             "axis": axis, "analytical": analytical, "differences": differences, "passed": passed})
        # A differentiable Angstrom-space objective must produce the same physical
        # displacement after the native-coordinate chain rule is accounted for.
        delta, factor = displacement(grad, .01, .5, scale, current["mask"])
        expected_world = .005 * (grad / scale) * current["mask"][..., None]
        expected_factor = (.025 / expected_world.norm(dim=-1).amax(1).clamp_min(1e-30)).clamp(max=1)
        expected_world *= expected_factor[:, None, None]
        units_ok = torch.allclose(delta * scale, expected_world, atol=1e-8, rtol=1e-5)
        rng_ok = tensor_digest(rng_state()) == before
        evidence.append({"teacher": teacher["teacher_id"], "units_passed": units_ok,
                         "max_atom_step_A": float((delta * scale).norm(dim=-1).max()),
                         "rng_unchanged": rng_ok, "affinity_vjp_calls": len(affinity_grad),
                         "parameters_frozen": all(not p.requires_grad for p in model.parameters()),
                         "passed": units_ok and rng_ok and not affinity_grad})
        del model
        torch.cuda.empty_cache()
    # Independent double-precision check of the scalar pseudo-Huber derivative.
    error = torch.tensor([.01, .3, 2., 20.], dtype=torch.float64, requires_grad=True)
    loss = error.square() / (torch.sqrt(1 + error.square()) + 1)
    derivative = torch.autograd.grad(loss.sum(), error)[0]
    epsilon = 1e-5
    rho = lambda e: e.square() / (torch.sqrt(1 + e.square()) + 1)
    fd = (rho(error.detach() + epsilon) - rho(error.detach() - epsilon)) / (2 * epsilon)
    evidence.append({"pseudo_huber_max_error": float((derivative - fd).abs().max()),
                     "passed": torch.allclose(derivative, fd, atol=1e-8, rtol=1e-7)})
    result = {"passed": all(r["passed"] for r in evidence), "checks": evidence,
              "source_fingerprint": source_fingerprint(),
              "additional_candidates": 0, "matching_frozen_for_derivative_test": True}
    write(root / "validation/gradient_checks.json", result)
    if not result["passed"]:
        raise RuntimeError("Gradient check failed; inspect validation/gradient_checks.json")
    print("Saved-state endpoint finite differences, coordinate units and RNG checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    check(parser.parse_args().root)
