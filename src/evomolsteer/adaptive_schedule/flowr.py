"""Audited native loop instrumentation, complete restart states and passive noise capture."""
import ast
import copy
import gzip
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time
import types
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .io import read, write, sha, value_sha, SEEDS
from .contracts import expanded
from .vendor.reward import LocalReward, displacement
from .vendor.common import EXPECTED_CHECKPOINT, EXPECTED_SAMPLER, campaign, protein, HIST
from evomolsteer.generation.rng import rng_state, set_rng


def tree(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [tree(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(tree(v, fn) for v in value)
    return copy.deepcopy(value)


def cpu(value):
    return tree(value, lambda x: x.detach().cpu().clone())


def device(value):
    return tree(value, lambda x: x.detach().clone().to("cuda"))


def detach(value):
    return tree(value, lambda x: x.detach())


def tensor_digest(value):
    def norm(v):
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            return [str(v.dtype), list(v.shape), hashlib.sha256(v.tobytes()).hexdigest()]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    return value_sha(norm(value))


def save_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        torch.save(cpu(value), f)
    os.replace(tmp, path)
    write(path.with_suffix(".sha.json"), {"sha256": sha(path)})


def load_pt(path):
    path = Path(path)
    if sha(path) != read(path.with_suffix(".sha.json"))["sha256"]:
        raise ValueError("Checkpoint checksum mismatch: " + str(path))
    with gzip.open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def load_initial(config, target, batch):
    path = Path(config["history"]) / "initials" / target / f"batch_{batch:03d}.pt.gz"
    meta = read(path.with_suffix(".json"))
    if sha(path) != meta["state_file_sha256"]:
        raise ValueError("Historical initial snapshot checksum mismatch")
    with gzip.open(path, "rb") as f:
        state = torch.load(f, map_location="cpu", weights_only=False)
    if state["seed"] != SEEDS[batch] or state["prior"]["coords"].shape[0] != 20:
        raise ValueError("Historical batch/seed mismatch")
    return state, meta


def load_model(config, target, root):
    flowr_root = Path(config["flowr"])
    sys.path.insert(0, str(flowr_root))
    import flowr
    from flowr.scriptutil import load_model as native_load
    if not Path(flowr.__file__).resolve().is_relative_to(flowr_root.resolve()):
        raise ValueError("Wrong FLOWR import path")
    ckpt = flowr_root / "checkpoints/flowr_root_v2.ckpt"
    if sha(ckpt) != EXPECTED_CHECKPOINT or sha(flowr_root / "flowr/models/fm_pocket.py") != EXPECTED_SAMPLER:
        raise ValueError("Native sampler or checkpoint no longer matches original audit")
    args = SimpleNamespace(**read(campaign("steer", target) / "config.json")["flowr_args"])
    for side in ["target", "untarget"]:
        setattr(args, "pdb_file_" + side, str(protein(target)))
        setattr(args, "ligand_file_" + side, str(HIST / "steer" / target / "inputs/target_ligand.sdf"))
    args.ckpt_path = str(ckpt)
    args.save_dir = str(root / "runtime" / target)
    args.seed, args.num_workers, args.batch_cost = 42, 0, 20
    args.sample_n_molecules_per_target = 100
    torch.set_float32_matmul_precision("high")
    model = native_load(args)[0].to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.integrator.use_sde_simulation = True
    model.integrator.coord_noise_level = .2
    if model.inpainting_mode or model.graph_inpainting or model.integrator.use_cosine_scheduler:
        raise ValueError("Unsupported inpainting or changed native coordinate scheduler")
    if model.integrator.type_strategy != "uniform-sample" or model.integrator.bond_strategy != "uniform-sample":
        raise ValueError("Native categorical strategy differs from audited uniform sampler")
    instrument(model, root / "provenance")
    return model


def instrument(model, provenance):
    """Insert callbacks without rewriting native forward/finalizer/transition formulas."""
    original = model._generate_selective
    source = textwrap.dedent(inspect.getsource(original))
    parsed = ast.parse(source)
    fn = parsed.body[0]
    guards = [n for n in fn.body if isinstance(n, ast.Assert) and isinstance(n.test, ast.Name)
              and n.test.id == "apply_guidance"]
    if len(guards) != 1:
        raise ValueError("Unsupported native selective assertion")
    fn.body.remove(guards[0])
    loop = [n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
            and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "enumerate"
            and ast.unparse(n.iter.args[0]) == "step_sizes"]
    if len(loop) != 1:
        raise ValueError("Native integration loop is ambiguous")
    loop = loop[0]
    for parent in ast.walk(fn):
        for field, nodes in ast.iter_fields(parent):
            if isinstance(nodes, list) and loop in nodes:
                pos = nodes.index(loop)
                nodes[pos:pos] = ast.parse(
                    "curr, cond_batch, times, pocket_equis_target, pocket_invs_target, "
                    "pocket_equis_untarget, pocket_invs_untarget = self._adaptive.initialize("
                    "curr, cond_batch, times, prior, pocket_data_target, pocket_data_untarget, "
                    "pocket_equis_target, pocket_invs_target, pocket_equis_untarget, pocket_invs_untarget, time_points)"
                ).body
                pos = nodes.index(loop)
                nodes[pos + 1:pos + 1] = ast.parse(
                    "self._adaptive.checkpoint(100, curr, cond_batch, times)"
                ).body
                break
        else:
            continue
        break
    forwards = 0
    integrates = 0
    for i in range(len(loop.body) - 1, -1, -1):
        n = loop.body[i]
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id == "out_target":
                loop.body[i] = ast.parse(
                    "out_target = self._adaptive.forward(curr,pocket_data_target,times,cond,"
                    "pocket_equis_target,pocket_invs_target,i,step_size)"
                ).body[0]
                forwards += 1
            if n.targets[0].id == "curr" and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "self.integrator.step":
                loop.body[i + 1:i + 1] = ast.parse("curr = self._adaptive.after(curr,i)").body
                integrates += 1
    if (forwards, integrates) != (1, 1):
        raise ValueError("Native forward or integration site changed")
    loop.body[:0] = ast.parse(
        "if i < self._adaptive.start_step:\n    continue\n"
        "self._adaptive.checkpoint(i,curr,cond_batch,times)"
    ).body
    ast.fix_missing_locations(parsed)
    modified = ast.unparse(parsed) + "\n"
    scope = dict(original.__func__.__globals__)
    exec(compile(modified, "adaptive_native_selective.py", "exec"), scope)
    model._generate_selective = types.MethodType(scope["_generate_selective"], model)
    provenance.mkdir(parents=True, exist_ok=True)
    for name, text in [("native_selective.py", source), ("instrumented_selective.py", modified)]:
        p = provenance / name
        if p.exists() and p.read_text() != text:
            raise ValueError("Different native instrumentation in an existing experiment")
        p.write_text(text)

    integrator = model.integrator
    coord = integrator.coord_step
    coord_source = textwrap.dedent(inspect.getsource(coord))

    class RandomRecorder(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "torch" and node.func.attr in {"randn", "randn_like"}:
                    return ast.copy_location(ast.Call(
                        func=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="_adaptive_randn", ctx=ast.Load()),
                        args=[ast.Constant(node.func.attr)] + node.args, keywords=node.keywords), node)
            return node
    tree_coord = ast.fix_missing_locations(RandomRecorder().visit(ast.parse(coord_source)))
    coord_namespace = dict(coord.__func__.__globals__)
    exec(compile(tree_coord, "adaptive_native_coord.py", "exec"), coord_namespace)
    integrator.coord_step = types.MethodType(coord_namespace["coord_step"], integrator)

    def draw(this, name, *args, **kwargs):
        sample = getattr(torch, name)(*args, **kwargs)
        this._adaptive_observer.noise_samples.append(sample.detach().clone())
        return sample
    integrator._adaptive_randn = types.MethodType(draw, integrator)
    cat = integrator._uniform_sample_step
    cat_source = textwrap.dedent(inspect.getsource(cat))
    token = "    return samples"
    if cat_source.count(token) != 1:
        raise ValueError("Native categorical sampler changed")
    cat_source = cat_source.replace(token,
        "    self._adaptive_observer.categories.append((step_probs.detach().clone(), samples.detach().clone()))\n"
        + token)
    cat_namespace = dict(cat.__func__.__globals__)
    exec(compile(cat_source, "adaptive_native_category.py", "exec"), cat_namespace)
    integrator._uniform_sample_step = types.MethodType(cat_namespace["_uniform_sample_step"], integrator)


def compose(states, slots=None):
    """Join independent slot rows only when all non-slot continuation context agrees."""
    if len(states) != 20:
        raise ValueError("A complete native batch of twenty parents is required")
    slots = list(range(20)) if slots is None else slots
    first = states[0]
    for s in states[1:]:
        for key in ["step", "context_sha256", "rng_sha256", "integrator"]:
            if s[key] != first[key]:
                raise ValueError("Parent contexts/RNG differ; require full-batch branch fallback")
        if tensor_digest(s["times"]) != tensor_digest(first["times"]):
            raise ValueError("Parent times differ")
    result = copy.deepcopy(first)
    for key in ["current", "cond", "previous_assignment"]:
        def join(values):
            v = values[0]
            if torch.is_tensor(v):
                if v.ndim and v.shape[0] == 20:
                    return torch.cat([x[slot:slot + 1] for x, slot in zip(values, slots)], 0)
                if any(not torch.equal(v, x) for x in values[1:]):
                    raise ValueError("Non-slot tensor differs between parents")
                return v.clone()
            if isinstance(v, dict):
                return {k: join([x[k] for x in values]) for k in v}
            if v is None:
                return None
            if any(x != v for x in values[1:]):
                raise ValueError("Non-slot context differs")
            return copy.deepcopy(v)
        result[key] = join([s[key] for s in states])
    return result


class Control:
    def __init__(self, model, template, initial, schedules, folder, context_path, resume=None):
        self.model, self.template, self.initial = model, template, initial
        self.folder, self.context_path = Path(folder), Path(context_path)
        self.resume = resume
        self.start_step = resume["step"] if resume else 0
        self.reward = LocalReward(template, "C", "endpoint")
        self.schedules = torch.tensor([expanded(x) for x in schedules], device="cuda")
        self.com = initial["target_com"].to("cuda").float()
        self.rows, self.nodes = [], {}
        self.trace = h5py.File(self.folder / "trace.h5", "w")
        self.noise_samples, self.categories = [], []
        self.last_assignment = None
        model.integrator._adaptive_observer = self

    def initialize(self, curr, cond, times, prior, pt, po, et, it, eo, io, grid):
        context = {"prior": prior, "pt": {k: v for k, v in pt.items() if k != "complex"},
                   "po": {k: v for k, v in po.items() if k != "complex"}, "com": self.com,
                   "et": et, "it": it, "eo": eo, "io": io, "grid": grid,
                   "coord_scale": float(self.model.coord_scale), "self_condition": self.model.self_condition,
                   "integrator": self.model.integrator.hparams,
                   "sampler_sha256": EXPECTED_SAMPLER, "checkpoint_sha256": EXPECTED_CHECKPOINT,
                   "precision": torch.get_float32_matmul_precision(), "device": torch.cuda.get_device_name(0)}
        self.context_sha = tensor_digest(context)
        if self.context_path.exists():
            saved = load_pt(self.context_path)
            if tensor_digest(saved) != self.context_sha:
                raise ValueError("Shared native context changed")
        else:
            save_pt(self.context_path, context)
        if self.resume:
            r = self.resume
            if r["context_sha256"] != self.context_sha:
                raise ValueError("Resume context/source/model mismatch")
            curr, cond, times = device(r["current"]), device(r["cond"]), device(r["times"])
            self.model.integrator._guard_warned = r["integrator"]["guard_warned"]
            self.last_assignment = device(r.get("previous_assignment"))
            set_rng(r["rng"])
            if tensor_digest(rng_state()) != r["rng_sha256"]:
                raise ValueError("RNG restore failed")
        return curr, cond, times, et, it, eo, io

    def checkpoint(self, step, curr, cond, times):
        if step % 10:
            return
        state = {"schema": 1, "step": step, "current": curr, "cond": cond, "times": times,
                 "rng": rng_state(), "context_sha256": self.context_sha,
                 "context_path": str(self.context_path), "previous_assignment": self.last_assignment,
                 "integrator": {"hparams": self.model.integrator.hparams,
                                "guard_warned": self.model.integrator._guard_warned}}
        state["rng_sha256"] = tensor_digest(state["rng"])
        save_pt(self.folder / f"checkpoint_{step:03d}.pt.gz", state)

    def store(self, key, step, value):
        a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if key not in self.trace:
            self.trace.create_dataset(key, shape=(100,) + a.shape, dtype=a.dtype,
                                      chunks=(1,) + a.shape, compression="gzip", compression_opts=1)
        self.trace[key][step] = a

    def forward(self, curr, pocket, times, cond, equis, invs, step, dt):
        self.noise_samples, self.categories = [], []
        self.before = curr["coords"].detach()
        self.times, self.dt = times, dt
        self.entry_rng = rng_state()
        before_hash = tensor_digest(self.entry_rng)
        with torch.enable_grad():
            x = self.before.clone().requires_grad_(True)
            out = self.model(dict(curr, coords=x), pocket, times, cond_batch=detach(cond),
                             pocket_equis=equis, pocket_invs=invs, training=False)
            for v in out.get("affinity", {}).values():
                if torch.is_tensor(v) and v.requires_grad:
                    def denied(g):
                        raise RuntimeError("Affinity head entered coordinate reward gradient")
                    v.register_hook(denied)
            pred, _ = self.model._get_predictions(out)
            endpoint = pred["coords"] * self.model.coord_scale + self.com[:, None]
            current = x * self.model.coord_scale + self.com[:, None]
            value, _, assignment, core = self.reward.loss(current, endpoint, out, curr["mask"].bool())
            gradient = torch.autograd.grad(-value.sum(), x)[0].detach()
        if tensor_digest(rng_state()) != before_hash:
            raise ValueError("Prediction/reward consumed native RNG")
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(value).all()):
            raise ValueError("Nonfinite reward/gradient")
        lam = self.schedules[:, step, None, None]
        self.delta, factor = displacement(gradient, dt, lam, self.model.coord_scale, curr["mask"])
        switches = (torch.zeros(20, device="cuda") if self.last_assignment is None
                    else (assignment != self.last_assignment).float().mean(1))
        self.last_assignment = assignment.detach()
        self.velocity = (pred["coords"].detach() - self.before) / (1 - times[0][:, None, None])
        gt = self.model.integrator.noise_scheduler(times[0][0])
        self.score = ((times[0][:, None, None] * self.velocity - self.before)
                      / (1 - times[0] + 1e-6)[:, None, None])
        self.drift = self.velocity + gt * self.score
        g = gradient / self.model.coord_scale
        drift_world = self.drift * self.model.coord_scale
        cosine = (g * drift_world).sum((1, 2)) / (
            g.flatten(1).norm(dim=1) * drift_world.flatten(1).norm(dim=1)).clamp_min(1e-30)
        self.rows.append({
            "step": step, "time": float(times[0][0]), "dt": float(dt),
            "lambda": self.schedules[:, step].tolist(), "loss": value.detach().tolist(),
            "clip_factor": factor.tolist(), "assignment_switch_fraction": switches.tolist(),
            "gradient_drift_cosine": cosine.tolist(), "rng_before": before_hash, "affinity_vjp": False})
        for key, v in {
            "current_coords_A": current, "endpoint_coords_A": endpoint, "gradient_world": g,
            "guidance_delta_A": self.delta * self.model.coord_scale, "assignment": assignment,
            "core_mask": core, "native_velocity": self.velocity, "native_drift": self.drift,
            "current_atomics": curr["atomics"].argmax(-1), "current_charges": curr["charges"].argmax(-1),
            "current_bonds": curr["bonds"].argmax(-1), "mask": curr["mask"],
        }.items():
            self.store(key, step, v)
        if step % 10 == 0:
            save_pt(self.folder / f"head_{step:03d}.pt.gz",
                    {"prediction": pred, "raw_head": out, "assignment": assignment,
                     "gradient_world": g, "rng_before": self.entry_rng, "rng_after": rng_state()})
            ids = assignment[:, self.reward.core_columns]
            batch = torch.arange(20, device="cuda")[:, None]
            self.nodes[str(step)] = {
                "step": step, "affinity_head_at_t": pred["affinity"]["pic50"].reshape(20).tolist(),
                "local_loss": value.detach().tolist(), "assignment": assignment.tolist(),
                "local_endpoint_A": endpoint[batch, ids].detach().tolist(),
                "local_current_A": current[batch, ids].detach().tolist(),
                "local_atom_labels": pred["atomics"].argmax(-1)[batch, ids].tolist(),
                "local_charge_labels": pred["charges"].argmax(-1)[batch, ids].tolist(),
                "atom_entropy": (-(pred["atomics"].clamp_min(1e-12).log() * pred["atomics"]).sum(-1)[batch, ids]).tolist()}
        return detach(out)

    def after(self, curr, step):
        if len(self.noise_samples) != 1 or len(self.categories) < 3:
            raise ValueError("Native random call structure changed")
        native = curr["coords"]
        gt = self.model.integrator.noise_scheduler(self.times[0][0])
        eps = self.noise_samples[0]
        std = torch.sqrt(2 * gt * self.model.integrator.coord_noise_level)
        noise_delta = eps * std * self.dt
        expected = (self.before + self.drift * self.dt + noise_delta) * curr["mask"][..., None]
        # Match the native expression's grouping, allowing only float rounding.
        if not torch.allclose(native, expected, atol=2e-5, rtol=2e-6):
            raise ValueError("Recorded drift/noise do not explain the native proposal")
        proposal = dict(curr, coords=native + self.delta)
        self.store("native_epsilon", step, eps)
        self.store("native_noise_delta", step, noise_delta)
        self.store("native_proposal_coords", step, native)
        self.store("guided_proposal_coords", step, proposal["coords"])
        for index, (probs, samples) in enumerate(self.categories):
            self.store(f"categorical_{index}_probabilities", step, probs)
            self.store(f"categorical_{index}_samples", step, samples)
        lengths = (self.delta * self.model.coord_scale).norm(dim=-1)
        core = self.trace["core_mask"][step]
        mask = curr["mask"].bool()
        core = torch.as_tensor(core, device=mask.device).bool() & mask
        outside = mask & ~core
        rms = lambda m: (lengths.square().masked_fill(~m, 0).sum(1) / m.sum(1).clamp_min(1)).sqrt().tolist()
        self.rows[-1].update(
            rng_after=tensor_digest(rng_state()), max_atom_guidance_A=lengths.amax(1).tolist(),
            inside_rms_A=rms(core), outside_rms_A=rms(outside),
            native_displacement_A=(native - self.before).flatten(1).norm(dim=1).mul(self.model.coord_scale).tolist(),
            guidance_native_ratio=(self.delta.flatten(1).norm(dim=1)
                                   / (native - self.before).flatten(1).norm(dim=1).clamp_min(1e-12)).tolist())
        if float(lengths.max()) > .025001:
            raise ValueError("Guidance displacement cap exceeded")
        if step % 10 == 9:
            self.trace.flush()
            write(self.folder / "steps.json", self.rows)
            write(self.folder / "nodes.json", self.nodes)
            write(self.folder / "progress.json", {"step_completed": step + 1, "time": time.time()})
        return proposal

    def close(self):
        self.trace.flush()
        self.trace.close()
        write(self.folder / "steps.json", self.rows)
        write(self.folder / "nodes.json", self.nodes)


def execute(model, config, template, batch, schedules, folder, context_path, resume=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    initial, meta = load_initial(config, template["target"], batch)
    pt, po, prior = device(initial["pocket_target"]), device(initial["pocket_off"]), device(initial["prior"])
    for pocket in [pt, po]:
        pocket["complex"] = [SimpleNamespace(com=x.numpy()) for x in initial["target_com"]]
    terminal = folder / "terminal.pt.gz"
    if terminal.exists():
        output = load_pt(terminal)
    else:
        control = Control(model, template, initial, schedules, folder, context_path, resume)
        model._adaptive = control
        set_rng(initial["rng"])
        started = time.time()
        try:
            output = model._generate_selective(
                prior=prior, pocket_data_target=pt, pocket_data_untarget=po, steps=100,
                times=[torch.zeros(20, device="cuda") for _ in range(3)],
                apply_guidance=False, coord_noise_level=.2)
            save_pt(terminal, output)
            save_pt(folder / "rng_terminal.pt.gz", rng_state())
            write(folder / "SAMPLING_COMPLETE.json",
                  {"seconds": time.time() - started, "start_step": control.start_step,
                   "end_step": 100, "steps_executed": len(control.rows),
                   "initial_sha256": meta["state_file_sha256"], "entry_seed": SEEDS[batch]})
        finally:
            control.close()
        del model._adaptive
    if not (folder / "EVALUATION_COMPLETE.json").exists():
        from evomolsteer.generation.controller import final_metrics
        # Evaluation must not change any subsequent branch's native random stream.
        shared_rng = rng_state()
        try:
            trace = SimpleNamespace(pt=pt, po=po, batch=batch, seed=SEEDS[batch],
                                    arm=folder.name, roots=np.arange(20), parents=np.arange(20), events=list(range(100)))
            records = final_metrics(model, device_output(output), trace, folder)
            write(folder / "final_records.json", records)
            write(folder / "EVALUATION_COMPLETE.json", {"attempts": len(records)})
        finally:
            set_rng(shared_rng)
    records = read(folder / "final_records.json")
    if len(records) != 20:
        raise ValueError("Native batch must retain all twenty attempts")
    return records


def device_output(output):
    # Native final coordinates/tensors are CPU; its nested affinity may be GPU.
    # final_metrics explicitly moves structural tensors itself.
    return tree(output, lambda x: x.detach().cpu())
