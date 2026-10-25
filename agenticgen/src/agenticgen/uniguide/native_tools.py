"""Source-pinned native instrumentation, copied from existing engineering only.

No legacy Control, reward, model loader, host config or strategy is imported.
"""
import ast, copy, gzip, hashlib, inspect, os, textwrap, types
from pathlib import Path
import numpy as np
import torch
from .io import read, write, sha, value_sha

def tree(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict) or hasattr(value, "items"):
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
        if isinstance(v, dict) or hasattr(v, "items"):
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
        observer = this._adaptive_observer
        sample = (observer.random_streams.normal(name, *args, **kwargs)
                  if hasattr(observer, "random_streams") else getattr(torch, name)(*args, **kwargs))
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
    native_draw = "torch.distributions.Categorical(step_probs).sample()"
    if cat_source.count(native_draw) != 1:
        raise ValueError("Native categorical random draw changed")
    cat_source = cat_source.replace(native_draw, "self._adaptive_categorical(step_probs)")
    def categorical(this, probabilities):
        observer = this._adaptive_observer
        if hasattr(observer, "random_streams"):
            return observer.random_streams.categorical(probabilities)
        return torch.distributions.Categorical(probabilities).sample()
    integrator._adaptive_categorical = types.MethodType(categorical, integrator)
    cat_namespace = dict(cat.__func__.__globals__)
    exec(compile(cat_source, "adaptive_native_category.py", "exec"), cat_namespace)
    integrator._uniform_sample_step = types.MethodType(cat_namespace["_uniform_sample_step"], integrator)

