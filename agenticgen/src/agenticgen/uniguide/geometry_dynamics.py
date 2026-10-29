"""Raw last-loop output and a candidate-local trust radius for guidance only."""
import math
import types
from pathlib import Path
import torch
from .io import write
from .native_tools import tree
from .geometry_chemistry import policy as chemistry_policy,STEP_FIELDS as CHEMICAL_STEP_FIELDS

WINDOWS=((10,25),(26,50),(51,75),(76,99))
LIMIT="maximum_single_atom_guidance_increment_angstrom"
STEP_FIELDS=("unconstrained_max_atom_increment_A","guidance_increment_scale_factor",
             "guidance_increment_limited")+CHEMICAL_STEP_FIELDS


def policy(config):
    from .absolute_coordinate import POLICY as coordinate_policy
    absolute = config.get("protocol") == coordinate_policy
    if absolute and LIMIT in config:
        raise ValueError("Absolute guidance is uncapped; remove the legacy increment-limit setting")
    limit=config.get(LIMIT)
    if not absolute and (type(limit) not in (int,float) or not math.isfinite(limit) or limit<=0):
        raise ValueError("Configure a positive finite "+LIMIT)
    if config.get("output_policy")!="raw999" or config.get("steps",100)!=100:
        raise ValueError("Geometry dynamics requires raw999 and 100 integration intervals")
    if config.get("windows",[list(w) for w in WINDOWS])!=[list(w) for w in WINDOWS]:
        raise ValueError("Geometry guidance windows must cover steps 10 through 99")
    restriction = (dict(limiter="none", limit_scope="none") if absolute else
        dict(maximum_single_atom_guidance_increment_angstrom=float(limit),
             limiter="candidate_uniform_direction_preserving_v1",limit_scope="guidance_increment_only"))
    return dict(output_policy="raw999",grid_end=.999,integration_intervals=100,
        intended_interval_width=.00999,windows=[list(w) for w in WINDOWS],
        **restriction,
        output_source="step_99_prediction_before_update",last_effective_guidance_update=98,
        extra_terminal_forward=False,coordinate_projection=False,chemical_projection=False,
        chemical_state_projection=chemistry_policy(config)["enabled"],chemical_inpainting=chemistry_policy(config))


def limit_increment(delta,valid,coord_scale,maximum):
    """Scale all valid-atom vectors of an offending molecule by one scalar.

    Use float64 norms to avoid misclassifying very large finite float32 vectors.
    Unaffected valid rows are bitwise unchanged. No random draw, atom clipping,
    native drift modification or batch-wide reduction enters this operation.
    """
    if type(maximum) not in (int,float) or not math.isfinite(maximum) or maximum<=0:
        raise ValueError("Positive finite guidance increment limit required")
    if not math.isfinite(float(coord_scale)) or coord_scale<=0:
        raise ValueError("Positive finite coordinate unit conversion required")
    if delta.ndim!=3 or delta.shape[-1]!=3 or valid.shape!=delta.shape[:2] or not delta.is_floating_point():
        raise ValueError("Guidance increment and atom mask shapes differ")
    mask=valid.bool()[...,None]
    raw=torch.where(mask,delta,torch.zeros_like(delta))
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("Nonfinite guidance cannot be repaired by a finite trust radius")
    raw_max=(raw.double()*float(coord_scale)).norm(dim=-1).amax(1)
    limited=raw_max>maximum
    # Small rounding margin only for limited rows ensures the stored vector
    # respects the bound after conversion back to the sampling dtype.
    radius=maximum*(1-4*torch.finfo(delta.dtype).eps)
    factor=torch.where(limited,radius/raw_max.clamp_min(torch.finfo(torch.float64).tiny),torch.ones_like(raw_max))
    result=torch.where(limited[:,None,None],(raw.double()*factor[:,None,None]).to(delta.dtype),raw)
    applied_max=(result.double()*float(coord_scale)).norm(dim=-1).amax(1)
    if bool((applied_max>maximum).any()):raise ArithmeticError("Guidance trust radius was not enforced")
    return result,dict(unconstrained_max_atom_increment_A=raw_max.tolist(),
        guidance_increment_scale_factor=factor.tolist(),guidance_increment_limited=limited.tolist())


def uncapped_increment(delta, valid, coord_scale, maximum=None):
    """Observe the original vector and return the same tensor, without a fallback."""
    from .condition import finite
    if maximum is not None:
        raise ValueError("Uncapped guidance cannot accept a radius")
    if delta.ndim != 3 or delta.shape[-1] != 3 or valid.shape != delta.shape[:2]:
        raise ValueError("Guidance vector/mask shapes differ")
    if not math.isfinite(float(coord_scale)) or coord_scale <= 0:
        raise ValueError("Positive coordinate conversion required")
    finite("uncapped guidance increment", delta)
    maximums = (delta.double() * float(coord_scale)).norm(dim=-1).amax(1)
    return delta, dict(unconstrained_max_atom_increment_A=maximums.tolist())


INCREMENT_OPERATORS = {"none": uncapped_increment,
                       "candidate_uniform_direction_preserving_v1": limit_increment}


class Raw999Sampling:
    """Sampling hooks shared by the geometry controller and engineering reference."""
    def sampling_function(self,config):
        from .joint_endpoint import compile_endpoint
        self.sampling_policy=policy(config)
        provenance=Path(config["root"])/"provenance/native"
        source=(provenance/"instrumented_selective.py").read_text(encoding="utf-8")
        function,rendered=compile_endpoint(self.model._generate_selective,source,True)
        target=provenance/"raw999_selective.py"
        if target.exists() and target.read_text(encoding="utf-8")!=rendered:
            raise ValueError("Raw999 implementation changed in an existing experiment")
        target.write_text(rendered,encoding="utf-8")
        return types.MethodType(function,self.model)

    def capture_loop_prediction(self,target,untarget,step,times):
        if step!=99:return
        self.last_loop_target=tree(target,lambda x:x.detach().clone())
        self.last_loop_untarget=tree(untarget,lambda x:x.detach().clone())
        self.last_loop=dict(step=step,network_times=[float(t[0]) for t in times],
            output_is_uncovered_prediction=True)

    def raw_output_checkpoint(self):
        if not hasattr(self,"last_loop_target"):return {}
        return dict(raw_output_prediction=dict(target=self.last_loop_target,untarget=self.last_loop_untarget,
            last_loop=self.last_loop))

    def restore_raw_output(self):
        if self.resume and self.resume.get("raw_output_prediction"):
            from .native_tools import device
            saved=self.resume["raw_output_prediction"]
            self.last_loop_target=device(saved["target"]);self.last_loop_untarget=device(saved["untarget"])
            self.last_loop=saved["last_loop"]

    def write_output_policy(self):
        if hasattr(self,"sampling_policy"):
            write(self.folder/"OUTPUT_POLICY.json",dict(self.sampling_policy,last_loop=getattr(self,"last_loop",None),
                chemical_condition_sha256=(getattr(self,"chemical_binding",None) or {}).get("chemical_condition_sha256")))


def increment_summary(folder):
    """Keep unclipped instability visible in both calibration and Agent history."""
    from .io import read
    rows=read(Path(folder)/"steps.json")
    if not rows or STEP_FIELDS[0] not in rows[0]:return {}
    output_policy=read(Path(folder)/"OUTPUT_POLICY.json")
    if output_policy["limiter"] == "none":
        return dict(guidance_increment_limiter="none",
            maximum_unconstrained_single_atom_guidance_increment_angstrom=max(
                max(r["unconstrained_max_atom_increment_A"]) for r in rows))
    flags=[r["guidance_increment_limited"] for r in rows]
    return dict(maximum_unconstrained_single_atom_guidance_increment_angstrom=max(
        max(r["unconstrained_max_atom_increment_A"]) for r in rows),
        guidance_increment_limited_candidate_steps=sum(sum(f) for f in flags),
        guidance_increment_limited_candidates=sum(any(f[i] for f in flags) for i in range(len(flags[0]))),
        minimum_guidance_increment_scale_factor=min(min(r["guidance_increment_scale_factor"]) for r in rows),
        guidance_increment_limit_angstrom=output_policy[LIMIT])
