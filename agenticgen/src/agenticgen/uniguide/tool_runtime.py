"""Versioned Agent configuration and instruction binding, separate from sampling."""
from pathlib import Path
from .io import frozen, read, sha, value_sha

RELEASE = "2.3.0"
DEFAULTS = dict(mode="readonly_tools_v1", skill_release=RELEASE,
    max_model_calls_per_round=30, max_output_tokens_per_round=30*16384)


def configure(config):
    result=dict(config)
    old=Path(config["root"])/"GEOMETRY_CONFIG.json" if "root" in config else None
    if "agent_tools" not in config and old is not None and old.exists() and "agent_tools" not in read(old):
        return result  # No automatic migration of a frozen historical experiment.
    from .absolute_coordinate import POLICY as coordinate_policy
    defaults = dict(DEFAULTS, skill_release="2.6.0") if config.get("protocol")==coordinate_policy else DEFAULTS
    settings=dict(defaults,**config.get("agent_tools",{}))
    if set(settings)!=set(DEFAULTS) or settings["mode"]!=DEFAULTS["mode"] or settings["skill_release"] not in ("2.1.0","2.2.0",RELEASE,"2.4.0","2.5.0","2.6.0"):
        raise ValueError("Unsupported Agent tool configuration/release")
    roles=2 if settings["skill_release"]=="2.6.0" else 4
    if type(settings["max_model_calls_per_round"]) is not int or not roles<=settings["max_model_calls_per_round"]<=1000:
        raise ValueError("Model call allowance cannot cover the configured roles")
    if type(settings["max_output_tokens_per_round"]) is not int or settings["max_output_tokens_per_round"]<roles*512:
        raise ValueError("Round output token allowance cannot cover the configured roles")
    result["agent_tools"]=settings
    if settings["skill_release"] == "2.6.0":
        if config.get("protocol") != coordinate_policy:
            raise ValueError("Linear skills require the absolute coordinate protocol")
        from .linear_scale import configure as configure_linear
        result=configure_linear(result)
    return result


def assets(config):
    folder=Path(__file__).resolve().parents[1]/"resources/skill_releases/guidance"/config["agent_tools"]["skill_release"]
    manifest=read(folder/"RELEASE.json")
    for relative,digest in manifest["files_sha256"].items():
        if sha(folder/relative)!=digest:raise ValueError("Skill release file differs from manifest")
    return {name+".md":folder/manifest.get("runtime_directory","runtime/internal_distance")/(name+".md")
        for name in ("analysis","proposal","synthesis","computation")}


def bind_experiment(config):
    files=assets(config)
    from agenticgen.providers import deepseek_tools
    from .linear_scale import enabled as linear_enabled
    council="linear_council.py" if linear_enabled(config) else "geometry_council.py"
    body=dict(settings=config["agent_tools"],skill_hashes={n:sha(p) for n,p in files.items()},
        implementation_sha256={"provider":sha(deepseek_tools.__file__),
            "registry":sha(Path(__file__).with_name("analysis_tools.py")),
            "council":sha(Path(__file__).with_name(council)),
            "runtime":sha(__file__)},scope="all targets and versions of this experiment; no in-experiment skill updates")
    if linear_enabled(config):
        body["implementation_sha256"].update({name:sha(Path(__file__).with_name(name+".py"))
            for name in ("linear_scale","linear_calibration","linear_runtime")})
    return frozen(Path(config["root"])/"GUIDANCE_AGENT_BINDING.json",body)
