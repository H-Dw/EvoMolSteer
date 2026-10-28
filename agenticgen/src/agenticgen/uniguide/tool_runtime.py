"""Versioned Agent configuration and instruction binding, separate from sampling."""
from pathlib import Path
from .io import frozen, read, sha, value_sha

RELEASE = "2.2.0"
DEFAULTS = dict(mode="readonly_tools_v1", skill_release=RELEASE,
    max_model_calls_per_round=30, max_output_tokens_per_round=30*16384)


def configure(config):
    result=dict(config)
    old=Path(config["root"])/"GEOMETRY_CONFIG.json" if "root" in config else None
    if "agent_tools" not in config and old is not None and old.exists() and "agent_tools" not in read(old):
        return result  # No automatic migration of a frozen historical experiment.
    settings=dict(DEFAULTS,**config.get("agent_tools",{}))
    if set(settings)!=set(DEFAULTS) or settings["mode"]!=DEFAULTS["mode"] or settings["skill_release"] not in ("2.1.0",RELEASE):
        raise ValueError("Unsupported Agent tool configuration/release")
    if type(settings["max_model_calls_per_round"]) is not int or not 4<=settings["max_model_calls_per_round"]<=1000:
        raise ValueError("At least four model calls required for the four-role round")
    if type(settings["max_output_tokens_per_round"]) is not int or settings["max_output_tokens_per_round"]<4*512:
        raise ValueError("Round output token allowance cannot cover the four roles")
    result["agent_tools"]=settings
    return result


def assets(config):
    folder=Path(__file__).resolve().parents[1]/"resources/skill_releases/guidance"/config["agent_tools"]["skill_release"]
    manifest=read(folder/"RELEASE.json")
    for relative,digest in manifest["files_sha256"].items():
        if sha(folder/relative)!=digest:raise ValueError("Skill release file differs from manifest")
    return {name+".md":folder/"runtime/internal_distance"/(name+".md")
        for name in ("analysis","proposal","synthesis","computation")}


def bind_experiment(config):
    files=assets(config)
    from agenticgen.providers import deepseek_tools
    body=dict(settings=config["agent_tools"],skill_hashes={n:sha(p) for n,p in files.items()},
        implementation_sha256={"provider":sha(deepseek_tools.__file__),
            "registry":sha(Path(__file__).with_name("analysis_tools.py")),
            "council":sha(Path(__file__).with_name("geometry_council.py")),
            "runtime":sha(__file__)},scope="all targets and versions of this experiment; no in-experiment skill updates")
    return frozen(Path(config["root"])/"GUIDANCE_AGENT_BINDING.json",body)
