"""Opt-in real provider acceptance using synthetic interface evidence; no sampler."""
import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tests/unit"))
from guidance_tool_fixture import build
from agenticgen.uniguide.geometry_council import advise
from agenticgen.uniguide.io import frozen, read, sha, write
from agenticgen.uniguide.tool_runtime import configure, assets, bind_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--live", action="store_true", help="Explicitly allow paid provider requests")
    args = parser.parse_args()
    if not args.live or not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("--live and process DEEPSEEK_API_KEY required; credential is never persisted")
    if not 1 <= args.rounds <= 3:
        parser.error("Acceptance is bounded to one through three rounds")
    if args.output.exists():
        parser.error("Use a new acceptance directory; failed sessions are never automatically retried")
    root = args.output.resolve()
    root.mkdir(parents=True)
    now = lambda: datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    config = configure(dict(root=str(root), llm=dict(model="deepseek-flash", thinking=True,
        reasoning_effort="high", max_output_tokens=16384, attempts=3, timeout=240,
        max_format_repairs=1)))
    frozen(root / "ACCEPTANCE_CONFIG.json", dict(config=config, synthetic_interface_only=True,
        no_molecular_sampling=True, started_at=now(), rounds=args.rounds))
    history, domain = build(root)
    binding=bind_experiment(config)
    source_files=dict(provider=PACKAGE/"src/agenticgen/providers/deepseek_tools.py",
        registry=PACKAGE/"src/agenticgen/uniguide/analysis_tools.py",
        council=PACKAGE/"src/agenticgen/uniguide/geometry_council.py",
        runtime=PACKAGE/"src/agenticgen/uniguide/tool_runtime.py")
    snapshot=root/"source_snapshot";snapshot.mkdir()
    for name,path in source_files.items():
        if sha(path)!=binding["implementation_sha256"][name]:raise ValueError("Implementation changed before acceptance")
        (snapshot/path.name).write_bytes(path.read_bytes())
    skills = root / "skills"
    skills.mkdir()
    for name, path in assets(config).items():
        (skills / name).write_bytes(path.read_bytes())
    protected = {p.relative_to(root).as_posix():sha(p) for directory in (root/"fixture", skills)
        for p in directory.rglob("*") if p.is_file()}
    summaries = []
    try:
        for index in range(1, args.rounds+1):
            folder = root / f"round_{index:02d}"
            print(f"{now()} real four-role acceptance round {index} started", flush=True)
            selected = advise(config, history, domain, folder, skills)
            before = len(list((folder/"model_budget").glob("call_*.json")))
            if advise(config, history, domain, folder, skills) != selected:
                raise ValueError("Cached role result changed")
            after = len(list((folder/"model_budget").glob("call_*.json")))
            if before != after:
                raise ValueError("Cached replay made another provider request")
            wires = [read(p) for p in folder.glob("*.session/step_*/RESPONSE.json")]
            tool_results = [read(p) for p in folder.glob("*.session/step_*/TOOL_*.json")]
            role_receipts = [read(folder/(role+".receipt.json"))
                for role in ("increase", "decrease", "redistribute", "synthesis")]
            counts = Counter(row["call"]["function"]["name"] for row in tool_results)
            summaries.append(dict(round=index, http_attempts=before, completed_responses=len(wires),
                tool_calls=dict(counts), tool_errors=sum(r["result"]["status"]=="tool_error" for r in tool_results),
                interface_repairs=sum(r["interface_repairs"] for r in role_receipts),
                completion_tokens=sum(w["usage"]["completion_tokens"] for w in wires),
                finish_reasons=dict(Counter(w["choices"][0]["finish_reason"] for w in wires)),
                selected_proposals=selected, original_action_validator_passed=True,
                cache_replay_added_http=after-before, completed_at=now()))
            write(root/"PROGRESS.json",summaries)
            print(f"{now()} round {index} passed: HTTP={before}, tools={dict(counts)}",flush=True)
        if not any(r["tool_calls"] for r in summaries):
            raise ValueError("No native tool call observed; cannot certify tool continuation")
        unchanged = all(sha(root/name)==digest for name,digest in protected.items())
        if not unchanged or (root/"attempts.json").exists():
            raise ValueError("Source mutation or molecular attempt ledger found")
        write(root/"COMPLETE.json",dict(synthetic_interface_only=True, completed_at=now(),
            rounds=summaries, preserved_source_files=len(protected), source_hashes=protected,
            sources_unchanged=unchanged,new_molecular_attempts=0,max_model_calls_per_round=30))
        print(f"{now()} acceptance complete; no molecular sampling",flush=True)
    except Exception as error:
        write(root/"FAILED.json",dict(failed_at=now(),error_type=type(error).__name__,
            completed_rounds=summaries,automatic_retry=False))
        print(f"{now()} acceptance stopped: {type(error).__name__}; preserve all receipts",flush=True)
        raise


if __name__ == "__main__":
    main()
