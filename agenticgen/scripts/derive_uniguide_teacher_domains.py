"""Offline teacher-domain audit from exported calibration JSON. No API or sampling.

The output is a proposal, never calibration/domain.json in an existing run.
Input keys: teachers.json and calibration/{coarse,fine,windows,full}_results.json.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agenticgen.uniguide.scale_domain import derive_domains, PHASE_COUNTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if output.name == "domain.json" or output == source:
        raise ValueError("Write a separate proposal, never an active domain or source artifact")
    raw = source.read_bytes()
    data = json.loads(gzip.decompress(raw) if source.suffix == ".gz" else raw)
    rows = [r for phase in PHASE_COUNTS for r in data[f"calibration/{phase}_results.json"]]
    result = dict(time_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        status="offline_proposal_not_deployed", api_calls=0, new_candidates=0,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_json_sha256=hashlib.sha256(gzip.decompress(raw) if source.suffix == ".gz" else raw).hexdigest(),
        domain=derive_domains(rows, data["teachers.json"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({target: {k: value[k] for k in ("initial", "lower", "upper", "risk_boundary")}
        for target, value in result["domain"]["by_teacher"].items()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
