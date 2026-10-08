"""Synthesize a compact regional reference and record this actual execution.

Example: --mining MINING --reference BRANCH_REF --source-receipt MULTI_RECEIPT
--branch-receipt BRANCH_RECEIPT --output SUMMARY --output-reference NEW_REF.gz
All destinations must be new; no model execution or terminal label access occurs.
"""
import argparse
import sys
import traceback
from pathlib import Path

from evomolsteer.continuous.regional_reference import prepare_inputs, synthesize
from evomolsteer.io import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--branch-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-reference", required=True)
    parser.add_argument("--receipt", help="Defaults to OUTPUT/execution_receipt.json")
    parser.add_argument("--q-threshold", type=float, default=.05)
    parser.add_argument("--minimum-batches", type=int, default=10)
    parser.add_argument("--minimum-loo-support", type=float, default=.70)
    parser.add_argument("--absolute-floor", type=float, default=1e-12)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    reference = Path(args.output_reference).resolve()
    receipt = Path(args.receipt).resolve() if args.receipt else out / "execution_receipt.json"
    if out.exists() or reference.exists() or receipt.exists():
        raise FileExistsError("Fresh summary, reference and receipt paths required")
    # Bind literal argv before performing synthesis, never reconstruct it later.
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    inputs = prepare_inputs(args.mining, args.reference, args.source_receipt, args.branch_receipt)
    loaded = [Path(module.__file__) for name, module in list(sys.modules.items())
              if name.startswith("evomolsteer") and getattr(module, "__file__", None)]
    codes = {str(path.resolve()): digest(path) for path in [Path(__file__), *loaded]}
    result, error = None, None
    try:
        result = synthesize(args.mining, args.reference, args.output, args.output_reference,
                            q_threshold=args.q_threshold, minimum_batches=args.minimum_batches,
                            minimum_loo_support=args.minimum_loo_support, absolute_floor=args.absolute_floor)
        returncode = 0
    except Exception:
        returncode = 1
        error = traceback.format_exc()
    unchanged = all(Path(path).is_file() and digest(path) == sha for path, sha in inputs.items()) and all(digest(path) == sha for path, sha in codes.items())
    generated = sorted(path for path in out.iterdir() if path.is_file()) if out.exists() else []
    if reference.is_file():
        generated.append(reference)
    outputs = {str(path.resolve()): digest(path) for path in generated}
    write_json(receipt, {
        "schema_version": "regional-reference-live-execution-receipt-1.0", "tool_id": "regional_reference",
        "command": command, "returncode": returncode, "error": error,
        "input_files": inputs, "code_files": codes, "output_files": outputs,
        "inputs_and_code_unchanged": unchanged, "complete_outputs": result is not None,
        "attestation": "Captured by the actual tool process before/after execution; not historical reconstruction.",
    })
    if returncode or not unchanged:
        raise RuntimeError(error or "Bound upstream input/source changed during execution")
    print("regional reference complete; receipt=" + str(receipt), flush=True)


if __name__ == "__main__":
    main()
