"""Synthesize one-lag regional reference with an actual-process source receipt."""
import argparse
import sys
import traceback
from pathlib import Path

from evomolsteer.continuous.regional_common_depth import prepare_inputs, synthesize_common_depth
from evomolsteer.io import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('mining', 'reference', 'source-receipt', 'branch-receipt', 'output', 'output-reference'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--receipt', help='Defaults to OUTPUT/execution_receipt.json')
    parser.add_argument('--depths', default='2,3,5,8,13')
    parser.add_argument('--q-threshold', type=float, default=.05)
    parser.add_argument('--minimum-batches', type=int, default=14)
    parser.add_argument('--minimum-loo-support', type=float, default=.70)
    parser.add_argument('--absolute-floor', type=float, default=1e-12)
    args = parser.parse_args()
    out, reference = Path(args.output).resolve(), Path(args.output_reference).resolve()
    receipt = Path(args.receipt).resolve() if args.receipt else out / 'execution_receipt.json'
    if out.exists() or reference.exists() or receipt.exists():
        raise FileExistsError('Fresh common-depth summary/reference/receipt paths required')
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    inputs = prepare_inputs(args.mining, args.reference, args.source_receipt, args.branch_receipt)
    loaded = [Path(module.__file__) for name, module in list(sys.modules.items())
              if name.startswith('evomolsteer') and getattr(module, '__file__', None)]
    codes = {str(path.resolve()): digest(path) for path in [Path(__file__), *loaded]}
    result, error = None, None
    try:
        result = synthesize_common_depth(args.mining, args.reference, args.output, args.output_reference,
                                        depths=tuple(int(v) for v in args.depths.split(',')),
                                        q_threshold=args.q_threshold, minimum_batches=args.minimum_batches,
                                        minimum_loo_support=args.minimum_loo_support, absolute_floor=args.absolute_floor)
        returncode = 0
    except Exception:
        returncode, error = 1, traceback.format_exc()
    unchanged = (all(Path(path).is_file() and digest(path) == sha for path, sha in inputs.items())
                 and all(digest(path) == sha for path, sha in codes.items()))
    generated = sorted(path for path in out.iterdir() if path.is_file()) if out.exists() else []
    if reference.is_file(): generated.append(reference)
    outputs = {str(path.resolve()): digest(path) for path in generated}
    write_json(receipt, {
        'schema_version': 'regional-common-depth-live-execution-receipt-1.0', 'tool_id': 'regional_common_depth',
        'command': command, 'returncode': returncode, 'error': error,
        'input_files': inputs, 'code_files': codes, 'output_files': outputs,
        'inputs_and_code_unchanged': unchanged, 'complete_outputs': result is not None,
        'attestation': 'Captured by the actual tool process before/after execution; not historical reconstruction.',
    })
    if returncode or not unchanged:
        raise RuntimeError(error or 'Bound upstream input/source changed during execution')
    print('common-depth regional reference complete; receipt=' + str(receipt), flush=True)


if __name__ == '__main__': main()
