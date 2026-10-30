"""Audit the two CPU-only recovery guards added after the long GPU run froze.

This does not transfer an acceptance receipt or bypass the runtime source binding.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

GUARDS = '''
if _recovery is not None and any(u["id"] not in _recovery["units"] for u in units):
    raise ValueError("OOM continuation candidate identity changed; no new reservation allowed")
if _recovery is not None:
    capacity = min(capacity, _recovery["next_capacity"])
    write(capacity_file, dict(capacity=capacity, device_binding=device_binding,
        reason="bound_by_clean_process_oom_recovery"))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(tree):
    return ast.dump(tree, include_attributes=False)


def run(baseline, latest, destination):
    relative = Path('agenticgen/src/agenticgen/uniguide')
    left, right = Path(baseline) / relative, Path(latest) / relative
    first = {p.name: sha(p) for p in left.glob('*.py')}
    second = {p.name: sha(p) for p in right.glob('*.py')}
    changed = sorted(name for name in first.keys() | second.keys() if first.get(name) != second.get(name))
    if changed != ['geometry_batch.py']:
        raise ValueError('Unexpected runtime source differences: ' + str(changed))
    before = ast.parse((left / 'geometry_batch.py').read_text())
    after = ast.parse((right / 'geometry_batch.py').read_text())
    allowed = {normalized(node) for node in ast.parse(GUARDS).body}
    execute = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == 'execute_many')
    removed = [node for node in execute.body if normalized(node) in allowed]
    if len(removed) != 2:
        raise ValueError('Recovery guards differ from the declared CPU-only change')
    execute.body = [node for node in execute.body if normalized(node) not in allowed]
    if normalized(before) != normalized(after):
        raise ValueError('GPU/candidate sampling implementation also changed')
    if sha(left / 'workspace_counter.cpp') != sha(right / 'workspace_counter.cpp'):
        raise ValueError('Native counter adapter changed')
    result = dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        baseline=str(baseline), latest=str(latest), changed_modules=changed,
        unchanged_modules={name: first[name] for name in first if name not in changed},
        geometry_batch_before_sha256=first['geometry_batch.py'],
        geometry_batch_after_sha256=second['geometry_batch.py'],
        removed_cpu_recovery_guards=2, remaining_module_ast_equal=True,
        fixed_stream_executor_unchanged=True, model_and_vjp_unchanged=True,
        default_non_oom_execution_unchanged=True,
        long_gpu_acceptance_receipt_not_transferred=True,
        fresh_runtime_acceptance_required_before_future_formal_run=True)
    Path(destination).write_text(json.dumps(result, ensure_ascii=True, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline')
    parser.add_argument('latest')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(run(args.baseline, args.latest, args.destination), ensure_ascii=True))
