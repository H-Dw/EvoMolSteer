"""Prepare a new fixed-code campaign and an explicit saved-sampling handoff."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from agenticgen.uniguide.absolute_recovery import RECOVERY_POLICY, STATIC_REFACTOR_SOURCES, preflight
from agenticgen.uniguide.absolute_runtime import prepare, validate_config
from agenticgen.uniguide.io import frozen, read, sha, value_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--failure-audit', type=Path, required=True)
    parser.add_argument('--equivalence-evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists() and any(args.root.iterdir()):
        raise ValueError('Restart requires a new result root; never clear old claims')
    original = read(args.source_root/'GEOMETRY_CONFIG.json')
    failure = read(args.failure_audit/'failure_audit.json')
    if failure['classification'] != 'engineering_postprocessing_failure_after_completed_raw_sampling':
        raise ValueError('Unexpected previous failure')
    batches = [b for b in failure['batches'] if b['terminal_saved'] and b['steps'] == 100]
    if len(batches) != 1 or not (args.source_root/'FINALIZED.json').exists():
        raise ValueError('Exactly one archived raw sampling batch is required')
    proof = read(args.equivalence_evidence)
    config = copy.deepcopy(original)
    config['root'] = str(args.root)
    config = validate_config(config)
    artifact_hashes = {name: row['sha256'] for name, row in read(args.failure_audit/'artifact_inventory.json').items()}
    for name in ('attempts.json', 'SUBJECTS.json', 'GEOMETRY_CONFIG.json', 'CAMPAIGN_FAILED.json', 'FINALIZED.json'):
        artifact_hashes[name] = sha(args.source_root/name)
    changes = {}
    for name in sorted(STATIC_REFACTOR_SOURCES):
        relative = 'agenticgen/src/agenticgen/uniguide/'+name+'.py'
        before = failure['runtime_commit']
        old_bytes = subprocess.check_output(['git', '-C', str(args.repo), 'show', before+':'+relative])
        checked_bytes = subprocess.check_output(['git', '-C', str(args.repo), 'show', proof['code_commit']+':'+relative])
        if hashlib.sha256(checked_bytes).hexdigest() != sha(args.repo/relative):
            raise ValueError('Sampling file differs from the checked static refactor: '+name)
        changes[name] = dict(previous=hashlib.sha256(old_bytes).hexdigest(), current=sha(args.repo/relative))
    authorization = dict(policy=RECOVERY_POLICY,
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        user_authorization='完成代码优化后，请提交至代码仓库，并在远端服务器重启两个教师分子的全量线上运行。',
        source_root=str(args.source_root), source_commit=failure['runtime_commit'],
        target_root=str(args.root), target_config_sha256=value_sha(config), source_batch=batches[0]['path'],
        source_artifact_hashes=artifact_hashes, source_attempts_sha256=sha(args.source_root/'attempts.json'),
        source_changes=changes, equivalence_evidence=str(args.equivalence_evidence),
        equivalence_evidence_sha256=sha(args.equivalence_evidence),
        expected_reserved_attempts=failure['calibration_reserved_attempts'],
        expected_saved_candidates=batches[0]['candidate_count'],
        aliases_are_not_additional_attempts=True, only_saved_postprocessing=True)
    authorization_path = args.config.with_name(args.config.stem+'-recovery.json')
    frozen(authorization_path, authorization)
    config['sampling_recovery'] = dict(authorization_path=str(authorization_path),
                                     authorization_sha256=sha(authorization_path))
    frozen(args.config, config)
    subjects = prepare(config)
    if subjects != read(args.source_root/'SUBJECTS.json'):
        raise ValueError('Restart must preserve both original teachers and molecular conditions')
    protected = dict(read(args.source_root/'SOURCE_PRESERVATION.json'))
    for path in args.source_root.rglob('*'):
        if (path.is_file() and 'monitoring' not in path.relative_to(args.source_root).parts
                and path.name not in ('.executor.lock', '.campaign.lock', '.worker.lock')):
            protected[str(path)] = sha(path)
    baseline = Path(failure['repo'])
    protected.update({str(baseline/name): digest for name, digest in read(args.failure_audit/'source_hashes.json').items()})
    frozen(args.root/'SOURCE_PRESERVATION.json', protected)
    frozen(args.root/'SAMPLING_RECOVERY_AUTHORIZATION.json', authorization)
    result = dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        repo=str(args.repo), commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
        root=str(args.root), config=str(args.config), config_sha256=sha(args.config),
        authorization=str(authorization_path), authorization_sha256=sha(authorization_path),
        source_preservation_count=len(protected), recovery_preflight=preflight(config),
        teacher_conditions={s['teacher']['target']:s['condition']['condition_sha256'] for s in subjects},
        no_sampler_or_llm_called=True, original_root_unchanged=True)
    frozen(args.root/'RESTART_PREPARED_AUDIT.json', result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
