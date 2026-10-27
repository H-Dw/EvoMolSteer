"""One audited same-request recovery for a frozen native UG05--UG20 runtime.

The original execution claim, request, failed answer and runtime stay intact.
A separate exclusive claim authorizes one provider invocation, with its original
network/format budgets. Successful calls replay through the frozen wrapper.
"""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(left, right):
    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return encode(left) == encode(right)


class TargetReached(Exception):
    pass


def prepare_target(runtime, provider, root, version, skills, binding, messages, schema, output, kwargs):
    """Reproduce and check every persisted input before bypassing a spent claim.

    This small reconstruction mirrors the frozen launcher's source preparation;
    the persisted source, metrics, request and wire must ALL match exactly. Any
    future launcher change therefore fails closed instead of silently adapting.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    version_root = root / 'versions' / version
    if not output.is_relative_to(version_root) or runtime.runtime_binding(skills) != binding:
        raise ValueError('Frozen runtime or target scope changed')
    opts = inspect.signature(provider).bind(messages, schema, output, **kwargs)
    opts.apply_defaults()
    expected = dict(model='deepseek-flash', thinking=True, reasoning_effort='high',
        max_output_tokens=16384, attempts=3, max_format_repairs=1, input_store=None,
        response_parser=None, format_builder=None, format_policy='json-interface-v5')
    if any(opts.arguments[k] != value for k, value in expected.items()):
        raise ValueError('Frozen provider protocol changed')
    policy = runtime.policy
    stable = [dict(m) for m in messages]
    context = json.loads(messages[1]['content'])
    decisions = []
    for row in context['history']:
        if row['id'].startswith('versions/' + version + '/'):
            group = (root / row['id']).resolve()
            if not group.is_relative_to(version_root) or not (group / 'COMPLETE.json').exists():
                raise ValueError('Action evidence lacks completed source')
            proposal = group / 'proposal.json'
            decisions.append(dict(source_group=row['id'], proposal=read(proposal), proposal_sha256=sha(proposal)))
    context['completed_action_evidence'] = decisions
    stable[1]['content'] = policy.canonical(context)
    rendered, contracted, metrics = policy.transform(stable, schema, skills)
    body = dict(model='deepseek-flash', messages=rendered, thinking={'type':'enabled'},
        reasoning_effort='high', max_tokens=16384, response_format={'type':'json_object'})
    request = dict(provider='DeepSeek', body=body, response_schema=contracted, format_policy=policy.POLICY)
    wire = dict(request, body=dict(body, messages=rendered + [dict(role='user', content=policy.format_contract(contracted))]))
    checks = {'.prompt_source.json':dict(messages=stable, response_schema=schema),
        '.prompt_binding.json':dict(runtime=binding, metrics=metrics),
        '.request.json':request, '.wire_1.request.json':wire,
        '.execution_claim.json':dict(version=version, output=str(output), policy=policy.POLICY)}
    for suffix, value in checks.items():
        if not same(read(output.with_suffix(suffix)), value):
            raise ValueError('Original input changed: ' + suffix)
    # Only the first length failure is eligible. Further recovery needs a new
    # explicit audit; neither this program nor repeated launches can loop.
    failures = list(output.parent.glob(output.stem + '.incomplete_*.receipt.json'))
    if len(failures) != 1 or list(output.parent.glob(output.stem + '.attempt_*.invalid.receipt.json')):
        raise ValueError('Expected exactly one initial length failure')
    failure = read(failures[0])
    if (failure['finish_reason'] != 'length' or failure['max_output_tokens'] != 16384
            or failure['request_sha256'] != sha(output.with_suffix('.request.json'))
            or failure['content_sha256'] != sha(failures[0].with_suffix('').with_suffix('.txt'))):
        raise ValueError('Failure receipt binding changed')
    if output.exists() or output.with_suffix('.receipt.json').exists():
        raise ValueError('Target already succeeded; never resend')
    return rendered, contracted


def recovery_call(runtime, provider, root, version, skills, binding, authorization, record, execute=False):
    auth = read(authorization)
    auth_sha = sha(authorization)
    target = (Path(root) / auth['output_relative_path']).resolve()
    native = runtime.wrap_provider(provider, root, version, skills, binding)
    def call(messages, schema, output, **kwargs):
        output = Path(output).resolve()
        if sha(authorization) != auth_sha:
            raise ValueError('Authorization changed')
        if output != target:
            relative = output.relative_to(Path(root).resolve()).as_posix()
            expected = auth['successful_outputs'].get(relative)
            if not expected or sha(output) != expected['output'] or sha(output.with_suffix('.receipt.json')) != expected['receipt']:
                raise ValueError('Unexpected or changed preceding cache')
            result = native(messages, schema, output, **kwargs)
            record(dict(event='validated_cache', output=relative))
            return result
        rendered, contracted = prepare_target(runtime, provider, root, version, skills, binding,
            messages, schema, output, kwargs)
        record(dict(event='exact_target_verified', output=auth['output_relative_path'], execute=execute))
        if execute:
            claim = output.with_suffix('.explicit_recovery_claim.json')
            with claim.open('x', encoding='utf-8') as stream:
                json.dump(dict(authorization_sha256=auth_sha, pid=os.getpid()), stream)
                stream.flush()
                os.fsync(stream.fileno())
            provider(rendered, contracted, output, **dict(kwargs,
                response_parser=runtime.policy.parse_output, format_builder=runtime.policy.format_contract,
                format_policy=runtime.policy.POLICY))
            record(dict(event='recovered', output=auth['output_relative_path'], receipt_sha256=sha(output.with_suffix('.receipt.json'))))
        raise TargetReached()
    return call


def replay(config, version, council, record):
    """Replay advice and completed rows only. No model, sampling or queue writes."""
    root = Path(config['root'])
    domain = read(root / 'calibration/domain.json')
    for teacher in read(root / 'teachers.json'):
        local_domain = domain
        if 'by_teacher' in domain:
            from agenticgen.uniguide.scale_domain import resolve_domain
            local_domain = resolve_domain(domain, teacher)
        troot = root / 'versions' / version / teacher['target']
        history = [read(root / 'R0' / teacher['target'] / 'aggregate.json')]
        saved = read(troot / 'history.json') if (troot / 'history.json').exists() else history
        for number in range(1, 6):
            if not same(history, saved[:len(history)]):
                raise ValueError('Reconstructed history differs from snapshot')
            proposals = council.advise(config, teacher, version, number, history, local_domain,
                root / 'versions' / version / 'skills', troot / f'advice_R{number:02d}')
            for proposal in proposals:
                folder = troot / f'R{number:02d}' / proposal['branch']
                if not (folder / 'COMPLETE.json').exists() or not same(read(folder / 'proposal.json'), proposal):
                    raise ValueError('Replay would require new or changed sampling')
                row = dict(read(folder / 'aggregate.json'), branch=proposal['branch'], round=number,
                    private_context=proposal['private_context'], synthesis_context=proposal['synthesis_context'],
                    quality=read(folder / 'QUALITY_LIGHT_COMPLETE.json'))
                history.append(row)
                record(dict(event='validated_group', group=str(folder.relative_to(root))))
        if not same(history, saved):
            raise ValueError('Completed teacher history mismatch')
        council.advise(config, teacher, version, 5, history, local_domain,
            root / 'versions' / version / 'skills', troot / 'analysis_R05', final=True)
    raise ValueError('Authorized target not reached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--frozen-repo', required=True)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    repo = Path(args.frozen_repo).resolve()
    auth = read(args.authorization)
    version = auth['version']
    if version not in [f'UG{i:02d}' for i in range(5,21)] or auth['scope'] != 'one_same_request_recovery' or not auth['basis']:
        raise ValueError('Invalid recovery authorization')
    own_repo = Path(__file__).resolve().parents[2]
    git = lambda path, *rest: subprocess.check_output(['git','-C',str(path),*rest])
    if (git(own_repo, 'rev-parse','HEAD').decode().strip() != auth['adapter_commit']
            or git(own_repo,'show','HEAD:agenticgen/scripts/resume_uniguide_prompt_call.py') != Path(__file__).read_bytes()):
        raise ValueError('Recovery script is not the authorized committed blob')
    sys.path.insert(0, str(repo / 'agenticgen/scripts'))
    import run_uniguide_prompt_version as runtime
    config = read(args.config)
    root = Path(config['root']).resolve()
    import fcntl
    with (root / '.executor.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, config['plip_dependencies'])
        from agenticgen.uniguide import council, pipeline
        from agenticgen.uniguide.io import value_sha
        folder = root / 'versions' / version
        manifest = read(folder / 'manifest.json')
        if (git(repo,'rev-parse','HEAD').decode().strip() != manifest['code_commit']
                or git(repo,'status','--porcelain','--untracked-files=no').strip()
                or Path(pipeline.__file__).resolve().parents[4] != repo
                or pipeline.source_binding() != manifest['source_binding']
                or sha(folder / 'manifest.json') != auth['manifest_sha256']):
            raise ValueError('Frozen scientific checkout changed')
        skills = pipeline.skill_location()
        binding = runtime.runtime_binding(skills)
        config = dict(config, guidance_prompt_runtime=binding)
        if not same(binding, read(folder / 'PROMPT_POLICY.json')) or value_sha(config) != manifest['config_sha256']:
            raise ValueError('Frozen runtime/config changed')
        preservation_path = root / auth['preservation_relative_path']
        if sha(preservation_path) != auth['preservation_sha256']:
            raise ValueError('Preservation ledger changed')
        preservation = read(preservation_path)
        def verify(include_history=True):
            for relative, expected in preservation['source_hashes'].items():
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or sha(path) != expected:
                    raise ValueError('Protected file changed: ' + relative)
            if include_history:
                for relative, value in preservation['history_snapshots'].items():
                    if sha(root / relative) != value['sha256']:
                        raise ValueError('History snapshot changed')
                if not same(read(root / 'attempts.json'), preservation['attempts']):
                    raise ValueError('Sampling budget changed during preflight')
        verify()
        stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        audit = root / 'engineering_audits' / f'{version}_native_recovery_{stamp}.jsonl'
        with audit.open('x', encoding='utf-8') as stream:
            def record(event):
                event['time_shanghai'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
                stream.write(json.dumps(event, ensure_ascii=False) + '\n');stream.flush();os.fsync(stream.fileno())
            record(dict(event='start', execute=args.execute, adapter_commit=auth['adapter_commit'],
                frozen_commit=manifest['code_commit'], authorization_sha256=sha(args.authorization), pid=os.getpid()))
            provider = council.call_json
            council.call_json = recovery_call(runtime, provider, root, version, skills, binding,
                args.authorization, record, args.execute)
            try:
                try:
                    replay(config, version, council, record)
                except TargetReached:
                    verify()
                    record(dict(event='preflight_complete', new_candidates=0, preceding_cache_count=len(auth['successful_outputs'])))
                if args.execute:
                    # The recovered output now passes the ordinary frozen wrapper.
                    # Any later failed claim still stops; there is no retry loop.
                    council.call_json = runtime.wrap_provider(provider, root, version, skills, binding)
                    pipeline.run_version(config, version)
                    record(dict(event='version_finalized', review_required=True))
            except BaseException as error:
                record(dict(event='stopped', error_type=type(error).__name__))
                raise
            finally:
                verify(include_history=not args.execute)
                record(dict(event='preserved_files_verified', count=len(preservation['source_hashes'])))
        print(json.dumps(dict(audit=str(audit), execute=args.execute)))


if __name__ == '__main__':
    main()
