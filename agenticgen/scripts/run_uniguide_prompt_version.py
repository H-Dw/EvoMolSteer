"""Run a new UG05--UG20 version with the audited v2 prompt contract throughout.

The scientific package and its numerical acceptance remain byte-identical.
The version config hash additionally binds this launcher, both policy modules,
and the accepted skills. Old UG04 recovery adapters are not used here.
"""
import argparse
import inspect
import json
from pathlib import Path

import uniguide_anchor_policy as policy


def runtime_binding(skills):
    return dict(policy=policy.POLICY, launcher_sha256=policy.file_sha(__file__),
        anchor_policy_sha256=policy.file_sha(policy.__file__),
        compact_policy_sha256=policy.file_sha(policy.legacy.__file__),
        skills={p.name: policy.file_sha(p) for p in sorted(Path(skills).glob('*.md'))})


def frozen_json(path, value):
    path = Path(path)
    text = policy.canonical(value)
    if path.exists():
        if policy.canonical(policy.read_json(path)) != text:
            raise ValueError('Frozen prompt evidence changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')


def wrap_provider(provider, root, version, skills, binding):
    signature = inspect.signature(provider)
    version_root = (Path(root) / 'versions' / version).resolve()
    def call(messages, schema, output, **kwargs):
        if runtime_binding(skills) != binding:
            raise ValueError('Prompt runtime assets changed mid-version')
        output = Path(output).resolve()
        if not output.is_relative_to(version_root):
            raise ValueError('Provider output outside bound version')
        opts = signature.bind(messages, schema, output, **kwargs)
        opts.apply_defaults()
        values = opts.arguments
        expected = dict(model='deepseek-flash', thinking=True, reasoning_effort='high',
            max_output_tokens=16384, attempts=3, max_format_repairs=1,
            input_store=None, response_parser=None, format_builder=None, format_policy='json-interface-v5')
        if any(values[k] != v for k, v in expected.items()):
            raise ValueError('Unexpected provider protocol')
        # Stable object-key traversal gives stable reference labels on cache replay.
        # Array ordering, numbers, types and every record remain unchanged.
        stable = [dict(m) for m in messages]
        context = json.loads(messages[1]['content'])
        decisions = []
        for row in context['history']:
            if row['id'].startswith('versions/' + version + '/'):
                group = (Path(root) / row['id']).resolve()
                if not group.is_relative_to(version_root) or not (group / 'COMPLETE.json').exists():
                    raise ValueError('Action evidence lacks a completed in-version source')
                proposal = group / 'proposal.json'
                decisions.append(dict(source_group=row['id'], proposal=policy.read_json(proposal),
                    proposal_sha256=policy.file_sha(proposal)))
        context['completed_action_evidence'] = decisions
        stable[1]['content'] = policy.canonical(context)
        rendered, contracted, metrics = policy.transform(stable, schema, skills)
        original = dict(messages=stable, response_schema=schema)
        frozen_json(output.with_suffix('.prompt_source.json'), original)
        frozen_json(output.with_suffix('.prompt_binding.json'), dict(runtime=binding, metrics=metrics))
        claim = output.with_suffix('.execution_claim.json')
        if not output.exists():
            # No implicit second launch after an exhausted/invalid provider call.
            with claim.open('x', encoding='utf-8') as stream:
                json.dump(dict(version=version, output=str(output), policy=policy.POLICY), stream)
        elif not claim.exists():
            raise ValueError('Cached output lacks an execution claim')
        return provider(rendered, contracted, output, **dict(kwargs,
            response_parser=policy.parse_output, format_builder=policy.format_contract,
            format_policy=policy.POLICY))
    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('version')
    args = parser.parse_args()
    if args.version not in [f'UG{i:02d}' for i in range(5, 21)]:
        raise ValueError('This launcher is only for new UG05--UG20 versions')
    import fcntl
    import sys
    config = policy.read_json(args.config)
    root = Path(config['root'])
    with (root / '.executor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, config['plip_dependencies'])
        from agenticgen.uniguide import council, pipeline
        skills = pipeline.skill_location()
        binding = runtime_binding(skills)
        config = dict(config, guidance_prompt_runtime=binding)
        folder = root / 'versions' / args.version
        # Reject an old/raw manifest before loading any model or making any call.
        from agenticgen.uniguide.io import value_sha
        if (folder / 'manifest.json').exists():
            if policy.read_json(folder / 'manifest.json')['config_sha256'] != value_sha(config):
                raise ValueError('Version was frozen under another prompt runtime')
        frozen_json(folder / 'PROMPT_POLICY.json', binding)
        council.call_json = wrap_provider(council.call_json, root, args.version, skills, binding)
        result = pipeline.run_version(config, args.version)
        print(json.dumps(dict(version=args.version, awaiting_gpt_review=True,
            overall=result['current']['overall'], next_version=result['next_version'])))


if __name__ == '__main__':
    main()
