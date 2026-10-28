"""Offline evidence for a design proposal; not an executable Agent tool registry."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'agenticgen/src'))
import httpx
from agenticgen.providers.deepseek_api import call_json


def main():
    results = []
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}
    messages = [{'role': 'user', 'content': 'Offline mock request, not an experiment.'}]
    with tempfile.TemporaryDirectory(prefix='guidance_tools_probe_') as tmp:
        base = Path(tmp)
        try:
            call_json(messages, schema, base/'unsupported.json', tools=[])
        except TypeError as error:
            assert 'tools' in str(error)
            results.append({'case': 'tools_keyword', 'observed': type(error).__name__, 'message': str(error)})
        else:
            raise AssertionError('Expected current provider to lack a tools parameter')
        for index, content in enumerate(['', None]):
            seen = []
            def handle(request):
                seen.append(json.loads(request.content))
                return httpx.Response(200, json={
                    'id': 'offline-mock', 'model': 'deepseek-flash',
                    'choices': [{'finish_reason': 'tool_calls', 'message': {
                        'role': 'assistant', 'content': content,
                        'reasoning_content': 'mock transport metadata',
                        'tool_calls': [{'id': 'mock-call', 'type': 'function', 'function': {
                            'name': 'summarize_population', 'arguments': '{"run_id":"mock-run"}'}}]}}],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-placeholder'}):
                try:
                    call_json(messages, schema, base/f'tool-{index}.json',
                              transport=httpx.MockTransport(handle), attempts=1, max_format_repairs=0)
                except (ValueError, TypeError) as error:
                    results.append({'case': 'tool_response', 'content': content,
                                    'observed': type(error).__name__, 'message': str(error),
                                    'mock_http_calls': len(seen), 'registered_tools_sent': 'tools' in seen[0]})
                else:
                    raise AssertionError('Current JSON-only provider unexpectedly accepted tool_calls')
            assert len(seen) == 1
        try:
            call_json([{'role': 'assistant', 'content': None, 'tool_calls': []}], schema,
                      base/'cache.json', input_store=base/'input-cache')
        except AttributeError as error:
            results.append({'case': 'null_assistant_cache', 'observed': type(error).__name__, 'message': str(error)})
        else:
            raise AssertionError('Expected text-only cache to reject null content')

    def summary(values):
        ordered = sorted(values)
        cut = len(ordered)//10
        return {'n': len(values), 'arithmetic_mean': statistics.mean(values),
                'median': statistics.median(values),
                'symmetric_ten_percent_trimmed_mean': statistics.mean(ordered[cut:-cut]),
                'maximum': max(values), 'worst_five_mean': statistics.mean(ordered[-5:]),
                'number_above_simulated_threshold_0_8': sum(v > .8 for v in values)}
    synthetic = {'not_real_experiment': True, 'unit': 'angstrom',
                 'not_a_recommended_scientific_threshold': True,
                 'candidate': summary([.25]*49+[12.]), 'reference': summary([.35]*50)}
    sources = ['agenticgen/src/agenticgen/providers/deepseek_api.py',
               'agenticgen/src/agenticgen/uniguide/geometry_council.py',
               'agenticgen/src/agenticgen/uniguide/geometry_sampling.py',
               'agenticgen/src/agenticgen/uniguide/geometry_protocol.py']
    output = {'method': 'actual current provider under httpx.MockTransport; no external API',
              'repository_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              'provider_probes': results, 'synthetic_statistics': synthetic,
              'new_molecular_attempts': 0, 'real_model_calls': 0, 'registered_tools': False}
    destination = Path(__file__).with_name('probe.json')
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output, ensure_ascii=True))


if __name__ == '__main__':
    main()
