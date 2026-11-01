"""Export only saved scientific coordinates and allowlisted prompt fields on CPU."""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import io
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    request = json.loads(Path(args.request).read_text())
    allowed = [Path(p).resolve() for p in request['allowed_roots']]

    def checked(path, expected=None):
        p = Path(path).resolve()
        assert any(p.is_relative_to(root) for root in allowed), str(p)
        assert 'launcher.log' not in p.name and not p.name.startswith('core')
        if expected:
            assert sha(p) == expected, str(p)
        return p

    result = dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  request_sha256=sha(Path(args.request)), collector_sha256=sha(Path(__file__)),
                  groups=[], prompts=[], new_model_forwards=0, new_sampling=0, new_llm_calls=0)
    conditions = {}
    for item in request['groups']:
        folder = checked(item['folder'])
        record_path = checked(folder/'records.json', item['record_sha256'])
        terminal_path = checked(folder/'terminal.pt.gz', item['terminal_sha256'])
        complete_path = checked(folder/'COMPLETE.json')
        rows = json.loads(record_path.read_text())
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, sha(p))
                  for p in [record_path, terminal_path, complete_path]}
        terminal = torch.load(io.BytesIO(gzip.decompress(terminal_path.read_bytes())),
                              map_location='cpu', weights_only=False)
        assert len(rows) == len(terminal['coords']) == item['population']
        root = checked(item['job_root'])
        condition_path = checked(root/'conditions'/f"{item['teacher']}.json")
        condition = json.loads(condition_path.read_text())
        conditions[str(condition_path)] = dict(sha256=sha(condition_path), value=condition)
        result['groups'].append(dict(**item, records=rows,
                                     coordinates_angstrom=terminal['coords'].double().numpy().tolist(),
                                     valid_mask=terminal['mask'].bool().numpy().tolist(),
                                     condition_path=str(condition_path), complete_sha256=sha(complete_path)))
        for p in [record_path, terminal_path, complete_path]:
            assert before[str(p)] == (p.stat().st_size, p.stat().st_mtime_ns, sha(p)), str(p)
    result['conditions'] = conditions
    for item in request['prompts']:
        path = checked(item['path'], item['sha256'])
        body = json.loads(path.read_text())
        if 'request' in body and 'messages' not in body:
            body = body['request']
        assert isinstance(body.get('messages'), list), str(path)
        # Explicitly omit headers, credentials, provider hidden reasoning and arbitrary metadata.
        messages = [{k:m[k] for k in ['role','content','name','tool_calls','tool_call_id'] if k in m}
                    for m in body['messages']]
        result['prompts'].append(dict(**item, messages=messages,
                                      model=body.get('model'), max_tokens=body.get('max_tokens'),
                                      tools=body.get('tools')))
    blob = json.dumps(result, ensure_ascii=True, allow_nan=False).encode()
    out = Path(args.out)
    assert not out.exists(), 'Preserve prior analysis input export'
    out.write_bytes(gzip.compress(blob, mtime=0))
    print(json.dumps(dict(timestamp=result['timestamp'], groups=len(result['groups']),
                          conditions=len(conditions), prompt_requests=len(result['prompts']),
                          output=str(out), bytes=out.stat().st_size, sha256=sha(out),
                          new_sampling=0, new_model_forwards=0, new_llm_calls=0), ensure_ascii=True))


if __name__ == '__main__':
    main()
