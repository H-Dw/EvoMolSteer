"""Export/import isolated Agent calls, compile rewards, or call a real API."""
import argparse
import json
import os
from pathlib import Path
import httpx
from evomolsteer.io import read_json, write_json
from evomolsteer.continuous.dependency_ablation import prepare, export, validate, compile_program

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'export', 'validate', 'compile', 'api'])
    p.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--input'); p.add_argument('--folder'); p.add_argument('--role', choices=['Analyst', 'Designer'])
    p.add_argument('--config-output', default='configs/experiments/steer_dependency_v1')
    p.add_argument('--report-output', default='docs/experiments/steer_dependency_20261009')
    p.add_argument('--requested-model', help='Record the explicitly requested Agent model outside Skills')
    p.add_argument('--compiled-output', help='Separate output directory for a new Agent-model cohort')
    a = p.parse_args(); repo = Path(a.repo)
    if a.action == 'prepare':
        prepare(repo, a.input, repo / a.config_output, repo / a.report_output)
    elif a.action == 'export': export(a.folder, a.role, a.requested_model)
    elif a.action == 'validate': validate(a.folder, a.role)
    elif a.action == 'compile': compile_program(repo, a.folder, a.compiled_output)
    else:
        request = read_json(Path(a.folder) / (a.role + '.request.json'))
        base, key, model = [os.environ.get(k) for k in ['EVOMOLSTEER_BASE_URL', 'EVOMOLSTEER_API_KEY', 'EVOMOLSTEER_MODEL']]
        if not all([base, key, model]): raise ValueError('Explicit API endpoint, key and model required')
        if request.get('requested_model') and request['requested_model'] != model:
            raise ValueError('API model conflicts with the explicitly requested Agent model')
        with httpx.Client(timeout=180) as client:
            result = client.post(base.rstrip('/') + '/chat/completions', headers={'Authorization': 'Bearer ' + key},
                json={'model': model, 'temperature': 0, 'messages': request['messages'], 'response_format': {'type': 'json_object'}})
            result.raise_for_status()
        write_json(Path(a.folder) / (a.role + '.response.json'), json.loads(result.json()['choices'][0]['message']['content']))
        validate(a.folder, a.role)
