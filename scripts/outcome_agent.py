import argparse
from evomolsteer.continuous.outcome_agents import run_tool, export_request, validate_response, compile_program, call_api

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--action', required=True, choices=['tool', 'export', 'validate', 'compile', 'api'])
    for key in ['plan', 'receipt', 'evidence', 'registry', 'output', 'analyst', 'request', 'response']:
        p.add_argument('--'+key)
    p.add_argument('--role', choices=['Analyst', 'Designer'])
    p.add_argument('--round', type=int)
    a = p.parse_args()
    if a.action == 'tool':
        run_tool(a.plan, a.receipt)
    elif a.action == 'export':
        print(export_request(a.role, a.evidence, a.registry, a.receipt, a.output, a.analyst))
    elif a.action == 'validate':
        validate_response(a.request, a.response)
    elif a.action == 'compile':
        compile_program(a.request, a.response, a.output, a.round)
    else:
        call_api(a.request, a.output)
