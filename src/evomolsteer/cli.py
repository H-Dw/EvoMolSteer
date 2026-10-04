"""Independent reusable stages; each stage records deterministic output files."""
import argparse
from pathlib import Path
from .io import read_json

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=['ingest','enrichment','differential','pca','trends','bundle','analyze','agent-export','agent-import','agent-call','reward-validate'])
    p.add_argument('--root');p.add_argument('--output',required=True);p.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/default.json'))
    p.add_argument('--split',default='discovery',choices=['discovery','validation','heldout']);p.add_argument('--role',choices=['Analyst','Designer']);p.add_argument('--response');p.add_argument('--request');p.add_argument('--program')
    a=p.parse_args(argv);out=Path(a.output)
    if a.command=='ingest':
        from .ingest import ingest
        result=ingest(a.root,out,read_json(a.config));print('Audited batches:',len(result))
    elif a.command in ['enrichment','differential','pca','trends','analyze']:
        import importlib
        names=['enrichment','differential','trends','pca'] if a.command=='analyze' else [a.command]
        for name in names:
            print('Running',name,flush=True);importlib.import_module('evomolsteer.'+name).run(out,a.split)
    elif a.command=='bundle':
        from .evidence import build_bundle
        build_bundle(out,a.split)
    elif a.command.startswith('agent-'):
        from .agents import export_request,import_response,call_api
        if a.command=='agent-export':export_request(out,a.role)
        elif a.command=='agent-import':import_response(Path(a.request),Path(a.response),out)
        else:call_api(Path(a.request),out)
    elif a.command=='reward-validate':
        from .reward import validate_on_fixture
        validate_on_fixture(out,Path(a.program))

if __name__=='__main__':main()
