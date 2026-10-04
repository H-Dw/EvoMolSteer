"""Check each term and the combined reward at every proposed stage midpoint."""
import argparse
import copy
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.reward import validate_on_fixture

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--program',required=True)
    a=p.parse_args();out=Path(a.output);program=read_json(a.program);cases=[]
    for i,term in enumerate(program['terms']):
        for mode in ['single_term','combined']:
            candidate=copy.deepcopy(program)
            candidate['terms']=[copy.deepcopy(term)] if mode=='single_term' else copy.deepcopy(program['terms'][i:]+program['terms'][:i])
            dest=out/'verification'/'reward_suite'/f'{i:02d}_{mode}'
            for name in ['config.json','feature_catalog.json','ingest_manifest.json']:write_json(dest/name,read_json(out/name))
            write_json(dest/'program.json',candidate)
            result=validate_on_fixture(dest,dest/'program.json')
            cases.append({'anchor_term':term['term_id'],'mode':mode,**result})
    report={'status':'passed' if cases else 'deferred_empty_program','n_cases':len(cases),'cases':cases,
        'original_program_sha256':digest(a.program),'runtime_sha256':digest(Path(__file__).resolve().parents[1]/'src/evomolsteer/reward.py'),
        'scope':'One saved particle per rule-stage midpoint; each term and combined reward. No live generator or efficacy validation.'}
    write_json(out/'agents/reward_suite.json',report)
    print('Numerical cases:',len(cases))
