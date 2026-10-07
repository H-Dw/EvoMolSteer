"""Validate literal designs and certify conservative execution equivalence."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.continuous.skill_ablation import validate_response,compile_response,effective_signature


def assemble(root,study,complete=False):
    root=Path(root).resolve();study=Path(study).resolve();rows=[];missing=[]
    design=read_json(study/'design.json')
    incumbent=read_json(root/'configs/experiments/skill_ablation_v1/incumbent.json')
    base_signature,_=effective_signature(incumbent)
    for variant in design['variants']:
        count=design['replicates_per_factorial_condition'] if variant['id'] in ('legacy_legacy','compact_legacy','legacy_compact','compact_compact') else design['replicates_per_advice_removal']
        for rep in range(count):
            folder=study/'agents'/variant['id']/f'replicate_{rep}'
            request=folder/'Designer.request.json';response=folder/'Designer.response.json'
            if not response.exists():missing.append(f'{variant["id"]}/{rep}');continue
            data=validate_response(request,response,study/'evidence.json')
            program=compile_response(root,request,response,study/'evidence.json',folder/'reward_program.json')
            a=read_json(request)['bindings'];row={'variant':variant['id'],'replicate':rep,
                'analyst_response_sha256':a['analyst_response_sha256'],'request_sha256':digest(request),
                'response_sha256':digest(response),'decision':data['decision'],
                'base_program_id':data['base_program_id'],'updates':data['updates']}
            if program is not None:
                signature,effective=effective_signature(program)
                row.update(effective_signature=signature,execution_equivalent_to_incumbent=signature==base_signature,
                    compiled_path=(folder/'reward_program.json').relative_to(root).as_posix())
                write_json(folder/'equivalence.json',{'signature':signature,'incumbent_signature':base_signature,
                    'equal_execution_fields':signature==base_signature,'effective_fields':effective,
                    'interpretation':'Same registered reward/configuration after excluding enumerated non-execution metadata; reuse inference only with identical checkpoint, input batches, RNG, integrator and runtime source. Not separate molecular replicates.'})
            rows.append(row)
    if complete and missing:raise ValueError('Missing literal responses: '+', '.join(missing))
    result={'responses':rows,'missing':missing,'unique_executable_signatures':sorted({r['effective_signature'] for r in rows if 'effective_signature' in r}),
        'incumbent_signature':base_signature,'all_executable_equal_incumbent':bool(rows) and all(r.get('execution_equivalent_to_incumbent',False) for r in rows),
        'scope':'Fixed discovery evidence and this subagent simulator; no general claim about LLMs or novel targets.'}
    write_json(study/'literal_designs.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--study',required=True);p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();r=assemble(a.repo,a.study,a.require_complete)
    print({'responses':len(r['responses']),'missing':len(r['missing']),'unique_programs':len(r['unique_executable_signatures'])})
