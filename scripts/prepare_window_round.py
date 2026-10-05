"""Freeze an LLM-proposed declarative revision and its auditable decision record.

This local tool does not choose a scientific hypothesis or launch remote jobs.
It records exact parameter changes and evidence, not private chain-of-thought.
"""
import argparse
import copy
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest


def prepare(base,revision,evidence,program_out,record_out,hypothesis,rationale,alternatives):
    original=read_json(base);patch=read_json(revision);program=copy.deepcopy(original)
    def update(target,changes):
        for key,value in changes.items():
            if isinstance(value,dict):update(target[key],value)
            else:target[key]=value
    update(program,patch)
    if not 1<=program['round']<=8:raise ValueError('This campaign authorizes at most eight rounds')
    if program['window']!=original['window'] or program['reference_sha256']!=original['reference_sha256']:
        raise ValueError('Do not change the target population/window during optimization')
    if Path(program_out).exists() or Path(record_out).exists():raise FileExistsError('Immutable round already exists')
    record={'round':program['round'],'parent_round':original['round'],'status':'prepared',
            'base_program':str(base),'base_program_sha256':digest(base),'exact_revision':patch,
            'evidence':[{'path':str(p),'sha256':digest(p),'results':read_json(p).get('results',{})} for p in evidence],
            'hypothesis':hypothesis,'decision_rationale':rationale,'alternatives':alternatives,
            'acceptance':'Frozen: primary mean improves >=2% and both paired adaptation batches improve over accepted incumbent; otherwise retain incumbent.',
            'scope':'Actual states within unchanged learned window; no new SMC, no physical-energy claim; all analysis local.',
            'interpretation':'Evidence, formula choices and parameter decisions; no private reasoning trace.'}
    write_json(program_out,program);record['program_sha256']=digest(program_out);write_json(record_out,record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--evidence',action='append',required=True);p.add_argument('--program-output',required=True)
    p.add_argument('--record-output',required=True);p.add_argument('--hypothesis',required=True)
    p.add_argument('--rationale',required=True);p.add_argument('--alternative',action='append',default=[])
    a=p.parse_args();prepare(a.base,a.revision,a.evidence,a.program_output,a.record_output,a.hypothesis,a.rationale,a.alternative)
