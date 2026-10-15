"""Enumerate legal untested actions, leaving scientific selection to the LLM."""
import hashlib
from .semantic_contract import variants,identity
from .reward import OPERATORS


def novel_actions(parents,regions,templates,master,constraint=None):
    constraint=constraint or {};seen=set(master.get('_seen_numeric_programs',[]));actions=[]
    for pid,entry in sorted(parents.items()):
        parent=entry['program'];parent_teacher=templates[parent['template_id']];parent_sig=identity(parent,parent_teacher)
        for factor in entry['legal_factors']:
            if constraint.get('factors') and factor not in constraint['factors']:continue
            targets=list(regions.values()) if factor=='region' else [regions[parent['region_id']]] if parent['region_id'] in regions else []
            for r in targets:
                t=templates[r['template_id']]
                operators=constraint.get('operators',OPERATORS) if factor=='operator' else [parent['operator']]
                for operator in operators:
                    a={'factor':factor,'observable_id':operator}
                    options=variants(parent,a,r,t,master)
                    options={k:q for k,q in options.items() if identity(q,t) not in seen and identity(q,t)!=parent_sig}
                    if not options:continue
                    key='|'.join([pid,r['region_id'],factor,operator]);aid='A'+hashlib.sha256(key.encode()).hexdigest()[:12]
                    def parameter_value(q):
                        if factor=='region':return q['region_id']
                        if factor=='window_start':return q['window'][0]
                        value=q
                        for key in factor.split('.'):value=value[key]
                        return value
                    actions.append({'action_id':aid,'parent_program_id':pid,'region_id':r['region_id'],
                        'factor':factor,'observable_id':operator,'untried_variants':list(options),
                        'new_values':{k:parameter_value(q) for k,q in options.items()},'options':options})
    return actions
