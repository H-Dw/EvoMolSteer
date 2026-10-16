"""Class-balanced elemental layout diagnostics, separate from chemical-role claims."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import read_json,write_json,digest
from .bank import lookup


def class_scores(wanted,got,vocabulary):
    wanted=np.asarray(wanted);got=np.asarray(got);row={};recalls=[];nc=[]
    for element in ('C','N','O','S','F','Cl','Br','P'):
        if element not in vocabulary:continue
        value=vocabulary[element];positive=wanted==value;prediction=got==value
        tp=int((positive&prediction).sum());expected=int(positive.sum());generated=int(prediction.sum())
        row[element+'_target_count']=expected;row[element+'_generated_count']=generated
        row[element+'_recall']=tp/expected if expected else None
        row[element+'_precision']=tp/generated if generated else None
        if expected:recalls.append(tp/expected)
        if element in ('N','C') and expected:nc.append(tp/expected)
    row['macro_element_recall']=float(np.mean(recalls)) if recalls else None
    row['NC_balanced_recall']=float(np.mean(nc)) if len(nc)==2 else None
    return row


def analyze(folder,bank,output):
    folder=Path(folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit=read_json(folder/'guidance_audit.json');p=audit['program'];geometry=read_json(folder/'geometry.json')
    last=next(r for r in reversed(audit['steps']) if r['active']);records=read_json(folder/'records.json');rows=[]
    for i,r in enumerate(records):
        s=p['mode_specs'][int(last['mode_indices'][i])];m=bank['modes'][s['template_id']];slots=s['atom_slots']
        wanted=np.asarray([m['atom_vocabulary'][m['native_elements'][j]] for j in slots]);tracked=np.asarray(last['actual_local_assignments'][i])
        for name,key in [('boundary','teacher_clock'),('terminal','terminal')]:
            xyz=np.asarray(geometry[key]['coords_A'])[i];atoms=np.asarray(geometry[key]['atomic_indices'])[i]
            target=lookup(m,p['learning_window'][1])[slots] if name=='boundary' else np.asarray(m['terminal_coords_A'])[slots]
            rr,cc=linear_sum_assignment(np.square(xyz[:,None]-target).sum(-1));matched=rr[np.argsort(cc)]
            for assignment,ids in [('tracked',tracked),('geometry_matched',matched)]:
                got=atoms[ids];row={'slot':i,'pic50':r['pic50'],'qualified':r['qualified_structure'],'state':name,'correspondence':assignment,'support_count':len(slots)}
                row.update(class_scores(wanted,got,m['atom_vocabulary']))
                rows.append(row)
    table=pd.DataFrame(rows);table.to_parquet(output/'element_layout.parquet',compression=None,index=False)
    summary=[]
    for (stage,correspondence),part in table.groupby(['state','correspondence']):
        summary.append({'state':stage,'correspondence':correspondence,'means':part.select_dtypes(include=[np.number]).drop(columns=['slot','pic50']).mean().replace({np.nan:None}).to_dict()})
    result={'sources':{name:digest(folder/name) for name in ('guidance_audit.json','geometry.json','records.json')},
        'n_attempts':len(records),'summaries':summary,'limits':['Carbon-dominated total accuracy is not N placement accuracy.',
            'Source-terminal labels are a retrospective layout comparator at the boundary, not actual early-state chemical roles.',
            'Geometric correspondence is type-blind; tracked correspondence is the last actual controller correspondence.',
            'No nitrogen acceptor/donor, microstate, H-bond or affinity causality is inferred.']}
    write_json(output/'summary.json',result);return result
