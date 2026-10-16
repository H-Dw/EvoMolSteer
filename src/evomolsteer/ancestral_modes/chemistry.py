"""Terminal graph-local group measurements; no noisy-state chemical assertions."""
from functools import lru_cache
from pathlib import Path
import pandas as pd
from evomolsteer.io import digest


def groups_from_graph(roles,edges):
    atoms={int(a['atom_index']):a for a in roles};adj={i:[] for i in atoms}
    for e in edges:
        i,j=int(e['i']),int(e['j']);kind=str(e['bond_type'])
        adj[i].append((j,kind));adj[j].append((i,kind))
    groups=[]
    for i,a in atoms.items():
        if a['element']!='C':continue
        carbonyl=[j for j,k in adj[i] if atoms[j]['element']=='O' and k=='DOUBLE']
        if not carbonyl:continue
        os=[j for j,k in adj[i] if atoms[j]['element']=='O' and k=='SINGLE']
        ns=[j for j,k in adj[i] if atoms[j]['element']=='N' and k=='SINGLE']
        for j in os:
            ids=[i,carbonyl[0],j];charge=sum(int(atoms[k]['formal_charge']) for k in ids)
            groups.append({'group':'carboxylate' if charge<0 else 'neutral_carboxyl_group',
                'atoms':sorted(ids),'formal_charge_sum':charge,'hydrogen_microstate_verified':False})
        for j in ns:groups.append({'group':'amide','atoms':sorted([i,carbonyl[0],j]),
            'formal_charge_sum':sum(int(atoms[k]['formal_charge']) for k in (i,carbonyl[0],j)),
            'hydrogen_microstate_verified':False})
        if not os and not ns:groups.append({'group':'other_carbonyl','atoms':sorted([i,carbonyl[0]]),'hydrogen_microstate_verified':False})
    return groups


@lru_cache(maxsize=32)
def terminal_group_context(report_path,node,roles_json):
    import json
    report=Path(report_path);source=report/'reference_inputs/inputs/final_bonds.csv.gz'
    table=pd.read_csv(source);edges=table[table.node_id.eq(node)].to_dict('records')
    groups=groups_from_graph(json.loads(roles_json),edges)
    return {'groups':groups,'bond_source_sha256':digest(source),'scope':'Observed terminal graph local groups; no early-state role assignment or new energy calculation.'}
