"""Role-conditioned pocket fields and actual candidate controls."""
import numpy as np


def context(node,slots,atoms,pocket,members):
    a=atoms[atoms.node_id.eq(node)].set_index('atom_index').sort_index()
    selected=a.loc[slots];polar=selected[selected.element.isin(['N','O'])]
    pp=pocket[pocket.element.isin(['N','O'])].copy()
    px=pp[['x_A','y_A','z_A']].to_numpy()
    xyz=polar[['x_A','y_A','z_A']].to_numpy()
    if len(xyz):order=np.argsort(np.linalg.norm(xyz[:,None]-px[None],axis=-1).min(0),kind='stable')[:6]
    else:order=np.argsort(np.linalg.norm(px-selected[['x_A','y_A','z_A']].to_numpy().mean(0),axis=-1))[:6]
    pp=pp.iloc[order];px=pp[['x_A','y_A','z_A']].to_numpy()
    m=members[members.node_id.eq(node)]
    same_graph=[] if m.empty else members.loc[members.smiles.eq(m.iloc[0].smiles)&~members.node_id.eq(node),'node_id'].tolist()
    same_root=[] if m.empty else members.loc[members.root_id.eq(m.iloc[0].root_id)&~members.node_id.eq(node),'node_id'].tolist()
    # Group labels are not silently replaced by one another.
    controls={'same_graph':same_graph,'same_root':same_root}
    grouped={kind:list(atoms[atoms.node_id.isin(nodes)].groupby('node_id')) for kind,nodes in controls.items()}
    fields=[]
    for role,select in [('O_acceptor',lambda g:g.element.eq('O')&g.acceptor),
                        ('N_acceptor',lambda g:g.element.eq('N')&g.acceptor)]:
        focal=a[select(a)][['x_A','y_A','z_A']].to_numpy()
        values=None if not len(focal) else np.linalg.norm(focal[:,None]-px[None],axis=-1).min(0)
        cohort_fields={}
        for kind,groups in grouped.items():
            ds=[]
            for other,g in groups:
                pts=g[select(g)][['x_A','y_A','z_A']].to_numpy()
                if len(pts):ds.append(np.linalg.norm(pts[:,None]-px[None],axis=-1).min(0))
            cohort_fields[kind]=np.asarray(ds)
        for j,p in enumerate(pp.itertuples()):
            row={'role':role,'pocket_index':int(p.pocket_index),'site':p.residue+':'+p.atom_name,
                 'coords_A':px[j].tolist(),'candidate_distance_A':None if values is None else float(values[j]),'controls':{}}
            for kind,array in cohort_fields.items():
                ds=array[:,j] if len(array) else []
                row['controls'][kind]={'n':len(ds),'mean_A':float(np.mean(ds)) if len(ds) else None,
                    'q25_q75_A':np.quantile(ds,[.25,.75]).tolist() if len(ds) else None}
            fields.append(row)
    return {'node_id':node,'support':selected.reset_index()[['atom_index','element','aromatic','acceptor','donor','degree','formal_charge']].to_dict('records'),
        'pocket_fields':fields,'control_nodes':controls,'n_same_graph':len(same_graph),'n_same_root':len(same_root),
        'limits':'Source terminal chemistry; field association is not a proven contact or energy. Same-root and same-graph controls are dependent and distinct.'}
