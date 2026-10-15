"""Registered deterministic evidence tools; no model calls or arbitrary code."""
import pandas as pd
import numpy as np
from pathlib import Path


def joint_layout_contrasts(source,patches):
    table=pd.read_csv(Path(source)/'all_layout_scores.csv.gz',dtype={'N_sites':str})
    rows=[]
    for node in sorted({p['node_id'] for p in patches}):
        g=table[table.node_id.eq(node)].sort_values('pic50_target',ascending=False)
        native=g[g.is_original].iloc[0]
        rows.append({'node_id':node,'n_measured_legal_layouts':len(g),'native_score':float(native.pic50_target),
            'native_N_sites':str(native.N_sites),'best_and_counterexample_layouts':
            g.head(3)[['N_sites','pic50_target','is_original']].to_dict('records')+g.tail(2)[['N_sites','pic50_target','is_original']].to_dict('records'),
            'scope':'Finite fixed-geometry fixed-count inputs; layouts are not independent generated molecules'})
    return rows


def local_role_context(source,patches):
    table=pd.read_csv(Path(source)/'all_atoms.csv.gz')
    pocket=pd.read_csv(Path(source)/'reference_inputs/inputs/pocket_atoms.csv')
    rows=[];px=pocket[['x_A','y_A','z_A']].to_numpy()
    for node in sorted({p['node_id'] for p in patches}):
        g=table[table.node_id.eq(node)];atoms=[]
        for a in g[g.element.eq('O')].itertuples():
            xyz=np.asarray([a.x_A,a.y_A,a.z_A]);dist=np.linalg.norm(px-xyz,axis=1);j=int(dist.argmin())
            atoms.append({'slot':int(a.atom_index),'coords_A':xyz.tolist(),'acceptor':bool(a.acceptor),
                'nearest_pocket_atom_row':j,'distance_A':float(dist[j])})
        rows.append({'node_id':node,'O_context':atoms,'scope':'Context evidence only; does not add points to the reward'})
    return rows
