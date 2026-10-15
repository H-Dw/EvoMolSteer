"""Immutable source-linked local hypotheses with compact shared evidence."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from .conditional_layout import contrasts
from .regional_context import context
from .hypothesis_ledger import load_history


def build(repo,master):
    out=repo/master['campaign_study'];out.mkdir(parents=True,exist_ok=True)
    if (out/'evidence.json').exists():return read_json(out/'evidence.json'),read_json(out/'templates.json')
    source=Path(master['layout_report']);all_templates=read_json(repo/master['prior_study']/'templates.json')
    templates={};seen=set()
    for k,t in all_templates.items():
        if t['counterfactual_labels'] or t['candidate_id'] in seen:continue
        seen.add(t['candidate_id']);templates[k]=copy.deepcopy(t)
    atoms=pd.read_csv(source/'all_atoms.csv.gz');layouts=pd.read_csv(source/'all_layout_scores.csv.gz',dtype={'N_sites':str,'eligible_sites':str})
    bonds=pd.read_csv(source/'reference_inputs/inputs/final_bonds.csv.gz')
    pocket=pd.read_csv(source/'reference_inputs/inputs/pocket_atoms.csv');members=pd.read_csv(source/'reference_inputs/inputs/frozen_membership.csv')
    regions=[];cards={};summary=[];prior=load_history(repo,Path(master['prior_study']))
    for tid,t in templates.items():
        node=t['evidence_origin'];a=atoms[atoms.node_id.eq(node)].sort_values('atom_index');x=np.asarray(t['terminal_coords_A'])
        adj={i:[] for i in range(len(x))};bb=bonds[bonds.node_id.eq(node)]
        for e in bb.itertuples():adj[int(e.i)].append(int(e.j));adj[int(e.j)].append(int(e.i))
        t['full_adjacency']={str(k):v for k,v in adj.items()}
        lay=contrasts(layouts[layouts.node_id.eq(node)])
        specs=[];nitrogen=a.loc[a.element.eq('N')&a.aromatic&a.acceptor,'atom_index'].astype(int).tolist()
        oxygen=a.loc[a.element.eq('O')&a.acceptor,'atom_index'].astype(int).tolist()
        def expand(ids,depth=1):
            result=set(ids)
            for _ in range(depth):result|={j for i in list(result) for j in adj[i]}
            return sorted(result)
        # Region identities contain observed atom slots, never borrowed labels.
        for n in nitrogen:specs.append(('N'+str(n),expand([n],2),'N neighborhood; candidate-specific conditional layout'))
        if nitrogen:specs.append(('NC',expand(nitrogen),'Joint observed N/C scaffold context'))
        if oxygen:specs.append(('O',expand(oxygen,2),'O acceptor plus attached scaffold; not generic O attraction'))
        if oxygen and nitrogen:specs.append(('NO',expand(oxygen+nitrogen[:1]),'Separated N/O and connected support; cooperative hypothesis'))
        if t['candidate_id']==master['reference_candidate']:
            reference=read_json(repo/master['initial_program'])
            specs.append(('reference',reference['atom_slots'],'Frozen broad proper-subset coordinate reference'))
        used=set()
        for label,slots,claim in specs:
            if len(slots)<2 or len(slots)>=len(x) or len(slots)>master['max_local_points'] or tuple(slots) in used:continue
            used.add(tuple(slots));rid=t['candidate_id']+'__'+label
            rc=context(node,slots,atoms,pocket,members)
            lengths=bb[bb.i.isin(slots)&bb.j.isin(slots)].length_A.to_numpy()
            # Spatial response calibration scale, not an inferred optimum width.
            width=float(np.median(lengths)*.25) if len(lengths) else .35
            r={'region_id':rid,'template_id':tid,'candidate_id':t['candidate_id'],'node_id':node,'atom_slots':slots,
               'claim':claim,'source_score':t['terminal_score'],'n_points':len(slots),'n_total':len(x),
               'native_elements':[t['native_elements'][i] for i in slots],
               'pocket_sites':[{'index':f['pocket_index'],'coords_A':f['coords_A'],'id':f['site']} for f in rc['pocket_fields'][:6]],
               'base_band_A':width,'band_basis':'Proposed calibration: quarter source local covalent-length median; not a proven success tolerance',
               'orientation_triplets':[[i,*adj[i][:2]] for i in slots if len(adj[i])>=2 and set(adj[i][:2])<=set(slots)][:3]}
            regions.append(r)
            cards[rid]={'context':rc,'conditional_layout':lay,'source':{'teacher_time':t['time'],'teacher_representation':t['representation'],
                'terminal_score':t['terminal_score'],'counterfactual_teacher':False,'trajectory_sha256':t['trajectory_sha256']}}
            summary.append({k:r[k] for k in ['region_id','candidate_id','claim','n_points','n_total','source_score','native_elements']})
    evidence={'schema_version':'local-motif-evidence-2','regions':regions,'catalog':summary,'cards':cards,'history':prior,
        'learning_window':master['learning_window'],
        'group_definitions':{'report_seed_groups':int(members.seed.nunique()),'root_ids':int(members.root_id.nunique()),
             'batches':int(members.batch.nunique()),'chemical_graphs':int(members.smiles.nunique()),
             'rule':'seed, batch and root_id are separate fields; clone and time counts are not independent observations'},
        'limits':['Source layout score is not a generated score forecast.','No universal Top10 motif survived full feature correction.',
            'O fields explain aggregate scores but O-only carboxyl torsion interventions were not significant.',
            'Neither N anchoring nor exact teacher copying is necessary or sufficient for high affinity.',
            'Recorded terminal roles are not asserted at noisy intermediate states.'],
        'O_evidence':{'heldout_full_R2':.3435,'drop_O_R2':.2473,'family_adjusted_p':.02532,
            'Top10_mean_prediction_increment':-.00067,'carboxyl_intervention_states':3340,'significant_carboxyl_intervention':False,
            'status':'Measured source-report explanation; not an established actionable O reward'},
        'source_hashes':{str(p):digest(source/p) for p in ['all_atoms.csv.gz','all_layout_scores.csv.gz','reference_inputs/inputs/final_bonds.csv.gz',
            'reference_inputs/inputs/pocket_atoms.csv','reference_inputs/inputs/frozen_membership.csv','Top10_spatial_motif_mining.zh-CN.html']}}
    write_json(out/'evidence.json',evidence);write_json(out/'templates.json',templates)
    return evidence,templates
