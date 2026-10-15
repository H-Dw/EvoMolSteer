"""Import finite, fixed-graph layout contrasts; never fit a scoring model."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evomolsteer.io import read_json,write_json,digest
from evomolsteer.final_guidance.evidence import trace_teacher,candidate_id


def sites(value):
    return [] if pd.isna(value) or str(value)=='' else [int(v) for v in str(value).split(',')]


def unit(value):
    value=np.asarray(value,float);return value/max(np.linalg.norm(value),1e-9)


def pocket_donors(protein,pocket):
    from rdkit import Chem,RDConfig
    from rdkit.Chem import ChemicalFeatures
    mol=Chem.MolFromPDBFile(str(protein),removeHs=True)
    if mol is None:raise ValueError('Protein chemistry unavailable')
    for atom in mol.GetAtoms():
        info=atom.GetPDBResidueInfo();res=info.GetResidueName().strip();name=info.GetName().strip()
        if res=='LYS' and name=='NZ':atom.SetFormalCharge(1)
        if res=='ARG' and name=='NH1':atom.SetFormalCharge(1)
        if res in ('ASP','GLU') and name in ('OD1','OD2','OE1','OE2') and any(b.GetBondType()==Chem.BondType.SINGLE for b in atom.GetBonds()):atom.SetFormalCharge(-1)
    Chem.SanitizeMol(mol)
    factory=ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'))
    donor_ids={i for f in factory.GetFeaturesForMol(mol) if f.GetFamily()=='Donor' for i in f.GetAtomIds()}
    xyz=mol.GetConformer().GetPositions();px=pocket[['x_A','y_A','z_A']].to_numpy();result=[]
    for i in sorted(donor_ids):
        a=mol.GetAtomWithIdx(i);info=a.GetPDBResidueInfo()
        if info.GetResidueName().strip()=='HIS' or np.linalg.norm(px-xyz[i],axis=1).min()>.01:continue
        near=[v.GetIdx() for v in a.GetNeighbors() if v.GetAtomicNum()>1]
        ray=-sum((unit(xyz[j]-xyz[i]) for j in near),np.zeros(3))
        result.append({'donor_id':f"{info.GetChainId()}:{info.GetResidueName().strip()}:{info.GetResidueNumber()}:{info.GetName().strip()}",
            'residue':f"{info.GetChainId()}:{info.GetResidueName().strip()}:{info.GetResidueNumber()}",
            'coords_A':xyz[i].tolist(),'outward':unit(ray).tolist()})
    return result


def prepare(repo,cfg):
    repo=Path(repo);source=Path(cfg['layout_report']);out=repo/cfg['campaign_study'];out.mkdir(parents=True,exist_ok=True)
    atoms=pd.read_csv(source/'all_atoms.csv.gz');layouts=pd.read_csv(source/'all_layout_scores.csv.gz',dtype={'N_sites':str,'eligible_sites':str})
    bonds=pd.read_csv(source/'reference_inputs/inputs/final_bonds.csv.gz')
    pocket=pd.read_csv(source/'reference_inputs/inputs/pocket_atoms.csv')
    donors=pocket_donors(repo/cfg['dataset']/'inputs'/cfg['protein'],pocket)
    members=pd.read_csv(source/'reference_inputs/inputs/frozen_membership.csv')
    names=sorted(set(members.loc[members.top10,'node_id'])|{cfg['r13_node_id']})
    regions=[];templates={};summaries=[];basecfg=read_json(repo/cfg['base_config'])
    for name in names:
        table=layouts[layouts.node_id.eq(name)].copy();native=table[table.is_original].iloc[0]
        a=atoms[atoms.node_id.eq(name)].sort_values('atom_index');x=a[['x_A','y_A','z_A']].to_numpy()
        ids=a.atom_index.to_numpy();assert np.array_equal(ids,np.arange(len(x)))
        b=bonds[bonds.node_id.eq(name)];adj={i:[] for i in ids}
        for edge in b.itertuples():adj[edge.i].append(edge.j);adj[edge.j].append(edge.i)
        batch=int(table.iloc[0]['batch']);slot=int(table.iloc[0]['slot']);key=candidate_id(batch,slot)
        anc=trace_teacher(repo,{**basecfg,'study':cfg['campaign_study']+'/templates'},
            {'candidate':{'candidate_id':key,'batch':batch,'slot':slot,'pic50_on_rescore':float(native.pic50_target)},'terminal_coords_A':x.tolist()})
        eligible=sites(native.eligible_sites);occupancy={}
        for i in eligible:
            present=table.N_sites.apply(lambda s:i in sites(s));yes=table.loc[present,'pic50_target'];no=table.loc[~present,'pic50_target']
            occupancy[i]={'slot':int(i),'N_minus_C':float(yes.mean()-no.mean()),'n_N':len(yes),'n_C':len(no)}
        ns=sites(native.N_sites);strength=np.zeros((len(eligible),len(donors)))
        for si,i in enumerate(eligible):
            ray=unit(-sum((unit(x[j]-x[i]) for j in adj[i]),np.zeros(3)))
            for di,d in enumerate(donors):
                vector=np.asarray(d['coords_A'])-x[i];distance=np.linalg.norm(vector);u=unit(vector)
                strength[si,di]=np.exp(-.5*((distance-3.)/.4)**2)*max(ray@u,0)**2*max(-u@np.asarray(d['outward']),0)**2
        site_context={i:{**occupancy[i],'native_element':a.iloc[i].element,'coords_A':x[i].tolist(),
            'native_aromatic':bool(a.iloc[i].aromatic),'native_acceptor':bool(a.iloc[i].acceptor),
            'native_donor':bool(a.iloc[i].donor),'native_degree':int(a.iloc[i].degree),'native_formal_charge':int(a.iloc[i].formal_charge),
            'exchange_class':'neutral degree-two pyridine-type aromatic N / aromatic CH; pyrrole [nH] and amide N excluded',
            'neighbors':adj[i],'best_donor':donors[int(np.argmax(strength[j]))]['donor_id'],
            'best_donor_strength':float(strength[j].max())} for j,i in enumerate(eligible)}
        # Native and best measured legal layouts remain distinct observed/counterfactual views.
        variants=[('native',native),('best_legal',table.sort_values('pic50_target',ascending=False,kind='stable').iloc[0])]
        seen=set()
        for label,row in variants:
            nitrogen=sites(row.N_sites)
            if tuple(nitrogen) in seen:continue
            seen.add(tuple(nitrogen));tid=key+'_'+label
            teacher={**anc,'terminal_representation':'terminal_world_A','terminal_coords_A':x.tolist(),
                'atom_vocabulary':read_json(repo/cfg['dataset']/'results'/cfg['campaign']/'config.json')['atom_vocabulary'],
                'charge_vocabulary':read_json(repo/cfg['dataset']/'results'/cfg['campaign']/'config.json')['charge_vocabulary'],
                'native_elements':a.element.tolist(),'desired_N_sites':nitrogen,'eligible_sites':eligible,
                'local_adjacency':{str(i):adj[i] for i in eligible},
                'counterfactual_labels':not bool(row.is_original),'template_id':tid,
                'layout_score':float(row.pic50_target),'evidence_origin':name,'source_kind':'fixed-geometry legal layout score; not a new generation'}
            templates[tid]=teacher
            rankedN=sorted(nitrogen,key=lambda i:(site_context[i]['best_donor_strength'],occupancy[i]['N_minus_C']),reverse=True)
            carbon=sorted(set(eligible)-set(nitrogen),key=lambda i:occupancy[i]['N_minus_C'])
            for ncount in sorted(set([1,min(2,len(rankedN)),len(rankedN)])):
                if ncount==0:continue
                for ccount in (1,2):
                    typed=sorted(rankedN[:ncount]+carbon[:ccount]);support=sorted(set(typed)|{j for i in rankedN[:ncount] for j in adj[i]})
                    if not 2<=len(support)<=cfg.get('max_sparse_points',cfg['max_local_points']) or len(support)>=len(x):continue
                    rid=f'L{len(regions):03d}';chosenN=rankedN[:ncount]
                    # Bind distinct donor RESIDUES, not duplicate contacts to one residue.
                    residues=sorted({d['residue'] for d in donors});cost=np.zeros((len(chosenN),len(residues)))
                    for k,i in enumerate(chosenN):
                        for j,res in enumerate(residues):cost[k,j]=max(strength[eligible.index(i),z] for z,d in enumerate(donors) if d['residue']==res)
                    rr,cc=linear_sum_assignment(-cost);anchors=[]
                    for k,j in zip(rr,cc):
                        i=chosenN[k];possible=[z for z,d in enumerate(donors) if d['residue']==residues[j]]
                        di=max(possible,key=lambda z:strength[eligible.index(i),z]);score=float(strength[eligible.index(i),di])
                        if score>=cfg['anchor_support_threshold']:anchors.append({'slot':i,'neighbor_slots':adj[i],
                            'donor':donors[di],'teacher_match_strength':score})
                    regions.append({'patch_id':rid,'evidence_id':rid+'/finite_layout','template_id':tid,'node_id':name,
                        'typed_slots':typed,'typed_elements':['N' if i in nitrogen else 'C' for i in typed],
                        'atom_slots':support,'anchors':anchors,'layout_score':float(row.pic50_target),
                        'native_score':float(native.pic50_target),'counterfactual_labels':not bool(row.is_original),
                        'site_evidence':[site_context[i] for i in typed],
                        'claim':'Within one fixed backbone, position labels affect the model; site contrasts are dependent, not atom energies'})
        summaries.append({'node_id':name,'native_score':float(native.pic50_target),'best_legal_score':float(table.pic50_target.max()),
            'n_layouts_scored':len(table),'native_minus_alternate_mean':float(native.pic50_target-table.loc[~table.is_original,'pic50_target'].mean()),
            'site_context':list(site_context.values())})
    # Append joint-swap patches without renumbering any previously frozen local ID.
    contexts={c['node_id']:{s['slot']:s for s in c['site_context']} for c in summaries}
    seen={(p['template_id'],tuple(p['typed_slots'])) for p in regions}
    for original in list(regions):
        t=templates[original['template_id']]
        if not t['counterfactual_labels']:continue
        formerN={i for i in t['eligible_sites'] if t['native_elements'][i]=='N'}
        compensation=sorted(formerN-set(t['desired_N_sites']))
        typed=sorted(set(original['typed_slots'])|set(compensation));support=sorted(set(original['atom_slots'])|set(compensation))
        if not compensation or len(support)>cfg.get('max_sparse_points',cfg['max_local_points']) or (original['template_id'],tuple(typed)) in seen:continue
        seen.add((original['template_id'],tuple(typed)));rid=f'L{len(regions):03d}'
        regions.append({**original,'patch_id':rid,'evidence_id':rid+'/finite_layout','typed_slots':typed,'atom_slots':support,
            'typed_elements':['N' if i in t['desired_N_sites'] else 'C' for i in typed],
            'site_evidence':[contexts[original['node_id']][i] for i in typed],
            'joint_layout_compensation':True,'compensating_C_slots':compensation,
            'claim':'Joint N relocation and C replacement within the selected local sites, not global N-count forcing'})
    if cfg.get('r13_context_comparison',False):
        support=read_json(repo/cfg['r13_analyst_path'])['atom_slots'];node=cfg['r13_node_id']
        choices=[(key,t) for key,t in templates.items() if t['evidence_origin']==node]
        native=next(t for key,t in choices if not t['counterfactual_labels']);other=next(t for key,t in choices if t['counterfactual_labels'])
        typed=sorted(set(native['desired_N_sites'])^set(other['desired_N_sites']))
        if not set(typed)<=set(support) or len(support)>=len(native['coords_A']):raise ValueError('R13 context must remain a proper subset containing compared sites')
        for key,t in choices:
            rid=f'L{len(regions):03d}';desired=set(t['desired_N_sites']);anchors=[]
            for prior in regions:
                if prior['template_id']==key:
                    for anchor in prior['anchors']:
                        if anchor['slot'] in typed and anchor not in anchors:anchors.append(anchor)
            regions.append({'patch_id':rid,'evidence_id':rid+'/finite_layout','template_id':key,'node_id':node,
                'typed_slots':typed,'typed_elements':['N' if i in desired else 'C' for i in typed],
                'atom_slots':support,'anchors':anchors,'layout_score':t['layout_score'],'native_score':native['layout_score'],
                'counterfactual_labels':t['counterfactual_labels'],'site_evidence':[contexts[node][i] for i in typed],
                'r13_context':True,'geometry_context_points':len(support),'local_NC_points':len(typed),
                'joint_layout_compensation':t['counterfactual_labels'],
                'claim':'Preserve the historical R13 proper-subset geometric context; compare N/C labels only on the measured two-site relocation, with no full-cloud loss'})
    for patch in regions:
        labels=dict(zip(patch['typed_slots'],patch['typed_elements']))
        patch['N_targets']=[s for s,e in labels.items() if e=='N'];patch['C_targets']=[s for s,e in labels.items() if e=='C']
        patch['site_evidence']=[{**s,'desired_element':labels[s['slot']],
            'native_to_desired':s['native_element']+' -> '+labels[s['slot']]} for s in patch['site_evidence']]
        patch['label_interpretation']='N_targets/C_targets and desired_element define this reward. native_element describes only the source molecule and may be the opposite label in a counterfactual.'
    write_json(out/'templates.json',templates)
    packet={'schema':'local-nc-evidence-2-role-metadata','patches':regions,'candidate_summaries':summaries,
        'support_definition':'ALL atom_slots, including context neighbors, receive direct geometry loss when a geometry term is enabled; only typed_slots receive type loss',
        'max_local_points':cfg['max_local_points'],'teacher_clock':cfg['teacher_time'],
        'report_limits':['Fixed-coordinate legal isomer inputs are not regenerated molecules or independent samples',
            'N matching is not a sufficient or necessary high-affinity rule; N--N spacing is not universally supported',
            'No calibrated physical atom energy or experimentally validated binding gain',
            'Terminal chemical identities are not claimed to exist at the noisy teacher clock',
            'No additional affinity predictor is fitted; generation will not differentiate affinity head'],
        'source_sha256':{n:digest(source/n) for n in ['all_atoms.csv.gz','all_layout_scores.csv.gz','key_findings.json','Top10_spatial_motif_mining.zh-CN.html']}}
    write_json(out/'evidence.json',packet);return packet
