"""Compact posthoc seed, graph-representation and historical-cohort diagnostics."""
import argparse
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors, rdMolDescriptors
from evomolsteer.io import read_json, write_json, digest
from .bank import lookup


def load_mol(folder, slot):
    with gzip.open(Path(folder)/f'{slot:03d}.sdf.gz', 'rt') as stream:
        mol = Chem.MolFromMolBlock(stream.read().split('$$$$')[0], sanitize=False, removeHs=False)
    Chem.SanitizeMol(mol); return mol


def chemical_description(mol, protein):
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'))
    f = factory.GetFeaturesForMol(mol)
    acceptors = {i for v in f if v.GetFamily() == 'Acceptor' for i in v.GetAtomIds()}
    donors = {i for v in f if v.GetFamily() == 'Donor' for i in v.GetAtomIds()}
    points = mol.GetConformer().GetPositions(); pairs = []; contact_residues = set()
    for j, atom in enumerate(mol.GetAtoms()):
        for name, position, element in protein:
            distance = float(np.linalg.norm(points[j]-position))
            if distance <= 4.: contact_residues.add(name.rsplit(':', 1)[0])
            if atom.GetSymbol() in ('N', 'O') and element in ('N', 'O') and distance <= 4.1:
                pairs.append({'ligand_atom': j, 'element': atom.GetSymbol(), 'aromatic': atom.GetIsAromatic(),
                              'acceptor_proxy': j in acceptors, 'donor_proxy': j in donors,
                              'protein_atom': name, 'distance_A': distance})
    return {'canonical_smiles': Chem.MolToSmiles(mol), 'formula': rdMolDescriptors.CalcMolFormula(mol),
            'formal_charge': Chem.GetFormalCharge(mol), 'rings': rdMolDescriptors.CalcNumRings(mol),
            'aromatic_atoms': sum(a.GetIsAromatic() for a in mol.GetAtoms()),
            'HBA': Descriptors.NumHAcceptors(mol), 'HBD': Descriptors.NumHDonors(mol),
            'contact_residues_4A': sorted(contact_residues), 'polar_proximities': sorted(pairs, key=lambda r:r['distance_A']),
            'limits': 'RDKit roles and heavy-atom distances are geometric hypotheses, not verified hydrogen bonds or binding free energies.'}


def analyze(repo, config, output):
    repo = Path(repo).resolve(); m = read_json(config); study = repo/m['campaign_study']
    output = Path(output); output.mkdir(parents=True, exist_ok=True); sources = {}
    def read(path):
        path = Path(path); sources[path.relative_to(repo).as_posix()] = digest(path); return read_json(path)
    state = read(study/'state.json'); root = repo/m['generated_root']
    focal_round = state['frozen_selection']['program_id']; p = read(study/focal_round/'selected_program.json')
    teacher = read(study/focal_round/'candidates/registered/teacher.json'); mode = teacher['modes'][p['mode_specs'][0]['template_id']]
    folders = {'R02': root/focal_round/focal_round, 'R15': repo/state['reference']['result_folder'],
               'unguided': root/'UNGUIDED_b00_n50/unguided'}
    records = {k: read(v/'records.json') for k,v in folders.items()}
    top = max((r for r in records['R02'] if r['qualified_structure']), key=lambda r:r['pic50']); slot = top['slot']
    original = repo/m['raw_dataset']/'results'/m['campaign']/m['arm']/f'batch_{mode["batch"]:03d}'
    source_record = read(original/'final_records.json')[mode['terminal_slot']]
    source_mol = next(v for v in Chem.SDMolSupplier(str(original/'molecules_all_built.sdf'), sanitize=False, removeHs=False)
                      if v is not None and v.GetProp('node_id') == source_record['node_id'])
    Chem.SanitizeMol(source_mol)
    protein_file = repo/'data/reference_terminal/original/inputs/3PE1_protein_aligned.pdb'; protein = []
    for path in (protein_file,original/'molecules_all_built.sdf',Path(__file__).resolve()):
        sources[path.relative_to(repo).as_posix()]=digest(path)
    for line in protein_file.read_text().splitlines():
        if line[:6].strip() not in ('ATOM', 'HETATM') or line[76:78].strip() == 'H': continue
        name = ':'.join([line[21],line[17:20].strip(),line[22:26].strip(),line[12:16].strip()])
        protein.append((name, np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]),line[76:78].strip()))
    molecules = {'teacher_F': source_mol, **{k:load_mol(v,slot) for k,v in folders.items()}}
    chemistry = {k:chemical_description(v,protein) for k,v in molecules.items()}
    write_json(output/'best_seed_chemistry.json', chemistry)
    geometries = {k:read(v/'geometry.json') for k,v in folders.items()}
    support = p['mode_specs'][0]['atom_slots']; geometry = {}
    boundary_time=float(p.get('teacher_time',p['learning_window'][1]))
    for name,g in geometries.items():
        if not np.isclose(g['teacher_clock']['time'],boundary_time,atol=2e-6):raise ValueError('Different captured boundary clocks')
        geometry[name] = {}
        for key,target in [('teacher_clock',lookup(mode,boundary_time)[support]), ('terminal',np.asarray(mode['terminal_coords_A'])[support])]:
            x = np.asarray(g[key]['coords_A'])[slot]; cost=((x[:,None]-target[None])**2).sum(-1)
            a,b = linear_sum_assignment(cost)
            geometry[name][key+'_common_geometry_only_rms_A'] = float(np.sqrt(cost[a,b].mean()))
    for key in ('teacher_clock','integration_current','terminal'):
        d=np.asarray(geometries['R02'][key]['coords_A'])[slot]-np.asarray(geometries['R15'][key]['coords_A'])[slot]
        geometry['R02_vs_R15_'+key+'_native_order_rms_A'] = float(np.sqrt((d*d).sum(-1).mean()))
    for d in folders.values():sources[(d/'categorical_states.npz').relative_to(repo).as_posix()]=digest(d/'categorical_states.npz')
    a=np.load(folders['R02']/'categorical_states.npz');b=np.load(folders['R15']/'categorical_states.npz')
    edge_diff=np.argwhere(np.triu(a['terminal_bonds'][slot]!=b['terminal_bonds'][slot],k=1));pairs=[]
    for i,j in edge_diff:
        pairs.append({'i':int(i),'j':int(j),'R02_raw_label':int(a['terminal_bonds'][slot,i,j]),
                      'R15_raw_label':int(b['terminal_bonds'][slot,i,j]),
                      'sanitized_bond':str(molecules['R02'].GetBondBetweenAtoms(int(i),int(j)).GetBondType())})
    graph_audit={'canonical_graph_equal':chemistry['R02']['canonical_smiles']==chemistry['R15']['canonical_smiles'],
                 'raw_atomic_labels_equal':bool(np.array_equal(a['terminal_atomics'][slot],b['terminal_atomics'][slot])),
                 'raw_charge_labels_equal':bool(np.array_equal(a['terminal_charges'][slot],b['terminal_charges'][slot])),
                 'changed_undirected_edges':pairs}
    graph_audit['native_to_sdf_coordinate_max_error_A']={name:float(np.max(np.abs(
        np.asarray(geometries[name]['terminal']['coords_A'])[slot]-molecules[name].GetConformer().GetPositions())))
        for name in ('R02','R15')}
    paired_rows=[];pair_stats={}
    for other in ('R15','unguided'):
        d=np.array([x['pic50']-y['pic50'] for x,y in zip(records['R02'],records[other])])
        pair_stats[other]={'mean':float(d.mean()),'median':float(np.median(d)),
            'improved_above_1e_3':int((d>.001).sum()),'declined_below_minus_1e_3':int((d<-.001).sum()),
            'within_1e_3':int((np.abs(d)<=.001).sum()),'top_slot_difference':float(d[slot]),
            'same_nonmissing_SMILES':sum(x['smiles']==y['smiles'] and x['smiles'] is not None for x,y in zip(records['R02'],records[other]))}
        paired_rows += [{'slot':i,'comparison':'R02_minus_'+other,'score_difference':float(v)} for i,v in enumerate(d)]
    pd.DataFrame(paired_rows).to_parquet(output/'paired_development.parquet',index=False,compression=None)
    rows=[];allg=[];allu=[];hashes=[]
    for r in state['rounds']:
        if r['stage']!='confirmation': continue
        for af,bf in zip(r['result_folders'],r['unguided_result_folders']):
            ar=read(repo/af/'records.json');br=read(repo/bf/'records.json');allg+=ar;allu+=br
            audit=read(repo/af/'guidance_audit.json');hashes.append({'batch':Path(af).parent.name,**{k:audit['input_pairing'][k] for k in ('seed','initial_sha256','initial_slots')}})
            f=pd.read_parquet(study/r['round']/('outcomes_'+Path(af).parent.name)/'per_seed.parquet')
            for row,aa,bb in zip(f.to_dict('records'),ar,br):
                rows.append({**row,'batch':Path(af).parent.name,'unguided_pic50':bb['pic50'],'paired_gain':aa['pic50']-bb['pic50']})
    frame=pd.DataFrame(rows);frame.to_parquet(output/'frozen_seed_features.parquet',index=False,compression=None)
    correlations={}
    for col in ('initial_target_error_A','last_target_error_A','boundary_posthoc_local_rms_A','terminal_posthoc_local_rms_A',
                'terminal_geometry_matched_terminal_element_fraction','actual_cumulative_step_rms_A'):
        correlations[col]={k:float(spearmanr(frame[col],frame[k]).statistic) for k in ('pic50','paired_gain')}
    occurrences={}
    for name,rs in {**records,'frozen_guidance':allg,'frozen_unguided':allu}.items():
        matches=[r for r in rs if r['smiles']==top['smiles']]
        occurrences[name]={'n':len(rs),'same_best_SMILES':len(matches),'scores':[r['pic50'] for r in matches],
                           'unique_valid_graphs':len({r['smiles'] for r in rs if r['qualified_structure']})}
    historical=[]
    for name in ('R15_v2','R19_v2','R20_v2'):
        d=repo/'docs/experiments/local_motif20_v2_20261015'/name/'results.json'
        r=read(d);metrics=r['metrics'];historical.append({'condition':'v2_'+name,**metrics})
    older=read(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/final_quality.json')
    hmean=older['M2']['policy_mean']['mean_pic50'];hbase=older['native']['policy_mean']['mean_pic50']
    vmean=np.mean([r['pic50'] for r in allg]);vbase=np.mean([r['pic50'] for r in allu])
    history={'M2_final400_mean':hmean,'M2_matched_native400_mean':hbase,'M2_gain':hmean-hbase,
             'v3_final400_mean':float(vmean),'v3_matched_native400_mean':float(vbase),'v3_gain':float(vmean-vbase),
             'mean_difference':float(vmean-hmean),'background_difference':float(vbase-hbase),
             'effect_difference':float((vmean-vbase)-(hmean-hbase)),
             'scope':'Arithmetic cohort decomposition, not a matched causal comparison of old M2 and v3.'}
    old24=read(repo/'data/generated/local_motif20_v2_20261015/R15_v2/R15_v2/records.json')
    x=np.array([r['pic50'] for r in old24]);y=np.array([r['pic50'] for r in records['R15']])
    small_n=len(old24)
    sample_size={'old_B24_mean':float(x.mean()),'B50_first24_mean':float(y[:small_n].mean()),'B50_last26_mean':float(y[small_n:].mean()),
                 'B50_all_mean':float(y.mean()),'same_priors_but_execution_shape_gap':float(y[:small_n].mean()-x.mean()),
                 'additional_slots_composition_gap':float(y.mean()-y[:small_n].mean()),
                 'scope':'Changing batch shape changes subsequent random-number consumption; first24 are not the identical inference paths.'}
    dev_scores=np.array([r['pic50'] for r in records['R02']])
    mean_path={'R15_B24':float(x.mean()),'R15_B50':float(y.mean()),'R02_B50':float(dev_scores.mean()),
               'R02_frozen400':float(vmean),'B24_to_B50_difference':float(y.mean()-x.mean()),
               'R15_to_R02_same_B50_difference':float(dev_scores.mean()-y.mean()),
               'R02_development_to_frozen_difference':float(vmean-dev_scores.mean())}
    replay_path=study/'mechanism_followup/exact_replay.json';replay_audit=None
    if replay_path.exists():
        replay=read(replay_path);folder=repo/replay['result_folder'];z=np.load(folder/'categorical_states.npz')
        replay_audit={'scores_bitwise_equal':replay['scores_bitwise_equal'],'max_abs_score_difference':replay['max_abs_score_difference'],
                      'n':replay['n'],'new_optimization_rounds':0,
                      'geometry_json_bytes_equal':(folders['R02']/'geometry.json').read_bytes()==(folder/'geometry.json').read_bytes(),
                      'all_categorical_arrays_equal':all(np.array_equal(a[k],z[k]) for k in a.files)}
    ref_atoms=a['terminal_atomics'][slot];ref_bonds=a['terminal_bonds'][slot]
    frontier=[];family=[]
    for r in state['rounds']:
        if r['stage']!='discovery':continue
        a=r['metrics'];dominated=any(c['stage']=='discovery' and c['metrics']['mean_all_pic50']>=a['mean_all_pic50'] and
            c['metrics']['best_qualified_pic50']>=a['best_qualified_pic50'] and
            (c['metrics']['mean_all_pic50']>a['mean_all_pic50'] or c['metrics']['best_qualified_pic50']>a['best_qualified_pic50']) for c in state['rounds'])
        frontier.append({'round':r['round'],'mean':a['mean_all_pic50'],'maximum':a['best_qualified_pic50'],'pareto_nondominated':not dominated})
        d=repo/r['result_folder'];best=max((v for v in read(d/'records.json') if v['qualified_structure']),key=lambda v:v['pic50'])
        if best['smiles']==top['smiles']:
            index=best['slot'];raw=np.load(d/'categorical_states.npz');pos=np.array(read(d/'geometry.json')['terminal']['coords_A'])[index]
            ref=np.array(geometries['R02']['terminal']['coords_A'])[slot]
            family.append({'round':r['round'],'best_slot':index,'pic50':best['pic50'],'support':len(r['program']['mode_specs'][0]['atom_slots']),
                'raw_atom_mismatches':int((raw['terminal_atomics'][index]!=ref_atoms).sum()),
                'raw_undirected_bond_mismatches':int(np.triu(raw['terminal_bonds'][index]!=ref_bonds,1).sum()),
                'native_order_coordinate_rms_A':float(np.sqrt(((pos-ref)**2).sum(1).mean()))})
    pd.DataFrame(frontier).to_csv(output/'development_frontier.csv',index=False)
    pd.DataFrame(family).to_csv(output/'best_graph_family.csv',index=False)
    scoring_path=study/'mechanism_followup/score_probe_results.json';attribution=None
    if scoring_path.exists():
        if not all(graph_audit[k] for k in ('canonical_graph_equal','raw_atomic_labels_equal','raw_charge_labels_equal')):
            raise ValueError('This interpretation requires equivalent graph representations, not distinct chemical graphs')
        if any(r['sanitized_bond']!='AROMATIC' for r in pairs) or max(graph_audit['native_to_sdf_coordinate_max_error_A'].values())>.001:
            raise ValueError('Kekule representation or native-to-molecule mapping not established')
        scores=read(scoring_path);manifest=read(study/'mechanism_followup/score_input_manifest.json')
        v={r['variant']:r for r in scores['rows']}
        errors={k:float(np.max(np.abs(np.array(v[name]['all_scores'])-manifest['sources'][k]['expected_scores'])))
                for k,name in [('A','A_coords_A_bonds'),('B','B_coords_B_bonds')]}
        if max(errors.values())>manifest['tolerance_pic50']:raise ValueError('Terminal score reconstruction failed')
        aa=v['A_coords_A_bonds']['focal_pic50'];bb=v['B_coords_B_bonds']['focal_pic50']
        ab=v['A_coords_B_bonds']['focal_pic50'];ba=v['B_coords_A_bonds']['focal_pic50']
        attribution={'variants':{k:r['focal_pic50'] for k,r in v.items()},'native_reconstruction_max_errors':errors,
            'total_R02_minus_R15':aa-bb,'coordinate_effect_at_R02_bonds':aa-ba,'coordinate_effect_at_R15_bonds':ab-bb,
            'coordinate_shapley_effect':.5*((aa-ba)+(ab-bb)),
            'bond_representation_effect_at_R02_coords':aa-ab,'bond_representation_effect_at_R15_coords':ba-bb,
            'bond_representation_shapley_effect':.5*((aa-ab)+(ba-bb)),
            'scope':'Exact two-factor head-input decomposition on equivalent canonical graphs. This is not a binding-free-energy or experimental affinity attribution.',
            'additional_generation':False,'not_a_new_optimization_best':True}
        assert abs(attribution['coordinate_shapley_effect']+attribution['bond_representation_shapley_effect']-(aa-bb))<1e-10
    report={'best_record':top,'source_record':{k:source_record[k] for k in ('node_id','pic50_on_rescore','smiles')},
        'geometry':geometry,'graph_representation':graph_audit,'paired_development':pair_stats,
        'best_graph_occurrence':occurrences,'frozen_correlations_descriptive':correlations,
        'old_M2_vs_current':history,'sample_size_decomposition':sample_size,'v2_conditions':historical,
        'mean_comparison_path':mean_path,'exact_replay':replay_audit,
        'focal_local_metrics':pd.read_parquet(study/focal_round/('outcomes_'+focal_round)/'per_seed.parquet').query('slot == @slot').to_dict('records')[0],
        'development_score_quantiles':{name:np.quantile([r['pic50'] for r in rs],[.1,.5,.9,.95,1]).tolist() for name,rs in records.items()},
        'terminal_score_factorial_attribution':attribution,
        'confirmation_initials':hashes,'development_initial':read(folders['R02']/'guidance_audit.json')['input_pairing'],
        'sources':sources,'limits':['No mechanistic affinity attribution from contacts alone.',
            'Same canonical graph can have different raw Kekule inputs; see factorial real-head scoring.',
            'Development frontier is search-biased and descriptive; no unconditional mean/maximum conflict theorem follows.',
            'Original 35-condition search is unchanged; exact replay is a separate repeatability diagnostic.']}
    write_json(output/'summary.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.repo,a.config,a.output);print(r['old_M2_vs_current']);print(r['geometry']);print(r['best_graph_occurrence'])


if __name__=='__main__':main()
