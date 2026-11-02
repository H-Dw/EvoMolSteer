"""Read saved complete populations with one metric definition; no sampling/API/model."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, gzip, hashlib, io, json, math, statistics, collections
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+'\n')


def stats(values, metric):
    vals = [(i, float(v)) for i, v in enumerate(values) if v is not None and math.isfinite(float(v))]
    if not vals:
        return dict(n=0, missing=len(values), mean=None, best=None, filtered_mean=None, top10_mean=None)
    numbers = [v for _, v in vals]
    q1, q3 = np.quantile(numbers, [.25, .75], method='linear')
    lo, hi = (-math.inf, q3+max(3*(q3-q1), 1.)) if metric == 'rmsd' else (q1-3*(q3-q1), q3+3*(q3-q1))
    removed = [i for i, v in vals if v < lo or v > hi] if len(vals) >= 10 else []
    eligible = len(vals) >= 10 and len(removed) <= .1*len(vals)+1e-12
    kept = [v for i, v in vals if i not in removed]
    top = sorted(numbers, reverse=metric == 'pic50')[:math.ceil(.1*len(vals))]
    return dict(n=len(vals), missing=len(values)-len(vals), mean=statistics.mean(numbers),
        best=max(numbers) if metric == 'pic50' else min(numbers),
        filtered_mean=statistics.mean(kept) if eligible and kept else None,
        top10_mean=statistics.mean(top), top10_n=len(top), excluded=removed,
        filter_eligible=eligible, lower=None if math.isinf(lo) else float(lo), upper=float(hi))


def main():
    import torch, rdkit
    from rdkit import Chem, RDLogger
    from rdkit.Contrib.SA_Score import sascorer
    from scipy.optimize import linear_sum_assignment
    torch.set_num_threads(1)
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cases = read(Path(args.cases))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    groups, candidates, cache, budgets = [], [], {}, {}
    for case in cases:
        folder, job = Path(case['folder']), Path(case['job_root'])
        assert (folder/'COMPLETE.json').exists()
        ag, records = read(folder/'aggregate.json'), read(folder/'records.json')
        assert ag['attempts'] == len(records) == 50 and ag['complete']
        if case['role'] == 'primary':
            version = read(job/'versions/GD01/FINALIZED.json')
            assert version['completed_rounds'] == 5
        if str(job) not in budgets:
            ledger = read(job/'attempts.json')
            budgets[str(job)] = dict(collections.Counter())
            for entry in ledger.values():
                stage = entry['stage']
                budgets[str(job)][stage] = budgets[str(job)].get(stage, 0)+entry['count']
        condition_path = next(p/'conditions'/f"{case['teacher']}.json" for p in folder.parents
                              if (p/'conditions'/f"{case['teacher']}.json").exists())
        condition = read(condition_path)
        lookup = {a['id']:i for i, a in enumerate(condition['atoms'])}
        target = np.array([condition['reference_coordinates_angstrom'][lookup[a]] for a in condition['controlled_atom_ids']], dtype=np.float64)
        assert target.shape == (9, 3)
        frame = condition['protein_alignment']
        terminal_sha = sha(folder/'terminal.pt.gz')
        terminal = torch.load(io.BytesIO(gzip.decompress((folder/'terminal.pt.gz').read_bytes())), map_location='cpu', weights_only=False)
        xyz = terminal['coords'].double().numpy() @ np.array(frame['rotation']) + np.array(frame['translation_angstrom'])
        mask = terminal['mask'].bool().numpy()
        assert len(xyz) == 50
        head = terminal.get('affinity', {}).get('pic50')
        pic50 = head.double().reshape(-1).tolist() if head is not None else [None]*50
        assert len(pic50) == 50
        binding = read(folder/'CHEMICAL_CONDITION.json') if (folder/'CHEMICAL_CONDITION.json').exists() else None
        features = {k:np.array(v).argmax(-1) for k, v in binding['features'].items()} if binding else {}
        categories = {k:v.argmax(-1).numpy() for k, v in terminal.items() if k in features}
        group_rows = []
        for i, row in enumerate(records):
            valid = np.flatnonzero(mask[i])
            cost = ((target[:, None]-xyz[i, valid][None])**2).sum(-1)
            assert np.isfinite(cost).all()
            left, right = linear_sum_assignment(cost)
            ids = valid[right[np.argsort(left)]]
            selected = xyz[i, ids]
            rmsd = float(np.sqrt(np.mean(np.sum((selected-target)**2, axis=-1))))
            u, _, vt = np.linalg.svd((selected-selected.mean(0)).T @ (target-target.mean(0)))
            rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
            fitted = (selected-selected.mean(0)) @ rotation
            fitted_rmsd = float(np.sqrt(np.mean(np.sum((fitted-(target-target.mean(0)))**2, axis=-1))))
            chemical = {}
            for key in ('atomics', 'charges', 'hybridization'):
                if key in features:
                    matched = categories[key][i, ids] == features[key]
                    chemical[key+'_fraction'] = float(matched.mean())
                    chemical[key+'_all'] = bool(matched.all())
            if 'bonds' in features:
                predicted = categories['bonds'][i][np.ix_(ids, ids)]
                upper = np.triu(np.ones((9,9), dtype=bool), 1)
                positive = upper & (features['bonds'] != 0)
                chemical['induced_bond_fraction'] = float((predicted[upper] == features['bonds'][upper]).mean())
                chemical['positive_bond_fraction'] = float((predicted[positive] == features['bonds'][positive]).mean()) if positive.any() else None
                chemical['induced_bonds_all'] = bool((predicted[upper] == features['bonds'][upper]).all())
                chemical['all_measured_channels_and_induced_bonds'] = all(value for key, value in chemical.items() if key.endswith('_all'))
            sa = None
            if row['build_success'] and row.get('smiles'):
                smi = row['smiles']
                if smi not in cache:
                    mol = Chem.MolFromSmiles(smi)
                    cache[smi] = float(sascorer.calculateScore(Chem.RemoveHs(mol))) if mol is not None else None
                sa = cache[smi]
            group_rows.append(dict(case_id=case['id'], seed=row['logical_seed'], slot=row['logical_slot'],
                node_id=row['node_id'], built=row['build_success'], connected=row['connected'],
                rmsd=rmsd, pic50=pic50[i], sa=sa, rigid_fit_rmsd=fitted_rmsd, loss=rmsd*rmsd/2,
                initial_assignment_rmsd=row.get('initial_assignment_condition_coordinate_root_mean_square_deviation_angstrom'),
                centroid_shift=float(np.linalg.norm(selected.mean(0)-target.mean(0))), chemistry=chemical,
                saved_rmsd=row.get('condition_coordinate_root_mean_square_deviation_angstrom')))
        assert len({(r['seed'],r['slot']) for r in group_rows}) == 50
        metrics = {k:stats([r[k] for r in group_rows], 'rmsd' if 'rmsd' in k else k)
                   for k in ('rmsd','pic50','sa','rigid_fit_rmsd','loss','initial_assignment_rmsd','centroid_shift')}
        assert abs(metrics['rmsd']['filtered_mean']-ag['ranking_condition_coordinate_root_mean_square_deviation_angstrom']) < 1e-4
        chemistry = {key:dict(n=sum(row['chemistry'].get(key) is not None for row in group_rows),
            mean=statistics.mean(row['chemistry'][key] for row in group_rows if row['chemistry'].get(key) is not None))
            for key in set(k for row in group_rows for k in row['chemistry'])
            if any(row['chemistry'].get(key) is not None for row in group_rows)}
        q = read(folder/'QUALITY_COMPLETE.json') if (folder/'QUALITY_COMPLETE.json').exists() else None
        quality_rows = read(folder/'quality.json') if (folder/'quality.json').exists() else []
        parts = {}
        for row in quality_rows:
            qual = row.get('quality') or {}
            if qual.get('posebusters_evaluable'):
                for name, value in (qual.get('posebusters_checks') or {}).items():
                    part = parts.setdefault(name, dict(evaluable=0, passed=0))
                    part['evaluable'] += 1
                    part['passed'] += int(bool(value))
        if q:
            assert (q['attempts'],q['checked_attempts'],q['not_selected'],q['coverage']) == (50,10,40,.2)
        source_sha = {name:sha(folder/name) for name in ('records.json','terminal.pt.gz','aggregate.json','COMPLETE.json')}
        if q:
            source_sha.update({name:sha(folder/name) for name in ('quality.json','QUALITY_COMPLETE.json')})
        assert source_sha['terminal.pt.gz'] == terminal_sha
        groups.append(dict(case, schedule=ag['effective_guidance_scale_schedule'], population=50,
            built=sum(row['built'] for row in group_rows), connected=sum(row['connected'] for row in group_rows),
            unique_smiles=ag.get('unique_smiles'), saved_ranking=ag['ranking_condition_coordinate_root_mean_square_deviation_angstrom'],
            metrics=metrics, chemistry=chemistry, quality=q, posebusters_components=parts,
            plip_mean=statistics.mean(row['plip_recovery'] for row in quality_rows if row.get('plip_recovery') is not None) if any(row.get('plip_recovery') is not None for row in quality_rows) else None,
            protein_alignment=frame, condition_sha256=condition['condition_sha256'],
            condition_file_sha256=sha(condition_path), source_sha256=source_sha,
            job_budget=budgets[str(job)], remove_aromaticity=binding.get('remove_aromaticity') if binding else None))
        candidates.extend(group_rows)
    report = dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        cases_sha256=sha(Path(args.cases)), analysis_code_sha256=sha(Path(__file__)),
        rdkit=rdkit.__version__, torch=torch.__version__, sascorer_sha256=sha(Path(sascorer.__file__)),
        sa_fragments_sha256=sha(Path(sascorer.__file__).with_name('fpscores.pkl.gz')),
        rmsd_definition='Double precision terminal injective Hungarian of nine original PLIP points in protein-registered frame; no ligand fit.',
        fitted_diagnostic_definition='Proper Kabsch of same terminal-assigned nine points, no rematching/scaling/reflection; not ranking.',
        affinity_definition='Saved terminal affinity.pic50 only; all predicted rows including failed builds; no forward/rescore.',
        SA_definition='Standard RDKit SA on successfully built SMILES only; not 3D conformation quality.',
        outlier_policy='At least10 finite, at most10% excluded; RMSD upper Q3+max(3IQR,1A); pIC50/SA outer3IQR; missing/nonfinite not outliers.',
        top10_policy='Best ceil(10% of finite observations), each metric selected independently; pIC50 descending, RMSD/SA ascending.',
        groups=groups, new_sampling=0,new_model_forwards=0,new_llm_calls=0)
    write(out/'saved_population_readings.json', report)
    with gzip.open(out/'comparison_candidates.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(out),groups=len(groups),candidate_views=len(candidates),
        output_sha256={p.name:sha(p) for p in out.iterdir()},new_sampling=0,new_model_forwards=0,new_llm_calls=0)),flush=True)


if __name__ == '__main__':
    main()
