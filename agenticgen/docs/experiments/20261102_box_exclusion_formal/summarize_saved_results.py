"""CPU-only review of immutable populations; no model loading, sampling or API."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, collections, gzip, hashlib, io, json, math, statistics
import numpy as np

ROOT = Path('/workspace/agenticgen_box_exclusion_20261102')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n')


def finite(value):
    return value is not None and math.isfinite(float(value))


def stats(values, metric):
    observed = [float(v) for v in values if finite(v)]
    if not observed:
        return dict(n=0, missing=len(values), mean=None, best=None, filtered_mean=None, top10_mean=None)
    q1, q3 = np.quantile(observed, [.25, .75], method='linear')
    lo, hi = (-math.inf, q3 + max(3*(q3-q1), 1.0)) if metric == 'rmsd' else (q1-3*(q3-q1), q3+3*(q3-q1))
    excluded = [i for i, v in enumerate(values) if finite(v) and (v < lo or v > hi)] if len(observed) >= 10 else []
    kept = [float(v) for i, v in enumerate(values) if finite(v) and i not in excluded]
    eligible = len(excluded) <= .1*len(observed) + 1e-12
    reverse = metric in ('pic50', 'fraction')
    top = sorted(observed, reverse=reverse)[:math.ceil(.1*len(observed))]
    return dict(n=len(observed), missing=len(values)-len(observed), mean=statistics.mean(observed),
                best=(max(observed) if reverse else min(observed)),
                filtered_mean=statistics.mean(kept) if kept and eligible else None,
                top10_mean=statistics.mean(top), top10_n=len(top), excluded_indices=excluded,
                filter_eligible=eligible, lower=None if math.isinf(lo) else float(lo), upper=float(hi))


def quality(folder):
    receipt = read(folder/'QUALITY_COMPLETE.json') if (folder/'QUALITY_COMPLETE.json').exists() else None
    rows = read(folder/'quality.json') if (folder/'quality.json').exists() else []
    parts = {}
    for row in rows:
        for key, value in (row.get('quality', {}).get('posebusters_checks') or {}).items():
            part = parts.setdefault(key, dict(evaluable=0, passed=0))
            part['evaluable'] += 1
            part['passed'] += int(bool(value))
    return dict(receipt=receipt, components=parts,
                plip_recovery=stats([row.get('plip_recovery') for row in rows], 'fraction'),
                sampled_rows=rows, not_extrapolated=True)


def main():
    import torch, rdkit
    from rdkit import Chem, RDLogger
    from rdkit.Contrib.SA_Score import sascorer
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--historical', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT/'PLAN.json')
    assert plan['commit'] == 'c79627ad35670862694fb6dd59dab7272a9e7c9d'
    sources = []
    evidence = []
    for job in plan['jobs']:
        base, teacher = Path(job['root']), job['teacher']
        selection = base/'box_exclusion'/teacher
        for p in sorted(selection.glob('profiles/*/population_50/COMPLETE.json')):
            sources.append(dict(teacher=teacher, label='负面排除预实验·'+p.parent.parent.name, folder=str(p.parent)))
        for p in sorted(base.glob('versions/GD01/*/iteration_*/action_*/COMPLETE.json')):
            sources.append(dict(teacher=teacher, label='多样组合搜索·'+p.parent.parent.name+'/'+p.parent.name, folder=str(p.parent)))
        decisions = []
        for receipt in sorted(base.glob('**/*.receipt.json')):
            stem = receipt.name.removesuffix('.receipt.json')
            session = receipt.with_name(stem+'.session')
            if not session.is_dir():
                continue
            output = receipt.with_name(stem+'.json')
            requests = sorted(session.glob('step_*/REQUEST.json'))
            decisions.append(dict(output=str(output), output_sha256=sha(output), decision=read(output),
                                  receipt=read(receipt),
                                  first_actual_request_messages=read(requests[0])['messages'],
                                  request_hashes={str(p): sha(p) for p in requests}))
        evidence.append(dict(job=job['id'], teacher=teacher,
            negative_evidence=read(selection/'NEGATIVE_EVIDENCE.json'),
            frozen=read(selection/'SELECTION_FROZEN.json'),
            search=read(selection/'SEARCH_STATE.json'),
            global_best=read(base/'versions/GD01'/teacher/'GLOBAL_BEST.json'),
            rounds=[read(p) for p in sorted(base.glob('versions/GD01/*/advice_*/ROUND_COMPLETE.json'))],
            actual_role_calls=decisions))
    sources.extend(read(Path(args.historical)))
    groups, allrows, sa_cache = [], [], {}
    for source in sources:
        folder, teacher = Path(source['folder']), source['teacher']
        assert (folder/'COMPLETE.json').exists(), folder
        rows, ag = read(folder/'records.json'), read(folder/'aggregate.json')
        assert len(rows) == 50 and ag['attempts'] == 50, folder
        terminal = torch.load(io.BytesIO(gzip.decompress((folder/'terminal.pt.gz').read_bytes())), map_location='cpu', weights_only=False)
        assert len(terminal['coords']) == len(rows)
        heads = terminal['affinity']['pic50'].double().reshape(-1).tolist()
        assert len(heads) == len(rows)
        chemicals = read(folder/'CHEMICAL_CONDITION.json') if (folder/'CHEMICAL_CONDITION.json').exists() else None
        condition_file = next(parent/'conditions'/f'{teacher}.json' for parent in folder.parents
                              if (parent/'conditions'/f'{teacher}.json').exists())
        condition = read(condition_file)
        lookup = {a['id']: i for i, a in enumerate(condition['atoms'])}
        reference = np.asarray([condition['reference_coordinates_angstrom'][lookup[a]] for a in condition['controlled_atom_ids']])
        frame = condition['protein_alignment']
        rotation, translation = np.asarray(frame['rotation']), np.asarray(frame['translation_angstrom'])
        group_rows = []
        for i, row in enumerate(rows):
            smi = row.get('smiles')
            score = None
            if row['build_success'] and smi:
                if smi not in sa_cache:
                    molecule = Chem.MolFromSmiles(smi)
                    if molecule is None:
                        raise ValueError('Built molecule SMILES cannot be parsed')
                    sa_cache[smi] = float(sascorer.calculateScore(Chem.RemoveHs(molecule)))
                score = sa_cache[smi]
            chemical = {}
            slots = row.get('condition_generated_slots')
            if chemicals and slots:
                ids = torch.tensor(slots, dtype=torch.long)
                channels = []
                for channel in ('atomics', 'charges', 'hybridization'):
                    if channel in terminal and channel in chemicals['features']:
                        actual = terminal[channel][i, ids].argmax(-1)
                        expected = torch.tensor(chemicals['features'][channel]).argmax(-1)
                        matched = actual == expected
                        chemical[channel+'_fraction'] = float(matched.double().mean())
                        chemical[channel+'_all'] = bool(matched.all())
                        channels.append(bool(matched.all()))
                expected = torch.tensor(chemicals['features']['bonds']).argmax(-1)
                actual = terminal['bonds'][i][ids[:, None], ids[None, :]].argmax(-1)
                offdiag = ~torch.eye(len(ids), dtype=torch.bool)
                chemical['induced_bonds_fraction'] = float((actual[offdiag] == expected[offdiag]).double().mean())
                chemical['induced_bonds_all'] = bool((actual[offdiag] == expected[offdiag]).all())
                positive = torch.triu(expected != 0, diagonal=1)
                chemical['positive_bond_fraction'] = float((actual[positive] == expected[positive]).double().mean()) if bool(positive.any()) else None
                chemical['all_measured_channels_and_induced_bonds'] = all(channels) and chemical['induced_bonds_all']
            rmsd = row['condition_coordinate_root_mean_square_deviation_angstrom']
            selected = terminal['coords'][i, slots].double().numpy() @ rotation + translation
            measured = float(np.sqrt(np.mean(np.sum((selected-reference)**2, axis=-1))))
            assert abs(measured-rmsd) < 1e-4, (folder, i, measured, rmsd)
            left, right = selected-selected.mean(0), reference-reference.mean(0)
            u, _, vt = np.linalg.svd(left.T @ right)
            proper = np.diag([1., 1., np.linalg.det(u @ vt)])
            fitted = left @ u @ proper @ vt
            fitted_rmsd = float(np.sqrt(np.mean(np.sum((fitted-right)**2, axis=-1))))
            group_rows.append(dict(folder=str(folder), teacher=teacher, index=i, node_id=row['node_id'],
                seed=row['logical_seed'], slot=row['logical_slot'], built=row['build_success'], connected=row['connected'],
                rmsd=rmsd, recomputed_absolute_rmsd=measured, condition_rigid_fit_rmsd=fitted_rmsd,
                loss=rmsd*rmsd/2, pic50=heads[i], sa=score, chemistry=chemical,
                initial_assignment_rmsd=row.get('initial_assignment_condition_coordinate_root_mean_square_deviation_angstrom'),
                last_guidance_assignment_rmsd=row.get('last_guidance_assignment_condition_coordinate_root_mean_square_deviation_angstrom')))
        metrics = {key: stats([r[key] for r in group_rows], 'rmsd' if 'rmsd' in key else key)
                   for key in ('rmsd', 'condition_rigid_fit_rmsd', 'pic50', 'sa', 'initial_assignment_rmsd', 'last_guidance_assignment_rmsd', 'loss')}
        chemical_keys = sorted({key for row in group_rows for key in row['chemistry']})
        chemistry = {key:dict(n=sum(key in row['chemistry'] for row in group_rows),
                              mean=statistics.mean(row['chemistry'][key] for row in group_rows if key in row['chemistry'] and row['chemistry'][key] is not None))
                     for key in chemical_keys}
        injection = None
        if folder.is_relative_to(ROOT):
            steps = read(folder/'steps.json')
            assert [s['step'] for s in steps] == list(range(100))
            for step in steps:
                assert all(bool(x) == (1 <= step['step'] <= 98) for x in step['chemical_inpainting_applied'])
                assert not step['affinity_vjp']
                if step['step'] in (0, 99):
                    assert all(x == 0 for x in step['gradient_norm']+step['max_atom_increment_A'])
            injection = {key:statistics.mean(x for step in steps if 1 <= step['step'] <= 98
                         for x in step[f'chemical_{key}_after_match_fraction']) for key in ('atomics', 'charges', 'bonds')}
            assert all(abs(x-1.) < 1e-7 for x in injection.values())
        groups.append(dict(source, population=50, built=sum(r['built'] for r in group_rows),
            connected=sum(r['connected'] for r in group_rows), unique_smiles=ag.get('unique_smiles'),
            schedule=ag['effective_guidance_scale_schedule'], metrics=metrics, chemistry=chemistry,
            quality=quality(folder), ranking=ag['ranking_condition_coordinate_root_mean_square_deviation_angstrom'],
            aggregate_loss=ag.get('coordinate_loss_statistics'),
            maximum_unconstrained_increment=ag.get('maximum_unconstrained_single_atom_guidance_increment_angstrom'),
            stored_assignment_diagnostics=ag.get('assignment_diagnostics_not_used_for_ranking'),
            chemical_after_injection=injection, condition_file_sha256=sha(condition_file),
            ligand_fit_definition='Diagnostic proper rigid fit of same nine terminal-assigned slots; no rematching, scaling or reflection; does not replace ranking.',
            source_sha256={name:sha(folder/name) for name in ('COMPLETE.json','records.json','terminal.pt.gz','aggregate.json')},
            chemical_binding_sha256=sha(folder/'CHEMICAL_CONDITION.json') if chemicals else None))
        allrows.extend(group_rows)
    write(out/'actual_evidence.json', evidence)
    write(out/'final_results.json', dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        commit=plan['commit'], plan_sha256=sha(ROOT/'PLAN.json'), analysis_source_sha256=sha(Path(__file__)),
        torch=torch.__version__, rdkit=rdkit.__version__, sascorer_sha256=sha(Path(sascorer.__file__)),
        sa_fragments_sha256=sha(Path(sascorer.__file__).with_name('fpscores.pkl.gz')),
        rmsd_definition='Saved terminal injective geometric Hungarian of original nine PLIP atoms in protein-registered frame; no ligand fit.',
        pic50_definition='Saved terminal affinity.pic50 on-target head; all 50 terminal rows; no new forward.',
        sa_definition='Standard RDKit Ertl-Schuffenhauer synthetic accessibility; 1 easy to 10 difficult; built SMILES only; not 3D quality.',
        chemistry_definition='Raw uncovered terminal atom/charge/induced-bond argmax at terminal geometric assignment, diagonal excluded; no stereochemistry guarantee.',
        filter_definition='Descriptive per complete50: >=10 finite; RMSD upper Q3+max(3IQR,1A); other metrics outer 3IQR; >10% gives NA; missing/nonfinite never outliers.',
        top10_definition='Best ceil(10% finite) per metric; pIC50 descending, RMSD/SA ascending; do not interpret as same five molecules.',
        evidence_sha256=sha(out/'actual_evidence.json'), groups=groups, new_sampling=0, new_model_forwards=0, llm_calls=0))
    with gzip.open(out/'candidate_rows.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for row in allrows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+'\n')
    print(json.dumps(dict(groups=len(groups), candidate_views=len(allrows), new_full_groups=34,
                          out=str(out), outputs={p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in out.iterdir() if p.is_file()})))


if __name__ == '__main__':
    main()
