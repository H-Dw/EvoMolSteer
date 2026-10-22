"""Analyze retained documents; never generate molecules or infer deleted coordinates."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from agenticgen.adaptive_schedule.io import read, write, eligible, sha
from agenticgen.adaptive_schedule.group_tools import score_summary, paired
from agenticgen.adaptive_schedule.comparisons import shared_batch_bootstrap


def chemistry(records):
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        return {'available': False}
    molecules = {}
    for r in records:
        text = r.get('smiles') or r.get('canonical_smiles') or r.get('diagnostic_sdf_smiles')
        m = Chem.MolFromSmiles(text) if text else None
        if eligible(r) and m is not None:
            molecules[Chem.MolToSmiles(m)] = m
    fps = [rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(m) for m in molecules.values()]
    similarities = [x for i,f in enumerate(fps) for x in DataStructs.BulkTanimotoSimilarity(f,fps[:i])]
    return {'available': True, 'unique_smiles':len(molecules),
            'uniqueness_over_valid':len(molecules)/max(1,sum(eligible(r) for r in records)),
            'diversity_one_minus_mean_tanimoto':1-statistics.fmean(similarities) if similarities else None}


def analyze(root):
    from .adaptive_schedule.protocol_state import load_config
    config, templates = load_config(root), read(root/'teachers.json')
    if config.get('controller',{}).get('protocol') == 'conditional_learning_v1':
        from .condition_reporting import analyze as condition_analyze
        return condition_analyze(root)
    collections, rows, paired_rows, cases, costs = {}, [], [], [], []
    quality_refs, baseline_rows = {}, []
    for template in templates:
        teacher = template['teacher_id']; target = template['target']
        refs = {'R0':read(root/'shared_r0'/teacher/'quality_records.json')}
        for arm in ['steer','unguided']:
            from agenticgen.adaptive_schedule.quality import metrics_folder
            original = read(root/'baselines'/f'{target}__{arm}.json')
            enriched=[]
            for record in original:
                profile=Path(config['history'])/'baseline_profiles_graph_v2'/target/arm/record['node_id']
                quality=metrics_folder(profile,template)
                enriched.append({**record,**quality})
                for name in ['complete.json','interactions.json']:
                    if (profile/name).exists(): quality_refs[str(profile/name)]=sha(profile/name)
            refs[arm]=enriched
            available=[x for x in enriched if x.get('quality_evaluated')]
            baseline_rows.append({'teacher':teacher,'target':target,'arm':arm,**score_summary(enriched),
                'quality_evaluated':len(available),
                'mean_plip_recovery':statistics.fmean(x['plip_recovery'] for x in available) if available else None,
                'posebusters_fraction_all_attempts':sum(x.get('posebusters_pass',False) for x in enriched)/100,
                'severe_clash_fraction_all_attempts':sum(x.get('severe_clash',False) for x in enriched)/100,
                'shared_historical_population':True})
        for version in config['versions']:
            base = root/'versions'/version/teacher
            if not base.exists(): continue
            for r in range(1,config['rounds']+1):
                wave = base/f'round_{r:02d}'/'wave'
                if not (wave/'quality_records.json').exists(): continue
                records = read(wave/'quality_records.json')
                groups = {f'G{g+1}':records[g*100:(g+1)*100] for g in range(4)}
                winner = max(groups,key=lambda g:(score_summary(groups[g])['mean'] or float('-inf'),
                                                  score_summary(groups[g])['max'] or float('-inf')))
                groups['group_selected'] = groups[winner]
                best_path=base/'incumbents'/f'r{r:02d}'/'best.json'
                if best_path.exists(): groups['TOP1']=[x['record'] for x in read(best_path)]
                for label, group in groups.items():
                    key=(version,teacher,r,label);collections[key]=group
                    scored=[x for x in group if eligible(x)]
                    quality=[x for x in group if x.get('quality_evaluated')]
                    row={'version':version,'teacher':teacher,'round':r,'collection':label,
                        'source_group':winner if label=='group_selected' else label,**score_summary(group),
                        'teacher_pic50':template['source_record']['pic50_on_rescore'],
                        'teacher_interactions':len(template.get('interactions',[])),
                        'teacher_fingerprints':len(template.get('fingerprints',[])),
                        'core_atoms':len(template['core_atoms']),
                        'recovery_fraction_all_attempts':sum(x['pic50_on_rescore']>=template['source_record']['pic50_on_rescore'] for x in scored)/100,
                        'exceed_fraction_all_attempts':sum(x['pic50_on_rescore']>template['source_record']['pic50_on_rescore'] for x in scored)/100,
                        'quality_evaluated':len(quality),
                        'posebusters_fraction_all_attempts':sum(x.get('posebusters_pass',False) for x in group)/100,
                        'severe_clash_fraction_all_attempts':sum(x.get('severe_clash',False) for x in group)/100,
                        'mean_plip_recovery':statistics.fmean(x['plip_recovery'] for x in quality) if quality else None,
                        **chemistry(group)}
                    for arm, original in refs.items():
                        b=score_summary(original)
                        for metric in ['mean','median','max','top10_mean']:
                            row[f'{metric}_minus_{arm}']=(row[metric]-b[metric] if row[metric] is not None and b[metric] is not None else None)
                    row['paired_vs_R0']=paired(refs['R0'],group)
                    # Historical Steer resampling breaks slot-level causal pairing.
                    row['historical_pairing']='batch-level descriptive only; cross-hardware slot equivalence not claimed'
                    for i,(a,b) in enumerate(zip(refs['R0'],group)):
                        paired_rows.append({'version':version,'teacher_id':teacher,'round':r,'collection':label,
                            'batch':i//20,'slot':i%20,'delta':b['pic50_on_rescore']-a['pic50_on_rescore'] if eligible(a) and eligible(b) else None})
                    rows.append(row)
                if (wave/'SAMPLING_COMPLETE.json').exists():
                    cost=read(wave/'SAMPLING_COMPLETE.json');costs.append({'version':version,'teacher':teacher,'round':r,**cost})
            best_path=base/'incumbents'/f'r{config["rounds"]:02d}'/'best.json'
            if best_path.exists():
                best=read(best_path);valid=[(i,t) for i,t in enumerate(best) if eligible(t['record'])]
                if valid:
                    high=max(valid,key=lambda x:x[1]['record']['pic50_on_rescore'])
                    improving=sorted(valid,key=lambda x:x[1]['record']['pic50_on_rescore']-(refs['R0'][x[0]].get('pic50_on_rescore') or 0))
                    for kind,(i,t) in [('best',high),('median_response',improving[len(improving)//2])]:
                        cases.append({'version':version,'teacher':teacher,'kind':kind,'slot':i,
                            'record':t['record'],'trial_source_id':t['source_id'],
                            'retained_snapshot':str(best_path.parent),'nodes':t['nodes']})
                failures=[(i,t) for i,t in enumerate(best) if not eligible(t['record']) or not t['record'].get('posebusters_pass',False)]
                if failures:
                    i,t=failures[0];cases.append({'version':version,'teacher':teacher,'kind':'failure_or_quality_conflict','slot':i,'record':t['record'],'nodes':t['nodes']})
    # Provider request IDs deduplicate cached receipts, retaining failed API usage.
    usage={}
    for p in (root/'versions').rglob('*.receipt.json'):
        receipt=read(p)
        key=receipt.get('provider_request_id') or str(p.relative_to(root))
        if receipt.get('usage'):
            usage[key]={'path':str(p.relative_to(root)), 'usage':receipt['usage'],
                        'finish_reason':receipt.get('finish_reason'), 'model':receipt.get('model')}
    unreceipted = [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p),
                   'reason': 'Failed response survives, but token usage receipt is absent; cost is unknown.'}
                  for pattern in ['*.invalid.txt', '*.incomplete_*.txt']
                  for p in (root/'versions').rglob(pattern) if not p.with_suffix('.receipt.json').exists()]
    comparisons=[]
    for (version,teacher,r,label), records in collections.items():
        n=int(version[1:]);previous=collections.get((f'v{n-1}',teacher,r,label))
        if previous is not None:
            comparisons.append({'version':version,'previous':f'v{n-1}','teacher':teacher,'round':r,'collection':label,**paired(previous,records)})
    equal=[]; bootstrap=[]
    for version,r,label in sorted({(x['version'],x['round'],x['collection']) for x in rows}):
        selected=[x for x in rows if (x['version'],x['round'],x['collection'])==(version,r,label)]
        if len(selected)!=len(templates):continue
        equal.append({'version':version,'round':r,'collection':label,
                      **{k:statistics.fmean(x[k] for x in selected) if all(x[k] is not None for x in selected) else None
                         for k in ['mean','max','top10_mean','mean_minus_R0','mean_minus_steer','mean_minus_unguided']}})
        b=[x for x in paired_rows if (x['version'],x['round'],x['collection'])==(version,r,label)]
        bootstrap.append({'version':version,'round':r,'collection':label,
                          **shared_batch_bootstrap(b,[t['teacher_id'] for t in templates],'delta')})
    out=root/'reports';out.mkdir(exist_ok=True)
    for name,value in [('populations',rows),('teacher_equal',equal),('shared_batch_bootstrap',bootstrap),
                       ('successive_versions',comparisons),('cases',cases),('gpu_cost',costs),('llm_usage',usage),
                       ('llm_missing_receipts',unreceipted),
                       ('historical_baselines',baseline_rows),('historical_quality_references',quality_refs)]:
        write(out/(name+'.json'),value)
    plain=[{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in rows]
    if plain:
        with (out/'populations.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(plain[0]));w.writeheader();w.writerows(plain)
    text=['# FLOWR population guidance comparisons','',
          'Mean is primary; Best/Top10 are secondary. Group selection and per-chain TOP1 are separate search outputs.',
          'Shared starts/baselines are not independent samples. Five-batch bootstrap is exploratory.',
          'Teacher-feature associations are descriptive with only two development teachers; no generalization or causal claim.',
          'Raw structures of discarded trials are deleted; their documentary analyses remain available.','',
          f'LLM responses missing token receipts: {len(unreceipted)}. Recorded usage excludes their unknown cost.' if unreceipted else 'No retained failed responses are missing token receipts.', '',
          '| Version | Round | Collection | Equal-weight Mean | Best | Top10 | Mean vs R0 | Mean vs Steer |',
          '|---|---:|---|---:|---:|---:|---:|---:|']
    for x in equal:
        fmt=lambda k:f'{x[k]:.5f}' if x[k] is not None else 'NA'
        text.append(f"|{x['version']}|{x['round']}|{x['collection']}|{fmt('mean')}|{fmt('max')}|{fmt('top10_mean')}|{fmt('mean_minus_R0')}|{fmt('mean_minus_steer')}|")
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return {'populations':len(rows),'versions_present':len({x['version'] for x in rows}), 'output':str(out)}


if __name__=='__main__':
    from agenticgen.adaptive_schedule.group_protocol import GROUP_ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(GROUP_ROOT))
    print(analyze(parser.parse_args().root))
