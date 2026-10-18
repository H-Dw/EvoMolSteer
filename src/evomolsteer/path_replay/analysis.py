"""Distribution summaries, fixed historical masks and source-batch paired CIs."""
import csv
from pathlib import Path

import numpy as np

from evomolsteer.io import read_json
from evomolsteer.storage.transactions import atomic_json
from .common import groups_for,read_batch


def distribution(records,planned,threshold):
    good=[r for r in records if r.get('pic50') is not None and np.isfinite(r['pic50'])]
    scores=np.array([r['pic50'] for r in good],dtype=float)
    valid=[r['pic50'] for r in good if r.get('valid_connected')]
    pb=[r['pic50'] for r in good if r.get('pb_pass')]
    energy=[r['mmff_relief_per_heavy'] for r in good if r.get('mmff_relief_per_heavy') is not None and np.isfinite(r['mmff_relief_per_heavy'])]
    out={'n_planned':planned,'n_records':len(records),'n_scored':len(good),'score_coverage':len(good)/planned,
         'valid_count':len(valid),'pb_pass_count':len(pb),'energy_count':len(energy),
         'best_valid':max(valid) if valid else None,'best_pb':max(pb) if pb else None,
         'strain_median':float(np.median(energy)) if energy else None,
         'strain_p90':float(np.quantile(energy,.9)) if energy else None,
         'high_score_count':int((scores>=threshold).sum()),'high_score_fraction_of_planned':float((scores>=threshold).sum()/planned)}
    for name,q in [('median',.5),('p90',.9),('p95',.95),('p99',.99)]: out[name]=float(np.quantile(scores,q)) if len(scores) else None
    out.update(mean=float(scores.mean()) if len(scores) else None,best=float(scores.max()) if len(scores) else None)
    return out


def paired_cluster(candidate,baseline,replicates=10000,seed=42):
    other={r['id']:r for r in baseline if r.get('pic50') is not None}
    pairs=[(r,other[r['id']]) for r in candidate if r.get('pic50') is not None and r['id'] in other]
    if not pairs: return {'n_paired':0}
    delta=np.array([a['pic50']-b['pic50'] for a,b in pairs]);batch=np.array([a['batch'] for a,b in pairs])
    keys=np.unique(batch);sums=np.array([delta[batch==k].sum() for k in keys]);counts=np.array([(batch==k).sum() for k in keys])
    result={'n_paired':len(pairs),'source_batches':len(keys),'mean_delta':float(delta.mean()),
            'fraction_improved':float((delta>0).mean()),'median_delta':float(np.median(delta))}
    if replicates and len(keys)>1:
        draws=np.random.default_rng(seed).integers(0,len(keys),(replicates,len(keys)))
        means=sums[draws].sum(1)/counts[draws].sum(1)
        result.update(mean_delta_ci_low=float(np.quantile(means,.025)),mean_delta_ci_high=float(np.quantile(means,.975)))
    return result


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def analyze(repo,cfg,work,output=None,bootstrap=True):
    repo=Path(repo);work=Path(work);out=Path(output) if output else work/'analysis';out.mkdir(parents=True,exist_ok=True)
    datasets={};performance=[];comparisons=[];attempts={a['id']:a for a in cfg['attempts']}
    for c in cfg['conditions']:
        source=next((x for x in cfg['conditions'] if x['id']==c.get('alias_of')),c);records=[]
        for g in groups_for(cfg,source):
            r=read_batch(work,cfg,source,g)
            if r: records += [dict(attempts[v['id']],**{k:x for k,x in v.items() if k not in attempts[v['id']]}) for v in r['records']]
        datasets[c['id']]=records
    historical=[dict(a,pic50=a['historical_pic50'],valid_connected=a['historical_valid_connected']) for a in cfg['attempts']]
    datasets['original_steer']=historical
    native=repo/'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/unguided'
    if native.is_dir():
        rows=[r for p in sorted(native.glob('batch_*/final_records.json')) for r in read_json(p)]
        if rows: datasets['historical_native_unpaired']=[{'id':f"native_b{r['batch']:03d}_s{r['slot']:03d}",'batch':r['batch'],
                     'pic50':r['pic50_on_rescore'],'valid_connected':bool(r.get('build_success') and r.get('connected'))} for r in rows]
    for name,records in datasets.items():
        c=next((x for x in cfg['conditions'] if x['id']==name),None)
        n=c['n'] if c else 1000
        cohorts=[('all1000' if n==1000 else 'assigned_top20',n)]
        if name!='historical_native_unpaired': cohorts += [('top05',50),('top10',100)]+([('top20',200)] if n==1000 else [])
        for label,size in cohorts:
            selected=records if label in ('all1000','assigned_top20') else [r for r in records if r['rank']<=size]
            performance.append({'condition':name,'cohort':label,'alias_of':c.get('alias_of') if c else None,
                                **distribution(selected,size,cfg['tail_threshold'])})
            if c and name!='unguided':
                for reference in ('unguided','M2','original_steer'):
                    if name==reference: continue
                    ref=datasets[reference]
                    if label not in ('all1000','assigned_top20'): ref=[r for r in ref if r['rank']<=size]
                    comparisons.append({'condition':name,'reference':reference,'cohort':label,
                        **paired_cluster(selected,ref,cfg['bootstrap_replicates'] if bootstrap else 0,cfg['bootstrap_seed'])})
    write_csv(out/'performance.csv',performance);write_csv(out/'paired.csv',comparisons)
    atomic_json(out/'summary.json',{'protocol_sha256':cfg['protocol_sha256'],'performance':performance,'paired':comparisons,
        'bootstrap_cluster':'original source batch, not execution group or cloned root','bootstrap_replicates':cfg['bootstrap_replicates'] if bootstrap else 0})
    lines=['# B50历史路径对齐实验','',
        '原Steer全部1000条终态路径均被保留；Top5/10/20按原始终态分数冻结。200例条件仅覆盖Top20%。',
        '无引导路径回放是已知历史路径条件下的对照。原生无引导历史池另列，不能据此宣称随机起点泛化。',
        'R15的两个历史名称共享数值等价结果，不是独立重复。缺失评分不以零填充；最高分另报有效结构与PB通过子集。','',
        '|条件|范围|评分/计划|均值|P95|P99|Best|有效Best|高分数|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def f(v): return '—' if v is None else f'{v:.6f}'
    for r in performance:
        lines.append(f"|{r['condition']}|{r['cohort']}|{r['n_scored']}/{r['n_planned']}|{f(r['mean'])}|{f(r['p95'])}|{f(r['p99'])}|{f(r['best'])}|{f(r['best_valid'])}|{r['high_score_count']}|")
    lines += ['', '配对效应及来源批次聚类区间见 paired.csv；结构、PB、应变覆盖与分位数见 performance.csv。',
              '未完成条件属于进行中结果；不能将部分批次的最高值当作完整N=1000性能。']
    (out/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'performance_rows':len(performance),'paired_rows':len(comparisons),'output':str(out)}
