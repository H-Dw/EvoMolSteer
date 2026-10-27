"""Read-only descriptive analysis of the exported UG01--UG05 pause snapshot."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'docs/evidence/uniguide/20261027_UG01_UG05_pause_results.json.gz'
DEST = BASE / 'docs/reports/20261027_UG_summary'
DEST.mkdir(exist_ok=True)
with gzip.open(SOURCE, 'rt', encoding='utf-8') as f:
    data = json.load(f)

teacher_scores = {x['target']: x['pic50'] for x in data['teachers']}
verified_groups = 0
for group in list(data['r0'].values()) + [g for v in data['versions'].values() for g in v['groups'].values()]:
    a, records = group['aggregate'], group['records']
    target = a['id'].split('/')[1 if a['id'].startswith('R0/') else 2]
    assert len(records) == a['attempts'] == 50
    assert len({(r['logical_seed'], r['logical_slot']) for r in records}) == 50
    assert len({r['node_id'] for r in records}) == 50
    for field, value in {
        'C': st.mean(r['condition_rmsd_A'] for r in records),
        'mean_pic50': st.mean(r['pic50_on_rescore'] for r in records),
        'best': max(r['pic50_on_rescore'] for r in records),
        'gap': abs(st.mean(r['pic50_on_rescore'] for r in records) - teacher_scores[target]),
        'built': sum(bool(r['build_success']) for r in records),
    }.items():
        assert math.isclose(a[field], value, rel_tol=1e-12, abs_tol=1e-12), (a['id'], field)
    selection = group['QUALITY_COMPLETE']['selection']
    expected = {(seed, slot) for seed in [42, 100045, 200048, 300051, 400054] for slot in [0, 1]}
    assert {(r['logical_seed'], r['logical_slot']) for r in selection} == expected
    indexed = {(r['logical_seed'], r['logical_slot']): r['node_id'] for r in records}
    assert all(indexed[(r['logical_seed'], r['logical_slot'])] == r['node_id'] for r in selection)
    verified_groups += 1


def choose(rows, band=.02):
    low = min(x['C'] for x in rows)
    return min((x for x in rows if x['C'] <= low+band), key=lambda x:(x['gap'],x['C'],-x['best'],x['id']))


def avg(values):
    values = list(values)
    return st.mean(values) if values else None


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST / name).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys, lineterminator='\n');w.writeheader();w.writerows(rows)


def quality(groups):
    counts = dict(groups=0, attempts=0, checked=0, not_selected=0, unbuilt=0,
        pb_pass=0, pb_measured_fail=0, pb_unevaluable=0, plip_evaluable=0, plip_unevaluable=0,
        pending_groups=0, engineering_failed_groups=0)
    recoveries=[]
    for g in groups:
        counts['groups']+=1;counts['attempts']+=g['aggregate']['attempts']
        if 'QUALITY_FAILED' in g:
            counts['engineering_failed_groups']+=1;continue
        q=g.get('QUALITY_COMPLETE')
        if not q:
            counts['pending_groups']+=1;continue
        assert q['checked_attempts']==10 and q['not_selected']==40 and q['coverage']==.2
        counts['checked']+=q['checked_attempts'];counts['not_selected']+=q['not_selected']
        counts['unbuilt']+=q['unbuilt_selected'];counts['pb_pass']+=q['posebusters_pass']
        counts['pb_measured_fail']+=q['posebusters_evaluable']-q['posebusters_pass']
        counts['pb_unevaluable']+=q['posebusters_unevaluable_built']
        counts['plip_evaluable']+=q['plip_evaluable'];counts['plip_unevaluable']+=q['plip_unevaluable_built']
        ids={s['node_id'] for s in q['selection']}
        for row in g['quality_rows']:
            if row['node_id'] in ids and row.get('quality',{}).get('plip_evaluable'):
                val=row.get('plip_recovery')
                if val is not None and math.isfinite(val):recoveries.append(val)
    assert sum(counts[k] for k in ['unbuilt','pb_pass','pb_measured_fail','pb_unevaluable']) == counts['checked']
    counts.update(plip_recovery_n=len(recoveries), plip_recovery_mean=avg(recoveries))
    return counts


groups=[];qrows=[];representatives=[];versions=[];llm=[];paired=[];details={};seen={};light=[];retained=[]
for name, v in data['versions'].items():
    rows=[g['aggregate'] for g in v['groups'].values()]
    finished='FINALIZED' in v['markers']
    overall=v['markers'].get('FINALIZED',{}).get('overall',{})
    version=dict(version=name,status='complete' if finished else 'partial',groups=len(rows),attempts=sum(x['attempts'] for x in rows),
        C=overall.get('C'),gap=overall.get('gap'),best=overall.get('best'),built=sum(x['built'] for x in rows))
    versions.append(version);qrows.append(dict(version=name,target='ALL',**quality(v['groups'].values())))
    light.append(dict(version=name, **{k: sum(g['QUALITY_LIGHT_COMPLETE'][k] for g in v['groups'].values())
        for k in ['attempts', 'built', 'valid_connected', 'severe_clash', 'clash_evaluable']}))
    for path, rows_kept in v['retention'].items():
        if path.endswith('/per_slot_top1.json'):
            retained.append(dict(version=name, target=path.split('/')[2], slots=len(rows_kept),
                built_slots=sum(bool(r['build_success']) for r in rows_kept)))
    unique_receipts={}
    for item in v['llm_receipts']:
        receipt=item['receipt'];key=receipt['provider_request_id']
        if key not in unique_receipts or 'receipt_kind' not in receipt:unique_receipts[key]=item
    receipts=[x['receipt'] for x in unique_receipts.values()]
    valid=[x for x in receipts if x.get('backend')=='real_api' and x.get('finish_reason')=='stop']
    failed_length=[x for x in receipts if x.get('finish_reason')=='length']
    invalid=[x for x in receipts if x.get('valid') is False and x.get('finish_reason')=='stop']
    assert len(valid)+len(failed_length)+len(invalid)==len(receipts)
    outputs=[x['usage']['completion_tokens'] for x in receipts]
    inputs=[x['usage']['prompt_tokens'] for x in receipts]
    reasoning=[x['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for x in receipts]
    llm.append(dict(version=name,logical_successes=len(valid),confirmed_http_responses=len(receipts),length_failures=len(failed_length),
        invalid_initial_answers=len(invalid),input_tokens=sum(inputs),output_tokens=sum(outputs),reasoning_tokens=sum(reasoning),
        reasoning_share=sum(reasoning)/sum(outputs),input_median=st.median(inputs),input_max=max(inputs),
        output_median=st.median(outputs),output_max=max(outputs),successful_calls_with_repair=sum(bool(x.get('interface_repairs')) for x in valid)))
    for target in data['r0']:
        selected={key:g for key,g in v['groups'].items() if key.split('/')[2]==target}
        if not selected:continue
        rr=[g['aggregate'] for g in selected.values()]
        winner=choose(rr);base=data['r0'][target]['aggregate']
        if finished:assert v['markers']['FINALIZED']['teachers'][target]['representative']==winner['id']
        rep=dict(version=name,target=target,status='official' if finished else 'partial_candidate',group=winner['id'],
            C=winner['C'],gap=winner['gap'],mean_pic50=winner['mean_pic50'],best=winner['best'],built=winner['built'],unique_smiles=winner['unique_smiles'],
            C_se_descriptive=winner['C_se'],delta_C_R0=winner['C']-base['C'],delta_gap_R0=winner['gap']-base['gap'])
        rep.update({f'W{i+1}':x for i,x in enumerate(winner['schedule'])});representatives.append(rep)
        qrows.append(dict(version=name,target=target,**quality(selected.values())))
        current=selected[winner['id']]['records'];baseline=data['r0'][target]['records']
        b={(x['logical_seed'],x['logical_slot']):x for x in baseline}
        for seed in [42,100045,200048,300051,400054]:
            block=[x for x in current if x['logical_seed']==seed]
            paired.append(dict(version=name,target=target,seed=seed,n=10,
                delta_C_R0=avg(x['condition_rmsd_A']-b[(seed,x['logical_slot'])]['condition_rmsd_A'] for x in block),
                delta_mean_pic50_R0=avg(x['pic50_on_rescore']-b[(seed,x['logical_slot'])]['pic50_on_rescore'] for x in block)))
        counts=seen.setdefault(target,{})
        repeats=0
        for g in rr:
            schedule=tuple(round(2*math.log2(x)) for x in g['schedule'])
            repeats+=int(schedule in counts);counts.setdefault(schedule,[]).append(g)
            row=dict(version=name,target=target,round=int(g['id'].split('/')[-2][1:]),branch=g['id'].split('/')[-1],
                group=g['id'],representative=g['id']==winner['id'],**{k:g[k] for k in ['attempts','C','C_se','mean_pic50','gap','best','built','unique_smiles']})
            row.update({f'W{i+1}':x for i,x in enumerate(g['schedule'])})
            for i,w in enumerate(g['window_diagnostics'],1):
                row.update({f'W{i}_{key}':w[key] for key in ['increment_vector_rms_A','native_increment_ratio_mean','max_atom_increment_A']})
            groups.append(row)
        details[name+'/'+target]=dict(partial=not finished,repeat_schedule_groups_from_earlier_versions=repeats,groups=len(rr),
            min_C=min(x['C'] for x in rr),median_C=st.median(x['C'] for x in rr),max_C=max(x['C'] for x in rr),
            min_gap=min(x['gap'] for x in rr),median_gap=st.median(x['gap'] for x in rr),max_gap=max(x['gap'] for x in rr),
            unique_smiles_union=len({r['smiles'] for g in selected.values() for r in g['records'] if r.get('smiles')}),
            max_window_rms=max(w['increment_vector_rms_A'] for g in rr for w in g['window_diagnostics']),
            max_atom_increment=max(w['max_atom_increment_A'] for g in rr for w in g['window_diagnostics']),
            band_sensitivity={str(band):choose(rr,band)['id'] for band in [.01,.02,.05]})

for target,g in data['r0'].items():
    qrows.append(dict(version='R0',target=target,**quality([g])))
    a=g['aggregate'];representatives.append(dict(version='R0',target=target,status='shared_baseline',group=a['id'],
        **{k:a[k] for k in ['C','gap','mean_pic50','best','built','unique_smiles']},**{f'W{i+1}':x for i,x in enumerate(a['schedule'])}))

repeat_audit={target:dict(unique_schedules=len(values),group_evaluations=sum(map(len,values.values())),
    repeated_schedules=sum(len(x)>1 for x in values.values()),max_C_range_same_lattice=max(max(g['C'] for g in x)-min(g['C'] for g in x) for x in values.values()),
    max_gap_range_same_lattice=max(max(g['gap'] for g in x)-min(g['gap'] for g in x) for x in values.values())) for target,values in seen.items()}
pooled={target:choose([g['aggregate'] for name,v in data['versions'].items() if name!='UG05' for key,g in v['groups'].items() if key.split('/')[2]==target]) for target in data['r0']}
budgets={stage:sum(x['count'] for x in data['attempts'].values() if x['stage']==stage) for stage in ['calibration','r0','formal']}
summary=dict(snapshot_time=data['time_shanghai'],source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    source_hash_scope='gzip archive bytes', source_json_sha256=hashlib.sha256(gzip.decompress(SOURCE.read_bytes())).hexdigest(),
    verification=dict(groups_recomputed_from_all_50_records=verified_groups, sample_seed_slot_node_ids_verified=verified_groups,
        source_file_hashes_in_export=len(data['source_sha256'])),
    budgets=budgets,versions=versions,representatives=representatives,quality=qrows,llm=llm,details=details,
    repeat_audit=repeat_audit,pooled_complete_version_groups=pooled,paired_seed_deltas=paired,
    lightweight_quality=light,retention=retained,
    limitations=['descriptive only; shared seeds and adaptive selection', 'UG05 partial, not ranked as a complete version',
        'fixed slots 0/1 are deterministic diagnostics, not a random population estimate', 'pIC50 is model rescore, not experimental affinity'])
(DEST/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
for name,rows in [('all_groups.csv',groups),('representatives.csv',representatives),('sampled_quality.csv',qrows),('llm_usage.csv',llm),('paired_seed_deltas.csv',paired),('lightweight_quality.csv',light),('retention.csv',retained)]:write_csv(name,rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'UG01':'#4782b4','UG02':'#c3913b','UG03':'#538a66','UG04':'#bb5263','UG05':'#8c76b4'}
fig,axs=plt.subplots(2,3,figsize=(16,9.8),layout='constrained')
complete=[x for x in versions if x['status']=='complete'];labels=['R0']+[x['version'] for x in complete]
base={key:avg(g['aggregate'][key] for g in data['r0'].values()) for key in ['C','gap']}
for ax,metric,title in zip(axs[0,:2],['C','gap'],['Equal-teacher representative C (lower is better)','Equal-teacher representative affinity gap']):
    vals=[base[metric]]+[x[metric] for x in complete]
    ax.plot(labels,vals,'o-',color='#354e68',lw=2)
    for i,value in enumerate(vals):ax.annotate(f'{value:.4f}',(i,value),xytext=(0,9),textcoords='offset points',ha='center')
    ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel('Angstrom' if metric=='C' else 'Absolute mean pIC50 gap');ax.margins(y=.23);ax.grid(axis='y',alpha=.2)
qall=[next(q for q in qrows if q['version']==v['version'] and q['target']=='ALL') for v in versions]
bottom=np.zeros(len(qall))
for key,color,label in [('pb_pass','#538a66','PB pass'),('pb_measured_fail','#cf7a4c','Measured fail'),('pb_unevaluable','#b1a2c7','Unknown'),('unbuilt','#b9bec6','Unbuilt')]:
    values=np.array([q[key] for q in qall]);axs[0,2].bar([q['version'] for q in qall],values,bottom=bottom,color=color,label=label);bottom+=values
for i,q in enumerate(qall):axs[0,2].text(i,bottom[i]+4,f'{q["pb_pass"]}/{q["checked"]}',ha='center',fontsize=9)
axs[0,2].set_title('Fixed 10/50 diagnostic slots only',loc='left',fontweight='bold');axs[0,2].set_ylabel('Selected attempts (not population pass rate)');axs[0,2].set_ylim(0,350);axs[0,2].legend(fontsize=8,ncol=2,loc='upper right')
for ax,target in zip(axs[1,:2],['6pyy_H7S_A_402','3w54_RNB_A_504']):
    for name in colors:
        rows=[g for g in groups if g['target']==target and g['version']==name]
        if not rows:continue
        ax.scatter([g['C'] for g in rows],[g['gap'] for g in rows],s=26,color=colors[name],alpha=.65,label=name+(' partial' if name=='UG05' else ''))
        rep=next(x for x in representatives if x['target']==target and x['version']==name)
        ax.scatter(rep['C'],rep['gap'],marker='*',s=190,color=colors[name],edgecolor='black',linewidth=.6)
    r=data['r0'][target]['aggregate'];ax.scatter(r['C'],r['gap'],marker='D',s=60,color='black',label='R0')
    ax.set_title(target.split('_')[0]+': all measured groups; stars = representatives',loc='left',fontweight='bold')
    ax.set_xlabel('C (Angstrom; logarithmic scale)');ax.set_ylabel('Absolute mean pIC50 gap');ax.set_xscale('log');ax.grid(alpha=.18);ax.legend(fontsize=8,ncol=2)
tokens=[]
for name,v in data['versions'].items():
    unique={x['receipt']['provider_request_id']:x['receipt'] for x in v['llm_receipts']}
    tokens.append([x['usage']['completion_tokens'] for x in unique.values()])
axs[1,2].boxplot(tokens,tick_labels=list(data['versions']),showfliers=True)
axs[1,2].axhline(16384,color='#b54a4a',ls='--',lw=1,label='16,384 cap (UG04 one-call exception)')
axs[1,2].set_title('Output tokens per confirmed HTTP response',loc='left',fontweight='bold');axs[1,2].set_ylabel('Reasoning + returned answer tokens');axs[1,2].legend(fontsize=8,loc='lower right');axs[1,2].grid(axis='y',alpha=.2)
fig.suptitle('UniGuide–FLOWR | Paused 27 Oct 2026 | UG01–04 complete, UG05 partial',fontsize=18,fontweight='bold')
fig.savefig(DEST/'overview.png',dpi=180);fig.savefig(DEST/'overview.pdf');plt.close(fig)
print(json.dumps({k:summary[k] for k in ['budgets','versions','representatives','quality','llm','details','repeat_audit']},ensure_ascii=True))
