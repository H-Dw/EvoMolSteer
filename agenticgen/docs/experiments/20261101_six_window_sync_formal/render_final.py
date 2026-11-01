"""Render audited readings; prose interpretation is written separately by GPT."""
import csv,gzip,hashlib,json,statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
RAW=HERE/'monitoring/20261102_0134/final_results_raw.json'
def dump(p,v):p.write_bytes((json.dumps(v,ensure_ascii=True,indent=2)+'\n').encode())
def f(v):return '—' if v is None else f'{v:.6f}'
def quartet(s):return ' / '.join(f(s[k]) for k in ('mean','best','filtered_mean','top10_mean'))
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rs]])
def main():
 if not RAW.exists():RAW.write_bytes((HERE/'final_results.json').read_bytes())
 d=json.loads(RAW.read_text());groups={(g['teacher'],g['id']):g for g in d['groups']};summary=[];chemical=[];actions=[];rounds=[];pairings=[]
 with gzip.open(HERE/'monitoring/20261102_0134/candidate_metrics.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
 physical={(r['teacher'],r['source_shard'],r['source_tensor_row']) for r in rows}
 assert len(physical)==2420
 for t in d['teachers']:
  name=t['teacher'].split('_')[0];baseline=groups[t['teacher'],t['baseline']];best=groups[t['teacher'],t['best']]
  for label,g in [('标定默认最优',baseline),('五轮全局最优',best)]:
   summary.append([name,label,str(g['schedule']),g['population'],g['built'],g['metrics']['sa']['n'],*[quartet(g['metrics'][m]) for m in ('rmsd','pic50','sa')]])
   c=g['terminal_chemical_matches'];ad=g['assignment_diagnostics']
   chemical.append([name,label,f(ad['initial_assignment_condition_coordinate_root_mean_square_deviation_angstrom']),
    f(ad['last_guidance_assignment_condition_coordinate_root_mean_square_deviation_angstrom']),f(g['metrics']['rmsd']['mean']),
    *[f(c[k]) for k in ('atomics_atom_fraction','charges_atom_fraction','positive_bond_fraction','all_induced_pair_fraction')]])
  a={(r['seed'],r['slot']):r for r in rows if r['teacher']==t['teacher'] and r['group']==t['baseline']}
  b={(r['seed'],r['slot']):r for r in rows if r['teacher']==t['teacher'] and r['group']==t['best']}
  assert len(a)==len(b)==50 and a.keys()==b.keys()
  for m in ('rmsd','pic50','sa'):
   deltas=[b[k][m]-a[k][m] for k in a if b[k][m] is not None and a[k][m] is not None]
   pairings.append(dict(teacher=t['teacher'],metric=m,finite_pairs=len(deltas),mean_change=statistics.mean(deltas),
    improved=sum(x>0 if m=='pic50' else x<0 for x in deltas),worse=sum(x<0 if m=='pic50' else x>0 for x in deltas),unchanged=sum(x==0 for x in deltas)))
  for r in t['rounds']:
   selected=r['synthesis']['selected'];props={p['id']:p for p in r['proposal']['proposals']}
   rr=[]
   for sid,gid in zip(selected,r['groups']):
    g=groups[t['teacher'],gid];p=props[sid]
    z=dict(teacher=name,round=r['iteration'],action=sid,scale=str(g['schedule']),tier=p['exploration_tier'],anchor=p['anchor_id'],
     raw_rmsd=g['metrics']['rmsd']['mean'],rank_rmsd=g['metrics']['rmsd']['filtered_mean'],mean_pic50=g['metrics']['pic50']['mean'],
     mean_sa=g['metrics']['sa']['mean'],sa_n=g['metrics']['sa']['n'],built=g['built'])
    actions.append(z);rr.append(z)
   rounds.append([name,r['iteration'],r['proposal_context']['checkpoint_learning']['rebase_from_best_checkpoint'],
    f(r['checkpoint']['performance']),f(r['checkpoint']['best_compatible_performance']),r['checkpoint']['improved_global_best'],
    '；'.join(z['scale'] for z in rr)])
 headers=['教师','节点','实际scale（六窗）','人口','构建','SA有效','RMSD Å：均值/最优/剔异常均值/前10%','预测pIC50：同序四统计','标准SA：同序四统计']
 text=['# 六窗口同步化学条件：最终机器读数','',table(headers,summary),'',
  'RMSD/SA取低值为优，pIC50取高值为优。前10%为各指标独立选择的ceil(10%×有限值数)；这四节点均取5个。RMSD仅蛋白对齐，未拟合配体；pIC50直接读取保存终态头输出。SA使用RDKit '+d['rdkit']+'，仅成功构建SMILES有定义。','',
  '异常规则按完整配置分别处理：RMSD高侧Q3+max(3IQR,1Å)，pIC50/SA双侧3IQR，至少10个有限值且排除不超过10%；缺测和构建失败不作为统计异常。6pyy标定pIC50剔3个高值，其余所列节点/指标无异常剔除。','',
  table(['教师','轮次','实际prompt回到最优','本轮新人口最优 Å','累计最优 Å','严格改善','本轮三项实际scale'],rounds),'',
  table(['教师','节点','初始对应误差 Å','最后引导对应误差 Å','终态动态覆盖 Å','终态原子类型匹配','终态电荷匹配','终态正键匹配','终态全部诱导对匹配'],chemical),'',
  '所有完整50人口组的1–98步注入后原子类型/电荷/条件内部键匹配均为1；0/99不注入、不做额外梯度。上表终态来自未覆盖预测，不能用注入后100%替代；动态几何对应也不保证原化学身份。','',
  table(['教师','正式构建/750','实际抽查/750','抽中构建失败','PB通过/抽中','PB可评价','PB未通过（可评价）','严重冲突/可评价','PLIP可评价'],
    [[t['teacher'].split('_')[0],t['formal_built'],t['quality_totals']['checked_attempts'],t['quality_totals']['unbuilt_selected'],
      str(t['quality_totals']['posebusters_pass'])+'/'+str(t['quality_totals']['checked_attempts']),t['quality_totals']['posebusters_evaluable'],
      t['quality_totals']['posebusters_evaluable']-t['quality_totals']['posebusters_pass'],
      str(t['quality_totals']['severe_clash'])+'/'+str(t['quality_totals']['clash_evaluable']),t['quality_totals']['plip_evaluable']] for t in d['teachers']]),'',
  'PB是已有PoseBusters '+', '.join(sorted({v for g in d['groups'] for v in g.get('posebusters_versions',[])}))+' 检查。每组只查五seed各slot0/1，共10/50；抽样结果不外推全人口。两个最优组均PB10/10；其PLIP指纹平均恢复分别0.90/0.85。标定默认组未做这些慢检查，不能补成0或当通过。','',
  table(['教师','键长通过/可评价','键角通过/可评价','内部冲突通过/可评价','内部能量通过/可评价','蛋白最近距离通过/可评价','蛋白体积重叠通过/可评价'],
   [[t['teacher'].split('_')[0],*[str(sum(g['posebusters_components'][k]['passed'] for g in d['groups'] if g['teacher']==t['teacher'] and g['stage']=='formal'))+'/'+str(sum(g['posebusters_components'][k]['evaluable'] for g in d['groups'] if g['teacher']==t['teacher'] and g['stage']=='formal')) for k in ('bond_lengths','bond_angles','internal_steric_clash','internal_energy','minimum_distance_to_protein','volume_overlap_with_protein')]] for t in d['teachers']]),'',
  '全部30正式动作及全部标定配置读数另见CSV和JSON；2900个保存视图含10→20→50复用，物理候选仅2420，不能把视图人口重复算入预算。']
 (HERE/'final_metrics.zh-CN.md').write_bytes(('\n'.join(text)+'\n').encode())
 with (HERE/'formal_actions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(actions[0]),lineterminator='\n');w.writeheader();w.writerows(actions)
 cal=[{k:g[k] for k in ('teacher','id','population','schedule','built')}|dict(raw_rmsd=g['metrics']['rmsd']['mean'],rank_rmsd=g['metrics']['rmsd']['filtered_mean']) for g in d['groups'] if g['stage']=='calibration']
 with (HERE/'calibration_readings.csv').open('w',encoding='utf-8-sig',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(cal[0]),lineterminator='\n');w.writeheader();w.writerows(cal)
 dump(HERE/'paired_comparison.json',pairings)
 raw_sha=hashlib.sha256(RAW.read_bytes()).hexdigest();compact={k:v for k,v in d.items() if k not in ('sources','teachers')}
 compact.update(raw_review_sha256=raw_sha,raw_review_remote='/workspace/uniguide_audits/six-window-sync-final-review-20261102_0134/final_results.json',physical_unique_candidates=2420,candidate_views=len(rows),teachers=[])
 for t in d['teachers']:
  z={k:v for k,v in t.items() if k not in ('rounds','best_state')};z['global_best']={k:t['best_state'][k] for k in ('representative','performance','change_from_calibration','best_path_sha256')}
  z['rounds']=[dict(iteration=r['iteration'],checkpoint=r['checkpoint'],groups=r['groups'],selected=r['synthesis']['selected'],
   rebase=r['proposal_context']['checkpoint_learning']['rebase_from_best_checkpoint'],path_sha256=r['proposal_context']['checkpoint_learning']['best_path_sha256']) for r in t['rounds']]
  compact['teachers'].append(z)
 dump(HERE/'final_results.json',compact)
 print(json.dumps(dict(raw_sha256=raw_sha,physical_candidates=len(physical),candidate_views=len(rows),formal_actions=len(actions),calibration_views=len(cal)),ensure_ascii=True))
if __name__=='__main__':main()
