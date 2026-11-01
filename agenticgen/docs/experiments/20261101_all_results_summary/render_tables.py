"""Render keyword-named, denominator-explicit comparison tables from audited data."""
from pathlib import Path
import json,csv,hashlib,datetime
B=Path(__file__).resolve().parent
X=json.loads((B/'raw/all_results_deduplicated.json').read_text(encoding='utf-8'))
PHASE={'历史参考':0,'初始对照':1,'固定预标定尺度':2,'固定对照实验':3,'正式探索':4,'标定预实验':5,'范围取证':6}
E=sorted(X['experiments'],key=lambda e:(PHASE[e['phase']],e['experiment'],e['teacher']))
def f(v):return '—'if v is None else f'{v:.4f}'
def t(v):return '6pyy靶蛋白'if v.startswith('6pyy')else'3w54靶蛋白'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |'for row in rows])+'\n'
def stat_cells(e,k):return [f(e['stats'][k][n])for n in ['mean','best','filtered_mean','top10_mean']]
def main_rows(items):
 return [[e['experiment'],t(e['teacher']),e['phase'],e['unique_saved_candidates'],e['stats']['sa']['n'],*stat_cells(e,'rmsd'),*stat_cells(e,'pic50'),*stat_cells(e,'sa')]for e in items]
HEAD=['实验关键词','教师','阶段','终态数','SA有效数','RMSD均值↓','RMSD最优↓','RMSD剔异常均值↓','RMSD前10%↓','pIC50均值↑','pIC50最优↑','pIC50剔异常均值↑','pIC50前10%↑','SA均值↓','SA最优↓','SA剔异常均值↓','SA前10%↓']
defs=[
 ['汇总范围','当前28个可见实验根；656个完整科学配置组，另列两组历史无引导配对参考。工程夹具不计科学结果。'],
 ['统计对象','每实验／教师／阶段汇总所有已保存的唯一物理测量槽位；不是只选择最优checkpoint。不同配置共用随机初态，不等于独立验证。'],
 ['RMSD','Å；统一九个原PLIP条件重原子，终态几何Hungarian覆盖误差；无配体刚体拟合。原实验自身的冻结对应或内部距离指标保留在补充表，不改历史排名。'],
 ['pIC50','对靶蛋白的保存终态输出头预测，所有终态均读取；不是实测活性。原生终态与未覆盖终态的网络时间不同，未新增模型前向统一重评分。历史重评分另表。'],
 ['SA','标准合成可及性，1易合成、10难合成；不代表三维构象合理性。[RDKit官方实现](https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py)。构象合理性另列已有PoseBusters检查。'],
 ['均值／最优','RMSD及SA越低越好，pIC50越高越好；各指标最优值可来自不同候选。SA仅在可构建且可评分分子上有定义，不把失败分子填成0。'],
 ['前10%','每个指标独立选取最优ceil(有效值数×10%)，从未剔异常的有限原始值计算。'],
 ['剔异常均值','每完整配置组至少10个观测；RMSD仅剔高于Q3+max(3IQR,1Å)；pIC50和SA采用双侧3IQR外栅栏。任一组异常比例超过10%时不输出该汇总的剔异常均值；原始均值、最优和前10%仍展示。'],
 ['分母与缺失','主表终态数不等于预留尝试预算；无终态的失败和未完成任务不伪装为成功。缺失／NaN／Inf不作为可剔异常；有效数、剔除数详见CSV。SA缺失分母与其他指标分开。'],
 ['构象检查','只汇总已有PoseBusters抽样，未把每组10/50检查外推为50/50；本次不新增剩余40个慢检查。'],
 ['比较限制','不同实验的输出策略、探索参数范围、候选数和匹配策略不同；应结合逐配置表进行配对，不能把样本量不同的最优值直接当作策略胜负。'],
 ['重复测量','推广视图按source_shard/source_row去重；同物理测量跨标定／范围取证／正式引用只计原始阶段。额外移除30个跨阶段重复槽位。'],
 ['新增计算','只读取既有终态与结构，新增标准SA和统一RMSD后处理；生成0、Agent0、模型前向0。RDKit '+X['rdkit']+'。'],
]
receipt=json.loads((B/'archive_receipt.json').read_text())
archive=table(['压缩包项目','结果'],[['状态','后台重建、全部成员校验完成；仅保留工作盘'],['路径',receipt['archive']],['大小',f"{receipt['bytes']}字节（约{receipt['bytes']/1e9:.2f}GB）"],['文件数',receipt['files']],['SHA256',receipt['sha256']],['权重／移动','不含模型权重；未移动到private_data']])
text=archive+'\n'+table(['统计口径','说明'],defs)+'\n'+table(HEAD,main_rows(E))
(B/'全部实验指标表.md').write_text(text,encoding='utf-8',newline='\n')
main=[e for e in E if e['phase']not in ['标定预实验','范围取证']]
(B/'正式与对照指标表.md').write_text(table(HEAD,main_rows(main)),encoding='utf-8',newline='\n')
with (B/'全部实验指标表.csv').open('w',encoding='utf-8-sig',newline='')as out:
 writer=csv.writer(out,lineterminator='\n');headers=['实验关键词','教师','阶段','唯一终态数','构建数','连通数','配置组数']
 for metric in ['rmsd','pic50','sa','rescore','native_rmsd']:headers +=[metric+'_'+v for v in ['n','missing','mean','best','filtered_mean','top10_mean','top10_n','excluded','filter_eligible']]
 writer.writerow(headers)
 for e in E:
  row=[e['experiment'],e['teacher'],e['phase'],e['unique_saved_candidates'],e['built'],e['connected'],e['groups']]
  for metric in ['rmsd','pic50','sa','rescore','native_rmsd']:row +=[e['stats'][metric].get(v)for v in ['n','missing','mean','best','filtered_mean','top10_mean','top10_n','excluded','filter_eligible']]
  writer.writerow(row)
with (B/'逐配置指标表.csv').open('w',encoding='utf-8-sig',newline='')as out:
 w=csv.writer(out,lineterminator='\n');h=['实验关键词','教师','阶段','样本数','构建数','尺度','来源路径','records_sha256','terminal_sha256']
 for metric in ['rmsd','pic50','sa','rescore','native_rmsd']:h +=[metric+'_'+v for v in ['n','mean','best','filtered_mean','top10_mean','excluded']]
 w.writerow(h)
 for g in X['groups']:
  row=[g['experiment'],g['teacher'],g['phase'],g['population'],g['built'],json.dumps(g['schedule']),g['folder'],g['record_sha256'],g['terminal_sha256']]
  for metric in ['rmsd','pic50','sa','rescore','native_rmsd']:row +=[g['stats'][metric].get(v)for v in ['n','mean','best','filtered_mean','top10_mean','excluded']]
  w.writerow(row)
supp=table(['口径','说明'],[['原记录RMSD','仅列保存了坐标RMSD的实验；内部距离误差没有冒充坐标RMSD。'],['历史重评分pIC50','原pic50_on_rescore字段；近期未存此字段的实验留空，不用终态输出头代填。']])
supp+=table(['实验关键词','教师','阶段','原RMSD有效数','原RMSD均值','原RMSD最优','原RMSD剔异常均值','原RMSD前10%','重评分有效数','重评分均值','重评分最优','重评分剔异常均值','重评分前10%'],[[e['experiment'],t(e['teacher']),e['phase'],e['stats']['native_rmsd']['n'],*stat_cells(e,'native_rmsd'),e['stats']['rescore']['n'],*stat_cells(e,'rescore')]for e in E])
(B/'原口径与重评分补充表.md').write_text(supp,encoding='utf-8',newline='\n')
qr=[]
for e in E:
 q=e['posebusters'];cells=[]
 for k in ['bond_lengths','bond_angles','internal_steric_clash','internal_energy','minimum_distance_to_protein']:
  v=q['components'].get(k,{});cells.append(str(v.get('passed',0))+'/'+str(v.get('evaluable',0)))
 qr.append([e['experiment'],t(e['teacher']),e['phase'],e['unique_saved_candidates'],e['built'],e['connected'],q['checked'],str(q['passed'])+'/'+str(q['evaluable']),*cells])
(B/'构象合理性抽样表.md').write_text(table(['实验关键词','教师','阶段','终态数','构建数','连通数','已检查数','整体通过/可评','键长','键角','内部冲突','内部能量','蛋白距离'],qr),encoding='utf-8',newline='\n')
summary={k:v for k,v in X.items()if k not in ['groups','experiments']};summary.update(experiment_rows=len(E),configuration_rows=len(X['groups']),model_forwards=0)
summary['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in B.iterdir()if p.suffix in ['.csv','.md']}
(B/'table_provenance.json').write_text(json.dumps(summary,ensure_ascii=True,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(experiment_rows=len(E),main_rows=len(main),configuration_rows=len(X['groups']),output=str(B)),ensure_ascii=True))
