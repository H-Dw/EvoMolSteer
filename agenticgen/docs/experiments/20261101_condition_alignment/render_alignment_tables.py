"""Publish keyword-named alignment tables without changing previous rankings."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent


def write(path, text):
    path.write_bytes(text.encode('utf-8'))


def number(value):
    return 'NA' if value is None else f'{value:.4f}'


def table(rows, path, best=False):
    headers = ['实验关键词','教师','阶段／原最优轮','有效数','对齐前均值 Å',
               '对齐后均值 Å','对齐后最优 Å','剔异常后均值 Å','对齐后前10%均值 Å','排除数']
    lines = ['| '+' | '.join(headers)+' |', '|'+ '|'.join(['---']*len(headers))+'|']
    flattened = []
    for row in rows:
        stat, raw = row['stats'], row['raw_stats']
        stage = f"第{row['selected_round']}轮／{row['suite']}" if best else row.get('phase','固定预标定尺度')
        values = [row['experiment'],row['teacher'].split('_')[0],stage,str(stat['n']),
                  number(raw['mean']),number(stat['mean']),number(stat['best']),
                  number(stat['filtered_mean']),number(stat['top10_mean']),str(stat['excluded'])]
        lines.append('| '+' | '.join(values)+' |')
        flattened.append(dict(experiment=row['experiment'], teacher=row['teacher'], phase=stage,
                              population=row['population'], raw_mean=raw['mean'],
                              **{'aligned_'+k:v for k,v in stat.items()}, folder=row.get('folder','')))
    write(path, '\n'.join(lines)+'\n')
    csv_path = path.with_suffix('.csv')
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(flattened)


def main():
    data = json.loads((HERE/'raw/alignment_results.json').read_text(encoding='utf-8'))
    rows = sorted(data['experiments'], key=lambda e:(e['experiment'],e['teacher'],e['phase']))
    table(rows, HERE/'全部实验condition对齐RMSD.md')
    table(data['paired_fixed_scale'], HERE/'梯度范围condition对齐RMSD.md')
    table(data['completed_job_best_configurations'], HERE/'完整五轮任务condition对齐RMSD.md', best=True)
    groups = [dict(g, phase=g['phase']) for g in data['groups']]
    table(groups, HERE/'逐配置condition对齐RMSD.md')
    (HERE/'alignment_summary.json').write_bytes((json.dumps(
        {k:v for k,v in data.items() if k!='groups'}, ensure_ascii=True,indent=2,
        allow_nan=False)+'\n').encode('utf-8'))
    (HERE/'validation.json').write_bytes((json.dumps(data['validation'], ensure_ascii=True,
                                                   indent=2)+'\n').encode('utf-8'))
    plan = '''| 项目 | 定义与实际执行 |
|---|---|
| 用户要求 | 对先前结果使用condition区域重原子做alignment，计算对齐后RMSD。 |
| 对应关系 | 沿用上一份共同口径：在原生成／蛋白坐标系对原九个PLIP重原子做单射几何Hungarian；确定对应后冻结，不在拟合后重新挑选或重新匹配。不是固定化学身份匹配。 |
| 拟合 | 每候选使用这同九个对应点、等权重、float64、正旋转Kabsch：最小化mean squared coordinate residual；允许刚体平移和旋转，det(R)=+1，不允许镜像、缩放、柔性变形。 |
| RMSD | 在用于拟合的同九个点上计算sqrt(mean(sum((P R+t-Q)^2)))，单位Å；原始蛋白坐标系RMSD另列，不被覆盖。 |
| 解释 | 对齐后RMSD描述消除整体平移／旋转后的条件区域形状误差，不能代表蛋白口袋中的姿态保持，也不能证明化学身份保持。 |
| 范围 | 上份汇总656个完整科学配置视图、另加两个历史无引导参考，共658组；按上一份物理源／阶段去重，19135个保存候选、79条实验教师阶段汇总。缺失历史根或不完整终态不伪造。 |
| 最优轮 | 六项已完成五轮任务沿用旧排名选中的轮次／配置，只给这些配置补算对齐RMSD，不按新指标重新选赢家。原未完成两任务保留在全结果表，未冒充完成任务。 |
| 统计 | 均值、最优（最低）、剔异常均值、最优ceil(10%有效数)均值。对齐值重新逐配置计算上侧Q3+max(3IQR,1Å)，至少10有限观测、最多10%排除；超过则滤后均值NA。低值、缺失／非有限值不被当作高侧异常删掉。 |
| 隔离 | 原实验只读；CPU计算，隐藏GPU、CPU库单线程，不加载生成模型、无新采样、无LLM，不重写原结构、终态、排名、pIC50或SA。保存每候选R/t、匹配槽位及对齐前后九个点，可复现或将同一刚体变换应用到完整结构。 |
| 验证 | 正旋转／平移恢复、禁止镜像和缩放；每候选对齐值不大于相同对应的未对齐值；每配置首候选和历史参考以SciPy Rotation.align_vectors独立核查；读取前后逐源size/mtime/SHA一致。 |
| 归档 | 新后处理独立保存，不修改此前已验证的大tar.gz，不恢复定时任务、不部署冻结执行目录。 |
'''
    write(HERE/'plan.zh-CN.md', plan)
    report = '''使用原九个condition重原子的既定几何对应做正旋转Kabsch拟合，已补算所有可读取科学终态的对齐RMSD。

对齐只移除条件片段的整体平移与旋转。它衡量片段形状，对蛋白口袋内的相对位置不再敏感；因此应与未对齐RMSD同时阅读。几何匹配也不保证原子类别或化学身份相同。

六个完整任务继续比较原先按运行时排名选出的最优轮配置；新对齐指标没有用于重新筛选配置。完整汇总保留标定、范围取证、正式和未完成任务已保存的实际结果。

所有计算在CPU上读取既有坐标，没有新生成、模型前向、LLM或质量慢查。旧结果、排名、亲和力和SA没有改变。逐候选匹配槽位、正旋转矩阵、平移及九点对齐坐标保存为机器可读压缩JSONL；所有输入与终态文件读取前后哈希核验一致。
'''
    write(HERE/'说明.zh-CN.md', report)
    manifest = {}
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ['delivery_manifest.json','.gitignore']:
            manifest[p.name] = dict(bytes=p.stat().st_size,
                                     sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (HERE/'delivery_manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=True,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(experiment_rows=len(rows), configuration_rows=len(groups),
                          paired_rows=len(data['paired_fixed_scale']), best_rows=len(data['completed_job_best_configurations']),
                          files=len(manifest)),ensure_ascii=True))


if __name__ == '__main__':
    main()
