"""Rigid-fit saved condition points and audit actual baseline prompts; no inference."""
from pathlib import Path
import collections
import csv
import datetime
import gzip
import hashlib
import json
import subprocess
import tarfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import compare_comparison5_selected as common
import summarize_results as saved_stats

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw/alignment_inputs.json.gz'
EXPECTED_INPUT_SHA = 'b42a726ffb5ec97caff62b85d81ee113d6500a9550540d6b511bd6f7b6a2aff7'
SCIENCE_COMMIT = '072e6f7831bd91b398f7e1cbd65f4725268ec9f4'


def review(out):
    selected=out['selected_current_best_round_populations']
    rounds=out['baseline_rounds']
    f=common.f
    table=common.table
    notes_6=[
        ('向上探测uniform80、90；另选[70,70,60,60]',
         '90晋级完整50得0.590092；80止于20未晋级；混合配置0.638535。两个范围配置名额用完。'),
        ('uniform110、120及[130,120,110,100]；依据70→90的收益优先上侧',
         '三项均budget_limited，新增0。prompt已有remaining_configurations=0，但合法性审计仍为legal_unmeasured。'),
        ('uniform100、110及[110,110,90,90]；继续寻找90上方的更好落点',
         '三项仍budget_limited，新增0。synthesis明说“50 remaining attempts but 0 remaining configurations”，仍选择无法准入的方案。'),
        ('[90,90,90,70]、[90,70,70,70]、[90,90,50,50]；改为确认档内下调后段',
         '分别0.590907、0.608252、0.590806 Å，绝对位置均未超过第1轮。最后一项片段叠合后略好。'),
        ('proposal建议[90,90,90,50]、[90,90,50,90]、[90,90,60,50]；synthesis选择hold',
         '预计晚段下调收益小，上侧又不可用，最终selected=[]；新增0。仍有550正式尝试余额，未耗尽整个参数空间。'),
    ]
    notes_3=[
        ('前段上调、后段下调：如[100,90,90,80]等三项',
         '三项均比初始标定uniform90差，轮最佳0.354690。'),
        ('保持后段90，仅抬高前段；另测[110,110,110,100]',
         '轮最佳[100,100,90,90]为0.340409；仍未超过标定。'),
        ('转向前段下调：[80,90,90,90]、[80,80,90,90]、[80,80,80,90]',
         '轮最佳0.318907，较前两轮接近标定；多窗下调更差。'),
        ('前段保留90并下调后段，另有前段一升一降的组合',
         '[90,90,80,80]达到0.280764，已接近但仍差于标定0.279239。'),
        ('缩小偏离：只降低最后窗口，或[90,90,100,80]',
         '[90,90,90,80]为0.279807；叠合指标则由[90,90,100,80]的0.216268领先。'),
    ]
    lines=['# RMSD对齐与两教师四窗口基线迭代复盘','',
           '当前RMSD已经在蛋白锚定坐标系中计算，没有对配体条件片段单独刚体拟合。'
           '本次补算同一终态对应九原子的proper Kabsch叠合RMSD。6pyy后续缺少改进的主要可确认原因是范围准入受限、两轮零采样及后期低收益落点；'
           '3w54的正式轮最佳逐轮改善，但原绝对坐标指标始终没有超过自身标定最优。','',
           '## 现有预对齐到底做了什么','',
           '实际condition中的protein_alignment要求由蛋白重原子身份决定旋转和平移，配体不参加拟合。'
           '两教师的moving/reference蛋白SHA相同，所以实际都是R=I、t=0、蛋白拟合RMSD=0；'
           '6pyy有8674个蛋白锚点，3w54有2147个。此前汇总是在这个已共用的蛋白坐标系中按终态Hungarian对应计算绝对位置误差，'
           '不存在漏掉两个不同蛋白坐标系之间变换的证据。','',
           '新增叠合先保留原记录中终态匹配的九个生成原子槽位，再对这九个点相对于原九个条件点拟合旋转和平移；'
           '不缩放、不允许镜像，不在拟合后重新挑原子。它测的是同一几何对应的条件片段形状，'
           '去除了片段相对蛋白的位置与朝向误差；不是全配体化学同构RMSD，也不保证原子化学身份。'
           '绝对位置指标与片段叠合指标应并报，后者没有替换原损失、在线排名或采样结果。', '',
           '## 当前六项最优轮配置的补算结果','',
           '沿用上一表已选择的六个完整50候选配置，不按新指标重选轮次；以下单位均为Å。'
           '叠合后均无按既有上侧规则剔除的异常值，前10%均值取最小的5个。','',
           table(['实验','教师','原选轮','蛋白对齐绝对RMSD均值','片段叠合均值','片段叠合最优值','叠合去异常均值','叠合最优10%均值'],
                 [[r['strategy'],r['teacher'].split('_')[0],r['round'],f(r['absolute_rmsd_A']['mean'],6),
                   *[f(r['condition_rigid_fit_rmsd_A'][key],6) for key in ['mean','best','filtered_mean','top10_mean']]] for r in selected]),'',
           '6pyy基线绝对均值0.590091→叠合均值0.480663，仍保留明显片段形状误差。'
           '按每候选平方误差分解后再人口平均，平移项占14.58%，可由旋转消除的项占17.32%，刚体拟合后的残差占68.10%；'
           '这些是均方误差份额，不是平均RMSD可直接相加的份额，也不能据此推断生成动力学。', '',
           '## 若按新增叠合指标重新选轮，排序是否变化','',
           table(['教师','绝对位置指标的轮最佳','叠合指标的轮最佳','含义'],[
               ['6pyy','第1轮[90,90,90,90]，0.590091','第4轮[90,90,50,50]，0.479485',
                '第1轮同口径叠合值0.480663；第4轮仅低0.001178 Å，约0.245%。绝对位置却升至0.590805。'],
               ['3w54','第5轮[90,90,90,80]，0.279807','第5轮[90,90,100,80]，0.216268',
                '仍是第5轮，但最优配置改变；原绝对位置最佳配置的叠合值为0.216905。']]),'',
           '因此，6pyy“最优轮是第一轮”对原来的绝对位置目标成立；改用片段形状指标后，第4轮有极小改善。'
           '这是评价目标变化带来的离线排序变化，不是已证明的普遍收益。', '',
           '## 6pyy五轮实际做了什么','',
           '初始完整标定确认档为40/50/60/70，起点uniform70的绝对RMSD为0.637971 Å。'
           '四窗口是10–25、26–50、51–75、76–99步。下表的策略来自实际proposal/synthesis可见说明，结果来自真实完成组与范围回执。', '',
           table(['轮次','实际思考与选择','执行及结果'],
                 [[i+1,a,b] for i,(a,b) in enumerate(notes_6)]),'',
           '第1轮主要获得了新的范围信息：uniform90比起点70低约7.50%，但同时成为确认域最高点。'
           'uniform80虽然没有晋级，也占用一个范围配置名额；uniform90占另一个，后续配置名额为0。'
           '第2/3轮的六项不能记作“做过但不如第一轮”，因为根本没有采样；两轮的未测100/110/120/130不能据此宣称差或安全。', '',
           '第4轮三个实测落点均为下调已确认档。两条晚段下调只改变千分之一Å左右；'
           '[90,70,70,70]则比uniform90高约0.018161 Å。它们提供了这几个完整配置的结果，不能证明全部晚段调整无效。'
           'Agent把很小的晚期increment视作“几乎无法触及主要误差”的理由，是未充分验证的机制推断。'
           '新增Kabsch结果也说明，第4轮有一项在形状上略好、在绝对位置上略差，不能只用“晚期无用”解释。', '',
           '第5轮hold是对现有候选预期收益的判断，不是组合空间已经穷尽。'
           '第2/3轮没有新完整50观测，不能制造连续下降；checkpoint_learning在五轮中均未触发rebase。', '',
           '## 与3w54逐轮对照','',
           '3w54初始确认档为80/90/100/110，uniform90位于确认域内部；四档完整均值分别约0.384321/0.279239/0.335355/0.373872 Å。'
           '前后两侧都可以直接构造正式动作，五轮选中3/3/3/3/2项，实际形成14个正式50候选组，共700次；没有被范围预算拦截。', '',
           table(['轮次','实际调整思路','实测反馈'],[[i+1,a,b] for i,(a,b) in enumerate(notes_3)]),'',
           table(['教师','标定最佳绝对RMSD','第1轮新配置最佳','第2轮','第3轮','第4轮','第5轮'],
                 [[j.split('_')[-1],f(next(c['best'] for c in out['baseline_calibration'] if c['job']==j)['absolute_rmsd_A']['mean'],6),
                   *[f(r['raw_winner']['absolute_rmsd_A']['mean'],6) if r['raw_winner'] else '无新完整测量'
                     for r in rounds if r['job']==j]] for j in ['01_baseline_6pyy','02_baseline_3w54']]),'',
           '3w54第1轮到第5轮的正式轮最佳从0.354690降到0.279807，下降21.11%；'
           '但标定已经是0.279239，第5轮仍高0.000567 Å（约0.203%）。'
           '所以这更准确地是：探索早期偏离较好的起点，随后逐渐回到附近；没有在原目标上创造超过标定的新最优。'
           '五轮实际prompt中的当前最佳一直是标定uniform90，不是以上一轮较差正式配置为新起点。'
           'skills实际prompt也明确要求包含标定最佳，不能改用较差正式组作为“优化成功”的起点。', '',
           '新增片段形状指标下，3w54标定uniform90为0.216704 Å，正式第5轮[90,90,100,80]为0.216268 Å，'
           '仅低约0.000436 Å（0.201%）。这和绝对位置目标不是一回事，不能混合宣称大幅收益。', '',
           '## 可以定位到哪些模块和prompt问题','',
           table(['位置','实际证据','影响'],[
             ['linear_scale.py / 标定确认域','6pyy初始40–70、首轮后最高90；3w54为80–110且90在内部。',
              '两教师的可执行探索余量不对称。相同五轮不等于相同搜索机会。'],
             ['linear_council.py → analysis_tools.audit_action_batch','真实工具回执能返回legal_unmeasured、task_type=range_evidence；提案/选择校验主要检查结构、证据、ID和动作边界。',
              '结构合法不代表此刻预算可执行；没有在选择前硬性排除已耗尽范围名额的动作。'],
             ['linear_runtime.range_allowance / range_evidence','执行阶段在remaining_configurations<1时才返回budget_limited。6pyy第2/3轮均被此门控拦截。',
              '预算保护按冻结协议生效，但准入检查过晚，浪费了两轮Agent决策机会；不是GPU崩溃或一次失败采样。'],
             ['skills 2.8.1的实际遵循','系统prompt已明确不要理解生成机制、不要消融反证；6pyy第4轮仍写“真实承载位置”，3w54第3轮仍写确定某方向“整体不利”。',
              '不是这些指引未传到服务端，而是语义遵循不足；现有schema不检查这类解释导向的选点。'],
             ['working_neighbourhoods 与范围额度同时展示','6pyy显示偏好区间[70,110]，但确认档仅40/50/60/70/90且新范围名额为0。',
              '容易把偏好范围误读为可直接执行范围；第3轮LLM已意识到风险却仍然全选该类动作。']]),'',
           '能由日志确认的是搜索流程和具体落点造成的有限覆盖，不能据此证明6pyy的生成机制存在何种物理缺陷。'
           '两教师基线都完整完成五轮且标定/质量有回执；本分析未发现NaN、GPU异常或接口中止来解释这两项差异。', '',
           '若下一步优化，优先让动作审计显式返回“此刻可执行/不可执行及预算原因”，并让proposal和synthesis共享同一准入过滤；'
           '标定起点应有两侧确认余量，前沿取证名额按实际可执行搜索规划分配；'
           'prompt应以“该完整落点为何可能超过包含标定的当前最佳”为选点依据，避免为了定位窗口贡献、补曲线或证伪而选择。'
           '这些是后续修改建议，本次没有改动科学代码、预算、旧prompt或启动任何新实验。', '',
           '## 证据与核验范围','',
           '2026-11-01 18:20（Asia/Shanghai）从仍可见的只读完成组读取57个人口视图、1910个坐标视图；'
           '其中含标定晋级视图，不能作为1910个独立新尝试。所有records/terminal文件都与既有SHA相符、读前读后未变。'
           '1910个刚体拟合逐个与SciPy Rotation.align_vectors核对，确认proper rotation、RMSD不升高、平移分解恒等式和原终态匹配成本。'
           '原RMSD汇总最大差2.23e-16 Å，与原float32记录最大差9.10e-7 Å。', '',
           '同时核对两教师五轮×两角色的20个实际上下文与49次真实REQUEST；读取既有proposal/synthesis输出、工具回执及范围事件。'
           '没有加载模型权重、重评分、重新采样或调用LLM。Kabsch参考：[SciPy官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html)。', '',
           '[完整机器读数、逐轮参数与实际角色说明](RMSD对齐与基线迭代读数.json)｜'
           '[57个人口视图逐口径四项统计CSV](RMSD对齐逐配置.csv)｜'
           '[原完整任务最优轮表](完整任务最优轮对比.md)','']
    common.write(HERE/'RMSD对齐与基线迭代复盘.md','\n'.join(lines))


def rigid_fit(x, y):
    xc, yc = x-x.mean(0), y-y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    translation = y.mean(0)-x.mean(0)@rotation
    error = float(np.sqrt(np.mean(np.sum((x@rotation+translation-y)**2, axis=-1))))
    assert np.linalg.det(rotation) > .999999999
    # Independent SciPy implementation: rssd is sqrt(sum), not RMSD; K=9.
    reference_rotation, rssd = Rotation.align_vectors(yc, xc)
    assert abs(rssd/np.sqrt(len(x))-error) < 1e-8
    assert np.max(np.abs(reference_rotation.as_matrix().T-rotation)) < 1e-7
    return error, rotation, translation


def statistic(values):
    excluded, eligible, fences = saved_stats.flags(values, 'rmsd')
    return dict(**saved_stats.stats(values, 'rmsd', excluded, eligible), fences=fences)


def main():
    assert common.sha(RAW) == EXPECTED_INPUT_SHA
    blob = json.load(gzip.open(RAW, 'rt', encoding='utf-8'))
    metadata = common.read(common.SOURCE)
    meta_groups = {g['folder']: g for g in metadata['groups']}
    best_source = common.read(HERE / '完整任务最优轮对比.json')
    audit_base = common.EXPERIMENTS/'20261031_guidance_comparison5/monitoring/20261031_2017'
    decisions = common.read(audit_base/'llm_round_decisions.json')
    round_events = common.read(audit_base/'rounds_and_range_events.json')
    groups = {}
    candidate_rows = []
    validations = dict(source_groups=len(blob['groups']), finite_coordinate_views=0,
                       independent_scipy_fit_checks=0, nonincreasing_fit_checks=0,
                       frozen_mapping_cost_checks=0, raw_rmsd_reconciliation_max_difference=0.,
                       saved_native_rmsd_max_difference=0.)
    frames = {}
    for path, item in blob['conditions'].items():
        c=item['value'];frame=c['protein_alignment']
        frames[path]=dict(condition_file_sha256=item['sha256'], condition_sha256=c['condition_sha256'],
                          **{k: v for k,v in frame.items() if k!='atom_identities'},
                          protein_anchor_count=len(frame['atom_identities']))
        assert frame['ligand_participates_in_fit'] is False
        assert np.array_equal(frame['rotation'],np.eye(3))
        assert np.array_equal(frame['translation_angstrom'],np.zeros(3))
    for g in blob['groups']:
        c=blob['conditions'][g['condition_path']]['value'];lookup={a['id']:i for i,a in enumerate(c['atoms'])}
        target=np.array([c['reference_coordinates_angstrom'][lookup[a]] for a in c['controlled_atom_ids']])
        assert target.shape == (9,3)
        frame=c['protein_alignment'];r=np.array(frame['rotation']);t=np.array(frame['translation_angstrom'])
        rows=[]
        for xyz,mask,record in zip(g['coordinates_angstrom'],g['valid_mask'],g['records']):
            xyz=np.asarray(xyz)@r+t;slots=np.asarray(record['condition_generated_slots'],dtype=int)
            assert len(set(slots))==9 and np.asarray(mask)[slots].all() and np.isfinite(xyz).all()
            x=xyz[slots]
            absolute=float(np.sqrt(np.mean(np.sum((x-target)**2,axis=-1))))
            valid=np.flatnonzero(mask);cost=np.sum((target[:,None]-xyz[valid][None])**2,axis=-1)
            a,b=linear_sum_assignment(cost)
            assert abs(float(np.sqrt(np.mean(cost[a,b])))-absolute)<1e-10
            fitted,rotation,translation=rigid_fit(x,target)
            shift=float(np.linalg.norm(x.mean(0)-target.mean(0)))
            centered=float(np.sqrt(np.mean(np.sum(((x-x.mean(0))-(target-target.mean(0)))**2,axis=-1))))
            assert fitted<=centered+1e-10 and centered<=absolute+1e-10
            assert abs(absolute**2-shift**2-centered**2)<1e-9
            native=record['condition_coordinate_root_mean_square_deviation_angstrom']
            validations['saved_native_rmsd_max_difference']=max(validations['saved_native_rmsd_max_difference'],abs(absolute-native))
            assert abs(absolute-native)<5e-6
            row=dict(group=g['folder'],teacher=g['teacher'],node_id=record['node_id'],
                     seed=record['logical_seed'],slot=record['logical_slot'],built=record['build_success'],
                     assigned_generated_slots=slots.tolist(),absolute_rmsd_A=absolute,
                     centroid_shift_A=shift,centered_unrotated_rmsd_A=centered,
                     ligand_condition_kabsch_rmsd_A=fitted,
                     rotation=rotation.tolist(),translation_A=translation.tolist())
            rows.append(row);candidate_rows.append(row)
        abs_values=[x['absolute_rmsd_A'] for x in rows];fit_values=[x['ligand_condition_kabsch_rmsd_A'] for x in rows]
        delta=abs(float(np.mean(abs_values))-meta_groups[g['folder']]['stats']['rmsd']['mean'])
        validations['raw_rmsd_reconciliation_max_difference']=max(validations['raw_rmsd_reconciliation_max_difference'],delta)
        assert delta<1e-10
        total_mse=float(np.mean(np.square(abs_values)))
        translation_mse=float(np.mean([x['centroid_shift_A']**2 for x in rows]))
        centered_mse=float(np.mean([x['centered_unrotated_rmsd_A']**2 for x in rows]))
        fitted_mse=float(np.mean(np.square(fit_values)))
        fractions=dict(translation=translation_mse/total_mse,
                       rotation_removable=(centered_mse-fitted_mse)/total_mse,
                       residual_after_rigid_fit=fitted_mse/total_mse)
        assert abs(sum(fractions.values())-1)<1e-10
        groups[g['folder']]=dict(folder=g['folder'],teacher=g['teacher'],population=len(rows),
                                schedule=meta_groups[g['folder']]['schedule'],
                                record_sha256=g['record_sha256'],terminal_sha256=g['terminal_sha256'],
                                condition_file_sha256=blob['conditions'][g['condition_path']]['sha256'],
                                absolute_rmsd_A=statistic(abs_values),
                                condition_rigid_fit_rmsd_A=statistic(fit_values),
                                centroid_shift_A=statistic([x['centroid_shift_A'] for x in rows]),
                                squared_error_decomposition_fractions=fractions)
        for key in ['finite_coordinate_views','independent_scipy_fit_checks','nonincreasing_fit_checks','frozen_mapping_cost_checks']:
            validations[key]+=len(rows)
    selected=[dict(strategy=t['strategy'],suite=t['suite'],round=t['selected_round'],
                   **groups[t['best_configuration']['folder']]) for t in best_source['selected']]

    # Recover what the provider actually saw; no reasoning_content is collected.
    packets={}
    for request in blob['prompts']:
        assert request['model']=='deepseek-flash' and request['max_tokens']==16384
        key=(request['job'],request['role'],request['advice'])
        context=next(json.loads(m['content']) for m in request['messages']
                     if m['role']=='user' and m['content'].startswith('{'))
        if key in packets:
            assert packets[key]['context']==context
            packets[key]['request_sha256'].append(request['sha256'])
        else:
            packets[key]=dict(context=context,request_sha256=[request['sha256']],
                              system_prompt_sha256=[hashlib.sha256(m['content'].encode()).hexdigest()
                                                    for m in request['messages'] if m['role']=='system'])
    roles=[]
    for r in decisions:
        key=(r['job'],r['role'],r['folder'])
        if key not in packets:
            continue
        p=packets[key];context=p['context']
        for name in ['range_evidence_allowance','confirmed_constant_levels','checkpoint_learning']:
            assert context[name]==r[name]
        roles.append(dict(job=r['job'],role=r['role'],folder=r['folder'],
                          source_prompt_sha256=r['prompt_source_sha256'],request_sha256=p['request_sha256'],
                          system_prompt_sha256=p['system_prompt_sha256'],
                          range_evidence_allowance=context['range_evidence_allowance'],
                          remaining_formal_attempts=context['remaining_formal_attempts'],
                          confirmed_constant_levels=context['confirmed_constant_levels'],
                          working_neighbourhoods=context['working_neighbourhoods'],
                          checkpoint_learning=context['checkpoint_learning'],
                          proposals_or_selection=r['answer']))
    assert len(roles)==20 and sum(len(r['request_sha256']) for r in roles)==49

    action_audits=[]
    with tarfile.open(audit_base/'full_final_readouts.tar.gz','r:gz') as archive:
        for job in ['01_baseline_6pyy','02_baseline_3w54']:
            payload=json.load(archive.extractfile(job+'_final.json'))
            for role in payload['roles']:
                if 'analysis_final' in role['folder']:
                    continue
                for receipt in role['tool_receipts']:
                    if receipt['call']['function']['name']!='audit_action_batch':
                        continue
                    action_audits.append(dict(job=job,role=role['role'],folder=role['folder'],
                        receipt_path=receipt['path'],receipt_sha256=receipt['sha256'],
                        evidence_id=receipt['result'].get('evidence_id'),
                        status=receipt['result'].get('status'),
                        rejected_result=receipt['result'] if 'actions' not in receipt['result'] else None,
                        actions=[dict(id=a['action']['id'],status=a['status'],
                                      schedule=a.get('resolved',{}).get('effective_guidance_scale_schedule'),
                                      task_type=a.get('resolved',{}).get('task_type'),
                                      observed_performance=a.get('observed_performance'))
                                 for a in receipt['result'].get('actions',[])]))

    round_analysis=[]
    for job in ['01_baseline_6pyy','02_baseline_3w54']:
        root=f'/workspace/agenticgen_guidance_comparison5_20261031/jobs/{job}/attempt_001'
        for event in (r for r in round_events if r['job']==job):
            options=[groups[root+'/'+g['id']] for g in event['groups'] if g.get('complete') and g.get('attempts')==50]
            raw_winner=min(options,key=lambda g:g['absolute_rmsd_A']['mean']) if options else None
            fit_winner=min(options,key=lambda g:g['condition_rigid_fit_rmsd_A']['mean']) if options else None
            synthesis=next(r for r in roles if r['job']==job and r['role']=='synthesis'
                           and r['folder'].endswith(f"advice_{event['iteration']:03d}"))
            round_analysis.append(dict(job=job,iteration=event['iteration'],
                                       requested_actions=event['requested_actions'],complete50_configurations=len(options),
                                       raw_winner=raw_winner,condition_rigid_fit_winner=fit_winner,
                                       groups=options,range_events=event['range_events'],
                                       synthesis=synthesis))

    calibration=[]
    for job in ['01_baseline_6pyy','02_baseline_3w54']:
        gg=[g for p,g in groups.items() if '/'+job+'/' in p and '/calibration/' in p and g['population']==50]
        best=min(gg,key=lambda g:g['absolute_rmsd_A']['mean'])
        calibration.append(dict(job=job,best=best))
    frozen_modules={}
    repo=HERE.parents[3]
    for name in ['absolute_coordinate.py','linear_scale.py','linear_council.py','linear_runtime.py','analysis_tools.py']:
        path='agenticgen/src/agenticgen/uniguide/'+name
        content=subprocess.check_output(['git','show',SCIENCE_COMMIT+':'+path],cwd=repo)
        frozen_modules[name]=dict(commit=SCIENCE_COMMIT,path=path,sha256=hashlib.sha256(content).hexdigest())
    out=dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             protocol='proper Kabsch rotation+translation of the same terminal-assigned nine condition atoms; '
                      'no rematching after ligand fit, no scaling/reflection, no whole-ligand isomorphism claim',
             initial_metric='protein-registered absolute coordinate RMSD; ligand does not participate in fit',
             input_export_sha256=common.sha(RAW),collector_sha256=blob['collector_sha256'],
             conditions=frames,selected_current_best_round_populations=selected,
             baseline_rounds=round_analysis,baseline_calibration=calibration,
             baseline_actual_role_prompts_and_decisions=roles,groups=list(groups.values()),
             actual_action_audit_receipts=action_audits,
             frozen_science_modules=frozen_modules,validation=validations,
             new_sampling=0,new_model_forwards=0,new_llm_calls=0,
             scipy_method_reference='https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html')
    with (HERE/'RMSD对齐逐配置.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['教师','配置','人口','口径','均值','最优值','剔除异常后均值','最优10%均值','异常数','原始来源'])
        for g in groups.values():
            for key,label in [('absolute_rmsd_A','蛋白对齐后的绝对位置RMSD'),('condition_rigid_fit_rmsd_A','同对应条件片段Kabsch叠合RMSD')]:
                s=g[key];w.writerow([g['teacher'],str(g['schedule']),g['population'],label,
                                     *[s[k] for k in ['mean','best','filtered_mean','top10_mean','excluded']],g['folder']])
    candidate_path=HERE/'raw/alignment_candidate_readouts.jsonl.gz'
    candidate_text=''.join(json.dumps(r,ensure_ascii=True,allow_nan=False)+'\n' for r in candidate_rows)
    candidate_path.write_bytes(gzip.compress(candidate_text.encode(),mtime=0))
    out['raw_candidate_readouts_sha256']=common.sha(candidate_path)
    common.write(HERE/'RMSD对齐与基线迭代读数.json',json.dumps(out,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    review(out)
    print(json.dumps(dict(validation=validations,selected=[dict(strategy=r['strategy'],teacher=r['teacher'],
        absolute=r['absolute_rmsd_A']['mean'],aligned=r['condition_rigid_fit_rmsd_A']['mean']) for r in selected],
        candidate_readout_sha256=common.sha(candidate_path),prompt_roles=len(roles),actual_requests=49),ensure_ascii=True))


if __name__ == '__main__':
    main()
