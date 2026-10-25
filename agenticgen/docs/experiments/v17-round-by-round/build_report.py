"""Rebuild a retrospective from frozen existing results; no model calls or tests."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'evidence.json').read_text(encoding='utf-8'))
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False, 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'savefig.facecolor': 'white'})
COL = {'top1': '#256A98', 'top2': '#BA663D'}
ROUND_COL = ['#B5C9DA', '#91B4C5', '#639AAF', '#36758F', '#164C69']

def write_csv(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)

def save(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        output = FIG / f'{name}.{ext}'
        fig.savefig(output, dpi=240, bbox_inches='tight')
        if ext == 'svg':
            output.write_bytes(output.read_bytes().replace(b'\r\n', b'\n'))
    plt.close(fig)

def axes_style(ax):
    ax.grid(axis='y', color='#E4E8EC', linewidth=.5)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis='y', useOffset=False, style='plain')

rows = []
baserows = {}
for r in D['report']['rows']:
    teacher = r['teacher'].split('__')[-1]
    if r['collection'] == 'R0':
        baserows[teacher] = r
    if not r['collection'].endswith('_unselected'):
        continue
    c, a = r['condition_rmsd_A_all_attempts'], r['affinity']
    rows.append({'teacher': teacher, 'round': r['round'],
        'group': r['collection'].replace('_unselected', ''),
        'ticks': json.dumps(r['ticks']), 'lambda': json.dumps([v / 10 for v in r['ticks']]),
        'n': r['attempts'], 'C_mean_A': c['mean'], 'C_min_A': c['min'],
        'C_median_A': c['median'], 'C_max_A': c['max'],
        'pic50_mean': a['mean'], 'pic50_best': a['max'], 'pic50_min': a['min'],
        'pic50_median': a['median'], 'teacher_pic50': r['teacher_pic50'],
        'gap': r['mean_affinity_absolute_gap'],
        'PB_fraction': r['posebusters_pass_fraction_all_attempts'],
        'severe_clash_fraction': r['severe_clash_fraction_all_attempts'],
        'plip_recovery': r['mean_plip_recovery'],
        'paired_mean_vs_R0': r['paired_affinity_vs_R0']['delta']['mean'],
        'paired_win_rate_vs_R0': r['paired_affinity_vs_R0']['win_rate_common_valid']})
rows.sort(key=lambda r: (r['teacher'], r['round'], r['group']))
LIMIT = {t: min(r['C_mean_A'] for r in rows if r['teacher'] == t) + .02 for t in COL}
for r in rows:
    r['in_primary_band'] = r['C_mean_A'] <= LIMIT[r['teacher']]
    r['primary_band_upper_A'] = LIMIT[r['teacher']]

def rank(r):
    return (r['gap'], r['C_mean_A'], -r['pic50_best'], r['round'], r['group'])

rounds = []
leaders = {}
for t in COL:
    prevmean = baserows[t]['affinity']['mean']
    for n in range(1, 6):
        current = [r for r in rows if r['teacher'] == t and r['round'] == n]
        pool = [r for r in rows if r['teacher'] == t and r['round'] <= n and r['in_primary_band']]
        lead = min(pool, key=rank)
        leaders[t, n] = lead
        rounds.append({'teacher': t, 'round': n, 'attempts': sum(r['n'] for r in current),
            'new_schedules_ticks': json.dumps([json.loads(r['ticks']) for r in current]),
            'round_C_mean_A': sum(r['C_mean_A'] for r in current) / 4,
            'round_C_min_A': min(r['C_min_A'] for r in current),
            'round_pic50_mean': sum(r['pic50_mean'] for r in current) / 4,
            'round_pic50_best': max(r['pic50_best'] for r in current),
            'in_band_groups_this_round': sum(r['in_primary_band'] for r in current),
            'leader': f"R{lead['round']}{lead['group']}", 'leader_ticks': lead['ticks'],
            'leader_C_A': lead['C_mean_A'], 'leader_mean': lead['pic50_mean'],
            'leader_best': lead['pic50_best'], 'leader_gap': lead['gap'],
            'leader_mean_delta_vs_previous': lead['pic50_mean'] - prevmean,
            'ranking_scope': 'retrospective cumulative pool, final-version min-C band; R0 excluded'})
        prevmean = lead['pic50_mean']

comparisons = []
for t in COL:
    score = baserows[t]['teacher_pic50']
    comparisons.append({'teacher': t, 'population': 'Teacher', 'n': 1,
        'C_mean_A': 0., 'C_min_A': 0., 'pic50_mean': score, 'pic50_best': score,
        'interpretation': 'Single reference, zero coordinate error is self-comparison'})
    for pop, label in [('unguided_historical_slotmatched25', 'Unguided25'),
                       ('shared_R0_guided25', 'GuidedR0'), ('V17_primary25', 'V17')]:
        r = next(r for r in D['slot_matched_comparison']['rows'] if r['teacher'] == t and r['population'] == pop)
        comparisons.append({'teacher': t, 'population': label, 'n': r['n'],
            'C_mean_A': r['template_RMSD_mean_A'], 'C_min_A': r['template_RMSD_min_A'],
            'pic50_mean': r['pIC50_mean'], 'pic50_best': r['pIC50_best'],
            'interpretation': 'Historical initial-slot-matched reference, not fresh paired zero replay' if label == 'Unguided25' else 'Shared schedule, 25 candidates'})

def find(t, ticks):
    return next(r for r in rows if r['teacher'] == t and json.loads(r['ticks']) == ticks)

contrast_specs = [
 ('T1_early_7_8','top1',[7,5,5,3],[8,5,5,3],'早段0.7→0.8，其他窗口固定'),
 ('T1_early_8_9','top1',[8,5,5,3],[9,5,5,3],'早段0.8→0.9，其他窗口固定'),
 ('T1_middle_at7','top1',[7,5,5,3],[7,7,5,3],'早段0.7背景：第二窗口0.5→0.7'),
 ('T1_middle_at8','top1',[8,5,5,3],[8,7,5,3],'早段0.8背景：第二窗口0.5→0.7'),
 ('T1_release_givenA','top1',[8,7,5,3],[8,7,5,1],'第二窗口0.7背景：末段0.3→0.1'),
 ('T1_joint_vsleader','top1',[8,5,5,3],[8,7,5,1],'第二窗口增加与末段释放的联合效果'),
 ('T2_early_3_1','top2',[3,3,5,5],[1,3,5,5],'第二窗口0.3背景：早段0.3→0.1'),
 ('T2_early_1_0_at3','top2',[1,3,5,5],[0,3,5,5],'第二窗口0.3背景：早段0.1→0'),
 ('T2_early_1_0_at5','top2',[1,5,5,5],[0,5,5,5],'第二窗口0.5背景：早段0.1→0'),
 ('T2_late_release','top2',[1,3,5,5],[1,3,5,3],'末段0.5→0.3，其他窗口固定'),
 ('T2_third_release','top2',[1,3,5,5],[1,3,3,5],'第三窗口0.5→0.3，其他窗口固定'),
 ('T2_second_3_1','top2',[1,3,5,5],[1,1,5,5],'第二窗口0.3→0.1，其他窗口固定')]
contrasts = []
for cid, t, a, b, note in contrast_specs:
    x, y = find(t,a), find(t,b)
    contrasts.append({'id': cid, 'teacher': t, 'reference_ticks': json.dumps(a),
        'target_ticks': json.dumps(b), 'reference': f"R{x['round']}{x['group']}",
        'target': f"R{y['round']}{y['group']}", 'delta_C_A': y['C_mean_A'] - x['C_mean_A'],
        'delta_mean': y['pic50_mean'] - x['pic50_mean'],
        'delta_best': y['pic50_best'] - x['pic50_best'],
        'target_in_band': y['in_primary_band'], 'description': note})
C = {r['id']: r for r in contrasts}
write_csv('v17_group_metrics.csv', rows)
write_csv('v17_round_summary.csv', rounds)
write_csv('baseline_comparison.csv', comparisons)
write_csv('matched_contrasts.csv', contrasts)
write_csv('campaign_version_trend.csv', D['version_trend'])

# Fig. 1: all versions plus a zoom that does not conceal the large regressions.
v = D['version_trend']
x = np.array([int(r['version'][1:]) for r in v])
best_gap = float(next(r for r in v if r['version'] == 'v17')['equal_gap'])
y = np.array([float(r['equal_gap']) - best_gap for r in v])
fig, aa = plt.subplots(2, 1, figsize=(9.2, 5.7), sharex=True, layout='constrained')
for ax in aa:
    ax.plot(x,y, '-o', color='#677785', lw=1.2, ms=3)
    ax.scatter([17],[0],s=105,marker='*',color='#CB7C2F',zorder=5,label='V17: historical best')
    ax.axhline(0,color='#A6B0BA',lw=.7)
    axes_style(ax);ax.set_xticks(x);ax.set_ylabel('Equal-teacher gap minus V17\n(pIC50; lower is better)')
aa[0].set_title('a  V6–V30: complete scale')
aa[0].legend(loc='upper right')
aa[1].set_ylim(-.00015,.0017);aa[1].set_title('b  Zoom near the best; off-scale versions labelled')
for xx,yy in zip(x,y):
    if yy > .0017:
        aa[1].plot(xx,.00162,marker='^',color='#677785',ms=5)
        aa[1].text(xx,.00142,f'V{xx}',ha='center',va='top',fontsize=7)
aa[1].set_xlabel('Version (one completed search per version)')
save(fig,'01_campaign')

# Fig. 2: four shared schedules per round and cumulative primary leader.
fig, aa = plt.subplots(2,3,figsize=(11,6.1),layout='constrained')
metric = [('C_mean_A','leader_C_A','Local RMSD (Å) ↓'),
          ('pic50_mean','leader_mean','Mean pIC50 ↑'),
          ('pic50_best','leader_best','Best pIC50 ↑')]
for i,t in enumerate(COL):
    rr = [r for r in rounds if r['teacher']==t]
    for j,(key,leadkey,title) in enumerate(metric):
        ax=aa[i,j]
        for g in range(1,5):
            pts=[r for r in rows if r['teacher']==t and r['group']==f'G{g}']
            for p in pts:
                ax.scatter(p['round']+(g-2.5)*.10,p[key],s=24,
                    marker='o' if p['in_primary_band'] else 'x',
                    color=COL[t] if p['in_primary_band'] else '#AB3E3E',alpha=.65)
        baseline = next(r for r in comparisons if r['teacher']==t and r['population']=='GuidedR0')
        u = next(r for r in comparisons if r['teacher']==t and r['population']=='Unguided25')
        ax.plot(range(6),[baseline[key]]+[r[leadkey] for r in rr],'-s',color=COL[t],lw=1.6,ms=4,label='Cumulative primary')
        ax.axhline(u[key],ls='--',lw=1,color='#73816B',label='Historical unguided25')
        if j==0:ax.axhline(LIMIT[t],ls=':',lw=1,color='#AB3E3E',label='Final 0.02 Å band limit')
        ax.set_title(f'{chr(97+i*3+j)}  {t.title()} · {title}')
        ax.set_xticks(range(6),['R0','R1','R2','R3','R4','R5'])
        axes_style(ax)
        if i==0 and j==0:ax.legend(loc='upper left',fontsize=6)
        ax.set_xlabel('Sampling round; R0 is guided')
save(fig,'02_round_progress')

# Fig. 3: the complete set of 40 executed four-window schedules.
cmap = LinearSegmentedColormap.from_list('guidance',['#F4F6F7','#AFCCD8','#337391','#163E58'])
fig, aa=plt.subplots(1,2,figsize=(8.6,8.3),layout='constrained')
for ax,t in zip(aa,COL):
    rr=[r for r in rows if r['teacher']==t]
    mat=np.array([json.loads(r['ticks']) for r in rr])/10
    im=ax.imshow(mat,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    labels=[]
    for k,r in enumerate(rr):
        isbest=r is leaders[t,5]
        labels.append(f"R{r['round']} {r['group']}"+(' ★' if isbest else '')+(' ×' if not r['in_primary_band'] else ''))
        for j in range(4):ax.text(j,k,f'{mat[k,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if mat[k,j]>=.65 else '#17394A')
    ax.set_yticks(range(20),labels);ax.set_xticks(range(4),['W1\n10–25','W2\n26–50','W3\n51–75','W4\n76–90'])
    ax.xaxis.tick_top();ax.set_title(t.title(),pad=35)
    for k in [3.5,7.5,11.5,15.5]:ax.axhline(k,color='white',lw=3)
    ax.tick_params(length=0)
fig.colorbar(im,ax=aa,shrink=.55,pad=.025,label='Guidance amplitude λ (tick / 10)')
fig.supxlabel('★ final primary shared schedule     × outside the final condition band',fontsize=8)
save(fig,'03_schedule_heatmap')

# Fig. 4: absolute performance, including teacher reference.
fig,aa=plt.subplots(2,3,figsize=(10.3,6.3),layout='constrained')
labels=['Teacher','Unguided\nmatched25','Guided\nR0','V17\nprimary25']
for i,t in enumerate(COL):
    data=[r for r in comparisons if r['teacher']==t]
    for j,(key,_,title) in enumerate(metric):
        ax=aa[i,j];values=[r[key] for r in data]
        bars=ax.bar(range(4),values,color=['#D5DADF','#ABB8A2','#C5D5DF',COL[t]],width=.65)
        ax.set_xticks(range(4),labels);ax.set_title(f'{chr(97+i*3+j)}  {t.title()} · {title}')
        ax.set_ylim(0,max(values)*1.18 if max(values)>0 else 1)
        for b,val in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(values)*.025,f'{val:.4f}',ha='center',fontsize=7)
        axes_style(ax)
fig.supxlabel('Teacher: one reference molecule; each generated population: n = 25. No error bars or significance claims.',fontsize=8)
save(fig,'04_reference_comparison')

# Fig. 5: matched local effects and the condition/function frontier.
fig,aa=plt.subplots(2,2,figsize=(10,7.2),layout='constrained')
ax=aa[0,0]
pts=[find('top1',[k,5,5,3]) for k in [7,8,9]]
ax.plot([.7,.8,.9],[r['pic50_mean'] for r in pts],'-o',color=COL['top1'])
for xx,p in zip([.7,.8,.9],pts):ax.annotate(f"{p['pic50_mean']:.6f}",(xx,p['pic50_mean']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7)
ax.set_xticks([.7,.8,.9]);ax.set_xlim(.68,.92);ax.margins(y=.25)
ax.set_title('a  Top1: local interior improves; higher dose reverses')
ax.set_xlabel('W1 amplitude (W2/W3/W4 = 0.5/0.5/0.3)');ax.set_ylabel('Mean pIC50');axes_style(ax)
ax=aa[0,1]
ids=['T1_middle_at7','T1_middle_at8','T1_release_givenA','T1_joint_vsleader']
labs=['A: W2 0.5→0.7 at W1=0.7','A: W2 0.5→0.7 at W1=0.8','B given A: W4 0.3→0.1','A+B versus R4 leader']
vals=[C[k]['delta_mean']*1e4 for k in ids]
ax.barh(range(4),vals,color=['#6F9EAD' if z>0 else '#BA775C' for z in vals],height=.6)
ax.set_yticks(range(4),labs);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.7)
ax.set_xlim(-3.25,2.9)
for i,z in enumerate(vals):ax.text(z+(.08 if z>=0 else -.08),i,f'{z:+.3f}',ha='left' if z>=0 else 'right',va='center',fontsize=7)
ax.set_title('b  Top1: response changes with background');ax.set_xlabel('Δ mean pIC50 (×10⁻⁴)')
ax=aa[1,0]
for r in rows:
    if r['teacher']!='top2':continue
    ax.scatter(r['C_mean_A'],r['pic50_mean'],color=ROUND_COL[r['round']-1],s=33,marker='o' if r['in_primary_band'] else 'x')
l=leaders['top2',5];ax.scatter(l['C_mean_A'],l['pic50_mean'],marker='*',s=130,color=COL['top2'],zorder=5)
ax.axvline(LIMIT['top2'],ls='--',color='#AB3E3E',lw=1)
ax.annotate('Final primary',(l['C_mean_A'],l['pic50_mean']),xytext=(-38,-30),textcoords='offset points',fontsize=7,arrowprops={'arrowstyle':'-','color':'#555'})
ax.text(LIMIT['top2']+.0007,4.14,'Band limit',rotation=90,color='#AB3E3E',fontsize=7)
ax.set_title('c  Top2: higher mean outside band is not selected')
ax.set_xlabel('Group mean local RMSD (Å)');ax.set_ylabel('Mean pIC50');axes_style(ax)
ax=aa[1,1]
ids=['T2_early_3_1','T2_early_1_0_at3','T2_late_release','T2_third_release']
labs=['W1 0.3→0.1 (W2=0.3)','W1 0.1→0 (W2=0.3)','W4 0.5→0.3','W3 0.5→0.3']
vals=[C[k]['delta_mean'] for k in ids]
ax.barh(range(4),vals,color=[COL['top2'] if C[k]['target_in_band'] else '#AA4444' for k in ids],height=.6)
ax.set_yticks(range(4),labs);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.7);ax.set_xlim(-.011,.023)
for i,z in enumerate(vals):
    ax.text(z+.0004 if z>=0 else z/2, i, f'{z:+.6f}',
            ha='left' if z>=0 else 'center', va='center', fontsize=7,
            color='#222222' if z>=0 else 'white')
ax.set_title('d  Top2: early release dominates; late gain is small');ax.set_xlabel('Δ mean pIC50; red = target outside band')
save(fig,'05_matched_effects')

BEHAVIOR = {
 'top1': {
  1: ('分别改变早段与末段，先比较两个窗口的方向',
      '四组为早段0.7、早段0.3、末段0.7、末段0.3，其余保持0.5。早段0.7的Mean最高；末段减弱有小幅功能收益，末段增强得到更低C但Mean更低。该轮体现条件与功能的取舍。',
      '这些是已执行数组及结果支持的行为描述；未取得本轮完整提案理由，不能声称agent已确定了具体物理机制。'),
  2: ('在早段0.7方向上组合末段释放，同时考察中间窗口',
      '执行[7,5,5,3]、[7,5,7,5]、[5,7,5,3]和[7,5,5,7]。G1把早段增强与末段释放组合，成为累计领先；第三窗口增强的G2与R1G1相比Mean几乎不变。',
      'G3与G1背景不同，不能将二者差直接归因于第二窗口。这里可以确认搜索沿已有较好方向推进，但不能从数组反推未保存的私有推理。'),
  3: ('探索早段高侧，并把第二窗口增强迁移到多个背景',
      'G1在[7,5,5,3]背景把第二窗口0.5→0.7，Mean增加0.000163527；G2把早段0.7→0.9，Mean反而降低0.025490618，尽管Best升高。G1成为累计领先，G2的更高单分子Best没有替代组均目标。',
      '这一轮四组混合Mean下降主要由高侧探索的坏组拉低，不能据此说保留的最佳共享调度退步。早段0.8此时尚未出现在已执行数组中。'),
  4: ('细化早段内侧0.8，并检查第二窗口继续增强',
      'G4执行[8,5,5,3]，得到全版Top1主代表。相同背景早段0.7→0.8使Mean提高0.001663303、C增加0.000190487Å；0.8→0.9则Mean降低0.027153921。另两条第二窗口0.7→0.9边均不利。',
      '本轮有获胜G4私有和汇总直接证据：正确保留共享调度领先者，区别逐槽TOP1拼接结果；汇总选出G4:H1=[8,7,5,3]及H2=[8,7,5,1]作为下一轮主要挑战，并明确两者共享“第二窗口收益可迁移”的风险。'),
  5: ('检验第二窗口迁移A与A加末段释放B，并保留两条较弱替代',
      'G1为AB=[8,7,5,1]，G2为A=[8,7,5,3]，G3/G4在早段0.7背景继续末段释放。A相对R4领先者Mean降低0.000227394；B在A背景增加0.000115862；AB仍低于原领先者0.000111532。四组均未超越R4G4。',
      '执行结果反证了该背景下A的有利迁移预期，却没有否定B的条件收益。缺少[8,5,5,1]的单独B对照，不能识别完整交互或断言联合失利就是非加性。未读取本轮新会话，不声称agent事后如何修订；R5后没有R6。')},
 'top2': {
  1: ('与Top1相同的初始窗口探测，却出现相反的早段偏好',
      '早段0.3的G2 Mean为4.193721142，相对guided R0增加0.029242649；早段0.7的G1得到全版最低C=1.902795711Å，但Mean降至4.128502283。两者均在主带内，累计功能领先者为G2。',
      '此处“更低C”与“更接近teacher的Mean”存在实际取舍，不能用Top1早段增强的结论直接替代Top2。'),
  2: ('比较早段内侧0.2与高侧0.9，并探测第二、第三窗口',
      '执行[2,5,5,5]、[9,5,5,5]、[3,3,5,5]、[5,5,3,7]。早段0.2没有胜过0.3；0.9虽给出全版最高Best=7.320987701，但Mean仅4.157487688。第二窗口减弱近乎持平，累计领先仍为R1G2。',
      '这不是连续降低早段就单调提高功能的证据；0.2局部较差也不自动排除其他背景的0.1。最高Best仍低于teacher，不能据其提升评价整组成功。'),
  3: ('在第二窗口0.3背景继续降低早段，找到新的主要功能增益',
      'G2=[1,3,5,5]成为新领先者；相对R2G3=[3,3,5,5]，早段0.3→0.1使Mean增加0.016824512、C增加0.000917175Å。对照G3=[2,3,5,5]与G4=[9,3,5,5]均落后。',
      '该结果限定于匹配背景和固定人口，不能声称早段响应全局单调。不同teacher的有利早段幅度明显不同，支持按响应选择调度。'),
  4: ('试探早段零端，同时比较第二窗口恢复与继续释放',
      'G1=[0,3,5,5]相对领先者Mean降低0.006983528、C增加0.007510833Å并出带。G2把第二窗口恢复到0.5，Mean仅低0.000028324；G3继续减到0.1，Mean低0.001252642。G4的末段释放发生在早段0.3背景，仍未超越领先者。',
      '本轮累计领先保持R3G2。零早段在该背景不利，但不能将这一结果解释成所有背景零端都具有相同功能效应。'),
  5: ('在领先背景释放末段，获得最终代表；另一个零端功能较高但出带',
      'G1=[1,3,5,3]相对R3G2 Mean增加0.000749311、C增加0.000645594Å，成为最终代表；G2第三窗口释放增加Mean0.000054369；G3第二窗口0.4没有胜出。G4=[0,5,5,5] Mean较高为4.214500885，但C=1.944389154Å出带。',
      '本轮有G1私有与汇总直接证据：汇总保留带内代表、去重提案，识别零端出带，后续建议仅归档。其“固定人口噪声”等措辞不能取消真实有符号差异；原生分支可中介幅度效果。私有未见同波兄弟结果，不应按汇总可见性追责。')}
}

ACTION = {
 'top1': [
  ['早段增强至0.7','早段减弱至0.3','末段增强至0.7','末段减弱至0.3'],
  ['早段0.7背景释放末段','早段0.7背景增强第三窗口','末段0.3背景增强第二窗口','早段0.7背景增强末段'],
  ['第二窗口增强至0.7','早段继续增强至0.9','第二、第三窗口均0.7','第二窗口与末段均0.7'],
  ['第二窗口继续至0.9','第三窗口0.7背景下第二窗口至0.9','第二窗口0.7背景回到末段0.5','细化早段内侧0.8'],
  ['第二窗口迁移+末段释放AB','第二窗口迁移A','旧早段0.7背景末段降到0.1','旧早段0.7背景末段降到0.2']],
 'top2': [
  ['早段增强至0.7','早段减弱至0.3','末段增强至0.7','末段减弱至0.3'],
  ['早段内侧0.2','早段高侧0.9','早段0.3背景降低第二窗口','末段0.7背景降低第三窗口'],
  ['早段0.3且第三窗口0.3、末段0.7','第二窗口0.3背景早段降到0.1','第二窗口0.3背景早段0.2','第二窗口0.3背景早段0.9'],
  ['领先背景试探早段零端','早段0.1背景第二窗口恢复0.5','早段0.1背景第二窗口继续降至0.1','早段0.3/第三窗口0.3背景释放末段'],
  ['领先背景末段释放至0.3','领先背景第三窗口释放至0.3','领先背景第二窗口细化为0.4','第二窗口0.5背景试探早段零端']]
}
actions=[]
for r in rows:
    actions.append({'teacher':r['teacher'],'round':r['round'],'execution_group':r['group'],
        'ticks':r['ticks'],'observed_action':ACTION[r['teacher']][r['round']-1][int(r['group'][1:])-1],
        'basis':'executed schedule, not inferred private reasoning',
        'direct_proposal_origin':('R4 G4:H2' if r['group']=='G1' else 'R4 G4:H1' if r['group']=='G2' else 'R4 G1:H1' if r['group']=='G3' else 'R4 G1:H2') if r['teacher']=='top1' and r['round']==5 else 'not established from available decision contexts'})
write_csv('agent_action_ledger.csv',actions)

lines=[]
def add(s=''): lines.append(s)
def table(header, data):
    add()
    add('| '+' | '.join(header)+' |');add('| '+' | '.join(['---']*len(header))+' |')
    for r in data:add('| '+' | '.join(str(x).replace('|',' / ').replace('\n',' ') for x in r)+' |')
    add()
def fmt(x):return f'{float(x):.9f}'
def sign(x):return f'{float(x):+.9f}'

add('# LLM协助gradient guidance实验总结与V17逐轮行为报告')
add('\n**结论：V6–V30实验执行完成，但科学目标未达成；V17是既定条件优先规则下的历史最好版本。** V17的Top1主要通过R4早段内侧0.8取得收益，Top2主要通过降低早段指导及R5末段释放取得收益。相对同初始槽位历史unguided25，两位teacher的区域RMSD均降低约0.03Å，但平均pIC50分别仍低0.004284325和0.006465616，且仍大幅低于teacher参考值。这里的“最好”不等同于已优于所有基线或证明LLM具有因果收益。\n')
add('## 1. 范围、证据与阅读方法\n')
add('- 本报告总结25个已完成版本，重点展示V17两位teacher各R1–R5，共40组、1000条正式结果。V17是版本，R1–R5是该版本内部采样轮次。')
add('- 全轮行为依据已执行四窗口数组与真实组统计重建；直接私有/汇总理由来自已缓存的Top1 R4G4及Top2 R5G1，共四上下文。此次远端11762连接失败，未补取其余轮次提案；没有虚构其思考或补作全会话审计。`full_review_completed=false`。')
add('- 表中G1–G4是当轮执行组，不能据相同编号认定跨轮是同一私有agent谱系。每轮四私有各提两案，汇总从合法提案中选四个供下一轮执行；R1由bootstrap产生。R5之后的提案只归档，无R6。')
add('- 五轮间模型、奖励、代码与skills冻结。变化的是四窗口的指导幅度，不是模型权重。每个调度从同一guided R0的第10步更新前状态恢复，不是把上一轮生成分子继续优化。')
add('- 数据源为已完成版报告及冻结收据；本次只计算统计、绘图和写文档，无采样、重评分、部署、自动化测试或定时任务恢复。\n')
add('## 2. 算法与评价口径\n')
add('四窗口W1–W4依次为步骤10–25、26–50、51–75、76–90；数组以tick表示，1 tick=0.1强度。例如[8,5,5,3]表示[0.8,0.5,0.5,0.3]。每组25候选，来自五种子各五槽；同轮四组共享随机流。')
add('\n内层奖励为teacher核心加一跳区域与当前一步模型**预测终态坐标**的几何Hungarian匹配pseudo-Huber损失的负值（delta=1Å），通过当前模型前向图回传至当前坐标；模型权重冻结，匹配离散选择不求导，未展开后续整条轨迹。指导叠加于原生积分结果，单原子单步指导位移上限0.025Å。没有pIC50梯度、PLIP距离/角度梯度或流场一致性惩罚；pIC50参与外层调度选择。')
add('\n主指标C为真实终态、蛋白固定坐标系中，指定区域逐分子RMSD再取25个候选的平均值。Top1区域2原子，Top2区域8原子。无配体独立平移/旋转拟合；几何匹配不强制元素或键身份一致。因此C同时含位置、朝向与局部形变误差，不等同纯内部构象差异，也不证明interaction位点恢复。PLIP恢复为单列诊断。')
add('\n每teacher的20组以最小组均C锚定0.02Å带，带内按Mean pIC50到teacher绝对差选优，Best只作次级评价。V17带上界分别为0.814247352Å、1.922795711Å。0.02Å是资格/排序规则，不是噪声阈值。所有pIC50均为模型预测分数，不是实验测定值。')
add('\n**R0是四段均0.5的guided基线，不是unguided。** unguided取历史100结果中与本实验相同的25初始槽位；未执行当前路径的严格配对零指导重跑，不能将其视为完整因果对照。旧100样本均值仅作历史描述，不用于本报告判断指导效果方向。\n')
add('## 3. 全部版本的主要结果\n')
s=D['campaign_summary']
add(f"V6–V30共25版，正式{int(s['formal_candidates']):,}结果，加一次共享R0的50条；历史失败{int(s['failed_candidates_excluded']):,}条独立排除。正式结果全部评分有效，记录PB通过率100%，严重clash为0；这些质量指标不能代替条件重构成功。")
add('\nV17相对V16两teacher的Mean分别增加0.001504755、0.000746431，等权功能gap下降0.001125593；相对先前最好V13，等权gap下降0.000343704。最终V30仍未超V17，末版连续退步规则只报告结果，没有执行V31或额外恢复采样。')
add('\n![V6–V30版本趋势](figures/01_campaign.png)\n')
add('**图1｜版本表现。** 每点为一个已完成版本两teacher主代表的等权Mean–teacher绝对差，相对V17作差；低值更好。上图完整尺度，下图局部放大并明确标出超出显示范围的版本。每组25个固定候选，不是独立重复实验；没有置信区间或显著性检验。来源：[版本数据](campaign_version_trend.csv)。\n')
add('## 4. V17、teacher与unguided的表现\n')
table(['Teacher','对象','n','区域RMSD均值 Å','区域RMSD最小值 Å','Mean pIC50','Best pIC50'],
      [[r['teacher'],r['population'],r['n'],fmt(r['C_mean_A']),fmt(r['C_min_A']),fmt(r['pic50_mean']),fmt(r['pic50_best'])] for r in comparisons])
add('teacher行是单个参考分子，自比坐标误差为0；不能把n=1当作生成群体。RMSD最小值与Best pIC50可能来自不同候选，不能拼成一条“最佳分子”。V17主代表为Top1 R4G4=[8,5,5,3]与Top2 R5G1=[1,3,5,3]。')
add('\n![参考性能比较](figures/04_reference_comparison.png)\n')
add('**图2｜绝对性能。** 三列分别为区域RMSD组均值、pIC50均值和最大值。teacher为单一参考；其余每组25候选。图展示绝对差距，小幅变化见下表与逐轮图。来源：[对照数据](baseline_comparison.csv)。\n')
delta=[]
for t in COL:
    lead=next(r for r in comparisons if r['teacher']==t and r['population']=='V17')
    for target in ['Unguided25','GuidedR0']:
        b=next(r for r in comparisons if r['teacher']==t and r['population']==target)
        delta.append([t,'V17 − '+target,sign(lead['C_mean_A']-b['C_mean_A']),sign(lead['pic50_mean']-b['pic50_mean']),sign(lead['pic50_best']-b['pic50_best'])])
table(['Teacher','比较','ΔRMSD均值 Å（负好）','ΔMean pIC50（正好）','ΔBest pIC50'],delta)
add('V17相对匹配unguided25的区域RMSD分别降低3.8061%和1.5604%，但Mean及Best均略低。相对guided R0，V17功能Mean提高，C却略增，这是条件带内选择功能更优调度的取舍。两teacher平均分仍分别低于teacher 3.624416485和3.200974360；V17全部正式组的最高Best也仅7.215617657、7.320987701，均低于teacher。不能将恢复部分功能称为完成teacher重构。\n')
add('## 5. V17五轮搜索概览\n')
add('![V17逐轮性能](figures/02_round_progress.png)\n')
add('**图3｜逐轮性能。** 圆点为当轮四个共享调度的25候选统计，红叉为最终主条件带外调度。实线是累计最优共享调度，R0点为guided基线；累计线用最终版条件带回顾重建，不暗示agent当时已知未来C最小值。Best列是该累计主代表的Best，不是全池最大Best；虚线为同槽位历史unguided25。来源：[40组数据](v17_group_metrics.csv)、[逐轮汇总](v17_round_summary.csv)。\n')
table(['Teacher','轮次','当轮100结果Mean','当轮Best','累计主代表','代表C Å','代表Mean','代表Mean较前轮变化'],
      [[r['teacher'],f"R{r['round']}",fmt(r['round_pic50_mean']),fmt(r['round_pic50_best']),r['leader'],fmt(r['leader_C_A']),fmt(r['leader_mean']),sign(r['leader_mean_delta_vs_previous'])] for r in rounds])
add('当轮100结果是四组探索方案的混合统计，不是100个独立起点；其中包含有意探测但失败的方案。累计主代表上升与当轮探索池均值下降可以同时出现。R1变化以R0为参照；后续变化以此前累计主代表为参照。\n')
add('![V17全部执行调度](figures/03_schedule_heatmap.png)\n')
add('**图4｜全部已执行调度。** 每行一组25候选，颜色与格内数值均为实际λ；星号是最终主代表，叉号是最终条件带外组。四列对应四个控制窗口，组间白线区分R1–R5。全部数组来自执行结果，无预测方案混入。来源：[动作台账](agent_action_ledger.csv)。\n')
for t in COL:
    add(f"## {6 if t=='top1' else 7}. {t.title()}：每轮行为与效果\n")
    for n in range(1,6):
        title,observed,limit=BEHAVIOR[t][n]
        add(f'### R{n}：{title}\n');add(observed+'\n')
        rr=[r for r in rows if r['teacher']==t and r['round']==n]
        table(['执行组','ticks [W1,W2,W3,W4]','动作描述','C均值 Å','C最小值 Å','Mean pIC50','Best pIC50','主带'],
              [[r['group'],r['ticks'],ACTION[t][n-1][int(r['group'][1:])-1],fmt(r['C_mean_A']),fmt(r['C_min_A']),fmt(r['pic50_mean']),fmt(r['pic50_best']),'内' if r['in_primary_band'] else '外'] for r in rr])
        r=next(r for r in rounds if r['teacher']==t and r['round']==n)
        add(f"**取得的效果：**累计主代表为{r['leader']} {r['leader_ticks']}，Mean={fmt(r['leader_mean'])}，相对此前代表变化{sign(r['leader_mean_delta_vs_previous'])}。{limit}\n")
add('## 8. 有直接提案证据的预测—实测对应\n')
add('Top1 R4 G4私有明确提出H1（A）与H2（AB），汇总确实选入下一轮。R5执行G2对应H1，执行G1对应H2；另外两个执行组来自R4 G1的末段低侧提案。这一映射来自已保存汇总与数组，不能泛化为所有轮次的agent谱系。\n')
table(['提案','当时预测Mean / gap','R5实际Mean / gap','结论'],[
 ['G4:H1 [8,7,5,3]','Mean约4.21358；gap约3.62425',f"{fmt(find('top1',[8,7,5,3])['pic50_mean'])} / {fmt(find('top1',[8,7,5,3])['gap'])}",'A迁移预测未实现；低于R4领先者'],
 ['G4:H2 [8,7,5,1]','Mean约4.21368；gap约3.62410–3.62420',f"{fmt(find('top1',[8,7,5,1])['pic50_mean'])} / {fmt(find('top1',[8,7,5,1])['gap'])}",'AB未超领先者，但B相对A仍有正收益']])
add('汇总提前写明两条主要挑战共享A的迁移风险，也坦承另两案预计落后；这不是完全忽视风险。可改进之处是私有提案阶段比较“A与AB”及“A与单独B”的末轮决策价值。未测单独B不能被报告为必胜方案。')
add('\nTop2 R5 G1提出[1,3,5,2]和[1,3,3,3]，汇总把它们列为归档建议，正确保持剩余采样轮数为0。这些数组没有实际结果，不纳入图表、性能差值或成功率。汇总正确排除早段零端出带方案，但将集中收益、原生分支称为“固定人口噪声”的表述缺少统计依据；这些现象可解释作用路径，不能取消固定人口组均改善。\n')
add('## 9. 匹配对照揭示的关键效应\n')
add('![关键匹配效应](figures/05_matched_effects.png)\n')
add('**图5｜局部效应与背景依赖。** a：Top1匹配背景下早段0.7/0.8/0.9的实测Mean。b：第二窗口增强在两个早段背景下符号相反；B在A背景有用，联合案仍未超原领先者。c：Top2全部20组条件—功能分布，星号为主代表，红色竖线为主带上界；叉号为带外方案。d：Top2匹配边的Mean变化，零早段边既功能不利又出带。均为固定25候选的描述性差值，没有额外重复或因果泛化检验。\n')
table(['Teacher','匹配改动','参照→目标','ΔC Å','ΔMean','目标主带'],[[r['teacher'],r['description'],r['reference']+'→'+r['target'],sign(r['delta_C_A']),sign(r['delta_mean']),'内' if r['target_in_band'] else '外'] for r in contrasts])
add('Top1第二窗口效应的背景差为−0.000390921 pIC50；该差中差说明此局部迁移不能直接沿用旧符号。它不识别完整的第二窗口×末段交互，因为缺少单独B的匹配端点。Top2两个背景的早段零端一处Mean下降、一处Mean上升，但都出条件带；不能把一个背景的功能收益与另一个背景的较低坐标代价拼成同一已测响应。\n')
add('## 10. V17之前改了什么，哪些能力得到支持\n')
add('V16→V17只改写三份通用skills中的两项能力：①分开条件带资格、功能排序、预测反证与各自单位；②依据同背景等步响应、反转、余量和共同外推风险选择步长。保持U1、coarse0.2、q0.1、max_range_probes0、max_changed_intervals2；模型、可微奖励、R0、随机流、采样与接口保持。')
add('\nV17的行为与这些目标部分一致：Top1实测合法内侧0.8并保留新的带内Mean领先；Top2坚持条件带，未让零端更高Mean或高早段的更高Best替代主代表。仍有两项局限：Top1末轮两个可信挑战依赖同一个迁移前提；小幅或集中收益有时被描述成“噪声”而缺少统计支持。版本改善、文字变化和一次行为吻合不能单独证明skills的因果效果；没有无LLM搜索或同任务多次独立搜索对照。')
add('\n当时V17→V18据此改写共同风险与末轮决策价值，以及总效应/条件效应/交互的区分；V18等权gap反而增加0.000728989。这一后续结果也不支持把规则改写视为必然增益。本报告不修改现有skills，不继续实验。\n')
add('## 11. 质量、敏感性、成本与结构保留\n')
add('V17全部40组均25/25有效、PB通过率1、严重clash0；Top1的20组全部在主带，Top2为18/20。化学身份与interaction恢复仍是独立维度：主代表中可获得严格化学映射诊断的候选仅Top1 5/25、Top2 2/25，不能把纯几何低误差当作同一化学模板完整恢复。')
table(['Teacher','主带代表PLIP恢复率均值','主带代表相对R0配对Mean胜率','0.01Å敏感性','0.05Å敏感性'],[
 ['top1',fmt(leaders['top1',5]['plip_recovery']),'0.68','R4G4（不变）','R4G4（不变）'],
 ['top2',fmt(leaders['top2',5]['plip_recovery']),'0.60','R2G2；C=1.906037015，gap=3.254717531','R5G4；C=1.944389154，gap=3.197704334']])
add('PLIP恢复率为单独指纹匹配诊断，不是本报告主RMSD或内层奖励；敏感性结果不替换冻结0.02Å主结论。')
add('\nV17正常完成耗时1861.756092秒（31.029268分钟），正式GPU时间535.428903秒（8.923815分钟）；59份provider-ID去重API收据共3,215,577输入、755,361输出tokens，无本版作废记录或已知usage缺失。这是资源统计，不是货币账单。全实验正常与失败API已分别统计并跨provider ID去重，不能再次把失败分数混入正式结果。')
add('\n已保留各轮评分、坐标误差、质量、LLM及来源记录；完整结构仅按每teacher的逐槽亲和力TOP1策略保留25条。这50条保留结构是多调度拼接归档，不等于两组主代表的完整25+25条原始结构。报告不得据现存TOP1文件重新替代共享调度统计。\n')
add('## 12. 交付文件与复现\n')
table(['文件','用途'],[
 ['[v17_group_metrics.csv](v17_group_metrics.csv)','40组原始聚合指标，包含区域RMSD均值/最小值、Mean/Best、质量与主带资格'],
 ['[v17_round_summary.csv](v17_round_summary.csv)','10个teacher轮次的探索池统计与回顾累计代表'],
 ['[agent_action_ledger.csv](agent_action_ledger.csv)','40条已执行动作说明；仅有证据时标注私有提案来源'],
 ['[matched_contrasts.csv](matched_contrasts.csv)','12条完整匹配端点及实际差值'],
 ['[baseline_comparison.csv](baseline_comparison.csv)','teacher、匹配unguided25、guided R0与V17主代表'],
 ['[campaign_version_trend.csv](campaign_version_trend.csv)','V6–V30主代表趋势及固定提交'],
 ['[evidence.json](evidence.json)','冻结V17报告、有限策略上下文、原始来源hash和对照输入'],
 ['[build_report.py](build_report.py)','从同目录evidence重建CSV、五套图与本文'],
 ['[artifact_manifest.json](artifact_manifest.json)','交付物SHA-256与范围记录'],
 ['[figures](figures/)','五套PNG、SVG、PDF图表'],
 ['[qa_notes.md](qa_notes.md)','数据口径与目视检查记录、证据限制']])
add('复现：在已具有Python、NumPy和matplotlib的环境运行 `python build_report.py`。只聚合已有文件，不访问服务器、不运行模型或测试。图表标签使用英文便于矢量编辑，中文图注解释口径。')
add('\n原始完整总结见[最终报告](../v6-v30-final/report.zh-CN.md)，人口对照修正见[同槽位比较](../v6-v30-followup/slot_matched_comparison.zh-CN.md)，原有限复盘见[V17策略评估](../v17_strategy_review/review.zh-CN.md)。新报告以已完成状态为准，不激活历史计划。')
(HERE/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

manifest={'scientific_commit':D['scientific_commit'],'mode':D['scope']['mode'],
    'full_review_completed':False,'new_sampling':False,'automated_tests_run':False,
    'source_evidence_sha256':hashlib.sha256((HERE/'evidence.json').read_bytes()).hexdigest(),
    'source_report_sha256':D['completion']['finalized']['report_sha256'],
    'figures':5,'formats':['png','svg','pdf'],'files_sha256':{}}
for p in sorted(HERE.rglob('*')):
    if p.is_file() and p.name!='artifact_manifest.json':
        manifest['files_sha256'][p.relative_to(HERE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(HERE/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'groups':len(rows),'teacher_rounds':len(rounds),'contrasts':len(contrasts),'figures':5,'report':str(HERE/'report.zh-CN.md')},ensure_ascii=False))
