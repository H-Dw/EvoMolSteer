"""Render the measured Luna screen and independent confirmation, locally."""
import argparse,time,subprocess
from pathlib import Path
from evomolsteer.io import read_json,write_json

def render(repo):
 s=repo/'docs/experiments/steer_dependency_20261009/luna_suite'
 qpath=repo/'results/steer_dependency_20261009/luna/screen/quality_summary.json';q=read_json(qpath)
 complete=(s/'confirmation_complete.json').exists()
 lines=['# GPT6-Luna：历史 Steer 依赖消融比较','',f"状态：{'筛查与独立确认均完成' if complete else '实验进行中；以下仅报告已经测量的结果'}。所有新角色调用显式使用 GPT6-Luna，26 次 Analyst / Designer 调用；旧模型队列已取消。",'', '## 已测量的筛查结果','', '每组 100 次生成，共用批次 75/76。I0 与冻结 R26 数值等价，因此复用同一测量，不是独立重复。','', '|条件|平均预测 pIC50|相对 R26|PB/N|应变中位数/重原子|应变 P90|精英有效数|','|---|---:|---:|---:|---:|---:|---:|']
 baseline=q['outcomes']['R26']['affinity_mean_all']
 for name,x in q['outcomes'].items():
  lines.append(f"|{name}|{x['affinity_mean_all']:.5f}|{x['affinity_mean_all']-baseline:+.5f}|{x['pb_fast_pass']}/{x['n']}|{x['strain_median_per_heavy']:.5f}|{x['strain_p90_per_heavy']:.5f}|{x['elite_valid']}|")
 missing=[r['condition'] for r in read_json(s/'protocol.json')['conditions'] if r['condition'] not in q['outcomes']]
 lines+=['', '未完成的主条件：'+(', '.join(missing) if missing else '无')+'。', '', '## 独立确认','']
 if complete:
  c=read_json(s/'confirmation_complete.json');cq=c['quality'];winner=c['selected_condition'];lines+=['筛查选中 `'+winner+'`，在未用于设计/选择的批次 77–82 各生成 300 个分子。','', '|条件|平均预测 pIC50|PB/N|应变中位数/重原子|精英有效数|','|---|---:|---:|---:|---:|']
  for name,x in cq['outcomes'].items():lines.append(f"|{name}|{x['affinity_mean_all']:.5f}|{x['pb_fast_pass']}/{x['n']}|{x['strain_median_per_heavy']:.5f}|{x['elite_valid']}|")
  lines+=['','按 batch 成对 bootstrap 的差异与区间保存在 `confirmation_quality.json`；不将仅六批的确认结果外推为跨目标泛化。']
  write_json(s/'confirmation_quality.json',cq)
 else:lines+=['预注册的六个新批次确认尚未全部完成，当前不提供其性能结论。']
 lines+=['','## 解释','', '本实验分别删除文字指引与历史数据权限。I 条件仍读取压缩的 Steer 统计；D1/D2 仍使用历史选中结构。只有 D3/D4 及其匹配剂量控制的 Agent 输入与奖励不读取生成的 Steer 数据。D4 的口袋裁剪和标准 FLOWR 输入仍继承原始结合结构，不能宣称完全无配体先验。', '', '各条件 Designer 同时改变函数族、系数和剂量，因此性能变化不能单独归因于被删除的一句话。重复调用产生不同方案，进一步显示模型输出变异。当前没有依据自动改写或替换历史 R26 默认配置。', '', '实测存在归一化抵消单项系数变化的情况：锚定权重不同，但接触权重为零、排斥项始终未激活时，保存的中间状态与最终输出完全一致。单独改数字不能保证引导改变。接触比例与 eta/ramp 则可能改变方向与剂量。', '', '历史 Steer 1000 个样本的平均预测 pIC50 为 7.65148，有效 961/PB960，应变中位数约 .53032，固定精英阈值以上有效样本 37。它是不同预算、不成对的参考；不可按不同样本量的最大值直接判优。', '', '原始结果：`screen_quality.json`、`confirmation_quality.json`（确认完成后生成）、`transport_records/`。方向、依据及实际程序：`agent_reasoning_evidence.json`、各条件的 `Analyst/Designer.response.json`、`compiled.json`。', '', '完整方法、数学函数、梯度路径和限制见 [实验设计与解释边界](study_design_and_limits.zh-CN.md)。实际指令交付与模型核验见 `agent_input_audit.json` 和 `role_model_ledger.json`。', '']
 (s/'analysis.zh-CN.md').write_text('\n'.join(lines),encoding='utf-8');write_json(s/'screen_quality.json',q)
 subprocess.run([str(repo/'.venv/Scripts/python.exe'),str(repo/'scripts/export_dependency_results.py'),'--study',str(s),'--quality',str(qpath),'--output',str(s/'tables')],check=True,cwd=repo)
 return complete

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--watch-seconds',type=int,default=0);p.add_argument('--commit-on-complete',action='store_true');a=p.parse_args();repo=Path(a.repo);deadline=time.monotonic()+a.watch_seconds
 while True:
  complete=render(repo)
  if complete:
   if a.commit_on_complete:
    subprocess.run(['git','add','docs/experiments/steer_dependency_20261009/luna_suite'],cwd=repo,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
    if any(not x.startswith('docs/experiments/steer_dependency_20261009/luna_suite/') for x in staged):raise ValueError('Unrelated staged changes; final report saved, commit deferred')
    if staged:
     subprocess.run(['git','commit','-m','Record completed Luna dependency ablations and independent confirmation'],cwd=repo,check=True)
     subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main'],cwd=repo,check=True)
   print('LUNA_COMPARISON_REPORT_COMPLETE',flush=True);break
  if time.monotonic()>=deadline:print('LUNA_COMPARISON_REPORT_PARTIAL',flush=True);break
  time.sleep(min(60,deadline-time.monotonic()))
