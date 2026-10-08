"""Register a fresh sequential, rollback-controlled twenty-round experiment."""
import argparse
import gzip
import json
import shutil
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.selection_workflow import restore_incumbent
from evomolsteer.generation.path_evaluation import summarize_tail


def prepare(repo):
    root=Path(repo).resolve();cfg=root/'configs/experiments/selection_path20_v1';docs=root/'docs/experiments/selection_path20_20261008'
    if (cfg/'campaign.json').exists():raise FileExistsError('Registered campaign already exists')
    cfg.mkdir(parents=True,exist_ok=True);docs.mkdir(parents=True,exist_ok=True)
    skills={role:{'path':f'skills/{role}/SKILL.md','sha256':digest(root/f'skills/{role}/SKILL.md')} for role in ['analyst','designer']}
    paths=['endpoint_reward.py','affinity_geometry_reward.py','coordinate_reward.py','local_reward.py','multistage_reward.py','scalar_guidance.py']
    write_json(cfg/'baseline_snapshot.json',{'program_path':'configs/experiments/skill_ablation_v1/incumbent.json',
      'program_sha256':digest(root/'configs/experiments/skill_ablation_v1/incumbent.json'),'baseline_skills':skills,
      'baseline_algorithm_files':{f'src/evomolsteer/generation/{p}':digest(root/f'src/evomolsteer/generation/{p}') for p in paths}})
    restore_incumbent(root,'selection_path20_v1','Initial immutable R26 default',0)
    write_json(cfg/'campaign.json',{'maximum_rounds':20,'rounds_completed':0,'rounds':[]})
    protocol={'schema_version':'selection-path20-protocol-1.0','master_seed':42,'screening_batches':[41,42],
      'confirmation_batches':[43,44,45,46],'maximum_rounds':20,'n_per_arm':100,
      'controls':'Matched initial coordinate/atom/bond hashes and same RNG; native and R26 per panel. Historical completed Steer is an unpaired reference, not a randomized equal-compute control.',
      'screening_acceptance':'All-attempt paired mean improvement over R26 >0.005 pIC50, positive in both batches, validity and PB-fast >=0.90; provisional only.',
      'confirmation_acceptance':'Frozen four-batch paired mean versus R26 >0.01 and bootstrap lower bound >0, native improvement >0; validity/PB no more than 0.03 below R26, strain median <=1.3*max(R26 median, historical Steer median). No automatic wet-lab claim.',
      'freeze':'Choose best admissible screen mean before confirmation labels. If none admissible, confirm best exploratory mean as an explicitly unaccepted hypothesis; default remains R26.',
      'rollback':'Restore byte-exact R26 active program, baseline Skills selector and baseline numerical source checks after every screen/failed confirmation. Research modules and their commits remain archived but disabled.',
      'round_schedule':{'1':'Fresh native/R26 screening controls','2-16':'Sequential single-mechanism tests, always anchored in R26 unless a screen passes predeclared criteria','17':'New native/R26 controls A','18':'Frozen candidate A','19':'New native/R26 controls B','20':'Same frozen candidate B'},
      'inference':'100 native integration steps, guidance on dynamically declared support only, then native to final; no particle selection, affinity-head gradient, or extra production model forward.',
      'scope':'Coordinate-only hypotheses; no chemical-graph equality gate. All posterior priors/correspondences conditioned on a frozen model endpoint.',
      'tail_threshold_pic50':8.258901977539063,'tail_threshold_source':'Previously frozen donor-valid p95, not optimized on new outcomes',
      'retention':'Verify and publish reports before deleting local/remote generated structures; original Steer and checkpoints protected.'}
    write_json(docs/'protocol.json',protocol)
    mine=root/'results/selection_path20/mining';target=docs/'mining';target.mkdir()
    for p in mine.iterdir():shutil.copy2(p,target/p.name)
    events=pd.read_parquet(mine/'events.parquet');effects=pd.read_parquet(mine/'effects.parquet')
    niches=read_json(mine/'niche_audit.json')
    summary={'event_count':len(events),'batches':14,'window':read_json(mine/'manifest.json')['window'],
      'expected_ess_fraction_mean':float(events.expected_ess_fraction.mean()),'expected_ess_fraction_min':float(events.expected_ess_fraction.min()),
      'realized_ess_fraction_mean':float(events.realized_ess_fraction.mean()),
      'mean_score_gain_from_expected_selection':float(events.score_selection_covariance.mean()),
      'root_count_first_mean':float(events[events.time==events.time.min()].surviving_roots.mean()),
      'root_count_last_mean':float(events[events.time==events.time.max()].surviving_roots.mean()),
      'q_significant_by_term':effects.groupby('term').q.apply(lambda v:int((v<.05).sum())).to_dict(),
      'niche_effective_n_min':min(v['effective_n'] for f in niches['frames'] for v in f['niches']),
      'top_expected_selection_features':effects[effects.term.eq('expected_selection')].sort_values('q').head(10).to_dict('records'),
      'limitations':'Expected probabilities audit the same selector; low mode ESS is regularized, not independent evidence. Realized copying can narrow roots even with mild one-step expected pressure.'}
    write_json(docs/'mining_summary.json',summary)
    historical=root/'docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv'
    d=pd.read_csv(historical);threshold=protocol['tail_threshold_pic50']
    write_json(docs/'steer_reference.json',{'source_path':historical.relative_to(root).as_posix(),'source_sha256':digest(historical),
      'all':summarize_tail(d,threshold),'donors':summarize_tail(d[d.batch<14],threshold),
      'non_donors':summarize_tail(d[d.batch>=14],threshold),
      'limitation':'Unpaired historical reference; donor data informs design and Steer has additional scoring/resampling compute.'})
    failure={'sources':{p:digest(root/p) for p in ['docs/experiments/elite_path20_20261007/report.zh-CN.md',
        'docs/experiments/dynamic_contrast10_20261008/report.zh-CN.md']},
      'measured_failures':['Terminal-descendant teacher replacement did not beat R26 independently.',
        'Background-subtracted potentials, doubling dose, and broad teacher expansion degraded mean affinity.',
        'Seven/eight marginal spatial-field contrast did not beat R26; increasing its coefficient worsened affinity.',
        'Hard reweighting can lower final control ESS below a partition-stage floor.'],
      'design_response':'Preserve R26 labels and full endpoint modes. Change prior, coordinate niche support or local SPD metric in isolation. Native-future calibration remains unavailable, never relabel censored branches as failure.'}
    write_json(docs/'failure_ledger.json',failure)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');a=p.parse_args();print(prepare(a.repo))
