"""Predeclared single-factor rollback hypotheses within the existing 30-round cap.

This is a bounded sensitivity campaign, not a learned optimizer or success proof.
Every actual inference result is retained before the next hypothesis is frozen.
"""
from pathlib import Path
import math
from ..io import read_json,digest


def outcome_record(path):
    """Normalize legacy reports without editing their immutable original files."""
    path=Path(path);v=read_json(path)
    if not isinstance(v,dict) or not isinstance(v.get('round'),int) or 'shape_improvement_fraction' not in v:
        raise ValueError('Complete outcome identity and geometry evidence required')
    if 'all_head_change_vs_native' not in v:
        native=read_json(path.parent/'round_01/local/terminal_report.json')['results']['unguided']
        terminal=read_json(path.parent/f"round_{v['round']:02d}/local/terminal_report.json")['results']['gradient']
        v={**v,'all_head_change_vs_native':terminal['all_pic50_on_rescore_mean']-native['all_pic50_on_rescore_mean'],
            'MMFF_relief_relative_change':terminal['all_mmff_relief_per_heavy_median']/native['all_mmff_relief_per_heavy_median']-1,
            'surround_RMS_improvement_fraction':1-terminal['all_relax_rms_surround_A_mean']/native['all_relax_rms_surround_A_mean']}
    return v


def proposal(number,evidence):
    if not 15<=number<=30:raise ValueError('Continuation covers rounds15--30 only')
    evidence=Path(evidence)
    if not (evidence/f'round_{number-1:02d}.outcome.json').is_file():
        raise ValueError('Preceding scientific result must be retained before choosing')
    previous=outcome_record(evidence/f'round_{number-1:02d}.outcome.json')
    if previous.get('round')!=number-1 or not math.isfinite(previous['shape_improvement_fraction']):
        raise ValueError('Complete preceding outcome required')
    fixed={
        15:(3,{'mixture_temperature':.05},'Sharper mode choice tests gradient blending; offline cosine changes in ~18% of native discovery particle-nodes.'),
        16:(3,{'mixture_temperature':1.},'Restore R3 and test broad mode averaging as the paired temperature alternative.'),
        17:(3,{'native_rms_ratio':.005},'Restore R3; test one-tenth dose to leave startup cap saturation and reduce categorical path disruption.'),
        18:(3,{'constraints.max_atom_step_A':.0125},'Restore R3; tighten every atom cap, unlike R12 which tightened only the first update.'),
        19:(13,{'native_rms_ratio':.025},'Restore directional shape R13 and halve predictive-flow dose to test its terminal strain tradeoff.'),
        20:(13,{'mixture_temperature':.05},'Restore R13; sharpen directional shape mode choice while keeping dose unchanged.'),
        21:(14,{'native_rms_ratio':.025},'Restore conditional shape R14 and halve dose without relaxing sparse-count support.'),
        22:(14,{'mixture_temperature':.05},'Restore R14; sharpen within-count geometry mixture, retaining all support thresholds.'),
        23:(4,{'native_rms_ratio':.1},'Restore joint predictive-flow R4; double its continuous small-step dose to test whether geometry was under-controlled.'),
        25:(3,{'robust_delta':.25},'Restore R3; increase robust saturation, changing relative mode influence rather than introducing an energy force.'),
        26:(3,{'robust_delta':4.},'Restore R3; compare weaker robust saturation to the strong-curvature alternative.'),
        28:(1,{'native_rms_ratio':.05},'Restore broad regional spread R1 at one-quarter dose as a representation control.'),
        29:(2,{'native_rms_ratio':.05},'Restore global spread R2 at one-quarter dose to separate common compaction from regional imitation.')}
    if number in fixed:return fixed[number]
    if number==24:
        previous=read_json(evidence/'round_23.outcome.json')
        acceptable=(all(previous.get(k) is not None for k in ('all_head_change_vs_native','MMFF_relief_relative_change','surround_RMS_improvement_fraction')) and
                    previous['all_head_change_vs_native']>=0 and previous['MMFF_relief_relative_change']<=0 and previous['surround_RMS_improvement_fraction']>=0)
        return (4,{'native_rms_ratio':.2 if acceptable else .01},
            'Restore R4; further increase dose only when R23 preserves head and both physical proxies, otherwise explore a smaller dose.')
    outcomes=[outcome_record(p) for p in evidence.glob('round_*.outcome.json')]
    if number==27:
        eligible=[v for v in outcomes if v['round']<number and v.get('all_head_change_vs_native') is not None and v['all_head_change_vs_native']>=0]
        chosen=max(eligible,key=lambda v:v['shape_improvement_fraction']) if eligible else outcome_record(evidence/'round_03.outcome.json')
        return chosen['round'],{'native_rms_ratio':.002},'Restore the best observed nonnegative-head geometry parent; test very small dose. Development selection is not independent validation.'
    eligible=[v for v in outcomes if v['round']<number and all(v.get(k) is not None for k in ('all_head_change_vs_native','MMFF_relief_relative_change','surround_RMS_improvement_fraction')) and v['shape_improvement_fraction']>0 and
              v['all_head_change_vs_native']>=0 and v['MMFF_relief_relative_change']<=0 and v['surround_RMS_improvement_fraction']>=0]
    chosen=max(eligible,key=lambda v:v['shape_improvement_fraction']) if eligible else outcome_record(evidence/'round_03.outcome.json')
    return chosen['round'],{'native_rms_ratio':.001},'Final small-dose check from a joint-outcome eligible parent, or restored R3 if none qualifies; do not declare success from proxy-only improvement.'


def resolve_parent(config_dir,parent_round,evidence=None):
    cfg=Path(config_dir)
    if parent_round==3:program=cfg/'endpoint_anchor_round03/program.json'
    elif parent_round in (1,2,4):program=cfg/f'round_{parent_round:02d}.json'
    elif parent_round==5:program=cfg/'endpoint_anchor_round05/program.json'
    else:program=cfg/f'backtrack_round{parent_round:02d}.json'
    p=read_json(program)
    references=[r for r in cfg.rglob('*.gz') if digest(r)==p['reference_sha256']]
    if len(references)!=1:raise ValueError('Parent requires one exact preserved reference')
    if evidence is not None:
        local=Path(evidence)/f'round_{parent_round:02d}/local'
        actual=local/'inference_config/reward_program.json';audit=read_json(local/'coordinate_audit.json')
        if read_json(actual)!=p or digest(actual)!=audit['program_sha256'] or digest(references[0])!=audit['reference_sha256']:
            raise ValueError('Parent must match the actual retained inference program/reference')
    return program,references[0]


def regression_triggers(outcome):
    fields={'shape_improvement_fraction':lambda v:v<0,'all_head_change_vs_native':lambda v:v<0,
        'MMFF_relief_relative_change':lambda v:v>0,'surround_RMS_improvement_fraction':lambda v:v<0}
    return [('missing:'+name) if outcome.get(name) is None else name for name,test in fields.items()
            if outcome.get(name) is None or test(outcome[name])]


def validate_retention(local,expected_rows=None):
    """Verify complete report bytes before deleting their underlying structures."""
    local=Path(local).resolve();manifest=read_json(local/'retention.json')
    if manifest['candidate_rows']<=0 or (expected_rows is not None and manifest['candidate_rows']!=expected_rows):
        raise ValueError('Failure-inclusive candidate count mismatch')
    files={r['path']:r for r in manifest['files']}
    required={'terminal_report.json','candidate_metrics.csv','execution_report.json','coordinate_audit.json','window/report.json'}
    if required-set(files):raise ValueError('Incomplete retained evidence')
    for name,item in files.items():
        path=(local/name).resolve()
        if not path.is_relative_to(local) or not path.is_file() or path.stat().st_size==0 or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('Retained report checksum/size/path mismatch')
    for name in required-{'candidate_metrics.csv'}:
        if not read_json(local/name):raise ValueError('Empty retained JSON')
    return digest(local/'retention.json')
