"""Compact diagnostics from retained reports; never infer another head score.

Separate dose, cancellation, instantaneous-window labels and final decoded
labels. This script does not infer a causal reason from a one-factor-looking
adaptive comparison or tune on frozen validation results.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.generation.affinity_reporting import paired_head


def diagnose(evidence, output, through=None):
    evidence, output = Path(evidence), Path(output)
    if output.exists(): raise FileExistsError(output)
    outcomes = sorted(evidence.glob('round_*.outcome.json'))
    baseline_path = evidence/'round_01/local/candidate_metrics.csv'
    baseline = pd.read_csv(baseline_path)
    sources = {baseline_path.as_posix(): digest(baseline_path)}
    batches, rounds = [], []
    for path in outcomes:
        r = read_json(path)
        n = r['round']
        if through is not None and n > through: continue
        local = evidence/f'round_{n:02d}/local'
        metric_path = local/'candidate_metrics.csv'
        actual = pd.read_csv(metric_path)
        native = actual if actual.arm.eq('unguided').any() else baseline
        pairs = paired_head(actual, native)
        program_path = local/'inference_config/reward_program.json'
        program = read_json(program_path)
        response_path = local/'guidance_response.json'
        response = read_json(response_path)
        window_path = local/'paired_head_window.parquet'
        window = pd.read_parquet(window_path) if window_path.exists() else None
        for p in (path, metric_path, program_path, response_path, window_path):
            if p.exists(): sources[p.as_posix()] = digest(p)
        dose = {d['batch']: d for d in response['dose']}
        for batch, group in pairs.groupby('batch'):
            signed, absolute = float(group.delta.mean()), float(group.delta.abs().mean())
            row = dict(round=n, split=r['split'], batch=int(batch), n=len(group),
                final_decoded_head_delta=signed, final_absolute_head_delta=absolute,
                final_positive_fraction=float(group.delta.gt(0).mean()),
                final_cancellation_fraction=1-abs(signed)/absolute if absolute > 0 else 0.,
                nominal_dose=program['native_rms_ratio'],
                measured_mean_injection_A=dose[batch]['mean_injection_A'],
                measured_mean_path_A=dose[batch]['mean_path_A'],
                window_instantaneous_head_delta=None, last_guided_score_time=None,
                final_minus_last_window_delta=None)
            if window is not None:
                g = window[window.batch.eq(batch)].sort_values('score_time')
                if len(g):
                    last = g.iloc[-1]
                    row.update(window_instantaneous_head_delta=float(last.mean_delta),
                        last_guided_score_time=float(last.score_time),
                        final_minus_last_window_delta=signed-float(last.mean_delta))
            batches.append(row)
        rounds.append(dict(round=n, split=r['split'], reward_view=r['reward_view'],
            final_head_delta=float(pairs.delta.mean()), final_absolute_head_delta=float(pairs.delta.abs().mean()),
            nominal_dose=program['native_rms_ratio'],
            reference_variant=program.get('reference_variant','instantaneous_or_legacy_coordinate_library'),
            teacher_neighbors=program.get('teacher_neighbors'), teacher_score_beta=program.get('teacher_score_beta'),
            teacher_temperature_A2=program.get('teacher_endpoint_temperature_A2'),
            region_radius_A=program.get('coordinate_region_radius_A'),
            time_ramp_power=program.get('time_ramp_power'),
            reference_sha256=program['reference_sha256'],
            interpretation='Descriptive paired effects; window and final head semantics differ'))
    if not rounds: raise ValueError('No completed retained rounds')
    output.mkdir(parents=True)
    rt, bt = pd.DataFrame(rounds), pd.DataFrame(batches)
    rt.to_parquet(output/'round_diagnostics.parquet',compression=None,index=False)
    bt.to_parquet(output/'batch_diagnostics.parquet',compression=None,index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    colors = ['#3264a0' if s=='discovery' else '#dd8b20' for s in rt.split]
    axes[0].bar(rt['round'],rt.final_head_delta,color=colors,label='Signed final decoded-head delta')
    axes[0].plot(rt['round'],rt.final_absolute_head_delta,'o-',color='#777',label='Mean absolute paired delta')
    axes[0].axhline(0,color='black',lw=.7);axes[0].set_ylabel('pIC50 difference');axes[0].legend()
    summary=bt.groupby('round')[['final_decoded_head_delta','window_instantaneous_head_delta']].mean()
    axes[1].plot(summary.index,summary.final_decoded_head_delta,'o-',label='Final decoded rescore')
    axes[1].plot(summary.index,summary.window_instantaneous_head_delta,'s--',label='Last guided instantaneous joint-latent head')
    axes[1].axhline(0,color='black',lw=.7);axes[1].set_xlabel('Fresh campaign round');axes[1].set_ylabel('Paired mean delta');axes[1].legend()
    axes[0].set_title('Active control can cancel in the mean; head readouts differ across stages')
    for ax in axes: ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'response_diagnostics.png',dpi=180)
    fig.savefig(output/'response_diagnostics.svg');plt.close(fig)
    manifest=dict(source_code_sha256=digest(__file__),sources=sources,rounds=len(rt),
        storage='Round and independent-batch summaries only; original candidate score tables reused',
        conclusions=['A small signed head response alone does not imply inactive control',
            'Overall reward scaling cancels under normalized-gradient control; dose and relative geometry matter',
            'Window-to-final label changes combine native continuation and different head inputs, not an identified decay mechanism',
            'Adaptive discovery contrasts are descriptive; frozen validation is not retuned'],
        output_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    p.add_argument('--through',type=int)
    a=p.parse_args();print({'rounds':diagnose(a.evidence,a.output,a.through)['rounds']})
