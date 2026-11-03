"""Recover postprocessing from complete frozen 2.14 rounds without inference."""
import argparse
import fcntl
import math
from pathlib import Path
import subprocess

from agenticgen.uniguide import condition_evaluation as fitted
from agenticgen.uniguide.absolute_runtime import retain
from agenticgen.uniguide.checkpoint_learning import best_from_calibration
from agenticgen.uniguide.geometry_sampling import validate_receipt
from agenticgen.uniguide.io import read, write, frozen, sha
from agenticgen.uniguide.linear_scale import EFFECTIVE, validate_domain


SOURCE_COMMIT = '52e32a1a1451928bb87dcc370938bd1ae6635970'


def recover_job(plan, job, output):
    root = Path(job['root'])
    result_path = Path(plan['root'])/'job_results'/(job['id']+'.json')
    if not result_path.exists():
        return dict(job_id=job['id'],status='pending_original_job')
    outcome = read(result_path)
    if outcome['status'] == 'completed':
        return dict(job_id=job['id'],status='original_job_completed')
    log = Path(plan['root'])/'logs'/job['id']/'formal.log'
    text = log.read_text(errors='replace')
    if ('retainer(root, current, winner' not in text or
            'No complete eligible finite geometry measurements' not in text):
        raise ValueError('Original failure is outside the supported retention repair')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'COMPLETE.json').exists():
        receipt = read(output/'COMPLETE.json')
        for name,digest in receipt['files'].items():
            if sha(output/name) != digest: raise ValueError('Recovered artifact changed')
        return read(output/'RESULT.json')
    ledger_before = sha(root/'attempts.json')
    rounds = [read(p) for p in sorted((root/'versions/GD01'/job['teacher']).glob('advice_*/ROUND_COMPLETE.json'))]
    if len(rounds) != 5 or [r['iteration'] for r in rounds] != list(range(1,6)):
        raise ValueError('Recovery requires all five complete original rounds')
    for row in rounds:
        if not row['hypothesis_execution']['complete']:
            raise ValueError('Recovery cannot replace failed hypotheses')
        for path,digest in row['source_receipts'].items():
            if sha(root/path/'COMPLETE.json') != digest: raise ValueError('Round source changed')
        for path,digest in row['advice_files'].items():
            if sha(root/path) != digest: raise ValueError('Round advice changed')
    groups = []
    for path in sorted((root/'measurement_registry').glob('*.json')):
        source = root/read(path)['source_group']
        group = validate_receipt(source)
        value = fitted.score(group)
        records = read(source/'records.json')
        if (len(records) != 50 or len({(r['logical_seed'],r['logical_slot']) for r in records}) != 50
                or any(not fitted.enabled(r) or not math.isfinite(r[fitted.METRIC]) for r in records)
                or not math.isclose(sum(r[fitted.METRIC] for r in records)/50,value,rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError('Frozen candidates do not reproduce the complete aligned mean')
        if (source/'proposal.json').exists():group['proposal']=read(source/'proposal.json')
        groups.append(group)
    domain = read(root/'calibration/domains.json')[job['teacher']]
    validate_domain(domain)
    winner = best_from_calibration(groups,domain,rounds)
    original_best = read(root/'versions/GD01'/job['teacher']/'GLOBAL_BEST.json')
    if winner['id'] != original_best['representative']:
        raise ValueError('Recovered winner differs from the published original checkpoint')
    formal = [g for row in rounds for g in row['groups']]
    if any(g['id'] not in {r['id'] for r in groups} for g in formal):
        raise ValueError('Formal round measurements are absent from frozen history')
    budget = read(root/'attempts.json')
    used = {stage:sum(r['count'] for r in budget.values() if r['stage']==stage)
            for stage in ['calibration','formal']}
    if used['calibration'] > 500 or used['formal'] > 1250 or len(formal)*50 != used['formal']:
        raise ValueError('Original budget does not match the completed populations')
    if list(root.rglob('synthesis.receipt.json')) or list(root.rglob('box_selection.receipt.json')):
        raise ValueError('A disabled model role was called')
    retain(root,groups,winner,output/'retention')
    baseline = next(g for g in groups if g['id']==domain['initial_source'])
    report = dict(job_id=job['id'],status='postprocessing_recovered',sampling_commit=SOURCE_COMMIT,
        repair_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        root=str(root),condition_evaluation=domain[fitted.CONFIG],completed_rounds=5,
        complete=True,formal_groups=len(formal),measured_populations=len(groups),attempts=used,
        representative=winner['id'],best_actual_schedule=winner[EFFECTIVE],
        best_mean=fitted.score(winner),calibration_root=baseline['id'],
        calibration_mean=fitted.score(baseline),
        ranking_measurement={k:winner[k] for k in ['id','complete','attempts',fitted.CONFIG,fitted.MEAN,
            'coordinate_loss_statistics','ranking_condition_coordinate_root_mean_square_deviation_angstrom']},
        original_failure_sha256=sha(root/'CAMPAIGN_FAILED.json'),
        original_job_result_sha256=sha(result_path),original_ledger_sha256=ledger_before,
        raw_measurements_reused=True,new_sampling_attempts=0,new_model_calls=0,
        round_receipts={str(p.relative_to(root)):sha(p) for p in
            sorted((root/'versions/GD01'/job['teacher']).glob('advice_*/ROUND_COMPLETE.json'))},
        population_receipts={g['id']:sha(root/g['id']/'COMPLETE.json') for g in groups})
    if sha(root/'attempts.json') != ledger_before:
        raise ValueError('Budget ledger changed during postprocessing')
    frozen(output/'RESULT.json',report)
    frozen(output/'ONLINE_PROTOCOL_AUDIT.json',dict(complete=True,rounds=5,
        complete_populations=len(groups),formal_groups=len(formal),disabled_roles_absent=True,
        condition_evaluation=domain[fitted.CONFIG],new_sampling_attempts=0))
    frozen(output/'COMPLETE.json',dict(files={str(p.relative_to(output)):sha(p)
        for p in output.rglob('*') if p.is_file()}))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True)
    args=parser.parse_args()
    suite=Path(args.suite).resolve();plan=read(suite/'PLAN.json')
    if plan['commit'] != SOURCE_COMMIT:raise ValueError('Unsupported source experiment')
    repair_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    output=suite/'postprocessing_recovery'/repair_commit
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=[]
        for job in plan['jobs']:
            write(output/'STATUS.json',dict(stage='validating_saved_job',job_id=job['id'],results=results,
                  new_sampling_attempts=0,new_model_calls=0))
            result=recover_job(plan,job,output/job['id']);results.append(result)
            write(output/'STATUS.json',dict(results=results,new_sampling_attempts=0,new_model_calls=0))
        print(str(output/'STATUS.json'))


if __name__=='__main__':main()
