"""Run the 2.14 CPU gate and the existing serial two-teacher online campaign."""
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

from agenticgen.uniguide.io import read, write, frozen, sha
from agenticgen.uniguide import condition_evaluation as fitted


CPU_TESTS = [
    'test_early_path_search', 'test_diverse_box_search', 'test_seven_box_search',
    'test_uniguide_geometry_batch', 'test_geometry_chemical_state', 'test_adaptive_calibration',
    'test_linear_guidance', 'test_coordinate_statistics', 'test_guidance_analysis_tools',
    'test_strict_final_submission', 'test_absolute_coordinate_protocol', 'test_dynamic_matching_jump',
    'test_open_domain_search', 'test_zero_start_guidance', 'test_synchronized_chemistry', 'test_deepseek_tools',
]


def load_runner():
    directory = Path(__file__).resolve().parents[1]/'20261031_guidance_comparison5'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('early_path_serial_runner', directory/'run_suite.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    original_sources, original_job = common.quality_sources, common.run_job

    def quality_sources(config):
        sources = set(original_sources(config))
        for path in (Path(config['root'])/'calibration/path').glob('*/*/COMPLETE.json'):
            row = read(path.parent/'aggregate.json')
            if row.get('complete') and row.get('attempts') == 50:
                sources.add(row['id'])
        return sorted(sources)

    def run_job(repo, suite, job):
        config = read(job['config'])
        if (config['agent_tools']['skill_release'] != '2.14.0'
                or config.get(fitted.CONFIG) != {'policy': fitted.POLICY}
                or config['box_exclusion'] != {'enabled': False}):
            raise ValueError('Online job differs from the 2.14 launch protocol')
        result = original_job(repo, suite, job)
        root = Path(config['root'])
        final = read(root/'versions/GD01/FINALIZED.json')
        if not final['complete'] or final['completed_rounds'] != 5:
            raise ValueError('Online search has failed hypotheses or incomplete rounds')
        score = fitted.score(final['teachers'][job['teacher']]['ranking_measurement'])
        measurements, failed_calibration = [], []
        for path in (root/'measurement_registry').glob('*.json'):
            source = root/read(path)['source_group']
            if not (source/'COMPLETE.json').exists():
                if source.is_relative_to(root/'calibration'):
                    failed_calibration.append(source.relative_to(root).as_posix())
                    continue
                raise ValueError('Online population has a failed or interrupted formal claim')
            row, records = read(source/'aggregate.json'), read(source/'records.json')
            value = fitted.score(row)
            if len(records) != 50 or not math.isclose(
                    sum(r[fitted.METRIC] for r in records)/50, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Candidate measurements differ from the complete aligned mean')
            measurements.append(dict(source_id=row['id'], mean=value, attempts=50))
        if list(root.rglob('synthesis.receipt.json')) or list(root.rglob('box_selection.receipt.json')):
            raise ValueError('A disabled model role was called')
        frozen(root/'ONLINE_PROTOCOL_AUDIT.json', dict(
            release='2.14.0', condition_evaluation=config[fitted.CONFIG], best_mean=score,
            complete_populations=measurements, failed_calibration=failed_calibration,
            formal_groups=final['measured_or_reused_groups'],
            rounds=5, disabled_roles_absent=True, final_sha256=sha(root/'versions/GD01/FINALIZED.json')))
        return dict(result, condition_aligned_mean=score, online_protocol_audited=True)

    common.quality_sources, common.run_job = quality_sources, run_job
    return common


def main():
    common = load_runner()
    plan_path = Path(sys.argv[sys.argv.index('--plan')+1])
    plan = read(plan_path)
    suite, repo = Path(plan['root']), Path(plan['repo'])
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != plan['commit']:
        raise ValueError('CPU preflight checkout differs from the fixed launch commit')
    frozen(suite/'CPU_PREFLIGHT_CLAIM.json', dict(pid=os.getpid(), commit=actual, plan_sha256=sha(plan_path)))
    write(suite/'STATUS.json', dict(stage='cpu_preflight', status='running', pid=os.getpid()))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-m', 'pytest', *['tests/unit/'+name+'.py' for name in CPU_TESTS],
               '-q', '--junitxml='+str(suite/'cpu_preflight.xml')]
    try:
        with (suite/'cpu_preflight.log').open('xb') as stream:
            completed = subprocess.run(command, cwd=repo/'agenticgen', env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        frozen(suite/'CPU_PREFLIGHT_EXIT.json', dict(returncode=completed.returncode,
               log_sha256=sha(suite/'cpu_preflight.log'), commit=actual))
        if completed.returncode:
            raise RuntimeError('CPU preflight failed before online inference')
        common.main()
    except Exception as error:
        write(suite/'SUITE_FAILED.json', dict(error_type=type(error).__name__, error=str(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
