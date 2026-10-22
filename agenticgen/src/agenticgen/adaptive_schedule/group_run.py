"""Durable formal population-council workflow; no legacy twenty-slot fallback."""
import concurrent.futures
import copy
import multiprocessing
import os
from pathlib import Path
import shutil
import time

from .io import read, write, sha, value_sha, source_fingerprint, repository, commit, eligible
from .group_protocol import configuration, GROUP_ROOT
from .grid import CHECKPOINT_STEPS
from .group_tools import score_summary, paired, interval_summary
from .contracts import earliest
from .protocol_state import load_config, config_path, prepared_path, acceptance_path
from .search_policy import effective_config
from agenticgen.runtime.paths import skill

SKILLS = {'analyst': 'agenticgen/src/agenticgen/resources/skills/plip-group-analyst/SKILL.md',
          'synthesizer': 'agenticgen/src/agenticgen/resources/skills/plip-group-synthesizer/SKILL.md'}


def prepare(root, source):
    root, source = Path(root), Path(source)
    c = configuration()
    if (root/'PREPARED.json').exists():
        assert_prepared(root)
        return read(prepared_path(root))
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use an empty new result directory')
    teachers = [t for t in read(source/'teachers.json') if t['rank'] in (1, 2)]
    from .cohort import select_templates
    teachers = select_templates(teachers, c['teacher_selection'])
    from .learning import expand_template
    teachers = [expand_template(t) for t in teachers]
    refs = read(source/'historical_references.json')
    for name, digest in refs.items():
        if sha(name) != digest:
            raise ValueError('Historical reference changed: '+name)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source/'baselines', root/'baselines', dirs_exist_ok=True)
    write(root/'config.json', c); write(root/'teachers.json', teachers)
    write(root/'historical_references.json', refs)
    for role, relative in SKILLS.items():
        shutil.copyfile(skill('plip-group-'+role), root/f'initial_{role}_SKILL.md')
    result = {'config_sha256': sha(root/'config.json'), 'teachers_sha256': sha(root/'teachers.json'),
              'historical_references_sha256': sha(root/'historical_references.json'),
              'source_fingerprint': source_fingerprint(), 'commit': commit(), 'created': time.time(),
              'source_root': str(source), 'old_sampling_imported': False,
              'teacher_count': 2, 'versions': len(c['versions']), 'rounds_per_version': 5, 'outcomes_per_version': 2*c['rounds']*4*c['population_size'],
              'total_outcomes_including_shared_r0': c['total_result_budget']}
    write(root/'PREPARED.json', result)
    return result


def assert_prepared(root):
    p, c = read(prepared_path(root)), load_config(root)
    if p['config_sha256'] != sha(config_path(root)) or p['teachers_sha256'] != sha(root/'teachers.json'):
        raise ValueError('Frozen experiment configuration changed')
    if p['source_fingerprint'] != source_fingerprint():
        raise ValueError('Runtime changed; explicit tested deployment adoption required')
    if p.get('historical_references_sha256') and p['historical_references_sha256'] != sha(root/'historical_references.json'):
        raise ValueError('Frozen historical references changed')
    for teacher, digest in p.get('shared_r0_sha256',{}).items():
        if sha(root/'shared_r0'/teacher/'COMPLETE.json') != digest:
            raise ValueError('Shared R0 changed after protocol migration')
    return c


def adopt_unstarted_runtime(root, reason):
    """Preflight-only repairs may rebind code, never an experiment with formal data."""
    from agenticgen.runtime.locks import lock
    root = Path(root)
    if not reason or len(reason.strip()) < 20:
        raise ValueError('A concrete repair reason is required')
    with lock(root):
        if any((root/'shared_r0').glob('**/manifest.json')) or any((root/'versions').glob('**/binding.json')):
            raise ValueError('Formal data exists: preflight-only adoption is forbidden')
        previous = read(root/'PREPARED.json')
        if previous['config_sha256'] != sha(root/'config.json') or previous['teachers_sha256'] != sha(root/'teachers.json'):
            raise ValueError('Cannot adopt changed scientific configuration')
        operation = root/'operations'/f'preflight_repair_{time.time_ns()}.json'
        updated = dict(previous, source_fingerprint=source_fingerprint(), commit=commit())
        write(operation, {'reason': reason, 'previous': previous, 'updated': updated, 'formal_sampling_started': False})
        write(root/'PREPARED.json', updated)
        return {'operation': str(operation), 'requires_fresh_acceptance': True}


def initialize_cpu():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def documents(folder, parent=None):
    nodes = read(folder/'nodes.json')
    steps = read(folder/'steps.json')
    if parent:
        from .wave_native import selected_document
        population = read(parent/'manifest.json').get('population_size',100)
        start = read(folder/'manifest.json')['start_step']
        pnodes, psteps = read(parent/'nodes.json'), read(parent/'steps.json')
        for s in CHECKPOINT_STEPS:
            if s < start:
                n = pnodes[str(s)]
                nodes[str(s)] = {k: v*4 if isinstance(v, list) and len(v) == population else v for k, v in n.items()}
        prefix = [{k: v*4 if isinstance(v, list) and len(v) == population else v for k, v in r.items()} for r in psteps[:start]]
        steps = prefix+steps
    if [r['step'] for r in steps] != list(range(100)) or set(nodes) != {str(s) for s in CHECKPOINT_STEPS}:
        raise ValueError('Incomplete documentary trajectory')
    return nodes, steps


def make_trials(folder, records, schedules, round_no, parent=None):
    from .wave_native import selected_document
    nodes, steps = documents(folder, parent)
    size = len(records)
    population = read(folder/'manifest.json').get('population_size',100)
    trials = []
    for row, record in enumerate(records):
        trials.append({'id': f'r{round_no:02d}', 'source_id': f'r{round_no:02d}:G{row//population+1}:c{row%population:03d}',
            'ticks': schedules[row], 'record': record, 'source_branch': str(folder), 'source_slot': row,
            'nodes': {s: selected_document(n, row, size) for s, n in nodes.items()},
            'intervals': interval_summary([selected_document(r, row, size) for r in steps])})
    write(folder/'trials.json', trials)
    return trials


def evidence(template, round_no, group, latest, parent_best, old=None):
    chains = []
    per_seed = len(latest)//5
    for i, trial in enumerate(latest):
        history = copy.deepcopy(old['chains'][i]['trials']) if old else []
        history = [h for h in history if h['id'] != 'parent']
        history.append(copy.deepcopy(trial))
        parent = copy.deepcopy(parent_best[i]); parent['id'] = 'parent'
        history.append(parent)
        chains.append({'candidate_id': f'b{i//per_seed:03d}:s{i%per_seed:03d}', 'batch': i//per_seed, 'slot': i%per_seed,
                       'incumbent_id': 'parent', 'latest_id': trial['id'], 'trials': history})
    return {'teacher': template, 'observed_round': round_no, 'group_id': group,
            'round_ids': [f'r{r:02d}' for r in range(round_no+1)], 'chains': chains}


def ensure_r0(model, root, config, template, pool):
    from .wave_native import execute_wave
    from .wave_retention import evaluate_quality
    folder = root/'shared_r0'/template['teacher_id']
    from .fixed_start import prepare_r0
    schedules = [config['initial_ticks']]*config['population_size']
    prepare_r0(model, root, config, template, folder)
    rows = evaluate_quality(folder, template, pool)
    if not (folder/'best.json').exists():
        write(folder/'best.json', make_trials(folder, rows, schedules, 0))
    from .fixed_start import finish_r0
    finish_r0(folder)
    return folder


def bind(root, version):
    c = assert_prepared(root)
    if version not in c['versions']:
        raise ValueError('Version outside authorized twenty-version budget')
    number = int(version[1:])
    if number > 1:
        from .group_review import review_check
        review_check(root, version)
    base = root/'versions'/version
    base.mkdir(parents=True, exist_ok=True)
    active = effective_config(root, version, c, freeze=True)
    skills = {}
    for role, relative in SKILLS.items():
        source = skill('plip-group-'+role)
        dest = base/f'{role}_SKILL.md'
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError('Skill changed within frozen version')
        if not dest.exists():
            shutil.copyfile(source, dest)
        skills[role] = str(dest)
    binding = {'config_sha256': sha(config_path(root)), 'source_fingerprint': source_fingerprint(),
               'search_policy_sha256':value_sha(active['active_search_policy']),
               'condition_comparison_sha256':value_sha(c.get('condition_comparison',{})),
               'skill_hashes': {role: sha(p) for role, p in skills.items()},
               'shared_r0': {p.parent.name: sha(p) for p in (root/'shared_r0').glob('*/COMPLETE.json')}}
    if (base/'binding.json').exists() and read(base/'binding.json') != binding:
        raise ValueError('Version inputs changed')
    write(base/'binding.json', binding)
    if not (base/'deployment.json').exists():
        write(base/'deployment.json', {'commit': commit(), 'time': time.time()})
    return skills


def run(root, version):
    from agenticgen.runtime.locks import lock
    from .flowr import load_model
    from .wave_native import execute_wave
    from .wave_retention import evaluate_quality, commit_best, verify_snapshot
    from .group_agent import plan_population
    from .council import analyze_four_groups, BoundedCaller
    from .teacher_pipeline import run_teacher_pipeline
    root = Path(root)
    if (root/f'FINALIZED_{version}.json').exists():
        raise ValueError('Version already finalized; do not repeat sampling or rewrite completion')
    config = assert_prepared(root)
    population = config['population_size']
    if version not in config['versions']:
        raise ValueError('Version outside authorized budget')
    if version != 'v1':
        from .group_review import review_check
        review_check(root, version)
    acceptance = read(acceptance_path(root))
    if acceptance['source_fingerprint'] != source_fingerprint() or acceptance['config_sha256'] != sha(config_path(root)):
        raise ValueError('Acceptance does not cover this runtime/config')
    config = effective_config(root, version, config)
    templates = {t['teacher_id']: t for t in read(root/'teachers.json')}
    with lock(root):
        write(root/'workers.json', {'pid': os.getpid(), 'version': version, 'commit': commit(), 'started': time.time()})
        write(root/'status.json', {'phase': 'running', 'version': version})
        # Preflight checks also forbid a second GPU owner on this result root.
        model = load_model(config, next(iter(templates.values()))['target'], root)
        with concurrent.futures.ProcessPoolExecutor(max_workers=config['execution']['quality_workers'],
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize_cpu) as pool:
            r0 = {t: ensure_r0(model, root, config, template, pool) for t, template in templates.items()}
            skills = bind(root, version)
            base = root/'versions'/version
            caller = BoundedCaller()
            initial_jobs = {}
            for teacher, template in templates.items():
                best = read(r0[teacher]/'best.json')
                data = evidence(template, 0, 'bootstrap', best, best)
                data.update(version=version, controller_protocol=config['controller']['protocol'])
                boot = plan_population(data, base/teacher/'bootstrap', config, hypotheses=4,
                                       skill=skills['analyst'], caller=caller)
                initial_jobs[teacher] = {'teacher_id': teacher, 'observed_round': 0,
                    'jobs': [{'ticks': j['ticks'], 'source_proposal_id': 'bootstrap:'+j['id'],
                              'proposal': j} for j in boot['jobs']]}
            def infer(teacher, round_no, job):
                if shutil.disk_usage(root).free < 10*2**30:
                    raise RuntimeError('Less than 10 GiB free; do not create another wave')
                tb = base/teacher
                saved_artifacts = tb/f'round_{round_no:02d}'/'artifacts.json'
                if (tb/f'RETENTION_COMPLETE_{round_no:02d}.json').exists():
                    return read(saved_artifacts)
                parent = r0[teacher] if round_no == 1 else Path(read(tb/f'commit_{round_no-1:02d}.json')['snapshot'])
                if round_no > 1:
                    verify_snapshot(parent)
                retention_parent = parent
                parent = r0[teacher]
                parent_best = read(parent/'best.json')
                schedules = [j['ticks'] for j in job['jobs'] for _ in range(population)]
                if len(schedules) != population*4 or len({tuple(j['ticks']) for j in job['jobs']}) != 4:
                    raise ValueError(f'Four different {population}-candidate hypotheses required')
                start = 10
                wave = tb/f'round_{round_no:02d}'/'wave'
                write(wave.parent/'job.json', job)
                execute_wave(model, config, templates[teacher], schedules, wave,
                             root/'contexts'/'population_context.pt.gz', parent=parent, start_step=start)
                artifacts = {'wave': str(wave), 'parent': str(parent), 'retention_parent': str(retention_parent),
                             'fixed_start': 10, 'job': job, 'schedules': schedules}
                write(saved_artifacts, artifacts)
                return artifacts
            def analyze(teacher, round_no, artifacts, shared_caller):
                tb, wave, parent = base/teacher, Path(artifacts['wave']), Path(artifacts['parent'])
                if (wave.parent/'council'/'COMPLETE.json').exists():
                    council = wave.parent/'council'
                    inventory = read(council/'MASTER_REVIEW_INPUTS.json')
                    for name, digest in inventory['files_sha256'].items():
                        if sha(council/name) != digest:
                            raise ValueError('Completed analysis evidence changed')
                    return read(council/'selection.json')
                rows = evaluate_quality(wave, templates[teacher], pool)
                all_trials = make_trials(wave, rows, artifacts['schedules'], round_no, parent)
                parent_best = read(Path(artifacts['retention_parent'])/'best.json')
                for i, trial in enumerate(all_trials):
                    trial['proposal'] = artifacts['job']['jobs'][i//population].get('proposal')
                    trial['parent_source_id'] = read(parent/'best.json')[i%population]['source_id']
                    trial['physical_parent_policy'] = 'fixed_r0_step10'
                write(wave/'trials.json', all_trials)
                specs = []
                for g, j in enumerate(artifacts['job']['jobs']):
                    old = None
                    if round_no == 1:
                        initial = read(r0[teacher]/'best.json')
                        old = evidence(templates[teacher], 0, 'bootstrap', initial, initial)
                    else:
                        source = j['source_proposal_id'].split(':')[0]
                        old = read(tb/f'round_{round_no-1:02d}'/f'{source}_evidence.json')
                    trials = all_trials[g*population:(g+1)*population]
                    data = evidence(templates[teacher], round_no, f'G{g+1}', trials, parent_best, old)
                    private_updates = []
                    if round_no > 1:
                        prior = read(tb/f'round_{round_no-1:02d}'/'council'/'analysts'/source/'hypotheses.json')
                        private_updates = list(old.get('private_learning_history', [])) + prior.get('learning_update', [])
                    data.update(version=version, controller_protocol=config['controller']['protocol'],
                                private_learning_history=private_updates)
                    path = wave.parent/f'G{g+1}_evidence.json'
                    write(path, data)
                    history = []
                    for r in range(round_no+1):
                        trial_rows = [next(t for t in c['trials'] if t['id'] == f'r{r:02d}') for c in data['chains']]
                        summary = score_summary([t['record'] for t in trial_rows])
                        h = {'ticks': trial_rows[0]['ticks'], 'attempts': population, 'valid': summary['valid']}
                        from .objective import condition_summary
                        objective = condition_summary(trial_rows,templates[teacher])
                        h['condition_mean_rmsd_A'] = objective['condition_rmsd_A_all_attempts']['mean']
                        h['affinity_absolute_gap'] = objective['mean_affinity_absolute_gap']
                        if r:
                            if h['affinity_absolute_gap'] is not None and history[-1]['affinity_absolute_gap'] is not None:
                                h['condition_error_delta'] = h['condition_mean_rmsd_A']-history[-1]['condition_mean_rmsd_A']
                                h['affinity_gap_delta'] = h['affinity_absolute_gap']-history[-1]['affinity_absolute_gap']
                            prev = [next(t for t in c['trials'] if t['id'] == f'r{r-1:02d}')['record'] for c in data['chains']]
                            comparison = paired(prev, [t['record'] for t in trial_rows])
                            h.update(paired_common_valid=comparison['common_valid'],
                                     paired_mean_delta=comparison['delta']['mean'], validity_delta=summary['valid']-score_summary(prev)['valid'])
                        history.append(h)
                    specs.append({'group_id': f'G{g+1}', 'data': data, 'anchor_ticks': j['ticks'], 'group_history': history})
                memories = [read(tb/f'round_{r:02d}'/'council'/'published_memory.json') for r in range(1, round_no)]
                return analyze_four_groups(specs, wave.parent/'council', config, published_reflections=memories,
                                           caller=shared_caller, analyst_skill=skills['analyst'], synthesis_skill=skills['synthesizer'])
            def commit_round(teacher, round_no, artifacts, selection):
                wave, parent = Path(artifacts['wave']), Path(artifacts['retention_parent'])
                all_trials = read(wave/'trials.json')
                return commit_best(base/teacher, parent, wave,
                                   [all_trials[g*population:(g+1)*population] for g in range(4)], round_no, r0[teacher])
            run_teacher_pipeline(initial_jobs, base, rounds=config['rounds'], infer=infer, analyze=analyze,
                                 commit=commit_round, caller=caller)
        report(root, version)
        from .group_review import review_pack
        review_pack(root, version)
        write(root/f'FINALIZED_{version}.json', {'version': version, 'rounds': config['rounds'],
            'outcomes': 2*config['rounds']*4*population, 'report_sha256': sha(base/'report.json'),
            'review_manifest_sha256': sha(base/'review/manifest.json'), 'time': time.time()})
        write(root/'status.json', {'phase': 'complete' if version == config['versions'][-1] else 'awaiting_master_review',
                                  'version': version, 'rounds': config['rounds']})


def report(root, version):
    if load_config(root).get('controller',{}).get('protocol') == 'conditional_learning_v1':
        from agenticgen.condition_reporting import version_report
        return version_report(root,version)
    config = load_config(root); base = root/'versions'/version
    population = config['population_size']
    rows = []
    for template in read(root/'teachers.json'):
        t = template['teacher_id']; reference = template['source_record']['pic50_on_rescore']
        def add(label, r, records):
            valid = [x for x in records if eligible(x)]
            rows.append({'teacher': t, 'round': r, 'collection': label, **score_summary(records),
                'teacher_recovery_fraction_all_attempts': sum(x['pic50_on_rescore'] >= reference for x in valid)/len(records),
                'teacher_exceed_fraction_all_attempts': sum(x['pic50_on_rescore'] > reference for x in valid)/len(records),
                'posebusters_pass_fraction_all_attempts': sum(x.get('posebusters_pass', False) for x in records)/len(records),
                'severe_clash_fraction': sum(x.get('severe_clash', False) for x in records)/len(records),
                'quality_evaluated': sum(x.get('quality_evaluated', False) for x in records)})
        add('R0', 0, read(root/'shared_r0'/t/'quality_records.json'))
        for r in range(1, config['rounds']+1):
            wave = base/t/f'round_{r:02d}'/'wave'
            if not (wave/'quality_records.json').exists():
                continue
            records = read(wave/'quality_records.json')
            for g in range(4):
                add(f'G{g+1}_unselected', r, records[g*population:(g+1)*population])
            snapshot = base/t/'incumbents'/f'r{r:02d}'/'best.json'
            if snapshot.exists():
                add('TOP1_search_output', r, [x['record'] for x in read(snapshot)])
    result = {'version': version, 'rows': rows, 'counts': {'teachers': 2, 'rounds': config['rounds'],
              'optimization_attempts': sum(r['attempts'] for r in rows if r['collection'].endswith('_unselected'))},
              'limitations': ['shared starts are not independent', 'TOP1 is selected search output',
                             'development teachers; no held-out generalization claim']}
    write(base/'report.json', result)
    text = ['# Population scheduling results', '', '| Teacher | Round | Collection | Valid | Mean | Best | Top10 |',
            '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda x: f'{x:.5f}' if isinstance(x, (float, int)) else 'NA'
        text.append(f"| {r['teacher']} | {r['round']} | {r['collection']} | {r['valid']} | {fmt(r['mean'])} | {fmt(r['max'])} | {fmt(r['top10_mean'])} |")
    (base/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return result
