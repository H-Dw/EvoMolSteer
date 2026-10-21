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

SKILLS = {'analyst': 'skills/plip-group-analyst/SKILL.md',
          'synthesizer': 'skills/plip-group-synthesizer/SKILL.md'}


def prepare(root, source):
    root, source = Path(root), Path(source)
    c = configuration()
    if (root/'PREPARED.json').exists():
        assert_prepared(root)
        return read(root/'PREPARED.json')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use an empty new result directory')
    teachers = [t for t in read(source/'teachers.json') if t['rank'] in (1, 2)]
    from .cohort import select_templates
    teachers = select_templates(teachers, c['teacher_selection'])
    refs = read(source/'historical_references.json')
    for name, digest in refs.items():
        if sha(name) != digest:
            raise ValueError('Historical reference changed: '+name)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source/'baselines', root/'baselines', dirs_exist_ok=True)
    write(root/'config.json', c); write(root/'teachers.json', teachers)
    write(root/'historical_references.json', refs)
    for role, relative in SKILLS.items():
        shutil.copyfile(repository()/relative, root/f'initial_{role}_SKILL.md')
    result = {'config_sha256': sha(root/'config.json'), 'teachers_sha256': sha(root/'teachers.json'),
              'source_fingerprint': source_fingerprint(), 'commit': commit(), 'created': time.time(),
              'source_root': str(source), 'old_sampling_imported': False,
              'teacher_count': 2, 'versions': 20, 'rounds_per_version': 5, 'outcomes_per_version': 4000,
              'total_outcomes_including_shared_r0': 80200}
    write(root/'PREPARED.json', result)
    return result


def assert_prepared(root):
    p, c = read(root/'PREPARED.json'), read(root/'config.json')
    if p['config_sha256'] != sha(root/'config.json') or p['teachers_sha256'] != sha(root/'teachers.json'):
        raise ValueError('Frozen experiment configuration changed')
    if p['source_fingerprint'] != source_fingerprint():
        raise ValueError('Runtime changed; explicit tested deployment adoption required')
    return c


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
        start = read(folder/'manifest.json')['start_step']
        pnodes, psteps = read(parent/'nodes.json'), read(parent/'steps.json')
        for s in CHECKPOINT_STEPS:
            if s < start:
                n = pnodes[str(s)]
                nodes[str(s)] = {k: v*4 if isinstance(v, list) and len(v) == 100 else v for k, v in n.items()}
        prefix = [{k: v*4 if isinstance(v, list) and len(v) == 100 else v for k, v in r.items()} for r in psteps[:start]]
        steps = prefix+steps
    if [r['step'] for r in steps] != list(range(100)) or set(nodes) != {str(s) for s in CHECKPOINT_STEPS}:
        raise ValueError('Incomplete documentary trajectory')
    return nodes, steps


def make_trials(folder, records, schedules, round_no, parent=None):
    from .wave_native import selected_document
    nodes, steps = documents(folder, parent)
    size = len(records)
    trials = []
    for row, record in enumerate(records):
        trials.append({'id': f'r{round_no:02d}', 'source_id': f'r{round_no:02d}:G{row//100+1}:c{row%100:03d}',
            'ticks': schedules[row], 'record': record, 'source_branch': str(folder), 'source_slot': row,
            'nodes': {s: selected_document(n, row, size) for s, n in nodes.items()},
            'intervals': interval_summary([selected_document(r, row, size) for r in steps])})
    write(folder/'trials.json', trials)
    return trials


def evidence(template, round_no, group, latest, parent_best, old=None):
    chains = []
    for i, trial in enumerate(latest):
        history = copy.deepcopy(old['chains'][i]['trials']) if old else []
        history = [h for h in history if h['id'] != 'parent']
        history.append(copy.deepcopy(trial))
        parent = copy.deepcopy(parent_best[i]); parent['id'] = 'parent'
        history.append(parent)
        chains.append({'candidate_id': f'b{i//20:03d}:s{i%20:03d}', 'batch': i//20, 'slot': i%20,
                       'incumbent_id': 'parent', 'latest_id': trial['id'], 'trials': history})
    return {'teacher': template, 'observed_round': round_no, 'group_id': group,
            'round_ids': [f'r{r:02d}' for r in range(round_no+1)], 'chains': chains}


def ensure_r0(model, root, config, template, pool):
    from .wave_native import execute_wave
    from .wave_retention import evaluate_quality
    folder = root/'shared_r0'/template['teacher_id']
    schedules = [config['initial_ticks']]*100
    execute_wave(model, config, template, schedules, folder, root/'contexts'/'b100.pt.gz')
    rows = evaluate_quality(folder, template, pool)
    if not (folder/'best.json').exists():
        write(folder/'best.json', make_trials(folder, rows, schedules, 0))
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
    skills = {}
    for role, relative in SKILLS.items():
        source = repository()/relative
        dest = base/f'{role}_SKILL.md'
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError('Skill changed within frozen version')
        if not dest.exists():
            shutil.copyfile(source, dest)
        skills[role] = str(dest)
    binding = {'config_sha256': sha(root/'config.json'), 'source_fingerprint': source_fingerprint(),
               'skill_hashes': {role: sha(p) for role, p in skills.items()},
               'shared_r0': {p.parent.name: sha(p) for p in (root/'shared_r0').glob('*/COMPLETE.json')}}
    if (base/'binding.json').exists() and read(base/'binding.json') != binding:
        raise ValueError('Version inputs changed')
    write(base/'binding.json', binding)
    if not (base/'deployment.json').exists():
        write(base/'deployment.json', {'commit': commit(), 'time': time.time()})
    return skills


def run(root, version):
    from .runner import lock
    from .flowr import load_model
    from .wave_native import execute_wave
    from .wave_retention import evaluate_quality, commit_best, verify_snapshot
    from .group_agent import plan_population
    from .council import analyze_four_groups, BoundedCaller
    from .teacher_pipeline import run_teacher_pipeline
    root = Path(root)
    config = assert_prepared(root)
    acceptance = read(root/'ACCEPTANCE_COMPLETE.json')
    if acceptance['source_fingerprint'] != source_fingerprint() or acceptance['config_sha256'] != sha(root/'config.json'):
        raise ValueError('Acceptance does not cover this runtime/config')
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
                parent_best = read(parent/'best.json')
                schedules = [j['ticks'] for j in job['jobs'] for _ in range(100)]
                if len(schedules) != 400 or len({tuple(j['ticks']) for j in job['jobs']}) != 4:
                    raise ValueError('Four different hundred-row hypotheses required')
                start = min(earliest(p['ticks'], j['ticks']) if p['ticks'] != j['ticks'] else 100
                            for p in parent_best for j in job['jobs'])
                wave = tb/f'round_{round_no:02d}'/'wave'
                write(wave.parent/'job.json', job)
                execute_wave(model, config, templates[teacher], schedules, wave,
                             root/'contexts'/'b400.pt.gz', parent=parent, start_step=start)
                artifacts = {'wave': str(wave), 'parent': str(parent), 'job': job, 'schedules': schedules}
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
                parent_best = read(parent/'best.json')
                for i, trial in enumerate(all_trials):
                    trial['proposal'] = artifacts['job']['jobs'][i//100].get('proposal')
                    trial['parent_source_id'] = parent_best[i%100]['source_id']
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
                    trials = all_trials[g*100:(g+1)*100]
                    data = evidence(templates[teacher], round_no, f'G{g+1}', trials, parent_best, old)
                    path = wave.parent/f'G{g+1}_evidence.json'
                    write(path, data)
                    history = []
                    for r in range(round_no+1):
                        trial_rows = [next(t for t in c['trials'] if t['id'] == f'r{r:02d}') for c in data['chains']]
                        summary = score_summary([t['record'] for t in trial_rows])
                        h = {'ticks': trial_rows[0]['ticks'], 'attempts': 100, 'valid': summary['valid']}
                        if r:
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
                wave, parent = Path(artifacts['wave']), Path(artifacts['parent'])
                all_trials = read(wave/'trials.json')
                return commit_best(base/teacher, parent, wave,
                                   [all_trials[g*100:(g+1)*100] for g in range(4)], round_no, r0[teacher])
            run_teacher_pipeline(initial_jobs, base, rounds=config['rounds'], infer=infer, analyze=analyze,
                                 commit=commit_round, caller=caller)
        report(root, version)
        from .group_review import review_pack
        review_pack(root, version)
        write(root/f'FINALIZED_{version}.json', {'version': version, 'rounds': config['rounds'],
            'outcomes': 4000, 'report_sha256': sha(base/'report.json'),
            'review_manifest_sha256': sha(base/'review/manifest.json'), 'time': time.time()})
        write(root/'status.json', {'phase': 'complete' if version == config['versions'][-1] else 'awaiting_master_review',
                                  'version': version, 'rounds': config['rounds']})


def report(root, version):
    config = read(root/'config.json'); base = root/'versions'/version
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
                add(f'G{g+1}_unselected', r, records[g*100:(g+1)*100])
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
