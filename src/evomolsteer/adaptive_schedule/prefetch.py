"""Bounded HTTP-only lookahead; the calling thread exclusively owns the GPU.

No worker imports torch, samples an RNG, changes histories or prunes payloads.
Each batch uses a frozen previous-round population and its own immutable history.
"""
import concurrent.futures
import copy
import time
import traceback

from .io import read, write, value_sha


class ProposalPool:
    def __init__(self, concurrency):
        if concurrency < 1:
            raise ValueError('Proposal concurrency must be positive')
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def submit(self, root, config, template, chains, skill, folder, population, proposer=None):
        if proposer is None:
            from .agent import propose
            proposer = propose
        chains, population = copy.deepcopy(chains), copy.deepcopy(population)
        round_no = population['decision_round']
        if (len(chains) != config['batch_size']
                or [c['slot'] for c in chains] != list(range(config['batch_size']))
                or any(len(c['history']) != round_no for c in chains)
                or population['teacher_id'] != template['teacher_id']):
            raise ValueError('Lookahead requires a complete previous-round batch')
        folder.mkdir(parents=True, exist_ok=True)
        futures = {self.pool.submit(proposer, root, config, template, c['slot'], c['history'], c['best_id'],
                     skill, folder / f"slot_{c['slot']:03d}.json", population=population): c['slot'] for c in chains}
        return {'history_sha256': value_sha(chains), 'population_sha256': value_sha(population),
                'futures': futures, 'submitted_at': time.time()}


def collect(bundle, chains, population, folder):
    if (bundle['history_sha256'] != value_sha(chains)
            or bundle['population_sha256'] != value_sha(population)):
        raise ValueError('Prefetched history/population changed before GPU execution')
    actions, failures = [None] * len(chains), []
    wait_start = time.time()
    for future in concurrent.futures.as_completed(bundle['futures']):
        slot = bundle['futures'][future]
        try:
            actions[slot] = future.result()
        except Exception:
            failures.append({'slot': slot, 'traceback': traceback.format_exc()})
    write(folder / 'prefetch_timing.json', {'submitted_at': bundle['submitted_at'],
          'gpu_owner_wait_seconds': time.time()-wait_start, 'collected_at': time.time(),
          'history_sha256': bundle['history_sha256'], 'population_sha256': bundle['population_sha256']})
    if failures:
        write(folder / 'errors.json', failures)
        raise RuntimeError(f'LLM proposal failure in {len(failures)} slots; successful calls retained')
    return actions


def job_order(templates, batches):
    return [(t, batch) for batch in range(batches) for t in templates]


def interleaved_run(root, config, templates, version, skill, fast):
    """Round barriers + alternating teachers + one next batch of HTTP lookahead."""
    from . import runner
    from .flowr import load_model
    from .population import round_context
    from .retention import enabled, compact_batch
    import torch

    if config['batch_size'] != 20:
        raise ValueError('Interleaving preserves the audited native batch of twenty')
    depth = config['execution'].get('proposal_lookahead_batches', 2)
    if depth != 2:
        raise ValueError('Only bounded current-plus-next batch lookahead is validated')
    pool = ProposalPool(config['llm']['concurrency'])
    model, target = None, None
    failures = []
    try:
        for round_no in range(config['rounds'] + 1):
            populations = {t['teacher_id']: round_context(root, config, t, version, round_no)
                           for t in templates} if round_no else {}
            jobs = []
            for template, batch in job_order(templates, config['batches']):
                folder = root / 'versions' / version / template['teacher_id'] / f'batch_{batch:03d}'
                path = folder / 'histories.json'
                if path.exists():
                    chains = read(path)
                    if all(len(c['history']) > round_no for c in chains):
                        # A crash after history commit must finish evaluation/retention
                        # before any subsequent round can read or delete structures.
                        if enabled(config):
                            compact_batch(root, config, template, batch, version, chains)
                        if all(len(c['history']) == config['rounds']+1 for c in chains):
                            write(folder / 'COMPLETE.json', {'chains': 20, 'rounds': config['rounds']})
                        continue
                jobs.append((template, batch, folder))
            pending = {}

            def submit(index):
                template, batch, folder = jobs[index]
                chains = read(folder / 'histories.json')
                if enabled(config):
                    compact_batch(root, config, template, batch, version, chains)
                return pool.submit(root, config, template, chains, skill,
                                   folder / f'round_{round_no:02d}', populations[template['teacher_id']])

            for index, (template, batch, folder) in enumerate(jobs):
                # Only two batches can have queued HTTP work. The global pool is
                # shared by both teachers, rather than multiplying API concurrency.
                if round_no:
                    for ahead in range(index, min(index + depth, len(jobs))):
                        if ahead not in pending:
                            pending[ahead] = submit(ahead)
                if target != template['target']:
                    if model is not None:
                        del model
                        torch.cuda.empty_cache()
                    model = load_model(config, template['target'], root)
                    target = template['target']
                for attempt in range(config['max_recoveries']):
                    try:
                        write(root / 'status.json', {'phase': 'running', 'version': version,
                            'operation': 'interleaved_prefetch', 'teacher': template['teacher_id'],
                            'batch': batch, 'round': round_no, 'time': time.time()})
                        runner.run_batch(model, root, config, template, batch, version, skill, fast,
                            round_no, populations.get(template['teacher_id']), proposals=pending.get(index))
                        break
                    except Exception:
                        error = {'teacher': template['teacher_id'], 'batch': batch, 'round': round_no,
                                 'attempt': attempt+1, 'traceback': traceback.format_exc(), 'time': time.time()}
                        write(folder / f'round_{round_no:02d}' / f'error_{attempt+1:02d}.json', error)
                        if attempt+1 == config['max_recoveries']:
                            failures.append(error)
                        else:
                            time.sleep([30, 120][min(attempt, 1)])
                            if round_no:
                                # Existing successful request receipts are reused by propose.
                                # If generation committed and only retention failed,
                                # run_batch skips proposals and completes the transaction.
                                chains = read(folder / 'histories.json')
                                if all(len(c['history']) == round_no for c in chains):
                                    pending[index] = submit(index)
                pending.pop(index, None)
            if failures:
                break  # Never open the next population round over incomplete attempts.
    finally:
        pool.close()
    return failures
