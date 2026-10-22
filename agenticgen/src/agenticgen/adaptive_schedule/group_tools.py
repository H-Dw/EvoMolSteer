"""Read-only numerical tools over one immutable candidate population.

The LLM chooses registered functions and bounded arguments. It never supplies
Python, shell commands, filenames, coordinates, or scores to be trusted as data.
"""
import copy
import math
import statistics
from pathlib import Path

import jsonschema

from .io import read, write, sha, value_sha, eligible, better
from .grid import CHECKPOINT_STEPS, DIAGNOSTIC_INTERVALS


def distribution(values):
    v = sorted(x for x in values if isinstance(x, (int, float)) and math.isfinite(x))
    def quantile(q):
        if not v:
            return None
        i = (len(v)-1)*q
        a = int(i)
        return v[a] + (v[min(a+1, len(v)-1)]-v[a])*(i-a)
    return {"n": len(v), "mean": statistics.fmean(v) if v else None,
            "min": min(v) if v else None, "max": max(v) if v else None,
            "q10": quantile(.1), "median": quantile(.5), "q90": quantile(.9)}


def score_summary(records):
    values = [r['pic50_on_rescore'] for r in records if eligible(r)]
    return {"attempts": len(records), "valid": len(values), "failed": len(records)-len(values),
            **distribution(values), "top10_mean": statistics.fmean(sorted(values)[-10:]) if len(values) >= 10 else None}


def paired(a, b):
    if len(a) != len(b):
        raise ValueError('Pairing requires every original candidate identity')
    common = [(x['pic50_on_rescore'], y['pic50_on_rescore']) for x, y in zip(a, b) if eligible(x) and eligible(y)]
    delta = [y-x for x, y in common]
    return {"direction": "B minus A", "all_attempts": len(a), "common_valid": len(common),
            "valid_to_invalid": sum(eligible(x) and not eligible(y) for x, y in zip(a, b)),
            "invalid_to_valid": sum(not eligible(x) and eligible(y) for x, y in zip(a, b)),
            "both_invalid": sum(not eligible(x) and not eligible(y) for x, y in zip(a, b)),
            "win_rate_common_valid": sum(d > 0 for d in delta)/len(delta) if delta else None,
            "delta": distribution(delta)}


def local_geometry(coords, teacher):
    if len(coords) != len(teacher) or not coords:
        raise ValueError('Local coordinates must match the saved teacher-core correspondence')
    if any(len(p) != 3 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in p)
           for p in coords + teacher):
        raise ValueError('Nonfinite or malformed three-dimensional coordinates')
    distances = [math.sqrt(sum((x-y)**2 for x, y in zip(p, q))) for p, q in zip(coords, teacher)]
    return {"rms_A": math.sqrt(statistics.fmean(d*d for d in distances)),
            "mean_distance_A": statistics.fmean(distances), "max_distance_A": max(distances),
            "pseudo_huber_delta1": statistics.fmean(math.sqrt(1+d*d)-1 for d in distances),
            "core_atoms": len(distances), "alignment": "protein frame; no rigid alignment; saved geometric correspondence"}


def interval_summary(rows):
    result = []
    for start, end in DIAGNOSTIC_INTERVALS:
        group = [r for r in rows if start <= r['step'] <= end]
        result.append({"start": start, "end": end, "steps": len(group),
            "clipped_steps": sum(r.get('clip_factor', 1) < 1 for r in group),
            **{k: distribution([r.get(k) for r in group]) for k in [
                'loss', 'clip_factor', 'assignment_switch_fraction', 'gradient_drift_cosine',
                'max_atom_guidance_A', 'inside_rms_A', 'outside_rms_A', 'guidance_native_ratio']}})
    return result


def export_legacy_population(root, teacher_id, version, observed_round, output):
    """Import analyses only, without changing frozen configurations or running GPU inference."""
    root, output = Path(root).resolve(), Path(output)
    if version not in {'v1', 'v2'} or type(observed_round) is not int or not 0 <= observed_round <= 10:
        raise ValueError('A valid completed observed round is required')
    templates = [t for t in read(root/'teachers.json') if t['teacher_id'] == teacher_id]
    if len(templates) != 1:
        raise ValueError('Unknown teacher')
    template = templates[0]
    cache, hashes = {}, {}
    def document(path):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('Analysis reference escaped the source root')
        if path not in cache:
            cache[path] = read(path)
            hashes[str(path.relative_to(root))] = sha(path)
        return cache[path]
    def diagnostic(branch, slot, step=None, depth=0):
        if depth > 64:
            raise ValueError('Cyclic or excessive parent depth')
        folder = root/branch
        manifest = document(folder/'manifest.json')
        if manifest.get('payload_branch'):
            return diagnostic(manifest['payload_branch'], slot, step, depth+1)
        n = len(manifest['schedules'])
        def select(row):
            return {k: v[slot] if isinstance(v, list) and len(v) == n else v for k, v in row.items()
                    if k not in {'rng_before', 'rng_after'}}
        if step is not None:
            nodes = document(folder/'nodes.json')
            if str(step) in nodes:
                return select(nodes[str(step)])
            parent = manifest['parents'][slot]
            if parent is None:
                raise ValueError('Missing diagnostic node')
            return diagnostic(parent, slot, step, depth+1)
        prefix = []
        if manifest['start_step']:
            prefix = [r for r in diagnostic(manifest['parents'][slot], slot, None, depth+1)
                      if r['step'] < manifest['start_step']]
        return prefix + [select(r) for r in document(folder/'steps.json')]
    chains = []
    for batch in range(5):
        entries = document(root/'versions'/version/teacher_id/f'batch_{batch:03d}'/'histories.json')
        if len(entries) != 20 or [c['slot'] for c in entries] != list(range(20)):
            raise ValueError('All five original twenty-slot batches are required')
        for c in entries:
            history = c['history'][:observed_round+1]
            if [h['id'] for h in history] != [f'r{i:02d}' for i in range(observed_round+1)]:
                raise ValueError('Every candidate must have completed the observed round')
            best = history[0]
            for h in history[1:]:
                if better(h['record'], best['record']):
                    best = h
            trials = []
            for h in history:
                trials.append({"id": h['id'], "ticks": h['ticks'], "record": h['record'],
                    "parent_id": h.get('parent_id'), "proposal": h.get('proposal'),
                    "source_branch": h['branch'],
                    "nodes": {str(s): diagnostic(h['branch'], c['slot'], s) for s in CHECKPOINT_STEPS},
                    "intervals": interval_summary(diagnostic(h['branch'], c['slot']))})
            index_path = root/'versions'/version/teacher_id/f'batch_{batch:03d}'/'retention.json'
            retained = document(index_path) if index_path.exists() else None
            restart_branch = (retained['restart_refs'][c['slot']] if retained and retained['best_ids'][c['slot']] == best['id']
                              else best['branch'])
            restart_folder = root/restart_branch
            restartable = all((restart_folder/f'checkpoint_{s:03d}.pt.gz').exists() for s in [10, 26, 51, 76])
            chains.append({"candidate_id": f'b{batch:03d}:s{c["slot"]:03d}', "batch": batch, "slot": c['slot'],
                           "restart_branch": restart_branch, "restartable": restartable,
                           "incumbent_id": best['id'], "latest_id": history[-1]['id'], "trials": trials})
    result = {"schema": 'plip-population-evidence-v1', "teacher": template,
              "observed_round": observed_round, "version": version, "chains": chains,
              "source_root": str(root), "source_hashes": hashes,
              "round_ids": [f'r{i:02d}' for i in range(observed_round+1)]}
    if output.exists() and read(output) != result:
        raise ValueError('Do not overwrite a frozen population export')
    write(output, result)
    return result


class PopulationTools:
    """Registered deterministic tools; all calculations use the same frozen input."""
    def __init__(self, data, output):
        self.data, self.output = copy.deepcopy(data), Path(output)
        ids = [c['candidate_id'] for c in self.data['chains']]
        if len(ids) not in (25,100) or len(set(ids)) != len(ids):
            raise ValueError('Exactly 25 (new protocol) or 100 (migration compatibility) unique candidate identities are required')
        self.digest = value_sha(self.data)
        self.selectors = ['incumbent', 'latest'] + list(self.data['round_ids'])
        node_sets = [set(t['nodes']) for c in self.data['chains'] for t in c['trials']]
        available_steps = sorted(int(s) for s in set.intersection(*node_sets))
        selector = {"enum": self.selectors}
        subset = {"type": "array", "items": {"enum": ids}, "uniqueItems": True, "maxItems": 20}
        def schema(props, required):
            return {"type": "object", "properties": props, "required": required, "additionalProperties": False}
        self.registry = {
            'population_summary': ("Mean, Best, Top10, quantiles and validity over all 100 attempts.", schema({'selection': selector}, ['selection'])),
            'paired_comparison': ("Paired B-A changes and validity conversions; failures remain counted.", schema({'a': selector, 'b': selector}, ['a', 'b'])),
            'geometry_diagnostics': ("Core coordinate errors in the fixed protein frame, computed from saved correspondences.",
                schema({'selection': selector, 'step': {'enum': available_steps}, 'state': {'enum': ['current', 'endpoint']}, 'candidate_ids': subset}, ['selection', 'step', 'state'])),
            'interval_diagnostics': ("Effective displacement, clipping and matching instability by guidance interval.", schema({'selection': selector}, ['selection'])),
            'subgroup_response': ("Paired response in score quartiles fixed from A, including missing-score counts.", schema({'a': selector, 'b': selector}, ['a', 'b'])),
            'candidate_history': ("All recorded trials and hypotheses for up to 20 identified candidates.", schema({'candidate_ids': dict(subset, minItems=1)}, ['candidate_ids'])),
            'history_summary': ("All fixed-round distributions plus the separately selected incumbent population.", schema({}, [])),
        }

        if self.data.get('controller_protocol') == 'conditional_learning_v1':
            from .learning_tools import registry
            self.registry.update(registry(self.selectors))

    def catalog(self):
        return [{"name": k, "description": d, "parameters": s} for k, (d, s) in self.registry.items()]

    def trial(self, chain, selection):
        key = chain[selection+'_id'] if selection in {'incumbent', 'latest'} else selection
        return next(t for t in chain['trials'] if t['id'] == key)

    def records(self, selection):
        return [self.trial(c, selection)['record'] for c in self.data['chains']]

    def calculate(self, name, args):
        if name not in self.registry:
            raise ValueError('Unregistered population tool')
        jsonschema.validate(args, self.registry[name][1])
        if name in {'teacher_condition', 'response_edge', 'execution_history', 'task_knowledge', 'region_response', 'condition_quality'}:
            from .learning_tools import calculate
            return calculate(self, name, args)
        if name == 'population_summary':
            return score_summary(self.records(args['selection']))
        if name == 'paired_comparison':
            return paired(self.records(args['a']), self.records(args['b']))
        if name == 'history_summary':
            return {s: score_summary(self.records(s)) for s in self.selectors if s != 'latest'}
        if name == 'candidate_history':
            return [{"candidate_id": c['candidate_id'], "incumbent_id": c['incumbent_id'],
                     "trials": [{k: v for k, v in t.items() if k not in {'nodes', 'intervals', 'source_branch'}} for t in c['trials']]}
                    for c in self.data['chains'] if c['candidate_id'] in args['candidate_ids']]
        if name == 'subgroup_response':
            a, b = self.records(args['a']), self.records(args['b'])
            indices = sorted((i for i, r in enumerate(a) if eligible(r)), key=lambda i: (a[i]['pic50_on_rescore'], i))
            groups = [indices[len(indices)*q//4:len(indices)*(q+1)//4] for q in range(4)]
            return {"stratification": "A eligible score quartiles; ties resolved by stable candidate identity",
                    "A_ineligible": len(a)-len(indices), "groups": [{"quartile": q+1,
                    **paired([a[i] for i in group], [b[i] for i in group])} for q, group in enumerate(groups)]}
        if name == 'geometry_diagnostics':
            template = self.data['teacher']
            target = [template['coords_A'][i] for i in template['core_atoms']]
            rows = []
            for c in self.data['chains']:
                if args.get('candidate_ids') and c['candidate_id'] not in args['candidate_ids']:
                    continue
                n = self.trial(c, args['selection'])['nodes'][str(args['step'])]
                rows.append({"candidate_id": c['candidate_id'], **local_geometry(n['local_'+args['state']+'_A'], target)})
            return {"attempts": len(rows), "rms_A": distribution([r['rms_A'] for r in rows]),
                    "pseudo_huber_delta1": distribution([r['pseudo_huber_delta1'] for r in rows]),
                    "largest_error_candidates": sorted(rows, key=lambda r: r['rms_A'], reverse=True)[:10],
                    "correspondence": "saved native geometric matching; no rigid alignment"}
        if name == 'interval_diagnostics':
            groups = [self.trial(c, args['selection'])['intervals'] for c in self.data['chains']]
            result = []
            for i, (start, end) in enumerate(DIAGNOSTIC_INTERVALS):
                rows = [g[i] for g in groups]
                result.append({"start": start, "end": end, "candidates": len(rows),
                    "clipped_fraction": sum(r['clipped_steps'] for r in rows)/max(1, sum(r['steps'] for r in rows)),
                    **{k: distribution([r[k]['mean'] for r in rows]) for k in rows[0] if isinstance(rows[0][k], dict)}})
            return result
        raise AssertionError('Missing registered implementation')

    def call(self, name, args):
        request = {'tool': name, 'arguments': args, 'population_sha256': self.digest}
        identifier = 'tool:' + value_sha(request)
        result = {'evidence_id': identifier, **request, 'result': self.calculate(name, args)}
        path = self.output/(identifier[5:]+'.json')
        if path.exists() and read(path) != result:
            raise ValueError('Frozen numerical tool evidence changed')
        write(path, result)
        return result
