"""Deterministic observations for conditional control, never schedule recommendations."""
from .io import eligible
from .group_tools import paired, score_summary, distribution, local_geometry
from .learning import belief_view, executed_schedules


def registry(selectors):
    def schema(props, required=()):
        return {'type': 'object', 'properties': props, 'required': list(required), 'additionalProperties': False}
    ab = schema({'a': {'enum': selectors}, 'b': {'enum': selectors}}, ['a', 'b'])
    return {
        'condition_quality': ('Continuous teacher condition error, affinity agreement and Best bonus; no hard thresholds.', schema({'selection':{'enum':selectors}}, ['selection'])),
        'teacher_condition': ('Static molecular condition, original core, one-hop region and direct/indirect control limits.', schema({})),
        'response_edge': ('Explicit A→B schedules, B−A contributions for all identities, validity and delivered control changes.', ab),
        'execution_history': ('Previously executed schedules and cached summary sources; no pending proposals.', schema({})),
        'task_knowledge': ('Latest published belief revisions and counterevidence, scoped to this teacher/version.', schema({})),
        'region_response': ('Original core versus expanded-region geometry at a saved node; no rigid alignment.',
            schema({'selection': {'enum': selectors}, 'step': {'enum': [0,10,25,26,50,51,75,76,90,91,100]},
                    'state': {'enum': ['current', 'endpoint']}}, ['selection','step','state']))}


def calculate(tools, name, args):
    data = tools.data
    if name == 'condition_quality':
        from .objective import condition_summary
        return condition_summary([tools.trial(c,args['selection']) for c in data['chains']],data['teacher'])
    if name == 'teacher_condition':
        return data['teacher']['teacher_profile']
    if name == 'task_knowledge':
        return {'version': data['version'], 'teacher_id': data['teacher']['teacher_id'],
                'beliefs': belief_view(data.get('published_reflections', [])),
                'private_prior_updates': data.get('private_learning_history', [])}
    if name == 'execution_history':
        return {'executed': executed_schedules(data, data.get('published_reflections', [])),
                'selection_warning': 'Incumbent is selected output, not a shared schedule or generation parent.'}
    if name == 'response_edge':
        a, b = args['a'], args['b']; ar, br = tools.records(a), tools.records(b)
        from .objective import condition_summary
        ac=condition_summary([tools.trial(c,a) for c in data['chains']],data['teacher'])
        bc=condition_summary([tools.trial(c,b) for c in data['chains']],data['teacher'])
        rows = []; common = sum(eligible(x) and eligible(y) for x,y in zip(ar,br))
        for c, x, y, ax, bx in zip(data['chains'], ar, br, ac['individual_condition_errors'],bc['individual_condition_errors']):
            ta, tb = tools.trial(c,a), tools.trial(c,b)
            delta = y['pic50_on_rescore']-x['pic50_on_rescore'] if eligible(x) and eligible(y) else None
            rows.append({'candidate_id': c['candidate_id'], 'A_source': ta.get('source_id'),
                'B_source': tb.get('source_id'), 'A_ticks': ta['ticks'], 'B_ticks': tb['ticks'],
                'delta_ticks': [v-u for u,v in zip(ta['ticks'],tb['ticks'])],
                'A_valid': eligible(x), 'B_valid': eligible(y), 'delta_pic50': delta,
                'A_condition_rmsd_A':ax['rms_A'],'B_condition_rmsd_A':bx['rms_A'],
                'delta_condition_rmsd_A':bx['rms_A']-ax['rms_A'],
                'contribution_to_condition_mean_delta':(bx['rms_A']-ax['rms_A'])/len(ar),
                'contribution_to_common_valid_mean': delta/common if delta is not None else None})
        return {'A': a, 'B': b, 'direction': 'B minus A', 'A_summary': score_summary(ar),
                'ranking_comparison':_ranking_comparison(ac,bc),
                'A_condition_function':ac,'B_condition_function':bc,
                'B_summary': score_summary(br), 'paired': paired(ar,br), 'identities': rows,
                'A_intervals': tools.calculate('interval_diagnostics', {'selection':a}),
                'B_intervals': tools.calculate('interval_diagnostics', {'selection':b}),
                'interpretation': 'Concentrated contributions count normally. All-valid means and common-valid deltas differ when validity changes. Multiple changed intervals give a joint effect, not isolated partial derivatives.'}
    if name == 'region_response':
        t = data['teacher']; all_atoms = t['core_atoms']
        out = {}
        for label, atoms in [('original_core',t['original_core_atoms']), ('guided_region',all_atoms)]:
            columns = [all_atoms.index(i) for i in atoms]; target = [t['coords_A'][i] for i in atoms]
            rows = []
            for c in data['chains']:
                node = tools.trial(c,args['selection'])['nodes'][str(args['step'])]
                coords = [node['local_'+args['state']+'_A'][i] for i in columns]
                rows.append(local_geometry(coords,target))
            out[label] = {k: distribution([r[k] for r in rows]) for k in ['rms_A','pseudo_huber_delta1']}
        return {'step':args['step'], 'state':args['state'], 'unit':'angstrom', 'regions':out,
                'caution':'Geometry is a proxy; verify final quality and affinity separately.'}
    raise ValueError('Unknown learning observation')


def _ranking_comparison(a,b):
    from .selection import adjacent
    def row(value):
        return {'condition':value['condition_rmsd_A_all_attempts']['mean'],
                'functional_gap':value['mean_affinity_absolute_gap']}
    return dict(adjacent(row(a),row(b)), note='Pairwise diagnostic only. Final shared-schedule choice uses a band anchored to the entire tested pool minimum. Not an equivalence/noise assertion.')
