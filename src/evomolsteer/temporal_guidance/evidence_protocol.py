"""Opt-in v3: explicit observables, actionable reads and per-feature source gates."""
import copy

from .evidence import temporal_sources

PROTOCOL = 'temporal-feature-curves-v3'
INTERFACE = 'evidence-tools-json-v2'


def identity_feature(identity):
    while identity['kind'] == 'field':
        identity = identity['base']
    if identity['kind'] in {'curve', 'stage_profile'}:
        return identity['source_id'].split('/', 1)[1]
    return None


def feature_handles(tools):
    result = {}
    for ref, identity in tools.delivered.items():
        feature = identity_feature(identity)
        if feature:
            result.setdefault(feature, []).append(ref)
    return {f: sorted(refs) for f, refs in result.items()}


def cited_summary_features(tools, refs):
    result = set()
    for ref in refs:
        identity = tools.delivered[ref]
        while identity['kind'] == 'field':
            identity = identity['base']
        if identity['kind'] == 'row':
            feature = tools.records[identity['source_id']].get('feature')
            if feature in tools.curves['functions']:
                result.add(feature)
    return result


class TemporalEvidenceProtocol:
    def __init__(self, role, hypothesis_sources=None):
        self.role = role
        self.hypothesis_sources = hypothesis_sources or {}

    def availability(self, tools):
        return {'observed_feature_handles': feature_handles(tools),
                'unread_curve_features': sorted(set(tools.curves['functions']) - set(feature_handles(tools))),
                'instruction': 'Declare curve_features in each hypothesis. Read missing features with '
                               'get_curve/get_curves; cite only returned handles. Summary handles are not curve handles.'}

    def _next_reads(self, tools, features):
        readable = sorted(set(features).intersection(tools.curves['functions']))
        return [{'tool': 'get_curves', 'arguments': {'features': readable[i:i+8]}}
                for i in range(0, len(readable), 8)]

    def validate(self, result, tools):
        if self.role == 'Designer':
            return self._designer(result, tools)
        errors, available = [], feature_handles(tools)
        if not available:
            errors.append({'code': 'observed_curve_not_read', 'pointer': '/hypotheses',
                           'reason': 'Read at least one actual curve before completing the temporal analysis.',
                           'allowed_tools': ['get_curve', 'get_curves']})
        for i, hypothesis in enumerate(result['hypotheses']):
            declared = set(hypothesis.get('curve_features', []))
            summary = cited_summary_features(tools, hypothesis['evidence_refs'])
            required = declared | summary
            covered = {identity_feature(tools.delivered[r]) for r in hypothesis['evidence_refs']}
            missing = sorted(required - covered)
            if not declared or summary - declared:
                errors.append({'code': 'hypothesis_feature_declaration', 'pointer': f'/hypotheses/{i}/curve_features',
                               'features_from_actual_summary_citations': sorted(summary),
                               'reason': 'Declare the actual coordinate features used by this hypothesis. '
                                         'Counterevidence may instead remain in findings/limitations.'})
            if missing:
                errors.append({'code': 'hypothesis_missing_feature_curve', 'pointer': f'/hypotheses/{i}/evidence_refs',
                               'missing_features': missing,
                               'already_observed_handles': {f: available[f] for f in missing if f in available},
                               'next_reads': self._next_reads(tools, [f for f in missing if f not in available]),
                               'reason': 'Cite a delivered curve for each declared feature. Read missing curves '
                                         'or remove unsupported hypotheses; do not relabel a summary as a curve.'})
        return errors

    def _designer(self, result, tools):
        if result['decision'] != 'edit':
            return []
        if not result['hypothesis_refs']:
            return [{'code': 'edit_missing_hypothesis', 'pointer': '/hypothesis_refs',
                     'reason': 'An edit needs an actual Analyst hypothesis; otherwise retain/defer.'}]
        cited = temporal_sources(tools, result['evidence_refs'])
        errors = []
        for h in result['hypothesis_refs']:
            sources = self.hypothesis_sources.get(h, [])
            features = sorted({s.split('/', 1)[1] for s in sources})
            if not features:
                errors.append({'code': 'hypothesis_without_observed_curve', 'hypothesis_id': h})
            for feature in features:
                eligible = {s for s in sources if s.split('/', 1)[1] == feature}
                if not eligible.intersection(cited):
                    handles = [r for r in tools.known if temporal_sources(tools, [r]).intersection(eligible)]
                    errors.append({'code': 'edit_missing_hypothesis_feature', 'hypothesis_id': h,
                                   'feature': feature, 'allowed_observed_handles': sorted(handles),
                                   'reason': 'Cite this adopted hypothesis feature, or do not adopt that hypothesis.'})
        return errors

    def enrich_schema(self, schema, tools, *, stalled=False, diagnostics=()):
        schema = copy.deepcopy(schema)
        finish = schema['oneOf'][1]['properties']['result']
        handles = feature_handles(tools)
        if self.role == 'Analyst':
            if not handles:
                # A real observation is required. The available tool branch
                # remains under Agent control; host does not manufacture reads.
                schema['oneOf'] = [schema['oneOf'][0]]
                return schema
            hypothesis = finish['properties']['hypotheses']['items']
            names = sorted(handles if stalled else tools.curves['functions'])
            hypothesis['properties']['curve_features'] = {'type': 'array', 'minItems': 1,
                'uniqueItems': True, 'items': {'enum': names}}
            hypothesis['required'] += ['curve_features']
            if stalled:
                hypothesis['allOf'] = [{
                    'if': {'properties': {'curve_features': {'contains': {'const': f}}}},
                    'then': {'properties': {'evidence_refs': {'contains': {'enum': refs}}}}}
                    for f, refs in handles.items()]
        else:
            edit_rules = [{'properties': {'hypothesis_refs': {'minItems': 1}}}]
            for h, sources in self.hypothesis_sources.items():
                for feature in sorted({s.split('/', 1)[1] for s in sources}):
                    eligible = {s for s in sources if s.split('/', 1)[1] == feature}
                    refs = sorted(r for r in tools.known if temporal_sources(tools, [r]).intersection(eligible))
                    edit_rules.append({'if': {'properties': {'hypothesis_refs': {'contains': {'const': h}}}},
                        'then': {'properties': {'evidence_refs': {'contains': {'enum': refs}}}}})
            finish.setdefault('allOf', []).append({'if': {'properties': {'decision': {'const': 'edit'}}},
                                                   'then': {'allOf': edit_rules}})
        return schema
