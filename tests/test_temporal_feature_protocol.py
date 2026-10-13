"""Source-specific gates and deterministic recovery of the actual C00 defect."""
import copy
from pathlib import Path

import jsonschema
import pytest

from evomolsteer.agent_workflow.contracts import action_schema
from evomolsteer.agent_workflow.engine import run_agent
from evomolsteer.io import read_json
from evomolsteer.temporal_guidance.evidence import temporal_sources
from evomolsteer.temporal_guidance.evidence_protocol import (
    PROTOCOL, INTERFACE, TemporalEvidenceProtocol, feature_handles, cited_summary_features)
from evomolsteer.temporal_guidance.workflow import TemporalEvidenceTools

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/experiments/compact_skills_verification_20261012/evidence'
FAILED = ROOT / 'docs/experiments/temporal_guidance_evidence_fix_20261013/calls/C00_canonical/chain_00/Analyst'


def tools():
    return TemporalEvidenceTools(DATA, bulk_curves=True)


def finish(features, refs):
    return {'action': 'finish', 'result': {'role': 'Analyst', 'findings': [],
        'hypotheses': [{'statement': 'Coordinate hypothesis to test', 'curve_features': features,
                        'evidence_refs': refs}], 'limitations': ['Not a measured guidance benefit']}}


def reconstruct_c00():
    t = tools()
    t.execute('read_metadata', {'section': 'clocks'})
    for path in sorted(FAILED.glob('*.observation.json')):
        action = read_json(path.with_name(path.name.replace('.observation.json', '.json')))
        assert t.execute(action['tool'], action['arguments']) == read_json(path)
    action = copy.deepcopy(read_json(FAILED / 'step_19.json'))
    # Replay fixtures add declarations derived from original cited rows. The
    # production host never writes these declarations into an LLM response.
    for h in action['result']['hypotheses']:
        h['curve_features'] = sorted(cited_summary_features(t, h['evidence_refs']))
    return t, action


def test_real_c00_replay_locates_seven_missing_features_with_exact_reads():
    t, action = reconstruct_c00()
    p = TemporalEvidenceProtocol('Analyst')
    errors = p.validate(action['result'], t)
    assert len(feature_handles(t)) == 3 and len(action['result']['hypotheses']) == 9
    assert len(errors) == 7 and all(e['code'] == 'hypothesis_missing_feature_curve' for e in errors)
    missing = {f for e in errors for f in e['missing_features']}
    requested = {f for e in errors for r in e['next_reads'] for f in r['arguments']['features']}
    assert missing == requested and len(requested) == 7


def test_no_observations_allows_only_tools_and_no_empty_fake_finish():
    t, p = tools(), TemporalEvidenceProtocol('Analyst')
    base = action_schema('Analyst', t.known, t.schemas)
    enriched = p.enrich_schema(base, t)
    assert len(enriched['oneOf']) == 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(finish([], []), enriched)
    jsonschema.validate({'action': 'tool', 'tool': 'get_curves', 'arguments': {'features': sorted(t.curves['functions'])[:2]}}, enriched)


def test_bulk_reads_are_bounded_and_feature_bound():
    t = tools()
    features = sorted(t.curves['functions'])[:8]
    result = t.execute('get_curves', {'features': features})
    assert result['ok'] and set(feature_handles(t)) == set(features)
    with pytest.raises(jsonschema.ValidationError):
        t.execute('get_curves', {'features': sorted(t.curves['functions'])[:9]})
    assert set(feature_handles(t)) == set(features)


def test_wrong_curve_and_summary_cannot_cover_declared_observable():
    t = tools()
    names = sorted(t.curves['functions'])[:2]
    ref = t.execute('get_curve', {'feature': names[0]})['observation']['ref']
    errors = TemporalEvidenceProtocol('Analyst').validate(finish([names[1]], [ref])['result'], t)
    assert errors[0]['missing_features'] == [names[1]]


def test_each_designer_adopted_hypothesis_feature_needs_own_source():
    t = tools()
    observations = t.execute('get_curves', {'features': sorted(t.curves['functions'])[:3]})['observation']['records']
    refs = [v['ref'] for v in observations]
    sources = {'H001': sorted(temporal_sources(t, refs[:2])), 'H002': sorted(temporal_sources(t, refs[2:]))}
    p = TemporalEvidenceProtocol('Designer', sources)
    bad = {'decision': 'edit', 'hypothesis_refs': ['H001', 'H002'], 'evidence_refs': [refs[0], refs[2]]}
    assert len(p.validate(bad, t)) == 1
    bad['evidence_refs'] = refs
    assert p.validate(bad, t) == []
    assert p.validate({'decision': 'retain'}, t) == []


def test_c00_no_progress_recovery_reads_missing_curves_then_finishes(tmp_path):
    t, bad = reconstruct_c00()
    p = TemporalEvidenceProtocol('Analyst')
    requests = []
    missing = sorted({f for e in p.validate(bad['result'], t) for f in e['missing_features']})
    def caller(messages, schema, output, **kwargs):
        requests.append(schema)
        if len(requests) <= 2:
            return copy.deepcopy(bad)
        if len(requests) == 3:
            hypothesis = schema['oneOf'][1]['properties']['result']['properties']['hypotheses']['items']
            assert set(hypothesis['properties']['curve_features']['items']['enum']) == set(feature_handles(t))
            assert hypothesis['allOf']
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(bad, schema)
            return {'action': 'tool', 'tool': 'get_curves', 'arguments': {'features': missing}}
        repaired = copy.deepcopy(bad)
        for h in repaired['result']['hypotheses']:
            for f in h['curve_features']:
                h['evidence_refs'] = sorted(set(h['evidence_refs'] + feature_handles(t)[f]))
        return repaired
    result = run_agent('Analyst', t, 'Replay source delivery', {}, tmp_path, caller=caller,
        max_steps=5, finish_validator=p.validate, finish_policy=PROTOCOL,
        schema_enricher=p.enrich_schema, evidence_availability=p.availability, interface_policy=INTERFACE,
        max_format_repairs=2)
    assert len(requests) == 4 and len(result['hypotheses']) == 9
    assert len(list(tmp_path.glob('*.finish_rejected.json'))) == 2
    assert p.validate(read_json(tmp_path / 'wire_result.json'), t) == []
    assert read_json(tmp_path / 'completed.json')['interface_version'] == INTERFACE
