"""Small real-API check of evidence interpretation, without new inference."""
import argparse
import json
from pathlib import Path

from evomolsteer.agentic_steer.workflow import obj
from evomolsteer.ancestral_modes.evidence_interpretation import contrast_reading
from evomolsteer.deepseek_api import call_json
from evomolsteer.io import digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--outcome', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo, folder = args.repo.resolve(), args.output
    if folder.exists():
        raise ValueError('Use a fresh audit output directory; prior API records are immutable')
    folder.mkdir(parents=True)
    evidence = read_json(args.evidence)
    expected = contrast_reading(evidence)
    schema = obj({'increment_status': {'enum': ['positive_descriptive_increment', 'not_established']},
        'guidance_gain_established': {'type': 'boolean'},
        'unsupported_quote': {'type': 'string', 'maxLength': 1800},
        'evidence_feature': {'enum': list(evidence['terminal_contrast']['features'])},
        'assessment': {'type': 'string', 'maxLength': 4000},
        'next_discriminating_test': {'type': 'string', 'maxLength': 1800}})
    prior, outcome = read_json(args.response), read_json(args.outcome)
    packet = {'task': 'Review the incremental-information claim in the prior response against these actual measurements. This is a narrowly scoped interface/evidence-reading audit, not a request to design or dispatch a reward. Assess information beyond the whole-cloud comparator and observed guidance value separately. If a claim is unsupported, copy one short exact excerpt into unsupported_quote and select the feature that tests that excerpt. Keep assessment concise (preferably under 1200 characters). Return only this audit schema; full-generation handoff fields are outside this task.',
        'evidence': {k: evidence[k] for k in ('region_id', 'terminal_contrast')},
        'deterministic_contrast_reading': expected,
        'prior_role_response': {k: prior[k] for k in ('scientific_hypothesis', 'support_reason')},
        'actual_outcome': {k: outcome[k] for k in ('comparison_scope', 'performance', 'limits') if k in outcome}}
    reviews, checks = {}, {}
    for role in ('Analyst', 'Designer'):
        skill = repo / ('skills/ancestral-mode-' + role.lower() + '/SKILL.md')
        text = skill.read_text(encoding='utf-8')
        (folder / (role + '.SKILL.md')).write_text(text, encoding='utf-8')
        observation = dict(packet)
        if role == 'Designer':
            observation['analyst_review'] = reviews['Analyst']
        response = call_json([{'role': 'system', 'content': text},
            {'role': 'user', 'content': json.dumps(observation, ensure_ascii=False)}],
            schema, folder / (role + '.json'), max_output_tokens=2200)
        reviews[role] = response
        expected_status = 'positive_descriptive_increment' if expected['positive_beyond_global_contrast_supported'] else 'not_established'
        raw = json.dumps(packet['prior_role_response'], ensure_ascii=False)
        checks[role] = {'interval_reading_matches_tool': response['increment_status'] == expected_status,
            'does_not_infer_guidance_gain_from_association': response['guidance_gain_established'] is False,
            'quote_is_real': bool(response['unsupported_quote']) and response['unsupported_quote'] in raw,
            'uses_increment_comparator': response['evidence_feature'] == 'beyond_global_error_A',
            'skill_sha256': digest(skill)}
    boolean_checks = [value for row in checks.values() for value in row.values() if isinstance(value, bool)]
    result = {'scope': 'real_api_evidence_reading_only; not new reward selection or generation',
        'checks': checks, 'passed': all(boolean_checks),
        'evidence_sha256': digest(args.evidence), 'prior_response_sha256': digest(args.response),
        'outcome_sha256': digest(args.outcome)}
    write_json(folder / 'validation.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
