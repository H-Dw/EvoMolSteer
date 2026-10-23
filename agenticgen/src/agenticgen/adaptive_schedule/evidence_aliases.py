"""Session-local citation labels; full immutable evidence remains on disk."""
import copy
from .io import value_sha

PROTOCOL = 'private-evidence-alias-v1'


class EvidenceAliases:
    def __init__(self):
        self.sources = {}
        self.aliases = {}

    def project(self, records):
        result = []
        for record in records:
            source = record['evidence_id']
            digest = value_sha(record)
            if source not in self.sources:
                alias = f'E{len(self.sources)+1:03d}'
                self.sources[source] = alias
                self.aliases[alias] = {'source_id': source, 'source_record_sha256': digest}
            alias = self.sources[source]
            if self.aliases[alias]['source_record_sha256'] != digest:
                raise ValueError('Session evidence changed for an existing source ID')
            result.append({**copy.deepcopy(record), 'evidence_id': alias,
                           'source_evidence_id': source, 'source_record_sha256': digest})
        return result

    def manifest(self):
        return {'protocol': PROTOCOL, 'aliases': copy.deepcopy(self.aliases)}

    def restore_citations(self, response):
        """Decode only exact, already schema-validated citation fields; never guess."""
        if isinstance(response, list):
            return [self.restore_citations(item) for item in response]
        if not isinstance(response, dict):
            return copy.deepcopy(response)
        result = {}
        for key, value in response.items():
            if key == 'evidence_ids':
                if not isinstance(value, list) or any(not isinstance(v, str) or v not in self.aliases for v in value):
                    raise ValueError('Unknown citation alias; no approximate replacement')
                result[key] = [self.aliases[v]['source_id'] for v in value]
            else:
                result[key] = self.restore_citations(value)
        return result
