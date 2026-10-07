import argparse
from pathlib import Path
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.continuous.terminal_design import audit_terminal_behavior, compile_terminal
from evomolsteer.generation.window_reference import load_reference

p = argparse.ArgumentParser(); p.add_argument('--repo', default=str(Path(__file__).resolve().parents[1])); a = p.parse_args()
root = Path(a.repo); ev = root/'docs/experiments/ck2_affinity_geometry30_20261007'
folder = ev/'terminal_skill_test_v4'; reference = ev/'terminal_lineage_v4/terminal_reference.json.gz'
r = audit_terminal_behavior(root/'skills/affinity-terminal-lineage/SKILL.md', folder/'input.json', folder/'prompt.txt', folder/'response.json', reference, folder/'behavior_audit.json')
compiled = compile_terminal(read_json(folder/'response.json'), read_json(root/'configs/experiments/ck2_affinity_geometry30_v1/backtrack_round07.json'), load_reference(reference), digest(reference))
write_json(folder/'compiled_design.json', compiled)
print(r)
