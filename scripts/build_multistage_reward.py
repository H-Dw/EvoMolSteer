"""Create reproducible discovery references, then compile the Designer program."""
import argparse
from evomolsteer.generation.prototypes import build_references, compile_program

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--analysis'); p.add_argument('--dataset')
p.add_argument('--references', required=True); p.add_argument('--tolerance', type=float, default=.1)
p.add_argument('--designer'); p.add_argument('--output')
a = p.parse_args()
if a.analysis or a.dataset:
    if not a.analysis or not a.dataset: p.error('Supply both analysis and dataset')
    packet = build_references(a.analysis, a.dataset, a.references, a.tolerance)
    print(packet['interpolation_audit'])
if a.designer or a.output:
    if not a.designer or not a.output: p.error('Supply both designer and output')
    program = compile_program(a.references, a.designer, a.output)
    print('Compiled', program['program_id'])
