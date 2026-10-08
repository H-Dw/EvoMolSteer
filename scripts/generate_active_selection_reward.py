"""Run the currently accepted reward through the real FLOWR endpoint adapter."""
import argparse
import sys
from pathlib import Path
from evomolsteer.io import read_json,digest

p=argparse.ArgumentParser(add_help=False);p.add_argument('--workflow',required=True)
a,args=p.parse_known_args();folder=Path(a.workflow).resolve();state=read_json(folder/'active_workflow.json')
program=folder/'active_program.json'
if digest(program)!=state['program_sha256']:raise ValueError('Active program hash mismatch')
if '--program' in args:raise ValueError('Use the active workflow, not an overriding program')
sys.argv=[sys.argv[0],'--program',str(program)]+args
from evomolsteer.generation.endpoint_controller import main
main()
