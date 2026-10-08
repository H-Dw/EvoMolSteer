"""Analyze recorded coordinate-control response without repeating inference."""
import argparse
from pathlib import Path
from evomolsteer.continuous.flow_response import mine_execution, conditional_formula_audit
from evomolsteer.io import digest, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--output',required=True);p.add_argument('--program');p.add_argument('--reference')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use a fresh immutable output directory')
    if bool(a.program)!=bool(a.reference):p.error('Program and reference must be supplied together')
    mine_execution(a.dataset,a.campaign,out)
    if a.program:write_json(out/'conditional_formula_audit.json',conditional_formula_audit(a.program,a.reference))
    root=Path(a.dataset)/'results'/a.campaign
    inputs={str(f.resolve()):digest(f) for f in sorted(root.glob('*/batch_*/guidance_trace.jsonl'))}
    for value in [a.program,a.reference]:
        if value:inputs[str(Path(value).resolve())]=digest(value)
    write_json(out/'manifest.json',{'schema_version':'flow-response-tool-1.0','inputs':inputs,
        'code_files':{str(Path(__file__).resolve()):digest(__file__),
        'src/evomolsteer/continuous/flow_response.py':digest(Path(__file__).resolve().parents[1]/'src/evomolsteer/continuous/flow_response.py')},
        'outputs':{f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file()},
        'storage':'Whole-window batch/time summaries, no particle coordinate duplication'})
if __name__=='__main__':main()
