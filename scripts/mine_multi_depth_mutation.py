"""Execute multi-depth mining and capture its literal live tool receipt."""
import argparse
import gzip
import json
import sys
import traceback
from pathlib import Path
from evomolsteer.continuous.multi_depth_mutation import mine,parse_depths
from evomolsteer.io import digest,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--depths',default='2,3,5,8,13');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--region-width',type=float,default=4.)
    p.add_argument('--nuisance',choices=('chemistry','chemistry_pose'),default='chemistry_pose')
    p.add_argument('--receipt',help='Defaults to OUTPUT/execution_receipt.json; written by this actual execution')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve()
    receipt=Path(a.receipt).resolve() if a.receipt else out/'execution_receipt.json'
    if out.exists() or receipt.exists():raise FileExistsError('Fresh output/receipt destination required')
    from evomolsteer.trajectory_source import trajectory_paths
    import evomolsteer.continuous.multi_depth_mutation as module
    import evomolsteer.continuous.branch_mutation as branch
    import evomolsteer.continuous.selection_innovation as innovation
    import evomolsteer.continuous.affinity_geometry as geometry
    import evomolsteer.continuous.coordinate_mining as coordinate
    import evomolsteer.continuous.dynamic_regions as dynamic
    sources=[Path(a.reference).resolve(),Path(a.dataset).resolve()/'results'/a.campaign/'config.json']
    reference=json.loads(gzip.decompress(Path(a.reference).read_bytes()));batches=reference['discovery_batches']
    sources.extend(path for path in trajectory_paths(Path(a.dataset)/'results'/a.campaign) if path.parent.parent.name=='single' and int(path.parent.name.split('_')[1]) in batches)
    sources.extend(Path(a.dataset)/'results'/a.campaign/f'frame_batch_{int(b):03d}.json' for b in batches)
    inputs={str(v.resolve()):digest(v) for v in sources}
    loaded=[Path(v.__file__) for key,v in list(sys.modules.items()) if key.startswith('evomolsteer') and getattr(v,'__file__',None)]
    codes={str(v.resolve()):digest(v) for v in [Path(__file__),*loaded]}
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]];error=None
    try:
        result=mine(a.dataset,a.campaign,a.reference,a.output,parse_depths(a.depths),a.seed,a.region_width,a.nuisance)
        returncode=0
    except Exception:
        returncode=1;error=traceback.format_exc();result=None
    unchanged=all(digest(path)==sha for path,sha in inputs.items()) and all(digest(path)==sha for path,sha in codes.items())
    outputs={str(v.resolve()):digest(v) for v in sorted(out.iterdir()) if v.is_file()} if out.exists() else {}
    write_json(receipt,{'schema_version':'multi-depth-live-execution-receipt-1.0','tool_id':'multi_depth_mutation',
      'command':command,'returncode':returncode,'input_files':inputs,'code_files':codes,'output_files':outputs,
      'inputs_and_code_unchanged':unchanged,'complete_outputs':result is not None,'error':error,
      'attestation':'Captured by the actual tool process before/after execution; not historical reconstruction.'})
    if returncode or not unchanged:raise RuntimeError(error or 'Bound input/code changed during execution')
    print('multi-depth complete; receipt='+str(receipt),flush=True)

if __name__=='__main__':main()
