"""Verify the new writer against a real completed batch; not model inference.

The source is immutable. Stored f16 historical probabilities remain f16; they
cannot be retroactively restored to the native float32 forecasts.
"""
import argparse
import gzip
from pathlib import Path
from time import perf_counter

import numpy as np

from evomolsteer.io import digest,read_json
from evomolsteer.storage.arrays import byte_equal
from evomolsteer.storage.selection_dataset import SelectionLearningWriter,verify_learning_batch,read_scoring_event
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.generation.steer_trace import terminal_numeric_arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',required=True)
    p.add_argument('--terminal-prediction',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--report',required=True)
    p.add_argument('--window-start',type=float,default=0.)
    p.add_argument('--window-end',type=float,default=.5)
    p.add_argument('--storage-codec',choices=['none','gzip_shuffle'],default='none')
    a=p.parse_args()
    source,final,out=map(Path,(a.trajectory,a.terminal_prediction,a.output))
    if out.exists() or out.resolve().is_relative_to(source.parent.resolve()):
        raise ValueError('Use a new replay output outside the original batch')
    hashes={str(source):digest(source),str(final):digest(final)}
    with np.load(source,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    import torch
    with gzip.open(final,'rb') as f:terminal=terminal_numeric_arrays(torch.load(f,map_location='cpu',weights_only=True))
    start=perf_counter()
    campaign=source.parent.parent.parent
    original_config=read_json(campaign/'config.json') if (campaign/'config.json').exists() else {}
    batch=int(source.parent.name.split('_')[1])
    scale=original_config.get('coord_scale')
    if scale is None:raise ValueError('Original campaign config with coord_scale is required')
    seed=int(original_config.get('experiment',{}).get('seed',42))+batch*100003
    writer=SelectionLearningWriter(out,score_times=data['score_time'][:,0],window=(a.window_start,a.window_end),codec=a.storage_codec,
        metadata={'arm':source.parent.parent.name,'seed':seed,'batch':batch,'coord_scale':scale,
                  'purpose':'stored-data replay, not GPU generation'})
    for i in range(len(data['score_time'])):writer.append(i,{k:v[i] for k,v in data.items()})
    manifest=writer.finish(terminal)
    indices=np.asarray(manifest['selection_steps'])
    with TrajectoryPackage(out/'trajectory.h5') as package:
        for k,v in data.items():
            if not byte_equal(package.read(k),v[indices]):raise AssertionError('Window array differs: '+k)
    with TrajectoryPackage(out/'terminal.h5') as package:
        for k,v in terminal.items():
            if not byte_equal(package.read(k)[0],v):raise AssertionError('Native terminal tensor differs: '+k)
    verification=verify_learning_batch(out)
    failed_slot=int(np.flatnonzero(data['offspring_count'][0]==0)[0])
    event=read_scoring_event(out,0,failed_slot,terminal_outcome=True)
    if event['terminal_outcome'][str(failed_slot)]:raise AssertionError('Eliminated sibling was assigned descendants')
    for name,h in hashes.items():
        if digest(name)!=h:raise AssertionError('Replay modified source')
    report={'schema_version':'steer-learning-real-replay-1.0','model_inference_executed':False,
            'source_sha256':hashes,'source_trajectory_arrays_bit_verified':len(data),
            'native_terminal_arrays_bit_verified':len(terminal),'full_integration_steps':len(data['score_time']),
            'window_steps':indices.tolist(),'last_proposal_time':manifest['last_selected_proposal_time'],
            'failed_slot_verified':failed_slot,'verification':verification,
            'result_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            'seconds':perf_counter()-start,'output':str(out.resolve()),'codec':a.storage_codec,
            'precision_note':'Historical f16 probabilities preserved as supplied; new generation trace records native dtype',
            'source_code_sha256':digest(__file__)}
    atomic_json(a.report,report)
    print(report)


if __name__=='__main__':main()
