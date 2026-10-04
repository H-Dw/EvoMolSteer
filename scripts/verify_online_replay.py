"""Replay real captured steps through the production writer; never delete input."""
import argparse
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
from evomolsteer.io import digest,read_json,write_json
from evomolsteer.storage.streaming import StepTrajectoryWriter
from evomolsteer.storage.trajectory import TrajectoryPackage


def replay(source,output,campaign,batch):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copytree(source/'inputs',output/'inputs')
    camp=source/'results'/campaign;dest=output/'results'/campaign;dest.mkdir(parents=True)
    for name in ['config.json',f'frame_batch_{batch:03d}.json']:
        shutil.copyfile(camp/name,dest/name)
    rows=[]
    for arm in ['unguided','single','joint']:
        path=camp/arm/f'batch_{batch:03d}'/'trajectory.npz'
        before=digest(path)
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        n=len(arrays['selected_indices']);bdir=dest/arm/f'batch_{batch:03d}'
        writer=StepTrajectoryWriter(bdir,expected_steps=n,metadata={'source_sha256':before,'kind':'stored_data_replay'})
        elapsed=[]
        for i in range(n):
            start=perf_counter();writer.append(i,{k:v[i] for k,v in arrays.items()});elapsed.append(perf_counter()-start)
        staging_bytes=sum(r['bytes'] for r in read_json(writer.manifest_path)['steps'])
        started=perf_counter();result=writer.finalize();merge_seconds=perf_counter()-started
        with TrajectoryPackage(writer.final_path) as p:count=p.verify(arrays)
        assert before==digest(path) and not writer.stages.exists() and not list(bdir.glob('*.npz'))
        shutil.copyfile(path.parent/'COMPLETE.json',bdir/'COMPLETE.json')
        rows.append({'arm':arm,'batch':batch,'steps':n,'verified_arrays':count,'source_unchanged':True,
            'source_npz_bytes':path.stat().st_size,'stages_total_bytes':staging_bytes,'final_bytes':writer.final_path.stat().st_size,
            'commit_seconds_median':float(np.median(elapsed)),'commit_seconds_max':max(elapsed),
            'finalize_seconds':merge_seconds,'per_stage_verified_before_return':True,'stage_cleanup_verified':True})
        print(arm,rows[-1],flush=True)
    write_json(dest/'COMPLETE.json',{'kind':'stored_data_replay','complete':True,'batches':[batch]})
    write_json(output/'storage_replay_verification.json',{'source':str(source),'campaign':campaign,'batches':rows,
        'scope':'Storage equivalence only; this is not a new molecular generation experiment','passed':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dataset',required=True);p.add_argument('--output-dataset',required=True)
    p.add_argument('--campaign',default='main1000_w050');p.add_argument('--batch',type=int,default=0)
    a=p.parse_args();replay(a.input_dataset,a.output_dataset,a.campaign,a.batch)
