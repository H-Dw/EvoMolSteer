"""Measure tar.gz on different real historical batches, without GPU inference.

Each view contains a selection window, full genealogy, native terminal tensors,
actual SDF/scores/failures and source pocket inputs. Historical f16 remains f16;
these are separate CK2 batches, not distinct newly generated CrossDocked targets.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from evomolsteer.io import digest,read_json
from evomolsteer.storage.target_archive import archive_target_directories
from evomolsteer.storage.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dataset',required=True);p.add_argument('--campaign',default='main1000_w050')
    p.add_argument('--output',required=True);p.add_argument('--report',required=True)
    p.add_argument('--batch-count',type=int,default=10);p.add_argument('--compression-level',type=int,default=6)
    p.add_argument('--storage-codec',choices=['none','gzip_shuffle'],default='none')
    a=p.parse_args();source=Path(a.raw_dataset).resolve();out=Path(a.output).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('Use a new disjoint benchmark directory')
    if a.batch_count<1:raise ValueError('At least one batch required')
    camp=source/'results'/a.campaign;paths=sorted((camp/'single').glob('batch_*/trajectory.npz'))[:a.batch_count]
    if len(paths)!=a.batch_count:raise ValueError('Not enough real source batches')
    out.mkdir(parents=True);directories={};original={};replays=[]
    script=Path(__file__).with_name('replay_steer_learning_storage.py')
    for path in paths:
        name=path.parent.name;b=int(name.split('_')[1]);view=out/'views'/name
        batch=view/'results/replay/single'/name
        report=out/'reports'/f'{name}.json'
        subprocess.run([sys.executable,str(script),'--trajectory',str(path),
            '--terminal-prediction',str(path.parent/'final_prediction.pt.gz'),'--output',str(batch),
            '--report',str(report),'--storage-codec',a.storage_codec],check=True,stdout=subprocess.DEVNULL)
        replay=read_json(report);replays.append(replay)
        original.update(replay['source_sha256'])
        for filename in ('events.json','final_records.json','molecules_all_built.sdf','molecules_raw_decodable.sdf','COMPLETE.json'):
            src=path.parent/filename
            if src.is_file():
                shutil.copyfile(src,batch/filename);original[str(src)]=digest(src)
        for filename in ('config.json',f'frame_batch_{b:03d}.json'):
            src=camp/filename;dest=view/'results/replay'/filename
            shutil.copyfile(src,dest);original[str(src)]=digest(src)
        inputs=view/'inputs';inputs.mkdir()
        from evomolsteer.generation.launcher import INPUT_FILES
        for filename in INPUT_FILES:
            src=source/'inputs'/filename;shutil.copyfile(src,inputs/filename);original[str(src)]=digest(src)
        directories[name]=view
        print(json.dumps({'event':'real_batch_view_ready','batch':name,'particles':replay['verification']['particles']}),flush=True)
    individual=[archive_target_directories({key:value},out/'individual'/f'{key}.tar.gz',
                compression_level=a.compression_level,scope='historical_single_batch_storage_replay') for key,value in directories.items()]
    group=archive_target_directories(directories,out/'group10.tar.gz',compression_level=a.compression_level,
                                    scope='historical_single_batch_storage_replay')
    for path,sha in original.items():
        if digest(path)!=sha:raise AssertionError('Source changed: '+path)
    size=sum(r['archive_bytes'] for r in individual)
    result={'schema_version':'steer-target-tar-benchmark-1.0','model_inference_executed':False,
        'tasks':len(paths),'scope':'Different real CK2 single-arm 50-candidate batches; not CrossDocked GPU generation',
        'particles_per_task':[r['verification']['particles'] for r in replays],
        'codec':a.storage_codec,'compression_level':a.compression_level,'group':group,
        'individual_archives_bytes':size,'grouping_extra_reduction_percent':100*(size-group['archive_bytes'])/size,
        'source_hashes_unchanged':True,'source_sha256':original,
        'limitations':['Historical event probabilities are f16, new generation saves native precision',
                       'One biological target; compression ratio for different CrossDocked targets is not measured',
                       'All tested source views remain; savings require retirement of working copies',
                       'These views contain recorded inputs/results but are not newly completed target jobs']}
    atomic_json(a.report,result)
    print(json.dumps({'source_bytes':group['source_bytes'],'archive_bytes':group['archive_bytes'],
                      'reduction_percent':group['reduction_percent'],'grouping_extra_reduction_percent':result['grouping_extra_reduction_percent']},indent=2))


if __name__=='__main__':main()
