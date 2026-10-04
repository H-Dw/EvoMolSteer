"""Capture code/environment identities without modifying a FLOWR installation."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
from evomolsteer.io import digest,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--flowr-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.flowr_root).resolve();files=sorted((root/'flowr').rglob('*.py'))
    if not files:raise ValueError('FLOWR source missing')
    def git(*args):
        r=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        return r.stdout.strip() if r.returncode==0 else None
    versions={}
    for name in ['torch','numpy','scipy','pandas','pyarrow','scikit-learn','h5py','rdkit']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    record={'python':sys.version,'executable':sys.executable,'flowr_root':str(root),'flowr_git_head':git('rev-parse','HEAD'),
        'flowr_tracked_status':git('status','--porcelain','--untracked-files=no'),'package_versions':versions,
        'flowr_sources':{p.relative_to(root).as_posix():{'sha256':digest(p),'mtime_ns':p.stat().st_mtime_ns} for p in files},
        'purpose':'Read-only snapshot. Capture time is separate from sampling start; do not claim a historical environment lock.'}
    write_json(a.output,record);print('Captured',len(files),'FLOWR source hashes')


if __name__=='__main__':main()
