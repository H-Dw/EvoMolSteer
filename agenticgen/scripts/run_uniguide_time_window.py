"""Detached paired time-grid/output/window experiment on a frozen historical engine."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','report','finalize'])
    for name in ('root','source-root','chemical-root','joint-root','frozen-repo'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    root,source,chemical,joint,frozen_repo=[Path(getattr(args,k)).resolve() for k in
        ('root','source_root','chemical_root','joint_root','frozen_repo')]
    if any(root==p or root.is_relative_to(p) or p.is_relative_to(root) for p in (source,chemical,joint,frozen_repo)):
        raise ValueError('Independent result directory required')
    execution_repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(frozen_repo/'agenticgen/src'))
    sys.path.insert(1,str(frozen_repo/'src'))
    path=execution_repo/'agenticgen/src/agenticgen/uniguide/time_window_experiment.py'
    spec=importlib.util.spec_from_file_location('agenticgen.uniguide.time_window_experiment',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        if args.mode=='prepare':module.prepare(root,source,chemical,joint,frozen_repo,execution_repo)
        elif args.mode=='run':module.run(root,source,chemical,joint,frozen_repo,execution_repo)
        elif args.mode=='finalize':module.finalize(root)
        else:module.report(root)
    except BaseException as error:
        if args.mode=='run' and root.exists():
            from agenticgen.uniguide.io import write
            import traceback
            write(root/'FAILED.json',dict(time_shanghai=module.now(),error_type=type(error).__name__,
                error=str(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__=='__main__':main()
