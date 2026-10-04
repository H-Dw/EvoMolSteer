"""Run inside the user's FLOWR.ROOT Python/GPU environment, never replace torch."""
import argparse
from pathlib import Path
import sys
from ..io import read_json
from ..storage.transactions import atomic_json
from .launcher import FlowrRunConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
    cfg=FlowrRunConfig(**read_json(a.request)).resolved()
    sys.path.insert(0,cfg.flowr_root)
    import flowr
    module_path=Path(flowr.__file__).resolve()
    if not module_path.is_relative_to(Path(cfg.flowr_root).resolve()):
        raise RuntimeError('Imported FLOWR from a different checkout: '+str(module_path))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('FLOWR controller requires its CUDA/ROCm runtime')
    from . import controller
    atomic_json(Path(a.request).parent/'runtime.json',{'python':sys.executable,'flowr_package':str(module_path),
        'torch':torch.__version__,'hip':torch.version.hip,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),'controller':str(Path(controller.__file__).resolve())})
    sys.argv=['evomolsteer-flowr','--root',cfg.output_dataset,'--checkpoint',cfg.checkpoint,
              '--campaign',cfg.campaign,'--n',str(cfg.samples),'--batch',str(cfg.batch_size),'--steps',str(cfg.steps),
              '--arms',','.join(cfg.arms),'--window-start',str(cfg.window_start),'--window',str(cfg.window_end),'--seed',str(cfg.seed)]
    if cfg.verify_passive:sys.argv.append('--verify-passive')
    controller.main()


if __name__=='__main__':main()
