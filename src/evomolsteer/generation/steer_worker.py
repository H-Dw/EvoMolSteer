"""Runs in the specified FLOWR.ROOT environment; no extra guidance model."""
import argparse
from pathlib import Path
import subprocess
import sys

from ..io import read_json
from ..storage.selection_dataset import selection_steps
from ..storage.transactions import atomic_json
from .steer_launcher import SteerLearningConfig


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--request',required=True)
    a=p.parse_args()
    cfg=SteerLearningConfig(**read_json(a.request)).resolved()
    sys.path.insert(0,cfg.flowr_root)
    import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(Path(cfg.flowr_root)):
        raise RuntimeError('Imported FLOWR from a different program path')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A working FLOWR CUDA/ROCm runtime is required')
    grid=torch.linspace(0,1,cfg.steps+1).numpy()[:-1]
    steps=selection_steps(grid,cfg.window_start,cfg.window_end)
    folder=Path(a.request).parent
    inputs=read_json(Path(cfg.output_dataset)/'inputs/pocket_inputs.json')['files']
    try:
        commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit=None
    atomic_json(folder/'runtime.json',{'python':sys.executable,'flowr_package':str(flowr.__file__),
                'torch':torch.__version__,'cuda':torch.version.cuda,'hip':torch.version.hip,
                'device':torch.cuda.get_device_name(0),'evomolsteer_commit':commit,
                'selection_steps':steps.tolist(),'score_times':grid[steps].tolist(),
                'selection_window':[cfg.window_start,cfg.window_end],'gradient_guidance':False})
    from . import controller
    from .steer_trace import SteerLearningExtension
    sys.argv=['evomolsteer-steer','--root',cfg.output_dataset,'--checkpoint',cfg.checkpoint,
              '--campaign',cfg.campaign,'--n',str(cfg.samples),'--batch',str(cfg.batch_size),
              '--steps',str(cfg.steps),'--arms',','.join(cfg.arms),'--window-start',str(cfg.window_start),
              '--window',str(cfg.window_end),'--seed',str(cfg.seed)]
    controller.main(extension=SteerLearningExtension(cfg,inputs))


if __name__=='__main__':main()
