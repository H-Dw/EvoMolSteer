"""Compact historical primitive randomness, independently indexed by ancestry.

N=1 multinomial sampling uses PyTorch's exponential race (argmax(p / q)).
We save generator states and original call shapes, not sampled categories or
large random tensors. New probabilities are always evaluated on the live path.
"""
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import torch

from evomolsteer.io import digest,read_json
from evomolsteer.storage.transactions import atomic_json
from .common import ancestry,identity,load_state

CHANNELS=('atomics','charges','bonds')


def verify_source(config):
    for b,s in config['sources'].items():
        root=Path(config['remote_dataset'])/s['relative']
        for name,sha in s['files'].items():
            if digest(root/name)!=sha: raise ValueError(f'Historical source changed: batch {b} / {name}')
    for name,sha in config['flowr_files'].items():
        if digest(Path(config['flowr_root'])/name)!=sha: raise ValueError('Historical FLOWR source mismatch: '+name)


def assert_rng(saved,step):
    if not torch.equal(torch.get_rng_state(),saved['torch_cpu']): raise ValueError(f'CPU RNG anchor mismatch at {step}')
    states=torch.cuda.get_rng_state_all()
    if len(states)!=len(saved['torch_cuda']) or any(not torch.equal(a,b) for a,b in zip(states,saved['torch_cuda'])):
        raise ValueError(f'CUDA RNG anchor mismatch at {step}')


def verify_native_primitive(device='cuda'):
    """Check actual native samples and generator advancement before using tapes."""
    checks=[]
    for shape in [(1250,15),(1250,8),(31250,5)]:
        g=torch.Generator(device=device).manual_seed(1729)
        p=torch.arange(1,shape[1]+1,device=device,dtype=torch.float32).expand(shape).contiguous()
        p=p/p.sum(-1,keepdim=True);state=g.get_state()
        actual=torch.multinomial(p,1,True,generator=g);after=g.get_state()
        g.set_state(state);q=torch.empty_like(p).exponential_(1,generator=g)
        expected=(p/q).argmax(-1,keepdim=True)
        ok=torch.equal(actual,expected) and torch.equal(after,g.get_state())
        checks.append({'shape':list(shape),'passed':ok})
    if not all(x['passed'] for x in checks): raise ValueError('Native categorical primitive is not the validated exponential race')
    return checks


def build_tapes(config,work):
    work=Path(work);directory=work/'tapes';directory.mkdir(parents=True,exist_ok=True)
    record=directory/'verification.json'
    verify_source(config)
    if record.exists():
        old=read_json(record)
        if old['protocol_sha256']==config['protocol_sha256'] and all(digest(directory/k)==v for k,v in old['artifacts'].items()):
            return old
        raise ValueError('An existing random tape has changed identity')
    primitives=verify_native_primitive()
    artifacts={};audits=[];saved_cpu=torch.get_rng_state();saved_cuda=torch.cuda.get_rng_state_all()
    try:
        for b in range(20):
            source=config['sources'][str(b)];folder=Path(config['remote_dataset'])/source['relative']
            initial=load_state(folder/'initial_state.pt.gz');prior=initial['prior']
            if len(prior['coords'])!=50 or tuple(prior['coords'].shape)!=(50,25,3): raise ValueError('Unsupported original coordinate shape')
            shapes=[list(prior[k].shape) for k in CHANNELS]
            if shapes!=[[50,25,15],[50,25,8],[50,25,25,5]]: raise ValueError('Unsupported categorical call schema')
            with np.load(folder/'trajectory.npz') as z:
                selected=z['selected_indices'].copy();prob=z['selection_probability'].copy();resampled=z['resampled'].copy()
            before,roots=ancestry(selected)
            restart=load_state(folder/'restart_step_000.pt.gz')
            for k in ('coords','atomics','bonds','charges','mask'):
                if not torch.equal(prior[k],restart['current'][k]): raise ValueError('Initial prior differs from actual t0 current: '+k)
            anchor=restart['rng']
            torch.set_rng_state(anchor['torch_cpu']);torch.cuda.set_rng_state_all(anchor['torch_cuda'])
            states=[];anchors=[];selections=0
            dummy=[torch.ones((int(np.prod(s[:-1])),s[-1]),device='cuda',dtype=torch.float32) for s in shapes]
            for step in range(config['steps']):
                if step in (0,10,20,30,40,50,75,99):
                    assert_rng(load_state(folder/f'restart_step_{step:03d}.pt.gz')['rng'],step);anchors.append(step)
                calls=[torch.cuda.get_rng_state().cpu().numpy().copy()]
                torch.randn(tuple(prior['coords'].shape),device='cuda',dtype=torch.float32)
                for p in dummy:
                    calls.append(torch.cuda.get_rng_state().cpu().numpy().copy())
                    torch.multinomial(p,1,True)
                if resampled[step]:
                    got=torch.multinomial(torch.as_tensor(prob[step],device='cuda'),50,True).cpu().numpy()
                    if not np.array_equal(got,selected[step]): raise ValueError(f'Historical selection RNG differs at batch {b}, step {step}')
                    selections+=1
                elif not np.array_equal(selected[step],np.arange(50)):
                    raise ValueError('Non-resampled step contains a nonidentity selection')
                states.append(calls)
            path=directory/f'batch_{b:03d}.npz';tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez(f,states=np.asarray(states,dtype=np.uint8),before=before.astype(np.uint8),
                         roots=roots.astype(np.uint8),shape_coords=np.array(prior['coords'].shape),
                         shape_atomics=np.array(shapes[0]),shape_charges=np.array(shapes[1]),shape_bonds=np.array(shapes[2]))
            tmp.replace(path);artifacts[path.name]=digest(path)
            audits.append({'batch':b,'anchors':anchors,'exact_selection_events':selections,
                           'source_files':source['files'],'tape_sha256':artifacts[path.name]})
            print({'phase':'random_tape_verified','batch':b,'anchors':len(anchors),'selections':selections},flush=True)
    finally:
        torch.set_rng_state(saved_cpu);torch.cuda.set_rng_state_all(saved_cuda)
    result={'passed':True,'protocol_sha256':config['protocol_sha256'],'torch':torch.__version__,
            'primitive_checks':primitives,'batches':audits,'artifacts':artifacts,'unix':time.time(),
            'primitive':'float32 native normal plus exponential race p/q argmax; no copied discrete outcomes',
            'model_forward_calls':0,'selection_replay':'offline RNG recovery only; generation never performs SMC'}
    atomic_json(record,result);return result


class ReplayRandom:
    """Inject only primitive random variables for a fixed B50 set of paths."""
    def __init__(self,config,work,attempts,device='cuda'):
        if len(attempts)!=config['batch_size']: raise ValueError('Replay requires exactly B50')
        self.device=torch.device(device);self.attempts=attempts;self.tables={};self.groups={}
        for row,a in enumerate(attempts): self.groups.setdefault(a['batch'],[]).append((row,a['final_slot']))
        for b in self.groups:
            with np.load(Path(work)/'tapes'/f'batch_{b:03d}.npz') as z: self.tables[b]={k:z[k].copy() for k in z.files}
        self.normal=torch.randn;self.multinomial=torch.multinomial;self.step=None;self.call=0;self.records=[]

    def begin_step(self,step):
        if self.step is not None or step!=len(self.records): raise ValueError('Noncontiguous random-tape steps')
        self.step=int(step);self.call=0

    def end_step(self):
        if self.call!=4: raise ValueError(f'Expected one normal and three categorical draws; got {self.call}')
        self.records.append({'step':self.step,'primitive_calls':self.call});self.step=None

    def primitive(self,index):
        if self.step is None or self.call!=index: raise ValueError('Native random call order differs from recorded schema')
        result=None;key='coords' if index==0 else CHANNELS[index-1]
        for b,pairs in self.groups.items():
            table=self.tables[b];shape=tuple(int(x) for x in table['shape_'+key])
            generator=torch.Generator(device=self.device)
            generator.set_state(torch.from_numpy(table['states'][self.step,index].copy()))
            if index==0:
                full=self.normal(shape,device=self.device,dtype=torch.float32,generator=generator)
            else:
                full=torch.empty((int(np.prod(shape[:-1])),shape[-1]),device=self.device,dtype=torch.float32).exponential_(1,generator=generator).reshape(shape)
            if result is None: result=torch.empty(shape,device=self.device,dtype=torch.float32)
            dest=torch.tensor([p[0] for p in pairs],device=self.device)
            parents=torch.tensor([int(table['before'][self.step,p[1]]) for p in pairs],device=self.device)
            result[dest]=full[parents]
        self.call+=1;return result

    def draw_normal(self,*args,**kwargs):
        shape=tuple(args[0]) if len(args)==1 and hasattr(args[0],'__iter__') else tuple(args)
        if shape!=(50,25,3) or kwargs.get('dtype',torch.float32)!=torch.float32 or kwargs.get('generator') is not None:
            raise ValueError('Unrecorded normal call signature')
        return self.primitive(0)

    def draw_categorical(self,probabilities,num_samples,replacement=False,*,generator=None,out=None):
        if self.step is None or self.call not in (1,2,3) or num_samples!=1 or not replacement or generator is not None or out is not None:
            raise ValueError('Unexpected categorical random call')
        q=self.primitive(self.call).reshape_as(probabilities)
        if probabilities.dtype!=torch.float32 or probabilities.device!=self.device:
            # cuda and cuda:0 are equivalent device spellings.
            if probabilities.dtype!=torch.float32 or probabilities.device.type!=self.device.type: raise ValueError('Categorical primitive device/dtype changed')
        return (probabilities/q).argmax(-1,keepdim=True)

    @contextmanager
    def activate(self):
        torch.randn=self.draw_normal;torch.multinomial=self.draw_categorical
        try: yield self
        finally: torch.randn=self.normal;torch.multinomial=self.multinomial
