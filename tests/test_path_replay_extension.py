import copy
from pathlib import Path
import json
import pytest
import torch

from evomolsteer.io import read_json,digest
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.path_replay.common import identity,batch_path,binding,read_batch
from evomolsteer.path_replay.extension import prepare,validate_extension,reuse_completed,status
from evomolsteer.path_replay.memory import saved_tensor_context
from evomolsteer.path_replay.resources import parse_vram

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'configs/experiments/path_replay_b50_20261018/config.json'


def example(tmp_path):
    cfg=read_json(BASE);ext=prepare(REPO,BASE,tmp_path/'extension.json');return cfg,ext


def test_extension_preserves_all_paths_groups_and_frozen_scientific_parameters(tmp_path):
    cfg,ext=example(tmp_path);original=next(c for c in cfg['conditions'] if c['id']=='NC_R11')
    assert ext['condition']==dict(original,n=1000)
    assert ext['base_protocol_sha256']==cfg['protocol_sha256']
    assert len(cfg['groups'])==20 and all(len(g['ids'])==50 for g in cfg['groups'])
    assert ext['base_config_sha256']==digest(BASE)
    assert ext['remote_work'].startswith('/') and '\\' not in ext['remote_work']
    assert status(cfg,ext,tmp_path/'missing')['pending']==1000


@pytest.mark.parametrize('field,value',[('window',[0,.5]),('n',200),('program_sha256','wrong')])
def test_extension_cannot_change_frozen_r11(tmp_path,field,value):
    cfg,ext=example(tmp_path);ext['condition'][field]=value
    ext['extension_sha256']=identity({k:v for k,v in ext.items() if k!='extension_sha256'})
    with pytest.raises(ValueError,match='frozen R11'):validate_extension(cfg,ext)


def test_only_complete_checksummed_original_batches_are_adopted(tmp_path):
    cfg,ext=example(tmp_path);base=tmp_path/'base';work=tmp_path/'extension';ext['base_work']=str(base)
    original=next(c for c in cfg['conditions'] if c['id']=='NC_R11')
    g=cfg['groups'][0];file=batch_path(base,original,g);file.parent.mkdir(parents=True)
    artifact=file.parent/'states.npz';artifact.write_bytes(b'exact original states')
    receipt={'status':'complete','binding':binding(cfg,original,g),'records':[{'id':k,'pic50':8.} for k in g['ids']],
        'artifacts':{'states.npz':digest(artifact)}};atomic_json(file,receipt)
    g2=cfg['groups'][1];atomic_json(batch_path(base,original,g2),{'status':'failed','binding':binding(cfg,original,g2),'error':'old OOM'})
    audit=reuse_completed(cfg,ext,work)
    assert audit['reused_n']==50 and audit['new_required_n']==950
    assert read_batch(work,cfg,ext['condition'],g)['reused_from']['source_binding']==receipt['binding']
    assert read_batch(work,cfg,ext['condition'],g2) is None
    assert read_json(file)==receipt
    assert reuse_completed(cfg,ext,work)['reused_n']==50
    artifact.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifact changed'):reuse_completed(cfg,ext,work)


def test_smi_uses_actual_memory_fraction_not_compute_utilization_or_other_devices():
    result=parse_vram({'card0':{'PCI Bus':'0000:0a:00.0','vram Total Memory (MiB)':'65520','vram Total Used Memory (MiB)':'2','GPU use (%)':'100'},
        'card6':{'vram Total Memory (MiB)':'65520','vram Total Used Memory (MiB)':'60000'}})
    assert result['percent']==pytest.approx(100*2/65520) and result['pci_bus']=='0000:0a:00.0'
    with pytest.raises(ValueError):parse_vram({'card0':{'GPU use (%)':'0'}})


def test_offload_preserves_full_batch_outputs_repeated_vjps_rng_and_parameter_flags():
    torch.manual_seed(42);x0=torch.randn(50,25,3,dtype=torch.double);w=torch.nn.Parameter(torch.randn(3,7,dtype=torch.double))
    outputs=[];rng=torch.get_rng_state().clone()
    for policy in ('native_device','saved_tensors_cpu_exact'):
        x=x0.clone().requires_grad_(True)
        with saved_tensor_context(policy):
            y=(x@w).sin();a=y.square().sum();b=y.cos().sum()
            ga=torch.autograd.grad(a,x,retain_graph=True)[0];gb=torch.autograd.grad(b,x)[0]
        outputs.append((y.detach(),ga,gb))
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    assert torch.equal(rng,torch.get_rng_state()) and w.requires_grad and w.grad is None
    with pytest.raises(ValueError):saved_tensor_context('approximate_half_precision')


def test_frozen_r11_control_matches_with_offload_for_full_b50():
    from evomolsteer.path_replay.common import condition_program
    from evomolsteer.nc_guidance.reward import NCControl
    cfg=read_json(BASE);c=next(c for c in cfg['conditions'] if c['id']=='NC_R11');program,teacher=condition_program(REPO,c)
    class Model(torch.nn.Module):
        coord_scale=1.
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.arange(45,dtype=torch.float32).reshape(3,15)/100)
        def forward(self,curr,*args,**kw):
            x=curr['coords']
            return {'coords':x+.05*torch.sin(x),'atomics':x@self.w}
        def _get_predictions(self,out):return {'coords':out['coords']},None
    model=Model();coords=torch.tensor(teacher['coords_A'])[None].repeat(50,1,1)+.17
    curr={'coords':coords,'mask':torch.ones(50,25,dtype=torch.bool)};results=[]
    for policy in ('native_device','saved_tensors_cpu_exact'):
        control=NCControl(model,program,teacher,torch.zeros(50,3),.5)
        with torch.no_grad(),saved_tensor_context(policy):
            out=control.forward(curr,None,[torch.full((50,),.2)]*3,None,None,None,20,.01)
        results.append((out,control.cached))
    for key in results[0][0]:assert torch.equal(results[0][0][key],results[1][0][key])
    for a,b in zip(results[0][1],results[1][1]):
        assert torch.equal(a,b) if torch.is_tensor(a) else a==b
    assert model.w.requires_grad and model.w.grad is None
