"""Existing-backward output sensitivities, not neural importance or attention.

Hooks return None and never alter values/gradients. Output scales differ across
modules; gradient RMS cannot rank their causal contribution to binding affinity.
"""
import torch


def tensors(value):
    if torch.is_tensor(value):yield value
    elif isinstance(value,(tuple,list)):
        for v in value:yield from tensors(v)
    elif isinstance(value,dict):
        for v in value.values():yield from tensors(v)


class DecoderSensitivity:
    def __init__(self,model):
        self.records={};self.handles=[]
        decoder=getattr(getattr(model,'gen',None),'ligand_dec',None)
        if decoder is None:return
        for name,module in decoder.named_modules():
            parts=name.split('.')
            if name in ('coord_emb','coord_out_proj') or len(parts)==2 and parts[0]=='layers' and parts[1].isdigit():
                self.handles.append(module.register_forward_hook(self._hook('gen.ligand_dec.'+name)))

    def _hook(self,name):
        def forward_hook(module,args,output):
            for index,value in enumerate(tensors(output)):
                if not value.requires_grad:continue
                key=name+'/'+str(index)
                def capture(gradient,key=key):
                    v=gradient.detach()
                    energy=v.square().sum()
                    n=v.numel()
                    self.records[key]={'gradient_RMS':float((energy/max(n,1)).sqrt()),
                        'elements':n,'finite':bool(torch.isfinite(v).all())}
                    if v.ndim>=3:
                        atomic=v.square().flatten(2).sum(2)
                        if atomic.shape[1]>0:
                            top=atomic.topk(max(1,atomic.shape[1]//4),dim=1).values.sum(1)
                            self.records[key]['top_quarter_slot_gradient_energy_fraction']=float((top/atomic.sum(1).clamp_min(1e-30)).mean())
                    # Returning None leaves the original gradient intact.
                value.register_hook(capture)
        return forward_hook

    def reset(self):self.records={}
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
