"""Capture policy for readable Steer datasets; the native sampler is unchanged."""
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..io import digest, read_json
from ..storage.selection_dataset import SelectionLearningWriter, _package_record
from ..storage.transactions import atomic_json


def numeric_array(value):
    if hasattr(value,'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def terminal_numeric_arrays(output):
    arrays={}
    def visit(value,prefix):
        if isinstance(value,Mapping):
            for key,item in value.items():
                visit(item,prefix+'__'+str(key) if prefix else str(key))
        elif hasattr(value,'detach') or isinstance(value,(np.ndarray,np.number,float,int,bool)):
            a=numeric_array(value)
            if a.dtype.kind not in 'biuf':
                raise TypeError('Unsupported terminal numeric dtype: '+prefix)
            arrays[prefix]=a
        else:
            raise TypeError('Unsupported native terminal field: '+prefix)
    visit(output,'')
    return arrays


class LearningTraceMixin:
    """Mixin over the existing Trace; suppress its restart/endpoint pickles.

    Coordinates are unchanged. Atom/charge forecast probabilities are recorded
    in their source dtype, instead of the historical f16 dump. Bond probability
    tensors are optional; hard bond labels are always retained.
    """
    def __init__(self,*args,score_grid,window,codec='none',capture_bond_probabilities=False,**kwargs):
        kwargs['save']=False
        super().__init__(*args,**kwargs)
        self.anchor_steps=set()
        self.capture_bond_probabilities=capture_bond_probabilities
        self.learning_store=SelectionLearningWriter(self.path,score_times=score_grid,window=window,codec=codec,
            metadata={'arm':self.arm,'seed':int(self.seed),'batch':int(self.batch),'coord_scale':float(self.scale),
                      'probability_precision':'native dtype; no added float16 conversion',
                      'bond_probability_tensors':capture_bond_probabilities})

    def score(self,pred,onoff):
        super().score(pred,onoff)
        for key in ('atomics','charges','hybridization'):
            if key in pred:
                self.arr.pop('predicted_'+key+'_probs_f16',None)
                self.arr['predicted_'+key+'_probs'].append(numeric_array(pred[key]))
        if self.capture_bond_probabilities and 'bonds' in pred:
            self.arr['predicted_bonds_probs'].append(numeric_array(pred['bonds']))

    def end(self,curr,cond,times):
        if any(len(v)!=1 for v in self.arr.values()):
            raise ValueError('Expected exactly one complete pre-selection observation')
        fields={k:v[0] for k,v in self.arr.items()}
        fields.update(selected_indices=self.selected.copy(),
                      offspring_count=np.bincount(self.selected,minlength=self.b),
                      resampled=np.uint8(self.did_resample),state_time=np.float32(float(times[0][0])))
        self.learning_store.append(int(self.i),fields)
        super().end(curr,cond,times)

    def finish(self,output):
        super().finish(output)
        atomic_json(self.path/'events.json',self.events)
        self.learning_store.finish(terminal_numeric_arrays(output))


class SteerLearningExtension:
    def __init__(self,cfg,input_files):
        self.cfg=cfg
        self.input_files=input_files

    def add_arguments(self,parser):
        pass

    def prepare(self,opt,out):
        if out.exists():
            # Launcher already created its request and state, but never batches.
            if any((out/arm).exists() for arm in opt.arms.split(',')) or (out/'COMPLETE.json').exists():
                raise FileExistsError('Generated campaign already exists')
        opt.input_files=self.input_files
        opt.verify_passive=False  # archival capture must not launch another inference

    def configure(self,model,opt,out):
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:
            raise ValueError('This capture policy currently supports de novo generation without inpainting')
        self.model,self.opt,self.out=model,opt,Path(out)

    def describe(self):
        return {'schema_version':'steer-selection-learning-1.0','sampling_mode':'selective_smc',
                'selection_window':[self.opt.window_start,self.opt.window],
                'gradient_guidance':False,'post_window_particle_resampling':False,
                'integration_end':1.0,'archive_container':'directory','storage_codec':self.cfg.storage_codec,
                'probability_precision':'native source dtype','bond_probability_tensors':self.cfg.capture_bond_probabilities}

    def amend_config(self,config):
        config['full_probability_steps']=[]
        config['probability_recording']={'events':'every retained scoring event','precision':'native dtype',
                                         'bond_probability_tensors':self.cfg.capture_bond_probabilities,
                                         'terminal':'all native numeric output tensors'}
        config['recording']='All pre-selection candidates inside score-time window; full all-step lineage; native terminal tensors and decoded results'
        config['trajectory_storage']='Readable learning directory with verified HDF5 and uncompressed event Parquet; no tar/zip envelope'
        config['deviations']='Common target frame, native matched SDE, reference-ligand atom count; user-defined window and population, not claimed identical to paper settings'
        config['input_files']=self.input_files
        config['selection_scope']={'window':[self.opt.window_start,self.opt.window],'bounds':'inclusive score_time',
                                   'proposal_time':'one native step later; not silently truncated at the upper score bound'}

    def instrument(self,model,path):
        from .controller import instrument
        return instrument(model,path)

    def make_trace(self,*args,**kwargs):
        import torch
        from .controller import Trace
        class LearningTrace(LearningTraceMixin,Trace):
            pass
        grid=torch.linspace(0,1,kwargs['steps']+1).numpy()[:-1]
        return LearningTrace(*args,**kwargs,score_grid=grid,window=[self.opt.window_start,self.opt.window],
                             codec=self.cfg.storage_codec,capture_bond_probabilities=self.cfg.capture_bond_probabilities)

    def final_metrics(self,model,output,trace,path):
        from .controller import final_metrics
        rows=final_metrics(model,output,trace,path)
        quality=[]
        for r in rows:
            quality.append({k:v for k,v in r.items() if k in ('node_id','slot','build_success','failure_reason','connected',
                             'heavy_atoms','qed','mw','rings','min_protein_distance_A','pairs_below_1_2A')})
            quality[-1]['energy_status']='not_evaluated'
        atomic_json(path/'quality_records.json',quality)
        atomic_json(path/'QUALITY_COMPLETE.json',{'scope':'native build/sanitization and geometry; no energy/force-field evaluation',
                                                 'records':len(rows),'build_failed':sum(not r['build_success'] for r in rows)})
        manifest=read_json(path/'learning_manifest.json')
        for name in ('events.json','final_records.json','quality_records.json','QUALITY_COMPLETE.json',
                     'molecules_all_built.sdf','molecules_raw_decodable.sdf'):
            manifest['files'].append(_package_record(Path(path),Path(path)/name))
        manifest['campaign_context']=[_package_record(self.out,self.out/name) for name in
                                      ('config.json',f'frame_batch_{trace.batch:03d}.json')]
        atomic_json(path/'learning_manifest.json',manifest)
        return rows
