"""Fixed-scale, fresh-noise teacher iteration. No search Agent or hidden retries."""
import copy
import functools
import math
import random
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .io import read,write,frozen,sha,value_sha,Budget
from .native_tools import cpu,device,load_pt,save_pt,tensor_digest
from . import condition_evaluation as fitted

POLICY='iterative-affinity-teacher-v2'


def seeds(round_number):
    # Blocks zero through four belong to the historical search; block six is acceptance.
    block=round_number if round_number else 6
    return [42+100003*(5*block+k) for k in range(5)]


def shortlist(records,score):
    eligible=[r for r in records if r.get('build_success') and r.get('connected')
        and all(type(r.get(k)) in (int,float) and math.isfinite(r[k]) for k in (fitted.METRIC,'pic50_on_rescore'))
        and r[fitted.METRIC]<.5 and r['pic50_on_rescore']>score]
    return sorted(eligible,key=lambda r:(-r['pic50_on_rescore'],r['node_id']))[:10]


def select(records,current,assess,make_condition):
    receipts=[];choices=[]
    for row in shortlist(records,current['pic50']):
        item=dict(node_id=row['node_id'],pic50=row['pic50_on_rescore'],rmsd=row[fitted.METRIC])
        try:
            quality=assess(row);item['quality']=quality
            if quality.get('posebusters_pass') is True and quality.get('posebusters_evaluable',True):
                condition=make_condition(row);choices.append((row,condition))
                item['usable_condition']=True
        except Exception as error:
            item.update(usable_condition=False,error_type=type(error).__name__,error=str(error))
        receipts.append(item)
    return (choices[0] if choices else None),receipts


def teacher_endpoint(channels,coordinates,device):
    """Build the full ligand used for baseline scoring, with native integer counts."""
    endpoint={k:v[None].to(device).float() for k,v in channels.items()}
    endpoint['coords']=torch.tensor(coordinates,device=device,dtype=torch.float32)[None]
    # FLOWR sums this mask and uses the result as an nn.Embedding index.
    endpoint['mask']=torch.ones(endpoint['coords'].shape[:2],device=device,dtype=torch.long)
    return endpoint


def rescore(model,output,pocket,com):
    """Same decoded endpoint and t=.9999 definition as runtime.scoring, no file mutation."""
    from .batching import take
    b=len(output['coords']);scores=[];before=tensor_digest(__import__('agenticgen.runtime.rng',fromlist=['rng_state']).rng_state())
    with torch.no_grad():
        for start in range(0,b,10):
            ids=list(range(start,min(start+10,b)));n=len(ids)
            # Keep the deployed network microbatch shape even for the baseline teacher.
            padded=ids+[ids[-1]]*(10-n)
            final={k:v for k,v in take(output,padded,b).items() if torch.is_tensor(v) and k!='affinity'}
            final['coords']=(final['coords']-com[padded,None,:])/model.coord_scale
            for k in ('atomics','bonds','charges','hybridization'):
                if k in final:final[k]=torch.nn.functional.one_hot(final[k].argmax(-1),final[k].shape[-1]).float()
            times=[torch.full((10,),.9999,device=final['coords'].device) for _ in range(3)]
            result=model._predict_affinity(final,final,take(pocket,padded,b),times)
            scores.extend(result['affinity']['pic50'].detach().cpu().reshape(-1).tolist()[:n])
    if before!=tensor_digest(__import__('agenticgen.runtime.rng',fromlist=['rng_state']).rng_state()):
        raise ValueError('Endpoint rescore consumed sampling RNG')
    if not all(math.isfinite(s) for s in scores):raise FloatingPointError('Nonfinite endpoint pIC50')
    return scores


def fresh_initial(config,target,n,round_number,prior_sampler,symmetrize):
    from flowr.util.molrepr import GeometricMolBatch
    from .sampler import population
    from agenticgen.runtime.rng import rng_state,set_rng
    initial=population(config,target,50);old=rng_state();mols=[];streams=[]
    try:
        for seed in seeds(round_number):
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            # Preserve native twenty-row prior draws; use the first ten slots.
            group=[prior_sampler.sample_molecule(n,symmetrize=symmetrize) for _ in range(20)]
            mols.extend(group[:10]);streams.append(rng_state())
        batch=GeometricMolBatch.from_list(mols)
        prior={k:getattr(batch,k) for k in ('coords','atomics','bonds','charges','mask')}
        # GeometricMolBatch builds a float padding mask; FLOWR's size embedding
        # requires the integral atom-count contract of the frozen native priors.
        prior['mask']=prior['mask'].long()
        if getattr(batch,'hybridization',None) is not None:prior['hybridization']=batch.hybridization
        for key in ('interactions','fragment_mode'):
            if key in initial['prior']:prior[key]=initial['prior'][key]
        if 'fragment_mask' in initial['prior']:
            previous=initial['prior']['fragment_mask']
            prior['fragment_mask']=torch.zeros((50,n),dtype=previous.dtype) if torch.is_tensor(previous) else previous
        initial.update(prior=cpu(prior),stream_states=streams)
        return initial
    finally:set_rng(old)


def validate(config):
    from .guidance_windows import SIX
    from .geometry_chemistry import policy
    if (config.get('experiment_mode')!=POLICY or config.get('rounds')!=5 or config.get('population')!=50
        or config.get('teacher_iteration')!=dict(rmsd_max_angstrom=.5,pb_top_k=10,rounds=5,population=50)
        or config.get('windows')!=[list(w) for w in SIX] or not fitted.enabled(config)
        or config.get('agent_tools') is not None or config.get('output_policy')!='raw999'
        or config.get('chemical_inpainting')!=dict(enabled=True,activation='guidance_phase')
        or config.get('prefix_inpainting')!={'enabled':False}):
        raise ValueError('Iterative teacher configuration differs from its fixed protocol')
    policy(config)
    if len(config['conditions'])!=1:raise ValueError('One teacher per iterative job')
    result=dict(config,initials_root=config['generation_context_root']);result.pop('agent_tools',None)
    return result


def run(config):
    from .prepare import load_model
    from .sampler import population,sample
    from .geometry_sampling import GeometryControl,scale_binding,terminal_records
    from .absolute_coordinate import prepare
    from .geometry_condition import from_molecule
    from .geometry_chemistry import condition_features
    from .box_search import early_shape
    from .profiling import quality,plip_profile
    from .quality import built_molecules
    from .batching import take
    config=validate(config)
    # Use the same PB/PLIP dependency bundle as experiment 1's quality workers.
    if config.get('plip_dependencies'):
        import sys
        sys.path.insert(0,config['plip_dependencies'])
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
    frozen(root/'GEOMETRY_CONFIG.json',config)
    frozen(root/'SOURCE_BINDING.json',dict(implementation={name:sha(Path(__file__).with_name(name)) for name in
        ['iterative_affinity.py','geometry_chemistry.py','geometry_sampling.py','condition_evaluation.py','box_search.py']},
        checkpoint_sha256=config['checkpoint_sha256']))
    target=config['conditions'][0]['teacher_id']
    template=next(t for t in read(Path(config['generation_context_root'])/'teachers.json') if t['target']==target)
    source=Path(config['fixed_scale_source']);final=read(source/'versions/GD01/FINALIZED.json')
    if not final['complete'] or final['configured_rounds']!=10:raise ValueError('Experiment 1 is not complete')
    representative=final['teachers'][target]['representative'];measured=read(source/representative/'aggregate.json')
    score=fitted.score(measured);schedule=measured['effective_guidance_scale_schedule']
    frozen(root/'SCALE_SOURCE.json',dict(source_root=str(source),source_id=representative,mean=score,
        scale=schedule,final_sha256=sha(source/'versions/GD01/FINALIZED.json'),aggregate_sha256=sha(source/representative/'aggregate.json')))
    frozen(root/'ATTEMPT_LIMITS.json',dict(stage_caps=dict(formal=250),per_teacher_stage_caps=dict(formal=250)))
    write(root/'BUDGET_SCOPE.json',dict(stage='formal',teacher_id=target))
    features=read(config['conditions'][0]['features_path']);condition=prepare(features,template['protein'])
    model,prior_sampler,symmetrize=load_model(config,template,datasets='prior')
    initial=population(config,target,50);first=device(take(initial['pocket_target'],[0],50));com=initial['target_com'][:1].to('cuda')
    source_mol=next(m for m in Chem.SDMolSupplier(template['sdf'],removeHs=True) if m is not None)
    baseline_features=from_molecule(source_mol,[],coordinate_frame='generation_context_angstrom')
    baseline_features['controlled_atom_ids']=[a['id'] for a in baseline_features['atoms']]
    # The affinity scorer needs the complete ligand; this is not the inpainting scope.
    scoring_condition=prepare(baseline_features,template['protein'])
    baseline_channels,_,_=condition_features(model,scoring_condition)
    teacher_output=teacher_endpoint(baseline_channels,baseline_features['coordinates_angstrom'],'cuda')
    baseline=rescore(model,teacher_output,first,com)[0]
    current=dict(id='initial',pic50=baseline,condition=condition,features=features,source_id=template['node_id'])
    frozen(root/'BASELINE.json',dict(current,historical_pic50=template['pic50'],scoring='decoded-endpoint-t0.9999',checkpoint_sha256=config['checkpoint_sha256']))

    def generate(round_number,destination,teacher):
        n=len(teacher['condition']['atoms'])
        init=fresh_initial(config,target,n,round_number,prior_sampler,symmetrize)
        save_pt(destination/'initial.pt.gz',init)
        frozen(destination/'INPUT.json',dict(teacher_id=teacher['id'],condition_sha256=teacher['condition']['condition_sha256'],
            chemical_template_sha256=value_sha(teacher['features']),
            chemical_controlled_atom_ids=teacher['condition']['controlled_atom_ids'],seeds=seeds(round_number),scale=schedule,initial_sha256=sha(destination/'initial.pt.gz')))
        binding=scale_binding(schedule,condition=teacher['condition'],window_bounds=config['windows'],early_shape=early_shape(config))
        output,pt,_,_=sample(model,config,template,schedule,destination,50,initial_override=init,
            control_factory=functools.partial(GeometryControl,condition=teacher['condition'],scale_binding=binding),check_affinity=False)
        rows=terminal_records(model,output,teacher['condition'],destination,50,config=config)
        scores=rescore(model,output,pt,init['target_com'].to('cuda'))
        for i,r in enumerate(rows):
            r.update(pic50_on_rescore=scores[i],logical_seed=seeds(round_number)[i//10],round=round_number,
                teacher_id=teacher['id'],source_id=destination.relative_to(root).as_posix()+'/'+r['node_id'])
            raw=output.get('affinity',{}).get('pic50')
            if raw is not None:r['pic50_on_upstream']=float(raw.reshape(-1)[i].detach().cpu())
        write(destination/'records.json',rows)
        return rows

    # New chemistry/score path must pass a real complete trajectory before science.
    acceptance=root/'acceptance'
    if not (acceptance/'COMPLETE.json').exists():
        if (acceptance/'INPUT.json').exists():raise RuntimeError('Interrupted acceptance; no automatic resampling')
        rows=generate(0,acceptance,current)
        steps=read(acceptance/'steps.json')
        if len(steps)!=100:raise ValueError('Acceptance did not complete full inference')
        for step in steps:
            active=1<=step['step']<=98
            if any(flag!=active for flag in step['chemical_inpainting_applied']):raise ValueError('Chemical endpoint policy changed')
            if active:
                for name,values in step.items():
                    if name.startswith('chemical_') and name.endswith('_after_match_fraction') and any(v!=1 for v in values):
                        raise ValueError('Interaction region was not completely inpainted')
        frozen(acceptance/'COMPLETE.json',dict(engineering_only=True,trajectory_evaluations=50,records_sha256=sha(acceptance/'records.json'),
            chemical_inpainting_scope='controlled_atoms_and_induced_bonds',
            terminal_projection=False,complete_terminal_rescore=True))
    for iteration in range(1,6):
        out=root/'rounds'/f'{iteration:03d}';done=out/'ROUND_COMPLETE.json'
        if done.exists():
            receipt=read(done)
            if receipt['input_teacher_sha256']!=value_sha(current):raise ValueError('Teacher replay changed')
            for name,digest in receipt['files'].items():
                if sha(out/name)!=digest:raise ValueError('Round evidence changed')
            current=receipt['output_teacher'];continue
        out.mkdir(parents=True,exist_ok=True)
        if (out/'CLAIM.json').exists():
            interrupted=read(out/'CLAIM.json')
            if interrupted['input_teacher_sha256']!=value_sha(current):raise ValueError('Interrupted parent differs')
            Budget(root).reserve(out.relative_to(root).as_posix(),'formal',50,value_sha(interrupted))
            write(out/'FAILED.json',dict(type='interrupted',attempts=50,no_retry=True))
            frozen(done,dict(input_teacher_sha256=value_sha(current),output_teacher=current,promoted=False,attempts=50,
                failed=True,files={n:sha(out/n) for n in ['CLAIM.json','FAILED.json']}))
            continue
        claim=dict(input_teacher_sha256=value_sha(current),round=iteration,seeds=seeds(iteration),scale=schedule)
        frozen(out/'CLAIM.json',claim);Budget(root).reserve(out.relative_to(root).as_posix(),'formal',50,value_sha(claim))
        old=copy.deepcopy(current)
        try:
            rows=generate(iteration,out,current);mols=built_molecules(out/'molecules_all_built.sdf')
            def assess(row):return quality(mols[row['node_id']],template['protein'])
            def build(row):
                mol=mols[row['node_id']];profile=plip_profile(mol,template['protein'],out/'profiles'/row['node_id'])
                if not profile.get('core_atoms'):raise ValueError('No usable direct PLIP site')
                atom_ids=[str(a.GetAtomMapNum()) if a.GetAtomMapNum() else 'atom_%04d'%a.GetIdx() for a in mol.GetAtoms()]
                new_features=from_molecule(mol,[atom_ids[i] for i in profile['core_atoms']],coordinate_frame='generation_context_angstrom')
                new_condition=prepare(new_features,template['protein'])
                condition_features(model,new_condition)
                return dict(features=new_features,condition=new_condition,profile=profile)
            choice,pb=select(rows,current,assess,build);write(out/'PROMOTION_AUDIT.json',dict(top_k=10,checks=pb))
            if choice:
                row,prepared=choice
                current=dict(id=f'round_{iteration:03d}',pic50=row['pic50_on_rescore'],source_id=row['source_id'],**prepared)
                save_pt(out/'promoted_teacher.pt.gz',take(load_pt(out/'terminal.pt.gz'),[row['tensor_row']],50))
                with Chem.SDWriter(str(out/'promoted_teacher.sdf')) as writer:writer.write(mols[row['node_id']])
            frozen(done,dict(input_teacher_sha256=value_sha(old),output_teacher=current,promoted=choice is not None,
                attempts=50,files={n:sha(out/n) for n in ['CLAIM.json','INPUT.json','records.json','PROMOTION_AUDIT.json','COORDINATE_ASSIGNMENT.json','promoted_teacher.sdf','promoted_teacher.pt.gz'] if (out/n).exists()}))
        except Exception as error:
            write(out/'FAILED.json',dict(type=type(error).__name__,error=str(error),attempts=50,no_retry=True))
            frozen(done,dict(input_teacher_sha256=value_sha(old),output_teacher=old,promoted=False,attempts=50,
                failed=True,files={n:sha(out/n) for n in ['CLAIM.json','FAILED.json']}));current=old
    rounds=[read(root/'rounds'/f'{r:03d}'/'ROUND_COMPLETE.json') for r in range(1,6)]
    ledger=read(root/'attempts.json')
    if len(ledger)!=5 or sum(x['count'] for x in ledger.values())!=250:raise ValueError('Iterative budget identity changed')
    previous=baseline
    for receipt in rounds:
        now=receipt['output_teacher']['pic50']
        if not math.isfinite(now) or now<previous or (receipt['promoted'] and now<=previous):raise ValueError('Teacher promotion is not strictly improving')
        previous=now
    result=dict(policy=POLICY,teacher=target,rounds=rounds,final_teacher=current,initial_pic50=baseline,
        improvement=current['pic50']-baseline,attempts=250,complete=not any(r.get('failed') for r in rounds))
    frozen(root/'FINALIZED.json',result);frozen(root/'ONLINE_PROTOCOL_AUDIT.json',dict(policy=POLICY,
        scientific_attempts=250,engineering_attempts=50,rounds=5,agent_calls=0,terminal_projection=False,
        chemical_inpainting_scope='controlled_atoms_and_induced_bonds',
        final_sha256=sha(root/'FINALIZED.json')))
    return result
