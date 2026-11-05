"""Read-only native teacher/score/inpainting validation on saved candidates.

No sampler or scientific-budget reservation is called by this entry point.
"""
import argparse
import copy
import fcntl
from pathlib import Path
from types import SimpleNamespace
import torch
from agenticgen.uniguide.io import read,sha,frozen
from agenticgen.uniguide.native_tools import load_pt,device,tensor_digest
from agenticgen.uniguide.batching import take
from agenticgen.uniguide import iterative_affinity as iteration


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source-attempt',required=True)
    p.add_argument('--output',required=True);p.add_argument('--lock',required=True);args=p.parse_args()
    source=Path(args.source_attempt);out=Path(args.output)
    if out.exists():raise RuntimeError('Use an independent validation directory; no silent revalidation')
    out.mkdir(parents=True)
    cfg=iteration.validate(read(args.config));cfg['root']=str(out/'runtime')
    if cfg.get('plip_dependencies'):
        import sys
        sys.path.insert(0,cfg['plip_dependencies'])
    paths=['BASELINE.json','SCALE_SOURCE.json','rounds/001/records.json','rounds/001/terminal.pt.gz',
        'rounds/001/initial.pt.gz','rounds/001/molecules_all_built.sdf','rounds/001/PROMOTION_AUDIT.json']
    hashes={name:sha(source/name) for name in paths}
    with open(args.lock,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from agenticgen.uniguide.prepare import load_model
        from agenticgen.uniguide.sampler import population
        from agenticgen.uniguide.quality import built_molecules
        from agenticgen.uniguide.profiling import quality,plip_profile
        from agenticgen.uniguide.teacher_preparation import prepare
        from agenticgen.uniguide.affinity_inputs import audit_teacher,audit_initial
        from agenticgen.uniguide.absolute_coordinate import match,prepare as geometry
        from agenticgen.uniguide.geometry_chemistry import ChemicalStateInpainting,condition_features,policy
        baseline=read(source/'BASELINE.json');target=cfg['conditions'][0]['teacher_id']
        template=next(x for x in read(Path(cfg['generation_context_root'])/'teachers.json') if x['target']==target)
        model,_,_=load_model(cfg,template,datasets='prior')
        initial=load_pt(source/'rounds/001/initial.pt.gz');terminal=load_pt(source/'rounds/001/terminal.pt.gz')
        rows=read(source/'rounds/001/records.json');mols=built_molecules(source/'rounds/001/molecules_all_built.sdf')
        pocket=device(initial['pocket_target']);com=initial['target_com'].cuda()
        scores=iteration.rescore(model,terminal,pocket,com)
        delta=max(abs(a-b['pic50_on_rescore']) for a,b in zip(scores,rows))
        if len(rows)!=50 or len(scores)!=50 or delta>1e-5:raise ValueError('Saved scientific endpoint scores changed')
        reference=population(cfg,target,50);checks=[]
        eligible=iteration.shortlist(rows,baseline['pic50'])
        if not eligible:raise ValueError('No saved eligible promotion candidates to validate')
        for row in eligible:
            folder=out/'candidates'/row['node_id'];folder.mkdir(parents=True)
            mol=mols[row['node_id']];pb=quality(mol,template['protein'])
            item=dict(node_id=row['node_id'],pic50=row['pic50_on_rescore'],posebusters=pb)
            if not pb.get('posebusters_pass'):
                checks.append(item);continue
            profile=plip_profile(mol,template['protein'],folder/'plip')
            if not profile['core_atoms']:raise ValueError('Previously passing candidate has no usable interaction site')
            ids=[str(a.GetAtomMapNum()) if a.GetAtomMapNum() else 'atom_%04d'%a.GetIdx() for a in mol.GetAtoms()]
            features,condition,graph=prepare(model,mol,[ids[i] for i in profile['core_atoms']],template['protein'])
            teacher=dict(id=row['node_id'],source_id=row['source_id'],features=features,condition=condition)
            teacher_audit=audit_teacher(model,teacher,template['protein'])
            generation_audit=audit_initial(initial,reference,condition)
            slots=match(terminal['coords'],condition,terminal['mask'].bool())
            controller=SimpleNamespace(sampling_policy=dict(chemical_inpainting=policy(cfg)),absolute_config=cfg,
                geometry_condition=condition,model=model,folder=folder,rows=[{}],
                absolute_state=SimpleNamespace(rows=[dict(slots=x) for x in slots.tolist()]),store=lambda *a:None)
            ChemicalStateInpainting.configure_chemistry(controller)
            before=tensor_digest(terminal);rng=torch.get_rng_state().clone()
            for step in (0,99):
                if ChemicalStateInpainting.apply_chemical_inpainting(controller,terminal,step) is not terminal:
                    raise ValueError('Native endpoint chemistry was overwritten')
            for step in (1,98):
                projected=ChemicalStateInpainting.apply_chemical_inpainting(controller,terminal,step)
                if projected['coords'] is not terminal['coords'] or projected['mask'] is not terminal['mask']:
                    raise ValueError('Chemical projection changed coordinates or masks')
                if any(any(v!=1 for v in values) for name,values in controller.rows[-1].items() if name.endswith('_after_match_fraction')):
                    raise ValueError('Promoted chemical condition was not completely applied')
            if before!=tensor_digest(terminal) or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Projection mutated the source/RNG')
            full=copy.deepcopy(features);full['controlled_atom_ids']=[a['id'] for a in features['atoms']]
            full_condition=geometry(full,template['protein']);channels,_,_=condition_features(model,full_condition)
            endpoint=iteration.teacher_endpoint(channels,features['coordinates_angstrom'],'cuda')
            diagnostic_score=iteration.rescore(model,endpoint,take(pocket,[0],50),com[:1])[0]
            item.update(prepared=True,graph=graph,teacher=teacher_audit,generation=generation_audit,
                source_rmsd=row['condition_aligned_rmsd_angstrom'],new_condition_sha256=condition['condition_sha256'],
                sdf_endpoint_score_diagnostic=diagnostic_score,frozen_promotion_score=row['pic50_on_rescore'],
                score_difference_after_sdf_roundtrip=diagnostic_score-row['pic50_on_rescore'],
                projection_steps_checked=[0,1,98,99],source_unchanged=True)
            frozen(folder/'FEATURES.json',features);frozen(folder/'CONDITION.json',condition)
            frozen(folder/'VALIDATION.json',item);checks.append(item)
        if not any(x.get('prepared') for x in checks):raise ValueError('No promoted teacher passed the full input pipeline')
    if any(sha(source/name)!=digest for name,digest in hashes.items()):raise ValueError('Saved source candidates changed')
    result=dict(complete=True,generated_candidates=0,scientific_reservations=0,agent_calls=0,
        source_attempt=str(source),source_files=hashes,baseline_pic50=baseline['pic50'],
        endpoint_rows_checked=50,maximum_endpoint_score_difference=delta,candidates=checks)
    frozen(out/'VALIDATION_COMPLETE.json',result)
    print('PROMOTED_TEACHER_VALIDATION:'+__import__('json').dumps(dict(complete=True,prepared=sum(bool(x.get('prepared')) for x in checks),
        generated_candidates=0,maximum_endpoint_score_difference=delta)),flush=True)


if __name__=='__main__':main()
