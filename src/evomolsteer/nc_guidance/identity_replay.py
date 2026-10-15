"""One frozen diagnostic replay, outside optimization, to localize identity loss."""
import argparse
import copy
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.partial_search.engine import Remote,infer
from .search import commit


def run(repo,master,source_round):
    repo=Path(repo).resolve();study=repo/master['campaign_study']
    if not (study/'COMPLETE.json').exists():raise ValueError('Finish official optimization and its remote workers first')
    oldcfg=read_json(repo/master['config_dir']/(source_round+'.json'));src=study/source_round
    oldbinding=read_json(next((src/'candidates').glob('*/bindings.json')))
    oldprogram=repo/oldbinding['programs'][0]['program'];p=read_json(oldprogram)
    label=source_round+'_identity_replay';cfg=copy.deepcopy(oldcfg)
    cfg.update(round_id=label,study=master['campaign_study']+'/'+label,remote_output=master['remote_campaign']+'/'+label)
    cfg['diagnostic_only']=True;cfg['new_reward_proposal']=False
    cfgpath=repo/master['config_dir']/(label+'.json');folder=repo/cfg['study'];target=folder/'candidates'/p['candidate_id']
    target.mkdir(parents=True,exist_ok=True);program=target/'programs'/(p['program_id']+'.json');program.parent.mkdir(exist_ok=True)
    program.write_bytes(oldprogram.read_bytes());teacher=target/'teacher.json';teacher.write_bytes((repo/oldbinding['teacher']).read_bytes())
    binding=copy.deepcopy(oldbinding);binding['programs'][0]['program']=program.relative_to(repo).as_posix();binding['teacher']=teacher.relative_to(repo).as_posix()
    write_json(target/'bindings.json',binding);write_json(cfgpath,cfg)
    write_json(folder/'frozen_replay.json',{'source_round':source_round,'program_bytes_unchanged':digest(program)==digest(oldprogram),
        'source_API_references':{role:(src/(role+'.json')).relative_to(repo).as_posix() for role in ('Analyst','Designer')},
        'new_LLM_calls':0,'new_reward_proposals':0,'purpose':'Save actual last typed assignment and half-state categorical identities; no parameter search'})
    sha=commit(repo,master,'diagnostic(nc-layout): replay frozen '+source_round+' with boundary identity telemetry')
    remote=Remote()
    try:
        output=repo/master['generated_root']/label
        if not (output/'COMPLETE.json').exists():output=infer(repo,master,cfg,cfgpath,remote,sha)
    finally:remote.close()
    output=output/p['program_id'];old=repo/master['generated_root']/source_round/p['program_id']
    audit=read_json(output/'guidance_audit.json');geometry=read_json(output/'geometry.json')
    original=read_json(old/'geometry.json');a=np.asarray(geometry['terminal']['atomic_indices'])
    boundary=next(s for s in audit['steps'] if 'boundary_typed_generated_slots' in s)
    ids=np.asarray(boundary['boundary_typed_generated_slots']);h=np.asarray(boundary['boundary_current_atomic_indices'])
    t=read_json(teacher);wanted=np.asarray([t['atom_vocabulary'][e] for e in p['typed_elements']])
    final=np.take_along_axis(a,ids,1);groups=[]
    for j,(slot,element) in enumerate(zip(p['typed_slots'],p['typed_elements'])):
        match=h[:,j]==wanted[j];after=final[:,j]==wanted[j]
        groups.append({'teacher_slot':slot,'desired_element':element,'n':len(h),
            'correct_in_actual_boundary_state':int(match.sum()),'correct_at_final_same_generated_slot':int(after.sum()),
            'boundary_correct_then_lost':int((match&~after).sum()),'boundary_wrong_then_correct':int((~match&after).sum())})
    oldrecords=read_json(old/'records.json');newrecords=read_json(output/'records.json')
    result={'source_round':source_round,'diagnostic_only':True,'extra_attempts':len(h),'independent_new_seeds':False,
        'same_program_bytes':digest(program)==digest(oldprogram),'inference_commit':sha,
        'terminal_coordinate_max_abs_difference_A':float(np.abs(np.asarray(geometry['terminal']['coords_A'])-np.asarray(original['terminal']['coords_A'])).max()),
        'pic50_max_abs_difference':float(max(abs(o['pic50']-n['pic50']) for o,n in zip(oldrecords,newrecords))),
        'terminal_atomic_labels_identical':bool(np.array_equal(a,np.asarray(original['terminal']['atomic_indices']))),
        'boundary_time':p['window'][1],'groups':groups,
        'last_decoder_probability_before_logit_step':boundary.get('typed_correct_probability'),
        'last_decoder_probability_after_logit_step':boundary.get('typed_probability_after_logit_step'),
        'meaning':'Actual native categorical state after the last active integration; then track those generated indices to the unguided final. No claim of intermediate chemical validity.'}
    write_json(folder/'identity_report.json',result);commit(repo,master,'results(nc-layout): frozen identity replay diagnostics');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True)
    p.add_argument('--source-round',default='R18');a=p.parse_args();print(run(a.repo,read_json(a.config),a.source_round))


if __name__=='__main__':main()
