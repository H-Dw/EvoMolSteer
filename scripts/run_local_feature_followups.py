"""At most two sequential local API follow-ups, driven by completed pilot data.

No free code synthesis: one observable experiment, then one dose experiment.
New programs are committed/pushed before the remote pulls and executes them.
Identical numerical functions share results; no extra random priors are made.
Credentials live only in inherited process environment.
"""
import argparse
import copy
import json
import inspect
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from evomolsteer.io import read_json,write_json
from evomolsteer.ancestral_modes.programs import numerical_identity
from deploy_path_replay import connect,command


def invoke(repo,args,log):
    p=subprocess.run([sys.executable,*args],cwd=repo,capture_output=True,text=True,encoding='utf-8')
    log.parent.mkdir(parents=True,exist_ok=True);log.write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError('Local stage failed; see '+str(log))
    return p.stdout


def queue_snapshot(work,proc_root='/proc'):
    from pathlib import Path
    import json
    folder=Path(work);status=folder/'queue_status.json'
    if status.exists():return json.loads(status.read_text())
    pid=folder/'queue.pid';active=False
    if pid.exists():
        command=Path(proc_root)/pid.read_text().strip()/'cmdline'
        active=command.exists() and b'evomolsteer.path_replay.local_research' in command.read_bytes()
    log=folder/'queue.log'
    return {'status':'starting' if active else 'failed_to_start','pid':pid.read_text().strip() if pid.exists() else None,
            'detail':'Queue imports may precede its first atomic status file.','log_tail':log.read_text()[-1200:] if log.exists() else None}


def wait_collect(repo,manifest_path,study,poll):
    m=read_json(manifest_path);work=m['remote_work']
    started=time.monotonic();network_errors=0
    while True:
        try:
            c=connect()
            try:
                code=inspect.getsource(queue_snapshot)+'\nimport json\nprint(json.dumps(queue_snapshot('+repr(work)+')))'
                state=json.loads(command(c,'python -c '+shlex.quote(code)))
            finally:c.close()
            network_errors=0
        except (OSError,RuntimeError) as error:
            network_errors+=1
            if network_errors>=5:raise
            write_json(study/'network_retry.json',{'attempt':network_errors,'error':str(error),'unix':time.time()})
            time.sleep(poll);continue
        write_json(study/'remote_observation.json',state)
        if state['status'] in ('complete','complete_with_errors'):break
        if state['status'] in ('blocked','failed_to_start'):raise RuntimeError('Remote startup/shared validation failed; see remote_observation.json')
        if state['status']=='starting' and time.monotonic()-started>300:raise RuntimeError('Queue startup exceeded 300 seconds')
        time.sleep(poll)
    invoke(repo,['scripts/deploy_local_feature_research.py','collect','--repo','.', '--manifest',str(manifest_path)],study/'collect.log')
    local=repo/'data/generated'/m['name']
    invoke(repo,['-m','evomolsteer.path_replay.local_research','analyze','--repo','.', '--manifest',str(manifest_path),'--work',str(local)],study/'analysis.log')
    return m,local,read_json(local/'analysis/summary.json')


def git_record(repo,master,label,log):
    # Only owned research artifacts/configs, never shared source or unrelated work.
    owned=(master['campaign_study'].replace('\\','/'),master['config_dir'].replace('\\','/'))
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
    if any(not any(p==root or p.startswith(root+'/') for root in owned) for p in staged):
        raise RuntimeError('Unrelated staged work exists; refusing to include it in an experiment commit')
    args=['git','add',*owned]
    subprocess.run(args,cwd=repo,check=True,capture_output=True)
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode
    if changed:
        subprocess.run(['git','commit','-m','experiment(local-feature): '+label+' freeze API decision and paired evidence'],cwd=repo,check=True,capture_output=True)
    p=subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7897','-c','http.version=HTTP/1.1','push','origin','main'],cwd=repo,capture_output=True,text=True)
    log.write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError('Push failed; no remote launch from unpushed code')


def calibrated_factors(target,observed,bounds=(.1,2.)):
    if not target or not observed or not math.isfinite(target+observed) or min(target,observed)<=0:
        raise ValueError('Finite measured positive support doses required')
    lower,upper=bounds
    if not 0<lower<=1<=upper:raise ValueError('Dose factor bounds must include the parent')
    factor=min(upper,max(lower,target/observed))
    return factor,sorted({1.,factor,math.sqrt(factor)})


def declined(outcome,parent,tolerance=.03):
    if outcome['best_valid'] is None:return parent['best_valid'] is not None
    return parent['best_valid'] is not None and outcome['best_valid']<parent['best_valid']-tolerance and outcome['mean']<parent['mean']-tolerance


def run(repo,master_path,first_manifest,poll,label_suffix=''):
    repo=Path(repo).resolve();master=read_json(master_path);study=repo/master['campaign_study'];bank=read_json(repo/master['teacher_bank'])
    state={'status':'waiting_for_first_pilot','followups':[],'max_followups':2,'poll_seconds':poll}
    state_path=study/'followup_state.json'
    if state_path.exists():write_json(study/'followup_history'/f'{time.time_ns()}.json',read_json(state_path))
    state['label_suffix']=label_suffix;write_json(state_path,state)
    current_path=Path(first_manifest).resolve();declines=0
    try:
        current,local,result=wait_collect(repo,current_path,study,poll)
        original_controls=current_path.relative_to(repo).as_posix()
        parent=next(c for c in current['conditions'] if c['kind']!='unguided' and c.get('role')!='frozen_current_reference')
        initial={r['condition']:r for r in result['performance']}
        tolerance=master.get('rollback_tolerance_pic50',.03)
        declines=int(declined(initial[parent['id']],initial['R02_reference'],tolerance))
        for phase,label in [('observable','L02_local_observable'),('dose','L03_local_dose')]:
            label+=label_suffix
            completed={r['id']:r for r in result['status']}
            if any(completed[c['id']]['complete']!=c['n'] for c in current['conditions']):
                raise RuntimeError('Pilot contains failed/incomplete data; no speculative next API call')
            parent_program=read_json(repo/parent['program']);region=parent_program['mode_specs'][0]['region_id']
            m=copy.deepcopy(master);m.update(initial_program=parent['program'],fixed_region_id=region,reuse_parent=False,
                research_phase=phase,comparison_scope=current['comparison_scope'],diagnostic_evidence=current['diagnostic_evidence'],
                native_zero_repeat_sources=current['native_zero_repeat_sources'],
                parent_feedback={'manifest':current_path.relative_to(repo).as_posix(),'work':local.relative_to(repo).as_posix(),'condition':parent['id']},
                reuse_current_controls={'manifest':original_controls})
            metrics={r['condition']:r for r in result['performance']}
            m['prior_campaign_evidence']={'source':str((local/'analysis/summary.json').relative_to(repo)),
                'comparison_scope':current['comparison_scope'],'current_paired_performance':result['performance'],
                'current_parent_response':next((r for r in result['responses'] if r['condition']==parent['id']),None),
                'limits':'Selected B50 diagnostic, not a passed strict historical replay or N1000 conclusion.'}
            if phase=='observable':
                operators=['pair_distance','pocket_distance']
                m['experiment_contract']={'primary_intervention':'one explicit local geometric observable, fixed previously selected support',
                    'frozen':'teacher, teacher clock, window, dose schedule, strength, and original B50 inputs',
                    'actual_comparators':['same-runtime zero replay','same-runtime frozen R02','completed local pointwise parent'],
                    'active_feature_rule':'Pair-distance or pocket-distance must be the actual registered operator, not just narrative metadata.'}
            else:
                operators=[parent_program['operator']]
                responses={r['condition']:r for r in result['responses']}
                target=responses['R02_reference'].get('direct_injected_rms_A_nonzero_mean',responses['R02_reference'].get('direct_injected_rms_A_mean'))
                observed=responses[parent['id']].get('direct_injected_rms_A_nonzero_mean',responses[parent['id']].get('direct_injected_rms_A_mean'))
                factor,m['dose_scale_multipliers']=calibrated_factors(target,observed,master['dose_factor_bounds'])
                m['experiment_contract']={'primary_intervention':'post-RMS requested displacement strength only',
                    'frozen':'selected support, operator, teacher, clock, window, schedule and input paths',
                    'calibration':{'reference_direct_rms_A':target,'parent_direct_rms_A':observed,'target_over_parent':factor},
                    'dose_coverage':'Condition on nonzero seed updates. Report coverage separately; do not mistake a smaller active fraction for weaker per-active-seed displacement.',
                    'warning':'Clipping and input-coordinate VJP coupling mean a multiplier need not change actual local displacement proportionally. Returning the parent function is allowed and reused.'}
            path=repo/master['config_dir']/(label+'.master.json')
            next_path=repo/master['config_dir']/(label+'.json')
            existing_launch=study/label/'launch_receipt.json'
            resumed=next_path.exists() and existing_launch.exists()
            if not resumed:
                write_json(path,m)
                state.update(status='local_api',current=label);write_json(state_path,state)
                args=['scripts/prepare_local_feature_research.py','--repo','.', '--master',str(path),'--round',label]
                for op in operators:args+=['--operator',op]
                try:invoke(repo,args,study/(label+'.api.log'))
                except Exception:
                    state['followups'].append({'label':label,'status':'skipped_interface_error','error':traceback.format_exc(),
                        'scientific_conclusion':'No inference; interface failure is not a negative generation result.'})
                    write_json(state_path,state);continue
            new=read_json(next_path)
            candidate=next(c for c in new['conditions'] if c['kind']!='unguided' and c.get('role')!='frozen_current_reference')
            p=read_json(repo/candidate['program'])
            if resumed:
                from evomolsteer.path_replay.local_research import load
                load(repo,next_path)
                delivered=read_json(study/label/'agents/delivery.json')
                a=read_json(study/label/'agents'/delivered['analyst_response_path'])
                if a['parent_program_id']!=parent_program['program_id']:raise ValueError('Existing launch belongs to a different Analyst parent')
            same=numerical_identity(p,bank)==numerical_identity(parent_program,bank)
            row={'label':label,'parent':parent['id'],'manifest':next_path.relative_to(repo).as_posix(),'reused_parent_function':same,
                 'program_id':p['program_id'],'operator':p['operator'],'slots':p['mode_specs'][0]['atom_slots'],'dose':p['dose'],'resumed_existing_launch':resumed}
            if same:
                row['reason']='Agent returned the same executable numerical function; no duplicate inference.'
                state['followups'].append(row);git_record(repo,m,label,study/(label+'.push.log'));break
            if not resumed:
                git_record(repo,m,label,study/(label+'.push.log'))
                invoke(repo,['scripts/deploy_local_feature_research.py','deploy','--repo','.', '--manifest',str(next_path)],study/(label+'.deploy.log'))
            state.update(status='remote_inference',current=label);write_json(state_path,state)
            next_current,next_local,next_result=wait_collect(repo,next_path,study,poll)
            counts={r['id']:r for r in next_result['status']}
            if counts[candidate['id']]['complete']!=candidate['n']:
                row.update(status='skipped_inference_error',counts=counts[candidate['id']]);state['followups'].append(row)
                write_json(state_path,state);continue
            current,local,result=next_current,next_local,next_result
            outcome=next(r for r in result['performance'] if r['condition']==candidate['id'])
            old=metrics[parent['id']]
            worse=declined(outcome,old,tolerance)
            declines=declines+1 if worse else 0
            row.update(performance=outcome,paired=result['paired'],consecutive_declines=declines)
            state['followups'].append(row);write_json(state_path,state)
            if declines>=2:
                state['rollback']='Retain frozen R02 incumbent; two consecutive affinity declines. Failed programs stay as immutable evidence, not active defaults.';break
            current_path=next_path;parent=candidate
        state.update(status='complete',current=None,finished_unix=time.time());write_json(state_path,state)
        git_record(repo,master,'record sequential exploration outcomes',study/'final.push.log')
    except BaseException:
        state.update(status='failed',error=traceback.format_exc(),finished_unix=time.time());write_json(state_path,state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--master',type=Path,required=True);p.add_argument('--first-manifest',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=60);p.add_argument('--label-suffix',default='');a=p.parse_args()
    if a.poll_seconds<20:raise ValueError('Avoid tight remote polling')
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]*',a.label_suffix):raise ValueError('Use a safe retry suffix')
    run(a.repo,a.master,a.first_manifest,a.poll_seconds,a.label_suffix)


if __name__=='__main__':main()
