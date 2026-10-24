"""Server-side V21 prior-selection context repair: scoped checks, audited failed-teacher restart.

Source comes only from GitHub fixed checkout. No arbitrary version/action input.
"""
import argparse
import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from agenticgen.adaptive_schedule.io import read, write, sha, repository, commit
from agenticgen.adaptive_schedule.group_run import assert_prepared
from agenticgen.adaptive_schedule.group_review import review_check
from agenticgen.adaptive_schedule.protocol_state import prepared_path, acceptance_path
from agenticgen.runtime.launch import idle_locks, check_api
from agenticgen.runtime.restart_failed_version import no_worker
from agenticgen.runtime import restart_failed_teacher as restart
from start_reviewed_version import passing_tests

ROOT = Path('/opt/MolSteer/generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022')
AUTH = 'agenticgen/docs/authorizations/20261024_v21_commitment_teacher_rerun.json'
TESTS = ('test_committed_wave.py', 'test_plip_group_agent.py', 'test_plip_council.py',
         'test_failed_teacher_restart.py', 'test_v21_repair_entry.py')


def aborted_costs(op, root):
    j = read(op/'journal.json'); archive = op/'failure_audit'; receipts = {}; unknown = []
    for path in archive.rglob('*.receipt.json'):
        item=read(path); identifier=item.get('provider_request_id'); usage=item.get('usage')
        if not identifier or not usage:
            unknown.append(str(path.relative_to(root))); continue
        if identifier in receipts and receipts[identifier]['usage'] != usage:
            raise ValueError('Conflicting API usage for the same provider ID')
        row=receipts.setdefault(identifier, {'usage':usage,'finish_reason':item.get('finish_reason'),'sources':{}})
        row['sources'][path.relative_to(root).as_posix()]=sha(path)
    pipeline=read(archive/'pipeline.json'); starts={}; durations=[]
    for event in pipeline['events']:
        if event.get('teacher') not in j['failed_teachers']: continue
        key=(event['teacher'],event['round'])
        if event['kind']=='gpu_started': starts[key]=event['time']
        elif event['kind']=='gpu_finished_analysis_queued':
            durations.append({'teacher':key[0],'round':key[1],'seconds':event['time']-starts[key]})
    result={'version':'v21','official_results_included':False,'currency_bill':False,
        'deleted_candidates':j['failed_candidate_count'],'receipts':receipts,'unknown_receipts':unknown,
        'provider_id_deduplicated_receipts':len(receipts),
        'prompt_tokens':sum(r['usage'].get('prompt_tokens',0) for r in receipts.values()),
        'completion_tokens':sum(r['usage'].get('completion_tokens',0) for r in receipts.values()),
        'gpu_waves':durations,'gpu_seconds':sum(d['seconds'] for d in durations),
        'excludes_R0_and_native_acceptance':True}
    write(op/'aborted_resource_usage.json',result)
    return {k:v for k,v in result.items() if k not in ('receipts','gpu_waves')}


def completed_source_commits(journal):
    earlier=journal['old_binding'].get('preserved_teacher_provenance',{})
    return {t:earlier.get(t,{}).get('deployment',journal['old_deployment'])['commit']
            for t in journal['completed_teachers']}


def renewed_approval(old, revision, prepared_sha, acceptance_sha, operation, journal):
    if old.get('mode') != 'best_round_strategy' or old.get('full_review_completed') is not False:
        raise ValueError('Retain the authorized limited review scope')
    if old.get('previous_version')!='v20' or old.get('next_version')!='v21':
        raise ValueError('Only renew existing V20 to V21 approval')
    result=copy.deepcopy(old)
    if 'review_policy_operation' in result:
        result['previous_review_policy_operation']=result.pop('review_policy_operation')
    if 'active_protocol_gate' in result:
        result['previous_active_protocol_gate']=result.pop('active_protocol_gate')
    result.update(review_commit=revision,time=time.time(),new_prepared_sha256=prepared_sha,
        new_acceptance_sha256=acceptance_sha,teacher_restart_operation=operation,
        rerun_failed_teachers=journal['failed_teachers'],preserved_completed_teachers=journal['completed_teachers'],
        preserved_teacher_source_commits=completed_source_commits(journal),analyst_output_limit=98304,
        citation_protocol='private-evidence-alias-v1',
        committed_schedule_context='prior_selected_schedules_v1',
        interface_repair_policy={'private_max_format_repairs':2,'format_policy':'json-interface-v5'},
        frozen_skills_unchanged=True,scientific_settings_unchanged=True,analysis_output_budget_unchanged=True,
        approval_scope='V21 failed teacher five-round restart; not V22 approval')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--commit',required=True)
    args=parser.parse_args(); revision=commit(); repo=repository(); base=ROOT/'versions/v21'
    if os.name!='posix' or not re.fullmatch('[0-9a-f]{40}',args.commit) or args.commit!=revision:
        raise ValueError('Use the fixed installed server commit')
    dep=Path('/opt/MolSteer/operations')/('agenticgen_deploy_'+revision[:8]); dep.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.pop('PYTHONPATH',None)
    env.update(AGENTICGEN_REPOSITORY=str(repo),AGENTICGEN_PYTHON=sys.executable)
    def event(phase):
        import json
        print('AGENTICGEN_EVENT '+json.dumps({'phase':phase,'commit':revision}),flush=True)
    def run(command, name):
        with (dep/name).open('wb') as log:
            result=subprocess.run(command,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('Repair stopped; inspect '+str(dep/name))
    with idle_locks(ROOT):
        no_worker(ROOT)
        if (ROOT/'FINALIZED_v21.json').exists(): raise ValueError('Completed V21 cannot restart')
    if (dep/'launch_receipt.json').exists(): raise ValueError('Repair was already launched')
    write(dep/'api_preflight.json',check_api())  # Fail before staging if login did not inherit API/proxy.
    xml=dep/'related-tests.xml'
    run([sys.executable,'-m','pytest','-q',*[str(repo/'agenticgen/tests/unit'/n) for n in TESTS],
         '--junitxml',str(xml)],'related-tests.log')
    passing_tests(xml);write(dep/'tests_receipt.json',{'commit':revision,'passed':True,'junit_sha256':sha(xml),'tests':list(TESTS),'scope':'prior_selected_schedules_v1'})
    event('related_tests_passed')
    op=ROOT/'operations'/('failed_teacher_rerun_v21_'+revision[:8])
    if not op.exists():
        write(dep/'stage_receipt.json',restart.stage(ROOT,'v21',repo/AUTH))
    if read(op/'journal.json')['commit']!=revision: raise ValueError('Repair operation belongs to another commit')
    event('protocol_staged_no_candidates_deleted')
    if not acceptance_path(ROOT).exists():
        write(dep/'native_reuse_receipt.json',restart.reuse_native_acceptance(ROOT,dep/'tests_receipt.json'))
    restart.native_ready(ROOT);event('native_evidence_reused_scoped_contract_checks_passed')
    write(dep/'finish_receipt.json',restart.finish(ROOT,op)); costs=aborted_costs(op,ROOT)
    write(dep/'aborted_cost_summary.json',costs);event('failed_top1_archived_and_deleted_top2_preserved')
    with idle_locks(ROOT):
        no_worker(ROOT);assert_prepared(ROOT);journal=restart.verify_ready(ROOT,'v21')
        if (base/'binding.json').exists(): raise ValueError('A new worker has already bound this version')
        if not (base/'review_approval.json').exists():
            approval=renewed_approval(read(op/'failure_audit/review_approval.json'),revision,
                sha(prepared_path(ROOT)),sha(acceptance_path(ROOT)),op.relative_to(ROOT).as_posix(),journal)
            write(base/'review_approval.json',approval)
        gate=review_check(ROOT,'v21')
        if gate.get('mode')!='best_round_strategy' or gate.get('sessions')!=4 or gate.get('full_review_completed') is not False:
            raise ValueError('Unexpected prior strategy-review gate')
        write(dep/'approval_receipt.json',{'commit':revision,'gate':gate,'approval_sha256':sha(base/'review_approval.json')})
    run(['bash',str(repo/'agenticgen/scripts/run.sh'),'launch','--root',str(ROOT),'--version','v21'],'launch.log')
    launch=read(ROOT/'launch.json')
    if launch['version']!='v21' or launch['commit']!=revision: raise ValueError('Launch identity differs')
    receipt={k:launch[k] for k in ('version','commit','pid','time','api_check')}
    receipt.update(rerun_teachers=journal['failed_teachers'],preserved_teachers=journal['completed_teachers'],
                   preserved_teacher_source_commits=completed_source_commits(journal),environment='full inherited; values omitted')
    write(dep/'launch_receipt.json',receipt);event('v21_top1_restarted')


if __name__=='__main__': main()
