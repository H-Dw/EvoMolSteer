"""Evaluate four acquisitions proposed by the corrected real-API rule.

This is a second development wave that saw 12 prior C6 outcomes; it is not a
fresh same-budget replacement for the original C6 condition.
"""
import argparse
import os
import shlex
from pathlib import Path

import numpy as np
import paramiko

from evomolsteer.agentic_steer.panel import make_job
from evomolsteer.agentic_steer.rules import evaluate_rule
from evomolsteer.agentic_steer.pilot import remote_trials,command
from evomolsteer.agentic_steer.workflow import observed_trials
from evomolsteer.io import read_json,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--runtime',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args();repo=Path.cwd();cfg=read_json(a.config);study=repo/cfg['study']
    panel=read_json(study/'panel.json');probe=read_json(study/'probes/scientific_contract_v2/coding.json')
    initial=read_json(study/'conditions/C6/initial.execution.json');feedback=read_json(study/'conditions/C6/feedback.execution.json')
    observed={(x['node_id'],r) for x in initial['directions'] for r in range(cfg['initial_replicates'])}|{(x['node_id'],x['replicate']) for x in feedback['next_trials']}
    features={k:np.asarray([c['features'][k] for c in panel['candidate_cards']]) for k in panel['candidate_cards'][0]['features']}
    scores,audit=evaluate_rule(probe['code'],features);requests=[]
    for i in np.argsort(-scores,kind='stable'):
        node=panel['candidate_cards'][int(i)]['node_id']
        available=[r for r in range(cfg['max_replicates']) if (node,r) not in observed]
        if available:requests.append((node,available[0]))
        if len(requests)==cfg['followup_trials']:break
    folder=study/'refinement_v2';folder.mkdir(exist_ok=True)
    make_job(repo,cfg,requests,folder/'acquisitions.job.json');job=read_json(folder/'acquisitions.job.json')
    write_json(folder/'decision.json',{'requests':requests,'rule_audit':audit,'observed_input_trials':len(observed),
        'logical_budget_after_this_wave':len(observed)+len(requests),'source_commit':a.source_commit,
        'scope':'Second development wave; no equal-budget superiority claim'})
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(a.host,port=a.port,username='root',password=os.environ['MOLSTEER_SCNET_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=25)
    client.get_transport().set_keepalive(30)
    try:
        if command(client,'cat '+shlex.quote(a.runtime+'/SOURCE_COMMIT'))!=a.source_commit:raise ValueError('Source commit mismatch')
        new=remote_trials(client,repo,cfg,[job],a.runtime,'refinement_v2')
    finally:client.close()
    observations=observed_trials(study,requests)
    write_json(folder/'results.json',{'new_unique_trials':new,'requests':requests,'observations':observations,
        'mean_pic50':float(np.mean([x['pic50'] for x in observations])),
        'elite_hits':sum(x['elite_hit'] for x in observations),'qualified':sum(x['qualified_structure'] for x in observations)})
    print('Refinement complete:',requests)


if __name__=='__main__':main()
