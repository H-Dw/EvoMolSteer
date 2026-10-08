"""Reassess early scientific proposals against the matched implementation control."""
import argparse
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.path_evaluation import paired_effect
from dispatch_path_round import verify_retention


def audit(directory):
    root=Path(directory).resolve();control_round=6
    inputs={}
    def load(number):
        folder=root/f'round{number:02d}';verify_retention(folder/'retention.json')
        inputs[str(number)]={'metrics_sha256':digest(folder/'candidate_metrics.csv'),
            'inference_commit':read_json(folder/'execution_report.json')['code_commit']}
        frame=pd.read_csv(folder/'candidate_metrics.csv')
        return frame[frame.arm.eq('gradient')],read_json(folder/'execution_report.json')
    control,execution=load(control_round);rows={}
    for number in [2,3,4,5]:
        candidate,candidate_execution=load(number)
        if candidate_execution['initial_state_signatures']!=execution['initial_state_signatures']:
            raise ValueError('Implementation control initial states differ')
        rows[str(number)]=paired_effect(candidate,control)
    output={'implementation_negative_control_round':control_round,'paired_effects':rows,'inputs':inputs,
        'interpretation':['The control has no newly learned spatial or selection rule.',
            'Round04 improvement versus R26 is reproduced by the normalization control.',
            'Round02 is positive on average against this control but not in both batches.',
            'These are adaptive screening comparisons, not independent confirmations.'],
        'inference_or_affinity_gradient_added':False}
    write_json(root/'early_implementation_control_audit.json',output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',required=True)
    result=audit(p.parse_args().reports)
    print({key:value['paired_mean_pic50'] for key,value in result['paired_effects'].items()})
