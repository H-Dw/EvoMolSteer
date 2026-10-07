import argparse
from pathlib import Path
from evomolsteer.generation.path_evaluation import retain_round
from evomolsteer.generation.terminal_evaluation import evaluate_terminal
from evomolsteer.io import read_json
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['dataset','campaign','work','output','credit']:p.add_argument('--'+key,required=True)
    p.add_argument('--quality-reference',default='configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz')
    p.add_argument('--baseline')
    a=p.parse_args();cfg=read_json(Path(a.dataset)/'results'/a.campaign/'config.json')['experiment']
    if not (Path(a.work)/'terminal_report.json').exists():
        evaluate_terminal(a.dataset,a.campaign,a.quality_reference,a.work,cfg['arms'].split(','),cfg['batch_indices'],workers=1)
    print(retain_round(a.dataset,a.campaign,a.work,a.output,read_json(a.credit)['threshold_pic50'],a.baseline))
