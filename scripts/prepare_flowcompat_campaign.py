"""Register a fresh budget; completed earlier experiments do not count here."""
import argparse
import shutil
from pathlib import Path
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.generation.selection_workflow import restore_incumbent


def prepare(repo):
    root=Path(repo).resolve();cfg=root/'configs/experiments/flowcompat30_v1';docs=root/'docs/experiments/flowcompat30_20261009'
    if (cfg/'campaign.json').exists():raise FileExistsError('Campaign already registered')
    cfg.mkdir(parents=True,exist_ok=True);docs.mkdir(parents=True,exist_ok=True)
    old=root/'configs/experiments/selection_path20_v1'
    shutil.copy2(old/'baseline_snapshot.json',cfg/'baseline_snapshot.json')
    restore_incumbent(root,'flowcompat30_v1','Fresh R26-based compatibility exploration',0)
    shutil.copy2(root/'docs/experiments/selection_path20_20261008/steer_reference.json',docs/'steer_reference.json')
    protocol={'schema_version':'flowcompat30-protocol-1.0','master_seed':42,'maximum_rounds':30,
        'screening_batches':[47,48],'confirmation_batches':[49,50,51,52,53,54],'n_per_arm':100,
        'schedule':{'1':'Original native/R26 interface','2':'New-interface empty-rule native/R26 numerical control',
            '3-24':'Sequential registered single-axis hypotheses, parameters chosen after prior retention; failed proposals restore R26',
            '25/27/29':'New paired native/R26 controls','26/28/30':'Same frozen candidate, no validation-driven tuning'},
        'screening_acceptance':'Mean versus R26 >.005, both batches positive, valid/PB >=.90; provisional only.',
        'confirmation_acceptance':'Frozen six-batch all-attempt mean versus R26 >.01, batch-bootstrap lower CI >0, and positive versus native; valid/PB drop <=.03; strain median/P90 <=1.3*max(R26,Steer).',
        'tail_threshold_pic50':8.258901977539063,'seed_stream':'42 + 100003 * batch; identical starting coordinates/atomics/bonds',
        'inference':'FLOWR_ROOT_v2,100 steps; dynamically bound learned window then native continuation; no particle Steer/resampling, head derivative, extra production forward, or chemical graph gate.',
        'implementation_feedback':'Literal Agent/tool/source hashes, null-control parity, conditional finite difference, new mechanism diagnostic and actual window response are distinct from efficacy.',
        'control_limit':'Historical Steer is an unpaired reference with design donors and greater scoring/copying compute.',
        'retention':'Preserve compact reports, exact hashes and code. Retire only owned generated tests after published report verification; original Steer/checkpoints protected.',
        'rationale':'Concise scientific hypotheses/formulas and evidence are reproducible design records; private reasoning transcripts are not recorded.'}
    write_json(docs/'protocol.json',protocol)
    write_json(cfg/'campaign.json',{'maximum_rounds':30,'rounds_completed':0,'rounds':[]})
    return protocol

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');a=p.parse_args();prepare(a.repo)
