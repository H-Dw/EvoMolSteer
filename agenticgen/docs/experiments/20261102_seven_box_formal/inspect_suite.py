"""Common process/lock/source inspection plus source-bound box Selection evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path


def main():
    path=Path(__file__).resolve().parents[1]/'20261031_guidance_comparison5/inspect_suite.py'
    spec=importlib.util.spec_from_file_location('shared_inspection',path)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    output=io.StringIO()
    with contextlib.redirect_stdout(output):common.main()
    result=json.loads(output.getvalue())
    for job in result['jobs']:
        root=Path(job['root']);folder=root/'box_exclusion'/job['teacher']
        job['box_exclusion']=dict(
            search_state=common.read(folder/'SEARCH_STATE.json'),
            negative_evidence=common.read(folder/'NEGATIVE_EVIDENCE.json'),
            frozen_selection=common.read(folder/'SELECTION_FROZEN.json'),
            selection_proposal=common.read(folder/'proposal/decision.json'),
            selection_final=common.read(folder/'final/decision.json'),
            actual_selection_requests=len(list(folder.glob('**/REQUEST.json'))),
            actual_selection_responses=len(list(folder.glob('**/RESPONSE.json'))),
            accepted_selection_receipts=[dict(path=str(p),sha256=common.sha(p),value=common.read(p))
                for p in folder.glob('**/decision.receipt.json')],
            model_budget_claims=[common.read(p) for p in (folder/'model_budget').glob('call_*.json')],
        )
        job['campaign_lock']=common.lock_status(root/'.campaign.lock')
    destination=Path(result['snapshot'])
    destination.write_text(json.dumps({k:v for k,v in result.items() if k!='snapshot'},ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
