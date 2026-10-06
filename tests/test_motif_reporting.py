import pandas as pd
import pytest
from evomolsteer.generation.prototypes import write_json
from evomolsteer.generation.motif_reporting import heldout,report


def fixture(tmp_path):
    for number,batches in ((14,[17]),(15,[18,19])):
        base=tmp_path/f'round_{number:02d}/local';base.mkdir(parents=True)
        rows=[];windows=[]
        for batch in batches:
            for arm in ('unguided','gradient'):
                windows.append({'arm':arm,'batch':batch,'n':2,'symmetric_shape_A':1. if arm=='unguided' else .9})
                for slot in (0,1):
                    failed=arm=='gradient' and batch==17 and slot==1
                    rows.append({'arm':arm,'batch':batch,'slot':slot,'seed':42+100003*batch,
                        'valid_connected':not failed,'pb_fast_pass':not failed,'smiles':f'{arm}_{batch}_{slot}',
                        'energy_status':'failure' if failed else 'converged',
                        'pic50_on_rescore':slot+(0 if arm=='unguided' else .1),
                        'mmff_relief_per_heavy':999 if failed else 1.,'relax_rms_surround_A':.3})
        pd.DataFrame(rows).to_csv(base/'candidate_metrics.csv',index=False)
        write_json(base/'window/report.json',{'batch_results':windows})
    return tmp_path


def test_failure_inclusive_heldout_summary_with_free_graph_changes(tmp_path):
    out=heldout(fixture(tmp_path));g=out['arms']['gradient']
    assert g['attempted']==6 and g['valid']==5 and g['energy_converged']==5
    assert g['all_head']==pytest.approx(.6) and g['MMFF_relief_per_heavy_p90']==1.
    assert out['comparison_vs_matched_native']['shape_improvement']==pytest.approx(.1)
    assert out['paired_batch_changes'][0]['MMFF_p90_change']==0
    assert len(out['paired_batch_changes'])==3


def test_misaligned_slots_or_incomplete_campaign_cannot_be_final(tmp_path):
    root=fixture(tmp_path);path=root/'round_14/local/candidate_metrics.csv';d=pd.read_csv(path)
    d.loc[(d.arm=='gradient')&(d.slot==1),'slot']=2;d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='alignment'):heldout(root)
    with pytest.raises(ValueError,match='fifteen'):report(root,root/'summary',require_complete=True)
