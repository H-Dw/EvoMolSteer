import importlib.util
from pathlib import Path
from agenticgen.adaptive_schedule.io import write, read


def test_report_separates_unselected_search_and_shared_baselines(tmp_path):
    from agenticgen import reporting as module
    module.chemistry=lambda records:{'available':False}
    root=tmp_path/'study';history=tmp_path/'history'
    template={'teacher_id':'t','target':'target','core_atoms':[0],'fingerprints':[['h','A','1','RES','role']],
              'interactions':[{}],'source_record':{'pic50_on_rescore':7}}
    write(root/'config.json',{'rounds':1,'versions':['v1'],'history':str(history)})
    write(root/'teachers.json',[template])
    def records(score):
        return [{'node_id':f'n{i}','pic50_on_rescore':score+i/100,'build_success':True,'connected':True,'failure_reason':'',
                 'quality_evaluated':True,'plip_recovery':.5,'posebusters_pass':True} for i in range(100)]
    write(root/'shared_r0/t/quality_records.json',records(3))
    for arm in ['steer','unguided']:
        write(root/'baselines'/f'target__{arm}.json',records(3.5))
    wave=root/'versions/v1/t/round_01/wave'
    write(wave/'quality_records.json',sum([records(4+g/10) for g in range(4)],[]))
    best=[{'record':r,'source_id':'r01:G4','nodes':{}} for r in records(4.3)]
    write(root/'versions/v1/t/incumbents/r01/best.json',best)
    missing = root/'versions/v1/t/bootstrap/saved_attempts/old/turn_01.attempt_1.invalid.txt'
    missing.parent.mkdir(parents=True, exist_ok=True)
    missing.write_text('{"invalid":true}', encoding='utf-8')
    result=module.analyze(root)
    assert result['populations']==6
    rows=read(root/'reports/populations.json')
    assert {r['collection'] for r in rows}=={'G1','G2','G3','G4','TOP1','group_selected'}
    winner=next(x for x in rows if x['collection']=='group_selected')
    assert winner['source_group']=='G4' and abs(winner['mean_minus_R0']-1.3)<1e-8
    baseline=read(root/'reports/historical_baselines.json')
    assert len(baseline)==2 and all(x['shared_historical_population'] for x in baseline)
    incomplete_costs=read(root/'reports/llm_missing_receipts.json')
    assert len(incomplete_costs)==1 and incomplete_costs[0]['path']==missing.relative_to(root).as_posix()
    assert 'unknown cost' in (root/'reports/report.md').read_text(encoding='utf-8')
