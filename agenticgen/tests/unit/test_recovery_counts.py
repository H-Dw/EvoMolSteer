import pytest
from agenticgen.adaptive_schedule.io import write,sha
from agenticgen.runtime.recover_plip_council_analysis import validate_wave_quality


@pytest.mark.parametrize('population',[25,100])
def test_recovery_requires_frozen_population_not_old_four_hundred(tmp_path,population):
    config={'population_size':population,'hypotheses':4}
    records=[{'attempt':i} for i in range(4*population)]
    write(tmp_path/'quality_records.json',records)
    write(tmp_path/'final_records.json',records)
    write(tmp_path/'QUALITY_COMPLETE.json',{'attempts':len(records),'records_sha256':sha(tmp_path/'quality_records.json')})
    assert validate_wave_quality(config,tmp_path)==len(records)
    write(tmp_path/'final_records.json',records[:-1])
    with pytest.raises(ValueError,match='count changed'):validate_wave_quality(config,tmp_path)
