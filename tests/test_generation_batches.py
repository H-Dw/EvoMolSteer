import pytest
from evomolsteer.generation.batches import resolve_batches


def test_fixed_global_batch_schedule_keeps_master_seed_derivation():
    assert resolve_batches(100,50)==[0,1]
    assert resolve_batches(100,50,'14,15')==[14,15]
    for bad in ['15,14','14,14','-1,0','14','14,15,16']:
        with pytest.raises(ValueError):resolve_batches(100,50,bad)
