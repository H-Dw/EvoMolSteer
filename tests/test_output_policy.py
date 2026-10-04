"""Storage changes must not change selection statistics or effective scope."""
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evomolsteer.io import digest,read_json,write_json
from evomolsteer.output_policy import extraction_representations,write_batch_table
from evomolsteer.scope import SCOPE
from test_selection_scope import test_four_methods_share_event_scope_and_same_time_background as make_analysis


def test_compact_preserves_all_four_methods_and_computed_rates(tmp_path):
    audit=tmp_path/'audit';audit.mkdir()
    make_analysis(audit)
    compact=tmp_path/'compact';compact.mkdir()
    for name in ['config.json','features.parquet','edges.parquet','feature_catalog.json','selection_scope.json']:
        shutil.copyfile(audit/name,compact/name)
    cfg=read_json(compact/'config.json');cfg['output_policy']={'profile':'compact'}
    write_json(compact/'config.json',cfg)
    from evomolsteer import enrichment,differential,trends,pca
    for module in [enrichment,differential,trends,pca]:module.run(compact)
    for method in ['enrichment','differential','trends','pca']:
        a,c=audit/'discovery'/method,compact/'discovery'/method
        for name in ['effects.csv','effect_rates.csv','event_diagnostics.json']:
            assert digest(a/name)==digest(c/name)
        for name in ['batch_effects','batch_effect_rates']:
            x,y=pd.read_csv(a/(name+'.csv')),pd.read_parquet(c/(name+'.parquet'))
            pd.testing.assert_frame_equal(x,y,check_dtype=False,atol=1e-10,rtol=1e-10)
        assert not (c/'events.parquet').exists()
        assert not (c/'event_rates.parquet').exists()
    for method in ['trends','pca']:
        assert digest(audit/'discovery'/method/'parent_child_diagnostics.json')==digest(compact/'discovery'/method/'parent_child_diagnostics.json')
    for name in ['basis.json','loadings.csv','explained_variance.csv']:
        assert digest(audit/'discovery/pca'/name)==digest(compact/'discovery/pca'/name)
    for name in ['pca/scores.parquet','pca/parent_child_velocity.parquet','trends/parent_child_rates.parquet']:
        assert not (compact/'discovery'/name).exists()


def test_float64_batch_table_is_uncompressed_and_bit_exact(tmp_path):
    values=np.array([1+2**-50,-0.,np.nextafter(1.,2.),1e-99],np.float64)
    table=pd.DataFrame({'feature':['x']*len(values),'value':values})
    write_batch_table(tmp_path/'batch.csv',table,{'output_policy':{'profile':'compact'}})
    with pq.ParquetFile(tmp_path/'batch.parquet') as p:
        assert p.schema_arrow.field('value').type==pa.float64()
        assert all(p.metadata.row_group(0).column(i).compression=='UNCOMPRESSED' for i in range(len(p.schema_arrow)))
        assert p.read().column('value').to_numpy().tobytes()==values.tobytes()


def test_storage_profile_never_adds_representations_or_nodes():
    cfg={'analysis_scope':SCOPE,'analysis_representations':['proposal_state']}
    for name in ['audit','compact']:
        cfg['output_policy']={'profile':name}
        assert extraction_representations(cfg)==['proposal_state']
    cfg['analysis_representations']=['current_state']
    with pytest.raises(ValueError):extraction_representations(cfg)


def test_compact_refuses_to_mix_in_stale_details(tmp_path):
    from evomolsteer.output_policy import validate_output_layout
    path=tmp_path/'events.parquet';path.write_bytes(b'historical')
    with pytest.raises(ValueError):validate_output_layout(tmp_path,{'output_policy':{'profile':'compact'}})
    assert path.read_bytes()==b'historical'
