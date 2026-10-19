import copy
from pathlib import Path

import pytest

from evomolsteer.io import read_json,write_json,digest
from evomolsteer.path_replay.common import groups_for,binding,batch_path,read_batch,identity
from evomolsteer.path_replay.local_research import reuse,load

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'configs/experiments/path_replay_b50_20261018/config.json'


def test_research_groups_gather_original_paths_without_regrouping_or_rng():
    cfg=read_json(BASE);c={'n':100,'group_indices':[8,0]}
    groups=groups_for(cfg,c)
    assert groups==[cfg['groups'][8],cfg['groups'][0]]
    assert len({v for g in groups for v in g['ids']})==100
    with pytest.raises(ValueError):groups_for(cfg,dict(c,group_indices=[0,0]))
    with pytest.raises(ValueError):groups_for(cfg,dict(c,n=50))
    with pytest.raises(KeyError):groups_for(cfg,dict(c,group_indices=[0,20]))


def test_reference_reuse_preserves_artifacts_and_checks_scientific_identity(tmp_path):
    cfg=read_json(BASE);original=next(c for c in cfg['conditions'] if c['id']=='ancestor_R02_v3')
    c=dict(original,id='reference',n=50,group_indices=[0],reuse_from=original['id'])
    g=cfg['groups'][0];base=tmp_path/'base';work=tmp_path/'work';old=batch_path(base,original,g)
    old.parent.mkdir(parents=True);artifact=old.parent/'states.npz';artifact.write_bytes(b'exact native B50 result')
    receipt={'binding':binding(cfg,original,g),'status':'complete','records':[{'id':x,'condition':original['id'],'pic50':8.} for x in g['ids']],
             'artifacts':{'states.npz':digest(artifact)}}
    write_json(old,receipt);m={'base_work':str(base)}
    reuse(cfg,m,c,work);result=read_batch(work,cfg,c,g)
    assert result['reused_from']['sha256']==digest(old)
    assert result['records'][0]['condition']=='reference'
    assert read_json(old)==receipt
    reuse(cfg,m,c,work)
    with pytest.raises(ValueError,match='scientific execution'):
        reuse(cfg,m,dict(c,program_sha256='different'),work)
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):reuse(cfg,m,c,work)


def test_manifest_cannot_rewrite_frozen_inputs_or_silently_remove_local_binding(tmp_path):
    cfg=read_json(BASE)
    c={'id':'zero','kind':'unguided','n':50,'window':None,'group_indices':[0]}
    m={'schema_version':'local-feature-path-research-1','batch_size':50,'saved_tensor_policy':'saved_tensors_cpu_exact',
       'base_config':BASE.relative_to(REPO).as_posix(),'base_config_sha256':digest(BASE),
       'base_protocol_sha256':cfg['protocol_sha256'],'base_work':cfg['remote_work'],
       'remote_work':cfg['remote_work']+'_independent','conditions':[c]}
    def save():
        m['research_sha256']=identity({k:v for k,v in m.items() if k!='research_sha256'})
        write_json(tmp_path/'m.json',m)
    save();assert load(REPO,tmp_path/'m.json')[0]==cfg
    m['base_protocol_sha256']='other';save()
    with pytest.raises(ValueError,match='base protocol'):load(REPO,tmp_path/'m.json')
    m['base_protocol_sha256']=cfg['protocol_sha256'];m['conditions'].append(dict(c,id='unbound',kind='native_local',
        **{k:v for k,v in next(c for c in cfg['conditions'] if c['id']=='ancestor_R02_v3').items() if k in ('program','reference','program_sha256','reference_sha256')}));save()
    with pytest.raises(ValueError,match='feature binding'):load(REPO,tmp_path/'m.json')


def test_formation_retention_keeps_half_slots_instead_of_hiding_suffix_changes():
    import numpy as np
    from evomolsteer.path_replay.local_features import measure
    x=np.asarray([[[0.,0,0],[1,0,0],[9,0,0],[10,0,0]]]);y=x.copy()
    y[:,0:2,0]+=10
    mode={'coords_A':x[0].tolist(),'path':{'times':[0.,.5],'coords_A':[x[0].tolist(),x[0].tolist()]}}
    teacher={'modes':{'A':mode}}
    p={'teacher_time':.5,'mode_specs':[{'template_id':'A','atom_slots':[0,1],'pocket_sites':[]}]}
    rows=measure(x,y,np.ones((1,4),bool),teacher,p)
    assert rows[0]['generated_slots_at_half']==[0,1]
    assert rows[0]['half_point_error_A']==0
    assert rows[0]['terminal_tracked_point_error_A']==10
    assert rows[0]['terminal_tracked_pair_error_A']==0
