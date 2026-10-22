import copy
import itertools
import numpy as np
import pytest
from agenticgen.adaptive_schedule.selection import choose, adjacent, rollback_advice
from agenticgen.adaptive_schedule.search_policy import policy, validate_policy, validate_probe_quota
from agenticgen.adaptive_schedule.geometry import protein_transform, transform, site_geometry, chemical_pose
from agenticgen.adaptive_schedule.contracts import expanded
from agenticgen.adaptive_schedule.learning import validate_controller_hypotheses


def row(name,c,f=1.,best=8.): return dict(id=name,condition=c,functional_gap=f,best=best)


@pytest.mark.parametrize('d,expected',[(.019,'b'),(.020,'b'),(.021,'a')])
def test_boundary(d,expected):
    assert choose([row('a',1,2),row('b',1+d,1)])['selected']['id']==expected


def test_pool_anchor_order_independence_and_no_chained_drift():
    rows=[row('a',1,3),row('b',1.019,2),row('c',1.038,1)]
    for order in itertools.permutations(rows):
        result=choose(order)
        assert result['selected']['id']=='b' and result['band_ids']==['a','b']
    assert choose(rows,.01)['selected']['id']=='a'
    assert choose(rows,.05)['selected']['id']=='c'
    assert choose([row('a',None)])['status']=='incomplete'


def test_regression_and_rollback_requires_two_complete_adjacent_versions():
    rows=[row('v6',1,1),row('v7',1.019,1.1),row('v8',1.02,1.2)]
    result=rollback_advice(rows)
    assert result['required'] and result['restore_version']=='v6'
    assert not adjacent(row('a',1,1),row('b',.979,3))['regression']
    assert not adjacent(row('a',1,1,9),row('b',1,1,8))['regression']
    rows.append(dict(row('v9',1.019,1.2),rollback_completed=True))
    assert not rollback_advice(rows)['required']
    assert rollback_advice([row('v6',1),row('v8',1)])['status']=='incomplete'
    assert rollback_advice([row('v6',None)])['status']=='incomplete'


def test_amplitude_range_coarse_scale_and_grid_are_independent():
    for v in range(6,11):
        with pytest.raises(ValueError): validate_policy(policy(5),f'v{v}',policy())
    validate_policy(policy(5),'v11',policy())
    validate_policy(policy(15,1.5),'v13',policy(10))
    with pytest.raises(ValueError): validate_policy(policy(10),'v11',policy())
    bad=policy(15);bad['fine_resolution']=1
    with pytest.raises(ValueError):validate_policy(bad,'v13',policy(10))
    assert expanded([5,11,150,7],150)[10]==.5
    assert expanded([5,11,150,7],150)[26]==1.1
    assert expanded([5,11,150,7],150)[51]==15
    with pytest.raises(ValueError):expanded([5,11,5,5])


def test_range_probe_quota_and_local_step_guard():
    def h(name,ticks,mode='local'):
        return dict(id=name,ticks=ticks,reference_ticks=[5]*4,evidence_ids=['e'],
                    search_mode=mode,scale_reason='Probe previously unavailable amplitude with measured delivered movement')
    hs=[h('H1',[45,5,5,5],'range_probe'),h('H2',[5,7,5,5])]
    validate_controller_hypotheses({'hypotheses':hs},[5]*4,2,{'e'},{},[],policy(5))
    with pytest.raises(ValueError):validate_controller_hypotheses({'hypotheses':hs},[5]*4,2,{'e'},{},[],policy())
    hs[1]=h('H2',[5,8,5,5])
    with pytest.raises(ValueError):validate_controller_hypotheses({'hypotheses':hs},[5]*4,2,{'e'},{},[],policy(5))
    hs[1]=h('H2',[5,45,5,5],'range_probe')
    with pytest.raises(ValueError):validate_probe_quota(hs,active=policy(5))
    hs[1]=h('H2',[7,7,7,5])
    with pytest.raises(ValueError):validate_controller_hypotheses({'hypotheses':hs},[5]*4,2,{'e'},{},[],policy(5))


def test_protein_alignment_preserves_complex_pose_but_detects_ligand_only_shift():
    protein=np.array([[0,0,0],[2,0,0],[0,3,0],[0,0,4.]],float)
    ligand=np.array([[.2,.3,.1],[1,1,1.]])
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    translation=np.array([7.,-9,2])
    moved=protein@r+translation
    fit=protein_transform(moved,protein,['A:1:N','A:1:CA','A:1:C','A:1:O'])
    assert fit['determinant']==pytest.approx(1)
    restored=transform(ligand@r+translation,fit)
    assert site_geometry(restored,ligand)['rmsd_A']<1e-12
    assert site_geometry(ligand+[5,0,0],ligand)['rmsd_A']>4
    with pytest.raises(ValueError):protein_transform(protein*[-1,1,1],protein,['A:1:N','A:1:CA','A:1:C','A:1:O'])
    moved[0,0]+=.2
    with pytest.raises(ValueError):protein_transform(moved,protein,['A:1:N','A:1:CA','A:1:C','A:1:O'])


def test_geometry_assignment_renumbering_and_active_mask():
    teacher=[[0,0,0],[2,0,0]]
    result=site_geometry([[2,0,0],[0,0,0],[500,0,0]],teacher,[True,True,False])
    assert result['assignment']==[1,0] and result['rmsd_A']==0
    with pytest.raises(ValueError):site_geometry(teacher,teacher,[True,False])


def test_chemical_mappings_and_two_point_shape_is_unavailable():
    Chem=pytest.importorskip('rdkit.Chem')
    def molecule(smiles,coords):
        m=Chem.MolFromSmiles(smiles); c=Chem.Conformer(m.GetNumAtoms())
        for i,position in enumerate(coords):c.SetAtomPosition(i,position)
        m.AddConformer(c);return m
    teacher=molecule('CCO',[(0,0,0),(1,0,0),(1,1,0)])
    reordered=Chem.RenumberAtoms(teacher,[2,0,1])
    assert chemical_pose(reordered,teacher,[0,1,2])['pose_rmsd_A']==0
    result=chemical_pose(teacher,teacher,[0,1])
    assert result['pose_rmsd_A']==0 and result['shape_rmsd_A'] is None
    other=molecule('CCN',[(0,0,0),(1,0,0),(1,1,0)])
    assert chemical_pose(other,teacher,[0,1,2])['status']=='not_available'
    moved=Chem.Mol(teacher)
    for i in range(3):moved.GetConformer().SetAtomPosition(i,np.asarray(teacher.GetConformer().GetAtomPosition(i))+[3,0,0])
    result=chemical_pose(moved,teacher,[0,1,2])
    assert result['pose_rmsd_A']==pytest.approx(3) and result['shape_rmsd_A']<1e-12


def test_head_prediction_rejected_as_terminal():
    from test_condition_controller import data25
    from agenticgen.adaptive_schedule.objective import condition_summary
    d=data25(); trials=[copy.deepcopy(c['trials'][-1]) for c in d['chains']]
    trials[0]['nodes']['100']['state_kind']='predicted_endpoint'
    with pytest.raises(ValueError,match='actual terminal'):condition_summary(trials,d['teacher'])


def test_sdf_audit_rejects_coordinate_and_covalent_graph_mismatches():
    Chem=pytest.importorskip('rdkit.Chem')
    from agenticgen.adaptive_schedule.geometry import audit_molecule_state
    m=Chem.MolFromSmiles('CO'); c=Chem.Conformer(2)
    c.SetAtomPosition(0,(0,0,0));c.SetAtomPosition(1,(1,0,0));m.AddConformer(c)
    args=([[0,0,0],[1,0,0]],['C','O'],[0,0],[[0,1],[1,0]],{0:'none',1:'single'})
    assert audit_molecule_state(m,*args)==[0,1]
    wrong=copy.deepcopy(args);wrong[3][0][1]=0;wrong[3][1][0]=0
    with pytest.raises(ValueError,match='graph'):audit_molecule_state(m,*wrong)
    m.GetConformer().SetAtomPosition(0,(.1,0,0))
    with pytest.raises(ValueError,match='coordinates'):audit_molecule_state(m,*args)


def test_native_kekule_and_sdf_aromatic_graphs_are_equivalent_but_real_change_is_not():
    Chem=pytest.importorskip('rdkit.Chem')
    from agenticgen.adaptive_schedule.geometry import audit_molecule_state
    mol=Chem.MolFromSmiles('c1ccccc1');conf=Chem.Conformer(6)
    coords=np.array([[np.cos(i*np.pi/3),np.sin(i*np.pi/3),0.] for i in range(6)])
    for i,x in enumerate(coords):conf.SetAtomPosition(i,x)
    mol.AddConformer(conf)
    kekule=Chem.Mol(mol);Chem.Kekulize(kekule,clearAromaticFlags=True)
    bonds=np.zeros((6,6),dtype=int)
    for b in kekule.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();bonds[i,j]=bonds[j,i]=int(b.GetBondTypeAsDouble())
    before=Chem.MolToMolBlock(mol)
    assert audit_molecule_state(mol,coords,['C']*6,[0]*6,bonds,{0:'none',1:'single',2:'double'})==list(range(6))
    assert Chem.MolToMolBlock(mol)==before
    bad=bonds.copy();i,j=np.argwhere(bad==2)[0];bad[i,j]=bad[j,i]=1
    with pytest.raises(ValueError,match='graph'):
        audit_molecule_state(mol,coords,['C']*6,[0]*6,bad,{0:'none',1:'single',2:'double'})
