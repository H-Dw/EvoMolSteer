import numpy as np
import pytest
from evomolsteer.elite_alignment.analysis import distribution,mean_ci,holm,nearest_teacher


def test_distribution_keeps_invalid_scores_but_separates_best_and_denominator():
    scores=[7.,8.,9.,np.nan]
    out=distribution(scores,[True,True,False,True],[True,False,False,False],5,8.,8.7)
    assert out['n_scored']==3 and out['score_coverage']==.6
    assert out['mean']==8. and out['best_all']==9. and out['best_valid']==8.
    assert out['q95']==pytest.approx(8.9)
    assert out['tail_count']==1 and out['tail_rate']==.2
    assert out['above_steer_count']==0


def test_root_bootstrap_retains_clone_dependence():
    roots=['a']*100+['b'];delta=[1.]*100+[-1.]
    lo,hi,p=mean_ci(roots,delta,repeats=2000)
    assert lo==-1. and hi==1. and p>0.1
    assert mean_ci(roots,delta,repeats=2000)==(lo,hi,p)


def test_holm_reserves_the_full_planned_family():
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    assert np.allclose(holm([.001,.02],family_size=93),[.093,1.])


def test_teacher_distance_is_permutation_invariant_without_ligand_repositioning():
    x=np.array([[0.,0.,0.],[3.,0.,0.],[0.,3.,0.]])
    teachers=x[[2,0,1]][None]
    assert nearest_teacher(x,[1,1,1],teachers)==0.
    assert nearest_teacher(x+[0.,0.,1.],[1,1,1],teachers)==pytest.approx(1.)


def test_residue_contact_proxy_keeps_the_receptor_frame(tmp_path):
    import gzip,json
    from rdkit import Chem
    from evomolsteer.elite_alignment.contacts import run
    pdb=tmp_path/'protein.pdb'
    pdb.write_text('ATOM      1  N   VAL A 116       0.000   0.000   0.000  1.00 20.00           N  \n')
    for method,z in [('unguided',0.),('M2',2.)]:
        mol=Chem.MolFromSmiles('CO');conf=Chem.Conformer(2);conf.SetAtomPosition(0,(3.,0.,z));conf.SetAtomPosition(1,(3.,1.,z));mol.AddConformer(conf)
        folder=tmp_path/'methods'/method/'case';folder.mkdir(parents=True)
        with gzip.open(folder/'terminal.sdf.gz','wt') as f:f.write(Chem.MolToMolBlock(mol)+'\n$$$$\n')
        (folder/'result.json').write_text(json.dumps({'pic50':7.}))
    residues,summary=run(tmp_path,pdb,['case'],['unguided','M2'],tmp_path/'output')
    rows=residues.set_index('method')
    assert rows.loc['unguided','min_polar_A']==pytest.approx(np.sqrt(10))
    assert rows.loc['M2','min_polar_A']==pytest.approx(np.sqrt(14))
    assert rows.loc['unguided','closest_protein_polar_atom']=='N'
    assert rows.loc['unguided','polar_pairs_lt3p5']==1
    assert rows.loc['M2','polar_pairs_lt3p5']==0
