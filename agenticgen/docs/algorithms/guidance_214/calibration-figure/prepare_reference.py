"""Prepare structure-grounded reference material for the requested image API."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def save_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition', required=True)
    ap.add_argument('--molecule', required=True)
    ap.add_argument('--readout', required=True)
    ap.add_argument('--outdir', required=True)
    args=ap.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    c=json.loads(Path(args.condition).read_text(encoding='utf-8'))
    mol=next(m for m in Chem.SDMolSupplier(args.molecule,removeHs=False) if m is not None)
    xyz=np.asarray(mol.GetConformer().GetPositions())
    assert mol.GetNumAtoms()==17 and mol.GetNumHeavyAtoms()==17
    assert np.allclose(xyz,np.asarray(c['reference_coordinates_angstrom']),atol=1e-4)
    ids={a['id']:i for i,a in enumerate(c['atoms'])}
    selected=[ids[i] for i in c['controlled_atom_ids']]
    assert len(selected)==9
    jobs=json.loads(gzip.decompress(Path(args.readout).read_bytes()))['jobs']
    j=next(j for j in jobs if j['job']['teacher']=='6pyy_H7S_A_402')
    measurements=[]
    for pop in j['populations']:
        a=pop['aggregate']
        assert a['condition_sha256']==c['condition_sha256']
        if a['id'].startswith('calibration/'):
            measurements.append(dict(name=a['id'].split('/')[-1],
                schedule=a['effective_guidance_scale_schedule'],
                condition_aligned_mean_rmsd_A=a['mean_condition_aligned_rmsd_angstrom']))
    assert len(measurements)==10
    save_json(out/'example_data.json',dict(example='6PYY / H7S',heavy_atoms=17,
        condition_atom_indices_zero_based=selected,condition_atoms=9,
        atoms=c['atoms'],bonds=c['bonds'],coordinates_angstrom=xyz.tolist(),
        condition_sha256=c['condition_sha256'],calibration=measurements,
        source_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
          for name,path in [('condition',args.condition),('molecule',args.molecule),('readout',args.readout)]}))
    export=Chem.Mol(mol)
    for name in export.GetPropNames(includePrivate=True):export.ClearProp(name)
    export.SetProp('_Name','H7S reference molecule')
    with Chem.SDWriter(str(out/'example_molecule.sdf')) as writer:writer.write(export)
    sdf_path=out/'example_molecule.sdf'
    sdf_path.write_bytes(sdf_path.read_text(encoding='utf-8').encode('utf-8'))
    graph=Chem.Mol(mol);rdDepictor.Compute2DCoords(graph)
    teal=(0.13,0.55,0.53)
    for fmt in ('svg','png'):
        drawer=(rdMolDraw2D.MolDraw2DSVG(760,620) if fmt=='svg' else rdMolDraw2D.MolDraw2DCairo(760,620))
        opts=drawer.drawOptions();opts.bondLineWidth=2.4;opts.padding=.12
        opts.addAtomIndices=False;opts.fillHighlights=False;opts.highlightBondWidthMultiplier=9
        drawer.DrawMolecule(graph,highlightAtoms=selected,highlightAtomColors={i:teal for i in selected},
            highlightAtomRadii={i:.33 for i in selected})
        drawer.FinishDrawing();value=drawer.GetDrawingText()
        (out/('atom_graph.'+fmt)).write_bytes(value.encode('utf-8') if isinstance(value,str) else value)
    print(json.dumps({'reference':'molecular_reference.png','heavy_atoms':17,'condition_atoms':9,'measured_calibration_profiles':10}))


if __name__=='__main__':main()
