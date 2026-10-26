Regression fixture from calibration/full/6pyy_H7S_A_402/p001, node_id
ug_b001_final_004. source.mol is the saved generated SDF molecule;
openbabel_roundtrip.sdf is its fixed-pose PDB/OpenBabel round-trip export.
The latter explicitly encodes the H defining imine E/Z. Default RDKit RemoveHs
keeps that H, although both representations describe the same heavy graph,
formal charges and attached hydrogen counts. Neither fixture is a new sample.

Only a validation copy uses RemoveHsParameters.removeDefiningBondStereo=True;
the molecule analyzed by PLIP, its hydrogen coordinates and source pose remain
unchanged. Official API documentation (accessed 2026-10-26):
https://www.rdkit.org/docs/cppapi/structRDKit_1_1MolOps_1_1RemoveHsParameters.html
