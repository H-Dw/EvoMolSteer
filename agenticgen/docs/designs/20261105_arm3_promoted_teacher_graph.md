# Follow-up: loss-checked promoted teacher graphs

The endpoint-device fix was verified on the real GPU: all fifty saved engineering
scores equal the experiment-1 scoring entry point, with maximum difference zero.
Attempt 007 then completed two scientific rounds. Its promotion receipts exposed
a second input-adapter defect: PB-passing candidates containing aromatic `[nH]`
could not be reconstructed from element, charge and aromatic bonds alone.
PLIP succeeded; the next geometry/chemistry condition failed during sanitization.

The first round contained two such candidates, with frozen endpoint pIC50 values
5.874032497406006 and 5.870794773101807 (baseline 5.81679630279541). These are
predicted scores, not measured affinity. They were not promoted by the old code.

`teacher_preparation` serializes single/double Kekule valences from a copy of the
candidate before reconstruction. It verifies the complete graph, per-atom hydrogen
counts, charges, radicals, isotopes, atom order and unchanged 3D coordinates.
It also checks that the shared condition encoder produces exactly the channels
obtained directly from the original native molecule under the model's existing
aromaticity setting. Unsupported round trips fail explicitly.

Only promoted-teacher preparation changes. The original teacher, shared geometry
and chemical inpainting modules, controlled region, step boundaries, endpoint
definition, PB requirements, pIC50 ranking and frozen promotion scores remain
unchanged. Sanitization/aromaticity handling does not optimize coordinates or
change protonation. Errors now identify whether PB or teacher preparation failed.

Attempt 007 was explicitly stopped after 150 scientific reservations; all data
and claims are preserved, including the interrupted third round. No reservation
is refunded. The campaign has therefore used/reserved 6,400 scientific candidates.
An independent new five-round run would need 250 further candidates (campaign
total 6,650; experiment-3/6pyy cumulative 400). This exceeds the original 250
ceiling for this teacher and must be explicitly approved before another run.

A prospective independent run can use scientific seed-block offset 7, providing
five disjoint new seed groups instead of repeating the already consumed groups.
The engineering block remains six and can reuse the validated fifty saved
engineering trajectories without generating additional engineering candidates.
The offset is frozen in configuration and every input/claim receipt.

`validate_arm3_promoted_teacher.py` checks saved real candidates through PB, PLIP,
condition construction, native feature encoding, fixed pocket/frame validation,
inpainting at steps 0/1/98/99 and full-teacher endpoint scoring. It also rescores
all fifty original terminal rows and checks source hashes. It never calls the
sampler or reserves scientific budget. A clean five-round scientific rerun is
separate from this zero-sampling validation.
