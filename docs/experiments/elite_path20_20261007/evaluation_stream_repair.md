# Local evaluation stream repair

Round 8's affinity rescores and converged MMFF outputs completed, but its
PoseBusters checks did not execute successfully. The retained check table is
empty and all 48 valid candidates have a PB error. The displayed zero pass rate
therefore represents an evaluator failure and cannot be interpreted as molecular
geometry failure.

The first evaluation in a persistent process lets RDKit/PoseBusters bind a global
stream to the redirected round log. That file closes before the next evaluation.
The second evaluation encounters `I/O operation on closed file`, including when
loading the protein. This was diagnosed from the full local log, not inferred
from generation metrics. The registered parent ranking does not use PB rate, so
round 9's already frozen intervention is not selected by the erroneous rate.

The local driver now starts a fresh evaluation interpreter for every round and
blocks report retention/cleanup whenever a PB evaluation error is present. A
real two-process PoseBusters test and an error-blocking regression test pass.
The inference reward, checkpoint, generation seed, window and graph policy are
unchanged. Further adaptive rounds pause while round 9 completes. Round 8 will
be replayed with its exact frozen reward to restore the missing checks; this is
an infrastructure repair of the same intervention, not an additional optimization
round. Both the invalid original report and the repaired provenance will be
preserved and clearly identified.

Resolved: round 8 replay produced exactly identical 50 affinity rescores and 48 converged MMFF relief values (maximum absolute difference zero). Its real PB-fast pass rate is 0.96, with zero PB evaluation errors. Reward JSON values are unchanged. See round08_replay/repair_comparison.json and both retained inference commits.
