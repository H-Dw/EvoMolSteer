# Frozen confirmation after round 17

No inference parameters are optimized during rounds 18–20. The source program
and reference hashes are frozen before the first independent panel. Round 18
uses the selected program and native control on batch IDs 32–33, round 19 uses
the old incumbent on the same panel, and round 20 repeats selected/native on
batch IDs 34–35. Each arm contributes 100 attempts per panel. All real initial
coordinate, atom and bond tensor signatures must agree within each paired batch.
Master seed 42 produces batch seeds by the existing rule `42 + batch*100003`;
different batch IDs provide separate deterministic initial-state panels.

The registered screening rank requires mean head score at least incumbent minus
0.02 and molecular validity at least 0.90, then ranks distinct elite graphs,
mean, p95 and secondary strain. This rule selects round 5, not the tail-leading
but lower-mean round 14. No claim of independently confirmed new path-kernel
performance follows from confirmation of the different endpoint-attractor family.

The final synthesis reads checksum-retained terminal records, reports all-output
and valid-only head scores, counts elites with all attempts as denominator, and
adds a separate PB-fast-qualified elite count. Deduplication is a reporting
operation, never an inference graph constraint. Energy remains a continuous
secondary MMFF local-relaxation diagnostic; no post hoc energy cutoff is added.
Two-batch panel uncertainty is limited, so the selected/native comparison also
pools all four frozen batches. No tuning follows confirmation outcomes.

The library builder's future metadata now names both supported consuming reward
families. Its description no longer assumes the consuming formula is an endpoint
attractor. Existing round 3/14 reference files and hashes remain frozen; for
those historical runs the actual `reward_program.json` identifies the executable
formula. Geometry-kernel bandwidth/curvature in round 16 apply to standardized
feature distance, whereas point-cloud distances use Angstrom units. Inherited
parameter key names do not make those standardized quantities physical distances.

After round 20 reports are retained and remotely verified, the finalizer removes
only disposable generated data and known campaign archives. Original Steer data,
checkpoints, source, reports and execution logs are protected. A terminal-only
transport capsule cannot recover intermediate coordinates after raw retirement;
new intermediate analysis requires deterministic regeneration from frozen inputs.
