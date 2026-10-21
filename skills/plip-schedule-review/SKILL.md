---
name: plip-schedule-review
description: Review a completed ten-teacher FLOWR scheduling cohort, audit all candidate-round decisions, and revise the scheduling skill using trace evidence before an identical-start rerun.
---

Review only after the V1 cohort has finished or a concrete technical failure needs diagnosis. Scientific skill changes are frozen within a cohort. Technical fixes require local tests, a local Git commit, GitHub push, and deployment of that exact commit; do not edit remote research code.

Read the cohort review manifest and every sharded candidate-round decision record. Shared prompt prefixes may be read once; preserve coverage of unique inputs, actions, outcomes, and cited evidence. Record reviewed shard IDs, counterexamples, and unresolved cases. Fetch complete raw prompts for contradictory, missing-evidence, repeated-action, or failure cases. Do not claim full review from an aggregate report alone.

Check whether agents:

- noticed previous failures and tested a different direction;
- distinguished intermediate head estimates from terminal rescores;
- confused lower coordinate loss with better affinity;
- considered clipping saturation, matching instability, and native-drift competition;
- used supplied evidence and respected the trust region;
- had enough relevant context without redundant tensors and logs.

Revise only rules with observed supporting cases and explicit counterexamples. Keep entries as condition, evidence IDs, suggested experiment, and falsifier. Add or deprecate entries instead of rewriting away failure history. GEPA, Reflexion and ACE motivate this trace-based context evolution; this is not an implementation of their complete algorithms or a model-weight update.

V2 retains all immutable experiment settings and starts from the same R0. Reset each candidate's optimization history. Do not transplant a lookup table of V1 teacher/slot-specific winning schedules. Commit the revised skill and review coverage manifest locally before pushing and remote deployment. Record both the skill hash and runtime hash.

Compare paired per-chain changes, teacher-equal Mean/Best/Top10, recovery/exceedance, chemistry quality, and compute/token budgets. This cohort is development data for the skill; do not infer held-out or cross-target generalization. Complete V2 reporting when both cohorts and the review are finished.

In TOP1 storage mode, review every trial through its preserved prompts, numerical node/step analyses, quality XML/JSON and failure records. Only each chain's incumbent trajectory plus the shared immutable R0 are physically retained. Validate retention journals and snapshot checksums; do not claim to inspect deleted raw tensors. Confirm proposals use the retained incumbent as parent and use discarded trials only as evidence. No task heartbeat is currently configured; do not create one without a new user request.
