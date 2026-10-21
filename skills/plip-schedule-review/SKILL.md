---
name: plip-schedule-review
description: Review a completed explicitly selected FLOWR scheduling cohort, audit all candidate-round decisions, and revise the scheduling skill using trace evidence before an identical-start rerun.
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

Read the actual frozen teacher selection and counts. The current study selects 5nn4 original ranks 1, 2, 3: three teachers, 300 chains and 3,000 decisions per version. Top1/3 share an origin; Top2 has a different origin, and all teachers reuse the same 100 initial slots. The only controls are four amplitudes at inclusive step ranges10..25,26..50,51..75,76..90. Verify exact entry-state restarts and unchanged updates outside an edited interval. The objective is Mean first, Best second; inspect whether population evidence is identical across the five batches of each round and whether weaker chains retain their budget. Distinguish fixed R10 unselected-proposal Mean from per-chain best-of-search Mean, which rises by selection. Retain failed trials and eligible counts. Top2 is intentionally included despite its smaller historical Steer–Unguided gap; it tests that different starting condition rather than serving as an abnormal data point. Development-set selection does not imply cross-target generalization.

In TOP1 storage mode, review every trial through its preserved prompts, numerical node/step analyses, quality XML/JSON and failure records. Only each chain's incumbent trajectory plus the shared immutable R0 are physically retained. Validate retention journals and snapshot checksums; do not claim to inspect deleted raw tensors. Confirm proposals use the retained incumbent as parent and use discarded trials only as evidence.

The user authorized a new run and thread heartbeat on ssh.zzai.scnet.cn:11762. The old 10104 run is excluded; it is accessed only for original-weight retrieval and safe handoff. Poll at two hours initially, adapting within one to six hours; stay quiet on unchanged healthy state. Delete the heartbeat after the full V2 report is delivered.

Before V2, use the strict review-check gate described in operations.zh-CN.md. Each committed shard review must cover every decision with teacher, batch, slot, id, a substantive assessment and actual prompt evidence_ids. Bind all shard, prompt, report and skill hashes to the exact deployed Git commit. An empty approval file or aggregate-only report is insufficient.
