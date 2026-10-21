---
name: plip-schedule-review
description: Review a completed FLOWR scheduling cohort, cover every private group analysis, synthesis and sampled hypothesis, and revise general skills using trace evidence before an identical-start rerun.
---

Review only after the V1 cohort has finished or a concrete technical failure needs diagnosis. Scientific skill changes are frozen within a cohort. Technical fixes require local tests, a local Git commit, GitHub push, and deployment of that exact commit; do not edit remote research code.

Read the cohort review manifest and every sharded decision record. For the four-plus-one protocol, cover all four private group sessions and the subsequent synthesis separately. Include all tool turns, all eight proposed hypotheses, four ranked selections, and the actual sampled outcomes. Shared prompt prefixes may be read once; preserve coverage of unique inputs, actions, outcomes, and cited evidence. Record reviewed shard IDs, counterexamples, and unresolved cases. Fetch complete raw prompts for contradictory, missing-evidence, repeated-action, or failure cases. Do not claim full review from an aggregate report alone.

Validate privacy before assessing scientific decisions: each analyst's actual inputs and registered tool evidence must belong only to its own 100-candidate group, its inherited group history, and reflections published in earlier rounds. Current sibling data belongs only to the synthesizer. Check that published reflection documents were available to the next round, not leaked back into the same round. The synthesis agent's reflection is evidence for GPT review, never a substitute for that review or permission to launch V2.

Check whether agents:

- noticed previous failures and tested a different direction;
- distinguished intermediate head estimates from terminal rescores;
- confused lower coordinate loss with better affinity;
- considered clipping saturation, matching instability, and native-drift competition;
- used supplied evidence and respected the trust region;
- had enough relevant context without redundant tensors and logs.

Revise only rules with observed supporting cases and explicit counterexamples. Keep entries as condition, evidence IDs, suggested experiment, and falsifier. Add or deprecate entries instead of rewriting away failure history. GEPA, Reflexion and ACE motivate this trace-based context evolution; this is not an implementation of their complete algorithms or a model-weight update.

V2 retains all immutable experiment settings and starts from the same R0. Reset all group/candidate optimization histories and cross-group reflection memory. Do not transplant a lookup table of V1 teacher/slot-specific winning schedules. Commit the revised analyst/synthesizer skills and review coverage manifest locally before pushing and remote deployment. Record both skill hashes and the runtime hash.

Compare paired per-chain changes, teacher-equal Mean/Best/Top10, recovery/exceedance, chemistry quality, and compute/token budgets. This cohort is development data for the skill; do not infer held-out or cross-target generalization. Complete V2 reporting when both cohorts and the review are finished.

Read the actual frozen teacher selection and counts. The current study selects 5nn4 original ranks 1 and 2: two teachers sharing 100 original initial slots. Each teacher runs four hypotheses of 100 candidates per round, for ten rounds: 8,000 optimization outcomes per version, 80 private analyst sessions and 20 synthesis sessions, plus explicitly recorded bootstrap/repair/tool requests. Each private analyst proposes two schedules. The synthesizer ranks eight suggestions and selects four distinct schedules; the final-round suggestions do not authorize an eleventh sampling round. Do not equate HTTP receipts with logical session counts, or reclassify repeated starts as independent samples.

The only controls are four amplitudes at inclusive steps 10..25,26..50,51..75,76..90. Verify checkpoint restart contracts for the frozen physical batch and RNG policy; the user accepts different stochastic finals across changed batch sizes. The objective is unselected 100-candidate Mean first, Best second. Distinguish each hypothesis distribution, the selected group-Mean winner, and per-chain best-of-search output; neither kind of selection alone proves distribution improvement. Inspect weaker-score subgroups, valid counts, common-valid paired changes and failures. Top2 remains intentionally included despite its smaller historical Steer–Unguided gap. Development-set selection does not imply cross-target generalization.

In TOP1 storage mode, review every trial through its preserved prompts, numerical node/step analyses, quality XML/JSON and failure records. Only each chain's incumbent trajectory plus the shared immutable R0 are physically retained. Validate retention journals and snapshot checksums; do not claim to inspect deleted raw tensors. Confirm proposals use the retained incumbent as parent and use discarded trials only as evidence.

The only current server is ssh.zzai.scnet.cn:11762. The old 10104 run is excluded and handoff is complete. Formal sampling and the heartbeat are currently paused by the user; do not resume them during implementation validation. After explicit resumption, inspect at 20 minutes initially and adapt to measured progress. Stay quiet on unchanged healthy state. Delete the heartbeat after the full V2 report is delivered.

Before V2, the review-check gate must support the active protocol, not the legacy per-slot decision count. Each committed review covers teacher, round, group, session role, proposal IDs, a substantive assessment and actual prompt evidence IDs. Reconcile all MASTER_REVIEW_INPUTS indexes with the frozen cohort and sampled-result manifest. Bind all shard, prompt, report, both skill hashes and exact deployed Git commit. An empty approval file, automatic reviewed=true flag, or aggregate-only report is insufficient. The current four-plus-one module emits an evidence index, not a completed strict V2 approval adapter; refuse V2 until that adapter and full real coverage are verified.
