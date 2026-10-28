# Decision precedence and integration boundaries

Use this reference when the supervising agent audits abnormal actions or maintains the decision policy. Keep detailed interviews out of inference prompts. Supporting research and the third interview are recorded in `agenticgen/docs/designs/20261028_guidance_abnormal_branches/`.

## Confirmed branch-level abstention

Policy identifier: `branch-evidence-abstention-v1`. On 2026-10-28 the human answered `branch_abstention`: “允许该方向暂不行动并说明缺什么证据，其他方向正常选择（推荐）”. No further reason or numerical threshold was supplied.

When a branch has no supported improvement and no informative experiment that discriminates relevant explanations, prefer abstention for that branch with a specific missing-evidence explanation. Keep supported branches eligible. A legal but unmeasured candidate can still be valuable: absence of measured success is not by itself a reason to abstain. Distinguish three states: no legal action, legal actions with unresolved value, and a useful action under the supplied evidence and constraints.

Implementation requires an explicit representation in the proposal schema, selection result, sampler dispatch, attempt ledger and expected-completion accounting. The inspected `geometry_council.py` at `f08cc928` requires two proposals and one selected candidate per branch outside final-analysis mode. It cannot express this policy. Do not emit invalid JSON, invent a candidate key or use final-analysis mode to impersonate an abstention. Disclose insufficiency in available narrative fields and identify the missing interface; do not claim the prompt alone stopped execution. No execution, retry or new sampling authorization follows from this policy.

## Evidence interpretation rules

| Potential wrong inference | Required distinction |
| --- | --- |
| The request failed, so its guidance weight is poor | A request/format failure is not a molecular evaluation. Keep partial execution evidence only to the extent actually measured |
| A diagnostic exception means zero error or a failed molecule | Missing evaluation, measured failure and valid poor performance are different statuses |
| The first legal examples are the best or only actions | The supplied examples may be a truncated deterministic enumeration; the full action contract defines the legal space |
| Every in-bound four-window combination is stable | Constant-weight calibration does not measure every mixed schedule; legality and empirical support differ |
| Opposite local effects mean one record should be discarded | Keep the anchor, unchanged intervals and protocol attached to a response; background dependence and interaction remain possible |
| Several roles agree, so the result has been independently verified | Agreement over shared evidence is not a new evaluation or an independent check |
| Repeated improvement on one fixed random panel proves generalization | Separate adaptive search-panel performance from independent validation; reused-seed sensitivity analysis is diagnostic |

Use the existing evidence, uncertainty, prediction and falsifier fields to make these distinctions. Do not add a long checklist or new output fields to each inference. Preserve full metric names, source identifiers and the controller's limits.

## Reconciled decisions and precedence

The six formerly pending choices are now confirmed by the direct human answer recorded in [confirmed-search-policy.md](confirmed-search-policy.md). Its free-text amendment takes precedence over the old option label. No further preference answer is needed for those six branches; numerical definitions and runtime support remain separate issues.

Apply these rules in order:

1. Verify evidence, teacher/protocol/population compatibility and action legality. Preserve failure and reuse provenance.
2. Address measured pathological displacement or structural failure; average gains do not waive existing limits.
3. Apply a verified external recovery trigger before ordinary local continuation; recover the best compatible strategy basis and develop a new allocation using all later lessons.
4. Respect condition preservation and explicit acceptance limits. Allow informative exploratory deterioration without declaring it a new incumbent.
5. Follow useful local improvements; on exhaustion choose a mechanism-complementary measured anchor. Prioritize joint performance without inventing individual causal effects.
6. Prefer mean primary-metric gains within limits; report seed variability without making consistency a separate objective. Keep fixed-panel conclusions scoped.
7. Use breadth-before-confirmation and branch abstention only through supported external contracts; report unsupported operations truthfully.

Compact working context does not discard the ledger or replace full supervising-GPT review. Recovery does not reset the ledger or repeat a saved evaluation. A risk-diagnostic contrast can have value despite uncertain mean benefit; an unmeasured legal action is not automatically useful or useless. Ordinary background-dependent uncertainty does not require a mechanism-isolation experiment before a supported joint improvement.

## Required implementation work, not solved by prose

The inspected internal-distance controller still uses an all-pair aggregate winner and fixed-count branch execution. It does not consume this policy record, implement the new consecutive-regression recovery, or select/retrieve the preferred working context. It lacks the complete seed-level input needed for seed-specific checks. These are adoption gaps, not evidence that prompts enforced the policy. Detailed code locations and manual conflict rehearsals are in `agenticgen/docs/designs/20261028_guidance_policy_reconciliation/`.

Missing numerical definitions remain explicit: round comparison and counter transitions, mean denominator/missing-outcome treatment, protection limits and plateau criteria. Do not convert their absence into a silently assumed threshold, or ask again for the qualitative choices already supplied. Current action schemas and output fields remain exact; unsupported abstention/recovery must be exposed to the supervisor instead of simulated with malformed actions.

## Research boundaries

- [KnowNo, CoRL 2023, section 3](https://proceedings.mlr.press/v229/ren23a/ren23a.pdf) motivates explicit uncertainty and help states. Its calibrated prediction sets and guarantees are not implemented by this abstention preference.
- [MAST, arXiv v3, failure taxonomy](https://arxiv.org/html/2503.13657v3) motivates distinguishing specification, coordination and verification errors. The molecular failure categories above are our adaptation, not the paper's exact labels or a claim of measured reliability improvement.
- [TuRBO, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf) and [NIST's interaction discussion](https://www.itl.nist.gov/div898/handbook/pri/section2/pri212.htm) motivate local scope and possible interaction. Neither proves a specific molecular mechanism from our sparse contrasts.
- [Cakmak et al., NeurIPS 2020, section 2.2](https://papers.nips.cc/paper/2020/file/e8f2779682fd11fa2067beffc27a9192-Paper.pdf) distinguishes mean, tail and worst-case objectives. No risk statistic or tail probability is inferred from a few seeds.
- [Cawley and Talbot, JMLR 2010](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) establishes selection-bias concerns when a noisy finite evaluation criterion is optimized. Applying that concern to fixed molecular random panels is a methodological analogy, not evidence that overfitting has already been measured here.

Sources inspected 2026-10-28, Asia/Shanghai. The inference agent is not assumed to have a human-question tool; the supervisor handles policy questions.
