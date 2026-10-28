# Elicit and apply scoped decision preferences

This procedure is for the supervising agent when the human requests preference teaching or an uncovered tradeoff materially affects a decision. It does not add interactive tool calls to each model inference or authorize experimental execution. Reuse already confirmed preferences within scope; do not repeatedly ask the same question.

## Research before asking

Identify the actual branch using compatible measured records. Separate scientific feasibility, which the controller checks, from a value judgment about acceptable tradeoffs. Read primary research or official implementation guidance for similar decisions. Offer three or four distinct feasible strategies, explain their implications, and identify any recommendation as the supervisor's recommendation. Do not offer fabricated data, forbidden resampling, or violated constraints as ordinary alternatives.

Useful source families are outcome-comparison preference learning, constrained multi-objective optimization, local experimental search, and evaluation of tail risk. Translate their decision principles, not their unimplemented numerical guarantees. If exact measurements for the current metric do not exist, label numerical examples hypothetical and specify which protocol they illustrate.

## Obtain the reason and boundary

Present one consequential tradeoff at a time, or bundle a few independent short questions. Include full metric names, units, relevant uncertainty, and what is kept fixed. Ask for a choice and its reason: which deterioration is unacceptable, what evidence would reverse the preference, and whether the rule is shared across reference molecules or requires teacher-specific calibration.

Use a boundary or counterexample question when a chosen strategy needs an unspecified tolerance, weighting, or stopping condition. Do not infer a numerical threshold from the values in an example. An answer about one protocol does not silently change another protocol's objective. Distinguish a dislike of excessive tail error from permission to add a new loss, acquire new random samples, or pause unrelated work.

If the human does not answer, leave the choice pending. A preselected recommendation is not confirmation. A delayed answer applies to the identified question and evidence snapshot; check whether that branch still exists before acting on it.

## Convert the answer into an auditable rule

Keep a decision record with: question identity; evidence snapshot; simulated-versus-measured status; offered options and sources; the human's actual answer and supplied reason; applicable objective/protocol/teacher scope; condition for changing the decision; numeric parameters explicitly supplied; and implementation status. Separate the direct answer from the supervisor's interpretation. Do not fabricate a reason if the human supplies only a selection.

Translate only confirmed preferences into a short conditional rule: **when the supported condition holds, prefer the specified legal choice, because of the stated priority; reconsider when the specified evidence appears**. Keep unknown thresholds explicit. Validate that the rule respects frozen scientific contracts and has the required measured inputs. Mark a rule as requiring controller integration if it changes ranking, feasibility, sampling allocation or execution. Store the evidence and interview outside the concise inference instructions; the controller should supply the relevant confirmed policy and identity.

## Check understanding through a scenario

Replay the rule on a clearly marked hypothetical or historical comparison. Explain the observable decision and its tradeoff, not a hidden chain of thought. Include a counterexample in which the rule should not apply. If two answers conflict, surface the exact conflict instead of averaging or silently choosing one. New explicit human instructions supersede earlier preferences within their stated scope.

## Primary references

- [Lin et al., Preference Exploration for Efficient Bayesian Optimization with Multiple Outcomes, AISTATS 2022](https://proceedings.mlr.press/v151/jerry-lin22a.html): pairwise outcome preferences guide subsequent expensive optimization. This interview does not fit their utility model.
- [BoTorch v0.16.0, Objectives](https://botorch.org/docs/v0.16.0/objectives): scalarization and constrained objectives are different design choices. They require actual definitions and implementations, not just narrative labels.
- [Eriksson et al., Scalable Global Optimization via Local Bayesian Optimization, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html): evidence-dependent local search regions. Borrowing a strategy preference does not instantiate its surrogate or trust-region algorithm.
- [Cakmak et al., Bayesian Optimization of Risk Measures, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/e8f2779682fd11fa2067beffc27a9192-Abstract.html): distribution-tail objectives differ from mean performance. A finite small sample is not a guarantee about population tails.

Sources checked 2026-10-28, Asia/Shanghai. No strategy listed here is a human-confirmed default merely because it is included in the interview.
