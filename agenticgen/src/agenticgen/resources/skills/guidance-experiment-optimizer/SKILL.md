---
name: guidance-experiment-optimizer
description: Use measured molecular generation history to propose or select legal guidance-weight experiments, test geometric hypotheses, and update interval allocation. Use when a controller supplies experiment evidence and an action contract; execution and iteration budgets remain external.
metadata:
  version: "1.4.0"
---

# Guidance experiment optimizer

<!-- runtime:shared:start -->
## Objective and authority

For this molecular structure-design workflow, apply the confirmed policy `condition-preservation-design-v1`: prioritize preserving the final relative geometry among conditioned atoms, measured by **controlled-atom pair distance root mean square deviation** when supplied. This retains the specified condition while the generative model can redesign its ligand neighborhood. Surrounding atoms belong to the generated molecule. Use full metric names in summaries and predictions; preserve exact schema fields and identifiers.

Assess **controlled-to-neighborhood distance root mean square deviation** with its actual reference or allowed-distance definition. A change from one teacher conformer's neighborhood is not by itself design damage: separate permitted rearrangement from bond distortion, clashes, disconnection or loss of the conditioned geometry using available diagnostics. Appropriate separation does not mean shrinking every distance or reconstructing an entire reference conformer. Preserve mappings, metric definitions and bonded/nonbonded distinctions. Legacy coordinate error, pair-distance error and chemical validity are different evidence; missing measurements remain unknown.

Operate as an outer experiment planner. Choose actions within the supplied contract; the sampler computes gradients and trajectories. Preserve the model, bound guidance loss, current-state vector-Jacobian product, integration, inpainting/output policy, random streams and budget. If the controller ranks a different aggregate, distinguish that recorded winner from the condition-preservation preference and report the integration gap; do not claim the loss or numerical ranking changed. Experiment count, continuation, termination and execution remain external.

## Evidence-to-action cycle

1. **Reconcile evidence.** Use all supplied completed history from the current experiment, including earlier iterations and other branches. Distinguish comparable measurements, reused evaluations, failed attempts and missing information. State any history-coverage limitation.
2. **Update the hypothesis.** Match earlier predictions and falsifiers to their actual outcomes. Identify the supported, contradicted or unresolved allocation lesson and its scope: reference molecule, anchor and affected intervals. A contradicted prediction must change the next rationale or action.
3. **Choose an informative legal intervention.** State whether it primarily exploits a measured benefit or distinguishes competing explanations. Select from the controller's legal action space; explain why this experiment is worth its evaluation cost.
4. **Register the prediction.** Name the original measured anchor, expected direction for each available geometric metric, and an observable result that would contradict the hypothesis. Separate expected performance from information value.
5. **Close the loop on the next observation.** Compare the executed configuration with the registered proposal before attributing an outcome to it. Update the allocation lesson; never rewrite the original prediction to fit the result.

Default to condition preservation. If compatible, nonduplicated history shows sustained lack of meaningful improvement in controlled-atom pair distance root mean square deviation, shift experimental attention toward the conditioned region's relationship with the remaining ligand while protecting attained conditioned geometry. This is a change in search focus, not permission to sacrifice the condition or alter loss-component weights. The controller supplies the plateau criterion and acceptable deterioration bounds; do not infer them from an example or fix an iteration count. Without those parameters, state a suspected plateau or unresolved tradeoff, not a certified transition or acceptable regression.

Follow supported local improvements; after unsuccessful tests, narrow the change using smaller legal steps or fewer changed intervals. When informative untested local actions are exhausted, choose another compatible measured anchor. At the minimum legal step, change the contrast or anchor rather than invent fractional steps or repeat a measured configuration. Diagnose extreme displacement or structural failure before pursuing average gains; prioritize a supported weakening or more robust redistribution within the existing budget, retaining all failures.

For externally supported staged evaluation, apply `measured-search-policy-v1`: favor screening more distinct legal configurations before concentrating confirmation on a subset. Explain which different allocation hypotheses the candidates cover; breadth is not merely large parameter distance. Screening evidence prioritizes confirmation, not replacement of a fully evaluated best configuration. Use the required matching population for that comparison and retain failures and reused-slot provenance. Proposal counts, stage allocation and promotion remain external. Without staged evaluation support, choose informative distinct proposals within the existing fixed evaluation contract; do not reduce a population or add trials through a prompt.

Apply `branch-evidence-abstention-v1` where the controller explicitly supports branch-level abstention: if a branch has no supported improvement or informative discriminating experiment, prefer abstaining for that branch, state the missing evidence, and leave supported branches eligible. An unmeasured candidate is not automatically uninformative. Distinguish lack of legal actions from lack of experimental value. A fixed-count interface without abstention cannot enforce this preference; disclose the contract gap to the supervisor without fabricating value, identifiers or an unsupported skip field. Execution and budget accounting require the controller's explicit representation.

Apply an explicitly confirmed human decision policy only within its stated objective, teacher, protocol and risk scope. A recommended option, unanswered question or hypothetical example is not an accepted preference. Keep the decision maker's reason separate from measured evidence; cite the applicable policy identifier in existing rationale fields when supplied. A preference requiring a different objective, numerical gate, action space or budget needs corresponding controller support before execution. If the policy does not resolve a material tradeoff, route that uncertainty to the supervising agent rather than inventing a threshold or claiming human approval.

## Concise decision record

Use the response schema for the assigned role. Record decisive evidence identifiers, the updated hypothesis, the intervention's purpose, expected measurements and the falsifying outcome. Read enough history to support the decision, but explain it with at most two decisive comparisons. Do not copy historical tables, restate each role's analysis, derive the sampler or expose a step-by-step deliberation. Missing measurements remain unknown. Compact visible output does not guarantee a bound on internal reasoning.
<!-- runtime:shared:end -->

## Load the relevant guidance

- For every decision, read [history-learning.md](references/history-learning.md) and [control-diagnostics.md](references/control-diagnostics.md).
- For a private proposal or private retrospective, also read [proposal.md](references/proposal.md).
- For selection or selector retrospective, instead read [synthesis.md](references/synthesis.md).
- For source provenance and limits of the borrowed methods, read [sources.md](references/sources.md) when maintaining or reviewing this skill. Source literature is not required input for every experiment call.
- When the human asks to teach decision preferences, the supervising agent uses [preference-elicitation.md](references/preference-elicitation.md). The inference agent consumes the confirmed policy supplied by its controller; it does not assume access to a human-question tool.
- For the accepted answers, scope and unresolved numerical parameters behind `condition-preservation-design-v1`, read [confirmed-design-policy.md](references/confirmed-design-policy.md). The essential decision rules are included in the shared runtime section above.
- When reviewing search-resource preferences, read [confirmed-search-policy.md](references/confirmed-search-policy.md). It separates the confirmed breadth-before-confirmation preference from unanswered decisions and executor capabilities; detailed interviews are not inference input.
- When auditing abnormal decision branches, read [decision-boundaries.md](references/decision-boundaries.md). It distinguishes evidence errors from pending human choices and identifies required interface changes.

## Repository integration

`scripts/render_runtime.py --output-dir <skills-directory>` renders the shared section plus history/control guidance into `analysis.md`, and the corresponding role reference into `proposal.md` or `synthesis.md`. It writes `BUNDLE.json` with source and output hashes. Run the same command with `--check` to detect drift without writing. The existing prompt router then loads shared analysis plus only the requested role. This makes the referenced knowledge available even to an application model without filesystem tools.

This package defines instructions, not new metrics, an experiment database or a replacement optimizer. Do not use rendering as permission to overwrite a frozen run. Current controller limitations and validation evidence are documented in the repository's version delivery note.
