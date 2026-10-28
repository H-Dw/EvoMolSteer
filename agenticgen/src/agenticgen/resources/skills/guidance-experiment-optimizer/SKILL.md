---
name: guidance-experiment-optimizer
description: Use measured molecular generation history to propose or select legal guidance-weight experiments, test geometric hypotheses, and update interval allocation. Use when a controller supplies experiment evidence and an action contract; execution and iteration budgets remain external.
metadata:
  version: "1.1.0"
---

# Guidance experiment optimizer

<!-- runtime:shared:start -->
## Objective and authority

Improve **controlled-atom coordinate root mean square deviation** and **controlled-to-surrounding-atom distance deviation** through evidence-based allocation of guidance weights across the supplied sampling intervals. Surrounding atoms belong to the generated molecule; exclude protein, receptor, pocket and other environment atoms. Use full metric names in reasoning summaries and predictions. Preserve exact machine field names and identifiers required by the response schema.

The first metric compares matched controlled atoms with their reference coordinates. The second compares supplied atom-pair distances with reference distances or allowed intervals. Preserve the supplied atom mapping, neighborhood, bonded/nonbonded distinction, tolerances and aggregation. Appropriate distances do not mean making all distances smaller. When the second metric or its definition is missing, mark it unavailable; coordinate agreement does not substitute for local-distance evidence.

Operate as an outer experiment planner. Choose weight schedules; the unchanged sampler computes gradients and trajectories. Preserve the model, coordinate loss, current-state vector-Jacobian product, integration, inpainting/output policy, random streams and budget. Do not change the scientific objective or turn diagnostic observations into an invented controller. Experiment count, continuation, termination and execution are external parameters. Learning from history grants no authority to launch, repeat or extend experiments.

## Evidence-to-action cycle

1. **Reconcile evidence.** Use all supplied completed history from the current experiment, including earlier iterations and other branches. Distinguish comparable measurements, reused evaluations, failed attempts and missing information. State any history-coverage limitation.
2. **Update the hypothesis.** Match earlier predictions and falsifiers to their actual outcomes. Identify the supported, contradicted or unresolved allocation lesson and its scope: reference molecule, anchor and affected intervals. A contradicted prediction must change the next rationale or action.
3. **Choose an informative legal intervention.** State whether it primarily exploits a measured benefit or distinguishes competing explanations. Select from the controller's legal action space; explain why this experiment is worth its evaluation cost.
4. **Register the prediction.** Name the original measured anchor, expected direction for each available geometric metric, and an observable result that would contradict the hypothesis. Separate expected performance from information value.
5. **Close the loop on the next observation.** Compare the executed configuration with the registered proposal before attributing an outcome to it. Update the allocation lesson; never rewrite the original prediction to fit the result.

Distinguish a coordinate improvement from an acceptable geometric tradeoff. Apply only externally supplied tolerances and selection priorities; do not invent a weighted score, universal distance threshold or stability guarantee. Prefer non-dominated measured alternatives when both metrics are available. If alternatives trade off and the controller supplies no preference, expose the tradeoff rather than claiming a unique geometric optimum.

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

## Repository integration

`scripts/render_runtime.py --output-dir <skills-directory>` renders the shared section plus history/control guidance into `analysis.md`, and the corresponding role reference into `proposal.md` or `synthesis.md`. It writes `BUNDLE.json` with source and output hashes. Run the same command with `--check` to detect drift without writing. The existing prompt router then loads shared analysis plus only the requested role. This makes the referenced knowledge available even to an application model without filesystem tools.

This package defines instructions, not new metrics, an experiment database or a replacement optimizer. Do not use rendering as permission to overwrite a frozen run. Current controller limitations and validation evidence are documented in the repository's version delivery note.
