---
name: guidance-postexperiment-rsi
description: Use an external agent to review a fully completed multi-target experiment and create a separate, immutable skill revision from one evidence-backed process problem. Not for within-experiment action selection or skill edits.
metadata:
  version: "1.0.0"
---

# Post-experiment skill improvement

## Entry gate and authority

Operate as the external review agent, separate from the Guidance Agent. Begin RSI only after all targets and all required work in the externally declared experiment scope have completed. A completed target, round or intermediate version is insufficient. Verify the original target manifest and authoritative completion/failure/quality/retention receipts using [completion-and-evidence.md](references/completion-and-evidence.md). Do not shorten the target list, treat a paused/aborted target as complete, or run missing experiments to pass this gate. If incomplete, record an incomplete-scope result and end this RSI invocation without making a skill revision.

Automated RSI is non-interactive: do not contact the human, ask questions, request approval or wait for a reply. Use confirmed policy and supported machine access. Missing critical evidence or authorization prevents dependent work; record the gap. Experiment execution, deployment and additional model evaluations require their own existing authorization and are not implied by this skill.

## One main problem, separate evidence layers

Read the completed experiment's compatible histories, original predictions, chosen and unchosen proposals, execution/quality receipts, retained structures, failures and reuse lineage. Separate valid measured outcomes from output/format/execution errors. Compare within target and matched populations; duplicated configurations and nested slots are not independent validation.

Choose one principal, evidence-backed process problem. Locate it in evidence availability/retrieval, interpretation, legal proposal construction, selection, or evaluation handling before blaming the prompt. A missing input or controller feature is not fixed by writing a reminder. A molecular outcome alone does not prove a particular skill caused it.

When a supported instruction change exists, change only the necessary one to three Guidance Agent modules around that one problem, one expected observable behavior and one falsifier. This is not a quota per experiment or a command to modify skills without a supported problem. If no actionable issue is established, record no revision. Keep observations, edits and evaluation results distinct. Use [revision-and-adoption.md](references/revision-and-adoption.md) for the evidence record and acceptance boundary.

## Abstraction without target leakage

Fixed skills teach how to analyze, retrieve and choose. Current-task allocation experiences remain in the task experience store; repeated support never turns a target's best interval or weight into a reusable fixed rule. Do not transfer a target identifier, unique measured value, literal schedule, molecule-specific example or source path into reusable skill text.

External RSI may derive general diagnostic methods and abstract, conditional hypotheses, such as whether large absolute increments indicate a need to inspect intervention timing. It must not disguise a local preferred weight as a universal rule. Put any empirical allocation-transfer hypothesis in a separate new-task prior context, not in fixed skills or compatible measured history. Validate using the receiving target's own calibration, measured anchors and legal actions in a later authorized experiment. Direct receiving-target counterevidence takes priority.

Keep exact experience sources and target-specific measurements only in the version's update log/evidence artifacts, excluded from runtime loading. Use abstract characteristics in reusable instructions, with no invented feature measurements or universal thresholds. See [abstraction-and-leakage.md](references/abstraction-and-leakage.md).

## Create a new version and assess it

Publish each revision in a previously nonexistent version directory with its own standard SKILL.md files, parent version/hash, changed-module record and manifest. Never overwrite source skills, a previous RSI skill, release files, experiment snapshots or accepted review receipts. Never hot-swap a running experiment. If revising this external RSI skill itself, use the same completed-experiment gate, single-problem evidence chain and new-directory rule; that revision does not authorize an unbounded self-improvement loop.

Check standard frontmatter, referenced resources, exact output schemas, role isolation, target leakage and rule conflicts. Replay only pre-decision inputs; later outcomes are review labels. Distinguish static checks, actual model behavior and actual molecular evaluations. Do not claim a hypothetical action's measured quality or ask for hidden reasoning.

A demonstrated process improvement may be adopted for a future authorized experiment without demonstrated geometry gain, but record it as process improvement only. It does not replace the best molecular-configuration checkpoint, alter the primary objective, claim geometry benefit, reset regression history or bypass recovery. A newly written and format-checked skill is only a proposed version until the relevant process behavior is actually checked. Record adoption separately from release creation and bind future runs explicitly.

## Preserve project decisions

Prioritize condition coordinate root mean square deviation; assess maximum condition atom coordinate error under explicit protection limits while permitting valid ligand redesign. Use full metric names. Preserve scientific compatibility, teacher-specific calibration, exact action identity, existing output modules, the fixed-panel claim boundary, budgets, failures and source artifacts. No empirical cross-target averaging of guidance strengths. Iteration limits and recovery counters remain external.

Resolve apparent conflicts explicitly: task-memory updates are not skill edits; background matching precedes recency; recency does not beat stronger contrary measurements; configuration recovery does not replace frozen skills; process acceptance is not molecular-best selection; abstract transfer is not copied experience or measured evidence. Do not restore obsolete policies through a checkpoint.

## Protein-anchored absolute-coordinate contract

The fixed protein atoms alone determine one proper Kabsch measurement transform. Apply it to the predicted complex for loss and evaluation, never to the evolving sampling state. The ligand cannot influence the alignment. At the first active step, the executor selects one injective geometric assignment from the uncovered clean prediction using Hungarian matching, then freezes that candidate's assignment for the whole trajectory. Coordinate guidance and chemical state inpainting share it. Never rematch the terminal structure to improve the ranking, align the ligand alone, alter the protein anchors or infer atom identity from a small geometry error.

The single coordinate loss is one half of the mean squared condition-atom coordinate error, not a sum over atom-pair distances. The executor owns alignment, assignment, teacher-specific calibrated multiplier, complete schedules and the full coordinate vector-Jacobian product. There are no condition/outer/interaction component switches. Chemical channels and induced condition bonds are injected only into the state; coordinates, self-conditioning and the raw999 final prediction are never overwritten. Evaluate final chemical retention separately.

Learn allocation from compatible historical configurations, measured coordinate errors, maximum and unconstrained increments, limiter counts, original predictions and counterexamples. Different alignment, assignment or loss protocols are not comparable measured anchors. Iteration counts remain external. Native tool calls remain intermediate; final responses keep the supplied role JSON, full receipt identities and text length limits.
