# History persistence and use contract

Read this reference when the supervising agent maintains Guidance Agent memory or integrates a controller. It defines responsibilities and proposed records; it is not an installed storage engine, a new response schema or authorization to run experiments. Essential inference rules are rendered from history-learning.md. Research provenance is in history-research.md.

## Four layers and their owners

| Layer | Preserve | Owner and mutation rule |
| --- | --- | --- |
| Source evidence | Received request/response/receipt, original proposal/prediction/falsifier, resolved action, configuration/skill/source hashes, slots and attempts, measurements/diagnostics, structures, terminal states and trajectories, failures and reuse provenance | Executor writes truthful artifacts at lifecycle boundaries; never overwrite a frozen run. Add correction/supersession receipts if an interpretation was wrong |
| Experiment index and lineage | Canonical configuration and scientific compatibility, parent action and parent skill version, proposal/attempt/evaluation identifiers, selected/unselected state, population and source links, completion/failure/reuse status | Controller maintains or rebuilds from source evidence. One physical evaluation can have several proposal/reuse links; preserve both without double-counting |
| Lesson archive | A short scoped observation and interpretation, source references, contrary evidence, applicability and revision links | Agent proposes; supervisor checks; supported persistence interface records with an acknowledgment. Do not infer successful storage from generated text |
| Working context | Coverage overview, current task/anchor, decisive compatible contrasts, counterexamples/risk records, relevant lessons and source access | Controller selects under an external size budget. Selection never deletes source evidence or relaxes full-version review |

Do not create a second competing truth store. Reuse existing manifests, registries, history snapshots, receipts and review paths; add derived indexes/lessons only through an explicitly integrated interface. Descriptive storage-layer names here do not imply files already exist.

## Save at meaningful lifecycle boundaries

1. Before evaluation, persist the proposed action, anchor, intended prediction/falsifier, prompt/skill binding and selection status. Record an unselected candidate as unexecuted, not as a failed experiment. Do not retroactively invent missing proposals or predictions.
2. Bind validation, attempt reservation and actual resolved execution to that proposal. Record request or action-validation errors separately from sampling and evaluation failures. An interrupted attempt remains incomplete with any actual partial evidence retained.
3. Link completed measurements to their exact population, seed/slot identities, metrics/denominators and raw output sources. Preserve failure counts, finite extremes and unknown diagnostics. Quality completion may arrive after base metrics; add its receipt without relabeling the original snapshot.
4. On reuse, save a reference to the original measurement and make the non-independent/new-attempt status explicit. Equivalent proposals retain their own provenance but do not get copied scores counted as fresh validation.
5. On review, bind the accepted/rejected lesson revision and actual skill changes to their evidence and predecessor. Retain rejected interpretations as review history where recorded; they are not active instruction authority.

The controller must make write/commit success observable and reconcile partial writes from receipts. This skill adds no filesystem permission, automatic retry or recovery loop. Frozen manifests, source structures and claims remain protected.

## A compact lesson record

These are proposed archive fields for the supervisor/persistence layer, not new Guidance Agent output keys. Reuse existing equivalent metadata when available; absent values are unknown, never fabricated.

| Field group | Required meaning |
| --- | --- |
| Identity and revision | Stable lesson identity, revision and predecessor/supersession link, author role, time and originating skill version |
| Applicability | Teacher/condition, protocol and metric definitions, output policy, calibration/action semantics, population scope, anchor or background for a local response |
| Source evidence | Actual observation/proposal/receipt identifiers and available content hashes; links to original prediction/falsifier and contrary observations |
| Observation | Full metric names, reported denominator and measured direction; actual diagnostic/failure status |
| Interpretation | Tentative or locally supported explanation, why it is useful, what remains unknown; separate from observed facts |
| Review status | Tentative, locally supported, disputed/contradicted, narrowed or superseded; status justified by evidence, not a vote or invented confidence score |
| Next-decision implication | Applicable legal allocation direction, known exception, and an observable event that would revise the lesson |

Maintain separate meanings for a measured result's status and a lesson's review status. Reused slots and multiple textual reflections are not independent support counts. There is no universal number of confirming rounds. Do not overwrite a source to fit a lesson, discard an inconvenient counterexample, or make a plausible interpretation an executable action identity.

## Retrieval and consolidation

Filter hard scientific incompatibilities before ranking relevance. Search the global configuration registry for duplicates independently of the small prompt view. Retrieve the relevant measured baseline/anchor, compatible contrasts on affected windows, counterexamples and failure signatures, and useful alternative lineages. Favor sources that can change the current decision rather than only top scores, the newest text or the most semantically similar narrative. The exact number and size are external budgets.

Consolidate duplicate narrative claims into one lesson with all source links; do not merge unequal populations or incompatible backgrounds into one numerical measurement. Keep raw values, precision, action identifiers, types and failure status intact. Distinguish global coverage from supplied detail. Record why a source was included and what is omitted; allow direct source inspection when a summary conflicts with a receipt. Without retrieval support, the model uses supplied evidence and explicitly marks missing information.

Conflicting local effects should be stored with their backgrounds. Narrow a lesson when new evidence contradicts a broader claim; do not force causal decomposition before a supported joint improvement. Retain non-best but informative configurations as possible next anchors. This implements the user's interest in complementary hypotheses without creating a new exploration quota or changing the incumbent objective.

## Two improvement loops, two archives

The experiment loop updates evidence about guidance-weight configurations. The supervising-GPT loop updates the Guidance Agent's skills from evaluated behavior. Bind both lineages, but do not confuse them: a better molecule is not proof that a new prompt caused the gain when the sampled action changed; a better narrative is not a measured scientific improvement.

For a proposed skill change, record the observed error/limitation, parent skill version, decisive evidence and counterevidence, concrete changed instruction, expected observable behavior and falsifier. Review archived cases for compatibility, legal actions, duplicate avoidance, failure classification and policy conflicts. Prevent hindsight leakage: reconstruct the pre-decision snapshot for inputs, and label later observations as evaluation evidence. Offline review cannot tell whether a different unexecuted schedule would have produced a better molecule. No extra model calls, sampling or unseen validation panel follows from this reference.

Recovery uses the best compatible strategy basis under the latest human policy while preserving later failures and lessons. It never deletes unsuccessful branches, resets attempts or recreates a saved molecular result. Existing user-defined regression/plateau gates and acceptance interfaces remain external.

## Current adoption boundary

The inspected code already saves raw proposals/receipts, measurement registries, frozen history snapshots and evidence-bound skill-review receipts. Its prompt uses completed history plus non-anchor evidence, and supplies original proposal predictions when present. It does not yet implement a dedicated lesson archive with review transitions, selected lesson retrieval/source-fetch, or acknowledgment of model-proposed memory writes. Do not claim these are active because this reference exists. See the accompanying repository assessment for exact source hashes and functions.
