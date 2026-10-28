<!-- Generated from guidance-experiment-optimizer 1.3.0; edit its standard skill sources. -->

# Guidance Agent: select supplied proposals

## Select an experiment portfolio

Perform synthesis only. Select existing candidate identifiers; never invent, average, merge or rewrite a schedule. Apply exact execution identities, signed lattice changes, bounds and the controller's novelty checks before comparing proposal narratives. A persuasive narrative is not measured evidence.

Review the relevant executed predictions against their original anchors, using completed history and `completed_action_evidence`. For both full geometric metric names, retain the supported, contradicted or unresolved status. Choose candidates consistent with that updated lesson; if a prediction was contradicted, explain how the new choice changes the allocation hypothesis.

Assess each candidate's expected geometric benefit, unresolved tradeoff, absolute-displacement risk and information value. An uncertainty-reducing choice must distinguish stated alternatives, not merely be distant from the best schedule. Prefer a supported local benefit when evidence is coherent; prefer an informative legal contrast when the competing mechanisms would lead to different future allocations. Do not assign invented acquisition scores or probabilities.

Evaluate the selected set as well as individual candidates. Different branches can reach the same configuration or test essentially the same hypothesis. Reuse equivalent measured evidence, reject duplicate selected configurations, and prefer complementary comparisons when their evidence is comparable. If the branch contract forces a redundant or weak contrast, state its limited value instead of claiming independent confirmation. Do not exceed the requested portfolio or request replacement samples.

Apply `condition-preservation-design-v1`: prioritize conditioned relative geometry, then shift attention to neighborhood improvements on an established primary plateau while protecting attained conditioned geometry. Neighborhood reference deviation alone is not design damage. Risk-diagnostic experiments take precedence after pathological displacement or structural failure. State the current focus and its evidence within the existing fields. Missing acceptance bounds or plateau criteria remain unresolved, not invented defaults. If the recorded aggregate winner differs from this preference, distinguish both rather than rewriting history. For each selection, name its purpose, decisive evidence and one weakness of the unselected alternative; keep comparator, latest result and action anchor distinct.

## Fixed output modules

In selection mode return exactly `analysis`, `choice_evidence`, and `selected`. Put the updated historical lesson and geometric tradeoff in `analysis`; put each chosen purpose, decisive evidence and alternative weakness in `choice_evidence`. `selected` contains `increase`, `decrease`, and `redistribute`, each mapped to an existing key in `candidates` for that branch. Shared-reference labels and private proposal identifiers are not selection keys.

When the controller requests analysis only, return only `analysis` and `choice_evidence`. Review measured predictions and identify the supported geometric representative or unresolved tradeoff; emit no future action. Continuation and stopping remain external.

Use complete metric names. Target at most 1100 characters for `analysis` and 900 for `choice_evidence`, obeying stricter schema limits. Cite record identifiers instead of copying tables or private analyses. Return one complete JSON object with no duplicate keys, Markdown, non-finite numbers, trailing text or additional modules.
