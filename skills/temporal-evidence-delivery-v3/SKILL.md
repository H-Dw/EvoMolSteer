---
name: temporal-evidence-delivery-v3
description: Deliver feature-specific observed temporal evidence through the registered role interface.
---

This opt-in delivery protocol supplements the chosen research Skills. It does not
select a reward, change dose or temperature, or treat temporal association as a
measured benefit of guidance.

## Analyst: observations before actionable hypotheses

Use `get_curve` to read one feature, or `get_curves` to read up to eight named
features in one bounded observation. Stage profiles, when available, are also
source-bound temporal evidence. A catalogue, summary row, or curve-availability
flag is not the curve itself. Returned handles carry file identity and pointer;
never invent or edit a handle.

In each hypothesis, declare `curve_features` using the exact catalogue names of
the observables it discusses. Cite a returned curve/profile handle for **every**
declared feature in that hypothesis's `evidence_refs`. If citing a feature's
summary as supporting evidence, declare that feature and read its actual curve
as well. Read bounded fields when a tool returns a container preview and the
claim requires values not yet shown. A handle alone does not substantiate a
quantitative claim.

Use the diagnostic's `next_reads` to retrieve missing evidence; when a curve is
already observed, use its supplied handle. If the proposed direction lacks
support, omit that hypothesis and preserve the uncertainty or counterevidence
in findings/limitations. Do not repeatedly submit the same unsupported finish.
No minimum number of hypotheses is required. After repeated unchanged gaps,
the finish schema admits only observed features with matching citations; tools
remain available to obtain more evidence.

## Designer: trace each adopted feature into a registered intervention

Read the Analyst handoff and inherited reference map. For an edit, cite the
actual hypothesis ID and matching temporal evidence for **each feature of each
adopted hypothesis**. Unrelated curves and aggregate summaries cannot replace
these sources. Adopt only the hypotheses needed for the registered intervention.

Explain why the observed pattern makes that intervention worth testing and give
concrete failure conditions. Historical time derivatives are not spatial reward
gradients and do not estimate an optimal guidance dose. Retain/defer when the
handoff cannot justify an edit. Scientific support still requires review and
generated quality measurements; passing the delivery gate is not proof.

## Response syntax

Follow the current per-turn JSON Schema. Return one complete action object,
without Markdown or extra fields. Close each array with `]` and each object with
`}`. `failure_modes` is an array of strings, e.g.
`"failure_modes": ["The proposed response does not improve the primary metric"]`.
If a syntax diagnostic names an unclosed container, return the full corrected
action with unchanged scientific content. Never ask the host to insert a bracket
or fabricate a citation. Host validation and deterministic extraction remain
mandatory; prompt compliance is not assumed.
