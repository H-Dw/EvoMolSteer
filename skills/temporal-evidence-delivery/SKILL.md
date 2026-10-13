---
name: evomolsteer-temporal-evidence-delivery
description: Deliver feature-specific observed temporal evidence through the Analyst and Designer interfaces.
---

This is the common source delivery protocol for all temporal experiment
conditions, including conditions without a temporal research extension. It does
not choose a reward, dose direction, stage boundary or preferred feature.

For Analyst, inspect the supplied catalogue. `curves_available` and
`stage_profiles_available` are independent. When profiles are absent, actual
continuous curves are still accessible with `get_curve`. Select an observable
relevant to the task, read its curve over the supplied support, and cite the
returned `ref` in each hypothesis that uses it. Trend summary rows and the
`curve_support` audit row do not deliver a feature's actual curve. Read a field
page if an observation explicitly indicates that its values were summarized.

For Designer, use the actual Analyst hypotheses and the supplied
`hypothesis_temporal_sources`. For an edit, cite the hypothesis ID and an
inherited handle for at least one of its relevant observed curves. An unrelated
curve is insufficient. The registered recipe levels are proposed interventions;
they are not optimal parameters learned from a temporal slope.

The host may reject a valid JSON finish with `finish_feedback`. Address its
specific diagnostic using the registered tools and actual observations, then
finish again within the remaining turn budget. Do not guess references or call
unavailable tools. Omit unsupported hypotheses; retain/defer when a reward edit
is not justified. Reading and citation prove delivery, not scientific adequacy
or an improvement in generated quality.
