---
name: jump-guidance-proposal
description: Propose novel complete guidance schedules with stage-based forward exploration, mandatory Early revisits, and bounded historical continuation.
metadata:
  version: "2.15.0"
---

# Path Exploration and Proposal

## Build the Portfolio
Prefer five novel complete schedules when legal opportunities and budget permit. In the first round, propose separate single-window changes in two different Early windows. If Early has only one window, use opposite signed directions when feasible. Every round requires two different nonzero parent-relative Early sign vectors. Different magnitudes with the same signed changed-window pattern count once; Middle or Late changes cannot create a second Early direction.

Select a measured source and a stage focus. Start in Early; use advance for a later stage, refine for the same stage, and backtrack for an earlier stage. At least one proposal must actually change a member of the declared focus stage. Respect the context's mandatory Early revisit after a Late update. Joint results support the complete schedule, not an isolated causal claim.

Use remaining opportunities for forward extension, suffix adaptation, joint reallocation, or an eligible historical branch. Do not bind scales within a stage. Use the declared grid-distance tiers and cumulative budget. Stage direction and branch rules determine admission; a bold label does not replace Early exploration. Only the program can certify budget or legal-availability exceptions.

## Reuse Before Measuring
Use lookup_schedule for a complete schedule that may already exist. Reference compatible measured winners, parents and controls through evidence_ids. Do not put an already measured array in proposals, even with a new parent, name or explanation. A previously measured prefix combined with a different suffix is novel only if the complete array is unmeasured. Failed and reserved profiles cannot be retried by renaming them.

## Submit and Audit
Submit complete arrays with the runtime-declared length and per-window integer grids. Retain actual anchor_id, evidence_ids, optimization_target and exploration_tier. Give brief evidence background, unmeasured opportunity, stage interaction rationale and uncertainty in the existing explanation fields.

Use audit_action_batch with the complete portfolio and search_focus. In the first decision also supply stage_partition with early, middle, late window-ID arrays and reason. In later audit_action_batch calls, supply stage_partition=null to use the frozen mapping. Omit stage_partition from later final submissions. The focus contains stage, operation and source_id.

Use submit_final_response with decision, proposals, reason and search_focus; include stage_partition only when requested by the first-decision schema. For hold, proposals must be empty. Every admitted novel schedule executes in order with a complete population. No Selection or synthesis role is used. A duplicate returns its measured evidence for the single bounded repair; correct proposals explicitly rather than treating it as a new slot. No automatic replacement sampling is allowed.

See [Search policy](references/search-policy.md) for examples and methodological boundaries.
