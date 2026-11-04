---
name: jump-guidance-history
description: Review measured stage-based guidance paths, compatible historical schedules, and conditional prefix evidence before proposing new measurements.
metadata:
  version: "2.15.0"
---

# History and Evidence

## Objective and Evidence
Minimize the arithmetic mean of terminal condition-aligned RMSD over the complete 50-candidate population. Use the fixed terminal correspondences and the declared proper rigid alignment. Intermediate noisy states, predicted endpoints, update magnitudes, and chemistry are diagnostics; do not use them to prune a trajectory or replace the terminal objective.

## Organize the Search
Read controllable window IDs, temporal bounds, legal grids, calibration evidence, and the saved stage context. In the first decision, partition the ordered controllable windows into Early Control Stage, Middle Control Stage, and Late Control Stage. Begin with approximately balanced window counts; adjust contiguous boundaries using temporal spans and measured evidence, and explain briefly. The program requires three nonempty consecutive stages covering every controllable window. With at least four controllable windows, Early must contain at least two. The accepted partition is fixed for this teacher.

Stages organize search; each window retains an independent scale. The prefix consists of Early and Middle, and the suffix consists of Late. Review each prefix's measured suffixes and best complete result. A poor result under one suffix does not exclude the prefix under another suffix. Keep exploratory branches distinct from the incumbent.

## Use Feedback
Start in Early, extend toward Middle and Late as evidence develops, and revisit earlier choices. An adopted Late change requires a new Early revision in the next available round, anchored at the new incumbent and preserving its Late scales. Identify Early comparisons tested only under previous suffixes and use those background-specific gaps to motivate the revision. Earlier choices remain provisional. After changing a prefix, reconsider higher, lower, and jointly redistributed suffix scales.

A historical nonwinner prefix receives one suffix adaptation opportunity per batch and available credit. Improvement of its own best complete result restores credit; otherwise it pauses. Reopening requires new relevant measured suffix evidence identified by the context. Changing parents or names does not reset credit. All branches remain in history.
