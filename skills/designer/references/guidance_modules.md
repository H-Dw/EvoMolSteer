# Optional Designer advice

## Module: coherent_modes

Consider whether a multimodal coordinate objective preserves real conformations
better than independently averaged feature targets. Keep alternatives distinct
when averaging would create unsupported geometry. Compare the simplest such
objective with the supplied incumbent and compressed alternatives.

## Module: regional_emphasis

If regional evidence is reproducible, consider spatial emphasis within a coherent
coordinate objective. Check whether it improves the target outcome beyond global
matching and whether surrounding coordinates remain compatible. Regional
enrichment does not itself prescribe an attraction direction.

## Module: history_matching

Use parent-child increments only when their coordinate meaning, support and
uncertainty are supplied. An increment-matching term is a separate hypothesis;
compare it with a design without history, and avoid promoting position evidence
into evidence for beneficial velocities.

## Module: dose_calibration

Distinguish reward shape from delivered displacement. If the controller normalizes
the gradient, multiplying the whole reward may not change the dose. Use actual
updates, clipping and paired quality changes to diagnose a weak dose or a wrong
target; do not assume stronger guidance is monotonically better.
