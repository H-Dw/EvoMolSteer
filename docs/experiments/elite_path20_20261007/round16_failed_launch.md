# Round 16 first execution

The geometry-kernel arm failed before producing any inference-step/terminal
result. `numpy_geometry` expects an array for landmark indexing; the JSON
reference supplies a list. The new caller had not converted this input. This is
an interface defect, not an observed decline in generation performance.

The caller now converts landmark lists to a NumPy array. A JSON-list geometry
test verifies finite coordinates and agreement of the scalar reward's exact
derivative with central finite differences. The frozen program, reference,
dose, window, checkpoint, seed and geometry formula remain unchanged. The
dispatcher retains the failed source commit, log, exit code and launch metadata
before retrying the same scientific round. No later trial proceeds before this
round completes its actual FLOWR VJP audit and terminal evaluation.
