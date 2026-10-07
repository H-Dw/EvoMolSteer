# Sequential path campaign: 20 rounds

Frozen winner: round 5; master seed 42; threshold 8.258902 pIC50.

Rounds 1–3 are local prerequisites; rounds 4–17 are inference controls/adaptive screens; rounds 18–20 are frozen confirmation.
All-output head scores include failed molecular decoding and are shown beside valid-output scores. PB is the dock_fast subset. Strain is MMFF local relaxation relief, not binding energy.

| Round | Module | Parent | N | Mean | Valid mean | Max | Elite | PB | Strain median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | "typed_path_graph" | — | — | — | — | — | — | — | — |
| 2 | "decoded_terminal_credit" | — | — | — | — | — | — | — | — |
| 3 | "terminal_path_teacher_library" | — | — | — | — | — | — | — | — |
| 4 | "incumbent_control" | None | 100 | 7.5377 | 7.5313 | 8.1320 | 0 | 0.97 | 0.5054 |
| 5 | "terminal_path_teacher_source" | 4 | 50 | 7.6119 | 7.6133 | 8.1691 | 0 | 0.94 | 0.5142 |
| 6 | {"reward_view": "endpoint_path_value"} | 5 | 50 | 7.4733 | 7.4733 | 8.1419 | 0 | 1.00 | 0.5456 |
| 7 | {"native_rms_ratio": 0.66} | 6 | 50 | 7.3931 | 7.3910 | 8.2627 | 1 | 0.94 | 0.5365 |
| 8 | {"teacher_neighbors": 1000} | 7 | 50 | 7.2405 | 7.2465 | 8.0006 | 0 | 0.96 | 0.7106 |
| 9 | {"teacher_score_beta": 4.0} | 7 | 50 | 7.3677 | 7.3473 | 8.1348 | 0 | 0.96 | 0.5571 |
| 10 | {"mixture_temperature": 1.0} | 7 | 50 | 7.3891 | 7.3864 | 8.1460 | 0 | 0.96 | 0.5676 |
| 11 | {"time_ramp_power": 1.0} | 7 | 50 | 7.4536 | 7.4536 | 8.5238 | 1 | 1.00 | 0.5595 |
| 12 | {"path_history_mix": 0.5} | 11 | 50 | 7.4348 | 7.4348 | 8.5238 | 1 | 1.00 | 0.5593 |
| 13 | {"path_history_mix": 0.8} | 11 | 50 | 7.4463 | 7.4604 | 8.5238 | 1 | 0.98 | 0.5570 |
| 14 | {"reference_sha256": "219182cde54b99b1bdacbc409677292dc91dc7bf7ce9fe72587921c19c39e651"} | 11 | 50 | 7.4718 | 7.4355 | 8.5204 | 3 | 0.88 | 0.5461 |
| 15 | {"teacher_endpoint_temperature_A2": 2.0} | 14 | 50 | 7.4369 | 7.4024 | 8.5205 | 2 | 0.92 | 0.5495 |
| 16 | {"path_kernel_space": "geometry"} | 14 | 50 | 7.3753 | 7.3521 | 8.1066 | 0 | 0.92 | 0.5701 |
| 17 | {"path_contrast_amplitude": true} | 14 | 50 | 7.4632 | 7.4632 | 8.2419 | 0 | 0.98 | 0.6366 |
| 18 | {} | 5 | 100 | 7.5118 | 7.5105 | 8.1247 | 0 | 0.96 | 0.5280 |
| 19 | {} | 4 | 100 | 7.6066 | 7.6001 | 8.1522 | 0 | 0.96 | 0.5137 |
| 20 | {} | 5 | 100 | 7.5998 | 7.6073 | 8.2579 | 0 | 0.96 | 0.5204 |

Pooled independent confirmation:

Selected mean 7.555761; native mean 7.440478.
Paired gain 0.115282; batch-bootstrap CI95 [0.05839667320251464, 0.17216752052307127].
Adaptive elite records 9, distinct graphs 4. No region-specific causal mechanism is established by these counts.

Round 8 uses the checksum-retained evaluator repair replay. Reported physical failures are never replaced by an evaluator exception.
See validation.json for per-panel, incumbent, PB-qualified tail, energy and historical Steer comparisons. Few deterministic batches do not establish broad generalization.
