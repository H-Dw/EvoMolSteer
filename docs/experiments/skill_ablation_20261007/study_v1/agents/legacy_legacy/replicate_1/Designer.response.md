# Designer decision

**retain_existing — program_20**, with `updates: {}`. Freeze the registered endpoint point-cloud program for the proposed matched-native paired comparison. This is an exploratory candidate, not an independently validated winner.

Its reported discovery mean affinity delta is **+0.230299**, validity change **+0.010000**, MMFF relief per heavy atom median/p90 **0.476946/0.954550**, and best valid predicted pIC50 **8.237148**. Programs 3/4 nearly tie the mean (**+0.229293/+0.229364**); program 4 has best valid **8.558043** and validity **-0.020000**. Program 19 has lower p90 **0.907866** and mean **+0.212839**. Program 8 has p90 **0.731024** and mean **+0.139001**. All 22 summaries reuse adaptive discovery batches; paired uncertainty and a strain noninferiority criterion are absent.

Keep K=4, offline score beta=2, teacher temperature=4 A², tau=0.5, source robust delta=2, relative RMS dose=0.33 against predictive flow, time ramp power=0 and the original reference. The registered Hungarian robust coordinate mixture is pulled through the actual FLOWR endpoint Jacobian, with self-conditioning, anchor, assignments and priors fixed. Affinity outputs are detached. No extra production forward/head call is added. Apply bounded guidance after native integration; its logged reward response is first-order at the old state. Use all 50 supplied nodes 0.00–0.49, then continue native inference to 1. Source caps and free native graph transitions remain part of the retained program.

Do not promote the observed associations to new control laws: all 24 fitted curves are disallowed and all 63 regional node curves cross zero. Region 00 soft-mass enrichment is **+0.639235 z**, q=**0.000187945**, positive in **14/14** batches, but first/last node effects are **-0.365951/-0.275338**. Overlap does not identify independent causal motifs. Compressed/full teacher-score correlations are **0.101826/0.064078**; the audit does not prove full geometry improves affinity. Zero of 400 survival tests passes joint BH; **349/700** nodes have fewer than 3 ancestors and **531/700** fewer than 6. Terminal labels at **0.99** are joint-head forecasts, distinct from final molecule rescoring; extinction credit stays null.

Registry round09/program_0 has mean **-0.041429** and round10/program_18, at dose **0.51**, has mean **+0.206789**, below round07/program_3 (**+0.229293**, dose **0.3**). These are not window-to-final response measurements. The four numeric diagnostic cases, finite-difference audit, full zero/native parity, delivered dose and terminal persistence are absent. No dose change, new region or lineage reference is inferred. Future comparisons must retain all failure denominators and report coverage, mean/unique mean, best valid/PB-fast, unique Top5 and physical tails separately.

| Binding | SHA256 |
|---|---|
| Literal request/prompt/input | `b32da2291d008ad429198fc460a9a6c6ea49ce4fc614f123e18555e626304716` |
| Combined role treatment, terminal LF preserved | `5d55a795865b9fff365daee5043a68d3ce856d7fe8eedeaec42fe14a7560bb49` |
| Evidence | `5c8f625fcbc1dc6cfba8c0fc0e8cfcc8aa92fe3ea7133dccbe7e3f308a5f5237` |
| Analyst response | `3f1ef005e103595fb901e742baafa40c0147b861666c12cf47b215c1899121d6` |
| Formula registry | `a79de7b4ea03ceb7dd590ead5b302ad2752cbf6b8d8be6d29ebb6e6f718daad8` |
| Retained source program | `7a0455c943619ae5a07e29590859ee14a2eaecd92a1accbea98389512015cce6` |
| Selected endpoint reference, bound metadata | `d706173b74a937c5becac08ccbcff187f48e50ca3303d9021667e466fa5fb77c` |
| Supplied regional prior, not activated | `5a6b23af25e4b322ae8d49a77741a08954b6360aaacf906c4862414aaeca6b3e` |

Validation: the exact supplied JSON schema passes, all response evidence IDs resolve, and the request/treatment/evidence/Analyst/registry/program hashes match. Only the request, its bound evidence/Analyst/registry and the registered retained program were used. No inference or source edits were performed. GPU behavior, numerical execution checks and final affinity remain unverified; independent review is required before activation.
