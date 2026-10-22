# V6 analysis repair and full rerun

The previous worker exited after both teachers' R1 sampling and quality completed. Neither teacher committed the round. No V6 finalization exists. The user explicitly requests a tested repair followed by removal of the failed candidate outputs and a complete five-round rerun.

## Causes and correction

- Top1 G1 calculated a proposal change from R0 `[5,5,5,5]` instead of its actual current action anchor `[7,5,5,5]`. The executor correctly rejected the illegal change. The request now names both references separately, provides exact per-interval eligible values, and binds `reference_ticks` to the current anchor in the response schema. The executor still checks all joint constraints. It does not substitute a scientific decision.
- Top2 synthesis had 327777 conservative input bytes against 262144. Repeated response-edge and condition evidence caused the excess; this was not output truncation. Shared evidence blocks preserve all values and candidates, with exact reconstruction and a source hash. On the failed payload, the encoded representation decreased from 297012 to 187903 bytes, with 34 shared blocks. Expanded evidence hash: `115999f43468f627bbb428a2325658c248eeb279fd8d52b418b1bf5bf764277f`.
- Authorized input cap becomes 524288 bytes; soft cap becomes 131072. Output remains 32768 private / 65536 synthesis tokens. These are separate budgets. The official [DeepSeek model documentation](https://api-docs.deepseek.com/quick_start/pricing/) was checked on 2026-10-23; the local byte cap is not presented as a tokenizer-exact token count.

## Integrity and testing

Local full suite: 78 passed, 12 dependency-related skips. New tests cover the original illegal-anchor failure, expanded-range compatibility, lossless evidence reconstruction, independent budget controls, scoped deletion, native-validation gating, interrupted deletion, R0 tampering, staged preservation, scientific-source-change rejection, and finalized-version rejection. Real LLM compliance and performance require the new run; unit tests are not evidence of either.

No changes to model, sampler, rewards, gradient cap, seeds, common random streams, native transitions, scoring, scientific matrix, frozen skills or shared R0. Only analysis presentation, eligibility diagnostics, input budget and the explicitly authorized rerun operation change. Remote installation, tests, native acceptance, deletion and launch must be documented by actual receipts before they are called complete.

Old candidate result files are removed only after fresh acceptance. Original requests, responses, receipts and deletion hashes live outside the new `versions/v6` tree and cannot populate new decisions or formal results. Failed-attempt cost is retained separately, not hidden as free work. The old 200 candidate outcomes are excluded; the new full V6 has 1000 optimization attempts plus the unchanged shared R0.
