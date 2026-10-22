# Deployment and continuation

Read [plan.zh-CN.md](plan.zh-CN.md) first. It is the active scientific protocol.
The former mean-first 100-candidate council is historical only. The current
automation remains PAUSED until native acceptance and full V5 review authorize V6.

## Source and environment

Use the existing repository/branch, with agenticgen as an installable src-layout
subproject. Migration commit `b105f0b82fa18e563e98a2719f627f477e571a95`
preserves the original source history. New scientific changes use later commits.
Always test and commit locally, push GitHub, then fetch a fixed commit remotely.
Do not deploy uncommitted working files as a scientific runtime.

The only execution host is root@ssh.zzai.scnet.cn:11762, workspace /opt/MolSteer.
Use its audited GPU Python as the basis for an isolated environment. Preserve
Torch/DTK and the original FLOWR checkpoint; do not resolve a GPU stack through
an automatic package upgrade. Check dependencies before installing the package.
No fallback import of evomolsteer or an old result-directory source tree is allowed.

Set AGENTICGEN_REPOSITORY to the exact deployed checkout, AGENTICGEN_HOST_CONFIG
to the host JSON if overridden, and AGENTICGEN_PYTHON to the installed interpreter
when using scripts/run.sh. The installed command must work from another directory.
API key is read only from DEEPSEEK_API_KEY; inherit the existing proxy environment
without logging values. Never persist SSH/API credentials.

## New experiment

Result root:
`/opt/MolSteer/generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022/`

1. `agenticgen doctor`: inspect installed source and package resource locations.
2. `agenticgen prepare --source-root <audited-history-root>`: new empty root only.
   This expands original PLIP core by exactly one covalent hop. It imports historical
   baselines/teacher metadata, not old generated trajectories or optimization answers.
3. `agenticgen validate`: create the 50 unique shared R0 records, validate fixed
   step-10 restarts, common randomness, four schedules in batch100, gradient finite
   differences, units and displacement cap. Diagnostic probes are not scientific
   comparison groups. They retain documentary records and are structurally pruned.
4. Complete substantive review of all 52 V5 logical sessions from the previous
   protocol, including every actual unique request, tool, proposal and failure.
   Preserve committed review document, individual coverage shards and source hashes.
5. Write a real V6 review approval only after all evidence is reviewed. It binds
   previous version/root, FINALIZED and manifest hashes, committed document/coverage,
   installed generic skills, exact deployment commit, new PREPARED and ACCEPTANCE
   hashes and the new objective priority. `agenticgen review-check --version v6`
   must pass. A file's existence is not approval.
6. `agenticgen launch --version v6`: API preflight, inherited environment, detached
   process. Verify command identity, worker/pipeline and real completed artifacts.
   Then resume the updated automation at 40-minute intervals.

Do not run prepare again or overwrite PREPARED/ACCEPTANCE for a runtime fix after
formal data exists. Preserve failure artifacts and follow a documented compatibility
assessment and tested fixed-commit deployment. Never change scientific settings to
bypass an exception. Already completed generation must not be re-sampled by an
analysis recovery. Full coverage is required separately for every next version.

## Evidence and storage

Each seed selects original slots 0–4. Native random calls still use twenty-row
shapes; padding is discarded and never counted as a candidate. Every hypothesis
has 25 attempts and every wave 100. Five rounds, two teachers give 1000 optimization
records per version. Counts differ from native integration steps, which remain 100.

All generation parents are shared R0 before update10. The per-chain affinity TOP1
archive is only selected output. A new best snapshot is verified and published
atomically after quality evaluation; only then may old structural/tensor payloads
be pruned. All scores, condition diagnostics, PLIP XML, quality, requests/responses,
learning updates and failed attempts remain. Incomplete evaluations retain payloads.

`agenticgen report --version vN` creates continuous condition/functional reports.
`agenticgen analyze` collects versions, historical descriptive references, costs
and missing receipts. `review-pack` binds actual evidence. Full GPT review and skill
revision precede each review-check/launch. New versions start with empty dynamic
knowledge but the same static teacher profile and shared R0.

Root running status is insufficient: inspect both teachers, each wave's sampling,
quality, council, retention and cleanup markers, real processes and recent errors.
Old failures are historical if corresponding later completion markers prove recovery.
Healthy unchanged states are quiet. V25 delivery ends the automation.

## Frozen continuous-RMSD protocol and range/rollback gates

The authorized matrix is V6–V25, 1000 optimization records per version plus 50 shared R0 records (maximum 20050). The primary ranking band is 0.02 A, with offline 0.01/0.05 A sensitivity. It never modifies rewards or displacements.

For the existing prepared root with R0 but no scientific version binding, use `agenticgen stage-protocol --reason <concrete rationale>` from the tested fixed commit. It creates an append-only protocol directory and active pointer; original preparation, configuration, acceptance and R0 remain intact. It refuses changes to the sampled R0 settings. Run `validate` against the staged protocol; its new acceptance receipt is written inside the protocol directory. Do not overwrite original root provenance. Until native acceptance and real V5 review pass, `launch` is forbidden.

Each version freezes `versions/vN/search_policy.json` with upper_amplitude, coarse_probe_scale, fine_resolution, max_range_probes and max_changed_intervals. Missing V6 policy defaults to U=1, s=0.2, q=0.1. Later versions require the preceding frozen policy. V6–V10 reject U>1. From V11 a reviewed increase follows 1→5→10→15… at most one level per version; prepare the explicit JSON before approval. Policy is bound to the version and every LLM prompt.

`review_approval.json` additionally binds `search_policy_sha256` (canonical value hash) and `condition_comparison_sha256`. Expansions require range_expansion with substantive boundary_response, delivered_movement_and_clipping, unresolved_hypothesis plus a committed document/document_sha256. The existing full 52-session evidence gate remains mandatory.

Reports record primary and sensitivity-band representatives, excluding R0 and selected per-chain archives. `analyze` writes version_decisions.json, but this is advice, not scientific approval. The launch gate recomputes advice from immutable finalized reports. After two adjacent regressions approval must bind rollback.restored_version, restored_binding_sha256, source_binding_sha256, and a committed code_compatibility_document/code_compatibility_sha256. Current skills and range policy must match the historical-best binding exactly. Code fixes need an explicit compatibility assessment. Preserve commit history; make a new restoration commit. New version memory starts empty.

Expanded probes have at most one selected range_probe and at most two edited intervals; other proposals follow local evidence-based step limits. Tenth-grid resolution never changes. The native tensor schedule validates the same approved bound; maximum displacement remains 0.025 A. Quality evaluation checks chemical pose mappings and records N/A when a predefined subgraph has no legal match. All documentary diagnostics remain after pruning.

At present remote deployment must be confirmed by an actual receipt. A local test or a prepared command is not evidence of server execution.
