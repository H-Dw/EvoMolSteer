# Reproduce the screening

The first wave uses the configuration in `configs/experiments/agentic_steer_20261014_v1/config.json`. Scientific Skills are pinned to commit `e13cc24c`; native generation and feedback definitions were deployed from `d3683d20`. The current live Skills include corrections tested in the separate `probes/scientific_contract_v2` calls. Do not silently replace the first wave's instructions.

From the repository with the package installed, or `PYTHONPATH=src`:

```sh
python scripts/run_agentic_steer.py --config configs/experiments/agentic_steer_20261014_v1/config.json --action prepare
python scripts/run_agentic_steer.py --config configs/experiments/agentic_steer_20261014_v1/config.json --action initial --condition C1
python scripts/run_agentic_steer.py --config configs/experiments/agentic_steer_20261014_v1/config.json --action feedback --condition C1
python scripts/report_agentic_steer.py --config configs/experiments/agentic_steer_20261014_v1/config.json
```

`prepare` requires the retained historical full snapshots, compact HDF5 packages and frozen terminal metrics. It does not copy coordinates into a second trajectory dataset. Request and response caches bind the exact instructions and schema; changed requests must use a new experiment directory rather than overwrite a previous call.

For fresh execution, expose `DEEPSEEK_API_KEY` and the named SSH password environment variable to the calling process. Use `run_agentic_pilot_remote.py` with an explicit SSH host/port, Git-deployed runtime and expected source commit. It sends JSON jobs through SFTP and verifies the runtime; it never uploads scientific source to bypass Git. The remote interpreter must use the server's existing FLOWR runtime libraries. Environment setup is outside scientific Skills.

Each job is independently runnable with `run_agentic_native_suffix.py --job PATH`. The checkpoint and FLOWR installation are configuration fields. A trial identity binds a historical full-state hash, node, step and reproducible child random stream. Common identities reuse one terminal observation regardless of the other transport-batch members.

Audit artifacts include native source hashes, exact restoring forecast differences, 70 executed suffix steps, no-SMC/no-gradient flags, final scoring and validity/energy records. The determinism check reran one identity and matched coordinate bytes, SMILES, predicted affinity and energy exactly.

Chemical decode failures retain compact float32 final coordinates and hard categorical channels in `*.failed_terminal.npz`, with vocabulary tied to the frozen checkpoint. Recovering those controls by deterministic replay is tracked separately and does not count as new independent observations.
