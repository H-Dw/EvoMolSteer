# Read-only tool batches: 32 calls per response

Raise the client-side `deepseek_tools` batch limit from 8 to 32. The model's
runtime instruction, session binding and final receipt expose the same limit.
Responses containing 13 or 15 evidence calls, as seen in the failed campaign,
now execute every registered call in order. Thirty-two calls are accepted;
thirty-three stop before any execution, without silent truncation or retries.

This changes calls per model response, not the separate per-round model-request
or output-token budgets. Strict final submission remains exactly one standalone
`submit_final_response` call; proposal, evidence and scientific-budget checks
remain enforced. No skills release is edited. Frozen experiments keep their
original checkout, and archived failures are not automatically restarted.
