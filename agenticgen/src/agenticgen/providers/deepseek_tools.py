"""Audited, bounded client-side tool sessions. Caller serializes a whole round."""
import json
import os
import time
from pathlib import Path

import httpx
import jsonschema

from agenticgen.uniguide.io import frozen, read, sha, value_sha, write
from .deepseek_api import response_diagnostics

POLICY = "deepseek-readonly-tools-v1"


class SessionStopped(RuntimeError):
    pass


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON number")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


class RoundBudget:
    """One ledger across private roles and synthesis, including failed HTTP attempts."""
    def __init__(self, folder, max_calls=30, max_output_tokens=30*16384):
        if type(max_calls) is not int or not 1 <= max_calls <= 1000:
            raise ValueError("Positive bounded per-round model call limit required")
        if type(max_output_tokens) is not int or max_output_tokens < 512:
            raise ValueError("Positive round output token allowance required")
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        frozen(self.folder/"POLICY.json", dict(max_model_calls_per_round=max_calls,
            max_output_tokens_per_round=max_output_tokens, unit="HTTP attempts across all roles",
            failed_attempts_count=True, cache_hits_count=False))
        self.max_calls, self.max_tokens = max_calls, max_output_tokens

    def rows(self):
        return [read(p) for p in sorted(self.folder.glob("call_*.json"))]

    def remaining(self):
        rows = self.rows()
        charged = sum(r.get("completion_tokens", r["reserved_output_tokens"]) for r in rows)
        return self.max_calls-len(rows), self.max_tokens-charged

    def reserve(self, role, request_hash, output_limit):
        calls, tokens = self.remaining()
        if calls < 1 or tokens < output_limit:
            raise SessionStopped("Round model-call/output-token budget exhausted")
        path = self.folder/f"call_{len(self.rows())+1:04d}.json"
        # Never refill an interrupted attempt. The caller holds the executor lock.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(dict(role=role, request_sha256=request_hash,
                           reserved_output_tokens=output_limit), stream)
        return path

    def account(self, path, wire):
        count = wire.get("usage", {}).get("completion_tokens")
        row = read(path)
        if type(count) is int and 0 <= count <= row["reserved_output_tokens"]:
            write(path, dict(row, completion_tokens=count))


def call_with_tools(messages, schema, output, *, registry, budget, response_validator=None,
                    response_parser=strict_json, model="deepseek-flash", timeout=240,
                    transport=None, attempts=3, sleep=time.sleep, max_output_tokens=16384,
                    max_format_repairs=1, thinking=True, reasoning_effort="high",
                    final_response_mode="text"):
    """Replay durable responses; never automatically repeat an ambiguous/failed model request.

    Registry.execute is deterministic/read-only and returns evidence receipts. Tool errors
    are returned to the model. Sampling and evaluation are not registered capabilities.
    """
    if model != "deepseek-flash" or type(thinking) is not bool or reasoning_effort not in {"low", "high", "max"}:
        raise ValueError("Explicit supported model/thinking configuration required")
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("Network attempts must be between one and three")
    if type(max_format_repairs) is not int or not 0 <= max_format_repairs <= 2:
        raise ValueError("Bounded format repair allowance required")
    if type(max_output_tokens) is not int or not 512 <= max_output_tokens <= 98304:
        raise ValueError("Bounded output token allowance required")
    if final_response_mode not in {"text", "strict_tool_v1"}:
        raise ValueError("Unsupported final response transport")
    strict = final_response_mode == "strict_tool_v1"
    endpoint = "https://api.deepseek.com/chat/completions"
    wire_tools = registry.definitions
    wire_contract = None
    if strict:
        from .strict_tools import FINAL_TOOL, ENDPOINT, definitions

        endpoint = ENDPOINT
        wire_tools, client_checks = definitions(registry.definitions, schema)
        wire_contract = dict(mode=final_response_mode, endpoint=endpoint,
            final_tool=FINAL_TOOL, tools=wire_tools, client_only_checks=client_checks,
            full_application_schema_sha256=value_sha(schema),
            implicit_field_completion=False, plaintext_final_accepted=False)
    output = Path(output)
    session = output.parent/(output.stem+".session")
    session.mkdir(parents=True, exist_ok=True)
    binding = dict(policy=POLICY, messages=messages, response_schema=schema,
        tools=registry.definitions, registry_binding=registry.binding,
        model=model, thinking=thinking, reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens, max_format_repairs=max_format_repairs,
        attempts=attempts, timeout=timeout)
    if strict:
        binding["final_response_transport"] = wire_contract
    frozen(session/"BINDING.json", binding)
    if (session/"FAILED.json").exists():
        raise SessionStopped("Archived failed session requires explicit audited recovery")
    completion = (
        f"When ready, call {FINAL_TOOL} alone with the complete final object as arguments. "
        "Never finish with ordinary text or JSON content. Include every required property, "
        "including explicit empty arrays when required. No field is inferred from your explanation. "
        if strict else "When ready, return one complete final JSON object, no Markdown or extra fields. "
    )
    transcript = list(messages) + [dict(role="user", content=
        "Use native registered tool calls only when required evidence is missing. "
        + completion +
        "Copy evidence_id strings verbatim (ev_ followed by all 20 hexadecimal characters); never shorten them. "
        "Keep each text field comfortably below its maxLength; cite evidence without repeating tables or numeric arrays. "
        "Final response schema: "+json.dumps(schema, ensure_ascii=False, separators=(",", ":")))]
    repairs, step = 0, 0
    schemas = {d["function"]["name"]:d["function"]["parameters"] for d in registry.definitions}
    try:
        with httpx.Client(timeout=timeout, transport=transport, trust_env=True) as client:
            while True:
                step += 1
                event = session/f"step_{step:04d}"
                event.mkdir(exist_ok=True)
                request_path, wire_path = event/"REQUEST.json", event/"RESPONSE.json"
                base = dict(model=model, messages=transcript, tools=wire_tools,
                    thinking={"type":"enabled" if thinking else "disabled"})
                if thinking:
                    base["reasoning_effort"] = reasoning_effort
                else:
                    base["temperature"] = 0
                if request_path.exists():
                    stored = read(request_path)
                    if {k:v for k,v in stored.items() if k not in ("max_tokens", "tool_choice")} != base:
                        raise ValueError("Replayed tool transcript differs from saved request")
                    if not wire_path.exists():
                        raise SessionStopped("Ambiguous interrupted API request; not resent")
                    wire = read(wire_path)
                else:
                    if output.exists():
                        raise ValueError("Final output has no complete replayable transcript")
                    calls, tokens = budget.remaining()
                    limit = min(max_output_tokens, tokens)
                    if calls < 1 or limit < 512:
                        raise SessionStopped("Round model-call/output-token budget exhausted")
                    # Thinking mode cannot force a named/required function. Keep the
                    # submission tool available even on the final allowed request.
                    body = dict(base, max_tokens=limit,
                        tool_choice="auto" if strict else "none" if calls == 1 else "auto")
                    key = os.environ.get("DEEPSEEK_API_KEY")
                    if not key:
                        raise SessionStopped("DEEPSEEK_API_KEY is absent from this process")
                    frozen(request_path, body)
                    for attempt in range(attempts):
                        claim = budget.reserve(output.stem, sha(request_path), limit)
                        frozen(event/f"HTTP_{attempt+1}.json", dict(budget_claim=claim.name))
                        try:
                            response = client.post(endpoint, json=body,
                                headers={"Authorization":"Bearer "+key})
                        except (httpx.TimeoutException, httpx.NetworkError):
                            if attempt+1 == attempts:
                                raise SessionStopped("Network attempt allowance exhausted") from None
                            sleep(min(30, 5*2**attempt))
                            continue
                        if response.status_code in {429,500,502,503,504} and attempt+1 < attempts:
                            sleep(min(30, 5*2**attempt))
                            continue
                        if response.status_code != 200:
                            raise SessionStopped(f"Provider HTTP {response.status_code}")
                        wire = response.json()
                        frozen(wire_path, wire)
                        budget.account(claim, wire)
                        break
                if wire.get("model") != model:
                    raise SessionStopped("Returned model differs from frozen model")
                choice = wire["choices"][0]
                message = choice["message"]
                calls = message.get("tool_calls")
                final_calls = ([c for c in calls if c.get("function", {}).get("name") == FINAL_TOOL]
                               if strict and calls else [])
                # Keep the full assistant message including opaque reasoning_content.
                assistant = dict(message, role="assistant")
                if calls:
                    if choice.get("finish_reason") != "tool_calls" or len(calls) > 8:
                        raise SessionStopped("Invalid or oversized tool-call batch")
                    ids = [c["id"] for c in calls]
                    if len(set(ids)) != len(ids) or any(not isinstance(i,str) or not i for i in ids):
                        raise SessionStopped("Duplicate or absent tool-call identity")
                    if not final_calls:
                        transcript.append(assistant)
                    for index, call in enumerate(calls if not final_calls else []):
                        try:
                            if call.get("type") != "function":
                                raise ValueError("Only registered function calls are supported")
                            name = call["function"]["name"]
                            if name not in schemas:
                                raise ValueError("Tool is not registered")
                            args = strict_json(call["function"]["arguments"])
                            jsonschema.validate(args, schemas[name])
                            result = registry.execute(name, args)
                        except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as error:
                            result = dict(status="tool_error", error_type=type(error).__name__,
                                message="Unknown tool, invalid arguments or unavailable compatible evidence; use registered schema and known IDs.")
                        saved = dict(tool_call_id=call["id"], call=call, result=result)
                        frozen(event/f"TOOL_{index:02d}.json", saved)
                        transcript.append(dict(role="tool", tool_call_id=call["id"],
                            content=json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))
                    if not final_calls:
                        continue
                if not calls and choice.get("finish_reason") != "stop":
                    raise SessionStopped("Model output incomplete; no automatic length retry")
                result = None
                try:
                    if strict:
                        if len(final_calls) != 1 or len(calls or []) != 1:
                            raise ValueError(f"Call {FINAL_TOOL} alone; required fields: {schema.get('required', [])}")
                        if final_calls[0].get("type") != "function":
                            raise ValueError("Final submission must be a native function call")
                        content = final_calls[0]["function"].get("arguments")
                    else:
                        content = message.get("content")
                    if not isinstance(content, str):
                        raise ValueError("Final response is not JSON text")
                    result = response_parser(content)
                    schema_errors = list(jsonschema.Draft7Validator(schema).iter_errors(result))
                    application_error = None
                    if response_validator is not None:
                        try:
                            response_validator(result)
                        except (ValueError, jsonschema.ValidationError) as error:
                            application_error = error
                        except (KeyError, TypeError, IndexError):
                            if not schema_errors:
                                raise
                    if schema_errors or application_error:
                        diagnostics = (response_diagnostics(schema_errors[0], result, schema)
                            if schema_errors else {})
                        if application_error is not None:
                            diagnostics["application_error"] = str(application_error)[:500]
                        diagnostics["valid_evidence_ids"] = sorted(registry.receipt_ids)
                        raise ValueError(json.dumps(diagnostics, ensure_ascii=False))
                except (ValueError, jsonschema.ValidationError) as error:
                    if repairs >= max_format_repairs:
                        raise SessionStopped("Final schema/action repair allowance exhausted") from None
                    repairs += 1
                    if result is None:
                        diagnostics = (response_diagnostics(error, result, schema)
                            if isinstance(error,(json.JSONDecodeError,jsonschema.ValidationError))
                            else dict(error_type=type(error).__name__,reason=str(error)[:300]))
                    transcript.append(assistant)
                    if strict and calls:
                        # Every native call needs a matching tool result before a
                        # repair turn. Mixed final/evidence batches execute neither.
                        for index, call in enumerate(calls):
                            result_error = dict(status="final_submission_rejected",
                                validation=diagnostics, evidence_executed=False)
                            frozen(event/f"TOOL_{index:02d}.json",
                                dict(tool_call_id=call["id"], call=call, result=result_error))
                            transcript.append(dict(role="tool", tool_call_id=call["id"],
                                content=json.dumps(result_error, ensure_ascii=False)))
                    instruction = (f"Replace the complete arguments in one {FINAL_TOOL} call, alone. "
                        f"All fields are mandatory: {schema.get('required', [])}. "
                        "For selection, supply selected explicitly: [] for hold, otherwise existing candidate IDs. "
                        if strict else "Replace the entire final JSON; ")
                    transcript.append(dict(role="user", content=instruction+
                        "preserve evidence and do not invent measurements. Validation: "+
                        json.dumps(diagnostics, ensure_ascii=False)))
                    continue
                frozen(output, result)
                if strict:
                    frozen(event/"TOOL_00.json", dict(tool_call_id=final_calls[0]["id"],
                        call=final_calls[0], result=dict(status="final_submission_accepted",
                        response_sha256=sha(output), evidence_executed=False)))
                evidence = sorted(registry.receipt_ids)
                receipt = dict(policy=POLICY, binding_sha256=sha(session/"BINDING.json"),
                    response_sha256=sha(output), steps=step, interface_repairs=repairs,
                    evidence_ids=evidence, transcript_files={p.relative_to(session).as_posix():sha(p)
                        for p in sorted(session.glob("step_*/*.json"))})
                if strict:
                    receipt.update(final_response_mode=final_response_mode,
                        final_tool=FINAL_TOOL, final_tool_call_id=final_calls[0]["id"],
                        complete_object_explicitly_submitted=True)
                frozen(output.with_suffix(".receipt.json"), receipt)
                return result
    except Exception as error:
        # No provider headers, key, proxy values or arbitrary response text in error receipts.
        frozen(session/"FAILED.json", dict(error_type=type(error).__name__,
            step=step, preserved_transcript=True, automatic_restart=False))
        raise
