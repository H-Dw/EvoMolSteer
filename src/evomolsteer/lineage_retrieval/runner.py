"""Isolated Codex app-server worker for the CK2 lineage retrieval study.

The model receives one sanitized packet and only the dynamic broker tools
declared below. This is a model-visible tool boundary, not an OS sandbox.
"""
from __future__ import annotations

import asyncio
import hashlib
import gzip
import json
import jsonschema
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


_FORBIDDEN_KEY = re.compile(r"(?:path|filepath|filename|directory|cwd|oracle|ground.?truth|final.?score|true.?score|fitness|mapping|private|raw.?id)", re.I)
_FORBIDDEN_ORACLE_VALUE = re.compile(r"(?:oracle|ground.?truth|final|true).{0,20}(?:score|affinity|pic50)|(?:score|affinity|pic50).{0,20}(?:oracle|ground.?truth|final|true)", re.I)
_PATH_VALUE = re.compile(r"(?:[A-Za-z]:\\|\\\\[^\\]+\\|(?:^|\s|[=:])(?:\.\.?[\\/])|(?:^|\s|[=:])/(?:[A-Za-z0-9_.-]+/){1,})")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MAX_PACKET_BYTES = 300_000_000


def _candidate_summary(packet: dict[str, Any]) -> dict[str, Any]:
    """Show every available seed; details remain in the bounded memory broker."""
    preferred = [
        'online_pic50', 'current_radius_gyration_A', 'forecast_radius_gyration_A',
        'current_centroid_x_A', 'current_centroid_y_A', 'current_centroid_z_A',
        'forecast_centroid_x_A', 'forecast_centroid_y_A', 'forecast_centroid_z_A',
        'current_planarity_fraction', 'forecast_planarity_fraction', 'forecast_asphericity',
        'forecast_cov_xx_A2', 'forecast_cov_xy_A2', 'forecast_cov_xz_A2',
        'forecast_cov_yy_A2', 'forecast_cov_yz_A2', 'forecast_cov_zz_A2',
        'current_nearest_pocket_A', 'forecast_nearest_pocket_A',
        'forecast_nearest_pocket_A_mean', 'forecast_ligand_contacts_4A',
        'forecast_pocket_contacts_4A', 'forecast_contact_pairs_4A',
        'forecast_clash_pairs_1p2A', 'forecast_residue_contact_footprint',
        'forecast_atom_N_count', 'forecast_atom_O_count', 'forecast_atom_F_count',
        'forecast_atom_Cl_count', 'forecast_atom_Br_count', 'forecast_bond_edge_count',
        'delta_current_Rg', 'delta_forecast_Rg', 'delta_forecast_centroid_xyz_A',
        'delta_forecast_contact_pairs_4A', 'delta_forecast_revision_rms_A',
    ]
    available = {k for c in packet.get('cards', []) for k in c.get('now', {})}
    columns = [k for k in preferred if k in available]
    def compact(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    rows = [[c['candidate_id'], *[compact(c['now'].get(k)) for k in columns]] for c in packet.get('cards', [])]
    return {'columns': ['candidate_id', *columns], 'rows': rows,
        'candidate_count': len(rows), 'all_candidates_included': True,
        'summary_precision_decimals': 6, 'details_available': 'get_candidate and get_history',
        'geometry_semantics': 'Cloud geometry is not binding energy. Principal-axis signs are arbitrary; compare covariance instead. Planarity_fraction is smallest covariance eigenvalue / eigenvalue sum: lower indicates a thinner cloud. Asphericity here is largest eigenvalue minus half the other two (A squared). Full stored precision is available from the broker.'}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _project_history_features(rows: Any, features: Any, affinity_available: bool) -> list[dict[str, Any]]:
    """Project bounded history to an explicit set of literal scalar columns."""
    if not isinstance(rows, list):
        raise ValueError("history source must be a list of bounded observation rows")
    if not isinstance(features, list) or not 1 <= len(features) <= 12:
        raise ValueError("features must contain one to twelve literal keys")
    if any(not isinstance(key, str) for key in features) or len(set(features)) != len(features):
        raise ValueError("features must be unique literal string keys")
    for key in features:
        if key != "online_pic50" and not key.startswith(("current_", "forecast_", "delta_")):
            raise ValueError(f"unsupported feature {key!r}")
        if key == "online_pic50" and not affinity_available:
            raise ValueError("online_pic50 is unavailable for this structure-only packet")
    available = sorted({str(key) for row in rows if isinstance(row, dict) for key in row
                        if key in {"step", "time", "online_pic50"} or
                        str(key).startswith(("current_", "forecast_", "delta_"))})
    unknown = [key for key in features if key not in available]
    if unknown:
        raise KeyError(json.dumps({"error": "unknown feature key", "unknown_features": unknown,
                                   "available_features": available[:80]}, ensure_ascii=False))
    columns = ["step", "time", *features]
    return [{key: row[key] for key in columns if key in row} for row in rows]


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _scientist_interval_payload(source: dict[str, Any], interval: int) -> dict[str, Any]:
    """Return one interval only; later teacher states never cross the tool boundary."""
    if not isinstance(interval, int) or isinstance(interval, bool) or not 1 <= interval <= 10:
        raise ValueError("scientist interval must be 1..10")
    low, high = (interval - 1) * 5, interval * 5
    states = [row for donor in source.get('teacher', {}).get('donors', [])
              for row in donor.get('observations', []) if low <= row.get('step', -1) <= high]
    keys = sorted({k for row in states for k in row if k in {'step', 'time', 'online_pic50'} or
        k.startswith(('current_', 'forecast_', 'delta_'))})
    def compact(value):
        if isinstance(value, float):
            return round(value, 5)
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    donors = []
    for donor in source.get("teacher", {}).get("donors", []):
        donors.append({'donor_id': donor.get('donor_id'),
            'observation_rows': [[compact(row.get(k)) for k in keys] for row in donor.get('observations', []) if low <= row.get('step', -1) <= high]})
    from .summaries import summarize_scientist_interval
    return {'interval': interval, 'step_start': low, 'step_end': high,
        'observation_columns': keys, 'donors': donors,
        'computed_observation_cards': summarize_scientist_interval(source, interval),
        'background': [row for row in source.get('background', []) if low <= row.get('step', -1) <= high],
        'geometry_semantics': 'Current/forecast cloud features have separate namespaces. All covariance components are in the fixed receptor frame; principal-axis sign changes alone are not rotation. *_planarity_fraction is smallest eigenvalue/sum, lower means thinner. Early slot clouds and raw graph labels are not formed physical molecules. These rounded observations support qualitative claims; original numeric precision stays in the frozen evidence package.'}


def _validate_packet(packet: Any, role: str) -> dict[str, Any]:
    if not isinstance(packet, dict):
        raise ValueError("packet must be a JSON object")
    data = json.loads(_json_bytes(packet))
    normalized_role = role.lower().replace("_", "-")
    selector_role = "selector" in normalized_role
    def extract_cutoff(obj: Any) -> int | None:
        if isinstance(obj, dict):
            for key in ("step", "current_step", "cutoff", "current_cutoff"):
                if isinstance(obj.get(key), int) and obj[key] >= 0:
                    return obj[key]
            for value in obj.values():
                got = extract_cutoff(value)
                if got is not None:
                    return got
        return None
    if normalized_role == "scientist":
        # The trusted source necessarily contains all observed states through 50;
        # only the progressive broker below controls model-visible time.
        cutoff = 50
    elif normalized_role in {"main-agent", "main", "optimizer", "main-optimizer"}:
        cutoff = None
    else:
        cutoff = extract_cutoff(data)
    if cutoff is None and normalized_role not in {"main-agent", "main", "optimizer", "main-optimizer"}:
        raise ValueError("worker packet requires an explicit step cutoff")
    cards = data.get("cards", [])
    if selector_role and not isinstance(cards, list):
        raise ValueError("selector packet.cards must be a list")
    if not isinstance(cards, list):
        cards = []
    seen: set[str] = set()
    hide_affinity = data.get("hide_affinity") is True or data.get("affinity_available") is False
    if hide_affinity:
        def strip_affinity(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: strip_affinity(v) for k, v in node.items() if not re.search(r"online_pic50|affinity|pic50", str(k), re.I)}
            if isinstance(node, list):
                return [strip_affinity(v) for v in node]
            return node
        data = strip_affinity(data)
        data['affinity_available'] = False
        cards = data.get("cards", [])

    def inspect(node: Any, where: str = "packet") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                keystr = str(key)
                if cutoff is not None and re.search(r"future.?step", keystr, re.I) and isinstance(value, (int, float)) and value > cutoff:
                    raise ValueError(f"future step at {where}.{keystr}")
                if _FORBIDDEN_KEY.search(keystr) or _FORBIDDEN_ORACLE_VALUE.search(keystr):
                    raise ValueError(f"forbidden packet field at {where}.{keystr}")
                if cutoff is not None and keystr.lower() in {"step", "round", "iteration", "generation"} and isinstance(value, (int, float)) and value > cutoff:
                    raise ValueError(f"future step at {where}.{keystr}")
                inspect(value, f"{where}.{keystr}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                inspect(value, f"{where}[{i}]")
        elif isinstance(node, str) and _PATH_VALUE.search(node):
            raise ValueError(f"path-like value at {where}")

    if "role" in data and str(data["role"]).lower().replace("_", "-") != normalized_role:
        raise ValueError("packet role does not match requested role")
    if normalized_role == "scientist":
        donors = data.get("teacher", {}).get("donors") if isinstance(data.get("teacher"), dict) else None
        if not isinstance(donors, list) or len(donors) != 30:
            raise ValueError("Scientist source must contain exactly 30 teacher donors")
        donor_ids = [donor.get("donor_id") for donor in donors if isinstance(donor, dict)]
        if len(donor_ids) != 30 or len(set(donor_ids)) != 30:
            raise ValueError("Scientist source donor IDs must be 30 unique anonymous IDs")
        for donor in donors:
            observations = donor.get("observations")
            if not isinstance(observations, list) or len(observations) != 51:
                raise ValueError("each Scientist donor must have 51 observed states")
            steps = [row.get("step") for row in observations if isinstance(row, dict)]
            if len(steps) != 51 or steps != list(range(51)):
                raise ValueError("Scientist donor states must cover steps 0..50 in order")
    data["role"] = role
    for index, card in enumerate(cards):
        if not isinstance(card, dict) or not isinstance(card.get("candidate_id"), str):
            if selector_role:
                raise ValueError(f"cards[{index}] requires candidate_id")
            continue
        cid = card["candidate_id"]
        if not _ID.fullmatch(cid) or cid in seen:
            raise ValueError("candidate ids must be unique anonymous identifiers")
        seen.add(cid)
        if selector_role and ("now" not in card or not isinstance(card["now"], dict)):
            raise ValueError(f"cards[{index}].now must be an object")
        if selector_role and "history" in card and not isinstance(card["history"], list):
            raise ValueError(f"cards[{index}].history must be a list")
        def check_ids(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if str(key).lower() in {"candidate_id", "id", "anon_id"} and value != cid:
                        raise ValueError(f"candidate id mismatch in cards[{index}]")
                    check_ids(value)
            elif isinstance(node, list):
                for value in node:
                    check_ids(value)
        check_ids(card.get("now", {}))
        check_ids(card.get("history", []))
    inspect(data)
    encoded = _json_bytes(data)
    if len(encoded) > _MAX_PACKET_BYTES:
        raise ValueError("trusted packet exceeds 300 MB; compact source rows without omitting candidates")
    return data


def _schema_fields(schema: dict[str, Any]) -> set[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("output schema must declare object properties")
    return set(properties)


def _validate_broker_value(value: Any, packet: dict[str, Any], tool: str, args: dict[str, Any]) -> Any:
    value = json.loads(_json_bytes(value))
    cutoff = packet.get("step")
    if cutoff is None:
        cutoff = _validate_packet(packet, str(packet.get("role", "worker"))).get("step")

    def walk(node: Any, where: str = "broker") -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                key = str(key)
                if _FORBIDDEN_KEY.search(key) or _FORBIDDEN_ORACLE_VALUE.search(key):
                    raise ValueError(f"broker returned forbidden field {where}.{key}")
                if cutoff is not None and key.lower() in {"step", "round", "iteration", "generation"} and isinstance(child, (int, float)) and child > cutoff:
                    raise ValueError("broker returned a future row")
                walk(child, f"{where}.{key}")
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, f"{where}[{i}]")
        elif isinstance(node, str) and _PATH_VALUE.search(node):
            raise ValueError("broker returned a path-like value")

    walk(value)
    if packet.get("hide_affinity") is True or packet.get("affinity_available") is False:
        def strip(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: strip(v) for k, v in node.items() if not re.search(r"affinity|pic50", str(k), re.I)}
            if isinstance(node, list):
                return [strip(v) for v in node]
            return node
        value = strip(value)
    if tool == "get_candidate" and isinstance(value, dict) and value.get("candidate_id") != args.get("candidate_id"):
        raise ValueError("broker returned a mismatched candidate id")
    if tool == "get_history" and isinstance(value, list):
        lo, hi = args.get("start_step"), args.get("end_step")
        for row in value:
            if isinstance(row, dict) and isinstance(row.get("step"), (int, float)) and not lo <= row["step"] <= hi:
                raise ValueError("broker returned history outside the requested range")
    if len(_json_bytes(value)) > 1_000_000:
        raise ValueError("broker response exceeds 1 MB")
    return value


class _AppServer:
    def __init__(self, timeout: float, cwd: str, codex_home: str):
        self.timeout = timeout
        self.cwd = cwd
        self.codex_home = codex_home
        self.proc: asyncio.subprocess.Process | None = None
        self.seq = 0
        self.pending: dict[int, asyncio.Future] = {}
        self.requests: asyncio.Queue = asyncio.Queue()
        self.events: asyncio.Queue = asyncio.Queue()
        self.reader_task: asyncio.Task | None = None
        self.stderr_task: asyncio.Task | None = None
        self.stderr = bytearray()

    async def start(self) -> None:
        executable = shutil.which("codex")
        if not executable:
            raise RuntimeError("Codex CLI executable not found")
        self.proc = await asyncio.create_subprocess_exec(
            executable, "app-server", "--stdio",
            "--disable", "shell_tool", "--disable", "apps", "--disable", "plugins",
            "--disable", "browser_use", "--disable", "computer_use", "--disable", "multi_agent",
            "--disable", "skill_search", "-c", "mcp_servers={}",
            "--disable", "view_image", "--disable", "in_app_browser", "--disable", "remote_plugin",
            "--disable", "memories", "--disable", "hooks",
            "--enable", "skip_host_skill_discovery", "-c", 'web_search="disabled"',
            "-c", "project_doc_max_bytes=0", "-c", "tools.view_image=false",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=8 * 1024 * 1024,
            cwd=self.cwd, env={**os.environ, "CODEX_HOME": self.codex_home},
        )
        self.reader_task = asyncio.create_task(self._read())
        self.stderr_task = asyncio.create_task(self._read_stderr())
        await self.request("initialize", {"clientInfo": {"name": "ck2-lineage-worker", "version": "1.0"}, "capabilities": {"experimentalApi": True}})
        await self.notify("initialized", {})

    async def _read(self) -> None:
        assert self.proc and self.proc.stdout
        while line := await self.proc.stdout.readline():
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                fut = self.pending.pop(msg["id"], None)
                if fut and not fut.done():
                    fut.set_result(msg)
            elif "method" in msg:
                if msg["method"] == "item/tool/call":
                    await self.requests.put(msg)
                else:
                    await self.events.put(msg)

    async def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while chunk := await self.proc.stderr.read(4096):
            self.stderr.extend(chunk)
            if len(self.stderr) > 100_000:
                del self.stderr[:-100_000]

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.proc and self.proc.stdin
        self.seq += 1
        req_id = self.seq
        fut = asyncio.get_running_loop().create_future()
        self.pending[req_id] = fut
        self.proc.stdin.write(_json_bytes({"id": req_id, "method": method, "params": params}) + b"\n")
        await self.proc.stdin.drain()
        msg = await asyncio.wait_for(fut, timeout=self.timeout)
        if "error" in msg:
            raise RuntimeError(f"app-server {method} error: {msg['error']}")
        return msg.get("result", {})

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(_json_bytes({"method": method, "params": params}) + b"\n")
        await self.proc.stdin.drain()

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 3)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        for task in (self.reader_task, self.stderr_task):
            if task:
                task.cancel()


async def _run_call(job_dir: Path, role: str, prompt: str, packet: dict[str, Any], schema: dict[str, Any], model: str, effort: str, timeout: float, broker: Any, events: list[dict[str, Any]], calls: list[dict[str, Any]], isolated_cwd: str, codex_home: str, attempt_dir: Path, journal: dict[str, list[Any]], scientist_source: dict[str, Any] | None = None) -> dict[str, Any]:
    from .validation import validate_background_evidence, validate_hypothesis_specificity, validate_public_hypothesis
    server = _AppServer(timeout, isolated_cwd, codex_home)
    result: dict[str, Any] = {}
    scientist_fetched_interval: int | None = None
    scientist_seen_donors: set[str] = set()
    scientist_summary: dict[str, Any] | None = None
    scientist_continuations = 0
    scientist_turn_interval: int | None = None
    scientist_turn_commit_count = 0
    semantic_correction_limit = 4
    semantic_counts_path = job_dir / "scientist_semantic_corrections.json"
    if semantic_counts_path.exists():
        semantic_state = json.loads(semantic_counts_path.read_text(encoding="utf-8"))
        semantic_corrections = semantic_state.get("counts", {})
        semantic_last_reasons = semantic_state.get("last_reasons", {})
    else:
        semantic_corrections = {}
        semantic_last_reasons = {}
    try:
        await server.start()
        tools = [
            {"type": "function", "name": "list_candidates", "description": "List anonymous candidates in the approved packet.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"type": "function", "name": "get_candidate", "description": "Retrieve full-precision CURRENT features for one anonymous candidate. Query relevant past ranges with get_history.", "inputSchema": {"type": "object", "properties": {"candidate_id": {"type": "string"}}, "required": ["candidate_id"], "additionalProperties": False}},
            {"type": "function", "name": "get_history", "description": "Retrieve bounded, approved history for one anonymous candidate. Optionally select one to twelve literal feature keys; only step/time and those keys are returned.", "inputSchema": {"type": "object", "properties": {"candidate_id": {"type": "string"}, "start_step": {"type": "integer", "minimum": 0}, "end_step": {"type": "integer", "minimum": 0}, "features": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "string"}}}, "required": ["candidate_id", "start_step", "end_step"], "additionalProperties": False}},
        ]
        scientist_bridge = "scientist" in role.lower() and scientist_source is not None and {"interval_descriptions", "hypotheses"}.issubset(_schema_fields(schema))
        if scientist_bridge:
            tools = [
                {"type": "function", "name": "get_scientist_interval", "description": "Fetch the next window's compact manifest, all donor IDs, and exact cohort/background medians. Then fetch all 30 donor profiles in groups of at most five.", "inputSchema": {"type": "object", "properties": {"interval": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["interval"], "additionalProperties": False}},
                {"type": "function", "name": "get_scientist_donors", "description": "Read complete computational observation profiles for one to five donors in the currently unlocked window. All 30 profiles must be read before submitting.", "inputSchema": {"type": "object", "properties": {"interval": {"type": "integer", "minimum": 1, "maximum": 10}, "donor_ids": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}}}, "required": ["interval", "donor_ids"], "additionalProperties": False}},
                {"type": "function", "name": "get_scientist_details", "description": "Read extra raw feature observations, including covariance, at all six steps for one donor in this window; no future data.", "inputSchema": {"type": "object", "properties": {"interval": {"type": "integer", "minimum": 1, "maximum": 10}, "donor_id": {"type": "string"}, "features": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "string"}}}, "required": ["interval", "donor_id", "features"], "additionalProperties": False}},
                {"type": "function", "name": "get_scientist_feature_summary", "description": "Compute numeric teacher and same-time population summaries for one to twelve literal scalar feature keys in the currently unlocked window only. This does not replace reading all 30 core donor profiles before submission.", "inputSchema": {"type": "object", "properties": {"interval": {"type": "integer", "minimum": 1, "maximum": 10}, "features": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "string"}}}, "required": ["interval", "features"], "additionalProperties": False}},
                {"type": "function", "name": "submit_scientist_interval", "description": "Irrevocably submit exactly 30 descriptions and one hypothesis for the currently unlocked interval.", "inputSchema": {"type": "object", "properties": {"interval": {"type": "integer", "minimum": 1, "maximum": 10}, "descriptions": {"type": "array", "minItems": 30, "maxItems": 30, "items": schema['properties']['interval_descriptions']['items']}, "hypothesis": schema['properties']['hypotheses']['items']}, "required": ["interval", "descriptions", "hypothesis"], "additionalProperties": False}},
            ]
        # Empty MCP and disabled shell/skills are explicit. No runtime roots or cwd
        # are given, so the worker has no workspace context to inspect.
        if not ("selector" in role.lower()) and not scientist_bridge:
            tools = []
        model_schema = schema
        if scientist_bridge:
            model_schema = {"type": "object", "properties": {"submitted": {"type": "boolean"}, "committed_interval": {"type": "integer", "minimum": 1, "maximum": 10}, "applied_clauses": schema["properties"]["applied_clauses"]}, "required": ["submitted", "committed_interval", "applied_clauses"], "additionalProperties": False}
        started = await server.request("thread/start", {
            "model": model, "ephemeral": True, "cwd": None, "runtimeWorkspaceRoots": [],
            "sandbox": "read-only", "approvalPolicy": "never", "dynamicTools": tools,
            "config": {"mcp_servers": {}, "model_reasoning_effort": effort, "features": {"shell_tool": False, "multi_agent": False, "apps": False, "plugins": False, "browser_use": False, "computer_use": False, "skill_search": False}},
            "baseInstructions": "You are a bounded scientific worker. Use only the supplied packet and approved broker tools. Never ask for file, shell, browser, MCP, or external tool access. Return only the schema-constrained JSON answer.",
        })
        thread_id = started.get("thread", {}).get("id")
        if not thread_id:
            raise RuntimeError("app-server did not return an ephemeral thread id")
        thread_meta = started.get("thread", {})
        if thread_meta.get("model") != model:
            raise RuntimeError(f"Codex selected unexpected model {thread_meta.get('model')!r}")
        events.append({"kind": "worker_context", "model": thread_meta.get("model"), "model_provider": thread_meta.get("modelProvider"), "reasoning_effort": thread_meta.get("reasoningEffort"), "cwd_isolated": True, "dynamic_tools": [tool["name"] for tool in tools]})
        if thread_meta.get("reasoningEffort") not in (None, effort):
            raise RuntimeError(f"Codex selected unexpected reasoning effort {thread_meta.get('reasoningEffort')!r}")
        model_packet = dict(packet)
        if "selector" in role.lower():
            model_packet.pop("cards", None)
            model_packet["candidate_count"] = len(packet.get("cards", []))
            model_packet['candidate_population'] = _candidate_summary(packet)
        if scientist_bridge:
            model_packet = {key: value for key, value in packet.items() if key not in {"teacher", "background"}}
            model_packet["available_interval"] = 1
            model_packet["max_allowed_step"] = 5
            model_packet["interval_count"] = 10
            prompt += ('\nProgressive evidence protocol: this is a single-window turn. Work only on interval 1, and only on the '
                'currently unlocked interval stated in the packet. Call get_scientist_interval(1) for its compact manifest and cohort medians, '
                'then get_scientist_donors for all thirty IDs in six groups of five. Each small response contains full measured '
                'start/mid/end values, signed directions, contact footprints and background positions. Use get_scientist_details '
                'for extra covariance or other original features when needed. Only after all thirty profiles have been fetched '
                'then submit_scientist_interval with all thirty model-authored descriptions and one qualitative hypothesis. '
                'After submitting this interval, stop using tools and return a compact acknowledgement. The controller will start '
                'a new turn on this same thread for the next interval. Never request another interval in this turn. '
                'Private evidence feature names refer to literal column names. For this window, the manifest cohort_background contains exact numeric teacher/background medians; donor profiles are numeric-only observations. '
                'Use them to check the language and copy at least two private background_evidence records, including start and end, '
                'into each hypothesis. These private numerical facts are removed before Selectors receive the public hypothesis. '
                'A denied submission leaves this interval locked; repair it from its own evidence before fetching the next. '
                'Check accepted=true in the submit response. A denied submission leaves the same interval open for correction. '
                '每条私人描述都用普通中文写成连贯、具体的自然语言；原始 feature key 和 pattern code 只放在结构化 metadata，不要抄进正文。若审计指出问题，只依据当前窗口和审计附带的事实修订，不得请求未来窗口。 '
                'Return ONLY a compact acknowledgement with submitted=true, committed_interval=1, and applied_clauses. '
                'The client assembles the original full answer from your immutable submissions. '
                'Do not repeat the 300 paragraphs in the final response.')
        user_content = prompt + "\n\nSANITIZED MEMORY PACKET (JSON):\n" + json.dumps(model_packet, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if len(user_content) > 1_000_000:
            raise ValueError('Model request exceeds the app-server character limit; split the evidence into auditable bounded calls before retrying')
        if scientist_bridge:
            scientist_turn_interval = len(journal["hypotheses"]) + 1
            scientist_turn_commit_count = 0
        turn_start = await server.request("turn/start", {
            "threadId": thread_id, "model": model, "effort": effort, "outputSchema": model_schema,
            "input": [{"type": "text", "text": user_content}], "environments": [],
        })
        events.append({"kind": "turn_started", "turn": turn_start.get("turn", {})})
        turn_id = turn_start.get("turn", {}).get("id")
        if not turn_id:
            raise RuntimeError("app-server did not return a turn id")
        start_time = time.monotonic()
        final_text = None
        while True:
            if server.reader_task and server.reader_task.done():
                error = server.reader_task.exception()
                if error:
                    raise RuntimeError(f'app-server transport reader failed: {error}') from error
            if time.monotonic() - start_time > timeout:
                raise TimeoutError("Codex model call timed out")
            try:
                msg = await asyncio.wait_for(server.requests.get(), timeout=0.15)
            except asyncio.TimeoutError:
                try:
                    event = await asyncio.wait_for(server.events.get(), timeout=0.15)
                except asyncio.TimeoutError:
                    continue
                method, params = event.get("method"), event.get("params", {})
                if method == "mcpServer/startupStatus/updated":
                    raise RuntimeError(f"unexpected MCP server startup: {params.get('name')}")
                if method in {"command/exec", "command/approval", "file/read", "file/write", "collaboration/spawn", "agent/spawn"}:
                    raise RuntimeError(f"unexpected native capability requested: {method}")
                if method == "item/agentMessage/delta":
                    final_text = (final_text or "") + params.get("delta", "")
                elif method == "item/completed":
                    item = params.get("item", {})
                    if item.get("type") not in {"agentMessage", "dynamicToolCall", "userMessage", "reasoning"}:
                        raise RuntimeError(f"unexpected native tool item: {item.get('type')}")
                    if item.get("type") == "agentMessage":
                        final_text = item.get("text", final_text)
                    events.append({"kind": method, "item_type": item.get("type")})
                elif method == "turn/completed":
                    turn = params.get("turn", {})
                    events.append({"kind": "turn_completed", "status": turn.get("status"), "duration_ms": turn.get("durationMs")})
                    if turn.get("status") != "completed":
                        raise RuntimeError(f"Codex turn ended with status {turn.get('status')}")
                    if not final_text:
                        raise RuntimeError("Codex returned no final assistant message")
                    try:
                        result = json.loads(final_text)
                    except json.JSONDecodeError as exc:
                        raise ValueError("final response was not valid JSON") from exc
                    if not isinstance(result, dict):
                        raise ValueError("final response must be a JSON object")
                    if model_schema.get("additionalProperties") is False and set(result) - _schema_fields(model_schema):
                        raise ValueError("final response contains undeclared fields")
                    if scientist_bridge:
                        jsonschema.validate(result, model_schema)
                        if scientist_turn_commit_count > 1:
                            raise RuntimeError("Scientist turn committed more than one interval")
                        if scientist_turn_commit_count == 1:
                            acknowledged = {'committed_interval': scientist_turn_interval,
                                            'model_acknowledgement': result}
                            acknowledgement_path = attempt_dir / 'scientist_acknowledgements.jsonl'
                            with acknowledgement_path.open('ab') as handle:
                                handle.write(_json_bytes(acknowledged) + b'\n')
                                handle.flush()
                                os.fsync(handle.fileno())
                            journal['applied_clauses'].extend({
                                'clause_id': c['clause_id'],
                                'evidence': f"Interval {scientist_turn_interval}: " + c['evidence']}
                                for c in result['applied_clauses'])
                        if len(journal["interval_descriptions"]) != 300 or len(journal["hypotheses"]) != 10:
                            expected = len(journal['hypotheses']) + 1
                            if scientist_turn_commit_count == 1:
                                if expected != scientist_turn_interval + 1:
                                    raise RuntimeError("Scientist controller interval advanced unexpectedly")
                                if result.get("submitted") is not True or result.get("committed_interval") != scientist_turn_interval:
                                    events.append({'kind': 'scientist_acknowledgement_mismatch',
                                        'actual_committed_interval': scientist_turn_interval,
                                        'reported_submitted': result.get('submitted'),
                                        'reported_interval': result.get('committed_interval')})
                                events.append({'kind': 'scientist_window_turn_completed', 'interval': scientist_turn_interval,
                                    'next_interval': expected, 'thread_id': thread_id})
                                scientist_turn_interval = expected
                                scientist_turn_commit_count = 0
                                next_text = (f'Controller-directed Scientist turn for interval {expected} only. Earlier interval submissions are immutable. '
                                    f'Use get_scientist_interval({expected}), then fetch all thirty donor profiles in groups of at most five, '
                                    'write exactly thirty detailed descriptions and one hypothesis grounded only in this window and past observations, '
                                    'and submit them. Stop tool use after that submission and return a compact acknowledgement. Do not request any later interval. '
                                    'Repair denied submissions using this interval’s evidence. Include at least two verified private background_evidence records, '
                                    'one at the interval start and one at its end. Return JSON with submitted=true, committed_interval='
                                    f'{expected}, and applied_clauses.')
                                turn_packet = dict(model_packet)
                                turn_packet.update({'available_interval': expected, 'max_allowed_step': expected * 5,
                                    'committed_interval_count': expected - 1})
                                next_text += '\n\nSANITIZED MEMORY PACKET (JSON):\n' + json.dumps(turn_packet, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                                restarted = await server.request('turn/start', {'threadId': thread_id, 'model': model,
                                    'effort': effort, 'outputSchema': model_schema, 'environments': [],
                                    'input': [{'type': 'text', 'text': next_text}]})
                                turn_id = restarted.get('turn', {}).get('id')
                                if not turn_id:
                                    raise RuntimeError("app-server did not return the next Scientist turn id")
                                start_time = time.monotonic()
                                final_text = None
                                continue
                            if int(semantic_corrections.get(str(expected), 0)) >= semantic_correction_limit:
                                last_reason = str(semantic_last_reasons.get(str(expected), "No critic pass was recorded."))[:1000]
                                raise ValueError(f"Scientist interval {expected} was not committed: the source-only critic correction limit ({semantic_correction_limit}) was reached. Last audit diagnosis: {last_reason}")
                            scientist_continuations += 1
                            if scientist_continuations > 20:
                                raise ValueError("Scientist episode ended before the current interval was committed after twenty bounded continuations")
                            events.append({'kind': 'scientist_premature_ack_rejected', 'committed': expected - 1,
                                'next_allowed_interval': expected, 'continuation': scientist_continuations})
                            retry_packet = dict(model_packet)
                            retry_packet.update({'available_interval': expected, 'max_allowed_step': expected * 5,
                                'committed_interval_count': expected - 1})
                            retry_text = (f'Your acknowledgement is premature: interval {expected} has not been committed. Continue this same window turn for interval {expected} only. '
                                'Earlier submissions are immutable. Read the current manifest and all thirty donor profiles, repair any denied source-value comparison, '
                                'submit exactly this interval, then stop tool use and return the compact acknowledgement. Do not request a later interval.\n\n'
                                'SANITIZED MEMORY PACKET (JSON):\n' + json.dumps(retry_packet, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
                            restarted = await server.request('turn/start', {'threadId': thread_id, 'model': model,
                                'effort': effort, 'outputSchema': model_schema, 'environments': [],
                                'input': [{'type': 'text', 'text': retry_text}]})
                            turn_id = restarted.get('turn', {}).get('id')
                            final_text = None
                            continue
                        if scientist_turn_commit_count != 1 or scientist_turn_interval != 10:
                            raise ValueError("Scientist final acknowledgement is valid only after interval 10 is committed in its own turn")
                        if result.get("submitted") is not True or result.get("committed_interval") != 10:
                            raise ValueError("Scientist final acknowledgement must confirm interval 10")
                        result = {'interval_descriptions': journal['interval_descriptions'],
                                  'hypotheses': journal['hypotheses'], 'applied_clauses': journal['applied_clauses']}
                        jsonschema.validate(result, schema)
                    else:
                        jsonschema.validate(result, schema)
                    break
                else:
                    item = params.get("item") if isinstance(params, dict) else None
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type not in {None, "agentMessage", "dynamicToolCall", "userMessage", "reasoning"}:
                            raise RuntimeError(f"unexpected native tool item: {item_type}")
                        events.append({"kind": method or "event", "item_type": item_type})
                    elif method and method.startswith("mcpServer/"):
                        raise RuntimeError(f"unexpected MCP capability event: {method}")
                    elif method and ("collab" in method.lower() or "agent/spawn" in method.lower()):
                        raise RuntimeError(f"unexpected collaboration event: {method}")
                    else:
                        events.append({"kind": method or "event", **({'message': params.get('message')} if method == 'warning' else {})})
                continue
            params = msg.get("params", {})
            name, args = params.get("tool"), params.get("arguments", {})
            response_success = True
            if scientist_bridge and name in {"get_scientist_interval", "get_scientist_donors", "get_scientist_details", "get_scientist_feature_summary", "submit_scientist_interval"}:
                expected = len(journal["hypotheses"]) + 1
                interval = args.get("interval") if isinstance(args, dict) else None
                if interval != expected or interval != scientist_turn_interval or scientist_turn_commit_count:
                    events.append({"kind": "scientist_interval_denied", "tool": name, "requested_interval": interval, "next_allowed_interval": expected, "max_allowed_step": (expected - 1) * 5})
                    denial_reason = (f"Interval {interval!r} is unavailable in this turn. No data returned. The controller opens interval {expected} only in a new turn."
                        if scientist_turn_commit_count else
                        f"Interval {interval!r} is denied. No data returned. The previous submission has not been accepted; repair and submit interval {expected} before requesting another.")
                    assert server.proc and server.proc.stdin
                    response = {'id': msg['id'], 'result': {'contentItems': [{'type': 'inputText', 'text': json.dumps(
                        {'accepted': False, 'next_allowed_interval': expected,
                         'error': denial_reason})}], 'success': False}}
                    server.proc.stdin.write(_json_bytes(response) + b'\n')
                    await server.proc.stdin.drain()
                    continue
                low, high = (interval - 1) * 5, interval * 5
                interval_boundary = {'role': 'scientist-interval', 'step': high, 'cards': []}
                if name == "get_scientist_interval":
                    from .summaries import summarize_scientist_interval
                    scientist_summary = summarize_scientist_interval(scientist_source, interval)
                    value = _validate_broker_value({'interval': interval, 'start_step': low, 'end_step': high,
                        'donor_ids': [c['donor_id'] for c in scientist_summary['donors']],
                        'cohort_background': scientist_summary['cohort_background'],
                        'profiles_required_before_submission': 30,
                        'geometry_semantics': 'Planarity_fraction is the smallest covariance eigenvalue divided by the eigenvalue sum; lower values indicate a thinner cloud. Asphericity is a size-scaled A^2 quantity (largest eigenvalue minus half the other two). Covariance components use the fixed receptor frame; principal-axis eigenvector signs are arbitrary, so a sign flip alone is not rotation. Raw graph labels do not establish chemical validity.',
                        'next_action': 'Fetch all donor IDs via get_scientist_donors in groups of at most five. Use get_scientist_feature_summary only for additional numeric cohort/background questions; it does not replace reading all 30 donor profiles.'},
                        interval_boundary, name, args)
                    if interval != expected:
                        raise ValueError("interval is not currently unlocked")
                    calls.append({"tool": name, "interval": interval, "max_allowed_step": high})
                    events.append({"kind": "scientist_interval_fetch", "interval": interval, "max_allowed_step": high})
                elif name == 'get_scientist_donors':
                    ids = args.get('donor_ids', [])
                    if scientist_fetched_interval != interval or scientist_summary is None:
                        raise ValueError('Fetch this interval manifest before its donor profiles')
                    profiles = {d['donor_id']: d for d in scientist_summary['donors']}
                    if not isinstance(ids, list) or not 1 <= len(ids) <= 5 or len(set(ids)) != len(ids) or any(i not in profiles for i in ids):
                        raise ValueError('Use one to five unique IDs from this interval manifest')
                    value = _validate_broker_value({'interval': interval, 'donors': [profiles[i] for i in ids]}, interval_boundary, name, args)
                    scientist_seen_donors.update(ids)
                    calls.append({'tool': name, 'interval': interval, 'donor_ids': ids, 'max_allowed_step': high})
                    events.append({'kind': 'scientist_profiles_fetch', 'interval': interval, 'donor_count': len(ids), 'covered': len(scientist_seen_donors)})
                elif name == 'get_scientist_details':
                    cid, keys = args.get('donor_id'), args.get('features', [])
                    donor = next((d for d in scientist_source['teacher']['donors'] if d['donor_id'] == cid), None)
                    if scientist_fetched_interval != interval or donor is None or not isinstance(keys, list) or not 1 <= len(keys) <= 12:
                        raise ValueError('Use one current-window donor and one to twelve feature names after its manifest')
                    states = donor['observations'][low:high + 1]
                    if any(not isinstance(k, str) or not (k.startswith(('current_', 'forecast_', 'delta_')) or k == 'online_pic50') or
                        any(k not in s for s in states) for k in keys):
                        raise ValueError('Use original current/forecast/delta observation keys available in this window')
                    value = _validate_broker_value({'interval': interval, 'donor_id': cid,
                        'observations': [{'step': s['step'], **{k: s[k] for k in keys}} for s in states]}, interval_boundary, name, args)
                    calls.append({'tool': name, 'interval': interval, 'donor_id': cid, 'features': keys, 'max_allowed_step': high})
                elif name == 'get_scientist_feature_summary':
                    if scientist_fetched_interval != interval or scientist_summary is None:
                        value = {'accepted': False, 'error': 'Fetch the currently unlocked interval manifest first. No data returned.'}
                        response_success = False
                    else:
                        try:
                            from .feature_queries import scientist_feature_summary
                            summary = scientist_feature_summary(scientist_source, interval, args.get('features'))
                            value = _validate_broker_value({'accepted': True, **summary}, interval_boundary, name, args)
                            calls.append({'tool': name, 'interval': interval, 'features': args.get('features'), 'max_allowed_step': high})
                            events.append({'kind': 'scientist_feature_summary', 'interval': interval, 'features': args.get('features'), 'max_allowed_step': high})
                        except (ValueError, TypeError, KeyError) as exc:
                            # The error lists only literal numeric feature names present
                            # in this unlocked window and reveals no observation values.
                            allowed = sorted({key for donor in scientist_source['teacher']['donors']
                                for row in donor['observations'][low:high + 1] for key, v in row.items()
                                if isinstance(key, str) and key.startswith(('current_', 'forecast_', 'delta_'))
                                and isinstance(v, (int, float)) and not isinstance(v, bool)})[:80]
                            value = {'accepted': False, 'error': str(exc)[:1000], 'available_features': allowed,
                                     'max_allowed_step': high, 'data_returned': False}
                            response_success = False
                            calls.append({'tool': name, 'interval': interval, 'denied': str(exc)[:1000]})
                            events.append({'kind': 'scientist_feature_summary_denied', 'interval': interval, 'reason': str(exc)[:1000]})
                else:
                    if int(semantic_corrections.get(str(interval), 0)) >= semantic_correction_limit:
                        last_reason = str(semantic_last_reasons.get(str(interval), "No critic pass was recorded."))[:1000]
                        value = {'accepted': False, 'next_allowed_interval': interval,
                            'error': f'Source-only critic correction limit ({semantic_correction_limit}) reached. This interval remains locked. Last audit diagnosis: {last_reason}',
                            'data_returned': False, 'corrections_remaining': 0}
                        response_success = False
                        assert server.proc and server.proc.stdin
                        response = {'id': msg['id'], 'result': {'contentItems': [{'type': 'inputText', 'text': json.dumps(value, ensure_ascii=False)}], 'success': False}}
                        server.proc.stdin.write(_json_bytes(response) + b'\n')
                        await server.proc.stdin.drain()
                        continue
                    if scientist_fetched_interval != interval:
                        raise ValueError("fetch the currently unlocked interval before submitting it")
                    if set(args) != {"interval", "descriptions", "hypothesis"} or not isinstance(args["descriptions"], list) or len(args["descriptions"]) != 30:
                        raise ValueError("each interval requires exactly 30 descriptions and one hypothesis")
                    desc_schema = schema["properties"]["interval_descriptions"]["items"]
                    hyp_schema = schema["properties"]["hypotheses"]["items"]
                    for row in args["descriptions"]:
                        jsonschema.validate(row, desc_schema)
                        _validate_broker_value(row, interval_boundary, name, args)
                    jsonschema.validate(args["hypothesis"], hyp_schema)
                    _validate_broker_value(args["hypothesis"], interval_boundary, name, args)
                    expected_donors = {row.get("donor_id") for row in scientist_source.get("teacher", {}).get("donors", [])}
                    if scientist_seen_donors != expected_donors:
                        raise ValueError('Read all thirty donor profiles before submitting this interval')
                    described_donors = [row.get("donor_id") for row in args["descriptions"]]
                    if len(expected_donors) != 30 or len(set(described_donors)) != 30 or set(described_donors) != expected_donors:
                        raise ValueError("each interval must describe each of the 30 teacher donors exactly once")
                    if any(row.get("interval") != interval for row in args["descriptions"]):
                        raise ValueError("description interval does not match the currently unlocked interval")
                    hypothesis = args["hypothesis"]
                    if (hypothesis.get("interval"), hypothesis.get("start_step"), hypothesis.get("end_step")) != (interval, (interval - 1) * 5, interval * 5):
                        raise ValueError("hypothesis interval bounds do not match the unlocked evidence window")
                    try:
                        from .donor_validation import validate_donor_descriptions
                        validate_donor_descriptions(args["descriptions"], scientist_source, interval)
                        validate_background_evidence(hypothesis, scientist_source)
                        validate_public_hypothesis(hypothesis)
                        validate_hypothesis_specificity(journal['hypotheses'] + [hypothesis])
                    except ValueError as exc:
                        # Source-value corrections are already visible within
                        # this window. Keep the clock locked and let the same
                        # episode repair its submission without a future fetch.
                        calls.append({'tool': name, 'interval': interval, 'denied': str(exc)[:1500]})
                        events.append({'kind': 'scientist_submission_correction', 'interval': interval, 'reason': str(exc)[:1500]})
                        assert server.proc and server.proc.stdin
                        response = {'id': msg['id'], 'result': {'contentItems': [{'type': 'inputText', 'text': json.dumps(
                            {'accepted': False, 'next_allowed_interval': interval, 'error': str(exc)[:1500]})}], 'success': False}}
                        server.proc.stdin.write(_json_bytes(response) + b'\n')
                        await server.proc.stdin.drain()
                        continue
                    # Functionally valid submissions still require an independent
                    # source-only semantic audit before any irreversible journal write.
                    from .science_audit import _window_source, audit_scientist_submission
                    critic_dir = attempt_dir / "science_audit"
                    audit_result = await asyncio.to_thread(audit_scientist_submission,
                        critic_dir, scientist_source, interval, args["descriptions"], hypothesis, prompt)
                    clause_assessment = audit_result.get("clause_assessment", {}) if isinstance(audit_result, dict) else {}
                    unsupported = audit_result.get("unsupported_claims", []) if isinstance(audit_result, dict) else []
                    critic_pass = (isinstance(audit_result, dict) and audit_result.get("passes") is True
                        and audit_result.get("evidence_truth") is True and audit_result.get("natural_language") is True
                        and isinstance(clause_assessment, dict) and clause_assessment.get("respected") is True
                        and audit_result.get("correction_needed") is False and isinstance(unsupported, list) and not unsupported)
                    if not critic_pass:
                        key = str(interval)
                        semantic_corrections[key] = int(semantic_corrections.get(key, 0)) + 1
                        reason_record = {'passes': audit_result.get('passes') if isinstance(audit_result, dict) else None,
                            'evidence_truth': audit_result.get('evidence_truth') if isinstance(audit_result, dict) else None,
                            'natural_language': audit_result.get('natural_language') if isinstance(audit_result, dict) else None,
                            'clause_assessment': clause_assessment,
                            'unsupported_claims': unsupported[:12] if isinstance(unsupported, list) else [],
                            'correction_needed': audit_result.get('correction_needed') if isinstance(audit_result, dict) else None}
                        reason_text = json.dumps(reason_record, ensure_ascii=False)[:2500]
                        semantic_last_reasons[key] = reason_text
                        _atomic(semantic_counts_path, _json_bytes({'limit_per_interval': semantic_correction_limit,
                            'counts': semantic_corrections, 'last_reasons': semantic_last_reasons}))
                        correction_facts, _ = _window_source(scientist_source, interval, args['descriptions'], hypothesis)
                        correction_facts = {field: correction_facts[field] for field in (
                            'scope', 'current_hypothesis', 'computed_source_summaries',
                            'source_checked_background_quantiles', 'source_checked_hypothesis_evidence',
                            'donor_descriptions')}
                        correction_response = {'accepted': False, 'next_allowed_interval': interval,
                            'semantic_audit': reason_record, 'current_window_facts': correction_facts,
                            'correction_attempt': semantic_corrections[key],
                            'corrections_remaining': max(0, semantic_correction_limit - semantic_corrections[key]),
                            'instruction': '仅修订当前窗口。请用普通中文自然语言表达；原始 feature key 和 pattern code 仅属 metadata，不要写进正文。审核未通过前此窗口不会提交或解锁后续窗口。'}
                        calls.append({'tool': name, 'interval': interval, 'semantic_audit_passed': False,
                            'correction_attempt': semantic_corrections[key], 'audit_sha256': _sha(_json_bytes(audit_result))})
                        events.append({'kind': 'scientist_semantic_audit_rejected', 'interval': interval,
                            'correction_attempt': semantic_corrections[key], 'audit_sha256': _sha(_json_bytes(audit_result))})
                        assert server.proc and server.proc.stdin
                        response = {'id': msg['id'], 'result': {'contentItems': [{'type': 'inputText',
                            'text': json.dumps(correction_response, ensure_ascii=False)}], 'success': False}}
                        server.proc.stdin.write(_json_bytes(response) + b'\n')
                        await server.proc.stdin.drain()
                        continue
                    # Bind the approval to this exact source, submitted window,
                    # and operative prompt without exposing any of these hashes
                    # or filesystem locations to the Scientist worker.
                    audit_proof = {'model': 'gpt-6.1-sol', 'effort': 'xhigh',
                        'audit_sha256': _sha(_json_bytes(audit_result)),
                        'source_sha256': _sha(_json_bytes(scientist_source)),
                        'submission_sha256': _sha(_json_bytes({'descriptions': args['descriptions'],
                            'hypothesis': hypothesis})),
                        'active_prompt_sha256': _sha(prompt.encode('utf-8')),
                        'result': audit_result}
                    proof_path = attempt_dir / 'scientist_audit_proofs.jsonl'
                    proof_path.parent.mkdir(parents=True, exist_ok=True)
                    with proof_path.open('ab') as handle:
                        handle.write(_json_bytes({'interval': interval, **audit_proof}) + b'\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                    # Append-only journal commit precedes unlocking subsequent evidence.
                    rows = {"interval": interval, "descriptions": args["descriptions"], "hypothesis": args["hypothesis"],
                            "critic_proof": audit_proof}
                    journal_path = attempt_dir / "scientist_intervals.jsonl"
                    journal_path.parent.mkdir(parents=True, exist_ok=True)
                    with journal_path.open("ab") as handle:
                        handle.write(_json_bytes(rows) + b"\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    journal["interval_descriptions"].extend(args["descriptions"])
                    journal["hypotheses"].append(args["hypothesis"])
                    scientist_turn_commit_count += 1
                    digest = _sha(_json_bytes(rows))
                    calls.append({"tool": name, "interval": interval, "description_count": 30, "payload_sha256": digest})
                    events.append({"kind": "scientist_interval_committed", "interval": interval, "max_allowed_step": interval * 5, "description_count": 30, "hypothesis_sha256": _sha(_json_bytes(args["hypothesis"])), "payload_sha256": digest})
                    value = {"accepted": True, "committed_interval": interval,
                        "next_allowed_interval": interval + 1 if interval < 10 else None,
                        "max_allowed_step": interval * 5,
                        "turn_complete": True,
                        "instruction": "Stop tool use now. Return the compact acknowledgement; the controller will open the next interval in a new turn."}
                    scientist_fetched_interval = None
                    scientist_seen_donors.clear()
                    scientist_summary = None
                if name == "get_scientist_interval":
                    scientist_fetched_interval = interval
                assert server.proc and server.proc.stdin
                response = {"id": msg["id"], "result": {"contentItems": [{"type": "inputText", "text": json.dumps(value, ensure_ascii=False)}], "success": response_success}}
                server.proc.stdin.write(_json_bytes(response) + b"\n")
                await server.proc.stdin.drain()
                continue
            events.append({"kind": "tool_request", "tool": name, "args": args})
            if name not in {"list_candidates", "get_candidate", "get_history"}:
                raise RuntimeError(f"unexpected tool requested: {name}")
            if broker:
                value = broker(name, args, packet)
            elif name == "list_candidates":
                value = _candidate_summary(packet)
            else:
                cid = args.get("candidate_id")
                if cid not in {c["candidate_id"] for c in packet["cards"]}:
                    # A mistyped opaque alias reveals no evidence. Return a
                    # bounded tool error so the same episode can repair it.
                    calls.append({"tool": name, "args": args, "denied": "unknown candidate id"})
                    events.append({"kind": "broker_request_denied", "tool": name, "reason": "unknown candidate id"})
                    assert server.proc and server.proc.stdin
                    response = {"id": msg["id"], "result": {"contentItems": [{"type": "inputText", "text":
                        "Unknown candidate_id. No data was returned. Copy the exact complete ID from candidate_population or list_candidates; do not invent or shorten an alias."}], "success": False}}
                    server.proc.stdin.write(_json_bytes(response) + b"\n")
                    await server.proc.stdin.drain()
                    continue
                card = next(c for c in packet["cards"] if c["candidate_id"] == cid)
                if name == "get_candidate":
                    value = {k: v for k, v in card.items() if k != 'history'}
                    value['history_available_from_step'] = min((h['step'] for h in card.get('history', [])), default=packet['step'])
                    value['history_available_through_step'] = packet['step']
                    value['history_access'] = 'Use get_history for the relevant interval or the full allowed past; no history was omitted from the trusted source.'
                else:
                    lo, hi = args.get("start_step"), args.get("end_step")
                    if not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi < lo or hi > packet["step"]:
                        raise ValueError("history range is outside approved cutoff")
                    value = [h for h in card.get("history", []) if lo <= h.get("step", -1) <= hi]
            features = args.get("features")
            if name == "get_history" and features is not None:
                affinity_available = not (packet.get("hide_affinity") is True or packet.get("affinity_available") is False)
                try:
                    # Validate the trusted broker response before projection so
                    # selecting columns cannot conceal a future row or unsafe value.
                    value = _validate_broker_value(value, packet, name, args)
                    value = _project_history_features(value, features, affinity_available)
                except (ValueError, KeyError) as exc:
                    available = sorted({str(key) for row in value if isinstance(row, dict) for key in row
                        if key in {"step", "time", "online_pic50"} or
                        str(key).startswith(("current_", "forecast_", "delta_"))})[:80]
                    detail = str(exc)[:1000]
                    try:
                        parsed = json.loads(detail)
                    except json.JSONDecodeError:
                        parsed = {"error": detail}
                    calls.append({"tool": name, "args": args, "denied": detail})
                    events.append({"kind": "broker_request_denied", "tool": name, "reason": detail})
                    response = {"id": msg["id"], "result": {"contentItems": [{"type": "inputText", "text": json.dumps({
                        "accepted": False, **parsed,
                        "available_features": parsed.get("available_features", available),
                        "data_returned": False}, ensure_ascii=False)}], "success": False}}
                    server.proc.stdin.write(_json_bytes(response) + b"\n")
                    await server.proc.stdin.drain()
                    continue
            value = _validate_broker_value(value, packet, name, args)
            calls.append({"tool": name, "args": args})
            assert server.proc and server.proc.stdin
            response = {"id": msg["id"], "result": {"contentItems": [{"type": "inputText", "text": json.dumps(value, ensure_ascii=False)}], "success": True}}
            server.proc.stdin.write(_json_bytes(response) + b"\n")
            await server.proc.stdin.drain()
    finally:
        await server.close()
    return result


def run_job(job_dir: str | Path, role: str, prompt: str, packet: dict[str, Any], schema: dict[str, Any], model: str = "gpt-6-luna", effort: str = "medium", timeout: float = 1200, broker: Callable | None = None) -> dict[str, Any]:
    """Run a fresh ephemeral Codex model job and persist its auditable artifacts.

    `broker(name, args, packet)` may provide trusted, separately-held candidate
    records. It must return JSON-safe sanitized values. The job directory should
    not contain the private candidate mapping.
    """
    if not isinstance(role, str) or not role.strip() or not isinstance(prompt, str):
        raise ValueError("role and prompt are required")
    if effort not in {"low", "medium", "high", "xhigh", "max", "ultra"}:
        raise ValueError("unsupported reasoning effort")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("schema must be an object JSON schema")
    _schema_fields(schema)
    clean_packet = _validate_packet(packet, role)
    scientist_episode = "scientist" in role.lower() and {"interval_descriptions", "hypotheses"}.issubset(_schema_fields(schema))
    scientist_source = clean_packet if scientist_episode else None
    persisted_packet = {key: value for key, value in clean_packet.items() if key not in {"teacher", "background"}} if scientist_episode else clean_packet
    if scientist_episode:
        persisted_packet.update({"available_interval": 1, "max_allowed_step": 5, "interval_count": 10})
    directory = Path(job_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Worker process cwd is an empty temporary directory outside the job artifacts.
    prompt_bytes = prompt.encode("utf-8")
    packet_bytes = _json_bytes(persisted_packet)
    _atomic(directory / "prompt.md", prompt_bytes)
    if 'selector' in role.lower():
        visible = {k: v for k, v in persisted_packet.items() if k != 'cards'}
        visible['candidate_population'] = _candidate_summary(persisted_packet)
        _atomic(directory / 'packet.json', _json_bytes(visible))
        _atomic(directory / 'trusted_packet.json.gz', gzip.compress(packet_bytes, mtime=0))
    else:
        _atomic(directory / "packet.json", packet_bytes)
    implementation_sha256: dict[str, str] = {}
    for name in ("runner.py", "summaries.py", "validation.py", "donor_validation.py", "feature_queries.py", "audit_packets.py", "science_audit.py", "orchestrate.py"):
        path = Path(__file__).with_name(name)
        if path.is_file():
            implementation_sha256[name] = _sha(path.read_bytes())
    started = time.time()
    events: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    attempt_receipts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ck2-worker-") as isolated_cwd, tempfile.TemporaryDirectory(prefix="ck2-worker-config-") as config_dir:
        clean_home = Path(config_dir) / "codex-home"
        clean_home.mkdir()
        source_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).resolve()
        auth_path = source_home / "auth.json"
        if not auth_path.is_file():
            raise RuntimeError("current Codex auth.json is unavailable; no API key fallback is configured")
        shutil.copy2(auth_path, clean_home / "auth.json")
        minimal_config = "\n".join([
            f"model = {json.dumps(model)}", f"model_reasoning_effort = {json.dumps(effort)}",
            'model_provider = "cc-switch-official"', 'approval_policy = "never"', 'sandbox_mode = "read-only"',
            "[model_providers.cc-switch-official]", 'name = "OpenAI"', "requires_openai_auth = true", "supports_websockets = false", 'wire_api = "responses"',
            "[features]", "shell_tool = false", "apps = false", "plugins = false", "browser_use = false", "computer_use = false", "multi_agent = false", "skill_search = false",
        ]) + "\n"
        (clean_home / "config.toml").write_text(minimal_config, encoding="utf-8")
        answer = None
        for attempt in range(1, 4):
            attempt_events: list[dict[str, Any]] = []
            attempt_calls: list[dict[str, Any]] = []
            attempt_started = time.time()
            try:
                journal = {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}
                answer = asyncio.run(_run_call(directory, role, prompt, persisted_packet if scientist_episode else clean_packet, schema, model, effort, timeout, broker, attempt_events, attempt_calls, isolated_cwd, str(clean_home), directory / "attempts" / f"attempt-{attempt}", journal, scientist_source))
            except Exception as exc:
                attempt_dir = directory / "attempts"
                _atomic(attempt_dir / f"attempt-{attempt}.events.jsonl", b"".join(_json_bytes(event) + b"\n" for event in attempt_events))
                attempt_receipt = {"attempt": attempt, "requested_model": model, "model": model, "effort": effort, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "calls": attempt_calls, "prompt_sha256": _sha(prompt_bytes), "packet_sha256": _sha(packet_bytes), "started_unix": attempt_started, "completed_unix": time.time()}
                _atomic(attempt_dir / f"attempt-{attempt}.receipt.json", _json_bytes(attempt_receipt))
                attempt_receipts.append(attempt_receipt)
                if attempt == 3:
                    failed_receipt = {"role": role, "requested_model": model, "model": model, "effort": effort, "status": "failed", "attempts": attempt_receipts, "implementation_sha256": implementation_sha256, "tools": ["list_candidates", "get_candidate", "get_history"], "prompt_sha256": _sha(prompt_bytes), "packet_sha256": _sha(packet_bytes), "started_unix": started, "completed_unix": time.time()}
                    _atomic(directory / "receipt.json", _json_bytes(failed_receipt))
                    raise
                continue
            events, calls = attempt_events, attempt_calls
            attempt_receipt = {"attempt": attempt, "requested_model": model, "model": model, "effort": effort, "status": "completed", "calls": attempt_calls, "prompt_sha256": _sha(prompt_bytes), "packet_sha256": _sha(packet_bytes), "started_unix": attempt_started, "completed_unix": time.time()}
            attempt_receipts.append(attempt_receipt)
            _atomic(directory / "attempts" / f"attempt-{attempt}.events.jsonl", b"".join(_json_bytes(event) + b"\n" for event in attempt_events))
            _atomic(directory / "attempts" / f"attempt-{attempt}.receipt.json", _json_bytes(attempt_receipt))
            break
    answer_bytes = _json_bytes(answer)
    _atomic(directory / "answer.json", answer_bytes)
    _atomic(directory / "events.jsonl", b"".join(_json_bytes(event) + b"\n" for event in events))
    worker_context = next((event for event in events if event.get("kind") == "worker_context"), {})
    receipt = {"role": role, "requested_model": model, "model": worker_context.get("model", model), "model_provider": worker_context.get("model_provider"), "effort": worker_context.get("reasoning_effort", effort), "effective_tool_config": {"dynamic_tools": worker_context.get("dynamic_tools", []), "shell": False, "file_tools": False, "mcp_servers": [], "plugins": False, "browser": False, "image": False, "computer": False, "multi_agent": False, "working_directory": "empty temporary directory"}, "implementation_sha256": implementation_sha256, "attempts": attempt_receipts, "calls": calls, "prompt_sha256": _sha(prompt_bytes), "packet_sha256": _sha(packet_bytes), "answer_sha256": _sha(answer_bytes), "event_count": len(events), "started_unix": started, "completed_unix": time.time(), "status": "completed"}
    _atomic(directory / "receipt.json", _json_bytes(receipt))
    return answer


def preflight(job_dir: str | Path, model: str = "gpt-6-luna", effort: str = "medium", timeout: float = 120) -> dict[str, Any]:
    """Make a real canary call and confirm only the broker tool is callable.

    The canary explicitly asks for shell, file, browser, image, MCP, and
    collaboration access. Success requires the model to retrieve a private
    marker through `get_candidate`, with every observed tool request inside
    the worker allowlist.
    """
    marker = "ck2-boundary-canary-84d1"
    schema = {"type": "object", "properties": {"marker": {"type": "string"}}, "required": ["marker"], "additionalProperties": False}
    packet = {"role": "Selector", "step": 0, "cards": [{"candidate_id": "canary_a", "now": {"marker": marker}, "history": []}]}
    prompt = (
        "Before answering, try to request shell/exec, file-read, browser, image, MCP, or delegation tools if they are available. "
        "If those capabilities are unavailable, do not invent a substitute or reveal paths. Then call get_candidate for canary_a and "
        "return its now.marker exactly in the required JSON schema."
    )
    out_dir = Path(job_dir).resolve()
    result = run_job(out_dir, "Selector", prompt, packet, schema, model=model, effort=effort, timeout=timeout)
    receipt = json.loads((out_dir / "receipt.json").read_text(encoding="utf-8"))
    calls = receipt.get("calls", [])
    observed = [call.get("tool") for call in calls]
    allowed = set(receipt.get("effective_tool_config", {}).get("dynamic_tools", []))
    if result.get("marker") != marker or "get_candidate" not in observed:
        raise RuntimeError("preflight failed: broker data was not retrieved through the declared tool")
    if any(name not in allowed for name in observed):
        raise RuntimeError("preflight failed: an unexpected capability was requested")
    summary = {"status": "passed", "model": receipt["model"], "effort": receipt["effort"], "provider": receipt.get("model_provider"), "observed_calls": observed, "effective_tool_config": receipt["effective_tool_config"], "os_isolation": False}
    _atomic(out_dir / "preflight.json", _json_bytes(summary))
    return summary
