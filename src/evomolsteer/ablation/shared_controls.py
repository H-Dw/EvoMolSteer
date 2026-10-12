"""Immutable paired controls and numerical-program reuse for resumed studies.

The original scientific activation stays frozen. This separate execution
amendment changes only the sample bank used by unfinished comparisons.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.continuous.m2_skill_api import effective_signature
from evomolsteer.io import digest, read_json


UNCHANGED_M2 = "本次 Agent 输出未改变执行函数"


def batch_key(batches):
    return ",".join(map(str, batches))


def bound_path(repo, relative):
    path = (Path(repo) / relative).resolve()
    if not path.is_relative_to(Path(repo).resolve()):
        raise ValueError("Control-bank path escapes repository")
    return path


def verify_attempts(frame, batches, batch_size):
    expected = {(b, slot) for b in batches for slot in range(batch_size)}
    if frame.duplicated(["batch", "slot"]).any() or set(map(tuple, frame[["batch", "slot"]].to_numpy())) != expected:
        raise ValueError("Control-bank all-attempt coverage mismatch")
    if not np.isfinite(pd.to_numeric(frame.pic50_on_rescore, errors="coerce")).all():
        raise ValueError("Control-bank missing affinity scores")
    if not {"valid_connected", "pb_fast_pass", "energy_status", "mmff_relief_per_heavy"} <= set(frame.columns):
        raise ValueError("Control-bank quality/failure records missing")


def verify_initial_states(result, controls, batches):
    for b in batches:
        key = f"gradient/batch_{b:03d}"
        if result["initial_state_signatures"].get(key) != controls["initial_state_signatures"][key]:
            raise ValueError("Candidate and M2 initial state mismatch")


class SharedControls:
    def __init__(self, repo, policy_path, protocol_sha256, state, prior_seed_mode):
        self.repo = Path(repo).resolve()
        self.policy_path = bound_path(self.repo, policy_path)
        policy = read_json(self.policy_path)
        if policy.get("schema_version") != "shared-controls-1.0" or policy.get("scope") != "all_unfinished_evaluations":
            raise ValueError("Unsupported shared-control execution policy")
        if policy["protocol_sha256"] != protocol_sha256:
            raise ValueError("Shared-control amendment is not bound to activation")
        self.policy_sha256 = digest(self.policy_path)
        if state.get("shared_control_policy_sha256", self.policy_sha256) != self.policy_sha256:
            raise ValueError("Shared-control policy changed after resume")
        manifest_path = bound_path(self.repo, policy["manifest"])
        if digest(manifest_path) != policy["manifest_sha256"]:
            raise ValueError("Control-bank manifest changed")
        self.manifest = read_json(manifest_path)
        m = self.manifest
        self.batches = list(m["batches"])
        if len(set(self.batches)) != len(self.batches) or m["batch_size"] * len(self.batches) != 400:
            raise ValueError("Exactly 400 fixed attempts per arm required")
        if m["prior_seed_mode"] != prior_seed_mode or prior_seed_mode != "indexed_batch":
            raise ValueError("Shared controls require stable batch-indexed RNG")
        if m["master_seed"] != 42 or m["steps"] != 100:
            raise ValueError("Control-bank inference settings changed")
        for relative, sha in {**m["files"], **m["generator_sources"]}.items():
            if digest(bound_path(self.repo, relative)) != sha:
                raise ValueError("Control-bank immutable file changed: " + relative)
        if effective_signature(read_json(bound_path(self.repo, m["program"]))) != m["effective_signature"]:
            raise ValueError("Frozen M2 numerical program changed")
        self.result = m["result"]
        if self.result["effective_signature"] != m["effective_signature"] or self.result["prior_seed_mode"] != prior_seed_mode:
            raise ValueError("Control-bank result identity mismatch")
        for arm in ("gradient", "unguided"):
            verify_attempts(pd.read_csv(bound_path(self.repo, self.result["outcomes"][arm]["metrics"])), self.batches, m["batch_size"])
            for b in self.batches:
                sig = self.result["initial_state_signatures"][f"{arm}/batch_{b:03d}"]
                if sig != self.result["initial_state_signatures"][f"gradient/batch_{b:03d}"]:
                    raise ValueError("Control-bank arms have unmatched initial states")
        self.summary = {"policy_sha256": self.policy_sha256, "manifest_sha256": policy["manifest_sha256"],
            "cohort": self.result["name"], "batches": self.batches, "attempts_per_arm": 400,
            "independent_confirmation": False, "sample_reuse_not_additional_observations": True,
            "scope": policy["scope"]}

    def measurement(self, owner, path, reference, signature):
        if effective_signature(read_json(path)) != signature:
            raise ValueError("Candidate numerical signature mismatch")
        if read_json(path)["reference_sha256"] != digest(reference):
            raise ValueError("Candidate numerical reference mismatch")
        key = signature + ":" + batch_key(self.batches)
        cached = owner.state["measured"].get(key)
        if signature == self.manifest["effective_signature"]:
            result, action = self.result, "reuse_M2"
        elif cached is not None:
            result, action = cached, "reuse_numerical_program"
        else:
            owner._budget()
            result = owner.infer(path, signature, lane="v3", reference_path=reference, batches=self.batches)
            action = "generate_new_numerical_program"
        if result["effective_signature"] != signature or result["prior_seed_mode"] != self.manifest["prior_seed_mode"]:
            raise ValueError("Cached numerical program identity mismatch")
        verify_initial_states(result, self.result, self.batches)
        for arm in result["outcomes"]:
            verify_attempts(owner.frames(result, arm), self.batches, self.manifest["batch_size"])
        owner.state["measured"][key] = result
        receipt = {"signature": signature, "cohort": result["name"], "batches": self.batches,
            "action": action, "note": UNCHANGED_M2 if action == "reuse_M2" else None,
            "reference_sha256": digest(reference), "paired_initial_states_verified": True,
            "attempts": 400, "independent_confirmation": False}
        prior = owner.state.setdefault("numerical_reuse", {}).get(key)
        # Preserve proof that a new numerical program was generated only once.
        if prior and prior["action"] == "generate_new_numerical_program":
            receipt["first_action"] = prior["action"]
        elif prior and "first_action" in prior:
            receipt["first_action"] = prior["first_action"]
        owner.state["numerical_reuse"][key] = receipt
        if prior != receipt:
            owner.save()
        return result

    def comparison(self, full, drop):
        f = sorted({r["signature"] for r in full["rows"]})
        d = sorted({r["signature"] for r in drop["rows"]})
        unchanged = f == d == [self.manifest["effective_signature"]]
        return {"shared_control_bank": self.summary, "full_numerical_signatures": f,
            "drop_numerical_signatures": d, "both_execute_M2": unchanged,
            "note": UNCHANGED_M2 if unchanged else None, "independent_confirmation": False}
