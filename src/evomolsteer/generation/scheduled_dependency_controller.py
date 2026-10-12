"""Opt-in scheduled M2 controller; inherits the audited no-SMC endpoint VJP."""
import argparse
import sys
from pathlib import Path

from .dependency_controller import DependencyExtension
from .scheduled_endpoint_reward import ScheduledEndpointReward
from ..io import digest
from ..temporal_guidance.policy import canonical_hash, validate_policy


class ScheduledDependencyExtension(DependencyExtension):
    def prepare(self, opt, out):
        super().prepare(opt, out)
        policy = validate_policy(self.program["temporal_policy"], self.reference["window"],
                                 self.reference["times"], opt.steps)
        if canonical_hash(policy) != self.program.get("temporal_policy_sha256"):
            raise ValueError("Temporal schedule digest mismatch")
        if self.program["reward_view"] != "endpoint_pointcloud":
            raise ValueError("This adapter preserves the M2 analytic point-cloud family")

    def configure(self, model, opt, out):
        super().configure(model, opt, out)
        self.reward = ScheduledEndpointReward(self.program, self.reference)

    def describe(self):
        return {**super().describe(), "generation_interface": "scheduled_m2_endpoint_v1",
                "temporal_adapter_sha256": digest(__file__),
                "temporal_reward_sha256": digest(Path(__file__).with_name("scheduled_endpoint_reward.py")),
                "temporal_policy_sha256": self.program["temporal_policy_sha256"],
                "temporal_policy": self.program["temporal_policy"],
                "reward_semantics": "Static / same-score-node historical endpoint forecast scalar mixture; no coordinate averaging",
                "ratio_budget_semantics": "Equal integral of requested RMS ratios; realized displacements are measured separately"}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--flowr-root", required=True)
    parser.add_argument("--allow-execution", action="store_true")
    args, remaining = parser.parse_known_args()
    if not args.allow_execution:
        raise ValueError("Inference is opt-in: explicitly provide --allow-execution")
    root = Path(args.flowr_root).resolve()
    if not (root / "flowr/models/fm_pocket.py").is_file():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):
        raise ValueError("FLOWR import does not match the declared program path")
    from . import controller
    sys.argv = [sys.argv[0]] + remaining
    controller.main(extension=ScheduledDependencyExtension())
