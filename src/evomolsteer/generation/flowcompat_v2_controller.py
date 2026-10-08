"""Separate v2 adapter: natural branch mixtures and raw scalar derivative audit.

Native generation and the frozen v1 controller stay unchanged. Trace correction
does not feed back into inference or add model calls/backward operations.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

from .branch_mixture_reward import BranchMixtureReward
from .flowcompat_controller import FlowCompatibilityExtension
from ..io import clean, digest


def raw_reward_response(raw_gradient, native_coords, result_coords, mask):
    """Realized coordinate increment paired with the evaluated scalar's VJP."""
    increment = result_coords - native_coords
    n = mask.sum(1).clamp_min(1)
    return {'first_order_reward_change': (raw_gradient * increment).sum((1, 2)),
            'flowcompat_raw_gradient_rms_native': (raw_gradient.square().sum((1, 2)) / n).sqrt(),
            'flowcompat_realized_injection_rms_native': (increment.square().sum((1, 2)) / n).sqrt()}


class FlowCompatibilityV2Extension(FlowCompatibilityExtension):
    def configure(self, model, opt, out):
        original = self.program
        if original['reward_view'] == 'endpoint_branch_mixture':
            self.program = copy.deepcopy(original)
            self.program['reward_view'] = 'endpoint_pointcloud'
            try:
                super().configure(model, opt, out)
            finally:
                self.program = original
            self.reward = BranchMixtureReward(original, self.reference)
        else:
            super().configure(model, opt, out)
        self.v2_controller_sha = digest(Path(__file__))

    def describe(self):
        return {**super().describe(), 'v2_controller_sha256': self.v2_controller_sha,
                'generation_interface': 'flowcompat_v2',
                'reward_view': self.program['reward_view'],
                'reward_derivative_trace': 'Raw evaluated scalar VJP dot realized injection; preconditioned inner product separate'}

    def after_native(self, curr):
        # Capture before the frozen parent optionally replaces g with P g.
        raw_gradient = self.cached[0].detach().clone() if self.cached is not None else None
        native_coords = curr['coords'].detach().clone() if raw_gradient is not None else None
        result = super().after_native(curr)
        path = self.model._lineage.path / 'guidance_trace.jsonl'
        lines = path.read_text().splitlines()
        row = json.loads(lines[-1])
        row['v2_controller_sha256'] = self.v2_controller_sha
        row['generation_interface'] = 'flowcompat_v2'
        if raw_gradient is not None:
            row['control_inner_product'] = row['first_order_reward_change']
            response = raw_reward_response(raw_gradient, native_coords, result['coords'].detach(), curr['mask'].bool())
            row.update({key: value.cpu().tolist() for key, value in response.items()})
        lines[-1] = json.dumps(clean(row), allow_nan=False)
        path.write_text('\n'.join(lines) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--flowr-root', required=True)
    args, remaining = parser.parse_known_args()
    root = Path(args.flowr_root).resolve()
    if not (root / 'flowr/models/fm_pocket.py').is_file():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    import flowr
    if not Path(flowr.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Wrong FLOWR import')
    from . import controller
    sys.argv = [sys.argv[0]] + remaining
    controller.main(extension=FlowCompatibilityV2Extension())


if __name__ == '__main__':
    main()
