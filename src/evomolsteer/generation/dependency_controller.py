"""Isolated ablation adapter; historical R26 uses the frozen v2 path unchanged."""
import argparse
import sys
from pathlib import Path
from . import flowcompat_controller, flowcompat_v2_controller
from .flowcompat_v2_controller import FlowCompatibilityV2Extension
from .endpoint_controller import EndpointCoordinateExtension
from .module_sensitivity import DecoderSensitivity
from ..io import digest


class DependencyExtension(FlowCompatibilityV2Extension):
    def configure(self, model, opt, out):
        if self.program['reward_view'] != 'endpoint_structure_field':
            return super().configure(model, opt, out)
        EndpointCoordinateExtension.configure(self, model, opt, out)
        self.previous_gradient = None
        self.module_sha = digest(Path(flowcompat_controller.__file__))
        self.v2_controller_sha = digest(Path(flowcompat_v2_controller.__file__))
        self.sensitivity = DecoderSensitivity(model)

    def describe(self):
        result = super().describe()
        result['generation_interface'] = 'steer_dependency_ablation_v1'
        if self.program['reward_view'] == 'endpoint_structure_field':
            result.update(evidence_domain=None, execution_domain=self.reward.window,
                reference_origin='Receptor/bound ligand; no generated Steer coordinates or labels',
                conditional_gradient='Forecast-based ligand assignment detached; contact field differentiable',
                support_semantics='Execution window inherited from matched R26 protocol; no learned temporal support')
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
    controller.main(extension=DependencyExtension())
