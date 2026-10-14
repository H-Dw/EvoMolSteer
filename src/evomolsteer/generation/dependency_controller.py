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
        if self.program['reward_view'] not in ('endpoint_structure_field', 'endpoint_static_hybrid',
                'endpoint_regional_pointcloud', 'endpoint_temporal_pointcloud', 'endpoint_teacher_region'):
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
        elif self.program['reward_view'] == 'endpoint_static_hybrid':
            result.update(execution_domain=self.reward.window,
                reference_origin='Original bound pose plus fixed selected Steer forecast teachers',
                conditional_gradient='Forecast correspondence, neighbor selection and priors detached; scalar mixture differentiated',
                support_semantics='Proposed static mixture and optional phase contact; no fitted temporal advantage claim')
        elif self.program['reward_view'] == 'endpoint_temporal_pointcloud':
            result.update(reference_origin='Frozen last-learning-node forecasts plus observed-score-node historical forecast teachers',
                temporal_teacher_blend=self.program['temporal_teacher_blend'],
                scientific_teacher_sources=self.reference.get('scientific_sources'),
                conditional_gradient='Both analytic losses use one differentiated FLOWR endpoint; assignments and priors detached',
                support_semantics='Historical forecast teacher at the current score node, not actual terminal structure or trajectory velocity')
        elif self.program['reward_view'] == 'endpoint_regional_pointcloud':
            result.update(reference_origin='Frozen intact teachers with detached coordinate-region attention',
                conditional_gradient='Endpoint coordinate matching with fixed anchor membership and positive surrounding weight',
                empirical_temporal_attention='coordinate_region_node_salience' in self.program)
        elif self.program['reward_view'] == 'endpoint_teacher_region':
            result.update(reference_origin='Terminal-ranked true ancestors with Agent-selected teacher regions',
                conditional_gradient='Teacher-region weights, assignments and priors detached; zero outside-region endpoint loss',
                source_learning_window=self.reference.get('source_learning_window'),
                regional_note='FLOWR VJP can still transmit coupled motion to other current-state coordinates')
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
