"""Trace correction cannot change a controller's coordinate arithmetic."""
import json
from types import SimpleNamespace

import pytest
import torch

from evomolsteer.generation.flowcompat_control import control_geometry
from evomolsteer.generation.flowcompat_controller import FlowCompatibilityExtension
from evomolsteer.generation.flowcompat_v2_controller import FlowCompatibilityV2Extension, raw_reward_response


def test_first_order_uses_raw_gradient_not_preconditioned_direction():
    raw = torch.tensor([[[1., 2., 0.]]], dtype=torch.float64)
    flow = torch.tensor([[[1., 0., 0.]]], dtype=torch.float64)
    mask = torch.ones((1, 1), dtype=torch.bool)
    adjusted, _, _ = control_geometry(raw, flow, mask, .2, [0., .5], {'parallel_component_scale': .25})
    native = torch.zeros_like(raw)
    result = native + .1 * adjusted
    response = raw_reward_response(raw, native, result, mask)
    original_control_inner_product = (adjusted * (result - native)).sum((1, 2))
    assert torch.allclose(response['first_order_reward_change'], torch.tensor([.425], dtype=torch.float64))
    assert torch.allclose(original_control_inner_product, torch.tensor([.40625], dtype=torch.float64))
    assert not torch.equal(response['first_order_reward_change'], original_control_inner_product)
    assert torch.allclose(response['flowcompat_raw_gradient_rms_native'], raw.norm(dim=(1, 2)))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('spec', [{}, {'parallel_component_scale': .25}])
def test_wrapper_correction_preserves_parent_coordinate_output_exactly(tmp_path, monkeypatch, dtype, spec):
    raw = torch.tensor([[[1., 2., 0.], [.5, -1., 0.]]], dtype=dtype)
    flow = torch.tensor([[[1., 0., 0.], [1., 0., 0.]]], dtype=dtype)
    native = torch.full_like(raw, .31)
    mask = torch.ones(raw.shape[:2], dtype=torch.bool)
    expected_adjusted, _, _ = control_geometry(raw, flow, mask, .2, [0., .5], spec)
    expected = native + .1 * expected_adjusted
    extension = FlowCompatibilityV2Extension()
    extension.cached = (raw, torch.tensor([1.], dtype=dtype), {})
    extension.model = SimpleNamespace(_lineage=SimpleNamespace(path=tmp_path))
    extension.v2_controller_sha = 'unit-test-module-hash'
    seen = []

    def frozen_parent(self, curr):
        seen.append(1)
        g, _, _ = self.cached
        adjusted, _, _ = control_geometry(g, flow, mask, .2, [0., .5], spec)
        injection = .1 * adjusted
        result = dict(curr)
        result['coords'] = curr['coords'] + injection
        (tmp_path / 'guidance_trace.jsonl').write_text(json.dumps({'first_order_reward_change': (adjusted * injection).sum((1, 2)).tolist()}) + '\n')
        self.cached = None
        return result

    monkeypatch.setattr(FlowCompatibilityExtension, 'after_native', frozen_parent)
    result = extension.after_native({'coords': native, 'mask': mask})
    row = json.loads((tmp_path / 'guidance_trace.jsonl').read_text())
    assert seen == [1] and torch.equal(result['coords'], expected)
    assert torch.equal(native, torch.full_like(raw, .31))
    actual = expected - native
    assert row['first_order_reward_change'] == (raw * actual).sum((1, 2)).tolist()
    assert row['control_inner_product'] == (expected_adjusted * (.1 * expected_adjusted)).sum((1, 2)).tolist()
    assert row['v2_controller_sha256'] == extension.v2_controller_sha
    assert row['generation_interface'] == 'flowcompat_v2'


def test_inactive_wrapper_preserves_object_and_coordinates(tmp_path, monkeypatch):
    extension = FlowCompatibilityV2Extension()
    extension.cached = None
    extension.model = SimpleNamespace(_lineage=SimpleNamespace(path=tmp_path))
    extension.v2_controller_sha = 'inactive-unit-test-hash'
    current = {'coords': torch.tensor([[[1., 2., 3.]]]), 'mask': torch.tensor([[True]])}

    def frozen_parent(self, curr):
        (tmp_path / 'guidance_trace.jsonl').write_text(json.dumps({'active': False}) + '\n')
        return curr

    monkeypatch.setattr(FlowCompatibilityExtension, 'after_native', frozen_parent)
    result = extension.after_native(current)
    row = json.loads((tmp_path / 'guidance_trace.jsonl').read_text())
    assert result is current and result['coords'] is current['coords']
    assert row['active'] is False and 'first_order_reward_change' not in row
