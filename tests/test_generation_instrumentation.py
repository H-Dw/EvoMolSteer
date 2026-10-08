import ast
from pathlib import Path
import pytest
from evomolsteer.generation.instrumentation import instrument_source


SOURCE = '''def _generate_selective(self):
    with torch.no_grad():

        pocket_equis_target, pocket_invs_target = encode()
        for i, step_size in enumerate(step_sizes):
            cond = cond_batch if self.self_condition else None
            # Integrate the ODE using Euler
            curr = integrate()
            # put into tuples
            predicted = prepare()
            curr = resample()
            times = update_times()
            # Inpainting for the ligand if required
            pass
        for j in range(corr_iters):
            cond = cond_batch if self.self_condition else None
            curr = correct()
    with torch.no_grad():
        cond = cond_batch if self.self_condition else None
        return predict()
'''


def callback_names(tree):
    return [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == '_lineage']


def test_callbacks_apply_only_to_integration_not_corrector():
    changed = instrument_source(SOURCE)
    tree = ast.parse(changed)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert callback_names(loops[0]) == ['before', 'score', 'propose', 'end']
    assert callback_names(loops[1]) == []
    assert sorted(callback_names(tree)) == ['before', 'begin', 'end', 'propose', 'score']
    # Removing only instrumentation restores every line of the actual algorithm.
    assert ''.join(line for line in changed.splitlines(True) if 'self._lineage.' not in line) == SOURCE


def test_changed_or_duplicated_capture_site_is_rejected():
    with pytest.raises(RuntimeError):
        instrument_source(SOURCE.replace('# Integrate the ODE using Euler', '# changed upstream loop'))
    with pytest.raises(RuntimeError):
        instrument_source(SOURCE.replace('            # put into tuples\n', '            # put into tuples\n' * 2))


def test_full_upstream_fixture_restores_native_algorithm_line_for_line():
    source=(Path(__file__).parent/'fixtures/upstream_generate_selective.py.txt').read_text()
    changed=instrument_source(source)
    assert ''.join(line for line in changed.splitlines(True) if 'self._lineage.' not in line)==source
    assert sorted(callback_names(ast.parse(changed)))==['before','begin','end','propose','score']
