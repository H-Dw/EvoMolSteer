"""Locate the five recording callbacks without changing FLOWR's integration loop."""
import ast


def instrument_source(source: str) -> str:
    """Require unique *contextual* sites; final corrector code is not a step.

    Both the integration loop and final prediction use ``torch.no_grad``;
    both the integration loop and corrector use ``cond_batch``. Matching the
    surrounding block prevents accidental insertion at the wrong occurrence.
    """
    substitutions = [
        (
            "    with torch.no_grad():\n\n        pocket_equis_target, pocket_invs_target",
            "    self._lineage.begin(prior, pocket_data_target, pocket_data_untarget)\n"
            "    with torch.no_grad():\n\n        pocket_equis_target, pocket_invs_target",
        ),
        (
            "        for i, step_size in enumerate(step_sizes):\n"
            "            cond = cond_batch if self.self_condition else None\n",
            "        for i, step_size in enumerate(step_sizes):\n"
            "            cond = cond_batch if self.self_condition else None\n"
            "            self._lineage.before(i, curr, cond, times, step_size, prior)\n",
        ),
        (
            "            # Integrate the ODE using Euler\n",
            "            self._lineage.score(predicted_target, predicted_untarget)\n"
            "            # Integrate the ODE using Euler\n",
        ),
        (
            "            # put into tuples\n",
            "            self._lineage.propose(curr)\n            # put into tuples\n",
        ),
        (
            "            # Inpainting for the ligand if required\n",
            "            self._lineage.end(curr, cond_batch, times)\n"
            "            # Inpainting for the ligand if required\n",
        ),
    ]
    changed = source
    for old, new in substitutions:
        if changed.count(old) != 1:
            raise RuntimeError("Unsupported FLOWR.ROOT selective loop; nonunique capture site: " + repr(old))
        changed = changed.replace(old, new, 1)
    ast.parse(changed)
    return changed
