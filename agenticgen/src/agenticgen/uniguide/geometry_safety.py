"""Candidate failure isolation without changing healthy autograd graphs.

The accounting unit is an immutable ten-row RNG shard. Inspection is detached;
only an already failed shard receives inert placeholders. Healthy tensors must
be returned by identity, including when another shard has failed.
"""

from functools import singledispatch

import torch

from .native_tools import save_pt

SHARD_ROWS = 10
PRESERVED_NETWORK_FIELDS = frozenset({"affinity", "mask"})


@singledispatch
def map_network_tensors(value, transform, path="network"):
    """Visit sampling tensors while preserving structural masks/affinity heads."""
    return value


@map_network_tensors.register(torch.Tensor)
def _map_tensor(value, transform, path="network"):
    return transform(path, value)


@map_network_tensors.register(dict)
def _map_mapping(value, transform, path="network"):
    return {
        key: (
            item
            if key in PRESERVED_NETWORK_FIELDS
            else map_network_tensors(item, transform, f"{path}/{key}")
        )
        for key, item in value.items()
    }


@map_network_tensors.register(list)
@map_network_tensors.register(tuple)
def _map_sequence(value, transform, path="network"):
    return type(value)(
        map_network_tensors(item, transform, f"{path}/{index}")
        for index, item in enumerate(value)
    )


class CandidateSafety:
    """Serial sampler failure accounting for immutable ten-row RNG shards."""

    def _candidate_ids(self, value, ids):
        resolved = list(range(self.initial["population"])) if ids is None else ids
        if value.shape[0] != len(resolved):
            raise ValueError("Cannot assign numerical failure to candidate")
        return resolved

    def _inspect_rows(self, name, value, ids):
        bad_rows = ~torch.isfinite(value.detach()).reshape(len(ids), -1).all(1)
        for local_row in bad_rows.nonzero().flatten().tolist():
            row = ids[local_row]
            unit = row // SHARD_ROWS
            if unit in self.failed_units:
                continue
            error = dict(
                type="nonfinite_candidate",
                field=name,
                row=row,
                step=getattr(self, "active_step", len(self.rows)),
                no_scientific_retry=True,
            )
            save_pt(
                self.folder / f"nonfinite_unit_{unit:03d}.pt.gz",
                dict(error=error, raw=value.detach()[local_row]),
            )
            self.failed_units[unit] = error
        return value

    def _quarantine_rows(self, value, ids):
        failed = tuple(row // SHARD_ROWS in self.failed_units for row in ids)
        # An identity-valued torch.where is NOT an autograd identity. Its extra
        # node changed branch accumulation at the first guided step in the real
        # model. Never attach a guard node to an entirely healthy microbatch.
        if not any(failed):
            return value
        invalid = torch.tensor(failed, dtype=torch.bool, device=value.device)
        shape = (-1,) + (1,) * (value.ndim - 1)
        return torch.where(invalid.reshape(shape), torch.zeros_like(value), value)

    def check_rows(self, name, value, ids=None):
        if not torch.is_tensor(value) or not value.ndim:
            return value
        ids = self._candidate_ids(value, ids)
        self._inspect_rows(name, value, ids)
        return self._quarantine_rows(value, ids)

    def prepare_output(self, output, ids):
        def inspect(name, value):
            if value.ndim and value.shape[0] == len(ids):
                self._inspect_rows(name, value, ids)
            return value

        def quarantine(name, value):
            if value.ndim and value.shape[0] == len(ids):
                return self._quarantine_rows(value, ids)
            return value

        # Discover all failed shards before replacing any tensors. Inspection
        # does not build throwaway autograd guards or mutate the model output.
        map_network_tensors(output, inspect)
        return map_network_tensors(output, quarantine)
