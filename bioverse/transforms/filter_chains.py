from __future__ import annotations

import awkward as ak

from ..data import Batch
from ..transform import Transform


def _as_labels(chains):
    if chains is None:
        return None
    if isinstance(chains, str):
        return (str(chains),)
    return tuple(str(chain) for chain in chains)


def _chain_ids(labels) -> list[str]:
    array = ak.Array(labels)
    while array.ndim > 1 and len(array) == 1:
        array = array[0]
    if array.ndim > 1:
        array = array[0]
        while array.ndim > 1 and len(array) == 1:
            array = array[0]
    return [str(item) for item in ak.to_list(array)]


class FilterChains(Transform):
    """Keep or drop polymer chains by ``chain_label``.

    Apply this *before* live graph transforms. Filtering after kNN leaves
    edges that point at removed nodes.

    Parameters
    ----------
    keep : str or sequence of str, optional
        Chain labels to retain. If omitted, every chain is eligible.
    drop : str or sequence of str, optional
        Chain labels to remove. Applied after ``keep``.
    """

    def __init__(self, keep=None, drop=None):
        self.keep = _as_labels(keep)
        self.drop = _as_labels(drop)
        if self.keep is None and self.drop is None:
            raise ValueError("FilterChains requires keep and/or drop.")

    def transform_batch(self, batch: Batch) -> Batch:
        labels = batch.data["chain_label"]
        present = _chain_ids(labels)
        selected = [
            i
            for i, chain in enumerate(present)
            if (self.keep is None or chain in self.keep)
            and (self.drop is None or chain not in self.drop)
        ]
        if not selected:
            raise ValueError(
                f"No chains in {present} match keep={self.keep} drop={self.drop}."
            )
        data = {key: value for key, value in batch.data.items()}
        if selected != list(range(len(present))):
            chain_axis = labels.ndim - 1
            slicer = (slice(None),) * chain_axis + (selected,)
            chain_level = batch.prefixes.index("chain")
            for col, value in list(data.items()):
                prefix = col.split("_")[0]
                if prefix in batch.prefixes and batch.prefixes.index(prefix) >= chain_level:
                    data[col] = value[slicer]
        return Batch(data, prefixes=list(batch.prefixes), resolution=batch.resolution)
