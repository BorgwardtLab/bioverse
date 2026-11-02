from collections import OrderedDict
from typing import Tuple

import awkward as ak
import numpy as np

from .data import Batch, _as_row_choice
from .utilities import SHARD_SIZE, config, load


def first_unique_rows(keys):
    """First-occurrence indices and an inverse map for row-wise keys.

    ``keys`` is ``(N,)`` or ``(N, K)``. The inverse maps every original row
    onto the unique-row index so live transforms can run once per structure.
    """
    keys = np.asarray(keys)
    if keys.ndim == 1:
        keys = keys.reshape(-1, 1)
    seen = {}
    uniq = []
    inverse = np.empty(len(keys), dtype=np.int64)
    for i, row in enumerate(map(tuple, keys.tolist())):
        found = seen.get(row)
        if found is None:
            found = len(uniq)
            seen[row] = found
            uniq.append(i)
        inverse[i] = found
    return np.asarray(uniq, dtype=np.int64), inverse


def _int_rows(value, n):
    if isinstance(value, (int, np.integer)):
        return np.full(n, int(value), dtype=np.int64)
    choice = np.asarray(_as_row_choice(value), dtype=np.int64).reshape(-1)
    if len(choice) == 1 and n != 1:
        return np.full(n, int(choice[0]), dtype=np.int64)
    return choice


def _take_rows(component, rows):
    if isinstance(component, (int, np.integer, slice)):
        return component
    if isinstance(component, ak.Array):
        return component[rows]
    return np.asarray(component)[rows]


class LRUCache(OrderedDict):
    """Least-recently-used cache for on-disk batch shards.

    Used by :class:`VirtualBatch` to limit memory when loading shards during
    task extraction.
    """

    def __init__(self, size: int = config.cache_size):
        super().__init__()
        self.size = size
        self._check_size()

    def __setitem__(self, shard: int, value: Batch):
        super().__setitem__(shard, value)
        self._check_size()

    def _check_size(self):
        while len(self) > self.size:
            self.popitem(last=False)  # Remove oldest item


class VirtualBatch:
    """Lazy, cache-backed view over sharded on-disk dataset data.

    ``VirtualBatch`` loads only the shards needed for a given index tuple,
    featurizes tokens using :class:`~bioverse.data.Assets`, and applies live
    transforms before returning a :class:`~bioverse.data.Batch` to the task.

    Examples
    --------
    .. code-block:: python

       vbatch = VirtualBatch(dataset.path, dataset.assets, dataset.live_transform)
       batch = vbatch[index["scene"], index["frame"], index["molecule"]]
    """

    def __init__(self, path, assets, live_transforms) -> None:
        self.path = path
        self.assets = assets
        self.cache = LRUCache(config.cache_size)
        self.superbatch = Batch({})
        self.live_transforms = live_transforms

    def featurize(self, batch: Batch) -> Batch:
        """
        Ad hoc featurization. This converts tokens to features only while loading the data, to reduce disk space usage.
        """
        for level in ["residue", "atom"]:
            if f"{level}_token" in batch and f"{level}_features" in self.assets:
                features = np.array(self.assets[f"{level}_features"])[
                    getattr(batch, f"{level}_token")
                ]
                # if there are existing features, conatenate
                if f"{level}_features" in batch:
                    features = ak.concatenate(
                        [getattr(batch, f"{level}_features"), features], axis=1
                    )
                setattr(batch, f"{level}_features", features)
        return batch

    def __getitem__(self, index: Tuple[ak.Array, ...]) -> Batch:
        # get shard index and scene index relative to shard
        shard_index = index[0] // SHARD_SIZE + 1
        scene_index = index[0] % SHARD_SIZE
        # get set of shards to load (must be sorted for indexing at the end)
        shard_set = np.sort(list(dict.fromkeys(shard_index).keys()))
        if not all(shard in self.cache for shard in shard_set):
            assert (
                len(shard_set) <= self.cache.size
            ), f"Cache size {self.cache.size} is too small. This can happen when the dataset has not been optimized for data loading. Try using the `OptimizeDataLoading` transform in either the dataset constructor or the `apply` method."
            # load new shards into cache
            for shard in shard_set:
                if not shard in self.cache:
                    self.cache[shard] = Batch(load(f"{self.path}/{shard}.ak"))
            # create scene index map
            batches = [self.cache[shard] for shard in shard_set]
            self.offsets = np.concatenate(
                [
                    [0],
                    np.cumsum([len(batch) for batch in batches])[:-1],
                ]
            )
            self.shards = shard_set
            # create new superbatch
            self.superbatch = sum(batches)
            # featurize superbatch
            self.superbatch = self.featurize(self.superbatch)  # type: ignore
        # map global scene index to superbatch index
        scene_index = (
            scene_index + self.offsets[np.searchsorted(self.shards, shard_index)]
        )
        index = (scene_index, *index[1:])
        scenes = _int_rows(index[0], len(np.asarray(ak.to_numpy(ak.ravel(index[0])))))
        n = len(scenes)
        keys = [scenes]
        if len(index) > 1:
            keys.append(_int_rows(index[1], n))
        if len(index) > 2:
            keys.append(_int_rows(index[2], n))
        uniq, inverse = first_unique_rows(np.stack(keys, axis=1))
        if len(uniq) != n:
            index = tuple(_take_rows(part, uniq) for part in index)
        vbatch = self.superbatch[index]  # type: ignore
        vbatch = self.live_transforms.transform_batch(vbatch)
        if len(uniq) != n:
            vbatch = vbatch[inverse]
        return vbatch
