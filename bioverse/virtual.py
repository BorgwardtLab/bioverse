from collections import OrderedDict
from typing import Tuple

import awkward as ak
import numpy as np

from .data import Batch
from .utilities import SHARD_SIZE, config, load


class LRUCache(OrderedDict):
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
        vbatch = self.superbatch[index]  # type: ignore
        vbatch = self.live_transforms.transform_batch(vbatch)
        return vbatch
