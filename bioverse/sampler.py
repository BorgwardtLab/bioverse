from abc import ABC, abstractmethod
from typing import Tuple, cast

import awkward as ak
import numpy as np

from .dataset import Dataset
from .utilities import config


class Sampler(ABC):

    @abstractmethod
    def index(self, toc: ak.Array) -> ak.Array:
        raise NotImplementedError

    def sample(
        self,
        dataset: Dataset,
        split: str,
        batch_size: int = 1,
        batch_on: str = "scenes",
        shuffle: bool = False,
        drop_last: bool = False,
        random_seed: int = config.seed,
        world_size: int = 1,
        rank: int = 0,
    ) -> Tuple[ak.Array, ak.Array]:
        self.rng = np.random.default_rng(random_seed)
        toc, tos, split_names = (dataset.toc, dataset.tos, dataset.split.attrs["names"])
        split_numbers = np.argwhere(np.array(split_names) == split)[:, 0]
        split_mask = ak.any(
            [ak.any(dataset.split == n, axis=1) for n in split_numbers], axis=0
        )
        split_toc = toc[split_mask]
        split_toc = cast(ak.Array, split_toc)
        # shards = cast(ak.Array, split_toc["shard"])
        # shard_order = shards[np.sort(np.unique(shards, return_index=True)[1])]
        # shard_order = cast(ak.Array, shard_order)
        if shuffle:
            # shard_sizes = ak.run_lengths(shards)
            # scenes = ak.unflatten(np.arange(len(shards)), shard_sizes)
            num_scenes_per_shard = ak.unflatten(split_mask, tos).sum(axis=1)
            shuffle_index = ak.unflatten(
                np.arange(len(split_toc)), num_scenes_per_shard
            )
            shuffle_index = ak.Array([self.rng.permutation(s) for s in shuffle_index])
            shard_perm = self.rng.permutation(len(shuffle_index))
            shuffle_index = ak.flatten(shuffle_index[shard_perm])
            split_toc = split_toc[shuffle_index]
            split_toc = cast(ak.Array, split_toc)
        index = self.index(split_toc)
        if batch_on == "scenes":
            # tie loose ends to make equal-length batch lists in DDP
            end = (len(split_toc) - world_size + 1) // world_size * world_size
            return ak.unflatten(index[rank:end:world_size], batch_size)
        elif batch_on == "frames":
            sizes = toc["frame"][index["scene"]]
        elif batch_on == "molecules":
            sizes = toc["molecule"][index["scene"]][index["frame"]]
        elif batch_on == "residues":
            sizes = toc["residue"][index["scene"]][index["frame"]][index["molecule"]]
        elif batch_on == "edges":
            sizes = toc["graph"][index["scene"]][index["frame"]][index["molecule"]]
        # compute batch list sizes for each rank
        rank_sizes = []
        sizes = ak.to_numpy(ak.ravel(sizes))
        for world in range(world_size):
            rank_sizes.append(
                ak.run_lengths(np.cumsum(sizes[world::world_size]) // batch_size)
            )
        # make batches
        index = ak.unflatten(index[rank::world_size], rank_sizes[rank])
        # ensure equal-length batch lists in DDP
        rank_min_batch_num = min(len(rank_sizes[world]) for world in range(world_size))
        if drop_last:
            rank_min_batch_num = rank_min_batch_num - 1
        index = index[:rank_min_batch_num]
        return index
