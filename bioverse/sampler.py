from abc import ABC, abstractmethod
from typing import Tuple, cast

import awkward as ak
import numpy as np

from .dataset import Dataset
from .utilities import SHARD_SIZE, config


class Sampler(ABC):
    """Select which dataset elements form each training or evaluation batch.

    Samplers map the dataset table-of-contents (``toc``) and an active split
    partition to Awkward index arrays at the scene, frame, molecule, or residue
    level. :meth:`sample` groups those indices into batches according to
    ``batch_size``, ``batch_on``, and distributed-training settings.

    Subclasses implement :meth:`index`. Common strategies include sampling
    every molecule (:class:`~bioverse.samplers.molecule.MoleculeSampler`) or
    every frame (:class:`~bioverse.samplers.frame.FrameSampler`).

    Examples
    --------
    .. code-block:: python

       from bioverse.samplers import MoleculeSampler

       sampler = MoleculeSampler()
       batch_indices = sampler.sample(
           dataset, partition="train", split="default", batch_size=32
       )
    """

    @abstractmethod
    def index(self, toc: ak.Array, mask: ak.Array) -> ak.Array:
        """Return row indices for elements in the active split partition.

        Parameters
        ----------
        toc
            Table-of-contents array describing dataset size at each level.
        mask
            Boolean mask selecting scenes in the current partition.

        Returns
        -------
        ak.Array
            Structured index with fields such as ``scene``, ``frame``,
            ``molecule``.
        """
        raise NotImplementedError

    def sample(
        self,
        dataset: Dataset,
        partition: str,
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
        partition = dataset.split.default if partition is None else partition
        toc, tos, mask = dataset.toc, dataset.tos, dataset.split[split, partition]
        # todo: reduce toc with split, index, then remap
        index = self.index(toc, mask)
        if shuffle:
            order = np.argsort(index["scene"])
            index = ak.Array({k: index[k][order] for k in index.fields})
            num_scenes_per_shard = np.unique(
                index["scene"] // SHARD_SIZE, return_counts=True
            )[1]
            shuffle_index = ak.unflatten(
                np.arange(len(index["scene"])), num_scenes_per_shard
            )
            shuffle_index = ak.Array([self.rng.permutation(s) for s in shuffle_index])
            shard_perm = self.rng.permutation(len(shuffle_index))
            shuffle_index = ak.flatten(shuffle_index[shard_perm])
            index = ak.Array({k: index[k][shuffle_index] for k in index.fields})
        if batch_on == "scenes" or batch_on == "mutations":
            # tie loose ends to make equal-length batch lists in DDP
            # end = (len(split_toc) - world_size + 1) // world_size * world_size # put this back if there are bugs
            end = (len(index["scene"]) - world_size + 1) // world_size * world_size
            return ak.unflatten(index[rank:end:world_size], min(batch_size, end - rank))
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
            rank_min_batch_num -= 1
        index = index[:rank_min_batch_num]
        return index
