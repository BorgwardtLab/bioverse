import awkward as ak
import numpy as np

from ..sampler import Sampler
from ..utilities import config


class MutationSampler(Sampler):
    """Sample indices at the mutation level for mutational scans.

    Each yielded index points at the same scene/frame/molecule and a different
    ``mutation`` entry, so loaders reuse one structure while varying mutation
    annotations.

    ``group_scenes=True`` keeps mutations of one protein contiguous after
    shuffle so live graphs and the encoder run once per structure in a batch.
    Leave it false for MEROPS contrastive batches that need mixed enzymes.
    """

    def __init__(self, group_scenes=False):
        self.group_scenes = bool(group_scenes)

    def _shuffle_grouped(self, index):
        scenes = np.asarray(ak.to_numpy(ak.ravel(index["scene"])), dtype=np.int64)
        order = np.argsort(scenes, kind="stable")
        index = ak.Array({k: index[k][order] for k in index.fields})
        scenes = np.asarray(ak.to_numpy(ak.ravel(index["scene"])), dtype=np.int64)
        _, counts = np.unique(scenes, return_counts=True)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        within = np.arange(len(scenes))
        for start, end in zip(offsets[:-1], offsets[1:]):
            within[start:end] = self.rng.permutation(within[start:end])
        blocks = [
            within[offsets[i] : offsets[i + 1]]
            for i in self.rng.permutation(len(counts))
        ]
        perm = np.concatenate(blocks) if blocks else within
        return ak.Array({k: index[k][perm] for k in index.fields})

    def sample(
        self,
        dataset,
        partition,
        split,
        batch_size=1,
        batch_on="scenes",
        shuffle=False,
        drop_last=False,
        random_seed=None,
        world_size=1,
        rank=0,
    ):
        if not self.group_scenes:
            return super().sample(
                dataset,
                partition,
                split,
                batch_size=batch_size,
                batch_on=batch_on,
                shuffle=shuffle,
                drop_last=drop_last,
                random_seed=random_seed,
                world_size=world_size,
                rank=rank,
            )
        if random_seed is None:
            random_seed = config.seed
        self.rng = np.random.default_rng(random_seed)
        partition = dataset.split.default if partition is None else partition
        toc, mask = dataset.toc, dataset.split[split, partition]
        index = self.index(toc, mask)
        if shuffle:
            index = self._shuffle_grouped(index)
        if batch_on not in ("scenes", "mutations"):
            return super().sample(
                dataset,
                partition,
                split,
                batch_size=batch_size,
                batch_on=batch_on,
                shuffle=False,
                drop_last=drop_last,
                random_seed=random_seed,
                world_size=world_size,
                rank=rank,
            )
        end = (len(index["scene"]) - world_size + 1) // world_size * world_size
        return ak.unflatten(index[rank:end:world_size], min(batch_size, end - rank))

    def index(self, toc, mask):
        n_mut = np.asarray(ak.to_numpy(ak.ravel(toc["mutations"])), dtype=int)
        scene_ids = np.asarray(ak.to_numpy(ak.ravel(toc["scene"])), dtype=int)
        scenes = np.repeat(scene_ids, n_mut)
        mutations = np.concatenate(
            [np.arange(int(count), dtype=int) for count in n_mut]
        )
        mask = np.asarray(mask, dtype=bool)
        if len(mask) == len(n_mut):
            mask = np.repeat(mask, n_mut)
        elif len(mask) != len(mutations):
            raise ValueError(
                "Mutation split mask length must match the number of scenes "
                f"({len(n_mut)}) or mutations ({len(mutations)}), got {len(mask)}."
            )
        scenes = scenes[mask]
        mutations = mutations[mask]
        n = len(scenes)
        index = ak.Array(
            {
                "scene": np.asarray(scenes, dtype=int),
                "frame": ak.Array([[0] for _ in range(n)]),
                "molecule": ak.Array([[0] for _ in range(n)]),
                "mutation": np.asarray(mutations, dtype=int),
            }
        )
        return index
