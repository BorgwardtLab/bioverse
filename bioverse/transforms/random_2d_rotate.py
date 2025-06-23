import awkward as ak
import numpy as np

from ..transform import Transform


class Random2DRotate(Transform):

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def transform_batch(self, batch):
        pos = batch.molecules.atom_pos
        angle = self.rng.rand(ak.num(pos, axis=0)) * 2 * np.pi
        angle = np.ones(ak.num(pos, axis=0)) * np.pi / 2
        cos, sin = np.cos(angle), np.sin(angle)
        rmat = np.stack([cos, -sin, sin, cos], axis=-1).reshape(-1, 2, 2)
        _, index = ak.broadcast_arrays(
            pos, np.arange(ak.num(pos, axis=0)), depth_limit=2
        )
        rmat = rmat[index.flatten()]
        batch.atom_pos = np.einsum("bx,bxy->by", batch.atom_pos.to_numpy(), rmat)  # type: ignore
        return batch
