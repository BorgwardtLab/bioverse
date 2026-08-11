from typing import Union

import awkward as ak
import numpy as np

from ..sampler import Sampler
from ..utilities import flatten


class ResidueSampler(Sampler):

    """Sample one index per residue in the active split."""

    def __init__(self, num_residues: Union[int, float] = 1):
        self.num_residues = num_residues

    def index(self, toc, mask):
        # reformat the toc to long format
        residues = toc[mask]["residue"]
        index = ak.zip([ak.local_index(residues, i) for i in range(residues.ndim)])
        index = flatten(index, exclude=4)
        scenes, frames, molecules, chains = ak.unzip(index)
        scenes = toc[mask]["scene"][scenes]  # remap to toc index
        # compute number of samples
        residues = ak.sum(residues, axis=-1)  # sum over chains
        residues = ak.flatten(residues, axis=None)
        if isinstance(self.num_residues, int):
            num_residues = ak.full_like(residues, self.num_residues)
        if isinstance(self.num_residues, float):
            num_residues = ak.values_astype(residues * self.num_residues, int)
            num_residues = ak.where(
                num_residues < 1, 1, num_residues
            )  # have at least one residue
        # create random mask
        random_mask = np.concatenate(
            [np.ones(ak.sum(num_residues)), np.zeros(ak.sum(residues - num_residues))]
        )
        self.rng.shuffle(random_mask)
        random_mask = ak.unflatten(random_mask, residues)
        # mask local index
        samples = ak.local_index(random_mask)[ak.values_astype(random_mask, bool)]
        # reshape
        frames = frames.unflatten(ak.full_like(scenes, 1))  # scenes are always 1
        molecules = molecules.unflatten(toc[mask]["frame"])
        residues = residues.unflatten(toc[mask]["frame"])
        # return samples
        index = ak.Array(
            {
                "scene": scenes,
                "frame": frames,
                "molecule": molecules,
                "residue": samples,
            }
        )
        return index
