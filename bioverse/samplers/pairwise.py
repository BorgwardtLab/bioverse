import itertools

import awkward as ak
import numpy as np

from ..sampler import Sampler
from ..utilities import flatten


class PairwiseSampler(Sampler):

    """Sample ordered pairs of molecules for pairwise tasks."""

    def index(self, toc, mask):
        molecules = toc[mask]["chain"]
        index = ak.zip([ak.local_index(molecules, i) for i in range(molecules.ndim)])
        index = flatten(index, exclude=4)
        scenes, frames, molecules = ak.unzip(index)
        scenes = toc[mask]["scene"][scenes]
        frames = frames.unflatten(ak.full_like(scenes, 1))
        molecules = molecules.unflatten(toc[mask]["frame"])

        local = np.arange(len(scenes))
        pairs = np.array(list(itertools.combinations(local, 2)), dtype=int)
        if len(pairs) == 0:
            return ak.Array(
                {
                    "scene": ak.Array([], dtype=int),
                    "frame": ak.Array([], dtype=int),
                    "molecule": ak.Array([], dtype=int),
                    "scene2": ak.Array([], dtype=int),
                    "frame2": ak.Array([], dtype=int),
                    "molecule2": ak.Array([], dtype=int),
                }
            )
        return ak.Array(
            {
                "scene": scenes[pairs[:, 0]],
                "frame": frames[pairs[:, 0]],
                "molecule": molecules[pairs[:, 0]],
                "scene2": scenes[pairs[:, 1]],
                "frame2": frames[pairs[:, 1]],
                "molecule2": molecules[pairs[:, 1]],
            }
        )
