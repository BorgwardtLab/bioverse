import awkward as ak
import numpy as np

from ..sampler import Sampler
from ..utilities import flatten


class MutationSampler(Sampler):

    """Sample indices at the mutation level for mutational scans."""

    def index(self, toc, mask):
        scenes = np.repeat(np.arange(len(toc["mutations"])), toc["mutations"])[mask]
        mutations = np.concatenate([np.arange(m) for m in toc["mutations"]])[mask]

        index = ak.Array(
            {
                "scene": scenes,
                "frame": ak.Array(np.full(len(scenes), 0)).unflatten(1, -1),
                "molecule": ak.Array(np.full(len(scenes), 0)).unflatten(1, -1),
                "mutation": ak.Array(mutations),
            }
        )
        return index
