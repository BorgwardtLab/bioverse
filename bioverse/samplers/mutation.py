import awkward as ak
import numpy as np

from ..sampler import Sampler
from ..utilities import flatten


class MutationSampler(Sampler):

    def index(self, toc):
        num_mutations = toc["mutations"].ravel().num(axis=0)

        index = ak.Array(
            {
                "scene": np.full(num_mutations, 0),
                "frame": ak.Array(np.full(num_mutations, 0)).unflatten(1, -1),
                "molecule": ak.Array(np.full(num_mutations, 0)).unflatten(1, -1),
                "mutation": ak.Array(self.rng.permutation(np.arange(num_mutations))),
            }
        )
        return index
