import awkward as ak
import numpy as np

from ..sampler import Sampler


class FrameSampler(Sampler):

    def index(self, toc, mask):
        num_frames = np.sum(mask)
        index = ak.Array(
            {
                "scene": ak.Array(np.full(num_frames, 0)),
                "frame": ak.unflatten(
                    np.arange(len(mask))[mask], np.ones(num_frames, dtype=int)
                ),
                "molecule": ak.unflatten(
                    np.full(num_frames, 0), np.ones(num_frames, dtype=int)
                ),
            }
        )
        return index
