import awkward as ak

from ..sampler import Sampler
from ..utilities import flatten


class FrameSampler(Sampler):

    def index(self, toc):
        molecules = toc["chain"]
        index = ak.zip([ak.local_index(molecules, i) for i in range(molecules.ndim)])
        index = flatten(index, exclude=4)
        scenes, frames, molecules = ak.unzip(index)
        scenes = toc["scene"][scenes]  # remap to toc index

        # reshape
        frames = frames.unflatten(ak.full_like(scenes, 1))  # scenes are always 1
        molecules = molecules.unflatten(toc["frame"])

        index = ak.Array(
            {
                "scene": scenes,
                "frame": frames,
                "molecule": molecules,
            }
        )
        return index
