import awkward as ak

from ..sampler import Sampler


class MoleculeAndMutationSampler(Sampler):

    def index(self, toc):
        for item in toc:
            scene = item["scene"]
            for frame in range(item["frames"]):
                for molecule in range(item["molecules"][frame]):
                    mutations = item["mutations"][frame][molecule]
                    mutation = self.rng.integers(mutations)
                    yield {
                        "scene": scene,
                        "frame": frame,
                        "molecule": molecule,
                        "mutation": mutation,
                    }

    def add_to_toc(self, shard):
        return {"mutations": ak.num(shard["mutations"], axis=3)}
