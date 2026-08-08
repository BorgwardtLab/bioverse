import awkward as ak
import numpy as np

from ..sampler import Sampler


class InterfacePairSampler(Sampler):

    def sample(self, dataset, partition, split, **kwargs):
        self.assets = dataset.assets
        return super().sample(dataset, partition, split, **kwargs)

    def index(self, toc, mask):
        scene_indices = np.where(mask)[0]
        protein_ids = self.assets["protein_ids"]
        interfaces = self.assets["interfaces"]
        pairs = []
        for scene in scene_indices:
            protein_id = protein_ids[int(scene)]
            if "_" not in protein_id:
                continue
            pdbid, chain = protein_id.split("_", 1)
            partners = interfaces.get(pdbid, {}).get(chain, {})
            for partner_chain in partners:
                partner_id = f"{pdbid}_{partner_chain}"
                try:
                    partner_scene = protein_ids.index(partner_id)
                except ValueError:
                    continue
                if partner_scene not in scene_indices or partner_scene <= scene:
                    continue
                pairs.append((int(scene), 0, 0, int(partner_scene), 0, 0))
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
        pairs = np.array(pairs, dtype=int)
        return ak.Array(
            {
                "scene": pairs[:, 0],
                "frame": pairs[:, 1],
                "molecule": pairs[:, 2],
                "scene2": pairs[:, 3],
                "frame2": pairs[:, 4],
                "molecule2": pairs[:, 5],
            }
        )
