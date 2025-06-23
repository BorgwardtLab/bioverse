import numpy as np

from ..transform import Transform
from ..utilities import BIOCHEMICAL_ATOM_ALPHABET


class OneHotAtomFeatures(Transform):

    def __init__(self, alphabet=BIOCHEMICAL_ATOM_ALPHABET):
        self.alphabet = alphabet

    def transform_assets(self, assets):
        if "features" not in assets:
            assets["atom_features"] = np.eye(len(self.alphabet)).tolist()
        else:
            assets["atom_features"] = np.concatenate(
                [
                    assets["atom_features"],
                    np.eye(len(self.alphabet)),
                ],
                axis=1,
            ).tolist()
        return assets
