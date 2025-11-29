import re

import numpy as np

from ..transform import Transform
from ..utilities import BIOCHEMICAL_ATOM_ALPHABET


class OneHotAtomFeatures(Transform):

    def __init__(self, alphabet=BIOCHEMICAL_ATOM_ALPHABET):
        self.alphabet = alphabet
        if not type(alphabet) == list:
            alphabet = re.findall(r"[A-Z][a-z]*", alphabet)
        self.n = len(alphabet)

    def transform_assets(self, assets):
        if "features" not in assets:
            assets["atom_features"] = np.eye(self.n).tolist()
        else:
            assets["atom_features"] = np.concatenate(
                [
                    assets["atom_features"],
                    np.eye(self.n),
                ],
                axis=1,
            ).tolist()
        return assets
