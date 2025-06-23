import numpy as np

from ..transform import Transform
from ..utilities import PROTEIN_ALPHABET


class OneHotResidueFeatures(Transform):

    def __init__(self, alphabet=PROTEIN_ALPHABET):
        self.alphabet = alphabet

    def transform_assets(self, assets):
        if "features" not in assets:
            assets["residue_features"] = np.eye(len(self.alphabet)).tolist()
        else:
            assets["residue_features"] = np.concatenate(
                [
                    assets["residue_features"],
                    np.eye(len(self.alphabet)),
                ],
                axis=1,
            ).tolist()
        return assets
