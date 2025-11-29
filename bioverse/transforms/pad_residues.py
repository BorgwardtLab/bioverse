import awkward as ak

from ..transform import Transform


class PadResidues(Transform):

    def __init__(self, max_length, token=-1, clip=True):
        super().__init__()
        self.max_length = max_length
        self.token = token
        self.clip = clip

    def transform_assets(self, assets):
        assets["_pad_max_length"] = self.max_length
        assets["_pad_clip"] = self.clip
        assets["_pad_token"] = self.token
        return assets
