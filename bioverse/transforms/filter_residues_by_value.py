import awkward as ak
import numpy as np

from ..transform import Transform


class FilterResiduesByValue(Transform):

    def __init__(
        self,
        attribute: str,
        value: any = None,
        below: float = None,
        above: float = None,
    ):
        self.attribute = attribute
        self.value = value
        self.below = below
        self.above = above
        assert (
            value is not None or below is not None or above is not None
        ), "Either value or threshold must be provided."

    def transform_batch(self, batch):
        if self.value is not None:
            mask = getattr(batch, self.attribute) == self.value
        elif self.below is not None:
            mask = getattr(batch, self.attribute) < self.below
        elif self.above is not None:
            mask = getattr(batch, self.attribute) > self.above
        batch.residues = batch.residues[mask]
        return batch
