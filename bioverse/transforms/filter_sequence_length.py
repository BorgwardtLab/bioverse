import awkward as ak

from bioverse.data import Batch

from ..transform import Transform


class FilterSequenceLength(Transform):
    filter = "scenes"

    def __init__(self, max_length: int):
        self.max_length = max_length

    def transform_batch(self, batch: Batch) -> Batch:
        batch.scene_filter = ak.all(
            ak.num(batch.scenes.chains.residue_label, axis=-1) < self.max_length,
            axis=-1,
        )
        return batch
