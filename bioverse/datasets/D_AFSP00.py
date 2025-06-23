from ..adapters import AlphaFoldAdapter
from ..dataset import Dataset
from ..transform import Compose
from ..transforms import FilterSequenceLength, SceneSplit


class D_AFSP00(Dataset):

    def release(self):
        batches, split, assets = AlphaFoldAdapter.download(name="swissprot_pdb")
        pipeline = Compose(
            FilterSequenceLength(1024),
            SceneSplit(test_size=1000, val_size=1000),
        )
        return pipeline(batches, split, assets)
