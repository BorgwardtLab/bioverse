from ..adapters import AlphaFoldAdapter
from ..dataset import Dataset
from ..transform import Compose
from ..transforms import FilterSequenceLength, SceneSplit


class D_AFEC00(Dataset):

    def release(self):
        batches, split, assets = AlphaFoldAdapter.download(
            name="UP000000625_83333_ECOLI"
        )
        pipeline = Compose(
            FilterSequenceLength(1024),
            SceneSplit(test_size=100, val_size=100),
        )
        return pipeline(batches, split, assets)
