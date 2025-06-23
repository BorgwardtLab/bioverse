from ..adapters import QuantumMachinesAdapter
from ..dataset import Dataset
from ..transform import Compose
from ..transforms import SceneSplit


class D_QNTMA9(Dataset):

    def release(self):
        batches, split, assets = QuantumMachinesAdapter.download()
        pipeline = Compose(
            SceneSplit(train_size=0.8, test_size=0.1, val_size=0.1),
        )
        return pipeline(batches, split, assets)
