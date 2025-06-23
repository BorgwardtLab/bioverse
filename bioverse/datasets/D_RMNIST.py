from ..adapters import RotMnistAdapter
from ..dataset import Dataset


class D_RMNIST(Dataset):

    def release(self):
        return RotMnistAdapter.download()
