from ..adapters import AresAdapter
from ..dataset import Dataset


class D_ARES00(Dataset):

    def release(self):
        return AresAdapter.download()
