from ..adapters import ProteinGymAdapter
from ..dataset import Dataset


class D_PROGYM(Dataset):

    def release(self):
        return ProteinGymAdapter.download()
