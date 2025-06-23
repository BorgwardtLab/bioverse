from ..adapters import ProteinInvBenchAdapter
from ..dataset import Dataset


class D_INVC42(Dataset):

    def release(self):
        return ProteinInvBenchAdapter.download()
