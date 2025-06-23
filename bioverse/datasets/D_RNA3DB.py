from ..adapters import Rna3dbAdapter
from ..dataset import Dataset


class D_RNA3DB(Dataset):

    def release(self):
        return Rna3dbAdapter.download()
