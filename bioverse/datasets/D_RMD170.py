from ..adapters import RevisedMolecularDynamicsAdapter
from ..dataset import Dataset


class D_RMD170(Dataset):

    def release(self):
        batches, split, assets = RevisedMolecularDynamicsAdapter.download()
        print(next(batches))
        exit()
        # todo: split
        return batches, split, assets
