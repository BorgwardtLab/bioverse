from ..dataset import ComposedDataset, Dataset
from .D_AFSP00 import D_AFSP00
from .D_PROGYM import D_PROGYM


class D_AFPGYM(Dataset):

    def release(self):
        composed = ComposedDataset(D_AFSP00(), D_PROGYM())
        batches, split, assets = composed.shards, composed.split, composed.assets
        rename = {
            "train_D_AFEC00": "pretrain",
            "test_D_AFEC00": "pretest",
            "val_D_AFEC00": "preval",
            "test_D_PROGYM": "test",
        }
        split.attrs["names"] = [rename[name] for name in split.attrs["names"]]
        return batches, split, assets
