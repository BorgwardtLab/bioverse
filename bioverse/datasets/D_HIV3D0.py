from ..adapters import DtpAidsAdapter
from ..dataset import Dataset
from ..transform import Compose
from ..transforms import GraphFromSmiles


class D_HIV3D0(Dataset):

    def release(self):
        batches, split, assets = DtpAidsAdapter.download()
        pipeline = Compose(
            GraphFromSmiles(geometric=True, dim=3),
        )
        return pipeline(batches, split, assets)
