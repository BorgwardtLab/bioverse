import os
from collections import defaultdict
import shutil

import pandas as pd

from ..logger import Logger
from ..utilities import save


class DiskLogger(Logger):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.trainer = trainer
        os.makedirs(trainer.root, exist_ok=True)

    def log_loss(self, data, mode='train'):
        data = {'Loss': data}
        data["Step"] = self.trainer.step
        data["Epoch"] = self.trainer.epoch
        data = {k: [v] for k, v in data.items()}
        if os.path.exists(self.trainer.root / f"{mode}_loss.csv"):
            df = pd.read_csv(self.trainer.root / f"{mode}_loss.csv")
        else:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        df.to_csv(self.trainer.root / f"{mode}_loss.csv", index=False)

    def log_dict(self, data, mode='train'):
        data["Step"] = self.trainer.step
        data["Epoch"] = self.trainer.epoch
        data = {k: [v] for k, v in data.items()}
        if os.path.exists(self.trainer.root / f"{mode}_results.csv"):
            df = pd.read_csv(self.trainer.root / f"{mode}_results.csv")
        else:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        df.to_csv(self.trainer.root / f"{mode}_results.csv", index=False)
