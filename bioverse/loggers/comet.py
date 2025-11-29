import os
import shutil
from collections import defaultdict

import pandas as pd

from ..logger import Logger
from ..utilities import save


class CometLogger(Logger):

    def __init__(
        self, trainer, api_key=None, project_name="Bioverse", experiment_name=None
    ):
        super().__init__(trainer)
        import comet_ml

        self.experiment = comet_ml.start(
            api_key=api_key or os.environ.get("COMET_API_KEY"),
            project_name=project_name,
            experiment_config=comet_ml.ExperimentConfig(name=experiment_name),
        )
        self.trainer = trainer
        os.makedirs(trainer.root, exist_ok=True)

    def log_loss(self, data, mode="train"):
        self.experiment.log_metric(
            f"{mode}_loss", data, step=self.trainer.step, epoch=self.trainer.epoch
        )

    def log_dict(self, data, mode="train"):
        self.experiment.log_metrics(
            data, step=self.trainer.step, epoch=self.trainer.epoch
        )
